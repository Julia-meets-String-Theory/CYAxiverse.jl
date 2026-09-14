"""Primary implementation of the CYAX-0168 scale generator 2.2.

This module implements the frozen, scientifically inert synthetic fixture
generator.  It has no backend imports, no wall-clock reads, and no runtime
random-number generator.  The only selection primitive is the closed SHA-256
PRF specified by the CYAX-0168 specification.

The public ``generate_snapshot`` function returns a small mapping-like
``Snapshot`` object.  Its records are ordinary dictionaries so that callers
can serialize them without depending on this module's classes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from functools import lru_cache
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
import unicodedata


GENERATOR_VERSION = "cyax-0168-scale-2.2"
NAMESPACE = "cyax-0168-synthetic-v2.2"
OBSERVED_AT = "2000-01-01T00:00:00.000000Z"
FIXED_TIME = OBSERVED_AT
BASE_VALID_FROM = "2000-01-02T00:00:00.000000Z"
AUTHORITY_RULE = "synthetic_fixture_v2.2"
SCHEMA_VERSION = "cyax-snapshot-v1"
AUTHORITY_RULE_VERSION = "cyax-authority-v1"
EVALUATOR_RULE_VERSION = "cyax-evaluator-v1"


class GenerationError(ValueError):
    """Raised when a frozen generator precondition or invariant fails."""


GenerationFailure = GenerationError


class _Profile(dict[str, float | int]):
    """Frozen profile row with both mapping and attribute access."""

    def __getattr__(self, key: str) -> float | int:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


PROFILES: dict[str, _Profile] = {
    "P-low": _Profile({
        "dependency_fanout": 2,
        "path_depth": 4,
        "supersession_depth": 2,
        "dispute_rate": 0.005,
        "cross_link_rate": 0.01,
        "cycle_rate": 0.0,
        "isolated_rate": 0.20,
        "parallel_evidence_rate": 0.01,
    }),
    "P-medium": _Profile({
        "dependency_fanout": 8,
        "path_depth": 8,
        "supersession_depth": 8,
        "dispute_rate": 0.020,
        "cross_link_rate": 0.05,
        "cycle_rate": 0.005,
        "isolated_rate": 0.05,
        "parallel_evidence_rate": 0.03,
    }),
    "P-high": _Profile({
        "dependency_fanout": 32,
        "path_depth": 16,
        "supersession_depth": 32,
        "dispute_rate": 0.100,
        "cross_link_rate": 0.20,
        "cycle_rate": 0.020,
        "isolated_rate": 0.01,
        "parallel_evidence_rate": 0.10,
    }),
}

# The profile values are frozen.  ``entity_count`` may be supplied explicitly
# for tiny conformance fixtures; named tiers retain the approved scale matrix.
TIERS: dict[str, dict[str, int]] = {
    "T0": {"entity_count": 1_000, "assertion_count": 5_000},
    "T1": {"entity_count": 10_000, "assertion_count": 50_000},
    "T2": {"entity_count": 100_000, "assertion_count": 500_000},
    "T3": {"entity_count": 200_000, "assertion_count": 1_000_000},
    "T4": {"entity_count": 1_000_000, "assertion_count": 5_000_000},
}

PRF_PURPOSES = {
    "dependency_target",
    "cross_component_dependency_target",
    "fill_concerns_pair",
}

ASSERTION_PREIMAGE_KEYS = (
    "subject_id", "predicate", "object_id", "literal_ref", "source_revision_id",
    "source_locator", "source_event_at", "asserted_at", "valid_from", "valid_to",
    "validity_basis", "authority_class", "authority_derivation_rule_id", "origin",
    "curation_state", "review_state", "epistemic_state", "dispute_state",
)

_ROLE_TYPES = (
    "WorkItem",
    "Decision",
    "Requirement",
    "Requirement",
    "Implementation",
    "Implementation",
    "Verification",
    "Claim",
    "Artifact",
    "Source",
)
_ROLE_LABELS = {
    0: "work_item",
    1: "decision",
    2: "requirement_a",
    3: "requirement_b",
    4: "implementation_a",
    5: "implementation_b",
    6: "verification",
    7: "claim",
    8: "artifact",
    9: "source",
}

_SIGNATURES: dict[str, tuple[set[str], set[str], bool]] = {
    "governs": ({"Decision", "Specification", "WorkItem"}, {"WorkItem", "Specification", "Requirement"}, False),
    "requires": ({"WorkItem", "Decision", "Specification", "Requirement"}, {"Requirement", "Implementation", "Verification"}, False),
    "implements": ({"Implementation", "Artifact"}, {"Requirement", "Specification"}, False),
    "verifies": ({"Verification", "Artifact"}, {"Claim", "Requirement", "Implementation"}, False),
    "supports": ({"Claim", "Verification", "Artifact", "Source"}, {"Claim", "Decision", "Requirement"}, False),
    "contradicts": ({"Claim", "Verification", "Artifact", "Source"}, {"Claim", "Decision", "Requirement"}, False),
    "supersedes": ({"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact", "Source"}, {"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact", "Source"}, False),
    "depends_on": ({"WorkItem", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}, {"WorkItem", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}, False),
    "derived_from": ({"Decision", "Requirement", "Implementation", "Verification", "Claim", "Artifact", "Source"}, {"Decision", "Requirement", "Implementation", "Verification", "Claim", "Artifact", "Source"}, False),
    "concerns": ({"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}, {"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}, False),
    "documented_in": ({"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}, {"Source"}, False),
}


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


@lru_cache(maxsize=8192)
def _frame_string(value: str) -> bytes:
    raw = _nfc(value).encode("utf-8")
    return b"S" + len(raw).to_bytes(8, "big") + raw


@lru_cache(maxsize=8192)
def _frame_integer(value: int) -> bytes:
    if value < 0:
        raise GenerationError("canonical generator integers must be nonnegative")
    raw = str(value).encode("ascii")
    return b"I" + len(raw).to_bytes(8, "big") + raw


def canonical_frame(value: Any) -> bytes:
    """Return the specification's domain-separated canonical frame."""

    if value is None:
        return b"N" + (0).to_bytes(8, "big")
    if isinstance(value, bool):
        return b"B" + (1).to_bytes(8, "big") + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return _frame_integer(value)
    if isinstance(value, bytes):
        return b"X" + len(value).to_bytes(8, "big") + value
    if isinstance(value, str):
        return _frame_string(value)
    if isinstance(value, (list, tuple)):
        return b"A" + len(value).to_bytes(8, "big") + b"".join(canonical_frame(v) for v in value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical object keys must be strings")
            nfc_key = _nfc(key)
            if nfc_key in normalized:
                raise GenerationError(f"canonical object has colliding NFC key: {nfc_key!r}")
            normalized[nfc_key] = item
        keys = sorted((key.encode("utf-8"), key) for key in normalized)
        out = [b"O", len(keys).to_bytes(8, "big")]
        for _, key in keys:
            out.extend((canonical_frame(key), canonical_frame(normalized[key])))
        return b"".join(out)
    raise TypeError(f"unsupported canonical-frame value: {type(value).__name__}")


frame = canonical_frame


def _hash_id(prefix: str, preimage: Any) -> str:
    return f"{prefix}{hashlib.sha256(canonical_frame(preimage)).hexdigest()}"


def _json_bytes(record: Mapping[str, Any]) -> bytes:
    # Input records are already restricted to canonical JSON scalar types.
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


canonical_json_bytes = _json_bytes


class _Record(dict):
    """Dict record with attribute access for the focused conformance API."""

    def __getattr__(self, key: str) -> Any:
        if key == "claim_key" and key not in self:
            return None
        # The source-bundle contract has richer optional provenance columns
        # than the compact v1 snapshot record.  Keep their synthetic null/
        # anchor values available to focused conformance callers without
        # adding unknown fields to the canonical snapshot record.
        if key == "actor_login" and "actor_login" not in self:
            return None
        if key == "author_association" and "author_association" not in self:
            return None
        if key == "event_state" and "event_state" not in self:
            return None
        if key == "role_evidence" and "role_evidence" not in self:
            return None
        if key == "metadata" and "metadata" not in self:
            return None
        if key == "anchors" and "anchors" not in self:
            return ["json-object"] if "/assertion/" in self.get("canonical_locator", "") else []
        if key == "assertion_ordinal" and hasattr(self, "_assertion_ordinal"):
            return self._assertion_ordinal
        if key == "raw_bytes" and hasattr(self, "_raw_bytes"):
            return self._raw_bytes
        if key == "locator" and "canonical_locator" in self:
            return self["canonical_locator"]
        if key == "byte_count" and "object_byte_count" in self:
            return self["object_byte_count"]
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def preimage(self) -> dict[str, Any]:
        return {key: self.get(key) for key in ASSERTION_PREIMAGE_KEYS}


def _primary(value: str) -> bytes:
    return value.encode("utf-8")


def _id_for_entity(entity_type: str, source_identity: Sequence[Any]) -> str:
    return _hash_id("cyax-entity-sha256:", ["cyax-entity-v1", NAMESPACE, entity_type, list(source_identity)])


def entity_id(namespace: str, entity_type: str, canonical_source_identity: Sequence[Any]) -> str:
    """Public stable entity ID helper used by conformance harnesses."""
    return _hash_id("cyax-entity-sha256:", ["cyax-entity-v1", namespace, entity_type, list(canonical_source_identity)])


def _id_for_literal(literal_type: str, value: str) -> str:
    return _hash_id("cyax-literal-sha256:", ["cyax-literal-v1", literal_type, value])


def literal_id(literal_type: str, value: str) -> str:
    return _id_for_literal(literal_type, value)


def _id_for_revision(source_kind: str, locator: str, digest: str, source_event_at: str | None) -> str:
    return _hash_id("cyax-source-revision-sha256:", ["cyax-source-revision-v1", source_kind, locator, digest, source_event_at])


def source_revision_id(source_kind: str, locator: str, digest: str, source_event_at: str | None) -> str:
    return _id_for_revision(source_kind, locator, digest, source_event_at)


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _id_for_assertion(record: Mapping[str, Any]) -> str:
    fields = {k: record[k] for k in ASSERTION_PREIMAGE_KEYS}
    return _hash_id("cyax-assertion-sha256:", ["cyax-assertion-v2", fields])


def assertion_id(record: Mapping[str, Any]) -> str:
    expected = set(ASSERTION_PREIMAGE_KEYS)
    present = set(record)
    extra_allowed = {"assertion_id", "assertion_ordinal"}
    if present - expected - extra_allowed or expected - present:
        missing = sorted(expected - present)
        extra = sorted(present - expected - extra_allowed)
        raise GenerationError(f"assertion preimage fields must be exact; missing={missing}, extra={extra}")
    return _id_for_assertion(record)


def _digest_frame(value: Any) -> str:
    return hashlib.sha256(canonical_frame(value)).hexdigest()


def _block_identity(tier: str, profile_id: str, seed: int, block: int, role: int) -> list[Any]:
    return [tier, profile_id, seed, block, role]


def _prf_digest(seed: int, profile_id: str, purpose: str, ordinal: int, counter: int) -> bytes:
    if purpose not in PRF_PURPOSES:
        raise GenerationError(f"unregistered PRF purpose: {purpose!r}")
    if min(seed, ordinal, counter) < 0 or counter > (2**64 - 1):
        raise GenerationError("PRF integer outside the frozen nonnegative u64 domain")
    return hashlib.sha256(canonical_frame(["cyax-gen-2.2", seed, profile_id, purpose, ordinal, counter])).digest()


def prf_digest(seed: int, profile_id: str, purpose: str, ordinal: int, counter: int = 0) -> bytes:
    return _prf_digest(seed, profile_id, purpose, ordinal, counter)


def prf_choice(candidates: Sequence[Any], *, seed: int, profile_id: str, purpose: str, ordinal: int, reject: Any = None, presorted: bool = False) -> tuple[Any, int]:
    """Apply the exact rejection-sampling rule and return ``(choice,counter)``.

    ``reject`` is a callback accepting a candidate, or ``None``.  The helper
    is public so conformance tests can exercise big-endian modulo/retry cases
    without constructing a complete snapshot.
    """

    if purpose not in PRF_PURPOSES:
        raise GenerationError(f"unregistered PRF purpose: {purpose!r}")
    def candidate_key(value: Any) -> bytes:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return canonical_frame([value[0], value[1]])
        return _primary(str(value))

    ordered = list(candidates) if presorted else sorted(candidates, key=candidate_key)
    if not ordered:
        raise GenerationError("empty candidate vector")
    counter = 0
    while True:
        if len(ordered) == 1:
            candidate = ordered[0]
        else:
            digest = prf_digest(seed, profile_id, purpose, ordinal, counter)
            x = int.from_bytes(digest, "big")
            limit = (1 << 256) - ((1 << 256) % len(ordered))
            if x >= limit:
                if counter == 2**64 - 1:
                    raise GenerationError("PRF counter overflow")
                counter += 1
                continue
            candidate = ordered[x % len(ordered)]
        if reject is not None and reject(candidate):
            if len(ordered) == 1:
                raise GenerationError("sole candidate rejected by the exhaustive purpose rule")
            if counter == 2**64 - 1:
                raise GenerationError("PRF counter overflow")
            counter += 1
            continue
        return candidate, counter


def prf_select(seed: int, profile_id: str, purpose: str, ordinal: int, candidates: Sequence[Any], *, sort_key: Any = None, rejects: Any = None) -> Any:
    """Compatibility façade returning only the selected candidate."""
    ordered = sorted(candidates, key=sort_key) if sort_key is not None else candidates
    selected, _ = prf_choice(ordered, seed=seed, profile_id=profile_id, purpose=purpose, ordinal=ordinal, reject=rejects, presorted=sort_key is not None)
    return selected


@dataclass
class Snapshot:
    """Complete canonical generated snapshot and construction trace."""

    tier: str
    profile_id: str
    seed: int
    entities: list[dict[str, Any]]
    literals: list[dict[str, Any]]
    source_revisions: list[dict[str, Any]]
    assertions: list[dict[str, Any]]
    source_objects: dict[str, bytes]
    manifest: dict[str, Any]
    trace: dict[str, Any]
    all_assertions: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.entities = [_Record(x) if not isinstance(x, _Record) else x for x in self.entities]
        self.literals = [_Record(x) if not isinstance(x, _Record) else x for x in self.literals]
        self.source_revisions = [_Record(x) if not isinstance(x, _Record) else x for x in self.source_revisions]
        self.assertions = [_Record(x) if not isinstance(x, _Record) else x for x in self.assertions]
        for record in self.source_revisions:
            record._raw_bytes = self.source_objects.get(record["object_sha256"], b"")
        for record in self.assertions:
            if "assertion_ordinal" in record:
                record._assertion_ordinal = record.pop("assertion_ordinal")
        if self.all_assertions is not None:
            self.all_assertions = [_Record(x) if not isinstance(x, _Record) else x for x in self.all_assertions]
            for record in self.all_assertions:
                if "assertion_ordinal" in record:
                    record._assertion_ordinal = record.pop("assertion_ordinal")

    @property
    def block_count(self) -> int:
        return len(self.entities) // 10

    @property
    def snapshot_id(self) -> str:
        return self.manifest["snapshot_id"]

    @property
    def logical_snapshot_checksum(self) -> str:
        return self.manifest["logical_snapshot_checksum"]

    @property
    def cycle_trace(self) -> list[dict[str, Any]]:
        phase = next((p for p in self.trace.get("phases", []) if p.get("phase") == 4), None)
        if phase is None:
            return []
        triples = phase.get("triples", [])
        removed = phase.get("removed_assertion_ids", [])
        result = []
        for i, triple in enumerate(triples):
            a, b, c = triple
            expected = [(a, b), (b, c), (c, a)]
            added = []
            for subject, object_id in expected:
                for record in self.assertions:
                    if record.get("predicate") == "depends_on" and record.get("subject_id") == subject and record.get("object_id") == object_id:
                        added.append(record["assertion_id"])
                        break
            result.append({"triple": tuple(triple), "removed_assertion_ids": removed[3 * i:3 * i + 3], "added_assertion_ids": added})
        return result

    @property
    def removed_assertions(self) -> list[_Record]:
        if self.all_assertions is None:
            return []
        final_ids = {record["assertion_id"] for record in self.assertions}
        return [record for record in self.all_assertions if record["assertion_id"] not in final_ids]

    @property
    def base_source_revisions(self) -> list[_Record]:
        return [record for record in self.source_revisions if "/base" in record["canonical_locator"]]

    @property
    def assertion_source_revisions(self) -> list[_Record]:
        return [record for record in self.source_revisions if "/assertion/" in record["canonical_locator"]]

    def __getitem__(self, key: str) -> Any:
        if key in {"logical_checksum", "logical_snapshot_checksum"}:
            return self.manifest["logical_snapshot_checksum"]
        if key == "snapshot_id":
            return self.manifest["snapshot_id"]
        if key == "generator_trace":
            return self.trace
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def keys(self):
        return ("tier", "profile_id", "seed", "entities", "literals", "source_revisions", "assertions", "source_objects", "manifest", "trace", "generator_trace", "logical_checksum", "logical_snapshot_checksum", "snapshot_id")

    def to_dict(self, *, include_source_objects: bool = True) -> dict[str, Any]:
        out = {
            "tier": self.tier,
            "profile_id": self.profile_id,
            "seed": self.seed,
            "entities": self.entities,
            "literals": self.literals,
            "source_revisions": self.source_revisions,
            "assertions": self.assertions,
            "source_objects": self.source_objects,
            "manifest": self.manifest,
            "trace": self.trace,
        }
        if include_source_objects:
            out["source_objects"] = {k: v.decode("utf-8") for k, v in self.source_objects.items()}
        else:
            out.pop("source_objects", None)
        return out


class _Builder:
    def __init__(self, tier: str, profile_id: str, seed: int, entity_count: int):
        if profile_id not in PROFILES:
            raise GenerationError(f"unknown profile {profile_id!r}")
        if not isinstance(seed, int) or seed < 0:
            raise GenerationError("seed must be a nonnegative integer")
        if not isinstance(entity_count, int) or entity_count < 10 or entity_count % 10:
            raise GenerationError("entity_count must be a positive multiple of ten")
        self.tier = tier
        self.profile_id = profile_id
        self.seed = seed
        self.entity_count = entity_count
        self.block_count = entity_count // 10
        self.profile = PROFILES[profile_id]
        self.entities: list[dict[str, Any]] = []
        self.entity_by_id: dict[str, dict[str, Any]] = {}
        self.block_entities: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        self.entity_block: dict[str, int] = {}
        self.block_by_work_id: dict[str, int] = {}
        self.literals: dict[str, dict[str, Any]] = {}
        self.block_claim_literal: dict[int, str] = {}
        self.source_revisions: dict[str, dict[str, Any]] = {}
        self.source_objects: dict[str, bytes] = {}
        self.assertions: dict[str, dict[str, Any]] = {}
        self.assertion_meta: dict[str, dict[str, Any]] = {}
        self.triples: set[tuple[str, str, str]] = set()
        self.base_assertion_ids: list[str] = []
        self.active_assertions: set[str] = set()
        self.next_ordinal = 0
        self.trace: dict[str, Any] = {
            "generator_version": GENERATOR_VERSION,
            "prf_purposes": sorted(PRF_PURPOSES),
            "prf_choices": [],
            "removed_assertion_ids": [],
            "phases": [],
        }
        self._source_base_revision: dict[int, str] = {}
        self._init_entities()

    def _init_entities(self) -> None:
        for b in range(self.block_count):
            for r, entity_type in enumerate(_ROLE_TYPES):
                identity = _block_identity(self.tier, self.profile_id, self.seed, b, r)
                entity_id = _id_for_entity(entity_type, identity)
                record = {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "namespace": NAMESPACE,
                    "canonical_source_identity": identity,
                    "display_label_ref": None,
                }
                if entity_type == "Claim":
                    record["claim_key"] = "synthetic.block_claim"
                if entity_id in self.entity_by_id:
                    raise GenerationError("entity ID collision")
                self.entity_by_id[entity_id] = record
                self.entity_block[entity_id] = b
                self.entities.append(record)
                self.block_entities[b][r] = record
                if r == 0:
                    self.block_by_work_id[entity_id] = b
            base_locator = self._base_locator(b)
            base_record = {
                "block": b,
                "generator": GENERATOR_VERSION,
                "profile": self.profile_id,
                "seed": self.seed,
                "tier": self.tier,
            }
            self._new_source_revision(b, base_locator, _json_bytes(base_record), base=True)
            literal_value = f"block={b};profile={self.profile_id};seed={self.seed}"
            literal_id = _id_for_literal("text", literal_value)
            self.literals[literal_id] = {
                "literal_id": literal_id,
                "literal_type": "text",
                "value": literal_value,
            }
            self.block_claim_literal[b] = literal_id

    def _base_locator(self, block: int) -> str:
        return f"cyax://0168/scale/2.2/{self.tier}/{self.profile_id}/{self.seed}/block/{block}/base"

    def _assertion_locator(self, ordinal: int, block: int) -> str:
        return f"cyax://0168/scale/2.2/{self.tier}/{self.profile_id}/{self.seed}/block/{block}/assertion/{ordinal}"

    def _new_source_revision(self, block: int, locator: str, content: bytes, *, base: bool = False) -> str:
        digest = hashlib.sha256(content).hexdigest()
        revision_id = _id_for_revision("synthetic_fixture", locator, digest, None)
        record = {
            "source_revision_id": revision_id,
            "source_entity_id": self.block_entities[block][9]["entity_id"],
            "source_kind": "synthetic_fixture",
            "canonical_locator": locator,
            "object_sha256": digest,
            "object_byte_count": len(content),
            "source_event_at": None,
            "observed_at": OBSERVED_AT,
            "actor_id": None,
            "authority_class": "ordinary_record",
            "authority_derivation_rule_id": AUTHORITY_RULE,
        }
        old = self.source_revisions.get(revision_id)
        if old is not None and old != record:
            raise GenerationError("source revision collision")
        self.source_revisions[revision_id] = record
        self.source_objects[digest] = content
        if base:
            self._source_base_revision[block] = revision_id
        return revision_id

    def _record_assertion(self, *, block: int, predicate: str, object_id: str | None, literal_ref: str | None,
                          phase: str, valid_from: str | None = None, dispute_state: str = "undisputed",
                          statement_source_block: int | None = None, ordinal_override: int | None = None,
                          source_revision_override: str | None = None) -> str:
        if predicate not in _SIGNATURES:
            raise GenerationError(f"unknown predicate: {predicate}")
        if (object_id is None) == (literal_ref is None):
            raise GenerationError("assertion must contain exactly one object or literal")
        subject = self.block_entities[block][0] if phase in {"dependency", "cycle", "supersession"} else None
        # Callers for motifs and additional records pass the actual block role
        # through ``_record_assertion_for_entities``; this method is retained as
        # a compact common record constructor.
        if subject is None:
            raise GenerationError("internal assertion subject missing")
        return self._record_assertion_for_entities(
            subject_id=subject["entity_id"], block=statement_source_block if statement_source_block is not None else block,
            predicate=predicate, object_id=object_id, literal_ref=literal_ref,
            phase=phase, valid_from=valid_from, dispute_state=dispute_state,
            ordinal_override=ordinal_override, source_revision_override=source_revision_override,
        )

    def _record_assertion_for_entities(self, *, subject_id: str, block: int, predicate: str,
                                       object_id: str | None, literal_ref: str | None, phase: str,
                                       valid_from: str | None = None, dispute_state: str = "undisputed",
                                       ordinal_override: int | None = None,
                                       source_revision_override: str | None = None) -> str:
        if (object_id is None) == (literal_ref is None):
            raise GenerationError("assertion must contain exactly one object or literal")
        subject_record = self.entity_by_id[subject_id]
        if object_id is not None:
            object_record = self.entity_by_id.get(object_id)
            if object_record is None:
                raise GenerationError("dangling assertion object")
            allowed_s, allowed_o, literal_allowed = _SIGNATURES[predicate]
            if subject_record["entity_type"] not in allowed_s or object_record["entity_type"] not in allowed_o or literal_allowed:
                raise GenerationError("predicate signature violation")
        else:
            if predicate != "documented_in" or subject_record["entity_type"] != "Claim" or literal_ref not in self.literals:
                raise GenerationError("literal predicate signature violation")
        ordinal = self.next_ordinal if ordinal_override is None else ordinal_override
        if ordinal_override is None:
            self.next_ordinal += 1
        statement = {
            "assertion_ordinal": ordinal,
            "generator": GENERATOR_VERSION,
            "literal_identity": literal_ref,
            "object_identity": object_id,
            "predicate": predicate,
            "profile": self.profile_id,
            "seed": self.seed,
            "subject_identity": subject_id,
            "tier": self.tier,
        }
        content = _json_bytes(statement)
        source_locator = self._assertion_locator(ordinal, block)
        source_revision_id = source_revision_override or self._new_source_revision(block, source_locator, content)
        record = {
            "assertion_ordinal": ordinal,
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "literal_ref": literal_ref,
            "source_revision_id": source_revision_id,
            "source_locator": source_locator,
            "source_event_at": None,
            "asserted_at": OBSERVED_AT,
            "valid_from": valid_from,
            "valid_to": None,
            "validity_basis": "explicit" if valid_from is not None else "unknown",
            "authority_class": "ordinary_record",
            "authority_derivation_rule_id": AUTHORITY_RULE,
            "origin": "source_direct",
            "curation_state": "independently_reviewed",
            "review_state": "not_required",
            "epistemic_state": "supported",
            "dispute_state": dispute_state,
        }
        assertion_id = _id_for_assertion(record)
        if assertion_id in self.assertions and self.assertions[assertion_id] != record:
            raise GenerationError("assertion ID collision")
        record["assertion_id"] = assertion_id
        self.assertions[assertion_id] = record
        self.assertion_meta[assertion_id] = {"phase": phase, "base": phase == "base"}
        self.active_assertions.add(assertion_id)
        if object_id is not None:
            self.triples.add((subject_id, predicate, object_id))
        return assertion_id

    def _add_base(self) -> None:
        for b in range(self.block_count):
            e = self.block_entities[b]
            pairs = [
                (1, "governs", 2, None), (0, "requires", 2, None), (0, "requires", 3, None),
                (2, "requires", 4, None), (3, "requires", 5, None), (4, "implements", 2, None),
                (5, "implements", 3, None), (6, "verifies", 4, None), (8, "verifies", 3, None),
                (8, "supports", 7, None), (7, "concerns", 0, None), (7, "documented_in", None, self.block_claim_literal[b]),
                (8, "documented_in", 9, None),
            ]
            for sr, predicate, orole, literal in pairs:
                aid = self._record_assertion_for_entities(
                    subject_id=e[sr]["entity_id"], block=b, predicate=predicate,
                    object_id=e[orole]["entity_id"] if orole is not None else None,
                    literal_ref=literal, phase="base",
                )
                self.base_assertion_ids.append(aid)

    def _work_items(self, blocks: Iterable[int]) -> list[str]:
        return sorted((self.block_entities[b][0]["entity_id"] for b in blocks), key=_primary)

    def _assertion_triple_present(self, subject_id: str, predicate: str, object_id: str) -> bool:
        return (subject_id, predicate, object_id) in self.triples

    def _phase_dependencies(self, isolated: set[int]) -> tuple[list[int], list[list[int]], list[str]]:
        # Components are consecutive block ordinals.  Phase 3 has a separate,
        # explicit primary-ID visit order for its source positions.
        connected = [b for b in range(self.block_count) if b not in isolated]
        connected_primary = sorted(connected, key=lambda b: _primary(self.block_entities[b][0]["entity_id"]))
        width = max(int(self.profile["path_depth"]) + 1, 64)
        components = [connected[i:i + width] for i in range(0, len(connected), width)]
        component_of = {b: i for i, comp in enumerate(components) for b in comp}
        chain_ids: list[str] = []
        for comp in components:
            works = [self.block_entities[b][0]["entity_id"] for b in comp]
            chain_length = min(int(self.profile["path_depth"]), len(works) - 1)
            for left, right in zip(works[:chain_length], works[1:chain_length + 1]):
                aid = self._record_assertion_for_entities(subject_id=left, block=self.block_by_work_id[left], predicate="depends_on", object_id=right, literal_ref=None, phase="dependency_chain")
                self.assertion_meta[aid]["chain"] = True
                chain_ids.append(aid)
        self.trace["phases"].append({
            "phase": 2,
            "component_blocks": [list(comp) for comp in components],
            "chain_assertion_ids": list(chain_ids),
            "chain_count": len(chain_ids),
        })
        dep_quota = min(int(self.profile["dependency_fanout"]) * len(connected), 30 * self.block_count)
        if self.tier not in {"C0", "C1", "C2", "C3", "T0", "T1", "T2", "T3", "T4"}:
            # Tiny synthetic probes can have fewer unique same-component
            # targets than the production fan-out.  Cap only those probes so
            # the exhaustive duplicate-rejection rule terminates; frozen
            # campaign tiers retain the exact quota above.
            dep_quota = min(dep_quota, sum((len(comp) - 1) ** 2 for comp in components) - len(components))
        additional = dep_quota - len(chain_ids)
        if additional < 0:
            raise GenerationError("dependency quota is below required chain edges")
        cross_count = math.floor(float(self.profile["cross_link_rate"]) * dep_quota)
        if self.tier not in {"C0", "C1", "C2", "C3", "T0", "T1", "T2", "T3", "T4"} and len(components) < 2:
            cross_count = 0
        if cross_count > additional:
            raise GenerationError("cross-link quota exceeds additional dependency positions")
        all_nonisolated = self._work_items(connected)
        block_component = {b: component_of[b] for b in connected}
        # Detect an exhausted no-replacement domain before invoking rejection
        # sampling.  A frozen candidate vector is never mutated, so an
        # impossible unique-target quota must fail rather than spin forever.
        positions_by_source: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
        for i in range(additional):
            source_block = connected_primary[i % len(connected_primary)]
            source_id = self.block_entities[source_block][0]["entity_id"]
            comp_id = block_component[source_block]
            if i < cross_count:
                domain = [wid for wid in all_nonisolated if block_component[self.block_by_work_id[wid]] != comp_id]
                purpose = "cross_component_dependency_target"
            else:
                domain = self._work_items(components[comp_id])
                purpose = "dependency_target"
            positions_by_source[source_id].append((domain, purpose))
        for source_id, positions in positions_by_source.items():
            domains = {target for domain, _ in positions for target in domain if target != source_id and not self._assertion_triple_present(source_id, "depends_on", target)}
            if len(positions) > len(domains):
                raise GenerationError("dependency target domain exhausted by duplicate rejection")
        for i in range(additional):
            source_block = connected_primary[i % len(connected_primary)] if connected_primary else None
            if source_block is None:
                raise GenerationError("no connected blocks")
            source_id = self.block_entities[source_block][0]["entity_id"]
            comp_id = block_component[source_block]
            if i < cross_count:
                candidates = [wid for wid in all_nonisolated if block_component[self.block_by_work_id[wid]] != comp_id]
                purpose = "cross_component_dependency_target"
            else:
                candidates = [wid for wid in self._work_items(components[comp_id])]
                purpose = "dependency_target"
            if not candidates:
                raise GenerationError("empty dependency candidate vector")
            def reject(target: str) -> bool:
                return target == source_id or self._assertion_triple_present(source_id, "depends_on", target)
            target, counter = prf_choice(candidates, seed=self.seed, profile_id=self.profile_id, purpose=purpose, ordinal=i, reject=reject)
            self.trace["prf_choices"].append({"phase": 3, "purpose": purpose, "ordinal": i, "counter": counter, "subject_id": source_id, "object_id": target})
            self._record_assertion_for_entities(subject_id=source_id, block=source_block, predicate="depends_on", object_id=target, literal_ref=None, phase="dependency")
        self.trace["phases"].append({"phase": 3, "isolated_blocks": sorted(isolated), "connected_blocks": len(connected), "dependency_quota": dep_quota, "chain_count": len(chain_ids), "additional_count": additional, "cross_link_count": cross_count})
        return connected, components, chain_ids

    def _phase_cycles(self, connected: list[int], components: list[list[int]]) -> set[int]:
        connected_work = self._work_items(connected)
        outgoing: dict[str, list[str]] = defaultdict(list)
        for aid in self.active_assertions:
            rec = self.assertions[aid]
            if rec["predicate"] == "depends_on" and not self.assertion_meta[aid].get("chain", False):
                outgoing[rec["subject_id"]].append(aid)
        need = math.floor(float(self.profile["cycle_rate"]) * len(connected))
        if need == 0:
            self.trace["phases"].append({"phase": 4, "cycle_count": 0, "removed_assertion_ids": [], "triples": []})
            return set()
        triples: list[tuple[str, str, str]] = []
        for i in range(0, max(0, len(connected_work) - 2)):
            tri = tuple(connected_work[i:i + 3])
            if any(not outgoing.get(wid) for wid in tri):
                continue
            a, b, c = tri
            if any(self._assertion_triple_present(x, "depends_on", y) for x, y in ((a, b), (b, c), (c, a))):
                continue
            triples.append(tri)
        chosen: list[tuple[str, str, str]] = []
        used: set[str] = set()
        for tri in triples:
            if any(w in used for w in tri):
                continue
            chosen.append(tri)
            used.update(tri)
            if len(chosen) == need:
                break
        if len(chosen) != need:
            raise GenerationError(f"cycle quota cannot be met: {len(chosen)} != {need}")
        cycle_blocks: set[int] = set()
        removed: list[str] = []
        for a, b, c in chosen:
            for subject, target in ((a, b), (b, c), (c, a)):
                candidates = [aid for aid in outgoing[subject] if aid in self.active_assertions]
                candidates.sort(key=lambda aid: (_primary(aid), _primary(self.assertions[aid]["subject_id"]), _primary(self.assertions[aid]["object_id"] or "")))
                if not candidates:
                    raise GenerationError("cycle member has no removable outgoing edge")
                removed_id = candidates[0]
                self.active_assertions.remove(removed_id)
                removed_record = self.assertions[removed_id]
                if removed_record["object_id"] is not None:
                    self.triples.discard((removed_record["subject_id"], removed_record["predicate"], removed_record["object_id"]))
                removed.append(removed_id)
                outgoing[subject].remove(removed_id)
            for subject, target in ((a, b), (b, c), (c, a)):
                sb = self.block_by_work_id[subject]
                aid = self._record_assertion_for_entities(subject_id=subject, block=sb, predicate="depends_on", object_id=target, literal_ref=None, phase="cycle")
                self.assertion_meta[aid]["cycle"] = True
            cycle_blocks.update(self.block_by_work_id[w] for w in (a, b, c))
        self.trace["removed_assertion_ids"].extend(removed)
        self.trace["phases"].append({"phase": 4, "cycle_count": len(chosen), "removed_assertion_ids": removed, "triples": [list(x) for x in chosen]})
        return cycle_blocks

    def _phase_supersession(self, connected: list[int], cycle_blocks: set[int]) -> None:
        depth = int(self.profile["supersession_depth"])
        eligible = [b for b in connected if b not in cycle_blocks]
        works = self._work_items(eligible)
        chains = [works[i:i + depth] for i in range(0, len(works), depth) if len(works[i:i + depth]) == depth]
        count = 0
        for chain in chains:
            for pos, (previous, successor) in enumerate(zip(chain, chain[1:]), start=1):
                sb = self.block_by_work_id[successor]
                valid_from = f"2000-01-02T00:00:{pos:02d}.000000Z"
                self._record_assertion_for_entities(subject_id=successor, block=sb, predicate="supersedes", object_id=previous, literal_ref=None, phase="supersession", valid_from=valid_from)
                count += 1
        self.trace["phases"].append({"phase": 5, "chain_count": len(chains), "supersession_count": count, "depth": depth})

    def _phase_disputes(self) -> None:
        claim_ids = sorted((self.block_entities[b][7]["entity_id"] for b in range(self.block_count)), key=_primary)
        claim_block = {self.block_entities[b][7]["entity_id"]: b for b in range(self.block_count)}
        pair_count = math.floor(float(self.profile["dispute_rate"]) * self.block_count)
        used = claim_ids[:2 * pair_count]
        for i in range(pair_count):
            second, first = used[2 * i + 1], used[2 * i]
            sb = claim_block[second]
            self._record_assertion_for_entities(subject_id=second, block=sb, predicate="contradicts", object_id=first, literal_ref=None, phase="dispute", dispute_state="disputed")
        self.trace["phases"].append({"phase": 6, "dispute_pair_count": pair_count, "contradiction_count": pair_count})

    def _phase_parallel(self) -> None:
        count = math.floor(float(self.profile["parallel_evidence_rate"]) * 13 * self.block_count)
        base = sorted(self.base_assertion_ids, key=_primary)[:count]
        for original_id in base:
            original = self.assertions[original_id]
            subject_id = original["subject_id"]
            block = self.block_by_work_id.get(subject_id)
            if block is None:
                block = next(b for b, roles in self.block_entities.items() if any(e["entity_id"] == subject_id for e in roles.values()))
            aid = self._record_assertion_for_entities(
                subject_id=subject_id, block=block, predicate=original["predicate"],
                object_id=original["object_id"], literal_ref=original["literal_ref"], phase="parallel",
            )
            self.assertion_meta[aid]["parallel_of"] = original_id
        self.trace["phases"].append({"phase": 7, "parallel_count": count, "parallel_of": base})

    def _signature_valid_concerns_pairs(self) -> list[tuple[str, str, int]]:
        pairs: list[tuple[str, str, int]] = []
        allowed_s, allowed_o, _ = _SIGNATURES["concerns"]
        for b, roles in self.block_entities.items():
            entities = [roles[r] for r in range(9)]  # source is not a concerns subject/object here
            for subject in entities:
                if subject["entity_type"] not in allowed_s:
                    continue
                for object_ in entities:
                    if object_["entity_type"] not in allowed_o or subject["entity_id"] == object_["entity_id"]:
                        continue
                    pairs.append((subject["entity_id"], object_["entity_id"], b))
        pairs.sort(key=lambda pair: canonical_frame([pair[0], pair[1]]))
        return pairs

    def _phase_filler(self) -> None:
        target = 50 * self.block_count
        remaining = target - len(self.active_assertions)
        if remaining < 0:
            raise GenerationError("profile phases exceed exact assertion quota")
        pairs = self._signature_valid_concerns_pairs()
        candidates = sorted(((s, o) for s, o, _ in pairs), key=canonical_frame)
        if not candidates and remaining:
            raise GenerationError("empty concerns filler vector")
        for i in range(remaining):
            def reject(pair: tuple[str, str]) -> bool:
                return pair[0] == pair[1] or self._assertion_triple_present(pair[0], "concerns", pair[1])
            pair, counter = prf_choice(candidates, seed=self.seed, profile_id=self.profile_id, purpose="fill_concerns_pair", ordinal=i, reject=reject, presorted=True)
            subject, object_id = pair
            block = self.entity_block[subject]
            self.trace["prf_choices"].append({"phase": 8, "purpose": "fill_concerns_pair", "ordinal": i, "counter": counter, "subject_id": subject, "object_id": object_id})
            self._record_assertion_for_entities(subject_id=subject, block=block, predicate="concerns", object_id=object_id, literal_ref=None, phase="filler")
        self.trace["phases"].append({"phase": 8, "filler_count": remaining, "candidate_count": len(candidates), "target_assertions": target})

    def _validate(self, isolated: set[int]) -> None:
        expected_entities = self.entity_count
        expected_assertions = 50 * self.block_count
        if len(self.entities) != expected_entities:
            raise GenerationError("entity count mismatch")
        if len(self.active_assertions) != expected_assertions:
            raise GenerationError(f"assertion count mismatch: {len(self.active_assertions)} != {expected_assertions}")
        if len({e["entity_id"] for e in self.entities}) != len(self.entities):
            raise GenerationError("duplicate entity ID")
        for aid in self.active_assertions:
            rec = self.assertions[aid]
            if aid != _id_for_assertion(rec):
                raise GenerationError("assertion ID does not match semantic preimage")
            if rec["object_id"] is not None and rec["object_id"] not in self.entity_by_id:
                raise GenerationError("dangling assertion reference")
            if rec["literal_ref"] is not None and rec["literal_ref"] not in self.literals:
                raise GenerationError("dangling literal reference")
            if rec["source_revision_id"] not in self.source_revisions:
                raise GenerationError("dangling source revision")
        # Removed assertions must not be emitted as active records or source
        # provenance.  Base revisions are always retained.
        expected_revisions = set(self._source_base_revision.values()) | {self.assertions[aid]["source_revision_id"] for aid in self.active_assertions}
        # Cycle substitution can have already allocated provenance bytes for a
        # removed phase-3 edge.  The final physical source payload contains
        # only base revisions and provenance for surviving assertions.
        self.source_revisions = {rid: record for rid, record in self.source_revisions.items() if rid in expected_revisions}
        if set(self.source_revisions) != expected_revisions:
            raise GenerationError("source revisions are not base-plus-final provenance")
        used_digests = {self.source_revisions[rid]["object_sha256"] for rid in expected_revisions}
        self.source_objects = {digest: content for digest, content in self.source_objects.items() if digest in used_digests}
        for b in isolated:
            if b not in range(self.block_count):
                raise GenerationError("invalid isolated block")

    def build(self) -> Snapshot:
        self._add_base()
        isolated_count = math.floor(float(self.profile["isolated_rate"]) * self.block_count)
        work_sorted = sorted(range(self.block_count), key=lambda b: _primary(self.block_entities[b][0]["entity_id"]))
        isolated = set(work_sorted[-isolated_count:]) if isolated_count else set()
        self.trace["phases"].append({
            "phase": 1,
            "isolated_blocks": sorted(isolated),
            "isolated_count": len(isolated),
        })
        connected, components, _ = self._phase_dependencies(isolated)
        cycle_blocks = self._phase_cycles(connected, components)
        self._phase_supersession(connected, cycle_blocks)
        self._phase_disputes()
        self._phase_parallel()
        self._phase_filler()
        self._validate(isolated)
        entities = sorted(self.entities, key=lambda r: _primary(r["entity_id"]))
        literals = sorted(self.literals.values(), key=lambda r: _primary(r["literal_id"]))
        revisions = sorted(self.source_revisions.values(), key=lambda r: _primary(r["source_revision_id"]))
        assertions = sorted((self.assertions[aid] for aid in self.active_assertions), key=lambda r: _primary(r["assertion_id"]))
        checksum = logical_snapshot_checksum(entities, literals, revisions, assertions, self.source_objects)
        manifest = {
            "generator_version": GENERATOR_VERSION,
            "namespace": NAMESPACE,
            "tier": self.tier,
            "profile_id": self.profile_id,
            "seed": self.seed,
            "profile": dict(self.profile),
            "schema_version": SCHEMA_VERSION,
            "authority_rule_version": AUTHORITY_RULE_VERSION,
            "semantic_evaluator_rule_version": EVALUATOR_RULE_VERSION,
            "claim_key_registry": [{
                "claim_key": "synthetic.block_claim",
                "literal_type": "text",
                "semantic_slot": "synthetic_fixture_block_statement",
            }],
            "owner_allowlist": [],
            "owner_decision_events": [],
            "entity_count": len(entities),
            "literal_count": len(literals),
            "source_revision_count": len(revisions),
            "assertion_count": len(assertions),
            "logical_snapshot_checksum": checksum,
            "snapshot_id": f"cyax-snapshot-sha256:{checksum}",
        }
        all_assertions = list(self.assertions.values())
        return Snapshot(self.tier, self.profile_id, self.seed, entities, literals, revisions, assertions, dict(self.source_objects), manifest, self.trace, all_assertions)


def logical_snapshot_checksum(entities: Sequence[Mapping[str, Any]], literals: Sequence[Mapping[str, Any]], source_revisions: Sequence[Mapping[str, Any]], assertions: Sequence[Mapping[str, Any]], source_objects: Mapping[str, bytes] | None = None) -> str:
    """Compute the explicit semantic checksum from canonical snapshot records."""

    semantic_entities = []
    for record in sorted(entities, key=lambda r: _primary(str(r["entity_id"]))):
        semantic_entities.append({k: v for k, v in record.items() if k != "display_label_ref"})
    semantic_literals = sorted((dict(r) for r in literals), key=lambda r: _primary(str(r["literal_id"])))
    semantic_assertions = sorted((
        {k: v for k, v in dict(r).items() if k != "assertion_ordinal"}
        for r in assertions
    ), key=lambda r: _primary(str(r["assertion_id"])))
    referenced_revision_ids = {str(r["source_revision_id"]) for r in semantic_assertions}
    semantic_revisions = sorted((dict(r) for r in source_revisions if r["source_revision_id"] in referenced_revision_ids), key=lambda r: _primary(str(r["source_revision_id"])))
    object_pairs = []
    if source_objects is not None:
        referenced_digests = {str(r["object_sha256"]) for r in semantic_revisions}
        object_pairs = sorted([[digest, len(source_objects[digest])] for digest in referenced_digests if digest in source_objects], key=lambda p: _primary(p[0]))
    source_projection = _digest_frame([
        "cyax-source-bundle-projection-v1",
        semantic_revisions,
        object_pairs,
        [],  # actor/role records: synthetic fixtures have none
        [],  # owner-decision registry: synthetic fixtures have none
    ])
    projection = [
        "cyax-logical-snapshot-v1", SCHEMA_VERSION, source_projection,
        AUTHORITY_RULE_VERSION, EVALUATOR_RULE_VERSION,
        ["semantic_entities", semantic_entities],
        ["semantic_literals", semantic_literals],
        ["semantic_source_revisions", semantic_revisions],
        ["semantic_assertions", semantic_assertions],
    ]
    return _digest_frame(projection)


def semantic_projection_checksum(snapshot: Snapshot) -> str:
    return logical_snapshot_checksum(snapshot.entities, snapshot.literals, snapshot.source_revisions, snapshot.assertions, snapshot.source_objects)


def compute_logical_snapshot_checksum(snapshot: Snapshot) -> str:
    return semantic_projection_checksum(snapshot)


def physical_source_revisions(snapshot: Snapshot) -> list[_Record]:
    """Return the complete physical source-revision sequence."""
    return list(snapshot.source_revisions)


def semantic_projection(snapshot: Snapshot) -> dict[str, Any]:
    """Return the explicit reachable semantic projection used for identity."""
    semantic_entities = [
        {k: v for k, v in record.items() if k != "display_label_ref"}
        for record in sorted(snapshot.entities, key=lambda r: _primary(str(r["entity_id"])))
    ]
    semantic_literals = sorted(
        (dict(record) for record in snapshot.literals),
        key=lambda r: _primary(str(r["literal_id"])),
    )
    semantic_assertions = sorted(
        ({k: v for k, v in dict(record).items() if k != "assertion_ordinal"} for record in snapshot.assertions),
        key=lambda r: _primary(str(r["assertion_id"])),
    )
    referenced = {str(record["source_revision_id"]) for record in semantic_assertions}
    semantic_revisions = sorted(
        (dict(record) for record in snapshot.source_revisions if record["source_revision_id"] in referenced),
        key=lambda r: _primary(str(r["source_revision_id"])),
    )
    return {
        "semantic_entities": semantic_entities,
        "semantic_literals": semantic_literals,
        "semantic_source_revisions": semantic_revisions,
        "semantic_assertions": semantic_assertions,
    }


def compute_semantic_source_bundle_projection_checksum(snapshot: Snapshot) -> str:
    """Hash the reachable synthetic source-bundle projection."""
    projection = semantic_projection(snapshot)
    digests = {str(record["object_sha256"]) for record in projection["semantic_source_revisions"]}
    objects = sorted(
        [[digest, len(snapshot.source_objects[digest])] for digest in digests if digest in snapshot.source_objects],
        key=lambda pair: _primary(pair[0]),
    )
    return _digest_frame([
        "cyax-source-bundle-projection-v1",
        projection["semantic_source_revisions"],
        objects,
        [],
        [],
    ])


def dependency_target_rejects(subject_id: str, object_id: str, existing_triples: Iterable[tuple[str, str, str]]) -> bool:
    """Apply the complete dependency-target rejection row."""
    return subject_id == object_id or (subject_id, "depends_on", object_id) in set(existing_triples)


def fill_concerns_pair_rejects(subject_id: str, object_id: str, existing_triples: Iterable[tuple[str, str, str]]) -> bool:
    """Apply the complete concerns-filler rejection row."""
    return subject_id == object_id or (subject_id, "concerns", object_id) in set(existing_triples)


def generate_snapshot(tier: str | int, profile_id: str, seed: int, entity_count: int | None = None) -> Snapshot:
    """Generate one deterministic frozen snapshot.

    ``tier`` may be ``T0``–``T4`` or an integer label used by tiny conformance
    fixtures.  Integer/custom tiers never grant access to decision fixtures;
    they only alter the synthetic identity and record counts.
    """

    if isinstance(tier, int):
        label = f"T{tier}"
    else:
        label = str(tier)
    if label in {"C0", "C1", "C2", "C3", "T0", "T1", "T2", "T3", "T4"}:
        # Registered calibration/decision and benchmark tiers remain outside
        # this G1 implementation rung.  Conformance callers must use an
        # unregistered synthetic label with an explicit tiny size.
        raise GenerationError(f"registered tier {label} generation/materialization is not authorized during CYAX-0168 G1")
    if entity_count is None:
        if label not in TIERS:
            raise GenerationError(f"unknown tier {tier!r}; provide entity_count for a tiny fixture")
        entity_count = TIERS[label]["entity_count"]
    return _Builder(label, profile_id, seed, entity_count).build()


# Short aliases used by focused conformance harnesses.
generate = generate_snapshot
build_snapshot = generate_snapshot


__all__ = [
    "GENERATOR_VERSION", "NAMESPACE", "PROFILES", "TIERS", "PRF_PURPOSES",
    "ASSERTION_PREIMAGE_KEYS", "FIXED_TIME", "Snapshot", "GenerationError", "GenerationFailure",
    "canonical_frame", "frame", "canonical_json_bytes", "entity_id", "literal_id", "source_revision_id", "assertion_id", "sha256_hex",
    "prf_digest", "prf_choice", "prf_select", "logical_snapshot_checksum", "semantic_projection_checksum", "compute_logical_snapshot_checksum", "physical_source_revisions", "semantic_projection", "compute_semantic_source_bundle_projection_checksum", "dependency_target_rejects", "fill_concerns_pair_rejects", "generate_snapshot", "generate", "build_snapshot",
]
