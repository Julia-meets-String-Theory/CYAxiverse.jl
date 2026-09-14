#!/usr/bin/env python3
"""CYAX-0168 common semantic/evidence core.

Implements, standard-library only, the frozen contract described in
``specs/0168-structured-graph-materialization/spec.md``: the closed
entity/literal/enum/predicate/Claim registry model, canonical
domain-separated framing and stable IDs, mechanical owner-decision
authority derivation, and the backend-independent evaluator primitives
(admissibility, temporal inclusion, supersession, dependency staleness)
that ``snapshot.py``, ``update.py``, and ``publication.py`` build on.

Scope boundary (CYAX-0168 G1, Worker A+B): this module implements the
common *identity and validation* layer only. Full Q01-Q12 population
selection, gold, and RetrievalBundle path-witness construction are a
separate worker's self-contained module (``evaluator.py``) per this
programme's deliberate no-shared-import convention between parallel G1
workers; this module does not import it and is not imported by it. This
module never materializes, executes, profiles, or inspects a T0-T4
fixture through either backend, and never enters G2+ semantic-parity
territory.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import re
import struct
import unicodedata
from collections.abc import Mapping as ABCMapping
from typing import Any, Mapping, Sequence


class SemanticError(ValueError):
    """Raised when a value violates a CYAX-0168 frozen semantic contract."""


# ---------------------------------------------------------------------------
# Canonical domain-separated framing (spec: "Canonical encoding and stable IDs")
# ---------------------------------------------------------------------------


def _u64be(n: int) -> bytes:
    if n < 0 or n > 0xFFFFFFFFFFFFFFFF:
        raise SemanticError(f"length/count out of u64 range: {n}")
    return struct.pack(">Q", n)


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _minimal_base10(value: int) -> str:
    # Python's str(int) never emits a leading zero or a "+" sign, and uses
    # a single leading "-" for negatives, matching "minimal-base10-ASCII".
    return str(value)


def frame(value: Any) -> bytes:
    """Encode ``value`` using the CYAX-0168 canonical domain-separated frame.

    Supported Python types: ``None``, ``bool``, ``int`` (arbitrary
    precision, not ``bool``), ``bytes``, ``str`` (NFC-normalized),
    ``list``/``tuple`` (array), and ``dict`` (object, NFC keys sorted by
    UTF-8 bytes). ``float`` and any other type are rejected.
    """

    if value is None:
        return b"N" + _u64be(0)
    if isinstance(value, bool):
        return b"B" + _u64be(1) + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        raw = _minimal_base10(value).encode("ascii")
        return b"I" + _u64be(len(raw)) + raw
    if isinstance(value, float):
        raise SemanticError("floats are forbidden in canonical framing")
    if isinstance(value, bytes):
        return b"X" + _u64be(len(value)) + value
    if isinstance(value, str):
        raw = _nfc(value).encode("utf-8")
        return b"S" + _u64be(len(raw)) + raw
    if isinstance(value, (list, tuple)):
        parts = [frame(item) for item in value]
        return b"A" + _u64be(len(parts)) + b"".join(parts)
    if isinstance(value, ABCMapping):
        return _frame_object(value)
    raise SemanticError(f"unsupported type for canonical framing: {type(value)!r}")


def _frame_object(value: Mapping[str, Any]) -> bytes:
    normalized: dict[str, str] = {}
    for key in value:
        if not isinstance(key, str):
            raise SemanticError("canonical object keys must be strings")
        nfc_key = _nfc(key)
        if nfc_key in normalized:
            raise SemanticError(f"canonical object has colliding NFC key: {nfc_key!r}")
        normalized[nfc_key] = key
    ordered = sorted(normalized.items(), key=lambda pair: pair[0].encode("utf-8"))
    body = b"".join(frame(nfc_key) + frame(value[orig_key]) for nfc_key, orig_key in ordered)
    return b"O" + _u64be(len(ordered)) + body


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame_hash(value: Any) -> str:
    """SHA-256 hex digest of ``frame(value)``."""

    return sha256_hex(frame(value))


# ---------------------------------------------------------------------------
# Stable ID derivation (spec: "Canonical encoding and stable IDs")
# ---------------------------------------------------------------------------


def compute_entity_id(namespace: str, entity_type: str, canonical_source_identity: Any) -> str:
    payload = frame(["cyax-entity-v1", namespace, entity_type, canonical_source_identity])
    return f"cyax-entity-sha256:{sha256_hex(payload)}"


def compute_source_revision_id(
    source_kind: str,
    canonical_locator: str,
    object_sha256: str,
    source_event_at: str | None,
) -> str:
    payload = frame(
        ["cyax-source-revision-v1", source_kind, canonical_locator, object_sha256, source_event_at]
    )
    return f"cyax-source-revision-sha256:{sha256_hex(payload)}"


def compute_literal_id(literal_type: str, canonical_value: Any) -> str:
    payload = frame(["cyax-literal-v1", literal_type, canonical_value])
    return f"cyax-literal-sha256:{sha256_hex(payload)}"


_ASSERTION_ID_FIELDS = (
    "subject_id",
    "predicate",
    "object_id",
    "literal_ref",
    "source_revision_id",
    "source_locator",
    "source_event_at",
    "asserted_at",
    "valid_from",
    "valid_to",
    "validity_basis",
    "authority_class",
    "authority_derivation_rule_id",
    "origin",
    "curation_state",
    "review_state",
    "epistemic_state",
    "dispute_state",
)


def assertion_semantic_preimage(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete canonical semantic assertion revision (excluding
    ``assertion_id`` itself), with every required field present."""

    missing = [name for name in _ASSERTION_ID_FIELDS if name not in fields]
    if missing:
        raise SemanticError(f"assertion preimage missing required fields: {missing}")
    return {name: fields[name] for name in _ASSERTION_ID_FIELDS}


def compute_assertion_id(fields: Mapping[str, Any]) -> str:
    preimage = assertion_semantic_preimage(fields)
    payload = frame(["cyax-assertion-v2", preimage])
    return f"cyax-assertion-sha256:{sha256_hex(payload)}"


# ---------------------------------------------------------------------------
# Frozen vocabularies (spec: "Common typed model", "Frozen vocabularies")
# ---------------------------------------------------------------------------

ENTITY_TYPES = frozenset(
    {
        "WorkItem",
        "Decision",
        "Specification",
        "Requirement",
        "Implementation",
        "Verification",
        "Claim",
        "Artifact",
        "Source",
    }
)

LITERAL_TYPES = frozenset(
    {
        "text",
        "integer",
        "boolean",
        "timestamp",
        "sha256",
        "work_item_state",
        "pull_request_state",
        "review_verdict",
    }
)

WORK_ITEM_STATES = frozenset({"open", "closed"})
PULL_REQUEST_STATES = frozenset({"draft_open", "open", "closed_unmerged", "merged"})
REVIEW_VERDICTS = frozenset({"pending", "pass", "changes_required", "rejected"})

SOURCE_KINDS = frozenset(
    {
        "github_issue",
        "github_issue_comment",
        "github_pull_request",
        "github_pull_request_comment",
        "github_review",
        "git_commit",
        "repository_file",
        "verification_artifact",
        "external_document",
        "synthetic_fixture",
    }
)

AUTHORITY_CLASSES = frozenset(
    {
        "owner_decision",
        "approved_specification",
        "canonical_work_item",
        "merged_implementation",
        "verification_evidence",
        "external_reference",
        "ordinary_record",
        "agent_proposal",
    }
)

ORIGINS = frozenset({"source_direct", "curator_interpretation", "rule_derived"})
CURATION_STATES = frozenset({"unreviewed", "curator_checked", "independently_reviewed"})
REVIEW_STATES = frozenset({"not_required", "pending", "passed", "changes_required", "rejected"})
EPISTEMIC_STATES = frozenset(
    {"extracted", "unresolved", "supported", "verified", "accepted", "rejected"}
)
DISPUTE_STATES = frozenset({"undisputed", "disputed", "resolved_upheld", "resolved_rejected"})
VALIDITY_BASES = frozenset({"explicit", "source_event", "unknown"})

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_timestamp_instant(value: str) -> dt.datetime:
    if not isinstance(value, str) or not TIMESTAMP_RE.match(value):
        raise SemanticError(f"timestamp must be YYYY-MM-DDTHH:MM:SS.ffffffZ: {value!r}")
    try:
        # datetime rejects second=60 (no leap seconds) and out-of-range fields.
        parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise SemanticError(f"invalid timestamp instant: {value!r}") from exc
    return parsed.replace(tzinfo=dt.timezone.utc)


def validate_literal_value(literal_type: str, value: Any) -> None:
    if literal_type not in LITERAL_TYPES:
        raise SemanticError(f"unregistered literal_type: {literal_type!r}")
    if literal_type == "text":
        if not isinstance(value, str):
            raise SemanticError("text literal value must be a string")
        if value != _nfc(value):
            raise SemanticError("text literal value must be NFC-normalized")
        if "\r" in value or "\x00" in value:
            raise SemanticError("text literal value must not contain CR or NUL")
    elif literal_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise SemanticError("integer literal value must be an arbitrary-precision int")
    elif literal_type == "boolean":
        if not isinstance(value, bool):
            raise SemanticError("boolean literal value must be true/false")
    elif literal_type == "timestamp":
        _validate_timestamp_instant(value)
    elif literal_type == "sha256":
        if not isinstance(value, str) or not SHA256_RE.match(value):
            raise SemanticError(f"sha256 literal value malformed: {value!r}")
    elif literal_type == "work_item_state":
        if value not in WORK_ITEM_STATES:
            raise SemanticError(f"unregistered work_item_state: {value!r}")
    elif literal_type == "pull_request_state":
        if value not in PULL_REQUEST_STATES:
            raise SemanticError(f"unregistered pull_request_state: {value!r}")
    elif literal_type == "review_verdict":
        if value not in REVIEW_VERDICTS:
            raise SemanticError(f"unregistered review_verdict: {value!r}")


# ---------------------------------------------------------------------------
# Predicate signatures (spec: "Predicate signatures")
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PredicateSignature:
    predicate: str
    subject_types: frozenset
    object_types: frozenset | None
    literal_allowed: bool = False
    same_type_as_subject: bool = False


_DEPENDS_ON_TYPES = frozenset(
    {"WorkItem", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}
)
_DERIVED_FROM_TYPES = frozenset(
    {"Decision", "Requirement", "Implementation", "Verification", "Claim", "Artifact", "Source"}
)
_CONCERNS_TYPES = frozenset(
    {
        "WorkItem",
        "Decision",
        "Specification",
        "Requirement",
        "Implementation",
        "Verification",
        "Claim",
        "Artifact",
    }
)
_DOCUMENTED_IN_ENTITY_SUBJECTS = frozenset(
    {
        "WorkItem",
        "Decision",
        "Specification",
        "Requirement",
        "Implementation",
        "Verification",
        "Claim",
        "Artifact",
    }
)

PREDICATE_SIGNATURES: tuple[PredicateSignature, ...] = (
    PredicateSignature(
        "governs",
        frozenset({"Decision", "Specification", "WorkItem"}),
        frozenset({"WorkItem", "Specification", "Requirement"}),
    ),
    PredicateSignature(
        "requires",
        frozenset({"WorkItem", "Decision", "Specification", "Requirement"}),
        frozenset({"Requirement", "Implementation", "Verification"}),
    ),
    PredicateSignature(
        "implements",
        frozenset({"Implementation", "Artifact"}),
        frozenset({"Requirement", "Specification"}),
    ),
    PredicateSignature(
        "verifies",
        frozenset({"Verification", "Artifact"}),
        frozenset({"Claim", "Requirement", "Implementation"}),
    ),
    PredicateSignature(
        "supports",
        frozenset({"Claim", "Verification", "Artifact", "Source"}),
        frozenset({"Claim", "Decision", "Requirement"}),
    ),
    PredicateSignature(
        "contradicts",
        frozenset({"Claim", "Verification", "Artifact", "Source"}),
        frozenset({"Claim", "Decision", "Requirement"}),
    ),
    PredicateSignature("supersedes", frozenset(ENTITY_TYPES), None, same_type_as_subject=True),
    PredicateSignature("depends_on", _DEPENDS_ON_TYPES, _DEPENDS_ON_TYPES),
    PredicateSignature("derived_from", _DERIVED_FROM_TYPES, _DERIVED_FROM_TYPES),
    PredicateSignature("concerns", _CONCERNS_TYPES, _CONCERNS_TYPES),
    PredicateSignature(
        "documented_in", _DOCUMENTED_IN_ENTITY_SUBJECTS, frozenset({"Source"})
    ),
    PredicateSignature("documented_in", frozenset({"Claim"}), None, literal_allowed=True),
)

PREDICATES = frozenset(sig.predicate for sig in PREDICATE_SIGNATURES)


def validate_predicate_signature(
    predicate: str,
    subject_type: str,
    object_type: str | None,
    has_literal_ref: bool,
) -> PredicateSignature:
    """Validate one (subject_type, predicate, object_type|literal) triple.

    Returns the matched :class:`PredicateSignature` row, or raises
    :class:`SemanticError`.
    """

    candidates = [sig for sig in PREDICATE_SIGNATURES if sig.predicate == predicate]
    if not candidates:
        raise SemanticError(f"unregistered predicate: {predicate!r}")
    for sig in candidates:
        if subject_type not in sig.subject_types:
            continue
        if sig.literal_allowed:
            if has_literal_ref and object_type is None:
                return sig
            continue
        if has_literal_ref:
            continue
        if sig.same_type_as_subject:
            if object_type == subject_type:
                return sig
            continue
        if object_type is not None and sig.object_types is not None and object_type in sig.object_types:
            return sig
    target = "literal" if has_literal_ref else object_type
    raise SemanticError(
        f"predicate signature violation: {subject_type} {predicate} {target!r}"
    )


# ---------------------------------------------------------------------------
# Claim key registry (spec: "Entity and Claim content")
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClaimKeyRegistryEntry:
    claim_key: str
    literal_type: str
    semantic_slot: str


def validate_claim_key_registry(
    entries: Sequence[ClaimKeyRegistryEntry],
) -> dict[str, ClaimKeyRegistryEntry]:
    registry: dict[str, ClaimKeyRegistryEntry] = {}
    previous_frame: bytes | None = None
    for entry in entries:
        if not isinstance(entry.claim_key, str) or not entry.claim_key:
            raise SemanticError("claim_key must be a non-empty string")
        if not isinstance(entry.semantic_slot, str) or not entry.semantic_slot:
            raise SemanticError(
                f"claim_key_registry entry {entry.claim_key!r} has an empty semantic_slot"
            )
        if entry.claim_key != _nfc(entry.claim_key) or entry.semantic_slot != _nfc(entry.semantic_slot):
            raise SemanticError("claim_key and semantic_slot must be NFC-normalized")
        if entry.literal_type not in LITERAL_TYPES:
            raise SemanticError(
                f"claim_key_registry entry {entry.claim_key!r} has unregistered literal_type"
            )
        if entry.claim_key in registry:
            raise SemanticError(f"duplicate claim_key in registry: {entry.claim_key!r}")
        current_frame = entry.claim_key.encode("utf-8")
        if previous_frame is not None and current_frame < previous_frame:
            raise SemanticError("claim_key_registry must be sorted by claim_key")
        previous_frame = current_frame
        registry[entry.claim_key] = entry
    return registry


# ---------------------------------------------------------------------------
# Canonical records: Entity, Literal, SourceRevision, Assertion
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    namespace: str
    canonical_source_identity: Any
    display_label_ref: str | None = None
    claim_key: str | None = None

    def validated(
        self, claim_key_registry: Mapping[str, ClaimKeyRegistryEntry] | None = None
    ) -> "Entity":
        if self.entity_type not in ENTITY_TYPES:
            raise SemanticError(f"unregistered entity_type: {self.entity_type!r}")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise SemanticError("entity namespace must be non-empty")
        expected_id = compute_entity_id(
            self.namespace, self.entity_type, self.canonical_source_identity
        )
        if expected_id != self.entity_id:
            raise SemanticError(
                f"entity_id {self.entity_id!r} does not match canonical source identity"
            )
        if self.entity_type == "Claim":
            if self.claim_key is None:
                raise SemanticError(f"Claim entity {self.entity_id} requires claim_key")
            if claim_key_registry is not None and self.claim_key not in claim_key_registry:
                raise SemanticError(f"unregistered claim_key: {self.claim_key!r}")
        elif self.claim_key is not None:
            raise SemanticError("claim_key is only defined on Claim entities")
        return self


@dataclasses.dataclass(frozen=True)
class Literal:
    literal_id: str
    literal_type: str
    value: Any

    def validated(self) -> "Literal":
        validate_literal_value(self.literal_type, self.value)
        expected_id = compute_literal_id(self.literal_type, self.value)
        if expected_id != self.literal_id:
            raise SemanticError(f"literal_id {self.literal_id!r} does not match canonical value")
        return self


@dataclasses.dataclass(frozen=True)
class SourceRevision:
    source_revision_id: str
    source_kind: str
    source_entity_id: str
    canonical_locator: str
    object_sha256: str
    object_byte_count: int
    source_event_at: str | None
    observed_at: str
    actor_id: str | None = None
    authority_class: str = "ordinary_record"
    authority_derivation_rule_id: str = "ordinary_record"

    def validated(self) -> "SourceRevision":
        if self.source_kind not in SOURCE_KINDS:
            raise SemanticError(f"unregistered source_kind: {self.source_kind!r}")
        if not isinstance(self.source_entity_id, str) or not self.source_entity_id:
            raise SemanticError("source_entity_id must be non-empty")
        if not isinstance(self.canonical_locator, str) or not self.canonical_locator:
            raise SemanticError("canonical_locator must be non-empty")
        if not SHA256_RE.match(self.object_sha256):
            raise SemanticError(f"malformed object_sha256: {self.object_sha256!r}")
        if isinstance(self.object_byte_count, bool) or not isinstance(self.object_byte_count, int) or self.object_byte_count < 0:
            raise SemanticError("object_byte_count must be nonnegative")
        if self.authority_class not in AUTHORITY_CLASSES:
            raise SemanticError(f"unregistered authority_class: {self.authority_class!r}")
        if self.authority_derivation_rule_id not in AUTHORITY_DERIVATION_RULE_IDS:
            raise SemanticError(
                f"unregistered authority_derivation_rule_id: {self.authority_derivation_rule_id!r}"
            )
        if self.source_event_at is not None:
            _validate_timestamp_instant(self.source_event_at)
        _validate_timestamp_instant(self.observed_at)
        expected_id = compute_source_revision_id(
            self.source_kind, self.canonical_locator, self.object_sha256, self.source_event_at
        )
        if expected_id != self.source_revision_id:
            raise SemanticError(
                f"source_revision_id {self.source_revision_id!r} does not match canonical fields"
            )
        return self


@dataclasses.dataclass(frozen=True)
class Assertion:
    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str | None
    literal_ref: str | None
    source_revision_id: str
    source_locator: str
    source_event_at: str | None
    asserted_at: str
    valid_from: str | None
    valid_to: str | None
    validity_basis: str
    authority_class: str
    authority_derivation_rule_id: str
    origin: str
    curation_state: str
    review_state: str
    epistemic_state: str
    dispute_state: str

    def semantic_fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _ASSERTION_ID_FIELDS}

    def validated(self) -> "Assertion":
        if not self.subject_id or not self.source_revision_id or not self.source_locator:
            raise SemanticError("assertion subject/source IDs and source_locator must be non-empty")
        if (self.object_id is None) == (self.literal_ref is None):
            raise SemanticError(
                f"assertion {self.assertion_id} requires exactly one of object_id/literal_ref"
            )
        if self.predicate not in PREDICATES:
            raise SemanticError(f"unregistered predicate: {self.predicate!r}")
        if self.validity_basis not in VALIDITY_BASES:
            raise SemanticError(f"invalid validity_basis: {self.validity_basis!r}")
        if self.authority_class not in AUTHORITY_CLASSES:
            raise SemanticError(f"invalid authority_class: {self.authority_class!r}")
        if self.authority_derivation_rule_id not in AUTHORITY_DERIVATION_RULE_IDS:
            raise SemanticError(
                f"unregistered authority_derivation_rule_id: {self.authority_derivation_rule_id!r}"
            )
        if self.origin not in ORIGINS:
            raise SemanticError(f"invalid origin: {self.origin!r}")
        if self.curation_state not in CURATION_STATES:
            raise SemanticError(f"invalid curation_state: {self.curation_state!r}")
        if self.review_state not in REVIEW_STATES:
            raise SemanticError(f"invalid review_state: {self.review_state!r}")
        if self.epistemic_state not in EPISTEMIC_STATES:
            raise SemanticError(f"invalid epistemic_state: {self.epistemic_state!r}")
        if self.dispute_state not in DISPUTE_STATES:
            raise SemanticError(f"invalid dispute_state: {self.dispute_state!r}")
        _validate_timestamp_instant(self.asserted_at)
        if self.source_event_at is not None:
            _validate_timestamp_instant(self.source_event_at)
        if self.valid_from is not None:
            _validate_timestamp_instant(self.valid_from)
        if self.valid_to is not None:
            _validate_timestamp_instant(self.valid_to)
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and not (self.valid_from < self.valid_to)
        ):
            raise SemanticError(
                f"assertion {self.assertion_id}: valid_from must precede valid_to"
            )
        expected_id = compute_assertion_id(self.semantic_fields())
        if expected_id != self.assertion_id:
            raise SemanticError(
                f"assertion_id {self.assertion_id!r} does not match canonical semantic fields "
                "(an ID collision with an unequal preimage must stop validation, not rehash)"
            )
        return self


def validate_asserted_at(
    assertion: Assertion,
    source_revision: SourceRevision,
    supporting_assertions: Sequence[Assertion] = (),
) -> None:
    """Validate deterministic assertion-revision provenance time.

    Source-direct facts use source event time when available, otherwise the
    captured observation time.  Curator interpretations require a captured
    action time (represented by ``source_event_at`` in the source revision),
    and rule-derived assertions use the maximum time of their explicit
    supporting assertions.  Build wall time is never accepted as a fallback.
    """

    expected: str | None
    if assertion.origin == "source_direct":
        expected = assertion.source_event_at or source_revision.source_event_at or source_revision.observed_at
    elif assertion.origin == "curator_interpretation":
        expected = assertion.source_event_at or source_revision.source_event_at
        if expected is None:
            raise SemanticError(
                f"curator_interpretation assertion {assertion.assertion_id} lacks captured action time"
            )
    elif assertion.origin == "rule_derived":
        if not supporting_assertions:
            raise SemanticError(
                f"rule_derived assertion {assertion.assertion_id} has no supporting assertions"
            )
        expected = max(a.asserted_at for a in supporting_assertions)
    else:  # Assertion.validated already rejects this, but keep fail-closed.
        raise SemanticError(f"unknown assertion origin: {assertion.origin!r}")
    if assertion.asserted_at != expected:
        raise SemanticError(
            f"asserted_at for {assertion.assertion_id} is not deterministic: "
            f"expected {expected!r}, got {assertion.asserted_at!r}"
        )


# ---------------------------------------------------------------------------
# Owner-decision authority derivation (spec: "Authority derivation")
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OwnerDecisionRegistryEntry:
    source_revision_id: str
    source_locator: str
    registry_entry_id: str


def validate_owner_decision_registry(
    entries: Sequence[OwnerDecisionRegistryEntry],
    known_source_locators: Sequence[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], OwnerDecisionRegistryEntry]:
    """Validate the immutable ``owner_decision_events.jsonl`` registry.

    Sorted by the framed ``(source_revision_id, source_locator)`` key;
    duplicate keys, duplicate ``registry_entry_id`` values, and (when
    ``known_source_locators`` is supplied) entries outside the source
    bundle all fail validation.
    """

    known = set(known_source_locators) if known_source_locators is not None else None
    registry: dict[tuple[str, str], OwnerDecisionRegistryEntry] = {}
    seen_entry_ids: set[str] = set()
    previous_key_frame: bytes | None = None
    for entry in entries:
        if not isinstance(entry.registry_entry_id, str) or not entry.registry_entry_id:
            raise SemanticError("owner decision registry_entry_id must be non-empty")
        key = (entry.source_revision_id, entry.source_locator)
        if key in registry:
            raise SemanticError(f"duplicate owner_decision_events entry: {key}")
        if entry.registry_entry_id in seen_entry_ids:
            raise SemanticError(
                f"duplicate registry_entry_id: {entry.registry_entry_id!r}"
            )
        seen_entry_ids.add(entry.registry_entry_id)
        key_frame = frame(list(key))
        if previous_key_frame is not None and key_frame < previous_key_frame:
            raise SemanticError(
                "owner_decision_events must be sorted by framed (source_revision_id, source_locator)"
            )
        previous_key_frame = key_frame
        if known is not None and key not in known:
            raise SemanticError(
                f"owner_decision_events entry references locator outside the source bundle: {key}"
            )
        registry[key] = entry
    return registry


@dataclasses.dataclass(frozen=True)
class SourceEvidence:
    """Structured, mechanically inspectable evidence about one captured
    source object -- sufficient to derive ``authority_class`` without
    parsing prose. A source-bundle compiler is responsible for populating
    this from captured actor/role/state evidence; this module only applies
    the frozen ordered derivation table to it.
    """

    source_kind: str
    source_revision_id: str
    source_locator: str
    actor_id: str | None = None
    actor_is_repository_owner: bool = False
    spec_status_approved: bool = False
    spec_approval_provenance_resolved: bool = False
    commit_reachable_from_base_branch: bool = False
    associated_pr_merged: bool = False
    verification_artifact_fingerprinted: bool = False
    external_reference_immutable_edition: bool = False
    agent_authored_without_approval: bool = False


_OWNER_COMMENT_KINDS = frozenset(
    {"github_issue_comment", "github_pull_request_comment", "github_review"}
)
_WORK_ITEM_KINDS = frozenset({"github_issue", "github_pull_request"})

AUTHORITY_DERIVATION_RULE_IDS = frozenset(
    {
        "owner_decision_registered_event",
        "approved_specification",
        "canonical_work_item",
        "merged_implementation",
        "verification_evidence",
        "external_reference",
        "agent_proposal",
        "ordinary_record",
        # Synthetic generator records carry their immutable generator rule
        # identifier while retaining authority_class=ordinary_record.
        "synthetic_fixture_v2.2",
    }
)


def derive_authority_class(
    evidence: SourceEvidence,
    owner_decision_registry: Mapping[tuple[str, str], OwnerDecisionRegistryEntry],
) -> tuple[str, str]:
    """Apply the ordered mechanical authority-derivation table.

    Returns ``(authority_class, authority_derivation_rule_id)``. Owner
    identity and decision-event registration are independent predicates:
    an owner comment is only ``owner_decision`` when both the actor is the
    repository owner *and* its exact
    ``(source_revision_id, source_locator)`` is a registered event: an
    unregistered owner acknowledgement/question/proposal, and the same
    registered text authored by a non-owner, both fall through to a lower
    rule (never invent authority from text similarity or registry error).
    """

    if evidence.source_kind not in SOURCE_KINDS:
        raise SemanticError(f"unregistered source_kind: {evidence.source_kind!r}")
    if not evidence.source_revision_id or not evidence.source_locator:
        raise SemanticError("authority evidence requires source revision and stable locator")
    key = (evidence.source_revision_id, evidence.source_locator)
    if (
        evidence.source_kind in _OWNER_COMMENT_KINDS
        and evidence.actor_is_repository_owner
        and key in owner_decision_registry
    ):
        return "owner_decision", "owner_decision_registered_event"
    if (
        evidence.source_kind == "repository_file"
        and evidence.spec_status_approved
        and evidence.spec_approval_provenance_resolved
    ):
        return "approved_specification", "approved_specification"
    if evidence.source_kind in _WORK_ITEM_KINDS:
        return "canonical_work_item", "canonical_work_item"
    if (
        evidence.source_kind == "git_commit"
        and evidence.commit_reachable_from_base_branch
        and evidence.associated_pr_merged
    ):
        return "merged_implementation", "merged_implementation"
    if evidence.source_kind == "verification_artifact" and evidence.verification_artifact_fingerprinted:
        return "verification_evidence", "verification_evidence"
    if (
        evidence.source_kind == "external_document"
        and evidence.external_reference_immutable_edition
    ):
        return "external_reference", "external_reference"
    if evidence.agent_authored_without_approval:
        return "agent_proposal", "agent_proposal"
    return "ordinary_record", "ordinary_record"


# ---------------------------------------------------------------------------
# State-machine transition validators (spec: "Frozen vocabularies")
# ---------------------------------------------------------------------------

ALLOWED_CURATION_TRANSITIONS: dict[str, frozenset] = {
    "unreviewed": frozenset({"curator_checked"}),
    "curator_checked": frozenset({"independently_reviewed"}),
    "independently_reviewed": frozenset(),
}
ALLOWED_REVIEW_TRANSITIONS: dict[str, frozenset] = {
    "not_required": frozenset(),
    "pending": frozenset({"passed", "changes_required", "rejected"}),
    "passed": frozenset(),
    "changes_required": frozenset({"pending"}),
    "rejected": frozenset(),
}
ALLOWED_EPISTEMIC_TRANSITIONS: dict[str, frozenset] = {
    "extracted": frozenset({"unresolved", "supported", "rejected"}),
    "unresolved": frozenset({"supported", "rejected"}),
    "supported": frozenset({"verified", "rejected"}),
    "verified": frozenset({"accepted", "rejected"}),
    "accepted": frozenset(),
    "rejected": frozenset(),
}
ALLOWED_DISPUTE_TRANSITIONS: dict[str, frozenset] = {
    "undisputed": frozenset({"disputed"}),
    "disputed": frozenset({"resolved_upheld", "resolved_rejected"}),
    "resolved_upheld": frozenset({"disputed"}),
    "resolved_rejected": frozenset({"disputed"}),
}


def validate_state_transition(table: Mapping[str, frozenset], old_state: str, new_state: str) -> None:
    """Reject any transition not on the frozen forward-only adjacency list.

    A "backward mutation" (in-place rewrite) is never expressed this way:
    callers only invoke this when comparing a predecessor assertion's
    frozen state to a genuinely new successor assertion's state.
    """

    allowed = table.get(old_state, frozenset())
    if new_state not in allowed:
        raise SemanticError(f"illegal transition {old_state!r} -> {new_state!r}")


def validate_review_state_transition(
    old_state: str,
    new_state: str,
    old_source_revision_id: str,
    new_source_revision_id: str,
) -> None:
    """``changes_required -> pending`` is legal only on a new source revision."""

    validate_state_transition(ALLOWED_REVIEW_TRANSITIONS, old_state, new_state)
    if old_state == "changes_required" and new_state == "pending":
        if old_source_revision_id == new_source_revision_id:
            raise SemanticError(
                "changes_required -> pending requires a new source revision"
            )


# ---------------------------------------------------------------------------
# Shared evaluator primitives: admissibility, temporal inclusion,
# supersession, dependency staleness (spec: "Frozen vocabularies",
# "Temporal model", "Shared semantic evaluator and RetrievalBundle v1")
# ---------------------------------------------------------------------------


def is_admissible(assertion: Assertion) -> bool:
    """Only curator_checked/independently_reviewed assertions whose
    epistemic state is supported/verified/accepted can affect semantic
    disposition; a pending/changes_required/rejected review_state also
    excludes an assertion.

    Assumption (not fully pinned down by the frozen contract): "review is
    required by the governing source" is read conservatively from the
    assertion's own ``review_state`` -- ``not_required`` or ``passed`` only
    -- since no other field is available to this backend-independent
    layer.
    """

    if assertion.curation_state not in {"curator_checked", "independently_reviewed"}:
        return False
    if assertion.epistemic_state not in {"supported", "verified", "accepted"}:
        return False
    if assertion.review_state not in {"not_required", "passed"}:
        return False
    # Disputed/rejected evidence remains inspectable, but cannot change the
    # semantic disposition until an admissible resolution explicitly upholds
    # it.  This is the fail-closed dispute rule in the frozen evaluator.
    if assertion.dispute_state not in {"undisputed", "resolved_upheld"}:
        return False
    return True


def temporal_domain_state(assertion: Assertion) -> str:
    """``'known'`` when ``valid_from`` is present, else ``'unknown'``.

    An assertion with unknown domain start is returned as provenance but
    cannot satisfy an as-of query requiring known state.
    """

    return "unknown" if assertion.valid_from is None else "known"


def as_of_holds(assertion: Assertion, as_of: str | None) -> bool:
    """Half-open ``[valid_from, valid_to)`` as-of inclusion test.

    ``as_of=None`` means the current structural query, but an unknown domain
    start still fails closed; timestamps compare lexicographically, which is
    valid because the canonical ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` form is
    order-preserving.
    """

    if as_of is None:
        # An unconstrained/current query still cannot promote an assertion
        # whose domain start is unknown to a known current fact.
        return assertion.valid_from is not None
    _validate_timestamp_instant(as_of)
    if assertion.valid_from is None:
        return False
    if assertion.valid_from > as_of:
        return False
    if assertion.valid_to is not None and not (as_of < assertion.valid_to):
        return False
    return True


def effective_supersessor(
    supersedes_targeting_entity: Sequence[Assertion],
    as_of: str | None,
) -> Assertion | str | None:
    """Return the admissible ``supersedes`` assertion in effect at ``as_of``.

    Supersession becomes effective at the admissible assertion's
    ``valid_from``; before that instant the predecessor remains current.
    Returns ``None`` when no admissible successor is yet effective, an
    :class:`Assertion` when exactly one is, or the string ``"ambiguous"``
    when two or more admissible successors tie at the same effective
    instant (equal-time competing successors fail closed).

    Callers are expected to have already filtered to admissible assertions
    targeting one entity via ``supersedes``.
    """

    effective = [
        a
        for a in supersedes_targeting_entity
        if is_admissible(a) and a.valid_from is not None and as_of_holds(a, as_of)
    ]
    if not effective:
        return None
    effective.sort(key=lambda a: (a.valid_from or "", a.assertion_id))
    earliest = effective[0].valid_from
    tied = [a for a in effective if a.valid_from == earliest]
    if len(tied) > 1:
        return "ambiguous"
    return tied[0]


# Keep the historical misspelling as the implementation name, but expose the
# two names commonly used by callers.  Both aliases deliberately preserve the
# same fail-closed equal-time behaviour.
effective_supersession = effective_supersessor
effective_successor = effective_supersessor


def dependency_stale_entities(
    admissible_depends_on_by_subject: Mapping[str, Sequence[str]],
    superseded_entities: Sequence[str],
) -> set[str]:
    """Fixed-point required-dependency staleness over explicit ``depends_on``
    assertions only.

    An entity is dependency-stale when any admissible ``depends_on`` target
    is superseded or itself dependency-stale (transitively).
    """

    superseded = set(superseded_entities)
    stale: set[str] = set()
    changed = True
    while changed:
        changed = False
        for subject, targets in admissible_depends_on_by_subject.items():
            if subject in stale:
                continue
            for target in targets:
                if target in superseded or target in stale:
                    stale.add(subject)
                    changed = True
                    break
    return stale


# ---------------------------------------------------------------------------
# RetrievalBundle v1 canonical schema and structural validator
# (spec: "Shared semantic evaluator and RetrievalBundle v1")
# ---------------------------------------------------------------------------

RETRIEVAL_BUNDLE_CONTRACT_VERSION = "cyax-retrieval-bundle-1.0"
PATH_DIRECTIONS = frozenset({"forward", "reverse"})


@dataclasses.dataclass(frozen=True)
class PathStep:
    assertion_id: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in PATH_DIRECTIONS:
            raise SemanticError(f"invalid path step direction: {self.direction!r}")


@dataclasses.dataclass(frozen=True)
class Path:
    path_id: str
    steps: tuple[PathStep, ...]


def compute_path_id(steps: Sequence[PathStep]) -> str:
    return frame_hash([[step.assertion_id, step.direction] for step in steps])


@dataclasses.dataclass(frozen=True)
class RetrievalBundle:
    contract_version: str
    query_instance_id: str
    snapshot_id: str
    as_of: str | None
    entities: tuple[Entity, ...]
    literals: tuple[Literal, ...]
    assertions: tuple[Assertion, ...]
    source_revisions: tuple[SourceRevision, ...]
    paths: tuple[Path, ...]
    diagnostics: Mapping[str, Any] | None = None


def _sorted_unique(records: Sequence[Any], id_attr: str, label: str) -> None:
    ids = [getattr(record, id_attr) for record in records]
    if ids != sorted(ids):
        raise SemanticError(f"RetrievalBundle {label} must be sorted by {id_attr}")
    if len(ids) != len(set(ids)):
        raise SemanticError(f"RetrievalBundle {label} contains a duplicate {id_attr}")


def validate_retrieval_bundle(bundle: RetrievalBundle) -> RetrievalBundle:
    """Validate the RetrievalBundle v1 structural contract.

    Checks: contract version; every semantic array present (never null)
    and sorted/unique by primary ID; deterministic, duplicate-free path
    ordering and recomputed ``path_id``; complete reference closure of
    every ``literal_ref``/``display_label_ref`` against ``literals[]``
    with no extraneous literal; and that every entity/assertion/
    source_revision referenced by a returned assertion or path step is
    itself present exactly once.
    """

    if bundle.contract_version != RETRIEVAL_BUNDLE_CONTRACT_VERSION:
        raise SemanticError(f"unsupported contract_version: {bundle.contract_version!r}")
    if not isinstance(bundle.query_instance_id, str) or not bundle.query_instance_id:
        raise SemanticError("query_instance_id must be non-empty")
    if not isinstance(bundle.snapshot_id, str) or not bundle.snapshot_id:
        raise SemanticError("snapshot_id must be non-empty")
    if bundle.as_of is not None:
        _validate_timestamp_instant(bundle.as_of)

    _sorted_unique(bundle.entities, "entity_id", "entities[]")
    _sorted_unique(bundle.literals, "literal_id", "literals[]")
    _sorted_unique(bundle.assertions, "assertion_id", "assertions[]")
    _sorted_unique(bundle.source_revisions, "source_revision_id", "source_revisions[]")

    entity_ids = {e.entity_id for e in bundle.entities}
    assertion_ids = {a.assertion_id for a in bundle.assertions}
    source_revision_ids = {s.source_revision_id for s in bundle.source_revisions}
    literal_ids = {l.literal_id for l in bundle.literals}

    for entity in bundle.entities:
        entity.validated()

    for assertion in bundle.assertions:
        assertion.validated()
        if assertion.subject_id not in entity_ids:
            raise SemanticError(f"assertion {assertion.assertion_id} subject not in entities[]")
        if assertion.object_id is not None and assertion.object_id not in entity_ids:
            raise SemanticError(f"assertion {assertion.assertion_id} object not in entities[]")
        object_type = (
            bundle.entities[next(i for i, item in enumerate(bundle.entities) if item.entity_id == assertion.object_id)].entity_type
            if assertion.object_id is not None
            else None
        )
        validate_predicate_signature(
            assertion.predicate,
            next(item.entity_type for item in bundle.entities if item.entity_id == assertion.subject_id),
            object_type,
            assertion.literal_ref is not None,
        )
        if assertion.source_revision_id not in source_revision_ids:
            raise SemanticError(
                f"assertion {assertion.assertion_id} source_revision not in source_revisions[]"
            )

    required_literal_ids: set[str] = set()
    for assertion in bundle.assertions:
        if assertion.literal_ref is not None:
            if assertion.literal_ref not in literal_ids:
                raise SemanticError(
                    f"assertion {assertion.assertion_id} literal_ref not in literals[]"
                )
            required_literal_ids.add(assertion.literal_ref)
    for entity in bundle.entities:
        if entity.display_label_ref is not None:
            if entity.display_label_ref not in literal_ids:
                raise SemanticError(
                    f"entity {entity.entity_id} display_label_ref not in literals[]"
                )
            required_literal_ids.add(entity.display_label_ref)
    extraneous = literal_ids - required_literal_ids
    if extraneous:
        raise SemanticError(f"literals[] contains unreferenced literal(s): {sorted(extraneous)}")

    for literal in bundle.literals:
        literal.validated()
    for revision in bundle.source_revisions:
        revision.validated()
        if revision.source_entity_id not in entity_ids:
            raise SemanticError(
                f"source revision {revision.source_revision_id} source_entity_id "
                "is not present in entities[]"
            )
    referenced_revision_ids = {a.source_revision_id for a in bundle.assertions}
    extra_revisions = source_revision_ids - referenced_revision_ids
    if extra_revisions:
        raise SemanticError(
            f"source_revisions[] contains unreferenced source revision(s): {sorted(extra_revisions)}"
        )

    seen_step_sequences: set[tuple[tuple[str, str], ...]] = set()
    previous_path_key: bytes | None = None
    for path in bundle.paths:
        current: str | None = None
        for step in path.steps:
            if step.direction not in PATH_DIRECTIONS:
                raise SemanticError(f"invalid path step direction: {step.direction!r}")
            if step.assertion_id not in assertion_ids:
                raise SemanticError(
                    f"path {path.path_id} step references assertion not in assertions[]"
                )
            assertion = next(item for item in bundle.assertions if item.assertion_id == step.assertion_id)
            if assertion.object_id is None:
                raise SemanticError("literal assertions cannot occur in a path witness")
            if current is not None:
                endpoint = assertion.subject_id if step.direction == "forward" else assertion.object_id
                if endpoint != current:
                    raise SemanticError(
                        f"path {path.path_id} has disconnected traversal at {step.assertion_id}"
                    )
            current = assertion.object_id if step.direction == "forward" else assertion.subject_id
        step_sequence = tuple((s.assertion_id, s.direction) for s in path.steps)
        if step_sequence in seen_step_sequences:
            raise SemanticError("RetrievalBundle paths contain a duplicate step sequence")
        seen_step_sequences.add(step_sequence)
        expected_path_id = compute_path_id(path.steps)
        if expected_path_id != path.path_id:
            raise SemanticError(f"path_id {path.path_id!r} does not match its step sequence")
        path_key = frame([[s.assertion_id, s.direction] for s in path.steps])
        if previous_path_key is not None and path_key < previous_path_key:
            raise SemanticError("RetrievalBundle paths[] must be ordered by framed step sequence")
        previous_path_key = path_key

    if bundle.diagnostics is not None and not isinstance(bundle.diagnostics, dict):
        raise SemanticError("diagnostics must be an object or null")

    return bundle


def strip_diagnostics(bundle: RetrievalBundle) -> RetrievalBundle:
    """Diagnostics-stripped, display-label-stripped projection for semantic
    gold/S/G parity comparison.

    Removes ``diagnostics`` and every ``display_label_ref``/display-only
    literal; assertion-``literal_ref`` closure remains mandatory.
    """

    stripped_entities = tuple(
        dataclasses.replace(e, display_label_ref=None) for e in bundle.entities
    )
    required_literal_ids = {a.literal_ref for a in bundle.assertions if a.literal_ref is not None}
    stripped_literals = tuple(
        l for l in bundle.literals if l.literal_id in required_literal_ids
    )
    return dataclasses.replace(
        bundle,
        entities=stripped_entities,
        literals=stripped_literals,
        diagnostics=None,
    )
