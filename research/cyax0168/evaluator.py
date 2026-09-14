#!/usr/bin/env python3
"""CYAX-0168 reference semantic evaluator, Q01-Q12 query/gold contract.

This module implements the backend-independent semantic evaluator described
in ``specs/0168-structured-graph-materialization/spec.md`` ("Shared semantic
evaluator and RetrievalBundle v1", "Temporal model", and "Frozen query
instances and gold"). It operates only on an in-memory :class:`Snapshot` of
canonical records; it never touches SQLite or a graph backend. Two adapters
(S and G, implemented elsewhere) are expected to reproduce the same
:class:`RetrievalBundle` this evaluator produces from the same snapshot.

Per the CYAX-0168 G1 hard guard this module performs no T0-T4 materialization
or backend access and never instantiates decision fixtures; it is exercised
only against tiny synthetic snapshots built directly in ``test_evaluator.py``.

The canonical encoding (`_frame*`) mirrors "Canonical encoding and stable
IDs" in the spec closely enough to produce deterministic, order-independent
`path_id` values; it is a self-contained copy rather than a shared import
because this worker's file scope excludes a shared canonical module.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

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

PREDICATES = frozenset(
    {
        "governs",
        "requires",
        "implements",
        "verifies",
        "supports",
        "contradicts",
        "supersedes",
        "depends_on",
        "derived_from",
        "concerns",
        "documented_in",
    }
)

# Predicate signature table: predicate -> (allowed subject types, allowed
# object types, literal allowed). Spec: "Predicate signatures".
PREDICATE_SIGNATURES: Mapping[str, tuple[frozenset[str], frozenset[str], bool]] = {
    "governs": (
        frozenset({"Decision", "Specification", "WorkItem"}),
        frozenset({"WorkItem", "Specification", "Requirement"}),
        False,
    ),
    "requires": (
        frozenset({"WorkItem", "Decision", "Specification", "Requirement"}),
        frozenset({"Requirement", "Implementation", "Verification"}),
        False,
    ),
    "implements": (
        frozenset({"Implementation", "Artifact"}),
        frozenset({"Requirement", "Specification"}),
        False,
    ),
    "verifies": (
        frozenset({"Verification", "Artifact"}),
        frozenset({"Claim", "Requirement", "Implementation"}),
        False,
    ),
    "supports": (
        frozenset({"Claim", "Verification", "Artifact", "Source"}),
        frozenset({"Claim", "Decision", "Requirement"}),
        False,
    ),
    "contradicts": (
        frozenset({"Claim", "Verification", "Artifact", "Source"}),
        frozenset({"Claim", "Decision", "Requirement"}),
        False,
    ),
    "supersedes": (frozenset(ENTITY_TYPES), frozenset(ENTITY_TYPES), False),
    "depends_on": (
        frozenset({"WorkItem", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}),
        frozenset({"WorkItem", "Requirement", "Implementation", "Verification", "Claim", "Artifact"}),
        False,
    ),
    "derived_from": (
        frozenset(
            {"Decision", "Requirement", "Implementation", "Verification", "Claim", "Artifact", "Source"}
        ),
        frozenset(
            {"Decision", "Requirement", "Implementation", "Verification", "Claim", "Artifact", "Source"}
        ),
        False,
    ),
    "concerns": (
        frozenset(
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
        ),
        frozenset(
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
        ),
        False,
    ),
    # documented_in has two signature rows in the spec: entity->Source (no
    # literal) and Claim->literal (yes literal, no object). Both share the
    # predicate name; callers disambiguate on which of object_id/literal_ref
    # is populated.
    "documented_in": (
        frozenset(
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
        ),
        frozenset({"Source"}),
        True,
    ),
}

# Q02's query-local authority rank (spec: "Semantic answers", Q02 note).
Q02_AUTHORITY_RANK: Mapping[str, int] = {
    "owner_decision": 0,
    "approved_specification": 1,
    "canonical_work_item": 2,
    "merged_implementation": 3,
    "verification_evidence": 4,
    "external_reference": 5,
    "ordinary_record": 6,
    "agent_proposal": 7,
}

Q11_ELIGIBLE_CLAIM_KEY = "synthetic.block_claim"
Q11_ELIGIBLE_SLOT = "synthetic_fixture_block_statement"

# Q11 rank order: accepted, then verified, then supported (lower is better).
Q11_EPISTEMIC_RANK: Mapping[str, int] = {"accepted": 0, "verified": 1, "supported": 2}

# Frozen workload metadata.  These are selectors only; they do not name or
# load any decision fixture.
QUERY_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "Q01": {"percentile": 0.50, "as_of": "2000-01-03T00:00:00.000000Z", "depth": None, "metric": "count"},
    "Q02": {"percentile": 0.90, "as_of": "2000-01-03T00:00:00.000000Z", "depth": None, "metric": "count"},
    "Q03": {"percentile": None, "as_of": None, "depth": "path_depth", "metric": "shortest_distance"},
    "Q04": {"percentile": 0.50, "as_of": "selected.valid_from", "depth": None, "metric": "valid_from"},
    "Q05": {"percentile": 0.50, "as_of": None, "depth": 1, "metric": "degree"},
    "Q06": {"percentile": 0.90, "as_of": None, "depth": "path_depth", "metric": "count"},
    "Q07": {"percentile": 0.90, "as_of": "2000-01-03T00:00:00.000000Z", "depth": "path_depth", "metric": "count"},
    "Q08": {"percentile": 0.50, "as_of": "2000-01-03T00:00:00.000000Z", "depth": None, "metric": "dispute_count"},
    "Q09": {"percentile": None, "as_of": None, "depth": "path_depth", "metric": "shortest_distance"},
    "Q10": {"percentile": 0.90, "as_of": None, "depth": "path_depth", "metric": "evidence_cover"},
    "Q11": {"percentile": 0.50, "as_of": "2000-01-03T00:00:00.000000Z", "depth": None, "metric": "count"},
    "Q12": {"percentile": 0.50, "as_of": "2000-01-03T00:00:00.000000Z", "depth": None, "metric": "combined_count"},
}


class EvaluatorError(ValueError):
    """Raised for malformed records or contract violations."""


@dataclass(frozen=True)
class ClaimKeyRegistryEntry:
    claim_key: str
    literal_type: str
    semantic_slot: str


def validate_claim_key_registry(entries: Sequence[ClaimKeyRegistryEntry]) -> dict[str, ClaimKeyRegistryEntry]:
    """Validate the fixture-local Claim registry before semantic evaluation."""
    ordered = tuple(entries)
    keys = [entry.claim_key for entry in ordered]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise EvaluatorError("claim_key_registry must be sorted and duplicate-free")
    result: dict[str, ClaimKeyRegistryEntry] = {}
    for entry in ordered:
        if not entry.claim_key or entry.literal_type not in LITERAL_TYPES or not entry.semantic_slot:
            raise EvaluatorError("claim registry entries require key, literal_type, and semantic_slot")
        result[entry.claim_key] = entry
    return result


# ---------------------------------------------------------------------------
# Canonical framing (spec: "Canonical encoding and stable IDs")
# ---------------------------------------------------------------------------


def _frame_null() -> bytes:
    return b"N" + (0).to_bytes(8, "big")


def _frame_bytes(value: bytes) -> bytes:
    return b"X" + len(value).to_bytes(8, "big") + value


def _frame_str(value: str) -> bytes:
    data = unicodedata.normalize("NFC", value).encode("utf-8")
    return b"S" + len(data).to_bytes(8, "big") + data


def _frame_int(value: int) -> bytes:
    data = str(value).encode("ascii")
    return b"I" + len(data).to_bytes(8, "big") + data


def _frame_bool(value: bool) -> bytes:
    return b"B" + (1).to_bytes(8, "big") + (b"\x01" if value else b"\x00")


def _frame_array(items: Sequence[Any]) -> bytes:
    out = bytearray(b"A" + len(items).to_bytes(8, "big"))
    for item in items:
        out += frame(item)
    return bytes(out)


def _frame_object(obj: Mapping[str, Any]) -> bytes:
    pairs = sorted(obj.items(), key=lambda kv: unicodedata.normalize("NFC", kv[0]).encode("utf-8"))
    out = bytearray(b"O" + len(pairs).to_bytes(8, "big"))
    for key, value in pairs:
        out += _frame_str(key) + frame(value)
    return bytes(out)


def frame(value: Any) -> bytes:
    """Canonical domain-separated frame for ``value`` (spec canonical encoding)."""

    if value is None:
        return _frame_null()
    if isinstance(value, bool):
        return _frame_bool(value)
    if isinstance(value, int):
        return _frame_int(value)
    if isinstance(value, str):
        return _frame_str(value)
    if isinstance(value, bytes):
        return _frame_bytes(value)
    if isinstance(value, (list, tuple)):
        return _frame_array(value)
    if isinstance(value, dict):
        return _frame_object(value)
    raise EvaluatorError(f"unsupported type for canonical frame: {type(value)!r}")


def frame_hash(value: Any) -> str:
    return hashlib.sha256(frame(value)).hexdigest()


# ---------------------------------------------------------------------------
# Canonical record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    display_label_ref: str | None = None
    # Spec: "A Claim additionally has claim_key, a registered fixture-local
    # semantic key... This field is absent on every other entity type."
    claim_key: str | None = None

    def __post_init__(self) -> None:
        if self.entity_type not in ENTITY_TYPES:
            raise EvaluatorError(f"unknown entity_type {self.entity_type!r}")
        if self.entity_type == "Claim" and self.claim_key is None:
            raise EvaluatorError(f"Claim entity {self.entity_id} requires claim_key")
        if self.entity_type != "Claim" and self.claim_key is not None:
            raise EvaluatorError(f"non-Claim entity {self.entity_id} must not have claim_key")


@dataclass(frozen=True)
class Literal:
    literal_id: str
    literal_type: str
    value: Any

    def __post_init__(self) -> None:
        if self.literal_type not in LITERAL_TYPES:
            raise EvaluatorError(f"unknown literal_type {self.literal_type!r}")
        value = self.value
        if self.literal_type == "text":
            if not isinstance(value, str) or value != unicodedata.normalize("NFC", value) or "\r" in value or "\x00" in value:
                raise EvaluatorError("text literal must be NFC and contain neither CR nor NUL")
        elif self.literal_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise EvaluatorError("integer literal must be an int")
        elif self.literal_type == "boolean":
            if not isinstance(value, bool):
                raise EvaluatorError("boolean literal must be bool")
        elif self.literal_type == "timestamp":
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value):
                raise EvaluatorError("timestamp literal has non-canonical format")
            try:
                _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError as exc:
                raise EvaluatorError("timestamp literal is not a valid instant") from exc
        elif self.literal_type == "sha256":
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise EvaluatorError("sha256 literal must be lowercase hexadecimal")
        elif self.literal_type == "work_item_state" and value not in {"open", "closed"}:
            raise EvaluatorError("unknown work_item_state literal")
        elif self.literal_type == "pull_request_state" and value not in {"draft_open", "open", "closed_unmerged", "merged"}:
            raise EvaluatorError("unknown pull_request_state literal")
        elif self.literal_type == "review_verdict" and value not in {"pending", "pass", "changes_required", "rejected"}:
            raise EvaluatorError("unknown review_verdict literal")


@dataclass(frozen=True)
class SourceRevision:
    source_revision_id: str
    source_kind: str
    canonical_locator: str
    object_sha256: str
    source_event_at: str | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in SOURCE_KINDS:
            raise EvaluatorError(f"unknown source_kind {self.source_kind!r}")
        if not isinstance(self.object_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.object_sha256):
            raise EvaluatorError("source revision object_sha256 must be lowercase hexadecimal")
        for name, value in (("source_event_at", self.source_event_at),):
            if value is not None and not _timestamp_is_valid(value):
                raise EvaluatorError(f"{name} must be canonical UTC timestamp")


@dataclass(frozen=True)
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

    def __post_init__(self) -> None:
        if self.predicate not in PREDICATES:
            raise EvaluatorError(f"unknown predicate {self.predicate!r}")
        if (self.object_id is None) == (self.literal_ref is None):
            raise EvaluatorError(
                f"assertion {self.assertion_id}: exactly one of object_id/literal_ref is required"
            )
        if self.validity_basis not in {"explicit", "source_event", "unknown"}:
            raise EvaluatorError(f"unknown validity_basis {self.validity_basis!r}")
        if self.authority_class not in AUTHORITY_CLASSES:
            raise EvaluatorError(f"unknown authority_class {self.authority_class!r}")
        if self.origin not in ORIGINS:
            raise EvaluatorError(f"unknown origin {self.origin!r}")
        if self.curation_state not in CURATION_STATES:
            raise EvaluatorError(f"unknown curation_state {self.curation_state!r}")
        if self.review_state not in REVIEW_STATES:
            raise EvaluatorError(f"unknown review_state {self.review_state!r}")
        if self.epistemic_state not in EPISTEMIC_STATES:
            raise EvaluatorError(f"unknown epistemic_state {self.epistemic_state!r}")
        if self.dispute_state not in DISPUTE_STATES:
            raise EvaluatorError(f"unknown dispute_state {self.dispute_state!r}")
        for name, value in (("source_event_at", self.source_event_at), ("asserted_at", self.asserted_at), ("valid_from", self.valid_from), ("valid_to", self.valid_to)):
            if value is not None and not _timestamp_is_valid(value):
                raise EvaluatorError(f"{name} must be canonical UTC timestamp")


def _timestamp_is_valid(value: str) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value):
        return False
    try:
        _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class Snapshot:
    """An immutable in-memory assertion snapshot (spec: "Immutable assertion
    snapshot and identity"). ``snapshot_id`` is caller-supplied; this module
    does not itself compile snapshots from a source bundle."""

    snapshot_id: str
    entities: Mapping[str, Entity]
    literals: Mapping[str, Literal]
    source_revisions: Mapping[str, SourceRevision]
    assertions: Mapping[str, Assertion]

    # Adjacency caches, built lazily via `index()`.
    _by_subject: Mapping[str, tuple[Assertion, ...]] = field(default=None, compare=False, repr=False)
    _by_object: Mapping[str, tuple[Assertion, ...]] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the closed in-memory snapshot boundary.

        Backend adapters are deliberately not trusted to supply a partially
        typed projection.  This inexpensive validation is the reference
        evaluator's equivalent of the canonical snapshot validator: all
        references must resolve, predicates must have their frozen signature,
        and the Claim/literal rule is enforced before a query can run.
        """
        for key, entity in self.entities.items():
            if key != entity.entity_id:
                raise EvaluatorError("entity mapping key does not match entity_id")
        for key, literal in self.literals.items():
            if key != literal.literal_id:
                raise EvaluatorError("literal mapping key does not match literal_id")
        for key, revision in self.source_revisions.items():
            if key != revision.source_revision_id:
                raise EvaluatorError("source revision mapping key does not match source_revision_id")
        for key, assertion in self.assertions.items():
            if key != assertion.assertion_id:
                raise EvaluatorError("assertion mapping key does not match assertion_id")
            if assertion.subject_id not in self.entities:
                raise EvaluatorError(f"dangling assertion subject {assertion.subject_id!r}")
            if assertion.object_id is not None and assertion.object_id not in self.entities:
                raise EvaluatorError(f"dangling assertion object {assertion.object_id!r}")
            if assertion.literal_ref is not None and assertion.literal_ref not in self.literals:
                raise EvaluatorError(f"dangling assertion literal {assertion.literal_ref!r}")
            if assertion.source_revision_id not in self.source_revisions:
                raise EvaluatorError(f"dangling source revision {assertion.source_revision_id!r}")
            subject = self.entities[assertion.subject_id]
            signature = PREDICATE_SIGNATURES[assertion.predicate]
            if subject.entity_type not in signature[0]:
                raise EvaluatorError(
                    f"predicate {assertion.predicate!r} disallows subject type {subject.entity_type!r}"
                )
            if assertion.literal_ref is not None:
                if assertion.predicate != "documented_in" or subject.entity_type != "Claim":
                    raise EvaluatorError("only a Claim documented_in assertion may carry literal_ref")
                if not signature[2]:
                    raise EvaluatorError("predicate does not permit literal_ref")
            else:
                object_type = self.entities[assertion.object_id].entity_type
                if object_type not in signature[1]:
                    raise EvaluatorError(
                        f"predicate {assertion.predicate!r} disallows object type {object_type!r}"
                    )
                if assertion.predicate == "documented_in" and subject.entity_type == "Claim":
                    raise EvaluatorError("Claim documented_in content must use literal_ref")

    def indexed(self) -> "Snapshot":
        if self._by_subject is not None and self._by_object is not None:
            return self
        by_subject: dict[str, list[Assertion]] = {}
        by_object: dict[str, list[Assertion]] = {}
        for assertion in self.assertions.values():
            by_subject.setdefault(assertion.subject_id, []).append(assertion)
            if assertion.object_id is not None:
                by_object.setdefault(assertion.object_id, []).append(assertion)
        object.__setattr__(
            self, "_by_subject", {k: tuple(v) for k, v in by_subject.items()}
        )
        object.__setattr__(self, "_by_object", {k: tuple(v) for k, v in by_object.items()})
        return self

    def outgoing(self, subject_id: str, predicate: str | None = None) -> tuple[Assertion, ...]:
        self.indexed()
        rows = self._by_subject.get(subject_id, ())
        if predicate is None:
            return rows
        return tuple(a for a in rows if a.predicate == predicate)

    def incoming(self, object_id: str, predicate: str | None = None) -> tuple[Assertion, ...]:
        self.indexed()
        rows = self._by_object.get(object_id, ())
        if predicate is None:
            return rows
        return tuple(a for a in rows if a.predicate == predicate)


# ---------------------------------------------------------------------------
# Admissibility and temporal evaluation (spec: "Frozen vocabularies",
# "Temporal model")
# ---------------------------------------------------------------------------


def is_admissible(assertion: Assertion) -> bool:
    """Whether ``assertion`` "can affect semantic disposition".

    Requires ``curation_state`` in {curator_checked, independently_reviewed}
    and ``epistemic_state`` in {supported, verified, accepted}.
    ``review_state=passed`` is additionally required "when review is
    required by the governing source"; this evaluator treats
    ``review_state`` itself as stating whether review is required
    (``not_required`` needs no further check, anything else must equal
    ``passed``).
    """

    if assertion.curation_state not in {"curator_checked", "independently_reviewed"}:
        return False
    if assertion.epistemic_state not in {"supported", "verified", "accepted"}:
        return False
    if assertion.review_state not in {"not_required", "passed"}:
        return False
    return True


def temporal_state(assertion: Assertion, as_of: str | None) -> str:
    """Return ``"current"``, ``"not_yet"``, ``"ended"``, or ``"unknown"``.

    When ``as_of`` is ``None`` the query applies no explicit time bound (the
    query parameter tables use ``as_of=null`` for structural/current
    queries); the assertion is treated as ``"current"`` unless its domain
    start is unknown, in which case it remains ``"unknown"`` per "An
    assertion with unknown domain start is returned as provenance with
    temporal state unknown but cannot satisfy a query requiring known
    state."
    """

    if assertion.valid_from is None:
        return "unknown"
    if as_of is None:
        return "current"
    if assertion.valid_from > as_of:
        return "not_yet"
    if assertion.valid_to is not None and not (as_of < assertion.valid_to):
        return "ended"
    return "current"


def temporal_ok(assertion: Assertion, as_of: str | None) -> bool:
    return temporal_state(assertion, as_of) == "current"


def admissible_current(assertions: Iterable[Assertion], as_of: str | None) -> list[Assertion]:
    return [a for a in assertions if is_admissible(a) and temporal_ok(a, as_of)]


def is_superseded(entity_id: str, snapshot: Snapshot, as_of: str | None) -> bool:
    """Whether ``entity_id`` has an admissible, currently-effective
    ``supersedes`` assertion naming it as the object.

    "Supersession becomes effective at the admissible supersedes assertion's
    valid_from; before that instant the predecessor remains current." When
    two admissible successors have equal effective time the entity is
    treated as superseded (an ambiguous/disputed successor state, but the
    predecessor itself is no longer current); callers that need the
    ambiguity itself should inspect :func:`competing_successors`.
    """

    for assertion in snapshot.incoming(entity_id, "supersedes"):
        if not is_admissible(assertion):
            continue
        if assertion.valid_from is None:
            continue
        if as_of is not None and assertion.valid_from > as_of:
            continue
        return True
    return False


def competing_successors(entity_id: str, snapshot: Snapshot, as_of: str | None) -> tuple[Assertion, ...]:
    """Admissible, effective ``supersedes`` assertions naming ``entity_id``
    as object, sharing the earliest effective ``valid_from`` among them.
    More than one entry means an equal-time competing-successor ambiguity.
    """

    effective = [
        a
        for a in snapshot.incoming(entity_id, "supersedes")
        if is_admissible(a) and a.valid_from is not None and (as_of is None or a.valid_from <= as_of)
    ]
    if not effective:
        return ()
    earliest = min(a.valid_from for a in effective)
    return tuple(sorted((a for a in effective if a.valid_from == earliest), key=lambda a: a.assertion_id))


# ---------------------------------------------------------------------------
# RetrievalBundle v1 (spec: "Shared semantic evaluator and RetrievalBundle v1")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathStep:
    assertion_id: str
    direction: str  # "forward" | "reverse"

    def __post_init__(self) -> None:
        if self.direction not in {"forward", "reverse"}:
            raise EvaluatorError(f"unknown path direction {self.direction!r}")


def compute_path_id(steps: Sequence[PathStep]) -> str:
    sequence = [[step.assertion_id, step.direction] for step in steps]
    return frame_hash(sequence)


@dataclass(frozen=True)
class Path:
    path_id: str
    steps: tuple[PathStep, ...]

    @staticmethod
    def from_steps(steps: Sequence[PathStep]) -> "Path":
        return Path(path_id=compute_path_id(steps), steps=tuple(steps))


@dataclass(frozen=True)
class RetrievalBundle:
    contract_version: str
    query_instance_id: str
    snapshot_id: str
    as_of: str | None
    entities: tuple[str, ...]
    literals: tuple[str, ...]
    assertions: tuple[str, ...]
    source_revisions: tuple[str, ...]
    paths: tuple[Path, ...]
    diagnostics: Mapping[str, Any] | None = None


RETRIEVAL_BUNDLE_CONTRACT_VERSION = "cyax-retrieval-bundle-1.0"


def close_literal_references(
    snapshot: Snapshot, assertion_ids: Iterable[str], entity_ids: Iterable[str]
) -> tuple[str, ...]:
    """Every ``literal_ref`` of a referenced assertion plus every non-null
    ``display_label_ref`` of a referenced entity, resolved exactly once
    (spec: "assertion-literal closure")."""

    literal_ids: set[str] = set()
    for assertion_id in assertion_ids:
        assertion = snapshot.assertions[assertion_id]
        if assertion.literal_ref is not None:
            if assertion.literal_ref not in snapshot.literals:
                raise EvaluatorError(f"dangling literal_ref {assertion.literal_ref!r}")
            literal_ids.add(assertion.literal_ref)
    for entity_id in entity_ids:
        label = snapshot.entities[entity_id].display_label_ref
        if label is not None:
            if label not in snapshot.literals:
                raise EvaluatorError(f"dangling display_label_ref {label!r}")
            literal_ids.add(label)
    return tuple(sorted(literal_ids))


def build_bundle(
    snapshot: Snapshot,
    query_instance_id: str,
    as_of: str | None,
    *,
    entity_ids: Iterable[str] = (),
    assertion_ids: Iterable[str] = (),
    paths: Sequence[Path] = (),
    diagnostics: Mapping[str, Any] | None = None,
    include_display_literals: bool = True,
) -> RetrievalBundle:
    entity_set = sorted(set(entity_ids))
    assertion_set = sorted(set(assertion_ids))
    for step in (s for p in paths for s in p.steps):
        if step.assertion_id not in snapshot.assertions:
            raise EvaluatorError(f"path references unknown assertion {step.assertion_id!r}")
        assertion_set = sorted(set(assertion_set) | {step.assertion_id})
    for assertion_id in assertion_set:
        if assertion_id not in snapshot.assertions:
            raise EvaluatorError(f"bundle references unknown assertion {assertion_id!r}")
        assertion = snapshot.assertions[assertion_id]
        entity_set = sorted(set(entity_set) | {assertion.subject_id} | ({assertion.object_id} if assertion.object_id else set()))
    for entity_id in entity_set:
        if entity_id not in snapshot.entities:
            raise EvaluatorError(f"bundle references unknown entity {entity_id!r}")
    source_revision_ids = sorted(
        {snapshot.assertions[aid].source_revision_id for aid in assertion_set}
    )
    literal_ids = close_literal_references(
        snapshot, assertion_set, entity_set if include_display_literals else ()
    )
    ordered_paths = tuple(
        sorted(paths, key=lambda p: frame([[s.assertion_id, s.direction] for s in p.steps]))
    )
    return RetrievalBundle(
        contract_version=RETRIEVAL_BUNDLE_CONTRACT_VERSION,
        query_instance_id=query_instance_id,
        snapshot_id=snapshot.snapshot_id,
        as_of=as_of,
        entities=tuple(entity_set),
        literals=literal_ids,
        assertions=tuple(assertion_set),
        source_revisions=tuple(source_revision_ids),
        paths=ordered_paths,
        diagnostics=diagnostics,
    )


def strip_diagnostics(bundle: RetrievalBundle) -> RetrievalBundle:
    """Diagnostics-stripped projection for S/G parity comparison."""

    # The evaluator bundle stores canonical record IDs rather than records;
    # labels are therefore removed by the adapter-level projection.  Preserve
    # assertion-literal closure IDs and clear only diagnostics here.
    return RetrievalBundle(
        contract_version=bundle.contract_version,
        query_instance_id=bundle.query_instance_id,
        snapshot_id=bundle.snapshot_id,
        as_of=bundle.as_of,
        entities=bundle.entities,
        literals=bundle.literals,
        assertions=bundle.assertions,
        source_revisions=bundle.source_revisions,
        paths=bundle.paths,
        diagnostics=None,
    )


def validate_retrieval_bundle(snapshot: Snapshot, bundle: RetrievalBundle) -> RetrievalBundle:
    """Validate a bundle's sorted/unique/reference-closed ID projection."""
    if bundle.contract_version != RETRIEVAL_BUNDLE_CONTRACT_VERSION:
        raise EvaluatorError("unsupported retrieval-bundle contract version")
    for name, values in (
        ("entities", bundle.entities), ("literals", bundle.literals),
        ("assertions", bundle.assertions), ("source_revisions", bundle.source_revisions),
    ):
        if tuple(values) != tuple(sorted(values)) or len(values) != len(set(values)):
            raise EvaluatorError(f"bundle {name} must be sorted and unique")
    if any(entity_id not in snapshot.entities for entity_id in bundle.entities):
        raise EvaluatorError("bundle contains an unknown entity")
    if any(literal_id not in snapshot.literals for literal_id in bundle.literals):
        raise EvaluatorError("bundle contains an unknown literal")
    if any(assertion_id not in snapshot.assertions for assertion_id in bundle.assertions):
        raise EvaluatorError("bundle contains an unknown assertion")
    if any(revision_id not in snapshot.source_revisions for revision_id in bundle.source_revisions):
        raise EvaluatorError("bundle contains an unknown source revision")
    entity_ids = set(bundle.entities)
    source_ids = set(bundle.source_revisions)
    literal_ids = set(bundle.literals)
    required_literals: set[str] = set()
    for assertion_id in bundle.assertions:
        assertion = snapshot.assertions[assertion_id]
        if assertion.subject_id not in entity_ids or (assertion.object_id is not None and assertion.object_id not in entity_ids):
            raise EvaluatorError(f"assertion {assertion_id!r} is not entity-closed")
        if assertion.source_revision_id not in source_ids:
            raise EvaluatorError(f"assertion {assertion_id!r} is not source-closed")
        if assertion.literal_ref is not None:
            required_literals.add(assertion.literal_ref)
    for entity_id in bundle.entities:
        label = snapshot.entities[entity_id].display_label_ref
        if label is not None:
            required_literals.add(label)
    if required_literals != literal_ids:
        raise EvaluatorError("bundle literals are not exact reference closure")
    previous_path_key: bytes | None = None
    seen_paths: set[tuple[tuple[str, str], ...]] = set()
    assertion_ids = set(bundle.assertions)
    for path in bundle.paths:
        sequence = tuple((step.assertion_id, step.direction) for step in path.steps)
        if sequence in seen_paths:
            raise EvaluatorError("duplicate path step sequence")
        seen_paths.add(sequence)
        if any(step.assertion_id not in assertion_ids or step.direction not in {"forward", "reverse"} for step in path.steps):
            raise EvaluatorError("path is not assertion-closed")
        if path.path_id != compute_path_id(path.steps):
            raise EvaluatorError("path_id does not match path steps")
        path_key = frame([[s.assertion_id, s.direction] for s in path.steps])
        if previous_path_key is not None and path_key < previous_path_key:
            raise EvaluatorError("paths must be framed-sequence ordered")
        previous_path_key = path_key
    return bundle


# ---------------------------------------------------------------------------
# Nearest-rank percentile selector (spec: "Frozen query instances and gold")
# ---------------------------------------------------------------------------


def nearest_rank_select(
    population: Sequence[tuple[Any, int]], percentile: float, tie_key
) -> Any:
    """Select the nearest-rank ``percentile`` member of ``population``.

    ``population`` is a sequence of ``(item, metric)`` pairs. The population
    is sorted by ``(metric, tie_key(item))``; nearest-rank order statistic
    ``ceil(percentile * n)`` (one-based) is returned. Raises
    :class:`EvaluatorError` on an empty population (the "invalid-fixture
    rule": empty population invalidates the fixture).
    """

    if not population:
        raise EvaluatorError("empty population: invalid fixture")
    ordered = sorted(population, key=lambda pair: (pair[1], frame(tie_key(pair[0]))))
    n = len(ordered)
    import math

    rank = math.ceil(percentile * n)
    rank = max(1, min(n, rank))
    return ordered[rank - 1][0]


# ---------------------------------------------------------------------------
# Q01-Q12 population/answer implementations
# ---------------------------------------------------------------------------


def _entity_ids(snapshot: Snapshot) -> list[str]:
    return sorted(snapshot.entities)


def q01_population(snapshot: Snapshot, as_of: str | None) -> list[tuple[str, int]]:
    """Entities with >=1 admissible current governs result (they are the
    object of an admissible current ``governs`` assertion); metric = count
    of such governors."""

    out: list[tuple[str, int]] = []
    for entity_id in _entity_ids(snapshot):
        governors = admissible_current(snapshot.incoming(entity_id, "governs"), as_of)
        if governors:
            out.append((entity_id, len(governors)))
    return out


def q01_answer(snapshot: Snapshot, target: str, as_of: str | None) -> RetrievalBundle:
    governors = admissible_current(snapshot.incoming(target, "governs"), as_of)
    assertion_ids = sorted(a.assertion_id for a in governors)
    entity_ids = {target} | {a.subject_id for a in governors}
    return build_bundle(
        snapshot,
        query_instance_id=f"Q01:{target}",
        as_of=as_of,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
    )


def _authority_evidence(snapshot: Snapshot, target: str, as_of: str | None) -> list[Assertion]:
    """"Admissible authority-evidence result" for X: admissible current
    ``supports``/``documented_in`` assertions whose object is X (evidence
    bearing directly on X's authority), used by Q01/Q02.
    """

    candidates = list(snapshot.incoming(target, "supports"))
    return admissible_current(candidates, as_of)


def q02_population(snapshot: Snapshot, as_of: str | None) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for entity_id in _entity_ids(snapshot):
        evidence = _authority_evidence(snapshot, entity_id, as_of)
        if evidence:
            out.append((entity_id, len(evidence)))
    return out


def q02_answer(snapshot: Snapshot, target: str, as_of: str | None) -> RetrievalBundle:
    """Minimal admissible authority evidence for X, ordered by
    ``(authority_rank, assertion_id)``."""

    evidence = _authority_evidence(snapshot, target, as_of)
    ranked = sorted(
        evidence,
        key=lambda a: (Q02_AUTHORITY_RANK[a.authority_class], a.assertion_id),
    )
    assertion_ids = tuple(a.assertion_id for a in ranked)
    entity_ids = {target} | {a.subject_id for a in ranked}
    return build_bundle(
        snapshot,
        query_instance_id=f"Q02:{target}",
        as_of=as_of,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
    )


_Q03_CHAIN = ("Decision", "Requirement", "Implementation", "Verification")
# The frozen predicate table stores ``verifies`` as Verification ->
# Implementation.  The Q03 witness is therefore traversed in reverse for
# its final step (Implementation -> Verification).
_Q03_STEP_PREDICATES = ("requires", "requires", "verifies")
_Q03_STEP_DIRECTIONS = ("forward", "forward", "reverse")


def _typed_neighbors(
    snapshot: Snapshot,
    node: str,
    predicate: str,
    object_type: str,
    direction: str,
    as_of: str | None,
) -> list[tuple[str, Assertion]]:
    out: list[tuple[str, Assertion]] = []
    candidates = (
        snapshot.outgoing(node, predicate)
        if direction == "forward"
        else snapshot.incoming(node, predicate)
    )
    for a in admissible_current(candidates, as_of):
        neighbour = a.object_id if direction == "forward" else a.subject_id
        if neighbour is not None and snapshot.entities[neighbour].entity_type == object_type:
            out.append((neighbour, a))
    return sorted(out, key=lambda pair: pair[1].assertion_id)


def _shortest_typed_path(
    snapshot: Snapshot,
    start: str,
    chain_types: Sequence[str],
    step_predicates: Sequence[str],
    as_of: str | None,
    max_depth: int | None,
) -> list[PathStep] | None:
    """Breadth-first shortest witness along an exact typed predicate chain,
    breaking ties by lexicographically smallest step-assertion-ID sequence.
    """

    if max_depth is not None and len(step_predicates) > max_depth:
        return None
    frontier: list[tuple[str, list[PathStep]]] = [(start, [])]
    for index, (predicate, object_type) in enumerate(zip(step_predicates, chain_types[1:])):
        direction = _Q03_STEP_DIRECTIONS[index] if chain_types == _Q03_CHAIN else "forward"
        next_frontier: list[tuple[str, list[PathStep]]] = []
        for node, steps in frontier:
            for neighbor, assertion in _typed_neighbors(
                snapshot, node, predicate, object_type, direction, as_of
            ):
                next_frontier.append(
                    (neighbor, steps + [PathStep(assertion.assertion_id, direction)])
                )
        if not next_frontier:
            return None
        frontier = next_frontier
    best = min(
        frontier, key=lambda pair: frame([[s.assertion_id, s.direction] for s in pair[1]])
    )
    return best[1]


def _all_q03_paths(snapshot: Snapshot, start: str, path_depth: int) -> dict[str, list[PathStep]]:
    """Return one deterministic shortest witness for every reachable V."""
    if path_depth < len(_Q03_STEP_PREDICATES):
        return {}
    frontier: list[tuple[str, list[PathStep]]] = [(start, [])]
    for index, (predicate, object_type) in enumerate(zip(_Q03_STEP_PREDICATES, _Q03_CHAIN[1:])):
        direction = _Q03_STEP_DIRECTIONS[index]
        next_frontier: list[tuple[str, list[PathStep]]] = []
        for node, steps in frontier:
            for neighbour, assertion in _typed_neighbors(snapshot, node, predicate, object_type, direction, None):
                next_frontier.append((neighbour, steps + [PathStep(assertion.assertion_id, direction)]))
        frontier = next_frontier
        if not frontier:
            break
    grouped: dict[str, list[PathStep]] = {}
    for endpoint, steps in frontier:
        previous = grouped.get(endpoint)
        if previous is None or frame([[s.assertion_id, s.direction] for s in steps]) < frame([[s.assertion_id, s.direction] for s in previous]):
            grouped[endpoint] = steps
    return grouped


def q03_population(snapshot: Snapshot, path_depth: int) -> list[tuple[tuple[str, str], int]]:
    """Reachable Decision/Requirement/Implementation/Verification pairs;
    metric = shortest distance not exceeding ``path_depth``."""

    out: list[tuple[tuple[str, str], int]] = []
    for entity_id, entity in snapshot.entities.items():
        if entity.entity_type != "Decision":
            continue
        for end_entity, witness in _all_q03_paths(snapshot, entity_id, path_depth).items():
            out.append(((entity_id, end_entity), len(witness)))
    return out


def q03_answer(snapshot: Snapshot, target: tuple[str, str], path_depth: int) -> RetrievalBundle:
    start, end = target
    witness = _all_q03_paths(snapshot, start, path_depth).get(end)
    if witness is None:
        raise EvaluatorError(f"Q03 target {target!r} has no witness at depth {path_depth}")
    path = Path.from_steps(witness)
    entity_ids = {start, target[1]}
    for step in witness:
        a = snapshot.assertions[step.assertion_id]
        entity_ids.add(a.subject_id)
        if a.object_id:
            entity_ids.add(a.object_id)
    return build_bundle(
        snapshot,
        query_instance_id=f"Q03:{start}->{target[1]}",
        as_of=None,
        entity_ids=entity_ids,
        assertion_ids=[s.assertion_id for s in witness],
        paths=[path],
    )


def q04_population(snapshot: Snapshot) -> list[tuple[str, int]]:
    """Supersession assertions; metric is a monotone ordinal over
    ``valid_from`` so that nearest-rank selection follows the spec's
    "supersession assertions; valid_from instant" ordering, with the
    required assertion-ID/subject-ID tie break applied via ``tie_key``.
    """

    rows = [a for a in snapshot.assertions.values() if a.predicate == "supersedes" and is_admissible(a)]
    rows = [a for a in rows if a.valid_from is not None]
    ordered = sorted(rows, key=lambda a: (a.valid_from, a.assertion_id, a.subject_id))
    return [(a.assertion_id, i) for i, a in enumerate(ordered)]


def q04_answer(snapshot: Snapshot, target_assertion_id: str) -> RetrievalBundle:
    if target_assertion_id not in snapshot.assertions:
        raise EvaluatorError(f"unknown Q04 target assertion {target_assertion_id!r}")
    target = snapshot.assertions[target_assertion_id]
    if target.predicate != "supersedes" or target.valid_from is None:
        raise EvaluatorError("Q04 target must be an admissible supersedes assertion with valid_from")
    as_of = target.valid_from
    entities_superseded = set()
    assertions_out = set()
    for a in snapshot.assertions.values():
        if a.predicate != "supersedes" or not is_admissible(a):
            continue
        if a.valid_from is None or as_of is None:
            continue
        if a.valid_from <= as_of and a.object_id is not None:
            entities_superseded.add(a.object_id)
            entities_superseded.add(a.subject_id)
            assertions_out.add(a.assertion_id)
    return build_bundle(
        snapshot,
        query_instance_id=f"Q04:{target_assertion_id}",
        as_of=as_of,
        entity_ids=entities_superseded,
        assertion_ids=assertions_out,
    )


def q05_population(snapshot: Snapshot) -> list[tuple[str, int]]:
    out = []
    for entity_id in _entity_ids(snapshot):
        neighbors = admissible_current(snapshot.incoming(entity_id, "depends_on"), None)
        if neighbors:
            out.append((entity_id, len(neighbors)))
    return out


def q05_answer(snapshot: Snapshot, target: str) -> RetrievalBundle:
    neighbors = admissible_current(snapshot.incoming(target, "depends_on"), None)
    assertion_ids = sorted(a.assertion_id for a in neighbors)
    entity_ids = {target} | {a.subject_id for a in neighbors}
    return build_bundle(
        snapshot,
        query_instance_id=f"Q05:{target}",
        as_of=None,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
    )


def _bounded_reverse_dependants(
    snapshot: Snapshot, target: str, depth: int
) -> dict[str, list[PathStep]]:
    """Bounded reverse ``depends_on`` reachability from ``target``: nodes N
    such that N (transitively, within ``depth`` hops) depends on ``target``.
    Returns, for each reached node, its lexicographically smallest shortest
    witness path (target -> ... -> node, traversed in reverse direction).
    """

    best: dict[str, list[PathStep]] = {}
    frontier = [(target, [])]
    seen_at_depth = {target: 0}
    for level in range(1, depth + 1):
        next_frontier = []
        for node, steps in frontier:
            for a in admissible_current(snapshot.incoming(node, "depends_on"), None):
                dependant = a.subject_id
                new_steps = steps + [PathStep(a.assertion_id, "reverse")]
                if dependant not in seen_at_depth or seen_at_depth[dependant] == level:
                    seen_at_depth[dependant] = level
                    if dependant not in best or frame(
                        [[s.assertion_id, s.direction] for s in new_steps]
                    ) < frame([[s.assertion_id, s.direction] for s in best[dependant]]):
                        best[dependant] = new_steps
                    next_frontier.append((dependant, new_steps))
        frontier = next_frontier
        if not frontier:
            break
    best.pop(target, None)
    return best


def q06_population(snapshot: Snapshot, path_depth: int) -> list[tuple[str, int]]:
    out = []
    for entity_id in _entity_ids(snapshot):
        reached = _bounded_reverse_dependants(snapshot, entity_id, path_depth)
        if reached:
            out.append((entity_id, len(reached)))
    return out


def q06_answer(snapshot: Snapshot, target: str, path_depth: int) -> RetrievalBundle:
    reached = _bounded_reverse_dependants(snapshot, target, path_depth)
    paths = [Path.from_steps(steps) for _node, steps in sorted(reached.items())]
    assertion_ids = {s.assertion_id for steps in reached.values() for s in steps}
    entity_ids = {target} | set(reached)
    return build_bundle(
        snapshot,
        query_instance_id=f"Q06:{target}",
        as_of=None,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
        paths=paths,
    )


def _dependency_stale_reachable(snapshot: Snapshot, root: str, path_depth: int) -> dict[str, list[PathStep]]:
    """Entities reachable (within ``path_depth`` forward ``depends_on`` hops)
    from ``root`` that are dependency-stale because they (transitively)
    depend on ``root`` after ``root``'s admissible supersession."""

    best: dict[str, list[PathStep]] = {}
    frontier = [(root, [])]
    for _level in range(path_depth):
        next_frontier = []
        for node, steps in frontier:
            for a in admissible_current(snapshot.incoming(node, "depends_on"), None):
                dependant = a.subject_id
                new_steps = steps + [PathStep(a.assertion_id, "reverse")]
                key = frame([[s.assertion_id, s.direction] for s in new_steps])
                if dependant not in best or key < frame(
                    [[s.assertion_id, s.direction] for s in best[dependant]]
                ):
                    best[dependant] = new_steps
                next_frontier.append((dependant, new_steps))
        frontier = next_frontier
        if not frontier:
            break
    best.pop(root, None)
    return best


def q07_population(snapshot: Snapshot, as_of: str | None, path_depth: int) -> list[tuple[str, int]]:
    out = []
    for entity_id in _entity_ids(snapshot):
        if not is_superseded(entity_id, snapshot, as_of):
            continue
        reached = _dependency_stale_reachable(snapshot, entity_id, path_depth)
        if reached:
            out.append((entity_id, len(reached)))
    return out


def q07_answer(snapshot: Snapshot, target: str, as_of: str | None, path_depth: int) -> RetrievalBundle:
    reached = _dependency_stale_reachable(snapshot, target, path_depth)
    paths = [Path.from_steps(steps) for _node, steps in sorted(reached.items())]
    assertion_ids = {s.assertion_id for steps in reached.values() for s in steps}
    entity_ids = {target} | set(reached)
    return build_bundle(
        snapshot,
        query_instance_id=f"Q07:{target}",
        as_of=as_of,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
        paths=paths,
    )


def _dispute_records(snapshot: Snapshot, target: str, as_of: str | None) -> list[Assertion]:
    """Admissible and inspectable dispute records ("contradicts" assertions)
    concerning ``target``: a ``contradicts`` assertion concerns ``target``
    when either endpoint Claim has an admissible-current ``concerns``
    assertion naming ``target``. "Inspectable" includes disputed and
    resolved records, not only currently-undisputed ones, so admissibility
    (not the dispute_state value) is the sole gate beyond concernment.
    """

    concerning_claims = {
        a.subject_id
        for a in admissible_current(snapshot.incoming(target, "concerns"), as_of)
    }
    out = []
    for a in snapshot.assertions.values():
        if a.predicate != "contradicts" or not is_admissible(a):
            continue
        if a.subject_id in concerning_claims or a.object_id in concerning_claims:
            out.append(a)
    return sorted(out, key=lambda a: a.assertion_id)


def q08_population(snapshot: Snapshot, as_of: str | None) -> list[tuple[str, int]]:
    out = []
    for entity_id in _entity_ids(snapshot):
        records = _dispute_records(snapshot, entity_id, as_of)
        if records:
            out.append((entity_id, len(records)))
    return out


def q08_answer(snapshot: Snapshot, target: str, as_of: str | None) -> RetrievalBundle:
    records = _dispute_records(snapshot, target, as_of)
    assertion_ids = [a.assertion_id for a in records]
    entity_ids = {target}
    for a in records:
        entity_ids.add(a.subject_id)
        if a.object_id:
            entity_ids.add(a.object_id)
    diagnostics = {"derived_dispute_state": {a.assertion_id: a.dispute_state for a in records}}
    return build_bundle(
        snapshot,
        query_instance_id=f"Q08:{target}",
        as_of=as_of,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
        diagnostics=diagnostics,
    )


def _shortest_evidence_witness(
    snapshot: Snapshot, claim_id: str, path_depth: int
) -> list[PathStep] | None:
    """One shortest Claim -> Verification/Artifact -> Source evidence
    witness, lexicographically tied."""

    if path_depth < 1:
        return None
    frontier: list[tuple[str, tuple[PathStep, ...], frozenset[str]]] = [(claim_id, (), frozenset({claim_id}))]
    for _level in range(path_depth):
        candidates: list[tuple[str, tuple[PathStep, ...], frozenset[str]]] = []
        for node, steps, visited_nodes in frontier:
            for a in admissible_current(snapshot.incoming(node, "verifies"), None):
                next_node = a.subject_id
                if next_node not in visited_nodes:
                    candidates.append((next_node, steps + (PathStep(a.assertion_id, "reverse"),), visited_nodes | {next_node}))
            for a in admissible_current(snapshot.incoming(node, "supports"), None):
                next_node = a.subject_id
                if next_node not in visited_nodes:
                    candidates.append((next_node, steps + (PathStep(a.assertion_id, "reverse"),), visited_nodes | {next_node}))
            for a in admissible_current(snapshot.outgoing(node, "documented_in"), None):
                if a.object_id is not None and a.object_id not in visited_nodes:
                    candidates.append((a.object_id, steps + (PathStep(a.assertion_id, "forward"),), visited_nodes | {a.object_id}))
        candidates.sort(key=lambda item: frame([[s.assertion_id, s.direction] for s in item[1]]))
        source_candidates = [item for item in candidates if snapshot.entities[item[0]].entity_type == "Source"]
        if source_candidates:
            return list(source_candidates[0][1])
        frontier = candidates
        if not frontier:
            return None
    return None


def q09_population(snapshot: Snapshot, path_depth: int) -> list[tuple[tuple[str, str], int]]:
    out = []
    for entity_id, entity in snapshot.entities.items():
        if entity.entity_type != "Claim":
            continue
        witness = _shortest_evidence_witness(snapshot, entity_id, path_depth)
        if witness is not None:
            last = snapshot.assertions[witness[-1].assertion_id]
            source_id = last.object_id if witness[-1].direction == "forward" else last.subject_id
            out.append(((entity_id, source_id), len(witness)))
    return out


def q09_answer(snapshot: Snapshot, target: tuple[str, str], path_depth: int) -> RetrievalBundle:
    claim_id, _source_id = target
    witness = _shortest_evidence_witness(snapshot, claim_id, path_depth)
    if witness is None:
        raise EvaluatorError(f"Q09 target {target!r} has no witness")
    path = Path.from_steps(witness)
    entity_ids = {target[0], target[1]}
    for step in witness:
        a = snapshot.assertions[step.assertion_id]
        entity_ids.add(a.subject_id)
        if a.object_id:
            entity_ids.add(a.object_id)
    return build_bundle(
        snapshot,
        query_instance_id=f"Q09:{claim_id}->{target[1]}",
        as_of=None,
        entity_ids=entity_ids,
        assertion_ids=[s.assertion_id for s in witness],
        paths=[path],
    )


def _claim_has_source_evidence(snapshot: Snapshot, claim_id: str, path_depth: int) -> bool:
    return _shortest_evidence_witness(snapshot, claim_id, path_depth) is not None


def _minimal_evidence_cover(
    snapshot: Snapshot, claim_a: str, claim_b: str, path_depth: int
) -> list[tuple[Assertion, ...]]:
    """Deterministic minimal evidence set covering both Claims: minimize
    assertion count, then source-revision count, then framed sorted
    assertion-ID list; return every exact tie.

    Each Claim's coverage is provided by any admissible-current evidence
    chain reaching a Source within ``path_depth``; this reference
    implementation enumerates every simple witness (bounded by
    ``path_depth``) per Claim and searches the Cartesian product of
    (witness_a, witness_b) covers, deduplicating shared assertions.
    """

    witnesses_a = _all_evidence_witnesses(snapshot, claim_a, path_depth)
    witnesses_b = _all_evidence_witnesses(snapshot, claim_b, path_depth)
    if not witnesses_a or not witnesses_b:
        return []
    covers: dict[tuple[str, ...], tuple[Assertion, ...]] = {}
    for wa in witnesses_a:
        for wb in witnesses_b:
            ids = sorted({s.assertion_id for s in wa} | {s.assertion_id for s in wb})
            covers[tuple(ids)] = tuple(snapshot.assertions[i] for i in ids)
    best_key = None
    best_covers: list[tuple[str, ...]] = []
    for ids, assertions in covers.items():
        source_revisions = len({a.source_revision_id for a in assertions})
        key = (len(ids), source_revisions, frame(list(ids)))
        if best_key is None or key < best_key:
            best_key = key
            best_covers = [ids]
        elif key == best_key:
            best_covers.append(ids)
    return [covers[ids] for ids in best_covers]


def _all_evidence_witnesses(
    snapshot: Snapshot, claim_id: str, path_depth: int
) -> list[tuple[PathStep, ...]]:
    results: list[tuple[PathStep, ...]] = []

    def walk(node: str, steps: tuple[PathStep, ...], depth: int, visited: frozenset[str]) -> None:
        if depth > path_depth:
            return
        if snapshot.entities[node].entity_type == "Source" and steps:
            results.append(steps)
            return
        for a in admissible_current(snapshot.incoming(node, "verifies"), None):
            if a.subject_id not in visited:
                walk(a.subject_id, steps + (PathStep(a.assertion_id, "reverse"),), depth + 1, visited | {a.subject_id})
        for a in admissible_current(snapshot.incoming(node, "supports"), None):
            if a.subject_id not in visited:
                walk(a.subject_id, steps + (PathStep(a.assertion_id, "reverse"),), depth + 1, visited | {a.subject_id})
        for a in admissible_current(snapshot.outgoing(node, "documented_in"), None):
            if a.object_id is not None and a.object_id not in visited:
                walk(a.object_id, steps + (PathStep(a.assertion_id, "forward"),), depth + 1, visited | {a.object_id})

    walk(claim_id, (), 0, frozenset({claim_id}))
    results.sort(key=lambda steps: frame([[s.assertion_id, s.direction] for s in steps]))
    return results


def q10_population(
    snapshot: Snapshot, as_of: str | None, path_depth: int
) -> list[tuple[tuple[str, str], int]]:
    claim_ids = sorted(
        eid for eid, e in snapshot.entities.items() if e.entity_type == "Claim"
    )
    eligible = [c for c in claim_ids if _claim_has_source_evidence(snapshot, c, path_depth)]
    out = []
    if len(eligible) < 2:
        return out
    for i, claim_a in enumerate(eligible):
        for claim_b in eligible[i + 1 :]:
            covers = _minimal_evidence_cover(snapshot, claim_a, claim_b, path_depth)
            if covers:
                metric = len(covers[0])
                out.append(((claim_a, claim_b), metric))
    return out


def q10_answer(
    snapshot: Snapshot, target: tuple[str, str], as_of: str | None, path_depth: int
) -> RetrievalBundle:
    claim_a, claim_b = target
    covers = _minimal_evidence_cover(snapshot, claim_a, claim_b, path_depth)
    all_assertion_ids = sorted({a.assertion_id for cover in covers for a in cover})
    entity_ids = {claim_a, claim_b}
    for aid in all_assertion_ids:
        a = snapshot.assertions[aid]
        entity_ids.add(a.subject_id)
        if a.object_id:
            entity_ids.add(a.object_id)
    diagnostics = {
        "tied_covers": [sorted(a.assertion_id for a in cover) for cover in covers]
    }
    return build_bundle(
        snapshot,
        query_instance_id=f"Q10:{claim_a},{claim_b}",
        as_of=as_of,
        entity_ids=entity_ids,
        assertion_ids=all_assertion_ids,
        diagnostics=diagnostics,
    )


def _q11_eligible_claims(
    snapshot: Snapshot, target: str, as_of: str | None, claim_slots: Mapping[str, str]
) -> list[tuple[Assertion, Assertion]]:
    """(concerns_assertion, claim_documented_in_assertion) pairs for Claims
    of the registered ``synthetic.block_claim``/``synthetic_fixture_block_statement``
    slot that concern ``target``, are current/unsuperseded, and have no
    unmet admissible ``depends_on`` requirement or unresolved dispute.

    Eligibility requires both: the specific Claim's own ``claim_key`` equals
    ``synthetic.block_claim`` (spec: "Q11 includes only the registered
    block-claim slot"), and the fixture's frozen ``claim_key_registry``
    (``claim_slots``) maps that key to ``synthetic_fixture_block_statement``.
    """

    out = []
    for a in admissible_current(snapshot.incoming(target, "concerns"), as_of):
        claim_id = a.subject_id
        claim = snapshot.entities[claim_id]
        if claim.entity_type != "Claim":
            continue
        if claim.claim_key != Q11_ELIGIBLE_CLAIM_KEY:
            continue
        if claim_slots.get(claim.claim_key) != Q11_ELIGIBLE_SLOT:
            continue
        if is_superseded(claim_id, snapshot, as_of):
            continue
        documented = [
            d
            for d in admissible_current(snapshot.outgoing(claim_id, "documented_in"), as_of)
            if d.literal_ref is not None and d.literal_ref in snapshot.literals
        ]
        if not documented:
            continue
        # Unmet admissible depends_on requirement check:
        unmet = False
        for dep in admissible_current(snapshot.outgoing(claim_id, "depends_on"), as_of):
            if dep.object_id is not None and is_superseded(dep.object_id, snapshot, as_of):
                unmet = True
        if unmet:
            continue
        # Unresolved dispute: any admissible contradicts assertion touching
        # the claim whose dispute_state is not resolved_* or undisputed.
        disputes = [
            d
            for d in snapshot.assertions.values()
            if d.predicate == "contradicts"
            and is_admissible(d)
            and (d.subject_id == claim_id or d.object_id == claim_id)
        ]
        if any(d.dispute_state == "disputed" for d in disputes):
            continue
        out.append((a, documented[0]))
    return out


def q11_population(
    snapshot: Snapshot, as_of: str | None, claim_slots: Mapping[str, str]
) -> list[tuple[str, int]]:
    """``claim_slots`` maps claim_key -> semantic_slot for the fixture's
    frozen ``claim_key_registry`` (spec: "registered fixture-local semantic
    key"); Q11 requires ``claim_key='synthetic.block_claim'`` with slot
    ``synthetic_fixture_block_statement``.
    """

    out = []
    for entity_id, entity in snapshot.entities.items():
        if entity.entity_type != "WorkItem":
            continue
        eligible = _q11_eligible_claims(snapshot, entity_id, as_of, claim_slots)
        if eligible:
            out.append((entity_id, len(eligible)))
    return out


def q11_answer(
    snapshot: Snapshot, target: str, as_of: str | None, claim_slots: Mapping[str, str]
) -> RetrievalBundle:
    eligible = _q11_eligible_claims(snapshot, target, as_of, claim_slots)
    # Rank reflects the Claim's own content assertion (documented_in), the
    # exact statement the Claim asserts, not the unrelated concerns edge.
    claim_epistemic = {
        pair[0].subject_id: pair[1].epistemic_state for pair in eligible
    }
    if not claim_epistemic:
        best_claims: list[str] = []
    else:
        best_rank = min(Q11_EPISTEMIC_RANK.get(s, 99) for s in claim_epistemic.values())
        best_claims = sorted(
            cid for cid, state in claim_epistemic.items() if Q11_EPISTEMIC_RANK.get(state, 99) == best_rank
        )
    assertion_ids = sorted(
        {pair[0].assertion_id for pair in eligible if pair[0].subject_id in best_claims}
        | {pair[1].assertion_id for pair in eligible if pair[0].subject_id in best_claims}
    )
    entity_ids = {target} | set(best_claims)
    return build_bundle(
        snapshot,
        query_instance_id=f"Q11:{target}",
        as_of=as_of,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
    )


def q12_population(snapshot: Snapshot, as_of: str | None) -> list[tuple[str, int]]:
    out = []
    for entity_id, entity in snapshot.entities.items():
        if entity.entity_type != "Requirement":
            continue
        implementers = admissible_current(snapshot.incoming(entity_id, "implements"), as_of)
        verifiers = admissible_current(snapshot.incoming(entity_id, "verifies"), as_of)
        if implementers and verifiers:
            out.append((entity_id, len(implementers) + len(verifiers)))
    return out


def q12_answer(snapshot: Snapshot, target: str, as_of: str | None) -> RetrievalBundle:
    implementers = admissible_current(snapshot.incoming(target, "implements"), as_of)
    verifiers = admissible_current(snapshot.incoming(target, "verifies"), as_of)
    assertion_ids = sorted({a.assertion_id for a in implementers} | {a.assertion_id for a in verifiers})
    entity_ids = {target} | {a.subject_id for a in implementers} | {a.subject_id for a in verifiers}
    return build_bundle(
        snapshot,
        query_instance_id=f"Q12:{target}",
        as_of=as_of,
        entity_ids=entity_ids,
        assertion_ids=assertion_ids,
    )


def select_query_instance(
    snapshot: Snapshot,
    query_id: str,
    *,
    path_depth: int = 3,
    claim_slots: Mapping[str, str] | None = None,
) -> Any:
    """Apply the frozen selector for one query population.

    The returned target is suitable for the corresponding ``qXX_answer``
    function.  Empty populations raise ``EvaluatorError`` so a malformed
    tier/profile/seed is rejected before backend work.
    """
    if query_id not in QUERY_DEFINITIONS:
        raise EvaluatorError(f"unknown query id {query_id!r}")
    if query_id == "Q01":
        return nearest_rank_select(q01_population(snapshot, QUERY_DEFINITIONS[query_id]["as_of"]), 0.50, lambda value: value)
    if query_id == "Q02":
        return nearest_rank_select(q02_population(snapshot, QUERY_DEFINITIONS[query_id]["as_of"]), 0.90, lambda value: value)
    if query_id == "Q03":
        population = q03_population(snapshot, path_depth)
        if not population:
            raise EvaluatorError("empty Q03 population: invalid fixture")
        maximum = max(metric for _item, metric in population)
        return min((item for item, metric in population if metric == maximum), key=lambda pair: frame(list(pair)))
    if query_id == "Q04":
        return nearest_rank_select(q04_population(snapshot), 0.50, lambda value: value)
    if query_id == "Q05":
        return nearest_rank_select(q05_population(snapshot), 0.50, lambda value: value)
    if query_id == "Q06":
        return nearest_rank_select(q06_population(snapshot, path_depth), 0.90, lambda value: value)
    if query_id == "Q07":
        return nearest_rank_select(q07_population(snapshot, QUERY_DEFINITIONS[query_id]["as_of"], path_depth), 0.90, lambda value: value)
    if query_id == "Q08":
        return nearest_rank_select(q08_population(snapshot, QUERY_DEFINITIONS[query_id]["as_of"]), 0.50, lambda value: value)
    if query_id == "Q09":
        population = q09_population(snapshot, path_depth)
        if not population:
            raise EvaluatorError("empty Q09 population: invalid fixture")
        maximum = max(metric for _item, metric in population)
        return min((item for item, metric in population if metric == maximum), key=lambda pair: frame(list(pair)))
    if query_id == "Q10":
        return nearest_rank_select(q10_population(snapshot, None, path_depth), 0.90, lambda value: value)
    if query_id == "Q11":
        if claim_slots is None:
            raise EvaluatorError("Q11 requires its frozen claim-key registry")
        return nearest_rank_select(q11_population(snapshot, QUERY_DEFINITIONS[query_id]["as_of"], claim_slots), 0.50, lambda value: value)
    return nearest_rank_select(q12_population(snapshot, QUERY_DEFINITIONS[query_id]["as_of"]), 0.50, lambda value: value)


def gold_bundle_checksum(bundle: RetrievalBundle) -> str:
    """Canonical checksum of the semantic gold ID projection."""
    record = {
        "contract_version": bundle.contract_version,
        "query_instance_id": bundle.query_instance_id,
        "snapshot_id": bundle.snapshot_id,
        "as_of": bundle.as_of,
        "entities": list(bundle.entities),
        "literals": list(bundle.literals),
        "assertions": list(bundle.assertions),
        "source_revisions": list(bundle.source_revisions),
        "paths": [[[step.assertion_id, step.direction] for step in path.steps] for path in bundle.paths],
    }
    return frame_hash(record)


def gold_record(bundle: RetrievalBundle) -> dict[str, Any]:
    """Return the frozen, diagnostics-free semantic gold record."""
    return {
        "contract_version": bundle.contract_version,
        "query_instance_id": bundle.query_instance_id,
        "snapshot_id": bundle.snapshot_id,
        "as_of": bundle.as_of,
        "entities": list(bundle.entities),
        "literals": list(bundle.literals),
        "assertions": list(bundle.assertions),
        "source_revisions": list(bundle.source_revisions),
        "paths": [
            {"path_id": path.path_id, "steps": [{"assertion_id": step.assertion_id, "direction": step.direction} for step in path.steps]}
            for path in bundle.paths
        ],
        "gold_checksum": gold_bundle_checksum(bundle),
    }
