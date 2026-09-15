"""Independent reproduction of CYAX-0168 F-scale generator version 2.3.

This module is Worker D's independent implementation for CYAX-0168 G1 (see
``specs/0168-structured-graph-materialization/spec.md``, section "F-scale
generator"). It was written from the specification text only, without
opening, importing, or otherwise consulting ``generator_primary.py`` (the
sibling independent implementation produced by another worker) or
``common.py``. All canonical framing, ID, and canonical-JSON primitives
below are re-derived locally from spec.md rather than reused.

Only small synthetic conformance parameters are exercised by the paired test
module (``tests/test_generator_independent.py``); this module never accesses
or materializes a real T0-T4 decision fixture.  The exact C0-C3 calibration
matrix is registered below for callers that explicitly request calibration
materialization.

Assumptions made where spec.md's generator-2.3 section itself is silent
(flagged here for manager reconciliation against the other independent
implementation; none of them affect entity/literal/source-revision/assertion
identity or the generator's own PRF/phase mechanics, which are fully pinned
by spec.md):

* The "Immutable assertion snapshot and identity" section's logical
  checksum formula (spec.md lines ~429-441) references top-level
  ``schema_version``, ``authority_rule_version``, and
  ``semantic_evaluator_rule_version`` tokens that are not given concrete
  values anywhere in spec.md. Given the document's pervasive "v1"/"v2"
  naming convention (``cyax-entity-v1``, ``cyax-source-revision-v1``,
  ``cyax-literal-v1``, ``cyax-assertion-v2``, ``cyax-logical-snapshot-v1``)
  and the absence of any other version token, this module uses the repository
  v1 schema/rule names as defaults (``cyax-snapshot-v1``,
  ``cyax-authority-v1``, and ``cyax-evaluator-v1``). Callers may override
  them.
* ``semantic_source_bundle_projection_checksum`` is defined for the general
  snapshot pipeline in terms of a captured *source bundle* that is outside
  generator 2.3's synthetic-only scope (generator-owned bytes ARE their own
  bundle). Absent a synthetic-bundle definition in spec.md, this module
  computes a bundle-projection checksum from the same synthetic source
  revisions referenced by semantic assertions, framed under the standard
  ``cyax-source-bundle-projection-v1`` tag. Callers may override this too.
  ``semantic_projection_checksum`` (no outer wrapper, no external
  constants) is exposed separately as an unambiguous, fully generator-scoped
  content-stability value.
* The canonical physical ``source_revisions`` record uses the exact common
  ``SourceRevision`` schema, including ``canonical_locator`` and
  ``object_byte_count`` plus nullable ``source_event_at`` and ``actor_id``.
  Generator-only raw bytes and diagnostic evidence are retained internally;
  they are not emitted in the canonical record.  The complete physical
  projection includes base and final-assertion provenance revisions, while
  the semantic projection filters to assertion-referenced revisions.
  The exact non-identity-bearing field set of a physical ``source_revisions``
  JSONL record for a *synthetic* revision (e.g. field names for the null
  actor/login/author-association/event-state placeholders) is not spelled
  out character-for-character in spec.md. This module names them
  ``actor_id``, ``actor_login``, ``author_association``, and ``event_state``
  and fixes them to ``None``, consistent with "Synthetic source-revision
  records use null actor ID, login, author association, event/state
  fields." These fields are excluded from every identity/checksum
  computation (they are diagnostic metadata only) so this naming choice
  cannot affect byte/ID/checksum reproduction.

Every other rule below is a direct, literal implementation of spec.md's
F-scale generator section, canonical encoding section, and common
entity/literal/assertion schema sections.
"""

from __future__ import annotations

import hashlib
import json as _json
import math
import unicodedata
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Canonical domain-separated framing (spec.md "Canonical encoding and stable
# IDs"). Values are encoded as: null, byte string, string, integer,
# false/true, array, object. Lengths/counts are u64be. Floats are forbidden.
# ---------------------------------------------------------------------------


def _u64be(n: int) -> bytes:
    if n < 0 or n > (2**64 - 1):
        raise ValueError(f"length/count out of u64 range: {n}")
    return n.to_bytes(8, "big")


def _frame_null() -> bytes:
    return b"N" + _u64be(0)


def _frame_bytes(raw: bytes) -> bytes:
    return b"X" + _u64be(len(raw)) + raw


def _frame_string(s: str) -> bytes:
    data = unicodedata.normalize("NFC", s).encode("utf-8")
    return b"S" + _u64be(len(data)) + data


def _frame_integer(n: int) -> bytes:
    if isinstance(n, bool):  # pragma: no cover - defensive
        raise TypeError("bool is not a canonical integer")
    text = str(n)
    data = text.encode("ascii")
    return b"I" + _u64be(len(data)) + data


def _frame_bool(v: bool) -> bytes:
    return b"B" + _u64be(1) + (b"\x01" if v else b"\x00")


def _frame_array(items: Sequence[object]) -> bytes:
    out = [b"A", _u64be(len(items))]
    for item in items:
        out.append(frame(item))
    return b"".join(out)


def _frame_object(obj: Dict[str, object]) -> bytes:
    keyed = []
    seen_keys = set()
    for key, value in obj.items():
        if not isinstance(key, str):
            raise TypeError("canonical object keys must be strings")
        normalized = unicodedata.normalize("NFC", key)
        if normalized in seen_keys:
            raise ValueError(f"canonical object has colliding NFC key: {normalized!r}")
        seen_keys.add(normalized)
        keyed.append((normalized.encode("utf-8"), normalized, value))
    keyed.sort(key=lambda t: t[0])
    out = [b"O", _u64be(len(keyed))]
    for _, k, v in keyed:
        out.append(_frame_string(k))
        out.append(frame(v))
    return b"".join(out)


def frame(value: object) -> bytes:
    """Canonical domain-separated frame of ``value`` per spec.md."""
    if value is None:
        return _frame_null()
    if isinstance(value, bool):
        return _frame_bool(value)
    if isinstance(value, int):
        return _frame_integer(value)
    if isinstance(value, str):
        return _frame_string(value)
    if isinstance(value, (bytes, bytearray)):
        return _frame_bytes(bytes(value))
    if isinstance(value, (list, tuple)):
        return _frame_array(value)
    if isinstance(value, dict):
        return _frame_object(value)
    raise TypeError(f"unsupported value type for canonical frame: {type(value)!r}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


# ---------------------------------------------------------------------------
# Stable ID renderers (spec.md "Canonical encoding and stable IDs").
# ---------------------------------------------------------------------------


def entity_id(namespace: str, entity_type: str, canonical_source_identity: Sequence[object]) -> str:
    digest = sha256_hex(frame(["cyax-entity-v1", namespace, entity_type, list(canonical_source_identity)]))
    return f"cyax-entity-sha256:{digest}"


def literal_id(literal_type: str, canonical_value: object) -> str:
    digest = sha256_hex(frame(["cyax-literal-v1", literal_type, canonical_value]))
    return f"cyax-literal-sha256:{digest}"


def source_revision_id(
    source_kind: str,
    canonical_locator: str,
    object_sha256: str,
    source_event_at: Optional[str],
) -> str:
    digest = sha256_hex(
        frame(["cyax-source-revision-v1", source_kind, canonical_locator, object_sha256, source_event_at])
    )
    return f"cyax-source-revision-sha256:{digest}"


ASSERTION_PREIMAGE_KEYS = (
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


def assertion_id(preimage: Dict[str, object]) -> str:
    missing = [k for k in ASSERTION_PREIMAGE_KEYS if k not in preimage]
    if missing:
        raise ValueError(f"assertion preimage missing keys: {missing}")
    extra = sorted(set(preimage) - set(ASSERTION_PREIMAGE_KEYS))
    if extra:
        raise ValueError(f"assertion preimage has unexpected keys: {extra}")
    digest = sha256_hex(frame(["cyax-assertion-v2", preimage]))
    return f"cyax-assertion-sha256:{digest}"


# ---------------------------------------------------------------------------
# Canonical JSON for generator-owned captured source bytes (spec.md
# "sorted-key canonical JSON plus LF").
# ---------------------------------------------------------------------------


def canonical_json_bytes(obj: Dict[str, object]) -> bytes:
    text = _json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return text.encode("ascii") + b"\n"


# ---------------------------------------------------------------------------
# PRF and candidate selection (spec.md "F-scale generator").
# ---------------------------------------------------------------------------

GENERATOR_NAME = "cyax-0168-scale-2.3"
GENERATOR_VERSION = "2.3"
PRF_DOMAIN = "cyax-gen-2.3"
NAMESPACE = "cyax-0168-synthetic-v2.3"
SOURCE_LOCATOR_VERSION = "/scale/2.3/"
AUTHORITY_DERIVATION_RULE_ID = "synthetic_fixture_v2.3"
PURPOSE_TOKENS = ("dependency_target", "cross_component_dependency_target", "fill_concerns_pair")

FIXED_TIME = "2000-01-01T00:00:00.000000Z"
MAX_COUNTER = 2**64 - 1


class GenerationFailure(Exception):
    """Raised whenever generator 2.3's own closed failure rules trigger."""


def dependency_target_rejects(subject_id: str, candidate_object_id: str, existing_triples: set) -> bool:
    """Exhaustive post-PRF rejection predicates shared by ``dependency_target``
    and ``cross_component_dependency_target`` (spec.md purpose table): (1)
    self-edge; (2) the exact ``(subject, depends_on, object)`` triple already
    present. No other candidate property may cause a retry."""
    return subject_id == candidate_object_id or (
        subject_id,
        "depends_on",
        candidate_object_id,
    ) in existing_triples


def fill_concerns_pair_rejects(subject_id: str, object_id: str, existing_triples: set) -> bool:
    """Exhaustive post-PRF rejection predicates for ``fill_concerns_pair``:
    (1) self-edge; (2) the exact ``(subject, concerns, object)`` triple
    already present."""
    return subject_id == object_id or (subject_id, "concerns", object_id) in existing_triples


def prf_digest(seed: int, profile_id: str, purpose: str, ordinal: int, counter: int) -> bytes:
    if purpose not in PURPOSE_TOKENS:
        raise ValueError(f"non-conforming purpose token: {purpose!r}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (seed, ordinal, counter)
    ):
        raise ValueError("seed, ordinal, and counter must be nonnegative integers")
    if counter > MAX_COUNTER:
        raise ValueError("counter exceeds u64 maximum")
    return sha256_bytes(frame([PRF_DOMAIN, seed, profile_id, purpose, ordinal, counter]))


def prf_select(
    seed: int,
    profile_id: str,
    purpose: str,
    ordinal: int,
    candidates: Sequence[object],
    sort_key: Callable[[object], object],
    rejects: Callable[[object], bool],
    *,
    candidates_sorted: bool = False,
    trace: Optional[List[Dict[str, object]]] = None,
    phase: Optional[str] = None,
    subject_id: Optional[str] = None,
) -> object:
    """Generic implementation of the closed generator-2.3 selection rule.

    ``rejects`` must implement exactly the purpose row's exhaustive
    post-PRF rejection predicates and nothing else.  If ``trace`` is supplied,
    append one complete choice record.  The trace is diagnostic output, but it
    is deliberately complete: it records the frozen candidate vector, every
    digest/counter attempt, each retry cause, and the accepted candidate.
    """
    if purpose not in PURPOSE_TOKENS:
        raise ValueError(f"non-conforming purpose token: {purpose!r}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (seed, ordinal)
    ):
        raise ValueError("seed and ordinal must be nonnegative integers")
    n = len(candidates)
    if n == 0:
        raise GenerationFailure(f"empty candidate vector for purpose={purpose} ordinal={ordinal}")
    # Candidate vectors are frozen by the generator before a phase begins.
    # Callers may pass ``candidates_sorted=True`` to avoid repeatedly sorting
    # the same immutable vector for every ordinal; the default retains the
    # standalone selector's defensive sort and its public conformance
    # behavior.
    sorted_candidates = list(candidates) if candidates_sorted else sorted(candidates, key=sort_key)

    def trace_value(value: object) -> object:
        # JSON-friendly and deterministic representation for pair candidates.
        if isinstance(value, tuple):
            return [trace_value(item) for item in value]
        if isinstance(value, list):
            return [trace_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): trace_value(item) for key, item in value.items()}
        return value

    attempts: List[Dict[str, object]] = []
    choice_record: Dict[str, object] = {
        "phase": phase,
        "purpose": purpose,
        "ordinal": ordinal,
        "candidate_count": n,
        "candidates": [trace_value(item) for item in sorted_candidates],
        "attempts": attempts,
    }
    if subject_id is not None:
        choice_record["subject_id"] = subject_id
    if trace is not None:
        trace.append(choice_record)

    if n == 1:
        cand = sorted_candidates[0]
        rejected = rejects(cand)
        attempts.append({
            "counter": 0,
            "prf_called": False,
            "candidate": trace_value(cand),
            "digest": None,
            "digest_rejected": False,
            "predicate_rejected": rejected,
        })
        if rejected:
            choice_record.update({"selected": None, "selected_counter": None, "retry_count": 0, "failed": True})
            raise GenerationFailure(
                f"sole candidate rejected without retry for purpose={purpose} ordinal={ordinal}"
            )
        choice_record.update({"selected": trace_value(cand), "selected_counter": 0, "retry_count": 0, "failed": False})
        return cand

    counter = 0
    while True:
        digest = prf_digest(seed, profile_id, purpose, ordinal, counter)
        x = int.from_bytes(digest, "big")
        limit = (2**256 // n) * n
        if x >= limit:
            attempts.append({
                "counter": counter,
                "prf_called": True,
                "candidate": None,
                "digest": digest.hex(),
                "digest_rejected": True,
                "predicate_rejected": False,
            })
            counter += 1
            if counter > MAX_COUNTER:
                choice_record.update({"selected": None, "selected_counter": None, "retry_count": counter, "failed": True})
                raise GenerationFailure("counter overflow during digest rejection")
            continue
        cand = sorted_candidates[x % n]
        rejected = rejects(cand)
        attempts.append({
            "counter": counter,
            "prf_called": True,
            "candidate": trace_value(cand),
            "digest": digest.hex(),
            "digest_rejected": False,
            "predicate_rejected": rejected,
        })
        if rejected:
            counter += 1
            if counter > MAX_COUNTER:
                choice_record.update({"selected": None, "selected_counter": None, "retry_count": counter, "failed": True})
                raise GenerationFailure("counter overflow during predicate rejection")
            continue
        choice_record.update({
            "selected": trace_value(cand),
            "selected_counter": counter,
            "retry_count": counter,
            "failed": False,
        })
        return cand


# ---------------------------------------------------------------------------
# Profiles (spec.md "Topology is separate from seed" table).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    dependency_fanout: int
    path_depth: int
    supersession_depth: int
    dispute_rate: Fraction
    cross_link_rate: Fraction
    cycle_rate: Fraction
    isolated_rate: Fraction
    parallel_evidence_rate: Fraction


PROFILES: Dict[str, Profile] = {
    "P-low": Profile(
        dependency_fanout=2,
        path_depth=4,
        supersession_depth=2,
        dispute_rate=Fraction("0.005"),
        cross_link_rate=Fraction("0.01"),
        cycle_rate=Fraction("0.000"),
        isolated_rate=Fraction("0.20"),
        parallel_evidence_rate=Fraction("0.01"),
    ),
    "P-medium": Profile(
        dependency_fanout=8,
        path_depth=8,
        supersession_depth=8,
        dispute_rate=Fraction("0.020"),
        cross_link_rate=Fraction("0.05"),
        cycle_rate=Fraction("0.005"),
        isolated_rate=Fraction("0.05"),
        parallel_evidence_rate=Fraction("0.03"),
    ),
    "P-high": Profile(
        dependency_fanout=32,
        path_depth=16,
        supersession_depth=32,
        dispute_rate=Fraction("0.100"),
        cross_link_rate=Fraction("0.20"),
        cycle_rate=Fraction("0.020"),
        isolated_rate=Fraction("0.01"),
        parallel_evidence_rate=Fraction("0.10"),
    ),
}

# The approved matrices are registered for calibration-only callers.  The
# ordinary ``generate_snapshot`` entry point remains useful for tiny synthetic
# conformance labels, but refuses every registered T0-T4 decision cell so a
# test or evidence script cannot accidentally materialize a decision fixture.
DECISION_MATRIX: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "T0": {"P-medium": (162000,)},
    "T1": {
        "P-low": (162011, 162012),
        "P-medium": (162013, 162014),
        "P-high": (162015, 162016),
    },
    "T2": {
        "P-low": (162021, 162022),
        "P-medium": (162023, 162024),
        "P-high": (162025, 162026),
    },
    "T3": {"P-low": (162031,), "P-medium": (162032,), "P-high": (162033,)},
    "T4": {"P-medium": (162041,)},
}

CALIBRATION_MATRIX: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "C0": {"P-medium": (168900,)},
    "C1": {
        "P-low": (168911, 168912),
        "P-medium": (168913, 168914),
        "P-high": (168915, 168916),
    },
    "C2": {
        "P-low": (168921, 168922),
        "P-medium": (168923, 168924),
        "P-high": (168925, 168926),
    },
    "C3": {"P-low": (168931,), "P-medium": (168932,), "P-high": (168933,)},
}

TIER_ENTITY_COUNTS = {"T0": 1_000, "T1": 10_000, "T2": 100_000, "T3": 200_000, "T4": 1_000_000}
CALIBRATION_ENTITY_COUNTS = {"C0": 1_000, "C1": 10_000, "C2": 100_000, "C3": 200_000}

ROLE_TYPE = {
    0: "WorkItem",
    1: "Decision",
    2: "Requirement",
    3: "Requirement",
    4: "Implementation",
    5: "Implementation",
    6: "Verification",
    7: "Claim",
    8: "Artifact",
    9: "Source",
}

# (subject_role, predicate, object_role) in exact base-motif construction order.
BASE_MOTIF_EDGES: Tuple[Tuple[int, str, int], ...] = (
    (1, "governs", 2),
    (0, "requires", 2),
    (0, "requires", 3),
    (2, "requires", 4),
    (3, "requires", 5),
    (4, "implements", 2),
    (5, "implements", 3),
    (6, "verifies", 4),
    (8, "verifies", 3),
    (8, "supports", 7),
    (7, "concerns", 0),
    # The Claim-to-literal and Artifact-to-Source records are appended in
    # this order by the block loop below; both are kept out of this tuple so
    # the frozen 13-record construction order is explicit.
)


# ---------------------------------------------------------------------------
# Record types.
# ---------------------------------------------------------------------------


@dataclass
class GeneratedEntity:
    entity_id: str
    entity_type: str
    namespace: str
    canonical_source_identity: List[object]
    display_label_ref: Optional[str] = None
    claim_key: Optional[str] = None


@dataclass
class GeneratedLiteral:
    literal_id: str
    literal_type: str
    value: object


@dataclass
class GeneratedSourceRevision:
    source_revision_id: str
    source_entity_id: str
    source_kind: str
    canonical_locator: str
    object_sha256: str
    object_byte_count: int
    source_event_at: Optional[str]
    observed_at: str
    authority_class: str
    authority_derivation_rule_id: str
    raw_bytes: bytes
    actor_id: Optional[str] = None
    actor_login: Optional[str] = None
    author_association: Optional[str] = None
    event_state: Optional[str] = None
    role_evidence: Optional[object] = None
    metadata: Optional[object] = None
    anchors: Optional[List[object]] = None

    @property
    def locator(self) -> str:
        """Compatibility view of the canonical locator."""
        return self.canonical_locator

    @property
    def byte_count(self) -> int:
        """Compatibility view of the canonical object byte count."""
        return self.object_byte_count


@dataclass
class GeneratedAssertion:
    assertion_id: str
    assertion_ordinal: int
    subject_id: str
    predicate: str
    object_id: Optional[str]
    literal_ref: Optional[str]
    source_revision_id: str
    source_locator: str
    source_event_at: Optional[str]
    asserted_at: str
    valid_from: Optional[str]
    valid_to: Optional[str]
    validity_basis: str
    authority_class: str
    authority_derivation_rule_id: str
    origin: str
    curation_state: str
    review_state: str
    epistemic_state: str
    dispute_state: str
    subject_block: int = field(compare=False, default=-1)  # bookkeeping only

    def preimage(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in ASSERTION_PREIMAGE_KEYS}


@dataclass
class GeneratedSnapshot:
    tier: str
    profile_id: str
    seed: int
    entity_count: int
    block_count: int
    entities: List[GeneratedEntity]
    literals: List[GeneratedLiteral]
    base_source_revisions: List[GeneratedSourceRevision]
    assertions: List[GeneratedAssertion]
    assertion_source_revisions: List[GeneratedSourceRevision]
    cycle_trace: List[Dict[str, object]]
    removed_assertions: List[GeneratedAssertion]
    # Complete construction/selection evidence.  These are intentionally
    # ordinary JSON-compatible values so two implementations can compare the
    # trace without importing one another's classes.
    prf_trace: List[Dict[str, object]] = field(default_factory=list)
    construction_trace: List[Dict[str, object]] = field(default_factory=list)

    @property
    def source_revisions(self) -> List[GeneratedSourceRevision]:
        """Complete physical source revisions, including base revisions."""
        return sorted(
            [*self.base_source_revisions, *self.assertion_source_revisions],
            key=lambda revision: revision.source_revision_id,
        )

    @property
    def generator_name(self) -> str:
        return GENERATOR_NAME

    @property
    def generator_version(self) -> str:
        return GENERATOR_VERSION

    @property
    def prf_domain(self) -> str:
        return PRF_DOMAIN

    @property
    def synthetic_namespace(self) -> str:
        return NAMESPACE

    @property
    def source_locator_version(self) -> str:
        return SOURCE_LOCATOR_VERSION

    @property
    def authority_derivation_rule_id(self) -> str:
        return AUTHORITY_DERIVATION_RULE_ID

    @property
    def assertion_ids(self) -> List[str]:
        return [assertion.assertion_id for assertion in self.assertions]

    @property
    def retry_trace(self) -> List[Dict[str, object]]:
        """Alias exposing the complete PRF choice/retry records."""
        return self.prf_trace

    @property
    def prf_choices(self) -> List[Dict[str, object]]:
        """Alias used by evidence consumers for PRF choice records."""
        return self.prf_trace

    @property
    def source_bytes(self) -> Dict[str, bytes]:
        """Canonical source bytes keyed by their canonical locator."""
        return {
            revision.canonical_locator: revision.raw_bytes
            for revision in physical_source_revisions(self)
        }

    @property
    def records(self) -> Dict[str, List[Dict[str, object]]]:
        """Complete physical canonical records (JSONL-ready projections)."""
        return physical_records(self)

    @property
    def canonical_records(self) -> Dict[str, List[Dict[str, object]]]:
        return self.records

    @property
    def logical_snapshot_checksum(self) -> str:
        """Raw SHA-256 logical semantic checksum (without the ID prefix)."""
        return compute_logical_snapshot_checksum(self)

    @property
    def snapshot_id(self) -> str:
        return f"cyax-snapshot-sha256:{self.logical_snapshot_checksum}"


# ---------------------------------------------------------------------------
# Main generation routine.
# ---------------------------------------------------------------------------


def _locator(tier: str, profile_id: str, seed: int, b: int, suffix: str) -> str:
    return f"cyax://0168{SOURCE_LOCATOR_VERSION}{tier}/{profile_id}/{seed}/block/{b}/{suffix}"


def generate_snapshot(tier: str, profile_id: str, seed: int, entity_count: int) -> GeneratedSnapshot:
    if profile_id not in PROFILES:
        raise ValueError(f"unknown profile: {profile_id!r}")
    if tier in DECISION_MATRIX:
        raise GenerationFailure("decision fixtures T0-T4 are outside the independent calibration-only generator")
    if entity_count <= 0 or entity_count % 10 != 0:
        raise GenerationFailure("entity_count must be a positive multiple of ten")
    profile = PROFILES[profile_id]
    B = entity_count // 10

    # --- Entities, literals, and base source revisions for every block ---
    # ``prf_trace`` and ``construction_trace`` are append-only evidence logs.
    # Construction trace entries describe every identity-bearing phase choice;
    # no diagnostic or host value enters an Entity/Assertion identity.
    prf_trace: List[Dict[str, object]] = []
    construction_trace: List[Dict[str, object]] = []
    entities: Dict[Tuple[int, int], GeneratedEntity] = {}
    workitem_id: Dict[int, str] = {}
    claim_id: Dict[int, str] = {}
    literals: List[GeneratedLiteral] = []
    base_revisions: List[GeneratedSourceRevision] = []
    source_entity_of_block: Dict[int, str] = {}

    for b in range(B):
        for r in range(10):
            canonical_identity = [tier, profile_id, seed, b, r]
            eid = entity_id(NAMESPACE, ROLE_TYPE[r], canonical_identity)
            claim_key = "synthetic.block_claim" if r == 7 else None
            entities[(b, r)] = GeneratedEntity(
                entity_id=eid,
                entity_type=ROLE_TYPE[r],
                namespace=NAMESPACE,
                canonical_source_identity=canonical_identity,
                display_label_ref=None,
                claim_key=claim_key,
            )
            if r == 0:
                workitem_id[b] = eid
            if r == 7:
                claim_id[b] = eid
            if r == 9:
                source_entity_of_block[b] = eid

        literal_value = f"block={b};profile={profile_id};seed={seed}"
        lit_id = literal_id("text", literal_value)
        literals.append(GeneratedLiteral(literal_id=lit_id, literal_type="text", value=literal_value))

        base_obj = {"block": b, "generator": GENERATOR_NAME, "profile": profile_id, "seed": seed, "tier": tier}
        base_bytes = canonical_json_bytes(base_obj)
        base_sha = sha256_hex(base_bytes)
        base_locator = _locator(tier, profile_id, seed, b, "base")
        base_rev_id = source_revision_id("synthetic_fixture", base_locator, base_sha, None)
        base_revisions.append(
            GeneratedSourceRevision(
                source_revision_id=base_rev_id,
                source_entity_id=source_entity_of_block[b],
                source_kind="synthetic_fixture",
                canonical_locator=base_locator,
                object_sha256=base_sha,
                object_byte_count=len(base_bytes),
                source_event_at=None,
                observed_at=FIXED_TIME,
                authority_class="ordinary_record",
                authority_derivation_rule_id=AUTHORITY_DERIVATION_RULE_ID,
                raw_bytes=base_bytes,
                anchors=[],
            )
        )

    block_literal: Dict[int, GeneratedLiteral] = {b: literals[b] for b in range(B)}

    # --- ordinal counter and assertion construction helper ---
    ordinal_counter = {"a": 0}
    all_constructed: List[GeneratedAssertion] = []
    assertion_revisions: Dict[str, GeneratedSourceRevision] = {}

    def make_assertion(
        subject_id: str,
        predicate: str,
        object_id: Optional[str],
        literal_ref: Optional[str],
        subject_block: int,
        valid_from: Optional[str],
        validity_basis: str,
        dispute_state: str,
        phase: str = "unknown",
    ) -> GeneratedAssertion:
        a = ordinal_counter["a"]
        ordinal_counter["a"] += 1
        locator = _locator(tier, profile_id, seed, subject_block, f"assertion/{a}")
        body = {
            "assertion_ordinal": a,
            "generator": GENERATOR_NAME,
            "literal_identity": literal_ref,
            "object_identity": object_id,
            "predicate": predicate,
            "profile": profile_id,
            "seed": seed,
            "subject_identity": subject_id,
            "tier": tier,
        }
        raw = canonical_json_bytes(body)
        obj_sha = sha256_hex(raw)
        rev_id = source_revision_id("synthetic_fixture", locator, obj_sha, None)
        preimage = {
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "literal_ref": literal_ref,
            "source_revision_id": rev_id,
            "source_locator": locator,
            "source_event_at": None,
            "asserted_at": FIXED_TIME,
            "valid_from": valid_from,
            "valid_to": None,
            "validity_basis": validity_basis,
            "authority_class": "ordinary_record",
            "authority_derivation_rule_id": AUTHORITY_DERIVATION_RULE_ID,
            "origin": "source_direct",
            "curation_state": "independently_reviewed",
            "review_state": "not_required",
            "epistemic_state": "supported",
            "dispute_state": dispute_state,
        }
        aid = assertion_id(preimage)
        rev = GeneratedSourceRevision(
            source_revision_id=rev_id,
            source_entity_id=source_entity_of_block[subject_block],
            source_kind="synthetic_fixture",
            canonical_locator=locator,
            object_sha256=obj_sha,
            object_byte_count=len(raw),
            source_event_at=None,
            observed_at=FIXED_TIME,
            authority_class="ordinary_record",
            authority_derivation_rule_id=AUTHORITY_DERIVATION_RULE_ID,
            raw_bytes=raw,
            anchors=["json-object"],
        )
        assertion = GeneratedAssertion(
            assertion_id=aid,
            assertion_ordinal=a,
            subject_block=subject_block,
            **preimage,
        )
        assertion_revisions[aid] = rev
        construction_trace.append(
            {
                "event": "assertion_constructed",
                "phase": phase,
                "assertion_ordinal": a,
                "assertion_id": aid,
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "literal_ref": literal_ref,
            }
        )
        return assertion

    existing_triples: set = set()
    base_motif_assertions: List[GeneratedAssertion] = []

    for b in range(B):
        for subj_role, predicate, obj_role in BASE_MOTIF_EDGES:
            subj_id = entities[(b, subj_role)].entity_id
            obj_id = entities[(b, obj_role)].entity_id
            asrt = make_assertion(subj_id, predicate, obj_id, None, b, None, "unknown", "undisputed", "base_motif")
            all_constructed.append(asrt)
            base_motif_assertions.append(asrt)
            existing_triples.add((subj_id, predicate, obj_id))
        # Claim documented_in <block literal>
        claim_subj = entities[(b, 7)].entity_id
        lit_ref = block_literal[b].literal_id
        asrt = make_assertion(claim_subj, "documented_in", None, lit_ref, b, None, "unknown", "undisputed", "base_motif")
        all_constructed.append(asrt)
        base_motif_assertions.append(asrt)
        # Artifact documented_in Source is the final base-motif assertion.
        artifact_subj = entities[(b, 8)].entity_id
        source_obj = entities[(b, 9)].entity_id
        asrt = make_assertion(artifact_subj, "documented_in", source_obj, None, b, None, "unknown", "undisputed", "base_motif")
        all_constructed.append(asrt)
        base_motif_assertions.append(asrt)

    # --- Phase 1: isolation ---
    blocks_by_id_order = sorted(range(B), key=lambda b: workitem_id[b])
    isolated_count = math.floor(profile.isolated_rate * B)
    isolated_blocks = set(blocks_by_id_order[B - isolated_count :]) if isolated_count > 0 else set()
    connected_order = [b for b in blocks_by_id_order if b not in isolated_blocks]
    connected_blocks_count = len(connected_order)
    construction_trace.append(
        {
            "event": "phase_complete",
            "phase": "phase-1-isolation",
            "block_primary_order": list(blocks_by_id_order),
            "isolated_block_ordinals": sorted(isolated_blocks),
            "connected_block_ordinals": list(connected_order),
        }
    )

    # --- Phase 2: components + fixed chain ---
    comp_size = max(profile.path_depth + 1, 64)
    components: List[List[int]] = [
        connected_order[i : i + comp_size] for i in range(0, connected_blocks_count, comp_size)
    ]
    component_of_block: Dict[int, int] = {}
    for ci, comp in enumerate(components):
        for blk in comp:
            component_of_block[blk] = ci
    construction_trace.append(
        {
            "event": "phase_complete",
            "phase": "phase-2-components",
            "component_capacity": comp_size,
            "components": [list(component) for component in components],
        }
    )

    non_chain_out: Dict[str, List[GeneratedAssertion]] = {}

    fixed_chain_edges = 0
    for comp in components:
        chain_len = min(profile.path_depth, len(comp) - 1)
        fixed_chain_edges += chain_len
        for i in range(1, chain_len + 1):
            subj_b = comp[i]
            obj_b = comp[i - 1]
            subj_id = workitem_id[subj_b]
            obj_id = workitem_id[obj_b]
            asrt = make_assertion(subj_id, "depends_on", obj_id, None, subj_b, None, "unknown", "undisputed", "phase-2-components")
            all_constructed.append(asrt)
            existing_triples.add((subj_id, "depends_on", obj_id))
            # chain edges are intentionally excluded from non_chain_out

    # frozen per-block candidate vectors
    same_component_candidates: Dict[int, List[str]] = {}
    cross_component_candidates: Dict[int, List[str]] = {}
    for comp in components:
        comp_ids = [workitem_id[x] for x in comp]
        for blk in comp:
            same_component_candidates[blk] = [wid for wid in comp_ids if wid != workitem_id[blk]]
    for blk in connected_order:
        ci = component_of_block[blk]
        cross_component_candidates[blk] = [
            workitem_id[x] for x in connected_order if component_of_block[x] != ci
        ]

    # --- Phase 3: additional dependency edges ---
    dependency_quota = min(profile.dependency_fanout * connected_blocks_count, 30 * B)
    additional_dependency_positions = dependency_quota - fixed_chain_edges
    if additional_dependency_positions < 0:
        raise GenerationFailure("fixed dependency chain exceeds dependency quota")
    cross_link_count = math.floor(profile.cross_link_rate * dependency_quota)
    if cross_link_count > additional_dependency_positions:
        raise GenerationFailure("cross-link count exceeds additional dependency positions")

    construction_trace.append(
        {
            "event": "phase_start",
            "phase": "phase-3-dependencies",
            "dependency_quota": dependency_quota,
            "fixed_chain_edges": fixed_chain_edges,
            "additional_dependency_positions": additional_dependency_positions,
            "cross_link_count": cross_link_count,
            "connected_block_ordinals": list(connected_order),
        }
    )

    for i in range(additional_dependency_positions):
        if connected_blocks_count == 0:
            raise GenerationFailure("no connected blocks available for dependency phase")
        src_b = connected_order[i % connected_blocks_count]
        subj_id = workitem_id[src_b]
        if i < cross_link_count:
            purpose = "cross_component_dependency_target"
            candidates = cross_component_candidates[src_b]
        else:
            purpose = "dependency_target"
            candidates = same_component_candidates[src_b]

        target_id = prf_select(
            seed,
            profile_id,
            purpose,
            i,
            candidates,
            sort_key=lambda x: x,
            rejects=lambda cand, _s=subj_id: dependency_target_rejects(_s, cand, existing_triples),
            candidates_sorted=True,
            trace=prf_trace,
            phase="phase-3-dependencies",
            subject_id=subj_id,
        )
        asrt = make_assertion(subj_id, "depends_on", target_id, None, src_b, None, "unknown", "undisputed", "phase-3-dependencies")
        all_constructed.append(asrt)
        existing_triples.add((subj_id, "depends_on", target_id))
        non_chain_out.setdefault(subj_id, []).append(asrt)

    # --- Phase 4: cycles (no PRF) ---
    removed_ids: set = set()
    removed_assertions: List[GeneratedAssertion] = []
    cycle_trace: List[Dict[str, object]] = []
    cycle_member_ids: set = set()
    workitem_block_of: Dict[str, int] = {workitem_id[b]: b for b in range(B)}

    cycle_quota = math.floor(profile.cycle_rate * connected_blocks_count)
    construction_trace.append(
        {
            "event": "phase_start",
            "phase": "phase-4-cycles",
            "cycle_quota": cycle_quota,
            "cycle_selection": "consecutive_primary_id_windows_lowest_id_disjoint",
            "uses_prf": False,
        }
    )
    if cycle_quota > 0:
        selected: List[Tuple[str, str, str]] = []
        used: set = set()
        for i in range(0, max(0, connected_blocks_count - 2)):
            if len(selected) >= cycle_quota:
                break
            a = workitem_id[connected_order[i]]
            bb = workitem_id[connected_order[i + 1]]
            c = workitem_id[connected_order[i + 2]]
            if a in used or bb in used or c in used:
                continue
            members_ok = all(len(non_chain_out.get(m, [])) >= 1 for m in (a, bb, c))
            pairs_ok = (
                (a, "depends_on", bb) not in existing_triples
                and (bb, "depends_on", c) not in existing_triples
                and (c, "depends_on", a) not in existing_triples
            )
            if members_ok and pairs_ok:
                selected.append((a, bb, c))
                used.update((a, bb, c))
        if len(selected) < cycle_quota:
            raise GenerationFailure("could not satisfy cycle quota with disjoint eligible triples")

        for a, bb, c in selected:
            removed_for_triple = []
            for member in (a, bb, c):
                candidates_list = non_chain_out[member]
                smallest = min(candidates_list, key=lambda x: (x.assertion_id, x.subject_id, x.object_id))
                candidates_list.remove(smallest)
                existing_triples.discard((smallest.subject_id, smallest.predicate, smallest.object_id))
                removed_ids.add(smallest.assertion_id)
                removed_assertions.append(smallest)
                removed_for_triple.append(smallest.assertion_id)

            added_for_triple = []
            for s, o in ((a, bb), (bb, c), (c, a)):
                sb = workitem_block_of[s]
                asrt = make_assertion(s, "depends_on", o, None, sb, None, "unknown", "undisputed", "phase-4-cycles")
                all_constructed.append(asrt)
                existing_triples.add((s, "depends_on", o))
                added_for_triple.append(asrt.assertion_id)

            cycle_member_ids.update((a, bb, c))
            cycle_trace.append(
                {
                    "triple": [a, bb, c],
                    "removed_assertion_ids": removed_for_triple,
                    "added_assertion_ids": added_for_triple,
                }
            )
            construction_trace.append(
                {
                    "event": "cycle_rewrite",
                    "phase": "phase-4-cycles",
                    "triple": [a, bb, c],
                    "removed_assertion_ids": list(removed_for_triple),
                    "added_assertion_ids": list(added_for_triple),
                }
            )

    # --- Phase 5: supersession chains ---
    eligible_workitems = sorted(
        (workitem_id[b] for b in range(B) if b not in isolated_blocks)
    )
    depth = profile.supersession_depth
    num_chains = len(eligible_workitems) // depth if depth > 0 else 0
    construction_trace.append(
        {
            "event": "phase_start",
            "phase": "phase-5-supersession",
            "eligible_workitem_ids": list(eligible_workitems),
            "supersession_depth": depth,
            "chain_count": num_chains,
            "isolated_excluded": sorted(isolated_blocks),
        }
    )
    for ci in range(num_chains):
        chain = eligible_workitems[ci * depth : (ci + 1) * depth]
        for pos in range(1, depth):
            subj_id = chain[pos]
            obj_id = chain[pos - 1]
            subj_b = workitem_block_of[subj_id]
            valid_from = f"2000-01-02T00:00:{pos:02d}.000000Z"
            asrt = make_assertion(subj_id, "supersedes", obj_id, None, subj_b, valid_from, "explicit", "undisputed", "phase-5-supersession")
            all_constructed.append(asrt)
            existing_triples.add((subj_id, "supersedes", obj_id))

    # --- Phase 6: disputes ---
    claims_sorted = sorted(claim_id[b] for b in range(B))
    claim_block_of: Dict[str, int] = {claim_id[b]: b for b in range(B)}
    pair_count = 2 * math.floor(profile.dispute_rate * B)
    disputing = claims_sorted[:pair_count]
    construction_trace.append(
        {
            "event": "phase_start",
            "phase": "phase-6-disputes",
            "claim_pair_count": pair_count // 2,
            "selected_claim_ids": list(disputing),
        }
    )
    for i in range(0, len(disputing), 2):
        c0 = disputing[i]
        c1 = disputing[i + 1]
        subj_b = claim_block_of[c1]
        asrt = make_assertion(c1, "contradicts", c0, None, subj_b, None, "unknown", "disputed", "phase-6-disputes")
        all_constructed.append(asrt)
        existing_triples.add((c1, "contradicts", c0))

    # --- Phase 7: parallel evidence ---
    base_sorted = sorted(base_motif_assertions, key=lambda x: x.assertion_id)
    parallel_count = math.floor(profile.parallel_evidence_rate * 13 * B)
    construction_trace.append(
        {
            "event": "phase_start",
            "phase": "phase-7-parallel-evidence",
            "parallel_count": parallel_count,
            "selected_base_assertion_ids": [a.assertion_id for a in base_sorted[:parallel_count]],
        }
    )
    for orig in base_sorted[:parallel_count]:
        asrt = make_assertion(
            orig.subject_id,
            orig.predicate,
            orig.object_id,
            orig.literal_ref,
            orig.subject_block,
            orig.valid_from,
            orig.validity_basis,
            orig.dispute_state,
            "phase-7-parallel-evidence",
        )
        all_constructed.append(asrt)

    # --- Phase 8: fill concerns to exactly 50*B ---
    pair_vector: List[Tuple[str, str, int]] = []
    for b in range(B):
        block_entities = [entities[(b, r)].entity_id for r in range(9)]  # roles 0..8, Source excluded
        for si in range(9):
            for oi in range(9):
                if si == oi:
                    continue
                pair_vector.append((block_entities[si], block_entities[oi], b))

    surviving_count = len(all_constructed) - len(removed_ids)
    target_total = 50 * B
    remaining = target_total - surviving_count
    if remaining < 0:
        raise GenerationFailure("assertion budget already exceeds 50*B before fill phase")

    def pair_sort_key(pair: Tuple[str, str, int]) -> bytes:
        return frame([pair[0], pair[1]])

    pair_vector.sort(key=pair_sort_key)

    construction_trace.append(
        {
            "event": "phase_start",
            "phase": "phase-8-filler",
            "target_assertion_count": target_total,
            "surviving_count_before_fill": surviving_count,
            "remaining": remaining,
            "candidate_pair_count": len(pair_vector),
            "candidate_pairs": [[s, o, b] for s, o, b in pair_vector],
        }
    )

    for p in range(remaining):
        chosen = prf_select(
            seed,
            profile_id,
            "fill_concerns_pair",
            p,
            pair_vector,
            sort_key=pair_sort_key,
            rejects=lambda cand: fill_concerns_pair_rejects(cand[0], cand[1], existing_triples),
            candidates_sorted=True,
            trace=prf_trace,
            phase="phase-8-filler",
        )
        s, o, b = chosen
        asrt = make_assertion(s, "concerns", o, None, b, None, "unknown", "undisputed", "phase-8-filler")
        all_constructed.append(asrt)
        existing_triples.add((s, "concerns", o))

    final_assertions = [x for x in all_constructed if x.assertion_id not in removed_ids]
    if len(final_assertions) != 50 * B:
        raise GenerationFailure(
            f"final assertion count {len(final_assertions)} != {50 * B} (invariant violated)"
        )

    final_revisions = [assertion_revisions[x.assertion_id] for x in final_assertions]

    return GeneratedSnapshot(
        tier=tier,
        profile_id=profile_id,
        seed=seed,
        entity_count=entity_count,
        block_count=B,
        entities=sorted(entities.values(), key=lambda e: e.entity_id),
        literals=sorted(literals, key=lambda l: l.literal_id),
        base_source_revisions=sorted(base_revisions, key=lambda r: r.source_revision_id),
        assertions=sorted(final_assertions, key=lambda a: a.assertion_id),
        assertion_source_revisions=sorted(final_revisions, key=lambda r: r.source_revision_id),
        cycle_trace=cycle_trace,
        removed_assertions=removed_assertions,
        prf_trace=prf_trace,
        construction_trace=construction_trace,
    )


def generate_calibration_snapshot(tier: str, profile_id: str, seed: int) -> GeneratedSnapshot:
    """Materialize one exact C0-C3 calibration cell.

    Decision cells are intentionally not accepted by this entry point.  The
    dimensions and seed membership are checked against the frozen calibration
    matrix before generation, which prevents a caller from silently using a
    calibration-shaped fixture with a decision seed.
    """
    if tier not in CALIBRATION_MATRIX or seed not in CALIBRATION_MATRIX[tier].get(profile_id, ()):
        raise GenerationFailure(f"not an approved calibration cell: {tier}/{profile_id}/{seed}")
    return generate_snapshot(tier, profile_id, seed, CALIBRATION_ENTITY_COUNTS[tier])


# ---------------------------------------------------------------------------
# Logical snapshot checksum (spec.md "Immutable assertion snapshot and
# identity"); see module docstring for the flagged outer-wrapper assumptions.
# ---------------------------------------------------------------------------


def _entity_record(e: GeneratedEntity) -> Dict[str, object]:
    rec: Dict[str, object] = {
        "entity_id": e.entity_id,
        "entity_type": e.entity_type,
        "namespace": e.namespace,
        "canonical_source_identity": list(e.canonical_source_identity),
        "display_label_ref": e.display_label_ref,
    }
    if e.entity_type == "Claim":
        rec["claim_key"] = e.claim_key
    return rec


def _semantic_entity_record(e: GeneratedEntity) -> Dict[str, object]:
    rec = _entity_record(e)
    rec.pop("display_label_ref", None)
    return rec


def _literal_record(l: GeneratedLiteral) -> Dict[str, object]:
    return {"literal_id": l.literal_id, "literal_type": l.literal_type, "value": l.value}


def _source_revision_record(r: GeneratedSourceRevision) -> Dict[str, object]:
    """Return the frozen common SourceRevision JSON record.

    ``GeneratedSourceRevision`` retains raw bytes and generator-only evidence
    for physical validation, but the canonical record is exactly the common
    snapshot schema.  In particular, the locator and byte-count field names
    are ``canonical_locator`` and ``object_byte_count``.
    """
    return {
        "source_revision_id": r.source_revision_id,
        "source_kind": r.source_kind,
        "source_entity_id": r.source_entity_id,
        "canonical_locator": r.canonical_locator,
        "object_sha256": r.object_sha256,
        "object_byte_count": r.object_byte_count,
        "source_event_at": r.source_event_at,
        "observed_at": r.observed_at,
        "actor_id": r.actor_id,
        "authority_class": r.authority_class,
        "authority_derivation_rule_id": r.authority_derivation_rule_id,
    }


def _assertion_record(a: GeneratedAssertion) -> Dict[str, object]:
    return {"assertion_id": a.assertion_id, **a.preimage()}


def semantic_projection(snapshot: GeneratedSnapshot) -> Dict[str, List[Dict[str, object]]]:
    entities = sorted((_semantic_entity_record(e) for e in snapshot.entities), key=lambda r: r["entity_id"])
    literals = sorted((_literal_record(l) for l in snapshot.literals), key=lambda r: r["literal_id"])
    revisions = sorted(
        (_source_revision_record(r) for r in snapshot.assertion_source_revisions),
        key=lambda r: r["source_revision_id"],
    )
    assertions = sorted(
        (_assertion_record(a) for a in snapshot.assertions), key=lambda r: r["assertion_id"]
    )
    return {
        "semantic_entities": entities,
        "semantic_literals": literals,
        "semantic_source_revisions": revisions,
        "semantic_assertions": assertions,
    }


def semantic_projection_checksum(snapshot: GeneratedSnapshot) -> str:
    """Generator-scoped content-stability checksum with no external version
    constants; unambiguous and fully reproducible from spec.md's generator
    2.3 section alone."""
    proj = semantic_projection(snapshot)
    payload = frame(
        [
            "cyax-0168-generator-semantic-projection-v1",
            ["semantic_entities", proj["semantic_entities"]],
            ["semantic_literals", proj["semantic_literals"]],
            ["semantic_source_revisions", proj["semantic_source_revisions"]],
            ["semantic_assertions", proj["semantic_assertions"]],
        ]
    )
    return sha256_hex(payload)


def physical_source_revisions(snapshot: GeneratedSnapshot) -> List[GeneratedSourceRevision]:
    """Return the complete physical source-revision set in primary-ID order.

    The base revision for every generated Source remains physical provenance;
    assertion revisions are added alongside it.  The semantic projection
    below intentionally filters back to only assertion-referenced revisions.
    """
    return sorted(
        [*snapshot.base_source_revisions, *snapshot.assertion_source_revisions],
        key=lambda revision: revision.source_revision_id,
    )


def source_objects(snapshot: GeneratedSnapshot, *, semantic_only: bool = False) -> Dict[str, bytes]:
    """Return the content-addressed synthetic source objects by digest."""
    revisions = snapshot.assertion_source_revisions if semantic_only else physical_source_revisions(snapshot)
    objects: Dict[str, bytes] = {}
    for revision in revisions:
        if sha256_hex(revision.raw_bytes) != revision.object_sha256:
            raise GenerationFailure("source object bytes do not match object_sha256")
        if len(revision.raw_bytes) != revision.object_byte_count:
            raise GenerationFailure("source object bytes do not match object_byte_count")
        previous = objects.get(revision.object_sha256)
        if previous is not None and previous != revision.raw_bytes:
            raise GenerationFailure("source object digest collision with unequal bytes")
        objects[revision.object_sha256] = revision.raw_bytes
    return objects


def canonical_source_bytes(snapshot: GeneratedSnapshot) -> Dict[str, bytes]:
    """Return every generated source object keyed by canonical locator.

    The mapping includes the one base object per block and the provenance
    object for every surviving assertion.  It is the lossless byte-level
    source view; :func:`source_objects` is the content-addressed digest view.
    """
    return snapshot.source_bytes


def source_records(snapshot: GeneratedSnapshot) -> List[Dict[str, object]]:
    """Return complete physical SourceRevision records in primary-ID order."""
    return physical_records(snapshot)["source_revisions"]


def assertion_records(snapshot: GeneratedSnapshot) -> List[Dict[str, object]]:
    return physical_records(snapshot)["assertions"]


def entity_records(snapshot: GeneratedSnapshot) -> List[Dict[str, object]]:
    return physical_records(snapshot)["entities"]


def literal_records(snapshot: GeneratedSnapshot) -> List[Dict[str, object]]:
    return physical_records(snapshot)["literals"]


def complete_output(snapshot: GeneratedSnapshot) -> Dict[str, object]:
    """Package the complete independent reproduction evidence.

    ``source_bytes`` is intentionally separate from JSON records because raw
    objects are content-addressed payloads rather than fields of a
    SourceRevision record.  All lists are already in their frozen primary-ID
    order; construction and PRF traces retain construction order.
    """
    return {
        "generator_name": snapshot.generator_name,
        "generator_version": snapshot.generator_version,
        "prf_domain": snapshot.prf_domain,
        "synthetic_namespace": snapshot.synthetic_namespace,
        "source_locator_version": snapshot.source_locator_version,
        "authority_derivation_rule_id": snapshot.authority_derivation_rule_id,
        "tier": snapshot.tier,
        "profile_id": snapshot.profile_id,
        "seed": snapshot.seed,
        "entities": entity_records(snapshot),
        "literals": literal_records(snapshot),
        "source_revisions": source_records(snapshot),
        "assertions": assertion_records(snapshot),
        "source_bytes": snapshot.source_bytes,
        "construction_trace": snapshot.construction_trace,
        "prf_choices": snapshot.prf_trace,
        "assertion_ids": snapshot.assertion_ids,
        "logical_snapshot_checksum": snapshot.logical_snapshot_checksum,
        "snapshot_id": snapshot.snapshot_id,
    }


def physical_records(snapshot: GeneratedSnapshot) -> Dict[str, List[Dict[str, object]]]:
    """Return complete canonical record projections for the physical view.

    Unlike the logical projection, this retains nullable display references,
    unreferenced base source revisions, and all physical source records. Raw
    source objects remain available through each revision's ``raw_bytes`` and
    are intentionally not duplicated inside JSONL-style revision records.
    """
    return {
        "entities": sorted((_entity_record(e) for e in snapshot.entities), key=lambda r: r["entity_id"]),
        "literals": sorted((_literal_record(l) for l in snapshot.literals), key=lambda r: r["literal_id"]),
        "source_revisions": [
            _source_revision_record(r) for r in physical_source_revisions(snapshot)
        ],
        "assertions": sorted(
            (_assertion_record(a) for a in snapshot.assertions), key=lambda r: r["assertion_id"]
        ),
    }


def compute_semantic_source_bundle_projection_checksum(snapshot: GeneratedSnapshot) -> str:
    """Hash the reachable synthetic source-bundle projection.

    Generator assertions reach their assertion-specific revisions and their
    captured objects.  Synthetic fixtures have no actor/role payloads or
    owner-decision registry entries, so those two ordered sections are empty.
    """
    projection = semantic_projection(snapshot)
    reachable_objects = source_objects(snapshot, semantic_only=True)
    ordered_objects = sorted(
        [[digest, len(raw_bytes)] for digest, raw_bytes in reachable_objects.items()],
        key=lambda pair: pair[0],
    )
    payload = frame(
        [
            "cyax-source-bundle-projection-v1",
            projection["semantic_source_revisions"],
            ordered_objects,
            [],
            [],
        ]
    )
    return sha256_hex(payload)


def compute_logical_snapshot_checksum(
    snapshot: GeneratedSnapshot,
    *,
    schema_version: str = "cyax-snapshot-v1",
    authority_rule_version: str = "cyax-authority-v1",
    semantic_evaluator_rule_version: str = "cyax-evaluator-v1",
    semantic_source_bundle_projection_checksum: Optional[str] = None,
) -> str:
    """Return the raw logical semantic SHA-256 checksum.

    The public ``snapshot_id`` is the separate rendered form
    ``cyax-snapshot-sha256:<checksum>``.
    """
    proj = semantic_projection(snapshot)
    if semantic_source_bundle_projection_checksum is None:
        semantic_source_bundle_projection_checksum = compute_semantic_source_bundle_projection_checksum(snapshot)
    payload = frame(
        [
            "cyax-logical-snapshot-v1",
            schema_version,
            semantic_source_bundle_projection_checksum,
            authority_rule_version,
            semantic_evaluator_rule_version,
            ["semantic_entities", proj["semantic_entities"]],
            ["semantic_literals", proj["semantic_literals"]],
            ["semantic_source_revisions", proj["semantic_source_revisions"]],
            ["semantic_assertions", proj["semantic_assertions"]],
        ]
    )
    digest = sha256_hex(payload)
    return digest
