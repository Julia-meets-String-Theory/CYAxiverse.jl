"""Independent reproduction of CYAX-0168 F-scale generator version 2.2.

This module is Worker D's independent implementation for CYAX-0168 G1 (see
``specs/0168-structured-graph-materialization/spec.md``, section "F-scale
generator"). It was written from the specification text only, without
opening, importing, or otherwise consulting ``generator_primary.py`` (the
sibling independent implementation produced by another worker) or
``common.py``. All canonical framing, ID, and canonical-JSON primitives
below are re-derived locally from spec.md rather than reused.

Only small synthetic conformance parameters are exercised by the paired test
module (``tests/test_generator_independent.py``); this module never accesses
or materializes a real T0-T4/C0-C3 decision or calibration fixture.

Assumptions made where spec.md's generator-2.2 section itself is silent
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
  generator 2.2's synthetic-only scope (generator-owned bytes ARE their own
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

GENERATOR_VERSION = "cyax-0168-scale-2.2"
NAMESPACE = "cyax-0168-synthetic-v2.2"
PURPOSE_TOKENS = ("dependency_target", "cross_component_dependency_target", "fill_concerns_pair")

FIXED_TIME = "2000-01-01T00:00:00.000000Z"
MAX_COUNTER = 2**64 - 1


class GenerationFailure(Exception):
    """Raised whenever generator 2.2's own closed failure rules trigger."""


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
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (seed, ordinal, counter)
    ):
        raise ValueError("seed, ordinal, and counter must be nonnegative integers")
    return sha256_bytes(frame(["cyax-gen-2.2", seed, profile_id, purpose, ordinal, counter]))


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
) -> object:
    """Generic implementation of the closed generator-2.2 selection rule.

    ``rejects`` must implement exactly the purpose row's exhaustive
    post-PRF rejection predicates and nothing else.
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
    if n == 1:
        cand = sorted_candidates[0]
        if rejects(cand):
            raise GenerationFailure(
                f"sole candidate rejected without retry for purpose={purpose} ordinal={ordinal}"
            )
        return cand

    counter = 0
    while True:
        digest = prf_digest(seed, profile_id, purpose, ordinal, counter)
        x = int.from_bytes(digest, "big")
        limit = (2**256 // n) * n
        if x >= limit:
            counter += 1
            if counter > MAX_COUNTER:
                raise GenerationFailure("counter overflow during digest rejection")
            continue
        cand = sorted_candidates[x % n]
        if rejects(cand):
            counter += 1
            if counter > MAX_COUNTER:
                raise GenerationFailure("counter overflow during predicate rejection")
            continue
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

    @property
    def source_revisions(self) -> List[GeneratedSourceRevision]:
        """Complete physical source revisions, including base revisions."""
        return sorted(
            [*self.base_source_revisions, *self.assertion_source_revisions],
            key=lambda revision: revision.source_revision_id,
        )


# ---------------------------------------------------------------------------
# Main generation routine.
# ---------------------------------------------------------------------------


def _locator(tier: str, profile_id: str, seed: int, b: int, suffix: str) -> str:
    return f"cyax://0168/scale/2.2/{tier}/{profile_id}/{seed}/block/{b}/{suffix}"


def generate_snapshot(tier: str, profile_id: str, seed: int, entity_count: int) -> GeneratedSnapshot:
    if profile_id not in PROFILES:
        raise ValueError(f"unknown profile: {profile_id!r}")
    if entity_count <= 0 or entity_count % 10 != 0:
        raise GenerationFailure("entity_count must be a positive multiple of ten")
    profile = PROFILES[profile_id]
    B = entity_count // 10

    # --- Entities, literals, and base source revisions for every block ---
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

        base_obj = {"block": b, "generator": GENERATOR_VERSION, "profile": profile_id, "seed": seed, "tier": tier}
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
                authority_derivation_rule_id="synthetic_fixture_v2.2",
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
    ) -> GeneratedAssertion:
        a = ordinal_counter["a"]
        ordinal_counter["a"] += 1
        locator = _locator(tier, profile_id, seed, subject_block, f"assertion/{a}")
        body = {
            "assertion_ordinal": a,
            "generator": GENERATOR_VERSION,
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
            "authority_derivation_rule_id": "synthetic_fixture_v2.2",
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
            authority_derivation_rule_id="synthetic_fixture_v2.2",
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
        return assertion

    existing_triples: set = set()
    base_motif_assertions: List[GeneratedAssertion] = []

    for b in range(B):
        for subj_role, predicate, obj_role in BASE_MOTIF_EDGES:
            subj_id = entities[(b, subj_role)].entity_id
            obj_id = entities[(b, obj_role)].entity_id
            asrt = make_assertion(subj_id, predicate, obj_id, None, b, None, "unknown", "undisputed")
            all_constructed.append(asrt)
            base_motif_assertions.append(asrt)
            existing_triples.add((subj_id, predicate, obj_id))
        # Claim documented_in <block literal>
        claim_subj = entities[(b, 7)].entity_id
        lit_ref = block_literal[b].literal_id
        asrt = make_assertion(claim_subj, "documented_in", None, lit_ref, b, None, "unknown", "undisputed")
        all_constructed.append(asrt)
        base_motif_assertions.append(asrt)
        # Artifact documented_in Source is the final base-motif assertion.
        artifact_subj = entities[(b, 8)].entity_id
        source_obj = entities[(b, 9)].entity_id
        asrt = make_assertion(artifact_subj, "documented_in", source_obj, None, b, None, "unknown", "undisputed")
        all_constructed.append(asrt)
        base_motif_assertions.append(asrt)

    # --- Phase 1: isolation ---
    blocks_by_id_order = sorted(range(B), key=lambda b: workitem_id[b])
    isolated_count = math.floor(profile.isolated_rate * B)
    isolated_blocks = set(blocks_by_id_order[B - isolated_count :]) if isolated_count > 0 else set()
    connected_order = [b for b in blocks_by_id_order if b not in isolated_blocks]
    connected_blocks_count = len(connected_order)

    # --- Phase 2: components + fixed chain ---
    comp_size = max(profile.path_depth + 1, 64)
    components: List[List[int]] = [
        connected_order[i : i + comp_size] for i in range(0, connected_blocks_count, comp_size)
    ]
    component_of_block: Dict[int, int] = {}
    for ci, comp in enumerate(components):
        for blk in comp:
            component_of_block[blk] = ci

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
            asrt = make_assertion(subj_id, "depends_on", obj_id, None, subj_b, None, "unknown", "undisputed")
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
        )
        asrt = make_assertion(subj_id, "depends_on", target_id, None, src_b, None, "unknown", "undisputed")
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
                asrt = make_assertion(s, "depends_on", o, None, sb, None, "unknown", "undisputed")
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

    # --- Phase 5: supersession chains ---
    eligible_workitems = sorted(
        (workitem_id[b] for b in range(B) if workitem_id[b] not in cycle_member_ids)
    )
    depth = profile.supersession_depth
    num_chains = len(eligible_workitems) // depth if depth > 0 else 0
    for ci in range(num_chains):
        chain = eligible_workitems[ci * depth : (ci + 1) * depth]
        for pos in range(1, depth):
            subj_id = chain[pos]
            obj_id = chain[pos - 1]
            subj_b = workitem_block_of[subj_id]
            valid_from = f"2000-01-02T00:00:{pos:02d}.000000Z"
            asrt = make_assertion(subj_id, "supersedes", obj_id, None, subj_b, valid_from, "explicit", "undisputed")
            all_constructed.append(asrt)
            existing_triples.add((subj_id, "supersedes", obj_id))

    # --- Phase 6: disputes ---
    claims_sorted = sorted(claim_id[b] for b in range(B))
    claim_block_of: Dict[str, int] = {claim_id[b]: b for b in range(B)}
    pair_count = 2 * math.floor(profile.dispute_rate * B)
    disputing = claims_sorted[:pair_count]
    for i in range(0, len(disputing), 2):
        c0 = disputing[i]
        c1 = disputing[i + 1]
        subj_b = claim_block_of[c1]
        asrt = make_assertion(c1, "contradicts", c0, None, subj_b, None, "unknown", "disputed")
        all_constructed.append(asrt)
        existing_triples.add((c1, "contradicts", c0))

    # --- Phase 7: parallel evidence ---
    base_sorted = sorted(base_motif_assertions, key=lambda x: x.assertion_id)
    parallel_count = math.floor(profile.parallel_evidence_rate * 13 * B)
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
        )
        s, o, b = chosen
        asrt = make_assertion(s, "concerns", o, None, b, None, "unknown", "undisputed")
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
    )


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
    2.2 section alone."""
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
    return f"cyax-snapshot-sha256:{digest}"
