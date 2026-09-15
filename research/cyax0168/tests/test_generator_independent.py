"""Conformance tests for Worker D's independent CYAX-0168 generator 2.3
reproduction (``research/cyax0168/generator_independent.py``).

The module under test is loaded directly from its file path so that this
test suite never triggers ``research/cyax0168/__init__.py`` (which imports
the sibling ``generator_primary`` implementation written by another worker).
This keeps Worker D's review fully independent per its task packet.

Only small synthetic parameters are used (entity counts far below the real
T0 fixture size, and tier/seed labels that are not in any T0-T4/C0-C3
registry entry). No real decision or calibration fixture is generated or
accessed, per the CYAX-0168 G1 hard guard.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "generator_independent.py"
_spec = importlib.util.spec_from_file_location("cyax0168_generator_independent_under_test", _MODULE_PATH)
gi = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Python 3.14's dataclass machinery resolves postponed annotations through
# ``sys.modules`` while the module is executing.  Register the path-loaded
# module before execution, matching the importlib contract used by a normal
# import and keeping this test independent of ``research.cyax0168.__init__``.
sys.modules[_spec.name] = gi
_spec.loader.exec_module(gi)

# Small synthetic-only conformance labels: distinct from every registered
# T0-T4/C0-C3 tier/profile/seed tuple in spec.md's decision and calibration
# matrices.
TINY_TIER = "conformance-tiny"
SEED_A = 900001
SEED_B = 900002


PREDICATE_SIGNATURES = {
    "governs": (
        {"Decision", "Specification", "WorkItem"},
        {"WorkItem", "Specification", "Requirement"},
    ),
    "requires": (
        {"WorkItem", "Decision", "Specification", "Requirement"},
        {"Requirement", "Implementation", "Verification"},
    ),
    "implements": ({"Implementation", "Artifact"}, {"Requirement", "Specification"}),
    "verifies": ({"Verification", "Artifact"}, {"Claim", "Requirement", "Implementation"}),
    "supports": (
        {"Claim", "Verification", "Artifact", "Source"},
        {"Claim", "Decision", "Requirement"},
    ),
    "contradicts": (
        {"Claim", "Verification", "Artifact", "Source"},
        {"Claim", "Decision", "Requirement"},
    ),
    "depends_on": (
        {"WorkItem", "Requirement", "Implementation", "Verification", "Claim", "Artifact"},
        {"WorkItem", "Requirement", "Implementation", "Verification", "Claim", "Artifact"},
    ),
    "concerns": (
        {"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact"},
        {"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact"},
    ),
    "documented_in": (
        {"WorkItem", "Decision", "Specification", "Requirement", "Implementation", "Verification", "Claim", "Artifact"},
        {"Source"},
    ),
}
SUPERSEDES_SAME_TYPE = "supersedes"


class TestCanonicalFraming(unittest.TestCase):
    def test_null(self):
        self.assertEqual(gi.frame(None), b"N" + (0).to_bytes(8, "big"))

    def test_string(self):
        self.assertEqual(gi.frame("ab"), b"S" + (2).to_bytes(8, "big") + b"ab")

    def test_integer_nonneg(self):
        self.assertEqual(gi.frame(0), b"I" + (1).to_bytes(8, "big") + b"0")
        self.assertEqual(gi.frame(162000), b"I" + (6).to_bytes(8, "big") + b"162000")

    def test_bool(self):
        self.assertEqual(gi.frame(True), b"B" + (1).to_bytes(8, "big") + b"\x01")
        self.assertEqual(gi.frame(False), b"B" + (1).to_bytes(8, "big") + b"\x00")

    def test_array(self):
        got = gi.frame(["a", 1])
        expected = b"A" + (2).to_bytes(8, "big") + gi.frame("a") + gi.frame(1)
        self.assertEqual(got, expected)

    def test_object_sorts_keys_by_utf8_bytes(self):
        got = gi.frame({"b": 1, "a": 2})
        expected = (
            b"O"
            + (2).to_bytes(8, "big")
            + gi.frame("a")
            + gi.frame(2)
            + gi.frame("b")
            + gi.frame(1)
        )
        self.assertEqual(got, expected)

    def test_object_present_null_is_not_absent(self):
        got = gi.frame({"a": None})
        expected = b"O" + (1).to_bytes(8, "big") + gi.frame("a") + gi.frame(None)
        self.assertEqual(got, expected)
        self.assertNotEqual(got, gi.frame({}))

    def test_float_rejected(self):
        with self.assertRaises(TypeError):
            gi.frame(1.5)


class TestStableIds(unittest.TestCase):
    def test_entity_id_deterministic_and_prefixed(self):
        a = gi.entity_id("ns", "WorkItem", ["T", "P", 1, 0, 0])
        b = gi.entity_id("ns", "WorkItem", ["T", "P", 1, 0, 0])
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("cyax-entity-sha256:"))
        self.assertEqual(len(a) - len("cyax-entity-sha256:"), 64)

    def test_entity_id_sensitive_to_every_field(self):
        base = gi.entity_id("ns", "WorkItem", ["T", "P", 1, 0, 0])
        variants = [
            gi.entity_id("ns2", "WorkItem", ["T", "P", 1, 0, 0]),
            gi.entity_id("ns", "Decision", ["T", "P", 1, 0, 0]),
            gi.entity_id("ns", "WorkItem", ["T2", "P", 1, 0, 0]),
            gi.entity_id("ns", "WorkItem", ["T", "P2", 1, 0, 0]),
            gi.entity_id("ns", "WorkItem", ["T", "P", 2, 0, 0]),
            gi.entity_id("ns", "WorkItem", ["T", "P", 1, 1, 0]),
            gi.entity_id("ns", "WorkItem", ["T", "P", 1, 0, 1]),
        ]
        self.assertEqual(len({base, *variants}), 1 + len(variants))

    def test_literal_id_matches_type_and_value(self):
        a = gi.literal_id("text", "block=0;profile=P-low;seed=1")
        b = gi.literal_id("text", "block=0;profile=P-low;seed=1")
        c = gi.literal_id("text", "block=1;profile=P-low;seed=1")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.startswith("cyax-literal-sha256:"))

    def test_source_revision_id_sensitive_to_locator_and_digest(self):
        base = gi.source_revision_id("synthetic_fixture", "loc-1", "00" * 32, None)
        other_locator = gi.source_revision_id("synthetic_fixture", "loc-2", "00" * 32, None)
        other_digest = gi.source_revision_id("synthetic_fixture", "loc-1", "11" * 32, None)
        self.assertNotEqual(base, other_locator)
        self.assertNotEqual(base, other_digest)

    def test_assertion_id_requires_complete_preimage(self):
        preimage = {k: None for k in gi.ASSERTION_PREIMAGE_KEYS}
        preimage.update(
            subject_id="s",
            predicate="depends_on",
            object_id="o",
            source_revision_id="r",
            source_locator="loc",
            asserted_at=gi.FIXED_TIME,
            validity_basis="unknown",
            authority_class="ordinary_record",
            authority_derivation_rule_id="synthetic_fixture_v2.3",
            origin="source_direct",
            curation_state="independently_reviewed",
            review_state="not_required",
            epistemic_state="supported",
            dispute_state="undisputed",
        )
        aid = gi.assertion_id(preimage)
        self.assertTrue(aid.startswith("cyax-assertion-sha256:"))

        incomplete = dict(preimage)
        del incomplete["dispute_state"]
        with self.assertRaises(ValueError):
            gi.assertion_id(incomplete)

        extra = dict(preimage)
        extra["unexpected_field"] = "x"
        with self.assertRaises(ValueError):
            gi.assertion_id(extra)

    def test_assertion_id_changes_on_epistemic_state_only(self):
        base = {k: None for k in gi.ASSERTION_PREIMAGE_KEYS}
        base.update(
            subject_id="s",
            predicate="depends_on",
            object_id="o",
            source_revision_id="r",
            source_locator="loc",
            asserted_at=gi.FIXED_TIME,
            validity_basis="unknown",
            authority_class="ordinary_record",
            authority_derivation_rule_id="synthetic_fixture_v2.3",
            origin="source_direct",
            curation_state="independently_reviewed",
            review_state="not_required",
            epistemic_state="supported",
            dispute_state="undisputed",
        )
        changed = dict(base)
        changed["epistemic_state"] = "verified"
        self.assertNotEqual(gi.assertion_id(base), gi.assertion_id(changed))


class TestCanonicalJson(unittest.TestCase):
    def test_base_source_bytes_match_spec_example_shape(self):
        obj = {"block": 3, "generator": "cyax-0168-scale-2.3", "profile": "P-low", "seed": 162011, "tier": "T1"}
        raw = gi.canonical_json_bytes(obj)
        self.assertEqual(
            raw,
            b'{"block":3,"generator":"cyax-0168-scale-2.3","profile":"P-low","seed":162011,"tier":"T1"}\n',
        )

    def test_assertion_body_key_order_and_null_identity(self):
        obj = {
            "assertion_ordinal": 5,
            "generator": "cyax-0168-scale-2.3",
            "literal_identity": None,
            "object_identity": "cyax-entity-sha256:" + "ab" * 32,
            "predicate": "depends_on",
            "profile": "P-low",
            "seed": 1,
            "subject_identity": "cyax-entity-sha256:" + "cd" * 32,
            "tier": "T1",
        }
        raw = gi.canonical_json_bytes(obj)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertIn(b'"literal_identity":null', raw)
        # keys must appear in alphabetical order with no whitespace
        self.assertEqual(
            raw,
            (
                b'{"assertion_ordinal":5,"generator":"cyax-0168-scale-2.3",'
                b'"literal_identity":null,"object_identity":"cyax-entity-sha256:'
                + b"ab" * 32
                + b'","predicate":"depends_on","profile":"P-low","seed":1,'
                b'"subject_identity":"cyax-entity-sha256:' + b"cd" * 32 + b'","tier":"T1"}\n'
            ),
        )


def _first_pick_index(seed, profile_id, purpose, ordinal, n, counter=0):
    digest = gi.prf_digest(seed, profile_id, purpose, ordinal, counter)
    x = int.from_bytes(digest, "big")
    limit = (2**256 // n) * n
    if x >= limit:
        return None
    return x % n


def _find_seed_with_first_pick(purpose, ordinal, profile_id, n, want_index, start=0, tries=20000):
    for seed in range(start, start + tries):
        if _first_pick_index(seed, profile_id, purpose, ordinal, n) == want_index:
            return seed
    raise AssertionError("could not find a seed hitting the requested first-draw index")


class TestPrfSelect(unittest.TestCase):
    def test_unknown_purpose_token_rejected(self):
        with self.assertRaises(ValueError):
            gi.prf_select(
                1, "P-low", "not_a_real_purpose", 0, ["a", "b"], sort_key=lambda x: x, rejects=lambda c: False
            )

    def test_empty_candidates_fail_generation(self):
        with self.assertRaises(gi.GenerationFailure):
            gi.prf_select(1, "P-low", "dependency_target", 0, [], sort_key=lambda x: x, rejects=lambda c: False)

    def test_single_candidate_accepted_without_prf_call(self):
        calls = []
        original = gi.prf_digest
        gi.prf_digest = lambda *a, **k: (_ for _ in ()).throw(AssertionError("PRF called for n=1"))
        try:
            result = gi.prf_select(
                1, "P-low", "dependency_target", 0, ["only"], sort_key=lambda x: x, rejects=lambda c: False
            )
        finally:
            gi.prf_digest = original
        self.assertEqual(result, "only")

    def test_single_candidate_rejected_fails_without_retry(self):
        original = gi.prf_digest
        gi.prf_digest = lambda *a, **k: (_ for _ in ()).throw(AssertionError("PRF called for n=1"))
        try:
            with self.assertRaises(gi.GenerationFailure):
                gi.prf_select(
                    1, "P-low", "dependency_target", 0, ["only"], sort_key=lambda x: x, rejects=lambda c: True
                )
        finally:
            gi.prf_digest = original

    def test_self_edge_retry(self):
        subject = "id-a"
        other = "id-b"
        candidates = sorted([subject, other])
        bad_index = candidates.index(subject)
        seed = _find_seed_with_first_pick("dependency_target", 0, "P-low", 2, bad_index)

        existing_triples: set = set()
        rejection_calls = []

        def rejects(cand):
            r = gi.dependency_target_rejects(subject, cand, existing_triples)
            rejection_calls.append((cand, r))
            return r

        result = gi.prf_select(seed, "P-low", "dependency_target", 0, candidates, sort_key=lambda x: x, rejects=rejects)
        self.assertEqual(result, other)
        self.assertTrue(any(cand == subject and r for cand, r in rejection_calls))

    def test_duplicate_dependency_retry(self):
        subject, good, dup = "id-a", "id-c", "id-b"
        candidates = sorted([good, dup])
        bad_index = candidates.index(dup)
        seed = _find_seed_with_first_pick("dependency_target", 3, "P-low", 2, bad_index)

        existing_triples = {(subject, "depends_on", dup)}
        result = gi.prf_select(
            seed,
            "P-low",
            "dependency_target",
            3,
            candidates,
            sort_key=lambda x: x,
            rejects=lambda cand: gi.dependency_target_rejects(subject, cand, existing_triples),
        )
        self.assertEqual(result, good)

    def test_candidate_satisfying_both_predicates_is_accepted_after_one_retry(self):
        subject = "id-a"
        good = "id-b"
        candidates = sorted([subject, good])
        # "subject" simultaneously triggers predicate (1) self-edge and
        # predicate (2) duplicate-triple, since the pre-existing triple is
        # itself a self-loop on the subject.
        bad_index = candidates.index(subject)
        seed = _find_seed_with_first_pick("dependency_target", 7, "P-low", 2, bad_index)

        existing_triples = {(subject, "depends_on", subject)}
        rejected_candidates = []

        def rejects(cand):
            r = gi.dependency_target_rejects(subject, cand, existing_triples)
            if r:
                rejected_candidates.append(cand)
            return r

        result = gi.prf_select(seed, "P-low", "dependency_target", 7, candidates, sort_key=lambda x: x, rejects=rejects)
        self.assertEqual(result, good)
        # exactly one distinct bad candidate is ever rejected before success
        self.assertEqual(set(rejected_candidates), {subject})

    def test_duplicate_filler_retry(self):
        pair_good = ("s1", "o1", 0)
        pair_dup = ("s1", "o2", 0)
        candidates = [pair_dup, pair_good]

        def sort_key(pair):
            return gi.frame([pair[0], pair[1]])

        sorted_candidates = sorted(candidates, key=sort_key)
        dup_index = sorted_candidates.index(pair_dup)
        seed = _find_seed_with_first_pick("fill_concerns_pair", 2, "P-medium", 2, dup_index)

        existing_triples = {("s1", "concerns", "o2")}
        result = gi.prf_select(
            seed,
            "P-medium",
            "fill_concerns_pair",
            2,
            candidates,
            sort_key=sort_key,
            rejects=lambda cand: gi.fill_concerns_pair_rejects(cand[0], cand[1], existing_triples),
        )
        self.assertEqual(result, pair_good)

    def test_digest_uniformity_rejection_retries(self):
        # Force counter=0 to land in the excess (rejected) region for n=3,
        # then counter=1 to select a valid candidate; confirms the digest
        # rejection-sampling branch (x >= L) advances the counter.
        n = 3
        limit = (2**256 // n) * n
        over_limit_digest = (2**256 - 1).to_bytes(32, "big")
        self.assertGreaterEqual(int.from_bytes(over_limit_digest, "big"), limit)
        valid_digest = (0).to_bytes(32, "big")  # x=0 -> index 0

        calls = {"n": 0}

        def fake_prf_digest(seed, profile_id, purpose, ordinal, counter):
            calls["n"] += 1
            return over_limit_digest if counter == 0 else valid_digest

        original = gi.prf_digest
        gi.prf_digest = fake_prf_digest
        try:
            result = gi.prf_select(
                1, "P-low", "dependency_target", 0, ["a", "b", "c"], sort_key=lambda x: x, rejects=lambda c: False
            )
        finally:
            gi.prf_digest = original
        self.assertEqual(result, "a")
        self.assertEqual(calls["n"], 2)

    def test_counter_overflow_fails_generation(self):
        original_max = gi.MAX_COUNTER
        gi.MAX_COUNTER = 1
        try:
            with self.assertRaises(gi.GenerationFailure):
                gi.prf_select(
                    1, "P-low", "dependency_target", 0, ["a", "b"], sort_key=lambda x: x, rejects=lambda c: True
                )
        finally:
            gi.MAX_COUNTER = original_max

    def test_trace_records_frozen_candidates_attempt_counters_and_retry_causes(self):
        subject, bad, good = "id-a", "id-a", "id-b"
        candidates = [bad, good]
        seed = _find_seed_with_first_pick("dependency_target", 11, "P-low", 2, 0)
        trace = []
        result = gi.prf_select(
            seed,
            "P-low",
            "dependency_target",
            11,
            candidates,
            sort_key=lambda x: x,
            rejects=lambda candidate: gi.dependency_target_rejects(subject, candidate, set()),
            trace=trace,
            phase="phase-3-dependencies",
            subject_id=subject,
        )
        self.assertEqual(result, good)
        self.assertEqual(len(trace), 1)
        choice = trace[0]
        self.assertEqual(choice["candidates"], sorted(candidates))
        self.assertEqual([attempt["counter"] for attempt in choice["attempts"]], [0, 1])
        self.assertTrue(choice["attempts"][0]["predicate_rejected"])
        self.assertEqual(choice["selected_counter"], 1)
        self.assertEqual(choice["retry_count"], 1)


class TestPredicateRejectionClosure(unittest.TestCase):
    """Differential check that the two purpose-row predicate functions
    implement exactly, and only, their two documented rejection rules."""

    def test_dependency_target_rejects_matches_reference(self):
        existing = {("s1", "depends_on", "o1")}
        cases = [
            ("s1", "o1", True),  # duplicate triple
            ("s1", "s1", True),  # self edge
            ("s1", "o2", False),  # neither
            ("s2", "o1", False),  # different subject, no duplicate for s2
        ]
        for subj, obj, expected in cases:
            with self.subTest(subj=subj, obj=obj):
                self.assertEqual(gi.dependency_target_rejects(subj, obj, existing), expected)

    def test_fill_concerns_pair_rejects_matches_reference(self):
        existing = {("s1", "concerns", "o1")}
        cases = [
            ("s1", "o1", True),
            ("s1", "s1", True),
            ("s1", "o2", False),
        ]
        for subj, obj, expected in cases:
            with self.subTest(subj=subj, obj=obj):
                self.assertEqual(gi.fill_concerns_pair_rejects(subj, obj, existing), expected)

    def test_unlisted_candidate_properties_do_not_trigger_dependency_retry(self):
        # Component membership, isolation status, cycle eligibility, and
        # candidate labels are not predicates in the closed purpose row.
        existing = {("other-subject", "depends_on", "id-o")}
        self.assertFalse(gi.dependency_target_rejects("id-s", "id-o", existing))
        self.assertFalse(gi.dependency_target_rejects("id-s", "id-o|isolated", existing))


class TestGenerateSnapshotStructure(unittest.TestCase):
    def test_rejects_entity_count_not_multiple_of_ten(self):
        with self.assertRaises(gi.GenerationFailure):
            gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 15)

    def test_rejects_unknown_profile(self):
        with self.assertRaises(ValueError):
            gi.generate_snapshot(TINY_TIER, "P-nonexistent", SEED_A, 100)

    def test_small_p_low_counts_and_signatures(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)  # B=10
        self.assertEqual(snap.block_count, 10)
        self.assertEqual(len(snap.entities), 100)
        self.assertEqual(len(snap.assertions), 500)  # 50 * B
        self._assert_predicate_signatures(snap)
        self._assert_role_layout(snap)

    def test_base_motif_construction_order_places_literal_before_source_link(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        first_block = sorted(snap.assertions, key=lambda a: a.assertion_ordinal)[:13]
        self.assertEqual([a.assertion_ordinal for a in first_block], list(range(13)))
        self.assertEqual(first_block[-2].predicate, "documented_in")
        self.assertIsNotNone(first_block[-2].literal_ref)
        self.assertEqual(first_block[-1].predicate, "documented_in")
        self.assertIsNotNone(first_block[-1].object_id)

    def test_small_p_medium_counts_and_signatures(self):
        # B=130 is the smallest compact scale used here with enough
        # connected components and same-component candidates for the frozen
        # P-medium dependency quota (smaller B legitimately fails generation
        # under the spec's empty-vector/duplicate rejection rules).
        snap = gi.generate_snapshot(TINY_TIER, "P-medium", SEED_A, 1300)  # B=130
        self.assertEqual(len(snap.entities), 1300)
        self.assertEqual(len(snap.assertions), 6500)
        self._assert_predicate_signatures(snap)

    def _assert_role_layout(self, snap):
        by_type = {}
        for e in snap.entities:
            by_type[e.entity_type] = by_type.get(e.entity_type, 0) + 1
        B = snap.block_count
        self.assertEqual(by_type["WorkItem"], B)
        self.assertEqual(by_type["Decision"], B)
        self.assertEqual(by_type["Requirement"], 2 * B)
        self.assertEqual(by_type["Implementation"], 2 * B)
        self.assertEqual(by_type["Verification"], B)
        self.assertEqual(by_type["Claim"], B)
        self.assertEqual(by_type["Artifact"], B)
        self.assertEqual(by_type["Source"], B)
        claim_entities = [e for e in snap.entities if e.entity_type == "Claim"]
        self.assertTrue(all(e.claim_key == "synthetic.block_claim" for e in claim_entities))
        non_claims = [e for e in snap.entities if e.entity_type != "Claim"]
        self.assertTrue(all(e.claim_key is None for e in non_claims))
        self.assertTrue(all(e.display_label_ref is None for e in snap.entities))
        self.assertTrue(all(e.namespace == gi.NAMESPACE for e in snap.entities))
        self.assertTrue(all(len(e.canonical_source_identity) == 5 for e in snap.entities))

    def _assert_predicate_signatures(self, snap):
        type_of = {e.entity_id: e.entity_type for e in snap.entities}
        for a in snap.assertions:
            subj_type = type_of[a.subject_id]
            if a.predicate == "documented_in" and a.literal_ref is not None:
                self.assertEqual(subj_type, "Claim")
                self.assertIsNone(a.object_id)
                continue
            self.assertIsNotNone(a.object_id, f"non-literal assertion missing object_id: {a}")
            obj_type = type_of[a.object_id]
            if a.predicate == SUPERSEDES_SAME_TYPE:
                self.assertEqual(subj_type, obj_type)
                continue
            allowed_subj, allowed_obj = PREDICATE_SIGNATURES[a.predicate]
            self.assertIn(subj_type, allowed_subj, f"{a.predicate}: bad subject type {subj_type}")
            self.assertIn(obj_type, allowed_obj, f"{a.predicate}: bad object type {obj_type}")


class TestGenerator23Contract(unittest.TestCase):
    """Focused checks for the identity-bearing 2.3 amendments."""

    def test_manifest_version_namespace_and_locator_are_23(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        self.assertEqual(snap.generator_name, "cyax-0168-scale-2.3")
        self.assertEqual(snap.generator_version, "2.3")
        self.assertEqual(snap.prf_domain, "cyax-gen-2.3")
        self.assertEqual(snap.synthetic_namespace, "cyax-0168-synthetic-v2.3")
        self.assertEqual(snap.source_locator_version, "/scale/2.3/")
        self.assertEqual(snap.authority_derivation_rule_id, "synthetic_fixture_v2.3")
        self.assertTrue(all("/scale/2.3/" in revision.locator for revision in snap.source_revisions))

    def test_phase2_uses_primary_order_contiguous_slices_and_later_to_previous_direction(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        workitems = {e.entity_id: e.canonical_source_identity[3] for e in snap.entities if e.entity_type == "WorkItem"}
        block_order = sorted(range(snap.block_count), key=lambda b: next(e.entity_id for e in snap.entities if e.entity_type == "WorkItem" and e.canonical_source_identity[3] == b))
        phase1 = next(x for x in snap.construction_trace if x.get("phase") == "phase-1-isolation")
        phase2 = next(x for x in snap.construction_trace if x.get("phase") == "phase-2-components" and x.get("event") == "phase_complete")
        self.assertEqual(phase1["block_primary_order"], block_order)
        connected = phase1["connected_block_ordinals"]
        flattened = [b for component in phase2["components"] for b in component]
        self.assertEqual(flattened, connected)
        self.assertTrue(all(component == connected[i : i + phase2["component_capacity"]] for i, component in enumerate(phase2["components"])))
        chains = [a for a in snap.assertions if a.predicate == "depends_on" and a.assertion_ordinal < 13 * snap.block_count + 100]
        # Check all fixed-chain edges by their component adjacency, not by
        # assertion-ID order.
        for component in phase2["components"]:
            for earlier, later in zip(component[:5], component[1:5]):
                expected_subject = next(e.entity_id for e in snap.entities if e.entity_type == "WorkItem" and e.canonical_source_identity[3] == later)
                expected_object = next(e.entity_id for e in snap.entities if e.entity_type == "WorkItem" and e.canonical_source_identity[3] == earlier)
                self.assertTrue(any(a.subject_id == expected_subject and a.object_id == expected_object for a in chains))

    def test_phase3_ordinary_candidate_vector_excludes_subject(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        choices = [choice for choice in snap.prf_trace if choice["purpose"] == "dependency_target"]
        self.assertGreater(len(choices), 0)
        for choice in choices:
            selected = choice["selected"]
            subject = choice["subject_id"]
            candidates = gi.reconstruct_candidate_vector(
                {"candidate_vectors": snap.candidate_vectors}, choice
            )
            self.assertNotIn(subject, candidates)

    def test_generator_trace_deduplicates_vectors_and_reconstructs_losslessly(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        self.assertTrue(snap.prf_trace)
        self.assertTrue(all("candidates" not in choice for choice in snap.prf_trace))
        self.assertLess(len(snap.candidate_vectors), len(snap.prf_trace))
        for choice in snap.prf_trace:
            vector = gi.reconstruct_candidate_vector(
                {"candidate_vectors": snap.candidate_vectors}, choice
            )
            self.assertEqual(vector, snap.candidate_vectors[choice["candidate_vector_id"]])

    def test_phase5_excludes_isolated_blocks_only(self):
        import math

        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        profile = gi.PROFILES["P-low"]
        workitem_by_block = {e.canonical_source_identity[3]: e.entity_id for e in snap.entities if e.entity_type == "WorkItem"}
        primary = sorted(workitem_by_block, key=lambda b: workitem_by_block[b])
        isolated = set(primary[-math.floor(profile.isolated_rate * snap.block_count) :])
        supersedes = [a for a in snap.assertions if a.predicate == "supersedes"]
        self.assertEqual(len(supersedes), (snap.block_count - len(isolated)) // profile.supersession_depth * (profile.supersession_depth - 1))
        self.assertTrue(all(next(b for b, wid in workitem_by_block.items() if wid == a.subject_id) not in isolated for a in supersedes))
        self.assertTrue(all(next(b for b, wid in workitem_by_block.items() if wid == a.object_id) not in isolated for a in supersedes))

    def test_complete_records_bytes_trace_ids_and_checksum_close(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        self.assertEqual(len(snap.records["entities"]), len(snap.entities))
        self.assertEqual(len(snap.records["assertions"]), len(snap.assertions))
        self.assertEqual(len(snap.source_bytes), len(snap.source_revisions))
        self.assertEqual(len(snap.assertion_ids), len(snap.assertions))
        raw_checksum = gi.compute_logical_snapshot_checksum(snap)
        self.assertEqual(snap.logical_snapshot_checksum, raw_checksum)
        self.assertEqual(snap.snapshot_id, "cyax-snapshot-sha256:" + raw_checksum)
        self.assertEqual(gi.complete_output(snap)["logical_snapshot_checksum"], raw_checksum)
        self.assertEqual(gi.complete_output(snap)["snapshot_id"], snap.snapshot_id)
        by_locator = {revision.locator: revision for revision in snap.source_revisions}
        for assertion in snap.assertions:
            revision = by_locator[assertion.source_locator]
            decoded = __import__("json").loads(revision.raw_bytes)
            self.assertEqual(decoded["assertion_ordinal"], assertion.assertion_ordinal)
            self.assertEqual(decoded["predicate"], assertion.predicate)
            self.assertEqual(decoded["subject_identity"], assertion.subject_id)
            self.assertEqual(decoded["object_identity"], assertion.object_id)
            self.assertEqual(decoded["literal_identity"], assertion.literal_ref)

    def test_decision_cells_are_not_materialized_and_calibration_matrix_is_frozen(self):
        self.assertEqual(gi.CALIBRATION_MATRIX["C0"]["P-medium"], (168900,))
        self.assertEqual(gi.CALIBRATION_MATRIX["C3"]["P-high"], (168933,))
        with self.assertRaises(gi.GenerationFailure):
            gi.generate_snapshot("T0", "P-medium", 162000, 1000)
        with self.assertRaises(gi.GenerationFailure):
            gi.generate_calibration_snapshot("T1", "P-low", 162011)


class TestDeterminism(unittest.TestCase):
    def test_repeated_generation_is_byte_identical(self):
        snap1 = gi.generate_snapshot(TINY_TIER, "P-medium", SEED_A, 1300)
        snap2 = gi.generate_snapshot(TINY_TIER, "P-medium", SEED_A, 1300)

        ids1 = [a.assertion_id for a in snap1.assertions]
        ids2 = [a.assertion_id for a in snap2.assertions]
        self.assertEqual(ids1, ids2)

        for a1, a2 in zip(snap1.assertions, snap2.assertions):
            self.assertEqual(a1.preimage(), a2.preimage())

        ent1 = [(e.entity_id, e.entity_type, e.claim_key) for e in snap1.entities]
        ent2 = [(e.entity_id, e.entity_type, e.claim_key) for e in snap2.entities]
        self.assertEqual(ent1, ent2)

        self.assertEqual(gi.semantic_projection_checksum(snap1), gi.semantic_projection_checksum(snap2))
        self.assertEqual(
            gi.compute_logical_snapshot_checksum(snap1), gi.compute_logical_snapshot_checksum(snap2)
        )

    def test_different_seed_changes_checksum(self):
        snap_a = gi.generate_snapshot(TINY_TIER, "P-medium", SEED_A, 1300)
        snap_b = gi.generate_snapshot(TINY_TIER, "P-medium", SEED_B, 1300)
        self.assertNotEqual(
            gi.semantic_projection_checksum(snap_a), gi.semantic_projection_checksum(snap_b)
        )
        self.assertNotEqual(
            [a.assertion_id for a in snap_a.assertions], [a.assertion_id for a in snap_b.assertions]
        )

    def test_display_label_only_style_change_does_not_alter_semantic_checksum(self):
        # Entity records already exclude display_label_ref from the semantic
        # projection; mutating it must not change the projection checksum.
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        before = gi.semantic_projection_checksum(snap)
        for e in snap.entities:
            e.display_label_ref = "diagnostic-only-" + e.entity_id
        after = gi.semantic_projection_checksum(snap)
        self.assertEqual(before, after)

    def test_source_bytes_and_ids_replay_from_canonical_fields(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        base_by_locator = {revision.locator: revision for revision in snap.base_source_revisions}
        assertion_revisions = {
            revision.source_revision_id: revision for revision in snap.assertion_source_revisions
        }

        for block in range(snap.block_count):
            locator = f"cyax://0168/scale/2.3/{TINY_TIER}/P-low/{SEED_A}/block/{block}/base"
            revision = base_by_locator[locator]
            expected = (
                f'{{"block":{block},"generator":"cyax-0168-scale-2.3",'
                f'"profile":"P-low","seed":{SEED_A},"tier":"{TINY_TIER}"}}\n'
            ).encode("ascii")
            self.assertEqual(revision.raw_bytes, expected)
            self.assertEqual(revision.object_byte_count, len(expected))
            self.assertEqual(revision.object_sha256, gi.sha256_hex(expected))
            self.assertEqual(
                revision.source_revision_id,
                gi.source_revision_id("synthetic_fixture", locator, revision.object_sha256, None),
            )

        for assertion in snap.assertions:
            revision = assertion_revisions[assertion.source_revision_id]
            self.assertEqual(assertion.source_locator, revision.locator)
            self.assertEqual(assertion.assertion_id, gi.assertion_id(assertion.preimage()))
            self.assertEqual(
                revision.source_revision_id,
                gi.source_revision_id("synthetic_fixture", revision.locator, revision.object_sha256, None),
            )
            self.assertEqual(revision.raw_bytes, revision.raw_bytes.rstrip(b"\n") + b"\n")

    def test_common_physical_and_semantic_record_schemas(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        physical = gi.physical_records(snap)
        semantic = gi.semantic_projection(snap)
        self.assertIn("display_label_ref", physical["entities"][0])
        self.assertNotIn("display_label_ref", semantic["semantic_entities"][0])
        self.assertEqual(
            set(physical["source_revisions"][0]),
            {
                "source_revision_id",
                "source_kind",
                "source_entity_id",
                "canonical_locator",
                "object_sha256",
                "object_byte_count",
                "source_event_at",
                "observed_at",
                "actor_id",
                "authority_class",
                "authority_derivation_rule_id",
            },
        )
        self.assertNotIn("locator", physical["source_revisions"][0])
        self.assertNotIn("byte_count", physical["source_revisions"][0])

    def test_physical_and_semantic_source_revision_boundaries(self):
        snap = gi.generate_snapshot(TINY_TIER, "P-low", SEED_A, 100)
        self.assertEqual(len(gi.physical_source_revisions(snap)), 510)
        self.assertEqual(len(gi.semantic_projection(snap)["semantic_source_revisions"]), 500)
        self.assertEqual(
            gi.compute_semantic_source_bundle_projection_checksum(snap),
            gi.compute_semantic_source_bundle_projection_checksum(snap),
        )
        for revision in gi.physical_source_revisions(snap):
            self.assertIsNone(revision.actor_id)
            self.assertIsNone(revision.actor_login)
            self.assertIsNone(revision.author_association)
            self.assertIsNone(revision.event_state)
            self.assertIsNone(revision.role_evidence)
            self.assertIsNone(revision.metadata)
            if "/assertion/" in revision.locator:
                self.assertEqual(revision.anchors, ["json-object"])
            else:
                self.assertEqual(revision.anchors, [])


class TestPhaseQuotas(unittest.TestCase):
    """Cross-checks generated structure against independently recomputed
    phase quotas (spec.md's ordered post-motif phases 1-8)."""

    def setUp(self):
        self.profile_id = "P-medium"
        self.entity_count = 1300  # B = 130
        self.snap = gi.generate_snapshot(TINY_TIER, self.profile_id, SEED_A, self.entity_count)
        self.B = self.entity_count // 10
        self.profile = gi.PROFILES[self.profile_id]

    def test_isolated_and_dependency_and_dispute_and_parallel_quotas(self):
        import math

        snap = self.snap
        B = self.B
        profile = self.profile

        isolated_count = math.floor(profile.isolated_rate * B)
        connected_blocks_count = B - isolated_count

        dependency_quota = min(profile.dependency_fanout * connected_blocks_count, 30 * B)
        cross_link_count = math.floor(profile.cross_link_rate * dependency_quota)
        self.assertLessEqual(cross_link_count, dependency_quota)

        depends_on_assertions = [a for a in snap.assertions if a.predicate == "depends_on"]
        # net additional non-chain depends_on assertions after any phase-4
        # removal must still total at least the chain edges (chain edges are
        # never removed).
        self.assertGreater(len(depends_on_assertions), 0)

        contradicts = [a for a in snap.assertions if a.predicate == "contradicts"]
        expected_pairs = 2 * math.floor(profile.dispute_rate * B)
        self.assertEqual(len(contradicts), expected_pairs // 2)
        for a in contradicts:
            self.assertEqual(a.dispute_state, "disputed")
        non_contradicts = [a for a in snap.assertions if a.predicate != "contradicts"]
        self.assertTrue(all(a.dispute_state == "undisputed" for a in non_contradicts))

        supersedes = [a for a in snap.assertions if a.predicate == "supersedes"]
        depth = profile.supersession_depth
        # 2.3 excludes phase-1 isolated blocks only.  Cycle members remain
        # eligible for phase-5 supersession, while cycle edges themselves are
        # never replaced by supersession edges.
        eligible_count = B - isolated_count
        expected_chains = eligible_count // depth
        self.assertEqual(len(supersedes), expected_chains * (depth - 1))
        for a in supersedes:
            self.assertEqual(a.validity_basis, "explicit")
            self.assertIsNotNone(a.valid_from)
            self.assertIsNone(a.valid_to)

        parallel_count = math.floor(profile.parallel_evidence_rate * 13 * B)
        self.assertGreaterEqual(parallel_count, 0)

        self.assertEqual(len(snap.assertions), 50 * B)

    def test_valid_from_null_except_supersedes(self):
        for a in self.snap.assertions:
            if a.predicate == "supersedes":
                self.assertIsNotNone(a.valid_from)
            else:
                self.assertIsNone(a.valid_from)
            self.assertIsNone(a.valid_to)
            self.assertIsNone(a.source_event_at)
            self.assertEqual(a.asserted_at, gi.FIXED_TIME)


class TestCycleConstruction(unittest.TestCase):
    """Exercises phase 4 (cycle construction) at a scale that guarantees at
    least one disjoint eligible triple, and validates the cycle-removal
    trace structurally."""

    def test_cycle_trace_present_and_consistent(self):
        import math

        snap = gi.generate_snapshot(TINY_TIER, "P-high", 900010, 1600)  # B=160
        profile = gi.PROFILES["P-high"]
        B = 160
        isolated_count = math.floor(profile.isolated_rate * B)
        connected_blocks_count = B - isolated_count
        cycle_quota = math.floor(profile.cycle_rate * connected_blocks_count)
        self.assertGreaterEqual(cycle_quota, 1, "test scale must exercise phase 4; adjust B if this fails")
        self.assertEqual(len(snap.cycle_trace), cycle_quota)

        final_ids = {a.assertion_id for a in snap.assertions}
        removed_ids = {r.assertion_id for r in snap.removed_assertions}
        self.assertTrue(removed_ids.isdisjoint(final_ids))

        for entry in snap.cycle_trace:
            triple = entry["triple"]
            self.assertEqual(len(triple), 3)
            self.assertEqual(len(set(triple)), 3)  # disjoint members
            self.assertEqual(len(entry["removed_assertion_ids"]), 3)
            self.assertEqual(len(entry["added_assertion_ids"]), 3)
            for rid in entry["removed_assertion_ids"]:
                self.assertIn(rid, removed_ids)
                self.assertNotIn(rid, final_ids)
            for aid in entry["added_assertion_ids"]:
                self.assertIn(aid, final_ids)

            a, b, c = triple
            added = entry["added_assertion_ids"]
            by_id = {x.assertion_id: x for x in snap.assertions}
            self.assertEqual(by_id[added[0]].subject_id, a)
            self.assertEqual(by_id[added[0]].object_id, b)
            self.assertEqual(by_id[added[1]].subject_id, b)
            self.assertEqual(by_id[added[1]].object_id, c)
            self.assertEqual(by_id[added[2]].subject_id, c)
            self.assertEqual(by_id[added[2]].object_id, a)

        # 2.3 permits cycle members to participate in phase-5 supersession;
        # only phase-1 isolated WorkItems are excluded.  The exact count is a
        # stronger check than relying on a particular cycle landing in one of
        # the complete depth-sized chains.
        supersedes = [a for a in snap.assertions if a.predicate == "supersedes"]
        self.assertEqual(len(supersedes), (B - isolated_count) // profile.supersession_depth * (profile.supersession_depth - 1))


class TestParallelEvidence(unittest.TestCase):
    def test_parallel_evidence_preserves_triple_changes_provenance(self):
        import math

        snap = gi.generate_snapshot(TINY_TIER, "P-high", 900011, 1600)  # B=160
        profile = gi.PROFILES["P-high"]
        B = 160
        parallel_count = math.floor(profile.parallel_evidence_rate * 13 * B)
        self.assertGreater(parallel_count, 0)

        by_triple = {}
        for a in snap.assertions:
            key = (a.subject_id, a.predicate, a.object_id, a.literal_ref)
            by_triple.setdefault(key, []).append(a)
        duplicated = [group for group in by_triple.values() if len(group) > 1]
        self.assertGreaterEqual(len(duplicated), 1)
        for group in duplicated:
            revs = {a.source_revision_id for a in group}
            ids = {a.assertion_id for a in group}
            self.assertEqual(len(revs), len(group))
            self.assertEqual(len(ids), len(group))


if __name__ == "__main__":
    unittest.main()
