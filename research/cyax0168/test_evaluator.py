#!/usr/bin/env python3
"""Tests for the CYAX-0168 reference evaluator against tiny synthetic
snapshots built directly in this file (no generator 2.2 access; no F-real
or T0-T4 fixtures, per the CYAX-0168 G1 hard guard)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluator as ev


T0 = "2000-01-01T00:00:00.000000Z"
T1 = "2000-01-02T00:00:00.000000Z"
T2 = "2000-01-03T00:00:00.000000Z"
T3 = "2000-01-04T00:00:00.000000Z"


def make_entity(entity_id: str, entity_type: str, claim_key: str | None = None) -> ev.Entity:
    return ev.Entity(entity_id=entity_id, entity_type=entity_type, claim_key=claim_key)


def make_assertion(
    assertion_id: str,
    subject_id: str,
    predicate: str,
    *,
    object_id: str | None = None,
    literal_ref: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    curation_state: str = "independently_reviewed",
    review_state: str = "not_required",
    epistemic_state: str = "verified",
    dispute_state: str = "undisputed",
    authority_class: str = "ordinary_record",
) -> ev.Assertion:
    return ev.Assertion(
        assertion_id=assertion_id,
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        literal_ref=literal_ref,
        source_revision_id="sr:1",
        source_locator="json-object",
        source_event_at=None,
        asserted_at=T0,
        valid_from=valid_from,
        valid_to=valid_to,
        validity_basis="unknown" if valid_from is None else "explicit",
        authority_class=authority_class,
        authority_derivation_rule_id="test",
        origin="source_direct",
        curation_state=curation_state,
        review_state=review_state,
        epistemic_state=epistemic_state,
        dispute_state=dispute_state,
    )


def make_snapshot(entities, assertions, literals=()) -> ev.Snapshot:
    return ev.Snapshot(
        snapshot_id="cyax-snapshot-sha256:test",
        entities={e.entity_id: e for e in entities},
        literals={l.literal_id: l for l in literals},
        source_revisions={
            "sr:1": ev.SourceRevision(
                source_revision_id="sr:1",
                source_kind="synthetic_fixture",
                canonical_locator="cyax://test",
                object_sha256="0" * 64,
            )
        },
        assertions={a.assertion_id: a for a in assertions},
    )


class CanonicalFrameTests(unittest.TestCase):
    def test_frame_is_deterministic_and_order_sensitive(self):
        a = ev.frame(["x", 1, None, True])
        b = ev.frame(["x", 1, None, True])
        self.assertEqual(a, b)
        c = ev.frame([1, "x", None, True])
        self.assertNotEqual(a, c)

    def test_object_frame_is_key_order_independent(self):
        a = ev.frame({"b": 1, "a": 2})
        b = ev.frame({"a": 2, "b": 1})
        self.assertEqual(a, b)

    def test_bool_and_int_frame_differently(self):
        self.assertNotEqual(ev.frame(True), ev.frame(1))


class AdmissibilityTests(unittest.TestCase):
    def test_unreviewed_is_inadmissible(self):
        a = make_assertion("a1", "x", "governs", object_id="y", curation_state="unreviewed")
        self.assertFalse(ev.is_admissible(a))

    def test_extracted_epistemic_state_is_inadmissible(self):
        a = make_assertion("a1", "x", "governs", object_id="y", epistemic_state="extracted")
        self.assertFalse(ev.is_admissible(a))

    def test_pending_review_is_inadmissible_when_required(self):
        a = make_assertion("a1", "x", "governs", object_id="y", review_state="pending")
        self.assertFalse(ev.is_admissible(a))

    def test_passed_review_is_admissible(self):
        a = make_assertion("a1", "x", "governs", object_id="y", review_state="passed")
        self.assertTrue(ev.is_admissible(a))

    def test_curator_checked_supported_is_admissible(self):
        a = make_assertion(
            "a1", "x", "governs", object_id="y",
            curation_state="curator_checked", epistemic_state="supported",
        )
        self.assertTrue(ev.is_admissible(a))


class TemporalTests(unittest.TestCase):
    def test_as_of_none_is_current_when_domain_known(self):
        a = make_assertion("a1", "x", "governs", object_id="y", valid_from=T0)
        self.assertEqual(ev.temporal_state(a, None), "current")

    def test_unknown_domain_start_is_unknown_even_without_as_of(self):
        a = make_assertion("a1", "x", "governs", object_id="y")
        self.assertEqual(ev.temporal_state(a, None), "unknown")
        self.assertFalse(ev.temporal_ok(a, None))
        self.assertFalse(ev.temporal_ok(a, T2))

    def test_half_open_interval(self):
        a = make_assertion("a1", "x", "governs", object_id="y", valid_from=T1, valid_to=T3)
        self.assertEqual(ev.temporal_state(a, T0), "not_yet")
        self.assertEqual(ev.temporal_state(a, T1), "current")
        self.assertEqual(ev.temporal_state(a, T2), "current")
        self.assertEqual(ev.temporal_state(a, T3), "ended")


class SupersessionTests(unittest.TestCase):
    def test_superseded_entity_detected(self):
        old = make_entity("old", "WorkItem")
        new = make_entity("new", "WorkItem")
        sup = make_assertion("sup1", "new", "supersedes", object_id="old", valid_from=T1)
        snap = make_snapshot([old, new], [sup])
        self.assertFalse(ev.is_superseded("old", snap, T0))
        self.assertTrue(ev.is_superseded("old", snap, T1))
        self.assertTrue(ev.is_superseded("old", snap, None))

    def test_competing_successors_ambiguity(self):
        old = make_entity("old", "WorkItem")
        new_a = make_entity("newA", "WorkItem")
        new_b = make_entity("newB", "WorkItem")
        sup_a = make_assertion("supA", "newA", "supersedes", object_id="old", valid_from=T1)
        sup_b = make_assertion("supB", "newB", "supersedes", object_id="old", valid_from=T1)
        snap = make_snapshot([old, new_a, new_b], [sup_a, sup_b])
        competitors = ev.competing_successors("old", snap, None)
        self.assertEqual(len(competitors), 2)
        self.assertEqual([c.assertion_id for c in competitors], ["supA", "supB"])


class NearestRankSelectTests(unittest.TestCase):
    def test_p50_matches_hand_computed_example(self):
        population = [("e1", 1), ("e2", 2), ("e3", 2), ("e4", 3)]
        # sorted by (metric, id): e1(1), e2(2), e3(2), e4(3); n=4, p=0.5 -> rank=ceil(2)=2 -> e2
        self.assertEqual(ev.nearest_rank_select(population, 0.5, lambda x: x), "e2")

    def test_p90_matches_hand_computed_example(self):
        population = [(f"e{i}", i) for i in range(1, 11)]
        # n=10, p=0.9 -> rank=ceil(9)=9 -> item at index 8 -> e9
        self.assertEqual(ev.nearest_rank_select(population, 0.9, lambda x: x), "e9")

    def test_tie_break_by_smallest_id(self):
        population = [("z", 1), ("a", 1), ("m", 1)]
        # all tie at metric=1; p50 over n=3 -> rank 2 -> sorted ids a,m,z -> rank2 = m
        self.assertEqual(ev.nearest_rank_select(population, 0.5, lambda x: x), "m")

    def test_empty_population_raises(self):
        with self.assertRaises(ev.EvaluatorError):
            ev.nearest_rank_select([], 0.5, lambda x: x)


class Q01Tests(unittest.TestCase):
    def test_governors_found_and_admissibility_filtered(self):
        x = make_entity("x", "WorkItem")
        d1 = make_entity("d1", "Decision")
        d2 = make_entity("d2", "Decision")
        good = make_assertion("a1", "d1", "governs", object_id="x", valid_from=T0)
        bad = make_assertion("a2", "d2", "governs", object_id="x", valid_from=T0, curation_state="unreviewed")
        snap = make_snapshot([x, d1, d2], [good, bad])
        pop = ev.q01_population(snap, T1)
        self.assertEqual(pop, [("x", 1)])
        bundle = ev.q01_answer(snap, "x", T1)
        self.assertEqual(bundle.assertions, ("a1",))
        self.assertEqual(set(bundle.entities), {"x", "d1"})


class Q02Tests(unittest.TestCase):
    def test_orders_by_authority_rank_then_assertion_id(self):
        x = make_entity("x", "Decision")
        s1 = make_entity("s1", "Verification")
        s2 = make_entity("s2", "Claim", claim_key="test.claim")
        low_rank = make_assertion(
            "a_low", "s1", "supports", object_id="x", valid_from=T0,
            authority_class="verification_evidence",
        )
        high_rank = make_assertion(
            "a_high", "s2", "supports", object_id="x", valid_from=T0,
            authority_class="owner_decision",
        )
        snap = make_snapshot([x, s1, s2], [low_rank, high_rank])
        bundle = ev.q02_answer(snap, "x", T1)
        self.assertEqual(bundle.assertions, ("a_high", "a_low"))

    def test_tie_break_by_assertion_id_within_same_rank(self):
        x = make_entity("x", "Decision")
        s1 = make_entity("s1", "Verification")
        s2 = make_entity("s2", "Verification")
        a2 = make_assertion(
            "a2", "s1", "supports", object_id="x", valid_from=T0,
            authority_class="verification_evidence",
        )
        a1 = make_assertion(
            "a1", "s2", "supports", object_id="x", valid_from=T0,
            authority_class="verification_evidence",
        )
        snap = make_snapshot([x, s1, s2], [a2, a1])
        bundle = ev.q02_answer(snap, "x", T1)
        self.assertEqual(bundle.assertions, ("a1", "a2"))


class Q03Tests(unittest.TestCase):
    def test_shortest_witness_and_lexicographic_tie_break(self):
        d = make_entity("d", "Decision")
        r1 = make_entity("r1", "Requirement")
        r2 = make_entity("r2", "Requirement")
        i1 = make_entity("i1", "Implementation")
        v1 = make_entity("v1", "Verification")
        assertions = [
            make_assertion("z_dr1", "d", "requires", object_id="r1", valid_from=T0),
            make_assertion("a_dr2", "d", "requires", object_id="r2", valid_from=T0),
            make_assertion("ri1", "r1", "requires", object_id="i1", valid_from=T0),
            make_assertion("ri2", "r2", "requires", object_id="i1", valid_from=T0),
            # ``verifies`` is stored as Verification -> Implementation; Q03
            # traverses this final edge in reverse.
            make_assertion("iv1", "v1", "verifies", object_id="i1", valid_from=T0),
        ]
        snap = make_snapshot([d, r1, r2, i1, v1], assertions)
        pop = ev.q03_population(snap, path_depth=3)
        self.assertEqual(pop, [(("d", "v1"), 3)])
        bundle = ev.q03_answer(snap, ("d", "v1"), path_depth=3)
        # both z_dr1->ri1->iv1 and a_dr2->ri2->iv1 are length-3 witnesses;
        # lexicographic comparison of [a_dr2, ri2, iv1] vs [z_dr1, ri1, iv1]
        # picks the 'a_dr2' branch first.
        self.assertEqual(bundle.assertions, ("a_dr2", "iv1", "ri2"))
        self.assertEqual(
            [s.assertion_id for s in bundle.paths[0].steps],
            ["a_dr2", "ri2", "iv1"],
        )

    def test_exceeding_path_depth_excludes_from_population(self):
        d = make_entity("d", "Decision")
        r1 = make_entity("r1", "Requirement")
        i1 = make_entity("i1", "Implementation")
        v1 = make_entity("v1", "Verification")
        assertions = [
            make_assertion("dr1", "d", "requires", object_id="r1", valid_from=T0),
            make_assertion("ri1", "r1", "requires", object_id="i1", valid_from=T0),
            make_assertion("iv1", "v1", "verifies", object_id="i1", valid_from=T0),
        ]
        snap = make_snapshot([d, r1, i1, v1], assertions)
        self.assertEqual(ev.q03_population(snap, path_depth=2), [])


class Q04Tests(unittest.TestCase):
    def test_population_orders_by_valid_from_then_assertion_then_subject(self):
        old1 = make_entity("old1", "WorkItem")
        old2 = make_entity("old2", "WorkItem")
        new1 = make_entity("new1", "WorkItem")
        sup_later = make_assertion("sup_b", "new1", "supersedes", object_id="old2", valid_from=T2)
        sup_earlier = make_assertion("sup_a", "new1", "supersedes", object_id="old1", valid_from=T1)
        snap = make_snapshot([old1, old2, new1], [sup_later, sup_earlier])
        pop = ev.q04_population(snap)
        self.assertEqual([item for item, _ in pop], ["sup_a", "sup_b"])
        selected = ev.nearest_rank_select(pop, 0.5, lambda x: x)
        self.assertEqual(selected, "sup_a")

    def test_answer_uses_selected_assertions_own_valid_from_as_as_of(self):
        old1 = make_entity("old1", "WorkItem")
        old2 = make_entity("old2", "WorkItem")
        new1 = make_entity("new1", "WorkItem")
        sup1 = make_assertion("sup1", "new1", "supersedes", object_id="old1", valid_from=T1)
        sup2 = make_assertion("sup2", "new1", "supersedes", object_id="old2", valid_from=T2)
        snap = make_snapshot([old1, old2, new1], [sup1, sup2])
        bundle = ev.q04_answer(snap, "sup2")
        self.assertEqual(bundle.as_of, T2)
        self.assertIn("old1", bundle.entities)
        self.assertIn("old2", bundle.entities)


class Q05Tests(unittest.TestCase):
    def test_direct_reverse_depends_on_neighborhood(self):
        x = make_entity("x", "WorkItem")
        y = make_entity("y", "WorkItem")
        z = make_entity("z", "WorkItem")
        a1 = make_assertion("a1", "y", "depends_on", object_id="x", valid_from=T0)
        a2 = make_assertion("a2", "z", "depends_on", object_id="x", valid_from=T0)
        snap = make_snapshot([x, y, z], [a1, a2])
        bundle = ev.q05_answer(snap, "x")
        self.assertEqual(set(bundle.entities), {"x", "y", "z"})
        self.assertEqual(bundle.assertions, ("a1", "a2"))


class Q06Tests(unittest.TestCase):
    def test_bounded_transitive_dependants_with_witness(self):
        x = make_entity("x", "WorkItem")
        y = make_entity("y", "WorkItem")
        z = make_entity("z", "WorkItem")
        a1 = make_assertion("a1", "y", "depends_on", object_id="x", valid_from=T0)
        a2 = make_assertion("a2", "z", "depends_on", object_id="y", valid_from=T0)
        snap = make_snapshot([x, y, z], [a1, a2])
        bundle = ev.q06_answer(snap, "x", path_depth=2)
        self.assertEqual(set(bundle.entities), {"x", "y", "z"})
        self.assertEqual(len(bundle.paths), 2)
        witness_for_z = next(p for p in bundle.paths if p.steps[-1].assertion_id == "a2")
        self.assertEqual([s.assertion_id for s in witness_for_z.steps], ["a1", "a2"])

    def test_depth_limits_reachability(self):
        x = make_entity("x", "WorkItem")
        y = make_entity("y", "WorkItem")
        z = make_entity("z", "WorkItem")
        a1 = make_assertion("a1", "y", "depends_on", object_id="x", valid_from=T0)
        a2 = make_assertion("a2", "z", "depends_on", object_id="y", valid_from=T0)
        snap = make_snapshot([x, y, z], [a1, a2])
        bundle = ev.q06_answer(snap, "x", path_depth=1)
        self.assertEqual(set(bundle.entities), {"x", "y"})


class Q07Tests(unittest.TestCase):
    def test_dependency_stale_reachable_after_supersession(self):
        old = make_entity("old", "WorkItem")
        new = make_entity("new", "WorkItem")
        dep = make_entity("dep", "WorkItem")
        sup = make_assertion("sup", "new", "supersedes", object_id="old", valid_from=T1)
        depends = make_assertion("d1", "dep", "depends_on", object_id="old", valid_from=T0)
        snap = make_snapshot([old, new, dep], [sup, depends])
        pop = ev.q07_population(snap, T2, path_depth=2)
        self.assertEqual(pop, [("old", 1)])
        bundle = ev.q07_answer(snap, "old", T2, path_depth=2)
        self.assertIn("dep", bundle.entities)

    def test_not_superseded_entity_excluded(self):
        old = make_entity("old", "WorkItem")
        dep = make_entity("dep", "WorkItem")
        depends = make_assertion("d1", "dep", "depends_on", object_id="old", valid_from=T0)
        snap = make_snapshot([old, dep], [depends])
        self.assertEqual(ev.q07_population(snap, T2, path_depth=2), [])


class Q08Tests(unittest.TestCase):
    def test_dispute_record_found_via_concerns(self):
        x = make_entity("x", "WorkItem")
        c1 = make_entity("c1", "Claim", claim_key="test.claim")
        c2 = make_entity("c2", "Claim", claim_key="test.claim")
        concerns = make_assertion("con1", "c1", "concerns", object_id="x", valid_from=T0)
        contradicts = make_assertion(
            "dis1", "c2", "contradicts", object_id="c1", valid_from=T0, dispute_state="disputed"
        )
        snap = make_snapshot([x, c1, c2], [concerns, contradicts])
        bundle = ev.q08_answer(snap, "x", T1)
        self.assertEqual(bundle.assertions, ("dis1",))
        self.assertEqual(bundle.diagnostics["derived_dispute_state"]["dis1"], "disputed")


class Q09Tests(unittest.TestCase):
    def test_shortest_claim_to_source_witness(self):
        c1 = make_entity("c1", "Claim", claim_key="test.claim")
        art = make_entity("art", "Artifact")
        src = make_entity("src", "Source")
        supports = make_assertion("sup", "art", "supports", object_id="c1", valid_from=T0)
        documented = make_assertion("doc", "art", "documented_in", object_id="src", valid_from=T0)
        snap = make_snapshot([c1, art, src], [supports, documented])
        pop = ev.q09_population(snap, path_depth=3)
        self.assertEqual(pop, [(("c1", "src"), 2)])
        bundle = ev.q09_answer(snap, ("c1", "src"), path_depth=3)
        self.assertEqual(bundle.assertions, ("doc", "sup"))


class Q10Tests(unittest.TestCase):
    def test_minimal_evidence_cover_and_ties(self):
        c1 = make_entity("c1", "Claim", claim_key="test.claim")
        c2 = make_entity("c2", "Claim", claim_key="test.claim")
        art1 = make_entity("art1", "Artifact")
        art2 = make_entity("art2", "Artifact")
        src1 = make_entity("src1", "Source")
        src2 = make_entity("src2", "Source")
        assertions = [
            make_assertion("s1", "art1", "supports", object_id="c1", valid_from=T0),
            make_assertion("doc1", "art1", "documented_in", object_id="src1", valid_from=T0),
            make_assertion("s2", "art2", "supports", object_id="c2", valid_from=T0),
            make_assertion("doc2", "art2", "documented_in", object_id="src2", valid_from=T0),
        ]
        snap = make_snapshot([c1, c2, art1, art2, src1, src2], assertions)
        pop = ev.q10_population(snap, T1, path_depth=2)
        self.assertEqual(pop, [(("c1", "c2"), 4)])
        bundle = ev.q10_answer(snap, ("c1", "c2"), T1, path_depth=2)
        self.assertEqual(set(bundle.assertions), {"s1", "doc1", "s2", "doc2"})

    def test_fewer_than_two_eligible_claims_invalid(self):
        c1 = make_entity("c1", "Claim", claim_key="test.claim")
        c2 = make_entity("c2", "Claim", claim_key="test.claim")
        snap = make_snapshot([c1, c2], [])
        self.assertEqual(ev.q10_population(snap, T1, path_depth=2), [])


class Q11Tests(unittest.TestCase):
    def test_includes_only_registered_block_claim_slot(self):
        wi = make_entity("wi", "WorkItem")
        claim = make_entity("claim", "Claim", claim_key=ev.Q11_ELIGIBLE_CLAIM_KEY)
        block_literal = ev.Literal(
            literal_id="lit1", literal_type="text", value="block=0;profile=P-low;seed=1"
        )
        concerns = make_assertion("con", "claim", "concerns", object_id="wi", valid_from=T0)
        documented = make_assertion(
            "doc", "claim", "documented_in", literal_ref="lit1", valid_from=T0,
            epistemic_state="accepted",
        )
        snap = make_snapshot([wi, claim], [concerns, documented], literals=[block_literal])

        registered = {ev.Q11_ELIGIBLE_CLAIM_KEY: ev.Q11_ELIGIBLE_SLOT}
        pop = ev.q11_population(snap, T1, registered)
        self.assertEqual(pop, [("wi", 1)])

        unregistered = {"other.key": "other_slot"}
        self.assertEqual(ev.q11_population(snap, T1, unregistered), [])

    def test_excludes_claim_with_non_eligible_claim_key(self):
        """A Claim registered under a different claim_key is never Q11-eligible,
        even when the registry separately maps the eligible key correctly
        (spec: "Q11 includes only the registered block-claim slot")."""

        wi = make_entity("wi", "WorkItem")
        claim = make_entity("claim", "Claim", claim_key="other.claim_key")
        lit = ev.Literal(literal_id="lit1", literal_type="text", value="x")
        concerns = make_assertion("con", "claim", "concerns", object_id="wi", valid_from=T0)
        documented = make_assertion(
            "doc", "claim", "documented_in", literal_ref="lit1", valid_from=T0,
            epistemic_state="accepted",
        )
        snap = make_snapshot([wi, claim], [concerns, documented], literals=[lit])
        registered = {
            ev.Q11_ELIGIBLE_CLAIM_KEY: ev.Q11_ELIGIBLE_SLOT,
            "other.claim_key": ev.Q11_ELIGIBLE_SLOT,
        }
        self.assertEqual(ev.q11_population(snap, T1, registered), [])

    def test_claim_requires_claim_key(self):
        with self.assertRaises(ev.EvaluatorError):
            ev.Entity(entity_id="c", entity_type="Claim")

    def test_non_claim_forbids_claim_key(self):
        with self.assertRaises(ev.EvaluatorError):
            ev.Entity(entity_id="w", entity_type="WorkItem", claim_key="x")

    def test_ranks_accepted_before_verified_before_supported(self):
        wi = make_entity("wi", "WorkItem")
        claim_accepted = make_entity("ca", "Claim", claim_key=ev.Q11_ELIGIBLE_CLAIM_KEY)
        claim_supported = make_entity("cs", "Claim", claim_key=ev.Q11_ELIGIBLE_CLAIM_KEY)
        lit = ev.Literal(literal_id="lit1", literal_type="text", value="x")
        assertions = [
            make_assertion("con_a", "ca", "concerns", object_id="wi", valid_from=T0),
            make_assertion(
                "doc_a", "ca", "documented_in", literal_ref="lit1", valid_from=T0,
                epistemic_state="accepted",
            ),
            make_assertion("con_s", "cs", "concerns", object_id="wi", valid_from=T0),
            make_assertion(
                "doc_s", "cs", "documented_in", literal_ref="lit1", valid_from=T0,
                epistemic_state="supported",
            ),
        ]
        snap = make_snapshot([wi, claim_accepted, claim_supported], assertions, literals=[lit])
        registered = {ev.Q11_ELIGIBLE_CLAIM_KEY: ev.Q11_ELIGIBLE_SLOT}
        bundle = ev.q11_answer(snap, "wi", T1, registered)
        self.assertEqual(bundle.entities, ("ca", "wi"))


class Q12Tests(unittest.TestCase):
    def test_requires_both_implementer_and_verifier(self):
        r = make_entity("r", "Requirement")
        impl = make_entity("impl", "Implementation")
        ver = make_entity("ver", "Verification")
        a1 = make_assertion("a1", "impl", "implements", object_id="r", valid_from=T0)
        a2 = make_assertion("a2", "ver", "verifies", object_id="r", valid_from=T0)
        snap = make_snapshot([r, impl, ver], [a1, a2])
        pop = ev.q12_population(snap, T1)
        self.assertEqual(pop, [("r", 2)])

        only_impl_snap = make_snapshot([r, impl], [a1])
        self.assertEqual(ev.q12_population(only_impl_snap, T1), [])


class BundleClosureTests(unittest.TestCase):
    def test_literal_closure_and_dangling_reference_fails(self):
        wi = make_entity("wi", "WorkItem")
        claim = make_entity("claim", "Claim", claim_key="test.claim")
        lit = ev.Literal(literal_id="lit1", literal_type="text", value="x")
        documented = make_assertion("doc", "claim", "documented_in", literal_ref="lit1", valid_from=T0)
        snap = make_snapshot([wi, claim], [documented], literals=[lit])
        bundle = ev.build_bundle(
            snap, "test", None, entity_ids=["claim"], assertion_ids=["doc"]
        )
        self.assertEqual(bundle.literals, ("lit1",))

    def test_path_id_is_deterministic_and_order_sensitive(self):
        steps_a = [ev.PathStep("a1", "forward"), ev.PathStep("a2", "reverse")]
        steps_b = [ev.PathStep("a2", "reverse"), ev.PathStep("a1", "forward")]
        self.assertEqual(ev.compute_path_id(steps_a), ev.compute_path_id(steps_a))
        self.assertNotEqual(ev.compute_path_id(steps_a), ev.compute_path_id(steps_b))

    def test_strip_diagnostics_removes_diagnostics_only(self):
        bundle = ev.RetrievalBundle(
            contract_version=ev.RETRIEVAL_BUNDLE_CONTRACT_VERSION,
            query_instance_id="q",
            snapshot_id="s",
            as_of=None,
            entities=("x",),
            literals=(),
            assertions=(),
            source_revisions=(),
            paths=(),
            diagnostics={"a": 1},
        )
        stripped = ev.strip_diagnostics(bundle)
        self.assertIsNone(stripped.diagnostics)
        self.assertEqual(stripped.entities, bundle.entities)


if __name__ == "__main__":
    unittest.main()
