"""Focused regression tests for the specialist toy QED assignment contract."""

import unittest
import tempfile
from fractions import Fraction

import h5py
import numpy as np

from qed_divisor_assignment import (
    QEDAssignmentFailure,
    classify_qed_leading_status,
    compute_leading_rank_order,
    prime_divisor_intersection_graph,
    select_qed_divisor,
    write_visible_sector_hdf5,
)


def _reference_rank(rows):
    """Rebuild-from-scratch exact-fraction rank: the pre-optimization oracle."""
    basis = {}
    for vector in rows:
        work = [Fraction(int(value)) for value in vector]
        for pivot, row in sorted(basis.items()):
            factor = work[pivot]
            if factor:
                work = [value - factor * row[index] for index, value in enumerate(work)]
        pivot = next((index for index, value in enumerate(work) if value), None)
        if pivot is not None:
            scale = work[pivot]
            basis[pivot] = tuple(value / scale for value in work)
    return len(basis)


def _reference_leading_rank_order(q, priorities):
    """Oracle: the original no-early-exit, rebuild-from-scratch selection."""
    order = sorted(range(q.shape[1]), key=lambda index: (-float(priorities[index]), index))
    selected = []
    rank = 0
    for index in order:
        next_rank = _reference_rank([q[:, i] for i in selected] + [q[:, index]])
        if next_rank > rank:
            selected.append(index)
            rank = next_rank
    return {"order": order, "selected": selected, "rank": rank}


class QEDAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([1, 2, 3])
        self.stable = ((0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3))
        self.charges = np.asarray([[1, 0], [0, 1], [1, 1]])
        self.neighbors, self.evidence = prime_divisor_intersection_graph(
            self.labels, np.asarray([[1, 2, 3]])
        )
        self.orientifold = {
            "requested": True,
            "status": "validated",
            "involution_type": "O3/O7",
            "h11_minus": 0,
            "prime_divisor_image_indices": np.arange(3),
        }

    def select(self, **kwargs):
        values = dict(
            policy="intersecting_d7",
            selection_policy="uniform_eligible",
            qcd_divisor_index=0,
            prime_toric_divisors=self.labels,
            prime_divisor_labels=self.stable,
            prime_divisor_charges_array=self.charges,
            prime_divisor_volumes=np.asarray([40.0, 3.0, 4.0]),
            neighbors=self.neighbors,
            intersection_evidence=self.evidence,
            orientifold=self.orientifold,
            effective_seed=17,
        )
        values.update(kwargs)
        return select_qed_divisor(**values)

    def test_uniform_selection_is_seeded_and_records_pool(self):
        first = self.select()
        second = self.select()
        self.assertEqual(first["qed_divisor_index"], second["qed_divisor_index"])
        self.assertEqual(first["candidate_pool_indices"], [1, 2])
        self.assertEqual(first["qed_divisor_index_user"], first["qed_divisor_index"] + 1)
        self.assertTrue(first["intersection_evidence"])

    def test_explicit_index_is_one_based(self):
        with self.assertRaises(QEDAssignmentFailure) as context:
            self.select(selection_policy="explicit", qed_divisor_index_user=0)
        self.assertEqual(context.exception.category, "invalid_explicit_index")

    def test_nonzero_h11_minus_is_recorded_but_not_a_visible_sector_rejection(self):
        orientifold = dict(self.orientifold)
        orientifold.update({"h11_plus": 1, "h11_minus": 1})
        result = self.select(orientifold=orientifold)
        self.assertEqual(result["orientifold_h11_plus"], 1)
        self.assertEqual(result["orientifold_h11_minus"], 1)

    def test_fan_invariant_orientifold_is_sufficient_without_kaehler_subspace_check(self):
        orientifold = dict(self.orientifold, status="fan_invariant")
        result = self.select(orientifold=orientifold)
        self.assertEqual(result["candidate_pool_indices"], [1, 2])

    def test_potential_source_has_exact_charge_match(self):
        status = classify_qed_leading_status(
            np.asarray([[1, 0, 1], [0, 1, 1]]),
            np.asarray([[1.0, 1.0, 1.0], [-2.0, -1.0, -3.0]]),
            1,
        )
        self.assertIn(status["status"], {"leading", "dependent"})
        self.assertEqual(status["method"], "exact_rational_incremental_rank")

    def test_leading_rank_order_matches_rebuild_from_scratch_oracle(self):
        # classify_qed_leading_status used to rebuild its exact-fraction basis
        # from scratch for every candidate column (O(rank^2) per candidate,
        # repeated once per column). compute_leading_rank_order instead
        # carries a persistent basis across candidates and exits once rank
        # saturates. Both must select the same leading columns in the same
        # order and reach the same rank, since neither change can alter which
        # vectors are linearly independent or when.
        rng = np.random.default_rng(2026)
        for trial in range(20):
            rows, cols = 5, 14
            q = rng.integers(-3, 4, size=(rows, cols))
            priorities = rng.random(cols)
            got = compute_leading_rank_order(q, priorities)
            want = _reference_leading_rank_order(q, priorities)
            self.assertEqual(got["order"], want["order"], msg=f"trial {trial}")
            self.assertEqual(got["selected"], want["selected"], msg=f"trial {trial}")
            self.assertEqual(got["rank"], want["rank"], msg=f"trial {trial}")

    def test_leading_rank_order_early_exit_matches_full_scan_with_duplicate_and_dependent_columns(self):
        # Columns 2 and 3 duplicate columns 0 and 1; column 4 is their sum
        # (dependent). Rank saturates at 2 well before all 5 columns are
        # scanned, exercising the early-exit path against a case with
        # explicit duplicates and a linear combination.
        q = np.asarray([[1, 0, 1, 0, 1], [0, 1, 0, 1, 1]])
        priorities = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
        got = compute_leading_rank_order(q, priorities)
        want = _reference_leading_rank_order(q, priorities)
        self.assertEqual(got, want)
        self.assertEqual(got["rank"], 2)
        self.assertEqual(got["selected"], [0, 1])

    def test_classify_qed_leading_status_accepts_precomputed_order(self):
        # A caller working through one geometry's whole assignment pool can
        # compute the geometry-only leading_rank_order once and pass it back
        # in on every per-assignment call instead of paying the elimination
        # cost again for every QED index in the pool.
        q = np.asarray([[1, 0, 1], [0, 1, 1]])
        l = np.asarray([[1.0, 1.0, 1.0], [-2.0, -1.0, -3.0]])
        precomputed = compute_leading_rank_order(q, l[1, :])

        direct = classify_qed_leading_status(q, l, 1)
        via_cache = classify_qed_leading_status(q, l, 1, leading_rank_order=precomputed)
        self.assertEqual(direct, via_cache)

    def test_visible_sector_hdf5_persists_divisor_labels(self):
        assignment = self.select()
        with tempfile.NamedTemporaryFile(suffix=".h5") as temporary:
            with h5py.File(temporary.name, "w") as handle:
                write_visible_sector_hdf5(handle.create_group("visible"), assignment)
            with h5py.File(temporary.name, "r") as handle:
                visible = handle["visible"]
                np.testing.assert_array_equal(
                    visible["qcd_divisor_label"][()], self.stable[0]
                )
                np.testing.assert_array_equal(
                    visible["qed_divisor_label"][()],
                    self.stable[assignment["qed_divisor_index"]],
                )
                np.testing.assert_array_equal(
                    visible["candidate_pool_labels"][()],
                    np.asarray([self.stable[index] for index in [1, 2]]),
                )


if __name__ == "__main__":
    unittest.main()
