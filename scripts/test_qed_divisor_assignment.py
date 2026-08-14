"""Focused regression tests for the specialist toy QED assignment contract."""

import unittest

import numpy as np

from qed_divisor_assignment import (
    QEDAssignmentFailure,
    classify_qed_leading_status,
    prime_divisor_intersection_graph,
    select_qed_divisor,
)


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

    def test_potential_source_has_exact_charge_match(self):
        status = classify_qed_leading_status(
            np.asarray([[1, 0, 1], [0, 1, 1]]),
            np.asarray([[1.0, 1.0, 1.0], [-2.0, -1.0, -3.0]]),
            1,
        )
        self.assertIn(status["status"], {"leading", "dependent"})
        self.assertEqual(status["method"], "exact_rational_incremental_rank")


if __name__ == "__main__":
    unittest.main()
