"""Small exact-arithmetic contracts for the inherited fixed-locus kernel."""

import unittest

import numpy as np

import orientifold_general_l_geometry as general_l
import toric_fixed_component_euler as toric_euler


AMBIENT_RAYS = (
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)


class ExactFixedLocusKernelTests(unittest.TestCase):
    def test_integer_rank_and_lattice_index_do_not_use_float_rounding(self):
        self.assertEqual(general_l._exact_rank([[10**18, 0], [0, 1]]), 2)
        self.assertEqual(
            general_l._integer_lattice_index([[2, 0], [0, 1]]), 2
        )

    def test_auxiliary_fan_uses_the_exact_fixed_eigenspace(self):
        matrix = np.eye(4, dtype=int)
        matrix[[0, 1]] = matrix[[1, 0]]
        fan = general_l.build_auxiliary_fan((AMBIENT_RAYS,), matrix)
        rays = {tuple(ray) for cone in fan for ray in cone["rays"]}
        self.assertIn((1, 1, 0, 0), rays)
        self.assertIn((0, 0, 1, 0), rays)

    def test_covariant_support_keeps_a_singleton_monomial(self):
        support = general_l.invariant_restricted_monomial_support(
            [(1, -1, -1, -1)],
            AMBIENT_RAYS,
            AMBIENT_RAYS[1:],
            np.eye(4, dtype=int),
            (0, 0, 0, 0),
            0,
            fan_cones=(AMBIENT_RAYS,),
        )
        self.assertEqual(support["status"], "certified")
        self.assertFalse(support["restriction_identically_zero"])
        self.assertEqual(support["support"][0]["cox_exponents"], [2, 0, 0, 0])

    def test_contained_projective_line_has_exact_euler_two(self):
        certificate = {
            "status": "certified",
            "fixed_toric_dimension": 1,
            "quotient_rays": [[1], [-1]],
            "quotient_maximal_cones": [[[1]], [[-1]]],
        }
        self.assertEqual(
            toric_euler.component_euler_from_certificate(certificate, contained=True),
            2,
        )

    def test_missing_component_evidence_is_unavailable(self):
        result = toric_euler.exact_fixed_locus_euler(
            [], np.eye(4, dtype=int), [{
                "fixed_toric_dimension": 0,
                "sigma_rays": [],
                "nu": {"numerator": [0, 0, 0, 0], "denominator": 1},
            }],
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason_code"], "missing_invariant_restricted_support")


if __name__ == "__main__":
    unittest.main()
