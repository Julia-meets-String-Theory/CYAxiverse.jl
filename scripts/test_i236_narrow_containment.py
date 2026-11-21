"""Issue #236 regressions: narrow containment rule (Owner decision D1(b))
and the fixed-singleton non-vanishing route (analyst A-6)."""

from fractions import Fraction as F
import itertools
import unittest

import numpy as np

import orientifold_general_l_geometry as g


SMOOTH_2D = [((1, 0), (0, 1)), ((0, 1), (-1, 0)), ((-1, 0), (0, -1)), ((0, -1), (1, 0))]
DIAG_2D = [((1, 1), (1, -1)), ((1, -1), (-1, -1)), ((-1, -1), (-1, 1)), ((-1, 1), (1, 1))]
MINUS_I = -np.eye(4, dtype=int)
IDENTITY = np.eye(4, dtype=int)
ZERO_T = (F(0),) * 4
BOUNDARY_L = np.diag([-1, -1, -1, 1]).astype(int)
BOUNDARY_T = (F(0), F(0), F(0), F(1, 2))
DV_DIAG = [
    (x, y, c, d)
    for (x, y) in [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for c, d in itertools.product((1, -1), repeat=2)
]
BINARY4 = {tuple(v) for v in itertools.product((0, 1), repeat=4)}
UNDETERMINED = "non_smooth_containment_undetermined"
DENSE = "eq_4.42_dense_torus_inversion_cancellation"
SINGLETON = "eq_4.42_fixed_singleton_nonvanishing_witness"
_FULL = object()


def product_fan(c2):
    cones = [
        tuple(sorted([(a, b, 0, 0) for a, b in c] + [(0, 0, s3, 0), (0, 0, 0, s4)]))
        for c in c2
        for s3 in (1, -1)
        for s4 in (1, -1)
    ]
    return cones, sorted({r for c in cones for r in c})


def full_dual(rays):
    return [
        q for q in itertools.product(range(-3, 4), repeat=4)
        if all(np.dot(r, q) >= -1 for r in rays)
    ]


def build(c2, L, t, lam, *, dual=_FULL, complete=False, with_fan_cones=True, keys=None):
    cones, rays = product_fan(c2)
    if dual is _FULL:
        dual = full_dual(rays)
    aux = g.build_auxiliary_fan(cones, L)
    if keys is None:
        keys = g._pointwise_invariant_cone_keys(cones, L)
    records = g._fixed_component_records(
        aux, L, t, lam,
        fixed_cone_keys=keys,
        dual_points=dual,
        ambient_rays=rays,
        fan_cones=cones if with_fan_cones else None,
        dual_points_complete=complete,
    )
    return records, aux, cones, rays


def labels(records):
    return {
        (tuple(tuple(r) for r in rec["sigma_rays"]), tuple(rec["nu"]["numerator"]), rec["nu"]["denominator"])
        for rec in records
    }


def torus_labels():
    return {((), nu, 1) for nu in BINARY4}


class DenseTorusInversionTests(unittest.TestCase):
    def test_diagonal_complete_support_contains_all_16_points(self):
        records, *_ = build(DIAG_2D, MINUS_I, ZERO_T, 1, complete=True)
        self.assertEqual(len(records), 16)
        self.assertEqual(labels(records), torus_labels())
        for rec in records:
            self.assertIs(rec["f_vanishes_identically"], True)
            self.assertEqual(rec["containment_method"], DENSE)
            self.assertEqual(rec["fixed_toric_dimension"], 0)
            self.assertNotIn("iff", rec["invariant_restricted_support"]["restriction_statement"])

    def test_smooth_control_same_16_labels(self):
        records, *_ = build(SMOOTH_2D, MINUS_I, ZERO_T, 1, complete=True)
        self.assertEqual(len(records), 16)
        self.assertEqual(labels(records), torus_labels())
        for rec in records:
            self.assertIs(rec["f_vanishes_identically"], True)
            self.assertEqual(rec["containment_method"], "eq_4.46_after_smooth_half_ray_eq_4.35")
            self.assertEqual(rec["fixed_toric_dimension"], 0)

    def _assert_undetermined_torus(self, records):
        self.assertEqual(len(records), 16)
        self.assertEqual(labels(records), torus_labels())
        for rec in records:
            self.assertIsNone(rec["f_vanishes_identically"])
            self.assertIsNone(rec["hypersurface_component_dimension"])
            self.assertEqual(rec["containment_method"], UNDETERMINED)

    def test_diagonal_without_completeness_is_undetermined(self):
        records, *_ = build(DIAG_2D, MINUS_I, ZERO_T, 1, complete=False)
        self._assert_undetermined_torus(records)

    def test_diagonal_pair_deleted_is_undetermined(self):
        _, rays = product_fan(DIAG_2D)
        dual = [q for q in full_dual(rays) if q not in ((1, 0, 0, 0), (-1, 0, 0, 0))]
        records, *_ = build(DIAG_2D, MINUS_I, ZERO_T, 1, dual=dual, complete=False)
        self._assert_undetermined_torus(records)

    def test_diagonal_without_fan_cones_is_undetermined(self):
        records, *_ = build(DIAG_2D, MINUS_I, ZERO_T, 1, complete=True, with_fan_cones=False)
        self._assert_undetermined_torus(records)

    def test_predicate_requires_exact_tuple(self):
        cones, rays = product_fan(DIAG_2D)
        support = g.invariant_restricted_monomial_support(
            full_dual(rays), rays, [], MINUS_I, ZERO_T, 1,
            fan_cones=cones,
            invariant_basis=g._integer_kernel_basis(np.eye(4, dtype=int) - MINUS_I),
        )
        ok = dict(matrix=MINUS_I, torus_shift=ZERO_T, lambda_f=1, sigma_rays=[],
                  ambient_dimension=0, support=support, dual_points_complete=True)
        self.assertTrue(g._dense_torus_inversion_cancellation_applies(**ok))
        inexact = MINUS_I.astype(float)
        inexact[0, 1] = 0.5
        for change in (
            {"lambda_f": F(3, 2)},
            {"lambda_f": True},
            {"matrix": inexact},
            {"torus_shift": (F(1, 2), F(0), F(0), F(0))},
            {"dual_points_complete": False},
            {"ambient_dimension": 1},
            {"sigma_rays": [(1, 1, 0, 0)]},
        ):
            self.assertFalse(g._dense_torus_inversion_cancellation_applies(**{**ok, **change}), change)


class ClassifierConsumerTests(unittest.TestCase):
    def test_undetermined_blocks_smooth_verdict(self):
        records, aux, *_ = build(DIAG_2D, MINUS_I, ZERO_T, 1, complete=False)
        result = g.classify_smoothness(
            MINUS_I, ZERO_T, 1, aux, records,
            {"non_smooth_facet_dual_vertices": set()}, DV_DIAG,
        )
        self.assertEqual(result["status"], "smoothness_verification_unavailable")
        self.assertEqual(result["verdict"], "not_verified")
        self.assertTrue(any("containment is undetermined" in r for r in result["reasons"]))

    def test_identity_sanity_contract_kept_and_records_preserved(self):
        # Analyst A-7: for exactly (I, 0, lambda_f=0) every point of X is fixed,
        # so the explicit identity-sanity contract stands; unresolved component
        # records keep their None value.
        records, *_ = build(DIAG_2D, MINUS_I, ZERO_T, 1, complete=False)
        cones, _ = product_fan(DIAG_2D)
        aux = g.build_auxiliary_fan(cones, IDENTITY)
        unresolved = records[0]
        result = g.classify_smoothness(IDENTITY, ZERO_T, 0, aux, [unresolved], {}, DV_DIAG)
        self.assertEqual(result["status"], "smooth")
        self.assertEqual(result["method"], "identity_sanity_contract")
        self.assertIsNone(unresolved["f_vanishes_identically"])
        other = g.classify_smoothness(IDENTITY, ZERO_T, 1, aux, [unresolved], {}, DV_DIAG)
        self.assertNotEqual(other["verdict"], "smooth")

    def test_identity_zero_dimensional_without_support_is_undetermined(self):
        cones, _ = product_fan(DIAG_2D)
        for lam in (0, 1):
            records, *_ = build(DIAG_2D, IDENTITY, ZERO_T, lam, dual=None, keys=[cones[0]])
            self.assertGreaterEqual(len(records), 1)
            for rec in records:
                self.assertEqual(rec["fixed_toric_dimension"], 0)
                self.assertIsNone(rec["f_vanishes_identically"])
                self.assertEqual(rec["containment_method"], UNDETERMINED)


class BoundaryControlTests(unittest.TestCase):
    def test_lambda0_full_support_is_undetermined(self):
        records, _, cones, rays = build(DIAG_2D, BOUNDARY_L, BOUNDARY_T, 0, complete=True)
        expected = {
            (((0, 0, 0, s),), (a, b, c, 0), 1)
            for s in (1, -1)
            for a, b, c in itertools.product((0, 1), repeat=3)
        }
        self.assertEqual(len(records), 16)
        self.assertEqual(labels(records), expected)
        for rec in records:
            self.assertEqual(rec["fixed_toric_dimension"], 0)
            self.assertIsNone(rec["f_vanishes_identically"])
            self.assertIsNone(rec["hypersurface_component_dimension"])
            self.assertEqual(rec["containment_method"], UNDETERMINED)
        for s in (1, -1):
            support = g.invariant_restricted_monomial_support(
                full_dual(rays), rays, [(0, 0, 0, s)], BOUNDARY_L, BOUNDARY_T, 0,
                fan_cones=cones,
                invariant_basis=g._integer_kernel_basis(np.eye(4, dtype=int) - BOUNDARY_L),
            )
            self.assertEqual(support["status"], "certified")
            self.assertEqual(support["chart"]["status"], "certified")
            self.assertTrue(support["support"])
            self.assertIsNone(g._singleton_nonvanishing_witness(BOUNDARY_L, support, [(0, 0, 0, s)]))

    def test_lambda1_singleton_witness_drops_points(self):
        dual = [(0, 0, 0, -1), (0, 0, 0, 1)]
        records, _, cones, rays = build(DIAG_2D, BOUNDARY_L, BOUNDARY_T, 1, dual=dual, complete=False)
        self.assertEqual(len(records), 0)
        for s, expected_phase in ((1, 0), (-1, 1)):
            support = g.invariant_restricted_monomial_support(
                dual, rays, [(0, 0, 0, s)], BOUNDARY_L, BOUNDARY_T, 1,
                fan_cones=cones,
                invariant_basis=g._integer_kernel_basis(np.eye(4, dtype=int) - BOUNDARY_L),
            )
            self.assertEqual(support["status"], "certified")
            self.assertEqual(support["chart"]["status"], "certified")
            witness = g._singleton_nonvanishing_witness(BOUNDARY_L, support, [(0, 0, 0, s)])
            self.assertIsNotNone(witness)
            self.assertEqual(witness["q"], [0, 0, 0, -s])
            self.assertEqual(witness["orbit_q"], [[0, 0, 0, -s]])
            self.assertEqual(witness["phase_exponent"], expected_phase)


class InheritedClassifierUndeterminedTests(unittest.TestCase):
    """Review v1 A1: the inherited classifier must not promote None to smooth."""

    def test_undetermined_containment_is_not_smooth(self):
        import inherited_orientifold_candidates as ioc

        component = {
            "sigma_rays": [],
            "sigma_dimension": 0,
            "fixed_toric_dimension": 0,
            "nu": [],
            "f_vanishes_identically": None,
            "containment_method": "non_smooth_containment_undetermined",
        }
        result = ioc.classify_smoothness(
            MINUS_I,
            ZERO_T,
            1,
            [],
            [component],
            {"non_smooth_facet_dual_vertices": set()},
            [],
        )
        self.assertTrue(result["dual_vertex_parity"]["available"])
        self.assertEqual(result["status"], "smoothness_verification_unavailable")
        self.assertEqual(result["verdict"], "not_verified")
        self.assertIn(
            "fixed-component containment is undetermined "
            "[non_smooth_containment_undetermined]",
            result["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
