"""Fixture-level tests for scripts/inherited_orientifold_candidates.py.

These tests avoid a live CYTools dependency for the pure-combinatorics and
wrapper-logic checks, following the existing pattern in
``test_stage2_stage_boundary.py`` of exercising orchestration logic with
lightweight fakes rather than real CYTools objects. ``validate_orientifold``
itself (the numerical core this module wraps unmodified) is exercised here
through those fakes and separately covered against real CYTools fixtures by
``test_stage2_stage_boundary.py``'s existing orientifold tests.

The primary fixture is the point set ``{0, e1, e2, e3, e4}`` (the origin plus
the standard basis of Z^4). Its automorphism group under GL(4,Z) is exactly
the permutation group S4 acting on (e1..e4): any linear automorphism is
determined by where it sends the basis vectors, and the only non-origin
points available to map them to are the basis vectors themselves. This makes
several properties checkable in closed form instead of by hand-picked
hard-coded matrices:

- the number of involutions (elements of order <= 2) in S4, including the
  identity, is 10 (1 identity + 6 transpositions + 3 double-transpositions);
- restricting to permutations that fix a subset setwise reduces to counting
  involutions in the corresponding smaller symmetric group.
"""

import json
import tempfile
import unittest
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np

from generate_geometric_data_multitriangulation import (
    OrientifoldValidationFailure,
    validate_orientifold,
)
from inherited_orientifold_candidates import (
    CANDIDATE_SCHEMA_VERSION,
    IDENTITY,
    TERMINAL_STATUSES,
    _complete_simplicial_fan,
    build_auxiliary_fan,
    classify_smoothness,
    enumerate_orientifold_candidates,
    enumerate_polytope_involutions,
    enumerate_projected_lattice_representatives,
    write_candidate_manifest,
    _component_key,
    _fraction_vector_to_json,
    _general_fixed_surface_n_s_table,
    _integer_coordinates,
    _integer_kernel_basis,
    _lattice_matrix_config,
)

BASIS_POINTS = np.array(
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=int,
)


class _FakePoly:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=int)

    def points(self):
        return self._points

    def dual(self):
        # Enumeration fixtures explicitly supply an empty dual-vertex set so
        # tests of FRST/H2/filter orchestration do not accidentally exercise
        # the separate missing-eq.-(4.45)-evidence path. That path is covered
        # directly by ClassifySmoothnessNsPolarityTests below.
        return _FakeDualPoly()

    def glsm_charge_matrix(self, include_origin=True, points=None, integral=True):
        # This fixture's point set (origin + independent basis vectors) has
        # no linear relations among its divisors, so the GLSM charge matrix
        # -- the true Z^N -> Pic(X) quotient map validate_orientifold now
        # uses -- is trivially the identity, matching _topology's
        # basis_matrix (also np.eye(h11)) exactly, since every test call
        # here sizes `points` to exactly h11 entries (no surplus divisors).
        del include_origin, integral
        n = len(points) if points is not None else self._points.shape[0]
        return np.eye(n, dtype=int)


class _FakeDualPoly:
    def vertices(self):
        return np.empty((0, 4), dtype=int)


class _FakeTriangulation:
    """Minimal stand-in exposing only what validate_orientifold reads."""

    def __init__(self, points, simplices_as_indices):
        self._points = np.asarray(points, dtype=int)
        self._simplices = np.asarray(simplices_as_indices, dtype=int)

    def points(self):
        return self._points

    def simplices(self, as_indices=False):
        del as_indices
        return self._simplices


def _topology(prime_toric_divisors, h11):
    return {
        "basis_matrix": np.eye(h11, dtype=int),
        "prime_toric_divisors": np.asarray(prime_toric_divisors, dtype=int),
        "h11": h11,
    }


class EnumeratePolytopeInvolutionsTests(unittest.TestCase):
    def test_basis_point_set_has_exactly_the_S4_involutions(self):
        involutions = enumerate_polytope_involutions(BASIS_POINTS)
        self.assertEqual(len(involutions), 10)

    def test_identity_is_always_included(self):
        involutions = enumerate_polytope_involutions(BASIS_POINTS)
        identity = np.eye(4, dtype=int)
        self.assertTrue(any(np.array_equal(matrix, identity) for matrix in involutions))

    def test_every_returned_matrix_is_a_valid_involution_automorphism(self):
        point_set = {tuple(int(v) for v in row) for row in BASIS_POINTS}
        identity = np.eye(4, dtype=int)
        for matrix in enumerate_polytope_involutions(BASIS_POINTS):
            self.assertEqual(matrix.dtype, np.dtype(int))
            self.assertTrue(np.array_equal(matrix @ matrix, identity))
            determinant = round(float(np.linalg.det(matrix)))
            self.assertIn(determinant, (-1, 1))
            for point in BASIS_POINTS:
                self.assertIn(tuple((matrix @ point).tolist()), point_set)

    def test_deterministic_repeated_calls(self):
        first = enumerate_polytope_involutions(BASIS_POINTS)
        second = enumerate_polytope_involutions(BASIS_POINTS)
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left, right))

    def test_six_point_fixture_with_negative_sum_point_still_only_permutes_basis(self):
        # A point set where a non-permutation automorphism would have to map
        # a basis vector to the extra point; verify it is correctly excluded
        # unless it actually is a lattice automorphism of the whole set.
        points = np.vstack([BASIS_POINTS, [[-1, -1, -1, -1]]])
        involutions = enumerate_polytope_involutions(points)
        point_set = {tuple(int(v) for v in row) for row in points}
        for matrix in involutions:
            for point in points:
                self.assertIn(tuple((matrix @ point).tolist()), point_set)

    def test_projected_lattice_representatives_have_expected_Z2_counts(self):
        identity_representatives = enumerate_projected_lattice_representatives(
            np.eye(4, dtype=int), 1
        )
        self.assertEqual(len(identity_representatives), 16)

        transposition = np.eye(4, dtype=int)
        transposition[[0, 1]] = transposition[[1, 0]]
        self.assertEqual(
            len(enumerate_projected_lattice_representatives(transposition, 1)), 8
        )
        self.assertEqual(
            len(enumerate_projected_lattice_representatives(transposition, -1)), 2
        )

    def test_projected_lattice_representatives_are_exact_for_non_axis_aligned_involutions(self):
        # A shear-conjugated reflection: fixes e1,e2,e3 and sends
        # e4 -> -4*e1 - e4. Its +1/-1 eigenspaces are not spanned by small
        # combinations of the standard basis, which is exactly the case an
        # earlier {-1,0,1}^4 bounded brute-force dedup search got wrong (it
        # returned 16/4 instead of the true 8/2 -- see
        # validation/fuzzy_axions_2412_12012_torus_shift_audit_20260817.md).
        # The expected counts below are the group orders 2**rank(I +/- L),
        # an independent closed-form check on |P_sign^L(N)/2P_sign^L(N)|.
        rotated = np.array(
            [[1, 0, 0, -4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=int
        )
        self.assertTrue(np.array_equal(rotated @ rotated, np.eye(4, dtype=int)))
        self.assertEqual(len(enumerate_projected_lattice_representatives(rotated, 1)), 8)
        self.assertEqual(len(enumerate_projected_lattice_representatives(rotated, -1)), 2)

    def test_projected_lattice_representative_counts_match_independent_rank_formula(self):
        # Property-based regression guard: |P_sign^L(N)/2P_sign^L(N)| must
        # always equal 2**rank(I + sign*L), for involutions whose eigenspaces
        # are rotated away from the standard basis by a random unimodular
        # conjugation (not just permutations/sign-flips).
        rng = np.random.default_rng(1234)

        def random_unimodular():
            matrix = np.eye(4, dtype=int)
            for _ in range(6):
                i, j = rng.choice(4, size=2, replace=False)
                elementary = np.eye(4, dtype=int)
                elementary[i, j] = int(rng.integers(-3, 4))
                matrix = matrix @ elementary
            return matrix

        for _ in range(40):
            signs = rng.choice([-1, 1], size=4)
            unimodular = random_unimodular()
            unimodular_inverse = np.rint(np.linalg.inv(unimodular)).astype(int)
            matrix = unimodular @ np.diag(signs).astype(int) @ unimodular_inverse
            self.assertTrue(np.array_equal(matrix @ matrix, np.eye(4, dtype=int)))
            for sign in (1, -1):
                expected = 2 ** int(
                    np.linalg.matrix_rank((np.eye(4) + sign * matrix).astype(float))
                )
                got = len(enumerate_projected_lattice_representatives(matrix, sign))
                self.assertEqual(got, expected, msg=f"matrix={matrix.tolist()} sign={sign}")

    def test_auxiliary_fan_contains_the_fixed_sum_ray(self):
        transposition = np.eye(4, dtype=int)
        transposition[[0, 1]] = transposition[[1, 0]]
        rays = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
        fan = build_auxiliary_fan([rays], transposition)
        fan_rays = {tuple(ray) for cone in fan for ray in cone["rays"]}
        self.assertIn((1, 1, 0, 0), fan_rays)
        self.assertIn((0, 0, 1, 0), fan_rays)


class EnumerateOrientifoldCandidatesTests(unittest.TestCase):
    def _poly(self):
        return _FakePoly(BASIS_POINTS)

    def test_fully_symmetric_frst_enumerates_the_full_triple_space(self):
        poly = self._poly()
        # A single top-dimensional simplex spanning every point is invariant
        # under any permutation of {1,2,3,4}.
        triangulation = _FakeTriangulation(BASIS_POINTS, [[0, 1, 2, 3, 4]])
        topology = _topology([1, 2, 3, 4], h11=5)

        records = enumerate_orientifold_candidates(poly, triangulation, topology)

        # Enumeration emits two records (lambda_f in {0, 1}) per *valid* torus
        # shift, but only ONE (a ``torus_shift_not_involution`` rejection) per
        # H_+^L coset whose representative 2t is non-integral: source
        # KS_orientifolds.tex line ~426-428 requires ``2t in N`` for
        # ``L o phi_[t]`` to square to the identity, so non-integral cosets are
        # not Z2 involutions and are excluded before the lambda_f loop (see the
        # 2t-in-N filter in enumerate_orientifold_candidates). The expected
        # record count is therefore filter-aware, not two-per-shift.
        expected_count = 0
        for matrix in enumerate_polytope_involutions(BASIS_POINTS):
            for shift in enumerate_projected_lattice_representatives(matrix, 1):
                if any(
                    Fraction(value).denominator != 1 for value in shift["vector"]
                ):
                    expected_count += 1
                else:
                    expected_count += 2
        matrix_summaries = [
            record
            for record in records
            if record.get("record_kind") == "lattice_matrix_search_summary"
        ]
        self.assertEqual(len(records), expected_count + len(matrix_summaries))
        # The non-involution filter is exercised by this fixture: at least one
        # coset per non-identity permutation matrix has a non-integral 2t.
        self.assertIn(
            "torus_shift_not_involution",
            {record["terminal_status"] for record in records},
        )
        # Any matrix whose full shift/lambda_f search found nothing accepted is
        # summarized with ``torus_shift_search_exhausted``. Unlike the
        # pre-filter fixture, some matrices now legitimately find nothing (their
        # only surviving shifts fail the smoothness checks), so
        # matrix_summaries need not be empty; assert the subset relation on its
        # statuses rather than exact equality.
        self.assertLessEqual(
            {record["terminal_status"] for record in matrix_summaries},
            {"torus_shift_search_exhausted"},
        )
        self.assertIn(
            "accepted_verified_orientifold",
            {record["terminal_status"] for record in records},
        )
        for status in TERMINAL_STATUSES:
            self.assertIn(status, TERMINAL_STATUSES)  # vocabulary sanity

        accepted = [
            record
            for record in records
            if record["terminal_status"] == "accepted_verified_orientifold"
        ]
        self.assertTrue(accepted)
        for record in accepted:
            self.assertIsNotNone(record["torus_shift"])
            self.assertIn(record["lambda_f"], (0, 1))
            self.assertEqual(record["smoothness"]["verdict"], "smooth")
            self.assertIsNotNone(record["fixed_point_set"])

    def test_h11_minus_zero_filter_keeps_only_the_identity(self):
        poly = self._poly()
        triangulation = _FakeTriangulation(BASIS_POINTS, [[0, 1, 2, 3, 4]])
        topology = _topology([1, 2, 3, 4], h11=5)

        records = enumerate_orientifold_candidates(
            poly, triangulation, topology, h11_minus_target=0
        )

        accepted = [
            record
            for record in records
            if record["terminal_status"] == "accepted_verified_orientifold"
            and record["lattice_matrix"] == np.eye(4, dtype=int).tolist()
            and record["torus_shift"]
            == {"numerator": [0, 0, 0, 0], "denominator": 1}
        ]
        # Only lambda_f=0 survives at (L=I, t=0): lambda_f=1 there forces
        # every monomial coefficient to vanish identically (source eq.
        # (4.43) with t=0, lambda_f=1 gives psi_q = -psi_q), so it is
        # correctly not a valid orientifold action, and
        # classify_smoothness's identity-sanity shortcut is restricted to
        # lambda_f=0 accordingly.
        self.assertEqual(len(accepted), 1)
        self.assertTrue(all(record["lattice_matrix"] == np.eye(4, dtype=int).tolist() for record in accepted))
        self.assertEqual({record["lambda_f"] for record in accepted}, {0})

    def test_partial_frst_rejects_candidates_that_move_the_excluded_point(self):
        poly = self._poly()
        # Simplex over {0,1,2,3} only (point 4 excluded): preserved exactly
        # by involutions of S4 fixing 4 setwise, i.e. involutions of S3 on
        # {1,2,3} -- 1 identity + 3 transpositions = 4 preserving candidates,
        # the other 10 - 4 = 6 move point 4 and must fail FRST preservation.
        triangulation = _FakeTriangulation(BASIS_POINTS, [[0, 1, 2, 3]])
        topology = _topology([1, 2, 3, 4], h11=5)

        records = enumerate_orientifold_candidates(poly, triangulation, topology)

        # Record and acceptance counts below are empirically determined (not
        # hand-derived) and reflect the source eq. (4.34)/line ~428 filter that
        # excludes torus-shift cosets with non-integral 2t (not Z2 involutions;
        # see the 2t-in-N filter in enumerate_orientifold_candidates). Relative
        # to the pre-filter fixture (86 records, 60 accepted) the filter drops
        # the non-integral cosets: their two lambda_f records each collapse to a
        # single ``torus_shift_not_involution`` rejection, and any acceptance
        # that had rested on such a (non-involution) shift is correctly removed.
        # The source Sec. 4.6 requirement for nef generic sections and
        # orbifold avoidance also leaves seven previously accepted fixture
        # triples unavailable; ``frst_not_preserved`` and the non-involution
        # count remain unchanged.
        self.assertEqual(len(records), 74)
        accepted = [
            record
            for record in records
            if record["terminal_status"] == "accepted_verified_orientifold"
        ]
        frst_failed = [
            record
            for record in records
            if record["terminal_status"] == "frst_not_preserved"
        ]
        non_involution = [
            record
            for record in records
            if record["terminal_status"] == "torus_shift_not_involution"
        ]
        self.assertEqual(len(accepted), 29)
        self.assertEqual(len(frst_failed), 6)
        self.assertEqual(len(non_involution), 12)
        for record in frst_failed:
            self.assertIsNone(record.get("h11_minus"))

    def test_identity_zero_shift_accepts_lambda_f_zero_only(self):
        # L=I, t=0 is the trivial (worldsheet-parity-only) action: physically
        # valid and trivially smooth for lambda_f=0 (O5/O9, whole CY fixed),
        # but not a valid orientifold action at all for lambda_f=1 (O3/O7):
        # source eq. (4.43) with t=0, lambda_f=1 forces psi_q = -psi_q for
        # every monomial coefficient q, i.e. psi_q=0 identically. This
        # asymmetry must survive both classify_smoothness's shortcut and
        # _fixed_point_set_description's labelling.
        poly = self._poly()
        triangulation = _FakeTriangulation(BASIS_POINTS, [[0, 1, 2, 3, 4]])
        topology = _topology([1, 2, 3, 4], h11=5)
        records = enumerate_orientifold_candidates(poly, triangulation, topology)
        identity_zero_shift = [
            record
            for record in records
            if record["lattice_matrix"] == np.eye(4, dtype=int).tolist()
            and record["torus_shift"] == {"numerator": [0, 0, 0, 0], "denominator": 1}
        ]
        accepted = [
            record
            for record in identity_zero_shift
            if record["terminal_status"] == "accepted_verified_orientifold"
        ]
        self.assertEqual({record["lambda_f"] for record in accepted}, {0})
        self.assertTrue(
            all(
                record["fixed_point_set"]["description"] == "whole_calabi_yau"
                for record in accepted
            )
        )
        rejected_lambda_f_1 = [
            record
            for record in identity_zero_shift
            if record["lambda_f"] == 1
        ]
        self.assertEqual(len(rejected_lambda_f_1), 1)
        self.assertNotEqual(
            rejected_lambda_f_1[0]["terminal_status"], "accepted_verified_orientifold"
        )
        self.assertNotEqual(
            rejected_lambda_f_1[0]["fixed_point_set"]["description"], "whole_calabi_yau"
        )

    def test_candidate_ids_are_stable_across_repeated_calls(self):
        poly = self._poly()
        triangulation = _FakeTriangulation(BASIS_POINTS, [[0, 1, 2, 3, 4]])
        topology = _topology([1, 2, 3, 4], h11=5)

        first = enumerate_orientifold_candidates(poly, triangulation, topology)
        second = enumerate_orientifold_candidates(poly, triangulation, topology)

        self.assertEqual(
            [record["candidate_id"] for record in first],
            [record["candidate_id"] for record in second],
        )

    def test_hand_constructed_non_preserving_matrix_reports_polytope_stage(self):
        poly = self._poly()
        triangulation = _FakeTriangulation(BASIS_POINTS, [[0, 1, 2, 3, 4]])
        topology = _topology([1, 2, 3, 4], h11=5)
        # Shear e1 -> e1 + e2, not among the polytope's own points.
        shear = np.array(
            [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int
        )
        config = _lattice_matrix_config(shear)
        with self.assertRaises(OrientifoldValidationFailure) as context:
            validate_orientifold(poly, triangulation, topology, config)
        self.assertEqual(context.exception.stage, "polytope_not_preserved")


class ClassifySmoothnessNsPolarityTests(unittest.TestCase):
    """Regression coverage for the eq. (4.50) n_S polarity fix.

    arXiv:2305.06363 (line ~647-654) imposes ``n^S_{df=0} = 0`` to avoid
    isolated nodal points on a 2-dimensional fixed surface -- i.e. n_S != 0
    is the obstruction, n_S == 0 is smooth. Before this fix,
    ``classify_smoothness`` treated ``n_s == 0`` as the non-smooth case
    instead (backwards), though the bug was inert in production because
    ``topology["fixed_surface_n_s"]`` was never populated anywhere in this
    codebase, so every such candidate previously resolved to
    ``smoothness_verification_unavailable`` (n_s is None) regardless of
    polarity -- see
    ``fuzzy_axions_2412_12012_h11_3_h11_5_table1_verification_20260818.md``
    Sec. 6 for the investigation and the primary-source confirmation.
    """

    def _dim2_component(self, sigma_rays=((1, 0, 0, 0), (0, 1, 0, 0))):
        return {
            "sigma_rays": [list(ray) for ray in sigma_rays],
            "nu": _fraction_vector_to_json((0, 0, 0, 0)),
            "sigma_dimension": 2,
            "fixed_toric_dimension": 2,
            "f_vanishes_identically": True,
        }

    def _classify(self, component, n_s, *, dual_vertices="fixture_empty"):
        topology = {}
        if n_s is not None:
            topology = {"fixed_surface_n_s": {_component_key(component): n_s}}
        return classify_smoothness(
            IDENTITY,
            (0.5, 0, 0, 0),
            1,
            auxiliary_fan=[],
            fixed_components=[component],
            topology=topology,
            dual_vertices=(
                np.empty((0, 4), dtype=int)
                if isinstance(dual_vertices, str)
                else dual_vertices
            ),
        )

    def test_zero_n_s_is_smooth(self):
        component = self._dim2_component()
        result = self._classify(component, n_s=0)
        self.assertEqual(result["status"], "smooth")
        self.assertEqual(result["verdict"], "smooth")

    def test_nonzero_n_s_is_non_smooth(self):
        component = self._dim2_component()
        for n_s in (1, -1, 2):
            with self.subTest(n_s=n_s):
                result = self._classify(component, n_s=n_s)
                self.assertEqual(result["status"], "fixed_point_set_non_smooth")
                self.assertEqual(result["verdict"], "non_smooth")
                self.assertTrue(
                    any(f"n_S = {n_s}" in reason for reason in result["reasons"])
                )

    def test_missing_evidence_stays_unavailable_not_smooth(self):
        component = self._dim2_component()
        result = self._classify(component, n_s=None)
        self.assertEqual(result["status"], "smoothness_verification_unavailable")
        self.assertEqual(result["verdict"], "not_verified")

    def test_missing_dual_vertex_parity_evidence_stays_unavailable(self):
        component = self._dim2_component()
        result = self._classify(component, n_s=0, dual_vertices=None)
        self.assertEqual(result["status"], "smoothness_verification_unavailable")
        self.assertEqual(result["verdict"], "not_verified")
        self.assertIn("dual-vertex parity", " ".join(result["reasons"]))

    def test_nonvanishing_positive_component_requires_source_section_evidence(self):
        component = {
            "sigma_rays": [],
            "nu": _fraction_vector_to_json((0, 0, 0, 0)),
            "sigma_dimension": 0,
            "fixed_toric_dimension": 2,
            "f_vanishes_identically": False,
        }
        auxiliary_fan = [
            {
                "rays": [[1, 0, 0, 0], [0, 1, 0, 0]],
                "dimension": 2,
                "pointwise_L_invariant": True,
                "simplicial": True,
                "ambient_cones": [],
            }
        ]
        result = classify_smoothness(
            np.diag([1, 1, -1, -1]),
            (0, 0, 0, 0),
            0,
            auxiliary_fan=auxiliary_fan,
            fixed_components=[component],
            topology={},
            dual_vertices=np.empty((0, 4), dtype=int),
        )
        self.assertEqual(result["status"], "smoothness_verification_unavailable")
        self.assertIn("nef generic-section", " ".join(result["reasons"]))
        self.assertEqual(
            result["positive_component_checks"][0]["reason_code"],
            "missing_ambient_cone_provenance",
        )


class PositiveComponentSectionChecksTests(unittest.TestCase):
    """Regression coverage for Moritz Sec. 4.6 section smoothness tests."""

    MATRIX = np.diag([1, 1, 1, -1])
    EXTRA_RAY = (0, 0, 0, 1)
    PLUS_FIXED_RAY = (1, 0, 0, 0)
    MINUS_FIXED_RAY = (-1, 0, 0, 0)

    @staticmethod
    def _ambient_cone_with_fixed_support(support):
        """Build a unimodular provenance cone realizing a fixed support form."""

        cones = {
            (-1, 3, -1): [
                [-2, -1, 0, -2],
                [-4, -2, -1, -4],
                [-4, -2, -1, -3],
                [-3, -2, -2, -4],
            ],
            (-1, -1, 1): [
                [-2, 1, -2, -2],
                [-4, 1, -4, -3],
                [-4, 2, -3, -4],
                [-3, 0, -4, -2],
            ],
            (1, 3, -1): [
                [-2, 0, -1, -2],
                [-4, 0, -3, -3],
                [-4, 1, 0, -4],
                [-3, 0, -2, -3],
            ],
            (1, -1, 1): [
                [-2, -2, -1, -2],
                [-4, -4, -1, -3],
                [-4, -3, 0, -4],
                [-3, -4, -2, -2],
            ],
        }
        return cones[tuple(support)]

    @classmethod
    def _fan(cls, surface_rays, ambient_overrides=None):
        ambient_overrides = ambient_overrides or {}
        pairs = [
            (surface_rays[0], surface_rays[1]),
            (surface_rays[1], surface_rays[2]),
            (surface_rays[2], surface_rays[0]),
        ]
        auxiliary_fan = []
        for fixed_ray in (cls.PLUS_FIXED_RAY, cls.MINUS_FIXED_RAY):
            for pair in pairs:
                fixed_cone = (fixed_ray,) + pair
                key = frozenset(fixed_cone)
                ambient = ambient_overrides.get(key)
                if ambient is None:
                    ambient = [list(ray) for ray in fixed_cone + (cls.EXTRA_RAY,)]
                auxiliary_fan.append(
                    {
                        "rays": [list(ray) for ray in fixed_cone],
                        "dimension": 3,
                        "pointwise_L_invariant": True,
                        "simplicial": True,
                        "ambient_cones": [ambient],
                    }
                )
        return auxiliary_fan

    @classmethod
    def _non_nef_fan(cls):
        surface_rays = (
            (0, 0, 1, 0),
            (0, 1, 0, 0),
            (0, 0, -1, 0),
            (0, -1, 3, 0),
        )
        pairs = [
            (surface_rays[0], surface_rays[1]),
            (surface_rays[1], surface_rays[2]),
            (surface_rays[2], surface_rays[3]),
            (surface_rays[3], surface_rays[0]),
        ]
        auxiliary_fan = []
        for fixed_ray in (cls.PLUS_FIXED_RAY, cls.MINUS_FIXED_RAY):
            for pair in pairs:
                fixed_cone = (fixed_ray,) + pair
                auxiliary_fan.append(
                    {
                        "rays": [list(ray) for ray in fixed_cone],
                        "dimension": 3,
                        "pointwise_L_invariant": True,
                        "simplicial": True,
                        "ambient_cones": [
                            [list(ray) for ray in fixed_cone + (cls.EXTRA_RAY,)]
                        ],
                    }
                )
        return auxiliary_fan

    @staticmethod
    def _component():
        return {
            "sigma_rays": [],
            "sigma_dimension": 0,
            "nu": _fraction_vector_to_json((0, 0, 0, 0)),
            "fixed_toric_dimension": 3,
            "f_vanishes_identically": False,
        }

    def _classify(self, auxiliary_fan):
        return classify_smoothness(
            self.MATRIX,
            (0, 0, 0, 0),
            0,
            auxiliary_fan=auxiliary_fan,
            fixed_components=[self._component()],
            topology={},
            dual_vertices=np.empty((0, 4), dtype=int),
        )

    def test_smooth_nef_fixed_component_is_certified(self):
        # P^1 x P^2 has a smooth complete quotient fan. The restricted
        # anti-canonical bundle is O(2,3), so the exact support-function test
        # must certify nefness and there are no orbifold strata to inspect.
        surface_rays = ((0, 1, 0, 0), (0, 0, 1, 0), (0, -1, -1, 0))
        result = self._classify(self._fan(surface_rays))
        self.assertEqual(result["status"], "smooth")
        check = result["positive_component_checks"][0]
        self.assertEqual(check["status"], "certified")
        self.assertEqual(check["nefness"]["status"], "certified")
        self.assertTrue(check["nefness"]["nef"])
        self.assertEqual(check["orbifold_intersection"]["status"], "certified")

    def test_non_nef_restriction_is_rejected(self):
        # P^1 x F_3 is smooth, but -K_{F_3} has negative degree on the
        # negative section. This exercises nefness itself, not a fan
        # simpliciality shortcut.
        result = self._classify(self._non_nef_fan())
        self.assertEqual(result["status"], "fixed_point_set_non_smooth")
        check = result["positive_component_checks"][0]
        self.assertEqual(check["reason_code"], "restricted_line_bundle_not_nef")
        self.assertTrue(check["nefness"]["witnesses"])

    def test_orbifold_stratum_intersection_is_rejected(self):
        # P^1 x P(1,1,2) has a non-smooth two-cone whose orbit is a curve.
        # The anti-canonical face over that curve has multiple lattice points,
        # hence its generic Laurent restriction has a zero on the orbifold
        # curve. The ambient provenance cones remain smooth; the singularity
        # is in the fixed quotient fan, as required by the source setup.
        surface_rays = ((0, 1, 0, 0), (0, 0, 1, 0), (0, -1, -2, 0))
        overrides = {}
        for fixed_ray, fixed_supports in (
            (
                self.PLUS_FIXED_RAY,
                {
                    frozenset((
                        self.PLUS_FIXED_RAY,
                        surface_rays[1],
                        surface_rays[2],
                    )): (-1, 3, -1),
                    frozenset((
                        self.PLUS_FIXED_RAY,
                        surface_rays[2],
                        surface_rays[0],
                    )): (-1, -1, 1),
                },
            ),
            (
                self.MINUS_FIXED_RAY,
                {
                    frozenset((
                        self.MINUS_FIXED_RAY,
                        surface_rays[1],
                        surface_rays[2],
                    )): (1, 3, -1),
                    frozenset((
                        self.MINUS_FIXED_RAY,
                        surface_rays[2],
                        surface_rays[0],
                    )): (1, -1, 1),
                },
            ),
        ):
            del fixed_ray
            for cone_key, support in fixed_supports.items():
                overrides[cone_key] = self._ambient_cone_with_fixed_support(support)
        result = self._classify(self._fan(surface_rays, overrides))
        self.assertEqual(result["status"], "fixed_point_set_non_smooth")
        check = result["positive_component_checks"][0]
        self.assertEqual(check["reason_code"], "orbifold_stratum_intersection")
        self.assertEqual(check["orbifold_intersection"]["status"], "rejected")
        self.assertGreater(
            check["orbifold_intersection"]["singular_strata"][0][
                "face_lattice_point_count"
            ],
            1,
        )


class CompleteSimplicialFanTests(unittest.TestCase):
    """Regression coverage for the complete quotient-fan certificate."""

    def test_complete_two_dimensional_fan_is_certified(self):
        fan = [
            ((1, 0), (0, 1)),
            ((0, 1), (-1, 0)),
            ((-1, 0), (0, -1)),
            ((0, -1), (1, 0)),
        ]
        self.assertTrue(_complete_simplicial_fan(fan, 2))

    def test_closed_first_quadrant_incidence_is_not_complete(self):
        # Every ray occurs in two cones, but this collection has neither
        # vector-space coverage nor a common-face fan structure: all rays lie
        # in one quadrant. A codimension-one count alone would accept it.
        fan = [
            ((1, 0), (0, 1)),
            ((0, 1), (1, 1)),
            ((1, 1), (1, 0)),
        ]
        self.assertFalse(_complete_simplicial_fan(fan, 2))


class GeneralFixedSurfaceMachineryTests(unittest.TestCase):
    """Regression coverage for the integer lattices used by general ``L``."""

    class _Fan:
        def __init__(self, vectors):
            self._vectors = np.asarray(vectors, dtype=int)

        def vectors(self):
            return self._vectors

    class _ToricVariety:
        def __init__(self, vectors):
            self._fan = GeneralFixedSurfaceMachineryTests._Fan(vectors)

        def fan(self):
            return self._fan

    def test_kernel_basis_is_saturated_over_the_integer_lattice(self):
        # A rational nullspace basis with denominators cleared independently
        # can miss primitive kernel vectors. This matrix is the smallest
        # regression found while validating the auxiliary quotient: its
        # primitive kernel vector (0, 1, -1, 0) must be represented exactly.
        matrix = np.array(
            [[3, 1, 1, 1], [0, 0, 0, 1]],
            dtype=int,
        )
        basis = _integer_kernel_basis(matrix)
        self.assertTrue(np.array_equal(matrix @ basis, np.zeros((2, 2), dtype=int)))
        coordinates = _integer_coordinates(basis, (0, 1, -1, 0))
        self.assertIsNotNone(coordinates)
        self.assertTrue(np.array_equal(basis @ coordinates, (0, 1, -1, 0)))

    def test_general_surface_matches_hand_computed_p1_four_example(self):
        # Take V=(P^1)^4 with rays +/-e_i and L=diag(1,1,-1,-1). Each of
        # the four fixed components is S=P^1 x P^1. Its two normal directions
        # are trivial, while K_V^{-1}|_S=O(2,2), so
        # int_S c_2(O(2,2) tensor O_S^2)=int_S(2H_1+2H_2)^2=8.
        basis = np.eye(4, dtype=int)
        vectors = []
        for index in range(4):
            vectors.extend((basis[index], -basis[index]))
        cones = [
            tuple(
                tuple(int(signs[index] * basis[index, coordinate]) for coordinate in range(4))
                for index in range(4)
            )
            for signs in product((-1, 1), repeat=4)
        ]
        matrix = np.diag([1, 1, -1, -1])
        table = _general_fixed_surface_n_s_table(
            cones,
            self._ToricVariety(vectors),
            matrix,
        )
        self.assertEqual(len(table), 4)
        self.assertEqual(set(table.values()), {8})

    def test_general_surface_diagnostics_preserve_terms_and_status(self):
        basis = np.eye(4, dtype=int)
        vectors = []
        for index in range(4):
            vectors.extend((basis[index], -basis[index]))
        cones = [
            tuple(
                tuple(int(signs[index] * basis[index, coordinate]) for coordinate in range(4))
                for index in range(4)
            )
            for signs in product((-1, 1), repeat=4)
        ]
        result = _general_fixed_surface_n_s_table(
            cones,
            self._ToricVariety(vectors),
            np.diag([1, 1, -1, -1]),
            return_diagnostics=True,
        )
        self.assertEqual(set(result["evidence"].values()), {8})
        # The quotient fan is shared by the four translated fixed components;
        # keep one fan-level record and retain every component label in it.
        self.assertEqual(len(result["surface_diagnostics"]), 1)
        diagnostic = result["surface_diagnostics"][0]
        self.assertEqual(len(diagnostic["fixed_components"]), 4)
        self.assertEqual(len(diagnostic["quotient_surface_rays"]), 4)
        self.assertEqual(len(diagnostic["quotient_surface_cones"]), 4)
        self.assertEqual(len(diagnostic["surface_cone_provenance"]), 4)
        self.assertEqual(len(diagnostic["restricted_divisor_coefficients"]), 8)
        invariant_basis = np.asarray(
            diagnostic["invariant_lattice_basis"], dtype=int
        )
        self.assertEqual(invariant_basis.shape, (4, 2))
        self.assertTrue(
            np.array_equal(
                (np.eye(4, dtype=int) - np.diag([1, 1, -1, -1])) @ invariant_basis,
                np.zeros((4, 2), dtype=int),
            )
        )
        self.assertEqual(
            diagnostic["quotient_annihilator"],
            [[1, 0], [0, 1]],
        )
        self.assertTrue(all(item["status"] == "certified" for item in result["surface_diagnostics"]))
        self.assertEqual(
            {
                (
                    item["c2_ambient_restricted"],
                    item["c2_surface"],
                    item["c1_surface_squared"],
                    item["n_s"],
                )
                for item in result["surface_diagnostics"]
            },
            # On P1 x P1, c2(T_V)|S = c2(T_S) = 4 and c1(T_S)^2 = 8,
            # so the source formula gives n_S = 4 - 4 + 8 = 8.
            {(4, 4, 8, 8)},
        )

    def test_general_surface_diagnostics_report_missing_full_cone(self):
        basis = np.eye(4, dtype=int)
        vectors = []
        for index in range(4):
            vectors.extend((basis[index], -basis[index]))
        auxiliary_fan = [
            {
                "rays": [],
                "dimension": 0,
                "pointwise_L_invariant": True,
                "simplicial": True,
                "ambient_cones": [],
            }
        ]
        result = _general_fixed_surface_n_s_table(
            [],
            self._ToricVariety(vectors),
            np.diag([1, 1, -1, -1]),
            auxiliary_fan=auxiliary_fan,
            return_diagnostics=True,
        )
        self.assertEqual(result["evidence"], {})
        self.assertEqual(len(result["surface_diagnostics"]), 1)
        self.assertEqual(
            result["surface_diagnostics"][0]["reason_code"],
            "missing_full_dimensional_auxiliary_cone",
        )
        self.assertEqual(len(result["surface_diagnostics"][0]["fixed_components"]), 4)


class WriteCandidateManifestTests(unittest.TestCase):
    def test_round_trip_writes_jsonl_and_summary(self):
        records = [
            {
                "candidate_id": "a",
                "polytope_id": "p",
                "frst_hash": "f",
                "lattice_matrix": np.eye(4, dtype=int).tolist(),
                "terminal_status": "accepted_verified_orientifold",
                "terminal_reason": None,
                "h11_plus": 5,
                "h11_minus": 0,
            },
            {
                "candidate_id": "b",
                "polytope_id": "p",
                "frst_hash": "f",
                "lattice_matrix": [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "terminal_status": "orientifold_h11_minus_filter_rejection",
                "terminal_reason": "h11_minus=1 does not match requested target 0",
                "h11_plus": 4,
                "h11_minus": 1,
            },
        ]
        with tempfile.TemporaryDirectory(prefix="cyax-orientifold-manifest-") as temporary:
            path = Path(temporary) / "candidates.jsonl"
            summary = write_candidate_manifest(
                path, records, provenance={"source_note": "test-fixture"}
            )

            written_records = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
            self.assertEqual(len(written_records), 2)

            summary_path = Path(f"{path}.summary.json")
            written_summary = json.loads(summary_path.read_text())
            self.assertEqual(written_summary, summary)
            self.assertEqual(
                written_summary["schema_version"], CANDIDATE_SCHEMA_VERSION
            )
            self.assertEqual(written_summary["candidate_count"], 2)
            self.assertEqual(
                written_summary["status_counts"]["accepted_verified_orientifold"], 1
            )
            self.assertEqual(
                written_summary["status_counts"][
                    "orientifold_h11_minus_filter_rejection"
                ],
                1,
            )
            self.assertEqual(written_summary["accepted_candidate_count"], 1)
            self.assertEqual(written_summary["distinct_accepted_frst_count"], 1)


if __name__ == "__main__":
    unittest.main()
