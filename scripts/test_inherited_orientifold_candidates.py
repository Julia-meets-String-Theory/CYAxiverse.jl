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
from glimmers_raw_frst import compute_polytope_normal_form_id, stable_hash
from inherited_orientifold_candidates import (
    CANDIDATE_SCHEMA_VERSION,
    IDENTITY,
    TERMINAL_STATUSES,
    _complete_simplicial_fan,
    _ambient_cartier_data,
    _ambient_anticanonical_cartier_data,
    _fixed_component_records,
    _half_ray_shortcut_proof,
    _nu_equal_mod_span,
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
    _integer_lattice_membership,
    _lattice_matrix_config,
    _pointwise_invariant_cone_keys,
    _primitive_quotient_vector,
    _sublattice_is_saturated,
    facets_with_non_smooth_cones,
)


class AmbientAnticanonicalCartierDataTests(unittest.TestCase):
    """Check exact Cartier data on smooth and Gorenstein singular cones."""

    def test_integral_cartier_data_accepts_nonunimodular_gorenstein_cone(self):
        # The fourth ray gives determinant -2, but all four rays lie on the
        # integral support hyperplane m dot p = -1 for m=(-1,-1,-1,-1).
        cone = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 1, -2],
            ],
            dtype=int,
        )
        self.assertEqual(
            _ambient_anticanonical_cartier_data(cone),
            (-1, -1, -1, -1),
        )

    def test_nonintegral_cartier_data_remains_unavailable(self):
        # The same determinant-two shape with a non-Gorenstein fourth ray
        # produces a half-integral local support function and must not pass.
        cone = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 1, 0, -2],
            ],
            dtype=int,
        )
        self.assertIsNone(_ambient_anticanonical_cartier_data(cone))

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

    def normal_form(self):
        # Deterministic stand-in for cytools.Polytope.normal_form(): the
        # fixture's origin-plus-basis point set is already a canonical
        # representative, so its normal form is itself.
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

        first_candidates = [
            record for record in first if record.get("record_kind") == "candidate"
        ]
        second_candidates = [
            record for record in second if record.get("record_kind") == "candidate"
        ]
        self.assertEqual(
            [record["candidate_id"] for record in first_candidates],
            [record["candidate_id"] for record in second_candidates],
        )
        self.assertGreater(len(first_candidates), 1)
        for record in first_candidates:
            if record.get("lambda_f") not in (0, 1):
                continue
            shifts = {
                tuple(shift["binary_source"]): shift
                for shift in enumerate_projected_lattice_representatives(
                    np.asarray(record["lattice_matrix"], dtype=int), 1
                )
            }
            shift = shifts[tuple(record["torus_shift_binary_source"])]
            expected = stable_hash(
                [
                    record["matrix_id"],
                    tuple(shift["numerator"]),
                    int(record["lambda_f"]),
                ]
            )
            self.assertEqual(record["candidate_id"], expected)

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

    def test_missing_non_smooth_facet_evidence_stays_unavailable(self):
        component = self._dim2_component()
        result = classify_smoothness(
            IDENTITY,
            (0.5, 0, 0, 0),
            1,
            auxiliary_fan=[],
            fixed_components=[component],
            topology={
                "fixed_surface_n_s": {_component_key(component): 0},
                "non_smooth_facet_dual_vertices": None,
            },
            dual_vertices=np.array([[1, 0, 0, 0]], dtype=int),
        )
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
        # Fixed components are now labelled by pointwise-``L``-invariant cones
        # of the original fan (source eqs. (4.33)--(4.35)), not by the
        # auxiliary fan ``Sigma_L``. Supply the trivial invariant cone ``()``
        # as the fixed-cone label explicitly; the bare ``auxiliary_fan`` below
        # deliberately contains no full-dimensional cone, so this fixed cone
        # must be reported as missing its containing full-dimensional cone
        # rather than silently certified.
        result = _general_fixed_surface_n_s_table(
            [],
            self._ToricVariety(vectors),
            np.diag([1, 1, -1, -1]),
            auxiliary_fan=auxiliary_fan,
            fixed_cone_keys=((),),
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


class PointwiseInvariantConeKeyTests(unittest.TestCase):
    """Source-faithful sigma-label universe from Moritz eqs. (4.33)--(4.35).

    ``_pointwise_invariant_cone_keys`` must return only faces of the ORIGINAL
    ambient fan ``Sigma`` whose every ray is individually fixed
    (``L @ ray == ray``).  The auxiliary fan ``Sigma_L`` (source eq. (4.26))
    carries additional intersection rays and is a strictly larger set, so it
    is not a valid component-label universe.  These two tests pin the two
    ways that distinction is load-bearing.
    """

    # Swap the first two ambient coordinates: e1 <-> e2 are moved, e3, e4 fixed.
    SWAP01 = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int
    )

    @staticmethod
    def _rays_in_keys(keys):
        return {ray for cone_key in keys for ray in cone_key}

    @staticmethod
    def _rays_in_auxiliary(fan):
        return {tuple(int(v) for v in ray) for cone in fan for ray in cone["rays"]}

    def test_coordinate_swap_auxiliary_ray_excluded_as_sigma_label(self):
        """A ray moved by L is never a sigma label; the auxiliary ray is larger.

        Regression guard: labelling fixed components off ``Sigma_L`` (the
        auxiliary fan) instead of the pointwise-fixed faces of ``Sigma`` would
        admit the intersection ray (1,1,0,0), which is NOT a face of the
        original fan and whose generating rays (1,0,0,0),(0,1,0,0) are not even
        individually L-fixed.
        """
        cone = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        keys = _pointwise_invariant_cone_keys([cone], self.SWAP01)
        key_rays = self._rays_in_keys(keys)

        # e1 is moved by the swap (L @ e1 = e2), so it cannot label a fixed
        # component and must not appear in any returned cone key.
        self.assertNotIn((1, 0, 0, 0), key_rays)
        self.assertNotIn((0, 1, 0, 0), key_rays)
        # The genuinely fixed rays e3, e4 do appear, individually and as the
        # maximal fixed face {e3, e4} (cone keys are stored with sorted rays).
        self.assertIn((0, 0, 1, 0), key_rays)
        self.assertIn((0, 0, 0, 1), key_rays)
        self.assertIn(((0, 0, 0, 1), (0, 0, 1, 0)), keys)
        # Every returned face is pointwise fixed under L (the defining property).
        for cone_key in keys:
            for ray in cone_key:
                self.assertEqual(
                    tuple((self.SWAP01 @ np.asarray(ray, dtype=int)).tolist()),
                    tuple(int(v) for v in ray),
                )

        # Contrast: the auxiliary fan Sigma_L contains the extra intersection
        # ray (1,1,0,0) (= e1+e2, the fixed direction inside the moved plane),
        # which is a strictly larger ray set than the sigma-label universe.
        auxiliary_rays = self._rays_in_auxiliary(
            build_auxiliary_fan([cone], self.SWAP01)
        )
        self.assertIn((1, 1, 0, 0), auxiliary_rays)
        self.assertNotIn((1, 1, 0, 0), key_rays)
        self.assertTrue(key_rays < auxiliary_rays)

    def test_nonprimitive_swapped_pair_scale_uses_original_ray_vectors(self):
        """Sigma labels keep the ORIGINAL lattice scale, not the primitive Sigma_L ray.

        The fixed subspace of the swap forces the auxiliary intersection ray of
        the nonprimitive original ray (2,2,0,0) to be its primitivisation
        (1,1,0,0).  ``_pointwise_invariant_cone_keys`` must yield the
        original-scale (2,2,0,0) (it labels components by original fan rays),
        while ``build_auxiliary_fan`` primitivises to (1,1,0,0).

        Regression guard: sourcing sigma labels from the primitivised auxiliary
        rays would silently rescale a component label, changing the eq. (4.35)
        integrality datum ``sum(ray/2)``.
        """
        # (2,2,0,0) is individually fixed by the swap and is nonprimitive.
        self.assertTrue(
            np.array_equal(self.SWAP01 @ np.array([2, 2, 0, 0], dtype=int), [2, 2, 0, 0])
        )
        cone = ((2, 2, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        keys = _pointwise_invariant_cone_keys([cone], self.SWAP01)
        key_rays = self._rays_in_keys(keys)

        # Original scale is retained; the primitivised auxiliary ray never
        # appears as a sigma label.
        self.assertIn((2, 2, 0, 0), key_rays)
        self.assertNotIn((1, 1, 0, 0), key_rays)
        # Every returned sigma ray is literally a ray of the original cone.
        self.assertTrue(key_rays <= set(cone))

        # The auxiliary fan does primitivise: it carries (1,1,0,0), not (2,2,0,0).
        auxiliary_rays = self._rays_in_auxiliary(
            build_auxiliary_fan([cone], self.SWAP01)
        )
        self.assertIn((1, 1, 0, 0), auxiliary_rays)
        self.assertNotIn((2, 2, 0, 0), auxiliary_rays)


class FixedComponentContainmentReductionTests(unittest.TestCase):
    """Exact phase reduction and containment in ``_fixed_component_records``.

    Source eqs. (4.30)--(4.35): reduce labels modulo the rational span of the
    vanishing cone, then remove a component when a proper face carries the
    compatible phase label. The proper face is the larger orbit-closure
    component and is retained.
    """

    def test_strict_subset_same_nu_retains_only_the_maximal_stratum(self):
        """The strict-subset sigma is kept; the superset cone is discarded.

        Regression guard: without the exact-``nu`` containment reduction both
        the proper face {(2,0,0,0)} and its superset {(2,0,0,0),(0,2,0,0)}
        would be persisted, double-counting a single fixed stratum and its
        proper face.
        """
        # L = I gives a single canonical nu = 0, so both cones share it exactly.
        subset_cone = ((2, 0, 0, 0),)
        superset_cone = ((2, 0, 0, 0), (0, 2, 0, 0))
        torus_shift = tuple(Fraction(0) for _ in range(4))
        records = _fixed_component_records(
            [],
            IDENTITY,
            torus_shift,
            0,
            fixed_cone_keys=(subset_cone, superset_cone),
        )

        retained_ray_sets = {
            frozenset(tuple(ray) for ray in record["sigma_rays"])
            for record in records
        }
        # Only the strict subset (the proper face / maximal stratum) survives.
        self.assertEqual(retained_ray_sets, {frozenset({(2, 0, 0, 0)})})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sigma_rays"], [[2, 0, 0, 0]])
        self.assertEqual(records[0]["sigma_dimension"], 1)
        # The superset ray set is not present under the same nu.
        self.assertNotIn(
            frozenset({(2, 0, 0, 0), (0, 2, 0, 0)}), retained_ray_sets
        )

    def test_distinct_labels_equivalent_modulo_sigma_span(self):
        """A minus-eigenspace label can be distinct but equivalent on sigma."""
        matrix = np.diag([1, -1, 1, 1])
        zero = (Fraction(0),) * 4
        e2 = (Fraction(0), Fraction(1), Fraction(0), Fraction(0))
        e1 = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
        sigma = ((0, 1, 0, 0),)

        self.assertTrue(_nu_equal_mod_span(zero, e2, sigma))
        self.assertFalse(_nu_equal_mod_span(zero, e2, ((1, 0, 0, 0),)))
        self.assertFalse(_nu_equal_mod_span(zero, e1, sigma))

        records = _fixed_component_records(
            [],
            matrix,
            zero,
            0,
            fixed_cone_keys=(sigma,),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["fixed_component_integrality"]["method"],
            "general_quotient_lattice_eq_4.30",
        )

    def test_genuinely_distinct_labels_are_retained(self):
        """Labels outside the sigma span describe separate components."""
        matrix = np.diag([1, -1, 1, 1])
        sigma = ((1, 0, 0, 0),)
        records = _fixed_component_records(
            [],
            matrix,
            (Fraction(0),) * 4,
            0,
            fixed_cone_keys=(sigma,),
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {
                tuple(Fraction(value) for value in record["nu"]["numerator"])
                for record in records
            },
            {(0, 0, 0, 0), (0, 1, 0, 0)},
        )

    def test_containment_uses_proper_face_and_span_equivalence(self):
        """A sigma component is removed when its proper face contains it."""
        matrix = np.diag([1, -1, 1, 1])
        face = ((0, 1, 0, 0),)
        sigma = ((0, 1, 0, 0), (0, 0, 1, 0))
        records = _fixed_component_records(
            [],
            matrix,
            (Fraction(0),) * 4,
            0,
            fixed_cone_keys=(face, sigma),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sigma_rays"], [[0, 1, 0, 0]])

    def test_conjugated_involution_fixed_components_use_exact_anti_lattice(self):
        """Verify a shear-conjugated reflection's non-axis-aligned label."""
        # This is a conjugate of diag(1, 1, 1, -1), not a permutation/sign
        # change of basis.  Its anti-invariant lattice contains (2,0,0,1),
        # which a coordinate-wise representative search can miss.
        matrix = np.array(
            [[1, 0, 0, -4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dtype=int,
        )
        zero = (Fraction(0),) * 4
        records = _fixed_component_records(
            [],
            matrix,
            zero,
            0,
            fixed_cone_keys=(((1, 0, 0, 0),),),
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(
            {
                tuple(record["nu"]["numerator"])
                for record in records
            },
            {(0, 0, 0, 0), (2, 0, 0, 1)},
        )
        self.assertTrue(
            all(
                record["fixed_component_integrality"]["method"]
                == "general_quotient_lattice_eq_4.30"
                for record in records
            )
        )

    def test_saturation_sensitive_span_merges_nonprimitive_phase_labels(self):
        """Use rational span to identify labels across a saturation defect."""
        matrix = np.diag([-1, 1, 1, 1])
        zero = (Fraction(0),) * 4
        e1 = (Fraction(1), Fraction(0), Fraction(0), Fraction(0))
        sigma = ((2, 0, 0, 0),)

        # e1 is not in Z*(2 e1), but it is in span_Q(2 e1).  This is the
        # saturation defect that must not split one fixed component into two.
        self.assertFalse(
            _sublattice_is_saturated(np.asarray(sigma, dtype=int).T)
        )
        self.assertFalse(
            _integer_lattice_membership(
                np.asarray(sigma, dtype=int).T,
                np.asarray(e1, dtype=int),
            )
        )
        self.assertTrue(_nu_equal_mod_span(zero, e1, sigma))

        records = _fixed_component_records(
            [],
            matrix,
            zero,
            0,
            fixed_cone_keys=(sigma,),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sigma_rays"], [[2, 0, 0, 0]])


class FixedComponentIntegralityPathTests(unittest.TestCase):
    """Use eq. (4.35) only when retained fan data prove its hypotheses."""

    def test_smooth_fixture_uses_half_ray_shortcut(self):
        cone = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
        auxiliary_fan = build_auxiliary_fan([cone], IDENTITY)
        proven, reason = _half_ray_shortcut_proof(auxiliary_fan, IDENTITY, ())
        self.assertTrue(proven)
        self.assertEqual(reason, "smooth_sigma_and_normal_directions_certified")
        records = _fixed_component_records(
            auxiliary_fan,
            IDENTITY,
            (Fraction(0),) * 4,
            0,
            fixed_cone_keys=((),),
        )
        self.assertEqual(
            records[0]["fixed_component_integrality"]["method"],
            "smooth_half_ray_eq_4.35",
        )

    def test_non_smooth_fixture_uses_general_quotient_condition(self):
        cone = (
            (1, 0, 0, 0),
            (1, 3, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
        sigma = cone[:2]
        auxiliary_fan = build_auxiliary_fan([cone], IDENTITY)
        proven, reason = _half_ray_shortcut_proof(auxiliary_fan, IDENTITY, sigma)
        self.assertFalse(proven)
        self.assertEqual(reason, "sigma_cone_not_smooth")
        records = _fixed_component_records(
            auxiliary_fan,
            IDENTITY,
            (Fraction(0),) * 4,
            0,
            fixed_cone_keys=(sigma,),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["fixed_component_integrality"]["method"],
            "general_quotient_lattice_eq_4.30",
        )


class FixedComponentZeroDimensionalFilterTests(unittest.TestCase):
    """Zero-dimensional non-vanishing strata are not persisted.

    A component with ``ambient_dimension == 0`` (fixed-subspace dimension equal
    to sigma dimension) and ``vanishes_identically == False``
    (``(sigma_dimension + lambda_f)`` even) has no hypersurface component: the
    generic section does not vanish at the point, so there is nothing to
    persist.  The odd-parity (vanishing) zero-dimensional stratum is kept.
    """

    def test_even_parity_zero_dim_stratum_is_filtered_odd_parity_kept(self):
        """Even-parity zero-dim stratum dropped; odd-parity zero-dim stratum kept.

        Regression guard: persisting the empty non-vanishing intersection would
        emit a spurious point-component record with no hypersurface.
        """
        # L = I: fixed_subspace_dimension = rank(2I) = 4. A 4-ray cone has
        # sigma_dimension 4, so ambient_dimension = 0.
        full_cone = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        # torus_shift chosen so eq. (4.35) integrality holds: t + sum(ray/2)
        # = (1/2,...) + (1/2,...) = (1,1,1,1).
        torus_shift = tuple(Fraction(1, 2) for _ in range(4))

        # lambda_f = 0 -> (4 + 0) even -> non-vanishing zero-dim stratum -> dropped.
        even_parity = _fixed_component_records(
            [], IDENTITY, torus_shift, 0, fixed_cone_keys=(full_cone,)
        )
        self.assertEqual(even_parity, [])

        # lambda_f = 1 -> (4 + 1) odd -> vanishing zero-dim stratum -> retained.
        odd_parity = _fixed_component_records(
            [], IDENTITY, torus_shift, 1, fixed_cone_keys=(full_cone,)
        )
        self.assertEqual(len(odd_parity), 1)
        record = odd_parity[0]
        self.assertEqual(record["sigma_dimension"], 4)
        self.assertEqual(record["fixed_toric_dimension"], 0)
        self.assertTrue(record["f_vanishes_identically"])


class CompleteSimplicialFanMalformedTests(unittest.TestCase):
    """``_complete_simplicial_fan`` rejects malformed 2D fans.

    The bounded quotient-surface certificate must reject any fan that is not a
    complete, simplicial, smooth fan covering the whole plane.  A codimension-
    one incidence count alone is insufficient, so distinct malformation modes
    are each rejected while a genuine complete smooth fan is certified.
    """

    def test_malformed_two_dimensional_fans_are_rejected(self):
        """Two distinct malformed 2D fans are False; the complete fan is True.

        Regression guard: a first-quadrant-only cover (strict subset of the
        plane) and a fan with a non-primitive ray must not be treated as a
        complete simplicial fan.
        """
        # (a) First-quadrant only: a single cone covering a strict subset of
        # the plane; the origin is not interior to the ray hull.
        first_quadrant_only = [((1, 0), (0, 1))]
        self.assertFalse(_complete_simplicial_fan(first_quadrant_only, 2))

        # (b) A non-primitive ray (2,0): the fan looks complete by incidence
        # but the ray gcd is 2, so it is not a valid smooth-fan ray.
        non_primitive_ray = [
            ((2, 0), (0, 1)),
            ((0, 1), (-1, 0)),
            ((-1, 0), (0, -1)),
            ((0, -1), (2, 0)),
        ]
        self.assertFalse(_complete_simplicial_fan(non_primitive_ray, 2))

        # (c) Overlapping cones sharing a ray but not a common face.
        overlapping_cones = [((1, 0), (0, 1)), ((1, 0), (1, 1))]
        self.assertFalse(_complete_simplicial_fan(overlapping_cones, 2))

        # A genuine complete smooth 2D fan: four cones around the origin from
        # rays (1,0),(0,1),(-1,0),(0,-1).
        complete_fan = [
            ((1, 0), (0, 1)),
            ((0, 1), (-1, 0)),
            ((-1, 0), (0, -1)),
            ((0, -1), (1, 0)),
        ]
        self.assertTrue(_complete_simplicial_fan(complete_fan, 2))


class FacetsWithNonSmoothConesPairingTests(unittest.TestCase):
    """Facet/cone pairing semantics behind ``facets_with_non_smooth_cones``.

    The function flags a dual vertex ``q`` when a non-simplicial/non-smooth
    cone meets its dual facet ``{m : <q,m> = -1}`` along ANY boundary ray, via
    ``np.any(rays @ vertex == -1)`` (eq. (4.45), line ~629 second clause).  The
    old ``np.all`` semantics required EVERY ray to pair -1 and so missed a cone
    meeting the facet only along a proper face.

    Building a full CYTools ``poly``/``triangulation`` with a genuine
    non-smooth cone intersecting a dual facet along one ray is impractical as a
    unit test, so this tests the two layers the function actually depends on:
    (1) the exact integer pairing predicate ``np.any(... == -1)`` versus
    ``np.all(...)`` on a cone that meets the facet along a single ray, and
    (2) the graceful ``None`` return when dual extraction is unavailable.
    """

    def test_single_ray_facet_pairing_is_flagged_by_any_not_all(self):
        """One ray pairing -1 among several flags the facet (np.any), not np.all.

        Regression guard for the eq. (4.45) fix from ``np.all`` to ``np.any``:
        a cone meeting the dual facet along a proper face (exactly one ray with
        ``<q, ray> = -1``) must be flagged.
        """
        # A 3-ray cone; only the first ray pairs -1 with the dual vertex.
        rays = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=int)
        vertex = np.array([-1, 0, 0, 0], dtype=int)
        pairings = rays @ vertex
        # Exact integer arithmetic; no float tolerance.
        self.assertEqual(tuple(int(v) for v in pairings), (-1, 0, 0))
        self.assertTrue(bool(np.any(pairings == -1)))  # new (correct) semantics
        self.assertFalse(bool(np.all(pairings == -1)))  # old (missed) semantics

    def test_returns_none_when_dual_extraction_is_unavailable(self):
        """Graceful None when ``poly.dual()`` raises; the fan is never touched.

        Regression guard: a missing/invalid dual must yield ``None`` (evidence
        unavailable), not an exception, and must short-circuit before reading
        the triangulation fan.
        """

        class _RaisingDualPoly:
            def points(self):
                return np.zeros((1, 4), dtype=int)

            def dual(self):
                raise RuntimeError("dual polytope unavailable")

        class _UnreachableTriangulation:
            def fan(self):
                raise AssertionError(
                    "fan() must not be reached when the dual is unavailable"
                )

        self.assertIsNone(
            facets_with_non_smooth_cones(
                _RaisingDualPoly(), _UnreachableTriangulation()
            )
        )


class NonOrthogonalBasisInvolutionTests(unittest.TestCase):
    """``enumerate_polytope_involutions`` uses the correct multiplication order.

    The involution is recovered as ``L = image_matrix @ basis_inverse`` with
    ``basis_matrix = basis_points.T``.  With a NON-orthogonal selected basis,
    the transposed order ``basis_inverse @ image_matrix`` gives a different,
    wrong matrix, so an order bug is observable here (it is invisible for the
    standard-basis fixture, whose ``basis_matrix`` is the identity).
    """

    # A shear-reflection: fixes e1,e2,e3, sends e4 -> -2 e1 - e4. Non-permutation.
    L = np.array(
        [[1, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=int
    )
    # Point set (origin + five points) closed under L. The first four
    # independent non-origin points are f1=(1,1,0,0), e2, e3, e4, whose
    # transpose basis_matrix is lower-triangular and NON-orthogonal
    # (f1 . e2 = 1). The moved point e4 -> (-2,0,0,-1) makes the set asymmetric
    # enough that basis_matrix @ L leaves the point set: a transposed
    # multiplication order would fail to produce L at all.
    POINTS = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-2, 0, 0, -1],
        ],
        dtype=int,
    )

    def test_non_orthogonal_basis_recovers_the_correct_involutions(self):
        """Every returned L is a valid involution mapping the set to itself.

        Regression guard: the specific shear-reflection L is present ONLY with
        the correct ``image_matrix @ basis_inverse`` order; the transposed
        order (``basis_inverse @ image_matrix``) would need image columns equal
        to ``basis_matrix @ L``, one of which leaves the point set, so it would
        never yield L on this non-orthogonal basis.
        """
        from sympy import Matrix as _SympyMatrix

        self.assertTrue(np.array_equal(self.L @ self.L, np.eye(4, dtype=int)))
        point_set = {tuple(int(v) for v in row) for row in self.POINTS}
        involutions = enumerate_polytope_involutions(self.POINTS)

        # The order-sensitive involution must be recovered exactly.
        self.assertTrue(any(np.array_equal(matrix, self.L) for matrix in involutions))

        identity = np.eye(4, dtype=int)
        for matrix in involutions:
            self.assertEqual(matrix.dtype, np.dtype(int))
            # Exact integer determinant in {-1, 1} (no float tolerance).
            determinant = int(_SympyMatrix(matrix.tolist()).det())
            self.assertIn(determinant, (-1, 1))
            self.assertTrue(np.array_equal(matrix @ matrix, identity))
            # L must actually map the point set onto itself.
            self.assertEqual(
                {tuple((matrix @ point).tolist()) for point in self.POINTS},
                point_set,
            )


class PolytopeNormalFormIdentityTests(unittest.TestCase):
    """The geometry identity keyed into records is lattice-invariant."""

    def test_identifier_is_row_order_invariant(self):
        points = [[1, 0, 0, 0], [0, 1, 0, 0], [-1, -1, -1, -1], [0, 0, 0, 0]]
        shuffled = [points[i] for i in (2, 0, 3, 1)]
        self.assertEqual(
            compute_polytope_normal_form_id(points),
            compute_polytope_normal_form_id(shuffled),
        )

    def test_distinct_normal_forms_get_distinct_identifiers(self):
        a = compute_polytope_normal_form_id([[1, 0, 0, 0], [0, 1, 0, 0]])
        b = compute_polytope_normal_form_id([[2, 0, 0, 0], [0, 1, 0, 0]])
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("normal-form-sha256:"))

    def test_candidate_records_carry_normal_form_identity(self):
        poly = _FakePoly(BASIS_POINTS)
        triangulation = _FakeTriangulation(BASIS_POINTS, [[0, 1, 2, 3, 4]])
        topology = _topology([1, 2, 3, 4], h11=5)

        records = enumerate_orientifold_candidates(poly, triangulation, topology)

        expected = compute_polytope_normal_form_id(poly.normal_form())
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record["polytope_normal_form_id"], expected)

    def test_gl_equivalent_presentations_share_identity_real_cytools(self):
        try:
            from cytools import Polytope
        except Exception as exc:  # pragma: no cover - depends on environment
            self.skipTest(f"cytools unavailable: {exc}")
        vertices = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
        )
        # A unimodular change of lattice basis (det = 1) is a GL(4,Z)
        # equivalence: it is the same geometry in a different presentation.
        unimodular = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]]
        )
        self.assertEqual(int(round(np.linalg.det(unimodular))), 1)
        first = Polytope(vertices.tolist())
        second = Polytope((vertices @ unimodular.T).tolist())
        self.assertEqual(
            compute_polytope_normal_form_id(np.asarray(first.normal_form(), dtype=int)),
            compute_polytope_normal_form_id(np.asarray(second.normal_form(), dtype=int)),
        )


if __name__ == "__main__":
    unittest.main()
