"""Synthetic regressions for CYAX-0244's cancellation-aware containment rule."""

from copy import deepcopy
from fractions import Fraction as F
import itertools
import unittest
from unittest import mock

import numpy as np

import orientifold_general_l_geometry as g
import toric_fixed_component_euler as toric_euler


SMOOTH_2D = [
    ((1, 0), (0, 1)),
    ((0, 1), (-1, 0)),
    ((-1, 0), (0, -1)),
    ((0, -1), (1, 0)),
]
DIAG_2D = [
    ((1, 1), (1, -1)),
    ((1, -1), (-1, -1)),
    ((-1, -1), (-1, 1)),
    ((-1, 1), (1, 1)),
]
IDENTITY = np.eye(4, dtype=int)
MINUS_I = -IDENTITY
SWAP = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=int,
)
ZERO_T = (F(0), F(0), F(0), F(0))
UNDETERMINED = "non_smooth_containment_undetermined"
DENSE = "eq_4.42_dense_torus_inversion_cancellation"
SINGLETON = "eq_4.42_fixed_singleton_nonvanishing_witness"
MATRIX_ZERO = "eq_4.42_restriction_matrix_zero"
MATRIX_NONZERO = "eq_4.42_restriction_matrix_nonzero_column"
SUPPORT_GATE = "support_incomplete_for_matrix_false"
_INFER_AMBIENT_RAYS = object()


def product_fan(c2):
    cones = [
        tuple(
            sorted(
                [(a, b, 0, 0) for a, b in cone]
                + [(0, 0, s3, 0), (0, 0, 0, s4)]
            )
        )
        for cone in c2
        for s3 in (1, -1)
        for s4 in (1, -1)
    ]
    return cones, sorted({ray for cone in cones for ray in cone})


def full_dual(rays):
    return [
        q
        for q in itertools.product(range(-3, 4), repeat=4)
        if all(sum(ray[i] * q[i] for i in range(4)) >= -1 for ray in rays)
    ]


def fixture_provenance(rows, *, complete=True, method="fixture_attested", source="fixture_attested"):
    return {
        "method": method,
        "complete": complete,
        "source": source,
        "row_count": len(rows),
        "rows_sha256": g._canonical_dual_rows_sha256(rows),
    }


def empty_sigma_witness(lattice=(0, 0, 0, 0)):
    return {"status": "certified", "alpha": [], "lattice_vector": list(lattice)}


def build(c2, matrix, shift, lambda_f, rows, *, complete=False, method="fixture_attested", source="fixture_attested", keys=None, decision_log=None, ambient_rays=_INFER_AMBIENT_RAYS):
    cones, rays = product_fan(c2)
    auxiliary = g.build_auxiliary_fan(cones, matrix)
    if keys is None:
        keys = g._pointwise_invariant_cone_keys(cones, matrix)
    provenance = (
        None
        if rows is None
        else fixture_provenance(
            rows,
            complete=complete,
            method=method,
            source=source,
        )
    )
    records = g._fixed_component_records(
        auxiliary,
        matrix,
        shift,
        lambda_f,
        fixed_cone_keys=keys,
        dual_points=rows,
        ambient_rays=(
            rays if ambient_rays is _INFER_AMBIENT_RAYS else ambient_rays
        ),
        fan_cones=cones,
        dual_points_complete=complete,
        dual_points_provenance=provenance,
        decision_log=decision_log,
    )
    return records, auxiliary, cones, rays


def direct_matrix_decision(
    *,
    rows,
    rays,
    cones,
    matrix=MINUS_I,
    shift=ZERO_T,
    lambda_f=1,
    sigma=(),
    nu=ZERO_T,
    cox_witness=None,
    provenance=None,
    fast_path=None,
):
    if cox_witness is None:
        cox_witness = empty_sigma_witness()
    if provenance is None and rows is not None:
        provenance = fixture_provenance(rows, complete=False)
    return g.restriction_matrix_decision(
        rows,
        provenance,
        rays,
        cones,
        sigma,
        nu,
        cox_witness,
        matrix,
        shift,
        lambda_f,
        fast_path=fast_path,
    )


class SyntheticCancellationContainmentTests(unittest.TestCase):
    def test_dual_extractor_attests_only_unfiltered_points_rows(self):
        rows = [(0, 0, 0, 0), (1, 0, 0, 0)]

        class PointsDual:
            def points(self):
                return np.asarray(rows, dtype=int)

            def vertices(self):
                return np.asarray([(0, 0, 0, 0)], dtype=int)

        class VerticesDual:
            def vertices(self):
                return np.asarray(rows, dtype=int)

        class Poly:
            def __init__(self, dual):
                self._dual = dual

            def dual(self):
                return self._dual

        extracted, provenance = g._extract_dual_lattice_points(Poly(PointsDual()))
        self.assertEqual(extracted.tolist(), [list(row) for row in rows])
        self.assertEqual(provenance["method"], "points")
        self.assertEqual(provenance["source"], "cytools_dual_points")
        self.assertIs(provenance["complete"], True)
        self.assertEqual(provenance["row_count"], 2)
        self.assertEqual(provenance["rows_sha256"], g._canonical_dual_rows_sha256(rows))

        extracted, provenance = g._extract_dual_lattice_points(Poly(VerticesDual()))
        self.assertEqual(extracted.tolist(), [list(row) for row in rows])
        self.assertEqual(provenance["method"], "vertices")
        self.assertIs(provenance["complete"], False)

        extracted, provenance = g._extract_dual_lattice_points(
            Poly(PointsDual()), PointsDual()
        )
        self.assertEqual(extracted.tolist(), [list(row) for row in rows])
        self.assertIs(provenance["complete"], False)

    def test_s1_complete_dense_inversion_agrees_with_a4_and_keeps_legacy_support(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        records, _, _, _ = build(DIAG_2D, MINUS_I, ZERO_T, 1, rows, complete=True)

        self.assertEqual(len(rows), 45)
        self.assertEqual(len(records), 16)
        for item in records:
            self.assertIs(item["f_vanishes_identically"], True)
            self.assertEqual(item["containment_method"], DENSE)
            self.assertIsNone(item["containment_reason_code"])
            self.assertIs(item["containment_generic"], False)
            self.assertIs(item["invariant_restricted_support"]["restriction_identically_zero"], False)
            witness = item["containment_witness"]
            self.assertEqual(
                set(witness),
                {
                    "failed_checks",
                    "dual_points_provenance",
                    "kappa",
                    "column_status",
                    "orbit_count",
                    "restriction_matrix_shape",
                },
            )
            self.assertEqual(witness["dual_points_provenance"]["source"], "fixture_attested")
            self.assertIs(witness["dual_points_provenance"]["complete"], True)
            self.assertEqual(witness["orbit_count"], 22)
            self.assertEqual(witness["restriction_matrix_shape"], [1, 22])
            self.assertEqual(witness["kappa"], -1)
            self.assertEqual(witness["failed_checks"], [])

    def test_s2_a6_singleton_fast_path_and_matrix_witness_are_preserved(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        decision_log = []
        records, _, _, _ = build(
            DIAG_2D,
            MINUS_I,
            ZERO_T,
            0,
            rows,
            complete=True,
            decision_log=decision_log,
        )

        self.assertEqual(records, [])
        self.assertEqual(len(decision_log), 16)
        for logged in decision_log:
            self.assertIs(logged["f_vanishes_identically"], False)
            self.assertEqual(logged["containment_method"], SINGLETON)
            self.assertIsNone(logged["containment_reason_code"])
            self.assertIs(logged["containment_generic"], True)
            self.assertFalse(logged["kept_in_output"])
            logged_witness = logged["containment_witness"]
            self.assertEqual(logged_witness["failed_checks"], [])
            self.assertEqual(
                set(logged_witness["matrix_witness"]),
                {"q", "Aq", "phase", "column"},
            )
        item = decision_log[0]
        self.assertIs(item["f_vanishes_identically"], False)
        self.assertEqual(item["containment_method"], SINGLETON)
        self.assertIsNone(item["containment_reason_code"])
        self.assertIs(item["containment_generic"], True)
        self.assertFalse(item["kept_in_output"])
        self.assertEqual(
            set(item["containment_witness"]),
            {
                "failed_checks",
                "dual_points_provenance",
                "kappa",
                "column_status",
                "support_complete",
                "matrix_witness",
                "q",
                "orbit_q",
                "phase_exponent",
            },
        )
        witness = item["containment_witness"]
        self.assertEqual(witness["q"], [0, 0, 0, 0])
        self.assertEqual(item["containment_witness"]["orbit_q"], [[0, 0, 0, 0]])
        self.assertEqual(
            set(witness["matrix_witness"]), {"q", "Aq", "phase", "column"}
        )

    def test_p_i2_n7_a6_witness_survives_matrix_input_refusal(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        decision_log = []
        records, _, _, _ = build(
            DIAG_2D,
            MINUS_I,
            (0.0, F(0), F(0), F(0)),
            0,
            rows,
            complete=True,
            keys=[[]],
            decision_log=decision_log,
        )

        self.assertEqual(records, [])
        self.assertEqual(len(decision_log), 16)
        for logged in decision_log:
            self.assertIs(logged["f_vanishes_identically"], False)
            self.assertEqual(logged["containment_method"], SINGLETON)
            self.assertIsNone(logged["containment_reason_code"])
            self.assertIs(logged["containment_generic"], True)
            witness = logged["containment_witness"]
            self.assertEqual(
                set(witness),
                {
                    "failed_checks",
                    "dual_points_provenance",
                    "q",
                    "orbit_q",
                    "phase_exponent",
                },
            )
            self.assertEqual(witness["failed_checks"], ["inexact_input"])
            self.assertIs(witness["dual_points_provenance"]["complete"], True)
            self.assertEqual(witness["q"], [0, 0, 0, 0])

    def test_p_i2_b1_base_kappa_predicate_includes_dead_orbit_records(self):
        first = [
            {
                "q": [-1, 0, 0, 0],
                "Aq": [1, 0, 0, 0],
                "free": True,
                "survives": True,
                "status": "zero",
                "column": {},
                "kappa": None,
                "pair_phase_consistent": True,
            },
            {
                "q": [0, 1, 0, 0],
                "Aq": [0, -1, 0, 0],
                "free": True,
                "survives": False,
                "status": "dead",
                "column": {},
                "kappa": 1,
                "pair_phase_consistent": True,
            },
        ]
        second = [dict(first[0], kappa=1), dict(first[1], kappa=-1)]

        with self.subTest(case="survivor without kappa; dead record has kappa"):
            result = g.decide_from_columns(first, [], {"complete": True})
            self.assertIsNone(result[0])
            self.assertEqual(result[1], UNDETERMINED)
            self.assertEqual(result[2], "column_uniformity_violated")

        with self.subTest(case="survivor and dead record have opposite kappas"):
            result = g.decide_from_columns(second, [], {"complete": True})
            self.assertIs(result[0], True)
            self.assertEqual(result[1], MATRIX_ZERO)
            self.assertIsNone(result[2])
            self.assertIs(result[3], False)

    def test_s11_a6_matrix_witness_is_separate_for_qt4_diagonal_lambda1(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        matrix = np.diag([-1, -1, -1, 1])
        shift = (F(0), F(0), F(0), F(1, 2))
        self.assertEqual(len(rows), 45)

        for sigma_ray, expected_q in (
            ((0, 0, 0, 1), [0, 0, 0, -1]),
            ((0, 0, 0, -1), [0, 0, 0, 1]),
        ):
            sigma = [sigma_ray]
            decision_log = []
            records, _, _, _ = build(
                DIAG_2D,
                matrix,
                shift,
                1,
                rows,
                complete=True,
                keys=[sigma],
                decision_log=decision_log,
            )

            self.assertEqual(records, [])
            self.assertEqual(len(decision_log), 8)
            for item in decision_log:
                self.assertEqual(item["sigma_rays"], [list(sigma_ray)])
                self.assertIs(item["f_vanishes_identically"], False)
                self.assertEqual(item["containment_method"], SINGLETON)
                self.assertIsNone(item["containment_reason_code"])
                self.assertIs(item["containment_generic"], True)
                witness = item["containment_witness"]
                self.assertEqual(witness["q"], expected_q)
                self.assertEqual(witness["orbit_q"], [expected_q])
                self.assertEqual(
                    set(witness["matrix_witness"]), {"q", "Aq", "phase", "column"}
                )
                self.assertEqual(witness["kappa"], 1)
                self.assertEqual(witness["failed_checks"], [])
                self.assertEqual(
                    witness["dual_points_provenance"]["method"], "fixture_attested"
                )
                self.assertIs(witness["dual_points_provenance"]["complete"], True)
                self.assertIs(witness["support_complete"], True)
                self.assertNotIn("Aq", witness)
                self.assertNotIn("phase", witness)
                self.assertNotIn("column", witness)

    def test_s2b_partial_pair_proves_generic_false_before_point_filter(self):
        cones, rays = product_fan(DIAG_2D)
        rows = [(1, 0, 0, 0), (-1, 0, 0, 0)]
        decision_log = []
        records, _, _, _ = build(
            DIAG_2D,
            MINUS_I,
            ZERO_T,
            0,
            rows,
            complete=False,
            decision_log=decision_log,
        )

        self.assertEqual(records, [])
        self.assertEqual(len(decision_log), 16)
        for item in decision_log:
            self.assertIs(item["f_vanishes_identically"], False)
            self.assertEqual(item["containment_method"], MATRIX_NONZERO)
            self.assertIsNone(item["containment_reason_code"])
            self.assertIs(item["containment_generic"], True)
            self.assertEqual(item["fixed_toric_dimension"], 0)
            self.assertFalse(item["kept_in_output"])
            witness = item["containment_witness"]
            self.assertEqual(witness["q"], [-1, 0, 0, 0])
            self.assertEqual(witness["Aq"], [1, 0, 0, 0])
            self.assertEqual(witness["column"], [{"character": [], "coefficient": 2}])
            self.assertIs(witness["support_complete"], False)
            self.assertEqual(witness["failed_checks"], [])

    def test_s3_missing_orbit_pair_is_undetermined_without_completeness(self):
        cones, rays = product_fan(DIAG_2D)
        rows = [q for q in full_dual(rays) if q not in ((0, 1, 0, 0), (0, -1, 0, 0))]
        decision_log = []
        records, _, _, _ = build(
            DIAG_2D,
            MINUS_I,
            ZERO_T,
            1,
            rows,
            complete=False,
            decision_log=decision_log,
        )

        self.assertEqual(len(records), 16)
        for item in decision_log:
            self.assertIsNone(item["f_vanishes_identically"])
            self.assertEqual(item["containment_method"], UNDETERMINED)
            self.assertEqual(item["containment_reason_code"], "dual_points_incomplete")
            self.assertIsNone(item["containment_generic"])
            self.assertEqual(item["containment_witness"]["failed_checks"], ["dual_points_incomplete"])
            self.assertIs(item["containment_witness"]["dual_points_provenance"]["complete"], False)
            self.assertEqual(item["containment_witness"]["kappa"], -1)
        self.assertTrue(all(item["kept_in_output"] for item in decision_log))

    def test_s4_vertices_provenance_cannot_certify_the_true_direction(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        decision_log = []
        records, _, _, _ = build(
            DIAG_2D,
            MINUS_I,
            ZERO_T,
            1,
            rows,
            complete=False,
            method="vertices",
            source="cytools_dual_points",
            decision_log=decision_log,
        )

        self.assertEqual(len(records), 16)
        for item in decision_log:
            self.assertIsNone(item["f_vanishes_identically"])
            self.assertEqual(item["containment_method"], UNDETERMINED)
            self.assertEqual(item["containment_reason_code"], "dual_points_incomplete")
            self.assertIsNone(item["containment_generic"])
            provenance = item["containment_witness"]["dual_points_provenance"]
            self.assertEqual(provenance["method"], "vertices")
            self.assertIs(provenance["complete"], False)

    def test_provenance_hash_mismatch_cannot_certify_the_true_direction(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        provenance = fixture_provenance(rows, complete=True)
        provenance["rows_sha256"] = "0" * 64
        result = direct_matrix_decision(
            rows=rows,
            rays=rays,
            cones=cones,
            provenance=provenance,
        )

        self.assertIsNone(result[0])
        self.assertEqual(result[1], UNDETERMINED)
        self.assertEqual(result[2], "dual_points_incomplete")
        self.assertIsNone(result[3])
        self.assertIs(result[4]["dual_points_provenance"]["complete"], False)

    def test_s5_conflicting_fast_path_and_matrix_return_both_certificates(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        provenance = fixture_provenance(rows, complete=True)
        orbit_records, failed_checks = g.restriction_matrix_columns(
            rows,
            provenance,
            rays,
            cones,
            (),
            ZERO_T,
            empty_sigma_witness(),
            MINUS_I,
            ZERO_T,
            0,
        )
        result = g.decide_from_columns(
            orbit_records,
            failed_checks,
            provenance,
            fast_path=(True, DENSE),
        )
        direct_result = direct_matrix_decision(
            rows=rows,
            rays=rays,
            cones=cones,
            lambda_f=0,
            provenance=provenance,
            fast_path=(True, DENSE),
        )
        self.assertEqual(result, direct_result)

        value, method, reason, generic, witness = result
        self.assertIsNone(value)
        self.assertEqual(method, "containment_certificate_conflict")
        self.assertEqual(reason, "certificate_conflict")
        self.assertIsNone(generic)
        self.assertEqual(witness["failed_checks"], ["certificate_conflict"])
        self.assertEqual(
            set(witness),
            {"failed_checks", "fast_path_certificate", "matrix_certificate"},
        )
        self.assertEqual(
            witness["fast_path_certificate"], {"value": True, "method": DENSE}
        )
        self.assertIs(witness["matrix_certificate"]["value"], False)
        self.assertEqual(witness["matrix_certificate"]["method"], MATRIX_NONZERO)

    def test_s12_fast_path_refusals_keep_verified_receipt(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        provenance = fixture_provenance(rows, complete=True)
        invalid_rows = [(2, 0, 0, 0)]
        frame_rays = [(1, 2, 0, 0), (2, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        frame_cones = [tuple(frame_rays)]
        frame_rows = [(0, 0, 0, 0)]
        frame_provenance = fixture_provenance(frame_rows, complete=True)
        cases = [
            (
                "action_not_validated",
                rows,
                provenance,
                rays,
                cones,
                (),
                ZERO_T,
                empty_sigma_witness(),
                np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                ZERO_T,
                1,
            ),
            (
                "inexact_input",
                rows,
                provenance,
                rays,
                cones,
                (),
                ZERO_T,
                empty_sigma_witness(),
                MINUS_I,
                (0.0, 0, 0, 0),
                1,
            ),
            (
                "component_label_unbound",
                rows,
                provenance,
                rays,
                cones,
                (),
                (F(1, 3), F(0), F(0), F(0)),
                empty_sigma_witness(),
                MINUS_I,
                ZERO_T,
                1,
            ),
            (
                "chart_unavailable",
                rows,
                provenance,
                rays,
                cones,
                [(1, 0, 0, 0)],
                ZERO_T,
                {"status": "certified", "alpha": [1], "lattice_vector": [1, 0, 0, 0]},
                IDENTITY,
                ZERO_T,
                0,
            ),
            (
                "dual_rows_invalid",
                invalid_rows,
                fixture_provenance(invalid_rows, complete=True),
                rays,
                cones,
                (),
                ZERO_T,
                empty_sigma_witness(),
                MINUS_I,
                ZERO_T,
                1,
            ),
            (
                "frame_unavailable",
                frame_rows,
                frame_provenance,
                frame_rays,
                frame_cones,
                frame_rays[:2],
                ZERO_T,
                {"status": "certified", "alpha": [F(1, 3), F(1, 3)], "lattice_vector": [1, 1, 0, 0]},
                IDENTITY,
                ZERO_T,
                0,
            ),
        ]
        for expected_failure, case_rows, receipt, case_rays, case_cones, sigma, nu, cox, matrix, shift, lambda_f in cases:
            with self.subTest(failure=expected_failure):
                result = g.restriction_matrix_decision(
                    case_rows,
                    receipt,
                    case_rays,
                    case_cones,
                    sigma,
                    nu,
                    cox,
                    matrix,
                    shift,
                    lambda_f,
                    fast_path=(True, DENSE),
                )
                self.assertIs(result[0], True)
                self.assertEqual(result[1], DENSE)
                self.assertIn(expected_failure, result[4]["failed_checks"])
                verified_receipt = g._verified_dual_points_provenance(
                    case_rows,
                    receipt.get("complete") is True,
                    receipt,
                )
                self.assertEqual(
                    result[4]["dual_points_provenance"], verified_receipt
                )

        no_receipt = g.restriction_matrix_decision(
            rows,
            None,
            rays,
            cones,
            (),
            ZERO_T,
            empty_sigma_witness(),
            MINUS_I,
            (0.0, 0, 0, 0),
            1,
            fast_path=(True, DENSE),
        )
        self.assertNotIn("dual_points_provenance", no_receipt[4])

    def test_s6_missing_integral_frame_fails_closed(self):
        rays = [(1, 2, 0, 0), (2, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        cones = [tuple(rays)]
        sigma = rays[:2]
        cox = {
            "status": "certified",
            "alpha": [F(1, 3), F(1, 3)],
            "lattice_vector": [1, 1, 0, 0],
        }
        rows = [(0, 0, 0, 0)]
        result = direct_matrix_decision(
            rows=rows,
            rays=rays,
            cones=cones,
            matrix=IDENTITY,
            lambda_f=0,
            sigma=sigma,
            cox_witness=cox,
            provenance=fixture_provenance(rows, complete=True),
        )

        self.assertIsNone(result[0])
        self.assertEqual(result[1], UNDETERMINED)
        self.assertEqual(result[2], "frame_unavailable")
        self.assertIsNone(result[3])
        self.assertEqual(result[4]["failed_checks"], ["frame_unavailable"])
        self.assertNotIn("kappa", result[4])

    def test_s7_pre_cancellation_support_retains_its_original_meaning(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        records, _, _, _ = build(DIAG_2D, MINUS_I, ZERO_T, 1, rows, complete=True)
        self.assertEqual(len(records), 16)
        for item in records:
            self.assertIs(item["f_vanishes_identically"], True)
            self.assertIs(
                item["invariant_restricted_support"]["restriction_identically_zero"],
                False,
            )

    def test_s8a_invalid_action_is_logged_with_the_first_reason_code(self):
        cones, rays = product_fan(DIAG_2D)
        sigma = [(1, 1, 0, 0)]
        rows = full_dual(rays)
        decision_log = []
        records, _, _, _ = build(
            DIAG_2D,
            IDENTITY,
            (F(1, 4), F(1, 4), F(0), F(0)),
            1,
            rows,
            complete=True,
            keys=[sigma],
            decision_log=decision_log,
        )

        self.assertTrue(records)
        self.assertTrue(decision_log)
        item = next(
            entry for entry in decision_log if entry["sigma_rays"] == [list(sigma[0])]
        )
        self.assertIsNone(item["f_vanishes_identically"])
        self.assertEqual(item["containment_method"], UNDETERMINED)
        self.assertEqual(item["containment_reason_code"], "action_not_validated")
        self.assertIsNone(item["containment_generic"])
        self.assertEqual(item["containment_witness"]["failed_checks"], ["action_not_validated"])
        self.assertNotIn("kappa", item["containment_witness"])

    def test_s8b_all_remaining_reason_codes_are_exact_and_fail_closed(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        complete = fixture_provenance(rows, complete=True)
        cases = []
        cases.append(
            (
                "inexact_input",
                direct_matrix_decision(
                    rows=rows,
                    rays=rays,
                    cones=cones,
                    shift=(0.0, 0, 0, 0),
                    provenance=complete,
                ),
            )
        )
        cases.append(
            (
                "component_label_unbound",
                direct_matrix_decision(
                    rows=rows,
                    rays=rays,
                    cones=cones,
                    nu=(F(1, 3), F(0), F(0), F(0)),
                    cox_witness=empty_sigma_witness(),
                    provenance=complete,
                ),
            )
        )
        missing_chart_sigma = [(1, 0, 0, 0)]
        chart_witness = {
            "status": "certified",
            "alpha": [1],
            "lattice_vector": [1, 0, 0, 0],
        }
        cases.append(
            (
                "chart_unavailable",
                direct_matrix_decision(
                    rows=rows,
                    rays=rays,
                    cones=cones,
                    matrix=IDENTITY,
                    lambda_f=0,
                    sigma=missing_chart_sigma,
                    cox_witness=chart_witness,
                    provenance=complete,
                ),
            )
        )
        nonsimplicial_rays = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 1, 1, 1),
        ]
        nonsimplicial_cones = [nonsimplicial_rays]
        zero_rows = [(0, 0, 0, 0)]
        cases.append(
            (
                "chart_unavailable",
                direct_matrix_decision(
                    rows=zero_rows,
                    rays=nonsimplicial_rays,
                    cones=nonsimplicial_cones,
                    matrix=IDENTITY,
                    lambda_f=0,
                    provenance=fixture_provenance(zero_rows, complete=True),
                ),
            )
        )
        cases.append(
            (
                "dual_rows_invalid",
                direct_matrix_decision(
                    rows=None,
                    rays=rays,
                    cones=cones,
                    matrix=IDENTITY,
                    lambda_f=0,
                ),
            )
        )
        cases.append(
            (
                "dual_rows_invalid",
                direct_matrix_decision(
                    rows=[(2, 0, 0, 0)],
                    rays=rays,
                    cones=cones,
                    matrix=IDENTITY,
                    lambda_f=0,
                ),
            )
        )
        missing_negative_e1 = [row for row in rows if row != (-1, 0, 0, 0)]
        cases.append(
            (
                "pair_phase_inconsistent",
                direct_matrix_decision(
                    rows=missing_negative_e1,
                    rays=rays,
                    cones=cones,
                    provenance=fixture_provenance(missing_negative_e1, complete=True),
                ),
            )
        )

        for expected_reason, result in cases:
            with self.subTest(reason=expected_reason, witness=result[4]):
                self.assertIsNone(result[0])
                self.assertEqual(result[1], UNDETERMINED)
                self.assertEqual(result[2], expected_reason)
                self.assertIsNone(result[3])
                self.assertEqual(result[4]["failed_checks"][0], expected_reason)

        columns, failed = g.restriction_matrix_columns(
            rows,
            complete,
            rays,
            cones,
            (),
            ZERO_T,
            empty_sigma_witness(),
            MINUS_I,
            ZERO_T,
            1,
        )
        active = next(
            record
            for record in columns
            if record["free"] and record["survives"] and record["character"] is not None
        )
        active["column"] = {tuple(active["character"]): F(2)}
        active["status"] = "nonzero"
        result = g.decide_from_columns(columns, failed, complete)
        self.assertIsNone(result[0])
        self.assertEqual(result[1], UNDETERMINED)
        self.assertEqual(result[2], "column_uniformity_violated")
        self.assertEqual(result[4]["failed_checks"], ["column_uniformity_violated"])

    def test_s8c_failed_checks_follow_condition_order(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        complete = fixture_provenance(rows, complete=True)
        incomplete_rows = [row for row in rows if row != (-1, 0, 0, 0)]
        incomplete = fixture_provenance(incomplete_rows, complete=False)

        cases = [
            (
                ["action_not_validated", "inexact_input"],
                direct_matrix_decision(
                    rows=rows,
                    rays=rays,
                    cones=cones,
                    shift=(0.25, 0, 0, 0),
                    sigma=[(7, 7, 7, 7)],
                    nu=(F(1, 3), F(0), F(0), F(0)),
                    cox_witness={
                        "status": "certified",
                        "alpha": [0],
                        "lattice_vector": [0, 0, 0, 0],
                    },
                    provenance=complete,
                ),
            ),
            (
                ["component_label_unbound"],
                direct_matrix_decision(
                    rows=incomplete_rows,
                    rays=rays,
                    cones=cones,
                    nu=(F(1, 3), F(0), F(0), F(0)),
                    provenance=incomplete,
                ),
            ),
            (
                ["pair_phase_inconsistent"],
                direct_matrix_decision(
                    rows=incomplete_rows,
                    rays=rays,
                    cones=cones,
                    provenance=incomplete,
                ),
            ),
            (
                ["component_label_unbound"],
                direct_matrix_decision(
                    rows=rows,
                    rays=rays,
                    cones=cones,
                    nu=(F(1, 3), F(0), F(0), F(0)),
                    provenance=fixture_provenance(rows, complete=False),
                ),
            ),
        ]
        invalid_rows = rows + [(2, 0, 0, 0)]
        cases.append(
            (
                ["component_label_unbound", "chart_unavailable", "dual_rows_invalid"],
                direct_matrix_decision(
                    rows=invalid_rows,
                    rays=rays,
                    cones=cones,
                    matrix=IDENTITY,
                    lambda_f=0,
                    sigma=[(1, 0, 0, 0)],
                    nu=(F(1, 2), F(0), F(0), F(0)),
                    cox_witness={
                        "status": "certified",
                        "alpha": [1],
                        "lattice_vector": [1, 0, 0, 0],
                    },
                    provenance=fixture_provenance(invalid_rows, complete=True),
                ),
            )
        )

        for expected, result in cases:
            with self.subTest(expected=expected):
                self.assertIsNone(result[0])
                self.assertEqual(result[1], UNDETERMINED)
                self.assertEqual(result[4]["failed_checks"], expected)

    def test_c4_omitted_ambient_rays_are_inferred_from_cones(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)

        with self.subTest(fixture="S-2b"):
            s2b_rows = [(1, 0, 0, 0), (-1, 0, 0, 0)]
            s2b_log = []
            s2b_records, _, _, _ = build(
                DIAG_2D,
                MINUS_I,
                ZERO_T,
                0,
                s2b_rows,
                complete=False,
                decision_log=s2b_log,
                ambient_rays=None,
            )
            self.assertEqual(s2b_records, [])
            self.assertEqual(len(s2b_log), 16)
            phases = []
            for item in s2b_log:
                self.assertIs(item["f_vanishes_identically"], False)
                self.assertEqual(item["containment_method"], MATRIX_NONZERO)
                self.assertIsNone(item["containment_reason_code"])
                witness = item["containment_witness"]
                self.assertEqual(witness["q"], [-1, 0, 0, 0])
                self.assertEqual(witness["Aq"], [1, 0, 0, 0])
                nu1 = F(item["nu"]["numerator"][0], item["nu"]["denominator"])
                self.assertEqual(witness["phase"], nu1)
                phases.append(witness["phase"])
                self.assertEqual(witness["column"], [{"character": [], "coefficient": 2}])
                self.assertIs(witness["support_complete"], False)
            self.assertEqual(phases.count(0), 8)
            self.assertEqual(phases.count(1), 8)

        with self.subTest(fixture="S-1"):
            s1_records, _, _, _ = build(
                DIAG_2D,
                MINUS_I,
                ZERO_T,
                1,
                rows,
                complete=True,
                ambient_rays=None,
            )
            self.assertEqual(len(s1_records), 16)
            expected_rays = [list(ray) for ray in rays]
            for item in s1_records:
                self.assertIs(item["f_vanishes_identically"], True)
                self.assertEqual(item["containment_method"], DENSE)
                self.assertIsNone(item["containment_reason_code"])
                witness = item["containment_witness"]
                self.assertEqual(
                    set(witness),
                    {
                        "failed_checks",
                        "dual_points_provenance",
                        "kappa",
                        "column_status",
                        "orbit_count",
                        "restriction_matrix_shape",
                    },
                )
                self.assertEqual(witness["failed_checks"], [])
                self.assertEqual(witness["dual_points_provenance"]["rows_sha256"], fixture_provenance(rows, complete=True)["rows_sha256"])
                self.assertEqual(
                    item["invariant_restricted_support"]["ambient_rays"], expected_rays
                )

        q_t4_matrix = np.diag([-1, -1, -1, 1])
        q_t4_shift = (F(0), F(0), F(0), F(1, 2))
        with self.subTest(fixture="Q-T4"):
            q_t4_records, _, _, _ = build(
                DIAG_2D,
                q_t4_matrix,
                q_t4_shift,
                0,
                rows,
                complete=True,
                keys=[[(0, 0, 0, 1)], [(0, 0, 0, -1)]],
                ambient_rays=None,
            )
            self.assertEqual(len(q_t4_records), 16)
            for item in q_t4_records:
                self.assertIs(item["f_vanishes_identically"], True)
                self.assertEqual(item["containment_method"], MATRIX_ZERO)
                self.assertEqual(item["containment_witness"]["failed_checks"], [])

        sigma = [(1, 1, 0, 0)]
        nu = (F(1, 2), F(-1, 2), F(0), F(0))
        cox = g._cox_alpha_lattice_witness(sigma, nu)
        provenance = fixture_provenance(rows, complete=True)
        with self.subTest(fixture="Q-T7 lambda=0"):
            q_t7_zero = direct_matrix_decision(
                rows=rows,
                rays=None,
                cones=cones,
                matrix=SWAP,
                lambda_f=0,
                sigma=sigma,
                nu=nu,
                cox_witness=cox,
                provenance=provenance,
            )
            self.assertIs(q_t7_zero[0], True)
            self.assertEqual(q_t7_zero[1], MATRIX_ZERO)
            self.assertEqual(q_t7_zero[4]["failed_checks"], [])

        with self.subTest(fixture="Q-T7 lambda=1"):
            q_t7_one = direct_matrix_decision(
                rows=rows,
                rays=None,
                cones=cones,
                matrix=SWAP,
                lambda_f=1,
                sigma=sigma,
                nu=nu,
                cox_witness=cox,
                provenance=provenance,
            )
            self.assertIs(q_t7_one[0], False)
            self.assertEqual(q_t7_one[1], MATRIX_NONZERO)
            self.assertIs(q_t7_one[4]["support_complete"], True)

        with self.subTest(fixture="omitted rays without cones"):
            no_cones = direct_matrix_decision(
                rows=rows,
                rays=None,
                cones=None,
                provenance=provenance,
            )
            self.assertIsNone(no_cones[0])

        swap_13 = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=int,
        )
        with self.subTest(fixture="invalid action"):
            invalid_action = direct_matrix_decision(
                rows=rows,
                rays=None,
                cones=cones,
                matrix=swap_13,
                provenance=provenance,
            )
            self.assertIsNone(invalid_action[0])
            self.assertEqual(invalid_action[2], "action_not_validated")

    def test_p_i2_b2_malformed_omitted_rays_fail_closed_in_direct_api(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        provenance = fixture_provenance(rows, complete=True)
        malformed = []
        dict_coordinate = [
            tuple(
                ({"numerator": 1, "denominator": 1},) + tuple(ray[1:])
                if ray == (1, 1, 0, 0)
                else tuple(ray)
                for ray in cone
            )
            for cone in cones
        ]
        malformed.extend(
            [
                ("dict coordinate", dict_coordinate),
                ("non-iterable cone", list(cones) + [7]),
                ("cone of scalars", [(1, 0, 0, 0)]),
                ("mixed coordinate types", [((1, 0, 0, 0), ("a", "b", "c", "d"))]),
            ]
        )
        expected = (
            None,
            UNDETERMINED,
            "action_not_validated",
            None,
            {
                "failed_checks": [
                    "action_not_validated",
                    "chart_unavailable",
                    "dual_rows_invalid",
                ],
                "dual_points_provenance": provenance,
            },
        )

        for label, bad_cones in malformed:
            with self.subTest(cones=label):
                result = direct_matrix_decision(
                    rows=rows,
                    rays=None,
                    cones=bad_cones,
                    provenance=provenance,
                )
                self.assertEqual(result, expected)

    def test_p_i2_n6_builder_preserves_base_malformed_cone_exceptions(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        provenance = fixture_provenance(rows, complete=True)
        auxiliary = g.build_auxiliary_fan(cones, MINUS_I)
        keys = g._pointwise_invariant_cone_keys(cones, MINUS_I)
        malformed = [
            (
                "dict coordinate",
                [
                    tuple(
                        ({"numerator": 1, "denominator": 1},) + tuple(ray[1:])
                        if ray == (1, 1, 0, 0)
                        else tuple(ray)
                        for ray in cone
                    )
                    for cone in cones
                ],
                TypeError,
                "int() argument must be a string, a bytes-like object or a real number, not 'dict'",
            ),
            (
                "mixed coordinate types",
                [((1, 0, 0, 0), ("a", "b", "c", "d"))],
                ValueError,
                "invalid literal for int() with base 10: 'a'",
            ),
            (
                "cone of scalars",
                [(1, 0, 0, 0)],
                TypeError,
                "'int' object is not iterable",
            ),
        ]

        for label, bad_cones, error_type, message in malformed:
            with self.subTest(cones=label):
                with self.assertRaises(error_type) as raised:
                    g._fixed_component_records(
                        auxiliary,
                        MINUS_I,
                        ZERO_T,
                        1,
                        fixed_cone_keys=keys,
                        dual_points=rows,
                        ambient_rays=None,
                        fan_cones=bad_cones,
                        dual_points_complete=True,
                        dual_points_provenance=provenance,
                    )
                self.assertEqual(str(raised.exception), message)

        empty = g._fixed_component_records(
            auxiliary,
            MINUS_I,
            ZERO_T,
            1,
            fixed_cone_keys=[],
            dual_points=rows,
            ambient_rays=None,
            fan_cones=malformed[0][1],
            dual_points_complete=True,
            dual_points_provenance=provenance,
        )
        self.assertEqual(empty, [])

    def test_p_i2_b3_builder_marks_both_missing_ray_inputs_unavailable(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        provenance = fixture_provenance(rows, complete=True)
        auxiliary = g.build_auxiliary_fan(cones, MINUS_I)
        keys = g._pointwise_invariant_cone_keys(cones, MINUS_I)
        records = g._fixed_component_records(
            auxiliary,
            MINUS_I,
            ZERO_T,
            1,
            fixed_cone_keys=keys,
            dual_points=rows,
            ambient_rays=None,
            fan_cones=None,
            dual_points_complete=True,
            dual_points_provenance=provenance,
        )

        self.assertTrue(records)
        self.assertEqual(
            records[0]["invariant_restricted_support"]["reason"],
            "ambient ray data is unavailable",
        )

    def test_c5_refusal_witness_flags_are_sparse_and_exclude_invalid_columns(self):
        cones, rays = product_fan(DIAG_2D)

        def columns_for(rows, lambda_f, *, complete=False):
            receipt = fixture_provenance(rows, complete=complete)
            columns, failures = g.restriction_matrix_columns(
                rows,
                receipt,
                rays,
                cones,
                (),
                ZERO_T,
                empty_sigma_witness(),
                MINUS_I,
                ZERO_T,
                lambda_f,
            )
            return columns, failures, receipt

        pair_rows = [(1, 0, 0, 0), (-1, 0, 0, 0)]
        columns, failures, receipt = columns_for(pair_rows, 0)
        for item in columns:
            if item.get("free") is True and item.get("survives") is True:
                item["kappa"] = -1
        mismatch = g.decide_from_columns(columns, failures, receipt)
        self.assertIsNone(mismatch[0])
        self.assertEqual(mismatch[1], UNDETERMINED)
        self.assertEqual(mismatch[2], "column_uniformity_violated")
        self.assertIs(mismatch[4]["kappa_mismatch"], True)
        self.assertNotIn("column_invalid", mismatch[4])

        columns, failures, receipt = columns_for(pair_rows, 1)
        for item in columns:
            if item.get("free") is True and item.get("survives") is True:
                item["kappa"] = 1
        mismatch = g.decide_from_columns(columns, failures, receipt)
        self.assertIsNone(mismatch[0])
        self.assertEqual(mismatch[2], "column_uniformity_violated")
        self.assertIs(mismatch[4]["kappa_mismatch"], True)
        self.assertNotIn("column_invalid", mismatch[4])

        columns, failures, receipt = columns_for(pair_rows, 1)
        for item in columns:
            if item.get("free") is True and item.get("survives") is True:
                item["status"] = "column_invalid"
                item["kappa"] = None
        invalid = g.decide_from_columns(columns, failures, receipt)
        self.assertEqual(invalid[2], "column_uniformity_violated")
        self.assertIs(invalid[4]["column_invalid"], True)
        self.assertNotIn("kappa_mismatch", invalid[4])

        mixed_rows = [
            (1, 0, 0, 0),
            (-1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, -1, 0, 0),
        ]
        columns, failures, receipt = columns_for(mixed_rows, 1)
        active = next(
            item
            for item in columns
            if item["free"] and item["survives"] and item["character"] is not None
        )
        active["column"] = {(): F(2)}
        active["status"] = "nonzero"
        mixed = g.decide_from_columns(columns, failures, receipt)
        self.assertEqual(mixed[2], "column_uniformity_violated")
        self.assertNotIn("kappa_mismatch", mixed[4])
        self.assertNotIn("column_invalid", mixed[4])

    def test_p_i2_n2_column_invalid_refusal_flag_requires_available_frame(self):
        record = {
            "q": [-1, 0, 0, 0],
            "Aq": [1, 0, 0, 0],
            "free": True,
            "survives": True,
            "status": "column_invalid",
            "column": {},
            "kappa": None,
            "pair_phase_consistent": True,
        }
        result = g.decide_from_columns(
            [record], ["dual_rows_invalid"], {"complete": True}
        )

        self.assertIsNone(result[0])
        self.assertEqual(result[1], UNDETERMINED)
        self.assertEqual(result[2], "dual_rows_invalid")
        self.assertNotIn("column_invalid", result[4])

    def test_p_i2_n4_invalid_kappa_is_excluded_from_mismatch_flag(self):
        records = [
            {
                "q": [-1, 0, 0, 0],
                "Aq": [1, 0, 0, 0],
                "free": True,
                "survives": True,
                "status": "zero",
                "column": {},
                "kappa": 1,
                "pair_phase_consistent": True,
            },
            {
                "q": [0, -1, 0, 0],
                "Aq": [0, 1, 0, 0],
                "free": True,
                "survives": True,
                "status": "column_invalid",
                "column": {},
                "kappa": -1,
                "pair_phase_consistent": True,
            },
        ]
        result = g.decide_from_columns(records, [], {"complete": True})

        self.assertIsNone(result[0])
        self.assertEqual(result[2], "column_uniformity_violated")
        self.assertIs(result[4]["kappa_mismatch"], True)
        self.assertIs(result[4]["column_invalid"], True)

    def test_s8b_m2_type_leaves_keep_exact_failure_lists(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        bad = (("bad", "inexact_input"), (b"bad", "inexact_input"), (None, "inexact_input"))

        for value, _ in bad:
            cases = {
                "torus_shift": ["action_not_validated", "component_label_unbound"],
                "matrix": ["action_not_validated", "component_label_unbound"],
                "lambda_f": ["action_not_validated"],
                "nu": ["component_label_unbound"],
                "alpha": ["component_label_unbound"],
                "lattice_vector": ["component_label_unbound"],
                "row_entry": ["dual_rows_invalid"],
            }
            for position, expected in cases.items():
                matrix = IDENTITY.tolist()
                shift = list(ZERO_T)
                lambda_f = 0
                sigma = []
                nu = list(ZERO_T)
                cox = empty_sigma_witness()
                case_rows = [list(row) for row in rows]
                if position == "torus_shift":
                    shift[0] = value
                elif position == "matrix":
                    matrix[0][0] = value
                elif position == "lambda_f":
                    lambda_f = value
                elif position == "nu":
                    nu[0] = value
                elif position in {"alpha", "lattice_vector"}:
                    sigma = [(1, 1, 0, 0)]
                    cox = {
                        "status": "certified",
                        "alpha": [0],
                        "lattice_vector": [0, 0, 0, 0],
                    }
                    if position == "alpha":
                        cox["alpha"][0] = value
                    else:
                        cox["lattice_vector"][0] = value
                else:
                    case_rows[0][0] = value
                result = direct_matrix_decision(
                    rows=case_rows,
                    rays=rays,
                    cones=cones,
                    matrix=matrix,
                    shift=shift,
                    lambda_f=lambda_f,
                    sigma=sigma,
                    nu=nu,
                    cox_witness=cox,
                    provenance=fixture_provenance(rows, complete=True),
                )
                with self.subTest(value=repr(value), position=position):
                    self.assertIsNone(result[0])
                    self.assertEqual(result[1], UNDETERMINED)
                    self.assertEqual(result[4]["failed_checks"], expected)

        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        complete = fixture_provenance(rows, complete=True)
        zero_scalar = {"numerator": 1, "denominator": 0}
        zero_vector = {"numerator": [0, 0, 0, 0], "denominator": 0}
        p3_cases = [
            ((zero_scalar, 0, 0, 0), ZERO_T, empty_sigma_witness(), ["action_not_validated", "component_label_unbound"]),
            (ZERO_T, (zero_scalar, 0, 0, 0), empty_sigma_witness(), ["component_label_unbound"]),
            (ZERO_T, ZERO_T, {"status": "certified", "alpha": [zero_scalar], "lattice_vector": [0, 0, 0, 0]}, ["component_label_unbound"]),
            (ZERO_T, ZERO_T, {"status": "certified", "alpha": [], "lattice_vector": [zero_scalar, 0, 0, 0]}, ["component_label_unbound"]),
            (zero_vector, ZERO_T, empty_sigma_witness(), ["action_not_validated", "component_label_unbound"]),
            (ZERO_T, zero_vector, empty_sigma_witness(), ["component_label_unbound"]),
        ]
        for shift, nu, cox, expected in p3_cases:
            result = direct_matrix_decision(
                rows=rows,
                rays=rays,
                cones=cones,
                matrix=MINUS_I,
                shift=shift,
                lambda_f=1,
                nu=nu,
                cox_witness=cox,
                provenance=complete,
            )
            with self.subTest(shift=shift, nu=nu, cox=cox):
                self.assertIsNone(result[0])
                self.assertEqual(result[4]["failed_checks"], expected)

        for value in ("bad", b"bad", None):
            result = direct_matrix_decision(
                rows=rows,
                rays=rays,
                cones=cones,
                matrix=MINUS_I,
                lambda_f=value,
                provenance=complete,
            )
            self.assertEqual(result[4]["failed_checks"], ["action_not_validated"])

        controls = [
            ZERO_T,
            {"numerator": [0, 0, 0, 0], "denominator": 1},
            ({"numerator": 0, "denominator": 1}, 0, 0, 0),
        ]
        for shift in controls:
            result = direct_matrix_decision(
                rows=rows,
                rays=rays,
                cones=cones,
                matrix=MINUS_I,
                shift=shift,
                lambda_f=1,
                provenance=complete,
            )
            self.assertIs(result[0], True)
            self.assertEqual(result[1], MATRIX_ZERO)
            self.assertEqual(result[4]["failed_checks"], [])

    def test_s9_decision_log_does_not_change_records_and_copies_nested_values(self):
        cones, rays = product_fan(DIAG_2D)
        full_rows = full_dual(rays)
        baseline, _, _, _ = build(DIAG_2D, MINUS_I, ZERO_T, 1, full_rows, complete=True)
        decision_log = []
        logged, _, _, _ = build(
            DIAG_2D,
            MINUS_I,
            ZERO_T,
            1,
            full_rows,
            complete=True,
            decision_log=decision_log,
        )

        self.assertEqual(logged, baseline)
        self.assertEqual(len(decision_log), 16)
        before = deepcopy(logged[0]["containment_witness"]["column_status"][0]["q"])
        decision_log[0]["containment_witness"]["column_status"][0]["q"][0] = 99
        self.assertEqual(logged[0]["containment_witness"]["column_status"][0]["q"], before)
        self.assertTrue(all(entry["kept_in_output"] for entry in decision_log))

        incomplete_pair_rows = [
            row for row in full_rows if row not in ((1, 0, 0, 0), (-1, 0, 0, 0))
        ]
        boundary_sigma = [(1, 1, 0, 0)]
        boundary_shift = (F(1, 4), F(1, 4), F(0), F(0))
        s10_rows = [
            row for row in full_rows if row not in ((-1, 0, 0, 0), (0, -1, 0, 0))
        ]
        s11_matrix = np.diag([-1, -1, -1, 1])
        s11_shift = (F(0), F(0), F(0), F(1, 2))
        cases = [
            (MINUS_I, ZERO_T, 0, full_rows, True, "fixture_attested", None),
            (MINUS_I, ZERO_T, 0, [(1, 0, 0, 0), (-1, 0, 0, 0)], False, "fixture_attested", None),
            (MINUS_I, ZERO_T, 1, incomplete_pair_rows, False, "fixture_attested", None),
            (MINUS_I, ZERO_T, 1, full_rows, False, "vertices", None),
            (IDENTITY, boundary_shift, 1, full_rows, True, "fixture_attested", [boundary_sigma]),
            (SWAP, ZERO_T, 1, full_rows, True, "fixture_attested", [boundary_sigma]),
            (SWAP, ZERO_T, 1, s10_rows, False, "fixture_attested", [boundary_sigma]),
            (s11_matrix, s11_shift, 1, full_rows, True, "fixture_attested", [[(0, 0, 0, 1)], [(0, 0, 0, -1)]]),
        ]
        for matrix, shift, lambda_f, rows, complete, method, keys in cases:
            with self.subTest(method=method, lambda_f=lambda_f, keys=keys):
                without_log, _, _, _ = build(
                    DIAG_2D,
                    matrix,
                    shift,
                    lambda_f,
                    rows,
                    complete=complete,
                    method=method,
                    source="cytools_dual_points" if method == "vertices" else "fixture_attested",
                    keys=keys,
                )
                call_log = []
                with_log, _, _, _ = build(
                    DIAG_2D,
                    matrix,
                    shift,
                    lambda_f,
                    rows,
                    complete=complete,
                    method=method,
                    source="cytools_dual_points" if method == "vertices" else "fixture_attested",
                    keys=keys,
                    decision_log=call_log,
                )
                self.assertEqual(with_log, without_log)
                self.assertTrue(call_log)

        smooth_log = []
        smooth_matrix = np.diag([-1, -1, -1, 1])
        build(
            SMOOTH_2D,
            smooth_matrix,
            (F(0), F(0), F(0), F(1, 2)),
            1,
            full_dual(product_fan(SMOOTH_2D)[1]),
            complete=True,
            keys=[[(0, 0, 0, 1)]],
            decision_log=smooth_log,
        )
        smooth_entry = next(
            entry for entry in smooth_log if entry["route"] == "smooth_half_ray"
        )
        self.assertIs(smooth_entry["containment_generic"], True)
        self.assertEqual(
            smooth_entry["containment_method"],
            "eq_4.46_after_smooth_half_ray_eq_4.35",
        )

    def test_s10_incomplete_false_is_gated_before_section_cache_and_euler(self):
        cones, rays = product_fan(DIAG_2D)
        sigma = [(1, 1, 0, 0)]
        nu = (F(1, 2), F(-1, 2), F(0), F(0))
        full_rows = full_dual(rays)
        rows_incomplete = [
            row
            for row in full_rows
            if row not in ((-1, 0, 0, 0), (0, -1, 0, 0))
        ]
        # The projected-lattice enumerator chooses the opposite representative
        # in the same P_-N / 2P_-N class as the SPEC's displayed nu.
        nu_json = g._fraction_vector_to_json((-nu[0], -nu[1], nu[2], nu[3]))
        selected_cases = []
        for rows, complete in ((rows_incomplete, False), (full_rows, True)):
            decision_log = []
            records, auxiliary, _, _ = build(
                DIAG_2D,
                SWAP,
                ZERO_T,
                1,
                rows,
                complete=complete,
                keys=[sigma],
                decision_log=decision_log,
            )
            component = next(
                item
                for item in records
                if item["sigma_rays"] == [list(sigma[0])] and item["nu"] == nu_json
            )
            log_entry = next(
                item
                for item in decision_log
                if item["sigma_rays"] == [list(sigma[0])] and item["nu"] == nu_json
            )
            self.assertEqual(log_entry["route"], "non_smooth_fallback")
            self.assertIs(component["f_vanishes_identically"], False)
            self.assertEqual(component["containment_method"], MATRIX_NONZERO)
            self.assertIs(component["containment_witness"]["support_complete"], complete)
            selected_cases.append((component, auxiliary, complete))

        incomplete_component, auxiliary, _ = selected_cases[0]
        section_cache = {}
        with mock.patch.object(
            g,
            "_positive_component_section_certificate",
            side_effect=AssertionError("incomplete support must not request a section certificate"),
        ) as section_certificate:
            result = g.classify_smoothness(
                SWAP,
                ZERO_T,
                1,
                auxiliary,
                [incomplete_component],
                {"non_smooth_facet_dual_vertices": set()},
                [],
                section_cache=section_cache,
            )
            euler = toric_euler.exact_fixed_locus_euler(
                auxiliary,
                SWAP,
                [incomplete_component],
            )
        self.assertEqual(section_certificate.call_count, 0)
        self.assertEqual(section_cache, {})
        section = result["positive_component_checks"][0]
        self.assertEqual(section["status"], "unavailable")
        self.assertEqual(section["reason_code"], SUPPORT_GATE)
        self.assertEqual(euler["status"], "unavailable")
        self.assertEqual(euler["components"][0]["euler_reason_code"], SUPPORT_GATE)
        self.assertTrue(
            all(item["euler_reason_code"] == SUPPORT_GATE for item in euler["components"])
        )

        complete_component, auxiliary, _ = selected_cases[1]
        with mock.patch.object(
            g,
            "_positive_component_section_certificate",
            return_value={
                "status": "unavailable",
                "reason_code": "synthetic_control",
                "reason": "synthetic test stub",
            },
        ) as section_certificate:
            complete_result = g.classify_smoothness(
                SWAP,
                ZERO_T,
                1,
                auxiliary,
                [complete_component],
                {"non_smooth_facet_dual_vertices": set()},
                [],
                section_cache={},
            )
            complete_euler = toric_euler.exact_fixed_locus_euler(
                auxiliary,
                SWAP,
                [complete_component],
            )
        self.assertGreater(section_certificate.call_count, 0)
        self.assertNotIn(
            SUPPORT_GATE,
            [item.get("reason_code") for item in complete_result["positive_component_checks"]],
        )
        self.assertNotEqual(complete_euler.get("reason_code"), SUPPORT_GATE)

    def test_support_gate_covers_missing_and_undetermined_witness_fields(self):
        base = {
            "f_vanishes_identically": False,
            "containment_method": MATRIX_NONZERO,
            "sigma_rays": [],
            "sigma_dimension": 0,
            "nu": [0, 0, 0, 0],
            "fixed_toric_dimension": 1,
            "invariant_restricted_support": {},
            "cox_phase_witness": {},
            "fixed_cone_lattice": {},
        }
        cases = [deepcopy(base)]
        with_none = deepcopy(base)
        with_none["containment_witness"] = None
        cases.append(with_none)
        with_undetermined = deepcopy(base)
        with_undetermined["containment_witness"] = {"support_complete": None}
        cases.append(with_undetermined)

        self.assertFalse(
            g._matrix_false_support_incomplete(
                {**base, "containment_method": SINGLETON}
            )
        )
        for component in cases:
            self.assertTrue(g._matrix_false_support_incomplete(component))
            with mock.patch.object(
                g,
                "_positive_component_section_certificate",
                side_effect=AssertionError("incomplete support must stop before section certification"),
            ):
                classified = g.classify_smoothness(
                    SWAP,
                    ZERO_T,
                    1,
                    [],
                    [component],
                    {"non_smooth_facet_dual_vertices": set()},
                    [],
                )
            self.assertEqual(
                classified["positive_component_checks"][0]["reason_code"],
                SUPPORT_GATE,
            )
            euler = toric_euler.exact_fixed_locus_euler([], SWAP, [component])
            self.assertEqual(euler["status"], "unavailable")
            self.assertEqual(euler["components"][0]["euler_reason_code"], SUPPORT_GATE)

    def test_p_i3_a_unlisted_failed_check_fails_closed(self):
        result = g.decide_from_columns([], ["unlisted_failure"], None)

        self.assertIsNone(result[0])
        self.assertEqual(result[1], "containment_certificate_conflict")
        self.assertEqual(result[2], "certificate_conflict")
        self.assertIsNone(result[3])
        self.assertEqual(
            result[4],
            {
                "failed_checks": ["certificate_conflict"],
                "offending_codes": ["unlisted_failure"],
                "totality_violation": True,
            },
        )

    def test_p_i3_b_bare_string_failed_checks_fails_closed(self):
        result = g.decide_from_columns([], "action_not_validated", None)

        self.assertIsNone(result[0])
        self.assertEqual(result[1], "containment_certificate_conflict")
        self.assertEqual(result[2], "certificate_conflict")
        self.assertIsNone(result[3])
        self.assertEqual(
            result[4],
            {
                "failed_checks": ["certificate_conflict"],
                "offending_codes": ["action_not_validated"],
                "totality_violation": True,
            },
        )

    def test_p_i3_c_matrix_route_exception_fails_component_closed(self):
        rows = full_dual(product_fan(DIAG_2D)[1])
        decision_log = []
        with mock.patch.object(
            g,
            "restriction_matrix_decision",
            side_effect=RuntimeError("synthetic matrix route failure"),
        ) as matrix_route:
            records, _, _, _ = build(
                DIAG_2D,
                MINUS_I,
                ZERO_T,
                1,
                rows,
                complete=True,
                decision_log=decision_log,
            )

        self.assertTrue(records)
        self.assertGreater(matrix_route.call_count, 0)
        self.assertEqual(len(records), 16)
        for component in records:
            self.assertIsNone(component["f_vanishes_identically"])
            self.assertEqual(
                component["containment_method"], "containment_certificate_conflict"
            )
            self.assertEqual(component["containment_reason_code"], "certificate_conflict")
            self.assertIsNone(component["containment_generic"])
            self.assertEqual(
                component["containment_witness"],
                {
                    "failed_checks": ["certificate_conflict"],
                    "totality_violation": True,
                    "exception_type": "RuntimeError",
                },
            )

    def test_a310_memory_error_propagates_from_matrix_route(self):
        rows = full_dual(product_fan(DIAG_2D)[1])
        with mock.patch.object(
            g,
            "restriction_matrix_decision",
            side_effect=MemoryError("synthetic matrix-route resource failure"),
        ):
            with self.assertRaises(MemoryError):
                build(DIAG_2D, MINUS_I, ZERO_T, 1, rows, complete=True)

    def test_p_i3_t8_non_string_failed_checks_record_type_names(self):
        cases = (
            (None, ["NoneType"]),
            (iter([]), ["list_iterator"]),
            (["action_not_validated", 3], ["int"]),
        )
        for failed_checks, expected in cases:
            with self.subTest(failed_checks=type(failed_checks).__name__):
                result = g.decide_from_columns([], failed_checks, None)
                self.assertIsNone(result[0])
                self.assertEqual(result[1], "containment_certificate_conflict")
                self.assertEqual(result[2], "certificate_conflict")
                self.assertIsNone(result[3])
                self.assertEqual(result[4]["offending_codes"], expected)

    def test_p_i3_d_vertices_receipt_cannot_claim_completeness(self):
        rows = full_dual(product_fan(DIAG_2D)[1])
        records, _, _, _ = build(
            DIAG_2D,
            MINUS_I,
            ZERO_T,
            1,
            rows,
            complete=True,
            method="vertices",
            source="cytools_dual_points",
        )

        self.assertEqual(len(records), 16)
        for component in records:
            self.assertIsNone(component["f_vanishes_identically"])
            self.assertEqual(component["containment_method"], UNDETERMINED)
            self.assertEqual(
                component["containment_reason_code"], "dual_points_incomplete"
            )
            self.assertIsNone(component["containment_generic"])
            provenance = component["containment_witness"]["dual_points_provenance"]
            self.assertEqual(provenance["method"], "vertices")
            self.assertIs(provenance["complete"], False)

    def test_p_i3_e_completeness_flag_without_provenance_is_incomplete(self):
        cones, rays = product_fan(DIAG_2D)
        rows = full_dual(rays)
        auxiliary = g.build_auxiliary_fan(cones, MINUS_I)
        records = g._fixed_component_records(
            auxiliary,
            MINUS_I,
            ZERO_T,
            1,
            fixed_cone_keys=g._pointwise_invariant_cone_keys(cones, MINUS_I),
            dual_points=rows,
            ambient_rays=rays,
            fan_cones=cones,
            dual_points_complete=True,
            dual_points_provenance=None,
        )

        self.assertEqual(len(records), 16)
        for component in records:
            self.assertIsNone(component["f_vanishes_identically"])
            self.assertEqual(component["containment_method"], UNDETERMINED)
            self.assertEqual(
                component["containment_reason_code"], "dual_points_incomplete"
            )
            self.assertIsNone(component["containment_generic"])
            self.assertNotIn(
                "dual_points_provenance", component["containment_witness"]
            )


if __name__ == "__main__":
    unittest.main()
