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


def build(c2, matrix, shift, lambda_f, rows, *, complete=False, method="fixture_attested", source="fixture_attested", keys=None, decision_log=None):
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
        ambient_rays=rays,
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
        result = direct_matrix_decision(
            rows=rows,
            rays=rays,
            cones=cones,
            lambda_f=0,
            provenance=provenance,
            fast_path=(True, DENSE),
        )

        value, method, reason, generic, witness = result
        self.assertIsNone(value)
        self.assertEqual(method, "containment_certificate_conflict")
        self.assertEqual(reason, "certificate_conflict")
        self.assertIsNone(generic)
        self.assertEqual(witness["failed_checks"], ["certificate_conflict"])
        self.assertEqual(
            witness["fast_path_certificate"], {"value": True, "method": DENSE}
        )
        self.assertIs(witness["matrix_certificate"]["value"], False)
        self.assertEqual(witness["matrix_certificate"]["method"], MATRIX_NONZERO)

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
        cases.append(
            (
                "pair_phase_inconsistent",
                direct_matrix_decision(
                    rows=[(1, 0, 0, 0)],
                    rays=rays,
                    cones=cones,
                    provenance=fixture_provenance([(1, 0, 0, 0)], complete=False),
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
        vertex_rows = [
            row for row in full_rows if row not in ((0, 1, 0, 0), (0, -1, 0, 0))
        ]
        boundary_sigma = [(1, 1, 0, 0)]
        boundary_shift = (F(1, 4), F(1, 4), F(0), F(0))
        cases = [
            (MINUS_I, ZERO_T, 0, full_rows, True, "fixture_attested", None),
            (MINUS_I, ZERO_T, 0, [(1, 0, 0, 0), (-1, 0, 0, 0)], False, "fixture_attested", None),
            (MINUS_I, ZERO_T, 1, incomplete_pair_rows, False, "fixture_attested", None),
            (MINUS_I, ZERO_T, 1, vertex_rows, False, "vertices", None),
            (IDENTITY, boundary_shift, 1, full_rows, True, "fixture_attested", [boundary_sigma]),
            (SWAP, ZERO_T, 1, full_rows, True, "fixture_attested", [boundary_sigma]),
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


if __name__ == "__main__":
    unittest.main()
