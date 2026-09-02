"""CYTools replay tests against the immutable h11=2 source rows."""

from __future__ import annotations

import copy
from fractions import Fraction
from itertools import combinations
import unittest

import numpy as np

from mpcp_bounded_analysis import (
    _construct_polytope,
    _construct_selected_triangulation,
    _source_identity_evidence,
    validate_replay_certificate,
)
from mpcp_bounded_analysis import point_identity, run_bounded_analysis
from mpcp_immutable_source import source_records


class ImmutableCYToolsMPCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cytools  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"CYTools is unavailable: {exc}")

    def test_three_source_rows_replay_both_boundary_frsts(self):
        result = run_bounded_analysis(
            source_records(),
            caps={"max_triangulations": 8, "max_seconds_per_index": 300},
        )
        self.assertEqual(set(result["reports"]), {"26", "31", "33"})
        expected_hodge = {
            "26": {"h11": 2, "h21": 120, "chi": -236},
            "31": {"h11": 2, "h21": 128, "chi": -252},
            "33": {"h11": 2, "h21": 132, "chi": -260},
        }
        for index, report in result["reports"].items():
            self.assertEqual(report["source_identity"]["status"], "source_identity_ready")
            self.assertEqual(report["source_identity"]["global_point_count"], 8)
            self.assertEqual(report["height_one_points"]["height_one_point_count"], 7)
            self.assertEqual(report["polytope_hodge"]["status"], "matched")
            self.assertEqual(report["polytope_hodge"]["observed"], expected_hodge[index])
            omission = report["selected_frst"]["omitted_point_facet_certificate"]
            self.assertEqual(omission["status"], "omitted_facet_interior_points_certified")
            self.assertEqual(omission["omitted_point_count"], 1)
            self.assertEqual(omission["omitted_points"], [report["source_identity"]["global_points"][7]])
            self.assertTrue(all(
                certificate["facet_interior"] for certificate in omission["certificates"]
            ))
            self.assertEqual(report["triangulation_enumeration"]["kwargs"]["include_points_interior_to_facets"], False)
            self.assertEqual(report["source_frst_catalog"]["status"], "matched")
            self.assertEqual(report["source_frst_catalog"]["observed_count"], 2)
            self.assertEqual(report["dual_action_checks"][0]["status"], "dual_check_verified")
            self.assertEqual(
                {row["point_count"] for row in report["refinement_records"]}, {7}
            )
            self.assertEqual(
                {row["terminal_status"] for row in report["refinement_records"]},
                {"refinement_enumerated"},
            )
            self.assertEqual(
                len(report["replay_certificates"]),
                2,
            )
            if report["replay_certificates"]:
                certificate = report["replay_certificates"][0]
                action_record = next(
                    action for action in report["action_records"]
                    if action.get("frst_hash") == certificate["frst"]["frst_hash"]
                    and action.get("terminal_status") == "refined_action_evaluated"
                )
                checked = validate_replay_certificate(
                    certificate,
                    report=report,
                    frst_hash=action_record["frst_hash"],
                    action=action_record["action"],
                    action_record=action_record,
                )
                self.assertEqual(checked["status"], "valid")
                self.assertEqual(
                    certificate["result"]["chi_F_I"],
                    {"26": 248, "31": 264, "33": 272}[index],
                )
                self.assertEqual(certificate["result"]["hodge_split"]["h21_plus"], 0)
            for action in report["action_records"]:
                fixed = action["fixed_locus_euler"]
                self.assertEqual(
                    len(fixed["components"]), fixed["component_count"]
                )
                self.assertTrue(all(
                    row["euler_status"] in {"computed", "unavailable"}
                    for row in fixed["components"]
                ))
                self.assertEqual(
                    fixed["certified_component_count"],
                    sum(row["euler_status"] == "computed" for row in fixed["components"]),
                )
            if index == "33":
                a_actions = [
                    action for action in report["action_records"]
                    if action.get("candidate_index") == 0
                ]
                self.assertEqual(len(a_actions), 1)
                fixed_a = a_actions[0]["fixed_locus_euler"]
                for expected_rays, determinant in (
                    (
                        ((-2, -1, 0, 0), (-2, 1, -1, -1), (0, 0, 0, 1), (0, 0, 1, 0)),
                        4,
                    ),
                    (
                        ((-2, 1, -1, -1), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0)),
                        2,
                    ),
                ):
                    target = next(
                        row for row in fixed_a["components"]
                        if {
                            tuple(ray) for ray in row["sigma_rays"]
                        } == set(expected_rays)
                    )
                    self.assertFalse(target["contained_in_hypersurface"])
                    self.assertEqual(
                        target["containment_method"],
                        "exact_eq_4.42_invariant_restricted_support",
                    )
                    support = target["invariant_restricted_support"]
                    self.assertEqual(support["status"], "certified")
                    self.assertEqual(len(support["support"]), 1)
                    self.assertEqual(support["support"][0]["q"], [1, -1, -1, -1])
                    exponents = dict(zip(
                        (tuple(ray) for ray in support["ambient_rays"]),
                        support["support"][0]["cox_exponents"],
                    ))
                    self.assertEqual(exponents[(1, 0, 0, 0)], 2)
                    self.assertEqual(target["euler_status"], "computed")
                    self.assertEqual(target["chi"], 0)
                    self.assertEqual(
                        abs(int(round(np.linalg.det(np.asarray(target["sigma_rays"], dtype=int))))),
                        determinant,
                    )
                self.assertEqual(fixed_a["status"], "computed")
                self.assertEqual(fixed_a["chi_F_I"], 272)
                self.assertEqual(
                    a_actions[0]["hodge_split"]["h21_plus"], 0
                )
                self.assertEqual(
                    tuple(
                        a_actions[0]["hodge_split"][key]
                        for key in ("h11_plus", "h11_minus", "h21_plus", "h21_minus")
                    ),
                    (2, 0, 0, 132),
                )

                b_actions = [
                    action for action in report["action_records"]
                    if action.get("candidate_index") == 1
                ]
                self.assertEqual(len(b_actions), 1)
                fixed_b = b_actions[0]["fixed_locus_euler"]
                surface = next(
                    row for row in fixed_b["components"]
                    if row["ambient_component_dimension"] == 2
                )
                self.assertFalse(surface["contained_in_hypersurface"])
                self.assertEqual(
                    surface["containment_method"],
                    "exact_eq_4.42_invariant_restricted_support",
                )
                self.assertEqual(
                    surface["invariant_restricted_support"]["support"][0]["q"],
                    [1, -1, -1, -1],
                )
                self.assertEqual(
                    len(surface["invariant_restricted_support"]["support"]), 1
                )
                self.assertEqual(surface["euler_status"], "computed")
                self.assertEqual(surface["chi"], 0)
                self.assertEqual(fixed_b["status"], "computed")
                self.assertEqual(fixed_b["chi_F_I"], 272)
                self.assertEqual(
                    tuple(
                        b_actions[0]["hodge_split"][key]
                        for key in ("h11_plus", "h11_minus", "h21_plus", "h21_minus")
                    ),
                    (2, 0, 0, 132),
                )
                self.assertNotIn("fixed_surface_n_s_diagnostics", fixed_b)

    def test_old_11_11_10_manifests_are_terminal(self):
        records = source_records()
        wrong_counts = {26: 11, 31: 11, 33: 10}
        wrong = {}
        for index, count in wrong_counts.items():
            record = copy.deepcopy(records[index])
            points = record["source"]["global_points"]
            while len(points) < count:
                points.append([100 + len(points), 0, 0, 0])
            record["source"]["global_points"] = points[:count]
            record["source"]["polytope_id"] = f"wrong-{index}-{count}"
            wrong[index] = record
        result = run_bounded_analysis(wrong)
        for index in ("26", "31", "33"):
            report = result["reports"][index]
            self.assertTrue(report["terminal_records"])
            self.assertEqual(report.get("refinement_records", []), [])
            self.assertIn(
                report["terminal_records"][0]["terminal_status"],
                {"source_polytope_id_mismatch", "source_point_count_mismatch"},
            )

    @staticmethod
    def _exact_determinant(matrix):
        """Evaluate a small integer determinant without floating-point rounding."""
        rows = [list(map(int, row)) for row in np.asarray(matrix).tolist()]
        size = len(rows)
        if size == 0:
            return 1
        sign = 1
        previous = 1
        for pivot_index in range(size - 1):
            if rows[pivot_index][pivot_index] == 0:
                swap = next(
                    (
                        index
                        for index in range(pivot_index + 1, size)
                        if rows[index][pivot_index] != 0
                    ),
                    None,
                )
                if swap is None:
                    return 0
                rows[pivot_index], rows[swap] = rows[swap], rows[pivot_index]
                sign = -sign
            pivot = rows[pivot_index][pivot_index]
            for row_index in range(pivot_index + 1, size):
                row_head = rows[row_index][pivot_index]
                for column_index in range(pivot_index + 1, size):
                    rows[row_index][column_index] = (
                        rows[row_index][column_index] * pivot
                        - row_head * rows[pivot_index][column_index]
                    ) // previous
            previous = pivot
        return sign * rows[-1][-1]

    @classmethod
    def _maximal_minor_index(cls, matrix):
        """Return the exact finite-index order from all maximal minors."""
        matrix = np.asarray(matrix, dtype=int)
        rank = min(matrix.shape)
        if matrix.shape[0] <= matrix.shape[1]:
            minors = (matrix[:, list(columns)] for columns in combinations(range(matrix.shape[1]), rank))
        else:
            minors = (matrix[list(rows), :] for rows in combinations(range(matrix.shape[0]), rank))
        values = [abs(cls._exact_determinant(minor)) for minor in minors]
        order = 0
        for value in values:
            order = int(np.gcd(order, value))
        return order

    def _cox_phase_witness(self, triangulation, cone_labels, expected_phase):
        """Return the exact Cox chart and GLSM stabilizer witness."""
        cone_labels = tuple(int(label) for label in cone_labels)
        fan_cones = {
            tuple(int(label) for label in cone)
            for cone in triangulation.fan().cones()
        }
        containing_cones = [
            cone for cone in fan_cones if set(cone_labels).issubset(set(cone))
        ]
        self.assertTrue(containing_cones)
        chart_cone = min(containing_cones, key=lambda cone: (len(cone), cone))
        labels = tuple(range(1, 7))
        complement = tuple(label for label in labels if label not in cone_labels)
        chart_complement = tuple(label for label in labels if label not in chart_cone)
        q_matrix = np.asarray(
            triangulation.get_cy().glsm_charge_matrix(include_origin=False),
            dtype=int,
        )
        q_complement = q_matrix[:, [label - 1 for label in complement]]
        phase = tuple(Fraction(value) for value in expected_phase)
        pairings = tuple(
            sum(Fraction(q_complement[row, column]) * phase[row] for row in range(2))
            for column in range(q_complement.shape[1])
        )
        self.assertTrue(all(value.denominator == 1 for value in pairings))
        chart_coordinates = {
            label: (0 if label in chart_cone else 1) for label in labels
        }
        sr_generators = triangulation.sr_ideal()
        for generator in sr_generators:
            support = {int(value) for value in generator}
            self.assertFalse(
                support.issubset(set(chart_cone)),
                msg=f"SR generator {support} obstructs cone chart {chart_cone}",
            )
        self.assertEqual(
            tuple(chart_coordinates[label] for label in chart_complement),
            (1,) * len(chart_complement),
        )
        return {
            "cone_labels": cone_labels,
            "chart_cone_labels": chart_cone,
            "complement_labels": complement,
            "q_matrix": q_matrix.tolist(),
            "q_complement": q_complement.tolist(),
            "phase": [
                {"numerator": value.numerator, "denominator": value.denominator}
                for value in phase
            ],
            "phase_pairings": [
                {"numerator": value.numerator, "denominator": value.denominator}
                for value in pairings
            ],
            "stabilizer_order": self._maximal_minor_index(q_complement),
            "irrelevant_ideal_chart": {
                "status": "exists",
                "complement_monomial": list(chart_complement),
                "cox_coordinates": chart_coordinates,
            },
        }

    @staticmethod
    def _component_for_labels(fixed, points, labels):
        target = {tuple(int(value) for value in points[label]) for label in labels}
        return next(
            component
            for component in fixed["components"]
            if {
                tuple(int(value) for value in ray)
                for ray in component["sigma_rays"]
            }
            == target
        )

    def test_class33_a_b_exact_cox_phase_and_fail_closed_witnesses(self):
        """Audit both immutable FRST phases without selecting on Euler output."""
        record = source_records()[33]
        result = run_bounded_analysis(
            {33: record},
            caps={"max_triangulations": 8, "max_seconds_per_index": 300},
        )
        report = result["reports"]["33"]
        actions = {
            action["candidate_index"]: action
            for action in report["action_records"]
        }
        self.assertEqual(set(actions), {0, 1})
        expected_q = [[4, 1, 1, 1, 1, 0], [-2, -1, -1, -1, 0, 1]]
        phase_fixtures = {
            "a": (
                (2, 3, 4, 5),
                (Fraction(1, 4), Fraction(0)),
                [[4, 0], [-2, 1]],
                4,
            ),
            "a_det2": (
                (2, 3, 4, 6),
                (Fraction(0), Fraction(1, 2)),
                [[4, 1], [-2, 0]],
                2,
            ),
            "b_surface": (
                (5, 6),
                (Fraction(1, 2), Fraction(1, 2)),
                [[4, 1, 1, 1], [-2, -1, -1, -1]],
                2,
            ),
        }
        triangulations = {}
        for candidate_index, label in enumerate(("a", "b")):
            candidate = copy.deepcopy(record)
            candidate["selected_frst"] = candidate["selected_frsts"][candidate_index]
            source = _source_identity_evidence(33, candidate)
            polytope, _ = _construct_polytope(candidate, source)
            triangulation, _ = _construct_selected_triangulation(polytope, candidate)
            triangulations[label] = triangulation
            self.assertEqual(
                np.asarray(
                    triangulation.get_cy().glsm_charge_matrix(include_origin=False),
                    dtype=int,
                ).tolist(),
                expected_q,
            )
        for fixture_name, (cone, phase, expected_complement_q, expected_order) in phase_fixtures.items():
            label = "b" if fixture_name == "b_surface" else "a"
            witness = self._cox_phase_witness(
                triangulations[label], cone, phase
            )
            self.assertEqual(witness["q_complement"], expected_complement_q)
            self.assertEqual(witness["stabilizer_order"], expected_order)
            self.assertEqual(witness["irrelevant_ideal_chart"]["status"], "exists")
            points = np.asarray(triangulations[label].points(), dtype=int)
            sigma_matrix = points[list(cone)]
            if len(cone) == 4:
                self.assertEqual(abs(self._exact_determinant(sigma_matrix)), expected_order)
            else:
                self.assertEqual(self._maximal_minor_index(sigma_matrix), expected_order)
            self.assertTrue(
                all(
                    Fraction(item["numerator"], item["denominator"]).denominator == 1
                    for item in witness["phase_pairings"]
                )
            )

        points_a = np.asarray(triangulations["a"].points(), dtype=int)
        fixed_a = actions[0]["fixed_locus_euler"]
        for labels, determinant in (((2, 3, 4, 5), 4), ((2, 3, 4, 6), 2)):
            component = self._component_for_labels(fixed_a, points_a, labels)
            self.assertFalse(component["contained_in_hypersurface"])
            support = component["invariant_restricted_support"]
            self.assertEqual(support["status"], "certified")
            self.assertEqual(support["support"][0]["q"], [1, -1, -1, -1])
            exponents = dict(zip(
                (tuple(ray) for ray in support["ambient_rays"]),
                support["support"][0]["cox_exponents"],
            ))
            self.assertEqual(exponents[(1, 0, 0, 0)], 2)
            self.assertTrue(support["chart"]["status"] == "certified")
            self.assertEqual(support["restriction_identically_zero"], False)
            self.assertEqual(component["ambient_component_dimension"], 0)
            self.assertEqual(component["euler_status"], "computed")
            self.assertEqual(component["chi"], 0)
            self.assertEqual(abs(self._exact_determinant(points_a[list(labels)])), determinant)
        self.assertEqual(fixed_a["status"], "computed")
        self.assertEqual(fixed_a["chi_F_I"], 272)

        points_b = np.asarray(triangulations["b"].points(), dtype=int)
        self.assertFalse(any(
            {1, 5, 6}.issubset(set(int(value) for value in simplex))
            for simplex in np.asarray(triangulations["b"].simplices(), dtype=int)
        ))
        fixed_b = actions[1]["fixed_locus_euler"]
        surface = self._component_for_labels(fixed_b, points_b, (5, 6))
        self.assertFalse(surface["contained_in_hypersurface"])
        self.assertEqual(surface["invariant_restricted_support"]["support"][0]["q"], [1, -1, -1, -1])
        self.assertEqual(surface["invariant_restricted_support"]["restriction_identically_zero"], False)
        self.assertEqual(surface["ambient_component_dimension"], 2)
        self.assertEqual(surface["euler_status"], "computed")
        self.assertEqual(surface["chi"], 0)
        self.assertEqual(fixed_b["status"], "computed")
        self.assertEqual(fixed_b["chi_F_I"], 272)
        self.assertNotIn("fixed_surface_n_s_diagnostics", fixed_b)
        self.assertEqual(len(report["replay_certificates"]), 2)
        self.assertTrue(all(
            certificate["certificate_schema_version"]
            == "cyaxiverse-bounded-mpcp-certificate-1.1"
            for certificate in report["replay_certificates"]
        ))

    def test_hodge_and_index_space_mismatches_are_terminal(self):
        hodge_wrong = copy.deepcopy(source_records()[26])
        hodge_wrong["source"]["expected_hodge"] = {"h11": 2, "h21": 121, "chi": -238}
        report = run_bounded_analysis({26: hodge_wrong})["reports"]["26"]
        self.assertIn("source_hodge_mismatch", {
            row["terminal_status"] for row in report["terminal_records"]
        })

        index_wrong = copy.deepcopy(source_records()[26])
        index_wrong["selected_frst"]["simplices_index_space"] = "standalone_triangulation_points"
        report = run_bounded_analysis({26: index_wrong})["reports"]["26"]
        self.assertIn("simplices_index_space_mismatch", {
            row["terminal_status"] for row in report["terminal_records"]
        })

        row_wrong = copy.deepcopy(source_records()[26])
        row_wrong["source"]["source_row"] = 22
        report = run_bounded_analysis({26: row_wrong})["reports"]["26"]
        self.assertIn("source_row_mismatch", {
            row["terminal_status"] for row in report["terminal_records"]
        })


if __name__ == "__main__":
    unittest.main()
