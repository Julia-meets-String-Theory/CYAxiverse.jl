"""Analytic and bounded replay fixtures for the MPCP driver."""

from __future__ import annotations

import unittest
import copy
from unittest import mock

import numpy as np

from mpcp_bounded_analysis import (
    CERTIFICATE_SCHEMA_VERSION,
    FORMULA_SCHEMA_VERSION,
    SUPPORTED_CYTOOLS_API_VERSION,
    _canonical_digest,
    _all_triangulations,
    analyze_replay_index,
    _certificate_key,
    build_replay_certificate,
    hodge_split_from_euler,
    _identity_conormal_n_s_terms,
    lower_subdivision_evidence,
    refined_glsm_evidence,
    run_bounded_analysis,
    symmetric_heights,
    triangulation_identity,
    validate_replay_certificate,
)
from orientifold_general_l_geometry import (
    _fixed_component_records,
    _cox_alpha_lattice_witness,
    invariant_restricted_monomial_support,
)
from mpcp_immutable_source import source_records


POINTS = np.asarray(
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.int64,
)


class _FakeCY:
    def __init__(self, labels=(1, 2, 3, 4)):
        self._labels = np.asarray(labels, dtype=np.int64)

    def glsm_charge_matrix(self, include_origin=False):
        del include_origin
        return np.eye(4, dtype=np.int64)

    def prime_toric_divisors(self):
        return self._labels

    def h11(self):
        return 4

    def h21(self):
        return 3

    def chi(self):
        return 2


class _FakeCYWithThreeLocalDivisors:
    """Expose local triangulation labels for the point-order regression."""

    def glsm_charge_matrix(self, include_origin=False):
        del include_origin
        return np.eye(3, dtype=np.int64)

    def prime_toric_divisors(self):
        return np.asarray([1, 2, 3], dtype=np.int64)


class _FakeTriangulation:
    def __init__(self, simplices, heights=None):
        self._simplices = np.asarray(simplices, dtype=np.int64)
        self._heights = np.asarray(heights if heights is not None else [0, 1, 2, 3, 4])

    def points(self):
        return POINTS

    def simplices(self, as_indices=False):
        del as_indices
        return self._simplices

    def heights(self):
        return self._heights

    def is_fine(self):
        return True

    def is_regular(self):
        return True

    def is_star(self):
        return True

    def is_valid(self):
        return True

    def get_cy(self):
        return _FakeCY()


class _FakePolytope:
    def __init__(self, triangulations):
        self._triangulations = tuple(triangulations)

    def points(self):
        return POINTS

    def all_triangulations(self, only_fine=True, only_regular=True, only_star=True,
                           include_points_interior_to_facets=False):
        self.seen_filters = (only_fine, only_regular, only_star,
                             include_points_interior_to_facets)
        return iter(self._triangulations)


class _FakeSubdivision:
    def cells(self):
        return [[0, 1, 2, 3, 4], [0, 1, 2, 3]]


class _FakeVectorConfiguration:
    def subdivide(self, heights, backend="ppl", make_fine=False, check_heights=False, cure_heights=True):
        del heights, backend, make_fine, check_heights, cure_heights
        return _FakeSubdivision()


class _FakePolytopeWithSubdivision(_FakePolytope):
    def vc(self):
        return _FakeVectorConfiguration()


ACTION = {
    "lattice_matrix": np.eye(4, dtype=np.int64).tolist(),
    "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 2},
    "lambda_f": 1,
}


class InvariantRestrictedSupportFixtures(unittest.TestCase):
    """Generic exact-support fixtures independent of KS class counts."""

    def test_eq_4_30_cox_witness_retains_solved_lattice_vector_sign(self):
        witness = _cox_alpha_lattice_witness(
            ((1, 0, 0, 0),),
            (0, 1, 0, 0),
        )
        self.assertEqual(witness["status"], "certified")
        self.assertEqual(witness["alpha"], [0])
        self.assertEqual(witness["lattice_vector"], [0, 1, 0, 0])

    def test_singleton_q0_support_has_x1_squared_and_chart(self):
        q0 = (1, -1, -1, -1)
        ambient_rays = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
        support = invariant_restricted_monomial_support(
            [q0],
            ambient_rays,
            ambient_rays[1:],
            np.eye(4, dtype=int),
            (0, 0, 0, 0),
            0,
            fan_cones=[ambient_rays],
        )
        self.assertEqual(support["status"], "certified")
        self.assertEqual(support["support"][0]["q"], list(q0))
        self.assertEqual(support["support"][0]["cox_exponents"], [2, 0, 0, 0])
        self.assertFalse(support["restriction_identically_zero"])
        self.assertEqual(support["chart"]["status"], "certified")

    def test_nonsmooth_empty_support_certifies_containment_without_parity(self):
        # The rank-two cone has index two.  Both available dual monomials have
        # a positive exponent on one of its zero Cox coordinates, so the exact
        # restricted support is empty even though no smooth-half-ray shortcut
        # is available.
        sigma = ((1, 0, 0, 0), (1, 2, 0, 0))
        ambient_rays = sigma + ((0, 0, 1, 0), (0, 0, 0, 1))
        dual_points = [(1, 0, -1, -1), (0, 0, 0, 0)]
        fan_cones = [ambient_rays]
        components = _fixed_component_records(
            [{"rays": [list(ray) for ray in ambient_rays], "dimension": 2,
              "simplicial": True, "ambient_cones": [
                  [list(ray) for ray in ambient_rays]
              ]}],
            np.eye(4, dtype=int),
            (0, 0, 0, 0),
            0,
            fixed_cone_keys=(sigma,),
            dual_points=dual_points,
            ambient_rays=ambient_rays,
            fan_cones=fan_cones,
        )
        self.assertEqual(len(components), 1)
        component = components[0]
        self.assertEqual(component["containment_method"], "exact_eq_4.42_invariant_restricted_support")
        self.assertEqual(component["fixed_cone_lattice"]["stabilizer_order"], 2)
        self.assertTrue(component["f_vanishes_identically"])
        self.assertEqual(component["invariant_restricted_support"]["support"], [])
        self.assertTrue(component["invariant_restricted_support"]["restriction_identically_zero"])
        lambda_one = _fixed_component_records(
            [{"rays": [list(ray) for ray in ambient_rays], "dimension": 2,
              "simplicial": True, "ambient_cones": [
                  [list(ray) for ray in ambient_rays]
              ]}],
            np.eye(4, dtype=int),
            (0, 0, 0, 0),
            1,
            fixed_cone_keys=(sigma,),
            dual_points=dual_points,
            ambient_rays=ambient_rays,
            fan_cones=fan_cones,
        )[0]
        self.assertEqual(
            lambda_one["containment_method"],
            "exact_eq_4.42_invariant_restricted_support",
        )
        self.assertTrue(lambda_one["f_vanishes_identically"])

    def test_bounded_structural_rank_three_four_support_reaudit(self):
        """Re-audit two bounded structural ranks without count-based tuning."""
        ambient_rays = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
        q0 = (1, -1, -1, -1)
        outcomes = []
        for sigma in ((ambient_rays[1],), ()):
            support = invariant_restricted_monomial_support(
                [q0],
                ambient_rays,
                sigma,
                np.eye(4, dtype=int),
                (0, 0, 0, 0),
                0,
                fan_cones=[ambient_rays],
            )
            outcomes.append({
                "component_rank": 4 - len(sigma),
                "support_count": len(support["support"]),
                "restriction_identically_zero": support["restriction_identically_zero"],
                "chart_status": support["chart"]["status"],
            })
        self.assertEqual(outcomes, [
            {
                "component_rank": 3,
                "support_count": 1,
                "restriction_identically_zero": False,
                "chart_status": "certified",
            },
            {
                "component_rank": 4,
                "support_count": 1,
                "restriction_identically_zero": False,
                "chart_status": "certified",
            },
        ])


def _certificate_fixture():
    """Build one certificate without importing CYTools or writing artifacts."""
    record = source_records()[26]
    source = record["source"]
    report = {
        "schema_version": "cyaxiverse-bounded-mpcp-replay-1.3",
        "runtime_provenance": {
            "cytools_version_guard": {
                "status": "verified",
                "expected": SUPPORTED_CYTOOLS_API_VERSION,
                "observed": SUPPORTED_CYTOOLS_API_VERSION,
                "reason": None,
            },
        },
        "caps": {"max_triangulations": 8},
        "source_identity": {
            "source_sha256": source["parquet_sha256"],
            "source_row": source["source_row"],
            "polytope_id": source["polytope_id"],
            "global_points": source["global_points"],
        },
        "selected_frst": {"identity": record["selected_frst"]["identity"]},
    }
    action_record = {
        "terminal_status": "refined_action_evaluated",
        "candidate_index": 1,
        "frst_hash": record["selected_frsts"][1]["identity"],
        "action": copy.deepcopy(ACTION),
        "fixed_locus_euler": {
            "status": "computed",
            "chi_F_I": 248,
            "components": [{"euler_status": "computed", "chi": 248}],
        },
        "refined_glsm": {
            "status": "refined_h2_action_verified",
            "h2_matrix": [[1, 0], [0, 1]],
            "proof": {"exact_residual_zero": True},
        },
        "hodge_split": {
            "h11_plus": 2,
            "h11_minus": 0,
            "h21_plus": 0,
            "h21_minus": 120,
            "chi_fixed_locus": 248,
            "chi_x": -236,
        },
    }
    certificate = build_replay_certificate(26, record, report, action_record)
    assert certificate is not None
    return record, report, action_record, certificate


def _reseal_certificate(certificate):
    """Recompute certificate digests for adversarial binding tests."""

    certificate["certificate_key"] = _certificate_key(certificate)
    certificate["certificate_key_digest"] = _canonical_digest(
        certificate["certificate_key"]
    )
    certificate["certificate_digest"] = _canonical_digest({
        key: value
        for key, value in certificate.items()
        if key not in {"certificate_digest", "certificate_key_digest"}
    })
    return certificate


class AnalyticMPCPTests(unittest.TestCase):
    def test_eq_4_50_identity_conormal_sign_fixture(self):
        tensor = np.zeros((4, 4, 4, 4), dtype=object)
        tensor[1, 2, 1, 2] = 1
        tensor[1, 2, 2, 2] = -1
        terms = _identity_conormal_n_s_terms(tensor, 1, 2, 3)
        self.assertEqual(terms, (0, 0, 0, 1, 1))

    def test_symmetric_height_formula_is_exact_and_invariant(self):
        poly = _FakePolytope([])
        tri = _FakeTriangulation([[0, 1, 2, 3, 4]], heights=[0, 1, 2, 3, 4])
        evidence = symmetric_heights(poly, tri, np.eye(4, dtype=np.int64))
        self.assertEqual(evidence["status"], "symmetric_heights_ready")
        self.assertEqual(
            evidence["heights"],
            [{"numerator": value, "denominator": 1} for value in range(5)],
        )
        self.assertTrue(evidence["invariant_exact"])
        self.assertIn("Moritz", evidence["source_anchor"])

    def test_full_refined_glsm_relation_includes_all_columns(self):
        action = np.eye(4, dtype=np.int64)
        action[[0, 1]] = action[[1, 0]]
        cy = _FakeCY()
        evidence = refined_glsm_evidence(cy, action, POINTS, original_prime_labels=[1, 2, 3])
        self.assertEqual(evidence["status"], "refined_h2_action_verified")
        self.assertEqual(evidence["Q_prime_shape"], [4, 4])
        self.assertEqual(evidence["prime_image_indices"], [1, 0, 2, 3])
        self.assertEqual(evidence["exceptional_prime_labels"], [4])
        self.assertTrue(evidence["proof"]["exact_residual_zero"])
        self.assertEqual(evidence["proof"]["equation"], "M Q_prime = Q_prime P")

    def test_lower_backend_retains_a_degenerate_non_simplicial_cell(self):
        tri = _FakeTriangulation([[0, 1, 2, 3, 4]], heights=[0, 1, 2, 3, 4])
        poly = _FakePolytopeWithSubdivision([])
        heights = symmetric_heights(poly, tri, np.eye(4, dtype=np.int64))
        lower = lower_subdivision_evidence(poly, heights)
        self.assertEqual(lower["status"], "lower_subdivision_retained")
        self.assertFalse(lower["simplicial"])
        self.assertFalse(lower["tie_breaking"])
        self.assertEqual(lower["backend"], "ppl")

    def test_lower_backend_reports_the_declared_cell_cap(self):
        tri = _FakeTriangulation([[0, 1, 2, 3, 4]], heights=[0, 1, 2, 3, 4])
        poly = _FakePolytopeWithSubdivision([])
        heights = symmetric_heights(poly, tri, np.eye(4, dtype=np.int64))
        lower = lower_subdivision_evidence(poly, heights, max_cells=1)
        self.assertEqual(lower["status"], "resource_capped_cells")
        self.assertEqual(lower["cell_count"], 2)
        self.assertEqual(len(lower["cells"]), 1)
        self.assertTrue(lower["cells_truncated"])
        self.assertTrue(lower["terminal"])
        self.assertFalse(lower["certification_allowed"])

    def test_triangulation_cap_is_terminal_and_not_exhaustive(self):
        tri = _FakeTriangulation([[0, 1, 2, 3, 4]])
        poly = _FakePolytope([tri])
        records, enumeration = _all_triangulations(
            poly, cap=0, deadline=float("inf")
        )
        self.assertEqual(records, [])
        self.assertEqual(enumeration["status"], "resource_capped")
        self.assertTrue(enumeration["terminal"])
        self.assertFalse(enumeration["complete"])
        self.assertFalse(enumeration["certification_allowed"])

    def test_top_level_triangulation_cap_is_terminal_and_uncertified(self):
        poly = _FakePolytope([])
        selected = _FakeTriangulation([[0, 1, 2, 3, 4]])
        source_identity = {
            "terminal": False,
            "expected_hodge": {"h11": 2, "h21": 120, "chi": -236},
            "expected_point_count": 5,
            "expected_boundary_point_count": 5,
            "global_point_count": 5,
            "global_points": POINTS.tolist(),
            "polytope_id": "fixture-polytope",
            "source_sha256": "fixture-source",
            "source_row": 21,
        }
        record = {"index": 26, "actions": [ACTION]}
        with mock.patch("mpcp_bounded_analysis._source_identity_evidence",
                        return_value=source_identity), \
             mock.patch("mpcp_bounded_analysis._construct_polytope",
                        return_value=(poly, {"status": "constructed"})), \
             mock.patch("mpcp_bounded_analysis._hodge_values",
                        return_value={"h11": 2, "h21": 120, "chi": -236}), \
             mock.patch("mpcp_bounded_analysis.height_one_point_evidence",
                        return_value={"height_one_point_count": 5}), \
             mock.patch("mpcp_bounded_analysis._construct_selected_triangulation",
                        return_value=(selected, {"status": "constructed"})), \
             mock.patch("mpcp_bounded_analysis._selected_identity_evidence",
                        return_value={"status": "matched"}), \
             mock.patch("mpcp_bounded_analysis.resolved_hodge_evidence",
                        return_value={"h11": 2, "h21": 120, "chi": -236}), \
             mock.patch("mpcp_bounded_analysis._hodge_match_evidence",
                        return_value={"status": "matched", "terminal": False}), \
             mock.patch("mpcp_bounded_analysis._points_from_object",
                        return_value=POINTS), \
             mock.patch("mpcp_bounded_analysis.omitted_point_facet_evidence",
                        return_value={"status": "omitted_facet_interior_points_certified",
                                      "triangulation_point_count": 5}), \
             mock.patch("mpcp_bounded_analysis._public_geometry_evidence",
                        return_value={}), \
             mock.patch("mpcp_bounded_analysis.dual_action_evidence",
                        return_value={"status": "verified", "terminal": False}), \
             mock.patch("mpcp_bounded_analysis._all_triangulations",
                        return_value=([], {"status": "resource_capped", "terminal": True,
                                          "complete": False,
                                          "certification_allowed": False})):
            report = analyze_replay_index(26, record,
                                          caps={"max_triangulations": 0})
        self.assertEqual(report["status"], "resource_capped")
        self.assertEqual(report["analysis_status"], "terminal_resource_capped")
        self.assertTrue(report["terminal"])
        self.assertFalse(report["complete"])
        self.assertFalse(report["certification_allowed"])
        self.assertIn(
            "resource_capped_triangulation_enumeration",
            {row["terminal_status"] for row in report["terminal_records"]},
        )

    def test_resolved_eq_4_51_does_not_assume_h11_two(self):
        split = hodge_split_from_euler(
            h11=4, h21=3, h11_minus=0, chi_fixed=10, chi_x=2
        )
        self.assertEqual(split["h11_plus"], 4)
        self.assertEqual(split["h21_minus"], 1)
        self.assertEqual(split["h21_plus"], 2)

    def test_eq_4_51_allows_negative_calabi_yau_euler(self):
        split = hodge_split_from_euler(
            h11=1, h21=101, h11_minus=0, chi_fixed=0, chi_x=-200
        )
        self.assertEqual(split["h21_minus"], 49)
        self.assertEqual(split["h21_plus"], 52)
        self.assertEqual(split["chi_x"], -200)

    def test_refined_glsm_uses_candidate_triangulation_point_order(self):
        origin = [0, 0, 0, 0]
        e1 = [1, 0, 0, 0]
        e2 = [0, 1, 0, 0]
        e3 = [0, 0, 1, 0]
        action = np.eye(4, dtype=np.int64)
        action[[0, 1]] = action[[1, 0]]
        # The full polytope has extra points before the triangulation labels.
        # Passing it would map prime label 3 to a nonexistent refined label.
        local_points = np.asarray([origin, e1, e2, e3], dtype=np.int64)
        evidence = refined_glsm_evidence(
            _FakeCYWithThreeLocalDivisors(), action, local_points,
            original_prime_labels=[1, 2],
            original_points=local_points[:3],
        )
        self.assertEqual(evidence["status"], "refined_h2_action_verified")
        self.assertEqual(evidence["prime_image_indices"], [1, 0, 2])
        self.assertEqual(evidence["exceptional_prime_labels"], [3])
        self.assertIn("coordinate comparison", evidence["exceptional_ray_comparison"])

    def test_bounded_replay_requires_source_identity_for_each_index(self):
        result = run_bounded_analysis({})
        self.assertEqual(set(result["reports"]), {"26", "31", "33"})
        for index in ("26", "31", "33"):
            report = result["reports"][index]
            self.assertEqual(report["scope_status"], "in_scope")
            self.assertIn(
                "source_identity_unavailable",
                {row["terminal_status"] for row in report["terminal_records"]},
            )

    def test_immutable_source_rows_are_exactly_the_three_bounded_classes(self):
        records = source_records()
        self.assertEqual(set(records), {26, 31, 33})
        expected_hodge = {
            26: {"h11": 2, "h21": 120, "chi": -236},
            31: {"h11": 2, "h21": 128, "chi": -252},
            33: {"h11": 2, "h21": 132, "chi": -260},
        }
        for index, record in records.items():
            source = record["source"]
            self.assertEqual(len(source["global_points"]), 8)
            self.assertEqual(source["expected_boundary_point_count"], 7)
            self.assertEqual(source["expected_hodge"], expected_hodge[index])
            self.assertEqual(len(record["selected_frsts"]), 2)
            self.assertTrue(all(
                frst["simplices_index_space"] == "triangulation_local"
                for frst in record["selected_frsts"]
            ))
            self.assertEqual(record["source"]["source_row"], {26: 21, 31: 27, 33: 29}[index])

    def test_missing_replay_inputs_are_terminal_for_each_index(self):
        result = run_bounded_analysis({})
        for index in ("26", "31", "33"):
            report = result["reports"][index]
            self.assertEqual(report["scope_status"], "in_scope")
            statuses = {row["terminal_status"] for row in report["terminal_records"]}
            self.assertIn("source_identity_unavailable", statuses)

    def test_boundary_scope_mismatch_is_terminal_without_selection(self):
        record = source_records()[26]
        record["source"] = dict(record["source"], global_points=record["source"]["global_points"][:-1])
        result = run_bounded_analysis({
            index: record
            for index in (26, 31, 33)
        })
        self.assertIn(
            "source_polytope_id_mismatch",
            {row["terminal_status"] for row in result["reports"]["26"]["terminal_records"]},
        )

    def test_identity_is_order_independent(self):
        first = _FakeTriangulation([[0, 1, 2], [0, 2, 3]])
        second = _FakeTriangulation([[0, 2, 3], [0, 1, 2]])
        self.assertEqual(triangulation_identity(first), triangulation_identity(second))

    def test_valid_certificate_replays_against_immutable_source_and_action(self):
        record, report, action_record, certificate = _certificate_fixture()
        self.assertEqual(certificate["certificate_schema_version"], CERTIFICATE_SCHEMA_VERSION)
        self.assertEqual(certificate["formula_schema_version"], FORMULA_SCHEMA_VERSION)
        checked = validate_replay_certificate(
            certificate,
            report=report,
            frst_hash=action_record["frst_hash"],
            action=action_record["action"],
            action_record=action_record,
        )
        self.assertEqual(checked["status"], "valid")
        self.assertFalse(checked["terminal"])

    def test_certificate_rejects_tampered_digest_source_frst_and_action(self):
        record, report, action_record, certificate = _certificate_fixture()
        tampered_digest = copy.deepcopy(certificate)
        tampered_digest["evidence"]["component_h2_evidence_digest"] = "tampered"
        self.assertEqual(
            validate_replay_certificate(tampered_digest, report=report)["status"],
            "mismatch",
        )
        tampered_source = copy.deepcopy(certificate)
        tampered_source["source"]["source_row"] += 1
        self.assertIn(
            "source source_row mismatch",
            validate_replay_certificate(tampered_source, report=report)["reasons"],
        )
        self.assertIn(
            "FRST hash mismatch",
            validate_replay_certificate(
                certificate,
                frst_hash=record["selected_frst"]["identity"],
            )["reasons"],
        )
        wrong_action = copy.deepcopy(action_record["action"])
        wrong_action["lambda_f"] = 0
        self.assertIn(
            "action digest does not match the live action",
            validate_replay_certificate(certificate, action=wrong_action)["reasons"],
        )

    def test_resealed_certificate_tamper_matrix_fails_closed(self):
        record, report, action_record, certificate = _certificate_fixture()
        tampered_row = copy.deepcopy(certificate)
        tampered_row["source"]["source_row"] = 27
        self.assertTrue(validate_replay_certificate(_reseal_certificate(tampered_row))["terminal"])

        tampered_coordinates = copy.deepcopy(certificate)
        tampered_coordinates["source"]["global_points"][0][0] = 99
        self.assertTrue(validate_replay_certificate(_reseal_certificate(tampered_coordinates))["terminal"])

        tampered_source_sha = copy.deepcopy(certificate)
        tampered_source_sha["source"]["source_sha256"] = "tampered"
        tampered_source_sha["source"]["parquet_sha256"] = "tampered"
        self.assertTrue(validate_replay_certificate(
            _reseal_certificate(tampered_source_sha), report=report
        )["terminal"])

        tampered_index = copy.deepcopy(certificate)
        tampered_index["index"] = 31
        self.assertTrue(validate_replay_certificate(_reseal_certificate(tampered_index))["terminal"])

        tampered_evidence = copy.deepcopy(certificate)
        tampered_evidence["evidence"]["fixed_locus_euler"]["chi_F_I"] += 1
        self.assertTrue(validate_replay_certificate(
            _reseal_certificate(tampered_evidence), action_record=action_record
        )["terminal"])

        tampered_schema = copy.deepcopy(certificate)
        tampered_schema["formula_schema_version"] = "stale"
        self.assertTrue(validate_replay_certificate(tampered_schema)["terminal"])
        stale_replay_schema = copy.deepcopy(certificate)
        stale_replay_schema["replay_schema_version"] = "cyaxiverse-bounded-mpcp-replay-1.2"
        self.assertTrue(validate_replay_certificate(stale_replay_schema)["terminal"])

        missing_source_sha = copy.deepcopy(certificate)
        del missing_source_sha["source"]["source_sha256"]
        self.assertTrue(validate_replay_certificate(missing_source_sha)["terminal"])

    def test_all_terminal_accounting_has_no_certificate_or_implicit_selection(self):
        result = run_bounded_analysis({})
        self.assertEqual(result["counts"]["indices_with_terminal_records"], 3)
        for report in result["reports"].values():
            self.assertEqual(report["replay_certificates"], [])
            self.assertEqual(report["counts"] if "counts" in report else {}, {})
            self.assertTrue(report["terminal_records"])


if __name__ == "__main__":
    unittest.main()
