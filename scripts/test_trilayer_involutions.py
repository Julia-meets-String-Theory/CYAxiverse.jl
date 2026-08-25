"""Focused exact reconstruction fixtures for the Sheridan trilayer action."""

from __future__ import annotations

import unittest

import numpy as np

from trilayer_involutions import (
    TERMINAL_STATUSES,
    decode_fraction_vector,
    eq_4_45_parity_evidence,
    enumerate_source_trilayer_candidates,
    evaluate_exact_trilayer_action,
    fan_preservation_evidence,
    frst_identifier,
    polytope_identifier,
    reconstruct_trilayer_actions,
)
from mpcp_bounded_analysis import build_replay_certificate


class _FakeDual:
    def __init__(self, vertices):
        self._vertices = np.asarray(vertices, dtype=np.int64)

    def vertices(self):
        return self._vertices


class _FakePoly:
    def __init__(self, vertices, dual_vertices):
        self._vertices = np.asarray(vertices, dtype=np.int64)
        self._dual = _FakeDual(dual_vertices)

    def vertices(self):
        return self._vertices

    def points(self):
        return self._vertices

    def dual(self):
        return self._dual


class _FakeTriangulation:
    def __init__(self, points, simplices):
        self._points = np.asarray(points, dtype=np.int64)
        self._simplices = np.asarray(simplices, dtype=np.int64)

    def points(self):
        return self._points

    def simplices(self, as_indices=False):
        del as_indices
        return self._simplices


# A small exact lattice fixture.  Its dual vertices are one facet at p0.q=-1
# and one outside vertex at p0.q=+1.  The fixture is used only for the source
# combinatorics; it is not presented as a CY geometry.
P0 = np.array([1, 0, 0, 0], dtype=np.int64)
FACET = [
    [-1, 0, 0, 0],
    [-1, 2, 0, 0],
    [-1, 0, 1, 0],
    [-1, 0, 0, 1],
]
DUAL_VERTICES = FACET + [[1, 0, 0, 0]]
SOURCE_POLY = _FakePoly([P0.tolist(), [0, 1, 0, 0]], DUAL_VERTICES)


class StructuralReconstructionTests(unittest.TestCase):
    def test_enumerates_source_action_without_class_labels_or_witnesses(self):
        records = enumerate_source_trilayer_candidates(SOURCE_POLY)
        accepted = [record for record in records if record["terminal_status"] == "structurally_reconstructed"]
        self.assertEqual(len(accepted), 1)
        record = accepted[0]
        self.assertEqual(record["p0"], P0.tolist())
        self.assertEqual(record["action"]["lattice_matrix"], np.eye(4, dtype=int).tolist())
        self.assertEqual(decode_fraction_vector(record["action"]["torus_shift"]), (1 / 2, 0, 0, 0))
        self.assertEqual(record["action"]["lambda_f"], 1)
        self.assertFalse(record["aggregate_labels_used"])
        self.assertFalse(record["unpublished_witnesses_used"])
        self.assertEqual(record["reconstruction_rule_version"], "moritz-4.64-4.66-source-gauge-1")
        self.assertTrue(record["action_digest"])

    def test_enumerates_all_qualifying_vertices_instead_of_selecting_one(self):
        second = np.array([0, 1, 0, 0], dtype=np.int64)
        dual = [
            [-1, 0, 0, 0],
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [-1, 0, 0, 1],
            [1, 0, 0, 0],
        ]
        # This second primal vertex is not a trilayer pole for the same dual
        # polytope, so the terminal record must retain that rejection.
        records = enumerate_source_trilayer_candidates(_FakePoly([P0.tolist(), second.tolist()], dual))
        self.assertEqual(sum(record["terminal_status"] == "structurally_reconstructed" for record in records), 1)
        self.assertIn(records[-1]["terminal_status"], {"not_trilayer", "dual_vertex_outside_facet_ambiguous"})

    def test_non_trilayer_outside_vertex_is_terminal(self):
        dual = DUAL_VERTICES + [[0, 0, 2, 0]]
        records = enumerate_source_trilayer_candidates(_FakePoly([P0.tolist()], dual))
        self.assertEqual(records[0]["terminal_status"], "dual_vertex_outside_facet_ambiguous")

    def test_non_three_dimensional_dual_face_is_terminal(self):
        dual = [
            [-1, 0, 0, 0],
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [1, 0, 0, 0],
        ]
        records = enumerate_source_trilayer_candidates(
            _FakePoly([P0.tolist()], dual)
        )
        self.assertEqual(records[0]["terminal_status"], "dual_facet_not_three_dimensional")
        self.assertEqual(records[0]["facet_affine_rank"], 2)

    def test_eq_4_45_is_exact_and_rejects_wrong_lambda(self):
        record = next(
            item
            for item in enumerate_source_trilayer_candidates(SOURCE_POLY)
            if item["terminal_status"] == "structurally_reconstructed"
        )
        passed = eq_4_45_parity_evidence(SOURCE_POLY, record["action"])
        self.assertEqual(passed["status"], "eq_4_45_parity_passed")
        wrong = dict(record["action"])
        wrong["lambda_f"] = 0
        failed = eq_4_45_parity_evidence(SOURCE_POLY, wrong)
        self.assertEqual(failed["status"], "eq_4_45_parity_failure")
        self.assertEqual(len(failed["violations"]), 5)

    def test_missing_fan_is_fail_closed(self):
        result = reconstruct_trilayer_actions(SOURCE_POLY)
        self.assertEqual(result["candidate_count"], 2)
        statuses = {record["terminal_status"] for record in result["candidates"]}
        self.assertIn("fan_evidence_unavailable", statuses)
        self.assertIn("not_trilayer", statuses)

    def test_fan_preservation_has_explicit_rejection_categories(self):
        tri = _FakeTriangulation(SOURCE_POLY.points(), [[0, 1]])
        preserved = fan_preservation_evidence(SOURCE_POLY, tri, np.eye(4, dtype=np.int64))
        self.assertEqual(preserved["status"], "fan_preserved")
        nonpreserving = fan_preservation_evidence(
            SOURCE_POLY,
            tri,
            np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int64),
        )
        self.assertEqual(nonpreserving["status"], "fan_not_preserved")

    def test_coordinate_basis_covariance(self):
        shear = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.int64,
        )
        inverse = np.array(
            [[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.int64,
        )
        transformed_primal = (shear @ np.asarray(SOURCE_POLY.points()).T).T
        transformed_dual = (inverse.T @ np.asarray(DUAL_VERTICES).T).T
        transformed = _FakePoly(transformed_primal, transformed_dual)
        original = next(
            item for item in enumerate_source_trilayer_candidates(SOURCE_POLY)
            if item["terminal_status"] == "structurally_reconstructed"
        )
        rotated = next(
            item for item in enumerate_source_trilayer_candidates(transformed)
            if item["terminal_status"] == "structurally_reconstructed"
        )
        self.assertEqual(rotated["p0"], (shear @ P0).tolist())
        original_t = np.asarray(decode_fraction_vector(original["action"]["torus_shift"]), dtype=object)
        rotated_t = np.asarray(decode_fraction_vector(rotated["action"]["torus_shift"]), dtype=object)
        self.assertEqual(tuple(rotated_t), tuple(shear @ original_t))
        self.assertEqual(rotated["action"]["lambda_f"], original["action"]["lambda_f"])
        self.assertEqual(rotated["pairings"], original["pairings"])

    def test_terminal_vocabulary_is_explicit(self):
        self.assertIn("fixed_locus_euler_unavailable", TERMINAL_STATUSES)
        self.assertIn("mpcp_certificate_unavailable", TERMINAL_STATUSES)
        self.assertIn("mpcp_certificate_mismatch", TERMINAL_STATUSES)
        self.assertIn("eq_4_45_parity_failure", TERMINAL_STATUSES)
        self.assertNotIn("accepted_from_population_label", TERMINAL_STATUSES)

    def test_valid_bounded_certificate_is_verified_before_exact_topology(self):
        structural = next(
            item for item in enumerate_source_trilayer_candidates(SOURCE_POLY)
            if item["terminal_status"] == "structurally_reconstructed"
        )
        triangulation = _FakeTriangulation(SOURCE_POLY.points(), [[0, 1]])
        source = {
            "source_sha256": "source-fixture-sha",
            "source_row": 21,
            "polytope_id": polytope_identifier(SOURCE_POLY.points()),
            "global_points": SOURCE_POLY.points(),
        }
        report = {
            "schema_version": "cyaxiverse-bounded-mpcp-replay-1.3",
            "runtime_provenance": {"cytools_version_guard": {
                "status": "verified", "expected": "1.4.12", "observed": "1.4.12",
            }},
            "caps": {"max_triangulations": 1},
            "source_identity": source,
            "selected_frst": {"identity": frst_identifier([[0, 1]])},
        }
        action_record = {
            "terminal_status": "refined_action_evaluated",
            "candidate_index": 0,
            "frst_hash": frst_identifier([[0, 1]]),
            "action": structural["action"],
            "fixed_locus_euler": {"status": "computed", "chi_F_I": 0, "components": []},
            "refined_glsm": {
                "status": "refined_h2_action_verified", "h2_matrix": [[1]],
                "proof": {"exact_residual_zero": True},
            },
            "hodge_split": {
                "h11_plus": 1, "h11_minus": 0, "h21_plus": 0,
                "h21_minus": 0, "chi_fixed_locus": 0, "chi_x": 0,
            },
        }
        certificate = build_replay_certificate(
            26, {"source": source}, report, action_record
        )
        self.assertIsNotNone(certificate)
        evaluated = evaluate_exact_trilayer_action(
            SOURCE_POLY, triangulation, None, structural,
            mpcp_certificate=certificate, source_record={"source": source},
        )
        self.assertEqual(evaluated["mpcp_certificate_verification"]["status"], "valid")
        self.assertEqual(evaluated["terminal_status"], "topology_evidence_unavailable")

    def test_tampered_bounded_certificate_is_terminal_in_exact_bridge(self):
        structural = next(
            item for item in enumerate_source_trilayer_candidates(SOURCE_POLY)
            if item["terminal_status"] == "structurally_reconstructed"
        )
        triangulation = _FakeTriangulation(SOURCE_POLY.points(), [[0, 1]])
        source = {
            "source_sha256": "source-fixture-sha", "source_row": 21,
            "polytope_id": polytope_identifier(SOURCE_POLY.points()),
            "global_points": SOURCE_POLY.points(),
        }
        report = {
            "schema_version": "cyaxiverse-bounded-mpcp-replay-1.3",
            "runtime_provenance": {"cytools_version_guard": {
                "status": "verified", "expected": "1.4.12", "observed": "1.4.12",
            }},
            "source_identity": source,
            "selected_frst": {"identity": frst_identifier([[0, 1]])},
        }
        action_record = {
            "terminal_status": "refined_action_evaluated", "candidate_index": 0,
            "frst_hash": frst_identifier([[0, 1]]), "action": structural["action"],
            "fixed_locus_euler": {"status": "computed", "chi_F_I": 0, "components": []},
            "refined_glsm": {"status": "refined_h2_action_verified", "h2_matrix": [[1]], "proof": {}},
            "hodge_split": {"h11_plus": 1, "h11_minus": 0, "h21_plus": 0, "h21_minus": 0},
        }
        certificate = build_replay_certificate(26, {"source": source}, report, action_record)
        certificate["source"]["source_row"] = 99
        evaluated = evaluate_exact_trilayer_action(
            SOURCE_POLY, triangulation, None, structural,
            mpcp_certificate=certificate, source_record={"source": source},
        )
        self.assertEqual(evaluated["terminal_status"], "mpcp_certificate_mismatch")

    def test_missing_certificate_is_unavailable_for_source_certified_bridge(self):
        structural = next(
            item for item in enumerate_source_trilayer_candidates(SOURCE_POLY)
            if item["terminal_status"] == "structurally_reconstructed"
        )
        triangulation = _FakeTriangulation(SOURCE_POLY.points(), [[0, 1]])
        source = {
            "source_sha256": "source-fixture-sha", "source_row": 21,
            "polytope_id": polytope_identifier(SOURCE_POLY.points()),
            "global_points": SOURCE_POLY.points(),
        }
        evaluated = evaluate_exact_trilayer_action(
            SOURCE_POLY, triangulation, None, structural,
            source_record={"source": source},
        )
        self.assertEqual(evaluated["terminal_status"], "mpcp_certificate_unavailable")


if __name__ == "__main__":
    unittest.main()
