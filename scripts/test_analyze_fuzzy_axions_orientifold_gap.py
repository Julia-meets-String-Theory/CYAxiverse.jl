"""Tests for the artifact-only Table 1 gap classifier."""

from pathlib import Path
import unittest

import analyze_fuzzy_axions_orientifold_gap as gap


H11_2 = Path("/private/tmp/cyax-orientifold-rerun-h11-2-20260820.json")
H11_3 = Path("/private/tmp/cyax-orientifold-rerun-h11-3-20260820.json")


def _detail(
    polytope_index=0,
    class_count=1,
    accepted=None,
    attempts=None,
    unresolved=None,
):
    return {
        "polytope_index": polytope_index,
        "frst_class_count": class_count,
        "orientifold_action_audit": {
            "h11_minus_zero_classes": list(accepted or []),
            "reason_diagnostics": {
                "surface_attempts": list(attempts or []),
                "unresolved_components": list(unresolved or []),
            },
        },
    }


class GapClassifierUnitTests(unittest.TestCase):
    def test_surface_unavailable_is_annotation_without_candidate_linkage(self):
        detail = _detail(
            attempts=[
                {
                    "frst_class_index": 0,
                    "status": "unavailable",
                    "reason_code": "non_smooth_ambient_cone",
                    "reason": "ambient certificate unavailable",
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )
        self.assertFalse(record["candidate_linked_unavailable"])

    def test_candidate_linked_unavailable_is_class_partition_evidence(self):
        detail = _detail(
            unresolved=[
                {
                    "candidate_id": "candidate-1",
                    "reason_code": "non_smooth_ambient_cone",
                    "candidate_terminal_status": "fixed_point_set_non_smooth",
                    "frst_class_index": 0,
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"], "unaccepted_with_candidate_linked_unavailable"
        )
        self.assertTrue(record["candidate_linked_unavailable"])
        self.assertEqual(record["candidate_linked_unavailable_component_count"], 1)
        self.assertEqual(record["candidate_linked_candidate_ids"], ["candidate-1"])
        self.assertIn("not a singularity determination", record["reason"])
        self.assertIn("paper-error finding", record["reason"])

    def test_partial_candidate_terminal_context_is_annotation_only(self):
        detail = _detail(
            attempts=[
                {
                    "frst_class_index": 0,
                    "status": "certified",
                    "reason_code": None,
                    "candidate_context": {
                        "candidate_terminal_status": "fixed_point_set_non_smooth"
                    },
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )
        self.assertFalse(record["candidate_linked_unavailable"])

    def test_no_diagnostic_is_not_a_rejection(self):
        record = gap._class_record(_detail(), 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )
        self.assertIn("not an exhaustive candidate verdict", record["reason"])

    def test_surface_only_evidence_is_not_promoted_to_a_class_verdict(self):
        detail = _detail(
            attempts=[
                {
                    "frst_class_index": 0,
                    "status": "certified",
                    "reason_code": None,
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )

    def test_category_partition_is_mutually_exclusive_for_fixture_records(self):
        details = [
            _detail(0, accepted=[0]),
            _detail(
                1,
                unresolved=[
                    {
                        "frst_class_index": 0,
                        "candidate_id": "candidate-1",
                        "reason_code": "non_smooth_ambient_cone",
                    }
                ],
            ),
            _detail(
                2,
                attempts=[
                    {
                        "frst_class_index": 0,
                        "status": "certified",
                        "reason_code": None,
                        "candidate_context": {
                            "candidate_terminal_status": "fixed_point_set_non_smooth"
                        },
                    }
                ],
            ),
            _detail(3),
        ]
        records = [
            gap._class_record(detail, 0, accepted=(index == 0))
            for index, detail in enumerate(details)
        ]
        categories = [record["category"] for record in records]

        self.assertEqual(
            categories,
            [
                "certified_inherited",
                "unaccepted_with_candidate_linked_unavailable",
                "unaccepted_not_classified_by_retained_terminal_ledger",
                "unaccepted_not_classified_by_retained_terminal_ledger",
            ],
        )
        self.assertEqual(len(categories), 4)

    def test_h11_four_is_rejected_by_scope(self):
        with self.assertRaises(gap.ArtifactError):
            gap.load_artifact(H11_2, 4)

    def test_aggregate_gap_is_not_treated_as_class_id_mapping(self):
        result = gap._comparison(253, 80)

        self.assertEqual(result["target_gap_count"], 173)
        self.assertEqual(result["code_output"], 80)
        self.assertNotIn("target_gap_class_ids", result)

    def test_inherited_partition_requires_equal_retained_h11_zero_ids(self):
        data = {
            "counts": {
                "source_evidence_inherited_orientifold_cys": 1,
                "source_evidence_h11_minus_zero_orientifold_cys": 0,
            },
            "details": [_detail()],
        }

        with self.assertRaisesRegex(
            gap.ArtifactError,
            "only h11_minus_zero_classes identifiers are retained",
        ):
            gap._class_level_audit(data, 2)


@unittest.skipUnless(H11_2.exists() and H11_3.exists(), "corrected audit artifacts are absent")
class CorrectedArtifactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = gap.analyze_paths({2: H11_2, 3: H11_3})
        cls.by_h11 = {entry["h11"]: entry for entry in cls.result["analyses"]}

    def test_scope_excludes_superseded_h11_four(self):
        self.assertEqual(self.result["scope"]["h11"], [2, 3])
        self.assertEqual(self.result["scope"]["excluded_h11"], [4])

    def test_h11_two_class_level_accounting(self):
        entry = self.by_h11[2]
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "target_gap_count"
            ],
            22,
        )
        self.assertEqual(entry["class_level_audit"]["total_frst_class_count"], 36)
        self.assertEqual(entry["class_level_audit"]["certified_class_count"], 10)
        self.assertEqual(
            entry["fixed_surface_diagnostics"],
            {
                "surface_attempt_count": 1164,
                "certified_surface_count": 762,
                "skipped_surface_count": 402,
                "skip_reason_counts": {"non_smooth_ambient_cone": 402},
                "unresolved_candidate_component_count": 33,
                "interpretation": (
                    "diagnostic rows are evidence attempts, not accepted orientifold "
                    "classes; surface statuses are annotations, and only unresolved "
                    "components linked to a candidate enter the class partition"
                ),
            },
        )
        self.assertEqual(
            entry["class_level_audit"]["category_counts"],
            {
                "unaccepted_not_classified_by_retained_terminal_ledger": 20,
                "unaccepted_with_candidate_linked_unavailable": 6,
            },
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_count"],
            16,
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_deficit"],
            16,
        )
        self.assertEqual(
            entry["orientifold_comparison"][
                "h11_minus_zero_h21_plus_zero_orientifold_cys"
            ]["target_gap_count"],
            0,
        )

    def test_h11_three_class_level_accounting(self):
        entry = self.by_h11[3]
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "target_gap_count"
            ],
            173,
        )
        self.assertEqual(entry["class_level_audit"]["total_frst_class_count"], 274)
        self.assertEqual(entry["class_level_audit"]["certified_class_count"], 80)
        self.assertEqual(
            entry["fixed_surface_diagnostics"]["surface_attempt_count"], 5510
        )
        self.assertEqual(
            entry["fixed_surface_diagnostics"]["skipped_surface_count"], 1892
        )
        self.assertEqual(
            entry["fixed_surface_diagnostics"]["skip_reason_counts"],
            {"non_smooth_ambient_cone": 1892},
        )
        self.assertEqual(
            entry["fixed_surface_diagnostics"][
                "unresolved_candidate_component_count"
            ],
            117,
        )
        self.assertEqual(
            entry["class_level_audit"]["category_counts"],
            {
                "unaccepted_not_classified_by_retained_terminal_ledger": 166,
                "unaccepted_with_candidate_linked_unavailable": 28,
            },
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_count"],
            108,
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_deficit"],
            145,
        )
        self.assertEqual(
            entry["orientifold_comparison"][
                "h11_minus_zero_h21_plus_zero_orientifold_cys"
            ]["target_gap_count"],
            0,
        )

    def test_every_candidate_linked_class_has_explicit_boundary_evidence(self):
        for entry in self.by_h11.values():
            for record in entry["class_level_audit"]["unaccepted_class_records"]:
                if record["category"] == "unaccepted_with_candidate_linked_unavailable":
                    self.assertTrue(record["candidate_linked_unavailable"])
                    self.assertTrue(record["candidate_linked_candidate_ids"])
                    self.assertTrue(record["unresolved_component_count"])


if __name__ == "__main__":
    unittest.main()
