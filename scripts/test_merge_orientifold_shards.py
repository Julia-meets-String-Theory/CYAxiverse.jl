"""Unit tests for scripts/merge_orientifold_shards.py.

These use hand-built shard summaries so the merge logic is exercised without a
CYTools dependency, following the fixture-level pattern of the other
``test_*`` modules in this directory.
"""

import unittest

from merge_orientifold_shards import ShardMergeError, merge_shard_summaries


def _shard_summary(index, count, total_favorable, *, counts, class_funnel, status_counts):
    return {
        "schema_version": "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.3",
        "kinetic_normalization": "sugra_c1",
        "input": {
            "requested_h11": 2,
            "shard": {
                "index": index,
                "count": count,
                "is_sharded": count > 1,
                "shard_favorable_polytopes": counts["favorable_polytopes"],
                "total_favorable_polytopes": total_favorable,
            },
        },
        "run_provenance": {"source_commit": "deadbeef"},
        "paper_targets": {
            "favorable_polytopes": 4,
            "frst_classes": 4,
            "h11_minus_zero_h21_plus_zero_orientifold_cys": 2,
        },
        "counts": counts,
        "terminal_ledger": {
            "record_count": sum(status_counts.values()),
            "record_kind_counts": {"candidate": sum(status_counts.values())},
            "terminal_status_counts": status_counts,
            "class_funnel": class_funnel,
        },
    }


def _counts(**overrides):
    base = {
        "favorable_polytopes": 2,
        "raw_frsts": 2,
        "frst_classes": 2,
        "raw_trilayer_polytopes": 0,
        "raw_trilayer_frst_classes": 0,
        "nonfrozen_trilayer_frst_classes": 0,
        "h21_plus_zero_trilayer_frst_classes": 1,
        "identity_torus_action_count": 0,
        "identity_torus_action_cy_count": 0,
        "identity_valid_o3o7_action_cy_count": 0,
        "source_evidence_inherited_orientifold_cys": 1,
        "source_evidence_h11_minus_zero_orientifold_cys": 1,
        "source_vertex_evidence_inherited_orientifold_cys": 1,
        "source_vertex_evidence_h11_minus_zero_orientifold_cys": 1,
        "kaehler_point_export_accepted_count": None,
        "kaehler_point_export_rejected_count": None,
    }
    base.update(overrides)
    return base


class MergeShardSummariesTests(unittest.TestCase):
    def _two_disjoint_shards(self):
        shard0 = _shard_summary(
            0, 2, 4,
            counts=_counts(),
            class_funnel=[
                {"polytope_normal_form_id": "nfA", "frst_class_index": 0, "accepted_for_table_1": True},
                {"polytope_normal_form_id": "nfA", "frst_class_index": 1, "accepted_for_table_1": False},
            ],
            status_counts={"accepted_verified_orientifold": 1, "fixed_point_set_non_smooth": 1},
        )
        shard1 = _shard_summary(
            1, 2, 4,
            counts=_counts(),
            class_funnel=[
                {"polytope_normal_form_id": "nfB", "frst_class_index": 0, "accepted_for_table_1": True},
                {"polytope_normal_form_id": "nfC", "frst_class_index": 0, "accepted_for_table_1": False},
            ],
            status_counts={"accepted_verified_orientifold": 1, "fixed_point_set_non_smooth": 1},
        )
        return [shard0, shard1]

    def test_counts_are_additive(self):
        merged = merge_shard_summaries(self._two_disjoint_shards())
        self.assertEqual(merged["counts"]["favorable_polytopes"], 4)
        self.assertEqual(merged["counts"]["frst_classes"], 4)
        self.assertEqual(merged["counts"]["h21_plus_zero_trilayer_frst_classes"], 2)
        self.assertIsNone(merged["counts"]["kaehler_point_export_accepted_count"])

    def test_class_funnel_union_and_certified_count(self):
        merged = merge_shard_summaries(self._two_disjoint_shards())
        self.assertEqual(merged["terminal_ledger"]["class_count"], 4)
        self.assertEqual(merged["table_1_accepted_class_count"], 2)
        self.assertEqual(
            merged["terminal_ledger"]["terminal_status_counts"],
            {"accepted_verified_orientifold": 2, "fixed_point_set_non_smooth": 2},
        )

    def test_population_complete_when_totals_match_target(self):
        merged = merge_shard_summaries(self._two_disjoint_shards())
        self.assertTrue(merged["population_complete"])
        self.assertEqual(merged["claim_status"]["favorable_polytopes"], "exact")

    def test_overlapping_geometry_is_rejected(self):
        shards = self._two_disjoint_shards()
        shards[1]["terminal_ledger"]["class_funnel"][0]["polytope_normal_form_id"] = "nfA"
        shards[1]["terminal_ledger"]["class_funnel"][0]["frst_class_index"] = 0
        with self.assertRaises(ShardMergeError):
            merge_shard_summaries(shards)

    def test_missing_shard_index_is_rejected(self):
        shards = self._two_disjoint_shards()
        shards[1]["input"]["shard"]["index"] = 0  # both claim index 0 of 2
        with self.assertRaises(ShardMergeError):
            merge_shard_summaries(shards)

    def test_inconsistent_source_commit_is_rejected(self):
        shards = self._two_disjoint_shards()
        shards[1]["run_provenance"]["source_commit"] = "cafef00d"
        with self.assertRaises(ShardMergeError):
            merge_shard_summaries(shards)

    def test_merged_summary_carries_the_common_label(self):
        merged = merge_shard_summaries(self._two_disjoint_shards())
        self.assertEqual(merged["schema_version"], "cyaxiverse-orientifold-shard-merge-1.1")
        self.assertEqual(merged["kinetic_normalization"], "sugra_c1")
        self.assertFalse(merged["kinetic_normalization_inferred"])

    def test_legacy_inputs_give_an_inferred_legacy_label(self):
        shards = self._two_disjoint_shards()
        for shard in shards:
            shard["schema_version"] = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.2"
            del shard["kinetic_normalization"]
        merged = merge_shard_summaries(shards)
        self.assertEqual(merged["kinetic_normalization"], "legacy_c_half")
        self.assertTrue(merged["kinetic_normalization_inferred"])

    def test_mixed_labels_are_rejected(self):
        shards = self._two_disjoint_shards()
        shards[1]["schema_version"] = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.2"
        del shards[1]["kinetic_normalization"]
        with self.assertRaisesRegex(ShardMergeError, "sugra_c1.*legacy_c_half"):
            merge_shard_summaries(shards, ["a.json", "b.json"])

    def test_invalid_input_labels_are_rejected(self):
        cases = [
            {"kinetic_normalization": None},
            {"kinetic_normalization": "legacy_c_half"},
            {"kinetic_normalization": "bogus"},
            {"schema_version": "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.2"},
            {"schema_version": "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1"},
            {"schema_version": "cyaxiverse-fuzzy-axions-h11-4-reproduction-9.9"},
        ]
        for edit in cases:
            shards = self._two_disjoint_shards()
            for key, value in edit.items():
                if value is None:
                    del shards[0][key]
                else:
                    shards[0][key] = value
            with self.assertRaisesRegex(ShardMergeError, "a.json"):
                merge_shard_summaries(shards, ["a.json", "b.json"])

    def test_merge_label_round_trip_and_rules(self):
        from kinetic_normalization_labels import shard_merge_label

        merged = merge_shard_summaries(self._two_disjoint_shards())
        self.assertEqual(shard_merge_label(merged, "m.json"), ("sugra_c1", False))
        legacy = {"schema_version": "cyaxiverse-orientifold-shard-merge-1.0"}
        self.assertEqual(shard_merge_label(legacy, "m.json"), ("legacy_c_half", True))
        derived = dict(merged, kinetic_normalization="legacy_c_half",
                       kinetic_normalization_inferred=True)
        self.assertEqual(shard_merge_label(derived, "m.json"), ("legacy_c_half", True))
        rejected = [
            dict(legacy, kinetic_normalization="sugra_c1"),
            {k: v for k, v in merged.items() if k != "kinetic_normalization"},
            {k: v for k, v in merged.items() if k != "kinetic_normalization_inferred"},
            dict(merged, kinetic_normalization="legacy_c_half"),
            dict(merged, kinetic_normalization="sugra_c1", kinetic_normalization_inferred=True),
            dict(merged, kinetic_normalization="a96_reproduction_raw_radian"),
            dict(merged, schema_version="cyaxiverse-orientifold-shard-merge-9.9"),
        ]
        for value in rejected:
            with self.assertRaisesRegex(ValueError, "m.json"):
                shard_merge_label(value, "m.json")

    def test_unsharded_input_is_rejected(self):
        shards = self._two_disjoint_shards()
        shards[0]["input"]["shard"]["is_sharded"] = False
        with self.assertRaises(ShardMergeError):
            merge_shard_summaries(shards)


if __name__ == "__main__":
    unittest.main()
