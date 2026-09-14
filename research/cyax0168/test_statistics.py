#!/usr/bin/env python3
"""Tiny known-truth tests; no calibration or decision fixture access."""

from __future__ import annotations

import unittest
from decimal import Decimal

try:
    from . import statistics as st
except ImportError:  # pragma: no cover - direct unittest discovery
    import statistics as st


class StatisticsTests(unittest.TestCase):
    def test_nearest_rank_and_type7(self):
        self.assertEqual(st.nearest_rank_percentile([1, 2, 3, 4], 0.95), 4)
        self.assertEqual(st._quantile_type7([0.0, 10.0], 0.25), 2.5)

    def test_draws_are_deterministic(self):
        self.assertEqual(st.draw_index(7, ["tiny", 1]), st.draw_index(7, ["tiny", 1]))
        self.assertEqual(st.draw_uniform(["tiny", 2]), st.draw_uniform(["tiny", 2]))

    def test_transform_targets_actual_classifier_estimands(self):
        ia, ib = Decimal("2"), Decimal("4")
        mean_a, mean_b = Decimal("2"), Decimal("4")
        a, b = st.transform_speedup_only(ia, ib, Decimal("3"))
        self.assertEqual(a, Decimal("1"))
        self.assertEqual(a * ia and b * ib / (a * ia), Decimal("3"))
        a, b = st.transform_saving_only(mean_a, mean_b, Decimal("5"))
        self.assertEqual(a, Decimal("1"))
        self.assertEqual(b * mean_b - mean_a, Decimal("5"))

    def test_truth_and_paired_estimands(self):
        self.assertEqual(st.empirical_truth([Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")]), Decimal("4"))
        instances = [([1, 2, 3, 4], [2, 4, 6, 8]), ([2, 3, 4, 5], [4, 6, 8, 10])]
        truth = st.paired_truth_statistics(instances)
        self.assertEqual(truth["speedup_b_over_a"], Decimal("2"))
        self.assertEqual(truth["saving_b_over_a"], Decimal("4.5"))

    def test_warm_blocks_remain_ordered_and_duration_projection_is_exact(self):
        blocks = [({"a": [1, 2, 3], "b": [2, 4, 6]}), ({"a": [2, 3, 4], "b": [4, 6, 8]})]
        self.assertEqual(st.warm_population_p95(blocks, "a"), Decimal("4"))
        category = st.duration_category("query_measured", "S", "C2", "P-low", "warm-process", "Q01")
        self.assertEqual(st.project_campaign_seconds({category: 4}, {category: 10}), 50.0)

    def test_bca_is_deterministic_on_tiny_units(self):
        units = (1.0, 2.0, 3.0, 4.0, 5.0)
        first = st.paired_bca_interval(units, lambda sample: sum(sample) / len(sample), resamples=100, seed_parts_prefix=("tiny",))
        second = st.paired_bca_interval(units, lambda sample: sum(sample) / len(sample), resamples=100, seed_parts_prefix=("tiny",))
        self.assertEqual(first, second)
        self.assertFalse(first["inconclusive"])


if __name__ == "__main__":
    unittest.main()
