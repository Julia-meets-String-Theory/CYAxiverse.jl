#!/usr/bin/env python3
from __future__ import annotations

import unittest
from decimal import Decimal

try:
    from .classifier import (
        ARCHITECTURE_PROBLEM, GRAPH, HYBRID, INCONCLUSIVE, OPERATIONAL_FAILURE,
        RETAIN_S, classify, direct_observation_margin, project_monotone_upper_envelope,
        reachable_outcomes, t4_condition4,
    )
except ImportError:  # pragma: no cover
    from classifier import (
        ARCHITECTURE_PROBLEM, GRAPH, HYBRID, INCONCLUSIVE, OPERATIONAL_FAILURE,
        RETAIN_S, classify, direct_observation_margin, project_monotone_upper_envelope,
        reachable_outcomes, t4_condition4,
    )


def cell(family, profile, mode, *, graph=True, relational=False):
    if relational:
        return {"family": family, "profile": profile, "mode": mode, "t3": {"s_over_g": 2.0, "lower95_s_over_g": 1.5, "saving_s_over_g": 10.0}, "t2": {"s_over_g": 1.1, "lower95_s_over_g": 1.0}}
    return {"family": family, "profile": profile, "mode": mode, "t3": {"speedup": 2.0, "lower95_speedup": 1.5, "saving": 10.0}, "t2": {"speedup": 1.1, "lower95_speedup": 1.0}}


class ClassifierTests(unittest.TestCase):
    def test_margin_equality_and_projection_monotonicity(self):
        self.assertTrue(direct_observation_margin({"speedup": 1.5, "saving": 7.5}, thresholds={"speedup": 2.0, "saving": 10.0}))
        self.assertEqual(project_monotone_upper_envelope({1: 10, 2: 5, 3: 20}), Decimal("1025.000000"))

    def test_reachable_ci_assignments_are_joint_and_monotone(self):
        intervals = [{"name": "x", "lower": 0.9, "upper": 1.1, "thresholds": (1.0,)}]
        outcomes = reachable_outcomes(intervals, lambda assignment: "pass" if assignment["x"] == "pass" else "fail")
        self.assertEqual(outcomes, frozenset({"pass", "fail"}))

    def test_classifier_precedence(self):
        self.assertEqual(classify({"host_valid": False, "architecture_problem": True}), INCONCLUSIVE)
        self.assertEqual(classify({"architecture_problem": True}), ARCHITECTURE_PROBLEM)
        self.assertEqual(classify({"combined_cache_capacity_breach": True}), OPERATIONAL_FAILURE)
        self.assertEqual(classify({"s_passes": True, "g_relative_breach": True}), RETAIN_S)

    def test_graph_and_hybrid_masks(self):
        graph_cells = [cell(family, profile, mode) for family in ("Q03", "Q06") for profile in ("P-low", "P-medium") for mode in ("fresh-process", "warm-process")]
        self.assertEqual(classify({"cells": graph_cells}), GRAPH)
        hybrid_cells = graph_cells + [cell("Q01", profile, mode, relational=True) for profile in ("P-low", "P-medium") for mode in ("fresh-process", "warm-process")]
        self.assertEqual(classify({"cells": hybrid_cells}), HYBRID)

    def test_t4_headroom_rows(self):
        self.assertTrue(t4_condition4({"rss": (75, 100)})["pass"])
        self.assertFalse(t4_condition4({"rss": (76, 100)})["pass"])


if __name__ == "__main__":
    unittest.main()
