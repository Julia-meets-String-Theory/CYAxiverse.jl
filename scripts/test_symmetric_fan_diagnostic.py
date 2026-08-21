"""Fixture tests for the bounded symmetric-fan diagnostic."""

import unittest

import numpy as np

from symmetric_fan_diagnostic import (
    TERMINAL_STATUSES,
    _cell_complex_id,
    _is_frst_preserved,
    classify_symmetric_subdivision_checks,
)


class _FakeTriangulation:
    def __init__(self, points, simplices):
        self._points = np.asarray(points, dtype=int)
        self._simplices = np.asarray(simplices, dtype=int)

    def points(self):
        return self._points

    def simplices(self, as_indices=False):
        del as_indices
        return self._simplices


POINTS = np.asarray(
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=int,
)


class SymmetricFanTerminalStatusTests(unittest.TestCase):
    def test_terminal_vocabulary_is_closed(self):
        self.assertEqual(
            set(TERMINAL_STATUSES),
            {
                "constructed",
                "already_represented",
                "two_face_failed",
                "regularity_failed",
                "star_failed",
                "fineness_failed",
                "resource_limited",
                "explicitly_unavailable",
            },
        )

    def test_successful_symmetric_subdivision_is_constructed(self):
        result = classify_symmetric_subdivision_checks(
            {
                "invariant": True,
                "fine": True,
                "regular": True,
                "star": True,
                "two_face": True,
            }
        )
        self.assertEqual(result["terminal_status"], "constructed")

    def test_simplicial_output_equivalent_to_known_class_is_separate(self):
        result = classify_symmetric_subdivision_checks(
            {
                "invariant": True,
                "fine": True,
                "regular": True,
                "star": True,
                "two_face": True,
                "class_equivalent": True,
            }
        )
        self.assertEqual(result["terminal_status"], "already_represented")

    def test_each_terminal_failure_family_is_retained(self):
        cases = {
            "two_face_failed": {"two_face": False},
            "regularity_failed": {"regular": False},
            "star_failed": {"star": False},
            "fineness_failed": {"fine": False},
        }
        for expected, failed_check in cases.items():
            with self.subTest(expected=expected):
                checks = {
                    "invariant": True,
                    "fine": True,
                    "regular": True,
                    "star": True,
                    "two_face": True,
                }
                checks.update(failed_check)
                self.assertEqual(
                    classify_symmetric_subdivision_checks(checks)["terminal_status"],
                    expected,
                )

    def test_budget_and_unavailable_stops_are_explicit(self):
        self.assertEqual(
            classify_symmetric_subdivision_checks({"resource_limited": True})[
                "terminal_status"
            ],
            "resource_limited",
        )
        self.assertEqual(
            classify_symmetric_subdivision_checks(
                {"unavailable_reason": "PPL unavailable"}
            )["terminal_status"],
            "explicitly_unavailable",
        )
        self.assertEqual(
            classify_symmetric_subdivision_checks({"invariant": False})[
                "terminal_status"
            ],
            "explicitly_unavailable",
        )


class SymmetricFanIdentityTests(unittest.TestCase):
    def test_supplied_frst_and_induced_complex_ids_are_deterministic(self):
        triangulation = _FakeTriangulation(POINTS, [[0, 1, 2, 3], [0, 1, 3, 4]])
        swap = np.eye(4, dtype=int)
        swap[[0, 1]] = swap[[1, 0]]
        self.assertFalse(_is_frst_preserved(triangulation, POINTS, swap))
        first = _cell_complex_id(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            ]
        )
        second = _cell_complex_id(
            [
                [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
            ]
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
