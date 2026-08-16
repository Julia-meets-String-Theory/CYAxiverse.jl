"""Focused tests for the Stage-1 raw-FRST collector diagnostics."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import generate_stage1_raw_frsts as stage1


class FakeKahlerCone:
    def hyperplanes(self):
        return np.eye(491)

    def tip_of_stretched_cone(self, value, **kwargs):
        assert value == 1.0
        return np.ones(491)


class FakeEffectiveCone:
    def rays(self):
        return np.eye(491)


class FakePolytope:
    def glsm_charge_matrix(self, include_origin=False):
        assert include_origin is False
        return np.eye(491)


class FakeCalabiYau:
    def __init__(self, divisor_volumes):
        self.divisor_volumes = np.asarray(divisor_volumes, dtype=float)

    def h11(self):
        return 491

    def toric_kahler_cone(self):
        return FakeKahlerCone()

    def toric_effective_cone(self):
        return FakeEffectiveCone()

    def polytope(self):
        return FakePolytope()

    def divisor_basis(self, as_matrix=False):
        assert as_matrix is True
        # Deliberately differ from the GLSM matrix: this is only a basis
        # selector and must not be used as the prime charge matrix.
        return 2.0 * np.eye(491)

    def prime_toric_divisors(self):
        return np.arange(491)

    def intersection_numbers(self, **kwargs):
        raise AssertionError(
            "high-h11 intersection reconstruction must not be a preflight gate"
        )

    def compute_cy_volume(self, point):
        del point
        return 1.0

    def compute_curve_volumes(self, point):
        del point
        return np.ones(491)

    def compute_divisor_volumes(self, point, in_basis=False):
        del point, in_basis
        return self.divisor_volumes

    def compute_inverse_kahler_metric(self, point):
        del point
        return np.eye(491)


class Stage1RawFRSTTests(unittest.TestCase):
    def test_preflight_summary_is_h491_only(self):
        records = [
            {"h11": 50, "terminal_status": "retained_raw_frst"},
            {
                "h11": 491,
                "terminal_status": "retained_raw_frst",
                "canonical_tip_preflight": {
                    "classification": "canonical_tip_divisor_volume_shortfall"
                },
            },
        ]
        self.assertEqual(
            stage1.summarize_nested_preflight_classifications(records),
            {"491": {"canonical_tip_divisor_volume_shortfall": 1}},
        )

    def test_negative_canonical_tip_divisors_are_classified_as_annotation(self):
        divisor_volumes = np.ones(491)
        divisor_volumes[0] = -2.0
        calabi_yau = FakeCalabiYau(divisor_volumes)
        with mock.patch.object(
            stage1.generator,
            "configure_mosek_license",
            return_value={"configured": False, "activated": False},
        ):
            result = stage1.run_h491_canonical_tip_preflight(calabi_yau)

        self.assertEqual(result["status"], "annotated_divisor_volume_shortfall")
        self.assertEqual(
            result["classification"], "canonical_tip_divisor_volume_shortfall"
        )
        self.assertEqual(result["shortfall_kind"], "negative_divisor_volume")
        self.assertEqual(result["diagnostic"]["point_status"], "failed")
        self.assertFalse(
            result["diagnostic"]["checks"]["positive_prime_divisor_volumes"]
        )
        self.assertTrue(
            result["residual_checks"]["prime_from_glsm_charge_matrix"]["passed"]
        )
        self.assertEqual(
            result["basis_selection_matrix"]["role"],
            "CYTools divisor_basis(as_matrix=True) basis selector; not a prime charge matrix",
        )
        self.assertFalse(result["intersection_reconstruction"]["authoritative"])
        self.assertEqual(result["intersection_reconstruction"]["status"], "omitted")

    def test_valid_h491_raw_frst_is_retained_with_preflight_annotation(self):
        class FakeTriangulation:
            labels = np.arange(5)

            def get_cy(self):
                return object()

            def points(self):
                return np.zeros((5, 4), dtype=int)

            def simplices(self, as_indices=False):
                del as_indices
                return np.asarray([[0, 1, 2, 3, 4]], dtype=int)

        preflight = {
            "schema_version": stage1.H491_CANONICAL_TIP_PREFLIGHT_SCHEMA_VERSION,
            "h11": 491,
            "status": "annotated_divisor_volume_shortfall",
            "classification": "canonical_tip_divisor_volume_shortfall",
            "shortfall_kind": "negative_divisor_volume",
            "divisor_volume_lower_bounds_enforced": False,
        }

        with tempfile.TemporaryDirectory(prefix="cyax-stage1-preflight-") as tmpdir:
            arguments = SimpleNamespace(
                outdir=tmpdir,
                sampling_scheme_by_h11={"491": "ntfe_fast"},
                proposal_budget_by_h11={"491": 1},
                retry_budget=0,
                backend="cgal",
                seed=17,
                fine_tune_steps=8,
                walk_step_size=1e-2,
                max_steps_to_wall=25,
                fast_height_scale=0.2,
                ntfe_max_face_points=17,
                ntfe_face_pool_size=5,
            )
            fake_polytope = SimpleNamespace(
                points=lambda: np.zeros((5, 4), dtype=int),
                vertices=lambda: np.zeros((5, 4), dtype=int),
            )
            fake_triangulation = FakeTriangulation()

            def fake_writer(path, **kwargs):
                del path
                return dict(kwargs["metadata"])

            with (
                mock.patch.object(
                    stage1.generator,
                    "triangulation_candidates",
                    return_value=iter([fake_triangulation]),
                ),
                mock.patch.object(stage1.generator, "validate_frst"),
                mock.patch.object(
                    stage1.generator,
                    "polytope_identity",
                    return_value=("polytope-id", [[0, 0, 0, 0]]),
                ),
                mock.patch.object(
                    stage1.generator,
                    "extract_topology",
                    return_value={"h11": 491, "h21": 11},
                ),
                mock.patch.object(stage1, "run_h491_canonical_tip_preflight", return_value=preflight),
                mock.patch.object(
                    stage1, "write_raw_frst_artifact", side_effect=fake_writer
                ),
            ):
                records = stage1.collect_raw_frsts_for_polytope(
                    arguments,
                    491,
                    1,
                    fake_polytope,
                    {"source_kind": "test"},
                    1,
                    {},
                )

        self.assertEqual(records[0]["terminal_status"], "retained_raw_frst")
        self.assertEqual(
            records[0]["canonical_tip_preflight"]["classification"],
            "canonical_tip_divisor_volume_shortfall",
        )
        self.assertEqual(
            records[0]["canonical_tip_preflight"]["shortfall_kind"],
            "negative_divisor_volume",
        )


if __name__ == "__main__":
    unittest.main()
