"""Verify the independent raw-FRST to stage-2 input boundary."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

import generate_stage2_eft_reference as stage2_entrypoint
from generate_stage2_eft_reference import main as stage2_main
from glimmers_raw_frst import (
    RawFRSTError,
    build_input_ledger,
    compute_triangulation_hash,
    read_raw_frst_artifact,
    write_raw_frst_artifact,
)
from glimmers_schema11 import atomic_jsonl_dump


class Stage2BoundaryTests(unittest.TestCase):
    def test_historical_sparse_coo_backend_contracts_stored_h491_tensor(self):
        generator = stage2_entrypoint.generator
        point = np.ones(491, dtype=float)
        kappa = np.asarray(
            [[0, 0, 0, 6.0], [1, 1, 1, 6.0]],
            dtype=float,
        )
        geometry = generator._compute_volume_geometry(
            None,
            point,
            volume_backend=generator.HISTORICAL_VOLUME_BACKEND,
            kappa=kappa,
            glsm_charge_matrix=np.eye(491),
            mori_cone=np.eye(491),
        )
        self.assertAlmostEqual(geometry["cy_volume"], 2.0)
        self.assertAlmostEqual(geometry["basis_divisor_volumes"][0], 3.0)
        self.assertAlmostEqual(geometry["basis_divisor_volumes"][1], 3.0)
        self.assertEqual(geometry["basis_divisor_volumes"].shape, (491,))
        self.assertTrue(np.allclose(geometry["prime_divisor_volumes"], geometry["basis_divisor_volumes"]))
        self.assertTrue(np.allclose(geometry["curve_volumes"], point))

    def test_fan_integer_constrained_backend_rounds_ambient_kappa_before_basis_reduction(self):
        generator = stage2_entrypoint.generator
        point = np.ones(2, dtype=float)
        cy = MagicMock()
        # Pre-seed `_fan` so the lazy-init branch (mirroring CYTools' own
        # `if not hasattr(self, "_fan")` pattern) is not exercised here.
        cy._fan = MagicMock()
        # Same numbers as the historical_sparse_coo test above (kappa[0,0,0]
        # = kappa[1,1,1] = 6, all else 0), expressed as Fan's ambient,
        # not-yet-basis-reduced, not-yet-rounded intersection numbers, with a
        # tiny float perturbation that must vanish after the 5e-2-tolerance
        # integer snap.
        cy._fan.intersection_numbers.return_value = {
            (0, 0, 0): 6.0 + 1e-9,
            (1, 1, 1): 6.0 - 1e-9,
        }
        cy.ambient_variety.return_value = MagicMock(
            canonical_divisor_is_smooth=MagicMock(return_value=True)
        )
        cy.divisor_basis.return_value = np.array([0, 1])
        cy.compute_curve_volumes.return_value = point.copy()

        geometry = generator._compute_volume_geometry(
            cy,
            point,
            volume_backend=generator.FAN_INTEGER_CONSTRAINED_VOLUME_BACKEND,
            glsm_charge_matrix=np.eye(2),
        )
        self.assertAlmostEqual(geometry["cy_volume"], 2.0)
        self.assertAlmostEqual(geometry["basis_divisor_volumes"][0], 3.0)
        self.assertAlmostEqual(geometry["basis_divisor_volumes"][1], 3.0)
        self.assertTrue(np.allclose(geometry["prime_divisor_volumes"], geometry["basis_divisor_volumes"]))
        self.assertTrue(np.allclose(geometry["curve_volumes"], point))
        # The un-rounded ambient values (6 +/- 1e-9) must have been snapped to
        # exact integers before contraction, not merely passed through.
        cy._fan.intersection_numbers.assert_called_once_with(
            pushed_down=True, in_basis=False, symmetrize=False, as_np_array=False, copy=True,
        )

    def test_fan_integer_constrained_backend_rejects_deviation_beyond_tolerance(self):
        generator = stage2_entrypoint.generator
        point = np.ones(2, dtype=float)
        cy = MagicMock()
        cy._fan = MagicMock()
        # A deviation from the nearest integer larger than the 5e-2 tolerance
        # must be treated as a real numerical failure, not silently rounded.
        cy._fan.intersection_numbers.return_value = {(0, 0, 0): 6.2, (1, 1, 1): 6.0}
        cy.ambient_variety.return_value = MagicMock(
            canonical_divisor_is_smooth=MagicMock(return_value=True)
        )
        cy.divisor_basis.return_value = np.array([0, 1])
        with self.assertRaises(ValueError):
            generator._compute_volume_geometry(
                cy,
                point,
                volume_backend=generator.FAN_INTEGER_CONSTRAINED_VOLUME_BACKEND,
                glsm_charge_matrix=np.eye(2),
            )

    def test_stage12_freezes_stage1_population_and_preserves_raw_identity_digest(self):
        ledger = [
            {
                "stage2_input_status": "retained_raw_frst",
                "h11": 50,
                "polytope_id": "polytope-a",
                "polytope_index": 1,
                "candidate_index": 2,
                "geometry_id": "geometry-a",
                "full_triangulation_hash": "triangulation-a",
                "raw_frst_path": "/stage1/frst-a.h5",
            },
            {
                "stage2_input_status": "missing_raw_frst",
                "geometry_id": "geometry-missing",
            },
        ]
        stage1_manifest = {"status": "completed", "accepted_geometry_count": 1400}
        provenance = stage2_entrypoint.build_frozen_stage1_population_provenance(
            ledger, stage1_manifest
        )
        reordered = stage2_entrypoint.build_frozen_stage1_population_provenance(
            list(reversed(ledger)), stage1_manifest
        )
        self.assertEqual(provenance["population_target"], 1400)
        self.assertEqual(provenance["retained_raw_input_count"], 1)
        self.assertTrue(provenance["population_frozen"])
        self.assertFalse(provenance["replenishment_allowed"])
        self.assertTrue(provenance["raw_frst_provenance_preserved"])
        self.assertEqual(
            provenance["retained_raw_identity_digest"],
            reordered["retained_raw_identity_digest"],
        )

    def test_stage10_overwrite_flag_is_explicit_and_safe_by_default(self):
        default_arguments = stage2_entrypoint.build_parser().parse_args(
            ["--stage1-root", "stage1", "--outdir", "stage2"]
        )
        self.assertFalse(default_arguments.allow_overwrite_existing_geometry)
        explicit_arguments = stage2_entrypoint.build_parser().parse_args(
            [
                "--stage1-root",
                "stage1",
                "--outdir",
                "stage2",
                "--allow-overwrite-existing-geometry",
            ]
        )
        self.assertTrue(explicit_arguments.allow_overwrite_existing_geometry)

    def test_stage10_atomic_geometry_write_audit_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="cyax-stage10-write-") as root:
            path = os.path.join(root, "cyax.h5")
            with h5py.File(path, "w") as file:
                file.attrs["schema_version"] = "old-schema"
                file.attrs["construction_metadata_json"] = json.dumps(
                    {"cy3_fingerprint": "old-geometry"}
                )
            old_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()

            with self.assertRaises(FileExistsError):
                stage2_entrypoint.generator.prepare_geometry_artifact_write(path)

            temporary_path, audit = (
                stage2_entrypoint.generator.prepare_geometry_artifact_write(
                    path, allow_overwrite_existing_geometry=True
                )
            )
            Path(temporary_path).write_bytes(b"complete-new-artifact")
            stage2_entrypoint.generator.finalize_geometry_artifact_write(
                temporary_path,
                path,
                allow_overwrite_existing_geometry=True,
            )
            self.assertEqual(Path(path).read_bytes(), b"complete-new-artifact")
            self.assertFalse(os.path.exists(temporary_path))
            self.assertTrue(audit["overwrite_performed"])
            self.assertEqual(audit["event"], "replaced_existing_geometry")
            self.assertEqual(audit["prior_artifact"]["sha256"], old_hash)
            self.assertEqual(
                audit["prior_artifact"]["geometry_id"], "old-geometry"
            )

            failed_temporary = os.path.join(root, "cyax.h5.tmp-failed")
            Path(failed_temporary).write_bytes(b"readable-but-incomplete")
            try:
                raise RuntimeError("simulated HDF5 write failure")
            except RuntimeError:
                stage2_entrypoint.generator.cleanup_temporary_geometry_artifact(
                    failed_temporary
                )
            self.assertFalse(os.path.exists(failed_temporary))

    def test_stage10_geometry_only_eft_and_pool_pending_statuses(self):
        generator = stage2_entrypoint.generator
        self.assertEqual(
            generator.geometry_artifact_status(False),
            generator.GEOMETRY_ONLY_ARTIFACT_STATUS,
        )
        self.assertEqual(
            generator.geometry_artifact_status(True),
            generator.POOL_PENDING_ARTIFACT_STATUS,
        )
        self.assertEqual(
            generator.geometry_artifact_status(
                True, "complete_eligible_ordered_pool"
            ),
            generator.ACCEPTED_GEOMETRY_ARTIFACT_STATUS,
        )

    def test_stage2_output_path_uses_unique_frst_index_before_candidate_index(self):
        first = {
            "h11": 491,
            "polytope_index": 1,
            "candidate_index": 1,
            "frst_index": 1,
        }
        later = {
            **first,
            "frst_index": 88,
        }

        first_path = stage2_entrypoint.build_geometry_output_path("stage2", first)
        later_path = stage2_entrypoint.build_geometry_output_path("stage2", later)

        self.assertNotEqual(first_path, later_path)
        self.assertTrue(str(first_path).endswith("h11_491/np_0000001/cy_0000001/cyax.h5"))
        self.assertTrue(str(later_path).endswith("h11_491/np_0000001/cy_0000088/cyax.h5"))

    def test_stage2_output_path_falls_back_to_candidate_index(self):
        record = {
            "h11": 50,
            "polytope_index": 2,
            "candidate_index": 7,
        }

        path = stage2_entrypoint.build_geometry_output_path("stage2", record)

        self.assertTrue(str(path).endswith("h11_050/np_0000002/cy_0000007/cyax.h5"))

    def test_stage5_defaults_to_canonical_qcd_and_records_diagnostic_budget(self):
        arguments = stage2_entrypoint.build_parser().parse_args(
            ["--stage1-root", "stage1", "--outdir", "stage2"]
        )
        self.assertEqual(arguments.moduli_policy, "canonical_qcd")
        self.assertEqual(arguments.max_kaehler_attempts, 100)
        self.assertFalse(arguments.allow_m_below_one)
        opt_in_arguments = stage2_entrypoint.build_parser().parse_args(
            [
                "--stage1-root",
                "stage1",
                "--outdir",
                "stage2",
                "--allow-m-below-one",
            ]
        )
        self.assertTrue(opt_in_arguments.allow_m_below_one)

    def test_volume_backend_defaults_to_fan_and_historical_mode_is_explicit(self):
        default_arguments = stage2_entrypoint.build_parser().parse_args(
            ["--stage1-root", "stage1", "--outdir", "stage2"]
        )
        self.assertEqual(
            default_arguments.volume_backend,
            stage2_entrypoint.VOLUME_BACKEND_FAN,
        )
        historical_arguments = stage2_entrypoint.build_parser().parse_args(
            [
                "--stage1-root",
                "stage1",
                "--outdir",
                "stage2",
                "--volume-backend",
                stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
            ]
        )
        self.assertEqual(
            historical_arguments.volume_backend,
            stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
        )
        auto_arguments = stage2_entrypoint.build_parser().parse_args(
            [
                "--stage1-root",
                "stage1",
                "--outdir",
                "stage2",
                "--volume-backend",
                stage2_entrypoint.VOLUME_BACKEND_AUTO,
            ]
        )
        self.assertEqual(
            auto_arguments.volume_backend,
            stage2_entrypoint.VOLUME_BACKEND_AUTO,
        )

    def test_auto_volume_backend_selects_historical_only_at_h11_491(self):
        self.assertEqual(
            stage2_entrypoint.resolve_volume_backend_for_h11(
                stage2_entrypoint.VOLUME_BACKEND_AUTO, 50
            ),
            stage2_entrypoint.VOLUME_BACKEND_FAN,
        )
        self.assertEqual(
            stage2_entrypoint.resolve_volume_backend_for_h11(
                stage2_entrypoint.VOLUME_BACKEND_AUTO, 491
            ),
            stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
        )
        self.assertEqual(
            stage2_entrypoint.generator.resolve_volume_backend(
                200, stage2_entrypoint.VOLUME_BACKEND_AUTO
            ),
            stage2_entrypoint.VOLUME_BACKEND_FAN,
        )

    def test_fan_integer_constrained_backend_has_no_h11_restriction_and_is_never_auto_selected(self):
        for h11 in (50, 100, 200, 491):
            self.assertEqual(
                stage2_entrypoint.validate_volume_backend_for_h11(
                    stage2_entrypoint.VOLUME_BACKEND_FAN_INTEGER_CONSTRAINED, h11
                ),
                stage2_entrypoint.VOLUME_BACKEND_FAN_INTEGER_CONSTRAINED,
            )
        # 'auto' policy is unchanged by adding this backend: still fan below
        # h11=491 and historical_sparse_coo at h11=491, never the new route.
        self.assertEqual(
            stage2_entrypoint.resolve_volume_backend_for_h11(
                stage2_entrypoint.VOLUME_BACKEND_AUTO, 50
            ),
            stage2_entrypoint.VOLUME_BACKEND_FAN,
        )
        self.assertEqual(
            stage2_entrypoint.resolve_volume_backend_for_h11(
                stage2_entrypoint.VOLUME_BACKEND_AUTO, 491
            ),
            stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
        )

    def test_historical_volume_backend_accepts_only_h11_491(self):
        self.assertEqual(
            stage2_entrypoint.validate_volume_backend_for_h11(
                stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
                491,
            ),
            stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
        )
        with self.assertRaises(stage2_entrypoint.VolumeBackendConfigurationError):
            stage2_entrypoint.validate_volume_backend_for_h11(
                stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
                490,
            )

    def test_volume_backend_propagates_to_geometry_generator_and_audits(self):
        arguments = stage2_entrypoint.build_parser().parse_args(
            [
                "--stage1-root",
                "stage1",
                "--outdir",
                "stage2",
                "--volume-backend",
                stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
                "--visible-sector-policy",
                "none",
            ]
        )
        raw_record = {
            "h11": 491,
            "polytope_id": "polytope-491",
            "geometry_id": "geometry-491",
            "polytope_index": 1,
            "candidate_index": 1,
            "full_triangulation_hash": "triangulation-491",
            "raw_frst_path": "/stage1/frst-491.h5",
        }
        persisted = {
            **raw_record,
            "arrays": {
                "polytope_points": np.zeros((2, 4), dtype=int),
                "simplices": np.zeros((1, 5), dtype=int),
            },
            "raw_frst_path": raw_record["raw_frst_path"],
        }
        fake_triangulation = MagicMock()
        fake_triangulation.get_cy.return_value = MagicMock()
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-volume-backend-") as temporary:
            with patch.object(
                stage2_entrypoint,
                "reconstruct_raw_frst",
                return_value=(persisted, MagicMock(), fake_triangulation),
            ), patch.object(
                stage2_entrypoint.generator,
                "inspect_geometry_artifact",
                return_value={"exists": False},
            ), patch.object(
                stage2_entrypoint.generator,
                "generate_and_save_geometry",
            ) as generate_geometry:
                terminal_record, topology_audit = (
                    stage2_entrypoint.process_raw_frst_artifact(
                        arguments,
                        raw_record,
                        {},
                        temporary,
                    )
                )

        self.assertTrue(generate_geometry.called)
        self.assertEqual(
            generate_geometry.call_args.kwargs["volume_backend"],
            stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
        )
        self.assertEqual(terminal_record["volume_backend_status"], "accepted")
        self.assertEqual(
            terminal_record["volume_backend_selected"],
            stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
        )
        self.assertEqual(topology_audit["volume_backend"], terminal_record["volume_backend"])
        self.assertEqual(
            topology_audit["volume_backend_selected"],
            terminal_record["volume_backend_selected"],
        )
        self.assertEqual(topology_audit["volume_backend_status"], "accepted")

    def test_non491_historical_backend_is_terminal_rejection(self):
        arguments = stage2_entrypoint.build_parser().parse_args(
            [
                "--stage1-root",
                "stage1",
                "--outdir",
                "stage2",
                "--volume-backend",
                stage2_entrypoint.VOLUME_BACKEND_HISTORICAL_H11_491,
                "--visible-sector-policy",
                "none",
            ]
        )
        raw_record = {
            "h11": 490,
            "polytope_id": "polytope-490",
            "geometry_id": "geometry-490",
            "polytope_index": 1,
            "candidate_index": 1,
            "full_triangulation_hash": "triangulation-490",
            "raw_frst_path": "/stage1/frst-490.h5",
        }
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-volume-rejection-") as temporary:
            with patch.object(
                stage2_entrypoint,
                "reconstruct_raw_frst",
            ) as reconstruct, patch.object(
                stage2_entrypoint.generator,
                "inspect_geometry_artifact",
                return_value={"exists": False},
            ):
                terminal_record, topology_audit = (
                    stage2_entrypoint.process_raw_frst_artifact(
                        arguments,
                        raw_record,
                        {},
                        temporary,
                    )
                )

        reconstruct.assert_not_called()
        self.assertEqual(
            terminal_record["terminal_status"], "volume_backend_rejection"
        )
        self.assertEqual(terminal_record["volume_backend_status"], "rejected")
        self.assertEqual(topology_audit["stage2_terminal_status"], "volume_backend_rejection")
        self.assertEqual(topology_audit["volume_backend_status"], "rejected")
        self.assertIn("h11=491", terminal_record["terminal_reason"])

    def test_kaehler_point_validation_records_domain_checks(self):
        class FakeKaehlerCone:
            def hyperplanes(self):
                return np.asarray([[1.0]])

        class FakeCalabiYau:
            def compute_cy_volume(self, point):
                return float(point[0] ** 3)

            def compute_curve_volumes(self, point):
                return np.asarray([point[0]])

            def compute_divisor_volumes(self, point, in_basis=False):
                return np.asarray([point[0] ** 2])

            def compute_inverse_kahler_metric(self, point):
                return np.asarray([[1.0]])

        accepted, values = stage2_entrypoint.generator.evaluate_kaehler_point(
            FakeCalabiYau(),
            FakeKaehlerCone(),
            np.asarray([[1]], dtype=int),
            np.asarray([1.0]),
            attempt_index=1,
            point_kind="canonical_tip",
            point_seed=17,
            solver="test",
        )
        self.assertEqual(accepted["point_status"], "accepted")
        self.assertTrue(accepted["checks"]["cone_membership"])
        self.assertTrue(accepted["checks"]["positive_cy_volume"])
        self.assertTrue(accepted["checks"]["positive_effective_divisor_volumes"])
        self.assertEqual(accepted["point_seed"], 17)
        self.assertIsNotNone(accepted["point_sha256"])
        self.assertIsNotNone(values)

        failed, failed_values = stage2_entrypoint.generator.evaluate_kaehler_point(
            FakeCalabiYau(),
            FakeKaehlerCone(),
            np.asarray([[1]], dtype=int),
            np.asarray([0.5]),
            attempt_index=2,
            point_kind="randomized_projection",
            point_seed=18,
            solver="test",
        )
        self.assertEqual(failed["point_status"], "failed")
        self.assertFalse(failed["checks"]["cone_membership"])
        self.assertIsNone(failed_values)

    def test_kaehler_point_validation_rejects_divisor_volume_shortfall(self):
        class FakeKaehlerCone:
            def hyperplanes(self):
                return np.asarray([[1.0]])

        class FakeCalabiYau:
            def compute_cy_volume(self, point):
                return 1.0

            def compute_curve_volumes(self, point):
                return np.asarray([1.0])

            def compute_divisor_volumes(self, point, in_basis=False):
                return np.asarray([0.5])

            def compute_inverse_kahler_metric(self, point):
                return np.asarray([[1.0]])

        diagnostic, values = stage2_entrypoint.generator.evaluate_kaehler_point(
            FakeCalabiYau(),
            FakeKaehlerCone(),
            np.asarray([[1]], dtype=int),
            np.asarray([1.0]),
            attempt_index=1,
            point_kind="canonical_tip",
            point_seed=17,
            solver="test",
        )
        self.assertEqual(diagnostic["point_status"], "failed")
        self.assertFalse(diagnostic["checks"]["prime_divisor_volume_lower_bound"])
        self.assertFalse(
            diagnostic["checks"]["effective_divisor_volume_lower_bound"]
        )
        self.assertIsNone(values)

    def test_potential_reference_reconstruction_is_deterministic_and_transient(self):
        reference = {
            "h11": 1,
            "kappa": np.asarray([[0, 0, 0, 6.0]]),
            "tip": np.asarray([1.0]),
            "effective_cone": np.asarray([[1]], dtype=np.int64),
            "basis_matrix": np.asarray([[1]], dtype=np.int64),
            "prime_toric_divisors": np.asarray([0], dtype=np.int64),
        }
        assignment = {"qed_divisor_index": 0}

        first = stage2_entrypoint.generator.reconstruct_potential_from_reference(
            reference, assignment
        )
        second = stage2_entrypoint.generator.reconstruct_potential_from_reference(
            reference, assignment
        )

        np.testing.assert_array_equal(first["Q"], second["Q"])
        np.testing.assert_array_equal(first["L"], second["L"])
        self.assertEqual(first["certificate"], second["certificate"])
        self.assertEqual(first["certificate"]["storage"], "geometry_references_only")
        self.assertEqual(
            first["certificate"]["difference_convention"],
            "q_pair[:, k] = q_direct[:, pair_j[k]] - q_direct[:, pair_i[k]]",
        )

    def _two_divisor_potential_reference(self):
        return {
            "h11": 1,
            "kappa": np.asarray([[0, 0, 0, 6.0]]),
            "tip": np.asarray([1.0]),
            "effective_cone": np.asarray([[1]], dtype=np.int64),
            "basis_matrix": np.asarray([[1, 2]], dtype=np.int64),
            "prime_toric_divisors": np.asarray([0, 1], dtype=np.int64),
        }

    def test_geometry_potential_terms_cached_across_assignments(self):
        # A geometry's assignment pool can call reconstruct_potential_from_reference
        # once per pool entry (thousands of times for large h11). The O(rays^2)
        # geometry-only terms must be computed once per reference, not once per
        # assignment, or EFT finalization stalls for hours on large pools.
        generator = stage2_entrypoint.generator
        reference = self._two_divisor_potential_reference()
        original = generator._reconstruct_intersection_geometry
        with patch.object(
            generator, "_reconstruct_intersection_geometry", side_effect=original
        ) as mocked:
            first = generator.reconstruct_potential_from_reference(
                reference, {"qed_divisor_index": 0}
            )
            second = generator.reconstruct_potential_from_reference(
                reference, {"qed_divisor_index": 1}
            )
            self.assertEqual(mocked.call_count, 1)

        self.assertEqual(first["Q"].shape[1], 1)
        self.assertEqual(second["Q"].shape[1], 2)
        np.testing.assert_array_equal(second["qed_charge"], np.asarray([2]))

        other_reference = self._two_divisor_potential_reference()
        with patch.object(
            generator, "_reconstruct_intersection_geometry", side_effect=original
        ) as mocked_other:
            generator.reconstruct_potential_from_reference(
                other_reference, {"qed_divisor_index": 0}
            )
            self.assertEqual(mocked_other.call_count, 1)

    def test_geometry_potential_terms_append_does_not_leak_across_calls(self):
        generator = stage2_entrypoint.generator
        reference = self._two_divisor_potential_reference()

        appended = generator.reconstruct_potential_from_reference(
            reference, {"qed_divisor_index": 1}
        )
        self.assertEqual(appended["Q"].shape[1], 2)

        direct_again = generator.reconstruct_potential_from_reference(
            reference, {"qed_divisor_index": 0}
        )
        self.assertEqual(direct_again["Q"].shape[1], 1)
        self.assertEqual(reference["_potential_terms"]["q"].shape[1], 1)

    def test_leading_rank_order_reused_when_direct_and_invalidated_on_append(self):
        # classify_qed_leading_status's exact-rational elimination is the
        # single most expensive per-call cost at large h11 (O(rank) columns
        # times an O(basis) reduction each, on a matrix with O(rays^2)
        # columns). It depends only on q/l, so _geometry_potential_terms
        # computes it once per geometry and reconstruct_potential_from_reference
        # must hand that cached order back on every pool entry that resolves
        # to a direct QED source -- but not on the rarer entry that appends a
        # new column, since that column was never part of the cached
        # elimination.
        generator = stage2_entrypoint.generator
        reference = self._two_divisor_potential_reference()
        original = generator.compute_leading_rank_order
        with patch.object(
            generator, "compute_leading_rank_order", side_effect=original
        ) as mocked:
            direct = generator.reconstruct_potential_from_reference(
                reference, {"qed_divisor_index": 0}
            )
            appended = generator.reconstruct_potential_from_reference(
                reference, {"qed_divisor_index": 1}
            )
            # _geometry_potential_terms itself only runs once per reference,
            # so compute_leading_rank_order is called exactly once regardless
            # of how many assignments are scored against this geometry.
            self.assertEqual(mocked.call_count, 1)

        self.assertIsNotNone(direct["leading_rank_order"])
        self.assertIs(
            direct["leading_rank_order"],
            reference["_potential_terms"]["leading_rank_order"],
        )
        self.assertIsNone(appended["leading_rank_order"])

    def test_materialize_row_potential_caches_across_pool_entries(self):
        # _materialize_row_potential is expand_eft_reference_rows's real call
        # site (via build_row_for_draw), invoked once per assignment-pool
        # entry for the SAME outer reference object. It must hand
        # reconstruct_potential_from_reference the same reconstruction dict
        # every time so the geometry-only cache actually persists across a
        # geometry's whole pool, not just across repeat calls with an
        # already-identical object (which reconstruct_potential_from_reference
        # alone cannot guarantee if its caller rebuilds a fresh dict per call).
        generator = stage2_entrypoint.generator
        outer_reference = {
            "h11": 1,
            "reconstruction": {
                "kappa": np.asarray([[0, 0, 0, 6.0]]),
                "tip": np.asarray([1.0]),
                "effective_cone": np.asarray([[1]], dtype=np.int64),
                "basis_matrix": np.asarray([[1, 2]], dtype=np.int64),
                "prime_toric_divisors": np.asarray([0, 1], dtype=np.int64),
            },
        }
        original = generator._reconstruct_intersection_geometry
        with patch.object(
            generator, "_reconstruct_intersection_geometry", side_effect=original
        ) as mocked:
            first = generator._materialize_row_potential(
                outer_reference, {"qed_divisor_index": 0}
            )
            second = generator._materialize_row_potential(
                outer_reference, {"qed_divisor_index": 1}
            )
            self.assertEqual(mocked.call_count, 1)

        self.assertEqual(first["Q"].shape[1], 1)
        self.assertEqual(second["Q"].shape[1], 2)

    def test_no_valid_kaehler_point_has_point_shortfall_status(self):
        error = stage2_entrypoint.generator.NoPhysicalKaehlerPoint(
            "no valid point"
        )
        self.assertEqual(
            stage2_entrypoint.classify_stage2_failure(error),
            "kaehler_point_shortfall",
        )

    def test_divisor_volume_contract_records_labels_basis_and_vectors(self):
        evidence = stage2_entrypoint.generator.build_divisor_volume_evidence(
            np.asarray([0, 1]),
            np.asarray([[10, 0, 0, 0], [11, 0, 0, 0]]),
            np.asarray([40.0, 1.0]),
            np.asarray([[1, 0], [0, 1]]),
            np.asarray([2.0, 1.0]),
            np.asarray([[10, 0], [11, 0]]),
            1.0,
            1.0,
        )
        self.assertEqual(evidence["validation_status"], "passed")
        self.assertEqual(evidence["prime_divisor_indices"], [0, 1])
        self.assertEqual(evidence["prime_divisor_labels"][0], [10, 0, 0, 0])
        self.assertEqual(evidence["basis_order"], [[10, 0], [11, 0]])
        self.assertEqual(evidence["effective_divisor_volumes"], [2.0, 1.0])
        self.assertEqual(evidence["volume_tolerance"], 1e-8)

    def test_divisor_volume_contract_rejects_lower_bound_failure(self):
        with self.assertRaises(
            stage2_entrypoint.generator.FinalGeometryValidationFailed
        ):
            stage2_entrypoint.generator.build_divisor_volume_evidence(
                np.asarray([0]),
                np.asarray([[10, 0, 0, 0]]),
                np.asarray([0.999]),
                np.asarray([[1, 0]]),
                np.asarray([1.0]),
                np.asarray([[10, 0]]),
                1.0,
                1.0,
            )

    def test_kaehler_sampler_metadata_keeps_canonical_tip_point(self):
        class FakeKaehlerCone:
            def hyperplanes(self):
                return np.asarray([[1.0]])

        proposals = list(
            stage2_entrypoint.generator.sample_stretched_kaehler_points(
                FakeKaehlerCone(),
                np.asarray([1.0]),
                np.random.default_rng(17),
                1,
                lambda message: None,
                point_seed=17,
                include_metadata=True,
            )
        )
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["point_kind"], "canonical_tip")
        np.testing.assert_array_equal(proposals[0]["point"], [1.0])

    def test_randomized_kaehler_points_are_reproducible_per_seed(self):
        class FakeKaehlerCone:
            def hyperplanes(self):
                return np.eye(2)

        def fake_solve_qp(_matrix, linear_term, **_kwargs):
            # Make the fake projection depend on the deterministic target while
            # keeping every returned point inside the unit-slack cone.
            scale = 1.0 + 0.01 * float(np.linalg.norm(linear_term))
            return np.full(2, scale)

        def collect(seed):
            with patch(
                "generate_geometric_data_multitriangulation.configure_mosek_license",
                return_value={"activated": False},
            ), patch("qpsolvers.available_solvers", ["test"]), patch(
                "qpsolvers.solve_qp", side_effect=fake_solve_qp
            ):
                return list(
                    stage2_entrypoint.generator.sample_stretched_kaehler_points(
                        FakeKaehlerCone(),
                        np.asarray([1.0, 1.0]),
                        np.random.default_rng(999),
                        4,
                        lambda message: None,
                        point_seed=seed,
                        include_metadata=True,
                    )
                )

        first = collect(123)
        second = collect(123)
        different_seed = collect(124)
        self.assertEqual(
            [proposal["point_seed"] for proposal in first],
            [proposal["point_seed"] for proposal in second],
        )
        for first_proposal, second_proposal in zip(first, second):
            np.testing.assert_array_equal(
                first_proposal["point"], second_proposal["point"]
            )
        self.assertNotEqual(
            [proposal["point_seed"] for proposal in first[1:]],
            [proposal["point_seed"] for proposal in different_seed[1:]],
        )

    def test_randomized_point_budget_records_skipped_attempts(self):
        class FakeKaehlerCone:
            def hyperplanes(self):
                return np.eye(2)

        diagnostics = []
        with patch(
            "generate_geometric_data_multitriangulation.configure_mosek_license",
            return_value={"activated": False},
        ), patch("qpsolvers.available_solvers", ["test"]), patch(
            "qpsolvers.solve_qp", side_effect=RuntimeError("infeasible")
        ):
            proposals = list(
                stage2_entrypoint.generator.sample_stretched_kaehler_points(
                    FakeKaehlerCone(),
                    np.asarray([1.0, 1.0]),
                    np.random.default_rng(999),
                    4,
                    lambda message: None,
                    point_seed=123,
                    diagnostics=diagnostics,
                    include_metadata=True,
                )
            )
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["point_kind"], "canonical_tip")
        self.assertEqual([record["attempt_index"] for record in diagnostics], [2, 3, 4])
        self.assertTrue(all(record["attempted"] for record in diagnostics))
        self.assertTrue(all(record["point_status"] == "skipped" for record in diagnostics))
        self.assertEqual(len({record["point_seed"] for record in diagnostics}), 3)

    def test_canonical_qcd_contraction_is_opt_in(self):
        generator = stage2_entrypoint.generator
        # The sole tip divisor has volume 100, so normalizing it to 40 needs
        # m=sqrt(40/100)<1.  The default policy rejects that contraction;
        # the explicit opt-in accepts it.
        prime_volumes = np.asarray([100.0])
        basis_volumes = np.asarray([100.0])
        effective_rays = np.asarray([[1.0]])

        self.assertIsNone(
            generator.select_canonical_qcd_candidate(
                prime_volumes,
                basis_volumes,
                effective_rays,
                [0],
                40.0,
                1.0,
                1.0,
                1_000_000.0,
            )
        )
        selected = generator.select_canonical_qcd_candidate(
            prime_volumes,
            basis_volumes,
            effective_rays,
            [0],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
            allow_m_below_one=True,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 0)
        self.assertAlmostEqual(selected[1], np.sqrt(0.4))

    def test_canonical_qcd_defers_subunit_tip_divisor_cut_until_scaling(self):
        generator = stage2_entrypoint.generator
        prime_tip_volumes = np.asarray([2.0, 0.5])
        basis_tip_volumes = np.asarray([2.0])
        effective_rays = np.asarray([[0.4]])

        class FakeKaehlerCone:
            def hyperplanes(self):
                return np.asarray([[1.0]])

        class FakeCalabiYau:
            def compute_cy_volume(self, point):
                return 1.0

            def compute_curve_volumes(self, point):
                return np.asarray([1.0])

            def compute_divisor_volumes(self, point, in_basis=False):
                return basis_tip_volumes if in_basis else prime_tip_volumes

            def compute_inverse_kahler_metric(self, point):
                return np.asarray([[1.0]])

        diagnostic, values = generator.evaluate_kaehler_point(
            FakeCalabiYau(),
            FakeKaehlerCone(),
            effective_rays,
            np.asarray([1.0]),
            attempt_index=1,
            point_kind="canonical_tip",
            min_prime_divisor_volume=1.0,
            min_divisor_volume=1.0,
            enforce_divisor_volume_lower_bounds=False,
        )

        selected = generator.select_canonical_qcd_candidate(
            prime_tip_volumes,
            basis_tip_volumes,
            effective_rays,
            [0],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
        )

        self.assertLess(np.min(prime_tip_volumes), 1.0)
        self.assertEqual(diagnostic["point_status"], "accepted")
        self.assertFalse(diagnostic["checks"]["prime_divisor_volume_lower_bound"])
        self.assertFalse(diagnostic["checks"]["effective_divisor_volume_lower_bound"])
        self.assertIsNotNone(values)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 0)
        self.assertAlmostEqual(selected[1] ** 2, 20.0)
        self.assertGreaterEqual(
            selected[1] ** 2 * np.min(prime_tip_volumes), 1.0
        )

    def test_canonical_qcd_rejects_subunit_divisor_that_stays_invalid_after_scaling(self):
        generator = stage2_entrypoint.generator
        prime_tip_volumes = np.asarray([2.0, 0.01])
        basis_tip_volumes = np.asarray([2.0, 0.01])
        effective_rays = np.eye(2)

        selected = generator.select_canonical_qcd_candidate(
            prime_tip_volumes,
            basis_tip_volumes,
            effective_rays,
            [0],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
        )

        self.assertLess(20.0 * np.min(prime_tip_volumes), 1.0)
        self.assertIsNone(selected)

    def test_canonical_final_divisor_scaling_avoids_dilated_recompute_drift(self):
        generator = stage2_entrypoint.generator
        tau0 = np.asarray([2.0, 0.5])
        prime_tau0 = np.asarray([2.0, 0.5])
        effective_rays = np.eye(2)
        radial_scale = np.sqrt(20.0)

        tau, prime_volumes, effective_volumes = (
            generator.scale_canonical_divisor_volumes(
                tau0, prime_tau0, effective_rays, radial_scale
            )
        )

        np.testing.assert_allclose(tau, [40.0, 10.0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            prime_volumes, [40.0, 10.0], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            effective_volumes, [40.0, 10.0], rtol=0.0, atol=1e-12
        )
        self.assertTrue(
            np.isclose(
                prime_volumes[0],
                40.0,
                rtol=0.0,
                atol=generator.QCD_VOLUME_TOLERANCE,
            )
        )

    def test_canonical_qcd_uses_largest_later_tip_volume_first(self):
        generator = stage2_entrypoint.generator
        selected = generator.select_canonical_qcd_candidate(
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 2.0]),
            np.eye(2),
            [0, 1],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
        )
        self.assertEqual(selected[0], 1)
        self.assertAlmostEqual(selected[1], np.sqrt(20.0))

    def test_production_qed_prefilter_skips_first_candidate_and_keeps_later_one(self):
        generator = stage2_entrypoint.generator
        prime_tip_volumes = np.asarray([8.0, 4.0, 30.0, 10.0])
        neighbors = ((2,), (3,), (0,), (1,))
        invariant = np.ones(4, dtype=bool)

        prefilter = generator.prefilter_canonical_qcd_candidates(
            prime_tip_volumes,
            [0, 1],
            neighbors,
            invariant,
            40.0,
            generator.QED_VOLUME_MAX,
            1_000_000.0,
        )

        self.assertEqual(prefilter["eligible_candidate_indices"], [1])
        self.assertEqual(prefilter["candidate_records"][0]["rejection_reason"], "no_eligible_qed_neighbor")
        self.assertEqual(prefilter["candidate_records"][1]["eligible_qed_indices"], [3])
        no_survivor = generator.prefilter_canonical_qcd_candidates(
            prime_tip_volumes,
            [0],
            neighbors,
            invariant,
            40.0,
            generator.QED_VOLUME_MAX,
            1_000_000.0,
        )
        self.assertEqual(no_survivor["failure_status"], "qcd_qed_prefilter_shortfall")
        failure = generator.QEDAssignmentFailure(
            "qcd_qed_prefilter_shortfall",
            "prefilter rejected all candidates",
            no_survivor,
        )
        self.assertEqual(
            stage2_entrypoint.classify_stage2_failure(failure),
            "qcd_qed_prefilter_shortfall",
        )
        selected = generator.select_canonical_qcd_candidate(
            prime_tip_volumes,
            prime_tip_volumes,
            np.eye(4),
            prefilter["eligible_candidate_indices"],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
        )
        self.assertEqual(selected[0], 1)
        self.assertAlmostEqual(selected[1] ** 2 * prime_tip_volumes[3], 100.0)

    def test_production_qed_prefilter_uses_candidate_specific_post_normalization_volume(self):
        generator = stage2_entrypoint.generator
        prefilter = generator.prefilter_canonical_qcd_candidates(
            np.asarray([4.0, 2.0, 13.0, 1.0]),
            [0, 1],
            ((2,), (3,), (0,), (1,)),
            np.ones(4, dtype=bool),
            40.0,
            127.5,
            1_000_000.0,
        )

        first_record, later_record = prefilter["candidate_records"]
        self.assertAlmostEqual(first_record["radial_scale"] ** 2, 10.0)
        self.assertAlmostEqual(first_record["neighbor_records"][0]["final_qed_volume"], 130.0)
        self.assertEqual(first_record["rejection_reason"], "no_eligible_qed_neighbor")
        self.assertAlmostEqual(later_record["radial_scale"] ** 2, 20.0)
        self.assertAlmostEqual(later_record["neighbor_records"][0]["final_qed_volume"], 20.0)
        self.assertEqual(prefilter["eligible_candidate_indices"], [1])

    def test_production_qed_prefilter_is_inactive_outside_eft_canonical_intersecting_d7(self):
        generator = stage2_entrypoint.generator
        self.assertFalse(
            generator.canonical_qcd_qed_prefilter_active(
                eft_mode=False,
                moduli_policy="canonical_qcd",
                visible_sector_policy="intersecting_d7",
            )
        )
        self.assertFalse(
            generator.canonical_qcd_qed_prefilter_active(
                eft_mode=True,
                moduli_policy="adaptive",
                visible_sector_policy="intersecting_d7",
            )
        )
        self.assertFalse(
            generator.canonical_qcd_qed_prefilter_active(
                eft_mode=True,
                moduli_policy="canonical_qcd",
                visible_sector_policy="none",
            )
        )

    def test_assignment_pool_remains_authoritative_after_qed_prefilter(self):
        generator = stage2_entrypoint.generator
        prefilter = generator.prefilter_canonical_qcd_candidates(
            np.asarray([40.0, 1.0]),
            [0],
            ((1,), (0,)),
            np.asarray([False, True]),
            40.0,
            generator.QED_VOLUME_MAX,
            1_000_000.0,
        )
        self.assertEqual(prefilter["eligible_candidate_indices"], [0])

        pool = generator.enumerate_assignment_pool(
            prime_labels=((0, 0, 0, 0), (1, 0, 0, 0)),
            prime_charges=np.asarray([[1, 0], [0, 1]]),
            prime_volumes_reference=np.asarray([40.0, 1.0]),
            effective_volumes_reference=np.asarray([40.0, 1.0]),
            neighbors=((1,), (0,)),
            intersection_evidence={(0, 1): ((0, 1),)},
            invariant_mask=np.asarray([False, True]),
            qed_volume_max=generator.QED_VOLUME_MAX,
        )
        self.assertEqual(pool, [])
        with self.assertRaises(generator.QEDAssignmentFailure) as context:
            generator.validate_assignment_pool(pool)
        self.assertEqual(context.exception.category, "assignment_pool_shortfall")

    def test_canonical_qcd_falls_back_after_final_lower_bound_failure(self):
        generator = stage2_entrypoint.generator
        # Index 2 has the larger eligible tip volume and is tried first, but
        # its m=20 scaling leaves index 1 below the final lower bound.  The
        # smaller candidate at index 0 then succeeds with m=40.
        selected = generator.select_canonical_qcd_candidate(
            np.asarray([1.0, 0.04, 2.0]),
            np.asarray([1.0, 0.04, 2.0]),
            np.eye(3),
            [0, 2],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 0)
        self.assertAlmostEqual(selected[1], np.sqrt(40.0))

    def test_canonical_qcd_contraction_opt_in_preserves_legacy_order(self):
        generator = stage2_entrypoint.generator
        selected = generator.select_canonical_qcd_candidate(
            np.asarray([100.0, 3.0]),
            np.asarray([100.0, 3.0]),
            np.eye(2),
            [0, 1],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
            allow_m_below_one=True,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 0)
        self.assertAlmostEqual(selected[1], np.sqrt(0.4))

    def test_qcd_normalization_uses_approved_tolerances_without_snapping(self):
        normalization = stage2_entrypoint.generator.normalize_qcd_assignment(
            np.asarray([2.0, 2.0]),
            np.asarray([1.0, 1.0]),
            0,
            qcd_volume_tolerance=1e-9,
            divisor_volume_tolerance=1e-8,
        )
        self.assertTrue(
            np.isclose(normalization["qcd_volume"], 40.0, rtol=0.0, atol=1e-9)
        )
        self.assertLessEqual(normalization["qcd_volume_residual"], 1e-9)
        self.assertEqual(normalization["qcd_volume_tolerance"], 1e-9)
        self.assertEqual(normalization["divisor_volume_tolerance"], 1e-8)

    def test_final_qcd_validation_is_strict_and_classifies_failures(self):
        generator = stage2_entrypoint.generator
        values = dict(
            point=np.asarray([1.0]),
            radial_scale=1.0,
            max_m=1_000_000.0,
            allow_m_below_one=False,
            qcd_divisor_index=0,
            qcd_volume_target=40.0,
            qcd_volume_min=40.0,
            qcd_volume_max=40.0,
            cy_volume=1.0,
            curve_volumes=np.asarray([1.0]),
            kaehler_slack=np.asarray([1.0]),
            inverse_metric=np.asarray([[1.0]]),
            prime_divisor_volumes=np.asarray([40.0, 1.0]),
            effective_divisor_volumes=np.asarray([1.0, 2.0]),
            min_prime_divisor_volume=1.0,
            min_divisor_volume=1.0,
        )
        result = generator.validate_final_qcd_normalization(**values)
        self.assertEqual(result["validation_status"], "passed")
        self.assertEqual(result["repair_policy"], "none")
        invalid = dict(values, prime_divisor_volumes=np.asarray([40.0, 0.999]))
        with self.assertRaises(generator.FinalGeometryValidationFailed) as context:
            generator.validate_final_qcd_normalization(**invalid)
        self.assertEqual(
            stage2_entrypoint.classify_stage2_failure(context.exception),
            "qcd_normalization_failure",
        )

    def _write_stage1_fixture(self, root, *, missing=False):
        raw_path = (
            Path(root)
            / "frst_candidates"
            / "h11_001"
            / "np_0000001"
            / "frst_0000001.h5"
        )
        if not missing:
            metadata = write_raw_frst_artifact(
                raw_path,
                h11=1,
                polytope_vertices=np.asarray(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
                ),
                polytope_points=np.asarray(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
                ),
                triangulation_labels=np.arange(5),
                triangulation_points=np.asarray(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1]]
                ),
                simplices=np.asarray([[0, 1, 2, 3, 4]]),
                simplex_indices=np.asarray([[0, 1, 2, 3, 4]]),
                metadata={"polytope_index": 1, "candidate_index": 1},
            )
            stage1_path = metadata["raw_frst_path"]
        else:
            stage1_path = str(raw_path.resolve())
            metadata = {
                "h11": 1,
                "polytope_id": "missing",
                "full_triangulation_hash": "missing",
                "geometry_id": "missing",
                "raw_frst_path": stage1_path,
            }
        atomic_jsonl_dump(
            Path(root) / "frst_terminal_statuses.jsonl",
            [{**metadata, "terminal_status": "retained_raw_frst"}],
        )
        return Path(stage1_path)

    def test_raw_identity_round_trip_and_separate_ledger(self):
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-boundary-") as temporary:
            root = Path(temporary) / "stage1"
            root.mkdir()
            raw_path = self._write_stage1_fixture(root)
            persisted = read_raw_frst_artifact(raw_path)
            self.assertEqual(persisted["stage1_status"], "retained")
            ledger = build_input_ledger(root)
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]["stage2_input_status"], "retained_raw_frst")
            self.assertEqual(ledger[0]["full_triangulation_hash"], persisted["full_triangulation_hash"])

    def test_missing_retained_raw_file_is_unavailable(self):
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-missing-") as temporary:
            root = Path(temporary) / "stage1"
            root.mkdir()
            self._write_stage1_fixture(root, missing=True)
            ledger = build_input_ledger(root)
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]["stage2_input_status"], "missing_raw_frst")

    def test_stage1_ledger_identity_mismatch_is_terminal(self):
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-identity-") as temporary:
            root = Path(temporary) / "stage1"
            root.mkdir()
            self._write_stage1_fixture(root)
            status_path = root / "frst_terminal_statuses.jsonl"
            status_record = json.loads(status_path.read_text().splitlines()[0])
            status_record["full_triangulation_hash"] = "tampered-ledger-hash"
            status_path.unlink()
            atomic_jsonl_dump(status_path, [status_record])
            ledger = build_input_ledger(root)
            self.assertEqual(ledger[0]["stage2_input_status"], "input_identity_mismatch")
            self.assertIn("identity_mismatches", ledger[0])

    def test_reconstructed_triangulation_hash_mismatch_is_terminal(self):
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-reconstruction-") as temporary:
            root = Path(temporary) / "stage1"
            root.mkdir()
            self._write_stage1_fixture(root)
            raw_record = build_input_ledger(root)[0]
            fake_polytope = MagicMock()
            fake_polytope.dim.return_value = 4
            fake_polytope.ambient_dim.return_value = 4
            fake_polytope.is_reflexive.return_value = True
            fake_polytope.labels_not_facet = np.arange(5)
            fake_triangulation = MagicMock()
            fake_triangulation.labels = np.arange(5)
            fake_triangulation.is_fine.return_value = True
            fake_triangulation.is_regular.return_value = True
            fake_triangulation.is_star.return_value = True
            fake_triangulation.is_valid.return_value = True
            altered_simplices = np.asarray([[0, 1, 2, 3, 3]])
            fake_triangulation.simplices.return_value = altered_simplices
            fake_polytope.triangulate.return_value = fake_triangulation
            topology_audit = stage2_entrypoint.build_topology_audit_record(
                raw_record, "qhull"
            )
            with patch.object(
                stage2_entrypoint.generator,
                "Polytope",
                return_value=fake_polytope,
            ):
                with self.assertRaises(RawFRSTError) as context:
                    stage2_entrypoint.reconstruct_raw_frst(
                        raw_record, "qhull", topology_audit
                    )
            self.assertEqual(context.exception.terminal_status, "input_identity_mismatch")
            self.assertEqual(
                topology_audit["reconstructed_triangulation_hash"],
                compute_triangulation_hash(altered_simplices),
            )

    def test_orientifold_preservation_failure_has_explicit_terminal_status(self):
        error = stage2_entrypoint.generator.OrientifoldValidationFailure(
            "selected FRST is not preserved"
        )
        self.assertEqual(
            stage2_entrypoint.classify_stage2_failure(error),
            "orientifold_invariance_failure",
        )

    def test_fan_invariant_orientifold_supports_visible_sector_without_even_kaehler_check(self):
        orientifold = {
            "requested": True,
            "status": "fan_invariant",
            "involution_type": "O3/O7",
            "prime_divisor_image_indices": np.arange(2),
        }
        candidates = stage2_entrypoint.generator._visible_qcd_candidates(
            "intersecting_d7", orientifold, ((1,), (0,))
        )
        self.assertEqual(candidates, [0, 1])

    def test_stage2_dry_run_writes_a_distinct_input_ledger(self):
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-dry-run-") as temporary:
            stage1_root = Path(temporary) / "stage1"
            stage1_root.mkdir()
            self._write_stage1_fixture(stage1_root)
            stage2_root = Path(temporary) / "stage2"
            stage2_main(
                [
                    "--stage1-root",
                    str(stage1_root),
                    "--outdir",
                    str(stage2_root),
                    "--visible-sector-policy",
                    "none",
                    "--dry-run",
                ]
            )
            self.assertTrue((stage2_root / "stage2_input_ledger.jsonl").is_file())
            self.assertTrue((stage2_root / "stage2_progress.jsonl").is_file())
            self.assertTrue((stage2_root / "run_manifest.json").is_file())
            self.assertTrue((stage2_root / "charge_factorized_manifest.json").is_file())
            self.assertTrue((stage2_root / "polytope_manifest.json").is_file())
            self.assertTrue(
                (stage2_root / "stage2_kaehler_point_diagnostics.jsonl").is_file()
            )
            self.assertTrue(
                (stage2_root / "stage2_assignment_pool_rejections.jsonl").is_file()
            )
            diagnostics_path = stage2_root / "stage2_topology_diagnostics.jsonl"
            self.assertTrue(diagnostics_path.is_file())
            diagnostics = [
                json.loads(line)
                for line in diagnostics_path.read_text().splitlines()
            ]
            self.assertEqual(len(diagnostics), 1)
            self.assertEqual(diagnostics[0]["audit_status"], "not_run")
            self.assertEqual(diagnostics[0]["stage2_terminal_status"], "dry_run")
            self.assertEqual(
                diagnostics[0]["orientifold_validation"]["h11_parity_policy"],
                "record_only_not_enforced",
            )
            self.assertFalse(list(stage2_root.glob("h11_*/np_*/cy_*/cyax.h5")))
            progress_events = [
                json.loads(line)
                for line in (stage2_root / "stage2_progress.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(progress_events[0]["event"], "run_started")
            self.assertEqual(progress_events[-1]["event"], "run_finalized")
            manifest = json.loads((stage2_root / "run_manifest.json").read_text())
            self.assertEqual(manifest["stage1_root"], str(stage1_root.resolve()))
            self.assertTrue(manifest["stage2_filters_do_not_replenish_stage1"])
            self.assertEqual(manifest["moduli_policy"], "canonical_qcd")
            self.assertEqual(manifest["kaehler_point_attempt_budget"], 100)
            self.assertEqual(
                manifest["divisor_volume_contract"]["prime_lower_bound"], 1.0
            )
            self.assertEqual(
                manifest["divisor_volume_contract"]["effective_lower_bound"], 1.0
            )
            self.assertEqual(
                manifest["divisor_volume_contract"]["tolerance"], 1e-8
            )
            self.assertEqual(manifest["qcd_volume_target"], 40.0)
            self.assertEqual(manifest["qcd_volume_tolerance"], 1e-9)
            self.assertEqual(manifest["divisor_volume_tolerance"], 1e-8)
            self.assertEqual(
                manifest["normalization_failure_status"],
                "qcd_normalization_failure",
            )
            self.assertEqual(
                manifest["stage2_assignment_pool_rejections"],
                "stage2_assignment_pool_rejections.jsonl",
            )
            self.assertEqual(
                manifest["assignment_pool_rejection_policy"]["hdf5"],
                "aggregate_rejection_counts_and_reasons_only",
            )
            self.assertEqual(manifest["normalization_repair_policy"], "none")
            self.assertEqual(
                manifest["selection_policy"],
                "explicit_qcd_divisor_index_or_deterministic_minimal_dilation",
            )
            self.assertEqual(
                manifest["post_selection_fallback"],
                "try_next_candidate_after_final_lower_bound_failure",
            )
            stage7_decisions = [
                decision for decision in manifest["user_decisions"]
                if decision["stage"] == 7
            ]
            self.assertEqual(len(stage7_decisions), 3)
            stage9_decisions = [
                decision for decision in manifest["user_decisions"]
                if decision["stage"] == 9
            ]
            self.assertEqual(len(stage9_decisions), 3)
            stage11_decisions = [
                decision for decision in manifest["user_decisions"]
                if decision["stage"] == 11
            ]
            self.assertEqual(len(stage11_decisions), 1)
            self.assertEqual(
                manifest["eft"]["sampling_policy"]["assignment_sampling"],
                "uniform_with_replacement",
            )
            self.assertEqual(
                manifest["eft"]["sampling_policy"]["draw_cap_formula"],
                "M_g = 10 * k_g",
            )
            self.assertNotIn(
                "Stage 10 temporary-artifact retention policy remains unresolved.",
                manifest["unresolved_scientific_choices"],
            )
            self.assertEqual(
                manifest["geometry_artifact_policy"]["temporary_artifact_policy"],
                "delete_after_status_recording",
            )
            self.assertFalse(
                manifest["orientifold_policy"]["required_for_visible_sector_policy"]
            )

    def test_eft_row_bounds_are_configurable_not_fixed(self):
        # --eft-minimum-rows/--eft-maximum-rows default to the approved
        # schema 1.1 values (100000/200000) but must accept other positive,
        # minimum-at-most-maximum values for bounded exploratory or
        # validation runs -- they are not locked to exactly those two
        # numbers.
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-row-bounds-") as temporary:
            stage1_root = Path(temporary) / "stage1"
            stage1_root.mkdir()
            self._write_stage1_fixture(stage1_root)
            orientifold_path = Path(temporary) / "orientifold.json"
            orientifold_path.write_text(
                json.dumps(
                    {
                        "involution_type": "O3/O7",
                        "lattice_matrix": np.eye(4, dtype=int).tolist(),
                    }
                )
            )
            stage2_root = Path(temporary) / "stage2"
            stage2_main(
                [
                    "--stage1-root",
                    str(stage1_root),
                    "--outdir",
                    str(stage2_root),
                    "--visible-sector-policy",
                    "intersecting_d7",
                    "--orientifold-file",
                    str(orientifold_path),
                    "--eft",
                    "--eft-minimum-rows",
                    "5",
                    "--eft-maximum-rows",
                    "180000",
                    "--dry-run",
                ]
            )
            manifest = json.loads((stage2_root / "run_manifest.json").read_text())
            self.assertEqual(manifest["eft"]["minimum_rows"], 5)
            self.assertEqual(manifest["eft"]["maximum_rows"], 180000)

    def test_stage12_subminimum_capacity_writes_diagnostic_partial_output(self):
        try:
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is not available")
        with tempfile.TemporaryDirectory(prefix="cyax-stage12-partial-") as temporary:
            stage1_root = Path(temporary) / "stage1"
            stage1_root.mkdir()
            self._write_stage1_fixture(stage1_root)
            orientifold_path = Path(temporary) / "orientifold.json"
            orientifold_path.write_text(
                json.dumps(
                    {
                        "involution_type": "O3/O7",
                        "lattice_matrix": np.eye(4, dtype=int).tolist(),
                    }
                )
            )
            stage2_root = Path(temporary) / "stage2"
            raw_record = build_input_ledger(stage1_root)[0]

            def fake_process(
                arguments,
                record,
                orientifold_config,
                output_root,
                progress_callback=None,
            ):
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "candidate_stage_started",
                            "stage": "fake_stage",
                            "geometry_id": record["geometry_id"],
                        }
                    )
                topology_audit = stage2_entrypoint.build_topology_audit_record(
                    record, arguments.backend
                )
                topology_audit.update(
                    {
                        "audit_status": "complete",
                        "stage2_terminal_status": "kaehler_point_shortfall",
                        "kaehler_point_scan": {
                            "policy": "canonical_qcd",
                            "attempt_budget": 100,
                            "point_status": "failed",
                            "diagnostics": [],
                        },
                        "assignment_pool_rejection_records": [],
                    }
                )
                return (
                    {
                        **record,
                        "terminal_status": "kaehler_point_shortfall",
                        "terminal_reason": "synthetic Stage-2 shortfall",
                        "output_path": str(stage2_root / "not-written" / "cyax.h5"),
                    },
                    topology_audit,
                )

            with patch.object(
                stage2_entrypoint.generator, "require_cytools_capabilities"
            ), patch.object(
                stage2_entrypoint,
                "process_raw_frst_artifact",
                side_effect=fake_process,
            ):
                stage2_main(
                    [
                        "--stage1-root",
                        str(stage1_root),
                        "--outdir",
                        str(stage2_root),
                        "--eft",
                        "--orientifold-file",
                        str(orientifold_path),
                    ]
                )

            eft_path = stage2_root / "eft_models.parquet"
            self.assertTrue(eft_path.is_file())
            metadata = parquet.read_schema(eft_path).metadata
            self.assertEqual(
                metadata[b"cyaxiverse_dataset_status"], b"diagnostic_partial"
            )
            self.assertEqual(metadata[b"production_complete"], b"False")
            manifest = json.loads((stage2_root / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed_diagnostic_partial")
            self.assertTrue(manifest["eft"]["diagnostic_success"])
            self.assertFalse(manifest["eft"]["production_complete"])
            self.assertTrue(manifest["eft"]["model_target_shortfall"])
            self.assertEqual(manifest["eft"]["rows_written"], 0)
            self.assertEqual(manifest["eft"]["validated_assignment_capacity"], 0)
            self.assertEqual(manifest["eft"]["minimum_rows"], 100_000)
            self.assertEqual(manifest["eft"]["maximum_rows"], 200_000)
            self.assertTrue(manifest["stage1_population_frozen"])
            self.assertFalse(manifest["stage1_replenishment_allowed"])
            model_statuses = [
                json.loads(line)
                for line in (stage2_root / "model_terminal_statuses.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(
                any(
                    record["terminal_status"] == "model_target_shortfall"
                    for record in model_statuses
                )
            )
            self.assertTrue(
                any(
                    record["terminal_status"]
                    == "accepted_diagnostic_partial_model_table"
                    for record in model_statuses
                )
            )

    def test_point_shortfall_retains_stage1_identity_in_stage2_accounting(self):
        with tempfile.TemporaryDirectory(prefix="cyax-stage2-point-shortfall-") as temporary:
            stage1_root = Path(temporary) / "stage1"
            stage1_root.mkdir()
            self._write_stage1_fixture(stage1_root)
            stage2_root = Path(temporary) / "stage2"
            raw_record = build_input_ledger(stage1_root)[0]

            def fake_process(
                arguments,
                record,
                orientifold_config,
                output_root,
                progress_callback=None,
            ):
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "candidate_stage_started",
                            "stage": "fake_stage",
                            "geometry_id": record["geometry_id"],
                        }
                    )
                topology_audit = stage2_entrypoint.build_topology_audit_record(
                    record, arguments.backend
                )
                topology_audit.update(
                    {
                        "orientifold_validation": stage2_entrypoint.orientifold_audit_record(
                            orientifold_config
                        ),
                        "audit_status": "complete",
                        "stage2_terminal_status": "kaehler_point_shortfall",
                        "kaehler_point_scan": {
                            "policy": "adaptive",
                            "attempt_budget": 100,
                            "point_status": "failed",
                            "diagnostics": [
                                {
                                    "attempt_index": 1,
                                    "point_kind": "canonical_tip",
                                    "point_seed": 77,
                                    "attempted": True,
                                    "point_status": "failed",
                                    "failure_reason": "no valid point",
                                }
                            ],
                        },
                        "assignment_pool_rejection_records": [
                            {
                                "geometry_id": record["geometry_id"],
                                "qcd_index": 0,
                                "qcd_label": [0, 0, 0, 0],
                                "qed_index": 1,
                                "qed_label": [1, 0, 0, 0],
                                "terminal_status": "qed_volume_rejection",
                                "terminal_reason": "boundary test rejection",
                            }
                        ],
                    }
                )
                terminal_record = {
                    **record,
                    "terminal_status": "kaehler_point_shortfall",
                    "terminal_reason": "no valid point within the bounded scan",
                    "output_path": str(
                        Path(output_root) / "not-written" / "cyax.h5"
                    ),
                }
                return terminal_record, topology_audit

            with patch.object(
                stage2_entrypoint.generator,
                "require_cytools_capabilities",
            ), patch.object(
                stage2_entrypoint,
                "process_raw_frst_artifact",
                side_effect=fake_process,
            ):
                stage2_main(
                    [
                        "--stage1-root",
                        str(stage1_root),
                        "--outdir",
                        str(stage2_root),
                        "--visible-sector-policy",
                        "none",
                    ]
                )

            terminal = json.loads(
                (stage2_root / "stage2_terminal_statuses.jsonl")
                .read_text()
                .splitlines()[0]
            )
            point_diagnostic = json.loads(
                (stage2_root / "stage2_kaehler_point_diagnostics.jsonl")
                .read_text()
                .splitlines()[0]
            )
            manifest = json.loads((stage2_root / "run_manifest.json").read_text())
            partial_rejection = json.loads(
                (stage2_root / "stage2_assignment_pool_rejections.partial.jsonl")
                .read_text()
                .splitlines()[0]
            )
            rejection = json.loads(
                (stage2_root / "stage2_assignment_pool_rejections.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(terminal["terminal_status"], "kaehler_point_shortfall")
            self.assertEqual(terminal["geometry_id"], raw_record["geometry_id"])
            self.assertEqual(terminal["raw_frst_path"], raw_record["raw_frst_path"])
            self.assertEqual(point_diagnostic["geometry_id"], raw_record["geometry_id"])
            self.assertEqual(point_diagnostic["point_status"], "failed")
            self.assertEqual(
                manifest["stage2_terminal_count_by_h11_and_status"]["1"],
                {"kaehler_point_shortfall": 1},
            )
            self.assertEqual(
                manifest["kaehler_point_status_counts"], {"failed": 1}
            )
            self.assertEqual(
                partial_rejection["terminal_status"], "qed_volume_rejection"
            )
            self.assertEqual(partial_rejection["terminal_reason"], "boundary test rejection")
            self.assertEqual(rejection["geometry_id"], raw_record["geometry_id"])
            self.assertEqual(rejection["qcd_label"], [0, 0, 0, 0])
            self.assertEqual(rejection["qed_index"], 1)
            self.assertEqual(rejection["terminal_status"], "qed_volume_rejection")
            self.assertEqual(rejection["terminal_reason"], "boundary test rejection")
            progress_events = [
                json.loads(line)
                for line in (stage2_root / "stage2_progress.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in progress_events],
                [
                    "run_started",
                    "candidate_started",
                    "candidate_stage_started",
                    "candidate_finished",
                    "run_finalized",
                ],
            )


if __name__ == "__main__":
    unittest.main()
