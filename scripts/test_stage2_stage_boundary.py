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

    def test_canonical_qcd_candidate_order_has_no_post_selection_fallback(self):
        generator = stage2_entrypoint.generator
        selected = generator.select_canonical_qcd_candidate(
            np.asarray([100.0, 1.0]),
            np.asarray([1.0, 1.0]),
            np.eye(2),
            [0, 1],
            40.0,
            1.0,
            1.0,
            1_000_000.0,
        )
        self.assertEqual(selected[0], 1)
        self.assertIsNone(
            generator.select_canonical_qcd_candidate(
                np.asarray([100.0, 1.0]),
                np.asarray([1.0, 1.0]),
                np.eye(2),
                [0],
                40.0,
                1.0,
                1.0,
                1_000_000.0,
            )
        )

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
                "explicit_qcd_divisor_index_or_first_eligible_ascending_index",
            )
            self.assertEqual(manifest["post_selection_fallback"], "none")
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
