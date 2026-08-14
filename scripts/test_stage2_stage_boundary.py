"""Verify the independent raw-FRST to stage-2 input boundary."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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

    def test_no_valid_kaehler_point_has_point_shortfall_status(self):
        error = stage2_entrypoint.generator.NoPhysicalKaehlerPoint(
            "no valid point"
        )
        self.assertEqual(
            stage2_entrypoint.classify_stage2_failure(error),
            "kaehler_point_shortfall",
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
            self.assertTrue((stage2_root / "run_manifest.json").is_file())
            self.assertTrue((stage2_root / "charge_factorized_manifest.json").is_file())
            self.assertTrue((stage2_root / "polytope_manifest.json").is_file())
            self.assertTrue(
                (stage2_root / "stage2_kaehler_point_diagnostics.jsonl").is_file()
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
            manifest = json.loads((stage2_root / "run_manifest.json").read_text())
            self.assertEqual(manifest["stage1_root"], str(stage1_root.resolve()))
            self.assertTrue(manifest["stage2_filters_do_not_replenish_stage1"])
            self.assertEqual(manifest["moduli_policy"], "canonical_qcd")
            self.assertEqual(manifest["kaehler_point_attempt_budget"], 100)
            self.assertFalse(
                manifest["orientifold_policy"]["required_for_visible_sector_policy"]
            )


if __name__ == "__main__":
    unittest.main()
