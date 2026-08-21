#!/usr/bin/env python3
"""Focused tests for audit-driver argument and population contracts."""

from contextlib import redirect_stderr
import hashlib
from io import StringIO
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reproduce_fuzzy_axions_h11_4 as driver


class ReproduceFuzzyAxionsDriverTest(unittest.TestCase):
    def test_known_table_population_requires_matching_favorable_count(self):
        complete = driver._population_completion_status(2, 36)
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["basis"], "table_1_favorable_polytope_target")

        incomplete = driver._population_completion_status(2, 35)
        self.assertFalse(incomplete["complete"])
        self.assertIn("target for h11=2 is 36", incomplete["reason"])

    def test_unknown_population_requires_explicit_basis(self):
        without_basis = driver._population_completion_status(99, 36)
        self.assertFalse(without_basis["complete"])
        self.assertEqual(without_basis["basis"], "no_explicit_basis")

        with_basis = driver._population_completion_status(
            99,
            36,
            explicit_basis={
                "favorable_polytopes": 36,
                "label": "checked_local_manifest",
            },
        )
        self.assertTrue(with_basis["complete"])
        self.assertEqual(with_basis["basis"], "explicit:checked_local_manifest")

    def test_unknown_population_completion_metadata_is_machine_readable(self):
        args = SimpleNamespace(
            parquet_dir="unused",
            h11=99,
            limit=10**9,
            progress=50,
            keep_details=False,
            orientifold_audit=False,
            orientifold_reason_diagnostics=False,
            export_kaehler_points=False,
            model_stage=False,
            qcd_divisor_domain="all_prime",
        )
        with patch.object(driver, "load_mirror_polytopes", return_value=[]):
            summary = driver.reproduce(args)

        input_data = summary["input"]
        self.assertFalse(input_data["population_complete"])
        self.assertEqual(input_data["population_completion_basis"], "no_explicit_basis")
        self.assertIn("no Table 1 favorable-polytope target", input_data["population_completion_reason"])
        self.assertIsNone(input_data["population_completion_expected_favorable_polytopes"])

    def test_provenance_manifest_hashes_scanned_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            first = directory_path / "polytopes-4d-05-vertices.parquet"
            second = directory_path / "polytopes-4d-06-vertices.parquet"
            first.write_bytes(b"first partition")
            second.write_bytes(b"second partition")
            records = [(None, {"parquet_file": str(first)})]

            manifest = driver._parquet_input_manifest(directory, records, limit=1)

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["scan_basis"], "loader_record_boundary")
        self.assertEqual(len(manifest["partitions"]), 1)
        partition = manifest["partitions"][0]
        self.assertEqual(partition["size_bytes"], len(b"first partition"))
        self.assertEqual(
            partition["sha256"],
            hashlib.sha256(b"first partition").hexdigest(),
        )

    def test_existing_output_is_rejected_before_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("{}", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                driver._parse_args(
                    ["--parquet-dir", "unused", "--output", str(output)]
                )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("refusing to overwrite existing output", stderr.getvalue())

    def test_schema_metadata_exposes_renamed_source_evidence_counts(self):
        args = SimpleNamespace(
            parquet_dir="unused",
            h11=99,
            limit=10**9,
            progress=50,
            keep_details=False,
            orientifold_audit=False,
            orientifold_reason_diagnostics=False,
            export_kaehler_points=False,
            model_stage=False,
            qcd_divisor_domain="all_prime",
        )
        with patch.object(driver, "load_mirror_polytopes", return_value=[]):
            summary = driver.reproduce(args)

        self.assertEqual(summary["schema_version"], driver.REPRODUCTION_SCHEMA_VERSION)
        aliases = summary["schema_metadata"]["deprecated_aliases"]
        self.assertEqual(
            aliases["counts.source_vertex_evidence_inherited_orientifold_cys"],
            "counts.source_evidence_inherited_orientifold_cys",
        )
        self.assertIn("source_evidence_inherited_orientifold_cys", summary["counts"])

    def test_runtime_provenance_has_explicit_environment_fields(self):
        args = SimpleNamespace(
            parquet_dir="unused",
            h11=99,
            limit=10**9,
            progress=50,
            keep_details=False,
            orientifold_audit=False,
            orientifold_reason_diagnostics=False,
            export_kaehler_points=False,
            model_stage=False,
            qcd_divisor_domain="all_prime",
        )
        with patch.object(driver, "load_mirror_polytopes", return_value=[]):
            summary = driver.reproduce(args)

        provenance = summary["run_provenance"]
        self.assertIn("source_commit", provenance)
        self.assertIn("git_dirty", provenance)
        self.assertEqual(provenance["package_version"], "0.2.0")
        self.assertIn("python", provenance["runtime_versions"])
        self.assertIn("cli_arguments", provenance)
        self.assertEqual(provenance["input_partition_manifest"]["status"], "unavailable")

    def test_reason_diagnostics_requires_orientifold_audit(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            driver._parse_args(
                [
                    "--parquet-dir",
                    "unused",
                    "--orientifold-reason-diagnostics",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "--orientifold-reason-diagnostics requires --orientifold-audit",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
