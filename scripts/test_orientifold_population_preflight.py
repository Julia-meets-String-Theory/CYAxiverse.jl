#!/usr/bin/env python3
"""Focused tests for the higher-h11 evidence preflight.

These tests exercise only text, checksum, and metadata validation.  They do
not import CYTools, read parquet files, or run a population scan.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orientifold_population_preflight import (
    HandoffRequirement,
    PopulationPreflightError,
    _read_checksum_manifest,
    _read_handoffs,
)


class PopulationPreflightTests(unittest.TestCase):
    def test_handoff_reader_reads_all_required_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CYAxiverse.jl"
            handoff = root.parent / "handoffs_checkpoints" / "run.md"
            handoff.parent.mkdir(parents=True)
            content = "fresh h11 = 4\nledger\nDo not regenerate\n"
            handoff.write_text(content, encoding="utf-8")
            result = _read_handoffs(
                root,
                (HandoffRequirement("handoffs_checkpoints/run.md", ("fresh h11 = 4", "ledger")),),
            )
            self.assertEqual(result[0]["bytes_read"], len(content.encode("utf-8")))
            self.assertEqual(result[0]["sha256"], hashlib.sha256(content.encode()).hexdigest())

    def test_handoff_reader_fails_closed_when_marker_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CYAxiverse.jl"
            handoff = root.parent / "handoffs_checkpoints" / "run.md"
            handoff.parent.mkdir(parents=True)
            handoff.write_text("fresh h11 = 4\n", encoding="utf-8")
            with self.assertRaises(PopulationPreflightError):
                _read_handoffs(
                    root,
                    (HandoffRequirement("handoffs_checkpoints/run.md", ("ledger",)),),
                )

    def test_checksum_manifest_rejects_malformed_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "SHA256SUMS.txt"
            manifest.write_text("not-a-checksum file.zst\n", encoding="utf-8")
            with self.assertRaises(PopulationPreflightError):
                _read_checksum_manifest(manifest)

    def test_source_contract_rejects_extra_partition(self):
        from orientifold_population_preflight import (
            IMPLEMENTATION_HANDOFF,
            REQUIRED_SOURCE_PARTITIONS,
            _implementation_source_contract,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CYAxiverse.jl"
            source = root.parent / "source"
            source.mkdir(parents=True)
            for name in REQUIRED_SOURCE_PARTITIONS + ("polytopes-4d-11-vertices.parquet",):
                (source / name).write_bytes(name.encode())
            record = {
                "type": "source",
                "path": str(source),
                "digest": "a" * 64,
                "hashes": {name[13:15]: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in REQUIRED_SOURCE_PARTITIONS},
            }
            with patch("orientifold_population_preflight._read_zstd_jsonl", return_value=[record]):
                with self.assertRaises(PopulationPreflightError):
                    _implementation_source_contract(root)

    def test_source_contract_rejects_partition_hash_mismatch(self):
        from orientifold_population_preflight import (
            REQUIRED_SOURCE_PARTITIONS,
            _implementation_source_contract,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CYAxiverse.jl"
            source = root.parent / "source"
            source.mkdir(parents=True)
            for name in REQUIRED_SOURCE_PARTITIONS:
                (source / name).write_bytes(name.encode())
            hashes = {
                name[13:15]: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in REQUIRED_SOURCE_PARTITIONS
            }
            hashes["05"] = "b" * 64
            record = {
                "type": "source",
                "path": str(source),
                "digest": "a" * 64,
                "hashes": hashes,
            }
            with patch("orientifold_population_preflight._read_zstd_jsonl", return_value=[record]):
                with self.assertRaises(PopulationPreflightError):
                    _implementation_source_contract(root)

    def test_source_contract_accepts_byte_identical_durable_alternate_path(self):
        from orientifold_population_preflight import (
            IMPLEMENTATION_HANDOFF,
            REQUIRED_SOURCE_PARTITIONS,
            _implementation_source_contract,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CYAxiverse.jl"
            source = root.parent / "durable-source"
            source.mkdir(parents=True)
            (root.parent / IMPLEMENTATION_HANDOFF).write_bytes(b"handoff")
            for name in REQUIRED_SOURCE_PARTITIONS:
                (source / name).write_bytes(name.encode())
            hashes = {
                name[13:15]: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in REQUIRED_SOURCE_PARTITIONS
            }
            record = {
                "type": "source",
                "path": "/private/tmp/old-ephemeral-mirror",
                "digest": "a" * 64,
                "hashes": hashes,
            }
            with patch("orientifold_population_preflight._read_zstd_jsonl", return_value=[record]):
                contract = _implementation_source_contract(root, source)
            self.assertEqual(contract["source_path"], str(source.resolve()))
            self.assertEqual(contract["handoff_source_path"], "/private/tmp/old-ephemeral-mirror")
            self.assertEqual(contract["source_path_origin"], "caller_override")

    def test_source_contract_rejects_alternate_path_hash_mismatch_and_extra_partition(self):
        from orientifold_population_preflight import (
            IMPLEMENTATION_HANDOFF,
            REQUIRED_SOURCE_PARTITIONS,
            _implementation_source_contract,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CYAxiverse.jl"
            source = root.parent / "durable-source"
            source.mkdir(parents=True)
            (root.parent / IMPLEMENTATION_HANDOFF).write_bytes(b"handoff")
            for name in REQUIRED_SOURCE_PARTITIONS:
                (source / name).write_bytes(name.encode())
            hashes = {
                name[13:15]: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in REQUIRED_SOURCE_PARTITIONS
            }
            record = {
                "type": "source",
                "path": "/private/tmp/old-ephemeral-mirror",
                "digest": "a" * 64,
                "hashes": hashes,
            }
            (source / "polytopes-4d-11-vertices.parquet").write_bytes(b"extra")
            with patch("orientifold_population_preflight._read_zstd_jsonl", return_value=[record]):
                with self.assertRaisesRegex(PopulationPreflightError, "exactly 05..10"):
                    _implementation_source_contract(root, source)
            (source / "polytopes-4d-11-vertices.parquet").unlink()
            (source / REQUIRED_SOURCE_PARTITIONS[0]).write_bytes(b"tampered")
            with patch("orientifold_population_preflight._read_zstd_jsonl", return_value=[record]):
                with self.assertRaisesRegex(PopulationPreflightError, "hash mismatch"):
                    _implementation_source_contract(root, source)

    def test_artifacts_require_explicit_merged_and_gap_hash_entries(self):
        from orientifold_population_preflight import ARTIFACT_SPECS, _verify_artifacts

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CYAxiverse.jl"
            data_dir = root.parent / "data" / ARTIFACT_SPECS[4]["directory"]
            data_dir.mkdir(parents=True)
            merged = data_dir / ARTIFACT_SPECS[4]["merged"]
            gap = data_dir / ARTIFACT_SPECS[4]["gap_analysis"]
            merged.write_bytes(b"merged")
            gap.write_bytes(b"gap")
            with patch(
                "orientifold_population_preflight._read_checksum_manifest",
                return_value=[("a" * 64, merged)],
            ):
                with self.assertRaisesRegex(PopulationPreflightError, "gap_analysis_artifact"):
                    _verify_artifacts(root, 4)


if __name__ == "__main__":
    unittest.main()
