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


if __name__ == "__main__":
    unittest.main()
