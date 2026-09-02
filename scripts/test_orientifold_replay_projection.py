"""Focused tests for the bounded h11=4 status projection contract."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orientifold_replay_projection as projection
from orientifold_replay_projection import (
    ProjectionError,
    canonical_projection_bytes,
    canonical_projection_sha256,
    compare_replay_artifacts,
)


class CanonicalProjectionTests(unittest.TestCase):
    def test_serialization_is_sorted_compact_utf8_jsonl_with_final_lf(self):
        rows = [
            {"row_identity": "b", "terminal_status": "h21_plus_nonzero"},
            {"row_identity": "a", "terminal_status": "accepted_exact_trilayer_action"},
        ]
        expected = (
            b'{"row_identity":"a","terminal_status":"accepted_exact_trilayer_action"}\n'
            b'{"row_identity":"b","terminal_status":"h21_plus_nonzero"}\n'
        )
        self.assertEqual(canonical_projection_bytes(rows), expected)
        self.assertEqual(canonical_projection_sha256(rows), hashlib.sha256(expected).hexdigest())

    def test_projection_rejects_duplicate_identity_and_missing_status(self):
        with self.assertRaisesRegex(ProjectionError, "duplicate"):
            canonical_projection_bytes(
                [
                    {"row_identity": "same", "terminal_status": "first"},
                    {"row_identity": "same", "terminal_status": "second"},
                ]
            )
        with self.assertRaisesRegex(ProjectionError, "terminal_status"):
            canonical_projection_bytes([{"row_identity": "missing-status"}])


class ArtifactComparisonTests(unittest.TestCase):
    @staticmethod
    def _artifact(
        path: Path,
        rows: list[dict[str, object]],
        *,
        code_hash: str,
        include_summary: bool = True,
        summary: dict[str, object] | None = None,
        extra_summaries: list[dict[str, object]] | None = None,
    ) -> None:
        config = {
            "requested_h11": 4,
            "max_rows": len(rows),
            "workers": 1,
            "shard_count": 1,
            "shard_index": 0,
            "source_code_sha256": code_hash,
        }
        payload = [
            {
                "record_type": "header",
                "config": config,
                "config_sha256": "config-digest",
            },
            *[{"record_type": "row", **row} for row in rows],
        ]
        if include_summary:
            summary_record = (
                {
                    "status": "completed",
                    "rows_evaluated": len(rows),
                    "database_writes": 0,
                    "duplicate_count": 0,
                }
                if summary is None
                else summary
            )
            payload.append({"record_type": "summary", **summary_record})
        for extra_summary in extra_summaries or []:
            payload.append({"record_type": "summary", **extra_summary})
        raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in payload)
        raw_path = path.with_suffix(".jsonl")
        raw_path.write_text(raw, encoding="utf-8", newline="\n")
        import subprocess

        subprocess.run(["zstd", "-19", "-q", "-f", "-o", str(path), str(raw_path)], check=True)
        raw_path.unlink()

    @staticmethod
    def _small_contract():
        return patch.multiple(
            projection,
            EXPECTED_SOURCE_ROWS=2,
            EXPECTED_AFFECTED_ROWS=1,
            EXPECTED_UNAFFECTED_ROWS=1,
            EXPECTED_H11=4,
        )

    def _assert_cli_failure(
        self, source: Path, repaired: Path, *, output: Path | None = None
    ) -> None:
        argv = [
            "orientifold_replay_projection.py",
            "--source",
            str(source),
            "--repaired",
            str(repaired),
        ]
        if output is not None:
            argv.extend(("--output", str(output)))
        with self._small_contract(), patch.object(
            projection,
            "IMMUTABLE_SOURCE_ARTIFACT_SHA256",
            projection._sha256_file(source),
        ), patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as raised:
                projection.main()
        self.assertNotEqual(raised.exception.code, 0)

    def test_comparison_includes_only_the_1088_baseline_unaffected_rows(self):
        # Keep the fixture small while exercising the exact selection logic.
        source_rows = [
            {"row_identity": f"{index:04d}", "terminal_status": "smoothness_verification_unavailable" if index == 0 else "h21_plus_nonzero"}
            for index in range(2)
        ]
        repaired_rows = [dict(row) for row in source_rows]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            self._artifact(source, source_rows, code_hash="source")
            self._artifact(repaired, repaired_rows, code_hash="repaired")
            with self._small_contract():
                report = compare_replay_artifacts(
                    source,
                    repaired,
                    expected_source_sha256=None,
                )
            self.assertEqual(report["projection"]["source_sha256"], report["projection"]["repaired_sha256"])
            self.assertTrue(report["projection"]["status_projection_matches"])
            self.assertEqual(report["contract"]["unaffected_rows"], 1)
            self.assertNotIn("rows_by_identity", report["source_artifact"])
            self.assertNotIn("rows_by_identity", report["repaired_artifact"])
            self.assertGreater(report["resource"]["peak_rss_bytes"], 0)

    def test_comparison_rejects_a_changed_unaffected_status(self):
        source_rows = [
            {"row_identity": f"{index:04d}", "terminal_status": "smoothness_verification_unavailable" if index == 0 else "h21_plus_nonzero"}
            for index in range(2)
        ]
        repaired_rows = [dict(row) for row in source_rows]
        repaired_rows[1]["terminal_status"] = "accepted_exact_trilayer_action"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            self._artifact(source, source_rows, code_hash="source")
            self._artifact(repaired, repaired_rows, code_hash="repaired")
            with self._small_contract():
                report = compare_replay_artifacts(source, repaired, expected_source_sha256=None)
            self.assertFalse(report["projection"]["status_projection_matches"])
            self.assertNotEqual(report["projection"]["source_sha256"], report["projection"]["repaired_sha256"])

            self._assert_cli_failure(source, repaired)

    def test_comparison_requires_one_completed_zero_write_summary(self):
        rows = [
            {"row_identity": "0000", "terminal_status": "smoothness_verification_unavailable"},
            {"row_identity": "0001", "terminal_status": "h21_plus_nonzero"},
        ]
        invalid_summaries = [
            None,
            {"status": "running", "rows_evaluated": 2, "database_writes": 0, "duplicate_count": 0},
            {"status": "completed", "rows_evaluated": 1, "database_writes": 0, "duplicate_count": 0},
            {"status": "completed", "rows_evaluated": 2, "database_writes": 1, "duplicate_count": 0},
            {"status": "completed", "rows_evaluated": 2, "database_writes": 0, "duplicate_count": 1},
        ]
        for summary in invalid_summaries:
            with self.subTest(summary=summary), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "source.jsonl.zst"
                repaired = Path(temporary) / "repaired.jsonl.zst"
                self._artifact(
                    source,
                    rows,
                    code_hash="source",
                    include_summary=summary is not None,
                    summary=summary,
                )
                self._artifact(repaired, rows, code_hash="repaired")
                with self._small_contract(), self.assertRaisesRegex(
                    ProjectionError, "summary"
                ):
                    compare_replay_artifacts(source, repaired, expected_source_sha256=None)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            self._artifact(source, rows, code_hash="source", extra_summaries=[{"status": "completed"}])
            self._artifact(repaired, rows, code_hash="repaired")
            with self._small_contract(), self.assertRaisesRegex(
                ProjectionError, "exactly one summary"
            ):
                compare_replay_artifacts(source, repaired, expected_source_sha256=None)

    def test_comparison_rejects_corrupt_and_truncated_zstd(self):
        rows = [
            {"row_identity": "0000", "terminal_status": "smoothness_verification_unavailable"},
            {"row_identity": "0001", "terminal_status": "h21_plus_nonzero"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            source.write_bytes(b"not a zstd stream")
            self._artifact(repaired, rows, code_hash="repaired")
            with self._small_contract(), self.assertRaisesRegex(
                ProjectionError, "cannot read replay artifact"
            ):
                compare_replay_artifacts(source, repaired, expected_source_sha256=None)

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            self._artifact(source, rows, code_hash="source")
            self._artifact(repaired, rows, code_hash="repaired")
            source.write_bytes(source.read_bytes()[:-8])
            with self._small_contract(), self.assertRaisesRegex(
                ProjectionError, "cannot read replay artifact"
            ):
                compare_replay_artifacts(source, repaired, expected_source_sha256=None)

    def test_comparison_rejects_wrong_source_sha256(self):
        rows = [
            {"row_identity": "0000", "terminal_status": "smoothness_verification_unavailable"},
            {"row_identity": "0001", "terminal_status": "h21_plus_nonzero"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            self._artifact(source, rows, code_hash="source")
            self._artifact(repaired, rows, code_hash="repaired")
            with self._small_contract(), self.assertRaisesRegex(
                ProjectionError, "SHA-256 mismatch"
            ):
                compare_replay_artifacts(source, repaired, expected_source_sha256="0" * 64)

    def test_comparison_rejects_duplicate_and_changed_row_identities(self):
        source_rows = [
            {"row_identity": "0000", "terminal_status": "smoothness_verification_unavailable"},
            {"row_identity": "0000", "terminal_status": "h21_plus_nonzero"},
        ]
        repaired_rows = [
            {"row_identity": "0000", "terminal_status": "smoothness_verification_unavailable"},
            {"row_identity": "0002", "terminal_status": "h21_plus_nonzero"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            self._artifact(source, source_rows, code_hash="source")
            self._artifact(repaired, repaired_rows, code_hash="repaired")
            with self._small_contract(), self.assertRaisesRegex(ProjectionError, "duplicate"):
                compare_replay_artifacts(source, repaired, expected_source_sha256=None)

        source_rows[1]["row_identity"] = "0001"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            self._artifact(source, source_rows, code_hash="source")
            self._artifact(repaired, repaired_rows, code_hash="repaired")
            with self._small_contract(), self.assertRaisesRegex(
                ProjectionError, "row-identity sets differ"
            ):
                compare_replay_artifacts(source, repaired, expected_source_sha256=None)

    def test_cli_rejects_output_overwrite(self):
        rows = [
            {"row_identity": "0000", "terminal_status": "smoothness_verification_unavailable"},
            {"row_identity": "0001", "terminal_status": "h21_plus_nonzero"},
        ]
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            source = Path(temporary) / "source.jsonl.zst"
            repaired = Path(temporary) / "repaired.jsonl.zst"
            output = Path(temporary) / "report.json.zst"
            self._artifact(source, rows, code_hash="source")
            self._artifact(repaired, rows, code_hash="repaired")
            output.write_bytes(b"existing report")
            self._assert_cli_failure(source, repaired, output=output)
            self.assertEqual(output.read_bytes(), b"existing report")


if __name__ == "__main__":
    unittest.main()
