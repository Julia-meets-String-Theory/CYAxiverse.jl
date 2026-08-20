"""Test the bounded inherited-orientifold terminal ledger."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import analyze_fuzzy_axions_orientifold_gap as gap
from orientifold_terminal_ledger import (
    LEDGER_SCHEMA_VERSION,
    TerminalLedgerError,
    TerminalLedgerWriter,
    read_terminal_ledger,
    validate_source_provenance,
    verify_terminal_ledger,
)


def _provenance(dirty=False):
    return {
        "source_commit": "abc123",
        "git_dirty": dirty,
        "working_tree_identity": (
            {"diff_sha256": "diff-digest"} if dirty else None
        ),
        "runtime_versions": {"python": "3.13"},
        "input_partition_manifest": {
            "status": "complete",
            "version": "fixture",
        },
    }


def _row(candidate_id, status, *, lambda_f=1, class_index=0):
    return {
        "polytope_index": 0,
        "frst_class_index": class_index,
        "polytope_id": "polytope-0",
        "frst_hash": f"frst-{class_index}",
        "matrix_id": "matrix-0",
        "candidate_id": candidate_id,
        "lambda_f": lambda_f,
        "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 2},
        "h11_plus": 2,
        "h11_minus": 0,
        "fixed_point_components": [],
        "fixed_point_set": {"description": "fixture"},
        "smoothness": {"verdict": "smooth"},
        "terminal_status": status,
        "terminal_reason_code": status,
    }


class TerminalLedgerTests(unittest.TestCase):
    def test_dirty_source_requires_cryptographic_identity(self):
        provenance = _provenance(dirty=True)
        provenance.pop("working_tree_identity")
        with self.assertRaisesRegex(TerminalLedgerError, "diff_sha256"):
            validate_source_provenance(provenance)

    def test_stream_round_trip_hash_and_class_funnel(self):
        with tempfile.TemporaryDirectory(prefix="cyax-terminal-ledger-") as directory:
            path = Path(directory) / "ledger.jsonl"
            writer = TerminalLedgerWriter(
                path,
                provenance=_provenance(),
                metadata={"fixture": True},
            )
            writer.write(
                {
                    "polytope_index": 0,
                    "frst_class_index": 0,
                    "polytope_id": "polytope-0",
                    "frst_hash": "frst-0",
                    "matrix_id": "matrix-0",
                    "candidate_id": "matrix-0",
                    "lambda_f": None,
                    "torus_shift": None,
                    "h11_parity": {
                        "status": "unavailable",
                        "reason": "matrix validation failed before parity extraction",
                    },
                    "fixed_component_evidence": {
                        "status": "not_evaluated",
                        "reason": "matrix validation failed before fixed-component evaluation",
                    },
                    "terminal_status": "matrix_validation_passed",
                    "terminal_reason_code": "matrix_validation_passed",
                }
            )
            writer.write(_row("candidate-a", "accepted_verified_orientifold"))
            writer.write(
                _row(
                    "candidate-b",
                    "accepted_verified_orientifold",
                    lambda_f=0,
                )
            )
            writer.write(
                _row(
                    "candidate-c",
                    "smoothness_verification_unavailable",
                    class_index=1,
                )
            )
            summary = writer.close()

            rows = read_terminal_ledger(path)
            self.assertEqual(len(rows), 4)
            self.assertEqual(summary["schema_version"], LEDGER_SCHEMA_VERSION)
            self.assertEqual(summary["record_count"], 4)
            self.assertEqual(summary["class_count"], 2)
            self.assertEqual(summary["sidecar_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(summary["class_funnel"][0]["accepted_for_table_1"])
            self.assertFalse(summary["class_funnel"][1]["accepted_for_table_1"])
            verify_terminal_ledger(path, summary["sidecar_sha256"])

            summary_path = Path(f"{path}.summary.json")
            self.assertEqual(json.loads(summary_path.read_text())["record_count"], 4)
            with self.assertRaises(FileExistsError):
                TerminalLedgerWriter(path, provenance=_provenance())

    def test_missing_required_field_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cyax-terminal-ledger-") as directory:
            writer = TerminalLedgerWriter(
                Path(directory) / "ledger.jsonl",
                provenance=_provenance(),
            )
            with self.assertRaisesRegex(TerminalLedgerError, "frst_hash"):
                writer.write({"terminal_status": "matrix_validation_passed"})
            writer.abort()

    def test_missing_or_null_scientific_evidence_is_rejected(self):
        for field, value in (
            ("h11_parity", None),
            ("fixed_component_evidence", None),
            ("h11_parity", {"h11_plus": 2, "h11_minus": None}),
            (
                "fixed_component_evidence",
                {
                    "fixed_point_components": [],
                    "fixed_point_set": None,
                    "smoothness": {"status": "smooth"},
                },
            ),
        ):
            with self.subTest(field=field, value=value):
                row = _row("missing-evidence", "fixed_point_set_non_smooth")
                row.pop("h11_plus", None)
                row.pop("h11_minus", None)
                row.pop("fixed_point_components", None)
                row.pop("fixed_point_set", None)
                row.pop("smoothness", None)
                row[field] = value
                with tempfile.TemporaryDirectory(prefix="cyax-terminal-ledger-") as directory:
                    writer = TerminalLedgerWriter(
                        Path(directory) / "ledger.jsonl",
                        provenance=_provenance(),
                    )
                    with self.assertRaises(TerminalLedgerError):
                        writer.write(row)
                    writer.abort()

    def test_analyzer_replaces_unclassified_classes_with_terminal_categories(self):
        with tempfile.TemporaryDirectory(prefix="cyax-terminal-ledger-") as directory:
            directory_path = Path(directory)
            path = directory_path / "ledger.jsonl"
            writer = TerminalLedgerWriter(path, provenance=_provenance())
            writer.write(
                {
                    "polytope_index": 0,
                    "frst_class_index": 0,
                    "polytope_id": "polytope-0",
                    "frst_hash": "frst-0",
                    "matrix_id": "matrix-0",
                    "candidate_id": "matrix-0",
                    "lambda_f": None,
                    "torus_shift": None,
                    "h11_parity": {
                        "status": "unavailable",
                        "reason": "matrix validation failed before parity extraction",
                    },
                    "fixed_component_evidence": {
                        "status": "not_evaluated",
                        "reason": "matrix validation failed before fixed-component evaluation",
                    },
                    "terminal_status": "matrix_validation_passed",
                    "terminal_reason_code": "matrix_validation_passed",
                }
            )
            writer.write(_row("accepted", "accepted_verified_orientifold"))
            writer.write(
                _row(
                    "unavailable",
                    "smoothness_verification_unavailable",
                    class_index=1,
                )
            )
            summary = writer.close()
            data = {
                "details": [{"polytope_index": 0, "frst_class_count": 2}],
                "terminal_ledger": summary,
            }

            result = gap._terminal_ledger_audit(data, directory_path / "artifact.json")

            self.assertEqual(result["terminal_record_count"], 3)
            self.assertEqual(
                result["category_counts"],
                {
                    "certified_inherited": 1,
                    "unaccepted_exhaustive_with_unavailable_evidence": 1,
                },
            )


if __name__ == "__main__":
    unittest.main()
