"""Focused regression tests for the orientifold axiverse database bridge.

Covers `load_ledger_accepted_classes`'s two preserved ledger shapes: a
single-shard summary with `class_funnel` at the top level (h11=2/3), and a
merged, sharded artifact with the same list nested at
`terminal_ledger.class_funnel` (h11=4's `h4.merged.json.zst`).
"""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_orientifold_axion_database import load_ledger_accepted_classes


def _write_compressed_ledger(directory: Path, name: str, payload: dict) -> None:
    """Write one zstd-compressed ledger fixture and its SHA256SUMS.txt entry."""
    path = directory / name
    encoded = json.dumps(payload).encode("utf-8")
    subprocess.run(
        ["zstd", "-q", "-f", "-o", str(path)],
        input=encoded,
        check=True,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sums_path = directory / "SHA256SUMS.txt"
    with open(sums_path, "a", encoding="utf-8") as stream:
        stream.write(f"{digest}  {name}\n")


_ACCEPTED_ENTRY = {
    "accepted_for_table_1": True,
    "accepted_witness": {"candidate_id": "c1", "lambda_f": 1, "torus_shift": {
        "numerator": [0, 0, 0, 1], "denominator": 2}},
    "candidate_attempt_count": 4,
    "frst_class_index": 0,
    "frst_hash": "deadbeef",
    "matrix_attempt_count": 1,
    "polytope_id": "lattice-points-sha256:aaaa",
    "polytope_index": 0,
    "polytope_normal_form_id": "normal-form-sha256:bbbb",
    "status_counts": {"accepted_verified_orientifold": 1},
}
_REJECTED_ENTRY = {
    "accepted_for_table_1": False,
    "accepted_witness": None,
    "candidate_attempt_count": 4,
    "frst_class_index": 0,
    "frst_hash": "cafef00d",
    "matrix_attempt_count": 1,
    "polytope_id": "lattice-points-sha256:cccc",
    "polytope_index": 1,
    "polytope_normal_form_id": "normal-form-sha256:dddd",
    "status_counts": {"fixed_point_set_non_smooth": 4},
}


class LoadLedgerAcceptedClassesTests(unittest.TestCase):
    def test_flat_class_funnel_shape(self):
        # h11=2/3 style: a single-shard summary with class_funnel at the
        # top level.
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {
                "schema_version": "test-1.0",
                "class_funnel": [_ACCEPTED_ENTRY, _REJECTED_ENTRY],
            }
            _write_compressed_ledger(directory, "flat.json.zst", payload)
            accepted, ledger, sha256 = load_ledger_accepted_classes(
                str(directory / "flat.json.zst"), str(directory / "SHA256SUMS.txt")
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["polytope_id"], _ACCEPTED_ENTRY["polytope_id"])
            self.assertEqual(ledger["schema_version"], "test-1.0")
            self.assertEqual(len(sha256), 64)

    def test_nested_terminal_ledger_class_funnel_shape(self):
        # h11=4 style: a merged, sharded artifact with class_funnel nested
        # under terminal_ledger.
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {
                "schema_version": "test-merged-1.0",
                "shard_count": 4,
                "terminal_ledger": {
                    "class_count": 2,
                    "class_funnel": [_ACCEPTED_ENTRY, _REJECTED_ENTRY],
                },
            }
            _write_compressed_ledger(directory, "merged.json.zst", payload)
            accepted, ledger, sha256 = load_ledger_accepted_classes(
                str(directory / "merged.json.zst"), str(directory / "SHA256SUMS.txt")
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["polytope_id"], _ACCEPTED_ENTRY["polytope_id"])
            self.assertEqual(ledger["shard_count"], 4)
            self.assertEqual(len(sha256), 64)

    def test_unrecognized_shape_raises(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {"schema_version": "test-1.0", "something_else": []}
            _write_compressed_ledger(directory, "malformed.json.zst", payload)
            with self.assertRaises(RuntimeError):
                load_ledger_accepted_classes(
                    str(directory / "malformed.json.zst"),
                    str(directory / "SHA256SUMS.txt"),
                )

    def test_sha256_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {"class_funnel": [_ACCEPTED_ENTRY]}
            _write_compressed_ledger(directory, "tampered.json.zst", payload)
            # Corrupt the recorded checksum so it no longer matches the file.
            sums_path = directory / "SHA256SUMS.txt"
            sums_path.write_text("0" * 64 + "  tampered.json.zst\n")
            with self.assertRaises(RuntimeError):
                load_ledger_accepted_classes(
                    str(directory / "tampered.json.zst"), str(sums_path)
                )


if __name__ == "__main__":
    unittest.main()
