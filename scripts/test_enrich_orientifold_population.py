"""Focused tests for the bounded orientifold population enrichment runner."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import enrich_orientifold_population as enrichment


class _FakePoly:
    pass


class _FakeTri:
    def points(self):
        return np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int64)

    def simplices(self, as_indices=False):
        if as_indices:
            return np.asarray([[0, 1, 2]], dtype=np.int64)
        return np.asarray([[[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]], dtype=np.int64)


def _source():
    return {
        "dataset": "calabi-yau-data/polytopes-4d",
        "dataset_revision": "test",
        "directory": "/private/tmp/source",
        "directory_sha256": enrichment.HISTORICAL_SOURCE_DIGEST,
        "partitions": [{"partition": "polytopes-4d-05-vertices.parquet", "sha256": "a" * 64}],
    }


def _entry():
    witness = {
        "matrix_id": "m" * 64,
        "lambda_f": 1,
        "torus_shift": {"numerator": [0, 1, 0, 0], "denominator": 2},
    }
    witness["candidate_id"] = enrichment.stable_hash(
        [witness["matrix_id"], tuple([0, 2, 0, 0]), witness["lambda_f"]]
    )
    return {
        "accepted_for_table_1": True,
        "frst_class_index": 3,
        "frst_hash": "f" * 64,
        "polytope_id": "lattice-points-sha256:" + "p" * 64,
        "accepted_witness": witness,
    }


class EnrichmentTests(unittest.TestCase):
    def test_nonzero_binary_shift_uses_source_projected_numerator(self):
        entry = _entry()
        raw = {
            "record": {
                "accepted_witness": entry["accepted_witness"],
                "torus_shift_binary_source": [0, 1, 0, 0],
                "lattice_matrix": np.eye(4, dtype=np.int64).tolist(),
            }
        }
        self.assertEqual(
            enrichment.validate_binary_shift(entry, raw),
            {"binary": [0, 1, 0, 0], "projected": [0, 2, 0, 0]},
        )

    def test_authoritative_selection_is_sorted_and_rejects_nonaccepted(self):
        merged = {
            "requested_h11": 4,
            "terminal_ledger": {"class_funnel": [_entry(), {"accepted_for_table_1": False}]},
        }
        rows = enrichment.authoritative_candidates(merged, 4)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1]["accepted_witness"]["lambda_f"], 1)

    def test_source_join_emits_input_identity_certificate_without_output_evidence(self):
        row = {
            "partition": "polytopes-4d-05-vertices.parquet",
            "partition_sha256": "a" * 64,
            "source_row": 12,
            "row_metadata": {"h12": 4},
            "poly": _FakePoly(),
            "physical_h11": 4,
            "global_points": [[0, 0, 0, 0]],
            "polytope_id": _entry()["polytope_id"],
            "triangulation_index": 0,
        }
        with mock.patch.object(enrichment, "locate_source", return_value=row), mock.patch.object(
            enrichment, "match_frst", return_value=(_FakeTri(), 0)
        ), mock.patch.object(
            enrichment, "match_matrix", return_value=(np.eye(4, dtype=np.int64), "m" * 64)
        ):
            entry = _entry()
            raw = {
                "path": "/private/tmp/raw.jsonl.zst",
                "record": {
                    "accepted_witness": entry["accepted_witness"],
                    "torus_shift_binary_source": [0, 1, 0, 0],
                    "lattice_matrix": np.eye(4, dtype=np.int64).tolist(),
                },
            }
            result = enrichment.enrich_candidate(_source(), 4, entry, raw)
        self.assertEqual(result["terminal_status"], "input_identity_verified")
        self.assertIsInstance(result["source_record"], dict)
        self.assertEqual(
            result["mpcp_certificate"]["certificate_schema_version"],
            enrichment.INPUT_CERTIFICATE_SCHEMA_VERSION,
        )
        self.assertNotIn("evidence", result["mpcp_certificate"])
        self.assertEqual(result["failure_categories"], [])

    def test_atomic_checkpoint_is_zstd_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl.zst"
            enrichment.zstd_jsonl_write_atomic(path, [{"record_type": "header"}, {"record_type": "row"}])
            self.assertEqual(enrichment.zstd_jsonl_read(path), [{"record_type": "header"}, {"record_type": "row"}])
            self.assertEqual(list(Path(directory).glob(".*")), [])

    def test_source_index_is_single_pass_and_preserves_favorable_route(self):
        rows = [
            {"polytope_id": "target", "favorable_index": 7, "physical_h11": 4},
            {"polytope_id": "other", "favorable_index": 8, "physical_h11": 4},
        ]
        with mock.patch.object(enrichment, "_iter_source_rows", return_value=iter(rows)):
            indexed = enrichment.index_source_rows(_source(), 4, {"target": 7})
        self.assertEqual(indexed["target"]["favorable_index"], 7)

    def test_source_index_rejects_favorable_route_mismatch(self):
        rows = [{"polytope_id": "target", "favorable_index": 8, "physical_h11": 4}]
        with mock.patch.object(enrichment, "_iter_source_rows", return_value=iter(rows)):
            with self.assertRaises(enrichment.EnrichmentError):
                enrichment.index_source_rows(_source(), 4, {"target": 7})


if __name__ == "__main__":
    unittest.main()
