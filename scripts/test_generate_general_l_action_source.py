"""Focused tests for immutable general-L source generation."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import generate_general_l_action_source as generator
from run_general_l_action_replacement_bounded import (
    account_terminal_rows,
    action_digest,
    compare_witnesses,
    load_json,
    load_jsonl,
    refingerprint_manifest,
)


class _FakePolytope:
    def points(self):
        return np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=int,
        )

    def normal_form(self):
        return self.points()


class _FakeTriangulation:
    def __init__(self, marker):
        self.marker = marker

    def simplices(self):
        return np.array(
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4]],
            dtype=int,
        ) + self.marker


def _candidate(poly, triangulation, topology, *, record_sink):
    del poly, topology
    action = {
        "lattice_matrix": np.eye(4, dtype=int).tolist(),
        "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1},
        "lambda_f": 1,
    }
    record = {
        "schema_version": generator.CANDIDATE_SCHEMA,
        "record_kind": "candidate",
        "candidate_id": f"candidate-{triangulation.marker}",
        "matrix_id": f"matrix-{triangulation.marker}",
        **action,
        "terminal_status": "accepted_verified_orientifold",
        "terminal_reason": None,
        "terminal_reason_code": "accepted_verified_orientifold",
        "h11_plus": 2,
        "h11_minus": 0,
        "fixed_point_components": [],
        "fixed_point_set": {"status": "verified"},
        "smoothness": {"status": "smooth"},
    }
    record_sink(record)
    return [record]


def _exact(poly, triangulation, witness):
    del poly, triangulation
    return {
        "schema_version": "cyaxiverse-exact-action-hodge-evidence-3.0",
        "status": "validated",
        "action_digest": action_digest(witness),
        "h11_plus": 2,
        "h11_minus": 0,
        "h21_plus": 0,
        "h21_minus": 132,
        "chi_fixed_locus": 272,
        "chi_x": -260,
    }


def _emits_then_fails(poly, triangulation, topology, *, record_sink):
    _candidate(poly, triangulation, topology, record_sink=record_sink)
    raise RuntimeError("fixture enumeration failure")


class GenerateGeneralLActionSourceTests(unittest.TestCase):
    def test_generation_orders_classes_and_preserves_witnesses(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(
                    f"partition-{partition}".encode("ascii")
                )
            calls = []

            def loader(path, *, h11, limit, favorable, partitions):
                calls.append((path, h11, limit, favorable, tuple(partitions)))
                return [
                    (
                        _FakePolytope(),
                        {
                            "parquet_file": str(
                                parquet / "polytopes-4d-05-vertices.parquet"
                            ),
                            "row_index": 17,
                            "physical_h11": h11,
                            "physical_h21": h11 + 10,
                            "mirror_h11": h11 + 10,
                            "mirror_h12": h11,
                        },
                    )
                ]

            def classes(poly):
                del poly
                return [_FakeTriangulation(1), _FakeTriangulation(0)]

            with patch.object(
                generator,
                "_partition_manifest",
                wraps=generator._partition_manifest,
            ), patch.object(generator, "_topology_for_class", return_value={}):
                result = generator.generate_source_rows(
                    parquet,
                    root / "published",
                    limit=1,
                    repository_root=Path(__file__).resolve().parent.parent,
                    loader=loader,
                    frst_classifier=classes,
                    candidate_enumerator=_candidate,
                    exact_diagnostic=_exact,
                )

            self.assertEqual(
                [call[-1] for call in calls],
                [tuple(range(5, 14))] * 4,
            )
            self.assertEqual(
                result["source_partition_manifest"]["partitions"][0]["partition"],
                5,
            )
            output = root / "published"
            source_rows, blank, malformed = load_jsonl(
                output / "h11-002.source-rows.jsonl.zst"
            )
            ledger_rows, _, _ = load_jsonl(
                output / "h11-002.terminal-ledger.jsonl.zst"
            )
            self.assertEqual((blank, malformed), (0, 0))
            self.assertEqual([row["frst_class_index"] for row in source_rows], [0, 1])
            self.assertEqual(
                compare_witnesses(source_rows, ledger_rows)["equal"], True
            )
            self.assertEqual(account_terminal_rows(source_rows)["terminal_rows"], 2)
            self.assertEqual(source_rows[0]["source_partition"], "polytopes-4d-05-vertices.parquet")
            self.assertEqual(source_rows[0]["source_provenance"]["physical_h11"], 2)
            self.assertEqual(source_rows[0]["source_provenance"]["physical_h21"], 12)

            input_manifest = load_json(output / "input-manifest.json.zst")
            self.assertEqual(input_manifest["schema"], "cyaxiverse-general-l-action-replacement-input-1.0")
            self.assertEqual(len(input_manifest["inputs"]), 8)
            self.assertEqual(input_manifest["run_scope"], "pilot")
            self.assertIsNone(input_manifest["population_completion"]["expected"])
            self.assertIn("scripts/reproduce_fuzzy_axions_h11_4.py", input_manifest["source_file_digests"])
            refingerprint_manifest(
                input_manifest,
                repo_root=Path(__file__).resolve().parent.parent,
            )
            self.assertTrue(
                all(
                    entry["source_commit"] == input_manifest["source_commit"]
                    and entry["output_root"] == str(output.resolve())
                    for entry in input_manifest["inputs"]
                )
            )
            self.assertEqual(input_manifest["production_gate"], "not_validated")
            self.assertEqual(input_manifest["scale_status"], "not_applicable")
            with self.assertRaises(FileExistsError):
                generator.generate_source_rows(
                    parquet,
                    output,
                    limit=1,
                    repository_root=Path(__file__).resolve().parent.parent,
                    loader=loader,
                    frst_classifier=classes,
                    candidate_enumerator=_candidate,
                    exact_diagnostic=_exact,
                )

    def test_missing_hodge_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(b"x")

            def loader(path, *, h11, limit, favorable, partitions):
                del path, h11, limit, favorable, partitions
                return [
                    (_FakePolytope(), {"parquet_file": "polytopes-4d-05-vertices.parquet", "row_index": 0})
                ]

            with self.assertRaisesRegex(RuntimeError, "Hodge-label mapping"):
                generator.generate_source_rows(
                    parquet,
                    root / "published",
                    limit=1,
                    repository_root=Path(__file__).resolve().parent.parent,
                    loader=loader,
                    frst_classifier=lambda poly: [],
                    candidate_enumerator=_candidate,
                    exact_diagnostic=_exact,
                )

    def test_emitted_terminal_rows_survive_class_failure(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(b"x")

            def loader(path, *, h11, limit, favorable, partitions):
                del path, limit, favorable, partitions
                return [
                    (
                        _FakePolytope(),
                        {
                            "parquet_file": str(
                                parquet / "polytopes-4d-05-vertices.parquet"
                            ),
                            "row_index": h11,
                            "physical_h11": h11,
                            "physical_h21": h11 + 10,
                            "mirror_h11": h11 + 10,
                            "mirror_h12": h11,
                        },
                    )
                ]

            with patch.object(generator, "_topology_for_class", return_value={}):
                generator.generate_source_rows(
                    parquet,
                    root / "published",
                    limit=1,
                    repository_root=Path(__file__).resolve().parent.parent,
                    loader=loader,
                    frst_classifier=lambda poly: [_FakeTriangulation(0)],
                    candidate_enumerator=_emits_then_fails,
                    exact_diagnostic=_exact,
                )
            rows, _, _ = load_jsonl(root / "published" / "h11-002.source-rows.jsonl.zst")
            counts = account_terminal_rows(rows)
            self.assertEqual(counts["candidate_action_attempts"], 1)
            self.assertEqual(counts["search_summary_rows"], 1)
            self.assertEqual(counts["terminal_rows"], 2)
            self.assertIn(
                "numerical_geometry_failure",
                {row["terminal_status"] for row in rows},
            )

    def test_returned_candidate_must_have_been_sent_to_sink(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(b"x")

            def loader(path, *, h11, limit, favorable, partitions):
                del path, limit, favorable, partitions
                return [
                    (
                        _FakePolytope(),
                        {
                            "parquet_file": str(parquet / "polytopes-4d-05-vertices.parquet"),
                            "row_index": h11,
                            "physical_h11": h11,
                            "physical_h21": h11 + 10,
                            "mirror_h11": h11 + 10,
                            "mirror_h12": h11,
                        },
                    )
                ]

            def bypass_sink(poly, triangulation, topology, *, record_sink):
                del poly, topology
                record = _candidate(
                    _FakePolytope(), triangulation, {}, record_sink=lambda row: None
                )[0]
                return [record]

            with patch.object(generator, "_topology_for_class", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "not emitted through record_sink"):
                    generator.generate_source_rows(
                        parquet,
                        root / "published",
                        limit=1,
                        repository_root=Path(__file__).resolve().parent.parent,
                        loader=loader,
                        frst_classifier=lambda poly: [_FakeTriangulation(0)],
                        candidate_enumerator=bypass_sink,
                        exact_diagnostic=_exact,
                    )

    def test_complete_run_enforces_owner_approved_population_counts(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(b"x")

            def loader(path, *, h11, limit, favorable, partitions):
                del limit, favorable, partitions
                return [
                    (
                        _FakePolytope(),
                        {
                            "parquet_file": str(parquet / "polytopes-4d-05-vertices.parquet"),
                            "row_index": h11,
                            "physical_h11": h11,
                            "physical_h21": h11 + 10,
                            "mirror_h11": h11 + 10,
                            "mirror_h12": h11,
                        },
                    )
                ]

            with patch.object(generator, "_topology_for_class", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "complete source count mismatch"):
                    generator.generate_source_rows(
                        parquet,
                        root / "published",
                        repository_root=Path(__file__).resolve().parent.parent,
                        loader=loader,
                        frst_classifier=lambda poly: [_FakeTriangulation(0)],
                        candidate_enumerator=_candidate,
                        exact_diagnostic=_exact,
                    )

    def test_complete_run_rejects_pseudo_frst_failure_class(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(b"x")

            def loader(path, *, h11, limit, favorable, partitions):
                del limit, favorable, partitions
                return [
                    (
                        _FakePolytope(),
                        {
                            "parquet_file": str(parquet / "polytopes-4d-05-vertices.parquet"),
                            "row_index": h11,
                            "physical_h11": h11,
                            "physical_h21": h11 + 10,
                            "mirror_h11": h11 + 10,
                            "mirror_h12": h11,
                        },
                    )
                ]

            def failed_classifier(poly):
                del poly
                raise RuntimeError("FRST fixture failure")

            with patch.object(generator, "_topology_for_class", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "complete source count mismatch"):
                    generator.generate_source_rows(
                        parquet,
                        root / "published",
                        repository_root=Path(__file__).resolve().parent.parent,
                        loader=loader,
                        frst_classifier=failed_classifier,
                        candidate_enumerator=_candidate,
                        exact_diagnostic=_exact,
                    )

    def test_rss_ceiling_fails_closed_before_loading_source(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(b"x")
            loaded = {"value": False}

            def loader(path, *, h11, limit, favorable, partitions):
                del path, h11, limit, favorable, partitions
                loaded["value"] = True
                return []

            with patch.object(
                generator,
                "_current_rss_bytes",
                return_value=generator.GLOBAL_LIMITS["max_rss_bytes"] + 1,
            ):
                with self.assertRaisesRegex(RuntimeError, "RSS ceiling"):
                    generator.generate_source_rows(
                        parquet,
                        root / "published",
                        limit=1,
                        repository_root=Path(__file__).resolve().parent.parent,
                        loader=loader,
                        frst_classifier=lambda poly: [],
                        candidate_enumerator=_candidate,
                        exact_diagnostic=_exact,
                    )
            self.assertFalse(loaded["value"])

    def test_checkpoint_segments_resume_without_reemitting_completed_class(self):
        with tempfile.TemporaryDirectory(prefix="cyax-general-l-source-") as directory:
            root = Path(directory)
            parquet = root / "parquet"
            parquet.mkdir()
            for partition in range(5, 14):
                (parquet / f"polytopes-4d-{partition:02d}-vertices.parquet").write_bytes(b"x")
            checkpoint = root / "checkpoints"
            output = root / "published"

            def loader(path, *, h11, limit, favorable, partitions):
                del path, limit, favorable, partitions
                return [
                    (
                        _FakePolytope(),
                        {
                            "parquet_file": str(parquet / "polytopes-4d-05-vertices.parquet"),
                            "row_index": h11,
                            "physical_h11": h11,
                            "physical_h21": h11 + 10,
                            "mirror_h11": h11 + 10,
                            "mirror_h12": h11,
                        },
                    )
                ]

            calls = {"count": 0}

            def interrupting(poly, triangulation, topology, *, record_sink):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise KeyboardInterrupt("simulated interruption")
                return _candidate(poly, triangulation, topology, record_sink=record_sink)

            with patch.object(generator, "_topology_for_class", return_value={}):
                with self.assertRaises(KeyboardInterrupt):
                    generator.generate_source_rows(
                        parquet, output, limit=1, checkpoint_root=checkpoint,
                        repository_root=Path(__file__).resolve().parent.parent,
                        loader=loader,
                        frst_classifier=lambda poly: [_FakeTriangulation(0)],
                        candidate_enumerator=interrupting,
                        exact_diagnostic=_exact,
                    )
                self.assertTrue(list(checkpoint.rglob("*.source.jsonl")))
                calls["count"] = 0

                def resuming(poly, triangulation, topology, *, record_sink):
                    calls["count"] += 1
                    return _candidate(poly, triangulation, topology, record_sink=record_sink)

                generator.generate_source_rows(
                    parquet, output, limit=1, checkpoint_root=checkpoint,
                    repository_root=Path(__file__).resolve().parent.parent,
                    loader=loader,
                    frst_classifier=lambda poly: [_FakeTriangulation(0)],
                    candidate_enumerator=resuming,
                    exact_diagnostic=_exact,
                )
                self.assertEqual(calls["count"], 3)
            self.assertEqual((root / "published" / "h11-002.source-rows.jsonl.zst").is_file(), True)
            ledger_segment = next(checkpoint.rglob("*.ledger.jsonl"))
            with ledger_segment.open("ab") as stream:
                stream.write(b"tampered\n")
            source_segment = next(
                checkpoint.rglob("*.source.jsonl")
            )
            metadata_segment = next(
                checkpoint.rglob("*.metadata.json.zst")
            )
            metadata = generator.load_json(metadata_segment)
            with self.assertRaisesRegex(RuntimeError, "checkpoint segment"):
                generator._verify_checkpoint_segment(
                    source_segment,
                    ledger_segment,
                    metadata_segment,
                    h11=metadata["h11"],
                    source_row=metadata["source_row"],
                    frst_class_index=metadata["frst_class_index"],
                    frst_hash=metadata["frst_hash"],
                    polytope_id=metadata["polytope_id"],
                    source_commit=metadata["source_commit"],
                )
            with self.assertRaisesRegex(RuntimeError, "source_commit"):
                generator._verify_checkpoint_segment(
                    source_segment,
                    ledger_segment,
                    metadata_segment,
                    h11=metadata["h11"],
                    source_row=metadata["source_row"],
                    frst_class_index=metadata["frst_class_index"],
                    frst_hash=metadata["frst_hash"],
                    polytope_id=metadata["polytope_id"],
                    source_commit="different-bound-source-commit",
                )


if __name__ == "__main__":
    unittest.main()
