"""Focused synthetic contract tests for the bounded general-L driver."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from run_general_l_action_replacement_bounded import (
    ContractError, action_digest, account_terminal_rows, atomic_create,
    bounded_gate, build_witness_record, canonical_bytes, compare_witnesses,
    finalize_terminal, hodge_split_from_euler, matrix_digest, validate_action,
    execute_bounded, terminal_identity, terminal_digest, validate_resume,
    repository_revision, CAPS, _witness_rows, load_json, refingerprint_manifest,
    input_manifest_digest, next_class, CANDIDATE_SCHEMA, GLOBAL_LIMITS,
    REQUIRED_SOURCE_FILES, create_approval_bound_manifest,
    prepare_bounded_manifest, main,
)


def _row(**changes):
    row = {"schema_version": CANDIDATE_SCHEMA, "polytope_id": "p1", "frst_hash": "f1", "frst_class_index": 0, "candidate_id": "c1",
           "action_digest": None, "record_kind": "candidate", "terminal_status": "accepted_verified_orientifold",
           "terminal_reason_code": "accepted_verified_orientifold", "evidence": {"smooth": True},
           "h11_minus": 0, "fixed_component_evidence": {"status": "verified"},
           "exact_action_h21_evidence": {"schema_version": "cyaxiverse-exact-action-hodge-evidence-3.0",
                                          "status": "validated", "h11_plus": 2, "h11_minus": 0,
                                          "h21_plus": 0, "h21_minus": 132,
                                          "chi_fixed_locus": 272, "chi_x": -260},
           "terminal_evidence": {"smoothness": {"status": "smooth"}},
           "lattice_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
           "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1}, "lambda_f": 1}
    row.update(changes)
    if row["record_kind"] != "candidate":
        row.update({"candidate_id": None, "action_digest": None, "lattice_matrix": None,
                    "torus_shift": None, "lambda_f": None})
    if row["action_digest"] is None and row["record_kind"] == "candidate": row["action_digest"] = action_digest(row)
    return finalize_terminal(row)


def _source_manifest_fixture(root: Path) -> dict:
    """Build a small unbound source manifest for preparation tests."""
    source_output = root / "source"
    source_checkpoint = root / "source-checkpoints"
    source_output.mkdir()
    source_checkpoint.mkdir()
    entries = []
    for h11 in (2, 3, 4, 5):
        for role in ("source_rows", "terminal_ledger"):
            path = source_output / f"h11-{h11}-{role}.jsonl"
            path.write_bytes(b"{}\n")
            entries.append({
                "h11": h11,
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "file_type": "jsonl",
                "source_row_or_partition_identity": f"h11={h11},row=1",
                "selection_route": "synthetic_fixture",
                "counting_unit": "synthetic class",
                "output_root": str(source_output),
            })
    manifest = {
        "schema": "cyaxiverse-general-l-action-replacement-input-1.0",
        "task_id": "synthetic-task",
        "program": "synthetic-program",
        "h11_values": [2, 3, 4, 5],
        "counting_unit": "synthetic class",
        "selection_route": "synthetic_fixture",
        "action_conventions": "synthetic actions",
        "terminal_conventions": "schema-1.2",
        "limits": CAPS,
        "global_limits": GLOBAL_LIMITS,
        "seed": 0,
        "dependency_manifest_sha256": "synthetic-dependencies",
        "runtime_versions": {
            "python_version": platform.python_version(),
            "julia_version": "1.12",
            "cytools_version": "not-used",
        },
        "relevant_environment_variables": {},
        "environment_revision": "synthetic-environment",
        "configuration_digest": "synthetic-configuration",
        "run_scope": "pilot",
        "output_root": str(source_output),
        "checkpoint_root": str(source_checkpoint),
        "production_gate": "not_validated",
        "scale_status": "not_applicable",
        "no_overwrite": True,
        "inputs": entries,
    }
    repo_root = Path(__file__).resolve().parent.parent
    manifest.update({
        **repository_revision(repo_root),
        "project_toml_sha256": hashlib.sha256(
            (repo_root / "Project.toml").read_bytes()
        ).hexdigest(),
        "manifest_toml_sha256": hashlib.sha256(
            (repo_root / "Manifest.toml").read_bytes()
        ).hexdigest(),
        "source_file_digests": {
            name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
            for name in REQUIRED_SOURCE_FILES
        },
    })
    manifest["input_manifest_sha256"] = input_manifest_digest(manifest)
    return manifest


def _multi_record_class_rows() -> list[dict]:
    """Build one class with structural attempts and two distinct actions."""
    common = {
        "schema_version": CANDIDATE_SCHEMA,
        "h11": 2,
        "source_row": 1,
        "polytope_id": "p-multi",
        "frst_hash": "f-multi",
        "frst_class_index": 0,
    }
    matrix = build_witness_record(
        {
            **common,
            "record_kind": "matrix_validation",
            "candidate_id": None,
            "lattice_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "matrix_id": "matrix-1",
        },
        None,
        {
            "record_kind": "matrix_validation",
            "terminal_status": "matrix_validation_passed",
            "terminal_reason_code": "matrix_validation_passed",
            "fixed_component_evidence": {"status": "not_evaluated"},
        },
    )
    summary = build_witness_record(
        {
            **common,
            "record_kind": "lattice_matrix_search_summary",
            "candidate_id": None,
            "matrix_id": "matrix-1",
        },
        None,
        {
            "record_kind": "lattice_matrix_search_summary",
            "terminal_status": "torus_shift_search_exhausted",
            "terminal_reason_code": "torus_shift_search_exhausted",
        },
    )
    exact_hodge = {
        "schema_version": "cyaxiverse-exact-action-hodge-evidence-3.0",
        "status": "validated",
        "h11_plus": 2,
        "h11_minus": 0,
        "h21_plus": 0,
        "h21_minus": 132,
        "chi_fixed_locus": 272,
        "chi_x": -260,
    }
    candidates = []
    for candidate_id, shift in (
        ("c1", {"numerator": [0, 0, 0, 0], "denominator": 1}),
        ("c2", {"numerator": [1, 0, 0, 0], "denominator": 2}),
    ):
        candidates.append(
            build_witness_record(
                {
                    **common,
                    "record_kind": "candidate",
                    "candidate_id": candidate_id,
                    "h11_minus": 0,
                    "fixed_component_evidence": {"status": "verified"},
                    "exact_action_h21_evidence": exact_hodge,
                },
                {
                    "lattice_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "torus_shift": shift,
                    "lambda_f": 1,
                },
                {
                    "terminal_status": "accepted_verified_orientifold",
                    "terminal_reason_code": "accepted_verified_orientifold",
                    "smoothness": {"status": "smooth"},
                },
            )
        )
    return [matrix, summary, *candidates]


class ContractTests(unittest.TestCase):
    def test_malformed_json_and_nonfinite_are_rejected(self):
        with self.assertRaises((json.JSONDecodeError, ValueError)): json.loads("{")
        with self.assertRaises(ContractError): canonical_bytes({"x": float("nan")})

    def test_exact_digest_and_reduced_rational_fixture(self):
        row = _row(torus_shift={"numerator": [1, 0, 0, 0], "denominator": 2})
        self.assertEqual(row["action_digest"], action_digest(row)); self.assertEqual(len(matrix_digest(row["lattice_matrix"])), 64)
        expected_matrix_digest = hashlib.sha256(
            json.dumps({"lattice_matrix": row["lattice_matrix"]}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(matrix_digest(row["lattice_matrix"]), expected_matrix_digest)
        with self.assertRaises(ContractError): validate_action({**row, "action_digest": "bad"})
        with self.assertRaises(ContractError): validate_action({**row, "torus_shift": {"numerator": [2, 0, 0, 0], "denominator": 4}})

    def test_matrix_terminal_uses_structural_fallback_identity(self):
        row = finalize_terminal({
            "polytope_id": "p1", "frst_hash": "f1", "record_kind": "matrix_validation",
            "terminal_status": "matrix_validation_passed",
            "terminal_reason_code": "matrix_validation_passed",
            "source_trilayer_candidate": {"matrix_candidate": "m1"},
        })
        self.assertIsNone(row.get("candidate_id"))
        self.assertEqual(len(row["terminal_record_identity"]), 64)

    def test_hodge_fixture_is_exact(self):
        result = hodge_split_from_euler(h11=2, h21=132, h11_minus=0, chi_fixed_locus=272, chi_x=-260)
        self.assertEqual((result["h11_plus"], result["h11_minus"], result["h21_plus"], result["h21_minus"]), (2, 0, 0, 132))
        with self.assertRaises(ContractError): hodge_split_from_euler(h11=2, h21=132, h11_minus=0, chi_fixed_locus=271, chi_x=-260)

    def test_terminal_tamper_missing_digest_and_comparison(self):
        left = _row(); right = copy.deepcopy(left)
        right["terminal_record_digest"] = "tampered"
        with self.assertRaises(ContractError): compare_witnesses([left], [right])
        with self.assertRaises(ContractError): validate_action({"lattice_matrix": left["lattice_matrix"], "torus_shift": left["torus_shift"], "lambda_f": 1})
        self.assertFalse(compare_witnesses([left], [_row(candidate_id="c2")])["equal"])
        missing = dict(left)
        missing["action_digest"] = None
        missing = finalize_terminal(missing)
        comparison = compare_witnesses([missing], [missing])
        self.assertFalse(comparison["equal"])
        self.assertEqual(comparison["missing_action_digest_count"], 2)

    def test_duplicate_orphan_missing_and_search_summary_accounting(self):
        rows = [_row(), _row(record_kind="lattice_matrix_search_summary", candidate_id=None, action_digest=None,
                             terminal_status="torus_shift_search_exhausted", terminal_reason_code="torus_shift_search_exhausted",
                             source_trilayer_candidate="structural-key")]
        counts = account_terminal_rows(rows); self.assertEqual(counts["search_summary_rows"], 1); self.assertEqual(counts["terminal_rows"], 2)
        self.assertEqual(counts["accepted_action_count"], 1)
        self.assertEqual(counts["selected_class_count"], 1)
        self.assertEqual(counts["representative_action_digest_by_class"], {"p1::f1": rows[0]["action_digest"]})
        self.assertEqual(compare_witnesses([rows[0]], [rows[0]])["equal"], True)

    def test_repeated_class_attempts_preserve_structural_rows_and_gate_contract(self):
        rows = _multi_record_class_rows()
        normalized = _witness_rows(copy.deepcopy(rows), 2)
        by_kind = {row["record_kind"]: row for row in normalized}
        for kind in ("matrix_validation", "lattice_matrix_search_summary"):
            for field in ("lattice_matrix", "torus_shift", "lambda_f", "action_digest"):
                self.assertIsNone(by_kind[kind][field])
        self.assertEqual(
            by_kind["matrix_validation"]["source_trilayer_candidate"]["matrix_id"],
            "matrix-1",
        )
        comparison = compare_witnesses(normalized, copy.deepcopy(normalized))
        self.assertTrue(comparison["equal"])
        self.assertTrue(comparison["class"]["live_duplicates"])
        self.assertEqual(comparison["class"]["live_duplicates"], ["p-multi::f-multi"])
        counts = account_terminal_rows(normalized)
        self.assertEqual(counts["terminal_rows"], 4)
        self.assertEqual(counts["duplicate_class_count"], 3)
        self.assertEqual(counts["duplicate_action_count"], 0)
        self.assertEqual(counts["duplicate_terminal_identity_count"], 0)
        self.assertEqual(
            bounded_gate(2, counts, witness_comparison=comparison)["status"],
            "passed",
        )

        partial_candidate = copy.deepcopy(normalized[-1])
        partial_candidate["torus_shift"] = None
        with self.assertRaisesRegex(ContractError, "action triple is incomplete"):
            _witness_rows([partial_candidate], 2)

        duplicate_action = copy.deepcopy(normalized[-1])
        duplicate_action["candidate_id"] = "c3"
        duplicate_action["source_candidate"]["candidate_id"] = "c3"
        duplicate_action["terminal_record_identity"] = terminal_identity(duplicate_action)
        duplicate_action["terminal_record_digest"] = terminal_digest(duplicate_action)
        duplicate_action_rows = normalized + [duplicate_action]
        duplicate_action_counts = account_terminal_rows(duplicate_action_rows)
        duplicate_action_comparison = compare_witnesses(
            duplicate_action_rows, normalized
        )
        self.assertEqual(duplicate_action_counts["duplicate_action_count"], 1)
        self.assertEqual(
            bounded_gate(
                2,
                duplicate_action_counts,
                witness_comparison=duplicate_action_comparison,
            )["status"],
            "blocked_on_evidence",
        )

        duplicate_terminal_rows = normalized + [copy.deepcopy(normalized[0])]
        duplicate_terminal_counts = account_terminal_rows(duplicate_terminal_rows)
        duplicate_terminal_comparison = compare_witnesses(
            duplicate_terminal_rows, normalized
        )
        self.assertEqual(
            duplicate_terminal_counts["duplicate_terminal_identity_count"], 1
        )
        self.assertEqual(
            bounded_gate(
                2,
                duplicate_terminal_counts,
                witness_comparison=duplicate_terminal_comparison,
            )["status"],
            "blocked_on_evidence",
        )

        multiplicity_comparison = compare_witnesses(normalized[:-1], normalized)
        self.assertFalse(multiplicity_comparison["class"]["equal"])
        self.assertEqual(
            bounded_gate(
                2,
                account_terminal_rows(normalized[:-1]),
                witness_comparison=multiplicity_comparison,
            )["status"],
            "blocked_on_evidence",
        )

        wrong_class = copy.deepcopy(normalized[-1])
        wrong_class["polytope_id"] = "p-other"
        wrong_class["source_candidate"]["polytope_id"] = "p-other"
        wrong_class["terminal_record_identity"] = terminal_identity(wrong_class)
        wrong_class["terminal_record_digest"] = terminal_digest(wrong_class)
        membership_comparison = compare_witnesses(
            normalized[:-1] + [wrong_class], normalized
        )
        self.assertFalse(membership_comparison["class"]["equal"])
        self.assertTrue(membership_comparison["class"]["live_minus_ledger"])
        self.assertEqual(
            bounded_gate(
                2,
                account_terminal_rows(normalized[:-1] + [wrong_class]),
                witness_comparison=membership_comparison,
            )["status"],
            "blocked_on_evidence",
        )

    def test_complete_witness_and_missing_digest_is_counted(self):
        record = build_witness_record({"polytope_id": "p1", "frst_hash": "f1", "candidate_id": "c1"},
                                      {"lattice_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1}, "lambda_f": 1},
                                      {"terminal_status": "accepted_verified_orientifold", "smoothness": {"status": "smooth"}})
        self.assertIn("source_candidate", record); self.assertIn("terminal_evidence", record)
        missing = copy.deepcopy(record); missing["action_digest"] = None
        missing["terminal_record_identity"] = terminal_identity(missing)
        missing["terminal_record_digest"] = terminal_digest(missing)
        self.assertEqual(account_terminal_rows([missing])["missing_action_digest_count"], 1)

    def test_named_sheridan_fixture_fails_closed_when_archive_is_absent(self):
        formula_source = Path(__file__).resolve().parent.parent / "validation" / "fuzzy_axions_supp" / "paper_source_2305_06363" / "KS_orientifolds.tex"
        self.assertTrue(formula_source.is_file())
        formula_text = formula_source.read_text(encoding="utf-8")
        self.assertIn("h^{2,1}_-", formula_text)
        self.assertIn("chi(\\mathcal{F}", formula_text)
        archive = Path("/Users/vmehta/Downloads/fuzzy-2412.12012v1.tar.gz")
        if not archive.is_file():
            self.assertFalse(archive.exists())
        else:
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), "905db55f2ab72e2b94ba9175148cd5a4976756e95ce37e64622bffdbc4d7bcea")
            with tarfile.open(archive, "r:gz") as source:
                tex = source.extractfile("main.tex").read().decode("utf-8")
            self.assertIn("h^{1,1}_+", tex)
            self.assertIn("(2, 0, 0, 132)", tex)
            self.assertIn("1 & -1 & 0 & 0 & -2 & 0", tex)

    def test_caps_ordering_resume_and_hash_mismatch(self):
        self.assertEqual(sorted(["b", "a"]), ["a", "b"])
        self.assertEqual(bounded_gate(2, {"favorable_polytopes_seen": 37, "frst_classes_seen": 1, "terminal_rows": 1})["status"], "blocked_on_evidence")
        expected = {"input_manifest_sha256": "a", "code_revision": "b", "environment_revision": "c", "configuration_digest": "d", "seed": 0, "limits": {"x": 1}, "output_root": "/new"}
        checkpoint = {**expected, "schema": "cyaxiverse-general-l-action-replacement-checkpoint-1.0", "last_class": [1, "p1", "f1", 0], "last_class_complete": True}
        validate_resume(checkpoint, expected)
        with self.assertRaises(ContractError): validate_resume({**checkpoint, "input_manifest_sha256": "bad"}, expected)
        with self.assertRaises(ContractError): validate_resume({**checkpoint, "last_class_complete": False}, expected)
        self.assertEqual(
            bounded_gate(2, {"candidate_action_attempts": CAPS[2]["max_action_attempts"] + 1})["status"],
            "blocked_on_evidence",
        )
        self.assertEqual(
            bounded_gate(2, {"terminal_rows": CAPS[2]["max_terminal_rows"] + 1})["status"],
            "blocked_on_evidence",
        )
        rows = [_row(source_row=2), _row(source_row=1, candidate_id="c2")]
        checkpoint_rows = {
            "schema": "cyaxiverse-general-l-action-replacement-checkpoint-1.0",
            "input_manifest_sha256": "a", "code_revision": "b", "environment_revision": "c",
            "configuration_digest": "d", "seed": 0, "limits": {"x": 1}, "output_root": "/new",
            "last_class_complete": True, "last_class": [1, "p1", "f1", 0],
        }
        self.assertEqual(len(next_class(rows, checkpoint_rows)), 1)
        with self.assertRaisesRegex(ContractError, "resume_mismatch"):
            next_class(rows, checkpoint_rows, {**checkpoint_rows, "output_root": "/other"})

    def test_unknown_status_is_retained_but_blocks_the_gate(self):
        row = _row(terminal_status="future_status", terminal_reason_code="future_status")
        counts = account_terminal_rows([row])
        self.assertEqual(counts["unknown_status_count"], 1)
        self.assertEqual(bounded_gate(2, counts)["status"], "blocked_on_evidence")

    def test_accepted_status_without_exact_hodge_evidence_blocks_selection(self):
        row = _row()
        row["exact_action_h21_evidence"]["h21_plus"] = 1
        row["terminal_record_identity"] = terminal_identity(row)
        row["terminal_record_digest"] = terminal_digest(row)
        counts = account_terminal_rows([row])
        self.assertEqual(counts["accepted_action_count"], 0)
        self.assertEqual(counts["acceptance_contract_failure_count"], 1)
        self.assertEqual(bounded_gate(2, counts)["status"], "blocked_on_evidence")

    def test_lambda_zero_is_retained_as_explicit_out_of_branch_rejection(self):
        record = build_witness_record(
            {"schema_version": CANDIDATE_SCHEMA, "polytope_id": "p1", "frst_hash": "f1",
             "candidate_id": "c0", "terminal_status": "accepted_verified_orientifold"},
            {"lattice_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
             "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1}, "lambda_f": 0},
            {"terminal_status": "accepted_verified_orientifold", "smoothness": {"status": "smooth"}},
        )
        self.assertEqual(record["terminal_reason_code"], "lambda_f_zero_out_of_branch")
        self.assertEqual(account_terminal_rows([record])["rejected_action_count"], 1)
        self.assertEqual(account_terminal_rows([record])["acceptance_contract_failure_count"], 0)

    def test_atomic_no_overwrite_and_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            atomic_create(path, b"one")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(b"one").hexdigest())
            with self.assertRaises(FileExistsError): atomic_create(path, b"two")

    def test_prepare_manifest_is_create_only_and_rejects_stale_or_aliased_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source_manifest_fixture(root)
            source_snapshot = copy.deepcopy(source)
            prepared_path = root / "prepared.json.zst"
            bounded_output = root / "bounded-output"
            bounded_checkpoint = root / "bounded-checkpoints"
            prepared = prepare_bounded_manifest(
                source,
                output_root=bounded_output,
                checkpoint_root=bounded_checkpoint,
                output_manifest_path=prepared_path,
            )
            self.assertEqual(source, source_snapshot)
            self.assertEqual(
                prepared["source_generation_output_root"], source["output_root"]
            )
            self.assertEqual(
                prepared["source_generation_checkpoint_root"], source["checkpoint_root"]
            )
            self.assertEqual(prepared["output_root"], str(bounded_output.resolve()))
            self.assertEqual(prepared["checkpoint_root"], str(bounded_checkpoint.resolve()))
            self.assertTrue(all(
                entry["output_root"] == str(bounded_output.resolve())
                and entry["checkpoint_root"] == str(bounded_checkpoint.resolve())
                for entry in prepared["inputs"]
            ))
            self.assertEqual(
                prepared["input_manifest_sha256"], input_manifest_digest(prepared)
            )
            self.assertEqual(load_json(prepared_path), prepared)

            for index, run_root in enumerate(
                (Path(source["output_root"]), Path(source["checkpoint_root"])),
                1,
            ):
                with self.assertRaises(ContractError):
                    prepare_bounded_manifest(
                        source,
                        output_root=root / f"source-overlap-output-{index}",
                        checkpoint_root=root / f"source-overlap-checkpoint-{index}",
                        output_manifest_path=run_root / f"nested-{index}.json.zst",
                    )
            for index, run_root in enumerate(
                (root / "nested-output-root", root / "nested-checkpoint-root"),
                1,
            ):
                with self.assertRaises(ContractError):
                    prepare_bounded_manifest(
                        source,
                        output_root=root / "nested-output-root",
                        checkpoint_root=root / "nested-checkpoint-root",
                        output_manifest_path=run_root / f"nested-{index}.json.zst",
                    )

            bounded_output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_bounded_manifest(
                    source,
                    output_root=bounded_output,
                    checkpoint_root=root / "other-checkpoints",
                    output_manifest_path=root / "second-prepared.json.zst",
                )
            with self.assertRaises(ContractError):
                prepare_bounded_manifest(
                    source,
                    output_root="relative-output",
                    checkpoint_root=root / "other-checkpoints",
                    output_manifest_path=root / "third-prepared.json.zst",
                )
            with self.assertRaises(ContractError):
                prepare_bounded_manifest(
                    source,
                    output_root=root / "other-output",
                    checkpoint_root=root / "other-output",
                    output_manifest_path=root / "fourth-prepared.json.zst",
                )
            alias_target = root / "alias-target"
            alias_root = root / "alias-root"
            alias_root.symlink_to(alias_target, target_is_directory=True)
            with self.assertRaises(ContractError):
                prepare_bounded_manifest(
                    source,
                    output_root=alias_root,
                    checkpoint_root=root / "alias-checkpoints",
                    output_manifest_path=root / "alias-prepared.json.zst",
                )
            with self.assertRaises(ContractError):
                prepare_bounded_manifest(
                    {**source, "input_manifest_sha256": "stale"},
                    output_root=root / "fifth-output",
                    checkpoint_root=root / "fifth-checkpoints",
                    output_manifest_path=root / "fifth-prepared.json.zst",
                )
            with self.assertRaises(ContractError):
                prepare_bounded_manifest(
                    {**source, "approval_fingerprint": {"sha256": "bound"}},
                    output_root=root / "sixth-output",
                    checkpoint_root=root / "sixth-checkpoints",
                    output_manifest_path=root / "sixth-prepared.json.zst",
                )

    def test_prepare_manifest_cli_uses_explicit_preparation_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source_manifest_fixture(root)
            source_path = root / "source-manifest.json.zst"
            from run_general_l_action_replacement_bounded import write_json_zst
            write_json_zst(source_path, source)
            output = root / "bounded-output"
            checkpoint = root / "bounded-checkpoints"
            prepared_path = root / "prepared-manifest.json.zst"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    os.fspath(Path(__file__).resolve().parent / "run_general_l_action_replacement_bounded.py"),
                    "--prepare-bounded-manifest",
                    "--input-manifest",
                    os.fspath(source_path),
                    "--output-root",
                    os.fspath(output),
                    "--checkpoint-root",
                    os.fspath(checkpoint),
                    "--output-manifest",
                    os.fspath(prepared_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn('"source_generation_output_root"', result.stdout)
            prepared = load_json(prepared_path)
            self.assertEqual(prepared["output_root"], str(output.resolve()))
            with self.assertRaisesRegex(SystemExit, "mutually exclusive"):
                main([
                    "--prepare-bounded-manifest",
                    "--bind-approval-manifest",
                ])
            with self.assertRaisesRegex(SystemExit, "mutually exclusive"):
                main([
                    "--prepare-bounded-manifest",
                    "--output-root",
                    "/tmp/a",
                    "--bounded-output-root",
                    "/tmp/b",
                ])
            with self.assertRaisesRegex(SystemExit, "mutually exclusive"):
                main([
                    "--prepare-bounded-manifest",
                    "--checkpoint-root",
                    "/tmp/a",
                    "--bounded-checkpoint-root",
                    "/tmp/b",
                ])
            with self.assertRaisesRegex(SystemExit, "execution-only"):
                main([
                    "--bind-approval-manifest",
                    "--output-root",
                    "/tmp/output",
                ])
            with self.assertRaisesRegex(SystemExit, "execution-only"):
                main([
                    "--bind-approval-manifest",
                    "--resume",
                    "/tmp/resume.json",
                ])
            with self.assertRaisesRegex(SystemExit, "--output-manifest requires"):
                main([
                    "--approval",
                    "/tmp/approval.json",
                    "--input-manifest",
                    "/tmp/input-manifest.json.zst",
                    "--output-root",
                    "/tmp/output",
                    "--output-manifest",
                    "/tmp/ignored-manifest.json.zst",
                ])

    def test_approval_binding_is_deterministic_and_non_circular(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_output = root / "source"
            source_checkpoint = root / "source-checkpoints"
            source_output.mkdir()
            source_checkpoint.mkdir()
            manifest = {
                "schema": "cyaxiverse-general-l-action-replacement-input-1.0",
                **{
                    field: {
                        "limits": {},
                        "global_limits": {},
                    }.get(field, f"{field}-fixture")
                    for field in (
                        "task_id", "program", "h11_values", "counting_unit",
                        "selection_route", "action_conventions", "terminal_conventions",
                        "limits", "global_limits", "seed", "dependency_manifest_sha256",
                        "project_toml_sha256", "manifest_toml_sha256", "runtime_versions",
                        "relevant_environment_variables", "environment_revision",
                        "source_file_digests", "source_commit", "tree_sha256",
                        "working_tree_diff_sha256", "configuration_digest", "output_root",
                        "checkpoint_root", "production_gate", "scale_status", "no_overwrite",
                    )
                },
            }
            manifest.update({
                "h11_values": [2, 3, 4, 5], "limits": CAPS,
                "global_limits": GLOBAL_LIMITS, "seed": 0,
                "runtime_versions": {"python_version": "fixture", "julia_version": "fixture", "cytools_version": "fixture"},
                "relevant_environment_variables": {}, "source_file_digests": {},
                "output_root": str(root / "published"),
                "checkpoint_root": str(root / "checkpoints"),
                "source_generation_output_root": str(source_output),
                "source_generation_checkpoint_root": str(source_checkpoint),
                "production_gate": "not_validated", "scale_status": "not_applicable",
                "no_overwrite": True,
            })
            manifest["input_manifest_sha256"] = input_manifest_digest(manifest)
            approval = {
                "status": "owner_approved", "approval_id": "fixture-approval",
                "approval_date": "2026-09-02", "new_bounded_run_authorized": True,
                **manifest,
                "schema": "cyaxiverse-general-l-action-replacement-approval-1.0",
            }
            approval_path = root / "approval.json"
            approval_path.write_bytes(canonical_bytes(approval))
            bound_one = create_approval_bound_manifest(
                manifest, approval, approval_path=approval_path,
                output_manifest_path=root / "bound-one.json.zst",
            )
            bound_two = create_approval_bound_manifest(
                manifest, approval, approval_path=approval_path,
                output_manifest_path=root / "bound-two.json.zst",
            )
            self.assertEqual(bound_one, bound_two)
            self.assertEqual(
                bound_one["input_manifest_sha256"], input_manifest_digest(bound_one)
            )
            self.assertEqual(
                load_json(root / "bound-one.json.zst"),
                json.loads(json.dumps(bound_one)),
            )
            for index, run_root in enumerate((source_output, source_checkpoint), 1):
                with self.assertRaises(ContractError):
                    create_approval_bound_manifest(
                        manifest,
                        approval,
                        approval_path=approval_path,
                        output_manifest_path=run_root / f"nested-bound-{index}.json.zst",
                    )
            for index, run_root in enumerate(
                (Path(manifest["output_root"]), Path(manifest["checkpoint_root"])),
                1,
            ):
                with self.assertRaises(ContractError):
                    create_approval_bound_manifest(
                        manifest,
                        approval,
                        approval_path=approval_path,
                        output_manifest_path=run_root / f"nested-bound-{index}.json.zst",
                    )
            approval_in_source_root = source_output / "approval.json"
            approval_in_source_root.write_bytes(approval_path.read_bytes())
            with self.assertRaises(ContractError):
                create_approval_bound_manifest(
                    manifest,
                    approval,
                    approval_path=approval_in_source_root,
                    output_manifest_path=root / "safe-bound.json.zst",
                )
            bad_approval_path = root / "bad-approval.json"
            bad_approval = dict(approval)
            bad_approval["approval_id"] = "different"
            bad_approval_path.write_bytes(canonical_bytes(bad_approval))
            with self.assertRaisesRegex(ContractError, "approval content mismatch"):
                create_approval_bound_manifest(
                    manifest, approval, approval_path=bad_approval_path,
                    output_manifest_path=root / "bad-bound.json.zst",
                )
            with self.assertRaises(ContractError):
                create_approval_bound_manifest(
                    bound_one, approval, approval_path=approval_path,
                    output_manifest_path=root / "bound-three.json.zst",
                )

    def test_tiny_approved_fixture_executes_and_publishes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_output = root / "source"
            source_checkpoint = root / "source-checkpoints"
            output = root / "published"
            checkpoint = root / "checkpoints"
            source_output.mkdir()
            source_checkpoint.mkdir()
            entries = []
            for h11 in (2, 3, 4, 5):
                raw = {"schema_version": CANDIDATE_SCHEMA, "h11": h11, "source_row": 1, "polytope_id": f"p{h11}", "frst_hash": f"f{h11}",
                       "frst_class_index": 0, "candidate_id": f"c{h11}", "record_kind": "candidate",
                       "terminal_status": "accepted_verified_orientifold", "terminal_reason_code": "accepted_verified_orientifold",
                       "lattice_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                       "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1}, "lambda_f": 1,
                       "h11_minus": 0,
                       "fixed_component_evidence": {"status": "verified"},
                       "exact_action_h21_evidence": {"schema_version": "cyaxiverse-exact-action-hodge-evidence-3.0", "status": "validated",
                                                       "h11_minus": 0, "h21_plus": 0, "h21_minus": 132, "h11_plus": h11,
                                                       "chi_fixed_locus": 272, "chi_x": -260},
                       "terminal_evidence": {"smoothness": {"status": "smooth"}}}
                raw["action_digest"] = action_digest(raw)
                for role in ("source_rows", "terminal_ledger"):
                    path = source_output / f"{h11}-{role}.jsonl"; path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
                    entries.append({"h11": h11, "role": role, "path": str(path), "size_bytes": path.stat().st_size,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "file_type": "jsonl",
                                    "source_row_or_partition_identity": f"h11={h11},row=1", "selection_route": "synthetic_fixture",
                                    "counting_unit": "favorable CY FRST class keyed by polytope_id::frst_hash"})
            revision = repository_revision(Path(__file__).resolve().parent.parent)
            repo_root = Path(__file__).resolve().parent.parent
            source_file_digests = {
                name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
                for name in REQUIRED_SOURCE_FILES
            }
            bindings = {"task_id": "general-l-action-replacement-bounded-run-h11-2-5", "program": "source-compatible inherited-orientifold general-L action validation",
                        "h11_values": [2, 3, 4, 5], "counting_unit": "favorable CY FRST class keyed by polytope_id::frst_hash",
                        "selection_route": "synthetic_fixture", "action_conventions": "exact (L,t,lambda_f), contragredient L, source order frozen",
                        "terminal_conventions": "schema-1.2 identity and complete-record digest", "limits": CAPS, "global_limits": GLOBAL_LIMITS, "seed": 0,
                        **revision, "dependency_manifest_sha256": "synthetic-dependencies",
                        "project_toml_sha256": hashlib.sha256((repo_root / "Project.toml").read_bytes()).hexdigest(),
                        "manifest_toml_sha256": hashlib.sha256((repo_root / "Manifest.toml").read_bytes()).hexdigest(),
                        "runtime_versions": {"python_version": platform.python_version(), "julia_version": "1.12", "cytools_version": "not-used"},
                        "relevant_environment_variables": {name: os.environ.get(name) for name in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "environment_revision": "synthetic-python",
                        "source_file_digests": source_file_digests, "configuration_digest": "fixture", "output_root": str(source_output),
                        "checkpoint_root": str(source_checkpoint), "run_scope": "pilot",
                        "production_gate": "not_validated", "scale_status": "not_applicable",
                        "no_overwrite": True}
            for entry in entries:
                entry.update({key: bindings[key] for key in ("source_commit", "tree_sha256", "working_tree_diff_sha256", "environment_revision", "configuration_digest", "seed", "limits", "global_limits", "output_root")})
            manifest = {"schema": "cyaxiverse-general-l-action-replacement-input-1.0", **bindings, "inputs": entries}
            manifest["input_manifest_sha256"] = input_manifest_digest(manifest)
            manifest = prepare_bounded_manifest(
                manifest,
                output_root=output,
                checkpoint_root=checkpoint,
                output_manifest_path=root / "prepared-input-manifest.json.zst",
                repo_root=repo_root,
            )
            approval = {"status": "approved",
                        "approval_id": "synthetic-approval", "approval_date": "2026-09-02",
                        "new_bounded_run_authorized": True, **manifest,
                        "schema": "cyaxiverse-general-l-action-replacement-approval-1.0"}
            # Exercise the JSON representation used by the CLI, including
            # stringified object keys in the limits map.
            manifest = json.loads(json.dumps(manifest))
            approval = json.loads(json.dumps(approval))
            approval_path = root / "approval.json"
            approval_path.write_text(
                json.dumps(approval, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            manifest = create_approval_bound_manifest(
                manifest,
                approval,
                approval_path=approval_path,
                output_manifest_path=root / "bound-input-manifest.json.zst",
            )
            result = execute_bounded(approval, manifest, output, repo_root=repo_root)
            self.assertEqual(result["production_gate"], "not_validated")
            self.assertEqual(result["scale_status"], "not_applicable")
            self.assertEqual(len(list(output.iterdir())), 22)
            self.assertTrue((output / "SHA256SUMS.txt").is_file())
            checksum_lines = (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(checksum_lines), 21)
            self.assertTrue(all("SHA256SUMS.txt" not in line for line in checksum_lines))
            for artifact in output.glob("*.zst"):
                subprocess.run(["zstd", "-tq", str(artifact)], check=True)
            changed_manifest = copy.deepcopy(manifest)
            changed_approval = copy.deepcopy(approval)
            changed_output = root / "changed-revision-output"
            changed_manifest["source_commit"] = "changed-source-revision"
            changed_approval["source_commit"] = "changed-source-revision"
            changed_manifest["output_root"] = str(changed_output)
            changed_approval["output_root"] = str(changed_output)
            with self.assertRaisesRegex(ContractError, "provenance_mismatch"):
                execute_bounded(changed_approval, changed_manifest, changed_output, repo_root=repo_root)

    def test_input_hash_revision_orphan_missing_and_duplicate_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            manifest = {"schema": "cyaxiverse-general-l-action-replacement-input-1.0", "inputs": [{"path": str(path), "size_bytes": path.stat().st_size,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "source_row_or_partition_identity": "row=1", "selection_route": "fixture",
                                     "counting_unit": "class"}]}
            path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "input_fingerprint_mismatch"):
                refingerprint_manifest(manifest)
        left = _row()
        duplicate = compare_witnesses([left, left], [left])
        self.assertFalse(duplicate["equal"])
        self.assertTrue(duplicate["class"]["live_duplicates"])
        orphan = compare_witnesses([left], [])
        self.assertIn("p1::f1", orphan["class"]["live_minus_ledger"])
        self.assertIn("p1::f1::" + left["action_digest"], orphan["action"]["live_minus_ledger"])

    def test_terminal_duplicate_deterministic_order_and_truncated_resume(self):
        first = _row(h11=2, source_row=1)
        second = _row(h11=2, source_row=2, candidate_id="c2")
        raw_first, raw_second = (dict(first), dict(second))
        for raw in (raw_first, raw_second):
            raw.pop("terminal_record_identity", None)
            raw.pop("terminal_record_digest", None)
        ordered = _witness_rows([raw_second, raw_first], 2)
        self.assertEqual([row["source_row"] for row in ordered], [1, 2])
        duplicate = compare_witnesses([first, first], [first])
        self.assertTrue(duplicate["terminal"]["live_duplicates"])
        with tempfile.TemporaryDirectory() as directory:
            truncated = Path(directory) / "checkpoint.json"
            truncated.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "malformed JSON"):
                load_json(truncated)


if __name__ == "__main__": unittest.main()
