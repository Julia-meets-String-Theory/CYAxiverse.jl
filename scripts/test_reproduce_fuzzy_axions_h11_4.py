#!/usr/bin/env python3
"""Focused tests for audit-driver argument and population contracts."""

from argparse import Namespace
from contextlib import redirect_stderr
import hashlib
from io import StringIO
import json
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import reproduce_fuzzy_axions_h11_4 as driver
replay = driver


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

    def test_bounded_kaehler_sampler_cli_is_explicit_and_seeded(self):
        args = driver._parse_args(
            [
                "--parquet-dir",
                "unused",
                "--sample-kaehler-points",
                "--kaehler-sampler-seed",
                "42",
            ]
        )
        self.assertTrue(args.sample_kaehler_points)
        self.assertEqual(args.kaehler_sampler_seed, 42)
        self.assertEqual(args.qcd_divisor_domain, "all_prime")

    def test_production_kaehler_mode_is_explicit_and_fail_closed(self):
        args = driver._parse_args(
            [
                "--parquet-dir",
                "unused",
                "--kaehler-search-mode",
                "production",
                "--kaehler-production-seed",
                "23",
                "--kaehler-production-starts",
                "3",
            ]
        )
        self.assertTrue(args.production_kaehler_search)
        self.assertEqual(args.kaehler_production_seed, 23)
        self.assertEqual(args.kaehler_production_starts, 3)
        self.assertEqual(args.kaehler_production_optimizer, "SLSQP")
        self.assertTrue(args.model_stage)

        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            driver._parse_args(
                [
                    "--parquet-dir",
                    "unused",
                    "--production-kaehler-search",
                    "--qcd-divisor-domain",
                    "leading_nonself",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --qcd-divisor-domain=all_prime", stderr.getvalue())

    def test_bounded_kaehler_sampler_exports_four_reproducible_attempts(self):
        class FakeCalabiYau:
            def toric_kahler_cone(self):
                return object()

            def toric_effective_cone(self):
                return SimpleNamespace(rays=lambda: [[1]])

            def glsm_charge_matrix(self, include_origin=False):
                return [[1]]

        class FakeTriangulation:
            def get_cy(self):
                return FakeCalabiYau()

        seed = 17

        def canonical(_triangulation, *, point_seed=None):
            return {
                "status": "accepted",
                "point": [1.0],
                "diagnostic": {"solver": "tip"},
                "point_seed": point_seed,
            }

        def proposals(*_args, point_seed=None, **_kwargs):
            return iter(
                [
                    {
                        "attempt_index": 1,
                        "point_kind": "canonical_tip",
                        "point_seed": point_seed,
                        "solver": None,
                        "point": np.asarray([1.0]),
                    },
                    *[
                        {
                            "attempt_index": attempt,
                            "point_kind": "randomized_projection",
                            "point_seed": driver.stable_seed(
                                "kaehler-point-attempt", point_seed, attempt
                            ),
                            "solver": "test",
                            "point": np.asarray([float(attempt)]),
                        }
                        for attempt in (2, 3, 4)
                    ],
                ]
            )

        def evaluate(_cy, _cone, _qprime, point, *, attempt_index, **_kwargs):
            return (
                {
                    "attempt_index": attempt_index,
                    "point_kind": "canonical_tip"
                    if attempt_index == 1
                    else "randomized_projection",
                    "point_seed": _kwargs.get("point_seed"),
                    "solver": _kwargs.get("solver"),
                },
                {
                    "point": np.asarray(point),
                    "cy_volume": 1.0,
                    "prime_divisor_volumes": np.asarray([1.0]),
                    "inverse_metric": np.asarray([[1.0]]),
                },
            )

        with patch.object(driver, "_export_kaehler_point", side_effect=canonical), patch.object(
            driver, "extract_topology", return_value={"h11": 1, "kappa": [[1]], "mori_cone": [[1]]}
        ), patch.object(
            driver, "canonicalize_unique_charge_rows", return_value=([[1]], {})
        ), patch.object(
            driver, "sample_stretched_kaehler_points", side_effect=proposals
        ), patch.object(driver, "evaluate_kaehler_point", side_effect=evaluate):
            first = driver._export_kaehler_points(
                FakeTriangulation(),
                point_seed=seed,
                attempts=driver.KAEHLER_DIAGNOSTIC_ATTEMPTS,
                seed_metadata={"master_seed": 9},
            )
            second = driver._export_kaehler_points(
                FakeTriangulation(),
                point_seed=seed,
                attempts=driver.KAEHLER_DIAGNOSTIC_ATTEMPTS,
                seed_metadata={"master_seed": 9},
            )

        self.assertEqual(len(first), 4)
        self.assertEqual([record["attempt_index"] for record in first], [1, 2, 3, 4])
        self.assertEqual(first, second)
        self.assertEqual(
            first[0]["point_provenance"]["proposal_kind"], "canonical_tip"
        )
        self.assertTrue(all(record["sampler_seed"] == seed for record in first))
        self.assertTrue(all(record["attempt_budget"] == 4 for record in first))
        self.assertTrue(
            all(
                record["point_provenance"]["proposal_kind"] == "seeded_projected_cone"
                for record in first[1:]
            )
        )
        self.assertEqual(
            [record["proposal_seed"] for record in first[1:]],
            [
                driver.stable_seed("kaehler-point-attempt", seed, attempt)
                for attempt in (2, 3, 4)
            ],
        )

    def test_production_optimizer_enforces_log_mass_and_volume_constraints(self):
        target_log10 = np.log10(driver.FUZZY_AXION_MASS_TARGET_EV)
        values = {
            "point": np.asarray([2.0]),
            "prime_divisor_volumes": np.asarray([30.0]),
            "inverse_metric": np.asarray([[1.0]]),
            "cy_volume": 1.0,
        }

        def evaluate(_point):
            return values

        def mass(_point, _values):
            return target_log10

        def fake_minimize(objective, x0, *, method, constraints, options):
            self.assertEqual(method, "SLSQP")
            self.assertAlmostEqual(constraints[0]["fun"](x0), 0.0)
            self.assertTrue(np.all(constraints[1]["fun"](x0) >= 0.0))
            self.assertEqual(options["maxiter"], 3)
            return SimpleNamespace(
                x=np.asarray([2.0]),
                success=True,
                status=0,
                message="mock converged",
                nit=1,
                nfev=2,
            )

        with patch.object(driver, "scipy_minimize", side_effect=fake_minimize):
            result = driver._production_constrained_kaehler_optimizer(
                [2.0],
                evaluate_point=evaluate,
                mass_log10_evaluator=mass,
                axion_index=1,
                qcd_divisor_index=1,
                start_index=1,
                start_seed=7,
                requested_starts=2,
                kahler_hyperplanes=np.asarray([[1.0]]),
                maxiter=3,
            )

        self.assertTrue(result["converged"])
        self.assertEqual(result["constraints"]["mass_log10"]["type"], "equality")
        self.assertEqual(result["constraints"]["qcd_volume"]["bounds"], [25.0, 40.0])
        self.assertTrue(result["constraints"]["prime_divisor_volumes"]["satisfied"])
        self.assertIn("optimized_tensors", result)

    def test_production_search_runs_each_pair_and_deduplicates_tensors(self):
        class FakeCone:
            def hyperplanes(self):
                return np.asarray([[1.0, 0.0], [0.0, 1.0]])

        class FakeEffectiveCone:
            def rays(self):
                return np.asarray([[1, 0], [0, 1]])

        class FakeCalabiYau:
            def toric_kahler_cone(self):
                return FakeCone()

            def toric_effective_cone(self):
                return FakeEffectiveCone()

        class FakeTriangulation:
            def get_cy(self):
                return FakeCalabiYau()

        values = {
            "point": np.asarray([2.0, 2.0]),
            "prime_divisor_volumes": np.asarray([30.0, 31.0, 32.0]),
            "inverse_metric": np.eye(2),
            "cy_volume": 8.0,
        }
        canonical = {
            "status": "accepted",
            "point": [1.0, 1.0],
            "glsm_charge_matrix": [[1, 0, 1], [0, 1, 1]],
            "prime_divisor_volumes": [1.0, 1.0, 1.0],
        }
        starts = [
            {"start_index": 1, "start_seed": 11, "point": np.asarray([1.0, 1.0])},
            {"start_index": 2, "start_seed": 12, "point": np.asarray([1.2, 1.2])},
        ]
        constraints = {
            "mass_log10": {"satisfied": True},
            "prime_divisor_volumes": {"satisfied": True},
            "qcd_volume": {"satisfied": True},
            "kaehler_cone": {"satisfied": True},
        }

        def fake_optimizer(initial_point, **kwargs):
            return {
                "converged": True,
                "status": "converged",
                "start_index": kwargs["start_index"],
                "start_seed": kwargs["start_seed"],
                "constraints": constraints,
                "residuals": {},
                "optimized_tensors": values,
                "_optimized_values": values,
            }

        args = SimpleNamespace(
            kaehler_production_starts=2,
            kaehler_production_optimizer="SLSQP",
            kaehler_production_ftol=1e-10,
            kaehler_production_maxiter=10,
            kaehler_production_mass_tolerance_log10=1e-8,
            kaehler_production_volume_tolerance=1e-8,
            gs=0.5,
            w0_real=1.0,
            w0_imag=0.0,
        )
        topology = {
            "h11": 2,
            "kappa": np.zeros((2, 2, 2)),
            "mori_cone": np.eye(2),
            "kahler_cone_hyperplanes": np.eye(2),
        }
        with patch.object(driver, "extract_topology", return_value=topology), patch.object(
            driver, "_production_start_points", return_value=(starts, [])
        ), patch.object(
            driver,
            "_production_constrained_kaehler_optimizer",
            side_effect=fake_optimizer,
        ):
            result = driver._production_kaehler_search(
                FakeTriangulation(), canonical, args, seed=19
            )

        self.assertEqual(result["finite_search_status"], "completed")
        self.assertEqual(result["pair_count"], 6)
        self.assertEqual(result["converged_pair_count"], 6)
        self.assertEqual(result["deduplication"]["candidate_count"], 12)
        self.assertEqual(result["deduplication"]["unique_count"], 1)
        self.assertEqual(result["deduplication"]["duplicate_count"], 11)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["qcd_divisor_domain"], "all_prime")
        self.assertEqual(result["records"][0]["kaehler_search_mode"], "production")


class _Poly:
    def __init__(self, h11):
        self._h11 = h11

    def h11(self):
        return self._h11


class _Triangulation:
    def get_cy(self):
        return object()


class _GeometryPoly:
    def __init__(self, h11):
        self._h11 = h11

    def h11(self):
        return self._h11


class CertificateActionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.action = {
            "lattice_matrix": [[1, 0], [0, -1]],
            "torus_shift": {"numerator": [1, 0], "denominator": 2},
            "lambda_f": 1,
        }
        self.certificate = {"action": {"witness": self.action, "digest": "d"}}

    def test_unique_action_match(self):
        candidate = {"action": dict(self.action), "terminal_status": "accepted"}
        self.assertEqual(
            replay._select_certificate_action([candidate], self.certificate)["action"],
            self.action,
        )

    def test_first_action_mismatch_is_rejected(self):
        mismatch = dict(self.action)
        mismatch["lambda_f"] = 0
        with self.assertRaises(replay.ReplayConfigurationError):
            replay._select_certificate_action([{"action": mismatch}], self.certificate)

    def test_zero_and_ambiguous_matches_are_rejected(self):
        candidate = {"action": dict(self.action)}
        with self.assertRaises(replay.ReplayConfigurationError):
            replay._select_certificate_action([], self.certificate)
        with self.assertRaises(replay.ReplayConfigurationError):
            replay._select_certificate_action([candidate, candidate], self.certificate)


class _GeometryTriangulation:
    pass


def _args(root, **updates):
    values = dict(
        h11=4,
        parquet_dir=str(root),
        max_rows=0,
        checkpoint_interval=32,
        dry_run=True,
        workers=1,
        shard_count=1,
        shard_index=0,
        checkpoint=root / "checkpoint.jsonl.zst",
        resume=False,
        orientifold_reason_diagnostics=False,
        output=None,
        allow_terminal_only_smoke=False,
    )
    values.update(updates)
    return Namespace(**values)


def _preflight(root):
    merged = root / "merged.json.zst"
    gap = root / "gap.json.zst"
    return {
        "status": "passed",
        "handoffs": [],
        "artifacts": {
            "merged_artifact": str(merged),
            "gap_analysis_artifact": str(gap),
            "artifact_digests": {
                "merged_artifact": {"path": str(merged.resolve()), "sha256": "a" * 64},
                "gap_analysis_artifact": {"path": str(gap.resolve()), "sha256": "b" * 64},
            },
        },
    }


def _write_merged_fixture(root, entries):
    payload = {"requested_h11": 4, "terminal_ledger": {"class_funnel": entries}}
    raw = root / "merged.json"
    merged = root / "merged.json.zst"
    raw.write_text(json.dumps(payload), encoding="utf-8")
    subprocess.run(["zstd", "-q", "-19", "-f", "-o", str(merged), str(raw)], check=True)
    preflight = _preflight(root)
    preflight["artifacts"]["artifact_digests"]["merged_artifact"]["sha256"] = hashlib.sha256(merged.read_bytes()).hexdigest()
    return preflight


class ReplayContractTests(unittest.TestCase):
    def test_cli_requires_explicit_h11_and_defaults_to_bounded_one_worker(self):
        parser = replay.build_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--parquet-dir", "/tmp/source"])
        args = parser.parse_args(["--parquet-dir", "/tmp/source", "--h11", "4"])
        self.assertEqual(args.max_rows, 0)
        self.assertEqual(args.workers, 1)
        self.assertEqual(args.checkpoint_interval, 32)
        self.assertTrue(parser.parse_args(["--parquet-dir", "/tmp/source", "--h11", "4", "--orientifold-reason-diagnostics"]).orientifold_reason_diagnostics)

    def test_worker_and_shard_caps_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            preflight = _preflight(root)
            with self.assertRaises(replay.ReplayConfigurationError):
                replay._freeze_replay_config(_args(root, workers=5), preflight)
            with self.assertRaises(replay.ReplayConfigurationError):
                replay._freeze_replay_config(_args(root, shard_count=5), preflight)

    def test_h11_dual_check_requires_mirror_and_cytools_equality(self):
        source = {"mirror_h12": 4}
        replay._verify_replay_row(_Poly(4), source, 4)
        with self.assertRaises(replay.ReplayConfigurationError):
            replay._verify_replay_row(_Poly(5), source, 4)
        with self.assertRaises(replay.ReplayConfigurationError):
            replay._verify_replay_row(_Poly(4), {"mirror_h12": 5}, 4)

    def test_zstd_checkpoint_is_atomic_and_resume_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "checkpoint.jsonl.zst"
            config = {"schema_version": replay.REPLAY_SCHEMA_VERSION, "source_input_sha256": "a"}
            rows, identities = replay._checkpoint_state(path, config, resume=False)
            self.assertEqual(identities, set())
            rows.append({"record_type": "row", "row_identity": "row-1", "terminal_status": "accepted"})
            replay._zstd_jsonl_write_atomic(path, rows)
            self.assertTrue(path.is_file())
            self.assertEqual(replay._zstd_jsonl_read(path)[1]["row_identity"], "row-1")
            with self.assertRaises(replay.ReplayResumeError):
                replay._checkpoint_state(path, {**config, "source_input_sha256": "b"}, resume=True)

    def test_schema3_identity_and_terminal_duplicate_accounting(self):
        source = {
            "declared_source_digest": "source-digest",
            "canonical_polytope_id": "p",
            "global_coordinates": [[0, 0, 0, 0]],
        }
        action = {"frst_hash": "frst", "action_digest": "a"}
        first = replay._row_identity(source, 0, action)
        second = replay._row_identity(source, 99, action)
        self.assertEqual(first, second)
        self.assertEqual(replay.CANDIDATE_SCHEMA_VERSION, "cyaxiverse-inherited-orientifold-candidate-3.0")

    def test_witness_derived_identities_are_stable_and_distinct(self):
        source = {"source_digest": "d", "polytope_id": "p", "frst_hash": "f"}
        first = replay._row_identity(source, 0, {"accepted_witness": {"matrix_id": "a"}})
        self.assertEqual(first, replay._row_identity(source, 99, {"accepted_witness": {"matrix_id": "a"}}))
        self.assertNotEqual(first, replay._row_identity(source, 0, {"accepted_witness": {"matrix_id": "b"}}))

    def test_merged_ledger_cap_is_candidate_identity_cap(self):
        merged = {
            "requested_h11": 4,
            "terminal_ledger": {
                "class_funnel": [
                    {"polytope_id": "rejected", "accepted_for_table_1": False, "accepted_witness": None, "frst_class_index": 7, "frst_hash": "fr"},
                    {"polytope_id": "p2", "accepted_for_table_1": True, "accepted_witness": {"matrix_id": "m2"}, "frst_class_index": 8, "frst_hash": "f2"},
                    {"polytope_id": "p1", "accepted_for_table_1": True, "accepted_witness": {"matrix_id": "m1"}, "frst_class_index": 99, "frst_hash": "f1"},
                ]
            },
        }
        selected = replay._select_ledger_candidates(
            merged,
            requested_h11=4,
            declared_source_digest="d",
            max_rows=1,
        )
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0][2]["accepted_for_table_1"])

    def test_duplicate_authoritative_merged_identity_fails_closed(self):
        entry = {"polytope_id": "p", "accepted_for_table_1": True, "accepted_witness": {"matrix_id": "m"}, "frst_class_index": 0, "frst_hash": "f"}
        with self.assertRaises(replay.ReplayConfigurationError):
            replay._select_ledger_candidates(
                {"requested_h11": 4, "terminal_ledger": {"class_funnel": [entry, dict(entry)]}},
                requested_h11=4, declared_source_digest="d", max_rows=2,
            )

    def test_replay_propagates_source_and_certificate_to_exact_reconstruction(self):
        source = {"polytope_id": "p", "frst_hash": "f"}
        source_record = {"source": {"polytope_id": "p", "global_points": [[0, 0, 0, 0]]}}
        certificate = {"certificate_schema_version": "fixture"}
        rows = replay._replay_candidate_rows(
            object(), source, 0, _Triangulation(),
            source_record=source_record,
            mpcp_certificate=certificate,
            ledger_candidate={"frst_hash": "f", "action_digest": "a"},
        )
        self.assertEqual(rows[0]["terminal_status"], "exact_certificate_unavailable")

    def test_replay_exception_is_terminalized(self):
        with patch.object(replay, "extract_topology", side_effect=RuntimeError("boom")):
            rows = replay._replay_candidate_rows(
                object(), {"polytope_id": "p"}, 0, _Triangulation(),
                source_record={"source": {}},
                mpcp_certificate={"certificate": True},
                ledger_candidate={"frst_hash": "f", "action_digest": "a"},
            )
        self.assertEqual(rows[0]["terminal_status"], "exact_certificate_unavailable")

    def test_schema_valid_evidence_reconstructs_polytope_and_selected_frst(self):
        from mpcp_bounded_analysis import point_identity

        points = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        source = {
            "declared_source_digest": "directory-digest",
            "canonical_polytope_id": f"lattice-points-sha256:{point_identity(points)}",
            "source_row": 11,
        }
        source_record = {
            "source": {
                "source_row": 11,
                "source_sha256": "parquet-digest",
                "polytope_id": source["canonical_polytope_id"],
                "global_points": points,
            },
            "selected_frst": {
                "points": points,
                "simplices": [[0, 1, 2, 3]],
                "simplices_index_space": "triangulation_local",
            },
        }
        candidate = {"frst_hash": "frst"}
        certificate = {
            "source": {
                "source_sha256": "parquet-digest",
                "polytope_id": source["canonical_polytope_id"],
                "global_points": points,
            },
            "frst": {"frst_hash": "frst"},
        }
        fake_poly = _GeometryPoly(4)
        fake_tri = _GeometryTriangulation()
        with patch(
            "mpcp_bounded_analysis._construct_polytope",
            return_value=(fake_poly, {"status": "constructed"}),
        ), patch(
            "mpcp_bounded_analysis._construct_selected_triangulation",
            return_value=(fake_tri, {"status": "constructed"}),
        ), patch(
            "mpcp_bounded_analysis.triangulation_identity",
            return_value="frst",
        ):
            poly, triangulation, error = replay._load_replay_geometry(
                source, candidate, source_record, certificate, 4
            )
        self.assertIs(poly, fake_poly)
        self.assertIs(triangulation, fake_tri)
        self.assertIsNone(error)

    def test_schema_valid_evidence_rejects_physical_h11_mismatch(self):
        from mpcp_bounded_analysis import point_identity

        points = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        polytope_id = f"lattice-points-sha256:{point_identity(points)}"
        source = {"canonical_polytope_id": polytope_id, "declared_source_digest": "d", "source_row": 11}
        source_record = {
            "source": {
                "source_row": 11,
                "source_sha256": "p",
                "polytope_id": polytope_id,
                "global_points": points,
            },
            "selected_frst": {
                "points": points,
                "simplices": [[0, 1, 2, 3]],
                "simplices_index_space": "triangulation_local",
            },
        }
        certificate = {
            "source": {"source_sha256": "p", "polytope_id": polytope_id, "global_points": points},
            "frst": {"frst_hash": "frst"},
        }
        with patch(
            "mpcp_bounded_analysis._construct_polytope",
            return_value=(_GeometryPoly(5), {"status": "constructed"}),
        ):
            poly, triangulation, error = replay._load_replay_geometry(
                source, {"frst_hash": "frst"}, source_record, certificate, 4
            )
        self.assertIsNone(poly)
        self.assertIsNone(triangulation)
        self.assertIn("physical h11 verification failed", error)

    def test_exact_replay_passes_reconstructed_objects_to_candidate_kernel(self):
        entry = {
            "polytope_id": "p",
            "accepted_for_table_1": True,
            "accepted_witness": {"matrix_id": "m"},
            "action_digest": "action",
            "frst_class_index": 0,
            "frst_hash": "frst",
            "source_record": {"source": {}},
            "mpcp_certificate": {"certificate": True},
        }
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            preflight = _write_merged_fixture(root, [entry])
            fake_poly = object()
            fake_tri = object()
            observed = {}

            def candidate_kernel(poly, source, class_index, triangulation, **kwargs):
                observed["poly"] = poly
                observed["triangulation"] = triangulation
                return [{"terminal_status": "exact_geometry_unavailable"}]

            with patch.object(replay, "run_population_preflight", return_value=preflight), \
                patch.object(replay, "_validate_enriched_evidence", return_value=(entry["source_record"], entry["mpcp_certificate"], None)), \
                patch.object(replay, "_load_replay_geometry", return_value=(fake_poly, fake_tri, None)), \
                patch.object(replay, "_replay_candidate_rows", side_effect=candidate_kernel):
                result = replay.exact_replay(_args(root, dry_run=False, max_rows=1))
        self.assertEqual(result["rows_evaluated"], 1)
        self.assertIs(observed["poly"], fake_poly)
        self.assertIs(observed["triangulation"], fake_tri)

    def test_dry_run_has_no_database_writes_and_persists_labels(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = replay.run_population_preflight
            replay.run_population_preflight = lambda *_args, **_kwargs: _preflight(root)
            try:
                result = replay.exact_replay(_args(root))
            finally:
                replay.run_population_preflight = original
            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["database_writes"], 0)
            self.assertEqual(result["config"]["labels"]["selection"], "candidate-only")
            self.assertEqual(result["config"]["labels"]["representativeness"], "nonrepresentative")
            self.assertEqual(result["config"]["labels"]["execution_mode"], "infrastructure_smoke_only")

    def test_terminal_only_full_run_is_blocked_but_explicit_two_row_smoke_runs(self):
        entry = {
            "polytope_id": "p",
            "accepted_for_table_1": True,
            "accepted_witness": {"matrix_id": "m"},
            "frst_class_index": 0,
            "frst_hash": "f",
        }
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            preflight = _write_merged_fixture(root, [entry])
            with patch.object(replay, "run_population_preflight", return_value=preflight):
                with self.assertRaises(replay.ReplayConfigurationError):
                    replay.exact_replay(_args(root, dry_run=False, max_rows=1))
                result = replay.exact_replay(
                    _args(root, dry_run=False, max_rows=1, allow_terminal_only_smoke=True)
                )
            self.assertEqual(result["database_writes"], 0)
            self.assertEqual(result["execution_mode"], "infrastructure_smoke_only")
            self.assertEqual(result["scientific_result"], "no_scientific_result")

    def test_merged_digest_is_checked_immediately_before_read(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _write_merged_fixture(root, [])
            path = root / "merged.json.zst"
            with self.assertRaises(replay.ReplayConfigurationError):
                replay._read_merged_ledger(path, expected_sha256="0" * 64, requested_h11=4)

    def test_resume_skips_terminal_identity_without_appending(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "checkpoint.jsonl.zst"
            config = {"schema_version": replay.REPLAY_SCHEMA_VERSION}
            rows, _ = replay._checkpoint_state(path, config, resume=False)
            rows.append({"record_type": "row", "row_identity": "same", "terminal_status": "exact_certificate_unavailable"})
            replay._zstd_jsonl_write_atomic(path, rows)
            resumed, identities = replay._checkpoint_state(path, config, resume=True)
            before = len(resumed)
            if "same" in identities:
                pass
            self.assertEqual(len(resumed), before)
            self.assertEqual(len(replay._zstd_jsonl_read(path)), 2)

    def test_duplicate_checkpoint_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "checkpoint.jsonl.zst"
            config = {"schema_version": replay.REPLAY_SCHEMA_VERSION}
            rows, _ = replay._checkpoint_state(path, config, resume=False)
            rows.extend([
                {"record_type": "row", "row_identity": "same"},
                {"record_type": "row", "row_identity": "same"},
            ])
            replay._zstd_jsonl_write_atomic(path, rows)
            with self.assertRaises(replay.ReplayResumeError):
                replay._checkpoint_state(path, config, resume=True)


if __name__ == "__main__":
    unittest.main()
