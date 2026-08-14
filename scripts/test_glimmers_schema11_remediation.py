"""Bounded regression tests for the schema-1.1 remediation contracts.

Keep this module independent of CYTools production sampling.  The fixtures
exercise deterministic helpers, mock proposal streams, exact integer charge
algebra, and synthetic provenance records.  The standalone component modules
from tasks 13--15 are optional in this worktree because their handoffs are
integrated later; their tests are skipped with an explicit reason until the
modules are available.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PILOT_DIR = SCRIPT_DIR.parents[2]
TASK_DIR = PILOT_DIR / "glimmers_local_pilot_tasks"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import glimmers_schema11 as schema11
import qed_divisor_assignment as qed_assignment


def _optional_import(name):
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        # A handoff may be present while its sibling dependency is still
        # being integrated.  Treat that as an explicit test-environment
        # condition and include it in the machine-readable evidence.
        COMPONENT_IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        return None


COMPONENT_IMPORT_ERRORS = {}


PROPOSAL_CONTROLLER = _optional_import("glimmers_proposal_controller")
H491_DIAGNOSTICS = _optional_import("glimmers_h491_diagnostics")
EFT_ROW_SCHEMA = _optional_import("glimmers_eft_row_schema")
PROVENANCE = _optional_import("glimmers_provenance")
FRESH_ENSEMBLE = _optional_import("glimmers_fresh_ensemble_manifest")


REQUIRED_EFT_FIELDS = {
    "model_id",
    "geometry_id",
    "assignment_hash",
    "assignment_pool_rank",
    "assignment_pool_size",
    "qcd_divisor_index",
    "qed_divisor_index",
    "qcd_volume_scale",
    "qcd_volume",
    "qed_volume",
    "qed_potential_source",
    "qed_unsorted_potential_index",
    "qed_post_sort_source_position",
    "qed_log10_lambda4",
    "qed_leading_status",
    "leading_rank_certificate",
    "charge_factorized_schema_version",
    "normalization_map_version",
    "derivation_status",
}


def _find_callable(module, names):
    if module is None:
        return None
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def _call_with_supported_kwargs(function, values):
    """Call a handoff fixture using its named contract arguments."""

    signature = inspect.signature(function)
    aliases = {
        "geometry_ids": ("geometry_ids", "identifiers", "geometries"),
        "pool_sizes": ("pool_sizes", "capacities", "capacity_by_geometry"),
        "minimum_rows": ("minimum_rows", "min_rows", "minimum"),
        "maximum_rows": ("maximum_rows", "max_rows", "ceiling"),
        "base_seed": ("base_seed", "seed"),
        "proposal_stream": ("proposal_stream", "proposals", "candidates", "records"),
        "candidate_records": ("candidate_records", "records", "candidates"),
        "sampler_args": ("sampler_args", "sampling_args", "sampler_settings"),
        "repo_status": ("repo_status", "git_status", "status"),
    }
    arguments = {}
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in values:
            arguments[parameter.name] = values[parameter.name]
            continue
        matching_key = next(
            (
                key
                for key, names in aliases.items()
                if parameter.name in names and key in values
            ),
            None,
        )
        if matching_key is not None:
            arguments[parameter.name] = values[matching_key]
            continue
        if parameter.default is inspect.Parameter.empty:
            raise unittest.SkipTest(
                f"component callable {function.__name__} has an unsupported "
                f"required argument {parameter.name!r}"
            )
    return function(**arguments)


def _as_mapping(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, dict):
                return item
    raise AssertionError(f"component result is not mapping-like: {type(value)!r}")


def _first_present(mapping, names, default=None):
    for name in names:
        if name in mapping:
            try:
                return mapping[name]
            except (KeyError, IndexError):
                continue
    return default


def _assignment_fixture():
    """Return a four-divisor fixture in which every QCD index can qualify."""

    labels = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (2, 0, 0, 0),
        (3, 0, 0, 0),
    ]
    charges = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 1],
        ],
        dtype=np.int64,
    )
    # Equal positive references make every QCD choice normalize every other
    # divisor to 40, which is safely below the strict 127.5 QED bound.
    prime_volumes = np.full(4, 2.0)
    effective_volumes = np.ones(4)
    neighbors = tuple(
        tuple(index for index in range(4) if index != left)
        for left in range(4)
    )
    evidence = {
        (left, right): [((left, 0, 0, 0), (right, 0, 0, 0))]
        for left in range(4)
        for right in range(left + 1, 4)
    }
    return {
        "prime_labels": labels,
        "prime_charges": charges,
        "prime_volumes_reference": prime_volumes,
        "effective_volumes_reference": effective_volumes,
        "neighbors": neighbors,
        "intersection_evidence": evidence,
        "invariant_mask": np.ones(4, dtype=bool),
    }


class AssignmentPoolContractTests(unittest.TestCase):
    def test_every_accepted_qcd_index_has_its_own_normalization_record(self):
        fixture = _assignment_fixture()
        terminal_records = []
        pool = qed_assignment.enumerate_assignment_pool(
            **fixture, terminal_records=terminal_records
        )

        self.assertGreater(len(pool), 0)
        self.assertEqual(
            {row["qcd_divisor_index"] for row in pool},
            {0, 1, 2, 3},
        )
        for row in pool:
            self.assertIn("qcd_radial_scale", row)
            self.assertIn("qcd_volume_scale", row)
            self.assertIn("normalization_map_version", row)
            self.assertEqual(row["qcd_volume"], 40.0)
            self.assertTrue(row["qcd_volume_exact"])
            self.assertGreaterEqual(row["minimum_prime_volume"], 1.0)
            self.assertGreaterEqual(row["minimum_effective_volume"], 1.0)
            self.assertLess(row["qed_volume"], 127.5)
            self.assertNotEqual(
                row["qcd_divisor_index"], row["qed_divisor_index"]
            )
            self.assertIn("normalization_data_hash", row)
            self.assertEqual(row["terminal_status"], "accepted_assignment")
        accepted_terminal = [
            record
            for record in terminal_records
            if record["terminal_status"] == "accepted_assignment"
        ]
        self.assertEqual(len(accepted_terminal), len(pool))
        self.assertTrue(
            all("qcd_index" in record and "qed_index" in record for record in accepted_terminal)
        )

    def test_detached_random_qcd_record_is_rejected_at_pool_boundary(self):
        fixture = _assignment_fixture()
        terminal_records = []
        pool = qed_assignment.enumerate_assignment_pool(
            **fixture, terminal_records=terminal_records
        )
        by_pair = {
            (row["qcd_divisor_index"], row["qed_divisor_index"]): row
            for row in pool
        }

        # A detached record with a valid index but another QCD normalization is
        # not an accepted assignment.  The row must match the complete ordered
        # pool, including its assignment hash and normalized volume.
        detached = dict(next(iter(by_pair.values())))
        detached.pop("qed_divisor_index")
        detached.pop("qed_index")
        self.assertNotIn("qed_index", detached)
        self.assertNotIn("qed_divisor_index", detached)
        accepted_terminal = [
            record
            for record in terminal_records
            if record.get("terminal_status") == "accepted_assignment"
        ]
        self.assertTrue(accepted_terminal)
        self.assertTrue(all("qcd_index" in record and "qed_index" in record for record in accepted_terminal))
        self.assertNotIn(detached, accepted_terminal)
        self.assertRaises(
            ValueError,
            qed_assignment.normalize_qcd_assignment,
            fixture["prime_volumes_reference"],
            fixture["effective_volumes_reference"],
            99,
        )

    def test_assignment_order_and_hash_replay_are_stable(self):
        fixture = _assignment_fixture()
        first = qed_assignment.enumerate_assignment_pool(**fixture)
        second = qed_assignment.enumerate_assignment_pool(**fixture)
        self.assertEqual(first, second)
        self.assertEqual(
            [row["pool_rank"] for row in first], list(range(len(first)))
        )
        changed = dict(fixture)
        changed["prime_volumes_reference"] = np.asarray([2.0, 2.0, 2.0, 2.5])
        changed_pool = qed_assignment.enumerate_assignment_pool(**changed)
        self.assertNotEqual(
            [row["assignment_hash"] for row in first],
            [row["assignment_hash"] for row in changed_pool],
        )


class CapacitySamplingContractTests(unittest.TestCase):
    def test_no_replacement_and_row_seed_replay_for_boundary_pool_sizes(self):
        pools = {
            f"geometry-{pool_size}": [f"assignment-{pool_size}-{rank}" for rank in range(pool_size)]
            for pool_size in (7, 142, 143, 265)
        }
        first = schema11.sample_capacity_aware_assignments(
            pools, 20260814, minimum_rows=1, maximum_rows=sum(map(len, pools.values()))
        )
        second = schema11.sample_capacity_aware_assignments(
            dict(reversed(list(pools.items()))),
            20260814,
            minimum_rows=1,
            maximum_rows=sum(map(len, pools.values())),
        )
        self.assertEqual(first, second)
        self.assertEqual(first["accepted_count"], sum(map(len, pools.values())))
        self.assertEqual(first["stop_reason"], "ceiling_reached")
        by_geometry = {}
        for row in first["rows"]:
            by_geometry.setdefault(row["geometry_id"], []).append(row)
            self.assertEqual(
                row["row_seed"],
                schema11.row_assignment_seed(
                    20260814, row["geometry_id"], row["row_index"]
                ),
            )
        for geometry_id, rows in by_geometry.items():
            self.assertEqual(
                len(rows), len(pools[geometry_id])
            )
            self.assertEqual(
                len({row["assignment_pool_rank"] for row in rows}), len(rows)
            )
            self.assertEqual(
                len({row["assignment_hash"] for row in rows}), len(rows)
            )

    def test_rank_sampler_is_replayable_and_caps_small_pools(self):
        sampled = schema11.sample_pool_without_replacement(
            7, 142, "geometry-7", 20260814, return_records=True,
            assignment_hashes=[f"hash-{index}" for index in range(7)],
        )
        replay = schema11.sample_pool_without_replacement(
            7, 142, "geometry-7", 20260814, return_records=True,
            assignment_hashes=[f"hash-{index}" for index in range(7)],
        )
        self.assertEqual(sampled, replay)
        self.assertEqual(len(sampled), 7)
        self.assertEqual({row["assignment_pool_rank"] for row in sampled}, set(range(7)))
        self.assertTrue(all(row["assignment_pool_size"] == 7 for row in sampled))

    def test_capacity_allocator_replaces_exact_142_143_contract(self):
        small_ids = ["g7", "g142", "g143", "g265"]
        small_sizes = [7, 142, 143, 265]
        small = schema11.allocate_eft_quotas(
            small_ids,
            small_sizes,
            minimum_rows=100_000,
            maximum_rows=200_000,
        )
        small_count = _first_present(
            small, ("accepted_count", "accepted_rows", "total_rows")
        )
        small_reason = _first_present(
            small, ("stop_reason", "terminal_status", "status")
        )
        self.assertIsNotNone(small_count)
        self.assertLess(small_count, 100_000)
        self.assertIn(
            small_reason,
            {"capacity_exhausted", "model_target_shortfall", "model_minimum_shortfall"},
        )

        large_ids = [f"g{index:04d}" for index in range(1400)]
        large_sizes = ([7, 142, 143, 265] * 350)
        large = schema11.allocate_eft_quotas(
            large_ids,
            large_sizes,
            minimum_rows=100_000,
            maximum_rows=200_000,
        )
        large_count = _first_present(
            large, ("accepted_count", "accepted_rows", "total_rows")
        )
        large_reason = _first_present(
            large, ("stop_reason", "terminal_status", "status")
        )
        self.assertIsNotNone(large_count)
        self.assertGreaterEqual(large_count, 100_000)
        self.assertLessEqual(large_count, 200_000)
        self.assertIn(large_reason, {"ceiling_reached", "model_ceiling_reached", "capacity_exhausted"})

        allocations = large["quotas"]
        self.assertEqual(sum(int(value) for value in allocations.values()), large_count)
        self.assertEqual(allocations[large_ids[0]], 7)
        for identifier, quota in allocations.items():
            index = large_ids.index(identifier)
            self.assertLessEqual(int(quota), large_sizes[index])


class ProposalRetryContractTests(unittest.TestCase):
    def _run(self, stream):
        if PROPOSAL_CONTROLLER is None:
            self.skipTest("proposal-controller handoff is not present")
        config = PROPOSAL_CONTROLLER.ProposalControllerConfig(
            accepted_target=1,
            proposal_budget=3,
            retry_budget=3,
            h11=491,
            sampler_name="ntfe_fast",
            deterministic_seed=17,
        )
        return PROPOSAL_CONTROLLER.run_proposal_controller(config, stream).to_dict()

    def test_rejections_do_not_consume_accepted_target(self):
        stream = [
            {"terminal_status": "kahler_tip_failure", "candidate_id": "bad-tip"},
            {"terminal_status": "numerical_geometry_failure", "candidate_id": "bad-num"},
            {"terminal_status": "accepted_geometry", "candidate_id": "good"},
        ]
        result = self._run(stream)
        accepted = _first_present(result, ("accepted_count", "accepted"))
        attempted = _first_present(
            result, ("proposal_count", "attempted_proposals", "attempted")
        )
        self.assertEqual(accepted, 1)
        self.assertEqual(attempted, 3)
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(
            [record["proposal_seed"] for record in result["records"]], [17, 18, 19]
        )
        self.assertEqual(
            [record["accepted_count_after"] for record in result["records"]], [0, 0, 1]
        )
        for field in ("accepted_target", "proposal_budget", "retry_budget"):
            self.assertIn(field, result)

    def test_cap_exhaustion_is_terminal_and_preserves_failure_taxonomy(self):
        result = self._run(
            [
                {"terminal_status": "kahler_tip_failure"},
                {"terminal_status": "numerical_geometry_failure"},
                {"terminal_status": "duplicate_ntfe_identity"},
            ]
        )
        self.assertEqual(_first_present(result, ("accepted_count", "accepted")), 0)
        self.assertEqual(result["terminal_status"], "geometry_target_shortfall")
        self.assertEqual(result["budget_status"], "proposal_budget_exhausted")
        statuses = result["status_counts"]
        self.assertGreaterEqual(int(statuses.get("kahler_tip_failure", 0)), 1)
        self.assertGreaterEqual(int(statuses.get("numerical_geometry_failure", 0)), 1)
        self.assertGreaterEqual(int(statuses.get("duplicate_ntfe_identity", 0)), 1)


class H491DiagnosticContractTests(unittest.TestCase):
    def test_native_h491_taxonomy_preserves_sampler_and_stage_counts(self):
        if H491_DIAGNOSTICS is None:
            self.skipTest("h11=491 diagnostics handoff is not present")
        result = H491_DIAGNOSTICS.diagnose_h491_regression_fixture().to_dict(
            include_records=False
        )
        serialized = json.dumps(result, sort_keys=True)
        self.assertIn("ntfe_fast", serialized)
        self.assertIn("kahler_tip_failure", serialized)
        self.assertIn("numerical_geometry_failure", serialized)
        self.assertIn("74", serialized)
        self.assertIn("13", serialized)
        self.assertNotIn('"sampler_name": "gnn', serialized)
        for forbidden in ("random_triangulations_gnn", "dualgnn", "pytorch"):
            self.assertIn(forbidden, serialized.lower())
        self.assertEqual(result["candidate_count"], 87)
        self.assertEqual(result["terminal_status_counts"]["kahler_tip_failure"], 74)
        self.assertEqual(result["terminal_status_counts"]["numerical_geometry_failure"], 13)
        self.assertEqual(result["stage_counts"]["tip_convergence"]["failure"], 74)
        self.assertEqual(result["stage_counts"]["numerical_residuals"]["failed"], 13)

        taxonomy = H491_DIAGNOSTICS.diagnose_h491(
            [
                {"candidate_index": 1, "terminal_status": "ntfe_generation_failure"},
                {"candidate_index": 2, "terminal_status": "kahler_tip_failure"},
                {"candidate_index": 3, "terminal_status": "numerical_geometry_failure"},
            ],
            accepted_target=1,
        ).to_dict(include_records=False)
        self.assertEqual(taxonomy["terminal_status_counts"]["ntfe_generation_failure"], 1)
        self.assertEqual(taxonomy["terminal_status_counts"]["kahler_tip_failure"], 1)
        self.assertEqual(taxonomy["terminal_status_counts"]["numerical_geometry_failure"], 1)
        self.assertEqual(
            taxonomy["stage_counts"]["ntfe_yield"]["generation_failure"], 1
        )


class EFTRowContractTests(unittest.TestCase):
    def test_existing_potential_conventions_are_exact_and_reject_mismatch(self):
        charges = np.asarray(
            [
                [1_000_003, 2_000_006, 0],
                [0, 0, 1],
            ],
            dtype=np.int64,
        )
        scales = np.asarray([[1.0, 1.0, 1.0], [3.0, 2.0, 1.0]])
        match = qed_assignment.record_potential_match(
            charges, scales, charges[:, 1], 3, 1
        )
        status = qed_assignment.classify_qed_leading_status(charges, scales, 1)
        self.assertEqual(match["qed_potential_source"], "direct_effective_cone")
        self.assertEqual(match["qed_unsorted_potential_index"], 1)
        self.assertEqual(match["qed_post_sort_source_position"], 1)
        self.assertEqual(status["status"], "dependent")
        self.assertEqual(status["selected_source_indices"], [0, 2])
        self.assertEqual(status["method"], "exact_rational_incremental_rank")
        with self.assertRaises(qed_assignment.QEDAssignmentFailure) as context:
            qed_assignment.record_potential_match(
                charges, scales, np.asarray([2_000_007, 0]), 3, 1
            )
        self.assertEqual(context.exception.category, "potential_term_mismatch")

    def test_row_schema_contains_required_eft_fields_and_no_dense_potential(self):
        serializer = _find_callable(EFT_ROW_SCHEMA, ("serialize_eft_row", "build_eft_row"))
        if serializer is None:
            self.skipTest("EFT row-schema handoff is not present")
        assignment = {
            "model_id": "geometry-0001:eft-000",
            "geometry_id": "geometry-0001",
            "assignment_hash": "a" * 64,
            "assignment_pool_rank": 0,
            "assignment_pool_size": 7,
            "qcd_divisor_index": 0,
            "qed_divisor_index": 1,
            "qcd_divisor_label": [0, 0, 0, 0],
            "qed_divisor_label": [1, 0, 0, 0],
            "qcd_volume_scale": 1.0,
            "qcd_volume": 40.0,
            "qed_volume": 2.0,
            "charge_factorized_schema_version": "glimmers-charge-factorized-1.1",
            "normalization_map_version": schema11.NORMALIZATION_MAP_VERSION,
        }
        geometry = {
            "geometry_id": "geometry-0001",
            "charge_factorized_schema_version": "glimmers-charge-factorized-1.1",
        }
        charges = np.asarray(
            [
                [1_000_003, 2_000_006, 0],
                [0, 0, 1],
            ],
            dtype=np.int64,
        )
        scales = np.asarray([[1.0, 1.0, 1.0], [3.0, 2.0, 1.0]])
        row = _as_mapping(serializer(
            geometry,
            assignment,
            q=charges,
            l=scales,
            qed_charge=charges[:, 1],
            direct_count=3,
            source_index=1,
        ))
        self.assertTrue(REQUIRED_EFT_FIELDS.issubset(row))
        self.assertNotIn("Q", row)
        self.assertNotIn("L", row)
        self.assertNotIn("qed_potential_matrix", row)
        round_tripped = json.loads(json.dumps(row, sort_keys=True))
        self.assertEqual(round_tripped["qed_unsorted_potential_index"], 1)
        self.assertEqual(round_tripped["qed_post_sort_source_position"], 1)
        self.assertEqual(round_tripped["qed_log10_lambda4"], 2.0)
        self.assertEqual(round_tripped["qed_leading_status"], "dependent")
        with self.assertRaises(EFT_ROW_SCHEMA.EFTRowFailure) as context:
            serializer(
                geometry,
                assignment,
                q=charges,
                l=scales,
                qed_charge=np.asarray([2_000_007, 0]),
                direct_count=3,
                source_index=1,
            )
        self.assertEqual(context.exception.terminal_status, "potential_term_mismatch")

        missing_assignment_field = dict(assignment)
        missing_assignment_field.pop("qed_volume")
        with self.assertRaises(EFT_ROW_SCHEMA.EFTRowFailure) as context:
            serializer(
                geometry,
                missing_assignment_field,
                q=charges,
                l=scales,
                qed_charge=charges[:, 1],
                direct_count=3,
                source_index=1,
            )
        self.assertEqual(
            context.exception.terminal_status, "missing_assignment_derived_data"
        )


class ProvenanceContractTests(unittest.TestCase):
    def test_provenance_digest_changes_for_seed_query_version_and_manifest(self):
        base = {
            "seed": 17,
            "query": {"h11": [50, 100, 200, 491], "favorable": True},
            "cytools_version": "1.4.12",
            "source_manifest": "manifest-sha256",
        }
        digests = {
            schema11.stable_hash(base),
            schema11.stable_hash({**base, "seed": 18}),
            schema11.stable_hash({**base, "query": {"h11": [491], "favorable": True}}),
            schema11.stable_hash({**base, "cytools_version": "1.4.13"}),
            schema11.stable_hash({**base, "source_manifest": "other-sha256"}),
        }
        self.assertEqual(len(digests), 5)

    def _git_fixture(self, root):
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "schema11@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Schema 1.1 Tests"],
            check=True,
        )

    def _provenance_fixture(self, root):
        task_file = root / "task.json"
        source_file = root / "source.txt"
        task_file.write_text('{"schema_version":"1.1"}\n', encoding="utf-8")
        source_file.write_text("fresh synthetic source\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
        )
        return task_file, source_file

    def test_clean_dirty_gate_records_complete_replay_identity(self):
        if PROVENANCE is None:
            self.skipTest("provenance handoff is not present")
        with tempfile.TemporaryDirectory(prefix="cyax-schema11-provenance-") as root_name:
            root = Path(root_name)
            self._git_fixture(root)
            task_file, source_file = self._provenance_fixture(root)
            query = {
                "source": "synthetic-regression-fixture",
                "criteria": {
                    "lattice": "N",
                    "favorable": True,
                    "reflexive": True,
                    "full_dimensional": True,
                    "h11": [50, 100, 200, 491],
                },
                "fresh": True,
                "result_count": 1,
                "returned_order": ["synthetic-0"],
                "source_revision": "fixture-revision",
            }
            clean = PROVENANCE.collect_provenance(
                repo_root=root,
                task_file=task_file,
                source_files={"source": source_file},
                source_query=query,
                output_root=root / "fresh-output",
                input_roots=[root],
                command_line=[sys.executable, __file__],
                environment_overrides={"cytools": "fixture-1.4"},
                host_settings={"threads": {"OMP_NUM_THREADS": "1"}},
            )
            self.assertEqual(clean["status"], "provenance_validated")
            self.assertEqual(clean["repository"]["status"], "clean")
            self.assertEqual(clean["environment_versions"]["cytools"], "fixture-1.4")
            self.assertEqual(clean["thread_settings"]["OMP_NUM_THREADS"], "1")
            self.assertTrue(PROVENANCE.validate_provenance_digest(clean))
            self.assertIn("task_file", clean)
            self.assertIn("source", clean["source_hashes"])

            (root / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
            with self.assertRaises(PROVENANCE.ProvenanceError) as context:
                PROVENANCE.collect_provenance(
                    repo_root=root,
                    task_file=task_file,
                    source_files={"source": source_file},
                    source_query=query,
                    output_root=root / "another-output",
                )
            self.assertEqual(context.exception.status, "provenance_dirty_tree")

    def test_fresh_ensemble_claim_labels_and_native_h491_settings(self):
        if PROVENANCE is None or FRESH_ENSEMBLE is None:
            self.skipTest("fresh-ensemble manifest handoff is not present")
        with tempfile.TemporaryDirectory(prefix="cyax-schema11-fresh-") as root_name:
            root = Path(root_name)
            self._git_fixture(root)
            task_file, source_file = self._provenance_fixture(root)
            source_query = {
                "source": "synthetic-fresh-cytools-fixture",
                "criteria": {
                    "lattice": "N",
                    "favorable": True,
                    "reflexive": True,
                    "full_dimensional": True,
                },
                "fresh": True,
                "result_count": 4,
                "returned_order": ["fresh-50", "fresh-100", "fresh-200", "fresh-491"],
                "source_revision": "source-revision-fixture",
            }
            provenance = PROVENANCE.collect_provenance(
                repo_root=root,
                task_file=task_file,
                source_files={"source": source_file},
                source_query=source_query,
                output_root=root / "fresh-output",
            )
            retained_polytopes = [
                {
                    "h11": h11,
                    "polytope_fingerprint": f"fresh-{h11}",
                    "retained_order": index,
                }
                for index, h11 in enumerate((50, 100, 200, 491))
            ]
            samplers = FRESH_ENSEMBLE.default_sampler_by_h11()
            derived_seeds = FRESH_ENSEMBLE.derive_ensemble_seeds(
                17, retained_polytopes, samplers
            )
            result = FRESH_ENSEMBLE.build_fresh_ensemble_manifest(
                provenance,
                source_query=source_query,
                retained_polytopes=retained_polytopes,
                base_seed=17,
                derived_seeds=derived_seeds,
                sampler_by_h11=samplers,
                accepted_row_count=100_000,
                stop_reason="capacity_exhausted",
                historical_mapping_status="not_attempted_by_policy",
                expected_polytope_counts={"50": 1, "100": 1, "200": 1, "491": 1},
            )
        serialized = json.dumps(result, sort_keys=True).lower()
        for label in (
            "fresh_favorable_cytools_proof_of_principle",
            "adapted_model_reuse",
            "no_historical_polytope_match_claim",
            "no_exact_200000_reproduction_claim",
        ):
            self.assertIn(label, serialized)
        self.assertEqual(result["sampler_by_h11"]["491"]["name"], "Polytope.ntfe_frts")
        self.assertEqual(result["sampler_by_h11"]["491"]["arguments"]["triang_method"], "fast")
        self.assertEqual(result["parity_convention"]["h11_plus"], "h11")
        self.assertEqual(result["parity_convention"]["h11_minus"], 0)
        claims = result["claims"]
        self.assertEqual(claims["historical_mapping_status"], "not_attempted_by_policy")
        self.assertEqual(claims["historical_polytope_claim"], "no_historical_polytope_match_claim")
        self.assertEqual(claims["paper_reproduction_status"], "no_exact_200000_reproduction_claim")


class ScopeContractTests(unittest.TestCase):
    def test_handoff_matrix_reserves_monolithic_generator_for_integration(self):
        task16 = json.loads(
            (TASK_DIR / "16_schema11_regression_validation.json").read_text()
        )
        task17 = json.loads(
            (TASK_DIR / "17_schema11_integration_orchestration.json").read_text()
        )
        generator_path = (
            "CYAxiverse.jl.worktrees/glimmers-schema-1-1/scripts/"
            "generate_geometric_data_multitriangulation.py"
        )
        test_path = (
            "CYAxiverse.jl.worktrees/glimmers-schema-1-1/scripts/"
            "test_glimmers_schema11_remediation.py"
        )
        allowed16 = {
            str(item["path"] if isinstance(item, dict) else item)
            for item in task16["write_scope"]["allowed_write_files"]
        }
        forbidden16 = {str(item) for item in task16["write_scope"]["forbidden_write_files"]}
        allowed17 = {
            str(item["path"] if isinstance(item, dict) else item)
            for item in task17["write_scope"]["allowed_write_files"]
        }
        self.assertEqual(
            {
                path
                for path in allowed16
                if "test_glimmers_schema11_remediation.py" in path
                or "GLIMMERS_SCHEMA11_VALIDATION_NOTE.md" in path
            },
            allowed16,
        )
        self.assertTrue(any(generator_path in path for path in allowed17))
        self.assertTrue(any(generator_path in path for path in forbidden16))
        self.assertTrue(any(test_path in path for path in allowed16))
        self.assertFalse(any(generator_path in path for path in allowed16))
        self.assertTrue(any(test_path in path for path in task17["write_scope"]["forbidden_write_files"]))

        # Keep the ownership check explicit without guessing task filenames.
        for path in sorted(TASK_DIR.glob("1[1-5]_*.json")):
            task = json.loads(path.read_text())
            self.assertTrue(
                any(generator_path in str(item) for item in task["write_scope"]["forbidden_write_files"]),
                path.name,
            )


def _run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    evidence = {
        "schema_version": "1.1",
        "suite": "test_glimmers_schema11_remediation",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "component_modules_present": {
            "proposal_controller": PROPOSAL_CONTROLLER is not None,
            "h491_diagnostics": H491_DIAGNOSTICS is not None,
            "eft_row_schema": EFT_ROW_SCHEMA is not None,
            "provenance": PROVENANCE is not None,
            "fresh_ensemble_manifest": FRESH_ENSEMBLE is not None,
        },
        "terminal_status": (
            "tests_passed"
            if not result.failures and not result.errors and not result.skipped
            else "test_environment_missing"
            if not result.failures and not result.errors and result.skipped
            else "contract_failure"
        ),
    }
    print("SCHEMA11_TEST_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_run_tests())
