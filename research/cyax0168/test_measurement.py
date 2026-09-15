#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from .measurement import (
        MeasurementError, condition_preconditioned_warm_cache, compare_manifest,
        hard_resource_envelope, inspect_materialization, one_query_execution_worker,
        pair_order_schedule, require_zero_sharing, ProcessMonitor,
    )
    from .host import ThermalObservation
except ImportError:  # pragma: no cover
    from measurement import (
        MeasurementError, condition_preconditioned_warm_cache, compare_manifest,
        hard_resource_envelope, inspect_materialization, one_query_execution_worker,
        pair_order_schedule, require_zero_sharing, ProcessMonitor,
    )
    from host import ThermalObservation


class MeasurementTests(unittest.TestCase):
    def test_manifest_cache_condition_and_post_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"a" * (8 * 1024 * 1024 + 17))
            before = inspect_materialization(root, backend="S")
            result = condition_preconditioned_warm_cache(before, reader=lambda path, size: (path.stat().st_size, before.files[0].sha256, 1_000_000.0))
            # The custom reader deliberately verifies only the single-file
            # fixture; pass stabilization remains deterministic.
            self.assertTrue(result.stabilized)
            after = inspect_materialization(root, backend="S")
            self.assertTrue(compare_manifest(before, after)["pass"])

    def test_mutation_and_unknown_clone_state_are_not_admissible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x").write_bytes(b"x")
            before = inspect_materialization(root, backend="S")
            (root / "x").write_bytes(b"changed")
            after = inspect_materialization(root, backend="S")
            self.assertFalse(compare_manifest(before, after)["pass"])
            with self.assertRaises(MeasurementError):
                require_zero_sharing(before)

    def test_hard_envelope_and_balanced_order(self):
        result = hard_resource_envelope(
            peak_rss_bytes=100, physical_ram_bytes=1000,
            logical_bytes=100, combined_logical_bytes=200,
            temporary_allocated_bytes=100, available_volume_capacity=1000,
            post_operation_available_capacity=500, total_volume_capacity=1000,
            cache_condition_pass=True, post_manifest_pass=True,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(pair_order_schedule(4, first_backend="S"), ("S", "G", "S", "G"))

    def test_query_worker_contract(self):
        self.assertTrue(one_query_execution_worker(worker_thread_names=("MainThread", "runtime-helper"))["valid"])
        self.assertFalse(one_query_execution_worker(worker_thread_names=("query-worker-1", "query-worker-2"))["valid"])

    def test_process_monitor_uses_direct_thermal_reader_by_default(self):
        calls = []
        def reader(*, phase):
            calls.append(phase)
            return ThermalObservation(phase, "nominal", observed_at_ns=len(calls))
        monitor = ProcessMonitor(power_source="AC", energy_mode="Automatic", thermal_reader=reader)
        sample = monitor.sample()
        self.assertEqual(calls, ["during"])
        self.assertEqual(sample.thermal_mechanism_version, "macos-foundation-nsprocessinfo-thermal-v1")
        self.assertTrue(monitor.final()["valid"])


if __name__ == "__main__":
    unittest.main()
