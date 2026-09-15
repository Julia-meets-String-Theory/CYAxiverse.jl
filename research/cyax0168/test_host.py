#!/usr/bin/env python3
from __future__ import annotations

import unittest

try:
    from .host import HostError, HostManifest, make_control_sample, validate_host_manifest, validate_process_control
except ImportError:  # pragma: no cover
    from host import HostError, HostManifest, make_control_sample, validate_host_manifest, validate_process_control


def manifest(**kwargs):
    defaults = dict(
        cpu_model_class="Apple ARM64 class", core_topology="P=4,E=4", physical_ram_bytes=16 * 1024**3,
        operating_system="macOS", operating_system_build="synthetic", filesystem_type="APFS",
        benchmark_volume_capacity_bytes=1_000_000, ordinary_available_volume_capacity_bytes=500_000,
        python_version="3.14", sqlite_version="3", sqlite_compile_options=("THREADSAFE=1",),
        power_source="AC", energy_mode="Automatic", host_api_versions={"energy": "v1"},
    )
    defaults.update(kwargs)
    return HostManifest(**defaults)


class HostTests(unittest.TestCase):
    def test_manifest_identity_is_stable_and_private_free(self):
        first = manifest()
        second = manifest()
        self.assertEqual(first.identity, second.identity)
        self.assertTrue(validate_host_manifest(first, require_initial_macos=True)["valid"])
        with self.assertRaises(HostError):
            manifest(extra_context={"hostname": "private"})

    def test_low_power_and_non_ac_are_invalid(self):
        with self.assertRaises(HostError):
            manifest(energy_mode="Low Power")
        with self.assertRaises(HostError):
            validate_host_manifest(manifest(power_source="battery"))

    def test_control_sample_truth_table(self):
        good = make_control_sample(power_source="AC", energy_mode="Automatic")
        self.assertTrue(validate_process_control(good)["valid"])
        bad = make_control_sample(power_source="AC", energy_mode="Automatic", page_out_delta=1)
        self.assertIn("page-out", validate_process_control(bad)["reasons"])


if __name__ == "__main__":
    unittest.main()
