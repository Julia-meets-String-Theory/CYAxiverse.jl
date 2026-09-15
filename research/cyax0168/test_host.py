#!/usr/bin/env python3
from __future__ import annotations

import unittest

try:
    from .host import (
        HostError,
        HostManifest,
        ThermalObservation,
        THERMAL_MECHANISM_VERSION,
        collect_thermal_observations,
        freeze_host_manifest,
        load_frozen_host_manifest,
        make_control_sample,
        read_macos_thermal_state,
        require_single_host_manifest,
        validate_host_manifest,
        validate_process_control,
        validate_thermal_observations,
    )
except ImportError:  # pragma: no cover
    from host import (
        HostError,
        HostManifest,
        ThermalObservation,
        THERMAL_MECHANISM_VERSION,
        collect_thermal_observations,
        freeze_host_manifest,
        load_frozen_host_manifest,
        make_control_sample,
        read_macos_thermal_state,
        require_single_host_manifest,
        validate_host_manifest,
        validate_process_control,
        validate_thermal_observations,
    )


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

    def test_frozen_manifest_binds_capacity_terms_and_rejects_tamper_or_mixing(self):
        frozen = freeze_host_manifest(manifest())
        self.assertEqual(frozen["manifest_id"], frozen["manifest_sha256"])
        self.assertEqual(
            frozen["manifest"]["capacity_terms"]["ordinary_available_volume_capacity_bytes"]["term"],
            "preflight_available_volume_capacity",
        )
        restored = load_frozen_host_manifest(frozen)
        self.assertEqual(restored.identity, frozen["manifest_id"])
        tampered = dict(frozen)
        tampered["manifest"] = dict(frozen["manifest"], ordinary_available_volume_capacity_bytes=499_999)
        with self.assertRaises(HostError):
            load_frozen_host_manifest(tampered)
        other = manifest(ordinary_available_volume_capacity_bytes=400_000)
        with self.assertRaises(HostError):
            require_single_host_manifest((restored, other))

    def test_direct_macos_thermal_route_maps_all_states_and_unavailable(self):
        for value, state in enumerate(("nominal", "fair", "serious", "critical")):
            observation = read_macos_thermal_state(
                state_reader=lambda _value=value: _value,
                platform_name="Darwin", now_ns=iter((10, 20)).__next__, phase="during"
            )
            self.assertEqual(observation.state, state)
            self.assertEqual(observation.mechanism_version, THERMAL_MECHANISM_VERSION)
        self.assertEqual(observation.source, "Foundation.framework:NSProcessInfo.processInfo.thermalState")
        self.assertIn("objc_msgSend", observation.runtime)
        unavailable = read_macos_thermal_state(
            state_reader=lambda: 99,
            platform_name="Darwin", now_ns=iter((10, 20)).__next__,
        )
        self.assertFalse(validate_thermal_observations((
            ThermalObservation("pre", "unavailable", available=False, observed_at_ns=10, reason="command-failed"),
            ThermalObservation("during", "unavailable", available=False, observed_at_ns=20, reason="command-failed"),
            ThermalObservation("post", "unavailable", available=False, observed_at_ns=30, reason="command-failed"),
        ))["valid"])
        self.assertEqual(unavailable.state, "unavailable")

    def test_thermal_series_has_frozen_phases_timing_and_invalidation(self):
        seen = []
        sleeps = []
        def sample(*, phase):
            seen.append(phase)
            return ThermalObservation(phase, "nominal", observed_at_ns=len(seen))
        observations = collect_thermal_observations(sample=sample, sleep=sleeps.append)
        self.assertEqual(seen, ["pre", "during", "post"])
        self.assertEqual(len(sleeps), 2)
        self.assertTrue(validate_thermal_observations(observations)["valid"])
        fair = tuple(ThermalObservation(item.phase, "fair", observed_at_ns=item.observed_at_ns) for item in observations)
        self.assertIn("during:thermal-fair", validate_thermal_observations(fair)["reasons"])


if __name__ == "__main__":
    unittest.main()
