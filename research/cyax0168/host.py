#!/usr/bin/env python3
"""Host and process validity controls for the CYAX-0168 G1 contract.

This module is deliberately backend-free.  It freezes replay-relevant host
context while excluding machine-unique values, and exposes the initial macOS
control vocabulary (AC power, an exact non-low Energy Mode, thermal/memory
pressure, page-out/swap deltas, and no descendants).  Unavailable optional
Mach fields are reported as diagnostics; they never become a validity claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
import sqlite3
import sys
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Sequence


class HostError(ValueError):
    """Raised when a host manifest or control sample is not admissible."""


NON_LOW_ENERGY_MODES = frozenset({"Automatic", "High Power"})
REQUIRED_PRESSURE_STATES = frozenset({"nominal", "normal"})
FORBIDDEN_MACHINE_FIELDS = frozenset({
    "hostname", "host_name", "serial", "serial_number", "account",
    "account_name", "username", "user", "machine_uuid", "udid",
})


def _is_ac_power(value: str) -> bool:
    token = str(value).strip().lower()
    return token in {"ac", "mains", "external", "ac power", "external power"} or token.startswith("ac ")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _reject_private_fields(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_MACHINE_FIELDS:
                raise HostError(f"host manifest contains machine-unique field {path}.{key}")
            _reject_private_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_private_fields(child, f"{path}[{index}]")


@dataclass(frozen=True)
class HostManifest:
    manifest_version: str = "cyax-host-manifest-v1"
    cpu_model_class: str = ""
    core_topology: str = ""
    physical_ram_bytes: int = 0
    operating_system: str = ""
    operating_system_build: str = ""
    filesystem_type: str = ""
    benchmark_volume_capacity_bytes: int = 0
    ordinary_available_volume_capacity_bytes: int = 0
    python_version: str = ""
    sqlite_version: str = ""
    sqlite_compile_options: tuple[str, ...] = ()
    ladybug_artifact: str = ""
    native_runtime_versions: Mapping[str, str] = field(default_factory=dict)
    measurement_tool_versions: Mapping[str, str] = field(default_factory=dict)
    power_source: str = ""
    energy_mode: str = ""
    host_api_versions: Mapping[str, str] = field(default_factory=dict)
    extra_context: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.manifest_version:
            raise HostError("manifest_version is required")
        for name in ("physical_ram_bytes", "benchmark_volume_capacity_bytes", "ordinary_available_volume_capacity_bytes"):
            if int(getattr(self, name)) <= 0:
                raise HostError(f"{name} must be positive")
        if self.ordinary_available_volume_capacity_bytes > self.benchmark_volume_capacity_bytes:
            raise HostError("ordinary available volume capacity exceeds volume capacity")
        if self.power_source and not _is_ac_power(self.power_source):
            raise HostError("primary campaign requires AC power")
        if self.energy_mode and self.energy_mode not in NON_LOW_ENERGY_MODES:
            raise HostError(f"unsupported or low Energy Mode {self.energy_mode!r}")
        _reject_private_fields(asdict(self))

    @property
    def identity(self) -> str:
        """Stable public identity, excluding any machine-unique locator."""
        return hashlib.sha256(_canonical_json(self.to_record())).hexdigest()

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["sqlite_compile_options"] = list(self.sqlite_compile_options)
        for key in ("native_runtime_versions", "measurement_tool_versions", "host_api_versions", "extra_context"):
            record[key] = dict(sorted(record[key].items()))
        return record


def validate_host_manifest(manifest: HostManifest, *, require_initial_macos: bool = False) -> dict[str, Any]:
    """Validate frozen campaign prerequisites and return a diagnostic report."""
    if require_initial_macos and (manifest.operating_system != "macOS" or "ARM64" not in manifest.cpu_model_class.upper()):
        raise HostError("initial reference campaign requires macOS ARM64")
    if not _is_ac_power(manifest.power_source):
        raise HostError("campaign host is not on AC power")
    if manifest.energy_mode not in NON_LOW_ENERGY_MODES:
        raise HostError("campaign host must freeze one exact non-low Energy Mode")
    if manifest.ordinary_available_volume_capacity_bytes <= 0:
        raise HostError("ordinary available volume capacity is unavailable")
    return {"valid": True, "manifest_id": manifest.identity, "private_fields": False}


def capture_host_manifest(
    *, power_source: str, energy_mode: str, cpu_model_class: str | None = None,
    core_topology: str | None = None, filesystem_type: str = "unknown",
    ladybug_artifact: str = "", measurement_tool_versions: Mapping[str, str] | None = None,
    host_api_versions: Mapping[str, str] | None = None,
) -> HostManifest:
    """Capture portable context without recording hostname/account/serial data."""
    usage = shutil.disk_usage(os.getcwd())
    system = platform.system()
    release = platform.release()
    machine = cpu_model_class or f"{platform.machine()}"
    if system == "Darwin":
        machine = machine if "ARM64" in machine.upper() else f"{machine}-ARM64" if machine in {"arm64", "aarch64"} else machine
    topology = core_topology or f"logical_cpus={os.cpu_count() or 1}"
    options = tuple(sorted(row[0] for row in sqlite3.connect(":memory:").execute("pragma compile_options")))
    return HostManifest(
        cpu_model_class=machine,
        core_topology=topology,
        physical_ram_bytes=int(_physical_ram_bytes()),
        operating_system="macOS" if system == "Darwin" else system,
        operating_system_build=release,
        filesystem_type=filesystem_type,
        benchmark_volume_capacity_bytes=int(usage.total),
        ordinary_available_volume_capacity_bytes=int(usage.free),
        python_version=platform.python_version(),
        sqlite_version=sqlite3.sqlite_version,
        sqlite_compile_options=options,
        ladybug_artifact=ladybug_artifact,
        measurement_tool_versions=dict(measurement_tool_versions or {}),
        power_source=power_source,
        energy_mode=energy_mode,
        host_api_versions=dict(host_api_versions or {"host": "python-platform-v1"}),
    )


def _physical_ram_bytes() -> int:
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    return 1


def ru_maxrss_bytes(usage: resource.struct_rusage | None = None) -> int:
    """Return macOS-byte/Linux-kilobyte ``ru_maxrss`` in common byte units."""
    value = (usage or resource.getrusage(resource.RUSAGE_SELF)).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def mach_task_diagnostics(pid: int | None = None) -> dict[str, Any]:
    """Best-effort versioned Mach metrics; unavailable values are explicit."""
    if platform.system() != "Darwin":
        return {"api_version": "mach-task-v1", "available": False, "reason": "not-macos"}
    # Calling task_info via ctypes is intentionally kept out of the validity
    # path.  A platform-specific collector can supply these fields later.
    return {"api_version": "mach-task-v1", "available": False, "reason": "collector-unavailable", "pid": pid}


@dataclass
class ProcessControlSample:
    power_source: str
    energy_mode: str
    thermal_pressure: str
    memory_pressure: str
    page_out_delta: int = 0
    swap_io_delta: int = 0
    descendants: tuple[int, ...] = ()
    monitor_gap: bool = False
    competing_load_ok: bool = True
    ru_maxrss_bytes: int = 0
    mach: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time.monotonic_ns)

    @property
    def valid(self) -> bool:
        return (
            _is_ac_power(self.power_source)
            and self.energy_mode in NON_LOW_ENERGY_MODES
            and self.thermal_pressure == "nominal"
            and self.memory_pressure == "normal"
            and self.page_out_delta == 0
            and self.swap_io_delta == 0
            and not self.descendants
            and not self.monitor_gap
            and self.competing_load_ok
        )


def validate_process_control(sample: ProcessControlSample, *, physical_ram_bytes: int | None = None) -> dict[str, Any]:
    """Map one process/control sample to the frozen validity decision."""
    reasons: list[str] = []
    if not _is_ac_power(sample.power_source): reasons.append("power-source")
    if sample.energy_mode not in NON_LOW_ENERGY_MODES: reasons.append("energy-mode")
    if sample.thermal_pressure != "nominal": reasons.append("thermal-pressure")
    if sample.memory_pressure != "normal": reasons.append("memory-pressure")
    if sample.page_out_delta != 0: reasons.append("page-out")
    if sample.swap_io_delta != 0: reasons.append("swap-io")
    if sample.descendants: reasons.append("descendant")
    if sample.monitor_gap: reasons.append("monitor-gap")
    if not sample.competing_load_ok: reasons.append("competing-load")
    if physical_ram_bytes is not None and sample.ru_maxrss_bytes > physical_ram_bytes * 0.25:
        reasons.append("memory-hard-envelope")
    return {"valid": not reasons, "reasons": tuple(reasons), "ru_maxrss_bytes": sample.ru_maxrss_bytes}


def child_pids(pid: int | None = None) -> tuple[int, ...]:
    """Return descendants using the host process API, with psutil fallback."""
    target = int(pid or os.getpid())
    try:
        import psutil  # type: ignore
        proc = psutil.Process(target)
        return tuple(sorted(child.pid for child in proc.children(recursive=True)))
    except (ImportError, OSError, RuntimeError):
        # ``pgrep -P`` is present on the initial macOS host and avoids adding
        # a runtime dependency to the benchmark process.  A command failure
        # is represented as an empty set only when the process has exited;
        # callers can mark the monitor gap separately.
        descendants: set[int] = set()
        frontier = [target]
        while frontier:
            parent = frontier.pop()
            try:
                completed = subprocess.run(
                    ["pgrep", "-P", str(parent)], capture_output=True, text=True,
                    check=False, timeout=1.0,
                )
            except (OSError, subprocess.SubprocessError):
                break
            children = []
            for line in completed.stdout.splitlines():
                try:
                    child = int(line.strip())
                except ValueError:
                    continue
                if child not in descendants:
                    descendants.add(child)
                    children.append(child)
            frontier.extend(children)
        return tuple(sorted(descendants))


def make_control_sample(
    *, power_source: str, energy_mode: str, thermal_pressure: str = "nominal",
    memory_pressure: str = "normal", page_out_delta: int = 0, swap_io_delta: int = 0,
    descendants: Sequence[int] | None = None, monitor_gap: bool = False,
    competing_load_ok: bool = True,
) -> ProcessControlSample:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ProcessControlSample(
        power_source=power_source, energy_mode=energy_mode,
        thermal_pressure=thermal_pressure, memory_pressure=memory_pressure,
        page_out_delta=int(page_out_delta), swap_io_delta=int(swap_io_delta),
        descendants=tuple(sorted(descendants or child_pids())),
        monitor_gap=monitor_gap, competing_load_ok=competing_load_ok,
        ru_maxrss_bytes=ru_maxrss_bytes(usage), mach=mach_task_diagnostics(os.getpid()),
    )


__all__ = [
    "HostError", "HostManifest", "ProcessControlSample", "NON_LOW_ENERGY_MODES",
    "capture_host_manifest", "validate_host_manifest", "capture_host", "validate_host", "ru_maxrss_bytes",
    "mach_task_diagnostics", "validate_process_control", "child_pids",
    "make_control_sample",
]

capture_host = capture_host_manifest
validate_host = validate_host_manifest
