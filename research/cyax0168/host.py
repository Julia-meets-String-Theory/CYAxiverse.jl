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
import ctypes
import json
import os
import platform
import resource
import re
import shutil
import sqlite3
import sys
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class HostError(ValueError):
    """Raised when a host manifest or control sample is not admissible."""


HOST_MANIFEST_SCHEMA_VERSION = "cyax-host-manifest-v2"
CAPACITY_TERM_MAPPING_VERSION = "cyax-capacity-terms-v1"
CAPACITY_API = "shutil.disk_usage.free"
CAPACITY_DEFINITION = (
    "ordinary bytes available to an unprivileged process on the benchmark "
    "volume, as returned by the frozen capacity API"
)
NON_LOW_ENERGY_MODES = frozenset({"Automatic", "High Power"})
THERMAL_STATES = ("nominal", "fair", "serious", "critical")
THERMAL_MECHANISM_VERSION = "macos-foundation-nsprocessinfo-thermal-v1"
THERMAL_SOURCE = "Foundation.framework:NSProcessInfo.processInfo.thermalState"
THERMAL_RUNTIME = "libobjc.A.dylib:objc_getClass+sel_registerName+objc_msgSend"
THERMAL_NOTIFICATION = "NSProcessInfoThermalStateDidChange (not subscribed; polling is authoritative)"
THERMAL_POLLING = "synchronous ProcessInfo.thermalState read at each declared phase"
THERMAL_STATE_MAPPING = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
# ``command`` is retained as the historical observation field name; its
# frozen value is now a source/runtime identity, never a shell command.
THERMAL_COMMAND = (THERMAL_SOURCE, THERMAL_RUNTIME)
THERMAL_SAMPLE_TIMEOUT_SECONDS = 2.0
THERMAL_SAMPLE_INTERVAL_SECONDS = 1.0
THERMAL_OBSERVATION_PHASES = ("pre", "during", "post")
THERMAL_MAX_OBSERVATION_GAP_SECONDS = 5.0
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
    manifest_version: str = HOST_MANIFEST_SCHEMA_VERSION
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
    capacity_api: str = CAPACITY_API
    capacity_definition: str = CAPACITY_DEFINITION

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
        if not self.capacity_api or not self.capacity_definition:
            raise HostError("capacity API and definition are required")
        _reject_private_fields(asdict(self))

    @property
    def identity(self) -> str:
        """Stable public identity, excluding any machine-unique locator."""
        return hashlib.sha256(_canonical_json(self.to_record())).hexdigest()

    @property
    def manifest_id(self) -> str:
        return self.identity

    @property
    def capacity_terms(self) -> dict[str, Any]:
        return capacity_term_mapping(self)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["sqlite_compile_options"] = list(self.sqlite_compile_options)
        for key in ("native_runtime_versions", "measurement_tool_versions", "host_api_versions", "extra_context"):
            record[key] = dict(sorted(record[key].items()))
        record["capacity_terms"] = self.capacity_terms
        return record


def capacity_term_mapping(manifest: HostManifest) -> dict[str, Any]:
    """Return the frozen mapping from host fields to decision terms.

    Keeping both values and their roles in the hashed record prevents a
    preflight value from being silently substituted for the normative disk
    denominator.  The mapping is data, not a second source of capacity.
    """
    return {
        "mapping_version": CAPACITY_TERM_MAPPING_VERSION,
        "benchmark_volume_capacity_bytes": {
            "term": "total_volume_capacity",
            "value_bytes": int(manifest.benchmark_volume_capacity_bytes),
            "api": "shutil.disk_usage.total",
            "role": "post_operation_reserve_denominator",
        },
        "ordinary_available_volume_capacity_bytes": {
            "term": "preflight_available_volume_capacity",
            "value_bytes": int(manifest.ordinary_available_volume_capacity_bytes),
            "api": manifest.capacity_api,
            "definition": manifest.capacity_definition,
            "role": "disk_hard_and_disk_excess_allowance_denominator",
        },
    }


def _manifest_record_from_value(value: HostManifest | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, HostManifest):
        return value.to_record()
    if not isinstance(value, Mapping):
        raise HostError("host manifest must be a HostManifest or mapping")
    if "manifest" in value:
        value = value["manifest"]
    record = dict(value)
    _reject_private_fields(record)
    return record


def freeze_host_manifest(manifest: HostManifest) -> dict[str, Any]:
    """Create the one immutable, current-run host-manifest envelope.

    The hash covers the complete canonical manifest record, including the
    capacity-term mapping.  No wall-clock or machine-unique field is added.
    """
    if not isinstance(manifest, HostManifest):
        raise HostError("freeze_host_manifest requires a HostManifest")
    if manifest.manifest_version != HOST_MANIFEST_SCHEMA_VERSION:
        raise HostError("only the current host-manifest schema can be frozen")
    record = manifest.to_record()
    digest = hashlib.sha256(_canonical_json(record)).hexdigest()
    return {
        "schema_version": HOST_MANIFEST_SCHEMA_VERSION,
        "manifest": record,
        "manifest_id": digest,
        "manifest_sha256": digest,
    }


def load_frozen_host_manifest(value: Mapping[str, Any] | str | os.PathLike[str]) -> HostManifest:
    """Load and verify a frozen manifest envelope or JSON path."""
    if isinstance(value, (str, os.PathLike)):
        try:
            with Path(value).open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise HostError(f"cannot load frozen host manifest: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "manifest", "manifest_id", "manifest_sha256"
    }:
        raise HostError("frozen host manifest envelope has unexpected fields")
    if value["schema_version"] != HOST_MANIFEST_SCHEMA_VERSION:
        raise HostError("unsupported host manifest schema version")
    record = _manifest_record_from_value(value["manifest"])
    expected = hashlib.sha256(_canonical_json(record)).hexdigest()
    if value["manifest_id"] != expected or value["manifest_sha256"] != expected:
        raise HostError("frozen host manifest hash mismatch")
    required = set(HostManifest.__dataclass_fields__) | {"capacity_terms"}
    if set(record) != required:
        raise HostError("host manifest record has unexpected or missing fields")
    try:
        manifest = HostManifest(**{key: record[key] for key in HostManifest.__dataclass_fields__})
    except (TypeError, ValueError) as exc:
        raise HostError(f"invalid frozen host manifest: {exc}") from exc
    if record["capacity_terms"] != capacity_term_mapping(manifest):
        raise HostError("capacity-term mapping does not match manifest values")
    if manifest.identity != expected:
        raise HostError("manifest identity does not match frozen hash")
    return manifest


def require_single_host_manifest(
    manifests: Sequence[HostManifest | Mapping[str, Any]],
) -> str:
    """Return one identity, rejecting paired runs that mix host contexts."""
    if not manifests:
        raise HostError("at least one host manifest is required")
    identities = {
        (manifest.identity if isinstance(manifest, HostManifest)
         else load_frozen_host_manifest(manifest).identity)
        for manifest in manifests
    }
    if len(identities) != 1:
        raise HostError("mixed host contexts are not admissible")
    return next(iter(identities))


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
    return {
        "valid": True,
        "manifest_id": manifest.identity,
        "manifest_sha256": manifest.identity,
        "capacity_terms": capacity_term_mapping(manifest),
        "private_fields": False,
    }


def capture_host_manifest(
    *, power_source: str, energy_mode: str, cpu_model_class: str | None = None,
    core_topology: str | None = None, filesystem_type: str = "unknown",
    ladybug_artifact: str = "", measurement_tool_versions: Mapping[str, str] | None = None,
    host_api_versions: Mapping[str, str] | None = None,
    capacity_api: str = CAPACITY_API,
    capacity_definition: str = CAPACITY_DEFINITION,
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
        capacity_api=capacity_api,
        capacity_definition=capacity_definition,
    )


def _physical_ram_bytes() -> int:
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            pass
    return 1


@dataclass(frozen=True)
class ThermalObservation:
    """One direct macOS thermal-pressure observation.

    ``state=unavailable`` is intentionally distinct from ``nominal``.  It
    fails the control gate and cannot be treated as an inferred pass.
    """

    phase: str
    state: str
    mechanism_version: str = THERMAL_MECHANISM_VERSION
    command: tuple[str, ...] = THERMAL_COMMAND
    available: bool = True
    returncode: int | None = 0
    observed_at_ns: int = 0
    elapsed_ns: int = 0
    reason: str | None = None
    output_sha256: str | None = None
    source: str = THERMAL_SOURCE
    runtime: str = THERMAL_RUNTIME
    notification: str = THERMAL_NOTIFICATION
    polling: str = THERMAL_POLLING

    def __post_init__(self) -> None:
        if self.state not in {*THERMAL_STATES, "unavailable"}:
            raise HostError(f"unknown thermal state {self.state!r}")
        if self.phase not in THERMAL_OBSERVATION_PHASES:
            raise HostError(f"unknown thermal observation phase {self.phase!r}")
        if self.state == "unavailable" and self.available:
            raise HostError("unavailable thermal observation cannot be marked available")
        if self.source != THERMAL_SOURCE or self.runtime != THERMAL_RUNTIME:
            raise HostError("thermal observation source/runtime identity is not frozen")
        if self.notification != THERMAL_NOTIFICATION or self.polling != THERMAL_POLLING:
            raise HostError("thermal observation timing identity is not frozen")

    def to_record(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "state": self.state,
            "mechanism_version": self.mechanism_version,
            "command": list(self.command),
            "available": self.available,
            "returncode": self.returncode,
            "observed_at_ns": self.observed_at_ns,
            "elapsed_ns": self.elapsed_ns,
            "reason": self.reason,
            "output_sha256": self.output_sha256,
            "source": self.source,
            "runtime": self.runtime,
            "notification": self.notification,
            "polling": self.polling,
        }


def _foundation_process_info_thermal_state() -> int:
    """Read ``NSProcessInfo.thermalState`` through the Foundation API.

    This deliberately uses the Objective-C runtime instead of a shell tool or
    an inferred metric.  The handles and selectors are fixed by the mechanism
    identity above and the returned integer is mapped only by the frozen enum
    table.  The function is called only on macOS; loader errors invalidate the
    observation rather than being converted to nominal.
    """
    foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    get_class = objc.objc_getClass
    get_class.argtypes = [ctypes.c_char_p]
    get_class.restype = ctypes.c_void_p
    register = objc.sel_registerName
    register.argtypes = [ctypes.c_char_p]
    register.restype = ctypes.c_void_p
    send = objc.objc_msgSend
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    send.restype = ctypes.c_void_p
    process_info_class = get_class(b"NSProcessInfo")
    process_info = send(process_info_class, register(b"processInfo"))
    if not process_info:
        raise RuntimeError("NSProcessInfo.processInfo returned nil")
    # ``thermalState`` returns an NSUInteger, not an Objective-C object.  A
    # separate typed view of objc_msgSend is required; using c_void_p would
    # turn the valid nominal value 0 into Python ``None``.
    send_unsigned = ctypes.CFUNCTYPE(
        ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_void_p
    )(ctypes.cast(send, ctypes.c_void_p).value)
    state = send_unsigned(process_info, register(b"thermalState"))
    # Keep the Foundation handle live until after the message dispatch.
    if not foundation:
        raise RuntimeError("Foundation.framework failed to load")
    return int(state)


def read_macos_thermal_state(
    *, state_reader: Callable[[], int] | None = None,
    platform_name: str | None = None,
    now_ns: Callable[[], int] = time.monotonic_ns,
    phase: str = "during",
) -> ThermalObservation:
    """Read macOS thermal pressure through Foundation ``NSProcessInfo``.

    The API source, Objective-C runtime, enum mapping, timeout, and
    unavailable-state behavior are fixed constants. ``state_reader`` and
    ``platform_name`` are injectable only for focused tests and preflight
    diagnostics; no unavailable result is inferred as nominal.
    """
    if phase not in THERMAL_OBSERVATION_PHASES:
        raise HostError(f"unknown thermal observation phase {phase!r}")
    started = int(now_ns())
    if (platform_name or platform.system()) != "Darwin":
        return ThermalObservation(
            phase=phase, state="unavailable", available=False,
            observed_at_ns=started, reason="not-macos",
        )
    try:
        reader = state_reader or _foundation_process_info_thermal_state
        raw_value = int(reader())
        state = THERMAL_STATE_MAPPING.get(raw_value, "unavailable")
        finished = int(now_ns())
        reason = None if state != "unavailable" else f"unknown-nsprocessinfo-enum:{raw_value}"
        raw = str(raw_value).encode("ascii")
        return ThermalObservation(
            phase=phase, state=state, available=state != "unavailable",
            returncode=None, observed_at_ns=finished,
            elapsed_ns=max(0, finished - started), reason=reason,
            output_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        finished = int(now_ns())
        return ThermalObservation(
            phase=phase, state="unavailable", available=False,
            returncode=None, observed_at_ns=finished,
            elapsed_ns=max(0, finished - started), reason=type(exc).__name__,
        )


def collect_thermal_observations(
    *, sample: Callable[..., ThermalObservation] = read_macos_thermal_state,
    sleep: Callable[[float], None] = time.sleep,
    phases: Sequence[str] = THERMAL_OBSERVATION_PHASES,
    interval_seconds: float = THERMAL_SAMPLE_INTERVAL_SECONDS,
) -> tuple[ThermalObservation, ...]:
    """Collect predeclared pre/during/post observations at a fixed interval."""
    if tuple(phases) != THERMAL_OBSERVATION_PHASES:
        raise HostError("thermal observation phases are frozen as pre/during/post")
    if interval_seconds < 0 or interval_seconds > THERMAL_MAX_OBSERVATION_GAP_SECONDS:
        raise HostError("thermal observation interval exceeds the frozen bound")
    observations: list[ThermalObservation] = []
    for index, phase in enumerate(phases):
        observation = sample(phase=phase)
        if observation.phase != phase:
            raise HostError("thermal sampler returned the wrong observation phase")
        observations.append(observation)
        if index + 1 < len(phases):
            sleep(interval_seconds)
    return tuple(observations)


def validate_thermal_observations(
    observations: Sequence[ThermalObservation],
    *, max_gap_seconds: float = THERMAL_MAX_OBSERVATION_GAP_SECONDS,
) -> dict[str, Any]:
    """Apply the frozen nominal-only thermal series gate."""
    reasons: list[str] = []
    expected = list(THERMAL_OBSERVATION_PHASES)
    actual = [item.phase for item in observations]
    if actual != expected:
        reasons.append("observation-phases")
    if not observations:
        reasons.append("observation-empty")
    mechanisms = {item.mechanism_version for item in observations}
    if mechanisms != {THERMAL_MECHANISM_VERSION}:
        reasons.append("mechanism-version")
    if any(tuple(item.command) != THERMAL_COMMAND for item in observations):
        reasons.append("mechanism-command")
    if any(item.source != THERMAL_SOURCE or item.runtime != THERMAL_RUNTIME for item in observations):
        reasons.append("source-runtime")
    if any(item.notification != THERMAL_NOTIFICATION or item.polling != THERMAL_POLLING for item in observations):
        reasons.append("notification-polling")
    for item in observations:
        if not item.available or item.state == "unavailable":
            reasons.append(f"{item.phase}:unavailable")
        elif item.state != "nominal":
            reasons.append(f"{item.phase}:thermal-{item.state}")
        if item.elapsed_ns > int(THERMAL_SAMPLE_TIMEOUT_SECONDS * 1e9):
            reasons.append(f"{item.phase}:sample-timeout")
    if len(observations) > 1:
        bound_ns = int(max_gap_seconds * 1e9)
        for previous, current in zip(observations, observations[1:]):
            if current.observed_at_ns < previous.observed_at_ns:
                reasons.append("observation-order")
            elif current.observed_at_ns - previous.observed_at_ns > bound_ns:
                reasons.append("observation-gap")
    return {
        "valid": not reasons,
        "reasons": tuple(dict.fromkeys(reasons)),
        "mechanism_version": THERMAL_MECHANISM_VERSION,
        "states": tuple(item.state for item in observations),
        "timing": {
            "sample_timeout_seconds": THERMAL_SAMPLE_TIMEOUT_SECONDS,
            "interval_seconds": THERMAL_SAMPLE_INTERVAL_SECONDS,
            "max_gap_seconds": max_gap_seconds,
            "phases": tuple(THERMAL_OBSERVATION_PHASES),
            "source": THERMAL_SOURCE,
            "runtime": THERMAL_RUNTIME,
            "state_mapping": dict(THERMAL_STATE_MAPPING),
            "notification": THERMAL_NOTIFICATION,
            "polling": THERMAL_POLLING,
            "transition_policy": "any fair/serious/critical or unavailable observation invalidates",
        },
    }


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
    thermal_mechanism_version: str = ""
    thermal_observation: Mapping[str, Any] = field(default_factory=dict)

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
    return {
        "valid": not reasons,
        "reasons": tuple(reasons),
        "ru_maxrss_bytes": sample.ru_maxrss_bytes,
        "thermal_mechanism_version": sample.thermal_mechanism_version,
        "thermal_observation": dict(sample.thermal_observation),
    }


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


def make_direct_control_sample(
    *, power_source: str, energy_mode: str, memory_pressure: str = "normal",
    page_out_delta: int = 0, swap_io_delta: int = 0,
    descendants: Sequence[int] | None = None, monitor_gap: bool = False,
    competing_load_ok: bool = True,
    thermal_reader: Callable[..., ThermalObservation] = read_macos_thermal_state,
) -> ProcessControlSample:
    """Build a control sample using the direct macOS thermal observation."""
    observation = thermal_reader(phase="during")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ProcessControlSample(
        power_source=power_source,
        energy_mode=energy_mode,
        thermal_pressure=observation.state,
        memory_pressure=memory_pressure,
        page_out_delta=int(page_out_delta),
        swap_io_delta=int(swap_io_delta),
        descendants=tuple(sorted(descendants or child_pids())),
        monitor_gap=monitor_gap,
        competing_load_ok=competing_load_ok,
        ru_maxrss_bytes=ru_maxrss_bytes(usage),
        mach=mach_task_diagnostics(os.getpid()),
        thermal_mechanism_version=observation.mechanism_version,
        thermal_observation=observation.to_record(),
    )


__all__ = [
    "HostError", "HostManifest", "ProcessControlSample", "NON_LOW_ENERGY_MODES",
    "HOST_MANIFEST_SCHEMA_VERSION", "CAPACITY_TERM_MAPPING_VERSION", "CAPACITY_API",
    "CAPACITY_DEFINITION", "capacity_term_mapping", "freeze_host_manifest",
    "load_frozen_host_manifest", "require_single_host_manifest", "capture_host_manifest",
    "validate_host_manifest", "capture_host", "validate_host", "ru_maxrss_bytes",
    "THERMAL_STATES", "THERMAL_MECHANISM_VERSION", "THERMAL_SOURCE", "THERMAL_RUNTIME",
    "THERMAL_NOTIFICATION", "THERMAL_POLLING", "THERMAL_STATE_MAPPING", "THERMAL_COMMAND",
    "THERMAL_SAMPLE_TIMEOUT_SECONDS", "THERMAL_SAMPLE_INTERVAL_SECONDS",
    "THERMAL_OBSERVATION_PHASES", "THERMAL_MAX_OBSERVATION_GAP_SECONDS",
    "ThermalObservation", "read_macos_thermal_state",
    "collect_thermal_observations", "validate_thermal_observations",
    "mach_task_diagnostics", "validate_process_control", "child_pids",
    "make_control_sample", "make_direct_control_sample",
]

capture_host = capture_host_manifest
validate_host = validate_host_manifest
