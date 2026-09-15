#!/usr/bin/env python3
"""Portable/macOS measurement primitives for CYAX-0168 G1.

The functions here measure only caller-supplied files and callables.  They do
not know how to open a backend, discover a fixture, or construct a decision
identity.  That separation is deliberate: G1 can validate cache/resource and
statistical machinery using calibration-only synthetic inputs.
"""

from __future__ import annotations

import gc
import hashlib
import os
import platform
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .host import (
        ProcessControlSample,
        ThermalObservation,
        make_control_sample,
        read_macos_thermal_state,
        validate_process_control,
    )
except ImportError:  # pragma: no cover - direct script/discovery mode
    from host import (
        ProcessControlSample,
        ThermalObservation,
        make_control_sample,
        read_macos_thermal_state,
        validate_process_control,
    )


class MeasurementError(ValueError):
    """Raised for malformed or invalid measurement state."""


READ_BUFFER_BYTES = 8 * 1024 * 1024
MIN_STABILIZATION_PASSES = 3
MAX_STABILIZATION_PASSES = 10
THROUGHPUT_RELATIVE_TOLERANCE = 0.05
TIMEOUT_NS = 120_000_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(READ_BUFFER_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileManifestEntry:
    relative_path: str
    logical_bytes: int
    sha256: str
    allocated_bytes: int
    sparse: bool | None = None
    clone_shared_bytes: int | None = None

    def identity_tuple(self) -> tuple[str, int, str]:
        return (self.relative_path, self.logical_bytes, self.sha256)

    def complete_tuple(self) -> tuple[Any, ...]:
        return (*self.identity_tuple(), self.allocated_bytes, self.sparse, self.clone_shared_bytes)


@dataclass(frozen=True)
class MaterializationManifest:
    backend: str
    root: str
    files: tuple[FileManifestEntry, ...]
    inspection_api_version: str = "posix-stat-v1"
    file_set_identity: str = ""
    post_measurement: bool = False

    def __post_init__(self) -> None:
        paths = [entry.relative_path for entry in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise MeasurementError("materialization files must be unique and sorted")
        if any(path.startswith("/") or ".." in Path(path).parts for path in paths):
            raise MeasurementError("materialization manifest paths must be relative")
        if any(entry.logical_bytes < 0 or entry.allocated_bytes < 0 for entry in self.files):
            raise MeasurementError("file sizes cannot be negative")
        if not self.file_set_identity:
            object.__setattr__(self, "file_set_identity", hashlib.sha256(_canonical_manifest(self.files)).hexdigest())

    @property
    def logical_materialization_bytes(self) -> int:
        return sum(entry.logical_bytes for entry in self.files)

    @property
    def allocated_materialization_bytes(self) -> int:
        return sum(entry.allocated_bytes for entry in self.files)

    @property
    def cache_identity(self) -> tuple[tuple[str, int, str], ...]:
        return tuple(entry.identity_tuple() for entry in self.files)

    @property
    def clone_shared_bytes(self) -> int | None:
        values = [entry.clone_shared_bytes for entry in self.files if entry.clone_shared_bytes is not None]
        return None if len(values) != len(self.files) else sum(values)


def _canonical_manifest(files: Sequence[FileManifestEntry]) -> bytes:
    return repr([entry.complete_tuple() for entry in files]).encode("utf-8")


def inspect_materialization(root: str | os.PathLike[str], *, backend: str, relative_paths: Iterable[str] | None = None, inspection_api_version: str | None = None) -> MaterializationManifest:
    """Freeze a complete regular-file manifest under ``root``.

    The returned root is diagnostic only; callers must not publish it as
    provenance.  APFS clone attribution is explicitly unknown unless a
    caller-supplied inspection implementation can establish zero sharing.
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise MeasurementError(f"materialization root is not a directory: {root!r}")
    if relative_paths is None:
        paths = [path.relative_to(base) for path in base.rglob("*") if path.is_file()]
    else:
        paths = [Path(path) for path in relative_paths]
    entries: list[FileManifestEntry] = []
    for relative in sorted(paths, key=lambda p: p.as_posix()):
        if relative.is_absolute() or ".." in relative.parts:
            raise MeasurementError("manifest path escapes materialization root")
        path = base / relative
        if not path.is_file() or path.is_symlink():
            raise MeasurementError(f"manifest path is not a regular file: {relative}")
        info = path.stat()
        allocated = int(info.st_blocks * 512) if hasattr(info, "st_blocks") else int(info.st_size)
        sparse = allocated < int(info.st_size)
        entries.append(FileManifestEntry(relative.as_posix(), int(info.st_size), _sha256_file(path), allocated, sparse, None))
    return MaterializationManifest(backend=backend, root=str(base), files=tuple(entries), inspection_api_version=inspection_api_version or ("apfs-stat-v1" if platform.system() == "Darwin" else "posix-stat-v1"))


def compare_manifest(before: MaterializationManifest, after: MaterializationManifest) -> dict[str, Any]:
    """Return the frozen pre/post immutability decision."""
    before_map = {entry.relative_path: entry for entry in before.files}
    after_map = {entry.relative_path: entry for entry in after.files}
    added = tuple(sorted(set(after_map) - set(before_map)))
    removed = tuple(sorted(set(before_map) - set(after_map)))
    changed = tuple(sorted(path for path in set(before_map) & set(after_map) if before_map[path].complete_tuple() != after_map[path].complete_tuple()))
    return {"pass": not added and not removed and not changed, "added": added, "removed": removed, "changed": changed}


def require_zero_sharing(manifest: MaterializationManifest) -> None:
    """Require detector-confirmed zero shared extents for disk attribution."""
    if manifest.clone_shared_bytes is None:
        raise MeasurementError("APFS clone/sparse capability check unavailable or indeterminate")
    if manifest.clone_shared_bytes != 0:
        raise MeasurementError("materialization has shared extents affecting allocated bytes")


def rebuild_or_exclude_disk(
    manifest: MaterializationManifest,
    rebuild: Callable[[str], MaterializationManifest] | None = None,
) -> MaterializationManifest:
    """Rebuild into a fresh clone-disabled destination or exclude the metric."""
    try:
        require_zero_sharing(manifest)
        return manifest
    except MeasurementError:
        if rebuild is None:
            raise MeasurementError("allocated disk comparison excluded: zero-sharing not established")
        rebuilt = rebuild(manifest.backend)
        require_zero_sharing(rebuilt)
        return rebuilt


@dataclass(frozen=True)
class ConditioningPass:
    pass_number: int
    bytes_read: int
    elapsed_ns: int
    throughput_bytes_per_second: float
    sha256_verified: bool
    exit_status: int = 0
    minor_faults: int | None = None
    major_faults: int | None = None


@dataclass(frozen=True)
class ConditioningResult:
    backend: str
    passes: tuple[ConditioningPass, ...]
    stabilized: bool
    pass_number: int | None
    cache_condition: str
    reason: str | None = None


@dataclass
class ProcessMonitor:
    """Small injectable monitor for the initial macOS validity controls."""
    power_source: str
    energy_mode: str
    # A supplied state is reserved for deterministic contract fixtures.  The
    # campaign path leaves it unset and reads the direct macOS route below.
    thermal_pressure: str | None = None
    memory_pressure: str = "normal"
    page_out_start: int = 0
    swap_io_start: int = 0
    samples: list[ProcessControlSample] = field(default_factory=list)
    monitor_gap: bool = False
    thermal_reader: Callable[..., ThermalObservation] = read_macos_thermal_state

    def sample(self, *, page_out: int | None = None, swap_io: int | None = None, descendants: Sequence[int] = (), competing_load_ok: bool = True) -> ProcessControlSample:
        page_delta = int((page_out if page_out is not None else self.page_out_start) - self.page_out_start)
        swap_delta = int((swap_io if swap_io is not None else self.swap_io_start) - self.swap_io_start)
        observation = None if self.thermal_pressure is not None else self.thermal_reader(phase="during")
        current = make_control_sample(
            power_source=self.power_source, energy_mode=self.energy_mode,
            thermal_pressure=(self.thermal_pressure if observation is None else observation.state),
            memory_pressure=self.memory_pressure,
            page_out_delta=page_delta, swap_io_delta=swap_delta,
            descendants=descendants, monitor_gap=self.monitor_gap,
            competing_load_ok=competing_load_ok,
        )
        if observation is not None:
            current.thermal_mechanism_version = observation.mechanism_version
            current.thermal_observation = observation.to_record()
        self.samples.append(current)
        return current

    def final(self) -> dict[str, Any]:
        if not self.samples:
            self.monitor_gap = True
        reports = [validate_process_control(sample) for sample in self.samples]
        return {"valid": bool(reports) and all(report["valid"] for report in reports) and not self.monitor_gap, "samples": tuple(reports)}


def _throughput_stable(values: Sequence[float]) -> bool:
    if len(values) < 3:
        return False
    recent = values[-3:]
    median = sorted(recent)[1]
    return median > 0 and (max(recent) - min(recent)) / median <= THROUGHPUT_RELATIVE_TOLERANCE


def condition_preconditioned_warm_cache(
    manifest: MaterializationManifest,
    *, first_read_backend: str | None = None,
    pass_limit: int = MAX_STABILIZATION_PASSES,
    reader: Callable[[Path, int], tuple[int, str]] | None = None,
) -> ConditioningResult:
    """Read every manifest file with 8 MiB buffers until three passes stabilize."""
    if pass_limit < MIN_STABILIZATION_PASSES or pass_limit > MAX_STABILIZATION_PASSES:
        raise MeasurementError("stabilization pass limit must be between 3 and 10")
    base = Path(manifest.root)
    results: list[ConditioningPass] = []
    throughputs: list[float] = []
    for pass_number in range(1, pass_limit + 1):
        started = time.monotonic_ns()
        total = 0
        verified = True
        synthetic_throughput: float | None = None
        for entry in manifest.files:
            path = base / entry.relative_path
            if reader is None:
                digest = hashlib.sha256()
                count = 0
                with path.open("rb") as handle:
                    while True:
                        block = handle.read(READ_BUFFER_BYTES)
                        if not block:
                            break
                        digest.update(block)
                        count += len(block)
                actual_digest = digest.hexdigest()
            else:
                read_result = reader(path, READ_BUFFER_BYTES)
                if len(read_result) == 2:
                    count, actual_digest = read_result
                elif len(read_result) == 3:
                    count, actual_digest, synthetic_throughput = read_result
                else:
                    raise MeasurementError("reader must return (bytes,digest) or (bytes,digest,throughput)")
            total += int(count)
            verified = verified and count == entry.logical_bytes and actual_digest == entry.sha256
        elapsed = max(1, time.monotonic_ns() - started)
        throughput = float(synthetic_throughput if synthetic_throughput is not None else total * 1_000_000_000 / elapsed)
        # A custom reader may provide a deterministic throughput for synthetic
        # conformance fixtures.  Real reads always use monotonic elapsed time.
        record = ConditioningPass(pass_number, total, elapsed, throughput, verified, 0)
        results.append(record)
        throughputs.append(throughput)
        if verified and _throughput_stable(throughputs):
            return ConditioningResult(manifest.backend, tuple(results), True, pass_number, "preconditioned-warm-cache")
        if not verified:
            return ConditioningResult(manifest.backend, tuple(results), False, None, "preconditioned-warm-cache", "short-read-or-digest-mismatch")
    return ConditioningResult(manifest.backend, tuple(results), False, None, "preconditioned-warm-cache", "throughput-did-not-stabilize-by-pass-10")


def condition_pair(
    manifest_s: MaterializationManifest,
    manifest_g: MaterializationManifest,
    *,
    first_backend: str = "S",
    reader: Callable[[Path, int], tuple[int, str] | tuple[int, str, float]] | None = None,
) -> dict[str, Any]:
    """Condition both manifests with an alternating first-read schedule."""
    if manifest_s.backend == manifest_g.backend:
        raise MeasurementError("paired manifests must have distinct backend labels")
    if first_backend not in {"S", "G"}:
        raise MeasurementError("first backend must be S or G")
    first, second = (manifest_s, manifest_g) if first_backend == "S" else (manifest_g, manifest_s)
    first_result = condition_preconditioned_warm_cache(first, reader=reader)
    second_result = condition_preconditioned_warm_cache(second, reader=reader)
    return {
        "first_backend": first_backend,
        "S": first_result if first_backend == "S" else second_result,
        "G": second_result if first_backend == "S" else first_result,
        "valid": first_result.stabilized and second_result.stabilized,
        "cache_condition": "preconditioned-warm-cache",
    }


def validate_non_mutating_pair(before: MaterializationManifest, after: MaterializationManifest) -> None:
    result = compare_manifest(before, after)
    if not result["pass"]:
        raise MeasurementError(f"post-measurement manifest changed: {result}")


def one_query_execution_worker(*, worker_thread_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validate that no more than one named query worker is active."""
    names = tuple(worker_thread_names or (thread.name for thread in threading.enumerate()))
    query_workers = tuple(name for name in names if "query" in name.lower() and "worker" in name.lower())
    return {"valid": len(query_workers) <= 1, "query_workers": query_workers}


def gc_before_measured_block() -> None:
    """Frozen policy: one collection immediately before each timed block."""
    gc.collect()


@dataclass(frozen=True)
class TimedCall:
    elapsed_ns: int
    value: Any = None
    timeout: bool = False
    breach: bool = False
    valid: bool = True


def timed_call(call: Callable[[], Any], *, timeout_ns: int = TIMEOUT_NS, monotonic_ns: Callable[[], int] = time.monotonic_ns) -> TimedCall:
    """Time a synchronous call with monotonic wall time and censored timeout."""
    gc_before_measured_block()
    started = monotonic_ns()
    value = call()
    elapsed = monotonic_ns() - started
    # A valid query reaching the frozen limit remains a censored observation;
    # callers classify the breach, rather than replacing it with missing data.
    timeout = elapsed >= timeout_ns
    return TimedCall(int(min(elapsed, timeout_ns) if timeout else elapsed), value, timeout=timeout, breach=timeout, valid=True)


def pair_order_schedule(size: int, *, first_backend: str = "S") -> tuple[str, ...]:
    """Balanced alternating first-backend schedule."""
    if size < 0:
        raise MeasurementError("schedule size cannot be negative")
    if first_backend not in {"S", "G"}:
        raise MeasurementError("first backend must be S or G")
    other = "G" if first_backend == "S" else "S"
    return tuple(first_backend if index % 2 == 0 else other for index in range(size))


def hard_resource_envelope(
    *, peak_rss_bytes: int, physical_ram_bytes: int,
    logical_bytes: int, combined_logical_bytes: int | None = None,
    temporary_allocated_bytes: int, available_volume_capacity: int,
    post_operation_available_capacity: int, total_volume_capacity: int,
    cache_condition_pass: bool, post_manifest_pass: bool,
    memory_pressure: str = "normal", page_out_delta: int = 0, swap_io_delta: int = 0,
) -> dict[str, Any]:
    """Evaluate one backend and the pair-level logical cache hard gates."""
    if physical_ram_bytes <= 0 or available_volume_capacity <= 0 or total_volume_capacity <= 0:
        raise MeasurementError("resource denominators must be positive")
    mem = peak_rss_bytes <= 0.25 * physical_ram_bytes and memory_pressure == "normal" and page_out_delta == 0 and swap_io_delta == 0
    disk = temporary_allocated_bytes <= 0.25 * available_volume_capacity and post_operation_available_capacity >= 0.20 * total_volume_capacity
    cache = (combined_logical_bytes if combined_logical_bytes is not None else logical_bytes) <= 0.25 * physical_ram_bytes and cache_condition_pass and post_manifest_pass
    return {"MEM_HARD": mem, "DISK_HARD": disk, "CACHE_HARD": cache, "valid": mem and disk and cache}


__all__ = [
    "MeasurementError", "READ_BUFFER_BYTES", "MaterializationManifest", "FileManifestEntry", "ManifestEntry", "CacheManifest",
    "ConditioningPass", "ConditioningResult", "ProcessMonitor", "TimedCall", "inspect_materialization",
    "compare_manifest", "require_zero_sharing", "rebuild_or_exclude_disk",
    "condition_preconditioned_warm_cache", "condition_warm_cache", "condition_pair", "validate_non_mutating_pair",
    "one_query_execution_worker", "gc_before_measured_block", "timed_call",
    "pair_order_schedule", "hard_resource_envelope", "TIMEOUT_NS",
]

# Short compatibility names used by calibration manifests and review tooling.
ManifestEntry = FileManifestEntry
CacheManifest = MaterializationManifest
condition_warm_cache = condition_preconditioned_warm_cache
