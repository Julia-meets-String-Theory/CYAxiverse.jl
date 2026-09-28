#!/usr/bin/env python3
"""Validation-only CYAX-0115 Uniform-FRST experiment driver.

This script is intentionally separate from the production sampler. It first
enumerates and freezes exact small supports, then binds all sampler seeds and
tests in a content-addressed manifest before running the public CYTools process.
"""

from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import hashlib
import inspect
import importlib.metadata
import io
import json
import os
import signal
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_VALIDATION_ROOT = REPOSITORY_ROOT / "validation" / "cyax_0115_uniform_frst"
REGISTRATION_PATH = DEFAULT_VALIDATION_ROOT / "preregistration_candidates.json"
CYTOOLS_SITE_PACKAGES = Path(
    "/opt/homebrew/Caskroom/miniforge/base/envs/cytools/lib/python3.14/site-packages"
)
EXPECTED_SOURCE_BLOBS = {
    "polytope.py": "c54a01651b850842128df00589fef981cf09c05c",
    "triangulation.py": "d83758d7c10ba9092fac66453a378e39e314d7a2",
}


class ValidationError(RuntimeError):
    """A frozen protocol, source, or artifact invariant failed."""


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob_sha1(contents: bytes) -> str:
    header = f"blob {len(contents)}\0".encode("ascii")
    return hashlib.sha1(header + contents).hexdigest()


def atomic_create(path: Path, contents: bytes) -> None:
    """Create one artifact without replacing an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    except FileExistsError as exc:
        raise ValidationError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_json_create(path: Path, value) -> None:
    atomic_create(path, canonical_json_bytes(value) + b"\n")


def append_jsonl_fsync(path: Path, value) -> None:
    """Append one durable progress event for recovery after a hard kill."""
    encoded = canonical_json_bytes(value) + b"\n"
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def run_bounded_process(command, timeout_seconds: float):
    """Run a process group with a hard wall-clock cap, including blocked C calls."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            tail_out, tail_err = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            tail_out, tail_err = process.communicate()
        stdout = tail_out or stdout
        stderr = tail_err or stderr
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return {
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "elapsed_wall_seconds": time.monotonic() - started,
    }


def read_progress_journal(path: Path):
    """Read complete durable lines and retain the identity of any torn tail."""
    if not path.is_file():
        return {
            "events": [],
            "journal_sha256": None,
            "unterminated_tail_sha256": None,
            "unterminated_tail_bytes": 0,
        }
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events = []
    tail = b""
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            tail = line
            if index != len(lines) - 1:
                raise ValidationError(f"non-final unterminated progress line in {path}")
            break
        events.append(json.loads(line))
    return {
        "events": events,
        "journal_sha256": sha256_bytes(raw),
        "unterminated_tail_sha256": sha256_bytes(tail) if tail else None,
        "unterminated_tail_bytes": len(tail),
    }


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_registration(path: Path = REGISTRATION_PATH):
    registration = load_json(path)
    if registration.get("schema") != "CYAX-0115-r4-preregistration-candidates-v1":
        raise ValidationError(f"unsupported registration schema in {path}")
    if registration.get("status") != "FROZEN_BEFORE_EXACT_SUPPORT_ENUMERATION":
        raise ValidationError("candidate registration is not frozen")
    return registration


def bootstrap_cytools(cache_dir: Path | None = None):
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    if str(CYTOOLS_SITE_PACKAGES) not in sys.path:
        sys.path.insert(0, str(CYTOOLS_SITE_PACKAGES))
    import cytools
    import cytools.polytope as polytope_module
    import cytools.triangulation as triangulation_module
    import numpy as np

    return cytools, polytope_module, triangulation_module, np


def environment_provenance(cytools, numpy_module):
    source_root = Path(cytools.__file__).resolve().parent
    source_hashes = {
        name: git_blob_sha1((source_root / name).read_bytes())
        for name in ("polytope.py", "triangulation.py")
    }
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": numpy_module.__version__,
        "cytools_version": importlib.metadata.version("cytools"),
        "installed_cytools_git_blob_sha": source_hashes,
        "platform": platform.system() + "-" + platform.machine(),
        "enumeration_backend": {
            "polytope_all_triangulations_backend_argument": None,
            "triangulation_engine": "bundled CYTools triangulumancer extension",
            "regularity_backend_argument": None,
            "regularity_optimizer_default": "highs",
            "topcom_executed": False,
        },
    }


def command_version(executable: str):
    path = shutil.which(executable)
    if path is None:
        return {"available": False}
    try:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        output = (result.stdout + result.stderr).strip()
        return {
            "available": True,
            "executable": executable,
            "version_output": output[:500],
            "version_exit_code": result.returncode,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "executable": executable, "version_error": type(exc).__name__}


def verify_source_bindings(registration, environment):
    expected = registration["source_bindings"]
    for key in ("cytools_version", "python_version", "numpy_version"):
        if environment[key] != expected[key]:
            raise ValidationError(
                f"source/environment drift for {key}: expected {expected[key]}, got {environment[key]}"
            )
    for filename, expected_blob in EXPECTED_SOURCE_BLOBS.items():
        actual_blob = environment["installed_cytools_git_blob_sha"][filename]
        if actual_blob != expected_blob:
            raise ValidationError(
                f"installed CYTools source drift for {filename}: {actual_blob} != {expected_blob}"
            )


def canonical_point_rows(poly):
    return sorted(tuple(int(value) for value in row) for row in poly.points().tolist())


def polytope_identity(poly):
    point_rows = canonical_point_rows(poly)
    return "lattice-points-sha256:" + sha256_bytes(canonical_json_bytes(point_rows))


def canonical_simplices(triangulation):
    rows = triangulation.simplices(as_indices=True).tolist()
    return sorted(tuple(sorted(int(value) for value in row)) for row in rows)


def full_triangulation_identity(triangulation):
    return sha256_bytes(canonical_json_bytes(canonical_simplices(triangulation)))


def canonical_two_face_rows(triangulation):
    face_simplices = triangulation.simplices(
        on_faces_dim=2,
        split_by_face=True,
        as_np_array=False,
        as_indices=True,
    )
    canonical_faces = []
    for simplices in face_simplices:
        simplices = list(simplices)
        if simplices and isinstance(simplices[0], (set, frozenset)):
            simplices = [sorted(simplex) for simplex in simplices]
        rows = [tuple(sorted(int(value) for value in simplex)) for simplex in simplices]
        canonical_faces.append(sorted(rows))
    return canonical_faces


def two_face_identity(triangulation):
    return sha256_bytes(canonical_json_bytes(canonical_two_face_rows(triangulation)))


def construct_candidate(cytools, candidate):
    poly = cytools.Polytope(candidate["vertices"])
    if candidate["construction"] == "Polytope(vertices).dual()":
        poly = poly.dual()
    elif candidate["construction"] != "Polytope(vertices)":
        raise ValidationError(f"unsupported construction {candidate['construction']!r}")
    return poly


def enumerate_candidate(
    cytools,
    candidate,
    output_dir: Path,
    cap: int,
    time_cap: int,
    progress_path: Path,
    started: float,
):
    poly = construct_candidate(cytools, candidate)
    if not poly.is_reflexive():
        result = {
            "candidate_id": candidate["candidate_id"],
            "source_locator": candidate["source_locator"],
            "status": "ineligible_not_reflexive",
            "support_complete": False,
            "reason": "CYTools does not identify this frozen source fixture as reflexive.",
            "polytope_id": polytope_identity(poly),
            "full_frst_support_count_or_lower_bound": 0,
            "full_frst_support_sha256": sha256_bytes(canonical_json_bytes([])),
            "elapsed_wall_seconds": time.monotonic() - started,
            "enumerator_script_git_blob_sha1": git_blob_sha1(SCRIPT_PATH.read_bytes()),
        }
        append_jsonl_fsync(
            progress_path,
            {
                "event": "candidate_complete",
                "candidate_id": candidate["candidate_id"],
                "status": result["status"],
                "elapsed_wall_seconds": result["elapsed_wall_seconds"],
                "support_count_or_lower_bound": 0,
                "support_sha256": result["full_frst_support_sha256"],
            },
        )
        return result

    point_rows = [list(map(int, row)) for row in poly.points().tolist()]
    public_point_labels = [int(label) for label in poly._triang_labels(None)]
    support = {}
    state_records = []
    iterator = poly.all_triangulations(
        points=None,
        only_fine=True,
        only_regular=True,
        only_star=True,
        backend=None,
        as_list=False,
    )
    terminal_reason = "exhausted"
    for triangulation in iterator:
        if time.monotonic() - started > time_cap:
            terminal_reason = "wall_clock_cap"
            break
        full_id = full_triangulation_identity(triangulation)
        if full_id in support:
            raise ValidationError(
                f"duplicate canonical full identity during exact enumeration: {candidate['candidate_id']} {full_id}"
            )
        if not (
            triangulation.is_fine()
            and triangulation.is_regular()
            and triangulation.is_star()
        ):
            raise ValidationError(
                f"public exact enumerator yielded a non-FRST for {candidate['candidate_id']}"
            )
        simplex_rows = [list(row) for row in canonical_simplices(triangulation)]
        two_face_rows = canonical_two_face_rows(triangulation)
        state = {
            "full_triangulation_sha256": full_id,
            "two_face_sha256": sha256_bytes(canonical_json_bytes(two_face_rows)),
            "simplices_as_point_indices": simplex_rows,
        }
        append_jsonl_fsync(
            progress_path,
            {
                "event": "enumerated_state",
                "index": len(state_records) + 1,
                "state": state,
            },
        )
        support[full_id] = state
        state_records.append(state)
        if len(state_records) >= cap:
            terminal_reason = "support_count_cap"
            break

    count = len(state_records)
    complete = terminal_reason == "exhausted"
    if complete and count < 2:
        status = "ineligible_support_below_two"
    elif terminal_reason == "support_count_cap":
        status = "ineligible_support_above_500"
    elif terminal_reason != "exhausted":
        status = "enumeration_incomplete"
    elif count <= 500:
        status = "qualified"
    else:
        status = "ineligible_support_above_500"

    state_records.sort(key=lambda row: row["full_triangulation_sha256"])
    result = {
        "candidate_id": candidate["candidate_id"],
        "source_locator": candidate["source_locator"],
        "source_coordinates": candidate["vertices"],
        "construction": candidate["construction"],
        "status": status,
        "support_complete": complete,
        "terminal_reason": terminal_reason,
        "elapsed_wall_seconds": time.monotonic() - started,
        "polytope_id": polytope_identity(poly),
        "full_polytope_points_in_public_order": point_rows,
        "public_wrapper_point_labels": public_point_labels,
        "full_frst_support_count_or_lower_bound": count,
        "full_frst_support_sha256": sha256_bytes(
            canonical_json_bytes(
                [state["full_triangulation_sha256"] for state in state_records]
            )
        ),
        "enumerator_script_git_blob_sha1": git_blob_sha1(SCRIPT_PATH.read_bytes()),
        "states": state_records,
        "enumerator": {
            "cytools": "1.4.12",
            "method": "Polytope.all_triangulations",
            "only_fine": True,
            "only_regular": True,
            "only_star": True,
            "points": None,
            "as_list": False,
        },
    }
    append_jsonl_fsync(
        progress_path,
        {
            "event": "candidate_complete",
            "candidate_id": candidate["candidate_id"],
            "status": status,
            "terminal_reason": terminal_reason,
            "elapsed_wall_seconds": result["elapsed_wall_seconds"],
            "support_count_or_lower_bound": count,
            "support_sha256": result["full_frst_support_sha256"],
        },
    )
    return result


def incomplete_candidate_record(
    candidate,
    output_dir: Path,
    progress_path: Path,
    timeout_result,
    limit_seconds: float,
    candidate_limit_seconds: float,
    terminal_reason: str,
):
    journal = read_progress_journal(progress_path)
    states = [
        event["state"]
        for event in journal["events"]
        if event.get("event") == "enumerated_state"
    ]
    state_ids = [state["full_triangulation_sha256"] for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise ValidationError(
            f"durable partial exact-support journal contains duplicate states: {candidate['candidate_id']}"
        )
    ordered_ids = sorted(state_ids)
    return {
        "candidate_id": candidate["candidate_id"],
        "source_locator": candidate["source_locator"],
        "source_coordinates": candidate["vertices"],
        "construction": candidate["construction"],
        "status": "enumeration_incomplete",
        "support_complete": False,
        "terminal_reason": terminal_reason,
        "elapsed_wall_seconds": timeout_result["elapsed_wall_seconds"],
        "full_frst_support_count_or_lower_bound": len(states),
        "full_frst_support_sha256": sha256_bytes(canonical_json_bytes(ordered_ids)),
        "states": states,
        "enumerator_script_git_blob_sha1": git_blob_sha1(SCRIPT_PATH.read_bytes()),
        "external_hard_cap": {
            "process_group_timeout_enforced": True,
            "per_candidate_limit_seconds": candidate_limit_seconds,
            "effective_deadline_seconds": limit_seconds,
            "timed_out": timeout_result["timed_out"],
            "process_returncode": timeout_result["returncode"],
            "termination_method": "SIGTERM then SIGKILL to isolated process group",
            "elapsed_wall_seconds": timeout_result["elapsed_wall_seconds"],
        },
        "progress_journal": {
            "path": progress_path.name,
            **journal,
            "complete_event_count": len(journal["events"]),
            "partial_state_count": len(states),
        },
        "worker_stdout_sha256": sha256_bytes(timeout_result["stdout"].encode("utf-8")),
        "worker_stderr_sha256": sha256_bytes(timeout_result["stderr"].encode("utf-8")),
    }


def enumerate_candidate_process(
    registration_path: Path,
    candidate,
    output_dir: Path,
    cache_dir: Path,
    support_cap: int,
    candidate_time_cap: float,
    effective_time_cap: float,
):
    progress_path = output_dir / f"progress-{candidate['candidate_id']}.jsonl"
    support_path = output_dir / f"support-{candidate['candidate_id']}.json"
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--phase",
        "enumerate-candidate",
        "--registration",
        str(registration_path),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_dir),
        "--candidate-id",
        candidate["candidate_id"],
        "--support-cap",
        str(support_cap),
        "--time-cap",
        str(candidate_time_cap),
    ]
    outcome = run_bounded_process(command, effective_time_cap)
    if outcome["timed_out"] or outcome["returncode"] != 0:
        terminal_reason = (
            "external_hard_wall_clock_cap"
            if effective_time_cap >= candidate_time_cap
            else "external_cumulative_wall_clock_cap"
        ) if outcome["timed_out"] else "enumeration_worker_failed"
        result = incomplete_candidate_record(
            candidate,
            output_dir,
            progress_path,
            outcome,
            effective_time_cap,
            candidate_time_cap,
            terminal_reason,
        )
        atomic_json_create(support_path, result)
        return result

    try:
        result = json.loads(outcome["stdout"])
    except json.JSONDecodeError as exc:
        failed_outcome = dict(outcome)
        failed_outcome["stderr"] += f"\ninvalid worker JSON: {exc}"
        result = incomplete_candidate_record(
            candidate,
            output_dir,
            progress_path,
            failed_outcome,
            effective_time_cap,
            candidate_time_cap,
            "enumeration_worker_output_invalid",
        )
        atomic_json_create(support_path, result)
        return result

    journal = read_progress_journal(progress_path)
    result["external_hard_cap"] = {
        "process_group_timeout_enforced": True,
        "per_candidate_limit_seconds": candidate_time_cap,
        "effective_deadline_seconds": effective_time_cap,
        "timed_out": False,
        "process_returncode": outcome["returncode"],
        "termination_method": "isolated process group; deadline expired only if incomplete",
        "elapsed_wall_seconds": outcome["elapsed_wall_seconds"],
    }
    result["progress_journal"] = {
        "path": progress_path.name,
        **journal,
        "complete_event_count": len(journal["events"]),
        "partial_state_count": sum(
            event.get("event") == "enumerated_state" for event in journal["events"]
        ),
    }
    atomic_json_create(support_path, result)
    return result


def enumerate_one_candidate(
    registration_path: Path,
    candidate_id: str,
    output_dir: Path,
    cache_dir: Path,
    support_cap: int,
    time_cap: float,
):
    started = time.monotonic()
    registration = load_registration(registration_path)
    candidates = registration["candidate_fixture_selection"]["candidates"]
    candidate = next(
        (item for item in candidates if item["candidate_id"] == candidate_id), None
    )
    if candidate is None:
        raise ValidationError(f"candidate is not present in frozen registration: {candidate_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"progress-{candidate_id}.jsonl"
    atomic_create(
        progress_path,
        canonical_json_bytes(
            {
                "event": "candidate_worker_started",
                "candidate_id": candidate_id,
                "monotonic_start": started,
                "wall_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "candidate_wall_cap_seconds": time_cap,
            }
        )
        + b"\n",
    )
    cytools, _, _, np = bootstrap_cytools(cache_dir)
    environment = environment_provenance(cytools, np)
    verify_source_bindings(registration, environment)
    append_jsonl_fsync(
        progress_path,
        {
            "event": "candidate_operation_started",
            "candidate_id": candidate_id,
            "monotonic_start": started,
            "environment_sha256": sha256_bytes(canonical_json_bytes(environment)),
        },
    )
    result = enumerate_candidate(
        cytools,
        candidate,
        output_dir,
        support_cap,
        time_cap,
        progress_path,
        started,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def verify_external_timeout_self_test(output_dir: Path):
    test_timeout_seconds = 1.0
    outcome = run_bounded_process(
        [
            sys.executable,
            "-c",
            "import time; print('WORKER_READY', flush=True); time.sleep(10)",
        ],
        test_timeout_seconds,
    )
    result = {
        "schema": "CYAX-0115-hard-timeout-self-test-v1",
        "method": "subprocess process-group deadline with SIGTERM/SIGKILL fallback",
        "test_timeout_seconds": test_timeout_seconds,
        "timed_out": outcome["timed_out"],
        "worker_ready_marker_observed": "WORKER_READY" in outcome["stdout"],
        "worker_returncode": outcome["returncode"],
        "elapsed_wall_seconds": outcome["elapsed_wall_seconds"],
        "worker_stdout_sha256": sha256_bytes(outcome["stdout"].encode("utf-8")),
        "worker_stderr_sha256": sha256_bytes(outcome["stderr"].encode("utf-8")),
    }
    if not result["timed_out"] or not result["worker_ready_marker_observed"]:
        raise ValidationError(f"external timeout self-test failed: {result}")
    atomic_json_create(output_dir / "hard-timeout-self-test.json", result)
    return result


class SamplerObserver:
    """Passive record of the frozen public augmented-process observations."""

    def __init__(self, support_ids):
        self.support_ids = set(support_ids)
        self.seen_hashes = set()
        self.raw_events = []
        self.duplicate_branches = []
        self.first_discoveries = []
        self.step_updates = []
        self.terminal_events = []
        self.consecutive_duplicates = 0

    @staticmethod
    def _point_snapshot(point):
        if point is None:
            return None
        if hasattr(point, "tolist"):
            point = point.tolist()
        return [float(value) for value in point]

    def _record_raw(self, triangulation, public_hash, n_retries, step_ctr, step_per_tri_ctr, old_pt, actual_seen_hashes=None):
        full_id = full_triangulation_identity(triangulation)
        simplex_rows = [list(row) for row in canonical_simplices(triangulation)]
        duplicate = public_hash in self.seen_hashes
        if actual_seen_hashes is not None and duplicate != (public_hash in actual_seen_hashes):
            raise ValidationError("observer hash history disagrees with CYTools seen-hash set")
        event = {
            "raw_index": len(self.raw_events) + 1,
            "full_triangulation_sha256": full_id,
            "two_face_sha256": two_face_identity(triangulation),
            "simplices_as_point_indices": simplex_rows,
            "cytools_python_hash": int(public_hash),
            "duplicate_predicted_from_public_hash_history": duplicate,
            "support_member": full_id in self.support_ids,
            "n_retries_before_branch": None if n_retries is None else int(n_retries),
            "step_ctr_before_branch": None if step_ctr is None else int(step_ctr),
            "step_per_tri_ctr_before_branch": None if step_per_tri_ctr is None else int(step_per_tri_ctr),
            "old_pt_before_branch": self._point_snapshot(old_pt),
            "public_seen_hash_count_before_branch": (
                None if actual_seen_hashes is None else len(actual_seen_hashes)
            ),
        }
        self.raw_events.append(event)
        if duplicate:
            self.consecutive_duplicates += 1
        else:
            self.seen_hashes.add(public_hash)
            self.consecutive_duplicates = 0
        return event

    def observe(self, event_name, triangulation=None, public_hash=None, n_retries=None,
                step_ctr=None, step_per_tri_ctr=None, old_pt=None,
                actual_seen_hashes=None, reason=None):
        if event_name == "raw_candidate":
            return self._record_raw(
                triangulation, public_hash, n_retries, step_ctr,
                step_per_tri_ctr, old_pt, actual_seen_hashes
            )
        if event_name == "duplicate_branch":
            if not self.raw_events or not self.raw_events[-1]["duplicate_predicted_from_public_hash_history"]:
                raise ValidationError("CYTools entered duplicate branch for a predicted new hash")
            record = {
                "raw_index": self.raw_events[-1]["raw_index"],
                "n_retries_after_increment": int(n_retries),
                "step_ctr": int(step_ctr),
                "step_per_tri_ctr": int(step_per_tri_ctr),
                "old_pt": self._point_snapshot(old_pt),
                "seen_hash_count": len(actual_seen_hashes),
            }
            self.duplicate_branches.append(record)
            return record
        if event_name == "first_discovery":
            if not self.raw_events or self.raw_events[-1]["duplicate_predicted_from_public_hash_history"]:
                raise ValidationError("CYTools added a hash predicted as duplicate")
            record = {
                "raw_index": self.raw_events[-1]["raw_index"],
                "full_triangulation_sha256": self.raw_events[-1]["full_triangulation_sha256"],
                "yield_order": len(self.first_discoveries) + 1,
                "n_retries_after_reset": int(n_retries),
                "step_ctr": int(step_ctr),
                "step_per_tri_ctr_after_reset": int(step_per_tri_ctr),
                "seen_hash_count_after_add": len(actual_seen_hashes),
            }
            self.first_discoveries.append(record)
            return record
        if event_name == "step_update":
            self.step_updates.append(
                {
                    "raw_index": len(self.raw_events),
                    "step_ctr": int(step_ctr),
                    "step_per_tri_ctr": int(step_per_tri_ctr),
                    "old_pt_after_update": self._point_snapshot(old_pt),
                }
            )
            return
        if event_name == "terminal":
            record = {
                "reason": reason,
                "n_retries": int(n_retries),
                "step_ctr": int(step_ctr),
                "step_per_tri_ctr": int(step_per_tri_ctr),
                "raw_observation_count": len(self.raw_events),
                "unique_count": len(self.first_discoveries),
            }
            self.terminal_events.append(record)
            return
        raise ValidationError(f"unknown sampler observer event: {event_name}")


def _observe_call(event_name, reason=None):
    positional = [ast.Constant(value=event_name)]
    keywords = []
    if event_name != "terminal":
        keywords.extend(
            [
                ast.keyword(arg="triangulation", value=ast.Name(id="temp_tri", ctx=ast.Load())),
                ast.keyword(arg="public_hash", value=ast.Name(id="h", ctx=ast.Load())),
                ast.keyword(arg="n_retries", value=ast.Name(id="n_retries", ctx=ast.Load())),
                ast.keyword(arg="step_ctr", value=ast.Name(id="step_ctr", ctx=ast.Load())),
                ast.keyword(arg="step_per_tri_ctr", value=ast.Name(id="step_per_tri_ctr", ctx=ast.Load())),
                ast.keyword(arg="old_pt", value=ast.Name(id="old_pt", ctx=ast.Load())),
                ast.keyword(arg="actual_seen_hashes", value=ast.Name(id="triang_hashes", ctx=ast.Load())),
            ]
        )
    else:
        keywords.extend(
            [
                ast.keyword(arg="n_retries", value=ast.Name(id="n_retries", ctx=ast.Load())),
                ast.keyword(arg="step_ctr", value=ast.Name(id="step_ctr", ctx=ast.Load())),
                ast.keyword(arg="step_per_tri_ctr", value=ast.Name(id="step_per_tri_ctr", ctx=ast.Load())),
                ast.keyword(arg="old_pt", value=ast.Name(id="old_pt", ctx=ast.Load())),
            ]
        )
    if reason is not None:
        keywords.append(ast.keyword(arg="reason", value=ast.Constant(value=reason)))
    expression = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="__cyax_observe__", ctx=ast.Load()),
            args=positional,
            keywords=keywords,
        )
    )
    expression._cyax_injected = True
    return expression


class FairGeneratorInstrumenter(ast.NodeTransformer):
    """Insert passive callbacks at the frozen pre-dedup and state-update sites."""

    @staticmethod
    def _is_name(node, name):
        return isinstance(node, ast.Name) and node.id == name

    def visit_FunctionDef(self, node):
        if node.name != "random_triangulations_fair_generator":
            return self.generic_visit(node)
        transformed = []
        for statement in node.body:
            visited = self.visit(statement)
            if isinstance(visited, list):
                transformed.extend(visited)
            elif visited is not None:
                transformed.append(visited)
        node.body = transformed
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if (
            len(node.targets) == 1
            and self._is_name(node.targets[0], "h")
            and isinstance(node.value, ast.Call)
            and self._is_name(node.value.func, "hash")
            and len(node.value.args) == 1
            and self._is_name(node.value.args[0], "temp_tri")
        ):
            return [node, _observe_call("raw_candidate")]
        return node

    def visit_AugAssign(self, node):
        node = self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id == "n_retries":
            return [node, _observe_call("duplicate_branch")]
        if isinstance(node.target, ast.Name) and node.target.id == "step_per_tri_ctr":
            return [node, _observe_call("step_update")]
        return node

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Yield) and self._is_name(node.value.value, "temp_tri"):
            return [_observe_call("first_discovery"), node]
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        test = node.test
        reason = None
        if (
            isinstance(test, ast.Compare)
            and self._is_name(test.left, "n_retries")
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.GtE)
            and len(test.comparators) == 1
            and self._is_name(test.comparators[0], "max_retries")
        ):
            reason = "max_retries"
        elif (
            isinstance(test, ast.BoolOp)
            and isinstance(test.op, ast.And)
            and len(test.values) == 2
            and isinstance(test.values[0], ast.Compare)
            and self._is_name(test.values[0].left, "N")
            and len(test.values[0].ops) == 1
            and isinstance(test.values[0].ops[0], ast.IsNot)
            and isinstance(test.values[0].comparators[0], ast.Constant)
            and test.values[0].comparators[0].value is None
        ):
            reason = "unique_count_limit"
        elif (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and self._is_name(test.operand, "outside_bounds")
        ):
            reason = "no_wall_found"
        if reason is not None:
            node.body.insert(0, _observe_call("terminal", reason))
        return node


class StripInjectedObservers(ast.NodeTransformer):
    def visit_Expr(self, node):
        if getattr(node, "_cyax_injected", False):
            return None
        return self.generic_visit(node)


def instrumented_fair_generator_source(triangulation_module):
    original = triangulation_module.random_triangulations_fair_generator
    source = inspect.getsource(original)
    original_ast = ast.parse(source)
    transformed_for_compile = FairGeneratorInstrumenter().visit(ast.parse(source))
    ast.fix_missing_locations(transformed_for_compile)
    stripped_for_compare = StripInjectedObservers().visit(copy.deepcopy(transformed_for_compile))
    if ast.dump(original_ast, include_attributes=False) != ast.dump(
        stripped_for_compare, include_attributes=False
    ):
        raise ValidationError("instrumented CYTools AST differs beyond passive observers")
    code = compile(transformed_for_compile, inspect.getsourcefile(original), "exec")
    namespace = triangulation_module.__dict__
    prior_observer = namespace.get("__cyax_observe__")
    namespace["__cyax_observe__"] = None
    exec(code, namespace)
    instrumented = namespace[original.__name__]
    namespace[original.__name__] = original
    if prior_observer is None:
        namespace.pop("__cyax_observe__", None)
    else:
        namespace["__cyax_observe__"] = prior_observer
    return instrumented, ast.dump(original_ast, include_attributes=False)


def _default_initial_walk_steps(poly):
    return 2 * len(poly.points()) // 10 + 10


def _run_chain_in_worker(
    manifest_path: Path,
    manifest_sha256: str,
    fixture_id: str,
    control_id: str,
    seed: int,
    mode: str,
    output_path: Path,
    cache_dir: Path,
    adversarial: bool = False,
):
    actual_manifest_sha = file_sha256(manifest_path)
    if actual_manifest_sha != manifest_sha256:
        raise ValidationError(
            f"frozen manifest hash mismatch: expected {manifest_sha256}, got {actual_manifest_sha}"
        )
    manifest = load_json(manifest_path)
    if manifest.get("status") != "FROZEN_BEFORE_SAMPLER_CALIBRATION":
        raise ValidationError("validation manifest was not frozen before sampler calibration")
    fixture = next((item for item in manifest["fixtures"] if item["candidate_id"] == fixture_id), None)
    if fixture is None:
        raise ValidationError(f"fixture is not in frozen manifest: {fixture_id}")
    control_record = next((item for item in fixture["controls"] if item["control_id"] == control_id), None)
    if control_record is None:
        raise ValidationError(f"control is not in frozen manifest for fixture: {control_id}")
    if seed not in control_record["fixed_restart_seeds"]:
        raise ValidationError(f"seed is outside frozen restart list: {seed}")

    support_path = manifest_path.parent / fixture["support_file"]
    if file_sha256(support_path) != fixture["support_file_sha256"]:
        raise ValidationError(f"exact-support file hash mismatch: {support_path}")
    support = load_json(support_path)
    support_ids = [state["full_triangulation_sha256"] for state in support["states"]]
    if sha256_bytes(canonical_json_bytes(sorted(support_ids))) != fixture["support_sha256"]:
        raise ValidationError(f"exact-support identity set mismatch: {fixture_id}")

    registration = load_registration()
    cytools, polytope_module, triangulation_module, np = bootstrap_cytools(cache_dir)
    environment = environment_provenance(cytools, np)
    verify_source_bindings(registration, environment)
    candidate = fixture["source_fixture"]
    poly = construct_candidate(cytools, candidate)
    if polytope_identity(poly) != fixture["polytope_id"]:
        raise ValidationError(f"fixture polytope identity mismatch: {fixture_id}")

    observer = SamplerObserver(support_ids) if mode in ("hash-observed", "instrumented") else None
    original_generator = triangulation_module.random_triangulations_fair_generator
    original_poly_generator = polytope_module.random_triangulations_fair_generator
    tri_class = triangulation_module.Triangulation
    original_tri_hash = tri_class.__hash__
    original_tri_class = triangulation_module.Triangulation
    prior_callback = triangulation_module.__dict__.get("__cyax_observe__")
    callback_installed = False
    hash_hook_installed = False
    instrumented_ast_sha256 = None
    captured_stdout = io.StringIO()
    started = time.monotonic()
    pid = os.getpid()
    control = control_record["sampler_controls"]
    initial_walk_steps = _default_initial_walk_steps(poly)
    if control_id == "01-stricter":
        initial_walk_steps *= 2
    elif control_id != "00-paper-source":
        raise ValidationError(f"unrecognized frozen sampler control: {control_id}")

    try:
        if mode == "instrumented":
            instrumented, source_ast = instrumented_fair_generator_source(triangulation_module)
            instrumented_ast_sha256 = sha256_bytes(source_ast.encode("utf-8"))
            triangulation_module.random_triangulations_fair_generator = instrumented
            polytope_module.random_triangulations_fair_generator = instrumented
            triangulation_module.__dict__["__cyax_observe__"] = observer.observe
            callback_installed = True
        elif mode == "hash-observed":
            observer_function_name = original_generator.__name__

            def observed_hash(triangulation):
                value = original_tri_hash(triangulation)
                frame = inspect.currentframe().f_back
                if frame is not None and frame.f_code.co_name == observer_function_name:
                    observer.observe("raw_candidate", triangulation, value)
                return value

            tri_class.__hash__ = observed_hash
            hash_hook_installed = True
        elif mode != "public":
            raise ValidationError(f"unsupported chain mode: {mode}")

        if adversarial:
            class AlwaysFineTriangulation:
                def __init__(self, *args, **kwargs):
                    pass

                def is_fine(self):
                    return True

                def random_flips(self, *args, **kwargs):
                    return self

                def __hash__(self):
                    return 0

                def simplices(self, *args, **kwargs):
                    raise AssertionError("adversarial stub must stop before candidate capture")

            triangulation_module.Triangulation = AlwaysFineTriangulation

        sampler_args = {
            "points": None,
            "N": None,
            "n_walk": int(control["n_walk"]),
            "n_flip": int(control["n_flip"]),
            "initial_walk_steps": initial_walk_steps,
            "walk_step_size": float(control["walk_step_size"]),
            "max_steps_to_wall": int(control["max_steps_to_wall"]),
            "fine_tune_steps": int(control["fine_tune_steps"]),
            "max_retries": 1 if adversarial else int(control["max_retries"]),
            "make_star": bool(control["make_star"]),
            "backend": str(control["backend"]),
            "seed": int(seed),
        }
        yielded = []
        with contextlib.redirect_stdout(captured_stdout):
            generator = poly.random_triangulations_fair(**sampler_args)
            for triangulation in generator:
                yielded.append(full_triangulation_identity(triangulation))
        source_root = Path(cytools.__file__).resolve().parent
        result = {
            "schema": "CYAX-0115-r4-chain-observation-v1",
            "manifest_sha256": manifest_sha256,
            "script_git_blob_sha1": git_blob_sha1(SCRIPT_PATH.read_bytes()),
            "fixture_id": fixture_id,
            "K": fixture["K"],
            "control_id": control_id,
            "seed": int(seed),
            "mode": mode,
            "adversarial_test": adversarial,
            "process_start_method": "spawn-subprocess",
            "process_id": pid,
            "parent_process_id": os.getppid(),
            "elapsed_wall_seconds": time.monotonic() - started,
            "environment": environment,
            "source_file_git_blobs": {
                name: git_blob_sha1((source_root / name).read_bytes())
                for name in ("polytope.py", "triangulation.py")
            },
            "instrumented_source_ast_sha256": instrumented_ast_sha256,
            "sampler_arguments": sampler_args,
            "initial_walk_steps_rule_evaluated": "2*len(poly.points())//10+10; doubled for 01-stricter",
            "yielded_unique_full_identities": yielded,
            "captured_stdout": captured_stdout.getvalue(),
            "natural_generator_exhaustion": True,
        }
        if observer is not None:
            result["raw_observations"] = observer.raw_events
            result["duplicate_branch_events"] = observer.duplicate_branches
            result["first_discoveries"] = observer.first_discoveries
            result["step_updates"] = observer.step_updates
            result["terminal_events"] = observer.terminal_events
            result["raw_observation_count"] = len(observer.raw_events)
            result["unique_projection_k_observed"] = min(10, fixture["K"] // 2)
            result["unique_projection_complete"] = len(yielded) >= result["unique_projection_k_observed"]
            result["all_raw_states_in_exact_support"] = all(
                event["support_member"] for event in observer.raw_events
            )
            if observer.terminal_events:
                result["termination_reason"] = observer.terminal_events[-1]["reason"]
            elif observer.consecutive_duplicates >= sampler_args["max_retries"]:
                result["termination_reason"] = "max_retries"
            elif "Couldn't find wall." in captured_stdout.getvalue():
                result["termination_reason"] = "no_wall_found"
            else:
                result["termination_reason"] = "natural_generator_exhaustion_unclassified"
            if adversarial:
                result["gate_result"] = (
                    "EARLY_TERMINATION_FAIL_CLOSED"
                    if len(observer.raw_events) == 0 and len(yielded) == 0
                    else "ADVERSARIAL_TEST_FAILED"
                )
                result["replacement_seed_count"] = 0
                result["survivor_conditioned_statistics_computed"] = False
        else:
            result["raw_observation_count"] = None
            result["termination_reason"] = (
                "no_wall_found"
                if "Couldn't find wall." in captured_stdout.getvalue()
                else "natural_generator_exhaustion_unclassified"
            )
        if observer is not None and not adversarial:
            required_j = manifest["raw_indices"]
            result["raw_required_indices_seen"] = all(
                len(observer.raw_events) >= index for index in required_j
            )
            result["early_termination_protocol_failure"] = (
                len(observer.raw_events) < max(required_j)
                or len(yielded) < result["unique_projection_k_observed"]
            )
            result["replacement_seed_count"] = 0
            result["survivor_conditioned_statistics_computed"] = False
        atomic_json_create(output_path, result)
        return result
    finally:
        triangulation_module.random_triangulations_fair_generator = original_generator
        polytope_module.random_triangulations_fair_generator = original_poly_generator
        triangulation_module.Triangulation = original_tri_class
        if hash_hook_installed:
            tri_class.__hash__ = original_tri_hash
        if callback_installed:
            if prior_callback is None:
                triangulation_module.__dict__.pop("__cyax_observe__", None)
            else:
                triangulation_module.__dict__["__cyax_observe__"] = prior_callback


def enumerate_frozen_candidates(registration_path: Path, output_dir: Path, cache_dir: Path):
    registration = load_registration(registration_path)
    registration_sha = file_sha256(registration_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValidationError(f"enumeration output directory already contains data: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cytools, _, _, np = bootstrap_cytools(cache_dir)
    environment = environment_provenance(cytools, np)
    verify_source_bindings(registration, environment)

    policy = registration["candidate_fixture_selection"]
    support_cap = policy["support_window"]["enumeration_stop_after_count"]
    resource_caps = registration["resource_caps_and_stops"]["exact_support_cap"]
    time_cap = resource_caps[
        "wall_clock_seconds_per_candidate"
    ]
    cumulative_cap = resource_caps["cumulative_enumeration_wall_clock_seconds"]
    hard_cap_self_test = verify_external_timeout_self_test(output_dir)
    cumulative_started = time.monotonic()
    results = []
    eligible = []
    for candidate in policy["candidates"]:
        cumulative_remaining = cumulative_cap - (time.monotonic() - cumulative_started)
        if cumulative_remaining <= 0:
            result = incomplete_candidate_record(
                candidate,
                output_dir,
                output_dir / f"progress-{candidate['candidate_id']}.jsonl",
                {
                    "elapsed_wall_seconds": 0.0,
                    "timed_out": True,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                },
                0.0,
                time_cap,
                "external_cumulative_wall_clock_cap",
            )
            atomic_json_create(
                output_dir / f"support-{candidate['candidate_id']}.json", result
            )
            results.append(result)
            break
        effective_cap = min(time_cap, cumulative_remaining)
        result = enumerate_candidate_process(
            registration_path,
            candidate,
            output_dir,
            cache_dir,
            support_cap,
            time_cap,
            effective_cap,
        )
        results.append(result)
        if result["status"] == "qualified":
            eligible.append(result)
        if result["status"] == "enumeration_incomplete":
            break
        if len(eligible) == 3:
            break

    summary = {
        "registration_sha256": registration_sha,
        "environment": environment,
        "resource_cap_enforcement": {
            "mechanism": "one spawned subprocess process group per candidate; communicate(timeout); SIGTERM then SIGKILL",
            "per_candidate_limit_seconds": time_cap,
            "cumulative_limit_seconds": cumulative_cap,
            "hard_timeout_self_test_file_sha256": file_sha256(
                output_dir / "hard-timeout-self-test.json"
            ),
            "hard_timeout_self_test": hard_cap_self_test,
            "total_elapsed_wall_seconds": time.monotonic() - cumulative_started,
        },
        "candidate_results": [
            {
                "candidate_id": result["candidate_id"],
                "status": result["status"],
                "support_complete": result["support_complete"],
                "terminal_reason": result.get("terminal_reason"),
                "elapsed_wall_seconds": result.get("elapsed_wall_seconds"),
                "external_hard_cap": result.get("external_hard_cap"),
                "support_count_or_lower_bound": result[
                    "full_frst_support_count_or_lower_bound"
                ],
                "support_file_sha256": file_sha256(
                    output_dir / f"support-{result['candidate_id']}.json"
                ),
            }
            for result in results
        ],
        "selected_candidate_ids": [result["candidate_id"] for result in eligible],
        "status": (
            "READY_TO_FREEZE_VALIDATION_MANIFEST"
            if len(eligible) == 3 and all(r["status"] != "enumeration_incomplete" for r in results)
            else "CALIBRATION_SUPPORT_INSUFFICIENT"
            if len(eligible) < 3 and all(r["status"] != "enumeration_incomplete" for r in results)
            else "UNIFORMITY_NOT_ESTABLISHED"
        ),
        "terminal_reason": (
            "fewer than three frozen source candidates have complete support sizes in 2..500"
            if len(eligible) < 3 and all(r["status"] != "enumeration_incomplete" for r in results)
            else "candidate exact-support enumeration did not complete within the frozen resource cap"
            if any(r["status"] == "enumeration_incomplete" for r in results)
            else None
        ),
    }
    atomic_json_create(output_dir / "support-selection.json", summary)
    return summary


def freeze_validation_manifest(registration_path: Path, output_dir: Path):
    registration = load_registration(registration_path)
    registration_sha = file_sha256(registration_path)
    summary_path = output_dir / "support-selection.json"
    if not summary_path.is_file():
        raise ValidationError("exact-support selection record is missing")
    summary = load_json(summary_path)
    if summary["registration_sha256"] != registration_sha:
        raise ValidationError("support selection is bound to a different registration")
    if summary["status"] != "READY_TO_FREEZE_VALIDATION_MANIFEST":
        status_record = {
            "status": summary["status"],
            "registration_sha256": registration_sha,
            "exact_support_selection_sha256": file_sha256(summary_path),
            "candidate_results": summary["candidate_results"],
            "terminal_reason": summary["terminal_reason"],
            "sampler_calibration_started": False,
        }
        atomic_json_create(output_dir / "calibration-support-status.json", status_record)
        return status_record

    support_by_id = {}
    for candidate_id in summary["selected_candidate_ids"]:
        path = output_dir / f"support-{candidate_id}.json"
        record = load_json(path)
        if record["status"] != "qualified" or not record["support_complete"]:
            raise ValidationError(f"selected support is incomplete: {candidate_id}")
        if len(record["states"]) != record["full_frst_support_count_or_lower_bound"]:
            raise ValidationError(f"support cardinality mismatch: {candidate_id}")
        support_by_id[candidate_id] = record

    controls = registration["sampler_control_grid"]
    fixtures = []
    fixture_by_candidate = {
        candidate["candidate_id"]: candidate
        for candidate in registration["candidate_fixture_selection"]["candidates"]
    }
    for fixture_order, candidate_id in enumerate(summary["selected_candidate_ids"], start=1):
        support = support_by_id[candidate_id]
        k_states = support["full_frst_support_count_or_lower_bound"]
        restart_count = max(10000, 50 * k_states)
        seed_base = 0x11500000 + fixture_order * 0x00100000
        seeds = [seed_base + index for index in range(restart_count)]
        if len(set(seeds)) != restart_count or max(seeds) > 0xFFFFFFFF:
            raise ValidationError(f"frozen restart seeds are not distinct uint32s: {candidate_id}")
        control_records = []
        for control_order, control in enumerate(controls, start=1):
            null_seeds = {
                "raw": {
                    str(index): 0x2A115000
                    + 100000 * fixture_order
                    + 10000 * control_order
                    + 100 * position
                    for position, index in enumerate(
                        registration["calibration_protocol"]["raw_observation_indices"],
                        start=1,
                    )
                },
                "unique": {
                    str(index): 0x3A115000
                    + 100000 * fixture_order
                    + 10000 * control_order
                    + 100 * position
                    for position, index in enumerate(
                        range(1, min(10, k_states // 2) + 1), start=1
                    )
                },
            }
            control_records.append(
                {
                    "control_id": control["control_id"],
                    "control_order": control_order,
                    "sampler_controls": control,
                    "fixed_restart_seeds": seeds,
                    "restart_count": restart_count,
                    "unique_projection_k": min(10, k_states // 2),
                    "null_simulation_seeds": null_seeds,
                }
            )
        fixtures.append(
            {
                "fixture_order": fixture_order,
                "candidate_id": candidate_id,
                "source_fixture": fixture_by_candidate[candidate_id],
                "polytope_id": support["polytope_id"],
                "point_rows_in_public_order": support["full_polytope_points_in_public_order"],
                "public_wrapper_point_labels": support["public_wrapper_point_labels"],
                "K": k_states,
                "support_file": f"support-{candidate_id}.json",
                "support_file_sha256": file_sha256(
                    output_dir / f"support-{candidate_id}.json"
                ),
                "support_sha256": support["full_frst_support_sha256"],
                "controls": control_records,
            }
        )

    manifest = {
        "schema": "CYAX-0115-r4-frozen-validation-manifest-v1",
        "status": "FROZEN_BEFORE_SAMPLER_CALIBRATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registration_sha256": registration_sha,
        "support_selection_sha256": file_sha256(summary_path),
        "script_git_blob_sha1": git_blob_sha1(SCRIPT_PATH.read_bytes()),
        "execution_base": registration["execution_base"],
        "current_repository_commit": subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "source_bindings": registration["source_bindings"],
        "environment": summary["environment"],
        "scientific_target": registration["scientific_target"],
        "raw_indices": registration["calibration_protocol"]["raw_observation_indices"],
        "statistical_protocol": registration["statistical_protocol"],
        "calibration_protocol": registration["calibration_protocol"],
        "resource_caps_and_stops": registration["resource_caps_and_stops"],
        "fixtures": fixtures,
        "appendix_a_benchmark": registration["appendix_a_benchmark"],
        "large_h11_plan": registration["large_h11_plan"],
        "comparison_plan": registration["comparison_plan"],
        "open_policy": registration["open_policy"],
    }
    manifest_path = output_dir / "validation_manifest.json"
    atomic_json_create(manifest_path, manifest)
    manifest_sha = file_sha256(manifest_path)
    atomic_create(output_dir / "validation_manifest.sha256", (manifest_sha + "  validation_manifest.json\n").encode())
    return {
        "status": "VALIDATION_MANIFEST_FROZEN",
        "manifest_path": str(manifest_path.resolve().relative_to(REPOSITORY_ROOT.resolve())),
        "manifest_sha256": manifest_sha,
        "selected_fixture_count": len(fixtures),
        "restart_seeds_per_control": [fixture["controls"][0]["restart_count"] for fixture in fixtures],
    }


def verify_existing_frozen_manifest(manifest_path: Path, expected_sha256: str | None = None):
    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    actual_sha256 = sha256_bytes(manifest_bytes)
    receipt_path = manifest_path.with_name("validation_manifest.sha256")
    receipt_fields = receipt_path.read_text(encoding="utf-8").strip().split()
    if len(receipt_fields) != 2 or receipt_fields[1] != manifest_path.name:
        raise ValidationError(f"invalid manifest SHA-256 receipt: {receipt_path}")
    if receipt_fields[0] != actual_sha256:
        raise ValidationError("frozen manifest bytes do not match their SHA-256 receipt")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValidationError(
            f"frozen manifest hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "FROZEN_BEFORE_SAMPLER_CALIBRATION":
        raise ValidationError("manifest is not marked frozen before calibration")
    schedules = []
    for fixture in manifest["fixtures"]:
        if fixture["K"] < 2 or fixture["K"] > 500:
            raise ValidationError(f"frozen fixture outside exact-support window: {fixture['candidate_id']}")
        expected_n = max(10000, 50 * fixture["K"])
        expected_seed_base = 0x11500000 + fixture["fixture_order"] * 0x00100000
        expected_seeds = [expected_seed_base + index for index in range(expected_n)]
        prior_control_seeds = None
        for control in fixture["controls"]:
            seeds = control["fixed_restart_seeds"]
            if len(seeds) != expected_n or control["restart_count"] != expected_n:
                raise ValidationError(f"frozen seed count mismatch: {fixture['candidate_id']}")
            if seeds != expected_seeds:
                raise ValidationError(f"frozen seed schedule does not match its registered formula: {fixture['candidate_id']}")
            if prior_control_seeds is not None and seeds != prior_control_seeds:
                raise ValidationError(f"paired controls do not share the frozen seed list: {fixture['candidate_id']}")
            prior_control_seeds = seeds
            schedules.append(
                {
                    "fixture_order": fixture["fixture_order"],
                    "candidate_id": fixture["candidate_id"],
                    "K": fixture["K"],
                    "control_id": control["control_id"],
                    "restart_count": control["restart_count"],
                    "fixed_restart_seeds": seeds,
                }
            )
            support_path = manifest_path.parent / fixture["support_file"]
            if file_sha256(support_path) != fixture["support_file_sha256"]:
                raise ValidationError(f"frozen exact-support artifact changed: {support_path}")
    schedule_identity = {
        "schema": "CYAX-0115-r4-fixed-seed-schedule-identity-v1",
        "schedule_count": len(schedules),
        "schedules": schedules,
    }
    schedule_sha256 = sha256_bytes(canonical_json_bytes(schedule_identity))
    return {
        "status": "PASS_EXISTING_FROZEN_MANIFEST_VERIFIED",
        "manifest_path": str(manifest_path.relative_to(REPOSITORY_ROOT.resolve())),
        "manifest_sha256": actual_sha256,
        "receipt_path": str(receipt_path.relative_to(REPOSITORY_ROOT.resolve())),
        "seed_schedule_identity_sha256": schedule_sha256,
        "seed_schedule_count": len(schedules),
        "chain_count": sum(item["restart_count"] for item in schedules),
        "no_manifest_bytes_written": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("enumerate", "enumerate-candidate", "freeze", "verify-frozen", "chain"),
        required=True,
    )
    parser.add_argument("--registration", type=Path, default=REGISTRATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION_ROOT / "exact_support")
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/cyax-0115-cytools-cache"))
    parser.add_argument("--candidate-id")
    parser.add_argument("--support-cap", type=int)
    parser.add_argument("--time-cap", type=float)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--fixture-id")
    parser.add_argument("--control-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("public", "hash-observed", "instrumented"))
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--adversarial-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.phase == "enumerate":
            result = enumerate_frozen_candidates(args.registration, args.output_dir, args.cache_dir)
        elif args.phase == "enumerate-candidate":
            if not args.candidate_id or args.support_cap is None or args.time_cap is None:
                raise ValidationError(
                    "enumerate-candidate requires --candidate-id, --support-cap, and --time-cap"
                )
            return enumerate_one_candidate(
                args.registration,
                args.candidate_id,
                args.output_dir,
                args.cache_dir,
                args.support_cap,
                args.time_cap,
            )
        elif args.phase == "freeze":
            result = freeze_validation_manifest(args.registration, args.output_dir)
        elif args.phase == "verify-frozen":
            if args.manifest is None or args.manifest_sha256 is None:
                raise ValidationError("verify-frozen requires --manifest and --manifest-sha256")
            result = verify_existing_frozen_manifest(args.manifest, args.manifest_sha256)
            if args.result_path is not None:
                atomic_json_create(args.result_path, result)
        else:
            if (
                args.manifest is None
                or not args.manifest_sha256
                or not args.fixture_id
                or not args.control_id
                or args.seed is None
                or args.mode is None
                or args.result_path is None
            ):
                raise ValidationError(
                    "chain requires --manifest, --manifest-sha256, --fixture-id, --control-id, --seed, --mode, and --result-path"
                )
            result = _run_chain_in_worker(
                args.manifest,
                args.manifest_sha256,
                args.fixture_id,
                args.control_id,
                args.seed,
                args.mode,
                args.result_path,
                args.cache_dir,
                args.adversarial_test,
            )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
