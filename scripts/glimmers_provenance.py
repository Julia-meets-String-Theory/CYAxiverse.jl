"""Dependency-light provenance gates for the Glimmers schema-1.1 pilot.

The generator owns geometry construction and artifact integration.  This
module owns the checks that must happen before a caller creates a production
output root.  It deliberately has no CYTools, NumPy, or HDF5 dependency so a
clean replay can be checked in a minimal Python environment.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


PROVENANCE_SCHEMA_VERSION = "glimmers-provenance-1.1"

TERMINAL_STATUSES = frozenset(
    {
        "provenance_validated",
        "provenance_dirty_tree",
        "missing_input_hash",
        "source_query_unrecorded",
        "fresh_source_shortfall",
        "historical_match_claim_forbidden",
        "output_collision",
        "storage_budget_exceeded",
        "user_decision_required",
    }
)

THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMBA_DISABLE_CACHING",
)

VERSION_DISTRIBUTIONS = {
    "cytools": "cytools",
    "numpy": "numpy",
    "h5py": "h5py",
    "qpsolvers": "qpsolvers",
    "pyarrow": "pyarrow",
    "python_flint": "python-flint",
}


class ProvenanceError(RuntimeError):
    """Raise a pre-output failure with a machine-readable terminal status."""

    def __init__(self, status: str, message: str, *, details=None):
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"unknown provenance terminal status: {status}")
        super().__init__(message)
        self.status = status
        self.details = {} if details is None else details


def _jsonable(value):
    """Convert common path and container values to deterministic JSON data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("provenance values must not contain non-finite floats")
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    return str(value)


def canonical_json(value) -> str:
    """Serialize a value using the digest convention shared by the helpers."""
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_digest(value) -> str:
    """Return a process-independent SHA-256 digest for serializable data."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_seed(*parts) -> int:
    """Derive a deterministic non-negative seed from serializable parts."""
    return int.from_bytes(bytes.fromhex(stable_digest(parts)[:16]), "big") & ((1 << 63) - 1)


def sha256_file(path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one input file, raising the required terminal status when absent."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProvenanceError(
            "missing_input_hash",
            f"cannot record input hash for missing file: {resolved}",
            details={"path": str(resolved)},
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_items(source_files):
    if source_files is None:
        return []
    if isinstance(source_files, Mapping):
        return [(str(label), Path(path)) for label, path in source_files.items()]
    return [(str(Path(path).expanduser().resolve()), Path(path)) for path in source_files]


def hash_source_files(source_files):
    """Return source hash maps and path records for the requested input files."""
    items = _source_items(source_files)
    hashes = {}
    records = []
    for label, path in items:
        resolved = Path(path).expanduser().resolve()
        digest = sha256_file(resolved)
        hashes[label] = digest
        records.append({"label": label, "path": str(resolved), "sha256": digest})
    return hashes, records


def _git_output(repo_root: Path, arguments):
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to collect clean-tree provenance")
    completed = subprocess.run(
        [git, *arguments],
        cwd=str(repo_root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def git_identity(
    repo_root,
    *,
    status_fixture=None,
    commit_fixture=None,
    branch_fixture=None,
):
    """Collect Git identity and status, with fixtures available for tests."""
    root = Path(repo_root).expanduser().resolve()
    if status_fixture is None:
        status_text = _git_output(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    elif isinstance(status_fixture, str):
        if status_fixture.strip().lower() == "clean":
            status_text = ""
        elif status_fixture.strip().lower() == "dirty":
            status_text = " M synthetic-provenance-fixture"
        else:
            status_text = status_fixture
    else:
        status_text = "\n".join(str(line) for line in status_fixture)
    status_lines = [line for line in status_text.splitlines() if line.strip()]
    commit = (
        str(commit_fixture)
        if commit_fixture is not None
        else _git_output(root, ["rev-parse", "HEAD"])
    )
    branch = (
        str(branch_fixture)
        if branch_fixture is not None
        else _git_output(root, ["branch", "--show-current"]) or "(detached HEAD)"
    )
    return {
        "commit": commit,
        "branch": branch,
        "status": "dirty" if status_lines else "clean",
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
    }


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def environment_versions(overrides=None):
    """Collect reproducibility-relevant package versions without importing them."""
    versions = {
        "python": platform.python_version(),
        **{name: _distribution_version(distribution) for name, distribution in VERSION_DISTRIBUTIONS.items()},
    }
    if overrides:
        versions.update({str(key): str(value) for key, value in overrides.items()})
    return versions


def host_and_thread_settings(overrides=None):
    """Collect host identity and only the environment variables relevant to threads."""
    host = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "logical_cpus": os.cpu_count(),
    }
    threads = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS}
    if overrides:
        overrides = dict(overrides)
        supplied_threads = overrides.pop("threads", None)
        host.update({str(key): _jsonable(value) for key, value in overrides.items()})
        if supplied_threads:
            threads.update({str(key): _jsonable(value) for key, value in supplied_threads.items()})
    return host, threads


def _command_record(command_line):
    if command_line is None:
        argv = [str(item) for item in sys.argv]
        return {"argv": argv, "shell": shlex.join(argv)}
    if isinstance(command_line, str):
        return {"argv": [], "shell": command_line}
    argv = [str(item) for item in command_line]
    return {"argv": argv, "shell": shlex.join(argv)}


def _absolute_roots(roots):
    if roots is None:
        return []
    if isinstance(roots, (str, os.PathLike, Path)):
        roots = [roots]
    return [str(Path(root).expanduser().resolve()) for root in roots]


def _require_source_query(source_query):
    if not isinstance(source_query, Mapping) or not source_query:
        raise ProvenanceError(
            "source_query_unrecorded",
            "a non-empty source-query record is required before production output",
        )
    source = _jsonable(source_query)
    missing = []
    if not source.get("source") and not source.get("source_identity"):
        missing.append("source")
    criteria = source.get("criteria", source.get("query_criteria"))
    if not isinstance(criteria, Mapping) or not criteria:
        missing.append("criteria")
    else:
        required_criteria = ("lattice", "favorable", "reflexive", "full_dimensional")
        missing.extend(key for key in required_criteria if key not in criteria)
        if (
            criteria.get("lattice") != "N"
            or any(criteria.get(key) is not True for key in required_criteria[1:])
        ):
            missing.append("N_favorable_reflexive_full_dimensional_true_flags")
    if source.get("fresh") is not True:
        missing.append("fresh")
    if not isinstance(source.get("result_count"), int) or source["result_count"] < 0:
        missing.append("result_count")
    returned_order = source.get("returned_order")
    if not isinstance(returned_order, Sequence) or isinstance(returned_order, (str, bytes)):
        missing.append("returned_order")
    elif len(returned_order) != source["result_count"]:
        missing.append("returned_order_length")
    if not (
        source.get("source_revision")
        or source.get("revision")
        or source.get("local_mirror_hash")
        or source.get("source_manifest_sha256")
        or source.get("local_mirror_manifest_sha256")
    ):
        missing.append("source_revision_or_local_mirror_hash")
    if missing:
        raise ProvenanceError(
            "source_query_unrecorded",
            "source query cannot be replayed; missing " + ", ".join(missing),
            details={"missing": missing},
        )
    return source


def _without_digest(record):
    return {key: value for key, value in record.items() if key != "provenance_digest"}


def validate_provenance_digest(record):
    """Verify the self-digest of a provenance record."""
    recorded = record.get("provenance_digest")
    expected = stable_digest(_without_digest(record))
    if not recorded or recorded != expected:
        raise ProvenanceError(
            "missing_input_hash",
            "provenance digest is absent or does not match the recorded inputs",
            details={"recorded": recorded, "expected": expected},
        )
    return True


def collect_provenance(
    *,
    repo_root,
    task_file,
    source_files,
    source_query,
    output_root,
    input_roots=None,
    command_line=None,
    environment_overrides=None,
    host_settings=None,
    seed=None,
    derived_seeds=None,
    git_status_fixture=None,
    git_commit_fixture=None,
    git_branch_fixture=None,
):
    """Run the clean-tree gate and return a complete immutable provenance record.

    Call this before creating ``output_root``.  Every failure is raised as a
    :class:`ProvenanceError` with one of the task's terminal statuses.
    """
    root = Path(repo_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    git = git_identity(
        root,
        status_fixture=git_status_fixture,
        commit_fixture=git_commit_fixture,
        branch_fixture=git_branch_fixture,
    )
    if git["dirty"]:
        raise ProvenanceError(
            "provenance_dirty_tree",
            "production provenance requires a clean worktree",
            details={"git": git},
        )
    if destination.exists():
        raise ProvenanceError(
            "output_collision",
            f"production output root already exists: {destination}",
            details={"output_root": str(destination)},
        )

    task_path = Path(task_file).expanduser().resolve()
    task_hash = sha256_file(task_path)
    source_hashes, source_records = hash_source_files(source_files)
    if not source_hashes:
        raise ProvenanceError(
            "missing_input_hash",
            "at least one source input hash is required before production output",
        )
    query = _require_source_query(source_query)
    versions = environment_versions(environment_overrides)
    host, threads = host_and_thread_settings(host_settings)
    command = _command_record(command_line)
    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "provenance_validated",
        "repository": {
            "root": str(root),
            "commit": git["commit"],
            "branch": git["branch"],
            "status": git["status"],
            "status_porcelain": git["status_porcelain"],
        },
        "task_file": {"path": str(task_path), "sha256": task_hash},
        "source_hashes": source_hashes,
        "source_inputs": source_records,
        "source_query": query,
        "source_query_digest": stable_digest(query),
        "environment_versions": versions,
        "host_settings": host,
        "thread_settings": threads,
        "command_line": command,
        "seeds": {
            "base_seed": None if seed is None else int(seed),
            "derived_seeds": [] if derived_seeds is None else _jsonable(derived_seeds),
        },
        "roots": {
            "input_roots": _absolute_roots(input_roots),
            "output_root": str(destination),
        },
    }
    record["provenance_digest"] = stable_digest(record)
    return record


def clean_tree_provenance(**kwargs):
    """Alias for :func:`collect_provenance` used by integration call sites."""
    return collect_provenance(**kwargs)


def production_provenance_gate(**kwargs):
    """Alias emphasizing that the gate must run before bulk output creation."""
    return collect_provenance(**kwargs)


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "TERMINAL_STATUSES",
    "ProvenanceError",
    "canonical_json",
    "clean_tree_provenance",
    "collect_provenance",
    "environment_versions",
    "git_identity",
    "hash_source_files",
    "host_and_thread_settings",
    "production_provenance_gate",
    "sha256_file",
    "stable_digest",
    "stable_seed",
    "validate_provenance_digest",
]
