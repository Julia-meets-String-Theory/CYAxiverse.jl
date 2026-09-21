#!/usr/bin/env python3
"""Run the pinned Gate A package certification harness.

R-023 requires a certification run against an immutable package checkout.
This module creates a detached checkout from an exact candidate commit in a
temporary clone, runs the package test command there, and proves that the
checkout's tracked tree did not change.  It also compares the candidate tree
with the closed iteration and final release trees.

The package commit and harness revision are separate inputs.  The harness
revision and an explicit harness file path are required so that a
certification record cannot silently use an unidentified or ambient test
runner.  The expected SHA-256 must also be supplied; the digest is the
authoritative byte binding for the local harness file, while the separately
recorded revision identifies the reviewed harness source.  This module does
not assume that the harness revision belongs to the package repository.  The
result contains Git identities and digests only; temporary local paths and
test output are deliberately omitted from the public result.

The command is shell-free.  The command line accepts a shell-like string for
convenience, while callers that need exact argument boundaries should use
``certify_exact_tree_checkout`` directly with a sequence of arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import platform
from collections.abc import Sequence
from typing import Any


PASS = "PASS"
INVALID = "INVALID"
BLOCKED = "BLOCKED"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# Keep the package test route identical to the repository's compact package
# verifier.  The executable is resolved from PATH, but its identity is
# recorded independently below.  In particular, a caller cannot turn a
# successful certification into a no-op by supplying ``true`` or ``pass``.
APPROVED_PACKAGE_TEST_COMMAND = (
    "julia",
    "--startup-file=no",
    "--project=.",
    "-e",
    "using Pkg; Pkg.test()",
)
_APPROVED_PACKAGE_TEST_ARGS = APPROVED_PACKAGE_TEST_COMMAND[1:]
_PUBLIC_ENVIRONMENT_KEYS = frozenset(
    {
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_RUNNER_OS",
        "JULIA_CPU_TARGET",
        "JULIA_DEPOT_PATH",
        "JULIA_LOAD_PATH",
        "JULIA_NUM_THREADS",
        "JULIA_PKG_OFFLINE",
        "JULIA_PROJECT",
        "LANG",
        "LC_ALL",
        "PATH",
        "TMPDIR",
        "TZ",
    }
)


def _result(status: str, reason_code: str | None = None, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if reason_code is not None:
        result["reason_code"] = reason_code
    result.update(details)
    return result


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _run_git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run Git without exposing local paths in an exception or result."""

    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )


def _git_output(repository: Path, *args: str) -> str | None:
    result = _run_git(repository, *args)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _commit_tree(repository: Path, commit: str) -> str | None:
    """Resolve an exact commit SHA and its cryptographic root tree."""

    if not _sha(commit):
        return None
    resolved = _git_output(repository, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        return None
    tree = _git_output(repository, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if tree is None or not _sha(tree):
        return None
    return tree


def _tracked_status(repository: Path) -> str | None:
    """Return the porcelain status for tracked files, or ``None`` on error."""

    result = _run_git(repository, "status", "--porcelain=v2", "--untracked-files=no")
    if result.returncode != 0:
        return None
    return result.stdout


def _tracked_tree_is_clean(repository: Path, commit: str) -> tuple[bool, str | None]:
    """Prove staged and unstaged tracked content equals ``commit``.

    Git's commit tree is the exact cryptographic identity.  The two diff
    checks cover both the index and the working tree, including modes and
    submodule entries.  Untracked test artifacts are intentionally excluded:
    R-023 constrains the tracked package tree.
    """

    status = _tracked_status(repository)
    if status is None:
        return False, None
    if status:
        return False, status
    for args in (
        ("diff", "--no-ext-diff", "--ignore-submodules=none", "--quiet", commit, "--"),
        (
            "diff",
            "--cached",
            "--no-ext-diff",
            "--ignore-submodules=none",
            "--quiet",
            commit,
            "--",
        ),
    ):
        result = _run_git(repository, *args)
        if result.returncode != 0:
            return False, status
    return True, status


def _harness_sha256(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _status_sha256(status: str | None) -> str | None:
    if status is None:
        return None
    return hashlib.sha256(status.encode("utf-8", errors="replace")).hexdigest()


def _safe_command(command: Sequence[str]) -> bool:
    return (
        isinstance(command, Sequence)
        and not isinstance(command, (str, bytes))
        and bool(command)
        and all(isinstance(item, str) and item and "\x00" not in item for item in command)
    )


def _canonical_package_test_command(command: Sequence[str]) -> list[str] | None:
    """Return the public command identity when ``command`` is approved.

    Absolute paths are accepted for the Julia executable so callers can pin a
    toolchain without putting a local path in public evidence.  The remaining
    arguments are exact, which prevents shell wrappers, alternate projects,
    and no-op snippets from receiving a package-test PASS.
    """

    if not _safe_command(command) or len(command) != len(APPROVED_PACKAGE_TEST_COMMAND):
        return None
    executable = Path(command[0]).name
    if executable != APPROVED_PACKAGE_TEST_COMMAND[0]:
        return None
    if tuple(command[1:]) != _APPROVED_PACKAGE_TEST_ARGS:
        return None
    return [executable, *_APPROVED_PACKAGE_TEST_ARGS]


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _resolve_executable(command: Sequence[str], environment: dict[str, str]) -> Path | None:
    raw = command[0]
    if os.path.dirname(raw):
        candidate = Path(raw)
    else:
        resolved = shutil.which(raw, path=environment.get("PATH"))
        if resolved is None:
            return None
        candidate = Path(resolved)
    try:
        candidate = candidate.resolve(strict=True)
    except OSError:
        return None
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return None
    return candidate


def _test_environment_identity(
    environment: dict[str, str],
    *,
    executable_version: str,
    executable_version_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Build replayable, path-free runtime identity and its private preimage.

    The digest covers every environment variable used for the child process,
    while the public projection exposes only a small allowlist of names.  This
    binds ambient settings without publishing home directories, depot paths,
    credentials, or other local values.
    """

    environment_preimage = {
        "variables": [[name, environment[name]] for name in sorted(environment)],
        "os_name": os.name,
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "test_executable_version": executable_version,
        "test_executable_version_sha256": executable_version_sha256,
    }
    public = {
        "os_name": os.name,
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "test_executable_version": executable_version,
        "test_executable_version_sha256": executable_version_sha256,
        "environment_keys": sorted(
            name for name in environment if name in _PUBLIC_ENVIRONMENT_KEYS
        ),
    }
    return public, _json_sha256(environment_preimage)


def _test_execution_identity(
    test_command: Sequence[str],
) -> tuple[list[str], Path, dict[str, Any], dict[str, str], str, str, str] | None:
    """Resolve and bind command, executable, and child runtime identities."""

    public_command = _canonical_package_test_command(test_command)
    if public_command is None:
        return None
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    executable = _resolve_executable(test_command, environment)
    if executable is None:
        return None
    executable_sha256 = _file_sha256(executable)
    if executable_sha256 is None:
        return None
    try:
        version = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            timeout=10.0,
            env=environment,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    version_bytes = (
        version.stdout.encode("utf-8", errors="replace")
        + b"\0"
        + version.stderr.encode("utf-8", errors="replace")
    )
    executable_version_sha256 = hashlib.sha256(version_bytes).hexdigest()
    if version.returncode != 0:
        return None
    version_label = version.stdout.strip().splitlines()[0] if version.stdout.strip() else "unknown"
    if (
        len(version_label) > 160
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in version_label)
        or any(marker in version_label.lower() for marker in ("/", "\\", "private", "users", "home", "tmp"))
    ):
        version_label = f"sha256:{executable_version_sha256}"
    environment_public, environment_sha256 = _test_environment_identity(
        environment,
        executable_version=version_label,
        executable_version_sha256=executable_version_sha256,
    )
    command_sha256 = _json_sha256(public_command)
    return (
        public_command,
        executable,
        environment_public,
        environment,
        command_sha256,
        executable_sha256,
        environment_sha256,
    )


def certify_exact_tree_checkout(
    *,
    repository: str | Path,
    candidate_commit: str,
    closed_iteration_commit: str,
    release_commit: str,
    test_command: Sequence[str],
    harness_revision: str,
    harness_path: str | Path | None = None,
    expected_harness_sha256: str | None = None,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Certify an exact candidate checkout and return a JSON-safe result.

    ``candidate_commit``, ``closed_iteration_commit`` and ``release_commit``
    must be full lower-case commit SHAs.  Symbolic refs are rejected because
    the checkout must remain pinned for the complete test run.  The package
    test command runs with ``cwd`` set to the temporary candidate checkout and
    never through a shell.
    """

    root = Path(repository)
    if not root.is_dir() or not (root / ".git").exists():
        return _result(BLOCKED, "PACKAGE_REPOSITORY_UNAVAILABLE")
    if not all(_sha(value) for value in (candidate_commit, closed_iteration_commit, release_commit)):
        return _result(INVALID, "PACKAGE_COMMIT_IDENTITY_INVALID")
    if not _sha(harness_revision):
        return _result(BLOCKED, "HARNESS_REVISION_UNPINNED")
    if harness_path is None:
        return _result(BLOCKED, "HARNESS_PATH_UNPINNED")
    if expected_harness_sha256 is None:
        return _result(BLOCKED, "HARNESS_DIGEST_UNPINNED")
    if not _safe_command(test_command):
        return _result(BLOCKED, "PACKAGE_TEST_COMMAND_UNAVAILABLE")
    public_command = _canonical_package_test_command(test_command)
    if public_command is None:
        return _result(BLOCKED, "PACKAGE_TEST_COMMAND_UNAPPROVED")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        return _result(BLOCKED, "PACKAGE_TEST_TIMEOUT_INVALID")

    source_harness = Path(harness_path)
    harness_sha256 = _harness_sha256(source_harness)
    if harness_sha256 is None:
        return _result(BLOCKED, "HARNESS_IDENTITY_UNAVAILABLE")
    if not _digest(expected_harness_sha256):
        return _result(BLOCKED, "HARNESS_DIGEST_INVALID")
    if harness_sha256 != expected_harness_sha256:
        return _result(BLOCKED, "HARNESS_IDENTITY_MISMATCH")

    candidate_tree = _commit_tree(root, candidate_commit)
    closed_tree = _commit_tree(root, closed_iteration_commit)
    release_tree = _commit_tree(root, release_commit)
    if candidate_tree is None or closed_tree is None or release_tree is None:
        return _result(BLOCKED, "PACKAGE_COMMIT_UNAVAILABLE")
    identities: dict[str, Any] = {
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "closed_iteration_commit": closed_iteration_commit,
        "closed_iteration_tree": closed_tree,
        "release_commit": release_commit,
        "release_tree": release_tree,
        "package_commit": candidate_commit,
        "package_tree": candidate_tree,
        "certified_tree": candidate_tree,
        "harness_revision": harness_revision,
        "harness_sha256": harness_sha256,
    }
    if len({candidate_tree, closed_tree, release_tree}) != 1:
        return _result(INVALID, "CERTIFIED_TREE_MISMATCH", **identities)

    execution_identity = _test_execution_identity(test_command)
    if execution_identity is None:
        return _result(BLOCKED, "PACKAGE_TEST_EXECUTABLE_UNAVAILABLE", **identities)
    (
        public_command,
        test_executable_path,
        test_environment,
        test_process_environment,
        test_command_sha256,
        test_executable_sha256,
        test_environment_sha256,
    ) = execution_identity
    identities.update(
        {
            "test_command_argv": public_command,
            "test_command_sha256": test_command_sha256,
            "test_executable": public_command[0],
            "test_executable_sha256": test_executable_sha256,
            "test_environment": test_environment,
            "test_environment_sha256": test_environment_sha256,
            "runtime_environment": test_environment,
            "runtime_environment_sha256": test_environment_sha256,
        }
    )

    try:
        with tempfile.TemporaryDirectory(prefix="cyax-certification-") as temporary:
            checkout = Path(temporary) / "package"
            clone = subprocess.run(
                ["git", "clone", "--no-local", "--quiet", str(root), str(checkout)],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
            if clone.returncode != 0:
                return _result(BLOCKED, "IMMUTABLE_CHECKOUT_UNAVAILABLE", **identities)
            checkout_result = _run_git(checkout, "checkout", "--detach", "--quiet", candidate_commit)
            if checkout_result.returncode != 0:
                return _result(BLOCKED, "IMMUTABLE_CHECKOUT_UNAVAILABLE", **identities)

            head_before = _git_output(checkout, "rev-parse", "--verify", "HEAD")
            tree_before = _git_output(checkout, "rev-parse", "--verify", "HEAD^{tree}")
            if head_before != candidate_commit or tree_before != candidate_tree:
                return _result(INVALID, "CHECKOUT_TREE_MISMATCH", **identities)
            clean_before, status_before = _tracked_tree_is_clean(checkout, candidate_commit)
            if not clean_before:
                return _result(
                    INVALID,
                    "CHECKOUT_TRACKED_TREE_CHANGED",
                    tracked_files_changed=bool(status_before),
                    tracked_status_sha256=_status_sha256(status_before),
                    **identities,
                )

            try:
                test = subprocess.run(
                    [str(test_executable_path), *public_command[1:]],
                    cwd=checkout,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_seconds),
                    env=test_process_environment,
                )
            except FileNotFoundError:
                return _result(BLOCKED, "PACKAGE_TEST_COMMAND_UNAVAILABLE", **identities)
            except subprocess.TimeoutExpired:
                return _result(BLOCKED, "PACKAGE_TEST_TIMEOUT", **identities)

            head_after = _git_output(checkout, "rev-parse", "--verify", "HEAD")
            tree_after = _git_output(checkout, "rev-parse", "--verify", "HEAD^{tree}")
            clean_after, status_after = _tracked_tree_is_clean(checkout, candidate_commit)
            run_details = {
                "test_exit_code": test.returncode,
                "test_stdout_sha256": hashlib.sha256(test.stdout.encode("utf-8", errors="replace")).hexdigest(),
                    "test_stderr_sha256": hashlib.sha256(test.stderr.encode("utf-8", errors="replace")).hexdigest(),
                    "head_before": head_before,
                    "head_after": head_after,
                    "tracked_tree_before": tree_before,
                    "tracked_tree_after": tree_after,
            }
            if head_after != candidate_commit:
                return _result(INVALID, "CHECKOUT_HEAD_CHANGED", **identities, **run_details)
            if tree_after != candidate_tree or not clean_after:
                return _result(
                    INVALID,
                    "TRACKED_TREE_CHANGED",
                    tracked_files_changed=bool(status_after),
                    tracked_status_sha256=_status_sha256(status_after),
                    **identities,
                    **run_details,
                )
            if test.returncode != 0:
                return _result(INVALID, "PACKAGE_TEST_FAILED", **identities, **run_details)
            return _result(PASS, **identities, **run_details)
    except (OSError, subprocess.SubprocessError):
        return _result(BLOCKED, "IMMUTABLE_CHECKOUT_UNAVAILABLE", **identities)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", "--repo", required=True, type=Path)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--closed-iteration-commit", required=True)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--harness-revision", required=True)
    parser.add_argument(
        "--harness-path",
        type=Path,
        help="explicit local path to the separately pinned harness source",
    )
    parser.add_argument(
        "--expected-harness-sha256",
        help="expected SHA-256 for --harness-path; required for PASS",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--test-command",
        nargs=argparse.REMAINDER,
        required=True,
        help="package test command and arguments; all remaining tokens are captured",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command_tokens = list(args.test_command)
    if command_tokens[:1] == ["--"]:
        command_tokens = command_tokens[1:]
    try:
        # A single quoted command remains convenient for callers that invoke
        # this script through a process wrapper.  Multiple tokens preserve
        # exact argv boundaries and avoid shell parsing altogether.
        command = shlex.split(command_tokens[0]) if len(command_tokens) == 1 else command_tokens
    except ValueError as error:
        print(json.dumps(_result(BLOCKED, "PACKAGE_TEST_COMMAND_INVALID", detail=str(error)), sort_keys=True))
        return 1
    result = certify_exact_tree_checkout(
        repository=args.repository,
        candidate_commit=args.candidate_commit,
        closed_iteration_commit=args.closed_iteration_commit,
        release_commit=args.release_commit,
        test_command=command,
        harness_revision=args.harness_revision,
        harness_path=args.harness_path,
        expected_harness_sha256=args.expected_harness_sha256,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
