#!/usr/bin/env python3
"""Compact verification wrapper for coding agents.

Emit exactly one JSON summary line on stdout per invocation.
Full child output is captured in a temporary log directory outside the checkout.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

SCHEMA_VERSION = "agent-verify-1.0"

_SAFE_GIT_PREFIXES = frozenset((
    "status", "diff", "log", "show", "rev-parse", "describe",
    "ls-files", "ls-tree", "name-rev", "branch",
))

_FORBIDDEN_GIT_SUBCOMMANDS = frozenset((
    "commit", "push", "pull", "fetch", "merge", "rebase",
    "checkout", "switch", "reset", "restore", "clean",
    "stash", "tag",
))


def _resolve_repo_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=script_dir,
        )
    except FileNotFoundError:
        return None, "git executable not found"
    if result.returncode != 0:
        return None, "not inside a Git worktree"
    return result.stdout.strip(), None


def _make_log_dir():
    return tempfile.mkdtemp(prefix="agent_verify_")


def _tail_nonempty(text, n=40):
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-n:]


def _capture_git_state(repo_root):
    state = {}
    for label, cmd in [
        ("status", ["git", "status", "--short", "--branch"]),
        ("diff_stat", ["git", "diff", "--stat"]),
        ("diff_names", ["git", "diff", "--name-only"]),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
            state[label] = r.stdout.strip()
        except Exception:
            state[label] = None
    try:
        r = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, cwd=repo_root,
        )
        state["untracked"] = r.stdout.strip()
    except Exception:
        state["untracked"] = None
    return state


def _run_child(argv, repo_root, log_dir, log_name):
    stdout_path = os.path.join(log_dir, f"{log_name}.stdout.log")
    stderr_path = os.path.join(log_dir, f"{log_name}.stderr.log")
    t0 = time.monotonic()
    try:
        with open(stdout_path, "w") as fo, open(stderr_path, "w") as fe:
            proc = subprocess.run(
                argv, stdout=fo, stderr=fe,
                cwd=repo_root, shell=False,
            )
        elapsed = time.monotonic() - t0
        with open(stdout_path) as f:
            out_text = f.read()
        with open(stderr_path) as f:
            err_text = f.read()
        return proc.returncode, elapsed, out_text, err_text, \
            {"stdout": stdout_path, "stderr": stderr_path}
    except FileNotFoundError as exc:
        elapsed = time.monotonic() - t0
        msg = str(exc)
        with open(stderr_path, "w") as f:
            f.write(msg + "\n")
        return None, elapsed, "", msg, \
            {"stdout": stdout_path, "stderr": stderr_path}


def _emit(result):
    json.dump(result, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _build_result(command, status, exit_code, duration, log_paths,
                  summary, warnings, git_state):
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": round(duration, 3),
        "log_paths": log_paths,
        "summary": summary,
        "warnings": warnings,
        "git_state": git_state,
    }


def _failure_summary(out_text, err_text, log_paths):
    combined = (out_text or "") + "\n" + (err_text or "")
    tail = _tail_nonempty(combined)
    parts = ["Last output:"] + tail
    parts.append(f"Full logs: {log_paths.get('stdout', '?')}")
    return "\n".join(parts)


def cmd_snapshot(repo_root, _args):
    log_dir = _make_log_dir()
    t0 = time.monotonic()
    git_state = _capture_git_state(repo_root)
    elapsed = time.monotonic() - t0
    result = _build_result(
        command="snapshot", status="passed", exit_code=0,
        duration=elapsed, log_paths={"log_dir": log_dir},
        summary="worktree snapshot captured",
        warnings=[], git_state=git_state,
    )
    _emit(result)
    return 0


def cmd_diff_check(repo_root, _args):
    log_dir = _make_log_dir()
    rc, elapsed, out, err, logs = _run_child(
        ["git", "diff", "--check"], repo_root, log_dir, "diff_check",
    )
    git_state = _capture_git_state(repo_root)
    if rc == 0:
        result = _build_result(
            command="diff-check", status="passed", exit_code=0,
            duration=elapsed, log_paths=logs,
            summary="no whitespace errors", warnings=[], git_state=git_state,
        )
        _emit(result)
        return 0
    if rc is None:
        result = _build_result(
            command="diff-check", status="unavailable", exit_code=127,
            duration=elapsed, log_paths=logs,
            summary="git not found", warnings=[err], git_state=git_state,
        )
        _emit(result)
        return 127
    summary = _failure_summary(out, err, logs)
    result = _build_result(
        command="diff-check", status="failed", exit_code=rc,
        duration=elapsed, log_paths=logs,
        summary=summary, warnings=[], git_state=git_state,
    )
    _emit(result)
    return rc


def cmd_run(repo_root, args):
    if not args.child_argv:
        result = _build_result(
            command="run", status="failed", exit_code=1,
            duration=0, log_paths={},
            summary="no command provided after --",
            warnings=[], git_state=_capture_git_state(repo_root),
        )
        _emit(result)
        return 1
    log_dir = _make_log_dir()
    rc, elapsed, out, err, logs = _run_child(
        args.child_argv, repo_root, log_dir, "run",
    )
    git_state = _capture_git_state(repo_root)
    if rc is None:
        result = _build_result(
            command="run", status="unavailable", exit_code=127,
            duration=elapsed, log_paths=logs,
            summary=f"executable not found: {args.child_argv[0]}",
            warnings=[err], git_state=git_state,
        )
        _emit(result)
        return 127
    if rc == 0:
        result = _build_result(
            command="run", status="passed", exit_code=0,
            duration=elapsed, log_paths=logs,
            summary="command succeeded", warnings=[], git_state=git_state,
        )
        _emit(result)
        return 0
    summary = _failure_summary(out, err, logs)
    result = _build_result(
        command="run", status="failed", exit_code=rc,
        duration=elapsed, log_paths=logs,
        summary=summary, warnings=[], git_state=git_state,
    )
    _emit(result)
    return rc


def cmd_package(repo_root, _args):
    julia = shutil.which("julia")
    if julia is None:
        result = _build_result(
            command="package", status="unavailable", exit_code=127,
            duration=0, log_paths={},
            summary="julia not found on PATH",
            warnings=["Julia must run in the approved regular local host "
                       "environment, never a sandbox or container."],
            git_state=_capture_git_state(repo_root),
        )
        _emit(result)
        return 127
    log_dir = _make_log_dir()
    argv = [julia, "--startup-file=no", "--project=.",
            "-e", "using Pkg; Pkg.test()"]
    rc, elapsed, out, err, logs = _run_child(argv, repo_root, log_dir, "package")
    git_state = _capture_git_state(repo_root)
    if rc == 0:
        result = _build_result(
            command="package", status="passed", exit_code=0,
            duration=elapsed, log_paths=logs,
            summary="Pkg.test() passed", warnings=[], git_state=git_state,
        )
        _emit(result)
        return 0
    summary = _failure_summary(out, err, logs)
    result = _build_result(
        command="package", status="failed", exit_code=rc,
        duration=elapsed, log_paths=logs,
        summary=summary, warnings=[], git_state=git_state,
    )
    _emit(result)
    return rc


def cmd_python_regressions(repo_root, _args):
    script = os.path.join(repo_root, "scripts", "run_python_regression_tests.py")
    if not os.path.isfile(script):
        result = _build_result(
            command="python-regressions", status="unavailable", exit_code=127,
            duration=0, log_paths={},
            summary="scripts/run_python_regression_tests.py not found",
            warnings=[], git_state=_capture_git_state(repo_root),
        )
        _emit(result)
        return 127
    log_dir = _make_log_dir()
    rc, elapsed, out, err, logs = _run_child(
        [sys.executable, script], repo_root, log_dir, "python_regressions",
    )
    git_state = _capture_git_state(repo_root)
    if rc is None:
        result = _build_result(
            command="python-regressions", status="unavailable", exit_code=127,
            duration=elapsed, log_paths=logs,
            summary="Python interpreter not found",
            warnings=[err], git_state=git_state,
        )
        _emit(result)
        return 127
    if rc == 0:
        result = _build_result(
            command="python-regressions", status="passed", exit_code=0,
            duration=elapsed, log_paths=logs,
            summary="Python regression tests passed",
            warnings=[], git_state=git_state,
        )
        _emit(result)
        return 0
    summary = _failure_summary(out, err, logs)
    result = _build_result(
        command="python-regressions", status="failed", exit_code=rc,
        duration=elapsed, log_paths=logs,
        summary=summary, warnings=[], git_state=git_state,
    )
    _emit(result)
    return rc


def cmd_all(repo_root, args):
    exit_code = cmd_diff_check(repo_root, args)
    if exit_code != 0:
        return exit_code
    return cmd_package(repo_root, args)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Token-efficient verification for coding agents.",
    )
    sub = parser.add_subparsers(dest="subcommand")
    sub.add_parser("snapshot", help="Capture worktree state.")
    sub.add_parser("diff-check", help="Run git diff --check.")
    run_p = sub.add_parser("run", help="Run an explicit command.")
    run_p.add_argument("child_argv", nargs="*",
                       help="Command and arguments (place after --).")
    sub.add_parser("package", help="Run julia Pkg.test().")
    sub.add_parser("python-regressions",
                    help="Run Python regression tests.")
    sub.add_parser("all", help="diff-check then package; stop on failure.")

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help(sys.stderr)
        sys.exit(2)

    repo_root, err = _resolve_repo_root()
    if repo_root is None:
        result = _build_result(
            command=args.subcommand, status="unavailable", exit_code=128,
            duration=0, log_paths={}, summary=err, warnings=[],
            git_state={},
        )
        _emit(result)
        sys.exit(128)

    dispatch = {
        "snapshot": cmd_snapshot,
        "diff-check": cmd_diff_check,
        "run": cmd_run,
        "package": cmd_package,
        "python-regressions": cmd_python_regressions,
        "all": cmd_all,
    }
    handler = dispatch[args.subcommand]
    exit_code = handler(repo_root, args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
