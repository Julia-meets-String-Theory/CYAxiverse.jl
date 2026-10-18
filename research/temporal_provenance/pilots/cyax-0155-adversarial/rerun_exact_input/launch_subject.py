#!/usr/bin/env python3
"""Launch exactly one fresh reconstruction subject through ``codex exec``.

The launcher is intentionally a direct ``subprocess.Popen`` call.  It passes
the committed request artifact as bytes to stdin, with ``shell=False`` and no
prompt interpolation.  Each run gets a new empty working directory, an
ephemeral/read-only Codex process, sanitized JSONL event capture, and the
exact final-message bytes emitted by ``--output-last-message``.  Local
thread/session identifiers are replaced with deterministic SHA-256
pseudonyms before event persistence.

This command is not a scoring tool.  It refuses repeated or out-of-order runs,
and any launch that returns tool/source-reopening events is durably marked
FAILED rather than retried.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
RUN_MANIFEST = RUNS_DIR / "manifest.json"
RUN_ORDER = ("A1", "B1", "B2", "A2", "A3", "B3")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DISPATCH = load_module("exact_dispatch", HERE / "exact_dispatch.py")
CAPTURE = load_module("capture_response", HERE / "capture_response.py")
SANITIZE = load_module("sanitize_identity", HERE / "sanitize_identity.py")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest() -> dict[str, Any]:
    return json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any]) -> None:
    CAPTURE.write_manifest(manifest)


def build_invocation(codex_argv0: str = "codex") -> list[str]:
    """Return the fixed argv; no prompt or source text is placed in argv."""
    return [
        codex_argv0,
        "exec",
        "--model",
        "gpt-5.6-sol",
        "-c",
        "model_reasoning_effort=high",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        "final.response",
        "-",
    ]


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def audit_events(event_bytes: bytes) -> dict[str, Any]:
    """Validate JSONL and reject tool, shell, browser, or command events."""
    lines = event_bytes.splitlines()
    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    identities: dict[str, str] = {}
    forbidden: list[dict[str, str]] = []
    blocked_types = (
        "tool", "function_call", "functioncall", "commandexecution", "shell",
        "browser", "mcp", "filesearch", "websearch",
    )
    identity_keys = ("thread_id", "session_id", "conversation_id", "turn_id")
    for index, line in enumerate(lines, 1):
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {index}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(event, dict):
            errors.append(f"line {index}: JSON event is not an object")
            continue
        parsed.append(event)
        for obj in _walk_dicts(event):
            for key in identity_keys:
                value = obj.get(key)
                if isinstance(value, str) and value:
                    identities.setdefault(key, value)
            event_type = obj.get("type")
            if isinstance(event_type, str):
                normalized = (
                    event_type.lower()
                    .replace("-", "")
                    .replace("_", "")
                    .replace(" ", "")
                )
                if any(pattern in normalized for pattern in blocked_types):
                    forbidden.append({"line": str(index), "type": event_type})
            for key in ("tool", "tool_name", "command", "shell_command", "function_call", "custom_tool_call", "mcp_tool_call"):
                if key in obj:
                    forbidden.append({"line": str(index), "field": key})
    return {
        "valid_jsonl": bool(lines) and not errors,
        "event_count": len(parsed),
        "errors": errors,
        "forbidden_events": forbidden,
        "tools_used": bool(forbidden),
        "source_reopenings": len(forbidden),
        "thread_id": identities.get("thread_id"),
        "session_id": identities.get("session_id"),
    }


def _entry(manifest: dict[str, Any], run_id: str) -> dict[str, Any]:
    for item in manifest.get("runs", []):
        if item.get("run_id") == run_id:
            return item
    raise RuntimeError(f"run {run_id} is not preregistered")


def preflight(manifest: dict[str, Any], run_id: str) -> tuple[dict[str, Any], bytes]:
    if tuple(manifest.get("launch_order", [])) != RUN_ORDER:
        raise RuntimeError("frozen launch order changed")
    if run_id not in RUN_ORDER:
        raise RuntimeError(f"unknown run ID {run_id}")
    entry = _entry(manifest, run_id)
    index = RUN_ORDER.index(run_id)
    entries = {item["run_id"]: item for item in manifest["runs"]}
    if any(entries[previous]["status"] != "CAPTURED" for previous in RUN_ORDER[:index]):
        raise RuntimeError(f"out-of-order launch refused for {run_id}")
    if entry.get("status") != "NOT_LAUNCHED":
        raise RuntimeError(f"repeated/non-fresh launch refused for {run_id}: {entry.get('status')}")
    DISPATCH.verify_artifacts(quiet=True)
    input_bytes = DISPATCH.verify_frozen_inputs()
    DISPATCH.verify_run_assignment(run_id, input_bytes)
    input_path = RUNS_DIR / entry["input_artifact"]
    data = input_path.read_bytes()
    if digest(data) != entry["input_sha256"]:
        raise RuntimeError(f"input hash changed for {run_id}")
    for key in ("event_artifact", "response_artifact"):
        if (RUNS_DIR / entry[key]).exists():
            raise RuntimeError(f"existing {key} blocks fresh launch for {run_id}")
    return entry, data


def _record_artifact(path: Path, data: bytes) -> dict[str, int | str]:
    CAPTURE.write_new(path, data)
    return {"bytes": len(data), "sha256": digest(data)}


def launch(run_id: str, codex_bin: str = "codex") -> dict[str, Any]:
    manifest = read_manifest()
    entry, input_bytes = preflight(manifest, run_id)
    argv = build_invocation(codex_bin)
    resolved = shutil.which(codex_bin)
    if resolved is None:
        raise RuntimeError(f"codex executable not found: {codex_bin}")
    # ``which`` only validates availability.  Popen receives the exact argv
    # recorded below, with shell=False; the OS performs PATH resolution.
    popen_argv = argv
    invocation = {
        "argv": argv,
        "shell": False,
        "stdin": "binary per-run input artifact",
        "cwd": "fresh empty temporary directory",
        "model": "gpt-5.6-sol",
        "reasoning": "high",
        "sandbox": "read-only",
        "ephemeral": True,
        "ignore_user_config": True,
        "ignore_rules": True,
        "skip_git_repo_check": True,
        "input_sha256": entry["input_sha256"],
    }
    entry["status"] = "RUNNING"
    entry["invocation"] = invocation
    entry["invocation_sha256"] = digest(json.dumps(invocation, sort_keys=True, separators=(",", ":")).encode())
    entry["launch_started_at"] = utc_now()
    write_manifest(manifest)

    stdout = b""
    stderr = b""
    final = b""
    returncode: int | None = None
    audit: dict[str, Any] = {}
    failure_reasons: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="cyax-0163-subject-") as isolated:
            isolated_path = Path(isolated)
            if any(isolated_path.iterdir()):
                raise RuntimeError("isolated working directory was not empty")
            process = subprocess.Popen(
                popen_argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=isolated,
                shell=False,
            )
            stdout, stderr = process.communicate(input=input_bytes)
            returncode = process.returncode
            final_path = isolated_path / "final.response"
            if final_path.is_file():
                final = final_path.read_bytes()
    except Exception as exc:  # preserve failure evidence before returning
        failure_reasons.append(f"launcher exception: {exc}")

    events_path = RUNS_DIR / entry["event_artifact"]
    event_original = {"bytes": len(stdout), "sha256": digest(stdout)}
    privacy_error: str | None = None
    try:
        sanitized_stdout, identity_sanitization = SANITIZE.sanitize_event_stream(stdout)
    except Exception as exc:
        # Never persist an unsanitized event stream.  Preserve only its
        # non-sensitive byte/hash provenance and mark the run failed.
        sanitized_stdout = b""
        identity_sanitization = {
            "schema": SANITIZE.SCHEMA,
            "original_bytes": len(stdout),
            "original_sha256": digest(stdout),
            "sanitized_bytes": 0,
            "sanitized_sha256": digest(b""),
            "hashed_identity_count": 0,
            "hashed_identities": [],
        }
        privacy_error = f"event identity sanitization failed: {exc}"
        failure_reasons.append(privacy_error)
    event_meta = _record_artifact(events_path, sanitized_stdout)
    # Keep only stderr's byte count/hash. Raw stderr can contain local paths
    # from the host and is not part of the subject response evidence.
    stderr_meta = {"bytes": len(stderr), "sha256": digest(stderr)}
    audit = audit_events(sanitized_stdout)
    if returncode != 0:
        failure_reasons.append(f"codex exit status {returncode}")
    if not audit["valid_jsonl"]:
        failure_reasons.append("stdout was not valid non-empty JSONL")
    if audit["forbidden_events"]:
        failure_reasons.append("tool/source-reopening event detected")
    if not final:
        failure_reasons.append("final-response artifact missing or empty")

    result: dict[str, Any] = {
        "event": event_meta,
        "event_original": event_original,
        "identity_sanitization": identity_sanitization,
        "stderr": stderr_meta,
        "returncode": returncode,
        "event_audit": audit,
        "thread_id": audit.get("thread_id"),
        "session_id": audit.get("session_id"),
        "completion_status": "COMPLETE" if not failure_reasons else "FAILED",
    }
    if final:
        response_path = RUNS_DIR / entry["response_artifact"]
        result["response"] = _record_artifact(response_path, final)
    if failure_reasons:
        entry["status"] = "FAILED"
        entry["failure_reasons"] = failure_reasons
    else:
        entry["status"] = "CAPTURED"
        entry["response"] = {
            **result["response"],
            "completion_status": "COMPLETE",
            "source_reopenings": audit["source_reopenings"],
        }
    entry["launch_finished_at"] = utc_now()
    entry["launch_result"] = result
    write_manifest(manifest)
    if failure_reasons:
        raise RuntimeError(f"{run_id} failed: {'; '.join(failure_reasons)}")
    return {"run_id": run_id, **result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=RUN_ORDER)
    parser.add_argument("--codex-bin", default="codex", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = launch(args.run_id, args.codex_bin)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
