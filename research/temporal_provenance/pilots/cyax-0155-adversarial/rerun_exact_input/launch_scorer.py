#!/usr/bin/env python3
"""Launch the one blind scorer through an exact stdin JSONL capture.

This command is intentionally separate from subject dispatch.  It is a
single-use launcher for a later scoring phase and is not invoked by tests or
the clean-rerun verifier.  The scorer receives only ``scorer_input.input``;
the working directory is fresh and empty and all tool/source events fail the
run.  A successful final response is split into six schema-valid scorecards.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import build_scorer_input
import exact_dispatch
import materialize_scorecards
import sanitize_identity


HERE = Path(__file__).resolve().parent
SCORING_DIR = HERE / "scoring"
EVENTS = SCORING_DIR / "scorer_events.jsonl"
FINAL = SCORING_DIR / "scorer.response"
RUN_RECORD = SCORING_DIR / "scorer_run.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_invocation(codex_argv0: str = "codex") -> list[str]:
    return [
        codex_argv0, "exec", "--model", "gpt-5.6-sol", "-c",
        "model_reasoning_effort=high", "--sandbox", "read-only", "--ephemeral",
        "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
        "--json", "--output-last-message", "scorer.response", "-",
    ]


def _write_record(value: dict[str, Any]) -> None:
    RUN_RECORD.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUN_RECORD.with_name(f".{RUN_RECORD.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(RUN_RECORD)


def _record(path: Path, data: bytes) -> dict[str, int | str]:
    exact_dispatch.write_immutable(path, data)
    return {"bytes": len(data), "sha256": sha256(data)}


def preflight() -> tuple[dict[str, Any], bytes]:
    build_scorer_input.verify()
    if RUN_RECORD.exists():
        record = json.loads(RUN_RECORD.read_text(encoding="utf-8"))
        if record.get("status") != "NOT_LAUNCHED":
            raise RuntimeError(f"repeated/non-fresh scorer launch refused: {record.get('status')}")
    if EVENTS.exists() or FINAL.exists():
        raise RuntimeError("existing scorer event/final artifact blocks fresh launch")
    data = build_scorer_input.INPUT.read_bytes()
    manifest = json.loads(build_scorer_input.MANIFEST.read_text(encoding="utf-8"))
    if manifest["input"] != {"bytes": len(data), "sha256": sha256(data)}:
        raise RuntimeError("scorer input hash changed")
    return manifest, data


def launch(codex_bin: str = "codex") -> dict[str, Any]:
    manifest, input_bytes = preflight()
    if shutil.which(codex_bin) is None:
        raise RuntimeError(f"codex executable not found: {codex_bin}")
    argv = build_invocation(codex_bin)
    invocation = {
        "argv": argv,
        "shell": False,
        "stdin": "exact scorer_input.input bytes",
        "cwd": "fresh empty temporary directory",
        "model": "gpt-5.6-sol",
        "reasoning": "high",
        "sandbox": "read-only",
        "ephemeral": True,
        "ignore_user_config": True,
        "ignore_rules": True,
        "skip_git_repo_check": True,
        "input_sha256": manifest["input"]["sha256"],
    }
    _write_record({
        "schema": "cyax-0163-blind-scorer-run-v1",
        "status": "RUNNING",
        "invocation": invocation,
        "invocation_sha256": sha256(json.dumps(invocation, sort_keys=True, separators=(",", ":")).encode()),
        "started_at": utc_now(),
    })
    stdout = b""
    stderr = b""
    final = b""
    returncode: int | None = None
    failure: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="cyax-0163-scorer-") as isolated:
            if any(Path(isolated).iterdir()):
                raise RuntimeError("isolated scorer directory was not empty")
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=isolated,
                shell=False,
            )
            stdout, stderr = process.communicate(input=input_bytes)
            returncode = process.returncode
            final_path = Path(isolated) / "scorer.response"
            if final_path.is_file():
                final = final_path.read_bytes()
    except Exception as exc:
        failure.append(f"launcher exception: {exc}")
    event_original = {"bytes": len(stdout), "sha256": sha256(stdout)}
    try:
        sanitized_stdout, identity_sanitization = sanitize_identity.sanitize_event_stream(stdout)
    except Exception as exc:
        sanitized_stdout = b""
        identity_sanitization = {
            "schema": sanitize_identity.SCHEMA,
            "original_bytes": len(stdout),
            "original_sha256": sha256(stdout),
            "sanitized_bytes": 0,
            "sanitized_sha256": sha256(b""),
            "hashed_identity_count": 0,
            "hashed_identities": [],
        }
        failure.append(f"event identity sanitization failed: {exc}")
    event_meta = _record(EVENTS, sanitized_stdout)
    audit = _audit(sanitized_stdout)
    if returncode != 0:
        failure.append(f"codex exit status {returncode}")
    if not audit["valid_jsonl"]:
        failure.append("stdout was not valid non-empty JSONL")
    if audit["forbidden_events"]:
        failure.append("tool/source-reopening event detected")
    if not final:
        failure.append("final scorer response missing or empty")
    if final and not failure:
        try:
            # Validate the exact captured bytes before creating scorecard files.
            materialize_scorecards.parse(final)
            # Save the exact final bytes first; the splitter never rewrites it.
            _record(FINAL, final)
            materialize_scorecards.materialize()
        except Exception as exc:
            failure.append(f"scorecard JSONL validation failed: {exc}")
    elif final:
        _record(FINAL, final)
    result = {
        "event": event_meta,
        "event_original": event_original,
        "identity_sanitization": identity_sanitization,
        "stderr": {"bytes": len(stderr), "sha256": sha256(stderr)},
        "returncode": returncode,
        "event_audit": audit,
        "thread_id": audit.get("thread_id"),
        "session_id": audit.get("session_id"),
        "completion_status": "COMPLETE" if not failure else "FAILED",
    }
    if final:
        result["response"] = {"bytes": len(final), "sha256": sha256(final)}
    record = {
        "schema": "cyax-0163-blind-scorer-run-v1",
        "status": "CAPTURED" if not failure else "FAILED",
        "invocation": invocation,
        "invocation_sha256": sha256(json.dumps(invocation, sort_keys=True, separators=(",", ":")).encode()),
        "started_at": json.loads(RUN_RECORD.read_text(encoding="utf-8")).get("started_at"),
        "finished_at": utc_now(),
        "result": result,
    }
    if failure:
        record["failure_reasons"] = failure
    _write_record(record)
    if failure:
        raise RuntimeError("blind scorer failed: " + "; ".join(failure))
    return record


def _audit(event_bytes: bytes) -> dict[str, Any]:
    # Reuse the subject launcher audit, which rejects command_execution after
    # normalization and collects the fresh thread/session identity.
    path = HERE / "launch_subject.py"
    spec = importlib.util.spec_from_file_location("subject_launcher_for_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load event auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_events(event_bytes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-bin", default="codex", help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(launch(args.codex_bin), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
