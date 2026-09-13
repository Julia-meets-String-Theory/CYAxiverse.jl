#!/usr/bin/env python3
"""One-time privacy migration for captured event streams and run records.

This command never reads or writes frozen inputs, response artifacts, or
scorecards.  It replaces only local Codex identity fields in event JSONL and
the associated run records with deterministic SHA-256 pseudonyms.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import launch_subject
import sanitize_identity


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
RUN_MANIFEST = RUNS_DIR / "manifest.json"
SCORING_DIR = HERE / "scoring"
SCORER_EVENTS = SCORING_DIR / "scorer_events.jsonl"
SCORER_RUN = SCORING_DIR / "scorer_run.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.privacy-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes(path, data)


def _sanitize_event(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    original = path.read_bytes()
    sanitized, privacy = sanitize_identity.sanitize_event_stream(original)
    if sanitized != original:
        _write_bytes(path, sanitized)
    audit = launch_subject.audit_events(sanitized)
    if not audit["valid_jsonl"] or audit["forbidden_events"]:
        raise RuntimeError(f"sanitized event failed no-tool audit: {path}")
    original_meta = {"bytes": len(original), "sha256": _sha256(original)}
    sanitized_meta = {"bytes": len(sanitized), "sha256": _sha256(sanitized)}
    return original_meta, sanitized_meta, {"privacy": privacy, "audit": audit}


def _update_subject_manifest() -> list[dict[str, Any]]:
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("identity_sanitization", {}).get("status") == "SANITIZED":
        raise RuntimeError("subject event artifacts are already privacy-sanitized")
    records = []
    for entry in manifest["runs"]:
        event_path = RUNS_DIR / entry["event_artifact"]
        original_meta, sanitized_meta, details = _sanitize_event(event_path)
        result = entry.get("launch_result")
        if not isinstance(result, dict):
            raise RuntimeError(f"missing launch result: {entry['run_id']}")
        result["event_original"] = original_meta
        result["event"] = sanitized_meta
        result["identity_sanitization"] = details["privacy"]
        result["event_audit"] = details["audit"]
        result["thread_id"] = details["audit"].get("thread_id")
        result["session_id"] = details["audit"].get("session_id")
        records.append({"run_id": entry["run_id"], "original": original_meta, "sanitized": sanitized_meta})
    manifest["identity_sanitization"] = {
        "schema": sanitize_identity.SCHEMA,
        "status": "SANITIZED",
        "identity_fields": list(sanitize_identity.IDENTITY_KEYS),
        "value_encoding": "sha256:<64 lowercase hex> with a fixed domain prefix",
        "event_streams": records,
        "note": "Original event bytes are not retained; original byte/hash provenance is recorded per run.",
    }
    _write_json(RUN_MANIFEST, manifest)
    return records


def _update_scorer_record() -> dict[str, Any] | None:
    if not SCORER_EVENTS.exists() or not SCORER_RUN.exists():
        return None
    original_meta, sanitized_meta, details = _sanitize_event(SCORER_EVENTS)
    record = json.loads(SCORER_RUN.read_text(encoding="utf-8"))
    if record.get("identity_sanitization", {}).get("status") == "SANITIZED":
        raise RuntimeError("scorer event artifacts are already privacy-sanitized")
    result = record.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("scorer run record has no result")
    result["event_original"] = original_meta
    result["event"] = sanitized_meta
    result["identity_sanitization"] = details["privacy"]
    result["event_audit"] = details["audit"]
    result["thread_id"] = details["audit"].get("thread_id")
    result["session_id"] = details["audit"].get("session_id")
    record["identity_sanitization"] = {
        "schema": sanitize_identity.SCHEMA,
        "status": "SANITIZED",
        "identity_fields": list(sanitize_identity.IDENTITY_KEYS),
        "value_encoding": "sha256:<64 lowercase hex> with a fixed domain prefix",
        "event_stream": {"original": original_meta, "sanitized": sanitized_meta},
        "note": "Original event bytes are not retained; original byte/hash provenance is recorded.",
    }
    _write_json(SCORER_RUN, record)
    return {"original": original_meta, "sanitized": sanitized_meta}


def main() -> int:
    subject_records = _update_subject_manifest()
    scorer_record = _update_scorer_record()
    print(json.dumps({"subject_event_streams": subject_records, "scorer_event_stream": scorer_record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
