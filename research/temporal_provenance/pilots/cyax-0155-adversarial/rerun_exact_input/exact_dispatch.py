#!/usr/bin/env python3
"""Materialize and verify exact-byte inputs for the clean CYAX-0163 rerun.

This module is deliberately byte-oriented.  It never uses ``read_text`` or
``write_text`` for frozen material and it never formats, normalizes, or
regenerates a prompt/context.  The only permitted input construction is:

    common_prompt_bytes + delimiter_bytes + condition_context_bytes

The committed dispatch manifest is the preregistration record for the full
input hashes.  ``materialize`` creates the two input artifacts once; later
invocations refuse to overwrite an artifact unless its bytes are identical.
``emit`` writes only the selected artifact to stdout's binary stream so a
byte-preserving execution surface can consume it without an interactive copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
ROOT = HERE.parents[4]
MANIFEST_PATH = HERE / "dispatch_manifest.json"
RUN_MANIFEST_PATH = HERE / "runs" / "manifest.json"
RUNS_DIR = HERE / "runs"

EXPECTED_FILES = {
    "common_prompt": {
        "path": PILOT / "common_subject_prompt.md",
        "bytes": 899,
        "words": 122,
        "sha256": "cf9b6aec4c7e232a9c2462e2f100dcefb6e49ea67422ce7670c4f85e66f3c52a",
    },
    "condition_a_context": {
        "path": PILOT / "condition_a_context.md",
        "bytes": 6429,
        "words": 852,
        "sha256": "aedd1350bef5389be90d2608318f7b9675f94aad0a50a8d5d59499fb017bc780",
    },
    "condition_b_context": {
        "path": PILOT / "condition_b_context.md",
        "bytes": 6702,
        "words": 895,
        "sha256": "33c61ad16a5228a557ae59f0ac80834674e40aae5b77a0a9e0044e82c1b2237c",
    },
}

DELIMITER = b"\n--- BEGIN FROZEN CONTEXT ---\n"
EXPECTED_DELIMITER_SHA256 = "aa23570e27878cc6f2b1871145b5cca8a53128ee24ecaa12af68aca8f6f3f2f3"

RUN_TO_CONDITION = {
    "A1": "A",
    "B1": "B",
    "B2": "B",
    "A2": "A",
    "A3": "A",
    "B3": "B",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words(data: bytes) -> int:
    return len(data.decode("utf-8").split())


def read_frozen(name: str) -> bytes:
    expected = EXPECTED_FILES[name]
    data = expected["path"].read_bytes()
    observed = {"bytes": len(data), "words": words(data), "sha256": sha256(data)}
    required = {
        "bytes": expected["bytes"],
        "words": expected["words"],
        "sha256": expected["sha256"],
    }
    if observed != required:
        raise RuntimeError(
            f"frozen {name} changed: expected {required}, observed {observed}"
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"frozen {name} is not UTF-8") from exc
    return data


def expected_inputs() -> dict[str, bytes]:
    prompt = read_frozen("common_prompt")
    delimiter_hash = sha256(DELIMITER)
    if len(DELIMITER) != 30 or delimiter_hash != EXPECTED_DELIMITER_SHA256:
        raise RuntimeError("dispatch delimiter changed")
    return {
        "A": prompt + DELIMITER + read_frozen("condition_a_context"),
        "B": prompt + DELIMITER + read_frozen("condition_b_context"),
    }


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cyax-0163-exact-dispatch-v1":
        raise RuntimeError("unsupported dispatch manifest schema")
    if manifest.get("delimiter", {}).get("sha256") != EXPECTED_DELIMITER_SHA256:
        raise RuntimeError("manifest delimiter hash mismatch")
    return manifest


def verify_manifest_inputs(manifest: dict, input_bytes: dict[str, bytes]) -> None:
    for condition, data in input_bytes.items():
        record = manifest["inputs"][condition]
        observed = {
            "bytes": len(data),
            "words": words(data),
            "sha256": sha256(data),
        }
        expected = {
            "bytes": record["bytes"],
            "words": record["words"],
            "sha256": record["sha256"],
        }
        if observed != expected:
            raise RuntimeError(
                f"preregistered {condition} input hash mismatch: "
                f"expected {expected}, observed {observed}"
            )


def verify_frozen_inputs() -> dict[str, bytes]:
    manifest = load_manifest()
    for name, expected in EXPECTED_FILES.items():
        record = manifest["frozen_files"][name]
        if record["sha256"] != expected["sha256"] or record["bytes"] != expected["bytes"]:
            raise RuntimeError(f"manifest frozen identity mismatch for {name}")
    prompt = read_frozen("common_prompt")
    for forbidden in (b"Condition A", b"Condition B", b"A1", b"A2", b"A3", b"B1", b"B2", b"B3", b"answer key", b"expected result"):
        if forbidden.lower() in prompt.lower():
            raise RuntimeError(f"common prompt contains forbidden dispatch metadata: {forbidden!r}")
    input_bytes = expected_inputs()
    verify_manifest_inputs(manifest, input_bytes)
    return input_bytes


def write_immutable(path: Path, data: bytes) -> None:
    """Create an artifact or prove an existing artifact is byte-identical."""
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize() -> None:
    manifest = load_manifest()
    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    input_bytes = verify_frozen_inputs()
    for condition, data in input_bytes.items():
        write_immutable(HERE / manifest["inputs"][condition]["artifact"], data)
    for run_id, condition in RUN_TO_CONDITION.items():
        entry = next(item for item in run_manifest["runs"] if item["run_id"] == run_id)
        write_immutable(RUNS_DIR / entry["input_artifact"], input_bytes[condition])
    print("PASS: exact A/B input artifacts materialized or already byte-identical")


def verify_artifacts(quiet: bool = False) -> None:
    manifest = load_manifest()
    input_bytes = verify_frozen_inputs()
    for condition, data in input_bytes.items():
        path = HERE / manifest["inputs"][condition]["artifact"]
        if not path.is_file():
            raise RuntimeError(f"missing exact input artifact: {path}")
        if path.read_bytes() != data:
            raise RuntimeError(f"exact input artifact changed: {path}")
    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in run_manifest["runs"]:
        path = RUNS_DIR / entry["input_artifact"]
        condition = entry["condition"]
        if not path.is_file() or path.read_bytes() != input_bytes[condition]:
            raise RuntimeError(f"per-run input artifact changed or missing: {path}")
    if not quiet:
        print("PASS: frozen source identities, delimiter, full input hashes, and artifacts verified")


def verify_run_assignment(run_id: str, input_bytes: dict[str, bytes]) -> None:
    """Assert the assigned run's preregistered hash before emitting bytes."""
    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    if tuple(run_manifest.get("launch_order", [])) != tuple(RUN_TO_CONDITION):
        raise RuntimeError("run launch order changed")
    entry = next((item for item in run_manifest.get("runs", []) if item.get("run_id") == run_id), None)
    if entry is None:
        raise RuntimeError(f"run {run_id} is not preregistered")
    condition = RUN_TO_CONDITION[run_id]
    if entry.get("condition") != condition:
        raise RuntimeError(f"run {run_id} condition assignment changed")
    expected_hash = sha256(input_bytes[condition])
    if entry.get("input_sha256") != expected_hash:
        raise RuntimeError(f"run {run_id} preregistered input hash changed")
    input_path = RUNS_DIR / entry["input_artifact"]
    if not input_path.is_file() or sha256(input_path.read_bytes()) != expected_hash:
        raise RuntimeError(f"run {run_id} input artifact changed or missing")
    if entry.get("status") not in {"NOT_LAUNCHED", "DISPATCHED"}:
        raise RuntimeError(f"run {run_id} is not dispatchable from status {entry.get('status')}")


def emit(run_id: str) -> None:
    if run_id not in RUN_TO_CONDITION:
        raise SystemExit(f"unknown run id {run_id}; expected {', '.join(RUN_TO_CONDITION)}")
    manifest = load_manifest()
    input_bytes = verify_frozen_inputs()
    verify_artifacts(quiet=True)
    verify_run_assignment(run_id, input_bytes)
    condition = RUN_TO_CONDITION[run_id]
    run_entry = next(
        item for item in json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))["runs"]
        if item["run_id"] == run_id
    )
    path = RUNS_DIR / run_entry["input_artifact"]
    # Never print status to stdout: stdout is the exact byte stream.
    sys.stdout.buffer.write(path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize")
    subparsers.add_parser("verify")
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("run_id")
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()
    elif args.command == "verify":
        verify_artifacts()
    elif args.command == "emit":
        emit(args.run_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
