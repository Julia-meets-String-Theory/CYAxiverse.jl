#!/usr/bin/env python3
"""Capture one fresh-subject response as immutable bytes.

The response must be supplied through stdin by the execution surface.  This
command does not decode, trim, normalize, or otherwise rewrite it.  It records
the response hash and completion metadata in the clean-rerun manifest only
after the exact response artifact has been created.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_MANIFEST = HERE / "runs" / "manifest.json"
RUNS_DIR = HERE / "runs"


def load_dispatch_module():
    spec = importlib.util.spec_from_file_location("exact_dispatch", HERE / "exact_dispatch.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load exact dispatch module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing response: {path}")
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


def write_manifest(manifest: dict) -> None:
    temporary = RUN_MANIFEST.with_name(f".{RUN_MANIFEST.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, RUN_MANIFEST)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=["A1", "B1", "B2", "A2", "A3", "B3"])
    parser.add_argument("--completion-status", default="COMPLETE")
    parser.add_argument("--source-reopenings", type=int, default=0)
    args = parser.parse_args()
    if args.source_reopenings < 0:
        raise SystemExit("--source-reopenings must be nonnegative")

    dispatch = load_dispatch_module()
    dispatch.verify_artifacts()
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    entry = next((item for item in manifest["runs"] if item["run_id"] == args.run_id), None)
    if entry is None:
        raise RuntimeError(f"run {args.run_id} is not in the frozen run manifest")
    if entry["status"] not in {"NOT_LAUNCHED", "DISPATCHED"}:
        raise RuntimeError(f"run {args.run_id} already has status {entry['status']}")

    data = sys.stdin.buffer.read()
    if not data:
        raise RuntimeError("empty response is not a complete capture")
    expected_input = dispatch.load_manifest()["inputs"][entry["condition"]]
    input_path = RUNS_DIR / entry["input_artifact"]
    if dispatch.sha256(input_path.read_bytes()) != expected_input["sha256"]:
        raise RuntimeError(f"run {args.run_id} input artifact changed")

    response_path = RUNS_DIR / entry["response_artifact"]
    write_new(response_path, data)
    entry["status"] = "CAPTURED"
    entry["response"] = {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "completion_status": args.completion_status,
        "source_reopenings": args.source_reopenings,
    }
    manifest["status"] = "PARTIALLY_CAPTURED"
    if all(item["status"] == "CAPTURED" for item in manifest["runs"]):
        manifest["status"] = "ALL_CAPTURED"
    write_manifest(manifest)
    print(json.dumps({"run_id": args.run_id, **entry["response"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
