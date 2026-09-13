#!/usr/bin/env python3
"""Materialize immutable, opaque response copies after capture.

The raw ``runs/responses`` files are the evidence record and are never edited.
This module copies their bytes to ``scoring/blind_responses/S1.response`` ...
``S6.response`` according to the frozen private mapping.  The mapping is used
only to build a private audit manifest; it is not put in the scorer input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import exact_dispatch


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
RUN_MANIFEST = RUNS_DIR / "manifest.json"
SCORING_DIR = HERE / "scoring"
MAPPING = SCORING_DIR / "condition_mapping.json"
OUTPUT_DIR = SCORING_DIR / "blind_responses"
BLIND_MANIFEST = SCORING_DIR / "blind_materialization_manifest.json"
LEAKAGE_REPORT = SCORING_DIR / "blind_content_leakage.json"
OPAQUE = tuple(f"S{i}" for i in range(1, 7))
RUNS = ("A1", "B1", "B2", "A2", "A3", "B3")

LEAKAGE_PATTERNS = {
    "graph": re.compile(rb"\bgraph\w*\b", re.IGNORECASE),
    "relational": re.compile(rb"\brelational\w*\b", re.IGNORECASE),
    "condition_label": re.compile(rb"\b[AB][0-9]{1,3}\b", re.IGNORECASE),
    "condition_phrase": re.compile(rb"\bcondition\s+[AB]\b", re.IGNORECASE),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _mapping() -> dict[str, str]:
    document = load_json(MAPPING)
    if document.get("schema") != "cyax-0163-clean-rerun-blind-mapping-v1":
        raise RuntimeError("unsupported blind mapping schema")
    mapping = document.get("mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(OPAQUE):
        raise RuntimeError("mapping must contain exactly S1-S6")
    if set(mapping.values()) != set(RUNS):
        raise RuntimeError("mapping must be one-to-one over A1/B1/B2/A2/A3/B3")
    return {str(key): str(value) for key, value in mapping.items()}


def _captured_entries() -> dict[str, dict[str, Any]]:
    document = load_json(RUN_MANIFEST)
    entries = document.get("runs")
    if not isinstance(entries, list) or tuple(item.get("run_id") for item in entries) != RUNS:
        raise RuntimeError("run manifest is not the frozen six-run order")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        run_id = entry.get("run_id")
        if entry.get("status") != "CAPTURED":
            raise RuntimeError(f"blind materialization requires CAPTURED: {run_id}")
        response = entry.get("response")
        if not isinstance(response, dict):
            raise RuntimeError(f"missing captured response metadata: {run_id}")
        source = RUNS_DIR / str(entry["response_artifact"])
        if not source.is_file():
            raise RuntimeError(f"missing raw response artifact: {source}")
        data = source.read_bytes()
        observed = {"bytes": len(data), "sha256": sha256(data)}
        expected = {"bytes": response.get("bytes"), "sha256": response.get("sha256")}
        if observed != expected:
            raise RuntimeError(f"raw response hash mismatch: {run_id}")
        if not data:
            raise RuntimeError(f"empty raw response artifact: {run_id}")
        result[str(run_id)] = {"entry": entry, "source": source, "data": data}
    return result


def leakage_record(opaque: str, data: bytes) -> dict[str, Any]:
    terms: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for name, pattern in LEAKAGE_PATTERNS.items():
        matches = pattern.findall(data)
        decoded = sorted({match.decode("ascii", errors="replace") for match in matches})
        terms[name] = decoded
        counts[name] = len(matches)
    return {
        "opaque_id": opaque,
        "bytes": len(data),
        "sha256": sha256(data),
        "flagged": any(counts.values()),
        "counts": counts,
        "terms": terms,
        "action": "flag_only; response bytes were not changed",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    exact_dispatch.write_immutable(path, data)


def materialize() -> tuple[dict[str, Any], dict[str, Any]]:
    mapping = _mapping()
    sources = _captured_entries()
    items: dict[str, Any] = {}
    leakage: dict[str, Any] = {
        "schema": "cyax-0163-blind-content-leakage-v1",
        "scope": "opaque response bytes; substantive terms are flagged, never removed",
        "patterns": list(LEAKAGE_PATTERNS),
        "items": {},
    }
    for opaque in OPAQUE:
        run_id = mapping[opaque]
        source = sources[run_id]["source"]
        data = sources[run_id]["data"]
        destination = OUTPUT_DIR / f"{opaque}.response"
        exact_dispatch.write_immutable(destination, data)
        opaque_data = destination.read_bytes()
        if opaque_data != data:
            raise RuntimeError(f"opaque response is not byte-identical: {opaque}")
        items[opaque] = {
            "source_run": run_id,
            "source_artifact": str(source.relative_to(HERE)),
            "source_bytes": len(data),
            "source_sha256": sha256(data),
            "opaque_artifact": str(destination.relative_to(HERE)),
            "opaque_bytes": len(opaque_data),
            "opaque_sha256": sha256(opaque_data),
            "byte_identical": True,
        }
        leakage["items"][opaque] = leakage_record(opaque, opaque_data)
    manifest = {
        "schema": "cyax-0163-blind-materialization-v1",
        "mapping_schema": "cyax-0163-clean-rerun-blind-mapping-v1",
        "opaque_ids": list(OPAQUE),
        "items": items,
        "note": "Private audit manifest. Do not provide this mapping-bearing file to the scorer.",
    }
    _write_json(BLIND_MANIFEST, manifest)
    _write_json(LEAKAGE_REPORT, leakage)
    return manifest, leakage


def verify() -> None:
    manifest, leakage = materialize()
    if set(manifest["items"]) != set(OPAQUE):
        raise RuntimeError("blind manifest is incomplete")
    for opaque in OPAQUE:
        item = manifest["items"][opaque]
        path = HERE / item["opaque_artifact"]
        data = path.read_bytes()
        if item["opaque_sha256"] != sha256(data) or item["opaque_bytes"] != len(data):
            raise RuntimeError(f"opaque hash changed: {opaque}")
        if not item["byte_identical"]:
            raise RuntimeError(f"byte identity was not recorded: {opaque}")
        if leakage["items"][opaque]["sha256"] != sha256(data):
            raise RuntimeError(f"leakage report hash changed: {opaque}")
    names = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file())
    if names != [f"S{i}.response" for i in range(1, 7)]:
        raise RuntimeError("scorer-facing directory contains non-opaque filenames")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest, _ = materialize()
    if args.verify:
        verify()
    print(json.dumps({"schema": manifest["schema"], "opaque_ids": list(OPAQUE)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
