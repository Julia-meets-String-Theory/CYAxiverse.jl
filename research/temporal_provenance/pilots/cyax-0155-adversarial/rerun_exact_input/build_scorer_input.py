#!/usr/bin/env python3
"""Build the frozen blind scorer stdin artifact.

Only the scorer prompt, the frozen answer key/schema, and opaque response
bytes are included.  The condition mapping, run manifest, invocation data,
and all response/run hashes are deliberately outside this artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import exact_dispatch


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
SCORING_DIR = HERE / "scoring"
PROMPT = SCORING_DIR / "scorer_prompt.md"
ANSWER_KEY = PILOT / "answer_key.md"
SCHEMA = PILOT / "scoring" / "scorecard_schema.json"
BLIND_DIR = SCORING_DIR / "blind_responses"
INPUT = SCORING_DIR / "scorer_input.input"
MANIFEST = SCORING_DIR / "scorer_input_manifest.json"
OPAQUE = tuple(f"S{i}" for i in range(1, 7))

ANSWER_DELIMITER = b"\n--- BEGIN FROZEN ANSWER KEY ---\n"
SCHEMA_DELIMITER = b"\n--- BEGIN FROZEN SCORECARD SCHEMA ---\n"
RESPONSE_DELIMITER = b"\n--- BEGIN BLIND RESPONSE "
RESPONSE_SUFFIX = b" ---\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_required(path: Path) -> bytes:
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"empty scorer material: {path}")
    return data


def build_bytes() -> tuple[bytes, dict[str, bytes]]:
    prompt = read_required(PROMPT)
    answer_key = read_required(ANSWER_KEY)
    schema = read_required(SCHEMA)
    responses: dict[str, bytes] = {}
    for opaque in OPAQUE:
        path = BLIND_DIR / f"{opaque}.response"
        responses[opaque] = read_required(path)
    pieces = [prompt, ANSWER_DELIMITER, answer_key, SCHEMA_DELIMITER, schema]
    for opaque in OPAQUE:
        pieces.extend((RESPONSE_DELIMITER, opaque.encode("ascii"), RESPONSE_SUFFIX, responses[opaque]))
    return b"".join(pieces), responses


def _metadata(data: bytes) -> dict[str, int | str]:
    return {"bytes": len(data), "sha256": sha256(data)}


def materialize() -> dict[str, Any]:
    data, responses = build_bytes()
    exact_dispatch.write_immutable(INPUT, data)
    observed = INPUT.read_bytes()
    if observed != data:
        raise RuntimeError("scorer input changed during materialization")
    manifest: dict[str, Any] = {
        "schema": "cyax-0163-blind-scorer-input-v1",
        "artifact": str(INPUT.relative_to(HERE)),
        "input": _metadata(data),
        "prompt": _metadata(PROMPT.read_bytes()),
        "answer_key": _metadata(ANSWER_KEY.read_bytes()),
        "scorecard_schema": _metadata(SCHEMA.read_bytes()),
        "opaque_responses": {opaque: _metadata(responses[opaque]) for opaque in OPAQUE},
        "opaque_ids": list(OPAQUE),
        "exclusions": [
            "condition_mapping.json",
            "runs/manifest.json",
            "dispatch_manifest.json",
            "source_snapshot.json",
            "run IDs, condition labels, invocation records, and response/run hashes",
        ],
    }
    exact_dispatch.write_immutable(
        MANIFEST,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def verify() -> None:
    manifest = materialize()
    expected, responses = build_bytes()
    observed = INPUT.read_bytes()
    if observed != expected:
        raise RuntimeError("scorer input bytes changed")
    if manifest["input"] != _metadata(observed):
        raise RuntimeError("scorer input manifest hash mismatch")
    for opaque in OPAQUE:
        if manifest["opaque_responses"][opaque] != _metadata(responses[opaque]):
            raise RuntimeError(f"opaque response metadata mismatch: {opaque}")
    # These are control-plane documents, not evidence terms in response text.
    prefix = observed[: len(observed) - sum(len(responses[o]) for o in OPAQUE)]
    for forbidden in (b"condition_mapping.json", b"runs/manifest.json", b"dispatch_manifest.json", b"source_snapshot.json"):
        if forbidden in prefix:
            raise RuntimeError(f"control-plane metadata leaked into scorer input: {forbidden!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = materialize()
    if args.verify:
        verify()
    print(json.dumps({"schema": manifest["schema"], "input": manifest["input"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
