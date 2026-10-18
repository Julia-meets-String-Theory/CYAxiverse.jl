#!/usr/bin/env python3
"""Validate and immutably split one blind scorer JSONL response."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import exact_dispatch


HERE = Path(__file__).resolve().parent
SCORING_DIR = HERE / "scoring"
SCORER_RESPONSE = SCORING_DIR / "scorer.response"
OUTPUT_DIR = SCORING_DIR / "blind_scorecards"
OPAQUE = tuple(f"S{i}" for i in range(1, 7))


def _validator():
    path = HERE.parent / "scoring" / "validate_scorecard.py"
    spec = importlib.util.spec_from_file_location("frozen_scorecard_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen scorecard validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse(data: bytes) -> dict[str, dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("scorer response is not UTF-8 JSONL") from exc
    lines = text.splitlines()
    if len(lines) != 6 or any(not line.strip() for line in lines):
        raise RuntimeError("scorer response must contain exactly six non-empty JSONL lines")
    validator = _validator()
    cards: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid scorecard JSON on line {index}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"scorecard line {index} is not an object")
        opaque = value.get("opaque_id")
        if opaque not in OPAQUE or opaque in cards:
            raise RuntimeError(f"scorecard line {index} has duplicate/invalid opaque_id")
        try:
            validator.validate(value)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"scorecard {opaque} failed frozen schema: {exc}") from exc
        cards[opaque] = value
    if set(cards) != set(OPAQUE):
        raise RuntimeError("scorer response does not contain S1-S6 exactly once")
    return cards


def materialize() -> dict[str, dict[str, Any]]:
    data = SCORER_RESPONSE.read_bytes()
    if not data:
        raise RuntimeError("scorer response is empty")
    cards = parse(data)
    for opaque in OPAQUE:
        serialized = (json.dumps(cards[opaque], indent=2, sort_keys=True) + "\n").encode("utf-8")
        exact_dispatch.write_immutable(OUTPUT_DIR / f"{opaque}.json", serialized)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=Path, default=None)
    args = parser.parse_args()
    global SCORER_RESPONSE
    if args.response is not None:
        SCORER_RESPONSE = args.response
    cards = materialize()
    print(json.dumps({"opaque_ids": list(cards)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
