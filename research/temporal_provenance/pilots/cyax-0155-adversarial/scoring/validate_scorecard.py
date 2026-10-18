#!/usr/bin/env python3
"""Strict offline validation for a scorer-facing opaque scorecard."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = [f"K{i}" for i in range(1, 13)]
CRITICAL = ["K6", "K7", "K8", "K9", "K11"]
INT_FIELDS = [
    "raw_total", "conflict_critical", "unsupported_assertions", "authority_errors",
    "temporal_errors", "supersession_errors", "unjustified_inferences",
    "correct_abstentions", "incorrect_abstentions", "source_reopenings",
]
BOOL_FIELDS = ["automatic_failure", "next_action_correct"]


def fail(message: str) -> None:
    raise ValueError(message)


def validate(value: object) -> None:
    schema = json.loads((HERE / "scorecard_schema.json").read_text())
    if not isinstance(value, dict):
        fail("root must be an object")
    required = set(schema["required"])
    if set(value) != required:
        fail(f"root fields mismatch: missing={sorted(required - set(value))}, unexpected={sorted(set(value) - required)}")
    if not isinstance(value["opaque_id"], str) or not re.fullmatch(r"S[1-6]", value["opaque_id"]):
        fail("opaque_id must match S1-S6")
    scores = value["scores"]
    if not isinstance(scores, dict) or set(scores) != set(KEYS):
        fail("scores must contain exactly K1-K12")
    if any(type(scores[key]) is not int or scores[key] not in (0, 1) for key in KEYS):
        fail("every score must be integer 0 or 1")
    for field in INT_FIELDS:
        if type(value[field]) is not int or value[field] < 0:
            fail(f"{field} must be a nonnegative integer")
    if value["raw_total"] != sum(scores.values()):
        fail("raw_total must equal sum K1-K12")
    if value["conflict_critical"] != sum(scores[key] for key in CRITICAL):
        fail("conflict_critical must equal sum K6,K7,K8,K9,K11")
    if value["raw_total"] > 12 or value["conflict_critical"] > 5:
        fail("score exceeds maximum")
    for field in BOOL_FIELDS:
        if type(value[field]) is not bool:
            fail(f"{field} must be boolean")
    reasons = value["automatic_failure_reasons"]
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        fail("automatic_failure_reasons must be an array of strings")
    if value["automatic_failure"] != bool(reasons):
        fail("automatic_failure must be true exactly when reasons are nonempty")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} SCORECARD.json", file=sys.stderr)
        return 2
    try:
        card = json.loads(Path(sys.argv[1]).read_text())
        validate(card)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    print(f"PASS: {card['opaque_id']} raw={card['raw_total']}/12 critical={card['conflict_critical']}/5 automatic_failure={card['automatic_failure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
