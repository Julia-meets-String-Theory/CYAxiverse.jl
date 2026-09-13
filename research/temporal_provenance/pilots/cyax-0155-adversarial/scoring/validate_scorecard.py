#!/usr/bin/env python3
"""Validate a CYAX-0155 adversarial scorecard against the frozen schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "scorecard_schema.json"
EXPECTED_KEYS = [f"K{i}" for i in range(1, 13)]
CONFLICT_CRITICAL = {"K6", "K7", "K8", "K9", "K11"}


def validate_scorecard(scorecard: dict) -> list[str]:
    errors: list[str] = []
    schema = json.loads(SCHEMA_PATH.read_text("utf-8"))
    required = schema.get("required", [])
    for field in required:
        if field not in scorecard:
            errors.append(f"missing required field: {field}")
    if "scores" in scorecard:
        scores = scorecard["scores"]
        for key in EXPECTED_KEYS:
            if key not in scores:
                errors.append(f"missing score: {key}")
            elif scores[key] not in (0, 1):
                errors.append(f"invalid score for {key}: {scores[key]}")
        total = sum(scores.get(k, 0) for k in EXPECTED_KEYS)
        cc = sum(scores.get(k, 0) for k in CONFLICT_CRITICAL)
        if scorecard.get("auto_failure"):
            if scorecard.get("total") != 0:
                errors.append(f"auto_failure is true but total is {scorecard.get('total')}, expected 0")
        else:
            if scorecard.get("total") != total:
                errors.append(f"total mismatch: recorded {scorecard.get('total')}, computed {total}")
        if scorecard.get("conflict_critical") != cc:
            errors.append(f"conflict_critical mismatch: recorded {scorecard.get('conflict_critical')}, computed {cc}")
    if "condition" in scorecard and scorecard["condition"] not in ("A", "B"):
        errors.append(f"invalid condition: {scorecard['condition']}")
    if "auto_failure_reasons" in scorecard:
        if scorecard.get("auto_failure") and not scorecard["auto_failure_reasons"]:
            errors.append("auto_failure is true but no reasons given")
        if not scorecard.get("auto_failure") and scorecard["auto_failure_reasons"]:
            errors.append("auto_failure is false but reasons are present")
    for field in ("unsupported_assertions", "authority_errors", "temporal_errors",
                  "supersession_errors", "unjustified_inferences",
                  "correct_abstentions", "incorrect_abstentions", "input_words",
                  "source_reopenings"):
        if field in scorecard and (not isinstance(scorecard[field], int) or scorecard[field] < 0):
            errors.append(f"invalid {field}: {scorecard[field]}")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <scorecard.json>")
        return 1
    path = Path(sys.argv[1])
    scorecard = json.loads(path.read_text("utf-8"))
    errors = validate_scorecard(scorecard)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: {path.name} ({scorecard.get('run_id', '?')}, condition {scorecard.get('condition', '?')}, total {scorecard.get('total', '?')}/12, cc {scorecard.get('conflict_critical', '?')}/5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
