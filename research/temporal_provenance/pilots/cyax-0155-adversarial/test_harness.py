#!/usr/bin/env python3
"""Focused offline tests for the CYAX-0155 experimental harness."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_validator():
    path = HERE / "scoring/validate_scorecard.py"
    spec = importlib.util.spec_from_file_location("scorecard_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


VALID = {
    "opaque_id": "S1",
    "scores": {f"K{i}": 1 for i in range(1, 13)},
    "raw_total": 12,
    "conflict_critical": 5,
    "automatic_failure": False,
    "automatic_failure_reasons": [],
    "unsupported_assertions": 0,
    "authority_errors": 0,
    "temporal_errors": 0,
    "supersession_errors": 0,
    "unjustified_inferences": 0,
    "correct_abstentions": 1,
    "incorrect_abstentions": 0,
    "next_action_correct": True,
    "source_reopenings": 0,
}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.validator = load_validator()

    def test_valid_scorecard(self):
        self.validator.validate(VALID.copy())

    def test_automatic_failure_preserves_raw_total(self):
        card = json.loads(json.dumps(VALID))
        card["automatic_failure"] = True
        card["automatic_failure_reasons"] = ["invented owner decision"]
        self.validator.validate(card)
        self.assertEqual(card["raw_total"], 12)

    def test_rejects_condition_label(self):
        card = json.loads(json.dumps(VALID))
        card["condition"] = "A"
        with self.assertRaises(ValueError):
            self.validator.validate(card)

    def test_rejects_non_boolean(self):
        card = json.loads(json.dumps(VALID))
        card["next_action_correct"] = "true"
        with self.assertRaises(ValueError):
            self.validator.validate(card)

    def test_rejects_bad_sum(self):
        card = json.loads(json.dumps(VALID))
        card["raw_total"] = 0
        with self.assertRaises(ValueError):
            self.validator.validate(card)

    def test_offline_builders(self):
        for command in (
            ["python3", str(HERE / "validate_snapshot.py")],
            ["python3", str(HERE / "context_builder.py"), "validate"],
            ["python3", str(HERE / "context_builder.py"), "determinism"],
        ):
            subprocess.run(command, cwd=HERE, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
