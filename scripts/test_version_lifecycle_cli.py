"""Read-only CLI contract checks for immutable lifecycle snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "version_lifecycle_cli.py"


class CliTests(unittest.TestCase):
    def test_invalid_repository_keeps_structured_json_contract(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI), "snapshot", "--repo", str(ROOT / "missing")],
            cwd=ROOT, check=False, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertIn("reason_code", payload)

    def test_source_is_not_an_event_ledger(self) -> None:
        text = (ROOT / "scripts" / "version_lifecycle_cli.py").read_text(encoding="utf-8")
        self.assertNotIn("ReleaseEventWriter", text)
        self.assertNotIn('"events"', text)
        self.assertIn("lifecycle_ref_snapshot", text)
        self.assertFalse((ROOT / "scripts/version_lifecycle/events.py").exists())
        self.assertFalse((ROOT / "scripts/version_lifecycle/writer.py").exists())

    def test_removed_events_command_is_rejected(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI), "events"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["reason_code"], "CLI_ARGUMENT_INVALID")

    def test_lifecycle_command_preserves_structured_blocked_contract(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI), "lifecycle", "--repo", str(ROOT / "missing")],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
