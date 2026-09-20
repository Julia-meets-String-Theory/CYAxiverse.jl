#!/usr/bin/env python3
"""Isolated Git fixture tests for the read-only Gate A lifecycle CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from version_lifecycle.writer import ReleaseEventWriter  # noqa: E402


CLI = SCRIPT_ROOT / "version_lifecycle_cli.py"


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class GateALifecycleCliFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cyax-gate-a-cli-")
        root = Path(self.temporary.name)
        self.remote = root / "origin.git"
        self.repo = root / "fixture"
        subprocess.run(["git", "init", "--bare", "-q", str(self.remote)], check=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.repo)], check=True)
        _git(self.repo, "config", "user.name", "Gate A Fixture")
        _git(self.repo, "config", "user.email", "gate-a@example.invalid")
        _git(self.repo, "remote", "add", "origin", str(self.remote))
        (self.repo / "Project.toml").write_text(
            'name = "Fixture"\nversion = "0.1.0"\n', encoding="utf-8"
        )
        (self.repo / "iterations.toml").write_text(
            "schema_version = 1\n"
            'target_iteration = "cli-fixture"\n'
            "iterations = []\nprospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _git(self.repo, "add", "Project.toml", "iterations.toml")
        _git(self.repo, "commit", "-q", "-m", "fixture main")
        (self.repo / "Project.toml").write_text(
            'name = "Fixture"\nversion = "0.2.0"\n', encoding="utf-8"
        )
        _git(self.repo, "add", "Project.toml")
        _git(self.repo, "commit", "-q", "-m", "fixture vmm")
        _git(self.repo, "branch", "vmm")
        _git(self.repo, "push", "-q", "origin", "main", "vmm")
        self.writer = ReleaseEventWriter(self.repo)
        self.writer.bootstrap()
        _git(
            self.repo,
            "push",
            "-q",
            "origin",
            "refs/heads/release-events:refs/heads/release-events",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_cli(self, *args: str) -> tuple[int, dict]:
        result = subprocess.run(
            [sys.executable, str(CLI), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.stderr, "", result.stderr)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            self.fail(f"CLI did not emit JSON: {result.stdout!r}: {error}")
        return result.returncode, payload

    def test_readiness_reads_snapshot_and_event_head_without_mutating_refs(self) -> None:
        before = _git(self.repo, "show-ref")
        fetch_head = self.repo / ".git" / "FETCH_HEAD"
        before_fetch_head = fetch_head.read_bytes() if fetch_head.exists() else None
        code, payload = self._run_cli(
            "readiness",
            "--repo",
            str(self.repo),
            "--dry-run",
            "--principal-closed",
            "0.2.0",
        )
        after = _git(self.repo, "show-ref")

        self.assertEqual(code, 0)
        self.assertEqual(after, before)
        after_fetch_head = fetch_head.read_bytes() if fetch_head.exists() else None
        self.assertEqual(after_fetch_head, before_fetch_head)
        self.assertNotIn(str(self.repo.resolve()), json.dumps(payload))
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["status"], "READY")
        self.assertEqual(payload["static_snapshot"]["status"], "READY")
        self.assertEqual(payload["event_head"]["status"], "READY")
        self.assertEqual(payload["event_head"]["event_count"], 0)
        view = payload["allocation_view"]
        self.assertEqual(view["status"], "READY")
        self.assertEqual(view["allocation"]["status"], "AVAILABLE")
        self.assertEqual(view["allocation"]["version"], "0.2.1-DEV")

    def test_missing_event_branch_is_a_machine_readable_block(self) -> None:
        # The local cache still exists, but the remote authority is absent.
        # The CLI must not treat local state as a substitute or bootstrap a
        # remote branch as a side effect of a readiness query.
        _git(self.remote, "update-ref", "-d", "refs/heads/release-events")
        before = _git(self.repo, "show-ref")
        code, payload = self._run_cli(
            "events",
            "--repo",
            str(self.repo),
            "--dry-run",
        )
        self.assertEqual(code, 2)
        self.assertEqual(_git(self.repo, "show-ref"), before)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertEqual(payload["reason_code"], "EVENT_AUTHORITY_UNAVAILABLE")

    def test_remote_advance_is_used_when_local_cache_is_stale(self) -> None:
        initial_head = self.writer.current_head()
        event = {
            "schema_version": 1,
            "event_id": "EVT-000000000001",
            "event_type": "development_reservation_prepared",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "cli-remote-advance",
            "static_iteration_snapshot": "a" * 64,
            "expected_event_head": initial_head,
            "owner_line": "principal",
            "final_version": "0.2.1",
            "intended_dev_version": "0.2.1-DEV",
            "expected_line_head": "c" * 40,
            "reservation_id": "cli-reservation-1",
        }
        append_writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _: True,
        )
        appended = append_writer.append(event, expected_head=initial_head)
        self.assertEqual(appended.status, "APPENDED")
        assert appended.head is not None
        _git(
            self.repo,
            "push",
            "-q",
            "origin",
            f"{appended.head}:refs/heads/release-events",
        )
        # Leave the checkout cache behind the advertised remote head.
        _git(self.repo, "update-ref", "refs/heads/release-events", initial_head)

        code, payload = self._run_cli(
            "events",
            "--repo",
            str(self.repo),
            "--dry-run",
        )
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "READY")
        event_head = payload["event_head"]
        self.assertEqual(event_head["authority"], "remote")
        self.assertEqual(event_head["head_commit"], appended.head)
        self.assertEqual(event_head["event_count"], 1)
        self.assertEqual(event_head["local_head_commit"], initial_head)
        self.assertTrue(event_head["local_head_stale"])

    def test_allocation_input_failure_keeps_combined_result_blocked(self) -> None:
        code, payload = self._run_cli(
            "allocation",
            "--repo",
            str(self.repo),
            "--dry-run",
            "--maintenance-closed",
            "0.2.0",
        )
        self.assertEqual(code, 2)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertEqual(payload["reason_code"], "MAINTENANCE_LINE_REQUIRED")
        self.assertEqual(payload["allocation_view"]["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
