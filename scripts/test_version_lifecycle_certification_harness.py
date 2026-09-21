#!/usr/bin/env python3
"""Focused fixture tests for the Gate A R-023 certification harness."""

from __future__ import annotations

import hashlib
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from certify_exact_tree_checkout import (  # noqa: E402
    BLOCKED,
    INVALID,
    PASS,
    certify_exact_tree_checkout,
)


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


class CertificationHarnessFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cyax-certification-harness-")
        self.repository = Path(self.temporary.name) / "package"
        subprocess.run(
            ["git", "init", "--quiet", "--initial-branch=main", str(self.repository)],
            check=True,
        )
        _git(self.repository, "config", "user.name", "Certification Fixture")
        _git(self.repository, "config", "user.email", "certification@example.invalid")
        (self.repository / "base.txt").write_text("base\n", encoding="utf-8")
        _git(self.repository, "add", "base.txt")
        _git(self.repository, "commit", "--quiet", "-m", "base")
        (self.repository / "tracked.txt").write_text("stable\n", encoding="utf-8")
        _git(self.repository, "add", "tracked.txt")
        _git(self.repository, "commit", "--quiet", "-m", "candidate")
        self.candidate_commit = _git(self.repository, "rev-parse", "HEAD")
        _git(self.repository, "commit", "--quiet", "--allow-empty", "-m", "closed iteration")
        self.closed_iteration_commit = _git(self.repository, "rev-parse", "HEAD")
        _git(self.repository, "commit", "--quiet", "--allow-empty", "-m", "release")
        self.release_commit = _git(self.repository, "rev-parse", "HEAD")
        self.harness_revision = "f" * 40
        self.harness_path = SCRIPT_ROOT / "certify_exact_tree_checkout.py"
        self.harness_sha256 = hashlib.sha256(self.harness_path.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _certify(self, command: list[str]) -> dict:
        return certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=self.release_commit,
            test_command=command,
            harness_revision=self.harness_revision,
            harness_path=self.harness_path,
            expected_harness_sha256=self.harness_sha256,
        )

    def test_runs_package_command_in_pinned_checkout_and_proves_all_trees(self) -> None:
        result = self._certify(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; assert Path('tracked.txt').read_text() "
                    "== 'stable\\n'; Path('test-ran').write_text('ok')"
                ),
            ]
        )
        self.assertEqual(result["status"], PASS)
        self.assertEqual(result["test_exit_code"], 0)
        self.assertEqual(result["candidate_tree"], result["closed_iteration_tree"])
        self.assertEqual(result["candidate_tree"], result["release_tree"])
        self.assertEqual(result["tracked_tree_before"], result["tracked_tree_after"])
        self.assertEqual(result["head_before"], self.candidate_commit)
        self.assertEqual(result["head_after"], self.candidate_commit)
        self.assertEqual(result["harness_revision"], self.harness_revision)
        self.assertEqual(len(result["harness_sha256"]), 64)

    def test_rejects_a_test_that_changes_a_tracked_file(self) -> None:
        result = self._certify(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('tracked.txt').write_text('changed\\n')",
            ]
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "TRACKED_TREE_CHANGED")
        self.assertTrue(result["tracked_files_changed"])
        self.assertEqual(len(result["tracked_status_sha256"]), 64)
        self.assertEqual(result["test_exit_code"], 0)

    def test_rejects_a_test_that_moves_head(self) -> None:
        result = self._certify(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess; subprocess.run("
                    "['git', 'checkout', '--detach', 'HEAD^'], check=True)"
                ),
            ]
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "CHECKOUT_HEAD_CHANGED")

    def test_rejects_candidate_closed_or_release_tree_mismatch_before_running_tests(self) -> None:
        (self.repository / "tracked.txt").write_text("release drift\n", encoding="utf-8")
        _git(self.repository, "add", "tracked.txt")
        _git(self.repository, "commit", "--quiet", "-m", "different release")
        mismatching_release = _git(self.repository, "rev-parse", "HEAD")
        result = certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=mismatching_release,
            test_command=[sys.executable, "-c", "raise SystemExit('must not run')"],
            harness_revision=self.harness_revision,
            harness_path=self.harness_path,
            expected_harness_sha256=self.harness_sha256,
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "CERTIFIED_TREE_MISMATCH")
        self.assertNotIn("test_exit_code", result)

    def test_reports_package_test_failure_and_timeout_as_fail_closed(self) -> None:
        failed = self._certify([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(failed["status"], INVALID)
        self.assertEqual(failed["reason_code"], "PACKAGE_TEST_FAILED")
        self.assertEqual(failed["test_exit_code"], 7)

        timeout = certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=self.release_commit,
            test_command=[sys.executable, "-c", "import time; time.sleep(1)"],
            harness_revision=self.harness_revision,
            harness_path=self.harness_path,
            expected_harness_sha256=self.harness_sha256,
            timeout_seconds=0.01,
        )
        self.assertEqual(timeout["status"], BLOCKED)
        self.assertEqual(timeout["reason_code"], "PACKAGE_TEST_TIMEOUT")

    def test_requires_an_independent_harness_revision_and_honors_digest_pin(self) -> None:
        missing = certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=self.release_commit,
            test_command=[sys.executable, "-c", "pass"],
            harness_revision="un-pinned",
            harness_path=self.harness_path,
            expected_harness_sha256=self.harness_sha256,
        )
        self.assertEqual(missing["status"], BLOCKED)
        self.assertEqual(missing["reason_code"], "HARNESS_REVISION_UNPINNED")

        missing_digest = certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=self.release_commit,
            test_command=[sys.executable, "-c", "pass"],
            harness_revision=self.harness_revision,
            harness_path=self.harness_path,
        )
        self.assertEqual(missing_digest["status"], BLOCKED)
        self.assertEqual(missing_digest["reason_code"], "HARNESS_DIGEST_UNPINNED")

        missing_path = certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=self.release_commit,
            test_command=[sys.executable, "-c", "pass"],
            harness_revision=self.harness_revision,
            expected_harness_sha256=self.harness_sha256,
        )
        self.assertEqual(missing_path["status"], BLOCKED)
        self.assertEqual(missing_path["reason_code"], "HARNESS_PATH_UNPINNED")

        expected = "0" * 64
        mismatch = certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=self.release_commit,
            test_command=[sys.executable, "-c", "pass"],
            harness_revision=self.harness_revision,
            harness_path=self.harness_path,
            expected_harness_sha256=expected,
        )
        self.assertEqual(mismatch["status"], BLOCKED)
        self.assertEqual(mismatch["reason_code"], "HARNESS_IDENTITY_MISMATCH")

    def test_cli_accepts_an_argv_command_and_emits_public_json(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_ROOT / "certify_exact_tree_checkout.py"),
                "--repo",
                str(self.repository),
                "--candidate-commit",
                self.candidate_commit,
                "--closed-iteration-commit",
                self.closed_iteration_commit,
                "--release-commit",
                self.release_commit,
                "--harness-revision",
                self.harness_revision,
                "--harness-path",
                str(self.harness_path),
                "--expected-harness-sha256",
                self.harness_sha256,
                "--test-command",
                sys.executable,
                "-c",
                "pass",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], PASS)
        self.assertNotIn(str(self.repository.resolve()), result.stdout)

    def test_cli_blocks_missing_or_mismatched_harness_digest(self) -> None:
        prefix = [
            sys.executable,
            str(SCRIPT_ROOT / "certify_exact_tree_checkout.py"),
            "--repo",
            str(self.repository),
            "--candidate-commit",
            self.candidate_commit,
            "--closed-iteration-commit",
            self.closed_iteration_commit,
            "--release-commit",
            self.release_commit,
            "--harness-revision",
            self.harness_revision,
            "--harness-path",
            str(self.harness_path),
        ]
        for expected_digest, reason_code in (
            (None, "HARNESS_DIGEST_UNPINNED"),
            ("0" * 64, "HARNESS_IDENTITY_MISMATCH"),
        ):
            command = [*prefix]
            if expected_digest is not None:
                command.extend(["--expected-harness-sha256", expected_digest])
            command.extend(["--test-command", sys.executable, "-c", "pass"])
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stderr, "")
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], BLOCKED)
            self.assertEqual(payload["reason_code"], reason_code)


if __name__ == "__main__":
    unittest.main()
