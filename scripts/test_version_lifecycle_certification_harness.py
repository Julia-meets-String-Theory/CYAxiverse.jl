#!/usr/bin/env python3
"""Focused fixture tests for the Gate A R-023 certification harness."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from certify_exact_tree_checkout import (  # noqa: E402
    APPROVED_PACKAGE_TEST_COMMAND,
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
        self.fake_bin = Path(self.temporary.name) / "bin"
        self.fake_bin.mkdir()
        self.fake_julia = self.fake_bin / "julia"
        self.fake_julia.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "import subprocess\n"
            "import sys\n"
            "import time\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print(os.environ.get('CYAX_CERTIFICATION_VERSION_BANNER', "
            "'julia version 1.12.0-certification-fixture'))\n"
            "    raise SystemExit(0)\n"
            "if sys.argv[1:] != ['--startup-file=no', '--project=.', '-e', 'using Pkg; Pkg.test()']:\n"
            "    raise SystemExit(41)\n"
            "action = os.environ.get('CYAX_CERTIFICATION_ACTION', 'pass')\n"
            "if action == 'tracked-change':\n"
            "    Path('tracked.txt').write_text('changed\\n')\n"
            "elif action == 'move-head':\n"
            "    subprocess.run(['git', 'checkout', '--detach', 'HEAD^'], check=True)\n"
            "elif action == 'fail':\n"
            "    raise SystemExit(7)\n"
            "elif action == 'timeout':\n"
            "    time.sleep(1)\n"
            "elif action == 'must-not-run':\n"
            "    raise SystemExit(41)\n",
            encoding="utf-8",
        )
        self.fake_julia.chmod(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
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

    def _child_environment(self, action: str = "pass") -> dict[str, str]:
        environment = dict(os.environ)
        environment["PATH"] = f"{self.fake_bin}{os.pathsep}{environment.get('PATH', '')}"
        environment["CYAX_CERTIFICATION_ACTION"] = action
        return environment

    def _certify(
        self,
        command: list[str] | None = None,
        *,
        action: str = "pass",
        timeout_seconds: float = 900.0,
    ) -> dict:
        original_path = os.environ.get("PATH")
        original_action = os.environ.get("CYAX_CERTIFICATION_ACTION")
        os.environ.update(self._child_environment(action))
        try:
            return certify_exact_tree_checkout(
                repository=self.repository,
                candidate_commit=self.candidate_commit,
                closed_iteration_commit=self.closed_iteration_commit,
                release_commit=self.release_commit,
                test_command=list(APPROVED_PACKAGE_TEST_COMMAND) if command is None else command,
                harness_revision=self.harness_revision,
                harness_path=self.harness_path,
                expected_harness_sha256=self.harness_sha256,
                timeout_seconds=timeout_seconds,
            )
        finally:
            if original_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = original_path
            if original_action is None:
                os.environ.pop("CYAX_CERTIFICATION_ACTION", None)
            else:
                os.environ["CYAX_CERTIFICATION_ACTION"] = original_action

    def test_runs_package_command_in_pinned_checkout_and_proves_all_trees(self) -> None:
        result = self._certify()
        self.assertEqual(result["status"], PASS)
        self.assertEqual(result["test_exit_code"], 0)
        self.assertEqual(result["candidate_tree"], result["closed_iteration_tree"])
        self.assertEqual(result["candidate_tree"], result["release_tree"])
        self.assertEqual(result["tracked_tree_before"], result["tracked_tree_after"])
        self.assertEqual(result["head_before"], self.candidate_commit)
        self.assertEqual(result["head_after"], self.candidate_commit)
        self.assertEqual(result["harness_revision"], self.harness_revision)
        self.assertEqual(len(result["harness_sha256"]), 64)
        self.assertEqual(result["test_command_argv"], list(APPROVED_PACKAGE_TEST_COMMAND))
        self.assertEqual(len(result["test_command_sha256"]), 64)
        self.assertEqual(result["test_executable"], "julia")
        self.assertEqual(len(result["test_executable_sha256"]), 64)
        self.assertEqual(len(result["test_environment_sha256"]), 64)
        self.assertEqual(result["runtime_environment"], result["test_environment"])
        self.assertEqual(result["runtime_environment_sha256"], result["test_environment_sha256"])
        self.assertNotIn(str(self.repository), json.dumps(result, sort_keys=True))
        self.assertNotIn(str(self.fake_bin), json.dumps(result, sort_keys=True))

    def test_rejects_a_test_that_changes_a_tracked_file(self) -> None:
        result = self._certify(action="tracked-change")
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "TRACKED_TREE_CHANGED")
        self.assertTrue(result["tracked_files_changed"])
        self.assertEqual(len(result["tracked_status_sha256"]), 64)
        self.assertEqual(result["test_exit_code"], 0)

    def test_unsafe_executable_version_banner_is_published_only_as_a_digest(self) -> None:
        original = os.environ.get("CYAX_CERTIFICATION_VERSION_BANNER")
        os.environ["CYAX_CERTIFICATION_VERSION_BANNER"] = (
            "julia version 1.12.0 /Users/private/toolchain"
        )
        try:
            result = self._certify()
        finally:
            if original is None:
                os.environ.pop("CYAX_CERTIFICATION_VERSION_BANNER", None)
            else:
                os.environ["CYAX_CERTIFICATION_VERSION_BANNER"] = original
        self.assertEqual(result["status"], PASS)
        banner = result["test_environment"]["test_executable_version"]
        self.assertRegex(banner, r"^sha256:[0-9a-f]{64}$")
        self.assertNotIn("/Users/private", json.dumps(result, sort_keys=True))

    def test_rejects_a_test_that_moves_head(self) -> None:
        result = self._certify(action="move-head")
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
            test_command=list(APPROVED_PACKAGE_TEST_COMMAND),
            harness_revision=self.harness_revision,
            harness_path=self.harness_path,
            expected_harness_sha256=self.harness_sha256,
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "CERTIFIED_TREE_MISMATCH")
        self.assertNotIn("test_exit_code", result)

    def test_reports_package_test_failure_and_timeout_as_fail_closed(self) -> None:
        failed = self._certify(action="fail")
        self.assertEqual(failed["status"], INVALID)
        self.assertEqual(failed["reason_code"], "PACKAGE_TEST_FAILED")
        self.assertEqual(failed["test_exit_code"], 7)

        timeout = self._certify(action="timeout", timeout_seconds=0.01)
        self.assertEqual(timeout["status"], BLOCKED)
        self.assertEqual(timeout["reason_code"], "PACKAGE_TEST_TIMEOUT")

    def test_requires_an_independent_harness_revision_and_honors_digest_pin(self) -> None:
        missing = certify_exact_tree_checkout(
            repository=self.repository,
            candidate_commit=self.candidate_commit,
            closed_iteration_commit=self.closed_iteration_commit,
            release_commit=self.release_commit,
            test_command=list(APPROVED_PACKAGE_TEST_COMMAND),
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
            test_command=list(APPROVED_PACKAGE_TEST_COMMAND),
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
            test_command=list(APPROVED_PACKAGE_TEST_COMMAND),
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
            test_command=list(APPROVED_PACKAGE_TEST_COMMAND),
            harness_revision=self.harness_revision,
            harness_path=self.harness_path,
            expected_harness_sha256=expected,
        )
        self.assertEqual(mismatch["status"], BLOCKED)
        self.assertEqual(mismatch["reason_code"], "HARNESS_IDENTITY_MISMATCH")

    def test_rejects_noop_or_unapproved_command_before_running_checkout(self) -> None:
        for command in (
            [sys.executable, "-c", "pass"],
            ["julia", "--startup-file=no", "--project=.", "-e", "pass"],
        ):
            result = self._certify(command)
            self.assertEqual(result["status"], BLOCKED)
            self.assertEqual(result["reason_code"], "PACKAGE_TEST_COMMAND_UNAPPROVED")
            self.assertNotIn("test_exit_code", result)

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
                *APPROVED_PACKAGE_TEST_COMMAND,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=self._child_environment(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], PASS)
        self.assertEqual(payload["test_command_argv"], list(APPROVED_PACKAGE_TEST_COMMAND))
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
            command.extend(["--test-command", *APPROVED_PACKAGE_TEST_COMMAND])
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=self._child_environment(),
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stderr, "")
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], BLOCKED)
            self.assertEqual(payload["reason_code"], reason_code)


if __name__ == "__main__":
    unittest.main()
