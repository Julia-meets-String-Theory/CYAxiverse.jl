"""Release-boundary checks against real fixture commits."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECKER = Path(__file__).with_name("check_version_bump.py")


class VersionBumpFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self._git("init")
        self._git("config", "user.name", "Fixture")
        self._git("config", "user.email", "fixture@example.invalid")
        (self.root / "src").mkdir()
        (self.root / "src" / "Fixture.jl").write_text("module Fixture\nend\n")
        (self.root / "Project.toml").write_text('name = "Fixture"\nversion = "0.1.0"\n')
        self._git("add", ".")
        self._git("commit", "-m", "base")
        self.base = self._git("rev-parse", "HEAD").strip()

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.root, capture_output=True, text=True
        )
        if result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout

    def _head(self, version: str) -> str:
        (self.root / "src" / "Fixture.jl").write_text(
            "module Fixture\nconst changed = true\nend\n"
        )
        (self.root / "Project.toml").write_text(
            f'name = "Fixture"\nversion = "{version}"\n'
        )
        self._git("add", ".")
        self._git("commit", "-m", "head")
        return self._git("rev-parse", "HEAD").strip()

    def _check(self, head: str, *, require_bump: bool) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable, str(CHECKER), "--base", self.base, "--head", head
        ]
        if require_bump:
            command.append("--require-bump")
        return subprocess.run(command, cwd=self.root, capture_output=True, text=True)

    def test_final_release_bump_passes(self) -> None:
        head = self._head("0.2.0")
        result = self._check(head, require_bump=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0.1.0 -> 0.2.0", result.stdout)

    def test_dev_identity_is_valid_for_development_but_not_main(self) -> None:
        head = self._head("0.2.0-DEV")
        self.assertEqual(self._check(head, require_bump=False).returncode, 0)
        result = self._check(head, require_bump=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("requires final package versions", result.stderr)

    def test_leading_zero_identity_fails_closed(self) -> None:
        head = self._head("00.2.0")
        result = self._check(head, require_bump=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid package version", result.stderr)


if __name__ == "__main__":
    unittest.main()
