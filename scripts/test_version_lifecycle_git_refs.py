"""Git fixture checks for lifecycle ref durability and closure identity."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.git_refs import (  # noqa: E402
    GitIdentityError,
    GitRepository,
    ProtectionEvidence,
)


class GitRefFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.remote = root / "remote.git"
        self.work = root / "work"
        self._cmd("git", "init", "--bare", str(self.remote), cwd=root)
        self._cmd("git", "init", str(self.work), cwd=root)
        self._cmd("git", "config", "user.name", "Fixture", cwd=self.work)
        self._cmd("git", "config", "user.email", "fixture@example.invalid", cwd=self.work)
        self._cmd("git", "remote", "add", "origin", str(self.remote), cwd=self.work)
        (self.work / "Project.toml").write_text('name = "Fixture"\nversion = "0.3.0"\n')
        self._cmd("git", "add", "Project.toml", cwd=self.work)
        self._cmd("git", "commit", "-m", "closure", cwd=self.work)
        self.commit = self._cmd("git", "rev-parse", "HEAD", cwd=self.work).strip()
        self.repository = GitRepository(self.work)
        self.protection = ProtectionEvidence(
            rule_id="fixture-rule",
            pattern="refs/tags/iterations/*",
            snapshot_sha256="0" * 64,
            retrieved_at_utc="2026-09-20T00:00:00Z",
            creation_guarded=True,
            update_guarded=True,
            deletion_guarded=True,
        )

    @staticmethod
    def _cmd(*command: str, cwd: Path) -> str:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout

    def test_protected_iteration_anchor_is_exact_and_remote_durable(self) -> None:
        tag = self.repository.make_annotated_iteration_tag(
            version="0.3.0",
            commit=self.commit,
            closure_timestamp_utc="2026-09-20T12:34:56Z",
            tagger_name="Fixture",
            tagger_email="fixture@example.invalid",
        )
        ref = "refs/tags/iterations/0.3.0"
        self.repository.push_create_only(ref, tag, self.protection)
        self.repository.push_create_only(ref, tag, self.protection)
        self.assertEqual(self.repository.remote_ref(ref), tag)
        self.assertEqual(self.repository.tree(self.commit), self._cmd(
            "git", "rev-parse", "HEAD^{tree}", cwd=self.work
        ).strip())
        self.assertEqual(self.repository.project_version(self.commit), "0.3.0")
        tag_text = self.repository.git("cat-file", "-p", tag).decode()
        self.assertIn("closure_timestamp_utc=2026-09-20T12:34:56Z", tag_text)

    def test_unsafe_or_conflicting_create_blocks(self) -> None:
        tag = self.repository.make_annotated_iteration_tag(
            version="0.3.0",
            commit=self.commit,
            closure_timestamp_utc="2026-09-20T12:34:56Z",
            tagger_name="Fixture",
            tagger_email="fixture@example.invalid",
        )
        ref = "refs/tags/iterations/0.3.0"
        weak = ProtectionEvidence(
            rule_id="fixture-rule", pattern="refs/tags/iterations/*",
            snapshot_sha256="0" * 64,
            retrieved_at_utc="2026-09-20T00:00:00Z",
            creation_guarded=True, update_guarded=False, deletion_guarded=True,
        )
        with self.assertRaisesRegex(GitIdentityError, "PROTECTION_EVIDENCE_UNAVAILABLE"):
            self.repository.push_create_only(ref, tag, weak)
        self.repository.push_create_only(ref, tag, self.protection)
        different_tag = self.repository.make_annotated_iteration_tag(
            version="0.3.0",
            commit=self.commit,
            closure_timestamp_utc="2026-09-20T12:34:57Z",
            tagger_name="Fixture",
            tagger_email="fixture@example.invalid",
        )
        with self.assertRaisesRegex(GitIdentityError, "REF_ALREADY_EXISTS"):
            self.repository.push_create_only(ref, different_tag, self.protection)

    def test_invalid_closure_time_is_not_inferred(self) -> None:
        with self.assertRaisesRegex(GitIdentityError, "invalid closure UTC timestamp"):
            self.repository.make_annotated_iteration_tag(
                version="0.3.0", commit=self.commit,
                closure_timestamp_utc="2026-02-30T12:34:56Z",
                tagger_name="Fixture", tagger_email="fixture@example.invalid",
            )

    def test_principal_interval_enumerates_drift_before_promotion(self) -> None:
        certified_tree = self.repository.tree(self.commit)
        no_drift = self.repository.principal_interval(self.commit, self.commit, certified_tree)
        self.assertEqual((no_drift["disposition"], no_drift["intervening_commits"]),
                         ("no_drift", []))

        self._cmd("git", "commit", "--allow-empty", "-m", "tree-neutral", cwd=self.work)
        neutral = self._cmd("git", "rev-parse", "HEAD", cwd=self.work).strip()
        interval = self.repository.principal_interval(self.commit, neutral, certified_tree)
        self.assertEqual(interval["disposition"], "tree_neutral_included")
        self.assertEqual([item["sha"] for item in interval["intervening_commits"]], [neutral])
        self.assertEqual(interval["freeze_main_tree"], certified_tree)

        (self.work / "Project.toml").write_text('name = "Fixture"\nversion = "0.3.1"\n')
        self._cmd("git", "add", "Project.toml", cwd=self.work)
        self._cmd("git", "commit", "-m", "content-drift", cwd=self.work)
        changed = self._cmd("git", "rev-parse", "HEAD", cwd=self.work).strip()
        blocked = self.repository.principal_interval(self.commit, changed, certified_tree)
        self.assertEqual(blocked["disposition"], "content_drift_blocked")
        self.assertEqual([item["sha"] for item in blocked["intervening_commits"]],
                         [neutral, changed])
        with self.assertRaisesRegex(GitIdentityError, "CANDIDATE_MAIN_NOT_ANCESTOR"):
            self.repository.principal_interval(changed, self.commit, certified_tree)


if __name__ == "__main__":
    unittest.main()
