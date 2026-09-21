"""Git fixture checks for lifecycle ref durability and closure identity."""

from __future__ import annotations

import subprocess
from contextlib import contextmanager, nullcontext
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.git_refs import (  # noqa: E402
    GitIdentityError,
    GitRepository,
    ProtectionEvidence,
    StaticMutationExclusion,
    require_candidate_ref,
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
        self.repository = GitRepository(
            self.work,
            exclusion_checker=lambda: True,
            exclusion_lease=lambda: nullcontext(True),
        )
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

    def test_create_only_lease_rejects_concurrent_fast_forward(self) -> None:
        ref = "refs/heads/maintenance/0.3"
        branch_protection = ProtectionEvidence(
            rule_id="fixture-maintenance-rule",
            pattern="refs/heads/maintenance/*",
            snapshot_sha256="0" * 64,
            retrieved_at_utc="2026-09-20T00:00:00Z",
            creation_guarded=True,
            update_guarded=True,
            deletion_guarded=True,
        )
        self._cmd("git", "push", "origin", f"{self.commit}:{ref}", cwd=self.work)
        (self.work / "Project.toml").write_text('name = "Fixture"\nversion = "0.3.1"\n')
        self._cmd("git", "commit", "-am", "later", cwd=self.work)
        later = self._cmd("git", "rev-parse", "HEAD", cwd=self.work).strip()
        remote_ref = self.repository.remote_ref
        calls = 0

        def stale_absence(observed_ref: str) -> str | None:
            nonlocal calls
            calls += 1
            return None if calls == 1 else remote_ref(observed_ref)

        self.repository.remote_ref = stale_absence
        with self.assertRaisesRegex(GitIdentityError, "REF_ALREADY_EXISTS"):
            self.repository.push_create_only(ref, later, branch_protection)
        self.assertEqual(remote_ref(ref), self.commit)

    def test_static_mutation_boundary_blocks_racing_iteration_and_public_tag(self) -> None:
        iteration_tag = self.repository.make_annotated_iteration_tag(
            version="0.3.0",
            commit=self.commit,
            closure_timestamp_utc="2026-09-20T12:34:56Z",
            tagger_name="Fixture",
            tagger_email="fixture@example.invalid",
        )
        public_protection = ProtectionEvidence(
            rule_id="fixture-canonical-rule",
            pattern="refs/tags/v0.*",
            snapshot_sha256="0" * 64,
            retrieved_at_utc="2026-09-20T00:00:00Z",
            creation_guarded=True,
            update_guarded=True,
            deletion_guarded=True,
        )
        held = self.repository.acquire_static_mutation()
        contender = GitRepository(
            self.work,
            exclusion_checker=lambda: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        try:
            for ref, object_id, protection in (
                ("refs/tags/iterations/0.3.0", iteration_tag, self.protection),
                ("refs/tags/v0.3.0", self.commit, public_protection),
            ):
                with self.subTest(ref=ref):
                    errors: list[BaseException] = []

                    def race() -> None:
                        try:
                            contender.push_create_only(ref, object_id, protection)
                        except BaseException as exc:  # report worker failure below
                            errors.append(exc)

                    thread = threading.Thread(target=race)
                    thread.start()
                    thread.join()
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], GitIdentityError)
                    self.assertRegex(
                        str(errors[0]), "STATIC_MUTATION_EXCLUSION_UNAVAILABLE"
                    )
        finally:
            held.release()

        self.repository.push_create_only(
            "refs/tags/iterations/0.3.0", iteration_tag, self.protection
        )
        self.repository.push_create_only(
            "refs/tags/v0.3.0", self.commit, public_protection
        )

    def test_remote_static_mutation_requires_live_exclusion_proof(self) -> None:
        tag = self.repository.make_annotated_iteration_tag(
            version="0.3.0",
            commit=self.commit,
            closure_timestamp_utc="2026-09-20T12:34:56Z",
            tagger_name="Fixture",
            tagger_email="fixture@example.invalid",
        )
        ref = "refs/tags/iterations/0.3.0"

        def unavailable() -> bool:
            raise OSError("authority unavailable")

        for label, checker in (
            ("missing", None),
            ("false", lambda: False),
            ("exception", unavailable),
        ):
            with self.subTest(checker=label):
                repository = GitRepository(
                    self.work,
                    exclusion_checker=checker,
                    exclusion_lease=lambda: nullcontext(True),
                )
                with self.assertRaisesRegex(GitIdentityError, "EXCLUSION_UNAVAILABLE"):
                    repository.push_create_only(ref, tag, self.protection)
                self.assertIsNone(self.repository.remote_ref(ref))

    def test_remote_static_mutation_holds_external_lease_through_push(self) -> None:
        held = False
        push_observed = False

        @contextmanager
        def external_lease():
            nonlocal held
            held = True
            try:
                yield True
            finally:
                held = False

        repository = GitRepository(
            self.work,
            exclusion_checker=lambda: held,
            exclusion_lease=external_lease,
        )
        original_git = repository.git

        def observed_git(*args, **kwargs):
            nonlocal push_observed
            if args and args[0] == "push":
                self.assertTrue(held)
                push_observed = True
            return original_git(*args, **kwargs)

        repository.git = observed_git
        repository.push_create_only(
            "refs/tags/iterations/0.3.0", self.commit, self.protection
        )
        self.assertTrue(push_observed)
        self.assertFalse(held)

    def test_remote_static_mutation_rejects_missing_or_failed_external_lease(self) -> None:
        for label, factory in (
            ("missing", None),
            ("false", lambda: nullcontext(False)),
            ("exception", lambda: (_ for _ in ()).throw(OSError("lease unavailable"))),
        ):
            with self.subTest(label=label):
                repository = GitRepository(
                    self.work,
                    exclusion_checker=lambda: True,
                    exclusion_lease=factory,
                )
                with self.assertRaisesRegex(GitIdentityError, "EXCLUSION_UNAVAILABLE"):
                    repository.push_create_only(
                        "refs/tags/iterations/0.3.0", self.commit, self.protection
                    )
                self.assertIsNone(self.repository.remote_ref("refs/tags/iterations/0.3.0"))

    def test_remote_static_mutation_lease_release_failure_is_uncertain(self) -> None:
        @contextmanager
        def failed_release():
            yield True
            raise OSError("lease release failed after remote push")

        repository = GitRepository(
            self.work,
            exclusion_checker=lambda: True,
            exclusion_lease=failed_release,
        )
        ref = "refs/tags/iterations/0.3.0"
        with self.assertRaisesRegex(
            GitIdentityError, "STATIC_MUTATION_OUTCOME_UNCERTAIN"
        ):
            repository.push_create_only(ref, self.commit, self.protection)
        self.assertEqual(self.repository.remote_ref(ref), self.commit)

    def test_static_mutation_boundary_requires_supported_lock(self) -> None:
        exclusion = StaticMutationExclusion(self.work / "missing" / "lock")
        with self.assertRaisesRegex(
            GitIdentityError, "STATIC_MUTATION_EXCLUSION_UNAVAILABLE"
        ):
            exclusion.acquire()

    def test_public_tag_requires_canonical_ruleset_and_excludes_legacy(self) -> None:
        public_ref = "refs/tags/v0.3.0"
        self._cmd(
            "git", "push", "origin", f"{self.commit}:refs/tags/v-0.1", cwd=self.work
        )
        good = ProtectionEvidence(
            rule_id="fixture-canonical-rule",
            pattern="refs/tags/v0.*",
            snapshot_sha256="0" * 64,
            retrieved_at_utc="2026-09-20T00:00:00Z",
            creation_guarded=True,
            update_guarded=True,
            deletion_guarded=True,
        )
        self.repository.push_create_only(public_ref, self.commit, good)
        self.assertEqual(self.repository.remote_ref(public_ref), self.commit)
        self.assertEqual(self.repository.remote_ref("refs/tags/v-0.1"), self.commit)

        for pattern, creation, update, deletion in (
            ("refs/tags/v*", True, True, True),
            ("refs/tags/v0.*", False, True, True),
            ("refs/tags/v0.*", True, False, True),
            ("refs/tags/v0.*", True, True, False),
        ):
            with self.subTest(pattern=pattern, creation=creation, update=update, deletion=deletion):
                weak = ProtectionEvidence(
                    rule_id="fixture-weak-rule",
                    pattern=pattern,
                    snapshot_sha256="0" * 64,
                    retrieved_at_utc="2026-09-20T00:00:00Z",
                    creation_guarded=creation,
                    update_guarded=update,
                    deletion_guarded=deletion,
                )
                with self.assertRaisesRegex(GitIdentityError, "PUBLIC_TAG_RULESET_UNAVAILABLE"):
                    weak.require_public_tag(public_ref)

        with self.assertRaisesRegex(GitIdentityError, "PUBLIC_TAG_RULESET_UNAVAILABLE"):
            good.require_public_tag("refs/tags/v-0.1")
        with self.assertRaisesRegex(GitIdentityError, "PUBLIC_TAG_RULESET_UNAVAILABLE"):
            good.require_public_tag("refs/tags/v00.3.0")
        with self.assertRaisesRegex(GitIdentityError, "PUBLIC_TAG_RULESET_UNAVAILABLE"):
            self.repository.push_create_only("refs/tags/v-0.1", self.commit, good)
        self.assertEqual(self.repository.remote_ref("refs/tags/v-0.1"), self.commit)

    def test_invalid_closure_time_is_not_inferred(self) -> None:
        with self.assertRaisesRegex(GitIdentityError, "invalid closure UTC timestamp"):
            self.repository.make_annotated_iteration_tag(
                version="0.3.0", commit=self.commit,
                closure_timestamp_utc="2026-02-30T12:34:56Z",
                tagger_name="Fixture", tagger_email="fixture@example.invalid",
            )

    def test_candidate_ref_validation_rejects_git_invalid_names(self) -> None:
        self.assertEqual(
            require_candidate_ref("refs/heads/candidates/0.3.0"),
            "refs/heads/candidates/0.3.0",
        )
        for ref in (
            "refs/heads/candidates/foo/../bar",
            "refs/heads/candidates/foo//bar",
            "refs/heads/candidates/foo/",
            "refs/heads/candidates/",
            "refs/heads/candidates/.hidden",
            "refs/heads/candidates/name.lock",
            "refs/heads/candidates/name.",
            "refs/heads/maintenance/0.3",
        ):
            with self.subTest(ref=ref):
                with self.assertRaises(GitIdentityError):
                    require_candidate_ref(ref)

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
