#!/usr/bin/env python3
"""Focused regression tests for the Gate A static/version library."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from version_lifecycle import (  # noqa: E402
    BlockedResult,
    MAX_VERSION_COMPONENT,
    Version,
    canonical_json,
    global_allocation_view,
    maintenance_line,
    parse_package_version,
    parse_public_tag,
    recompute_snapshot_digests,
    select_maintenance_version,
    select_principal_sentinel,
    sha256_hex,
    snapshot_is_stale,
    static_snapshot,
    validate_static_snapshot,
)
from version_lifecycle.static import (  # noqa: E402
    StaticValidationError,
    _remote_refs,
    sanitize_source_repository,
)
from version_lifecycle.writer import LedgerHead, ReleaseEventWriter  # noqa: E402
from version_lifecycle.events import (  # noqa: E402
    EventSchemaError,
    canonical_event_bytes,
    validate_event,
)


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _fixture_event_head(
    repo: Path,
    occupied_versions: tuple[str, ...] = (),
    *,
    branch: str = "release-events",
    events: tuple[dict, ...] | None = None,
) -> LedgerHead:
    """Build a test head through the writer's verified Git authority path."""

    if events is None:
        events = tuple(
            {
                "schema_version": 1,
                "event_id": f"EVT-{index:012d}",
                "event_type": "development_reservation_prepared",
                "timestamp_utc": "2026-09-20T12:34:56Z",
                "transaction_id": f"fixture-{index}",
                "static_iteration_snapshot": "a" * 64,
                "owner_line": "principal",
                "final_version": value.removesuffix("-DEV"),
                "intended_dev_version": f"{value.removesuffix('-DEV')}-DEV",
                "expected_line_head": "c" * 40,
                "reservation_id": f"reservation-{index}",
            }
            for index, value in enumerate(occupied_versions, start=1)
        )
    writer = ReleaseEventWriter(repo, branch=branch)
    root = writer._make_commit(b"", parent=None, message="fixture bootstrap")
    _run(repo, "update-ref", writer.ref, root)
    raw = b""
    parent = root
    for event in events:
        event = dict(event)
        event["expected_event_head"] = parent
        raw += canonical_event_bytes(event) + b"\n"
        parent = writer._make_commit(raw, parent=parent, message="fixture append")
        _run(repo, "update-ref", writer.ref, parent)
    return writer.read_head()


def _fixture() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="cyax-version-lifecycle-"))
    remote = repo.parent / f"{repo.name}-remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    _run(repo, "init", "-q", "-b", "main")
    _run(repo, "config", "user.email", "tests@example.invalid")
    _run(repo, "config", "user.name", "Lifecycle Tests")
    _run(repo, "remote", "add", "origin", str(remote))
    (repo / "Project.toml").write_text('name = "Fixture"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "iterations.toml").write_text(
        "schema_version = 1\n"
        'target_iteration = "fixture"\n'
        "iterations = []\nprospective = []\nretrospective = []\n",
        encoding="utf-8",
    )
    _run(repo, "add", "Project.toml", "iterations.toml")
    _run(repo, "commit", "-q", "-m", "fixture main")
    (repo / "Project.toml").write_text('name = "Fixture"\nversion = "0.2.0"\n', encoding="utf-8")
    _run(repo, "add", "Project.toml")
    _run(repo, "commit", "-q", "-m", "fixture vmm")
    _run(repo, "branch", "vmm")
    _run(repo, "tag", "v0.1.0")
    _run(repo, "tag", "v-0.1")
    _run(repo, "push", "-q", "origin", "main", "vmm", "--tags")
    return repo


class VersionLifecycleStaticTests(unittest.TestCase):
    def test_static_remote_namespace_rejects_ambiguous_framing(self) -> None:
        repo = _fixture()
        sha = "a" * 40
        valid = f"{sha}\trefs/heads/vmm\n"
        for output in (
            valid + "not-an-advertisement\n",
            valid + valid,
            valid.removesuffix("\n") + "\x1c",
            valid.removesuffix("\n"),
        ):
            with self.subTest(output=output), patch(
                "version_lifecycle.static._git", return_value=output.encode("ascii")
            ):
                with self.assertRaisesRegex(
                    StaticValidationError, "remote ref advertisement is malformed"
                ):
                    _remote_refs(repo, "origin")

    def test_option_like_remote_is_blocked_before_git_subprocess(self) -> None:
        repo = _fixture()
        with patch("version_lifecycle.static.subprocess.run") as run:
            result = static_snapshot(
                repo,
                source_repository="fixture/repo",
                remote="--upload-pack=touch /tmp/pwned",
            )
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")
        run.assert_not_called()

    def test_canonical_codec(self) -> None:
        self.assertEqual(canonical_json({"z": 1, "a": [True, "ok"]}), b'{"a":[true,"ok"],"z":1}')
        self.assertEqual(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        for value in (None, 1.0, float("nan"), "é", "line\nfeed"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_json(value)

    def test_version_and_tag_grammar(self) -> None:
        version = parse_package_version("0.7.1-DEV")
        self.assertTrue(version.is_dev)
        self.assertEqual(version.canonical, "0.7.1-DEV")
        self.assertEqual(parse_public_tag("v12.0.3").canonical, "12.0.3")
        for value in ("01.2.3", "1.2.3-alpha", "1.2.3-rc1", "1.2.3+build", "1.2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_package_version(value)
        for value in ("v01.2.3", "v1.2.3-DEV", "v-0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_public_tag(value)

    def test_version_components_match_julia_uint32_bounds(self) -> None:
        maximum = MAX_VERSION_COMPONENT
        for value in (
            f"{maximum}.0.0",
            f"0.{maximum}.0",
            f"0.0.{maximum}",
        ):
            with self.subTest(value=value):
                self.assertEqual(parse_package_version(value).canonical, value)
                self.assertEqual(parse_public_tag(f"v{value}").canonical, value)
        for value in (
            f"{maximum + 1}.0.0",
            f"0.{maximum + 1}.0",
            f"0.0.{maximum + 1}",
            f"{maximum + 1}.0.0-DEV",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_package_version(value)
        with self.assertRaises(ValueError):
            Version(maximum + 1, 0, 0)

    def test_maintenance_lines_use_bounded_shared_components(self) -> None:
        for value in (
            "maintenance/10.1",
            "maintenance/172.16",
            "maintenance/169.254",
            "maintenance/192.168",
        ):
            with self.subTest(value=value):
                expected = tuple(map(int, value.split("/")[1].split(".")))
                self.assertEqual(maintenance_line(value), expected)
        with self.assertRaises(ValueError):
            maintenance_line(f"maintenance/{MAX_VERSION_COMPONENT + 1}.0")

    def test_event_validator_rejects_julia_overflow_component(self) -> None:
        event = {
            "schema_version": 1,
            "event_id": "EVT-000000000001",
            "event_type": "development_reservation_prepared",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "overflow-regression",
            "static_iteration_snapshot": "a" * 64,
            "expected_event_head": "c" * 40,
            "owner_line": "principal",
            "final_version": "0.3.1",
            "intended_dev_version": "0.3.1-DEV",
            "expected_line_head": "b" * 40,
            "reservation_id": "overflow-regression",
        }
        validate_event(event)
        event["final_version"] = f"{MAX_VERSION_COMPONENT + 1}.0.0"
        event["intended_dev_version"] = f"{MAX_VERSION_COMPONENT + 1}.0.0-DEV"
        with self.assertRaises(EventSchemaError):
            validate_event(event)

    @unittest.skipUnless(shutil.which("julia"), "Julia is required for grammar equivalence")
    def test_supported_versions_round_trip_through_julia(self) -> None:
        for value in (
            "0.2.0",
            "0.2.1-DEV",
            "12.0.3",
            f"{MAX_VERSION_COMPONENT}.0.0",
        ):
            with self.subTest(value=value):
                result = subprocess.run(
                    ["julia", "--startup-file=no", "-e", "print(VersionNumber(ARGS[1]))", value],
                    check=True, capture_output=True, text=True,
                )
                self.assertEqual(result.stdout, parse_package_version(value).canonical)
        for value in (
            f"{MAX_VERSION_COMPONENT + 1}.0.0",
            f"0.{MAX_VERSION_COMPONENT + 1}.0",
            f"0.0.{MAX_VERSION_COMPONENT + 1}",
        ):
            with self.subTest(value=value):
                result = subprocess.run(
                    ["julia", "--startup-file=no", "-e", "print(VersionNumber(ARGS[1]))", value],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)

    def test_snapshot_contains_four_verified_digests_and_legacy_tag_is_excluded(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        self.assertEqual(snapshot["canonical_static_iteration_source"], "refs/heads/vmm:iterations.toml")
        self.assertEqual(snapshot["source_repository"], "fixture/repo")
        self.assertEqual(snapshot["public_tag_bindings"][0]["tag"], "v0.1.0")
        self.assertNotIn("v-0.1", [item["tag"] for item in snapshot["public_tag_bindings"]])
        self.assertEqual(
            recompute_snapshot_digests(snapshot)["snapshot_digest"], snapshot.snapshot_digest
        )
        self.assertEqual(validate_static_snapshot(snapshot, repository=repo).snapshot_digest, snapshot.snapshot_digest)
        self.assertIn("0.1.0", snapshot.occupied_versions)
        self.assertIn("0.2.0", snapshot.occupied_versions)

    def test_source_repository_is_a_safe_public_identity(self) -> None:
        repo = _fixture()
        unsafe = (
            "/Users/private/repo",
            "file:///Users/private/repo",
            "local://repo",
            "https://user:password@example.com/org/repo.git",
            "https://100.64.0.1/org/repo",
            "https://127.0.0.1./org/repo",
            "https://10.0.0.1./org/repo",
            "https://169.254.169.254./org/repo",
            "https://2130706433/org/repo",
            "https://0x7f000001/org/repo",
            "https://127.1/org/repo",
            "https://0x7f.0.0.1/org/repo",
            "https://0177.0.0.1/org/repo",
            "https://foo.lan/org/repo",
            "https://foo.corp/org/repo",
            "https://foo.localdomain/org/repo",
            "https://foo.test/org/repo",
            "https://example.com/org/repo",
            "https://example.org/org/repo",
            "https://example.net/org/repo",
            "https://sub.example.com/org/repo",
            "https://home.arpa/org/repo",
            "127.0.0.1",
            "10.0.0.1",
            "0x7f000001",
            "2130706433",
            "017700000001",
            "127.1/repo",
            "10.1/repo",
            "10.0.1/repo",
            "172.16.1/repo",
            "192.168.1/repo",
            "org/10.0.0.1",
            "org/10.0.0.1.git",
            "org/2130706433.git",
            "org/0x7f000001.git",
            "org/017700000001.git",
            "git@github.com:org/10.0.0.1.git",
            "git@github.com:org/2130706433.git",
            "https://github.com/org/10.0.0.1.git",
            "example.com/repo",
            "https://127.0.0.1%2e/org/repo",
            "https://github%2ecom/org/repo",
            "https://foo..com/org/repo",
            "https://github.com/org/repo?",
            "https://github.com/org/repo#",
            "https://[::1]/org/repo",
            "https://intranet/org/repo",
            "https://localhost/org/repo",
        )
        for value in unsafe:
            with self.subTest(value=value):
                result = static_snapshot(repo, source_repository=value)
                self.assertIsInstance(result, BlockedResult)
                assert isinstance(result, BlockedResult)
                self.assertEqual(result.reason_code, "STATIC_SOURCE_REPOSITORY_UNSAFE")
                self.assertNotIn(value, result.detail)

        self.assertEqual(
            sanitize_source_repository("git@github.com:Org/Repo.git"),
            "https://github.com/Org/Repo",
        )
        self.assertEqual(
            sanitize_source_repository("https://8.8.8.8/Org/Repo"),
            "https://8.8.8.8/Org/Repo",
        )
        self.assertEqual(
            sanitize_source_repository("https://[::ffff:8.8.8.8]/Org/Repo"),
            "https://[::ffff:8.8.8.8]/Org/Repo",
        )
        self.assertEqual(
            sanitize_source_repository("https://[2001:4860:4860::8888]/Org/Repo"),
            "https://[2001:4860:4860::8888]/Org/Repo",
        )
        snapshot = static_snapshot(
            repo,
            source_repository="https://github.com/Org/Repo.git",
        )
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        self.assertEqual(snapshot.source_repository, "https://github.com/Org/Repo")

    def test_structural_snapshot_requires_the_canonical_selector(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        tampered = snapshot.to_dict()
        tampered["canonical_static_iteration_source"] = "refs/heads/other:iterations.toml"
        tampered["source_ref"] = "refs/heads/other"
        tampered["snapshot_digest"] = sha256_hex(
            canonical_json({
                key: tampered[key]
                for key in (
                    "snapshot_schema_version", "canonical_static_iteration_source",
                    "source_repository", "source_ref", "source_path", "source_commit",
                    "source_tree", "iterations_toml_sha256", "iteration_ref_bindings",
                    "ref_set_digest", "public_tag_bindings", "tag_set_digest",
                    "occupied_versions",
                )
            })
        )
        with self.assertRaises(StaticValidationError):
            validate_static_snapshot(tampered, structural_only=True)

    def test_caller_set_authority_boolean_cannot_enter_allocation_view(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        forged = type(snapshot)(
            snapshot.to_dict(),
            source_bytes=snapshot.source_bytes,
            authority_verified=True,
        )
        result = global_allocation_view(
            forged,
            _fixture_event_head(repo),
        )
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")

    def test_snapshot_mapping_mutation_cannot_retain_authority(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        # The dataclass is frozen, but its public mapping is intentionally
        # exposed for serialization compatibility.  A caller can therefore
        # alter a field and recompute the public digest; the private binding
        # must still reject the altered authority.
        snapshot.data["source_commit"] = "f" * 40
        snapshot.data["snapshot_digest"] = recompute_snapshot_digests(snapshot)[
            "snapshot_digest"
        ]
        result = global_allocation_view(
            snapshot,
            _fixture_event_head(repo),
        )
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")

    def test_implicit_local_remote_identity_is_blocked(self) -> None:
        repo = _fixture()
        result = static_snapshot(repo)
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_SOURCE_REPOSITORY_UNSAFE")
        self.assertNotIn(repo.name, result.detail)

    def test_schema_versions_require_exact_integer_one(self) -> None:
        repo = _fixture()
        for schema_version in ("true", '"1"'):
            with self.subTest(schema_version=schema_version):
                (repo / "iterations.toml").write_text(
                    f"schema_version = {schema_version}\n"
                    "target_iteration = \"fixture\"\n"
                    "iterations = []\nprospective = []\nretrospective = []\n",
                    encoding="utf-8",
                )
                _run(repo, "add", "iterations.toml")
                _run(repo, "commit", "-q", "-m", "invalid registry schema version")
                _run(repo, "branch", "-f", "vmm")
                _run(repo, "push", "-q", "--force", "origin", "vmm")
                result = static_snapshot(repo, source_repository="fixture/repo")
                self.assertIsInstance(result, BlockedResult)
                assert isinstance(result, BlockedResult)

        valid = static_snapshot(_fixture(), source_repository="fixture/repo")
        assert not isinstance(valid, BlockedResult)
        for value in (True, "1"):
            with self.subTest(snapshot_schema_version=value):
                tampered = valid.to_dict()
                tampered["snapshot_schema_version"] = value
                with self.assertRaises(StaticValidationError):
                    recompute_snapshot_digests(tampered)

    def test_malformed_occupied_versions_raise_structured_validation_error(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        for value in ([{"version": "0.1.0"}], [True], ["0.1.0-DEV"]):
            with self.subTest(value=value):
                tampered = snapshot.to_dict()
                tampered["occupied_versions"] = value
                with self.assertRaises(StaticValidationError):
                    recompute_snapshot_digests(tampered)

    def test_prospective_release_line_is_canonical_and_matches_version(self) -> None:
        repo = _fixture()
        for release_line in ("maintenance/0.1", "maintenance/01.2", "release/0.2"):
            with self.subTest(release_line=release_line):
                (repo / "iterations.toml").write_text(
                    "schema_version = 1\ntarget_iteration = \"fixture\"\n"
                    "[[iterations]]\niteration_id = \"prospective-fixture\"\n"
                    "final_version = \"0.2.0\"\nkind = \"prospective\"\n"
                    f"release_line = \"{release_line}\"\n"
                    "aggregate_impact = {}\nanchor_ref = \"iterations/0.2.0\"\n"
                    "contributing_identities = [{ role = \"issue\", identity = \"#125\" }]\n"
                    "prospective = []\nretrospective = []\n",
                    encoding="utf-8",
                )
                _run(repo, "add", "iterations.toml")
                _run(repo, "commit", "-q", "-m", f"invalid release line {release_line}")
                _run(repo, "branch", "-f", "vmm")
                _run(repo, "push", "-q", "--force", "origin", "vmm")
                result = static_snapshot(repo, source_repository="fixture/repo")
                self.assertIsInstance(result, BlockedResult)
                assert isinstance(result, BlockedResult)
                self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")

        (repo / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"fixture\"\n"
            "iterations = []\nprospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "valid canonical root registry")
        _run(repo, "branch", "-f", "vmm")
        _run(repo, "push", "-q", "--force", "origin", "vmm")
        (repo / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"fixture\"\n"
            "[[iterations]]\niteration_id = \"prospective-fixture\"\n"
            "final_version = \"0.2.1\"\nkind = \"prospective\"\n"
            "release_line = \"maintenance/0.2\"\naggregate_impact = {}\n"
            "anchor_ref = \"iterations/0.2.1\"\n"
            "contributing_identities = [{ role = \"issue\", identity = \"#125\" }]\n"
            "prospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "valid maintenance release line")
        _run(repo, "tag", "-a", "iterations/0.2.1", "-m", "fixture anchor")
        _run(repo, "push", "-q", "origin", "--tags")
        result = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(result, BlockedResult)

    def test_snapshot_tampering_and_staleness_fail_closed(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        self.assertFalse(snapshot_is_stale(snapshot.to_dict(), repo))
        tampered = snapshot.to_dict()
        tampered["occupied_versions"] = list(reversed(tampered["occupied_versions"]))
        with self.assertRaises(StaticValidationError):
            validate_static_snapshot(tampered)
        (repo / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"fixture-new\"\n"
            "iterations = []\nprospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "change static source")
        _run(repo, "branch", "-f", "vmm")
        _run(repo, "push", "-q", "--force", "origin", "vmm")
        self.assertTrue(snapshot_is_stale(snapshot, repo))

    def test_remote_vmm_wins_over_stale_local_vmm(self) -> None:
        repo = _fixture()
        local_vmm = _run(repo, "rev-parse", "refs/heads/vmm")
        (repo / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"remote-update\"\n"
            "iterations = []\nprospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "remote-only vmm update")
        remote_vmm = _run(repo, "rev-parse", "HEAD")
        _run(repo, "push", "-q", "origin", "HEAD:refs/heads/vmm")
        self.assertEqual(_run(repo, "rev-parse", "refs/heads/vmm"), local_vmm)
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        self.assertEqual(snapshot.source_commit, remote_vmm)
        self.assertNotEqual(snapshot.source_commit, local_vmm)

    def test_remote_object_is_fetched_without_updating_local_tracking_ref(self) -> None:
        repo = _fixture()
        remote_url = _run(repo, "config", "--get", "remote.origin.url")
        publisher = repo.parent / f"{repo.name}-publisher"
        subprocess.run(
            ["git", "clone", "-q", "--no-local", "--branch", "vmm", remote_url, str(publisher)],
            check=True,
        )
        _run(publisher, "config", "user.email", "publisher@example.invalid")
        _run(publisher, "config", "user.name", "Publisher")
        (publisher / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"fetched-object\"\n"
            "iterations = []\nprospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(publisher, "add", "iterations.toml")
        _run(publisher, "commit", "-q", "-m", "publisher vmm update")
        published = _run(publisher, "rev-parse", "HEAD")
        _run(publisher, "push", "-q", "origin", "HEAD:refs/heads/vmm")
        local_tracking = _run(repo, "rev-parse", "refs/remotes/origin/vmm")
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        self.assertEqual(snapshot.source_commit, published)
        self.assertEqual(_run(repo, "rev-parse", "refs/remotes/origin/vmm"), local_tracking)

    def test_remote_tag_change_and_cross_namespace_anchor_conflict_are_detected(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        _run(repo, "tag", "v0.2.0")
        _run(repo, "push", "-q", "origin", "v0.2.0")
        self.assertTrue(snapshot_is_stale(snapshot, repo))

        _run(repo, "tag", "-a", "iterations/0.3.0", "-m", "anchor tag")
        _run(repo, "update-ref", "refs/heads/iterations/0.3.0", "HEAD")
        _run(repo, "push", "-q", "origin", "v0.2.0", "refs/tags/iterations/0.3.0", "refs/heads/iterations/0.3.0")
        result = static_snapshot(repo, source_repository="fixture/repo")
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")

    def test_branch_only_iteration_anchor_is_rejected(self) -> None:
        repo = _fixture()
        (repo / "iterations.toml").write_text(
            "schema_version = 1\n"
            'target_iteration = "fixture"\n'
            "[[iterations]]\n"
            'iteration_id = "prospective-fixture"\n'
            'final_version = "0.3.0"\n'
            'kind = "prospective"\n'
            'release_line = "principal"\n'
            "aggregate_impact = {}\n"
            'anchor_ref = "iterations/0.3.0"\n'
            'contributing_identities = [{ role = "issue", identity = "#125" }]\n'
            "prospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "branch-only anchor fixture")
        _run(repo, "push", "-q", "origin", "HEAD:refs/heads/iterations/0.3.0")
        result = static_snapshot(repo, source_repository="fixture/repo")
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")

    def test_structural_snapshot_without_source_bytes_cannot_enter_allocation_view(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        structural = snapshot.to_dict()
        result = global_allocation_view(structural, _fixture_event_head(repo))
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_SOURCE_BYTES_UNAVAILABLE")

    def test_principal_and_maintenance_allocation_use_global_occupancy(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        head = _fixture_event_head(
            repo,
            occupied_versions=("0.7.1-DEV", "0.7.3", "1.2.0"),
        )
        view = global_allocation_view(snapshot, head)
        self.assertEqual(view.status, "READY")
        principal = select_principal_sentinel("0.7.0", snapshot, head)
        self.assertEqual(principal.status, "BLOCKED")
        self.assertEqual(principal.reason_code, "PRINCIPAL_SENTINEL_UNAVAILABLE")
        maintenance = select_maintenance_version("maintenance/0.7", "0.7.0", snapshot, head)
        self.assertEqual(maintenance.status, "AVAILABLE")
        self.assertEqual(maintenance.version.canonical, "0.7.2-DEV")
        self.assertTrue(view.is_available("0.7.2"))
        self.assertFalse(view.is_available("0.7.3"))

    def test_allocation_blocks_when_julia_patch_domain_is_exhausted(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        head = _fixture_event_head(repo)
        closed = f"0.7.{MAX_VERSION_COMPONENT}"
        principal = select_principal_sentinel(closed, snapshot, head)
        self.assertEqual(
            (principal.status, principal.reason_code),
            ("BLOCKED", "PRINCIPAL_VERSION_EXHAUSTED"),
        )
        maintenance = select_maintenance_version("maintenance/0.7", closed, snapshot, head)
        self.assertEqual(
            (maintenance.status, maintenance.reason_code),
            ("BLOCKED", "MAINTENANCE_PATCH_EXHAUSTED"),
        )

    def test_invalid_event_head_never_declares_a_version_available(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        for head in (
            None,
            {"status": "VALIDATED", "head_commit": "a" * 40, "occupied_versions": ["not-a-version"], "proof_digest": "0" * 64},
            {
                "status": "VALIDATED",
                "head_commit": "a" * 40,
                "occupied_versions": ["0.7.1"],
                "proof_digest": "0" * 64,
            },
        ):
            result = global_allocation_view(snapshot, head)
            self.assertIsInstance(result, BlockedResult)
            assert isinstance(result, BlockedResult)
            self.assertEqual(result.reason_code, "ALLOCATION_EVENT_HEAD_INVALID")

        result = select_principal_sentinel(
            "0.7.0", snapshot, _fixture_event_head(repo, occupied_versions=("0.7.1",))
        )
        self.assertEqual(result.reason_code, "PRINCIPAL_SENTINEL_UNAVAILABLE")

    def test_caller_forged_mapping_occupancy_proof_is_rejected(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        proof = {
            "head_commit": "a" * 40,
            "occupied_versions": [],
        }
        forged = {
            "status": "VALIDATED",
            **proof,
            "proof_digest": sha256_hex(canonical_json(proof)),
        }
        result = global_allocation_view(snapshot, forged)
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "ALLOCATION_EVENT_HEAD_INVALID")

    def test_writer_ledger_head_is_the_event_head_boundary(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        head = _fixture_event_head(repo)
        view = global_allocation_view(snapshot, head)
        self.assertEqual(view.status, "READY")
        self.assertEqual(view.event_head_commit, head.commit)
        malformed = LedgerHead(commit="b" * 40, raw=b'{"not":"an event"}\n', events=())
        result = global_allocation_view(snapshot, malformed)
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "ALLOCATION_EVENT_HEAD_INVALID")

    def test_static_and_event_authorities_must_share_one_repository(self) -> None:
        repository_a = _fixture()
        repository_b = _fixture()
        snapshot_a = static_snapshot(
            repository_a, source_repository="fixture/repository-a"
        )
        assert not isinstance(snapshot_a, BlockedResult)
        event_head_b = _fixture_event_head(repository_b)

        result = global_allocation_view(snapshot_a, event_head_b)

        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "ALLOCATION_EVENT_HEAD_INVALID")
        self.assertIn("different repositories", result.detail)

    def test_noncanonical_orphan_event_history_cannot_enter_allocation(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)

        canonical = global_allocation_view(snapshot, _fixture_event_head(repo))
        self.assertEqual(canonical.status, "READY")

        second_ledger = global_allocation_view(
            snapshot,
            _fixture_event_head(repo, branch="second-ledger"),
        )
        self.assertIsInstance(second_ledger, BlockedResult)
        assert isinstance(second_ledger, BlockedResult)
        self.assertEqual(second_ledger.reason_code, "ALLOCATION_EVENT_HEAD_INVALID")

    def test_historical_event_snapshot_remains_occupied_after_static_refresh(self) -> None:
        repo = _fixture()
        current = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(current, BlockedResult)
        historical_event = {
            "schema_version": 1,
            "event_id": "EVT-000000000001",
            "event_type": "development_reservation_prepared",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "historical-reservation",
            "static_iteration_snapshot": "a" * 64,
            "expected_event_head": "b" * 40,
            "owner_line": "principal",
            "final_version": "0.3.1",
            "intended_dev_version": "0.3.1-DEV",
            "expected_line_head": "c" * 40,
            "reservation_id": "reservation-1",
        }
        head = _fixture_event_head(repo, events=(historical_event,))
        view = global_allocation_view(current, head)
        self.assertEqual(view.status, "READY")
        self.assertIn("0.3.1", view.mutable_occupied)
        self.assertFalse(view.is_available("0.3.1"))

    def test_preentry_abort_releases_reservation_for_allocation_replay(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        prepared = {
            "schema_version": 1,
            "event_id": "EVT-000000000001",
            "event_type": "development_reservation_prepared",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "reservation-prepare",
            "static_iteration_snapshot": "a" * 64,
            "expected_event_head": "b" * 40,
            "owner_line": "principal",
            "final_version": "0.3.1",
            "intended_dev_version": "0.3.1-DEV",
            "expected_line_head": "c" * 40,
            "reservation_id": "reservation-1",
        }
        active = global_allocation_view(
            snapshot,
            _fixture_event_head(repo, events=(prepared,)),
        )
        self.assertEqual(active.status, "READY")
        self.assertFalse(active.is_available("0.3.1"))

        aborted = dict(
            prepared,
            event_id="EVT-000000000002",
            event_type="development_reservation_aborted",
            transaction_id="reservation-abort",
            expected_event_head="d" * 40,
            abort_reason="branch_not_created",
            non_entry_evidence={
                "verified": True,
                "reservation_id": prepared["reservation_id"],
                "owner_line": "principal",
                "final_version": "0.3.1",
                "intended_dev_version": "0.3.1-DEV",
                "line_ref": "refs/heads/vmm",
                "expected_line_head": prepared["expected_line_head"],
                "line_state": "unchanged",
                "observed_line_head": prepared["expected_line_head"],
                "dev_not_entered": True,
                "exclusion_verified": True,
                "observed_at_utc": "2026-09-20T12:34:55Z",
                "evidence_ref": "evidence/non-entry.json",
                "evidence_digest": "d" * 64,
            },
        )
        aborted.pop("expected_line_head")
        after_abort = global_allocation_view(
            snapshot,
            _fixture_event_head(repo, events=(prepared, aborted)),
        )
        self.assertEqual(after_abort.status, "READY")
        self.assertNotIn("0.3.1", after_abort.mutable_occupied)
        self.assertTrue(after_abort.is_available("0.3.1"))

    def test_unresolved_selector_returns_required_blocked_result(self) -> None:
        repo = _fixture()
        result = static_snapshot(repo, selector="refs/heads/missing:iterations.toml")
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")

    def test_prospective_anchor_binding_and_retrospective_truth_are_separate(self) -> None:
        repo = _fixture()
        (repo / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"fixture\"\n"
            "[[iterations]]\n"
            "iteration_id = \"prospective-fixture\"\nfinal_version = \"0.3.0\"\n"
            "kind = \"prospective\"\nrelease_line = \"principal\"\naggregate_impact = {}\n"
            "anchor_ref = \"iterations/0.3.0\"\n"
            "contributing_identities = [{ role = \"issue\", identity = \"#125\" }]\n"
            "prospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "prospective registry")
        _run(repo, "branch", "-f", "vmm")
        _run(repo, "tag", "-a", "iterations/0.3.0", "-m", "fixture anchor")
        _run(repo, "push", "-q", "--force", "origin", "vmm", "--tags")
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        self.assertIn("0.3.0", snapshot.occupied_versions)

        # The retrospective schema keeps an actual historical DEV spelling as
        # a separate fact while occupation uses its final namespace member.
        (repo / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"fixture\"\n"
            "prospective = []\niterations = []\n"
            "[[retrospective]]\niteration_id = \"history\"\n"
            "declared_version = \"0.4.0\"\nproject_toml_version = \"0.4.0-DEV\"\n"
            "project_toml_carried = true\nhistorical_release_status = \"unreleased\"\n"
            "anchor_sha = \"a\"\nanchor_tree = \"b\"\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "retrospective fixture")
        _run(repo, "branch", "-f", "vmm")
        _run(repo, "push", "-q", "--force", "origin", "vmm")
        retrospective = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(retrospective, BlockedResult)
        assert not isinstance(retrospective, BlockedResult)
        self.assertIn("0.4.0", retrospective.occupied_versions)
        self.assertNotIn("0.4.0-DEV", retrospective.occupied_versions)

    def test_retrospective_truth_requires_explicit_consistent_fields(self) -> None:
        cases = {
            "contradictory final_version": (
                "final_version contradicts declared_version",
                'declared_version = "0.4.0"\n'
                'final_version = "0.4.1"\n'
                'project_toml_version = "0.4.0-DEV"\n'
                "project_toml_carried = true\n",
            ),
            "missing carried status": (
                "requires project_toml_carried",
                'declared_version = "0.4.0"\n'
                'project_toml_version = "0.4.0-DEV"\n',
            ),
            "carried without actual version": (
                "marks Project.toml carried but omits actual version",
                'declared_version = "0.4.0"\n'
                "project_toml_carried = true\n",
            ),
        }
        for label, (expected_detail, historical_fields) in cases.items():
            with self.subTest(label=label):
                repo = _fixture()
                (repo / "iterations.toml").write_text(
                    "schema_version = 1\ntarget_iteration = \"fixture\"\n"
                    "iterations = []\nprospective = []\n"
                    "[[retrospective]]\niteration_id = \"history\"\n"
                    + historical_fields
                    + 'historical_release_status = "unreleased"\n'
                    + 'anchor_sha = "a"\nanchor_tree = "b"\n',
                    encoding="utf-8",
                )
                _run(repo, "add", "iterations.toml")
                _run(repo, "commit", "-q", "-m", "invalid retrospective fixture")
                _run(repo, "branch", "-f", "vmm")
                _run(repo, "push", "-q", "--force", "origin", "vmm")
                result = static_snapshot(repo, source_repository="fixture/repo")
                self.assertIsInstance(result, BlockedResult)
                assert isinstance(result, BlockedResult)
                self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")
                self.assertIn(expected_detail, result.detail)

    def test_root_registry_rejects_forwardported_maintenance_metadata(self) -> None:
        repo = _fixture()
        (repo / "iterations.toml").write_text(
            "schema_version = 1\ntarget_iteration = \"fixture\"\n"
            "[[iterations]]\niteration_id = \"maintenance\"\nfinal_version = \"0.1.1\"\n"
            "kind = \"prospective\"\nrelease_line = \"maintenance/0.1\"\n"
            "aggregate_impact = {}\nanchor_ref = \"iterations/0.1.1\"\n"
            "contributing_identities = [{ role = \"issue\", identity = \"#125\" }]\n"
            "prospective = []\nretrospective = []\n",
            encoding="utf-8",
        )
        _run(repo, "add", "iterations.toml")
        _run(repo, "commit", "-q", "-m", "invalid maintenance registry")
        _run(repo, "branch", "-f", "vmm")
        _run(repo, "push", "-q", "--force", "origin", "vmm")
        result = static_snapshot(repo, source_repository="fixture/repo")
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
