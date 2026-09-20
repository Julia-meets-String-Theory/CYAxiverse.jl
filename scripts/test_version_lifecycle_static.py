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


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from version_lifecycle import (  # noqa: E402
    BlockedResult,
    canonical_json,
    global_allocation_view,
    parse_package_version,
    parse_public_tag,
    recompute_snapshot_digests,
    select_maintenance_version,
    select_principal_sentinel,
    sha256_hex,
    snapshot_is_stale,
    static_snapshot,
    validate_static_snapshot,
    validated_occupancy_proof,
)
from version_lifecycle.static import StaticValidationError  # noqa: E402
from version_lifecycle.writer import LedgerHead  # noqa: E402
from version_lifecycle.events import canonical_event_bytes  # noqa: E402


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


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

    @unittest.skipUnless(shutil.which("julia"), "Julia is required for grammar equivalence")
    def test_supported_versions_round_trip_through_julia(self) -> None:
        for value in ("0.2.0", "0.2.1-DEV", "12.0.3"):
            with self.subTest(value=value):
                result = subprocess.run(
                    ["julia", "--startup-file=no", "-e", "print(VersionNumber(ARGS[1]))", value],
                    check=True, capture_output=True, text=True,
                )
                self.assertEqual(result.stdout, parse_package_version(value).canonical)

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

    def test_snapshot_tampering_and_staleness_fail_closed(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
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
        subprocess.run(["git", "clone", "-q", "--no-local", remote_url, str(publisher)], check=True)
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

    def test_structural_snapshot_without_source_bytes_cannot_enter_allocation_view(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        self.assertNotIsInstance(snapshot, BlockedResult)
        assert not isinstance(snapshot, BlockedResult)
        structural = snapshot.to_dict()
        result = global_allocation_view(structural, validated_occupancy_proof("a" * 40, []))
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_SOURCE_BYTES_UNAVAILABLE")

    def test_principal_and_maintenance_allocation_use_global_occupancy(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        head = validated_occupancy_proof(
            "a" * 40,
            ["0.7.1-DEV", "0.7.3", "1.2.0"],
            static_snapshot_digest=snapshot.snapshot_digest,
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

    def test_invalid_event_head_never_declares_a_version_available(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        for head in (
            None,
            {"status": "VALIDATED", "head_commit": "a" * 40, "occupied_versions": ["not-a-version"], "proof_digest": "0" * 64},
            validated_occupancy_proof("a" * 40, ["0.7.1"] ) | {"proof_digest": "0" * 64},
        ):
            result = global_allocation_view(snapshot, head)
            self.assertIsInstance(result, BlockedResult)
            assert isinstance(result, BlockedResult)
            self.assertEqual(result.reason_code, "ALLOCATION_EVENT_HEAD_INVALID")

        proof = validated_occupancy_proof(
            "a" * 40,
            ["0.7.1"],
            static_snapshot_digest=snapshot.snapshot_digest,
        )
        result = select_principal_sentinel("0.7.0", snapshot, proof)
        self.assertEqual(result.reason_code, "PRINCIPAL_SENTINEL_UNAVAILABLE")

    def test_writer_ledger_head_is_the_event_head_boundary(self) -> None:
        repo = _fixture()
        snapshot = static_snapshot(repo, source_repository="fixture/repo")
        assert not isinstance(snapshot, BlockedResult)
        head = LedgerHead(commit="b" * 40, raw=b"", events=())
        view = global_allocation_view(snapshot, head)
        self.assertEqual(view.status, "READY")
        self.assertEqual(view.event_head_commit, "b" * 40)
        malformed = LedgerHead(commit="b" * 40, raw=b'{"not":"an event"}\n', events=())
        result = global_allocation_view(snapshot, malformed)
        self.assertIsInstance(result, BlockedResult)
        assert isinstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "ALLOCATION_EVENT_HEAD_INVALID")

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
        raw = canonical_event_bytes(historical_event) + b"\n"
        head = LedgerHead(commit="c" * 40, raw=raw, events=(historical_event,))
        view = global_allocation_view(current, head)
        self.assertEqual(view.status, "READY")
        self.assertIn("0.3.1", view.mutable_occupied)
        self.assertFalse(view.is_available("0.3.1"))

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
