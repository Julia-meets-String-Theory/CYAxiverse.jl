"""Focused tests for the immutable lifecycle-manifest authority."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.manifests import (  # noqa: E402
    CreateOnlyLifecycleWriter,
    CreateOnlyLifecycleManifestPort,
    ManifestError,
    ManifestConflict,
    build_lifecycle_ref_snapshot,
    canonical_manifest_bytes,
    lifecycle_ref_snapshot,
    lifecycle_ref_for_manifest,
    make_manifest_commit,
    manifest_id,
    publication_key_fixture,
    validate_complete_lifecycle_refs,
    validate_lifecycle_ref_snapshot,
    seal_manifest,
    validate_manifest,
)
from version_lifecycle.git_refs import (  # noqa: E402
    ProtectionEvidence,
    canonical_remote_authority,
)
from version_lifecycle.static import _mark_authority_verified  # noqa: E402
from version_lifecycle.authorization import (  # noqa: E402
    AuthorizationResolution,
    canonical_authorization_bytes,
    seal_authorization,
)


OWNER_AUTHORIZATION = "AUTH-SHA256-" + "7" * 64
OWNER_AUTHORIZATION_REF = "owner-authority://synthetic/grant-manifest-tests"
OWNER_AUTHORIZATION_DIGEST = "7" * 64


def authorization_fields() -> dict[str, str]:
    return {
        "owner_authorization": OWNER_AUTHORIZATION,
        "owner_authorization_ref": OWNER_AUTHORIZATION_REF,
        "owner_authorization_digest": OWNER_AUTHORIZATION_DIGEST,
    }


class ManifestAuthority:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.references: dict[str, str] = {}

    def bind(
        self,
        manifest: dict[str, object],
        *,
        action: str = "create-release-manifest",
    ) -> dict[str, object]:
        provisional = seal_manifest(manifest)
        target = lifecycle_ref_for_manifest(provisional)
        reference = f"owner-authority://fixture/manifest-{len(self.records) + 1}"
        record = seal_authorization({
            "schema_version": 1,
            "repository": "fixture-repository",
            "owner_account": "fixture-owner",
            "authority_source_ref": reference,
            "issued_at_utc": "2026-09-21T00:00:00Z",
            "expires_at_utc": "2026-09-23T00:00:00Z",
            "transaction_id": "tx-1",
            "owner_line": "principal",
            "final_version": str(manifest["final_version"]),
            "authorized_actions": [action],
            "target_refs": [target],
        })
        self.records[reference] = record
        self.references[target] = reference
        rebound = dict(manifest)
        rebound.update({
            "owner_authorization": record["owner_authorization"],
            "owner_authorization_ref": reference,
            "owner_authorization_digest": record["owner_authorization_digest"],
        })
        return seal_manifest(rebound)

    def reference(self, action: str, target: str, version: str) -> str:
        return self.references[target]

    def fetch_owner_authorization(self, reference: str) -> AuthorizationResolution:
        record = self.records[reference]
        return AuthorizationResolution(
            record, canonical_authorization_bytes(record), True
        )


WRITER_CONTEXT = {
    "transaction_id": "tx-1",
    "owner_line": "principal",
    "final_version": "1.2.3",
    "action": "create-release-manifest",
}


def writer_snapshot(
    occupied: list[str] | None = None,
    *,
    static_occupied: list[str] | None = None,
    source_commit: str = "a" * 40,
) -> object:
    from test_version_lifecycle_static import (  # noqa: PLC0415
        lifecycle_snapshot,
        static_snapshot,
    )
    from version_lifecycle.allocation import global_allocation_view  # noqa: PLC0415

    return global_allocation_view(
        static_snapshot(static_occupied, source_commit=source_commit),
        lifecycle_snapshot(occupied or []),
    )


def remote_writer_snapshot(repository: Path) -> object:
    from test_version_lifecycle_static import (  # noqa: PLC0415
        SOURCE_REPOSITORY,
        static_snapshot,
    )
    from version_lifecycle.allocation import global_allocation_view  # noqa: PLC0415

    authority = canonical_remote_authority(repository, "origin")
    root_parent = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    static = _mark_authority_verified(
        static_snapshot(source_commit=root_parent), authority
    )
    lifecycle = build_lifecycle_ref_snapshot(
        repository,
        source_repository=SOURCE_REPOSITORY,
        static_source_commit=root_parent,
    )
    return global_allocation_view(static, lifecycle)


class ManifestTests(unittest.TestCase):
    def draft(self, **extra: object) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "manifest_type": "version-claimed",
            "timestamp_utc": "2026-09-22T00:00:00Z",
            "predecessor_refs": [],
            **authorization_fields(),
            "transaction_id": "tx-1",
            "static_iteration_snapshot": "a" * 64,
            "lifecycle_ref_snapshot": "b" * 64,
            "final_version": "1.2.3",
            "owner_line": "principal",
        }
        result.update(extra)
        return result

    def prepared_draft(self, **extra: object) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "manifest_type": "reservation-prepared",
            "timestamp_utc": "2026-09-22T00:00:00Z",
            "predecessor_refs": [],
            **authorization_fields(),
            "transaction_id": "tx-1",
            "static_iteration_snapshot": "a" * 64,
            "lifecycle_ref_snapshot": "b" * 64,
            "final_version": "1.2.3",
            "reserved_final": "1.2.3",
            "owner_line": "principal",
            "intended_dev_version": "1.2.3-DEV",
            "expected_line_head": "a" * 40,
        }
        result.update(extra)
        return result

    def chain(self) -> list[dict[str, object]]:
        common = {
            "schema_version": 1,
            "timestamp_utc": "2026-09-22T00:00:00Z",
            **authorization_fields(),
            "transaction_id": "tx-chain",
            "static_iteration_snapshot": "a" * 64,
            "lifecycle_ref_snapshot": "b" * 64,
        }
        prepared_reservation = seal_manifest({
            **common, "manifest_type": "reservation-prepared", "predecessor_refs": [],
            "final_version": "1.2.3", "owner_line": "principal",
            "reserved_final": "1.2.3", "intended_dev_version": "1.2.3-DEV",
            "expected_line_head": "1" * 40,
        })
        opened_reservation = seal_manifest({
            **common, "manifest_type": "reservation-opened",
            "predecessor_refs": [lifecycle_ref_for_manifest(prepared_reservation)],
            "final_version": "1.2.3", "owner_line": "principal",
            "reserved_final": "1.2.3", "intended_dev_version": "1.2.3-DEV",
            "actual_dev_head": "2" * 40,
        })
        claim = seal_manifest({
            **common, "manifest_type": "version-claimed",
            "predecessor_refs": [lifecycle_ref_for_manifest(opened_reservation)],
            "final_version": "1.2.3", "owner_line": "principal",
        })
        consumed_reservation = seal_manifest({
            **common, "manifest_type": "reservation-consumed",
            "predecessor_refs": [lifecycle_ref_for_manifest(opened_reservation)],
            "final_version": "1.2.3", "owner_line": "principal",
            "reserved_final": "1.2.3", "closed_final_version": "1.2.3",
            "closure_anchor_ref": "refs/tags/iterations/1.2.3",
            "terminal_disposition": "closed",
        })
        candidate = seal_manifest({
            **common, "manifest_type": "candidate-opened",
            "predecessor_refs": [lifecycle_ref_for_manifest(claim)],
            "candidate_id": "candidate-1", "candidate_ref": "refs/heads/candidates/v1.2.3/candidate-1",
            "candidate_sha": "c" * 40, "candidate_tree": "d" * 40,
            "final_version": "1.2.3", "release_line": "principal",
            "anchor_ref": "refs/tags/iterations/1.2.3", "anchor_sha": "c" * 40,
            "anchor_tree": "d" * 40,
            "main_at_candidate_sha": "1" * 40,
            "main_at_candidate_version": "1.2.2",
        })
        intent = seal_manifest({
            **common, "manifest_type": "release-intent-prepared",
            "predecessor_refs": [lifecycle_ref_for_manifest(candidate)],
            "candidate_id": "candidate-1", "candidate_ref": candidate["candidate_ref"],
            "candidate_sha": candidate["candidate_sha"], "candidate_tree": candidate["candidate_tree"],
            "final_version": "1.2.3", "release_line": "principal", "public_tag": "v1.2.3",
            "final_release_sha": "e" * 40, "final_release_tree": "d" * 40,
            "certification_binding": "tree-bound", "certification_subject_sha": "c" * 40,
            "certification_subject_tree": "d" * 40, "certification_policy_revision": "policy-r1",
            "certification_harness_revision": "harness-r1", "certification_environment": "ci",
            "certification_evidence_refs": ["evidence/certification-r1"],
            "certification_transfer_evidence": {
                "candidate_tree": "d" * 40,
                "final_release_tree": "d" * 40,
                "anchor_tree": "d" * 40,
                "evidence_ref": "evidence/transfer-r1",
            },
            "anchor_ref": candidate["anchor_ref"], "anchor_sha": candidate["anchor_sha"],
            "anchor_tree": candidate["anchor_tree"],
        })
        release = seal_manifest({
            **common, "manifest_type": "released",
            "predecessor_refs": [lifecycle_ref_for_manifest(intent)],
            "final_version": "1.2.3", "release_line": "principal", "public_tag": "v1.2.3",
            "candidate_ref": candidate["candidate_ref"], "candidate_sha": candidate["candidate_sha"],
            "candidate_tree": candidate["candidate_tree"], "anchor_ref": candidate["anchor_ref"],
            "anchor_sha": candidate["anchor_sha"], "anchor_tree": candidate["anchor_tree"],
            "final_release_sha": intent["final_release_sha"], "final_release_tree": intent["final_release_tree"],
            "certification_binding": intent["certification_binding"],
            "certification_subject_sha": intent["certification_subject_sha"],
            "certification_subject_tree": intent["certification_subject_tree"],
            "certification_policy_revision": intent["certification_policy_revision"],
            "certification_harness_revision": intent["certification_harness_revision"],
            "certification_environment": intent["certification_environment"],
            "certification_evidence_refs": intent["certification_evidence_refs"],
            "certification_transfer_evidence": intent["certification_transfer_evidence"],
            "closure_timestamp_utc": "2026-09-22T00:10:00Z", "previous_main_sha": "1" * 40,
            "previous_main_version": "1.2.2", "main_at_release_sha": "e" * 40,
            "main_at_release_version": "1.2.3",
            "main_at_candidate_sha": candidate["main_at_candidate_sha"],
            "main_at_candidate_version": candidate["main_at_candidate_version"],
            "evidence_refs": ["evidence/release-r1"],
        })
        publication = seal_manifest({
            **{key: value for key, value in common.items() if key not in {"transaction_id", "static_iteration_snapshot", "lifecycle_ref_snapshot"}},
            "manifest_type": "publication", "timestamp_utc": "2026-09-22T01:00:00Z",
            "predecessor_refs": [lifecycle_ref_for_manifest(release)],
            "released_manifest_ref": lifecycle_ref_for_manifest(release), "released_manifest_id": release["manifest_id"],
            "released_manifest_digest": hashlib.sha256(canonical_manifest_bytes(release)).hexdigest(),
            "public_tag": "v1.2.3", "tag_commit": release["final_release_sha"], "tag_tree": release["final_release_tree"],
            "github_release_id": 1, "github_release_url": "https://github.com/cyaxiverse/CYAxiverse.jl/releases/tag/v1.2.3",
            "published_at_utc": "2026-09-22T01:00:00Z", "publication_evidence_ref": "evidence/publication-r1",
            "publication_evidence_digest": "9" * 64,
        })
        return [
            prepared_reservation, opened_reservation, claim,
            consumed_reservation, candidate, intent, release, publication,
        ]

    def test_content_derived_id_and_exact_wire_bytes(self) -> None:
        value = seal_manifest(self.draft())
        self.assertEqual(value["manifest_id"], manifest_id(value))
        self.assertEqual(canonical_manifest_bytes(value), json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(lifecycle_ref_for_manifest(value), "refs/heads/lifecycle/v1/claims/v1.2.3")

    def test_manifests_reject_private_authorization_references(self) -> None:
        unsafe = (
            "file:///Users/alice/private/grant.json",
            "/Users/alice/private/grant.json",
            "owner-authority://alice:password@synthetic/grant",
            "owner-authority://synthetic/grant?secret=value",
            "owner-authority://synthetic/grant#token-locator",
            "owner-authority://synthetic/secret-grant",
            "owner-authority://synthetic/ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "owner-authority://localhost/grant",
            "owner-authority://synthetic/Users/alice/grant",
        )
        for reference in unsafe:
            allocation = self.draft(owner_authorization_ref=reference)
            publication = dict(self.chain()[-1])
            publication.pop("manifest_id")
            publication.pop("publication_id")
            publication["owner_authorization_ref"] = reference
            for manifest in (allocation, publication):
                with self.subTest(
                    kind=manifest["manifest_type"], reference=reference
                ), self.assertRaisesRegex(ManifestError, "public-safe"):
                    validate_manifest(seal_manifest(manifest))

    def test_authorization_binding_does_not_create_identity_cycle(self) -> None:
        first = seal_manifest(self.draft())
        second = seal_manifest(self.draft(
            owner_authorization="AUTH-SHA256-" + "8" * 64,
            owner_authorization_ref="owner-authority://synthetic/grant-other",
            owner_authorization_digest="8" * 64,
        ))
        self.assertEqual(first["manifest_id"], second["manifest_id"])
        self.assertEqual(lifecycle_ref_for_manifest(first), lifecycle_ref_for_manifest(second))
        self.assertNotEqual(canonical_manifest_bytes(first), canonical_manifest_bytes(second))

    def test_fixed_publication_fixture(self) -> None:
        fixture = publication_key_fixture()
        self.assertEqual(fixture["sha256"], "ebd4b5200df5c8e3aa30bee44a17edbfceef805b21fde551196ca1c006ed8c04")
        self.assertEqual(fixture["publication_id"], "pub-ebd4b5200df5c8e3aa30bee44a17edbfceef805b21fde551196ca1c006ed8c04")
        self.assertEqual(fixture["ref"], "refs/heads/lifecycle/v1/publications/v1.2.3/pub-ebd4b5200df5c8e3aa30bee44a17edbfceef805b21fde551196ca1c006ed8c04")

    def test_tampering_and_noncanonical_manifest_are_rejected(self) -> None:
        value = seal_manifest(self.draft())
        value["final_version"] = "1.2.4"
        with self.assertRaises(ManifestError):
            validate_manifest(value)
        raw = canonical_manifest_bytes(seal_manifest(self.draft())) + b"\n"
        self.assertTrue(raw.endswith(b"\n"))

    def test_schema_rejects_extra_fields_and_unsorted_sets(self) -> None:
        value = seal_manifest(self.draft(extra_field="forbidden"))
        with self.assertRaisesRegex(ManifestError, "undeclared manifest fields"):
            validate_manifest(value)
        value = seal_manifest(self.draft(certification_evidence_refs=["z", "a"]))
        value["manifest_type"] = "released"
        with self.assertRaises(ManifestError):
            validate_manifest(value)

    def test_publication_rejects_forbidden_predecessor_fields(self) -> None:
        value = seal_manifest({
            "schema_version": 1,
            "manifest_type": "publication",
            "timestamp_utc": "2026-09-22T01:00:00Z",
            "predecessor_refs": ["refs/heads/lifecycle/v1/releases/v1.2.3"],
            **authorization_fields(),
            "released_manifest_ref": "refs/heads/lifecycle/v1/releases/v1.2.3",
            "released_manifest_id": "LIF-SHA256-" + "a" * 64,
            "released_manifest_digest": "b" * 64,
            "public_tag": "v1.2.3",
            "tag_commit": "c" * 40,
            "tag_tree": "d" * 40,
            "github_release_id": "release-1",
            "github_release_url": "https://github.com/cyaxiverse/CYAxiverse.jl/releases/tag/v1.2.3",
            "published_at_utc": "2026-09-22T01:00:00Z",
            "publication_evidence_ref": "evidence/publication",
            "publication_evidence_digest": "e" * 64,
            "candidate_ref": "refs/heads/lifecycle/v1/candidates/v1.2.3/candidate-1",
        })
        with self.assertRaisesRegex(ManifestError, "undeclared manifest fields"):
            validate_manifest(value)

    def test_schema_rejects_missing_timestamp_bad_array_and_bad_publication_path(self) -> None:
        value = self.draft()
        value.pop("owner_authorization")
        with self.assertRaises(ManifestError):
            validate_manifest(seal_manifest(value))
        value = self.draft(certification_evidence_refs=["z", "a"])
        value["manifest_type"] = "released"
        with self.assertRaises(ManifestError):
            validate_manifest(seal_manifest(value))
        value = self.draft()
        value["schema_version"] = True
        with self.assertRaises(ManifestError):
            validate_manifest(seal_manifest(value))
        publication = dict(self.chain()[-1])
        publication.pop("manifest_id")
        publication.pop("publication_id")
        publication["github_release_url"] = (
            "https://user@github.com/cyaxiverse/CYAxiverse.jl/releases/tag/v1.2.3"
        )
        with self.assertRaisesRegex(ManifestError, "sanitized HTTPS URL"):
            validate_manifest(seal_manifest(publication))

    def test_schema_rejects_missing_transition_identity_and_unsafe_public_values(self) -> None:
        intent = next(
            item for item in self.chain()
            if item["manifest_type"] == "release-intent-prepared"
        )
        missing = dict(intent)
        missing.pop("manifest_id")
        missing.pop("anchor_tree")
        with self.assertRaisesRegex(ManifestError, "missing required fields"):
            validate_manifest(seal_manifest(missing))

        candidate = next(
            item for item in self.chain() if item["manifest_type"] == "candidate-opened"
        )
        bad_candidate = dict(candidate)
        bad_candidate.pop("manifest_id")
        bad_candidate["candidate_id"] = "candidate bad~ref"
        with self.assertRaisesRegex(ManifestError, "candidate_id is not canonical"):
            validate_manifest(seal_manifest(bad_candidate))

        unsafe = dict(intent)
        unsafe.pop("manifest_id")
        unsafe["certification_environment"] = "/Users/private/workstation"
        with self.assertRaisesRegex(ManifestError, "nonpublic value"):
            validate_manifest(seal_manifest(unsafe))

        unsafe = dict(intent)
        unsafe.pop("manifest_id")
        unsafe["certification_evidence_refs"] = ["https://user:secret@example.com/evidence"]
        with self.assertRaisesRegex(ManifestError, "nonpublic value"):
            validate_manifest(seal_manifest(unsafe))

    def test_duplicate_key_raw_json_and_one_file_commit_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for key, value in (("user.name", "Fixture"), ("user.email", "fixture@example.invalid")):
                subprocess.run(["git", "-C", str(root), "config", key, value], check=True)
            raw = b'{"manifest_id":"x","manifest_id":"y"}'
            blob = subprocess.check_output(["git", "-C", str(root), "hash-object", "-w", "--stdin"], input=raw).decode().strip()
            tree = subprocess.check_output(["git", "-C", str(root), "mktree"], input=f"100644 blob {blob}\tmanifest.json\n".encode()).decode().strip()
            commit = subprocess.check_output(["git", "-C", str(root), "commit-tree", tree], input=b"fixture\n").decode().strip()
            with self.assertRaises(ManifestError):
                import version_lifecycle.manifests as module
                module._manifest_from_commit(root, commit)
            manifest = seal_manifest(self.draft())
            commit = make_manifest_commit(root, manifest)
            self.assertEqual(subprocess.check_output(["git", "-C", str(root), "ls-tree", "--name-only", commit], text=True).splitlines(), ["manifest.json"])

    def test_complete_graph_replay_and_terminal_reversal(self) -> None:
        chain = self.chain()
        records = {lifecycle_ref_for_manifest(item): item for item in chain}
        graph = validate_complete_lifecycle_refs(records)
        self.assertEqual(graph.occupied_versions, ("1.2.3",))
        bad = list(chain)
        candidate = next(item for item in bad if item["manifest_type"] == "candidate-opened")
        withdrawn_draft = {
            "schema_version": 1, "manifest_type": "candidate-withdrawn",
            "timestamp_utc": "2026-09-22T00:20:00Z", "predecessor_refs": [lifecycle_ref_for_manifest(candidate)],
            **authorization_fields(), "transaction_id": "tx-chain",
            "static_iteration_snapshot": "a" * 64, "lifecycle_ref_snapshot": "b" * 64,
            "final_version": "1.2.3", "release_line": "principal", "candidate_id": "candidate-1",
            "candidate_ref": candidate["candidate_ref"],
            "withdrawal_evidence": {
                "public_tag_ref": "refs/tags/v1.2.3", "tag_absent": True,
            },
        }
        withdrawn_draft.pop("manifest_id", None)
        withdrawn = seal_manifest(withdrawn_draft)
        reopened_draft = {
            **candidate, "candidate_id": "candidate-2", "candidate_ref": "refs/heads/candidates/v1.2.3/candidate-2",
            "predecessor_refs": [lifecycle_ref_for_manifest(withdrawn)],
        }
        reopened_draft.pop("manifest_id", None)
        reopened = seal_manifest(reopened_draft)
        with self.assertRaisesRegex(ManifestError, "invalid for transition"):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(item): item
                for item in [*bad[:5], withdrawn, reopened]
            })

    def test_graph_rejects_orphan_candidate_and_certification_substitution(self) -> None:
        chain = self.chain()
        candidate = dict(next(
            item for item in chain if item["manifest_type"] == "candidate-opened"
        ))
        candidate.pop("manifest_id")
        candidate["predecessor_refs"] = []
        orphan = seal_manifest(candidate)
        with self.assertRaisesRegex(ManifestError, "predecessor cardinality"):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(orphan): orphan
            })

        intent = next(
            item for item in chain
            if item["manifest_type"] == "release-intent-prepared"
        )
        release = dict(next(
            item for item in chain if item["manifest_type"] == "released"
        ))
        release.pop("manifest_id")
        release["certification_policy_revision"] = "substituted-policy"
        substituted = seal_manifest(release)
        prefix = [
            item for item in chain
            if item["manifest_type"] not in {"released", "publication"}
        ]
        with self.assertRaisesRegex(
            ManifestError, "certification_policy_revision"
        ):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(item): item
                for item in [*prefix, substituted]
            })
        self.assertEqual(
            substituted["predecessor_refs"],
            [lifecycle_ref_for_manifest(intent)],
        )

    def test_pre_entry_abort_releases_version_and_opened_abort_is_invalid(self) -> None:
        common = {
            "schema_version": 1,
            "timestamp_utc": "2026-09-22T00:00:00Z",
            **authorization_fields(),
            "transaction_id": "tx-abort",
            "static_iteration_snapshot": "a" * 64,
            "lifecycle_ref_snapshot": "b" * 64,
            "owner_line": "principal",
            "final_version": "1.2.3",
            "reserved_final": "1.2.3",
            "intended_dev_version": "1.2.3-DEV",
        }
        prepared = seal_manifest({
            **common,
            "manifest_type": "reservation-prepared",
            "predecessor_refs": [],
            "expected_line_head": "1" * 40,
        })
        aborted = seal_manifest({
            **common,
            "manifest_type": "reservation-aborted",
            "predecessor_refs": [lifecycle_ref_for_manifest(prepared)],
            "non_entry_evidence": {"verified": True},
            "abort_reason": "definite-non-entry",
        })
        graph = validate_complete_lifecycle_refs({
            lifecycle_ref_for_manifest(item): item for item in (prepared, aborted)
        })
        self.assertNotIn("1.2.3", graph.occupied_versions)

        opened = seal_manifest({
            **common,
            "manifest_type": "reservation-opened",
            "predecessor_refs": [lifecycle_ref_for_manifest(prepared)],
            "actual_dev_head": "2" * 40,
        })
        invalid_abort = dict(aborted)
        invalid_abort.pop("manifest_id")
        invalid_abort["predecessor_refs"] = [lifecycle_ref_for_manifest(opened)]
        invalid_abort = seal_manifest(invalid_abort)
        with self.assertRaisesRegex(ManifestError, "invalid for transition"):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(item): item
                for item in (prepared, opened, invalid_abort)
            })

    def test_maintenance_successor_uses_bound_lowest_available_patch(self) -> None:
        common = {
            "schema_version": 1,
            "timestamp_utc": "2026-09-22T00:00:00Z",
            **authorization_fields(),
            "transaction_id": "tx-maintenance",
            "static_iteration_snapshot": "a" * 64,
            "lifecycle_ref_snapshot": "b" * 64,
            "owner_line": "maintenance/1.2",
        }
        prepared = seal_manifest({
            **common,
            "manifest_type": "reservation-prepared",
            "predecessor_refs": [],
            "final_version": "1.2.3",
            "reserved_final": "1.2.3",
            "intended_dev_version": "1.2.3-DEV",
            "expected_line_head": "1" * 40,
        })
        opened = seal_manifest({
            **common,
            "manifest_type": "reservation-opened",
            "predecessor_refs": [lifecycle_ref_for_manifest(prepared)],
            "final_version": "1.2.3",
            "reserved_final": "1.2.3",
            "intended_dev_version": "1.2.3-DEV",
            "actual_dev_head": "2" * 40,
        })
        consumed = seal_manifest({
            **common,
            "manifest_type": "reservation-consumed",
            "predecessor_refs": [lifecycle_ref_for_manifest(opened)],
            "final_version": "1.2.3",
            "reserved_final": "1.2.3",
            "closed_final_version": "1.2.3",
            "terminal_disposition": "closed",
            "closure_anchor_ref": "refs/tags/iterations/1.2.3",
        })
        successor = seal_manifest({
            **common,
            "manifest_type": "reservation-prepared",
            "predecessor_refs": [lifecycle_ref_for_manifest(consumed)],
            "final_version": "1.2.5",
            "reserved_final": "1.2.5",
            "intended_dev_version": "1.2.5-DEV",
            "expected_line_head": "3" * 40,
        })
        records = {
            lifecycle_ref_for_manifest(item): item
            for item in (prepared, opened, consumed, successor)
        }
        allocation_key = (
            successor["static_iteration_snapshot"],
            successor["lifecycle_ref_snapshot"],
        )
        graph = validate_complete_lifecycle_refs(
            records,
            allocation_occupied_by_snapshot={allocation_key: {"1.2.4"}},
        )
        self.assertIn("1.2.5", graph.occupied_versions)
        with self.assertRaisesRegex(ManifestError, "allocation view"):
            validate_complete_lifecycle_refs(records)

    def test_graph_rejects_multiple_active_reservations_for_one_line(self) -> None:
        first = seal_manifest(self.prepared_draft())
        second = seal_manifest(self.prepared_draft(
            timestamp_utc="2026-09-22T00:01:00Z",
            final_version="1.2.4",
            reserved_final="1.2.4",
            intended_dev_version="1.2.4-DEV",
        ))
        with self.assertRaisesRegex(ManifestError, "multiple active reservations"):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(item): item for item in (first, second)
            })

    def test_graph_keeps_claimed_reservation_active_until_consumed(self) -> None:
        prepared, opened, claim = self.chain()[:3]
        second = seal_manifest(self.prepared_draft(
            timestamp_utc="2026-09-22T00:01:00Z",
            final_version="1.2.4",
            reserved_final="1.2.4",
            intended_dev_version="1.2.4-DEV",
        ))
        with self.assertRaisesRegex(ManifestError, "multiple active reservations"):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(item): item
                for item in (prepared, opened, claim, second)
            })

    def test_graph_rejects_cross_line_version_ownership(self) -> None:
        principal = seal_manifest(self.prepared_draft())
        maintenance = seal_manifest(self.prepared_draft(
            timestamp_utc="2026-09-22T00:01:00Z",
            owner_line="maintenance/1.2",
        ))
        with self.assertRaisesRegex(ManifestError, "multiple active owner lines"):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(item): item
                for item in (principal, maintenance)
            })

    def test_graph_rejects_reservation_for_permanently_occupied_version(self) -> None:
        chain = self.chain()
        competing = seal_manifest(self.prepared_draft(
            timestamp_utc="2026-09-22T00:30:00Z",
            owner_line="maintenance/1.2",
        ))
        for history in (chain[:4], chain):
            with self.subTest(history=history[-1]["manifest_type"]), self.assertRaisesRegex(
                ManifestError, "permanently occupied"
            ):
                validate_complete_lifecycle_refs({
                    lifecycle_ref_for_manifest(item): item
                    for item in [*history, competing]
                })

    def test_candidate_must_preserve_claim_owner_line(self) -> None:
        chain = self.chain()
        candidate = dict(chain[4])
        candidate.pop("manifest_id")
        candidate["release_line"] = "maintenance/1.2"
        candidate = seal_manifest(candidate)
        with self.assertRaisesRegex(ManifestError, "changes claim owner line"):
            validate_complete_lifecycle_refs({
                lifecycle_ref_for_manifest(item): item
                for item in [*chain[:4], candidate]
            })

    def test_withdrawal_requires_exact_proof_and_remote_tag_absence(self) -> None:
        candidate = next(
            item for item in self.chain()
            if item["manifest_type"] == "candidate-opened"
        )
        draft = {
            "schema_version": 1,
            "manifest_type": "candidate-withdrawn",
            "timestamp_utc": "2026-09-22T00:20:00Z",
            "predecessor_refs": [lifecycle_ref_for_manifest(candidate)],
            **authorization_fields(),
            "transaction_id": "tx-chain",
            "static_iteration_snapshot": "a" * 64,
            "lifecycle_ref_snapshot": "b" * 64,
            "final_version": "1.2.3",
            "release_line": "principal",
            "candidate_id": "candidate-1",
            "candidate_ref": candidate["candidate_ref"],
            "withdrawal_evidence": {
                "public_tag_ref": "refs/tags/v1.2.3",
                "tag_absent": True,
            },
        }
        withdrawn = seal_manifest(draft)
        invalid = dict(draft)
        invalid["withdrawal_evidence"] = {"tag_absent": True}
        with self.assertRaisesRegex(ManifestError, "exact absent canonical"):
            validate_manifest(seal_manifest(invalid))

        import version_lifecycle.manifests as module
        writer = object.__new__(CreateOnlyLifecycleWriter)
        writer.repository = Path(".")
        writer.remote = "origin"
        advertisement = (
            f"{'f' * 40}\trefs/tags/v1.2.3\n".encode("ascii")
        )
        with patch.object(module, "_git", return_value=advertisement):
            with self.assertRaisesRegex(ManifestConflict, "PUBLIC_TAG_ALREADY_EXISTS"):
                writer._require_no_public_tag(withdrawn)
        with patch.object(module, "_git", return_value=b""):
            writer._require_no_public_tag(withdrawn)

    def test_publication_must_match_released_manifest_exactly(self) -> None:
        chain = self.chain()

        def records_with_publication(**changes: object) -> dict[str, dict[str, object]]:
            publication = dict(chain[-1])
            publication.pop("manifest_id")
            publication.pop("publication_id")
            publication.update(changes)
            if "public_tag" in changes and "github_release_url" not in changes:
                publication["github_release_url"] = (
                    "https://github.com/cyaxiverse/CYAxiverse.jl/releases/tag/"
                    + str(publication["public_tag"])
                )
            publication = seal_manifest(publication)
            values = [*chain[:-1], publication]
            return {lifecycle_ref_for_manifest(item): item for item in values}

        with self.assertRaisesRegex(ManifestError, "released manifest ID mismatch"):
            validate_complete_lifecycle_refs(
                records_with_publication(released_manifest_id="LIF-SHA256-" + "0" * 64)
            )
        with self.assertRaisesRegex(ManifestError, "released manifest digest mismatch"):
            validate_complete_lifecycle_refs(
                records_with_publication(released_manifest_digest="0" * 64)
            )
        with self.assertRaisesRegex(ManifestError, "public tag mismatch"):
            validate_complete_lifecycle_refs(
                records_with_publication(public_tag="v1.2.4")
            )
        with self.assertRaisesRegex(ManifestError, "release commit mismatch"):
            validate_complete_lifecycle_refs(
                records_with_publication(tag_commit="0" * 40)
            )
        with self.assertRaisesRegex(ManifestError, "release tree mismatch"):
            validate_complete_lifecycle_refs(
                records_with_publication(tag_tree="0" * 40)
            )

    def test_snapshot_occupied_set_must_match_replayed_graph(self) -> None:
        chain = self.chain()
        records = []
        for item in chain:
            raw = canonical_manifest_bytes(item)
            records.append({
                "ref": lifecycle_ref_for_manifest(item), "manifest": item,
                "commit": "a" * 40, "tree": "b" * 40,
                "manifest_id": item["manifest_id"], "manifest_digest": hashlib.sha256(raw).hexdigest(),
            })
        graph = validate_complete_lifecycle_refs(records)
        bindings = sorted([{key: entry[key] for key in ("ref", "commit", "tree", "manifest_id", "manifest_digest")} for entry in records], key=lambda item: item["ref"])
        snapshot = {
            "snapshot_schema_version": 1, "source_repository": "fixture",
            "lifecycle_ref_bindings": bindings,
            "lifecycle_ref_set_digest": hashlib.sha256(json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "occupied_versions": list(graph.occupied_versions),
        }
        snapshot["lifecycle_snapshot_digest"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(validate_lifecycle_ref_snapshot(snapshot, graph_records=records).occupied_versions, ("1.2.3",))
        tampered = dict(snapshot, occupied_versions=["9.9.9"])
        tampered["lifecycle_snapshot_digest"] = hashlib.sha256(json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.assertRaisesRegex(ManifestError, "occupied versions"):
            validate_lifecycle_ref_snapshot(tampered, graph_records=records)

    def test_create_only_writer_replays_one_file_commit_and_idempotent_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "origin.git"
            work = root / "work"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            subprocess.run(["git", "init", "-q", str(work)], check=True)
            for key, value in (("user.name", "Fixture"), ("user.email", "fixture@example.invalid")):
                subprocess.run(["git", "-C", str(work), "config", key, value], check=True)
            subprocess.run(["git", "-C", str(work), "remote", "add", "origin", str(remote)], check=True)
            (work / "base").write_text("base", encoding="utf-8")
            subprocess.run(["git", "-C", str(work), "add", "base"], check=True)
            subprocess.run(["git", "-C", str(work), "commit", "-qm", "base"], check=True)
            root_parent = subprocess.check_output(
                ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
            ).strip()
            authority = ManifestAuthority()
            manifest = authority.bind(self.prepared_draft())
            protection = ProtectionEvidence("fixture", "refs/heads/lifecycle/v1/*", "0" * 64, "2026-09-22T00:00:00Z", True, True, True)
            snapshot = remote_writer_snapshot(work)
            writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: nullcontext(True),
                snapshot_callback=lambda: remote_writer_snapshot(work),
                expected_snapshot=snapshot,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=root_parent,
            )
            first = writer.create(manifest, protection, authorization_context=WRITER_CONTEXT)
            second = writer.create(manifest, protection, authorization_context=WRITER_CONTEXT)
            retry_writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: nullcontext(True),
                snapshot_callback=lambda: remote_writer_snapshot(work),
                expected_snapshot=snapshot,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=root_parent,
            )
            retry = retry_writer.create(
                manifest, protection, authorization_context=WRITER_CONTEXT
            )
            port = CreateOnlyLifecycleManifestPort(
                writer, protection, lambda _kind, _manifest: WRITER_CONTEXT
            )
            self.assertEqual(
                port.create_manifest("reservation-prepared", manifest), manifest
            )
            opened_draft = {
                **self.prepared_draft(),
                "manifest_type": "reservation-opened",
                "predecessor_refs": [lifecycle_ref_for_manifest(manifest)],
                "actual_dev_head": "b" * 40,
            }
            opened_draft.pop("expected_line_head")
            opened = authority.bind(opened_draft)
            successor = writer.create(
                opened, protection, authorization_context=WRITER_CONTEXT
            )
            self.assertEqual(first.status, "CREATED")
            self.assertEqual(second.status, "IDEMPOTENT")
            self.assertEqual(retry.status, "IDEMPOTENT")
            self.assertEqual(successor.status, "CREATED")
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(work), "rev-parse", f"{first.commit}^"],
                    text=True,
                ).strip(),
                root_parent,
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(work), "rev-parse", f"{successor.commit}^"],
                    text=True,
                ).strip(),
                first.commit,
            )
            entries = subprocess.check_output(["git", "-C", str(work), "ls-tree", "--name-only", first.tree], text=True).splitlines()
            self.assertEqual(entries, ["manifest.json"])
            ref = first.ref
            self.assertEqual(subprocess.check_output(["git", "-C", str(work), "ls-remote", "origin", ref], text=True).split("\t")[1].strip(), ref)
            conflicting = authority.bind(self.prepared_draft(timestamp_utc="2026-09-22T00:00:01Z"))
            with self.assertRaises(ManifestError):
                writer.create(conflicting, protection, authorization_context=WRITER_CONTEXT)

    def test_writer_rejects_authorization_scope_mismatch_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "origin.git"
            work = root / "work"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            subprocess.run(["git", "init", "-q", str(work)], check=True)
            for key, value in (
                ("user.name", "Fixture"),
                ("user.email", "fixture@example.invalid"),
            ):
                subprocess.run(
                    ["git", "-C", str(work), "config", key, value], check=True
                )
            subprocess.run(
                ["git", "-C", str(work), "remote", "add", "origin", str(remote)],
                check=True,
            )
            (work / "base").write_text("base", encoding="utf-8")
            subprocess.run(["git", "-C", str(work), "add", "base"], check=True)
            subprocess.run(
                ["git", "-C", str(work), "commit", "-qm", "base"], check=True
            )
            root_parent = subprocess.check_output(
                ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
            ).strip()
            authority = ManifestAuthority()
            manifest = authority.bind(self.prepared_draft())
            protection = ProtectionEvidence(
                "fixture", "refs/heads/lifecycle/v1/*", "0" * 64,
                "2026-09-22T00:00:00Z", True, True, True,
            )
            snapshot = remote_writer_snapshot(work)
            writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: nullcontext(True),
                snapshot_callback=lambda: remote_writer_snapshot(work),
                expected_snapshot=snapshot,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=root_parent,
            )
            mismatches = (
                {**WRITER_CONTEXT, "transaction_id": "tx-other"},
                {**WRITER_CONTEXT, "owner_line": "maintenance/1.2"},
                {**WRITER_CONTEXT, "final_version": "1.2.4"},
            )
            for context in mismatches:
                with self.subTest(context=context), self.assertRaisesRegex(
                    ManifestError, "OWNER_AUTHORIZATION_UNVERIFIED"
                ):
                    writer.create(
                        manifest, protection, authorization_context=context
                    )
                self.assertEqual(
                    subprocess.check_output(
                        ["git", "-C", str(work), "ls-remote", "--heads", "origin"],
                        text=True,
                    ),
                    "",
                )
            wrong_action_manifest = authority.bind(
                self.prepared_draft(), action="create-tag"
            )
            with self.assertRaisesRegex(
                ManifestError, "OWNER_AUTHORIZATION_UNVERIFIED"
            ):
                writer.create(
                    wrong_action_manifest,
                    protection,
                    authorization_context={**WRITER_CONTEXT, "action": "create-tag"},
                )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(work), "ls-remote", "--heads", "origin"],
                    text=True,
                ),
                "",
            )

    def test_writer_rejects_static_reservation_collision_without_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "origin.git"
            work = root / "work"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            subprocess.run(["git", "init", "-q", str(work)], check=True)
            for key, value in (
                ("user.name", "Fixture"),
                ("user.email", "fixture@example.invalid"),
            ):
                subprocess.run(
                    ["git", "-C", str(work), "config", key, value], check=True
                )
            subprocess.run(
                ["git", "-C", str(work), "remote", "add", "origin", str(remote)],
                check=True,
            )
            (work / "base").write_text("base", encoding="utf-8")
            subprocess.run(["git", "-C", str(work), "add", "base"], check=True)
            subprocess.run(
                ["git", "-C", str(work), "commit", "-qm", "base"], check=True
            )
            source_commit = subprocess.check_output(
                ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
            ).strip()
            authority = ManifestAuthority()
            manifest = authority.bind(self.prepared_draft())
            protection = ProtectionEvidence(
                "fixture", "refs/heads/lifecycle/v1/*", "0" * 64,
                "2026-09-22T00:00:00Z", True, True, True,
            )
            snapshot = writer_snapshot(
                static_occupied=["1.2.3"], source_commit=source_commit
            )
            writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: nullcontext(True),
                snapshot_callback=lambda: snapshot,
                expected_snapshot=snapshot,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=source_commit,
            )
            with self.assertRaisesRegex(
                ManifestError, "RESERVATION_VERSION_UNAVAILABLE"
            ):
                writer.create(
                    manifest, protection, authorization_context=WRITER_CONTEXT
                )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(work), "ls-remote", "--heads", "origin"],
                    text=True,
                ),
                "",
            )

    def test_publication_authorization_scope_comes_from_released_predecessor(self) -> None:
        chain = self.chain()
        records = {
            lifecycle_ref_for_manifest(item): {"manifest": item}
            for item in chain[:-1]
        }
        self.assertEqual(
            CreateOnlyLifecycleWriter._authorization_scope(chain[-1], records),
            ("tx-chain", "principal", "1.2.3"),
        )

    def test_authoritative_reader_rejects_invalid_commit_parent_topology(self) -> None:
        for topology in ("rootless", "wrong-parent", "multi-parent"):
            with self.subTest(topology=topology), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                remote = root / "origin.git"
                work = root / "work"
                subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
                subprocess.run(["git", "init", "-q", str(work)], check=True)
                for key, value in (
                    ("user.name", "Fixture"),
                    ("user.email", "fixture@example.invalid"),
                ):
                    subprocess.run(
                        ["git", "-C", str(work), "config", key, value],
                        check=True,
                    )
                subprocess.run(
                    ["git", "-C", str(work), "remote", "add", "origin", str(remote)],
                    check=True,
                )
                (work / "base").write_text("base", encoding="utf-8")
                subprocess.run(["git", "-C", str(work), "add", "base"], check=True)
                subprocess.run(
                    ["git", "-C", str(work), "commit", "-qm", "base"],
                    check=True,
                )
                root_parent = subprocess.check_output(
                    ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
                ).strip()
                wrong_parent = subprocess.check_output(
                    ["git", "-C", str(work), "commit-tree", f"{root_parent}^{{tree}}"],
                    input="wrong parent\n",
                    text=True,
                ).strip()
                manifest = seal_manifest(self.prepared_draft())
                if topology == "rootless":
                    commit = make_manifest_commit(work, manifest)
                elif topology == "wrong-parent":
                    commit = make_manifest_commit(
                        work, manifest, parent=wrong_parent
                    )
                else:
                    single = make_manifest_commit(
                        work, manifest, parent=root_parent
                    )
                    tree = subprocess.check_output(
                        ["git", "-C", str(work), "rev-parse", f"{single}^{{tree}}"],
                        text=True,
                    ).strip()
                    commit = subprocess.check_output(
                        [
                            "git", "-C", str(work), "commit-tree", tree,
                            "-p", root_parent, "-p", wrong_parent,
                        ],
                        input="multi parent\n",
                        text=True,
                    ).strip()
                ref = lifecycle_ref_for_manifest(manifest)
                subprocess.run(
                    ["git", "-C", str(work), "push", "-q", "origin", f"{commit}:{ref}"],
                    check=True,
                )
                with self.assertRaisesRegex(
                    ManifestError, "LIFECYCLE_PARENT_UNVERIFIED"
                ):
                    build_lifecycle_ref_snapshot(
                        work,
                        source_repository="fixture-repository",
                        static_source_commit=root_parent,
                    )

    def test_lifecycle_genesis_survives_static_source_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "origin.git"
            work = root / "work"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            subprocess.run(["git", "init", "-q", str(work)], check=True)
            for key, value in (
                ("user.name", "Fixture"),
                ("user.email", "fixture@example.invalid"),
            ):
                subprocess.run(
                    ["git", "-C", str(work), "config", key, value], check=True
                )
            subprocess.run(
                ["git", "-C", str(work), "remote", "add", "origin", str(remote)],
                check=True,
            )
            (work / "base").write_text("base", encoding="utf-8")
            subprocess.run(["git", "-C", str(work), "add", "base"], check=True)
            subprocess.run(
                ["git", "-C", str(work), "commit", "-qm", "base"], check=True
            )
            lifecycle_genesis = subprocess.check_output(
                ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
            ).strip()
            manifest = seal_manifest(self.prepared_draft())
            commit = make_manifest_commit(
                work, manifest, parent=lifecycle_genesis
            )
            ref = lifecycle_ref_for_manifest(manifest)
            subprocess.run(
                ["git", "-C", str(work), "push", "-q", "origin", f"{commit}:{ref}"],
                check=True,
            )
            (work / "base").write_text("advanced", encoding="utf-8")
            subprocess.run(["git", "-C", str(work), "add", "base"], check=True)
            subprocess.run(
                ["git", "-C", str(work), "commit", "-qm", "advance vmm"],
                check=True,
            )
            current_static = subprocess.check_output(
                ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
            ).strip()
            subprocess.run(
                [
                    "git", "-C", str(work), "push", "-q", "origin",
                    "HEAD:refs/heads/vmm",
                ],
                check=True,
            )
            snapshot = build_lifecycle_ref_snapshot(
                work,
                source_repository="fixture-repository",
                static_source_commit=current_static,
            )
            self.assertEqual(len(snapshot.ref_bindings), 1)

            combined = remote_writer_snapshot(work)
            authority = ManifestAuthority()
            writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: nullcontext(True),
                snapshot_callback=lambda: remote_writer_snapshot(work),
                expected_snapshot=combined,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=current_static,
            )
            self.assertEqual(len(writer._complete_graph_records()), 1)
            self.assertEqual(
                writer.lifecycle_genesis_commit, lifecycle_genesis
            )

    def test_create_revalidates_snapshot_inside_exclusion_and_blocks_uncertain_create(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "origin.git"
            work = root / "work"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            subprocess.run(["git", "init", "-q", str(work)], check=True)
            for key, value in (("user.name", "Fixture"), ("user.email", "fixture@example.invalid")):
                subprocess.run(["git", "-C", str(work), "config", key, value], check=True)
            subprocess.run(["git", "-C", str(work), "remote", "add", "origin", str(remote)], check=True)
            (work / "base").write_text("base", encoding="utf-8")
            subprocess.run(["git", "-C", str(work), "add", "base"], check=True)
            subprocess.run(["git", "-C", str(work), "commit", "-qm", "base"], check=True)
            root_parent = subprocess.check_output(
                ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
            ).strip()
            authority = ManifestAuthority()
            manifest = authority.bind(self.prepared_draft())
            protection = ProtectionEvidence("fixture", "refs/heads/lifecycle/v1/*", "0" * 64, "2026-09-22T00:00:00Z", True, True, True)
            held = {"value": False}
            class Guard:
                def __enter__(self) -> bool:
                    held["value"] = True
                    return True
                def __exit__(self, *_args: object) -> None:
                    held["value"] = False
            first_snapshot = writer_snapshot(source_commit=root_parent)
            second_snapshot = writer_snapshot(
                ["9.9.9"], source_commit=root_parent
            )
            values = iter((first_snapshot, second_snapshot))
            def callback() -> object:
                self.assertTrue(held["value"])
                return next(values)
            writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: Guard(),
                snapshot_callback=callback,
                expected_snapshot=first_snapshot,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=root_parent,
            )
            with self.assertRaisesRegex(ManifestError, "SNAPSHOT_STALE"):
                writer.create(manifest, protection, authorization_context=WRITER_CONTEXT)
            self.assertFalse(held["value"])

            original_fetch = authority.fetch_owner_authorization
            authority.fetch_owner_authorization = lambda reference: (
                lambda resolution: AuthorizationResolution(
                    resolution.record,
                    resolution.canonical_bytes + b" ",
                    resolution.repository_owner_verified,
                )
            )(original_fetch(reference))
            writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: nullcontext(True),
                snapshot_callback=lambda: first_snapshot,
                expected_snapshot=first_snapshot,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=root_parent,
            )
            with self.assertRaisesRegex(Exception, "bytes changed"):
                writer.create(
                    manifest, protection, authorization_context=WRITER_CONTEXT
                )
            authority.fetch_owner_authorization = original_fetch

            manifest = authority.bind(self.prepared_draft(timestamp_utc="2026-09-22T00:01:00Z"))
            writer = CreateOnlyLifecycleWriter(
                work,
                exclusion_lease=lambda: nullcontext(True),
                snapshot_callback=lambda: first_snapshot,
                expected_snapshot=first_snapshot,
                owner_authorization_authority=authority,
                authorization_reference=authority.reference,
                authorization_clock=lambda: "2026-09-22T00:00:00Z",
                repository_identity="fixture-repository",
                static_source_commit=root_parent,
            )
            import version_lifecycle.manifests as module
            real_git = module._git
            def fail_push(repository: Path, *args: str, input_bytes: bytes | None = None, check: bool = True) -> bytes:
                if args and args[0] == "push":
                    raise ManifestError("transport unavailable")
                return real_git(repository, *args, input_bytes=input_bytes, check=check)
            with patch.object(module, "_git", side_effect=fail_push):
                with self.assertRaisesRegex(Exception, "CREATE_ONCE_OUTCOME_UNCERTAIN"):
                    writer.create(manifest, protection, authorization_context=WRITER_CONTEXT)


if __name__ == "__main__":
    unittest.main()
