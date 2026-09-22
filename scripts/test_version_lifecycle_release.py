"""Focused immutable release/publication consistency tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.codec import sha256_hex  # noqa: E402
from version_lifecycle.manifests import (  # noqa: E402
    canonical_manifest_bytes,
    lifecycle_ref_for_manifest,
    seal_manifest,
)
import test_version_lifecycle_manifests as manifest_tests  # noqa: E402
from version_lifecycle.release import (  # noqa: E402
    BLOCKED,
    PUBLICATION_RECONCILIATION_PENDING,
    TERMINAL_CONSISTENT,
    validate_publication_evidence,
    validate_release_consistency,
    validate_released_manifest,
)


SHA = "a" * 40
OTHER_SHA = "c" * 40
TREE = "b" * 40
OTHER_TREE = "d" * 40
OWNER_AUTHORIZATION = "AUTH-SHA256-" + "7" * 64
OWNER_AUTHORIZATION_REF = "owner-authority://synthetic/grant-release-tests"
OWNER_AUTHORIZATION_DIGEST = "7" * 64


def released(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": "released",
        "timestamp_utc": "2026-09-22T00:00:00Z",
        "predecessor_refs": [
            "refs/heads/lifecycle/v1/intents/v1.2.3/candidate-1/"
            + "LIF-SHA256-"
            + "c" * 64
        ],
        "owner_authorization": OWNER_AUTHORIZATION,
        "owner_authorization_ref": OWNER_AUTHORIZATION_REF,
        "owner_authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "transaction_id": "release-tx",
        "static_iteration_snapshot": "d" * 64,
        "lifecycle_ref_snapshot": "e" * 64,
        "final_version": "1.2.3",
        "release_line": "principal",
        "public_tag": "v1.2.3",
        "candidate_ref": "refs/heads/lifecycle/v1/candidates/v1.2.3/candidate-1",
        "candidate_sha": SHA,
        "candidate_tree": TREE,
        "anchor_ref": "refs/tags/iterations/1.2.3",
        "anchor_sha": SHA,
        "anchor_tree": TREE,
        "final_release_sha": SHA,
        "final_release_tree": TREE,
        "certification_binding": "tree-bound",
        "certification_subject_sha": SHA,
        "certification_subject_tree": TREE,
        "certification_policy_revision": "policy-r1",
        "certification_harness_revision": "harness-r1",
        "certification_environment": "ci-linux",
        "certification_evidence_refs": ["evidence/certification-r1"],
        "closure_timestamp_utc": "2026-09-21T00:00:00Z",
        "previous_main_sha": OTHER_SHA,
        "previous_main_version": "1.2.2",
        "main_at_candidate_sha": OTHER_SHA,
        "main_at_candidate_version": "1.2.2",
        "main_at_release_sha": SHA,
        "main_at_release_version": "1.2.3",
        "evidence_refs": ["evidence/release-r1"],
    }
    value.update(overrides)
    value = {key: item for key, item in value.items() if item is not None}
    return seal_manifest(value)


def publication(release: dict[str, object], **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "manifest_type": "publication",
        "timestamp_utc": "2026-09-22T01:00:00Z",
        "predecessor_refs": ["refs/heads/lifecycle/v1/releases/v1.2.3"],
        "owner_authorization": OWNER_AUTHORIZATION,
        "owner_authorization_ref": OWNER_AUTHORIZATION_REF,
        "owner_authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "released_manifest_ref": "refs/heads/lifecycle/v1/releases/v1.2.3",
        "released_manifest_id": release["manifest_id"],
        "released_manifest_digest": sha256_hex(canonical_manifest_bytes(release)),
        "public_tag": "v1.2.3",
        "tag_commit": SHA,
        "tag_tree": TREE,
        "github_release_id": 9001,
        "github_release_url": "https://github.com/cyaxiverse/CYAxiverse.jl/releases/tag/v1.2.3",
        "published_at_utc": "2026-09-22T01:00:00Z",
        "publication_evidence_ref": "evidence/publication-r1",
        "publication_evidence_digest": "1" * 64,
    }
    value.update(overrides)
    return seal_manifest(value)


class ReleaseTests(unittest.TestCase):
    @staticmethod
    def complete_context() -> tuple[
        dict[str, object], dict[str, object], dict[str, object]
    ]:
        chain = manifest_tests.ManifestTests().chain()
        release = next(item for item in chain if item["manifest_type"] == "released")
        publication_manifest = next(
            item for item in chain if item["manifest_type"] == "publication"
        )
        records = {lifecycle_ref_for_manifest(item): item for item in chain}
        kwargs: dict[str, object] = {
            "public_tag": publication_manifest["public_tag"],
            "tag_commit": publication_manifest["tag_commit"],
            "tag_tree": publication_manifest["tag_tree"],
            "certified_tree": release["anchor_tree"],
            "github_release_id": publication_manifest["github_release_id"],
            "publication_evidence_digest": publication_manifest[
                "publication_evidence_digest"
            ],
            "lifecycle_records": records,
            "anchor_tag_object": release["anchor_sha"],
            "anchor_tree": release["anchor_tree"],
            "anchor_closure_timestamp_utc": release["closure_timestamp_utc"],
            "candidate_ref": release["candidate_ref"],
            "candidate_commit": release["candidate_sha"],
            "candidate_tree": release["candidate_tree"],
            "canonical_tag_observations": [{
                "ref": f"refs/tags/{publication_manifest['public_tag']}",
                "tag": publication_manifest["public_tag"],
                "commit": publication_manifest["tag_commit"],
                "tree": publication_manifest["tag_tree"],
            }],
            "github_release_observations": [{
                "id": publication_manifest["github_release_id"],
                "tag": publication_manifest["public_tag"],
                "url": publication_manifest["github_release_url"],
            }],
            "require_complete_namespace": True,
        }
        return release, publication_manifest, kwargs

    def test_terminal_validation_replays_complete_namespace_and_direct_identities(self) -> None:
        release, publication_manifest, kwargs = self.complete_context()
        result = validate_release_consistency(
            release,
            publication_manifest,
            **kwargs,
        )
        self.assertEqual(result["status"], TERMINAL_CONSISTENT)
        missing = validate_release_consistency(
            release,
            publication_manifest,
            require_complete_namespace=True,
        )
        self.assertEqual(missing["reason_code"], "COMPLETE_RELEASE_UNIVERSE_UNAVAILABLE")
        extra_tags = list(kwargs["canonical_tag_observations"])
        extra_tags.append({
            "ref": "refs/tags/v1.2.4", "tag": "v1.2.4",
            "commit": "f" * 40, "tree": "e" * 40,
        })
        mismatch = validate_release_consistency(
            release,
            publication_manifest,
            **{**kwargs, "canonical_tag_observations": extra_tags},
        )
        self.assertEqual(
            mismatch["reason_code"], "COMPLETE_RELEASE_UNIVERSE_MISMATCH"
        )
        closure_mismatch = validate_release_consistency(
            release,
            publication_manifest,
            **{
                **kwargs,
                "anchor_closure_timestamp_utc": "2026-09-22T00:10:01Z",
            },
        )
        self.assertEqual(
            closure_mismatch["reason_code"], "ANCHOR_IDENTITY_MISMATCH"
        )
        rogue_github_release = list(kwargs["github_release_observations"])
        rogue_github_release.append({
            "id": 2,
            "tag": "nightly",
            "url": "https://github.com/cyaxiverse/CYAxiverse.jl/releases/tag/nightly",
        })
        invalid = validate_release_consistency(
            release,
            publication_manifest,
            **{
                **kwargs,
                "github_release_observations": rogue_github_release,
            },
        )
        self.assertEqual(
            invalid["reason_code"], "COMPLETE_RELEASE_UNIVERSE_INVALID"
        )
        wrong_url = [dict(item) for item in kwargs["github_release_observations"]]
        wrong_url[0]["url"] = (
            "https://github.com/other/project/releases/tag/v1.2.3"
        )
        mismatch = validate_release_consistency(
            release,
            publication_manifest,
            **{**kwargs, "github_release_observations": wrong_url},
        )
        self.assertEqual(mismatch["reason_code"], "GITHUB_RELEASE_URL_MISMATCH")

    def test_checked_in_principal_fixture_is_internally_consistent(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "specs/0125-version-iteration-release/fixtures/synthetic-lifecycle-principal.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        release = fixture["released_manifest"]
        publication_manifest = fixture["publication_manifest"]
        self.assertEqual(validate_released_manifest(release)["status"], "PASS")
        self.assertEqual(
            validate_publication_evidence(publication_manifest, release)["status"],
            "PASS",
        )

    def test_missing_released_manifest_is_blocked(self) -> None:
        self.assertEqual(validate_release_consistency()["status"], BLOCKED)

    def test_missing_publication_is_forward_pending(self) -> None:
        result = validate_release_consistency(released())
        self.assertEqual(result["status"], PUBLICATION_RECONCILIATION_PENDING)

    def test_exact_release_and_publication_are_terminal(self) -> None:
        release, publication_manifest, kwargs = self.complete_context()
        result = validate_release_consistency(
            release,
            publication_manifest,
            **kwargs,
        )
        self.assertEqual(result["status"], TERMINAL_CONSISTENT)

    def test_publication_released_digest_is_verified(self) -> None:
        release = released()
        result = validate_release_consistency(
            release, publication(release, released_manifest_digest="f" * 64)
        )
        self.assertEqual(result["reason_code"], "RELEASED_MANIFEST_DIGEST_MISMATCH")

    def test_publication_evidence_digest_is_verified(self) -> None:
        release = released()
        result = validate_release_consistency(
            release,
            publication(release),
            publication_evidence_digest="2" * 64,
        )
        self.assertEqual(result["reason_code"], "PUBLICATION_EVIDENCE_DIGEST_MISMATCH")

    def test_publication_tag_and_github_identity_mismatches_are_invalid(self) -> None:
        release = released()
        tag = validate_release_consistency(
            release, publication(release), tag_commit=OTHER_SHA
        )
        self.assertEqual(tag["reason_code"], "RELEASE_COMMIT_MISMATCH")
        github = validate_release_consistency(
            release, publication(release), github_release_id=9002
        )
        self.assertEqual(github["reason_code"], "GITHUB_RELEASE_ID_MISMATCH")

    def test_principal_main_must_equal_final_release(self) -> None:
        result = validate_released_manifest(released(main_at_release_sha=OTHER_SHA))
        self.assertEqual(result["reason_code"], "MAIN_RELEASE_COMMIT_MISMATCH")
        result = validate_released_manifest(released(main_at_release_version="1.2.2"))
        self.assertEqual(result["reason_code"], "MAIN_RELEASE_VERSION_MISMATCH")

    def test_maintenance_release_binds_line_and_unchanged_final_main(self) -> None:
        valid = released(
            release_line="maintenance/1.2",
            previous_main_sha=None,
            previous_main_version=None,
            main_at_candidate_sha=OTHER_SHA,
            main_at_candidate_version="2.0.0",
            main_at_release_sha=OTHER_SHA,
            main_at_release_version="2.0.0",
        )
        self.assertEqual(validate_released_manifest(valid)["status"], "PASS")
        for changes in (
            {"release_line": "maintenance/9.9"},
            {"main_at_release_version": "2.0.0-DEV"},
            {"main_at_release_sha": SHA},
        ):
            draft = dict(valid)
            draft.pop("manifest_id")
            draft.update(changes)
            result = validate_released_manifest(seal_manifest(draft))
            self.assertEqual(result["reason_code"], "RELEASED_MANIFEST_INVALID")

    def test_exact_tree_mismatch_is_invalid(self) -> None:
        result = validate_released_manifest(released(final_release_tree=OTHER_TREE))
        self.assertEqual(result["reason_code"], "RELEASE_TREE_MISMATCH")

    def test_unsupported_certification_binding_blocks(self) -> None:
        result = validate_released_manifest(released(certification_binding="future-bound"))
        self.assertEqual(result["reason_code"], "UNSUPPORTED_CERTIFICATION_BINDING")

    def test_commit_bound_different_final_requires_recertification(self) -> None:
        result = validate_released_manifest(
            released(certification_binding="commit-bound", certification_subject_sha=OTHER_SHA)
        )
        self.assertEqual(result["reason_code"], "RECERTIFICATION_REQUIRED")

    def test_tree_bound_transfer_requires_exact_durable_evidence(self) -> None:
        missing = validate_released_manifest(released(certification_subject_sha=OTHER_SHA))
        self.assertEqual(missing["reason_code"], "TREE_BOUND_TRANSFER_EVIDENCE_REQUIRED")
        evidence = {
            "candidate_tree": TREE,
            "certified_tree": TREE,
            "final_release_tree": TREE,
            "anchor_tree": TREE,
            "evidence_ref": "evidence/transfer-r1",
        }
        passed = validate_released_manifest(
            released(
                certification_subject_sha=OTHER_SHA,
                certification_transfer_evidence=evidence,
            )
        )
        self.assertEqual(passed["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
