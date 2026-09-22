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
    ManifestError,
    ManifestConflict,
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
from version_lifecycle.git_refs import ProtectionEvidence  # noqa: E402


OWNER_AUTHORIZATION = "AUTH-SHA256-" + "7" * 64
OWNER_AUTHORIZATION_REF = "owner-authority://synthetic/grant-manifest-tests"
OWNER_AUTHORIZATION_DIGEST = "7" * 64


def authorization_fields() -> dict[str, str]:
    return {
        "owner_authorization": OWNER_AUTHORIZATION,
        "owner_authorization_ref": OWNER_AUTHORIZATION_REF,
        "owner_authorization_digest": OWNER_AUTHORIZATION_DIGEST,
    }


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

    def chain(self) -> list[dict[str, object]]:
        common = {
            "schema_version": 1,
            "timestamp_utc": "2026-09-22T00:00:00Z",
            **authorization_fields(),
            "transaction_id": "tx-chain",
            "static_iteration_snapshot": "a" * 64,
            "lifecycle_ref_snapshot": "b" * 64,
        }
        claim = seal_manifest({
            **common, "manifest_type": "version-claimed", "predecessor_refs": [],
            "final_version": "1.2.3", "owner_line": "principal",
        })
        candidate = seal_manifest({
            **common, "manifest_type": "candidate-opened",
            "predecessor_refs": [lifecycle_ref_for_manifest(claim)],
            "candidate_id": "candidate-1", "candidate_ref": "refs/heads/candidates/v1.2.3/candidate-1",
            "candidate_sha": "c" * 40, "candidate_tree": "d" * 40,
            "final_version": "1.2.3", "release_line": "principal",
            "anchor_ref": "refs/tags/iterations/1.2.3", "anchor_sha": "c" * 40,
            "anchor_tree": "d" * 40,
        })
        intent = seal_manifest({
            **common, "manifest_type": "release-intent-prepared",
            "predecessor_refs": [lifecycle_ref_for_manifest(candidate)],
            "candidate_id": "candidate-1", "candidate_ref": candidate["candidate_ref"],
            "candidate_sha": candidate["candidate_sha"], "candidate_tree": candidate["candidate_tree"],
            "final_version": "1.2.3", "release_line": "principal", "public_tag": "v1.2.3",
            "final_release_sha": "e" * 40, "final_release_tree": "f" * 40,
            "certification_binding": "tree-bound", "certification_subject_sha": "c" * 40,
            "certification_subject_tree": "d" * 40, "certification_policy_revision": "policy-r1",
            "certification_harness_revision": "harness-r1", "certification_environment": "ci",
            "certification_evidence_refs": ["evidence/certification-r1"],
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
            "closure_timestamp_utc": "2026-09-22T00:10:00Z", "previous_main_sha": "1" * 40,
            "previous_main_version": "1.2.2", "main_at_release_sha": "2" * 40,
            "main_at_release_version": "1.2.3",
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
        return [claim, candidate, intent, release, publication]

    def test_content_derived_id_and_exact_wire_bytes(self) -> None:
        value = seal_manifest(self.draft())
        self.assertEqual(value["manifest_id"], manifest_id(value))
        self.assertEqual(canonical_manifest_bytes(value), json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(lifecycle_ref_for_manifest(value), "refs/heads/lifecycle/v1/claims/v1.2.3")

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
        withdrawn_draft = {
            "schema_version": 1, "manifest_type": "candidate-withdrawn",
            "timestamp_utc": "2026-09-22T00:20:00Z", "predecessor_refs": [lifecycle_ref_for_manifest(bad[1])],
            **authorization_fields(), "transaction_id": "tx-chain",
            "static_iteration_snapshot": "a" * 64, "lifecycle_ref_snapshot": "b" * 64,
            "final_version": "1.2.3", "release_line": "principal", "candidate_id": "candidate-1",
            "candidate_ref": bad[1]["candidate_ref"], "withdrawal_evidence": {"reason": "no-tag"},
        }
        withdrawn_draft.pop("manifest_id", None)
        withdrawn = seal_manifest(withdrawn_draft)
        reopened_draft = {
            **bad[1], "candidate_id": "candidate-2", "candidate_ref": "refs/heads/candidates/v1.2.3/candidate-2",
            "predecessor_refs": [lifecycle_ref_for_manifest(withdrawn)],
        }
        reopened_draft.pop("manifest_id", None)
        reopened = seal_manifest(reopened_draft)
        with self.assertRaisesRegex(ManifestError, "terminal-to-active"):
            validate_complete_lifecycle_refs({lifecycle_ref_for_manifest(item): item for item in [bad[0], bad[1], withdrawn, reopened]})

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
            manifest = seal_manifest(self.draft())
            protection = ProtectionEvidence("fixture", "refs/heads/lifecycle/v1/*", "0" * 64, "2026-09-22T00:00:00Z", True, True, True)
            writer = CreateOnlyLifecycleWriter(work)
            first = writer.create(manifest, protection)
            second = writer.create(manifest, protection)
            self.assertEqual(first.status, "CREATED")
            self.assertEqual(second.status, "IDEMPOTENT")
            entries = subprocess.check_output(["git", "-C", str(work), "ls-tree", "--name-only", first.tree], text=True).splitlines()
            self.assertEqual(entries, ["manifest.json"])
            ref = first.ref
            self.assertEqual(subprocess.check_output(["git", "-C", str(work), "ls-remote", "origin", ref], text=True).split("\t")[1].strip(), ref)
            conflicting = seal_manifest(self.draft(timestamp_utc="2026-09-22T00:00:01Z"))
            with self.assertRaisesRegex(ManifestConflict, "CREATE_ONCE_CONFLICT"):
                writer.create(conflicting, protection)

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
            manifest = seal_manifest(self.draft())
            protection = ProtectionEvidence("fixture", "refs/heads/lifecycle/v1/*", "0" * 64, "2026-09-22T00:00:00Z", True, True, True)
            held = {"value": False}
            class Guard:
                def __enter__(self) -> bool:
                    held["value"] = True
                    return True
                def __exit__(self, *_args: object) -> None:
                    held["value"] = False
            values = iter(({"snapshot": 1}, {"snapshot": 2}))
            def callback() -> object:
                self.assertTrue(held["value"])
                return next(values)
            writer = CreateOnlyLifecycleWriter(work, exclusion=lambda: Guard(), snapshot_callback=callback)
            with self.assertRaisesRegex(ManifestError, "SNAPSHOT_STALE"):
                writer.create(manifest, protection)
            self.assertFalse(held["value"])

            manifest = seal_manifest(self.draft(timestamp_utc="2026-09-22T00:01:00Z"))
            writer = CreateOnlyLifecycleWriter(work, exclusion=lambda: nullcontext(True))
            import version_lifecycle.manifests as module
            real_git = module._git
            def fail_push(repository: Path, *args: str, input_bytes: bytes | None = None, check: bool = True) -> bytes:
                if args and args[0] == "push":
                    raise ManifestError("transport unavailable")
                return real_git(repository, *args, input_bytes=input_bytes, check=check)
            with patch.object(module, "_git", side_effect=fail_push):
                with self.assertRaisesRegex(Exception, "CREATE_ONCE_OUTCOME_UNCERTAIN"):
                    writer.create(manifest, protection)


if __name__ == "__main__":
    unittest.main()
