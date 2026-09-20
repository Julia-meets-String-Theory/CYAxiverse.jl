#!/usr/bin/env python3
"""Focused Gate A certification and release evidence fixtures."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.certification import (
    BLOCKED as CERT_BLOCKED,
    PASS as CERT_PASS,
    certify_exact_tree,
    validate_certification_identity,
    validate_certification_transfer,
)
from version_lifecycle.codec import canonical_json, sha256_hex
from version_lifecycle.events import validate_event
from version_lifecycle.release import (
    BLOCKED,
    LEGACY_EXCLUDED,
    PASS,
    INVALID,
    PUBLICATION_RECONCILIATION_PENDING,
    TAG_RECONCILIATION_PENDING,
    TERMINAL_CONSISTENT,
    validate_publication_evidence,
    validate_release_catalog,
    validate_release_consistency,
    validate_released_event,
)


SHA_A = "a" * 40
SHA_B = "b" * 40
TREE = "c" * 40
EVENT_ID = "EVT-000000000123"
EVENT_ID_TWO = "EVT-000000000124"


def certification(binding: str = "tree-bound", subject_sha: str = SHA_A) -> dict:
    return {
        "binding": binding,
        "package_commit": subject_sha,
        "package_tree": TREE,
        "policy_revision": "policy-2026-09",
        "harness_revision": "harness-2026-09",
        "environment": "julia-1.12-python-3.14",
        "evidence_digest": "d" * 64,
        "evidence_ref": "evidence/certification.json",
    }


def released_event(
    line: str = "principal",
    *,
    version: str = "0.3.0",
    event_id: str = EVENT_ID,
    main_sha: str | None = None,
    main_version: str | None = None,
) -> dict:
    if main_sha is None:
        main_sha = SHA_A if line == "principal" else SHA_B
    if main_version is None:
        main_version = version if line == "principal" else "0.2.0"
    event = {
        "schema_version": 1,
        "event_id": event_id,
        "event_type": "released",
        "timestamp_utc": "2026-09-20T14:00:00Z",
        "transaction_id": f"release-{event_id}",
        "static_iteration_snapshot": "a" * 64,
        "expected_event_head": "0" * 40,
        "closure_timestamp_utc": "2026-09-20T13:00:00Z",
        "final_version": version,
        "release_line": line,
        "anchor_ref": f"refs/tags/iterations/{version}",
        "anchor_sha": SHA_A,
        "anchor_tree": TREE,
        "candidate_ref": f"refs/heads/candidates/{version}",
        "candidate_sha": SHA_A,
        "candidate_tree": TREE,
        "final_release_sha": SHA_A,
        "final_release_tree": TREE,
        "certification_binding": "tree-bound",
        "certification_subject_sha": SHA_A,
        "certification_subject_tree": TREE,
        "certification_policy_revision": "policy-2026-09",
        "certification_harness_revision": "harness-2026-09",
        "certification_environment": "julia-1.12-python-3.14",
        "certification_evidence_refs": ["evidence/certification.json"],
        "evidence_refs": ["evidence/released.json"],
        "public_tag": f"v{version}",
        "main_at_event_sha": main_sha,
        "main_at_event_version": main_version,
    }
    if line == "principal":
        event.update({
            "previous_main_sha": "e" * 40,
            "previous_main_version": "0.2.0",
        })
    return event


def project_versions(
    line: str = "principal",
    *,
    version: str = "0.3.0",
    main_version: str | None = None,
) -> dict:
    if main_version is None:
        main_version = version if line == "principal" else "0.2.0"
    return {
        "closure": version,
        "candidate": version,
        "anchor": version,
        "final_release": version,
        "certified": version,
        "main": main_version,
    }


def tag(version: str = "0.3.0") -> dict:
    return {"name": f"v{version}", "sha": SHA_A, "tree": TREE, "version": version}


def github_release(version: str = "0.3.0", release_id: int = 9001) -> dict:
    return {
        "id": release_id,
        "tag_name": f"v{version}",
        "target_sha": SHA_A,
        "target_tree": TREE,
        "published_at": "2026-09-20T15:00:00Z",
    }


def publication_evidence(version: str = "0.3.0", event_id: str = EVENT_ID, release_id: int = 9001) -> dict:
    return {
        "event_id": event_id,
        "public_tag": f"v{version}",
        "github_release_id": release_id,
        "published_at": "2026-09-20T15:00:00Z",
        "evidence_ref": "evidence/publication/v0.3.0.json",
        "evidence_digest": "f" * 64,
    }


def release_intent(version: str = "0.3.0") -> dict:
    return {
        "event_type": "release_intent_prepared",
        "intent_id": "INT-0.3.0",
        "public_tag": f"v{version}",
        "final_version": version,
        "final_release_sha": SHA_A,
        "final_release_tree": TREE,
    }


class TestCertification(unittest.TestCase):
    def test_exact_tree_positive(self):
        result = certify_exact_tree(
            candidate_tree=TREE,
            certified_tree=TREE,
            anchor_tree=TREE,
            final_release_tree=TREE,
        )
        self.assertEqual(result["status"], CERT_PASS)

    def test_changed_tree_rejected(self):
        result = certify_exact_tree(
            candidate_tree=TREE,
            certified_tree=TREE,
            anchor_tree="f" * 40,
        )
        self.assertEqual(result["reason_code"], "CERTIFIED_TREE_MISMATCH")

    def test_unsupported_binding_blocks(self):
        result = validate_certification_identity({"binding": "content-bound"})
        self.assertEqual(result, {"status": CERT_BLOCKED, "reason_code": "UNSUPPORTED_CERTIFICATION_BINDING"})

    def test_tree_bound_transfer_requires_durable_proof(self):
        record = certification(subject_sha=SHA_A)
        result = validate_certification_transfer(
            record,
            candidate_commit=SHA_A,
            candidate_tree=TREE,
            final_release_commit=SHA_B,
            final_release_tree=TREE,
            anchor_tree=TREE,
        )
        self.assertEqual(result["status"], CERT_BLOCKED)
        self.assertEqual(result["reason_code"], "TREE_BOUND_TRANSFER_EVIDENCE_REQUIRED")

    def test_tree_bound_transfer_with_proof(self):
        record = certification(subject_sha=SHA_A)
        record["candidate_commit"] = SHA_A
        record["candidate_tree"] = TREE
        record["anchor_tree"] = TREE
        record["transfer_evidence"] = {
            "certified_tree": TREE,
            "candidate_tree": TREE,
            "final_release_tree": TREE,
            "anchor_tree": TREE,
            "evidence_ref": "evidence/transfer.json",
        }
        result = validate_certification_transfer(
            record,
            candidate_commit=SHA_A,
            candidate_tree=TREE,
            final_release_commit=SHA_B,
            final_release_tree=TREE,
            anchor_tree=TREE,
        )
        self.assertEqual(result["status"], CERT_PASS)
        self.assertTrue(result["transferred"])

    def test_commit_bound_transfer_requires_recertification(self):
        record = certification(binding="commit-bound", subject_sha=SHA_A)
        result = validate_certification_transfer(
            record,
            candidate_commit=SHA_A,
            candidate_tree=TREE,
            final_release_commit=SHA_B,
            final_release_tree=TREE,
            anchor_tree=TREE,
        )
        self.assertEqual(result["reason_code"], "RECERTIFICATION_REQUIRED")


class TestReleaseEvidence(unittest.TestCase):
    def test_principal_identity_and_consistency(self):
        event = released_event()
        self.assertEqual(
            validate_released_event(
                event,
                certification=certification(),
                project_versions=project_versions(),
            )["status"],
            PASS,
        )
        result = validate_release_consistency(
            event,
            tag(),
            github_release(),
            publication_evidence(),
            release_intent=release_intent(),
            certification=certification(),
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], TERMINAL_CONSISTENT)

    def test_durable_tree_transfer_proof_matches_event_and_certification(self):
        event = released_event(main_sha=SHA_B)
        event["final_release_sha"] = SHA_B
        transfer = {
            "verified": True,
            "candidate_sha": SHA_A,
            "final_release_sha": SHA_B,
            "certified_tree": TREE,
            "candidate_tree": TREE,
            "final_release_tree": TREE,
            "anchor_tree": TREE,
            "evidence_ref": "evidence/transfer.json",
        }
        event["certification_transfer_evidence"] = dict(transfer)
        record = certification()
        record["transfer_evidence"] = dict(transfer)
        result = validate_released_event(
            event,
            certification=record,
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], PASS)

        missing = dict(event)
        missing.pop("certification_transfer_evidence")
        self.assertEqual(
            validate_released_event(
                missing,
                certification=record,
                project_versions=project_versions(),
            )["status"],
            INVALID,
        )

        missing_cert = validate_released_event(
            event,
            certification=certification(),
            project_versions=project_versions(),
        )
        self.assertEqual(missing_cert["status"], INVALID)

    def test_maintenance_forbids_previous_main_claim(self):
        event = released_event("maintenance/0.3")
        event["previous_main_sha"] = "e" * 40
        result = validate_released_event(
            event,
            certification=certification(),
            project_versions=project_versions("maintenance/0.3"),
            principal_main={"sha": SHA_B, "version": "0.2.0"},
        )
        self.assertEqual(result["status"], INVALID)
        self.assertIn("maintenance_previous_main_forbidden", result["errors"])

    def test_maintenance_line_must_match_final_version(self):
        event = released_event("maintenance/0.4")
        result = validate_released_event(
            event,
            certification=certification(),
            project_versions=project_versions("maintenance/0.4"),
            principal_main={"sha": SHA_B, "version": "0.2.0"},
        )
        self.assertEqual(result["status"], INVALID)
        self.assertIn("maintenance_line_version_mismatch", result["errors"])

    def test_project_version_evidence_is_required(self):
        result = validate_released_event(
            released_event(), certification=certification()
        )
        self.assertEqual(result["status"], BLOCKED)
        self.assertEqual(result["reason_code"], "PROJECT_VERSION_EVIDENCE_UNAVAILABLE")

    def test_pinned_certification_is_required(self):
        result = validate_released_event(
            released_event(), project_versions=project_versions()
        )
        self.assertEqual(result["status"], BLOCKED)
        self.assertEqual(result["reason_code"], "CERTIFICATION_EVIDENCE_UNAVAILABLE")

    def test_terminal_released_event_requires_schema_version_one(self):
        for schema_version in (None, 2):
            with self.subTest(schema_version=schema_version):
                event = released_event()
                if schema_version is None:
                    event.pop("schema_version")
                else:
                    event["schema_version"] = schema_version
                result = validate_released_event(
                    event,
                    certification=certification(),
                    project_versions=project_versions(),
                )
                self.assertEqual(result["status"], INVALID)
                self.assertIn("schema_version_must_be_one", result["errors"])

        event = released_event()
        event["unrecognized_alias"] = "value"
        result = validate_released_event(
            event,
            certification=certification(),
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], INVALID)
        self.assertTrue(any(error.startswith("unknown_event_fields:") for error in result["errors"]))

    def test_terminal_released_event_requires_all_common_schema_fields(self):
        for field in ("transaction_id", "static_iteration_snapshot", "expected_event_head"):
            with self.subTest(field=field):
                event = released_event()
                event.pop(field)
                result = validate_released_event(
                    event,
                    certification=certification(),
                    project_versions=project_versions(),
                )
                self.assertEqual(result["status"], INVALID)
                self.assertTrue(result["errors"])

    def test_anchor_and_candidate_refs_are_exact(self):
        event = released_event()
        event["anchor_ref"] = "refs/tags/other"
        result = validate_released_event(
            event,
            certification=certification(),
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], INVALID)
        self.assertIn("anchor_ref_version_mismatch", result["errors"])

    def test_principal_version_must_increase_and_main_match(self):
        event = released_event()
        event["previous_main_version"] = "0.3.0"
        result = validate_released_event(event, certification=certification())
        self.assertEqual(result["status"], INVALID)
        self.assertIn("principal_version_not_monotonic", result["errors"])

    def test_principal_main_at_event_must_equal_release_sha(self):
        event = released_event(main_sha=SHA_B)
        result = validate_released_event(
            event,
            certification=certification(),
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], INVALID)
        self.assertIn("principal_main_release_sha_mismatch", result["errors"])

    def test_certification_event_schema_aliases_match_identity(self):
        event = released_event()
        event_certification = {
            "certification_binding": "tree_bound",
            "certification_subject_sha": SHA_A,
            "certification_subject_tree": TREE,
            "certification_policy_revision": "policy-2026-09",
            "certification_harness_revision": "harness-2026-09",
            "certification_environment": "julia-1.12-python-3.14",
            "certification_evidence_refs": ["evidence/certification.json"],
        }
        result = validate_released_event(
            event,
            certification=event_certification,
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], PASS)

    def test_certification_identity_fields_must_match_event(self):
        cases = (
            ("policy_revision", "policy-2026-10", "certification_policy_revision_mismatch"),
            ("harness_revision", "harness-2026-10", "certification_harness_revision_mismatch"),
            ("environment", "julia-1.13-python-3.14", "certification_environment_mismatch"),
            ("evidence_ref", "evidence/other-certification.json", "certification_evidence_refs_mismatch"),
        )
        for field, value, expected_error in cases:
            with self.subTest(field=field):
                record = certification()
                record[field] = value
                result = validate_released_event(
                    released_event(),
                    certification=record,
                    project_versions=project_versions(),
                )
                self.assertEqual(result["status"], INVALID)
                self.assertIn(expected_error, result["errors"])

    def test_certification_evidence_reference_list_must_match_event(self):
        record = certification()
        record["evidence_refs"] = [
            "evidence/certification.json",
            "evidence/extra-certification.json",
        ]
        result = validate_released_event(
            released_event(),
            certification=record,
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], INVALID)
        self.assertIn("certification_evidence_refs_mismatch", result["errors"])

    def test_certification_evidence_identity_must_match_reference_list(self):
        record = certification()
        record["evidence"] = "evidence/other-certification.json"
        record["evidence_refs"] = ["evidence/certification.json"]
        result = validate_released_event(
            released_event(),
            certification=record,
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], INVALID)
        self.assertIn("certification_evidence_identity_mismatch", result["errors"])

    def test_conflicting_certification_aliases_are_invalid(self):
        record = certification()
        record["certification_policy_revision"] = "different-policy"
        result = validate_released_event(
            released_event(), certification=record, project_versions=project_versions()
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "CERTIFICATION_IDENTITY_AMBIGUOUS")

        record = certification()
        record["evidence_refs"] = ["evidence/certification.json"]
        record["certification_evidence_refs"] = ["evidence/another.json"]
        result = validate_released_event(
            released_event(), certification=record, project_versions=project_versions()
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "CERTIFICATION_IDENTITY_AMBIGUOUS")

    def test_tree_mismatch_is_invalid(self):
        event = released_event()
        event["final_release_tree"] = "f" * 40
        result = validate_released_event(event, certification=certification())
        self.assertEqual(result["status"], INVALID)
        self.assertIn("release_tree_mismatch", result["errors"])

    def test_tag_without_event_with_matching_intent_is_pending(self):
        intent = {
            "public_tag": "v0.3.0",
            "final_version": "0.3.0",
            "final_release_sha": SHA_A,
            "final_release_tree": TREE,
        }
        result = validate_release_consistency(None, tag(), release_intent=intent)
        self.assertEqual(result["status"], TAG_RECONCILIATION_PENDING)

    def test_released_event_without_publication_is_pending(self):
        result = validate_release_consistency(
            released_event(),
            tag(),
            certification=certification(),
            project_versions=project_versions(),
            release_intent=release_intent(),
        )
        self.assertEqual(result["status"], PUBLICATION_RECONCILIATION_PENDING)

    def test_bidirectional_mismatch_is_invalid(self):
        release = github_release()
        release["target_sha"] = SHA_B
        result = validate_release_consistency(
            released_event(),
            tag(),
            release,
            publication_evidence(),
            release_intent=release_intent(),
            certification=certification(),
            project_versions=project_versions(),
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "GITHUB_RELEASE_SHA_MISMATCH")

    def test_publication_evidence_is_immutable_identity(self):
        result = validate_publication_evidence(
            publication_evidence(), event=released_event(), tag=tag(), github_release=github_release()
        )
        self.assertEqual(result["status"], PASS)
        bad = publication_evidence()
        bad["evidence_ref"] = "/Users/owner/private.json"
        self.assertEqual(validate_publication_evidence(bad)["reason_code"], "UNSAFE_PUBLIC_EVIDENCE")

        for value in (
            "local://runner/evidence.json",
            "ssh://runner/evidence.json",
            "private/evidence.json",
            "../private/evidence.json",
            "credentials/evidence.json",
            "https://intranet/org/evidence.json",
        ):
            with self.subTest(value=value):
                bad = publication_evidence()
                bad["evidence_ref"] = value
                self.assertEqual(
                    validate_publication_evidence(bad)["reason_code"],
                    "UNSAFE_PUBLIC_EVIDENCE",
                )

    def test_certification_rejects_private_public_boundary_values(self):
        for field, value in (
            ("evidence_ref", "local://runner/certification.json"),
            ("policy_revision", "ssh://runner/policy"),
            ("harness_revision", "private/harness"),
            ("environment", "https://intranet/ci"),
        ):
            with self.subTest(field=field):
                record = certification()
                record[field] = value
                result = validate_certification_identity(record)
                self.assertEqual(result["status"], INVALID)
                self.assertEqual(result["reason_code"], "UNSAFE_PUBLIC_EVIDENCE")

    def test_legacy_tag_is_excluded(self):
        result = validate_release_consistency(tag={"name": "v-0.1"})
        self.assertEqual(result["status"], LEGACY_EXCLUDED)

    def test_release_event_cannot_use_legacy_tag(self):
        event = released_event()
        event["public_tag"] = "v-0.1"
        result = validate_released_event(event, certification=certification())
        self.assertEqual(result["status"], INVALID)

    def test_catalog_terminal_cardinality(self):
        result = validate_release_catalog(
            [released_event()],
            [tag()],
            [github_release()],
            [publication_evidence()],
            release_intents=[release_intent()],
            certifications={EVENT_ID: certification()},
            project_versions={EVENT_ID: project_versions()},
        )
        self.assertEqual(result["status"], TERMINAL_CONSISTENT)
        self.assertEqual(result["release_count"], 1)

    def test_catalog_rejects_noncanonical_released_event(self):
        event = released_event()
        event.pop("transaction_id")
        result = validate_release_catalog(
            [event],
            [tag()],
            [github_release()],
            [publication_evidence()],
            release_intents=[release_intent()],
            certifications={EVENT_ID: certification()},
            project_versions={EVENT_ID: project_versions()},
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "RELEASED_EVENT_INVALID")

    def test_catalog_duplicate_tag_is_invalid(self):
        result = validate_release_catalog(
            [released_event()],
            [tag(), tag()],
            [github_release()],
            [publication_evidence()],
            certifications={EVENT_ID: certification()},
            project_versions={EVENT_ID: project_versions()},
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "DUPLICATE_CANONICAL_TAG")

    def test_catalog_canonical_string_tag_is_validated_as_incomplete(self):
        result = validate_release_catalog(
            [released_event()],
            ["v0.3.0"],
            [github_release()],
            [publication_evidence()],
            release_intents=[release_intent()],
            certifications={EVENT_ID: certification()},
            project_versions={EVENT_ID: project_versions()},
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "TAG_IDENTITY_INCOMPLETE")

    def test_catalog_tag_pending_requires_intent(self):
        intent = {
            "public_tag": "v0.3.0",
            "final_version": "0.3.0",
            "final_release_sha": SHA_A,
            "final_release_tree": TREE,
        }
        result = validate_release_catalog(
            [], [tag()], [], [], release_intents=[intent]
        )
        self.assertEqual(result["status"], TAG_RECONCILIATION_PENDING)

    def test_catalog_publication_pending(self):
        result = validate_release_catalog(
            [released_event()],
            [tag()],
            [],
            [],
            certifications={EVENT_ID: certification()},
            project_versions={EVENT_ID: project_versions()},
            release_intents=[release_intent()],
        )
        self.assertEqual(result["status"], PUBLICATION_RECONCILIATION_PENDING)

    def test_catalog_orphan_release_is_invalid(self):
        result = validate_release_catalog(
            [], [tag()], [github_release()], [],
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "GITHUB_RELEASE_EVENT_MISSING")

    def test_catalog_malformed_names_are_structured_invalid(self):
        result = validate_release_catalog(
            [released_event()],
            [{"name": []}],
            [github_release()],
            [publication_evidence()],
            release_intents=[release_intent()],
            certifications={EVENT_ID: certification()},
            project_versions={EVENT_ID: project_versions()},
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "INVALID_CANONICAL_TAG_RECORD")

    def test_catalog_legacy_record_is_ignored_with_future_release(self):
        result = validate_release_catalog(
            [released_event()],
            ["v-0.1", tag()],
            [github_release()],
            [publication_evidence()],
            release_intents=[release_intent()],
            certifications={EVENT_ID: certification()},
            project_versions={EVENT_ID: project_versions()},
        )
        self.assertEqual(result["status"], TERMINAL_CONSISTENT)

    def test_crosslinked_legacy_tuple_is_invalid(self):
        result = validate_release_consistency(
            released_event(), {"name": "v-0.1"}, release_intent=release_intent()
        )
        self.assertEqual(result["status"], INVALID)
        self.assertEqual(result["reason_code"], "MIXED_LEGACY_CANONICAL_IDENTITIES")

    def test_catalog_wholly_legacy_is_excluded(self):
        result = validate_release_catalog(
            [], [{"name": "v-0.1"}], [{"tag_name": "v-0.1", "id": 1}], []
        )
        self.assertEqual(result["status"], LEGACY_EXCLUDED)

    def test_catalog_uses_event_bound_maintenance_main_snapshots(self):
        first = released_event(
            "maintenance/0.3",
            main_sha=SHA_B,
            main_version="0.2.0",
        )
        second = released_event(
            "maintenance/0.3",
            version="0.3.1",
            event_id=EVENT_ID_TWO,
            main_sha="d" * 40,
            main_version="0.2.1",
        )
        first_main = {"sha": SHA_B, "version": "0.2.0"}
        second_main = {"sha": "d" * 40, "version": "0.2.1"}
        result = validate_release_catalog(
            [first, second],
            [tag(), tag("0.3.1")],
            [github_release(), github_release("0.3.1", 9002)],
            [
                publication_evidence(),
                publication_evidence("0.3.1", EVENT_ID_TWO, 9002),
            ],
            release_intents=[release_intent(), release_intent("0.3.1")],
            certifications={EVENT_ID: certification(), EVENT_ID_TWO: certification()},
            project_versions={
                EVENT_ID: project_versions("maintenance/0.3"),
                EVENT_ID_TWO: project_versions("maintenance/0.3", version="0.3.1", main_version="0.2.1"),
            },
            principal_main={EVENT_ID: first_main, EVENT_ID_TWO: second_main},
        )
        self.assertEqual(result["status"], TERMINAL_CONSISTENT)


class TestSyntheticEvidencePacket(unittest.TestCase):
    def test_stored_packet_and_publication_digests(self):
        repository = Path(__file__).resolve().parents[1]
        fixture = repository / "specs/0125-version-iteration-release/fixtures/synthetic-release-evidence.json"
        raw = fixture.read_bytes()
        packet = json.loads(raw)
        self.assertEqual(raw, canonical_json(packet))
        self.assertIs(packet["fixture_only"], True)
        for line in ("principal", "maintenance"):
            with self.subTest(line=line):
                record = packet[line]
                publication = record["publication_evidence"]
                payload_path = repository / publication["evidence_ref"]
                payload_raw = payload_path.read_bytes()
                self.assertEqual(payload_raw, canonical_json(json.loads(payload_raw)))
                self.assertEqual(sha256_hex(payload_raw), publication["evidence_digest"])
                event = record["event"]
                self.assertEqual(validate_event(event)["event_id"], event["event_id"])
                self.assertEqual(
                    validate_release_consistency(
                        event=event, tag=record["tag"],
                        github_release=record["github_release"],
                        publication_evidence=publication,
                        release_intent=record["release_intent"],
                        certification=record["certification"],
                        project_versions=record["project_versions"],
                        principal_main={
                            "sha": record["event"]["main_at_event_sha"],
                            "version": record["event"]["main_at_event_version"],
                        },
                    )["status"], TERMINAL_CONSISTENT,
                )


if __name__ == "__main__":
    raise SystemExit(unittest.main())
