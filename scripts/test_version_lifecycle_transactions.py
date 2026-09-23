"""Fixture coverage for the reduced first-principal lifecycle coordinator."""

from __future__ import annotations

import json
import sys
import unittest
from hashlib import sha256
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.authorization import (  # noqa: E402
    AuthorizationResolution,
    canonical_authorization_bytes,
    seal_authorization,
)
from version_lifecycle.manifests import (  # noqa: E402
    lifecycle_ref_for_manifest,
    seal_manifest,
    validate_complete_lifecycle_refs,
    validate_manifest,
)
from version_lifecycle.publication_evidence import (  # noqa: E402
    publication_evidence_for_tag,
)
from version_lifecycle.transactions import (  # noqa: E402
    AllocationView,
    BootstrapIntent,
    ClosureIntent,
    ReleaseIntent,
    run_closure,
    run_maintenance_bootstrap,
    run_rare_recovery,
    run_release,
)


SHA_A = "a" * 40
SHA_B = "b" * 40
TREE = "c" * 40
SNAPSHOT = "0" * 64
LIFECYCLE_SNAPSHOT = "1" * 64
OWNER = "owner-principal"
REPOSITORY = "fixture-repository"
RESERVATION_REF = (
    "refs/heads/lifecycle/v1/reservations/principal/v0.3.0-DEV/"
    + "LIF-SHA256-" + "e" * 64
)
CONSUMED_RESERVATION_REF = (
    "refs/heads/lifecycle/v1/reservations/principal/v0.3.0-DEV/"
    + "LIF-SHA256-" + "f" * 64
)
CLAIM_REF = "refs/heads/lifecycle/v1/claims/v0.3.0"
ANCHOR_REF = "refs/tags/iterations/0.3.0"
RELEASE_TARGET = "refs/heads/lifecycle/v1/releases/v0.3.0"


def make_authorization(
    transaction_id: str,
    target_refs: list[str],
    *,
    final_version: str = "0.3.0",
    authority_source_ref: str | None = None,
    authorized_actions: list[str] | None = None,
) -> dict[str, object]:
    return seal_authorization({
        "schema_version": 1,
        "repository": REPOSITORY,
        "owner_account": OWNER,
        "authority_source_ref": authority_source_ref or f"owner-authority://fixture/{transaction_id}",
        "issued_at_utc": "2026-09-20T00:00:00Z",
        "expires_at_utc": "2026-09-21T00:00:00Z",
        "transaction_id": transaction_id,
        "owner_line": "principal",
        "final_version": final_version,
        "authorized_actions": authorized_actions or ["create-release-manifest", "create-tag"],
        "target_refs": sorted(set(target_refs)),
    })


class FixtureAuthority:
    def __init__(self, record: dict[str, object], *, owner: bool = True):
        self.record = record
        self.raw = canonical_authorization_bytes(record)
        self.owner = owner

    def fetch_owner_authorization(self, reference: str):
        return AuthorizationResolution(self.record, self.raw, self.owner)


class IssuingAuthority:
    def __init__(self, transaction_id: str, *, owner: bool = True):
        self.transaction_id = transaction_id
        self.owner = owner
        self.records: dict[str, dict[str, object]] = {}

    def issue(self, action: str, target_ref: str, final_version: str) -> str:
        reference = (
            f"owner-authority://fixture/{self.transaction_id}/"
            f"grant-{len(self.records) + 1}"
        )
        self.records[reference] = make_authorization(
            self.transaction_id,
            [target_ref],
            final_version=final_version,
            authority_source_ref=reference,
            authorized_actions=[action],
        )
        return reference

    def fetch_owner_authorization(self, reference: str):
        record = self.records[reference]
        return AuthorizationResolution(
            record, canonical_authorization_bytes(record), self.owner
        )


CLOSURE_AUTHORIZATION = make_authorization(
    "tx-closure", [RELEASE_TARGET, ANCHOR_REF, "refs/heads/vmm"]
)
RELEASE_AUTHORIZATION = make_authorization(
    "tx-release", [RELEASE_TARGET, "refs/heads/main", "refs/tags/v0.3.0",
                    "refs/heads/candidates/v0.3.0/candidate-0-3-0"]
)
CLOSURE_AUTHORITY = FixtureAuthority(CLOSURE_AUTHORIZATION)
RELEASE_AUTHORITY = FixtureAuthority(RELEASE_AUTHORIZATION)


class PrincipalClosureFixture:
    def __init__(self, *, fail: str | None = None, occupied: frozenset[str] = frozenset()):
        self.fail = fail
        self.calls: list[str] = []
        self.manifest_types: list[str] = []
        self.manifests: list[dict[str, object]] = []
        self.occupied = occupied
        self.static_epoch = 0
        self.lifecycle_epoch = 1
        self.now_utc = "2026-09-20T12:34:56Z"
        self.authorization = IssuingAuthority("tx-closure")
        self.owner_authorization_authority = self.authorization

    def owner_authorization_ref_for(self, action, target_ref, final_version):
        return self.authorization.issue(action, target_ref, final_version)

    def authorization_now_utc(self):
        return self.now_utc

    def freeze_line(self, intent):
        self.calls.append("freeze")
        if self.fail == "freeze":
            return None
        return "line-freeze"

    def acquire_static_mutation(self, intent):
        self.calls.append("serialize")
        if self.fail == "serialize":
            raise RuntimeError("exclusion unavailable")
        return "exclusion"

    def release_static_mutation(self, lease):
        self.calls.append("release-exclusion")

    def allocation_view(self):
        self.calls.append("view")
        return AllocationView(
            f"{self.static_epoch:x}" * 64,
            f"{self.lifecycle_epoch:x}" * 64,
            self.occupied,
        )

    def verify_closure_target(self, intent, view):
        self.calls.append("verify-closure")
        return {"verified": True}

    def create_closure_anchor(self, intent):
        self.calls.append("anchor")
        if self.fail == "anchor":
            return {"version": intent.final_version, "commit": "bad", "tree": TREE,
                    "closure_timestamp_utc": intent.closure_timestamp_utc}
        return {"version": intent.final_version, "commit": SHA_A, "tree": TREE,
                "closure_timestamp_utc": intent.closure_timestamp_utc,
                "ref": ANCHOR_REF}

    def create_anchor(self, intent, closure):
        self.calls.append("create-anchor")
        self.static_epoch += 2
        return dict(closure)

    def create_manifest(self, manifest_type, payload):
        self.calls.append(f"manifest:{manifest_type}")
        self.manifest_types.append(manifest_type)
        if self.fail == manifest_type:
            raise RuntimeError(f"failed {manifest_type}")
        result = seal_manifest(dict(payload))
        validate_manifest(result)
        self.manifests.append(result)
        self.lifecycle_epoch += 1
        return result

    def verify_outgoing_terminal(self, intent, consumption):
        self.calls.append("outgoing-terminal")
        return {"verified": True}

    def reopen_dev(self, intent, preparation):
        self.calls.append("reopen")
        self.static_epoch += 1
        return {"version": preparation["intended_dev_version"], "head": SHA_B}

    def activate_next(self, intent, preparation, reopened):
        self.calls.append("activate")
        return {"head": reopened["head"]}

    def verify_closure_correspondence(self, intent, anchor, consumed, prepared, reopened, active, claim):
        self.calls.append("correspondence")
        if self.fail == "correspondence":
            raise RuntimeError("correspondence unavailable")
        return {"verified": True}

    def unfreeze_line(self, token):
        self.calls.append("unfreeze")


class NoExclusionClosureFixture(PrincipalClosureFixture):
    def __getattribute__(self, name):
        if name in {"acquire_static_mutation", "acquire_allocation_exclusion"}:
            raise AttributeError(name)
        return super().__getattribute__(name)


class CachedClosureFixture(PrincipalClosureFixture):
    def allocation_view(self):
        self.calls.append("view")
        return AllocationView(SNAPSHOT, LIFECYCLE_SNAPSHOT, self.occupied)


class PrincipalReleaseFixture:
    def __init__(self, *, binding="tree-bound", fail: str | None = None):
        self.binding = binding
        self.fail = fail
        self.calls: list[str] = []
        self.manifest_types: list[str] = []
        self.manifests: list[dict[str, object]] = []
        self.static_epoch = 0
        self.lifecycle_epoch = 1
        self.now_utc = "2026-09-20T13:00:00Z"
        self.publication_evidence_ref = "evidence/publication.json"
        self.publication_evidence_override = None
        self.persisted_publication_evidence_payload = None
        self.authorization = IssuingAuthority("tx-release")
        self.owner_authorization_authority = self.authorization

    def owner_authorization_ref_for(self, action, target_ref, final_version):
        return self.authorization.issue(action, target_ref, final_version)

    def authorization_now_utc(self):
        return self.now_utc

    def acquire_static_mutation(self, intent):
        self.calls.append("serialize")
        return "exclusion"

    def allocation_view(self):
        self.calls.append("view")
        return AllocationView(
            f"{self.static_epoch:x}" * 64,
            f"{self.lifecycle_epoch:x}" * 64,
            frozenset({"0.3.0"}),
        )

    def release_static_mutation(self, lease):
        self.calls.append("release-exclusion")

    def verify_anchor(self, intent):
        self.calls.append("anchor")
        return {
            "ref": ANCHOR_REF,
            "sha": SHA_B,
            "tree": TREE,
            "closure_timestamp_utc": intent.closure_timestamp_utc,
        }

    def create_candidate(self, intent):
        self.calls.append("candidate")
        return {"candidate_id": "candidate-0-3-0", "ref": intent.candidate_ref,
                "sha": SHA_A, "tree": TREE, "version": intent.final_version,
                "durable": True, "main_at_candidate_sha": SHA_B,
                "main_at_candidate_version": "0.2.0"}

    def create_manifest(self, manifest_type, payload):
        self.calls.append(f"manifest:{manifest_type}")
        self.manifest_types.append(manifest_type)
        if self.fail == manifest_type:
            raise RuntimeError(f"failed {manifest_type}")
        result = seal_manifest(dict(payload))
        validate_manifest(result)
        self.manifests.append(result)
        self.lifecycle_epoch += 1
        return result

    def certify_candidate(self, intent, candidate):
        self.calls.append("certify")
        return {"binding": self.binding, "subject_sha": candidate["sha"],
                "subject_tree": candidate["tree"], "policy_revision": "policy",
                "harness_revision": "harness", "environment": "fixture",
                "evidence_refs": ["evidence/candidate.json"]}

    def freeze_main(self, intent):
        self.calls.append("freeze-main")
        return {"token": "main-freeze", "sha": SHA_B, "version": "0.2.0"}

    def verify_principal_interval(self, intent, candidate, freeze):
        self.calls.append("interval")
        return {"verified": True, "intervening_commits": [], "disposition": "no_drift"}

    def promote_principal(self, intent, candidate, certification, freeze):
        self.calls.append("promote")
        return {"sha": SHA_B, "tree": TREE, "version": intent.final_version,
                "main_at_release_sha": SHA_B,
                "main_at_release_version": intent.final_version,
                "evidence_refs": ["evidence/released.json"]}

    def verify_tree_transfer(self, intent, candidate, certification, final):
        self.calls.append("tree-transfer")
        return {"verified": True, "candidate_sha": candidate["sha"],
                "final_release_sha": final["sha"], "candidate_tree": TREE,
                "final_release_tree": TREE, "anchor_tree": TREE,
                "evidence_ref": "evidence/transfer.json"}

    def recertify_final(self, intent, final):
        self.calls.append("recertify")
        return {"binding": "commit-bound", "subject_sha": final["sha"],
                "subject_tree": final["tree"], "policy_revision": "policy-final",
                "harness_revision": "harness-final", "environment": "fixture-final",
                "evidence_refs": ["evidence/final.json"]}

    class Protection:
        def require_public_tag(self, ref):
            return None

    def verify_public_tag_ruleset(self, intent, ref):
        self.calls.append("tag-ruleset")
        return self.Protection()

    def create_tag(self, intent, prepared, final):
        self.calls.append("tag")
        if self.fail == "tag":
            return {"name": intent.public_tag, "commit": "bad", "tree": TREE, "protected": True}
        self.static_epoch += 2
        return {"name": intent.public_tag, "commit": final["sha"], "tree": final["tree"], "protected": True}

    def publish_github_release(self, intent, released):
        self.calls.append("github-release")
        return {"id": 1, "tag": intent.public_tag,
                "url": "https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/releases/tag/v0.3.0",
                "published_at_utc": "2026-09-20T13:30:00Z"}

    def publication_evidence_target(self, intent, released, tag):
        self.calls.append("publication-evidence-target")
        if self.fail == "unsafe-publication-evidence-ref":
            return "evidence/chatgpt.com/share/private"
        return self.publication_evidence_ref

    def publication_evidence_payload(self, intent, released, tag):
        self.calls.append("publication-evidence-payload")
        if self.fail == "unsafe-publication-evidence-payload":
            return {"share": "https://chatgpt.com/share/synthetic-private-chat"}
        if self.publication_evidence_override is not None:
            return self.publication_evidence_override
        return publication_evidence_for_tag(released, tag)

    def persist_publication_evidence(self, intent, target, payload):
        self.calls.append("publication-evidence")
        if self.fail == "publication-evidence":
            raise RuntimeError("publication evidence timeout")
        if self.fail != "publication-evidence-no-store":
            self.persisted_publication_evidence_payload = payload
        return {"ref": target, "digest": sha256(payload).hexdigest()}

    def read_publication_evidence(self, intent, target):
        self.calls.append("publication-evidence-readback")
        if self.persisted_publication_evidence_payload is None:
            return None
        payload = self.persisted_publication_evidence_payload
        if self.fail == "publication-evidence-readback-mismatch":
            record = json.loads(payload)
            record["tag_tree"] = "f" * 40
            payload = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        return {"ref": target, "bytes": payload}

    def verify_released(self, intent, released, certification, final):
        self.calls.append("verify-released")
        return {"verified": True}

    def verify_terminal(self, intent, released, publication, evidence):
        self.calls.append("terminal")
        if self.fail == "terminal":
            raise RuntimeError("terminal proof unavailable")
        return {"verified": True}

    def dispatch_documentation(self, publication_ref):
        self.calls.append("documentation-dispatch")
        if self.fail == "documentation-dispatch":
            raise RuntimeError("documentation dispatch unavailable")
        if self.fail == "documentation-dispatch-mismatch":
            return {"status": "requested", "publication_ref": "refs/heads/wrong"}
        return {"status": "requested", "publication_ref": publication_ref}

    def unfreeze_main(self, token):
        self.calls.append("unfreeze-main")


class NoTagProtectionReleaseFixture(PrincipalReleaseFixture):
    def __getattribute__(self, name):
        if name == "verify_public_tag_ruleset":
            raise AttributeError(name)
        return super().__getattribute__(name)


class MissingClosureProofFixture(PrincipalClosureFixture):
    def __init__(self, missing: str):
        super().__init__()
        self.missing = missing

    def __getattribute__(self, name):
        missing = object.__getattribute__(self, "__dict__").get("missing")
        if name not in {"missing", "__dict__"} and name == missing:
            raise AttributeError(name)
        return super().__getattribute__(name)


class MissingReleaseProofFixture(PrincipalReleaseFixture):
    def __init__(self, missing: str):
        super().__init__()
        self.missing = missing

    def __getattribute__(self, name):
        missing = object.__getattribute__(self, "__dict__").get("missing")
        if name not in {"missing", "__dict__"} and name == missing:
            raise AttributeError(name)
        return super().__getattribute__(name)


class TransactionTests(unittest.TestCase):
    closure_intent = ClosureIntent(
        owner_line="principal", final_version="0.3.0",
        closure_timestamp_utc="2026-09-20T12:34:56Z",
        outgoing_reserved_final="0.3.0", transaction_id="tx-closure",
        expected_line_head=SHA_B, static_snapshot_digest=SNAPSHOT,
        lifecycle_snapshot_digest=LIFECYCLE_SNAPSHOT,
        owner_authorization=CLOSURE_AUTHORIZATION["owner_authorization"],
        owner_authorization_ref=CLOSURE_AUTHORIZATION["authority_source_ref"],
        owner_authorization_digest=CLOSURE_AUTHORIZATION["owner_authorization_digest"],
        repository=REPOSITORY,
        predecessor_refs=(RESERVATION_REF,),
    )
    release_intent = ReleaseIntent(
        owner_line="principal", final_version="0.3.0",
        candidate_ref="refs/heads/candidates/v0.3.0/candidate-0-3-0",
        public_tag="v0.3.0", candidate_sha=SHA_A, candidate_tree=TREE,
        transaction_id="tx-release", anchor_ref=ANCHOR_REF,
        anchor_sha=SHA_B, anchor_tree=TREE,
        closure_timestamp_utc="2026-09-20T12:34:56Z",
        timestamp_utc="2026-09-20T13:00:00Z",
        owner_authorization=RELEASE_AUTHORIZATION["owner_authorization"],
        owner_authorization_ref=RELEASE_AUTHORIZATION["authority_source_ref"],
        owner_authorization_digest=RELEASE_AUTHORIZATION["owner_authorization_digest"],
        repository=REPOSITORY,
        static_snapshot_digest=SNAPSHOT,
        lifecycle_snapshot_digest=LIFECYCLE_SNAPSHOT,
        predecessor_refs=(CLAIM_REF,),
    )

    def test_principal_closure_is_serialized_and_terminal_before_reopen(self):
        port = PrincipalClosureFixture()
        result = run_closure(port, self.closure_intent)
        self.assertTrue(result.complete)
        self.assertFalse(result.frozen)
        self.assertEqual(result.evidence["next_final"], "0.3.1")
        self.assertEqual(port.manifest_types,
                         ["reservation-consumed", "reservation-prepared", "reservation-opened", "version-claimed"])
        self.assertLess(port.calls.index("manifest:reservation-consumed"), port.calls.index("manifest:reservation-prepared"))
        self.assertLess(port.calls.index("manifest:reservation-prepared"), port.calls.index("reopen"))
        self.assertEqual(port.calls[-2:], ["release-exclusion", "unfreeze"])
        snapshots = [
            (
                manifest["static_iteration_snapshot"],
                manifest["lifecycle_ref_snapshot"],
            )
            for manifest in port.manifests
        ]
        self.assertEqual(
            snapshots,
            [
                ("2" * 64, "1" * 64),
                ("2" * 64, "2" * 64),
                ("3" * 64, "3" * 64),
                ("3" * 64, "4" * 64),
            ],
        )

    def test_closure_rejects_cached_authority_after_anchor_mutation(self):
        port = CachedClosureFixture()
        result = run_closure(port, self.closure_intent)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "ALLOCATION_SNAPSHOT_STALE", True),
        )
        self.assertNotIn("reservation-consumed", port.manifest_types)

    def test_emitted_manifests_pass_the_real_schema(self):
        closure_port = PrincipalClosureFixture()
        self.assertTrue(run_closure(closure_port, self.closure_intent).complete)
        for manifest in closure_port.manifests:
            self.assertEqual(validate_manifest(manifest), manifest)

        release_port = PrincipalReleaseFixture()
        self.assertTrue(run_release(release_port, self.release_intent).complete)
        for manifest in release_port.manifests:
            self.assertEqual(validate_manifest(manifest), manifest)

    def test_actual_principal_outputs_replay_from_reservation_to_publication(self):
        common = {
            "schema_version": 1,
            "timestamp_utc": "2026-09-20T12:00:00Z",
            "owner_authorization": CLOSURE_AUTHORIZATION["owner_authorization"],
            "owner_authorization_ref": CLOSURE_AUTHORIZATION["authority_source_ref"],
            "owner_authorization_digest": CLOSURE_AUTHORIZATION["owner_authorization_digest"],
            "transaction_id": "tx-prior-reservation",
            "static_iteration_snapshot": SNAPSHOT,
            "lifecycle_ref_snapshot": LIFECYCLE_SNAPSHOT,
        }
        prepared = seal_manifest({
            **common,
            "manifest_type": "reservation-prepared",
            "predecessor_refs": [],
            "owner_line": "principal",
            "final_version": "0.3.0",
            "reserved_final": "0.3.0",
            "intended_dev_version": "0.3.0-DEV",
            "expected_line_head": SHA_B,
        })
        opened = seal_manifest({
            **common,
            "manifest_type": "reservation-opened",
            "predecessor_refs": [lifecycle_ref_for_manifest(prepared)],
            "owner_line": "principal",
            "final_version": "0.3.0",
            "reserved_final": "0.3.0",
            "intended_dev_version": "0.3.0-DEV",
            "actual_dev_head": SHA_B,
        })
        claim = seal_manifest({
            **common,
            "manifest_type": "version-claimed",
            "predecessor_refs": [lifecycle_ref_for_manifest(opened)],
            "owner_line": "principal",
            "final_version": "0.3.0",
        })
        closure_port = PrincipalClosureFixture()
        closure = run_closure(
            closure_port,
            replace(
                self.closure_intent,
                predecessor_refs=(lifecycle_ref_for_manifest(opened),),
            ),
        )
        release_port = PrincipalReleaseFixture()
        release = run_release(
            release_port,
            replace(
                self.release_intent,
                predecessor_refs=(lifecycle_ref_for_manifest(claim),),
            ),
        )
        self.assertTrue(closure.complete)
        self.assertTrue(release.complete)
        manifests = [
            prepared, opened, claim,
            *closure_port.manifests,
            *release_port.manifests,
        ]
        graph = validate_complete_lifecycle_refs({
            lifecycle_ref_for_manifest(item): item for item in manifests
        })
        self.assertIn("0.3.0", graph.occupied_versions)
        self.assertIn("0.3.1", graph.occupied_versions)

    def test_exact_utc_timestamp_is_required(self):
        port = PrincipalClosureFixture()
        bad = ClosureIntent("principal", "0.3.0", "2026-02-30T12:34:56Z", "0.3.0")
        result = run_closure(port, bad)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "INVALID_CLOSURE_TIMESTAMP", False))
        self.assertNotIn("freeze", port.calls)

    def test_outgoing_failure_keeps_line_frozen_and_prevents_new_allocation(self):
        port = PrincipalClosureFixture(fail="reservation-consumed")
        result = run_closure(port, self.closure_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "PORT_OPERATION_FAILED", True))
        self.assertNotIn("reservation-prepared", port.manifest_types)
        self.assertNotIn("unfreeze", port.calls)

    def test_missing_exclusion_fails_closed_before_closure_writes(self):
        port = NoExclusionClosureFixture()
        result = run_closure(port, self.closure_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "EXCLUSION_UNAVAILABLE", True))
        self.assertNotIn("anchor", port.calls)
        self.assertNotIn("unfreeze", port.calls)

    def test_closure_missing_authority_blocks_before_anchor_write(self):
        port = PrincipalClosureFixture()
        intent = replace(self.closure_intent, owner_authorization="", owner_authorization_ref="",
                         owner_authorization_digest="")
        result = run_closure(port, intent)
        self.assertEqual((result.status, result.reason_code),
                         ("BLOCKED", "OWNER_AUTHORIZATION_UNVERIFIED"))
        self.assertNotIn("anchor", port.calls)

    def test_authority_must_come_from_trusted_port_configuration(self):
        port = PrincipalClosureFixture()
        port.owner_authorization_authority = None
        result = run_closure(port, self.closure_intent)
        self.assertEqual(result.reason_code, "OWNER_AUTHORIZATION_UNVERIFIED")

    def test_authorization_expiry_uses_trusted_execution_clock(self):
        closure_port = PrincipalClosureFixture()
        closure_port.now_utc = "2026-09-22T00:00:00Z"
        closure = run_closure(closure_port, self.closure_intent)
        self.assertEqual(
            closure.reason_code, "OWNER_AUTHORIZATION_UNVERIFIED"
        )
        self.assertNotIn("create-anchor", closure_port.calls)

        release_port = PrincipalReleaseFixture()
        release_port.now_utc = "2026-09-22T00:00:00Z"
        release = run_release(release_port, self.release_intent)
        self.assertEqual(release.reason_code, "OWNER_AUTHORIZATION_UNVERIFIED")
        self.assertNotIn("candidate", release_port.calls)

    def test_release_authorization_scope_failures_precede_candidate_write(self):
        for field, intent in (
            ("repository", replace(self.release_intent, repository="other/repository")),
            ("transaction", replace(self.release_intent, transaction_id="other-transaction")),
        ):
            with self.subTest(field=field):
                port = PrincipalReleaseFixture()
                result = run_release(port, intent)
                self.assertEqual((result.status, result.reason_code),
                                 ("BLOCKED", "OWNER_AUTHORIZATION_UNVERIFIED"))
                self.assertNotIn("candidate", port.calls)

        port = PrincipalReleaseFixture()
        port.authorization.owner = False
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code),
                         ("BLOCKED", "OWNER_AUTHORIZATION_UNVERIFIED"))
        self.assertNotIn("candidate", port.calls)

    def test_release_rejects_anchor_closure_timestamp_mismatch(self):
        port = PrincipalReleaseFixture()
        port.verify_anchor = lambda intent: {
            "ref": ANCHOR_REF,
            "sha": SHA_B,
            "tree": TREE,
            "closure_timestamp_utc": "2026-09-20T12:34:57Z",
        }
        result = run_release(port, self.release_intent)
        self.assertEqual(
            (result.status, result.reason_code),
            ("INVALID", "ANCHOR_IDENTITY_MISMATCH"),
        )
        self.assertNotIn("candidate", port.calls)

        port = PrincipalReleaseFixture()
        port.owner_authorization_ref_for = lambda action, target, version: (
            port.authorization.issue(action, "refs/heads/wrong-target", version)
        )
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code),
                         ("BLOCKED", "OWNER_AUTHORIZATION_UNVERIFIED"))
        self.assertNotIn("candidate", port.calls)

    def test_occupied_principal_sentinel_is_not_skipped_or_reused(self):
        port = PrincipalClosureFixture(occupied=frozenset({"0.3.1"}))
        result = run_closure(port, self.closure_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "PRINCIPAL_SENTINEL_UNAVAILABLE", True))
        self.assertNotIn("reservation-prepared", port.manifest_types)
        self.assertNotIn("unfreeze", port.calls)

    def test_release_happy_path_uses_exact_manifest_types_and_unfreezes_last(self):
        port = PrincipalReleaseFixture()
        result = run_release(port, self.release_intent)
        self.assertTrue(result.complete)
        self.assertFalse(result.frozen)
        self.assertEqual(port.manifest_types,
                         ["candidate-opened", "release-intent-prepared", "released", "publication"])
        self.assertLess(port.calls.index("tag"), port.calls.index("manifest:released"))
        self.assertLess(port.calls.index("manifest:released"), port.calls.index("manifest:publication"))
        self.assertLess(port.calls.index("publication-evidence-target"), port.calls.index("github-release"))
        self.assertLess(port.calls.index("publication-evidence-payload"), port.calls.index("github-release"))
        self.assertLess(port.calls.index("github-release"), port.calls.index("publication-evidence"))
        self.assertLess(port.calls.index("publication-evidence"), port.calls.index("publication-evidence-readback"))
        self.assertLess(port.calls.index("publication-evidence-readback"), port.calls.index("manifest:publication"))
        self.assertLess(port.calls.index("terminal"), port.calls.index("documentation-dispatch"))
        self.assertLess(port.calls.index("documentation-dispatch"), port.calls.index("release-exclusion"))
        self.assertEqual(port.calls[-2:], ["release-exclusion", "unfreeze-main"])
        self.assertEqual(
            port.manifests[-1]["publication_evidence_digest"],
            sha256(port.persisted_publication_evidence_payload).hexdigest(),
        )
        allocation_manifests = [
            manifest for manifest in port.manifests
            if manifest["manifest_type"] != "publication"
        ]
        self.assertEqual(
            [
                (
                    manifest["static_iteration_snapshot"],
                    manifest["lifecycle_ref_snapshot"],
                )
                for manifest in allocation_manifests
            ],
            [
                ("0" * 64, "1" * 64),
                ("0" * 64, "2" * 64),
                ("2" * 64, "3" * 64),
            ],
        )

    def test_principal_release_rejects_version_regression_before_tag(self):
        port = PrincipalReleaseFixture()
        port.freeze_main = lambda intent: {
            "token": "main-freeze",
            "sha": SHA_B,
            "version": "0.4.0",
        }
        result = run_release(port, self.release_intent)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "PRINCIPAL_VERSION_REGRESSION", True),
        )
        self.assertNotIn("tag", port.calls)

    def test_mandatory_closure_and_release_proofs_fail_closed(self):
        closure = run_closure(
            MissingClosureProofFixture("verify_closure_correspondence"),
            self.closure_intent,
        )
        self.assertEqual(
            closure.reason_code, "CLOSURE_CORRESPONDENCE_UNPROVEN"
        )
        self.assertTrue(closure.frozen)

        for missing, reason in (
            ("verify_principal_interval", "PRINCIPAL_ANCESTRY_UNPROVEN"),
            ("verify_released", "RELEASED_STATE_UNPROVEN"),
            ("verify_terminal", "TERMINAL_CONSISTENCY_UNPROVEN"),
        ):
            with self.subTest(missing=missing):
                port = MissingReleaseProofFixture(missing)
                result = run_release(port, self.release_intent)
                self.assertEqual(result.reason_code, reason)
                self.assertTrue(result.frozen)
                self.assertNotIn("unfreeze-main", port.calls)

    def test_unsupported_certification_binding_blocks_before_tag(self):
        port = PrincipalReleaseFixture(binding="content-bound")
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "UNSUPPORTED_CERTIFICATION_BINDING", False))
        self.assertNotIn("tag", port.calls)

    def test_local_certification_environment_is_rejected_before_intent_or_tag(self):
        port = PrincipalReleaseFixture()
        original = port.certify_candidate

        def private_environment(intent, candidate):
            certification = original(intent, candidate)
            certification["environment"] = {
                "schema_version": 1,
                "environment_id": "ci-linux",
                "local_hostname": "runner-17",
            }
            return certification

        port.certify_candidate = private_environment
        result = run_release(port, self.release_intent)
        self.assertEqual(result.reason_code, "CERTIFICATION_IDENTITY_UNPROVEN")
        self.assertNotIn("manifest:release-intent-prepared", port.calls)
        self.assertNotIn("tag", port.calls)
        self.assertNotIn("manifest:released", port.calls)
        self.assertFalse(any("certification_environment" in item for item in port.manifests))

    def test_evidence_references_require_duplicate_free_string_lists(self):
        port = PrincipalReleaseFixture()
        original_certify = port.certify_candidate

        def string_evidence(intent, candidate):
            record = original_certify(intent, candidate)
            record["evidence_refs"] = "evidence/candidate.json"
            return record

        port.certify_candidate = string_evidence
        result = run_release(port, self.release_intent)
        self.assertEqual(result.reason_code, "CERTIFICATION_IDENTITY_UNPROVEN")
        self.assertNotIn("tag", port.calls)

        port = PrincipalReleaseFixture()
        original_promote = port.promote_principal

        def mapping_evidence(intent, candidate, certification, freeze):
            final = original_promote(intent, candidate, certification, freeze)
            final["evidence_refs"] = {"ref": "evidence/released.json"}
            return final

        port.promote_principal = mapping_evidence
        result = run_release(port, self.release_intent)
        self.assertEqual(result.reason_code, "RELEASE_EVIDENCE_INVALID")
        self.assertNotIn("tag", port.calls)

    def test_missing_tag_protection_blocks_before_intent_or_tag(self):
        port = NoTagProtectionReleaseFixture()
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "PUBLIC_TAG_RULESET_UNAVAILABLE", True))
        self.assertNotIn("tag", port.calls)
        self.assertNotIn("manifest:release-intent-prepared", port.calls)

    def test_tree_bound_transfer_requires_equal_trees(self):
        port = PrincipalReleaseFixture()
        original = port.verify_tree_transfer

        def wrong_tree(intent, candidate, certification, final):
            evidence = original(intent, candidate, certification, final)
            evidence["final_release_tree"] = "e" * 40
            return evidence

        port.verify_tree_transfer = wrong_tree
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "CERTIFICATION_TRANSFER_UNPROVEN", True))
        self.assertNotIn("tag", port.calls)

    def test_commit_bound_certification_recertifies_final_commit(self):
        port = PrincipalReleaseFixture(binding="commit-bound")
        result = run_release(port, self.release_intent)
        self.assertTrue(result.complete)
        self.assertIn("recertify", port.calls)
        self.assertNotIn("tree-transfer", port.calls)

    def test_tag_failure_is_invalid_and_keeps_main_frozen(self):
        port = PrincipalReleaseFixture(fail="tag")
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("INVALID", "PUBLIC_TAG_IDENTITY_MISMATCH", True))
        self.assertNotIn("manifest:released", port.calls)
        self.assertNotIn("unfreeze-main", port.calls)

    def test_post_tag_failure_is_forward_reconciliation(self):
        port = PrincipalReleaseFixture(fail="released")
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("tag_reconciliation_pending", "PORT_OPERATION_FAILED", True))
        self.assertNotIn("unfreeze-main", port.calls)

    def test_post_release_failure_is_publication_reconciliation(self):
        port = PrincipalReleaseFixture(fail="publication-evidence")
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("publication_reconciliation_pending", "PORT_OPERATION_FAILED", True))
        self.assertNotIn("unfreeze-main", port.calls)

        port = PrincipalReleaseFixture(fail="documentation-dispatch")
        result = run_release(port, self.release_intent)
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("publication_reconciliation_pending", "PORT_OPERATION_FAILED", True))
        self.assertIn("manifest:publication", port.calls)
        self.assertIn("terminal", port.calls)
        self.assertTrue(result.evidence["documentation_dispatch_reconciliation_required"])
        self.assertNotIn("release-exclusion", port.calls)
        self.assertNotIn("unfreeze-main", port.calls)

        port = PrincipalReleaseFixture(fail="documentation-dispatch-mismatch")
        result = run_release(port, self.release_intent)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen, result.phase),
            ("publication_reconciliation_pending", "DOCS_DEPLOY_DISPATCH_MISMATCH", True, "terminal_verified"),
        )
        self.assertTrue(result.evidence["documentation_dispatch_reconciliation_required"])
        self.assertNotIn("release-exclusion", port.calls)
        self.assertNotIn("unfreeze-main", port.calls)

    def test_publication_manifest_requires_exact_durable_evidence_readback(self):
        for failure in (
            "publication-evidence-no-store",
            "publication-evidence-readback-mismatch",
        ):
            with self.subTest(failure=failure):
                port = PrincipalReleaseFixture(fail=failure)
                result = run_release(port, self.release_intent)
                self.assertEqual(
                    (result.status, result.reason_code, result.frozen),
                    (
                        "publication_reconciliation_pending",
                        "PUBLICATION_EVIDENCE_READBACK_UNPROVEN",
                        True,
                    ),
                )
                self.assertIn("publication-evidence-readback", port.calls)
                self.assertNotIn("manifest:publication", port.calls)
                self.assertNotIn("terminal", port.calls)
                self.assertNotIn("documentation-dispatch", port.calls)
                self.assertNotIn("release-exclusion", port.calls)
                self.assertNotIn("unfreeze-main", port.calls)
                self.assertTrue(
                    result.evidence[
                        "publication_evidence_readback_reconciliation_required"
                    ]
                )

    def test_missing_publication_evidence_readback_port_blocks_publication(self):
        port = MissingReleaseProofFixture("read_publication_evidence")
        result = run_release(port, self.release_intent)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("publication_reconciliation_pending", "PUBLICATION_EVIDENCE_UNAVAILABLE", True),
        )
        self.assertNotIn("github-release", port.calls)
        self.assertNotIn("manifest:publication", port.calls)
        self.assertNotIn("unfreeze-main", port.calls)

    def test_publication_evidence_preflight_blocks_invalid_values_before_persist(self):
        for failure, reason in (
            ("unsafe-publication-evidence-ref", "PUBLICATION_EVIDENCE_TARGET_INVALID"),
            ("unsafe-publication-evidence-payload", "PUBLICATION_EVIDENCE_MISMATCH"),
        ):
            with self.subTest(failure=failure):
                port = PrincipalReleaseFixture(fail=failure)
                result = run_release(port, self.release_intent)
                self.assertEqual(
                    (result.status, result.reason_code, result.frozen),
                    ("publication_reconciliation_pending" if failure == "unsafe-publication-evidence-ref" else "INVALID", reason, True),
                )
                self.assertNotIn("publication-evidence", port.calls)
                self.assertNotIn("github-release", port.calls)
                self.assertNotIn("manifest:publication", port.calls)
                self.assertNotIn("unfreeze-main", port.calls)

    def test_contradictory_publication_evidence_identity_is_rejected_before_publication(self):
        port = PrincipalReleaseFixture()
        port.publication_evidence_override = {
            "schema_version": 1,
            "public_tag": "v9.9.9",
            "released_manifest_id": "LIF-SHA256-" + "a" * 64,
            "released_manifest_ref": "refs/heads/lifecycle/v1/releases/v0.3.0",
            "released_manifest_digest": "b" * 64,
            "tag_commit": SHA_A,
            "tag_tree": TREE,
        }
        result = run_release(port, self.release_intent)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("INVALID", "PUBLICATION_EVIDENCE_MISMATCH", True),
        )
        self.assertNotIn("github-release", port.calls)
        self.assertNotIn("publication-evidence", port.calls)
        self.assertNotIn("manifest:publication", port.calls)
        self.assertNotIn("unfreeze-main", port.calls)

        port = PrincipalReleaseFixture()
        original_payload = port.publication_evidence_payload

        def floating_schema_version(intent, released, tag):
            payload = original_payload(intent, released, tag)
            payload["schema_version"] = 1.0
            return payload

        port.publication_evidence_payload = floating_schema_version
        result = run_release(port, self.release_intent)
        self.assertEqual(result.reason_code, "PUBLICATION_EVIDENCE_MISMATCH")
        self.assertNotIn("github-release", port.calls)
        self.assertNotIn("publication-evidence", port.calls)

    def test_maintenance_bootstrap_and_rare_recovery_are_deferred(self):
        bootstrap = run_maintenance_bootstrap(
            object(), BootstrapIntent("maintenance/1.2", "1.2.0")
        )
        recovery = run_rare_recovery(object())
        self.assertEqual(bootstrap.reason_code, "DEFERRED_MAINTENANCE_BOOTSTRAP")
        self.assertEqual(recovery.reason_code, "DEFERRED_RARE_RECOVERY")
        self.assertFalse(bootstrap.frozen)
        self.assertFalse(recovery.frozen)


if __name__ == "__main__":
    unittest.main()
