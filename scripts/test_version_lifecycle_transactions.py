"""Observable closure/bootstrap ordering and freeze recovery fixtures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.git_refs import ProtectionEvidence  # noqa: E402
from version_lifecycle.transactions import (  # noqa: E402
    AllocationView,
    BootstrapIntent,
    ClosureIntent,
    ReleaseIntent,
    run_closure,
    run_maintenance_bootstrap,
    run_release,
)


class ClosureFixture:
    def __init__(
        self, *, fail_consumption=False, occupied=frozenset(),
        closure_fields=None, anchor_fields=None, exclusion_available=True,
    ) -> None:
        self.calls: list[str] = []
        self.fail_consumption = fail_consumption
        self.occupied = occupied
        self.exclusion_available = exclusion_available
        self.closure_fields = closure_fields or {}
        self.anchor_fields = anchor_fields or {}
        self.views: list[AllocationView] = []
        self.consumed_view: AllocationView | None = None

    def freeze_line(self, intent):
        self.calls.append("freeze")
        return "freeze-proof"

    def acquire_static_mutation(self, intent):
        if not self.exclusion_available:
            raise RuntimeError("racing static mutation")
        return "fixture-exclusion"

    def release_static_mutation(self, lease):
        if lease != "fixture-exclusion":
            raise AssertionError("unexpected exclusion lease")

    def allocation_view(self):
        self.calls.append("view")
        view = AllocationView(
            f"snapshot-{len(self.views)}", f"event-head-{len(self.views)}", self.occupied
        )
        self.views.append(view)
        return view

    def verify_closure_target(self, intent, view):
        self.calls.append("target")

    def merge_final(self, intent):
        self.calls.append("merge")
        result = {
            "version": intent.final_version,
            "commit": "a" * 40,
            "tree": "b" * 40,
        }
        result.update(self.closure_fields)
        return result

    def create_anchor(self, intent, closure):
        self.calls.append("anchor")
        result = {
            "version": intent.final_version, "commit": closure["commit"],
            "tree": closure["tree"],
            "closure_timestamp_utc": intent.closure_timestamp_utc,
        }
        result.update(self.anchor_fields)
        return result

    def consume_outgoing(self, intent, anchor, view):
        self.calls.append("consume")
        self.consumed_view = view
        if self.fail_consumption:
            raise RuntimeError("uncertain append")
        return {
            "reserved_final": intent.outgoing_reserved_final,
            "closed_final_version": intent.final_version,
            "terminal_disposition": (
                "closed" if intent.outgoing_reserved_final == intent.final_version
                else "CONSUMED_UNUSED_DEV_RESERVATION"
            ),
            "event_id": "EVT-000000000002",
        }

    def verify_outgoing_terminal(self, intent, consumption):
        self.calls.append("terminal")

    def prepare_next(self, intent, next_final, view):
        self.calls.append("prepare")
        return {"version": next_final, "intended_dev": next_final + "-DEV"}

    def reopen_dev(self, intent, preparation):
        self.calls.append("reopen")
        return {"version": preparation["intended_dev"], "head": "reopened"}

    def activate_next(self, intent, preparation, reopened):
        self.calls.append("activate")
        return {"head": reopened["head"]}

    def verify_closure_correspondence(self, *args):
        self.calls.append("correspondence")

    def unfreeze_line(self, token):
        self.calls.append("unfreeze")


class BootstrapFixture:
    def __init__(self, branch_state="created") -> None:
        self.calls: list[str] = []
        self.branch_state = branch_state

    def verify_base_and_absence(self, intent):
        self.calls.append("base")

    def freeze_bootstrap(self, intent):
        self.calls.append("freeze")
        return "freeze-proof"

    def allocation_view(self):
        self.calls.append("view")
        return AllocationView("snapshot", "event-head", frozenset({"1.2.0", "1.2.1"}))

    def prepare_reservation(self, intent, final, view):
        self.calls.append("prepare")
        return {"version": final, "intended_dev": final + "-DEV"}

    def create_line_if_absent(self, intent, preparation):
        self.calls.append("create")
        return {"state": self.branch_state, "head": "base" if self.branch_state == "created" else ""}

    def abort_nonentry(self, intent, preparation, branch):
        self.calls.append("abort")

    def install_dev(self, intent, branch, preparation):
        self.calls.append("install")
        return {"version": preparation["intended_dev"], "head": "dev-head"}

    def activate_reservation(self, intent, preparation, dev_head):
        self.calls.append("activate")
        return {"head": dev_head["head"]}

    def record_line_opened(self, intent, branch, activation):
        self.calls.append("line-open")
        return {"event_id": "EVT-000000000003"}

    def verify_bootstrap_correspondence(self, *args):
        self.calls.append("correspondence")

    def unfreeze_bootstrap(self, token):
        self.calls.append("unfreeze")


class ReleaseFixture:
    def __init__(
        self, *, fail_after_tag=False, fail_publication=False, bad_tag=False,
        exclusion_available=True,
    ):
        self.calls: list[str] = []
        self.fail_after_tag = fail_after_tag
        self.fail_publication = fail_publication
        self.bad_tag = bad_tag
        self.exclusion_available = exclusion_available
        self.candidate_sha = "a" * 40
        self.final_sha = "b" * 40
        self.tree = "c" * 40
        self.intent_certification = None
        self.released_certification = None

    def acquire_static_mutation(self, intent):
        if not self.exclusion_available:
            raise RuntimeError("racing static mutation")
        return "fixture-exclusion"

    def release_static_mutation(self, lease):
        if lease != "fixture-exclusion":
            raise AssertionError("unexpected exclusion lease")

    def verify_anchor(self, intent):
        self.calls.append("anchor")

    def make_durable_candidate(self, intent):
        self.calls.append("candidate")
        return {"ref": intent.candidate_ref, "sha": self.candidate_sha,
                "tree": self.tree, "version": intent.final_version, "durable": True,
                "main_at_candidate_sha": "d" * 40,
                "main_at_candidate_version": "0.2.0"}

    def append_candidate_opened(self, intent, candidate):
        self.calls.append("opened")
        return {"candidate_sha": candidate["sha"], "event_id": "EVT-000000000001"}

    def certify_candidate(self, intent, candidate):
        self.calls.append("certify")
        return {"binding": "tree-bound", "subject_sha": candidate["sha"],
                "subject_tree": candidate["tree"], "policy_revision": "policy",
                "harness_revision": "harness", "environment": "env",
                "evidence_refs": ["evidence"]}

    def freeze_main(self, intent):
        self.calls.append("freeze-main")
        return {"token": "freeze", "sha": "d" * 40, "version": "0.2.0"}

    def verify_principal_interval(self, intent, candidate, freeze):
        self.calls.append("verify-interval")
        return {"verified": True,
                "candidate_main_sha": candidate["main_at_candidate_sha"],
                "freeze_main_sha": freeze["sha"],
                "candidate_main_tree": "e" * 40,
                "freeze_main_tree": "e" * 40,
                "intervening_commits": [],
                "disposition": "no_drift"}

    def promote_principal(self, intent, candidate, certification, freeze):
        self.calls.append("promote")
        return {"sha": self.final_sha, "tree": self.tree,
                "version": intent.final_version,
                "previous_main_sha": freeze["sha"],
                "previous_main_version": freeze["version"],
                "main_at_event_sha": self.final_sha,
                "main_at_event_version": intent.final_version,
                "ancestry_disposition": "no_drift"}

    def verify_maintenance(self, intent, candidate, certification):
        self.calls.append("maintenance")
        return {"sha": self.final_sha, "tree": self.tree,
                "version": intent.final_version,
                "main_before_sha": "d" * 40,
                "main_at_event_sha": "d" * 40,
                "main_before_version": "1.0.0",
                "main_at_event_version": "1.0.0"}

    def recertify_final(self, intent, final):
        self.calls.append("recertify")
        return {"binding": "commit-bound", "subject_sha": final["sha"],
                "subject_tree": final["tree"], "evidence_refs": ["evidence"],
                "policy_revision": "policy", "harness_revision": "harness",
                "environment": "env"}

    def verify_tree_transfer(self, intent, candidate, certification, final):
        self.calls.append("verify-transfer")
        return {"verified": True, "candidate_sha": candidate["sha"],
                "final_release_sha": final["sha"],
                "candidate_tree": candidate["tree"],
                "final_release_tree": final["tree"],
                "anchor_tree": intent.anchor_tree,
                "evidence_ref": "evidence/transfer.json"}

    def append_release_intent(self, intent, candidate, certification, final):
        self.calls.append("intent")
        self.intent_certification = dict(certification)
        result = {"public_tag": "v" + intent.final_version,
                  "release_sha": final["sha"], "release_tree": final["tree"],
                  "event_id": "EVT-000000000002"}
        if "transfer_evidence" in certification:
            result["certification_transfer_evidence"] = dict(
                certification["transfer_evidence"]
            )
        return result

    def verify_public_tag_ruleset(self, intent, tag_ref):
        self.calls.append("ruleset")
        major = tag_ref.removeprefix("refs/tags/v").split(".", 1)[0]
        return ProtectionEvidence(
            rule_id="fixture-future-tags",
            pattern=f"refs/tags/v{major}.*",
            snapshot_sha256="0" * 64,
            retrieved_at_utc="2026-09-20T00:00:00Z",
            creation_guarded=True,
            update_guarded=True,
            deletion_guarded=True,
        )

    def create_protected_tag(self, intent, prepared, final, protection):
        self.calls.append("tag")
        protection.require_public_tag(f"refs/tags/{prepared['public_tag']}")
        return {"name": prepared["public_tag"], "commit": "0" * 40 if self.bad_tag else final["sha"],
                "tree": final["tree"], "protected": True}

    def append_released(self, intent, candidate, certification, final, tag):
        self.calls.append("released")
        if self.fail_after_tag:
            raise RuntimeError("transport timeout")
        self.released_certification = dict(certification)
        result = {"event_id": "EVT-000000000003", "public_tag": tag["name"]}
        if "transfer_evidence" in certification:
            result["certification_transfer_evidence"] = dict(
                certification["transfer_evidence"]
            )
        return result

    def verify_released(self, intent, event, certification, final):
        self.calls.append("verify-released")

    def publish_github_release(self, intent, event):
        self.calls.append("publish")
        if self.fail_publication:
            raise RuntimeError("publication timeout")
        return {"id": "release-1", "tag": event["public_tag"]}

    def persist_publication_evidence(self, intent, event, publication):
        self.calls.append("publication-evidence")
        return {"event_id": event["event_id"],
                "public_tag": event["public_tag"],
                "github_release_id": publication["id"]}

    def verify_terminal(self, intent, event, publication, evidence):
        self.calls.append("terminal")

    def unfreeze_main(self, token):
        self.calls.append("unfreeze-main")


class TransactionTests(unittest.TestCase):
    closure = ClosureIntent(
        "tx", "principal", "0.3.0", "0.3.0", "old-head", "2026-09-20T12:34:56Z"
    )
    bootstrap = BootstrapIntent("tx", "maintenance/1.2", "base", "main", "2.0.0")
    release = ReleaseIntent(
        "tx", "principal", "0.3.0", "refs/tags/iterations/0.3.0",
        "f" * 40, "c" * 40, "2026-09-20T12:34:56Z",
        "refs/heads/candidates/0.3.0",
    )

    def test_successful_closure_unfreezes_only_after_correspondence(self):
        port = ClosureFixture()
        result = run_closure(port, self.closure)
        self.assertEqual((result.status, result.frozen, result.evidence["next_final"]),
                         ("COMPLETE", False, "0.3.1"))
        self.assertEqual(port.calls, [
            "freeze", "view", "target", "merge", "anchor", "view", "consume", "terminal",
            "view", "prepare", "reopen", "activate", "correspondence", "unfreeze",
        ])

    def test_closure_consumption_uses_post_anchor_allocation_view(self):
        port = ClosureFixture()
        result = run_closure(port, self.closure)
        self.assertIs(port.consumed_view, result.evidence["post_anchor_view"])
        self.assertIsNot(port.consumed_view, result.evidence["bound_view"])

    def test_failed_outgoing_consumption_never_allocates_or_unfreezes(self):
        port = ClosureFixture(fail_consumption=True)
        result = run_closure(port, self.closure)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "BLOCKED", "OUTGOING_RESERVATION_RECONCILIATION_FAILED", True
        ))
        self.assertNotIn("prepare", port.calls)
        self.assertNotIn("unfreeze", port.calls)

    def test_racing_static_mutation_blocks_closure_under_line_freeze(self):
        port = ClosureFixture(exclusion_available=False)
        result = run_closure(port, self.closure)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "EXCLUSION_UNAVAILABLE", True),
        )
        self.assertEqual(port.calls, ["freeze"])
        self.assertNotIn("merge", port.calls)
        self.assertNotIn("unfreeze", port.calls)

    def test_closure_and_anchor_require_complete_matching_git_identities(self):
        cases = [
            ("closure", "commit", None, "CLOSURE_IDENTITY_MISMATCH"),
            ("closure", "commit", "not-a-sha", "CLOSURE_IDENTITY_MISMATCH"),
            ("closure", "tree", None, "CLOSURE_IDENTITY_MISMATCH"),
            ("closure", "tree", "not-a-sha", "CLOSURE_IDENTITY_MISMATCH"),
            ("anchor", "commit", None, "ANCHOR_IDENTITY_MISMATCH"),
            ("anchor", "commit", "not-a-sha", "ANCHOR_IDENTITY_MISMATCH"),
            ("anchor", "tree", None, "ANCHOR_IDENTITY_MISMATCH"),
            ("anchor", "tree", "not-a-sha", "ANCHOR_IDENTITY_MISMATCH"),
        ]
        for stage, field, value, reason_code in cases:
            with self.subTest(stage=stage, field=field, value=value):
                fields = {field: value}
                port = ClosureFixture(
                    closure_fields=fields if stage == "closure" else {},
                    anchor_fields=fields if stage == "anchor" else {},
                )
                result = run_closure(port, self.closure)
                self.assertEqual(
                    (result.status, result.reason_code, result.frozen),
                    ("BLOCKED", reason_code, True),
                )
                self.assertNotIn("unfreeze", port.calls)

    def test_different_final_consumes_unused_dev_before_reopen(self):
        intent = ClosureIntent(
            "different-final", "principal", "0.3.2", "0.3.1",
            "old-head", "2026-09-20T12:34:56Z",
        )
        port = ClosureFixture()
        result = run_closure(port, intent)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(result.evidence["next_final"], "0.3.3")
        self.assertEqual(
            result.evidence["consumption"]["terminal_disposition"],
            "CONSUMED_UNUSED_DEV_RESERVATION",
        )
        self.assertLess(port.calls.index("terminal"), port.calls.index("prepare"))

    def test_different_final_requires_exact_unused_disposition(self):
        intent = ClosureIntent(
            "different-final", "principal", "0.3.2", "0.3.1",
            "old-head", "2026-09-20T12:34:56Z",
        )
        port = ClosureFixture()
        original = port.consume_outgoing
        def wrong_disposition(intent, anchor, view):
            result = original(intent, anchor, view)
            result["terminal_disposition"] = "closed"
            return result
        port.consume_outgoing = wrong_disposition
        result = run_closure(port, intent)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "OUTGOING_RESERVATION_RECONCILIATION_FAILED", True),
        )
        self.assertNotIn("prepare", port.calls)

    def test_exact_principal_sentinel_cannot_skip_occupied_patch(self):
        port = ClosureFixture(occupied=frozenset({"0.3.1"}))
        result = run_closure(port, self.closure)
        self.assertEqual((result.status, result.reason_code), (
            "BLOCKED", "PRINCIPAL_SENTINEL_UNAVAILABLE"
        ))
        self.assertNotIn("prepare", port.calls)

    def test_bootstrap_from_first_available_patch(self):
        port = BootstrapFixture()
        result = run_maintenance_bootstrap(port, self.bootstrap)
        self.assertEqual((result.status, result.evidence["preparation"]["version"]),
                         ("COMPLETE", "1.2.2"))
        self.assertEqual(port.calls[:4], ["base", "freeze", "base", "view"])
        self.assertEqual(port.calls[-2:], ["correspondence", "unfreeze"])

    def test_uncertain_branch_creation_keeps_prepared_version_unavailable(self):
        port = BootstrapFixture("uncertain")
        result = run_maintenance_bootstrap(port, self.bootstrap)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "BLOCKED", "BOOTSTRAP_CREATION_UNCERTAIN", True
        ))
        self.assertNotIn("abort", port.calls)
        self.assertNotIn("install", port.calls)
        self.assertNotIn("unfreeze", port.calls)

    def test_definite_noncreation_allows_abort_but_keeps_freeze(self):
        port = BootstrapFixture("not_created")
        result = run_maintenance_bootstrap(port, self.bootstrap)
        self.assertEqual(result.reason_code, "BOOTSTRAP_BRANCH_NOT_CREATED")
        self.assertIn("abort", port.calls)
        self.assertNotIn("unfreeze", port.calls)

    def test_release_publication_unfreezes_only_after_terminal_proof(self):
        port = ReleaseFixture()
        result = run_release(port, self.release)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(port.calls[-3:], ["publication-evidence", "terminal", "unfreeze-main"])
        self.assertLess(port.calls.index("intent"), port.calls.index("tag"))
        self.assertLess(port.calls.index("ruleset"), port.calls.index("intent"))
        self.assertLess(port.calls.index("tag"), port.calls.index("released"))
        self.assertLess(port.calls.index("verify-interval"), port.calls.index("promote"))
        self.assertLess(port.calls.index("verify-transfer"), port.calls.index("tag"))
        self.assertEqual(
            result.evidence["public_tag_ruleset"]["rule_id"],
            "fixture-future-tags",
        )

    def test_release_blocks_before_intent_without_canonical_public_tag_ruleset(self):
        for failure in ("missing", "legacy-inclusive", "creation", "update", "deletion"):
            with self.subTest(failure=failure):
                port = ReleaseFixture()
                original = port.verify_public_tag_ruleset

                def bad_ruleset(intent, tag_ref, failure=failure):
                    if failure == "missing":
                        return None
                    proof = original(intent, tag_ref)
                    values = dict(proof.__dict__)
                    if failure == "legacy-inclusive":
                        values["pattern"] = "refs/tags/v*"
                    else:
                        values[f"{failure}_guarded"] = False
                    return ProtectionEvidence(**values)

                port.verify_public_tag_ruleset = bad_ruleset
                result = run_release(port, self.release)
                self.assertEqual(
                    (result.status, result.reason_code, result.frozen),
                    ("BLOCKED", "PUBLIC_TAG_RULESET_UNAVAILABLE", True),
                )
                self.assertNotIn("intent", port.calls)
                self.assertNotIn("tag", port.calls)

    def test_racing_static_mutation_blocks_release_before_candidate(self):
        port = ReleaseFixture(exclusion_available=False)
        result = run_release(port, self.release)
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "EXCLUSION_UNAVAILABLE", False),
        )
        self.assertEqual(port.calls, ["anchor"])
        self.assertNotIn("candidate", port.calls)
        self.assertNotIn("tag", port.calls)

    def test_tree_transfer_proof_is_bound_to_both_release_events(self):
        port = ReleaseFixture()
        result = run_release(port, self.release)
        self.assertEqual(result.status, "COMPLETE")
        proof = result.evidence["certification_transfer"]
        self.assertEqual(result.evidence["certification"]["transfer_evidence"], proof)
        self.assertEqual(result.evidence["final_certification"]["transfer_evidence"], proof)
        self.assertEqual(port.intent_certification["transfer_evidence"], proof)
        self.assertEqual(port.released_certification["transfer_evidence"], proof)
        self.assertEqual(result.evidence["intent"]["certification_transfer_evidence"], proof)
        self.assertEqual(result.evidence["released"]["certification_transfer_evidence"], proof)

    def test_missing_durable_transfer_proof_blocks_before_public_tag(self):
        port = ReleaseFixture()
        original = port.append_release_intent

        def missing_transfer(intent, candidate, certification, final):
            result = original(intent, candidate, certification, final)
            result.pop("certification_transfer_evidence", None)
            return result

        port.append_release_intent = missing_transfer
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "BLOCKED", "RELEASE_INTENT_TRANSFER_EVIDENCE_UNPROVEN", True
        ))
        self.assertNotIn("tag", port.calls)

    def test_missing_ancestry_or_transfer_proof_blocks_public_tag(self):
        for missing in ("interval", "transfer"):
            with self.subTest(missing=missing):
                port = ReleaseFixture()
                if missing == "interval":
                    port.verify_principal_interval = lambda *args: {}
                    reason = "PRINCIPAL_ANCESTRY_UNPROVEN"
                else:
                    port.verify_tree_transfer = lambda *args: {}
                    reason = "CERTIFICATION_TRANSFER_UNPROVEN"
                result = run_release(port, self.release)
                self.assertEqual((result.status, result.reason_code, result.frozen), (
                    "BLOCKED", reason, True
                ))
                self.assertNotIn("tag", port.calls)

    def test_tree_bound_transfer_is_required_when_candidate_and_final_differ(self):
        port = ReleaseFixture()
        original = port.certify_candidate

        def incorrectly_final_certified(intent, candidate):
            result = original(intent, candidate)
            result["subject_sha"] = port.final_sha
            return result

        port.certify_candidate = incorrectly_final_certified
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "BLOCKED", "CERTIFICATION_IDENTITY_UNPROVEN", False
        ))
        self.assertNotIn("verify-transfer", port.calls)
        self.assertNotIn("tag", port.calls)

    def test_commit_bound_recertification_accepts_updated_reviewed_pins(self):
        port = ReleaseFixture()

        def commit_bound_certification(intent, candidate):
            return {
                "binding": "commit-bound",
                "subject_sha": candidate["sha"],
                "subject_tree": candidate["tree"],
                "policy_revision": "policy",
                "harness_revision": "harness",
                "environment": "env",
                "evidence_refs": ["evidence/candidate-certification.json"],
            }

        port.certify_candidate = commit_bound_certification
        original = port.recertify_final

        def recertified_with_updated_pins(intent, final):
            result = original(intent, final)
            result.update({
                "policy_revision": "policy-2026-10",
                "harness_revision": "harness-2026-10",
                "environment": "julia-1.13-python-3.14",
                "evidence_refs": ["evidence/final-certification.json"],
            })
            return result

        port.recertify_final = recertified_with_updated_pins
        result = run_release(port, self.release)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(
            result.evidence["candidate_certification"]["policy_revision"], "policy"
        )
        self.assertEqual(
            result.evidence["final_certification"]["policy_revision"], "policy-2026-10"
        )
        self.assertEqual(
            result.evidence["final_certification"]["harness_revision"], "harness-2026-10"
        )
        self.assertEqual(
            result.evidence["final_certification"]["environment"],
            "julia-1.13-python-3.14",
        )

    def test_commit_bound_recertification_requires_valid_final_pins(self):
        for field in ("policy_revision", "harness_revision", "environment"):
            for case, value in (("missing", None), ("invalid", 123)):
                with self.subTest(field=field, case=case):
                    port = ReleaseFixture()

                    def commit_bound_certification(intent, candidate):
                        return {
                            "binding": "commit-bound",
                            "subject_sha": candidate["sha"],
                            "subject_tree": candidate["tree"],
                            "policy_revision": "policy",
                            "harness_revision": "harness",
                            "environment": "env",
                            "evidence_refs": ["evidence"],
                        }

                    port.certify_candidate = commit_bound_certification
                    original = port.recertify_final

                    def recertified_with_bad_pin(
                        intent, final, field=field, case=case, value=value
                    ):
                        result = original(intent, final)
                        if case == "missing":
                            result.pop(field, None)
                        else:
                            result[field] = value
                        return result

                    port.recertify_final = recertified_with_bad_pin
                    result = run_release(port, self.release)
                    self.assertEqual((result.status, result.reason_code, result.frozen), (
                        "BLOCKED", "RECERTIFICATION_REQUIRED", True
                    ))
                    self.assertNotIn("intent", port.calls)
                    self.assertNotIn("tag", port.calls)

    def test_principal_interval_requires_tree_comparison_for_no_drift(self):
        port = ReleaseFixture()
        original = port.verify_principal_interval

        def mismatched_trees(intent, candidate, freeze):
            result = original(intent, candidate, freeze)
            result["freeze_main_tree"] = "f" * 40
            return result

        port.verify_principal_interval = mismatched_trees
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "BLOCKED", "PRINCIPAL_ANCESTRY_UNPROVEN", True
        ))
        self.assertNotIn("promote", port.calls)
        self.assertNotIn("tag", port.calls)

    def test_principal_interval_requires_each_intervening_tree_record(self):
        port = ReleaseFixture()
        original = port.verify_principal_interval

        def missing_tree_record(intent, candidate, freeze):
            result = original(intent, candidate, freeze)
            result["intervening_commits"] = [{"sha": "f" * 40}]
            result["disposition"] = "tree_neutral_included"
            return result

        port.verify_principal_interval = missing_tree_record
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "BLOCKED", "PRINCIPAL_ANCESTRY_UNPROVEN", True
        ))
        self.assertNotIn("promote", port.calls)
        self.assertNotIn("tag", port.calls)

    def test_principal_regression_stops_before_main_promotion(self):
        port = ReleaseFixture()
        port.freeze_main = lambda intent: {
            "token": "freeze", "sha": "d" * 40, "version": "0.3.0"
        }
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "BLOCKED", "PRINCIPAL_VERSION_REGRESSION", True
        ))
        self.assertNotIn("promote", port.calls)
        self.assertNotIn("tag", port.calls)

    def test_post_tag_event_failure_is_forward_only(self):
        port = ReleaseFixture(fail_after_tag=True)
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.frozen),
                         ("tag_reconciliation_pending", True))
        self.assertNotIn("unfreeze-main", port.calls)
        self.assertNotIn("publish", port.calls)

    def test_post_event_publication_failure_is_forward_only(self):
        port = ReleaseFixture(fail_publication=True)
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.frozen),
                         ("publication_reconciliation_pending", True))
        self.assertNotIn("unfreeze-main", port.calls)

    def test_irreversible_mismatched_tag_is_invalid(self):
        port = ReleaseFixture(bad_tag=True)
        result = run_release(port, self.release)
        self.assertEqual((result.status, result.reason_code, result.frozen), (
            "INVALID", "PUBLIC_TAG_IDENTITY_MISMATCH", True
        ))
        self.assertNotIn("released", port.calls)
        self.assertNotIn("unfreeze-main", port.calls)

    def test_maintenance_release_does_not_freeze_or_change_main(self):
        port = ReleaseFixture()
        intent = ReleaseIntent(
            "tx", "maintenance/0.3", "0.3.1",
            "refs/tags/iterations/0.3.1", "f" * 40, "c" * 40,
            "2026-09-20T12:34:56Z", "refs/heads/candidates/0.3.1",
        )
        result = run_release(port, intent)
        self.assertEqual(result.status, "COMPLETE")
        self.assertNotIn("freeze-main", port.calls)
        self.assertIn("maintenance", port.calls)


if __name__ == "__main__":
    unittest.main()
