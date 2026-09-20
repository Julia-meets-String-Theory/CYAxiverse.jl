"""Focused canonical event and append-only writer tests for CYAX-0125."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.events import (  # noqa: E402
    EventCanonicalEncodingError,
    EventIdExhausted,
    EventSchemaError,
    EventTransitionError,
    MAX_EVENT_SEQUENCE,
    append_only_prefix,
    canonical_event_bytes,
    event_id,
    next_event_id,
    parse_stream,
    validate_event,
    validate_transition,
)
from version_lifecycle.writer import (  # noqa: E402
    AppendOutcomeUncertain,
    DuplicateTransactionError,
    ReleaseEventWriter,
    StaleHeadError,
)


SNAPSHOT = "a" * 64


def reservation_event(
    event_id_value: str = "EVT-000000000001",
    *,
    transaction_id: str = "tx-1",
    expected_head: str = "",
    event_type: str = "development_reservation_prepared",
) -> dict:
    event = {
        "schema_version": 1,
        "event_id": event_id_value,
        "event_type": event_type,
        "timestamp_utc": "2026-09-20T12:34:56Z",
        "transaction_id": transaction_id,
        "static_iteration_snapshot": SNAPSHOT,
        "expected_event_head": expected_head,
        "owner_line": "principal",
        "final_version": "0.3.1",
        "intended_dev_version": "0.3.1-DEV",
        "expected_line_head": "b" * 40,
        "reservation_id": "reservation-1",
    }
    if event_type == "development_reservation_opened":
        event.pop("expected_line_head")
        event["actual_dev_head"] = "c" * 40
    return event


def candidate_event(
    event_id_value: str = "EVT-000000000001",
    *,
    transaction_id: str = "candidate-1",
    expected_head: str = "b" * 40,
    candidate_id: str = "candidate-1",
) -> dict:
    return {
        "schema_version": 1,
        "event_id": event_id_value,
        "event_type": "candidate_opened",
        "timestamp_utc": "2026-09-20T12:34:56Z",
        "transaction_id": transaction_id,
        "static_iteration_snapshot": SNAPSHOT,
        "expected_event_head": expected_head,
        "candidate_id": candidate_id,
        "candidate_ref": "refs/heads/candidates/0.3.1",
        "candidate_sha": "c" * 40,
        "candidate_tree": "f" * 40,
        "final_version": "0.3.1",
        "release_line": "principal",
        "anchor_ref": "refs/tags/iterations/0.3.1",
        "anchor_sha": "e" * 40,
        "anchor_tree": "f" * 40,
    }


def release_intent_event(
    event_id_value: str = "EVT-000000000002",
    *,
    transaction_id: str = "intent-1",
    intent_id: str = "intent-1",
    candidate_id: str = "candidate-1",
    expected_head: str = "b" * 40,
) -> dict:
    return {
        "schema_version": 1,
        "event_id": event_id_value,
        "event_type": "release_intent_prepared",
        "timestamp_utc": "2026-09-20T12:34:56Z",
        "transaction_id": transaction_id,
        "static_iteration_snapshot": SNAPSHOT,
        "expected_event_head": expected_head,
        "intent_id": intent_id,
        "candidate_id": candidate_id,
        "candidate_ref": "refs/heads/candidates/0.3.1",
        "candidate_sha": "c" * 40,
        "candidate_tree": "f" * 40,
        "anchor_ref": "refs/tags/iterations/0.3.1",
        "anchor_sha": "e" * 40,
        "anchor_tree": "f" * 40,
        "certification_binding": "tree-bound",
        "certification_subject_sha": "f" * 40,
        "certification_subject_tree": "f" * 40,
        "certification_policy_revision": "policy-r1",
        "certification_harness_revision": "harness-r1",
        "certification_environment": "ci-linux",
        "certification_evidence_refs": ["evidence/certification-1"],
        "final_release_sha": "f" * 40,
        "final_release_tree": "f" * 40,
        "final_version": "0.3.1",
        "release_line": "principal",
        "public_tag": "v0.3.1",
    }


def release_intent_abort_event(
    event_id_value: str = "EVT-000000000006",
    *,
    transaction_id: str = "abort-1",
    intent_id: str = "intent-1",
    candidate_id: str = "candidate-1",
    expected_head: str = "b" * 40,
) -> dict:
    return {
        "schema_version": 1,
        "event_id": event_id_value,
        "event_type": "release_intent_aborted",
        "timestamp_utc": "2026-09-20T12:34:56Z",
        "transaction_id": transaction_id,
        "static_iteration_snapshot": SNAPSHOT,
        "expected_event_head": expected_head,
        "intent_id": intent_id,
        "candidate_id": candidate_id,
        "public_tag": "v0.3.1",
        "no_public_tag_evidence": {
            "public_tag": "v0.3.1",
            "tag_absent": True,
            "checked_at_utc": "2026-09-20T12:34:56Z",
            "exclusion_verified": True,
            "exclusion_snapshot": "d" * 64,
        },
    }


class EventValidationTests(unittest.TestCase):
    def test_event_is_canonical_and_jsonl_framing_is_exact(self):
        event = dict(reservation_event(), expected_event_head="b" * 40)
        encoded = canonical_event_bytes(event)
        self.assertEqual(encoded, canonical_event_bytes(encoded))
        self.assertEqual(parse_stream(encoded + b"\n"), [event])
        with self.assertRaises(EventCanonicalEncodingError):
            validate_event(encoded + b"\n")

    def test_duplicate_keys_and_noncanonical_order_are_rejected(self):
        raw = b'{"schema_version":1,"schema_version":1}'
        with self.assertRaises(EventCanonicalEncodingError):
            validate_event(raw)
        raw = json.dumps(reservation_event(), separators=(",", ":")).encode()
        # json.dumps preserves insertion order, while lifecycle canonical JSON
        # sorts keys.  This is a valid JSON object with noncanonical bytes.
        with self.assertRaises(EventCanonicalEncodingError):
            validate_event(raw)

    def test_event_id_sequence_and_exhaustion(self):
        self.assertEqual(event_id(1), "EVT-000000000001")
        self.assertEqual(event_id(MAX_EVENT_SEQUENCE), "EVT-999999999999")
        self.assertEqual(
            next_event_id([{"event_id": "EVT-000000000001"}]),
            "EVT-000000000002",
        )
        with self.assertRaises(EventIdExhausted):
            event_id(MAX_EVENT_SEQUENCE + 1)
        with self.assertRaises(EventIdExhausted):
            next_event_id([{"event_id": "EVT-999999999999"}])

    def test_schema_and_prefix_fail_closed(self):
        bad = reservation_event()
        bad["schema_version"] = 2
        with self.assertRaises(EventSchemaError):
            validate_event(bad)
        old = canonical_event_bytes(
            dict(reservation_event(), expected_event_head="b" * 40)
        ) + b"\n"
        changed = old.replace(b"principal", b"maintainer", 1)
        with self.assertRaises(EventTransitionError):
            append_only_prefix(old, changed)

    def test_maintenance_reservation_matches_final_version_lineage(self):
        event = reservation_event(expected_head="b" * 40)
        event.update(owner_line="maintenance/0.3")
        validate_event(event)
        event["final_version"] = "0.4.1"
        event["intended_dev_version"] = "0.4.1-DEV"
        with self.assertRaises(EventSchemaError):
            validate_event(event)

    def test_git_identity_fields_require_full_sha(self):
        event = reservation_event(expected_head="b" * 40)
        event["expected_line_head"] = "line-head"
        with self.assertRaises(EventSchemaError):
            validate_event(event)

        opened = reservation_event(
            event_id_value="EVT-000000000001",
            transaction_id="open-1",
            expected_head="b" * 40,
            event_type="development_reservation_opened",
        )
        opened["actual_dev_head"] = "dev-head"
        with self.assertRaises(EventSchemaError):
            validate_event(opened)

        consumed = reservation_event(
            event_id_value="EVT-000000000001",
            transaction_id="consume-1",
            expected_head="b" * 40,
            event_type="development_reservation_consumed",
        )
        consumed.pop("expected_line_head")
        consumed["closure_anchor"] = "iteration-anchor"
        consumed["terminal_disposition"] = "closed"
        with self.assertRaises(EventSchemaError):
            validate_event(consumed)

    def test_durable_fields_reject_local_and_secret_like_values(self):
        event = reservation_event()
        event["non_entry_evidence"] = "/Users/fixture/private-head"
        event.pop("expected_line_head")
        event["event_type"] = "development_reservation_aborted"
        with self.assertRaises(EventSchemaError):
            validate_event(event)

        intent = release_intent_event()
        intent["certification_environment"] = "token=private-value"
        with self.assertRaises(EventSchemaError):
            validate_event(intent)

    def test_release_intent_abort_requires_structured_exclusion_proof(self):
        abort = {
            "schema_version": 1,
            "event_id": "EVT-000000000003",
            "event_type": "release_intent_aborted",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "abort-1",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": "b" * 40,
            "intent_id": "intent-1",
            "candidate_id": "candidate-1",
            "public_tag": "v0.3.1",
            "no_public_tag_evidence": "tag absent",
        }
        with self.assertRaises(EventSchemaError):
            validate_event(abort)

        abort["no_public_tag_evidence"] = {
            "public_tag": "v0.3.1",
            "tag_absent": True,
            "checked_at_utc": "2026-09-20T12:34:56Z",
            "exclusion_verified": True,
            "exclusion_snapshot": "d" * 64,
        }
        validate_event(abort)

    def test_release_intent_transition_has_one_active_candidate_intent(self):
        candidate = candidate_event()
        first = release_intent_event()
        second = release_intent_event(
            "EVT-000000000003",
            transaction_id="intent-2",
            intent_id="intent-2",
        )
        with self.assertRaises(EventTransitionError):
            validate_transition([candidate, first], second)

        abort = {
            "schema_version": 1,
            "event_id": "EVT-000000000003",
            "event_type": "release_intent_aborted",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "abort-1",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": "b" * 40,
            "intent_id": "intent-1",
            "candidate_id": "candidate-1",
            "public_tag": "v0.3.1",
            "no_public_tag_evidence": {
                "public_tag": "v0.3.1",
                "tag_absent": True,
                "checked_at_utc": "2026-09-20T12:34:56Z",
                "exclusion_verified": True,
                "exclusion_snapshot": "d" * 64,
            },
        }
        validate_transition([candidate, first], abort)

        replacement = release_intent_event(
            "EVT-000000000004",
            transaction_id="intent-2",
            intent_id="intent-2",
        )
        validate_transition([candidate, first, abort], replacement)

        mismatched_abort = dict(abort)
        mismatched_abort["candidate_id"] = "candidate-2"
        mismatched_abort["event_id"] = "EVT-000000000004"
        mismatched_abort["transaction_id"] = "abort-2"
        mismatched_abort["no_public_tag_evidence"] = dict(
            abort["no_public_tag_evidence"]
        )
        with self.assertRaises(EventTransitionError):
            validate_transition([candidate, first], mismatched_abort)

    def test_withdrawn_candidate_keeps_version_occupied(self):
        candidate = candidate_event()
        withdrawn = {
            "schema_version": 1,
            "event_id": "EVT-000000000002",
            "event_type": "candidate_withdrawn",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "withdraw-1",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": "b" * 40,
            "candidate_id": "candidate-1",
            "candidate_ref": "refs/heads/candidates/0.3.1",
            "candidate_sha": "c" * 40,
            "candidate_tree": "f" * 40,
            "withdrawal_evidence": "evidence/withdrawal-1",
        }
        prepared = reservation_event(
            event_id_value="EVT-000000000003",
            transaction_id="reuse-withdrawn",
            expected_head="b" * 40,
        )
        with self.assertRaises(EventTransitionError):
            validate_transition([candidate, withdrawn], prepared)


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cyax-event-")
        self.repo = Path(self.directory.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Fixture"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "fixture@example.invalid"],
            check=True,
        )
        (self.repo / "README").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "fixture"], check=True)
        # Fixture writes carry explicit live-proof callbacks.  A proofless
        # writer is retained for the fail-closed negative cases below.
        self.writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
        )
        self.unproved_writer = ReleaseEventWriter(self.repo)
        self.writer.bootstrap()

    def tearDown(self):
        self.directory.cleanup()

    def _prepare_release_intent_abort(self):
        """Append a valid pre-tag lifecycle through a prepared intent."""

        head = self.writer.current_head()
        prepared = reservation_event(expected_head=head)
        opened_result = self.writer.append(prepared, expected_head=head)
        opened = reservation_event(
            "EVT-000000000002",
            transaction_id="open-for-abort",
            expected_head=opened_result.head,
            event_type="development_reservation_opened",
        )
        opened_result = self.writer.append(opened, expected_head=opened_result.head)
        consumed = dict(
            opened,
            event_id="EVT-000000000003",
            transaction_id="consume-for-abort",
            expected_event_head=opened_result.head,
            event_type="development_reservation_consumed",
            closure_anchor="e" * 40,
            terminal_disposition="closed",
        )
        consumed.pop("actual_dev_head")
        consumed_result = self.writer.append(consumed, expected_head=opened_result.head)
        candidate = candidate_event(
            "EVT-000000000004",
            transaction_id="candidate-for-abort",
            expected_head=consumed_result.head,
        )
        candidate_result = self.writer.append(candidate, expected_head=consumed_result.head)
        intent = release_intent_event(
            "EVT-000000000005",
            transaction_id="intent-for-abort",
            expected_head=candidate_result.head,
        )
        intent_result = self.writer.append(intent, expected_head=candidate_result.head)
        abort = release_intent_abort_event(expected_head=intent_result.head)
        return abort

    def test_bootstrap_is_one_file_empty_orphan_stream(self):
        head = self.writer.read_head()
        self.assertEqual(head.raw, b"")
        self.assertEqual(head.events, ())
        tree = subprocess.check_output(
            ["git", "-C", str(self.repo), "ls-tree", "--name-only", head.commit],
            text=True,
        ).splitlines()
        self.assertEqual(tree, ["release-events.jsonl"])

    def test_append_expected_head_and_idempotent_transaction(self):
        head = self.writer.current_head()
        event = reservation_event(expected_head=head)
        result = self.writer.append(event, expected_head=head)
        self.assertEqual(result.status, "APPENDED")
        self.assertEqual(self.writer.read_head().events, (event,))
        replay = self.writer.append(event, expected_head=head)
        self.assertEqual((replay.status, replay.idempotent), ("IDEMPOTENT", True))

    def test_transaction_payload_collision_is_invalid(self):
        head = self.writer.current_head()
        event = reservation_event(expected_head=head)
        self.writer.append(event)
        changed = dict(
            event,
            final_version="0.3.2",
            intended_dev_version="0.3.2-DEV",
        )
        with self.assertRaises(DuplicateTransactionError):
            self.writer.append(changed, expected_head=head)

    def test_stale_expected_head_does_not_mutate_branch(self):
        head = self.writer.current_head()
        self.writer.append(reservation_event(expected_head=head), expected_head=head)
        stale = reservation_event(
            "EVT-000000000002", transaction_id="tx-2", expected_head=head
        )
        with self.assertRaises(StaleHeadError):
            self.writer.append(stale, expected_head=head)
        self.assertEqual(len(self.writer.read_head().events), 1)

    def test_reservation_transition_requires_prepared_identity(self):
        head = self.writer.current_head()
        prepared = reservation_event(expected_head=head)
        first = self.writer.append(prepared, expected_head=head)
        opened = reservation_event(
            "EVT-000000000002",
            transaction_id="tx-2",
            expected_head=first.head,
            event_type="development_reservation_opened",
        )
        self.writer.append(opened, expected_head=first.head)
        mismatched = dict(
            opened,
            event_id="EVT-000000000003",
            transaction_id="tx-3",
            expected_event_head=self.writer.current_head(),
            final_version="0.3.2",
            intended_dev_version="0.3.2-DEV",
        )
        with self.assertRaises(EventTransitionError):
            self.writer.append(mismatched)

    def test_consumed_owner_can_open_candidate(self):
        head = self.writer.current_head()
        prepared = reservation_event(expected_head=head)
        opened_result = self.writer.append(prepared, expected_head=head)
        opened = reservation_event(
            "EVT-000000000002",
            transaction_id="tx-open",
            expected_head=opened_result.head,
            event_type="development_reservation_opened",
        )
        consumed_result = self.writer.append(opened, expected_head=opened_result.head)
        consumed = dict(
            opened,
            event_id="EVT-000000000003",
            transaction_id="tx-consume",
            expected_event_head=consumed_result.head,
            event_type="development_reservation_consumed",
        )
        consumed.pop("actual_dev_head")
        consumed["closure_anchor"] = "e" * 40
        consumed["terminal_disposition"] = "closed"
        consumed_result = self.writer.append(consumed, expected_head=consumed_result.head)
        candidate = {
            "schema_version": 1,
            "event_id": "EVT-000000000004",
            "event_type": "candidate_opened",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "tx-candidate",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": consumed_result.head,
            "candidate_id": "candidate-1",
            "candidate_ref": "refs/heads/candidates/0.3.1",
            "candidate_sha": "c" * 40,
            "candidate_tree": "f" * 40,
            "final_version": "0.3.1",
            "release_line": "principal",
            "anchor_ref": "refs/tags/iterations/0.3.1",
            "anchor_sha": "e" * 40,
            "anchor_tree": "f" * 40,
        }
        result = self.writer.append(candidate, expected_head=consumed_result.head)
        self.assertEqual(result.status, "APPENDED")

    def test_competing_line_cannot_open_owner_candidate(self):
        head = self.writer.current_head()
        prepared = reservation_event(expected_head=head)
        opened_result = self.writer.append(prepared, expected_head=head)
        opened = reservation_event(
            "EVT-000000000002",
            transaction_id="tx-open",
            expected_head=opened_result.head,
            event_type="development_reservation_opened",
        )
        consumed_result = self.writer.append(opened, expected_head=opened_result.head)
        consumed = dict(
            opened,
            event_id="EVT-000000000003",
            transaction_id="tx-consume",
            expected_event_head=consumed_result.head,
            event_type="development_reservation_consumed",
        )
        consumed.pop("actual_dev_head")
        consumed["closure_anchor"] = "e" * 40
        consumed["terminal_disposition"] = "closed"
        consumed_result = self.writer.append(consumed, expected_head=consumed_result.head)
        candidate = {
            "schema_version": 1,
            "event_id": "EVT-000000000004",
            "event_type": "candidate_opened",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "tx-competing-candidate",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": consumed_result.head,
            "candidate_id": "candidate-competing",
            "candidate_ref": "refs/heads/candidates/0.3.1",
            "candidate_sha": "c" * 40,
            "candidate_tree": "f" * 40,
            "final_version": "0.3.1",
            "release_line": "maintenance/0.3",
            "anchor_ref": "refs/tags/iterations/0.3.1",
            "anchor_sha": "e" * 40,
            "anchor_tree": "f" * 40,
        }
        with self.assertRaises(EventTransitionError):
            self.writer.append(candidate, expected_head=consumed_result.head)

    def test_proven_preentry_abort_frees_version_for_reallocation(self):
        head = self.writer.current_head()
        prepared = reservation_event(expected_head=head)
        aborted = dict(
            prepared,
            event_id="EVT-000000000002",
            transaction_id="tx-abort",
            expected_event_head=None,
            event_type="development_reservation_aborted",
            abort_reason="branch_not_created",
            non_entry_evidence="branch_absent_at_reconciliation",
        )
        aborted["expected_event_head"] = self.writer.append(
            prepared, expected_head=head
        ).head
        aborted.pop("expected_line_head")
        after_abort = self.writer.append(aborted, expected_head=aborted["expected_event_head"])
        replacement = dict(
            prepared,
            event_id="EVT-000000000003",
            transaction_id="tx-replacement",
            reservation_id="reservation-2",
            expected_event_head=after_abort.head,
        )
        self.assertEqual(
            self.writer.append(replacement, expected_head=after_abort.head).status,
            "APPENDED",
        )

    def test_consumed_version_cannot_be_reallocated(self):
        head = self.writer.current_head()
        prepared = reservation_event(expected_head=head)
        opened_result = self.writer.append(prepared, expected_head=head)
        opened = reservation_event(
            "EVT-000000000002",
            transaction_id="tx-open",
            expected_head=opened_result.head,
            event_type="development_reservation_opened",
        )
        consumed_result = self.writer.append(opened, expected_head=opened_result.head)
        consumed = dict(
            opened,
            event_id="EVT-000000000003",
            transaction_id="tx-consume",
            expected_event_head=consumed_result.head,
            event_type="development_reservation_consumed",
            closure_anchor="e" * 40,
            terminal_disposition="closed",
        )
        consumed.pop("actual_dev_head")
        consumed_result = self.writer.append(consumed, expected_head=consumed_result.head)
        replacement = dict(
            prepared,
            event_id="EVT-000000000004",
            transaction_id="tx-reuse",
            reservation_id="reservation-2",
            expected_event_head=consumed_result.head,
        )
        with self.assertRaises(EventTransitionError):
            self.writer.append(replacement, expected_head=consumed_result.head)

    def test_uncertain_remote_push_retries_only_when_head_is_unchanged(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
        )
        head = writer.current_head()
        event = reservation_event(expected_head=head)
        calls = []
        observations = [
            (head, writer.read_head().raw),
            (head, writer.read_head().raw),
        ]
        expected_raw = writer.read_head().raw + canonical_event_bytes(event) + b"\n"

        def push(commit, _branch, expected):
            calls.append((commit, expected))
            if len(calls) == 1:
                raise OSError("connection lost after remote acceptance window")

        def reconcile():
            if observations:
                return observations.pop(0)
            return calls[-1][0], expected_raw

        result = writer.append_remote(
            event,
            expected_head=head,
            push=push,
            reconcile=reconcile,
        )
        self.assertEqual(result.status, "APPENDED")
        self.assertEqual(len(calls), 2)

    def test_remote_advance_without_transaction_returns_stale_block(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
        )
        head = writer.current_head()
        event = reservation_event(expected_head=head)
        advanced = writer._make_commit(b"", parent=head, message="other")
        observations = [(head, writer.read_head().raw), (advanced, b"")]

        def push(_commit, _branch, _expected):
            raise OSError("unknown remote response")

        def reconcile():
            return observations.pop(0)

        result = writer.append_remote(
            event,
            expected_head=head,
            push=push,
            reconcile=reconcile,
        )
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "EXPECTED_EVENT_HEAD_STALE", False))

    def test_remote_push_with_unclassifiable_reconciliation_freezes(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
        )
        head = writer.current_head()
        event = reservation_event(expected_head=head)
        calls = 0

        def reconcile():
            nonlocal calls
            calls += 1
            if calls == 1:
                return head, writer.read_head().raw
            raise OSError("remote read unavailable")

        result = writer.append_remote(
            event,
            expected_head=head,
            push=lambda *_args: (_ for _ in ()).throw(OSError("connection lost")),
            reconcile=reconcile,
        )
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "APPEND_OUTCOME_UNCERTAIN", True))

    def test_remote_append_without_live_proofs_is_blocked(self):
        head = self.unproved_writer.current_head()
        result = self.unproved_writer.append_remote(
            reservation_event(expected_head=head),
            expected_head=head,
            push=lambda *_args: None,
            reconcile=lambda: (head, b""),
        )
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "EXCLUSION_UNAVAILABLE", True))

    def test_local_append_without_live_proofs_is_blocked(self):
        head = self.unproved_writer.current_head()
        result = self.unproved_writer.append(
            reservation_event(expected_head=head), expected_head=head
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "EXCLUSION_UNAVAILABLE", True),
        )

        static_unproved = ReleaseEventWriter(self.repo, exclusion_checker=lambda: True)
        result = static_unproved.append(
            reservation_event(expected_head=head), expected_head=head
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "STATIC_AUTHORITY_SELECTOR_UNRESOLVED", True),
        )

    def test_abort_requires_live_public_tag_absence_checker(self):
        abort = self._prepare_release_intent_abort()
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
        )
        result = writer.append(abort, expected_head=abort["expected_event_head"])
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "PUBLIC_TAG_ABSENCE_UNAVAILABLE", True),
        )

    def test_abort_blocks_when_public_tag_exists(self):
        abort = self._prepare_release_intent_abort()
        checked: list[str] = []
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            public_tag_absence_checker=lambda tag: checked.append(tag) or False,
        )
        result = writer.append(abort, expected_head=abort["expected_event_head"])
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "PUBLIC_TAG_EXISTS", True),
        )
        self.assertEqual(checked, ["v0.3.1"])

    def test_abort_appends_only_after_verified_public_tag_absence(self):
        abort = self._prepare_release_intent_abort()
        checked: list[str] = []
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            public_tag_absence_checker=lambda tag: checked.append(tag) or True,
        )
        result = writer.append(abort, expected_head=abort["expected_event_head"])
        self.assertEqual(result.status, "APPENDED")
        self.assertEqual(checked, ["v0.3.1"])

    def test_remote_abort_also_requires_live_public_tag_absence(self):
        abort = self._prepare_release_intent_abort()
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
        )
        head = writer.current_head()
        result = writer.append_remote(
            abort,
            expected_head=head,
            push=lambda *_args: None,
            reconcile=lambda: (head, writer.read_head().raw),
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "PUBLIC_TAG_ABSENCE_UNAVAILABLE", True),
        )

    def test_callback_only_remote_append_without_observation_is_blocked(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
        )
        head = writer.current_head()
        result = writer.append_remote(
            reservation_event(expected_head=head),
            expected_head=head,
            push=lambda *_args: None,
        )
        self.assertEqual((result.status, result.reason_code, result.frozen),
                         ("BLOCKED", "APPEND_OUTCOME_UNCERTAIN", True))

    def test_callback_only_bootstrap_requires_and_uses_observation(self):
        writer = ReleaseEventWriter(self.repo, branch="callback-bootstrap")
        observations = [(None, b"")]
        pushed: list[str] = []

        def push(commit, _branch, _expected):
            pushed.append(commit)
            observations.append((commit, b""))

        result = writer.bootstrap_remote(
            push=push,
            reconcile=lambda: observations.pop(0),
            protection_checker=lambda: True,
        )
        self.assertEqual(result, pushed[0])

    def test_callback_only_bootstrap_without_observation_is_uncertain(self):
        writer = ReleaseEventWriter(self.repo, branch="callback-bootstrap-no-proof")
        with self.assertRaises(AppendOutcomeUncertain) as context:
            writer.bootstrap_remote(push=lambda *_args: None, protection_checker=lambda: True)
        self.assertEqual(context.exception.reason_code, "APPEND_OUTCOME_UNCERTAIN")

    def test_bootstrap_push_uncertainty_reconciles_before_freezing(self):
        writer = ReleaseEventWriter(self.repo, branch="callback-bootstrap-uncertain")
        observations = [(None, b""), (None, b"")]

        with self.assertRaises(AppendOutcomeUncertain) as context:
            writer.bootstrap_remote(
                push=lambda *_args: (_ for _ in ()).throw(OSError("connection lost")),
                reconcile=lambda: observations.pop(0),
                protection_checker=lambda: True,
            )
        self.assertEqual(context.exception.reason_code, "APPEND_OUTCOME_UNCERTAIN")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
