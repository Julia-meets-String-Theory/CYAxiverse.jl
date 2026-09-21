"""Focused canonical event and append-only writer tests for CYAX-0125."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

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
    ExclusionUnavailable,
    ReleaseEventWriter,
    StaleHeadError,
    WriterError,
    bootstrap_release_events,
)
from version_lifecycle.certification import is_safe_public_value  # noqa: E402
from version_lifecycle.allocation import _event_occupied  # noqa: E402
from version_lifecycle.git_refs import GitRepository  # noqa: E402


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


def non_entry_proof(prepared: dict, *, absent: bool = False) -> dict:
    proof = {
        "verified": True,
        "reservation_id": prepared["reservation_id"],
        "owner_line": prepared["owner_line"],
        "final_version": prepared["final_version"],
        "intended_dev_version": prepared["intended_dev_version"],
        "line_ref": (
            "refs/heads/vmm" if prepared["owner_line"] == "principal"
            else f"refs/heads/{prepared['owner_line']}"
        ),
        "expected_line_head": prepared["expected_line_head"],
        "line_state": "absent" if absent else "unchanged",
        "dev_not_entered": True,
        "exclusion_verified": True,
        "observed_at_utc": "2026-09-20T12:34:55Z",
        "evidence_ref": "evidence/non-entry.json",
        "evidence_digest": "d" * 64,
    }
    if not absent:
        proof["observed_line_head"] = prepared["expected_line_head"]
    return proof


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
        "certification_subject_sha": "c" * 40,
        "certification_subject_tree": "f" * 40,
        "certification_policy_revision": "policy-r1",
        "certification_harness_revision": "harness-r1",
        "certification_environment": "ci-linux",
        "certification_evidence_refs": ["evidence/certification-1"],
        "certification_transfer_evidence": {
            "verified": True,
            "candidate_sha": "c" * 40,
            "final_release_sha": "f" * 40,
            "certified_tree": "f" * 40,
            "candidate_tree": "f" * 40,
            "final_release_tree": "f" * 40,
            "anchor_tree": "f" * 40,
            "evidence_ref": "evidence/transfer-1",
        },
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
    def test_valid_maintenance_refs_pass_event_validation(self):
        for line in ("10.2", "172.16", "169.254", "192.168"):
            with self.subTest(line=line):
                owner = f"maintenance/{line}"
                prepared = reservation_event(expected_head="b" * 40)
                prepared.update(
                    owner_line=owner,
                    final_version=f"{line}.1",
                    intended_dev_version=f"{line}.1-DEV",
                )
                validate_event(prepared)
                aborted = dict(
                    prepared,
                    event_id="EVT-000000000002",
                    transaction_id="maintenance-abort",
                    event_type="development_reservation_aborted",
                    abort_reason="definite_non_entry",
                    non_entry_evidence=non_entry_proof(prepared),
                )
                aborted.pop("expected_line_head")
                validate_event(aborted)
                line_opened = {
                    "schema_version": 1,
                    "event_id": "EVT-000000000003",
                    "event_type": "maintenance_line_opened",
                    "timestamp_utc": "2026-09-20T12:34:56Z",
                    "transaction_id": "maintenance-opened",
                    "static_iteration_snapshot": SNAPSHOT,
                    "expected_event_head": "b" * 40,
                    "release_line": owner,
                    "approved_base_version": f"{line}.0",
                    "branch_ref": f"refs/heads/{owner}",
                    "branch_head": "d" * 40,
                    "reservation_id": prepared["reservation_id"],
                    "dev_version": f"{line}.1-DEV",
                }
                validate_event(line_opened)

    def test_typed_public_refs_allow_maintenance_lines_but_generic_values_stay_strict(self):
        self.assertTrue(
            is_safe_public_value(
                "refs/heads/maintenance/10.2", key="branch_ref"
            )
        )
        self.assertTrue(
            is_safe_public_value(
                "refs/heads/maintenance/10.2", key="line_ref"
            )
        )
        for key in ("metadata", "evidence_ref"):
            with self.subTest(key=key):
                self.assertFalse(
                    is_safe_public_value("refs/heads/10.0.0.1", key=key)
                )

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

    def test_reservation_abort_requires_matching_structured_non_entry_proof(self):
        prepared = reservation_event(expected_head="b" * 40)
        aborted = dict(
            prepared,
            event_id="EVT-000000000002",
            transaction_id="tx-abort",
            event_type="development_reservation_aborted",
            abort_reason="definite_non_entry",
            non_entry_evidence="not-proof",
        )
        aborted.pop("expected_line_head")
        with self.assertRaises(EventSchemaError):
            validate_event(aborted)
        with self.assertRaises(EventSchemaError):
            _event_occupied([prepared, aborted])

        aborted["non_entry_evidence"] = non_entry_proof(prepared)
        validate_transition([prepared], aborted)
        self.assertNotIn("0.3.1", _event_occupied([prepared, aborted]))

        wrong = dict(aborted)
        wrong["non_entry_evidence"] = dict(aborted["non_entry_evidence"])
        wrong["non_entry_evidence"]["expected_line_head"] = "d" * 40
        wrong["non_entry_evidence"]["observed_line_head"] = "d" * 40
        with self.assertRaises(EventTransitionError):
            validate_transition([prepared], wrong)

        wrong["non_entry_evidence"]["verified"] = False
        with self.assertRaises(EventSchemaError):
            validate_event(wrong)

        late = dict(aborted)
        late["non_entry_evidence"] = dict(aborted["non_entry_evidence"])
        late["non_entry_evidence"]["observed_at_utc"] = "2026-09-20T12:35:00Z"
        with self.assertRaises(EventSchemaError):
            validate_event(late)

        maintenance_prepared = reservation_event(expected_head="b" * 40)
        maintenance_prepared.update(
            owner_line="maintenance/1.2",
            final_version="1.2.1",
            intended_dev_version="1.2.1-DEV",
        )
        maintenance_abort = dict(
            maintenance_prepared,
            event_id="EVT-000000000002",
            transaction_id="maintenance-abort",
            event_type="development_reservation_aborted",
            abort_reason="branch_not_created",
            non_entry_evidence=non_entry_proof(maintenance_prepared, absent=True),
        )
        maintenance_abort.pop("expected_line_head")
        validate_transition([maintenance_prepared], maintenance_abort)

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

        intent = release_intent_event()
        intent["certification_environment"] = "https://user:pass@example.com/env"
        with self.assertRaises(EventSchemaError):
            validate_event(intent)

        for alias in (
            "https://0177.0.0.1/org/repo",
            "https://127.0.0.1%2e/org/repo",
            "10.0.0.1",
            "::1",
            "runner=fc00::1",
            "10.0.0.1:443",
            "runner@10.0.0.1",
            "runner-10.0.0.1",
            "10.0.0.1\\path",
            "10.0.0.1.example.com",
            "10.1",
            "10.0.1",
            "172.16.1",
            "192.168.1",
            "runner=10.1",
            "10.0.0.1#frag",
        ):
            with self.subTest(alias=alias):
                intent = release_intent_event()
                intent["certification_environment"] = alias
                with self.assertRaises(EventSchemaError):
                    validate_event(intent)

    def test_release_intent_binds_certified_subject_to_exact_trees(self):
        intent = release_intent_event()
        intent["certification_subject_tree"] = "d" * 40
        with self.assertRaises(EventSchemaError):
            validate_event(intent)

        intent = release_intent_event()
        intent["certification_binding"] = "commit-bound"
        intent.pop("certification_transfer_evidence")
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

    def test_tree_bound_release_events_require_durable_transfer_proof(self):
        missing = release_intent_event()
        missing.pop("certification_transfer_evidence")
        with self.assertRaises(EventSchemaError):
            validate_event(missing)

        mismatched = release_intent_event()
        mismatched["certification_transfer_evidence"] = dict(
            mismatched["certification_transfer_evidence"]
        )
        mismatched["certification_transfer_evidence"]["final_release_sha"] = "e" * 40
        with self.assertRaises(EventSchemaError):
            validate_event(mismatched)

        released = {
            **{key: value for key, value in release_intent_event().items() if key != "intent_id"},
            "event_id": "EVT-000000000003",
            "event_type": "released",
            "transaction_id": "released-1",
            "closure_timestamp_utc": "2026-09-20T12:34:56Z",
            "evidence_refs": ["evidence/released-1"],
            "main_at_event_sha": "f" * 40,
            "main_at_event_version": "0.3.1",
            "previous_main_sha": "d" * 40,
            "previous_main_version": "0.2.0",
        }
        released.pop("certification_transfer_evidence")
        with self.assertRaises(EventSchemaError):
            validate_event(released)

    def test_released_transition_matches_durable_transfer_proof(self):
        candidate = candidate_event()
        intent = release_intent_event()
        released = {
            **{key: value for key, value in intent.items() if key != "intent_id"},
            "event_id": "EVT-000000000003",
            "event_type": "released",
            "transaction_id": "released-1",
            "closure_timestamp_utc": "2026-09-20T12:34:56Z",
            "evidence_refs": ["evidence/released-1"],
            "main_at_event_sha": "f" * 40,
            "main_at_event_version": "0.3.1",
            "previous_main_sha": "d" * 40,
            "previous_main_version": "0.2.0",
        }
        released["certification_transfer_evidence"] = dict(
            released["certification_transfer_evidence"]
        )
        released["certification_transfer_evidence"]["evidence_ref"] = (
            "evidence/other-transfer"
        )
        with self.assertRaises(EventTransitionError):
            validate_transition([candidate, intent], released)

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

    def test_candidate_cannot_be_reopened_for_the_same_final_after_withdrawal(self):
        candidate = candidate_event()
        replacement = candidate_event(
            "EVT-000000000002", transaction_id="candidate-2", candidate_id="candidate-2"
        )
        with self.assertRaisesRegex(EventTransitionError, "durable candidate"):
            validate_transition([candidate], replacement)
        withdrawn = {
            "schema_version": 1,
            "event_id": "EVT-000000000002",
            "event_type": "candidate_withdrawn",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "withdraw-1",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": "b" * 40,
            "candidate_id": candidate["candidate_id"],
            "candidate_ref": candidate["candidate_ref"],
            "candidate_sha": candidate["candidate_sha"],
            "candidate_tree": candidate["candidate_tree"],
            "withdrawal_evidence": "evidence/withdrawal-1",
        }
        validate_transition([candidate], withdrawn)
        replacement["event_id"] = "EVT-000000000003"
        with self.assertRaisesRegex(EventTransitionError, "durable candidate"):
            validate_transition([candidate, withdrawn], replacement)

    def test_withdrawal_requires_intent_abort_and_cannot_follow_release(self):
        candidate = candidate_event()
        intent = release_intent_event()
        withdrawn = {
            "schema_version": 1,
            "event_id": "EVT-000000000003",
            "event_type": "candidate_withdrawn",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "withdraw-after-intent",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": "b" * 40,
            "candidate_id": candidate["candidate_id"],
            "candidate_ref": candidate["candidate_ref"],
            "candidate_sha": candidate["candidate_sha"],
            "candidate_tree": candidate["candidate_tree"],
            "withdrawal_evidence": "evidence/withdrawal-1",
        }
        with self.assertRaisesRegex(EventTransitionError, "intent must be aborted"):
            validate_transition([candidate, intent], withdrawn)

        abort = release_intent_abort_event("EVT-000000000003")
        validate_transition([candidate, intent], abort)
        withdrawn["event_id"] = "EVT-000000000004"
        validate_transition([candidate, intent, abort], withdrawn)

        released = {
            **{key: value for key, value in intent.items() if key not in {"intent_id"}},
            "event_id": "EVT-000000000003",
            "event_type": "released",
            "transaction_id": "released-1",
            "closure_timestamp_utc": "2026-09-20T12:34:56Z",
            "evidence_refs": ["evidence/released-1"],
            "main_at_event_sha": "f" * 40,
            "main_at_event_version": "0.3.1",
            "previous_main_sha": "d" * 40,
            "previous_main_version": "0.2.0",
        }
        validate_transition([candidate, intent], released)
        with self.assertRaisesRegex(EventTransitionError, "released candidate"):
            validate_transition([candidate, intent, released], withdrawn)
        with self.assertRaisesRegex(EventTransitionError, "released intent"):
            validate_transition(
                [candidate, intent, released],
                release_intent_abort_event("EVT-000000000004", transaction_id="late-abort"),
            )

    def test_different_final_consumes_reserved_identity_and_opens_one_candidate(self):
        prepared = reservation_event(expected_head="b" * 40)
        opened = reservation_event(
            "EVT-000000000002", transaction_id="open-1",
            expected_head="b" * 40, event_type="development_reservation_opened",
        )
        consumed = dict(
            opened,
            event_id="EVT-000000000003",
            transaction_id="consume-unused",
            event_type="development_reservation_consumed",
            closed_final_version="0.3.2",
            closure_anchor="refs/tags/iterations/0.3.2",
            terminal_disposition="CONSUMED_UNUSED_DEV_RESERVATION",
        )
        consumed.pop("actual_dev_head")
        for prior, event in (([], prepared), ([prepared], opened), ([prepared, opened], consumed)):
            validate_transition(prior, event)
        self.assertEqual(_event_occupied([consumed]), {"0.3.1", "0.3.2"})

        reused = reservation_event(
            "EVT-000000000004", transaction_id="reuse-closed",
            expected_head="b" * 40,
        )
        reused.update(final_version="0.3.2", intended_dev_version="0.3.2-DEV")
        with self.assertRaises(EventTransitionError):
            validate_transition([prepared, opened, consumed], reused)

        candidate = candidate_event(
            "EVT-000000000004", transaction_id="candidate-closed",
        )
        candidate.update(
            final_version="0.3.2",
            candidate_ref="refs/heads/candidates/0.3.2",
            anchor_ref="refs/tags/iterations/0.3.2",
        )
        validate_transition([prepared, opened, consumed], candidate)
        duplicate = dict(
            candidate,
            event_id="EVT-000000000005",
            transaction_id="duplicate-candidate",
            candidate_id="candidate-2",
        )
        with self.assertRaisesRegex(EventTransitionError, "durable candidate"):
            validate_transition([prepared, opened, consumed, candidate], duplicate)

        bad = dict(consumed, terminal_disposition="something-else")
        with self.assertRaises(EventSchemaError):
            validate_event(bad)
        bad = dict(consumed, closure_anchor="refs/tags/iterations/0.3.1")
        with self.assertRaises(EventSchemaError):
            validate_event(bad)
        bad = dict(consumed, closed_final_version="0.3.1")
        with self.assertRaises(EventSchemaError):
            validate_event(bad)

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
            reservation_non_entry_checker=lambda _event: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        self.unproved_writer = ReleaseEventWriter(
            self.repo,
            exclusion_lease=lambda: nullcontext(True),
        )
        self.writer.bootstrap(protection_checker=lambda: True)

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

    def test_public_bootstrap_requires_lease_and_protection(self):
        for label, lease, protection in (
            ("missing lease", None, lambda: True),
            ("false lease", lambda: nullcontext(False), lambda: True),
            ("missing protection", lambda: nullcontext(True), None),
            ("false protection", lambda: nullcontext(True), lambda: False),
        ):
            with self.subTest(label=label):
                writer = ReleaseEventWriter(
                    self.repo,
                    branch=f"bootstrap-requires-{label.replace(' ', '-')}",
                    exclusion_lease=lease,
                )
                with self.assertRaises(ExclusionUnavailable):
                    writer.bootstrap(protection_checker=protection)
                self.assertFalse(writer.branch_exists())

    def test_public_bootstrap_holds_both_exclusions_through_protected_cas(self):
        state = {"lease": False, "checked": False}

        class TrackingLease:
            def __enter__(self):
                state["lease"] = True
                return True

            def __exit__(self, *_args):
                state["lease"] = False
                return False

        writer = ReleaseEventWriter(
            self.repo,
            branch="bootstrap-controlled",
            exclusion_lease=lambda: TrackingLease(),
        )

        def protection_checker():
            state["checked"] = True
            self.assertTrue(state["lease"])
            self.assertTrue(writer.static_exclusion.held())
            return True

        head = writer.bootstrap(protection_checker=protection_checker)
        self.assertTrue(state["checked"])
        self.assertFalse(state["lease"])
        self.assertFalse(writer.static_exclusion.held())
        self.assertEqual(writer.current_head(), head)
        self.assertEqual(writer.read_head().raw, b"")

    def test_public_bootstrap_static_contention_fails_closed(self):
        writer = ReleaseEventWriter(
            self.repo,
            branch="bootstrap-contention",
            exclusion_lease=lambda: nullcontext(True),
        )
        held = writer.static_exclusion.acquire()
        outcomes: list[BaseException] = []

        def run_bootstrap():
            try:
                writer.bootstrap(protection_checker=lambda: True)
            except BaseException as error:  # noqa: BLE001 - capture worker result
                outcomes.append(error)

        try:
            thread = threading.Thread(target=run_bootstrap)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        finally:
            held.release()
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], ExclusionUnavailable)
        self.assertFalse(writer.branch_exists())

    def test_public_bootstrap_release_failure_is_uncertain(self):
        class RaisingLease:
            def __enter__(self):
                return True

            def __exit__(self, *_args):
                raise OSError("lease release failed after local bootstrap")

        writer = ReleaseEventWriter(
            self.repo,
            branch="bootstrap-release-failure",
            exclusion_lease=lambda: RaisingLease(),
        )
        with self.assertRaises(AppendOutcomeUncertain):
            writer.bootstrap(protection_checker=lambda: True)
        self.assertTrue(writer.branch_exists())
        self.assertEqual(writer.read_head().raw, b"")

    def test_external_lease_cannot_suppress_bootstrap_protection_failure(self):
        class SuppressingLease:
            def __enter__(self):
                return True

            def __exit__(self, *_args):
                return True

        writer = ReleaseEventWriter(
            self.repo,
            branch="bootstrap-suppression",
            exclusion_lease=lambda: SuppressingLease(),
        )
        with self.assertRaises(ExclusionUnavailable):
            writer.bootstrap(protection_checker=lambda: False)
        self.assertFalse(writer.branch_exists())

    def test_bootstrap_wrapper_uses_the_protected_boundary(self):
        head = bootstrap_release_events(
            self.repo,
            branch="bootstrap-wrapper",
            exclusion_lease=lambda: nullcontext(True),
            protection_checker=lambda: True,
        )
        writer = ReleaseEventWriter(self.repo, branch="bootstrap-wrapper")
        self.assertEqual(writer.current_head(), head)
        self.assertEqual(writer.read_head().events, ())

    def test_event_writer_reuses_outer_governed_lease(self):
        state = {"held": False, "enters": 0}

        @contextmanager
        def nonreentrant_lease():
            if state["held"]:
                raise RuntimeError("external lease cannot be entered twice")
            state["held"] = True
            state["enters"] += 1
            try:
                yield True
            finally:
                state["held"] = False

        repository = GitRepository(
            self.repo,
            exclusion_checker=lambda: state["held"],
            exclusion_lease=nonreentrant_lease,
        )
        writer = ReleaseEventWriter(
            self.repo,
            branch="nested-governed",
            exclusion_checker=lambda: state["held"],
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=nonreentrant_lease,
        )
        with repository.acquire_static_mutation():
            head = writer.bootstrap(protection_checker=lambda: state["held"])
            appended = writer.append(reservation_event(expected_head=head), expected_head=head)
            self.assertEqual(appended.status, "APPENDED")
            self.assertTrue(state["held"])
            self.assertEqual(state["enters"], 1)
        self.assertFalse(state["held"])
        self.assertEqual(state["enters"], 1)

    def test_read_head_rejects_noncanonical_event_trees(self):
        head = self.writer.current_head()
        blob = subprocess.check_output(
            ["git", "-C", str(self.repo), "hash-object", "-w", "--stdin"],
            input=b"",
        ).decode().strip()
        for label, tree_input in (
            ("executable", f"100755 blob {blob}\trelease-events.jsonl\n"),
            ("symlink", f"120000 blob {blob}\trelease-events.jsonl\n"),
            ("gitlink", f"160000 commit {head}\trelease-events.jsonl\n"),
            ("extra file", f"100644 blob {blob}\tREADME\n100644 blob {blob}\trelease-events.jsonl\n"),
        ):
            with self.subTest(label=label):
                tree = subprocess.check_output(
                    ["git", "-C", str(self.repo), "mktree"],
                    input=tree_input.encode(),
                ).decode().strip()
                malformed = subprocess.check_output(
                    ["git", "-C", str(self.repo), "commit-tree", tree, "-p", head, "-m", "bad tree"],
                    text=True,
                ).strip()
                subprocess.run(
                    ["git", "-C", str(self.repo), "update-ref", "refs/heads/release-events", malformed],
                    check=True,
                )
                with self.assertRaises((AppendOutcomeUncertain, WriterError)):
                    self.writer.read_head()

    def test_read_head_rejects_historical_tree_merge_and_nonprefix_topology(self):
        root = self.writer.current_head()
        first_event = reservation_event(expected_head=root)
        first_raw = canonical_event_bytes(first_event) + b"\n"

        def commit_with_tree(
            raw: bytes, *, mode: str = "100644", parents: tuple[str, ...] = ()
        ) -> str:
            blob = subprocess.check_output(
                ["git", "-C", str(self.repo), "hash-object", "-w", "--stdin"],
                input=raw,
            ).decode().strip()
            tree = subprocess.check_output(
                ["git", "-C", str(self.repo), "mktree"],
                input=f"{mode} blob {blob}\trelease-events.jsonl\n".encode(),
            ).decode().strip()
            args = ["git", "-C", str(self.repo), "commit-tree", tree]
            for parent in parents:
                args.extend(("-p", parent))
            return subprocess.check_output(
                [*args, "-m", "topology fixture"], text=True
            ).strip()

        malformed_parent = commit_with_tree(first_raw, mode="100755", parents=(root,))
        second_event = reservation_event(
            "EVT-000000000002",
            transaction_id="topology-second",
            expected_head=malformed_parent,
        )
        descendant = self.writer._make_commit(
            first_raw + canonical_event_bytes(second_event) + b"\n",
            parent=malformed_parent,
            message="descendant",
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "update-ref", self.writer.ref, descendant],
            check=True,
        )
        with self.assertRaises(AppendOutcomeUncertain):
            self.writer.read_head()

        merge_child = self.writer._make_commit(first_raw, parent=root, message="child")
        merge = commit_with_tree(first_raw, parents=(merge_child, root))
        subprocess.run(
            ["git", "-C", str(self.repo), "update-ref", self.writer.ref, merge],
            check=True,
        )
        with self.assertRaises(AppendOutcomeUncertain):
            self.writer.read_head()

        valid_child = self.writer._make_commit(first_raw, parent=root, message="child")
        changed_first = first_raw.replace(b'"tx-1"', b'"tx-x"', 1)
        second_event = reservation_event(
            "EVT-000000000002",
            transaction_id="nonprefix-second",
            expected_head=valid_child,
        )
        nonprefix = self.writer._make_commit(
            changed_first + canonical_event_bytes(second_event) + b"\n",
            parent=valid_child,
            message="nonprefix",
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "update-ref", self.writer.ref, nonprefix],
            check=True,
        )
        with self.assertRaises(AppendOutcomeUncertain):
            self.writer.read_head()

    def test_read_head_rejects_event_expected_head_that_differs_from_parent(self):
        root = self.writer.current_head()
        event = reservation_event(expected_head="f" * 40)
        malformed = self.writer._make_commit(
            canonical_event_bytes(event) + b"\n", parent=root, message="wrong parent"
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "update-ref", self.writer.ref, malformed],
            check=True,
        )
        with self.assertRaises(AppendOutcomeUncertain):
            self.writer.read_head()

    def test_callback_exceptions_block_and_freeze_without_appending(self):
        head = self.writer.current_head()
        event = reservation_event(expected_head=head)

        def unavailable(*_args):
            raise OSError("fixture authority unavailable")

        for callbacks, reason in (
            ({"exclusion_checker": unavailable, "static_snapshot_checker": lambda _: True},
             "EXCLUSION_UNAVAILABLE"),
            ({"exclusion_checker": lambda: True, "static_snapshot_checker": unavailable},
             "STATIC_AUTHORITY_SELECTOR_UNRESOLVED"),
        ):
            with self.subTest(reason=reason):
                writer = ReleaseEventWriter(
                    self.repo,
                    exclusion_lease=lambda: nullcontext(True),
                    **callbacks,
                )
                result = writer.append(event, expected_head=head)
                self.assertEqual((result.status, result.reason_code, result.frozen),
                                 ("BLOCKED", reason, True))
                remote_result = writer.append_remote(
                    event,
                    expected_head=head,
                    push=lambda *_args: None,
                    reconcile=lambda: (head, b""),
                )
                self.assertEqual(
                    (remote_result.status, remote_result.reason_code, remote_result.frozen),
                    ("BLOCKED", reason, True),
                )
                self.assertEqual(self.writer.current_head(), head)

        with self.assertRaisesRegex(ExclusionUnavailable, "event branch creation protection"):
            self.writer.bootstrap_remote(
                remote="origin",
                protection_checker=unavailable,
            )

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

    def test_opened_reservation_cannot_be_opened_again(self):
        head = self.writer.current_head()
        first = self.writer.append(reservation_event(expected_head=head), expected_head=head)
        opened = reservation_event(
            "EVT-000000000002",
            transaction_id="tx-open",
            expected_head=first.head,
            event_type="development_reservation_opened",
        )
        second = self.writer.append(opened, expected_head=first.head)
        duplicate = dict(
            opened,
            event_id="EVT-000000000003",
            transaction_id="tx-open-again",
            expected_event_head=second.head,
        )
        with self.assertRaisesRegex(EventTransitionError, "already opened"):
            self.writer.append(duplicate, expected_head=second.head)
        self.assertEqual(self.writer.current_head(), second.head)

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
            non_entry_evidence=non_entry_proof(prepared),
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

    def test_abort_blocks_without_live_non_entry_recheck(self):
        head = self.writer.current_head()
        prepared = reservation_event(expected_head=head)
        prepared_result = self.writer.append(prepared, expected_head=head)
        aborted = dict(
            prepared,
            event_id="EVT-000000000002",
            transaction_id="tx-abort",
            expected_event_head=prepared_result.head,
            event_type="development_reservation_aborted",
            abort_reason="definite_non_entry",
            non_entry_evidence=non_entry_proof(prepared),
        )
        aborted.pop("expected_line_head")

        def unavailable(_event):
            raise OSError("live line state unavailable")

        for checker in (None, unavailable, lambda _event: False):
            with self.subTest(checker=checker):
                writer = ReleaseEventWriter(
                    self.repo,
                    exclusion_checker=lambda: True,
                    static_snapshot_checker=lambda _digest: True,
                    reservation_non_entry_checker=checker,
                    exclusion_lease=lambda: nullcontext(True),
                )
                result = writer.append(aborted, expected_head=prepared_result.head)
                self.assertEqual(
                    (result.status, result.reason_code, result.frozen),
                    ("BLOCKED", "RESERVATION_NON_ENTRY_UNAVAILABLE", True),
                )
                self.assertEqual(writer.current_head(), prepared_result.head)

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
            exclusion_lease=lambda: nullcontext(True),
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

    def test_remote_head_read_rejects_ref_change_between_listing_and_fetch(self):
        writer = ReleaseEventWriter(
            self.repo,
            remote="origin",
            exclusion_lease=lambda: nullcontext(True),
        )
        advertised = writer.current_head()
        fetched = "f" * 40

        def raced_git(
            args: list[str], *, input_bytes: bytes | None = None, check: bool = True
        ):
            del input_bytes, check
            if args[:2] == ["ls-remote", "--refs"]:
                return subprocess.CompletedProcess(
                    ["git", *args],
                    0,
                    stdout=f"{advertised}\t{writer.ref}\n".encode("ascii"),
                    stderr=b"",
                )
            if args and args[0] == "fetch":
                return subprocess.CompletedProcess(
                    ["git", *args], 0, stdout=b"", stderr=b""
                )
            if args and args[0] == "rev-parse":
                return subprocess.CompletedProcess(
                    ["git", *args], 0, stdout=f"{fetched}\n".encode("ascii"), stderr=b""
                )
            if args[:2] == ["update-ref", "-d"]:
                return subprocess.CompletedProcess(
                    ["git", *args], 0, stdout=b"", stderr=b""
                )
            self.fail(f"unexpected Git command: {args}")

        with patch.object(writer, "_git", side_effect=raced_git):
            with self.assertRaisesRegex(
                AppendOutcomeUncertain, "advanced during reconciliation"
            ):
                writer.read_remote_head()

    def test_remote_advance_without_transaction_returns_stale_block(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.current_head()
        event = reservation_event(expected_head=head)
        advanced_event = reservation_event(
            "EVT-000000000001",
            transaction_id="remote-other",
            expected_head=head,
        )
        advanced_raw = canonical_event_bytes(advanced_event) + b"\n"
        advanced = writer._make_commit(advanced_raw, parent=head, message="other")
        observations = [(head, writer.read_head().raw), (advanced, advanced_raw)]

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
            exclusion_lease=lambda: nullcontext(True),
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

    def test_static_mutation_race_blocks_event_append(self):
        head = self.writer.current_head()
        held = self.writer.static_exclusion.acquire()
        outcomes = []

        def contender():
            outcomes.append(
                self.writer.append(
                    reservation_event(expected_head=head), expected_head=head
                )
            )

        try:
            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        finally:
            held.release()
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(
            (outcomes[0].status, outcomes[0].reason_code, outcomes[0].frozen),
            ("BLOCKED", "EXCLUSION_UNAVAILABLE", True),
        )
        self.assertEqual(self.writer.current_head(), head)

    def test_remote_append_cannot_recreate_missing_authority_branch(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.current_head()
        pushed = []
        result = writer.append_remote(
            reservation_event(expected_head=head),
            expected_head=head,
            push=lambda *_args: pushed.append(True),
            reconcile=lambda: (None, b""),
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "EVENT_BRANCH_UNAVAILABLE", True),
        )
        self.assertEqual(pushed, [])
        self.assertEqual(writer.current_head(), head)

    def test_remote_replay_does_not_use_local_only_idempotency(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.current_head()
        event = reservation_event(expected_head=head)
        appended = writer.append(event, expected_head=head)
        self.assertEqual(appended.status, "APPENDED")
        pushed: list[bool] = []
        result = writer.append_remote(
            event,
            expected_head=head,
            push=lambda *_args: pushed.append(True),
            reconcile=lambda: (None, b""),
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "EVENT_BRANCH_UNAVAILABLE", True),
        )
        self.assertEqual(pushed, [])

    def test_default_remote_append_uses_expected_head_cas_after_remote_deletion(self):
        remote = self.repo.parent / "remote-append-cas-race-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        branch = "release-events-cas-race"
        writer = ReleaseEventWriter(
            self.repo,
            branch=branch,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.bootstrap_remote(protection_checker=lambda: True)
        event = reservation_event(expected_head=head)
        original_git = writer._git
        push_commands: list[list[str]] = []

        def delete_before_push(
            args: list[str], *, input_bytes: bytes | None = None, check: bool = True
        ):
            if args and args[0] == "push":
                push_commands.append(list(args))
                subprocess.run(
                    ["git", "-C", str(remote), "update-ref", "-d", writer.ref],
                    check=True,
                )
            return original_git(args, input_bytes=input_bytes, check=check)

        with patch.object(writer, "_git", side_effect=delete_before_push):
            result = writer.append_remote(event, expected_head=head)

        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "APPEND_OUTCOME_UNCERTAIN", True),
        )
        self.assertEqual(len(push_commands), 1)
        self.assertIn(
            f"--force-with-lease={writer.ref}:{head}", push_commands[0]
        )
        advertised = subprocess.run(
            ["git", "-C", str(remote), "show-ref", "--heads"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(advertised.returncode, 1)
        self.assertEqual(advertised.stdout, "")

    def test_stale_checkout_replays_verified_remote_transaction_idempotently(self):
        remote = self.repo.parent / f"{self.repo.name}-stale-cache-replay-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        branch = "release-events-stale-replay"
        authority = ReleaseEventWriter(
            self.repo,
            branch=branch,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        empty_head = authority.bootstrap_remote(protection_checker=lambda: True)
        event = reservation_event(expected_head=empty_head)

        with tempfile.TemporaryDirectory(prefix="cyax-event-stale-cache-") as fresh:
            fresh_repo = Path(fresh)
            subprocess.run(["git", "init", "-q", str(fresh_repo)], check=True)
            subprocess.run(
                ["git", "-C", str(fresh_repo), "remote", "add", "origin", str(remote)],
                check=True,
            )
            cached = ReleaseEventWriter(
                fresh_repo,
                branch=branch,
                remote="origin",
                exclusion_checker=lambda: True,
                static_snapshot_checker=lambda _digest: True,
                exclusion_lease=lambda: nullcontext(True),
            )
            self.assertEqual(
                cached.bootstrap_remote(protection_checker=lambda: True), empty_head
            )

            appended = authority.append_remote(event, expected_head=empty_head)
            self.assertEqual(appended.status, "APPENDED")
            assert appended.head is not None

            mismatch = dict(
                event,
                final_version="0.3.2",
                intended_dev_version="0.3.2-DEV",
                reservation_id="reservation-2",
            )
            with self.assertRaises(DuplicateTransactionError):
                cached.append_remote(mismatch, expected_head=empty_head)
            replay = cached.append_remote(event, expected_head=empty_head)
            self.assertEqual((replay.status, replay.idempotent), ("IDEMPOTENT", True))
            self.assertEqual(replay.head, appended.head)
            self.assertEqual(cached.current_head(), appended.head)

    def test_verified_remote_replay_is_idempotent_when_cache_cas_loses_race(self):
        remote = self.repo.parent / f"{self.repo.name}-cache-cas-replay-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        branch = "release-events-cache-cas-replay"
        writer = ReleaseEventWriter(
            self.repo,
            branch=branch,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.bootstrap_remote(protection_checker=lambda: True)
        event = reservation_event(expected_head=head)
        original_git = writer._git

        def lose_cache_cas(
            args: list[str], *, input_bytes: bytes | None = None, check: bool = True
        ):
            if (
                len(args) == 4
                and args[:2] == ["update-ref", writer.ref]
                and args[3] == head
            ):
                return subprocess.CompletedProcess(
                    ["git", *args], 1, stdout=b"", stderr=b"simulated cache CAS loss"
                )
            return original_git(args, input_bytes=input_bytes, check=check)

        with patch.object(writer, "_git", side_effect=lose_cache_cas):
            first = writer.append_remote(event, expected_head=head)
            self.assertEqual(first.status, "APPENDED")
            self.assertEqual(writer.current_head(), head)
            replay = writer.append_remote(event, expected_head=head)

        self.assertEqual((replay.status, replay.idempotent), ("IDEMPOTENT", True))
        self.assertEqual(replay.head, first.head)
        self.assertEqual(writer.current_head(), head)
        advertised = subprocess.check_output(
            ["git", "-C", str(remote), "rev-parse", f"refs/heads/{branch}"],
            text=True,
        ).strip()
        self.assertEqual(advertised, first.head)

    def test_remote_append_does_not_create_missing_bare_remote_branch(self):
        remote = self.repo.parent / "missing-event-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        writer = ReleaseEventWriter(
            self.repo,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.current_head()
        result = writer.append_remote(
            reservation_event(expected_head=head), expected_head=head
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "EVENT_BRANCH_UNAVAILABLE", True),
        )
        advertised = subprocess.run(
            ["git", "-C", str(remote), "show-ref", "--heads"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(advertised.returncode, 1)
        self.assertEqual(advertised.stdout, "")

    def test_remote_unsupported_certification_binding_is_frozen_at_early_gates(self):
        unsupported = release_intent_event()
        unsupported["certification_binding"] = "unsupported"
        head = self.unproved_writer.current_head()
        for writer, kwargs in (
            (
                self.unproved_writer,
                {"reconcile": lambda: (head, b"")},
            ),
            (
                ReleaseEventWriter(
                    self.repo,
                    exclusion_checker=lambda: True,
                    static_snapshot_checker=lambda _digest: True,
                    exclusion_lease=lambda: nullcontext(True),
                ),
                {},
            ),
        ):
            with self.subTest(proofs=writer.exclusion_checker is not None):
                result = writer.append_remote(
                    unsupported,
                    expected_head=head,
                    push=lambda *_args: None,
                    **kwargs,
                )
                self.assertEqual(
                    (result.status, result.reason_code, result.frozen),
                    ("BLOCKED", "UNSUPPORTED_CERTIFICATION_BINDING", True),
                )
                self.assertEqual(writer.current_head(), head)

    def test_local_append_without_live_proofs_is_blocked(self):
        head = self.unproved_writer.current_head()
        result = self.unproved_writer.append(
            reservation_event(expected_head=head), expected_head=head
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "EXCLUSION_UNAVAILABLE", True),
        )

        static_unproved = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        result = static_unproved.append(
            reservation_event(expected_head=head), expected_head=head
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "STATIC_AUTHORITY_SELECTOR_UNRESOLVED", True),
        )

    def test_local_append_requires_external_exclusion_lease(self):
        head = self.writer.current_head()

        def raising_lease():
            raise OSError("external lease unavailable")

        for lease in (None, lambda: nullcontext(False), raising_lease):
            with self.subTest(lease=lease):
                writer = ReleaseEventWriter(
                    self.repo,
                    exclusion_checker=lambda: True,
                    static_snapshot_checker=lambda _digest: True,
                    exclusion_lease=lease,
                )
                result = writer.append(
                    reservation_event(expected_head=head), expected_head=head
                )
                self.assertEqual(
                    (result.status, result.reason_code, result.frozen),
                    ("BLOCKED", "EXCLUSION_UNAVAILABLE", True),
                )
                self.assertEqual(writer.current_head(), head)

    def test_abort_requires_live_public_tag_absence_checker(self):
        abort = self._prepare_release_intent_abort()
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
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
            exclusion_lease=lambda: nullcontext(True),
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
            exclusion_lease=lambda: nullcontext(True),
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
            exclusion_lease=lambda: nullcontext(True),
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
            exclusion_lease=lambda: nullcontext(True),
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
        writer = ReleaseEventWriter(
            self.repo,
            branch="callback-bootstrap",
            exclusion_lease=lambda: nullcontext(True),
        )
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
        writer = ReleaseEventWriter(
            self.repo,
            branch="callback-bootstrap-no-proof",
            exclusion_lease=lambda: nullcontext(True),
        )
        with self.assertRaises(AppendOutcomeUncertain) as context:
            writer.bootstrap_remote(push=lambda *_args: None, protection_checker=lambda: True)
        self.assertEqual(context.exception.reason_code, "APPEND_OUTCOME_UNCERTAIN")

    def test_bootstrap_push_uncertainty_reconciles_before_freezing(self):
        writer = ReleaseEventWriter(
            self.repo,
            branch="callback-bootstrap-uncertain",
            exclusion_lease=lambda: nullcontext(True),
        )
        observations = [(None, b""), (None, b"")]

        with self.assertRaises(AppendOutcomeUncertain) as context:
            writer.bootstrap_remote(
                push=lambda *_args: (_ for _ in ()).throw(OSError("connection lost")),
                reconcile=lambda: observations.pop(0),
                protection_checker=lambda: True,
            )
        self.assertEqual(context.exception.reason_code, "APPEND_OUTCOME_UNCERTAIN")

    def test_bootstrap_remote_rejects_nonempty_local_ledger_when_remote_is_absent(self):
        remote = self.repo.parent / "nonempty-bootstrap-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        local_head = self.writer.current_head()
        self.writer.append(
            reservation_event(expected_head=local_head), expected_head=local_head
        )
        writer = ReleaseEventWriter(
            self.repo,
            remote="origin",
            exclusion_lease=lambda: nullcontext(True),
        )
        with self.assertRaises(AppendOutcomeUncertain):
            writer.bootstrap_remote(protection_checker=lambda: True)
        advertised = subprocess.run(
            ["git", "-C", str(remote), "show-ref", "--heads"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(advertised.returncode, 1)
        self.assertEqual(advertised.stdout, "")

    def test_bootstrap_remote_rejects_local_empty_descendant_when_remote_is_absent(self):
        remote = self.repo.parent / "descendant-bootstrap-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        local_head = self.writer.current_head()
        appended = self.writer.append(
            reservation_event(expected_head=local_head), expected_head=local_head
        )
        assert appended.head is not None
        empty_descendant = self.writer._make_commit(
            b"", parent=appended.head, message="invalid empty descendant"
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "update-ref",
                "refs/heads/release-events",
                empty_descendant,
            ],
            check=True,
        )
        writer = ReleaseEventWriter(
            self.repo,
            remote="origin",
            exclusion_lease=lambda: nullcontext(True),
        )
        with self.assertRaises(AppendOutcomeUncertain):
            writer.bootstrap_remote(protection_checker=lambda: True)
        advertised = subprocess.run(
            ["git", "-C", str(remote), "show-ref", "--heads"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(advertised.returncode, 1)
        self.assertEqual(advertised.stdout, "")

    def test_bootstrap_remote_materializes_nonempty_head_in_fresh_cache(self):
        remote = self.repo / "fresh-cache-bootstrap-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        branch = "fresh-cache-events"
        authority = ReleaseEventWriter(
            self.repo,
            branch=branch,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        empty_head = authority.bootstrap_remote(protection_checker=lambda: True)
        first = reservation_event(expected_head=empty_head)
        first_result = authority.append_remote(first, expected_head=empty_head)
        self.assertEqual(first_result.status, "APPENDED")
        assert first_result.head is not None

        fresh = self.repo / "fresh-cache-checkout"
        subprocess.run(["git", "init", "-q", str(fresh)], check=True)
        subprocess.run(
            ["git", "-C", str(fresh), "remote", "add", "origin", str(remote)],
            check=True,
        )
        cached = ReleaseEventWriter(
            fresh,
            branch=branch,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        observed = cached.bootstrap_remote(protection_checker=lambda: True)
        self.assertEqual(observed, first_result.head)
        self.assertTrue(cached.branch_exists())
        self.assertEqual(cached.read_head().raw, authority.read_head().raw)

        continuation = reservation_event(
            "EVT-000000000002",
            transaction_id="fresh-cache-continuation",
            expected_head=observed,
        )
        continuation.update(
            final_version="0.3.2",
            intended_dev_version="0.3.2-DEV",
            reservation_id="reservation-2",
        )
        result = cached.append_remote(continuation, expected_head=observed)
        self.assertEqual(result.status, "APPENDED")

    def test_bootstrap_remote_advances_stale_cache_before_append(self):
        remote = self.repo / "stale-cache-bootstrap-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        branch = "stale-cache-events"
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        writer = ReleaseEventWriter(
            self.repo,
            branch=branch,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        empty_head = writer.bootstrap_remote(protection_checker=lambda: True)
        first = reservation_event(expected_head=empty_head)
        first_result = writer.append_remote(first, expected_head=empty_head)
        self.assertEqual(first_result.status, "APPENDED")
        assert first_result.head is not None
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "update-ref",
                f"refs/heads/{branch}",
                empty_head,
                first_result.head,
            ],
            check=True,
        )

        observed = writer.bootstrap_remote(protection_checker=lambda: True)
        self.assertEqual(observed, first_result.head)
        self.assertEqual(writer.current_head(), first_result.head)
        continuation = reservation_event(
            "EVT-000000000002",
            transaction_id="stale-cache-continuation",
            expected_head=observed,
        )
        continuation.update(
            final_version="0.3.2",
            intended_dev_version="0.3.2-DEV",
            reservation_id="reservation-2",
        )
        result = writer.append_remote(continuation, expected_head=observed)
        self.assertEqual(result.status, "APPENDED")

    def test_bootstrap_remote_refuses_to_rewind_ahead_local_cache(self):
        remote = self.repo / "divergent-cache-bootstrap-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        branch = "divergent-cache-events"
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        writer = ReleaseEventWriter(
            self.repo,
            branch=branch,
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        empty_head = writer.bootstrap_remote(protection_checker=lambda: True)
        first = reservation_event(expected_head=empty_head)
        first_result = writer.append_remote(first, expected_head=empty_head)
        self.assertEqual(first_result.status, "APPENDED")
        assert first_result.head is not None
        local_only = reservation_event(
            "EVT-000000000002",
            transaction_id="local-only-continuation",
            expected_head=first_result.head,
        )
        local_only.update(
            final_version="0.3.2",
            intended_dev_version="0.3.2-DEV",
            reservation_id="reservation-2",
        )
        local_result = writer.append(local_only, expected_head=first_result.head)
        self.assertEqual(local_result.status, "APPENDED")
        assert local_result.head is not None

        with self.assertRaises(AppendOutcomeUncertain):
            writer.bootstrap_remote(protection_checker=lambda: True)
        self.assertEqual(writer.current_head(), local_result.head)
        advertised = subprocess.check_output(
            ["git", "-C", str(remote), "rev-parse", f"refs/heads/{branch}"],
            text=True,
        ).strip()
        self.assertEqual(advertised, first_result.head)

    def test_bootstrap_remote_requires_external_exclusion_lease_before_push(self):
        for lease in (None, lambda: nullcontext(False)):
            with self.subTest(lease=lease):
                writer = ReleaseEventWriter(
                    self.repo,
                    branch=(
                        "bootstrap-missing-lease"
                        if lease is None
                        else "bootstrap-false-lease"
                    ),
                    exclusion_lease=lease,
                )
                pushed: list[bool] = []
                with self.assertRaises(ExclusionUnavailable) as context:
                    writer.bootstrap_remote(
                        push=lambda *_args: pushed.append(True),
                        reconcile=lambda: (None, b""),
                        protection_checker=lambda: True,
                    )
                self.assertEqual(context.exception.reason_code, "EXCLUSION_UNAVAILABLE")
                self.assertEqual(pushed, [])

    def test_bootstrap_remote_holds_lock_and_external_lease_through_push(self):
        active = False
        observations = [(None, b"")]

        class TrackingLease:
            def __enter__(self):
                nonlocal active
                active = True
                return True

            def __exit__(self, *_args):
                nonlocal active
                active = False
                return False

        writer = ReleaseEventWriter(
            self.repo,
            branch="bootstrap-lease-lifetime",
            exclusion_lease=lambda: TrackingLease(),
        )
        pushed: list[str] = []

        def reconcile():
            self.assertTrue(active)
            self.assertTrue(writer.static_exclusion.held())
            return observations.pop(0)

        def push(commit, _branch, _expected):
            self.assertTrue(active)
            self.assertTrue(writer.static_exclusion.held())
            pushed.append(commit)
            observations.append((commit, b""))

        result = writer.bootstrap_remote(
            push=push,
            reconcile=reconcile,
            protection_checker=lambda: active,
        )
        self.assertEqual(result, pushed[0])
        self.assertFalse(active)
        self.assertFalse(writer.static_exclusion.held())

    def test_remote_append_lease_release_failure_returns_uncertain_result(self):
        class RaisingLease:
            def __enter__(self):
                return True

            def __exit__(self, *_args):
                raise OSError("lease release failed after remote push")

        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: RaisingLease(),
        )
        head = writer.current_head()
        event = reservation_event(expected_head=head)
        expected_raw = writer.read_head().raw + canonical_event_bytes(event) + b"\n"
        observations = [(head, writer.read_head().raw)]
        pushed: list[str] = []

        def push(commit, _branch, _expected):
            pushed.append(commit)
            observations.append((commit, expected_raw))

        result = writer.append_remote(
            event,
            expected_head=head,
            push=push,
            reconcile=lambda: observations.pop(0),
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "APPEND_OUTCOME_UNCERTAIN", True),
        )
        self.assertEqual(len(pushed), 1)

    def test_bootstrap_remote_static_exclusion_race_blocks_before_observation(self):
        writer = ReleaseEventWriter(
            self.repo,
            branch="bootstrap-static-race",
            exclusion_lease=lambda: nullcontext(True),
        )
        held = writer.static_exclusion.acquire()
        pushed: list[bool] = []
        outcomes: list[BaseException] = []

        def run_bootstrap():
            try:
                writer.bootstrap_remote(
                    push=lambda *_args: pushed.append(True),
                    reconcile=lambda: (None, b""),
                    protection_checker=lambda: True,
                )
            except BaseException as error:  # noqa: BLE001 - capture worker result
                outcomes.append(error)

        try:
            thread = threading.Thread(target=run_bootstrap)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        finally:
            held.release()
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], ExclusionUnavailable)
        self.assertEqual(outcomes[0].reason_code, "EXCLUSION_UNAVAILABLE")
        self.assertEqual(pushed, [])

    def test_real_bare_remote_bootstrap_and_append_verify_one_file_topology(self):
        remote = self.repo.parent / f"{self.repo.name}-release-events-origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "remote", "add", "origin", str(remote)],
            check=True,
        )
        writer = ReleaseEventWriter(
            self.repo,
            branch="release-events-real-remote",
            remote="origin",
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.bootstrap_remote(
            protection_checker=lambda: True,
        )
        event = reservation_event(expected_head=head)
        result = writer.append_remote(event, expected_head=head)
        self.assertEqual(result.status, "APPENDED")
        advertised = subprocess.check_output(
            [
                "git",
                "-C",
                str(remote),
                "rev-parse",
                "refs/heads/release-events-real-remote",
            ],
            text=True,
        ).strip()
        self.assertEqual(advertised, result.head)
        tree = subprocess.check_output(
            [
                "git",
                "-C",
                str(remote),
                "ls-tree",
                "--name-only",
                advertised,
            ],
            text=True,
        ).splitlines()
        self.assertEqual(tree, ["release-events.jsonl"])

    def test_commit_identity_ignores_private_ambient_git_identity(self):
        ambient = {
            "GIT_AUTHOR_NAME": "Private Ambient Author",
            "GIT_AUTHOR_EMAIL": "private-author@example.invalid",
            "GIT_COMMITTER_NAME": "Private Ambient Committer",
            "GIT_COMMITTER_EMAIL": "private-committer@example.invalid",
        }
        with patch.dict(os.environ, ambient, clear=False):
            commit = self.writer._make_commit(
                b"", parent=self.writer.current_head(), message="ambient identity test"
            )
        identity = subprocess.check_output(
            [
                "git",
                "-C",
                str(self.repo),
                "show",
                "-s",
                "--format=%an%x00%ae%x00%cn%x00%ce",
                commit,
            ],
            text=True,
        ).strip()
        self.assertEqual(
            identity,
            "CYAxiverse Lifecycle Writer\x00release-events@cyaxiverse.invalid\x00"
            "CYAxiverse Lifecycle Writer\x00release-events@cyaxiverse.invalid",
        )

    def test_callback_reconciliation_cannot_claim_unverified_topology(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.current_head()
        pushed: list[bool] = []
        result = writer.append_remote(
            reservation_event(expected_head=head),
            expected_head=head,
            push=lambda *_args: pushed.append(True),
            reconcile=lambda: ("f" * 40, b""),
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "APPEND_OUTCOME_UNCERTAIN", True),
        )
        self.assertEqual(pushed, [])

    def test_callback_reconciliation_rejects_malformed_history(self):
        writer = ReleaseEventWriter(
            self.repo,
            exclusion_checker=lambda: True,
            static_snapshot_checker=lambda _digest: True,
            exclusion_lease=lambda: nullcontext(True),
        )
        head = writer.current_head()
        malformed = writer._make_commit(b"", parent=head, message="empty descendant")
        pushed: list[bool] = []
        result = writer.append_remote(
            reservation_event(expected_head=head),
            expected_head=head,
            push=lambda *_args: pushed.append(True),
            reconcile=lambda: (malformed, b""),
        )
        self.assertEqual(
            (result.status, result.reason_code, result.frozen),
            ("BLOCKED", "APPEND_OUTCOME_UNCERTAIN", True),
        )
        self.assertEqual(pushed, [])


class EventIntegrationAuditTests(unittest.TestCase):
    """Regression coverage for public-boundary event validation."""

    def test_abort_proof_binds_tag_and_exclusion_fields(self):
        abort = release_intent_abort_event()
        proof = dict(abort["no_public_tag_evidence"])
        proof["public_tag"] = "v0.3.2"
        abort["no_public_tag_evidence"] = proof
        with self.assertRaises(EventSchemaError):
            validate_event(abort)

        abort = release_intent_abort_event()
        proof = dict(abort["no_public_tag_evidence"])
        proof["exclusion_verified"] = False
        abort["no_public_tag_evidence"] = proof
        with self.assertRaises(EventSchemaError):
            validate_event(abort)

        candidate = candidate_event()
        intent = release_intent_event()
        abort = release_intent_abort_event(candidate_id="candidate-other")
        with self.assertRaises(EventTransitionError):
            validate_transition([candidate, intent], abort)

    def test_public_boundary_rejects_durable_locators_and_does_not_echo_values(self):
        cases = (
            (reservation_event(), "expected_line_head", "/Users/secret/line-head"),
            (release_intent_event(), "certification_environment", "file:///Users/secret/env"),
            (release_intent_event(), "certification_evidence_refs", ["Bearer SECRET-CREDENTIAL"]),
        )
        for event, field, value in cases:
            event[field] = value
            with self.subTest(field=field):
                with self.assertRaises(EventSchemaError) as context:
                    validate_event(event)
                self.assertNotIn("SECRET-CREDENTIAL", str(context.exception))
                self.assertNotIn("/Users/secret", str(context.exception))

    def test_maintenance_reservation_rejects_wrong_major_minor_for_each_state_shape(self):
        prepared = reservation_event()
        prepared.update(
            owner_line="maintenance/1.2",
            final_version="1.3.1",
            intended_dev_version="1.3.1-DEV",
        )
        with self.assertRaises(EventSchemaError):
            validate_event(prepared)

        opened = reservation_event(event_type="development_reservation_opened")
        opened.update(
            owner_line="maintenance/1.2",
            final_version="1.3.1",
            intended_dev_version="1.3.1-DEV",
        )
        with self.assertRaises(EventSchemaError):
            validate_event(opened)

        aborted = reservation_event(event_type="development_reservation_aborted")
        aborted.pop("expected_line_head")
        aborted.update(
            owner_line="maintenance/1.2",
            final_version="1.3.1",
            intended_dev_version="1.3.1-DEV",
            abort_reason="fixture",
            non_entry_evidence="evidence/non-entry",
        )
        with self.assertRaises(EventSchemaError):
            validate_event(aborted)

    def test_schema_version_requires_exact_integer_one(self):
        for schema_version in (True, False, "1"):
            event = reservation_event()
            event["schema_version"] = schema_version
            with self.subTest(schema_version=schema_version):
                with self.assertRaises(EventSchemaError):
                    validate_event(event)

    def test_maintenance_line_open_binds_base_and_branch(self):
        opened = reservation_event(
            event_id_value="EVT-000000000001",
            transaction_id="maintenance-opened",
            expected_head="b" * 40,
            event_type="development_reservation_opened",
        )
        opened.update(
            owner_line="maintenance/1.2",
            final_version="1.2.1",
            intended_dev_version="1.2.1-DEV",
        )
        line_opened = {
            "schema_version": 1,
            "event_id": "EVT-000000000002",
            "event_type": "maintenance_line_opened",
            "timestamp_utc": "2026-09-20T12:34:56Z",
            "transaction_id": "maintenance-line",
            "static_iteration_snapshot": SNAPSHOT,
            "expected_event_head": "b" * 40,
            "release_line": "maintenance/1.2",
            "approved_base_version": "1.2.0",
            "branch_ref": "refs/heads/maintenance/1.2",
            "branch_head": "d" * 40,
            "reservation_id": opened["reservation_id"],
            "dev_version": "1.2.1-DEV",
        }
        validate_event(line_opened)
        validate_transition([opened], line_opened)

        wrong_base = dict(line_opened, approved_base_version="1.3.0")
        with self.assertRaises(EventSchemaError):
            validate_event(wrong_base)
        wrong_branch = dict(line_opened, branch_ref="refs/heads/maintenance/1.2-extra")
        with self.assertRaises(EventSchemaError):
            validate_event(wrong_branch)
        wrong_shape = dict(line_opened, approved_base_version="1.2")
        with self.assertRaises(EventSchemaError):
            validate_event(wrong_shape)

    def test_candidate_refs_use_git_invalid_component_policy(self):
        released = dict(
            (key, value)
            for key, value in release_intent_event().items()
            if key != "intent_id"
        )
        released.update(
            event_id="EVT-000000000003",
            event_type="released",
            transaction_id="released-1",
            closure_timestamp_utc="2026-09-20T12:34:56Z",
            evidence_refs=["evidence/released-1"],
            main_at_event_sha="f" * 40,
            main_at_event_version="0.3.1",
            previous_main_sha="d" * 40,
            previous_main_version="0.2.0",
        )
        shapes = (
            ("candidate_opened", candidate_event()),
            ("release_intent_prepared", release_intent_event()),
            ("released", released),
        )
        invalid_refs = (
            "refs/heads/candidates/../0.3.1",
            "refs/heads/candidates//0.3.1",
            "refs/heads/candidates/.0.3.1",
            "refs/heads/candidates/0.3.1.lock",
        )
        for event_type, event in shapes:
            for candidate_ref in invalid_refs:
                bad = dict(event, candidate_ref=candidate_ref)
                with self.subTest(event_type=event_type, candidate_ref=candidate_ref):
                    with self.assertRaises(EventSchemaError):
                        validate_event(bad)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
