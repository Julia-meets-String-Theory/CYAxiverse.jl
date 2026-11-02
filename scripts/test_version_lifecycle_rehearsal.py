"""Offline contract, approval, and fixed-route tests for rehearsal runtime."""

from __future__ import annotations

import base64
import hashlib
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.codec import sha256_hex
from version_lifecycle.github_protection import (
    ApiResponse,
    GitHubProtectionAdapter,
    GitHubProtectionError,
    RehearsalFreezeProof,
)
from version_lifecycle.rehearsal import (
    APPROVAL_MARKER_PREFIX,
    CASE_ORDER,
    OPERATION_TABLE,
    REPOSITORY,
    STATUS_AWARE_VECTORS,
    CaseObservation,
    AnonymousGitHubTransport,
    RehearsalError,
    ValidatedRouteRequest,
    VerifiedRunApproval,
    _APPROVAL_VERIFICATION_TOKEN,
    _CONTRACT_SHA256,
    _canonical,
    _result,
    run_rehearsal,
    validate_seed_receipt_bundle,
    validate_approval_receipt,
    validate_driver_result,
    validate_route_request_bytes,
    validate_rehearsal_route_trigger,
    verify_run_approval,
    rehydrate_approval_envelope,
    serialize_approval_envelope,
    rehydrate_seed_receipt_evidence,
    rehydrate_seed_receipt_evidence_pair,
    serialize_seed_receipt_evidence,
    serialize_seed_receipt_evidence_pair,
    _validate_jobs_response,
)


RUN_ID = 9854321
HEAD_SHA = "d" * 40
REF = "refs/heads/rehearsal/trigger/route-0125-1"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
SEED_TARGET_SHA = "f12fbb8ec52e4c7510564f602a15f26173fe772e"
SEED_TARGET_TREE = "c515bf2745061da9150285d6d7c6035b061a1caa"
SEED_REFS = {
    "principal": "refs/heads/rehearsal/principal/cyax125-20261101-route",
    "P2-update": "refs/heads/rehearsal/trigger/cyax125-p2-app-update",
    "4f-move": "refs/heads/rehearsal/restricted/cyax125-20261101-move",
}


def _synthetic_seed_bundle(intent_id: str) -> dict[str, object]:
    def envelope(status: int, body: bytes, suffix: str) -> dict[str, object]:
        return {
            "status": status,
            "github_request_id": f"SYNTHETIC-{suffix}",
            "date_header": "Mon, 02 Nov 2026 12:00:00 GMT",
            "body_b64": base64.b64encode(body).decode("ascii"),
            "body_sha256": sha256_hex(body),
        }

    records: list[dict[str, object]] = []
    for index, role in enumerate(("principal", "P2-update", "4f-move"), start=1):
        ref = SEED_REFS[role]
        identity_body = _canonical({"id": 102535039, "login": "vmmhep", "type": "User"})
        ref_body = _canonical({"ref": ref, "object": {"sha": SEED_TARGET_SHA, "type": "commit"}})
        journal = {
            "journal_version": 1,
            "repository": REPOSITORY,
            "intent_id": intent_id,
            "role": role,
            "ref": ref,
            "target_sha": SEED_TARGET_SHA,
            "target_tree": SEED_TARGET_TREE,
            "actor_identity": envelope(200, identity_body, f"{role}-identity"),
            "create_method": "POST",
            "create_path": f"/repos/{REPOSITORY}/git/refs",
            "create_payload": {"ref": ref, "sha": SEED_TARGET_SHA},
            "create_response": envelope(201, ref_body, f"{role}-create"),
            "readback_method": "GET",
            "readback_path": f"/repos/{REPOSITORY}/git/ref/{ref.removeprefix('refs/')}",
            "readback_response": envelope(200, ref_body, f"{role}-readback"),
            "operation_started_at": "2026-11-02T12:00:00Z",
            "operation_completed_at": "2026-11-02T12:00:01Z",
            "terminal_state": "CREATED_AND_READ_BACK",
            "no_pending_writes": True,
        }
        journal_hash = sha256_hex(_canonical(journal))
        claim = {
            "claim_version": 1,
            "repository": REPOSITORY,
            "issue_number": 125,
            "intent_id": intent_id,
            "role": role,
            "ref": ref,
            "target_sha": SEED_TARGET_SHA,
            "target_tree": SEED_TARGET_TREE,
            "journal_sha256": journal_hash,
            "owner_user_id": 102535039,
            "terminal_state": "CREATED_AND_READ_BACK",
            "no_pending_writes": True,
        }
        claim_body = b"CYAX125_SEED_V1:" + _canonical(claim)
        records.append({
            "role": role,
            "journal": journal,
            "journal_sha256": journal_hash,
            "claim": claim,
            "owner_action_claim_sha256": sha256_hex(claim_body),
            "owner_receipt_comment_id": 100 + index,
        })
    return {
        "seed_version": 1,
        "repository": REPOSITORY,
        "intent_id": intent_id,
        "records": records,
    }


def _request_bytes(
    *,
    implementation_sha: str = "a" * 40,
    candidate_tree: str = "b" * 40,
    workflow_blob_sha: str = "c" * 40,
    operation_table_sha256: str | None = None,
    called_code_blobs: dict[str, str] | None = None,
) -> bytes:
    code = called_code_blobs or {
        f"scripts/version_lifecycle/fake_{index:02d}.py": f"{index:040x}"
        for index in range(16)
    }
    seed_bundle = _synthetic_seed_bundle("intent-route-0125")
    plan = {
        "plan_version": 1,
        "repository": REPOSITORY,
        "candidate_sha": implementation_sha,
        "candidate_tree": candidate_tree,
        "workflow_blob_sha": workflow_blob_sha,
        "called_code_blobs": code,
        "fixture_base": "f12fbb8ec52e4c7510564f602a15f26173fe772e",
        "fixture_ff": "a6df7c78e27ee81771cc06b0783a508edcaed320",
        "fixture_workflow_path": "32af89296955ad684517e7a57ce5d8a431f5a971",
        "setup_inventory_sha256": "1" * 64,
        "environment_export_sha256": "2" * 64,
        "seed_receipts_sha256": sha256_hex(_canonical(seed_bundle)),
        "operation_table_sha256": operation_table_sha256 or sha256_hex(_canonical(OPERATION_TABLE)),
        "case_order": list(CASE_ORDER),
        "owner_actor_id": 102535039,
        "app_id": 5371554,
        "installation_id": 172789630,
        "intent_id": "intent-route-0125",
    }
    plan_sha = sha256_hex(_canonical(plan))
    attestation = {
        "attestation_version": 1,
        "request_id": "route-0125-1",
        "intent_id": plan["intent_id"],
        "plan_sha256": plan_sha,
        "candidate_sha": plan["candidate_sha"],
        "epoch_id": "owner-epoch-1",
        "freeze_ruleset_id": 26316866,
        "full_definition_sha256": "4" * 64,
        "effective_rule_ids": [26314086, 26314227, 26316866],
        "owner_actor_id": 102535039,
        "last_operation_terminal_receipt_sha256": "5" * 64,
        "no_pending_or_unknown_writes": True,
        "exclusive_owner_writer_until_reconciliation": True,
        "epoch_started_at": "2026-11-02T11:00:00Z",
    }
    request = {
        "schema_version": 3,
        "request_id": "route-0125-1",
        "implementation_sha": implementation_sha,
        "workflow_blob_sha": plan["workflow_blob_sha"],
        "expected_vmm_sha": plan["fixture_base"],
        "called_code_blobs": code,
        "purpose": "gate-a-route-proof",
        "plan": plan,
        "plan_sha256": plan_sha,
        "owner_attestation": attestation,
        "owner_attestation_sha256": sha256_hex(_canonical(attestation)),
        "seed_receipt_bundle": seed_bundle,
    }
    return _canonical(request)


def _request() -> ValidatedRouteRequest:
    parsed = validate_route_request_bytes(_request_bytes())
    return replace(parsed, commit_sha=HEAD_SHA)


def _marker(request: ValidatedRouteRequest, **overrides: object) -> str:
    marker = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "run_id": RUN_ID,
        "run_attempt": 1,
        "head_sha": HEAD_SHA,
        "request_id": request.request_id,
        "plan_sha256": request.plan_sha256,
        "environment_id": 24648183439,
        "environment_name": "lifecycle",
        "owner_user_id": 102535039,
        "no_admin_bypass_used": True,
    }
    marker.update(overrides)
    return APPROVAL_MARKER_PREFIX + _canonical(marker).decode("ascii")


def _provider_payloads(request: ValidatedRouteRequest) -> dict[str, object]:
    run_path = f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}"
    return {
        f"/repos/{REPOSITORY}": {
            "id": 527718962, "full_name": REPOSITORY, "private": False,
        },
        f"{run_path}/attempts/1": {
            "id": RUN_ID,
            "run_attempt": 1,
            "event": "push",
            "head_sha": HEAD_SHA,
            "head_branch": REF.removeprefix("refs/heads/"),
            "path": ".github/workflows/cyax-lifecycle-dispatch.yml@refs/heads/main",
            "repository": {"id": 527718962, "full_name": REPOSITORY},
            "actor": {"id": 102535039, "login": "owner"},
        },
        f"{run_path}/approvals": [{
            "state": "approved",
            "user": {"id": 102535039, "login": "owner"},
            "environments": [{"id": 24648183439, "name": "lifecycle"}],
            "comment": _marker(request),
        }],
        f"{run_path}/attempts/1/jobs?per_page=100": {
            "total_count": 1,
            "jobs": [{
                "id": 55779911,
                "run_id": RUN_ID,
                "name": "Validate and fence an Owner rehearsal request",
                "head_sha": HEAD_SHA,
                "status": "in_progress",
                "steps": [
                    {"name": "Fetch the exact trigger commit with anonymous read-only access", "status": "completed", "conclusion": "success"},
                    {"name": "Validate exact route and approval evidence before App mint", "status": "in_progress", "conclusion": None},
                    {"name": "Mint a scoped token and execute the fixed rehearsal route", "status": "queued", "conclusion": None},
                ],
            }],
        },
    }


class FakeApprovalTransport:
    def __init__(self, payloads: dict[str, object] | None = None) -> None:
        self.payloads = payloads or _provider_payloads(_request())
        self.calls: list[str] = []
        self.overrides: dict[tuple[str, int], ApiResponse] = {}
        self.default_headers = {"content-type": "application/json; charset=utf-8", "etag": '"stable"', "x-github-request-id": "fake-request"}

    def get(self, path: str) -> ApiResponse:
        count = self.calls.count(path)
        self.calls.append(path)
        if (path, count) in self.overrides:
            return self.overrides[(path, count)]
        body = json.dumps(self.payloads[path], sort_keys=True, separators=(",", ":")).encode("utf-8")
        return ApiResponse(200, body, dict(self.default_headers))


def _seed_provider_payloads(request: ValidatedRouteRequest) -> dict[str, object]:
    payloads: dict[str, object] = {}
    bundle = request.request["seed_receipt_bundle"]
    for record in bundle["records"]:
        claim_body = b"CYAX125_SEED_V1:" + _canonical(record["claim"])
        payloads[f"/repos/{REPOSITORY}/issues/comments/{record['owner_receipt_comment_id']}"] = {
            "id": record["owner_receipt_comment_id"],
            "user": {"id": 102535039, "type": "User"},
            "issue_url": f"https://api.github.com/repos/{REPOSITORY}/issues/125",
            "body": claim_body.decode("ascii"),
        }
        payloads[f"/repos/{REPOSITORY}/git/ref/heads/{record['journal']['ref'].removeprefix('refs/heads/')}"] = {
            "ref": record["journal"]["ref"],
            "object": {"sha": SEED_TARGET_SHA, "type": "commit"},
        }
    payloads[f"/repos/{REPOSITORY}/git/commits/{SEED_TARGET_SHA}"] = {
        "sha": SEED_TARGET_SHA, "tree": {"sha": SEED_TARGET_TREE},
    }
    return payloads


class FakeSeedTransport:
    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []
        self.overrides: dict[str, ApiResponse] = {}

    def get(self, path: str) -> ApiResponse:
        self.calls.append(path)
        if path in self.overrides:
            return self.overrides[path]
        body = json.dumps(self.payloads[path], sort_keys=True, separators=(",", ":")).encode("utf-8")
        return ApiResponse(200, body, {
            "content-type": "application/json", "date": "Mon, 02 Nov 2026 12:00:00 GMT",
            "x-github-request-id": "SYNTHETIC-READ",
        })


def _approval(request: ValidatedRouteRequest) -> VerifiedRunApproval:
    return verify_run_approval(
        request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, ref=REF,
        transport=FakeApprovalTransport(_provider_payloads(request)),
        monotonic=lambda: 1.0, utc_now=lambda: NOW,
    )


class RehearsalContractTests(unittest.TestCase):
    def test_embedded_contract_pins_and_all_status_vectors(self) -> None:
        self.assertEqual(len(STATUS_AWARE_VECTORS), 33)
        for vector in STATUS_AWARE_VECTORS:
            with self.subTest(vector=vector["name"]):
                try:
                    validate_driver_result(vector["return"])
                    accepted = True
                except RehearsalError:
                    accepted = False
                self.assertEqual(accepted, vector["expected_valid"])
                self.assertEqual(accepted, vector["actual_valid"])

    def test_request_is_canonical_closed_and_cross_bound(self) -> None:
        request = _request()
        self.assertEqual(request.request_id, "route-0125-1")
        self.assertEqual(request.plan_sha256, request.owner_attestation["plan_sha256"])
        self.assertEqual(
            request.plan["operation_table_sha256"],
            sha256_hex(_canonical(OPERATION_TABLE)),
        )
        raw = request.request_bytes
        with self.assertRaises(RehearsalError):
            validate_route_request_bytes(raw + b"\n")
        with self.assertRaises(RehearsalError):
            validate_route_request_bytes(raw.replace(b'"schema_version":3', b'"schema_version":true'))
        with self.assertRaises(RehearsalError):
            validate_route_request_bytes(raw.replace(b'"schema_version":3', b'"schema_version":3,"schema_version":3'))
        with self.assertRaisesRegex(RehearsalError, "ROUTE_REQUEST_BINDING_INVALID"):
            validate_route_request_bytes(_request_bytes(
                operation_table_sha256=_CONTRACT_SHA256["operation_table"]
            ))

    def test_schema3_seed_receipts_verify_journals_claims_comments_and_fixed_readbacks(self) -> None:
        request = _request()
        transport = FakeSeedTransport(_seed_provider_payloads(request))
        evidence = validate_seed_receipt_bundle(request, transport)
        self.assertEqual(evidence.status, "PASS_ACCEPTED")
        self.assertEqual(evidence.verified_roles, ("principal", "P2-update", "4f-move"))
        self.assertEqual(len(transport.calls), 7)
        self.assertTrue(all("issues/comments/" in path or "/git/ref/heads/rehearsal/" in path or "/git/commits/" in path for path in transport.calls))
        self.assertEqual(len(evidence.observations), 7)
        first = rehydrate_seed_receipt_evidence(
            serialize_seed_receipt_evidence(evidence), request
        )
        second = validate_seed_receipt_bundle(
            request, FakeSeedTransport(_seed_provider_payloads(request))
        )
        pair = serialize_seed_receipt_evidence_pair(first, second)
        restored = rehydrate_seed_receipt_evidence_pair(pair, request)
        self.assertEqual(len(restored["first"]["observations"]), 7)
        self.assertEqual(len(restored["second"]["observations"]), 7)
        self.assertNotEqual(
            restored["first"]["live_observation_sha256"],
            restored["second"]["live_observation_sha256"],
        )
        self.assertTrue(all(
            item["method"] == "GET" and item["body_b64"]
            and sha256_hex(base64.b64decode(item["body_b64"])) == item["body_sha256"]
            for item in restored["first"]["observations"]
        ))

        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
            token_permissions={"contents": "write", "metadata": "read"},
            token_expires_at=NOW + timedelta(hours=1), clock=lambda: NOW,
        )
        def create_fence(*_args, **_kwargs):
            adapter._initial_approval_evidence[request.request_id] = {"approval": "verified"}
            return {"observed_commit_sha": "8" * 40}
        with mock.patch.object(adapter, "create_rehearsal_request_fence", side_effect=create_fence):
            adapter.create_rehearsal_route_request_fence(
                request, approval=_approval(request),
                intent={"repository": REPOSITORY, "request_id": request.request_id},
                seed_validation_evidence=pair,
            )
        first = OPERATION_TABLE[0]
        proof = RehearsalFreezeProof(
            request.owner_attestation["epoch_id"], request.plan["intent_id"],
            request.request_id, request.plan_sha256, first["ref"], 26316866,
            request.owner_attestation["full_definition_sha256"],
            (26314086, 26314227, 26316866),
            request.owner_attestation["last_operation_terminal_receipt_sha256"],
            request.owner_attestation["full_definition_sha256"], 0, None,
            "2026-11-02T12:00:00Z", adapter._nonce,
        )
        claim_record = {}
        with (
            adapter.case_lock(first["ref"]),
            mock.patch.object(adapter, "_create_rehearsal_record_ref", side_effect=lambda **kwargs: claim_record.update(kwargs["record"]) or "9" * 40),
        ):
            adapter.claim_case(request, first["case"], proof)
        durable_pair = claim_record["approval_evidence"]["seed_validation_evidence"]
        self.assertEqual(len(durable_pair["first"]["observations"]), 7)
        self.assertEqual(len(durable_pair["second"]["observations"]), 7)

    def test_seed_evidence_rehydration_replays_producer_authority_checks(self) -> None:
        request = _request()
        payloads = _seed_provider_payloads(request)
        first = validate_seed_receipt_bundle(request, FakeSeedTransport(payloads))
        second = validate_seed_receipt_bundle(request, FakeSeedTransport(payloads))
        original = json.loads(serialize_seed_receipt_evidence(first))

        def set_observation_body(observation, value):
            body = _canonical(value)
            observation["body_b64"] = base64.b64encode(body).decode("ascii")
            observation["body_sha256"] = sha256_hex(body)
            observation["body_bytes"] = len(body)

        def rehash(value):
            value.pop("evidence_sha256", None)
            value["live_observation_sha256"] = sha256_hex(
                _canonical(value["observations"])
            )
            value["evidence_sha256"] = sha256_hex(_canonical(value))
            return _canonical(value)

        def changed(mutator):
            value = json.loads(json.dumps(original))
            value.pop("evidence_sha256", None)
            mutator(value)
            return rehash(value)

        def comment(observation):
            return json.loads(base64.b64decode(observation["body_b64"]))

        def update_comment(field, replacement):
            def apply(value):
                observation = value["observations"][0]
                row = comment(observation)
                row[field] = replacement
                set_observation_body(observation, row)
            return apply

        def update_ref(field, replacement):
            def apply(value):
                observation = value["observations"][1]
                row = json.loads(base64.b64decode(observation["body_b64"]))
                row[field] = replacement
                set_observation_body(observation, row)
            return apply

        mutations = [
            ("rehash-503", lambda value: (
                value["observations"][0].__setitem__("status", 503),
                set_observation_body(value["observations"][0], {"message": "unavailable"}),
            )),
            ("rehash-404", lambda value: (
                value["observations"][0].__setitem__("status", 404),
                set_observation_body(value["observations"][0], {"message": "missing"}),
            )),
            ("wrong-issue-126", update_comment(
                "issue_url", f"https://api.github.com/repos/{REPOSITORY}/issues/126"
            )),
            ("wrong-owner-actor", update_comment("user", {"id": 100, "type": "User"})),
            ("wrong-claim-body", update_comment("body", "CYAX125_SEED_V1:{}")),
            ("wrong-current-ref-sha", lambda value: set_observation_body(
                value["observations"][1],
                {"ref": SEED_REFS["principal"],
                 "object": {"sha": "0" * 40, "type": "commit"}},
            )),
            ("wrong-current-ref-name", update_ref("ref", "refs/heads/rehearsal/other")),
            ("wrong-observation-path", lambda value: value["observations"][0].__setitem__(
                "path", "/repos/Julia-meets-String-Theory/CYAxiverse.jl/issues/125"
            )),
            ("wrong-seed-tree", lambda value: set_observation_body(
                value["observations"][-1],
                {"sha": SEED_TARGET_SHA, "tree": {"sha": "0" * 40}},
            )),
            ("wrong-seed-commit", lambda value: set_observation_body(
                value["observations"][-1],
                {"sha": "0" * 40, "tree": {"sha": SEED_TARGET_TREE}},
            )),
            ("wrong-role", lambda value: value["observations"][0].__setitem__("role", "P2-update")),
            ("boolean-status-alias", lambda value: value["observations"][0].__setitem__("status", True)),
            ("boolean-body-size-alias", lambda value: value["observations"][0].__setitem__("body_bytes", True)),
            ("non-string-header-value", lambda value: value["observations"][0]["headers"].__setitem__(
                "date", 1
            )),
            ("missing-observation-field", lambda value: value["observations"][0].pop("started_at")),
            ("extra-observation-field", lambda value: value["observations"][0].__setitem__("unbound", "x")),
            ("invalid-time", lambda value: value["observations"][0].__setitem__(
                "started_at", "2026-99-99T99:99:99.000000+00:00"
            )),
            ("unknown-header", lambda value: value["observations"][0]["headers"].__setitem__(
                "x-unreviewed", "value"
            )),
            ("boolean-schema-version", lambda value: value.__setitem__("schema_version", True)),
        ]

        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
            token_permissions={"contents": "write", "metadata": "read"},
            token_expires_at=NOW + timedelta(hours=1), clock=lambda: NOW,
        )
        for name, mutate in mutations:
            with self.subTest(case=name):
                bad_first = changed(mutate)
                with self.assertRaisesRegex(RehearsalError, "SEED_EVIDENCE_REHYDRATION_INVALID"):
                    rehydrate_seed_receipt_evidence(bad_first, request)

                bad_value = json.loads(bad_first)
                bad_value.pop("evidence_sha256")
                bad_pair = _canonical({
                    "schema_version": 1, "first": bad_value, "second": second.to_dict(),
                })
                with self.assertRaisesRegex(RehearsalError, "SEED_EVIDENCE_PAIR_REHYDRATION_INVALID"):
                    rehydrate_seed_receipt_evidence_pair(bad_pair, request)
                with mock.patch.object(adapter, "create_rehearsal_request_fence") as create_fence:
                    with self.assertRaisesRegex(GitHubProtectionError, "SEED_VALIDATION_EVIDENCE_INVALID"):
                        adapter.create_rehearsal_route_request_fence(
                            request, approval=None, intent={}, seed_validation_evidence=bad_pair,
                        )
                    create_fence.assert_not_called()

        float_value = json.loads(json.dumps(original))
        float_value.pop("evidence_sha256", None)
        float_value["observations"][0]["status"] = 200.0
        float_value["live_observation_sha256"] = "0" * 64
        float_value["evidence_sha256"] = "0" * 64
        float_raw = json.dumps(
            float_value, sort_keys=True, separators=(",", ":"),
        ).encode("ascii")
        with self.assertRaisesRegex(RehearsalError, "SEED_EVIDENCE_REHYDRATION_INVALID"):
            rehydrate_seed_receipt_evidence(float_raw, request)
        float_value.pop("evidence_sha256")
        float_pair = json.dumps({
            "schema_version": 1, "first": float_value, "second": second.to_dict(),
        }, sort_keys=True, separators=(",", ":")).encode("ascii")
        with self.assertRaisesRegex(RehearsalError, "SEED_EVIDENCE_PAIR_REHYDRATION_INVALID"):
            rehydrate_seed_receipt_evidence_pair(float_pair, request)
        with mock.patch.object(adapter, "create_rehearsal_request_fence") as create_fence:
            with self.assertRaisesRegex(GitHubProtectionError, "SEED_VALIDATION_EVIDENCE_INVALID"):
                adapter.create_rehearsal_route_request_fence(
                    request, approval=None, intent={}, seed_validation_evidence=float_pair,
                )
            create_fence.assert_not_called()

        valid_pair = serialize_seed_receipt_evidence_pair(first, second)
        pair_value = json.loads(valid_pair)
        pair_value["schema_version"] = True
        invalid_pair_schema = _canonical(pair_value)
        with self.assertRaisesRegex(RehearsalError, "SEED_EVIDENCE_PAIR_REHYDRATION_INVALID"):
            rehydrate_seed_receipt_evidence_pair(invalid_pair_schema, request)
        with mock.patch.object(adapter, "create_rehearsal_request_fence") as create_fence:
            with self.assertRaisesRegex(GitHubProtectionError, "SEED_VALIDATION_EVIDENCE_INVALID"):
                adapter.create_rehearsal_route_request_fence(
                    request, approval=None, intent={}, seed_validation_evidence=invalid_pair_schema,
                )
            create_fence.assert_not_called()

    def test_actual_embedded_preflight_writes_rehashable_second_failure_evidence(self) -> None:
        workflow = (
            Path(__file__).resolve().parent.parent
            / ".github/workflows/cyax-lifecycle-dispatch.yml"
        ).read_text(encoding="utf-8")
        match = re.search(
            r"(?m)^\s*preflight_python = <<~PY\n(?P<body>.*?)^\s*PY\s*$",
            workflow, re.S,
        )
        self.assertIsNotNone(match)
        preflight = textwrap.dedent(match.group("body"))
        self.assertIn('"observation_preimage": [', workflow)

        request = _request()
        first_raw = serialize_seed_receipt_evidence(
            validate_seed_receipt_bundle(
                request, FakeSeedTransport(_seed_provider_payloads(request))
            )
        )
        wrapper = f'''\
import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from test_version_lifecycle_rehearsal import FakeSeedTransport, _request, _seed_provider_payloads
from version_lifecycle.github_protection import ApiResponse
import version_lifecycle.rehearsal as rehearsal

rehearsal.validate_rehearsal_route_trigger = lambda *_args, **_kwargs: _request()
rehearsal.rehydrate_approval_envelope = lambda *_args, **_kwargs: object()
producer_validate = rehearsal.validate_seed_receipt_bundle
def fail_second(request, _transport):
    if isinstance(_transport, rehearsal._RecordedSeedTransport):
        return producer_validate(request, _transport)
    mode = os.environ["SEED_FAILURE_CASE"]
    transport = FakeSeedTransport(_seed_provider_payloads(request))
    comment_path = next(path for path in transport.payloads if "/issues/comments/" in path)
    calls = []
    original_get = transport.get
    def synthetic_get(path):
        calls.append(path)
        if path == comment_path:
            if mode == "503":
                return ApiResponse(503, b'{{"message":"unavailable"}}', {{}})
            if mode == "404":
                return ApiResponse(404, b'{{"message":"missing"}}', {{}})
            if mode == "changed":
                row = dict(transport.payloads[comment_path])
                row["issue_url"] = f"https://api.github.com/repos/{{rehearsal.REPOSITORY}}/issues/126"
                return ApiResponse(200, rehearsal._canonical(row), {{}})
            if mode == "transport":
                raise OSError("synthetic transport unavailable")
        return original_get(path)
    transport.get = synthetic_get
    try:
        return producer_validate(request, transport)
    finally:
        Path(os.environ["RUNNER_TEMP"], "seed-get-calls.json").write_text(
            json.dumps(calls), encoding="ascii"
        )
rehearsal.validate_seed_receipt_bundle = fail_second
exec({preflight!r}, {{"__name__": "__main__"}})
'''
        cases = ("503", "404", "changed", "transport", "malformed-first")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = root / "runner"
            runner.mkdir(mode=0o700)
            event = root / "event.json"
            event.write_text("{}", encoding="ascii")
            approval = root / "approval.json"
            approval.write_bytes(b"{}")
            script_dir = str(Path(__file__).resolve().parent)
            for case in cases:
                with self.subTest(case=case):
                    shutil.rmtree(runner)
                    runner.mkdir(mode=0o700)
                    seed_path = runner / "cyax125-seed-validation-first.json"
                    seed_path.write_bytes(b"{\"invalid\":true}" if case == "malformed-first" else first_raw)
                    env = {
                        "PATH": os.environ["PATH"],
                        "PYTHONPATH": str(Path(__file__).resolve().parent.parent) + os.pathsep + script_dir,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "RUNNER_TEMP": str(runner),
                        "GITHUB_EVENT_PATH": str(event),
                        "GITHUB_REPOSITORY": REPOSITORY,
                        "GITHUB_REF": REF,
                        "GITHUB_SHA": HEAD_SHA,
                        "GITHUB_ACTOR": "vmmhep",
                        "GITHUB_RUN_ID": str(RUN_ID),
                        "GITHUB_RUN_ATTEMPT": "1",
                        "ROUTE_MODE": "true",
                        "APPROVAL_RAW_ENVELOPE_PATH": str(approval),
                        "SEED_VALIDATION_RAW_PATH": str(seed_path),
                        "SEED_FAILURE_CASE": case,
                    }
                    result = subprocess.run(
                        [sys.executable, "-c", wrapper], cwd=root, env=env,
                        capture_output=True, text=True, timeout=20, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    failure_path = runner / "cyax125-seed-validation-pair-failure.json"
                    self.assertTrue(failure_path.is_file(), result.stderr)
                    self.assertEqual(failure_path.stat().st_mode & 0o777, 0o600)
                    failure_raw = failure_path.read_bytes()
                    failure = json.loads(failure_raw)
                    self.assertEqual(_canonical(failure), failure_raw)
                    self.assertIs(type(failure["observation_preimage"]), list)
                    self.assertEqual(
                        base64.b64decode(failure["first_validation_preimage_b64"]),
                        seed_path.read_bytes(),
                    )
                    if case == "malformed-first":
                        self.assertIsNone(failure["receipt"])
                        self.assertEqual(failure["observation_preimage"], [])
                    else:
                        receipt = failure["receipt"]
                        self.assertIsNotNone(receipt, failure)
                        observations = failure["observation_preimage"]
                        self.assertEqual(
                            sha256_hex(_canonical(observations)),
                            receipt["observations_sha256"],
                        )
                        self.assertEqual(len(observations), 1)
                        if case in {"503", "transport"}:
                            self.assertEqual(receipt["status"], "BLOCKED_UNKNOWN")
                        else:
                            self.assertEqual(receipt["status"], "BLOCKED_NOT_VERIFIABLE")
                    calls_path = runner / "seed-get-calls.json"
                    calls = json.loads(calls_path.read_bytes()) if calls_path.exists() else []
                    if case == "malformed-first":
                        self.assertEqual(calls, [])
                    else:
                        self.assertEqual(len(calls), 1)
                        self.assertIn("/issues/comments/", calls[0])
                    self.assertNotIn("LIFECYCLE_APP_PRIVATE_KEY", env)

    def test_seed_receipt_failures_keep_absence_and_unknown_distinct(self) -> None:
        request = _request()
        payloads = _seed_provider_payloads(request)
        first_comment = next(path for path in payloads if "/issues/comments/" in path)
        transport = FakeSeedTransport(payloads)
        transport.overrides[first_comment] = ApiResponse(404, b'{"message":"Not Found"}', {})
        with self.assertRaises(RehearsalError) as absent:
            validate_seed_receipt_bundle(request, transport)
        self.assertEqual(absent.exception.failure_receipt["status"], "BLOCKED_NOT_VERIFIABLE")
        self.assertIsNone(absent.exception.failure_receipt["next_case"])

        transport = FakeSeedTransport(payloads)
        transport.overrides[first_comment] = ApiResponse(503, b'{"message":"unavailable"}', {})
        with self.assertRaises(RehearsalError) as unknown:
            validate_seed_receipt_bundle(request, transport)
        self.assertEqual(unknown.exception.failure_receipt["status"], "BLOCKED_UNKNOWN")
        self.assertEqual(len(unknown.exception.observation_preimage), 1)
        self.assertEqual(unknown.exception.observation_preimage[0]["status"], 503)
        self.assertEqual(
            base64.b64decode(unknown.exception.observation_preimage[0]["body_b64"]),
            b'{"message":"unavailable"}',
        )

        wrong_issue = dict(payloads)
        wrong_issue[first_comment] = {**wrong_issue[first_comment], "issue_url": "https://api.github.com/repos/other/repo/issues/125"}
        with self.assertRaisesRegex(RehearsalError, "SEED_COMMENT_BINDING_INVALID"):
            validate_seed_receipt_bundle(request, FakeSeedTransport(wrong_issue))

    def test_schema3_rejects_schema2_and_enforces_full_array_and_numeric_keywords(self) -> None:
        request = _request()
        with self.assertRaisesRegex(RehearsalError, "ROUTE_REQUEST_SCHEMA_INVALID"):
            validate_route_request_bytes(_canonical({"schema_version": 2}))
        altered = json.loads(request.request_bytes)
        altered["seed_receipt_bundle"]["records"].append(altered["seed_receipt_bundle"]["records"][0])
        altered["plan"]["seed_receipts_sha256"] = sha256_hex(_canonical(altered["seed_receipt_bundle"]))
        altered["plan_sha256"] = sha256_hex(_canonical(altered["plan"]))
        altered["owner_attestation"]["plan_sha256"] = altered["plan_sha256"]
        altered["owner_attestation_sha256"] = sha256_hex(_canonical(altered["owner_attestation"]))
        with self.assertRaises(RehearsalError):
            validate_route_request_bytes(_canonical(altered))

    def test_p5_observes_durable_4a_terminal_result_with_optional_off_ref_event(self) -> None:
        request = _request()
        approval = _approval(request)
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
            token_permissions={"contents": "write", "metadata": "read"},
            token_expires_at=NOW + timedelta(hours=1), clock=lambda: NOW,
        )
        principal_ref = SEED_REFS["principal"]
        prior_result = {
            "status": "PASS_ACCEPTED", "next_case": "4b-nff", "unresolved": False,
            "evidence_sha256": "8" * 64,
        }
        prior_evidence = {
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "plan_sha256": request.plan_sha256,
            "approval_receipt_sha256": approval.receipt_sha256,
            "ref": principal_ref,
            "expected_old_sha": SEED_TARGET_SHA,
            "intended_sha": "a6df7c78e27ee81771cc06b0783a508edcaed320",
            "api_mutation_count": 0,
            "git_push_count": 1,
            "target_rule_id": None,
            "failure_reason": None,
            "pre_state": {"object_type": None, "sha": None, "absence_known": True},
            "cas_result": {
                "write_outcome": "provider_accepted_and_read_back",
                "request_id": request.request_id,
                "expected_old_sha": SEED_TARGET_SHA,
                "intended_new_sha": "a6df7c78e27ee81771cc06b0783a508edcaed320",
                "intended_new_tree": SEED_TARGET_TREE,
                "observed_sha": "a6df7c78e27ee81771cc06b0783a508edcaed320",
                "postwrite_rules_verified": True,
            },
        }
        run_row = {
            "event": "push",
            "head_sha": "a6df7c78e27ee81771cc06b0783a508edcaed320",
            "head_branch": principal_ref.removeprefix("refs/heads/"),
            "actor": {"id": 5371554, "login": "cyaxiverse-lifecycle[bot]"},
            "repository": {"id": 527718962},
        }
        proof = RehearsalFreezeProof(
            request.owner_attestation["epoch_id"], request.plan["intent_id"],
            request.request_id, request.plan_sha256, principal_ref, 26316866,
            request.owner_attestation["full_definition_sha256"],
            (26314086, 26314227, 26316866),
            request.owner_attestation["last_operation_terminal_receipt_sha256"],
            request.owner_attestation["full_definition_sha256"], 0,
            "a6df7c78e27ee81771cc06b0783a508edcaed320", "2026-11-02T12:00:00Z",
            adapter._nonce,
        )

        for runs, expected_event in (([run_row], True), ([], False)):
            response = ApiResponse(
                200, _canonical({"total_count": len(runs), "workflow_runs": runs}),
                {"x-github-request-id": "SYNTHETIC-RUN-LIST"},
            )
            with (
                adapter.case_lock(principal_ref),
                mock.patch.object(adapter, "read_case_result_evidence", return_value=(prior_result, prior_evidence)),
                mock.patch.object(adapter, "_read_rehearsal_target_ref_with_evidence", return_value=(
                    ("commit", "a6df7c78e27ee81771cc06b0783a508edcaed320"),
                    {"status": 200, "body_sha256": "9" * 64},
                )),
                mock.patch.object(adapter, "_read_rehearsal_commit", return_value={
                    "sha": "a6df7c78e27ee81771cc06b0783a508edcaed320",
                    "tree": {"sha": SEED_TARGET_TREE}, "parents": [SEED_TARGET_SHA],
                }),
                mock.patch.object(adapter, "_api_request", return_value=response) as get,
            ):
                observation = adapter._execute_fixed_case(
                    "P5-event-observe", request, approval, proof
                )
            self.assertEqual(observation.status, "PASS_ACCEPTED")
            self.assertEqual(observation.p5_subresult["observed_off_ref_event"], expected_event)
            self.assertFalse(observation.p5_subresult["matching_app_trigger_guard_exercised"])
            self.assertEqual(observation.p5_subresult["second_driver_dispatch_count"], 0)
            self.assertEqual(observation.p5_subresult["api_mutation_count"], 0)
            self.assertEqual(observation.p5_subresult["git_push_count"], 0)
            self.assertEqual(get.call_args.args[1], f"/repos/{REPOSITORY}/actions/runs?event=push&head_sha=a6df7c78e27ee81771cc06b0783a508edcaed320&per_page=100")
            result = _result(request, "P5-event-observe", observation.status, observation.evidence,
                             p5_subresult=observation.p5_subresult)
            validate_driver_result(result)

        with (
            adapter.case_lock(principal_ref),
            mock.patch.object(adapter, "read_case_result_evidence", return_value=None),
            mock.patch.object(adapter, "rehearsal_effect_counters", return_value={
                "driver_dispatch_count": 0, "api_mutation_count": 0, "git_push_count": 0,
            }),
        ):
            blocked = adapter._execute_fixed_case("P5-event-observe", request, approval, proof)
        self.assertEqual(blocked.status, "BLOCKED_NOT_VERIFIABLE")
        self.assertEqual(blocked.p5_subresult["failure_reason"], "PRIOR_4A_TERMINAL_RECEIPT_ABSENT")
        self.assertFalse(blocked.p5_subresult["matching_app_trigger_guard_exercised"])
        self.assertEqual(blocked.p5_subresult["second_driver_dispatch_count"], 0)
        blocked_result = _result(
            request, "P5-event-observe", blocked.status, blocked.evidence,
            p5_subresult=blocked.p5_subresult,
        )
        validate_driver_result(blocked_result)

    def test_approval_job_accepts_known_runner_bookends_and_rejects_other_steps(self) -> None:
        job = _provider_payloads(_request())[
            f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/1/jobs?per_page=100"
        ]
        row = job["jobs"][0]
        row["steps"].insert(0, {
            "name": "Set up job", "status": "completed", "conclusion": "success",
        })
        row["steps"].append({
            "name": "Complete job", "status": "queued", "conclusion": None,
        })
        self.assertEqual(_validate_jobs_response(job, run_id=RUN_ID, head_sha=HEAD_SHA), 55779911)

        bad = json.loads(json.dumps(job))
        bad["jobs"][0]["steps"].insert(-1, {
            "name": "unreviewed executable", "status": "queued", "conclusion": None,
        })
        with self.assertRaisesRegex(RehearsalError, "APPROVAL_JOB_STEPS_INVALID"):
            _validate_jobs_response(bad, run_id=RUN_ID, head_sha=HEAD_SHA)

        bad_setup = json.loads(json.dumps(job))
        bad_setup["jobs"][0]["steps"][0]["conclusion"] = "failure"
        with self.assertRaisesRegex(RehearsalError, "APPROVAL_RUNNER_SETUP_NOT_COMPLETE"):
            _validate_jobs_response(bad_setup, run_id=RUN_ID, head_sha=HEAD_SHA)

    def test_raw_approval_envelope_round_trip_revalidates_response_provenance(self) -> None:
        request = _request()
        approval = _approval(request)
        envelope = serialize_approval_envelope(approval)
        restored = rehydrate_approval_envelope(
            envelope, request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA,
        )
        self.assertEqual(len(restored.raw_responses), 8)
        self.assertEqual(restored.receipt_sha256, approval.receipt_sha256)
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
        )
        self.assertEqual(
            adapter._approval_evidence_payload(request, restored)["receipt_sha256"],
            approval.receipt_sha256,
        )

        original = json.loads(envelope)
        for label, mutate in (
            ("path", lambda value: value["responses"][0].__setitem__("path", "/repos/other/repo")),
            ("receipt", lambda value: value.__setitem__("receipt_sha256", "0" * 64)),
            ("body", lambda value: value["responses"][0].__setitem__("body_b64", "e30=")),
            ("credential", lambda value: value["responses"][0]["headers"].__setitem__("authorization", "Bearer x")),
        ):
            bad = json.loads(json.dumps(original))
            mutate(bad)
            with self.subTest(label=label), self.assertRaises(RehearsalError):
                rehydrate_approval_envelope(
                    _canonical(bad), request, run_id=RUN_ID,
                    run_attempt=1, head_sha=HEAD_SHA,
                )

    def test_wrapped_fenced_request_contents_decode_to_exact_canonical_request(self) -> None:
        request = _request()
        encoded = base64.b64encode(request.request_bytes).decode("ascii")
        wrapped = "\n".join(encoded[index:index + 88] for index in range(0, len(encoded), 88)) + "\n"
        payload = {
            "path": request.request_path,
            "type": "file",
            "encoding": "base64",
            "content": wrapped,
            "size": len(request.request_bytes),
            "sha": hashlib.sha1(
                b"blob " + str(len(request.request_bytes)).encode("ascii")
                + b"\0" + request.request_bytes
            ).hexdigest(),
        }
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(
                200, json.dumps(payload, separators=(",", ":")).encode("ascii"),
                {"content-type": "application/json"},
            ),
        )
        self.assertEqual(adapter._read_fenced_route_request(request), request.request_bytes)

    def test_case_result_journals_and_retrieves_the_canonical_evidence_preimage(self) -> None:
        request = _request()
        case_id = CASE_ORDER[0]
        operation_ref = OPERATION_TABLE[0]["ref"]
        evidence = {
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "plan_sha256": request.plan_sha256,
            "ref": operation_ref,
            "provider_response": {
                "status": 422,
                "headers": {"x-github-request-id": "fixture"},
                "body_b64": base64.b64encode(b'{"errors":[]}').decode("ascii"),
                "body_sha256": sha256_hex(b'{"errors":[]}'),
                "body_bytes": len(b'{"errors":[]}'),
            },
            "pre_state": {"object_type": None, "sha": None, "absence_known": True},
            "post_state": {"object_type": None, "sha": None, "absence_known": True},
            "freeze_proof": {"snapshot_sha256": "4" * 64, "effective_rule_ids": [26314085]},
        }
        result = _result(request, case_id, "BLOCKED_UNKNOWN", evidence)
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
        )
        adapter._claimed_rehearsal_cases[(request.request_id, request.plan_sha256, case_id)] = "a" * 40
        written: list[dict[str, object]] = []
        with mock.patch.object(
            adapter, "_create_rehearsal_record_ref",
            side_effect=lambda **kwargs: written.append(dict(kwargs["record"])) or "b" * 40,
        ):
            with adapter.case_lock(operation_ref):
                adapter.append_case_result(result, evidence)
        record = written[0]
        record_bytes = _canonical(record)
        path = f"case-results/{request.request_id}/{request.plan_sha256}/{case_id}.json"
        encoded = base64.b64encode(record_bytes).decode("ascii")
        wrapped = "\n".join(encoded[index:index + 88] for index in range(0, len(encoded), 88)) + "\n"
        content_response = ApiResponse(200, json.dumps({
            "path": path, "encoding": "base64", "content": wrapped,
        }, separators=(",", ":")).encode("ascii"), {"content-type": "application/json"})
        with (
            mock.patch.object(adapter, "_read_rehearsal_target_ref", return_value=("commit", "c" * 40)),
            mock.patch.object(adapter, "_read_rehearsal_commit", return_value={"parents": [request.expected_vmm_sha]}),
            mock.patch.object(adapter, "_api_request", return_value=content_response),
        ):
            observed_result, observed_evidence = adapter.read_case_result_evidence(
                request.request_id, request.plan_sha256, case_id
            )
        self.assertEqual(observed_result, result)
        self.assertEqual(observed_evidence, evidence)
        self.assertEqual(sha256_hex(_canonical(observed_evidence)), observed_result["evidence_sha256"])

        bad_record = dict(record)
        bad_record["evidence"] = {**evidence, "reason": "tampered"}
        bad_encoded = base64.b64encode(_canonical(bad_record)).decode("ascii")
        bad_response = ApiResponse(200, json.dumps({
            "path": path, "encoding": "base64", "content": bad_encoded,
        }, separators=(",", ":")).encode("ascii"), {"content-type": "application/json"})
        with (
            mock.patch.object(adapter, "_read_rehearsal_target_ref", return_value=("commit", "c" * 40)),
            mock.patch.object(adapter, "_read_rehearsal_commit", return_value={"parents": [request.expected_vmm_sha]}),
            mock.patch.object(adapter, "_api_request", return_value=bad_response),
        ):
            with self.assertRaisesRegex(GitHubProtectionError, "CASE_RESULT_EVIDENCE_DIGEST_MISMATCH"):
                adapter.read_case_result_evidence(request.request_id, request.plan_sha256, case_id)

    def test_approval_evidence_payload_retains_only_receipt_bound_raw_responses(self) -> None:
        request = _request()
        approval = _approval(request)
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
        )
        payload = adapter._approval_evidence_payload(request, approval)
        self.assertEqual(payload["receipt_b64"], base64.b64encode(approval.receipt_bytes).decode("ascii"))
        self.assertEqual(len(payload["responses"]), 8)
        self.assertEqual(
            base64.b64decode(payload["responses"][0]["body_b64"], validate=True),
            approval.raw_responses[0]["body"],
        )
        self.assertEqual(payload["responses"][0]["body_sha256"], approval.raw_responses[0]["body_sha256"])

        changed_body = dict(approval.raw_responses[0])
        changed_body["body"] = changed_body["body"] + b" "
        changed_rows = (changed_body, *approval.raw_responses[1:])
        with self.assertRaisesRegex(GitHubProtectionError, "APPROVAL_RAW_EVIDENCE_INVALID"):
            adapter._approval_evidence_payload(request, replace(approval, raw_responses=changed_rows))

        changed_path = dict(approval.raw_responses[0])
        changed_path["path"] = "/repos/attacker/repository"
        changed_rows = (changed_path, *approval.raw_responses[1:])
        with self.assertRaisesRegex(GitHubProtectionError, "APPROVAL_RAW_EVIDENCE_INVALID"):
            adapter._approval_evidence_payload(request, replace(approval, raw_responses=changed_rows))

        changed_headers = dict(approval.raw_responses[0])
        changed_headers["headers"] = {**changed_headers["headers"], "authorization": "Bearer secret"}
        changed_rows = (changed_headers, *approval.raw_responses[1:])
        with self.assertRaisesRegex(GitHubProtectionError, "APPROVAL_RAW_EVIDENCE_CONTAINS_CREDENTIAL"):
            adapter._approval_evidence_payload(request, replace(approval, raw_responses=changed_rows))

    def test_first_case_claim_requires_and_carries_approval_evidence_once(self) -> None:
        request = _request()
        approval = _approval(request)
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
        )
        evidence = adapter._approval_evidence_payload(request, approval)
        records: list[dict[str, object]] = []

        def proof(case_index: int) -> RehearsalFreezeProof:
            case_id = CASE_ORDER[case_index]
            return RehearsalFreezeProof(
                request.owner_attestation["epoch_id"], request.plan["intent_id"],
                request.request_id, request.plan_sha256, OPERATION_TABLE[case_index]["ref"],
                26316866, request.owner_attestation["full_definition_sha256"],
                adapter._rehearsal_case_effective_ids(case_id),
                request.owner_attestation["last_operation_terminal_receipt_sha256"],
                request.owner_attestation["full_definition_sha256"], 0, None,
                "2026-11-02T12:00:00Z", adapter._nonce,
            )

        with adapter.case_lock(OPERATION_TABLE[0]["ref"]):
            with self.assertRaisesRegex(GitHubProtectionError, "APPROVAL_EVIDENCE_REQUIRED"):
                adapter.claim_case(request, CASE_ORDER[0], proof(0))

        adapter._initial_approval_evidence[request.request_id] = evidence
        with mock.patch.object(
            adapter, "_create_rehearsal_record_ref",
            side_effect=lambda **kwargs: records.append(dict(kwargs["record"])) or "a" * 40,
        ):
            with adapter.case_lock(OPERATION_TABLE[0]["ref"]):
                adapter.claim_case(request, CASE_ORDER[0], proof(0))
            with adapter.case_lock(OPERATION_TABLE[1]["ref"]):
                adapter.claim_case(request, CASE_ORDER[1], proof(1))

        self.assertEqual(records[0]["approval_evidence"], evidence)
        self.assertNotIn("approval_evidence", records[1])

    def test_route_trigger_binds_single_added_request_and_exact_python_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args: str) -> str:
                result = subprocess.run(
                    ["git", "-C", str(root), *args], capture_output=True,
                    check=True, env={
                        "PATH": os.environ["PATH"], "HOME": temp,
                        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                    },
                )
                return result.stdout.decode("utf-8").strip()
            git("init", "-q")
            git("config", "user.name", "Offline Owner")
            git("config", "user.email", "owner@example.invalid")
            workflow = root / ".github/workflows/cyax-lifecycle-dispatch.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text("name: fixed\n", encoding="ascii")
            package = root / "scripts/version_lifecycle"
            package.mkdir(parents=True)
            for index in range(16):
                (package / f"module_{index:02d}.py").write_text(
                    f"VALUE = {index}\n", encoding="ascii"
                )
            git("add", ".")
            git("commit", "-q", "-m", "parent")
            parent = git("rev-parse", "HEAD")
            tree = git("rev-parse", "HEAD^{tree}")
            workflow_blob = git("rev-parse", "HEAD:.github/workflows/cyax-lifecycle-dispatch.yml")
            paths = git("ls-tree", "-r", "--name-only", "HEAD", "--", "scripts/version_lifecycle").splitlines()
            closure = {
                path: git("rev-parse", f"HEAD:{path}")
                for path in paths if path.endswith(".py")
            }
            raw = _request_bytes(
                implementation_sha=parent,
                candidate_tree=tree,
                workflow_blob_sha=workflow_blob,
                called_code_blobs=closure,
            )
            request_path = root / "lifecycle-requests/route-0125-1.json"
            request_path.parent.mkdir(parents=True)
            request_path.write_bytes(raw)
            git("add", "lifecycle-requests/route-0125-1.json")
            git("commit", "-q", "-m", "trigger")
            commit = git("rev-parse", "HEAD")
            event = {
                "ref": REF, "before": "0" * 40, "after": commit,
                "created": True, "deleted": False, "forced": False,
                "repository": {"full_name": REPOSITORY},
                "sender": {"type": "User", "login": "owner"},
            }
            request = validate_rehearsal_route_trigger(
                str(root), repository=REPOSITORY, ref=REF,
                commit_sha=commit, actor="owner", event=event,
            )
            self.assertEqual(request.commit_sha, commit)
            self.assertEqual(request.parent_sha, parent)
            self.assertEqual(request.called_code_blobs, closure)

    def test_valid_approval_uses_exact_order_twice_and_unknown_time_labels(self) -> None:
        request = _request()
        transport = FakeApprovalTransport(_provider_payloads(request))
        approval = verify_run_approval(
            request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, ref=REF,
            transport=transport, monotonic=lambda: 2.0, utc_now=lambda: NOW,
        )
        run_prefix = f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}"
        expected = [
            f"/repos/{REPOSITORY}", f"{run_prefix}/attempts/1",
            f"{run_prefix}/approvals", f"{run_prefix}/attempts/1/jobs?per_page=100",
        ] * 2
        self.assertEqual(transport.calls, expected)
        self.assertEqual(approval.receipt["approval_event_time"], None)
        self.assertEqual(approval.receipt["approval_time_evidence"], "UNKNOWN_NOT_DOCUMENTED")
        self.assertEqual(approval.receipt["admin_bypass_evidence"], "OWNER_PROCEDURAL_ATTESTATION_NOT_PROVIDER_VERIFIED")
        self.assertEqual(len(approval.raw_responses), 8)
        self.assertEqual(approval.receipt["job_id"], 55779911)

    def test_approval_context_rejects_bool_attempt_without_provider_reads(self) -> None:
        request = _request()
        transport = FakeApprovalTransport(_provider_payloads(request))
        with self.assertRaisesRegex(RehearsalError, "APPROVAL_CONTEXT_INVALID"):
            verify_run_approval(
                request, run_id=RUN_ID, run_attempt=True, head_sha=HEAD_SHA,
                ref=REF, transport=transport,
            )
        self.assertEqual(transport.calls, [])

    def test_anonymous_http_transport_is_fixed_origin_tls_and_has_no_auth_headers(self) -> None:
        body = b'{"id":527718962,"full_name":"Julia-meets-String-Theory/CYAxiverse.jl","private":false}'

        class FakeResponse:
            def __init__(self, raw: bytes) -> None:
                self.status = 200
                self.raw = raw
            def read(self, _size: int) -> bytes: return self.raw
            def getheaders(self): return [("Content-Type", "application/json")]

        class FakeConnection:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs
                self.request_args = None
                self.closed = False
            def request(self, *args: object, **kwargs: object) -> None:
                self.request_args = (args, kwargs)
            def getresponse(self) -> FakeResponse: return FakeResponse(body)
            def close(self) -> None: self.closed = True

        connections: list[FakeConnection] = []
        def connection_factory(*args: object, **kwargs: object) -> FakeConnection:
            connection = FakeConnection(*args, **kwargs)
            connections.append(connection)
            return connection

        path = f"/repos/{REPOSITORY}"
        with mock.patch.dict(os.environ, {
            "GH_TOKEN": "offline-sentinel",
            "GITHUB_TOKEN": "offline-sentinel",
            "HTTPS_PROXY": "https://user:secret@proxy.invalid:443",
        }), mock.patch(
            "version_lifecycle.rehearsal.http.client.HTTPSConnection",
            side_effect=connection_factory,
        ):
            response = AnonymousGitHubTransport().get(path)
        self.assertEqual(response.status, 200)
        connection = connections[0]
        self.assertEqual(connection.args[:2], ("api.github.com", 443))
        self.assertTrue(connection.kwargs["context"].check_hostname)
        self.assertEqual(connection.kwargs["context"].verify_mode, 2)
        args, kwargs = connection.request_args
        self.assertEqual(args[:2], ("GET", path))
        headers = kwargs["headers"]
        self.assertEqual(set(headers), {"Accept", "X-GitHub-Api-Version", "User-Agent"})
        self.assertFalse({key.lower() for key in headers} & {
            "authorization", "proxy-authorization", "cookie", "x-github-token",
        })
        self.assertTrue(connection.closed)
        with mock.patch(
            "version_lifecycle.rehearsal.http.client.HTTPSConnection",
            side_effect=AssertionError("not reached"),
            ), self.assertRaisesRegex(RehearsalError, "ANONYMOUS_PATH_NOT_ALLOWLISTED"):
            AnonymousGitHubTransport().get("/repos/attacker/repo")

    def test_approval_read_window_is_enforced_during_the_eight_gets(self) -> None:
        request = _request()
        transport = FakeApprovalTransport(_provider_payloads(request))
        ticks = iter((0.0, 31.0))
        with self.assertRaisesRegex(RehearsalError, "APPROVAL_READ_WINDOW_EXCEEDED"):
            verify_run_approval(
                request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA,
                ref=REF, transport=transport, monotonic=lambda: next(ticks),
                utc_now=lambda: NOW,
            )
        self.assertEqual(len(transport.calls), 1)

    def test_provider_mutations_fail_closed(self) -> None:
        request = _request()
        base = _provider_payloads(request)
        cases: list[tuple[str, object, int]] = []
        private = json.loads(json.dumps(base)); private[f"/repos/{REPOSITORY}"]["private"] = True
        cases.append(("private", private, 200))
        wrong_attempt = json.loads(json.dumps(base)); wrong_attempt[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/1"]["run_attempt"] = True
        cases.append(("bool attempt", wrong_attempt, 200))
        duplicate = json.loads(json.dumps(base)); duplicate[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/approvals"].append(duplicate[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/approvals"][0])
        cases.append(("duplicate approval", duplicate, 200))
        stale = json.loads(json.dumps(base)); stale[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/approvals"][0]["state"] = "rejected"
        cases.append(("stale approval", stale, 200))
        wrong_owner = json.loads(json.dumps(base)); wrong_owner[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/approvals"][0]["user"]["id"] = 9
        cases.append(("wrong owner", wrong_owner, 200))
        wrong_env = json.loads(json.dumps(base)); wrong_env[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/approvals"][0]["environments"][0]["id"] = 9
        cases.append(("wrong environment", wrong_env, 200))
        wrong_marker = json.loads(json.dumps(base)); wrong_marker[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/approvals"][0]["comment"] = _marker(request, no_admin_bypass_used=False)
        cases.append(("marker mismatch", wrong_marker, 200))
        too_many_jobs = json.loads(json.dumps(base)); too_many_jobs[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/1/jobs?per_page=100"]["total_count"] = 101
        cases.append(("pagination count", too_many_jobs, 200))
        secret_started = json.loads(json.dumps(base)); secret_started[f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/1/jobs?per_page=100"]["jobs"][0]["steps"][2]["status"] = "in_progress"
        cases.append(("secret step started", secret_started, 200))
        for label, payloads, status in cases:
            with self.subTest(label=label):
                transport = FakeApprovalTransport(payloads)
                path = f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/1"
                if label == "private":
                    path = f"/repos/{REPOSITORY}"
                elif label in {"duplicate approval", "stale approval", "wrong owner", "wrong environment", "marker mismatch"}:
                    path = f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/approvals"
                elif label in {"pagination count", "secret step started"}:
                    path = f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}/attempts/1/jobs?per_page=100"
                if label == "marker mismatch":
                    transport.payloads[path][0]["comment"] = _marker(request, plan_sha256="0" * 64)
                with self.assertRaises(RehearsalError):
                    verify_run_approval(
                        request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA,
                        ref=REF, transport=transport, monotonic=lambda: 1.0,
                        utc_now=lambda: NOW,
                    )

    def test_approval_status_link_size_changed_and_json_failures(self) -> None:
        request = _request()
        root = f"/repos/{REPOSITORY}"
        for status in (302, 401, 403, 429, 500):
            transport = FakeApprovalTransport(_provider_payloads(request))
            transport.overrides[(root, 0)] = ApiResponse(status, b"{}", {"content-type": "application/json"})
            with self.subTest(status=status), self.assertRaises(RehearsalError):
                verify_run_approval(request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, ref=REF, transport=transport)

        linked = FakeApprovalTransport(_provider_payloads(request))
        linked.overrides[(root, 0)] = ApiResponse(200, b'{"id":527718962,"full_name":"Julia-meets-String-Theory/CYAxiverse.jl","private":false}', {"content-type": "application/json", "link": '<next>; rel="next"'})
        with self.assertRaisesRegex(RehearsalError, "APPROVAL_PAGINATION_NOT_ALLOWED"):
            verify_run_approval(request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, ref=REF, transport=linked)

        oversized = FakeApprovalTransport(_provider_payloads(request))
        oversized.overrides[(root, 0)] = ApiResponse(200, b" " * (1024 * 1024 + 1), {"content-type": "application/json"})
        with self.assertRaisesRegex(RehearsalError, "APPROVAL_RESPONSE_OVERSIZE"):
            verify_run_approval(request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, ref=REF, transport=oversized)

        changed = FakeApprovalTransport(_provider_payloads(request))
        changed.overrides[(root, 1)] = ApiResponse(200, b'{"id":527718963,"full_name":"Julia-meets-String-Theory/CYAxiverse.jl","private":false}', dict(changed.default_headers))
        with self.assertRaisesRegex(RehearsalError, "APPROVAL_READS_CHANGED"):
            verify_run_approval(request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, ref=REF, transport=changed)

        for body in (b'{"id":527718962,"id":527718962}', b'{"id":527718962,"x":1.5}'):
            malformed = FakeApprovalTransport(_provider_payloads(request))
            malformed.overrides[(root, 0)] = ApiResponse(200, body, dict(malformed.default_headers))
            with self.assertRaises(RehearsalError):
                verify_run_approval(request, run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, ref=REF, transport=malformed)

    def test_receipt_is_closed_and_revalidated(self) -> None:
        request = _request()
        approval = _approval(request)
        restored = validate_approval_receipt(
            approval.receipt_bytes, request, run_id=RUN_ID,
            run_attempt=1, head_sha=HEAD_SHA,
        )
        self.assertEqual(restored.receipt_sha256, approval.receipt_sha256)
        altered = dict(approval.receipt)
        altered["admin_bypass_evidence"] = "PROVIDER_VERIFIED"
        with self.assertRaises(RehearsalError):
            validate_approval_receipt(
                _canonical(altered), request, run_id=RUN_ID,
                run_attempt=1, head_sha=HEAD_SHA,
            )


class FakeDriverAdapter:
    def __init__(self, *, stop_at: str | None = None, preflight: str = "PASS_ACCEPTED", change_epoch_at: str | None = None, post_sha_at: str | None = None) -> None:
        self.stop_at = stop_at
        self.preflight_status = preflight
        self.change_epoch_at = change_epoch_at
        self.post_sha_at = post_sha_at
        self.executed: list[str] = []
        self.locks: list[str] = []
        self.proof_calls: dict[tuple[str, str], int] = {}
        self.postproof_markers: dict[str, dict[str, object]] = {}
        self.ref_shas = {row["ref"]: row["expected_old"] for row in OPERATION_TABLE if not row["ref"].startswith("DETERMINISTIC ")}

    def preflight_route(self, request: ValidatedRouteRequest, approval: VerifiedRunApproval) -> CaseObservation:
        return CaseObservation(self.preflight_status, {"reason": self.preflight_status})

    @contextmanager
    def case_lock(self, ref: str):
        self.locks.append(ref)
        yield

    def verify_rehearsal_freeze_epoch(self, request: ValidatedRouteRequest, case_id: str, phase: str) -> object:
        key = (case_id, phase)
        count = self.proof_calls.get(key, 0)
        self.proof_calls[key] = count + 1
        generation = 1
        if self.change_epoch_at == case_id and phase == "post":
            generation = 2
        row = OPERATION_TABLE[CASE_ORDER.index(case_id)]
        ref = row["ref"]
        if ref.startswith("DETERMINISTIC "):
            ref = f"refs/heads/rehearsal/lifecycle/requests/{request.request_id}"
        sha = self.ref_shas.get(ref)
        if phase == "post" and case_id == self.post_sha_at:
            sha = "e" * 40
        proof = SimpleNamespace(
            generation=generation, ref=ref, ref_sha=sha,
            ref_object_type="commit" if sha is not None else None,
            to_dict=lambda: {
                "case_id": case_id, "phase": phase, "generation": generation,
                "ref": ref, "ref_sha": sha,
                "ref_object_type": "commit" if sha is not None else None,
            },
        )
        return proof

    def retain_unreconciled_case_postproof(self, request, case_id: str, evidence) -> None:
        self.postproof_markers[case_id] = dict(evidence)

    def execute_case(self, case_id: str, request: ValidatedRouteRequest, approval: VerifiedRunApproval, proof: object) -> CaseObservation:
        self.executed.append(case_id)
        if case_id == self.stop_at:
            return CaseObservation("BLOCKED_UNKNOWN", {"reason": "provider timeout"}, failure_reason="provider timeout")
        if case_id == "4c":
            return CaseObservation("PASS_OWNER_ROUTE_NO_PUSH", {"api_mutation_count": 0, "git_push_count": 0})
        if case_id == "P5-event-observe":
            return CaseObservation("PASS_ACCEPTED", {}, p5_subresult={
                "matching_app_trigger_guard_exercised": False,
                "observed_off_ref_event": False,
                "second_driver_dispatch_count": 0,
                "api_mutation_count": 0,
                "git_push_count": 0,
                "existing_fence_sha": None,
                "fence_state": "UNKNOWN",
                "request_plan_match": None,
                "failure_reason": None,
            })
        if case_id == "P5-duplicate-fence":
            return CaseObservation("PASS_DENIED_TARGET_CONTROL", {}, p5_subresult={
                "matching_app_trigger_guard_exercised": False,
                "observed_off_ref_event": None,
                "second_driver_dispatch_count": 0,
                "api_mutation_count": 0,
                "git_push_count": 0,
                "existing_fence_sha": "3" * 40,
                "fence_state": "VERIFIED_MATCH",
                "request_plan_match": True,
                "failure_reason": None,
            })
        if case_id == "4g":
            return CaseObservation("STOP_OWNER_ACTION_REQUIRED", {"reason": "Owner action"})
        if case_id in {"4d-branch-create", "4d-tag-create"}:
            return CaseObservation(
                "BLOCKED_NOT_VERIFIABLE",
                {"reason": "WORKFLOWS_CREATE_DISCRIMINATOR_UNQUALIFIED"},
                failure_reason="WORKFLOWS_CREATE_DISCRIMINATOR_UNQUALIFIED",
            )
        if case_id == "4a":
            self.ref_shas[OPERATION_TABLE[CASE_ORDER.index(case_id)]["ref"]] = OPERATION_TABLE[CASE_ORDER.index(case_id)]["target"]
            return CaseObservation("PASS_ACCEPTED", {"reason": "fixed provider outcome"})
        return CaseObservation("PASS_DENIED_TARGET_CONTROL", {"reason": "fixed provider outcome"})


class FakeJournal:
    def __init__(self, *, fail_claim: bool = False, fail_result: bool = False) -> None:
        self.fail_claim = fail_claim
        self.fail_result = fail_result
        self.claims: list[str] = []
        self.results: list[dict[str, object]] = []
        self.evidence: list[dict[str, object]] = []

    def claim_case(self, request: ValidatedRouteRequest, case_id: str, proof: object) -> str:
        self.claims.append(case_id)
        if self.fail_claim:
            raise GitHubProtectionError("CASE_CLAIM_WRITE_UNKNOWN")
        return "claim-sha"

    def append_case_result(
        self, result: dict[str, object], evidence: dict[str, object]
    ) -> str:
        if self.fail_result:
            raise GitHubProtectionError("CASE_RESULT_WRITE_UNKNOWN")
        self.results.append(dict(result))
        self.evidence.append(dict(evidence))
        return "result-sha"


class RehearsalDriverTests(unittest.TestCase):
    def test_fixed_driver_stops_at_unqualified_workflows_boundary(self) -> None:
        request = _request()
        approval = _approval(request)
        adapter = FakeDriverAdapter()
        journal = FakeJournal()
        result = run_rehearsal(request, approval, adapter, journal)
        stop_index = CASE_ORDER.index("4d-branch-create")
        self.assertEqual(adapter.executed, list(CASE_ORDER[:stop_index + 1]))
        self.assertEqual(journal.claims, list(CASE_ORDER[:stop_index + 1]))
        self.assertEqual(len(journal.results), stop_index + 1)
        for journal_result, evidence in zip(journal.results, journal.evidence):
            self.assertEqual(
                journal_result["evidence_sha256"], sha256_hex(_canonical(evidence))
            )
        self.assertEqual(result["case_id"], "4d-branch-create")
        self.assertEqual(result["status"], "BLOCKED_NOT_VERIFIABLE")
        self.assertEqual(adapter.locks[-1], "refs/heads/rehearsal/lifecycle/proof/cyax125-20261101")

    def test_missing_seed_preflight_stops_before_claim_or_execution(self) -> None:
        request = _request()
        adapter = FakeDriverAdapter(preflight="STOP_OWNER_ACTION_REQUIRED")
        journal = FakeJournal()
        result = run_rehearsal(request, _approval(request), adapter, journal)
        self.assertEqual(result["status"], "STOP_OWNER_ACTION_REQUIRED")
        self.assertEqual(adapter.executed, [])
        self.assertEqual(journal.claims, [])

    def test_unknown_claim_does_not_retry_or_write_a_result(self) -> None:
        request = _request()
        adapter = FakeDriverAdapter()
        journal = FakeJournal(fail_claim=True)
        result = run_rehearsal(request, _approval(request), adapter, journal)
        self.assertEqual(result["status"], "BLOCKED_UNKNOWN")
        self.assertEqual(len(adapter.executed), 0)
        self.assertEqual(journal.claims, [CASE_ORDER[0]])
        self.assertEqual(journal.results, [])

    def test_unknown_case_outcome_and_changed_epoch_stop_sequence(self) -> None:
        request = _request()
        adapter = FakeDriverAdapter(stop_at="P2-app-create")
        journal = FakeJournal()
        result = run_rehearsal(request, _approval(request), adapter, journal)
        self.assertEqual(result["status"], "BLOCKED_UNKNOWN")
        self.assertEqual(adapter.executed, ["P2-app-create"])

        adapter = FakeDriverAdapter(change_epoch_at="P2-app-create")
        journal = FakeJournal()
        result = run_rehearsal(request, _approval(request), adapter, journal)
        self.assertEqual(result["status"], "BLOCKED_UNKNOWN")
        self.assertEqual(adapter.executed, ["P2-app-create"])

    def test_final_postproof_sha_interleave_blocks_pass_and_retains_case_write_marker(self) -> None:
        request = _request()
        adapter = FakeDriverAdapter(post_sha_at="P2-app-create")
        journal = FakeJournal()
        result = run_rehearsal(request, _approval(request), adapter, journal)
        self.assertEqual(result["status"], "BLOCKED_UNKNOWN")
        self.assertIsNone(result["next_case"])
        self.assertEqual(adapter.executed, ["P2-app-create"])
        self.assertIn("P2-app-create", adapter.postproof_markers)
        self.assertEqual(
            adapter.postproof_markers["P2-app-create"]["post"]["ref_sha"], "e" * 40
        )

    def test_bad_p5_counts_cannot_pass_and_failed_p5_never_advances(self) -> None:
        request = _request()
        approval = _approval(request)
        bad = {
            "schema_version": 1,
            "request_id": request.request_id,
            "plan_sha256": request.plan_sha256,
            "case_id": "P5-event-observe",
            "status": "PASS_ACCEPTED",
            "evidence_sha256": "2" * 64,
            "next_case": "P5-duplicate-fence",
            "unresolved": False,
            "p5_subresult": {
                "matching_app_trigger_guard_exercised": False,
                "observed_off_ref_event": None,
                "second_driver_dispatch_count": True,
                "api_mutation_count": 0,
                "git_push_count": 0,
                "existing_fence_sha": None,
                "fence_state": "UNKNOWN",
                "request_plan_match": None,
                "failure_reason": None,
            },
        }
        with self.assertRaises(RehearsalError):
            validate_driver_result(bad)


class GitHubCaseAdapterTests(unittest.TestCase):
    def _case_fixture(self, *, status: int, body: bytes, positive_case: str = "4b-nff"):
        ref = "refs/heads/rehearsal/principal/cyax125-20261101-route"
        old = "a6df7c78e27ee81771cc06b0783a508edcaed320"
        target = "f12fbb8ec52e4c7510564f602a15f26173fe772e"
        state = {"sha": old}
        calls: list[tuple[str, str, object]] = []

        def transport(method: str, path: str, payload: object) -> ApiResponse:
            calls.append((method, path, payload))
            if method == "GET" and "/git/ref/" in path:
                if state["sha"] is None:
                    return ApiResponse(404, b"{}", {})
                return ApiResponse(200, json.dumps({"ref": ref, "object": {"type": "commit", "sha": state["sha"]}}).encode(), {"content-type": "application/json"})
            if method == "GET" and "/git/commits/" in path:
                sha = path.rsplit("/", 1)[-1]
                if sha == "f12fbb8ec52e4c7510564f602a15f26173fe772e":
                    parent = "d8f5141615b11e88bb9a5aa23c5b3bc7befe8fc3"
                elif sha in {"a6df7c78e27ee81771cc06b0783a508edcaed320", "32af89296955ad684517e7a57ce5d8a431f5a971", "749c211debb344d556db5a1f74e6759a217fe9da"}:
                    parent = "f12fbb8ec52e4c7510564f602a15f26173fe772e"
                else:
                    parent = old
                return ApiResponse(200, json.dumps({"sha": sha, "tree": {"sha": "e" * 40}, "parents": [{"sha": parent}]}).encode(), {"content-type": "application/json"})
            if method == "PATCH":
                if status in {200, 201}:
                    state["sha"] = payload["sha"]
                return ApiResponse(status, body, {"content-type": "application/json", "x-github-request-id": "fake-write"})
            return ApiResponse(500, b"{}", {"content-type": "application/json"})

        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=transport, token_permissions={"contents": "write", "metadata": "read"},
            token_expires_at=NOW + timedelta(hours=1), clock=lambda: NOW,
        )
        request = _request()
        approval = VerifiedRunApproval(
            receipt={}, receipt_bytes=b"{}", receipt_sha256="6" * 64,
            request_id=request.request_id, plan_sha256=request.plan_sha256,
            run_id=RUN_ID, run_attempt=1, head_sha=HEAD_SHA, job_id=1,
            marker_sha256="7" * 64, raw_responses=(),
            _verification_token=_APPROVAL_VERIFICATION_TOKEN,
        )
        proof = RehearsalFreezeProof(
            "owner-epoch-1", request.plan["intent_id"], request.request_id,
            request.plan_sha256, ref, 26316866,
            request.owner_attestation["full_definition_sha256"],
            (26314086, 26314227, 26316866), "5" * 64,
            request.owner_attestation["full_definition_sha256"], 0,
            old, "2026-11-02T12:00:00Z", adapter._nonce,
        )
        return adapter, request, approval, proof, state, calls, ref, target

    def test_unknown_case_postread_is_not_treated_as_known_absence(self) -> None:
        adapter, request, approval, proof, _state, _calls, ref, target = self._case_fixture(
            status=422, body=b'{"errors":[{"rule_id":26314086}]}',
        )
        reads = iter([
            (("commit", proof.ref_sha), {"status": 200, "body_sha256": "1" * 64}),
            TimeoutError("postread unavailable"),
        ])

        def read_ref(_ref: str):
            result = next(reads)
            if isinstance(result, Exception):
                raise result
            return result

        with (
            adapter.case_lock(ref),
            mock.patch.object(adapter, "_read_rehearsal_target_ref_with_evidence", side_effect=read_ref),
            mock.patch.object(adapter, "_api_request", return_value=ApiResponse(
                422, b'{"errors":[{"rule_id":26314086}]}',
                {"x-github-request-id": "postread-test"},
            )),
        ):
            observation = adapter._case_observation(
                "4b-nff", request, proof, method="PATCH", path="/fixture/ref",
                payload={"sha": target, "force": True}, expected_sha=target,
                expected_rule=26314086, positive=False,
                approval_receipt_sha256=approval.receipt_sha256,
            )
        self.assertEqual(observation.status, "BLOCKED_UNKNOWN")
        self.assertEqual(observation.evidence["post_state"]["absence_known"], False)
        self.assertIsNotNone(observation.evidence["post_state"]["readback_error"])
        self.assertEqual(len(adapter._unreconciled_case_writes), 1)

    def test_unqualified_workflows_create_denials_and_network_errors_stay_blocked_not_verifiable(self) -> None:
        operation = next(row for row in OPERATION_TABLE if row["case"] == "4d-branch-create")
        for status, response_body, transport_error, expected_state, expected_reason in (
            (403, b'{"message":"generic"}', None,
             "BLOCKED_UNQUALIFIED_PROVIDER_DISCRIMINATOR",
             "WORKFLOWS_DENIAL_DISCRIMINATOR_UNQUALIFIED"),
            (422, b'{"message":"generic"}', None,
             "BLOCKED_UNQUALIFIED_PROVIDER_DISCRIMINATOR",
             "WORKFLOWS_DENIAL_DISCRIMINATOR_UNQUALIFIED"),
            (401, b'{"message":"authentication required"}', None,
             "BLOCKED_UNQUALIFIED_PROVIDER_OUTCOME",
             "WORKFLOWS_PROVIDER_OUTCOME_UNQUALIFIED"),
            (503, b'{"message":"unavailable"}', None,
             "BLOCKED_UNQUALIFIED_PROVIDER_OUTCOME",
             "WORKFLOWS_PROVIDER_OUTCOME_UNQUALIFIED"),
            (None, b"", TimeoutError("offline"),
             "BLOCKED_UNQUALIFIED_PROVIDER_OUTCOME",
             "WORKFLOWS_PROVIDER_OUTCOME_UNQUALIFIED"),
        ):
            with self.subTest(status=status, network=transport_error is not None):
                adapter, request, approval, proof, _state, _calls, _ref, _target = self._case_fixture(
                    status=200, body=b"{}", positive_case="4d-branch-create",
                )
                capability_proof = replace(proof, ref=operation["ref"], ref_sha=None)
                response = ApiResponse(status or 503, response_body, {
                    "x-github-request-id": "SYNTHETIC-CAPABILITY",
                })
                with (
                    adapter.case_lock(operation["ref"]),
                    mock.patch.object(adapter, "_read_rehearsal_target_ref_with_evidence", return_value=(
                        None, {"status": 404, "body_sha256": "1" * 64},
                    )),
                    mock.patch.object(
                        adapter, "_api_request",
                        side_effect=transport_error if transport_error is not None else None,
                        return_value=response,
                    ),
                ):
                    observation = adapter._case_observation(
                        "4d-branch-create", request, capability_proof,
                        method="POST", path=f"/repos/{REPOSITORY}/git/refs",
                        payload={"ref": operation["ref"], "sha": operation["target"]},
                        expected_sha=operation["target"], expected_rule=None,
                        positive=True, approval_receipt_sha256=approval.receipt_sha256,
                    )
                self.assertEqual(observation.status, "BLOCKED_NOT_VERIFIABLE")
                self.assertEqual(observation.failure_reason, expected_reason)
                self.assertEqual(observation.evidence["capability_fallback_state"], expected_state)
                self.assertIsNone(observation.p5_subresult)
                result = _result(
                    request, "4d-branch-create", observation.status,
                    observation.evidence,
                )
                self.assertIsNone(result["next_case"])
                self.assertTrue(result["unresolved"])

    def test_duplicate_fence_exercises_installed_dispatch_boundary_and_measures_effects(self) -> None:
        request = _request()
        approval = _approval(request)
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="offline-token", api_base="https://fixture.invalid",
            transport=lambda *_args: ApiResponse(500, b"{}", {}),
            token_permissions={"contents": "write", "metadata": "read"},
            token_expires_at=NOW + timedelta(hours=1), clock=lambda: NOW,
        )
        fence_ref = f"refs/heads/rehearsal/lifecycle/requests/{request.request_id}"
        fence_sha = "a" * 40
        adapter._initial_request_fences[request.request_id] = fence_sha
        adapter._route_admitted_dispatches = 1
        proof = RehearsalFreezeProof(
            request.owner_attestation["epoch_id"], request.plan["intent_id"],
            request.request_id, request.plan_sha256, fence_ref, 26316866,
            request.owner_attestation["full_definition_sha256"],
            (26314086, 26314227, 26316866),
            request.owner_attestation["last_operation_terminal_receipt_sha256"],
            request.owner_attestation["full_definition_sha256"], 0, fence_sha,
            "2026-11-02T12:00:00Z", adapter._nonce,
        )
        evidence = {"status": 200, "body_sha256": "2" * 64}
        with (
            adapter.case_lock(fence_ref),
            mock.patch.object(
                adapter, "_read_rehearsal_target_ref_with_evidence",
                return_value=(("commit", fence_sha), evidence),
            ),
            mock.patch.object(
                adapter, "_read_fenced_route_request",
                return_value=(request.request_bytes, evidence),
            ),
            mock.patch.object(
                adapter, "_vmm_head_and_tree",
                return_value=(request.expected_vmm_sha, "b" * 40),
            ),
            mock.patch.object(adapter, "_read_request_file", return_value=False),
            mock.patch.object(
                adapter, "_read_rehearsal_fence",
                return_value=("commit", fence_sha),
            ),
        ):
            observation = adapter.reject_existing_same_request_fence(
                request, approval, proof
            )
        self.assertEqual(observation.status, "PASS_DENIED_TARGET_CONTROL", observation.evidence)
        duplicate_result = _result(
            request, "P5-duplicate-fence", observation.status,
            observation.evidence, p5_subresult=observation.p5_subresult,
        )
        validate_driver_result(duplicate_result)
        self.assertEqual(
            observation.evidence["original_fence"]["state"]["sha"], fence_sha
        )
        case_id = "P5-duplicate-fence"
        adapter._claimed_rehearsal_cases[(request.request_id, request.plan_sha256, case_id)] = "claim-sha"
        written_record = {}
        result_path = f"case-results/{request.request_id}/{request.plan_sha256}/{case_id}.json"
        with (
            adapter.case_lock(fence_ref),
            mock.patch.object(
                adapter, "_create_rehearsal_record_ref",
                side_effect=lambda **kwargs: written_record.update(kwargs["record"]) or "result-sha",
            ),
        ):
            adapter.append_case_result(duplicate_result, observation.evidence)
        returned_content = base64.b64encode(_canonical(written_record)).decode("ascii")
        returned_response = ApiResponse(200, _canonical({
            "path": result_path, "encoding": "base64", "content": returned_content,
        }), {})
        with (
            mock.patch.object(
                adapter, "_read_rehearsal_target_ref", return_value=("commit", "9" * 40)
            ),
            mock.patch.object(adapter, "_read_rehearsal_commit", return_value={
                "sha": "9" * 40, "tree": {"sha": "8" * 40},
                "parents": ["f12fbb8ec52e4c7510564f602a15f26173fe772e"],
            }),
            mock.patch.object(adapter, "_api_request", return_value=returned_response),
        ):
            retrieved = adapter.read_case_result_evidence(
                request.request_id, request.plan_sha256, case_id
            )
        self.assertEqual(retrieved, (duplicate_result, observation.evidence))
        self.assertEqual(observation.evidence["reason"], "REQUEST_ID_ALREADY_USED")
        self.assertEqual(
            observation.evidence["effect_deltas"],
            {"second_driver_dispatch_count": 0, "api_mutation_count": 0, "git_push_count": 0},
        )
        self.assertEqual(
            adapter.rehearsal_effect_counters(),
            {"driver_dispatch_count": 1, "api_mutation_count": 0, "git_push_count": 0},
        )

        def unexpected_acceptance(*_args: object, **_kwargs: object) -> dict[str, object]:
            adapter._route_admitted_dispatches += 1
            return {"fence": {}, "result": {"status": "PASS_ACCEPTED"}}

        with (
            adapter.case_lock(fence_ref),
            mock.patch.object(
                adapter, "_read_rehearsal_target_ref_with_evidence",
                return_value=(("commit", fence_sha), evidence),
            ),
            mock.patch.object(
                adapter, "_read_fenced_route_request",
                return_value=(request.request_bytes, evidence),
            ),
            mock.patch.object(adapter, "dispatch_rehearsal_route", side_effect=unexpected_acceptance),
        ):
            unexpected = adapter.reject_existing_same_request_fence(
                request, approval, proof
            )
        self.assertEqual(unexpected.status, "BLOCKED_NOT_VERIFIABLE")
        self.assertEqual(
            unexpected.evidence["effect_deltas"]["second_driver_dispatch_count"], 1
        )

    def test_only_fixed_nff_case_sets_force_true_and_requires_rule_attribution(self) -> None:
        adapter, request, approval, proof, _state, calls, ref, _target = self._case_fixture(
            status=422, body=b'{"errors":[{"rule_id":26314086}]}',
        )
        with adapter.case_lock(ref):
            result = adapter.execute_case("4b-nff", request, approval, proof)
        self.assertEqual(result.status, "PASS_DENIED_TARGET_CONTROL")
        patch = [entry for entry in calls if entry[0] == "PATCH"]
        self.assertEqual(len(patch), 1)
        self.assertEqual(patch[0][2], {"sha": "f12fbb8ec52e4c7510564f602a15f26173fe772e", "force": True})

        adapter, request, approval, proof, _state, _calls, ref, _target = self._case_fixture(
            status=422, body=b'{"message":"generic protected branch rejection"}',
        )
        with adapter.case_lock(ref):
            result = adapter.execute_case("4b-nff", request, approval, proof)
        self.assertEqual(result.status, "BLOCKED_NOT_VERIFIABLE")

    def test_ff_route_requires_local_git_cas_source_without_api_fallback(self) -> None:
        adapter, request, approval, proof, state, calls, ref, _target = self._case_fixture(
            status=200, body=b'{"ref":"ok"}', positive_case="4a",
        )
        proof = replace(proof, ref_sha="f12fbb8ec52e4c7510564f602a15f26173fe772e")
        state["sha"] = proof.ref_sha
        with adapter.case_lock(ref):
            result = adapter.execute_case("4a", request, approval, proof)
        self.assertEqual(result.status, "BLOCKED_NOT_VERIFIABLE")
        self.assertEqual(result.failure_reason, "LOCAL_GIT_SOURCE_UNAVAILABLE")
        self.assertFalse(any(entry[0] == "PATCH" for entry in calls))
        self.assertEqual(state["sha"], proof.ref_sha)

    def test_unknown_write_is_not_retried_on_the_same_adapter(self) -> None:
        adapter, request, approval, proof, _state, calls, ref, _target = self._case_fixture(
            status=422, body=b"{}",
        )
        original = adapter._transport
        def timeout(method: str, path: str, payload: object) -> ApiResponse:
            if method == "PATCH":
                calls.append((method, path, payload))
                raise TimeoutError("offline timeout")
            return original(method, path, payload)
        adapter._transport = timeout
        with adapter.case_lock(ref):
            first = adapter.execute_case("4b-nff", request, approval, proof)
        self.assertEqual(first.status, "BLOCKED_UNKNOWN")
        with adapter.case_lock(ref), self.assertRaisesRegex(
            GitHubProtectionError, "REHEARSAL_EPOCH_INVALIDATED"
        ):
            adapter.execute_case("4b-nff", request, approval, proof)
        self.assertEqual(len([call for call in calls if call[0] == "PATCH"]), 1)

    def test_expired_or_wrong_scope_token_is_rejected(self) -> None:
        adapter, request, approval, proof, _state, _calls, ref, _target = self._case_fixture(
            status=422, body=b'{}',
        )
        adapter._token_permissions = {"contents": "write"}
        with self.assertRaisesRegex(GitHubProtectionError, "APP_TOKEN_SCOPE_INVALID"):
            adapter.execute_case("4b-nff", request, approval, proof)
        adapter._token_permissions = {"contents": "write", "metadata": "read"}
        adapter._token_expires_at = NOW + timedelta(seconds=1)
        with self.assertRaisesRegex(GitHubProtectionError, "APP_TOKEN_EXPIRED"):
            adapter.execute_case("4b-nff", request, approval, proof)


if __name__ == "__main__":
    unittest.main()
