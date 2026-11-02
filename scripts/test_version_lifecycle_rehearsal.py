"""Offline contract, approval, and fixed-route tests for rehearsal runtime."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
    run_rehearsal,
    validate_approval_receipt,
    validate_driver_result,
    validate_route_request_bytes,
    validate_rehearsal_route_trigger,
    verify_run_approval,
)


RUN_ID = 9854321
HEAD_SHA = "d" * 40
REF = "refs/heads/rehearsal/trigger/route-0125-1"
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def _request_bytes(
    *,
    implementation_sha: str = "a" * 40,
    candidate_tree: str = "b" * 40,
    workflow_blob_sha: str = "c" * 40,
    called_code_blobs: dict[str, str] | None = None,
) -> bytes:
    code = called_code_blobs or {
        f"scripts/version_lifecycle/fake_{index:02d}.py": f"{index:040x}"
        for index in range(16)
    }
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
        "seed_receipts_sha256": "3" * 64,
        "operation_table_sha256": _CONTRACT_SHA256["operation_table"],
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
        "schema_version": 2,
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
        raw = request.request_bytes
        with self.assertRaises(RehearsalError):
            validate_route_request_bytes(raw + b"\n")
        with self.assertRaises(RehearsalError):
            validate_route_request_bytes(raw.replace(b'"schema_version":2', b'"schema_version":true'))
        with self.assertRaises(RehearsalError):
            validate_route_request_bytes(raw.replace(b'"schema_version":2', b'"schema_version":2,"schema_version":2'))

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
        ), self.assertRaisesRegex(RehearsalError, "APPROVAL_PATH_NOT_ALLOWLISTED"):
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
    def __init__(self, *, stop_at: str | None = None, preflight: str = "PASS_ACCEPTED", change_epoch_at: str | None = None) -> None:
        self.stop_at = stop_at
        self.preflight_status = preflight
        self.change_epoch_at = change_epoch_at
        self.executed: list[str] = []
        self.locks: list[str] = []
        self.proof_calls: dict[tuple[str, str], int] = {}

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
        return SimpleNamespace(generation=generation, to_dict=lambda: {"case_id": case_id, "phase": phase})

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
        positive = {"4a", "4d-branch-create", "4d-tag-create"}
        return CaseObservation(
            "PASS_ACCEPTED" if case_id in positive else "PASS_DENIED_TARGET_CONTROL",
            {"reason": "fixed provider outcome"},
        )


class FakeJournal:
    def __init__(self, *, fail_claim: bool = False, fail_result: bool = False) -> None:
        self.fail_claim = fail_claim
        self.fail_result = fail_result
        self.claims: list[str] = []
        self.results: list[dict[str, object]] = []

    def claim_case(self, request: ValidatedRouteRequest, case_id: str, proof: object) -> str:
        self.claims.append(case_id)
        if self.fail_claim:
            raise GitHubProtectionError("CASE_CLAIM_WRITE_UNKNOWN")
        return "claim-sha"

    def append_case_result(self, result: dict[str, object]) -> str:
        if self.fail_result:
            raise GitHubProtectionError("CASE_RESULT_WRITE_UNKNOWN")
        self.results.append(dict(result))
        return "result-sha"


class RehearsalDriverTests(unittest.TestCase):
    def test_fixed_driver_executes_full_order_then_stops_at_owner_action(self) -> None:
        request = _request()
        approval = _approval(request)
        adapter = FakeDriverAdapter()
        journal = FakeJournal()
        result = run_rehearsal(request, approval, adapter, journal)
        self.assertEqual(adapter.executed, list(CASE_ORDER))
        self.assertEqual(journal.claims, list(CASE_ORDER))
        self.assertEqual(len(journal.results), len(CASE_ORDER))
        self.assertEqual(result["case_id"], "4g")
        self.assertEqual(result["status"], "STOP_OWNER_ACTION_REQUIRED")
        self.assertEqual(adapter.locks[15], f"refs/heads/rehearsal/principal/cyax125-20261101-route")
        self.assertEqual(adapter.locks[16], f"refs/heads/rehearsal/lifecycle/requests/{request.request_id}")

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
        self.assertEqual(result["status"], "BLOCKED_NOT_VERIFIABLE")
        self.assertEqual(adapter.executed, ["P2-app-create"])

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

    def test_valid_ff_move_uses_force_false_and_readback(self) -> None:
        adapter, request, approval, proof, state, calls, ref, _target = self._case_fixture(
            status=200, body=b'{"ref":"ok"}', positive_case="4a",
        )
        proof = replace(proof, ref_sha="f12fbb8ec52e4c7510564f602a15f26173fe772e")
        state["sha"] = proof.ref_sha
        with adapter.case_lock(ref):
            result = adapter.execute_case("4a", request, approval, proof)
        self.assertEqual(result.status, "PASS_ACCEPTED")
        patch = [entry for entry in calls if entry[0] == "PATCH"]
        self.assertEqual(len(patch), 1)
        self.assertEqual(patch[0][2], {"sha": "a6df7c78e27ee81771cc06b0783a508edcaed320", "force": False})
        self.assertEqual(state["sha"], "a6df7c78e27ee81771cc06b0783a508edcaed320")

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
