"""Adversarial tests for the authenticated GitHub protection adapter."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import traceback
import unittest
from unittest import mock
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.github_protection import (  # noqa: E402
    ApiResponse,
    GitHubProtectionAdapter,
    GitHubProtectionError,
    classify_principal_branch_move,
    classify_ref_creation,
    validate_rehearsal_trigger,
)


REPOSITORY = "Julia-meets-String-Theory/CYAxiverse.jl"
OBSERVATION_REF = "refs/heads/candidates/cyax-0125/gate-a-observation"
SHA_A = "a" * 40
SHA_B = "b" * 40
TIMESTAMP = "2026-09-24T15:00:00Z"
APP_BYPASS = {"actor_id": 5371554, "actor_type": "Integration", "bypass_mode": "always"}
ADMIN_BYPASS = {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}


def rule(kind: str) -> dict[str, object]:
    return {"type": kind}


def ruleset(
    identifier: int,
    name: str,
    *,
    target: str = "branch",
    enforcement: str = "active",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    rules: list[dict[str, object]] | None = None,
    actors: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": identifier,
        "name": name,
        "target": target,
        "enforcement": enforcement,
        "conditions": {
            "ref_name": {
                "include": include or [],
                "exclude": exclude or [],
            }
        },
        "rules": rules or [],
        "bypass_actors": actors or [],
        "updated_at": TIMESTAMP,
    }


def candidate_rulesets() -> list[dict[str, object]]:
    pattern = "refs/heads/candidates/**/*"
    return [
        ruleset(
            23967983,
            "CYAx candidate refs immutable",
            include=[pattern],
            rules=[rule("deletion"), rule("non_fast_forward"), rule("update")],
        ),
        ruleset(
            23967991,
            "CYAx candidate refs creation",
            include=[pattern],
            rules=[rule("creation")],
            actors=[{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}],
        ),
    ]


def freeze_rulesets() -> list[dict[str, object]]:
    return [
        ruleset(
            23948123,
            "CYAx vmm lifecycle freeze",
            enforcement="disabled",
            include=["refs/heads/vmm"],
            rules=[rule("update"), rule("deletion"), rule("non_fast_forward")],
        ),
        ruleset(
            23948127,
            "CYAx main lifecycle freeze",
            enforcement="disabled",
            include=["refs/heads/main"],
            rules=[rule("update"), rule("deletion"), rule("non_fast_forward")],
        ),
    ]


def public_tag_rulesets() -> list[dict[str, object]]:
    return [
        ruleset(
            23948086,
            "CYAx canonical release tags immutable",
            target="tag",
            include=["refs/tags/v*.*.*"],
            exclude=["refs/tags/v-0.1"],
            rules=[rule("update"), rule("deletion"), rule("non_fast_forward")],
        ),
        ruleset(
            23948090,
            "CYAx canonical release tag creation",
            target="tag",
            include=["refs/tags/v*.*.*"],
            exclude=["refs/tags/v-0.1"],
            rules=[rule("creation")],
            actors=[{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}],
        ),
    ]


def rehearsal_rulesets() -> list[dict[str, object]]:
    lifecycle = "refs/heads/rehearsal/lifecycle/**"
    principal = "refs/heads/rehearsal/principal/*"
    return [
        ruleset(
            51001, "CYAx rehearsal lifecycle creation", include=[lifecycle],
            rules=[rule("creation")], actors=[ADMIN_BYPASS, APP_BYPASS],
        ),
        ruleset(
            51002, "CYAx rehearsal lifecycle immutable", include=[lifecycle],
            rules=[rule("deletion"), rule("non_fast_forward"), rule("update")],
        ),
        ruleset(
            52001, "CYAx rehearsal principal integrity", include=[principal],
            rules=[rule("deletion"), rule("non_fast_forward")],
        ),
        ruleset(
            52002, "CYAx rehearsal principal review", include=[principal],
            rules=[{
                "type": "pull_request",
                "parameters": {
                    "required_approving_review_count": 0,
                    "dismiss_stale_reviews_on_push": False,
                    "require_code_owner_review": False,
                    "require_last_push_approval": False,
                },
            }],
            actors=[ADMIN_BYPASS, APP_BYPASS],
        ),
        ruleset(
            52003, "CYAx rehearsal principal freeze", include=[principal],
            rules=[rule("update")], actors=[ADMIN_BYPASS, APP_BYPASS],
        ),
    ]


class FakeGitHub:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows if rows is not None else (
            candidate_rulesets() + freeze_rulesets() + rehearsal_rulesets()
        )
        self.calls: list[tuple[str, str, object]] = []
        self.fail_next = False
        self.stale_detail = False
        self.change_main_after_write = False
        self.move_main_after_disable = False
        self.freeze_write_mode: str | None = None
        self.late_freeze_apply: tuple[int, str] | None = None
        self.branch_reads = {"vmm": 0, "main": 0}
        self.branch_shas = {"vmm": "a" * 40, "main": "a" * 40}
        self.change_vmm_after_write = False
        self.fail_branch_read_at: dict[str, set[int]] = {}
        self.fail_project_version = False
        self.vmm_tree = "c" * 40
        self.request_files: set[str] = set()
        self.git_ref_objects: dict[str, str] = {}
        self.git_commit_objects: dict[str, dict[str, object]] = {}
        self.ref_create_mode: str | None = None
        self.ref_create_calls = 0
        self.fail_fence_readback = False
        self.object_write_mode: str | None = None
        self.object_write_calls = 0
        self.project_version = "0.2.0"
        self.evaluated_rules = [
            {"ruleset_id": 23967991, "type": "creation"},
            {"ruleset_id": 23967983, "type": "update"},
            {"ruleset_id": 23967983, "type": "deletion"},
            {"ruleset_id": 23967983, "type": "non_fast_forward"},
        ]
        self.lifecycle_evaluated_rules = [
            {"ruleset_id": 51001, "type": "creation"},
            {"ruleset_id": 51002, "type": "deletion"},
            {"ruleset_id": 51002, "type": "non_fast_forward"},
            {"ruleset_id": 51002, "type": "update"},
        ]
        self.principal_evaluated_rules = [
            {"ruleset_id": 52001, "type": "deletion"},
            {"ruleset_id": 52001, "type": "non_fast_forward"},
            {"ruleset_id": 52002, "type": "pull_request"},
            {"ruleset_id": 52003, "type": "update"},
        ]

    def __call__(self, method: str, path: str, payload=None) -> ApiResponse:
        self.calls.append((method, path, payload))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Bearer private-token must not escape")
        if "/rulesets?" in path and method == "GET":
            summaries = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "target": row["target"],
                    "enforcement": row["enforcement"],
                    "updated_at": (
                        "2026-09-24T14:59:59Z"
                        if self.stale_detail and row["id"] == 23967983
                        else row["updated_at"]
                    ),
                }
                for row in reversed(self.rows)
            ]
            return self._response(summaries, {"ETag": '"ruleset-list"'})
        if "/rulesets/" in path and method == "GET":
            identifier = int(path.rsplit("/", 1)[1])
            row = next((row for row in self.rows if row["id"] == identifier), None)
            if row is None:
                return ApiResponse(404, b"{}")
            response = self._response(row)
            if self.late_freeze_apply and self.late_freeze_apply[0] == identifier:
                _, enforcement = self.late_freeze_apply
                self.late_freeze_apply = None
                row["enforcement"] = enforcement
                row["updated_at"] = "2026-09-24T15:00:02.002Z"
            return response
        if "/rulesets/" in path and method in {"PATCH", "PUT"}:
            identifier = int(path.rsplit("/", 1)[1])
            row = next(row for row in self.rows if row["id"] == identifier)
            mode = self.freeze_write_mode
            self.freeze_write_mode = None
            if mode == "terminal-rejection":
                return ApiResponse(403, b"{}")
            if mode in {"response-lost-before", "late-apply-after-readback"}:
                if mode == "late-apply-after-readback":
                    self.late_freeze_apply = (identifier, payload["enforcement"])
                raise RuntimeError("Bearer private-token must not escape")
            if mode == "response-lost-readback-fails":
                self.fail_next = True
                raise RuntimeError("Bearer private-token must not escape")
            row["enforcement"] = payload["enforcement"]
            row["updated_at"] = "2026-09-24T15:00:01.001Z"
            if (
                identifier == 23948127
                and payload["enforcement"] == "disabled"
                and self.move_main_after_disable
            ):
                self.branch_shas["main"] = SHA_B
            if mode == "response-lost-after":
                raise RuntimeError("Bearer private-token must not escape")
            return self._response(row)
        if "/rules/branches/" in path and method == "GET":
            branch = unquote(path.split("/rules/branches/", 1)[1])
            if branch.startswith("rehearsal/lifecycle/"):
                evaluated = self.lifecycle_evaluated_rules
            elif branch.startswith("rehearsal/principal/"):
                evaluated = self.principal_evaluated_rules
            else:
                evaluated = self.evaluated_rules
            return self._response(evaluated, {
                "ETag": '"effective-rules"',
                "X-GitHub-Request-Id": "REQ-0125",
            })
        if "/git/ref/heads/rehearsal/lifecycle/requests/" in path and method == "GET":
            branch = path.split("/git/ref/heads/", 1)[1]
            if self.fail_fence_readback and self.ref_create_calls:
                raise RuntimeError("readback unavailable")
            if branch not in self.git_ref_objects:
                return ApiResponse(404, b"{}")
            return self._response({
                "ref": f"refs/heads/{branch}",
                "object": {"type": "commit", "sha": self.git_ref_objects[branch]},
            })
        if "/git/commits/" in path and method == "GET":
            commit_sha = path.rsplit("/", 1)[1]
            value = self.git_commit_objects.get(commit_sha)
            return ApiResponse(404, b"{}") if value is None else self._response(value)
        if "/contents/lifecycle-requests/" in path and method == "GET":
            file_path = path.split("/contents/", 1)[1].split("?", 1)[0]
            return self._response({"path": file_path}) if file_path in self.request_files else ApiResponse(404, b"{}")
        if "/branches/vmm" in path and method == "GET":
            self.branch_reads["vmm"] = self.branch_reads.get("vmm", 0) + 1
            if self.branch_reads["vmm"] in self.fail_branch_read_at.get("vmm", set()):
                raise RuntimeError("branch read unavailable")
            sha = self.branch_shas["vmm"]
            if self.change_vmm_after_write and self.branch_reads["vmm"] > 1:
                sha = SHA_B
            return self._response({
                "name": "vmm",
                "commit": {
                    "sha": sha,
                    "commit": {"tree": {"sha": self.vmm_tree}},
                },
            })
        if path.endswith("/git/blobs") and method == "POST":
            self.object_write_calls += 1
            raw = base64.b64decode(payload["content"])
            sha = hashlib.sha1(
                f"blob {len(raw)}\0".encode("ascii") + raw
            ).hexdigest()
            if self.object_write_mode == "response-lost-after-blob":
                self.object_write_mode = None
                raise RuntimeError("Bearer private-token must not escape")
            return self._response({"sha": sha})
        if path.endswith("/git/trees") and method == "POST":
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            sha = hashlib.sha1(b"tree\0" + raw).hexdigest()
            return self._response({"sha": sha})
        if path.endswith("/git/commits") and method == "POST":
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            sha = hashlib.sha1(b"commit\0" + raw).hexdigest()
            self.git_commit_objects[sha] = {
                "sha": sha,
                "tree": {"sha": payload["tree"]},
                "parents": [{"sha": parent} for parent in payload["parents"]],
            }
            return self._response({"sha": sha})
        if path.endswith("/git/refs") and method == "POST":
            self.ref_create_calls += 1
            ref = payload["ref"]
            branch = ref.removeprefix("refs/heads/")
            mode = self.ref_create_mode
            self.ref_create_mode = None
            if branch in self.git_ref_objects:
                return ApiResponse(422, b"{}")
            if mode == "reject-race":
                self.git_ref_objects[branch] = SHA_B
                return ApiResponse(422, b"{}")
            if mode == "terminal-reject":
                return ApiResponse(422, b"{}")
            if mode == "terminal-reject-readback-fails":
                self.fail_fence_readback = True
                return ApiResponse(422, b"{}")
            if mode == "response-lost-after":
                self.git_ref_objects[branch] = payload["sha"]
                raise RuntimeError("Bearer private-token must not escape")
            self.git_ref_objects[branch] = payload["sha"]
            return self._response({
                "ref": ref,
                "object": {"type": "commit", "sha": payload["sha"]},
            }, {"X-GitHub-Request-Id": "REQ-FENCE-1"})
        if "/branches/" in path and method == "GET":
            branch = unquote(path.rsplit("/", 1)[1])
            self.branch_reads[branch] = self.branch_reads.get(branch, 0) + 1
            if self.branch_reads[branch] in self.fail_branch_read_at.get(branch, set()):
                raise RuntimeError("branch read unavailable")
            sha = self.branch_shas.get(branch, SHA_A)
            if branch == "main" and self.change_main_after_write and self.branch_reads[branch] > 1:
                sha = SHA_B
            if branch == "vmm" and self.change_vmm_after_write and self.branch_reads[branch] > 1:
                sha = SHA_B
            return self._response({"name": branch, "commit": {"sha": sha}})
        if "/contents/Project.toml?" in path and method == "GET":
            if self.fail_project_version:
                raise RuntimeError("version read unavailable")
            content = base64.b64encode(
                f'version = "{self.project_version}"\n'.encode("ascii")
            ).decode("ascii")
            return self._response({"path": "Project.toml", "encoding": "base64", "content": content})
        return ApiResponse(404, b"{}")

    @staticmethod
    def _response(value, headers=None) -> ApiResponse:
        return ApiResponse(200, json.dumps(value, sort_keys=True).encode(), headers or {})


def adapter_for(fake: FakeGitHub, *, gate=None) -> GitHubProtectionAdapter:
    return GitHubProtectionAdapter(
        REPOSITORY,
        token="private-token",
        transport=fake,
        clock=lambda: datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc),
        authorization_gate=gate,
    )


class GitHubProtectionTests(unittest.TestCase):
    @staticmethod
    def _git(repository: Path, *args: str, input_bytes: bytes | None = None) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            input=input_bytes,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
        return result.stdout.decode("utf-8").strip()

    def _trigger_repo(self, root: Path):
        self._git(root, "init", "-q")
        self._git(root, "config", "user.name", "Owner Fixture")
        self._git(root, "config", "user.email", "owner@example.invalid")
        workflow = root / ".github/workflows/cyax-lifecycle-dispatch.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("name: test workflow\n", encoding="utf-8")
        code_dir = root / "scripts/version_lifecycle"
        code_dir.mkdir(parents=True)
        for name in ("__init__.py", "github_protection.py", "transactions.py"):
            (code_dir / name).write_text(f"# {name}\n", encoding="utf-8")
        self._git(root, "add", ".")
        self._git(root, "commit", "-q", "-m", "reviewed implementation")
        parent = self._git(root, "rev-parse", "HEAD")
        workflow_blob = self._git(
            root, "rev-parse", f"{parent}:.github/workflows/cyax-lifecycle-dispatch.yml"
        )
        called_code_blobs = {}
        for name in ("__init__.py", "github_protection.py", "transactions.py"):
            path = f"scripts/version_lifecycle/{name}"
            called_code_blobs[path] = self._git(root, "rev-parse", f"{parent}:{path}")
        request = {
            "schema_version": 1,
            "request_id": "request-0125-1",
            "implementation_sha": parent,
            "workflow_blob_sha": workflow_blob,
            "expected_vmm_sha": SHA_A,
            "called_code_blobs": called_code_blobs,
            "purpose": "gate-a-rehearsal",
        }
        request_path = root / "lifecycle-requests/request-0125-1.json"
        request_path.parent.mkdir()
        request_path.write_bytes(
            json.dumps(
                request, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        )
        self._git(root, "add", "lifecycle-requests/request-0125-1.json")
        self._git(root, "commit", "-q", "-m", "Owner rehearsal request")
        commit = self._git(root, "rev-parse", "HEAD")
        event = {
            "ref": "refs/heads/rehearsal/trigger/request-0125-1",
            "before": "0" * 40,
            "after": commit,
            "repository": {"full_name": REPOSITORY},
            "sender": {"type": "User", "login": "owner"},
            "created": True,
            "deleted": False,
            "forced": False,
        }
        return parent, commit, event

    def test_real_transport_is_pinned_to_the_governed_github_api_host(self) -> None:
        with self.assertRaisesRegex(GitHubProtectionError, "API_ORIGIN_INVALID"):
            GitHubProtectionAdapter(
                REPOSITORY,
                token="private-token",
                api_base="https://api.github.com.attacker.example",
            )

        # A supplied transport is the test seam and never sends Authorization
        # headers through api_base.
        fake = FakeGitHub()
        adapter = GitHubProtectionAdapter(
            REPOSITORY,
            token="private-token",
            api_base="https://fixture.invalid",
            transport=fake,
        )
        self.assertEqual(adapter.read_snapshot().repository, REPOSITORY)

    def test_dispatch_workflow_is_one_environment_gated_push_job_with_closed_permissions(self) -> None:
        workflow = (
            Path(__file__).resolve().parent.parent
            / ".github/workflows/cyax-lifecycle-dispatch.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("branches:\n      - rehearsal/trigger/*", workflow)
        self.assertIn("paths:\n      - lifecycle-requests/**", workflow)
        self.assertIn("permissions: {}", workflow)
        self.assertIn("environment:\n      name: lifecycle", workflow)
        self.assertEqual(workflow.count("\n  lifecycle:\n"), 1)
        self.assertEqual(workflow.count("      - name:"), 3)
        self.assertLess(
            workflow.index("Validate the exact Owner trigger"),
            workflow.index("Mint a scoped token and create the request fence"),
        )
        self.assertEqual(workflow.count("secrets.LIFECYCLE_APP_PRIVATE_KEY"), 1)
        for forbidden in (
            "workflow_dispatch:",
            "pull_request:",
            "pull_request_target:",
            "schedule:",
            "repository_dispatch:",
        ):
            self.assertNotIn(forbidden, workflow)
        self.assertIn('"contents" => "write"', workflow)
        self.assertNotIn("permission-workflows:", workflow)
        self.assertNotIn("permission-administration:", workflow)
        self.assertNotIn("uses:", workflow)
        self.assertIn("fetch --no-tags --depth=2 origin", workflow)
        self.assertIn("unset auth GITHUB_TOKEN", workflow)
        self.assertIn("github.event.installation.id != 172789630", workflow)
        self.assertIn("github.run_attempt == 1", workflow)

    def test_trigger_validation_binds_one_parent_one_added_request_and_code_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, commit, event = self._trigger_repo(root)
            request = validate_rehearsal_trigger(
                root,
                repository=REPOSITORY,
                ref=event["ref"],
                commit_sha=commit,
                actor="owner",
                event=event,
            )
            self.assertEqual(request.parent_sha, parent)
            self.assertEqual(request.commit_sha, commit)
            self.assertEqual(request.request_id, "request-0125-1")
            self.assertEqual(
                set(request.called_code_blobs),
                {
                    "scripts/version_lifecycle/__init__.py",
                    "scripts/version_lifecycle/github_protection.py",
                    "scripts/version_lifecycle/transactions.py",
                },
            )
            wrong_actor = dict(event)
            wrong_actor["sender"] = {"type": "Bot", "login": "cyaxiverse-lifecycle[bot]"}
            wrong_actor["installation"] = {"id": 172789630}
            with self.assertRaisesRegex(GitHubProtectionError, "TRIGGER_ACTOR_INVALID"):
                validate_rehearsal_trigger(
                    root, repository=REPOSITORY, ref=event["ref"],
                    commit_sha=commit, actor="cyaxiverse-lifecycle[bot]", event=wrong_actor,
                )
            wrong_parent = dict(event, before=SHA_A)
            with self.assertRaisesRegex(
                GitHubProtectionError, "TRIGGER_PUSH_RANGE_INVALID"
            ):
                validate_rehearsal_trigger(
                    root, repository=REPOSITORY, ref=event["ref"],
                    commit_sha=commit, actor="owner", event=wrong_parent,
                )
            replayed = dict(event, created=False, before=commit)
            with self.assertRaisesRegex(
                GitHubProtectionError, "TRIGGER_IDENTITY_INVALID"
            ):
                validate_rehearsal_trigger(
                    root, repository=REPOSITORY, ref=event["ref"],
                    commit_sha=commit, actor="owner", event=replayed,
                )

    def test_local_git_validation_never_inherits_tokens_or_git_injection_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, commit, event = self._trigger_repo(root)
            captured: list[dict[str, str]] = []
            original_run = subprocess.run

            def capture_git_environment(command, **kwargs):
                if command[0] == "git":
                    captured.append(dict(kwargs.get("env", {})))
                return original_run(command, **kwargs)

            sentinels = {
                "GH_TOKEN": "gh-secret-sentinel",
                "GITHUB_TOKEN": "github-secret-sentinel",
                "GH_ENTERPRISE_TOKEN": "enterprise-secret-sentinel",
                "GIT_DIR": "/tmp/injected-git-dir",
                "GIT_WORK_TREE": "/tmp/injected-work-tree",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "credential.helper",
                "GIT_CONFIG_VALUE_0": "!printf injected",
            }
            with mock.patch.dict(os.environ, sentinels, clear=False), mock.patch(
                "version_lifecycle.github_protection.subprocess.run",
                side_effect=capture_git_environment,
            ):
                request = validate_rehearsal_trigger(
                    root,
                    repository=REPOSITORY,
                    ref=event["ref"],
                    commit_sha=commit,
                    actor="owner",
                    event=event,
                )
            self.assertEqual(request.request_id, "request-0125-1")
            self.assertGreater(len(captured), 0)
            for environment in captured:
                for key in set(sentinels) - {
                    "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"
                }:
                    self.assertNotIn(key, environment)
                for value in set(sentinels.values()) - {"1", "credential.helper", "!printf injected"}:
                    self.assertNotIn(value, environment.values())
                self.assertEqual(environment["GIT_CONFIG_NOSYSTEM"], "1")
                self.assertEqual(environment["GIT_CONFIG_GLOBAL"], os.devnull)
                self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
                self.assertEqual(environment["GCM_INTERACTIVE"], "never")
                self.assertEqual(environment["GIT_CONFIG_COUNT"], "2")
                self.assertEqual(environment["GIT_CONFIG_KEY_0"], "credential.helper")
                self.assertEqual(environment["GIT_CONFIG_VALUE_0"], "")
                self.assertEqual(environment["GIT_CONFIG_KEY_1"], "core.hooksPath")
                self.assertEqual(
                    environment["GIT_CONFIG_VALUE_1"], os.devnull
                )

    def test_trigger_validation_rejects_extra_or_modified_workflow_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, commit, event = self._trigger_repo(root)
            request_bytes = (root / "lifecycle-requests/request-0125-1.json").read_bytes()
            self._git(root, "reset", "--hard", parent)
            extra = root / "extra.txt"
            extra.write_text("unexpected\n", encoding="utf-8")
            request_path = root / "lifecycle-requests/request-0125-1.json"
            request_path.parent.mkdir()
            request_path.write_bytes(request_bytes)
            self._git(root, "add", "extra.txt", "lifecycle-requests/request-0125-1.json")
            self._git(root, "commit", "-q", "-m", "request plus extra path")
            commit = self._git(root, "rev-parse", "HEAD")
            event = dict(event, after=commit)
            with self.assertRaisesRegex(GitHubProtectionError, "TRIGGER_DIFF_INVALID"):
                validate_rehearsal_trigger(
                    root, repository=REPOSITORY, ref=event["ref"],
                    commit_sha=commit, actor="owner", event=event,
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, commit, event = self._trigger_repo(root)
            request_bytes = (root / "lifecycle-requests/request-0125-1.json").read_bytes()
            self._git(root, "reset", "--hard", parent)
            workflow = root / ".github/workflows/cyax-lifecycle-dispatch.yml"
            workflow.write_text("name: changed\n", encoding="utf-8")
            request_path = root / "lifecycle-requests/request-0125-1.json"
            request_path.parent.mkdir()
            request_path.write_bytes(request_bytes)
            self._git(
                root, "add", ".github/workflows/cyax-lifecycle-dispatch.yml",
                "lifecycle-requests/request-0125-1.json",
            )
            self._git(root, "commit", "-q", "-m", "request plus workflow change")
            commit = self._git(root, "rev-parse", "HEAD")
            event = dict(event, after=commit)
            with self.assertRaisesRegex(GitHubProtectionError, "TRIGGER_DIFF_INVALID"):
                validate_rehearsal_trigger(
                    root, repository=REPOSITORY, ref=event["ref"],
                    commit_sha=commit, actor="owner", event=event,
                )

    def test_owner_route_classifier_scans_every_commit_and_rename_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._git(root, "init", "-q")
            self._git(root, "config", "user.name", "Fixture")
            self._git(root, "config", "user.email", "fixture@example.invalid")
            workflows = root / ".github/workflows"
            workflows.mkdir(parents=True)
            (workflows / "one.yml").write_text("name: first\n", encoding="utf-8")
            (root / "README.md").write_text("readme\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-q", "-m", "base")
            old = self._git(root, "rev-parse", "HEAD")
            (workflows / "one.yml").write_text("name: changed\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-q", "-m", "modify workflow")
            (workflows / "one.yml").write_text("name: first\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-q", "-m", "restore workflow")
            (workflows / "one.yml").rename(workflows / "renamed.yml")
            self._git(root, "add", "-A")
            self._git(root, "commit", "-q", "-m", "rename workflow")
            new = self._git(root, "rev-parse", "HEAD")
            decision = classify_principal_branch_move(
                root,
                ref="refs/heads/rehearsal/principal/line",
                expected_old_sha=old,
                intended_new_sha=new,
            )
            self.assertEqual(decision.status, "OWNER_ROUTE_REQUIRED")
            self.assertEqual(decision.reason_code, "WORKFLOW_PATH_CHANGED")
            self.assertEqual(len(decision.commits_examined), 3)
            self.assertEqual(set(decision.workflow_paths), {
                ".github/workflows/one.yml", ".github/workflows/renamed.yml"
            })

    def test_ref_creation_classifier_keeps_canonical_tags_and_rehearsal_paths_separate(self) -> None:
        self.assertEqual(
            classify_ref_creation("refs/tags/v1.2.3").status,
            "OWNER_ROUTE_REQUIRED",
        )
        self.assertEqual(
            classify_ref_creation("refs/tags/rehearsal/v1.2.3").status,
            "REJECTED_BY_RULESET",
        )
        self.assertEqual(
            classify_ref_creation(
                "refs/heads/rehearsal/lifecycle/requests/r-1"
            ).status,
            "APP_CREATE_ELIGIBLE",
        )
        self.assertEqual(
            classify_ref_creation(
                "refs/heads/rehearsal/lifecycle/requests/r-2",
                workflow_content_rejected=True,
            ).status,
            "OWNER_ROUTE_REQUIRED",
        )

    def test_principal_cas_checks_owner_route_authorization_and_exact_old_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._git(root, "init", "-q")
            self._git(root, "config", "user.name", "Fixture")
            self._git(root, "config", "user.email", "fixture@example.invalid")
            (root / "README.md").write_text("base\n", encoding="utf-8")
            self._git(root, "add", "README.md")
            self._git(root, "commit", "-q", "-m", "base")
            old = self._git(root, "rev-parse", "HEAD")
            (root / "README.md").write_text("next\n", encoding="utf-8")
            self._git(root, "add", "README.md")
            self._git(root, "commit", "-q", "-m", "next")
            new = self._git(root, "rev-parse", "HEAD")
            tree = self._git(root, "rev-parse", f"{new}^{{tree}}")

            fake = FakeGitHub()
            branch = "rehearsal/principal/line"
            fake.branch_shas[branch] = old
            gate_calls = []
            adapter = GitHubProtectionAdapter(
                REPOSITORY,
                token="private-token",
                transport=fake,
                authorization_gate=lambda intent, action, ref: (
                    gate_calls.append(("authorization", action, ref)) or True
                ),
                rehearsal_gate=lambda intent, action, ref, _rules: (
                    gate_calls.append(("rules", action, ref)) or True
                ),
            )
            original_run = subprocess.run
            push_calls = []
            push_environments = []

            def successful_push(command, **kwargs):
                if command[0] == "git" and "push" in command:
                    push_calls.append(command)
                    push_environments.append(kwargs["env"])
                    fake.branch_shas[branch] = new
                    return subprocess.CompletedProcess(command, 0, b"", b"")
                return original_run(command, **kwargs)

            expected_rules = fake.principal_evaluated_rules
            fake.principal_evaluated_rules = []
            with mock.patch(
                "version_lifecycle.github_protection.subprocess.run",
                side_effect=successful_push,
            ):
                with self.assertRaisesRegex(
                    GitHubProtectionError, "REHEARSAL_RULESET_UNVERIFIED"
                ):
                    adapter.compare_and_swap_rehearsal_principal(
                        root,
                        ref=f"refs/heads/{branch}",
                        expected_old_sha=old,
                        intended_new_sha=new,
                        intended_new_tree=tree,
                        intent={"repository": REPOSITORY, "transaction_id": "tx-rules"},
                    )
            self.assertEqual(push_calls, [])
            fake.principal_evaluated_rules = expected_rules
            gate_calls.clear()

            intent = {
                "repository": REPOSITORY,
                "transaction_id": "tx-rehearsal",
            }
            with mock.patch(
                "version_lifecycle.github_protection.subprocess.run",
                side_effect=successful_push,
            ):
                result = adapter.compare_and_swap_rehearsal_principal(
                    root,
                    ref=f"refs/heads/{branch}",
                    expected_old_sha=old,
                    intended_new_sha=new,
                    intended_new_tree=tree,
                    intent=intent,
                )
            self.assertEqual(result["observed_sha"], new)
            self.assertTrue(result["postwrite_rules_verified"])
            self.assertIn(
                f"--force-with-lease=refs/heads/{branch}:{old}", push_calls[0]
            )
            self.assertNotIn("GH_TOKEN", push_environments[0])
            self.assertNotIn("GITHUB_TOKEN", push_environments[0])
            self.assertEqual(push_environments[0]["GIT_CONFIG_VALUE_1"], "")
            self.assertEqual(push_environments[0]["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(
                gate_calls,
                [
                    ("authorization", "move-rehearsal-principal", f"refs/heads/{branch}"),
                    ("rules", "move-rehearsal-principal", f"refs/heads/{branch}"),
                    ("rules", "verify-principal-move-postwrite", f"refs/heads/{branch}"),
                ],
            )

            fake.branch_shas[branch] = SHA_B
            push_calls.clear()
            with mock.patch(
                "version_lifecycle.github_protection.subprocess.run",
                side_effect=successful_push,
            ):
                with self.assertRaisesRegex(
                    GitHubProtectionError, "CAS_STALE_EXPECTED_OLD"
                ):
                    adapter.compare_and_swap_rehearsal_principal(
                        root,
                        ref=f"refs/heads/{branch}",
                        expected_old_sha=old,
                        intended_new_sha=new,
                        intended_new_tree=tree,
                        intent=intent,
                    )
            self.assertEqual(push_calls, [])

    def test_principal_cas_stops_before_push_for_workflow_changes_and_unknown_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._git(root, "init", "-q")
            self._git(root, "config", "user.name", "Fixture")
            self._git(root, "config", "user.email", "fixture@example.invalid")
            workflow_dir = root / ".github/workflows"
            workflow_dir.mkdir(parents=True)
            workflow = workflow_dir / "reviewed.yml"
            workflow.write_text("name: reviewed\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-q", "-m", "base")
            old = self._git(root, "rev-parse", "HEAD")
            workflow.write_text("name: changed\n", encoding="utf-8")
            self._git(root, "add", ".")
            self._git(root, "commit", "-q", "-m", "change workflow")
            new = self._git(root, "rev-parse", "HEAD")
            tree = self._git(root, "rev-parse", f"{new}^{{tree}}")
            branch = "rehearsal/principal/line"
            fake = FakeGitHub()
            fake.branch_shas[branch] = old
            adapter = GitHubProtectionAdapter(
                REPOSITORY,
                token="private-token",
                transport=fake,
                authorization_gate=lambda *_: True,
                rehearsal_gate=lambda *_: True,
            )
            original_run = subprocess.run
            push_calls = []

            def failed_push(command, **kwargs):
                if command[0] == "git" and "push" in command:
                    push_calls.append(command)
                    return subprocess.CompletedProcess(command, 1, b"", b"unknown")
                return original_run(command, **kwargs)

            with mock.patch(
                "version_lifecycle.github_protection.subprocess.run",
                side_effect=failed_push,
            ):
                with self.assertRaisesRegex(
                    GitHubProtectionError, "OWNER_ROUTE_REQUIRED"
                ):
                    adapter.compare_and_swap_rehearsal_principal(
                        root,
                        ref=f"refs/heads/{branch}",
                        expected_old_sha=old,
                        intended_new_sha=new,
                        intended_new_tree=tree,
                        intent={"repository": REPOSITORY, "transaction_id": "tx-1"},
                    )
            self.assertEqual(push_calls, [])

            # A clean-range push with an unknown transport result is read back
            # once and fenced against any second write in this process.
            (root / "README.md").write_text("readme\n", encoding="utf-8")
            self._git(root, "add", "README.md")
            self._git(root, "commit", "-q", "-m", "restore non-workflow range")
            # Use the previous workflow commit as the expected old head so this
            # final commit contains no workflow path change.
            clean_old = new
            clean_new = self._git(root, "rev-parse", "HEAD")
            clean_tree = self._git(root, "rev-parse", f"{clean_new}^{{tree}}")
            fake.branch_shas[branch] = clean_old
            push_calls.clear()
            with mock.patch(
                "version_lifecycle.github_protection.subprocess.run",
                side_effect=failed_push,
            ):
                with self.assertRaisesRegex(GitHubProtectionError, "REF_MOVE_UNKNOWN"):
                    adapter.compare_and_swap_rehearsal_principal(
                        root,
                        ref=f"refs/heads/{branch}",
                        expected_old_sha=clean_old,
                        intended_new_sha=clean_new,
                        intended_new_tree=clean_tree,
                        intent={"repository": REPOSITORY, "transaction_id": "tx-2"},
                    )
                with self.assertRaisesRegex(
                    GitHubProtectionError, "REF_MOVE_UNRECONCILED"
                ):
                    adapter.compare_and_swap_rehearsal_principal(
                        root,
                        ref=f"refs/heads/{branch}",
                        expected_old_sha=clean_old,
                        intended_new_sha=clean_new,
                        intended_new_tree=clean_tree,
                        intent={"repository": REPOSITORY, "transaction_id": "tx-2"},
                    )
            self.assertEqual(len(push_calls), 1)

    def test_request_fence_is_create_once_and_never_moves_vmm(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, event = self._trigger_repo(root)
            request = validate_rehearsal_trigger(
                root,
                repository=REPOSITORY,
                ref=event["ref"],
                commit_sha=event["after"],
                actor="owner",
                event=event,
            )
        fake = FakeGitHub()
        gate_calls: list[tuple[str, str, dict[str, object]]] = []

        def approve_request_fence(intent, action, ref, rules):
            gate_calls.append((action, ref, rules))
            return (
                action == "create-request-fence"
                and intent.get("owner_environment_approved") is True
                and rules.get("repository") == REPOSITORY
                and rules.get("ref") == ref
                and rules.get("policy") == "cyax-r15-rehearsal-request-fence-v1"
                and adapter.is_live_rehearsal_rules_evidence(
                    rules, ref=ref, policy="request-fence"
                )
            )

        adapter = GitHubProtectionAdapter(
            REPOSITORY,
            token="private-token",
            transport=fake,
            clock=lambda: datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc),
            rehearsal_gate=approve_request_fence,
        )
        result = adapter.create_rehearsal_request_fence(
            request,
            intent={
                "repository": REPOSITORY,
                "request_id": request.request_id,
                "owner_environment_approved": True,
            },
        )
        self.assertEqual(result["expected_vmm_sha"], SHA_A)
        self.assertEqual(result["observed_commit_sha"], result["intended_commit_sha"])
        self.assertEqual(fake.branch_shas["vmm"], SHA_A)
        self.assertEqual(
            len([
                call for call in fake.calls
                if call[0] == "POST" and call[1].endswith("/git/refs")
            ]),
            1,
        )
        with self.assertRaisesRegex(
            GitHubProtectionError, "REQUEST_ID_ALREADY_USED"
        ):
            adapter.create_rehearsal_request_fence(
                request,
                intent={
                    "repository": REPOSITORY,
                    "request_id": request.request_id,
                    "owner_environment_approved": True,
                },
            )
        self.assertEqual(
            len([
                call for call in fake.calls
                if call[0] == "POST" and call[1].endswith("/git/refs")
            ]),
            1,
        )
        self.assertEqual(len(gate_calls), 2)
        self.assertEqual(gate_calls[0][0], "create-request-fence")
        self.assertEqual(
            gate_calls[0][1],
            f"refs/heads/rehearsal/lifecycle/requests/{request.request_id}",
        )
        self.assertEqual(gate_calls[0][2]["ref"], gate_calls[0][1])
        self.assertEqual(
            set(gate_calls[0][2]["rule_types"]),
            {"creation", "deletion", "non_fast_forward", "update"},
        )

        unprotected = FakeGitHub()
        unprotected.lifecycle_evaluated_rules = []
        unprotected_adapter = GitHubProtectionAdapter(
            REPOSITORY,
            token="private-token",
            transport=unprotected,
            rehearsal_gate=lambda *_: True,
        )
        with self.assertRaisesRegex(
            GitHubProtectionError, "REHEARSAL_RULESET_UNVERIFIED"
        ):
            unprotected_adapter.create_rehearsal_request_fence(
                request,
                intent={
                    "repository": REPOSITORY,
                    "request_id": request.request_id,
                },
            )
        self.assertFalse(any(call[0] == "POST" for call in unprotected.calls))

    def test_request_fence_rejects_noncanonical_effective_rules_before_writes(self) -> None:
        cases = (
            "wrong_coverage",
            "excluded_coverage",
            "disabled_immutable",
            "app_bypasses_immutable",
            "missing_creation_app_bypass",
            "inconsistent_effective_id",
            "unexpected_effective_rule",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _, _, event = self._trigger_repo(root)
                request = validate_rehearsal_trigger(
                    root, repository=REPOSITORY, ref=event["ref"],
                    commit_sha=event["after"], actor="owner", event=event,
                )
                fake = FakeGitHub()
                immutable = next(row for row in fake.rows if row["id"] == 51002)
                creation = next(row for row in fake.rows if row["id"] == 51001)
                if case == "wrong_coverage":
                    immutable["conditions"]["ref_name"]["include"] = [
                        "refs/heads/rehearsal/lifecycle/requests/*"
                    ]
                elif case == "excluded_coverage":
                    immutable["conditions"]["ref_name"]["exclude"] = [
                        "refs/heads/rehearsal/lifecycle/requests/blocked-*"
                    ]
                elif case == "disabled_immutable":
                    immutable["enforcement"] = "disabled"
                elif case == "app_bypasses_immutable":
                    immutable["bypass_actors"] = [APP_BYPASS]
                elif case == "missing_creation_app_bypass":
                    creation["bypass_actors"] = [ADMIN_BYPASS]
                elif case == "inconsistent_effective_id":
                    fake.lifecycle_evaluated_rules[0]["ruleset_id"] = 51999
                else:
                    fake.rows.append(ruleset(
                        51003,
                        "unexpected rehearsal request rule",
                        include=["refs/heads/rehearsal/lifecycle/**"],
                        rules=[rule("pull_request")],
                    ))
                    fake.lifecycle_evaluated_rules.append(
                        {"ruleset_id": 51003, "type": "pull_request"}
                    )
                adapter = GitHubProtectionAdapter(
                    REPOSITORY, token="private-token", transport=fake,
                    rehearsal_gate=lambda *_: True,
                )
                with self.assertRaisesRegex(
                    GitHubProtectionError, "REHEARSAL_RULESET_UNVERIFIED"
                ):
                    adapter.create_rehearsal_request_fence(
                        request,
                        intent={"repository": REPOSITORY, "request_id": request.request_id},
                    )
                self.assertEqual(fake.object_write_calls, 0)
                self.assertEqual(fake.ref_create_calls, 0)
                self.assertFalse(any(
                    call[0] in {"POST", "PUT", "PATCH"} for call in fake.calls
                ))

    def test_principal_move_requires_active_freeze_and_exact_effective_binding(self) -> None:
        cases = ("missing_freeze", "disabled_freeze", "excluded_freeze", "inconsistent_id")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._git(root, "init", "-q")
            self._git(root, "config", "user.name", "Fixture")
            self._git(root, "config", "user.email", "fixture@example.invalid")
            (root / "README.md").write_text("base\n", encoding="utf-8")
            self._git(root, "add", "README.md")
            self._git(root, "commit", "-q", "-m", "base")
            old = self._git(root, "rev-parse", "HEAD")
            (root / "README.md").write_text("next\n", encoding="utf-8")
            self._git(root, "add", "README.md")
            self._git(root, "commit", "-q", "-m", "next")
            new = self._git(root, "rev-parse", "HEAD")
            tree = self._git(root, "rev-parse", f"{new}^{{tree}}")
            for case in cases:
                with self.subTest(case=case):
                    fake = FakeGitHub()
                    freeze = next(row for row in fake.rows if row["id"] == 52003)
                    if case == "missing_freeze":
                        fake.rows.remove(freeze)
                        fake.principal_evaluated_rules = [
                            item for item in fake.principal_evaluated_rules
                            if item["ruleset_id"] != 52003
                        ]
                    elif case == "disabled_freeze":
                        freeze["enforcement"] = "disabled"
                    elif case == "excluded_freeze":
                        freeze["conditions"]["ref_name"]["exclude"] = [
                            "refs/heads/rehearsal/principal/line"
                        ]
                    else:
                        fake.principal_evaluated_rules[-1]["ruleset_id"] = 52999
                    fake.branch_shas["rehearsal/principal/line"] = old
                    adapter = GitHubProtectionAdapter(
                        REPOSITORY, token="private-token", transport=fake,
                        authorization_gate=lambda *_: True,
                        rehearsal_gate=lambda *_: True,
                    )
                    original_run = subprocess.run
                    pushes = []

                    def record_push(command, **kwargs):
                        if command[0] == "git" and "push" in command:
                            pushes.append(command)
                        return original_run(command, **kwargs)

                    with mock.patch(
                        "version_lifecycle.github_protection.subprocess.run",
                        side_effect=record_push,
                    ):
                        with self.assertRaisesRegex(
                            GitHubProtectionError, "REHEARSAL_RULESET_UNVERIFIED"
                        ):
                            adapter.compare_and_swap_rehearsal_principal(
                                root,
                                ref="refs/heads/rehearsal/principal/line",
                                expected_old_sha=old,
                                intended_new_sha=new,
                                intended_new_tree=tree,
                                intent={"repository": REPOSITORY, "transaction_id": "tx"},
                            )
                    self.assertEqual(pushes, [])
                    self.assertFalse(any(
                        call[0] in {"POST", "PUT", "PATCH"} for call in fake.calls
                    ))

    def test_request_fence_rechecks_rules_before_ref_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, event = self._trigger_repo(root)
            request = validate_rehearsal_trigger(
                root, repository=REPOSITORY, ref=event["ref"],
                commit_sha=event["after"], actor="owner", event=event,
            )
        fake = FakeGitHub()
        immutable = next(row for row in fake.rows if row["id"] == 51002)
        gate_calls = []
        adapter = None

        def approve_then_weaken(intent, action, ref, rules):
            approved = adapter.is_live_rehearsal_rules_evidence(
                rules, ref=ref, policy="request-fence"
            )
            gate_calls.append(action)
            if len(gate_calls) == 1:
                immutable["bypass_actors"] = [APP_BYPASS]
            return approved

        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="private-token", transport=fake,
            rehearsal_gate=approve_then_weaken,
        )
        with self.assertRaisesRegex(
            GitHubProtectionError, "REHEARSAL_RULESET_UNVERIFIED"
        ):
            adapter.create_rehearsal_request_fence(
                request,
                intent={"repository": REPOSITORY, "request_id": request.request_id},
            )
        self.assertEqual(gate_calls, ["create-request-fence"])
        self.assertEqual(fake.ref_create_calls, 0)
        self.assertFalse(any(
            call[0] == "POST" and call[1].endswith("/git/refs")
            for call in fake.calls
        ))

    def test_request_fence_race_and_lost_response_stop_without_retry(self) -> None:
        for mode, expected in (
            ("reject-race", "REQUEST_ID_ALREADY_USED"),
            ("terminal-reject", "REQUEST_FENCE_CREATE_REJECTED"),
            ("terminal-reject-readback-fails", "REQUEST_FENCE_WRITE_UNKNOWN"),
            ("response-lost-after", "REQUEST_FENCE_WRITE_UNKNOWN"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                _, _, event = self._trigger_repo(root)
                request = validate_rehearsal_trigger(
                    root, repository=REPOSITORY, ref=event["ref"],
                    commit_sha=event["after"], actor="owner", event=event,
                )
                fake = FakeGitHub()
                fake.ref_create_mode = mode
                adapter = GitHubProtectionAdapter(
                    REPOSITORY,
                    token="private-token",
                    transport=fake,
                    rehearsal_gate=lambda *_: True,
                )
                with self.assertRaisesRegex(GitHubProtectionError, expected) as raised:
                    adapter.create_rehearsal_request_fence(
                        request,
                        intent={
                            "repository": REPOSITORY,
                            "request_id": request.request_id,
                        },
                    )
                ref_posts = [
                    call for call in fake.calls
                    if call[0] == "POST" and call[1].endswith("/git/refs")
                ]
                self.assertEqual(len(ref_posts), 1)
                if mode == "terminal-reject-readback-fails":
                    self.assertEqual(
                        raised.exception.evidence["readback_error"],
                        "REQUEST_FENCE_READBACK_UNAVAILABLE",
                    )
                    self.assertEqual(
                        raised.exception.evidence["write_outcome"], "unknown"
                    )
                if expected == "REQUEST_FENCE_WRITE_UNKNOWN":
                    with self.assertRaisesRegex(
                        GitHubProtectionError, "REQUEST_FENCE_WRITE_UNRECONCILED"
                    ):
                        adapter.create_rehearsal_request_fence(
                            request,
                            intent={
                                "repository": REPOSITORY,
                                "request_id": request.request_id,
                            },
                        )
                    self.assertEqual(
                        len([
                            call for call in fake.calls
                            if call[0] == "POST" and call[1].endswith("/git/refs")
                        ]),
                        1,
                    )

    def test_request_fence_unknown_object_write_blocks_same_adapter_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, event = self._trigger_repo(root)
            request = validate_rehearsal_trigger(
                root, repository=REPOSITORY, ref=event["ref"],
                commit_sha=event["after"], actor="owner", event=event,
            )
        fake = FakeGitHub()
        fake.object_write_mode = "response-lost-after-blob"
        adapter = GitHubProtectionAdapter(
            REPOSITORY,
            token="private-token",
            transport=fake,
            rehearsal_gate=lambda *_: True,
        )
        intent = {"repository": REPOSITORY, "request_id": request.request_id}
        with self.assertRaisesRegex(
            GitHubProtectionError, "REQUEST_FENCE_OBJECT_CREATION_UNKNOWN"
        ):
            adapter.create_rehearsal_request_fence(request, intent=intent)
        with self.assertRaisesRegex(
            GitHubProtectionError, "REQUEST_FENCE_WRITE_UNRECONCILED"
        ):
            adapter.create_rehearsal_request_fence(request, intent=intent)
        self.assertEqual(fake.object_write_calls, 1)
        self.assertFalse(any(
            call[0] == "POST" and call[1].endswith("/git/refs")
            for call in fake.calls
        ))

    def test_snapshot_is_canonical_and_digested_from_exact_details(self) -> None:
        fake = FakeGitHub()
        adapter = adapter_for(fake)
        first = adapter.read_snapshot()
        second = adapter.read_snapshot()
        self.assertEqual(first.snapshot_sha256, second.snapshot_sha256)
        self.assertEqual(first.snapshot_sha256, __import__("hashlib").sha256(first.canonical_bytes).hexdigest())
        self.assertEqual([row.id for row in first.rulesets], sorted(row["id"] for row in fake.rows))
        self.assertIn(("GET", "/repos/Julia-meets-String-Theory/CYAxiverse.jl/rulesets/23967983", None), fake.calls)

    def test_candidate_pair_produces_exact_live_protection_evidence(self) -> None:
        adapter = adapter_for(FakeGitHub())
        evidence = adapter.verify_candidate_ref(OBSERVATION_REF)
        self.assertEqual(evidence.ruleset_ids, (23967983, 23967991))
        evidence.require(OBSERVATION_REF, creation=True)
        self.assertTrue(adapter.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository=REPOSITORY
        ))
        self.assertFalse(adapter.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository="other/repository"
        ))
        with self.assertRaisesRegex(Exception, "TARGET_MISMATCH"):
            evidence.require("refs/heads/candidates/other/candidate", creation=True)
        single = adapter.verify_candidate_ref("refs/heads/candidates/single")
        single.require("refs/heads/candidates/single", creation=True)

    def test_candidate_pair_fails_closed_on_missing_or_wrong_definitions(self) -> None:
        cases = []
        missing = candidate_rulesets()[:1]
        cases.append((missing, "CANDIDATE_PROTECTION_INCOMPLETE"))
        wrong_pattern = candidate_rulesets()
        wrong_pattern[0]["conditions"]["ref_name"]["include"] = ["refs/heads/candidates/*"]
        cases.append((wrong_pattern, "CANDIDATE_PROTECTION_IMMUTABLE_INVALID"))
        wrong_enforcement = candidate_rulesets()
        wrong_enforcement[1]["enforcement"] = "disabled"
        cases.append((wrong_enforcement, "CANDIDATE_PROTECTION_CREATION_INVALID"))
        wrong_immutable_bypass = candidate_rulesets()
        wrong_immutable_bypass[0]["bypass_actors"] = [
            {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}
        ]
        cases.append((wrong_immutable_bypass, "CANDIDATE_PROTECTION_IMMUTABLE_INVALID"))
        wrong_creation_bypass = candidate_rulesets()
        wrong_creation_bypass[1]["bypass_actors"] = []
        cases.append((wrong_creation_bypass, "CANDIDATE_PROTECTION_CREATION_INVALID"))
        missing_update = candidate_rulesets()
        missing_update[0]["rules"] = [rule("deletion"), rule("non_fast_forward")]
        cases.append((missing_update, "CANDIDATE_PROTECTION_IMMUTABLE_INVALID"))
        for rows, reason in cases:
            with self.subTest(reason=reason, rows=rows):
                with self.assertRaises(GitHubProtectionError) as caught:
                    adapter_for(FakeGitHub(rows)).verify_candidate_ref(OBSERVATION_REF)
                self.assertEqual(caught.exception.reason_code, reason)

    def test_candidate_ref_pattern_ambiguity_and_bad_ref_are_rejected(self) -> None:
        rows = candidate_rulesets() + [
            ruleset(991, "overlap", include=["refs/heads/candidates/cyax*"], rules=[rule("update")])
        ]
        with self.assertRaisesRegex(GitHubProtectionError, "OVERLAP_AMBIGUOUS"):
            adapter_for(FakeGitHub(rows)).verify_candidate_ref(OBSERVATION_REF)
        with self.assertRaisesRegex(GitHubProtectionError, "CANDIDATE_REF_INVALID"):
            adapter_for(FakeGitHub()).verify_candidate_ref("refs/heads/candidates")

    def test_stale_and_cross_repository_snapshots_are_rejected(self) -> None:
        fake = FakeGitHub()
        fake.stale_detail = True
        with self.assertRaisesRegex(GitHubProtectionError, "SNAPSHOT_CHANGED"):
            adapter_for(fake).read_snapshot()
        first = adapter_for(FakeGitHub())
        evidence = first.verify_candidate_ref(OBSERVATION_REF)
        other = adapter_for(FakeGitHub())
        self.assertFalse(other.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository=REPOSITORY
        ))

    def test_live_protection_evidence_rejects_changed_snapshot_and_read_failure(self) -> None:
        fake = FakeGitHub()
        adapter = adapter_for(fake)
        evidence = adapter.verify_candidate_ref(OBSERVATION_REF)
        self.assertTrue(adapter.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository=REPOSITORY
        ))
        target = next(row for row in fake.rows if row["id"] == 23967983)
        target["updated_at"] = "2026-09-24T15:00:03Z"
        self.assertFalse(adapter.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository=REPOSITORY
        ))

        fake.fail_next = True
        self.assertFalse(adapter.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository=REPOSITORY
        ))

    def test_public_tag_protection_rejects_extra_conditions_and_parameters(self) -> None:
        cases = ("immutable-condition", "creation-condition", "immutable-parameters", "creation-parameters")
        for case in cases:
            with self.subTest(case=case):
                fake = FakeGitHub()
                fake.rows.extend(public_tag_rulesets())
                immutable = next(row for row in fake.rows if row["id"] == 23948086)
                creation = next(row for row in fake.rows if row["id"] == 23948090)
                if case == "immutable-condition":
                    immutable["conditions"]["repository_id"] = 42
                elif case == "creation-condition":
                    creation["conditions"]["repository_id"] = 42
                elif case == "immutable-parameters":
                    immutable["rules"][0]["parameters"] = {"unexpected": True}
                else:
                    creation["rules"][0]["parameters"] = {"unexpected": True}
                with self.assertRaisesRegex(
                    GitHubProtectionError, "PUBLIC_TAG_PROTECTION_INVALID"
                ):
                    adapter_for(fake).verify_public_tag("refs/tags/v0.3.0")

    def test_updated_at_preserves_fractional_provider_precision(self) -> None:
        self.assertEqual(
            GitHubProtectionAdapter._updated_at("2026-09-24T15:00:00.123Z"),
            "2026-09-24T15:00:00.123000Z",
        )
        first = candidate_rulesets()
        second = candidate_rulesets()
        first[0]["updated_at"] = "2026-09-24T15:00:00.123Z"
        second[0]["updated_at"] = "2026-09-24T15:00:00.124Z"
        self.assertNotEqual(
            adapter_for(FakeGitHub(first)).read_snapshot().snapshot_sha256,
            adapter_for(FakeGitHub(second)).read_snapshot().snapshot_sha256,
        )

    def test_api_errors_are_reason_coded_without_credentials(self) -> None:
        fake = FakeGitHub()
        fake.fail_next = True
        with self.assertRaises(GitHubProtectionError) as caught:
            adapter_for(fake).read_snapshot()
        self.assertEqual(caught.exception.reason_code, "GITHUB_API_REQUEST_FAILED")
        self.assertNotIn("private-token", str(caught.exception))
        self.assertNotIn("private-token", repr(caught.exception))
        self.assertNotIn("private-token", "".join(traceback.format_exception(caught.exception)))

    def test_read_only_rule_evaluation_binds_provider_and_snapshot_identity(self) -> None:
        fake = FakeGitHub()
        observation = adapter_for(fake).observe_rules_for_ref(OBSERVATION_REF)
        self.assertEqual(observation.ruleset_ids, (23967983, 23967991))
        self.assertEqual(
            observation.rule_types,
            ("creation", "deletion", "non_fast_forward", "update"),
        )
        self.assertEqual(observation.provider_etag, '"effective-rules"')
        self.assertEqual(observation.provider_request_id, "REQ-0125")
        self.assertEqual(len(observation.response_sha256), 64)
        self.assertIn("/repos/Julia-meets-String-Theory/CYAxiverse.jl/rules/branches/candidates/cyax-0125/gate-a-observation", [call[1] for call in fake.calls])

    def test_rule_evaluation_mismatch_and_wrong_namespace_fail_closed(self) -> None:
        fake = FakeGitHub()
        fake.evaluated_rules[0]["ruleset_id"] = 800
        with self.assertRaisesRegex(GitHubProtectionError, "EVALUATION_CANDIDATE_MISMATCH"):
            adapter_for(fake).observe_rules_for_ref(OBSERVATION_REF)
        with self.assertRaisesRegex(GitHubProtectionError, "EVALUATION_REF_INVALID"):
            adapter_for(FakeGitHub()).observe_rules_for_ref("refs/tags/v0.3.0")

    def test_freeze_controls_require_authorization_and_verify_write_readback(self) -> None:
        fake = FakeGitHub()
        calls = []
        adapter = adapter_for(
            fake,
            gate=lambda intent, action, ref: calls.append((action, ref)) or True,
        )
        result = adapter.freeze_main({"transaction": "test", "repository": REPOSITORY})
        lease = result["token"]
        self.assertEqual(result["sha"], SHA_A)
        self.assertEqual(result["version"], "0.2.0")
        self.assertTrue(adapter.is_live_freeze_lease(lease, ref="refs/heads/main"))
        self.assertEqual(next(row for row in fake.rows if row["id"] == 23948127)["enforcement"], "active")
        put_calls = [call for call in fake.calls if call[0] == "PUT"]
        self.assertEqual(len(put_calls), 1)
        self.assertEqual(set(put_calls[0][2]), {"name", "target", "enforcement", "conditions", "rules", "bypass_actors"})
        adapter.unfreeze_main(lease)
        self.assertFalse(adapter.is_live_freeze_lease(lease, ref="refs/heads/main"))
        self.assertEqual(next(row for row in fake.rows if row["id"] == 23948127)["enforcement"], "disabled")
        self.assertEqual(len([call for call in fake.calls if call[0] == "PUT"]), 2)
        self.assertEqual(calls, [("activate", "refs/heads/main"), ("deactivate", "refs/heads/main")])

    def test_freeze_failures_leave_control_active_or_do_not_write(self) -> None:
        no_gate = adapter_for(FakeGitHub())
        with self.assertRaisesRegex(GitHubProtectionError, "AUTHORIZATION_UNVERIFIED"):
            no_gate.freeze_line({"repository": REPOSITORY})
        fake = FakeGitHub()
        fake.change_main_after_write = True
        adapter = adapter_for(fake, gate=lambda *_: True)
        with self.assertRaisesRegex(GitHubProtectionError, "TARGET_MOVED"):
            adapter.freeze_main({"repository": REPOSITORY})
        self.assertEqual(next(row for row in fake.rows if row["id"] == 23948127)["enforcement"], "active")

    def test_uncertain_freeze_activation_is_read_back_and_never_retried(self) -> None:
        for mode, observed in (
            ("response-lost-after", "active"),
            ("response-lost-before", "disabled"),
            ("late-apply-after-readback", "disabled"),
            ("response-lost-readback-fails", "unreadable"),
        ):
            with self.subTest(mode=mode):
                fake = FakeGitHub()
                fake.freeze_write_mode = mode
                adapter = adapter_for(fake, gate=lambda *_: True)
                with self.assertRaisesRegex(
                    GitHubProtectionError, "FREEZE_WRITE_UNKNOWN"
                ) as raised:
                    adapter.freeze_line({"repository": REPOSITORY})
                self.assertEqual(
                    raised.exception.evidence["observed_enforcement"], observed
                )
                self.assertEqual(
                    len([call for call in fake.calls if call[0] == "PUT"]), 1
                )
                if mode == "late-apply-after-readback":
                    self.assertEqual(
                        next(row for row in fake.rows if row["id"] == 23948123)[
                            "enforcement"
                        ],
                        "active",
                    )
                with self.assertRaisesRegex(
                    GitHubProtectionError, "FREEZE_WRITE_UNRECONCILED"
                ):
                    adapter.freeze_line({"repository": REPOSITORY})
                self.assertEqual(
                    len([call for call in fake.calls if call[0] == "PUT"]), 1
                )

    def test_uncertain_freeze_release_does_not_issue_an_opposite_or_repeat_write(self) -> None:
        for mode, observed in (
            ("response-lost-after", "disabled"),
            ("response-lost-before", "active"),
            ("late-apply-after-readback", "active"),
        ):
            with self.subTest(mode=mode):
                fake = FakeGitHub()
                adapter = adapter_for(fake, gate=lambda *_: True)
                lease = adapter.freeze_main(
                    {"transaction": "release-unknown", "repository": REPOSITORY}
                )["token"]
                fake.freeze_write_mode = mode
                with self.assertRaisesRegex(
                    GitHubProtectionError, "FREEZE_WRITE_UNKNOWN"
                ) as raised:
                    adapter.unfreeze_main(lease)
                self.assertEqual(
                    raised.exception.evidence["observed_enforcement"], observed
                )
                put_count = len([call for call in fake.calls if call[0] == "PUT"])
                self.assertEqual(put_count, 2)
                with self.assertRaisesRegex(
                    GitHubProtectionError, "FREEZE_WRITE_UNRECONCILED"
                ):
                    adapter.unfreeze_main(lease)
                self.assertEqual(
                    len([call for call in fake.calls if call[0] == "PUT"]),
                    put_count,
                )

    def test_freeze_inflight_fence_serializes_activation_and_release_writes(self) -> None:
        class DelayedTransport:
            def __init__(self, fake, enforcement):
                self.fake = fake
                self.enforcement = enforcement
                self.started = threading.Event()
                self.resume = threading.Event()
                self.blocked = False

            def __call__(self, method, path, payload=None):
                if (
                    method == "PUT" and "/rulesets/" in path
                    and payload.get("enforcement") == self.enforcement
                    and not self.blocked
                ):
                    self.blocked = True
                    self.started.set()
                    if not self.resume.wait(5):
                        raise RuntimeError("timed out waiting to release fake write")
                return self.fake(method, path, payload)

        fake = FakeGitHub()
        transport = DelayedTransport(fake, "active")
        adapter = GitHubProtectionAdapter(
            REPOSITORY, token="private-token", transport=transport,
            authorization_gate=lambda *_: True,
        )
        activation_results = []

        def activate():
            try:
                activation_results.append(
                    adapter.freeze_main({"repository": REPOSITORY})
                )
            except Exception as error:
                activation_results.append(error)

        first_activation = threading.Thread(target=activate)
        first_activation.start()
        self.assertTrue(transport.started.wait(5))
        with self.assertRaisesRegex(
            GitHubProtectionError, "FREEZE_WRITE_UNRECONCILED"
        ):
            adapter.freeze_main({"repository": REPOSITORY})
        stale_disabled = adapter._freeze_ruleset(
            "refs/heads/main", adapter.read_snapshot()
        )
        with self.assertRaisesRegex(
            GitHubProtectionError, "FREEZE_WRITE_UNRECONCILED"
        ):
            adapter._set_enforcement("refs/heads/main", stale_disabled, "disabled")
        transport.resume.set()
        first_activation.join(5)
        self.assertFalse(first_activation.is_alive())
        self.assertEqual(len(activation_results), 1)
        self.assertNotIsInstance(activation_results[0], Exception)
        self.assertEqual(len([call for call in fake.calls if call[0] == "PUT"]), 1)

        lease = activation_results[0]["token"]
        fake.freeze_write_mode = "response-lost-after"
        delayed_release = DelayedTransport(fake, "disabled")
        adapter._transport = delayed_release
        release_results = []

        def release():
            try:
                adapter.unfreeze_main(lease)
                release_results.append(None)
            except Exception as error:
                release_results.append(error)

        first_release = threading.Thread(target=release)
        first_release.start()
        self.assertTrue(delayed_release.started.wait(5))
        with self.assertRaisesRegex(
            GitHubProtectionError, "FREEZE_WRITE_UNRECONCILED"
        ):
            adapter.unfreeze_main(lease)
        delayed_release.resume.set()
        first_release.join(5)
        self.assertFalse(first_release.is_alive())
        self.assertEqual(len(release_results), 1)
        self.assertIsInstance(release_results[0], GitHubProtectionError)
        self.assertEqual(release_results[0].reason_code, "FREEZE_WRITE_UNKNOWN")
        self.assertTrue(adapter.is_live_freeze_lease(lease, ref="refs/heads/main"))
        self.assertEqual(len([call for call in fake.calls if call[0] == "PUT"]), 2)

    def test_terminal_freeze_rejection_is_distinct_from_an_uncertain_write(self) -> None:
        fake = FakeGitHub()
        fake.freeze_write_mode = "terminal-rejection"
        adapter = adapter_for(fake, gate=lambda *_: True)
        with self.assertRaisesRegex(
            GitHubProtectionError, "FREEZE_WRITE_REJECTED"
        ) as raised:
            adapter.freeze_line({"repository": REPOSITORY})
        self.assertEqual(
            raised.exception.evidence["write_outcome"], "terminal_rejection"
        )
        self.assertEqual(
            raised.exception.evidence["observed_enforcement"], "disabled"
        )
        self.assertEqual(
            next(row for row in fake.rows if row["id"] == 23948123)["enforcement"],
            "disabled",
        )
        self.assertNotIn("refs/heads/vmm", adapter._unreconciled_freeze_writes)

    def test_freeze_rejects_wrong_rule_or_stale_lease_without_releasing(self) -> None:
        wrong = FakeGitHub()
        wrong_freeze = next(row for row in wrong.rows if row["id"] == 23948127)
        wrong_freeze["bypass_actors"] = [
            {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}
        ]
        adapter = adapter_for(wrong, gate=lambda *_: True)
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_RULESET_INVALID"):
            adapter.freeze_main({"repository": REPOSITORY})
        self.assertEqual(next(row for row in wrong.rows if row["id"] == 23948127)["enforcement"], "disabled")

        fake = FakeGitHub()
        adapter = adapter_for(fake, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        target = next(row for row in fake.rows if row["id"] == 23948127)
        target["updated_at"] = "2026-09-24T15:00:02Z"
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_RULESET_STALE"):
            adapter.unfreeze_main(lease)
        self.assertEqual(target["enforcement"], "active")

    def test_live_lease_check_reloads_enforcement_and_timestamp(self) -> None:
        fake = FakeGitHub()
        adapter = adapter_for(fake, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        self.assertTrue(adapter.verify_live_freeze_lease(lease, ref="refs/heads/main"))
        target = next(row for row in fake.rows if row["id"] == 23948127)
        target["enforcement"] = "disabled"
        target["updated_at"] = "2026-09-24T15:00:02Z"
        self.assertFalse(adapter.verify_live_freeze_lease(lease, ref="refs/heads/main"))

    def test_release_rejects_branch_move_before_write_and_restores_after_move(self) -> None:
        before_write = FakeGitHub()
        adapter = adapter_for(before_write, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        put_count = len([call for call in before_write.calls if call[0] == "PUT"])
        before_write.branch_shas["main"] = SHA_B
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_TARGET_MOVED"):
            adapter.unfreeze_main(lease)
        self.assertEqual(
            len([call for call in before_write.calls if call[0] == "PUT"]),
            put_count,
        )
        self.assertEqual(
            next(row for row in before_write.rows if row["id"] == 23948127)["enforcement"],
            "active",
        )

        after_write = FakeGitHub()
        adapter = adapter_for(after_write, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        after_write.move_main_after_disable = True
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_TARGET_MOVED"):
            adapter.unfreeze_main(lease)
        target = next(row for row in after_write.rows if row["id"] == 23948127)
        self.assertEqual(target["enforcement"], "active")
        self.assertEqual(after_write.branch_shas["main"], SHA_B)
        self.assertEqual(
            len([call for call in after_write.calls if call[0] == "PUT"]), 3,
            "activation, attempted release, then verified restoration",
        )


if __name__ == "__main__":
    unittest.main()
