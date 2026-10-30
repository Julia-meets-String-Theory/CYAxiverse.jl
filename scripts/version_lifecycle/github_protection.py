"""Authenticated GitHub ruleset and lifecycle-freeze adapter.

The adapter is intentionally narrow. It reads repository rulesets and effective
rules for an exact branch, and it can acquire or release the already-approved
``vmm`` and ``main`` freeze rulesets when an authorized lifecycle port asks it
to do so. It does not create refs or change lifecycle state.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
import uuid

from .codec import canonical_json, sha256_hex
from .git_refs import (
    GitIdentityError,
    ProtectionEvidence,
    require_candidate_ref,
    require_full_ref,
)


_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_PART = re.compile(r"[A-Za-z0-9_.-]+\Z")
_CANDIDATE_PATTERN = "refs/heads/candidates/**/*"
_CANDIDATE_IMMUTABLE_ID = 23967983
_CANDIDATE_CREATION_ID = 23967991
_CANDIDATE_IMMUTABLE_NAME = "CYAx candidate refs immutable"
_CANDIDATE_CREATION_NAME = "CYAx candidate refs creation"
_FREEZE_CONTROLS = {
    "refs/heads/vmm": (23948123, "CYAx vmm lifecycle freeze"),
    "refs/heads/main": (23948127, "CYAx main lifecycle freeze"),
}
_PUBLIC_TAG_IMMUTABLE_ID = 23948086
_PUBLIC_TAG_CREATION_ID = 23948090
_PUBLIC_TAG_PATTERN = "refs/tags/v*.*.*"
_FULL_REF_SHA = re.compile(r"[0-9a-f]{40}\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?\Z")
_APP_LOGIN = "cyaxiverse-lifecycle[bot]"
_APP_INSTALLATION_ID = 172789630


class GitHubProtectionError(RuntimeError):
    """A reason-coded, privacy-safe failure from the GitHub adapter."""

    def __init__(
        self,
        reason_code: str,
        *,
        provider_status: int | None = None,
        provider_request_id: str | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.provider_status = provider_status
        self.provider_request_id = provider_request_id
        self.evidence = dict(evidence or {})

    @property
    def provider_rejection_is_terminal(self) -> bool:
        return (
            self.provider_status is not None
            and 400 <= self.provider_status < 500
            and self.provider_status not in {408, 425, 429}
        )


@dataclass(frozen=True, slots=True)
class ApiResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


Transport = Callable[[str, str, Mapping[str, Any] | None], ApiResponse]


@dataclass(frozen=True, slots=True)
class CanonicalRuleset:
    id: int
    name: str
    target: str
    enforcement: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    conditions_extra: Mapping[str, Any]
    rules: tuple[Mapping[str, Any], ...]
    bypass_actors: tuple[Mapping[str, Any], ...]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        conditions: dict[str, Any] = {
            "ref_name": {
                "include": list(self.include),
                "exclude": list(self.exclude),
            }
        }
        conditions.update(self.conditions_extra)
        return {
            "id": self.id,
            "name": self.name,
            "target": self.target,
            "enforcement": self.enforcement,
            "conditions": conditions,
            "rules": [dict(rule) for rule in self.rules],
            "bypass_actors": [dict(actor) for actor in self.bypass_actors],
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class RulesetSnapshot:
    repository: str
    retrieved_at_utc: str
    rulesets: tuple[CanonicalRuleset, ...]
    canonical_bytes: bytes
    snapshot_sha256: str
    collection_etags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "retrieved_at_utc": self.retrieved_at_utc,
            "rulesets": [rule.to_dict() for rule in self.rulesets],
            "snapshot_sha256": self.snapshot_sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedProtectionEvidence:
    """Live evidence tied to one adapter, repository, snapshot, and ref."""

    protection: ProtectionEvidence
    repository: str
    target_ref: str
    ruleset_ids: tuple[int, ...]
    snapshot_sha256: str
    retrieved_at_utc: str
    _adapter_nonce: object = field(repr=False, compare=False)

    @property
    def rule_id(self) -> str:
        return self.protection.rule_id

    @property
    def pattern(self) -> str:
        return self.protection.pattern

    @property
    def creation_guarded(self) -> bool:
        return self.protection.creation_guarded

    @property
    def update_guarded(self) -> bool:
        return self.protection.update_guarded

    @property
    def deletion_guarded(self) -> bool:
        return self.protection.deletion_guarded

    def require(self, ref: str, *, creation: bool = False) -> None:
        if ref != self.target_ref:
            raise GitIdentityError("PROTECTION_EVIDENCE_TARGET_MISMATCH")
        self.protection.require(ref, creation=creation)

    def require_public_tag(self, ref: str) -> None:
        if ref != self.target_ref:
            raise GitIdentityError("PROTECTION_EVIDENCE_TARGET_MISMATCH")
        self.protection.require_public_tag(ref)


@dataclass(frozen=True, slots=True)
class RuleEvaluationObservation:
    repository: str
    ref: str
    retrieved_at_utc: str
    ruleset_snapshot_sha256: str
    response_sha256: str
    provider_etag: str | None
    provider_request_id: str | None
    ruleset_ids: tuple[int, ...]
    rule_types: tuple[str, ...]
    canonical_response: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "ref": self.ref,
            "retrieved_at_utc": self.retrieved_at_utc,
            "ruleset_snapshot_sha256": self.ruleset_snapshot_sha256,
            "response_sha256": self.response_sha256,
            "provider_etag": self.provider_etag,
            "provider_request_id": self.provider_request_id,
            "ruleset_ids": list(self.ruleset_ids),
            "rule_types": list(self.rule_types),
        }


@dataclass(frozen=True, slots=True)
class FreezeLease:
    """Opaque capability to release one verified freeze acquired here."""

    lease_id: str
    ref: str
    ruleset_id: int
    branch_sha: str
    version: str
    ruleset_updated_at: str
    snapshot_sha256: str
    retrieved_at_utc: str
    _adapter_nonce: object = field(repr=False, compare=False)
    _intent: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class OwnerRouteDecision:
    status: str
    reason_code: str
    commits_examined: tuple[str, ...] = ()
    workflow_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RehearsalTriggerRequest:
    request_id: str
    request_path: str
    commit_sha: str
    parent_sha: str
    workflow_blob_sha: str
    expected_vmm_sha: str
    called_code_blobs: Mapping[str, str]
    request_sha256: str
    request_bytes: bytes = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_path": self.request_path,
            "commit_sha": self.commit_sha,
            "parent_sha": self.parent_sha,
            "workflow_blob_sha": self.workflow_blob_sha,
            "expected_vmm_sha": self.expected_vmm_sha,
            "called_code_blobs": dict(self.called_code_blobs),
            "request_sha256": self.request_sha256,
        }


def _git_read(repository: str | Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise GitHubProtectionError("GIT_EVIDENCE_UNAVAILABLE")
    return result.stdout


def _request_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate request key")
        value[key] = item
    return value


def classify_principal_branch_move(
    repository: str | Path,
    *,
    ref: str,
    expected_old_sha: str,
    intended_new_sha: str,
) -> OwnerRouteDecision:
    """Classify every commit in an App rehearsal move before any push."""

    try:
        require_full_ref(ref)
    except (GitIdentityError, TypeError):
        return OwnerRouteDecision("REJECTED_BY_RULESET", "TARGET_OR_SHA_INVALID")
    if (
        not ref.startswith("refs/heads/rehearsal/principal/")
        or not ref.removeprefix("refs/heads/rehearsal/principal/")
        or any(part in {"", ".", ".."} for part in ref.split("/")[4:])
        or not _FULL_REF_SHA.fullmatch(expected_old_sha)
        or not _FULL_REF_SHA.fullmatch(intended_new_sha)
    ):
        return OwnerRouteDecision("REJECTED_BY_RULESET", "TARGET_OR_SHA_INVALID")
    if expected_old_sha == intended_new_sha:
        return OwnerRouteDecision("REJECTED_BY_RULESET", "REF_MOVE_IS_EMPTY")
    try:
        ancestor = subprocess.run(
            [
                "git", "-C", str(repository), "merge-base", "--is-ancestor",
                expected_old_sha, intended_new_sha,
            ],
            capture_output=True,
            check=False,
        )
        if ancestor.returncode == 1:
            return OwnerRouteDecision("REJECTED_BY_RULESET", "NON_FAST_FORWARD")
        if ancestor.returncode != 0:
            return OwnerRouteDecision("CAPABILITY_UNAVAILABLE", "ANCESTRY_UNPROVEN")
        commits = _git_read(
            repository, "rev-list", "--reverse", f"{expected_old_sha}..{intended_new_sha}"
        ).decode("ascii").splitlines()
        workflow_paths: set[str] = set()
        for commit in commits:
            parents = _git_read(repository, "show", "-s", "--format=%P", commit).decode(
                "ascii"
            ).split()
            comparisons = (
                [(parents[index], commit) for index in range(len(parents))]
                if parents
                else [(commit, commit)]
            )
            for before, after in comparisons:
                command = [
                    "git", "-C", str(repository), "diff-tree", "--no-commit-id",
                    "--name-only", "-r", "--no-renames",
                ]
                if before == after:
                    command.append("--root")
                    command.append(after)
                else:
                    command.extend((before, after))
                command.extend(("--", ".github/workflows"))
                result = subprocess.run(
                    command, capture_output=True, check=False
                )
                if result.returncode:
                    return OwnerRouteDecision(
                        "OWNER_ROUTE_REQUIRED", "WORKFLOW_RANGE_UNPROVEN",
                        tuple(commits), tuple(sorted(workflow_paths)),
                    )
                for path in result.stdout.decode("utf-8").splitlines():
                    if path.startswith(".github/workflows/"):
                        workflow_paths.add(path)
    except (UnicodeError, GitHubProtectionError, OSError):
        return OwnerRouteDecision("OWNER_ROUTE_REQUIRED", "WORKFLOW_RANGE_UNPROVEN")
    ordered_commits = tuple(commits)
    ordered_paths = tuple(sorted(workflow_paths))
    if ordered_paths:
        return OwnerRouteDecision(
            "OWNER_ROUTE_REQUIRED", "WORKFLOW_PATH_CHANGED",
            ordered_commits, ordered_paths,
        )
    return OwnerRouteDecision("APP_CAS_ELIGIBLE", "NO_WORKFLOW_CHANGE", ordered_commits)


def classify_ref_creation(ref: str, *, workflow_content_rejected: bool = False) -> OwnerRouteDecision:
    """Classify the narrow rehearsal create-once namespace."""

    try:
        require_full_ref(ref)
    except (GitIdentityError, TypeError):
        return OwnerRouteDecision("REJECTED_BY_RULESET", "REF_OUTSIDE_REHEARSAL_ALLOWLIST")
    if re.fullmatch(r"refs/tags/v\d+\.\d+\.\d+", ref):
        return OwnerRouteDecision("OWNER_ROUTE_REQUIRED", "CANONICAL_PUBLIC_TAG")
    if re.fullmatch(r"refs/tags/rehearsal/v\d+\.\d+\.\d+", ref):
        return OwnerRouteDecision("REJECTED_BY_RULESET", "REHEARSAL_CANONICAL_TAG")
    allowed = (
        ref.startswith("refs/heads/rehearsal/lifecycle/")
        or re.fullmatch(r"refs/tags/rehearsal/iterations/[^/]+", ref) is not None
    )
    if not allowed or any(part in {"", ".", ".."} for part in ref.split("/")):
        return OwnerRouteDecision("REJECTED_BY_RULESET", "REF_OUTSIDE_REHEARSAL_ALLOWLIST")
    if workflow_content_rejected:
        return OwnerRouteDecision("OWNER_ROUTE_REQUIRED", "WORKFLOW_CONTENT_CREATION_REJECTED")
    return OwnerRouteDecision("APP_CREATE_ELIGIBLE", "CREATE_ONCE_REHEARSAL_REF")


def validate_rehearsal_trigger(
    repository_path: str | Path,
    *,
    repository: str,
    ref: str,
    commit_sha: str,
    actor: str,
    event: Mapping[str, Any],
) -> RehearsalTriggerRequest:
    """Verify the exact Owner-pushed one-file trigger before minting an App token."""

    if (
        not isinstance(repository, str)
        or not repository
        or not isinstance(ref, str)
        or not ref.startswith("refs/heads/rehearsal/trigger/")
        or any(part in {"", ".", ".."} for part in ref.removeprefix("refs/heads/").split("/"))
        or not isinstance(commit_sha, str)
        or _FULL_REF_SHA.fullmatch(commit_sha) is None
        or not isinstance(actor, str)
        or not actor
    ):
        raise GitHubProtectionError("TRIGGER_IDENTITY_INVALID")
    try:
        require_full_ref(ref)
    except (GitIdentityError, TypeError):
        raise GitHubProtectionError("TRIGGER_IDENTITY_INVALID") from None
    try:
        if (
            event.get("ref") != ref
            or event.get("after") != commit_sha
            or event.get("repository", {}).get("full_name") != repository
            or event.get("created") is not True
            or event.get("deleted") is not False
            or event.get("forced") is not False
        ):
            raise GitHubProtectionError("TRIGGER_IDENTITY_INVALID")
        sender = event.get("sender")
        installation = event.get("installation")
        if not isinstance(sender, Mapping) or (
            sender.get("type") != "User"
            or sender.get("login") != actor
            or sender.get("login") == _APP_LOGIN
            or (
                isinstance(installation, Mapping)
                and installation.get("id") == _APP_INSTALLATION_ID
            )
        ):
            raise GitHubProtectionError("TRIGGER_ACTOR_INVALID")
        row = _git_read(
            repository_path, "show", "-s", "--format=%H %P", commit_sha
        ).decode("ascii").strip().split()
        if len(row) != 2 or row[0] != commit_sha or _FULL_REF_SHA.fullmatch(row[1]) is None:
            raise GitHubProtectionError("TRIGGER_PARENT_INVALID")
        parent_sha = row[1]
        before = event.get("before")
        if before != "0" * 40:
            raise GitHubProtectionError("TRIGGER_PUSH_RANGE_INVALID")
        changes = _git_read(
            repository_path,
            "diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames",
            parent_sha, commit_sha,
        ).decode("utf-8").splitlines()
        if len(changes) != 1 or not changes[0].startswith("A\t"):
            raise GitHubProtectionError("TRIGGER_DIFF_INVALID")
        request_path = changes[0][2:]
        if not request_path.startswith("lifecycle-requests/") or "/" in request_path.removeprefix(
            "lifecycle-requests/"
        ):
            raise GitHubProtectionError("TRIGGER_DIFF_INVALID")
        mode_rows = _git_read(
            repository_path, "ls-tree", "-z", commit_sha, "--", request_path
        ).decode("utf-8").split("\0")
        entries = [item for item in mode_rows if item]
        if len(entries) != 1:
            raise GitHubProtectionError("TRIGGER_REQUEST_INVALID")
        metadata, actual_path = entries[0].split("\t", 1)
        mode, object_type, _ = metadata.split()
        if mode != "100644" or object_type != "blob" or actual_path != request_path:
            raise GitHubProtectionError("TRIGGER_REQUEST_INVALID")
        try:
            request_bytes = _git_read(repository_path, "show", f"{commit_sha}:{request_path}")
            if not request_bytes or len(request_bytes) > 65536 or request_bytes.endswith(b"\n"):
                raise ValueError
            request = json.loads(
                request_bytes.decode("utf-8"), object_pairs_hook=_request_object_pairs
            )
            if not isinstance(request, dict):
                raise ValueError
            if canonical_json(request) != request_bytes:
                raise ValueError
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise GitHubProtectionError("TRIGGER_REQUEST_INVALID") from None
        fields = {
            "schema_version", "request_id", "implementation_sha",
            "workflow_blob_sha", "expected_vmm_sha", "called_code_blobs", "purpose",
        }
        if set(request) != fields or type(request.get("schema_version")) is not int or request[
            "schema_version"
        ] != 1:
            raise GitHubProtectionError("TRIGGER_REQUEST_INVALID")
        request_id = request.get("request_id")
        if (
            not isinstance(request_id, str)
            or _REQUEST_ID.fullmatch(request_id) is None
            or request_path != f"lifecycle-requests/{request_id}.json"
            or ref != f"refs/heads/rehearsal/trigger/{request_id}"
            or request.get("implementation_sha") != parent_sha
            or request.get("purpose") != "gate-a-rehearsal"
        ):
            raise GitHubProtectionError("TRIGGER_REQUEST_INVALID")
        expected_vmm_sha = request.get("expected_vmm_sha")
        if (
            not isinstance(expected_vmm_sha, str)
            or _FULL_REF_SHA.fullmatch(expected_vmm_sha) is None
        ):
            raise GitHubProtectionError("TRIGGER_REQUEST_INVALID")
        workflow_path = ".github/workflows/cyax-lifecycle-dispatch.yml"
        workflow_blob = _git_read(
            repository_path, "rev-parse", f"{parent_sha}:{workflow_path}"
        ).decode("ascii").strip()
        current_workflow_blob = _git_read(
            repository_path, "rev-parse", f"{commit_sha}:{workflow_path}"
        ).decode("ascii").strip()
        if (
            _FULL_REF_SHA.fullmatch(workflow_blob) is None
            or current_workflow_blob != workflow_blob
            or request.get("workflow_blob_sha") != workflow_blob
        ):
            raise GitHubProtectionError("TRIGGER_WORKFLOW_IDENTITY_INVALID")
        code_paths = _git_read(
            repository_path, "ls-tree", "-r", "--name-only", parent_sha,
            "--", "scripts/version_lifecycle",
        ).decode("utf-8").splitlines()
        called_code_blobs: dict[str, str] = {}
        for path in code_paths:
            if not path.endswith(".py"):
                continue
            parent_blob = _git_read(
                repository_path, "rev-parse", f"{parent_sha}:{path}"
            ).decode("ascii").strip()
            current_blob = _git_read(
                repository_path, "rev-parse", f"{commit_sha}:{path}"
            ).decode("ascii").strip()
            if (
                _FULL_REF_SHA.fullmatch(parent_blob) is None
                or current_blob != parent_blob
            ):
                raise GitHubProtectionError("TRIGGER_CALLED_CODE_CHANGED")
            called_code_blobs[path] = parent_blob
        if request.get("called_code_blobs") != called_code_blobs:
            raise GitHubProtectionError("TRIGGER_CALLED_CODE_IDENTITY_INVALID")
        return RehearsalTriggerRequest(
            request_id,
            request_path,
            commit_sha,
            parent_sha,
            workflow_blob,
            expected_vmm_sha,
            called_code_blobs,
            sha256_hex(request_bytes),
            request_bytes,
        )
    except GitHubProtectionError:
        raise
    except (UnicodeError, OSError, ValueError, TypeError):
        raise GitHubProtectionError("TRIGGER_EVIDENCE_UNAVAILABLE") from None


class GitHubProtectionAdapter:
    """Read and manage the governed repository rulesets through GitHub REST."""

    api_version = "2026-03-10"

    def __init__(
        self,
        repository: str,
        *,
        token: str | None = None,
        api_base: str = "https://api.github.com",
        transport: Transport | None = None,
        clock: Callable[[], datetime] | None = None,
        authorization_gate: Callable[[Any, str, str], bool] | None = None,
        rehearsal_gate: Callable[[Any, str, str, Mapping[str, Any]], bool] | None = None,
    ) -> None:
        parts = repository.split("/") if isinstance(repository, str) else []
        if len(parts) != 2 or any(_REPOSITORY_PART.fullmatch(part) is None for part in parts):
            raise GitHubProtectionError("REPOSITORY_IDENTITY_INVALID")
        if not isinstance(api_base, str):
            raise GitHubProtectionError("API_ORIGIN_INVALID")
        parsed = urlsplit(api_base)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise GitHubProtectionError("API_ORIGIN_INVALID")
        if transport is None:
            try:
                github_api_origin = (
                    parsed.hostname == "api.github.com"
                    and parsed.port in {None, 443}
                    and parsed.path in {"", "/"}
                )
            except ValueError:
                github_api_origin = False
            if not github_api_origin:
                raise GitHubProtectionError("API_ORIGIN_INVALID")
        self.repository = "/".join(parts)
        self.owner, self.repo = parts
        self.api_base = (
            "https://api.github.com" if transport is None else api_base.rstrip("/")
        )
        self._token = token if token is not None else (
            os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        )
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._authorization_gate = authorization_gate
        self._rehearsal_gate = rehearsal_gate
        self._nonce = object()
        self._leases: dict[str, FreezeLease] = {}
        self._unreconciled_freeze_writes: dict[str, dict[str, Any]] = {}
        self._unreconciled_branch_moves: dict[str, dict[str, Any]] = {}
        self._unreconciled_request_fences: dict[str, dict[str, Any]] = {}

    def _timestamp(self) -> str:
        try:
            value = self._clock()
            if value.tzinfo is None:
                raise ValueError
            return value.astimezone(timezone.utc).replace(microsecond=0).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except Exception as error:
            raise GitHubProtectionError("CLOCK_UNAVAILABLE") from error

    def _api_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        allow_not_found: bool = False,
    ) -> ApiResponse:
        if not self._token:
            raise GitHubProtectionError("GITHUB_AUTHENTICATION_UNAVAILABLE")
        if not path.startswith("/") or ".." in path.split("/"):
            raise GitHubProtectionError("API_PATH_INVALID")
        if self._transport is not None:
            try:
                response = self._transport(method, path, payload)
            except GitHubProtectionError:
                raise
            except Exception:
                # Transport implementations can include request headers in an
                # exception message. Keep the public failure reason private.
                raise GitHubProtectionError(
                    "GITHUB_API_REQUEST_FAILED", evidence={"outcome": "unknown"}
                ) from None
        else:
            raw = None if payload is None else json.dumps(
                payload, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            request = Request(
                self.api_base + path,
                data=raw,
                method=method,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {self._token}",
                    "X-GitHub-Api-Version": self.api_version,
                    "Content-Type": "application/json",
                    "User-Agent": "CYAxiverse-lifecycle-control",
                },
            )
            try:
                with urlopen(request, timeout=20) as result:
                    response = ApiResponse(
                        result.status,
                        result.read(),
                        {key.lower(): value for key, value in result.headers.items()},
                    )
            except HTTPError as error:
                status = error.code
                headers = {key.lower(): value for key, value in error.headers.items()}
                error.close()
                if allow_not_found and status == 404:
                    return ApiResponse(status, b"", headers)
                raise GitHubProtectionError(
                    "GITHUB_API_REQUEST_FAILED",
                    provider_status=status,
                    provider_request_id=headers.get("x-github-request-id"),
                    evidence={
                        "outcome": (
                            "terminal_rejection"
                            if 400 <= status < 500 and status not in {408, 425, 429}
                            else "unknown"
                        )
                    },
                ) from None
            except (URLError, TimeoutError, OSError, ValueError):
                raise GitHubProtectionError(
                    "GITHUB_API_REQUEST_FAILED", evidence={"outcome": "unknown"}
                ) from None
        if not isinstance(response, ApiResponse):
            raise GitHubProtectionError(
                "GITHUB_API_REQUEST_FAILED", evidence={"outcome": "unknown"}
            )
        normalized_headers = {
            str(key).lower(): str(value) for key, value in response.headers.items()
        }
        if allow_not_found and response.status == 404:
            return ApiResponse(response.status, response.body, normalized_headers)
        if not 200 <= response.status < 300:
            raise GitHubProtectionError(
                "GITHUB_API_REQUEST_FAILED",
                provider_status=response.status,
                provider_request_id=normalized_headers.get("x-github-request-id"),
                evidence={
                    "outcome": (
                        "terminal_rejection"
                        if 400 <= response.status < 500
                        and response.status not in {408, 425, 429}
                        else "unknown"
                    )
                },
            )
        return ApiResponse(response.status, response.body, normalized_headers)

    def _repo_path(self, suffix: str) -> str:
        return f"/repos/{quote(self.owner, safe='')}/{quote(self.repo, safe='')}{suffix}"

    def _read_ref(self, branch: str) -> tuple[str, str] | None:
        path = self._repo_path(f"/git/ref/heads/{quote(branch, safe='/')}")
        response = self._api_request("GET", path, allow_not_found=True)
        if response.status == 404:
            return None
        payload = self._json(response)
        try:
            ref = payload["ref"]
            object_type = payload["object"]["type"]
            object_sha = payload["object"]["sha"]
        except (KeyError, TypeError):
            raise GitHubProtectionError("GITHUB_REF_IDENTITY_INVALID") from None
        if (
            ref != f"refs/heads/{branch}"
            or object_type != "commit"
            or not isinstance(object_sha, str)
            or _FULL_REF_SHA.fullmatch(object_sha) is None
        ):
            raise GitHubProtectionError("GITHUB_REF_IDENTITY_INVALID")
        return object_type, object_sha

    def _vmm_head_and_tree(self, expected_sha: str) -> tuple[str, str]:
        if _FULL_REF_SHA.fullmatch(expected_sha) is None:
            raise GitHubProtectionError("REQUEST_VMM_IDENTITY_INVALID")
        response = self._api_request(
            "GET", self._repo_path("/branches/vmm")
        )
        payload = self._json(response)
        try:
            name = payload["name"]
            commit = payload["commit"]["sha"]
            tree = payload["commit"]["commit"]["tree"]["sha"]
        except (KeyError, TypeError):
            raise GitHubProtectionError("REQUEST_VMM_IDENTITY_INVALID") from None
        if (
            name != "vmm"
            or commit != expected_sha
            or not isinstance(tree, str)
            or _FULL_REF_SHA.fullmatch(tree) is None
        ):
            raise GitHubProtectionError("REQUEST_VMM_HEAD_CHANGED")
        return commit, tree

    def _read_request_file(self, path: str, ref: str) -> bool:
        encoded = quote(path, safe="/")
        query = urlencode({"ref": ref})
        response = self._api_request(
            "GET", self._repo_path(f"/contents/{encoded}?{query}"),
            allow_not_found=True,
        )
        return response.status != 404

    def _read_rehearsal_fence(
        self, branch: str
    ) -> tuple[str, str] | None:
        path = self._repo_path(f"/git/ref/heads/{quote(branch, safe='/')}")
        response = self._api_request("GET", path, allow_not_found=True)
        if response.status == 404:
            return None
        payload = self._json(response)
        try:
            ref = payload["ref"]
            object_type = payload["object"]["type"]
            object_sha = payload["object"]["sha"]
        except (KeyError, TypeError):
            raise GitHubProtectionError("GITHUB_REF_IDENTITY_INVALID") from None
        if (
            ref != f"refs/heads/{branch}"
            or object_type != "commit"
            or not isinstance(object_sha, str)
            or _FULL_REF_SHA.fullmatch(object_sha) is None
        ):
            raise GitHubProtectionError("GITHUB_REF_IDENTITY_INVALID")
        return object_type, object_sha

    def create_rehearsal_request_fence(
        self,
        request: RehearsalTriggerRequest,
        *,
        intent: Any,
    ) -> dict[str, Any]:
        """Create one immutable rehearsal request fence on the current vmm tree.

        The marker commit is based on the exact observed vmm head and adds only
        the already-validated request file. It never moves or creates vmm.
        """

        if not isinstance(request, RehearsalTriggerRequest):
            raise GitHubProtectionError("REQUEST_FENCE_INPUT_INVALID")
        if (
            _REQUEST_ID.fullmatch(request.request_id) is None
            or request.request_path != f"lifecycle-requests/{request.request_id}.json"
            or _FULL_REF_SHA.fullmatch(request.commit_sha) is None
            or _FULL_REF_SHA.fullmatch(request.parent_sha) is None
            or _FULL_REF_SHA.fullmatch(request.workflow_blob_sha) is None
            or _FULL_REF_SHA.fullmatch(request.expected_vmm_sha) is None
            or not isinstance(request.request_bytes, bytes)
            or not re.fullmatch(r"[0-9a-f]{64}\Z", request.request_sha256)
            or sha256_hex(request.request_bytes) != request.request_sha256
        ):
            raise GitHubProtectionError("REQUEST_FENCE_INPUT_INVALID")
        try:
            payload = json.loads(
                request.request_bytes.decode("utf-8"),
                object_pairs_hook=_request_object_pairs,
            )
            if (
                not isinstance(payload, dict)
                or set(payload) != {
                    "schema_version", "request_id", "implementation_sha",
                    "workflow_blob_sha", "expected_vmm_sha", "called_code_blobs",
                    "purpose",
                }
                or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") != 1
                or canonical_json(payload) != request.request_bytes
                or payload.get("request_id") != request.request_id
                or payload.get("implementation_sha") != request.parent_sha
                or payload.get("workflow_blob_sha") != request.workflow_blob_sha
                or payload.get("expected_vmm_sha") != request.expected_vmm_sha
                or payload.get("called_code_blobs") != dict(request.called_code_blobs)
                or payload.get("purpose") != "gate-a-rehearsal"
            ):
                raise ValueError
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise GitHubProtectionError("REQUEST_FENCE_INPUT_INVALID") from None
        fence_ref = f"refs/heads/rehearsal/lifecycle/requests/{request.request_id}"
        if request.request_id in self._unreconciled_request_fences:
            raise GitHubProtectionError(
                "REQUEST_FENCE_WRITE_UNRECONCILED",
                evidence=self._unreconciled_request_fences[request.request_id],
            )
        decision = classify_ref_creation(fence_ref)
        if decision.status != "APP_CREATE_ELIGIBLE":
            raise GitHubProtectionError(decision.reason_code)
        intent_repository = (
            intent.get("repository") if isinstance(intent, Mapping)
            else getattr(intent, "repository", None)
        )
        if intent_repository != self.repository:
            raise GitHubProtectionError("REHEARSAL_REPOSITORY_MISMATCH")
        intent_request_id = (
            intent.get("request_id") if isinstance(intent, Mapping)
            else getattr(intent, "request_id", None)
        )
        if intent_request_id != request.request_id:
            raise GitHubProtectionError("REQUEST_FENCE_IDENTITY_MISMATCH")
        expected_vmm, base_tree = self._vmm_head_and_tree(request.expected_vmm_sha)
        if self._read_request_file(request.request_path, "vmm"):
            raise GitHubProtectionError("REQUEST_ID_ALREADY_USED")
        fence_branch = fence_ref.removeprefix("refs/heads/")
        if self._read_rehearsal_fence(fence_branch) is not None:
            raise GitHubProtectionError("REQUEST_ID_ALREADY_USED")
        rules_evidence = self._require_rehearsal_rules(
            fence_ref, {"creation", "deletion", "non_fast_forward", "update"}
        )
        if self._rehearsal_gate is None:
            raise GitHubProtectionError("REQUEST_FENCE_AUTHORIZATION_UNVERIFIED")
        try:
            authorized = self._rehearsal_gate(
                intent, "create-request-fence", fence_ref, rules_evidence
            )
        except Exception:
            authorized = False
        if authorized is not True:
            raise GitHubProtectionError("REQUEST_FENCE_AUTHORIZATION_UNVERIFIED")

        def create_object(path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
            try:
                response = self._api_request("POST", self._repo_path(path), payload)
                value = self._json(response)
                if (
                    not isinstance(value, dict)
                    or not isinstance(value.get("sha"), str)
                    or _FULL_REF_SHA.fullmatch(value["sha"]) is None
                ):
                    raise GitHubProtectionError(
                        "REQUEST_FENCE_OBJECT_RESPONSE_INVALID"
                    )
                return value
            except GitHubProtectionError as error:
                if error.provider_rejection_is_terminal:
                    raise GitHubProtectionError(
                        "REQUEST_FENCE_OBJECT_REJECTED",
                        provider_status=error.provider_status,
                        provider_request_id=error.provider_request_id,
                    ) from None
                evidence = {
                    "request_id": request.request_id,
                    "target_ref": fence_ref,
                    "object_path": path,
                    "write_outcome": "unknown",
                    "provider_status": error.provider_status,
                    "provider_request_id": error.provider_request_id,
                }
                self._unreconciled_request_fences[request.request_id] = evidence
                raise GitHubProtectionError(
                    "REQUEST_FENCE_OBJECT_CREATION_UNKNOWN",
                    provider_status=error.provider_status,
                    provider_request_id=error.provider_request_id,
                    evidence=evidence,
                ) from None

        blob = create_object(
            "/git/blobs",
            {
                "content": base64.b64encode(request.request_bytes).decode("ascii"),
                "encoding": "base64",
            },
        )
        blob_sha = blob.get("sha")
        if not isinstance(blob_sha, str) or _FULL_REF_SHA.fullmatch(blob_sha) is None:
            raise GitHubProtectionError("REQUEST_FENCE_OBJECT_INVALID")
        tree = create_object(
            "/git/trees",
            {
                "base_tree": base_tree,
                "tree": [{
                    "path": request.request_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha,
                }],
            },
        )
        tree_sha = tree.get("sha")
        if not isinstance(tree_sha, str) or _FULL_REF_SHA.fullmatch(tree_sha) is None:
            raise GitHubProtectionError("REQUEST_FENCE_OBJECT_INVALID")
        commit = create_object(
            "/git/commits",
            {
                "message": f"CYAx rehearsal request fence {request.request_id}",
                "tree": tree_sha,
                "parents": [expected_vmm],
            },
        )
        commit_sha = commit.get("sha")
        if not isinstance(commit_sha, str) or _FULL_REF_SHA.fullmatch(commit_sha) is None:
            raise GitHubProtectionError("REQUEST_FENCE_OBJECT_INVALID")
        verified_commit = self._json(
            self._api_request(
                "GET", self._repo_path(f"/git/commits/{commit_sha}")
            )
        )
        try:
            observed_commit_sha = verified_commit["sha"]
            commit_tree = verified_commit["tree"]["sha"]
            parents = [parent["sha"] for parent in verified_commit["parents"]]
        except (KeyError, TypeError):
            raise GitHubProtectionError("REQUEST_FENCE_COMMIT_INVALID") from None
        if (
            observed_commit_sha != commit_sha
            or commit_tree != tree_sha
            or parents != [expected_vmm]
        ):
            raise GitHubProtectionError("REQUEST_FENCE_COMMIT_INVALID")
        current_vmm, current_tree = self._vmm_head_and_tree(expected_vmm)
        if current_vmm != expected_vmm or current_tree != base_tree:
            raise GitHubProtectionError("REQUEST_VMM_HEAD_CHANGED")
        if self._read_rehearsal_fence(fence_branch) is not None:
            raise GitHubProtectionError("REQUEST_ID_ALREADY_USED")

        attempt = {
            "write_id": uuid.uuid4().hex,
            "request_id": request.request_id,
            "target_ref": fence_ref,
            "intended_commit_sha": commit_sha,
            "expected_vmm_sha": expected_vmm,
            "started_at_utc": self._timestamp(),
        }
        try:
            response = self._api_request(
                "POST",
                self._repo_path("/git/refs"),
                {"ref": fence_ref, "sha": commit_sha},
            )
        except GitHubProtectionError as error:
            try:
                observed = self._read_rehearsal_fence(fence_branch)
                observed_sha = observed[1] if observed else None
                readback_error = None
            except Exception:
                observed_sha = None
                readback_error = "REQUEST_FENCE_READBACK_UNAVAILABLE"
            evidence = {
                **attempt,
                "write_outcome": (
                    "terminal_rejection"
                    if error.provider_rejection_is_terminal
                    else "unknown"
                ),
                "provider_status": error.provider_status,
                "provider_request_id": error.provider_request_id,
                "observed_commit_sha": observed_sha,
                "readback_error": readback_error,
            }
            if error.provider_rejection_is_terminal:
                if observed_sha is None and readback_error is None:
                    raise GitHubProtectionError(
                        "REQUEST_FENCE_CREATE_REJECTED",
                        provider_status=error.provider_status,
                        provider_request_id=error.provider_request_id,
                        evidence=evidence,
                    ) from None
                raise GitHubProtectionError(
                    "REQUEST_ID_ALREADY_USED",
                    provider_status=error.provider_status,
                    provider_request_id=error.provider_request_id,
                    evidence=evidence,
                ) from None
            self._unreconciled_request_fences[request.request_id] = evidence
            raise GitHubProtectionError(
                "REQUEST_FENCE_WRITE_UNKNOWN",
                provider_status=error.provider_status,
                provider_request_id=error.provider_request_id,
                evidence=evidence,
            ) from None
        try:
            observed = self._read_rehearsal_fence(fence_branch)
        except Exception:
            observed = None
        observed_sha = observed[1] if observed else None
        evidence = {
            **attempt,
            "write_outcome": "provider_accepted",
            "provider_status": response.status,
            "provider_request_id": response.headers.get("x-github-request-id"),
            "observed_commit_sha": observed_sha,
        }
        if observed_sha != commit_sha:
            evidence["write_outcome"] = "unknown"
            self._unreconciled_request_fences[request.request_id] = evidence
            raise GitHubProtectionError(
                "REQUEST_FENCE_WRITE_UNKNOWN",
                provider_status=response.status,
                provider_request_id=response.headers.get("x-github-request-id"),
                evidence=evidence,
            )
        return evidence

    def _rehearsal_branch_head(self, ref: str) -> str:
        branch = ref.removeprefix("refs/heads/")
        response = self._api_request(
            "GET",
            self._repo_path(f"/branches/{quote(branch, safe='')}"),
        )
        payload = self._json(response)
        try:
            name = payload["name"]
            sha = payload["commit"]["sha"]
        except (KeyError, TypeError):
            raise GitHubProtectionError("REHEARSAL_BRANCH_IDENTITY_INVALID") from None
        if name != branch or not isinstance(sha, str) or _FULL_REF_SHA.fullmatch(sha) is None:
            raise GitHubProtectionError("REHEARSAL_BRANCH_IDENTITY_INVALID")
        return sha

    def compare_and_swap_rehearsal_principal(
        self,
        repository_path: str | Path,
        *,
        ref: str,
        expected_old_sha: str,
        intended_new_sha: str,
        intended_new_tree: str,
        intent: Any,
    ) -> dict[str, Any]:
        """Fast-forward one rehearsal principal ref with an exact-old lease."""

        if ref in self._unreconciled_branch_moves:
            raise GitHubProtectionError(
                "REF_MOVE_UNRECONCILED",
                evidence=self._unreconciled_branch_moves[ref],
            )
        decision = classify_principal_branch_move(
            repository_path,
            ref=ref,
            expected_old_sha=expected_old_sha,
            intended_new_sha=intended_new_sha,
        )
        if decision.status == "OWNER_ROUTE_REQUIRED":
            raise GitHubProtectionError(
                "OWNER_ROUTE_REQUIRED",
                evidence={
                    "reason_code": decision.reason_code,
                    "commits_examined": list(decision.commits_examined),
                    "workflow_paths": list(decision.workflow_paths),
                },
            )
        if decision.status != "APP_CAS_ELIGIBLE":
            raise GitHubProtectionError(decision.reason_code)
        if _FULL_REF_SHA.fullmatch(intended_new_tree) is None:
            raise GitHubProtectionError("CAS_TARGET_TREE_INVALID")
        try:
            actual_tree = _git_read(
                repository_path, "rev-parse", f"{intended_new_sha}^{{tree}}"
            ).decode("ascii").strip()
        except GitHubProtectionError:
            raise GitHubProtectionError("CAS_TARGET_TREE_INVALID") from None
        if actual_tree != intended_new_tree:
            raise GitHubProtectionError("CAS_TARGET_TREE_MISMATCH")
        intent_repository = (
            intent.get("repository") if isinstance(intent, Mapping)
            else getattr(intent, "repository", None)
        )
        if intent_repository != self.repository:
            raise GitHubProtectionError("REHEARSAL_REPOSITORY_MISMATCH")
        if self._authorization_gate is None:
            raise GitHubProtectionError("REHEARSAL_AUTHORIZATION_UNVERIFIED")
        try:
            authorized = self._authorization_gate(
                intent, "move-rehearsal-principal", ref
            )
        except Exception:
            authorized = False
        if authorized is not True:
            raise GitHubProtectionError("REHEARSAL_AUTHORIZATION_UNVERIFIED")
        if self._rehearsal_gate is None:
            raise GitHubProtectionError("REHEARSAL_RULESET_UNVERIFIED")
        rules_evidence = self._require_rehearsal_rules(
            ref, {"deletion", "non_fast_forward", "update"}
        )
        try:
            rules_verified = self._rehearsal_gate(
                intent, "move-rehearsal-principal", ref, rules_evidence
            )
        except Exception:
            rules_verified = False
        if rules_verified is not True:
            raise GitHubProtectionError("REHEARSAL_RULESET_UNVERIFIED")
        if self._rehearsal_branch_head(ref) != expected_old_sha:
            raise GitHubProtectionError("CAS_STALE_EXPECTED_OLD")
        attempt = {
            "write_id": uuid.uuid4().hex,
            "target_ref": ref,
            "expected_old_sha": expected_old_sha,
            "intended_new_sha": intended_new_sha,
            "intended_new_tree": intended_new_tree,
            "request_id": (
                intent.get("transaction_id") if isinstance(intent, Mapping)
                else getattr(intent, "transaction_id", None)
            ),
            "started_at_utc": self._timestamp(),
        }
        if not self._token:
            raise GitHubProtectionError("GITHUB_AUTHENTICATION_UNAVAILABLE")
        credential = base64.b64encode(
            f"x-access-token:{self._token}".encode("utf-8")
        ).decode("ascii")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"GH_TOKEN", "GITHUB_TOKEN", "GIT_DIR", "GIT_WORK_TREE"}
            and not key.startswith("GIT_CONFIG_")
        }
        environment.update({
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_COUNT": "4",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "",
            "GIT_CONFIG_KEY_1": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_1": "",
            "GIT_CONFIG_KEY_2": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_2": f"AUTHORIZATION: basic {credential}",
            "GIT_CONFIG_KEY_3": "core.hooksPath",
            "GIT_CONFIG_VALUE_3": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        })
        result = subprocess.run(
            [
                "git", "-C", str(repository_path), "push", "--porcelain",
                f"--force-with-lease={ref}:{expected_old_sha}",
                f"https://github.com/{self.owner}/{self.repo}.git",
                f"{intended_new_sha}:{ref}",
            ],
            capture_output=True,
            check=False,
            env=environment,
        )
        if result.returncode:
            try:
                observed_sha = self._rehearsal_branch_head(ref)
                readback_error = None
            except Exception:
                observed_sha = None
                readback_error = "REF_READBACK_UNAVAILABLE"
            evidence = {
                **attempt,
                "write_outcome": "unknown",
                "observed_sha": observed_sha,
                "readback_error": readback_error,
            }
            self._unreconciled_branch_moves[ref] = evidence
            raise GitHubProtectionError(
                "REF_MOVE_UNKNOWN", evidence=evidence
            )
        try:
            observed_sha = self._rehearsal_branch_head(ref)
        except Exception:
            observed_sha = None
        if observed_sha != intended_new_sha:
            evidence = {
                **attempt,
                "write_outcome": "unknown",
                "observed_sha": observed_sha,
            }
            self._unreconciled_branch_moves[ref] = evidence
            raise GitHubProtectionError(
                "REF_MOVE_UNKNOWN", evidence=evidence
            )
        try:
            postwrite_rules_evidence = self._require_rehearsal_rules(
                ref, {"deletion", "non_fast_forward", "update"}
            )
            postwrite_rules_verified = self._rehearsal_gate(
                intent, "verify-principal-move-postwrite", ref,
                postwrite_rules_evidence,
            )
        except Exception:
            postwrite_rules_verified = False
        if postwrite_rules_verified is not True:
            evidence = {
                **attempt,
                "write_outcome": "provider_accepted",
                "observed_sha": observed_sha,
                "postwrite_rules_verified": False,
            }
            self._unreconciled_branch_moves[ref] = evidence
            raise GitHubProtectionError(
                "REF_MOVE_POSTWRITE_UNVERIFIED", evidence=evidence
            )
        return {
            **attempt,
            "write_outcome": "provider_accepted_and_read_back",
            "observed_sha": observed_sha,
            "postwrite_rules_verified": True,
        }

    @staticmethod
    def _json(response: ApiResponse) -> Any:
        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, TypeError):
            raise GitHubProtectionError("GITHUB_RESPONSE_INVALID") from None

    def _list_rulesets(self) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        rows: list[dict[str, Any]] = []
        etags: list[str] = []
        page = 1
        while True:
            query = urlencode({"includes_parents": "true", "per_page": 100, "page": page})
            response = self._api_request("GET", self._repo_path(f"/rulesets?{query}"))
            payload = self._json(response)
            if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
                raise GitHubProtectionError("RULESET_LIST_INVALID")
            rows.extend(payload)
            etag = response.headers.get("etag") or response.headers.get("ETag")
            if isinstance(etag, str) and etag:
                etags.append(etag)
            if len(payload) < 100:
                break
            page += 1
            if page > 100:
                raise GitHubProtectionError("RULESET_LIST_INCOMPLETE")
        return rows, tuple(etags)

    @staticmethod
    def _updated_at(value: Any) -> str:
        if not isinstance(value, str):
            raise GitHubProtectionError("RULESET_DETAIL_INVALID")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            raise GitHubProtectionError("RULESET_DETAIL_INVALID") from None

    @staticmethod
    def _rule(value: Any) -> Mapping[str, Any]:
        if not isinstance(value, dict) or not isinstance(value.get("type"), str):
            raise GitHubProtectionError("RULESET_DETAIL_INVALID")
        if "parameters" in value and not isinstance(value["parameters"], dict):
            raise GitHubProtectionError("RULESET_DETAIL_INVALID")
        return json.loads(canonical_json(value))

    @staticmethod
    def _actor(value: Any) -> Mapping[str, Any]:
        if (
            not isinstance(value, dict)
            or type(value.get("actor_id")) is not int
            or not isinstance(value.get("actor_type"), str)
            or not isinstance(value.get("bypass_mode"), str)
        ):
            raise GitHubProtectionError("RULESET_DETAIL_INVALID")
        return json.loads(canonical_json(value))

    @classmethod
    def _canonical_ruleset(cls, payload: Any) -> CanonicalRuleset:
        try:
            if not isinstance(payload, dict):
                raise ValueError
            identifier = payload["id"]
            name = payload["name"]
            target = payload["target"]
            enforcement = payload["enforcement"]
            all_conditions = payload["conditions"]
            conditions = all_conditions["ref_name"]
            include = conditions["include"]
            exclude = conditions["exclude"]
            rules = payload["rules"]
            actors = payload["bypass_actors"]
            if (
                type(identifier) is not int or identifier <= 0
                or not isinstance(name, str) or not name
                or target not in {"branch", "tag", "push", "repository"}
                or enforcement not in {"active", "disabled", "evaluate"}
                or not isinstance(include, list)
                or any(not isinstance(pattern, str) or not pattern for pattern in include)
                or not isinstance(exclude, list)
                or any(not isinstance(pattern, str) or not pattern for pattern in exclude)
                or not isinstance(rules, list)
                or not isinstance(actors, list)
            ):
                raise ValueError
            normalized_rules = tuple(sorted((cls._rule(rule) for rule in rules), key=canonical_json))
            rule_types = [rule["type"] for rule in normalized_rules]
            if len(rule_types) != len(set(rule_types)):
                raise ValueError
            normalized_actors = tuple(
                sorted((cls._actor(actor) for actor in actors), key=canonical_json)
            )
            if len({canonical_json(actor) for actor in normalized_actors}) != len(normalized_actors):
                raise ValueError
            return CanonicalRuleset(
                identifier,
                name,
                target,
                enforcement,
                tuple(sorted(include)),
                tuple(sorted(exclude)),
                json.loads(canonical_json({
                    name: value for name, value in all_conditions.items()
                    if name != "ref_name"
                })),
                normalized_rules,
                normalized_actors,
                cls._updated_at(payload["updated_at"]),
            )
        except (KeyError, TypeError, ValueError):
            raise GitHubProtectionError("RULESET_DETAIL_INVALID") from None

    def read_snapshot(self) -> RulesetSnapshot:
        summaries, etags = self._list_rulesets()
        summary_by_id: dict[int, dict[str, Any]] = {}
        for summary in summaries:
            identifier = summary.get("id")
            if type(identifier) is not int or identifier <= 0 or identifier in summary_by_id:
                raise GitHubProtectionError("RULESET_LIST_INVALID")
            summary_by_id[identifier] = summary
        rulesets: list[CanonicalRuleset] = []
        for identifier in sorted(summary_by_id):
            detail_response = self._api_request(
                "GET", self._repo_path(f"/rulesets/{identifier}")
            )
            detail = self._json(detail_response)
            ruleset = self._canonical_ruleset(detail)
            summary = summary_by_id[identifier]
            if (
                ruleset.id != identifier
                or summary.get("name") != ruleset.name
                or summary.get("target") != ruleset.target
                or summary.get("enforcement") != ruleset.enforcement
                or self._updated_at(summary.get("updated_at")) != ruleset.updated_at
            ):
                raise GitHubProtectionError("RULESET_SNAPSHOT_CHANGED")
            rulesets.append(ruleset)
        normalized = {
            "repository": self.repository,
            "rulesets": [item.to_dict() for item in rulesets],
        }
        payload = canonical_json(normalized)
        return RulesetSnapshot(
            self.repository,
            self._timestamp(),
            tuple(rulesets),
            payload,
            sha256_hex(payload),
            etags,
        )

    @staticmethod
    def _rule_types(ruleset: CanonicalRuleset) -> tuple[str, ...]:
        return tuple(sorted(str(rule["type"]) for rule in ruleset.rules))

    @staticmethod
    def _exact_actors(ruleset: CanonicalRuleset) -> tuple[tuple[int, str, str], ...]:
        return tuple(
            sorted(
                (int(actor["actor_id"]), str(actor["actor_type"]), str(actor["bypass_mode"]))
                for actor in ruleset.bypass_actors
            )
        )

    @staticmethod
    def _overlaps_candidate_namespace(pattern: str) -> bool:
        prefix = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
        namespace = "refs/heads/candidates/"
        return prefix.startswith(namespace) or namespace.startswith(prefix)

    def _candidate_pair(self, snapshot: RulesetSnapshot) -> tuple[CanonicalRuleset, CanonicalRuleset]:
        by_id = {ruleset.id: ruleset for ruleset in snapshot.rulesets}
        immutable = by_id.get(_CANDIDATE_IMMUTABLE_ID)
        creation = by_id.get(_CANDIDATE_CREATION_ID)
        if immutable is None or creation is None:
            raise GitHubProtectionError("CANDIDATE_PROTECTION_INCOMPLETE")
        expected_common = (
            "branch", "active", (_CANDIDATE_PATTERN,), (),
        )
        if (
            immutable.name != _CANDIDATE_IMMUTABLE_NAME
            or (immutable.target, immutable.enforcement, immutable.include, immutable.exclude) != expected_common
            or immutable.conditions_extra
            or self._rule_types(immutable) != ("deletion", "non_fast_forward", "update")
            or self._exact_actors(immutable)
            or any(rule.get("parameters") for rule in immutable.rules)
        ):
            raise GitHubProtectionError("CANDIDATE_PROTECTION_IMMUTABLE_INVALID")
        if (
            creation.name != _CANDIDATE_CREATION_NAME
            or (creation.target, creation.enforcement, creation.include, creation.exclude) != expected_common
            or creation.conditions_extra
            or self._rule_types(creation) != ("creation",)
            or self._exact_actors(creation) != ((5, "RepositoryRole", "always"),)
            or any(rule.get("parameters") for rule in creation.rules)
        ):
            raise GitHubProtectionError("CANDIDATE_PROTECTION_CREATION_INVALID")
        for ruleset in snapshot.rulesets:
            if ruleset.id in {_CANDIDATE_IMMUTABLE_ID, _CANDIDATE_CREATION_ID}:
                continue
            if (
                ruleset.target == "branch"
                and ruleset.enforcement == "active"
                and any(self._overlaps_candidate_namespace(pattern) for pattern in ruleset.include)
            ):
                raise GitHubProtectionError("CANDIDATE_PROTECTION_OVERLAP_AMBIGUOUS")
        return immutable, creation

    def verify_candidate_ref(self, ref: str) -> VerifiedProtectionEvidence:
        try:
            require_candidate_ref(ref)
        except GitIdentityError:
            raise GitHubProtectionError("CANDIDATE_REF_INVALID") from None
        if not ref.startswith("refs/heads/candidates/"):
            raise GitHubProtectionError("CANDIDATE_REF_INVALID")
        snapshot = self.read_snapshot()
        immutable, creation = self._candidate_pair(snapshot)
        protection = ProtectionEvidence(
            rule_id=f"{immutable.id}+{creation.id}",
            pattern=_CANDIDATE_PATTERN,
            snapshot_sha256=snapshot.snapshot_sha256,
            retrieved_at_utc=snapshot.retrieved_at_utc,
            creation_guarded=True,
            update_guarded=True,
            deletion_guarded=True,
        )
        proof = VerifiedProtectionEvidence(
            protection,
            self.repository,
            ref,
            (immutable.id, creation.id),
            snapshot.snapshot_sha256,
            snapshot.retrieved_at_utc,
            self._nonce,
        )
        proof.require(ref, creation=True)
        return proof

    def verify_public_tag(self, ref: str) -> VerifiedProtectionEvidence:
        if not isinstance(ref, str) or not ref.startswith("refs/tags/v"):
            raise GitHubProtectionError("PUBLIC_TAG_REF_INVALID")
        snapshot = self.read_snapshot()
        by_id = {ruleset.id: ruleset for ruleset in snapshot.rulesets}
        immutable = by_id.get(_PUBLIC_TAG_IMMUTABLE_ID)
        creation = by_id.get(_PUBLIC_TAG_CREATION_ID)
        common = ("tag", "active", (_PUBLIC_TAG_PATTERN,), ("refs/tags/v-0.1",))
        if (
            immutable is None or creation is None
            or (immutable.name, immutable.target, immutable.enforcement, immutable.include, immutable.exclude)
            != ("CYAx canonical release tags immutable", *common)
            or immutable.conditions_extra
            or self._rule_types(immutable) != ("deletion", "non_fast_forward", "update")
            or self._exact_actors(immutable)
            or any(rule.get("parameters") for rule in immutable.rules)
            or (creation.name, creation.target, creation.enforcement, creation.include, creation.exclude)
            != ("CYAx canonical release tag creation", *common)
            or creation.conditions_extra
            or self._rule_types(creation) != ("creation",)
            or self._exact_actors(creation) != ((5, "RepositoryRole", "always"),)
            or any(rule.get("parameters") for rule in creation.rules)
        ):
            raise GitHubProtectionError("PUBLIC_TAG_PROTECTION_INVALID")
        protection = ProtectionEvidence(
            rule_id=f"{immutable.id}+{creation.id}",
            pattern=_PUBLIC_TAG_PATTERN,
            snapshot_sha256=snapshot.snapshot_sha256,
            retrieved_at_utc=snapshot.retrieved_at_utc,
            creation_guarded=True,
            update_guarded=True,
            deletion_guarded=True,
            canonical_public_tags_globally_guarded=True,
        )
        proof = VerifiedProtectionEvidence(
            protection,
            self.repository,
            ref,
            (immutable.id, creation.id),
            snapshot.snapshot_sha256,
            snapshot.retrieved_at_utc,
            self._nonce,
        )
        proof.require_public_tag(ref)
        return proof

    def is_live_protection_evidence(
        self, evidence: Any, *, ref: str, repository: str
    ) -> bool:
        if not (
            isinstance(evidence, VerifiedProtectionEvidence)
            and evidence._adapter_nonce is self._nonce
            and evidence.repository == self.repository == repository
            and evidence.target_ref == ref
            and re.fullmatch(r"[0-9a-f]{64}", evidence.snapshot_sha256) is not None
        ):
            return False
        try:
            current = self.read_snapshot()
        except GitHubProtectionError:
            return False
        return current.snapshot_sha256 == evidence.snapshot_sha256

    def observe_rules_for_ref(self, ref: str) -> RuleEvaluationObservation:
        """Read the non-mutating GitHub effective-rules evaluation for a ref."""

        if not isinstance(ref, str) or not ref.startswith("refs/heads/"):
            raise GitHubProtectionError("EVALUATION_REF_INVALID")
        branch = ref.removeprefix("refs/heads/")
        if not branch or any(part in {"", ".", ".."} for part in branch.split("/")):
            raise GitHubProtectionError("EVALUATION_REF_INVALID")
        snapshot = self.read_snapshot()
        encoded_branch = "/".join(quote(part, safe="") for part in branch.split("/"))
        response = self._api_request(
            "GET", self._repo_path(f"/rules/branches/{encoded_branch}")
        )
        payload = self._json(response)
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise GitHubProtectionError("RULE_EVALUATION_INVALID")
        ids: set[int] = set()
        rule_types: list[str] = []
        for item in payload:
            identifier = item.get("ruleset_id")
            rule_type = item.get("type")
            if type(identifier) is not int or identifier <= 0 or not isinstance(rule_type, str):
                raise GitHubProtectionError("RULE_EVALUATION_IDENTITY_UNAVAILABLE")
            ids.add(identifier)
            rule_types.append(rule_type)
        if ref.startswith("refs/heads/candidates/"):
            immutable, creation = self._candidate_pair(snapshot)
            expected_ids = {immutable.id, creation.id}
            expected_types = {"creation", "deletion", "non_fast_forward", "update"}
            if ids != expected_ids or set(rule_types) != expected_types:
                raise GitHubProtectionError("RULE_EVALUATION_CANDIDATE_MISMATCH")
        normalized_response = canonical_json(payload)
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        return RuleEvaluationObservation(
            self.repository,
            ref,
            self._timestamp(),
            snapshot.snapshot_sha256,
            sha256_hex(normalized_response),
            headers.get("etag"),
            headers.get("x-github-request-id"),
            tuple(sorted(ids)),
            tuple(sorted(set(rule_types))),
            normalized_response,
        )

    def _require_rehearsal_rules(
        self, ref: str, required_types: set[str]
    ) -> dict[str, Any]:
        try:
            require_full_ref(ref)
        except (GitIdentityError, TypeError):
            raise GitHubProtectionError("REHEARSAL_REF_INVALID") from None
        if not ref.startswith("refs/heads/rehearsal/"):
            raise GitHubProtectionError("REHEARSAL_REF_INVALID")
        observation = self.observe_rules_for_ref(ref)
        observed_types = set(observation.rule_types)
        if (
            observation.repository != self.repository
            or observation.ref != ref
            or not required_types.issubset(observed_types)
            or not observation.ruleset_ids
        ):
            raise GitHubProtectionError(
                "REHEARSAL_RULESET_UNVERIFIED",
                evidence={
                    "ref": ref,
                    "required_rule_types": sorted(required_types),
                    "observed_rule_types": sorted(observed_types),
                    "ruleset_snapshot_sha256": observation.ruleset_snapshot_sha256,
                },
            )
        return observation.to_dict()

    def _authorize_freeze(self, intent: Any, action: str, ref: str) -> None:
        intent_repository = (
            intent.get("repository") if isinstance(intent, Mapping)
            else getattr(intent, "repository", None)
        )
        if intent_repository != self.repository:
            raise GitHubProtectionError("FREEZE_REPOSITORY_MISMATCH")
        if self._authorization_gate is None:
            raise GitHubProtectionError("FREEZE_AUTHORIZATION_UNVERIFIED")
        try:
            authorized = self._authorization_gate(intent, action, ref)
        except Exception:
            raise GitHubProtectionError("FREEZE_AUTHORIZATION_UNVERIFIED") from None
        if authorized is not True:
            raise GitHubProtectionError("FREEZE_AUTHORIZATION_UNVERIFIED")

    def _freeze_ruleset(self, ref: str, snapshot: RulesetSnapshot) -> CanonicalRuleset:
        expected_id, expected_name = _FREEZE_CONTROLS[ref]
        matches = [rule for rule in snapshot.rulesets if rule.id == expected_id]
        if len(matches) != 1:
            raise GitHubProtectionError("FREEZE_RULESET_UNAVAILABLE")
        rule = matches[0]
        if (
            rule.name != expected_name
            or rule.target != "branch"
            or rule.include != (ref,)
            or rule.exclude
            or rule.conditions_extra
            or self._rule_types(rule) != ("deletion", "non_fast_forward", "update")
            or self._exact_actors(rule)
            or any(item.get("parameters") for item in rule.rules)
        ):
            raise GitHubProtectionError("FREEZE_RULESET_INVALID")
        return rule

    @staticmethod
    def _same_freeze_definition(
        expected: CanonicalRuleset, observed: CanonicalRuleset
    ) -> bool:
        left = expected.to_dict()
        right = observed.to_dict()
        left.pop("enforcement")
        right.pop("enforcement")
        left.pop("updated_at")
        right.pop("updated_at")
        return left == right

    def _require_reconciled_freeze(self, ref: str) -> None:
        if ref in self._unreconciled_freeze_writes:
            raise GitHubProtectionError(
                "FREEZE_WRITE_UNRECONCILED",
                evidence=self._unreconciled_freeze_writes[ref],
            )

    def _freeze_write_readback(
        self,
        *,
        ref: str,
        ruleset: CanonicalRuleset,
        desired: str,
        attempt: Mapping[str, Any],
        outcome: str,
        provider_request_id: str | None,
        provider_status: int | None,
    ) -> tuple[RulesetSnapshot | None, CanonicalRuleset | None, dict[str, Any]]:
        observation: dict[str, Any] = {
            **dict(attempt),
            "requested_enforcement": desired,
            "write_outcome": outcome,
            "provider_status": provider_status,
            "provider_request_id": provider_request_id,
        }
        try:
            snapshot = self.read_snapshot()
            observed = self._freeze_ruleset(ref, snapshot)
            observation.update({
                "readback_at_utc": snapshot.retrieved_at_utc,
                "readback_snapshot_sha256": snapshot.snapshot_sha256,
                "observed_enforcement": observed.enforcement,
                "observed_ruleset_updated_at": observed.updated_at,
                "definition_matches": self._same_freeze_definition(ruleset, observed),
            })
            return snapshot, observed, observation
        except Exception as error:
            reason = getattr(error, "reason_code", None)
            try:
                readback_at = self._timestamp()
            except GitHubProtectionError:
                readback_at = None
            observation.update({
                "readback_at_utc": readback_at,
                "readback_error": (
                    reason if isinstance(reason, str) else "RULESET_READBACK_UNAVAILABLE"
                ),
                "observed_enforcement": "unreadable",
            })
            return None, None, observation

    def _set_enforcement(
        self,
        ref: str,
        ruleset: CanonicalRuleset,
        enforcement: str,
    ) -> tuple[RulesetSnapshot, CanonicalRuleset]:
        if enforcement not in {"active", "disabled"}:
            raise GitHubProtectionError("FREEZE_ENFORCEMENT_INVALID")
        self._require_reconciled_freeze(ref)
        attempt = {
            "write_id": uuid.uuid4().hex,
            "target_ref": ref,
            "ruleset_id": ruleset.id,
            "prior_enforcement": ruleset.enforcement,
            "started_at_utc": self._timestamp(),
        }
        payload = {
            "name": ruleset.name,
            "target": ruleset.target,
            "enforcement": enforcement,
            "conditions": {
                "ref_name": {
                    "include": list(ruleset.include),
                    "exclude": list(ruleset.exclude),
                },
                **dict(ruleset.conditions_extra),
            },
            "rules": [dict(rule) for rule in ruleset.rules],
            "bypass_actors": [dict(actor) for actor in ruleset.bypass_actors],
        }
        try:
            response = self._api_request(
                "PUT", self._repo_path(f"/rulesets/{ruleset.id}"), payload
            )
        except GitHubProtectionError as error:
            terminal = error.provider_rejection_is_terminal
            outcome = "terminal_rejection" if terminal else "unknown"
            snapshot, observed, evidence = self._freeze_write_readback(
                ref=ref,
                ruleset=ruleset,
                desired=enforcement,
                attempt=attempt,
                outcome=outcome,
                provider_request_id=error.provider_request_id,
                provider_status=error.provider_status,
            )
            if (
                terminal
                and observed is not None
                and observed.enforcement == ruleset.enforcement
                and observed.updated_at == ruleset.updated_at
                and evidence.get("definition_matches") is True
            ):
                raise GitHubProtectionError(
                    "FREEZE_WRITE_REJECTED",
                    provider_status=error.provider_status,
                    provider_request_id=error.provider_request_id,
                    evidence=evidence,
                ) from None
            self._unreconciled_freeze_writes[ref] = evidence
            raise GitHubProtectionError(
                "FREEZE_WRITE_UNKNOWN",
                provider_status=error.provider_status,
                provider_request_id=error.provider_request_id,
                evidence=evidence,
            ) from None

        response_request_id = response.headers.get("x-github-request-id")
        snapshot, observed, evidence = self._freeze_write_readback(
            ref=ref,
            ruleset=ruleset,
            desired=enforcement,
            attempt=attempt,
            outcome="provider_accepted",
            provider_request_id=response_request_id,
            provider_status=response.status,
        )
        if (
            snapshot is None
            or observed is None
            or observed.enforcement != enforcement
            or evidence.get("definition_matches") is not True
        ):
            evidence["write_outcome"] = "unknown"
            self._unreconciled_freeze_writes[ref] = evidence
            raise GitHubProtectionError(
                "FREEZE_WRITE_UNKNOWN",
                provider_status=response.status,
                provider_request_id=response_request_id,
                evidence=evidence,
            )
        return snapshot, observed

    def _branch(self, ref: str) -> tuple[str, str]:
        branch = ref.removeprefix("refs/heads/")
        if ref not in _FREEZE_CONTROLS:
            raise GitHubProtectionError("FREEZE_TARGET_INVALID")
        response = self._api_request(
            "GET", self._repo_path(f"/branches/{quote(branch, safe='')}")
        )
        payload = self._json(response)
        try:
            name = payload["name"]
            sha = payload["commit"]["sha"]
        except (KeyError, TypeError):
            raise GitHubProtectionError("FREEZE_TARGET_IDENTITY_INVALID") from None
        if name != branch or not isinstance(sha, str) or _SHA1.fullmatch(sha) is None:
            raise GitHubProtectionError("FREEZE_TARGET_IDENTITY_INVALID")
        return name, sha

    def _version_at(self, commit_sha: str) -> str:
        query = urlencode({"ref": commit_sha})
        response = self._api_request(
            "GET", self._repo_path(f"/contents/Project.toml?{query}")
        )
        payload = self._json(response)
        try:
            if payload["path"] != "Project.toml" or payload["encoding"] != "base64":
                raise ValueError
            content = base64.b64decode(payload["content"], validate=True)
            version = tomllib.loads(content.decode("utf-8"))["version"]
            if not isinstance(version, str) or not version:
                raise ValueError
            return version
        except (KeyError, TypeError, ValueError, UnicodeError, tomllib.TOMLDecodeError):
            raise GitHubProtectionError("FREEZE_VERSION_IDENTITY_INVALID") from None

    def acquire_freeze(self, ref: str, intent: Any) -> FreezeLease:
        if ref not in _FREEZE_CONTROLS:
            raise GitHubProtectionError("FREEZE_TARGET_INVALID")
        self._require_reconciled_freeze(ref)
        self._authorize_freeze(intent, "activate", ref)
        before = self.read_snapshot()
        ruleset = self._freeze_ruleset(ref, before)
        if ruleset.enforcement != "disabled":
            raise GitHubProtectionError("FREEZE_RULESET_STATE_AMBIGUOUS")
        branch_name, branch_sha = self._branch(ref)
        after, active = self._set_enforcement(ref, ruleset, "active")
        after_name, after_sha = self._branch(ref)
        if after_name != branch_name or after_sha != branch_sha:
            # The control remains active. The caller must reconcile the moved ref.
            raise GitHubProtectionError("FREEZE_TARGET_MOVED")
        version = self._version_at(after_sha)
        lease = FreezeLease(
            uuid.uuid4().hex,
            ref,
            active.id,
            after_sha,
            version,
            active.updated_at,
            after.snapshot_sha256,
            after.retrieved_at_utc,
            self._nonce,
            intent,
        )
        self._leases[lease.lease_id] = lease
        return lease

    def freeze_line(self, intent: Any) -> FreezeLease:
        return self.acquire_freeze("refs/heads/vmm", intent)

    def freeze_main(self, intent: Any) -> dict[str, Any]:
        lease = self.acquire_freeze("refs/heads/main", intent)
        return {
            "token": lease,
            "sha": lease.branch_sha,
            "version": lease.version,
            "ruleset_id": lease.ruleset_id,
            "ruleset_snapshot_sha256": lease.snapshot_sha256,
            "retrieved_at_utc": lease.retrieved_at_utc,
        }

    def is_live_freeze_lease(self, lease: Any, *, ref: str) -> bool:
        return (
            isinstance(lease, FreezeLease)
            and lease._adapter_nonce is self._nonce
            and lease.ref == ref
            and self._leases.get(lease.lease_id) is lease
            and _SHA1.fullmatch(lease.branch_sha) is not None
            and re.fullmatch(r"[0-9a-f]{64}", lease.snapshot_sha256) is not None
        )

    def release_freeze(
        self, lease: FreezeLease, *, expected_branch_sha: str | None = None
    ) -> None:
        if not self.is_live_freeze_lease(lease, ref=getattr(lease, "ref", "")):
            raise GitHubProtectionError("FREEZE_LEASE_UNVERIFIED")
        self._require_reconciled_freeze(lease.ref)
        self._authorize_freeze(lease._intent, "deactivate", lease.ref)
        current = self.read_snapshot()
        ruleset = self._freeze_ruleset(lease.ref, current)
        if ruleset.enforcement != "active" or ruleset.updated_at != lease.ruleset_updated_at:
            raise GitHubProtectionError("FREEZE_RULESET_STALE")
        expected = expected_branch_sha or lease.branch_sha
        if _SHA1.fullmatch(expected) is None:
            raise GitHubProtectionError("FREEZE_TARGET_IDENTITY_INVALID")
        branch_name, before_sha = self._branch(lease.ref)
        if before_sha != expected:
            raise GitHubProtectionError("FREEZE_TARGET_MOVED")
        after, disabled = self._set_enforcement(lease.ref, ruleset, "disabled")
        after_name, after_sha = self._branch(lease.ref)
        if after_name != branch_name or after_sha != expected:
            # Restore the freeze before reporting a moved ref. If the provider
            # cannot verify restoration, the operation remains blocked.
            self._set_enforcement(lease.ref, disabled, "active")
            raise GitHubProtectionError("FREEZE_TARGET_MOVED")
        del self._leases[lease.lease_id]

    def verify_live_freeze_lease(
        self,
        lease: FreezeLease,
        *,
        ref: str,
        expected_branch_sha: str | None = None,
    ) -> bool:
        if not self.is_live_freeze_lease(lease, ref=ref):
            return False
        snapshot = self.read_snapshot()
        ruleset = self._freeze_ruleset(ref, snapshot)
        if ruleset.enforcement != "active" or ruleset.updated_at != lease.ruleset_updated_at:
            return False
        branch_name, branch_sha = self._branch(ref)
        if branch_name != ref.removeprefix("refs/heads/"):
            return False
        expected = expected_branch_sha
        return expected is None or branch_sha == expected

    def unfreeze_line(
        self, lease: FreezeLease, *, expected_branch_sha: str | None = None
    ) -> None:
        if lease.ref != "refs/heads/vmm":
            raise GitHubProtectionError("FREEZE_LEASE_TARGET_MISMATCH")
        self.release_freeze(lease, expected_branch_sha=expected_branch_sha)

    def unfreeze_main(
        self, lease: FreezeLease, *, expected_branch_sha: str | None = None
    ) -> None:
        if lease.ref != "refs/heads/main":
            raise GitHubProtectionError("FREEZE_LEASE_TARGET_MISMATCH")
        self.release_freeze(lease, expected_branch_sha=expected_branch_sha)
