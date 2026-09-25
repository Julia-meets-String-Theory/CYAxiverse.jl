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
import tomllib
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
import uuid

from .codec import canonical_json, sha256_hex
from .git_refs import GitIdentityError, ProtectionEvidence, require_candidate_ref


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


class GitHubProtectionError(RuntimeError):
    """A reason-coded, privacy-safe failure from the GitHub adapter."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


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
        self._nonce = object()
        self._leases: dict[str, FreezeLease] = {}

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
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
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
                raise GitHubProtectionError("GITHUB_API_REQUEST_FAILED") from None
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
                error.close()
                raise GitHubProtectionError("GITHUB_API_REQUEST_FAILED") from None
            except (URLError, TimeoutError, OSError, ValueError):
                raise GitHubProtectionError("GITHUB_API_REQUEST_FAILED") from None
        if not isinstance(response, ApiResponse) or not 200 <= response.status < 300:
            raise GitHubProtectionError("GITHUB_API_REQUEST_FAILED")
        return response

    def _repo_path(self, suffix: str) -> str:
        return f"/repos/{quote(self.owner, safe='')}/{quote(self.repo, safe='')}{suffix}"

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
            or self._rule_types(immutable) != ("deletion", "non_fast_forward", "update")
            or self._exact_actors(immutable)
            or (creation.name, creation.target, creation.enforcement, creation.include, creation.exclude)
            != ("CYAx canonical release tag creation", *common)
            or self._rule_types(creation) != ("creation",)
            or self._exact_actors(creation) != ((5, "RepositoryRole", "always"),)
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
        return (
            isinstance(evidence, VerifiedProtectionEvidence)
            and evidence._adapter_nonce is self._nonce
            and evidence.repository == self.repository == repository
            and evidence.target_ref == ref
            and re.fullmatch(r"[0-9a-f]{64}", evidence.snapshot_sha256) is not None
        )

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
        self._authorize_freeze(intent, "activate", ref)
        before = self.read_snapshot()
        ruleset = self._freeze_ruleset(ref, before)
        if ruleset.enforcement != "disabled":
            raise GitHubProtectionError("FREEZE_RULESET_STATE_AMBIGUOUS")
        branch_name, branch_sha = self._branch(ref)
        self._set_enforcement(ruleset, "active")
        after = self.read_snapshot()
        active = self._freeze_ruleset(ref, after)
        if active.enforcement != "active":
            raise GitHubProtectionError("FREEZE_POSTWRITE_UNVERIFIED")
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
        self._set_enforcement(ruleset, "disabled")
        after = self.read_snapshot()
        disabled = self._freeze_ruleset(lease.ref, after)
        if disabled.enforcement != "disabled":
            raise GitHubProtectionError("FREEZE_RELEASE_UNVERIFIED")
        after_name, after_sha = self._branch(lease.ref)
        if after_name != branch_name or after_sha != expected:
            # Restore the freeze before reporting a moved ref. If the provider
            # cannot verify restoration, the operation remains blocked.
            self._set_enforcement(disabled, "active")
            restored = self.read_snapshot()
            active = self._freeze_ruleset(lease.ref, restored)
            if active.enforcement != "active":
                raise GitHubProtectionError("FREEZE_RELEASE_UNVERIFIED")
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

    def _set_enforcement(self, ruleset: CanonicalRuleset, enforcement: str) -> None:
        if enforcement not in {"active", "disabled"}:
            raise GitHubProtectionError("FREEZE_ENFORCEMENT_INVALID")
        self._api_request(
            "PUT",
            self._repo_path(f"/rulesets/{ruleset.id}"),
            {
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
            },
        )

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
