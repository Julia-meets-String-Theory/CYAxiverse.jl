"""Pure validators for Gate A exact-tree certification evidence.

The release transaction code owns Git operations.  This module only checks
the identity and binding of a certification record, so it can be used with a
real repository, an API response, or a small fixture without mutating refs.
The returned mappings are deliberately JSON serialisable and use the same
``status``/``reason_code`` vocabulary as the lifecycle writers.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from urllib.parse import urlsplit


SUPPORTED_BINDINGS = frozenset(("tree-bound", "commit-bound"))
BLOCKED = "BLOCKED"
INVALID = "INVALID"
PASS = "PASS"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def is_safe_public_value(value: Any) -> bool:
    """Return whether a durable value is safe to expose publicly.

    This lexical gate rejects private locators and credential-like values.
    It is shared by the certification and release validators; it does not
    claim that an otherwise safe value is approved evidence.
    """

    if isinstance(value, Mapping):
        return all(
            isinstance(key, str)
            and is_safe_public_value(key)
            and is_safe_public_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(is_safe_public_value(item) for item in value)
    if not isinstance(value, str):
        return True
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        return False
    lowered = value.lower()
    if (
        value.startswith(("/", "~", "\\\\"))
        or _DRIVE_PATH.match(value) is not None
    ):
        return False
    normalized = lowered.replace("\\", "/")
    if normalized.startswith(("file:", "local:", "ssh:")):
        return False
    if "://" in normalized:
        try:
            parsed = urlsplit(value)
            host = (parsed.hostname or "").lower()
            username = parsed.username
            password = parsed.password
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"} or username or password:
            return False
        if (
            not host
            or "." not in host
            or host.endswith((".local", ".internal", ".lan", ".corp"))
            or host in {"localhost", "intranet", "internal"}
        ):
            return False
    components = {part for part in normalized.split("/") if part in {
        "private", "users", "home", "tmp", "var", "codex"
    }}
    if components or re.search(r"(?:^|[./\s])private(?:/|$)", normalized):
        return False
    forbidden = (
        "credential",
        "authorization",
        "bearer ",
        "password",
        "token",
        "secret",
        "api_key",
        "apikey",
        ".env",
        "localhost",
        "127.0.0.1",
    )
    return not any(marker in lowered for marker in forbidden)


def _value(record: Mapping[str, Any], *names: str) -> Any:
    """Return the first present value from a set of compatibility aliases."""

    for name in names:
        if name in record:
            return record[name]
    return None


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value


def _git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None


def _result(
    status: str,
    reason_code: str | None = None,
    errors: list[str] | None = None,
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if reason_code is not None:
        result["reason_code"] = reason_code
    if errors:
        result["errors"] = list(errors)
    result.update(details)
    return result


def certification_binding(record: Mapping[str, Any]) -> str | None:
    """Return the canonical binding name, accepting one legacy spelling."""

    value = _value(record, "binding", "certification_binding")
    if value == "tree_bound":
        return "tree-bound"
    if value == "commit_bound":
        return "commit-bound"
    return value if isinstance(value, str) else None


def validate_certification_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the independent pinned identities in a certification record.

    The record must identify the package, policy, harness, environment and
    evidence independently.  A missing or unsupported binding is blocked
    before a public tag can be created.  A malformed identity is invalid.
    """

    if not isinstance(record, Mapping):
        return _result(BLOCKED, "CERTIFICATION_RECORD_UNAVAILABLE")
    if not is_safe_public_value(record):
        return _result(INVALID, "UNSAFE_PUBLIC_EVIDENCE")

    binding = certification_binding(record)
    if binding not in SUPPORTED_BINDINGS:
        return _result(BLOCKED, "UNSUPPORTED_CERTIFICATION_BINDING")

    required = {
        "package_commit": ("package_commit", "package_sha", "subject_sha"),
        "package_tree": ("package_tree", "subject_tree", "certified_tree"),
        "policy_revision": ("policy_revision", "policy_sha"),
        "harness_revision": ("harness_revision", "harness_sha"),
        "environment": ("environment", "environment_id", "environment_ref"),
        "evidence": ("evidence", "evidence_ref", "evidence_digest", "evidence_sha"),
    }
    errors: list[str] = []
    identity: dict[str, Any] = {"binding": binding}
    for canonical, aliases in required.items():
        value = _value(record, *aliases)
        if canonical in {"package_commit", "package_tree"}:
            valid = _git_sha(value)
        else:
            valid = _text(value)
        if not valid:
            errors.append(f"missing_or_invalid_{canonical}")
        else:
            identity[canonical] = value

    # The package commit is pinned independently from the tree.  A tree-only
    # identity is useful for transfer, but it is not certification evidence.
    if errors:
        return _result(INVALID, "CERTIFICATION_IDENTITY_INVALID", errors, identity=identity)
    return _result(PASS, identity=identity)


def certify_exact_tree(
    *,
    candidate_tree: str,
    certified_tree: str,
    anchor_tree: str,
    final_release_tree: str | None = None,
    binding: str = "tree-bound",
    candidate_commit: str | None = None,
    subject_commit: str | None = None,
    final_release_commit: str | None = None,
) -> dict[str, Any]:
    """Prove the immutable candidate, anchor and release trees are equal.

    This helper is intentionally independent of Git.  Callers obtain the
    immutable SHA/tree identities from their Git provider and pass them here.
    """

    if binding not in SUPPORTED_BINDINGS:
        return _result(BLOCKED, "UNSUPPORTED_CERTIFICATION_BINDING")
    values = {
        "candidate_tree": candidate_tree,
        "certified_tree": certified_tree,
        "anchor_tree": anchor_tree,
        "final_release_tree": final_release_tree,
    }
    if any(not _git_sha(value) for value in (candidate_tree, certified_tree, anchor_tree)):
        return _result(INVALID, "TREE_IDENTITY_INVALID")
    if final_release_tree is not None and not _git_sha(final_release_tree):
        return _result(INVALID, "TREE_IDENTITY_INVALID")
    if candidate_commit is not None and not _git_sha(candidate_commit):
        return _result(INVALID, "COMMIT_IDENTITY_INVALID")
    if subject_commit is not None and not _git_sha(subject_commit):
        return _result(INVALID, "COMMIT_IDENTITY_INVALID")
    if final_release_commit is not None and not _git_sha(final_release_commit):
        return _result(INVALID, "COMMIT_IDENTITY_INVALID")
    if len({candidate_tree, certified_tree, anchor_tree}) != 1:
        return _result(INVALID, "CERTIFIED_TREE_MISMATCH", trees=values)
    if final_release_tree is not None and final_release_tree != certified_tree:
        return _result(INVALID, "RELEASE_TREE_MISMATCH", trees=values)
    if binding == "commit-bound":
        if not _git_sha(subject_commit) or not _git_sha(final_release_commit):
            return _result(BLOCKED, "RECERTIFICATION_REQUIRED")
        if subject_commit != final_release_commit:
            return _result(BLOCKED, "RECERTIFICATION_REQUIRED")
    return _result(
        PASS,
        certified_tree=certified_tree,
        candidate_commit=candidate_commit,
        subject_commit=subject_commit,
        final_release_commit=final_release_commit,
        binding=binding,
    )


def validate_certification_transfer(
    record: Mapping[str, Any],
    *,
    candidate_commit: str,
    candidate_tree: str,
    final_release_commit: str,
    final_release_tree: str,
    anchor_tree: str,
    public_tag_exists: bool = False,
) -> dict[str, Any]:
    """Validate the only two permitted candidate-to-release transfers.

    Tree-bound evidence can transfer to a different commit when every
    certified tree is equal and durable transfer evidence is present.
    Commit-bound evidence cannot transfer; a new certification is required
    before a tag is created.  ``public_tag_exists`` is accepted to make the
    call site explicit; a failed transfer never authorises retagging.
    """

    identity = validate_certification_identity(record)
    if identity["status"] != PASS:
        return identity
    binding = identity["identity"]["binding"]
    subject_commit = _value(record, "package_commit", "package_sha", "subject_sha")
    subject_tree = _value(record, "package_tree", "subject_tree", "certified_tree")
    record_candidate_commit = _value(record, "candidate_commit", "candidate_sha")
    record_candidate_tree = _value(record, "candidate_tree")
    record_anchor_tree = _value(record, "anchor_tree")

    if not _git_sha(candidate_commit) or not _git_sha(final_release_commit):
        return _result(INVALID, "COMMIT_IDENTITY_INVALID")
    if not all(_git_sha(value) for value in (candidate_tree, final_release_tree, anchor_tree)):
        return _result(INVALID, "TREE_IDENTITY_INVALID")

    errors: list[str] = []
    if record_candidate_commit is not None and record_candidate_commit != candidate_commit:
        errors.append("candidate_commit_mismatch")
    if record_candidate_tree is not None and record_candidate_tree != candidate_tree:
        errors.append("candidate_tree_mismatch")
    if record_anchor_tree is not None and record_anchor_tree != anchor_tree:
        errors.append("anchor_tree_mismatch")
    if subject_tree != candidate_tree or subject_tree != anchor_tree:
        errors.append("certified_tree_mismatch")
    if errors:
        return _result(INVALID, "CERTIFICATION_TREE_MISMATCH", errors)

    if binding == "commit-bound" and subject_commit != final_release_commit:
        return _result(
            BLOCKED,
            "RECERTIFICATION_REQUIRED",
            transfer_forbidden=True,
            public_tag_exists=public_tag_exists,
        )

    if binding == "tree-bound" and subject_commit != final_release_commit:
        transfer = _value(
            record,
            "transfer_evidence",
            "transfer_proof",
            "certification_transfer_evidence",
            "certification_transfer",
        )
        if not isinstance(transfer, Mapping):
            return _result(BLOCKED, "TREE_BOUND_TRANSFER_EVIDENCE_REQUIRED")
        transfer_certified_tree = _value(transfer, "certified_tree", "subject_tree")
        if (
            transfer_certified_tree is not None
            and transfer_certified_tree != subject_tree
            or _value(transfer, "candidate_tree") != candidate_tree
            or _value(transfer, "final_release_tree", "release_tree") != final_release_tree
            or _value(transfer, "anchor_tree") != anchor_tree
        ):
            return _result(INVALID, "TREE_BOUND_TRANSFER_MISMATCH")
        if transfer_certified_tree is not None and not _git_sha(transfer_certified_tree):
            return _result(INVALID, "TREE_BOUND_TRANSFER_EVIDENCE_INVALID")
        if not all(
            _git_sha(_value(transfer, name))
            for name in ("candidate_tree", "final_release_tree", "anchor_tree")
        ):
            return _result(INVALID, "TREE_BOUND_TRANSFER_EVIDENCE_INVALID")
        if not _text(_value(transfer, "evidence_ref", "evidence_digest", "digest")):
            return _result(INVALID, "TREE_BOUND_TRANSFER_EVIDENCE_INVALID")

    if final_release_tree != subject_tree:
        return _result(INVALID, "RELEASE_TREE_MISMATCH")
    return _result(
        PASS,
        binding=binding,
        transferred=subject_commit != final_release_commit,
        candidate_commit=candidate_commit,
        final_release_commit=final_release_commit,
        public_tag_exists=public_tag_exists,
    )


def validate_certification(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility entry point for callers that use the shorter name."""

    if args and isinstance(args[0], Mapping) and not kwargs:
        return validate_certification_identity(args[0])
    return certify_exact_tree(*args, **kwargs)


__all__ = [
    "BLOCKED",
    "INVALID",
    "PASS",
    "SUPPORTED_BINDINGS",
    "certification_binding",
    "certify_exact_tree",
    "is_safe_public_value",
    "validate_certification",
    "validate_certification_identity",
    "validate_certification_transfer",
]
