"""Pure validators for Gate A exact-tree certification evidence.

The release transaction code owns Git operations.  This module only checks
the identity and binding of a certification record, so it can be used with a
real repository, an API response, or a small fixture without mutating refs.
The returned mappings are deliberately JSON serialisable and use the same
``status``/``reason_code`` vocabulary as the lifecycle writers.
"""

from __future__ import annotations

from collections.abc import Mapping
import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit

from .public_ip import parse_ipv4_compat
from .versions import maintenance_line, parse_package_version, parse_public_tag


SUPPORTED_BINDINGS = frozenset(("tree-bound", "commit-bound"))
BLOCKED = "BLOCKED"
INVALID = "INVALID"
PASS = "PASS"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_PUBLIC_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_NONPUBLIC_DNS_SUFFIXES = (
    ".arpa", ".corp", ".example.com", ".example.net", ".example.org",
    ".home", ".internal", ".intranet", ".lan",
    ".local", ".localdomain", ".localhost", ".private",
    ".test", ".example", ".invalid", ".onion",
)
_NONPUBLIC_DNS_NAMES = {"example.com", "example.net", "example.org"}
_PUBLIC_VALUE_TOKEN_SPLIT = re.compile(r"[\s=,;/()\[\]{}<>\"'@\\]+")
_DOTTED_IP_CANDIDATE = re.compile(
    r"(?<![0-9A-Za-z])(?:0[xX][0-9A-Fa-f]+|[0-9]+)"
    r"(?:\.(?:0[xX][0-9A-Fa-f]+|[0-9]+)){1,3}(?![0-9])"
)
_TOKEN_SHAPED_VALUE = re.compile(
    r"(?:"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{16,}|"
    r"sk_(?:live|test|proj)_?[A-Za-z0-9_-]{16,}|"
    r"rk_(?:live|test)_[A-Za-z0-9]{16,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"AIza[A-Za-z0-9_-]{30,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r")",
    re.I,
)
_HTTP_URL_IN_TEXT = re.compile(r"https?://[^\s,;\"'<>]+", re.I)
_PRIVATE_SHARE_URL = re.compile(
    r"(?:chatgpt\.com|chat\.openai\.com)/share(?:/|[?#\s]|$)", re.I
)
_EMBEDDED_UNIX_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_./-])/(?:[^/\s,;\"'<>]+/)*[^/\s,;\"'<>]+"
)
_EMBEDDED_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|~[\\/]|"
    r"\\\\(?:\?\\|\.\\)?[^\\/\s,;\"'<>]+[\\/])"
)
_PUBLIC_ENVIRONMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PUBLIC_ENVIRONMENT_FIELDS = frozenset({
    "schema_version",
    "environment_id",
    "os",
    "architecture",
    "julia_version",
    "python_version",
    "ci_provider",
    "harness_runtime",
    "runner_image_digest",
    "container_image_digest",
})
_PUBLIC_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _version_identity(value: str, key: str | None) -> bool:
    """Recognize typed release identities before interpreting IP aliases."""

    if key is None:
        return False
    try:
        if key == "version" or key.endswith("_version") or key in {
            "closure", "candidate", "anchor", "final_release", "certified", "main"
        }:
            parse_package_version(value)
            return True
        if key == "line" or key.endswith("_line"):
            maintenance_line(value)
            return True
        if key == "public_tag":
            parse_public_tag(value)
            return True
        if key in {"anchor_ref", "closure_anchor", "candidate_ref", "intent_id"}:
            prefix = {
                "anchor_ref": "refs/tags/iterations/",
                "closure_anchor": "refs/tags/iterations/",
                "candidate_ref": "refs/heads/candidates/",
                "intent_id": "INT-",
            }[key]
            return value.startswith(prefix) and parse_package_version(
                value[len(prefix):]
            ).is_final
        if key in {"branch_ref", "line_ref"}:
            if value == "refs/heads/vmm":
                return True
            prefix = "refs/heads/"
            if value.startswith(prefix):
                maintenance_line(value[len(prefix):])
                return True
    except (TypeError, ValueError):
        return False
    return False


def _contains_nonpublic_locator(value: str) -> bool:
    """Find local hosts and IP locators in labels and URL shaped values."""

    # Scan the whole string first. A private address followed by a port,
    # path, punctuation, or DNS suffix is still a private locator.
    for match in _DOTTED_IP_CANDIDATE.finditer(value):
        candidate = match.group()
        address = parse_ipv4_compat(candidate)
        if address is not None and not address.is_global:
            return True

    for token in _PUBLIC_VALUE_TOKEN_SPLIT.split(value):
        if not token:
            continue
        host_label = token.lower().rstrip(".")
        if host_label.count(":") == 1 and host_label.rsplit(":", 1)[1].isdigit():
            host_label = host_label.rsplit(":", 1)[0]
        if (
            host_label in _NONPUBLIC_DNS_NAMES
            or host_label in {"localhost", "localhost.localdomain", "intranet", "internal"}
            or host_label.endswith(_NONPUBLIC_DNS_SUFFIXES)
        ):
            return True
        try:
            address = ipaddress.ip_address(token)
        except ValueError:
            integer_alias = (
                parse_ipv4_compat(token)
                if "." in token or token.lower().startswith("0x")
                or (9 <= len(token) <= 12 and token.isascii() and token.isdecimal())
                else None
            )
            if integer_alias is not None and not integer_alias.is_global:
                return True
            continue
        if not address.is_global:
            return True
    return False


def _contains_embedded_absolute_path(value: str) -> bool:
    """Find Unix, Windows-drive, or home paths embedded in durable labels."""

    # URLs have their own host/path validation below. Remove them before
    # looking for filesystem paths so the ``//`` in a URL is not mistaken for
    # a UNC or Unix path prefix.
    non_url_text = _HTTP_URL_IN_TEXT.sub(" ", value)
    return (
        _EMBEDDED_UNIX_ABSOLUTE_PATH.search(non_url_text) is not None
        or _EMBEDDED_WINDOWS_ABSOLUTE_PATH.search(non_url_text) is not None
    )


def is_safe_public_value(value: Any, *, key: str | None = None) -> bool:
    """Return whether a durable value is safe to expose publicly.

    This lexical gate rejects private locators and credential-like values.
    It is shared by the certification and release validators; it does not
    claim that an otherwise safe value is approved evidence.
    """

    if isinstance(value, Mapping):
        return all(
            isinstance(key, str)
            and is_safe_public_value(key)
            and is_safe_public_value(item, key=key)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(is_safe_public_value(item, key=key) for item in value)
    if not isinstance(value, str):
        return True
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        return False
    if _TOKEN_SHAPED_VALUE.search(value) is not None:
        return False
    if _PRIVATE_SHARE_URL.search(value) is not None:
        return False
    if _contains_embedded_absolute_path(value):
        return False
    if "%" in value:
        # Percent escapes can hide a private host or local path in an
        # otherwise ordinary durable label.
        return False
    lowered = value.lower()
    if value != "v-0.1" and not _version_identity(value, key) and _contains_nonpublic_locator(value):
        return False
    if (
        value.startswith(("/", "~", "\\\\"))
        or _DRIVE_PATH.match(value) is not None
    ):
        return False
    normalized = lowered.replace("\\", "/")
    if normalized.startswith(("file:", "local:", "ssh:")):
        return False
    # A URL-like value with a missing slash must not fall through as an
    # ordinary evidence label; doing so bypasses the host/userinfo checks.
    if re.match(r"^[a-z][a-z0-9+.-]*:", normalized) and not normalized.startswith(
        ("http://", "https://")
    ):
        return False
    if "://" in normalized:
        try:
            parsed = urlsplit(value)
            host = (parsed.hostname or "").lower().rstrip(".")
            username = parsed.username
            password = parsed.password
            port = parsed.port
        except ValueError:
            return False
        if (
            parsed.scheme not in {"http", "https"}
            or username is not None
            or password is not None
            or port is not None
            or "?" in value
            or "#" in value
            or "%" in parsed.netloc
            or "%" in parsed.path
        ):
            return False
        if (
            not host
            or host.endswith(_NONPUBLIC_DNS_SUFFIXES)
            or host in _NONPUBLIC_DNS_NAMES
            or host in {"localhost", "localhost.localdomain", "intranet", "internal"}
        ):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # URL clients can interpret abbreviated and nondecimal dotted
            # hosts as private IPv4 addresses despite ipaddress rejecting them.
            if parse_ipv4_compat(host) is not None:
                return False
            address = None
        if address is not None and not address.is_global:
            return False
        if address is None and (
            len(host) > 253
            or "." not in host
            or not all(_PUBLIC_DNS_LABEL.fullmatch(label) for label in host.split("."))
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


def is_safe_public_reference(value: Any, *, key: str | None = None) -> bool:
    """Validate a durable public reference or an array of such references."""

    if isinstance(value, str):
        return (
            bool(value)
            and "?" not in value
            and "#" not in value
            and is_safe_public_value(value, key=key)
        )
    if isinstance(value, (list, tuple)):
        return all(is_safe_public_reference(item, key=key) for item in value)
    return False


def is_safe_public_environment(value: Any) -> bool:
    """Validate a public environment label or the exact typed v1 object."""

    if isinstance(value, str):
        return (
            _PUBLIC_ENVIRONMENT_ID.fullmatch(value) is not None
            and is_safe_public_value(value, key="certification_environment")
        )
    if not isinstance(value, Mapping):
        return False
    keys = set(value)
    if (
        not {"schema_version", "environment_id"} <= keys
        or not keys <= _PUBLIC_ENVIRONMENT_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
    ):
        return False
    for key, item in value.items():
        if key == "schema_version":
            continue
        if (
            not isinstance(item, str)
            or not item
            or not is_safe_public_value(item, key=key)
        ):
            return False
        if key == "environment_id" and _PUBLIC_ENVIRONMENT_ID.fullmatch(item) is None:
            return False
        if key in {"runner_image_digest", "container_image_digest"} and _PUBLIC_IMAGE_DIGEST.fullmatch(item) is None:
            return False
    return True


def has_token_shape(value: str) -> bool:
    """Return whether a string contains a common credential-shaped token."""

    return _TOKEN_SHAPED_VALUE.search(value) is not None


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

    environment = _value(record, "environment", "environment_id", "environment_ref")
    if not is_safe_public_environment(environment):
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
    "is_safe_public_reference",
    "is_safe_public_environment",
    "validate_certification",
    "validate_certification_identity",
    "validate_certification_transfer",
]
