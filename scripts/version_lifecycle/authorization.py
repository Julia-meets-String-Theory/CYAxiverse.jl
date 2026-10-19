"""Canonical synthetic owner-authority records for reduced Gate A.

The authority is injected by the caller's typed port.  This module never
contacts a production owner service and never treats an intent field as proof
of ownership.  A port must return the current canonical bytes plus an
independent repository-owner assertion; every mutation re-fetches and
revalidates the record while its exclusion is held.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import ipaddress
import re
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

from .codec import canonical_json, sha256_hex
from .certification import has_token_shape, is_safe_public_value
from .versions import parse_package_version


AUTHORIZATION_ID_PREFIX = "AUTH-SHA256-"
AUTHORIZATION_ID_RE = re.compile(r"^AUTH-SHA256-[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version", "owner_authorization", "repository", "owner_account",
        "authority_source_ref", "issued_at_utc", "expires_at_utc",
        "transaction_id", "owner_line", "final_version", "authorized_actions",
        "target_refs", "owner_authorization_digest",
    }
)
_AUTHORITY_COMPONENT_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$"
)
_PRIVATE_REFERENCE_MARKER_RE = re.compile(
    r"(?:credential|password|passwd|secret|token|bearer|api[-_]?key)", re.I
)
_LOCAL_REFERENCE_COMPONENTS = frozenset(
    {"codex", "home", "private", "tmp", "users", "var"}
)


class AuthorizationError(ValueError):
    """A malformed, stale, or nonmatching owner authorization."""


@dataclass(frozen=True, slots=True)
class AuthorizationResolution:
    """Exact current bytes and independent owner identity from an authority."""

    record: Mapping[str, Any]
    canonical_bytes: bytes
    repository_owner_verified: bool


class OwnerAuthorizationAuthority(Protocol):
    def fetch_owner_authorization(self, reference: str) -> Any: ...


def _text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or not value.isascii():
        raise AuthorizationError(f"{field} is not printable ASCII")
    if any(ord(char) < 0x20 or ord(char) > 0x7E for char in value):
        raise AuthorizationError(f"{field} is not printable ASCII")


def validate_authorization_reference(value: Any) -> str:
    """Return one public, immutable owner-authority reference."""

    _text(value, "authority_source_ref")
    assert isinstance(value, str)
    if (
        len(value) > 256
        or not value.startswith("owner-authority://")
        or "?" in value
        or "#" in value
        or "%" in value
        or _PRIVATE_REFERENCE_MARKER_RE.search(value)
        or has_token_shape(value)
    ):
        raise AuthorizationError("authority_source_ref is not public-safe")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise AuthorizationError("authority_source_ref is not canonical") from error
    try:
        authority_ip = ipaddress.ip_address(parsed.netloc)
    except ValueError:
        authority_ip = None
    components = parsed.path.removeprefix("/").split("/")
    local_authority = parsed.netloc.lower() in {
        "localhost", "localhost.localdomain", "local", "private"
    }
    if (
        parsed.scheme != "owner-authority"
        or not parsed.netloc
        or parsed.netloc != parsed.netloc.lower()
        or authority_ip is not None
        or local_authority
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not _AUTHORITY_COMPONENT_RE.fullmatch(parsed.netloc)
        or not parsed.path.startswith("/")
        or not components
        or any(
            component in {"", ".", ".."}
            or component.lower() in _LOCAL_REFERENCE_COMPONENTS
            or _AUTHORITY_COMPONENT_RE.fullmatch(component) is None
            for component in components
        )
    ):
        raise AuthorizationError("authority_source_ref is not canonical")
    return value


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        raise AuthorizationError(f"{field} is not explicit UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise AuthorizationError(f"{field} is not a real UTC timestamp") from error


def _canonical_version(value: Any) -> str:
    try:
        parsed = parse_package_version(value)
    except (TypeError, ValueError) as error:
        raise AuthorizationError("final_version is not canonical") from error
    if not parsed.is_final:
        raise AuthorizationError("final_version must be final")
    return parsed.canonical


def _sorted_strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise AuthorizationError(f"{field} must be a nonempty string array")
    if len(set(value)) != len(value) or value != sorted(value, key=lambda item: item.encode("utf-8")):
        raise AuthorizationError(f"{field} must be sorted and duplicate-free")
    return list(value)


def authorization_id_preimage(record: Mapping[str, Any]) -> bytes:
    """Return the exact record bytes used for the content-derived ID."""

    body = {key: value for key, value in record.items()
            if key not in {"owner_authorization", "owner_authorization_digest"}}
    return canonical_json(body)


def authorization_digest_preimage(record: Mapping[str, Any]) -> bytes:
    """Return the same noncircular bytes used for ID and digest."""

    return authorization_id_preimage(record)


def authorization_id(record: Mapping[str, Any]) -> str:
    return AUTHORIZATION_ID_PREFIX + sha256_hex(authorization_id_preimage(record))


def authorization_digest(record: Mapping[str, Any]) -> str:
    # R-046 uses one noncircular preimage.  The digest and the ID suffix are
    # therefore equal and neither self-referential field is hashed.
    return sha256_hex(authorization_id_preimage(record))


def seal_authorization(record: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one synthetic owner grant with its canonical ID and digest."""

    body = dict(record)
    body.pop("owner_authorization", None)
    body.pop("owner_authorization_digest", None)
    body["owner_authorization"] = authorization_id(body)
    body["owner_authorization_digest"] = authorization_digest(body)
    validate_authorization(body)
    return body


def canonical_authorization_bytes(record: Mapping[str, Any]) -> bytes:
    checked = validate_authorization(record)
    raw = canonical_json(checked)
    if raw.endswith(b"\n"):
        raise AuthorizationError("authorization bytes have trailing LF")
    return raw


def validate_authorization(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise AuthorizationError("authorization record is not an object")
    if set(record) != AUTHORIZATION_FIELDS:
        raise AuthorizationError("authorization record fields are not exact")
    if record["schema_version"] != 1 or isinstance(record["schema_version"], bool):
        raise AuthorizationError("unsupported authorization schema")
    if not AUTHORIZATION_ID_RE.fullmatch(str(record["owner_authorization"])):
        raise AuthorizationError("invalid owner_authorization ID")
    if record["owner_authorization"] != authorization_id(record):
        raise AuthorizationError("owner_authorization ID mismatch")
    if not SHA256_RE.fullmatch(str(record["owner_authorization_digest"])):
        raise AuthorizationError("invalid owner_authorization_digest")
    if record["owner_authorization_digest"] != authorization_digest(record):
        raise AuthorizationError("owner_authorization_digest mismatch")
    for field in ("repository", "owner_account", "transaction_id", "owner_line"):
        _text(record[field], field)
        if not is_safe_public_value(record[field], key=field):
            raise AuthorizationError(f"{field} is not a safe public value")
    validate_authorization_reference(record["authority_source_ref"])
    _canonical_version(record["final_version"])
    issued = _utc(record["issued_at_utc"], "issued_at_utc")
    expires = _utc(record["expires_at_utc"], "expires_at_utc")
    if expires <= issued:
        raise AuthorizationError("authorization expiry must follow issue time")
    _sorted_strings(record["authorized_actions"], "authorized_actions")
    _sorted_strings(record["target_refs"], "target_refs")
    return dict(record)


def _normalise_resolution(value: Any) -> AuthorizationResolution:
    if isinstance(value, AuthorizationResolution):
        if not isinstance(value.repository_owner_verified, bool):
            raise AuthorizationError("owner authority owner assertion is not boolean")
        return value
    if isinstance(value, tuple) and len(value) == 3:
        record, raw, owner_verified = value
        if not isinstance(owner_verified, bool):
            raise AuthorizationError("owner authority owner assertion is not boolean")
        return AuthorizationResolution(record, raw, owner_verified)
    if isinstance(value, Mapping):
        record = value.get("record", value.get("authorization"))
        raw = value.get("canonical_bytes", value.get("bytes"))
        owner_verified = value.get("repository_owner_verified")
        if record is not None and raw is not None and isinstance(owner_verified, bool):
            return AuthorizationResolution(record, raw, owner_verified)
    raise AuthorizationError("owner authority returned no exact resolution")


def verify_owner_authorization(
    authority: Any,
    reference: str,
    *,
    repository: str,
    transaction_id: str,
    action: str,
    owner_line: str,
    final_version: str,
    target_ref: str,
    now_utc: str,
) -> dict[str, Any]:
    """Fetch and verify one current grant for one exact mutation."""

    try:
        fetch = getattr(authority, "fetch_owner_authorization", None)
        if fetch is None and callable(authority):
            fetch = authority
        if fetch is None:
            raise AuthorizationError("owner authority callback is unavailable")
        resolution = _normalise_resolution(fetch(reference))
        record = validate_authorization(resolution.record)
        raw = resolution.canonical_bytes
        if not isinstance(raw, bytes) or raw != canonical_json(record):
            raise AuthorizationError("owner authorization bytes changed or are noncanonical")
        if not resolution.repository_owner_verified:
            raise AuthorizationError("owner account is not repository owner")
        if record["authority_source_ref"] != reference:
            raise AuthorizationError("owner authorization source reference mismatch")
        if record["repository"] != repository:
            raise AuthorizationError("owner authorization repository mismatch")
        if record["transaction_id"] != transaction_id:
            raise AuthorizationError("owner authorization transaction mismatch")
        if record["owner_line"] != owner_line:
            raise AuthorizationError("owner authorization line mismatch")
        if _canonical_version(record["final_version"]) != _canonical_version(final_version):
            raise AuthorizationError("owner authorization version mismatch")
        if action not in record["authorized_actions"]:
            raise AuthorizationError("owner authorization action mismatch")
        if target_ref not in record["target_refs"]:
            raise AuthorizationError("owner authorization target mismatch")
        now = _utc(now_utc, "now_utc")
        issued = _utc(record["issued_at_utc"], "issued_at_utc")
        expires = _utc(record["expires_at_utc"], "expires_at_utc")
        if now < issued or now >= expires:
            raise AuthorizationError("owner authorization is expired or not yet valid")
        return record
    except AuthorizationError:
        raise
    except Exception as error:
        raise AuthorizationError("owner authorization is unverifiable") from error


__all__ = [
    "AUTHORIZATION_FIELDS", "AUTHORIZATION_ID_PREFIX", "AuthorizationError",
    "AuthorizationResolution", "OwnerAuthorizationAuthority",
    "authorization_digest", "authorization_digest_preimage", "authorization_id",
    "authorization_id_preimage", "canonical_authorization_bytes",
    "seal_authorization", "validate_authorization", "verify_owner_authorization",
]
