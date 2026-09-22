"""Immutable lifecycle manifests and create-once Git-ref authority.

The version lifecycle is intentionally modelled as a set of small immutable
objects.  Each object is stored in a commit with one file, ``manifest.json``;
the protected ref is created once and is never used as a mutable head.  This
module contains the dependency-free wire and replay rules.  The network-facing
writer is deliberately small and only performs create-if-absent operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any, Mapping
from urllib.parse import urlsplit

from .codec import canonical_json, sha256_hex
from .authorization import (
    AUTHORIZATION_ID_RE,
    AuthorizationError,
    verify_owner_authorization,
)
from .certification import is_safe_public_value
from .git_refs import (
    canonical_remote_authority,
    GitIdentityError,
    ProtectionEvidence,
    parse_remote_ref_advertisements,
    require_full_ref,
    validate_remote,
)
from .versions import parse_package_version, parse_public_tag


SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
MANIFEST_ID_RE = re.compile(r"^LIF-SHA256-[0-9a-f]{64}$")
PUBLICATION_ID_RE = re.compile(r"^pub-[0-9a-f]{64}$")
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
LIFECYCLE_REF_PREFIX = "refs/heads/lifecycle/v1/"

MANIFEST_TYPES = frozenset(
    {
        "version-claimed",
        "reservation-prepared",
        "reservation-opened",
        "reservation-aborted",
        "reservation-consumed",
        "candidate-opened",
        "candidate-withdrawn",
        "release-intent-prepared",
        "release-intent-aborted",
        "released",
        "publication",
    }
)

_COMMON_REQUIRED = frozenset(
    {
        "schema_version", "manifest_type", "manifest_id", "timestamp_utc",
        "predecessor_refs", "owner_authorization", "owner_authorization_ref",
        "owner_authorization_digest",
    }
)
_ALLOCATION_REQUIRED = frozenset(
    {"transaction_id", "static_iteration_snapshot", "lifecycle_ref_snapshot"}
)
_HEX_FIELDS = frozenset(
    {
        "static_iteration_snapshot",
        "lifecycle_ref_snapshot",
        "evidence_digest",
        "publication_evidence_digest",
        "manifest_digest",
        "released_manifest_digest",
        "owner_authorization_digest",
    }
)
_SHA_FIELDS = frozenset(
    {
        "expected_line_head",
        "actual_dev_head",
        "candidate_sha",
        "candidate_tree",
        "anchor_sha",
        "anchor_tree",
        "final_release_sha",
        "final_release_tree",
        "certification_subject_sha",
        "certification_subject_tree",
        "certified_tree",
        "previous_main_sha",
        "main_at_release_sha",
        "tag_commit",
        "tag_tree",
    }
)
_PUBLICATION_FORBIDDEN = frozenset(
    {
        "owner_line",
        "final_version",
        "reserved_final",
        "intended_dev_version",
        "candidate_ref",
        "candidate_sha",
        "candidate_tree",
        "anchor_ref",
        "anchor_sha",
        "anchor_tree",
        "intent_ref",
        "certification_binding",
        "certification_subject_sha",
        "certification_subject_tree",
        "certification_policy_revision",
        "certification_harness_revision",
        "certification_environment",
        "certification_evidence_refs",
        "previous_main_sha",
        "previous_main_version",
        "closure_timestamp_utc",
    }
)

_COMMON_FIELDS = frozenset(
    {"schema_version", "manifest_type", "manifest_id", "timestamp_utc", "predecessor_refs", "owner_authorization", "owner_authorization_ref", "owner_authorization_digest"}
)
_ALLOCATION_FIELDS = frozenset(
    {"transaction_id", "static_iteration_snapshot", "lifecycle_ref_snapshot"}
)
_TYPE_FIELDS: dict[str, frozenset[str]] = {
    "version-claimed": frozenset({"final_version", "owner_line"}),
    "reservation-prepared": frozenset({"owner_line", "final_version", "intended_dev_version", "expected_line_head", "reserved_final"}),
    "reservation-opened": frozenset({"owner_line", "final_version", "intended_dev_version", "actual_dev_head", "reserved_final"}),
    "reservation-aborted": frozenset({"owner_line", "final_version", "intended_dev_version", "non_entry_evidence", "abort_reason", "reserved_final"}),
    "reservation-consumed": frozenset({"owner_line", "final_version", "terminal_disposition", "closure_anchor_ref", "closed_final_version", "reserved_final"}),
    "candidate-opened": frozenset({"candidate_id", "candidate_ref", "candidate_sha", "candidate_tree", "final_version", "release_line", "anchor_ref", "anchor_sha", "anchor_tree", "main_at_candidate_sha", "main_at_candidate_version"}),
    "candidate-withdrawn": frozenset({"candidate_id", "candidate_ref", "final_version", "release_line", "withdrawal_evidence"}),
    "release-intent-prepared": frozenset({"candidate_id", "candidate_ref", "candidate_sha", "candidate_tree", "final_version", "release_line", "public_tag", "final_release_sha", "final_release_tree", "certification_binding", "certification_subject_sha", "certification_subject_tree", "certification_policy_revision", "certification_harness_revision", "certification_environment", "certification_evidence_refs", "certification_transfer_evidence", "anchor_ref", "anchor_sha", "anchor_tree"}),
    "release-intent-aborted": frozenset({"intent_ref", "candidate_id", "final_version", "no_public_tag_evidence"}),
    "released": frozenset({"final_version", "release_line", "public_tag", "candidate_ref", "candidate_sha", "candidate_tree", "anchor_ref", "anchor_sha", "anchor_tree", "final_release_sha", "final_release_tree", "certification_binding", "certification_subject_sha", "certification_subject_tree", "certification_policy_revision", "certification_harness_revision", "certification_environment", "certification_evidence_refs", "certification_transfer_evidence", "closure_timestamp_utc", "previous_main_sha", "previous_main_version", "main_at_release_sha", "main_at_release_version", "main_at_candidate_sha", "main_at_candidate_version", "evidence_refs"}),
    "publication": frozenset({"publication_id", "released_manifest_ref", "released_manifest_id", "released_manifest_digest", "public_tag", "tag_commit", "tag_tree", "github_release_id", "github_release_url", "published_at_utc", "publication_evidence_ref", "publication_evidence_digest"}),
}


class ManifestError(ValueError):
    """Raised when canonical manifest bytes or identity are invalid."""


class ManifestConflict(ManifestError):
    """A create-once ref exists with a different immutable payload."""


class CreateOutcomeUncertain(RuntimeError):
    """The remote result cannot be classified without weakening authority."""


def _is_sha1(value: Any) -> bool:
    return isinstance(value, str) and SHA1_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _check_timestamp(value: Any, field: str = "timestamp_utc") -> None:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        raise ManifestError(f"{field} must be RFC3339 UTC")
    try:
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ManifestError(f"{field} must be a real UTC timestamp") from error


def _check_ref(value: Any, *, prefix: str | None = None) -> None:
    if not isinstance(value, str):
        raise ManifestError("predecessor refs must be strings")
    try:
        require_full_ref(value)
    except GitIdentityError as error:
        raise ManifestError("invalid lifecycle ref") from error
    if prefix is not None and not value.startswith(prefix):
        raise ManifestError("ref is outside its canonical namespace")


def _check_identity_text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ManifestError(f"{name} must be nonempty printable ASCII")
    if any(ord(char) < 0x20 or ord(char) > 0x7E for char in value):
        raise ManifestError(f"{name} must be nonempty printable ASCII")
    if name not in {"github_release_url", "publication_evidence_ref", "owner_authorization_ref"} and ("/" in value or ".." in value):
        raise ManifestError(f"{name} contains a forbidden ref component")


def _check_publication_evidence_ref(value: Any) -> None:
    """Require a safe repository-relative POSIX evidence path."""

    if not isinstance(value, str) or not value or not value.isascii():
        raise ManifestError("publication_evidence_ref must be a safe relative path")
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(not part or part in {".", ".."} for part in value.split("/"))
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
    ):
        raise ManifestError("publication_evidence_ref must be a safe relative path")


def _check_github_release_url(value: Any, public_tag: str) -> None:
    """Require one canonical public GitHub release-page URL."""

    if not isinstance(value, str) or not value or not value.isascii():
        raise ManifestError("github_release_url must be a sanitized HTTPS URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ManifestError("github_release_url must be a sanitized HTTPS URL") from error
    parts = parsed.path.split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or len(parts) != 6
        or parts[0] != ""
        or parts[3:] != ["releases", "tag", public_tag]
        or any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", component) is None
            for component in parts[1:3]
        )
    ):
        raise ManifestError("github_release_url must be a sanitized HTTPS URL")


def _check_positive_decimal(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{field} must be a positive decimal integer")


def _line_ref_component(value: Any) -> str:
    if value == "principal":
        return "principal"
    if isinstance(value, str) and value.startswith("maintenance/"):
        suffix = value.removeprefix("maintenance/")
        if re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", suffix):
            return f"maintenance-{suffix}"
    raise ManifestError("owner line is not canonical")


def publication_key(public_tag: str, released_manifest_id: str) -> str:
    """Return the exact deterministic publication key preimage digest."""

    try:
        parsed = parse_public_tag(public_tag)
    except (TypeError, ValueError) as error:
        raise ManifestError("publication requires a canonical public tag") from error
    if parsed.canonical != public_tag[1:] or not MANIFEST_ID_RE.fullmatch(released_manifest_id):
        raise ManifestError("publication key has invalid identity")
    preimage = canonical_json(
        {"public_tag": public_tag, "released_manifest_id": released_manifest_id}
    )
    return hashlib.sha256(preimage).hexdigest()


def publication_id(public_tag: str, released_manifest_id: str) -> str:
    return f"pub-{publication_key(public_tag, released_manifest_id)}"


def publication_ref(public_tag: str, released_manifest_id: str) -> str:
    return (
        f"{LIFECYCLE_REF_PREFIX}publications/{public_tag}/"
        f"{publication_id(public_tag, released_manifest_id)}"
    )


def publication_key_fixture() -> dict[str, str]:
    """Return the fixed normative publication-key fixture."""

    released = "LIF-SHA256-" + "a" * 64
    tag = "v1.2.3"
    preimage = canonical_json({"public_tag": tag, "released_manifest_id": released})
    digest = sha256_hex(preimage)
    return {
        "preimage": preimage.decode("ascii"),
        "sha256": digest,
        "publication_id": f"pub-{digest}",
        "ref": publication_ref(tag, released),
    }


def manifest_identity_preimage(manifest: Mapping[str, Any]) -> bytes:
    """Encode the authorization-independent manifest identity fields."""

    if not isinstance(manifest, Mapping):
        raise ManifestError("manifest must be an object")
    excluded = {
        "manifest_id", "owner_authorization", "owner_authorization_ref",
        "owner_authorization_digest",
    }
    body = {key: value for key, value in manifest.items() if key not in excluded}
    try:
        return canonical_json(body)
    except (TypeError, ValueError) as error:
        raise ManifestError(f"manifest is not canonical JSON: {error}") from error


def manifest_id(manifest: Mapping[str, Any]) -> str:
    return f"LIF-SHA256-{sha256_hex(manifest_identity_preimage(manifest))}"


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Add the content-derived ID to a manifest draft."""

    sealed = dict(manifest)
    if sealed.get("manifest_type") == "publication":
        if "public_tag" in sealed and "released_manifest_id" in sealed:
            derived_publication = publication_id(
                str(sealed["public_tag"]), str(sealed["released_manifest_id"])
            )
            if "publication_id" in sealed and sealed["publication_id"] != derived_publication:
                raise ManifestError("publication_id conflicts with its deterministic key")
            sealed["publication_id"] = derived_publication
    derived = manifest_id(sealed)
    if "manifest_id" in sealed and sealed["manifest_id"] != derived:
        raise ManifestError("manifest_id conflicts with canonical manifest content")
    sealed["manifest_id"] = derived
    return sealed


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Validate and encode one standalone no-LF manifest."""

    checked = dict(manifest)
    validate_manifest(checked)
    return canonical_json(checked)


def _validate_publication(manifest: Mapping[str, Any]) -> None:
    required = {
        "publication_id", "released_manifest_ref", "released_manifest_id",
        "released_manifest_digest", "public_tag", "tag_commit", "tag_tree",
        "github_release_id", "github_release_url", "published_at_utc",
        "publication_evidence_ref", "publication_evidence_digest", "owner_authorization",
        "owner_authorization_ref", "owner_authorization_digest",
    }
    missing = required - set(manifest)
    if missing:
        raise ManifestError("publication missing required fields: " + ",".join(sorted(missing)))
    if set(manifest) & _PUBLICATION_FORBIDDEN:
        raise ManifestError("publication contains forbidden release-predecessor fields")
    if not PUBLICATION_ID_RE.fullmatch(str(manifest["publication_id"])):
        raise ManifestError("invalid publication_id")
    if publication_id(str(manifest["public_tag"]), str(manifest["released_manifest_id"])) != manifest["publication_id"]:
        raise ManifestError("publication_id does not match public tag and released manifest")
    expected = publication_ref(str(manifest["public_tag"]), str(manifest["released_manifest_id"]))
    if manifest.get("ref") not in {None, expected}:
        raise ManifestError("publication ref identity mismatch")
    _check_ref(manifest["released_manifest_ref"], prefix=f"{LIFECYCLE_REF_PREFIX}releases/")
    if not MANIFEST_ID_RE.fullmatch(str(manifest["released_manifest_id"])):
        raise ManifestError("invalid released_manifest_id")
    if not _is_sha256(manifest["released_manifest_digest"]):
        raise ManifestError("invalid released_manifest_digest")
    try:
        parsed = parse_public_tag(manifest["public_tag"])
    except (TypeError, ValueError) as error:
        raise ManifestError("invalid public_tag") from error
    if parsed.canonical != str(manifest["public_tag"])[1:]:
        raise ManifestError("noncanonical public_tag")
    for field in ("tag_commit",):
        if not _is_sha1(manifest[field]):
            raise ManifestError(f"{field} must be a full Git SHA")
    if not _is_sha1(manifest["tag_tree"]):
        raise ManifestError("tag_tree must be a full Git SHA")
    _check_timestamp(manifest["published_at_utc"], "published_at_utc")
    _check_positive_decimal(manifest["github_release_id"], "github_release_id")
    _check_identity_text(manifest["owner_authorization"], "owner_authorization")
    if not AUTHORIZATION_ID_RE.fullmatch(manifest["owner_authorization"]):
        raise ManifestError("invalid owner_authorization")
    _check_identity_text(manifest["owner_authorization_ref"], "owner_authorization_ref")
    if not _is_sha256(manifest["owner_authorization_digest"]):
        raise ManifestError("invalid owner_authorization_digest")
    _check_github_release_url(manifest["github_release_url"], manifest["public_tag"])
    _check_publication_evidence_ref(manifest["publication_evidence_ref"])
    if not _is_sha256(manifest["publication_evidence_digest"]):
        raise ManifestError("invalid publication_evidence_digest")
    predecessors = manifest.get("predecessor_refs")
    if not isinstance(predecessors, list) or predecessors != [manifest["released_manifest_ref"]]:
        raise ManifestError("publication must have exactly one released predecessor")


def _validate_type_specific(manifest: Mapping[str, Any]) -> None:
    kind = manifest["manifest_type"]
    required: dict[str, set[str]] = {
        "version-claimed": {"final_version", "owner_line"},
        "reservation-prepared": {"owner_line", "final_version", "intended_dev_version", "expected_line_head"},
        "reservation-opened": {"owner_line", "final_version", "intended_dev_version", "actual_dev_head"},
        "reservation-aborted": {"owner_line", "final_version", "intended_dev_version", "non_entry_evidence"},
        "reservation-consumed": {"owner_line", "final_version", "terminal_disposition", "closure_anchor_ref"},
        "candidate-opened": {"candidate_id", "candidate_ref", "candidate_sha", "candidate_tree", "final_version", "release_line", "anchor_ref", "anchor_sha", "anchor_tree", "main_at_candidate_sha", "main_at_candidate_version"},
        "candidate-withdrawn": {"candidate_id", "candidate_ref", "final_version", "release_line", "withdrawal_evidence"},
        "release-intent-prepared": {"candidate_id", "candidate_ref", "candidate_sha", "candidate_tree", "final_version", "release_line", "public_tag", "final_release_sha", "final_release_tree", "certification_binding", "certification_subject_sha", "certification_subject_tree", "certification_policy_revision", "certification_harness_revision", "certification_environment", "certification_evidence_refs", "anchor_ref", "anchor_sha", "anchor_tree"},
        "release-intent-aborted": {"intent_ref", "candidate_id", "final_version", "no_public_tag_evidence"},
        "released": {"final_version", "release_line", "public_tag", "candidate_ref", "candidate_sha", "candidate_tree", "anchor_ref", "anchor_sha", "anchor_tree", "final_release_sha", "final_release_tree", "certification_binding", "certification_subject_sha", "certification_subject_tree", "certification_policy_revision", "certification_harness_revision", "certification_environment", "certification_evidence_refs", "closure_timestamp_utc", "main_at_release_sha", "main_at_release_version", "main_at_candidate_sha", "main_at_candidate_version", "evidence_refs"},
    }
    if kind == "publication":
        _validate_publication(manifest)
        return
    missing = required[kind] - set(manifest)
    if missing:
        raise ManifestError(f"{kind} missing required fields: " + ",".join(sorted(missing)))
    if "owner_line" in manifest:
        _line_ref_component(manifest["owner_line"])
    if "release_line" in manifest:
        _line_ref_component(manifest["release_line"])
        if manifest["release_line"] != "principal":
            final = parse_package_version(manifest["final_version"])
            expected_line = f"maintenance/{final.major}.{final.minor}"
            if manifest["release_line"] != expected_line:
                raise ManifestError("maintenance release line does not match final version")
    if "candidate_id" in manifest:
        _check_identity_text(manifest["candidate_id"], "candidate_id")
        if CANDIDATE_ID_RE.fullmatch(str(manifest["candidate_id"])) is None:
            raise ManifestError("candidate_id is not canonical")
    for field in ("candidate_ref", "anchor_ref", "intent_ref", "closure_anchor_ref"):
        if field in manifest:
            _check_ref(manifest[field], prefix=None)
    for field in ("reserved_final", "closed_final_version"):
        if field in manifest:
            try:
                parsed = parse_package_version(manifest[field])
            except (TypeError, ValueError) as error:
                raise ManifestError(f"{field} is not a canonical package version") from error
            if parsed.is_dev:
                raise ManifestError(f"{field} must be final")
    for field in (
        "non_entry_evidence", "withdrawal_evidence", "no_public_tag_evidence",
        "certification_transfer_evidence",
    ):
        if field in manifest:
            try:
                canonical_json(manifest[field])
            except (TypeError, ValueError) as error:
                raise ManifestError(f"{field} is not canonical evidence") from error
            if not is_safe_public_value(manifest[field], key=field):
                raise ManifestError(f"{field} contains a nonpublic value")
    if kind in {"candidate-withdrawn", "release-intent-aborted"}:
        field = (
            "withdrawal_evidence"
            if kind == "candidate-withdrawn"
            else "no_public_tag_evidence"
        )
        proof = manifest[field]
        expected_tag_ref = f"refs/tags/v{manifest['final_version']}"
        if (
            not isinstance(proof, Mapping)
            or set(proof) != {"public_tag_ref", "tag_absent"}
            or proof.get("public_tag_ref") != expected_tag_ref
            or proof.get("tag_absent") is not True
        ):
            raise ManifestError(
                f"{field} must bind the exact absent canonical public tag"
            )
    for field in ("abort_reason", "terminal_disposition", "certification_binding"):
        if field in manifest:
            _check_identity_text(manifest[field], field)
    if kind == "released" and manifest["release_line"] == "principal":
        for field in ("previous_main_sha", "previous_main_version"):
            if field not in manifest:
                raise ManifestError(f"principal released manifest missing {field}")
    if kind == "released" and manifest.get("release_line") != "principal":
        if "previous_main_sha" in manifest or "previous_main_version" in manifest:
            raise ManifestError("maintenance released manifest cannot contain previous main claim")
        if (
            manifest["main_at_candidate_sha"] != manifest["main_at_release_sha"]
            or manifest["main_at_candidate_version"]
            != manifest["main_at_release_version"]
        ):
            raise ManifestError(
                "maintenance release changes contemporaneous principal main"
            )
    for field in ("certification_evidence_refs", "evidence_refs"):
        if field in manifest:
            values = manifest[field]
            if not isinstance(values, list) or not values or any(not isinstance(item, str) for item in values):
                raise ManifestError(f"{field} must be a nonempty string array")
            if len(set(values)) != len(values) or values != sorted(values, key=lambda item: item.encode("utf-8")):
                raise ManifestError(f"{field} must be a sorted duplicate-free set array")
            if not is_safe_public_value(values, key=field):
                raise ManifestError(f"{field} contains a nonpublic value")
    for field in (
        "certification_environment", "certification_policy_revision",
        "certification_harness_revision", "github_release_url",
        "publication_evidence_ref",
    ):
        if field in manifest and not is_safe_public_value(manifest[field], key=field):
            raise ManifestError(f"{field} contains a nonpublic value")
    if kind in {"candidate-opened", "candidate-withdrawn", "release-intent-prepared"}:
        expected_candidate_ref = (
            f"refs/heads/candidates/v{manifest['final_version']}/"
            f"{manifest['candidate_id']}"
        )
        if manifest["candidate_ref"] != expected_candidate_ref:
            raise ManifestError("candidate_ref does not match candidate identity")
    if kind in {"candidate-opened", "release-intent-prepared", "released"}:
        expected_anchor_ref = f"refs/tags/iterations/{manifest['final_version']}"
        if manifest["anchor_ref"] != expected_anchor_ref:
            raise ManifestError("anchor_ref does not match final version")


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one manifest and return a detached canonical mapping."""

    if not isinstance(manifest, Mapping):
        raise ManifestError("manifest must be an object")
    if any(not isinstance(key, str) for key in manifest):
        raise ManifestError("manifest keys must be strings")
    keys = set(manifest)
    if not _COMMON_REQUIRED <= keys:
        raise ManifestError("manifest is missing common fields")
    if isinstance(manifest["schema_version"], bool) or manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError("unsupported manifest schema_version")
    kind = manifest["manifest_type"]
    if kind not in MANIFEST_TYPES:
        raise ManifestError("unsupported manifest_type")
    allowed = _COMMON_FIELDS | _TYPE_FIELDS[kind]
    if kind != "publication":
        allowed |= _ALLOCATION_FIELDS
    unknown = keys - allowed
    if unknown:
        raise ManifestError("undeclared manifest fields: " + ",".join(sorted(unknown)))
    if not MANIFEST_ID_RE.fullmatch(str(manifest["manifest_id"])):
        raise ManifestError("invalid manifest_id")
    try:
        derived_manifest_id = manifest_id(manifest)
    except (TypeError, ValueError) as error:
        raise ManifestError("manifest identity is not canonical JSON") from error
    if derived_manifest_id != manifest["manifest_id"]:
        raise ManifestError("manifest_id does not match canonical identity preimage")
    _check_timestamp(manifest["timestamp_utc"])
    predecessors = manifest["predecessor_refs"]
    if not isinstance(predecessors, list) or len(set(predecessors)) != len(predecessors):
        raise ManifestError("predecessor_refs must be a duplicate-free ordered array")
    for ref in predecessors:
        _check_ref(ref, prefix=LIFECYCLE_REF_PREFIX)
    if kind != "publication":
        if "transaction_id" in keys:
            _check_identity_text(manifest["transaction_id"], "transaction_id")
        if kind in {"version-claimed", "reservation-prepared", "reservation-opened", "reservation-aborted", "reservation-consumed", "candidate-opened", "candidate-withdrawn", "release-intent-prepared", "release-intent-aborted", "released"}:
            if not _ALLOCATION_REQUIRED <= keys:
                raise ManifestError("allocation manifest is missing snapshot bindings")
        for field in ("static_iteration_snapshot", "lifecycle_ref_snapshot"):
            if not _is_sha256(manifest[field]):
                raise ManifestError(f"{field} must be a SHA-256 digest")
        _check_identity_text(manifest["owner_authorization"], "owner_authorization")
        if not AUTHORIZATION_ID_RE.fullmatch(manifest["owner_authorization"]):
            raise ManifestError("invalid owner_authorization")
        _check_identity_text(manifest["owner_authorization_ref"], "owner_authorization_ref")
        if not _is_sha256(manifest["owner_authorization_digest"]):
            raise ManifestError("invalid owner_authorization_digest")
    for field in _SHA_FIELDS:
        if field in manifest and not _is_sha1(manifest[field]):
            raise ManifestError(f"{field} must be a full Git SHA")
    for field in _HEX_FIELDS:
        if field in manifest and not _is_sha256(manifest[field]):
            raise ManifestError(f"{field} must be a SHA-256 digest")
    for field in ("closure_timestamp_utc",):
        if field in manifest:
            _check_timestamp(manifest[field], field)
    for field in (
        "final_version", "previous_main_version", "main_at_release_version",
        "main_at_candidate_version",
    ):
        if field in manifest:
            try:
                parsed = parse_package_version(manifest[field])
            except (TypeError, ValueError) as error:
                raise ManifestError(f"{field} is not a canonical package version") from error
            if parsed.is_dev:
                raise ManifestError(f"{field} must be final")
    if "intended_dev_version" in manifest:
        try:
            intended = parse_package_version(manifest["intended_dev_version"])
        except (TypeError, ValueError) as error:
            raise ManifestError("intended_dev_version is not canonical") from error
        if not intended.is_dev or "final_version" in manifest and intended.final.canonical != manifest["final_version"]:
            raise ManifestError("intended_dev_version does not match final_version")
    if "public_tag" in manifest:
        try:
            tag = parse_public_tag(manifest["public_tag"])
        except (TypeError, ValueError) as error:
            raise ManifestError("public_tag is not canonical") from error
        if tag.canonical != str(manifest["public_tag"])[1:]:
            raise ManifestError("public_tag is not canonical")
        if "final_version" in manifest and tag.canonical != manifest["final_version"]:
            raise ManifestError("public_tag does not match final_version")
    _validate_type_specific(manifest)
    return dict(manifest)


def lifecycle_ref_for_manifest(manifest: Mapping[str, Any]) -> str:
    """Derive the only valid protected ref for a manifest."""

    checked = validate_manifest(manifest)
    kind = checked["manifest_type"]
    if kind == "version-claimed":
        return f"{LIFECYCLE_REF_PREFIX}claims/v{checked['final_version']}"
    if kind.startswith("reservation-"):
        owner = _line_ref_component(checked["owner_line"])
        dev = str(checked.get("intended_dev_version", checked["final_version"] + "-DEV"))
        return f"{LIFECYCLE_REF_PREFIX}reservations/{owner}/v{dev}/{checked['manifest_id']}"
    if kind.startswith("candidate-"):
        if kind == "candidate-withdrawn":
            return f"{LIFECYCLE_REF_PREFIX}candidates/v{checked['final_version']}/{checked['candidate_id']}/{checked['manifest_id']}"
        return f"{LIFECYCLE_REF_PREFIX}candidates/v{checked['final_version']}/{checked['candidate_id']}"
    if kind.startswith("release-intent-"):
        return f"{LIFECYCLE_REF_PREFIX}intents/v{checked['final_version']}/{checked['candidate_id']}/{checked['manifest_id']}"
    if kind == "released":
        return f"{LIFECYCLE_REF_PREFIX}releases/v{checked['final_version']}"
    if kind == "publication":
        return publication_ref(str(checked["public_tag"]), str(checked["released_manifest_id"]))
    raise ManifestError("unsupported manifest ref namespace")


def _git(repository: Path, *args: str, input_bytes: bytes | None = None, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *args], input=input_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if check and result.returncode:
        raise ManifestError(result.stderr.decode("utf-8", errors="replace").strip() or "git command failed")
    return result.stdout


def _resolve_tree(repository: Path, commit: str) -> str:
    value = _git(repository, "rev-parse", "--verify", f"{commit}^{{tree}}").decode().strip()
    if not SHA1_RE.fullmatch(value):
        raise ManifestError("manifest commit has invalid tree")
    return value


def _manifest_from_commit(repository: Path, commit: str) -> tuple[bytes, str]:
    tree = _resolve_tree(repository, commit)
    entries = _git(repository, "ls-tree", tree).decode("ascii").splitlines()
    if len(entries) != 1:
        raise ManifestError("lifecycle manifest commit must contain exactly manifest.json")
    fields = entries[0].split("\t")
    if len(fields) != 2 or fields[1] != MANIFEST_FILE:
        raise ManifestError("lifecycle manifest commit must contain exactly manifest.json")
    # Keep parsing explicit: a tree entry must be a regular blob with a full
    # object identity, never a symlink, submodule, or nested tree.
    mode_fields = fields[0].split(" ")
    if len(mode_fields) != 3 or mode_fields[0] != "100644" or mode_fields[1] != "blob" or not SHA1_RE.fullmatch(mode_fields[2]):
        raise ManifestError("lifecycle manifest commit must contain exactly manifest.json")
    manifest_blob = mode_fields[2]
    raw = _git(repository, "show", f"{commit}:{MANIFEST_FILE}")
    if not raw or raw.endswith(b"\n"):
        raise ManifestError("manifest bytes must not have a trailing LF")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ManifestError("manifest bytes are not valid JSON") from error
    if canonical_json(value) != raw:
        raise ManifestError("manifest bytes are not canonical JSON")
    actual_blob = _git(repository, "hash-object", "--stdin", input_bytes=raw).decode("ascii").strip()
    if actual_blob != manifest_blob:
        raise ManifestError("manifest blob does not match exact canonical bytes")
    validate_manifest(value)
    return raw, tree


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class LifecycleRefSnapshot:
    data: Mapping[str, Any]
    _authority_token: object | None = field(default=None, repr=False, compare=False)
    _repository_authority: str | None = field(default=None, repr=False, compare=False)

    @property
    def status(self) -> str:
        return "READY"

    @property
    def snapshot_digest(self) -> str:
        return str(self.data["lifecycle_snapshot_digest"])

    @property
    def ref_set_digest(self) -> str:
        return str(self.data["lifecycle_ref_set_digest"])

    @property
    def occupied_versions(self) -> tuple[str, ...]:
        return tuple(self.data["occupied_versions"])

    @property
    def ref_bindings(self) -> tuple[Mapping[str, str], ...]:
        return tuple(self.data["lifecycle_ref_bindings"])

    @property
    def source_repository(self) -> str:
        return str(self.data["source_repository"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    @property
    def authority_verified(self) -> bool:
        return (
            self._authority_token is _LIFECYCLE_AUTHORITY_TOKEN
            and isinstance(self._repository_authority, str)
            and bool(self._repository_authority)
        )


_LIFECYCLE_AUTHORITY_TOKEN = object()


def _has_verified_lifecycle_authority(snapshot: LifecycleRefSnapshot) -> bool:
    """Return true only for snapshots built from the remote authority."""

    return snapshot.authority_verified


def _mark_lifecycle_authority_verified(
    snapshot: LifecycleRefSnapshot,
    repository_authority: str,
) -> LifecycleRefSnapshot:
    """Attach the private authority marker after authoritative verification."""

    if not isinstance(repository_authority, str) or not repository_authority:
        raise ManifestError("lifecycle repository authority is unavailable")
    return LifecycleRefSnapshot(
        snapshot.data, _LIFECYCLE_AUTHORITY_TOKEN, repository_authority
    )


def _version_values(manifest: Mapping[str, Any]) -> set[str]:
    """Derive occupation from typed state, never from incidental fields."""

    if manifest.get("manifest_type") == "reservation-aborted":
        return set()
    occupied: set[str] = set()
    kind = manifest.get("manifest_type")
    for key in ("final_version", "reserved_final", "closed_final_version"):
        value = manifest.get(key)
        if isinstance(value, str):
            try:
                occupied.add(parse_package_version(value).final.canonical)
            except (TypeError, ValueError):
                raise ManifestError(f"{key} is not a canonical occupied version")
    if kind == "publication" and isinstance(manifest.get("public_tag"), str):
        occupied.add(parse_public_tag(manifest["public_tag"]).canonical)
    return occupied


_PREDECESSOR_TYPES: dict[str, frozenset[str]] = {
    "version-claimed": frozenset({"reservation-prepared", "reservation-opened"}),
    "reservation-prepared": frozenset({"reservation-consumed"}),
    "reservation-opened": frozenset({"reservation-prepared"}),
    "reservation-aborted": frozenset({"reservation-prepared"}),
    "reservation-consumed": frozenset({"reservation-opened", "reservation-prepared"}),
    "candidate-opened": frozenset({"version-claimed"}),
    "candidate-withdrawn": frozenset({"candidate-opened"}),
    "release-intent-prepared": frozenset({"candidate-opened"}),
    "release-intent-aborted": frozenset({"release-intent-prepared"}),
    "released": frozenset({"release-intent-prepared"}),
    "publication": frozenset({"released"}),
}
_PREDECESSOR_COUNTS: dict[str, frozenset[int]] = {
    "version-claimed": frozenset({1}),
    "reservation-prepared": frozenset({0, 1}),
    "reservation-opened": frozenset({1}),
    "reservation-aborted": frozenset({1}),
    "reservation-consumed": frozenset({1}),
    "candidate-opened": frozenset({1}),
    "candidate-withdrawn": frozenset({1}),
    "release-intent-prepared": frozenset({1}),
    "release-intent-aborted": frozenset({1}),
    "released": frozenset({1}),
    "publication": frozenset({1}),
}
_TERMINAL_TYPES = frozenset(
    {"reservation-aborted", "reservation-consumed", "candidate-withdrawn", "release-intent-aborted", "released", "publication"}
)
_ACTIVE_TYPES = frozenset(
    {"version-claimed", "reservation-prepared", "reservation-opened", "candidate-opened", "release-intent-prepared"}
)


@dataclass(frozen=True, slots=True)
class LifecycleGraph:
    """The validated complete immutable lifecycle graph."""

    manifests_by_ref: Mapping[str, Mapping[str, Any]]
    occupied_versions: tuple[str, ...]

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifests_by_ref))


def _normalise_graph_records(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any] | None]]:
    normalized: list[tuple[str, Mapping[str, Any], Mapping[str, Any] | None]] = []
    if isinstance(records, Mapping):
        iterable: Iterable[tuple[str | None, Any]] = records.items()
    else:
        iterable = ((None, item) for item in records)
    for key, raw in iterable:
        binding: Mapping[str, Any] | None = None
        if isinstance(raw, Mapping) and isinstance(raw.get("manifest"), Mapping):
            binding = raw
            manifest = raw["manifest"]
            ref = raw.get("ref", key)
        else:
            manifest = raw
            ref = key
        if not isinstance(manifest, Mapping) or not isinstance(ref, str):
            raise ManifestError("lifecycle graph records require ref and manifest")
        normalized.append((ref, manifest, binding))
    return normalized


def validate_lifecycle_graph(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    allocation_occupied_by_snapshot: Mapping[tuple[str, str], Iterable[str]] | None = None,
) -> LifecycleGraph:
    """Replay every protected ref and reject an incomplete or contradictory graph.

    ``records`` accepts either ``{ref: manifest}`` or records containing a
    ``ref`` and ``manifest`` pair.  A record may also carry commit/tree and
    manifest-digest fields; when present, each is checked against the manifest.
    """

    entries = _normalise_graph_records(records)
    by_ref: dict[str, Mapping[str, Any]] = {}
    by_id: dict[str, str] = {}
    publication_pairs: dict[tuple[str, str], str] = {}
    github_ids: dict[int, str] = {}
    for ref, raw_manifest, binding in entries:
        _check_ref(ref, prefix=LIFECYCLE_REF_PREFIX)
        if ref in by_ref:
            raise ManifestError("duplicate lifecycle ref identity")
        manifest = validate_manifest(raw_manifest)
        expected_ref = lifecycle_ref_for_manifest(manifest)
        if ref != expected_ref:
            raise ManifestError("lifecycle ref does not match manifest identity")
        identity = str(manifest["manifest_id"])
        if identity in by_id:
            raise ManifestError("duplicate manifest ID")
        if binding is not None:
            if "manifest_id" in binding and binding["manifest_id"] != manifest["manifest_id"]:
                raise ManifestError("manifest ID does not match lifecycle binding")
            if "manifest_digest" in binding and binding["manifest_digest"] != sha256_hex(canonical_manifest_bytes(manifest)):
                raise ManifestError("manifest digest does not match canonical bytes")
            for field in ("commit", "tree"):
                if field in binding and not _is_sha1(binding[field]):
                    raise ManifestError(f"invalid lifecycle binding {field}")
        by_ref[ref] = manifest
        by_id[identity] = ref
        if manifest["manifest_type"] == "publication":
            pair = (str(manifest["public_tag"]), str(manifest["released_manifest_id"]))
            if pair in publication_pairs:
                raise ManifestError("duplicate publication pair")
            publication_pairs[pair] = ref
            github_id = manifest["github_release_id"]
            if github_id in github_ids:
                raise ManifestError("duplicate GitHub Release identity")
            github_ids[github_id] = ref

    # Validate all predecessor edges only after the complete ref set is known.
    children: dict[str, list[str]] = {ref: [] for ref in by_ref}
    for ref, manifest in by_ref.items():
        kind = str(manifest["manifest_type"])
        predecessors = manifest["predecessor_refs"]
        if len(predecessors) not in _PREDECESSOR_COUNTS[kind]:
            raise ManifestError(
                f"{kind} has invalid predecessor cardinality"
            )
        allowed = _PREDECESSOR_TYPES[kind]
        seen: set[str] = set()
        for predecessor in predecessors:
            if predecessor in seen or predecessor == ref:
                raise ManifestError("duplicate or self predecessor")
            seen.add(predecessor)
            previous = by_ref.get(predecessor)
            if previous is None:
                raise ManifestError("predecessor ref is not present in the complete graph")
            previous_kind = str(previous["manifest_type"])
            if previous_kind not in allowed:
                raise ManifestError("predecessor manifest type is invalid for transition")
            if (
                previous_kind in _TERMINAL_TYPES
                and kind in _ACTIVE_TYPES
                and not (
                    previous_kind == "reservation-consumed"
                    and kind == "reservation-prepared"
                )
            ):
                raise ManifestError("terminal-to-active lifecycle reversal")
            children[predecessor].append(ref)
            _validate_transition_identity(
                previous,
                manifest,
                previous_kind,
                kind,
                predecessor,
                allocation_occupied_by_snapshot=allocation_occupied_by_snapshot,
            )
    for predecessor, child_refs in children.items():
        if len(child_refs) <= 1:
            continue
        child_types = [str(by_ref[child]["manifest_type"]) for child in child_refs]
        allowed_reservation_fork = (
            by_ref[predecessor]["manifest_type"] == "reservation-opened"
            and sorted(child_types) == ["reservation-consumed", "version-claimed"]
        )
        if not allowed_reservation_fork:
            raise ManifestError("lifecycle graph has a branching predecessor")

    for manifest in by_ref.values():
        kind = str(manifest["manifest_type"])
        predecessors = manifest["predecessor_refs"]
        if kind == "publication":
            released_ref = str(manifest["released_manifest_ref"])
            released = by_ref.get(released_ref)
            if released is None or released["manifest_type"] != "released":
                raise ManifestError("publication released predecessor is absent or not released")
            if released["manifest_id"] != manifest["released_manifest_id"]:
                raise ManifestError("publication released manifest ID mismatch")
            if sha256_hex(canonical_manifest_bytes(released)) != manifest["released_manifest_digest"]:
                raise ManifestError("publication released manifest digest mismatch")
            if released.get("public_tag") != manifest["public_tag"]:
                raise ManifestError("publication public tag mismatch")
            if released.get("final_release_sha") != manifest["tag_commit"]:
                raise ManifestError("publication release commit mismatch")
            if released.get("final_release_tree") != manifest["tag_tree"]:
                raise ManifestError("publication release tree mismatch")
        if kind == "released":
            for predecessor in predecessors:
                if by_ref[predecessor]["manifest_type"] != "release-intent-prepared":
                    raise ManifestError("released manifest must follow a prepared intent")

    # Singleton and active-state constraints are graph properties, not ref-name checks.
    singleton: dict[tuple[str, str], str] = {}
    active_reservations: dict[str, str] = {}
    for ref, manifest in by_ref.items():
        kind = str(manifest["manifest_type"])
        version = manifest.get("final_version")
        if not isinstance(version, str):
            version = None
        if kind in {"version-claimed", "released"} and version is not None:
            key = (kind, version)
            if key in singleton:
                raise ManifestError("duplicate singleton lifecycle identity")
            singleton[key] = ref
        if kind in {"reservation-prepared", "reservation-opened"} and not any(
            by_ref[child]["manifest_type"] in _ACTIVE_TYPES
            for child in children[ref]
        ) and not any(
            by_ref[child]["manifest_type"] in {"reservation-aborted", "reservation-consumed"}
            for child in children[ref]
        ):
            key = str(manifest["owner_line"])
            if key in active_reservations:
                raise ManifestError("owner line has multiple active reservations")
            active_reservations[key] = ref
    pre_entry_aborted = {
        predecessor
        for manifest in by_ref.values()
        if manifest["manifest_type"] == "reservation-aborted"
        for predecessor in manifest["predecessor_refs"]
    }
    occupied: set[str] = set()
    for ref, manifest in by_ref.items():
        if ref in pre_entry_aborted:
            continue
        occupied.update(_version_values(manifest))
    return LifecycleGraph(by_ref, tuple(sorted(occupied)))


def _validate_transition_identity(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    previous_kind: str,
    current_kind: str,
    predecessor_ref: str,
    *,
    allocation_occupied_by_snapshot: Mapping[tuple[str, str], Iterable[str]] | None,
) -> None:
    """Check lineage identity in addition to the predecessor type."""

    def same(field: str) -> None:
        if field in previous and field in current and previous[field] != current[field]:
            raise ManifestError(f"{current_kind} changes predecessor {field}")

    for field in ("final_version", "owner_line", "intended_dev_version"):
        if previous_kind.startswith("reservation") and field in previous and field in current:
            if (
                previous_kind == "reservation-consumed"
                and current_kind == "reservation-prepared"
                and field == "final_version"
            ):
                continue
            same(field)
    if previous_kind == "reservation-consumed" and current_kind == "reservation-prepared":
        previous_version = parse_package_version(str(previous["final_version"]))
        current_version = parse_package_version(str(current["final_version"]))
        owner_line = str(current["owner_line"])
        if owner_line == "principal":
            if (
                current_version.major != previous_version.major
                or current_version.minor != previous_version.minor
                or current_version.patch != previous_version.patch + 1
            ):
                raise ManifestError("next principal reservation is not the exact sentinel")
        else:
            match = re.fullmatch(r"maintenance/(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", owner_line)
            if match is None:
                raise ManifestError("maintenance reservation owner line is invalid")
            line = (int(match.group(1)), int(match.group(2)))
            key = (
                str(current["static_iteration_snapshot"]),
                str(current["lifecycle_ref_snapshot"]),
            )
            if allocation_occupied_by_snapshot is None or key not in allocation_occupied_by_snapshot:
                raise ManifestError("maintenance allocation view is unavailable")
            occupied = {
                parse_package_version(value).final.canonical
                for value in allocation_occupied_by_snapshot[key]
            }
            patch = previous_version.patch + 1
            while f"{line[0]}.{line[1]}.{patch}" in occupied:
                patch += 1
            if (
                current_version.major,
                current_version.minor,
                current_version.patch,
            ) != (line[0], line[1], patch):
                raise ManifestError(
                    "next maintenance reservation is not the lowest available patch"
                )
    if current_kind == "version-claimed" and previous_kind.startswith("reservation"):
        same("final_version")
    if current_kind == "candidate-opened":
        same("final_version")
    if current_kind == "candidate-withdrawn":
        for field in ("candidate_id", "candidate_ref", "final_version", "release_line"):
            same(field)
    if current_kind == "release-intent-prepared":
        for field in (
            "candidate_id", "candidate_ref", "candidate_sha", "candidate_tree",
            "final_version", "release_line", "anchor_ref", "anchor_sha", "anchor_tree",
        ):
            same(field)
    if current_kind == "release-intent-aborted":
        if current.get("intent_ref") != predecessor_ref:
            raise ManifestError("release-intent-aborted intent_ref does not name predecessor")
        for field in ("candidate_id", "final_version"):
            same(field)
    if current_kind == "released":
        for field in (
            "candidate_id", "candidate_ref", "candidate_sha", "candidate_tree",
            "final_version", "release_line", "public_tag", "final_release_sha",
            "final_release_tree", "anchor_ref", "anchor_sha", "anchor_tree",
            "certification_binding", "certification_subject_sha", "certification_subject_tree",
            "certification_policy_revision", "certification_harness_revision",
            "certification_environment", "certification_evidence_refs",
        ):
            same(field)
        marker = object()
        if previous.get("certification_transfer_evidence", marker) != current.get(
            "certification_transfer_evidence", marker
        ):
            raise ManifestError(
                "released changes predecessor certification_transfer_evidence"
            )


# Descriptive aliases make the complete-ref requirement explicit to callers.
validate_complete_lifecycle_refs = validate_lifecycle_graph
replay_lifecycle_graph = validate_lifecycle_graph


def validate_lifecycle_ref_snapshot(
    snapshot: LifecycleRefSnapshot | Mapping[str, Any],
    *,
    graph_records: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    expected_occupied_versions: Iterable[str] | None = None,
) -> LifecycleRefSnapshot:
    authority_verified = (
        isinstance(snapshot, LifecycleRefSnapshot)
        and _has_verified_lifecycle_authority(snapshot)
    )
    data = snapshot.to_dict() if isinstance(snapshot, LifecycleRefSnapshot) else dict(snapshot)
    required = {"snapshot_schema_version", "source_repository", "lifecycle_ref_bindings", "lifecycle_ref_set_digest", "occupied_versions", "lifecycle_snapshot_digest"}
    if set(data) != required:
        raise ManifestError("lifecycle snapshot fields are incomplete or noncanonical")
    if data["snapshot_schema_version"] != SCHEMA_VERSION:
        raise ManifestError("unsupported lifecycle snapshot schema")
    if not isinstance(data["source_repository"], str) or not data["source_repository"] or not data["source_repository"].isascii():
        raise ManifestError("lifecycle snapshot source repository is required")
    refs = data["lifecycle_ref_bindings"]
    if not isinstance(refs, list) or any(not isinstance(item, Mapping) for item in refs) or refs != sorted(refs, key=lambda item: item["ref"]):
        raise ManifestError("lifecycle refs must be sorted")
    if len({item.get("ref") for item in refs}) != len(refs):
        raise ManifestError("duplicate lifecycle refs")
    for item in refs:
        if set(item) != {"ref", "commit", "tree", "manifest_id", "manifest_digest"}:
            raise ManifestError("invalid lifecycle ref binding")
        _check_ref(item["ref"], prefix=LIFECYCLE_REF_PREFIX)
        if not _is_sha1(item["commit"]) or not _is_sha1(item["tree"]):
            raise ManifestError("lifecycle ref binding has invalid Git identity")
        if not MANIFEST_ID_RE.fullmatch(item["manifest_id"]) or not _is_sha256(item["manifest_digest"]):
            raise ManifestError("lifecycle ref binding has invalid manifest identity")
    ref_digest = sha256_hex(canonical_json(refs))
    if ref_digest != data["lifecycle_ref_set_digest"]:
        raise ManifestError("lifecycle ref-set digest mismatch")
    occupied = data["occupied_versions"]
    if not isinstance(occupied, list) or occupied != sorted(occupied) or len(set(occupied)) != len(occupied):
        raise ManifestError("occupied versions must be sorted and unique")
    for value in occupied:
        try:
            parsed = parse_package_version(value)
        except (TypeError, ValueError) as error:
            raise ManifestError("occupied versions must be canonical package versions") from error
        if parsed.is_dev or parsed.canonical != value:
            raise ManifestError("occupied versions must be canonical final versions")
    if graph_records is not None:
        graph = validate_lifecycle_graph(graph_records)
        if list(graph.occupied_versions) != occupied:
            raise ManifestError("occupied versions do not match validated lifecycle graph")
    if expected_occupied_versions is not None:
        expected = sorted(set(expected_occupied_versions))
        if expected != occupied:
            raise ManifestError("occupied versions do not match derived lifecycle state")
    preimage = {key: value for key, value in data.items() if key != "lifecycle_snapshot_digest"}
    if sha256_hex(canonical_json(preimage)) != data["lifecycle_snapshot_digest"]:
        raise ManifestError("lifecycle snapshot digest mismatch")
    validated = LifecycleRefSnapshot(data)
    if authority_verified:
        return _mark_lifecycle_authority_verified(
            validated, str(snapshot._repository_authority)
        )
    return validated


def _validate_lifecycle_commit_topology(
    repository: str | Path,
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    root_parent_commit: str,
) -> None:
    root = Path(repository)
    if not isinstance(root_parent_commit, str) or not SHA1_RE.fullmatch(
        root_parent_commit
    ):
        raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
    _git(root, "cat-file", "-e", f"{root_parent_commit}^{{commit}}")
    normalized = _normalise_graph_records(records)
    commits: dict[str, str] = {}
    manifests: dict[str, Mapping[str, Any]] = {}
    for ref, manifest, binding in normalized:
        commit = None if binding is None else binding.get("commit")
        if not isinstance(commit, str) or not SHA1_RE.fullmatch(commit):
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
        commits[ref] = commit
        manifests[ref] = manifest
    for ref, manifest in manifests.items():
        predecessors = manifest.get("predecessor_refs")
        if not isinstance(predecessors, list):
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
        if not predecessors:
            expected_parent = root_parent_commit
        elif len(predecessors) == 1 and predecessors[0] in commits:
            expected_parent = commits[predecessors[0]]
        else:
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
        commit = commits[ref]
        parents = _git(
            root, "rev-list", "--parents", "-n", "1", commit
        ).decode("ascii").strip().split()
        if parents != [commit, expected_parent]:
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")


def build_lifecycle_ref_snapshot(
    repository: str | Path,
    remote: str = "origin",
    *,
    source_repository: str | None = None,
    root_parent_commit: str,
) -> LifecycleRefSnapshot:
    """Resolve and validate the complete protected lifecycle-ref namespace."""

    root = Path(repository)
    remote = validate_remote(remote)
    repository_authority = canonical_remote_authority(root, remote)
    output = _git(root, "ls-remote", "--refs", remote, f"{LIFECYCLE_REF_PREFIX}*")
    advertisements = parse_remote_ref_advertisements(output)
    bindings: list[dict[str, str]] = []
    graph_records: list[dict[str, Any]] = []
    for ref, commit in sorted(advertisements.items()):
        if not ref.startswith(LIFECYCLE_REF_PREFIX) or not SHA1_RE.fullmatch(commit):
            raise ManifestError("invalid lifecycle ref advertisement")
        # The authority repository may be a fresh isolated bare reader. Fetch
        # only the advertised immutable ref into that disposable object store;
        # never write FETCH_HEAD or mutate the inspected checkout.
        _git(root, "fetch", "--no-tags", "--no-write-fetch-head", remote, ref)
        raw, tree = _manifest_from_commit(root, commit)
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        expected_ref = lifecycle_ref_for_manifest(value)
        if expected_ref != ref:
            raise ManifestError("lifecycle ref does not match manifest identity")
        binding = {"ref": ref, "commit": commit, "tree": tree, "manifest_id": value["manifest_id"], "manifest_digest": sha256_hex(raw)}
        bindings.append(binding)
        graph_records.append({"ref": ref, "manifest": value, **binding})
    final_output = _git(
        root, "ls-remote", "--refs", remote, f"{LIFECYCLE_REF_PREFIX}*"
    )
    if parse_remote_ref_advertisements(final_output) != advertisements:
        raise ManifestError("lifecycle ref namespace changed during snapshot")
    graph = validate_lifecycle_graph(graph_records)
    _validate_lifecycle_commit_topology(
        root, graph_records, root_parent_commit
    )
    data: dict[str, Any] = {
        "snapshot_schema_version": SCHEMA_VERSION,
        "source_repository": source_repository or "configured",
        "lifecycle_ref_bindings": bindings,
        "lifecycle_ref_set_digest": sha256_hex(canonical_json(bindings)),
        "occupied_versions": list(graph.occupied_versions),
    }
    data["lifecycle_snapshot_digest"] = sha256_hex(canonical_json(data))
    validated = validate_lifecycle_ref_snapshot(
        data,
        graph_records=graph_records,
        expected_occupied_versions=graph.occupied_versions,
    )
    return _mark_lifecycle_authority_verified(validated, repository_authority)


def lifecycle_ref_snapshot(*args: Any, **kwargs: Any) -> LifecycleRefSnapshot:
    return build_lifecycle_ref_snapshot(*args, **kwargs)


def make_manifest_commit(repository: str | Path, manifest: Mapping[str, Any], *, parent: str | None = None, message: str | None = None) -> str:
    """Create a local one-file manifest commit without creating a ref."""

    root = Path(repository)
    raw = canonical_manifest_bytes(manifest)
    blob = _git(root, "hash-object", "-w", "--stdin", input_bytes=raw).decode().strip()
    index_tree = _git(root, "mktree", input_bytes=f"100644 blob {blob}\t{MANIFEST_FILE}\n".encode()).decode().strip()
    args = ["commit-tree", index_tree]
    if parent is not None:
        if not SHA1_RE.fullmatch(parent):
            raise ManifestError("invalid manifest commit parent")
        args.extend(["-p", parent])
    commit = _git(root, *args, input_bytes=(message or "lifecycle manifest\n").encode()).decode().strip()
    if not SHA1_RE.fullmatch(commit):
        raise ManifestError("Git did not return a commit identity")
    _manifest_from_commit(root, commit)
    return commit


@dataclass(frozen=True, slots=True)
class CreateResult:
    status: str
    ref: str
    commit: str
    tree: str
    manifest_id: str


def _snapshot_components(snapshot: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    static_snapshot = getattr(snapshot, "static_snapshot", None)
    lifecycle_snapshot = getattr(snapshot, "lifecycle_ref_snapshot", None)
    from .static import _has_verified_authority

    if (
        static_snapshot is None
        or lifecycle_snapshot is None
        or not _has_verified_authority(static_snapshot)
        or not _has_verified_lifecycle_authority(lifecycle_snapshot)
        or static_snapshot._repository_authority
        != lifecycle_snapshot._repository_authority
    ):
        raise ManifestError("COMBINED_AUTHORITY_SNAPSHOT_UNVERIFIED")
    return static_snapshot.to_dict(), lifecycle_snapshot.to_dict()


def _snapshot_fingerprint(snapshot: Any) -> bytes:
    static_data, lifecycle_data = _snapshot_components(snapshot)
    data = {
        "static_snapshot": static_data,
        "lifecycle_ref_snapshot": lifecycle_data,
    }
    try:
        return canonical_json(data)
    except (TypeError, ValueError) as error:
        raise ManifestError("lifecycle snapshot callback returned noncanonical data") from error


def _snapshot_fingerprint_from_data(
    static_data: Mapping[str, Any], lifecycle_data: Mapping[str, Any]
) -> bytes:
    try:
        return canonical_json({
            "static_snapshot": dict(static_data),
            "lifecycle_ref_snapshot": dict(lifecycle_data),
        })
    except (TypeError, ValueError) as error:
        raise ManifestError("COMBINED_AUTHORITY_SNAPSHOT_REQUIRED")


def _successor_snapshot_fingerprint(
    snapshot: Any,
    binding: Mapping[str, str],
    occupied_versions: Iterable[str],
) -> bytes:
    static_data, lifecycle_data = _snapshot_components(snapshot)
    bindings = [dict(item) for item in lifecycle_data["lifecycle_ref_bindings"]]
    if any(item["ref"] == binding["ref"] for item in bindings):
        raise ManifestError("CREATE_ONCE_CONFLICT")
    bindings.append(dict(binding))
    bindings.sort(key=lambda item: item["ref"])
    successor = dict(lifecycle_data)
    successor["lifecycle_ref_bindings"] = bindings
    successor["lifecycle_ref_set_digest"] = sha256_hex(canonical_json(bindings))
    successor["occupied_versions"] = sorted(set(occupied_versions))
    successor.pop("lifecycle_snapshot_digest", None)
    successor["lifecycle_snapshot_digest"] = sha256_hex(canonical_json(successor))
    return _snapshot_fingerprint_from_data(static_data, successor)


class CreateOnlyLifecycleWriter:
    """Create protected lifecycle refs with exact retry semantics."""

    def __init__(
        self,
        repository: str | Path,
        remote: str = "origin",
        *,
        exclusion_lease: Callable[[], Any],
        snapshot_callback: Callable[[], Any],
        expected_snapshot: Any,
        owner_authorization_authority: Any,
        authorization_reference: Callable[[str, str, str], str],
        authorization_clock: Callable[[], str],
        repository_identity: str,
        root_parent_commit: str,
    ) -> None:
        self.repository = Path(repository)
        self.remote = validate_remote(remote)
        if not callable(exclusion_lease):
            raise GitIdentityError("EXCLUSION_UNAVAILABLE")
        if not callable(snapshot_callback) or expected_snapshot is None:
            raise ManifestError("LIFECYCLE_SNAPSHOT_REVALIDATION_REQUIRED")
        if (
            owner_authorization_authority is None
            or not callable(authorization_reference)
            or not callable(authorization_clock)
        ):
            raise ManifestError("OWNER_AUTHORIZATION_UNVERIFIED")
        if not isinstance(repository_identity, str) or not repository_identity:
            raise ManifestError("OWNER_AUTHORIZATION_UNVERIFIED")
        if not isinstance(root_parent_commit, str) or not SHA1_RE.fullmatch(
            root_parent_commit
        ):
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
        try:
            _git(self.repository, "cat-file", "-e", f"{root_parent_commit}^{{commit}}")
        except ManifestError as error:
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED") from error
        static_data, _lifecycle_data = _snapshot_components(expected_snapshot)
        if static_data.get("source_commit") != root_parent_commit:
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
        self.exclusion = exclusion_lease
        self.snapshot_callback = snapshot_callback
        self.expected_snapshot = expected_snapshot
        self.owner_authorization_authority = owner_authorization_authority
        self.authorization_reference = authorization_reference
        self.authorization_clock = authorization_clock
        self.repository_identity = repository_identity
        self.root_parent_commit = root_parent_commit

    @contextmanager
    def _governed_boundary(self) -> Any:
        exclusion = self.exclusion
        try:
            if hasattr(exclusion, "acquire") and not hasattr(exclusion, "__enter__"):
                context = exclusion.acquire()
            elif callable(exclusion) and not hasattr(exclusion, "__enter__"):
                context = exclusion()
            else:
                context = exclusion
            if context is None or not hasattr(context, "__enter__"):
                raise GitIdentityError("EXCLUSION_UNAVAILABLE")
            with context as held:
                if held is False:
                    raise GitIdentityError("EXCLUSION_UNAVAILABLE")
                yield
        except GitIdentityError:
            raise
        except ManifestError:
            raise
        except AuthorizationError:
            raise
        except CreateOutcomeUncertain:
            raise
        except Exception as error:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE") from error

    def _snapshot(self) -> tuple[Any, bytes]:
        value = self.snapshot_callback()
        return value, _snapshot_fingerprint(value)

    def _fetch_remote_ref(self, ref: str) -> None:
        _git(self.repository, "fetch", "--no-tags", "--no-write-fetch-head", self.remote, ref)

    def _read_remote_manifest(self, ref: str, commit: str) -> tuple[bytes, str]:
        try:
            self._fetch_remote_ref(ref)
            return _manifest_from_commit(self.repository, commit)
        except ManifestError as error:
            raise CreateOutcomeUncertain("CREATE_ONCE_OUTCOME_UNCERTAIN") from error

    def remote_ref(self, ref: str) -> str | None:
        _check_ref(ref, prefix=LIFECYCLE_REF_PREFIX)
        output = _git(self.repository, "ls-remote", "--refs", self.remote, ref)
        records = parse_remote_ref_advertisements(output)
        return records.get(ref)

    def _expected_parent(
        self,
        manifest: Mapping[str, Any],
        records: Mapping[str, Mapping[str, Any]],
    ) -> str:
        predecessors = manifest["predecessor_refs"]
        if not predecessors:
            return self.root_parent_commit
        if len(predecessors) != 1:
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
        predecessor = records.get(predecessors[0])
        if predecessor is None or not SHA1_RE.fullmatch(str(predecessor.get("commit", ""))):
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")
        return str(predecessor["commit"])

    def _require_commit_parent(self, commit: str, expected_parent: str) -> None:
        line = _git(
            self.repository, "rev-list", "--parents", "-n", "1", commit
        ).decode("ascii").strip().split()
        if line != [commit, expected_parent]:
            raise ManifestError("LIFECYCLE_PARENT_UNVERIFIED")

    def _require_no_public_tag(self, manifest: Mapping[str, Any]) -> None:
        kind = manifest["manifest_type"]
        if kind not in {"candidate-withdrawn", "release-intent-aborted"}:
            return
        tag_ref = f"refs/tags/v{manifest['final_version']}"
        output = _git(
            self.repository, "ls-remote", "--refs", self.remote, tag_ref
        )
        if parse_remote_ref_advertisements(output):
            raise ManifestConflict("PUBLIC_TAG_ALREADY_EXISTS")

    def _complete_graph_records(self) -> dict[str, dict[str, Any]]:
        output = _git(
            self.repository,
            "ls-remote",
            "--refs",
            self.remote,
            f"{LIFECYCLE_REF_PREFIX}*",
        )
        advertisements = parse_remote_ref_advertisements(output)
        records: dict[str, dict[str, Any]] = {}
        for ref, commit in sorted(advertisements.items()):
            raw, tree = self._read_remote_manifest(ref, commit)
            try:
                manifest = json.loads(
                    raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                raise ManifestError("remote lifecycle manifest is invalid") from error
            records[ref] = {
                "ref": ref,
                "manifest": manifest,
                "commit": commit,
                "tree": tree,
                "manifest_id": manifest.get("manifest_id"),
                "manifest_digest": sha256_hex(raw),
            }
        final_output = _git(
            self.repository,
            "ls-remote",
            "--refs",
            self.remote,
            f"{LIFECYCLE_REF_PREFIX}*",
        )
        if parse_remote_ref_advertisements(final_output) != advertisements:
            raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")
        validate_lifecycle_graph(records)
        _validate_lifecycle_commit_topology(
            self.repository, records, self.root_parent_commit
        )
        return records

    @staticmethod
    def _snapshot_matches_records(snapshot: Any, records: Mapping[str, Mapping[str, Any]]) -> None:
        _static_data, lifecycle_data = _snapshot_components(snapshot)
        graph = validate_lifecycle_graph(records)
        bindings = sorted(
            [
                {
                    key: record[key]
                    for key in (
                        "ref", "commit", "tree", "manifest_id", "manifest_digest"
                    )
                }
                for record in records.values()
            ],
            key=lambda item: item["ref"],
        )
        if lifecycle_data["lifecycle_ref_bindings"] != bindings:
            raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")
        if lifecycle_data["occupied_versions"] != list(graph.occupied_versions):
            raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")

    def _find_manifest_id_elsewhere(self, manifest_identity: str, expected_ref: str) -> None:
        output = _git(self.repository, "ls-remote", "--refs", self.remote, f"{LIFECYCLE_REF_PREFIX}*")
        records = parse_remote_ref_advertisements(output)
        for ref, commit in sorted(records.items()):
            if ref == expected_ref:
                continue
            try:
                raw, _tree = self._read_remote_manifest(ref, commit)
                value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
            except (ManifestError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if value.get("manifest_id") == manifest_identity:
                raise ManifestConflict("CREATE_ONCE_CONFLICT")

    def create(
        self,
        manifest: Mapping[str, Any],
        protection: ProtectionEvidence,
        *,
        authorization_context: Mapping[str, str],
    ) -> CreateResult:
        checked = validate_manifest(manifest)
        raw = canonical_manifest_bytes(checked)
        ref = lifecycle_ref_for_manifest(checked)
        protection.require(ref, creation=True)
        required_context = {
            "transaction_id", "owner_line", "final_version", "action"
        }
        if set(authorization_context) != required_context:
            raise ManifestError("OWNER_AUTHORIZATION_UNVERIFIED")

        def verify_authorization() -> None:
            reference = self.authorization_reference(
                authorization_context["action"], ref,
                authorization_context["final_version"],
            )
            try:
                authorization_now = self.authorization_clock()
            except Exception as error:
                raise AuthorizationError(
                    "owner authorization clock is unavailable"
                ) from error
            record = verify_owner_authorization(
                self.owner_authorization_authority,
                reference,
                repository=self.repository_identity,
                transaction_id=authorization_context["transaction_id"],
                action=authorization_context["action"],
                owner_line=authorization_context["owner_line"],
                final_version=authorization_context["final_version"],
                target_ref=ref,
                now_utc=authorization_now,
            )
            if (
                checked["owner_authorization"] != record["owner_authorization"]
                or checked["owner_authorization_ref"] != reference
                or checked["owner_authorization_digest"]
                != record["owner_authorization_digest"]
            ):
                raise ManifestError("OWNER_AUTHORIZATION_UNVERIFIED")

        snapshot = self.expected_snapshot
        with self._governed_boundary():
            observed_snapshot, fingerprint = self._snapshot()
            verify_authorization()
            records = self._complete_graph_records()
            self._snapshot_matches_records(observed_snapshot, records)
            current_record = records.get(ref)
            current = None if current_record is None else str(current_record["commit"])
            commit: str | None = None
            tree: str | None = None
            if current is not None:
                existing_raw, existing_tree = self._read_remote_manifest(ref, current)
                if existing_raw == raw:
                    graph = validate_lifecycle_graph(records)
                    expected = _snapshot_fingerprint(snapshot)
                    if fingerprint != expected:
                        binding = {
                            key: str(current_record[key])
                            for key in (
                                "ref", "commit", "tree", "manifest_id",
                                "manifest_digest",
                            )
                        }
                        allowed = _successor_snapshot_fingerprint(
                            snapshot, binding, graph.occupied_versions
                        )
                        if fingerprint != allowed:
                            raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")
                    self.expected_snapshot = observed_snapshot
                    return CreateResult("IDEMPOTENT", ref, current, existing_tree, checked["manifest_id"])
                raise ManifestConflict("CREATE_ONCE_CONFLICT")
            if fingerprint != _snapshot_fingerprint(snapshot):
                raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")
            proposed_records = dict(records)
            proposed_records[ref] = {"ref": ref, "manifest": checked}
            validate_lifecycle_graph(proposed_records)
            self._require_no_public_tag(checked)
            self._find_manifest_id_elsewhere(checked["manifest_id"], ref)
            parent = self._expected_parent(checked, records)
            commit = make_manifest_commit(
                self.repository,
                checked,
                parent=parent,
                message=f"lifecycle: {checked['manifest_type']}\n",
            )
            self._require_commit_parent(commit, parent)
            tree = _resolve_tree(self.repository, commit)
            # Revalidate immediately before the create-once remote operation;
            # the surrounding governed exclusion remains held until reread.
            before_push_snapshot, before_push_fingerprint = self._snapshot()
            if before_push_fingerprint != fingerprint:
                raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")
            self._require_no_public_tag(checked)
            verify_authorization()
            try:
                _git(self.repository, "push", "--porcelain", f"--force-with-lease={ref}:", self.remote, f"{commit}:{ref}")
            except ManifestError as error:
                observed = self.remote_ref(ref)
                if observed is None:
                    raise CreateOutcomeUncertain("CREATE_ONCE_OUTCOME_UNCERTAIN") from error
                existing_raw, existing_tree = self._read_remote_manifest(ref, observed)
                if existing_raw == raw:
                    records = self._complete_graph_records()
                    graph = validate_lifecycle_graph(records)
                    after_snapshot, after_fingerprint = self._snapshot()
                    self._snapshot_matches_records(after_snapshot, records)
                    binding = {
                        "ref": ref,
                        "commit": observed,
                        "tree": existing_tree,
                        "manifest_id": checked["manifest_id"],
                        "manifest_digest": sha256_hex(raw),
                    }
                    if after_fingerprint != _successor_snapshot_fingerprint(
                        before_push_snapshot, binding, graph.occupied_versions
                    ):
                        raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")
                    self.expected_snapshot = after_snapshot
                    return CreateResult("IDEMPOTENT", ref, observed, existing_tree, checked["manifest_id"])
                raise ManifestConflict("CREATE_ONCE_CONFLICT") from error
            observed = self.remote_ref(ref)
            if observed is None:
                raise CreateOutcomeUncertain("CREATE_ONCE_OUTCOME_UNCERTAIN")
            existing_raw, existing_tree = self._read_remote_manifest(ref, observed)
            if observed != commit or existing_raw != raw or existing_tree != tree:
                if existing_raw != raw:
                    raise ManifestConflict("CREATE_ONCE_CONFLICT")
                raise CreateOutcomeUncertain("CREATE_ONCE_OUTCOME_UNCERTAIN")
            records = self._complete_graph_records()
            graph = validate_lifecycle_graph(records)
            after_snapshot, after_fingerprint = self._snapshot()
            self._snapshot_matches_records(after_snapshot, records)
            binding = {
                "ref": ref,
                "commit": commit,
                "tree": tree,
                "manifest_id": checked["manifest_id"],
                "manifest_digest": sha256_hex(raw),
            }
            if after_fingerprint != _successor_snapshot_fingerprint(
                before_push_snapshot, binding, graph.occupied_versions
            ):
                raise ManifestError("LIFECYCLE_SNAPSHOT_STALE")
            self.expected_snapshot = after_snapshot
            return CreateResult("CREATED", ref, commit, tree, checked["manifest_id"])


class CreateOnlyLifecycleManifestPort:
    """Bind the coordinator's manifest port to the protected network writer."""

    def __init__(
        self,
        writer: CreateOnlyLifecycleWriter,
        protection: ProtectionEvidence,
        authorization_context: Callable[[str, Mapping[str, Any]], Mapping[str, str]],
    ) -> None:
        if not callable(authorization_context):
            raise ManifestError("OWNER_AUTHORIZATION_UNVERIFIED")
        self.writer = writer
        self.protection = protection
        self.authorization_context = authorization_context

    def authorization_now_utc(self) -> str:
        return self.writer.authorization_clock()

    def create_manifest(
        self, manifest_type: str, manifest: Mapping[str, Any]
    ) -> dict[str, Any]:
        checked = validate_manifest(manifest)
        if checked["manifest_type"] != manifest_type:
            raise ManifestError("manifest type does not match mutation port")
        context = self.authorization_context(manifest_type, checked)
        self.writer.create(
            checked, self.protection, authorization_context=context
        )
        return dict(checked)


__all__ = [
    "CreateOnlyLifecycleManifestPort", "CreateOnlyLifecycleWriter",
    "CreateOutcomeUncertain", "CreateResult",
    "LIFECYCLE_REF_PREFIX", "LifecycleRefSnapshot", "MANIFEST_FILE", "MANIFEST_ID_RE",
    "LifecycleGraph", "MANIFEST_TYPES", "ManifestConflict", "ManifestError", "SCHEMA_VERSION",
    "build_lifecycle_ref_snapshot", "canonical_manifest_bytes", "lifecycle_ref_for_manifest",
    "lifecycle_ref_snapshot", "manifest_id", "manifest_identity_preimage", "make_manifest_commit", "seal_manifest",
    "publication_id", "publication_key", "publication_key_fixture", "publication_ref",
    "replay_lifecycle_graph", "validate_complete_lifecycle_refs", "validate_lifecycle_graph",
    "validate_lifecycle_ref_snapshot", "validate_manifest",
]
