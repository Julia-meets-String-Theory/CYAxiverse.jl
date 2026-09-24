"""Typed canonical evidence binding a release manifest to its public tag.

This payload is prepared before GitHub Release creation.  It intentionally
contains no GitHub Release ID or URL; those post-publication identities live
in the separate publication manifest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .certification import is_safe_public_value
from .codec import canonical_json, sha256_hex
from .manifests import (
    ManifestError,
    canonical_manifest_bytes,
    lifecycle_ref_for_manifest,
    validate_manifest,
)


PUBLICATION_EVIDENCE_FIELDS = frozenset({
    "schema_version",
    "public_tag",
    "released_manifest_id",
    "released_manifest_ref",
    "released_manifest_digest",
    "tag_commit",
    "tag_tree",
})
_FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PublicationEvidenceError(ValueError):
    """Raised when a public publication-evidence record is not exact."""


def validate_publication_evidence_record(
    value: Any,
    released: Mapping[str, Any],
    *,
    public_tag: str,
    tag_commit: str,
    tag_tree: str,
) -> dict[str, Any]:
    """Validate exact fields and every identity in a pre-publication record."""

    if not isinstance(value, Mapping) or set(value) != PUBLICATION_EVIDENCE_FIELDS:
        raise PublicationEvidenceError("publication evidence fields are not exact")
    if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
        raise PublicationEvidenceError("unsupported publication evidence schema")
    try:
        checked_release = validate_manifest(released)
    except ManifestError as error:
        raise PublicationEvidenceError("released manifest is invalid") from error
    if checked_release.get("manifest_type") != "released":
        raise PublicationEvidenceError("publication evidence requires a released manifest")
    if (
        not isinstance(public_tag, str)
        or not isinstance(tag_commit, str)
        or not isinstance(tag_tree, str)
        or not _FULL_SHA1.fullmatch(tag_commit)
        or not _FULL_SHA1.fullmatch(tag_tree)
    ):
        raise PublicationEvidenceError("publication evidence tag identity is invalid")
    expected = {
        "schema_version": 1,
        "public_tag": public_tag,
        "released_manifest_id": checked_release["manifest_id"],
        "released_manifest_ref": lifecycle_ref_for_manifest(checked_release),
        "released_manifest_digest": sha256_hex(
            canonical_manifest_bytes(checked_release)
        ),
        "tag_commit": tag_commit,
        "tag_tree": tag_tree,
    }
    if (
        public_tag != checked_release.get("public_tag")
        or tag_commit != checked_release.get("final_release_sha")
        or tag_tree != checked_release.get("final_release_tree")
        or dict(value) != expected
    ):
        raise PublicationEvidenceError("publication evidence identity mismatch")
    if not _FULL_SHA256.fullmatch(str(value["released_manifest_digest"])):
        raise PublicationEvidenceError("released manifest digest is invalid")
    if not is_safe_public_value(value):
        raise PublicationEvidenceError("publication evidence contains a nonpublic value")
    return dict(expected)


def publication_evidence_for_tag(
    released: Mapping[str, Any], tag: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the exact evidence mapping from checked release/tag identities."""

    public_tag = tag.get("name", tag.get("public_tag"))
    tag_commit = tag.get("commit", tag.get("tag_commit"))
    tag_tree = tag.get("tree", tag.get("tag_tree"))
    if (
        not isinstance(public_tag, str)
        or not isinstance(tag_commit, str)
        or not isinstance(tag_tree, str)
    ):
        raise PublicationEvidenceError("public tag identity is incomplete")
    checked_release = validate_manifest(released)
    try:
        value = {
            "schema_version": 1,
            "public_tag": public_tag,
            "released_manifest_id": checked_release["manifest_id"],
            "released_manifest_ref": lifecycle_ref_for_manifest(checked_release),
            "released_manifest_digest": sha256_hex(
                canonical_manifest_bytes(checked_release)
            ),
            "tag_commit": tag_commit,
            "tag_tree": tag_tree,
        }
    except ManifestError as error:
        raise PublicationEvidenceError("released manifest is invalid") from error
    return validate_publication_evidence_record(
        value,
        checked_release,
        public_tag=public_tag,
        tag_commit=tag_commit,
        tag_tree=tag_tree,
    )


def canonical_publication_evidence_bytes(
    value: Any,
    released: Mapping[str, Any],
    *,
    public_tag: str,
    tag_commit: str,
    tag_tree: str,
) -> bytes:
    """Return exact canonical no-LF evidence bytes after identity validation."""

    checked = validate_publication_evidence_record(
        value,
        released,
        public_tag=public_tag,
        tag_commit=tag_commit,
        tag_tree=tag_tree,
    )
    try:
        return canonical_json(checked)
    except (TypeError, ValueError) as error:
        raise PublicationEvidenceError("publication evidence is not canonical JSON") from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationEvidenceError("publication evidence contains duplicate keys")
        result[key] = value
    return result


def parse_publication_evidence(
    raw: bytes,
    released: Mapping[str, Any],
    *,
    public_tag: str,
    tag_commit: str,
    tag_tree: str,
) -> dict[str, Any]:
    """Parse exact canonical evidence bytes and verify release/tag identities."""

    if not isinstance(raw, bytes) or not raw or raw.endswith(b"\n"):
        raise PublicationEvidenceError("publication evidence bytes are not exact")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationEvidenceError("publication evidence is not valid JSON") from error
    checked = validate_publication_evidence_record(
        value,
        released,
        public_tag=public_tag,
        tag_commit=tag_commit,
        tag_tree=tag_tree,
    )
    try:
        canonical = canonical_json(checked)
    except (TypeError, ValueError) as error:
        raise PublicationEvidenceError("publication evidence is not canonical JSON") from error
    if canonical != raw:
        raise PublicationEvidenceError("publication evidence bytes are not canonical")
    return checked


__all__ = [
    "PUBLICATION_EVIDENCE_FIELDS",
    "PublicationEvidenceError",
    "canonical_publication_evidence_bytes",
    "parse_publication_evidence",
    "publication_evidence_for_tag",
    "validate_publication_evidence_record",
]
