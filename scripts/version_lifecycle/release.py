"""Release and publication consistency for immutable lifecycle manifests.

This module contains pure validators only. A release is terminal only when
the released manifest, canonical public tag, exact tree identities and the
single deterministic publication manifest agree. No mutable stream is imported
or used as an authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any

from .certification import validate_certification_transfer
from .codec import sha256_hex
from .manifests import (
    ManifestError,
    canonical_manifest_bytes,
    lifecycle_ref_for_manifest,
    publication_ref,
    validate_lifecycle_graph,
    validate_manifest,
)
from .versions import parse_package_version, parse_public_tag

PASS = "PASS"
INVALID = "INVALID"
BLOCKED = "BLOCKED"
TERMINAL_CONSISTENT = "terminal_consistent"
TAG_RECONCILIATION_PENDING = "tag_reconciliation_pending"
PUBLICATION_RECONCILIATION_PENDING = "publication_reconciliation_pending"
LEGACY_EXCLUDED = "LEGACY_EXCLUDED"
LEGACY_PUBLIC_TAG = "v-0.1"
_FULL_SHA1 = re.compile(r"[0-9a-f]{40}\Z")


def _result(status: str, reason_code: str | None = None, errors: list[str] | None = None, **details: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"status": status}
    if reason_code is not None:
        value["reason_code"] = reason_code
    if errors:
        value["errors"] = list(errors)
    value.update(details)
    return value


def is_canonical_public_tag(value: Any) -> bool:
    if not isinstance(value, str) or value == LEGACY_PUBLIC_TAG:
        return False
    try:
        parsed = parse_public_tag(value)
    except (TypeError, ValueError):
        return False
    return parsed.canonical == value[1:]


def tag_version(value: str) -> str | None:
    return value[1:] if is_canonical_public_tag(value) else None


def validate_publication_evidence(
    publication: Mapping[str, Any], released: Mapping[str, Any] | None = None,
    *, tag_commit: str | None = None, tag_tree: str | None = None,
    publication_evidence_digest: str | None = None,
) -> dict[str, Any]:
    """Validate a publication manifest and its unique released predecessor."""

    try:
        value = validate_manifest(publication)
    except ManifestError as error:
        return _result(INVALID, "PUBLICATION_MANIFEST_INVALID", [str(error)])
    if value["manifest_type"] != "publication":
        return _result(INVALID, "PUBLICATION_MANIFEST_INVALID", ["manifest_type must be publication"])
    expected_ref = publication_ref(value["public_tag"], value["released_manifest_id"])
    if value.get("ref") not in {None, expected_ref}:
        return _result(INVALID, "PUBLICATION_REF_MISMATCH")
    if released is None:
        return _result(PUBLICATION_RECONCILIATION_PENDING, "RELEASED_MANIFEST_UNAVAILABLE", expected_ref=expected_ref)
    try:
        released_value = validate_manifest(released)
    except ManifestError as error:
        return _result(INVALID, "RELEASED_MANIFEST_INVALID", [str(error)])
    if released_value["manifest_type"] != "released":
        return _result(INVALID, "RELEASED_PREDECESSOR_INVALID")
    if value["released_manifest_id"] != released_value["manifest_id"]:
        return _result(INVALID, "RELEASED_MANIFEST_ID_MISMATCH")
    actual_released_digest = sha256_hex(canonical_manifest_bytes(released_value))
    if value["released_manifest_digest"] != actual_released_digest:
        return _result(INVALID, "RELEASED_MANIFEST_DIGEST_MISMATCH")
    expected_released_ref = f"refs/heads/lifecycle/v1/releases/v{released_value['final_version']}"
    if value["released_manifest_ref"] != expected_released_ref:
        return _result(INVALID, "RELEASED_MANIFEST_REF_MISMATCH")
    if tag_commit is not None and value["tag_commit"] != tag_commit:
        return _result(INVALID, "PUBLIC_TAG_COMMIT_MISMATCH")
    if tag_tree is not None and value["tag_tree"] != tag_tree:
        return _result(INVALID, "PUBLIC_TAG_TREE_MISMATCH")
    if value["public_tag"] != released_value.get("public_tag"):
        return _result(INVALID, "PUBLIC_TAG_MISMATCH")
    if value["tag_commit"] != released_value.get("final_release_sha"):
        return _result(INVALID, "RELEASE_COMMIT_MISMATCH")
    if value["tag_tree"] != released_value.get("final_release_tree"):
        return _result(INVALID, "RELEASE_TREE_MISMATCH")
    if (
        publication_evidence_digest is not None
        and value["publication_evidence_digest"] != publication_evidence_digest
    ):
        return _result(INVALID, "PUBLICATION_EVIDENCE_DIGEST_MISMATCH")
    return _result(PASS, publication_id=value["publication_id"])


def validate_released_manifest(
    released: Mapping[str, Any], *, public_tag: str | None = None,
    tag_commit: str | None = None, tag_tree: str | None = None,
    certified_tree: str | None = None,
) -> dict[str, Any]:
    """Validate the durable release identity before GitHub publication."""

    try:
        value = validate_manifest(released)
    except ManifestError as error:
        return _result(INVALID, "RELEASED_MANIFEST_INVALID", [str(error)])
    if value["manifest_type"] != "released":
        return _result(INVALID, "RELEASED_MANIFEST_INVALID", ["manifest_type must be released"])
    if public_tag is not None and value.get("public_tag") != public_tag:
        return _result(INVALID, "PUBLIC_TAG_MISMATCH")
    if tag_commit is not None and value.get("final_release_sha") != tag_commit:
        return _result(INVALID, "RELEASE_COMMIT_MISMATCH")
    if tag_tree is not None and value.get("final_release_tree") != tag_tree:
        return _result(INVALID, "RELEASE_TREE_MISMATCH")
    if certified_tree is not None and value.get("anchor_tree") != certified_tree:
        return _result(INVALID, "CERTIFIED_TREE_MISMATCH")
    certification_record: dict[str, Any] = {
        "binding": value.get("certification_binding"),
        "package_commit": value.get("certification_subject_sha"),
        "package_tree": value.get("certification_subject_tree"),
        "policy_revision": value.get("certification_policy_revision"),
        "harness_revision": value.get("certification_harness_revision"),
        "environment": value.get("certification_environment"),
        "evidence": value.get("certification_evidence_refs", [None])[0],
        "candidate_commit": value.get("candidate_sha"),
        "candidate_tree": value.get("candidate_tree"),
        "anchor_tree": value.get("anchor_tree"),
    }
    if "certification_transfer_evidence" in value:
        certification_record["certification_transfer_evidence"] = value[
            "certification_transfer_evidence"
        ]
    certification = validate_certification_transfer(
        certification_record,
        candidate_commit=str(value.get("candidate_sha", "")),
        candidate_tree=str(value.get("candidate_tree", "")),
        final_release_commit=str(value.get("final_release_sha", "")),
        final_release_tree=str(value.get("final_release_tree", "")),
        anchor_tree=str(value.get("anchor_tree", "")),
        public_tag_exists=public_tag is not None or tag_commit is not None,
    )
    if certification["status"] != PASS:
        return certification
    if value.get("release_line") == "principal":
        try:
            if parse_package_version(value["main_at_release_version"]).is_dev:
                return _result(INVALID, "MAIN_RELEASE_VERSION_IS_DEV")
        except (TypeError, ValueError):
            return _result(INVALID, "MAIN_RELEASE_VERSION_INVALID")
        if value["main_at_release_version"] != value["final_version"]:
            return _result(INVALID, "MAIN_RELEASE_VERSION_MISMATCH")
        if value["main_at_release_sha"] != value["final_release_sha"]:
            return _result(INVALID, "MAIN_RELEASE_COMMIT_MISMATCH")
    return _result(PASS, released_manifest_id=value["manifest_id"], public_tag=value["public_tag"])


def validate_release_consistency(
    released: Mapping[str, Any] | None = None,
    publication: Mapping[str, Any] | None = None,
    *, public_tag: str | None = None, tag_commit: str | None = None,
    tag_tree: str | None = None, certified_tree: str | None = None,
    github_release_id: object | None = None,
    publication_evidence_digest: str | None = None,
    lifecycle_records: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    anchor_tag_object: str | None = None,
    anchor_tree: str | None = None,
    anchor_closure_timestamp_utc: str | None = None,
    candidate_ref: str | None = None,
    candidate_commit: str | None = None,
    candidate_tree: str | None = None,
    canonical_tag_observations: Iterable[Mapping[str, Any]] | None = None,
    github_release_observations: Iterable[Mapping[str, Any]] | None = None,
    require_complete_namespace: bool = False,
) -> dict[str, Any]:
    """Perform bidirectional tag/release/publication consistency checks."""

    if released is None:
        return _validate_tag_reconciliation_pending(
            public_tag=public_tag,
            tag_commit=tag_commit,
            tag_tree=tag_tree,
            github_release_id=github_release_id,
            lifecycle_records=lifecycle_records,
            canonical_tag_observations=canonical_tag_observations,
            github_release_observations=github_release_observations,
            require_complete_namespace=require_complete_namespace,
        )
    release_result = validate_released_manifest(
        released, public_tag=public_tag, tag_commit=tag_commit,
        tag_tree=tag_tree, certified_tree=certified_tree,
    )
    if release_result["status"] != PASS:
        return release_result
    if publication is None:
        return _result(PUBLICATION_RECONCILIATION_PENDING, "PUBLICATION_MANIFEST_UNAVAILABLE", released_manifest_id=released["manifest_id"])
    publication_result = validate_publication_evidence(
        publication, released, tag_commit=tag_commit, tag_tree=tag_tree,
        publication_evidence_digest=publication_evidence_digest,
    )
    if publication_result["status"] != PASS:
        return publication_result
    if github_release_id is not None and publication.get("github_release_id") != github_release_id:
        return _result(INVALID, "GITHUB_RELEASE_ID_MISMATCH")
    if (
        lifecycle_records is None
        or canonical_tag_observations is None
        or github_release_observations is None
    ):
        return _result(BLOCKED, "COMPLETE_RELEASE_UNIVERSE_UNAVAILABLE")
    try:
        graph = validate_lifecycle_graph(lifecycle_records)
        tags = _canonical_tag_index(canonical_tag_observations)
        github_releases = _github_release_index(github_release_observations)
    except (ManifestError, TypeError, ValueError) as error:
        return _result(INVALID, "COMPLETE_RELEASE_UNIVERSE_INVALID", [str(error)])
    universe = _validate_complete_release_universe(
        graph.manifests_by_ref, tags, github_releases
    )
    if universe is not None:
        return universe
    released_ref = lifecycle_ref_for_manifest(released)
    publication_ref_name = lifecycle_ref_for_manifest(publication)
    if graph.manifests_by_ref.get(released_ref) != released:
        return _result(INVALID, "RELEASED_MANIFEST_NAMESPACE_MISMATCH")
    if graph.manifests_by_ref.get(publication_ref_name) != publication:
        return _result(INVALID, "PUBLICATION_MANIFEST_NAMESPACE_MISMATCH")
    observations = (
        anchor_tag_object,
        anchor_tree,
        anchor_closure_timestamp_utc,
        candidate_ref,
        candidate_commit,
        candidate_tree,
    )
    if any(value is None for value in observations):
        return _result(BLOCKED, "DIRECT_IDENTITY_OBSERVATION_INCOMPLETE")
    if (
        released.get("anchor_sha") != anchor_tag_object
        or released.get("anchor_tree") != anchor_tree
        or released.get("closure_timestamp_utc")
        != anchor_closure_timestamp_utc
    ):
        return _result(INVALID, "ANCHOR_IDENTITY_MISMATCH")
    if (
        released.get("candidate_ref") != candidate_ref
        or released.get("candidate_sha") != candidate_commit
        or released.get("candidate_tree") != candidate_tree
    ):
        return _result(INVALID, "CANDIDATE_IDENTITY_MISMATCH")
    if any(
        value is None
        for value in (
            public_tag,
            tag_commit,
            tag_tree,
            certified_tree,
            github_release_id,
            publication_evidence_digest,
        )
    ):
        return _result(BLOCKED, "TERMINAL_OBSERVATION_INCOMPLETE")
    return _result(TERMINAL_CONSISTENT, released_manifest_id=released["manifest_id"], publication_id=publication["publication_id"], public_tag=released["public_tag"])


def _validate_tag_reconciliation_pending(
    *,
    public_tag: str | None,
    tag_commit: str | None,
    tag_tree: str | None,
    github_release_id: object | None,
    lifecycle_records: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
    canonical_tag_observations: Iterable[Mapping[str, Any]] | None,
    github_release_observations: Iterable[Mapping[str, Any]] | None,
    require_complete_namespace: bool,
) -> dict[str, Any]:
    """Recover the irreversible tag boundary from intent and immutable tag evidence."""

    if lifecycle_records is None or canonical_tag_observations is None:
        return _result(BLOCKED, "RELEASED_MANIFEST_UNAVAILABLE")
    if public_tag is None:
        return _result(BLOCKED, "PUBLIC_TAG_UNAVAILABLE")
    if not is_canonical_public_tag(public_tag):
        return _result(INVALID, "PUBLIC_TAG_INVALID")
    if require_complete_namespace and github_release_observations is None:
        return _result(BLOCKED, "COMPLETE_RELEASE_UNIVERSE_UNAVAILABLE")
    try:
        graph = validate_lifecycle_graph(lifecycle_records)
        tags = _canonical_tag_index(canonical_tag_observations)
        github_releases = (
            None
            if github_release_observations is None
            else _github_release_index(github_release_observations)
        )
    except (ManifestError, TypeError, ValueError) as error:
        return _result(INVALID, "COMPLETE_RELEASE_UNIVERSE_INVALID", [str(error)])

    tag_observation = tags.get(public_tag)
    if github_release_id is not None or (
        github_releases is not None and public_tag in github_releases
    ):
        return _result(INVALID, "GITHUB_RELEASE_WITHOUT_RELEASED_MANIFEST")
    if tag_observation is None and (tag_commit is not None or tag_tree is not None):
        return _result(INVALID, "CANONICAL_TAG_OBSERVATION_MISMATCH")
    intents = [
        manifest
        for manifest in graph.manifests_by_ref.values()
        if manifest["manifest_type"] == "release-intent-prepared"
        and manifest.get("public_tag") == public_tag
    ]
    if tag_observation is None:
        # Intent without a tag is still on the reversible, pre-boundary side.
        return _result(BLOCKED, "PUBLIC_TAG_UNAVAILABLE")
    if len(intents) != 1:
        return _result(INVALID, "TAG_RECONCILIATION_INTENT_MISMATCH")

    intent = intents[0]
    intent_ref = lifecycle_ref_for_manifest(intent)
    children = {
        ref
        for ref, manifest in graph.manifests_by_ref.items()
        if intent_ref in manifest.get("predecessor_refs", [])
    }
    # A durable released child means this is not a pending transition; an
    # aborted intent cannot authorize a tag that remains publicly reachable.
    child_types = {
        graph.manifests_by_ref[ref]["manifest_type"] for ref in children
    }
    if child_types.intersection({"released", "release-intent-aborted"}):
        return _result(INVALID, "TAG_RECONCILIATION_INTENT_MISMATCH")

    expected_version = tag_version(public_tag)
    if (
        expected_version != intent.get("final_version")
        or tag_observation["commit"] != intent.get("final_release_sha")
        or tag_observation["tree"] != intent.get("final_release_tree")
        or (tag_commit is not None and tag_observation["commit"] != tag_commit)
        or (tag_tree is not None and tag_observation["tree"] != tag_tree)
    ):
        return _result(INVALID, "TAG_RECONCILIATION_IDENTITY_MISMATCH")

    if require_complete_namespace:
        assert github_releases is not None
        released_by_tag: dict[str, list[Mapping[str, Any]]] = {}
        publications_by_tag: dict[str, list[Mapping[str, Any]]] = {}
        for manifest in graph.manifests_by_ref.values():
            kind = manifest["manifest_type"]
            if kind == "released":
                released_by_tag.setdefault(str(manifest["public_tag"]), []).append(manifest)
            elif kind == "publication":
                publications_by_tag.setdefault(str(manifest["public_tag"]), []).append(manifest)
        closed_tags = set(tags) - {public_tag}
        if (
            public_tag not in tags
            or set(released_by_tag) != closed_tags
            or set(publications_by_tag) != closed_tags
            or set(github_releases) != closed_tags
            or any(len(items) != 1 for items in released_by_tag.values())
            or any(len(items) != 1 for items in publications_by_tag.values())
        ):
            return _result(INVALID, "COMPLETE_RELEASE_UNIVERSE_MISMATCH")
        closed_universe = _validate_complete_release_universe(
            graph.manifests_by_ref,
            {tag: value for tag, value in tags.items() if tag != public_tag},
            github_releases,
        )
        if closed_universe is not None:
            return closed_universe

    return _result(
        TAG_RECONCILIATION_PENDING,
        "RELEASED_MANIFEST_UNAVAILABLE",
        intent_ref=intent_ref,
        intent_manifest_id=intent["manifest_id"],
        public_tag=public_tag,
        tag_commit=tag_observation["commit"],
        tag_tree=tag_observation["tree"],
    )


def _canonical_tag_index(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise ValueError("canonical tag observation must be an object")
        value = dict(raw)
        if set(value) != {"ref", "tag", "commit", "tree"}:
            raise ValueError("canonical tag observation fields are invalid")
        tag = value["tag"]
        if (
            not is_canonical_public_tag(tag)
            or value["ref"] != f"refs/tags/{tag}"
            or not isinstance(value["commit"], str)
            or _FULL_SHA1.fullmatch(value["commit"]) is None
            or not isinstance(value["tree"], str)
            or _FULL_SHA1.fullmatch(value["tree"]) is None
        ):
            raise ValueError("canonical tag observation is invalid")
        if tag in result:
            raise ValueError("duplicate canonical tag observation")
        result[tag] = value
    return result


def _github_release_index(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    ids: set[int] = set()
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise ValueError("GitHub Release observation must be an object")
        value = dict(raw)
        if set(value) != {"id", "tag", "url"}:
            raise ValueError("GitHub Release observation fields are invalid")
        tag = value["tag"]
        release_id = value["id"]
        release_url = value["url"]
        if tag == LEGACY_PUBLIC_TAG:
            continue
        if not is_canonical_public_tag(tag):
            raise ValueError("GitHub Release tag is not canonical")
        if (
            isinstance(release_id, bool)
            or not isinstance(release_id, int)
            or release_id <= 0
            or not isinstance(release_url, str)
            or not release_url
            or tag in result
            or release_id in ids
        ):
            raise ValueError("duplicate or invalid GitHub Release observation")
        result[tag] = value
        ids.add(release_id)
    return result


def _validate_complete_release_universe(
    manifests_by_ref: Mapping[str, Mapping[str, Any]],
    tags: Mapping[str, Mapping[str, Any]],
    github_releases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    released_by_tag: dict[str, list[Mapping[str, Any]]] = {}
    publications_by_tag: dict[str, list[Mapping[str, Any]]] = {}
    for manifest in manifests_by_ref.values():
        kind = manifest["manifest_type"]
        if kind == "released":
            released_by_tag.setdefault(str(manifest["public_tag"]), []).append(manifest)
        elif kind == "publication":
            publications_by_tag.setdefault(str(manifest["public_tag"]), []).append(manifest)
    names = set(tags)
    if names != set(released_by_tag) or names != set(publications_by_tag) or names != set(github_releases):
        return _result(INVALID, "COMPLETE_RELEASE_UNIVERSE_MISMATCH")
    for tag in sorted(names):
        releases = released_by_tag[tag]
        publications = publications_by_tag[tag]
        if len(releases) != 1 or len(publications) != 1:
            return _result(INVALID, "COMPLETE_RELEASE_UNIVERSE_MISMATCH")
        released = releases[0]
        publication = publications[0]
        tag_observation = tags[tag]
        github_release = github_releases[tag]
        released_result = validate_released_manifest(
            released,
            public_tag=tag,
            tag_commit=str(tag_observation["commit"]),
            tag_tree=str(tag_observation["tree"]),
            certified_tree=str(released["anchor_tree"]),
        )
        if released_result["status"] != PASS:
            return released_result
        publication_result = validate_publication_evidence(
            publication,
            released,
            tag_commit=str(tag_observation["commit"]),
            tag_tree=str(tag_observation["tree"]),
        )
        if publication_result["status"] != PASS:
            return publication_result
        if publication["github_release_id"] != github_release["id"]:
            return _result(INVALID, "GITHUB_RELEASE_ID_MISMATCH")
        if publication["github_release_url"] != github_release["url"]:
            return _result(INVALID, "GITHUB_RELEASE_URL_MISMATCH")
    return None


def validate_bidirectional_consistency(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return validate_release_consistency(*args, **kwargs)


def validate_release_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return validate_release_consistency(*args, **kwargs)


# Compatibility spelling for callers that migrated from old event-shaped
# input. The value is intentionally a manifest validator.
validate_released_event = validate_released_manifest


__all__ = [
    "BLOCKED", "INVALID", "LEGACY_EXCLUDED", "LEGACY_PUBLIC_TAG", "PASS",
    "PUBLICATION_RECONCILIATION_PENDING", "TAG_RECONCILIATION_PENDING",
    "TERMINAL_CONSISTENT", "is_canonical_public_tag", "tag_version",
    "validate_bidirectional_consistency", "validate_publication_evidence",
    "validate_release_consistency", "validate_release_evidence",
    "validate_released_event", "validate_released_manifest",
]
