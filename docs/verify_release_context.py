#!/usr/bin/env python3
"""Verify immutable release and publication manifests for documentation.

Documentation deployment consumes an exact released manifest and its one
publication predecessor. Tracked documentation source is never edited for a
release, and a missing, ambiguous, or mismatched manifest blocks deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from version_lifecycle.manifests import (  # noqa: E402
    ManifestError,
    canonical_manifest_bytes,
    validate_lifecycle_graph,
    validate_manifest,
)
from version_lifecycle.publication_evidence import (  # noqa: E402
    PublicationEvidenceError,
    parse_publication_evidence,
)
from version_lifecycle.release import (  # noqa: E402
    LEGACY_PUBLIC_TAG,
    PASS,
    TERMINAL_CONSISTENT,
    is_canonical_public_tag,
    validate_release_consistency,
)
from version_lifecycle.versions import parse_package_version  # noqa: E402

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def fail(message: str) -> int:
    print(f"release documentation context blocked: {message}", file=sys.stderr)
    return 2


def _full_sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        checked = validate_manifest(value)
        if canonical_manifest_bytes(checked) != raw:
            raise ManifestError("manifest file is not exact canonical no-LF JSON")
    except (OSError, UnicodeError, json.JSONDecodeError, ManifestError) as error:
        raise ValueError(f"invalid immutable lifecycle manifest: {error}") from error
    return checked


def write_environment(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"newline in environment value: {key}")
            handle.write(f"{key}={value}\n")


def verify(args: argparse.Namespace) -> int:
    if args.tag is None or args.tag_ref is None:
        return fail("canonical tag identity is required")
    if not is_canonical_public_tag(args.tag) or args.tag_ref != f"refs/tags/{args.tag}":
        return fail("tag is not an exact canonical vX.Y.Z ref")
    if not _full_sha(args.tag_sha) or not _full_sha(args.tag_tree) or not _full_sha(args.main_sha):
        return fail("tag/main identities must be full Git SHAs")
    try:
        tag_version = parse_package_version(args.tag_version)
        main_version = parse_package_version(args.main_version)
    except (TypeError, ValueError) as error:
        return fail(f"package version is not canonical: {error}")
    if tag_version.is_dev or tag_version.canonical != args.tag[1:]:
        return fail("tag Project.toml version does not match canonical tag")
    if main_version.is_dev:
        return fail("principal main must not carry a DEV version")
    try:
        released = _read_manifest(args.manifest)
        publication = _read_manifest(args.publication_manifest)
        evidence_bytes = args.publication_evidence.read_bytes()
    except ValueError as error:
        return fail(str(error))
    except OSError as error:
        return fail(f"publication evidence is unavailable: {error}")
    try:
        parse_publication_evidence(
            evidence_bytes,
            released,
            public_tag=args.tag,
            tag_commit=args.tag_sha,
            tag_tree=args.tag_tree,
        )
    except PublicationEvidenceError as error:
        return fail(f"publication evidence is invalid: {error}")
    evidence_digest = hashlib.sha256(evidence_bytes).hexdigest()
    lifecycle_records = None
    canonical_tags = None
    github_releases = None
    if args.require_complete_lifecycle:
        if (
            args.lifecycle_index is None
            or args.canonical_tags is None
            or args.github_releases is None
        ):
            return fail("complete release-universe observations are required")
        try:
            lifecycle_records = json.loads(
                args.lifecycle_index.read_text(encoding="utf-8")
            )
            canonical_tags = json.loads(
                args.canonical_tags.read_text(encoding="utf-8")
            )
            github_releases = json.loads(
                args.github_releases.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return fail(f"complete release universe is invalid: {error}")
    result = validate_release_consistency(
        released, publication,
        public_tag=args.tag, tag_commit=args.tag_sha, tag_tree=args.tag_tree,
        certified_tree=args.tag_tree,
        github_release_id=args.github_release_id,
        publication_evidence_digest=evidence_digest,
        lifecycle_records=lifecycle_records,
        anchor_tag_object=args.anchor_tag_object,
        anchor_tree=args.anchor_tree,
        anchor_closure_timestamp_utc=args.anchor_closure_timestamp_utc,
        candidate_ref=args.candidate_ref,
        candidate_commit=args.candidate_commit,
        candidate_tree=args.candidate_tree,
        canonical_tag_observations=canonical_tags,
        github_release_observations=github_releases,
        require_complete_namespace=args.require_complete_lifecycle,
    )
    if result.get("status") != PASS and result.get("status") != "terminal_consistent":
        return fail(f"immutable release/publication validation failed: {result}")
    try:
        if args.anchor_observations is None or args.stable_publication_evidence is None:
            return fail("independent stable-release authority observations are required")
        anchor_observations = json.loads(
            args.anchor_observations.read_text(encoding="utf-8")
        )
        stable_evidence_bytes = args.stable_publication_evidence.read_bytes()
        stable_tag = _verified_current_principal_tag(
            args.main_sha, main_version.canonical,
            lifecycle_records, canonical_tags, github_releases,
            anchor_observations, stable_evidence_bytes,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ManifestError,
            TypeError, ValueError) as error:
        return fail(f"current stable-selector authority is invalid: {error}")
    is_current_principal = (
        released.get("release_line") == "principal"
        and released.get("final_release_sha") == args.main_sha
        and released.get("final_version") == main_version.canonical
        and stable_tag == args.tag
    )
    values = {
        "CYAX_DOCS_MANIFEST_STATUS": "verified",
        "CYAX_DOCS_RELEASE_LINE": str(released["release_line"]),
        "CYAX_DOCS_RELEASE_VERSION": str(released["final_version"]),
        "CYAX_DOCS_RELEASE_SHA": str(released["final_release_sha"]),
        "CYAX_DOCS_RELEASE_MANIFEST_ID": str(released["manifest_id"]),
        "CYAX_DOCS_PUBLICATION_ID": str(publication["publication_id"]),
        "CYAX_DOCS_PRINCIPAL_MAIN_SHA": args.main_sha,
        "CYAX_DOCS_STABLE": "true" if is_current_principal else "false",
        # Empty is an explicit clearing of any inherited/unverified value.
        "CYAX_DOCS_STABLE_TAG": stable_tag or "",
    }
    write_environment(args.github_env, values)
    print(json.dumps(values, sort_keys=True))
    return 0


def verify_stable_context(args: argparse.Namespace) -> int:
    if not _full_sha(args.main_sha):
        return fail("principal main SHA is not a full Git SHA")
    try:
        version = parse_package_version(args.main_version)
    except (TypeError, ValueError) as error:
        return fail(f"principal main version is not canonical: {error}")
    if version.is_dev:
        return fail("principal main must not carry a DEV version")
    if (
        args.lifecycle_index is None
        or args.canonical_tags is None
        or args.github_releases is None
        or args.anchor_observations is None
        or args.stable_publication_evidence is None
    ):
        return fail("complete immutable release and anchor observations are required")
    try:
        lifecycle_records = json.loads(
            args.lifecycle_index.read_text(encoding="utf-8")
        )
        canonical_tags = json.loads(
            args.canonical_tags.read_text(encoding="utf-8")
        )
        github_releases = json.loads(
            args.github_releases.read_text(encoding="utf-8")
        )
        anchor_observations = json.loads(
            args.anchor_observations.read_text(encoding="utf-8")
        )
        stable_evidence_bytes = args.stable_publication_evidence.read_bytes()
        stable_tag = _verified_current_principal_tag(
            args.main_sha, version.canonical,
            lifecycle_records, canonical_tags, github_releases,
            anchor_observations, stable_evidence_bytes,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ManifestError,
            TypeError, ValueError) as error:
        return fail(f"current stable-selector authority is invalid: {error}")
    values = {
        "CYAX_DOCS_PRINCIPAL_MAIN_SHA": args.main_sha,
        "CYAX_DOCS_STABLE": "false",
        # Do not let an inherited or caller-provided tag become a selector.
        "CYAX_DOCS_STABLE_TAG": stable_tag or "",
    }
    write_environment(args.github_env, values)
    print(json.dumps(values, sort_keys=True))
    return 0


def _verified_current_principal_tag(
    main_sha: str,
    main_version: str,
    lifecycle_records: object,
    canonical_tags: object,
    github_releases: object,
    anchor_observations: object,
    stable_evidence_bytes: bytes,
) -> str | None:
    """Resolve stable only from a complete, terminal immutable release universe."""

    if (
        not isinstance(canonical_tags, list)
        or not isinstance(github_releases, list)
        or not isinstance(anchor_observations, list)
        or not isinstance(stable_evidence_bytes, bytes)
    ):
        raise ValueError("release, tag and anchor observations must be exact arrays")
    graph = validate_lifecycle_graph(lifecycle_records)
    manifests = graph.manifests_by_ref

    tags_by_name: dict[str, dict[str, object]] = {}
    for value in canonical_tags:
        if (
            not isinstance(value, dict)
            or set(value) != {"ref", "tag", "commit", "tree"}
        ):
            raise ValueError("canonical tag observation fields are invalid")
        tag = value["tag"]
        if (
            not is_canonical_public_tag(tag)
            or value["ref"] != f"refs/tags/{tag}"
            or not _full_sha(value["commit"])
            or not _full_sha(value["tree"])
            or tag in tags_by_name
        ):
            raise ValueError("canonical tag observation is invalid or duplicated")
        tags_by_name[tag] = value

    releases_by_tag: dict[str, dict[str, object]] = {}
    seen_release_ids: set[int] = set()
    for value in github_releases:
        if (
            not isinstance(value, dict)
            or set(value) != {"id", "tag", "url"}
        ):
            raise ValueError("GitHub Release observation fields are invalid")
        tag = value["tag"]
        release_id = value["id"]
        if tag == LEGACY_PUBLIC_TAG:
            continue
        if (
            not is_canonical_public_tag(tag)
            or isinstance(release_id, bool)
            or not isinstance(release_id, int)
            or release_id <= 0
            or not isinstance(value["url"], str)
            or not value["url"]
            or tag in releases_by_tag
            or release_id in seen_release_ids
        ):
            raise ValueError("GitHub Release observation is invalid or duplicated")
        releases_by_tag[tag] = value
        seen_release_ids.add(release_id)

    released_by_tag: dict[str, dict[str, object]] = {}
    publications_by_tag: dict[str, dict[str, object]] = {}
    for manifest in manifests.values():
        kind = manifest["manifest_type"]
        if kind == "released":
            tag = str(manifest["public_tag"])
            if tag in released_by_tag:
                raise ValueError("duplicate released public tag")
            released_by_tag[tag] = dict(manifest)
        elif kind == "publication":
            tag = str(manifest["public_tag"])
            if tag in publications_by_tag:
                raise ValueError("duplicate publication public tag")
            publications_by_tag[tag] = dict(manifest)

    anchors_by_tag: dict[str, dict[str, object]] = {}
    expected_anchor_fields = {
        "public_tag", "released_manifest_id", "anchor_ref", "tag_object",
        "commit", "tree", "closure_timestamp_utc",
    }
    for value in anchor_observations:
        if (
            not isinstance(value, dict)
            or set(value) != expected_anchor_fields
        ):
            raise ValueError("iteration anchor observation fields are invalid")
        tag = value["public_tag"]
        if (
            not is_canonical_public_tag(tag)
            or not _full_sha(value["tag_object"])
            or not _full_sha(value["commit"])
            or not _full_sha(value["tree"])
            or not isinstance(value["anchor_ref"], str)
            or not isinstance(value["closure_timestamp_utc"], str)
            or tag in anchors_by_tag
        ):
            raise ValueError("iteration anchor observation is invalid or duplicated")
        anchors_by_tag[tag] = value

    if (
        set(tags_by_name) != set(released_by_tag)
        or set(tags_by_name) != set(publications_by_tag)
        or set(tags_by_name) != set(releases_by_tag)
        or set(tags_by_name) != set(anchors_by_tag)
    ):
        raise ValueError(
            "complete public tag/release/lifecycle/anchor universe disagrees"
        )

    current = [
        tag
        for tag, manifest in released_by_tag.items()
        if manifest.get("release_line") == "principal"
        and manifest.get("final_release_sha") == main_sha
        and manifest.get("final_version") == main_version
    ]
    if len(current) > 1:
        raise ValueError("multiple terminal principal releases match current main")
    selected_tag = current[0] if current else None

    if selected_tag is not None:
        released = released_by_tag[selected_tag]
        publication = publications_by_tag[selected_tag]
        tag_observation = tags_by_name[selected_tag]
        try:
            parse_publication_evidence(
                stable_evidence_bytes,
                released,
                public_tag=selected_tag,
                tag_commit=str(tag_observation["commit"]),
                tag_tree=str(tag_observation["tree"]),
            )
        except PublicationEvidenceError as error:
            raise ValueError("current principal publication evidence is invalid") from error
        stable_evidence_digest = hashlib.sha256(stable_evidence_bytes).hexdigest()
        if stable_evidence_digest != publication["publication_evidence_digest"]:
            raise ValueError("current principal publication evidence digest mismatch")

    # Validate each released manifest against an independently observed
    # annotated iteration anchor. The observations are generated from fetched
    # Git objects, not copied from the released manifest fields.
    for tag in sorted(released_by_tag):
        released = released_by_tag[tag]
        publication = publications_by_tag[tag]
        tag_observation = tags_by_name[tag]
        github_release = releases_by_tag[tag]
        anchor = anchors_by_tag[tag]
        if (
            anchor["released_manifest_id"] != released["manifest_id"]
            or anchor["anchor_ref"] != released["anchor_ref"]
            or anchor["tag_object"] != released["anchor_sha"]
            or anchor["tree"] != released["anchor_tree"]
            or anchor["closure_timestamp_utc"]
            != released["closure_timestamp_utc"]
        ):
            raise ValueError(f"released manifest {tag} does not match its observed anchor")
        evidence_digest = (
            stable_evidence_digest
            if tag == selected_tag
            else str(publication["publication_evidence_digest"])
        )
        result = validate_release_consistency(
            released,
            publication,
            public_tag=tag,
            tag_commit=str(tag_observation["commit"]),
            tag_tree=str(tag_observation["tree"]),
            certified_tree=str(anchor["tree"]),
            github_release_id=github_release["id"],
            publication_evidence_digest=evidence_digest,
            lifecycle_records=lifecycle_records,
            anchor_tag_object=str(anchor["tag_object"]),
            anchor_tree=str(anchor["tree"]),
            anchor_closure_timestamp_utc=str(anchor["closure_timestamp_utc"]),
            candidate_ref=str(released["candidate_ref"]),
            candidate_commit=str(released["candidate_sha"]),
            candidate_tree=str(released["candidate_tree"]),
            canonical_tag_observations=canonical_tags,
            github_release_observations=github_releases,
            require_complete_namespace=True,
        )
        if result.get("status") != TERMINAL_CONSISTENT:
            raise ValueError(
                f"release universe is not terminal and consistent for {tag}: {result}"
            )

    return selected_tag


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--publication-manifest", type=Path)
    result.add_argument("--publication-evidence", type=Path)
    result.add_argument("--lifecycle-index", type=Path)
    result.add_argument("--canonical-tags", type=Path)
    result.add_argument("--github-releases", type=Path)
    result.add_argument("--anchor-observations", type=Path)
    result.add_argument("--stable-publication-evidence", type=Path)
    result.add_argument("--tag")
    result.add_argument("--tag-ref")
    result.add_argument("--tag-sha")
    result.add_argument("--tag-tree")
    result.add_argument("--tag-version")
    result.add_argument("--main-sha", required=True)
    result.add_argument("--main-version", required=True)
    result.add_argument("--github-env", type=Path, required=True)
    result.add_argument("--github-release-id", type=int, default=None)
    result.add_argument("--anchor-tag-object")
    result.add_argument("--anchor-tree")
    result.add_argument("--anchor-closure-timestamp-utc")
    result.add_argument("--candidate-ref")
    result.add_argument("--candidate-commit")
    result.add_argument("--candidate-tree")
    result.add_argument("--require-complete-lifecycle", action="store_true")
    result.add_argument("--stable-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.stable_only:
        return verify_stable_context(args)
    if (
        args.manifest is None
        or args.publication_manifest is None
        or args.publication_evidence is None
    ):
        return fail(
            "--manifest, --publication-manifest and --publication-evidence are required"
        )
    return verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
