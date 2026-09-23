#!/usr/bin/env python3
"""Fetch independent Git and GitHub evidence for documentation stable routing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FULL_SHA1 = re.compile(r"^[0-9a-f]{40}$")
GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
sys.path.insert(0, str(ROOT / "scripts"))

from version_lifecycle.manifests import (  # noqa: E402
    ManifestError,
    validate_lifecycle_graph,
)
from version_lifecycle.publication_evidence import (  # noqa: E402
    PublicationEvidenceError,
    parse_publication_evidence,
)
from version_lifecycle.release import (  # noqa: E402
    LEGACY_PUBLIC_TAG,
    is_canonical_public_tag,
)
from version_lifecycle.static import (  # noqa: E402
    StaticValidationError,
    _annotated_anchor_payload,
)
from version_lifecycle.versions import parse_package_version  # noqa: E402


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def _canonical_tag_index(observations: object) -> dict[str, dict[str, Any]]:
    if not isinstance(observations, list):
        raise ValueError("canonical tag observations must be an array")
    result: dict[str, dict[str, Any]] = {}
    for value in observations:
        if (
            not isinstance(value, dict)
            or set(value) != {"ref", "tag", "commit", "tree"}
            or not is_canonical_public_tag(value.get("tag"))
            or not isinstance(value.get("commit"), str)
            or not FULL_SHA1.fullmatch(value["commit"])
            or not isinstance(value.get("tree"), str)
            or not FULL_SHA1.fullmatch(value["tree"])
        ):
            raise ValueError("canonical tag observation is invalid")
        tag = value["tag"]
        if value["ref"] != f"refs/tags/{tag}" or tag in result:
            raise ValueError("canonical tag observation is inconsistent or duplicated")
        result[tag] = value
    return result


def _canonical_github_release_index(
    observations: object,
) -> dict[str, dict[str, Any]]:
    if not isinstance(observations, list):
        raise ValueError("GitHub Release observations must be an array")
    result: dict[str, dict[str, Any]] = {}
    seen_tags: set[str] = set()
    seen_ids: set[int] = set()
    for value in observations:
        if (
            not isinstance(value, dict)
            or set(value) != {"id", "tag", "url"}
        ):
            raise ValueError("GitHub Release observation fields are invalid")
        tag = value["tag"]
        release_id = value["id"]
        release_url = value["url"]
        if (
            not isinstance(tag, str)
            or tag in seen_tags
            or isinstance(release_id, bool)
            or not isinstance(release_id, int)
            or release_id <= 0
            or release_id in seen_ids
            or not isinstance(release_url, str)
            or not release_url
        ):
            raise ValueError("duplicate or invalid GitHub Release observation")
        seen_tags.add(tag)
        seen_ids.add(release_id)
        if tag == LEGACY_PUBLIC_TAG:
            # GitHub still advertises the grandfathered legacy release. It is
            # not a canonical public tag and must not enter lifecycle matching.
            continue
        if not is_canonical_public_tag(tag):
            raise ValueError("GitHub Release tag is not canonical")
        result[tag] = value
    return result


def _stable_release(
    manifests: dict[str, dict[str, Any]],
    main_sha: str,
    main_version: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = [
        manifest
        for manifest in manifests.values()
        if manifest.get("manifest_type") == "released"
        and manifest.get("release_line") == "principal"
        and manifest.get("final_release_sha") == main_sha
        and manifest.get("final_version") == main_version
    ]
    if len(candidates) > 1:
        raise ValueError("multiple principal releases match current main")
    if not candidates:
        return None
    released = candidates[0]
    publications = [
        manifest
        for manifest in manifests.values()
        if manifest.get("manifest_type") == "publication"
        and manifest.get("released_manifest_id") == released.get("manifest_id")
        and manifest.get("public_tag") == released.get("public_tag")
    ]
    if len(publications) != 1:
        raise ValueError("current principal release has no unique publication manifest")
    return released, publications[0]


def prepare(args: argparse.Namespace) -> None:
    if not FULL_SHA1.fullmatch(args.main_sha):
        raise ValueError("principal main SHA must be a full Git SHA")
    if not GITHUB_REPOSITORY.fullmatch(args.github_repository):
        raise ValueError("GitHub repository must be an exact owner/repository identity")
    lifecycle_records = json.loads(args.lifecycle_index.read_text(encoding="utf-8"))
    canonical_tags = json.loads(args.canonical_tags.read_text(encoding="utf-8"))
    github_releases = json.loads(args.github_releases.read_text(encoding="utf-8"))
    graph = validate_lifecycle_graph(lifecycle_records)
    manifests = {ref: dict(value) for ref, value in graph.manifests_by_ref.items()}
    main_version = parse_package_version(args.main_version)
    if main_version.is_dev:
        raise ValueError("principal main must not carry a DEV version")
    tag_observations = _canonical_tag_index(canonical_tags)
    releases_by_tag = _canonical_github_release_index(github_releases)

    anchor_observations = []
    released_manifests = sorted(
        (
            manifest
            for manifest in manifests.values()
            if manifest.get("manifest_type") == "released"
        ),
        key=lambda manifest: str(manifest["public_tag"]),
    )
    for released in released_manifests:
        ref = str(released["anchor_ref"])
        if not ref.startswith("refs/tags/iterations/"):
            raise ValueError("released manifest anchor ref is not protected")
        subprocess.run(
            ["git", "fetch", "--no-tags", "--", "origin", f"{ref}:{ref}"],
            cwd=args.repository,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        tag_object = _git(args.repository, "rev-parse", ref)
        commit = _git(args.repository, "rev-parse", f"{ref}^{{commit}}")
        tree = _git(args.repository, "rev-parse", f"{ref}^{{tree}}")
        closure_timestamp = _annotated_anchor_payload(
            args.repository,
            tag_object,
            ref.removeprefix("refs/tags/"),
            commit,
        )
        if (
            tag_object != released["anchor_sha"]
            or tree != released["anchor_tree"]
            or closure_timestamp != released["closure_timestamp_utc"]
        ):
            raise ValueError("released manifest does not match its observed iteration anchor")
        anchor_observations.append({
            "public_tag": released["public_tag"],
            "released_manifest_id": released["manifest_id"],
            "anchor_ref": ref,
            "tag_object": tag_object,
            "commit": commit,
            "tree": tree,
            "closure_timestamp_utc": closure_timestamp,
        })

    selected = _stable_release(
        manifests, args.main_sha, main_version.canonical
    )
    evidence_bytes = b""
    if selected is not None:
        released, publication = selected
        tag = str(released["public_tag"])
        if not is_canonical_public_tag(tag):
            raise ValueError("stable principal release has a noncanonical public tag")
        tag_observation = tag_observations.get(tag)
        github_observation = releases_by_tag.get(tag)
        if tag_observation is None or github_observation is None:
            raise ValueError("stable principal release lacks canonical tag or GitHub Release")
        if (
            github_observation.get("id") != publication.get("github_release_id")
            or github_observation.get("url") != publication.get("github_release_url")
        ):
            raise ValueError("stable publication does not match GitHub Release observation")

        release_detail = json.loads(subprocess.check_output(
            [
                "gh", "api",
                f"repos/{args.github_repository}/releases/tags/{tag}",
            ],
            cwd=args.repository,
        ))
        if (
            release_detail.get("id") != publication["github_release_id"]
            or release_detail.get("tag_name") != tag
            or release_detail.get("html_url") != publication["github_release_url"]
        ):
            raise ValueError("stable GitHub Release identity is contradictory")
        asset_name = publication["publication_evidence_ref"]
        matching_assets = [
            asset
            for asset in release_detail.get("assets", [])
            if isinstance(asset, dict) and asset.get("name") == asset_name
        ]
        if len(matching_assets) != 1:
            raise ValueError("stable publication evidence asset is missing or ambiguous")
        asset_id = matching_assets[0].get("id")
        if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id <= 0:
            raise ValueError("stable publication evidence asset ID is invalid")
        evidence_bytes = subprocess.check_output(
            [
                "gh", "api", "-H", "Accept: application/octet-stream",
                f"repos/{args.github_repository}/releases/assets/{asset_id}",
            ],
            cwd=args.repository,
        )
        try:
            parse_publication_evidence(
                evidence_bytes,
                released,
                public_tag=tag,
                tag_commit=str(tag_observation["commit"]),
                tag_tree=str(tag_observation["tree"]),
            )
        except PublicationEvidenceError as error:
            raise ValueError("stable publication evidence bytes are invalid") from error
        if hashlib.sha256(evidence_bytes).hexdigest() != publication[
            "publication_evidence_digest"
        ]:
            raise ValueError("stable publication evidence digest does not match manifest")

    args.anchor_observations.write_text(
        json.dumps(anchor_observations, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    args.stable_evidence.write_bytes(evidence_bytes)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=ROOT)
    result.add_argument("--github-repository", required=True)
    result.add_argument("--lifecycle-index", type=Path, required=True)
    result.add_argument("--canonical-tags", type=Path, required=True)
    result.add_argument("--github-releases", type=Path, required=True)
    result.add_argument("--main-sha", required=True)
    result.add_argument("--main-version", required=True)
    result.add_argument("--anchor-observations", type=Path, required=True)
    result.add_argument("--stable-evidence", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        prepare(args)
    except (
        OSError, UnicodeError, json.JSONDecodeError, ManifestError,
        PublicationEvidenceError, StaticValidationError, TypeError, ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"stable release authority blocked: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
