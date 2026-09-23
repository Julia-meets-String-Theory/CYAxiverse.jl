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
    validate_manifest,
)
from version_lifecycle.publication_evidence import (  # noqa: E402
    PublicationEvidenceError,
    parse_publication_evidence,
)
from version_lifecycle.release import PASS, validate_release_consistency, is_canonical_public_tag  # noqa: E402
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
    is_current_principal = (
        released.get("release_line") == "principal"
        and released.get("final_release_sha") == args.main_sha
        and released.get("final_version") == main_version.canonical
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
        # Pin Documenter's stable selector to the current principal main
        # version.  For maintenance tags this preserves stable while the
        # maintenance version receives its own immutable directory.
        "CYAX_DOCS_STABLE_TAG": f"v{main_version.canonical}",
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
    values = {
        "CYAX_DOCS_PRINCIPAL_MAIN_SHA": args.main_sha,
        "CYAX_DOCS_STABLE": "false",
        # Development builds keep the stable selector bound to the verified
        # current-main package version, without treating the vmm build as
        # stable or accepting a caller-provided tag.
        "CYAX_DOCS_STABLE_TAG": f"v{version.canonical}",
    }
    write_environment(args.github_env, values)
    print(json.dumps(values, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--publication-manifest", type=Path)
    result.add_argument("--publication-evidence", type=Path)
    result.add_argument("--lifecycle-index", type=Path)
    result.add_argument("--canonical-tags", type=Path)
    result.add_argument("--github-releases", type=Path)
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
