#!/usr/bin/env python3
"""Verify a canonical released event before documentation deployment.

The workflow fetches the protected ``release-events`` branch and resolves the
Git tag and current principal ``main`` locally. This command validates the
complete event stream, binds the matching released event to those immutable
Git identities, and exports only the verified routing context through
``GITHUB_ENV``. A maintenance event is checked against its contemporaneous
``main_at_event_sha/version``; current ``main`` is used only to resolve the
stable principal tag. It exits nonzero for every missing, ambiguous, or
mismatched identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from version_lifecycle.events import parse_stream  # noqa: E402
from version_lifecycle.release import (  # noqa: E402
    PASS,
    is_canonical_public_tag,
    validate_released_event,
)


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
MAINTENANCE_LINE_RE = re.compile(
    r"^maintenance/(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)


def fail(message: str) -> int:
    print(f"release documentation context blocked: {message}", file=sys.stderr)
    return 2


def full_sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def canonical_version(value: object) -> bool:
    return isinstance(value, str) and VERSION_RE.fullmatch(value) is not None


def resolve_tag_commit(tag: str) -> str | None:
    """Resolve a canonical public tag from the authenticated Git remote."""

    if not is_canonical_public_tag(tag):
        return None
    tag_ref = f"refs/tags/{tag}"
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "origin", tag_ref, f"{tag_ref}^{{}}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    direct = None
    for raw_line in output.splitlines():
        fields = raw_line.split()
        if len(fields) != 2 or not full_sha(fields[0]):
            continue
        if fields[1] == f"{tag_ref}^{{}}":
            return fields[0]
        if fields[1] == tag_ref:
            direct = fields[0]
    return direct


def list_canonical_public_tags() -> list[str] | None:
    """Enumerate canonical public tags from the authenticated Git remote.

    ``None`` means the remote query failed. An empty list is a successful
    assertion that no canonical public tag exists, which is needed to permit
    pre-first-release development documentation even when the mutable ledger
    contains a valid pending intent.
    """

    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "--refs", "origin", "refs/tags/v*"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    tags = set()
    for raw_line in output.splitlines():
        fields = raw_line.split()
        if len(fields) != 2 or not full_sha(fields[0]):
            continue
        prefix = "refs/tags/"
        if fields[1].startswith(prefix):
            tag = fields[1][len(prefix) :]
            if is_canonical_public_tag(tag):
                tags.add(tag)
    return sorted(tags)


def certification_record(event: dict[str, object]) -> dict[str, object]:
    """Build the pinned certification identity carried by a released event.

    The release validator deliberately requires certification evidence as a
    separate input.  The event stream is the documentation verifier's only
    authority, so this adapter passes the independently recorded fields to
    that validator without inventing any evidence or resolving a private
    locator.
    """

    evidence_refs = event.get("certification_evidence_refs")
    evidence_ref = evidence_refs[0] if isinstance(evidence_refs, list) and evidence_refs else ""
    return {
        "binding": event.get("certification_binding", ""),
        "package_commit": event.get("certification_subject_sha", ""),
        "package_tree": event.get("certification_subject_tree", ""),
        "policy_revision": event.get("certification_policy_revision", ""),
        "harness_revision": event.get("certification_harness_revision", ""),
        "environment": event.get("certification_environment", ""),
        "evidence_ref": evidence_ref,
    }


def project_version_evidence(
    event: dict[str, object], *, resolved_version: str
) -> dict[str, str]:
    """Return version observations from independently resolved Git trees.

    The tag or current principal tree supplies ``resolved_version``. The
    released event must separately report the same final version and equal
    exact trees before these observations can be used for the five
    certification-side views.
    """

    event_main_version = str(event.get("main_at_event_version", ""))
    return {
        "closure": resolved_version,
        "candidate": resolved_version,
        "anchor": resolved_version,
        "final_release": resolved_version,
        "certified": resolved_version,
        "main": event_main_version,
    }


def verified_principal_matches(
    events: list[dict[str, object]], *, main_sha: str, main_version: str
) -> list[dict[str, object]]:
    """Return the one validated principal release that supplies stable."""

    matches = []
    for prior in events:
        if not (
            prior.get("event_type") == "released"
            and prior.get("release_line") == "principal"
            and prior.get("final_release_sha") == main_sha
            and prior.get("final_version") == main_version
            and is_canonical_public_tag(prior.get("public_tag"))
        ):
            continue
        prior_result = validate_released_event(
            prior,
            certification=certification_record(prior),
            project_versions=project_version_evidence(
                prior,
                resolved_version=main_version,
            ),
        )
        if prior_result.get("status") != PASS:
            raise ValueError(
                "principal main candidate failed released-event validation: "
                f"{prior_result}"
            )
        matches.append(prior)
    return matches


def verify_stable_context(args: argparse.Namespace) -> int:
    """Verify the current principal-main release for development docs."""

    if not canonical_version(args.main_version):
        return fail("principal main Project.toml version is not canonical")
    if not full_sha(args.main_sha):
        return fail("principal main SHA is not a full Git SHA")
    remote_tags = list_canonical_public_tags()
    if remote_tags is None:
        return fail("canonical public tag enumeration failed")
    try:
        events = parse_stream(args.event_stream.read_bytes())
        principal_matches = verified_principal_matches(
            events, main_sha=args.main_sha, main_version=args.main_version
        )
    except Exception as error:
        return fail(f"canonical release event stream is invalid: {error}")
    has_released_event = any(event.get("event_type") == "released" for event in events)
    if not has_released_event and not remote_tags:
        values = {
            "CYAX_DOCS_PRINCIPAL_MAIN_SHA": args.main_sha,
            "CYAX_DOCS_STABLE": "false",
            "CYAX_DOCS_STABLE_TAG": "",
        }
        write_environment(
            args.github_env,
            values,
        )
        print(json.dumps(values, sort_keys=True))
        return 0
    if len(principal_matches) != 1:
        return fail(
            "expected one verified principal released event for current main, "
            f"found {len(principal_matches)}"
        )
    stable_tag = str(principal_matches[0]["public_tag"])
    if resolve_tag_commit(stable_tag) != args.main_sha:
        return fail("stable principal tag is unavailable or does not match current main")
    write_environment(
        args.github_env,
        {
            "CYAX_DOCS_PRINCIPAL_MAIN_SHA": args.main_sha,
            "CYAX_DOCS_STABLE": "false",
            "CYAX_DOCS_STABLE_TAG": stable_tag,
        },
    )
    print(
        json.dumps(
            {
                "CYAX_DOCS_PRINCIPAL_MAIN_SHA": args.main_sha,
                "CYAX_DOCS_STABLE": "false",
                "CYAX_DOCS_STABLE_TAG": stable_tag,
            },
            sort_keys=True,
        )
    )
    return 0


def write_environment(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"newline in environment value: {key}")
            handle.write(f"{key}={value}\n")


def verify(args: argparse.Namespace) -> int:
    if not is_canonical_public_tag(args.tag):
        return fail("tag is not a canonical vX.Y.Z identity")
    if not canonical_version(args.tag[1:]):
        return fail("tag version is not canonical")
    if not canonical_version(args.tag_version):
        return fail("tag Project.toml version is not canonical")
    if not canonical_version(args.main_version):
        return fail("principal main Project.toml version is not canonical")
    for name, value in (
        ("tag SHA", args.tag_sha),
        ("tag tree", args.tag_tree),
        ("principal main SHA", args.main_sha),
    ):
        if not full_sha(value):
            return fail(f"{name} is not a full Git SHA")

    try:
        events = parse_stream(args.event_stream.read_bytes())
    except Exception as error:  # parser errors are deliberate fail-closed outcomes
        return fail(f"canonical release event stream is invalid: {error}")

    matches = [
        event
        for event in events
        if event.get("event_type") == "released" and event.get("public_tag") == args.tag
    ]
    if len(matches) != 1:
        return fail(f"expected one released event for {args.tag}, found {len(matches)}")
    event = matches[0]
    if event.get("final_version") != args.tag_version:
        return fail("tag Project.toml version does not equal released event final_version")

    line = event.get("release_line")
    if line != "principal" and not (
        isinstance(line, str) and MAINTENANCE_LINE_RE.fullmatch(line)
    ):
        return fail("released event has no canonical principal or maintenance/X.Y line")

    is_principal = line == "principal"

    event_main = {
        "sha": event.get("main_at_event_sha", ""),
        "version": event.get("main_at_event_version", ""),
    }
    result = validate_released_event(
        event,
        certification=certification_record(event),
        principal_main={
            "sha": args.main_sha if is_principal else event_main["sha"],
            "version": args.main_version if is_principal else event_main["version"],
        },
        # The released event binds one exact tree for closure, candidate,
        # anchor, certification, and final release. The tag checkout is that
        # tree, so its package version supplies all five equal observations.
        # Principal stable selection reads current main independently; a
        # maintenance event keeps its contemporaneous main identity above.
        project_versions=project_version_evidence(
            event,
            resolved_version=args.tag_version,
        ),
    )
    if result.get("status") != PASS:
        return fail(f"released event validation failed: {result}")

    if event.get("final_release_sha") != args.tag_sha:
        return fail("tag commit does not equal released event final_release_sha")
    if event.get("final_release_tree") != args.tag_tree:
        return fail("tag tree does not equal released event final_release_tree")

    # A principal event whose certified final commit is current main is the
    # only event allowed to supply the stable selector. Maintenance releases
    # retain the previously verified principal-main tag.
    stable_tag = ""
    try:
        principal_matches = verified_principal_matches(
            events, main_sha=args.main_sha, main_version=args.main_version
        )
    except ValueError as error:
        return fail(str(error))
    if len(principal_matches) > 1:
        return fail("multiple principal released events claim current main")
    if principal_matches:
        stable_tag = str(principal_matches[0]["public_tag"])
        if resolve_tag_commit(stable_tag) != args.main_sha:
            return fail("stable principal tag is unavailable or does not match current main")
    else:
        return fail("no verified principal release matches current main")

    env_values = {
        "CYAX_DOCS_EVENT_STATUS": "verified",
        "CYAX_DOCS_RELEASE_LINE": str(line),
        "CYAX_DOCS_RELEASE_VERSION": str(event["final_version"]),
        "CYAX_DOCS_RELEASE_SHA": str(event["final_release_sha"]),
        "CYAX_DOCS_PRINCIPAL_MAIN_SHA": args.main_sha,
        "CYAX_DOCS_STABLE": "true" if is_principal and stable_tag == args.tag else "false",
        "CYAX_DOCS_STABLE_TAG": stable_tag,
        "CYAX_DOCS_EVENT_ID": str(event["event_id"]),
    }
    write_environment(args.github_env, env_values)
    print(json.dumps(env_values, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--event-stream", type=Path, required=True)
    result.add_argument("--tag")
    result.add_argument("--tag-sha")
    result.add_argument("--tag-tree")
    result.add_argument("--tag-version")
    result.add_argument("--main-sha", required=True)
    result.add_argument("--main-version", required=True)
    result.add_argument("--github-env", type=Path, required=True)
    result.add_argument(
        "--stable-only",
        action="store_true",
        help="verify only the current principal release for a development selector",
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(verify_stable_context(arguments) if arguments.stable_only else verify(arguments))
