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
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from version_lifecycle.events import parse_stream  # noqa: E402
from version_lifecycle.release import (  # noqa: E402
    PASS,
    is_canonical_public_tag,
    validate_released_event,
)
from version_lifecycle.versions import (  # noqa: E402
    maintenance_line,
    parse_package_version,
)


SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def fail(message: str) -> int:
    print(f"release documentation context blocked: {message}", file=sys.stderr)
    return 2


def full_sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def canonical_version(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return parse_package_version(value).is_final
    except (TypeError, ValueError):
        return False


def canonical_tag_ref(value: object) -> str | None:
    """Return the tag name only for an exact canonical full tag ref."""

    if not isinstance(value, str) or not value.startswith("refs/tags/"):
        return None
    tag = value.removeprefix("refs/tags/")
    if value != f"refs/tags/{tag}" or not is_canonical_public_tag(tag):
        return None
    return tag


def _command_text(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        list(command), cwd=ROOT if cwd is None else cwd,
        stderr=subprocess.STDOUT, text=True
    ).strip()


def _remote_ref_commit(ref: str) -> str:
    output = _command_text("git", "ls-remote", "origin", ref, f"{ref}^{{}}")
    direct = None
    peeled = None
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 2 or not full_sha(fields[0]):
            continue
        if fields[1] == ref:
            direct = fields[0]
        elif fields[1] == f"{ref}^{{}}":
            peeled = fields[0]
    commit = peeled or direct
    if commit is None:
        raise ValueError(f"remote ref is unavailable: {ref}")
    return commit


def _project_version(repository: Path, commit: str) -> str:
    raw = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{commit}:Project.toml"],
        stderr=subprocess.STDOUT,
    )
    value = tomllib.loads(raw.decode("utf-8")).get("version")
    if not canonical_version(value):
        raise ValueError("resolved Project.toml version is not canonical")
    return str(value)


def resolve_repository_evidence(
    event: dict[str, object],
    *,
    tag_ref: str,
    tag_sha: str,
    tag_tree: str,
    tag_version: str,
    main_sha: str,
    main_version: str,
) -> dict[str, dict[str, str]]:
    """Resolve Git-backed release identities independently of the event.

    Gate A cannot re-run external certification policy or live repository
    settings here.  It can and must independently resolve every Git ref,
    commit, tree, and Project.toml version used to authorize documentation.
    The event remains the durable authority for non-Git attestation metadata.
    """

    if canonical_tag_ref(tag_ref) != event.get("public_tag"):
        raise ValueError("workflow tag ref is not the event's exact canonical tag ref")
    anchor_ref = event.get("anchor_ref")
    candidate_ref = event.get("candidate_ref")
    if not isinstance(anchor_ref, str) or not isinstance(candidate_ref, str):
        raise ValueError("released event is missing durable Git refs")
    expected_refs = {
        "tag": (tag_ref, tag_sha),
        "anchor": (anchor_ref, event.get("anchor_sha")),
        "candidate": (candidate_ref, event.get("candidate_sha")),
        "main": ("refs/heads/main", main_sha),
    }
    for name, (ref, expected_sha) in expected_refs.items():
        if not full_sha(expected_sha) or _remote_ref_commit(ref) != expected_sha:
            raise ValueError(f"{name} ref does not resolve to the required commit")

    endpoint = _command_text("git", "remote", "get-url", "origin")
    with tempfile.TemporaryDirectory(prefix="cyax-docs-release-evidence-") as temporary:
        repository = Path(temporary) / "authority.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(repository)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for ref, _ in expected_refs.values():
            subprocess.run(
                [
                    "git", "-C", str(repository), "fetch", "--quiet",
                    "--no-tags", "--no-write-fetch-head", endpoint, ref,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        event_main_sha = event.get("main_at_event_sha")
        certification_sha = event.get("certification_subject_sha")
        for commit in (event_main_sha, certification_sha):
            if not full_sha(commit):
                raise ValueError("event carries an invalid independently resolved commit")
            present = subprocess.run(
                ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if present.returncode != 0:
                subprocess.run(
                    [
                        "git", "-C", str(repository), "fetch", "--quiet",
                        "--no-tags", "--no-write-fetch-head", endpoint, str(commit),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

        def identity(commit: object) -> dict[str, str]:
            if not full_sha(commit):
                raise ValueError("resolved identity is not a full Git SHA")
            sha = str(commit)
            tree = _command_text(
                "git", "rev-parse", "--verify", f"{sha}^{{tree}}", cwd=repository
            )
            if not full_sha(tree):
                raise ValueError("resolved tree is not a full Git SHA")
            return {
                "sha": sha,
                "tree": tree,
                "version": _project_version(repository, sha),
            }

        evidence = {
            name: identity(expected_sha)
            for name, (_, expected_sha) in expected_refs.items()
        }
        evidence["certified"] = identity(certification_sha)
        evidence["event_main"] = identity(event_main_sha)

    if evidence["tag"] != {
        "sha": tag_sha, "tree": tag_tree, "version": tag_version
    }:
        raise ValueError("workflow tag checkout does not match remote Git evidence")
    if evidence["main"]["version"] != main_version:
        raise ValueError("workflow main version does not match remote Git evidence")
    return evidence


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


def certification_record(
    event: dict[str, object], repository_evidence: dict[str, dict[str, str]]
) -> dict[str, object]:
    """Build the pinned certification identity carried by a released event.

    The release validator deliberately requires certification evidence as a
    separate input. Git commit/tree identity comes from the independent
    repository resolver. Policy, harness, environment, and evidence-reference
    fields remain durable ledger assertions at Gate A; live settings and
    external attestation resolution belong to the later release gate.
    """

    evidence_refs = event.get("certification_evidence_refs")
    evidence_ref = evidence_refs[0] if isinstance(evidence_refs, list) and evidence_refs else ""
    record = {
        "binding": event.get("certification_binding", ""),
        "package_commit": repository_evidence["certified"]["sha"],
        "package_tree": repository_evidence["certified"]["tree"],
        "policy_revision": event.get("certification_policy_revision", ""),
        "harness_revision": event.get("certification_harness_revision", ""),
        "environment": event.get("certification_environment", ""),
        "evidence_ref": evidence_ref,
        "evidence_refs": evidence_refs,
    }
    if "certification_transfer_evidence" in event:
        record["transfer_evidence"] = event["certification_transfer_evidence"]
    return record


def project_version_evidence(
    repository_evidence: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Return version observations from independently resolved Git trees.

    Every returned version was read from an independently resolved Git tree.
    The released event must separately agree with these observations.
    """

    return {
        "closure": repository_evidence["anchor"]["version"],
        "candidate": repository_evidence["candidate"]["version"],
        "anchor": repository_evidence["anchor"]["version"],
        "final_release": repository_evidence["tag"]["version"],
        "certified": repository_evidence["certified"]["version"],
        "main": repository_evidence["event_main"]["version"],
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
        prior_evidence = resolve_repository_evidence(
            prior,
            tag_ref=f"refs/tags/{prior['public_tag']}",
            tag_sha=main_sha,
            tag_tree=str(prior.get("final_release_tree", "")),
            tag_version=main_version,
            main_sha=main_sha,
            main_version=main_version,
        )
        prior_result = validate_released_event(
            prior,
            certification=certification_record(prior, prior_evidence),
            project_versions=project_version_evidence(prior_evidence),
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
    if canonical_tag_ref(args.tag_ref) != args.tag:
        return fail("tag ref is not exactly refs/tags/<canonical-tag>")
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
    if line != "principal":
        try:
            maintenance_line(line)
        except (TypeError, ValueError):
            return fail("released event has no canonical principal or maintenance/X.Y line")

    is_principal = line == "principal"

    try:
        repository_evidence = resolve_repository_evidence(
            event,
            tag_ref=args.tag_ref,
            tag_sha=args.tag_sha,
            tag_tree=args.tag_tree,
            tag_version=args.tag_version,
            main_sha=args.main_sha,
            main_version=args.main_version,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return fail(f"independent repository identity resolution failed: {error}")

    result = validate_released_event(
        event,
        certification=certification_record(event, repository_evidence),
        principal_main={
            "sha": repository_evidence["main" if is_principal else "event_main"]["sha"],
            "version": repository_evidence["main" if is_principal else "event_main"]["version"],
        },
        # Git refs and commits were resolved independently above. Principal
        # stable selection uses current main; a maintenance event uses its
        # independently resolved contemporaneous main commit.
        project_versions=project_version_evidence(repository_evidence),
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
    result.add_argument("--tag-ref")
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
