#!/usr/bin/env python3
"""Read-only Gate A lifecycle readiness CLI.

The CLI deliberately has no writer operation.  It reads the canonical static
snapshot and the local ``release-events`` branch, then reports a stable JSON
object that can be consumed by CI or a release operator.  ``--dry-run`` is
accepted explicitly and is always true in the report; the command never
creates refs, commits events, or changes a checkout.

Examples::

    python scripts/version_lifecycle_cli.py readiness \
        --repo /path/to/fixture --dry-run --principal-closed 0.2.0
    python scripts/version_lifecycle_cli.py snapshot --repo /path/to/fixture
    python scripts/version_lifecycle_cli.py events --repo /path/to/fixture

Exit status is 0 for a ready result, 2 for a structured blocked result, and 1
for invalid CLI input or an unexpected implementation error.  JSON is always
written to stdout, including errors.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import uuid


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from version_lifecycle import (  # noqa: E402
    BlockedResult,
    GlobalAllocationView,
    global_allocation_view,
    select_maintenance_version,
    select_principal_sentinel,
    static_snapshot,
)
from version_lifecycle.static import StaticSnapshot  # noqa: E402
from version_lifecycle.events import parse_stream  # noqa: E402
from version_lifecycle.certification import is_safe_public_value  # noqa: E402
from version_lifecycle.git_refs import validate_remote  # noqa: E402
from version_lifecycle.writer import (  # noqa: E402
    BranchUnavailable,
    LedgerHead,
    ReleaseEventWriter,
    WriterError,
)


DEFAULT_EVENT_BRANCH = "release-events"
DEFAULT_EVENT_STREAM = "release-events.jsonl"
DEFAULT_REMOTE = "origin"
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")


class CliArgumentError(ValueError):
    """An argparse failure that can be reported through the JSON contract."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _jsonable(value: Any) -> Any:
    """Convert lifecycle dataclasses and version values to JSON primitives."""

    if hasattr(value, "canonical"):
        return str(value.canonical)
    if isinstance(value, StaticSnapshot):
        return value.to_dict()
    if isinstance(value, GlobalAllocationView):
        return {
            "status": value.status,
            "event_head_commit": value.event_head_commit,
            "static_snapshot_digest": value.snapshot_digest,
            "static_occupied_versions": sorted(value.static_occupied),
            "mutable_occupied_versions": sorted(value.mutable_occupied),
            "occupied_versions": sorted(value.occupied),
        }
    if isinstance(value, BlockedResult):
        return {
            "status": value.status,
            "reason_code": value.reason_code,
            "detail": value.detail,
        }
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _redact_paths(value: Any, paths: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for path in paths:
            if path:
                value = value.replace(path, "<repository>")
        return value
    if isinstance(value, Mapping):
        return {key: _redact_paths(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item, paths) for item in value]
    return value


def _sanitize_report(value: Any, *, key: str | None = None) -> Any:
    """Keep transport identities and untrusted diagnostics out of CLI JSON."""

    if isinstance(value, Mapping):
        return {name: _sanitize_report(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_report(item, key=key) for item in value]
    if isinstance(value, str):
        if key == "remote":
            return "configured"
        if key in {"detail", "local_read_detail"}:
            # Exception text can contain a configured URL, credentials, or a
            # machine path.  The stable reason_code carries the failure.
            return "detail omitted; use reason_code"
        if key in {"advertised_ref", "advertised_head"}:
            return "<redacted>"
        if key in {
            "selector", "branch", "stream_path", "remote_ref",
        } and (
            "@" in value
            or "://" in value
            or not is_safe_public_value(value, key=key)
        ):
            return "<redacted>"
    return value


def _write(
    result: Mapping[str, Any],
    *,
    exit_code: int = 0,
    redact_paths: tuple[str, ...] = (),
) -> int:
    """Emit one deterministic JSON object and return the desired exit code."""

    payload = _sanitize_report(_redact_paths(_jsonable(result), redact_paths))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return exit_code


def _blocked(reason_code: str, detail: str, **fields: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "BLOCKED",
        "reason_code": reason_code,
        "detail": detail,
    }
    result.update(fields)
    # Callers may attach a READY component report (for example the combined
    # allocation view); the enclosing result remains blocked.
    result["status"] = "BLOCKED"
    return result


def _static_report(args: argparse.Namespace) -> tuple[dict[str, Any], StaticSnapshot | None]:
    result = static_snapshot(
        args.repo,
        source_repository=args.source_repository,
        selector=args.selector,
        remote=args.remote,
    )
    if isinstance(result, BlockedResult):
        return _blocked(
            result.reason_code,
            result.detail,
            selector=args.selector,
            remote=args.remote,
        ), None
    assert isinstance(result, StaticSnapshot)
    return {
        "status": "READY",
        "snapshot_digest": result.snapshot_digest,
        "source_commit": result.source_commit,
        "source_tree": result.source_tree,
        "source_repository": result.source_repository,
        "canonical_static_iteration_source": result.canonical_source,
        "iterations_toml_sha256": result.file_digest,
        "ref_set_digest": result.ref_set_digest,
        "tag_set_digest": result.tag_set_digest,
        "occupied_versions": list(result.occupied_versions),
        "iteration_ref_bindings": list(result.ref_bindings),
        "public_tag_bindings": list(result.tag_bindings),
    }, result


def _event_report(args: argparse.Namespace) -> tuple[dict[str, Any], LedgerHead | None]:
    """Read the remote event authority and retain local state as diagnostics.

    The local branch is a checkout cache.  The advertised remote object is
    fetched into a disposable ref so the head, tree topology, and stream bytes
    all come from one exact remote identity.  The disposable ref is removed in
    ``finally``; no branch or tag is created by the CLI.
    """

    try:
        writer = ReleaseEventWriter(
            args.repo,
            branch=args.event_branch,
            stream_path=args.event_stream,
        )
        remote_ref = f"refs/heads/{args.event_branch}"
        advertised = writer._git(  # type: ignore[attr-defined]
            ["ls-remote", "--refs", args.remote, remote_ref]
        ).stdout.decode("utf-8", errors="strict").splitlines()
        if not advertised:
            return _blocked(
                "EVENT_AUTHORITY_UNAVAILABLE",
                f"remote {args.remote!r} does not advertise {remote_ref}",
                remote=args.remote,
                remote_ref=remote_ref,
                branch=args.event_branch,
                stream_path=args.event_stream,
            ), None
        if len(advertised) != 1 or "\t" not in advertised[0]:
            return _blocked(
                "EVENT_AUTHORITY_DIVERGENT",
                "remote event ref advertisement is ambiguous",
                remote=args.remote,
                remote_ref=remote_ref,
            ), None
        remote_head, advertised_ref = advertised[0].split("\t", 1)
        if advertised_ref != remote_ref or _GIT_OBJECT_RE.fullmatch(remote_head) is None:
            return _blocked(
                "EVENT_AUTHORITY_DIVERGENT",
                "remote event ref advertisement has an invalid identity",
                remote=args.remote,
                remote_ref=remote_ref,
                advertised_ref=advertised_ref,
                advertised_head=remote_head,
            ), None

        disposable_ref = f"refs/codex/cli-event/{uuid.uuid4().hex}"
        try:
            writer._git(  # type: ignore[attr-defined]
                [
                    "fetch",
                    "--no-tags",
                    "--no-write-fetch-head",
                    args.remote,
                    f"{remote_ref}:{disposable_ref}",
                ]
            )
            fetched_head = writer._git(  # type: ignore[attr-defined]
                ["rev-parse", "--verify", disposable_ref]
            ).stdout.decode("ascii", errors="strict").strip()
            if fetched_head != remote_head:
                return _blocked(
                    "EVENT_AUTHORITY_DIVERGENT",
                    "fetched event object differs from advertised remote head",
                    remote=args.remote,
                    remote_ref=remote_ref,
                    advertised_head=remote_head,
                    fetched_head=fetched_head,
                ), None
            try:
                raw = writer._git(  # type: ignore[attr-defined]
                    ["show", f"{disposable_ref}:{args.event_stream}"]
                ).stdout
                writer._verify_stream_topology(fetched_head, raw)
                events = tuple(parse_stream(raw))
            except (WriterError, ValueError) as error:
                return _blocked(
                    "EVENT_AUTHORITY_DIVERGENT",
                    f"remote event stream is invalid: {error}",
                    remote=args.remote,
                    remote_ref=remote_ref,
                    remote_head=remote_head,
                    stream_path=args.event_stream,
                ), None
        finally:
            writer._git(  # type: ignore[attr-defined]
                ["update-ref", "-d", disposable_ref], check=False
            )

        # Read the local cache through the writer's public read path when it
        # exists.  A stale cache is diagnostic only; allocation binds to the
        # remote head above.  A malformed local cache cannot make a valid
        # remote authority appear valid or change its bytes.
        local_head: LedgerHead | None = None
        local_error: str | None = None
        if writer.branch_exists():
            try:
                local_head = writer.read_head()
            except (BranchUnavailable, OSError, ValueError, RuntimeError) as error:
                local_error = str(error)
        head = LedgerHead(commit=remote_head, raw=raw, events=events)
        report: dict[str, Any] = {
            "status": "READY",
            "authority": "remote",
            "remote": args.remote,
            "remote_ref": remote_ref,
            "branch": args.event_branch,
            "stream_path": args.event_stream,
            "head_commit": head.commit,
            "remote_head_commit": head.commit,
            "stream_sha256": hashlib.sha256(head.raw).hexdigest(),
            "event_count": len(head.events),
            "event_ids": [event["event_id"] for event in head.events],
            "event_types": [event["event_type"] for event in head.events],
            "local_head_commit": None if local_head is None else local_head.commit,
            "local_head_stale": local_head is not None and local_head.commit != head.commit,
        }
        if local_error is not None:
            report["local_read_status"] = "INVALID"
            report["local_read_detail"] = local_error
        elif local_head is None:
            report["local_read_status"] = "ABSENT"
        else:
            report["local_read_status"] = "READY"
        return report, head
    except (BranchUnavailable, OSError, ValueError, RuntimeError, WriterError) as error:
        return _blocked(
            "EVENT_AUTHORITY_INVALID",
            str(error),
            remote=args.remote,
            remote_ref=f"refs/heads/{args.event_branch}",
            branch=args.event_branch,
            stream_path=args.event_stream,
        ), None


def _allocation_report(
    args: argparse.Namespace,
    snapshot: StaticSnapshot,
    head: LedgerHead,
) -> dict[str, Any]:
    view = global_allocation_view(snapshot, head)
    if isinstance(view, BlockedResult):
        return _blocked(view.reason_code, view.detail)

    report: dict[str, Any] = _jsonable(view)
    principal_closed = args.principal_closed
    maintenance_closed = args.maintenance_closed
    if principal_closed is not None and maintenance_closed is not None:
        return _blocked(
            "ALLOCATION_INPUT_AMBIGUOUS",
            "provide only one of --principal-closed and --maintenance-closed",
            **report,
        )
    if maintenance_closed is not None and args.maintenance_line is None:
        return _blocked(
            "MAINTENANCE_LINE_REQUIRED",
            "--maintenance-closed requires --maintenance-line",
            **report,
        )
    if args.maintenance_line is not None and maintenance_closed is None:
        return _blocked(
            "MAINTENANCE_CLOSED_VERSION_REQUIRED",
            "--maintenance-line requires --maintenance-closed",
            **report,
        )

    if principal_closed is not None:
        decision = select_principal_sentinel(principal_closed, snapshot, head)
    elif maintenance_closed is not None:
        decision = select_maintenance_version(
            args.maintenance_line,
            maintenance_closed,
            snapshot,
            head,
        )
    else:
        report["allocation_status"] = "NOT_REQUESTED"
        return report

    decision_data = _jsonable(decision)
    report["allocation"] = decision_data
    if decision.status != "AVAILABLE":
        report["status"] = "BLOCKED"
        report["reason_code"] = decision.reason_code
        report["detail"] = decision.detail
    return report


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    base: dict[str, Any] = {
        "command": args.command,
        "dry_run": True,
    }
    static_result, snapshot = _static_report(args)
    if static_result.get("status") == "READY":
        # This is the stable source identity from the validated snapshot, not
        # the local checkout path.
        base["source_repository"] = static_result["source_repository"]
    if args.command == "snapshot":
        base["static_snapshot"] = static_result
        base["status"] = static_result["status"]
        if static_result["status"] == "BLOCKED":
            base.update(
                reason_code=static_result["reason_code"],
                detail=static_result["detail"],
            )
            return base, 2
        return base, 0

    # A full readiness/allocation result cannot use an unverified static
    # authority.  Stop before touching the event branch so the reported
    # reason identifies the first failed gate.
    if args.command in {"readiness", "allocation"} and snapshot is None:
        base["static_snapshot"] = static_result
        base["event_head"] = {
            "status": "NOT_CHECKED",
            "reason_code": static_result["reason_code"],
        }
        base["status"] = "BLOCKED"
        base["reason_code"] = static_result["reason_code"]
        base["detail"] = static_result["detail"]
        return base, 2

    event_result, head = _event_report(args)
    if args.command == "events":
        base["event_head"] = event_result
        base["status"] = event_result["status"]
        if event_result["status"] == "BLOCKED":
            base.update(
                reason_code=event_result["reason_code"],
                detail=event_result["detail"],
            )
            return base, 2
        return base, 0

    base["static_snapshot"] = static_result
    base["event_head"] = event_result
    if head is None:
        base["status"] = "BLOCKED"
        base["reason_code"] = event_result["reason_code"]
        base["detail"] = event_result["detail"]
        return base, 2

    allocation = _allocation_report(args, snapshot, head)
    base["allocation_view"] = allocation
    base["status"] = allocation.get("status", "READY")
    if base["status"] == "BLOCKED":
        base["reason_code"] = allocation.get("reason_code")
        base["detail"] = allocation.get("detail", "")
        return base, 2
    return base, 0


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    # Suppress child-parser defaults so options supplied before the subcommand
    # are retained.  ``main`` installs defaults after parsing.
    absent = argparse.SUPPRESS
    parser.add_argument("--repo", "--repository", default=absent, help="Git checkout to inspect")
    parser.add_argument("--remote", default=absent, help="remote used for static authority")
    parser.add_argument(
        "--source-repository",
        default=absent,
        help="stable source identity in the snapshot",
    )
    parser.add_argument(
        "--selector",
        default=absent,
        help="canonical static source selector",
    )
    parser.add_argument("--event-branch", "--branch", default=absent)
    parser.add_argument("--event-stream", "--stream-path", default=absent)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=absent,
        help="explicitly request the read-only Gate A mode (the only supported mode)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=absent,
        help="accepted for callers that select machine-readable output explicitly",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    _add_common_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=False)
    for name in ("readiness", "snapshot", "events", "allocation"):
        child = subparsers.add_parser(name, help=f"read {name} lifecycle state")
        _add_common_options(child)
        child.add_argument("--principal-closed", metavar="VERSION")
        child.add_argument("--maintenance-line", metavar="LINE")
        child.add_argument("--maintenance-closed", metavar="VERSION")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        defaults = {
            "repo": ".",
            "remote": DEFAULT_REMOTE,
            "source_repository": None,
            "selector": "refs/heads/vmm:iterations.toml",
            "event_branch": DEFAULT_EVENT_BRANCH,
            "event_stream": DEFAULT_EVENT_STREAM,
            "dry_run": True,
            "json": True,
        }
        for name, value in defaults.items():
            if not hasattr(args, name):
                setattr(args, name, value)
        args.remote = validate_remote(args.remote)
        if args.command is None:
            args.command = "readiness"
            # Root-only invocation has no allocation inputs.
            args.principal_closed = None
            args.maintenance_line = None
            args.maintenance_closed = None
        result, exit_code = _run(args)
    except SystemExit:
        raise
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        result = {
            "command": getattr(locals().get("args", None), "command", "readiness"),
            "dry_run": True,
            "status": "BLOCKED",
            "reason_code": "CLI_INPUT_INVALID",
            "detail": str(error),
        }
        exit_code = 1
    redact: list[str] = []
    if "args" in locals():
        raw_repository = str(getattr(args, "repo", ""))
        if raw_repository:
            repository_path = Path(raw_repository)
            if repository_path.is_absolute():
                redact.append(raw_repository)
            redact.append(str(repository_path.resolve()))
        raw_remote = str(getattr(args, "remote", ""))
        if raw_remote.startswith("/"):
            redact.extend((raw_remote, str(Path(raw_remote).resolve())))
    return _write(
        result,
        exit_code=exit_code,
        redact_paths=tuple(sorted(set(redact), key=len, reverse=True)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
