#!/usr/bin/env python3
"""Read-only Gate A lifecycle readiness CLI.

The command resolves the static iteration source and the complete immutable
``refs/heads/lifecycle/v1/*`` namespace in an isolated bare repository. It
never creates lifecycle objects, changes a ref, or performs a production
release. The ``lifecycle`` command reports the complete immutable ref snapshot.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

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
from version_lifecycle.manifests import LifecycleRefSnapshot, lifecycle_ref_snapshot  # noqa: E402
from version_lifecycle.static import StaticSnapshot  # noqa: E402
from version_lifecycle.certification import is_safe_public_value  # noqa: E402
from version_lifecycle.git_refs import validate_remote  # noqa: E402

DEFAULT_REMOTE = "origin"


class CliArgumentError(ValueError):
    """An argparse failure that can be reported through the JSON contract."""


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "canonical"):
        return str(value.canonical)
    if isinstance(value, StaticSnapshot):
        return value.to_dict()
    if isinstance(value, GlobalAllocationView):
        return {
            "status": value.status,
            "lifecycle_snapshot_digest": value.lifecycle_snapshot_digest,
            "static_snapshot_digest": value.snapshot_digest,
            "static_occupied_versions": sorted(value.static_occupied),
            "lifecycle_occupied_versions": sorted(value.lifecycle_occupied),
            "occupied_versions": sorted(value.occupied),
        }
    if isinstance(value, LifecycleRefSnapshot):
        return value.to_dict()
    if isinstance(value, BlockedResult):
        return {"status": value.status, "reason_code": value.reason_code, "detail": value.detail}
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _sanitize_report(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {name: _sanitize_report(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_report(item, key=key) for item in value]
    if isinstance(value, str):
        if key == "remote":
            return "configured"
        if key in {"detail", "local_read_detail"}:
            return "detail omitted; use reason_code"
        if key in {"selector", "remote_ref"} and ("@" in value or "://" in value or not is_safe_public_value(value, key=key)):
            return "<redacted>"
    return value


def _write(result: Mapping[str, Any], *, exit_code: int = 0, redact_paths: tuple[str, ...] = ()) -> int:
    payload = _sanitize_report(_jsonable(result))
    for path in redact_paths:
        payload = _redact_paths(payload, path)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return exit_code


def _redact_paths(value: Any, path: str) -> Any:
    if isinstance(value, str):
        return value.replace(path, "<repository>")
    if isinstance(value, Mapping):
        return {key: _redact_paths(item, path) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item, path) for item in value]
    return value


def _blocked(reason_code: str, detail: str, **fields: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "BLOCKED", "reason_code": reason_code, "detail": detail}
    result.update(fields)
    return result


def _static_report(args: argparse.Namespace) -> tuple[dict[str, Any], StaticSnapshot | None]:
    result = static_snapshot(
        args.authority_repo,
        source_repository=args.source_repository,
        selector=args.selector,
        remote=args.authority_remote,
    )
    if isinstance(result, BlockedResult):
        return _blocked(result.reason_code, result.detail, selector=args.selector, remote=args.remote), None
    assert isinstance(result, StaticSnapshot)
    return {"status": "READY", **result.to_dict()}, result


def _lifecycle_report(args: argparse.Namespace, source_repository: str | None = None) -> tuple[dict[str, Any], LifecycleRefSnapshot | None]:
    try:
        snapshot = lifecycle_ref_snapshot(
            args.authority_repo, remote=args.authority_remote, source_repository=source_repository or args.source_repository,
        )
    except (OSError, ValueError, RuntimeError) as error:
        return _blocked("LIFECYCLE_REF_SNAPSHOT_UNAVAILABLE", str(error), remote=args.remote, namespace="refs/heads/lifecycle/v1/*"), None
    return {
        "status": "READY", "authority": "remote", "remote": args.remote,
        "namespace": "refs/heads/lifecycle/v1/*",
        "lifecycle_snapshot_digest": snapshot.snapshot_digest,
        "lifecycle_ref_set_digest": snapshot.ref_set_digest,
        "occupied_versions": list(snapshot.occupied_versions),
        "ref_bindings": list(snapshot.ref_bindings),
    }, snapshot


def _remote_transport(repository: str | Path, remote: str) -> str:
    root = Path(repository).resolve()
    configured = subprocess.run(
        ["git", "-C", str(root), "config", "--get", f"remote.{remote}.url"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    endpoint = configured.stdout.strip() if configured.returncode == 0 else remote
    if "://" not in endpoint and not re.match(r"^[^/@:]+@[^/:]+:", endpoint):
        candidate = Path(endpoint).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        endpoint = str(candidate.resolve())
    return validate_remote(endpoint)


def _isolated_authority_repository(args: argparse.Namespace, root: Path) -> None:
    subprocess.run(["git", "init", "--bare", "--quiet", str(root)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    endpoint = _remote_transport(args.repo, args.remote)
    subprocess.run(["git", "-C", str(root), "remote", "add", "authority", endpoint], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    args.authority_repo = str(root)
    args.authority_remote = "authority"


def _allocation_report(args: argparse.Namespace, snapshot: StaticSnapshot, lifecycle: LifecycleRefSnapshot) -> dict[str, Any]:
    view = global_allocation_view(snapshot, lifecycle)
    if isinstance(view, BlockedResult):
        return _blocked(view.reason_code, view.detail)
    report: dict[str, Any] = _jsonable(view)
    if args.principal_closed is not None and args.maintenance_closed is not None:
        return _blocked("ALLOCATION_INPUT_AMBIGUOUS", "provide only one allocation input", **report)
    if args.maintenance_closed is not None and args.maintenance_line is None:
        return _blocked("MAINTENANCE_LINE_REQUIRED", "--maintenance-closed requires --maintenance-line", **report)
    if args.maintenance_line is not None and args.maintenance_closed is None:
        return _blocked("MAINTENANCE_CLOSED_VERSION_REQUIRED", "--maintenance-line requires --maintenance-closed", **report)
    if args.principal_closed is not None:
        decision = select_principal_sentinel(args.principal_closed, snapshot, lifecycle)
    elif args.maintenance_closed is not None:
        decision = select_maintenance_version(args.maintenance_line, args.maintenance_closed, snapshot, lifecycle)
    else:
        report["allocation_status"] = "NOT_REQUESTED"
        return report
    report["allocation"] = _jsonable(decision)
    if decision.status != "AVAILABLE":
        report.update(status="BLOCKED", reason_code=decision.reason_code, detail=decision.detail)
    return report


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    base: dict[str, Any] = {"command": args.command, "dry_run": True}
    static_result, snapshot = _static_report(args)
    if static_result.get("status") == "READY":
        base["source_repository"] = static_result["source_repository"]
    if args.command == "snapshot":
        base["static_snapshot"] = static_result
        if static_result["status"] == "BLOCKED":
            base.update(status="BLOCKED", reason_code=static_result["reason_code"], detail=static_result["detail"])
            return base, 2
        base["status"] = "READY"
        return base, 0
    if snapshot is None:
        base["static_snapshot"] = static_result
        base["lifecycle_snapshot"] = {"status": "NOT_CHECKED", "reason_code": static_result.get("reason_code")}
        base.update(status="BLOCKED", reason_code=static_result.get("reason_code"), detail=static_result.get("detail", ""))
        return base, 2
    lifecycle_result, lifecycle = _lifecycle_report(args, static_result.get("source_repository"))
    base["static_snapshot"] = static_result
    base["lifecycle_snapshot"] = lifecycle_result
    if args.command == "lifecycle":
        base["status"] = lifecycle_result["status"]
        if lifecycle_result["status"] == "BLOCKED":
            base.update(reason_code=lifecycle_result["reason_code"], detail=lifecycle_result["detail"])
            return base, 2
        return base, 0
    if lifecycle is None:
        base.update(status="BLOCKED", reason_code=lifecycle_result["reason_code"], detail=lifecycle_result["detail"])
        return base, 2
    allocation = _allocation_report(args, snapshot, lifecycle)
    base["allocation_view"] = allocation
    base["status"] = allocation.get("status", "READY")
    if base["status"] == "BLOCKED":
        base.update(reason_code=allocation.get("reason_code"), detail=allocation.get("detail", ""))
        return base, 2
    return base, 0


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    absent = argparse.SUPPRESS
    parser.add_argument("--repo", "--repository", default=absent)
    parser.add_argument("--remote", default=absent)
    parser.add_argument("--source-repository", default=absent)
    parser.add_argument("--selector", default=absent)
    parser.add_argument("--dry-run", action="store_true", default=absent)
    parser.add_argument("--json", action="store_true", default=absent)


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(description=__doc__)
    _add_common_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=False)
    for name in ("readiness", "snapshot", "lifecycle", "allocation"):
        child = subparsers.add_parser(name, help=f"read {name} lifecycle state")
        _add_common_options(child)
        child.add_argument("--principal-closed")
        child.add_argument("--maintenance-line")
        child.add_argument("--maintenance-closed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        defaults = {"repo": ".", "remote": DEFAULT_REMOTE, "source_repository": None, "selector": "refs/heads/vmm:iterations.toml", "dry_run": True, "json": True}
        for name, value in defaults.items():
            if not hasattr(args, name):
                setattr(args, name, value)
        args.remote = validate_remote(args.remote)
        if args.command is None:
            args.command = "readiness"
            args.principal_closed = args.maintenance_line = args.maintenance_closed = None
        with tempfile.TemporaryDirectory(prefix="cyax-gate-a-readonly-") as temporary:
            _isolated_authority_repository(args, Path(temporary) / "authority.git")
            result, exit_code = _run(args)
    except SystemExit:
        raise
    except CliArgumentError as error:
        result = {
            "command": "readiness",
            "dry_run": True,
            "status": "BLOCKED",
            "reason_code": "CLI_ARGUMENT_INVALID",
            "detail": str(error),
        }
        exit_code = 2
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        result = {"command": "readiness", "dry_run": True, "status": "BLOCKED", "reason_code": "CLI_INPUT_INVALID", "detail": str(error)}
        exit_code = 1
    redact: list[str] = []
    raw_repository = str(getattr(locals().get("args", None), "repo", ""))
    if raw_repository:
        redact.extend((raw_repository, str(Path(raw_repository).resolve())))
    return _write(result, exit_code=exit_code, redact_paths=tuple(sorted(set(redact), key=len, reverse=True)))


if __name__ == "__main__":
    raise SystemExit(main())
