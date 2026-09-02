#!/usr/bin/env python3
"""Canonical status projection for the bounded h11=4 replay comparison.

This module docstring is the authoritative human-readable contract for the
verification gate.  Keep the projection fields, ordering, encoding, newline
policy, and hash recipe below stable.

The immutable replay artifact is a complete bounded exact-replay output, not a
new population input.  This module compares that output with one replay made
after the zero-dimensional certificate repair.  It deliberately projects only
the stable row identity and the terminal status.  Geometry evidence, timing,
and enumeration order are not part of this compatibility gate.

Projection recipe
-----------------
Select every ``record_type == "row"`` record from the source artifact whose
``terminal_status`` is not ``smoothness_verification_unavailable``.  This is
the handoff's 1,088-row unaffected set; the 58 rows with that baseline status
are the repaired set and are excluded.  Require the repaired artifact to have
the same complete row-identity set, then project the selected identities from
both artifacts.

Sort records by ``row_identity`` using Python's Unicode code-point order.  For
each record, encode exactly this object as UTF-8::

    {"row_identity": "...", "terminal_status": "..."}\n

The object uses ``json.dumps(..., sort_keys=True, separators=(",", ":"),
ensure_ascii=True, allow_nan=False)``.  The LF byte is included after every
record, including the final record.  Hash the resulting bytes with SHA-256.
The report binds both compressed artifact SHA-256 fingerprints, each replay
header's ``config_sha256`` and ``config.source_code_sha256``, and the repaired
implementation file fingerprint when supplied.

Each artifact must contain exactly one header and exactly one summary.  The
summary must have ``status == "completed"``, ``rows_evaluated`` equal to the
number of streamed row records, and zero ``database_writes`` and
``duplicate_count``.  zstd output is consumed line by line; the final report
contains bounded metadata and digests only.

The module has no CYTools, NumPy, Julia, or database dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PROJECTION_SCHEMA_VERSION = "cyaxiverse-orientifold-unaffected-projection-1.0"
EXPECTED_H11 = 4
EXPECTED_SOURCE_ROWS = 1146
EXPECTED_AFFECTED_ROWS = 58
EXPECTED_UNAFFECTED_ROWS = 1088
AFFECTED_SOURCE_STATUSES = frozenset({"smoothness_verification_unavailable"})
IMMUTABLE_SOURCE_ARTIFACT_SHA256 = (
    "b7cb293ea369d52fa22aa01db6b487303b40279f87e40d14a1509bb1a062dfa3"
)


class ProjectionError(ValueError):
    """Raise when a replay artifact cannot satisfy the projection contract."""


def canonical_projection_record(row: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact two-field projection for one replay row."""

    identity = row.get("row_identity")
    status = row.get("terminal_status")
    if not isinstance(identity, str) or not identity:
        raise ProjectionError("projection row has no non-empty string row_identity")
    if not isinstance(status, str) or not status:
        raise ProjectionError(f"projection row {identity!r} has no terminal_status")
    return {"row_identity": identity, "terminal_status": status}


def _rows_by_identity(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        projected = canonical_projection_record(row)
        identity = projected["row_identity"]
        if identity in indexed:
            raise ProjectionError(f"{label} artifact has duplicate row_identity {identity}")
        indexed[identity] = projected
    return indexed


def canonical_projection_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize rows using the canonical sorted compact JSONL recipe."""

    indexed = _rows_by_identity(rows, label="projection")
    chunks = []
    for identity in sorted(indexed):
        encoded = json.dumps(
            indexed[identity],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        chunks.append(encoded + b"\n")
    return b"".join(chunks)


def canonical_projection_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash the canonical sorted compact JSONL projection with SHA-256."""

    return hashlib.sha256(canonical_projection_bytes(rows)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_zstd_jsonl(
    path: Path, on_record: Callable[[dict[str, Any]], None]
) -> None:
    """Stream and validate one zstd-compressed JSONL artifact.

    Consume stdout one line at a time and check zstd's exit status after EOF.
    The latter is required because a truncated stream can still produce valid
    JSONL lines before the decompressor reports corruption.
    """

    zstd = shutil.which("zstd")
    if zstd is None:
        raise ProjectionError("zstd is required to read replay artifacts")
    try:
        process = subprocess.Popen(
            [zstd, "-dcq", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ProjectionError(f"cannot start zstd for replay artifact {path}") from exc
    try:
        assert process.stdout is not None
        for line_number, line in enumerate(iter(process.stdout.readline, b""), 1):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProjectionError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ProjectionError(
                    f"replay artifact line {line_number} is not an object"
                )
            on_record(value)
        assert process.stderr is not None
        stderr = process.stderr.read()
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise ProjectionError(f"cannot read replay artifact {path}: {detail}")


def _peak_rss_bytes() -> int:
    """Return the process peak RSS in bytes for bounded resource reporting."""

    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    return peak_rss


def _artifact_payload(path: Path, *, label: str, expected_sha256: str | None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ProjectionError(f"{label} replay artifact is missing: {path}")
    artifact_sha256 = _sha256_file(path)
    if expected_sha256 is not None and artifact_sha256 != expected_sha256:
        raise ProjectionError(
            f"{label} replay artifact SHA-256 mismatch: expected {expected_sha256}, "
            f"got {artifact_sha256}"
        )
    header_count = 0
    config_sha256: Any = None
    source_code_sha256: Any = None
    row_count = 0
    rows_by_identity: dict[str, dict[str, str]] = {}
    summary_count = 0
    summary_metadata: dict[str, Any] | None = None

    def consume(record: dict[str, Any]) -> None:
        nonlocal header_count, config_sha256, source_code_sha256
        nonlocal row_count, summary_count, summary_metadata
        record_type = record.get("record_type")
        if record_type == "header":
            header_count += 1
            if header_count > 1:
                raise ProjectionError(f"{label} artifact must contain exactly one header")
            config = record.get("config")
            if not isinstance(config, dict):
                raise ProjectionError(f"{label} artifact header has no config object")
            if config.get("requested_h11") != EXPECTED_H11:
                raise ProjectionError(f"{label} artifact requested_h11 is not 4")
            for name, expected in (
                ("max_rows", EXPECTED_SOURCE_ROWS),
                ("workers", 1),
                ("shard_count", 1),
                ("shard_index", 0),
            ):
                if config.get(name) != expected:
                    raise ProjectionError(
                        f"{label} artifact config {name} is not {expected}"
                    )
            config_sha256 = record.get("config_sha256")
            source_code_sha256 = config.get("source_code_sha256")
        elif record_type == "row":
            row_count += 1
            projected = canonical_projection_record(record)
            identity = projected["row_identity"]
            if identity in rows_by_identity:
                raise ProjectionError(f"{label} artifact has duplicate row_identity {identity}")
            rows_by_identity[identity] = projected
        elif record_type == "summary":
            summary_count += 1
            if summary_count > 1:
                raise ProjectionError(f"{label} artifact must contain exactly one summary")
            # Retain only bounded summary metadata, not the complete summary.
            summary_metadata = {
                name: record.get(name)
                for name in (
                    "status",
                    "rows_evaluated",
                    "database_writes",
                    "duplicate_count",
                )
            }

    _read_zstd_jsonl(path, consume)
    if header_count != 1:
        raise ProjectionError(f"{label} artifact must contain exactly one header")
    if summary_count != 1 or summary_metadata is None:
        raise ProjectionError(f"{label} artifact must contain exactly one summary")
    if summary_metadata.get("status") != "completed":
        raise ProjectionError(f"{label} artifact summary status is not completed")
    if (
        type(summary_metadata.get("rows_evaluated")) is not int
        or summary_metadata["rows_evaluated"] != row_count
    ):
        raise ProjectionError(f"{label} artifact summary row count does not match rows")
    if (
        type(summary_metadata.get("database_writes")) is not int
        or summary_metadata["database_writes"] != 0
    ):
        raise ProjectionError(f"{label} artifact summary database_writes is not zero")
    if (
        type(summary_metadata.get("duplicate_count")) is not int
        or summary_metadata["duplicate_count"] != 0
    ):
        raise ProjectionError(f"{label} artifact summary duplicate_count is not zero")
    if row_count != EXPECTED_SOURCE_ROWS:
        raise ProjectionError(
            f"{label} artifact has {row_count} rows; expected {EXPECTED_SOURCE_ROWS}"
        )
    return {
        "path": str(path),
        "sha256": artifact_sha256,
        "size_bytes": path.stat().st_size,
        "config_sha256": config_sha256,
        "source_code_sha256": source_code_sha256,
        "requested_h11": EXPECTED_H11,
        "rows_by_identity": rows_by_identity,
    }


def _implementation_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ProjectionError(f"repaired implementation file is missing: {path}")
    return {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def compare_replay_artifacts(
    source_path: Path,
    repaired_path: Path,
    *,
    implementation_path: Path | None = None,
    expected_source_sha256: str | None = IMMUTABLE_SOURCE_ARTIFACT_SHA256,
) -> dict[str, Any]:
    """Compare the unaffected status projection of two bounded h11=4 outputs."""

    source = _artifact_payload(
        source_path, label="source", expected_sha256=expected_source_sha256
    )
    repaired = _artifact_payload(repaired_path, label="repaired", expected_sha256=None)
    source_by_identity = source.pop("rows_by_identity")
    repaired_by_identity = repaired.pop("rows_by_identity")
    source_ids = set(source_by_identity)
    repaired_ids = set(repaired_by_identity)
    if source_ids != repaired_ids:
        missing = sorted(source_ids - repaired_ids)
        extra = sorted(repaired_ids - source_ids)
        raise ProjectionError(
            f"source/repaired row-identity sets differ (missing={missing[:3]}, extra={extra[:3]})"
        )
    affected = {
        identity
        for identity, row in source_by_identity.items()
        if row["terminal_status"] in AFFECTED_SOURCE_STATUSES
    }
    unaffected = [
        repaired_by_identity[identity]
        for identity in sorted(source_ids - affected)
    ]
    source_unaffected = [source_by_identity[identity] for identity in sorted(source_ids - affected)]
    if len(affected) != EXPECTED_AFFECTED_ROWS:
        raise ProjectionError(
            f"source affected-row count is {len(affected)}; expected {EXPECTED_AFFECTED_ROWS}"
        )
    if len(unaffected) != EXPECTED_UNAFFECTED_ROWS:
        raise ProjectionError(
            f"source unaffected-row count is {len(unaffected)}; expected {EXPECTED_UNAFFECTED_ROWS}"
        )
    source_digest = canonical_projection_sha256(source_unaffected)
    repaired_digest = canonical_projection_sha256(unaffected)
    report = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "contract": {
            "h11": EXPECTED_H11,
            "source_rows": EXPECTED_SOURCE_ROWS,
            "affected_rows": EXPECTED_AFFECTED_ROWS,
            "unaffected_rows": EXPECTED_UNAFFECTED_ROWS,
            "affected_source_statuses": sorted(AFFECTED_SOURCE_STATUSES),
            "inclusion_rule": "include source row identities whose baseline terminal_status is not in affected_source_statuses",
            "projected_fields": ["row_identity", "terminal_status"],
            "ordering": "ascending row_identity using Python Unicode code-point order",
            "json_encoding": "UTF-8; json.dumps(sort_keys=True, separators=(\",\", \":\"), ensure_ascii=True, allow_nan=False)",
            "newline": "LF after every JSON object, including the final object",
            "hash": "SHA-256 of serialized projection bytes",
            "streaming": "zstd stdout is consumed line-by-line; only row_identity and terminal_status are retained for comparison",
        },
        "source_artifact": source,
        "repaired_artifact": repaired,
        "repaired_implementation": _implementation_fingerprint(implementation_path),
        "projection": {
            "source_sha256": source_digest,
            "repaired_sha256": repaired_digest,
            "status_projection_matches": source_digest == repaired_digest,
            "row_identity_projection_matches": True,
        },
        "resource": {
            "peak_rss_bytes": _peak_rss_bytes(),
            "measurement": "process getrusage(RUSAGE_SELF).ru_maxrss after streaming both artifacts",
        },
    }
    return report


def _write_zstd_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    private_tmp = Path("/private/tmp").resolve()
    try:
        path.relative_to(private_tmp)
    except ValueError as exc:
        raise ProjectionError("verification output must be written under /private/tmp") from exc
    if path.exists() or path.is_symlink():
        raise ProjectionError(f"refusing to overwrite verification output: {path}")
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ProjectionError("zstd is required to write verification output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=str(private_tmp), delete=False
    ) as stream:
        raw_path = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    compressed_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        completed = subprocess.run(
            [zstd, "-19", "-q", "-f", "-o", str(compressed_path), str(raw_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ProjectionError(f"zstd compression failed: {completed.stderr.strip()}")
        os.replace(compressed_path, path)
    finally:
        raw_path.unlink(missing_ok=True)
        compressed_path.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repaired", required=True, type=Path)
    parser.add_argument("--implementation", type=Path)
    parser.add_argument("--output", type=Path, help="Compressed JSON report under /private/tmp.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    try:
        report = compare_replay_artifacts(
            args.source,
            args.repaired,
            implementation_path=args.implementation,
            expected_source_sha256=IMMUTABLE_SOURCE_ARTIFACT_SHA256,
        )
        if not report["projection"]["status_projection_matches"]:
            raise ProjectionError(
                "status projection mismatch: canonical SHA-256 digests differ"
            )
        if args.output is not None:
            _write_zstd_json(args.output, report)
        print(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True))
    except ProjectionError as exc:
        raise SystemExit(f"projection validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
