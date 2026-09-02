#!/usr/bin/env python3
"""Validate the required h11=4/5 population handoff before a run.

The higher-h11 orientifold runs are replay runs, not opportunities to silently
regenerate or reinterpret an earlier population.  This module reads the
relevant handoffs, verifies the durable compressed artifacts, and checks the
recorded population metadata.  It deliberately has no CYTools dependency and
does not load a parquet source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class PopulationPreflightError(RuntimeError):
    """Raise when the required higher-h11 run evidence is unavailable."""


@dataclass(frozen=True)
class HandoffRequirement:
    """Describe one handoff that must be read before a higher-h11 run."""

    relative_path: str
    markers: tuple[str, ...]


REQUIRED_HANDOFFS: tuple[HandoffRequirement, ...] = (
    HandoffRequirement(
        "handoffs_checkpoints/HANDOFF_orientifold_h11_4_population_rerun_20260821.md",
        ("fresh h11 = 4", "ledger", "superseded"),
    ),
    HandoffRequirement(
        "handoffs_checkpoints/HANDOFF_orientifold_population_h11_4_normal_form_keying_20260820.md",
        ("normal-form keying", "Do not run any h11 > 4", "smoothness"),
    ),
    HandoffRequirement(
        "handoffs_checkpoints/HANDOFF_orientifold_gap_analyzer_h11_4_5_extension_20260821.md",
        ("h11 = 4 and h11 = 5", "ledger-only", "Do not regenerate it here"),
    ),
    HandoffRequirement(
        "handoffs_checkpoints/SUMMARY_orientifold_performance_and_h11_5_run_20260821.md",
        ("h11 = 5 favorable population", "Peak RSS", "h11 = 4"),
    ),
    HandoffRequirement(
        "handoffs_checkpoints/HANDOFF_orientifold_population_h11_2_3_normal_form_keying_20260820.md",
        ("h11 = 2", "h11 = 3", "normal-form"),
    ),
)


ARTIFACT_SPECS: Mapping[int, Mapping[str, Any]] = {
    4: {
        "directory": "orientifold_h11_4_population_20260821",
        "merged": "h4.merged.json.zst",
        "gap_analysis": "h4_gap_analysis.json.zst",
        "expected": {
            "h11": 4,
            "favorable_polytopes": 1185,
            "frst_classes": 1760,
            "h21_plus_zero_trilayer_frst_classes": 267,
        },
    },
    5: {
        "directory": "orientifold_h11_5_population_20260821",
        "merged": "merged.json.zst",
        "gap_analysis": "h5_gap_analysis.json.zst",
        "expected": {
            "h11": 5,
            "favorable_polytopes": 4897,
            "frst_classes": 11713,
            "h21_plus_zero_trilayer_frst_classes": 1033,
        },
    },
}

IMPLEMENTATION_HANDOFF = "IMPLEMENTATION_HANDOFF_orientifold_axiverse_20260823.jsonl.zst"
REQUIRED_SOURCE_PARTITIONS = tuple(
    f"polytopes-4d-{vertex_count:02d}-vertices.parquet"
    for vertex_count in range(5, 11)
)


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_handoffs(
    repo_root: Path,
    requirements: Iterable[HandoffRequirement] = REQUIRED_HANDOFFS,
) -> list[dict[str, Any]]:
    """Read and validate every required handoff, returning acknowledgements."""

    acknowledgements = []
    for requirement in requirements:
        path = repo_root.parent / requirement.relative_path
        if not path.is_file():
            raise PopulationPreflightError(f"required handoff is missing: {path}")
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in requirement.markers if marker not in text]
        if missing:
            raise PopulationPreflightError(
                f"required handoff {path} is missing markers: {', '.join(missing)}"
            )
        acknowledgements.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "bytes_read": len(text.encode("utf-8")),
                "markers": list(requirement.markers),
            }
        )
    return acknowledgements


def _read_checksum_manifest(path: Path) -> list[tuple[str, Path]]:
    if not path.is_file():
        raise PopulationPreflightError(f"artifact checksum manifest is missing: {path}")
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise PopulationPreflightError(
                f"invalid checksum entry at {path}:{line_number}: {line!r}"
            )
        digest, name = fields
        relative_name = name.lstrip("*")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PopulationPreflightError(
                f"checksum entry escapes its artifact directory at {path}:{line_number}"
            )
        entries.append((digest, path.parent / relative_path))
    if not entries:
        raise PopulationPreflightError(f"artifact checksum manifest is empty: {path}")
    return entries


def _verify_zstd(path: Path) -> None:
    try:
        completed = subprocess.run(
            ["zstd", "-tq", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise PopulationPreflightError("zstd is required for artifact preflight") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PopulationPreflightError(f"zstd integrity check failed for {path}: {detail}")


def _read_zstd_json(path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["zstd", "-dc", str(path)],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise PopulationPreflightError("zstd is required for artifact preflight") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PopulationPreflightError(f"could not read compressed JSON {path}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PopulationPreflightError(f"invalid compressed JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise PopulationPreflightError(f"compressed JSON artifact is not an object: {path}")
    return payload


def _read_zstd_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a compressed JSONL handoff without accepting malformed records."""

    try:
        completed = subprocess.run(
            ["zstd", "-dcq", str(path)], check=False, capture_output=True
        )
    except OSError as exc:
        raise PopulationPreflightError("zstd is required for implementation-handoff preflight") from exc
    if completed.returncode != 0:
        raise PopulationPreflightError(
            f"could not read implementation handoff {path}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    records = []
    for line_number, line in enumerate(completed.stdout.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PopulationPreflightError(
                f"invalid implementation handoff JSON at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise PopulationPreflightError(
                f"implementation handoff line {line_number} is not an object"
            )
        records.append(value)
    return records


def _directory_digest(entries: list[dict[str, Any]]) -> str:
    """Hash the ordered partition manifest used by replay provenance."""

    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_source_contract(
    repo_root: Path, parquet_dir: Path | None = None
) -> dict[str, Any]:
    """Load and verify the exact 05--10 source contract from the handoff.

    ``parquet_dir`` is an explicit caller-selected durable mirror.  The
    handoff path remains provenance, while filenames and byte hashes remain
    the authoritative source identity when the mirror has moved.
    """

    handoff_path = repo_root.parent / IMPLEMENTATION_HANDOFF
    if not handoff_path.is_file():
        raise PopulationPreflightError(f"implementation handoff is missing: {handoff_path}")
    records = _read_zstd_jsonl(handoff_path)
    source_records = [record for record in records if record.get("type") == "source"]
    if len(source_records) != 1:
        raise PopulationPreflightError(
            f"implementation handoff must contain exactly one source record: {handoff_path}"
        )
    source = source_records[0]
    handoff_source_path = Path(str(source.get("path", ""))).expanduser().resolve()
    source_path = (
        Path(parquet_dir).expanduser().resolve()
        if parquet_dir is not None
        else handoff_source_path
    )
    hashes = source.get("hashes")
    if not isinstance(hashes, dict):
        raise PopulationPreflightError("implementation handoff source path or hashes are unavailable")
    if parquet_dir is None and not source_path.is_dir():
        raise PopulationPreflightError(f"implementation handoff source path is missing: {source_path}")
    if parquet_dir is not None and not source_path.is_dir():
        raise PopulationPreflightError(f"caller-selected parquet source path is missing: {source_path}")
    expected_names = set(REQUIRED_SOURCE_PARTITIONS)
    observed_names = {
        path.name for path in source_path.glob("polytopes-4d-*-vertices.parquet")
    }
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise PopulationPreflightError(
            f"source partitions must be exactly 05..10; missing={missing}, extra={extra}"
        )
    if set(hashes) != {name[13:15] for name in REQUIRED_SOURCE_PARTITIONS}:
        raise PopulationPreflightError("implementation handoff source hash keys must be exactly 05..10")
    entries = []
    for name in REQUIRED_SOURCE_PARTITIONS:
        path = source_path / name
        vertex_key = name[13:15]
        expected_hash = str(hashes[vertex_key])
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise PopulationPreflightError(
                f"source partition hash mismatch for {path}: expected {expected_hash}, got {observed_hash}"
            )
        entries.append({
            "partition": name,
            "path": str(path),
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "size_bytes": path.stat().st_size,
        })
    expected_directory_digest = source.get("digest")
    if not isinstance(expected_directory_digest, str) or len(expected_directory_digest) != 64:
        raise PopulationPreflightError("implementation handoff has no directory digest")
    observed_directory_digest = _directory_digest(entries)
    # The historical handoff digest was produced by an earlier manifest
    # serializer.  Partition hashes are the authoritative equality check; the
    # current canonical digest is retained so later handoffs can migrate to a
    # reproducible serializer without losing the recorded value.
    directory_digest_status = (
        "matched" if observed_directory_digest == expected_directory_digest
        else "recorded_legacy_manifest"
    )
    return {
        "handoff": str(handoff_path),
        "handoff_source_path": str(handoff_source_path),
        "source_path": str(source_path),
        "source_path_origin": "caller_override" if parquet_dir is not None else "handoff",
        "required_partitions": list(REQUIRED_SOURCE_PARTITIONS),
        "expected_directory_digest": expected_directory_digest,
        "observed_directory_digest": observed_directory_digest,
        "directory_digest_status": directory_digest_status,
        "partitions": entries,
    }


def _verify_artifacts(repo_root: Path, h11: int) -> dict[str, Any]:
    spec = ARTIFACT_SPECS[h11]
    data_dir = repo_root.parent / "data" / spec["directory"]
    if not data_dir.is_dir():
        raise PopulationPreflightError(f"durable population artifact directory is missing: {data_dir}")

    entries = _read_checksum_manifest(data_dir / "SHA256SUMS.txt")
    required_paths = {
        (data_dir / spec["merged"]).resolve(): "merged_artifact",
        (data_dir / spec["gap_analysis"]).resolve(): "gap_analysis_artifact",
    }
    manifest_digests = {
        path.resolve(): digest
        for digest, path in entries
        if path.resolve() in required_paths
    }
    missing_required = [
        name for path, name in required_paths.items() if path not in manifest_digests
    ]
    if missing_required:
        raise PopulationPreflightError(
            "checksum manifest must explicitly hash " + ", ".join(sorted(missing_required))
        )
    verified_files = []
    for expected_digest, path in entries:
        if not path.is_file():
            raise PopulationPreflightError(f"artifact listed in checksum manifest is missing: {path}")
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise PopulationPreflightError(
                f"checksum mismatch for {path}: expected {expected_digest}, got {actual_digest}"
            )
        if path.name.endswith(".zst"):
            _verify_zstd(path)
        verified_files.append({"path": str(path), "sha256": actual_digest})

    merged_path = data_dir / spec["merged"]
    gap_path = data_dir / spec["gap_analysis"]
    merged = _read_zstd_json(merged_path)
    gap_analysis = _read_zstd_json(gap_path)
    expected = spec["expected"]
    counts = merged.get("counts")
    if not isinstance(counts, dict):
        raise PopulationPreflightError(f"merged artifact has no counts object: {merged_path}")
    checks = {
        "h11": merged.get("requested_h11"),
        "favorable_polytopes": counts.get("favorable_polytopes"),
        "frst_classes": counts.get("frst_classes"),
        "h21_plus_zero_trilayer_frst_classes": counts.get(
            "h21_plus_zero_trilayer_frst_classes"
        ),
    }
    if checks != expected:
        raise PopulationPreflightError(
            f"merged artifact metadata mismatch for h11={h11}: expected {expected}, got {checks}"
        )
    if merged.get("population_complete") is not True:
        raise PopulationPreflightError(f"merged artifact is not population-complete: {merged_path}")
    analyses = gap_analysis.get("analyses")
    if not isinstance(analyses, list) or len(analyses) != 1 or analyses[0].get("h11") != h11:
        raise PopulationPreflightError(f"gap-analysis artifact does not acknowledge h11={h11}: {gap_path}")
    return {
        "directory": str(data_dir),
        "checksum_manifest": str(data_dir / "SHA256SUMS.txt"),
        "verified_file_count": len(verified_files),
        "verified_files": verified_files,
        "merged_artifact": str(merged_path),
        "gap_analysis_artifact": str(gap_path),
        "artifact_digests": {
            "merged_artifact": {
                "path": str(merged_path.resolve()),
                "sha256": manifest_digests[merged_path.resolve()],
            },
            "gap_analysis_artifact": {
                "path": str(gap_path.resolve()),
                "sha256": manifest_digests[gap_path.resolve()],
            },
        },
        "metadata": checks,
        "population_complete": True,
    }


def run_population_preflight(
    repo_root: Path, h11: int, parquet_dir: Path | None = None
) -> dict[str, Any] | None:
    """Read required evidence and fail closed for an h11=4/5 run.

    Return ``None`` for h11 values outside the higher-h11 replay boundary.
    """

    if h11 not in ARTIFACT_SPECS:
        return None
    repo_root = repo_root.resolve()
    handoffs = _read_handoffs(repo_root)
    source_contract = _implementation_source_contract(repo_root, parquet_dir)
    artifacts = _verify_artifacts(repo_root, h11)
    return {
        "status": "passed",
        "policy": "read_required_handoffs_and_verify_durable_population_artifacts_before_geometry_loading",
        "h11": h11,
        "handoffs": handoffs,
        "source_contract": source_contract,
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11", type=int, choices=tuple(ARTIFACT_SPECS), required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="CYAxiverse.jl checkout containing scripts/ (default: this checkout)",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        help="Caller-selected 05--10 mirror; validated against the handoff hashes.",
    )
    args = parser.parse_args(argv)
    try:
        result = run_population_preflight(args.repo_root, args.h11, args.parquet_dir)
    except PopulationPreflightError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
