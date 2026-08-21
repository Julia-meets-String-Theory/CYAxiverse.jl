#!/usr/bin/env python3
"""Upgrade an existing frozen Stage-1 raw-FRST population with topology caches.

The migration preserves the input FRST identities.  It reconstructs each
persisted polytope/triangulation, extracts the Stage-2-independent topology,
and writes a new raw-FRST artifact using the current HDF5 interchange writer.
The input root is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

# CYTools writes an inequality cache at import time/exit.  Keep that cache
# outside the repository and user cache so a migration is self-contained.
CYTOOLS_CACHE = os.environ.get(
    "CYAX_TOPOLOGY_CACHE",
    os.path.join(tempfile.gettempdir(), "cyax-topology-cache-migration"),
)
os.makedirs(CYTOOLS_CACHE, exist_ok=True)
try:
    import platformdirs

    platformdirs.user_cache_dir = lambda *args, **kwargs: CYTOOLS_CACHE
except ImportError:  # pragma: no cover - CYTools normally depends on platformdirs
    pass

import numpy as np
from cytools import Polytope

import generate_geometric_data_multitriangulation as generator
from glimmers_raw_frst import (
    RAW_FRST_SCHEMA_VERSION,
    TOPOLOGY_CACHE_CONVENTIONS,
    TOPOLOGY_CACHE_SCHEMA_VERSION,
    build_raw_frst_geometry_id,
    compute_polytope_id,
    compute_triangulation_hash,
    file_sha256,
    read_raw_frst_artifact,
    write_raw_frst_artifact,
)
from glimmers_schema11 import append_jsonl_record, atomic_json_dump, atomic_jsonl_dump


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Migrate a frozen raw-FRST population to the cached HDF5 format."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--backend",
        choices=("cgal", "qhull"),
        default=None,
        help="Override the backend recorded in each raw artifact.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially written output root after checking completed artifacts.",
    )
    return parser.parse_args(argv)


def git_commit():
    repository_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def rewrite_paths(value, source_root, output_root):
    """Rewrite provenance paths recursively when copying JSON manifests."""
    source_text = str(source_root)
    output_text = str(output_root)
    if isinstance(value, str):
        return (
            output_text + value[len(source_text) :]
            if value == source_text or value.startswith(source_text + os.sep)
            else value
        )
    if isinstance(value, list):
        return [rewrite_paths(item, source_root, output_root) for item in value]
    if isinstance(value, dict):
        return {
            key: rewrite_paths(item, source_root, output_root)
            for key, item in value.items()
        }
    return value


def raw_paths(root):
    return sorted(
        Path(path)
        for path in (root / "frst_candidates").glob("h11_*/np_*/frst_*.h5")
    )


def output_path_for(source_path, source_root, output_root):
    return output_root / source_path.relative_to(source_root)


def backend_for(metadata, override):
    if override is not None:
        return override
    backend = (metadata.get("sampling_metadata") or {}).get("backend")
    if backend not in {"cgal", "qhull"}:
        raise RuntimeError(
            f"raw artifact {metadata.get('raw_frst_path')} has no usable triangulation backend"
        )
    return backend


def reconstruct_and_cache(source_path, destination_path, backend_override):
    started = time.monotonic()
    persisted = read_raw_frst_artifact(source_path)
    arrays = persisted["arrays"]
    backend = backend_for(persisted, backend_override)
    vertices = np.asarray(arrays["polytope_vertices"], dtype=int)
    labels = np.asarray(arrays["triangulation_labels"], dtype=int)
    simplices = np.asarray(arrays["simplices"], dtype=int)

    polytope = Polytope(vertices, deterministic_glsm_basis=True)
    triangulation = polytope.triangulate(
        points=labels.tolist(),
        simplices=simplices.tolist(),
        make_star=False,
        check_input_simplices=True,
        backend=backend,
        verbosity=0,
    )
    reconstructed_points = np.asarray(polytope.points(), dtype=int)
    reconstructed_simplices = np.asarray(triangulation.simplices(), dtype=int)
    polytope_id = compute_polytope_id(reconstructed_points)
    full_hash = compute_triangulation_hash(reconstructed_simplices)
    if polytope_id != str(persisted["polytope_id"]):
        raise RuntimeError(
            "polytope identity changed during reconstruction: "
            f"{persisted['polytope_id']} != {polytope_id}"
        )
    if full_hash != str(persisted["full_triangulation_hash"]):
        raise RuntimeError(
            "triangulation identity changed during reconstruction: "
            f"{persisted['full_triangulation_hash']} != {full_hash}"
        )

    calabi_yau = triangulation.get_cy()
    topology = generator.extract_topology(
        calabi_yau, triangulation, export_kahler_rays=False
    )
    h11 = int(persisted["h11"])
    if int(topology["h11"]) != h11:
        raise RuntimeError(
            f"topology h11={topology['h11']} disagrees with raw h11={h11}"
        )
    geometry_id = build_raw_frst_geometry_id(h11, polytope_id, full_hash)
    cache_metadata = {
        "schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
        "h11": int(topology["h11"]),
        "h21": int(topology["h21"]),
        "geometry_id": geometry_id,
        "raw_geometry_id": geometry_id,
        "polytope_id": polytope_id,
        "full_triangulation_hash": full_hash,
        "cytools_version": getattr(generator.cytools, "version", None),
        "backend": backend,
        "conventions": TOPOLOGY_CACHE_CONVENTIONS,
        "kahler_rays_exported": False,
        "migration": "legacy raw-FRST topology-cache upgrade",
    }
    metadata = {
        key: value
        for key, value in persisted.items()
        if key not in {"arrays", "topology_cache", "raw_frst_file_sha256", "raw_frst_path"}
    }
    metadata.update(
        {
            "topology_cache_status": "computed",
            "topology_cache_reason": "migrated from the held CYTools reconstruction",
            "topology_cache_migration_source": str(source_path.resolve()),
            "topology_cache_migration_source_schema": persisted.get(
                "raw_frst_schema_version"
            ),
        }
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    record = write_raw_frst_artifact(
        destination_path,
        h11=h11,
        polytope_vertices=vertices,
        polytope_points=reconstructed_points,
        triangulation_labels=np.asarray(triangulation.labels, dtype=int),
        triangulation_points=np.asarray(triangulation.points(), dtype=int),
        simplices=reconstructed_simplices,
        simplex_indices=np.asarray(
            triangulation.simplices(as_indices=True), dtype=int
        ),
        metadata=metadata,
        topology_cache=topology,
        topology_cache_metadata=cache_metadata,
    )
    record["migration_elapsed_seconds"] = time.monotonic() - started
    record["migration_backend"] = backend
    return record


def validate_resumable_output(path):
    persisted = read_raw_frst_artifact(path)
    cache = persisted.get("topology_cache") or {}
    if cache.get("status") != "available":
        raise RuntimeError(f"existing output cache is unavailable: {path}")
    return persisted


def copy_and_rewrite_json(source_path, output_path, source_root, output_root):
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    atomic_json_dump(output_path, rewrite_paths(payload, source_root, output_root))


def finalize_manifests(
    source_root,
    output_root,
    source_paths,
    converted_records,
    started_unix_ns,
):
    source_status_path = source_root / "frst_terminal_statuses.jsonl"
    statuses = []
    by_source = {
        str(source.resolve()): record
        for source, record in zip(source_paths, converted_records)
    }
    if source_status_path.is_file():
        for line in source_status_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            source_path = record.get("raw_frst_path")
            converted = by_source.get(str(Path(source_path).resolve())) if source_path else None
            if converted is not None:
                record.update(
                    {
                        key: converted.get(key)
                        for key in (
                            "raw_frst_schema_version",
                            "geometry_id",
                            "topology_cache_status",
                            "topology_cache_reason",
                            "file_size_bytes",
                        )
                    }
                )
                record["raw_frst_path"] = converted["raw_frst_path"]
            elif source_path:
                record["raw_frst_path"] = str(
                    output_path_for(Path(source_path), source_root, output_root)
                )
            statuses.append(record)
    else:
        statuses = [
            {
                **record,
                "terminal_status": "retained_raw_frst",
                "terminal_reason": "migrated legacy retained raw FRST",
            }
            for record in converted_records
        ]
    atomic_jsonl_dump(output_root / "frst_terminal_statuses.jsonl", statuses)

    for source_path in (
        source_root / "polytope_manifest.json",
        source_root / "frst_candidates" / "frst_seed_pool_deficit_ledger.json",
        source_root / "frst_candidates" / "frst_seed_pool_probe_ledger.json",
        source_root
        / "frst_candidates"
        / "frst_selection_luck_replenishment_ledger.json",
    ):
        if source_path.is_file():
            copy_and_rewrite_json(
                source_path,
                output_root / source_path.relative_to(source_root),
                source_root,
                output_root,
            )

    source_manifest_path = source_root / "run_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    count_by_h11 = Counter(str(record["h11"]) for record in converted_records)
    output_manifest = dict(source_manifest)
    output_manifest.update(
        {
            "output_root": str(output_root.resolve()),
            "raw_frst_schema_version": RAW_FRST_SCHEMA_VERSION,
            "stage_boundary": (
                "stage 1 writes raw FRSTs plus an optional stage-2-independent "
                "topology cache; stage 2 is a separate run"
            ),
            "stage2_started": False,
            "retained_raw_frst_count": len(converted_records),
            "retained_raw_frst_count_by_h11": dict(sorted(count_by_h11.items())),
            "topology_cache_schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
            "topology_cache_compression": {
                "filter": "gzip",
                "compression_opts": 9,
                "shuffle": True,
            },
            "topology_cache_migration": {
                "source_root": str(source_root.resolve()),
                "source_run_manifest_sha256": (
                    file_sha256(source_manifest_path)
                    if source_manifest_path.is_file()
                    else None
                ),
                "source_population_preserved": True,
                "source_raw_frst_count": len(source_paths),
                "converted_raw_frst_count": len(converted_records),
                "converted_raw_frst_count_by_h11": dict(sorted(count_by_h11.items())),
                "migration_source_commit": git_commit(),
                "started_unix_ns": started_unix_ns,
                "finished_unix_ns": time.time_ns(),
                "python_version": platform.python_version(),
                "cytools_cache_directory": CYTOOLS_CACHE,
            },
        }
    )
    atomic_json_dump(output_root / "run_manifest.json", output_manifest)


def main(argv=None):
    arguments = parse_args(argv)
    source_root = Path(arguments.input_root).resolve()
    output_root = Path(arguments.output_root).resolve()
    if not source_root.is_dir():
        raise SystemExit(f"input root does not exist: {source_root}")
    if source_root == output_root:
        raise SystemExit("input and output roots must be different")
    sources = raw_paths(source_root)
    if not sources:
        raise SystemExit(f"no raw FRST artifacts found under {source_root}")
    if output_root.exists() and not arguments.resume:
        raise SystemExit(
            f"output root already exists; choose a new sibling or pass --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "topology_cache_migration_progress.jsonl"
    started_unix_ns = time.time_ns()
    append_jsonl_record(
        progress_path,
        {
            "event": "migration_started",
            "source_root": str(source_root),
            "output_root": str(output_root),
            "source_count": len(sources),
            "source_count_by_h11": dict(
                sorted(Counter(path.parts[-4] for path in sources).items())
            ),
            "started_unix_ns": started_unix_ns,
            "source_commit": git_commit(),
            "cytools_cache_directory": CYTOOLS_CACHE,
        },
    )
    converted_records = []
    for sequence, source_path in enumerate(sources, start=1):
        destination_path = output_path_for(source_path, source_root, output_root)
        try:
            if arguments.resume and destination_path.is_file():
                persisted = validate_resumable_output(destination_path)
                record = {
                    key: persisted.get(key)
                    for key in (
                        "h11",
                        "polytope_id",
                        "full_triangulation_hash",
                        "geometry_id",
                        "raw_frst_schema_version",
                        "topology_cache_status",
                        "topology_cache_reason",
                        "raw_frst_path",
                        "file_size_bytes",
                    )
                }
                record["migration_resumed"] = True
            else:
                record = reconstruct_and_cache(
                    source_path, destination_path, arguments.backend
                )
            converted_records.append(record)
            append_jsonl_record(
                progress_path,
                {
                    "event": "artifact_completed",
                    "sequence": sequence,
                    "source_path": str(source_path),
                    "output_path": str(destination_path),
                    "h11": record.get("h11"),
                    "geometry_id": record.get("geometry_id"),
                    "topology_cache_status": record.get("topology_cache_status"),
                    "elapsed_seconds": record.get("migration_elapsed_seconds"),
                },
            )
            if sequence == 1 or sequence % 25 == 0 or sequence == len(sources):
                print(
                    json.dumps(
                        {
                            "event": "artifact_completed",
                            "sequence": sequence,
                            "total": len(sources),
                            "h11": record.get("h11"),
                            "geometry_id": record.get("geometry_id"),
                            "elapsed_seconds": record.get("migration_elapsed_seconds"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except Exception as error:
            append_jsonl_record(
                progress_path,
                {
                    "event": "artifact_failed",
                    "sequence": sequence,
                    "source_path": str(source_path),
                    "output_path": str(destination_path),
                    "h11": source_path.parts[-4],
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                },
            )
            raise
    finalize_manifests(
        source_root,
        output_root,
        sources,
        converted_records,
        started_unix_ns,
    )
    append_jsonl_record(
        progress_path,
        {
            "event": "migration_finalized",
            "converted_count": len(converted_records),
            "finished_unix_ns": time.time_ns(),
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "source_root": str(source_root),
                "output_root": str(output_root),
                "converted_count": len(converted_records),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
