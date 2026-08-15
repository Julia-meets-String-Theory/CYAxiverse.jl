"""Append tested continuation-polytopes to a raw Stage-1 FRST population.

This is intentionally a Stage-1-only command.  It uses the unchanged CYTools
``fast`` sampler on favorable continuation candidates from the selection-luck
probe, writes only the number of FRSTs still needed, and never replaces an
existing raw artifact or starts Stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from cytools import Polytope

import generate_geometric_data_multitriangulation as generator
from glimmers_raw_frst import (
    RAW_FRST_SCHEMA_VERSION,
    compute_triangulation_hash,
    discover_raw_frst_paths,
    read_raw_frst_artifact,
    stable_hash,
    write_raw_frst_artifact,
)


TARGET_BY_H11 = {50: 500, 100: 500, 200: 300, 491: 100}
FRSTS_PER_REPLACEMENT_POLYTOPE = 10
SAMPLER_CONTRACT = {
    "sampler": "Polytope.random_triangulations_fast",
    "N": 10,
    "c": 0.2,
    "max_retries": 500,
    "make_star": True,
    "only_fine": True,
    "backend": "cgal",
    "as_list": False,
    "progress_bar": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_json_dump(path: Path, payload) -> None:
    """Atomically update a ledger/manifest without touching raw HDF5 files."""
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_record(record):
    # ``read_raw_frst_artifact`` also exposes the optional decoded topology
    # cache.  It contains NumPy arrays and is deliberately not part of the
    # JSON ledger/manifest representation.
    return {
        key: value
        for key, value in record.items()
        if key not in {"arrays", "topology_cache"}
    }


def recount_raw_population(stage1_root: Path):
    """Validate every existing artifact and return counts, identities, and records."""
    records = []
    identities = {}
    by_h11 = Counter()
    by_polytope = defaultdict(list)
    for path in discover_raw_frst_paths(stage1_root):
        metadata = read_raw_frst_artifact(path, include_topology_cache=False)
        identity = (metadata["polytope_id"], metadata["full_triangulation_hash"])
        if identity in identities:
            raise RuntimeError(
                "existing duplicate full triangulation: "
                f"{path} duplicates {identities[identity]}"
            )
        identities[identity] = str(path.resolve())
        by_h11[int(metadata["h11"])] += 1
        by_polytope[(int(metadata["h11"]), int(metadata.get("polytope_index", 0)))].append(
            metadata
        )
        record = _canonical_record(metadata)
        record.update(
            {
                "raw_frst_path": str(path.resolve()),
                "terminal_status": "retained_raw_frst",
                "terminal_reason": "validated existing Stage-1 raw FRST",
            }
        )
        records.append(record)
    return records, identities, by_h11, by_polytope


def load_probe_candidates(pool_path: Path, probe_path: Path):
    with pool_path.open(encoding="utf-8") as stream:
        pool = json.load(stream)
    with probe_path.open(encoding="utf-8") as stream:
        probe = json.load(stream)
    if pool.get("schema_version") != "cyaxiverse-selection-luck-pool-1":
        raise RuntimeError("unexpected selection pool schema")
    if probe.get("schema_version") != "cyaxiverse-selection-luck-fast-probe-1":
        raise RuntimeError("unexpected selection probe schema")
    result_by_key = {
        (int(result["h11"]), int(result["pool_index"])): result
        for result in probe.get("results", [])
    }
    candidates = {}
    for raw_h11, entries in pool.get("candidates", {}).items():
        h11 = int(raw_h11)
        candidates[h11] = []
        for entry in entries:
            if entry.get("in_original_manifest"):
                continue
            key = (h11, int(entry["pool_index"]))
            result = result_by_key.get(key)
            if result is None or not result.get("target_met"):
                continue
            if int(result.get("returned_count", 0)) < FRSTS_PER_REPLACEMENT_POLYTOPE:
                continue
            candidates[h11].append({"pool": entry, "probe": result})
    return pool, probe, candidates


def choose_replacements(deficits, candidates):
    """Choose the earliest tested continuation candidates needed for each deficit."""
    chosen = []
    for h11 in sorted(deficits):
        remaining = int(deficits[h11])
        if remaining <= 0:
            continue
        for candidate in candidates.get(h11, []):
            if remaining <= 0:
                break
            target = min(FRSTS_PER_REPLACEMENT_POLYTOPE, remaining)
            chosen.append({"h11": h11, "target": target, **candidate})
            remaining -= target
        if remaining:
            raise RuntimeError(
                f"tested continuation pool cannot fill h11={h11} deficit {deficits[h11]}"
            )
    return chosen


def _source_record(pool_entry, pool_path: Path, pool_sha256: str):
    return {
        "source_kind": "local_parquet_mirror_selection_luck_continuation",
        "selection_pool_path": str(pool_path.resolve()),
        "selection_pool_sha256": pool_sha256,
        "selection_pool_index": int(pool_entry["pool_index"]),
        "selection_pool_row_index": int(pool_entry["row_index"]),
        "selection_pool_partition": pool_entry["partition"],
        "selection_pool_polytope_key_sha256": pool_entry["polytope_key_sha256"],
        "selection_rule": "parquet partition order, row order, unique vertex sets, CYTools is_favorable(lattice=N)",
        "original_manifest_member": False,
    }


def append_candidate(
    stage1_root: Path,
    selected,
    *,
    run_seed: int,
    pool_path: Path,
    probe_path: Path,
    pool_sha256: str,
    probe_sha256: str,
    identities,
):
    h11 = int(selected["h11"])
    target = int(selected["target"])
    pool_entry = selected["pool"]
    probe_entry = selected["probe"]
    pool_index = int(pool_entry["pool_index"])
    stream_seed = int(probe_entry["seed"])
    polytope = Polytope(
        np.asarray(pool_entry["vertices"], dtype=int), deterministic_glsm_basis=True
    )
    if int(polytope.h11()) != h11:
        raise RuntimeError(
            f"pool candidate h11 mismatch: requested {h11}, CYTools returned {polytope.h11()}"
        )
    if not bool(polytope.is_favorable(lattice="N")):
        raise RuntimeError(f"pool candidate {h11}/{pool_index} is not favorable")

    polytope_id, polytope_points = generator.polytope_identity(polytope)
    output_dir = (
        stage1_root
        / "frst_candidates"
        / f"h11_{h11:03d}"
        / f"np_{pool_index:07d}"
    )
    existing_paths = sorted(output_dir.glob("frst_*.h5")) if output_dir.is_dir() else []
    if existing_paths:
        raise FileExistsError(
            "replacement output directory already contains artifacts; refusing to "
            f"append without an explicit resume decision: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    source = _source_record(pool_entry, pool_path, pool_sha256)
    source["selection_probe_path"] = str(probe_path.resolve())
    source["selection_probe_sha256"] = probe_sha256
    stage1_run_id = stable_hash(
        {
            "mechanism": "selection_luck_continuation_replenishment",
            "run_seed": run_seed,
            "h11": h11,
            "pool_index": pool_index,
            "stream_seed": stream_seed,
        }
    )
    candidates = generator.triangulation_candidates(
        polytope,
        "fast",
        FRSTS_PER_REPLACEMENT_POLYTOPE,
        SAMPLER_CONTRACT["max_retries"],
        SAMPLER_CONTRACT["backend"],
        stream_seed,
        None,
        None,
        None,
        8,
        1e-2,
        25,
        0.2,
        "fast",
        17,
        1000,
    )
    polytope_vertices = np.asarray(polytope.vertices(), dtype=int)
    stream_records = []
    retained = 0
    proposals = 0
    duplicates = 0
    failures = 0
    for triangulation in candidates:
        proposals += 1
        proposal_seed = stream_seed + proposals - 1
        record = {
            "h11": h11,
            "polytope_index": pool_index,
            "candidate_index": proposals,
            "proposal_index": proposals,
            "proposal_seed": int(proposal_seed),
            "stream_index": 1,
            "stream_seed": stream_seed,
            "sampler": "fast",
            "raw_frst_path": None,
        }
        try:
            generator.validate_frst(polytope, triangulation)
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            full_hash = compute_triangulation_hash(simplices)
            record["full_triangulation_hash"] = full_hash
            identity = (polytope_id, full_hash)
            if identity in identities:
                duplicates += 1
                record.update(
                    {
                        "terminal_status": "duplicate_full_triangulation",
                        "terminal_reason": "identity already present in retained raw population",
                    }
                )
                stream_records.append(record)
                continue
            raw_path = output_dir / f"frst_{retained + 1:07d}.h5"
            metadata = {
                "artifact_type": "raw_frst_candidate",
                "stage": "frst_collection",
                "stage1_run_id": stage1_run_id,
                "run_seed": int(run_seed),
                "h11": h11,
                "polytope_index": pool_index,
                "polytope_source": source,
                "candidate_index": proposals,
                "proposal_index": proposals,
                "proposal_seed": int(proposal_seed),
                "stream_index": 1,
                "stream_seed": stream_seed,
                "seed_derivation": "selection-luck probe seed recorded verbatim",
                "sampler": "fast",
                "sampling_metadata": {
                    **SAMPLER_CONTRACT,
                    "seed": stream_seed,
                    "proposal_budget": FRSTS_PER_REPLACEMENT_POLYTOPE,
                    "retry_budget": SAMPLER_CONTRACT["max_retries"],
                    "sampling_unit": "frst",
                    "selection_status": "biased_random_height_proposal",
                    "replacement_target": target,
                    "replacement_mechanism": "tested_continuation_polytope",
                },
                "ks_database_version": "local parquet mirror; selection-luck continuation pool",
                "replenishment": {
                    "mechanism": "alternate_favorable_polytope_selection",
                    "pool_index": pool_index,
                    "probe_result": probe_entry,
                    "selection_pool_sha256": pool_sha256,
                    "selection_probe_sha256": probe_sha256,
                },
            }
            retained_record = write_raw_frst_artifact(
                raw_path,
                h11=h11,
                polytope_vertices=polytope_vertices,
                polytope_points=np.asarray(polytope.points(), dtype=int),
                triangulation_labels=np.asarray(triangulation.labels, dtype=int),
                triangulation_points=np.asarray(triangulation.points(), dtype=int),
                simplices=simplices,
                simplex_indices=np.asarray(
                    triangulation.simplices(as_indices=True), dtype=int
                ),
                metadata=metadata,
            )
            retained += 1
            identities[identity] = str(raw_path.resolve())
            record = retained_record
            record.update(
                {
                    "terminal_status": "retained_raw_frst",
                    "terminal_reason": "selection-luck replacement serialized before Stage 2",
                }
            )
        except Exception as exc:
            failures += 1
            record.update(
                {
                    "terminal_status": "invalid_frst",
                    "terminal_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        stream_records.append(record)
        if retained >= target:
            break

    if retained < target:
        raise RuntimeError(
            f"replacement stream h11={h11}, pool_index={pool_index} retained "
            f"{retained} of requested {target}"
        )
    stream_records.append(
        {
            "h11": h11,
            "polytope_index": pool_index,
            "stream_index": 1,
            "stream_seed": stream_seed,
            "sampler": "fast",
            "proposal_budget": FRSTS_PER_REPLACEMENT_POLYTOPE,
            "retry_budget": SAMPLER_CONTRACT["max_retries"],
            "candidate_count": proposals,
            "accepted_new_count": retained,
            "duplicate_full_triangulation_count": duplicates,
            "generation_failure_count": failures,
            "source_exhausted": False,
            "terminal_status": "target_reached",
            "terminal_reason": "replacement target reached without Stage-2 processing",
        }
    )
    return {
        "h11": h11,
        "polytope_index": pool_index,
        "pool_index": pool_index,
        "row_index": int(pool_entry["row_index"]),
        "polytope_id": polytope_id,
        "stream_seed": stream_seed,
        "requested": target,
        "candidate_count": proposals,
        "accepted_new_count": retained,
        "duplicate_full_triangulation_count": duplicates,
        "generation_failure_count": failures,
        "source_exhausted": False,
        "raw_paths": [
            str(path.resolve()) for path in sorted(output_dir.glob("frst_*.h5"))
        ],
        "terminal_records": stream_records,
    }


def existing_replenishment_records(records):
    """Recover an interrupted replenishment run from persisted artifact metadata."""
    grouped = defaultdict(list)
    for record in records:
        replenishment = record.get("replenishment")
        if not isinstance(replenishment, dict):
            continue
        pool_index = replenishment.get("pool_index")
        if pool_index is None:
            continue
        grouped[(int(record["h11"]), int(pool_index))].append(record)
    recovered = []
    for (h11, pool_index), group in sorted(grouped.items()):
        first = group[0]
        probe_result = first["replenishment"].get("probe_result", {})
        recovered.append(
            {
                "h11": h11,
                "polytope_index": pool_index,
                "pool_index": pool_index,
                "row_index": int(probe_result.get("row_index", -1)),
                "polytope_id": first.get("polytope_id"),
                "stream_seed": int(first.get("stream_seed", first.get("sampling_metadata", {}).get("seed", 0))),
                "requested": len(group),
                "candidate_count": int(probe_result.get("returned_count", len(group))),
                "accepted_new_count": len(group),
                "duplicate_full_triangulation_count": 0,
                "generation_failure_count": 0,
                "source_exhausted": False,
                "raw_paths": sorted(record["raw_frst_path"] for record in group),
                "recovered_from_artifacts": True,
            }
        )
    return recovered


def rebuild_manifests(stage1_root: Path, records, identities, before_counts, replenishment):
    by_h11 = Counter(int(record["h11"]) for record in records)
    by_polytope = defaultdict(list)
    for record in records:
        by_polytope[(int(record["h11"]), int(record.get("polytope_index", 0)))].append(record)
    missing = {
        str(h11): max(0, TARGET_BY_H11[h11] - by_h11[h11]) for h11 in TARGET_BY_H11
    }
    if any(missing.values()):
        raise RuntimeError(f"raw population remains incomplete: {missing}")

    all_terminal_records = []
    polytope_entries = []
    for key in sorted(by_polytope):
        h11, polytope_index = key
        group = sorted(by_polytope[key], key=lambda record: str(record["raw_frst_path"]))
        first = read_raw_frst_artifact(
            group[0]["raw_frst_path"], include_topology_cache=False
        )
        arrays = first["arrays"]
        source = first.get("polytope_source") or first.get("source")
        polytope_entries.append(
            {
                "h11": h11,
                "polytope_index": polytope_index,
                "polytope_id": first["polytope_id"],
                "vertices": np.asarray(arrays["polytope_vertices"], dtype=int).tolist(),
                "source": source,
                "target_raw_frsts": len(group),
            }
        )
        all_terminal_records.extend(group)

    counts = {
        str(h11): {
            "target": TARGET_BY_H11[h11],
            "observed": by_h11[h11],
            "missing": 0,
            "observed_polytope_count": sum(1 for key in by_polytope if key[0] == h11),
            "polytope_target": 10,
        }
        for h11 in sorted(TARGET_BY_H11)
    }
    ledger = {
        "schema_version": "cyaxiverse-frst-deficit-ledger-2",
        "artifact_glob": "frst_candidates/h11_*/np_*/frst_*.h5",
        "target_by_h11": {str(k): v for k, v in TARGET_BY_H11.items()},
        "by_h11": counts,
        "complete_polytope_count": sum(len(values) == 10 for values in by_polytope.values()),
        "incomplete_polytopes": [],
        "duplicate_full_triangulation_groups": [],
        "recount_before_replenishment": {str(k): v for k, v in before_counts.items()},
        "replenishment_status": "complete",
        "replenishment_ledger": "frst_candidates/frst_selection_luck_replenishment_ledger.json",
        "created_unix_ns": time.time_ns(),
    }
    _replace_json_dump(stage1_root / "frst_candidates" / "frst_seed_pool_deficit_ledger.json", ledger)
    _replace_json_dump(
        stage1_root / "polytope_manifest.json",
        {"schema_version": RAW_FRST_SCHEMA_VERSION, "polytopes": polytope_entries},
    )
    _write_jsonl(stage1_root / "frst_terminal_statuses.jsonl", all_terminal_records)

    try:
        cytools_version = importlib.metadata.version("cytools")
    except importlib.metadata.PackageNotFoundError:
        cytools_version = None
    run_manifest = {
        "schema_version": RAW_FRST_SCHEMA_VERSION,
        "stage": "stage1_raw_frst_collection",
        "status": "completed",
        "output_root": str(stage1_root.resolve()),
        "source_commit": _git_commit(),
        "python_version": platform.python_version(),
        "cytools_version": cytools_version,
        "hardware_threads": os.cpu_count(),
        "approved_plan": {str(k): v for k, v in TARGET_BY_H11.items()},
        "target_raw_frst_count": sum(TARGET_BY_H11.values()),
        "target_raw_frst_count_by_h11": {str(k): v for k, v in TARGET_BY_H11.items()},
        "retained_raw_frst_count": len(records),
        "retained_raw_frst_count_by_h11": {str(k): by_h11[k] for k in sorted(by_h11)},
        "duplicate_full_triangulation_count": 0,
        "stage_boundary": (
            "stage 1 writes raw FRSTs plus an optional stage-2-independent "
            "topology cache; stage 2 is a separate run"
        ),
        "stage2_started": False,
        "replenishment": {
            "mechanism": "tested alternate favorable polytope selection",
            "records": [
                {key: value for key, value in item.items() if key != "terminal_records"}
                for item in replenishment
            ],
        },
        "user_decision": "approved population completion with successful continuation candidates",
    }
    _replace_json_dump(stage1_root / "run_manifest.json", run_manifest)
    return ledger, run_manifest


def _git_commit():
    repository_root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", required=True)
    parser.add_argument("--selection-pool", required=True)
    parser.add_argument("--probe-report", required=True)
    parser.add_argument("--run-seed", type=int, default=20260813)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    stage1_root = Path(arguments.stage1_root).resolve()
    pool_path = Path(arguments.selection_pool).resolve()
    probe_path = Path(arguments.probe_report).resolve()
    if not stage1_root.is_dir():
        raise FileNotFoundError(stage1_root)
    if not pool_path.is_file() or not probe_path.is_file():
        raise FileNotFoundError("selection pool or probe report is missing")

    records, identities, by_h11, _ = recount_raw_population(stage1_root)
    deficits = {
        h11: max(0, TARGET_BY_H11[h11] - by_h11[h11]) for h11 in TARGET_BY_H11
    }
    print(
        json.dumps(
            {
                "status": "recounted",
                "counts": {str(k): by_h11[k] for k in sorted(TARGET_BY_H11)},
                "deficits": {str(k): deficits[k] for k in sorted(deficits)},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if arguments.dry_run:
        return 0

    pool, probe, candidates = load_probe_candidates(pool_path, probe_path)
    selected = choose_replacements(deficits, candidates)
    pool_sha256 = _sha256_file(pool_path)
    probe_sha256 = _sha256_file(probe_path)
    replenishment = []
    terminal_records = []
    before_counts = dict(by_h11)
    for item in selected:
        result = append_candidate(
            stage1_root,
            item,
            run_seed=arguments.run_seed,
            pool_path=pool_path,
            probe_path=probe_path,
            pool_sha256=pool_sha256,
            probe_sha256=probe_sha256,
            identities=identities,
        )
        replenishment.append(result)
        terminal_records.extend(result.pop("terminal_records"))
        refreshed_records, _, refreshed_counts, _ = recount_raw_population(stage1_root)
        records = refreshed_records
        by_h11 = refreshed_counts
        print(
            json.dumps(
                {
                    "status": "replacement_completed",
                    "h11": result["h11"],
                    "pool_index": result["pool_index"],
                    "accepted_new_count": result["accepted_new_count"],
                    "counts": {str(k): by_h11[k] for k in sorted(TARGET_BY_H11)},
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # If raw generation completed before manifest finalization (for example,
    # after an intentional no-overwrite collision), recover its append records
    # from the persisted artifact metadata instead of regenerating anything.
    if not selected:
        recovered = existing_replenishment_records(records)
        if recovered:
            replenishment = recovered
            for item in recovered:
                before_counts[item["h11"]] -= item["accepted_new_count"]

    # Rebuild terminal records from persisted files so the ledger remains valid
    # even if a future run is interrupted after an individual append.
    records, identities, by_h11, _ = recount_raw_population(stage1_root)
    ledger, run_manifest = rebuild_manifests(
        stage1_root, records, identities, before_counts, replenishment
    )
    replenishment_ledger = {
        "schema_version": "cyaxiverse-selection-luck-replenishment-1",
        "status": "completed",
        "stage": "frst_collection",
        "stage2_started": False,
        "run_seed": arguments.run_seed,
        "sampler_contract": SAMPLER_CONTRACT,
        "selection_pool_path": str(pool_path),
        "selection_pool_sha256": pool_sha256,
        "selection_probe_path": str(probe_path),
        "selection_probe_sha256": probe_sha256,
        "counts_before": {str(k): before_counts.get(k, 0) for k in sorted(TARGET_BY_H11)},
        "counts_after": {str(k): by_h11.get(k, 0) for k in sorted(TARGET_BY_H11)},
        "deficits_filled": {
            str(k): by_h11.get(k, 0) - before_counts.get(k, 0)
            for k in sorted(TARGET_BY_H11)
        },
        "selected_replacements": replenishment,
        "approved_target": {str(k): v for k, v in TARGET_BY_H11.items()},
        "unique_identity_count": len(identities),
        "raw_artifact_count": len(records),
        "ledger_paths": {
            "deficit": str(
                (stage1_root / "frst_candidates" / "frst_seed_pool_deficit_ledger.json").resolve()
            ),
            "terminal": str((stage1_root / "frst_terminal_statuses.jsonl").resolve()),
            "run_manifest": str((stage1_root / "run_manifest.json").resolve()),
        },
    }
    _replace_json_dump(
        stage1_root / "frst_candidates" / "frst_selection_luck_replenishment_ledger.json",
        replenishment_ledger,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "counts": {str(k): by_h11[k] for k in sorted(TARGET_BY_H11)},
                "total": len(records),
                "unique_identities": len(identities),
                "target": sum(TARGET_BY_H11.values()),
                "stage2_started": run_manifest["stage2_started"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
