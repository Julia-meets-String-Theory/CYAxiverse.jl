#!/usr/bin/env python3
"""Run bounded, explicitly biased exact-support comparison samplers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "validation/cyax_0115_uniform_frst"
SUPPORT_DIR = VALIDATION / "exact_support_hard_capped"
MANIFEST = SUPPORT_DIR / "validation_manifest.json"
MANIFEST_SHA = "04254692b901a0b2a2d18d2bb21d7647af9aa48e0ce7d7173307a82ff6563263"
CACHE_ROOT = Path("/private/tmp/cyax-0115-comparison-cache")
PROPOSAL_COUNT = 100
WALL_CAP_SECONDS = 300


def load_validator():
    path = ROOT / "scripts/validate_uniform_frst_ensemble.py"
    spec = importlib.util.spec_from_file_location("cyax0115_validator_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validation driver: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def comparison_seed(fixture_order: int, proposal_index: int = 1) -> int:
    return 0x4A115000 + fixture_order * 1000 + proposal_index


def run_worker(fixture_id: str, method: str, seed: int, result_path: Path):
    v = load_validator()
    manifest = json.loads(MANIFEST.read_bytes())
    if v.file_sha256(MANIFEST) != MANIFEST_SHA:
        raise v.ValidationError("frozen manifest hash changed before comparison")
    fixture = next(item for item in manifest["fixtures"] if item["candidate_id"] == fixture_id)
    if seed != comparison_seed(fixture["fixture_order"], 1):
        raise v.ValidationError("worker seed does not match the frozen first proposal seed")
    support = json.loads((SUPPORT_DIR / fixture["support_file"]).read_bytes())
    support_ids = {state["full_triangulation_sha256"] for state in support["states"]}
    cache_dir = CACHE_ROOT / f"{fixture_id}-{method}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # platformdirs uses ~/Library/Caches on macOS, which is outside this task's
    # writable roots. Redirect only CYTools' runtime cache to this task cache.
    import platformdirs

    platformdirs.user_cache_dir = lambda *args, **kwargs: str(cache_dir)
    sys.path.insert(0, str(ROOT / "scripts"))
    generation_path = ROOT / "scripts/generate_geometric_data_multitriangulation.py"
    generation_spec = importlib.util.spec_from_file_location(
        f"cyax0115_generation_{fixture['fixture_order']}_{method}", generation_path
    )
    if generation_spec is None or generation_spec.loader is None:
        raise RuntimeError(f"cannot load repository comparison route: {generation_path}")
    generation = importlib.util.module_from_spec(generation_spec)
    sys.modules[generation_spec.name] = generation
    generation_spec.loader.exec_module(generation)
    cytools, _, _, _ = v.bootstrap_cytools(cache_dir)
    poly = v.construct_candidate(cytools, fixture["source_fixture"])
    journal = result_path.with_suffix(".progress.jsonl")
    if method not in ("fast", "ntfe_fast"):
        raise ValueError(f"unsupported comparison method: {method}")

    full_ids = []
    two_face_ids = []
    member_count = 0
    failures = []
    for proposal_index in range(1, PROPOSAL_COUNT + 1):
        proposal_seed = 0x4A115000 + fixture["fixture_order"] * 1000 + proposal_index
        v.append_jsonl_fsync(
            journal,
            {
                "event": "proposal_started",
                "fixture_id": fixture_id,
                "method": method,
                "seed": proposal_seed,
                "proposal_index": proposal_index,
            },
        )
        stream = generation.triangulation_candidates(
            poly,
            method,
            1,
            50,
            "cgal",
            proposal_seed,
            None,
            None,
            None,
            8,
            0.01,
            25,
            0.2,
            "fast",
            17,
            1000,
        )
        try:
            triangulation = next(iter(stream))
        except StopIteration:
            record = {
                "event": "proposal_no_yield",
                "fixture_id": fixture_id,
                "method": method,
                "seed": proposal_seed,
                "proposal_index": proposal_index,
            }
            failures.append(record)
            v.append_jsonl_fsync(journal, record)
            continue
        except Exception as exc:
            record = {
                "event": "proposal_error",
                "fixture_id": fixture_id,
                "method": method,
                "seed": proposal_seed,
                "proposal_index": proposal_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(record)
            v.append_jsonl_fsync(journal, record)
            continue
        full_id = v.full_triangulation_identity(triangulation)
        two_face_id = v.two_face_identity(triangulation)
        full_ids.append(full_id)
        two_face_ids.append(two_face_id)
        member_count += full_id in support_ids
        v.append_jsonl_fsync(
            journal,
            {
                "event": "comparison_candidate",
                "fixture_id": fixture_id,
                "method": method,
                "seed": proposal_seed,
                "proposal_index": proposal_index,
                "full_triangulation_sha256": full_id,
                "two_face_sha256": two_face_id,
                "support_member": full_id in support_ids,
            },
        )
    result = {
        "schema": "CYAX-0115-r4-biased-comparison-v1",
        "manifest_sha256": MANIFEST_SHA,
        "fixture_id": fixture_id,
        "fixture_order": fixture["fixture_order"],
        "K": fixture["K"],
        "method": method,
        "interpretation": "biased diagnostic only; no Uniform-FRST or population claim",
        "proposal_seed_rule": "one fresh frozen seed per 1-based proposal index",
        "seed_rule": "0x4A115000 + fixture_order*1000 + proposal_index",
        "first_proposal_seed": 0x4A115000 + fixture["fixture_order"] * 1000 + 1,
        "last_proposal_seed": 0x4A115000 + fixture["fixture_order"] * 1000 + PROPOSAL_COUNT,
        "proposal_count": PROPOSAL_COUNT,
        "yielded_count": len(full_ids),
        "unique_full_identity_count": len(set(full_ids)),
        "unique_two_face_identity_count": len(set(two_face_ids)),
        "exact_support_member_count": member_count,
        "no_yield_or_error_count": len(failures),
        "no_yield_or_error_records": failures,
        "full_identity_sequence": full_ids,
        "two_face_identity_sequence": two_face_ids,
        "progress_journal": journal.name,
    }
    v.atomic_json_create(result_path, result)
    print(json.dumps(result, sort_keys=True))


def run_all(output_dir: Path):
    v = load_validator()
    manifest = json.loads(MANIFEST.read_bytes())
    if v.file_sha256(MANIFEST) != MANIFEST_SHA:
        raise v.ValidationError("frozen manifest hash changed before comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for fixture in manifest["fixtures"]:
        for method in ("fast", "ntfe_fast"):
            initial_seed = comparison_seed(fixture["fixture_order"], 1)
            output_path = output_dir / f"{fixture['candidate_id']}-{method}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--fixture-id",
                fixture["candidate_id"],
                "--method",
                method,
                "--seed",
                str(initial_seed),
                "--result-path",
                str(output_path),
            ]
            outcome = v.run_bounded_process(command, WALL_CAP_SECONDS)
            entry = {
                "fixture_id": fixture["candidate_id"],
                "fixture_order": fixture["fixture_order"],
                "method": method,
                "first_proposal_seed": initial_seed,
                "last_proposal_seed": comparison_seed(fixture["fixture_order"], PROPOSAL_COUNT),
                "proposal_count": PROPOSAL_COUNT,
                "yield_target": PROPOSAL_COUNT,
                "wall_cap_seconds": WALL_CAP_SECONDS,
                "elapsed_wall_seconds": outcome["elapsed_wall_seconds"],
                "timed_out": outcome["timed_out"],
                "returncode": outcome["returncode"],
                "worker_stdout_sha256": v.sha256_bytes(outcome["stdout"].encode("utf-8")),
                "worker_stderr_sha256": v.sha256_bytes(outcome["stderr"].encode("utf-8")),
                "result_path": str(output_path.relative_to(ROOT)),
                "progress_path": str(output_path.with_suffix(".progress.jsonl").relative_to(ROOT)),
            }
            if output_path.exists():
                entry["result_sha256"] = v.file_sha256(output_path)
                result = json.loads(output_path.read_bytes())
                entry["yielded_count"] = result["yielded_count"]
                entry["unique_full_identity_count"] = result["unique_full_identity_count"]
                entry["unique_two_face_identity_count"] = result["unique_two_face_identity_count"]
                entry["exact_support_member_count"] = result["exact_support_member_count"]
                entry["no_yield_or_error_count"] = result["no_yield_or_error_count"]
            else:
                entry["partial_progress_present"] = output_path.with_suffix(".progress.jsonl").exists()
            comparisons.append(entry)
            # Preserve a bounded checkpoint after every method, including failures.
            checkpoint = {
                "schema": "CYAX-0115-r4-biased-comparison-checkpoint-v1",
                "manifest_sha256": MANIFEST_SHA,
                "wall_cap_seconds_per_fixture_method": WALL_CAP_SECONDS,
                "proposal_cap_per_fixture_method": PROPOSAL_COUNT,
                "comparisons": list(comparisons),
            }
            checkpoint_path = output_dir / "comparison-checkpoints.jsonl"
            v.append_jsonl_fsync(checkpoint_path, checkpoint)
            print(json.dumps(entry, sort_keys=True), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fixture-id")
    parser.add_argument("--method", choices=("fast", "ntfe_fast"))
    parser.add_argument("--seed", type=int, help="The first proposal seed, fixed by the manifest formula")
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=VALIDATION / "calibration/comparisons_per_proposal")
    args = parser.parse_args()
    if args.worker:
        if not args.fixture_id or not args.method or args.seed is None or args.result_path is None:
            parser.error("worker mode requires fixture, method, seed, and result path")
        run_worker(args.fixture_id, args.method, args.seed, args.result_path)
    else:
        raise SystemExit(run_all(args.output_dir))


if __name__ == "__main__":
    main()
