"""Collect the fixed stage-1 raw FRST population.

This command deliberately stops after serializing retained FRST data.  It does
not construct a CY3, choose an orientifold, search the Kähler cone, assign
visible-sector divisors, or write EFT rows.  Run
``generate_stage2_eft_reference.py`` separately on its output.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from cytools import Polytope, fetch_polytopes

import generate_geometric_data_multitriangulation as generator
from glimmers_raw_frst import (
    RAW_FRST_SCHEMA_VERSION,
    count_by_h11,
    stable_hash,
    compute_triangulation_hash,
    write_raw_frst_artifact,
)
from glimmers_schema11 import atomic_json_dump, atomic_jsonl_dump, ensure_fresh_output_root


APPROVED_PLAN = {50: 500, 100: 500, 200: 300, 491: 100}
POLYTOPE_COUNTS = {50: 50, 100: 50, 200: 30, 491: 1}
FRSTS_PER_POLYTOPE = {50: 10, 100: 10, 200: 10, 491: 100}


def parse_raw_frst_plan(value):
    """Parse a comma-separated ``h11:raw-FRST-count`` plan."""
    plan = {}
    for token in str(value).replace(" ", "").split(","):
        if not token or ":" not in token:
            raise ValueError("plan must use comma-separated h11:count entries")
        raw_h11, raw_count = token.split(":", 1)
        try:
            h11, count = int(raw_h11), int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid plan entry {token!r}") from exc
        if h11 < 1 or count < 1 or h11 in plan:
            raise ValueError(f"invalid or duplicate plan entry {token!r}")
        plan[h11] = count
    if not plan:
        raise ValueError("the plan must not be empty")
    return dict(sorted(plan.items()))


def load_source_polytopes(arguments, h11, polytope_limit, polytope_manifest):
    if polytope_manifest is not None:
        entries = polytope_manifest["by_h11"].get(h11, [])[:polytope_limit]
        return [
            (
                Polytope(polytope_vertices, deterministic_glsm_basis=True),
                {
                    "source_kind": "local_polytope_manifest",
                    "manifest_path": os.path.abspath(arguments.polytope_manifest),
                    "manifest_source": polytope_manifest.get("source"),
                    "selection_index": index,
                },
            )
            for index, polytope_vertices in enumerate(entries, start=1)
        ]
    polytopes = fetch_polytopes(
        h11=h11,
        limit=polytope_limit,
        lattice="N",
        favorable=True,
        deterministic_glsm_basis=True,
    )
    return [
        (
            polytope,
            {
                "source_kind": "cytools_fetch_polytopes",
                "query": {
                    "h11": h11,
                    "lattice": "N",
                    "favorable": True,
                    "limit": polytope_limit,
                    "deterministic_glsm_basis": True,
                },
                "selection_index": index,
            },
        )
        for index, polytope in enumerate(polytopes, start=1)
    ]


def allocate_raw_frst_targets(h11, raw_frst_target, available_polytopes):
    """Distribute a requested h11 target over the retained polytope order."""
    if not available_polytopes:
        return []
    if h11 in APPROVED_PLAN and raw_frst_target == APPROVED_PLAN[h11]:
        count = min(POLYTOPE_COUNTS[h11], len(available_polytopes))
        per_polytope = FRSTS_PER_POLYTOPE[h11]
        total = min(raw_frst_target, count * per_polytope)
    else:
        count = min(
            len(available_polytopes),
            max(1, int(np.ceil(raw_frst_target / max(FRSTS_PER_POLYTOPE.get(h11, 1), 1)))),
        )
        total = raw_frst_target
    targets = [0] * count
    for index in range(total):
        targets[index % count] += 1
    return targets


def build_raw_frst_metadata(
    arguments, h11, polytope_index, polytope, source, proposal_index, proposal_seed
):
    identifier, polytope_points = generator.polytope_identity(polytope)
    return {
        "stage1_run_id": stable_hash(
            {"seed": arguments.seed, "h11": h11, "polytope_index": polytope_index}
        ),
        "polytope_index": polytope_index,
        "polytope_source": source,
        "candidate_index": proposal_index,
        "proposal_index": proposal_index,
        "proposal_seed": int(proposal_seed),
        "sampler": arguments.sampling_scheme_by_h11.get(str(h11)),
        "sampler_metadata": {
            "scheme": arguments.sampling_scheme_by_h11.get(str(h11)),
            "backend": arguments.backend,
            "proposal_budget": arguments.proposal_budget_by_h11.get(str(h11)),
            "retry_budget": arguments.retry_budget,
        },
        "polytope_point_count": len(polytope_points),
    }


def collect_raw_frsts_for_polytope(
    arguments,
    h11,
    polytope_index,
    polytope,
    source,
    raw_frst_target,
    seen_triangulation_identities,
):
    """Collect one polytope's raw FRSTs and return stage-1 records."""
    if raw_frst_target <= 0:
        return []
    sampler_name = arguments.sampling_scheme_by_h11[str(h11)]
    proposal_budget = arguments.proposal_budget_by_h11[str(h11)]
    candidates = generator.triangulation_candidates(
        polytope,
        sampler_name,
        proposal_budget,
        arguments.retry_budget,
        arguments.backend,
        arguments.seed + h11 + polytope_index,
        None,
        None,
        None,
        arguments.fine_tune_steps,
        arguments.walk_step_size,
        arguments.max_steps_to_wall,
        arguments.fast_height_scale,
        "fast",
        arguments.ntfe_max_face_points,
        arguments.ntfe_face_pool_size,
    )
    polytope_points = np.asarray(polytope.points(), dtype=int)
    polytope_vertices = np.asarray(polytope.vertices(), dtype=int)
    polytope_identifier, _ = generator.polytope_identity(polytope)
    terminal_records = []
    retained_count = 0
    attempted_count = 0
    for proposal_index, triangulation in enumerate(candidates, start=1):
        attempted_count += 1
        proposal_seed = arguments.seed + h11 * 1_000_000 + polytope_index * 10_000 + proposal_index
        terminal_record = {
            "h11": h11,
            "polytope_index": polytope_index,
            "candidate_index": proposal_index,
            "proposal_seed": int(proposal_seed),
            "sampler": sampler_name,
            "polytope_id": polytope_identifier,
            "raw_frst_path": None,
        }
        try:
            generator.validate_frst(polytope, triangulation)
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            full_hash = compute_triangulation_hash(simplices)
            terminal_record["full_triangulation_hash"] = full_hash
            triangulation_identity = (polytope_identifier, full_hash)
            if triangulation_identity in seen_triangulation_identities:
                terminal_record.update(
                    {
                        "terminal_status": "duplicate_full_triangulation",
                        "terminal_reason": (
                            "duplicate of "
                            f"{seen_triangulation_identities[triangulation_identity]}"
                        ),
                    }
                )
                terminal_records.append(terminal_record)
                continue
            raw_path = (
                Path(arguments.outdir)
                / "frst_candidates"
                / f"h11_{h11:03d}"
                / f"np_{polytope_index:07d}"
                / f"frst_{retained_count + 1:07d}.h5"
            )
            metadata = build_raw_frst_metadata(
                arguments,
                h11,
                polytope_index,
                polytope,
                source,
                proposal_index,
                proposal_seed,
            )
            retained = write_raw_frst_artifact(
                raw_path,
                h11=h11,
                polytope_vertices=polytope_vertices,
                polytope_points=polytope_points,
                triangulation_labels=np.asarray(triangulation.labels, dtype=int),
                triangulation_points=np.asarray(triangulation.points(), dtype=int),
                simplices=simplices,
                simplex_indices=np.asarray(triangulation.simplices(as_indices=True), dtype=int),
                metadata=metadata,
            )
            retained["raw_frst_path"] = str(raw_path.resolve())
            retained.update(
                {
                    "terminal_status": "retained_raw_frst",
                    "terminal_reason": "FRST serialized before stage-2 processing",
                }
            )
            terminal_record = retained
            seen_triangulation_identities[triangulation_identity] = str(
                raw_path.resolve()
            )
            retained_count += 1
        except FileExistsError as exc:
            terminal_record.update(
                {"terminal_status": "output_collision", "terminal_reason": str(exc)}
            )
        except Exception as exc:
            terminal_record.update(
                {
                    "terminal_status": "invalid_frst",
                    "terminal_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        terminal_records.append(terminal_record)
        if retained_count >= raw_frst_target:
            break
    if retained_count < raw_frst_target:
        terminal_records.append(
            {
                "h11": h11,
                "polytope_index": polytope_index,
                "sampler": sampler_name,
                "terminal_status": "proposal_budget_exhausted",
                "terminal_reason": f"retained {retained_count} of requested {raw_frst_target} raw FRSTs",
                "attempted": attempted_count,
                "requested": raw_frst_target,
            }
        )
    return terminal_records


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect stage-1 raw FRST artifacts for a separate stage-2 run."
    )
    parser.add_argument("--outdir", required=True, help="Fresh stage-1 output directory.")
    parser.add_argument("--h11-plan", default="50:500,100:500,200:300,491:100")
    parser.add_argument("--polytope-manifest", default=None)
    parser.add_argument("--ks-database-version", default="CYTools fetch_polytopes endpoint (version not exposed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cgal", "qhull"), default="cgal")
    parser.add_argument("--retry-budget", type=int, default=500)
    parser.add_argument("--lower-proposal-budget", type=int, default=10)
    parser.add_argument("--h491-proposal-budget", type=int, default=100)
    parser.add_argument("--fine-tune-steps", type=int, default=8)
    parser.add_argument("--walk-step-size", type=float, default=1e-2)
    parser.add_argument("--max-steps-to-wall", type=int, default=25)
    parser.add_argument("--fast-height-scale", type=float, default=0.2)
    parser.add_argument("--ntfe-max-face-points", type=int, default=17)
    parser.add_argument("--ntfe-face-pool-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def current_source_commit():
    """Return the current repository commit when Git metadata is available."""
    repository_root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    arguments.plan = parse_raw_frst_plan(arguments.h11_plan)
    arguments.outdir = ensure_fresh_output_root(arguments.outdir)
    if arguments.retry_budget < 0 or arguments.lower_proposal_budget < 1 or arguments.h491_proposal_budget < 1:
        raise ValueError("proposal and retry budgets are invalid")
    arguments.sampling_scheme_by_h11 = {
        str(h11): "ntfe_fast" if h11 == 491 else "fast" for h11 in arguments.plan
    }
    arguments.proposal_budget_by_h11 = {
        str(h11): arguments.h491_proposal_budget if h11 == 491 else arguments.lower_proposal_budget
        for h11 in arguments.plan
    }
    polytope_manifest = (
        None
        if arguments.polytope_manifest is None
        else generator.load_polytope_manifest(arguments.polytope_manifest)
    )
    if not arguments.dry_run:
        generator.require_cytools_capabilities("ntfe_fast", "fast")
    started = time.perf_counter()
    candidate_records = []
    polytope_entries = []
    seen_triangulation_identities = {}
    for h11, raw_frst_target in arguments.plan.items():
        polytope_limit = POLYTOPE_COUNTS.get(
            h11, max(1, int(np.ceil(raw_frst_target / 10)))
        )
        source_polytopes = (
            load_source_polytopes(arguments, h11, polytope_limit, polytope_manifest)
            if not arguments.dry_run
            else []
        )
        raw_frst_targets = allocate_raw_frst_targets(
            h11, raw_frst_target, source_polytopes
        )
        for polytope_index, ((polytope, source), per_polytope_target) in enumerate(
            zip(source_polytopes, raw_frst_targets), start=1
        ):
            polytope_identifier, _ = generator.polytope_identity(polytope)
            polytope_entries.append(
                {
                    "h11": h11,
                    "polytope_index": polytope_index,
                    "polytope_id": polytope_identifier,
                    "vertices": np.asarray(polytope.vertices(), dtype=int).tolist(),
                    "source": source,
                    "target_raw_frsts": per_polytope_target,
                }
            )
            if not arguments.dry_run:
                candidate_records.extend(
                    collect_raw_frsts_for_polytope(
                        arguments,
                        h11,
                        polytope_index,
                        polytope,
                        source,
                        per_polytope_target,
                        seen_triangulation_identities,
                    )
                )
    retained_records = [
        record
        for record in candidate_records
        if record.get("terminal_status") == "retained_raw_frst"
    ]
    target_count = sum(arguments.plan.values())
    run_manifest = {
        "schema_version": RAW_FRST_SCHEMA_VERSION,
        "stage": "stage1_raw_frst_collection",
        "status": (
            "completed"
            if len(retained_records) == target_count
            else "raw_frst_target_shortfall"
        ),
        "output_root": arguments.outdir,
        "command_line": sys.argv if argv is None else [sys.argv[0], *argv],
        "source_commit": current_source_commit(),
        "python_version": platform.python_version(),
        "cytools_version": getattr(generator.cytools, "version", None),
        "hardware_threads": os.cpu_count(),
        "seed": arguments.seed,
        "approved_plan": arguments.plan,
        "target_raw_frst_count": target_count,
        "target_raw_frst_count_by_h11": arguments.plan,
        "retained_raw_frst_count": len(retained_records),
        "retained_raw_frst_count_by_h11": count_by_h11(
            retained_records, status_key="terminal_status"
        ),
        "terminal_status_count_by_h11": count_by_h11(
            candidate_records, status_key="terminal_status"
        ),
        "duplicate_full_triangulation_count": sum(
            record.get("terminal_status") == "duplicate_full_triangulation"
            for record in candidate_records
        ),
        "stage_boundary": "stage 1 writes raw FRSTs only; stage 2 is a separate run",
        "user_decisions": [
            {
                "stage": 1,
                "question": "May stage 2 consume only retained raw FRST files?",
                "answer_value": "confirmed",
                "answer": "yes",
            },
            {
                "stage": 1,
                "question": "How should a missing or corrupt raw FRST be handled?",
                "answer_value": "confirmed",
                "answer": "record it as unavailable; do not replace it in stage 2",
            },
            {
                "stage": 1,
                "question": "Where should the stage-2 input record live?",
                "answer_value": "confirmed",
                "answer": "in a separate JSONL input ledger",
            },
        ],
        "stage2_population_policy": {
            "replacement_frsts_in_stage2": False,
            "missing_raw_frst_status": "unavailable",
            "input_ledger": "separate JSONL artifact in the stage-2 output root",
        },
        "timing_seconds": time.perf_counter() - started,
    }
    atomic_jsonl_dump(Path(arguments.outdir) / "frst_terminal_statuses.jsonl", candidate_records)
    atomic_json_dump(Path(arguments.outdir) / "polytope_manifest.json", {"schema_version": RAW_FRST_SCHEMA_VERSION, "polytopes": polytope_entries})
    atomic_json_dump(Path(arguments.outdir) / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": run_manifest["status"],
                "retained": len(retained_records),
                "target": target_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    main()
