#!/usr/bin/env python3
"""Bind CYAX-0115 bounded failure and comparison evidence for review."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "validation/cyax_0115_uniform_frst"
CALIBRATION = BASE / "calibration"
SUPPORT = BASE / "exact_support_hard_capped"
MANIFEST = SUPPORT / "validation_manifest.json"
MANIFEST_SHA = "04254692b901a0b2a2d18d2bb21d7647af9aa48e0ce7d7173307a82ff6563263"
SCHEDULE_SHA = "a351d05ce60a5cb00f6298b567e045d8e6e18d9ab49055453919872f16feb92d"
HANDOFF_SHA = "bc130d684459c5f221d0e03f708ef48ca41044777d37ebecbe22c8745102ef68"


def load_validator():
    path = ROOT / "scripts/validate_uniform_frst_ensemble.py"
    spec = importlib.util.spec_from_file_location("cyax0115_validator_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validation driver: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(v, path):
    return v.file_sha256(path)


def git_blob(path: Path) -> str:
    return subprocess.run(
        ["git", "hash-object", str(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_chain_results(v, manifest):
    pairs = []
    path_stems = {
        (1, "00-paper-source"): "fixture1-control00-final",
        (1, "01-stricter"): "fixture1-control01",
        (2, "00-paper-source"): "fixture2-control00-final",
        (2, "01-stricter"): "fixture2-control01",
        (3, "00-paper-source"): "fixture3-control00",
        (3, "01-stricter"): "fixture3-control01",
    }
    mode_records = {}
    for fixture in manifest["fixtures"]:
        order = fixture["fixture_order"]
        for control in fixture["controls"]:
            control_id = control["control_id"]
            stem = path_stems[(order, control_id)]
            evidence = {}
            for mode in ("public", "hash-observed", "instrumented"):
                path = CALIBRATION / "equivalence" / f"{stem}-{mode}.json"
                result = json.loads(path.read_bytes())
                require(result["manifest_sha256"] == MANIFEST_SHA, "chain bound to other manifest")
                require(result["fixture_id"] == fixture["candidate_id"], "chain fixture changed")
                require(result["control_id"] == control_id, "chain control changed")
                expected_seed = control["fixed_restart_seeds"][0]
                require(result["seed"] == expected_seed, "chain used a non-frozen first seed")
                require(result["mode"] == mode, "chain evidence mode mismatch")
                require(result["termination_reason"] == "no_wall_found", "expected no-wall terminal outcome")
                require(result["captured_stdout"] == "Couldn't find wall.\n", "unexpected generator terminal text")
                require(result["yielded_unique_full_identities"] == [], "chain yielded an unexpected unique identity")
                if mode == "public":
                    require(result["raw_observation_count"] is None, "public mode must not claim passive raw count")
                else:
                    require(result["raw_observation_count"] == 0, "observer did not confirm zero raw candidates")
                    require(result["replacement_seed_count"] == 0, "observer reports a replacement seed")
                    require(result["survivor_conditioned_statistics_computed"] is False,
                            "survivor-conditioned statistics were computed")
                if mode == "instrumented":
                    require(len(result["terminal_events"]) == 1, "instrumented terminal event missing")
                    require(result["terminal_events"][0]["reason"] == "no_wall_found",
                            "instrumented terminal reason differs")
                    require(result["terminal_events"][0]["raw_observation_count"] == 0,
                            "instrumented terminal raw count differs")
                evidence[mode] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha(v, path),
                    "script_git_blob_sha1": result["script_git_blob_sha1"],
                    "process_id": result["process_id"],
                    "termination_reason": result["termination_reason"],
                    "raw_observation_count": result["raw_observation_count"],
                    "yielded_unique_count": len(result["yielded_unique_full_identities"]),
                    "step_update_count": len(result.get("step_updates", [])),
                    "terminal_event_count": len(result.get("terminal_events", [])),
                }
            require(len({evidence[m]["process_id"] for m in evidence}) == 3,
                    "public/hash/instrumented modes did not use distinct processes")
            scheduled = control["restart_count"]
            pairs.append(
                {
                    "fixture_order": order,
                    "fixture_id": fixture["candidate_id"],
                    "K": fixture["K"],
                    "control_id": control_id,
                    "seed": control["fixed_restart_seeds"][0],
                    "scheduled_restarts": scheduled,
                    "attempted_restarts": 1,
                    "unattempted_restarts": scheduled - 1,
                    "raw_indices_required": manifest["raw_indices"],
                    "raw_indices_observed": [],
                    "raw_uniform_marginal_gate": "FAIL_EARLY_TERMINATION_BEFORE_J1",
                    "unique_projection_k": control["unique_projection_k"],
                    "unique_projection_gate": "FAIL_EARLY_TERMINATION_BEFORE_K_FIRST_DISCOVERIES",
                    "public_hash_instrumented": evidence,
                    "replacement_seed_count": 0,
                    "survivor_conditioned_statistics_computed": False,
                }
            )
            mode_records[(order, control_id)] = evidence
    return pairs, mode_records


def wilson95(successes: int, total: int):
    """Descriptive interval for paired seed agreement, not a target-uniformity test."""
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_comparison_report(v, manifest):
    items = {}
    for fixture in manifest["fixtures"]:
        fixture_id = fixture["candidate_id"]
        expected_support = {
            state["full_triangulation_sha256"]
            for state in json.loads((SUPPORT / fixture["support_file"]).read_bytes())["states"]
        }
        for method in ("fast", "ntfe_fast"):
            result_path = CALIBRATION / "comparisons_per_proposal" / f"{fixture_id}-{method}.json"
            progress_path = result_path.with_suffix(".progress.jsonl")
            result = json.loads(result_path.read_bytes())
            events = [json.loads(line) for line in progress_path.read_text().splitlines()]
            starts = [event for event in events if event["event"] == "proposal_started"]
            yielded = [event for event in events if event["event"] == "comparison_candidate"]
            expected_seeds = [
                0x4A115000 + fixture["fixture_order"] * 1000 + i
                for i in range(1, 101)
            ]
            actual_seeds = [event["seed"] for event in starts]
            require(actual_seeds == expected_seeds, f"comparison seed list changed: {fixture_id}/{method}")
            require(len(yielded) == 100, f"comparison denominator is short: {fixture_id}/{method}")
            require(result["yielded_count"] == 100, f"comparison result count changed: {fixture_id}/{method}")
            require(result["no_yield_or_error_count"] == 0, f"comparison had failed seeds: {fixture_id}/{method}")
            require(all(event["full_triangulation_sha256"] in expected_support for event in yielded),
                    f"comparison emitted a state outside exact support: {fixture_id}/{method}")
            require(result["exact_support_member_count"] == 100, "comparison exact-support denominator changed")
            items[(fixture["fixture_order"], method)] = {
                "path": str(result_path.relative_to(ROOT)),
                "sha256": sha(v, result_path),
                "progress_path": str(progress_path.relative_to(ROOT)),
                "progress_sha256": sha(v, progress_path),
                "result": result,
                "events": yielded,
                "full_counts": Counter(event["full_triangulation_sha256"] for event in yielded),
                "two_face_counts": Counter(event["two_face_sha256"] for event in yielded),
            }

    comparison_rows = []
    for fixture in manifest["fixtures"]:
        order = fixture["fixture_order"]
        fast = items[(order, "fast")]
        ntfe = items[(order, "ntfe_fast")]
        fast_seed_map = {event["proposal_index"]: event["seed"] for event in fast["events"]}
        ntfe_seed_map = {event["proposal_index"]: event["seed"] for event in ntfe["events"]}
        require(fast_seed_map == ntfe_seed_map, "comparison routes do not use paired seeds")
        fast_full = [event["full_triangulation_sha256"] for event in fast["events"]]
        ntfe_full = [event["full_triangulation_sha256"] for event in ntfe["events"]]
        agreement = sum(a == b for a, b in zip(fast_full, ntfe_full))
        states = sorted(set(fast_full) | set(ntfe_full))
        tv = 0.5 * sum(
            abs(fast["full_counts"].get(state, 0) / 100 - ntfe["full_counts"].get(state, 0) / 100)
            for state in states
        )
        comparison_rows.append(
            {
                "fixture_id": fixture["candidate_id"],
                "K": fixture["K"],
                "proposal_seeds_paired": True,
                "proposal_count_per_route": 100,
                "fast": {
                    "attempted": 100,
                    "yielded": 100,
                    "no_yield_or_error": 0,
                    "exact_support_member_count": 100,
                    "unique_full_states": len(fast["full_counts"]),
                    "unique_two_face_states": len(fast["two_face_counts"]),
                    "full_state_frequencies": dict(sorted(fast["full_counts"].items())),
                    "two_face_state_frequencies": dict(sorted(fast["two_face_counts"].items())),
                    "result_path": fast["path"],
                    "result_sha256": fast["sha256"],
                    "progress_path": fast["progress_path"],
                    "progress_sha256": fast["progress_sha256"],
                },
                "ntfe_fast": {
                    "attempted": 100,
                    "yielded": 100,
                    "no_yield_or_error": 0,
                    "exact_support_member_count": 100,
                    "unique_full_states": len(ntfe["full_counts"]),
                    "unique_two_face_states": len(ntfe["two_face_counts"]),
                    "full_state_frequencies": dict(sorted(ntfe["full_counts"].items())),
                    "two_face_state_frequencies": dict(sorted(ntfe["two_face_counts"].items())),
                    "result_path": ntfe["path"],
                    "result_sha256": ntfe["sha256"],
                    "progress_path": ntfe["progress_path"],
                    "progress_sha256": ntfe["progress_sha256"],
                },
                "descriptive_paired_identity_agreement": {
                    "agreement_count": agreement,
                    "denominator": 100,
                    "wilson_95_interval": wilson95(agreement, 100),
                    "empirical_total_variation_between_methods": tv,
                    "interpretation": "descriptive seed-pair contrast; no fairness or population inference",
                },
                "fair_reference_uncertainty": {
                    "status": "UNDEFINED_NO_VALID_FAIR_RAW_SAMPLE",
                    "reason": "both fair controls terminate before raw index 1; their comparison denominator is 0/1 attempted restarts per cell",
                },
            }
        )
    return {
        "schema": "CYAX-0115-r4-biased-comparison-report-v1",
        "manifest_sha256": MANIFEST_SHA,
        "route_contract": {
            "methods": ["fast", "ntfe_fast"],
            "proposal_seed_rule": "0x4A115000 + fixture_order*1000 + one_based_proposal_index",
            "proposal_count_per_fixture_method": 100,
            "wall_cap_seconds_per_fixture_method": 300,
            "source_generation_helper_git_blob_sha1": "52774fc2cab228af29b82a3a0b96797c1500bed5",
            "interpretation": "explicitly biased comparisons only; do not transfer any population claim",
        },
        "paired_comparison_rows": comparison_rows,
        "uncertainty_boundary": "Descriptive paired-seed agreement has a Wilson interval. A fair-versus-biased uncertainty interval is undefined because all fair cells terminate before J=1; it is not zero and no test was run.",
        "status": "PASS_BOUNDED_BIASED_COMPARISONS_ONLY",
    }


def main():
    v = load_validator()
    manifest_bytes = MANIFEST.read_bytes()
    manifest_sha = v.sha256_bytes(manifest_bytes)
    require(manifest_sha == MANIFEST_SHA, "frozen manifest SHA changed")
    manifest = json.loads(manifest_bytes)
    schedule_receipt = json.loads((SUPPORT / "seed-schedule-identity.json").read_bytes())
    require(schedule_receipt["identity_sha256"] == SCHEDULE_SHA, "seed schedule identity changed")
    local_verification = CALIBRATION / "local-verification.json"
    require(local_verification.is_file(), "local invariant verification artifact missing")
    local_record = json.loads(local_verification.read_bytes())
    chain_rows, mode_records = load_chain_results(v, manifest)
    require(len(chain_rows) == 6, "not all frozen fixture/control gates were attempted")
    require(all(row["raw_uniform_marginal_gate"].startswith("FAIL_") for row in chain_rows),
            "expected every raw marginal gate to fail early")
    comparison_report = load_comparison_report(v, manifest)
    script_path = ROOT / "scripts/validate_uniform_frst_ensemble.py"
    code_blob = git_blob(script_path)
    summary = {
        "schema": "CYAX-0115-r4-bounded-failure-evidence-v1",
        "handoff_sha256": HANDOFF_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "registration_sha256": manifest["registration_sha256"],
        "seed_schedule_identity_sha256": SCHEDULE_SHA,
        "manifest_status": "FROZEN_BEFORE_SAMPLER_CALIBRATION_BYTES_UNCHANGED",
        "current_validation_driver_git_blob_sha1": code_blob,
        "installed_source_blobs": manifest["source_bindings"],
        "execution_environment": manifest["environment"],
        "overall_gate": "UNIFORMITY_NOT_ESTABLISHED",
        "failure_reason": "Every frozen fixture/control first restart terminates with `Couldn't find wall.` before any raw candidate; the r4 contract fails closed before the first required observation J=1.",
        "denominator_accounting": {
            "fixture_control_cells": 6,
            "scheduled_restarts_per_cell": 10000,
            "attempted_restarts_per_cell": 1,
            "unattempted_restarts_per_cell": 9999,
            "attempted_fixture_control_restart_slots": 6,
            "scheduled_fixture_control_restart_slots": 60000,
            "replacement_seed_count": 0,
            "raw_candidate_observations": 0,
            "unique_yields": 0,
            "condition_on_survivors": False,
        },
        "fixture_control_results": chain_rows,
        "statistical_gates": {
            "raw_multinomial_pearson": "GATED_NOT_RUN_NO_RAW_OBSERVATIONS",
            "tv_99_percent_null_envelopes": "GATED_NOT_RUN_NO_RAW_OBSERVATIONS",
            "max_frequency_99_percent_null_envelopes": "GATED_NOT_RUN_NO_RAW_OBSERVATIONS",
            "holm_familywise_adjustment": "UNDEFINED_NO_PREREGISTERED_TESTS_COULD_BE_RUN",
            "unique_projection_calibration": "GATED_NOT_RUN_NO_FIRST_DISCOVERIES",
            "test_statistics_and_p_values": "UNDEFINED_NOT_ZERO",
            "appendix_a_h11_11_sampling_benchmark": "GATED_NOT_RUN_BEFORE_REQUIRED_RAW_INDEX; paper source/control provenance was checked",
            "large_h11_diagnostics": "GATED_NOT_RUN_RAW_AND_EQUIVALENCE_GATES_FAILED",
            "worker_count_reversed_schedule_test": "GATED_NOT_RUN_STOP_AT_FIRST_FROZEN_SEED_FAILURE",
            "comparative_uncertainty_against_fair": "UNDEFINED_NO_VALID_FAIR_RAW_SAMPLE",
        },
        "biased_comparison_report": comparison_report,
        "open_policy_subcontract": {
            "status": "OWNER_DECISION_REQUIRED_OPEN_POLICY",
            "evidence": "Manager read-only search of Issue #115/current repository governance records found the fair/fast/open requirement but no owner-approved open-policy semantics.",
            "effect": "open-policy sub-contract only; does not alter the bounded fair/fast failure evidence",
        },
        "local_verification": {
            "status": local_record["overall_status"],
            "path": str(local_verification.relative_to(ROOT)),
            "sha256": sha(v, local_verification),
        },
        "superseded_records": [
            {
                "path": "validation/cyax_0115_uniform_frst/calibration/equivalence/fixture1-control00-instrumented.json",
                "disposition": "superseded; first AST check operated on a mutation-modified comparison tree and recorded no injected callbacks; final instrumented evidence uses the AST-verified copy and the same frozen seed",
            },
            {
                "path_prefix": "validation/cyax_0115_uniform_frst/calibration/comparisons/",
                "disposition": "single-stream pilot excluded from the final comparison report because the frozen formula binds a distinct seed to each one-based proposal index; the final per-proposal comparison batch applies that formula to all 100 proposal indices",
            },
        ],
        "production_boundary": {
            "production_ensemble_generated": False,
            "downstream_population_scan_run": False,
            "population_claim_made": False,
            "merge_or_publish_authorized": False,
        },
    }
    v.atomic_json_create(CALIBRATION / "comparison-report.json", comparison_report)
    v.atomic_json_create(CALIBRATION / "gate-and-comparison-summary.json", summary)
    print(json.dumps({
        "status": summary["overall_gate"],
        "chain_cells": len(chain_rows),
        "comparison_cells": 2 * len(comparison_report["paired_comparison_rows"]),
        "manifest_sha256": MANIFEST_SHA,
        "seed_schedule_identity_sha256": SCHEDULE_SHA,
        "validation_driver_git_blob_sha1": code_blob,
        "gate_summary_sha256": sha(v, CALIBRATION / "gate-and-comparison-summary.json"),
        "comparison_report_sha256": sha(v, CALIBRATION / "comparison-report.json"),
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
