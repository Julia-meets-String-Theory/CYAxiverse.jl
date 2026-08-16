"""Collect the fixed stage-1 raw FRST population.

This command deliberately stops after serializing retained FRST data and an
optional stage-2-independent topology cache.  It does not choose an
orientifold, search the Kähler cone, assign visible-sector divisors, or write
EFT rows.  Run ``generate_stage2_eft_reference.py`` separately on its output.
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
    TOPOLOGY_CACHE_CONVENTIONS,
    TOPOLOGY_CACHE_SCHEMA_VERSION,
    build_raw_frst_geometry_id,
    count_by_h11,
    stable_hash,
    compute_triangulation_hash,
    write_raw_frst_artifact,
)
from glimmers_schema11 import atomic_json_dump, atomic_jsonl_dump, ensure_fresh_output_root


APPROVED_PLAN = {50: 500, 100: 500, 200: 300, 491: 100}
POLYTOPE_COUNTS = {50: 50, 100: 50, 200: 30, 491: 1}
FRSTS_PER_POLYTOPE = {50: 10, 100: 10, 200: 10, 491: 100}
H491_CANONICAL_TIP_PREFLIGHT_SCHEMA_VERSION = (
    "cyaxiverse-h491-canonical-tip-preflight-1.0"
)


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


def run_h491_canonical_tip_preflight(calabi_yau):
    """Annotate an h11=491 FRST with a CYTools canonical-tip diagnostic.

    Keep this check observational: a valid FRST remains the Stage-1 retention
    unit even when the canonical stretched-cone tip has negative or subunit
    divisor volumes.  Stage 2 remains responsible for physical point
    selection and filtering.
    """
    result = {
        "schema_version": H491_CANONICAL_TIP_PREFLIGHT_SCHEMA_VERSION,
        "h11": 491,
        "method": "tip_of_stretched_cone(1.0)",
        "tip_parameter": 1.0,
        "cytools_version": getattr(generator.cytools, "version", None),
        "status": "error",
        "classification": "canonical_tip_preflight_error",
        "solver": None,
        "solver_policy": "mosek_if_licensed_then_cytools_default",
        "mosek_configured": None,
        "mosek_activated": None,
        "mosek_fallback_reason": None,
        "divisor_volume_lower_bounds_enforced": False,
        "basis_selection_matrix": None,
        "prime_charge_matrix": None,
        "intersection_reconstruction": {
            "status": "omitted",
            "authoritative": False,
            "method": "CalabiYau.intersection_numbers(in_basis=True, format='coo')",
            "reason": (
                "Omitted from the h11=491 preflight gate because CYTools warns "
                "that high-h11 intersection reconstruction is unreliable."
            ),
        },
        "diagnostic": None,
    }
    try:
        if int(calabi_yau.h11()) != 491:
            result.update(
                {
                    "status": "not_applicable",
                    "classification": "not_h11_491",
                }
            )
            return result

        kahler_cone = calabi_yau.toric_kahler_cone()
        license_status = generator.configure_mosek_license()
        result["mosek_configured"] = bool(license_status.get("configured", False))
        result["mosek_activated"] = bool(license_status.get("activated", False))
        if result["mosek_activated"]:
            try:
                tip = np.asarray(
                    kahler_cone.tip_of_stretched_cone(1.0, backend="mosek"),
                    dtype=float,
                )
                result["solver"] = "mosek"
            except Exception as exc:
                result["mosek_fallback_reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                tip = np.asarray(
                    kahler_cone.tip_of_stretched_cone(1.0), dtype=float
                )
                result["solver"] = "cytools-default-after-mosek-failure"
        else:
            tip = np.asarray(kahler_cone.tip_of_stretched_cone(1.0), dtype=float)
            result["solver"] = "cytools-default"

        hyperplanes = np.asarray(kahler_cone.hyperplanes(), dtype=float)
        effective_cone_rays = np.asarray(
            calabi_yau.toric_effective_cone().rays(), dtype=float
        )
        result["kahler_hyperplanes_shape"] = list(hyperplanes.shape)
        result["effective_cone_rays_shape"] = list(effective_cone_rays.shape)
        if (
            hyperplanes.ndim != 2
            or hyperplanes.shape[1] != 491
            or effective_cone_rays.ndim != 2
            or effective_cone_rays.shape[1] != 491
        ):
            raise ValueError(
                "unexpected h11=491 cone shape: "
                f"Kähler={hyperplanes.shape}, effective={effective_cone_rays.shape}"
            )

        residual_checks = {
            "status": "unavailable",
            "authoritative": True,
            "absolute_tolerance": 1e-8,
            "relative_tolerance": 1e-10,
            "prime_from_glsm_charge_matrix": None,
            "effective_from_basis": None,
        }
        try:
            basis_selection_matrix_raw = np.asarray(
                calabi_yau.divisor_basis(as_matrix=True)
            )
            basis_selection_matrix = np.asarray(
                basis_selection_matrix_raw, dtype=float
            )
            prime_charge_matrix_raw = np.asarray(
                calabi_yau.polytope().glsm_charge_matrix(include_origin=False)
            )
            prime_charge_matrix = np.asarray(prime_charge_matrix_raw, dtype=float)
            prime_labels = np.asarray(
                calabi_yau.prime_toric_divisors(), dtype=int
            ).reshape(-1)
            basis_volumes = np.asarray(
                calabi_yau.compute_divisor_volumes(tip, in_basis=True),
                dtype=float,
            ).reshape(-1)
            prime_volumes = np.asarray(
                calabi_yau.compute_divisor_volumes(tip), dtype=float
            ).reshape(-1)
            if basis_selection_matrix.ndim != 2:
                raise ValueError(
                    "CYTools divisor_basis(as_matrix=True) must be two-dimensional"
                )
            if prime_charge_matrix.ndim != 2:
                raise ValueError(
                    "CYTools GLSM charge matrix must be two-dimensional"
                )
            if basis_selection_matrix.shape[0] != basis_volumes.size:
                raise ValueError(
                    "basis-selection matrix row count does not match basis volumes"
                )
            if prime_charge_matrix.shape[0] != basis_volumes.size:
                raise ValueError(
                    "GLSM charge matrix row count does not match basis volumes"
                )
            if prime_charge_matrix.shape[1] != prime_volumes.size:
                raise ValueError(
                    "GLSM charge matrix column count does not match prime volumes"
                )
            prime_from_glsm_charge_matrix = prime_charge_matrix.T @ basis_volumes
            effective_from_basis = effective_cone_rays @ basis_volumes

            def residual_entry(reference, observed):
                reference = np.asarray(reference, dtype=float).reshape(-1)
                observed = np.asarray(observed, dtype=float).reshape(-1)
                if reference.shape != observed.shape:
                    raise ValueError(
                        "residual arrays have different shapes: "
                        f"{reference.shape} versus {observed.shape}"
                    )
                difference = np.asarray(reference) - np.asarray(observed)
                allowed = 1e-8 + 1e-10 * np.maximum(
                    np.abs(reference), np.abs(observed)
                )
                normalized = np.divide(
                    np.abs(difference),
                    allowed,
                    out=np.zeros_like(difference, dtype=float),
                    where=allowed > 0.0,
                )
                maximum = (
                    float(np.max(np.abs(difference)))
                    if difference.size
                    else 0.0
                )
                maximum_normalized = (
                    float(np.max(normalized)) if normalized.size else 0.0
                )
                return {
                    "max_abs_residual": maximum,
                    "max_normalized_residual": maximum_normalized,
                    "finite": bool(np.all(np.isfinite(difference))),
                    "passed": bool(
                        np.all(np.isfinite(difference))
                        and maximum_normalized <= 1.0
                    ),
                }

            result["basis_selection_matrix"] = {
                "shape": list(basis_selection_matrix_raw.shape),
                "dtype": str(basis_selection_matrix_raw.dtype),
                "sha256": stable_hash(basis_selection_matrix_raw.tolist()),
                "role": (
                    "CYTools divisor_basis(as_matrix=True) basis selector; "
                    "not a prime charge matrix"
                ),
            }
            result["prime_charge_matrix"] = {
                "shape": list(prime_charge_matrix_raw.shape),
                "dtype": str(prime_charge_matrix_raw.dtype),
                "sha256": stable_hash(prime_charge_matrix_raw.tolist()),
                "role": (
                    "CYTools polytope().glsm_charge_matrix(include_origin=False); "
                    "prime volumes are Q.T @ tau_basis"
                ),
            }
            residual_checks.update(
                {
                    "status": "complete",
                    "prime_from_glsm_charge_matrix": residual_entry(
                        prime_volumes, prime_from_glsm_charge_matrix
                    ),
                    "effective_from_basis": residual_entry(
                        effective_from_basis, effective_cone_rays @ basis_volumes
                    ),
                    "basis_selection_matrix_shape": list(
                        basis_selection_matrix.shape
                    ),
                    "prime_charge_matrix_shape": list(prime_charge_matrix.shape),
                    "prime_toric_divisors_shape": list(prime_labels.shape),
                }
            )
        except Exception as exc:
            residual_checks["reason"] = f"{type(exc).__name__}: {exc}"
        result["residual_checks"] = residual_checks

        diagnostic, _ = generator.evaluate_kaehler_point(
            calabi_yau,
            kahler_cone,
            effective_cone_rays,
            tip,
            attempt_index=1,
            point_kind="canonical_tip",
            solver=result["solver"],
            min_prime_divisor_volume=1.0,
            min_divisor_volume=1.0,
            enforce_divisor_volume_lower_bounds=False,
        )
        result["diagnostic"] = generator._jsonable(diagnostic)
        checks = diagnostic.get("checks", {})
        residual_status = result.get("residual_checks", {}).get("status")
        residual_checks_passed = (
            residual_status == "complete"
            and all(
                result["residual_checks"].get(name, {}).get("passed") is True
                for name in (
                    "prime_from_glsm_charge_matrix",
                    "effective_from_basis",
                )
            )
        )
        divisor_checks = {
            "finite_basis_divisor_volumes",
            "positive_basis_divisor_volumes",
            "finite_prime_divisor_volumes",
            "positive_prime_divisor_volumes",
            "finite_effective_divisor_volumes",
            "positive_effective_divisor_volumes",
            "prime_divisor_volume_lower_bound",
            "effective_divisor_volume_lower_bound",
        }
        core_checks = {
            name: passed
            for name, passed in checks.items()
            if name not in divisor_checks
        }
        divisor_shortfall = any(
            checks.get(name) is False for name in divisor_checks
        )
        if residual_status == "complete" and not residual_checks_passed:
            result["status"] = "evaluation_failed"
            result["classification"] = "topology_validation_failure"
            result["failure_stage"] = "basis_convention"
        elif divisor_shortfall and all(value is True for value in core_checks.values()):
            minima = [
                diagnostic.get("minimum_basis_divisor_volume"),
                diagnostic.get("minimum_prime_divisor_volume"),
                diagnostic.get("minimum_effective_divisor_volume"),
            ]
            finite_minima = [
                value
                for value in minima
                if isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
            ]
            result["shortfall_kind"] = (
                "negative_divisor_volume"
                if any(value < 0.0 for value in finite_minima)
                else "subunit_divisor_volume"
            )
            result["status"] = "annotated_divisor_volume_shortfall"
            result["classification"] = "canonical_tip_divisor_volume_shortfall"
        elif diagnostic.get("point_status") == "accepted":
            result["status"] = "passed"
            result["classification"] = "passed"
        else:
            result["status"] = "evaluation_failed"
            result["classification"] = "canonical_tip_preflight_failure"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


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
            topology_cache = None
            topology_cache_metadata = None
            raw_geometry_id = build_raw_frst_geometry_id(
                h11, polytope_identifier, full_hash
            )
            calabi_yau = None
            try:
                # Reuse the CYTools triangulation already held by the sampler.
                # This work is deliberately best-effort: retaining the raw
                # FRST remains the stage-1 accounting unit if a topology
                # extraction or serialization step fails.
                calabi_yau = triangulation.get_cy()
                topology_cache = generator.extract_topology(
                    calabi_yau, triangulation, export_kahler_rays=False
                )
                topology_cache_metadata = {
                    "schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
                    "h11": int(topology_cache["h11"]),
                    "h21": int(topology_cache["h21"]),
                    "geometry_id": raw_geometry_id,
                    "raw_geometry_id": raw_geometry_id,
                    "polytope_id": polytope_identifier,
                    "full_triangulation_hash": full_hash,
                    "cytools_version": getattr(generator.cytools, "version", None),
                    "backend": arguments.backend,
                    "conventions": TOPOLOGY_CACHE_CONVENTIONS,
                    "kahler_rays_exported": False,
                }
                metadata.update(
                    {
                        "topology_cache_status": "computed",
                        "topology_cache_reason": "computed from the held CYTools triangulation",
                    }
                )
            except Exception as cache_error:
                metadata.update(
                    {
                        "topology_cache_status": "compute_failed",
                        "topology_cache_reason": (
                            f"{type(cache_error).__name__}: {cache_error}"
                        ),
                    }
                )
            if h11 == 491:
                if calabi_yau is None:
                    canonical_tip_preflight = {
                        "schema_version": H491_CANONICAL_TIP_PREFLIGHT_SCHEMA_VERSION,
                        "h11": 491,
                        "method": "tip_of_stretched_cone(1.0)",
                        "tip_parameter": 1.0,
                        "cytools_version": getattr(generator.cytools, "version", None),
                        "status": "unavailable",
                        "classification": "canonical_tip_preflight_unavailable",
                        "reason": "CYTools Calabi-Yau construction failed before preflight",
                        "divisor_volume_lower_bounds_enforced": False,
                    }
                else:
                    canonical_tip_preflight = run_h491_canonical_tip_preflight(
                        calabi_yau
                    )
                metadata["canonical_tip_preflight"] = canonical_tip_preflight
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
                topology_cache=topology_cache,
                topology_cache_metadata=topology_cache_metadata,
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


def summarize_nested_preflight_classifications(records):
    """Count nested h11=491 preflight classifications without new statuses."""
    counts = {}
    for record in records:
        preflight = record.get("canonical_tip_preflight") or {}
        if int(record.get("h11", -1)) != 491 or not preflight:
            continue
        classification = preflight.get("classification", "unknown")
        h11_counts = counts.setdefault("491", {})
        h11_counts[classification] = h11_counts.get(classification, 0) + 1
    return {
        h11: dict(sorted(classifications.items()))
        for h11, classifications in sorted(counts.items())
    }


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
        "canonical_tip_preflight_classification_count_by_h11": summarize_nested_preflight_classifications(
            retained_records
        ),
        "terminal_status_count_by_h11": count_by_h11(
            candidate_records, status_key="terminal_status"
        ),
        "duplicate_full_triangulation_count": sum(
            record.get("terminal_status") == "duplicate_full_triangulation"
            for record in candidate_records
        ),
        "stage_boundary": (
            "stage 1 writes raw FRSTs plus an optional stage-2-independent "
            "topology cache; stage 2 is a separate run"
        ),
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
