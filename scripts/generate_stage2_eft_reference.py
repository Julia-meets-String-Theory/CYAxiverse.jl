"""Run stage 2 on a fixed collection of retained raw FRST artifacts.

Read only the raw FRST files emitted by
``generate_stage1_raw_frsts.py``.  Reconstruct each FRST, apply the existing
CYTools topology, orientifold, Kähler, divisor, visible-sector, and potential
checks, and optionally build the compact EFT-reference table.  This command
never samples a replacement FRST.
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

import generate_geometric_data_multitriangulation as generator
from glimmers_raw_frst import (
    RawFRSTError,
    TOPOLOGY_CACHE_CONVENTIONS,
    TOPOLOGY_CACHE_SCHEMA_VERSION,
    build_input_ledger,
    compute_triangulation_hash,
    count_by_h11,
    read_raw_frst_artifact,
    validate_topology_cache,
)
from glimmers_schema11 import (
    MAXIMUM_EFT_ROWS,
    MINIMUM_EFT_ROWS,
    QCD_VOLUME_TARGET,
    QED_VOLUME_MAX,
    SCHEMA_VERSION as SCHEMA_1_1_VERSION,
    TARGET_GEOMETRY_COUNT,
    append_jsonl_record,
    atomic_json_dump,
    atomic_jsonl_dump,
    ensure_fresh_output_root,
    estimate_storage,
    reconcile_eft_capacity,
    summarize_terminal_records,
    stable_hash,
    stable_seed,
    write_eft_parquet,
)


def build_parser():
    """Build the stage-2 command-line parser."""
    parser = argparse.ArgumentParser(
        description="Process retained raw FRSTs in an independent stage-2 run."
    )
    parser.add_argument("--stage1-root", required=True, help="Completed stage-1 output root.")
    parser.add_argument("--outdir", required=True, help="Fresh stage-2 output root.")
    parser.add_argument("--backend", choices=("cgal", "qhull"), default="cgal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--orientifold-file", default=None)
    parser.add_argument(
        "--moduli-policy",
        choices=("adaptive", "canonical_qcd"),
        default="canonical_qcd",
        help=(
            "Use the deterministic Glimmers-style tip and QCD-volume-40 "
            "dilation, or select adaptive only for an explicitly labelled "
            "randomized Kähler diagnostic."
        ),
    )
    parser.add_argument(
        "--allow-m-below-one",
        action="store_true",
        help=(
            "Allow canonical_qcd to contract the canonical tip (m<1); "
            "disabled by default and recorded in the run metadata."
        ),
    )
    parser.add_argument("--visible-sector-policy", choices=("none", "intersecting_d7"), default="intersecting_d7")
    parser.add_argument(
        "--orientifold-kaehler-policy",
        choices=("none", "require_even_subspace"),
        default="none",
        help=(
            "Require the orientifold-even Kaehler subspace to intersect the "
            "stretched cone, or record that this check is not required."
        ),
    )
    parser.add_argument("--max-m", type=float, default=1_000_000.0)
    parser.add_argument("--max-kaehler-attempts", type=int, default=100)
    parser.add_argument("--min-divisor-volume", type=float, default=1.0)
    parser.add_argument("--min-prime-divisor-volume", type=float, default=1.0)
    parser.add_argument("--qcd-volume-target", type=float, default=QCD_VOLUME_TARGET)
    parser.add_argument("--qcd-divisor-index", type=int, default=None)
    parser.add_argument("--qed-volume-max", type=float, default=QED_VOLUME_MAX)
    parser.add_argument(
        "--export-kaehler-rays",
        "--export-kahler-rays",
        dest="export_kaehler_rays",
        action="store_true",
        help=(
            "Enumerate and store Kaehler-cone rays; effective-cone rays for Q "
            "are always retained."
        ),
    )
    parser.add_argument("--materialize-dense-potential", action="store_true")
    parser.add_argument("--eft", action="store_true", help="Build compact EFT-reference rows after geometry acceptance.")
    parser.add_argument("--eft-minimum-rows", type=int, default=MINIMUM_EFT_ROWS)
    parser.add_argument("--eft-maximum-rows", type=int, default=MAXIMUM_EFT_ROWS)
    parser.add_argument("--eft-output-path", default=None)
    parser.add_argument(
        "--allow-overwrite-existing-geometry",
        action="store_true",
        help=(
            "Explicitly authorize replacement of an existing cyax.h5 artifact; "
            "disabled by default and recorded in geometry provenance."
        ),
    )
    parser.add_argument("--ks-database-version", default="inherited from stage-1 raw FRST records")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_topology_audit_record(raw_frst_record, backend):
    """Create the compact structural audit record for one raw FRST."""
    return {
        "audit_schema_version": "cyaxiverse-stage2-topology-audit-1.0",
        "audit_status": "not_run",
        "reconstruction_identity_status": "not_run",
        "topology_validation_status": "not_run",
        "stage2_terminal_status": None,
        "h11": raw_frst_record.get("h11"),
        "polytope_id": raw_frst_record.get("polytope_id"),
        "geometry_id": raw_frst_record.get("geometry_id"),
        "raw_frst_path": raw_frst_record.get("raw_frst_path"),
        "backend": backend,
        "cytools_version": getattr(generator.cytools, "version", None),
        "topology_cache_status": "not_run",
        "topology_cache_reason": None,
        "ledger_full_triangulation_hash": raw_frst_record.get(
            "full_triangulation_hash"
        ),
    }


def orientifold_audit_record(orientifold_config):
    """Record orientifold scope without inferring downstream physics."""
    return {
        "requested": bool(orientifold_config.get("requested", False)),
        "input_status": orientifold_config.get("status"),
        "involution_type": orientifold_config.get("involution_type"),
        "h11_parity_policy": "record_only_not_enforced",
        "fixed_locus_validation": "not_performed",
        "tadpole_validation": "not_performed",
        "physical_orientifold_claim": "not_made",
    }


def reconstruct_raw_frst(
    raw_frst_record, backend, topology_audit, *, export_kahler_rays=False
):
    """Reconstruct one CYTools polytope and triangulation from persisted arrays."""
    persisted = read_raw_frst_artifact(raw_frst_record["raw_frst_path"])
    topology_audit.update(
        {
            "raw_frst_file_sha256": persisted.get("raw_frst_file_sha256"),
            "raw_dataset_full_triangulation_hash": persisted.get(
                "full_triangulation_hash"
            ),
            "raw_frst_schema_version": persisted.get("raw_frst_schema_version"),
            "polytope_point_count": int(
                np.asarray(persisted["arrays"]["polytope_points"]).shape[0]
            ),
            "polytope_vertex_count": int(
                np.asarray(persisted["arrays"]["polytope_vertices"]).shape[0]
            ),
            "raw_triangulation_label_shape": list(
                np.asarray(persisted["arrays"]["triangulation_labels"]).shape
            ),
            "raw_triangulation_simplex_shape": list(
                np.asarray(persisted["arrays"]["simplices"]).shape
            ),
        }
    )
    arrays = persisted["arrays"]
    polytope = generator.Polytope(
        arrays["polytope_vertices"], deterministic_glsm_basis=True
    )
    # Legacy raw-FRST artifacts carry vertices but only a placeholder point
    # table.  CYTools prime-divisor labels index the reconstructed full
    # lattice-point table, so use the authoritative points from this
    # reconstruction for all downstream geometry writes and label mapping.
    reconstructed_polytope_points = np.asarray(polytope.points(), dtype=int)
    persisted["arrays"]["polytope_points"] = reconstructed_polytope_points
    topology_audit["polytope_point_count"] = int(
        reconstructed_polytope_points.shape[0]
    )
    triangulation = polytope.triangulate(
        points=np.asarray(arrays["triangulation_labels"], dtype=int).tolist(),
        simplices=np.asarray(arrays["simplices"], dtype=int).tolist(),
        make_star=False,
        check_input_simplices=True,
        backend=backend,
        verbosity=0,
    )
    topology_audit["polytope_dimension"] = int(polytope.dim())
    topology_audit["polytope_ambient_dimension"] = int(polytope.ambient_dim())
    topology_audit["polytope_reflexive"] = bool(polytope.is_reflexive())
    topology_audit["frst_validation"] = generator.validate_frst(
        polytope, triangulation
    )
    topology_audit["reconstructed_simplex_shape"] = list(
        np.asarray(triangulation.simplices(), dtype=int).shape
    )
    topology_audit["reconstructed_triangulation_hash"] = compute_triangulation_hash(
        np.asarray(triangulation.simplices(), dtype=int)
    )
    if persisted["full_triangulation_hash"] != raw_frst_record["full_triangulation_hash"]:
        raise RawFRSTError(
            "input_identity_mismatch",
            "reconstructed triangulation hash differs from the stage-1 ledger",
            record={
                "ledger_full_triangulation_hash": raw_frst_record[
                    "full_triangulation_hash"
                ],
                "raw_dataset_full_triangulation_hash": persisted[
                    "full_triangulation_hash"
                ],
            },
        )
    if (
        topology_audit["reconstructed_triangulation_hash"]
        != persisted["full_triangulation_hash"]
    ):
        raise RawFRSTError(
            "input_identity_mismatch",
            "CYTools reconstructed a triangulation with a different identity",
            record={
                "raw_dataset_full_triangulation_hash": persisted[
                    "full_triangulation_hash"
                ],
                "reconstructed_triangulation_hash": topology_audit[
                    "reconstructed_triangulation_hash"
                ],
            },
        )
    topology_audit["reconstruction_identity_status"] = "passed"
    expected_cache_identity = {
        "schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
        "h11": int(persisted["h11"]),
        "geometry_id": persisted.get("geometry_id"),
        "raw_geometry_id": persisted.get("geometry_id"),
        "polytope_id": persisted.get("polytope_id"),
        "full_triangulation_hash": persisted.get("full_triangulation_hash"),
        "cytools_version": getattr(generator.cytools, "version", None),
        "backend": backend,
        "conventions": TOPOLOGY_CACHE_CONVENTIONS,
        "kahler_rays_exported": bool(export_kahler_rays),
    }
    cache_info = persisted.get("topology_cache")
    if cache_info is None:
        topology = None
        cache_reason = persisted.get(
            "topology_cache_reason", "raw FRST has no topology cache group"
        )
    else:
        topology, cache_reason = validate_topology_cache(
            cache_info, expected_cache_identity
        )
    if topology is None:
        topology_audit["topology_cache_status"] = "fallback_recompute"
        topology_audit["topology_cache_reason"] = cache_reason
    else:
        topology_audit["topology_cache_status"] = "hit"
        topology_audit["topology_cache_reason"] = cache_reason
        persisted["topology_override"] = topology
    return persisted, polytope, triangulation


def build_geometry_output_path(output_root, raw_frst_record):
    """Map one raw identity to a deterministic stage-2 HDF5 path."""
    return (
        Path(output_root)
        / f"h11_{int(raw_frst_record['h11']):03d}"
        / f"np_{int(raw_frst_record.get('polytope_index', 0)):07d}"
        / f"cy_{int(raw_frst_record.get('candidate_index', 0)):07d}"
        / "cyax.h5"
    )


def classify_stage2_failure(error):
    """Map implementation failures onto the fixed stage-2 status vocabulary."""
    if isinstance(error, RawFRSTError):
        return error.terminal_status
    if isinstance(error, generator.OrientifoldValidationFailure):
        return "orientifold_invariance_failure"
    if isinstance(error, generator.QEDAssignmentFailure):
        if error.category == "no_eligible_qed_divisor":
            return "no_eligible_intersecting_qed_pair"
        if error.category == "qed_volume_rejection":
            return "volume_filter_rejection"
        if error.category == "orientifold_invariance_failure":
            return "orientifold_invariance_failure"
        if error.category == "qcd_normalization_failure":
            return "qcd_normalization_failure"
        if error.category == "assignment_pool_shortfall":
            return "assignment_pool_shortfall"
        if error.category == "output_collision":
            return "output_collision"
        if error.category == "potential_term_mismatch":
            return "potential_term_mismatch"
        return "numerical_geometry_failure"
    status = generator._candidate_terminal_status(error)
    return {
        "no_eligible_intersecting_qed_pair": "no_eligible_intersecting_qed_pair",
        "divisor_volume_filter_rejection": "volume_filter_rejection",
        "topology_or_cone_error": "topology_validation_failure",
        "io_failure": "storage_failure",
        "no_eligible_qed_divisor": "no_eligible_intersecting_qed_pair",
        "invalid_charge_basis_mapping": "numerical_geometry_failure",
        "intersection_failure": "topology_validation_failure",
        "qed_volume_rejection": "volume_filter_rejection",
        "kaehler_tip_failure": "kaehler_tip_failure",
        "kaehler_point_shortfall": "kaehler_point_shortfall",
        "qcd_normalization_failure": "qcd_normalization_failure",
        "output_collision": "output_collision",
        "numerical_geometry_failure": "numerical_geometry_failure",
    }.get(status, "numerical_geometry_failure")


def _stage2_identity(record):
    """Return the stable raw-FRST identity fields used in progress records."""
    return {
        key: record.get(key)
        for key in (
            "h11",
            "polytope_id",
            "polytope_index",
            "candidate_index",
            "geometry_id",
            "full_triangulation_hash",
            "raw_frst_path",
        )
    }


def process_raw_frst_artifact(
    arguments,
    raw_frst_record,
    orientifold_config,
    output_root,
    progress_callback=None,
):
    """Process one retained raw FRST and return terminal and audit records."""
    candidate_started = time.monotonic()
    stage_timings = []
    active_stage = None
    active_stage_started = None

    def start_stage(stage):
        """Record a stage start before entering a potentially slow operation."""
        nonlocal active_stage, active_stage_started
        finish_stage()
        active_stage = str(stage)
        active_stage_started = time.monotonic()
        if progress_callback is not None:
            progress_callback(
                {
                    **_stage2_identity(raw_frst_record),
                    "event": "candidate_stage_started",
                    "stage": active_stage,
                    "elapsed_seconds": time.monotonic() - candidate_started,
                }
            )

    def finish_stage():
        """Close the current stage interval when the next stage begins."""
        nonlocal active_stage, active_stage_started
        if active_stage is None:
            return
        stage_timings.append(
            {
                "stage": active_stage,
                "elapsed_seconds": time.monotonic() - active_stage_started,
            }
        )
        active_stage = None
        active_stage_started = None

    topology_audit = build_topology_audit_record(raw_frst_record, arguments.backend)
    point_diagnostics = []
    assignment_pool_rejection_records = []
    topology_audit["kaehler_point_scan"] = {
        "policy": arguments.moduli_policy,
        "allow_m_below_one": bool(arguments.allow_m_below_one),
        "qcd_volume_target": generator.QCD_VOLUME_TARGET,
        "qcd_volume_tolerance": generator.QCD_VOLUME_TOLERANCE,
        "divisor_volume_tolerance": generator.DIVISOR_VOLUME_TOLERANCE,
        "normalization_failure_status": "qcd_normalization_failure",
        "normalization_repair_policy": "none",
        "selection_policy": (
            "explicit_qcd_divisor_index_or_first_eligible_ascending_index"
        ),
        "post_selection_fallback": "none",
        "attempt_budget": arguments.max_kaehler_attempts,
        "point_status": "not_run",
        "diagnostics": point_diagnostics,
    }
    topology_audit["orientifold_validation"] = orientifold_audit_record(
        orientifold_config
    )
    terminal_record = {
        key: raw_frst_record.get(key)
        for key in (
            "h11",
            "polytope_id",
            "raw_frst_path",
            "full_triangulation_hash",
            "geometry_id",
            "polytope_index",
            "candidate_index",
        )
    }
    terminal_record.update(
        {
            "artifact_status": (
                generator.POOL_PENDING_ARTIFACT_STATUS
                if arguments.eft
                else generator.GEOMETRY_ONLY_ARTIFACT_STATUS
            ),
            "artifact_written": False,
            "allow_overwrite_existing_geometry": bool(
                arguments.allow_overwrite_existing_geometry
            ),
        }
    )
    output_path = build_geometry_output_path(output_root, raw_frst_record)
    terminal_record["output_path"] = str(output_path.resolve())
    existing_artifact = generator.inspect_geometry_artifact(output_path)
    if existing_artifact["exists"]:
        terminal_record["existing_artifact_audit"] = existing_artifact
    try:
        start_stage("reconstruct_raw_frst")
        persisted, polytope, triangulation = reconstruct_raw_frst(
            raw_frst_record,
            arguments.backend,
            topology_audit,
            export_kahler_rays=arguments.export_kaehler_rays,
        )
        finish_stage()
        start_stage("construct_calabi_yau")
        calabi_yau = triangulation.get_cy()
        finish_stage()
        sampling_metadata = dict(
            persisted.get("sampler_metadata")
            or persisted.get("sampling_metadata")
            or {}
        )
        sampling_metadata.setdefault(
            "scheme", persisted.get("sampler", "raw_frst_reconstruction")
        )
        sampling_metadata.update(
            {
                "stage1_raw_frst_path": persisted["raw_frst_path"],
                "stage1_full_triangulation_hash": persisted["full_triangulation_hash"],
                "stage2_sampler": "none_raw_frst_reconstruction",
            }
        )
        kaehler_point_seed = stable_seed(
            "stage2-kaehler-point",
            arguments.seed,
            persisted["geometry_id"],
        )
        sampling_metadata.update(
            {
                "stage2_kaehler_point_seed": kaehler_point_seed,
                "stage2_kaehler_point_attempt_budget": arguments.max_kaehler_attempts,
            }
        )
        topology_audit["kaehler_point_scan"].update(
            {
                "point_seed": kaehler_point_seed,
                "point_status": "running",
            }
        )

        def report(message):
            start_stage(message)
            if arguments.verbose:
                print(
                    f"{persisted['geometry_id']}: {message}", flush=True
                )

        generator.generate_and_save_geometry(
            int(persisted["h11"]),
            calabi_yau,
            np.asarray(persisted["arrays"]["polytope_points"], dtype=int),
            np.asarray(persisted["arrays"]["simplices"], dtype=int),
            str(output_path),
            arguments.max_m,
            arguments.max_kaehler_attempts,
            arguments.min_divisor_volume,
            arguments.min_prime_divisor_volume,
            25.0,
            40.0,
            arguments.moduli_policy,
            arguments.qcd_volume_target,
            arguments.qcd_divisor_index,
            arguments.visible_sector_policy,
            None,
            np.random.default_rng(arguments.seed + int(persisted["h11"])),
            report,
            poly=polytope,
            triangulation=triangulation,
            polytope_id=persisted["polytope_id"],
            sampling_metadata=sampling_metadata,
            ks_database_version=arguments.ks_database_version,
            orientifold_config=orientifold_config,
            orientifold_kaehler_policy=arguments.orientifold_kaehler_policy,
            export_kahler_rays=arguments.export_kaehler_rays,
            qed_selection_policy="uniform_eligible",
            qed_volume_max=arguments.qed_volume_max,
            materialize_dense_potential=arguments.materialize_dense_potential,
            eft_mode=arguments.eft,
            raw_frst_metadata=persisted,
            topology_audit=topology_audit,
            topology_override=persisted.get("topology_override"),
            kaehler_point_seed=kaehler_point_seed,
            kaehler_point_diagnostics=point_diagnostics,
            assignment_pool_rejection_records=assignment_pool_rejection_records,
            allow_m_below_one=arguments.allow_m_below_one,
            allow_overwrite_existing_geometry=(
                arguments.allow_overwrite_existing_geometry
            ),
        )
        finish_stage()
        terminal_record.update(
            {
                "terminal_status": (
                    "accepted_geometry"
                    if arguments.eft
                    else "geometry_only"
                ),
                "terminal_reason": "stage-2 geometry artifact written from retained raw FRST",
                "artifact_status": (
                    generator.ACCEPTED_GEOMETRY_ARTIFACT_STATUS
                    if arguments.eft
                    else generator.GEOMETRY_ONLY_ARTIFACT_STATUS
                ),
                "artifact_written": True,
                "overwrite_event": (
                    "replaced_existing_geometry"
                    if existing_artifact.get("exists")
                    else "created_new_geometry"
                ),
            }
        )
    except Exception as error:
        finish_stage()
        terminal_status = classify_stage2_failure(error)
        topology_audit.update(
            {
                "audit_status": (
                    "complete"
                    if topology_audit["topology_validation_status"] == "passed"
                    else "failed"
                ),
                "stage2_terminal_status": terminal_status,
                "artifact_status": terminal_record["artifact_status"],
                "artifact_written": terminal_record["artifact_written"],
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
            }
        )
        topology_audit["kaehler_point_scan"].update(
            {
                "point_status": "failed",
                "diagnostic_count": len(point_diagnostics),
            }
        )
        failure_record = getattr(error, "record", None)
        if failure_record:
            topology_audit["failure_record"] = failure_record
        terminal_record.update(
            {
                "terminal_status": terminal_status,
                "terminal_reason": f"{type(error).__name__}: {error}",
                "artifact_status": (
                    generator.POOL_PENDING_ARTIFACT_STATUS
                    if arguments.eft
                    and getattr(error, "category", None) == "assignment_pool_shortfall"
                    else "not_written"
                ),
                "artifact_written": False,
            }
        )
        topology_audit["artifact_status"] = terminal_record["artifact_status"]
        topology_audit["artifact_written"] = terminal_record["artifact_written"]
    else:
        finish_stage()
        topology_audit.update(
            {
                "audit_status": "complete",
                "stage2_terminal_status": terminal_record["terminal_status"],
                "artifact_status": terminal_record["artifact_status"],
                "artifact_written": terminal_record["artifact_written"],
            }
        )
        topology_audit["kaehler_point_scan"].update(
            {
                "point_status": "accepted",
                "diagnostic_count": len(point_diagnostics),
            }
        )
    terminal_record["stage2_elapsed_seconds"] = time.monotonic() - candidate_started
    terminal_record["stage_timings"] = stage_timings
    topology_audit["stage2_elapsed_seconds"] = terminal_record["stage2_elapsed_seconds"]
    topology_audit["stage_timings"] = stage_timings
    topology_audit["assignment_pool_rejection_records"] = assignment_pool_rejection_records
    return terminal_record, topology_audit


def read_stage1_manifest(stage1_root):
    """Read the stage-1 manifest when present without treating it as geometry data."""
    manifest_path = Path(stage1_root).resolve() / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_stage1_polytope_manifest(stage1_root):
    """Read the stage-1 polytope provenance without altering it."""
    manifest_path = Path(stage1_root).resolve() / "polytope_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def build_frozen_stage1_population_provenance(input_ledger, stage1_manifest):
    """Record the immutable Stage-1 population used by this Stage-2 run."""
    retained = [
        {
            key: record.get(key)
            for key in (
                "h11",
                "polytope_id",
                "polytope_index",
                "candidate_index",
                "geometry_id",
                "full_triangulation_hash",
                "raw_frst_path",
            )
        }
        for record in input_ledger
        if record.get("stage2_input_status") == "retained_raw_frst"
    ]
    retained.sort(
        key=lambda record: (
            str(record.get("geometry_id", "")),
            str(record.get("full_triangulation_hash", "")),
            str(record.get("raw_frst_path", "")),
        )
    )
    declared_count = None if stage1_manifest is None else stage1_manifest.get(
        "accepted_geometry_count",
        stage1_manifest.get("accepted_count"),
    )
    return {
        "population_target": TARGET_GEOMETRY_COUNT,
        "stage1_manifest_status": (
            None if stage1_manifest is None else stage1_manifest.get("status")
        ),
        "stage1_declared_accepted_count": declared_count,
        "retained_raw_input_count": len(retained),
        "accepted_stage2_geometry_count": None,
        "population_frozen": True,
        "replenishment_allowed": False,
        "population_change_policy": (
            "stage1_population_frozen_no_replenishment_or_population_change"
        ),
        "raw_frst_provenance_preserved": True,
        "raw_frst_identity_fields": [
            "geometry_id",
            "full_triangulation_hash",
            "raw_frst_path",
        ],
        "retained_raw_identity_digest": stable_hash(retained),
    }


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
    arguments.outdir = ensure_fresh_output_root(arguments.outdir)
    if arguments.eft and arguments.visible_sector_policy != "intersecting_d7":
        raise ValueError("--eft requires --visible-sector-policy intersecting_d7")
    if arguments.eft and arguments.orientifold_file is None:
        raise ValueError("--eft requires --orientifold-file")
    if arguments.eft and arguments.moduli_policy != "canonical_qcd":
        raise ValueError("--eft requires --moduli-policy canonical_qcd")
    if arguments.materialize_dense_potential:
        raise ValueError(
            "schema 1.1 uses on-demand potential reconstruction; "
            "dense potential materialization is not permitted"
        )
    if arguments.allow_m_below_one and arguments.moduli_policy != "canonical_qcd":
        raise ValueError(
            "--allow-m-below-one requires --moduli-policy canonical_qcd"
        )
    if arguments.moduli_policy == "canonical_qcd" and not np.isclose(
        arguments.qcd_volume_target, QCD_VOLUME_TARGET, rtol=0.0, atol=1e-12
    ):
        raise ValueError("canonical_qcd requires --qcd-volume-target 40.0")
    if arguments.eft and arguments.qed_volume_max != QED_VOLUME_MAX:
        raise ValueError("--eft requires the inclusive QED volume bound 127.5")
    if arguments.eft_minimum_rows != MINIMUM_EFT_ROWS or arguments.eft_maximum_rows != MAXIMUM_EFT_ROWS:
        raise ValueError("schema 1.1 EFT row bounds are fixed at 100000 and 200000")
    if not arguments.dry_run:
        generator.require_cytools_capabilities("fair", "fast")
    orientifold_config = generator.load_orientifold(arguments.orientifold_file)
    input_ledger = build_input_ledger(arguments.stage1_root)
    atomic_jsonl_dump(Path(arguments.outdir) / "stage2_input_ledger.jsonl", input_ledger)
    progress_path = Path(arguments.outdir) / "stage2_progress.jsonl"
    partial_terminal_path = Path(arguments.outdir) / "stage2_terminal_statuses.partial.jsonl"
    partial_topology_path = Path(arguments.outdir) / "stage2_topology_diagnostics.partial.jsonl"
    partial_kaehler_path = Path(arguments.outdir) / "stage2_kaehler_point_diagnostics.partial.jsonl"
    partial_rejection_path = Path(arguments.outdir) / "stage2_assignment_pool_rejections.partial.jsonl"

    def record_progress(event):
        """Append one flushed progress event for interruption-safe diagnostics."""
        append_jsonl_record(
            progress_path,
            {
                "progress_schema_version": "cyaxiverse-stage2-progress-1.0",
                "timestamp_unix_ns": time.time_ns(),
                **event,
            },
        )

    record_progress(
        {
            "event": "run_started",
            "raw_input_count": len(input_ledger),
            "retained_raw_input_count": sum(
                record.get("stage2_input_status") == "retained_raw_frst"
                for record in input_ledger
            ),
            "command_line": sys.argv if argv is None else [sys.argv[0], *argv],
        }
    )
    stage1_manifest = read_stage1_manifest(arguments.stage1_root)
    stage1_polytope_manifest = read_stage1_polytope_manifest(arguments.stage1_root)
    topology_diagnostics = []
    kaehler_point_diagnostics = []
    assignment_pool_rejection_records = []
    retained_inputs = [
        record
        for record in input_ledger
        if record.get("stage2_input_status") == "retained_raw_frst"
    ]
    stage2_records = [
        {
            **record,
            "terminal_status": record["stage2_input_status"],
            "terminal_reason": record.get("terminal_reason", "stage-1 input unavailable"),
        }
        for record in input_ledger
        if record.get("stage2_input_status") != "retained_raw_frst"
    ]
    for input_record in stage2_records:
        append_jsonl_record(partial_terminal_path, input_record)
        topology_diagnostics.append(
            {
                **build_topology_audit_record(input_record, arguments.backend),
                "orientifold_validation": orientifold_audit_record(
                    orientifold_config
                ),
                "audit_status": "not_run",
                "stage2_terminal_status": input_record["terminal_status"],
                "failure_reason": input_record.get("terminal_reason"),
            }
        )
        kaehler_point_diagnostics.append(
            {
                key: input_record.get(key)
                for key in (
                    "h11",
                    "polytope_id",
                    "raw_frst_path",
                    "full_triangulation_hash",
                    "geometry_id",
                )
            }
            | {
                "attempted": False,
                "point_status": "not_run",
                "scan_status": "stage2_input_unavailable",
                "failure_reason": input_record.get("terminal_reason"),
            }
        )
    if not arguments.dry_run:
        completed_count = len(stage2_records)
        for retained_input_sequence, raw_frst_record in enumerate(
            retained_inputs, start=1
        ):
            record_progress(
                {
                    "event": "candidate_started",
                    "retained_input_sequence": retained_input_sequence,
                    "completed_count": completed_count,
                    **_stage2_identity(raw_frst_record),
                }
            )
            terminal_record, topology_audit = process_raw_frst_artifact(
                arguments,
                raw_frst_record,
                orientifold_config,
                arguments.outdir,
                progress_callback=record_progress,
            )
            terminal_record.update(
                {
                    "retained_input_sequence": retained_input_sequence,
                    "completed_count": completed_count + 1,
                }
            )
            stage2_records.append(terminal_record)
            assignment_pool_rejection_records.extend(
                topology_audit.pop("assignment_pool_rejection_records", ())
            )
            topology_diagnostics.append(topology_audit)
            for point_record in topology_audit.get(
                "kaehler_point_scan", {}
            ).get("diagnostics", ()):
                kaehler_point_diagnostics.append(
                    {
                        key: raw_frst_record.get(key)
                        for key in (
                            "h11",
                            "polytope_id",
                            "raw_frst_path",
                            "full_triangulation_hash",
                            "geometry_id",
                        )
                    }
                    | dict(point_record)
                )
            append_jsonl_record(partial_terminal_path, terminal_record)
            append_jsonl_record(partial_topology_path, topology_audit)
            for point_record in topology_audit.get(
                "kaehler_point_scan", {}
            ).get("diagnostics", ()):
                append_jsonl_record(
                    partial_kaehler_path,
                    {
                        key: raw_frst_record.get(key)
                        for key in (
                            "h11",
                            "polytope_id",
                            "raw_frst_path",
                            "full_triangulation_hash",
                            "geometry_id",
                        )
                    }
                    | dict(point_record),
                )
            for rejection_record in topology_audit.get(
                "assignment_pool_rejection_records", ()
            ):
                append_jsonl_record(partial_rejection_path, rejection_record)
            completed_count += 1
            record_progress(
                {
                    "event": "candidate_finished",
                    "retained_input_sequence": retained_input_sequence,
                    "completed_count": completed_count,
                    "terminal_status": terminal_record.get("terminal_status"),
                    "terminal_reason": terminal_record.get("terminal_reason"),
                    "stage2_elapsed_seconds": terminal_record.get(
                        "stage2_elapsed_seconds"
                    ),
                    "stage_timings": terminal_record.get("stage_timings", ()),
                    **_stage2_identity(raw_frst_record),
                }
            )
    else:
        for raw_frst_record in retained_inputs:
            topology_diagnostics.append(
                {
                    **build_topology_audit_record(
                        raw_frst_record, arguments.backend
                    ),
                    "orientifold_validation": orientifold_audit_record(
                        orientifold_config
                    ),
                    "audit_status": "not_run",
                    "stage2_terminal_status": "dry_run",
                    "failure_reason": "stage-2 reconstruction was not run",
                }
            )
            kaehler_point_diagnostics.append(
                {
                    key: raw_frst_record.get(key)
                    for key in (
                        "h11",
                        "polytope_id",
                        "raw_frst_path",
                        "full_triangulation_hash",
                        "geometry_id",
                    )
                }
                | {
                    "attempted": False,
                    "point_status": "not_run",
                    "scan_status": "dry_run",
                    "failure_reason": "stage-2 reconstruction was not run",
                }
            )

    accepted_geometry_paths = [
        record["output_path"]
        for record in stage2_records
        if record.get("terminal_status") == "accepted_geometry"
        and Path(record.get("output_path", "")).is_file()
    ]
    model_records = []
    model_rows = []
    model_error = None
    allocation = None
    if arguments.eft and not arguments.dry_run:
        try:
            model_rows, model_records, allocation = generator.expand_eft_reference_rows(
                accepted_geometry_paths,
                arguments.seed,
                arguments.eft_minimum_rows,
                arguments.eft_maximum_rows,
            )
        except Exception as error:
            model_error = error
            if isinstance(error, generator.ModelTargetShortfall):
                allocation = error.allocation
                model_records.extend(error.records)
            else:
                model_records.append(
                    {
                        "terminal_status": "invalid_row_schema",
                        "terminal_reason": f"{type(error).__name__}: {error}",
                    }
                )
    model_dataset_status = None
    model_reconciliation = None
    if arguments.eft and not arguments.dry_run and model_error is None:
        validated_capacity = (
            0
            if allocation is None
            else int(allocation.get("validated_assignment_capacity", 0))
        )
        model_reconciliation = reconcile_eft_capacity(
            validated_capacity,
            len(model_rows),
            minimum_rows=arguments.eft_minimum_rows,
            maximum_rows=arguments.eft_maximum_rows,
        )
        model_dataset_status = model_reconciliation["dataset_status"]
        try:
            eft_path = arguments.eft_output_path or str(Path(arguments.outdir) / "eft_models.parquet")
            if not os.path.isabs(eft_path):
                eft_path = str(Path(arguments.outdir) / eft_path)
            eft_path = Path(eft_path).resolve()
            try:
                eft_path.relative_to(Path(arguments.outdir).resolve())
            except ValueError as error:
                raise ValueError("--eft-output-path must remain inside --outdir") from error
            write_eft_parquet(
                str(eft_path),
                model_rows,
                metadata={
                    "cyaxiverse_dataset_status": model_dataset_status,
                    "production_complete": model_reconciliation[
                        "production_complete"
                    ],
                    "diagnostic_success": model_reconciliation[
                        "diagnostic_success"
                    ],
                    "model_target_shortfall": model_reconciliation[
                        "model_target_shortfall"
                    ],
                    "requested_target": model_reconciliation["requested_target"],
                    "minimum_acceptable": model_reconciliation[
                        "minimum_acceptable"
                    ],
                },
            )
            model_records.append(
                {
                    "terminal_status": (
                        "accepted_model_table"
                        if model_reconciliation["production_complete"]
                        else "accepted_diagnostic_partial_model_table"
                    ),
                    "terminal_reason": (
                        "compact EFT-reference Parquet written atomically"
                        if model_reconciliation["production_complete"]
                        else "diagnostic partial EFT-reference Parquet written atomically"
                    ),
                    "output_path": str(eft_path),
                    "row_count": len(model_rows),
                    "dataset_status": model_dataset_status,
                    "production_complete": model_reconciliation[
                        "production_complete"
                    ],
                    "diagnostic_success": model_reconciliation[
                        "diagnostic_success"
                    ],
                }
            )
        except Exception as error:
            model_error = error
            model_dataset_status = "storage_failure"
            model_reconciliation = {
                **model_reconciliation,
                "dataset_status": "storage_failure",
                "diagnostic_success": False,
                "production_complete": False,
            }
            model_records.append(
                {
                    "terminal_status": "storage_failure",
                    "terminal_reason": f"{type(error).__name__}: {error}",
                }
            )
    storage_estimate = estimate_storage(arguments.outdir, len(model_rows))
    kaehler_point_status_counts = {}
    for point_record in kaehler_point_diagnostics:
        status = point_record.get("point_status", "unknown")
        kaehler_point_status_counts[status] = (
            kaehler_point_status_counts.get(status, 0) + 1
        )
    kaehler_point_attempted_count = sum(
        bool(point_record.get("attempted", False))
        for point_record in kaehler_point_diagnostics
    )
    assignment_pool_rejection_records.sort(
        key=lambda record: (
            str(record.get("geometry_id", "")),
            int(record.get("qcd_index", -1)),
            int(record.get("qed_index", -1))
            if record.get("qed_index") is not None
            else -1,
            str(record.get("terminal_status", "")),
            str(record.get("terminal_reason", "")),
        )
    )
    stage1_population = build_frozen_stage1_population_provenance(
        input_ledger, stage1_manifest
    )
    stage1_population["accepted_stage2_geometry_count"] = len(
        accepted_geometry_paths
    )
    model_failure_status = None
    if model_error is not None:
        model_failure_status = next(
            (
                record.get("terminal_status")
                for record in reversed(model_records)
                if record.get("terminal_status")
                in {"storage_failure", "invalid_row_schema", "model_target_shortfall"}
            ),
            "model_target_shortfall",
        )
    run_manifest = {
        "schema_version": f"cyaxiverse-stage2-evaluation-{SCHEMA_1_1_VERSION}",
        "stage": "stage2_geometry_and_eft_generation",
        "status": (
            "dry_run"
            if arguments.dry_run
            else model_failure_status
            if model_error is not None
            else "completed_geometry_only"
            if not arguments.eft
            else "completed"
            if model_reconciliation is not None
            and model_reconciliation["production_complete"]
            else "completed_diagnostic_partial"
        ),
        "output_root": arguments.outdir,
        "stage1_root": str(Path(arguments.stage1_root).resolve()),
        "stage1_manifest_status": None if stage1_manifest is None else stage1_manifest.get("status"),
        "command_line": sys.argv if argv is None else [sys.argv[0], *argv],
        "source_commit": current_source_commit(),
        "python_version": platform.python_version(),
        "cytools_version": getattr(generator.cytools, "version", None),
        "hardware_threads": os.cpu_count(),
        "seed": arguments.seed,
        "moduli_policy": arguments.moduli_policy,
        "allow_m_below_one": bool(arguments.allow_m_below_one),
        "allow_overwrite_existing_geometry": bool(
            arguments.allow_overwrite_existing_geometry
        ),
        "geometry_artifact_policy": {
            "geometry_only_status": generator.GEOMETRY_ONLY_ARTIFACT_STATUS,
            "accepted_geometry_status": generator.ACCEPTED_GEOMETRY_ARTIFACT_STATUS,
            "pool_pending_status": generator.POOL_PENDING_ARTIFACT_STATUS,
            "geometry_only_pool_policy": "may_finalize_before_pool_construction",
            "eft_pool_policy": "complete_validated_hashed_pool_required_before_finalization",
            "pool_pending_eft_policy": "not_accepted_and_no_final_artifact",
            "overwrite_policy": "explicit_allow_overwrite_existing_geometry_only",
            "temporary_artifact_policy": "delete_after_status_recording",
        },
        "kaehler_point_attempt_budget": arguments.max_kaehler_attempts,
        "kaehler_point_attempt_budget_semantics": (
            "one canonical tip evaluation"
            if arguments.moduli_policy == "canonical_qcd"
            else "at most this many evaluations including the canonical tip"
        ),
        "kaehler_point_seed_derivation": (
            "stable_seed('stage2-kaehler-point', base_seed, geometry_id)"
        ),
        "kaehler_point_status_counts": kaehler_point_status_counts,
        "kaehler_point_attempted_count": kaehler_point_attempted_count,
        "qcd_volume_target": generator.QCD_VOLUME_TARGET,
        "qcd_volume_tolerance": generator.QCD_VOLUME_TOLERANCE,
        "normalization_failure_status": "qcd_normalization_failure",
        "normalization_repair_policy": "none",
        "selection_policy": (
            "explicit_qcd_divisor_index_or_first_eligible_ascending_index"
        ),
        "post_selection_fallback": "none",
        "divisor_volume_tolerance": generator.DIVISOR_VOLUME_TOLERANCE,
        "divisor_volume_contract": {
            "prime_lower_bound": arguments.min_prime_divisor_volume,
            "effective_lower_bound": arguments.min_divisor_volume,
            "tolerance": generator.DIVISOR_VOLUME_TOLERANCE,
            "failure_status": "qcd_normalization_failure",
            "evidence_group": "cytools/geometric/divisor_volume_evidence",
            "point_failure_policy": (
                "retry_within_adaptive_budget"
                if arguments.moduli_policy == "adaptive"
                else "single_canonical_point"
            ),
        },
        "raw_input_count": len(input_ledger),
        "raw_input_count_by_h11_and_status": count_by_h11(input_ledger),
        "retained_raw_input_count": len(retained_inputs),
        "duplicate_full_triangulation_count": sum(
            record.get("stage2_input_status") == "duplicate_full_triangulation"
            for record in input_ledger
        ),
        "unavailable_raw_input_count": sum(
            record.get("stage2_input_status")
            in {"missing_raw_frst", "invalid_frst", "input_identity_mismatch"}
            for record in input_ledger
        ),
        "stage2_terminal_count_by_h11_and_status": count_by_h11(
            stage2_records, status_key="terminal_status"
        ),
        "stage2_artifact_count_by_status": count_by_h11(
            stage2_records, status_key="artifact_status"
        ),
        "geometry_overwrite_event_count": sum(
            record.get("overwrite_event") == "replaced_existing_geometry"
            for record in stage2_records
        ),
        "accepted_geometry_count": len(accepted_geometry_paths),
        "stage2_filters_do_not_replenish_stage1": True,
        "stage1_population": stage1_population,
        "stage1_population_frozen": True,
        "stage1_replenishment_allowed": False,
        "stage1_population_repair_policy": "forbidden_after_stage1_completion",
        "stage2_input_ledger": "stage2_input_ledger.jsonl",
        "stage2_progress": "stage2_progress.jsonl",
        "stage2_progress_schema_version": "cyaxiverse-stage2-progress-1.0",
        "stage2_progress_flush_policy": (
            "append and fsync run, candidate, stage-start, and candidate-finish events"
        ),
        "stage2_partial_terminal_statuses": "stage2_terminal_statuses.partial.jsonl",
        "stage2_partial_topology_diagnostics": "stage2_topology_diagnostics.partial.jsonl",
        "stage2_partial_kaehler_point_diagnostics": (
            "stage2_kaehler_point_diagnostics.partial.jsonl"
        ),
        "stage2_partial_assignment_pool_rejections": (
            "stage2_assignment_pool_rejections.partial.jsonl"
        ),
        "stage2_topology_diagnostics": "stage2_topology_diagnostics.jsonl",
        "stage2_kaehler_point_diagnostics": "stage2_kaehler_point_diagnostics.jsonl",
        "stage2_assignment_pool_rejections": "stage2_assignment_pool_rejections.jsonl",
        "assignment_pool_rejection_policy": {
            "sidecar": "all_candidate_pair_records_with_labels_indices_status_reason",
            "hdf5": "aggregate_rejection_counts_and_reasons_only",
        },
        "topology_audit_schema_version": "cyaxiverse-stage2-topology-audit-1.0",
        "accepted_stage2_status": "accepted_geometry",
        "geometry_only_stage2_status": "geometry_only",
        "orientifold_policy": {
            "required_for_visible_sector_policy": (
                arguments.visible_sector_policy == "intersecting_d7"
            ),
            "visible_sector_policy": arguments.visible_sector_policy,
            "input_status": orientifold_config.get("status"),
            "involution_type": orientifold_config.get("involution_type"),
            "h11_parity_policy": "record_only_not_enforced",
            "preservation_failure_status": "orientifold_invariance_failure",
            "fixed_locus_validation": "not_performed",
            "tadpole_validation": "not_performed",
            "physical_orientifold_claim": "not_made",
            "kaehler_subspace_policy": arguments.orientifold_kaehler_policy,
            "effective_cone_rays_required_for_Q": True,
            "kaehler_cone_rays_exported": arguments.export_kaehler_rays,
        },
        "user_decisions": [
            {
                "stage": 1,
                "answer_value": "confirmed",
                "answer": "stage 2 uses only retained raw FRST files",
            },
            {
                "stage": 1,
                "answer_value": "confirmed",
                "answer": "missing or corrupt raw FRSTs are unavailable and are not replaced",
            },
            {
                "stage": 1,
                "answer_value": "confirmed",
                "answer": "the stage-2 input ledger is separate",
            },
            {
                "stage": 3,
                "answer_value": "confirmed",
                "answer": (
                    "h11_plus and h11_minus are recorded; h11_minus=0 is not "
                    "enforced by the intersecting-D7 policy"
                ),
            },
            {
                "stage": 4,
                "answer_value": "confirmed",
                "answer": (
                    "retain the current CYTools stretched-cone definition "
                    "and slack threshold"
                ),
            },
            {
                "stage": 4,
                "answer_value": "confirmed",
                "answer": (
                    "the reference run does not require an orientifold-even "
                    "Kaehler-subspace intersection; that check remains opt-in"
                ),
            },
            {
                "stage": 4,
                "answer_value": "confirmed",
                "answer": (
                    "effective-cone rays are required for Q; Kaehler-cone rays "
                    "remain optional"
                ),
            },
            {
                "stage": 3,
                "answer_value": "confirmed",
                "answer": (
                    "orientifold-preservation failures reject the current geometry; "
                    "visible_sector_policy=none remains an explicit non-orientifold mode"
                ),
            },
            {
                "stage": 3,
                "answer_value": "confirmed",
                "answer": (
                    "fixed-locus, tadpole, and physical orientifold claims are "
                    "outside this generator"
                ),
            },
            {
                "stage": 5,
                "answer_value": "confirmed",
                "answer": (
                    "canonical_qcd is the Glimmers-style production policy; "
                    "adaptive randomized Kähler points are diagnostic only"
                ),
            },
            {
                "stage": 5,
                "answer_value": "confirmed",
                "answer": (
                    "the adaptive diagnostic budget is 100 total point evaluations, "
                    "including the canonical tip"
                ),
            },
            {
                "stage": 5,
                "answer_value": "confirmed",
                "answer": (
                    "a point shortfall retains the Stage-1 FRST in Stage-2 accounting "
                    "with kaehler_point_shortfall"
                ),
            },
            {
                "stage": 7,
                "answer_value": "confirmed",
                "answer": (
                    "retain Vol(D_QCD)=40.0 with final post-normalization absolute "
                    "tolerance 1e-9 and divisor lower-bound tolerance 1e-8"
                ),
            },
            {
                "stage": 7,
                "answer_value": "confirmed",
                "answer": (
                    "post-normalization validation is strict: finite data, divisor "
                    "lower bounds, cone membership, and positivity are required; "
                    "failures are qcd_normalization_failure with no repair"
                ),
            },
            {
                "stage": 7,
                "answer_value": "confirmed",
                "answer": (
                    "honor explicit qcd_divisor_index; otherwise choose the first "
                    "eligible candidate in ascending index order after visible-sector "
                    "compatibility filtering, with allow_m_below_one retained"
                ),
            },
            {
                "stage": 8,
                "answer_value": "confirmed",
                "answer": (
                    "accept a geometry only after the complete eligible ordered "
                    "QCD/QED assignment pool is built, validated, and deterministically hashed; "
                    "empty or incomplete pools are failures"
                ),
            },
            {
                "stage": 8,
                "answer_value": "confirmed",
                "answer": "use the inclusive QED condition Vol(D_QED) <= 127.5",
            },
            {
                "stage": 8,
                "answer_value": "confirmed",
                "answer": (
                    "write every candidate-pair rejection with labels, indices, status, "
                    "and reason to a sidecar JSONL; retain aggregate rejection counts "
                    "and reasons in production HDF5"
                ),
            },
            {
                "stage": 9,
                "answer_value": "confirmed",
                "answer": (
                    "retain Q as h11 x N_instanton with charge vectors as columns, "
                    "and use q_pair[:, k] = q_direct[:, pair_j[k]] - "
                    "q_direct[:, pair_i[k]]"
                ),
            },
            {
                "stage": 9,
                "answer_value": "confirmed",
                "answer": (
                    "defer potential construction and validation to EFT-row "
                    "generation; geometry HDF5 acceptance requires sufficient "
                    "references for deterministic reconstruction"
                ),
            },
            {
                "stage": 9,
                "answer_value": "confirmed",
                "answer": (
                    "store no dense Q, L, K-inverse, volume, or potential arrays "
                    "in production HDF5 or EFT rows; persist reconstruction inputs, "
                    "provenance, references, and replay certificates"
                ),
            },
            {
                "stage": 10,
                "answer_value": "confirmed",
                "answer": (
                    "an existing cyax.h5 may be overwritten only with the explicit "
                    "allow-overwrite-existing-geometry flag; record the prior artifact "
                    "identity/hash and overwrite event in provenance"
                ),
            },
            {
                "stage": 10,
                "answer_value": "confirmed",
                "answer": (
                    "delete readable-but-incomplete temporary HDF5 artifacts after "
                    "recording the failure in status artifacts"
                ),
            },
            {
                "stage": 10,
                "answer_value": "confirmed",
                "answer": (
                    "geometry-only runs may finalize an explicitly labelled geometry_only "
                    "artifact before pool construction; EFT mode requires a complete "
                    "validated and hashed pool for accepted_geometry; pool_pending is "
                    "not accepted under the EFT contract"
                ),
            },
            {
                "stage": 11,
                "answer_value": "confirmed",
                "answer": (
                    "sample ordered assignment-pool entries with replacement; keep row identity "
                    "as geometry identity plus ordered assignment, collapse duplicate draws, "
                    "retry row-level failures within the same accepted geometry, and cap draws "
                    "per geometry at M_g = 10 * k_g"
                ),
            },
            {
                "stage": 12,
                "answer_value": "confirmed",
                "answer": (
                    "the exact EFT row target is 200000 and the minimum acceptable "
                    "count is 100000"
                ),
            },
            {
                "stage": 12,
                "answer_value": "confirmed",
                "answer": (
                    "below-minimum validated capacity emits a clearly labelled "
                    "diagnostic partial dataset, records model_target_shortfall and "
                    "complete accounting, and is diagnostically successful but not "
                    "production-complete"
                ),
            },
            {
                "stage": 12,
                "answer_value": "confirmed",
                "answer": (
                    "the completed 1400-FRST Stage-1 population is frozen; row "
                    "shortfalls cannot be repaired by replenishment or population "
                    "changes, and raw-FRST provenance must be preserved"
                ),
            },
        ],
        "unresolved_scientific_choices": [],
        "eft": {
            "minimum_rows": arguments.eft_minimum_rows,
            "minimum_acceptable_rows": arguments.eft_minimum_rows,
            "maximum_rows": arguments.eft_maximum_rows,
            "target_rows": arguments.eft_maximum_rows,
            "rows_written": len(model_rows),
            "terminal_status": (
                model_failure_status
                if model_error is not None
                else "model_target_shortfall"
                if model_reconciliation is not None
                and model_reconciliation["model_target_shortfall"]
                else "accepted_model_table"
                if model_reconciliation is not None
                else None
            ),
            "dataset_status": model_dataset_status,
            "production_complete": (
                False
                if model_reconciliation is None
                else model_reconciliation["production_complete"]
            ),
            "diagnostic_success": (
                False
                if model_reconciliation is None
                else model_reconciliation["diagnostic_success"]
            ),
            "model_target_shortfall": (
                None
                if model_reconciliation is None
                else model_reconciliation["model_target_shortfall"]
            ),
            "raw_assignment_capacity": (
                None
                if allocation is None
                else allocation.get("raw_assignment_capacity")
            ),
            "validated_assignment_capacity": (
                None
                if allocation is None
                else allocation.get("validated_assignment_capacity")
            ),
            "capacity_shortfall": (
                None
                if model_reconciliation is None
                else model_reconciliation["capacity_shortfall"]
            ),
            "row_shortfall": (
                None
                if model_reconciliation is None
                else model_reconciliation["row_shortfall"]
            ),
            "minimum_shortfall": (
                None
                if model_reconciliation is None
                else model_reconciliation["minimum_shortfall"]
            ),
            "counts_reconcile": (
                None
                if model_reconciliation is None
                else {
                    **model_reconciliation,
                    "raw_assignment_capacity": allocation.get(
                        "raw_assignment_capacity", 0
                    ),
                    "rows_do_not_exceed_validated_capacity": len(model_rows)
                    <= model_reconciliation["validated_assignment_capacity"],
                }
            ),
            "sampling_policy": {
                "assignment_sampling": "uniform_with_replacement",
                "row_identity": "geometry_id_plus_ordered_assignment",
                "duplicate_policy": "collapse_duplicate_assignment_draws",
                "row_failure_policy": "retry_same_geometry",
                "draw_cap_formula": "M_g = 10 * k_g",
            },
            "allocation": allocation,
            "draw_accounting": (
                None if allocation is None else allocation.get("per_geometry_sampling")
            ),
        },
    }
    atomic_jsonl_dump(Path(arguments.outdir) / "stage2_terminal_statuses.jsonl", stage2_records)
    atomic_jsonl_dump(
        Path(arguments.outdir) / "stage2_topology_diagnostics.jsonl",
        topology_diagnostics,
    )
    atomic_jsonl_dump(
        Path(arguments.outdir) / "stage2_kaehler_point_diagnostics.jsonl",
        kaehler_point_diagnostics,
    )
    atomic_jsonl_dump(
        Path(arguments.outdir) / "stage2_assignment_pool_rejections.jsonl",
        assignment_pool_rejection_records,
    )
    if arguments.eft:
        atomic_jsonl_dump(Path(arguments.outdir) / "model_terminal_statuses.jsonl", model_records)
    charge_factorized_manifest = generator.factorized_manifest_for_paths(
        accepted_geometry_paths
    )
    atomic_json_dump(
        Path(arguments.outdir) / "charge_factorized_manifest.json",
        charge_factorized_manifest,
    )
    atomic_json_dump(
        Path(arguments.outdir) / "polytope_manifest.json",
        {
            "stage": "stage2_input_provenance",
            "source_stage1_manifest": stage1_polytope_manifest,
            "raw_input_count_by_h11_and_status": count_by_h11(input_ledger),
        },
    )
    atomic_json_dump(
        Path(arguments.outdir) / "summary_by_h11_and_status.json",
        summarize_terminal_records(stage2_records, model_records),
    )
    atomic_json_dump(Path(arguments.outdir) / "storage_estimate.json", storage_estimate)
    atomic_json_dump(Path(arguments.outdir) / "run_manifest.json", run_manifest)
    record_progress(
        {
            "event": "run_finalized",
            "status": run_manifest["status"],
            "completed_count": len(stage2_records),
            "accepted_geometry_count": len(accepted_geometry_paths),
            "eft_row_count": len(model_rows),
        }
    )
    if arguments.eft and model_error is not None:
        raise RuntimeError(str(model_error)) from model_error
    print(
        json.dumps(
            {
                "status": run_manifest["status"],
                "raw_inputs": len(input_ledger),
                "accepted_geometries": len(accepted_geometry_paths),
                "eft_rows": len(model_rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    main()
