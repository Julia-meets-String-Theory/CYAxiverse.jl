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
from pathlib import Path

import numpy as np

import generate_geometric_data_multitriangulation as generator
from glimmers_raw_frst import (
    RawFRSTError,
    build_input_ledger,
    compute_triangulation_hash,
    count_by_h11,
    read_raw_frst_artifact,
)
from glimmers_schema11 import (
    MAXIMUM_EFT_ROWS,
    MINIMUM_EFT_ROWS,
    QCD_VOLUME_TARGET,
    QED_VOLUME_MAX,
    SCHEMA_VERSION as SCHEMA_1_1_VERSION,
    atomic_json_dump,
    atomic_jsonl_dump,
    ensure_fresh_output_root,
    estimate_storage,
    summarize_terminal_records,
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


def reconstruct_raw_frst(raw_frst_record, backend, topology_audit):
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


def process_raw_frst_artifact(
    arguments, raw_frst_record, orientifold_config, output_root
):
    """Process one retained raw FRST and return terminal and audit records."""
    topology_audit = build_topology_audit_record(raw_frst_record, arguments.backend)
    point_diagnostics = []
    topology_audit["kaehler_point_scan"] = {
        "policy": arguments.moduli_policy,
        "allow_m_below_one": bool(arguments.allow_m_below_one),
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
    output_path = build_geometry_output_path(output_root, raw_frst_record)
    terminal_record["output_path"] = str(output_path.resolve())
    try:
        persisted, polytope, triangulation = reconstruct_raw_frst(
            raw_frst_record, arguments.backend, topology_audit
        )
        calabi_yau = triangulation.get_cy()
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
            kaehler_point_seed=kaehler_point_seed,
            kaehler_point_diagnostics=point_diagnostics,
            allow_m_below_one=arguments.allow_m_below_one,
        )
        terminal_record.update(
            {
                "terminal_status": "accepted_geometry",
                "terminal_reason": "stage-2 geometry artifact written from retained raw FRST",
            }
        )
    except Exception as error:
        terminal_status = classify_stage2_failure(error)
        topology_audit.update(
            {
                "audit_status": (
                    "complete"
                    if topology_audit["topology_validation_status"] == "passed"
                    else "failed"
                ),
                "stage2_terminal_status": terminal_status,
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
            }
        )
    else:
        topology_audit.update(
            {
                "audit_status": "complete",
                "stage2_terminal_status": terminal_record["terminal_status"],
            }
        )
        topology_audit["kaehler_point_scan"].update(
            {
                "point_status": "accepted",
                "diagnostic_count": len(point_diagnostics),
            }
        )
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
    if arguments.allow_m_below_one and arguments.moduli_policy != "canonical_qcd":
        raise ValueError(
            "--allow-m-below-one requires --moduli-policy canonical_qcd"
        )
    if arguments.moduli_policy == "canonical_qcd" and not np.isclose(
        arguments.qcd_volume_target, QCD_VOLUME_TARGET, rtol=0.0, atol=1e-12
    ):
        raise ValueError("canonical_qcd requires --qcd-volume-target 40.0")
    if arguments.eft and arguments.qed_volume_max != QED_VOLUME_MAX:
        raise ValueError("--eft requires the strict QED volume bound 127.5")
    if arguments.eft_minimum_rows != MINIMUM_EFT_ROWS or arguments.eft_maximum_rows != MAXIMUM_EFT_ROWS:
        raise ValueError("schema 1.1 EFT row bounds are fixed at 100000 and 200000")
    if not arguments.dry_run:
        generator.require_cytools_capabilities("fair", "fast")
    orientifold_config = generator.load_orientifold(arguments.orientifold_file)
    input_ledger = build_input_ledger(arguments.stage1_root)
    atomic_jsonl_dump(Path(arguments.outdir) / "stage2_input_ledger.jsonl", input_ledger)
    stage1_manifest = read_stage1_manifest(arguments.stage1_root)
    stage1_polytope_manifest = read_stage1_polytope_manifest(arguments.stage1_root)
    topology_diagnostics = []
    kaehler_point_diagnostics = []
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
        for raw_frst_record in retained_inputs:
            terminal_record, topology_audit = process_raw_frst_artifact(
                arguments, raw_frst_record, orientifold_config, arguments.outdir
            )
            stage2_records.append(terminal_record)
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
    if arguments.eft and not arguments.dry_run and not accepted_geometry_paths:
        model_error = generator.ModelTargetShortfall(
            "no accepted geometry artifacts are available for EFT row generation",
            [],
        )
        model_records.append(
            {
                "terminal_status": "model_target_shortfall",
                "terminal_reason": str(model_error),
            }
        )
    if arguments.eft and not arguments.dry_run and accepted_geometry_paths:
        try:
            model_rows, model_records, allocation = generator.expand_eft_reference_rows(
                accepted_geometry_paths,
                arguments.seed,
                arguments.eft_minimum_rows,
                arguments.eft_maximum_rows,
            )
        except Exception as error:
            model_error = error
            model_records.append(
                {
                    "terminal_status": "model_target_shortfall"
                    if isinstance(error, generator.ModelTargetShortfall)
                    else "invalid_row_schema",
                    "terminal_reason": f"{type(error).__name__}: {error}",
                }
            )
    if arguments.eft and not arguments.dry_run and model_error is None and model_rows:
        try:
            eft_path = arguments.eft_output_path or str(Path(arguments.outdir) / "eft_models.parquet")
            if not os.path.isabs(eft_path):
                eft_path = str(Path(arguments.outdir) / eft_path)
            eft_path = Path(eft_path).resolve()
            try:
                eft_path.relative_to(Path(arguments.outdir).resolve())
            except ValueError as error:
                raise ValueError("--eft-output-path must remain inside --outdir") from error
            write_eft_parquet(str(eft_path), model_rows)
            model_records.append(
                {
                    "terminal_status": "accepted_model_row",
                    "terminal_reason": "compact EFT-reference Parquet written atomically",
                    "output_path": str(eft_path),
                    "row_count": len(model_rows),
                }
            )
        except Exception as error:
            model_error = error
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
    run_manifest = {
        "schema_version": f"cyaxiverse-stage2-evaluation-{SCHEMA_1_1_VERSION}",
        "stage": "stage2_geometry_and_eft_generation",
        "status": (
            "dry_run"
            if arguments.dry_run
            else "model_target_shortfall"
            if model_error is not None
            else "completed_geometry_only"
            if not arguments.eft
            else "completed"
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
        "accepted_geometry_count": len(accepted_geometry_paths),
        "stage2_filters_do_not_replenish_stage1": True,
        "stage2_input_ledger": "stage2_input_ledger.jsonl",
        "stage2_topology_diagnostics": "stage2_topology_diagnostics.jsonl",
        "stage2_kaehler_point_diagnostics": "stage2_kaehler_point_diagnostics.jsonl",
        "topology_audit_schema_version": "cyaxiverse-stage2-topology-audit-1.0",
        "accepted_stage2_status": "accepted_geometry",
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
        ],
        "unresolved_scientific_choices": [
            "Divisor, visible-sector, QCD/QED-pool, potential, and EFT stopping questions remain recorded by the stage-2 command options and require explicit confirmation before production claims.",
        ],
        "eft": {
            "minimum_rows": arguments.eft_minimum_rows,
            "maximum_rows": arguments.eft_maximum_rows,
            "rows_written": len(model_rows),
            "allocation": allocation,
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
