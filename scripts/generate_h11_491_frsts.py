#!/usr/bin/env python3
"""Generate full CYAxiverse HDF5 geometries for the h11=491 KS polytope.

This is the h11=491-focused companion to
``generate_geometric_data_multitriangulation.py``.  It uses the checked-in
manifest for the unique favorable N-lattice polytope with Hodge numbers
``(491, 11)`` and reuses the package's ``generate_and_save_geometry`` writer.
The resulting files have the normal layout::

    OUTDIR/h11_491/np_0000001/cy_0000001/cyax.h5

and contain the full ``cytools/geometric`` and ``cytools/potential`` groups,
including ``L``, ``Q``, ``Kinv``, intersection data, cone data, divisor
volumes, and construction metadata.  The downstream physical selection
controls therefore remain available and default to the package values.

The default sampler is the benchmarked ``ntfe_fast`` proposal with explicit
two-face-combination accounting.  On this polytope the fast two-face sampler
can exhaust its finite source after one proposal; the run report records that
shortfall rather than presenting it as a large or representative sample.  The
optional ``fast``, ``fair``, and ``gnn_ntfe`` proposal families retain the
package's controls and provenance metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import sys
import tempfile
import time

# Keep CYTools' inequality cache outside the repository.  The location is
# configurable for persistent runs and defaults to a temporary directory.
CYTOOLS_CACHE = os.environ.get(
    "CYAX_H11_491_CACHE",
    os.environ.get(
        "CYAX_BENCH_CACHE",
        os.path.join(tempfile.gettempdir(), "cyax-h11-491-frsts-cache"),
    ),
)
os.makedirs(CYTOOLS_CACHE, exist_ok=True)
try:
    import platformdirs

    platformdirs.user_cache_dir = lambda *args, **kwargs: CYTOOLS_CACHE
except ImportError:  # pragma: no cover - CYTools normally depends on platformdirs
    pass

import cytools
import numpy as np
from cytools import Polytope

import generate_geometric_data_multitriangulation as package_generator
import probe_h11_491_sampler as sampler_probe


H11 = 491
H21 = 11
POLYTOPE_INDEX = 1
SCHEMA_VERSION = package_generator.SCHEMA_VERSION
DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(__file__), "manifests", "h11_491_11_ks.json"
)
DEFAULT_OUTDIR = "h11_491_geometry"
EXPECTED_REJECTIONS = (
    package_generator.PrefactorCriterionNotMet,
    package_generator.NoPhysicalKaehlerPoint,
    package_generator.NoQcdDivisorVolume,
    package_generator.NoStandardModelAssignment,
    package_generator.FinalGeometryValidationFailed,
    package_generator.NoVisibleSectorAssignment,
)


def file_sha256(path):
    """Hash a manifest without depending on its JSON formatting."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(path, payload):
    """Write a complete run report before replacing its destination."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=".h11_491_geometry-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                package_generator._jsonable(payload),
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_h491_polytope(manifest_path):
    """Load and validate the one local manifest entry used by this script."""
    manifest = package_generator.load_polytope_manifest(manifest_path)
    vertices_list = manifest["by_h11"].get(H11, [])
    if len(vertices_list) != 1:
        raise RuntimeError(
            "The h11=491 manifest must contain exactly one polytope; "
            f"found {len(vertices_list)}."
        )
    vertices = np.asarray(vertices_list[0], dtype=int)
    poly = Polytope(vertices, deterministic_glsm_basis=True)
    observed = (int(poly.h11()), int(poly.h21()))
    if observed != (H11, H21):
        raise RuntimeError(
            "The manifest does not reconstruct the expected Hodge data: "
            f"expected {(H11, H21)}, observed {observed}."
        )
    if poly.ambient_dim() != 4 or poly.dim() != 4:
        raise RuntimeError("The h11=491 target must be a full-dimensional 4-polytope.")
    if not poly.is_reflexive():
        raise RuntimeError("The h11=491 manifest polytope is not reflexive.")
    if not poly.is_favorable(lattice="N"):
        raise RuntimeError("The h11=491 manifest polytope is not favorable on N.")
    return manifest, vertices, poly


def polytope_metadata(poly, vertices, manifest_path, manifest):
    """Build JSON-safe identity and point-configuration metadata."""
    polytope_id, canonical_points = package_generator.polytope_identity(poly)
    return {
        "h11": H11,
        "h21": H21,
        "polytope_index": POLYTOPE_INDEX,
        "identity": polytope_id,
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_source": manifest.get("source"),
        "vertices": vertices.tolist(),
        "reflexive": bool(poly.is_reflexive()),
        "favorable_n": bool(poly.is_favorable(lattice="N")),
        "lattice_points": len(poly.points()),
        "triangulated_points": len(poly.labels_not_facet),
        "facets": len(poly.facets()),
        "two_faces": len(poly.faces(2)),
        "canonical_lattice_points_sha256": package_generator._sha256_json(
            canonical_points
        ),
    }


def build_parser():
    """Construct a package-compatible h11=491 geometry-generation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h11", type=int, default=H11, help="Fixed target Hodge number (must be 491)."
    )
    parser.add_argument(
        "--manifest",
        "--polytope-manifest",
        dest="manifest",
        default=DEFAULT_MANIFEST,
        help="Local KS manifest containing the unique h11=491 polytope.",
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--report",
        default=None,
        help="Run-report path (default: OUTDIR/report.json).",
    )
    parser.add_argument(
        "--n",
        "--target-frsts",
        dest="target_frsts",
        type=int,
        default=1,
        help="Number of accepted full CY3 HDF5 geometries to retain.",
    )
    parser.add_argument(
        "--proposal-budget",
        "--max-tip-attempts",
        dest="proposal_budget",
        type=int,
        default=300,
        help="Bound on sampler proposals or explicit NTFE extensions.",
    )
    parser.add_argument(
        "--sampling-scheme",
        "--sampler",
        dest="sampling_scheme",
        choices=package_generator.SAMPLING_SCHEMES,
        default="ntfe_fast",
        help="FRST proposal family; ntfe_fast is the efficient default.",
    )
    parser.add_argument(
        "--backend", choices=("cgal", "qhull"), default="cgal", help="CYTools backend."
    )
    parser.add_argument(
        "--cores",
        "--workers",
        dest="cores",
        type=int,
        default=1,
        help="Worker count; this single-polytope writer currently requires 1.",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-retries", type=int, default=50)
    parser.add_argument("--n-walk", type=int, default=None)
    parser.add_argument("--n-flip", type=int, default=None)
    parser.add_argument("--initial-walk-steps", type=int, default=None)
    parser.add_argument("--fine-tune-steps", type=int, default=8)
    parser.add_argument("--walk-step-size", type=float, default=1e-2)
    parser.add_argument("--max-steps-to-wall", type=int, default=25)
    parser.add_argument("--fast-height-scale", type=float, default=0.2)
    parser.add_argument(
        "--ntfe-face-sampler",
        choices=package_generator.NTFE_FACE_SAMPLERS,
        default="fast",
    )
    parser.add_argument("--ntfe-max-face-points", type=int, default=0)
    parser.add_argument("--ntfe-face-pool-size", type=int, default=5)
    parser.add_argument(
        "--exact-proposals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For NTFE modes, count explicit two-face extensions (default); "
            "--no-exact-proposals delegates proposal counting to CYTools."
        ),
    )
    parser.add_argument(
        "--gnn-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="dualGNN device override for gnn_ntfe or dualgnn face sampling.",
    )

    # These options match the full package writer.  They are not applied to
    # the FRST proposal; they control the CY3/Kähler/potential artifact after
    # a candidate has been selected.
    parser.add_argument("--max-m", type=float, default=1_000_000.0)
    parser.add_argument("--max-kaehler-attempts", type=int, default=100)
    parser.add_argument("--min-divisor-volume", type=float, default=1.0)
    parser.add_argument("--min-prime-divisor-volume", type=float, default=1.0)
    parser.add_argument("--qcd-volume-min", type=float, default=25.0)
    parser.add_argument("--qcd-volume-max", type=float, default=40.0)
    parser.add_argument(
        "--moduli-policy", choices=("adaptive", "canonical_qcd"), default="adaptive"
    )
    parser.add_argument("--qcd-volume-target", type=float, default=40.0)
    parser.add_argument("--qcd-divisor-index", type=int, default=None)
    parser.add_argument(
        "--visible-sector-policy", choices=("none", "intersecting_d7"), default="none"
    )
    parser.add_argument("--qed-divisor-index", type=int, default=None)
    parser.add_argument("--orientifold-file", type=str, default=None)
    parser.add_argument("--export-kahler-rays", action="store_true")
    parser.add_argument(
        "--ks-database-version",
        default="local h11=491 manifest",
        help="Database/manifest label recorded in each HDF5 construction metadata record.",
    )
    parser.add_argument(
        "--wall-clock-seconds",
        type=float,
        default=None,
        help="Optional hard wall-clock cap that preserves a partial report.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing cyax.h5 output slots."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and write the planned run metadata without sampling.",
    )
    return parser


def validate_args(args):
    """Validate h11=491 and full-geometry option contracts."""
    if args.h11 != H11:
        raise ValueError("--h11 is fixed at 491 for this script.")
    if args.target_frsts < 1:
        raise ValueError("--n/--target-frsts must be positive.")
    if args.proposal_budget < 1:
        raise ValueError("--proposal-budget/--max-tip-attempts must be positive.")
    if args.cores != 1:
        raise ValueError(
            "This writer handles one high-h11 polytope per process; use --cores 1."
        )
    if args.max_retries < 1:
        raise ValueError("--max-retries must be positive.")
    if args.ntfe_max_face_points < 0:
        raise ValueError("--ntfe-max-face-points cannot be negative.")
    if args.ntfe_face_pool_size < 1:
        raise ValueError("--ntfe-face-pool-size must be positive.")
    if args.fine_tune_steps < 1 or args.max_steps_to_wall < 1:
        raise ValueError("Fair sampler step counts must be positive.")
    if args.walk_step_size <= 0.0 or args.fast_height_scale <= 0.0:
        raise ValueError("--walk-step-size and --fast-height-scale must be positive.")
    if args.exact_proposals and args.sampling_scheme not in {"ntfe_fast", "gnn_ntfe"}:
        raise ValueError("--exact-proposals is only applicable to NTFE modes.")
    if args.max_m <= 0.0 or args.max_kaehler_attempts < 1:
        raise ValueError("--max-m and --max-kaehler-attempts must be positive.")
    if args.min_divisor_volume <= 0.0 or args.min_prime_divisor_volume <= 0.0:
        raise ValueError("Divisor-volume lower bounds must be positive.")
    if args.qcd_volume_min <= 0.0 or args.qcd_volume_max < args.qcd_volume_min:
        raise ValueError("QCD volume bounds must be positive and ordered.")
    if args.qcd_volume_target <= 0.0:
        raise ValueError("--qcd-volume-target must be positive.")
    if args.qcd_divisor_index is not None and args.qcd_divisor_index < 0:
        raise ValueError("--qcd-divisor-index must be non-negative.")
    if args.qcd_divisor_index is not None and args.moduli_policy != "canonical_qcd":
        raise ValueError("--qcd-divisor-index requires --moduli-policy canonical_qcd.")
    if args.qed_divisor_index is not None and args.qed_divisor_index < 0:
        raise ValueError("--qed-divisor-index must be non-negative.")
    if args.visible_sector_policy == "intersecting_d7" and args.orientifold_file is None:
        raise ValueError(
            "--visible-sector-policy intersecting_d7 requires --orientifold-file."
        )
    if args.qed_divisor_index is not None and args.visible_sector_policy != "intersecting_d7":
        raise ValueError(
            "--qed-divisor-index requires --visible-sector-policy intersecting_d7."
        )
    if args.export_kahler_rays and args.dry_run:
        # This is only informational; the real export remains opt-in.
        pass
    for name, value in (
        ("--n-walk", args.n_walk),
        ("--n-flip", args.n_flip),
        ("--initial-walk-steps", args.initial_walk_steps),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive.")
    if args.wall_clock_seconds is not None and args.wall_clock_seconds <= 0.0:
        raise ValueError("--wall-clock-seconds must be positive.")


def environment_metadata():
    """Record versions, hardware, thread controls, and the CYTools cache."""
    return {
        "cytools_version": getattr(cytools, "version", None),
        "dualgnn_version": sampler_probe.package_version("dualgnn"),
        "torch_version": sampler_probe.package_version("torch"),
        "numpy_version": sampler_probe.package_version("numpy"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        "numba_disable_caching": os.environ.get("NUMBA_DISABLE_CACHING"),
        "cytools_cache_directory": CYTOOLS_CACHE,
    }


def exact_state():
    """Return mutable counters used by the explicit NTFE proposal path."""
    return {
        "attempted_proposals": 0,
        "yielded_triangulations": 0,
        "non_solid_attempts": 0,
        "solid_attempts": 0,
        "invalid_extensions": 0,
        "duplicate_two_face_classes": 0,
        "seen_two_face_hashes": set(),
        "two_face_combination_hashes": [],
        "extension_errors": [],
        "pool_decomposition": None,
        "terminal_status": None,
    }


def candidate_stream(args, poly, state):
    """Create the selected bounded FRST proposal stream."""
    if args.exact_proposals:
        face_sampler = (
            "dualgnn"
            if args.sampling_scheme == "gnn_ntfe"
            else args.ntfe_face_sampler
        )
        return sampler_probe.exact_ntfe_candidates(
            poly,
            face_sampler,
            args.proposal_budget,
            args.seed,
            args.ntfe_max_face_points,
            args.ntfe_face_pool_size,
            state,
        )
    return package_generator.triangulation_candidates(
        poly,
        args.sampling_scheme,
        args.proposal_budget,
        args.max_retries,
        args.backend,
        args.seed,
        args.n_walk,
        args.n_flip,
        args.initial_walk_steps,
        args.fine_tune_steps,
        args.walk_step_size,
        args.max_steps_to_wall,
        args.fast_height_scale,
        args.ntfe_face_sampler,
        args.ntfe_max_face_points,
        args.ntfe_face_pool_size,
    )


def sampling_metadata(args, poly):
    """Construct the same sampler metadata embedded by the package writer."""
    n_points = len(poly.points())
    n_walk = n_points // 10 + 10 if args.n_walk is None else args.n_walk
    n_flip = n_points // 10 + 10 if args.n_flip is None else args.n_flip
    initial_walk_steps = (
        2 * n_points // 10 + 10
        if args.initial_walk_steps is None
        else args.initial_walk_steps
    )
    mosek_license = package_generator.configure_mosek_license()
    if args.sampling_scheme in {"ntfe_fast", "gnn_ntfe"}:
        selection_status = "direct_ntfe_proposal_with_finite_face_pool"
        sampling_unit = "two_face_inequivalent_frst"
    elif args.sampling_scheme == "fair":
        selection_status = "provisionally_fair_frst_markov_chain"
        sampling_unit = "frst"
    else:
        selection_status = "biased_random_height_proposal"
        sampling_unit = "frst"
    return {
        "scheme": args.sampling_scheme,
        "backend": args.backend,
        "seed": args.seed,
        "N_walk": n_walk,
        "N_flip": n_flip,
        "initial_walk_steps": initial_walk_steps,
        "fine_tune_steps": args.fine_tune_steps,
        "walk_step_size": args.walk_step_size,
        "max_steps_to_wall": args.max_steps_to_wall,
        "max_retries": args.max_retries,
        "fast_height_scale": args.fast_height_scale,
        "sampling_unit": sampling_unit,
        "selection_status": selection_status,
        "proposal_budget": args.proposal_budget,
        "exact_proposals": bool(args.exact_proposals),
        "ntfe_face_sampler": args.ntfe_face_sampler,
        "ntfe_max_face_points": args.ntfe_max_face_points,
        "ntfe_face_pool_size": args.ntfe_face_pool_size,
        "qp_solver_preference": "mosek_if_licensed_then_available",
        "mosek_license_configured": mosek_license["configured"],
        "mosek_license_activated": mosek_license["activated"],
    }


def existing_output_indices(outdir):
    """Find the contiguous accepted-output prefix used by the package layout."""
    indices = []
    index = 1
    while os.path.exists(
        package_generator.output_path(outdir, H11, POLYTOPE_INDEX, index)
    ):
        indices.append(index)
        index += 1
    return indices


def make_report(args, poly_metadata_value, outdir, report_path, orientifold):
    direct_ntfe = args.sampling_scheme in {"ntfe_fast", "gnn_ntfe"}
    return {
        "schema_version": f"{SCHEMA_VERSION}-h11-491-run",
        "target_population": (
            "full CY3 geometries from two-face-inequivalent FRSTs of the unique "
            "favorable KS h11=491 polytope"
            if direct_ntfe
            else "full CY3 geometries from FRSTs of the unique favorable KS h11=491 polytope"
        ),
        "realised_sample": (
            "bounded local-manifest HDF5 generation with explicit CY3, Kähler, "
            "potential, and physical-selection outcomes; not population-representative"
        ),
        "construction": {
            "h11": H11,
            "h21": H21,
            "hdf5_schema": SCHEMA_VERSION,
            "ks_database_version": args.ks_database_version,
            "writer": "generate_geometric_data_multitriangulation.generate_and_save_geometry",
            "physical_selection": True,
            "moduli_policy": args.moduli_policy,
            "visible_sector_policy": args.visible_sector_policy,
            "orientifold": package_generator._jsonable(orientifold),
        },
        "sampling": {
            "scheme": args.sampling_scheme,
            "sampling_unit": "two_face_inequivalent_frst" if direct_ntfe else "frst",
            "seed": args.seed,
            "backend": args.backend,
            "target_frsts_requested": args.target_frsts,
            "proposal_budget_requested": args.proposal_budget,
            "proposal_budget_semantics": (
                "explicit face-combination extension attempts"
                if args.exact_proposals
                else "CYTools sampler N request; not an explicit extension-attempt count"
            ),
            "exact_proposals": bool(args.exact_proposals),
            "ntfe_face_sampler": args.ntfe_face_sampler,
            "ntfe_max_face_points": args.ntfe_max_face_points,
            "ntfe_face_pool_size": args.ntfe_face_pool_size,
        },
        "selection": {
            "max_m": args.max_m,
            "max_kaehler_attempts": args.max_kaehler_attempts,
            "min_divisor_volume": args.min_divisor_volume,
            "min_prime_divisor_volume": args.min_prime_divisor_volume,
            "qcd_volume_window": [args.qcd_volume_min, args.qcd_volume_max],
            "qcd_volume_target": args.qcd_volume_target,
            "qcd_divisor_index": args.qcd_divisor_index,
            "qed_divisor_index": args.qed_divisor_index,
            "export_kahler_rays": args.export_kahler_rays,
        },
        "polytope": poly_metadata_value,
        "environment": environment_metadata(),
        "source": sampler_probe.git_metadata(),
        "outputs": {
            "outdir": os.path.abspath(outdir),
            "report": os.path.abspath(report_path),
            "geometry_root": f"h11_{H11:03d}/np_{POLYTOPE_INDEX:07d}",
            "geometry_file": "cy_XXXXXXX/cyax.h5",
            "hdf5_groups": [
                "cytools/geometric",
                "cytools/potential",
                "construction_metadata",
            ],
        },
        "counts": {
            "existing_geometries": 0,
            "yielded_triangulations": 0,
            "attempted_candidates": 0,
            "valid_frsts": 0,
            "accepted_geometries": 0,
            "rejected_frsts": 0,
            "candidate_errors": 0,
            "duplicate_full_triangulations": 0,
            "duplicate_two_face_classes": 0,
            "written_hdf5": 0,
        },
        "candidates": [],
        "terminal_status": None,
    }


def generate_candidate(
    args,
    poly,
    poly_points,
    polytope_id,
    sampling_info,
    triangulation,
    candidate_index,
    output_index,
    outdir,
    orientifold_config,
    seen_full,
    seen_two_face,
    report,
):
    """Run the package CY3 writer for one FRST and return an audit record."""
    started = time.perf_counter()
    cpu_started = sampler_probe.resource_snapshot()["cpu_seconds"]
    candidate = {"candidate_index": candidate_index, "output_index": output_index}
    triangulation_hash, simplex_shape = sampler_probe.canonical_triangulation_hash(
        triangulation
    )
    candidate["triangulation_sha256"] = triangulation_hash
    candidate["simplices_shape"] = simplex_shape
    try:
        candidate["two_face_sha256"] = sampler_probe.canonical_two_face_hash(
            triangulation
        )
    except Exception as exc:  # pragma: no cover - CYTools-specific fallback
        candidate["two_face_sha256"] = None
        candidate["two_face_hash_error"] = type(exc).__name__
    report["counts"]["valid_frsts"] += 1
    if triangulation_hash in seen_full:
        report["counts"]["duplicate_full_triangulations"] += 1
        candidate["terminal_status"] = "duplicate_full_triangulation"
        return candidate
    seen_full.add(triangulation_hash)
    if candidate.get("two_face_sha256") is not None:
        if candidate["two_face_sha256"] in seen_two_face:
            report["counts"]["duplicate_two_face_classes"] += 1
        seen_two_face.add(candidate["two_face_sha256"])

    filepath = package_generator.output_path(
        outdir, H11, POLYTOPE_INDEX, output_index
    )
    candidate["h5_path"] = os.path.relpath(filepath, outdir)
    simplices = np.asarray(triangulation.simplices(), dtype=int)

    def report_message(message):
        if args.verbose:
            elapsed = time.perf_counter() - started
            print(
                f"np_{POLYTOPE_INDEX:07d} [{elapsed:8.1f}s]: {message}",
                flush=True,
            )

    try:
        package_generator.generate_and_save_geometry(
            H11,
            triangulation.get_cy(),
            poly_points,
            simplices,
            filepath,
            args.max_m,
            args.max_kaehler_attempts,
            args.min_divisor_volume,
            args.min_prime_divisor_volume,
            args.qcd_volume_min,
            args.qcd_volume_max,
            args.moduli_policy,
            args.qcd_volume_target,
            args.qcd_divisor_index,
            args.visible_sector_policy,
            args.qed_divisor_index,
            np.random.default_rng(args.seed + candidate_index - 1),
            report_message,
            poly=poly,
            triangulation=triangulation,
            polytope_id=polytope_id,
            sampling_metadata=sampling_info,
            ks_database_version=args.ks_database_version,
            orientifold_config=orientifold_config,
            export_kahler_rays=args.export_kahler_rays,
        )
    except EXPECTED_REJECTIONS as exc:
        report["counts"]["rejected_frsts"] += 1
        candidate["terminal_status"] = "rejected_frst"
        candidate["rejection_type"] = type(exc).__name__
        candidate["rejection_reason"] = str(exc)[:1000]
        return candidate
    except Exception as exc:
        report["counts"]["candidate_errors"] += 1
        candidate["terminal_status"] = "geometry_generation_error"
        candidate["error_type"] = type(exc).__name__
        candidate["error"] = str(exc)[:1000]
        raise
    finally:
        candidate["cpu_seconds"] = (
            sampler_probe.resource_snapshot()["cpu_seconds"] - cpu_started
        )
        candidate["elapsed_seconds"] = time.perf_counter() - started

    report["counts"]["accepted_geometries"] += 1
    report["counts"]["written_hdf5"] += 1
    candidate["terminal_status"] = "accepted_geometry"
    candidate["h5_size_bytes"] = os.path.getsize(filepath)
    candidate["h5_groups"] = [
        "cytools/geometric",
        "cytools/potential",
        "construction_metadata",
    ]
    return candidate


def run_generation(args):
    """Generate full HDF5 geometries and atomically persist the run report."""
    validate_args(args)
    manifest, vertices, poly = load_h491_polytope(args.manifest)
    package_generator.require_cytools_capabilities(
        args.sampling_scheme, args.ntfe_face_sampler
    )
    if args.sampling_scheme == "gnn_ntfe" or args.ntfe_face_sampler == "dualgnn":
        device = sampler_probe.configure_gnn_device(args.gnn_device)
    else:
        device = {}
    orientifold = package_generator.load_orientifold(args.orientifold_file)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    report_path = os.path.abspath(args.report or os.path.join(outdir, "report.json"))
    poly_metadata_value = polytope_metadata(poly, vertices, args.manifest, manifest)
    report = make_report(args, poly_metadata_value, outdir, report_path, orientifold)
    report["device"] = device
    report["command"] = sys.argv
    report["sampling_metadata"] = sampling_metadata(args, poly)

    if args.dry_run:
        report["terminal_status"] = "dry_run"
        atomic_json_dump(report_path, report)
        return report

    existing = [] if args.overwrite else existing_output_indices(outdir)
    accepted = len(existing)
    next_output_index = 1 if args.overwrite else accepted + 1
    report["counts"]["existing_geometries"] = accepted
    if accepted >= args.target_frsts and not args.overwrite:
        report["terminal_status"] = "target_already_present"
        atomic_json_dump(report_path, report)
        return report

    exact = exact_state()
    seen_full = set()
    seen_two_face = set()
    poly_points = np.asarray(poly.points(), dtype=int)
    polytope_id, _ = package_generator.polytope_identity(poly)
    sampling_info = report["sampling_metadata"]
    started = time.perf_counter()
    process_started = sampler_probe.resource_snapshot()
    sampler_started = time.perf_counter()
    previous_alarm = None
    raised_error = None
    if args.wall_clock_seconds is not None:
        if not hasattr(signal, "setitimer"):
            raise RuntimeError("The wall-clock cap requires signal.setitimer on this platform.")

        def raise_resource_limit(signum, frame):
            del signum, frame
            raise sampler_probe.ResourceLimitExceeded("wall-clock cap reached")

        previous_alarm = signal.signal(signal.SIGALRM, raise_resource_limit)
        signal.setitimer(signal.ITIMER_REAL, args.wall_clock_seconds)

    try:
        candidates = candidate_stream(args, poly, exact)
        for candidate_index, triangulation in enumerate(candidates, start=1):
            report["counts"]["yielded_triangulations"] += 1
            report["counts"]["attempted_candidates"] += 1
            candidate = generate_candidate(
                args,
                poly,
                poly_points,
                polytope_id,
                sampling_info,
                triangulation,
                candidate_index,
                next_output_index,
                outdir,
                orientifold,
                seen_full,
                seen_two_face,
                report,
            )
            report["candidates"].append(candidate)
            if args.verbose:
                print(
                    f"candidate {candidate_index}: {candidate['terminal_status']} "
                    f"({candidate.get('elapsed_seconds', 0.0):.2f}s)",
                    flush=True,
                )
            if candidate["terminal_status"] == "accepted_geometry":
                accepted += 1
                next_output_index += 1
            if accepted >= args.target_frsts:
                break
        if accepted >= args.target_frsts:
            report["terminal_status"] = "target_reached"
        elif args.exact_proposals and exact["terminal_status"] is not None:
            report["terminal_status"] = exact["terminal_status"]
        elif not report["candidates"]:
            report["terminal_status"] = "sampler_retry_exhausted"
        elif report["counts"]["accepted_geometries"] == 0:
            report["terminal_status"] = "no_accepted_geometry"
        else:
            report["terminal_status"] = "proposal_budget_exhausted"
    except sampler_probe.ResourceLimitExceeded as exc:
        report["terminal_status"] = "resource_cap"
        report["termination_reason"] = "wall_clock_resource_cap"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    except KeyboardInterrupt as exc:
        report["terminal_status"] = "external_interrupt"
        report["termination_reason"] = "external_interrupt"
        report["error_type"] = type(exc).__name__
        report["error"] = "Generation interrupted before normal completion."
    except Exception as exc:
        raised_error = exc
        report["terminal_status"] = "sampler_or_cytools_error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:1000]
    finally:
        if args.wall_clock_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
        if args.exact_proposals:
            report["exact_budget"] = sampler_probe.exact_budget_payload(
                exact, args.proposal_budget
            )
            report["counts"]["duplicate_two_face_classes"] = max(
                report["counts"]["duplicate_two_face_classes"],
                exact["duplicate_two_face_classes"],
            )
        report["timing"] = {
            "sampler_iteration_seconds": time.perf_counter() - sampler_started,
            "total_seconds": time.perf_counter() - started,
        }
        finished = sampler_probe.resource_snapshot()
        report["resource"] = {
            "cpu_seconds": finished["cpu_seconds"] - process_started["cpu_seconds"],
            "peak_rss_bytes": finished["peak_rss_bytes"],
        }
        atomic_json_dump(report_path, report)

    if raised_error is not None:
        raise raised_error
    return report


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        report = run_generation(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(package_generator._jsonable(report), indent=2, sort_keys=True))
    print(f"\nWrote {report['outputs']['report']}")


if __name__ == "__main__":
    main()
