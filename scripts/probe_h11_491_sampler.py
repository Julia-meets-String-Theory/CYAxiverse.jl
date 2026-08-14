"""Run a bounded, reproducible h11=491 CYTools sampler probe.

The probe uses the published vertices of the unique favorable KS polytope with
Hodge numbers (491, 11).  It validates only the triangulation sampler by
default; ``--include-cy`` additionally constructs each CY hypersurface and
records its Hodge data.  It never writes an HDF5 geometry artifact.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time
from importlib import metadata, resources

import numpy as np
import platformdirs

_benchmark_cache = os.environ.get("CYAX_BENCH_CACHE")
if _benchmark_cache:
    os.makedirs(_benchmark_cache, exist_ok=True)
    platformdirs.user_cache_dir = lambda *args, **kwargs: _benchmark_cache

import cytools
from cytools import Polytope
from cytools.helpers.matrix import LIL_stack
from cytools.ntfe.ntfe import _find_interior_point_highs

from generate_geometric_data_multitriangulation import (
    NTFE_FACE_SAMPLERS,
    SAMPLING_SCHEMES,
    extract_topology,
    load_polytope_manifest,
    require_cytools_capabilities,
    triangulation_candidates,
)


class ResourceLimitExceeded(TimeoutError):
    """Signal that the benchmark-only wall-clock cap was reached."""


DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(__file__), "manifests", "h11_491_11_ks.json"
)
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def canonical_triangulation_hash(triangulation):
    """Hash a triangulation independently of simplex enumeration order."""
    simplices = np.asarray(triangulation.simplices(as_indices=True), dtype=np.int32)
    simplices = np.sort(simplices, axis=1)
    simplices = simplices[np.lexsort(simplices.T[::-1])]
    return hashlib.sha256(simplices.tobytes()).hexdigest(), list(simplices.shape)


def canonical_two_face_hash(triangulation):
    """Hash the set of two-face restrictions for duplicate accounting."""
    face_simplices = triangulation.simplices(
        on_faces_dim=2,
        split_by_face=True,
        as_np_array=False,
        as_indices=True,
    )
    canonical_faces = []
    for simplices in face_simplices:
        simplices = list(simplices)
        if simplices and isinstance(simplices[0], (set, frozenset)):
            simplices = [sorted(simplex) for simplex in simplices]
        array = np.asarray(simplices, dtype=np.int32)
        if array.size == 0:
            canonical_faces.append([])
            continue
        array = np.sort(array, axis=1)
        array = array[np.lexsort(array.T[::-1])]
        canonical_faces.append(array.tolist())
    payload = json.dumps(canonical_faces, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resource_snapshot():
    """Return process CPU time and a portable peak-RSS estimate."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    # Darwin reports bytes; Linux reports KiB.
    peak_rss_bytes = peak_rss if platform.system() == "Darwin" else peak_rss * 1024
    return {
        "cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "peak_rss_bytes": peak_rss_bytes,
    }


def canonical_face_choice_hash(face_triangs):
    """Hash a selected collection of two-face triangulations."""
    canonical_faces = []
    for triangulation in face_triangs:
        simplices = np.asarray(
            triangulation.simplices(as_indices=True), dtype=np.int32
        )
        if simplices.size == 0:
            canonical_faces.append([])
            continue
        simplices = np.sort(simplices, axis=1)
        simplices = simplices[np.lexsort(simplices.T[::-1])]
        canonical_faces.append(simplices.tolist())
    payload = json.dumps(canonical_faces, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_ntfe_candidates(
    poly,
    face_sampler,
    max_draws,
    seed,
    max_face_points,
    face_pool_size,
    state,
):
    """Yield NTFE extensions with an observable explicit proposal counter.

    The installed CYTools ``ntfe_frts`` API counts requested expanded cones,
    not the individual face-combination extension attempts.  This diagnostic
    path follows its local logic after building the same two-face FRT pools:
    select a face combination, stack its secondary-cone inequalities, find an
    interior height vector, and construct the star triangulation.  It does not
    patch CYTools or change the production adapter.
    """
    pool_started = time.perf_counter()
    pool_cpu_started = resource_snapshot()["cpu_seconds"]
    state["pool_decomposition"] = {
        "status": "in_progress",
        "face_sampler": face_sampler,
        "face_pool_size_requested": face_pool_size,
        "max_face_points": max_face_points,
        "pool_build_started": True,
    }
    face_pools = poly.face_triangs(
        dim=2,
        only_regular=True,
        max_npts=max_face_points,
        N_face_triangs=face_pool_size,
        triang_method=face_sampler,
        seed=seed,
        verbosity=0,
    )
    pool_wall = time.perf_counter() - pool_started
    pool_cpu = resource_snapshot()["cpu_seconds"] - pool_cpu_started
    pool_sizes = [len(pool) for pool in face_pools]
    state["pool_decomposition"] = {
        "face_count": len(face_pools),
        "face_pool_sizes": pool_sizes,
        "face_pool_size_requested": face_pool_size,
        "max_face_points": max_face_points,
        "face_sampler": face_sampler,
        "pool_build_wall_seconds": pool_wall,
        "pool_build_cpu_seconds": pool_cpu,
        "total_face_frt_count": int(sum(pool_sizes)),
    }
    if not face_pools or any(size == 0 for size in pool_sizes):
        state["terminal_status"] = "source_exhausted"
        state["attempted_proposals"] = 0
        return

    inequality_started = time.perf_counter()
    inequality_cpu_started = resource_snapshot()["cpu_seconds"]
    face_ineqs = poly.triangface_ineqs(
        face_triangs=face_pools,
        require_star=False,
        verbosity=0,
    )
    state["pool_decomposition"].update(
        {
            "inequality_build_wall_seconds": time.perf_counter() - inequality_started,
            "inequality_build_cpu_seconds": (
                resource_snapshot()["cpu_seconds"] - inequality_cpu_started
            ),
            "face_inequality_choice_counts": [len(block) for block in face_ineqs],
        }
    )
    choice_counts = [len(block) for block in face_ineqs]
    total_combinations = math.prod(choice_counts)
    state["pool_decomposition"]["total_face_combinations"] = int(total_combinations)
    if total_combinations == 0:
        state["terminal_status"] = "source_exhausted"
        state["attempted_proposals"] = 0
        return

    rng = np.random.default_rng(seed)
    seen_choices = set()
    n_draws = min(int(max_draws), int(total_combinations))
    for _ in range(n_draws):
        choice = tuple(int(rng.integers(count)) for count in choice_counts)
        while choice in seen_choices:
            choice = tuple(int(rng.integers(count)) for count in choice_counts)
        seen_choices.add(choice)
        state["attempted_proposals"] += 1
        selected_faces = [
            face_pools[face_index][choice[face_index]]
            for face_index in range(len(choice))
        ]
        selected_hash = canonical_face_choice_hash(selected_faces)
        if selected_hash in state["seen_two_face_hashes"]:
            state["duplicate_two_face_classes"] += 1
        state["seen_two_face_hashes"].add(selected_hash)
        state["two_face_combination_hashes"].append(selected_hash)
        hypers = LIL_stack(face_ineqs, choice, choice_counts)
        try:
            heights = _find_interior_point_highs(
                hypers, len(poly.labels), c=1
            )
        except Exception as exc:
            state["extension_errors"].append(
                {"type": type(exc).__name__, "message": str(exc)[:300]}
            )
            continue
        if heights is None:
            state["non_solid_attempts"] += 1
            continue
        state["solid_attempts"] += 1
        try:
            triangulation = poly.triangulate(heights=heights, make_star=True)
        except Exception as exc:
            state["extension_errors"].append(
                {"type": type(exc).__name__, "message": str(exc)[:300]}
            )
            continue
        if triangulation is None or not triangulation:
            state["invalid_extensions"] += 1
            continue
        state["yielded_triangulations"] += 1
        yield triangulation

    state["terminal_status"] = (
        "source_exhausted" if n_draws < int(max_draws) else "completed"
    )


def exact_budget_payload(exact_state, requested_proposals):
    """Serialize explicit-harness counters, including partial capped runs."""
    return {
        "requested_proposals": requested_proposals,
        "attempted_proposals": exact_state["attempted_proposals"],
        "yielded_triangulations": exact_state["yielded_triangulations"],
        "non_solid_attempts": exact_state["non_solid_attempts"],
        "solid_attempts": exact_state["solid_attempts"],
        "invalid_extensions": exact_state["invalid_extensions"],
        "duplicate_two_face_classes": exact_state["duplicate_two_face_classes"],
        "extension_errors": exact_state["extension_errors"],
        "two_face_combination_hashes": exact_state["two_face_combination_hashes"],
        "pool_decomposition": exact_state["pool_decomposition"],
    }


def package_version(name):
    """Return an installed distribution version without making it required."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def configure_gnn_device(requested):
    """Select or override dualGNN's default device for this probe only."""
    try:
        import torch
        from dualgnn import device as dualgnn_device
        from dualgnn import model as dualgnn_model
    except ImportError:
        return {
            "requested": requested,
            "selected": None,
            "error": "dualgnn_or_torch_import_failed",
        }

    selected = dualgnn_device.default_device() if requested == "auto" else requested
    if requested != "auto":
        dualgnn_device.default_device = lambda: selected
        dualgnn_model.default_device = lambda: selected
    return {
        "requested": requested,
        "selected": str(selected),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "checkpoint": str(resources.files("dualgnn") / "ckpts" / "reinforce.pt"),
    }


def git_metadata():
    """Record source revision and status without mutating the checkout."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"revision": revision, "status_short": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)[:300]}


def atomic_json_dump(path, payload):
    """Write one complete sampler report without exposing a partial JSON file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=".h11_491_sampler-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="JSON report path.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--sampler", choices=SAMPLING_SCHEMES, default="ntfe_fast")
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument(
        "--proposal-budget",
        type=int,
        default=None,
        help=(
            "Number of sampler proposals passed to CYTools. When set, the "
            "probe consumes the full yielded stream unless --accepted-target "
            "is also set."
        ),
    )
    parser.add_argument(
        "--accepted-target",
        type=int,
        default=None,
        help="Stop after this many valid FRSTs; zero consumes the full proposal budget.",
    )
    parser.add_argument(
        "--exact-proposals",
        action="store_true",
        help=(
            "Use the benchmark-only explicit face-combination harness for "
            "ntfe_fast or gnn_ntfe. This counts extension attempts directly."
        ),
    )
    parser.add_argument(
        "--wall-clock-seconds",
        type=float,
        default=None,
        help="Optional benchmark-only wall-clock cap that still writes an atomic report.",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--backend", choices=("cgal", "qhull"), default="cgal")
    parser.add_argument("--ntfe-face-sampler", choices=NTFE_FACE_SAMPLERS, default="fast")
    parser.add_argument("--ntfe-max-face-points", type=int, default=0)
    parser.add_argument("--ntfe-face-pool-size", type=int, default=5)
    parser.add_argument("--include-cy", action="store_true")
    parser.add_argument(
        "--gnn-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Override dualGNN's default device for this probe.",
    )
    parser.add_argument(
        "--include-topology",
        action="store_true",
        help=(
            "Also extract the serializable CY3 topology/cone boundary used by "
            "the HDF5 generator, without materialising its dense potential matrix."
        ),
    )
    args = parser.parse_args()

    if args.candidate_count < 1:
        parser.error("--candidate-count must be positive")
    if args.proposal_budget is not None and args.proposal_budget < 1:
        parser.error("--proposal-budget must be positive")
    if args.accepted_target is not None and args.accepted_target < 0:
        parser.error("--accepted-target cannot be negative")
    if args.ntfe_max_face_points < 0:
        parser.error("--ntfe-max-face-points cannot be negative")
    if args.ntfe_face_pool_size < 1:
        parser.error("--ntfe-face-pool-size must be positive")
    if args.include_topology and not args.include_cy:
        parser.error("--include-topology requires --include-cy")
    if args.exact_proposals and args.sampler not in {"ntfe_fast", "gnn_ntfe"}:
        parser.error("--exact-proposals requires ntfe_fast or gnn_ntfe")
    if args.wall_clock_seconds is not None and args.wall_clock_seconds <= 0:
        parser.error("--wall-clock-seconds must be positive")

    proposal_budget = (
        args.proposal_budget if args.proposal_budget is not None else args.candidate_count
    )
    accepted_target = args.accepted_target
    if accepted_target is None:
        accepted_target = 0 if args.proposal_budget is not None else args.candidate_count
    manifest = load_polytope_manifest(args.manifest)
    vertices_list = manifest["by_h11"].get(491, [])
    if len(vertices_list) != 1:
        raise RuntimeError("The h11=491 manifest must contain exactly one polytope.")
    gnn_device = configure_gnn_device(args.gnn_device) if args.sampler == "gnn_ntfe" else {}
    require_cytools_capabilities(args.sampler, args.ntfe_face_sampler)

    started = time.perf_counter()
    poly = Polytope(vertices_list[0], deterministic_glsm_basis=True)
    expected = (491, 11)
    observed = (int(poly.h11()), int(poly.h21()))
    if observed != expected:
        raise RuntimeError(f"Manifest does not reconstruct hodge data {expected}: {observed}.")

    report = {
        "schema_version": "cyaxiverse-h11-491-sampler-probe-v2",
        "source": manifest.get("source"),
        "target_population": (
            "two-face-inequivalent FRSTs of the unique favorable KS h11=491 polytope"
            if args.sampler in {"ntfe_fast", "gnn_ntfe"}
            else "FRSTs of the unique favorable KS h11=491 polytope"
        ),
        "realised_sample": "bounded sampler probe; not a population-representative sample",
        "sampling": {
            "scheme": args.sampler,
            "sampling_unit": (
                "two_face_inequivalent_frst"
                if args.sampler in {"ntfe_fast", "gnn_ntfe"}
                else "frst"
            ),
            "ntfe_face_sampler": args.ntfe_face_sampler,
            "ntfe_max_face_points": args.ntfe_max_face_points,
            "ntfe_face_pool_size": args.ntfe_face_pool_size,
            "candidate_count_requested": args.candidate_count,
            "proposal_budget_requested": proposal_budget,
            "accepted_target": accepted_target,
            "seed": args.seed,
            "backend": args.backend,
            "exact_proposal_harness": bool(args.exact_proposals),
            "proposal_budget_semantics": (
                "explicit face-combination extension attempts"
                if args.exact_proposals
                else "CYTools sampler N request; not an explicit extension-attempt count"
            ),
        },
        "environment": {
            "cytools_version": cytools.version,
            "dualgnn_version": package_version("dualgnn"),
            "torch_version": package_version("torch"),
            "numpy_version": package_version("numpy"),
            "qpsolvers_version": package_version("qpsolvers"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
            "numba_disable_caching": os.environ.get("NUMBA_DISABLE_CACHING"),
            "cache_directory": os.environ.get("CYAX_BENCH_CACHE"),
        },
        "device": gnn_device,
        "source": git_metadata(),
        "polytope": {
            "h11": observed[0],
            "h21": observed[1],
            "reflexive": bool(poly.is_reflexive()),
            "favorable": bool(poly.is_favorable(lattice="N")),
            "lattice_points": len(poly.points()),
            "triangulated_points": len(poly.labels_not_facet),
            "two_faces": len(poly.faces(2)),
            "facets": len(poly.facets()),
        },
        "candidates": [],
        "counts": {
            "yielded_triangulations": 0,
            "valid_frsts": 0,
            "invalid_frsts": 0,
            "duplicate_full_triangulations": 0,
            "duplicate_two_face_classes": 0,
        },
        "terminal_status": None,
    }
    exact_state = {
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
    try:
        import psutil

        report["environment"]["memory_bytes_total"] = int(psutil.virtual_memory().total)
    except ImportError:
        report["environment"]["memory_bytes_total"] = None

    process_started = resource_snapshot()
    sampler_started = time.perf_counter()
    seen_full = set()
    seen_two_face = set()
    first_valid_topology_done = False
    previous_alarm = None
    if args.wall_clock_seconds is not None:
        if not hasattr(signal, "setitimer"):
            raise RuntimeError("The wall-clock cap requires signal.setitimer on this platform.")

        def raise_resource_limit(signum, frame):
            del signum, frame
            raise ResourceLimitExceeded("benchmark wall-clock cap reached")

        previous_alarm = signal.signal(signal.SIGALRM, raise_resource_limit)
        signal.setitimer(signal.ITIMER_REAL, args.wall_clock_seconds)
    try:
        if args.exact_proposals:
            candidates = exact_ntfe_candidates(
                poly,
                "dualgnn" if args.sampler == "gnn_ntfe" else args.ntfe_face_sampler,
                proposal_budget,
                args.seed,
                args.ntfe_max_face_points,
                args.ntfe_face_pool_size,
                exact_state,
            )
        else:
            candidates = triangulation_candidates(
                poly,
                args.sampler,
                proposal_budget,
                50,
                args.backend,
                args.seed,
                None,
                None,
                None,
                8,
                1e-2,
                25,
                0.2,
                args.ntfe_face_sampler,
                args.ntfe_max_face_points,
                args.ntfe_face_pool_size,
            )
        previous = time.perf_counter()
        for candidate_index, triangulation in enumerate(candidates, start=1):
            candidate_started = time.perf_counter()
            candidate_cpu_started = resource_snapshot()["cpu_seconds"]
            triangulation_hash, simplex_shape = canonical_triangulation_hash(triangulation)
            checks = {
                "fine": bool(triangulation.is_fine()),
                "regular": bool(triangulation.is_regular()),
                "star": bool(triangulation.is_star()),
                "valid": bool(triangulation.is_valid()),
            }
            candidate = {
                "candidate_index": candidate_index,
                "sampler_latency_seconds": candidate_started - previous,
                "frst_validation_seconds": time.perf_counter() - candidate_started,
                "frst_validation": checks,
                "triangulation_sha256": triangulation_hash,
                "simplices_shape": simplex_shape,
                "terminal_status": "accepted_frst" if all(checks.values()) else "invalid_frst",
            }
            candidate["cpu_seconds"] = resource_snapshot()["cpu_seconds"] - candidate_cpu_started
            candidate["two_face_sha256"] = None
            try:
                candidate["two_face_sha256"] = canonical_two_face_hash(triangulation)
            except Exception as exc:
                candidate["two_face_hash_error"] = type(exc).__name__
            report["counts"]["yielded_triangulations"] += 1
            if triangulation_hash in seen_full:
                report["counts"]["duplicate_full_triangulations"] += 1
            seen_full.add(triangulation_hash)
            if candidate["two_face_sha256"] is not None:
                if candidate["two_face_sha256"] in seen_two_face:
                    report["counts"]["duplicate_two_face_classes"] += 1
                seen_two_face.add(candidate["two_face_sha256"])
            if all(checks.values()):
                report["counts"]["valid_frsts"] += 1
            else:
                report["counts"]["invalid_frsts"] += 1
            if args.include_cy and all(checks.values()) and not first_valid_topology_done:
                cy_started = time.perf_counter()
                cy = triangulation.get_cy()
                candidate["cy_construction_seconds"] = time.perf_counter() - cy_started
                candidate["cy_h11"] = int(cy.h11())
                candidate["cy_h21"] = int(cy.h21())
                candidate["cy_smooth"] = bool(cy.is_smooth())
                candidate["terminal_status"] = (
                    "cy3_topology_verified"
                    if (candidate["cy_h11"], candidate["cy_h21"]) == expected
                    and candidate["cy_smooth"]
                    else "cy3_validation_failed"
                )
                first_valid_topology_done = True
            if args.include_topology and candidate.get("terminal_status") == "cy3_topology_verified":
                topology_started = time.perf_counter()
                topology = extract_topology(cy, triangulation, export_kahler_rays=False)
                candidate["topology_extraction_seconds"] = time.perf_counter() - topology_started
                candidate["topology"] = {
                    "basis_matrix_shape": list(topology["basis_matrix"].shape),
                    "kappa_coo_shape": list(topology["kappa"].shape),
                    "kappa_nonzero_entries": int(topology["kappa"].shape[0]),
                    "c2_shape": list(topology["c2"].shape),
                    "mori_cone_shape": list(topology["mori_cone"].shape),
                    "kahler_hyperplanes_shape": list(
                        topology["kahler_cone_hyperplanes"].shape
                    ),
                }
                candidate["terminal_status"] = "cy3_topology_and_cones_verified"
            report["candidates"].append(candidate)
            previous = time.perf_counter()
            if accepted_target and report["counts"]["valid_frsts"] >= accepted_target:
                break
        if args.exact_proposals:
            report["exact_budget"] = exact_budget_payload(
                exact_state, proposal_budget
            )
            report["counts"]["duplicate_two_face_classes"] = exact_state[
                "duplicate_two_face_classes"
            ]
            report["terminal_status"] = exact_state["terminal_status"]
            if report["terminal_status"] is None:
                report["terminal_status"] = "completed"
        elif not report["candidates"]:
            report["terminal_status"] = "sampler_retry_exhausted"
        elif report["counts"]["valid_frsts"] == 0:
            report["terminal_status"] = "invalid_frst"
        else:
            report["terminal_status"] = "completed"
    except ResourceLimitExceeded:
        report["terminal_status"] = "resource_cap"
        report["error_type"] = "ResourceLimitExceeded"
        report["error"] = (
            "Probe interrupted by the explicit wall-clock resource cap; "
            "the partial bounded run is retained in this atomic report."
        )
        report["termination_reason"] = "wall_clock_resource_cap"
    except KeyboardInterrupt:
        cap_reached = (
            args.wall_clock_seconds is not None
            and (time.perf_counter() - sampler_started) >= args.wall_clock_seconds
        )
        report["terminal_status"] = "resource_cap" if cap_reached else "sampler_or_cytools_error"
        report["error_type"] = "KeyboardInterrupt"
        report["error"] = (
            "Probe interrupted externally after the wall-clock cap."
            if cap_reached
            else "Probe interrupted externally before normal completion."
        )
        report["termination_reason"] = (
            "wall_clock_resource_cap" if cap_reached else "external_interrupt"
        )
    except Exception as exc:
        report["terminal_status"] = "sampler_or_cytools_error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:1000]
        raise
    finally:
        if args.wall_clock_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
        if args.exact_proposals and "exact_budget" not in report:
            report["exact_budget"] = exact_budget_payload(
                exact_state, proposal_budget
            )
        report["timing"] = {
            "sampler_iteration_seconds": time.perf_counter() - sampler_started,
            "cold_sampler_initialization_seconds": (
                report["candidates"][0]["sampler_latency_seconds"]
                if report["candidates"]
                else None
            ),
            "warm_sampler_seconds": sum(
                candidate["sampler_latency_seconds"]
                for candidate in report["candidates"][1:]
            ),
        }
        process_finished = resource_snapshot()
        report["resource"] = {
            "cpu_seconds": process_finished["cpu_seconds"] - process_started["cpu_seconds"],
            "peak_rss_bytes": process_finished["peak_rss_bytes"],
        }
        report["total_seconds"] = time.perf_counter() - started
        atomic_json_dump(args.report, report)


if __name__ == "__main__":
    main()
