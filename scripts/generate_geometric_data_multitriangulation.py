"""Generate CYTools data, including several FRSTs for rare polytopes.

``--n`` is the number of Calabi--Yau geometries requested for each h11, not
the number of distinct polytopes.  The script first fetches up to ``n``
favorable N-lattice polytopes.  If fewer exist (as for h11=491), it samples
several bounded, pseudorandom FRSTs of each available polytope instead of ever
attempting to enumerate all triangulations.

Each FRST has its own toric Kähler cone and stretched-cone tip.  A candidate is
saved only when its tip can be dilated to satisfy the control criterion of
arXiv:2309.01831, eq. (21).  Failed candidates are rejected and replaced by
new FRST samples up to a user-configurable attempt budget.
"""

import argparse
import itertools
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from cytools import Polytope, fetch_polytopes


class PrefactorCriterionNotMet(RuntimeError):
    """The current FRST's tip cannot satisfy the potential-control criterion."""


def generate_and_save_geometry(h11, cy, poly_points, simplices, filepath, max_m, report):
    """Compute the CYAxiverse datasets and write one HDF5 geometry file."""
    report("computing Hodge, GLSM, and divisor-basis data")
    h21 = int(cy.h21())
    glsm = np.asarray(cy.glsm_charge_matrix(include_origin=False), dtype=int)
    basis = np.asarray(cy.divisor_basis(), dtype=int)

    n_val, m_val = 1.0, 1.0
    report("finding the stretched Kähler-cone tip (this can be slow without Mosek)")
    tip = np.asarray(
        cy.toric_kahler_cone().tip_of_stretched_cone(math.sqrt(n_val)), dtype=float
    )
    report("computing effective-cone rays")
    qprime = np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    nq = qprime.shape[0]

    report("computing divisor volumes and inverse Kähler metric")
    div_vols = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
    python_basis = basis - 1
    tau0 = div_vols[python_basis]
    kinv0_raw = np.asarray(cy.compute_inverse_kahler_metric(tip), dtype=float)
    kinv0 = 0.5 * (kinv0_raw + kinv0_raw.T)
    tau, kinv = tau0.copy(), kinv0.copy()

    # The original script evaluates this condition in nested Python loops for
    # every 0.01 increment of m.  At h11=491 that dominates the runtime.  The
    # bilinear form and ray volumes only scale as m**4 and m**2, respectively,
    # so precompute their m=1 values and test all lower-triangular pairs in
    # NumPy instead.
    report(f"searching the stretched-cone prefactor over {nq * (nq - 1) // 2} ray pairs")
    lower_i, lower_j = np.tril_indices(nq, k=-1)
    bilinear0 = (qprime @ kinv0) @ qprime.T
    tauq0 = qprime @ tau0
    def prefactor_is_valid(candidate_m):
        """Evaluate the original pairwise criterion at one candidate value."""
        candidate_m2 = candidate_m**2
        candidate_tauq = candidate_m2 * tauq0
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate_lhs = np.abs(
                np.log(
                    np.abs(
                        math.pi
                        * candidate_m2**2
                        * bilinear0[lower_i, lower_j]
                    )
                )
                - 2
                * math.pi
                * (candidate_tauq[lower_i] + candidate_tauq[lower_j])
            )
            candidate_rhs = np.abs(
                np.log(np.abs(candidate_tauq)) - 2 * math.pi * candidate_tauq
            )
        return np.all(candidate_lhs > candidate_rhs[lower_i])

    # Doubling reaches the large-m regime in O(log(m)) checks.  Once it finds
    # a valid upper bound, binary refinement recovers the old 0.01 resolution.
    lower_m, upper_m = 1.0, 1.0
    if not prefactor_is_valid(upper_m):
        while upper_m < max_m:
            lower_m = upper_m
            upper_m = min(2.0 * upper_m, max_m)
            report(f"testing stretched-cone prefactor m={upper_m:.2f}")
            if prefactor_is_valid(upper_m):
                break
        else:
            raise PrefactorCriterionNotMet(
                f"The stretched-cone prefactor did not converge before m={max_m}. "
                "Reject this FRST and try a different tip."
            )

        if not prefactor_is_valid(upper_m):
            raise PrefactorCriterionNotMet(
                f"The stretched-cone prefactor did not converge before m={max_m}. "
                "Reject this FRST and try a different tip."
            )

        while upper_m - lower_m > 1e-2:
            midpoint = 0.5 * (lower_m + upper_m)
            if prefactor_is_valid(midpoint):
                upper_m = midpoint
            else:
                lower_m = midpoint
    m_val = upper_m
    m2 = m_val**2
    tau = m2 * tau0
    kinv = m2**2 * kinv0

    if np.min(tau) <= 1.0:
        n_val = 1.0 / np.min(tau)
        tip = math.sqrt(n_val) * tip
        div_vols = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
        tau = div_vols[python_basis]
        kinv_raw = np.asarray(cy.compute_inverse_kahler_metric(tip), dtype=float)
        kinv = 0.5 * (kinv_raw + kinv_raw.T)

    volume = float(cy.compute_cy_volume(tip))
    tip_prefactor = np.asarray([math.sqrt(n_val), m_val], dtype=float)

    report(f"building potential data from {nq} effective-cone rays")
    num_cross = nq * (nq - 1) // 2
    q = np.empty((nq + num_cross, h11), dtype=float)
    q[:nq] = qprime
    l2 = np.empty((num_cross, 2), dtype=float)
    idx = 0
    for i in range(nq - 1):
        qi = qprime[i]
        for j in range(i + 1, nq):
            qj = qprime[j]
            q[nq + idx] = qj - qi
            l2[idx] = [
                (math.pi * (qi @ (kinv @ qj)) + ((qi + qj) @ tau))
                * 8 * math.pi / volume**2,
                -2 * math.log10(math.e) * math.pi * ((qi @ tau) + (qj @ tau)),
            ]
            idx += 1

    l1 = np.empty((nq, 2), dtype=float)
    for j, qj in enumerate(qprime):
        q_tau = qj @ tau
        l1[j] = [
            (8 * math.pi / volume**2) * q_tau,
            -2 * math.log10(math.e) * math.pi * q_tau,
        ]
    l_raw = np.vstack((l1, l2))
    l = np.empty_like(l_raw)
    l[:, 0] = np.sign(l_raw[:, 0])
    l[:, 1] = np.log10(np.abs(l_raw[:, 0])) + l_raw[:, 1]

    report("writing HDF5 data")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as file:
        cytools_group = file.create_group("cytools")
        geometric = cytools_group.create_group("geometric")
        geometric.create_dataset("points", data=poly_points, compression="gzip", compression_opts=9)
        geometric.create_dataset("simplices", data=simplices, compression="gzip", compression_opts=9)
        geometric.create_dataset("h21", data=h21)
        geometric.create_dataset("glsm", data=glsm, compression="gzip", compression_opts=9)
        geometric.create_dataset("basis", data=basis, compression="gzip", compression_opts=9)
        geometric.create_dataset("tip", data=tip, compression="gzip", compression_opts=9)
        geometric.create_dataset("tip_prefactor", data=tip_prefactor, compression="gzip", compression_opts=9)
        geometric.create_dataset("CY_volume", data=volume)
        geometric.create_dataset("divisor_volumes", data=tau, compression="gzip", compression_opts=9)
        geometric.create_dataset("Kinv", data=kinv, compression="gzip", compression_opts=9)
        potential = cytools_group.create_group("potential")
        potential.create_dataset("L", data=l, compression="gzip", compression_opts=9)
        potential.create_dataset("Q", data=q, compression="gzip", compression_opts=9)


def output_path(base_dir, h11, polytope_index, triangulation_index):
    return os.path.join(
        base_dir,
        f"h11_{h11:03d}",
        f"np_{polytope_index:07d}",
        f"cy_{triangulation_index:07d}",
        "cyax.h5",
    )


def process_polytope(task):
    """Generate a bounded number of distinct FRSTs for one polytope."""
    (
        polytope_index,
        vertices,
        h11,
        requested,
        base_dir,
        seed,
        max_retries,
        max_tip_attempts,
        overwrite,
        max_m,
        verbose,
    ) = task
    try:
        started = time.perf_counter()

        def report(message):
            if verbose:
                elapsed = time.perf_counter() - started
                print(
                    f"np_{polytope_index:07d} [{elapsed:8.1f}s]: {message}",
                    flush=True,
                )

        # Vertices, rather than all lattice points, make reconstruction cheap for
        # the h11=491 N-lattice polytope (which has 680 lattice points).
        report("constructing polytope")
        poly = Polytope(vertices)
        points = np.asarray(poly.points(), dtype=int)

        # Output indices represent accepted geometries, not raw triangulation
        # attempts.  This lets a resumed scan retain prior successful samples.
        existing_indices = []
        index = 1
        if not overwrite:
            while os.path.exists(output_path(base_dir, h11, polytope_index, index)):
                existing_indices.append(index)
                index += 1
        accepted = len(existing_indices)
        next_output_index = index
        if accepted >= requested and not overwrite:
            return {
                "ok": True,
                "polytope_index": polytope_index,
                "requested": requested,
                "attempted": 0,
                "accepted": accepted,
                "saved": 0,
                "rejected": 0,
                "skipped": accepted,
            }

        # The deterministic triangulation is the first candidate.  Subsequent
        # candidates are streamed, so a large retry budget does not retain a
        # large list of triangulation objects in memory.
        report("constructing the initial FRST candidate")
        candidates = [poly.triangulate(backend="cgal")]
        if max_tip_attempts > 1:
            candidates = itertools.chain(
                candidates,
                poly.random_triangulations_fast(
                    N=max_tip_attempts - 1,
                    max_retries=max_retries,
                    backend="cgal",
                    as_list=False,
                    seed=seed,
                ),
            )

        saved = rejected = attempted = 0
        for triangulation in candidates:
            if accepted >= requested:
                break
            attempted += 1
            triangulation_index = next_output_index
            filepath = output_path(base_dir, h11, polytope_index, triangulation_index)
            report(
                f"testing FRST candidate {attempted}/{max_tip_attempts} "
                f"for accepted geometry {accepted + 1}/{requested}"
            )
            report(f"processing accepted-output slot {triangulation_index}")
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            try:
                generate_and_save_geometry(
                    h11,
                    triangulation.get_cy(),
                    points,
                    simplices,
                    filepath,
                    max_m,
                    report,
                )
            except PrefactorCriterionNotMet:
                rejected += 1
                report("rejected FRST: its tip fails the prefactor criterion")
                continue
            accepted += 1
            saved += 1
            next_output_index += 1
        return {
            "ok": True,
            "polytope_index": polytope_index,
            "requested": requested,
            "attempted": attempted,
            "accepted": accepted,
            "saved": saved,
            "rejected": rejected,
            "skipped": len(existing_indices),
        }
    except Exception as exc:
        return {"ok": False, "polytope_index": polytope_index, "error": repr(exc)}


def plan_tasks(
    h11,
    n_geometries,
    base_dir,
    seed,
    max_retries,
    max_tip_attempts,
    overwrite,
    max_m,
    verbose,
):
    """Fetch at most n polytopes and spread n requested geometries over them."""
    polytopes = list(
        fetch_polytopes(h11=h11, limit=n_geometries, lattice="N", favorable=True)
    )
    if not polytopes:
        return []

    # Give every available polytope one geometry, then distribute the remaining
    # requests round-robin as extra triangulations of those polytopes.
    counts = [1] * len(polytopes)
    for index in itertools.islice(itertools.cycle(range(len(polytopes))), n_geometries - len(polytopes)):
        counts[index] += 1

    return [
        (
            polytope_index,
            np.asarray(poly.vertices(), dtype=int),
            h11,
            count,
            base_dir,
            seed + polytope_index,
            max_retries,
            max_tip_attempts,
            overwrite,
            max_m,
            verbose,
        )
        for polytope_index, (poly, count) in enumerate(zip(polytopes, counts), start=1)
    ]


def run_batch(
    h11,
    n_geometries,
    base_dir,
    n_cores,
    seed,
    max_retries,
    max_tip_attempts,
    overwrite,
    max_m,
    verbose,
):
    tasks = plan_tasks(
        h11,
        n_geometries,
        base_dir,
        seed,
        max_retries,
        max_tip_attempts,
        overwrite,
        max_m,
        verbose,
    )
    if not tasks:
        print(f"No favorable N-lattice polytopes found for h11={h11}.")
        return 0

    print(
        f"Found {len(tasks)} favorable polytope(s); requesting {n_geometries} "
        f"geometry/triangulation output(s)."
    )
    saved = 0
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_polytope, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if not result["ok"]:
                print(f"ERROR np_{result['polytope_index']:07d}: {result['error']}")
                continue
            saved += result["saved"]
            print(
                f"np_{result['polytope_index']:07d}: accepted "
                f"{result['accepted']}/{result['requested']} geometries after "
                f"{result['attempted']} FRST attempts; saved {result['saved']}, "
                f"rejected {result['rejected']}, skipped {result['skipped']}."
            )
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate CYTools geometry data for Julia.")
    parser.add_argument("--h11_min", type=int, default=4, help="Starting h11 value.")
    parser.add_argument("--h11_max", type=int, default=4, help="Ending h11 value (inclusive).")
    parser.add_argument("--n", type=int, default=1, help="Target number of CY geometries per h11.")
    parser.add_argument("--outdir", type=str, default=".", help="Base directory for output data.")
    parser.add_argument("--cores", type=int, default=None, help="Worker count (default: all available).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible random triangulations.")
    parser.add_argument("--max-retries", type=int, default=50, help="Bound per-FRST search retries.")
    parser.add_argument(
        "--max-tip-attempts",
        type=int,
        default=50,
        help="Maximum FRST candidates tried per polytope before reporting a shortfall.",
    )
    parser.add_argument(
        "--max-m",
        type=float,
        default=1_000_000.0,
        help="Maximum stretched-cone prefactor before reporting a non-convergence.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cyax.h5 files.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-worker stages and elapsed times; recommended for large h11.",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be positive")
    if args.max_tip_attempts < 1:
        parser.error("--max-tip-attempts must be positive")
    if args.h11_max < args.h11_min:
        args.h11_max = args.h11_min
    os.makedirs(args.outdir, exist_ok=True)

    total_saved = 0
    for h11 in range(args.h11_min, args.h11_max + 1):
        print(f"\n>>> Processing h11={h11} <<<")
        total_saved += run_batch(
            h11,
            args.n,
            args.outdir,
            args.cores,
            args.seed,
            args.max_retries,
            args.max_tip_attempts,
            args.overwrite,
            args.max_m,
            args.verbose,
        )
    print(f"\nSaved {total_saved} geometry file(s).")


if __name__ == "__main__":
    main()
