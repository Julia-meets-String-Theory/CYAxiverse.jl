#!/usr/bin/env python3
"""Diagnose CYTools coordinate conventions for a high-h11 hypersurface.

This script is intentionally read-only with respect to CY geometry: it creates
one deterministic FRST and writes diagnostics only.  It does not sample tips,
search prefactors, or write CYAxiverse HDF5 output.
"""

import argparse
import json
import os

import numpy as np
import cytools
from cytools import fetch_polytopes


def stats(values):
    values = np.asarray(values, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"size": int(values.size), "finite": 0}
    return {
        "size": int(values.size),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "nonpositive": int(np.count_nonzero(finite <= 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11", type=int, default=491)
    parser.add_argument("--outdir", default="cytools_h11_diagnostic")
    parser.add_argument("--backend", default="cgal")
    parser.add_argument(
        "--exact-intersections",
        action="store_true",
        help="Also request CYTools exact sparse intersection numbers; this can be slow.",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.exact_intersections:
        # CYTools deliberately guards rational intersection reconstruction as
        # experimental. Enable it only for this explicitly requested run.
        cytools.config.enable_experimental_features()

    polytopes = list(fetch_polytopes(h11=args.h11, limit=1, lattice="N", favorable=True))
    if not polytopes:
        raise RuntimeError(f"No favorable N-lattice polytope found for h11={args.h11}.")

    poly = polytopes[0]
    triangulation = poly.triangulate(backend=args.backend)
    cy = triangulation.get_cy()
    tip = np.asarray(cy.toric_kahler_cone().tip_of_stretched_cone(1.0), dtype=float)

    # These are the three objects whose coordinate conventions must agree.
    raw_tau = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
    basis_tau = np.asarray(cy.compute_divisor_volumes(tip, in_basis=True), dtype=float)
    divisor_basis = np.asarray(cy.divisor_basis())
    divisor_basis_matrix = np.asarray(cy.divisor_basis(as_matrix=True))
    effective_rays = np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    effective_pairing = effective_rays @ basis_tau

    curve_volumes = np.asarray(cy.compute_curve_volumes(tip), dtype=float)
    mori_basis_rays = np.asarray(cy.toric_mori_cone(in_basis=True).rays(), dtype=float)
    kahler_hyperplanes = np.asarray(cy.toric_kahler_cone().hyperplanes(), dtype=float)
    kahler_slack = kahler_hyperplanes @ tip

    # With the default index basis these must agree.  If they do not, the
    # stored basis is a matrix/general basis and the raw indexing shortcut is
    # invalid.
    indexed_basis_tau = None
    if divisor_basis.ndim == 1 and np.issubdtype(divisor_basis.dtype, np.integer):
        # CYTools divisor indices include the canonical/origin index, while
        # compute_divisor_volumes() omits it.
        indexed_basis_tau = raw_tau[divisor_basis.astype(int) - 1]

    report = {
        "h11_requested": args.h11,
        "h11_computed": int(cy.h11()),
        "h21": int(cy.h21()),
        "n_polytope_points": int(len(poly.points())),
        "n_prime_toric_divisors": int(len(cy.prime_toric_divisors())),
        "tip": stats(tip),
        "curve_volumes": stats(curve_volumes),
        "kahler_hyperplane_slack": stats(kahler_slack),
        "raw_prime_divisor_volumes": stats(raw_tau),
        "basis_divisor_volumes": stats(basis_tau),
        "effective_rays": {"shape": list(effective_rays.shape)},
        "effective_ray_dot_basis_tau": stats(effective_pairing),
        "mori_basis_rays": {"shape": list(mori_basis_rays.shape)},
        "basis": {
            "indices_shape": list(divisor_basis.shape),
            "matrix_shape": list(divisor_basis_matrix.shape),
            "raw_indexed_basis_match_max_abs_error": (
                None
                if indexed_basis_tau is None
                else float(np.max(np.abs(indexed_basis_tau - basis_tau)))
            ),
        },
    }

    np.savez_compressed(
        os.path.join(args.outdir, "h11_diagnostic_arrays.npz"),
        tip=tip,
        raw_tau=raw_tau,
        basis_tau=basis_tau,
        divisor_basis=divisor_basis,
        divisor_basis_matrix=divisor_basis_matrix,
        effective_rays=effective_rays,
        effective_pairing=effective_pairing,
        curve_volumes=curve_volumes,
        mori_basis_rays=mori_basis_rays,
        kahler_hyperplanes=kahler_hyperplanes,
        kahler_slack=kahler_slack,
    )

    if args.exact_intersections:
        intersections = cy.intersection_numbers(
            in_basis=True, format="coo", exact_arithmetic=True
        )
        np.save(os.path.join(args.outdir, "basis_intersections_exact.npy"), intersections)
        report["exact_intersections"] = {"shape": list(np.asarray(intersections).shape)}

    report_path = os.path.join(args.outdir, "h11_diagnostic_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
