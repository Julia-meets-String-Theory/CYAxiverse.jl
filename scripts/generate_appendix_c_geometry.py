#!/usr/bin/env python3
"""Generate the exact Appendix C N=8 benchmark geometry and potential."""

import argparse
import math
import os

import h5py
import numpy as np
from cytools import Polytope


VERTICES = np.array(
    [
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [-1, -1, 1, 0],
        [-1, 1, -1, 0],
        [1, -1, -1, -1],
        [1, 1, 1, -1],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ],
    dtype=int,
)

EXPECTED_TAU = np.array([45, 17, 17, 14.5, 14.5, 15.5, 15.5, 25], dtype=float)
EXPECTED_K_EIGENVALUES = np.sort(
    np.array([8.20e-4, 6.35e-4, 5.97e-4, 3.13e-4, 1.24e-4, 9.15e-5, 8.30e-5, 5.84e-5])
)


def potential_data(cy, tip, tau, kinv, volume):
    qprime = np.asarray(cy.toric_effective_cone().rays(), dtype=int)
    nq = qprime.shape[0]
    charges = [row for row in qprime]
    raw_coefficients = []

    for charge in qprime:
        q_tau = float(charge @ tau)
        prefactor = (8 * math.pi / volume**2) * q_tau
        raw_coefficients.append((prefactor, -2 * math.pi * math.log10(math.e) * q_tau))

    for i in range(nq - 1):
        for j in range(i + 1, nq):
            qi, qj = qprime[i], qprime[j]
            charges.append(qj - qi)
            prefactor = (
                math.pi * float(qi @ kinv @ qj) + float((qi + qj) @ tau)
            ) * 8 * math.pi / volume**2
            exponent = -2 * math.pi * math.log10(math.e) * float((qi + qj) @ tau)
            raw_coefficients.append((prefactor, exponent))

    raw = np.asarray(raw_coefficients, dtype=float)
    L = np.empty((2, raw.shape[0]), dtype=float)
    L[0, :] = np.sign(raw[:, 0])
    L[1, :] = np.log10(np.abs(raw[:, 0])) + raw[:, 1]
    Q = np.asarray(charges, dtype=int).T
    return Q, L


def validate(cy, tau, volume, kinetic, curve_volumes):
    failures = []
    if int(cy.h11()) != 8 or int(cy.h21()) != 28:
        failures.append(f"Hodge numbers are ({cy.h11()}, {cy.h21()}), expected (8, 28)")
    if not np.isclose(volume, 126.0, rtol=0, atol=1e-9):
        failures.append(f"CY volume is {volume}, expected 126")
    if not np.allclose(tau, EXPECTED_TAU, rtol=0, atol=1e-9):
        failures.append(f"divisor volumes are {tau.tolist()}, expected {EXPECTED_TAU.tolist()}")
    if not np.allclose([curve_volumes.min(), curve_volumes.max()], [1, 3], atol=1e-9):
        failures.append("curve-volume range does not match [1, 3]")
    if not np.allclose(
        np.linalg.eigvalsh(kinetic), EXPECTED_K_EIGENVALUES, rtol=6e-3, atol=0
    ):
        failures.append("kinetic eigenvalues do not match Appendix C")
    if failures:
        raise RuntimeError("Appendix C validation failed: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="paper_benchmarks/appendix_c",
        help="Output database root (default: %(default)s)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    filepath = os.path.abspath(
        os.path.join(args.outdir, "h11_008", "np_0000001", "cy_0000001", "cyax.h5")
    )
    if os.path.exists(filepath) and not args.overwrite:
        raise FileExistsError(f"{filepath} already exists; pass --overwrite to replace it")

    polytope = Polytope(VERTICES)
    triangulation = polytope.triangulate(backend="cgal")
    cy = triangulation.get_cy()
    tip = np.asarray(cy.toric_kahler_cone().tip_of_stretched_cone(1.0), dtype=float)
    tau = np.asarray(cy.compute_divisor_volumes(tip, in_basis=True), dtype=float)
    kinv_raw = np.asarray(cy.compute_inverse_kahler_metric(tip), dtype=float)
    kinv = 0.5 * (kinv_raw + kinv_raw.T)
    kinetic = np.linalg.inv(kinv)
    volume = float(cy.compute_cy_volume(tip))
    curve_volumes = np.asarray(cy.compute_curve_volumes(tip), dtype=float)
    validate(cy, tau, volume, kinetic, curve_volumes)
    Q, L = potential_data(cy, tip, tau, kinv, volume)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as file:
        cytools = file.create_group("cytools")
        geometric = cytools.create_group("geometric")
        geometric.create_dataset("points", data=np.asarray(polytope.points(), dtype=int), compression="gzip")
        geometric.create_dataset("vertices", data=VERTICES, compression="gzip")
        geometric.create_dataset("simplices", data=np.asarray(triangulation.simplices(), dtype=int), compression="gzip")
        geometric.create_dataset("h21", data=int(cy.h21()))
        geometric.create_dataset("glsm", data=np.asarray(cy.glsm_charge_matrix(include_origin=False), dtype=int), compression="gzip")
        geometric.create_dataset("basis", data=np.asarray(cy.divisor_basis(), dtype=int), compression="gzip")
        geometric.create_dataset("tip", data=tip, compression="gzip")
        geometric.create_dataset("tip_prefactor", data=np.ones(2))
        geometric.create_dataset("CY_volume", data=volume)
        geometric.create_dataset("divisor_volumes", data=tau, compression="gzip")
        geometric.create_dataset("curve_volumes", data=curve_volumes, compression="gzip")
        geometric.create_dataset("Kinv", data=kinv, compression="gzip")
        geometric.attrs["source"] = "KS_axiverse_inflation draft, Appendix C"
        geometric.attrs["volume_scale_k"] = 1.0

        potential = cytools.create_group("potential")
        potential.create_dataset("L", data=L, compression="gzip")
        potential.create_dataset("Q", data=Q, compression="gzip")

    print(f"Saved validated Appendix C geometry to {filepath}")
    print(f"h11={cy.h11()}, h21={cy.h21()}, volume={volume:.15g}")
    print(f"divisor volumes={tau.tolist()}")
    print(f"potential terms={Q.shape[1]}")


if __name__ == "__main__":
    main()
