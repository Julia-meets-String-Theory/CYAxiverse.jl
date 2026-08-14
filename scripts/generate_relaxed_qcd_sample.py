#!/usr/bin/env python3
"""Rebuild a bounded KS/FRST sample with selectable moduli normalization.

The local CYAxiverse data already contains KS polytopes and FRSTs, while the
remote CYTools KS endpoint is not reachable from this host.  This script
reconstructs those stored polytopes/FRSTs in CYTools and supports two modes:

* ``relaxed`` reruns the draft geometric algorithm with no QCD-divisor filter;
* ``canonical_qcd`` keeps the stretched-cone tip direction and dilates the
  same FRST until a selected prime divisor has the requested QCD volume.

The output is therefore a *local KS-record reconstruction*, not a fresh
unfiltered query of the remote KS endpoint.  The source geometry, FRST
identifiers, and normalization policy are recorded in each HDF5 file and in
the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from cytools import Polytope

from geometry_charge_conventions import canonicalize_unique_charge_rows


SCHEMA_VERSION = "cyaxiverse-moduli-reconstruction-v2"


def source_path(source_dir: Path, h11: int, polytope: int) -> Path:
    return (
        source_dir
        / f"h11_{h11:03d}"
        / f"np_{polytope:07d}"
        / "cy_0000001"
        / "cyax.h5"
    )


def output_path(output_dir: Path, h11: int, polytope: int) -> Path:
    return (
        output_dir
        / f"h11_{h11:03d}"
        / f"np_{polytope:07d}"
        / "cy_0000001"
        / "cyax.h5"
    )


def safe_log_abs(value: float) -> float:
    return -math.inf if value == 0.0 else math.log(abs(value))


def draft_controlled_tip(cy, max_m: float, step: float) -> tuple[np.ndarray, float, float]:
    """Apply the draft's potential-control loop without a QCD-volume gate."""
    n_val = 1.0
    m_val = 1.0
    tip = np.asarray(
        cy.toric_kahler_cone().tip_of_stretched_cone(math.sqrt(n_val)), dtype=float
    )
    basis = np.asarray(cy.divisor_basis(), dtype=int)
    basis_indices = basis - 1
    div_volumes = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
    tau0 = div_volumes[basis_indices]
    kinv0 = np.asarray(cy.compute_inverse_kahler_metric(tip), dtype=float)
    kinv0 = 0.5 * (kinv0 + kinv0.T)

    qprime_raw = np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    qprime, _ = canonicalize_unique_charge_rows(qprime_raw)

    while True:
        m2 = m_val**2
        tau = m2 * tau0
        kinv = m2**2 * kinv0
        converged = True
        for j in range(len(qprime)):
            qj = qprime[j]
            tau_qj = float(np.dot(tau, qj))
            rhs = abs(safe_log_abs(tau_qj) - 2.0 * math.pi * tau_qj)
            for i in range(j + 1, len(qprime)):
                qi = qprime[i]
                lhs_argument = math.pi * float(np.dot(qi, kinv @ qj))
                lhs = abs(
                    safe_log_abs(lhs_argument)
                    - 2.0 * math.pi * float(np.dot(tau, qi + qj))
                )
                if lhs <= rhs:
                    converged = False
                    break
            if not converged:
                break
        if converged:
            break
        m_val += step
        if m_val > max_m:
            raise RuntimeError(f"draft potential-control loop exceeded max_m={max_m:g}")

    # Match the draft's post-loop lower-bound correction.
    if np.min(tau) <= 1.0:
        n_val = 1.0 / float(np.min(tau))
        tip = m_val * math.sqrt(n_val) * tip
        div_volumes = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
        tau = div_volumes[basis_indices]
        kinv = np.asarray(cy.compute_inverse_kahler_metric(tip), dtype=float)
        kinv = 0.5 * (kinv + kinv.T)
    else:
        tip = m_val * tip
        tau = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)[basis_indices]
        kinv = np.asarray(cy.compute_inverse_kahler_metric(tip), dtype=float)
        kinv = 0.5 * (kinv + kinv.T)
    return tip, math.sqrt(n_val), m_val


def canonical_qcd_tip(
    cy,
    qcd_volume_target: float,
    qcd_divisor_index: int | None,
    max_m: float,
) -> tuple[np.ndarray, float, float, int]:
    """Dilate the canonical stretched-cone ray to a prime-divisor target."""
    tip0 = np.asarray(
        cy.toric_kahler_cone().tip_of_stretched_cone(1.0), dtype=float
    )
    prime_tau0 = np.asarray(cy.compute_divisor_volumes(tip0), dtype=float)
    if prime_tau0.ndim != 1 or not np.all(np.isfinite(prime_tau0)):
        raise RuntimeError("CYTools returned invalid prime toric divisor volumes.")
    candidates = (
        [qcd_divisor_index]
        if qcd_divisor_index is not None
        else range(len(prime_tau0))
    )
    for candidate_index in candidates:
        if not 0 <= candidate_index < len(prime_tau0):
            continue
        prime_volume = float(prime_tau0[candidate_index])
        if prime_volume <= 0.0:
            continue
        m_scale = math.sqrt(qcd_volume_target / prime_volume)
        if m_scale < 1.0 or m_scale > max_m:
            continue
        final_prime_tau = m_scale**2 * prime_tau0
        if np.min(final_prime_tau) < 1.0 - 1e-8:
            continue
        return m_scale * tip0, 1.0, m_scale, int(candidate_index)
    requested = (
        f"prime toric divisor index {qcd_divisor_index}"
        if qcd_divisor_index is not None
        else "any prime toric divisor"
    )
    raise RuntimeError(
        f"No canonical stretched-cone ray reaches QCD volume "
        f"{qcd_volume_target:g} for {requested} while keeping every prime "
        "divisor volume at least one."
    )


def potential_data(cy, tip: np.ndarray, qprime: np.ndarray):
    tau = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
    basis = np.asarray(cy.divisor_basis(), dtype=int) - 1
    tau = tau[basis]
    kinv = np.asarray(cy.compute_inverse_kahler_metric(tip), dtype=float)
    kinv = 0.5 * (kinv + kinv.T)
    volume = float(cy.compute_cy_volume(tip))

    nq = qprime.shape[0]
    pair_count = nq * (nq - 1) // 2
    term_count = nq + pair_count
    q = np.zeros((int(cy.h11()), term_count), dtype=float)
    l_raw = np.zeros((2, term_count), dtype=float)
    q[:, :nq] = qprime.T

    prefactor = 8.0 * math.pi / volume**2
    for j in range(nq):
        charge = qprime[j]
        qtau = float(np.dot(charge, tau))
        l_raw[0, j] = prefactor * qtau
        l_raw[1, j] = -2.0 * math.log10(math.e) * math.pi * qtau

    index = nq
    for i in range(nq - 1):
        for j in range(i + 1, nq):
            qi, qj = qprime[i], qprime[j]
            q[:, index] = qj - qi
            qsum = qi + qj
            term = prefactor * (
                math.pi * float(np.dot(qi, kinv @ qj)) + float(np.dot(qsum, tau))
            )
            exponent = -2.0 * math.log10(math.e) * math.pi * float(
                np.dot(qsum, tau)
            )
            l_raw[0, index] = term
            l_raw[1, index] = exponent
            index += 1

    l = np.zeros_like(l_raw)
    l[0, :] = np.sign(l_raw[0, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        l[1, :] = np.log10(np.abs(l_raw[0, :])) + l_raw[1, :]
    return tau, kinv, volume, q, l


def rebuild_one(
    source: Path,
    target: Path,
    h11: int,
    polytope: int,
    max_m: float,
    step: float,
    moduli_policy: str,
    qcd_volume_target: float,
    qcd_divisor_index: int | None,
):
    with h5py.File(source, "r") as file:
        points = np.asarray(file["cytools/geometric/points"], dtype=int)
        simplices = np.asarray(file["cytools/geometric/simplices"], dtype=int)

    poly = Polytope(points, deterministic_glsm_basis=True)
    triangulation = poly.triangulate(
        simplices=simplices,
        backend="cgal",
        check_input_simplices=True,
        verbosity=0,
    )
    cy = triangulation.get_cy()
    if int(cy.h11()) != h11:
        raise RuntimeError(f"reconstructed h11={cy.h11()} but expected {h11}")

    basis = np.asarray(cy.divisor_basis(), dtype=int)
    qprime, charge_metadata = canonicalize_unique_charge_rows(
        np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    )
    if moduli_policy == "canonical_qcd":
        tip, n_scale, m_scale, selected_qcd_index = canonical_qcd_tip(
            cy,
            qcd_volume_target=qcd_volume_target,
            qcd_divisor_index=qcd_divisor_index,
            max_m=max_m,
        )
    else:
        tip, n_scale, m_scale = draft_controlled_tip(cy, max_m=max_m, step=step)
        selected_qcd_index = None
    tau, kinv, volume, q, l = potential_data(cy, tip, qprime)
    prime_tau = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
    curve_volumes = np.asarray(cy.compute_curve_volumes(tip), dtype=float)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "moduli_policy": moduli_policy,
        "source_geometry": str(source),
        "source_h11": h11,
        "source_polytope": polytope,
        "source_frst": 1,
        "frst_reconstructed_from_stored_simplices": True,
        "draft_control_max_m": max_m,
        "draft_control_step": step,
        "tip_prefactor": [n_scale, m_scale],
        "potential_charge_convention": charge_metadata["convention"],
        "canonical_effective_cone_ray_count": charge_metadata["canonical_count"],
        "duplicate_effective_cone_rows_removed": charge_metadata["duplicates_removed"],
    }
    if selected_qcd_index is None:
        metadata["qcd_divisor_requirement"] = "relaxed"
    else:
        metadata.update({
            "qcd_divisor_requirement": "targeted_prime_toric_divisor",
            "qcd_divisor_index": selected_qcd_index,
            "qcd_divisor_index_base": 0,
            "qcd_volume_target": qcd_volume_target,
            "qcd_divisor_volume": float(prime_tau[selected_qcd_index]),
        })
    with h5py.File(target, "r+") as file:
        geometric = file["cytools/geometric"]

        def replace_dataset(group, name, data):
            if name in group:
                del group[name]
            array = np.asarray(data)
            if array.ndim == 0:
                group.create_dataset(name, data=data)
            else:
                group.create_dataset(name, data=data, compression="gzip", compression_opts=9)

        for name, data in {
            "basis": basis,
            "tip": tip,
            "tip_prefactor": np.asarray([n_scale, m_scale], dtype=float),
            "CY_volume": volume,
            "divisor_volumes": tau,
            "Kinv": kinv,
            "effective_cone": qprime,
            "prime_divisor_volumes": prime_tau,
            "curve_volumes": curve_volumes,
        }.items():
            replace_dataset(geometric, name, data)
        geometric.attrs["moduli_policy"] = metadata["moduli_policy"]
        geometric.attrs["qcd_divisor_requirement"] = metadata["qcd_divisor_requirement"]
        potential = file["cytools/potential"]
        for name, data in {"Q": q, "L": l}.items():
            replace_dataset(potential, name, data)
        if "construction_metadata" not in file:
            construction = file.create_group("construction_metadata")
        else:
            construction = file["construction_metadata"]
        construction.attrs["construction_metadata_json"] = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )
    return {
        "h11": h11,
        "polytope": polytope,
        "frst": 1,
        "source": str(source),
        "target": str(target),
        "n_prime_divisors": int(len(prime_tau)),
        "min_prime_divisor_volume": float(np.min(prime_tau)),
        "max_prime_divisor_volume": float(np.max(prime_tau)),
        "min_basis_divisor_volume": float(np.min(tau)),
        "cy_volume": volume,
        "n_effective_cone_rays": int(qprime.shape[0]),
        "m_scale": m_scale,
        "moduli_policy": moduli_policy,
        "qcd_divisor_index": selected_qcd_index,
        "status": "saved",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("../data"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--h11-min", type=int, default=4)
    parser.add_argument("--h11-max", type=int, default=10)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--polytope-start", type=int, default=1)
    parser.add_argument("--max-m", type=float, default=1_000_000.0)
    parser.add_argument("--m-step", type=float, default=1e-2)
    parser.add_argument(
        "--moduli-policy",
        choices=("relaxed", "canonical_qcd"),
        default="relaxed",
        help=(
            "Moduli normalization: relaxed omits the QCD-divisor filter; "
            "canonical_qcd dilates the stretched-cone tip to the target."
        ),
    )
    parser.add_argument("--qcd-volume-target", type=float, default=40.0)
    parser.add_argument("--qcd-divisor-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.qcd_volume_target <= 0.0:
        parser.error("--qcd-volume-target must be positive")
    if args.qcd_divisor_index is not None and args.qcd_divisor_index < 0:
        parser.error("--qcd-divisor-index must be non-negative")
    if args.qcd_divisor_index is not None and args.moduli_policy != "canonical_qcd":
        parser.error("--qcd-divisor-index requires --moduli-policy canonical_qcd")
    source_dir = args.source_dir.resolve()
    output_dir = args.outdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = []
    started = time.time()
    for h11 in range(args.h11_min, args.h11_max + 1):
        for offset in range(args.n):
            polytope = args.polytope_start + offset
            source = source_path(source_dir, h11, polytope)
            target = output_path(output_dir, h11, polytope)
            if not source.is_file():
                failures.append({"h11": h11, "polytope": polytope, "status": "missing_source"})
                print(f"MISSING h11={h11} polytope={polytope}", flush=True)
                continue
            if target.exists() and not args.overwrite:
                print(f"SKIP existing h11={h11} polytope={polytope}", flush=True)
                continue
            try:
                row = rebuild_one(
                    source,
                    target,
                    h11,
                    polytope,
                    args.max_m,
                    args.m_step,
                    args.moduli_policy,
                    args.qcd_volume_target,
                    args.qcd_divisor_index,
                )
                manifest.append(row)
                print(
                    f"SAVED h11={h11} polytope={polytope} "
                    f"m={row['m_scale']:.6g} min_prime_tau={row['min_prime_divisor_volume']:.6g}",
                    flush=True,
                )
            except Exception as exc:
                failures.append({
                    "h11": h11,
                    "polytope": polytope,
                    "status": "failure",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"FAIL h11={h11} polytope={polytope}: {type(exc).__name__}: {exc}", flush=True)

    manifest_path = output_dir / "relaxed_qcd_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "schema_version": SCHEMA_VERSION,
        "h11_min": args.h11_min,
        "h11_max": args.h11_max,
        "n_per_h11": args.n,
        "polytope_start": args.polytope_start,
        "moduli_policy": args.moduli_policy,
        "qcd_volume_target": args.qcd_volume_target,
        "qcd_divisor_index": args.qcd_divisor_index,
        "qcd_divisor_requirement": (
            "targeted_prime_toric_divisor"
            if args.moduli_policy == "canonical_qcd"
            else "relaxed"
        ),
        "draft_control_max_m": args.max_m,
        "draft_control_step": args.m_step,
        "saved": manifest,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Saved {len(manifest)} geometries; failures={len(failures)}")
    print(f"Manifest: {manifest_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
