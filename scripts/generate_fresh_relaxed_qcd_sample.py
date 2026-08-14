#!/usr/bin/env python3
"""Generate a fresh, unfiltered KS/CYTools sample for h11=4..10.

The parquet files are the official complete 4D KS mirror linked from the
Kreuzer--Skarke data site.  This script selects the first ten favorable N-lattice
polytopes at each h11, constructs one CYTools triangulation (an FRST), and
applies the draft geometric algorithm without imposing a QCD-divisor volume
window.  The selected polytope provenance is stored in the manifest and each
HDF5 record.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from cytools import Polytope

from geometry_charge_conventions import canonicalize_unique_charge_rows
from generate_relaxed_qcd_sample import draft_controlled_tip, potential_data


def output_path(output_dir: Path, h11: int, index: int) -> Path:
    return output_dir / f"h11_{h11:03d}" / f"np_{index:07d}" / "cy_0000001" / "cyax.h5"


def write_geometry(target: Path, poly, triangulation, cy, h11: int, index: int,
                   source: dict, max_m: float, m_step: float) -> dict:
    if int(cy.h11()) != h11:
        raise RuntimeError(f"CYTools returned h11={cy.h11()} for requested h11={h11}")
    frst = {
        "valid": bool(triangulation.is_valid()),
        "fine": bool(triangulation.is_fine()),
        "regular": bool(triangulation.is_regular()),
        "star": bool(triangulation.is_star()),
    }
    if not all(frst.values()):
        raise RuntimeError(f"triangulation is not an FRST: {frst}")

    points = np.asarray(poly.points(), dtype=int)
    simplices = np.asarray(triangulation.simplices(), dtype=int)
    h21 = int(cy.h21())
    glsm = np.asarray(cy.glsm_charge_matrix(include_origin=False), dtype=int)
    basis = np.asarray(cy.divisor_basis(), dtype=int)
    qprime, charge_metadata = canonicalize_unique_charge_rows(
        np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    )
    tip, n_scale, m_scale = draft_controlled_tip(cy, max_m=max_m, step=m_step)
    tau, kinv, volume, q, l = potential_data(cy, tip, qprime)
    prime_tau = np.asarray(cy.compute_divisor_volumes(tip), dtype=float)
    curve_volumes = np.asarray(cy.compute_curve_volumes(tip), dtype=float)

    metadata = {
        "schema_version": "cyaxiverse-fresh-ks-relaxed-qcd-v1",
        "moduli_policy": "draft_no_qcd_divisor_requirement",
        "qcd_divisor_requirement": "relaxed",
        "source_dataset": "calabi-yau-data/polytopes-4d",
        "source_dataset_url": "https://huggingface.co/datasets/calabi-yau-data/polytopes-4d",
        "source_parquet": source["parquet"],
        "source_row_index": source["row_index"],
        "source_h11": h11,
        "source_h12": source["physical_h12"],
        "source_ks_h11": source["ks_h11"],
        "source_ks_h12": source["ks_h12"],
        "source_vertex_count": source["vertex_count"],
        "source_favorable_N": True,
        "frst_validation": frst,
        "draft_control_max_m": max_m,
        "draft_control_step": m_step,
        "tip_prefactor": [n_scale, m_scale],
        "potential_charge_convention": charge_metadata["convention"],
        "canonical_effective_cone_ray_count": charge_metadata["canonical_count"],
        "duplicate_effective_cone_rows_removed": charge_metadata["duplicates_removed"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as file:
        cytools = file.create_group("cytools")
        geometric = cytools.create_group("geometric")

        def dataset(name, data):
            array = np.asarray(data)
            if array.ndim == 0:
                geometric.create_dataset(name, data=data)
            else:
                geometric.create_dataset(name, data=data, compression="gzip", compression_opts=9)

        dataset("points", points)
        dataset("simplices", simplices)
        dataset("h21", h21)
        dataset("glsm", glsm)
        dataset("basis", basis)
        dataset("tip", tip)
        dataset("tip_prefactor", np.asarray([n_scale, m_scale], dtype=float))
        dataset("CY_volume", volume)
        dataset("divisor_volumes", tau)
        dataset("Kinv", kinv)
        dataset("effective_cone", qprime)
        dataset("prime_divisor_volumes", prime_tau)
        dataset("curve_volumes", curve_volumes)
        geometric.attrs["moduli_policy"] = metadata["moduli_policy"]
        geometric.attrs["qcd_divisor_requirement"] = "relaxed"
        geometric.attrs["source_dataset"] = metadata["source_dataset"]

        potential = cytools.create_group("potential")
        potential.create_dataset("L", data=l, compression="gzip", compression_opts=9)
        potential.create_dataset("Q", data=q, compression="gzip", compression_opts=9)

        construction = file.create_group("construction_metadata")
        construction.attrs["construction_metadata_json"] = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )

    return {
        "h11": h11,
        "polytope": index,
        "frst": 1,
        "target": str(target),
        "source_parquet": source["parquet"],
        "source_row_index": source["row_index"],
        "h12": source["physical_h12"],
        "vertex_count": source["vertex_count"],
        "frst_validation": frst,
        "n_prime_divisors": int(len(prime_tau)),
        "min_prime_divisor_volume": float(np.min(prime_tau)),
        "max_prime_divisor_volume": float(np.max(prime_tau)),
        "min_basis_divisor_volume": float(np.min(tau)),
        "cy_volume": volume,
        "n_effective_cone_rays": int(qprime.shape[0]),
        "m_scale": m_scale,
        "status": "saved",
    }


def select_polytopes(parquet_dir: Path, per_h11: int) -> dict[int, list[dict]]:
    selected = {h11: [] for h11 in range(4, 11)}
    for filename in ("05", "06", "07", "08", "09"):
        path = parquet_dir / f"polytopes-4d-{filename}-vertices.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = pq.read_table(path, columns=["vertices", "vertex_count", "h11", "h12"]).to_pylist()
        for row_index, row in enumerate(rows):
            # The parquet mirror stores the dual (M-lattice) Hodge labels.
            # CYTools fetch_polytopes(..., lattice="N") swaps these labels,
            # so physical h11 is the parquet h12 column.
            h11 = int(row["h12"])
            if h11 not in selected or len(selected[h11]) >= per_h11:
                continue
            poly = Polytope(np.asarray(row["vertices"], dtype=int), deterministic_glsm_basis=True)
            if not poly.is_favorable(lattice="N"):
                continue
            selected[h11].append({
                "vertices": row["vertices"],
                "vertex_count": int(row["vertex_count"]),
                "h11": h11,
                "physical_h12": int(row["h11"]),
                "ks_h11": int(row["h11"]),
                "ks_h12": int(row["h12"]),
                "parquet": str(path),
                "row_index": row_index,
            })
        if all(len(selected[h11]) >= per_h11 for h11 in selected):
            break
    missing = {h11: len(rows) for h11, rows in selected.items() if len(rows) < per_h11}
    if missing:
        raise RuntimeError(f"insufficient favorable rows in downloaded partitions: {missing}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--per-h11", type=int, default=10)
    parser.add_argument("--max-m", type=float, default=1_000_000.0)
    parser.add_argument("--m-step", type=float, default=1e-2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.per_h11 <= 0:
        raise ValueError("--per-h11 must be positive")
    output_dir = args.outdir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_polytopes(args.parquet_dir.resolve(), args.per_h11)
    manifest = []
    failures = []
    for h11 in range(4, 11):
        for index, source in enumerate(selected[h11], start=1):
            try:
                poly = Polytope(np.asarray(source["vertices"], dtype=int), deterministic_glsm_basis=True)
                triangulation = poly.triangulate(backend="cgal", verbosity=0)
                cy = triangulation.get_cy()
                record = write_geometry(
                    output_path(output_dir, h11, index), poly, triangulation, cy,
                    h11, index, source, args.max_m, args.m_step
                )
                manifest.append(record)
                print(f"SAVED h11={h11} polytope={index} source={source['parquet']}#{source['row_index']} "
                      f"m={record['m_scale']:.6g} min_prime_tau={record['min_prime_divisor_volume']:.6g}",
                      flush=True)
            except Exception as error:
                failure = {
                    "h11": h11,
                    "polytope": index,
                    "source_parquet": source["parquet"],
                    "source_row_index": source["row_index"],
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                print(f"FAILED h11={h11} polytope={index}: {failure['error']}", flush=True)

    manifest_path = output_dir / "fresh_relaxed_qcd_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "cyaxiverse-fresh-ks-relaxed-qcd-v1",
        "source_dataset": "calabi-yau-data/polytopes-4d",
        "source_dataset_url": "https://huggingface.co/datasets/calabi-yau-data/polytopes-4d",
        "qcd_divisor_requirement": "relaxed",
        "selection": "first favorable N-lattice rows in parquet partitions 05-09, ten per h11",
        "saved": manifest,
        "failures": failures,
    }, indent=2, sort_keys=True))
    print(f"Saved {len(manifest)} geometries; failures={len(failures)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
