#!/usr/bin/env python3
"""Run the original draft catastrophe scanner on local fresh HDF5 records.

Unlike the Julia comparison scanner, this adapter calls the draft's own
`dim_reductor` and reduced-model solvers.  It reads the saved GLSM matrix,
divisor volumes, CY volume, and inverse Kahler metric directly, so no KS
network request or CYTools reconstruction is needed during the scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from cytools_catastrophe_scan import (
    GeometryContext,
    changed_signature,
    model_signature_summary,
    solve_at_k,
)


def parse_grid(text: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("k grid must contain finite positive values")
    return values


def geometry_paths(root: Path):
    for h11_dir in sorted(root.glob("h11_*")):
        if not h11_dir.is_dir():
            continue
        h11 = int(h11_dir.name.split("_")[1])
        if not 4 <= h11 <= 10:
            continue
        for poly_dir in sorted(h11_dir.glob("np_*")):
            path = poly_dir / "cy_0000001" / "cyax.h5"
            if path.is_file():
                yield h11, int(poly_dir.name.split("_")[1]), path


def load_context(path: Path) -> GeometryContext:
    with h5py.File(path, "r") as file:
        geometric = file["cytools/geometric"]
        # The fresh generator writes glsm=glsm_charge_matrix(include_origin=False)
        # with shape h11 × (N_divisors-1). This is exactly the draft scan's
        # transpose-after-origin-removal matrix.
        charges = np.unique(np.asarray(geometric["glsm"], dtype=float).T, axis=0)
        divisor_vol = np.asarray(geometric["divisor_volumes"], dtype=float)
        kinv = np.asarray(geometric["Kinv"], dtype=float)
        cy_vol = float(geometric["CY_volume"][()])
    return GeometryContext(cy_vol, divisor_vol, charges, kinv)


def count_for_compare(result, field: str) -> int:
    if field == "theory":
        return int(round(result.theory_minima))
    return int(result.n_minima)


def scanner_args(starts: int) -> SimpleNamespace:
    return SimpleNamespace(
        triang_backend="cgal",
        cone_backend="glop",
        threshold=0.01,
        flag_perturbative=1,
        samples=starts,
        seed=0,
        one_dim_domain=10 * math.pi,
        one_dim_step=0.01,
        hessian_tol=1e-6,
        grad_tol=1e-8,
        maxfev=200,
        cluster_tol=1e-4,
        first_k_require_flag=None,
        first_k_min_qdot_tau=1.0,
        allow_nonpositive_reduced_sign_after_first_k=True,
        run_later_k_only_if_first_minima_gt=1,
        minima_field="reduced",
        candidate_type="minima",
        allow_model_switch_candidates=False,
        allow_minima_change_without_critical_change=False,
        only_candidates=False,
    )


def solve_path(context, h11: int, poly_index: int, args, base_k, extension_k,
               adaptive: bool):
    results = []
    for index, k in enumerate(base_k):
        result = solve_at_k(
            None, h11, poly_index, k, args, context=context,
            required_min_qdot_tau=(args.first_k_min_qdot_tau
                                   if adaptive and index == 0 else None),
            allow_nonpositive_reduced_sign=(index > 0),
        )
        results.append(result)
        if adaptive and index == 0:
            if not result.valid_truncation or result.n_minima <= 1:
                break
    if adaptive and len(results) == len(base_k):
        last = results[-1]
        if last.valid_truncation and count_for_compare(last, args.minima_field) > 1:
            for k in extension_k:
                result = solve_at_k(
                    None, h11, poly_index, k, args, context=context,
                    allow_nonpositive_reduced_sign=True,
                )
                results.append(result)
                if (not result.valid_truncation or
                        count_for_compare(result, args.minima_field) <= 1):
                    break

    complete = len(results) >= len(base_k)
    signature = changed_signature(results, minima_field=args.minima_field) if complete else {
        "minima_changed": False, "critical_changed": False, "index_changed": False,
    }
    model = model_signature_summary(results) if results else {
        "model_consistent": False, "model_signature_count": 0, "model_summary": [],
    }
    raw_candidate = complete and signature["minima_changed"]
    model_switch = raw_candidate and not model["model_consistent"]
    minima_without_critical = (
        raw_candidate and signature["minima_changed"] and
        not signature["critical_changed"]
    )
    candidate = raw_candidate and model["model_consistent"] and not minima_without_critical
    return {
        "complete": complete,
        "results": results,
        "signature": signature,
        "model": model,
        "raw_candidate": raw_candidate,
        "model_switch": model_switch,
        "minima_without_critical": minima_without_critical,
        "candidate": candidate,
    }


def row_from_result(h11: int, poly_index: int, result):
    row = asdict(result)
    row.update({"h11": h11, "poly_index": poly_index})
    return row


def run(root: Path, output: Path, adaptive: bool, starts: int):
    output.mkdir(parents=True, exist_ok=True)
    base_k = parse_grid("0.5,1.0" if adaptive else "0.5,1.0,1.25,1.5,1.75,2.0,2.5,3.0")
    extension_k = parse_grid("1.25,1.5,1.75,2.0,2.5,3.0") if adaptive else []
    args = scanner_args(starts)
    summaries = []
    rows = []
    errors = []
    paths = list(geometry_paths(root))
    for rank, (h11, poly_index, path) in enumerate(paths, start=1):
        try:
            context = load_context(path)
            scan = solve_path(context, h11, poly_index, args, base_k, extension_k, adaptive)
            for result in scan["results"]:
                rows.append(row_from_result(h11, poly_index, result))
            results = scan["results"]
            summaries.append({
                "sample_rank": rank,
                "h11": h11,
                "polytope": poly_index,
                "frst": 1,
                "k_values": ";".join(str(result.k) for result in results),
                "minima_counts": ";".join(str(count_for_compare(result, args.minima_field)) for result in results),
                "critical_counts": ";".join(str(result.n_critical) for result in results),
                "qdot_first": results[0].qdot_tau_min if results else None,
                "valid_truncation": all(result.valid_truncation for result in results),
                "complete": scan["complete"],
                "model_consistent": scan["model"]["model_consistent"],
                "minima_changed": scan["signature"]["minima_changed"],
                "critical_changed": scan["signature"]["critical_changed"],
                "raw_candidate": scan["raw_candidate"],
                "candidate": scan["candidate"],
                "model_switch": scan["model_switch"],
                "minima_without_critical": scan["minima_without_critical"],
                "flags": ";".join(str(result.flag) for result in results),
                "reduced_dimensions": ";".join(str(result.reduced_dim) for result in results),
                "status": "scanned",
            })
            print(f"[{rank}/{len(paths)}] h11={h11} polytope={poly_index} "
                  f"candidate={scan['candidate']} raw={scan['raw_candidate']}", flush=True)
        except Exception as error:
            errors.append({"sample_rank": rank, "h11": h11, "polytope": poly_index,
                           "frst": 1, "error": f"{type(error).__name__}: {error}"})
            print(f"[{rank}/{len(paths)}] h11={h11} polytope={poly_index} FAILED: {error}", flush=True)

    def write_csv(name, records):
        if not records:
            (output / name).write_text("")
            return
        fields = sorted({key for record in records for key in record})
        with (output / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    write_csv("scan_rows.csv", rows)
    write_csv("geometry_summary.csv", summaries)
    write_csv("errors.csv", errors)
    metadata = {
        "data_dir": str(root),
        "geometries": len(paths),
        "scanned": len(summaries),
        "errors": len(errors),
        "adaptive": adaptive,
        "base_k": base_k,
        "extension_k": extension_k,
        "first_k_min_qdot_tau": 1.0 if adaptive else None,
        "reduction": "original Python dim_reductor",
        "threshold": 0.01,
        "candidate_type": "minima",
        "minima_field": "reduced",
        "solver_starts": starts,
        "candidates": sum(record["candidate"] for record in summaries),
        "raw_minima_changing": sum(record["raw_candidate"] for record in summaries),
        "model_switch_symptoms": sum(record["model_switch"] for record in summaries),
        "minima_without_critical_symptoms": sum(record["minima_without_critical"] for record in summaries),
        "first_gate_failures": sum(
            record["qdot_first"] is not None and record["qdot_first"] < 1.0
            for record in summaries
        ) if adaptive else 0,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--all-grid", action="store_true")
    parser.add_argument("--starts", type=int, default=2048)
    args = parser.parse_args()
    if args.adaptive == args.all_grid:
        parser.error("choose exactly one of --adaptive or --all-grid")
    run(args.data_dir.resolve(), args.output_dir.resolve(), args.adaptive, args.starts)


if __name__ == "__main__":
    main()
