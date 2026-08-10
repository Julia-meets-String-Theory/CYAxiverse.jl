#!/usr/bin/env python3
"""Run the actual author coefficient/reduction code on one exported geometry.

The ``--author-src`` directory should normally be
the external author-code validation directory.  That directory routes the archived author
implementation through the reviewed prefactor correction; passing the raw
archive is allowed only for an explicit diagnostic.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def install_import_stubs():
    """Stub optional construction/solver packages at author-module import time."""
    scipy = types.ModuleType("scipy")
    scipy_linalg = types.ModuleType("scipy.linalg")
    scipy_integrate = types.ModuleType("scipy.integrate")
    scipy_optimize = types.ModuleType("scipy.optimize")
    def qr(matrix, pivoting=False):
        matrix = np.asarray(matrix)
        q, r = np.linalg.qr(matrix, mode="reduced")
        if pivoting:
            # The author uses scipy's pivoted QR only to choose independent
            # rows.  Preserve that contract without importing SciPy.
            pivots = []
            rank = 0
            for column in range(matrix.shape[1]):
                candidate = matrix[:, pivots + [column]]
                candidate_rank = np.linalg.matrix_rank(candidate)
                if candidate_rank > rank:
                    pivots.append(column)
                    rank = candidate_rank
            pivots.extend(
                column for column in range(matrix.shape[1]) if column not in pivots
            )
            return q, r, np.asarray(pivots, dtype=int)
        return q, r

    scipy_linalg.qr = qr
    scipy_integrate.solve_ivp = lambda *args, **kwargs: None
    scipy_optimize.minimize = lambda *args, **kwargs: None
    scipy.linalg = scipy_linalg
    scipy.integrate = scipy_integrate
    scipy.optimize = scipy_optimize
    sys.modules.update({
        "scipy": scipy,
        "scipy.linalg": scipy_linalg,
        "scipy.integrate": scipy_integrate,
        "scipy.optimize": scipy_optimize,
    })

    cytools = types.ModuleType("cytools")
    cytools.Polytope = object
    cytools.Cone = object
    cytools.read_polytopes = lambda *args, **kwargs: None
    cytools.fetch_polytopes = lambda *args, **kwargs: None
    sys.modules["cytools"] = cytools

    def smith_normal_form(matrix):
        """Small integer-SNF shim for the archived reducer's transforms."""
        array = np.asarray(np.rint(matrix), dtype=object)
        rows, columns = array.shape
        left = np.eye(rows, dtype=object)
        right = np.eye(columns, dtype=object)

        def swap_rows(target, first, second):
            target[[first, second], :] = target[[second, first], :]

        def swap_columns(target, first, second):
            target[:, [first, second]] = target[:, [second, first]]

        for pivot_index in range(min(rows, columns)):
            nonzero = None
            for row in range(pivot_index, rows):
                for column in range(pivot_index, columns):
                    if array[row, column] != 0:
                        nonzero = (row, column)
                        break
                if nonzero is not None:
                    break
            if nonzero is None:
                break
            swap_rows(array, pivot_index, nonzero[0])
            swap_rows(left, pivot_index, nonzero[0])
            swap_columns(array, pivot_index, nonzero[1])
            swap_columns(right, pivot_index, nonzero[1])

            while True:
                if array[pivot_index, pivot_index] < 0:
                    array[pivot_index, :] *= -1
                    left[pivot_index, :] *= -1
                changed = False
                pivot = int(array[pivot_index, pivot_index])
                for row in range(pivot_index + 1, rows):
                    value = int(array[row, pivot_index])
                    if value == 0:
                        continue
                    quotient = value // pivot
                    array[row, :] -= quotient * array[pivot_index, :]
                    left[row, :] -= quotient * left[pivot_index, :]
                    if array[row, pivot_index] != 0:
                        swap_rows(array, row, pivot_index)
                        swap_rows(left, row, pivot_index)
                    changed = True
                    break
                if changed:
                    continue
                for column in range(pivot_index + 1, columns):
                    value = int(array[pivot_index, column])
                    if value == 0:
                        continue
                    quotient = value // pivot
                    array[:, column] -= quotient * array[:, pivot_index]
                    right[:, column] -= quotient * right[:, pivot_index]
                    if array[pivot_index, column] != 0:
                        swap_columns(array, column, pivot_index)
                        swap_columns(right, column, pivot_index)
                    changed = True
                    break
                if changed:
                    continue
                offending = None
                for row in range(pivot_index + 1, rows):
                    for column in range(pivot_index + 1, columns):
                        if int(array[row, column]) % pivot != 0:
                            offending = row
                            break
                    if offending is not None:
                        break
                if offending is None:
                    break
                array[pivot_index, :] += array[offending, :]
                left[pivot_index, :] += left[offending, :]

        return (
            np.asarray(array, dtype=float),
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
        )

    hsnf = types.ModuleType("hsnf")
    hsnf.smith_normal_form = smith_normal_form
    sys.modules["hsnf"] = hsnf

    joan = types.ModuleType("Joan_v_minima_phase_fraction_with_critical")
    joan.multi_axion_solver = lambda *args, **kwargs: None
    one_dim = types.ModuleType("one_dim_phase_fraction_with_critical")
    one_dim.one_dim_axion_solver = lambda *args, **kwargs: None
    sys.modules[joan.__name__] = joan
    sys.modules[one_dim.__name__] = one_dim


def read_array(path, integer=False):
    values = np.loadtxt(path, dtype=float)
    return np.rint(values).astype(int) if integer else np.asarray(values)


def charge_key(charge):
    values = tuple(np.rint(charge).astype(int).tolist())
    return min(values, tuple(-value for value in values))


def oriented_q(path, h11):
    values = read_array(path, integer=True)
    values = np.atleast_2d(values)
    return values if values.shape[1] == h11 else values.T


def oriented_l(path, term_count):
    values = np.atleast_2d(read_array(path))
    if values.shape[0] == 2 and values.shape[1] == term_count:
        return values
    if values.shape[1] == 2 and values.shape[0] == term_count:
        return values.T
    raise ValueError(f"L has incompatible shape {values.shape}")


def coefficient_map(reduced):
    result = {}
    for charges, logs, signs in zip(
        (reduced[0], reduced[1]), (reduced[2], reduced[3]), (reduced[4], reduced[5])
    ):
        charges = np.atleast_2d(charges)
        for charge, log_value, sign_value in zip(
            charges, np.atleast_1d(logs), np.atleast_1d(signs)
        ):
            result.setdefault(charge_key(charge), []).append(
                (float(sign_value), float(log_value) / np.log(10.0))
            )
    return result


def compare_maps(package_q, package_l, author_map):
    package_map = {}
    for charge, sign_value, log_value in zip(
        package_q, package_l[0], package_l[1]
    ):
        package_map.setdefault(charge_key(charge), []).append(
            (float(sign_value), float(log_value))
        )
    mismatches = 0
    max_error = 0.0
    for key in set(package_map) | set(author_map):
        expected = sorted(package_map.get(key, []), key=lambda item: item[1])
        observed = sorted(author_map.get(key, []), key=lambda item: item[1])
        if len(expected) != len(observed):
            mismatches += abs(len(expected) - len(observed))
            continue
        for (expected_sign, expected_log), (observed_sign, observed_log) in zip(
            expected, observed
        ):
            mismatches += int(expected_sign != observed_sign)
            if np.isfinite(expected_log) and np.isfinite(observed_log):
                max_error = max(max_error, abs(expected_log - observed_log))
            elif expected_log != observed_log:
                max_error = float("inf")
    return {"count_or_sign_mismatches": mismatches, "max_log10_error": max_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--author-src",
        type=Path,
        default=Path(
            "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/"
            "CN_Axiverse_code/ks_axiverse_python_collaborator/"
            "validation/cyaxiverse_fixed"
        ),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    install_import_stubs()
    sys.path.insert(0, str(args.author_src.resolve()))
    import Camcode_full_2 as author_code
    import cytools_catastrophe_scan as author_scan

    input_dir = args.input_dir
    tau = read_array(input_dir / "tau.txt")
    kinv = read_array(input_dir / "kinv.txt")
    volume = float((input_dir / "cy_volume.txt").read_text().strip())
    charges = np.unique(read_array(input_dir / "author_charges.txt", integer=True), axis=0)
    context = author_scan.GeometryContext(volume, tau, charges, kinv)
    result = {"author_source": str(author_code.AUTHOR_SOURCE), "scales": {}}

    for scale in (0.9, 1.0, 1.1):
        scaled_volume, scaled_tau, scaled_charges, scaled_kinv = (
            author_scan.geometric_quantities(
                None, scale, "unused", "unused", context=context
            )
        )
        reduced = author_code.dim_reductor(
            scaled_charges, scaled_tau, scaled_kinv, scaled_volume, 0.1, False
        )
        author_map = coefficient_map(reduced)
        scale_name = str(scale).replace(".", "p")
        scale_result = {}
        for mode in ("fixed", "full"):
            package_q = oriented_q(
                input_dir / f"{mode}_{scale_name}_Q.txt", len(tau)
            )
            package_l = oriented_l(
                input_dir / f"{mode}_{scale_name}_L.txt", len(package_q)
            )
            scale_result[mode] = compare_maps(package_q, package_l, author_map)
        result["scales"][str(scale)] = scale_result

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
