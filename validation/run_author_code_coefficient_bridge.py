#!/usr/bin/env python3
"""Run the archived author coefficient code on package-exported geometry data.

This deliberately imports and calls the author's ``Camcode_full_2`` routines.
CYTools and the solver modules are stubbed only at import time because this
check supplies the already-generated geometry tensors and compares the
coefficient/reduction path, not polytope construction or minima solving.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def _install_import_stubs():
    scipy = types.ModuleType("scipy")
    scipy_linalg = types.ModuleType("scipy.linalg")
    scipy_integrate = types.ModuleType("scipy.integrate")
    scipy_optimize = types.ModuleType("scipy.optimize")

    def qr(matrix, pivoting=False):
        matrix = np.asarray(matrix)
        q, r = np.linalg.qr(matrix, mode="reduced")
        if pivoting:
            # scipy.linalg.qr(..., pivoting=True) is used only to select
            # independent rows in the author's reducer.  Reproduce that
            # contract with a deterministic rank-greedy pivot list.
            pivots = []
            rank = 0
            for column in range(matrix.shape[1]):
                candidate = matrix[:, pivots + [column]]
                candidate_rank = np.linalg.matrix_rank(candidate)
                if candidate_rank > rank:
                    pivots.append(column)
                    rank = candidate_rank
            pivots.extend(column for column in range(matrix.shape[1]) if column not in pivots)
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
        """Small integer-SNF compatibility shim for the archived import.

        The author's reducer only needs the unimodular row/column transforms.
        This implements the standard Euclidean row/column reduction and
        returns ``D, L, R`` with ``L @ A @ R == D``. It is a dependency shim;
        the coefficient and dimensional-reduction code remains the archived
        author code.
        """
        array = np.asarray(np.rint(matrix), dtype=object)
        rows, columns = array.shape
        left = np.eye(rows, dtype=object)
        right = np.eye(columns, dtype=object)

        def swap_rows(target, first, second):
            target[[first, second], :] = target[[second, first], :]

        def swap_columns(target, first, second):
            target[:, [first, second]] = target[:, [second, first]]

        limit = min(rows, columns)
        for pivot_index in range(limit):
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

                pivot = int(array[pivot_index, pivot_index])
                offending = None
                for row in range(pivot_index + 1, rows):
                    for column in range(pivot_index + 1, columns):
                        if int(array[row, column]) % pivot != 0:
                            offending = row
                            break
                    if offending is not None:
                        break
                if offending is not None:
                    # The offending row has zero pivot-column entry, so this
                    # preserves the cleared column while exposing the entry
                    # to the Euclidean reduction above.
                    array[pivot_index, :] += array[offending, :]
                    left[pivot_index, :] += left[offending, :]
                    continue
                break

        return np.asarray(array, dtype=float), np.asarray(left, dtype=float), np.asarray(right, dtype=float)

    hsnf = types.ModuleType("hsnf")
    hsnf.smith_normal_form = smith_normal_form
    sys.modules["hsnf"] = hsnf

    joan = types.ModuleType("Joan_v_minima_phase_fraction_with_critical")
    joan.multi_axion_solver = lambda *args, **kwargs: None
    one_dim = types.ModuleType("one_dim_phase_fraction_with_critical")
    one_dim.one_dim_axion_solver = lambda *args, **kwargs: None
    sys.modules[joan.__name__] = joan
    sys.modules[one_dim.__name__] = one_dim


def _read(path):
    return np.loadtxt(path, dtype=float)


def _read_int(path):
    return np.rint(_read(path)).astype(int)


def _charge_key(charge):
    """Canonicalize q and -q, which define the same cosine term."""
    values = tuple(np.rint(charge).astype(int).tolist())
    opposite = tuple(-value for value in values)
    return min(values, opposite)


def _coefficient_map(reduced):
    q_tilde, q_bar = reduced[0], reduced[1]
    l_tilde, l_bar = reduced[2], reduced[3]
    s_tilde, s_bar = reduced[4], reduced[5]
    entries = {}
    for charges, logs, signs in (
        (q_tilde, l_tilde, s_tilde), (q_bar, l_bar, s_bar)
    ):
        charges = np.atleast_2d(charges)
        logs = np.atleast_1d(logs)
        signs = np.atleast_1d(signs)
        for charge, log_value, sign_value in zip(charges, logs, signs):
            key = _charge_key(charge)
            entries.setdefault(key, []).append(
                (float(sign_value), float(log_value) / np.log(10.0))
            )
    return entries


def _compare_maps(package_q, package_l, author_map):
    package_map = {}
    for charge, sign_value, log_value in zip(
        package_q.T, package_l[0, :], package_l[1, :]
    ):
        package_map.setdefault(_charge_key(charge), []).append(
            (float(sign_value), float(log_value))
        )
    keys_match = set(package_map) == set(author_map)
    sign_mismatches = 0
    max_log_error = 0.0
    count_mismatch = 0
    for key in set(package_map) | set(author_map):
        package_values = sorted(package_map.get(key, []), key=lambda value: value[1])
        author_values = sorted(author_map.get(key, []), key=lambda value: value[1])
        if len(package_values) != len(author_values):
            count_mismatch += abs(len(package_values) - len(author_values))
            continue
        for (package_sign, package_log), (author_sign, author_log) in zip(
            package_values, author_values
        ):
            sign_mismatches += int(package_sign != author_sign)
            if np.isfinite(package_log) and np.isfinite(author_log):
                max_log_error = max(max_log_error, abs(package_log - author_log))
            elif package_log != author_log:
                max_log_error = float("inf")
    return {
        "keys_match": keys_match,
        "count_mismatch": count_mismatch,
        "sign_mismatches": sign_mismatches,
        "max_finite_log10_error": max_log_error,
    }


def _has_validation_failure(value):
    """Return whether a nested comparison result contains a failed check."""
    if isinstance(value, dict):
        if "error" in value:
            return True
        for key, item in value.items():
            if key == "keys_match" and item is False:
                return True
            if key in {"count_mismatch", "sign_mismatches"} and item != 0:
                return True
            if key in {"max_finite_log10_error", "max_error"} and (
                not np.isfinite(item) or item > 1e-10
            ):
                return True
            if _has_validation_failure(item):
                return True
    elif isinstance(value, list):
        return any(_has_validation_failure(item) for item in value)
    return False


def compare_input_dir(input_dir, author_code, author_scan, author_src):
    tau = _read(input_dir / "tau.txt")
    kinv = _read(input_dir / "kinv.txt")
    cy_volume = float((input_dir / "cy_volume.txt").read_text().strip())
    # ``geometric_base`` calls np.unique(..., axis=0) on the CYTools GLSM
    # rows.  The package export preserves the stored row order, so reproduce
    # that author operation here rather than relying on file ordering.
    charges_path = input_dir / "author_charges.txt"
    charges = np.unique(_read_int(charges_path), axis=0)
    context = author_scan.GeometryContext(cy_volume, tau, charges, kinv)
    results = {
        "author_source": str(author_src / "Camcode_full_2.py"),
        "geometric_quantities_source": str(
            author_src / "cytools_catastrophe_scan.py"
        ),
        "geometry_context": {
            "h11": int(charges.shape[1]),
            "leading_count": int(charges.shape[0]),
            "cy_volume": cy_volume,
        },
        "scales": {},
    }
    for scale_name in ("0p9", "1p0", "1p1"):
        k = float(scale_name.replace("p", "."))
        cy_vol_k, tau_k, charges_k, kinv_k = author_scan.geometric_quantities(
            None, k, "unused", "unused", context=context
        )
        reduced = author_code.dim_reductor(
            charges_k, tau_k, kinv_k, cy_vol_k, 0.1, False
        )
        author_map = _coefficient_map(reduced)
        scale_result = {}
        for mode in ("fixed", "full"):
            package_q = _read_int(input_dir / f"{mode}_{scale_name}_Q.txt")
            package_l = _read(input_dir / f"{mode}_{scale_name}_L.txt")
            scale_result[mode] = {
                "package_vs_author": _compare_maps(package_q, package_l, author_map)
            }
        fixed_l = _read(input_dir / f"fixed_{scale_name}_L.txt")
        full_l = _read(input_dir / f"full_{scale_name}_L.txt")
        finite = np.isfinite(fixed_l[1, :]) & np.isfinite(full_l[1, :])
        expected_delta = -3.0 * np.log10(k)
        delta_error = np.max(
            np.abs((full_l[1, finite] - fixed_l[1, finite]) - expected_delta)
        )
        scale_result["full_minus_fixed_log10"] = {
            "expected": float(expected_delta),
            "max_error": float(delta_error),
            "sign_mismatches": int(np.count_nonzero(
                full_l[0, :] != fixed_l[0, :]
            )),
        }
        results["scales"][scale_name] = scale_result
    geometry_index = input_dir / "geometry_index.txt"
    if geometry_index.exists():
        results["geometry_index"] = geometry_index.read_text().strip()
    results["input_dir"] = str(input_dir)
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path)
    group.add_argument("--input-root", type=Path)
    parser.add_argument("--author-src", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    _install_import_stubs()
    sys.path.insert(0, str(args.author_src))
    import Camcode_full_2 as author_code
    import cytools_catastrophe_scan as author_scan

    if args.input_dir is not None:
        result = compare_input_dir(
            args.input_dir, author_code, author_scan, args.author_src
        )
    else:
        input_dirs = sorted(
            path for path in args.input_root.iterdir()
            if path.is_dir() and path.name.startswith("geometry_")
        )
        geometries = []
        for input_dir in input_dirs:
            try:
                geometries.append(
                    compare_input_dir(
                        input_dir, author_code, author_scan, args.author_src
                    )
                )
            except Exception as error:  # preserve per-geometry failures
                geometries.append({
                    "geometry_index": (
                        (input_dir / "geometry_index.txt").read_text().strip()
                        if (input_dir / "geometry_index.txt").exists()
                        else input_dir.name
                    ),
                    "input_dir": str(input_dir),
                    "error": f"{type(error).__name__}: {error}",
                })
        result = {
            "author_source": str(args.author_src / "Camcode_full_2.py"),
            "geometric_quantities_source": str(
                args.author_src / "cytools_catastrophe_scan.py"
            ),
            "geometry_count": len(geometries),
            "geometries": geometries,
        }

    serialized = json.dumps(result, sort_keys=True, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(serialized + "\n")
    print(serialized)
    if _has_validation_failure(result):
        raise SystemExit("author-code coefficient validation failed")


if __name__ == "__main__":
    main()
