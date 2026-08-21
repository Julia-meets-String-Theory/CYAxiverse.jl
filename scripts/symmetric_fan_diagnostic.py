#!/usr/bin/env python3
"""Run a bounded diagnostic for Moritz et al.'s L-symmetric fan route.

This module does not change the supplied-FRST orientifold enumerator.  It
finds involutions that preserve the polytope but not the selected FRST, then
retains the lower-hull subdivision produced by symmetrized heights.

The artifact keeps three identities separate: the supplied FRST, the induced
cell complex, and the final two-face-equivalence FRST class (when the output
is simplicial).  A non-simplicial ambient fan is diagnostic evidence only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

from generate_geometric_data_multitriangulation import load_mirror_polytopes
from glimmers_raw_frst import compute_polytope_id, stable_hash
from inherited_orientifold_candidates import enumerate_polytope_involutions


SCHEMA_VERSION = "cyaxiverse-symmetric-fan-diagnostic-1.0"
SOURCE_CONSTRUCTION = (
    "Moritz et al. arXiv:2305.06363 sec. 4.3: "
    "h_p=(h'_p+h'_{L(p)})/2 followed by a lower regular subdivision"
)
TERMINAL_STATUSES = (
    "constructed",
    "already_represented",
    "two_face_failed",
    "regularity_failed",
    "star_failed",
    "fineness_failed",
    "resource_limited",
    "explicitly_unavailable",
)


class DiagnosticError(RuntimeError):
    """Report an unavailable diagnostic input or capability."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _point_tuple(point: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(point, dtype=int).tolist())


def _canonical_cells(cells: Iterable[Iterable[Iterable[int]]]) -> list[list[list[int]]]:
    """Return a deterministic coordinate representation of a cell complex."""

    normalized = []
    for cell in cells:
        normalized.append(sorted({_point_tuple(ray) for ray in cell}))
    return [[list(ray) for ray in cell] for cell in sorted(normalized)]


def _cell_complex_id(cells: Iterable[Iterable[Iterable[int]]]) -> str:
    """Hash a cell complex without depending on CYTools object ordering."""

    return f"symmetric-subdivision:{stable_hash(_canonical_cells(cells))}"


def _global_simplex_key(triangulation: Any, polytope_points: np.ndarray) -> list[list[int]]:
    """Return selected FRST simplices in global polytope-point indices."""

    point_lookup = {_point_tuple(point): index for index, point in enumerate(polytope_points)}
    local_points = np.asarray(triangulation.points(), dtype=int)
    local_to_global = []
    for point in local_points:
        try:
            local_to_global.append(point_lookup[_point_tuple(point)])
        except KeyError as exc:
            raise DiagnosticError("the selected FRST contains an unknown polytope point") from exc
    return sorted(
        sorted(int(local_to_global[int(index)]) for index in simplex)
        for simplex in np.asarray(triangulation.simplices(as_indices=True), dtype=int)
    )


def _frst_id(triangulation: Any, polytope_points: np.ndarray) -> str:
    """Hash the supplied FRST independently of CYTools labels."""

    return f"supplied-frst:{stable_hash(_global_simplex_key(triangulation, polytope_points))}"


def _mapped_cell_complex(cells: Iterable[Iterable[Iterable[int]]], matrix: np.ndarray) -> list[list[list[int]]]:
    mapped = []
    for cell in cells:
        mapped.append([tuple((matrix @ np.asarray(ray, dtype=int)).tolist()) for ray in cell])
    return _canonical_cells(mapped)


def _is_polytope_preserved(points: np.ndarray, matrix: np.ndarray) -> bool:
    point_set = {_point_tuple(point) for point in points}
    return all(
        _point_tuple(matrix @ np.asarray(point, dtype=int)) in point_set
        for point in points
    )


def _is_frst_preserved(triangulation: Any, polytope_points: np.ndarray, matrix: np.ndarray) -> bool:
    """Check exact FRST preservation using global lattice-point labels."""

    simplices = {tuple(simplex) for simplex in _global_simplex_key(triangulation, polytope_points)}
    point_lookup = {_point_tuple(point): index for index, point in enumerate(polytope_points)}
    mapped = set()
    for simplex in simplices:
        mapped.add(
            tuple(
                sorted(
                    point_lookup[_point_tuple(matrix @ polytope_points[index])]
                    for index in simplex
                )
            )
        )
    return mapped == simplices


def _check_status(
    *,
    invariant: bool | None,
    fine: bool | None,
    regular: bool | None,
    star: bool | None,
    two_face: bool | None,
    class_equivalent: bool = False,
    unavailable_reason: str | None = None,
    resource_limited: bool = False,
) -> tuple[str, str]:
    """Return one terminal status from the ordered source checks."""

    if resource_limited:
        return "resource_limited", unavailable_reason or "diagnostic budget exhausted"
    if unavailable_reason is not None:
        return "explicitly_unavailable", unavailable_reason
    if invariant is not True:
        return "explicitly_unavailable", "cell-complex invariance under L was not certified"
    if fine is False:
        return "fineness_failed", "the subdivision does not use the full point configuration"
    if regular is False:
        return "regularity_failed", "the regularity witness or API check failed"
    if star is False:
        return "star_failed", "the subdivision is not a star subdivision"
    if two_face is False:
        return "two_face_failed", "a two-face restriction contains a non-triangular cell"
    if class_equivalent:
        return "already_represented", "the simplicial subdivision is two-face equivalent to a known FRST class"
    return "constructed", "all requested checks passed"


def classify_symmetric_subdivision_checks(checks: dict[str, Any]) -> dict[str, Any]:
    """Classify synthetic or live check results using the finite vocabulary."""

    status, reason = _check_status(
        invariant=checks.get("invariant"),
        fine=checks.get("fine"),
        regular=checks.get("regular"),
        star=checks.get("star"),
        two_face=checks.get("two_face"),
        class_equivalent=bool(checks.get("class_equivalent", False)),
        unavailable_reason=checks.get("unavailable_reason"),
        resource_limited=bool(checks.get("resource_limited", False)),
    )
    return {"terminal_status": status, "terminal_reason": reason}


def _height_inputs(poly: Any, triangulation: Any, matrix: np.ndarray) -> dict[str, Any]:
    """Build and retain original and symmetrized origin-relative heights."""

    raw_heights = np.asarray(triangulation.heights(), dtype=float).reshape(-1)
    tri_points = np.asarray(triangulation.points(), dtype=int)
    if raw_heights.size != tri_points.shape[0]:
        raise DiagnosticError("height vector length does not match the selected FRST points")
    point_to_height = {
        _point_tuple(point): float(height)
        for point, height in zip(tri_points, raw_heights)
    }
    origin = (0, 0, 0, 0)
    if origin not in point_to_height:
        raise DiagnosticError("the selected FRST height vector has no origin entry")

    vectors = np.asarray(poly.vc().vectors(), dtype=int)
    original = []
    symmetric = []
    for vector in vectors:
        key = _point_tuple(vector)
        image = _point_tuple(matrix @ vector)
        if key not in point_to_height or image not in point_to_height:
            raise DiagnosticError("the FRST heights do not cover the vector configuration")
        original_height = point_to_height[key] - point_to_height[origin]
        image_height = point_to_height[image] - point_to_height[origin]
        original.append(original_height)
        symmetric.append(0.5 * (original_height + image_height))
    return {
        "method": "triangulation.heights; origin-relative for Polytope.vc",
        "origin": list(origin),
        "vector_configuration": vectors.tolist(),
        "original_heights": original,
        "symmetrized_heights": symmetric,
        "formula": "h[p]=(h_prime[p]+h_prime[L(p)])/2",
    }


def _fan_cells(fan: Any) -> list[list[list[int]]]:
    try:
        return _canonical_cells(fan.cells(as_rays=True))
    except (AttributeError, TypeError) as exc:
        raise DiagnosticError("CYTools Fan does not expose maximal cells as rays") from exc


def _fan_boolean(fan: Any, method_name: str) -> bool | None:
    method = getattr(fan, method_name, None)
    if method is None:
        return None
    try:
        return bool(method())
    except (NotImplementedError, RuntimeError, ValueError):
        return None


def _regularity_check(fan: Any) -> dict[str, Any]:
    """Check regularity or retain the exact PPL lower-hull certificate."""

    direct = _fan_boolean(fan, "is_regular")
    if direct is not None:
        return {"passed": direct, "method": "CYTools Fan.is_regular"}
    return {
        "passed": True,
        "method": "source_certified_ppl_lower_hull",
        "note": "Fan.is_regular is not implemented for non-triangulations; retained symmetrized heights are the witness",
    }


def _star_check(fan: Any) -> dict[str, Any]:
    """Use CYTools' Gorenstein-Fano certificate for a conical star fan."""

    method = getattr(fan, "is_gorenstein_fano", None)
    if method is None:
        return {"passed": None, "method": "unavailable"}
    try:
        return {"passed": bool(method()), "method": "CYTools Fan.is_gorenstein_fano"}
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        return {"passed": None, "method": "unavailable", "reason": str(exc)}


def _fineness_check(fan: Any, vector_count: int) -> dict[str, Any]:
    direct = _fan_boolean(fan, "is_fine")
    if direct is not None:
        return {"passed": direct, "method": "CYTools Fan.is_fine", "point_count": vector_count}
    used_labels = getattr(fan, "used_labels", None)
    if used_labels is not None:
        try:
            labels = {int(label) for label in used_labels}
            return {
                "passed": len(labels) == vector_count,
                "method": "CYTools Fan.used_labels",
                "point_count": vector_count,
                "used_label_count": len(labels),
            }
        except (TypeError, ValueError):
            pass
    return {"passed": None, "method": "unavailable", "point_count": vector_count}


def _two_face_check(poly: Any, cells: list[list[list[int]]]) -> dict[str, Any]:
    """Apply the source two-face cell-intersection and coverage checks."""

    faces = getattr(poly, "faces", None)
    if faces is None:
        return {"passed": None, "method": "unavailable", "failures": []}
    try:
        two_faces = faces(d=2)
    except (TypeError, RuntimeError, ValueError):
        return {"passed": None, "method": "unavailable", "failures": []}

    failures = []
    for face_index, face in enumerate(two_faces):
        face_points = {_point_tuple(point) for point in face.points()}
        covered = set()
        for cell_index, cell in enumerate(cells):
            intersection = sorted(face_points.intersection({_point_tuple(ray) for ray in cell}))
            if not intersection:
                continue
            covered.update(intersection)
            if len(intersection) > 3:
                failures.append(
                    {
                        "face_index": face_index,
                        "cell_index": cell_index,
                        "intersection": [list(point) for point in intersection],
                        "reason": "cell meets a two-face in more than three rays",
                    }
                )
        missing = sorted(face_points - covered)
        if missing:
            failures.append(
                {
                    "face_index": face_index,
                    "missing_points": [list(point) for point in missing],
                    "reason": "two-face point configuration is not covered",
                }
            )
    return {
        "passed": not failures,
        "method": "source_two_face_cell_intersection_criterion",
        "face_count": len(two_faces),
        "failures": failures,
    }


def _known_class_equivalence(fan: Any, representatives: Iterable[Any]) -> dict[str, Any]:
    """Compare a simplicial output with existing FRST classes."""

    is_triangulation = getattr(fan, "is_triangulation", None)
    if is_triangulation is None:
        return {"applicable": False, "equivalent": False, "reason": "fan API unavailable"}
    try:
        if not bool(is_triangulation()):
            return {"applicable": False, "equivalent": False, "reason": "non_simplicial_subdivision"}
        candidate = fan.get_pc_triangulation()
    except (NotImplementedError, RuntimeError, ValueError, AttributeError) as exc:
        return {"applicable": False, "equivalent": False, "reason": str(exc)}

    for class_index, reference in enumerate(representatives):
        try:
            if bool(candidate.is_equivalent(reference, on_faces_dim=2)):
                return {
                    "applicable": True,
                    "equivalent": True,
                    "class_index": class_index,
                    "method": "CYTools Triangulation.is_equivalent(on_faces_dim=2)",
                }
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            return {"applicable": False, "equivalent": False, "reason": str(exc)}
    return {
        "applicable": True,
        "equivalent": False,
        "method": "CYTools Triangulation.is_equivalent(on_faces_dim=2)",
    }


def diagnose_pair(
    poly: Any,
    triangulation: Any,
    matrix: np.ndarray,
    *,
    polytope_index: int,
    frst_class_index: int,
    representatives: Iterable[Any] = (),
    deadline: float | None = None,
) -> dict[str, Any]:
    """Attempt one non-preserving ``(FRST, L)`` pair with full provenance."""

    started = time.monotonic()
    points = np.asarray(poly.points(), dtype=int)
    matrix = np.asarray(matrix, dtype=int)
    record: dict[str, Any] = {
        "record_kind": "symmetric_subdivision_attempt",
        "polytope_index": int(polytope_index),
        "frst_class_index": int(frst_class_index),
        "polytope_id": compute_polytope_id(points),
        "supplied_frst_id": _frst_id(triangulation, points),
        "lattice_matrix": matrix.tolist(),
        "source_construction": SOURCE_CONSTRUCTION,
        "population_scope": "supplied_frst_to_induced_symmetric_subdivision",
    }
    if deadline is not None and time.monotonic() >= deadline:
        record.update(classify_symmetric_subdivision_checks({"resource_limited": True}))
        record["elapsed_seconds"] = time.monotonic() - started
        return record
    if not _is_polytope_preserved(points, matrix):
        record.update(classify_symmetric_subdivision_checks({"unavailable_reason": "L does not preserve the supplied polytope"}))
        record["elapsed_seconds"] = time.monotonic() - started
        return record

    try:
        height_data = _height_inputs(poly, triangulation, matrix)
        fan = poly.vc().subdivide(
            heights=np.asarray(height_data["symmetrized_heights"], dtype=float),
            backend="ppl",
            make_fine=False,
            check_heights=False,
            cure_heights=True,
        )
        cells = _fan_cells(fan)
        record["height_inputs"] = height_data
        record["symmetric_subdivision_id"] = _cell_complex_id(cells)
        record["cells"] = cells
        record["maximal_cell_ray_counts"] = [len(cell) for cell in cells]
    except (DiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record.update(classify_symmetric_subdivision_checks({"unavailable_reason": str(exc)}))
        record["elapsed_seconds"] = time.monotonic() - started
        return record

    invariant = _mapped_cell_complex(cells, matrix) == _canonical_cells(cells)
    fine_data = _fineness_check(fan, len(height_data["vector_configuration"]))
    regular_data = _regularity_check(fan)
    star_data = _star_check(fan)
    two_face_data = _two_face_check(poly, cells)
    equivalence = _known_class_equivalence(fan, representatives)
    record["checks"] = {
        "invariant": {"passed": invariant, "method": "exact cell-complex ray-set mapping"},
        "fine": fine_data,
        "regular": regular_data,
        "star": star_data,
        "two_face": two_face_data,
        "class_equivalence": equivalence,
    }
    record.update(
        classify_symmetric_subdivision_checks(
            {
                "invariant": invariant,
                "fine": fine_data["passed"],
                "regular": regular_data["passed"],
                "star": star_data["passed"],
                "two_face": two_face_data["passed"],
                "class_equivalent": equivalence["equivalent"],
            }
        )
    )
    if equivalence.get("equivalent"):
        record["final_frst_class_id"] = f"frst-class:{int(equivalence['class_index'])}"
    elif equivalence.get("applicable"):
        record["final_frst_class_id"] = f"induced-frst:{stable_hash(cells)}"
    else:
        record["final_frst_class_id"] = None
    record["elapsed_seconds"] = time.monotonic() - started
    return _jsonable(record)


def _class_representatives(poly: Any) -> list[Any]:
    raw = poly.all_triangulations(
        only_fine=True,
        only_regular=True,
        only_star=True,
        as_list=True,
    )
    representatives = []
    for triangulation in raw:
        if not any(
            triangulation.is_equivalent(reference, on_faces_dim=2)
            for reference in representatives
        ):
            representatives.append(triangulation)
    return representatives


def scan_population(
    records: Iterable[tuple[Any, dict[str, Any]]],
    *,
    polytope_limit: int | None = None,
    class_limit: int | None = None,
    pair_limit: int | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Scan a bounded population and retain every attempted pair status."""

    started = time.monotonic()
    attempts = []
    pair_count = 0
    selected_polytopes = 0
    selected_classes = 0
    for polytope_index, (poly, _source) in enumerate(records):
        if polytope_limit is not None and selected_polytopes >= polytope_limit:
            break
        selected_polytopes += 1
        representatives = _class_representatives(poly)
        for class_index, triangulation in enumerate(representatives):
            if class_limit is not None and selected_classes >= class_limit:
                break
            selected_classes += 1
            for matrix in enumerate_polytope_involutions(np.asarray(poly.points(), dtype=int)):
                if np.array_equal(matrix, np.eye(4, dtype=int)):
                    continue
                if _is_frst_preserved(triangulation, np.asarray(poly.points(), dtype=int), matrix):
                    continue
                if pair_limit is not None and pair_count >= pair_limit:
                    break
                pair_count += 1
                deadline = None if max_seconds is None else started + max_seconds
                attempts.append(
                    diagnose_pair(
                        poly,
                        triangulation,
                        matrix,
                        polytope_index=polytope_index,
                        frst_class_index=class_index,
                        representatives=representatives,
                        deadline=deadline,
                    )
                )
                if max_seconds is not None and time.monotonic() >= deadline:
                    break
            if pair_limit is not None and pair_count >= pair_limit:
                break
            if max_seconds is not None and time.monotonic() >= started + max_seconds:
                break
        if pair_limit is not None and pair_count >= pair_limit:
            break
        if max_seconds is not None and time.monotonic() >= started + max_seconds:
            break

    status_counts = Counter(attempt["terminal_status"] for attempt in attempts)
    for status in TERMINAL_STATUSES:
        status_counts.setdefault(status, 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_construction": SOURCE_CONSTRUCTION,
        "population": {
            "supplied_frst_class_representatives": "Triangulation.is_equivalent(on_faces_dim=2)",
            "induced_symmetric_subdivision": "one record per (supplied_frst_id, lattice_matrix)",
            "final_frst_class": "two-face equivalence only for simplicial induced subdivisions",
            "selected_polytopes": selected_polytopes,
            "selected_classes": selected_classes,
            "tested_pairs": pair_count,
            "complete": pair_limit is None and max_seconds is None,
        },
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "attempts": attempts,
    }


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11", type=int, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--polytope-limit", type=int)
    parser.add_argument("--class-limit", type=int)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        records = load_mirror_polytopes(
            str(args.parquet_dir),
            args.h11,
            limit=args.polytope_limit or 10**9,
            favorable=True,
        )
        result = scan_population(
            records,
            polytope_limit=args.polytope_limit,
            class_limit=args.class_limit,
            pair_limit=args.pair_limit,
            max_seconds=args.max_seconds,
        )
        result["provenance"] = {
            "h11": args.h11,
            "parquet_dir": str(args.parquet_dir.resolve()),
            "source_commit": _git_commit(Path(__file__).resolve().parent.parent),
            "cli": vars(args),
        }
        encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            output = args.output.resolve()
            if output.exists():
                raise DiagnosticError(f"refusing to overwrite existing output {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
    except (DiagnosticError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
