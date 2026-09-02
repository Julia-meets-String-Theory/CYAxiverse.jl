"""Reconstruct the exact Sheridan trilayer action from lattice data.

The trilayer construction is a source-defined special case, not a label that
can be inferred from a population count.  Moritz, arXiv:2305.06363v1,
§4.8 (``KS_orientifolds.tex`` lines 749--766), gives the source gauge

``L = id``, ``t = p0 / 2``, and ``lambda_f = 1``

when the dual polytope is the convex hull of a facet on which ``p0`` pairs to
``-1`` and one outside vertex on which it pairs to ``+1``.  Sheridan et al.,
arXiv:2412.12012v1, §4.1 (``main.tex`` lines 1295--1299), identifies these
trilayer involutions with the ``h21_plus=0`` inherited orientifolds in the
reported population.

This module reconstructs that finite source-authorized action set from the
ordered primal and dual lattice points.  It does not read aggregate labels or
an accepted witness.  Optional evaluation against a chosen FRST and topology
is deliberately lazy so the structural constructor remains usable without
CYTools.  Missing fan, GLSM, parity, fixed-component, or smoothness evidence
is terminal ``unavailable`` rather than an acceptance shortcut.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import json
from typing import Any

import numpy as np
from sympy import Matrix


RECONSTRUCTION_SCHEMA_VERSION = "cyaxiverse-sheridan-trilayer-exact-1.0"
RECONSTRUCTION_RULE_VERSION = "moritz-4.64-4.66-source-gauge-1"
SOURCE_ANCHORS = {
    "sheridan_population": "arXiv:2412.12012v1 main.tex lines 1295-1299",
    "sheridan_example": "arXiv:2412.12012v1 main.tex lines 1337-1349",
    "moritz_trilayer": "arXiv:2305.06363v1 KS_orientifolds.tex lines 749-766",
    "moritz_fan": "arXiv:2305.06363v1 KS_orientifolds.tex lines 330-339",
    "moritz_h2": "arXiv:2305.06363v1 KS_orientifolds.tex lines 375-385",
    "moritz_parity": "arXiv:2305.06363v1 KS_orientifolds.tex lines 621-629",
    "moritz_fixed_components": "arXiv:2305.06363v1 KS_orientifolds.tex lines 491-555",
    "moritz_smoothness": "arXiv:2305.06363v1 KS_orientifolds.tex lines 610-659",
    "moritz_hodge": "arXiv:2305.06363v1 KS_orientifolds.tex lines 661-668",
}
SOURCE_ARCHIVE_SHA256 = {
    "sheridan_arXiv_2412_12012v1": "905db55f2ab72e2b94ba9175148cd5a4976756e95ce37e64622bffdbc4d7bcea",
    "moritz_arXiv_2305_06363v1": "9e36c8a0a9fa9c5e876329fc6f1d56b46896125842da5f5a44652ea16e48a992",
}
SOURCE_ARCHIVE_PATHS = {
    "sheridan_arXiv_2412_12012v1": "/Users/vmehta/Downloads/fuzzy-2412.12012v1.tar.gz",
    "moritz_arXiv_2305_06363v1": "validation/fuzzy_axions_supp/paper_source_2305_06363/KS_orientifolds.tex",
}


TERMINAL_STATUSES = (
    "accepted_exact_trilayer_action",
    "structurally_reconstructed",
    "not_trilayer",
    "invalid_primal_vertex_data",
    "invalid_dual_vertex_data",
    "dual_facet_not_three_dimensional",
    "dual_convex_hull_certificate_unavailable",
    "dual_vertex_outside_facet_ambiguous",
    "fan_evidence_unavailable",
    "fan_not_preserved",
    "topology_evidence_unavailable",
    "polytope_not_preserved",
    "frst_not_preserved",
    "prime_divisor_set_not_preserved",
    "nonintegral_h2_action",
    "h2_action_not_involution",
    "action_not_involution",
    "eq_4_45_parity_unavailable",
    "eq_4_45_parity_failure",
    "fixed_component_unavailable",
    "fixed_point_set_non_smooth",
    "smoothness_verification_unavailable",
    "fixed_locus_euler_unavailable",
    "mpcp_certificate_unavailable",
    "mpcp_certificate_mismatch",
    "h21_plus_nonzero",
)


def _integer_rows(values: Any, *, name: str) -> np.ndarray:
    """Convert an ordered lattice-point collection to an exact integer array."""
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n, 4), got {array.shape}")
    try:
        integer = np.asarray(array, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not np.array_equal(array, integer):
        raise ValueError(f"{name} must contain exact integers")
    if len({tuple(int(value) for value in row) for row in integer}) != len(integer):
        raise ValueError(f"{name} contains duplicate lattice points")
    return integer


def _points_from_polytope(poly, *, name: str) -> np.ndarray:
    """Read one exact point array from a CYTools-like polytope object."""
    method = getattr(poly, name, None)
    if method is None:
        raise ValueError(f"polytope does not expose {name}()")
    return _integer_rows(method(), name=name)


def _dual_polytope(poly, dual_polytope=None):
    if dual_polytope is not None:
        return dual_polytope
    method = getattr(poly, "dual", None)
    if method is None:
        raise ValueError("polytope does not expose dual()")
    return method()


def _primitive_normal(rows):
    """Return a primitive integer normal to exact rows, or ``None``."""
    nullspace = Matrix(np.asarray(rows, dtype=object).tolist()).nullspace()
    if len(nullspace) != 1:
        return None
    vector = [Fraction(value) for value in nullspace[0]]
    denominator = 1
    for value in vector:
        denominator = np.lcm(denominator, value.denominator)
    integer = [int(value * denominator) for value in vector]
    divisor = 0
    for value in integer:
        divisor = int(np.gcd(divisor, abs(value)))
    if divisor == 0:
        return None
    integer = [value // divisor for value in integer]
    first = next(value for value in integer if value)
    if first < 0:
        integer = [-value for value in integer]
    return tuple(integer)


def _facet_support_inequalities(facet):
    """Compute exact supporting inequalities of a 3-dimensional facet.

    The facet lies in the affine hyperplane ``p0.q=-1``.  Three affinely
    independent facet vertices determine each candidate two-face normal.  A
    candidate is retained only when all facet vertices lie on one exact side;
    no floating-point convex-hull tolerance is used.
    """
    facet = [tuple(int(value) for value in point) for point in facet]
    if len(facet) < 4:
        return None
    inequalities = set()
    for points in itertools.combinations(facet, 3):
        anchor, left, right = points
        rows = [
            [left[index] - anchor[index] for index in range(4)],
            [right[index] - anchor[index] for index in range(4)],
        ]
        # The facet's affine hyperplane is supplied by the caller through the
        # first row of ``_facet_membership``.  A 2-face normal is determined
        # after that row is appended there.
        if Matrix(rows).rank() != 2:
            continue
        inequalities.add((anchor, tuple(rows)))
    # The function is completed by _facet_membership, which knows p0.  The
    # provisional tuple keeps the exact point combinations deterministic.
    return tuple(sorted(inequalities))


def _facet_membership(point, facet, p0):
    """Test exact membership in the convex hull of a dual facet."""
    point = tuple(Fraction(value) for value in point)
    facet = [tuple(Fraction(value) for value in row) for row in facet]
    p0 = tuple(Fraction(value) for value in p0)
    if not facet:
        return False
    if sum(a * b for a, b in zip(p0, point)) != -1:
        return False
    supports = []
    for anchor, left, right in itertools.combinations(facet, 3):
        rows = [
            list(p0),
            [left[index] - anchor[index] for index in range(4)],
            [right[index] - anchor[index] for index in range(4)],
        ]
        normal = _primitive_normal(rows)
        if normal is None:
            continue
        values = [
            sum(Fraction(normal[index]) * (vertex[index] - anchor[index]) for index in range(4))
            for vertex in facet
        ]
        if all(value >= 0 for value in values) or all(value <= 0 for value in values):
            supports.append((normal, anchor, min(values), max(values)))
    # A full-dimensional facet has a finite supporting half-space description
    # from its two-faces.  If no such description can be constructed, retain a
    # terminal unavailable result rather than accepting an unproved hull.
    if not supports:
        return None
    for normal, anchor, _, _ in supports:
        values = [
            sum(Fraction(normal[index]) * (vertex[index] - anchor[index]) for index in range(4))
            for vertex in facet
        ]
        point_value = sum(
            Fraction(normal[index]) * (point[index] - anchor[index]) for index in range(4)
        )
        if all(value >= 0 for value in values) and point_value < 0:
            return False
        if all(value <= 0 for value in values) and point_value > 0:
            return False
    return True


def _canonical_fraction(value):
    value = Fraction(value)
    return {"numerator": int(value.numerator), "denominator": int(value.denominator)}


def fraction_vector(vector):
    """Return a JSON-stable exact rational vector representation."""
    values = tuple(Fraction(value) for value in vector)
    denominator = 1
    for value in values:
        denominator = int(np.lcm(denominator, value.denominator))
    numerator = [int(value * denominator) for value in values]
    divisor = 0
    for value in numerator + [denominator]:
        divisor = int(np.gcd(divisor, abs(value)))
    if divisor > 1:
        numerator = [value // divisor for value in numerator]
        denominator //= divisor
    return {"numerator": numerator, "denominator": int(denominator)}


def decode_fraction_vector(value):
    """Decode the exact vector representation emitted by this module."""
    if isinstance(value, dict) and "numerator" in value:
        denominator = int(value["denominator"])
        if denominator <= 0:
            raise ValueError("rational vector denominator must be positive")
        return tuple(Fraction(int(item), denominator) for item in value["numerator"])
    return tuple(Fraction(item) for item in value)


def action_digest(action):
    """Hash exactly ``(L,t,lambda_f)`` under the source gauge."""
    payload = {
        "lattice_matrix": action["lattice_matrix"],
        "torus_shift": action["torus_shift"],
        "lambda_f": int(action["lambda_f"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def lattice_matrix_digest(matrix):
    """Hash the exact lattice matrix independently of shift/parity."""
    array = np.asarray(matrix, dtype=np.int64)
    if array.shape != (4, 4):
        raise ValueError("lattice matrix must have shape (4, 4)")
    return hashlib.sha256(
        json.dumps(array.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def frst_identifier(simplices):
    """Hash an exact simplex set independently of enumeration order."""
    array = np.asarray(simplices, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("FRST simplices must be a two-dimensional array")
    canonical = sorted(tuple(sorted(int(value) for value in row)) for row in array.tolist())
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def polytope_identifier(points):
    """Hash the canonical lattice-point set as a replay identity."""
    rows = _integer_rows(points, name="polytope points")
    canonical_rows = sorted(tuple(int(value) for value in row) for row in rows.tolist())
    digest = hashlib.sha256(
        json.dumps(canonical_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"lattice-points-sha256:{digest}"


def _source_provenance():
    return {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "reconstruction_rule_version": RECONSTRUCTION_RULE_VERSION,
        "construction": "enumerate every primal vertex p0 satisfying Moritz trilayer convex-hull criterion; emit L=I, t=p0/2, lambda_f=1",
        "source_gauge": "N=Z^4 ordered coordinates; no affine translation; dual action is contragredient",
        "source_anchors": dict(SOURCE_ANCHORS),
        "source_archive_sha256": dict(SOURCE_ARCHIVE_SHA256),
        "source_archive_paths": dict(SOURCE_ARCHIVE_PATHS),
        "aggregate_labels_used": False,
        "unpublished_witnesses_used": False,
    }


def enumerate_source_trilayer_candidates(poly, *, dual_polytope=None):
    """Enumerate all source-authorized structural trilayer actions.

    Return one record per primal vertex, including terminal rejection records.
    Multiple qualifying vertices are retained: the source construction does
    not authorize selecting one by class index or by a population target.
    """
    provenance = _source_provenance()
    try:
        primal_vertices = _points_from_polytope(poly, name="vertices")
    except ValueError as exc:
        return [{**provenance, "terminal_status": "invalid_primal_vertex_data", "reason": str(exc)}]
    try:
        dual = _dual_polytope(poly, dual_polytope)
        dual_vertices = _points_from_polytope(dual, name="vertices")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return [{
            **provenance,
            "polytope_id": polytope_identifier(_points_from_polytope(poly, name="points"))
            if getattr(poly, "points", None) is not None else None,
            "terminal_status": "invalid_dual_vertex_data",
            "reason": str(exc),
        }]
    if primal_vertices.shape[1] != 4 or dual_vertices.shape[1] != 4:
        return [{**provenance, "terminal_status": "invalid_dual_vertex_data", "reason": "source construction is four-dimensional"}]
    try:
        primal_points = _points_from_polytope(poly, name="points")
        identifier = polytope_identifier(primal_points)
    except ValueError:
        identifier = None
    records = []
    for vertex_index, p0_array in enumerate(primal_vertices):
        p0 = tuple(int(value) for value in p0_array)
        base = {
            **provenance,
            "polytope_id": identifier,
            "primal_vertex_index": int(vertex_index),
            "p0": list(p0),
            "dual_vertex_count": int(len(dual_vertices)),
        }
        pairings = [sum(int(a) * int(b) for a, b in zip(p0, q)) for q in dual_vertices]
        facet = [tuple(int(value) for value in q) for q, pairing in zip(dual_vertices, pairings) if pairing == -1]
        outside = [
            (index, tuple(int(value) for value in q), pairing)
            for index, (q, pairing) in enumerate(zip(dual_vertices, pairings))
            if pairing > -1
        ]
        if not facet:
            records.append({**base, "terminal_status": "not_trilayer", "reason": "p0 has no dual facet at pairing -1"})
            continue
        facet_rank = Matrix(
            [
                [point[index] - facet[0][index] for index in range(4)]
                for point in facet[1:]
            ]
        ).rank()
        if facet_rank != 3:
            records.append({
                **base,
                "terminal_status": "dual_facet_not_three_dimensional",
                "reason": "the p0.q=-1 dual face is not an affine three-dimensional facet",
                "pairings": pairings,
                "facet_vertex_count": len(facet),
                "facet_affine_rank": int(facet_rank),
            })
            continue
        if any(pairing < -1 for pairing in pairings):
            records.append({**base, "terminal_status": "not_trilayer", "reason": "dual vertex violates reflexive facet inequality p0.q >= -1", "pairings": pairings})
            continue
        if len(outside) != 1 or outside[0][2] != 1:
            records.append({
                **base,
                "terminal_status": "dual_vertex_outside_facet_ambiguous",
                "reason": "source criterion requires exactly one outside dual vertex with p0.q=+1",
                "pairings": pairings,
                "outside_vertices": [[int(index), list(q), int(pairing)] for index, q, pairing in outside],
            })
            continue
        q0_index, q0, _ = outside[0]
        if any(pairing not in (-1, 0, 1) for pairing in pairings):
            records.append({**base, "terminal_status": "not_trilayer", "reason": "dual polytope has a layer outside {-1,0,+1}", "pairings": pairings})
            continue
        # Every zero-layer vertex must lie in the convex hull of the facet and
        # q0.  At height zero this is equivalent to 2q-q0 lying in the facet.
        hull_checks = []
        hull_unavailable = False
        hull_outside = False
        for q, pairing in zip(dual_vertices, pairings):
            if pairing != 0:
                continue
            target = tuple(Fraction(2 * int(value) - int(q0[index])) for index, value in enumerate(q))
            membership = _facet_membership(target, facet, p0)
            if membership is None:
                hull_unavailable = True
                break
            hull_checks.append({"vertex": [int(value) for value in q], "translated_facet_point": [int(value) for value in target], "in_facet_hull": bool(membership)})
            if not membership:
                hull_outside = True
                records.append({
                    **base,
                    "terminal_status": "not_trilayer",
                    "reason": "a zero-layer dual vertex is outside conv(facet,q0)",
                    "pairings": pairings,
                    "q0": list(q0),
                    "hull_checks": hull_checks,
                })
                break
        if hull_outside:
            continue
        if hull_unavailable:
            records.append({**base, "terminal_status": "dual_convex_hull_certificate_unavailable", "reason": "exact facet hull inequalities could not be constructed", "pairings": pairings, "q0": list(q0)})
            continue
        else:
            action = {
                "lattice_matrix": np.eye(4, dtype=np.int64).tolist(),
                "torus_shift": fraction_vector(tuple(Fraction(value, 2) for value in p0)),
                "lambda_f": 1,
            }
            candidate = {
                **base,
                "terminal_status": "structurally_reconstructed",
                "reason": None,
                "pairings": pairings,
                "dual_facet_vertex_indices": [int(index) for index, pairing in enumerate(pairings) if pairing == -1],
                "q0_vertex_index": int(q0_index),
                "q0": list(q0),
                "hull_checks": hull_checks,
                "action": action,
                "action_digest": action_digest(action),
                # Keep the complete action at record level as well as under
                # ``action`` so downstream serializers cannot accidentally
                # persist only the source-derived p0 label.
                "lattice_matrix": action["lattice_matrix"],
                "torus_shift": action["torus_shift"],
                "lambda_f": int(action["lambda_f"]),
            }
            records.append(candidate)
            continue
        # The loop broke because a zero-layer point failed membership.
        continue
    if not records:
        records.append({**provenance, "terminal_status": "not_trilayer", "reason": "no primal vertices were available"})
    return records


def fan_preservation_evidence(poly, triangulation, matrix):
    """Check exact preservation of the chosen fan and return replay evidence."""
    if triangulation is None:
        return {"status": "fan_evidence_unavailable", "reason": "a chosen FRST/fan is required"}
    matrix = np.asarray(matrix, dtype=np.int64)
    try:
        points = _points_from_polytope(poly, name="points")
        point_lookup = {tuple(int(value) for value in row): index for index, row in enumerate(points)}
        tri_points = _points_from_polytope(triangulation, name="points")
        tri_global = [point_lookup[tuple(int(value) for value in row)] for row in tri_points]
        simplices = np.asarray(triangulation.simplices(as_indices=True), dtype=np.int64)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return {"status": "fan_evidence_unavailable", "reason": str(exc)}
    mapped_indices = []
    for point in points:
        mapped = tuple(int(value) for value in matrix @ point)
        if mapped not in point_lookup:
            return {"status": "fan_not_preserved", "reason": "matrix does not preserve the polytope point set", "mapped_point": list(mapped)}
        mapped_indices.append(point_lookup[mapped])
    fan = {tuple(sorted(int(tri_global[int(index)]) for index in simplex)) for simplex in simplices}
    mapped_fan = {tuple(sorted(int(mapped_indices[index]) for index in simplex)) for simplex in fan}
    return {
        "status": "fan_preserved" if fan == mapped_fan else "fan_not_preserved",
        "reason": None if fan == mapped_fan else "matrix does not preserve the selected FRST simplices",
        "simplex_count": len(fan),
        "point_count": len(points),
        "frst_hash": frst_identifier(simplices),
    }


def action_involution_evidence(action):
    """Check the exact affine involution conditions for ``(L,t)``."""
    try:
        matrix = np.asarray(action["lattice_matrix"], dtype=np.int64)
        shift = decode_fraction_vector(action["torus_shift"])
        matrix_shape_ok = matrix.shape == (4, 4)
        linear_involution = matrix_shape_ok and np.array_equal(
            matrix @ matrix, np.eye(4, dtype=np.int64)
        )
        two_shift = tuple(2 * value for value in shift)
        integral_shift = all(value.denominator == 1 for value in two_shift)
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            "status": "action_not_involution",
            "linear_involution": False,
            "integral_two_t": False,
        }
    return {
        "status": "action_involution_passed"
        if linear_involution and integral_shift
        else "action_not_involution",
        "linear_involution": bool(linear_involution),
        "integral_two_t": bool(integral_shift),
        "two_t": [_canonical_fraction(value) for value in two_shift],
    }


def eq_4_45_parity_evidence(
    poly, action, *, dual_vertices=None, extra_dual_vertices=None
):
    """Evaluate Moritz eq. (4.45) with exact rational arithmetic."""
    matrix = np.asarray(action["lattice_matrix"], dtype=np.int64)
    t = decode_fraction_vector(action["torus_shift"])
    if dual_vertices is None:
        try:
            dual_vertices = _points_from_polytope(_dual_polytope(poly), name="vertices")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return {"status": "eq_4_45_parity_unavailable", "reason": str(exc), "checked_vertices": [], "violations": []}
    else:
        try:
            dual_vertices = _integer_rows(dual_vertices, name="dual vertices")
        except ValueError as exc:
            return {"status": "eq_4_45_parity_unavailable", "reason": str(exc), "checked_vertices": [], "violations": []}
    two_t = tuple(2 * value for value in t)
    checked = []
    violations = []
    extra = {
        tuple(int(value) for value in vertex)
        for vertex in (extra_dual_vertices or ())
    }
    for q in dual_vertices:
        q_tuple = tuple(int(value) for value in q)
        if not np.array_equal(matrix.T @ q, q) and q_tuple not in extra:
            continue
        total = sum(left * right for left, right in zip(two_t, q_tuple)) + int(action["lambda_f"])
        row = {"q": list(q_tuple), "pairing_2t_q": _canonical_fraction(total - int(action["lambda_f"])), "total": _canonical_fraction(total)}
        checked.append(row)
        if total.denominator != 1 or int(total) % 2:
            violations.append(row)
    return {
        "status": "eq_4_45_parity_failure" if violations else "eq_4_45_parity_passed",
        "condition": (
            "2*<t,q> + lambda_f = 0 mod 2 for L-fixed dual vertices and "
            "dual vertices at non-smooth/non-simplicial facets"
        ),
        "checked_vertices": checked,
        "violations": violations,
    }


def _hodge_split(h11, h21, h11_minus, chi_fixed, chi_x):
    values = (h11, h21, h11_minus, chi_fixed, chi_x)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("Hodge and Euler inputs must be exact integers")
    delta = int(chi_fixed) - int(chi_x)
    if delta % 4:
        raise ValueError("chi(F_I)-chi(X) is not divisible by four")
    h21_minus = int(h11_minus) + delta // 4 - 1
    result = {
        "h11_plus": int(h11) - int(h11_minus),
        "h11_minus": int(h11_minus),
        "h21_plus": int(h21) - h21_minus,
        "h21_minus": h21_minus,
    }
    if min(result.values()) < 0:
        raise ValueError("eq. (4.51) produced a negative Hodge eigenspace")
    result.update({"chi_fixed_locus": int(chi_fixed), "chi_x": int(chi_x)})
    return result


def evaluate_exact_trilayer_action(
    poly,
    triangulation,
    topology,
    structural_record,
    *,
    mpcp_certificate=None,
    source_record=None,
):
    """Evaluate one reconstructed action against fan, GLSM, parity and smoothness.

    This function imports the existing exact source kernels only when called.
    Every missing or failed stage retains a terminal status and its evidence.
    It never falls back to a summary label or an identity-only special shift.
    """
    if structural_record.get("terminal_status") != "structurally_reconstructed":
        return dict(structural_record)
    result = dict(structural_record)
    action = dict(structural_record["action"])
    result["action"] = action
    result["action_digest"] = action_digest(action)
    involution = action_involution_evidence(action)
    result["action_involution"] = involution
    if involution["status"] != "action_involution_passed":
        result["terminal_status"] = "action_not_involution"
        result["reason"] = "source action does not define an affine involution"
        return result
    result["matrix_digest"] = lattice_matrix_digest(action["lattice_matrix"])
    fan = fan_preservation_evidence(poly, triangulation, np.asarray(action["lattice_matrix"], dtype=np.int64))
    result["fan_preservation"] = fan
    if fan.get("frst_hash") is not None:
        result["frst_hash"] = fan["frst_hash"]
        result["candidate_id"] = hashlib.sha256(
            json.dumps(
                [result.get("polytope_id"), fan["frst_hash"], result["action_digest"]],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    # A certificate is an optional bounded replay witness, never an outcome
    # selector.  Verify immutable identity before using any certificate data;
    # the exact action path below still recomputes its own topology evidence.
    if mpcp_certificate is not None:
        try:
            from mpcp_bounded_analysis import (
                validate_population_input_certificate,
                validate_replay_certificate,
            )
            source = source_record.get("source", source_record) if isinstance(source_record, dict) else None
            if isinstance(source, dict):
                global_points = source.get("global_points")
                if isinstance(global_points, np.ndarray):
                    global_points = global_points.tolist()
                source = {
                    "source_sha256": source.get("source_sha256", source.get("parquet_sha256")),
                    "source_row": source.get("source_row", source.get("row_index")),
                    "polytope_id": source.get("polytope_id"),
                    "global_points": global_points,
                }
            certificate_source = mpcp_certificate.get("source", {})
            live_points = _points_from_polytope(poly, name="points")
            live_polytope_id = polytope_identifier(live_points)
            expected_polytope_id = certificate_source.get("polytope_id")
            expected_points = certificate_source.get("global_points")
            if expected_polytope_id != live_polytope_id:
                raise ValueError("certificate polytope_id does not match live global coordinates")
            if expected_points is None or polytope_identifier(expected_points) != live_polytope_id:
                raise ValueError("certificate global coordinates do not match the live polytope")
            if mpcp_certificate.get("certificate_schema_version") == "cyaxiverse-population-input-certificate-1.0":
                certificate_check = validate_population_input_certificate(
                    mpcp_certificate,
                    source=source,
                    frst_hash=fan.get("frst_hash"),
                    action=action,
                    requested_h11=int(poly.h11()),
                )
            else:
                certificate_check = validate_replay_certificate(
                    mpcp_certificate,
                    source=source,
                    frst_hash=fan.get("frst_hash"),
                    action=action,
                )
        except (ImportError, TypeError, ValueError) as exc:
            certificate_check = {
                "status": "invalid",
                "terminal": True,
                "reasons": [f"certificate validator unavailable: {exc}"],
            }
        # Preserve the full certificate for downstream writers; keep the
        # verification result in a separate field so adding runtime metadata
        # cannot invalidate the certificate digest.
        result["mpcp_certificate"] = dict(mpcp_certificate)
        result["mpcp_certificate_verification"] = certificate_check
        if certificate_check.get("status") != "valid":
            result["terminal_status"] = "mpcp_certificate_mismatch"
            result["reason"] = "; ".join(certificate_check.get("reasons", []))
            return result
    elif source_record is not None:
        # Source-certified callers must explicitly supply the bounded replay
        # witness.  The no-certificate path remains available for historical
        # structural diagnostics, but it cannot produce source-certified
        # acceptance until the bounded replay is rerun.
        result["mpcp_certificate"] = {
            "status": "missing_recompute_required",
            "terminal": True,
            "reasons": ["no bounded MPCP certificate was supplied; bounded replay must be rerun"],
        }
        result["terminal_status"] = "mpcp_certificate_unavailable"
        result["reason"] = result["mpcp_certificate"]["reasons"][0]
        return result
    if fan["status"] != "fan_preserved":
        result["terminal_status"] = fan["status"]
        return result
    if not isinstance(topology, dict):
        result["terminal_status"] = "topology_evidence_unavailable"
        result["reason"] = "exact GLSM and fixed-component topology are required"
        return result
    try:
        from generate_geometric_data_multitriangulation import (
            OrientifoldValidationFailure,
            validate_orientifold,
        )
    except ImportError as exc:
        result["terminal_status"] = "topology_evidence_unavailable"
        result["reason"] = f"exact GLSM validator unavailable: {exc}"
        return result
    config = {
        "requested": True,
        "status": "input_loaded",
        "lattice_matrix": np.asarray(action["lattice_matrix"], dtype=np.int64),
        "involution_type": "O3/O7",
        "torus_shift": action["torus_shift"],
        "lambda_f": 1,
        "canonical_action_required": True,
        "action_digest": result["action_digest"],
    }
    try:
        validated = validate_orientifold(poly, triangulation, topology, config)
    except OrientifoldValidationFailure as exc:
        result["terminal_status"] = exc.stage or "topology_evidence_unavailable"
        result["reason"] = str(exc)
        return result
    except (ArithmeticError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        result["terminal_status"] = "topology_evidence_unavailable"
        result["reason"] = str(exc)
        return result
    result["h2_action"] = {
        "matrix": np.asarray(validated["h2_involution_matrix"], dtype=np.int64).tolist(),
        "proof": validated["h2_action_proof"],
        "h11_plus": int(validated["h11_plus"]),
        "h11_minus": int(validated["h11_minus"]),
    }
    parity = eq_4_45_parity_evidence(
        poly,
        action,
        extra_dual_vertices=topology.get("non_smooth_facet_dual_vertices", ()),
    )
    result["eq_4_45_parity"] = parity
    if parity["status"] != "eq_4_45_parity_passed":
        result["terminal_status"] = parity["status"]
        result["reason"] = "source eq. (4.45) parity failed"
        return result
    try:
        import inherited_orientifold_candidates as ioc
        import orientifold_general_l_geometry as general_l
        triangulation_cones = ioc._triangulation_cones(poly, triangulation)
        matrix = np.asarray(action["lattice_matrix"], dtype=np.int64)
        fixed_cone_keys = general_l._pointwise_invariant_cone_keys(triangulation_cones, matrix)
        dual_polytope = _dual_polytope(poly)
        dual_points_method = getattr(dual_polytope, "points", None)
        if not callable(dual_points_method):
            dual_points_method = getattr(dual_polytope, "vertices", None)
        dual_points = None if not callable(dual_points_method) else np.asarray(
            dual_points_method(), dtype=np.int64
        )
        ambient_rays = sorted({ray for cone in triangulation_cones for ray in cone})
        auxiliary_fan = general_l.build_auxiliary_fan(triangulation_cones, matrix)
        fixed_components = general_l._fixed_component_records(
            auxiliary_fan,
            matrix,
            decode_fraction_vector(action["torus_shift"]),
            1,
            fixed_cone_keys=fixed_cone_keys,
            dual_points=dual_points,
            ambient_rays=ambient_rays,
            fan_cones=triangulation_cones,
        )
        result["fixed_components"] = fixed_components
        fixed_surface_evidence = topology.get("fixed_surface_n_s", {})
        if not fixed_surface_evidence and any(
            item.get("f_vanishes_identically") is True
            and int(item.get("fixed_toric_dimension", -1)) == 2
            for item in fixed_components
        ):
            fixed_surface_evidence = ioc.identity_fixed_surface_n_s_table(triangulation_cones, triangulation)
        local_topology = dict(topology)
        local_topology["fixed_surface_n_s"] = fixed_surface_evidence
        local_topology.setdefault("non_smooth_facet_dual_vertices", ioc.facets_with_non_smooth_cones(poly, triangulation))
        smoothness = general_l.classify_smoothness(
            matrix,
            decode_fraction_vector(action["torus_shift"]),
            1,
            auxiliary_fan,
            fixed_components,
            local_topology,
            _points_from_polytope(_dual_polytope(poly), name="vertices"),
        )
        result["smoothness"] = smoothness
        if smoothness.get("status") != "smooth":
            result["terminal_status"] = smoothness.get("status", "smoothness_verification_unavailable")
            result["reason"] = "; ".join(smoothness.get("reasons", []))
            return result
        import toric_fixed_component_euler as toric_euler
        fixed_euler = toric_euler.exact_fixed_locus_euler(
            auxiliary_fan,
            matrix,
            fixed_components,
            fixed_surface_n_s_evidence=fixed_surface_evidence,
        )
        result["fixed_locus_euler"] = fixed_euler
        if fixed_euler.get("status") != "computed":
            result["terminal_status"] = "fixed_locus_euler_unavailable"
            result["reason"] = fixed_euler.get("reason", "fixed-locus Euler evidence unavailable")
            return result
    except (ArithmeticError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        result["terminal_status"] = "fixed_component_unavailable"
        result["reason"] = str(exc)
        return result
    cy = getattr(triangulation, "get_cy", lambda: None)()
    if cy is not None and hasattr(cy, "h11") and hasattr(cy, "h21") and hasattr(cy, "chi"):
        try:
            result["hodge_split"] = _hodge_split(
                int(cy.h11()),
                int(cy.h21()),
                int(validated["h11_minus"]),
                int(result["fixed_locus_euler"]["chi_F_I"]),
                int(cy.chi()),
            )
            if result["hodge_split"]["h21_plus"] != 0:
                result["terminal_status"] = "h21_plus_nonzero"
                result["reason"] = "the exact Eq. (4.51) split is not a Sheridan h21_plus=0 action"
                return result
            if (
                mpcp_certificate is not None
                and mpcp_certificate.get("certificate_schema_version")
                != "cyaxiverse-population-input-certificate-1.0"
            ):
                expected_result = mpcp_certificate.get("result", {})
                if expected_result.get("chi_F_I") != result["fixed_locus_euler"].get("chi_F_I"):
                    result["terminal_status"] = "mpcp_certificate_mismatch"
                    result["reason"] = "certificate fixed-locus Euler does not match exact recomputation"
                    return result
                if expected_result.get("hodge_split") != result["hodge_split"]:
                    result["terminal_status"] = "mpcp_certificate_mismatch"
                    result["reason"] = "certificate Hodge split does not match exact recomputation"
                    return result
        except (TypeError, ValueError) as exc:
            result["terminal_status"] = "fixed_locus_euler_unavailable"
            result["reason"] = str(exc)
            return result
    result["terminal_status"] = "accepted_exact_trilayer_action"
    result["reason"] = None
    return result


def _certificate_for_structural_action(certificates, structural_record):
    """Select a certificate only by its exact action witness digest.

    A source row or class index alone never selects a certificate.  A mapping
    may contain one certificate, a ``certificates`` list, or be keyed by an
    action digest; unmatched actions are evaluated without that certificate
    so all structural terminal records remain visible.
    """

    if certificates is None:
        return None
    candidates = certificates
    if isinstance(certificates, dict) and "certificates" in certificates:
        candidates = certificates["certificates"]
    if isinstance(candidates, dict) and "certificate_digest" not in candidates:
        candidates = list(candidates.values())
    if isinstance(candidates, dict):
        candidates = [candidates]
    if not isinstance(candidates, (list, tuple)):
        return None
    expected_digest = structural_record.get("action_digest")
    for certificate in candidates:
        if not isinstance(certificate, dict):
            continue
        action = certificate.get("action")
        if isinstance(action, dict) and action.get("digest") == expected_digest:
            return certificate
    return None


def reconstruct_trilayer_actions(
    poly,
    triangulation=None,
    topology=None,
    *,
    dual_polytope=None,
    mpcp_certificate=None,
    source_record=None,
):
    """Reconstruct and, when evidence is supplied, exactly evaluate all actions."""
    structural = enumerate_source_trilayer_candidates(poly, dual_polytope=dual_polytope)
    evaluated = []
    for record in structural:
        if record.get("terminal_status") != "structurally_reconstructed":
            evaluated.append(record)
        else:
            certificate = _certificate_for_structural_action(
                mpcp_certificate, record
            )
            evaluated.append(
                evaluate_exact_trilayer_action(
                    poly,
                    triangulation,
                    topology,
                    record,
                    mpcp_certificate=certificate,
                    source_record=source_record,
                )
            )
    return {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "reconstruction_rule_version": RECONSTRUCTION_RULE_VERSION,
        "provenance": _source_provenance(),
        "polytope_id": next((record.get("polytope_id") for record in evaluated if record.get("polytope_id")), None),
        "candidate_count": len(evaluated),
        "candidates": evaluated,
        "terminal_status_counts": {
            status: sum(record.get("terminal_status") == status for record in evaluated)
            for status in sorted({record.get("terminal_status") for record in evaluated})
        },
        "selection_rule": "retain every source-authorized p0/action; no class-index, witness, or target-count selection",
    }
