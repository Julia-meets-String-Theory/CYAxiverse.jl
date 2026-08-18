#!/usr/bin/env python3
"""Reproduce the h11=4 population benchmarks of arXiv:2412.12012.

This is a source-matched audit driver, not a replacement for the production
geometry generator.  It reads the same KS Parquet mirror as the generator,
counts FRST classes using the paper's two-face equivalence relation, and
records the special trilayer involution together with the frozen-conifold
diagnostic used by the paper's orientifold cut.

The orientifold and model stages are intentionally represented as explicit
diagnostic records.  A count is labelled ``exact`` only when the implementation
has the corresponding source criterion and complete input evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from generate_geometric_data_multitriangulation import (
    DIVISOR_VOLUME_TOLERANCE,
    configure_mosek_license,
    evaluate_kaehler_point,
    extract_topology,
    load_mirror_polytopes,
    sample_stretched_kaehler_points,
)
from geometry_charge_conventions import canonicalize_unique_charge_rows
from inherited_orientifold_candidates import (
    enumerate_polytope_involutions,
    enumerate_projected_lattice_representatives,
)


PAPER_TARGETS = {
    "favorable_polytopes": 1185,
    "frst_classes": 1760,
    "inherited_orientifold_cys": 1559,
    "h11_minus_zero_orientifold_cys": 1554,
    "h11_minus_zero_h21_plus_zero_orientifold_cys": 267,
    "models": 3348,
}


def _as_int_rows(values: Any) -> np.ndarray:
    rows = np.asarray(values, dtype=int)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError(f"expected an (n, 4) integer array, got {rows.shape}")
    return rows


def _frst_classes(poly):
    """Return one representative per paper FRST class.

    The paper identifies triangulations when their restrictions to all
    two-faces agree up to a polytope automorphism.  CYTools exposes this as
    ``Triangulation.is_equivalent(..., on_faces_dim=2)``.
    """

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
    return raw, representatives


def _trilayer_candidate(poly):
    """Return the source [21] trilayer candidate, if one exists.

    For a primal vertex p0, the dual facet is q.p0=-1.  The special source
    construction requires the dual polytope to be the convex hull of that
    facet and one vertex q0 outside it, with q0.p0=+1.  The associated data are
    L=I, t=p0/2, and lambda_f=1.
    """

    primal_vertices = _as_int_rows(poly.vertices())
    dual = poly.dual()
    dual_vertices = _as_int_rows(dual.vertices())
    for p0 in primal_vertices:
        heights = dual_vertices @ p0
        outside = np.flatnonzero(heights > -1)
        if outside.size != 1:
            continue
        q0_index = int(outside[0])
        if int(heights[q0_index]) != 1:
            continue
        if np.any(heights < -1):
            continue
        return {
            "p0": p0.tolist(),
            "q0": dual_vertices[q0_index].tolist(),
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift_numerator": p0.tolist(),
            "torus_shift_denominator": 2,
            "lambda_f": 1,
            "criterion": "Moritz eqs. (4.64)-(4.66), trilayer sufficient condition",
        }
    return None


def _identity_torus_actions(poly):
    """Return the source identity-linear-action torus representatives.

    CYTools returns integer numerators q for representatives t=q/2 modulo
    lattice automorphisms.  For an O3/O7 hypersurface (lambda_f=1), the
    vertex part of source eq. (4.45) requires q.pairing+1 to be even for all
    dual vertices.  This is only an action-level diagnostic: fixed-locus
    smoothness and the paper's frozen-conifold cut are recorded separately.
    """

    actions = np.asarray(poly.inequivalent_Z2_actions(), dtype=int)
    dual_vertices = _as_int_rows(poly.dual().vertices())
    valid = [
        numerator
        for numerator in actions
        if np.all((dual_vertices @ numerator + 1) % 2 == 0)
    ]
    return actions, np.asarray(valid, dtype=int)


def _ambient_intersection_tensor(triangulation):
    """Return the toric fourfold intersection tensor.

    CYTools includes the canonical divisor as index zero and the nonzero fan
    rays as indices one onward.  The frozen-conifold formula below uses only
    the latter, so the convention is retained explicitly in the caller.
    """

    tensor = np.asarray(
        triangulation.fan().intersection_numbers(as_np_array=True), dtype=float
    )
    if tensor.ndim != 4 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"unexpected ambient intersection tensor shape {tensor.shape}")
    return tensor


def _frozen_conifold_diagnostic(triangulation, p0):
    """Evaluate the source fixed-surface conifold diagnostic for L=I.

    For L=I, fixed components are indexed by fan cones satisfying
    ``t + 1/2 sum(sigma(1)) in N``.  With t=p0/2 this is the parity condition
    p0+sum(sigma(1)) in 2N.  A two-dimensional fixed component is contained in
    the hypersurface for lambda_f=1.  For a smooth cone generated by p and q,
    eq. (4.49) reduces to

        int_F (K_V^{-1}-D_p)(K_V^{-1}-D_q).

    A positive value is a frozen-conifold obstruction for the paper's model
    population.  Non-smooth cones are reported as unavailable evidence rather
    than silently classified.
    """

    fan = triangulation.fan()
    vectors = _as_int_rows(fan.vectors())
    tensor = _ambient_intersection_tensor(triangulation)
    if tensor.shape[0] != vectors.shape[0] + 1:
        return {
            "status": "unavailable",
            "reason": "ambient intersection tensor does not match fan rays",
            "surface_records": [],
        }

    surface_records = []
    unavailable = []
    for cone in fan.cones(dim=2, formal=True):
        ray_points = _as_int_rows(cone.rays())
        if ray_points.shape[0] != 2:
            continue
        if not cone.is_smooth():
            unavailable.append(ray_points.tolist())
            continue
        if not np.all((np.asarray(p0, dtype=int) + ray_points.sum(axis=0)) % 2 == 0):
            continue
        ray_indices = []
        for point in ray_points:
            matches = np.flatnonzero(np.all(vectors == point, axis=1))
            if matches.size != 1:
                unavailable.append(ray_points.tolist())
                ray_indices = []
                break
            ray_indices.append(int(matches[0]) + 1)
        if not ray_indices:
            continue
        p_index, q_index = ray_indices
        n_rays = vectors.shape[0]
        l2 = 0.0
        l_dp = 0.0
        l_dq = 0.0
        for r in range(1, n_rays + 1):
            for s in range(1, n_rays + 1):
                l2 += tensor[p_index, q_index, r, s]
            l_dp += tensor[p_index, q_index, r, p_index]
            l_dq += tensor[p_index, q_index, r, q_index]
        dpdq = tensor[p_index, q_index, p_index, q_index]
        n_s = int(round(l2 - l_dp - l_dq + dpdq))
        surface_records.append(
            {
                "rays": ray_points.tolist(),
                "frozen_conifold_count": n_s,
                "formula": "int_F (K_V^-1-D_p)(K_V^-1-D_q)",
            }
        )

    frozen = [record for record in surface_records if record["frozen_conifold_count"] != 0]
    if unavailable:
        status = "unavailable"
    elif frozen:
        status = "frozen_conifold_obstruction"
    else:
        status = "no_frozen_conifold_obstruction"
    return {
        "status": status,
        "surface_records": surface_records,
        "unavailable_cones": unavailable,
        "frozen_surface_count": len(frozen),
    }


def _h11_minus_from_divisor_basis(poly, triangulation, matrix):
    """Derive the integral H2 involution and its odd dimension."""

    basis_matrix = np.asarray(triangulation.get_cy().divisor_basis(as_matrix=True), dtype=float)
    prime_labels = np.asarray(triangulation.get_cy().prime_toric_divisors(), dtype=int)
    points = _as_int_rows(poly.points())
    lookup = {
        tuple(point): int(label)
        for point, label in zip(points, poly.points_to_indices(points))
    }
    mapped = np.asarray(
        [lookup[tuple(np.asarray(matrix) @ point)] for point in points], dtype=int
    )
    divisor_points = np.concatenate(([0], prime_labels))
    mapped_divisors = mapped[divisor_points]
    positions = {int(label): position for position, label in enumerate(divisor_points)}
    try:
        mapped_positions = np.asarray([positions[int(label)] for label in mapped_divisors], dtype=int)
    except KeyError:
        return None
    permutation = np.zeros((divisor_points.size, divisor_points.size), dtype=float)
    permutation[np.arange(divisor_points.size), mapped_positions] = 1.0
    transformed_basis = basis_matrix @ permutation
    coefficients, _, _, _ = np.linalg.lstsq(basis_matrix.T, transformed_basis.T, rcond=None)
    h2 = np.rint(coefficients.T).astype(int)
    if not np.allclose(coefficients, h2, atol=1e-8):
        return None
    if not np.allclose(h2 @ basis_matrix, transformed_basis, atol=1e-8):
        return None
    if not np.array_equal(h2 @ h2, np.eye(h2.shape[0], dtype=int)):
        return None
    _, singular_values, _ = np.linalg.svd(h2.T + np.eye(h2.shape[0]))
    rank = int(np.count_nonzero(singular_values > 1e-10))
    return int(h2.shape[0] - rank)


def _class_invariant_under_matrix(poly, triangulation, matrix):
    """Test source FRST-class invariance under a lattice involution."""

    from cytools.triangulation import Triangulation

    points = _as_int_rows(triangulation.points())
    transformed_labels = poly.points_to_indices(
        [tuple(np.asarray(matrix, dtype=int) @ point) for point in points]
    )
    transformed = Triangulation(
        poly,
        transformed_labels,
        simplices=triangulation.simplices(as_indices=True),
        make_star=True,
        check_input_simplices=False,
    )
    return bool(triangulation.is_equivalent(transformed, on_faces_dim=2))


def _source_vertex_parity_allows(poly, matrix, torus_shift_numerator, lambda_f=1):
    """Apply the fixed-dual-vertex part of Moritz eq. (4.45)."""

    dual_vertices = _as_int_rows(poly.dual().vertices())
    dual_action = np.asarray(matrix, dtype=int).T
    fixed = [
        vertex
        for vertex in dual_vertices
        if np.array_equal(dual_action @ vertex, vertex)
    ]
    numerator = np.asarray(torus_shift_numerator, dtype=int)
    return bool(
        all((int(np.dot(numerator, vertex)) + int(lambda_f)) % 2 == 0 for vertex in fixed)
    )


def _orientifold_action_audit(poly, classes):
    """Count class-level affine O3/O7 candidates with source vertex evidence."""

    matrices = enumerate_polytope_involutions(poly.points())
    if not classes:
        return {"inherited": 0, "h11_minus_zero": 0, "h11_minus_zero_classes": []}
    h11_minus = {
        tuple(np.asarray(matrix, dtype=int).flatten()): _h11_minus_from_divisor_basis(
            poly, classes[0], matrix
        )
        for matrix in matrices
    }
    inherited_classes = set()
    h11_zero_classes = set()
    for matrix in matrices:
        key = tuple(np.asarray(matrix, dtype=int).flatten())
        odd_dimension = h11_minus[key]
        if odd_dimension is None:
            continue
        shifts = enumerate_projected_lattice_representatives(matrix, 1)
        valid_shift = any(
            _source_vertex_parity_allows(poly, matrix, shift["numerator"], lambda_f=1)
            for shift in shifts
        )
        if not valid_shift:
            continue
        for class_index, triangulation in enumerate(classes):
            if _class_invariant_under_matrix(poly, triangulation, matrix):
                inherited_classes.add(class_index)
                if odd_dimension == 0:
                    h11_zero_classes.add(class_index)
    return {
        "inherited": len(inherited_classes),
        "h11_minus_zero": len(h11_zero_classes),
        "h11_minus_zero_classes": sorted(h11_zero_classes),
    }


def _export_kaehler_point(triangulation):
    """Export the Algorithm-1 reference point t0 for one accepted FRST.

    ``t0`` is the tip of the stretched Kähler cone (arXiv:2412.12012 Sec.
    4.1, Algorithm 1), reusing ``sample_stretched_kaehler_points``'s
    ``attempt_index=1``/``canonical_tip`` convention (attempts=1, so only the
    tip itself is yielded; no qpsolvers/MOSEK angular sampling runs).
    ``kahler_cone.hyperplanes()`` is the *Mori*-cone dual (
    ``toric_kahler_cone() = toric_mori_cone(in_basis=True).dual()``), so
    ``tip_of_stretched_cone(1.0)`` bounds curve volumes, not divisor volumes:
    the paper's criterion 1 (tau_alpha >= 1 for all h11+4 prime toric
    divisors) is genuinely *not* guaranteed to hold at t0 itself -- confirmed
    empirically (divisor volumes ~0.5 at t0 on a live h11=4 sample, while the
    Kahler-cone slack sits at exactly 1.0 as expected). This matches the
    paper's own Algorithm 1, which starts at t0 and only requires criterion 1
    to hold after a homogeneous dilatation lambda*t0 (Sec. 3.3, quoted in the
    scope note). ``evaluate_kaehler_point`` is therefore called with
    ``enforce_divisor_volume_lower_bounds=False`` -- the same deferral the
    existing ``canonical_qcd`` moduli policy uses for its own post-dilation
    check (see generate_geometric_data_multitriangulation.py around line
    2207) -- so this export records t0's raw geometry data for the Phase 3
    lambda solver to evaluate criterion 1 against, rather than silently
    discarding every candidate whose *undilated* tip fails a check the paper
    never requires there.  Only the plain-data outputs needed by the Phase
    2/3 model-stage math are returned -- no live CYTools object is retained,
    so this record is safe to serialize and consume from Julia.

    ``glsm_charge_matrix`` is Q in the ``potential_matrix_convention``
    documented around generate_geometric_data_multitriangulation.py line
    ~3082 (h11 x N, instanton charges are columns); N here indexes all h11+4
    prime toric divisors, matching eq. 3.9's Q^i_alpha.
    """

    cy = triangulation.get_cy()
    topology = extract_topology(cy, triangulation)
    kahler_cone = cy.toric_kahler_cone()
    mosek_license = configure_mosek_license()
    tip_solver = "cytools-default"
    if mosek_license["activated"]:
        try:
            reference_tip = np.asarray(
                kahler_cone.tip_of_stretched_cone(1.0, backend="mosek"), dtype=float
            )
            tip_solver = "mosek"
        except Exception:
            reference_tip = np.asarray(
                kahler_cone.tip_of_stretched_cone(1.0), dtype=float
            )
            tip_solver = "cytools-default-after-mosek-failure"
    else:
        reference_tip = np.asarray(
            kahler_cone.tip_of_stretched_cone(1.0), dtype=float
        )

    qprime_raw = np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    qprime, _ = canonicalize_unique_charge_rows(qprime_raw)
    qprime = np.asarray(qprime, dtype=np.int64)

    glsm = np.asarray(cy.glsm_charge_matrix(include_origin=False), dtype=int)
    volume_context = {
        "volume_backend": "fan",
        "kappa": topology["kappa"],
        "glsm_charge_matrix": glsm,
        "mori_cone": topology["mori_cone"],
    }

    tip_proposal = next(
        sample_stretched_kaehler_points(
            kahler_cone,
            reference_tip,
            np.random.default_rng(),
            1,
            lambda message: None,
            point_seed=None,
            diagnostics=None,
            include_metadata=True,
        )
    )
    point = np.asarray(tip_proposal["point"], dtype=float)

    diagnostic, values = evaluate_kaehler_point(
        cy,
        kahler_cone,
        qprime,
        point,
        attempt_index=1,
        point_kind="canonical_tip",
        solver=tip_solver,
        min_prime_divisor_volume=1.0,
        min_divisor_volume=1.0,
        volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
        enforce_divisor_volume_lower_bounds=False,
        **volume_context,
    )
    if values is None:
        return {
            "status": "rejected",
            "reason": diagnostic.get(
                "failure_reason", "canonical tip failed the Kähler-point domain checks"
            ),
            "diagnostic": diagnostic,
        }
    return {
        "status": "accepted",
        "h11": int(topology["h11"]),
        "point": values["point"].tolist(),
        "cy_volume": float(values["cy_volume"]),
        "prime_divisor_volumes": values["prime_divisor_volumes"].tolist(),
        "inverse_metric": values["inverse_metric"].tolist(),
        "glsm_charge_matrix": glsm.tolist(),
        "potential_matrix_convention": {
            "Q": "h11 x N; instanton charges are columns; N indexes all h11+4 prime toric divisors",
        },
        "diagnostic": diagnostic,
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def reproduce(args):
    records = load_mirror_polytopes(
        args.parquet_dir,
        h11=4,
        limit=args.limit,
        favorable=True,
    )
    polytopes = []
    total_raw = 0
    total_classes = 0
    trilayer_polytope_count = 0
    trilayer_class_count = 0
    trilayer_nonfrozen_class_count = 0
    identity_action_cy_count = 0
    identity_valid_action_cy_count = 0
    identity_action_count = 0
    orientifold_inherited_count = 0
    orientifold_h11_zero_count = 0
    kaehler_export_accepted_count = 0
    kaehler_export_rejected_count = 0
    details = []
    for poly_index, (poly, provenance) in enumerate(records):
        raw, classes = _frst_classes(poly)
        total_raw += len(raw)
        total_classes += len(classes)
        trilayer = _trilayer_candidate(poly)
        actions, valid_actions = _identity_torus_actions(poly)
        identity_action_count += len(actions)
        if len(actions):
            identity_action_cy_count += len(classes)
        if len(valid_actions):
            identity_valid_action_cy_count += len(classes)
        frozen = None
        frozen_per_class = None
        nonfrozen_class_count_this_polytope = 0
        orientifold = None
        if trilayer is not None:
            trilayer_polytope_count += 1
            trilayer_class_count += len(classes)
            # The special trilayer involution (L=I) is compatible with every
            # FRST -- but the frozen-conifold diagnostic is evaluated on a
            # specific simplicial fan, and different FRST classes of the same
            # polytope can subdivide the fixed toric divisors differently.
            # Measured directly: the diagnostic's status varies across FRST
            # classes of the same polytope in ~22% of a sampled subset (see
            # validation/fuzzy_axions_2412_12012_frst_dependent_frozen_conifold_20260817.md).
            # It must therefore be evaluated per FRST class, not propagated
            # from a single representative.
            frozen_per_class = [
                _frozen_conifold_diagnostic(triangulation, trilayer["p0"])
                for triangulation in classes
            ]
            nonfrozen_class_count_this_polytope = sum(
                1
                for result in frozen_per_class
                if result["status"] == "no_frozen_conifold_obstruction"
            )
            trilayer_nonfrozen_class_count += nonfrozen_class_count_this_polytope
            frozen = frozen_per_class[0] if frozen_per_class else None
        if args.orientifold_audit:
            orientifold = _orientifold_action_audit(poly, classes)
            orientifold_inherited_count += orientifold["inherited"]
            orientifold_h11_zero_count += orientifold["h11_minus_zero"]
        kaehler_export_per_class = None
        if args.export_kaehler_points and frozen_per_class is not None:
            # Only the classes actually accepted by the h21_plus_zero
            # trilayer/frozen-conifold path (Algorithm 1's model-stage input
            # population, see validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md
            # section 6) get a live-CYTools export; the rest stay `None` so
            # index alignment with `frozen_conifold_per_class` is preserved.
            kaehler_export_per_class = []
            for class_index, triangulation in enumerate(classes):
                if frozen_per_class[class_index]["status"] != "no_frozen_conifold_obstruction":
                    kaehler_export_per_class.append(None)
                    continue
                record = _export_kaehler_point(triangulation)
                kaehler_export_per_class.append(record)
                if record["status"] == "accepted":
                    kaehler_export_accepted_count += 1
                else:
                    kaehler_export_rejected_count += 1
        details.append(
            {
                "polytope_index": poly_index,
                "provenance": provenance,
                "raw_frst_count": len(raw),
                "frst_class_count": len(classes),
                "trilayer": trilayer,
                "frozen_conifold": frozen,
                "frozen_conifold_per_class": frozen_per_class,
                "nonfrozen_class_count": nonfrozen_class_count_this_polytope,
                "identity_torus_action_numerators": actions.tolist(),
                "identity_valid_o3o7_action_numerators": valid_actions.tolist(),
                "orientifold_action_audit": orientifold,
                "kaehler_point_export_per_class": kaehler_export_per_class,
            }
        )
        if args.progress and (poly_index + 1) % args.progress == 0:
            print(
                f"processed {poly_index + 1}/{len(records)} polytopes; "
                f"raw={total_raw}, classes={total_classes}, "
                f"trilayer_classes={trilayer_class_count}, "
                f"nonfrozen_trilayer_classes={trilayer_nonfrozen_class_count}",
                flush=True,
            )

    summary = {
        "schema_version": "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.0",
        "paper": "arXiv:2412.12012",
        "orientifold_source": "arXiv:2305.06363",
        "input": {
            "source": "generator.load_mirror_polytopes",
            "parquet_dir": str(Path(args.parquet_dir).resolve()),
            "requested_h11": 4,
            "favorable_lattice": "N",
            "record_count": len(records),
            "population_complete": bool(args.limit >= 10**9),
        },
        "counts": {
            "favorable_polytopes": len(records),
            "raw_frsts": total_raw,
            "frst_classes": total_classes,
            "raw_trilayer_polytopes": trilayer_polytope_count,
            "raw_trilayer_frst_classes": trilayer_class_count,
            "nonfrozen_trilayer_frst_classes": trilayer_nonfrozen_class_count,
            "identity_torus_action_count": identity_action_count,
            "identity_torus_action_cy_count": identity_action_cy_count,
            "identity_valid_o3o7_action_cy_count": identity_valid_action_cy_count,
            "source_vertex_evidence_inherited_orientifold_cys": orientifold_inherited_count
            if args.orientifold_audit
            else None,
            "source_vertex_evidence_h11_minus_zero_orientifold_cys": orientifold_h11_zero_count
            if args.orientifold_audit
            else None,
            "kaehler_point_export_accepted_count": kaehler_export_accepted_count
            if args.export_kaehler_points
            else None,
            "kaehler_point_export_rejected_count": kaehler_export_rejected_count
            if args.export_kaehler_points
            else None,
        },
        "paper_targets": PAPER_TARGETS,
        "claim_status": {
            "favorable_polytopes": "exact" if len(records) == PAPER_TARGETS["favorable_polytopes"] else "mismatch",
            "frst_classes": "exact" if total_classes == PAPER_TARGETS["frst_classes"] else "mismatch",
            "h21_plus_zero": (
                "benchmark_match_candidate"
                if trilayer_nonfrozen_class_count == PAPER_TARGETS["h11_minus_zero_h21_plus_zero_orientifold_cys"]
                else "diagnostic_only"
            ),
        },
        "details": details if args.keep_details else None,
    }
    return _jsonable(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--limit", type=int, default=10**9)
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument("--keep-details", action="store_true")
    parser.add_argument("--orientifold-audit", action="store_true")
    parser.add_argument(
        "--export-kaehler-points",
        action="store_true",
        help=(
            "Export the Algorithm-1 canonical-tip Kahler point (cy_volume, "
            "prime_divisor_volumes, inverse_metric, GLSM charge matrix Q) for "
            "every h21_plus_zero-accepted FRST class. Requires --keep-details "
            "to appear in the output."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reproduce(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
