#!/usr/bin/env python3
"""Reproduce the h11=4 population benchmarks of arXiv:2412.12012.

This is a source-matched audit driver, not a replacement for the production
geometry generator.  It reads the same KS Parquet mirror as the generator,
counts FRST classes using the paper's two-face equivalence relation, and
records the special trilayer involution together with two independent
per-FRST-class diagnostics: the frozen-conifold smoothness check (Moritz
eqs. 4.48-4.50, a separate orientifold-background smoothness condition)
and the h^{2,1}_+(X,I)=0 Hodge-number identity (eq. 4.51), the latter
being the actual gate for the paper's h11_minus_zero_h21_plus_zero
population.

The orientifold and model stages are intentionally represented as explicit
diagnostic records.  A count is labelled ``exact`` only when the implementation
has the corresponding source criterion and complete input evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import h5py
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


def _fixed_locus_components(triangulation, p0):
    """Enumerate the irreducible fixed-locus components F_I(sigma), L=I.

    Per Moritz eqs. (4.33)-(4.35), with L=I the nu label is trivially 0 and
    every cone is pointwise L-invariant, so the smooth-cone components of
    the fixed point set are labelled by cones sigma (of any dimension 0-4)
    satisfying the parity condition t + (1/2) sum(sigma(1)) in N, i.e. with
    t=p0/2, p0 + sum(rays in sigma(1)) in 2N.  Per the reduction described
    just before eq. (4.35) ("one then removes the F~(sigma,nu) that are
    already contained in a higher dimensional component"), a candidate
    cone is discarded whenever a proper face of it also satisfies the
    parity condition, since V(sigma) is then already contained in that
    smaller face's own (larger) stratum.
    """

    fan = triangulation.fan()
    vectors = _as_int_rows(fan.vectors())
    p0 = np.asarray(p0, dtype=int)

    def parity_ok(idxs):
        total = p0 + (vectors[list(idxs)].sum(axis=0) if idxs else 0)
        return bool(np.all(total % 2 == 0))

    def indices_of(cone):
        ray_points = _as_int_rows(cone.rays())
        idxs = []
        for point in ray_points:
            matches = np.flatnonzero(np.all(vectors == point, axis=1))
            if matches.size != 1:
                return None
            idxs.append(int(matches[0]))
        return tuple(sorted(idxs))

    candidates = []
    unavailable = False
    reasons = []
    if parity_ok(()):
        candidates.append((0, ()))
    for k in (1, 2, 3, 4):
        for cone in fan.cones(dim=k, formal=True):
            idxs = indices_of(cone)
            if idxs is None:
                unavailable = True
                reasons.append(f"ray_lookup_failed dim={k}")
                continue
            if not parity_ok(idxs):
                continue
            if k % 2 == 0 and not cone.is_smooth():
                # Eq. (4.46): f vanishes identically on F~(sigma) for even
                # dim(sigma), so all of F~(sigma) becomes part of F_I; the
                # smoothness discussion of Sec. 4.6 requires this toric
                # stratum itself to be smooth for its contribution to
                # chi(F_I) to be well defined.
                unavailable = True
                reasons.append(f"non_smooth_even_dim_component dim={k} rays={idxs}")
                continue
            # For odd dim(sigma), f is a generic section (eq. 4.46) and the
            # paper's own requirement is weaker than cone smoothness ("no
            # orbifold singularities of F~(sigma,nu) intersect the
            # hypersurface", Sec. 4.6) -- independently verified not to
            # change the population count (see the fixed-locus validation
            # note), so a non-smooth odd-dimension cone is kept, not
            # excluded.
            candidates.append((k, idxs))

    sets = [set(idxs) for _, idxs in candidates]
    minimal = [
        (k, idxs)
        for i, (k, idxs) in enumerate(candidates)
        if not any(j != i and sets[j] < sets[i] for j in range(len(candidates)))
    ]
    return minimal, unavailable, reasons


def _fixed_locus_euler_characteristic(poly, triangulation, p0):
    """Compute chi(F_I) for the L=I trilayer fixed locus.

    Dispatches each irreducible component F_I(sigma) from
    ``_fixed_locus_components`` by dim(sigma) parity (Moritz eq. 4.46):

    - even dim(sigma): F_I(sigma)=F~(sigma) in full, a smooth complete
      toric variety whose Euler characteristic equals its number of
      maximal cones (each toric torus-fixed point contributes 1).
    - dim(sigma)=1 (a ray p, F~(sigma)=D_p): F_I(sigma)=D_p . X, a
      generic-hypersurface-section surface in the Calabi-Yau threefold X.
      By adjunction (K_X=0), chi(D_p.X) = D_p^3 + c2(X).D_p -- verified
      to agree with an independent ambient toric-Chern-class derivation
      to numerical precision on real h11=4 examples, see
      validation/fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md.
    - dim(sigma)=3 (rays p,q,r spanning a curve V(sigma)): F_I(sigma) is
      the point set cut by X on that curve, whose count is the ambient
      quadruple intersection D_p.D_q.D_r.X = sum_s kappa[p,q,r,s].
    """

    components, unavailable, reasons = _fixed_locus_components(triangulation, p0)
    if unavailable:
        return {"chi_F_I": None, "status": "unavailable", "reasons": reasons, "components": []}

    fan = triangulation.fan()
    vectors = _as_int_rows(fan.vectors())
    maximal_cones = [set(cone) for cone in fan.cones(as_inds=True)]
    tensor = _ambient_intersection_tensor(triangulation)
    kappa = tensor[1:, 1:, 1:, 1:]

    cy = triangulation.get_cy()
    prime_divs = set(cy.prime_toric_divisors())
    inter_dense = None
    c2_vec = None

    chi_total = 0.0
    detail = []
    for k, idxs in components:
        if k % 2 == 0:
            chi = float(sum(1 for cone in maximal_cones if set(idxs) <= cone))
            detail.append({"dim": k, "rays": idxs, "chi": chi, "kind": "toric_stratum"})
        elif k == 1:
            ray_index = idxs[0]
            point_index = int(poly.points_to_indices(vectors[ray_index].reshape(1, -1))[0])
            if point_index not in prime_divs:
                return {
                    "chi_F_I": None,
                    "status": "unavailable",
                    "reasons": reasons + [f"ray {ray_index} is not a prime toric divisor of X"],
                    "components": detail,
                }
            if inter_dense is None:
                inter_dense = cy.intersection_numbers(in_basis=False, format="dense")
                c2_vec = cy.second_chern_class(in_basis=False, include_origin=True)
            chi = float(
                inter_dense[point_index, point_index, point_index] + c2_vec[point_index]
            )
            detail.append({"dim": k, "rays": idxs, "chi": chi, "kind": "divisor_surface"})
        else:
            p_index, q_index, r_index = idxs
            chi = float(kappa[p_index, q_index, r_index, :].sum())
            detail.append({"dim": k, "rays": idxs, "chi": chi, "kind": "curve_points"})
        chi_total += chi

    return {"chi_F_I": chi_total, "status": "computed", "reasons": reasons, "components": detail}


def _h21_plus_zero_diagnostic(poly, triangulation, p0):
    """Test whether h^{2,1}_+(X,I)=0 exactly, via Moritz eq. (4.51).

    For L=I, h^{1,1}_-(X,I)=0 identically (the identity lattice action
    fixes every toric divisor class), so eq. (4.51) reduces to
    h^{2,1}_-(X,I) = (chi(F_I) - chi(X))/4 - 1, and h^{2,1}_+ = h^{2,1}(X)
    - h^{2,1}_-(X,I).  Independently validated against the paper's own
    h11=2 worked example (Sec. 4.2.1, eq. 4.2): reproduces the stated
    (h^{1,1}_+,h^{1,1}_-,h^{2,1}_+,h^{2,1}_-) = (2,0,0,132) exactly, and
    reproduces the paper's 267-class h11=4 population target exactly when
    applied across all trilayer FRST classes.  See
    validation/fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md.
    """

    result = _fixed_locus_euler_characteristic(poly, triangulation, p0)
    if result["status"] != "computed":
        return {
            "status": "unavailable",
            "reasons": result["reasons"],
            "chi_F_I": None,
            "h21_minus": None,
            "h21_plus": None,
            "components": result["components"],
        }
    cy = triangulation.get_cy()
    chi_X = cy.chi()
    h21_X = cy.h21()
    chi_FI = result["chi_F_I"]
    h21_minus = (chi_FI - chi_X) / 4.0 - 1.0
    h21_plus = h21_X - h21_minus
    is_zero = abs(h21_plus) < 1e-6
    return {
        "status": "h21_plus_zero" if is_zero else "h21_plus_nonzero",
        "reasons": result["reasons"],
        "chi_F_I": chi_FI,
        "h21_minus": h21_minus,
        "h21_plus": h21_plus,
        "components": result["components"],
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


MODEL_STAGE_DRIVER = Path(__file__).resolve().parent / "fuzzy_axion_model_stage_driver.jl"


def _write_model_stage_input(records: list[dict], path: Path) -> None:
    """Write flat (X, FRST) Kähler-point export records to HDF5 for the Julia driver.

    HDF5 is used rather than JSON because it is already a hard dependency on
    both sides of this package (``h5py`` here, ``HDF5.jl`` in
    ``Project.toml``); ``CYAxiverse`` has no JSON-parsing package, and this
    keeps priority 4 from having to add one. ``Q``/``inverse_metric`` are
    written in their Python-documented shapes (h11 x N, h11 x h11); the Julia
    driver is responsible for undoing HDF5.jl's dimension-order reversal on
    read (see its module docstring -- confirmed empirically, not assumed,
    before writing that script).
    """

    with h5py.File(path, "w") as file:
        file.create_dataset("record_count", data=len(records))
        for index, record in enumerate(records):
            group = file.create_group(f"records/{index}")
            group.create_dataset("Q", data=np.array(record["glsm_charge_matrix"], dtype=np.int64))
            group.create_dataset("tau", data=np.array(record["prime_divisor_volumes"], dtype=np.float64))
            group.create_dataset("cy_volume", data=float(record["cy_volume"]))
            group.create_dataset(
                "inverse_metric", data=np.array(record["inverse_metric"], dtype=np.float64)
            )


def _run_model_stage(records: list[dict], args) -> dict[str, Any]:
    """Run priority 3's Julia model-stage evaluator over the exported Kähler points.

    ``gs``/``W0`` are a single documented convention applied uniformly to
    every record -- never tuned per record or towards the 3,348 target (see
    validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md
    Sec. 6, "Acceptance tests for the 3,348 comparison").
    """

    convention = {
        "gs": args.gs,
        "gs_justification": (
            "paper's stated main-analysis reference value, eq. 3.28-3.29 (gs=0.5 -> P~5e-4)"
        ),
        "w0_real": args.w0_real,
        "w0_imag": args.w0_imag,
        "w0_justification": (
            "no ensemble-specific W0 value is stated in the source (Sec. 3.4 gives only "
            "the reheating-example illustrative value W0=1e-5, and argues -- eq. 3.26-3.27 "
            "-- that the fuzzy axion's own decay constant is near-insensitive to W0 once its "
            "mass is pinned by construction, though this does not bound the model *count*, "
            "see the scope doc Sec. 3 point 2); defaults to W0=1, matching the paper's own "
            "Sec. 4.2.1 hand-worked-example convention, and is not tuned to approach the "
            "3,348 target"
        ),
    }

    if not records:
        return {
            "input_record_count": 0,
            "total_model_count": None,
            "model_count_per_record": [],
            "models": [] if args.keep_details else None,
            "convention": convention,
            "diagnostic_reason": (
                "no accepted h21_plus_zero-class Kahler-point export records in this run"
            ),
        }

    with tempfile.TemporaryDirectory(prefix="fuzzy_axion_model_stage_") as workdir_name:
        workdir = Path(workdir_name)
        input_path = workdir / "model_stage_input.h5"
        output_path = workdir / "model_stage_output.h5"
        _write_model_stage_input(records, input_path)

        julia_project = args.julia_project or Path(__file__).resolve().parent.parent
        subprocess.run(
            [
                args.julia_binary,
                f"--project={julia_project}",
                str(MODEL_STAGE_DRIVER),
                str(input_path),
                str(output_path),
                str(args.gs),
                str(args.w0_real),
                str(args.w0_imag),
            ],
            check=True,
        )

        with h5py.File(output_path, "r") as file:
            total_model_count = int(file["total_model_count"][()])
            model_count_per_record = np.asarray(file["model_count_per_record"]).tolist()
            prefactor_P = float(file["prefactor_P"][()])
            models = None
            if args.keep_details:
                record_index = np.asarray(file["model_record_index"])
                axion_index = np.asarray(file["model_axion_index"])
                qcd_divisor_index = np.asarray(file["model_qcd_divisor_index"])
                lam = np.asarray(file["model_lambda"])
                mass_reference_log10_ev = np.asarray(file["model_mass_reference_log10_ev"])
                tau_reference = np.asarray(file["model_tau_reference"])
                models = [
                    {
                        "record_index": int(record_index[i]),
                        "axion_index": int(axion_index[i]),
                        "qcd_divisor_index": int(qcd_divisor_index[i]),
                        "lambda": float(lam[i]),
                        "mass_reference_log10_ev": float(mass_reference_log10_ev[i]),
                        "tau_reference": float(tau_reference[i]),
                    }
                    for i in range(len(record_index))
                ]

    convention["prefactor_P"] = prefactor_P
    return {
        "input_record_count": len(records),
        "total_model_count": total_model_count,
        "model_count_per_record": model_count_per_record,
        "models": models,
        "convention": convention,
        "diagnostic_reason": None,
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
    trilayer_h21_plus_zero_class_count = 0
    identity_action_cy_count = 0
    identity_valid_action_cy_count = 0
    identity_action_count = 0
    orientifold_inherited_count = 0
    orientifold_h11_zero_count = 0
    kaehler_export_accepted_count = 0
    kaehler_export_rejected_count = 0
    export_kaehler_points = args.export_kaehler_points or args.model_stage
    model_stage_records = [] if args.model_stage else None
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
        h21_per_class = None
        h21_plus_zero_class_count_this_polytope = 0
        orientifold = None
        if trilayer is not None:
            trilayer_polytope_count += 1
            trilayer_class_count += len(classes)
            # The special trilayer involution (L=I) is compatible with every
            # FRST -- but both per-class diagnostics below are evaluated on a
            # specific simplicial fan, and different FRST classes of the same
            # polytope can subdivide the fixed toric divisors differently.
            # Measured directly: the frozen-conifold diagnostic's status
            # varies across FRST classes of the same polytope in ~22% of a
            # sampled subset (see
            # validation/fuzzy_axions_2412_12012_frst_dependent_frozen_conifold_20260817.md).
            # Both must therefore be evaluated per FRST class, not propagated
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
            # The actual h11_minus_zero_h21_plus_zero population gate (eq.
            # 4.51 applied directly): independently validated to reproduce
            # the paper's own h11=2 worked example's Hodge splitting exactly
            # and the h11=4 population's 267-class target exactly -- unlike
            # `frozen_per_class` above, which is a separate smoothness
            # diagnostic for the orientifold background, not part of this
            # population's definition. See
            # validation/fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md.
            h21_per_class = [
                _h21_plus_zero_diagnostic(poly, triangulation, trilayer["p0"])
                for triangulation in classes
            ]
            h21_plus_zero_class_count_this_polytope = sum(
                1 for result in h21_per_class if result["status"] == "h21_plus_zero"
            )
            trilayer_h21_plus_zero_class_count += h21_plus_zero_class_count_this_polytope
        if args.orientifold_audit:
            orientifold = _orientifold_action_audit(poly, classes)
            orientifold_inherited_count += orientifold["inherited"]
            orientifold_h11_zero_count += orientifold["h11_minus_zero"]
        kaehler_export_per_class = None
        if export_kaehler_points and h21_per_class is not None:
            # Only the classes actually accepted by the h21_plus_zero
            # population gate (Algorithm 1's model-stage input population,
            # see validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md
            # section 6) get a live-CYTools export; the rest stay `None` so
            # index alignment with `h21_per_class` is preserved.
            kaehler_export_per_class = []
            for class_index, triangulation in enumerate(classes):
                if h21_per_class[class_index]["status"] != "h21_plus_zero":
                    kaehler_export_per_class.append(None)
                    continue
                record = _export_kaehler_point(triangulation)
                kaehler_export_per_class.append(record)
                if record["status"] == "accepted":
                    kaehler_export_accepted_count += 1
                    if args.model_stage:
                        model_stage_records.append(
                            {
                                "polytope_index": poly_index,
                                "frst_class_index": class_index,
                                "glsm_charge_matrix": record["glsm_charge_matrix"],
                                "prime_divisor_volumes": record["prime_divisor_volumes"],
                                "cy_volume": record["cy_volume"],
                                "inverse_metric": record["inverse_metric"],
                            }
                        )
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
                "h21_plus_zero_per_class": h21_per_class,
                "h21_plus_zero_class_count": h21_plus_zero_class_count_this_polytope,
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
                f"nonfrozen_trilayer_classes={trilayer_nonfrozen_class_count}, "
                f"h21_plus_zero_trilayer_classes={trilayer_h21_plus_zero_class_count}",
                flush=True,
            )

    population_complete = bool(args.limit >= 10**9)
    population_exact_267 = (
        trilayer_h21_plus_zero_class_count == PAPER_TARGETS["h11_minus_zero_h21_plus_zero_orientifold_cys"]
        and population_complete
    )
    model_stage = _run_model_stage(model_stage_records, args) if args.model_stage else None
    if model_stage is not None and model_stage["diagnostic_reason"] is None:
        reasons = []
        if not population_complete:
            reasons.append("run was limited via --limit; population is not the full h11=4 set")
        if trilayer_h21_plus_zero_class_count != PAPER_TARGETS["h11_minus_zero_h21_plus_zero_orientifold_cys"]:
            reasons.append(
                f"input population is {trilayer_h21_plus_zero_class_count} h21_plus_zero-accepted "
                f"FRST classes, not the exact {PAPER_TARGETS['h11_minus_zero_h21_plus_zero_orientifold_cys']}"
                "-target population (see "
                "fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md)"
            )
        if model_stage["total_model_count"] != PAPER_TARGETS["models"]:
            reasons.append(
                f"total model count {model_stage['total_model_count']} != paper target "
                f"{PAPER_TARGETS['models']}"
            )
        if reasons:
            model_stage["diagnostic_reason"] = "; ".join(reasons)

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
            "population_complete": population_complete,
        },
        "counts": {
            "favorable_polytopes": len(records),
            "raw_frsts": total_raw,
            "frst_classes": total_classes,
            "raw_trilayer_polytopes": trilayer_polytope_count,
            "raw_trilayer_frst_classes": trilayer_class_count,
            # A separate orientifold-background smoothness diagnostic, not
            # this population's own gate -- see h21_plus_zero_trilayer_frst_classes.
            "nonfrozen_trilayer_frst_classes": trilayer_nonfrozen_class_count,
            "h21_plus_zero_trilayer_frst_classes": trilayer_h21_plus_zero_class_count,
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
            if export_kaehler_points
            else None,
            "kaehler_point_export_rejected_count": kaehler_export_rejected_count
            if export_kaehler_points
            else None,
        },
        "paper_targets": PAPER_TARGETS,
        "claim_status": {
            "favorable_polytopes": "exact" if len(records) == PAPER_TARGETS["favorable_polytopes"] else "mismatch",
            "frst_classes": "exact" if total_classes == PAPER_TARGETS["frst_classes"] else "mismatch",
            "h21_plus_zero": (
                "benchmark_match_candidate"
                if trilayer_h21_plus_zero_class_count == PAPER_TARGETS["h11_minus_zero_h21_plus_zero_orientifold_cys"]
                else "diagnostic_only"
            ),
            "model_count": (
                None
                if model_stage is None
                else (
                    "benchmark_match_candidate"
                    if population_exact_267 and model_stage["total_model_count"] == PAPER_TARGETS["models"]
                    else "diagnostic_only"
                )
            ),
        },
        "model_stage": model_stage,
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
    parser.add_argument(
        "--model-stage",
        action="store_true",
        help=(
            "Run priority 4: per h21_plus_zero-accepted FRST class, export the "
            "Algorithm-1 canonical-tip Kahler point (implies --export-kaehler-points) "
            "and enumerate (QCD divisor, fuzzy axion) models via "
            "CYAxiverse.paper_benchmarks.enumerate_fuzzy_axion_models (bridged through "
            "Julia via HDF5), comparing the total model count against the paper's "
            "target of 3,348 under the acceptance-test discipline in "
            "validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md "
            "Sec. 6."
        ),
    )
    parser.add_argument(
        "--gs",
        type=float,
        default=0.5,
        help="String coupling used for the prefactor P = gs^4/128 (eq. 3.28-3.29); "
        "the paper's stated main-analysis value is 0.5.",
    )
    parser.add_argument(
        "--w0-real",
        type=float,
        default=1.0,
        help="Real part of the flux superpotential W0 (eq. 3.12); no ensemble-specific "
        "value is given in the source, so this defaults to the paper's own Sec. 4.2.1 "
        "hand-worked-example convention W0=1.",
    )
    parser.add_argument("--w0-imag", type=float, default=0.0)
    parser.add_argument(
        "--julia-binary",
        default="julia",
        help="Julia executable used to run scripts/fuzzy_axion_model_stage_driver.jl.",
    )
    parser.add_argument(
        "--julia-project",
        type=Path,
        default=None,
        help="Julia --project path for the model-stage driver; defaults to this "
        "repository's root (the parent of scripts/).",
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
