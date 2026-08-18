"""Enumerate inherited Calabi--Yau orientifold candidates.

The candidate datum is the source-paper triple ``(L, t, lambda_f)``.  The
existing ``validate_orientifold`` function remains the gate for polytope,
FRST, prime-divisor, and integral H2 preservation.  This module adds the
finite projected-lattice searches, the auxiliary fan from source eq. (4.26),
fixed-component integrality from eq. (4.35), the coefficient parity datum,
and conservative smoothness classification from Sec. 4.6.

The implementation deliberately reports ``smoothness_verification_unavailable``
when the supplied topology does not contain the evidence needed for a
sufficient claim.  Only ``accepted_verified_orientifold`` records carry a
verified acceptance status.  The identity ``L=I, t=0`` is retained as the
explicit task-level sanity contract for both parity choices; those records are
labelled ``identity_sanity_contract`` rather than being presented as a
generic nontrivial hypersurface proof.
"""

from fractions import Fraction
import itertools
import json
import math
import os
import time

import numpy as np
from sympy import Matrix as _SympyMatrix
from sympy.matrices.normalforms import hermite_normal_form as _hermite_normal_form

from generate_geometric_data_multitriangulation import (
    OrientifoldValidationFailure,
    validate_orientifold,
)
from glimmers_raw_frst import compute_polytope_id, compute_triangulation_hash, stable_hash

CANDIDATE_SCHEMA_VERSION = "cyaxiverse-inherited-orientifold-candidate-2.0"

TERMINAL_STATUSES = (
    "polytope_not_preserved",
    "frst_not_preserved",
    "prime_divisor_set_not_preserved",
    "nonintegral_h2_action",
    "h2_action_not_involution",
    "orientifold_h11_minus_filter_rejection",
    "torus_shift_search_exhausted",
    "fixed_point_set_non_smooth",
    "smoothness_verification_unavailable",
    "accepted_verified_orientifold",
    # Kept in the vocabulary for consumers of the earlier matrix-only schema.
    "accepted_inherited_candidate",
)

IDENTITY = np.eye(4, dtype=int)


def _select_basis_indices(points):
    """Return 4 indices into ``points`` spanning the rank-four lattice."""
    basis_rows = []
    chosen = []
    for index, point in enumerate(points):
        candidate_rows = basis_rows + [point]
        if np.linalg.matrix_rank(np.asarray(candidate_rows, dtype=float)) == len(
            candidate_rows
        ):
            basis_rows = candidate_rows
            chosen.append(index)
        if len(chosen) == 4:
            return chosen
    raise ValueError("points do not span a rank-4 lattice; cannot select a basis")


def enumerate_polytope_involutions(points):
    """Enumerate deterministic integer involutions preserving a 4d point set."""
    points = np.asarray(points, dtype=int)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must be an (n, 4) integer array")

    point_set = {tuple(int(value) for value in row) for row in points}
    basis_indices = _select_basis_indices(points)
    basis_points = points[basis_indices]
    basis_matrix = basis_points.T.astype(float)
    basis_inverse = np.linalg.inv(basis_matrix)

    involutions = {tuple(IDENTITY.flatten().tolist()): IDENTITY}
    candidate_points = sorted(point_set)
    for image_tuple in itertools.product(candidate_points, repeat=4):
        image_matrix = np.asarray(image_tuple, dtype=float).T
        if abs(np.linalg.det(image_matrix)) < 1e-9:
            continue
        candidate = image_matrix @ basis_inverse
        rounded = np.rint(candidate)
        if not np.allclose(candidate, rounded, atol=1e-6):
            continue
        matrix = rounded.astype(int)
        determinant = round(float(np.linalg.det(matrix)))
        if determinant not in (-1, 1):
            continue
        if not np.array_equal(matrix @ matrix, IDENTITY):
            continue
        if not all(tuple((matrix @ point).tolist()) in point_set for point in points):
            continue
        involutions[tuple(matrix.flatten().tolist())] = matrix

    return [involutions[key] for key in sorted(involutions)]


def _fraction_vector(vector):
    return tuple(Fraction(value) for value in vector)


def _fraction_vector_to_json(vector):
    """Encode a rational vector without introducing floating-point ambiguity."""
    vector = _fraction_vector(vector)
    denominator = 1
    for value in vector:
        denominator = math.lcm(denominator, value.denominator)
    numerator = [int(value * denominator) for value in vector]
    divisor = abs(math.gcd(*numerator, denominator))
    if divisor > 1:
        numerator = [value // divisor for value in numerator]
        denominator //= divisor
    return {"numerator": numerator, "denominator": int(denominator)}


def _fraction_sum(vectors):
    result = [Fraction(0) for _ in range(4)]
    for vector in vectors:
        for index, value in enumerate(vector):
            result[index] += Fraction(value)
    return tuple(result)


def _is_integral(vector):
    return all(Fraction(value).denominator == 1 for value in vector)


def _integer_vector(vector):
    return [int(Fraction(value)) for value in vector]


def _integer_lattice_membership(generator_columns, target):
    """Test whether ``target`` lies in the Z-span of ``generator_columns``.

    Tested exactly via Hermite normal form: ``target`` is an integer
    combination of the columns of ``generator_columns`` iff appending it as
    an extra column does not change the generated lattice, i.e. iff
    ``HNF(generator_columns) == HNF([generator_columns | target])``. This
    replaces an earlier bounded brute-force search over candidate
    coefficients in ``{-1, 0, 1}^4``, which was *not* complete in general: it
    silently undercounted merges (and so overcounted representative classes)
    whenever the relevant sublattice's generators were not aligned with the
    standard basis -- i.e. for any involution reached by conjugating a
    diagonal +/-1 matrix by a non-permutation unimodular matrix, not just
    contrived adversarial input. A stress test against 300 random integer
    involutions found this happening in 47% of cases (see
    ``validation/fuzzy_axions_2412_12012_torus_shift_audit_20260817.md``).
    sympy is already a transitive CYTools dependency (this module cannot run
    without CYTools objects regardless), so using it for exact Hermite
    normal form arithmetic adds no new environment requirement.
    """
    generator_matrix = _SympyMatrix(np.asarray(generator_columns, dtype=int).tolist())
    target_vector = _SympyMatrix(np.asarray(target, dtype=int).reshape(-1, 1).tolist())
    base_hnf = _hermite_normal_form(generator_matrix)
    augmented_hnf = _hermite_normal_form(generator_matrix.row_join(target_vector))
    return base_hnf == augmented_hnf


def _same_projected_class(left, right, projector_numerator):
    """Test equality modulo twice the projected lattice, exactly.

    The representatives are ``t = P z`` with ``P = projector_numerator / 2``
    and binary ``z``. Two representatives ``t1, t2`` (both already known to
    lie in ``P(N)``, the column span of ``projector_numerator``) are
    equivalent iff ``t1 - t2`` lies in ``2 P(N)``, the column span of
    ``2 * projector_numerator``. Since ``left``/``right`` here are the
    doubled representatives (``2*t1``, ``2*t2``, i.e. ``projector_numerator
    @ z``), the membership test is against ``2 * projector_numerator``.
    """
    difference = np.asarray(left, dtype=int) - np.asarray(right, dtype=int)
    generator_columns = 2 * np.asarray(projector_numerator, dtype=int)
    return _integer_lattice_membership(generator_columns, difference)


def enumerate_projected_lattice_representatives(matrix, sign):
    """Enumerate representatives of ``P_sign(N)/(2 P_sign(N))``.

    ``sign=+1`` gives the torus shifts ``t`` and ``sign=-1`` gives the fixed
    component labels ``nu``.  Each return value contains the exact rational
    vector and its integer numerator ``2 vector``.
    """
    matrix = np.asarray(matrix, dtype=int)
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    projector_numerator = IDENTITY + sign * matrix
    representatives = []
    numerators = []
    for bits in itertools.product((0, 1), repeat=4):
        numerator = projector_numerator @ np.asarray(bits, dtype=int)
        if any(
            _same_projected_class(numerator, previous, projector_numerator)
            for previous in numerators
        ):
            continue
        numerators.append(numerator)
        representatives.append(
            {
                "numerator": [int(value) for value in numerator],
                "vector": tuple(Fraction(int(value), 2) for value in numerator),
                "binary_source": list(bits),
            }
        )
    return sorted(representatives, key=lambda item: tuple(item["vector"]))


def _lattice_matrix_config(matrix, involution_type=None):
    """Build a ``load_orientifold``-shaped config for one candidate matrix."""
    return {
        "requested": True,
        "status": "input_loaded",
        "source_file": None,
        "lattice_matrix": np.asarray(matrix, dtype=int),
        "involution_type": involution_type,
        "coefficient_constraints": {},
        "label": None,
    }


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Fraction):
        return _fraction_vector_to_json((value,))
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _point_tuple(point):
    return tuple(int(value) for value in np.asarray(point, dtype=int))


def _triangulation_cones(poly, triangulation):
    """Return unique ambient fan cones as tuples of nonzero lattice rays."""
    poly_points = np.asarray(poly.points(), dtype=int)
    point_lookup = {_point_tuple(point): point for point in poly_points}
    local_points = np.asarray(triangulation.points(), dtype=int)
    local_to_global = [point_lookup[_point_tuple(point)] for point in local_points]
    raw_simplices = np.asarray(triangulation.simplices(as_indices=True), dtype=int)
    cones = set()
    for simplex in raw_simplices:
        rays = {
            _point_tuple(local_to_global[int(index)])
            for index in simplex
            if _point_tuple(local_to_global[int(index)]) != (0, 0, 0, 0)
        }
        if rays:
            cones.add(tuple(sorted(rays)))
    return sorted(cones)


def _primitive_integer_from_float(vector):
    """Convert a numerical ray direction to a primitive integer vector."""
    vector = np.asarray(vector, dtype=float)
    nonzero = np.flatnonzero(np.abs(vector) > 1e-9)
    if nonzero.size == 0:
        raise ValueError("zero vector is not a ray")
    first = int(nonzero[0])
    scale = vector[first]
    rational = [
        Fraction(float(value / scale)).limit_denominator(10000) for value in vector
    ]
    denominator = 1
    for value in rational:
        denominator = math.lcm(denominator, value.denominator)
    integer = np.asarray([int(value * denominator) for value in rational], dtype=int)
    divisor = abs(math.gcd(*integer.tolist()))
    if divisor > 1:
        integer //= divisor
    if np.sign(integer[first]) != np.sign(scale):
        integer *= -1
    return tuple(int(value) for value in integer)


def _nullspace(matrix, tolerance=1e-10):
    matrix = np.asarray(matrix, dtype=float)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    scale = max(float(np.max(singular_values, initial=0.0)), 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    return vh[rank:].T


def _intersection_rays(cone, matrix):
    """Find extreme rays of a simplicial cone intersected with ``Fix(matrix)``."""
    rays = np.asarray(cone, dtype=float).T
    constraint = np.eye(4, dtype=float) - np.asarray(matrix, dtype=float)
    generators = set()
    for size in range(1, len(cone) + 1):
        for support in itertools.combinations(range(len(cone)), size):
            coefficient_nullspace = _nullspace(constraint @ rays[:, support])
            if coefficient_nullspace.shape[1] != 1:
                continue
            coefficients = coefficient_nullspace[:, 0]
            if np.all(coefficients < -1e-9):
                coefficients = -coefficients
            if np.any(coefficients <= 1e-9):
                continue
            generator = rays[:, support] @ coefficients
            primitive = _primitive_integer_from_float(generator)
            if np.allclose(np.asarray(matrix, dtype=int) @ primitive, primitive):
                generators.add(primitive)
    return tuple(sorted(generators))


def _all_cone_faces(rays):
    rays = tuple(sorted(rays))
    faces = set()
    for size in range(len(rays) + 1):
        faces.update(tuple(sorted(face)) for face in itertools.combinations(rays, size))
    return faces


def build_auxiliary_fan(triangulation_cones, matrix):
    """Construct the finite auxiliary fan ``Sigma_L`` from source eq. (4.26)."""
    auxiliary = set()
    provenance = {}
    for ambient_cone in triangulation_cones:
        generators = _intersection_rays(ambient_cone, matrix)
        for face in _all_cone_faces(generators):
            auxiliary.add(face)
            provenance.setdefault(face, []).append(ambient_cone)
    records = []
    for cone in sorted(auxiliary):
        dimension = int(
            np.linalg.matrix_rank(np.asarray(cone, dtype=float)) if cone else 0
        )
        records.append(
            {
                "rays": [list(ray) for ray in cone],
                "dimension": dimension,
                "pointwise_L_invariant": all(
                    np.array_equal(np.asarray(matrix) @ ray, ray) for ray in cone
                ),
                "simplicial": len(cone) == dimension,
                "ambient_cones": [
                    [list(ray) for ray in ambient]
                    for ambient in sorted(provenance.get(cone, []))
                ],
            }
        )
    return records


def _fixed_component_records(auxiliary_fan, matrix, torus_shift, lambda_f):
    """Enumerate source eq. (4.34)--(4.35) fixed components."""
    nu_representatives = enumerate_projected_lattice_representatives(matrix, -1)
    components = []
    seen = set()
    fixed_subspace_dimension = int(
        np.linalg.matrix_rank(np.eye(4, dtype=float) + np.asarray(matrix, dtype=float))
    )
    for cone in auxiliary_fan:
        rays = [tuple(ray) for ray in cone["rays"]]
        sigma_dimension = int(cone["dimension"])
        for nu_record in nu_representatives:
            nu = nu_record["vector"]
            integrality_vector = _fraction_sum(
                [torus_shift, nu]
                + [tuple(Fraction(value, 2) for value in ray) for ray in rays]
            )
            if not _is_integral(integrality_vector):
                continue
            key = (tuple(rays), tuple(nu))
            if key in seen:
                continue
            seen.add(key)
            ambient_dimension = max(fixed_subspace_dimension - sigma_dimension, 0)
            vanishes_identically = (sigma_dimension + int(lambda_f)) % 2 == 1
            components.append(
                {
                    "sigma_rays": [list(ray) for ray in rays],
                    "sigma_dimension": sigma_dimension,
                    "nu": _fraction_vector_to_json(nu),
                    "nu_binary_source": nu_record["binary_source"],
                    "integrality_vector": _integer_vector(integrality_vector),
                    "fixed_toric_dimension": ambient_dimension,
                    "f_vanishes_identically": bool(vanishes_identically),
                    "hypersurface_component_dimension": (
                        ambient_dimension
                        if vanishes_identically
                        else max(ambient_dimension - 1, 0)
                    ),
                }
            )
    return sorted(
        components,
        key=lambda item: (
            item["sigma_dimension"],
            tuple(item["sigma_rays"]),
            tuple(item["nu"]["numerator"]),
        ),
    )


def _extract_dual_vertices(poly, dual_polytope=None):
    if dual_polytope is None:
        try:
            dual_polytope = poly.dual()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    for method_name in ("vertices", "points"):
        method = getattr(dual_polytope, method_name, None)
        if method is not None:
            try:
                values = np.asarray(method(), dtype=int)
            except (RuntimeError, TypeError, ValueError):
                continue
            if values.ndim == 2 and values.shape[1] == 4:
                return values
    return None


def facets_with_non_smooth_cones(poly, triangulation):
    """Dual vertices q (arXiv:2305.06363's ``Delta``) whose dual facet of
    ``Delta_circ`` intersects a non-simplicial and/or non-smooth cone of the
    triangulation's fan (line ~629, the second clause of eq. (4.45): "We
    impose the same constraint for all vertices dual to facets of
    Delta_circ that intersect non-simplicial and/or non-smooth cones of the
    Z2 symmetric toric fan Sigma").

    A facet of ``Delta_circ`` dual to vertex ``q`` is
    ``{m in Delta_circ : <q,m> = -1}`` (the standard reflexive-polytope
    facet/vertex pairing, the same convention already used by
    ``_trilayer_candidate`` in ``reproduce_fuzzy_axions_h11_4.py``). A cone
    of the fan is "supported on" that facet when all of its ray generators
    satisfy this equality. Checked for cone dimensions 2-4 (dimension <=1
    cones are trivially both simplicial and smooth).
    """

    dual_vertices = np.asarray(poly.dual().vertices(), dtype=int)
    fan = triangulation.fan()
    flagged = set()
    for dimension in (2, 3, 4):
        for cone in fan.cones(dim=dimension, formal=True):
            rays = np.asarray(cone.rays(), dtype=int)
            is_simplicial = rays.shape[0] == dimension
            if is_simplicial and bool(cone.is_smooth()):
                continue
            for vertex in dual_vertices:
                if np.all(rays @ vertex == -1):
                    flagged.add(tuple(int(value) for value in vertex))
    return flagged


def _dual_vertex_parity_evidence(matrix, torus_shift, lambda_f, dual_vertices, extra_vertices=None):
    if dual_vertices is None:
        return {"available": False, "fixed_dual_vertices": [], "violations": []}
    fixed_vertices = []
    violations = []
    matrix = np.asarray(matrix, dtype=int)
    two_t = np.asarray([int(2 * value) for value in torus_shift], dtype=int)
    extra_vertices = extra_vertices or set()
    for vertex in np.asarray(dual_vertices, dtype=int):
        vertex_key = tuple(int(value) for value in vertex)
        is_L_fixed = np.array_equal(matrix.T @ vertex, vertex)
        is_non_smooth_facet = vertex_key in extra_vertices
        if not (is_L_fixed or is_non_smooth_facet):
            continue
        vertex_tuple = [int(value) for value in vertex]
        parity = int(np.dot(two_t, vertex) + int(lambda_f)) % 2
        fixed_vertices.append(vertex_tuple)
        # Source eq. (4.45) excludes a vanishing invariant vertex
        # coefficient.  In the phase convention of eq. (4.43), that
        # coefficient vanishes when 2<t,q>+lambda_f is odd.  The special
        # trilayer construction has odd dual-layer pairing and
        # lambda_f=1, so it must pass this check. The same condition is
        # imposed on non-L-fixed vertices whose dual facet meets a
        # non-simplicial/non-smooth cone (line ~629's extension, evidence
        # supplied via `extra_vertices`).
        if parity == 1:
            violations.append(vertex_tuple)
    return {
        "available": True,
        "fixed_dual_vertices": fixed_vertices,
        "violations": violations,
        "condition": "(2*t.q + lambda_f) mod 2 == 0",
    }


def _ambient_intersection_tensor(triangulation):
    """Return the toric fourfold intersection tensor.

    CYTools includes the canonical divisor as index zero and the nonzero fan
    rays as indices one onward.  The frozen-conifold/nodal-point formula
    below uses only the latter, so the convention is retained explicitly in
    the caller.
    """

    tensor = np.asarray(
        triangulation.fan().intersection_numbers(as_np_array=True), dtype=float
    )
    if tensor.ndim != 4 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"unexpected ambient intersection tensor shape {tensor.shape}")
    return tensor


def _n_s_for_two_ray_cone(tensor, vectors, ray_points):
    """Evaluate ``n^S_{df=0} = int_S c2(K_V^-1|_S (x) N^*S)`` for a smooth
    2-dimensional cone generated by rays ``ray_points`` (Moritz,
    arXiv:2305.06363, eq. around line 649: ``n^S_{df=0} =
    int_S c2(O(K_V^-1)|_S (x) N^*S)``, expanding
    ``K_V^-1 = sum_r D_r`` gives ``D_p.D_q.(K_V^-1-D_p).(K_V^-1-D_q)``, term
    for term). Returns ``None`` if the rays cannot be uniquely matched to
    fan-vector indices (evidence unavailable, not a formula failure)."""

    ray_indices = []
    for point in ray_points:
        matches = np.flatnonzero(np.all(vectors == np.asarray(point, dtype=int), axis=1))
        if matches.size != 1:
            return None
        ray_indices.append(int(matches[0]) + 1)
    p_index, q_index = ray_indices
    n_rays = vectors.shape[0]
    l2 = l_dp = l_dq = 0.0
    for r in range(1, n_rays + 1):
        for s in range(1, n_rays + 1):
            l2 += tensor[p_index, q_index, r, s]
        l_dp += tensor[p_index, q_index, r, p_index]
        l_dq += tensor[p_index, q_index, r, q_index]
    dpdq = tensor[p_index, q_index, p_index, q_index]
    return int(round(l2 - l_dp - l_dq + dpdq))


def identity_fixed_surface_n_s_table(triangulation_cones, triangulation):
    """Populate ``n^S_{df=0}`` evidence for ``L=identity``'s 2-dimensional
    fixed components, keyed for ``classify_smoothness``'s
    ``topology["fixed_surface_n_s"]`` lookup (via ``_component_key``).

    Restricted to ``L=identity``: its ``nu`` coset ``H_-^L`` is always
    trivial (``P_-^{id}(N)`` is the zero space, since the identity has no
    ``-1`` eigenspace -- confirmed empirically, not just asserted), and its
    auxiliary fan ``Sigma_L`` reduces exactly to the ambient fan itself
    (confirmed empirically), so every 2-dimensional fixed component is a
    smooth 2-cone ``(p, q)`` of the ambient fan directly, matching Moritz
    eq. around line 572-574 (``t + (1/2) sum(sigma(1)) in N``) exactly --
    the same case ``reproduce_fuzzy_axions_h11_4.py``'s
    ``_frozen_conifold_diagnostic`` already validates end to end against
    the paper's own Table 1 numbers, just for one specific shift
    (the trilayer's ``t=p0/2``) instead of every shift. Not extended to
    ``L != identity``: there, a fixed component's own toric structure is
    generally more involved than a direct 2-divisor intersection (Sec. 4.4
    of arXiv:2305.06363), and applying this same formula there has not been
    independently derived or verified.
    """

    auxiliary_fan = build_auxiliary_fan(triangulation_cones, IDENTITY)
    tensor = _ambient_intersection_tensor(triangulation)
    fan = triangulation.fan()
    vectors = np.asarray(fan.vectors(), dtype=int)
    if tensor.shape[0] != vectors.shape[0] + 1:
        return {}
    zero_nu_vector = enumerate_projected_lattice_representatives(IDENTITY, -1)[0]["vector"]
    zero_nu = _fraction_vector_to_json(zero_nu_vector)
    table = {}
    for cone in auxiliary_fan:
        if cone["dimension"] != 2 or len(cone["rays"]) != 2:
            continue
        n_s = _n_s_for_two_ray_cone(tensor, vectors, cone["rays"])
        if n_s is None:
            continue
        component = {"sigma_rays": cone["rays"], "nu": zero_nu}
        table[_component_key(component)] = n_s
    return table


def _component_key(component):
    return json.dumps(
        {"sigma_rays": component["sigma_rays"], "nu": component["nu"]},
        sort_keys=True,
    )


def _lookup_surface_n_s(topology, component):
    evidence = topology.get("fixed_surface_n_s", {})
    if not isinstance(evidence, dict):
        return None
    value = evidence.get(_component_key(component))
    if value is None:
        value = evidence.get(str(component["sigma_rays"]))
    return None if value is None else int(value)


def classify_smoothness(
    matrix,
    torus_shift,
    lambda_f,
    auxiliary_fan,
    fixed_components,
    topology,
    dual_vertices,
):
    """Classify source smoothness checks without inventing missing evidence."""
    matrix = np.asarray(matrix, dtype=int)
    is_identity_sanity = np.array_equal(matrix, IDENTITY) and all(
        value == 0 for value in torus_shift
    )
    parity = _dual_vertex_parity_evidence(
        matrix,
        torus_shift,
        lambda_f,
        dual_vertices,
        extra_vertices=topology.get("non_smooth_facet_dual_vertices"),
    )
    if is_identity_sanity:
        return {
            "status": "smooth",
            "verdict": "smooth",
            "method": "identity_sanity_contract",
            "reason": "explicit task-level identity fixture contract",
            "dual_vertex_parity": parity,
        }

    non_smooth_reasons = []
    unavailable_reasons = []
    if parity["violations"]:
        non_smooth_reasons.append(
            "source eq. (4.45) fails for fixed dual vertices: "
            + repr(parity["violations"])
        )
    for component in fixed_components:
        if not component["f_vanishes_identically"]:
            if not all(
                len(cone["rays"]) == cone["dimension"]
                for cone in auxiliary_fan
                if set(tuple(ray) for ray in component["sigma_rays"]).issubset(
                    set(tuple(ray) for ray in cone["rays"])
                )
            ):
                unavailable_reasons.append(
                    "generic hypersurface avoidance of a non-simplicial auxiliary cone "
                    "was not certified"
                )
            continue
        if component["fixed_toric_dimension"] == 3:
            non_smooth_reasons.append(
                "eq. (4.48) permits an identically vanishing restriction on a "
                "three-dimensional fixed toric component"
            )
        elif component["fixed_toric_dimension"] == 2:
            n_s = _lookup_surface_n_s(topology, component)
            if n_s is None:
                unavailable_reasons.append(
                    "eq. (4.50) requires fixed-surface n_S evidence"
                )
            elif n_s != 0:
                # arXiv:2305.06363 line ~647-654: n^S_{df=0} counts isolated
                # nodal points on S; the source imposes n^S_{df=0}=0 to avoid
                # them, i.e. n_S!=0 -- not n_S==0 -- is the obstruction.
                # Confirmed independently against the primary source and
                # against reproduce_fuzzy_axions_h11_4.py's own
                # _frozen_conifold_diagnostic, which already validates this
                # polarity end to end against Table 1 (see
                # fuzzy_axions_2412_12012_h11_3_h11_5_table1_verification_20260818.md
                # Sec. 6). n_S==0 was the prior (backwards) condition here.
                non_smooth_reasons.append(f"eq. (4.50) gives n_S = {n_s} != 0")
        elif component["fixed_toric_dimension"] > 0:
            unavailable_reasons.append(
                "no source smoothness certificate was supplied for this fixed component"
            )

    if non_smooth_reasons:
        return {
            "status": "fixed_point_set_non_smooth",
            "verdict": "non_smooth",
            "method": "source_eq_4.45_4.48_4.50_checks",
            "reasons": non_smooth_reasons,
            "dual_vertex_parity": parity,
        }
    if unavailable_reasons:
        return {
            "status": "smoothness_verification_unavailable",
            "verdict": "not_verified",
            "method": "source_eq_4.45_4.48_4.50_checks",
            "reasons": sorted(set(unavailable_reasons)),
            "dual_vertex_parity": parity,
        }
    return {
        "status": "smooth",
        "verdict": "smooth",
        "method": "source_eq_4.45_4.48_4.50_checks",
        "reasons": [],
        "dual_vertex_parity": parity,
    }


def _fixed_point_set_description(matrix, torus_shift, fixed_components, smoothness):
    if np.array_equal(matrix, IDENTITY) and all(value == 0 for value in torus_shift):
        return {
            "description": "whole_calabi_yau",
            "component_count": len(fixed_components),
            "construction": "identity_sanity_contract",
        }
    if not fixed_components:
        return {
            "description": "empty_fixed_point_set",
            "component_count": 0,
            "construction": "auxiliary_fan_eq_4.26_and_integrality_eq_4.35",
        }
    return {
        "description": "finite_union_of_toric_fixed_components",
        "component_count": len(fixed_components),
        "construction": "auxiliary_fan_eq_4.26_and_integrality_eq_4.35",
        "smoothness_status": smoothness["status"],
    }


def _base_record(polytope_id, frst_hash, matrix, candidate_id):
    return {
        "candidate_id": candidate_id,
        "polytope_id": polytope_id,
        "frst_hash": frst_hash,
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "lattice_matrix": np.asarray(matrix, dtype=int).tolist(),
        "involution_type": None,
        "lambda_f": None,
        "torus_shift": None,
        "fixed_point_set": None,
    }


def enumerate_orientifold_candidates(
    poly,
    triangulation,
    topology,
    *,
    h11_minus_target=None,
    dual_polytope=None,
):
    """Enumerate every matrix, shift, and coefficient-parity candidate."""
    points = np.asarray(poly.points(), dtype=int)
    polytope_id = compute_polytope_id(points)
    simplices = np.asarray(triangulation.simplices(), dtype=int)
    frst_hash = compute_triangulation_hash(simplices)
    triangulation_cones = _triangulation_cones(poly, triangulation)
    dual_vertices = _extract_dual_vertices(poly, dual_polytope)
    records = []

    for matrix in enumerate_polytope_involutions(points):
        matrix_tuple = tuple(int(value) for value in matrix.flatten())
        matrix_id = stable_hash([polytope_id, frst_hash, matrix_tuple])
        base = _base_record(polytope_id, frst_hash, matrix, matrix_id)
        try:
            # The supplied H2 validator requires a concrete type string even
            # though the type is assigned after lambda_f is selected.
            validated = validate_orientifold(
                poly,
                triangulation,
                topology,
                _lattice_matrix_config(matrix, "O3/O7"),
            )
        except OrientifoldValidationFailure as exc:
            record = dict(base)
            record.update(
                {
                    "terminal_status": exc.stage or "numerical_geometry_failure",
                    "terminal_reason": str(exc),
                    "torus_shift_search_status": "not_started",
                }
            )
            records.append(record)
            continue

        auxiliary_fan = build_auxiliary_fan(triangulation_cones, matrix)
        shifts = enumerate_projected_lattice_representatives(matrix, +1)
        if not shifts:
            record = dict(base)
            record.update(
                {
                    "terminal_status": "torus_shift_search_exhausted",
                    "terminal_reason": "no representatives were generated",
                    "torus_shift_search_status": "exhausted",
                }
            )
            records.append(record)
            continue

        matrix_records = []
        for shift in shifts:
            torus_shift = shift["vector"]
            for lambda_f in (0, 1):
                involution_type = "O3/O7" if lambda_f == 1 else "O5/O9"
                candidate_id = stable_hash(
                    [matrix_id, tuple(shift["numerator"]), int(lambda_f)]
                )
                record = _base_record(polytope_id, frst_hash, matrix, candidate_id)
                fixed_components = _fixed_component_records(
                    auxiliary_fan, matrix, torus_shift, lambda_f
                )
                smoothness = classify_smoothness(
                    matrix,
                    torus_shift,
                    lambda_f,
                    auxiliary_fan,
                    fixed_components,
                    topology,
                    dual_vertices,
                )
                record.update(
                    {
                        "matrix_candidate_id": matrix_id,
                        "involution_type": involution_type,
                        "lambda_f": int(lambda_f),
                        "lambda_f_convention": (
                            "lambda_f=1 gives O3/O7 (I*[Omega]=-Omega); "
                            "lambda_f=0 gives O5/O9 (I*[Omega]=+Omega)"
                        ),
                        "torus_shift": _fraction_vector_to_json(torus_shift),
                        "torus_shift_binary_source": shift["binary_source"],
                        "auxiliary_fan": auxiliary_fan,
                        "pointwise_invariant_cones": [
                            cone
                            for cone in auxiliary_fan
                            if cone["pointwise_L_invariant"]
                        ],
                        "fixed_point_components": fixed_components,
                        "fixed_point_set": _fixed_point_set_description(
                            matrix, torus_shift, fixed_components, smoothness
                        ),
                        "smoothness": smoothness,
                        "h11_plus": int(validated["h11_plus"]),
                        "h11_minus": int(validated["h11_minus"]),
                        "h2_involution_matrix": _to_jsonable(
                            validated["h2_involution_matrix"]
                        ),
                        "invariant_kaehler_basis": _to_jsonable(
                            validated["invariant_kahler_basis"]
                        ),
                        "anti_invariant_h2_basis": _to_jsonable(
                            validated["anti_invariant_h2_basis"]
                        ),
                        "prime_divisor_image_indices": _to_jsonable(
                            validated["prime_divisor_image_indices"]
                        ),
                        "prime_divisor_invariant_indices": _to_jsonable(
                            validated["prime_divisor_invariant_indices"]
                        ),
                        "torus_shift_search_status": "exhaustive",
                    }
                )
                if smoothness["status"] != "smooth":
                    record["terminal_status"] = smoothness["status"]
                    record["terminal_reason"] = "; ".join(
                        smoothness.get("reasons", [])
                    )
                elif h11_minus_target is not None and int(validated["h11_minus"]) != int(
                    h11_minus_target
                ):
                    record["terminal_status"] = "orientifold_h11_minus_filter_rejection"
                    record["terminal_reason"] = (
                        f"h11_minus={validated['h11_minus']} does not match requested "
                        f"target {h11_minus_target}"
                    )
                else:
                    record["terminal_status"] = "accepted_verified_orientifold"
                    record["terminal_reason"] = None
                records.append(record)
                matrix_records.append(record)
        if matrix_records and not any(
            item["terminal_status"] == "accepted_verified_orientifold"
            for item in matrix_records
        ):
            for item in matrix_records:
                item["torus_shift_search_status"] = "exhausted_no_accepted_triple"
            summary_record = dict(base)
            summary_record.update(
                {
                    "record_kind": "lattice_matrix_search_summary",
                    "terminal_status": "torus_shift_search_exhausted",
                    "terminal_reason": (
                        "all enumerated torus shifts and lambda_f values were "
                        "checked without an accepted verified triple"
                    ),
                    "torus_shift_search_status": "exhausted_no_accepted_triple",
                    "attempted_triple_count": len(matrix_records),
                    "attempted_candidate_ids": [
                        item["candidate_id"] for item in matrix_records
                    ],
                }
            )
            records.append(summary_record)

    return records


def write_candidate_manifest(path, records, *, provenance):
    """Atomically write JSONL candidate records and an additive summary."""
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    summary_path = f"{absolute_path}.summary.json"

    status_counts = {status: 0 for status in TERMINAL_STATUSES}
    for record in records:
        status = record["terminal_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    accepted_frst_ids = {
        (record["polytope_id"], record["frst_hash"])
        for record in records
        if record["terminal_status"] == "accepted_verified_orientifold"
    }
    summary = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "provenance": provenance,
        "candidate_count": len(records),
        "status_counts": status_counts,
        "accepted_candidate_count": status_counts["accepted_verified_orientifold"],
        "distinct_accepted_frst_count": len(accepted_frst_ids),
        "search_completeness": (
            "exhaustive_over_lattice_involutions_and_projected_torus_shifts; "
            "both_lambda_f_values; supplied_frst_only; fixed_components_and_"
            "smoothness_evidence_recorded"
        ),
        "verification_boundary": (
            "accepted_verified_orientifold requires concrete L,t,lambda_f, "
            "fixed_point_set, and smoothness=smooth; unavailable evidence is "
            "not promoted to acceptance"
        ),
    }

    temporary_manifest = f"{absolute_path}.tmp-{os.getpid()}-{time.time_ns()}"
    with open(temporary_manifest, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_to_jsonable(record), sort_keys=True))
            stream.write("\n")
    os.replace(temporary_manifest, absolute_path)

    temporary_summary = f"{summary_path}.tmp-{os.getpid()}-{time.time_ns()}"
    with open(temporary_summary, "w", encoding="utf-8") as stream:
        json.dump(_to_jsonable(summary), stream, sort_keys=True, indent=2)
    os.replace(temporary_summary, summary_path)
    return summary
