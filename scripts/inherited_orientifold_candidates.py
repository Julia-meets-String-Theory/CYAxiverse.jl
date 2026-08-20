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
try:
    from scipy.optimize import linprog as _linprog
except ImportError:  # pragma: no cover - the cytools environment supplies scipy
    _linprog = None
from sympy import Matrix as _SympyMatrix
from sympy.matrices.normalforms import hermite_normal_form as _hermite_normal_form
from sympy.polys.matrices import DomainMatrix as _DomainMatrix
from sympy.polys.matrices.normalforms import (
    smith_normal_decomp as _smith_normal_decomp,
)

from generate_geometric_data_multitriangulation import (
    OrientifoldValidationFailure,
    validate_orientifold,
)
from glimmers_raw_frst import (
    compute_polytope_id,
    compute_polytope_normal_form_id,
    compute_triangulation_hash,
    stable_hash,
)

CANDIDATE_SCHEMA_VERSION = "cyaxiverse-inherited-orientifold-candidate-2.5"

GENERAL_FIXED_SURFACE_REASON_CODES = (
    "nonsaturated_fixed_cone_lattice",
    "missing_full_dimensional_auxiliary_cone",
    "non_simplicial_full_dimensional_auxiliary_cone",
    "incomplete_quotient_surface_fan",
    "non_smooth_ambient_cone",
    "non_smooth_surface_fan",
    "missing_restricted_cartier_data",
    "nonintegral_restricted_cartier_data",
    "inconsistent_restricted_cartier_data",
    "nonintegral_final_n_s",
)

TERMINAL_STATUSES = (
    "matrix_validation_passed",
    "numerical_geometry_failure",
    "polytope_not_preserved",
    "frst_not_preserved",
    "prime_divisor_set_not_preserved",
    "nonintegral_h2_action",
    "h2_action_not_involution",
    "torus_shift_not_involution",
    "orientifold_h11_minus_filter_rejection",
    "torus_shift_search_exhausted",
    "fixed_point_set_non_smooth",
    "smoothness_verification_unavailable",
    "accepted_verified_orientifold",
    # Kept in the vocabulary for consumers of the earlier matrix-only schema.
    "accepted_inherited_candidate",
)

IDENTITY = np.eye(4, dtype=int)
# Distinguish an omitted optional extension check from an explicitly failed
# extraction.  The latter must remain unavailable evidence, not an empty set.
_EXTRA_VERTEX_EVIDENCE_NOT_REQUESTED = object()


def _exact_rank(matrix):
    """Return the rank of an integer matrix without floating-point rounding."""

    array = np.asarray(matrix, dtype=int)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        return 0
    return int(_SympyMatrix(array.tolist()).rank())


def _exact_determinant(matrix):
    """Return the exact determinant of an integer matrix."""

    array = np.asarray(matrix, dtype=int)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return int(_SympyMatrix(array.tolist()).det())


def _select_basis_indices(points):
    """Return 4 indices into ``points`` spanning the rank-four lattice."""
    basis_rows = []
    chosen = []
    for index, point in enumerate(points):
        candidate_rows = basis_rows + [point]
        if _exact_rank(np.asarray(candidate_rows, dtype=int)) == len(candidate_rows):
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
    basis_matrix = _SympyMatrix(basis_points.T.tolist())
    basis_inverse = basis_matrix.inv()

    involutions = {tuple(IDENTITY.flatten().tolist()): IDENTITY}
    candidate_points = sorted(point_set)
    for image_tuple in itertools.product(candidate_points, repeat=4):
        image_matrix = np.asarray(image_tuple, dtype=int).T
        if _exact_determinant(image_matrix) == 0:
            continue
        candidate = _SympyMatrix(image_matrix.tolist()) * basis_inverse
        if any(getattr(value, "q", 1) != 1 for value in candidate):
            continue
        matrix = np.asarray(candidate, dtype=int)
        determinant = _exact_determinant(matrix)
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


def _fraction_vector_from_json(value):
    """Decode the exact common-denominator representation of a vector."""

    if not isinstance(value, dict):
        raise ValueError("fraction vector must be an object")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if not isinstance(numerator, (list, tuple)) or denominator is None:
        raise ValueError("fraction vector must contain numerator and denominator")
    denominator = int(denominator)
    if denominator == 0:
        raise ValueError("fraction vector denominator must be nonzero")
    return tuple(Fraction(int(item), denominator) for item in numerator)


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


def _vector_in_rational_span(vector, generators):
    """Test exact membership in the rational span of integer generators."""

    vector = tuple(Fraction(value) for value in vector)
    generators = tuple(tuple(Fraction(value) for value in generator) for generator in generators)
    if not generators:
        return not any(vector)
    generator_matrix = _SympyMatrix(
        [[value for value in column] for column in zip(*generators)]
    )
    augmented = generator_matrix.row_join(
        _SympyMatrix([[value] for value in vector])
    )
    return generator_matrix.rank() == augmented.rank()


def _nu_equal_mod_span(left, right, sigma_rays):
    """Test exact phase-label equality modulo ``span_Q(sigma)``.

    Moritz eq. (4.30) permits arbitrary coefficients on the rays whose
    coordinates vanish.  Since the labels are rational, the relevant real
    span is the same as the exact rational span.  This test deliberately uses
    rational linear algebra rather than floating rank or a bounded coefficient
    search.
    """

    left = tuple(Fraction(value) for value in left)
    right = tuple(Fraction(value) for value in right)
    difference = tuple(a - b for a, b in zip(left, right))
    return _vector_in_rational_span(difference, sigma_rays)


def _quotient_lattice_coordinates(vector, sigma_rays):
    """Return exact coordinates in the quotient lattice dual to ``sigma``.

    If ``U = span_Q(sigma)``, an exact integer basis of
    ``U^perp ∩ M`` maps ``N`` to ``Z``.  The class of ``vector`` is integral
    in ``N_R/U`` exactly when every such pairing is an integer.  The Smith
    normal form used by ``_integer_kernel_basis`` also handles nonprimitive
    and saturation-sensitive cone generators without a tolerance.
    """

    sigma_rays = tuple(tuple(int(value) for value in ray) for ray in sigma_rays)
    if sigma_rays:
        sigma_matrix = np.asarray(sigma_rays, dtype=int).T
        annihilator = _integer_kernel_basis(sigma_matrix.T).T
    else:
        annihilator = IDENTITY.copy()
    vector = tuple(Fraction(value) for value in vector)
    coordinates = tuple(_fraction_dot(row, vector) for row in annihilator)
    return annihilator, coordinates


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


def _primitive_integer_from_exact(vector):
    """Convert an exact rational ray direction to a primitive integer vector."""

    rational = [Fraction(_exact_fraction(value)) for value in vector]
    if not any(rational):
        raise ValueError("zero vector is not a ray")
    denominator = 1
    for value in rational:
        denominator = math.lcm(denominator, value.denominator)
    integer = np.asarray([int(value * denominator) for value in rational], dtype=int)
    divisor = abs(math.gcd(*integer.tolist()))
    if divisor > 1:
        integer //= divisor
    return tuple(int(value) for value in integer)


def _intersection_rays(cone, matrix):
    """Find extreme rays of a simplicial cone intersected with ``Fix(matrix)``.

    Use exact rational nullspaces.  Floating-point SVD followed by
    ``limit_denominator`` can change a ray for a numerically delicate or
    large-coordinate cone before the lattice checks run.
    """

    rays = _SympyMatrix(np.asarray(cone, dtype=int).T.tolist())
    constraint = _SympyMatrix(
        (np.eye(4, dtype=int) - np.asarray(matrix, dtype=int)).tolist()
    )
    generators = set()
    for size in range(1, len(cone) + 1):
        for support in itertools.combinations(range(len(cone)), size):
            coefficient_nullspace = (
                constraint * rays[:, list(support)]
            ).nullspace()
            if len(coefficient_nullspace) != 1:
                continue
            coefficients = [
                _exact_fraction(value) for value in coefficient_nullspace[0]
            ]
            if all(value < 0 for value in coefficients):
                coefficients = [-value for value in coefficients]
            if any(value <= 0 for value in coefficients):
                continue
            coefficient_matrix = _SympyMatrix(
                [
                    _SympyMatrix([[value.numerator]])[0, 0]
                    / value.denominator
                    for value in coefficients
                ]
            )
            generator = rays[:, list(support)] * coefficient_matrix
            primitive = _primitive_integer_from_exact(generator)
            if np.array_equal(np.asarray(matrix, dtype=int) @ primitive, primitive):
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
        dimension = _exact_rank(np.asarray(cone, dtype=int)) if cone else 0
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


def _pointwise_invariant_cone_keys(triangulation_cones, matrix):
    """Return original-fan faces whose rays are individually fixed by ``L``.

    Moritz eqs. (4.33)--(4.35) label fixed components by cones of the
    original ambient fan ``Sigma``.  The auxiliary fan ``Sigma_L`` contains
    additional intersection rays and must not be used as the component-label
    universe.
    """

    matrix = np.asarray(matrix, dtype=int)
    fixed = set()
    for ambient_cone in triangulation_cones:
        for face in _all_cone_faces(ambient_cone):
            if all(
                np.array_equal(matrix @ np.asarray(ray, dtype=int), ray)
                for ray in face
            ):
                fixed.add(tuple(sorted(tuple(int(value) for value in ray) for ray in face)))
    return tuple(sorted(fixed))


def _half_ray_shortcut_proof(auxiliary_fan, matrix, sigma_rays):
    """Return whether eq. (4.35) is proven for one fixed component.

    The source shortcut requires a smooth ``sigma`` and smooth normal
    directions.  The retained auxiliary-fan records prove the latter only
    when every full-dimensional auxiliary cone containing ``sigma`` is
    simplicial and every ambient provenance cone is a smooth unimodular
    four-cone.  Missing provenance is therefore an unavailable certificate,
    not permission to apply the shortcut.
    """

    sigma_rays = tuple(tuple(int(value) for value in ray) for ray in sigma_rays)
    if not _cone_is_smooth_in_lattice(sigma_rays, 4):
        return False, "sigma_cone_not_smooth"
    if auxiliary_fan is None:
        return False, "auxiliary_fan_missing"

    matrix = np.asarray(matrix, dtype=int)
    fixed_rank = _exact_rank(np.eye(4, dtype=int) + matrix)
    sigma_set = set(sigma_rays)
    containing_cones = []
    for cone in auxiliary_fan:
        rays = tuple(tuple(int(value) for value in ray) for ray in cone.get("rays", []))
        if cone.get("dimension") != fixed_rank or not sigma_set.issubset(set(rays)):
            continue
        containing_cones.append(cone)
    if not containing_cones:
        return False, "containing_auxiliary_cone_unavailable"

    for cone in containing_cones:
        rays = cone.get("rays", [])
        if not cone.get("simplicial") or len(rays) != fixed_rank:
            return False, "non_simplicial_auxiliary_normal_fan"
        ambient_cones = cone.get("ambient_cones", [])
        if not ambient_cones:
            return False, "ambient_provenance_unavailable"
        for ambient in ambient_cones:
            if len(ambient) != 4 or abs(_exact_determinant(np.asarray(ambient, dtype=int))) != 1:
                return False, "non_smooth_ambient_normal_direction"
    return True, "smooth_sigma_and_normal_directions_certified"


def _fixed_component_is_contained_in(component, container):
    """Test source containment from a larger cone to its proper face.

    ``container`` is the higher-dimensional fixed component, so its sigma
    rays must be a strict subset of ``component``'s rays.  The phase labels
    must then agree modulo the rational span of the container cone.
    """

    component_rays = frozenset(tuple(ray) for ray in component["sigma_rays"])
    container_rays = frozenset(tuple(ray) for ray in container["sigma_rays"])
    if not container_rays < component_rays:
        return False
    return _nu_equal_mod_span(
        _fraction_vector_from_json(component["nu"]),
        _fraction_vector_from_json(container["nu"]),
        tuple(container_rays),
    )


def _fixed_component_records(
    auxiliary_fan,
    matrix,
    torus_shift,
    lambda_f,
    *,
    fixed_cone_keys,
):
    """Enumerate source eq. (4.30)--(4.35) fixed components.

    ``fixed_cone_keys`` must come from faces of the original ambient fan.
    ``auxiliary_fan`` is retained by the caller for the geometry and
    smoothness checks of each component, but its extra intersection rays are
    not valid component labels.  For a smooth ``sigma`` with retained proof of
    smooth normal directions, use the source half-ray shortcut eq. (4.35).
    Otherwise use the general quotient-lattice condition from eq. (4.30).
    Reduce phase labels modulo the exact rational span of ``sigma`` and remove
    a component when it is contained in a component labelled by a proper
    face.  All three operations use exact rational or Smith-normal-form
    arithmetic.
    """
    nu_representatives = enumerate_projected_lattice_representatives(matrix, -1)
    # The connected fixed torus has the +1 eigenspace of L.  Keep the
    # source convention (rank(I+L)) while evaluating the rank exactly.
    fixed_subspace_dimension = _exact_rank(
        np.eye(4, dtype=int) + np.asarray(matrix, dtype=int)
    )
    admissible = []
    for source_rays in fixed_cone_keys:
        rays = [tuple(int(value) for value in ray) for ray in source_rays]
        sigma_dimension = _exact_rank(np.asarray(rays, dtype=int)) if rays else 0
        use_half_ray, shortcut_reason = _half_ray_shortcut_proof(
            auxiliary_fan,
            matrix,
            rays,
        )
        integrality_method = (
            "smooth_half_ray_eq_4.35"
            if use_half_ray
            else "general_quotient_lattice_eq_4.30"
        )
        for nu_record in nu_representatives:
            nu = nu_record["vector"]
            if use_half_ray:
                integrality_vector = _fraction_sum(
                    [torus_shift, nu]
                    + [tuple(Fraction(value, 2) for value in ray) for ray in rays]
                )
                if not _is_integral(integrality_vector):
                    continue
                quotient_annihilator = None
                quotient_coordinates = None
            else:
                integrality_vector = _fraction_sum([torus_shift, nu])
                quotient_annihilator, quotient_coordinates = _quotient_lattice_coordinates(
                    integrality_vector,
                    rays,
                )
                if any(value.denominator != 1 for value in quotient_coordinates):
                    continue
            ambient_dimension = fixed_subspace_dimension - sigma_dimension
            if ambient_dimension < 0:
                continue
            vanishes_identically = (sigma_dimension + int(lambda_f)) % 2 == 1
            # A non-vanishing section has no zero-dimensional hypersurface
            # component; do not persist the empty intersection as a fixed
            # component record.
            if ambient_dimension == 0 and not vanishes_identically:
                continue
            admissible.append(
                {
                    "sigma_rays": [list(ray) for ray in rays],
                    "sigma_dimension": sigma_dimension,
                    "nu": _fraction_vector_to_json(nu),
                    "nu_binary_source": nu_record["binary_source"],
                    "integrality_vector": (
                        _integer_vector(integrality_vector)
                        if _is_integral(integrality_vector)
                        else None
                    ),
                    "fixed_component_integrality": {
                        "method": integrality_method,
                        "shortcut_proof": shortcut_reason,
                        "quotient_annihilator": (
                            None
                            if quotient_annihilator is None
                            else np.asarray(quotient_annihilator, dtype=int).tolist()
                        ),
                        "quotient_coordinates": (
                            None
                            if quotient_coordinates is None
                            else [_fraction_to_json(value) for value in quotient_coordinates]
                        ),
                    },
                    "fixed_toric_dimension": ambient_dimension,
                    "f_vanishes_identically": bool(vanishes_identically),
                    "hypersurface_component_dimension": (
                        ambient_dimension
                        if vanishes_identically
                        else max(ambient_dimension - 1, 0)
                    ),
                }
            )

    # First keep one deterministic representative for each phase class modulo
    # span(sigma).  The first record is deterministic because the projected
    # representatives are enumerated in sorted rational-vector order.
    canonical = []
    for component in admissible:
        if any(
            tuple(other["sigma_rays"]) == tuple(component["sigma_rays"])
            and _nu_equal_mod_span(
                _fraction_vector_from_json(component["nu"]),
                _fraction_vector_from_json(other["nu"]),
                tuple(tuple(ray) for ray in component["sigma_rays"]),
            )
            for other in canonical
        ):
            continue
        canonical.append(component)

    # Remove lower-dimensional fixed components already contained in a
    # higher-dimensional component.  A proper face has the larger orbit
    # closure, so the containment direction is ``component -> container``.
    retained = []
    for component in canonical:
        rays = frozenset(tuple(ray) for ray in component["sigma_rays"])
        contained_in_admissible_face = any(
            _fixed_component_is_contained_in(component, other)
            for other in canonical
        )
        if contained_in_admissible_face:
            continue
        retained.append(component)

    return sorted(
        retained,
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
    of the fan geometrically intersects that facet when at least one boundary
    ray satisfies this equality. Requiring every ray to pair ``-1`` misses
    cones that meet the facet along a proper face. Checked for cone dimensions
    2-4 (dimension <=1 cones are trivially both simplicial and smooth).
    """

    dual_vertices = _extract_dual_vertices(poly)
    if dual_vertices is None:
        return None
    fan = triangulation.fan()
    flagged = set()
    for dimension in (2, 3, 4):
        for cone in fan.cones(dim=dimension, formal=True):
            rays = np.asarray(cone.rays(), dtype=int)
            is_simplicial = rays.shape[0] == dimension
            if is_simplicial and bool(cone.is_smooth()):
                continue
            for vertex in dual_vertices:
                if np.any(rays @ vertex == -1):
                    flagged.add(tuple(int(value) for value in vertex))
    return flagged


def _dual_vertex_parity_evidence(
    matrix,
    torus_shift,
    lambda_f,
    dual_vertices,
    extra_vertices=_EXTRA_VERTEX_EVIDENCE_NOT_REQUESTED,
):
    if dual_vertices is None:
        return {"available": False, "fixed_dual_vertices": [], "violations": []}
    if extra_vertices is None:
        # ``facets_with_non_smooth_cones`` returns None when dual-vertex
        # extraction fails.  Do not interpret that failed check as evidence
        # that no additional vertices require the parity condition.
        return {"available": False, "fixed_dual_vertices": [], "violations": []}
    fixed_vertices = []
    violations = []
    noninteger_pairings = []
    matrix = np.asarray(matrix, dtype=int)
    # `2t` must stay EXACT. For a non-identity `L` the torus shift carries
    # quarter-integer components (`torus_shift = (I+L)*bits / 4`), so `2t` can
    # be a genuine half-integer vector. The prior `int(2*value)` rounded each
    # such component toward zero -- e.g. `2t=(1/2,1/2,0,0)` became `(0,0,0,0)`
    # -- silently corrupting eq. (4.45)'s parity for every non-identity-`L`
    # candidate, the dominant rejection channel in that sector. Identity `L`
    # never triggered it (its `2t` is always integral), which is exactly why
    # identity-only validation could not have caught it. Keep `2t` as exact
    # `Fraction`s and contract against the (integer) dual vertex before any
    # rounding.
    two_t = [Fraction(2) * Fraction(value) for value in torus_shift]
    if extra_vertices is _EXTRA_VERTEX_EVIDENCE_NOT_REQUESTED:
        extra_vertices = set()
    for vertex in np.asarray(dual_vertices, dtype=int):
        vertex_key = tuple(int(value) for value in vertex)
        is_L_fixed = np.array_equal(matrix.T @ vertex, vertex)
        is_non_smooth_facet = vertex_key in extra_vertices
        if not (is_L_fixed or is_non_smooth_facet):
            continue
        vertex_tuple = [int(value) for value in vertex]
        fixed_vertices.append(vertex_tuple)
        # Source eq. (4.45) excludes a vanishing invariant vertex
        # coefficient.  In the phase convention of eq. (4.43), that
        # coefficient vanishes when 2<t,q>+lambda_f is odd.  The special
        # trilayer construction has odd dual-layer pairing and
        # lambda_f=1, so it must pass this check. The same condition is
        # imposed on non-L-fixed vertices whose dual facet meets a
        # non-simplicial/non-smooth cone (line ~629's extension, evidence
        # supplied via `extra_vertices`).
        #
        # `<2t, q>` is provably integral for an `L`-fixed `q` (there
        # `<2t,q> = <(I+L)bits, q>/... ` reduces to an integer pairing). The
        # only way it can be non-integral is an extension vertex whose facet
        # meets a non-smooth cone; such a vertex cannot satisfy the integer
        # "== 0 mod 2" condition at all, so it is a violation by the same
        # token. These are tallied separately so their frequency is visible.
        pairing = sum(
            coeff * int(component) for coeff, component in zip(two_t, vertex)
        )
        total = pairing + int(lambda_f)
        if total.denominator != 1:
            noninteger_pairings.append(vertex_tuple)
            violations.append(vertex_tuple)
        elif int(total) % 2 == 1:
            violations.append(vertex_tuple)
    return {
        "available": True,
        "fixed_dual_vertices": fixed_vertices,
        "violations": violations,
        "noninteger_pairings": noninteger_pairings,
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


def _cone_has_smooth_star(fan, sigma_rays):
    """Is ``S = D_p . D_q`` (the toric surface dual to the 2-cone
    ``sigma=(p,q)``) itself smooth?

    ``sigma`` being a smooth (unimodular) 2-cone is not sufficient: cone
    multiplicity is multiplicative across a smooth sub-face's star (choose
    lattice coordinates with ``p,q`` as the first two basis vectors -- any
    cone ``tau ⊇ sigma`` then has block-triangular generator matrix, so
    ``mult(tau) = mult(sigma) * mult(star_image(tau)) = mult(star_image(tau))``
    since ``mult(sigma)=1``), so ``tau`` is smooth iff its star image in
    ``N/Z{p,q}`` is -- and that star image *is* the local toric structure of
    ``S`` itself. So ``sigma`` being a face of any non-simplicial-or-non-
    smooth cone elsewhere in the fan means ``S`` has a genuine orbifold
    point there, independent of whether ``sigma`` itself looks smooth in
    isolation. Verified directly (not just cited) against real fan data in
    validation/fuzzy_axions_2412_12012_n_s_orbifold_contamination_20260819.md
    Sec. 4: the star-image determinant matched ``cone.is_smooth()`` for the
    containing cone exactly, with zero exceptions.
    """

    sigma_set = set(tuple(int(value) for value in ray) for ray in sigma_rays)
    for dimension in (3, 4):
        for cone in fan.cones(dim=dimension, formal=True):
            rays = np.asarray(cone.rays(), dtype=int)
            ray_set = set(tuple(int(value) for value in ray) for ray in rays)
            if not sigma_set.issubset(ray_set):
                continue
            is_simplicial = rays.shape[0] == dimension
            if not (is_simplicial and bool(cone.is_smooth())):
                return False
    return True


def _integer_kernel_basis(matrix):
    """Return an integer basis for the kernel of an integer matrix."""

    array = np.asarray(matrix, dtype=int)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rows, columns = array.shape
    if rows == 0:
        return np.eye(columns, dtype=int)
    domain_matrix = _DomainMatrix.from_Matrix(_SympyMatrix(array.tolist()))
    smith, _, right = _smith_normal_decomp(domain_matrix)
    diagonal = smith.to_Matrix()
    rank = sum(
        index < diagonal.rows
        and index < diagonal.cols
        and diagonal[index, index] != 0
        for index in range(min(diagonal.rows, diagonal.cols))
    )
    return np.asarray(right.to_Matrix()[:, rank:], dtype=int)


def _integer_coordinates(basis, vector):
    """Express ``vector`` in the columns of ``basis`` over the integers."""

    basis = _SympyMatrix(np.asarray(basis, dtype=int).tolist())
    vector = _SympyMatrix(np.asarray(vector, dtype=int).reshape(-1, 1).tolist())
    try:
        coordinates = basis.gauss_jordan_solve(vector)[0]
    except (ValueError, TypeError):
        return None
    values = []
    for value in coordinates:
        if getattr(value, "q", 1) != 1:
            return None
        values.append(int(value))
    return np.asarray(values, dtype=int)


def _sublattice_is_saturated(generators):
    """Check that a sublattice has no finite-index saturation defect."""

    generators = np.asarray(generators, dtype=int)
    rank, count = generators.shape
    if count == 0:
        return True
    if count > rank:
        return False
    minors = []
    for rows in itertools.combinations(range(rank), count):
        minors.append(_exact_determinant(generators[np.ix_(rows, range(count))]))
    return abs(math.gcd(*minors)) == 1


def _surface_divisor_intersection(left, right, rays, cones):
    """Intersect two toric divisors on a smooth complete surface fan."""

    left = {tuple(ray): value for ray, value in left.items()}
    right = {tuple(ray): value for ray, value in right.items()}
    rays = list(rays)
    angles = {
        ray: math.atan2(float(ray[1]), float(ray[0])) for ray in rays
    }
    ordered = sorted(rays, key=lambda ray: angles[ray])
    result = Fraction(0)
    cone_set = {frozenset(cone) for cone in cones}
    for index, ray in enumerate(ordered):
        previous_ray = ordered[index - 1]
        next_ray = ordered[(index + 1) % len(ordered)]
        previous = np.asarray(previous_ray, dtype=int)
        current = np.asarray(ray, dtype=int)
        following = np.asarray(next_ray, dtype=int)
        denominator = _exact_determinant(np.column_stack((previous, current)))
        next_denominator = _exact_determinant(
            np.column_stack((current, following))
        )
        if abs(denominator) != 1 or abs(next_denominator) != 1:
            raise ValueError("surface fan contains a non-smooth two-cone")
        self_intersection = -_exact_determinant(
            np.column_stack((previous, following))
        )
        result += self_intersection * left[ray] * right[ray]
        if frozenset((ray, next_ray)) not in cone_set:
            raise ValueError("surface fan is not complete around a ray")
        result += left[ray] * right[next_ray]
        result += left[next_ray] * right[ray]
    return result


def _ambient_cartier_data(ambient_cone, divisor_point):
    """Return the local support function for one ambient toric divisor."""

    rays = np.asarray(ambient_cone, dtype=int)
    if rays.shape != (4, 4) or abs(_exact_determinant(rays)) != 1:
        return None
    target = np.asarray(
        [-int(np.array_equal(ray, divisor_point)) for ray in rays], dtype=int
    )
    solution = _SympyMatrix(rays.tolist()).inv() * _SympyMatrix(target.tolist())
    return tuple(solution)


def _fraction_to_json(value):
    """Encode one exact rational as an integer or numerator/denominator pair."""

    value = Fraction(value)
    if value.denominator == 1:
        return int(value)
    return {"numerator": int(value.numerator), "denominator": int(value.denominator)}


POSITIVE_COMPONENT_MAX_DIMENSION = 3
MAX_SECTION_LATTICE_POINTS = 100_000


def _exact_fraction(value):
    """Convert an exact SymPy or Python rational to ``Fraction``."""

    if isinstance(value, Fraction):
        return value
    numerator = getattr(value, "p", None)
    denominator = getattr(value, "q", None)
    if numerator is not None and denominator is not None:
        return Fraction(int(numerator), int(denominator))
    return Fraction(value)


def _ambient_anticanonical_cartier_data(ambient_cone):
    """Return exact local Cartier data for ``-K_V`` on a simplicial cone.

    A non-unimodular simplicial cone can still be Gorenstein: the local
    anticanonical support function is an integral dual-lattice vector whose
    pairing with every primitive ray is ``-1``.  Solve that condition exactly
    and retain only integral solutions.  Non-simplicial, rank-deficient, or
    genuinely non-Cartier provenance remains unavailable.
    """

    rays = np.asarray(ambient_cone, dtype=int)
    if rays.shape != (4, 4):
        return None
    ray_matrix = _SympyMatrix(rays.tolist())
    if ray_matrix.det() == 0:
        return None
    solution = ray_matrix.inv() * _SympyMatrix([-1, -1, -1, -1])
    if any(getattr(value, "q", 1) != 1 for value in solution):
        return None
    return tuple(_exact_fraction(value) for value in solution)


def _rational_coordinates(basis, vector):
    """Express a rational vector in a full-column-rank rational basis."""

    basis = _SympyMatrix(np.asarray(basis, dtype=object).tolist())
    vector = _SympyMatrix([_exact_fraction(value) for value in vector])
    try:
        solution = basis.gauss_jordan_solve(vector)[0]
    except (ValueError, TypeError):
        return None
    if any(getattr(value, "free_symbols", set()) for value in solution):
        return None
    return tuple(_exact_fraction(value) for value in solution)


def _fraction_dot(left, right):
    """Take an exact dot product."""

    return sum(
        (_exact_fraction(left_value) * _exact_fraction(right_value))
        for left_value, right_value in zip(left, right)
    )


def _primitive_quotient_vector(vector):
    """Return a primitive quotient ray and its positive integral scale."""

    vector = np.asarray(vector, dtype=int)
    divisor = abs(math.gcd(*[int(value) for value in vector.tolist()]))
    if divisor == 0:
        return None
    primitive = tuple(int(value // divisor) for value in vector)
    return primitive, divisor


def _cone_is_smooth_in_lattice(cone, dimension):
    """Test smoothness of a simplicial cone in ``Z^dimension`` exactly."""

    cone = tuple(tuple(int(value) for value in ray) for ray in cone)
    cone_dimension = len(cone)
    if cone_dimension == 0:
        return True
    matrix = np.asarray(cone, dtype=int).T
    if matrix.shape != (dimension, cone_dimension):
        return False
    if cone_dimension == dimension:
        return abs(int(_SympyMatrix(matrix.tolist()).det())) == 1
    minors = []
    for rows in itertools.combinations(range(dimension), cone_dimension):
        minors.append(
            abs(int(_SympyMatrix(matrix[list(rows), :].tolist()).det()))
        )
    return math.gcd(*minors) == 1


def _cone_facet_normals(cone):
    """Return exact facet covectors for a full-dimensional simplicial cone."""

    matrix = _SympyMatrix(np.asarray(cone, dtype=int).T.tolist())
    inverse = matrix.inv()
    return tuple(
        tuple(_exact_fraction(value) for value in row)
        for row in inverse.tolist()
    )


def _intersection_cone_extreme_rays(left, right, dimension):
    """Return the exact extreme rays of two full-dimensional cones.

    The inverse ray matrices give the inequalities defining each simplicial
    cone.  In dimensions one through three, every nonzero extreme ray of the
    intersection lies on ``dimension - 1`` of those facet hyperplanes, so
    enumerating their exact rational nullspaces is complete.  This is used to
    distinguish a genuine common face from a pair of cones that merely has
    the same codimension-one incidence counts.
    """

    if dimension == 1:
        return (left[0],) if left[0] == right[0] else ()
    normals = _cone_facet_normals(left) + _cone_facet_normals(right)
    intersection_rays = set()
    for selected in itertools.combinations(normals, dimension - 1):
        normal_matrix = _SympyMatrix([list(normal) for normal in selected])
        nullspace = normal_matrix.nullspace()
        if len(nullspace) != 1:
            continue
        direction = tuple(_exact_fraction(value) for value in nullspace[0])
        for sign in (1, -1):
            candidate = tuple(sign * value for value in direction)
            if not any(candidate):
                continue
            if all(
                _fraction_dot(normal, candidate) >= 0 for normal in normals
            ):
                intersection_rays.add(_primitive_integer_from_exact(candidate))
    return tuple(sorted(intersection_rays))


def _strict_positive_hull_certificate(rays, dimension):
    """Check that the origin is strictly inside the convex hull of ``rays``.

    A complete fan must contain the origin in the interior of its ray hull.
    The LP maximizes the common positive weight in
    ``sum_i w_i ray_i = 0``, ``sum_i w_i = 1``.  An inconclusive numerical
    margin is rejected conservatively; it never promotes a fan to certified.
    """

    if _linprog is None or not rays:
        return False
    ray_array = np.asarray(rays, dtype=float)
    if ray_array.shape != (len(rays), dimension) or not np.all(
        np.isfinite(ray_array)
    ):
        return False
    count = len(rays)
    equality = np.zeros((dimension + 1, count + 1), dtype=float)
    equality[:dimension, :count] = ray_array.T
    equality[dimension, :count] = 1.0
    upper = np.zeros((count, count + 1), dtype=float)
    upper[:, :count] = -np.eye(count, dtype=float)
    upper[:, -1] = 1.0
    result = _linprog(
        c=np.r_[np.zeros(count, dtype=float), -1.0],
        A_ub=upper,
        b_ub=np.zeros(count, dtype=float),
        A_eq=equality,
        b_eq=np.r_[np.zeros(dimension, dtype=float), 1.0],
        bounds=[(0.0, 1.0)] * count + [(0.0, 1.0)],
        method="highs",
    )
    return bool(result.success and result.x[-1] > 1e-9)


def _complete_simplicial_fan(maximal_cones, dimension):
    """Check completeness, intersections, and simplicity of a finite fan.

    The bounded section certificate requires more than a codimension-one
    incidence count.  This predicate therefore verifies primitive rays,
    full-dimensional simplicial cones, exact pairwise common-face
    intersections, the two-sided codimension-one incidence condition, and
    vector-space coverage via a strict positive-hull test.  Any unsupported
    or numerically ambiguous case returns ``False`` so callers keep it
    unavailable instead of treating it as a smoothness certificate.
    """

    try:
        dimension = int(dimension)
    except (TypeError, ValueError):
        return False
    if dimension < 1 or dimension > POSITIVE_COMPONENT_MAX_DIMENSION:
        return False
    if not maximal_cones:
        return False

    normalized_cones = []
    for cone in maximal_cones:
        try:
            normalized_rays = []
            for ray in cone:
                normalized_ray = []
                for value in ray:
                    rational = _exact_fraction(value)
                    if rational.denominator != 1:
                        return False
                    normalized_ray.append(int(rational))
                normalized_rays.append(tuple(normalized_ray))
            normalized = tuple(normalized_rays)
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if len(normalized) != dimension:
            return False
        if any(len(ray) != dimension for ray in normalized):
            return False
        if len(set(normalized)) != dimension:
            return False
        if any(
            math.gcd(*[abs(int(value)) for value in ray]) != 1
            for ray in normalized
        ):
            return False
        if abs(int(_SympyMatrix(np.asarray(normalized, dtype=int).T.tolist()).det())) == 0:
            return False
        normalized_cones.append(tuple(sorted(normalized)))

    if len(set(normalized_cones)) != len(normalized_cones):
        return False
    all_rays = tuple(sorted({ray for cone in normalized_cones for ray in cone}))
    if not _strict_positive_hull_certificate(all_rays, dimension):
        return False

    codimension_one_faces = {}
    for cone in normalized_cones:
        for face in itertools.combinations(sorted(cone), max(dimension - 1, 0)):
            key = frozenset(face)
            codimension_one_faces[key] = codimension_one_faces.get(key, 0) + 1
    if not codimension_one_faces or any(
        count != 2 for count in codimension_one_faces.values()
    ):
        return False

    for left_index, left in enumerate(normalized_cones):
        for right in normalized_cones[left_index + 1 :]:
            common_rays = set(left).intersection(right)
            intersection_rays = set(
                _intersection_cone_extreme_rays(left, right, dimension)
            )
            if intersection_rays != common_rays:
                return False
    return True


def _toric_section_lattice_points(rays, coefficients, dimension):
    """Enumerate lattice points of a bounded toric divisor polytope exactly."""

    if dimension < 1 or dimension > POSITIVE_COMPONENT_MAX_DIMENSION:
        return {
            "status": "unavailable",
            "reason_code": "unsupported_section_polytope_dimension",
            "reason": (
                "exact section-polytope enumeration is bounded to toric "
                f"dimension {POSITIVE_COMPONENT_MAX_DIMENSION}"
            ),
        }
    rays = tuple(sorted(tuple(int(value) for value in ray) for ray in rays))
    vertices = set()
    for selected in itertools.combinations(range(len(rays)), dimension):
        matrix = _SympyMatrix([rays[index] for index in selected])
        determinant = int(matrix.det())
        if determinant == 0:
            continue
        right_hand_side = _SympyMatrix(
            [-_exact_fraction(coefficients[rays[index]]) for index in selected]
        )
        solution = matrix.inv() * right_hand_side
        point = tuple(_exact_fraction(value) for value in solution)
        if all(
            _fraction_dot(point, ray) >= -_exact_fraction(coefficients[ray])
            for ray in rays
        ):
            vertices.add(point)
    if not vertices:
        return {
            "status": "unavailable",
            "reason_code": "section_polytope_unavailable",
            "reason": (
                "the restricted anti-canonical divisor has no certifiable "
                "bounded polytope vertices"
            ),
        }

    lower = [min(point[index] for point in vertices) for index in range(dimension)]
    upper = [max(point[index] for point in vertices) for index in range(dimension)]
    ranges = [
        range(math.floor(value), math.ceil(upper[index]) + 1)
        for index, value in enumerate(lower)
    ]
    search_size = math.prod(len(values) for values in ranges)
    if search_size > MAX_SECTION_LATTICE_POINTS:
        return {
            "status": "unavailable",
            "reason_code": "section_lattice_search_bounded",
            "reason": (
                f"the exact section-lattice search would inspect {search_size} "
                f"integer points, above the bound {MAX_SECTION_LATTICE_POINTS}"
            ),
        }
    points = []
    for candidate in itertools.product(*ranges):
        point = tuple(Fraction(int(value)) for value in candidate)
        if all(
            _fraction_dot(point, ray) >= -_exact_fraction(coefficients[ray])
            for ray in rays
        ):
            points.append(point)
    return {
        "status": "certified",
        "reason_code": None,
        "reason": None,
        "points": tuple(sorted(set(points))),
        "vertices": tuple(sorted(vertices)),
    }


def _positive_component_section_certificate(auxiliary_fan, matrix, component):
    """Certify Sec. 4.6 for a positive-dimensional non-vanishing component.

    The component is the toric orbit closure of ``sigma`` in the auxiliary
    fixed fan.  For dimensions one through three, construct its star fan in
    the exact quotient lattice, restrict the ambient anti-canonical Cartier
    data, and apply two source-matched tests:

    * nefness: every local support-function representative lies in the
      restricted divisor polytope, equivalently the support function is
      convex on the complete fan;
    * orbifold avoidance: for each non-smooth cone with positive-dimensional
      orbit, the restricted section polytope face has exactly one lattice
      point.  More than one point gives a non-monomial Laurent restriction,
      which generically has a zero on that orbit.

    Missing provenance, non-Cartier data, incomplete fans, unsupported
    dimensions, and bounded-search limits are unavailable rather than
    accepted.
    """

    matrix = np.asarray(matrix, dtype=int)
    invariant_basis = _integer_kernel_basis(np.eye(4, dtype=int) - matrix)
    fixed_rank = int(invariant_basis.shape[1])
    sigma_rays = tuple(
        tuple(int(value) for value in ray) for ray in component.get("sigma_rays", [])
    )
    sigma_dimension = int(component.get("sigma_dimension", len(sigma_rays)))
    component_dimension = fixed_rank - sigma_dimension
    supplied_component_dimension = component.get("fixed_toric_dimension")
    base = {
        "component_sigma_rays": [list(ray) for ray in sigma_rays],
        "component_sigma_dimension": sigma_dimension,
        "fixed_toric_dimension": component_dimension,
        "fixed_lattice_basis": np.asarray(invariant_basis, dtype=int).tolist(),
    }

    def unavailable(reason_code, reason, **extra):
        return {
            **base,
            **extra,
            "status": "unavailable",
            "reason_code": reason_code,
            "reason": reason,
            "nefness": {
                "status": "unavailable",
                "reason_code": reason_code,
                "reason": reason,
            },
            "orbifold_intersection": {
                "status": "unavailable",
                "reason_code": reason_code,
                "reason": reason,
            },
        }

    if component_dimension < 1:
        return unavailable(
            "nonpositive_fixed_component_dimension",
            "the supplied fixed-component dimension is not positive",
        )
    if (
        supplied_component_dimension is not None
        and int(supplied_component_dimension) != component_dimension
    ):
        return unavailable(
            "inconsistent_fixed_component_dimension",
            "the fixed-component dimension disagrees with the invariant lattice rank",
        )
    if component_dimension > POSITIVE_COMPONENT_MAX_DIMENSION:
        return unavailable(
            "unsupported_fixed_component_dimension",
            (
                "the bounded toric support-function and section-polytope "
                f"certificate supports dimensions 1 through {POSITIVE_COMPONENT_MAX_DIMENSION}"
            ),
        )
    if sigma_dimension != len(sigma_rays):
        return unavailable(
            "nonsaturated_fixed_cone_lattice",
            "the fixed cone ray count does not match its supplied dimension",
        )
    if any(
        not np.array_equal(matrix @ np.asarray(ray, dtype=int), np.asarray(ray, dtype=int))
        for ray in sigma_rays
    ):
        return unavailable(
            "nonintegral_fixed_component_lattice",
            "a fixed-component ray is not pointwise invariant under L",
        )

    sigma_coordinates = []
    for ray in sigma_rays:
        coordinates = _integer_coordinates(invariant_basis, ray)
        if coordinates is None:
            return unavailable(
                "nonintegral_fixed_component_lattice",
                "a fixed-component ray has no integral coordinate in the invariant lattice",
            )
        if math.gcd(*[abs(int(value)) for value in coordinates.tolist()]) != 1:
            return unavailable(
                "nonsaturated_fixed_cone_lattice",
                "a fixed-component ray is not primitive in the invariant lattice",
            )
        sigma_coordinates.append(coordinates)
    sigma_matrix = (
        np.column_stack(sigma_coordinates)
        if sigma_coordinates
        else np.empty((fixed_rank, 0), dtype=int)
    )
    if not _sublattice_is_saturated(sigma_matrix):
        return unavailable(
            "nonsaturated_fixed_cone_lattice",
            "the fixed-cone rays generate a proper finite-index sublattice of the invariant lattice",
        )
    quotient_annihilator = _integer_kernel_basis(sigma_matrix.T).T
    if quotient_annihilator.shape != (component_dimension, fixed_rank):
        return unavailable(
            "incomplete_fixed_component_fan",
            "the fixed-cone quotient does not have the supplied dimension",
        )

    maximal_cones = {}
    reference_support = None
    for full_cone in auxiliary_fan:
        full_rays = tuple(
            tuple(int(value) for value in ray) for ray in full_cone.get("rays", [])
        )
        if int(full_cone.get("dimension", -1)) != fixed_rank:
            continue
        if not set(sigma_rays).issubset(full_rays):
            continue
        if not bool(full_cone.get("simplicial")) or len(full_rays) != fixed_rank:
            return unavailable(
                "non_simplicial_fixed_component_fan",
                "a full-dimensional auxiliary cone containing the fixed component is non-simplicial",
            )
        ambient_cones = full_cone.get("ambient_cones", [])
        if not ambient_cones:
            return unavailable(
                "missing_ambient_cone_provenance",
                "the full-dimensional auxiliary cone has no ambient-cone provenance for -K_V",
            )
        quotient_rays = []
        quotient_scales = {}
        for ray in full_rays:
            if ray in sigma_rays:
                continue
            coordinates = _integer_coordinates(invariant_basis, ray)
            if coordinates is None:
                return unavailable(
                    "nonintegral_fixed_component_lattice",
                    "an auxiliary fixed ray has no integral coordinate in the invariant lattice",
                )
            quotient_vector = quotient_annihilator @ coordinates
            primitive_data = _primitive_quotient_vector(quotient_vector)
            if primitive_data is None:
                return unavailable(
                    "incomplete_fixed_component_fan",
                    "an auxiliary ray vanishes in the fixed-cone quotient lattice",
                )
            primitive, scale = primitive_data
            if primitive in quotient_rays:
                return unavailable(
                    "incomplete_fixed_component_fan",
                    "a full-dimensional auxiliary cone induces duplicate quotient rays",
                )
            quotient_rays.append(primitive)
            quotient_scales[primitive] = int(scale)
        if len(quotient_rays) != component_dimension:
            return unavailable(
                "incomplete_fixed_component_fan",
                "a full-dimensional auxiliary cone does not induce the required quotient cone dimension",
            )

        ambient_supports = []
        for ambient in ambient_cones:
            ambient_data = _ambient_anticanonical_cartier_data(ambient)
            if ambient_data is None:
                return unavailable(
                    "missing_ambient_cartier_data",
                    "an ambient provenance cone is not a smooth unimodular four-cone",
                )
            ambient_supports.append(
                tuple(
                    sum(
                        _exact_fraction(invariant_basis[row, column])
                        * ambient_data[row]
                        for row in range(4)
                    )
                    for column in range(fixed_rank)
                )
            )
        if any(support != ambient_supports[0] for support in ambient_supports[1:]):
            return unavailable(
                "inconsistent_ambient_cartier_data",
                "ambient provenance cones give inconsistent restrictions of -K_V",
            )
        local_ambient_support = ambient_supports[0]
        if reference_support is None:
            reference_support = local_ambient_support
        difference = tuple(
            local - reference
            for local, reference in zip(local_ambient_support, reference_support)
        )
        if any(_fraction_dot(difference, coordinates) != 0 for coordinates in sigma_coordinates):
            return unavailable(
                "inconsistent_restricted_cartier_data",
                "local -K_V support functions do not agree along the fixed cone",
            )
        quotient_support = _rational_coordinates(quotient_annihilator.T, difference)
        if quotient_support is None:
            return unavailable(
                "inconsistent_restricted_cartier_data",
                "the restricted -K_V support function is not integral in the quotient dual lattice",
            )
        coefficients = {
            ray: -_fraction_dot(quotient_support, ray) for ray in quotient_rays
        }
        cone_key = frozenset(quotient_rays)
        record = {
            "rays": tuple(sorted(quotient_rays)),
            "support": quotient_support,
            "coefficients": coefficients,
            "scales": quotient_scales,
        }
        previous = maximal_cones.get(cone_key)
        if previous is not None and (
            previous["support"] != record["support"]
            or previous["coefficients"] != record["coefficients"]
        ):
            return unavailable(
                "inconsistent_restricted_cartier_data",
                "duplicate quotient cones have inconsistent restricted Cartier data",
            )
        maximal_cones[cone_key] = record

    if not maximal_cones:
        return unavailable(
            "missing_full_dimensional_auxiliary_cone",
            "no full-dimensional auxiliary cone contains the fixed component",
        )
    maximal_records = [maximal_cones[key] for key in sorted(maximal_cones, key=lambda value: tuple(sorted(value)))]
    maximal_ray_sets = [record["rays"] for record in maximal_records]
    if not _complete_simplicial_fan(maximal_ray_sets, component_dimension):
        return unavailable(
            "incomplete_fixed_component_fan",
            "the containing auxiliary cones do not form a complete simplicial quotient fan",
        )

    coefficients = {}
    for record in maximal_records:
        for ray, coefficient in record["coefficients"].items():
            previous = coefficients.get(ray)
            if previous is not None and previous != coefficient:
                return unavailable(
                    "inconsistent_restricted_cartier_data",
                    "the restricted -K_V divisor has inconsistent coefficients on a quotient ray",
                )
            coefficients[ray] = coefficient
    nef_witnesses = []
    support_inequality_margins = []
    for record in maximal_records:
        for ray in record["rays"]:
            degree = _fraction_dot(record["support"], ray) + coefficients[ray]
            support_inequality_margins.append(
                {
                    "maximal_cone": [list(item) for item in record["rays"]],
                    "ray": list(ray),
                    "margin": _fraction_to_json(degree),
                }
            )
        for ray in sorted(coefficients):
            margin = _fraction_dot(record["support"], ray) + coefficients[ray]
            if margin < 0:
                nef_witnesses.append(
                    {
                        "maximal_cone": [list(item) for item in record["rays"]],
                        "ray": list(ray),
                        "support_inequality_margin": _fraction_to_json(margin),
                    }
                )
    nefness = {
        "status": "rejected" if nef_witnesses else "certified",
        "nef": not nef_witnesses,
        "method": "exact_toric_support_function_convexity",
        "support_inequality_margins": support_inequality_margins,
        "witnesses": nef_witnesses,
    }
    fan_data = {
        **base,
        "quotient_annihilator": np.asarray(quotient_annihilator, dtype=int).tolist(),
        "quotient_rays": [list(ray) for ray in sorted(coefficients)],
        "quotient_maximal_cones": [
            [list(ray) for ray in record["rays"]] for record in maximal_records
        ],
        "restricted_anticanonical_coefficients": [
            {"ray": list(ray), "coefficient": _fraction_to_json(coefficients[ray])}
            for ray in sorted(coefficients)
        ],
    }
    if nef_witnesses:
        return {
            **fan_data,
            "status": "rejected",
            "reason_code": "restricted_line_bundle_not_nef",
            "reason": (
                "the restricted anti-canonical support function violates a "
                "nefness inequality on the quotient fan"
            ),
            "nefness": nefness,
            "orbifold_intersection": {
                "status": "not_evaluated",
                "reason_code": "restricted_line_bundle_not_nef",
                "reason": "orbifold avoidance is not evaluated after a nefness rejection",
            },
        }

    all_cones = {frozenset()}
    for record in maximal_records:
        rays = record["rays"]
        for size in range(1, component_dimension + 1):
            all_cones.update(frozenset(face) for face in itertools.combinations(rays, size))
    singular_cones = [
        tuple(sorted(cone))
        for cone in all_cones
        if cone and not _cone_is_smooth_in_lattice(tuple(cone), component_dimension)
    ]
    positive_dimensional_singular_cones = [
        cone for cone in singular_cones if component_dimension - len(cone) > 0
    ]
    if not positive_dimensional_singular_cones:
        return {
            **fan_data,
            "status": "certified",
            "reason_code": None,
            "reason": None,
            "nefness": nefness,
            "orbifold_intersection": {
                "status": "certified",
                "avoided": True,
                "method": "exact_toric_orbit_face_test",
                "singular_cones": [
                    [list(ray) for ray in cone] for cone in singular_cones
                ],
                "positive_dimensional_singular_cones": [],
                "reason": (
                    "all singular toric strata are zero-dimensional, so a "
                    "generic nonzero section avoids them"
                ),
            },
        }

    section_points = _toric_section_lattice_points(
        tuple(coefficients), coefficients, component_dimension
    )
    if section_points["status"] != "certified":
        return {
            **fan_data,
            "status": "unavailable",
            "reason_code": section_points["reason_code"],
            "reason": section_points["reason"],
            "nefness": nefness,
            "orbifold_intersection": {
                "status": "unavailable",
                "reason_code": section_points["reason_code"],
                "reason": section_points["reason"],
            },
        }
    all_section_points = section_points["points"]
    singular_strata = []
    for cone in positive_dimensional_singular_cones:
        face_points = tuple(
            point
            for point in all_section_points
            if all(
                _fraction_dot(point, ray) == -coefficients[ray]
                for ray in cone
            )
        )
        singular_strata.append(
            {
                "cone": [list(ray) for ray in cone],
                "orbit_dimension": component_dimension - len(cone),
                "face_lattice_point_count": len(face_points),
                "face_lattice_points": [
                    [_fraction_to_json(value) for value in point]
                    for point in face_points
                ],
            }
        )
        if len(face_points) != 1:
            return {
                **fan_data,
                "status": "rejected",
                "reason_code": "orbifold_stratum_intersection",
                "reason": (
                    "the restricted section polytope has "
                    f"{len(face_points)} lattice points on the positive-dimensional "
                    "orbifold stratum, so a generic section intersects it"
                ),
                "nefness": nefness,
                "orbifold_intersection": {
                    "status": "rejected",
                    "avoided": False,
                    "method": "exact_toric_orbit_face_test",
                    "singular_strata": singular_strata,
                },
            }
    return {
        **fan_data,
        "status": "certified",
        "reason_code": None,
        "reason": None,
        "nefness": nefness,
        "orbifold_intersection": {
            "status": "certified",
            "avoided": True,
            "method": "exact_toric_orbit_face_test",
            "singular_strata": singular_strata,
        },
    }


def _general_fixed_surface_n_s_table(
    triangulation_cones,
    triangulation,
    matrix,
    auxiliary_fan=None,
    *,
    fixed_cone_keys=None,
    return_diagnostics=False,
):
    """Compute eq. (4.50) evidence for general-``L`` fixed surfaces.

    For a two-dimensional component, the fixed surface is the toric orbit
    closure of ``sigma`` in the auxiliary fan ``Sigma_L``.  Use the toric
    Euler sequence and adjunction before evaluating the Chern-class integral:

    ``n_S = int_S(c_2(T_V)|_S - c_2(T_S) + c_1(T_S)^2)``.

    The identity case reduces to
    ``D_p D_q (K_V^-1-D_p)(K_V^-1-D_q)``.  Return no entry when the local
    fan, Cartier data, or lattice quotient cannot certify a smooth surface.
    Set ``return_diagnostics`` to return the evidence table together with a
    machine-readable record for every skipped two-dimensional fixed surface.
    """

    matrix = np.asarray(matrix, dtype=int)
    if auxiliary_fan is None:
        auxiliary_fan = build_auxiliary_fan(triangulation_cones, matrix)
    if fixed_cone_keys is None:
        fixed_cone_keys = _pointwise_invariant_cone_keys(triangulation_cones, matrix)
    source_sigma_keys = []
    for source_rays in fixed_cone_keys:
        normalized_rays = tuple(
            tuple(int(value) for value in ray) for ray in source_rays
        )
        if not all(
            np.array_equal(matrix @ np.asarray(ray, dtype=int), ray)
            for ray in normalized_rays
        ):
            continue
        source_sigma_keys.append(normalized_rays)
    tables = {}
    diagnostics = []
    global_reasons = []

    def finish():
        if return_diagnostics:
            return {
                "evidence": tables,
                "surface_diagnostics": diagnostics,
                "global_reasons": global_reasons,
            }
        return tables

    nu_records = None

    def record_surface_reason(
        sigma_rays,
        sigma_dimension,
        reason_code,
        reason,
    ):
        if not return_diagnostics:
            return
        if reason_code not in GENERAL_FIXED_SURFACE_REASON_CODES:
            raise ValueError(f"unknown general fixed-surface reason code: {reason_code}")
        fixed_components = []
        if nu_records is not None:
            fixed_components = [
                {
                    "sigma_rays": [list(ray) for ray in sigma_rays],
                    "nu": _fraction_vector_to_json(record["vector"]),
                }
                for record in nu_records
            ]
        diagnostics.append(
            {
                "status": "unavailable",
                "reason_code": reason_code,
                "reason": reason,
                "sigma_rays": [list(ray) for ray in sigma_rays],
                "sigma_dimension": int(sigma_dimension),
                "fixed_components": fixed_components,
            }
        )

    def record_surface_evidence(
        sigma_rays,
        sigma_dimension,
        c2_ambient_restricted,
        c2_surface,
        c1_surface_squared,
        n_s,
        diagnostic_data,
    ):
        if not return_diagnostics:
            return
        diagnostics.append(
            {
                "status": "certified",
                "reason_code": None,
                "reason": None,
                "sigma_rays": [list(ray) for ray in sigma_rays],
                "sigma_dimension": int(sigma_dimension),
                "fixed_components": [
                    {
                        "sigma_rays": [list(ray) for ray in sigma_rays],
                        "nu": _fraction_vector_to_json(record["vector"]),
                    }
                    for record in nu_records
                ],
                "c2_ambient_restricted": _fraction_to_json(c2_ambient_restricted),
                "c2_surface": _fraction_to_json(c2_surface),
                "c1_surface_squared": _fraction_to_json(c1_surface_squared),
                "n_s": _fraction_to_json(n_s),
                **diagnostic_data,
            }
        )

    fixed_rank = _exact_rank(np.eye(4, dtype=int) + matrix)
    if fixed_rank < 2:
        global_reasons.append(
            {
                "reason_code": "fixed_subspace_dimension_below_surface",
                "reason": "the invariant subspace has dimension below two",
            }
        )
        return finish()

    invariant_basis = _integer_kernel_basis(np.eye(4, dtype=int) - matrix)
    if invariant_basis.shape[1] != fixed_rank:
        global_reasons.append(
            {
                "reason_code": "invariant_basis_rank_mismatch",
                "reason": "the integer invariant basis rank does not match the fixed subspace rank",
            }
        )
        return finish()
    vectors = np.asarray(triangulation.fan().vectors(), dtype=int)
    if vectors.ndim != 2 or vectors.shape[1] != 4:
        global_reasons.append(
            {
                "reason_code": "invalid_ambient_ray_data",
                "reason": "the triangulation fan does not expose an integer (n, 4) ray array",
            }
        )
        return finish()

    nu_records = enumerate_projected_lattice_representatives(matrix, -1)

    ambient_cone_data = {}

    def local_cartier_data(ambient):
        ambient_key = tuple(tuple(int(value) for value in ray) for ray in ambient)
        if ambient_key not in ambient_cone_data:
            if len(ambient_key) != 4:
                ambient_cone_data[ambient_key] = None
            else:
                data = {
                    tuple(point): _ambient_cartier_data(ambient_key, point)
                    for point in vectors
                }
                ambient_cone_data[ambient_key] = (
                    data if all(value is not None for value in data.values()) else None
                )
        return ambient_cone_data[ambient_key]

    for sigma_rays in source_sigma_keys:
        sigma_dimension = _exact_rank(np.asarray(sigma_rays, dtype=int)) if sigma_rays else 0
        if fixed_rank - sigma_dimension != 2:
            continue
        if len(sigma_rays) != sigma_dimension:
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "nonsaturated_fixed_cone_lattice",
                "the auxiliary cone ray count does not match its lattice dimension",
            )
            continue
        sigma_coordinates = []
        for ray in sigma_rays:
            coordinates = _integer_coordinates(invariant_basis, ray)
            if coordinates is None:
                sigma_coordinates = []
                break
            sigma_coordinates.append(coordinates)
        if sigma_dimension and not sigma_coordinates:
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "nonsaturated_fixed_cone_lattice",
                "an auxiliary fixed-cone ray has no integral coordinate in the invariant lattice",
            )
            continue
        sigma_matrix = (
            np.column_stack(sigma_coordinates)
            if sigma_coordinates
            else np.empty((fixed_rank, 0), dtype=int)
        )
        if not _sublattice_is_saturated(sigma_matrix):
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "nonsaturated_fixed_cone_lattice",
                "the fixed-cone rays generate a proper finite-index sublattice of the invariant lattice",
            )
            continue
        annihilator = _integer_kernel_basis(sigma_matrix.T).T
        if annihilator.shape != (2, fixed_rank):
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "incomplete_quotient_surface_fan",
                "the quotient annihilator does not have the required two-dimensional rank",
            )
            continue

        surface_cones = {}
        boundary_rays = {}
        boundary_scales = {}
        local_ambient_cones = []
        full_dimensional_cone_count = 0
        failure_code = None
        failure_reason = None
        for full_cone in auxiliary_fan:
            full_rays = tuple(tuple(int(value) for value in ray) for ray in full_cone["rays"])
            if full_cone["dimension"] != fixed_rank or not set(sigma_rays).issubset(full_rays):
                continue
            full_dimensional_cone_count += 1
            if not full_cone["simplicial"] or len(full_rays) != fixed_rank:
                failure_code = "non_simplicial_full_dimensional_auxiliary_cone"
                failure_reason = "a full-dimensional auxiliary cone containing the fixed cone is non-simplicial"
                break
            ambient_cones = full_cone.get("ambient_cones", [])
            if not ambient_cones:
                failure_code = "missing_full_dimensional_auxiliary_cone"
                failure_reason = "the full-dimensional auxiliary cone has no ambient-cone provenance"
                break
            if any(
                len(ambient) != 4
                or abs(_exact_determinant(np.asarray(ambient, dtype=int))) != 1
                for ambient in ambient_cones
            ):
                failure_code = "non_smooth_ambient_cone"
                failure_reason = "a containing ambient cone is not a smooth unimodular four-cone"
                break
            for ambient in ambient_cones:
                ambient_key = tuple(
                    tuple(int(value) for value in ray) for ray in ambient
                )
                local_data = local_cartier_data(ambient_key)
                if local_data is None:
                    failure_code = "missing_restricted_cartier_data"
                    failure_reason = (
                        "an ambient provenance cone has no complete integral "
                        "Cartier data for the ambient divisors"
                    )
                    break
            if failure_code is not None:
                break
            quotient_rays = []
            for ray in full_rays:
                if ray in sigma_rays:
                    continue
                coordinates = _integer_coordinates(invariant_basis, ray)
                if coordinates is None:
                    failure_code = "incomplete_quotient_surface_fan"
                    failure_reason = "a full-dimensional auxiliary ray has no integral invariant-lattice coordinate"
                    break
                quotient = annihilator @ coordinates
                divisor = abs(math.gcd(*quotient.tolist()))
                if divisor == 0:
                    failure_code = "incomplete_quotient_surface_fan"
                    failure_reason = "a quotient ray vanishes in the two-dimensional quotient lattice"
                    break
                primitive = tuple(int(value // divisor) for value in quotient)
                if primitive in boundary_rays and (
                    boundary_rays[primitive] != ray
                    or boundary_scales[primitive] != divisor
                ):
                    failure_code = "incomplete_quotient_surface_fan"
                    failure_reason = (
                        "multiple ambient rays map to one quotient ray with "
                        "incompatible divisor provenance"
                    )
                    break
                quotient_rays.append(primitive)
                boundary_rays[primitive] = ray
                boundary_scales[primitive] = divisor
            if failure_code is not None:
                break
            if len(quotient_rays) != 2 or quotient_rays[0] == quotient_rays[1]:
                failure_code = "incomplete_quotient_surface_fan"
                failure_reason = "a full-dimensional auxiliary cone does not induce two distinct quotient rays"
                break
            cone_key = frozenset(quotient_rays)
            surface_cones.setdefault(cone_key, []).append(full_cone)
            local_ambient_cones.extend(full_cone["ambient_cones"])
        if failure_code is not None:
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                failure_code,
                failure_reason,
            )
            continue
        if not surface_cones:
            reason_code = (
                "missing_full_dimensional_auxiliary_cone"
                if full_dimensional_cone_count == 0
                else "incomplete_quotient_surface_fan"
            )
            reason = (
                "no full-dimensional auxiliary cone contains this fixed cone"
                if full_dimensional_cone_count == 0
                else "the containing full-dimensional cones do not form a quotient surface fan"
            )
            record_surface_reason(sigma_rays, sigma_dimension, reason_code, reason)
            continue

        if not _complete_simplicial_fan(
            [tuple(sorted(cone)) for cone in surface_cones],
            2,
        ):
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "incomplete_quotient_surface_fan",
                "the quotient surface fan failed the complete simplicial fan certificate",
            )
            continue

        boundary_keys = tuple(boundary_rays)
        if len(boundary_keys) < 3 or any(
            sum(ray in cone for cone in surface_cones) != 2 for ray in boundary_keys
        ):
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "incomplete_quotient_surface_fan",
                "the quotient surface fan is not complete around every boundary ray",
            )
            continue
        try:
            _surface_divisor_intersection(
                {ray: Fraction(1) for ray in boundary_keys},
                {ray: Fraction(1) for ray in boundary_keys},
                boundary_keys,
                surface_cones,
            )
        except ValueError as exc:
            message = str(exc)
            reason_code = (
                "non_smooth_surface_fan"
                if "non-smooth" in message
                else "incomplete_quotient_surface_fan"
            )
            record_surface_reason(sigma_rays, sigma_dimension, reason_code, message)
            continue

        if not local_ambient_cones:
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "missing_restricted_cartier_data",
                "the quotient surface fan has no ambient-cone provenance for restriction",
            )
            continue
        try:
            # The reference support function is the Cartier data on sigma.
            reference_ambient = None
            for ambient in local_ambient_cones:
                ambient_key = tuple(tuple(int(value) for value in ray) for ray in ambient)
                if local_cartier_data(ambient_key) is not None:
                    reference_ambient = ambient_key
                    break
            if reference_ambient is None:
                record_surface_reason(
                    sigma_rays,
                    sigma_dimension,
                    "missing_restricted_cartier_data",
                    "no containing ambient cone supplied integral Cartier support data for all divisors",
                )
                continue
            reference_data = ambient_cone_data[reference_ambient]
            divisor_coefficients = {}
            for divisor_point in vectors:
                divisor_key = tuple(int(value) for value in divisor_point)
                reference = reference_data[divisor_key]
                coefficients = {}
                for cone_key, full_cones in surface_cones.items():
                    for full_cone in full_cones:
                        for ambient in full_cone.get("ambient_cones", []):
                            ambient_key = tuple(
                                tuple(int(value) for value in ray)
                                for ray in ambient
                            )
                            local_data = local_cartier_data(ambient_key)
                            if local_data is None:
                                raise ValueError("missing ambient Cartier data")
                            difference = [
                                value - ref
                                for value, ref in zip(
                                    local_data[divisor_key], reference
                                )
                            ]
                            for quotient_ray in cone_key:
                                ambient_ray = boundary_rays[quotient_ray]
                                coefficient = -sum(
                                    value * int(component)
                                    for value, component in zip(
                                        difference, ambient_ray
                                    )
                                ) / boundary_scales[quotient_ray]
                                if coefficient.q != 1:
                                    raise ValueError(
                                        "non-integral restricted Cartier coefficient"
                                    )
                                coefficient = Fraction(int(coefficient))
                                previous = coefficients.get(quotient_ray)
                                if previous is not None and previous != coefficient:
                                    raise ValueError(
                                        "inconsistent restricted Cartier data"
                                    )
                                coefficients[quotient_ray] = coefficient
                if set(coefficients) != set(boundary_keys):
                    raise ValueError("missing restricted Cartier data")
                divisor_coefficients[divisor_key] = coefficients

            unit_coefficients = {ray: Fraction(1) for ray in boundary_keys}
            c2_surface = Fraction(len(surface_cones))
            c1_surface_squared = _surface_divisor_intersection(
                unit_coefficients, unit_coefficients, boundary_keys, surface_cones
            )
            c2_ambient_restricted = Fraction(0)
            divisor_keys = [tuple(int(value) for value in point) for point in vectors]
            for left_index, left_key in enumerate(divisor_keys):
                for right_key in divisor_keys[left_index + 1 :]:
                    c2_ambient_restricted += _surface_divisor_intersection(
                        divisor_coefficients[left_key],
                        divisor_coefficients[right_key],
                        boundary_keys,
                        surface_cones,
                    )
            n_s = c2_ambient_restricted - c2_surface + c1_surface_squared
            if n_s.denominator != 1:
                record_surface_reason(
                    sigma_rays,
                    sigma_dimension,
                    "nonintegral_final_n_s",
                    f"the final fixed-surface n_S is nonintegral: {n_s}",
                )
                continue
        except ValueError as exc:
            message = str(exc)
            if "non-integral restricted Cartier" in message:
                reason_code = "nonintegral_restricted_cartier_data"
            elif "inconsistent restricted Cartier" in message:
                reason_code = "inconsistent_restricted_cartier_data"
            elif "missing" in message or "incomplete" in message:
                reason_code = "missing_restricted_cartier_data"
            elif "non-smooth" in message:
                reason_code = "non_smooth_surface_fan"
            else:
                reason_code = "missing_restricted_cartier_data"
            record_surface_reason(sigma_rays, sigma_dimension, reason_code, message)
            continue
        except (TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
            record_surface_reason(
                sigma_rays,
                sigma_dimension,
                "missing_restricted_cartier_data",
                f"restricted Cartier calculation could not be completed: {exc}",
            )
            continue

        ordered_boundary_keys = tuple(sorted(boundary_keys))
        ordered_surface_cones = tuple(
            sorted(surface_cones, key=lambda cone: tuple(sorted(cone)))
        )
        surface_cone_provenance = []
        for cone_key in ordered_surface_cones:
            ambient_cones = {
                tuple(
                    tuple(int(value) for value in ray)
                    for ray in ambient
                )
                for full_cone in surface_cones[cone_key]
                for ambient in full_cone.get("ambient_cones", [])
            }
            surface_cone_provenance.append(
                {
                    "quotient_cone": [list(ray) for ray in sorted(cone_key)],
                    "ambient_cones": [
                        [list(ray) for ray in sorted(ambient)]
                        for ambient in sorted(ambient_cones)
                    ],
                }
            )
        diagnostic_data = {
            "invariant_lattice_basis": np.asarray(
                invariant_basis, dtype=int
            ).tolist(),
            "quotient_annihilator": np.asarray(annihilator, dtype=int).tolist(),
            "quotient_surface_rays": [
                list(ray) for ray in ordered_boundary_keys
            ],
            "quotient_surface_cones": [
                [list(ray) for ray in sorted(cone)]
                for cone in ordered_surface_cones
            ],
            "surface_cone_provenance": surface_cone_provenance,
            "boundary_ray_scales": [
                {
                    "quotient_ray": list(ray),
                    "scale": int(boundary_scales[ray]),
                }
                for ray in ordered_boundary_keys
            ],
            "reference_ambient_cone": [
                list(ray) for ray in sorted(reference_ambient)
            ],
            "restricted_divisor_coefficients": [
                {
                    "ambient_ray": list(divisor_key),
                    "coefficients": [
                        {
                            "quotient_ray": list(ray),
                            "coefficient": _fraction_to_json(
                                divisor_coefficients[divisor_key][ray]
                            ),
                        }
                        for ray in ordered_boundary_keys
                    ],
                }
                for divisor_key in sorted(divisor_coefficients)
            ],
        }
        record_surface_evidence(
            sigma_rays,
            sigma_dimension,
            c2_ambient_restricted,
            c2_surface,
            c1_surface_squared,
            n_s,
            diagnostic_data,
        )

        # Every admissible fixed surface is keyed with nu below. The value is
        # independent of nu because nu translates the toric component without
        # changing its fan or restricted line bundles.
        for nu_record in nu_records:
            tables[
                _component_key(
                    {
                        "sigma_rays": [list(ray) for ray in sigma_rays],
                        "nu": _fraction_vector_to_json(nu_record["vector"]),
                    }
                )
            ] = int(n_s)

    return finish()


def identity_fixed_surface_n_s_table(triangulation_cones, triangulation):
    """Populate ``n^S_{df=0}`` evidence for ``L=identity``'s 2-dimensional
    fixed components, keyed for ``classify_smoothness``'s
    ``topology["fixed_surface_n_s"]`` lookup (via ``_component_key``).

    Restricted to ``L=identity``: its ``nu`` coset ``H_-^L`` is always
    trivial (``P_-^{id}(N)`` is the zero space, since the identity has no
    ``-1`` eigenspace -- confirmed empirically, not just asserted), and its
    auxiliary fan ``Sigma_L`` reduces exactly to the ambient fan itself
    (confirmed empirically), so every 2-dimensional fixed component is a
    2-cone ``(p, q)`` of the ambient fan directly, matching Moritz eq.
    around line 572-574 (``t + (1/2) sum(sigma(1)) in N``) exactly -- the
    same case ``reproduce_fuzzy_axions_h11_4.py``'s
    ``_frozen_conifold_diagnostic`` already validates end to end against
    the paper's own Table 1 numbers, just for one specific shift
    (the trilayer's ``t=p0/2``) instead of every shift. Not extended to
    ``L != identity``: there, a fixed component's own toric structure is
    generally more involved than a direct 2-divisor intersection (Sec. 4.4
    of arXiv:2305.06363), and applying this same formula there has not been
    independently derived or verified.

    ``(p, q)`` being simplicial does *not* mean ``S = D_p.D_q`` is smooth --
    see ``_cone_has_smooth_star``. Source line 656-657 separately requires
    ``S`` itself to be smooth (on top of ``n_S=0``); when it isn't, no table
    entry is written, so ``classify_smoothness``'s existing
    ``_lookup_surface_n_s -> None`` path reports
    ``"eq. (4.50) requires fixed-surface n_S evidence"`` in
    ``unavailable_reasons`` rather than trusting a computed ``n_S`` whose
    own precondition (a smooth complete intersection with a split normal
    bundle) does not hold.
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
        if not _cone_has_smooth_star(fan, cone["rays"]):
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
    # L=I, t=0 is only a valid (trivially smooth) orientifold action for
    # lambda_f=0 (worldsheet-parity-only, O9-filling-everything). For
    # lambda_f=1, source eq. (4.43) with t=0 forces every monomial
    # coefficient psi_q = -psi_q, i.e. psi_q=0 for all q -- there is no
    # hypersurface to speak of, and eq. (4.45) (line ~627) correctly reports
    # a violation at every L-fixed dual vertex (all of them, since L=I).
    # This branch must not paper over that with the same unconditional
    # "smooth" verdict used for lambda_f=0.
    is_identity_sanity = (
        np.array_equal(matrix, IDENTITY)
        and all(value == 0 for value in torus_shift)
        and int(lambda_f) == 0
    )
    extra_vertices = topology.get(
        "non_smooth_facet_dual_vertices",
        _EXTRA_VERTEX_EVIDENCE_NOT_REQUESTED,
    )
    parity = _dual_vertex_parity_evidence(
        matrix,
        torus_shift,
        lambda_f,
        dual_vertices,
        extra_vertices=extra_vertices,
    )
    if is_identity_sanity:
        return {
            "status": "smooth",
            "verdict": "smooth",
            "method": "identity_sanity_contract",
            "reason": "explicit task-level identity fixture contract",
            "dual_vertex_parity": parity,
            "positive_component_checks": [],
        }

    non_smooth_reasons = []
    unavailable_reasons = []
    positive_component_checks = []
    if not parity["available"]:
        unavailable_reasons.append(
            "source eq. (4.45) dual-vertex parity evidence is unavailable"
        )
    elif parity["violations"]:
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
            if component["fixed_toric_dimension"] > 0:
                section_check = _positive_component_section_certificate(
                    auxiliary_fan,
                    matrix,
                    component,
                )
                section_check["component"] = {
                    "sigma_rays": component["sigma_rays"],
                    "sigma_dimension": component["sigma_dimension"],
                    "nu": component["nu"],
                }
                positive_component_checks.append(section_check)
                if section_check["status"] == "rejected":
                    non_smooth_reasons.append(
                        "source Sec. 4.6 rejects positive-dimensional "
                        "non-vanishing fixed component "
                        f"[{section_check['reason_code']}]: {section_check['reason']}"
                    )
                elif section_check["status"] == "unavailable":
                    unavailable_reasons.append(
                        "source Sec. 4.6 requires nef generic-section and "
                        "orbifold-avoidance evidence for a positive-dimensional "
                        "non-vanishing fixed component "
                        f"[{section_check['reason_code']}]: {section_check['reason']}"
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
            "method": "source_eq_4.45_4.48_4.50_sec_4.6_checks",
            "reasons": non_smooth_reasons,
            "dual_vertex_parity": parity,
            "positive_component_checks": positive_component_checks,
        }
    if unavailable_reasons:
        return {
            "status": "smoothness_verification_unavailable",
            "verdict": "not_verified",
            "method": "source_eq_4.45_4.48_4.50_sec_4.6_checks",
            "reasons": sorted(set(unavailable_reasons)),
            "dual_vertex_parity": parity,
            "positive_component_checks": positive_component_checks,
        }
    return {
        "status": "smooth",
        "verdict": "smooth",
        "method": "source_eq_4.45_4.48_4.50_sec_4.6_checks",
        "reasons": [],
        "dual_vertex_parity": parity,
        "positive_component_checks": positive_component_checks,
    }


def _fixed_point_set_description(matrix, torus_shift, fixed_components, smoothness):
    """Describe the fixed-point set; the whole-CY label must agree with
    ``classify_smoothness``'s own verdict rather than re-deriving it, so the
    two cannot fall out of sync (see ``classify_smoothness``'s lambda_f=0
    restriction on the identity/zero-shift case)."""
    if (
        np.array_equal(matrix, IDENTITY)
        and all(value == 0 for value in torus_shift)
        and smoothness["status"] == "smooth"
    ):
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


def _base_record(polytope_id, polytope_normal_form_id, frst_hash, matrix, candidate_id):
    return {
        "candidate_id": candidate_id,
        "matrix_id": candidate_id,
        "polytope_id": polytope_id,
        "polytope_normal_form_id": polytope_normal_form_id,
        "frst_hash": frst_hash,
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "record_kind": "matrix_validation",
        "lattice_matrix": np.asarray(matrix, dtype=int).tolist(),
        "involution_type": None,
        "lambda_f": None,
        "torus_shift": None,
        "fixed_point_set": None,
    }


def _not_evaluated_fixed_component_evidence(reason):
    """Record an explicit proof boundary before fixed components are evaluated."""

    return {
        "status": "not_evaluated",
        "reason": str(reason),
    }


def enumerate_orientifold_candidates(
    poly,
    triangulation,
    topology,
    *,
    h11_minus_target=None,
    dual_polytope=None,
    general_fixed_surface_diagnostics=None,
    record_sink=None,
):
    """Enumerate every matrix, shift, and coefficient-parity candidate."""
    points = np.asarray(poly.points(), dtype=int)
    polytope_id = compute_polytope_id(points)
    # Geometry-unique identity: the affine normal form is invariant under
    # GL(4, Z) and translation, so it distinguishes lattice-inequivalent
    # polytopes that share the same index-combinatorial triangulation hash.
    normal_form_points = np.asarray(poly.normal_form(), dtype=int)
    polytope_normal_form_id = compute_polytope_normal_form_id(normal_form_points)
    simplices = np.asarray(triangulation.simplices(), dtype=int)
    frst_hash = compute_triangulation_hash(simplices)
    triangulation_cones = _triangulation_cones(poly, triangulation)
    dual_vertices = _extract_dual_vertices(poly, dual_polytope)
    class _RecordCollection(list):
        def append(self, record):
            if (
                record_sink is not None
                and record.get("record_kind") != "lattice_matrix_search_summary"
            ):
                record_sink(dict(record))
            super().append(record)

    records = _RecordCollection()

    for matrix in enumerate_polytope_involutions(points):
        fixed_cone_keys = _pointwise_invariant_cone_keys(triangulation_cones, matrix)
        matrix_tuple = tuple(int(value) for value in matrix.flatten())
        matrix_id = stable_hash([polytope_id, frst_hash, matrix_tuple])
        base = _base_record(
            polytope_id, polytope_normal_form_id, frst_hash, matrix, matrix_id
        )
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
                    "terminal_reason_code": exc.stage or "numerical_geometry_failure",
                    "torus_shift_search_status": "not_started",
                    "h11_parity": {
                        "status": "unavailable",
                        "reason": "matrix validation did not produce an H2 parity decomposition",
                    },
                    "fixed_component_evidence": _not_evaluated_fixed_component_evidence(
                        "matrix validation failed before fixed-component evaluation"
                    ),
                }
            )
            records.append(record)
            continue

        if record_sink is not None:
            matrix_success = dict(base)
            matrix_success.update(
                {
                    "record_kind": "matrix_validation",
                    "matrix_id": matrix_id,
                    "terminal_status": "matrix_validation_passed",
                    "terminal_reason": "polytope, FRST, divisor, and H2 validation passed",
                    "terminal_reason_code": "matrix_validation_passed",
                    "h11_plus": validated["h11_plus"],
                    "h11_minus": validated["h11_minus"],
                    "fixed_component_evidence": _not_evaluated_fixed_component_evidence(
                        "matrix validation passed; no torus-shift fixed component was evaluated"
                    ),
                    "h2_involution_matrix": _to_jsonable(
                        validated["h2_involution_matrix"]
                    ),
                }
            )
            record_sink(matrix_success)

        auxiliary_fan = build_auxiliary_fan(triangulation_cones, matrix)
        fixed_surface_n_s = topology.get("fixed_surface_n_s", {})
        if topology.get("compute_general_fixed_surface_n_s") and not np.array_equal(
            matrix, IDENTITY
        ):
            n_s_result = _general_fixed_surface_n_s_table(
                triangulation_cones,
                triangulation,
                matrix,
                auxiliary_fan,
                fixed_cone_keys=fixed_cone_keys,
                return_diagnostics=general_fixed_surface_diagnostics is not None,
            )
            if general_fixed_surface_diagnostics is not None:
                fixed_surface_n_s = n_s_result["evidence"]
                general_fixed_surface_diagnostics[matrix_id] = {
                    "matrix_id": matrix_id,
                    "lattice_matrix": matrix.tolist(),
                    "surface_diagnostics": n_s_result["surface_diagnostics"],
                    "global_reasons": n_s_result["global_reasons"],
                }
            else:
                fixed_surface_n_s = n_s_result
        matrix_topology = dict(topology)
        matrix_topology["fixed_surface_n_s"] = fixed_surface_n_s
        shifts = enumerate_projected_lattice_representatives(matrix, +1)
        if not shifts:
            record = dict(base)
            record.update(
                {
                    "matrix_id": matrix_id,
                    "record_kind": "matrix_validation",
                    "terminal_status": "torus_shift_search_exhausted",
                    "terminal_reason": "no representatives were generated",
                    "terminal_reason_code": "torus_shift_search_exhausted",
                    "torus_shift_search_status": "exhausted",
                    "h11_plus": validated["h11_plus"],
                    "h11_minus": validated["h11_minus"],
                    "fixed_component_evidence": _not_evaluated_fixed_component_evidence(
                        "no torus-shift representative was generated"
                    ),
                }
            )
            records.append(record)
            continue

        matrix_records = []
        for shift in shifts:
            # ``shift["vector"]`` is a representative of source eq. (4.34)'s
            # H_+^L = P_+^L(N)/(2 P_+^L(N)), whose elements are explicitly
            # labelled "[2t]" there -- it IS 2t, not t.
            two_t = shift["vector"]
            # Source eq. (4.34) derivation (KS_orientifolds.tex line ~426-428):
            # a Z2 involution ``L o phi_[t]`` squares to ``phi_[2t]``, so it is
            # an involution *only if* ``2t in N``. For a non-identity ``L``,
            # ``enumerate_projected_lattice_representatives`` returns cosets of
            # ``H_+^L`` whose representative ``2t`` need not be integral (e.g.
            # ``2t=(1/2,1/2,0,0)`` for a coordinate-swap ``L``); those are not
            # Z2 involutions at all and must be excluded here, before any
            # smoothness/parity evaluation -- a non-integral ``2t`` also makes
            # eq. (4.45)'s ``<2t,q>`` parity ill-defined. Identity ``L`` never
            # triggers this (its ``2t`` is always integral), which is exactly
            # why identity-only validation never surfaced it.
            if any(Fraction(value).denominator != 1 for value in two_t):
                record = dict(base)
                shift_candidate_id = stable_hash(
                    [matrix_id, tuple(shift["numerator"]), "noninvolution"]
                )
                record.update(
                    {
                        "record_kind": "candidate",
                        "matrix_id": matrix_id,
                        "candidate_id": shift_candidate_id,
                        "attempt_kind": "torus_shift_rejected_noninvolution",
                        "matrix_candidate_id": matrix_id,
                        "terminal_status": "torus_shift_not_involution",
                        "terminal_reason": (
                            "source eq. (4.34)/line ~428 requires 2t in N for "
                            "L o phi_[t] to be an involution; this H_+^L coset "
                            "representative 2t is not integral"
                        ),
                        "terminal_reason_code": "torus_shift_not_involution",
                        "torus_shift_binary_source": shift["binary_source"],
                        "torus_shift_search_status": "rejected_noninvolution",
                        "h11_plus": validated["h11_plus"],
                        "h11_minus": validated["h11_minus"],
                        "fixed_component_evidence": _not_evaluated_fixed_component_evidence(
                            "torus shift was rejected before fixed-component evaluation"
                        ),
                    }
                )
                records.append(record)
                matrix_records.append(record)
                continue
            # Downstream consumers (eq. 4.45's dual-vertex parity, eq. 4.33's
            # fixed-component integrality condition) use ``t`` directly, so the
            # integral ``2t`` is halved once more here. Confirmed empirically
            # for L=identity against CYTools' own poly.inequivalent_Z2_actions()
            # (t in {0, 1/2}^4, not {0, 1}^4 as enumerate_projected_lattice_
            # representatives' "vector" gives before this correction).
            torus_shift = tuple(value / 2 for value in shift["vector"])
            for lambda_f in (0, 1):
                involution_type = "O3/O7" if lambda_f == 1 else "O5/O9"
                candidate_id = stable_hash(
                    [matrix_id, tuple(shift["numerator"]), int(lambda_f)]
                )
                record = _base_record(
                    polytope_id,
                    polytope_normal_form_id,
                    frst_hash,
                    matrix,
                    candidate_id,
                )
                record.update(
                    {
                        "record_kind": "candidate",
                        "matrix_id": matrix_id,
                        "matrix_candidate_id": matrix_id,
                        "involution_type": involution_type,
                        "lambda_f": int(lambda_f),
                        "lambda_f_convention": (
                            "lambda_f=1 gives O3/O7 (I*[Omega]=-Omega); "
                            "lambda_f=0 gives O5/O9 (I*[Omega]=+Omega)"
                        ),
                        "torus_shift": _fraction_vector_to_json(torus_shift),
                        "torus_shift_binary_source": shift["binary_source"],
                        "h11_plus": int(validated["h11_plus"]),
                        "h11_minus": int(validated["h11_minus"]),
                    }
                )
                try:
                    fixed_components = _fixed_component_records(
                        auxiliary_fan,
                        matrix,
                        torus_shift,
                        lambda_f,
                        fixed_cone_keys=fixed_cone_keys,
                    )
                    smoothness = classify_smoothness(
                        matrix,
                        torus_shift,
                        lambda_f,
                        auxiliary_fan,
                        fixed_components,
                        matrix_topology,
                        dual_vertices,
                    )
                except Exception as exc:
                    record.update(
                        {
                            "terminal_status": "numerical_geometry_failure",
                            "terminal_reason": (
                                "fixed-component construction or smoothness evaluation "
                                f"failed: {type(exc).__name__}: {exc}"
                            ),
                            "terminal_reason_code": "candidate_geometry_evaluation_failure",
                            "fixed_component_evidence": {
                                "status": "unavailable",
                                "reason": (
                                    "fixed-component construction or smoothness evaluation "
                                    f"failed: {type(exc).__name__}: {exc}"
                                ),
                            },
                        }
                    )
                    records.append(record)
                    matrix_records.append(record)
                    continue
                record.update(
                    {
                        "auxiliary_fan": auxiliary_fan,
                        "pointwise_invariant_cones": [
                            cone
                            for cone in auxiliary_fan
                            if cone["pointwise_L_invariant"]
                        ],
                        "fixed_point_components": fixed_components,
                        "fixed_surface_n_s_evidence": fixed_surface_n_s,
                        "fixed_point_set": _fixed_point_set_description(
                            matrix, torus_shift, fixed_components, smoothness
                        ),
                        "smoothness": smoothness,
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
                    record["terminal_reason_code"] = (
                        smoothness.get("reason_code")
                        or smoothness["status"]
                    )
                elif h11_minus_target is not None and int(validated["h11_minus"]) != int(
                    h11_minus_target
                ):
                    record["terminal_status"] = "orientifold_h11_minus_filter_rejection"
                    record["terminal_reason"] = (
                        f"h11_minus={validated['h11_minus']} does not match requested "
                        f"target {h11_minus_target}"
                    )
                    record["terminal_reason_code"] = "orientifold_h11_minus_filter_rejection"
                else:
                    record["terminal_status"] = "accepted_verified_orientifold"
                    record["terminal_reason"] = None
                    record["terminal_reason_code"] = "accepted_verified_orientifold"
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
