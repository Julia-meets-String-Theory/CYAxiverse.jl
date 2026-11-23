"""Source-validated exact general-L fixed-component geometry.

This is the reviewed dependency closure ported from sibling commit
a2018d98bd558ff9903c2296d7e8e122c84c1434. It implements Moritz
arXiv:2305.06363v1 Secs. 4.4--4.6 with exact arithmetic.
"""
from fractions import Fraction
import copy
import hashlib
import itertools
import json
import math

import numpy as np
try:
    from scipy.optimize import linprog as _linprog
except ImportError:  # pragma: no cover
    _linprog = None
from sympy import Matrix as _SympyMatrix
from sympy.matrices.normalforms import hermite_normal_form as _hermite_normal_form
from sympy.polys.matrices import DomainMatrix as _DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp as _smith_normal_decomp

IDENTITY = np.eye(4, dtype=int)
_EXTRA_VERTEX_EVIDENCE_NOT_REQUESTED = object()
_BASE_HNF_CACHE = {}
POSITIVE_COMPONENT_MAX_DIMENSION = 3
MAX_SECTION_LATTICE_POINTS = 100_000
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


def _integer_lattice_index(generators):
    """Return the exact index of a ray sublattice in its saturated span."""

    array = np.asarray(generators, dtype=int)
    if array.ndim != 2:
        raise ValueError("lattice generators must be a two-dimensional array")
    if array.shape[1] == 0:
        return 1
    rank = _exact_rank(array)
    if rank != array.shape[1]:
        return 0
    minors = []
    for rows in itertools.combinations(range(array.shape[0]), array.shape[1]):
        minors.append(_exact_determinant(array[np.ix_(rows, range(array.shape[1]))]))
    return abs(math.gcd(*minors))

def _exact_rank(matrix):
    """Return the rank of an integer matrix without floating-point rounding.

    Exact rational Gaussian elimination over ``Fraction`` on the small integer
    matrices this module works with (at most a handful of rows in ``Z^4``).
    This is bit-for-bit identical to the previous ``sympy.Matrix.rank()`` but
    ~11x faster; it is a dominant hot spot in the fixed-component search (see
    validation/2026-08_orientifold_performance_review.md, finding F0).
    """

    array = np.asarray(matrix, dtype=int)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rows, columns = array.shape
    if rows == 0 or columns == 0:
        return 0
    reduced = [[Fraction(int(value)) for value in row] for row in array.tolist()]
    rank = 0
    pivot_row = 0
    for column in range(columns):
        pivot = next(
            (index for index in range(pivot_row, rows) if reduced[index][column] != 0),
            None,
        )
        if pivot is None:
            continue
        reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
        pivot_value = reduced[pivot_row][column]
        for index in range(rows):
            if index == pivot_row:
                continue
            factor = reduced[index][column] / pivot_value
            if factor == 0:
                continue
            for target in range(column, columns):
                reduced[index][target] -= factor * reduced[pivot_row][target]
        pivot_row += 1
        rank += 1
        if pivot_row == rows:
            break
    return rank

def _exact_determinant(matrix):
    """Return the exact determinant of an integer matrix.

    Fraction-free Bareiss elimination in pure Python ints: exact, no
    floating-point rounding, and ~54x faster than the previous
    ``sympy.Matrix.det()`` on the 4x4-and-smaller integer matrices this module
    evaluates in its tightest loops (see
    validation/2026-08_orientifold_performance_review.md, finding F0). The
    result is bit-for-bit identical to the sympy determinant.
    """

    array = np.asarray(matrix, dtype=int)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    entries = [[int(value) for value in row] for row in array.tolist()]
    size = len(entries)
    if size == 0:
        return 1
    sign = 1
    previous_pivot = 1
    for step in range(size - 1):
        if entries[step][step] == 0:
            swap = next(
                (index for index in range(step + 1, size) if entries[index][step] != 0),
                None,
            )
            if swap is None:
                return 0
            entries[step], entries[swap] = entries[swap], entries[step]
            sign = -sign
        pivot = entries[step][step]
        for index in range(step + 1, size):
            row_head = entries[index][step]
            for column in range(step + 1, size):
                entries[index][column] = (
                    entries[index][column] * pivot - row_head * entries[step][column]
                ) // previous_pivot
        previous_pivot = pivot
    return sign * entries[size - 1][size - 1]

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
    # The base HNF depends only on ``generator_columns`` (fixed across a whole
    # projected-lattice enumeration), so cache it and recompute only the
    # augmented HNF per target (finding F6). Exact; only avoids recomputation.
    generator_key = tuple(
        tuple(int(value) for value in row)
        for row in np.asarray(generator_columns, dtype=int).tolist()
    )
    cached = _BASE_HNF_CACHE.get(generator_key)
    if cached is None:
        generator_matrix = _SympyMatrix([list(row) for row in generator_key])
        cached = (_hermite_normal_form(generator_matrix), generator_matrix)
        _BASE_HNF_CACHE[generator_key] = cached
    base_hnf, generator_matrix = cached
    target_vector = _SympyMatrix(np.asarray(target, dtype=int).reshape(-1, 1).tolist())
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

def _exact_fraction_or_none(value):
    """Return ``Fraction(value)`` or None for booleans and non-numeric input."""
    if isinstance(value, bool):
        return None
    try:
        return Fraction(value)
    except (TypeError, ValueError):
        return None


def _exact_integer_matrix_4x4(matrix):
    """Return a 4x4 matrix as exact integer tuples, or None if any entry is inexact."""
    try:
        rows = [list(row) for row in matrix]
    except TypeError:
        return None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return None
    exact = []
    for row in rows:
        exact_row = []
        for value in row:
            fraction = _exact_fraction_or_none(value)
            if fraction is None or fraction.denominator != 1:
                return None
            exact_row.append(int(fraction))
        exact.append(tuple(exact_row))
    return tuple(exact)


def _support_chart_certified(support, sigma_rays):
    """True when the support and its Cox chart are certified for this exact cone."""
    if not isinstance(support, dict) or support.get("status") != "certified":
        return False
    chart = support.get("chart")
    if not isinstance(chart, dict) or chart.get("status") != "certified":
        return False
    if chart.get("irrelevant_ideal_avoided") is not True:
        return False
    sigma = sorted([int(value) for value in ray] for ray in sigma_rays)
    return chart.get("chart_cone") == sigma


def _dense_torus_inversion_cancellation_applies(
    matrix,
    torus_shift,
    lambda_f,
    sigma_rays,
    ambient_dimension,
    support,
    *,
    dual_points_complete,
):
    """Return True only for the proved issue #236 dense-torus inversion case.

    For L = -I, t = 0 and lambda_f = 1, Moritz eq. (4.42) forces psi_0 = 0
    and makes every swapped pair restrict as psi_q (eps^q - eps^-q) = 0 at
    each dense-torus sign point eps, so the restriction vanishes identically.
    ``dual_points_complete=True`` is the producer's attestation that
    ``dual_points`` is an exact enumeration of every lattice point of the
    dual polytope for this fan; internally consistent truncated input cannot
    prove its own completeness.  Owner decision D1(b), issue #236.
    """
    if dual_points_complete is not True:
        return False
    if len(sigma_rays) != 0 or ambient_dimension != 0:
        return False
    if _exact_fraction_or_none(lambda_f) != 1:
        return False
    minus_identity = tuple(
        tuple(-1 if row == column else 0 for column in range(4)) for row in range(4)
    )
    if _exact_integer_matrix_4x4(matrix) != minus_identity:
        return False
    try:
        shift = [_exact_fraction_or_none(value) for value in torus_shift]
    except TypeError:
        return False
    if len(shift) != 4 or any(value != 0 for value in shift):
        return False
    return _support_chart_certified(support, sigma_rays)


def _singleton_nonvanishing_witness(matrix, support, sigma_rays):
    """Return a fixed-singleton witness of a nonzero restriction, or None.

    Analyst A-6 (issue #236): if one allowed L-fixed dual point q has a
    monomial that is nonzero on the component chart, the covariant section
    with psi_q = 1 and every other independent orbit coefficient zero has a
    nonzero restriction, so the generic restriction is not identically zero.
    No complete dual-point enumeration is required for this direction.
    """
    exact = _exact_integer_matrix_4x4(matrix)
    if exact is None or not _support_chart_certified(support, sigma_rays):
        return None
    sigma = {tuple(int(value) for value in ray) for ray in sigma_rays}
    try:
        ambient = [tuple(int(value) for value in ray) for ray in support.get("ambient_rays") or ()]
    except (TypeError, ValueError):
        return None
    for row in support.get("support") or ():
        try:
            q = tuple(int(value) for value in row["q"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(q) != 4:
            continue
        image = tuple(sum(exact[k][i] * q[k] for k in range(4)) for i in range(4))
        if image != q:
            continue
        if row.get("orbit_q") != [list(q)] or row.get("image_q") != list(q):
            continue
        if row.get("chart_nonzero") is not True:
            continue
        phase = row.get("phase_exponent")
        if isinstance(phase, bool) or not isinstance(phase, int):
            continue
        exponents = row.get("cox_exponents")
        if not isinstance(exponents, list) or len(exponents) != len(ambient):
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != sum(a * b for a, b in zip(ray, q)) + 1
            or value < 0
            or (ray in sigma and value != 0)
            for value, ray in zip(exponents, ambient)
        ):
            continue
        return {"q": list(q), "orbit_q": [list(q)], "phase_exponent": phase}
    return None


_RESTRICTION_FAILURE_ORDER = (
    "action_not_validated",
    "inexact_input",
    "component_label_unbound",
    "chart_unavailable",
    "dual_rows_invalid",
    "frame_unavailable",
    "pair_phase_inconsistent",
    "column_uniformity_violated",
    "dual_points_incomplete",
    "certificate_conflict",
)


def _matrix_input_is_exact(value):
    """Check the numeric leaves accepted by the exact matrix decision API."""
    if value is None or isinstance(value, (str, bytes)):
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer, Fraction)):
        return True
    if isinstance(value, np.ndarray):
        return _matrix_input_is_exact(value.tolist())
    if isinstance(value, dict):
        return all(_matrix_input_is_exact(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_matrix_input_is_exact(item) for item in value)
    return False


def _matrix_validation_fraction(value):
    """Convert an input for ordered domain diagnostics, including a float.

    Floats are used only to check an earlier action condition when condition 2
    has already recorded ``inexact_input``. They never enter a column or a
    returned decision.
    """
    if isinstance(value, (bool, np.bool_)):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return Fraction(float(value))
    return None


def _matrix_fraction(value):
    """Decode one exact Python/NumPy integer or rational value."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, dict):
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        if (
            isinstance(numerator, (int, np.integer))
            and not isinstance(numerator, (bool, np.bool_))
            and isinstance(denominator, (int, np.integer))
            and not isinstance(denominator, (bool, np.bool_))
            and int(denominator) != 0
        ):
            return Fraction(int(numerator), int(denominator))
    return None


def _matrix_fraction_vector(value):
    """Decode a four-vector supplied as exact scalars or the JSON vector form."""
    if isinstance(value, dict) and "numerator" in value and "denominator" in value:
        numerators = value.get("numerator")
        denominator = value.get("denominator")
        if not isinstance(numerators, (list, tuple)):
            return None
        denominator = _matrix_fraction(denominator)
        if denominator is None or denominator.denominator != 1 or denominator == 0:
            return None
        values = [_matrix_fraction(item) for item in numerators]
        if any(item is None for item in values):
            return None
        return tuple(item / denominator for item in values)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        values = tuple(_matrix_fraction(item) for item in value)
    except TypeError:
        return None
    if any(item is None for item in values):
        return None
    return values


def _matrix_int_vector(value, *, diagnostic=False):
    values = _matrix_fraction_vector(value)
    if values is None and diagnostic:
        try:
            values = tuple(_matrix_validation_fraction(item) for item in value)
        except TypeError:
            return None
    if values is None or len(values) != 4:
        return None
    if any(item is None or item.denominator != 1 for item in values):
        return None
    return tuple(int(item) for item in values)


def _matrix_int_rows(value, *, diagnostic=False):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        rows = tuple(_matrix_int_vector(row, diagnostic=diagnostic) for row in value)
    except TypeError:
        return None
    if any(row is None for row in rows):
        return None
    return rows


def _matrix_cone_rows(value, *, diagnostic=False):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        cones = tuple(
            _matrix_int_rows(cone, diagnostic=diagnostic)
            for cone in value
        )
    except TypeError:
        return None
    if any(cone is None for cone in cones):
        return None
    return cones


def _matrix_integer_linear_solution(matrix, rhs):
    """Return one exact integer solution of ``matrix * x = rhs``, if it exists."""
    rows = [list(row) for row in matrix]
    if not rows:
        return (0, 0, 0, 0)
    row_count = len(rows)
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows) or len(rhs) != row_count:
        return None
    domain = _DomainMatrix.from_Matrix(_SympyMatrix(rows))
    diagonal_form, left, right = _smith_normal_decomp(domain)
    diagonal_form = diagonal_form.to_Matrix()
    left = left.to_Matrix()
    right = right.to_Matrix()
    transformed_rhs = left * _SympyMatrix(list(rhs))
    solution_coordinates = [0] * column_count
    diagonal_count = min(row_count, column_count)
    rank = 0
    for index in range(diagonal_count):
        diagonal = diagonal_form[index, index]
        if diagonal == 0:
            continue
        rank += 1
        quotient = transformed_rhs[index] / diagonal
        if getattr(quotient, "q", 1) != 1:
            return None
        solution_coordinates[index] = int(quotient)
    for index in range(rank, row_count):
        if transformed_rhs[index] != 0:
            return None
    solution = right * _SympyMatrix(solution_coordinates)
    values = tuple(int(value) for value in solution)
    if _SympyMatrix(rows) * _SympyMatrix(values) != _SympyMatrix(list(rhs)):
        return None
    return values


def _matrix_action_valid(matrix, torus_shift, lambda_f, ambient_rays, fan_cones):
    raw_matrix = matrix.tolist() if isinstance(matrix, np.ndarray) else matrix
    try:
        rows = [list(row) for row in raw_matrix]
    except TypeError:
        return False
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return False
    fractions = [
        [_matrix_validation_fraction(value) for value in row]
        for row in rows
    ]
    if any(value is None for row in fractions for value in row):
        return False
    if any(value.denominator != 1 for row in fractions for value in row):
        return False
    action = tuple(tuple(int(value) for value in row) for row in fractions)
    if any(
        sum(action[i][k] * action[k][j] for k in range(4)) != int(i == j)
        for i in range(4)
        for j in range(4)
    ):
        return False
    rays = _matrix_int_rows(ambient_rays, diagnostic=True)
    cones = _matrix_cone_rows(fan_cones, diagnostic=True)
    if rays is None or not rays or cones is None or not cones:
        return False
    ray_set = set(rays)
    if len(ray_set) != len(rays):
        return False
    mapped_rays = {
        tuple(sum(action[i][j] * ray[j] for j in range(4)) for i in range(4))
        for ray in rays
    }
    if mapped_rays != ray_set:
        return False
    cone_set = {frozenset(cone) for cone in cones}
    for cone in cones:
        image = frozenset(
            tuple(sum(action[i][j] * ray[j] for j in range(4)) for i in range(4))
            for ray in cone
        )
        if image not in cone_set:
            return False
    shift = _matrix_fraction_vector(torus_shift)
    if shift is None and isinstance(torus_shift, (list, tuple, np.ndarray)):
        try:
            shift = tuple(_matrix_validation_fraction(value) for value in torus_shift)
        except TypeError:
            shift = None
    if shift is None or len(shift) != 4 or any(value is None for value in shift):
        return False
    shift_delta = tuple(
        sum(action[i][j] * shift[j] for j in range(4)) - shift[i]
        for i in range(4)
    )
    if any(value.denominator != 1 for value in shift_delta):
        return False
    if any((2 * value).denominator != 1 for value in shift):
        return False
    parity = _matrix_validation_fraction(lambda_f)
    return parity is not None and parity.denominator == 1 and int(parity) in (0, 1)


def _matrix_label_valid(matrix, torus_shift, sigma_rays, nu, cox_phase_witness):
    exact_matrix = _exact_integer_matrix_4x4(matrix)
    shift = _matrix_fraction_vector(torus_shift)
    nu = _matrix_fraction_vector(nu)
    sigma = _matrix_int_rows(sigma_rays)
    if exact_matrix is None or shift is None or len(shift) != 4:
        return False
    if nu is None or len(nu) != 4 or sigma is None:
        return False
    action = np.asarray(exact_matrix, dtype=int)
    sigma_matrix = np.eye(4, dtype=int) - action
    if any(
        tuple(sum(action[i, j] * ray[j] for j in range(4)) for i in range(4)) != ray
        for ray in sigma
    ):
        return False
    image_nu = tuple(
        sum(Fraction(int(action[i, j])) * nu[j] for j in range(4))
        for i in range(4)
    )
    if image_nu != tuple(-value for value in nu):
        return False
    twice_nu = tuple(2 * value for value in nu)
    if any(value.denominator != 1 for value in twice_nu):
        return False
    if _matrix_integer_linear_solution(sigma_matrix.tolist(), [int(value) for value in twice_nu]) is None:
        return False
    if not isinstance(cox_phase_witness, dict) or cox_phase_witness.get("status") != "certified":
        return False
    alpha_raw = cox_phase_witness.get("alpha")
    lattice_raw = cox_phase_witness.get("lattice_vector")
    if isinstance(alpha_raw, np.ndarray):
        alpha_raw = alpha_raw.tolist()
    try:
        alpha = tuple(_matrix_fraction(value) for value in alpha_raw)
    except TypeError:
        return False
    lattice_vector = _matrix_int_vector(lattice_raw)
    if len(alpha) != len(sigma) or any(value is None for value in alpha) or lattice_vector is None:
        return False
    phase_vector = tuple(shift[index] + nu[index] for index in range(4))
    reconstructed = tuple(
        phase_vector[index]
        + sum(Fraction(ray[index]) * alpha[ray_index] for ray_index, ray in enumerate(sigma))
        for index in range(4)
    )
    return reconstructed == tuple(Fraction(value) for value in lattice_vector)


def _matrix_chart_valid(sigma_rays, ambient_rays, fan_cones):
    sigma = _matrix_int_rows(sigma_rays)
    ambient = _matrix_int_rows(ambient_rays)
    cones = _matrix_cone_rows(fan_cones)
    if sigma is None or ambient is None or cones is None:
        return False
    sigma_set = set(sigma)
    if len(sigma_set) != len(sigma) or not sigma_set.issubset(set(ambient)):
        return False
    for cone in cones:
        cone_set = set(cone)
        if not sigma_set.issubset(cone_set):
            continue
        if len(cone) <= 4 and _exact_rank(np.asarray(cone, dtype=int)) == len(cone):
            return True
    return False


def _matrix_rows_valid(dual_points, ambient_rays, fan_cones):
    rows = _matrix_int_rows(dual_points)
    ambient = _matrix_int_rows(ambient_rays)
    cones = _matrix_cone_rows(fan_cones)
    if rows is None or not rows or ambient is None or not ambient or cones is None:
        return False
    if len(set(rows)) != len(rows) or len(set(ambient)) != len(ambient):
        return False
    full_rays = {ray for cone in cones for ray in cone}
    if set(ambient) != full_rays:
        return False
    return all(
        1 + sum(ray[index] * point[index] for index in range(4)) >= 0
        for point in rows
        for ray in ambient
    )


def _matrix_frame(sigma_rays):
    sigma = _matrix_int_rows(sigma_rays)
    if sigma is None:
        return None
    if not sigma:
        return (0, 0, 0, 0)
    return _matrix_integer_linear_solution(sigma, [-1] * len(sigma))


def _canonical_dual_rows_sha256(rows):
    """Hash canonical sorted integer rows using compact JSON serialization."""
    canonical_rows = _matrix_int_rows(rows)
    if canonical_rows is None:
        return None
    payload = json.dumps(
        [list(row) for row in sorted(canonical_rows)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _extract_dual_lattice_points(poly, dual_polytope=None):
    """Return exact dual rows and provenance without importing CYTools.

    ``points()`` on the implicit dual is the only source that attests the full
    lattice-point set.  A vertices fallback, or an explicitly supplied dual
    polytope, carries rows but cannot attest completeness on its own.
    """
    explicit_dual_polytope = dual_polytope is not None
    if dual_polytope is None:
        try:
            dual_polytope = poly.dual()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None, None
    for method_name in ("points", "vertices"):
        method = getattr(dual_polytope, method_name, None)
        if callable(method):
            try:
                values = np.asarray(method(), dtype=int)
            except (RuntimeError, TypeError, ValueError):
                continue
            if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
                continue
            if len({tuple(int(value) for value in row) for row in values.tolist()}) != values.shape[0]:
                continue
            provenance = {
                "method": method_name,
                "complete": bool(method_name == "points" and not explicit_dual_polytope),
                "source": "cytools_dual_points",
                "row_count": int(values.shape[0]),
                "rows_sha256": _canonical_dual_rows_sha256(values),
            }
            return values, provenance
    return None, None


def _verified_dual_points_provenance(dual_points, dual_points_complete, provenance):
    """Return a copied provenance receipt with completeness rechecked."""
    if not isinstance(provenance, dict):
        return None
    receipt = copy.deepcopy(provenance)
    rows = _matrix_int_rows(dual_points)
    actual_hash = _canonical_dual_rows_sha256(dual_points)
    hash_matches = (
        actual_hash is not None
        and receipt.get("rows_sha256") == actual_hash
        and rows is not None
        and receipt.get("row_count") == len(rows)
    )
    method = receipt.get("method")
    source = receipt.get("source")
    source_is_authoritative = (
        (source == "cytools_dual_points" and method == "points")
        or (source == "fixture_attested" and method == "fixture_attested")
    )
    receipt["complete"] = bool(
        dual_points_complete is True
        and provenance.get("complete") is True
        and hash_matches
        and source_is_authoritative
    )
    return receipt


def _matrix_false_support_incomplete(component):
    """True when a new matrix False lacks the full support for consumers."""
    if not isinstance(component, dict):
        return False
    if component.get("containment_method") != "eq_4.42_restriction_matrix_nonzero_column":
        return False
    witness = component.get("containment_witness") or {}
    return not isinstance(witness, dict) or witness.get("support_complete") is not True


def _ordered_restriction_failures(failed_checks):
    found = set(failed_checks)
    return [code for code in _RESTRICTION_FAILURE_ORDER if code in found]


def _matrix_character_coordinates(q, frame, matrix, sigma_rays):
    """Return exact residual-character coordinates in the fixed torus."""
    action = np.asarray(matrix, dtype=int)
    invariant_basis = _integer_kernel_basis(np.eye(4, dtype=int) - action)
    sigma_coordinates = []
    for ray in sigma_rays:
        coordinates = _integer_coordinates(invariant_basis, ray)
        if coordinates is None:
            return None
        sigma_coordinates.append(coordinates)
    if sigma_coordinates:
        sigma_matrix = np.column_stack(sigma_coordinates)
    else:
        sigma_matrix = np.empty((invariant_basis.shape[1], 0), dtype=int)
    quotient_annihilator = _integer_kernel_basis(sigma_matrix.T).T
    framed_character = tuple(q[index] - frame[index] for index in range(4))
    return _quotient_character_coordinates(
        framed_character,
        invariant_basis,
        quotient_annihilator,
    )


def _matrix_phase_sign(phase):
    phase = Fraction(phase) % 1
    if phase == 0:
        return 1
    if phase == Fraction(1, 2):
        return -1
    return None


def _matrix_column_entries(column):
    return [
        {
            "character": [_fraction_to_json(value) for value in character],
            "coefficient": _fraction_to_json(coefficient),
        }
        for character, coefficient in sorted(column.items())
    ]


def restriction_matrix_columns(
    rows,
    provenance,
    ambient_rays,
    fan_cones,
    sigma_rays,
    nu,
    cox_phase_witness,
    matrix,
    torus_shift,
    lambda_f,
):
    """Build the exact Eq. 4.42 restriction column for each free orbit."""
    witness_alpha = (
        cox_phase_witness.get("alpha")
        if isinstance(cox_phase_witness, dict)
        else None
    )
    witness_lattice = (
        cox_phase_witness.get("lattice_vector")
        if isinstance(cox_phase_witness, dict)
        else None
    )
    inputs_are_exact = _matrix_input_is_exact(
        (
            rows,
            ambient_rays,
            fan_cones,
            sigma_rays,
            nu,
            witness_alpha,
            witness_lattice,
            matrix,
            torus_shift,
            lambda_f,
        )
    )
    failed_checks = []
    action_valid = _matrix_action_valid(
        matrix, torus_shift, lambda_f, ambient_rays, fan_cones
    )
    if not action_valid:
        failed_checks.append("action_not_validated")
    if not inputs_are_exact:
        failed_checks.append("inexact_input")
        return [], _ordered_restriction_failures(failed_checks)

    if not _matrix_label_valid(
        matrix, torus_shift, sigma_rays, nu, cox_phase_witness
    ):
        failed_checks.append("component_label_unbound")
    if not _matrix_chart_valid(sigma_rays, ambient_rays, fan_cones):
        failed_checks.append("chart_unavailable")
    if not _matrix_rows_valid(rows, ambient_rays, fan_cones):
        failed_checks.append("dual_rows_invalid")
    frame = _matrix_frame(sigma_rays)
    if frame is None:
        failed_checks.append("frame_unavailable")
    failed_checks = _ordered_restriction_failures(failed_checks)
    if failed_checks:
        return [], failed_checks

    exact_action = _exact_integer_matrix_4x4(matrix)
    action = np.asarray(exact_action, dtype=int)
    shift = _matrix_fraction_vector(torus_shift)
    nu_vector = _matrix_fraction_vector(nu)
    lam = int(_matrix_fraction(lambda_f))
    points = _matrix_int_rows(rows)
    sigma = _matrix_int_rows(sigma_rays)
    alpha = tuple(
        _matrix_fraction(value) for value in cox_phase_witness["alpha"]
    )
    kappa_phase = sum(alpha, Fraction(0)) + Fraction(lam, 2)
    kappa_sign = _matrix_phase_sign(kappa_phase)
    point_set = set(points)
    pending = set(points)
    orbit_records = []
    while pending:
        point = min(pending)
        image = tuple(
            sum(int(action[k, i]) * point[k] for k in range(4))
            for i in range(4)
        )
        if image == point:
            pending.remove(point)
            phase = _fraction_dot(shift, point) + Fraction(lam, 2)
            allowed = phase.denominator == 1
            survives = all(
                1 + sum(ray[index] * point[index] for index in range(4)) == 0
                for ray in sigma
            )
            character = (
                _matrix_character_coordinates(point, frame, exact_action, sigma)
                if allowed and survives
                else None
            )
            invalid_character = allowed and survives and character is None
            column = {character: Fraction(1)} if character is not None else {}
            status = (
                "forced_singleton"
                if not allowed
                else "column_invalid"
                if invalid_character
                else "nonzero"
                if column
                else "dead"
            )
            orbit_records.append(
                {
                    "q": list(point),
                    "Aq": list(point),
                    "h_q": _fraction_to_json(phase),
                    "h_Aq": _fraction_to_json(phase),
                    "survives": bool(survives),
                    "free": bool(allowed),
                    "pair_phase_consistent": True,
                    "phase": _fraction_to_json(phase),
                    "character": character,
                    "column": column,
                    "status": status,
                    "kappa": kappa_sign,
                }
            )
            continue

        present = image in point_set
        pending.discard(point)
        if present:
            pending.discard(image)
            representative, partner = sorted((point, image))
            # A is an involution, so the image of the representative is the
            # other member regardless of which row sorted first.
            partner = tuple(
                sum(int(action[k, i]) * representative[k] for k in range(4))
                for i in range(4)
            )
        else:
            representative, partner = point, None
        h_q = _fraction_dot(shift, representative) + Fraction(lam, 2)
        if partner is None:
            h_partner = None
            pair_consistent = False
            survives = all(
                1 + sum(ray[index] * representative[index] for index in range(4)) == 0
                for ray in sigma
            )
            character = None
            column = {}
            phase = _fraction_dot(
                tuple(shift[index] - nu_vector[index] for index in range(4)),
                representative,
            ) + Fraction(lam, 2)
            status = "partner_missing"
        else:
            h_partner = _fraction_dot(shift, partner) + Fraction(lam, 2)
            pair_consistent = (h_q + h_partner).denominator == 1
            survives = all(
                1 + sum(ray[index] * representative[index] for index in range(4)) == 0
                for ray in sigma
            )
            phase = _fraction_dot(
                tuple(shift[index] - nu_vector[index] for index in range(4)),
                representative,
            ) + Fraction(lam, 2)
            sign = _matrix_phase_sign(phase)
            character = (
                _matrix_character_coordinates(representative, frame, exact_action, sigma)
                if survives
                else None
            )
            invalid_character = survives and character is None
            invalid_phase = survives and sign is None
            if invalid_character or invalid_phase:
                column = {}
                status = "column_invalid"
            elif not survives:
                column = {}
                status = "dead"
            elif sign == 1:
                column = {character: Fraction(2)}
                status = "nonzero"
            else:
                column = {}
                status = "zero"
        orbit_records.append(
            {
                "q": list(representative),
                "Aq": None if partner is None else list(partner),
                "image_q": list(image),
                "h_q": _fraction_to_json(h_q),
                "h_Aq": None if h_partner is None else _fraction_to_json(h_partner),
                "survives": bool(survives),
                "free": bool(partner is not None),
                "pair_phase_consistent": bool(pair_consistent),
                "phase": _fraction_to_json(phase),
                "character": character,
                "column": column,
                "status": status,
                "kappa": kappa_sign,
            }
        )
    orbit_records.sort(key=lambda item: tuple(item["q"]))
    return orbit_records, []


def _restriction_column_status(orbit_records):
    return [
        {
            "q": list(record["q"]),
            "Aq": copy.deepcopy(record.get("Aq")),
            "survives": record.get("survives") is True,
            "free": record.get("free") is True,
            "status": record.get("status"),
        }
        for record in orbit_records
    ]


def _restriction_matrix_shape(orbit_records):
    characters = {
        tuple(record["character"])
        for record in orbit_records
        if record.get("free") is True
        and record.get("survives") is True
        and record.get("character") is not None
    }
    columns = sum(record.get("free") is True for record in orbit_records)
    return [len(characters), columns]


def _restriction_nonzero_orbits(orbit_records):
    return [
        record
        for record in orbit_records
        if record.get("free") is True
        and record.get("survives") is True
        and bool(record.get("column"))
    ]


def _restriction_matrix_witness(
    orbit_records,
    failed_checks,
    provenance,
    value,
    *,
    frame_available,
    totality_violation=False,
):
    witness = {"failed_checks": list(failed_checks)}
    if provenance is not None:
        witness["dual_points_provenance"] = copy.deepcopy(provenance)
    if frame_available:
        kappas = {
            record.get("kappa")
            for record in orbit_records
            if record.get("kappa") in (-1, 1)
        }
        witness["kappa"] = next(iter(kappas)) if len(kappas) == 1 else None
        witness["column_status"] = _restriction_column_status(orbit_records)
    nonzero = sorted(
        _restriction_nonzero_orbits(orbit_records),
        key=lambda record: tuple(record["q"]),
    )
    if value is True:
        witness["orbit_count"] = sum(
            record.get("free") is True for record in orbit_records
        )
        witness["restriction_matrix_shape"] = _restriction_matrix_shape(orbit_records)
    elif value is False and nonzero:
        record = nonzero[0]
        witness.update(
            {
                "q": list(record["q"]),
                "Aq": copy.deepcopy(record.get("Aq")),
                "phase": copy.deepcopy(record.get("phase")),
                "column": _matrix_column_entries(record.get("column") or {}),
                "support_complete": (
                    provenance.get("complete") is True
                    if isinstance(provenance, dict)
                    else False
                ),
            }
        )
    if totality_violation:
        witness["totality_violation"] = True
    return witness


def decide_from_columns(orbit_records, failed_checks, provenance, fast_path=None):
    """Decide from exact orbit columns, checking the invariant and fast paths."""
    orbit_records = list(orbit_records or ())
    failures = list(failed_checks or ())
    frame_available = not any(
        reason in failures
        for reason in _RESTRICTION_FAILURE_ORDER[:6]
    )
    if frame_available:
        if any(record.get("pair_phase_consistent") is not True for record in orbit_records):
            failures.append("pair_phase_inconsistent")
        if any(record.get("status") == "column_invalid" for record in orbit_records):
            failures.append("column_uniformity_violated")
        surviving = [
            record
            for record in orbit_records
            if record.get("free") is True and record.get("survives") is True
        ]
        statuses = {
            "nonzero" if record.get("column") else "zero"
            for record in surviving
            if record.get("status") not in ("forced_singleton", "partner_missing")
        }
        if len(statuses) > 1:
            failures.append("column_uniformity_violated")
        kappas = {
            record.get("kappa")
            for record in orbit_records
            if record.get("kappa") in (-1, 1)
        }
        if len(kappas) == 1 and statuses:
            kappa = next(iter(kappas))
            if (kappa == -1 and "nonzero" in statuses) or (
                kappa == 1 and "zero" in statuses
            ):
                failures.append("column_uniformity_violated")
    failures = _ordered_restriction_failures(failures)
    nonzero = _restriction_nonzero_orbits(orbit_records)
    all_zero = not nonzero
    provenance_complete = (
        isinstance(provenance, dict) and provenance.get("complete") is True
    )
    if frame_available and not failures and all_zero and not provenance_complete:
        failures.append("dual_points_incomplete")
        failures = _ordered_restriction_failures(failures)
    reason_code = failures[0] if failures else None
    if reason_code is not None:
        matrix_value = None
        matrix_method = "non_smooth_containment_undetermined"
    elif nonzero:
        matrix_value = False
        matrix_method = "eq_4.42_restriction_matrix_nonzero_column"
    else:
        matrix_value = True
        matrix_method = "eq_4.42_restriction_matrix_zero"
    matrix_witness = _restriction_matrix_witness(
        orbit_records,
        failures,
        provenance,
        matrix_value,
        frame_available=frame_available,
    )
    if fast_path is not None:
        fast_value, fast_method = fast_path
        if fast_value not in (True, False):
            conflict_witness = {
                "failed_checks": ["certificate_conflict"],
                "totality_violation": True,
            }
            return (
                None,
                "containment_certificate_conflict",
                "certificate_conflict",
                None,
                conflict_witness,
            )
        if matrix_value is not None and matrix_value is not fast_value:
            conflict_witness = {
                "failed_checks": ["certificate_conflict"],
                "fast_path_certificate": {
                    "value": fast_value,
                    "method": fast_method,
                },
                "matrix_certificate": {
                    "value": matrix_value,
                    "method": matrix_method,
                    "witness": matrix_witness,
                },
            }
            return (
                None,
                "containment_certificate_conflict",
                "certificate_conflict",
                None,
                conflict_witness,
            )
        final_witness = matrix_witness
        if matrix_value is None and not frame_available:
            final_witness = {"failed_checks": failures}
        return (
            fast_value,
            fast_method,
            None,
            True if fast_value is False else False,
            final_witness,
        )
    if matrix_value is None and not failures:
        # This is an internal totality guard. It fails closed instead of
        # allowing an unclassified component to escape the builder.
        witness = _restriction_matrix_witness(
            orbit_records,
            ["certificate_conflict"],
            provenance,
            None,
            frame_available=frame_available,
            totality_violation=True,
        )
        return (
            None,
            "containment_certificate_conflict",
            "certificate_conflict",
            None,
            witness,
        )
    generic = False if matrix_value is True else True if matrix_value is False else None
    return matrix_value, matrix_method, reason_code, generic, matrix_witness


def restriction_matrix_decision(
    rows,
    provenance,
    ambient_rays,
    fan_cones,
    sigma_rays,
    nu,
    cox_phase_witness,
    matrix,
    torus_shift,
    lambda_f,
    *,
    fast_path=None,
):
    """Compose exact geometry-column construction and the decision procedure."""
    orbit_records, failed_checks = restriction_matrix_columns(
        rows,
        provenance,
        ambient_rays,
        fan_cones,
        sigma_rays,
        nu,
        cox_phase_witness,
        matrix,
        torus_shift,
        lambda_f,
    )
    return decide_from_columns(
        orbit_records,
        failed_checks,
        _verified_dual_points_provenance(
            rows,
            provenance.get("complete") is True if isinstance(provenance, dict) else False,
            provenance,
        ),
        fast_path=fast_path,
    )


def _fixed_component_records(
    auxiliary_fan,
    matrix,
    torus_shift,
    lambda_f,
    *,
    fixed_cone_keys,
    half_ray_cache=None,
    dual_points=None,
    ambient_rays=None,
    fan_cones=None,
    dual_points_complete=False,
    dual_points_provenance=None,
    decision_log=None,
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
    arithmetic.  Every component also retains its exact Cox alpha/lattice
    witness and, when dual lattice points are supplied, its invariant
    restricted monomial support.
    """
    nu_representatives = enumerate_projected_lattice_representatives(matrix, -1)
    verified_dual_provenance = _verified_dual_points_provenance(
        dual_points,
        dual_points_complete,
        dual_points_provenance,
    )
    verified_dual_points_complete = (
        verified_dual_provenance is not None
        and verified_dual_provenance.get("complete") is True
    )
    decision_log_indices = {}
    # The connected fixed torus has the +1 eigenspace of L.  Keep the
    # source convention (rank(I+L)) while evaluating the rank exactly.
    fixed_subspace_dimension = _exact_rank(
        np.eye(4, dtype=int) + np.asarray(matrix, dtype=int)
    )
    admissible = []
    for source_rays in fixed_cone_keys:
        rays = [tuple(int(value) for value in ray) for ray in source_rays]
        sigma_dimension = _exact_rank(np.asarray(rays, dtype=int)) if rays else 0
        # The half-ray shortcut proof depends only on (auxiliary_fan, matrix,
        # rays) -- all fixed across the 16 torus shifts and both lambda_f
        # values of a matrix -- so cache it per matrix (finding F3). Bit-for-bit
        # identical, ~18s of recomputation removed on a 6-polytope h11=4 slice.
        half_ray_key = tuple(rays)
        if half_ray_cache is not None and half_ray_key in half_ray_cache:
            use_half_ray, shortcut_reason = half_ray_cache[half_ray_key]
        else:
            use_half_ray, shortcut_reason = _half_ray_shortcut_proof(
                auxiliary_fan,
                matrix,
                rays,
            )
            if half_ray_cache is not None:
                half_ray_cache[half_ray_key] = (use_half_ray, shortcut_reason)
        # A cache is an optimization, not a smoothness certificate.  Keep the
        # source boundary fail-closed even when a caller supplies stale or
        # malformed cache contents for a nonsmooth original cone.
        if use_half_ray and not _cone_is_smooth_in_lattice(rays, 4):
            use_half_ray = False
            shortcut_reason = "sigma_cone_not_smooth"
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
            phase_vector = _fraction_sum([torus_shift, nu])
            cox_witness = _cox_alpha_lattice_witness(rays, phase_vector)
            sigma_lattice_matrix = (
                np.asarray(rays, dtype=int).T
                if rays else np.empty((4, 0), dtype=int)
            )
            lattice_index = _integer_lattice_index(sigma_lattice_matrix)
            support = invariant_restricted_monomial_support(
                dual_points,
                ambient_rays if ambient_rays is not None else tuple(
                    tuple(int(value) for value in ray)
                    for cone in (fan_cones or ())
                    for ray in cone
                ),
                rays,
                matrix,
                torus_shift,
                lambda_f,
                fan_cones=fan_cones,
                invariant_basis=_integer_kernel_basis(np.eye(4, dtype=int) - matrix),
            )
            singleton_witness = None
            containment_reason_code = None
            if use_half_ray:
                vanishes_identically = (sigma_dimension + int(lambda_f)) % 2 == 1
                containment_method = "eq_4.46_after_smooth_half_ray_eq_4.35"
                containment_generic = (
                    False if vanishes_identically else True
                )
                containment_witness = None
            else:
                fast_path = None
                if _dense_torus_inversion_cancellation_applies(
                    matrix,
                    torus_shift,
                    lambda_f,
                    rays,
                    ambient_dimension,
                    support,
                    dual_points_complete=verified_dual_points_complete,
                ):
                    fast_path = (
                        True,
                        "eq_4.42_dense_torus_inversion_cancellation",
                    )
                else:
                    # Keep the accepted #236 singleton proof as the second
                    # fast path. The general matrix below must agree with any
                    # decisive fast path whose domain is independently valid.
                    singleton_witness = _singleton_nonvanishing_witness(
                        matrix, support, rays
                    )
                    if singleton_witness is not None:
                        fast_path = (
                            False,
                            "eq_4.42_fixed_singleton_nonvanishing_witness",
                        )
                try:
                    (
                        vanishes_identically,
                        containment_method,
                        containment_reason_code,
                        containment_generic,
                        containment_witness,
                    ) = restriction_matrix_decision(
                        dual_points,
                        verified_dual_provenance,
                        ambient_rays if ambient_rays is not None else tuple(
                            tuple(int(value) for value in ray)
                            for cone in (fan_cones or ())
                            for ray in cone
                        ),
                        fan_cones,
                        rays,
                        nu,
                        {
                            "status": cox_witness["status"],
                            "equation": cox_witness.get(
                                "equation", "t + nu + sum(alpha_p p) = n in N"
                            ),
                            "alpha": cox_witness.get("alpha"),
                            "lattice_vector": cox_witness.get("lattice_vector"),
                            "phase_vector": _fraction_vector_to_json(phase_vector),
                        },
                        matrix,
                        torus_shift,
                        lambda_f,
                        fast_path=fast_path,
                    )
                except AssertionError:
                    # A failed totality assertion is a component-level
                    # unavailable certificate, never an exception escaping
                    # the builder (clarification M-5).
                    vanishes_identically = None
                    containment_method = "containment_certificate_conflict"
                    containment_reason_code = "certificate_conflict"
                    containment_generic = None
                    containment_witness = {
                        "failed_checks": ["certificate_conflict"],
                        "totality_violation": True,
                    }
                if (
                    singleton_witness is not None
                    and containment_method
                    == "eq_4.42_fixed_singleton_nonvanishing_witness"
                ):
                    # Rule W keeps A-6 identity fields authoritative at top level.
                    matrix_fields = ("q", "Aq", "phase", "column")
                    matrix_witness = {}
                    if isinstance(containment_witness, dict):
                        if all(key in containment_witness for key in matrix_fields):
                            matrix_witness = {
                                key: copy.deepcopy(containment_witness.pop(key))
                                for key in matrix_fields
                            }
                        for key, value in singleton_witness.items():
                            containment_witness[key] = copy.deepcopy(value)
                        if matrix_witness:
                            containment_witness["matrix_witness"] = matrix_witness
                    elif containment_witness is None:
                        containment_witness = copy.deepcopy(singleton_witness)
                elif singleton_witness is not None and isinstance(containment_witness, dict):
                    # Preserve the 40fecd89 layout for conflict records.
                    for key, value in singleton_witness.items():
                        containment_witness.setdefault(key, copy.deepcopy(value))
                elif singleton_witness is not None and containment_witness is None:
                    containment_witness = copy.deepcopy(singleton_witness)
            # A non-vanishing section has no zero-dimensional hypersurface
            # component. Log the decision first: the zero-dimensional False
            # record is intentionally absent from the returned records.
            component_record = {
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
                "cox_phase_witness": {
                    "status": cox_witness["status"],
                    "equation": cox_witness.get(
                        "equation", "t + nu + sum(alpha_p p) = n in N"
                    ),
                    "alpha": cox_witness.get("alpha"),
                    "lattice_vector": cox_witness.get("lattice_vector"),
                    "phase_vector": _fraction_vector_to_json(phase_vector),
                },
                "fixed_cone_lattice": {
                    "rank": int(sigma_dimension),
                    "index": int(lattice_index),
                    "saturated": bool(lattice_index == 1),
                    "stabilizer_order": int(lattice_index),
                    "method": "exact_gcd_of_maximal_minors",
                },
                "invariant_restricted_support": support,
                "containment_method": containment_method,
                "containment_reason_code": containment_reason_code,
                "containment_generic": containment_generic,
                "containment_witness": containment_witness,
                "fixed_toric_dimension": ambient_dimension,
                "f_vanishes_identically": vanishes_identically,
                "hypersurface_component_dimension": (
                    None
                    if vanishes_identically is None
                    else ambient_dimension
                    if vanishes_identically
                    else max(ambient_dimension - 1, 0)
                ),
            }
            if decision_log is not None:
                decision_log.append(
                    {
                        "sigma_rays": copy.deepcopy(component_record["sigma_rays"]),
                        "nu": copy.deepcopy(component_record["nu"]),
                        "route": "smooth_half_ray" if use_half_ray else "non_smooth_fallback",
                        "f_vanishes_identically": vanishes_identically,
                        "containment_method": containment_method,
                        "containment_reason_code": containment_reason_code,
                        "containment_generic": containment_generic,
                        "containment_witness": copy.deepcopy(containment_witness),
                        "fixed_toric_dimension": ambient_dimension,
                        "kept_in_output": False,
                    }
                )
                decision_log_indices[id(component_record)] = len(decision_log) - 1
            if (
                vanishes_identically is True
                and ambient_dimension == 0
            ):
                component_record[
                    "zero_dimensional_local_smoothness_cartier_evidence"
                ] = _derive_zero_dimensional_local_evidence(
                    auxiliary_fan,
                    matrix,
                    component_record,
                )
            if ambient_dimension == 0 and vanishes_identically is False:
                continue
            admissible.append(component_record)

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

    result = sorted(
        retained,
        key=lambda item: (
            item["sigma_dimension"],
            tuple(item["sigma_rays"]),
            tuple(item["nu"]["numerator"]),
        ),
    )
    for component in result:
        log_index = decision_log_indices.get(id(component))
        if log_index is not None:
            decision_log[log_index]["kept_in_output"] = True
    return result

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


def _phase_json(value):
    """Encode a phase exponent in the exact common scalar format."""

    return _fraction_to_json(Fraction(value))


def _cox_alpha_lattice_witness(sigma_rays, vector):
    """Solve the exact Cox phase equation for one fixed component.

    The witness is the rational solution of

    ``vector + sum(alpha_p * p) = n``

    with ``n`` in the ambient lattice.  The quotient-lattice condition from
    Moritz eq. (4.30) guarantees that the displayed system has a solution;
    setting all free parameters to zero gives a deterministic representative.
    Keep both ``alpha`` and ``n`` because the latter is the lattice witness,
    not a floating-point reconstruction of the phase.
    """

    sigma_rays = tuple(tuple(int(value) for value in ray) for ray in sigma_rays)
    vector = tuple(Fraction(value) for value in vector)
    rank = _exact_rank(np.asarray(sigma_rays, dtype=int)) if sigma_rays else 0
    if rank != len(sigma_rays):
        return {
            "status": "unavailable",
            "reason_code": "nonsaturated_fixed_cone_lattice",
            "reason": "fixed-cone generators are linearly dependent",
            "alpha": None,
            "lattice_vector": None,
        }
    if not sigma_rays:
        if not _is_integral(vector):
            return {
                "status": "unavailable",
                "reason_code": "phase_not_in_ambient_lattice",
                "reason": "the empty-cone phase has no integral lattice witness",
                "alpha": [],
                "lattice_vector": None,
            }
        return {
            "status": "certified",
            "equation": "t + nu + sum(alpha_p p) = n in N",
            "alpha": [],
            "lattice_vector": _integer_vector(vector),
        }

    ray_matrix = _SympyMatrix(np.asarray(sigma_rays, dtype=int).T.tolist())
    identity = _SympyMatrix.eye(4)
    rhs = _SympyMatrix([
        -_SympyMatrix([value.numerator for value in vector])[index, 0]
        / vector[index].denominator
        for index in range(4)
    ])
    system = ray_matrix.row_join(-identity)
    try:
        solution, parameters = system.gauss_jordan_solve(rhs)
    except (ValueError, TypeError):
        return {
            "status": "unavailable",
            "reason_code": "cox_phase_witness_unavailable",
            "reason": "the exact Cox phase system has no rational solution",
            "alpha": None,
            "lattice_vector": None,
        }
    substitutions = {parameter: 0 for parameter in parameters}
    solution = [value.subs(substitutions) for value in solution]
    alpha = tuple(_exact_fraction(value) for value in solution[: len(sigma_rays)])
    # ``system`` is ``[A | -I] [alpha; n] = -vector``, so the second
    # solution block is already the ambient lattice witness ``n``.  Negating
    # it would record ``-n`` and no longer certify
    # ``vector + A*alpha = n`` in Eq. (4.30).
    lattice_vector = tuple(
        _exact_fraction(value) for value in solution[len(sigma_rays) :]
    )
    if not _is_integral(lattice_vector):
        return {
            "status": "unavailable",
            "reason_code": "cox_phase_lattice_witness_nonintegral",
            "reason": "the exact Cox phase solution does not end at an ambient lattice point",
            "alpha": [_fraction_to_json(value) for value in alpha],
            "lattice_vector": [_fraction_to_json(value) for value in lattice_vector],
        }
    return {
        "status": "certified",
        "equation": "t + nu + sum(alpha_p p) = n in N",
        "alpha": [_fraction_to_json(value) for value in alpha],
        "lattice_vector": _integer_vector(lattice_vector),
    }


def _quotient_character_coordinates(q, invariant_basis, quotient_annihilator):
    """Express a restricted dual point in exact quotient-lattice coordinates."""

    q = tuple(Fraction(value) for value in q)
    basis = _SympyMatrix(np.asarray(invariant_basis, dtype=int).tolist())
    annihilator = _SympyMatrix(np.asarray(quotient_annihilator, dtype=int).T.tolist())
    target = basis.T * _SympyMatrix([_exact_fraction(value) for value in q])
    try:
        solution = annihilator.gauss_jordan_solve(target)[0]
    except (ValueError, TypeError):
        return None
    if any(getattr(value, "free_symbols", set()) for value in solution):
        return None
    return tuple(_exact_fraction(value) for value in solution)


def _component_chart_evidence(sigma_rays, fan_cones, ambient_rays, stabilizer_order):
    """Record an exact Cox chart for the fixed toric orbit."""

    sigma = frozenset(tuple(int(value) for value in ray) for ray in sigma_rays)
    fan_cone_rows = fan_cones if fan_cones is not None else ()
    fan_cones = tuple(
        frozenset(tuple(int(value) for value in ray) for ray in cone)
        for cone in fan_cone_rows
    )
    chart_exists = any(sigma.issubset(cone) for cone in fan_cones)
    ambient_ray_rows = ambient_rays if ambient_rays is not None else ()
    ambient_rays = tuple(
        tuple(int(value) for value in ray) for ray in ambient_ray_rows
    )
    return {
        "status": "certified" if chart_exists else "unavailable",
        "reason_code": None if chart_exists else "cox_chart_unavailable",
        "reason": None if chart_exists else "the fixed cone is absent from the supplied fan",
        "chart_cone": [list(ray) for ray in sorted(sigma)],
        "cox_coordinates": {
            str(ray): (0 if ray in sigma else 1) for ray in sorted(ambient_rays)
        },
        "nonzero_coordinate_rays": [list(ray) for ray in sorted(set(ambient_rays) - sigma)],
        "irrelevant_ideal_avoided": bool(chart_exists),
        "stabilizer_order": int(stabilizer_order),
        "stabilizer_method": "exact_gcd_of_maximal_cone_minors",
    }


def invariant_restricted_monomial_support(
    dual_points,
    ambient_rays,
    sigma_rays,
    matrix,
    torus_shift,
    lambda_f,
    *,
    fan_cones=None,
    invariant_basis=None,
    quotient_annihilator=None,
):
    """Compute the exact covariant monomials surviving on one fixed component.

    Use all exact dual lattice points, the Cox monomial
    ``s_q = prod_p x_p^(<p,q>+1)``, and the coefficient covariance in Moritz
    eq. (4.42).  This is the owner-approved derived implementation for the
    nonsmooth branch.  Issue #236: ``restriction_identically_zero`` only
    reports that the supplied pre-cancellation support is empty.  A nonempty
    support does not prove a nonzero restriction, because swapped pairs can
    cancel.  An empty support proves vanishing only when the dual points are
    complete; the non-smooth containment fallback does not rely on that and
    records such cases as undetermined.  No complete restricted line system
    is reconstructed.
    """

    if dual_points is None:
        return {
            "status": "unavailable",
            "reason_code": "dual_lattice_points_unavailable",
            "reason": "exact dual lattice points are required for invariant support",
            "support": [],
            "newton_face": [],
            "restriction_identically_zero": None,
        }
    try:
        dual = np.asarray(dual_points, dtype=int)
        rays = tuple(tuple(int(value) for value in ray) for ray in ambient_rays)
        sigma = tuple(tuple(int(value) for value in ray) for ray in sigma_rays)
        if dual.ndim != 2 or dual.shape[1] != 4 or not len(dual):
            raise ValueError("dual lattice points must have shape (n, 4)")
        if len(set(tuple(int(value) for value in row) for row in dual.tolist())) != len(dual):
            raise ValueError("dual lattice points must be unique")
        if not rays or any(len(ray) != 4 for ray in rays):
            raise ValueError("ambient ray data is unavailable")
        matrix = np.asarray(matrix, dtype=int)
        torus_shift = tuple(Fraction(value) for value in torus_shift)
        if len(torus_shift) != 4:
            raise ValueError("torus shift must have rank four")
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "status": "unavailable",
            "reason_code": "invalid_exact_support_input",
            "reason": str(exc),
            "support": [],
            "newton_face": [],
            "restriction_identically_zero": None,
        }

    point_rows = [tuple(int(value) for value in row) for row in dual.tolist()]
    point_set = set(point_rows)
    mapped = {
        point: tuple(int(value) for value in (matrix.T @ np.asarray(point, dtype=int)).tolist())
        for point in point_rows
    }
    if any(image not in point_set for image in mapped.values()):
        return {
            "status": "unavailable",
            "reason_code": "dual_lattice_action_not_preserved",
            "reason": "the exact dual lattice point set is not preserved by L",
            "support": [],
            "newton_face": [],
            "restriction_identically_zero": None,
        }
    if invariant_basis is None:
        invariant_basis = _integer_kernel_basis(np.eye(4, dtype=int) - matrix)
    if quotient_annihilator is None:
        sigma_matrix = np.asarray(sigma, dtype=int).T if sigma else np.empty((4, 0), dtype=int)
        sigma_coordinates = (
            _integer_coordinates(invariant_basis, ray)
            for ray in sigma
        )
        coordinates = [value for value in sigma_coordinates if value is not None]
        sigma_matrix = (
            np.column_stack(coordinates)
            if coordinates
            else np.empty((invariant_basis.shape[1], 0), dtype=int)
        )
        quotient_annihilator = _integer_kernel_basis(sigma_matrix.T).T

    support = []
    reference_q = None
    covariance = []
    for q in point_rows:
        image = mapped[q]
        phase = _fraction_dot(torus_shift, q) + Fraction(int(lambda_f), 2)
        exponent = tuple(_fraction_dot(ray, q) + 1 for ray in rays)
        if any(value.denominator != 1 or value < 0 for value in exponent):
            return {
                "status": "unavailable",
                "reason_code": "invalid_anticanonical_monomial_exponent",
                "reason": "a dual lattice point produced a nonintegral or negative Cox exponent",
                "support": [],
                "newton_face": [],
                "restriction_identically_zero": None,
            }
        survives_chart = all(
            exponent[rays.index(ray)] == 0 for ray in sigma
        )
        fixed_q = image == q
        coefficient_allowed = (phase.denominator == 1) if fixed_q else True
        covariance.append({
            "q": list(q),
            "image_q": list(image),
            "fixed_dual_point": bool(fixed_q),
            "phase_exponent": _phase_json(phase),
            "coefficient_allowed": bool(coefficient_allowed),
            "equation": "psi[L(q)] = exp(2*pi*i*(<t,q> + lambda_f/2))*psi[q]",
        })
        if not survives_chart or not coefficient_allowed:
            continue
        if reference_q is None:
            reference_q = q
        character = _quotient_character_coordinates(
            tuple(left - right for left, right in zip(q, reference_q)),
            invariant_basis,
            quotient_annihilator,
        )
        if character is None:
            return {
                "status": "unavailable",
                "reason_code": "restricted_character_coordinates_unavailable",
                "reason": "a surviving dual point has no exact quotient-character coordinate",
                "support": [],
                "newton_face": [],
                "restriction_identically_zero": None,
            }
        support.append({
            "q": list(q),
            "orbit_q": [list(q), list(image)] if image != q else [list(q)],
            "image_q": list(image),
            "cox_exponents": [int(value) for value in exponent],
            "restricted_character": [_fraction_to_json(value) for value in character],
            "phase_exponent": _phase_json(phase),
            "chart_nonzero": True,
        })
    unique_support = {}
    for item in support:
        orbit_key = tuple(sorted(tuple(point) for point in item["orbit_q"]))
        unique_support.setdefault(orbit_key, item)
    support = [unique_support[key] for key in sorted(unique_support)]
    return {
        "status": "certified",
        "reason_code": None,
        "reason": None,
        "method": "exact_dual_lattice_monomials_and_eq_4.42_covariance",
        "source_anchor": "Moritz KS_orientifolds.tex lines 285-296 and 587-601",
        "dual_point_count": int(len(point_rows)),
        "dual_point_source": "CYTools dual.points/vertices exact integer fallback",
        "ambient_rays": [list(ray) for ray in rays],
        "covariance": covariance,
        "support": support,
        "newton_face": [item["restricted_character"] for item in support],
        "restriction_identically_zero": not support,
        "chart": _component_chart_evidence(
            sigma,
            fan_cones,
            rays,
            _integer_lattice_index(np.asarray(sigma, dtype=int).T if sigma else np.empty((4, 0), dtype=int)),
        ),
        "restriction_statement": (
            "restriction_identically_zero reports only that the supplied "
            "pre-cancellation invariant support is empty; it is not a "
            "containment decision (issue #236)"
        ),
    }

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

def _derive_zero_dimensional_local_evidence(auxiliary_fan, matrix, component):
    """Derive the exact local certificate for a contained point.

    Use the invariant lattice of ``matrix`` for the local cone, and use the
    auxiliary-fan ambient provenance for the restriction of ``-K_V``.  The
    returned record is evidence, not an acceptance shortcut: a nonsmooth
    local cone or missing/inconsistent Cartier provenance remains explicit so
    ``_zero_dimensional_component_local_certificate`` can fail closed.
    """

    schema = "cyaxiverse-zero-dimensional-local-evidence-1.0"
    matrix = np.asarray(matrix, dtype=int)
    sigma_rays = tuple(
        tuple(int(value) for value in ray)
        for ray in component.get("sigma_rays", [])
    )
    sigma_dimension = int(component.get("sigma_dimension", len(sigma_rays)))
    invariant_basis = _integer_kernel_basis(np.eye(4, dtype=int) - matrix)
    fixed_rank = int(invariant_basis.shape[1])
    base = {
        "local_evidence_schema": schema,
        "method": "exact_invariant_lattice_and_auxiliary_fan_provenance",
        "source_anchor": (
            "Moritz KS_orientifolds.tex lines 631-657; "
            "owner-approved zero-dimensional local certificate contract"
        ),
        "fixed_lattice_basis": np.asarray(invariant_basis, dtype=int).tolist(),
        "sigma_rays": [list(ray) for ray in sigma_rays],
        "sigma_dimension": sigma_dimension,
        "fixed_lattice_rank": fixed_rank,
    }

    def unavailable(reason_code, reason, **extra):
        return {
            **base,
            **extra,
            "status": "unavailable",
            "reason_code": reason_code,
            "reason": reason,
        }

    if fixed_rank != sigma_dimension or fixed_rank < 1:
        return unavailable(
            "zero_dimensional_local_cone_unavailable",
            (
                "the contained point has no positive-rank full-dimensional "
                "local cone in the invariant lattice"
            ),
        )
    if len(sigma_rays) != fixed_rank:
        return unavailable(
            "zero_dimensional_local_cone_unavailable",
            "the local fixed cone is not simplicial and full-dimensional",
        )

    sigma_coordinates = []
    for ray in sigma_rays:
        coordinates = _integer_coordinates(invariant_basis, ray)
        if coordinates is None:
            return unavailable(
                "zero_dimensional_local_cone_unavailable",
                "a fixed-cone ray has no integral coordinate in the invariant lattice",
            )
        sigma_coordinates.append(coordinates)
    local_matrix = np.column_stack(sigma_coordinates)
    determinant = abs(_exact_determinant(local_matrix))
    local_cone = {
        "rays": [list(ray) for ray in sigma_rays],
        "coordinates": [coordinates.tolist() for coordinates in sigma_coordinates],
        "determinant": int(determinant),
    }

    containing_cones = []
    sigma_set = set(sigma_rays)
    for cone in auxiliary_fan or ():
        rays = tuple(
            tuple(int(value) for value in ray)
            for ray in cone.get("rays", [])
        )
        if int(cone.get("dimension", -1)) != fixed_rank:
            continue
        if not sigma_set.issubset(set(rays)):
            continue
        if not bool(cone.get("simplicial")) or len(rays) != fixed_rank:
            return unavailable(
                "zero_dimensional_local_cone_unavailable",
                "the containing auxiliary local cone is not simplicial",
                local_cone=local_cone,
            )
        containing_cones.append((cone, rays))
    if not containing_cones:
        return unavailable(
            "zero_dimensional_local_cone_unavailable",
            "no full-dimensional auxiliary local cone has been identified",
            local_cone=local_cone,
        )

    restricted_supports = []
    ambient_determinants = []
    provenance = []
    for cone, rays in containing_cones:
        ambient_cones = cone.get("ambient_cones", [])
        if not ambient_cones:
            return unavailable(
                "zero_dimensional_restricted_cartier_unavailable",
                "the auxiliary local cone has no selected-fan provenance",
                local_cone=local_cone,
            )
        for ambient in ambient_cones:
            ambient_rays = tuple(
                tuple(int(value) for value in ray)
                for ray in ambient
            )
            ambient_array = np.asarray(ambient_rays, dtype=int)
            if ambient_array.shape != (4, 4):
                return unavailable(
                    "zero_dimensional_ambient_provenance_unavailable",
                    "selected-fan provenance is not a full-dimensional ambient cone",
                    local_cone=local_cone,
                    ambient_cone_provenance=provenance,
                    ambient_cone_determinants=ambient_determinants,
                )
            ambient_determinant = abs(_exact_determinant(ambient_array))
            ambient_determinants.append(int(ambient_determinant))
            ambient_data = _ambient_anticanonical_cartier_data(ambient_rays)
            if ambient_data is None:
                reason_code = (
                    "zero_dimensional_ambient_provenance_cone_not_smooth"
                    if ambient_determinant != 1
                    else "zero_dimensional_restricted_cartier_unavailable"
                )
                reason = (
                    "selected-fan provenance contains a non-unimodular ambient cone"
                    if ambient_determinant != 1
                    else (
                        "selected-fan provenance has no integral local "
                        "anticanonical Cartier support"
                    )
                )
                return unavailable(
                    reason_code,
                    reason,
                    local_cone=local_cone,
                    ambient_cone_provenance=provenance,
                    ambient_cone_determinants=ambient_determinants,
                )
            restricted = tuple(
                sum(
                    int(invariant_basis[row, column])
                    * _exact_fraction(ambient_data[row])
                    for row in range(4)
                )
                for column in range(fixed_rank)
            )
            if any(value.denominator != 1 for value in restricted):
                reason_code = (
                    "zero_dimensional_ambient_provenance_cone_not_smooth"
                    if ambient_determinant != 1
                    else "zero_dimensional_restricted_cartier_unavailable"
                )
                reason = (
                    "selected-fan provenance contains a non-unimodular ambient cone"
                    if ambient_determinant != 1
                    else "the restricted anticanonical Cartier support is not integral"
                )
                return unavailable(
                    reason_code,
                    reason,
                    local_cone=local_cone,
                    ambient_cone_provenance=provenance,
                    ambient_cone_determinants=ambient_determinants,
                )
            restricted_supports.append(restricted)
            provenance.append(
                {
                    "auxiliary_cone": [list(ray) for ray in rays],
                    "ambient_cone": [list(ray) for ray in ambient_rays],
                    "ambient_cone_determinant": int(ambient_determinant),
                    "ambient_cone_smooth_unimodular": ambient_determinant == 1,
                    "ambient_cartier_support": [
                        _fraction_to_json(value) for value in ambient_data
                    ],
                    "restricted_cartier_support": [
                        _fraction_to_json(value) for value in restricted
                    ],
                }
            )

    if any(support != restricted_supports[0] for support in restricted_supports[1:]):
        return unavailable(
            "zero_dimensional_restricted_cartier_unavailable",
            "selected-fan provenance gives inconsistent restricted Cartier support",
            local_cone=local_cone,
            ambient_cone_provenance=provenance,
            ambient_cone_determinants=ambient_determinants,
        )

    smooth = determinant == 1
    ambient_smooth = bool(ambient_determinants) and all(
        value == 1 for value in ambient_determinants
    )
    evidence_certified = ambient_smooth and smooth
    evidence_reason_code = (
        None
        if evidence_certified
        else "zero_dimensional_ambient_provenance_cone_not_smooth"
        if not ambient_smooth
        else "zero_dimensional_local_fan_not_smooth"
    )
    evidence_reason = (
        None
        if evidence_certified
        else (
            "selected-FRST ambient provenance contains a non-unimodular cone: "
            f"determinants={ambient_determinants}"
        )
        if not ambient_smooth
        else (
            "the invariant fixed-cone determinant is non-unimodular: "
            f"determinant={determinant}"
        )
    )
    return {
        **base,
        "status": "certified" if evidence_certified else "unavailable",
        "reason_code": evidence_reason_code,
        "reason": evidence_reason,
        "local_cone_determinant": int(determinant),
        "ambient_cone_determinants": ambient_determinants,
        "ambient_provenance_smoothness": {
            "status": "certified" if ambient_smooth else "rejected",
            "all_unimodular": ambient_smooth,
            "method": "exact_selected_frst_provenance_cone_determinants",
            "determinants": ambient_determinants,
        },
        "local_cone": local_cone,
        "local_smoothness": {
            "status": "certified" if smooth else "rejected",
            "smooth": smooth,
            "method": "exact_unimodular_local_cone_determinant",
            "determinant": int(determinant),
        },
        "restricted_cartier": {
            "status": "certified",
            "integral": True,
            "method": "exact_restriction_of_anticanonical_cartier_support",
            "support": [
                _fraction_to_json(value) for value in restricted_supports[0]
            ],
            "provenance": provenance,
        },
    }


def _zero_dimensional_component_local_certificate(auxiliary_fan, matrix, component):
    """Require explicit local smoothness and Cartier evidence for a point.

    A contained zero-dimensional component is not certified by its dimension
    alone.  Callers must supply a local certificate with integral restricted
    Cartier data, smooth selected-FRST ambient provenance cones, and a
    unimodular invariant local cone.  Determinant-two and determinant-four
    ambient or invariant cones remain unavailable; no ordinary-Euler point
    contribution is inferred for them.
    """
    evidence = component.get("zero_dimensional_local_smoothness_cartier_evidence")
    base = {
        "fixed_toric_dimension": 0,
        "local_evidence_schema": "cyaxiverse-zero-dimensional-local-evidence-1.0",
    }
    if not isinstance(evidence, dict):
        return {
            **base,
            "status": "unavailable",
            "reason_code": "missing_zero_dimensional_local_evidence",
            "reason": (
                "contained zero-dimensional component requires explicit local "
                "smoothness and Cartier evidence"
            ),
        }
    if evidence.get("status") != "certified":
        return {
            **base,
            "status": "unavailable",
            "reason_code": evidence.get(
                "reason_code", "zero_dimensional_local_evidence_unavailable"
            ),
            "reason": evidence.get(
                "reason", "local smoothness/Cartier evidence is not certified"
            ),
            "local_evidence": evidence,
    }

    # Production-generated evidence carries separate ambient provenance
    # determinants and the invariant fixed-cone determinant.  Keep the
    # legacy scalar fallback for the hand-authored unit fixtures, whose
    # contract predates this provenance split and intentionally supplies only
    # one local-certificate determinant.
    generated_provenance = "ambient_cone_determinants" in evidence
    if not generated_provenance and auxiliary_fan:
        return {
            **base,
            "status": "unavailable",
            "reason_code": "zero_dimensional_ambient_provenance_unavailable",
            "reason": (
                "production zero-dimensional evidence lacks exact ambient "
                "selected-FRST provenance determinants"
            ),
            "local_evidence": evidence,
        }
    if generated_provenance:
        ambient_determinants = evidence.get("ambient_cone_determinants")
        if not isinstance(ambient_determinants, (list, tuple)) or not ambient_determinants:
            return {
                **base,
                "status": "unavailable",
                "reason_code": "zero_dimensional_ambient_provenance_unavailable",
                "reason": (
                    "contained zero-dimensional component lacks exact ambient "
                    "selected-FRST provenance determinants"
                ),
                "local_evidence": evidence,
            }
        parsed_ambient_determinants = []
        for value in ambient_determinants:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                return {
                    **base,
                    "status": "unavailable",
                    "reason_code": "zero_dimensional_ambient_provenance_unavailable",
                    "reason": "ambient provenance determinant is not an exact integer",
                    "local_evidence": evidence,
                }
            parsed_ambient_determinants.append(abs(int(value)))
        ambient_smoothness = evidence.get("ambient_provenance_smoothness", {})
        if any(value != 1 for value in parsed_ambient_determinants):
            return {
                **base,
                "status": "unavailable",
                "reason_code": "zero_dimensional_ambient_provenance_cone_not_smooth",
                "reason": (
                    "selected-FRST ambient provenance contains a non-unimodular "
                    f"local cone: determinants={parsed_ambient_determinants}"
                ),
                "local_evidence": evidence,
            }
        if (
            not isinstance(ambient_smoothness, dict)
            or ambient_smoothness.get("status") != "certified"
            or ambient_smoothness.get("all_unimodular") is not True
            or ambient_smoothness.get("determinants") != parsed_ambient_determinants
        ):
            return {
                **base,
                "status": "unavailable",
                "reason_code": "zero_dimensional_ambient_provenance_unavailable",
                "reason": "ambient provenance smoothness evidence is incomplete",
                "local_evidence": evidence,
            }
        determinant = evidence.get("local_cone_determinant")
    else:
        determinant = evidence.get("ambient_cone_determinant")
    if isinstance(determinant, bool) or not isinstance(
        determinant, (int, np.integer)
    ):
        determinant = None
    else:
        determinant = abs(int(determinant))
    if determinant != 1:
        return {
            **base,
            "status": "unavailable",
            "reason_code": "zero_dimensional_local_fan_not_smooth",
            "reason": (
                "contained zero-dimensional component lacks a unimodular local "
                "cone; singular quotient points are unavailable"
            ),
            "local_evidence": evidence,
        }
    smoothness = evidence.get("local_smoothness", {})
    cartier = evidence.get("restricted_cartier", {})
    if (
        not isinstance(smoothness, dict)
        or smoothness.get("status") != "certified"
        or smoothness.get("smooth") is not True
    ):
        return {
            **base,
            "status": "unavailable",
            "reason_code": "zero_dimensional_local_smoothness_unavailable",
            "reason": "the local fixed stratum is not certified smooth",
            "local_evidence": evidence,
        }
    if (
        not isinstance(cartier, dict)
        or cartier.get("status") != "certified"
        or cartier.get("integral") is not True
    ):
        return {
            **base,
            "status": "unavailable",
            "reason_code": "zero_dimensional_restricted_cartier_unavailable",
            "reason": "the local restricted Cartier data is not certified integral",
            "local_evidence": evidence,
        }
    return {
        **base,
        "status": "certified",
        "reason_code": None,
        "reason": None,
        "local_evidence": evidence,
        "fixed_cone_lattice": {
            "source_ray_sublattice_saturated": True,
            "local_cone_determinant": int(determinant),
            "smooth_unimodular": True,
        },
    }


def _positive_component_section_certificate(
    auxiliary_fan, matrix, component, *, restricted_support=None,
    allow_contained=False,
):
    """Certify Sec. 4.6 for a positive-dimensional non-vanishing component.

    The component is the toric orbit closure of ``sigma`` in the auxiliary
    fixed fan.  For dimensions one through three, construct its star fan in
    the exact quotient lattice, restrict the ambient anti-canonical Cartier
    data, and apply two source-matched tests:

    * nefness: every local support-function representative lies in the
      restricted divisor polytope, equivalently the support function is
      convex on the complete fan;
    * orbifold avoidance: for each non-smooth cone with positive-dimensional
      orbit, the **actual invariant restricted support** face has exactly one
      lattice point.  More than one point gives a non-monomial Laurent
      restriction, which generically has a zero on that orbit.

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

    if restricted_support is None:
        restricted_support = component.get("invariant_restricted_support")

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
    if not isinstance(restricted_support, dict) or restricted_support.get("status") != "certified":
        return unavailable(
            "missing_invariant_restricted_support",
            "transverse nondegeneracy requires the exact invariant restricted support",
            restricted_support=restricted_support,
        )
    contained_support = restricted_support.get("restriction_identically_zero") is True
    if contained_support and not allow_contained:
        return unavailable(
            "component_is_contained_in_hypersurface",
            "a contained component cannot use the transverse section certificate",
            restricted_support=restricted_support,
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
    source_cone_sublattice_saturated = _sublattice_is_saturated(sigma_matrix)
    # The orbit closure uses the saturated real span of sigma.  Its character
    # lattice is therefore the exact integer annihilator below, even when the
    # source simplicial cone has a finite quotient index.  This does not make
    # a contained component smooth; callers retain the separate smooth-only
    # boundary for that branch.
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
    anchor_record = maximal_records[0]
    anchor_matrix = _SympyMatrix([list(ray) for ray in anchor_record["rays"]])
    anchor_rhs = _SympyMatrix([
        -_exact_fraction(coefficients[ray]) for ray in anchor_record["rays"]
    ])
    newton_origin = tuple(
        _exact_fraction(value)
        for value in (anchor_matrix.inv() * anchor_rhs)
    )
    actual_newton_face = []
    for item in restricted_support.get("newton_face", []):
        actual_newton_face.append([
            _fraction_to_json(
                newton_origin[index] + _exact_fraction(value)
            )
            for index, value in enumerate(item)
        ])
    restricted_support_with_origin = dict(restricted_support)
    restricted_support_with_origin["newton_face"] = actual_newton_face
    restricted_support_with_origin["newton_origin"] = [
        _fraction_to_json(value) for value in newton_origin
    ]
    fan_data = {
        **base,
        "fixed_cone_lattice": {
            "source_ray_sublattice_saturated": source_cone_sublattice_saturated,
            "quotient_character_lattice_saturated": True,
            "method": "exact_integer_annihilator_of_saturated_real_span",
        },
        "quotient_annihilator": np.asarray(quotient_annihilator, dtype=int).tolist(),
        "quotient_rays": [list(ray) for ray in sorted(coefficients)],
        "quotient_maximal_cones": [
            [list(ray) for ray in record["rays"]] for record in maximal_records
        ],
        "restricted_anticanonical_coefficients": [
            {"ray": list(ray), "coefficient": _fraction_to_json(coefficients[ray])}
            for ray in sorted(coefficients)
        ],
        "restricted_support": restricted_support_with_origin,
        "section_genericity": {
            "status": "certified",
            "scope": "generic_member_of_actual_invariant_restricted_support",
            "nondegeneracy": "required_on_each_full_dimensional_orbit_newton_face",
            "method": "exact_support_face_plus_eq_4.42_coefficient_covariance",
            "support_count": len(restricted_support.get("support", [])),
            "newton_face": actual_newton_face,
        },
    }
    if allow_contained:
        # Contained components need only their exact toric fan/Cartier data
        # for the smooth Chern/Eq. (4.50) branch.  They must never trigger a
        # reconstructed complete section line system.
        contained_singular_cones = []
        all_faces = {frozenset()}
        for record in maximal_records:
            rays = record["rays"]
            for size in range(1, component_dimension + 1):
                all_faces.update(
                    frozenset(face)
                    for face in itertools.combinations(rays, size)
                )
        contained_singular_cones = [
            tuple(sorted(cone))
            for cone in all_faces
            if cone and not _cone_is_smooth_in_lattice(
                tuple(cone), component_dimension
            )
        ]
        fan_data["section_genericity"] = {
            "status": "not_applicable",
            "scope": "contained_component_toric_fan_only",
            "reason": "Eq. (4.42) support is empty; no transverse section is evaluated",
        }
        if contained_singular_cones:
            return {
                **fan_data,
                "status": "unavailable",
                "reason_code": "singular_contained_component_fan",
                "reason": "a contained component requires a smooth toric fan",
                "nefness": nefness,
                "orbifold_intersection": {
                    "status": "unavailable",
                    "reason_code": "singular_contained_component_fan",
                    "reason": "a contained component requires a smooth toric fan",
                    "singular_cones": [
                        [list(ray) for ray in cone]
                        for cone in contained_singular_cones
                    ],
                },
            }
        return {
            **fan_data,
            "status": "certified",
            "reason_code": None,
            "reason": None,
            "nefness": nefness,
            "orbifold_intersection": {
                "status": "not_applicable",
                "avoided": None,
                "method": "contained_component_smooth_fan_gate",
            },
        }
    support_points = tuple(
        tuple(_exact_fraction(value) for value in item)
        for item in actual_newton_face
    )
    if not support_points:
        return {
            **fan_data,
            "status": "unavailable",
            "reason_code": "empty_invariant_restricted_support",
            "reason": "a transverse component has no invariant restricted monomial",
            "nefness": nefness,
            "orbifold_intersection": {
                "status": "unavailable",
                "reason_code": "empty_invariant_restricted_support",
                "reason": "a transverse component has no invariant restricted monomial",
            },
        }
    restricted_support_coefficients = {
        ray: -min(_fraction_dot(point, ray) for point in support_points)
        for ray in coefficients
    }
    fan_data["restricted_support_coefficients"] = [
        {
            "ray": list(ray),
            "coefficient": _fraction_to_json(restricted_support_coefficients[ray]),
        }
        for ray in sorted(restricted_support_coefficients)
    ]
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

    section_points = support_points
    restricted_support_coefficients = {
        ray: -min(_fraction_dot(point, ray) for point in section_points)
        for ray in coefficients
    }
    fan_data["restricted_support_coefficients"] = [
        {
            "ray": list(ray),
            "coefficient": _fraction_to_json(restricted_support_coefficients[ray]),
        }
        for ray in sorted(restricted_support_coefficients)
    ]
    all_section_points = section_points
    singular_strata = []
    for cone in positive_dimensional_singular_cones:
        face_points = tuple(
            point
            for point in all_section_points
            if all(
                _fraction_dot(point, ray) == -restricted_support_coefficients[ray]
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
    section_cache=None,
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
    zero_dimensional_component_checks = []
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
        if component["f_vanishes_identically"] is None:
            unavailable_reasons.append(
                "fixed-component containment is undetermined ["
                + str(component.get("containment_method"))
                + "]"
            )
            continue
        if not component["f_vanishes_identically"]:
            if _matrix_false_support_incomplete(component):
                section_check = {
                    "status": "unavailable",
                    "reason_code": "support_incomplete_for_matrix_false",
                    "reason": (
                        "matrix False used incomplete dual rows; the full restricted "
                        "support is unavailable"
                    ),
                    "component": {
                        "sigma_rays": component["sigma_rays"],
                        "sigma_dimension": component["sigma_dimension"],
                        "nu": component["nu"],
                    },
                }
                positive_component_checks.append(section_check)
                unavailable_reasons.append(
                    "section certificate unavailable [support_incomplete_for_matrix_false]"
                )
                continue
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
                # The section certificate depends only on (auxiliary_fan,
                # matrix, sigma_rays, nu); torus_shift and lambda_f do not
                # enter it, so cache it per matrix keyed by the component's
                # (sigma_rays, nu). Bit-for-bit identical (finding F3). The
                # cached value never carries the per-call "component" field, so
                # a fresh copy is returned before that field is attached.
                section_key = (
                    tuple(tuple(int(value) for value in ray) for ray in component["sigma_rays"]),
                    json.dumps(component["nu"], sort_keys=True),
                )
                if section_cache is not None and section_key in section_cache:
                    section_check = dict(section_cache[section_key])
                else:
                    section_check = _positive_component_section_certificate(
                        auxiliary_fan,
                        matrix,
                        component,
                    )
                    if section_cache is not None:
                        section_cache[section_key] = section_check
                    section_check = dict(section_check)
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
        elif component["fixed_toric_dimension"] == 0:
            local_check = _zero_dimensional_component_local_certificate(
                auxiliary_fan, matrix, component
            )
            zero_dimensional_component_checks.append({
                **local_check,
                "component": {
                    "sigma_rays": component["sigma_rays"],
                    "sigma_dimension": component["sigma_dimension"],
                    "nu": component["nu"],
                },
            })
            if local_check["status"] != "certified":
                unavailable_reasons.append(
                    "contained zero-dimensional fixed component requires "
                    "local smoothness/Cartier evidence "
                    f"[{local_check['reason_code']}]: {local_check['reason']}"
                )
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
            "zero_dimensional_component_checks": zero_dimensional_component_checks,
        }
    if unavailable_reasons:
        return {
            "status": "smoothness_verification_unavailable",
            "verdict": "not_verified",
            "method": "source_eq_4.45_4.48_4.50_sec_4.6_checks",
            "reasons": sorted(set(unavailable_reasons)),
            "dual_vertex_parity": parity,
            "positive_component_checks": positive_component_checks,
            "zero_dimensional_component_checks": zero_dimensional_component_checks,
        }
    return {
        "status": "smooth",
        "verdict": "smooth",
        "method": "source_eq_4.45_4.48_4.50_sec_4.6_checks",
        "reasons": [],
        "dual_vertex_parity": parity,
        "positive_component_checks": positive_component_checks,
        "zero_dimensional_component_checks": zero_dimensional_component_checks,
    }
