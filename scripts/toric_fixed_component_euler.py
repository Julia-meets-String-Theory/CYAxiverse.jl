"""Exact ordinary Euler characteristics for toric fixed components.

The toric Chern-class and adjunction formulas used here are derived in
``validation/orientifold_exact_action_formula_ledger_20260822.md``.  This
Contained components use the smooth toric Chern calculation only. Transverse
components may additionally use toric-orbit additivity and Khovanskii's
Newton-polytope formula. These are ordinary topological Euler
characteristics, never stringy/orbifold Euler characteristics.
"""

from fractions import Fraction
import itertools
import math

import numpy as np
from sympy import Matrix as SympyMatrix
from sympy import linsolve
from sympy.matrices.normalforms import smith_normal_form
from sympy.polys.domains import ZZ

try:
    from scipy.spatial import ConvexHull as _ConvexHull
except ImportError:  # pragma: no cover - exact fallback remains available
    _ConvexHull = None

import orientifold_general_l_geometry as general_l


MAX_EHRHART_SEARCH_POINTS = 3_000_000


def _fraction(value):
    if isinstance(value, dict):
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    return Fraction(value)


def _degree_monomials(variable_count, degree):
    for indices in itertools.combinations_with_replacement(range(variable_count), degree):
        exponent = [0] * variable_count
        for index in indices:
            exponent[index] += 1
        yield tuple(exponent)


def _add_exponent(left, right):
    return tuple(a + b for a, b in zip(left, right))


def _support(exponent):
    return frozenset(index for index, power in enumerate(exponent) if power)


def _intersection_functional(rays, maximal_cones):
    """Return the exact degree map of a smooth complete toric fan.

    Linear equivalences and Stanley--Reisner vanishing determine all degree-d
    monomials.  Maximal square-free monomials are normalized by their exact
    lattice determinant.  A non-smooth cone is terminal unavailable.
    """
    rays = tuple(tuple(int(value) for value in ray) for ray in rays)
    dimension = len(rays[0]) if rays else 0
    ray_index = {ray: index for index, ray in enumerate(rays)}
    cone_indices = []
    for cone in maximal_cones:
        indices = tuple(ray_index[tuple(int(value) for value in ray)] for ray in cone)
        exact_determinant = abs(int(SympyMatrix([rays[i] for i in indices]).det()))
        if exact_determinant != 1:
            raise ValueError("quotient fan is not smooth unimodular")
        cone_indices.append(frozenset(indices))
    monomials = tuple(_degree_monomials(len(rays), dimension))
    position = {monomial: index for index, monomial in enumerate(monomials)}
    rows = []
    values = []

    for monomial in monomials:
        if not any(_support(monomial).issubset(cone) for cone in cone_indices):
            row = [Fraction(0)] * len(monomials)
            row[position[monomial]] = Fraction(1)
            rows.append(row)
            values.append(Fraction(0))

    if dimension:
        for lower in _degree_monomials(len(rays), dimension - 1):
            for coordinate in range(dimension):
                row = [Fraction(0)] * len(monomials)
                for index, ray in enumerate(rays):
                    exponent = list(lower)
                    exponent[index] += 1
                    row[position[tuple(exponent)]] += Fraction(ray[coordinate])
                rows.append(row)
                values.append(Fraction(0))

    for cone in cone_indices:
        exponent = tuple(1 if index in cone else 0 for index in range(len(rays)))
        row = [Fraction(0)] * len(monomials)
        row[position[exponent]] = Fraction(1)
        rows.append(row)
        values.append(Fraction(1))

    solution_set = linsolve((SympyMatrix(rows), SympyMatrix(values)))
    solutions = list(solution_set)
    if len(solutions) != 1 or any(value.free_symbols for value in solutions[0]):
        raise ValueError("smooth complete fan degree map is not uniquely determined")
    degrees = {
        monomial: Fraction(int(value.p), int(value.q))
        for monomial, value in zip(monomials, solutions[0])
    }
    return degrees


def _multiply(left, right, maximum_degree):
    result = {}
    for left_exp, left_value in left.items():
        for right_exp, right_value in right.items():
            exponent = _add_exponent(left_exp, right_exp)
            if sum(exponent) <= maximum_degree:
                result[exponent] = result.get(exponent, Fraction(0)) + left_value * right_value
    return {key: value for key, value in result.items() if value}


def _power(polynomial, power, maximum_degree, variable_count):
    result = {(0,) * variable_count: Fraction(1)}
    for _ in range(power):
        result = _multiply(result, polynomial, maximum_degree)
    return result


def component_euler_from_certificate(certificate, *, contained):
    """Evaluate a certified component with exact smooth toric intersection.

    If ``contained`` is true this returns ``int_C c_top(TC)``.  Otherwise it
    returns ``int_C [c(TC) D/(1+D)]_dim(C)`` for the restricted anticanonical
    divisor.  Dimension-zero non-contained components have empty generic
    intersection and Euler characteristic zero.
    """
    if certificate.get("status") != "certified":
        raise ValueError("fixed-component fan/Cartier certificate is unavailable")
    dimension = int(certificate["fixed_toric_dimension"])
    if dimension == 0:
        return 1 if contained else 0
    rays = tuple(tuple(int(value) for value in ray) for ray in certificate["quotient_rays"])
    maximal_cones = certificate["quotient_maximal_cones"]
    degrees = _intersection_functional(rays, maximal_cones)
    variable_count = len(rays)
    one = (0,) * variable_count
    total_chern = {one: Fraction(1)}
    for index in range(variable_count):
        exponent = [0] * variable_count
        exponent[index] = 1
        total_chern = _multiply(
            total_chern,
            {one: Fraction(1), tuple(exponent): Fraction(1)},
            dimension,
        )
    if contained:
        integrand = {key: value for key, value in total_chern.items() if sum(key) == dimension}
    else:
        coefficient_rows = certificate.get(
            "restricted_support_coefficients",
            certificate["restricted_anticanonical_coefficients"],
        )
        coefficients = {
            tuple(item["ray"]): _fraction(item["coefficient"])
            for item in coefficient_rows
        }
        divisor = {}
        for index, ray in enumerate(rays):
            exponent = [0] * variable_count
            exponent[index] = 1
            divisor[tuple(exponent)] = coefficients[ray]
        integrand = {}
        for power in range(1, dimension + 1):
            chern_degree = dimension - power
            chern_piece = {key: value for key, value in total_chern.items() if sum(key) == chern_degree}
            term = _multiply(chern_piece, _power(divisor, power, dimension, variable_count), dimension)
            sign = Fraction(1 if power % 2 else -1)
            for exponent, value in term.items():
                integrand[exponent] = integrand.get(exponent, Fraction(0)) + sign * value
    answer = sum(value * degrees[exponent] for exponent, value in integrand.items())
    if answer.denominator != 1:
        raise ValueError("component Euler characteristic is nonintegral")
    return int(answer)


def _matrix_rank(vectors):
    if not vectors:
        return 0
    return int(SympyMatrix(vectors).rank())


def _complete_difference_lattice_evidence(differences, rank):
    """Compute the exact index of the complete integer difference span."""
    differences = np.asarray(differences, dtype=int)
    if differences.ndim != 2:
        raise ValueError("difference matrix must be two-dimensional")
    rank = int(rank)
    smith = smith_normal_form(SympyMatrix(differences.tolist()), domain=ZZ)
    invariants = [
        abs(int(smith[index, index]))
        for index in range(min(smith.rows, smith.cols))
        if smith[index, index] != 0
    ]
    if len(invariants) != rank:
        raise ValueError("difference lattice rank does not match Newton-face rank")
    return {
        "rank": rank,
        "index": int(math.prod(invariants)),
        "smith_invariants": invariants,
        "method": "exact_smith_normal_form_of_complete_integer_difference_matrix",
    }


def _cross_2d(origin, left, right):
    return ((left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0]))


def _convex_hull_2d(points):
    points = sorted(set(tuple(Fraction(value) for value in point) for point in points))
    if len(points) <= 1:
        return tuple(points)

    def half(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and _cross_2d(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result

    return tuple(half(points)[:-1] + half(reversed(points))[:-1])


def _support_polytope_vertices(points, dimension):
    """Return exact Newton-support vertices with a bounded hull prefilter."""

    points = tuple(sorted(set(tuple(Fraction(value) for value in point) for point in points)))
    if len(points) <= dimension:
        return points
    if dimension == 1:
        return (min(points), max(points))
    if dimension == 2:
        return _convex_hull_2d(points)
    if dimension != 3 or _ConvexHull is None:
        return points
    try:
        hull = _ConvexHull(np.asarray(points, dtype=float))
        candidates = tuple(points[index] for index in sorted(set(int(index) for index in hull.vertices)))
        # The candidate list is selected numerically, but all later faces and
        # volumes use these exact rational coordinates.  Recheck that every
        # support point lies in the candidate hull before accepting it.
        facets = _hull_3d_facets(candidates)
        centre = tuple(sum(point[axis] for point in candidates) / len(candidates) for axis in range(3))
        if all(
            all(
                (sum(facet["normal"][axis] * point[axis] for axis in range(3)) - facet["offset"])
                * (sum(facet["normal"][axis] * centre[axis] for axis in range(3)) - facet["offset"]) >= 0
                for facet in facets
            )
            for point in points
        ):
            return candidates
    except (ValueError, RuntimeError, TypeError):
        pass
    return points


def _primitive_hyperplane(normal, offset):
    values = [int(value) for value in normal] + [int(offset)]
    divisor = math.gcd(*[abs(value) for value in values])
    values = [value // divisor for value in values]
    first = next(value for value in values if value)
    if first < 0:
        values = [-value for value in values]
    return tuple(values[:-1]), values[-1]


def _det3(rows):
    return Fraction(SympyMatrix(rows).det())


def _hull_3d_facets(points):
    """Return exact supporting planes and cyclic vertex orders."""
    points = tuple(tuple(Fraction(value) for value in point) for point in points)
    planes = {}
    for indices in itertools.combinations(range(len(points)), 3):
        a, b, c = (points[index] for index in indices)
        left = tuple(b[i] - a[i] for i in range(3))
        right = tuple(c[i] - a[i] for i in range(3))
        normal = (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )
        if not any(normal):
            continue
        if any(value.denominator != 1 for value in normal):
            raise ValueError("nonintegral lattice hull normal")
        normal = tuple(int(value) for value in normal)
        offset = sum(normal[i] * a[i] for i in range(3))
        if offset.denominator != 1:
            raise ValueError("nonintegral lattice hull offset")
        signed = [sum(normal[i] * point[i] for i in range(3)) - offset for point in points]
        if not (all(value >= 0 for value in signed) or all(value <= 0 for value in signed)):
            continue
        key = _primitive_hyperplane(normal, int(offset))
        planes[key] = tuple(index for index, value in enumerate(signed) if value == 0)

    facets = []
    for (normal, offset), indices in sorted(planes.items()):
        projected = None
        for pair in ((0, 1), (0, 2), (1, 2)):
            candidate = [(points[index][pair[0]], points[index][pair[1]]) for index in indices]
            differences = [
                [point[axis] - candidate[0][axis] for axis in range(2)]
                for point in candidate[1:]
            ]
            if _matrix_rank(differences) == 2:
                projected = candidate
                break
        if projected is None:
            raise ValueError("three-dimensional hull facet is not two-dimensional")
        hull = _convex_hull_2d(projected)
        lookup = {point: index for point, index in zip(projected, indices)}
        facets.append({
            "normal": normal,
            "offset": offset,
            "vertices": tuple(lookup[point] for point in hull),
        })
    if len(facets) < 4:
        raise ValueError("three-dimensional lattice hull is incomplete")
    return tuple(facets)


def normalized_lattice_volume(points):
    """Return ``k!`` times covolume-one Euclidean volume exactly.

    Khovanskii's Euler formula consumes this normalized volume directly; no
    additional factorial is applied by callers.
    """
    points = tuple(sorted(set(tuple(Fraction(value) for value in point) for point in points)))
    dimension = len(points[0]) if points else 0
    if dimension == 0:
        return 1
    differences = [
        [point[axis] - points[0][axis] for axis in range(dimension)]
        for point in points[1:]
    ]
    if _matrix_rank(differences) != dimension:
        raise ValueError("Newton polytope is not full-dimensional")
    if dimension == 1:
        answer = max(point[0] for point in points) - min(point[0] for point in points)
    elif dimension == 2:
        hull = _convex_hull_2d(points)
        answer = abs(sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[index][1] * hull[(index + 1) % len(hull)][0]
            for index in range(len(hull))
        ))
    elif dimension == 3:
        facets = _hull_3d_facets(points)
        centre = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
        answer = Fraction(0)
        for facet in facets:
            vertices = facet["vertices"]
            anchor = points[vertices[0]]
            for index in range(1, len(vertices) - 1):
                triangle = (anchor, points[vertices[index]], points[vertices[index + 1]])
                answer += abs(_det3([
                    [vertex[axis] - centre[axis] for axis in range(3)]
                    for vertex in triangle
                ]))
    else:
        raise ValueError("Newton-polytope volume is supported only through dimension three")
    if answer.denominator != 1:
        raise ValueError("normalized lattice volume is nonintegral")
    return int(answer)


def _lattice_points_in_dilate(points, dilation):
    dimension = len(points[0])
    if dilation == 0:
        return 1
    scaled = tuple(tuple(dilation * value for value in point) for point in points)
    bounds = [
        range(math.floor(min(point[axis] for point in scaled)),
              math.ceil(max(point[axis] for point in scaled)) + 1)
        for axis in range(dimension)
    ]
    search_size = math.prod(len(values) for values in bounds)
    if search_size > MAX_EHRHART_SEARCH_POINTS:
        raise ValueError(
            f"independent Ehrhart search requires {search_size} points, above "
            f"the bound {MAX_EHRHART_SEARCH_POINTS}"
        )
    if dimension == 1:
        return len(bounds[0])
    if dimension == 2:
        hull = _convex_hull_2d(scaled)
        return sum(
            all(_cross_2d(hull[index], hull[(index + 1) % len(hull)], candidate) >= 0
                for index in range(len(hull)))
            for candidate in itertools.product(*bounds)
        )
    facets = _hull_3d_facets(scaled)
    centre = tuple(sum(point[axis] for point in scaled) / len(scaled) for axis in range(3))
    return sum(
        all(
            (sum(facet["normal"][axis] * candidate[axis] for axis in range(3))
             - facet["offset"])
            * (sum(facet["normal"][axis] * centre[axis] for axis in range(3))
               - facet["offset"]) >= 0
            for facet in facets
        )
        for candidate in itertools.product(*bounds)
    )


def normalized_lattice_volume_ehrhart(points):
    """Independent exact normalized volume from the kth Ehrhart difference."""
    points = tuple(sorted(set(tuple(Fraction(value) for value in point) for point in points)))
    dimension = len(points[0]) if points else 0
    if dimension == 0:
        return 1
    counts = [_lattice_points_in_dilate(points, dilation) for dilation in range(dimension + 1)]
    return sum(
        (-1) ** (dimension - dilation) * math.comb(dimension, dilation) * count
        for dilation, count in enumerate(counts)
    )


def _all_fan_cones(maximal_cones):
    cones = {frozenset()}
    for maximal in maximal_cones:
        maximal = tuple(tuple(int(value) for value in ray) for ray in maximal)
        for size in range(1, len(maximal) + 1):
            cones.update(frozenset(face) for face in itertools.combinations(maximal, size))
    return tuple(sorted((tuple(sorted(cone)) for cone in cones), key=lambda cone: (len(cone), cone)))


def _face_coordinates(section_points, cone, dimension):
    face = tuple(
        point for point in section_points
        if all(general_l._fraction_dot(point, ray) == -coefficient for ray, coefficient in cone)
    )
    if not face:
        raise ValueError("restricted section Newton face is empty")
    rays = np.asarray([ray for ray, _ in cone], dtype=int)
    basis = general_l._integer_kernel_basis(rays.reshape((len(cone), dimension)))
    if basis.shape != (dimension, dimension - len(cone)):
        raise ValueError("orbit character lattice basis has the wrong rank")
    origin = face[0]
    coordinates = []
    for point in face:
        difference = []
        for value, reference in zip(point, origin):
            value = value - reference
            if value.denominator != 1:
                raise ValueError("Newton-face difference is not integral")
            difference.append(int(value))
        coordinate = general_l._integer_coordinates(basis, difference)
        if coordinate is None:
            raise ValueError("Newton-face difference is not in the saturated orbit character lattice")
        coordinates.append(tuple(int(value) for value in coordinate))
    return tuple(sorted(set(coordinates)))


def transverse_component_euler_orbit(certificate):
    """Compute ordinary Euler by exact toric-orbit Newton-face additivity."""
    if certificate.get("status") != "certified":
        raise ValueError("fixed-component fan/Cartier certificate is unavailable")
    nefness = certificate.get("nefness", {})
    if nefness.get("status") != "certified" or nefness.get("nef") is not True:
        raise ValueError("orbit Euler requires certified nef restricted divisor")
    avoidance = certificate.get("orbifold_intersection", {})
    if avoidance.get("status") != "certified" or avoidance.get("avoided") is not True:
        raise ValueError("orbit Euler requires certified orbifold-stratum avoidance")
    genericity = certificate.get("section_genericity", {})
    if genericity.get("status") != "certified":
        raise ValueError("orbit Euler requires certified generic section nondegeneracy")
    dimension = int(certificate["fixed_toric_dimension"])
    if dimension == 0:
        return 0, {"ordinary_euler": 0, "orbits": []}
    rays = tuple(tuple(int(value) for value in ray) for ray in certificate["quotient_rays"])
    coefficient_rows = certificate.get("restricted_support_coefficients")
    if coefficient_rows is None:
        coefficient_rows = certificate.get("restricted_anticanonical_coefficients")
    restricted_support = certificate.get("restricted_support")
    if not isinstance(restricted_support, dict) or restricted_support.get("status") != "certified":
        raise ValueError("actual invariant restricted support is unavailable")
    if restricted_support.get("restriction_identically_zero") is True:
        raise ValueError("a contained component cannot be evaluated as transverse")
    support_points = restricted_support.get("newton_face", [])
    if not support_points:
        raise ValueError("actual invariant restricted support is empty")
    if coefficient_rows is None:
        coefficient_rows = [
            {
                "ray": list(ray),
                "coefficient": general_l._fraction_to_json(
                    -min(
                        general_l._fraction_dot(point, ray)
                        for point in support_points
                    )
                ),
            }
            for ray in rays
        ]
    coefficients = {
        tuple(item["ray"]): _fraction(item["coefficient"])
        for item in coefficient_rows
    }
    section_vertices = _support_polytope_vertices(
        support_points, dimension
    )
    section = {
        "status": "certified",
        "method": "actual_invariant_restricted_support_newton_face",
        "points": tuple(
            tuple(general_l._exact_fraction(value) for value in point)
            for point in support_points
        ),
        "vertices": tuple(
            tuple(general_l._exact_fraction(value) for value in point)
            for point in section_vertices
        ),
    }
    orbit_evidence = []
    total = 0
    for cone_rays in _all_fan_cones(certificate["quotient_maximal_cones"]):
        orbit_dimension = dimension - len(cone_rays)
        cone = tuple((ray, coefficients[ray]) for ray in cone_rays)
        face_lattice_points = _face_coordinates(section["points"], cone, dimension)
        face = _face_coordinates(section["vertices"], cone, dimension)
        affine_rank = _matrix_rank([
            [value - face[0][axis] for axis, value in enumerate(point)]
            for point in face[1:]
        ])
        difference_lattice = None
        if orbit_dimension == 0 or affine_rank == 0:
            chi = 0
            method = "nonzero_monomial_has_empty_zero_locus"
            geometric_volume = ehrhart_volume = None
        elif affine_rank != orbit_dimension:
            differences = np.asarray([
                [int(value - face_lattice_points[0][axis]) for axis, value in enumerate(point)]
                for point in face_lattice_points[1:]
            ], dtype=int).T
            difference_lattice = _complete_difference_lattice_evidence(
                differences, affine_rank
            )
            # Danilov--Khovanskii 1987 Section 5.1: the hypersurface is
            # Z' x (C*)^(k-r). Since k-r>0 and chi(C*)=0, its ordinary Euler
            # contribution is exactly zero. The complete support-difference
            # lattice can have finite index in its saturated affine span; that
            # index changes component or Hodge-Deligne data but not this Euler
            # contribution.
            chi = 0
            method = "lower_dimensional_newton_face_torus_product_zero_euler"
            geometric_volume = ehrhart_volume = None
        else:
            geometric_volume = normalized_lattice_volume(face)
            ehrhart_volume = normalized_lattice_volume_ehrhart(face)
            if geometric_volume != ehrhart_volume:
                raise ValueError("geometric and Ehrhart normalized volumes disagree")
            chi = (-1) ** (orbit_dimension - 1) * geometric_volume
            method = "khovanskii_signed_normalized_lattice_volume"
        orbit_record = {
            "cone": [list(ray) for ray in cone_rays],
            "orbit_dimension": orbit_dimension,
            "newton_face_affine_dimension": affine_rank,
            "newton_face_vertices": [list(point) for point in face],
            "newton_face_lattice_point_count": len(face_lattice_points),
            "normalized_lattice_volume": geometric_volume,
            "independent_ehrhart_normalized_lattice_volume": ehrhart_volume,
            "chi": chi,
            "method": method,
        }
        if difference_lattice is not None:
            orbit_record["newton_face_difference_lattice"] = difference_lattice
        orbit_evidence.append(orbit_record)
        total += chi
    return total, {
        "ordinary_euler": total,
        "method": "toric_orbit_additivity_and_khovanskii_normalized_volume",
        "normalization": "normalized_lattice_volume_equals_k_factorial_times_covolume_one_volume",
        "genericity": "generic_non_degenerate_section_of_actual_invariant_restricted_support",
        "restricted_support": restricted_support,
        "orbits": orbit_evidence,
    }


def exact_fixed_locus_euler(
    auxiliary_fan, matrix, components, *, fixed_surface_n_s_evidence=None
):
    """Sum certified disjoint maximal components, failing unavailable."""
    evidence = []
    total = 0
    first_failure = None

    def component_evidence(
        component,
        *,
        contained,
        dimension,
        surface_n_s=None,
        chi=None,
        certificate=None,
        method=None,
        orbit_evidence=None,
        status="unavailable",
        reason=None,
        reason_code=None,
    ):
        """Retain one status record for every enumerated source component."""
        return {
            "sigma_rays": component["sigma_rays"],
            "nu": component["nu"],
            "contained_in_hypersurface": contained,
            "containment_method": component.get("containment_method"),
            "invariant_restricted_support": component.get("invariant_restricted_support"),
            "cox_phase_witness": component.get("cox_phase_witness"),
            "fixed_cone_lattice": component.get("fixed_cone_lattice"),
            "ambient_component_dimension": dimension,
            "chi": chi,
            "fixed_surface_n_s": None if surface_n_s is None else int(surface_n_s),
            "certificate": certificate,
            "method": method,
            "orbit_evidence": orbit_evidence,
            "euler_status": status,
            "euler_reason_code": reason_code,
            "euler_reason": reason,
        }

    for component in components:
        dimension = int(component["fixed_toric_dimension"])
        containment_value = component.get("f_vanishes_identically")
        if containment_value is None:
            evidence.append(component_evidence(
                component,
                contained=False,
                dimension=dimension,
                reason="exact invariant restricted support is unavailable",
                reason_code="missing_invariant_restricted_support",
            ))
            if first_failure is None:
                first_failure = {
                    "reason": "exact invariant restricted support is unavailable",
                    "reason_code": "missing_invariant_restricted_support",
                }
            continue
        contained = containment_value is True
        surface_n_s = None
        if contained and dimension == 2:
            surface_table = fixed_surface_n_s_evidence or {}
            surface_n_s = surface_table.get(general_l._component_key(component))
            if surface_n_s is None:
                reason = "contained fixed surface lacks certified Moritz eq. 4.50 n_S=0 evidence"
                reason_code = "missing_fixed_surface_n_s_zero_evidence"
            elif int(surface_n_s) != 0:
                reason = (
                    "contained fixed surface has nonzero Moritz eq. 4.50 "
                    f"n_S={int(surface_n_s)}; source requires n_S=0"
                )
                reason_code = "fixed_surface_n_s_nonzero"
            else:
                reason = None
                reason_code = None
            if reason is not None:
                evidence.append(component_evidence(
                    component,
                    contained=contained,
                    dimension=dimension,
                    surface_n_s=surface_n_s,
                    reason=reason,
                    reason_code=reason_code,
                ))
                if first_failure is None:
                    first_failure = {
                        "reason": reason,
                        "reason_code": reason_code,
                    }
                continue
        if dimension == 0:
            if not contained:
                support = component.get("invariant_restricted_support", {})
                chart = support.get("chart", {}) if isinstance(support, dict) else {}
                if support.get("status") == "certified" and support.get("support") and chart.get("status") == "certified":
                    evidence.append(component_evidence(
                        component,
                        contained=contained,
                        dimension=dimension,
                        chi=0,
                        method="nonzero_invariant_monomial_has_empty_zero_locus",
                        status="computed",
                    ))
                    continue
                evidence.append(component_evidence(
                    component,
                    contained=contained,
                    dimension=dimension,
                    reason="non-vanishing zero-dimensional component lacks exact chart evidence",
                    reason_code="missing_invariant_restricted_support_chart",
                ))
                if first_failure is None:
                    first_failure = {
                        "reason": "non-vanishing zero-dimensional component lacks exact chart evidence",
                        "reason_code": "missing_invariant_restricted_support_chart",
                    }
                continue
            certificate = general_l._zero_dimensional_component_local_certificate(
                auxiliary_fan, matrix, component
            )
        else:
            certificate = general_l._positive_component_section_certificate(
                auxiliary_fan,
                matrix,
                component,
                allow_contained=contained,
            )
        if certificate.get("status") != "certified":
            evidence.append(component_evidence(
                component,
                contained=contained,
                dimension=dimension,
                surface_n_s=surface_n_s,
                certificate=certificate,
                reason=certificate.get("reason", "component certificate unavailable"),
                reason_code=certificate.get("reason_code"),
            ))
            if first_failure is None:
                first_failure = {
                    "reason": certificate.get("reason", "component certificate unavailable"),
                    "reason_code": certificate.get("reason_code"),
                }
            continue
        try:
            orbit_evidence = None
            if contained:
                lattice = certificate.get("fixed_cone_lattice", {})
                if lattice and lattice.get("source_ray_sublattice_saturated") is not True:
                    raise ValueError(
                        "contained component has a nonsaturated source fixed cone"
                    )
                chi = component_euler_from_certificate(certificate, contained=True)
                component_method = "smooth_complete_toric_chow_ring"
            else:
                chi, orbit_evidence = transverse_component_euler_orbit(certificate)
                component_method = "ordinary_euler_toric_orbit_stratification"
                # The orbit sum is authoritative for the actual invariant
                # support.  Do not reconstruct or cross-check against the
                # complete anticanonical line system here: that would discard
                # the Eq. (4.42) restricted Newton face on a nonsmooth or
                # support-sparse component.
        except ValueError as exc:
            evidence.append(component_evidence(
                component,
                contained=contained,
                dimension=dimension,
                surface_n_s=surface_n_s,
                certificate=certificate,
                reason=str(exc),
            ))
            if first_failure is None:
                first_failure = {"reason": str(exc), "reason_code": None}
            continue
        evidence.append(component_evidence(
            component,
            contained=contained,
            dimension=dimension,
            surface_n_s=surface_n_s,
            chi=chi,
            certificate=certificate,
            method=component_method,
            orbit_evidence=orbit_evidence,
            status="computed",
        ))
        total += chi
    if first_failure is not None:
        return {
            "status": "unavailable",
            "reason": first_failure["reason"],
            "reason_code": first_failure.get("reason_code"),
            "components": evidence,
        }
    return {
        "status": "computed",
        "chi_F_I": total,
        "components": evidence,
        "component_count": len(evidence),
        "certified_component_count": len(evidence),
        "method": "componentwise_smooth_chern_or_ordinary_euler_orbit_stratification",
        "component_union": "maximal_source_components_after_containment_removal",
    }
