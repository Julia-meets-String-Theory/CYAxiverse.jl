"""Generate CYTools data, including several FRSTs for rare polytopes.

``--n`` is the number of Calabi--Yau geometries requested for each h11, not
the number of distinct polytopes.  The script first fetches up to ``n``
favorable N-lattice polytopes.  If fewer exist (as for h11=491), it samples
several bounded, pseudorandom FRSTs of each available polytope instead of ever
attempting to enumerate all triangulations.

Each FRST has its own toric Kähler cone and stretched-cone tip.  A candidate is
saved only when its tip can be dilated to satisfy the control criterion of
arXiv:2309.01831, eq. (21).  Failed candidates are rejected and replaced by
new FRST samples up to a user-configurable attempt budget.

The default ``fair`` sampler delegates secondary-fan walks and flips to
CYTools. ``fast`` is available for explicitly biased coverage/training scans.
Kähler-cone quadratic programs prefer MOSEK when it is licensed and available; if
``$MOSEKLM_LICENSE_FILE`` is unset, a standard ``$HOME/mosek.lic`` is exposed
to the child solver without reading or copying the license contents.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import cytools
from cytools import Polytope, fetch_polytopes


SCHEMA_VERSION = "cyaxiverse-ks-cy3-v5"
MIN_CYTOOLS_VERSION = (1, 4, 0)
SOURCE_PAPER_SET = (
    "arXiv:2008.01730v1",  # fair secondary-fan/triangulation sampling
    "arXiv:2309.01831v1",  # stretched-cone potential-control criterion
    "arXiv:2305.06363v1",  # orientifold-compatible KS constructions
    "arXiv:2412.12012v1",  # fuzzy-axion QCD divisor-volume requirement
)


def configure_mosek_license():
    """Expose the conventional user MOSEK license path without embedding it."""
    configured = os.environ.get("MOSEKLM_LICENSE_FILE")
    if configured:
        source = "MOSEKLM_LICENSE_FILE"
    else:
        home_license = os.path.join(os.path.expanduser("~"), "mosek.lic")
        if not os.path.isfile(home_license):
            return {"configured": False, "activated": False, "source": None}
        os.environ["MOSEKLM_LICENSE_FILE"] = home_license
        source = "$HOME/mosek.lic"

    # CYTools caches this status, so refresh it after exposing a non-default
    # license path and before any cone optimization is requested.
    try:
        cytools.config.check_mosek_license(silent=True)
        activated = bool(cytools.config.mosek_is_activated())
    except (ImportError, OSError, RuntimeError):
        activated = False
    return {"configured": True, "activated": activated, "source": source}


def require_cytools_capabilities(sampling_scheme):
    """Fail early if the selected CYTools construction path is unavailable."""
    version = getattr(cytools, "version", None)
    if version is None:
        raise RuntimeError("CYTools does not expose its version; refusing to write data.")
    try:
        parsed_version = tuple(int(component) for component in version.split("."))
    except ValueError as exc:
        raise RuntimeError(f"Unparseable CYTools version {version!r}.") from exc
    if parsed_version < MIN_CYTOOLS_VERSION:
        required = ".".join(str(component) for component in MIN_CYTOOLS_VERSION)
        raise RuntimeError(
            f"CYTools {version} is too old for this generator; require >= {required}."
        )

    required = [
        (Polytope, "triangulate"),
        (Polytope, "random_triangulations_fast"),
        (Polytope, "random_triangulations_fair"),
    ]
    missing = [
        f"{owner.__name__}.{name}"
        for owner, name in required
        if not hasattr(owner, name)
    ]
    if sampling_scheme not in {"fast", "fair"}:
        raise ValueError(f"Unsupported sampling scheme {sampling_scheme!r}.")
    if missing:
        raise RuntimeError(
            "Installed CYTools is missing required public APIs: " + ", ".join(missing)
        )
    configure_mosek_license()


def _jsonable(value):
    """Convert CYTools/numpy scalar containers into deterministic JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _sha256_json(value):
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nullspace(matrix, tolerance=1e-10):
    """Return an orthonormal basis for the numerical nullspace of a matrix."""
    matrix = np.asarray(matrix, dtype=float)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    scale = max(float(np.max(singular_values, initial=0.0)), 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    return vh[rank:].T


def load_orientifold(path):
    """Load and validate the explicit orientifold input contract."""
    if path is None:
        return {"requested": False, "status": "not_requested"}
    try:
        with open(path, encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read orientifold JSON {path!r}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError("Orientifold JSON must contain an object.")
    try:
        raw_matrix = np.asarray(config.get("lattice_matrix"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("orientifold.lattice_matrix must contain integers.") from exc
    if raw_matrix.shape != (4, 4) or not np.all(np.isfinite(raw_matrix)):
        raise RuntimeError("orientifold.lattice_matrix must be an integer 4x4 matrix.")
    if not np.allclose(raw_matrix, np.rint(raw_matrix)):
        raise RuntimeError("orientifold.lattice_matrix must contain integers.")
    matrix = np.rint(raw_matrix).astype(int)
    orientifold_type = config.get("involution_type")
    if orientifold_type not in {"O3/O7", "O5/O9"}:
        raise RuntimeError(
            "orientifold.involution_type must be exactly 'O3/O7' or 'O5/O9'."
        )
    if not np.array_equal(matrix @ matrix, np.eye(4, dtype=int)):
        raise RuntimeError("orientifold.lattice_matrix must square to the identity.")
    determinant = round(float(np.linalg.det(matrix)))
    if determinant not in {-1, 1} or not np.isclose(np.linalg.det(matrix), determinant):
        raise RuntimeError("orientifold.lattice_matrix must be unimodular.")
    if not np.array_equal(matrix @ np.zeros(4, dtype=int), np.zeros(4, dtype=int)):
        raise RuntimeError("The orientifold lattice action must fix the origin.")
    return {
        "requested": True,
        "status": "input_loaded",
        "source_file": os.path.abspath(path),
        "lattice_matrix": matrix,
        "involution_type": orientifold_type,
        "coefficient_constraints": config.get("coefficient_constraints", {}),
        "label": config.get("label"),
    }


def validate_orientifold(poly, triangulation, topology, config):
    """Validate an explicit lattice involution and derive its H2 action."""
    if not config["requested"]:
        return config
    matrix = np.asarray(config["lattice_matrix"], dtype=int)
    points = np.asarray(poly.points(), dtype=int)
    point_lookup = {tuple(point): index for index, point in enumerate(points)}
    mapped_indices = []
    for point in points:
        mapped = tuple((matrix @ point).tolist())
        if mapped not in point_lookup:
            raise RuntimeError(
                "The orientifold lattice action does not preserve the KS polytope."
            )
        mapped_indices.append(point_lookup[mapped])
    mapped_indices = np.asarray(mapped_indices, dtype=int)
    triangulation_points = np.asarray(triangulation.points(), dtype=int)
    triangulation_global_indices = np.asarray(
        [point_lookup[tuple(point)] for point in triangulation_points], dtype=int
    )
    simplices = {
        tuple(
            sorted(int(triangulation_global_indices[vertex]) for vertex in simplex)
        )
        for simplex in np.asarray(triangulation.simplices(as_indices=True), dtype=int)
    }
    mapped_simplices = {
        tuple(sorted(int(mapped_indices[vertex]) for vertex in simplex))
        for simplex in simplices
    }
    if mapped_simplices != simplices:
        raise RuntimeError(
            "The orientifold lattice action does not preserve the selected FRST."
        )

    basis_matrix = np.asarray(topology["basis_matrix"], dtype=float)
    divisor_points = np.concatenate(
        (np.asarray([0], dtype=int), topology["prime_toric_divisors"])
    )
    if basis_matrix.shape[1] != divisor_points.size:
        raise RuntimeError(
            "The exported divisor basis does not match CYTools' canonical "
            "origin-plus-prime-divisor configuration."
        )
    mapped_divisor_points = mapped_indices[divisor_points]
    divisor_positions = {point: position for position, point in enumerate(divisor_points)}
    try:
        mapped_divisor_positions = np.asarray(
            [divisor_positions[point] for point in mapped_divisor_points], dtype=int
        )
    except KeyError as exc:
        raise RuntimeError(
            "The orientifold action does not preserve the prime toric divisor set."
        ) from exc
    permutation = np.zeros((divisor_points.size, divisor_points.size), dtype=float)
    permutation[np.arange(divisor_points.size), mapped_divisor_positions] = 1.0
    transformed_basis = basis_matrix @ permutation
    coefficients, _, _, _ = np.linalg.lstsq(
        basis_matrix.T, transformed_basis.T, rcond=None
    )
    h2_matrix = coefficients.T
    integral_h2 = np.rint(h2_matrix).astype(int)
    if not np.allclose(h2_matrix, integral_h2, atol=1e-8):
        raise RuntimeError(
            "The orientifold action does not induce an integral action in the "
            "exported divisor basis."
        )
    if not np.allclose(integral_h2 @ basis_matrix, transformed_basis, atol=1e-8):
        raise RuntimeError("Could not express the orientifold action in H2.")
    if not np.array_equal(integral_h2 @ integral_h2, np.eye(topology["h11"], dtype=int)):
        raise RuntimeError("The induced H2 action is not an involution.")

    invariant_basis = _nullspace(integral_h2.T - np.eye(topology["h11"]))
    anti_invariant_basis = _nullspace(integral_h2.T + np.eye(topology["h11"]))
    config = dict(config)
    config.update(
        {
            "status": "fan_invariant",
            "h2_involution_matrix": integral_h2,
            "invariant_kahler_basis": invariant_basis,
            "anti_invariant_h2_basis": anti_invariant_basis,
            "h11_plus": int(invariant_basis.shape[1]),
            "h11_minus": int(anti_invariant_basis.shape[1]),
            "polytope_preserved": True,
            "frst_preserved": True,
        }
    )
    return config


def validate_invariant_kaehler_subspace(kahler_cone, reference_tip, orientifold):
    """Check that the orientifold-even Kähler subspace reaches the cone."""
    if not orientifold["requested"]:
        return orientifold
    h2_matrix = np.asarray(orientifold["h2_involution_matrix"], dtype=float)
    invariant_basis = np.asarray(orientifold["invariant_kahler_basis"], dtype=float)
    hyperplanes = np.asarray(kahler_cone.hyperplanes(), dtype=float)
    if invariant_basis.shape[1] == 0:
        raise RuntimeError("The orientifold has no invariant H2/Kähler direction.")
    invariant_tip = invariant_basis @ (invariant_basis.T @ reference_tip)
    if np.min(hyperplanes @ invariant_tip) < 1.0 - 1e-6:
        try:
            from qpsolvers import available_solvers, solve_qp
        except ImportError as exc:
            raise RuntimeError(
                "qpsolvers is required to verify the invariant Kähler subspace."
            ) from exc
        equality = h2_matrix.T - np.eye(h2_matrix.shape[0])
        solvers = sorted(available_solvers)
        if not solvers:
            raise RuntimeError("qpsolvers found no solver for invariant Kähler validation.")
        invariant_tip = None
        for solver in solvers:
            try:
                invariant_tip = solve_qp(
                    np.eye(reference_tip.size),
                    np.zeros(reference_tip.size),
                    G=-hyperplanes,
                    h=-np.ones(hyperplanes.shape[0]),
                    A=equality,
                    b=np.zeros(equality.shape[0]),
                    solver=solver,
                    verbose=False,
                )
            except Exception:
                invariant_tip = None
            if invariant_tip is not None:
                break
        if invariant_tip is None:
            raise RuntimeError(
                "The orientifold-even Kähler subspace did not intersect the "
                "stretched Kähler cone."
            )
        invariant_tip = np.asarray(invariant_tip, dtype=float)
    if not np.all(np.isfinite(invariant_tip)) or np.min(hyperplanes @ invariant_tip) < 1.0 - 1e-6:
        raise RuntimeError(
            "The orientifold-even Kähler subspace did not intersect the "
            "stretched Kähler cone."
        )
    result = dict(orientifold)
    result["invariant_kahler_point"] = invariant_tip
    result["invariant_kahler_cone_intersection"] = True
    result["status"] = "validated"
    return result


def canonical_points(poly):
    """Return all lattice points in a stable order for construction metadata."""
    return sorted(tuple(int(coordinate) for coordinate in point) for point in poly.points())


def polytope_identity(poly):
    """Identify a fetched KS polytope without inventing an unstable database ID."""
    points = canonical_points(poly)
    return f"lattice-points-sha256:{_sha256_json(points)}", points


def validate_frst(poly, triangulation):
    """Validate the CYTools FRST contract before constructing the hypersurface."""
    if poly.ambient_dim() != 4 or poly.dim() != 4:
        raise RuntimeError(
            f"KS-to-CY3 construction requires a full-dimensional 4-polytope; "
            f"got ambient_dim={poly.ambient_dim()}, dim={poly.dim()}."
        )
    if not poly.is_reflexive():
        raise RuntimeError("The fetched KS polytope is not reflexive.")

    expected_labels = set(poly.labels_not_facet)
    actual_labels = set(int(label) for label in triangulation.labels)
    if actual_labels != expected_labels:
        raise RuntimeError(
            "FRST point configuration is not the reflexive default: it must "
            "contain exactly the lattice points outside facet interiors, including "
            "the origin."
        )

    checks = {
        "fine": bool(triangulation.is_fine()),
        "regular": bool(triangulation.is_regular()),
        "star": bool(triangulation.is_star()),
        "valid": bool(triangulation.is_valid()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("FRST validation failed for: " + ", ".join(failed))
    return checks


def extract_topology(cy, triangulation, *, export_kahler_rays=False):
    """Extract serializable CYTools topology used by Julia and fingerprints.

    Kähler-cone hyperplanes are sufficient for the physical validation and
    stretched-cone optimization performed by this generator. Enumerating the
    dual Kähler-cone rays is therefore opt-in: it can be prohibitively
    expensive at large h11 even though the hyperplanes are already available.
    """
    h11 = int(cy.h11())
    h21 = int(cy.h21())
    basis = np.asarray(cy.divisor_basis(), dtype=int)
    basis_matrix = np.asarray(cy.divisor_basis(as_matrix=True), dtype=int)
    prime_toric_divisors = np.asarray(cy.prime_toric_divisors(), dtype=int)
    kappa = np.asarray(
        cy.intersection_numbers(in_basis=True, format="coo"), dtype=float
    )
    if kappa.ndim != 2 or kappa.shape[1] != 4:
        raise RuntimeError(
            "CYTools returned an unexpected sparse intersection-number layout; "
            f"expected four [i, j, k, value] columns, got shape {kappa.shape}."
        )
    if basis.ndim == 1 and basis.size != h11:
        raise RuntimeError(
            f"Divisor basis has length {basis.size}, but CYTools reports h11={h11}."
        )
    if basis.ndim == 2 and basis.shape[0] != h11:
        raise RuntimeError(
            f"Generic divisor basis has {basis.shape[0]} rows, but h11={h11}."
        )

    c2 = np.asarray(cy.second_chern_class(in_basis=True), dtype=float)
    mori = np.asarray(cy.toric_mori_cone(in_basis=True).rays(), dtype=float)
    kahler = cy.toric_kahler_cone()
    kahler_rays = None
    if export_kahler_rays:
        kahler_rays = np.asarray(kahler.rays(), dtype=float)
    kahler_hyperplanes = np.asarray(kahler.hyperplanes(), dtype=float)
    if c2.shape != (h11,):
        raise RuntimeError(f"Unexpected c2 shape {c2.shape}; expected {(h11,)}.")
    finite_arrays = [
        ("intersection numbers", kappa),
        ("c2", c2),
        ("Mori cone", mori),
        ("Kähler cone hyperplanes", kahler_hyperplanes),
    ]
    if kahler_rays is not None:
        finite_arrays.append(("Kähler cone rays", kahler_rays))
    for name, array in finite_arrays:
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"CYTools returned non-finite {name}.")

    topology = {
        "h11": h11,
        "h21": h21,
        "basis": basis,
        "basis_matrix": basis_matrix,
        "prime_toric_divisors": prime_toric_divisors,
        "kappa": kappa,
        "c2": c2,
        "mori_cone": mori,
        "kahler_cone_rays": kahler_rays,
        "kahler_cone_hyperplanes": kahler_hyperplanes,
        "face_restriction_dim2": triangulation.simplices(on_faces_dim=2),
    }
    return topology


def topology_identity(polytope_id, triangulation, topology):
    """Build a conservative, explicitly non-complete CY3 topology fingerprint."""
    triangulation_id = f"frst-sha256:{_sha256_json(triangulation.simplices().tolist())}"
    fingerprint_payload = {
        "polytope_id": polytope_id,
        "simplices": triangulation.simplices().tolist(),
        "face_restriction_dim2": topology["face_restriction_dim2"],
        "h11": topology["h11"],
        "h21": topology["h21"],
        "kappa": topology["kappa"],
        "c2": topology["c2"],
    }
    cy3_fingerprint = f"topological-sha256:{_sha256_json(fingerprint_payload)}"
    return triangulation_id, cy3_fingerprint


class PrefactorCriterionNotMet(RuntimeError):
    """The current FRST's tip cannot satisfy the potential-control criterion."""


class NoPhysicalKaehlerPoint(RuntimeError):
    """No sampled point has positive volumes on effective-divisor cone rays."""


class NoQcdDivisorVolume(RuntimeError):
    """No stretched point satisfies the prime-divisor QCD volume window."""


def sample_stretched_kaehler_points(
    kahler_cone, reference_tip, rng, attempts, report, solver_used=None
):
    """Yield randomized points in the same stretched Kähler region.

    The canonical SKC tip is the norm-minimizing point in ``H t >= 1``.  At
    high h11 it need not lie in the useful angular region.  Each later point
    is the Euclidean projection of a random target onto that same polyhedron,
    so it remains inside the Kähler cone with every curve-wall distance >= 1.
    """
    mosek_license = configure_mosek_license()
    yield np.asarray(reference_tip, dtype=float)
    if attempts <= 1:
        return
    try:
        from qpsolvers import available_solvers, solve_qp
    except ImportError as exc:
        raise RuntimeError(
            "Angular Kähler-point sampling requires qpsolvers (a CYTools "
            "dependency). Reinstall CYTools with its solver extras."
        ) from exc

    hyperplanes = np.asarray(kahler_cone.hyperplanes(), dtype=float)
    if hyperplanes.ndim != 2 or hyperplanes.shape[1] != reference_tip.size:
        raise RuntimeError("Unexpected toric Kähler-cone hyperplane representation.")
    if not available_solvers:
        raise RuntimeError("qpsolvers found no installed quadratic-program solver.")
    solvers = (["mosek"] if mosek_license["activated"] and "mosek" in available_solvers else []) + [
        name for name in sorted(available_solvers) if name != "mosek"
    ]
    identity = np.eye(reference_tip.size)
    target_norm = max(float(np.linalg.norm(reference_tip)), 1.0)
    report(
        "MOSEK is "
        + (
            "licensed and preferred"
            if mosek_license["activated"] and "mosek" in available_solvers
            else "not licensed/available; using the first available qpsolvers backend"
        )
    )

    for number in range(2, attempts + 1):
        direction = rng.normal(size=reference_tip.size)
        direction /= max(float(np.linalg.norm(direction)), np.finfo(float).tiny)
        # A logarithmic range makes this explore angles rather than merely a
        # tiny neighborhood of the norm-minimizing reference tip.
        target = target_norm * (2.0 ** rng.uniform(-1.0, 4.0)) * direction
        report(f"projecting randomized Kähler point {number}/{attempts}")
        point = None
        selected_solver = None
        for solver in solvers:
            try:
                point = solve_qp(
                    identity,
                    -target,
                    G=-hyperplanes,
                    h=-np.ones(hyperplanes.shape[0]),
                    solver=solver,
                    verbose=False,
                )
            except Exception:
                point = None
            if point is not None:
                selected_solver = solver
                break
        if point is None:
            report(
                "randomized Kähler projection failed with all available "
                f"solvers ({', '.join(solvers)}); skipping candidate"
            )
            continue
        point = np.asarray(point, dtype=float)
        if np.all(np.isfinite(point)) and np.min(hyperplanes @ point) >= 1.0 - 1e-6:
            report(f"randomized Kähler projection succeeded with {selected_solver}")
            if solver_used is not None:
                solver_used.append(selected_solver)
            yield point
        else:
            report("randomized Kähler projection was infeasible; skipping candidate")


def generate_and_save_geometry(
    h11,
    cy,
    poly_points,
    simplices,
    filepath,
    max_m,
    max_kaehler_attempts,
    min_divisor_volume,
    min_prime_divisor_volume,
    qcd_volume_min,
    qcd_volume_max,
    rng,
    report,
    *,
    poly,
    triangulation,
    polytope_id,
    sampling_metadata,
    ks_database_version,
    orientifold_config,
    export_kahler_rays=False,
):
    """Compute the CYAxiverse datasets and write one HDF5 geometry file."""
    report("validating the CYTools FRST")
    frst_validation = validate_frst(poly, triangulation)
    if not bool(cy.is_smooth()):
        raise RuntimeError("CYTools reports that the generic CY hypersurface is not smooth.")
    report("computing Hodge, intersection, and divisor-basis data")
    topology = extract_topology(
        cy, triangulation, export_kahler_rays=export_kahler_rays
    )
    orientifold = validate_orientifold(poly, triangulation, topology, orientifold_config)
    h21 = topology["h21"]
    if topology["h11"] != int(h11) or topology["h11"] != int(cy.h11()):
        raise RuntimeError(
            f"h11 mismatch between request ({h11}) and CYTools ({topology['h11']})."
        )
    triangulation_id, cy3_fingerprint = topology_identity(
        polytope_id, triangulation, topology
    )
    favorable = bool(poly.is_favorable(lattice="N"))
    glsm = np.asarray(cy.glsm_charge_matrix(include_origin=False), dtype=int)
    basis = topology["basis"]

    report("finding the stretched Kähler-cone tip (this can be slow without Mosek)")
    kahler_cone = cy.toric_kahler_cone()
    mosek_license = configure_mosek_license()
    tip_solver = "cytools-default"
    if mosek_license["activated"]:
        try:
            reference_tip = np.asarray(
                kahler_cone.tip_of_stretched_cone(1.0, backend="mosek"),
                dtype=float,
            )
            tip_solver = "mosek"
        except Exception as exc:
            report(f"licensed MOSEK tip solve failed; falling back to CYTools default: {exc}")
            reference_tip = np.asarray(
                kahler_cone.tip_of_stretched_cone(1.0), dtype=float
            )
            tip_solver = "cytools-default-after-mosek-failure"
    else:
        reference_tip = np.asarray(
            kahler_cone.tip_of_stretched_cone(1.0), dtype=float
        )
    orientifold = validate_invariant_kaehler_subspace(
        kahler_cone, reference_tip, orientifold
    )
    report("computing effective-cone rays")
    qprime = np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    nq = qprime.shape[0]
    if qprime.ndim != 2 or qprime.shape[1] != int(h11):
        raise RuntimeError(
            "Effective-cone rays are not expressed in the exported divisor basis: "
            f"got shape {qprime.shape}, expected (*, {h11})."
        )

    # Eq. (20) of arXiv:2309.01831 concerns effective four-cycles on the CY.
    # Do *not* require every ambient toric divisor returned by CYTools to be
    # positive: at large h11 that list can include divisors with trivial or
    # redundant restriction to the hypersurface.  The toric effective-cone
    # rays are the relevant generators, and qprime and tau_basis share the
    # divisor-basis convention below.
    report("searching angular Kähler directions with positive effective-divisor volumes")
    kaehler_point = None
    divisor_scale = None
    projection_solvers = []
    for kaehler_attempt, candidate in enumerate(
        sample_stretched_kaehler_points(
            kahler_cone,
            reference_tip,
            rng,
            max_kaehler_attempts,
            report,
            projection_solvers,
        ),
        start=1,
    ):
        candidate_tau = np.asarray(
            cy.compute_divisor_volumes(candidate, in_basis=True), dtype=float
        )
        effective_volumes = qprime @ candidate_tau
        minimum_volume = float(np.min(effective_volumes))
        if not np.isfinite(minimum_volume) or minimum_volume <= 0.0:
            report(
                f"rejected Kähler point {kaehler_attempt}/{max_kaehler_attempts}: "
                f"minimum effective-divisor volume {minimum_volume:.3e}"
            )
            continue
        divisor_scale = max(1.0, math.sqrt(min_divisor_volume / minimum_volume))
        kaehler_point = divisor_scale * candidate
        report(
            f"accepted Kähler point {kaehler_attempt}/{max_kaehler_attempts}; "
            f"four-cycle scale={divisor_scale:.3e}"
        )
        break
    if kaehler_point is None:
        raise NoPhysicalKaehlerPoint(
            f"No positive-effective-divisor Kähler point among {max_kaehler_attempts} "
            "stretched-cone samples."
        )

    report("computing divisor volumes and inverse Kähler metric")
    tau0 = np.asarray(cy.compute_divisor_volumes(kaehler_point, in_basis=True), dtype=float)
    if tau0.shape != (int(h11),) or not np.all(np.isfinite(tau0)):
        raise RuntimeError(
            f"CYTools returned invalid basis divisor volumes with shape {tau0.shape}."
        )
    kinv0_raw = np.asarray(cy.compute_inverse_kahler_metric(kaehler_point), dtype=float)
    kinv0 = 0.5 * (kinv0_raw + kinv0_raw.T)
    if kinv0.shape != (int(h11), int(h11)) or not np.all(np.isfinite(kinv0)):
        raise RuntimeError("CYTools returned an invalid inverse Kähler metric.")
    if np.min(np.linalg.eigvalsh(kinv0)) <= 0.0:
        raise RuntimeError("The selected Kähler point has a non-positive metric.")
    tau, kinv = tau0.copy(), kinv0.copy()
    prime_tau0 = np.asarray(cy.compute_divisor_volumes(kaehler_point), dtype=float)
    if prime_tau0.ndim != 1 or not np.all(np.isfinite(prime_tau0)):
        raise RuntimeError("CYTools returned invalid prime toric divisor volumes.")

    # The original script evaluates this condition in nested Python loops for
    # every 0.01 increment of m.  At h11=491 that dominates the runtime.  The
    # bilinear form and ray volumes only scale as m**4 and m**2, respectively,
    # so precompute their m=1 values and test all lower-triangular pairs in
    # NumPy instead.
    report(f"searching the stretched-cone prefactor over {nq * (nq - 1) // 2} ray pairs")
    lower_i, lower_j = np.tril_indices(nq, k=-1)
    bilinear0 = (qprime @ kinv0) @ qprime.T
    tauq0 = qprime @ tau0
    if np.min(tauq0) <= 0.0:
        raise NoPhysicalKaehlerPoint(
            "The positive prime-divisor point has non-positive effective-ray "
            "volumes in the selected divisor basis."
        )
    def prefactor_is_valid(candidate_m):
        """Evaluate the original pairwise criterion at one candidate value."""
        candidate_m2 = candidate_m**2
        candidate_tauq = candidate_m2 * tauq0
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate_lhs = np.abs(
                np.log(
                    np.abs(
                        math.pi
                        * candidate_m2**2
                        * bilinear0[lower_i, lower_j]
                    )
                )
                - 2
                * math.pi
                * (candidate_tauq[lower_i] + candidate_tauq[lower_j])
            )
            candidate_rhs = np.abs(
                np.log(np.abs(candidate_tauq)) - 2 * math.pi * candidate_tauq
            )
        return np.all(candidate_lhs > candidate_rhs[lower_i])

    # Doubling reaches the large-m regime in O(log(m)) checks.  Once it finds
    # a valid upper bound, binary refinement recovers the old 0.01 resolution.
    lower_m, upper_m = 1.0, 1.0
    if not prefactor_is_valid(upper_m):
        while upper_m < max_m:
            lower_m = upper_m
            upper_m = min(2.0 * upper_m, max_m)
            report(f"testing stretched-cone prefactor m={upper_m:.2f}")
            if prefactor_is_valid(upper_m):
                break
        else:
            raise PrefactorCriterionNotMet(
                f"The stretched-cone prefactor did not converge before m={max_m}. "
                "Reject this FRST and try a different tip."
            )

        if not prefactor_is_valid(upper_m):
            raise PrefactorCriterionNotMet(
                f"The stretched-cone prefactor did not converge before m={max_m}. "
                "Reject this FRST and try a different tip."
            )

        while upper_m - lower_m > 1e-2:
            midpoint = 0.5 * (lower_m + upper_m)
            if prefactor_is_valid(midpoint):
                upper_m = midpoint
            else:
                lower_m = midpoint
    m_val = upper_m
    minimum_m_for_prime_divisors = math.sqrt(
        min_prime_divisor_volume / float(np.min(prime_tau0))
    )
    qcd_interval = None
    for prime_volume in prime_tau0:
        lower = math.sqrt(qcd_volume_min / float(prime_volume))
        upper = math.sqrt(qcd_volume_max / float(prime_volume))
        lower = max(lower, m_val, minimum_m_for_prime_divisors)
        if lower <= upper and lower <= max_m and prefactor_is_valid(lower):
            qcd_interval = (lower, upper)
            break
    if qcd_interval is None:
        raise NoQcdDivisorVolume(
            "No stretched-cone prefactor satisfies the prime-toric-divisor "
            f"lower bound {min_prime_divisor_volume:g} and QCD window "
            f"[{qcd_volume_min:g}, {qcd_volume_max:g}]."
        )
    m_val = qcd_interval[0]
    m2 = m_val**2
    tau = m2 * tau0
    kinv = m2**2 * kinv0

    # Store a self-consistent physical point: tau, Kinv and the CY volume are
    # all evaluated at the same final J = m * kaehler_point.
    tip = m_val * kaehler_point
    volume = float(cy.compute_cy_volume(tip))
    prime_divisor_volumes = m2 * prime_tau0
    qcd_divisor_indices = np.flatnonzero(
        (prime_divisor_volumes >= qcd_volume_min)
        & (prime_divisor_volumes <= qcd_volume_max)
    )
    if (
        np.min(prime_divisor_volumes) < min_prime_divisor_volume - 1e-8
        or qcd_divisor_indices.size == 0
    ):
        raise NoQcdDivisorVolume("Final prime toric divisor volumes failed validation.")
    qcd_divisor_index = int(qcd_divisor_indices[0])
    curve_volumes = np.asarray(cy.compute_curve_volumes(tip), dtype=float)
    kahler_slack = np.asarray(kahler_cone.hyperplanes(), dtype=float) @ tip
    if (
        not np.isfinite(volume)
        or volume <= 0.0
        or not np.all(np.isfinite(curve_volumes))
        or np.min(curve_volumes) <= 0.0
        or np.min(kahler_slack) < 1.0 - 1e-6
    ):
        raise RuntimeError("Final CY geometry failed volume or Kähler-cone validation.")
    tip_prefactor = np.asarray([divisor_scale, m_val], dtype=float)

    report(f"building potential data from {nq} effective-cone rays")
    num_cross = nq * (nq - 1) // 2
    q = np.empty((nq + num_cross, h11), dtype=float)
    q[:nq] = qprime
    l2 = np.empty((num_cross, 2), dtype=float)
    idx = 0
    for i in range(nq - 1):
        qi = qprime[i]
        for j in range(i + 1, nq):
            qj = qprime[j]
            q[nq + idx] = qj - qi
            l2[idx] = [
                (math.pi * (qi @ (kinv @ qj)) + ((qi + qj) @ tau))
                * 8 * math.pi / volume**2,
                -2 * math.log10(math.e) * math.pi * ((qi @ tau) + (qj @ tau)),
            ]
            idx += 1

    l1 = np.empty((nq, 2), dtype=float)
    for j, qj in enumerate(qprime):
        q_tau = qj @ tau
        l1[j] = [
            (8 * math.pi / volume**2) * q_tau,
            -2 * math.log10(math.e) * math.pi * q_tau,
        ]
    l_raw = np.vstack((l1, l2))
    if not np.all(np.isfinite(l_raw)) or np.any(l_raw[:, 0] == 0.0):
        raise RuntimeError("Potential coefficients contain zero or non-finite amplitudes.")
    l = np.empty_like(l_raw)
    l[:, 0] = np.sign(l_raw[:, 0])
    l[:, 1] = np.log10(np.abs(l_raw[:, 0])) + l_raw[:, 1]

    report("writing HDF5 data")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temporary_path = f"{filepath}.tmp-{os.getpid()}-{time.time_ns()}"
    construction_metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_paper_set": SOURCE_PAPER_SET,
        "cytools_version": cytools.version,
        "ks_database_version": ks_database_version,
        "polytope_id": polytope_id,
        "polytope_id_kind": "canonical_lattice_point_sha256",
        "triangulation_id": triangulation_id,
        "cy3_fingerprint": cy3_fingerprint,
        "cy3_fingerprint_status": "topological_fingerprint",
        "sampling": sampling_metadata,
        "mosek_license": {
            "configured": mosek_license["configured"],
            "activated": mosek_license["activated"],
            "source": mosek_license["source"],
            "tip_solver": tip_solver,
            "projection_solvers": sorted(set(projection_solvers)),
        },
        "favorable": favorable,
        "basis_convention": "CYTools divisor_basis(include_origin=True); all numerical vectors in basis",
        "intersection_convention": "CYTools CalabiYau.intersection_numbers(in_basis=True, format='coo')",
        "frst_validation": frst_validation,
        "kappa_format": "coo",
        "kappa_columns": ["i", "j", "k", "value"],
        "kappa_index_base": 0,
        "prime_divisor_volume_lower_bound": min_prime_divisor_volume,
        "prime_divisor_convention": (
            "CYTools compute_divisor_volumes(tip), ordered by "
            "CYTools prime_toric_divisors()"
        ),
        "qcd_divisor_volume_window": [qcd_volume_min, qcd_volume_max],
        "qcd_divisor_index": qcd_divisor_index,
        "qcd_divisor_index_base": 0,
        "qcd_divisor_volume": float(prime_divisor_volumes[qcd_divisor_index]),
        "kahler_cone_rays_exported": bool(export_kahler_rays),
        "orientifold": orientifold,
    }
    try:
        with h5py.File(temporary_path, "w") as file:
            file.attrs["schema_version"] = SCHEMA_VERSION
            file.attrs["construction_metadata_json"] = json.dumps(
                _jsonable(construction_metadata), sort_keys=True, separators=(",", ":")
            )
            cytools_group = file.create_group("cytools")
            geometric = cytools_group.create_group("geometric")
            geometric.create_dataset("points", data=poly_points, compression="gzip", compression_opts=9)
            geometric.create_dataset(
                "triangulation_points",
                data=np.asarray(triangulation.points(), dtype=int),
                compression="gzip",
                compression_opts=9,
            )
            geometric.create_dataset("simplices", data=simplices, compression="gzip", compression_opts=9)
            geometric.create_dataset("h11", data=topology["h11"])
            geometric.create_dataset("h21", data=h21)
            geometric.create_dataset("glsm", data=glsm, compression="gzip", compression_opts=9)
            geometric.create_dataset("basis", data=basis, compression="gzip", compression_opts=9)
            geometric.create_dataset(
                "basis_matrix", data=topology["basis_matrix"], compression="gzip", compression_opts=9
            )
            geometric.create_dataset(
                "prime_toric_divisors",
                data=topology["prime_toric_divisors"],
                compression="gzip",
                compression_opts=9,
            )
            geometric.create_dataset("tip", data=tip, compression="gzip", compression_opts=9)
            geometric.create_dataset("tip_prefactor", data=tip_prefactor, compression="gzip", compression_opts=9)
            geometric.create_dataset("CY_volume", data=volume)
            geometric.create_dataset("divisor_volumes", data=tau, compression="gzip", compression_opts=9)
            geometric.create_dataset(
                "prime_divisor_volumes",
                data=prime_divisor_volumes,
                compression="gzip",
                compression_opts=9,
            )
            geometric.create_dataset("curve_volumes", data=curve_volumes, compression="gzip", compression_opts=9)
            geometric.create_dataset("Kinv", data=kinv, compression="gzip", compression_opts=9)
            geometric.create_dataset("kappa", data=topology["kappa"], compression="gzip", compression_opts=9)
            geometric.create_dataset("c2", data=topology["c2"], compression="gzip", compression_opts=9)
            geometric.create_dataset("effective_cone", data=qprime, compression="gzip", compression_opts=9)
            geometric.create_dataset("mori_cone", data=topology["mori_cone"], compression="gzip", compression_opts=9)
            if topology["kahler_cone_rays"] is not None:
                geometric.create_dataset(
                    "kahler_cone",
                    data=topology["kahler_cone_rays"],
                    compression="gzip",
                    compression_opts=9,
                )
            geometric.create_dataset(
                "kahler_hyperplanes",
                data=topology["kahler_cone_hyperplanes"],
                compression="gzip",
                compression_opts=9,
            )
            geometric.attrs["favorable"] = favorable
            geometric.attrs["polytope_id"] = polytope_id
            geometric.attrs["triangulation_id"] = triangulation_id
            geometric.attrs["cy3_fingerprint"] = cy3_fingerprint
            geometric.attrs["sampling_scheme"] = sampling_metadata["scheme"]
            geometric.attrs["kappa_format"] = construction_metadata["kappa_format"]
            geometric.attrs["kappa_index_base"] = construction_metadata["kappa_index_base"]
            geometric.attrs["basis_convention"] = construction_metadata["basis_convention"]
            geometric.attrs["intersection_convention"] = construction_metadata["intersection_convention"]
            if orientifold["requested"]:
                orientifold_group = geometric.create_group("orientifold")
                orientifold_group.create_dataset(
                    "lattice_matrix", data=orientifold["lattice_matrix"]
                )
                orientifold_group.create_dataset(
                    "h2_involution_matrix",
                    data=orientifold["h2_involution_matrix"],
                )
                orientifold_group.create_dataset(
                    "invariant_kahler_basis",
                    data=orientifold["invariant_kahler_basis"],
                )
                orientifold_group.create_dataset(
                    "anti_invariant_h2_basis",
                    data=orientifold["anti_invariant_h2_basis"],
                )
                orientifold_group.create_dataset(
                    "invariant_kahler_point",
                    data=orientifold["invariant_kahler_point"],
                )
                orientifold_group.attrs["involution_type"] = orientifold["involution_type"]
                orientifold_group.attrs["h11_plus"] = orientifold["h11_plus"]
                orientifold_group.attrs["h11_minus"] = orientifold["h11_minus"]
            construction_metadata_group = file.create_group("construction_metadata")
            construction_metadata_group.create_dataset(
                "canonical_lattice_points",
                data=np.asarray(canonical_points(poly), dtype=int),
                compression="gzip",
            )
            construction_metadata_group.create_dataset(
                "face_restriction_dim2",
                data=np.asarray(topology["face_restriction_dim2"], dtype=int),
                compression="gzip",
            )
            construction_metadata_group.attrs["construction_metadata_json"] = json.dumps(
                _jsonable(construction_metadata), sort_keys=True, separators=(",", ":")
            )
            potential = cytools_group.create_group("potential")
            potential.create_dataset("L", data=l, compression="gzip", compression_opts=9)
            potential.create_dataset("Q", data=q, compression="gzip", compression_opts=9)
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def output_path(base_dir, h11, polytope_index, triangulation_index):
    return os.path.join(
        base_dir,
        f"h11_{h11:03d}",
        f"np_{polytope_index:07d}",
        f"cy_{triangulation_index:07d}",
        "cyax.h5",
    )


def triangulation_candidates(
    poly,
    sampling_scheme,
    max_tip_attempts,
    max_retries,
    backend,
    seed,
    n_walk,
    n_flip,
    initial_walk_steps,
    fine_tune_steps,
    walk_step_size,
    max_steps_to_wall,
    fast_height_scale,
):
    """Yield CYTools FRSTs with an explicit fast or fair sampling contract."""
    point_labels = tuple(poly.labels_not_facet)
    if sampling_scheme == "fast":
        # The deterministic candidate is useful for coverage scans, but is
        # deliberately recorded as part of the biased fast ensemble.
        yield poly.triangulate(
            points=point_labels,
            make_star=True,
            backend=backend,
            verbosity=0,
        )
        remaining = max_tip_attempts - 1
        if remaining > 0:
            yield from poly.random_triangulations_fast(
                N=remaining,
                c=fast_height_scale,
                max_retries=max_retries,
                make_star=True,
                only_fine=True,
                points=point_labels,
                backend=backend,
                as_list=False,
                progress_bar=False,
                seed=seed,
            )
        return

    # CYTools implements the secondary-fan walk and random-flip sampler from
    # arXiv:2008.01730; keep this adapter declarative and pass every control
    # parameter explicitly so artifacts are reproducible.
    yield from poly.random_triangulations_fair(
        N=max_tip_attempts,
        n_walk=n_walk,
        n_flip=n_flip,
        initial_walk_steps=initial_walk_steps,
        walk_step_size=walk_step_size,
        max_steps_to_wall=max_steps_to_wall,
        fine_tune_steps=fine_tune_steps,
        max_retries=max_retries,
        make_star=True,
        points=point_labels,
        backend=backend,
        as_list=False,
        progress_bar=False,
        seed=seed,
    )


def process_polytope(task):
    """Generate a bounded number of distinct FRSTs for one polytope."""
    (
        polytope_index,
        vertices,
        h11,
        requested,
        base_dir,
        seed,
        max_retries,
        max_tip_attempts,
        overwrite,
        max_m,
        max_kaehler_attempts,
        min_divisor_volume,
        min_prime_divisor_volume,
        qcd_volume_min,
        qcd_volume_max,
        verbose,
        sampling_scheme,
        backend,
        n_walk,
        n_flip,
        initial_walk_steps,
        fine_tune_steps,
        walk_step_size,
        max_steps_to_wall,
        fast_height_scale,
        ks_database_version,
        orientifold_config,
        export_kahler_rays,
    ) = task
    try:
        started = time.perf_counter()

        def report(message):
            if verbose:
                elapsed = time.perf_counter() - started
                print(
                    f"np_{polytope_index:07d} [{elapsed:8.1f}s]: {message}",
                    flush=True,
                )

        # Vertices, rather than all lattice points, make reconstruction cheap for
        # the h11=491 N-lattice polytope (which has 680 lattice points).
        report("constructing polytope")
        poly = Polytope(vertices, deterministic_glsm_basis=True)
        points = np.asarray(poly.points(), dtype=int)
        n_points = len(points)
        n_walk = n_points // 10 + 10 if n_walk is None else n_walk
        n_flip = n_points // 10 + 10 if n_flip is None else n_flip
        initial_walk_steps = (
            2 * n_points // 10 + 10
            if initial_walk_steps is None
            else initial_walk_steps
        )
        polytope_id, _ = polytope_identity(poly)
        mosek_license = configure_mosek_license()
        sampling_metadata = {
            "scheme": sampling_scheme,
            "backend": backend,
            "seed": seed,
            "N_walk": n_walk,
            "N_flip": n_flip,
            "initial_walk_steps": initial_walk_steps,
            "fine_tune_steps": fine_tune_steps,
            "walk_step_size": walk_step_size,
            "max_steps_to_wall": max_steps_to_wall,
            "max_retries": max_retries,
            "fast_height_scale": fast_height_scale,
            "qp_solver_preference": "mosek_if_licensed_then_available",
            "mosek_license_configured": mosek_license["configured"],
            "mosek_license_activated": mosek_license["activated"],
        }

        # Output indices represent accepted geometries, not raw triangulation
        # attempts.  This lets a resumed scan retain prior successful samples.
        existing_indices = []
        index = 1
        if not overwrite:
            while os.path.exists(output_path(base_dir, h11, polytope_index, index)):
                existing_indices.append(index)
                index += 1
        accepted = len(existing_indices)
        next_output_index = index
        if accepted >= requested and not overwrite:
            return {
                "ok": True,
                "polytope_index": polytope_index,
                "requested": requested,
                "attempted": 0,
                "accepted": accepted,
                "saved": 0,
                "rejected": 0,
                "skipped": accepted,
            }

        report(f"sampling FRST candidates with scheme={sampling_scheme}")
        candidates = triangulation_candidates(
            poly,
            sampling_scheme,
            max_tip_attempts,
            max_retries,
            backend,
            seed,
            n_walk,
            n_flip,
            initial_walk_steps,
            fine_tune_steps,
            walk_step_size,
            max_steps_to_wall,
            fast_height_scale,
        )

        saved = rejected = attempted = 0
        for triangulation in candidates:
            if accepted >= requested:
                break
            attempted += 1
            triangulation_index = next_output_index
            filepath = output_path(base_dir, h11, polytope_index, triangulation_index)
            report(
                f"testing FRST candidate {attempted}/{max_tip_attempts} "
                f"for accepted geometry {accepted + 1}/{requested}"
            )
            report(f"processing accepted-output slot {triangulation_index}")
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            try:
                generate_and_save_geometry(
                    h11,
                    triangulation.get_cy(),
                    points,
                    simplices,
                    filepath,
                    max_m,
                    max_kaehler_attempts,
                    min_divisor_volume,
                    min_prime_divisor_volume,
                    qcd_volume_min,
                    qcd_volume_max,
                    np.random.default_rng(seed + attempted - 1),
                    report,
                    poly=poly,
                    triangulation=triangulation,
                    polytope_id=polytope_id,
                    sampling_metadata=sampling_metadata,
                    ks_database_version=ks_database_version,
                    orientifold_config=orientifold_config,
                    export_kahler_rays=export_kahler_rays,
                )
            except (
                PrefactorCriterionNotMet,
                NoPhysicalKaehlerPoint,
                NoQcdDivisorVolume,
            ) as exc:
                rejected += 1
                report(f"rejected FRST: {exc}")
                continue
            accepted += 1
            saved += 1
            next_output_index += 1
        return {
            "ok": True,
            "polytope_index": polytope_index,
            "requested": requested,
            "attempted": attempted,
            "accepted": accepted,
            "saved": saved,
            "rejected": rejected,
            "skipped": len(existing_indices),
        }
    except Exception as exc:
        return {"ok": False, "polytope_index": polytope_index, "error": repr(exc)}


def parse_h11_values(values):
    """Parse explicit h11 values from comma- or whitespace-separated arguments."""
    tokens = []
    for value in values:
        cleaned = value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        tokens.extend(cleaned.replace(",", " ").split())
    if not tokens:
        raise ValueError("at least one h11 value is required")

    try:
        h11_values = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("h11 values must be integers") from exc
    if any(h11 < 1 for h11 in h11_values):
        raise ValueError("h11 values must be positive")
    if len(set(h11_values)) != len(h11_values):
        raise ValueError("h11 values must be unique")
    return h11_values


def plan_tasks(
    h11,
    n_geometries,
    base_dir,
    seed,
    max_retries,
    max_tip_attempts,
    overwrite,
    max_m,
    max_kaehler_attempts,
    min_divisor_volume,
    min_prime_divisor_volume,
    qcd_volume_min,
    qcd_volume_max,
    verbose,
    sampling_scheme,
    backend,
    n_walk,
    n_flip,
    initial_walk_steps,
    fine_tune_steps,
    walk_step_size,
    max_steps_to_wall,
    fast_height_scale,
    ks_database_version,
    favorable,
    orientifold_config,
    export_kahler_rays,
):
    """Fetch at most n polytopes and spread n requested geometries over them."""
    polytopes = list(
        fetch_polytopes(
            h11=h11,
            limit=n_geometries,
            lattice="N",
            favorable=favorable,
            deterministic_glsm_basis=True,
        )
    )
    if not polytopes:
        return []

    # Give every available polytope one geometry, then distribute the remaining
    # requests round-robin as extra triangulations of those polytopes.
    counts = [1] * len(polytopes)
    for index in itertools.islice(itertools.cycle(range(len(polytopes))), n_geometries - len(polytopes)):
        counts[index] += 1

    return [
        (
            polytope_index,
            np.asarray(poly.vertices(), dtype=int),
            h11,
            count,
            base_dir,
            seed + polytope_index,
            max_retries,
            max_tip_attempts,
            overwrite,
            max_m,
            max_kaehler_attempts,
            min_divisor_volume,
            min_prime_divisor_volume,
            qcd_volume_min,
            qcd_volume_max,
            verbose,
            sampling_scheme,
            backend,
            n_walk,
            n_flip,
            initial_walk_steps,
            fine_tune_steps,
            walk_step_size,
            max_steps_to_wall,
            fast_height_scale,
            ks_database_version,
            orientifold_config,
            export_kahler_rays,
        )
        for polytope_index, (poly, count) in enumerate(zip(polytopes, counts), start=1)
    ]


def run_batch(
    h11,
    n_geometries,
    base_dir,
    n_cores,
    seed,
    max_retries,
    max_tip_attempts,
    overwrite,
    max_m,
    max_kaehler_attempts,
    min_divisor_volume,
    min_prime_divisor_volume,
    qcd_volume_min,
    qcd_volume_max,
    verbose,
    sampling_scheme,
    backend,
    n_walk,
    n_flip,
    initial_walk_steps,
    fine_tune_steps,
    walk_step_size,
    max_steps_to_wall,
    fast_height_scale,
    ks_database_version,
    favorable,
    orientifold_config,
    export_kahler_rays,
):
    tasks = plan_tasks(
        h11,
        n_geometries,
        base_dir,
        seed,
        max_retries,
        max_tip_attempts,
        overwrite,
        max_m,
        max_kaehler_attempts,
        min_divisor_volume,
        min_prime_divisor_volume,
        qcd_volume_min,
        qcd_volume_max,
        verbose,
        sampling_scheme,
        backend,
        n_walk,
        n_flip,
        initial_walk_steps,
        fine_tune_steps,
        walk_step_size,
        max_steps_to_wall,
        fast_height_scale,
        ks_database_version,
        favorable,
        orientifold_config,
        export_kahler_rays,
    )
    if not tasks:
        print(f"No favorable N-lattice polytopes found for h11={h11}.")
        return 0

    print(
        f"Found {len(tasks)} favorable polytope(s); requesting {n_geometries} "
        f"geometry/triangulation output(s)."
    )
    saved = 0
    failures = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_polytope, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if not result["ok"]:
                print(f"ERROR np_{result['polytope_index']:07d}: {result['error']}")
                failures.append(result)
                continue
            saved += result["saved"]
            print(
                f"np_{result['polytope_index']:07d}: accepted "
                f"{result['accepted']}/{result['requested']} geometries after "
                f"{result['attempted']} FRST attempts; saved {result['saved']}, "
                f"rejected {result['rejected']}, skipped {result['skipped']}."
            )
    if failures:
        details = "; ".join(
            f"np_{failure['polytope_index']:07d}: {failure['error']}"
            for failure in failures
        )
        raise RuntimeError(f"CYTools generation failed; no complete batch result: {details}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate CYTools geometry data for Julia.")
    parser.add_argument("--h11_min", type=int, default=4, help="Starting h11 value.")
    parser.add_argument("--h11_max", type=int, default=4, help="Ending h11 value (inclusive).")
    parser.add_argument(
        "--h11_interval",
        type=int,
        default=1,
        help="Step between h11_min and h11_max (inclusive; ignored with --h11s).",
    )
    parser.add_argument(
        "--h11s",
        "--h11-list",
        "--h11_list",
        dest="h11s",
        nargs="+",
        metavar="H11",
        help="Explicit h11 values, e.g. '[4,10,20,50]' or '4 10 20 50'.",
    )
    parser.add_argument("--n", type=int, default=1, help="Target number of CY geometries per h11.")
    parser.add_argument("--outdir", type=str, default=".", help="Base directory for output data.")
    parser.add_argument("--cores", type=int, default=None, help="Worker count (default: all available).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible random triangulations.")
    parser.add_argument(
        "--sampling-scheme",
        choices=("fair", "fast"),
        default="fair",
        help="FRST ensemble: fair secondary-fan walk (default) or biased fast heights.",
    )
    parser.add_argument(
        "--backend",
        choices=("cgal", "qhull"),
        default="cgal",
        help="CYTools triangulation backend.",
    )
    parser.add_argument(
        "--favorable",
        choices=("true", "false", "any"),
        default="true",
        help="KS favorability filter; 'any' also permits expanded non-favorable bases.",
    )
    parser.add_argument(
        "--ks-database-version",
        default="CYTools fetch_polytopes endpoint (version not exposed)",
        help="Version/endpoint label recorded in every artifact.",
    )
    parser.add_argument("--max-retries", type=int, default=50, help="Bound per-FRST search retries.")
    parser.add_argument(
        "--max-tip-attempts",
        type=int,
        default=50,
        help="Maximum FRST candidates tried per polytope before reporting a shortfall.",
    )
    parser.add_argument(
        "--max-m",
        type=float,
        default=1_000_000.0,
        help="Maximum stretched-cone prefactor before reporting a non-convergence.",
    )
    parser.add_argument(
        "--max-kaehler-attempts",
        type=int,
        default=100,
        help="Angular Kähler points tested per FRST (including the canonical tip).",
    )
    parser.add_argument("--n-walk", type=int, default=None, help="Fair sampler walk steps per sample.")
    parser.add_argument("--n-flip", type=int, default=None, help="Fair sampler flips per sample.")
    parser.add_argument(
        "--initial-walk-steps",
        type=int,
        default=None,
        help="Fair sampler burn-in walk steps before recording samples.",
    )
    parser.add_argument(
        "--fine-tune-steps",
        type=int,
        default=8,
        help="Fair sampler wall-location refinement steps.",
    )
    parser.add_argument(
        "--walk-step-size",
        type=float,
        default=1e-2,
        help="Fair sampler secondary-fan walk step size.",
    )
    parser.add_argument(
        "--max-steps-to-wall",
        type=int,
        default=25,
        help="Fair sampler maximum search steps toward a secondary-fan wall.",
    )
    parser.add_argument(
        "--fast-height-scale",
        type=float,
        default=0.2,
        help="Fast sampler Gaussian height scale passed to CYTools.",
    )
    parser.add_argument(
        "--min-divisor-volume",
        type=float,
        default=1.0,
        help="Required lower bound for every effective-divisor cone ray volume.",
    )
    parser.add_argument(
        "--min-prime-divisor-volume",
        type=float,
        default=1.0,
        help="Required lower bound for every prime toric divisor volume.",
    )
    parser.add_argument(
        "--qcd-volume-min",
        type=float,
        default=25.0,
        help="Lower edge of the required prime-divisor QCD volume window.",
    )
    parser.add_argument(
        "--qcd-volume-max",
        type=float,
        default=40.0,
        help="Upper edge of the required prime-divisor QCD volume window.",
    )
    parser.add_argument(
        "--orientifold-file",
        type=str,
        default=None,
        help="JSON file containing an explicit lattice involution and orientifold type.",
    )
    parser.add_argument(
        "--export-kahler-rays",
        action="store_true",
        help=(
            "Enumerate and store Kähler-cone rays. This is optional and can be "
            "prohibitively expensive at large h11; hyperplanes are always stored."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cyax.h5 files.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-worker stages and elapsed times; recommended for large h11.",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be positive")
    if args.sampling_scheme not in {"fair", "fast"}:
        parser.error("--sampling-scheme must be fair or fast")
    if args.max_tip_attempts < 1:
        parser.error("--max-tip-attempts must be positive")
    if args.max_kaehler_attempts < 1:
        parser.error("--max-kaehler-attempts must be positive")
    if args.min_divisor_volume <= 0.0:
        parser.error("--min-divisor-volume must be positive")
    if args.min_prime_divisor_volume <= 0.0:
        parser.error("--min-prime-divisor-volume must be positive")
    if args.qcd_volume_min <= 0.0 or args.qcd_volume_max < args.qcd_volume_min:
        parser.error("--qcd-volume-min must be positive and no greater than --qcd-volume-max")
    for name, value in (
        ("--n-walk", args.n_walk),
        ("--n-flip", args.n_flip),
        ("--initial-walk-steps", args.initial_walk_steps),
    ):
        if value is not None and value < 1:
            parser.error(f"{name} must be positive")
    if args.fine_tune_steps < 1 or args.max_steps_to_wall < 1:
        parser.error("--fine-tune-steps and --max-steps-to-wall must be positive")
    if args.walk_step_size <= 0.0 or args.fast_height_scale <= 0.0:
        parser.error("--walk-step-size and --fast-height-scale must be positive")
    if args.h11_interval < 1:
        parser.error("--h11_interval must be positive")
    if args.h11s is not None:
        if args.h11_interval != 1:
            parser.error("--h11_interval cannot be combined with --h11s")
        try:
            h11_values = parse_h11_values(args.h11s)
        except ValueError as exc:
            parser.error(f"--h11s: {exc}")
    else:
        if args.h11_min < 1 or args.h11_max < 1:
            parser.error("--h11_min and --h11_max must be positive")
        if args.h11_max < args.h11_min:
            args.h11_max = args.h11_min
        h11_values = list(range(args.h11_min, args.h11_max + 1, args.h11_interval))
    require_cytools_capabilities(args.sampling_scheme)
    orientifold_config = load_orientifold(args.orientifold_file)
    favorable = {"true": True, "false": False, "any": None}[args.favorable]
    os.makedirs(args.outdir, exist_ok=True)

    total_saved = 0
    for h11 in h11_values:
        print(f"\n>>> Processing h11={h11} <<<")
        total_saved += run_batch(
            h11,
            args.n,
            args.outdir,
            args.cores,
            args.seed,
            args.max_retries,
            args.max_tip_attempts,
            args.overwrite,
            args.max_m,
            args.max_kaehler_attempts,
            args.min_divisor_volume,
            args.min_prime_divisor_volume,
            args.qcd_volume_min,
            args.qcd_volume_max,
            args.verbose,
            args.sampling_scheme,
            args.backend,
            args.n_walk,
            args.n_flip,
            args.initial_walk_steps,
            args.fine_tune_steps,
            args.walk_step_size,
            args.max_steps_to_wall,
            args.fast_height_scale,
            args.ks_database_version,
            favorable,
            orientifold_config,
            args.export_kahler_rays,
        )
    print(f"\nSaved {total_saved} geometry file(s).")


if __name__ == "__main__":
    main()
