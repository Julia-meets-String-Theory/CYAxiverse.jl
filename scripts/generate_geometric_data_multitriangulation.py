"""Generate CYTools data, including several FRSTs for rare polytopes.

``--n`` is the number of Calabi--Yau geometries requested for each h11, not
the number of distinct polytopes.  The script first fetches up to ``n``
favorable N-lattice polytopes.  If fewer exist (as for h11=491), it samples
several bounded, pseudorandom FRSTs of each available polytope instead of ever
attempting to enumerate all triangulations.

Each FRST has its own toric Kähler cone and stretched-cone tip.  Under the
default ``adaptive`` moduli policy, a candidate is saved only when an angular
Kähler point can be dilated to satisfy the control criterion of
arXiv:2309.01831, eq. (21), and a prime divisor lies in the QCD volume window.
The optional ``canonical_qcd`` policy keeps the canonical stretched-cone ray
and applies a homogeneous radial rescaling so an explicit or deterministically
ordered eligible prime divisor has a requested target volume.  By default,
eligible candidates are ordered by descending positive finite tip volume so
the selected divisor minimizes the required dilation; an explicit divisor
index remains an override.  This
is a normalization of an existing FRST, not a new triangulation or a D-brane
model.  The optional
``intersecting_d7`` visible-sector policy adds the paper-style toy assignment:
it requires a validated O3/O7 involution, selects an invariant QED divisor
intersecting QCD, and exports the corresponding QED charge and Euclidean-D3
instanton term.  On the production EFT path, a QCD candidate must first have
an orientifold-invariant intersecting QED neighbor whose candidate-specific
post-normalization volume is within the inclusive QED bound.  It does not
claim global tadpole or matter cancellation.

The default ``fair`` sampler delegates secondary-fan walks and flips to
CYTools. ``fast`` is available for explicitly biased coverage/training scans.
Kähler-cone quadratic programs prefer MOSEK when it is licensed and available; if
``$MOSEKLM_LICENSE_FILE`` is unset, a standard ``$HOME/mosek.lic`` is exposed
to the child solver without reading or copying the license contents.
"""

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait

import h5py
import numpy as np
import cytools
from cytools import Polytope, fetch_polytopes
from geometry_charge_conventions import canonicalize_unique_charge_rows
from qed_divisor_assignment import (
    QEDAssignmentFailure,
    TERMINAL_FAILURE_CATEGORIES,
    classify_qed_leading_status,
    enumerate_assignment_pool,
    normalize_qcd_assignment,
    prime_divisor_charges,
    prime_divisor_intersection_graph,
    record_potential_match,
    select_qed_divisor,
    summarize_assignment_pool_rejections,
    stable_divisor_labels,
    validate_assignment_pool,
    write_visible_sector_hdf5,
)
from glimmers_schema11 import (
    CHARGE_FACTORIZED_SCHEMA_VERSION,
    MAXIMUM_EFT_ROWS,
    MINIMUM_EFT_ROWS,
    NORMALIZATION_MAP_VERSION,
    QCD_VOLUME_TARGET,
    QED_VOLUME_MAX,
    SCHEMA_VERSION as SCHEMA_1_1_VERSION,
    TARGET_GEOMETRY_COUNT,
    atomic_json_dump,
    atomic_jsonl_dump,
    ensure_fresh_output_root,
    estimate_storage,
    factorized_charge_metadata,
    sample_capacity_aware_assignments,
    stable_hash,
    stable_seed,
    write_eft_parquet,
    summarize_terminal_records,
)
from glimmers_proposal_controller import (
    ProposalControllerConfig,
    ProposalDecision,
    run_proposal_controller,
)
from glimmers_h491_diagnostics import NativeH491SamplerSettings, diagnose_h491
from glimmers_eft_row_schema import serialize_eft_row, validate_eft_row
from glimmers_provenance import ProvenanceError, production_provenance_gate
from glimmers_fresh_ensemble_manifest import (
    build_fresh_ensemble_manifest,
    default_sampler_by_h11,
    derive_ensemble_seeds,
    write_fresh_ensemble_manifest,
)


SCHEMA_VERSION = "cyaxiverse-ks-cy3-v9-schema-1.1"
MIN_CYTOOLS_VERSION = (1, 4, 0)
SOURCE_REFERENCES = (
    "arXiv:2008.01730v1",  # fair secondary-fan/triangulation sampling
    "arXiv:2309.10855v3",  # direct NTFE FRST construction
    "arXiv:2309.01831v1",  # stretched-cone potential-control criterion
    "arXiv:2309.13145v3",  # axion-photon light-threshold toy construction
    "arXiv:2305.06363v1",  # orientifold-compatible KS constructions
    "arXiv:2412.12012v1",  # fuzzy-axion QCD divisor-volume requirement
)
SAMPLING_SCHEMES = ("fair", "fast", "ntfe_fast")
NTFE_FACE_SAMPLERS = ("fast", "fair", "grow2d")
VOLUME_BACKENDS = ("fan", "historical_sparse_coo", "auto")
HISTORICAL_VOLUME_BACKEND = "historical_sparse_coo"
AUTO_VOLUME_BACKEND = "auto"
KS_MIRROR_DATASET = "calabi-yau-data/polytopes-4d"
KS_MIRROR_DATASET_URL = "https://huggingface.co/datasets/calabi-yau-data/polytopes-4d"
QCD_VOLUME_TOLERANCE = 1e-9
DIVISOR_VOLUME_TOLERANCE = 1e-8
CANONICAL_QCD_SELECTION_POLICY = "deterministic_minimal_dilation"
CANONICAL_QCD_CANDIDATE_ORDER = (
    "descending_positive_finite_tip_volume_at_or_below_target_then_ascending_divisor_index"
)
CANONICAL_QCD_CONTRACTION_CANDIDATE_ORDER = "legacy_input_order_when_allow_m_below_one"
CANONICAL_QCD_POST_SELECTION_FALLBACK = (
    "try_next_candidate_after_final_lower_bound_failure"
)
CANONICAL_QCD_QED_PREFILTER_SCHEMA_VERSION = (
    "cyaxiverse-canonical-qcd-qed-prefilter-1.0"
)
CANONICAL_QCD_QED_PREFILTER_POLICY = (
    "eft_canonical_qcd_intersecting_d7_final_qed_volume"
)
CANONICAL_QCD_QED_PREFILTER_FAILURE_STATUS = "qcd_qed_prefilter_shortfall"
KAEHLER_SLACK_TOLERANCE = 1e-6
POTENTIAL_RECONSTRUCTION_SCHEMA_VERSION = "cyaxiverse-potential-reconstruction-1.0"
POTENTIAL_RECONSTRUCTION_RTOL = 1e-10
POTENTIAL_RECONSTRUCTION_ATOL = 1e-10
GEOMETRY_ONLY_ARTIFACT_STATUS = "geometry_only"
ACCEPTED_GEOMETRY_ARTIFACT_STATUS = "accepted_geometry"
POOL_PENDING_ARTIFACT_STATUS = "pool_pending"


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


def require_cytools_capabilities(sampling_scheme, ntfe_face_sampler):
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
    if sampling_scheme == "ntfe_fast":
        required.append((Polytope, "ntfe_frts"))
    missing = [
        f"{owner.__name__}.{name}"
        for owner, name in required
        if not hasattr(owner, name)
    ]
    if sampling_scheme not in SAMPLING_SCHEMES:
        raise ValueError(f"Unsupported sampling scheme {sampling_scheme!r}.")
    if missing:
        raise RuntimeError(
            "Installed CYTools is missing required public APIs: " + ", ".join(missing)
        )
    if ntfe_face_sampler == "dualgnn":
        raise ValueError(
            "dualGNN face sampling is forbidden in schema 1.1; use ntfe_fast "
            "with triang_method=fast"
        )
    configure_mosek_license()


def _jsonable(value):
    """Convert CYTools/numpy scalar containers into deterministic JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def _sha256_file(path):
    """Return the SHA-256 digest of one persisted artifact without loading it whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_geometry_artifact(path):
    """Inspect an existing geometry artifact for overwrite provenance."""
    absolute_path = os.path.abspath(path)
    if not os.path.exists(absolute_path):
        return {"exists": False, "path": absolute_path}
    record = {
        "exists": True,
        "path": absolute_path,
        "byte_size": os.path.getsize(absolute_path),
        "sha256": None,
        "readable_hdf5": False,
        "schema_version": None,
        "geometry_id": None,
        "raw_geometry_id": None,
    }
    try:
        record["sha256"] = _sha256_file(absolute_path)
    except OSError as error:
        record["hash_error"] = f"{type(error).__name__}: {error}"
    try:
        with h5py.File(absolute_path, "r") as file:
            record["readable_hdf5"] = True
            schema_version = file.attrs.get("schema_version")
            if isinstance(schema_version, bytes):
                schema_version = schema_version.decode("utf-8", errors="replace")
            record["schema_version"] = schema_version
            metadata_json = file.attrs.get("construction_metadata_json")
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode("utf-8", errors="replace")
            if metadata_json:
                metadata = json.loads(metadata_json)
                record["geometry_id"] = metadata.get("cy3_fingerprint")
                raw_input = metadata.get("raw_frst_input") or {}
                record["raw_geometry_id"] = raw_input.get("raw_geometry_id")
    except (
        OSError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        record["read_error"] = f"{type(error).__name__}: {error}"
    return record


def prepare_geometry_artifact_write(path, allow_overwrite_existing_geometry=False):
    """Prepare a same-directory temporary path and record any prior artifact."""
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    prior_artifact = inspect_geometry_artifact(absolute_path)
    if prior_artifact["exists"] and not allow_overwrite_existing_geometry:
        raise FileExistsError(f"output collision: {absolute_path}")
    write_audit = {
        "allow_overwrite_existing_geometry": bool(allow_overwrite_existing_geometry),
        "overwrite_requested": bool(allow_overwrite_existing_geometry),
        "overwrite_performed": bool(prior_artifact["exists"]),
        "event": (
            "replaced_existing_geometry"
            if prior_artifact["exists"]
            else "created_new_geometry"
        ),
        "prior_artifact": prior_artifact if prior_artifact["exists"] else None,
    }
    temporary_path = (
        f"{absolute_path}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    return temporary_path, write_audit


def finalize_geometry_artifact_write(
    temporary_path, path, allow_overwrite_existing_geometry=False
):
    """Publish a complete HDF5 temporary file without an implicit overwrite."""
    absolute_path = os.path.abspath(path)
    if allow_overwrite_existing_geometry:
        os.replace(temporary_path, absolute_path)
        return
    os.link(temporary_path, absolute_path)
    os.unlink(temporary_path)


def cleanup_temporary_geometry_artifact(temporary_path):
    """Delete a failed or otherwise unpublished temporary geometry artifact."""
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)


def geometry_artifact_status(eft_mode, assignment_pool_status=None):
    """Classify the final artifact boundary without changing terminal failures."""
    if not eft_mode:
        return GEOMETRY_ONLY_ARTIFACT_STATUS
    if assignment_pool_status == "complete_eligible_ordered_pool":
        return ACCEPTED_GEOMETRY_ARTIFACT_STATUS
    return POOL_PENDING_ARTIFACT_STATUS


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
            raise OrientifoldValidationFailure(
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
        raise OrientifoldValidationFailure(
            "The orientifold lattice action does not preserve the selected FRST."
        )

    basis_matrix = np.asarray(topology["basis_matrix"], dtype=float)
    divisor_points = np.concatenate(
        (np.asarray([0], dtype=int), topology["prime_toric_divisors"])
    )
    if basis_matrix.shape[1] != divisor_points.size:
        raise OrientifoldValidationFailure(
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
        raise OrientifoldValidationFailure(
            "The orientifold action does not preserve the prime toric divisor set."
        ) from exc
    if mapped_divisor_positions[0] != 0:
        raise OrientifoldValidationFailure("The orientifold action must fix the origin label.")
    prime_image_indices = mapped_divisor_positions[1:] - 1
    if np.any(prime_image_indices < 0) or np.any(
        prime_image_indices >= topology["prime_toric_divisors"].size
    ):
        raise OrientifoldValidationFailure("The orientifold prime-divisor image map is invalid.")
    permutation = np.zeros((divisor_points.size, divisor_points.size), dtype=float)
    permutation[np.arange(divisor_points.size), mapped_divisor_positions] = 1.0
    transformed_basis = basis_matrix @ permutation
    coefficients, _, _, _ = np.linalg.lstsq(
        basis_matrix.T, transformed_basis.T, rcond=None
    )
    h2_matrix = coefficients.T
    integral_h2 = np.rint(h2_matrix).astype(int)
    if not np.allclose(h2_matrix, integral_h2, atol=1e-8):
        raise OrientifoldValidationFailure(
            "The orientifold action does not induce an integral action in the "
            "exported divisor basis."
        )
    if not np.allclose(integral_h2 @ basis_matrix, transformed_basis, atol=1e-8):
        raise OrientifoldValidationFailure("Could not express the orientifold action in H2.")
    if not np.array_equal(integral_h2 @ integral_h2, np.eye(topology["h11"], dtype=int)):
        raise OrientifoldValidationFailure("The induced H2 action is not an involution.")

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
            "prime_divisor_image_indices": prime_image_indices,
            "prime_divisor_invariant_indices": np.flatnonzero(
                prime_image_indices == np.arange(prime_image_indices.size)
            ),
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
        from scipy.sparse import csc_matrix, eye as sparse_eye

        equality = h2_matrix.T - np.eye(h2_matrix.shape[0])
        qp_hyperplanes = csc_matrix(hyperplanes)
        solvers = sorted(available_solvers)
        if not solvers:
            raise RuntimeError("qpsolvers found no solver for invariant Kähler validation.")
        invariant_tip = None
        for solver in solvers:
            try:
                invariant_tip = solve_qp(
                    sparse_eye(reference_tip.size, format="csc"),
                    np.zeros(reference_tip.size),
                    G=-qp_hyperplanes,
                    h=-np.ones(hyperplanes.shape[0]),
                    A=csc_matrix(equality),
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


def summarize_array_structure(array):
    """Summarize one numerical array for the structural topology audit."""
    values = np.asarray(array)
    try:
        finite = bool(np.all(np.isfinite(values)))
    except TypeError:
        finite = None
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": finite,
    }


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
    if basis_matrix.ndim != 2 or basis_matrix.shape[0] != h11:
        raise RuntimeError(
            "CYTools returned an unexpected divisor-basis matrix shape; "
            f"got {basis_matrix.shape}, expected ({h11}, n_divisors)."
        )
    if prime_toric_divisors.ndim != 1:
        raise RuntimeError(
            "CYTools returned an unexpected prime-divisor label shape; "
            f"got {prime_toric_divisors.shape}, expected (n_prime_divisors,)."
        )
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
    if mori.ndim != 2 or mori.shape[1] != h11:
        raise RuntimeError(
            "CYTools returned an unexpected Mori-cone shape; "
            f"got {mori.shape}, expected (n_rays, {h11})."
        )
    kahler = cy.toric_kahler_cone()
    kahler_rays = None
    if export_kahler_rays:
        kahler_rays = np.asarray(kahler.rays(), dtype=float)
    kahler_hyperplanes = np.asarray(kahler.hyperplanes(), dtype=float)
    if kahler_hyperplanes.ndim != 2 or kahler_hyperplanes.shape[1] != h11:
        raise RuntimeError(
            "CYTools returned an unexpected Kähler-hyperplane shape; "
            f"got {kahler_hyperplanes.shape}, expected (n_hyperplanes, {h11})."
        )
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
    simplices = np.asarray(triangulation.simplices(), dtype=int)
    triangulation_id = f"frst-sha256:{_sha256_json(simplices.tolist())}"
    fingerprint_payload = {
        "polytope_id": polytope_id,
        "simplices": simplices.tolist(),
        "face_restriction_dim2": topology["face_restriction_dim2"],
        "h11": topology["h11"],
        "h21": topology["h21"],
        "kappa": topology["kappa"],
        "c2": topology["c2"],
    }
    cy3_fingerprint = f"topological-sha256:{_sha256_json(fingerprint_payload)}"
    return triangulation_id, cy3_fingerprint


def prime_divisor_neighbors(prime_labels, face_simplices):
    """Build the prime-divisor intersection graph from triangulated two-faces.

    The returned tuple is indexed by the CYTools ``prime_toric_divisors``
    order.  Two divisors are neighbors when their point labels occur together
    in an edge of a triangulated two-face, which is the toric intersection
    criterion used by the axion-photon toy construction.
    """
    labels = np.asarray(prime_labels, dtype=int).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != labels.size:
        raise RuntimeError("prime toric divisor labels must be unique and non-empty")
    positions = {int(label): index for index, label in enumerate(labels)}
    simplices = np.asarray(face_simplices, dtype=int)
    if simplices.size == 0:
        return tuple(() for _ in labels)
    if simplices.ndim != 2:
        raise RuntimeError(
            "triangulation two-face simplices must be a two-dimensional array"
        )
    neighbors = [set() for _ in labels]
    for simplex in simplices:
        face_positions = sorted(
            {positions[int(label)] for label in simplex if int(label) in positions}
        )
        for first, second in itertools.combinations(face_positions, 2):
            neighbors[first].add(second)
            neighbors[second].add(first)
    return tuple(tuple(sorted(values)) for values in neighbors)


def _visible_qcd_candidates(policy, orientifold, neighbors):
    """Return QCD candidates that have an orientifold-compatible QED partner."""
    if policy == "none":
        return None
    if policy != "intersecting_d7":
        raise ValueError(f"unsupported visible-sector policy {policy!r}")
    # ``fan_invariant`` means that the lattice action, selected FRST, prime
    # divisor map, and induced H2 involution have been validated.  The
    # optional invariant-Kahler-subspace check upgrades this to ``validated``
    # when requested, but it is deliberately not part of the reference run.
    if not orientifold.get("requested", False) or orientifold.get("status") not in {
        "fan_invariant",
        "validated",
    }:
        raise NoVisibleSectorAssignment(
            "intersecting_d7 requires a validated lattice-action orientifold"
        )
    if orientifold.get("involution_type") != "O3/O7":
        raise NoVisibleSectorAssignment(
            "intersecting_d7 requires an O3/O7 orientifold for D7 gauge cycles"
        )
    image_indices = np.asarray(orientifold["prime_divisor_image_indices"], dtype=int)
    invariant = image_indices == np.arange(image_indices.size)
    candidates = [
        qcd_index
        for qcd_index, qeds in enumerate(neighbors)
        if any(invariant[qed_index] for qed_index in qeds)
    ]
    return candidates


def canonical_qcd_qed_prefilter_active(
    *, eft_mode, moduli_policy, visible_sector_policy
):
    """Return whether the production-only canonical QED prefilter is active."""
    return bool(
        eft_mode
        and moduli_policy == "canonical_qcd"
        and visible_sector_policy == "intersecting_d7"
    )


def _inactive_canonical_qcd_qed_prefilter_metadata(
    *, eft_mode, moduli_policy, visible_sector_policy
):
    """Describe why the production-only canonical QED prefilter is inactive."""
    return {
        "schema_version": CANONICAL_QCD_QED_PREFILTER_SCHEMA_VERSION,
        "policy": CANONICAL_QCD_QED_PREFILTER_POLICY,
        "active": False,
        "status": "inactive_outside_production_path",
        "activation_contract": {
            "eft_mode": True,
            "moduli_policy": "canonical_qcd",
            "visible_sector_policy": "intersecting_d7",
        },
        "requested_path": {
            "eft_mode": bool(eft_mode),
            "moduli_policy": str(moduli_policy),
            "visible_sector_policy": str(visible_sector_policy),
        },
        "reason": (
            "apply only to eft_mode=true, moduli_policy=canonical_qcd, "
            "visible_sector_policy=intersecting_d7"
        ),
    }


def prefilter_canonical_qcd_candidates(
    prime_tau0,
    candidate_indices,
    neighbors,
    invariant_mask,
    qcd_volume_target,
    effective_qed_volume_max=QED_VOLUME_MAX,
    max_m=1_000_000.0,
    *,
    allow_m_below_one=False,
):
    """Filter canonical QCD candidates by their normalized QED neighbors.

    Compute the existing canonical radial scale for every candidate that is
    admissible under the current ``m`` policy.  Keep a candidate only when a
    distinct intersecting neighbor is orientifold-invariant and its final
    volume ``m**2 * prime_tau0[qed_index]`` is at most the inclusive QED
    bound.  Do not apply charge or assignment-pool checks here; the complete
    assignment pool remains the authoritative production gate.
    """
    prime_tau0 = np.asarray(prime_tau0, dtype=float).reshape(-1)
    try:
        qcd_volume_target = float(qcd_volume_target)
        effective_qed_volume_max = float(effective_qed_volume_max)
        max_m = float(max_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("QCD/QED prefilter parameters must be numeric") from exc
    if (
        not np.isfinite(qcd_volume_target)
        or qcd_volume_target <= 0.0
        or not np.isfinite(effective_qed_volume_max)
        or effective_qed_volume_max <= 0.0
        or not np.isfinite(max_m)
        or max_m <= 0.0
    ):
        raise ValueError("QCD/QED prefilter parameters must be finite and positive")
    if len(neighbors) != prime_tau0.size:
        raise ValueError("QCD/QED prefilter neighbors have an inconsistent shape")
    invariant = np.asarray(invariant_mask, dtype=bool).reshape(-1)
    if invariant.shape != prime_tau0.shape:
        raise ValueError("QCD/QED prefilter invariant mask has an inconsistent shape")

    normalized_candidate_indices = []
    for value in candidate_indices:
        try:
            index = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("QCD candidate indices must be integers") from exc
        normalized_candidate_indices.append(index)

    candidate_records = []
    eligible_candidate_indices = []
    for qcd_index in normalized_candidate_indices:
        record = {
            "qcd_index": qcd_index,
            "status": "rejected",
            "eligible_qed_indices": [],
            "neighbor_records": [],
        }
        if not 0 <= qcd_index < prime_tau0.size:
            record["rejection_reason"] = "qcd_index_out_of_range"
            candidate_records.append(record)
            continue
        qcd_tip_volume = float(prime_tau0[qcd_index])
        record["qcd_tip_volume"] = qcd_tip_volume
        if not np.isfinite(qcd_tip_volume) or qcd_tip_volume <= 0.0:
            record["rejection_reason"] = "qcd_tip_volume_not_positive_finite"
            candidate_records.append(record)
            continue
        candidate_m = math.sqrt(qcd_volume_target / qcd_tip_volume)
        record["radial_scale"] = float(candidate_m)
        if not np.isfinite(candidate_m):
            record["rejection_reason"] = "radial_scale_not_finite"
            candidate_records.append(record)
            continue
        if not allow_m_below_one and candidate_m < 1.0:
            record["rejection_reason"] = "radial_contraction_disallowed"
            candidate_records.append(record)
            continue
        if candidate_m > max_m:
            record["rejection_reason"] = "radial_scale_exceeds_max_m"
            candidate_records.append(record)
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            volume_scale = candidate_m**2
            final_qed_volumes = volume_scale * prime_tau0
        record["volume_scale"] = float(volume_scale)
        for qed_index in sorted({int(value) for value in neighbors[qcd_index]}):
            neighbor_record = {
                "qed_index": qed_index,
                "distinct": qed_index != qcd_index,
                "orientifold_invariant": False,
                "final_qed_volume": None,
                "within_effective_max": False,
                "eligible": False,
            }
            if not 0 <= qed_index < prime_tau0.size:
                neighbor_record["reason"] = "qed_index_out_of_range"
            elif qed_index == qcd_index:
                neighbor_record["reason"] = "qcd_qed_must_be_distinct"
            else:
                neighbor_record["orientifold_invariant"] = bool(invariant[qed_index])
                if not invariant[qed_index]:
                    neighbor_record["reason"] = "qed_divisor_not_orientifold_invariant"
                    record["neighbor_records"].append(neighbor_record)
                    continue
                final_qed_volume = float(final_qed_volumes[qed_index])
                neighbor_record["final_qed_volume"] = final_qed_volume
                within_effective_max = bool(
                    np.isfinite(final_qed_volume)
                    and final_qed_volume <= effective_qed_volume_max
                )
                neighbor_record["within_effective_max"] = within_effective_max
                if within_effective_max:
                    neighbor_record["eligible"] = True
                    record["eligible_qed_indices"].append(qed_index)
                    neighbor_record["reason"] = "accepted"
                else:
                    neighbor_record["reason"] = "qed_volume_exceeds_effective_max"
            record["neighbor_records"].append(neighbor_record)
        if record["eligible_qed_indices"]:
            record["status"] = "accepted"
            eligible_candidate_indices.append(qcd_index)
        else:
            record["rejection_reason"] = "no_eligible_qed_neighbor"
        candidate_records.append(record)

    return {
        "schema_version": CANONICAL_QCD_QED_PREFILTER_SCHEMA_VERSION,
        "policy": CANONICAL_QCD_QED_PREFILTER_POLICY,
        "active": True,
        "status": "passed" if eligible_candidate_indices else "no_candidate_survived",
        "failure_status": (
            None
            if eligible_candidate_indices
            else CANONICAL_QCD_QED_PREFILTER_FAILURE_STATUS
        ),
        "qcd_volume_target": qcd_volume_target,
        "effective_qed_volume_max": effective_qed_volume_max,
        "qed_volume_comparison": "less_than_or_equal_to_effective_max",
        "allow_m_below_one": bool(allow_m_below_one),
        "max_m": max_m,
        "candidate_indices_input": normalized_candidate_indices,
        "eligible_candidate_indices": eligible_candidate_indices,
        "candidate_records": candidate_records,
        "assignment_pool_authoritative": True,
        "assignment_pool_authority_policy": (
            "prefilter_does_not_replace_complete_validated_ordered_assignment_pool"
        ),
    }


class PrefactorCriterionNotMet(RuntimeError):
    """The current FRST's tip cannot satisfy the potential-control criterion."""


class NoPhysicalKaehlerPoint(RuntimeError):
    """No sampled point has positive volumes on effective-divisor cone rays."""


class NoQcdDivisorVolume(RuntimeError):
    """No stretched point satisfies the prime-divisor QCD volume window."""


class NoStandardModelAssignment(RuntimeError):
    """No pairwise-intersecting prime-divisor triple exists for the FRST."""


class FinalGeometryValidationFailed(RuntimeError):
    """The rescaled candidate leaves the physical Kähler-domain checks."""


def build_divisor_volume_evidence(
    prime_divisor_indices,
    prime_divisor_labels,
    prime_divisor_volumes,
    effective_cone_rays,
    effective_divisor_volumes,
    basis,
    min_prime_divisor_volume,
    min_divisor_volume,
    *,
    tolerance=DIVISOR_VOLUME_TOLERANCE,
):
    """Validate final divisor volumes and return replayable geometry evidence."""
    prime_indices = np.asarray(prime_divisor_indices, dtype=np.int64).reshape(-1)
    prime_labels = np.asarray(prime_divisor_labels, dtype=np.int64)
    prime_volumes = np.asarray(prime_divisor_volumes, dtype=float).reshape(-1)
    effective_rays = np.asarray(effective_cone_rays, dtype=np.int64)
    effective_volumes = np.asarray(effective_divisor_volumes, dtype=float).reshape(-1)
    basis_array = np.asarray(basis, dtype=np.int64)
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("divisor-volume tolerance must be finite and non-negative")
    if prime_indices.size != prime_volumes.size:
        raise RuntimeError("prime-divisor indices and volumes have inconsistent lengths")
    if not np.array_equal(prime_indices, np.arange(prime_indices.size)):
        raise RuntimeError(
            "prime-divisor indices must be the zero-based CYTools ordering"
        )
    if prime_labels.ndim != 2 or prime_labels.shape[0] != prime_indices.size:
        raise RuntimeError("prime-divisor labels and indices have inconsistent shapes")
    if prime_labels.shape[0] and np.unique(prime_labels, axis=0).shape[0] != prime_labels.shape[0]:
        raise RuntimeError("prime-divisor labels must be unique")
    if effective_rays.ndim != 2 or effective_rays.shape[0] != effective_volumes.size:
        raise RuntimeError(
            "effective-cone rays and volumes have inconsistent shapes"
        )
    if basis_array.ndim == 0:
        raise RuntimeError("divisor-basis ordering must be an array")
    if not (
        np.all(np.isfinite(prime_volumes))
        and np.all(np.isfinite(effective_volumes))
    ):
        raise RuntimeError("required divisor volumes are non-finite")
    min_prime = float(np.min(prime_volumes)) if prime_volumes.size else math.inf
    min_effective = (
        float(np.min(effective_volumes)) if effective_volumes.size else math.inf
    )
    prime_bound_passed = bool(
        min_prime >= float(min_prime_divisor_volume) - tolerance
    )
    effective_bound_passed = bool(
        min_effective >= float(min_divisor_volume) - tolerance
    )
    evidence = {
        "schema_version": "cyaxiverse-divisor-volume-evidence-1.0",
        "validation_status": "passed"
        if prime_bound_passed and effective_bound_passed
        else "failed",
        "volume_tolerance": tolerance,
        "prime_divisor_volume_lower_bound": float(min_prime_divisor_volume),
        "effective_divisor_volume_lower_bound": float(min_divisor_volume),
        "prime_divisor_index_base": 0,
        "prime_divisor_indices": prime_indices.tolist(),
        "prime_divisor_labels": prime_labels.tolist(),
        "prime_divisor_volumes": prime_volumes.tolist(),
        "effective_cone_ray_index_base": 0,
        "effective_cone_ray_indices": list(range(effective_rays.shape[0])),
        "effective_cone_rays": effective_rays.tolist(),
        "effective_divisor_volumes": effective_volumes.tolist(),
        "basis_order": basis_array.tolist(),
        "basis_convention": (
            "CYTools divisor_basis(include_origin=True); vectors are in this order"
        ),
        "minimum_prime_divisor_volume": min_prime,
        "minimum_effective_divisor_volume": min_effective,
        "checks": {
            "finite_prime_divisor_volumes": True,
            "finite_effective_divisor_volumes": True,
            "prime_divisor_lower_bound": prime_bound_passed,
            "effective_divisor_lower_bound": effective_bound_passed,
        },
    }
    if not prime_bound_passed or not effective_bound_passed:
        raise FinalGeometryValidationFailed(
            "final divisor-volume lower-bound validation failed: "
            f"minimum_prime={min_prime:.6g}, minimum_effective={min_effective:.6g}, "
            f"required_prime={float(min_prime_divisor_volume):.6g}, "
            f"required_effective={float(min_divisor_volume):.6g}, "
            f"tolerance={tolerance:.3g}"
        )
    return evidence


def validate_final_qcd_normalization(
    *,
    point,
    radial_scale,
    max_m,
    allow_m_below_one,
    qcd_divisor_index,
    qcd_volume_target,
    qcd_volume_min,
    qcd_volume_max,
    cy_volume,
    curve_volumes,
    kaehler_slack,
    inverse_metric,
    prime_divisor_volumes,
    effective_divisor_volumes,
    min_prime_divisor_volume,
    min_divisor_volume,
):
    """Validate one final normalized point without repairing its data."""
    point = np.asarray(point, dtype=float).reshape(-1)
    curve_volumes = np.asarray(curve_volumes, dtype=float).reshape(-1)
    kaehler_slack = np.asarray(kaehler_slack, dtype=float).reshape(-1)
    inverse_metric = np.asarray(inverse_metric, dtype=float)
    prime_volumes = np.asarray(prime_divisor_volumes, dtype=float).reshape(-1)
    effective_volumes = np.asarray(effective_divisor_volumes, dtype=float).reshape(-1)
    radial_scale = float(radial_scale)
    qcd_volume_target = (
        None if qcd_volume_target is None else float(qcd_volume_target)
    )
    qcd_volume_min = float(qcd_volume_min)
    qcd_volume_max = float(qcd_volume_max)
    qcd_index_valid = 0 <= int(qcd_divisor_index) < prime_volumes.size
    qcd_volume = (
        float(prime_volumes[int(qcd_divisor_index)])
        if qcd_index_valid
        else math.nan
    )
    metric_eigenvalues = (
        np.linalg.eigvalsh(0.5 * (inverse_metric + inverse_metric.T))
        if inverse_metric.ndim == 2 and inverse_metric.shape[0] == inverse_metric.shape[1]
        else np.asarray([], dtype=float)
    )
    checks = {
        "finite_point": bool(np.all(np.isfinite(point))),
        "finite_cy_volume": bool(np.isfinite(cy_volume)),
        "positive_cy_volume": bool(np.isfinite(cy_volume) and cy_volume > 0.0),
        "finite_curve_volumes": bool(np.all(np.isfinite(curve_volumes))),
        "positive_curve_volumes": bool(
            np.all(np.isfinite(curve_volumes))
            and (not curve_volumes.size or np.min(curve_volumes) > 0.0)
        ),
        "finite_kaehler_slack": bool(np.all(np.isfinite(kaehler_slack))),
        "cone_membership": bool(
            np.all(np.isfinite(kaehler_slack))
            and (not kaehler_slack.size or np.min(kaehler_slack) >= 1.0 - KAEHLER_SLACK_TOLERANCE)
        ),
        "finite_inverse_metric": bool(np.all(np.isfinite(inverse_metric))),
        "positive_inverse_metric": bool(
            metric_eigenvalues.size
            and np.all(np.isfinite(metric_eigenvalues))
            and np.min(metric_eigenvalues) > 0.0
        ),
        "finite_prime_divisor_volumes": bool(np.all(np.isfinite(prime_volumes))),
        "finite_effective_divisor_volumes": bool(
            np.all(np.isfinite(effective_volumes))
        ),
        "prime_divisor_lower_bound": bool(
            prime_volumes.size
            and np.all(np.isfinite(prime_volumes))
            and np.min(prime_volumes)
            >= float(min_prime_divisor_volume) - DIVISOR_VOLUME_TOLERANCE
        ),
        "effective_divisor_lower_bound": bool(
            effective_volumes.size
            and np.all(np.isfinite(effective_volumes))
            and np.min(effective_volumes)
            >= float(min_divisor_volume) - DIVISOR_VOLUME_TOLERANCE
        ),
        "qcd_divisor_index": qcd_index_valid,
        "qcd_volume_target": bool(
            qcd_index_valid
            and np.isclose(
                qcd_volume,
                qcd_volume_target,
                rtol=0.0,
                atol=QCD_VOLUME_TOLERANCE,
            )
        )
        if qcd_volume_target is not None
        else True,
        "qcd_volume_window": bool(
            qcd_index_valid
            and np.isclose(
                qcd_volume,
                qcd_volume_target,
                rtol=0.0,
                atol=QCD_VOLUME_TOLERANCE,
            )
        )
        if qcd_volume_target is not None
        else bool(
            qcd_index_valid
            and qcd_volume_min - DIVISOR_VOLUME_TOLERANCE
            <= qcd_volume
            <= qcd_volume_max + DIVISOR_VOLUME_TOLERANCE
        ),
        "radial_scale_finite": bool(np.isfinite(radial_scale)),
        "radial_scale_positive": bool(np.isfinite(radial_scale) and radial_scale > 0.0),
        "radial_scale_upper_bound": bool(
            np.isfinite(radial_scale) and radial_scale <= float(max_m)
        ),
        "radial_scale_lower_bound": bool(
            np.isfinite(radial_scale)
            and (allow_m_below_one or radial_scale >= 1.0)
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    result = {
        "schema_version": "cyaxiverse-qcd-normalization-validation-1.0",
        "validation_status": "passed" if not failed_checks else "failed",
        "failure_checks": failed_checks,
        "checks": checks,
        "qcd_volume": qcd_volume,
        "qcd_volume_target": qcd_volume_target,
        "qcd_volume_residual": (
            None if qcd_volume_target is None else abs(qcd_volume - qcd_volume_target)
        ),
        "qcd_volume_tolerance": QCD_VOLUME_TOLERANCE,
        "divisor_volume_tolerance": DIVISOR_VOLUME_TOLERANCE,
        "kaehler_slack_tolerance": KAEHLER_SLACK_TOLERANCE,
        "radial_scale": radial_scale,
        "allow_m_below_one": bool(allow_m_below_one),
        "repair_policy": "none",
        "failure_status": "qcd_normalization_failure",
    }
    if failed_checks:
        raise FinalGeometryValidationFailed(
            "post-normalization QCD validation failed: "
            + ", ".join(failed_checks)
        )
    return result


class NoVisibleSectorAssignment(RuntimeError):
    """No orientifold-compatible intersecting QCD/QED divisor pair exists."""


class OrientifoldValidationFailure(RuntimeError):
    """The supplied orientifold does not preserve the selected geometry."""


def _candidate_terminal_status(exc):
    """Map implementation failures onto the schema 1.1 terminal vocabulary."""
    if isinstance(exc, QEDAssignmentFailure):
        if exc.category == "no_eligible_qed_divisor":
            return "no_eligible_intersecting_qed_pair"
        return exc.category
    if isinstance(exc, FileExistsError):
        return "output_collision"
    if isinstance(exc, OSError):
        return "io_failure"
    if isinstance(exc, PrefactorCriterionNotMet):
        return "kaehler_tip_failure"
    if isinstance(exc, NoPhysicalKaehlerPoint):
        return "kaehler_point_shortfall"
    if isinstance(exc, NoQcdDivisorVolume):
        return "qcd_normalization_failure"
    if isinstance(exc, NoStandardModelAssignment):
        return "topology_or_cone_error"
    if isinstance(exc, NoVisibleSectorAssignment):
        return "no_eligible_intersecting_qed_pair"
    if isinstance(exc, OrientifoldValidationFailure):
        return "orientifold_invariance_failure"
    if isinstance(exc, FinalGeometryValidationFailed):
        return "qcd_normalization_failure"
    return "numerical_geometry_failure"


def _triangulation_hashes(triangulation):
    """Return full and two-face identities without retaining live CYTools data."""
    full_hash = stable_hash(np.asarray(triangulation.simplices()).tolist())
    two_face_hash = None
    try:
        two_face_hash = stable_hash(
            np.asarray(triangulation.simplices(on_faces_dim=2), dtype=object).tolist()
        )
    except Exception:
        pass
    return full_hash, two_face_hash


def sample_stretched_kaehler_points(
    kahler_cone,
    reference_tip,
    rng,
    attempts,
    report,
    solver_used=None,
    *,
    point_seed=None,
    diagnostics=None,
    include_metadata=False,
):
    """Yield randomized points in the same stretched Kähler region.

    The canonical SKC tip is the norm-minimizing point in ``H t >= 1``.  At
    high h11 it need not lie in the useful angular region.  Each later point
    is the Euclidean projection of a random target onto that same polyhedron,
    so it remains inside the Kähler cone with every curve-wall distance >= 1.
    """
    mosek_license = configure_mosek_license()
    reference_point = np.asarray(reference_tip, dtype=float)
    reference_metadata = {
        "attempt_index": 1,
        "point_kind": "canonical_tip",
        "point_seed": None if point_seed is None else int(point_seed),
        "solver": None,
        "point": reference_point,
    }
    yield reference_metadata if include_metadata else reference_point
    if attempts <= 1:
        return
    try:
        from qpsolvers import available_solvers, solve_qp
        from scipy.sparse import csc_matrix, eye as sparse_eye
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
    qp_hyperplanes = csc_matrix(hyperplanes)
    identity = sparse_eye(reference_tip.size, format="csc")
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
        attempt_seed = (
            None
            if point_seed is None
            else stable_seed("kaehler-point-attempt", point_seed, number)
        )
        attempt_rng = (
            rng if attempt_seed is None else np.random.default_rng(attempt_seed)
        )
        direction = attempt_rng.normal(size=reference_tip.size)
        direction /= max(float(np.linalg.norm(direction)), np.finfo(float).tiny)
        # A logarithmic range makes this explore angles rather than merely a
        # tiny neighborhood of the norm-minimizing reference tip.
        target = target_norm * (2.0 ** attempt_rng.uniform(-1.0, 4.0)) * direction
        report(f"projecting randomized Kähler point {number}/{attempts}")
        point = None
        selected_solver = None
        solver_errors = []
        for solver in solvers:
            try:
                point = solve_qp(
                    identity,
                    -target,
                    G=-qp_hyperplanes,
                    h=-np.ones(hyperplanes.shape[0]),
                    solver=solver,
                    verbose=False,
                )
            except Exception as exc:
                solver_errors.append(f"{solver}: {type(exc).__name__}: {exc}")
                point = None
            if point is not None:
                selected_solver = solver
                break
        if point is None:
            report(
                "randomized Kähler projection failed with all available "
                f"solvers ({', '.join(solvers)}); skipping candidate"
            )
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "attempt_index": number,
                        "point_kind": "randomized_projection",
                        "point_seed": attempt_seed,
                        "attempted": True,
                        "point_status": "skipped",
                        "solver": None,
                        "failure_reason": "all quadratic-program solvers failed",
                        "solver_errors": solver_errors,
                    }
                )
            continue
        point = np.asarray(point, dtype=float)
        if np.all(np.isfinite(point)) and np.min(hyperplanes @ point) >= 1.0 - 1e-6:
            report(f"randomized Kähler projection succeeded with {selected_solver}")
            if solver_used is not None:
                solver_used.append(selected_solver)
            proposal = {
                "attempt_index": number,
                "point_kind": "randomized_projection",
                "point_seed": attempt_seed,
                "solver": selected_solver,
                "point": point,
            }
            yield proposal if include_metadata else point
        else:
            report("randomized Kähler projection was infeasible; skipping candidate")
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "attempt_index": number,
                        "point_kind": "randomized_projection",
                        "point_seed": attempt_seed,
                        "attempted": True,
                        "point_status": "skipped",
                        "solver": selected_solver,
                        "failure_reason": "projected point was non-finite or outside the stretched cone",
                    }
                )


def resolve_volume_backend(h11, volume_backend=None, sampling_metadata=None):
    """Resolve and validate the selected Stage-2 volume backend.

    Default to CYTools' current Fan path.  The historical sparse COO path is
    deliberately restricted to h11=491 because it is a reproduction
    compatibility route for the high-h11 construction, not a general
    replacement for CYTools.  The explicit ``auto`` policy selects that
    historical route only at h11=491 and selects Fan elsewhere.
    """
    if volume_backend is None:
        metadata_backend = (
            None
            if sampling_metadata is None
            else sampling_metadata.get("volume_backend")
        )
        volume_backend = metadata_backend or os.environ.get(
            "CYAX_VOLUME_BACKEND", "fan"
        )
    volume_backend = str(volume_backend)
    if volume_backend not in VOLUME_BACKENDS:
        raise ValueError(
            f"volume_backend must be one of {VOLUME_BACKENDS}, got {volume_backend!r}"
        )
    if volume_backend == AUTO_VOLUME_BACKEND:
        volume_backend = (
            HISTORICAL_VOLUME_BACKEND if int(h11) == 491 else "fan"
        )
    if volume_backend == HISTORICAL_VOLUME_BACKEND and int(h11) != 491:
        raise ValueError(
            "historical_sparse_coo is restricted to h11=491; "
            f"received h11={int(h11)}"
        )
    return volume_backend


def evaluate_kaehler_point(
    cy,
    kahler_cone,
    effective_cone_rays,
    point,
    *,
    attempt_index,
    point_kind,
    point_seed=None,
    solver=None,
    min_prime_divisor_volume=1.0,
    min_divisor_volume=1.0,
    volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
    enforce_divisor_volume_lower_bounds=True,
    volume_backend="fan",
    kappa=None,
    glsm_charge_matrix=None,
    mori_cone=None,
):
    """Evaluate one Kähler point without retaining live CYTools objects.

    Defer only the divisor lower-bound checks when the point is an angular
    direction that will be homogeneously normalized later.
    """
    diagnostic = {
        "attempt_index": int(attempt_index),
        "point_kind": point_kind,
        "point_seed": None if point_seed is None else int(point_seed),
        "attempted": True,
        "solver": solver,
        "point_status": "failed",
        "checks": {},
        "divisor_volume_tolerance": float(volume_tolerance),
        "prime_divisor_volume_lower_bound": float(min_prime_divisor_volume),
        "effective_divisor_volume_lower_bound": float(min_divisor_volume),
        "divisor_volume_lower_bounds_enforced": bool(
            enforce_divisor_volume_lower_bounds
        ),
        "volume_backend": str(volume_backend),
    }
    values = None
    try:
        point_array = np.asarray(point, dtype=float).reshape(-1)
        diagnostic["coordinate_dimension"] = int(point_array.size)
        diagnostic["point_norm"] = float(np.linalg.norm(point_array))
        diagnostic["point_sha256"] = stable_hash(point_array.tolist())
        hyperplanes = np.asarray(kahler_cone.hyperplanes(), dtype=float)
        slack = hyperplanes @ point_array
        finite_coordinates = bool(np.all(np.isfinite(point_array)))
        finite_slack = bool(np.all(np.isfinite(slack)))
        minimum_slack = float(np.min(slack)) if slack.size else math.inf
        diagnostic["minimum_kaehler_slack"] = minimum_slack
        diagnostic["checks"]["finite_coordinates"] = finite_coordinates
        diagnostic["checks"]["cone_membership"] = bool(
            finite_slack and minimum_slack >= 1.0 - 1e-6
        )
        if not finite_coordinates:
            raise ValueError("Kähler point coordinates are non-finite")
        if not diagnostic["checks"]["cone_membership"]:
            raise ValueError(
                f"Kähler point is outside the stretched cone: minimum slack {minimum_slack:.6g}"
            )

        geometry = _compute_volume_geometry(
            cy,
            point_array,
            volume_backend=volume_backend,
            kappa=kappa,
            glsm_charge_matrix=glsm_charge_matrix,
            mori_cone=mori_cone,
        )
        cy_volume = geometry["cy_volume"]
        curve_volumes = geometry["curve_volumes"]
        basis_divisor_volumes = geometry["basis_divisor_volumes"]
        prime_divisor_volumes = geometry["prime_divisor_volumes"]
        effective_divisor_volumes = np.asarray(
            effective_cone_rays, dtype=float
        ) @ basis_divisor_volumes
        inverse_metric = geometry["inverse_metric"]
        metric_eigenvalues = np.linalg.eigvalsh(inverse_metric)

        diagnostic.update(
            {
                "cy_volume": cy_volume,
                "minimum_curve_volume": (
                    float(np.min(curve_volumes)) if curve_volumes.size else math.inf
                ),
                "minimum_basis_divisor_volume": (
                    float(np.min(basis_divisor_volumes))
                    if basis_divisor_volumes.size
                    else math.inf
                ),
                "minimum_prime_divisor_volume": (
                    float(np.min(prime_divisor_volumes))
                    if prime_divisor_volumes.size
                    else math.inf
                ),
                "minimum_effective_divisor_volume": (
                    float(np.min(effective_divisor_volumes))
                    if effective_divisor_volumes.size
                    else math.inf
                ),
                "minimum_metric_eigenvalue": (
                    float(np.min(metric_eigenvalues))
                    if metric_eigenvalues.size
                    else math.inf
                ),
            }
        )
        diagnostic["checks"].update(
            {
                "finite_cy_volume": bool(np.isfinite(cy_volume)),
                "positive_cy_volume": bool(np.isfinite(cy_volume) and cy_volume > 0.0),
                "finite_curve_volumes": bool(np.all(np.isfinite(curve_volumes))),
                "positive_curve_volumes": bool(
                    np.all(np.isfinite(curve_volumes))
                    and (not curve_volumes.size or np.min(curve_volumes) > 0.0)
                ),
                "finite_basis_divisor_volumes": bool(
                    np.all(np.isfinite(basis_divisor_volumes))
                ),
                "positive_basis_divisor_volumes": bool(
                    np.all(np.isfinite(basis_divisor_volumes))
                    and (not basis_divisor_volumes.size or np.min(basis_divisor_volumes) > 0.0)
                ),
                "finite_prime_divisor_volumes": bool(
                    np.all(np.isfinite(prime_divisor_volumes))
                ),
                "positive_prime_divisor_volumes": bool(
                    np.all(np.isfinite(prime_divisor_volumes))
                    and (not prime_divisor_volumes.size or np.min(prime_divisor_volumes) > 0.0)
                ),
                "prime_divisor_volume_lower_bound": bool(
                    np.all(np.isfinite(prime_divisor_volumes))
                    and (
                        not prime_divisor_volumes.size
                        or np.min(prime_divisor_volumes)
                        >= float(min_prime_divisor_volume) - float(volume_tolerance)
                    )
                ),
                "finite_effective_divisor_volumes": bool(
                    np.all(np.isfinite(effective_divisor_volumes))
                ),
                "positive_effective_divisor_volumes": bool(
                    np.all(np.isfinite(effective_divisor_volumes))
                    and (
                        not effective_divisor_volumes.size
                        or np.min(effective_divisor_volumes) > 0.0
                    )
                ),
                "effective_divisor_volume_lower_bound": bool(
                    np.all(np.isfinite(effective_divisor_volumes))
                    and (
                        not effective_divisor_volumes.size
                        or np.min(effective_divisor_volumes)
                        >= float(min_divisor_volume) - float(volume_tolerance)
                    )
                ),
                "finite_inverse_metric": bool(np.all(np.isfinite(inverse_metric))),
                "positive_inverse_metric": bool(
                    np.all(np.isfinite(metric_eigenvalues))
                    and (not metric_eigenvalues.size or np.min(metric_eigenvalues) > 0.0)
                ),
            }
        )
        deferred_checks = {
            "prime_divisor_volume_lower_bound",
            "effective_divisor_volume_lower_bound",
        }
        failed_checks = [
            name
            for name, passed in diagnostic["checks"].items()
            if not passed
            and (
                enforce_divisor_volume_lower_bounds
                or name not in deferred_checks
            )
        ]
        if failed_checks:
            raise ValueError("failed point checks: " + ", ".join(failed_checks))
        diagnostic["point_status"] = "accepted"
        values = {
            "point": point_array,
            "basis_divisor_volumes": basis_divisor_volumes,
            "prime_divisor_volumes": prime_divisor_volumes,
            "effective_divisor_volumes": effective_divisor_volumes,
            "curve_volumes": curve_volumes,
            "inverse_metric": inverse_metric,
            "cy_volume": cy_volume,
        }
    except Exception as exc:
        diagnostic["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if "checks" not in diagnostic:
            diagnostic["checks"] = {}
    return diagnostic, values


def select_canonical_qcd_candidate(
    prime_tau0,
    tau0,
    qprime,
    candidate_indices,
    qcd_volume_target,
    min_prime_divisor_volume,
    min_divisor_volume,
    max_m,
    *,
    allow_m_below_one=False,
    report=None,
):
    """Select a canonical QCD divisor and radial scale.

    With the default ``m >= 1`` policy, order positive finite candidates at
    or below the target by descending tip volume, then by ascending divisor
    index.  This minimizes the required dilation while retaining a
    deterministic fallback when final lower-bound checks reject a candidate.
    When contraction is explicitly enabled, preserve the legacy input order
    while admitting candidates above the target; this keeps the opt-in branch
    conservative and separate from the default policy.
    """
    prime_tau0 = np.asarray(prime_tau0, dtype=float)
    tau0 = np.asarray(tau0, dtype=float)
    qprime = np.asarray(qprime, dtype=float)

    candidates = []
    for candidate_index in candidate_indices:
        candidate_index = int(candidate_index)
        if not 0 <= candidate_index < len(prime_tau0):
            continue
        prime_volume = float(prime_tau0[candidate_index])
        if not np.isfinite(prime_volume) or prime_volume <= 0.0:
            continue
        candidate_m = math.sqrt(qcd_volume_target / prime_volume)
        # The canonical reference run only dilates the tip by default.
        # Contraction is retained as an explicit opt-in for studies that
        # reproduce the unrestricted normalization convention.
        if not allow_m_below_one and candidate_m < 1.0:
            continue
        if not np.isfinite(candidate_m) or candidate_m > max_m:
            continue
        candidates.append((candidate_index, prime_volume, candidate_m))

    if not allow_m_below_one:
        candidates.sort(key=lambda item: (-item[1], item[0]))

    for candidate_index, prime_volume, candidate_m in candidates:
        if report is not None:
            report(
                f"QCD tip volume={prime_volume:.6g}; "
                f"homogeneous radial scale m={candidate_m:.6g}"
            )
        candidate_tau = candidate_m**2 * tau0
        candidate_prime_volumes = candidate_m**2 * prime_tau0
        candidate_effective_volumes = qprime @ candidate_tau
        if (
            not np.all(np.isfinite(candidate_prime_volumes))
            or not np.all(np.isfinite(candidate_effective_volumes))
            or not candidate_prime_volumes.size
            or not candidate_effective_volumes.size
            or np.min(candidate_prime_volumes)
            < min_prime_divisor_volume - DIVISOR_VOLUME_TOLERANCE
        ):
            continue
        if (
            np.min(candidate_effective_volumes)
            < min_divisor_volume - DIVISOR_VOLUME_TOLERANCE
        ):
            continue
        return candidate_index, candidate_m
    return None


def scale_canonical_divisor_volumes(tau0, prime_tau0, qprime, radial_scale):
    """Apply the exact homogeneous divisor-volume scaling for a canonical tip."""
    tau0 = np.asarray(tau0, dtype=float).reshape(-1)
    prime_tau0 = np.asarray(prime_tau0, dtype=float).reshape(-1)
    qprime = np.asarray(qprime, dtype=float)
    radial_scale = float(radial_scale)
    if (
        not np.isfinite(radial_scale)
        or radial_scale <= 0.0
        or tau0.size == 0
        or prime_tau0.size == 0
        or qprime.ndim != 2
        or qprime.shape[1] != tau0.size
        or not np.all(np.isfinite(tau0))
        or not np.all(np.isfinite(prime_tau0))
        or not np.all(np.isfinite(qprime))
    ):
        raise FinalGeometryValidationFailed(
            "canonical divisor-volume scaling references are invalid"
        )
    try:
        volume_scale = radial_scale**2
        scaled_tau = volume_scale * tau0
        scaled_prime_tau = volume_scale * prime_tau0
        scaled_effective_tau = qprime @ scaled_tau
    except (FloatingPointError, ValueError) as exc:
        raise FinalGeometryValidationFailed(
            "canonical divisor-volume scaling overflowed or has invalid data"
        ) from exc
    if not (
        np.all(np.isfinite(scaled_tau))
        and np.all(np.isfinite(scaled_prime_tau))
        and np.all(np.isfinite(scaled_effective_tau))
    ):
        raise FinalGeometryValidationFailed(
            "canonical post-normalization divisor volumes are non-finite"
        )
    return scaled_tau, scaled_prime_tau, scaled_effective_tau


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
    moduli_policy,
    qcd_volume_target,
    qcd_divisor_index,
    visible_sector_policy,
    qed_divisor_index,
    rng,
    report,
    *,
    poly,
    triangulation,
    polytope_id,
    sampling_metadata,
    ks_database_version,
    orientifold_config,
    orientifold_kaehler_policy="none",
    polytope_source=None,
    export_kahler_rays=False,
    allow_overwrite_existing_geometry=False,
    qed_selection_policy="uniform_eligible",
    qed_divisor_index_user=None,
    qed_selection_seed=0,
    qed_volume_max=None,
    materialize_dense_potential=False,
    eft_mode=False,
    raw_frst_metadata=None,
    topology_override=None,
    topology_audit=None,
    kaehler_point_seed=None,
    kaehler_point_diagnostics=None,
    assignment_pool_rejection_records=None,
    allow_m_below_one=False,
    volume_backend=None,
):
    """Compute the CYAxiverse datasets and write one HDF5 geometry file.

    Resolve ``volume_backend`` to the current Fan path by default.  The
    historical sparse COO compatibility path can be selected explicitly, or
    through ``sampling_metadata['volume_backend']`` / ``CYAX_VOLUME_BACKEND``;
    it is accepted only for h11=491.
    """
    volume_backend_requested = volume_backend
    volume_backend = resolve_volume_backend(
        h11, volume_backend, sampling_metadata=sampling_metadata
    )
    # Preserve the package writer's historical zero-based positional option
    # while making the specialist CLI's explicit index one-based and auditable.
    if qed_divisor_index_user is None and qed_divisor_index is not None:
        qed_divisor_index_user = int(qed_divisor_index) + 1
        qed_selection_policy = "explicit"
    if moduli_policy not in {"adaptive", "canonical_qcd"}:
        raise ValueError(
            "moduli_policy must be 'adaptive' or 'canonical_qcd'"
        )
    if allow_m_below_one and moduli_policy != "canonical_qcd":
        raise ValueError(
            "allow_m_below_one requires moduli_policy='canonical_qcd'"
        )
    if materialize_dense_potential:
        raise ValueError(
            "schema 1.1 stores potential reconstruction references only; "
            "dense potential materialization is not permitted in production HDF5"
        )
    if orientifold_kaehler_policy not in {"none", "require_even_subspace"}:
        raise ValueError(
            "orientifold_kaehler_policy must be 'none' or "
            "'require_even_subspace'"
        )
    if qcd_volume_target <= 0.0:
        raise ValueError("qcd_volume_target must be positive")
    if moduli_policy == "canonical_qcd" and not np.isclose(
        qcd_volume_target, QCD_VOLUME_TARGET, rtol=0.0, atol=1e-12
    ):
        raise ValueError("canonical_qcd requires the schema 1.1 QCD target volume 40.0")
    if eft_mode and moduli_policy != "canonical_qcd":
        raise ValueError("--eft requires --moduli-policy canonical_qcd")
    if eft_mode and visible_sector_policy != "intersecting_d7":
        raise ValueError("--eft requires --visible-sector-policy intersecting_d7")
    if eft_mode and not np.isclose(
        float(qed_volume_max if qed_volume_max is not None else np.inf),
        QED_VOLUME_MAX,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("--eft requires the inclusive QED volume bound 127.5")
    if qcd_divisor_index is not None and qcd_divisor_index < 0:
        raise ValueError("qcd_divisor_index must be non-negative")
    if qcd_divisor_index is not None and moduli_policy != "canonical_qcd":
        raise ValueError(
            "qcd_divisor_index requires moduli_policy='canonical_qcd'"
        )
    if visible_sector_policy not in {"none", "intersecting_d7"}:
        raise ValueError(
            "visible_sector_policy must be 'none' or 'intersecting_d7'"
        )
    if qed_selection_policy not in {"uniform_eligible", "explicit"}:
        raise ValueError(
            "qed_selection_policy must be 'uniform_eligible' or 'explicit'"
        )
    if visible_sector_policy == "none" and (
        qed_selection_policy == "explicit" or qed_divisor_index_user is not None
    ):
        raise QEDAssignmentFailure(
            "invalid_explicit_index",
            "QED selection requires visible_sector_policy='intersecting_d7'",
        )
    if visible_sector_policy == "intersecting_d7" and not orientifold_config.get(
        "requested", False
    ):
        raise NoVisibleSectorAssignment(
            "intersecting_d7 requires --orientifold-file"
        )
    if qed_divisor_index_user is not None and visible_sector_policy != "intersecting_d7":
        raise ValueError(
            "qed_divisor_index_user requires visible_sector_policy='intersecting_d7'"
        )
    if qed_selection_policy == "explicit" and qed_divisor_index_user is None:
        raise ValueError("explicit QED selection requires qed_divisor_index_user")
    if qed_selection_policy == "uniform_eligible" and qed_divisor_index_user is not None:
        raise ValueError("an explicit QED index requires explicit selection")
    if kaehler_point_seed is None:
        kaehler_point_seed = stable_seed(
            "kaehler-point",
            sampling_metadata.get("seed", 0),
            sampling_metadata.get("proposal_seed"),
            polytope_id,
        )
    point_diagnostics = (
        [] if kaehler_point_diagnostics is None else kaehler_point_diagnostics
    )
    report("validating the CYTools FRST")
    frst_validation = validate_frst(poly, triangulation)
    if topology_audit is not None:
        topology_audit["frst_validation"] = frst_validation
        topology_audit["smooth_hypersurface"] = bool(cy.is_smooth())
    if not bool(cy.is_smooth()):
        raise RuntimeError("CYTools reports that the generic CY hypersurface is not smooth.")
    if topology_override is None:
        report("computing Hodge, intersection, and divisor-basis data")
        topology = extract_topology(
            cy, triangulation, export_kahler_rays=export_kahler_rays
        )
    else:
        report("loading validated topology cache")
        topology = dict(topology_override)
        topology.setdefault("kahler_cone_rays", None)
        missing = [
            name
            for name in (
                "h11",
                "h21",
                "basis",
                "basis_matrix",
                "prime_toric_divisors",
                "kappa",
                "c2",
                "mori_cone",
                "kahler_cone_hyperplanes",
                "face_restriction_dim2",
            )
            if name not in topology
        ]
        if missing:
            raise RuntimeError(f"validated topology cache is missing fields: {missing}")
    if topology_audit is not None:
        topology_audit.update(
            {
                "cytools_h11": topology["h11"],
                "cytools_h21": topology["h21"],
                "basis_convention": (
                    "CYTools divisor_basis(include_origin=True); "
                    "all numerical vectors in basis"
                ),
                "intersection_convention": (
                    "CYTools CalabiYau.intersection_numbers "
                    "(in_basis=True, format='coo')"
                ),
                "topology_arrays": {
                    "basis": summarize_array_structure(topology["basis"]),
                    "basis_matrix": summarize_array_structure(
                        topology["basis_matrix"]
                    ),
                    "prime_toric_divisors": summarize_array_structure(
                        topology["prime_toric_divisors"]
                    ),
                    "intersection_numbers": summarize_array_structure(
                        topology["kappa"]
                    ),
                    "second_chern_class": summarize_array_structure(topology["c2"]),
                    "mori_cone": summarize_array_structure(topology["mori_cone"]),
                    "kahler_cone_hyperplanes": summarize_array_structure(
                        topology["kahler_cone_hyperplanes"]
                    ),
                    "kahler_cone_rays": (
                        None
                        if topology["kahler_cone_rays"] is None
                        else summarize_array_structure(topology["kahler_cone_rays"])
                    ),
                    "face_restriction_dim2": {
                        "count": len(topology["face_restriction_dim2"]),
                        "dtype": str(
                            np.asarray(topology["face_restriction_dim2"]).dtype
                        ),
                    },
                },
                "topology_validation_status": "passed",
            }
        )
    orientifold = validate_orientifold(poly, triangulation, topology, orientifold_config)
    if topology_audit is not None:
        topology_audit["orientifold_validation"] = {
            "requested": bool(orientifold.get("requested", False)),
            "input_status": orientifold.get("status"),
            "status": orientifold.get("status"),
            "involution_type": orientifold.get("involution_type"),
            "h11_plus": orientifold.get("h11_plus"),
            "h11_minus": orientifold.get("h11_minus"),
            "h11_parity_policy": "record_only_not_enforced",
            "fixed_locus_validation": "not_performed",
            "tadpole_validation": "not_performed",
            "physical_orientifold_claim": "not_made",
        }
    prime_labels = np.asarray(topology["prime_toric_divisors"], dtype=int)
    prime_labels_stable = stable_divisor_labels(prime_labels, poly_points)
    prime_charges = None
    prime_intersection_evidence = {}
    prime_neighbors = None
    if visible_sector_policy != "none":
        try:
            prime_charges = prime_divisor_charges(
                topology["basis_matrix"], prime_labels
            )
            prime_neighbors, prime_intersection_evidence = (
                prime_divisor_intersection_graph(
                    prime_labels, topology["face_restriction_dim2"]
                )
            )
        except QEDAssignmentFailure:
            raise
        except Exception as exc:
            raise QEDAssignmentFailure(
                "invalid_charge_basis_mapping", str(exc)
            ) from exc
    neighbors = None
    standard_model_divisors = None
    standard_model_qcd_selection = None
    visible_qcd_candidates = None
    visible_qcd_candidate_set = None
    canonical_qcd_qed_prefilter = _inactive_canonical_qcd_qed_prefilter_metadata(
        eft_mode=eft_mode,
        moduli_policy=moduli_policy,
        visible_sector_policy=visible_sector_policy,
    )
    if moduli_policy == "canonical_qcd" or visible_sector_policy == "intersecting_d7":
        neighbors = prime_divisor_neighbors(
            topology["prime_toric_divisors"], topology["face_restriction_dim2"]
        )
    h21 = topology["h21"]
    if topology["h11"] != int(h11) or topology["h11"] != int(cy.h11()):
        raise RuntimeError(
            f"h11 mismatch between request ({h11}) and CYTools ({topology['h11']})."
    )
    if topology_audit is not None:
        topology_audit.setdefault("volume_backend", volume_backend)
        topology_audit["volume_backend_selected"] = volume_backend
        topology_audit["historical_contraction"] = (
            "sparse COO multiplicity-aware contraction from "
            "CalabiYau.intersection_numbers(in_basis=True, format='coo')"
            if volume_backend == HISTORICAL_VOLUME_BACKEND
            else "not_selected"
        )
    triangulation_id, cy3_fingerprint = topology_identity(
        polytope_id, triangulation, topology
    )
    favorable = bool(poly.is_favorable(lattice="N"))
    glsm = np.asarray(cy.glsm_charge_matrix(include_origin=False), dtype=int)
    basis = topology["basis"]
    volume_context = {
        "volume_backend": volume_backend,
        "kappa": topology["kappa"],
        "glsm_charge_matrix": glsm,
        "mori_cone": topology["mori_cone"],
    }
    if moduli_policy == "canonical_qcd":
        # The complete ordered assignment pool is now the visible-sector
        # acceptance unit.  Keep an explicit QCD index as a deterministic
        # geometry reference when supplied; otherwise use the deterministic
        # minimal-dilation candidate order after the immutable tip data are
        # available below.  There is no detached random-QCD record in the
        # schema-1.1 flow.
        standard_model_divisors = None
        standard_model_qcd_selection = (
            "explicit_geometry_reference_qcd"
            if qcd_divisor_index is not None
            else "deterministic_geometry_reference_qcd"
        )

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
    if orientifold_kaehler_policy == "require_even_subspace":
        orientifold = validate_invariant_kaehler_subspace(
            kahler_cone, reference_tip, orientifold
        )
    else:
        orientifold = dict(orientifold)
        orientifold["kaehler_subspace_validation_status"] = (
            "not_required_for_declared_reference_run"
        )
        orientifold["invariant_kahler_cone_intersection"] = None
        orientifold["invariant_kahler_point"] = None
    if topology_audit is not None and orientifold["requested"]:
        topology_audit["orientifold_validation"].update(
            {
                "status": orientifold.get("status"),
                "kaehler_subspace_policy": orientifold_kaehler_policy,
                "kaehler_subspace_validation_status": orientifold.get(
                    "kaehler_subspace_validation_status", "validated"
                ),
                "invariant_kahler_cone_intersection": orientifold.get(
                    "invariant_kahler_cone_intersection"
                ),
            }
        )
    if visible_sector_policy == "intersecting_d7":
        visible_qcd_candidates = _visible_qcd_candidates(
            visible_sector_policy, orientifold, neighbors
        )
        visible_qcd_candidate_set = set(visible_qcd_candidates)
        if (
            qcd_divisor_index is not None
            and qcd_divisor_index not in visible_qcd_candidate_set
            and not canonical_qcd_qed_prefilter_active(
                eft_mode=eft_mode,
                moduli_policy=moduli_policy,
                visible_sector_policy=visible_sector_policy,
            )
        ):
            raise NoVisibleSectorAssignment(
                f"QCD divisor index {qcd_divisor_index} has no invariant "
                "intersecting QED divisor"
            )
    report("computing effective-cone rays")
    qprime_raw = np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    if qprime_raw.ndim != 2 or qprime_raw.shape[1] != int(h11):
        raise RuntimeError(
            "Effective-cone rays are not expressed in the exported divisor basis: "
            f"got shape {qprime_raw.shape}, expected (*, {h11})."
        )
    qprime, charge_metadata = canonicalize_unique_charge_rows(qprime_raw)
    qprime = np.asarray(qprime, dtype=np.int64)
    nq = qprime.shape[0]
    if charge_metadata["duplicates_removed"]:
        report(
            "removed "
            f"{charge_metadata['duplicates_removed']} redundant effective-cone "
            "charge rows before potential construction"
        )

    # Eq. (20) of arXiv:2309.01831 concerns effective four-cycles on the CY.
    # Do *not* require every ambient toric divisor returned by CYTools to be
    # positive: at large h11 that list can include divisors with trivial or
    # redundant restriction to the hypersurface.  The toric effective-cone
    # rays are the relevant generators, and qprime and tau_basis share the
    # divisor-basis convention below.
    selected_point_diagnostic = None
    if moduli_policy == "canonical_qcd":
        # Use the canonical stretched-cone ray and impose the visible-sector
        # normalization by a later homogeneous rescaling.  This is the
        # geometry-level QCD normalization; it does not select a new FRST.
        kaehler_point = reference_tip.copy()
        divisor_scale = 1.0
        projection_solvers = []
        selected_point_diagnostic, selected_point_values = evaluate_kaehler_point(
            cy,
            kahler_cone,
            qprime,
            kaehler_point,
            attempt_index=1,
            point_kind="canonical_tip",
            point_seed=kaehler_point_seed,
            solver=tip_solver,
            min_prime_divisor_volume=min_prime_divisor_volume,
            min_divisor_volume=min_divisor_volume,
            volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
            # Apply the final >=1 control cut only after QCD normalization.
            enforce_divisor_volume_lower_bounds=False,
            **volume_context,
        )
        point_diagnostics.append(selected_point_diagnostic)
        if selected_point_values is None:
            raise NoPhysicalKaehlerPoint(
                "The canonical stretched-cone tip failed the Kähler-point domain checks."
            )
        report("using the canonical stretched-cone ray for QCD normalization")
    else:
        report("searching angular Kähler directions with positive effective-divisor volumes")
        kaehler_point = None
        divisor_scale = None
        projection_solvers = []
        for proposal in sample_stretched_kaehler_points(
            kahler_cone,
            reference_tip,
            rng,
            max_kaehler_attempts,
            report,
            projection_solvers,
            point_seed=kaehler_point_seed,
            diagnostics=point_diagnostics,
            include_metadata=True,
        ):
            kaehler_attempt = int(proposal["attempt_index"])
            candidate = np.asarray(proposal["point"], dtype=float)
            candidate_diagnostic, candidate_values = evaluate_kaehler_point(
                cy,
                kahler_cone,
                qprime,
                candidate,
                attempt_index=kaehler_attempt,
                point_kind=proposal["point_kind"],
                point_seed=proposal.get("point_seed"),
                solver=proposal.get("solver"),
                min_prime_divisor_volume=min_prime_divisor_volume,
                min_divisor_volume=min_divisor_volume,
                volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
                **volume_context,
            )
            point_diagnostics.append(candidate_diagnostic)
            if candidate_values is None:
                report(
                    f"rejected Kähler point {kaehler_attempt}/{max_kaehler_attempts}: "
                    f"{candidate_diagnostic.get('failure_reason', 'domain checks failed')}"
                )
                continue
            effective_volumes = candidate_values["effective_divisor_volumes"]
            minimum_volume = float(np.min(effective_volumes))
            divisor_scale = max(1.0, math.sqrt(min_divisor_volume / minimum_volume))
            kaehler_point = divisor_scale * candidate
            selected_point_diagnostic = candidate_diagnostic
            selected_point_values = candidate_values
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
    reference_geometry = _compute_volume_geometry(
        cy, kaehler_point, **volume_context
    )
    tau0 = reference_geometry["basis_divisor_volumes"]
    if tau0.shape != (int(h11),) or not np.all(np.isfinite(tau0)):
        raise RuntimeError(
            f"CYTools returned invalid basis divisor volumes with shape {tau0.shape}."
        )
    kinv0 = reference_geometry["inverse_metric"]
    if kinv0.shape != (int(h11), int(h11)) or not np.all(np.isfinite(kinv0)):
        raise RuntimeError("CYTools returned an invalid inverse Kähler metric.")
    if np.min(np.linalg.eigvalsh(kinv0)) <= 0.0:
        raise RuntimeError("The selected Kähler point has a non-positive metric.")
    tau, kinv = tau0.copy(), kinv0.copy()
    prime_tau0 = reference_geometry["prime_divisor_volumes"]
    if prime_tau0.ndim != 1 or not np.all(np.isfinite(prime_tau0)):
        raise RuntimeError("CYTools returned invalid prime toric divisor volumes.")

    tauq0 = qprime @ tau0
    if np.min(tauq0) <= 0.0:
        raise NoPhysicalKaehlerPoint(
            "The positive prime-divisor point has non-positive effective-ray "
            "volumes in the selected divisor basis."
        )
    pre_normalization_tip = np.asarray(kaehler_point, dtype=float).copy()
    pre_normalization_geometry = _compute_volume_geometry(
        cy, pre_normalization_tip, **volume_context
    )
    pre_normalization_volume = pre_normalization_geometry["cy_volume"]
    pre_normalization_curve_volumes = pre_normalization_geometry["curve_volumes"]
    if (
        not np.isfinite(pre_normalization_volume)
        or not np.all(np.isfinite(pre_normalization_curve_volumes))
    ):
        raise RuntimeError("pre-normalization geometry data are non-finite")
    volume_backend_diagnostics = _volume_backend_diagnostics(
        cy,
        kaehler_point,
        reference_geometry,
        effective_cone_rays=qprime,
        **volume_context,
    )
    if moduli_policy == "canonical_qcd":
        # This is the paper-style geometry prescription: keep the canonical
        # stretched-cone direction and fix only the radial scale from the
        # selected QCD divisor.  The adaptive potential-control search is not
        # part of this normalization and would add avoidable O(nq^2) work.
        production_qed_prefilter = canonical_qcd_qed_prefilter_active(
            eft_mode=eft_mode,
            moduli_policy=moduli_policy,
            visible_sector_policy=visible_sector_policy,
        )
        if production_qed_prefilter:
            # Evaluate every QCD candidate before minimal-dilation ordering so
            # the QED volume test is candidate-specific.  An explicit index
            # remains a singleton override and is never replaced by another
            # candidate.
            candidate_indices = (
                [qcd_divisor_index]
                if qcd_divisor_index is not None
                else list(range(len(prime_tau0)))
            )
            invariant_mask = np.asarray(
                orientifold["prime_divisor_image_indices"], dtype=int
            ) == np.arange(prime_labels.size)
            canonical_qcd_qed_prefilter = prefilter_canonical_qcd_candidates(
                prime_tau0,
                candidate_indices,
                prime_neighbors,
                invariant_mask,
                qcd_volume_target,
                QED_VOLUME_MAX if qed_volume_max is None else float(qed_volume_max),
                max_m,
                allow_m_below_one=allow_m_below_one,
            )
            canonical_qcd_qed_prefilter["explicit_qcd_index_override"] = (
                qcd_divisor_index is not None
            )
            if topology_audit is not None:
                topology_audit["canonical_qcd_qed_prefilter"] = (
                    canonical_qcd_qed_prefilter
                )
            candidate_indices = canonical_qcd_qed_prefilter[
                "eligible_candidate_indices"
            ]
            if not candidate_indices:
                raise QEDAssignmentFailure(
                    CANONICAL_QCD_QED_PREFILTER_FAILURE_STATUS,
                    "The canonical QCD QED prefilter rejected every candidate: "
                    "no distinct intersecting orientifold-invariant QED neighbor "
                    f"has final volume <= {canonical_qcd_qed_prefilter['effective_qed_volume_max']:g}.",
                    canonical_qcd_qed_prefilter,
                )
        else:
            candidate_indices = (
                [qcd_divisor_index]
                if qcd_divisor_index is not None
                else sorted(visible_qcd_candidate_set)
                if visible_qcd_candidate_set is not None
                else list(range(len(prime_tau0)))
            )
            if visible_qcd_candidate_set is not None:
                candidate_indices = [
                    index
                    for index in candidate_indices
                    if index in visible_qcd_candidate_set
                ]
        selected_qcd = select_canonical_qcd_candidate(
            prime_tau0,
            tau0,
            qprime,
            candidate_indices,
            qcd_volume_target,
            min_prime_divisor_volume,
            min_divisor_volume,
            max_m,
            allow_m_below_one=allow_m_below_one,
            report=report,
        )
        if selected_qcd is None:
            requested = (
                f"prime toric divisor index {qcd_divisor_index}"
                if qcd_divisor_index is not None
                else "any prime toric divisor"
            )
            raise NoQcdDivisorVolume(
                f"No canonical stretched-cone ray satisfies QCD volume "
                f"{qcd_volume_target:g} for {requested}, the divisor lower "
                f"bound {min_prime_divisor_volume:g}, the effective-divisor "
                "lower bound, and final Kähler-cone validation."
            )
        qcd_divisor_index, m_val = selected_qcd
        if production_qed_prefilter:
            canonical_qcd_qed_prefilter.update(
                {
                    "selected_qcd_index": int(qcd_divisor_index),
                    "selected_radial_scale": float(m_val),
                    "selection_status": "selected_after_qed_prefilter",
                }
            )
            if topology_audit is not None:
                topology_audit["canonical_qcd_qed_prefilter"] = (
                    canonical_qcd_qed_prefilter
                )
        qcd_volume_min = qcd_volume_target
        qcd_volume_max = qcd_volume_target
    else:
        # The adaptive policy retains the existing potential-control search.
        # Its pairwise form is evaluated on m=1 data and scaled analytically
        # because the bilinear and ray-volume terms scale as m**4 and m**2.
        report(f"searching the stretched-cone prefactor over {nq * (nq - 1) // 2} ray pairs")
        lower_i, lower_j = np.tril_indices(nq, k=-1)
        bilinear0 = (qprime @ kinv0) @ qprime.T

        def prefactor_is_valid(candidate_m):
            """Evaluate the adaptive pairwise criterion at one m."""
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

        # Doubling reaches the large-m regime in O(log(m)) checks.  Once it
        # finds a valid upper bound, binary refinement recovers the old 0.01
        # resolution.
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
        for candidate_index, prime_volume in enumerate(prime_tau0):
            if visible_qcd_candidate_set is not None and candidate_index not in visible_qcd_candidate_set:
                continue
            lower = math.sqrt(qcd_volume_min / float(prime_volume))
            upper = math.sqrt(qcd_volume_max / float(prime_volume))
            lower = max(lower, m_val, minimum_m_for_prime_divisors)
            if lower <= upper and lower <= max_m and prefactor_is_valid(lower):
                qcd_interval = (candidate_index, lower, upper)
                break
        if qcd_interval is None:
            raise NoQcdDivisorVolume(
                "No stretched-cone prefactor satisfies the prime-toric-divisor "
                f"lower bound {min_prime_divisor_volume:g} and QCD window "
                f"[{qcd_volume_min:g}, {qcd_volume_max:g}]."
            )
        qcd_divisor_index, m_val, _ = qcd_interval
    m2 = m_val**2
    kinv = m2**2 * kinv0

    # Store a self-consistent physical point: tau, Kinv and the CY volume are
    # all evaluated at the same final J = m * kaehler_point.  For the
    # canonical policy, use the exact homogeneous divisor scaling from the
    # selected reference point rather than a second floating-point CYTools
    # evaluation at the dilated point.  The latter can drift beyond the strict
    # QCD target tolerance even though the homogeneous normalization is exact.
    tip = m_val * kaehler_point
    final_geometry = _compute_volume_geometry(cy, tip, **volume_context)
    volume = final_geometry["cy_volume"]
    if moduli_policy == "canonical_qcd":
        tau, prime_divisor_volumes, effective_divisor_volumes = (
            scale_canonical_divisor_volumes(
                tau0, prime_tau0, qprime, m_val
            )
        )
    else:
        tau = final_geometry["basis_divisor_volumes"]
        prime_divisor_volumes = final_geometry["prime_divisor_volumes"]
        effective_divisor_volumes = qprime @ tau
    if tau.shape != (int(h11),) or not np.all(np.isfinite(tau)):
        raise FinalGeometryValidationFailed(
            "post-normalization basis divisor volumes are non-finite or have an invalid shape"
        )
    if prime_divisor_volumes.ndim != 1 or not np.all(
        np.isfinite(prime_divisor_volumes)
    ):
        raise FinalGeometryValidationFailed(
            "post-normalization prime toric divisor volumes are non-finite"
        )
    if effective_divisor_volumes.ndim != 1 or not np.all(
        np.isfinite(effective_divisor_volumes)
    ):
        raise FinalGeometryValidationFailed(
            "post-normalization effective-cone divisor volumes are non-finite"
        )
    selected_normalization = None
    if moduli_policy == "canonical_qcd":
        try:
            selected_normalization = normalize_qcd_assignment(
                prime_tau0,
                tauq0,
                qcd_divisor_index,
                target=QCD_VOLUME_TARGET,
                min_prime=min_prime_divisor_volume,
                min_effective=min_divisor_volume,
                qcd_volume_tolerance=QCD_VOLUME_TOLERANCE,
                divisor_volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
            )
        except ValueError as exc:
            raise NoQcdDivisorVolume(str(exc)) from exc
    divisor_volume_evidence = build_divisor_volume_evidence(
        np.arange(prime_labels.size, dtype=np.int64),
        prime_labels_stable,
        prime_divisor_volumes,
        qprime,
        effective_divisor_volumes,
        topology["basis"],
        min_prime_divisor_volume,
        min_divisor_volume,
    )
    curve_volumes = final_geometry["curve_volumes"]
    kaehler_slack = np.asarray(kahler_cone.hyperplanes(), dtype=float) @ tip
    normalization_checks = validate_final_qcd_normalization(
        point=tip,
        radial_scale=m_val,
        max_m=max_m,
        allow_m_below_one=allow_m_below_one,
        qcd_divisor_index=qcd_divisor_index,
        qcd_volume_target=(QCD_VOLUME_TARGET if moduli_policy == "canonical_qcd" else None),
        qcd_volume_min=qcd_volume_min,
        qcd_volume_max=qcd_volume_max,
        cy_volume=volume,
        curve_volumes=curve_volumes,
        kaehler_slack=kaehler_slack,
        inverse_metric=kinv,
        prime_divisor_volumes=prime_divisor_volumes,
        effective_divisor_volumes=effective_divisor_volumes,
        min_prime_divisor_volume=min_prime_divisor_volume,
        min_divisor_volume=min_divisor_volume,
    )
    if selected_point_diagnostic is not None:
        selected_point_diagnostic.update(
            {
                "selected_for_normalization": True,
                "normalization_status": "passed",
                "radial_scale": float(m_val),
                "angular_scale": float(divisor_scale),
            }
        )
    tip_prefactor = np.asarray([divisor_scale, m_val], dtype=float)

    assignment_pool = None
    assignment_pool_validation = None
    assignment_pool_rejection_summary = None
    if eft_mode and moduli_policy == "canonical_qcd" and visible_sector_policy == "intersecting_d7":
        invariant_mask = np.asarray(
            orientifold["prime_divisor_image_indices"], dtype=int
        ) == np.arange(prime_labels.size)
        assignment_pool = enumerate_assignment_pool(
            prime_labels=prime_labels_stable,
            prime_charges=prime_charges,
            prime_volumes_reference=prime_tau0,
            # The assignment helper requires one effective-volume entry per
            # prime divisor.  The immutable CYTools prime-divisor volumes are
            # the assignment-level divisor family; the separate effective-cone
            # ray volumes remain persisted and validated above.
            effective_volumes_reference=prime_tau0,
            neighbors=prime_neighbors,
            intersection_evidence=prime_intersection_evidence,
            invariant_mask=invariant_mask,
            qcd_volume_target=QCD_VOLUME_TARGET,
            min_prime_volume=min_prime_divisor_volume,
            min_effective_volume=min_divisor_volume,
            qcd_volume_tolerance=QCD_VOLUME_TOLERANCE,
            divisor_volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
            qed_volume_max=(
                QED_VOLUME_MAX if qed_volume_max is None else float(qed_volume_max)
            ),
        )
        assignment_pool_rejection_summary = summarize_assignment_pool_rejections(
            getattr(assignment_pool, "terminal_records", ())
        )
        if assignment_pool_rejection_records is not None:
            context = {
                "stage": "stage2",
                "geometry_id": (
                    None
                    if raw_frst_metadata is None
                    else raw_frst_metadata.get("geometry_id")
                ),
                "h11": None
                if raw_frst_metadata is None
                else raw_frst_metadata.get("h11"),
                "polytope_id": (
                    None
                    if raw_frst_metadata is None
                    else raw_frst_metadata.get("polytope_id")
                ),
                "raw_frst_path": (
                    None
                    if raw_frst_metadata is None
                    else raw_frst_metadata.get("raw_frst_path")
                ),
            }
            assignment_pool_rejection_records.extend(
                [
                    {**context, **dict(record)}
                    for record in getattr(assignment_pool, "terminal_records", ())
                    if record.get("terminal_status") != "accepted_assignment"
                ]
            )
        if not assignment_pool:
            raise QEDAssignmentFailure(
                "assignment_pool_shortfall",
                "the complete ordered QCD-QED assignment pool is empty",
                {
                    "assignment_pool_status": "empty",
                    "assignment_pool_rejection_summary": assignment_pool_rejection_summary,
                },
            )
        try:
            assignment_pool_validation = validate_assignment_pool(assignment_pool)
        except QEDAssignmentFailure as error:
            error.record.update(
                {
                    "assignment_pool_status": "incomplete",
                    "assignment_pool_rejection_summary": assignment_pool_rejection_summary,
                }
            )
            raise

    visible_sector = None
    if visible_sector_policy != "none" and not eft_mode:
        visible_sector = select_qed_divisor(
            policy=visible_sector_policy,
            selection_policy=qed_selection_policy,
            qcd_divisor_index=qcd_divisor_index,
            prime_toric_divisors=prime_labels,
            prime_divisor_labels=prime_labels_stable,
            prime_divisor_charges_array=prime_charges,
            prime_divisor_volumes=prime_divisor_volumes,
            neighbors=prime_neighbors,
            intersection_evidence=prime_intersection_evidence,
            orientifold=orientifold,
            effective_seed=qed_selection_seed,
            qed_divisor_index_user=qed_divisor_index_user,
            qed_volume_max=qed_volume_max,
        )

    report(f"recording potential reconstruction references for {nq} effective-cone rays")
    num_cross = nq * (nq - 1) // 2
    q_direct = np.asarray(qprime.T, dtype=np.int64)
    factorized_charges = factorized_charge_metadata(q_direct)
    qed_charge = None if visible_sector is None else visible_sector["qed_charge"]
    qed_direct_index = None
    qed_potential_source_index = None
    if qed_charge is not None:
        for direct_index, direct_charge in enumerate(qprime):
            if np.array_equal(direct_charge, qed_charge):
                qed_direct_index = direct_index
                break
        qed_potential_source_index = (
            qed_direct_index if qed_direct_index is not None else nq + num_cross
        )
        visible_sector["qed_instanton_index"] = int(qed_potential_source_index)
    if visible_sector is not None:
        visible_sector.update(
            {
                "qed_potential_source": (
                    "direct_effective_cone"
                    if qed_direct_index is not None
                    else "appended_prime_divisor_e3"
                ),
                "qed_charge_exact_match": True,
                "qed_potential_source_index": int(qed_potential_source_index),
                "qed_leading_status": "deferred_to_eft_row_reconstruction",
                "leading_rank_certificate": {
                    "status": "deferred_to_eft_row_reconstruction",
                    "method": "compact_geometry_reference_reconstruction",
                },
            }
        )
    if visible_sector is not None:
        visible_sector["terminal_status"] = "accepted_assignment"
        visible_sector["terminal_reason"] = "geometry-derived QED assignment accepted"
    basis_matrix = np.asarray(topology["basis_matrix"], dtype=int)
    prime_labels = np.asarray(topology["prime_toric_divisors"], dtype=int)
    if basis_matrix.ndim != 2 or basis_matrix.shape[1] <= int(np.max(prime_labels)):
        raise RuntimeError("the divisor basis matrix cannot represent prime divisors")
    divisor_volume_evidence_compact = {
        key: value
        for key, value in divisor_volume_evidence.items()
        if key
        not in {
            "prime_divisor_volumes",
            "effective_divisor_volumes",
        }
    }
    divisor_volume_evidence_compact.update(
        {
            "prime_divisor_volume_count": int(prime_divisor_volumes.size),
            "effective_divisor_volume_count": int(effective_divisor_volumes.size),
            "prime_divisor_volumes_sha256": stable_hash(prime_divisor_volumes.tolist()),
            "effective_divisor_volumes_sha256": stable_hash(
                effective_divisor_volumes.tolist()
            ),
        }
    )

    artifact_status = geometry_artifact_status(
        eft_mode,
        None if assignment_pool_validation is None else assignment_pool_validation[
            "pool_status"
        ],
    )
    if eft_mode and artifact_status != ACCEPTED_GEOMETRY_ARTIFACT_STATUS:
        raise QEDAssignmentFailure(
            "assignment_pool_shortfall",
            "EFT geometry finalization requires a complete validated assignment pool",
            {"artifact_status": artifact_status},
        )
    report("writing HDF5 data")
    temporary_path, artifact_write_audit = prepare_geometry_artifact_write(
        filepath, allow_overwrite_existing_geometry
    )
    kaehler_point_status_counts = {}
    for point_record in point_diagnostics:
        status = point_record.get("point_status", "unknown")
        kaehler_point_status_counts[status] = (
            kaehler_point_status_counts.get(status, 0) + 1
        )
    kaehler_point_attempted_count = sum(
        bool(point_record.get("attempted", False))
        for point_record in point_diagnostics
    )
    construction_metadata = {
        "schema_version": SCHEMA_VERSION,
        "schema_semantic_version": SCHEMA_1_1_VERSION,
        "charge_factorized_schema_version": CHARGE_FACTORIZED_SCHEMA_VERSION,
        "normalization_map_version": NORMALIZATION_MAP_VERSION,
        "source_references": SOURCE_REFERENCES,
        "cytools_version": cytools.version,
        "ks_database_version": ks_database_version,
        "polytope_id": polytope_id,
        "polytope_id_kind": "canonical_lattice_point_sha256",
        "polytope_source": polytope_source,
        "triangulation_id": triangulation_id,
        "cy3_fingerprint": cy3_fingerprint,
        "cy3_fingerprint_status": "topological_fingerprint",
        "sampling": sampling_metadata,
        "volume_backend_requested": (
            "fan" if volume_backend_requested is None else str(volume_backend_requested)
        ),
        "volume_backend": volume_backend,
        "volume_backend_selection": (
            "h11=491 historical_sparse_coo; all other h11 fan"
            if volume_backend_requested == AUTO_VOLUME_BACKEND
            else "explicit_or_default_backend"
        ),
        "volume_backend_scope": (
            "h11=491 compatibility reproduction only"
            if volume_backend == HISTORICAL_VOLUME_BACKEND
            else "CYTools Fan path default"
        ),
        "historical_contraction": (
            {
                "selected": True,
                "intersection_source": (
                    "CYTools CalabiYau.intersection_numbers "
                    "(in_basis=True, format='coo')"
                ),
                "formula": (
                    "V=1/6*kappa_ijk*t_i*t_j*t_k; "
                    "tau_i=1/2*kappa_ijk*t_j*t_k; "
                    "repeated-index multiplicities from COO entries"
                ),
                "prime_divisor_formula": "glsm_charge_matrix.T @ tau_basis",
                "effective_divisor_formula": "effective_cone_rays @ tau_basis",
                "curve_volume_formula": "mori_cone_rays @ t",
            }
            if volume_backend == HISTORICAL_VOLUME_BACKEND
            else {"selected": False},
        ),
        "volume_backend_diagnostics": volume_backend_diagnostics,
        "kaehler_point_scan": {
            "policy": moduli_policy,
            "attempt_budget": int(max_kaehler_attempts),
            "attempt_budget_semantics": (
                "one canonical tip evaluation"
                if moduli_policy == "canonical_qcd"
                else "at most this many evaluations including the canonical tip"
            ),
            "point_seed": int(kaehler_point_seed),
            "canonical_tip_included": True,
            "selected_attempt_index": (
                None
                if selected_point_diagnostic is None
                else int(selected_point_diagnostic["attempt_index"])
            ),
            "status_counts": kaehler_point_status_counts,
            "attempted_point_count": kaehler_point_attempted_count,
            "diagnostics": point_diagnostics,
        },
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
            (
                (
                    "historical sparse COO basis contraction followed by "
                    "GLSM.T @ tau_basis at reference tip, then homogeneous m^2 "
                    "scaling; ordered by CYTools prime_toric_divisors()"
                )
                if volume_backend == HISTORICAL_VOLUME_BACKEND
                else (
                    "CYTools compute_divisor_volumes(reference tip), then "
                    "homogeneous m^2 scaling; ordered by CYTools "
                    "prime_toric_divisors()"
                )
            )
            if moduli_policy == "canonical_qcd"
            else (
                (
                    "historical sparse COO basis contraction followed by "
                    "GLSM.T @ tau_basis, ordered by CYTools prime_toric_divisors()"
                )
                if volume_backend == HISTORICAL_VOLUME_BACKEND
                else (
                    "CYTools compute_divisor_volumes(tip), ordered by "
                    "CYTools prime_toric_divisors()"
                )
            )
        ),
        "qcd_divisor_volume_window": [qcd_volume_min, qcd_volume_max],
        "qcd_divisor_index": qcd_divisor_index,
        "qcd_divisor_index_base": 0,
        "qcd_divisor_label": np.asarray(
            prime_labels_stable[qcd_divisor_index]
        ).tolist(),
        "qcd_divisor_volume": float(prime_divisor_volumes[qcd_divisor_index]),
        "qcd_divisor_volume_exact": bool(
            moduli_policy == "canonical_qcd"
            and np.isclose(
                float(prime_divisor_volumes[qcd_divisor_index]),
                QCD_VOLUME_TARGET,
                rtol=0.0,
                atol=QCD_VOLUME_TOLERANCE,
            )
        ),
        "post_normalization_min_prime_divisor_volume": float(
            np.min(prime_divisor_volumes)
        ),
        "post_normalization_min_effective_divisor_volume": float(
            np.min(effective_divisor_volumes)
        ),
        "qcd_volume_tolerance": QCD_VOLUME_TOLERANCE,
        "divisor_volume_tolerance": DIVISOR_VOLUME_TOLERANCE,
        "divisor_volume_evidence": divisor_volume_evidence_compact,
        "qcd_normalization_validation": normalization_checks,
        "qcd_normalization_failure_status": "qcd_normalization_failure",
        "qcd_normalization_repair_policy": "none",
        "moduli_policy": moduli_policy,
        "standard_model": (
            None
            if standard_model_divisors is None
            else {
                "selection_policy": "uniform_pairwise_intersection_triangle",
                "divisor_indices": list(standard_model_divisors),
                "qcd_selection": standard_model_qcd_selection,
                "qcd_divisor_index": int(qcd_divisor_index),
                "divisor_index_base": 0,
            }
        ),
        "homogeneous_scaling": (
            "J_final = m J_tip; m = sqrt(qcd_volume_target / "
            "vol(D_QCD)_tip); tau scales as m^2, Kinv as m^4"
            if moduli_policy == "canonical_qcd"
            else "adaptive prefactor search"
        ),
        "potential_control": (
            "adaptive_pairwise_search"
            if moduli_policy == "adaptive"
            else "not_applied_in_canonical_qcd"
        ),
        "qcd_volume_target": qcd_volume_target,
        "allow_m_below_one": bool(allow_m_below_one),
        "canonical_qcd_normalization": (
            {
                "candidate_order": (
                    "explicit_qcd_divisor_index"
                    if qcd_divisor_index is not None
                    else (
                        CANONICAL_QCD_CONTRACTION_CANDIDATE_ORDER
                        if allow_m_below_one
                        else CANONICAL_QCD_CANDIDATE_ORDER
                    )
                ),
                "selection_policy": (
                    "explicit_qcd_divisor_index"
                    if qcd_divisor_index is not None
                    else CANONICAL_QCD_SELECTION_POLICY
                ),
                "visible_sector_compatibility_filter": (
                    visible_sector_policy == "intersecting_d7"
                ),
                "contraction_policy": (
                    "allowed_opt_in" if allow_m_below_one else "disallowed_by_default"
                ),
                "selected_radial_scale": float(m_val),
                "selected_qcd_divisor_index": int(qcd_divisor_index),
                "selected_qcd_divisor_label": np.asarray(
                    prime_labels_stable[qcd_divisor_index]
                ).tolist(),
                "post_selection_fallback": (
                    "not_applicable_explicit_override"
                    if qcd_divisor_index is not None
                    else CANONICAL_QCD_POST_SELECTION_FALLBACK
                ),
                "qed_prefilter": canonical_qcd_qed_prefilter,
                "repair_policy": "none",
            }
            if moduli_policy == "canonical_qcd"
            else None
        ),
        "visible_sector_policy": visible_sector_policy,
        "orientifold_kaehler_policy": orientifold_kaehler_policy,
        "qed_selection_policy": qed_selection_policy,
        "qed_selection_seed": int(qed_selection_seed),
        "qed_volume_upper_bound": (
            None if qed_volume_max is None else float(qed_volume_max)
        ),
        "qed_volume_filter_policy": (
            "qcd_qed_prefilter_then_less_than_or_equal_to_127.5_complete_pool"
            if assignment_pool is not None
            else ("disabled" if qed_volume_max is None else "pre_filter_pool_then_reject")
        ),
        "canonical_qcd_qed_prefilter": canonical_qcd_qed_prefilter,
        "assignment_pool_size": 0 if assignment_pool is None else len(assignment_pool),
        "assignment_pool_hash": (
            None
            if assignment_pool_validation is None
            else assignment_pool_validation["pool_hash"]
        ),
        "artifact_status": artifact_status,
        "artifact_acceptance_policy": (
            "complete_validated_hashed_assignment_pool_required"
            if eft_mode
            else "geometry_only_before_assignment_pool"
        ),
        "assignment_pool_status": (
            "not_requested"
            if assignment_pool is None
            else assignment_pool_validation["pool_status"]
        ),
        "assignment_pool_normalization_scope": (
            "each_ordered_qcd_qed_assignment"
            if assignment_pool is not None
            else "not_requested"
        ),
        "assignment_pool_rejection_summary": assignment_pool_rejection_summary,
        "assignment_pool_rejection_policy": (
            "detailed_candidate_pair_records_in_stage2_sidecar_jsonl_aggregate_hdf5"
            if assignment_pool is not None
            else "not_requested"
        ),
        "artifact_write_audit": artifact_write_audit,
        "detached_random_qcd_record": False,
        "c4_basis_convention": "full_cytools_h11_declared_all_c4_assumption",
        "all_h11_c4_assumption": {
            "enabled": True,
            "assumed_h11_minus": 0,
            "status": "declared_modeling_assumption",
            "computed_h11_plus": orientifold.get("h11_plus"),
            "computed_h11_minus": orientifold.get("h11_minus"),
            "provenance": (
                "Paper-style full-CYTools-basis convention; not an inferred "
                "physical orientifold parity result."
            ),
        },
        "eft_mode": bool(eft_mode),
        "visible_sector": visible_sector,
        "claim_boundary": (
            "geometry-derived integer divisor-class toy assignment; not a physical "
            "Standard Model brane construction or source-ensemble reproduction"
        ),
        "potential_matrix_convention": {
            "Q": "h11 x N; instanton charges are columns",
            "L": "2 x N; rows are sign/mantissa and log10 scale",
        },
        "potential_storage": "reconstruct_on_demand_geometry_references_only",
        "potential_reconstruction": {
            "schema_version": POTENTIAL_RECONSTRUCTION_SCHEMA_VERSION,
            "storage": "geometry_references_only",
            "source_datasets": [
                "cytools/geometric/kappa",
                "cytools/geometric/basis_matrix",
                "cytools/geometric/prime_toric_divisors",
                "cytools/geometric/effective_cone",
                "cytools/geometric/tip",
            ],
            "q_orientation": "h11 x N_instanton; charge vectors are columns",
            "difference_convention": factorized_charges["difference_convention"],
            "pair_ordering": factorized_charges["pair_ordering"],
            "direct_source_count": int(nq),
            "pair_source_count": int(num_cross),
            "q_direct_sha256": stable_hash(q_direct.tolist()),
            "pair_source_index_sha256": stable_hash(
                {
                    "pair_i": factorized_charges["pair_i"].tolist(),
                    "pair_j": factorized_charges["pair_j"].tolist(),
                }
            ),
            "coefficient_formula": (
                "CYAxiverse sign/log10 Lambda^4 from reconstructed tau, Kinv, and CY volume"
            ),
            "replay_rtol": POTENTIAL_RECONSTRUCTION_RTOL,
            "replay_atol": POTENTIAL_RECONSTRUCTION_ATOL,
            "qed_source_index": (
                None
                if qed_potential_source_index is None
                else int(qed_potential_source_index)
            ),
        },
        "factorized_charge_convention": factorized_charges["difference_convention"],
        "tip_scale_components": ["angular_scale", "radial_scale"],
        "angular_scale": float(divisor_scale),
        "radial_scale": float(m_val),
        "kahler_cone_rays_exported": bool(export_kahler_rays),
        "orientifold": orientifold,
        "potential_charge_convention": charge_metadata["convention"],
        "raw_effective_cone_ray_count": charge_metadata["raw_count"],
        "canonical_effective_cone_ray_count": charge_metadata["canonical_count"],
        "duplicate_effective_cone_rows_removed": charge_metadata[
            "duplicates_removed"
        ],
        "raw_frst_input": (
            None
            if raw_frst_metadata is None
            else {
                "raw_frst_path": raw_frst_metadata.get("raw_frst_path"),
                "raw_frst_schema_version": raw_frst_metadata.get(
                    "raw_frst_schema_version"
                ),
                "raw_geometry_id": raw_frst_metadata.get("geometry_id"),
                "raw_polytope_id": raw_frst_metadata.get("polytope_id"),
                "raw_full_triangulation_hash": raw_frst_metadata.get(
                    "full_triangulation_hash"
                ),
                "stage1_status": raw_frst_metadata.get("stage1_status"),
            }
        ),
    }
    try:
        with h5py.File(temporary_path, "w") as file:
            file.attrs["schema_version"] = SCHEMA_VERSION
            file.attrs["artifact_status"] = artifact_status
            file.attrs["assignment_pool_status"] = construction_metadata[
                "assignment_pool_status"
            ]
            file.attrs["allow_overwrite_existing_geometry"] = bool(
                allow_overwrite_existing_geometry
            )
            file.attrs["overwrite_performed"] = bool(
                artifact_write_audit["overwrite_performed"]
            )
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
            geometric.create_dataset(
                "tip_pre_normalization",
                data=pre_normalization_tip,
                compression="gzip",
                compression_opts=9,
            )
            geometric.create_dataset("tip_prefactor", data=tip_prefactor, compression="gzip", compression_opts=9)
            geometric.create_dataset("CY_volume", data=volume)
            geometric.create_dataset("CY_volume_pre_normalization", data=pre_normalization_volume)
            divisor_evidence_group = geometric.create_group(
                "divisor_volume_evidence"
            )
            divisor_evidence_group.attrs["schema_version"] = (
                divisor_volume_evidence["schema_version"]
            )
            divisor_evidence_group.attrs["validation_status"] = (
                divisor_volume_evidence["validation_status"]
            )
            divisor_evidence_group.attrs["volume_tolerance"] = (
                DIVISOR_VOLUME_TOLERANCE
            )
            divisor_evidence_group.attrs["qcd_volume_target"] = QCD_VOLUME_TARGET
            divisor_evidence_group.attrs["qcd_volume_tolerance"] = QCD_VOLUME_TOLERANCE
            divisor_evidence_group.attrs["normalization_failure_status"] = (
                "qcd_normalization_failure"
            )
            divisor_evidence_group.attrs["normalization_repair_policy"] = "none"
            divisor_evidence_group.attrs["normalization_checks_json"] = json.dumps(
                _jsonable(normalization_checks), sort_keys=True
            )
            divisor_evidence_group.attrs["prime_divisor_volume_count"] = int(
                prime_divisor_volumes.size
            )
            divisor_evidence_group.attrs["effective_divisor_volume_count"] = int(
                effective_divisor_volumes.size
            )
            divisor_evidence_group.attrs["prime_divisor_volumes_sha256"] = stable_hash(
                prime_divisor_volumes.tolist()
            )
            divisor_evidence_group.attrs[
                "effective_divisor_volumes_sha256"
            ] = stable_hash(effective_divisor_volumes.tolist())
            divisor_evidence_group.attrs[
                "minimum_prime_divisor_volume"
            ] = float(np.min(prime_divisor_volumes))
            divisor_evidence_group.attrs[
                "minimum_effective_divisor_volume"
            ] = float(np.min(effective_divisor_volumes))
            divisor_evidence_group.attrs["prime_divisor_index_base"] = 0
            divisor_evidence_group.attrs["effective_cone_ray_index_base"] = 0
            divisor_evidence_group.create_dataset(
                "basis_order",
                data=np.asarray(topology["basis"], dtype=np.int64),
                compression="gzip",
                compression_opts=9,
            )
            divisor_evidence_group.create_dataset(
                "prime_divisor_indices",
                data=np.arange(prime_labels.size, dtype=np.int64),
            )
            divisor_evidence_group.create_dataset(
                "prime_divisor_labels",
                data=np.asarray(prime_labels_stable, dtype=np.int64),
                compression="gzip",
                compression_opts=9,
            )
            divisor_evidence_group.create_dataset(
                "effective_cone_ray_indices",
                data=np.arange(qprime.shape[0], dtype=np.int64),
            )
            divisor_evidence_group.create_dataset(
                "effective_cone_rays",
                data=qprime,
                compression="gzip",
                compression_opts=9,
            )
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
            if standard_model_divisors is not None:
                standard_model = geometric.create_group("standard_model")
                standard_model.create_dataset(
                    "divisor_indices",
                    data=np.asarray(standard_model_divisors, dtype=int),
                )
                standard_model.create_dataset(
                    "qcd_divisor_index", data=int(qcd_divisor_index)
                )
                standard_model.attrs["selection_policy"] = (
                    "uniform_pairwise_intersection_triangle"
                )
                standard_model.attrs["qcd_selection"] = standard_model_qcd_selection
                standard_model.attrs["divisor_index_base"] = 0
            geometric.attrs["favorable"] = favorable
            geometric.attrs["polytope_id"] = polytope_id
            geometric.attrs["triangulation_id"] = triangulation_id
            geometric.attrs["cy3_fingerprint"] = cy3_fingerprint
            geometric.attrs["sampling_scheme"] = sampling_metadata["scheme"]
            geometric.attrs["volume_backend_requested"] = construction_metadata[
                "volume_backend_requested"
            ]
            geometric.attrs["volume_backend"] = volume_backend
            geometric.attrs["historical_contraction"] = json.dumps(
                _jsonable(construction_metadata["historical_contraction"]),
                sort_keys=True,
                separators=(",", ":"),
            )
            geometric.attrs["kappa_format"] = construction_metadata["kappa_format"]
            geometric.attrs["kappa_index_base"] = construction_metadata["kappa_index_base"]
            geometric.attrs["basis_convention"] = construction_metadata["basis_convention"]
            geometric.attrs["intersection_convention"] = construction_metadata["intersection_convention"]
            geometric.attrs["potential_charge_convention"] = charge_metadata[
                "convention"
            ]
            geometric.attrs["raw_effective_cone_ray_count"] = charge_metadata[
                "raw_count"
            ]
            geometric.attrs["canonical_effective_cone_ray_count"] = charge_metadata[
                "canonical_count"
            ]
            geometric.attrs["duplicate_effective_cone_rows_removed"] = charge_metadata[
                "duplicates_removed"
            ]
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
                if orientifold.get("invariant_kahler_point") is not None:
                    orientifold_group.create_dataset(
                        "invariant_kahler_point",
                        data=orientifold["invariant_kahler_point"],
                    )
                orientifold_group.create_dataset(
                    "prime_divisor_image_indices",
                    data=orientifold["prime_divisor_image_indices"],
                )
                orientifold_group.create_dataset(
                    "prime_divisor_invariant_indices",
                    data=orientifold["prime_divisor_invariant_indices"],
                )
                orientifold_group.attrs["involution_type"] = orientifold["involution_type"]
                orientifold_group.attrs["h11_plus"] = orientifold["h11_plus"]
                orientifold_group.attrs["h11_minus"] = orientifold["h11_minus"]
                orientifold_group.attrs["kaehler_subspace_policy"] = (
                    orientifold_kaehler_policy
                )
                orientifold_group.attrs["kaehler_subspace_validation_status"] = (
                    orientifold.get(
                        "kaehler_subspace_validation_status", "validated"
                    )
                )
            if visible_sector is not None:
                write_visible_sector_hdf5(
                    geometric.create_group("visible_sector"), visible_sector
                )
            if assignment_pool is not None:
                pool_group = geometric.create_group("assignment_pool")
                pool_group.attrs["schema_version"] = "ordered-qcd-qed-pool-1.1"
                pool_group.attrs["pool_status"] = "complete_eligible_ordered_pool"
                pool_group.attrs["qed_volume_comparison"] = "less_than_or_equal_to_127.5"
                pool_group.attrs["pool_hash"] = assignment_pool_validation["pool_hash"]
                pool_group.attrs["rejection_record_policy"] = (
                    "aggregate_counts_and_reasons_only_hdf5_detailed_sidecar_jsonl"
                )
                pool_group.attrs["rejection_summary_json"] = json.dumps(
                    assignment_pool_rejection_summary,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                pool_group.create_dataset(
                    "pool_rank", data=np.asarray([item["pool_rank"] for item in assignment_pool], dtype=np.int64)
                )
                pool_group.create_dataset(
                    "qcd_divisor_index", data=np.asarray([item["qcd_divisor_index"] for item in assignment_pool], dtype=np.int64)
                )
                pool_group.create_dataset(
                    "qed_divisor_index", data=np.asarray([item["qed_divisor_index"] for item in assignment_pool], dtype=np.int64)
                )
                pool_group.create_dataset(
                    "qcd_divisor_label",
                    data=np.asarray([item["qcd_divisor_label"] for item in assignment_pool], dtype=np.int64),
                    compression="gzip",
                    compression_opts=9,
                )
                pool_group.create_dataset(
                    "qed_divisor_label",
                    data=np.asarray([item["qed_divisor_label"] for item in assignment_pool], dtype=np.int64),
                    compression="gzip",
                    compression_opts=9,
                )
                for name in (
                    "qcd_radial_scale", "qcd_volume_scale", "qcd_volume",
                    "qcd_volume_target", "qcd_volume_tolerance",
                    "divisor_volume_tolerance", "qcd_volume_residual", "qed_volume",
                    "minimum_prime_volume", "minimum_effective_volume",
                ):
                    pool_group.create_dataset(
                        name,
                        data=np.asarray([item[name] for item in assignment_pool], dtype=float),
                        compression="gzip",
                        compression_opts=9,
                    )
                string_type = h5py.string_dtype(encoding="utf-8")
                pool_group.create_dataset(
                    "assignment_hash",
                    data=np.asarray([item["assignment_hash"] for item in assignment_pool], dtype=object),
                    dtype=string_type,
                )
                pool_group.create_dataset(
                    "intersection_evidence_json",
                    data=np.asarray(
                        [json.dumps(item["intersection_evidence"], sort_keys=True) for item in assignment_pool],
                        dtype=object,
                    ),
                    dtype=string_type,
                )
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
            potential.attrs["storage_schema"] = "reconstruct_on_demand"
            potential.attrs["schema_version"] = POTENTIAL_RECONSTRUCTION_SCHEMA_VERSION
            potential.attrs["orientation"] = (
                "h11 x N_instanton; charge vectors are columns"
            )
            potential.attrs["difference_convention"] = factorized_charges[
                "difference_convention"
            ]
            potential.attrs["pair_ordering"] = factorized_charges["pair_ordering"]
            potential.attrs["reconstruction_metadata_json"] = json.dumps(
                _jsonable(construction_metadata["potential_reconstruction"]),
                sort_keys=True,
                separators=(",", ":"),
            )
            file.flush()
        finalize_geometry_artifact_write(
            temporary_path,
            filepath,
            allow_overwrite_existing_geometry,
        )
    finally:
        cleanup_temporary_geometry_artifact(temporary_path)


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
    ntfe_face_sampler,
    ntfe_max_face_points,
    ntfe_face_pool_size,
    ntfe_as_generator=True,
):
    """Yield bounded FRST candidates with an explicit sampling contract.

    ``ntfe_fast`` draws fine 2-face triangulations and uses CYTools' direct
    NTFE extension algorithm.  It avoids repeatedly constructing CY data for
    FRSTs that differ only away from the two-faces, but its finite 2-face pools
    define a deliberately restricted, non-uniform proposal distribution.
    Schema 1.1 permits only the package-native ``ntfe_frts`` path with the
    ``fast`` two-face sampler.  Learned/GNN proposals are intentionally not
    exposed by this generator.
    """
    if sampling_scheme not in SAMPLING_SCHEMES:
        raise ValueError(
            f"unsupported sampler {sampling_scheme!r}; GNN samplers are unavailable"
        )
    point_labels = tuple(poly.labels_not_facet)
    if sampling_scheme == "fast":
        yield from poly.random_triangulations_fast(
            N=max_tip_attempts,
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

    if sampling_scheme == "ntfe_fast":
        if ntfe_face_sampler != "fast":
            raise ValueError("schema 1.1 ntfe_fast requires triang_method='fast'")
        if not ntfe_as_generator:
            raise ValueError("schema 1.1 ntfe_fast requires as_generator=True")
        yield from poly.ntfe_frts(
            N=max_tip_attempts,
            make_star=True,
            seed=seed,
            max_npts=ntfe_max_face_points,
            N_face_triangs=ntfe_face_pool_size,
            triang_method="fast",
            as_generator=True,
            backend=backend,
            verbosity=0,
        )
        return

    if sampling_scheme == "ntfe_fast":
        yield from poly.ntfe_frts(
            N=max_tip_attempts,
            make_star=True,
            seed=seed,
            max_npts=ntfe_max_face_points,
            N_face_triangs=ntfe_face_pool_size,
            triang_method=ntfe_face_sampler,
            as_generator=True,
            backend=backend,
            verbosity=0,
        )
        return

    if sampling_scheme == "gnn_ntfe":
        yield from poly.random_triangulations_gnn(
            N=max_tip_attempts,
            make_star=True,
            max_npts=ntfe_max_face_points,
            N_face_triangs=ntfe_face_pool_size,
            as_generator=True,
            seed=seed,
            verbosity=0,
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
        allow_overwrite_existing_geometry,
        max_m,
        max_kaehler_attempts,
        min_divisor_volume,
        min_prime_divisor_volume,
        qcd_volume_min,
        qcd_volume_max,
        moduli_policy,
        qcd_volume_target,
        qcd_divisor_index,
        visible_sector_policy,
        qed_divisor_index,
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
        ntfe_face_sampler,
        ntfe_max_face_points,
        ntfe_face_pool_size,
        ks_database_version,
        orientifold_config,
        export_kahler_rays,
        qed_selection_policy,
        qed_volume_max,
        eft_mode,
        materialize_dense_potential,
        volume_backend,
        proposal_budget,
        retry_budget,
        polytope_source,
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
            "proposal_budget": proposal_budget,
            "retry_budget": retry_budget,
            "fast_height_scale": fast_height_scale,
            "sampling_unit": (
                "two_face_inequivalent_frst"
                if sampling_scheme == "ntfe_fast"
                else "frst"
            ),
            "selection_status": (
                "direct_ntfe_proposal_with_finite_face_pool"
                if sampling_scheme == "ntfe_fast"
                else (
                    "provisionally_fair_frst_markov_chain"
                    if sampling_scheme == "fair"
                    else "biased_random_height_proposal"
                )
            ),
            "ntfe_face_sampler": ntfe_face_sampler,
            "ntfe_max_face_points": ntfe_max_face_points,
            "ntfe_face_pool_size": ntfe_face_pool_size,
            "ntfe_as_generator": True,
            "ntfe_contract": {
                "N": proposal_budget,
                "max_npts": ntfe_max_face_points,
                "N_face_triangs": ntfe_face_pool_size,
                "as_generator": True,
                "backend": backend,
                "triang_method": "fast",
            }
            if sampling_scheme == "ntfe_fast"
            else None,
            "qp_solver_preference": "mosek_if_licensed_then_available",
            "mosek_license_configured": mosek_license["configured"],
            "mosek_license_activated": mosek_license["activated"],
        }

        # Output indices represent accepted geometries, not raw triangulation
        # attempts.  This lets a resumed scan retain prior successful samples.
        existing_indices = []
        index = 1
        if not allow_overwrite_existing_geometry:
            while os.path.exists(output_path(base_dir, h11, polytope_index, index)):
                existing_indices.append(index)
                index += 1
        accepted = len(existing_indices)
        next_output_index = index
        if accepted >= requested and not allow_overwrite_existing_geometry:
            return {
                "ok": True,
                "h11": h11,
                "polytope_index": polytope_index,
                "requested": requested,
                "attempted": 0,
                "accepted": accepted,
                "saved": 0,
                "rejected": 0,
                "skipped": accepted,
            }

        report(
            f"sampling FRST candidates with scheme={sampling_scheme}; "
            f"proposal_budget={proposal_budget}, retry_budget={retry_budget}"
        )
        candidates = triangulation_candidates(
            poly,
            sampling_scheme,
            proposal_budget,
            retry_budget,
            backend,
            seed,
            n_walk,
            n_flip,
            initial_walk_steps,
            fine_tune_steps,
            walk_step_size,
            max_steps_to_wall,
            fast_height_scale,
            ntfe_face_sampler,
            ntfe_max_face_points,
            ntfe_face_pool_size,
        )

        saved = rejected = 0
        duplicate_full_triangulations = 0
        duplicate_ntfe_identity = 0
        seen_full_hashes = set()
        seen_two_face_hashes = set()

        def evaluate_candidate(triangulation, context):
            nonlocal saved, rejected, duplicate_full_triangulations, duplicate_ntfe_identity
            proposal_index = int(context["proposal_index"])
            proposal_seed = int(context["proposal_seed"])
            filepath = output_path(base_dir, h11, polytope_index, next_output_index + saved)
            candidate_base = {
                "h11": h11,
                "polytope_index": polytope_index,
                "polytope_id": polytope_id,
                "candidate_index": proposal_index,
                "proposal_index": proposal_index,
                "sampler": sampling_scheme,
                "proposal_seed": proposal_seed,
            }
            existing_artifact = inspect_geometry_artifact(filepath)
            if existing_artifact["exists"]:
                candidate_base["existing_artifact_audit"] = existing_artifact
            try:
                full_hash, two_face_hash = _triangulation_hashes(triangulation)
                candidate_base.update(
                    {
                        "full_triangulation_hash": full_hash,
                        "two_face_hash": two_face_hash,
                    }
                )
            except Exception as exc:
                rejected += 1
                return ProposalDecision(
                    "invalid_frst",
                    f"could not fingerprint candidate: {exc}",
                    candidate_base,
                )
            if full_hash in seen_full_hashes:
                duplicate_full_triangulations += 1
                return ProposalDecision(
                    "duplicate_full_triangulation",
                    "full triangulation identity already emitted",
                    candidate_base,
                )
            seen_full_hashes.add(full_hash)
            if sampling_scheme == "ntfe_fast" and two_face_hash in seen_two_face_hashes:
                duplicate_ntfe_identity += 1
                return ProposalDecision(
                    "duplicate_ntfe_identity",
                    "two-face identity already emitted",
                    candidate_base,
                )
            if sampling_scheme == "ntfe_fast" and two_face_hash is not None:
                seen_two_face_hashes.add(two_face_hash)
            try:
                report(
                    f"testing FRST proposal {proposal_index}/{proposal_budget} "
                    f"for accepted geometry {accepted + saved + 1}/{requested}"
                )
                generate_and_save_geometry(
                    h11,
                    triangulation.get_cy(),
                    points,
                    np.asarray(triangulation.simplices(), dtype=int),
                    filepath,
                    max_m,
                    max_kaehler_attempts,
                    min_divisor_volume,
                    min_prime_divisor_volume,
                    qcd_volume_min,
                    qcd_volume_max,
                    moduli_policy,
                    qcd_volume_target,
                    qcd_divisor_index,
                    visible_sector_policy,
                    qed_divisor_index,
                    np.random.default_rng(proposal_seed),
                    report,
                    poly=poly,
                    triangulation=triangulation,
                    polytope_id=polytope_id,
                    sampling_metadata=sampling_metadata,
                    ks_database_version=ks_database_version,
                    orientifold_config=orientifold_config,
                    polytope_source=polytope_source,
                    export_kahler_rays=export_kahler_rays,
                    qed_selection_policy=qed_selection_policy,
                    qed_selection_seed=proposal_seed,
                    qed_volume_max=qed_volume_max,
                    materialize_dense_potential=materialize_dense_potential,
                    eft_mode=eft_mode,
                    allow_overwrite_existing_geometry=(
                        allow_overwrite_existing_geometry
                    ),
                    volume_backend=volume_backend,
                )
            except Exception as exc:
                rejected += 1
                report(f"rejected FRST proposal {proposal_index}: {exc}")
                status = _candidate_terminal_status(exc)
                if status == "io_failure":
                    status = "numerical_geometry_failure"
                return ProposalDecision(status, str(exc), candidate_base)
            saved += 1
            return ProposalDecision(
                "accepted_geometry",
                "geometry artifact written atomically",
                {
                    **candidate_base,
                    "output_path": os.path.abspath(filepath),
                    "artifact_status": geometry_artifact_status(
                        eft_mode,
                        "complete_eligible_ordered_pool"
                        if eft_mode
                        else None,
                    ),
                    "overwrite_event": (
                        "replaced_existing_geometry"
                        if existing_artifact.get("exists")
                        else "created_new_geometry"
                    ),
                },
            )

        controller_config = ProposalControllerConfig(
            accepted_target=max(0, requested - accepted),
            proposal_budget=proposal_budget,
            retry_budget=retry_budget,
            h11=h11,
            sampler_name=sampling_scheme,
            deterministic_seed=int(seed),
        )
        controller_report = run_proposal_controller(
            controller_config, candidates, evaluator=evaluate_candidate
        )
        candidate_records = []
        for record in controller_report.records:
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "proposal_index": record.proposal_index,
                    "proposal_seed": record.proposal_seed,
                    "retry_index": record.retry_index,
                    "accepted_count_after": record.accepted_count_after,
                    "terminal_status": record.terminal_status,
                    "terminal_reason": record.reason,
                }
            )
            candidate_records.append(metadata)
        if controller_report.terminal_status != "accepted_geometry":
            candidate_records.append(
                {
                    "h11": h11,
                    "polytope_index": polytope_index,
                    "sampler": sampling_scheme,
                    "terminal_status": controller_report.terminal_status,
                    "terminal_reason": (
                        f"accepted={accepted + controller_report.accepted_count} "
                        f"requested={requested}; budget_status="
                        f"{controller_report.budget_status}"
                    ),
                    "proposal_budget": proposal_budget,
                    "retry_budget": retry_budget,
                    "proposal_count": controller_report.proposal_count,
                    "retry_count": controller_report.retry_count,
                }
            )
        h491_diagnostics = None
        if h11 == 491:
            settings = NativeH491SamplerSettings(
                proposal_budget=proposal_budget,
                accepted_target=requested,
                deterministic_seed=int(seed),
                retry_budget=retry_budget,
            )
            h491_diagnostics = diagnose_h491(
                [record for record in candidate_records if "candidate_index" in record],
                settings=settings,
                accepted_target=requested,
            ).to_dict()
        accepted_total = accepted + controller_report.accepted_count
        return {
            "ok": True,
            "h11": h11,
            "polytope_index": polytope_index,
            "requested": requested,
            "attempted": controller_report.proposal_count,
            "accepted": accepted_total,
            "saved": saved,
            "rejected": rejected,
            "skipped": len(existing_indices),
            "proposal_count": controller_report.proposal_count,
            "retry_count": controller_report.retry_count,
            "controller_report": controller_report.to_dict(),
            "h491_diagnostics": h491_diagnostics,
            "duplicate_full_triangulations": duplicate_full_triangulations,
            "duplicate_ntfe_identity": duplicate_ntfe_identity,
            "candidate_terminal_records": candidate_records,
        }
    except Exception as exc:
        return {
            "ok": False,
            "h11": h11,
            "polytope_index": polytope_index,
            "error": repr(exc),
        }


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


def load_polytope_manifest(path):
    """Load explicit KS polytope vertices when the remote endpoint is unavailable."""
    try:
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read polytope manifest {path!r}: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("polytopes"), list):
        raise RuntimeError("Polytope manifest must be an object with a 'polytopes' list.")

    by_h11 = {}
    for position, entry in enumerate(manifest["polytopes"], start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Polytope manifest entry {position} must be an object.")
        try:
            raw_h11 = entry["h11"]
            raw_vertices = np.asarray(entry["vertices"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Polytope manifest entry {position} needs integer h11 and vertices."
            ) from exc
        if isinstance(raw_h11, bool) or not isinstance(raw_h11, (int, np.integer)):
            raise RuntimeError(f"Polytope manifest entry {position} has a non-integer h11.")
        if raw_vertices.dtype.kind not in "iu":
            raise RuntimeError(
                f"Polytope manifest entry {position} has non-integer vertex coordinates."
            )
        h11 = int(raw_h11)
        vertices = np.asarray(raw_vertices, dtype=int)
        if h11 < 1 or vertices.ndim != 2 or vertices.shape[1] != 4 or vertices.shape[0] < 5:
            raise RuntimeError(
                f"Polytope manifest entry {position} is not a four-dimensional "
                "vertex list."
            )
        by_h11.setdefault(h11, []).append(vertices.tolist())
    return {"source": manifest.get("source"), "by_h11": by_h11}


def load_mirror_polytopes(parquet_dir, h11, limit, favorable):
    """Read favorable N-lattice polytopes from the KS Parquet mirror."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "The Parquet mirror source requires pyarrow in the CYTools environment."
        ) from exc

    parquet_dir = os.path.abspath(os.fspath(parquet_dir))
    if not os.path.isdir(parquet_dir):
        raise RuntimeError(f"Parquet mirror directory does not exist: {parquet_dir}")
    paths = sorted(
        glob.glob(os.path.join(parquet_dir, "polytopes-4d-*-vertices.parquet")),
        key=lambda path: int(os.path.basename(path).split("-")[2]),
    )
    if not paths:
        raise RuntimeError(
            "No polytopes-4d-*-vertices.parquet files found in mirror directory "
            f"{parquet_dir}."
        )

    records = []
    for path in paths:
        table = parquet.read_table(
            path, columns=["vertices", "vertex_count", "h11", "h12"]
        )
        for row_index, row in enumerate(table.to_pylist()):
            # The published mirror uses the dual Hodge-label convention:
            # physical h11 is the mirror h12 column.
            physical_h11 = int(row["h12"])
            if physical_h11 != int(h11):
                continue
            vertices = np.asarray(row["vertices"], dtype=int)
            poly = Polytope(vertices, deterministic_glsm_basis=True)
            if int(poly.h11()) != int(h11):
                raise RuntimeError(
                    "KS mirror Hodge-label convention check failed: "
                    f"{os.path.basename(path)} row {row_index} has requested "
                    f"h11={h11}, but CYTools constructed h11={poly.h11()}."
                )
            if favorable is not None and bool(poly.is_favorable(lattice="N")) != favorable:
                continue
            records.append(
                (
                    poly,
                    {
                        "source_kind": "huggingface_parquet_mirror",
                        "dataset": KS_MIRROR_DATASET,
                        "dataset_url": KS_MIRROR_DATASET_URL,
                        "parquet_file": os.path.abspath(path),
                        "row_index": int(row_index),
                        "mirror_h11": int(row["h11"]),
                        "mirror_h12": int(row["h12"]),
                        "physical_h11": physical_h11,
                        "physical_h21": int(row["h11"]),
                        "vertex_count": int(row["vertex_count"]),
                        "favorable_checked_by": (
                            "CYTools Polytope.is_favorable(lattice='N')"
                        ),
                    },
                )
            )
            if len(records) >= limit:
                return records
    return records


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
    moduli_policy,
    qcd_volume_target,
    qcd_divisor_index,
    visible_sector_policy,
    qed_divisor_index,
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
    ntfe_face_sampler,
    ntfe_max_face_points,
    ntfe_face_pool_size,
    ks_database_version,
    favorable,
    orientifold_config,
    export_kahler_rays,
    frsts_per_polytope=None,
    replacement_pool_size=0,
    return_replacement_tasks=False,
    polytope_manifest=None,
    qed_selection_policy="uniform_eligible",
    qed_volume_max=None,
    eft_mode=False,
    materialize_dense_potential=False,
    volume_backend=None,
    proposal_budget=None,
    retry_budget=None,
):
    """Fetch favorable polytopes and assign each its FRST output target.

    When ``return_replacement_tasks`` is true, the first ``n_geometries``
    polytopes are active and any additional fetched polytopes are returned as
    zero-target replacement tasks.  The replacement tasks receive their real
    target only after an active polytope produces a shortfall.
    """
    fetch_limit = n_geometries + max(0, replacement_pool_size)
    proposal_budget = max_tip_attempts if proposal_budget is None else int(proposal_budget)
    retry_budget = max_retries if retry_budget is None else int(retry_budget)
    if proposal_budget < 1 or retry_budget < 0:
        raise ValueError("proposal_budget must be positive and retry_budget cannot be negative")
    if polytope_manifest is None:
        polytopes = list(
            fetch_polytopes(
                h11=h11,
                limit=fetch_limit,
                lattice="N",
                favorable=favorable,
                deterministic_glsm_basis=True,
            )
        )
        polytope_sources = [
            {
                "source_kind": "cytools_fetch_polytopes",
                "query": {
                    "h11": int(h11),
                    "lattice": "N",
                    "favorable": favorable,
                    "limit": fetch_limit,
                    "deterministic_glsm_basis": True,
                },
                "selection_index": index,
            }
            for index, _ in enumerate(polytopes, start=1)
        ]
    else:
        polytopes = []
        for vertices in polytope_manifest["by_h11"].get(h11, []):
            poly = Polytope(vertices, deterministic_glsm_basis=True)
            if int(poly.h11()) != h11:
                raise RuntimeError(
                    "The local polytope manifest h11 does not match CYTools: "
                    f"requested {h11}, obtained {poly.h11()}."
                )
            if favorable is None or bool(poly.is_favorable(lattice="N")) == favorable:
                polytopes.append(poly)
        polytopes = polytopes[:fetch_limit]
        polytope_sources = [
            {
                "source_kind": "local_polytope_manifest",
                "manifest_source": polytope_manifest.get("source"),
            }
            for _ in polytopes
        ]
    if not polytopes:
        return ([], []) if return_replacement_tasks else []

    active_polytopes = polytopes[:n_geometries]
    replacement_polytopes = polytopes[n_geometries:]
    active_sources = polytope_sources[:n_geometries]
    replacement_sources = polytope_sources[n_geometries:]

    if frsts_per_polytope is not None:
        # In per-polytope mode, --n is the requested number of favorable
        # polytopes. Preserve the combined target if fewer are available by
        # distributing it as evenly as possible across those that were found.
        total_target = n_geometries * frsts_per_polytope
        base_target, remainder = divmod(total_target, len(active_polytopes))
        counts = [
            base_target + (1 if index < remainder else 0)
            for index in range(len(active_polytopes))
        ]
    else:
        # Default mode: give every available polytope one geometry, then
        # distribute remaining requests round-robin as extra triangulations.
        counts = [1] * len(active_polytopes)
        for index in itertools.islice(
            itertools.cycle(range(len(active_polytopes))),
            n_geometries - len(active_polytopes),
        ):
            counts[index] += 1

    def make_task(polytope_index, poly, count, polytope_source):
        return (
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
            moduli_policy,
            qcd_volume_target,
            qcd_divisor_index,
            visible_sector_policy,
            qed_divisor_index,
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
            ntfe_face_sampler,
            ntfe_max_face_points,
            ntfe_face_pool_size,
            ks_database_version,
            orientifold_config,
            export_kahler_rays,
            qed_selection_policy,
            qed_volume_max,
            eft_mode,
            materialize_dense_potential,
            volume_backend,
            proposal_budget,
            retry_budget,
            polytope_source,
        )

    tasks = [
        make_task(polytope_index, poly, count, polytope_source)
        for polytope_index, (poly, count, polytope_source) in enumerate(
            zip(active_polytopes, counts, active_sources), start=1
        )
    ]
    if not return_replacement_tasks:
        return tasks

    replacement_tasks = [
        make_task(polytope_index, poly, 0, polytope_source)
        for polytope_index, (poly, polytope_source) in enumerate(
            zip(replacement_polytopes, replacement_sources),
            start=len(active_polytopes) + 1,
        )
    ]
    return tasks, replacement_tasks


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
    moduli_policy,
    qcd_volume_target,
    qcd_divisor_index,
    visible_sector_policy,
    qed_divisor_index,
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
    ntfe_face_sampler,
    ntfe_max_face_points,
    ntfe_face_pool_size,
    ks_database_version,
    favorable,
    orientifold_config,
    export_kahler_rays,
    frsts_per_polytope=None,
    polytope_manifest=None,
    qed_selection_policy="uniform_eligible",
    qed_volume_max=None,
    eft_mode=False,
    materialize_dense_potential=False,
    volume_backend=None,
    proposal_budget=None,
    retry_budget=None,
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
        moduli_policy,
        qcd_volume_target,
        qcd_divisor_index,
        visible_sector_policy,
        qed_divisor_index,
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
        ntfe_face_sampler,
        ntfe_max_face_points,
        ntfe_face_pool_size,
        ks_database_version,
        favorable,
        orientifold_config,
        export_kahler_rays,
        frsts_per_polytope,
        polytope_manifest=polytope_manifest,
        qed_selection_policy=qed_selection_policy,
        qed_volume_max=qed_volume_max,
        eft_mode=eft_mode,
        materialize_dense_potential=materialize_dense_potential,
        volume_backend=volume_backend,
        proposal_budget=proposal_budget,
        retry_budget=retry_budget,
    )
    if not tasks:
        print(f"No favorable N-lattice polytopes found for h11={h11}.")
        return 0

    requested_outputs = sum(task[3] for task in tasks)
    if frsts_per_polytope is not None and len(tasks) < n_geometries:
        targets = [task[3] for task in tasks]
        print(
            f"Requested {n_geometries} favorable polytopes but found "
            f"{len(tasks)}; redistributed the combined target of "
            f"{requested_outputs} FRSTs across per-polytope targets {targets}."
        )
    print(
        f"Found {len(tasks)} favorable polytope(s); requesting "
        f"{requested_outputs} geometry/triangulation output(s)."
    )
    return _run_tasks(tasks, n_cores)


def _run_tasks(tasks, n_cores):
    """Run task tuples in one shared pool and aggregate their results."""
    saved = 0
    failures = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_polytope, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            label = f"h11={result['h11']} np_{result['polytope_index']:07d}"
            if not result["ok"]:
                print(f"ERROR {label}: {result['error']}")
                failures.append(result)
                continue
            saved += result["saved"]
            print(
                f"{label}: accepted "
                f"{result['accepted']}/{result['requested']} geometries after "
                f"{result['attempted']} FRST attempts; saved {result['saved']}, "
                f"rejected {result['rejected']}, skipped {result['skipped']}."
            )
    if failures:
        details = "; ".join(
            f"h11={failure['h11']} np_{failure['polytope_index']:07d}: "
            f"{failure['error']}"
            for failure in failures
        )
        raise RuntimeError(f"CYTools generation failed; no complete batch result: {details}")
    return saved


def run_batches(
    h11_values,
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
    moduli_policy,
    qcd_volume_target,
    qcd_divisor_index,
    visible_sector_policy,
    qed_divisor_index,
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
    ntfe_face_sampler,
    ntfe_max_face_points,
    ntfe_face_pool_size,
    ks_database_version,
    favorable,
    orientifold_config,
    export_kahler_rays,
    frsts_per_polytope=None,
    replace_rejected_polytopes=False,
    max_polytope_replacements=10,
    polytope_manifest=None,
    geometry_targets=None,
    qed_selection_policy="uniform_eligible",
    qed_volume_max=None,
    eft_mode=False,
    materialize_dense_potential=False,
    volume_backend=None,
    collect_records=False,
    proposal_budget=None,
    retry_budget=None,
):
    """Plan all h11 values into one pool, without an h11 completion barrier.

    Tasks for a later h11 are submitted as soon as that h11 has been planned.
    Therefore a slow polytope does not prevent idle workers from taking work
    from another h11 value.
    """
    total_saved = 0
    all_results = []
    pending = set()
    replacement_tasks_by_h11 = {}

    def set_task_target(task, target):
        """Return a task tuple with a new accepted-output target."""
        return task[:3] + (target,) + task[4:]

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        for h11 in h11_values:
            print(f"\n>>> Processing h11={h11} <<<")
            if geometry_targets is None:
                geometry_target = n_geometries
                planning_polytope_count = n_geometries
                planning_frsts_per_polytope = frsts_per_polytope
            else:
                geometry_target = int(geometry_targets[h11])
                planning_polytope_count = {50: 50, 100: 50, 200: 30, 491: 1}[h11]
                planning_frsts_per_polytope = 100 if h11 == 491 else 10
            planning_sampling_scheme = (
                sampling_scheme
                if geometry_targets is None
                else ("ntfe_fast" if h11 == 491 else "fast")
            )
            planning_max_tip_attempts = (
                max_tip_attempts
                if geometry_targets is None
                else (100 if h11 == 491 else 10)
            )
            planning_max_retries = 500 if geometry_targets is not None else max_retries
            tasks, replacement_tasks = plan_tasks(
                h11,
                planning_polytope_count,
                base_dir,
                seed,
                planning_max_retries,
                planning_max_tip_attempts,
                overwrite,
                max_m,
                max_kaehler_attempts,
                min_divisor_volume,
                min_prime_divisor_volume,
                qcd_volume_min,
                qcd_volume_max,
                moduli_policy,
                qcd_volume_target,
                qcd_divisor_index,
                visible_sector_policy,
                qed_divisor_index,
                verbose,
                planning_sampling_scheme,
                backend,
                n_walk,
                n_flip,
                initial_walk_steps,
                fine_tune_steps,
                walk_step_size,
                max_steps_to_wall,
                fast_height_scale,
                ntfe_face_sampler,
                ntfe_max_face_points,
                ntfe_face_pool_size,
                ks_database_version,
                favorable,
                orientifold_config,
                export_kahler_rays,
                planning_frsts_per_polytope,
                max_polytope_replacements if replace_rejected_polytopes else 0,
                True,
                polytope_manifest,
                qed_selection_policy=qed_selection_policy,
                qed_volume_max=qed_volume_max,
                eft_mode=eft_mode,
                materialize_dense_potential=materialize_dense_potential,
                volume_backend=volume_backend,
                proposal_budget=(
                    planning_max_tip_attempts
                    if proposal_budget is None
                    else proposal_budget
                ),
                retry_budget=(
                    planning_max_retries if retry_budget is None else retry_budget
                ),
            )
            replacement_tasks_by_h11[h11] = replacement_tasks
            if not tasks:
                print(f"No favorable N-lattice polytopes found for h11={h11}.")
                continue

            requested_outputs = sum(task[3] for task in tasks)
            if planning_frsts_per_polytope is not None and len(tasks) < planning_polytope_count:
                targets = [task[3] for task in tasks]
                print(
                    f"Requested {geometry_target} favorable polytopes but found "
                    f"{len(tasks)}; redistributed the combined target of "
                    f"{requested_outputs} FRSTs across per-polytope targets {targets}."
                )
            print(
                f"Found {len(tasks)} favorable polytope(s) for h11={h11}; "
                f"requesting {requested_outputs} geometry/triangulation output(s)."
            )
            if replacement_tasks:
                print(
                    f"Replacement mode enabled for h11={h11}; fetched "
                    f"{len(replacement_tasks)} spare favorable polytope(s)."
                )
            pending.update(executor.submit(process_polytope, task) for task in tasks)

        failures = []
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                all_results.append(result)
                label = f"h11={result['h11']} np_{result['polytope_index']:07d}"
                if not result["ok"]:
                    print(f"ERROR {label}: {result['error']}")
                    failures.append(result)
                    continue
                total_saved += result["saved"]
                print(
                    f"{label}: accepted {result['accepted']}/{result['requested']} "
                    f"geometries after {result['attempted']} FRST attempts; "
                    f"saved {result['saved']}, rejected {result['rejected']}, "
                    f"skipped {result['skipped']}."
                )

                shortfall = max(0, result["requested"] - result["accepted"])
                replacement_tasks = replacement_tasks_by_h11[result["h11"]]
                if (
                    replace_rejected_polytopes
                    and shortfall > 0
                    and replacement_tasks
                ):
                    replacement = set_task_target(replacement_tasks.pop(0), shortfall)
                    print(
                        f"Replacing {label}'s shortfall of {shortfall} "
                        f"accepted geometry/geometries with "
                        f"np_{replacement[0]:07d}."
                    )
                    pending.add(executor.submit(process_polytope, replacement))
                elif replace_rejected_polytopes and shortfall > 0:
                    print(
                        f"No spare favorable polytope remains for {label}; "
                        f"shortfall={shortfall}."
                    )

        if failures:
            details = "; ".join(
                f"h11={failure['h11']} np_{failure['polytope_index']:07d}: "
                f"{failure['error']}"
                for failure in failures
            )
            if not collect_records:
                raise RuntimeError(
                    f"CYTools generation failed; no complete batch result: {details}"
                )
            print(f"Generation completed with worker failures: {details}")
    if collect_records:
        return {"saved": total_saved, "results": all_results}
    return total_saved


def parse_eft_geometry_plan(value):
    """Parse and validate the approved schema 1.1 geometry allocation."""
    plan = {}
    for token in str(value).replace(" ", "").split(","):
        if not token or ":" not in token:
            raise ValueError("geometry plan must use comma-separated h11:count entries")
        raw_h11, raw_count = token.split(":", 1)
        try:
            h11, count = int(raw_h11), int(raw_count)
        except ValueError as exc:
            raise ValueError(f"invalid geometry plan entry {token!r}") from exc
        if h11 < 1 or count < 1 or h11 in plan:
            raise ValueError(f"invalid or duplicate geometry plan entry {token!r}")
        plan[h11] = count
    expected = {50: 500, 100: 500, 200: 300, 491: 100}
    if plan != expected:
        raise ValueError(
            "schema 1.1 requires geometry plan 50:500,100:500,200:300,491:100"
        )
    if sum(plan.values()) != TARGET_GEOMETRY_COUNT:
        raise ValueError("schema 1.1 geometry plan must total 1400")
    return plan


class ModelTargetShortfall(RuntimeError):
    """The complete assignment pools cannot satisfy the requested row quotas."""

    def __init__(self, message, records, allocation=None):
        super().__init__(message)
        self.records = records
        self.allocation = allocation


def _reconstruct_intersection_geometry(kappa, tip):
    """Reconstruct volume, divisor volumes, and the inverse metric from COO data."""
    point = np.asarray(tip, dtype=float).reshape(-1)
    sparse_kappa = np.asarray(kappa, dtype=float)
    if sparse_kappa.ndim != 2 or sparse_kappa.shape[1] != 4:
        raise ValueError("intersection data must be a COO array with four columns")
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("the stored accepted Kaehler point is invalid")
    indices = sparse_kappa[:, :3]
    if not np.all(np.isfinite(sparse_kappa)):
        raise ValueError("intersection data contain non-finite values")
    integer_indices = np.asarray(indices, dtype=np.int64)
    if not np.array_equal(indices, integer_indices):
        raise ValueError("intersection indices must be integral")
    h11 = point.size
    if integer_indices.size and (
        np.min(integer_indices) < 0 or np.max(integer_indices) >= h11
    ):
        raise ValueError("intersection indices are outside the Kaehler-point basis")

    kappa_matrix = np.zeros((h11, h11), dtype=float)
    divisor_volumes = np.zeros(h11, dtype=float)
    cy_volume = 0.0
    for (i, j, k), value in zip(integer_indices, sparse_kappa[:, 3]):
        indices_tuple = (int(i), int(j), int(k))
        permutations = tuple(set(itertools.permutations(indices_tuple)))
        value = float(value)
        cy_volume += (
            len(permutations)
            * value
            * point[i]
            * point[j]
            * point[k]
            / 6.0
        )
        for first, second, third in permutations:
            kappa_matrix[first, second] += value * point[third]
            divisor_volumes[first] += 0.5 * value * point[second] * point[third]

    inverse_metric = 4.0 * (
        np.outer(divisor_volumes, divisor_volumes) - kappa_matrix * cy_volume
    )
    inverse_metric = 0.5 * (inverse_metric + inverse_metric.T)
    if (
        not np.isfinite(cy_volume)
        or cy_volume <= 0.0
        or not np.all(np.isfinite(divisor_volumes))
        or not np.all(np.isfinite(inverse_metric))
    ):
        raise ValueError("reconstructed geometric quantities are non-finite or non-positive")
    return {
        "cy_volume": float(cy_volume),
        "divisor_volumes": divisor_volumes,
        "inverse_metric": inverse_metric,
    }


def _compute_volume_geometry(
    cy,
    tip,
    *,
    volume_backend,
    kappa=None,
    glsm_charge_matrix=None,
    mori_cone=None,
):
    """Compute all Stage-2 geometric volumes through one selected backend."""
    point = np.asarray(tip, dtype=float).reshape(-1)
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("the Kähler point is empty or non-finite")
    volume_backend = resolve_volume_backend(point.size, volume_backend)
    if volume_backend == "fan":
        cy_volume = float(cy.compute_cy_volume(point))
        basis_divisor_volumes = np.asarray(
            cy.compute_divisor_volumes(point, in_basis=True), dtype=float
        )
        prime_divisor_volumes = np.asarray(
            cy.compute_divisor_volumes(point), dtype=float
        )
        inverse_metric = np.asarray(
            cy.compute_inverse_kahler_metric(point), dtype=float
        )
        curve_volumes = np.asarray(cy.compute_curve_volumes(point), dtype=float)
    else:
        if kappa is None:
            raise ValueError("historical_sparse_coo requires COO intersection data")
        if glsm_charge_matrix is None:
            raise ValueError(
                "historical_sparse_coo requires the existing GLSM charge matrix"
            )
        if mori_cone is None:
            raise ValueError("historical_sparse_coo requires Mori-cone rays")
        reconstructed = _reconstruct_intersection_geometry(kappa, point)
        cy_volume = float(reconstructed["cy_volume"])
        basis_divisor_volumes = np.asarray(
            reconstructed["divisor_volumes"], dtype=float
        )
        inverse_metric = np.asarray(reconstructed["inverse_metric"], dtype=float)
        glsm = np.asarray(glsm_charge_matrix, dtype=float)
        if glsm.ndim != 2 or glsm.shape[0] != point.size:
            raise ValueError(
                "the GLSM charge matrix must have one row per divisor-basis coordinate"
            )
        prime_divisor_volumes = np.asarray(glsm.T @ basis_divisor_volumes, dtype=float)
        mori = np.asarray(mori_cone, dtype=float)
        if mori.ndim != 2 or mori.shape[1] != point.size:
            raise ValueError("Mori-cone rays must be expressed in the divisor basis")
        curve_volumes = np.asarray(mori @ point, dtype=float)

    inverse_metric = 0.5 * (inverse_metric + inverse_metric.T)
    expected_shape = (point.size,)
    if (
        basis_divisor_volumes.shape != expected_shape
        or not np.all(np.isfinite(basis_divisor_volumes))
        or prime_divisor_volumes.ndim != 1
        or not np.all(np.isfinite(prime_divisor_volumes))
        or curve_volumes.ndim != 1
        or not np.all(np.isfinite(curve_volumes))
        or inverse_metric.shape != (point.size, point.size)
        or not np.all(np.isfinite(inverse_metric))
        or not np.isfinite(cy_volume)
    ):
        raise ValueError(
            f"{volume_backend} returned non-finite or incorrectly shaped geometry data"
        )
    return {
        "cy_volume": cy_volume,
        "basis_divisor_volumes": basis_divisor_volumes,
        "prime_divisor_volumes": prime_divisor_volumes,
        "inverse_metric": inverse_metric,
        "curve_volumes": curve_volumes,
    }


def _volume_backend_diagnostics(
    cy,
    tip,
    selected_geometry,
    *,
    volume_backend,
    kappa=None,
    glsm_charge_matrix=None,
    mori_cone=None,
    effective_cone_rays=None,
):
    """Record a bounded Fan comparison when historical mode is selected."""
    diagnostics = {
        "selected_backend": volume_backend,
        "status": "not_requested",
    }
    if volume_backend != HISTORICAL_VOLUME_BACKEND:
        return diagnostics
    try:
        fan_geometry = _compute_volume_geometry(
            cy,
            tip,
            volume_backend="fan",
            kappa=kappa,
            glsm_charge_matrix=glsm_charge_matrix,
            mori_cone=mori_cone,
        )
        historical_basis = np.asarray(
            selected_geometry["basis_divisor_volumes"], dtype=float
        )
        fan_basis = np.asarray(fan_geometry["basis_divisor_volumes"], dtype=float)
        comparison = {
            "status": "recorded",
            "point_sha256": stable_hash(np.asarray(tip, dtype=float).tolist()),
            "fan_cy_volume": float(fan_geometry["cy_volume"]),
            "historical_cy_volume": float(selected_geometry["cy_volume"]),
            "fan_min_basis_divisor_volume": float(np.min(fan_basis)),
            "historical_min_basis_divisor_volume": float(np.min(historical_basis)),
            "fan_nonpositive_basis_count": int(np.count_nonzero(fan_basis <= 0.0)),
            "historical_nonpositive_basis_count": int(
                np.count_nonzero(historical_basis <= 0.0)
            ),
            "max_abs_basis_volume_difference": float(
                np.max(np.abs(fan_basis - historical_basis))
            ),
        }
        if effective_cone_rays is not None:
            qprime = np.asarray(effective_cone_rays, dtype=float)
            fan_effective = qprime @ fan_basis
            historical_effective = qprime @ historical_basis
            comparison.update(
                {
                    "fan_min_effective_divisor_volume": float(np.min(fan_effective)),
                    "historical_min_effective_divisor_volume": float(
                        np.min(historical_effective)
                    ),
                    "fan_nonpositive_effective_count": int(
                        np.count_nonzero(fan_effective <= 0.0)
                    ),
                    "historical_nonpositive_effective_count": int(
                        np.count_nonzero(historical_effective <= 0.0)
                    ),
                }
            )
        diagnostics.update(comparison)
    except Exception as exc:
        diagnostics.update(
            {
                "status": "fan_comparison_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return diagnostics


def _signed_log_scale(raw_amplitude, raw_exponent):
    """Encode one finite coefficient in the CYAxiverse sign/log10 convention."""
    raw_amplitude = np.asarray(raw_amplitude, dtype=float)
    raw_exponent = np.asarray(raw_exponent, dtype=float)
    if (
        raw_amplitude.shape != raw_exponent.shape
        or not np.all(np.isfinite(raw_amplitude))
        or not np.all(np.isfinite(raw_exponent))
        or np.any(raw_amplitude == 0.0)
    ):
        raise ValueError("potential coefficients must be finite and nonzero")
    return np.vstack(
        (
            np.sign(raw_amplitude),
            np.log10(np.abs(raw_amplitude)) + raw_exponent,
        )
    )


def reconstruct_potential_from_reference(reference, assignment):
    """Reconstruct one bounded potential view from compact geometry references.

    The HDF5 artifact stores the intersection tensor, divisor basis, effective
    cone, and accepted Kaehler point, but not dense potential or metric arrays.
    Reconstruct those arrays transiently for exact source matching and rank
    certification of one EFT row.
    """
    h11 = int(reference["h11"])
    effective_cone = np.asarray(reference["effective_cone"], dtype=np.int64)
    if effective_cone.ndim != 2 or effective_cone.shape[1] != h11:
        raise ValueError("effective-cone reconstruction input has an invalid shape")
    if np.unique(effective_cone, axis=0).shape[0] != effective_cone.shape[0]:
        raise ValueError("effective-cone reconstruction input contains duplicate rays")
    q_direct = effective_cone.T
    geometry = _reconstruct_intersection_geometry(reference["kappa"], reference["tip"])
    tau = geometry["divisor_volumes"]
    inverse_metric = geometry["inverse_metric"]
    cy_volume = geometry["cy_volume"]
    basis_matrix = np.asarray(reference["basis_matrix"], dtype=np.int64)
    prime_labels = np.asarray(reference["prime_toric_divisors"], dtype=np.int64).reshape(-1)
    if basis_matrix.ndim != 2 or basis_matrix.shape[0] != h11:
        raise ValueError("divisor-basis reconstruction input has an invalid shape")
    if prime_labels.size == 0 or np.any(prime_labels < 0) or np.any(
        prime_labels >= basis_matrix.shape[1]
    ):
        raise ValueError("prime-divisor reconstruction labels are invalid")
    prime_charges = np.asarray(basis_matrix[:, prime_labels].T, dtype=np.int64)
    prime_volumes = prime_charges @ tau
    effective_volumes = effective_cone @ tau
    if not np.all(np.isfinite(prime_volumes)) or not np.all(np.isfinite(effective_volumes)):
        raise ValueError("reconstructed divisor volumes are non-finite")

    direct_count = effective_cone.shape[0]
    pair_i = []
    pair_j = []
    for i in range(direct_count - 1):
        for j in range(i + 1, direct_count):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    q_pair = q_direct[:, pair_j] - q_direct[:, pair_i]
    q = np.concatenate((q_direct, q_pair), axis=1)

    prefactor = 8.0 * math.pi / cy_volume**2
    direct_tau = effective_cone @ tau
    direct_amplitude = prefactor * direct_tau
    direct_exponent = -2.0 * math.log10(math.e) * math.pi * direct_tau
    pair_sum = effective_cone[pair_i] + effective_cone[pair_j]
    pair_metric = np.einsum(
        "ai,ij,aj->a", effective_cone[pair_i], inverse_metric, effective_cone[pair_j]
    )
    pair_tau_sum = pair_sum @ tau
    pair_amplitude = prefactor * (math.pi * pair_metric + pair_tau_sum)
    pair_exponent = -2.0 * math.log10(math.e) * math.pi * (
        direct_tau[pair_i] + direct_tau[pair_j]
    )
    l = np.concatenate(
        (
            _signed_log_scale(direct_amplitude, direct_exponent),
            _signed_log_scale(pair_amplitude, pair_exponent),
        ),
        axis=1,
    )

    qed_charge = np.asarray(
        prime_charges[int(assignment["qed_divisor_index"])], dtype=np.int64
    )
    source_index = next(
        (
            index
            for index in range(direct_count)
            if np.array_equal(q[:, index], qed_charge)
        ),
        None,
    )
    if source_index is None:
        qed_tau = float(qed_charge @ tau)
        qed_raw_amplitude = prefactor * qed_tau
        qed_raw_exponent = -2.0 * math.log10(math.e) * math.pi * qed_tau
        qed_l = _signed_log_scale(
            np.asarray([qed_raw_amplitude]), np.asarray([qed_raw_exponent])
        )
        source_index = q.shape[1]
        q = np.concatenate((q, qed_charge.reshape(-1, 1)), axis=1)
        l = np.concatenate((l, qed_l), axis=1)

    certificate = {
        "schema_version": POTENTIAL_RECONSTRUCTION_SCHEMA_VERSION,
        "storage": "geometry_references_only",
        "q_orientation": "h11 x N_instanton; charge vectors are columns",
        "difference_convention": (
            "q_pair[:, k] = q_direct[:, pair_j[k]] - q_direct[:, pair_i[k]]"
        ),
        "pair_ordering": "lexicographic_i_then_j_with_i_less_than_j",
        "direct_count": int(direct_count),
        "pair_count": int(pair_i.size),
        "qed_source_index": int(source_index),
        "qed_source_kind": (
            "direct_effective_cone" if source_index < direct_count else "appended_prime_divisor_e3"
        ),
        "effective_cone_sha256": stable_hash(effective_cone.tolist()),
        "pair_source_index_sha256": stable_hash(
            {"pair_i": pair_i.tolist(), "pair_j": pair_j.tolist()}
        ),
        "q_sha256": stable_hash(q.tolist()),
        "l_sha256": stable_hash(l.tolist()),
        "replay_rtol": POTENTIAL_RECONSTRUCTION_RTOL,
        "replay_atol": POTENTIAL_RECONSTRUCTION_ATOL,
    }
    return {
        "Q": q,
        "L": l,
        "qed_charge": qed_charge,
        "direct_count": direct_count,
        "source_index": int(source_index),
        "reconstruction": {
            **geometry,
            "prime_divisor_volumes": prime_volumes,
            "effective_divisor_volumes": effective_volumes,
            "q_direct": q_direct,
            "pair_i": pair_i,
            "pair_j": pair_j,
        },
        "certificate": certificate,
    }


def _geometry_reference(path):
    """Read compact geometry/pool references needed by the EFT row layer."""
    with h5py.File(path, "r") as file:
        metadata = json.loads(file.attrs["construction_metadata_json"])
        geometric = file["cytools/geometric"]
        geometry_id = str(metadata.get("cy3_fingerprint"))
        if not geometry_id or geometry_id == "None":
            geometry_id = stable_hash({"path": os.path.abspath(path), "metadata": metadata})
        result = {
            "geometry_id": geometry_id,
            "geometry_file": os.path.abspath(path),
            "geometry_hash": str(metadata.get("cy3_fingerprint", "")),
            "geometry_schema_version": str(metadata.get("schema_version", SCHEMA_VERSION)),
            "charge_factorized_schema_version": str(
                metadata.get("charge_factorized_schema_version", CHARGE_FACTORIZED_SCHEMA_VERSION)
            ),
            "normalization_map_version": str(
                metadata.get("normalization_map_version", NORMALIZATION_MAP_VERSION)
            ),
            "h11": int(geometric["h11"][()]),
            "h21": int(geometric["h21"][()]),
            "sampler": str(metadata.get("sampling", {}).get("scheme", "unknown")),
        }
        if "assignment_pool" not in geometric:
            raise ModelTargetShortfall(
                f"geometry {geometry_id} has no persisted complete assignment pool",
                [],
            )
        pool_group = geometric["assignment_pool"]
        pool_size = int(pool_group["pool_rank"].shape[0])
        result["pool_size"] = pool_size
        result["pool"] = {
            name: pool_group[name][()]
            for name in (
                "pool_rank", "qcd_divisor_index", "qed_divisor_index",
                "qcd_divisor_label", "qed_divisor_label", "qcd_radial_scale",
                "qcd_volume_scale", "qcd_volume", "qcd_volume_target",
                "qcd_volume_tolerance", "divisor_volume_tolerance",
                "qcd_volume_residual", "qed_volume",
                "minimum_prime_volume", "minimum_effective_volume", "assignment_hash",
            )
        }
        required_reconstruction_datasets = (
            "kappa",
            "basis_matrix",
            "prime_toric_divisors",
            "effective_cone",
            "tip",
        )
        missing_references = [
            name for name in required_reconstruction_datasets if name not in geometric
        ]
        if missing_references:
            raise ValueError(
                "geometry is missing potential reconstruction references: "
                + ", ".join(missing_references)
            )
        result["reconstruction"] = {
            name: geometric[name][()]
            for name in required_reconstruction_datasets
        }
        result["reconstruction_certificate"] = metadata.get(
            "potential_reconstruction", {}
        )
    return result


def _decode_hdf5_text(value):
    """Decode one scalar HDF5 UTF-8 value without changing its identity."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _materialize_row_potential(reference, assignment):
    """Materialize one bounded Q/L view from geometry references only."""
    reconstruction = dict(reference["reconstruction"])
    reconstruction["h11"] = reference["h11"]
    return reconstruct_potential_from_reference(reconstruction, assignment)


def expand_eft_reference_rows(
    accepted_geometry_paths, base_seed, minimum_rows, maximum_rows
):
    """Build compact rows with replacement sampling and bounded retries.

    Each geometry receives a deterministic requested unique-row quota ``k_g``.
    Draws use replacement, accepted duplicate assignment identities collapse,
    row-construction failures trigger another draw from the same geometry, and
    the draw cap is ``M_g = 10 * k_g``.  The returned allocation contains the
    per-geometry accounting needed to audit cap-induced capacity shortfalls.
    """
    references = [_geometry_reference(path) for path in sorted(accepted_geometry_paths)]
    references.sort(key=lambda reference: reference["geometry_id"])
    assignment_pools = {
        reference["geometry_id"]: [
            _decode_hdf5_text(value) for value in reference["pool"]["assignment_hash"]
        ]
        for reference in references
    }
    reference_by_id = {reference["geometry_id"]: reference for reference in references}

    def build_row_for_draw(geometry_id, pool_rank, draw_seed, draw_index):
        reference = reference_by_id[geometry_id]
        pool = reference["pool"]
        pool_positions = np.flatnonzero(pool["pool_rank"] == int(pool_rank))
        if len(pool_positions) != 1:
            return {
                "accepted": False,
                "status": "invalid_geometry_reference",
                "reason": f"persisted pool rank {pool_rank} is not unique for {geometry_id}",
            }
        position = int(pool_positions[0])
        assignment_hash = _decode_hdf5_text(pool["assignment_hash"][position])
        assignment = {
            "geometry_id": geometry_id,
            "geometry_file": reference["geometry_file"],
            "geometry_hash": reference["geometry_hash"],
            "geometry_schema_version": reference["geometry_schema_version"],
            "charge_factorized_schema_version": reference[
                "charge_factorized_schema_version"
            ],
            "normalization_map_version": reference["normalization_map_version"],
            "h11": reference["h11"],
            "h21": reference["h21"],
            "qcd_divisor_index": int(pool["qcd_divisor_index"][position]),
            "qed_divisor_index": int(pool["qed_divisor_index"][position]),
            "qcd_divisor_label": np.asarray(pool["qcd_divisor_label"][position]).tolist(),
            "qed_divisor_label": np.asarray(pool["qed_divisor_label"][position]).tolist(),
            "assignment_hash": assignment_hash,
            "assignment_pool_rank": int(pool_rank),
            "assignment_pool_size": reference["pool_size"],
            "model_seed": int(draw_seed),
            "draw_seed": int(draw_seed),
            "draw_index": int(draw_index),
            "qcd_radial_scale": float(pool["qcd_radial_scale"][position]),
            "qcd_volume_scale": float(pool["qcd_volume_scale"][position]),
            "qcd_volume": float(pool["qcd_volume"][position]),
            "qcd_volume_target": float(pool["qcd_volume_target"][position]),
            "qcd_volume_tolerance": float(pool["qcd_volume_tolerance"][position]),
            "divisor_volume_tolerance": float(pool["divisor_volume_tolerance"][position]),
            "qcd_volume_residual": float(pool["qcd_volume_residual"][position]),
            "qed_volume": float(pool["qed_volume"][position]),
            "minimum_prime_volume": float(pool["minimum_prime_volume"][position]),
            "minimum_effective_volume": float(pool["minimum_effective_volume"][position]),
        }
        try:
            potential = _materialize_row_potential(reference, assignment)
            row = serialize_eft_row(
                assignment,
                assignment,
                potential,
                model_id=f"{geometry_id}:assignment-{assignment_hash}",
            )
        except Exception as error:
            status = getattr(error, "terminal_status", getattr(error, "category", "invalid_row_schema"))
            if status not in {
                "invalid_geometry_reference",
                "potential_term_mismatch",
                "rank_span_classification_failure",
                "missing_assignment_derived_data",
                "invalid_row_schema",
            }:
                status = "invalid_row_schema"
            return {
                "accepted": False,
                "status": status,
                "reason": f"{type(error).__name__}: {error}",
            }
        return {"accepted": True, "record": row}

    # Validate every persisted assignment before sampling.  This makes the
    # capacity claim the number of distinct ordered assignments that can
    # actually produce a schema-valid row, rather than the raw pool size or
    # the number encountered by the replacement sampler.
    validated_pool_ranks = {}
    prevalidated_rows = {}
    validation_failure_records = []
    for geometry_id in sorted(assignment_pools):
        valid_ranks = []
        pool = reference_by_id[geometry_id]["pool"]
        for pool_rank in range(reference_by_id[geometry_id]["pool_size"]):
            validation_seed = stable_seed(
                "stage12-capacity-validation", base_seed, geometry_id, pool_rank
            )
            result = build_row_for_draw(
                geometry_id, pool_rank, validation_seed, pool_rank
            )
            if result.get("accepted"):
                valid_ranks.append(pool_rank)
                prevalidated_rows[(geometry_id, pool_rank)] = result["record"]
                continue
            validation_failure_records.append(
                {
                    "terminal_status": result.get(
                        "status", "invalid_row_schema"
                    ),
                    "terminal_reason": result.get(
                        "reason", "row construction failed during capacity validation"
                    ),
                    "geometry_id": geometry_id,
                    "assignment_pool_rank": pool_rank,
                    "assignment_hash": _decode_hdf5_text(
                        pool["assignment_hash"][pool_rank]
                    ),
                    "capacity_validation": True,
                }
            )
        validated_pool_ranks[geometry_id] = valid_ranks

    def capacity_validated_row_callback(geometry_id, pool_rank, draw_seed, draw_index):
        prevalidated = prevalidated_rows.get((geometry_id, int(pool_rank)))
        if prevalidated is None:
            return {
                "accepted": False,
                "status": "invalid_row_schema",
                "reason": "assignment was not schema-valid during capacity validation",
            }
        return {"accepted": True, "record": dict(prevalidated)}

    capacity = sample_capacity_aware_assignments(
        assignment_pools,
        base_seed,
        minimum_rows=minimum_rows,
        maximum_rows=maximum_rows,
        row_callback=capacity_validated_row_callback,
        eligible_pool_ranks=validated_pool_ranks,
    )
    allocation = dict(capacity["allocation"])
    allocation.update(
        {
            "planned_accepted_count": capacity["planned_accepted_count"],
            "accepted_count": capacity["accepted_count"],
            "minimum_reached": capacity["minimum_reached"],
            "successful": capacity["successful"],
            "stop_reason": capacity["actual_stop_reason"],
            "terminal_status": capacity["terminal_status"],
            "per_geometry_sampling": capacity["per_geometry_sampling"],
            "sampling_policy": {
                "assignment_sampling": "uniform_with_replacement",
                "row_identity": "geometry_id_plus_ordered_assignment",
                "duplicate_policy": "collapse_duplicate_assignment_draws",
                "row_failure_policy": "retry_same_geometry",
                "draw_cap_formula": "M_g = 10 * k_g",
            },
            "raw_assignment_capacity": capacity["raw_assignment_capacity"],
            "validated_assignment_capacity": capacity[
                "validated_assignment_capacity"
            ],
            "rows_written": capacity["accepted_count"],
            "requested_target": capacity["ceiling"],
            "minimum_acceptable": capacity["requested_minimum"],
            "capacity_shortfall": capacity["reconciliation"]["capacity_shortfall"],
            "row_shortfall": capacity["reconciliation"]["row_shortfall"],
            "minimum_shortfall": capacity["reconciliation"]["minimum_shortfall"],
            "production_complete": capacity["reconciliation"][
                "production_complete"
            ],
            "diagnostic_success": capacity["reconciliation"]["diagnostic_success"],
            "dataset_status": capacity["reconciliation"]["dataset_status"],
        }
    )
    terminal_records = [
        {
            "terminal_status": (
                "accepted_model_target"
                if capacity["reconciliation"]["production_complete"]
                else "model_target_shortfall"
            ),
            "terminal_reason": (
                "exact EFT row target reached"
                if capacity["reconciliation"]["production_complete"]
                else "validated assignment capacity or row generation did not reach the exact target"
            ),
            "requested_minimum": capacity["requested_minimum"],
            "ceiling": capacity["ceiling"],
            "accepted_count": capacity["accepted_count"],
            "planned_accepted_count": capacity["planned_accepted_count"],
            "maximum_feasible_rows": capacity["maximum_feasible_rows"],
            "raw_assignment_capacity": capacity["raw_assignment_capacity"],
            "validated_assignment_capacity": capacity[
                "validated_assignment_capacity"
            ],
            "rows_written": capacity["accepted_count"],
            "dataset_status": capacity["reconciliation"]["dataset_status"],
            "production_complete": capacity["reconciliation"][
                "production_complete"
            ],
            "capacity_shortfall": capacity["reconciliation"][
                "capacity_shortfall"
            ],
            "row_shortfall": capacity["reconciliation"]["row_shortfall"],
            "minimum_shortfall": capacity["reconciliation"][
                "minimum_shortfall"
            ],
        }
    ]
    terminal_records.extend(validation_failure_records)
    rows = capacity["rows"]
    for row_index, row in enumerate(rows):
        geometry_id = row["geometry_id"]
        reference = reference_by_id[geometry_id]
        sampling = capacity["per_geometry_sampling"][geometry_id]
        row.update(
            {
                "row_order": row_index,
                "requested_minimum": capacity["requested_minimum"],
                "ceiling": capacity["ceiling"],
                "accepted_count": capacity["accepted_count"],
                "stop_reason": capacity["actual_stop_reason"],
                "sampling_unit": "ordered_qcd_qed_assignment",
                "assignment_sampling": "uniform_with_replacement_duplicate_collapse",
                "requested_unique_rows": sampling["requested_unique_rows"],
                "draw_cap": sampling["draw_cap"],
                "accepted_unique_rows": sampling["accepted_unique_rows"],
                "duplicate_draws": sampling["duplicate_draws"],
                "failed_draws": sampling["failed_draws"],
                "cap_induced_capacity_shortfall": sampling[
                    "cap_induced_capacity_shortfall"
                ],
            }
        )
        validate_eft_row(row)
        terminal_records.append(
            {
                "model_id": row["model_id"],
                "geometry_id": geometry_id,
                "h11": reference["h11"],
                "sampler": reference["sampler"],
                "terminal_status": "accepted_model_row",
                "terminal_reason": "compact Parquet row prepared",
                "pool_rank": row["assignment_pool_rank"],
                "model_seed": int(row["model_seed"]),
                "draw_index": row["draw_index"],
            }
        )
    for geometry_id in sorted(capacity["per_geometry_sampling"]):
        sampling = capacity["per_geometry_sampling"][geometry_id]
        terminal_records.append(
            {
                "terminal_status": "model_sampling_accounting",
                "terminal_reason": "per-geometry replacement-draw accounting",
                **sampling,
            }
        )
    return rows, terminal_records, allocation


def write_schema11_artifacts(
    output_root,
    *,
    run_manifest,
    candidate_records,
    model_records,
    summary,
    storage_estimate,
    charge_factorized_manifest,
    polytope_manifest,
    include_model_statuses,
    fresh_ensemble_manifest=None,
):
    """Write all external accounting artifacts with no-overwrite atomic writes."""
    atomic_jsonl_dump(
        os.path.join(output_root, "candidate_terminal_statuses.jsonl"), candidate_records
    )
    if include_model_statuses:
        atomic_jsonl_dump(
            os.path.join(output_root, "model_terminal_statuses.jsonl"), model_records
        )
    atomic_json_dump(os.path.join(output_root, "run_manifest.json"), run_manifest)
    atomic_json_dump(
        os.path.join(output_root, "summary_by_h11_and_status.json"), summary
    )
    atomic_json_dump(os.path.join(output_root, "storage_estimate.json"), storage_estimate)
    atomic_json_dump(os.path.join(output_root, "polytope_manifest.json"), polytope_manifest)
    atomic_json_dump(
        os.path.join(output_root, "charge_factorized_manifest.json"),
        charge_factorized_manifest,
    )
    if fresh_ensemble_manifest is not None:
        write_fresh_ensemble_manifest(
            os.path.join(output_root, "fresh_ensemble_manifest.json"),
            fresh_ensemble_manifest,
        )


def factorized_manifest_for_paths(paths):
    """Summarize reference-only potential inputs without loading dense arrays."""
    entries = []
    for path in sorted(paths):
        with h5py.File(path, "r") as file:
            metadata = json.loads(file.attrs["construction_metadata_json"])
            geometric = file["cytools/geometric"]
            effective_cone = np.asarray(geometric["effective_cone"], dtype=np.int64)
            h11 = int(geometric["h11"][()])
            direct_count = int(effective_cone.shape[0])
            pair_i = []
            pair_j = []
            for i in range(direct_count - 1):
                for j in range(i + 1, direct_count):
                    pair_i.append(i)
                    pair_j.append(j)
            reconstruction = metadata.get("potential_reconstruction", {})
            entries.append(
                {
                    "geometry_file": os.path.abspath(path),
                    "geometry_id": metadata.get("cy3_fingerprint"),
                    "schema_version": metadata.get("schema_version"),
                    "charge_factorized_schema_version": str(
                        metadata.get(
                            "charge_factorized_schema_version",
                            CHARGE_FACTORIZED_SCHEMA_VERSION,
                        )
                    ),
                    "direct_shape": [h11, direct_count],
                    "pair_count": len(pair_i),
                    "pair_i_sha256": stable_hash(pair_i),
                    "pair_j_sha256": stable_hash(pair_j),
                    "q_direct_sha256": stable_hash(effective_cone.T.tolist()),
                    "difference_convention": reconstruction.get(
                        "difference_convention",
                        "q_pair[:, k] = q_direct[:, pair_j[k]] - q_direct[:, pair_i[k]]",
                    ),
                    "storage": "geometry_references_only",
                    "dense_potential_present": False,
                }
            )
    return {
        "schema_version": CHARGE_FACTORIZED_SCHEMA_VERSION,
        "representation": "effective_cone_references_plus_reconstructed_pair_sources",
        "materialization": "on_demand_during_eft_row_generation",
        "geometries": entries,
    }


def _fresh_source_query(args, h11_values, polytope_manifest):
    """Record the complete fresh source query without making population claims."""
    criteria = {
        "lattice": "N",
        "favorable": True,
        "reflexive": True,
        "full_dimensional": True,
    }
    returned_polytopes = []
    if polytope_manifest is not None:
        for h11 in sorted(h11_values):
            for ordinal, vertices in enumerate(
                polytope_manifest["by_h11"].get(h11, []), start=1
            ):
                returned_polytopes.append(
                    {
                        "h11": int(h11),
                        "polytope_fingerprint": stable_hash(vertices),
                        "returned_index": ordinal,
                    }
                )
        source = polytope_manifest.get("source") or "local_polytope_manifest"
        revision = f"local-manifest:{os.path.abspath(args.polytope_manifest)}"
    else:
        expected = {50: 50, 100: 50, 200: 30, 491: 1}
        for h11 in sorted(h11_values):
            polytopes = list(
                fetch_polytopes(
                    h11=h11,
                    limit=expected[h11],
                    lattice="N",
                    favorable=True,
                    deterministic_glsm_basis=True,
                )
            )
            for ordinal, poly in enumerate(polytopes, start=1):
                fingerprint, _ = polytope_identity(poly)
                returned_polytopes.append(
                    {
                        "h11": int(h11),
                        "polytope_fingerprint": fingerprint,
                        "returned_index": ordinal,
                    }
                )
        source = args.ks_database_version
        revision = str(args.ks_database_version)
    return {
        "source": source,
        "source_revision": revision,
        "query_criteria": criteria,
        "fresh": True,
        "result_count": len(returned_polytopes),
        "returned_order": [
            item["polytope_fingerprint"] for item in returned_polytopes
        ],
        "returned_polytopes": returned_polytopes,
        "query_ordering": "CYTools returned order preserved within each h11",
    }


def _task17_path():
    """Return the immutable Task 17 JSON path used by the provenance gate."""
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "glimmers_local_pilot_tasks",
            "17_schema11_integration_orchestration.json",
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an adapted finite KS geometry reference. Optional --eft "
            "expands compact ordered QCD-QED assignment rows; it does not "
            "reproduce the Glimmers paper ensemble."
        )
    )
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
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help=(
            "Target number of accepted geometries per h11, or number of "
            "favorable polytopes when --frsts-per-polytope is set."
        ),
    )
    parser.add_argument(
        "--frsts-per-polytope",
        "--triangulations-per-polytope",
        dest="frsts_per_polytope",
        type=int,
        default=None,
        help=(
            "Request this many accepted FRSTs from every favorable polytope; "
            "with this option --n is the number of polytopes to fetch."
        ),
    )
    parser.add_argument(
        "--replace-rejected-polytopes",
        action="store_true",
        help=(
            "Refill accepted-geometry shortfalls with additional favorable "
            "polytopes fetched for the same h11."
        ),
    )
    parser.add_argument(
        "--max-polytope-replacements",
        type=int,
        default=10,
        help=(
            "Maximum spare favorable polytopes fetched per h11 when "
            "--replace-rejected-polytopes is enabled."
        ),
    )
    parser.add_argument("--outdir", type=str, default=".", help="Base directory for output data.")
    parser.add_argument(
        "--allow-overwrite-existing-geometry",
        action="store_true",
        help=(
            "Explicitly authorize replacement of an existing cyax.h5 artifact; "
            "disabled by default and recorded in geometry provenance."
        ),
    )
    parser.add_argument("--cores", type=int, default=None, help="Worker count (default: all available).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible random triangulations.")
    parser.add_argument(
        "--sampling-scheme",
        choices=SAMPLING_SCHEMES,
        default="fair",
        help=(
            "FRST proposal: fair secondary-fan walk (default), biased fast "
            "heights, or package-native ntfe_fast with triang_method=fast."
        ),
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
    parser.add_argument(
        "--database-source",
        choices=("cytools", "mirror", "manifest"),
        default="cytools",
        help=(
            "Polytope source: live CYTools fetch_polytopes (default), a KS "
            "Parquet mirror, or an explicit JSON manifest."
        ),
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=None,
        help=(
            "Directory containing polytopes-4d-*-vertices.parquet files for "
            "--database-source mirror."
        ),
    )
    parser.add_argument(
        "--polytope-manifest",
        default=None,
        help=(
            "JSON file with explicit KS polytope vertices. Use this for a "
            "replay or when the CYTools KS endpoint is unavailable."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=50, help="Bound per-FRST search retries.")
    parser.add_argument(
        "--max-tip-attempts",
        "--ntfe-sample-count",
        dest="max_tip_attempts",
        type=int,
        default=100,
        help=(
            "Maximum FRST candidates tried per polytope; for ntfe_fast this "
            "is Polytope.ntfe_frts N (default: 100)."
        ),
    )
    parser.add_argument(
        "--proposal-budget",
        type=int,
        default=None,
        help="Accepted-geometry proposal cap; defaults to --max-tip-attempts.",
    )
    parser.add_argument(
        "--retry-budget",
        type=int,
        default=None,
        help="Rejected-proposal retry cap; defaults to --max-retries.",
    )
    parser.add_argument(
        "--max-m",
        type=float,
        default=1_000_000.0,
        help=(
            "Maximum radial prefactor for canonical_qcd or potential-control "
            "search for adaptive mode."
        ),
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
        "--ntfe-face-sampler",
        choices=NTFE_FACE_SAMPLERS,
        default="fast",
        help=(
            "2-face FRT proposal used by ntfe_fast. The finite pool is a "
            "coverage proposal, not a uniform NTFE measure."
        ),
    )
    parser.add_argument(
        "--ntfe-max-face-points",
        type=int,
        default=17,
        help="Native ntfe_frts max_npts control (schema 1.1 default: 17).",
    )
    parser.add_argument(
        "--ntfe-face-pool-size",
        "--ntfe-face-triangulations",
        dest="ntfe_face_pool_size",
        type=int,
        default=1000,
        help="Native ntfe_frts N_face_triangs control (schema 1.1 default: 1000).",
    )
    parser.add_argument(
        "--ntfe-as-generator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require ntfe_frts as_generator=True; disabling it is rejected.",
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
        "--volume-backend",
        choices=VOLUME_BACKENDS,
        default="fan",
        help=(
            "Stage-2 volume backend. fan is the CYTools default; "
            "historical_sparse_coo reproduces the historical sparse COO "
            "contraction and is restricted to h11=491; auto selects "
            "historical_sparse_coo at h11=491 and fan otherwise."
        ),
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
        "--moduli-policy",
        choices=("adaptive", "canonical_qcd"),
        default="adaptive",
        help=(
            "Choose the Kähler-moduli construction: adaptive samples angular "
            "points and applies the QCD window; canonical_qcd uses the canonical "
            "stretched-cone ray, samples a pairwise-intersecting divisor triple, "
            "and scales its QCD member to the target."
        ),
    )
    parser.add_argument(
        "--qcd-volume-target",
        type=float,
        default=40.0,
        help=(
            "Target QCD prime-divisor volume for --moduli-policy canonical_qcd "
            "(default: 40)."
        ),
    )
    parser.add_argument(
        "--qcd-divisor-index",
        type=int,
        default=None,
        help=(
            "Zero-based CYTools prime-toric-divisor index for canonical_qcd; "
            "if omitted, choose the QCD member uniformly from a sampled "
            "pairwise-intersecting divisor triple."
        ),
    )
    parser.add_argument(
        "--visible-sector-policy",
        choices=("none", "intersecting_d7"),
        default="none",
        help=(
            "Visible-sector assignment policy. intersecting_d7 requires a "
            "validated O3/O7 involution and selects an invariant QED divisor "
            "intersecting the QCD divisor."
        ),
    )
    parser.add_argument(
        "--qed-divisor-index",
        type=int,
        default=None,
        help=(
            "Zero-based CYTools prime-toric-divisor index for the invariant "
            "QED divisor under intersecting_d7; if omitted, choose the "
            "lowest-volume eligible intersection."
        ),
    )
    parser.add_argument(
        "--qed-selection-policy",
        choices=("uniform_eligible", "explicit"),
        default="uniform_eligible",
        help="Non-EFT visible-sector selection policy; --eft samples its complete pool.",
    )
    parser.add_argument(
        "--qed-volume-max",
        type=float,
        default=None,
        help="Inclusive QED volume upper bound; --eft fixes this to 127.5.",
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
    parser.add_argument(
        "--materialize-dense-potential",
        action="store_true",
        help=(
            "Explicit compatibility opt-in to write dense Q/L in addition to "
            "the canonical factorized charge artifact."
        ),
    )
    parser.add_argument(
        "--eft",
        action="store_true",
        help=(
            "Opt in to the approved 1400-geometry compact EFT-reference adaptation "
            "with capacity-aware ordered-pool sampling."
        ),
    )
    parser.add_argument(
        "--eft-geometry-plan",
        default="50:500,100:500,200:300,491:100",
        help="Schema 1.1 geometry plan h11:count,... (the approved plan is fixed).",
    )
    parser.add_argument(
        "--eft-minimum-rows",
        type=int,
        default=MINIMUM_EFT_ROWS,
        help="Minimum accepted EFT-reference rows (schema 1.1: 100000).",
    )
    parser.add_argument(
        "--eft-maximum-rows",
        type=int,
        default=MAXIMUM_EFT_ROWS,
        help="Hard EFT-reference row ceiling (schema 1.1: 200000).",
    )
    parser.add_argument(
        "--eft-output-format",
        choices=("parquet",),
        default="parquet",
        help="Compressed columnar EFT table format.",
    )
    parser.add_argument(
        "--eft-output-path",
        default=None,
        help="EFT Parquet path, relative to --outdir unless absolute.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-worker stages and elapsed times; recommended for large h11.",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be positive")
    if args.frsts_per_polytope is not None and args.frsts_per_polytope < 1:
        parser.error("--frsts-per-polytope must be positive")
    if args.max_polytope_replacements < 0:
        parser.error("--max-polytope-replacements cannot be negative")
    if args.sampling_scheme not in SAMPLING_SCHEMES:
        parser.error("--sampling-scheme has an unsupported value")
    if not args.ntfe_as_generator:
        parser.error("--ntfe-as-generator cannot be disabled; schema 1.1 requires True")
    if args.max_tip_attempts < 1:
        parser.error("--max-tip-attempts must be positive")
    proposal_budget = (
        args.max_tip_attempts
        if args.proposal_budget is None
        else args.proposal_budget
    )
    retry_budget = args.max_retries if args.retry_budget is None else args.retry_budget
    if proposal_budget < 1:
        parser.error("--proposal-budget must be positive")
    if retry_budget < 0:
        parser.error("--retry-budget cannot be negative")
    if args.max_kaehler_attempts < 1:
        parser.error("--max-kaehler-attempts must be positive")
    if args.min_divisor_volume <= 0.0:
        parser.error("--min-divisor-volume must be positive")
    if args.min_prime_divisor_volume <= 0.0:
        parser.error("--min-prime-divisor-volume must be positive")
    if args.qcd_volume_min <= 0.0 or args.qcd_volume_max < args.qcd_volume_min:
        parser.error("--qcd-volume-min must be positive and no greater than --qcd-volume-max")
    if args.qcd_volume_target <= 0.0:
        parser.error("--qcd-volume-target must be positive")
    if args.moduli_policy == "canonical_qcd" and not np.isclose(
        args.qcd_volume_target, QCD_VOLUME_TARGET, rtol=0.0, atol=1e-12
    ):
        parser.error("canonical_qcd requires --qcd-volume-target 40.0")
    if args.qcd_divisor_index is not None and args.qcd_divisor_index < 0:
        parser.error("--qcd-divisor-index must be non-negative")
    if args.qcd_divisor_index is not None and args.moduli_policy != "canonical_qcd":
        parser.error("--qcd-divisor-index requires --moduli-policy canonical_qcd")
    if args.qed_divisor_index is not None and args.qed_divisor_index < 0:
        parser.error("--qed-divisor-index must be non-negative")
    if args.visible_sector_policy == "intersecting_d7" and args.orientifold_file is None:
        parser.error("--visible-sector-policy intersecting_d7 requires --orientifold-file")
    if args.qed_divisor_index is not None and args.visible_sector_policy != "intersecting_d7":
        parser.error("--qed-divisor-index requires --visible-sector-policy intersecting_d7")
    if args.qed_selection_policy == "explicit" and args.qed_divisor_index is None:
        parser.error("--qed-selection-policy explicit requires --qed-divisor-index")
    if args.qed_volume_max is not None and args.qed_volume_max <= 0.0:
        parser.error("--qed-volume-max must be positive")
    if args.eft:
        if args.sampling_scheme != "ntfe_fast":
            parser.error("--eft requires --sampling-scheme ntfe_fast")
        if args.backend != "cgal":
            parser.error("--eft requires --backend cgal")
        if args.ntfe_face_sampler != "fast":
            parser.error("--eft requires ntfe_fast triang_method=fast")
        if args.proposal_budget is not None and args.proposal_budget != 100:
            parser.error("--eft requires a native proposal budget of 100")
        if args.retry_budget is not None and args.retry_budget != 500:
            parser.error("--eft requires a native retry budget of 500")
        if args.max_tip_attempts != 100 or args.ntfe_max_face_points != 17 or args.ntfe_face_pool_size != 1000:
            parser.error(
                "--eft requires ntfe N=100, max_npts=17, and N_face_triangs=1000"
            )
        if not np.isclose(args.fast_height_scale, 0.2, rtol=0.0, atol=1e-12):
            parser.error("--eft requires lower-h11 fast height scale 0.2")
        if args.moduli_policy != "canonical_qcd":
            parser.error("--eft requires --moduli-policy canonical_qcd")
        if args.visible_sector_policy != "intersecting_d7":
            parser.error("--eft requires --visible-sector-policy intersecting_d7")
        if args.orientifold_file is None:
            parser.error("--eft requires --orientifold-file")
        if args.eft_minimum_rows != MINIMUM_EFT_ROWS:
            parser.error("--eft-minimum-rows must be 100000 for schema 1.1")
        if args.eft_maximum_rows != MAXIMUM_EFT_ROWS:
            parser.error("--eft-maximum-rows must be 200000 for schema 1.1")
        try:
            eft_geometry_plan = parse_eft_geometry_plan(args.eft_geometry_plan)
        except ValueError as exc:
            parser.error(f"--eft-geometry-plan: {exc}")
        args.qed_volume_max = QED_VOLUME_MAX
    else:
        eft_geometry_plan = None
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
    if args.ntfe_max_face_points < 0:
        parser.error("--ntfe-max-face-points cannot be negative")
    if args.ntfe_face_pool_size < 1:
        parser.error("--ntfe-face-pool-size must be positive")
    if args.h11_interval < 1:
        parser.error("--h11_interval must be positive")
    if args.database_source == "mirror":
        if args.parquet_dir is None:
            parser.error("--database-source mirror requires --parquet-dir")
        if args.polytope_manifest is not None:
            parser.error("--parquet-dir and --polytope-manifest are mutually exclusive")
    elif args.database_source == "manifest":
        if args.polytope_manifest is None:
            parser.error("--database-source manifest requires --polytope-manifest")
        if args.parquet_dir is not None:
            parser.error("--parquet-dir requires --database-source mirror")
    elif args.parquet_dir is not None or args.polytope_manifest is not None:
        parser.error("select --database-source mirror or manifest for the supplied local source")
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
    if args.volume_backend == HISTORICAL_VOLUME_BACKEND and any(
        int(value) != 491 for value in h11_values
    ):
        parser.error("--volume-backend historical_sparse_coo requires h11=491 only")
    if args.eft:
        if args.h11s is not None and set(h11_values) != set(eft_geometry_plan):
            parser.error("--eft --h11s must contain exactly 50,100,200,491")
        h11_values = sorted(eft_geometry_plan)
        parser.error(
            "--eft is now split across generate_stage1_raw_frsts.py and "
            "generate_stage2_eft_reference.py; run stage 1 first and pass its "
            "raw-FRST output to stage 2"
        )
    require_cytools_capabilities(args.sampling_scheme, args.ntfe_face_sampler)
    orientifold_config = load_orientifold(args.orientifold_file)
    favorable = {"true": True, "false": False, "any": None}[args.favorable]
    polytope_manifest = (
        None
        if args.polytope_manifest is None
        else load_polytope_manifest(args.polytope_manifest)
    )
    if polytope_manifest is not None and args.ks_database_version == (
        "CYTools fetch_polytopes endpoint (version not exposed)"
    ):
        args.ks_database_version = (
            f"local polytope manifest: {os.path.abspath(args.polytope_manifest)}"
        )
    provenance_record = None
    source_query = None
    if args.eft:
        source_query = _fresh_source_query(args, h11_values, polytope_manifest)
        source_files = [
            os.path.abspath(__file__),
            os.path.join(os.path.dirname(__file__), "qed_divisor_assignment.py"),
            os.path.join(os.path.dirname(__file__), "glimmers_schema11.py"),
            os.path.join(os.path.dirname(__file__), "glimmers_proposal_controller.py"),
            os.path.join(os.path.dirname(__file__), "glimmers_h491_diagnostics.py"),
            os.path.join(os.path.dirname(__file__), "glimmers_eft_row_schema.py"),
            os.path.join(os.path.dirname(__file__), "glimmers_provenance.py"),
            os.path.join(os.path.dirname(__file__), "glimmers_fresh_ensemble_manifest.py"),
        ]
        if args.polytope_manifest is not None:
            source_files.append(os.path.abspath(args.polytope_manifest))
        try:
            provenance_record = production_provenance_gate(
                repo_root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                task_file=_task17_path(),
                source_files=source_files,
                source_query=source_query,
                output_root=os.path.abspath(args.outdir),
                input_roots=(
                    []
                    if args.polytope_manifest is None
                    else [os.path.abspath(args.polytope_manifest)]
                ),
                command_line=None,
                seed=args.seed,
            )
        except ProvenanceError as exc:
            parser.error(f"{exc.status}: {exc}")
    try:
        output_root = ensure_fresh_output_root(args.outdir)
    except FileExistsError as exc:
        parser.error(str(exc))

    batch_result = run_batches(
        h11_values,
        args.n,
        output_root,
        args.cores,
        args.seed,
        args.max_retries,
        args.max_tip_attempts,
        args.allow_overwrite_existing_geometry,
        args.max_m,
        args.max_kaehler_attempts,
        args.min_divisor_volume,
        args.min_prime_divisor_volume,
        args.qcd_volume_min,
        args.qcd_volume_max,
        args.moduli_policy,
        args.qcd_volume_target,
        args.qcd_divisor_index,
        args.visible_sector_policy,
        args.qed_divisor_index,
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
        args.ntfe_face_sampler,
        args.ntfe_max_face_points,
        args.ntfe_face_pool_size,
        args.ks_database_version,
        favorable,
        orientifold_config,
        args.export_kahler_rays,
        args.frsts_per_polytope,
        args.replace_rejected_polytopes,
        args.max_polytope_replacements,
        polytope_manifest,
        geometry_targets=eft_geometry_plan,
        qed_selection_policy=args.qed_selection_policy,
        qed_volume_max=args.qed_volume_max,
        eft_mode=args.eft,
        materialize_dense_potential=args.materialize_dense_potential,
        volume_backend=args.volume_backend,
        collect_records=True,
        proposal_budget=(None if args.proposal_budget is None else proposal_budget),
        retry_budget=(None if args.retry_budget is None else retry_budget),
    )
    candidate_records = []
    for result in batch_result["results"]:
        candidate_records.extend(result.get("candidate_terminal_records", []))
        if not result.get("ok", False):
            candidate_records.append(
                {
                    "h11": result.get("h11"),
                    "polytope_index": result.get("polytope_index"),
                    "sampler": args.sampling_scheme,
                    "terminal_status": "geometry_target_shortfall",
                    "terminal_reason": result.get("error", "worker failure"),
                }
            )
    accepted_paths = [
        record["output_path"]
        for record in candidate_records
        if record.get("terminal_status") == "accepted_geometry"
        and os.path.isfile(record.get("output_path", ""))
    ]
    model_records = []
    model_rows = []
    model_error = None
    allocation = None
    if args.eft:
        accepted_count = len(accepted_paths)
        if accepted_count != TARGET_GEOMETRY_COUNT:
            model_error = ModelTargetShortfall(
                f"geometry target shortfall: requested={TARGET_GEOMETRY_COUNT} emitted={accepted_count}",
                [
                    {
                        "terminal_status": "model_target_shortfall",
                        "terminal_reason": "fewer than 1400 accepted geometries",
                        "requested_geometries": TARGET_GEOMETRY_COUNT,
                        "accepted_geometries": accepted_count,
                    }
                ],
            )
            model_records.extend(model_error.records)
        else:
            try:
                model_rows, model_records, allocation = expand_eft_reference_rows(
                    accepted_paths,
                    args.seed,
                    args.eft_minimum_rows,
                    args.eft_maximum_rows,
                )
            except ModelTargetShortfall as exc:
                model_error = exc
                allocation = exc.allocation
                model_records.extend(exc.records)
            except Exception as exc:
                model_error = exc
                model_records.append(
                    {
                        "terminal_status": "model_assignment_rejected",
                        "terminal_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    storage_estimate = estimate_storage(
        output_root, len(model_rows) if args.eft else 0
    )
    if args.eft and storage_estimate["status"] != "within_budget" and model_error is None:
        model_error = RuntimeError("schema 1.1 persistent storage preflight exceeded 2 GiB")
        model_records.append(
            {
                "terminal_status": "storage_budget_exceeded",
                "terminal_reason": storage_estimate["status"],
            }
        )
    if args.eft and model_error is None:
        try:
            eft_path = args.eft_output_path or os.path.join(output_root, "eft_models.parquet")
            if not os.path.isabs(eft_path):
                eft_path = os.path.join(output_root, eft_path)
            if os.path.commonpath((os.path.abspath(eft_path), output_root)) != output_root:
                raise ValueError("--eft-output-path must be inside the fresh output root")
            write_eft_parquet(eft_path, model_rows)
            model_records.append(
                {
                    "terminal_status": "accepted_model_table",
                    "terminal_reason": "one compressed Parquet table written atomically",
                    "output_path": os.path.abspath(eft_path),
                    "row_count": len(model_rows),
                }
            )
        except Exception as exc:
            model_error = exc
            model_records.append(
                {
                    "terminal_status": "model_row_write_failure",
                    "terminal_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    charge_manifest = factorized_manifest_for_paths(accepted_paths)
    polytope_entries = {}
    for record in candidate_records:
        if "polytope_id" in record:
            key = (record.get("h11"), record.get("polytope_index"), record.get("polytope_id"))
            polytope_entries[key] = {
                "h11": record.get("h11"),
                "retained_order": record.get("polytope_index"),
                "polytope_id": record.get("polytope_id"),
                "polytope_fingerprint": record.get("polytope_id"),
            }
    retained_polytopes = sorted(
        polytope_entries.values(),
        key=lambda item: (
            int(item["h11"]),
            int(item["retained_order"]),
            str(item["polytope_fingerprint"]),
        ),
    )
    if not retained_polytopes and source_query is not None:
        retained_polytopes = [
            {
                "h11": item["h11"],
                "retained_order": item["returned_index"],
                "polytope_fingerprint": item["polytope_fingerprint"],
            }
            for item in source_query["returned_polytopes"]
        ]
    polytope_manifest = {
        "schema_version": SCHEMA_1_1_VERSION,
        "source": args.ks_database_version,
        "selection_route": "fresh favorable N-lattice query; deterministic returned order",
        "favorable": True,
        "lattice": "N",
        "fresh": True,
        "expected_counts": {"50": 50, "100": 50, "200": 30, "491": 1}
        if args.eft
        else None,
        "polytopes": retained_polytopes,
    }
    fresh_ensemble_manifest = None
    if args.eft:
        derived_seeds = derive_ensemble_seeds(
            args.seed, retained_polytopes, default_sampler_by_h11()
        )
        stop_reason = (
            allocation.get("stop_reason")
            if allocation is not None
            else "geometry_target_shortfall" if model_error is not None else "completed"
        )
        fresh_ensemble_manifest = build_fresh_ensemble_manifest(
            provenance_record,
            source_query=source_query,
            retained_polytopes=retained_polytopes,
            base_seed=args.seed,
            derived_seeds=derived_seeds,
            source_manifest=args.polytope_manifest,
            cytools_version=cytools.version,
            sampler_by_h11=default_sampler_by_h11(),
            accepted_row_count=len(model_rows),
            stop_reason=stop_reason,
            expected_polytope_counts={50: 50, 100: 50, 200: 30, 491: 1},
            run_metadata={
                "proposal_budget": proposal_budget,
                "retry_budget": retry_budget,
                "minimum_rows": args.eft_minimum_rows,
                "maximum_rows": args.eft_maximum_rows,
                "accepted_geometry_count": len(accepted_paths),
            },
        )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema_semantic_version": SCHEMA_1_1_VERSION,
        "run_id": stable_hash({"seed": args.seed, "output_root": output_root}),
        "status": "model_target_shortfall" if model_error else "completed_geometry_only" if not args.eft else "completed",
        "output_root": output_root,
        "provenance_digest": (
            None if provenance_record is None else provenance_record["provenance_digest"]
        ),
        "fresh_ensemble_manifest_status": (
            None if fresh_ensemble_manifest is None else fresh_ensemble_manifest["status"]
        ),
        "fresh_ensemble_manifest_path": (
            None
            if fresh_ensemble_manifest is None
            else os.path.join(output_root, "fresh_ensemble_manifest.json")
        ),
        "c4_basis_convention": "full_cytools_h11_declared_all_c4_assumption",
        "all_h11_c4_assumption": {
            "enabled": True,
            "assumed_h11_minus": 0,
            "status": "declared_modeling_assumption",
            "provenance": (
                "Paper-style full-CYTools-basis convention; not an inferred "
                "physical orientifold parity result."
            ),
        },
        "sampling_unit": "accepted geometry record; EFT row is an ordered QCD-QED assignment",
        "population_label": "adapted_fresh_favorable_filtered_geometry_reference",
        "paper_mapping_status": "adapted_model_reuse_not_exact_paper_multiplicity",
        "exact_paper_non_reproduction_boundary": (
            "This finite fresh-favorable sample and its compact model reuse do not "
            "reproduce the paper ensemble, its undocumented model weighting, or a "
            "uniform/representative KS population."
        ),
        "downstream_boundary": "No GNN, PyTorch, axion-photon, cosmology, or inflation analysis is run.",
        "args": {key: value for key, value in vars(args).items() if key != "h11s"},
        "h11_values": h11_values,
        "proposal_budget": proposal_budget,
        "retry_budget": retry_budget,
        "sampler_by_h11": (
            {"50": "fast", "100": "fast", "200": "fast", "491": "ntfe_fast"}
            if args.eft
            else "configured_per_run"
        ),
        "fresh_favorable_polytope_counts": (
            {"50": 50, "100": 50, "200": 30, "491": 1}
            if args.eft
            else "recorded_by_candidate_tasks"
        ),
        "triangulations_per_polytope": (
            {"50": 10, "100": 10, "200": 10, "491": 100}
            if args.eft
            else "configured_per_run"
        ),
        "accepted_geometry_count": len(accepted_paths),
        "candidate_proposal_count": sum(
            result.get("proposal_count", 0) for result in batch_result["results"]
        ),
        "candidate_retry_count": sum(
            result.get("retry_count", 0) for result in batch_result["results"]
        ),
        "duplicate_full_triangulation_count": sum(
            result.get("duplicate_full_triangulations", 0)
            for result in batch_result["results"]
        ),
        "duplicate_ntfe_identity_count": sum(
            result.get("duplicate_ntfe_identity", 0) for result in batch_result["results"]
        ),
        "output_collision_status": (
            "detected"
            if any(
                record.get("terminal_status") == "output_collision"
                for record in candidate_records
            )
            else "none_detected"
        ),
        "geometry_overwrite_event_count": sum(
            record.get("overwrite_event") == "replaced_existing_geometry"
            for record in candidate_records
        ),
        "allow_overwrite_existing_geometry": bool(
            args.allow_overwrite_existing_geometry
        ),
        "geometry_artifact_policy": {
            "geometry_only_status": GEOMETRY_ONLY_ARTIFACT_STATUS,
            "accepted_geometry_status": ACCEPTED_GEOMETRY_ARTIFACT_STATUS,
            "pool_pending_status": POOL_PENDING_ARTIFACT_STATUS,
            "overwrite_policy": "explicit_allow_overwrite_existing_geometry_only",
            "temporary_artifact_policy": "delete_after_status_recording",
        },
        "eft_allocation": allocation,
    }
    summary = summarize_terminal_records(candidate_records, model_records)
    write_schema11_artifacts(
        output_root,
        run_manifest=run_manifest,
        candidate_records=candidate_records,
        model_records=model_records,
        summary=summary,
        storage_estimate=storage_estimate,
        charge_factorized_manifest=charge_manifest,
        polytope_manifest=polytope_manifest,
        include_model_statuses=args.eft,
        fresh_ensemble_manifest=fresh_ensemble_manifest,
    )
    print(
        f"\nSchema 1.1 summary: saved {len(accepted_paths)} geometry file(s); "
        f"proposals={run_manifest['candidate_proposal_count']} "
        f"retries={run_manifest['candidate_retry_count']} "
        f"duplicates={run_manifest['duplicate_full_triangulation_count'] + run_manifest['duplicate_ntfe_identity_count']}."
    )
    if args.eft:
        if model_error is not None:
            print(f"EFT model target not completed: {model_error}")
            raise RuntimeError(str(model_error)) from model_error
        print(
            f"EFT reference rows: {len(model_rows)}; "
            f"minimum={args.eft_minimum_rows}, ceiling={args.eft_maximum_rows}."
        )


if __name__ == "__main__":
    main()
