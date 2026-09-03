"""Read and write the stage-1 raw-FRST interchange contract.

Keep this module independent of CYTools.  Stage 1 writes lattice data and
triangulation labels, and may additionally persist a validated,
stage-2-independent topology cache.  Stage 2 remains responsible for
reconstructing CYTools objects and applying all geometry and EFT cuts.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


RAW_FRST_SCHEMA_VERSION = "cyaxiverse-raw-frst-1.0"
LEGACY_RAW_FRST_SCHEMA_VERSION = "cyaxiverse-raw-frst-1.1"
TOPOLOGY_CACHE_SCHEMA_VERSION = "cyaxiverse-topology-cache-1.1"
TOPOLOGY_CACHE_GROUP = "topology_cache"
TOPOLOGY_CACHE_FIELDS = (
    "h11",
    "h21",
    "basis",
    "basis_matrix",
    "glsm",
    "prime_toric_divisors",
    "kappa",
    "c2",
    "mori_cone",
    "kahler_cone_hyperplanes",
    "face_restriction_dim2",
)
TOPOLOGY_CACHE_CONVENTIONS = {
    "basis": "CYTools divisor_basis(include_origin=True); all numerical vectors in basis",
    "basis_matrix": "CSR rows are the CYTools divisor_basis(as_matrix=True) rows",
    "glsm": "CYTools polytope.glsm_charge_matrix(include_origin=False), rows are H2 relations and columns are prime toric divisors",
    "kappa": "CYTools CalabiYau.intersection_numbers(in_basis=True, format='coo'), zero-based indices",
    "prime_toric_divisors": "CYTools prime_toric_divisors labels, zero-based lattice-point labels",
    "mori_cone": "CYTools toric_mori_cone(in_basis=True).rays(), rows are rays",
    "kahler_cone_hyperplanes": "CYTools toric_kahler_cone().hyperplanes(), rows are hyperplanes",
    "face_restriction_dim2": "CYTools triangulation.simplices(on_faces_dim=2), zero-based point labels",
}
RAW_FRST_ARTIFACT_STATUS = "retained"
RAW_FRST_TERMINAL_STATUSES = (
    "retained_raw_frst",
    "invalid_frst",
    "missing_raw_frst",
    "duplicate_full_triangulation",
    "input_identity_mismatch",
    "user_decision_required",
)


class RawFRSTError(RuntimeError):
    """Report a raw-FRST input failure with its stable stage-2 status."""

    def __init__(self, terminal_status, reason, *, record=None):
        if terminal_status not in RAW_FRST_TERMINAL_STATUSES[1:]:
            raise ValueError(f"unknown raw-FRST status {terminal_status!r}")
        super().__init__(str(reason))
        self.terminal_status = terminal_status
        self.reason = str(reason)
        self.record = {} if record is None else dict(record)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def stable_hash(value):
    """Return a deterministic digest for JSON-compatible identity data."""
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_rows(rows, *, name):
    array = np.asarray(rows)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    return sorted(tuple(int(value) for value in row) for row in array.tolist())


def compute_polytope_id(polytope_points):
    """Return the canonical lattice-point identity for one raw FRST."""
    points = _canonical_rows(polytope_points, name="polytope_points")
    return f"lattice-points-sha256:{stable_hash(points)}"


def compute_polytope_normal_form_id(normal_form_points):
    """Return the lattice-invariant geometry identity of a polytope.

    ``normal_form_points`` are the polytope's affine normal form, for example
    ``cytools.Polytope.normal_form()``: the canonical representative under
    ``GL(n, Z)`` and lattice translation.  Two reflexive polytopes describe the
    same geometry exactly when their normal forms agree, so this identity is
    stable across polytope presentations.  Prefer it over ``compute_polytope_id``
    (which hashes the raw lattice points as supplied, so lattice-equivalent
    presentations differ) and over ``compute_triangulation_hash`` (whose simplex
    indices can coincide for lattice-inequivalent polytopes, so distinct
    geometries collide).
    """
    points = _canonical_rows(normal_form_points, name="normal_form_points")
    return f"normal-form-sha256:{stable_hash(points)}"


def compute_triangulation_hash(simplices):
    """Return the order-independent identity of the full triangulation.

    This hashes the simplices as supplied (point indices into the polytope's
    own point list), so it is a combinatorial fingerprint of one triangulation
    presentation, not a geometry identity: lattice-inequivalent polytopes with
    the same index combinatorics collide.  Use ``compute_polytope_normal_form_id``
    for a geometry-unique identity.
    """
    return stable_hash(_canonical_rows(simplices, name="simplices"))


def build_raw_frst_geometry_id(h11, polytope_identifier, full_triangulation_hash):
    """Build the stable identity of a retained raw FRST."""
    return (
        f"raw-frst:h11-{int(h11):03d}:{polytope_identifier}:"
        f"frst-{full_triangulation_hash}"
    )


def _atomic_hdf5_write(path, write_payload):
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"output collision: {destination}")
    temporary = destination.with_name(
        f"{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with h5py.File(temporary, "w") as handle:
            write_payload(handle)
            handle.flush()
        # Linking the completed temporary file is atomic and refuses to replace
        # a destination that appeared concurrently.
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _compressed_dataset(group, name, data):
    """Write one cache dataset with lossless compression and shuffling."""
    return group.create_dataset(
        name,
        data=np.asarray(data),
        compression="gzip",
        compression_opts=9,
        shuffle=True,
    )


def _csr_components(matrix):
    """Return lossless CSR components for a two-dimensional numeric matrix."""
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("basis_matrix must be two-dimensional")
    if values.dtype.kind not in "biufc":
        raise ValueError("basis_matrix must contain HDF5-compatible numeric values")
    row_indices, column_indices = np.nonzero(values)
    indptr = np.zeros(values.shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, row_indices + 1, 1)
    np.cumsum(indptr, out=indptr)
    return values[row_indices, column_indices], column_indices.astype(np.int64), indptr


def _write_topology_cache(group, topology, cache_metadata):
    """Serialize the stage-2-independent topology payload into an HDF5 group."""
    missing = [name for name in TOPOLOGY_CACHE_FIELDS if name not in topology]
    if missing:
        raise ValueError(f"topology cache is missing fields: {missing}")
    cache = group.create_group(TOPOLOGY_CACHE_GROUP)
    cache.attrs["schema_version"] = TOPOLOGY_CACHE_SCHEMA_VERSION
    cache.attrs["cache_metadata_json"] = json.dumps(
        _jsonable(cache_metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    cache.attrs["compression"] = "gzip"
    cache.attrs["compression_opts"] = 9
    cache.attrs["shuffle"] = True
    cache.attrs["index_base"] = 0
    _compressed_dataset(cache, "h11", np.asarray([int(topology["h11"])], dtype=np.int64))
    _compressed_dataset(cache, "h21", np.asarray([int(topology["h21"])], dtype=np.int64))
    _compressed_dataset(cache, "basis", np.asarray(topology["basis"]))
    basis_data, basis_indices, basis_indptr = _csr_components(topology["basis_matrix"])
    basis_matrix = cache.create_group("basis_matrix")
    basis_matrix.attrs["format"] = "csr"
    basis_matrix.attrs["index_base"] = 0
    _compressed_dataset(basis_matrix, "data", basis_data)
    _compressed_dataset(basis_matrix, "indices", basis_indices)
    _compressed_dataset(basis_matrix, "indptr", basis_indptr)
    _compressed_dataset(
        basis_matrix, "shape", np.asarray(np.asarray(topology["basis_matrix"]).shape, dtype=np.int64)
    )
    _compressed_dataset(
        cache, "prime_toric_divisors", np.asarray(topology["prime_toric_divisors"])
    )
    glsm = np.asarray(topology["glsm"])
    if glsm.ndim != 2 or glsm.shape[0] != int(topology["h11"]):
        raise ValueError("glsm must have shape (h11, n_prime_divisors)")
    _compressed_dataset(cache, "glsm", glsm.astype(np.int64, copy=False))

    kappa = np.asarray(topology["kappa"])
    if kappa.ndim != 2 or kappa.shape[1] != 4:
        raise ValueError("kappa must have shape (n, 4) in COO row format")
    kappa_indices = np.asarray(kappa[:, :3])
    if not np.all(np.isfinite(kappa_indices)) or not np.all(kappa_indices == np.floor(kappa_indices)):
        raise ValueError("kappa indices must be finite integers")
    kappa_group = cache.create_group("kappa")
    kappa_group.attrs["format"] = "coo"
    kappa_group.attrs["index_base"] = 0
    _compressed_dataset(kappa_group, "indices", kappa_indices.astype(np.int64))
    _compressed_dataset(kappa_group, "values", np.asarray(kappa[:, 3]))
    _compressed_dataset(
        kappa_group,
        "shape",
        np.asarray([int(topology["h11"])] * 3, dtype=np.int64),
    )
    for name in (
        "c2",
        "mori_cone",
        "kahler_cone_hyperplanes",
        "face_restriction_dim2",
    ):
        _compressed_dataset(cache, name, np.asarray(topology[name]))
    return cache


def _read_topology_cache(handle):
    """Read one topology cache group into the generator's topology mapping."""
    if TOPOLOGY_CACHE_GROUP not in handle:
        return None
    cache = handle[TOPOLOGY_CACHE_GROUP]
    schema_version = cache.attrs.get("schema_version")
    if isinstance(schema_version, bytes):
        schema_version = schema_version.decode("utf-8")
    if schema_version != TOPOLOGY_CACHE_SCHEMA_VERSION:
        return {
            "schema_version": schema_version,
            "metadata": {},
            "payload": None,
            "status": "schema_mismatch",
            "reason": f"unsupported topology cache schema {schema_version!r}",
        }
    metadata_json = cache.attrs.get("cache_metadata_json", "{}")
    if isinstance(metadata_json, bytes):
        metadata_json = metadata_json.decode("utf-8")
    try:
        metadata = json.loads(str(metadata_json))
    except (TypeError, ValueError) as exc:
        return {
            "schema_version": schema_version,
            "metadata": {},
            "payload": None,
            "status": "invalid",
            "reason": f"cache metadata is not valid JSON: {exc}",
        }
    if not isinstance(metadata, dict):
        return {
            "schema_version": schema_version,
            "metadata": {},
            "payload": None,
            "status": "invalid",
            "reason": "cache metadata must be a JSON object",
        }
    try:
        h11 = cache["h11"][()]
        h21 = cache["h21"][()]
        if h11.shape != (1,) or h21.shape != (1,):
            raise ValueError("topology cache Hodge-number datasets must have shape (1,)")
        h11 = int(h11[0])
        h21 = int(h21[0])
        if h11 < 1 or h21 < 0:
            raise ValueError("topology cache Hodge numbers are out of range")
        basis_matrix_group = cache["basis_matrix"]
        basis_shape_values = basis_matrix_group["shape"][()]
        if basis_shape_values.shape != (2,):
            raise ValueError("basis_matrix CSR shape must have two entries")
        basis_shape = tuple(int(value) for value in basis_shape_values.tolist())
        if basis_shape[0] != h11:
            raise ValueError("basis_matrix row count does not match h11")
        basis_matrix = np.zeros(basis_shape, dtype=basis_matrix_group["data"].dtype)
        indptr = basis_matrix_group["indptr"][()].astype(np.int64, copy=False)
        indices = basis_matrix_group["indices"][()].astype(np.int64, copy=False)
        data = basis_matrix_group["data"][()]
        if (
            len(basis_shape) != 2
            or any(value < 0 for value in basis_shape)
            or indptr.size != basis_shape[0] + 1
            or indptr[-1] != data.size
            or indices.size != data.size
            or np.any(indptr[1:] < indptr[:-1])
            or (indices.size and (np.min(indices) < 0 or np.max(indices) >= basis_shape[1]))
        ):
            raise ValueError("basis_matrix CSR components have inconsistent lengths")
        for row in range(basis_shape[0]):
            basis_matrix[row, indices[indptr[row] : indptr[row + 1]]] = data[
                indptr[row] : indptr[row + 1]
            ]
        basis = cache["basis"][()]
        if basis.ndim not in (1, 2) or basis.shape[0] != h11:
            raise ValueError("divisor basis shape does not match h11")
        prime_toric_divisors = cache["prime_toric_divisors"][()]
        if prime_toric_divisors.ndim != 1 or (
            prime_toric_divisors.size
            and (
                np.min(prime_toric_divisors) < 0
                or np.max(prime_toric_divisors) >= basis_shape[1]
            )
        ):
            raise ValueError("prime toric divisor labels do not fit basis_matrix")
        glsm = cache["glsm"][()].astype(np.int64, copy=False)
        if glsm.ndim != 2 or glsm.shape != (h11, prime_toric_divisors.size):
            raise ValueError("glsm shape does not match h11 and prime toric divisors")
        kappa_group = cache["kappa"]
        kappa_shape = kappa_group["shape"][()]
        if kappa_shape.shape != (3,) or not np.all(kappa_shape == h11):
            raise ValueError("kappa shape does not match h11")
        kappa_indices = kappa_group["indices"][()].astype(np.int64, copy=False)
        kappa_values = kappa_group["values"][()]
        if (
            kappa_indices.ndim != 2
            or kappa_indices.shape[1] != 3
            or kappa_values.ndim != 1
            or kappa_indices.shape[0] != kappa_values.size
        ):
            raise ValueError("kappa COO components have inconsistent shapes")
        kappa = np.column_stack((kappa_indices, kappa_values))
        c2 = cache["c2"][()]
        mori_cone = cache["mori_cone"][()]
        kahler_cone_hyperplanes = cache["kahler_cone_hyperplanes"][()]
        if c2.shape != (h11,):
            raise ValueError("c2 shape does not match h11")
        if mori_cone.ndim != 2 or mori_cone.shape[1] != h11:
            raise ValueError("Mori-cone shape does not match h11")
        if (
            kahler_cone_hyperplanes.ndim != 2
            or kahler_cone_hyperplanes.shape[1] != h11
        ):
            raise ValueError("Kahler-hyperplane shape does not match h11")
        payload = {
            "h11": h11,
            "h21": h21,
            "basis": basis,
            "basis_matrix": basis_matrix,
            "glsm": glsm,
            "prime_toric_divisors": prime_toric_divisors,
            "kappa": kappa,
            "c2": c2,
            "mori_cone": mori_cone,
            "kahler_cone_rays": None,
            "kahler_cone_hyperplanes": kahler_cone_hyperplanes,
            "face_restriction_dim2": cache["face_restriction_dim2"][()],
        }
    except (IndexError, KeyError, TypeError, ValueError, OSError) as exc:
        return {
            "schema_version": schema_version,
            "metadata": metadata,
            "payload": None,
            "status": "invalid",
            "reason": f"cache payload is invalid: {exc}",
        }
    return {
        "schema_version": schema_version,
        "metadata": metadata,
        "payload": payload,
        "status": "available",
        "reason": "validated by HDF5 codec",
    }


def validate_topology_cache(cache, expected):
    """Validate cache identity and return its topology or a fallback reason."""
    if cache is None:
        return None, "missing topology cache group"
    if cache.get("status") != "available":
        return None, str(cache.get("reason", "topology cache is unavailable"))
    metadata = cache.get("metadata") or {}
    mismatches = {}
    for key in expected:
        recorded = metadata.get(key)
        wanted = expected.get(key)
        equal = (
            stable_hash(recorded) == stable_hash(wanted)
            if isinstance(recorded, (dict, list, tuple))
            or isinstance(wanted, (dict, list, tuple))
            else str(recorded) == str(wanted)
        )
        if not equal:
            mismatches[key] = {"expected": wanted, "recorded": recorded}
    if mismatches:
        return None, f"topology cache identity mismatch: {mismatches}"
    return cache["payload"], "cache identity validated"


def write_raw_frst_artifact(
    path,
    *,
    h11,
    polytope_vertices,
    polytope_points,
    triangulation_labels,
    triangulation_points,
    simplices,
    simplex_indices=None,
    metadata=None,
    topology_cache=None,
    topology_cache_metadata=None,
):
    """Write one retained FRST and an optional stage-2-independent topology cache."""
    vertices = np.asarray(polytope_vertices, dtype=int)
    points = np.asarray(polytope_points, dtype=int)
    labels = np.asarray(triangulation_labels, dtype=int).reshape(-1)
    tri_points = np.asarray(triangulation_points, dtype=int)
    raw_simplices = np.asarray(simplices, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 4:
        raise ValueError("polytope_vertices must have shape (n, 4)")
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("polytope_points must have shape (n, 4)")
    if tri_points.ndim != 2 or tri_points.shape[1] != 4:
        raise ValueError("triangulation_points must have shape (n, 4)")
    if raw_simplices.ndim != 2 or raw_simplices.shape[1] != 5:
        raise ValueError("simplices must have shape (n, 5) for a four-dimensional FRST")
    if labels.size != tri_points.shape[0]:
        raise ValueError("triangulation_labels and triangulation_points disagree")
    if simplex_indices is None:
        simplex_indices = np.empty((0, 5), dtype=int)
    simplex_indices = np.asarray(simplex_indices, dtype=int)
    if simplex_indices.size and simplex_indices.shape != raw_simplices.shape:
        raise ValueError("simplex_indices must have the same shape as simplices")

    identifier = compute_polytope_id(points)
    full_hash = compute_triangulation_hash(raw_simplices)
    raw_geometry_id = build_raw_frst_geometry_id(h11, identifier, full_hash)
    record = dict(metadata or {})
    record.update(
        {
            "raw_frst_schema_version": RAW_FRST_SCHEMA_VERSION,
            "stage1_status": RAW_FRST_ARTIFACT_STATUS,
            "h11": int(h11),
            "polytope_id": identifier,
            "polytope_id_kind": "canonical_lattice_point_sha256",
            "full_triangulation_hash": full_hash,
            "geometry_id": raw_geometry_id,
            "triangulation_label_order": "CYTools triangulation labels",
            "simplices_convention": "full FRST simplex labels, sorted within simplex",
        }
    )
    record.setdefault("topology_cache_status", "not_requested")
    record.setdefault("topology_cache_reason", "stage-1 topology cache was not requested")
    if topology_cache is not None:
        record["topology_cache_status"] = "available"
        record["topology_cache_reason"] = "stage-1 topology cache computed from held CYTools objects"

    def write_payload(handle):
        handle.attrs["raw_frst_schema_version"] = RAW_FRST_SCHEMA_VERSION
        handle.attrs["stage1_status"] = RAW_FRST_ARTIFACT_STATUS
        handle.attrs["metadata_json"] = json.dumps(
            _jsonable(record), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        handle.create_dataset("polytope_vertices", data=vertices, compression="gzip", compression_opts=9)
        handle.create_dataset("polytope_points", data=points, compression="gzip", compression_opts=9)
        handle.create_dataset("triangulation_labels", data=labels, compression="gzip", compression_opts=9)
        handle.create_dataset("triangulation_points", data=tri_points, compression="gzip", compression_opts=9)
        handle.create_dataset("simplices", data=raw_simplices, compression="gzip", compression_opts=9)
        handle.create_dataset("simplex_indices", data=simplex_indices, compression="gzip", compression_opts=9)
        if topology_cache is not None:
            try:
                _write_topology_cache(
                    handle,
                    topology_cache,
                    topology_cache_metadata or {},
                )
            except Exception as exc:
                # The raw FRST is the stage-1 retention unit.  A cache codec
                # failure must not turn a valid raw FRST into a rejection.
                cache = handle.get(TOPOLOGY_CACHE_GROUP)
                if cache is not None:
                    del handle[TOPOLOGY_CACHE_GROUP]
                record["topology_cache_status"] = "write_failed"
                record["topology_cache_reason"] = f"{type(exc).__name__}: {exc}"
                handle.attrs["metadata_json"] = json.dumps(
                    _jsonable(record), sort_keys=True, separators=(",", ":"), allow_nan=False
                )

    _atomic_hdf5_write(path, write_payload)
    record["raw_frst_path"] = str(Path(path).resolve())
    record["file_size_bytes"] = Path(path).stat().st_size
    return record


def _read_metadata(handle):
    raw = handle.attrs.get("metadata_json")
    if raw is None:
        raise RawFRSTError("missing_raw_frst", "raw FRST metadata_json is absent")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise RawFRSTError("invalid_frst", f"raw FRST metadata is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RawFRSTError("invalid_frst", "raw FRST metadata must be an object")
    return metadata


def read_raw_frst_artifact(path, *, include_topology_cache=True):
    """Read one raw FRST, optionally decoding its topology cache."""
    resolved = Path(path).resolve()
    legacy_schema = False
    if not resolved.is_file():
        raise RawFRSTError("missing_raw_frst", f"raw FRST file does not exist: {resolved}")
    try:
        with h5py.File(resolved, "r") as handle:
            metadata = _read_metadata(handle)
            if handle.attrs.get("raw_frst_schema_version") == RAW_FRST_SCHEMA_VERSION:
                required = ("h11", "polytope_id", "full_triangulation_hash", "geometry_id")
                missing = [field for field in required if metadata.get(field) in (None, "")]
                if missing:
                    raise RawFRSTError("input_identity_mismatch", f"raw FRST identity is missing: {missing}")
                if metadata.get("stage1_status") != RAW_FRST_ARTIFACT_STATUS:
                    raise RawFRSTError("invalid_frst", "raw FRST was not retained by stage 1")
                datasets = (
                    "polytope_vertices",
                    "polytope_points",
                    "triangulation_labels",
                    "triangulation_points",
                    "simplices",
                    "simplex_indices",
                )
                if any(name not in handle for name in datasets):
                    missing = [name for name in datasets if name not in handle]
                    raise RawFRSTError("missing_raw_frst", f"raw FRST datasets are missing: {missing}")
                arrays = {name: handle[name][()] for name in datasets}
                topology_cache = (
                    _read_topology_cache(handle) if include_topology_cache else None
                )
            elif handle.attrs.get("schema_version") == LEGACY_RAW_FRST_SCHEMA_VERSION and "frst" in handle:
                # The original Stage-1 collector stored the same serializable
                # triangulation under a ``frst`` group and omitted the full
                # lattice-point and geometry-id datasets.  Keep this reader
                # compatible without rewriting those append-only artifacts.
                legacy_schema = True
                group = handle["frst"]
                datasets = ("polytope_vertices", "triangulation_labels", "simplices")
                if any(name not in group for name in datasets):
                    missing = [name for name in datasets if name not in group]
                    raise RawFRSTError("missing_raw_frst", f"legacy raw FRST datasets are missing: {missing}")
                vertices = group["polytope_vertices"][()]
                simplices = group["simplices"][()]
                arrays = {
                    "polytope_vertices": vertices,
                    # Legacy files do not persist all lattice points.  Stage 2
                    # reconstructs them from the vertices before using the
                    # triangulation; this placeholder preserves the array
                    # contract for dependency-free ledger validation.
                    "polytope_points": vertices.copy(),
                    "triangulation_labels": group["triangulation_labels"][()],
                    "triangulation_points": np.empty((0, 4), dtype=int),
                    "simplices": simplices,
                    "simplex_indices": simplices.copy(),
                }
                topology_cache = None
                metadata = dict(metadata)
                metadata.setdefault("stage1_status", RAW_FRST_ARTIFACT_STATUS)
                metadata.setdefault("raw_frst_schema_version", LEGACY_RAW_FRST_SCHEMA_VERSION)
                required = ("h11", "polytope_id", "full_triangulation_hash")
                missing = [field for field in required if metadata.get(field) in (None, "")]
                if missing:
                    raise RawFRSTError("input_identity_mismatch", f"legacy raw FRST identity is missing: {missing}")
            else:
                raise RawFRSTError("invalid_frst", "unsupported raw FRST schema version")
    except RawFRSTError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise RawFRSTError("invalid_frst", f"could not read raw FRST {resolved}: {exc}") from exc

    actual_full_hash = compute_triangulation_hash(arrays["simplices"])
    if legacy_schema:
        actual_polytope_id = str(metadata["polytope_id"])
        actual_geometry_id = metadata.get("geometry_id") or build_raw_frst_geometry_id(
            metadata["h11"], actual_polytope_id, actual_full_hash
        )
    else:
        actual_polytope_id = compute_polytope_id(arrays["polytope_points"])
        actual_geometry_id = build_raw_frst_geometry_id(
            metadata["h11"], actual_polytope_id, actual_full_hash
        )
    if legacy_schema:
        metadata.setdefault("geometry_id", actual_geometry_id)
    expected = {
        "polytope_id": actual_polytope_id,
        "full_triangulation_hash": actual_full_hash,
        "geometry_id": actual_geometry_id,
    }
    mismatches = {
        key: {"recorded": metadata.get(key), "reconstructed": value}
        for key, value in expected.items()
        if str(metadata.get(key)) != str(value)
    }
    if mismatches:
        raise RawFRSTError("input_identity_mismatch", "raw FRST identity does not match its datasets", record=mismatches)
    metadata = dict(metadata)
    metadata["raw_frst_schema_version"] = metadata.get(
        "raw_frst_schema_version",
        LEGACY_RAW_FRST_SCHEMA_VERSION if legacy_schema else RAW_FRST_SCHEMA_VERSION,
    )
    metadata["stage1_status"] = metadata.get("stage1_status", RAW_FRST_ARTIFACT_STATUS)
    metadata["raw_frst_path"] = str(resolved)
    metadata["raw_frst_file_sha256"] = file_sha256(resolved)
    metadata["arrays"] = arrays
    metadata["topology_cache"] = topology_cache
    return metadata


def file_sha256(path, *, chunk_size=1024 * 1024):
    """Hash one persisted input for the stage-2 ledger."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_raw_frst_paths(stage1_root):
    """Return raw FRST paths in deterministic stage-1 order."""
    root = Path(stage1_root).resolve() / "frst_candidates"
    return sorted(root.glob("h11_*/np_*/frst_*.h5"), key=lambda path: str(path))


def build_input_ledger(stage1_root):
    """Validate every discovered raw FRST and retain one terminal record each."""
    records = []
    seen_triangulation_identities = {}
    stage1_root = Path(stage1_root).resolve()
    listed_paths = {}
    stage1_status_path = stage1_root / "frst_terminal_statuses.jsonl"
    if stage1_status_path.is_file():
        with stage1_status_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                status_record = json.loads(line)
                if status_record.get("terminal_status") == "retained_raw_frst":
                    raw_path = status_record.get("raw_frst_path")
                    if raw_path:
                        listed_paths[str(Path(raw_path).resolve())] = status_record

    discovered_paths = {
        str(path.resolve()): path for path in discover_raw_frst_paths(stage1_root)
    }
    candidate_paths = {
        **discovered_paths,
        **{path: Path(path) for path in listed_paths},
    }
    for path_string in sorted(candidate_paths):
        path = candidate_paths[path_string]
        base = dict(listed_paths.get(path_string, {}))
        base["raw_frst_path"] = str(path.resolve())
        try:
            metadata = read_raw_frst_artifact(path, include_topology_cache=False)
            if listed_paths and path_string not in listed_paths:
                raise RawFRSTError(
                    "input_identity_mismatch",
                    "raw FRST is not listed as retained by the stage-1 status ledger",
                )
            identity_mismatches = {
                field: {
                    "ledger": base.get(field),
                    "raw_artifact": metadata.get(field),
                }
                for field in (
                    "h11",
                    "polytope_id",
                    "full_triangulation_hash",
                    "geometry_id",
                )
                if field in base
                and base.get(field) not in (None, "")
                and str(base.get(field)) != str(metadata.get(field))
            }
            if identity_mismatches:
                raise RawFRSTError(
                    "input_identity_mismatch",
                    "stage-1 ledger identity does not match the raw FRST artifact",
                    record={"identity_mismatches": identity_mismatches},
                )
            record = {
                key: value
                for key, value in metadata.items()
                if key not in {"arrays", "topology_cache"}
            }
            record.update(
                {
                    "stage2_input_status": "retained_raw_frst",
                    "terminal_status": "retained_raw_frst",
                    "terminal_reason": "raw FRST identity and datasets validated",
                }
            )
            triangulation_identity = (
                record["polytope_id"],
                record["full_triangulation_hash"],
            )
            if triangulation_identity in seen_triangulation_identities:
                record.update(
                    {
                        "stage2_input_status": "duplicate_full_triangulation",
                        "terminal_status": "duplicate_full_triangulation",
                        "terminal_reason": (
                            "duplicate of "
                            f"{seen_triangulation_identities[triangulation_identity]}"
                        ),
                    }
                )
            else:
                seen_triangulation_identities[triangulation_identity] = record[
                    "raw_frst_path"
                ]
            records.append(record)
        except RawFRSTError as exc:
            record = dict(base)
            record.update(exc.record)
            record.update(
                {
                    "stage2_input_status": exc.terminal_status,
                    "terminal_status": exc.terminal_status,
                    "terminal_reason": exc.reason,
                }
            )
            records.append(record)
    return records


def count_by_h11(records, *, status_key="stage2_input_status"):
    """Count records by h11 and the selected status field."""
    counts = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[str(record.get("h11", "unknown"))][record.get(status_key, "unknown")] += 1
    return {h11: dict(sorted(values.items())) for h11, values in sorted(counts.items())}
