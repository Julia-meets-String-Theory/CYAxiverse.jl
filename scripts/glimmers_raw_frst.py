"""Read and write the stage-1 raw-FRST interchange contract.

Keep this module independent of CYTools.  Stage 1 writes lattice data and
triangulation labels; stage 2 is responsible for reconstructing CYTools
objects and applying all geometry and EFT cuts.
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


def compute_triangulation_hash(simplices):
    """Return the order-independent identity of the full triangulation."""
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
):
    """Write one retained FRST without topology or stage-2 derived data."""
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


def read_raw_frst_artifact(path):
    """Read and validate one raw FRST artifact without reconstructing CYTools."""
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
            metadata = read_raw_frst_artifact(path)
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
            record = {key: value for key, value in metadata.items() if key != "arrays"}
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
