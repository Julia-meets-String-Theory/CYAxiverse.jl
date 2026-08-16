"""Geometry-derived toy QED divisor selection for the CYTools generator.

This module deliberately has no CYTools dependency.  The generator supplies
the stable prime-divisor labels, divisor-basis charges, two-face intersection
evidence, volumes, and validated orientifold image map.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import hashlib
import json

import numpy as np


TERMINAL_FAILURE_CATEGORIES = (
    "qcd_normalization_failure",
    "qcd_qed_prefilter_shortfall",
    "assignment_pool_shortfall",
    "no_eligible_qed_divisor",
    "invalid_explicit_index",
    "orientifold_invariance_failure",
    "intersection_failure",
    "qed_volume_rejection",
    "invalid_charge_basis_mapping",
    "potential_term_mismatch",
    "numerical_geometry_failure",
    "output_collision",
    "user_decision_required",
    "accepted_assignment",
)

QCD_VOLUME_TARGET = 40.0
QCD_VOLUME_TOLERANCE = 1e-9
DIVISOR_VOLUME_TOLERANCE = 1e-8
QED_VOLUME_MAX = 127.5
NORMALIZATION_MAP_VERSION = "homogeneous-qcd-volume-40-v1"


class AssignmentPool(list):
    """List-compatible accepted assignments with explicit terminal accounting."""

    def __init__(self, assignments, terminal_records, pool_hash):
        super().__init__(assignments)
        self.terminal_records = list(terminal_records)
        self.pool_hash = str(pool_hash)
        self.rejection_records = [
            record
            for record in self.terminal_records
            if record.get("terminal_status") != "accepted_assignment"
        ]
        self.rejection_summary = summarize_assignment_pool_rejections(
            self.rejection_records
        )

    def serializable(self):
        """Return accepted rows and aggregate rejection data as JSON-compatible data."""
        return {
            "accepted_assignments": list(self),
            "rejection_summary": dict(self.rejection_summary),
            "pool_hash": self.pool_hash,
            "pool_status": "complete_eligible_ordered_pool",
        }


class QEDAssignmentFailure(RuntimeError):
    """A visible-sector outcome with an explicit terminal category."""

    def __init__(self, category, reason, record=None):
        if category not in TERMINAL_FAILURE_CATEGORIES:
            raise ValueError(f"unknown QED assignment failure category {category!r}")
        super().__init__(reason)
        self.category = category
        self.reason = str(reason)
        self.record = {} if record is None else record


def summarize_assignment_pool_rejections(records):
    """Aggregate assignment-pool rejection statuses and reasons for HDF5 metadata."""
    status_counts = Counter()
    reason_counts = Counter()
    for record in records:
        if record.get("terminal_status") == "accepted_assignment":
            continue
        status = str(record.get("terminal_status", "unknown"))
        reason = str(record.get("terminal_reason", "unknown"))
        status_counts[status] += 1
        reason_counts[reason] += 1
    return {
        "total_rejections": int(sum(status_counts.values())),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def validate_assignment_pool(pool):
    """Validate non-empty ordered-pool integrity before geometry acceptance."""
    if not isinstance(pool, (list, AssignmentPool)) or not pool:
        raise QEDAssignmentFailure(
            "assignment_pool_shortfall",
            "the complete eligible ordered assignment pool is empty",
        )
    required = {
        "assignment_hash",
        "pool_rank",
        "qcd_divisor_index",
        "qed_divisor_index",
        "qcd_divisor_label",
        "qed_divisor_label",
        "qcd_volume",
        "qed_volume",
        "terminal_status",
    }
    pairs = []
    hashes = []
    ordering = []
    for assignment in pool:
        missing = sorted(required.difference(assignment))
        if missing:
            raise QEDAssignmentFailure(
                "assignment_pool_shortfall",
                "assignment pool entry is missing required fields: "
                + ", ".join(missing),
            )
        if assignment["terminal_status"] != "accepted_assignment":
            raise QEDAssignmentFailure(
                "assignment_pool_shortfall",
                "assignment pool contains a non-accepted entry",
            )
        pair = (
            int(assignment["qcd_divisor_index"]),
            int(assignment["qed_divisor_index"]),
        )
        if pair in pairs:
            raise QEDAssignmentFailure(
                "assignment_pool_shortfall",
                f"assignment pool contains duplicate ordered pair {pair}",
            )
        pairs.append(pair)
        hashes.append(str(assignment["assignment_hash"]))
        ordering.append(
            (
                tuple(int(value) for value in assignment["qcd_divisor_label"]),
                tuple(int(value) for value in assignment["qed_divisor_label"]),
                pair[0],
                pair[1],
            )
        )
    if [int(assignment["pool_rank"]) for assignment in pool] != list(range(len(pool))):
        raise QEDAssignmentFailure(
            "assignment_pool_shortfall",
            "assignment pool ranks are not contiguous and zero-based",
        )
    if len(set(hashes)) != len(hashes):
        raise QEDAssignmentFailure(
            "assignment_pool_shortfall",
            "assignment pool hashes are not unique",
        )
    if ordering != sorted(ordering):
        raise QEDAssignmentFailure(
            "assignment_pool_shortfall",
            "assignment pool ordering is not deterministic",
        )
    expected_hash = stable_hash(list(pool))
    if isinstance(pool, AssignmentPool) and pool.pool_hash != expected_hash:
        raise QEDAssignmentFailure(
            "assignment_pool_shortfall",
            "assignment pool hash does not match its accepted entries",
        )
    return {
        "pool_status": "complete_eligible_ordered_pool",
        "pool_size": len(pool),
        "pool_hash": expected_hash,
    }


def _integer_array(values, name):
    array = np.asarray(values)
    if array.ndim == 0 or not np.all(np.isfinite(array)):
        raise QEDAssignmentFailure("invalid_charge_basis_mapping", f"{name} is not finite")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise QEDAssignmentFailure("invalid_charge_basis_mapping", f"{name} must contain integers")
    return rounded.astype(np.int64, copy=False)


def charge_hash(charge):
    vector = _integer_array(np.asarray(charge).reshape(-1), "charge")
    return hashlib.sha256(
        json.dumps(vector.tolist(), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_hash(value):
    """Return a deterministic digest for JSON-compatible assignment data."""

    def jsonable(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            return item.item()
        if isinstance(item, tuple):
            return [jsonable(value) for value in item]
        if isinstance(item, list):
            return [jsonable(value) for value in item]
        if isinstance(item, dict):
            return {str(key): jsonable(value) for key, value in item.items()}
        return item

    encoded = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_divisor_labels(prime_toric_divisors, lattice_points):
    """Return prime-divisor labels in CYTools order as lattice coordinates."""
    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    points = np.asarray(lattice_points, dtype=int)
    if labels.size == 0 or np.unique(labels).size != labels.size:
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "prime-divisor labels must be unique and non-empty"
        )
    if points.ndim != 2 or points.shape[1] != 4:
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "lattice points must be an n by 4 array"
        )
    if np.any(labels < 0) or np.any(labels >= len(points)):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "prime-divisor label is outside the lattice-point table"
        )
    result = tuple(tuple(int(value) for value in points[label]) for label in labels)
    if len(set(result)) != len(result):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "prime-divisor lattice labels are not unique"
        )
    return result


def prime_divisor_charges(basis_matrix, prime_toric_divisors):
    """Extract exact integer charges from the CYTools divisor basis."""
    basis = _integer_array(basis_matrix, "divisor basis matrix")
    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    if basis.ndim != 2 or labels.size == 0 or np.any(labels < 0) or np.any(labels >= basis.shape[1]):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "the divisor basis cannot represent prime divisors"
        )
    return np.asarray(basis[:, labels].T, dtype=np.int64)


def prime_divisor_intersection_graph(prime_toric_divisors, face_simplices):
    """Return neighbors and the triangulated two-face evidence for each edge."""
    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    positions = {int(label): index for index, label in enumerate(labels)}
    simplices = np.asarray(face_simplices, dtype=int)
    if simplices.size == 0:
        return tuple(() for _ in labels), {}
    if simplices.ndim != 2:
        raise QEDAssignmentFailure("intersection_failure", "two-face simplices must be two-dimensional")
    neighbors = [set() for _ in labels]
    evidence = {}
    for simplex in simplices:
        face_labels = tuple(sorted({int(label) for label in simplex if int(label) in positions}))
        face_positions = tuple(sorted(positions[label] for label in face_labels))
        for left in range(len(face_positions)):
            for right in range(left + 1, len(face_positions)):
                first, second = face_positions[left], face_positions[right]
                neighbors[first].add(second)
                neighbors[second].add(first)
                evidence.setdefault((first, second), []).append(face_labels)
    return tuple(tuple(sorted(values)) for values in neighbors), evidence


def _orientifold_mask(orientifold, divisor_count):
    if not orientifold or not orientifold.get("requested", False):
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure", "intersecting_d7 requires an orientifold input"
        )
    if orientifold.get("status") not in {"fan_invariant", "validated"} or orientifold.get(
        "involution_type"
    ) != "O3/O7":
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure", "intersecting_d7 requires a validated O3/O7 lattice action"
        )
    images = np.asarray(orientifold.get("prime_divisor_image_indices", []), dtype=int).reshape(-1)
    if images.shape != (divisor_count,) or np.any((images < 0) | (images >= divisor_count)):
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure", "orientifold prime-divisor image map has an invalid shape"
        )
    return images == np.arange(divisor_count), images


def normalize_qcd_assignment(
    prime_volumes,
    effective_volumes,
    qcd_index,
    *,
    target=40.0,
    min_prime=1.0,
    min_effective=1.0,
    qcd_volume_tolerance=QCD_VOLUME_TOLERANCE,
    divisor_volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
):
    """Normalize one QCD assignment and validate its volume-domain contract.

    Apply the homogeneous dilation ``J -> m J`` for one selected prime toric
    divisor.  Four-cycle and effective-divisor volumes therefore scale by
    ``m**2``.  Return only JSON-compatible scalars and lists; pool rows retain
    compact summaries and digests rather than embedding these vectors.
    """
    prime = np.asarray(prime_volumes, dtype=float).reshape(-1)
    effective = np.asarray(effective_volumes, dtype=float).reshape(-1)
    try:
        qcd_scalar = float(qcd_index)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("qcd_index must be a finite integer") from exc
    target = float(target)
    min_prime = float(min_prime)
    min_effective = float(min_effective)
    qcd_volume_tolerance = float(qcd_volume_tolerance)
    divisor_volume_tolerance = float(divisor_volume_tolerance)
    if not np.isfinite(qcd_scalar) or qcd_scalar != int(qcd_scalar):
        raise ValueError("qcd_index must be a finite integer")
    qcd_index = int(qcd_scalar)
    if (
        prime.size == 0
        or effective.size == 0
        or effective.shape != prime.shape
        or not np.all(np.isfinite(prime))
        or not np.all(np.isfinite(effective))
        or not np.isfinite(qcd_scalar)
        or qcd_scalar != qcd_index
        or not 0 <= qcd_index < prime.size
    ):
        raise ValueError("normalization reference volumes or qcd_index are invalid")
    if (
        not np.isfinite(target)
        or target <= 0.0
        or not np.isfinite(min_prime)
        or not np.isfinite(min_effective)
        or min_prime < 1.0
        or min_effective < 1.0
        or not np.isfinite(qcd_volume_tolerance)
        or qcd_volume_tolerance < 0.0
        or not np.isfinite(divisor_volume_tolerance)
        or divisor_volume_tolerance < 0.0
    ):
        raise ValueError("normalization targets and tolerances must be finite and valid")
    if target != 40.0:
        raise ValueError("assignment-level QCD normalization requires target volume 40.0")
    if prime[qcd_index] <= 0.0:
        raise ValueError("QCD reference volume must be positive")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        radial_scale = float(np.sqrt(target / prime[qcd_index]))
        volume_scale = float(radial_scale**2)
        normalized_prime = volume_scale * prime
        normalized_effective = volume_scale * effective
    qcd_volume = float(normalized_prime[qcd_index])
    minimum_prime = float(np.min(normalized_prime))
    minimum_effective = float(np.min(normalized_effective))
    if not np.all(np.isfinite(normalized_prime)) or not np.all(np.isfinite(normalized_effective)):
        raise ValueError("post-normalization volumes are not finite")
    if not np.isclose(
        qcd_volume, target, rtol=0.0, atol=qcd_volume_tolerance
    ):
        raise ValueError("post-normalization QCD volume is not exactly the target")
    if minimum_prime < min_prime - divisor_volume_tolerance:
        raise ValueError("post-normalization prime-divisor minimum is below one")
    if minimum_effective < min_effective - divisor_volume_tolerance:
        raise ValueError("post-normalization effective-divisor minimum is below one")
    return {
        "qcd_index": qcd_index,
        "radial_scale": radial_scale,
        "volume_scale": volume_scale,
        "qcd_volume": qcd_volume,
        "prime_volumes": [float(value) for value in normalized_prime],
        "effective_volumes": [float(value) for value in normalized_effective],
        "minimum_prime_volume": minimum_prime,
        "minimum_effective_volume": minimum_effective,
        "target": target,
        "qcd_volume_tolerance": qcd_volume_tolerance,
        "divisor_volume_tolerance": divisor_volume_tolerance,
        "qcd_volume_residual": abs(qcd_volume - target),
        "normalization_map_version": NORMALIZATION_MAP_VERSION,
        "qcd_volume_exact": bool(
            np.isclose(qcd_volume, target, rtol=0.0, atol=qcd_volume_tolerance)
        ),
    }


def enumerate_assignment_pool(
    *,
    prime_labels,
    prime_charges,
    prime_volumes_reference,
    effective_volumes_reference,
    neighbors,
    intersection_evidence,
    invariant_mask=None,
    qcd_volume_target=40.0,
    min_prime_volume=1.0,
    min_effective_volume=1.0,
    qcd_volume_tolerance=QCD_VOLUME_TOLERANCE,
    divisor_volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
    qed_volume_max=127.5,
    terminal_records=None,
):
    """Enumerate the complete eligible ordered ``(QCD, QED)`` assignment pool.

    Normalize every QCD candidate independently before considering its QED
    neighbors.  The returned list contains accepted assignments only and is
    stably ordered by the pair of lattice-coordinate labels.  The returned
    list-like object exposes a serializable ``terminal_records`` sidecar for
    every rejected candidate and every accepted assignment.  When
    ``terminal_records`` is supplied, the same records are also appended to
    that caller-owned list; rejected records never become pool members.
    """

    normalization_map_version = globals().get(
        "NORMALIZATION_MAP_VERSION", "homogeneous-qcd-volume-40-v1"
    )

    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return [_jsonable(item) for item in value.tolist()]
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, (tuple, list)):
            return [_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        return value

    def _stable_hash(value):
        encoded = json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    collected_terminal_records = []

    def _emit(status, reason, qcd_index, qed_index=None, **fields):
        record = {
            "candidate_kind": "qcd" if qed_index is None else "qed",
            "terminal_status": str(status),
            "terminal_reason": str(reason),
            "qcd_index": int(qcd_index),
            "qcd_label": list(labels[qcd_index]),
            "normalization_map_version": normalization_map_version,
        }
        if qed_index is not None:
            record["qed_index"] = int(qed_index)
            record["qed_label"] = list(labels[qed_index])
        record.update(fields)
        collected_terminal_records.append(record)
        if terminal_records is not None:
            try:
                terminal_records.append(record)
            except AttributeError as exc:
                raise TypeError("terminal_records must provide append(record)") from exc
        return record

    try:
        labels = [tuple(int(value) for value in label) for label in prime_labels]
    except (TypeError, ValueError) as exc:
        raise ValueError("prime labels must be iterable integer coordinate labels") from exc
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("prime labels must be unique and non-empty")
    charges = _integer_array(prime_charges, "prime-divisor charges")
    if charges.ndim != 2 or charges.shape[0] != len(labels):
        raise ValueError("prime labels and charges have inconsistent shapes")
    prime_reference = np.asarray(prime_volumes_reference, dtype=float).reshape(-1)
    effective_reference = np.asarray(effective_volumes_reference, dtype=float).reshape(-1)
    if prime_reference.shape != (len(labels),):
        raise ValueError("prime reference volumes have an inconsistent shape")
    if not np.all(np.isfinite(prime_reference)) or not np.all(np.isfinite(effective_reference)):
        raise ValueError("reference volumes must be finite")
    if len(neighbors) != len(labels):
        raise ValueError("neighbors have an inconsistent shape")
    neighbor_lists = []
    for qcd_index, values in enumerate(neighbors):
        converted = set()
        for value in values:
            try:
                candidate = int(value)
                if float(value) != candidate:
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("neighbor indices must be integers") from exc
            if not 0 <= candidate < len(labels):
                raise ValueError("neighbor index is outside the prime-divisor list")
            converted.add(candidate)
        neighbor_lists.append(
            tuple(sorted(converted, key=lambda index: (labels[index], index)))
        )
    if intersection_evidence is None or not hasattr(intersection_evidence, "get"):
        raise ValueError("intersection_evidence must be a mapping")
    if invariant_mask is None:
        invariant = np.ones(len(labels), dtype=bool)
    else:
        invariant = np.asarray(invariant_mask, dtype=bool).reshape(-1)
        if invariant.shape != (len(labels),):
            raise ValueError("invariant_mask has an inconsistent shape")
    qcd_volume_target = float(qcd_volume_target)
    qed_volume_max = float(qed_volume_max)
    if (
        not np.isfinite(qcd_volume_target)
        or qcd_volume_target <= 0.0
        or not np.isfinite(qed_volume_max)
        or qed_volume_max <= 0.0
    ):
        raise ValueError("QCD target and QED upper bound must be finite and positive")
    if not np.isclose(qcd_volume_target, 40.0, rtol=0.0, atol=1e-12):
        raise ValueError("assignment pools require QCD target volume 40.0")
    if not np.isclose(qed_volume_max, 127.5, rtol=0.0, atol=1e-12):
        raise ValueError("assignment pools require QED volume upper bound 127.5")

    pool = []
    accepted_terminal_records = []
    for qcd_index in range(len(labels)):
        try:
            normalization = normalize_qcd_assignment(
                prime_reference,
                effective_reference,
                qcd_index,
                target=qcd_volume_target,
                min_prime=min_prime_volume,
                min_effective=min_effective_volume,
                qcd_volume_tolerance=qcd_volume_tolerance,
                divisor_volume_tolerance=divisor_volume_tolerance,
            )
        except (FloatingPointError, OverflowError, ValueError) as exc:
            _emit("qcd_normalization_failure", str(exc), qcd_index)
            continue

        # Under the identity O3/O7 convention a non-invariant QCD divisor is
        # not an eligible assignment source.  Perform this check only after
        # independent normalization so every QCD candidate crosses the same
        # normalization boundary before its ordered QED partners are tested.
        if not invariant[qcd_index]:
            _emit(
                "orientifold_invariance_failure",
                "QCD divisor is not invariant under the approved identity O3/O7 convention",
                qcd_index,
                qcd_radial_scale=float(normalization["radial_scale"]),
                qcd_volume_scale=float(normalization["volume_scale"]),
                qcd_volume=float(normalization["qcd_volume"]),
                minimum_prime_volume=float(normalization["minimum_prime_volume"]),
                minimum_effective_volume=float(normalization["minimum_effective_volume"]),
            )
            continue

        normalized_prime = normalization["prime_volumes"]
        normalized_effective = normalization["effective_volumes"]
        normalization_data = {
            "qcd_index": qcd_index,
            "qcd_reference_volume": float(prime_reference[qcd_index]),
            "effective_reference_volumes": effective_reference,
            "normalized_prime_volumes": normalized_prime,
            "normalized_effective_volumes": normalized_effective,
            "radial_scale": normalization["radial_scale"],
            "volume_scale": normalization["volume_scale"],
            "target": normalization["target"],
            "normalization_map_version": normalization_map_version,
        }
        normalization_data_hash = _stable_hash(normalization_data)
        accepted_for_qcd = 0
        for qed_index in neighbor_lists[qcd_index]:
            pair = tuple(sorted((qcd_index, qed_index)))
            evidence = intersection_evidence.get(pair, ())
            if qed_index == qcd_index:
                _emit(
                    "intersection_failure",
                    "QCD and QED divisors must be distinct",
                    qcd_index,
                    qed_index,
                )
                continue
            if not invariant[qed_index]:
                _emit(
                    "orientifold_invariance_failure",
                    "QED divisor is not invariant under the approved identity O3/O7 convention",
                    qcd_index,
                    qed_index,
                )
                continue
            if not np.any(charges[qed_index]):
                _emit(
                    "invalid_charge_basis_mapping",
                    "QED divisor has a zero charge vector",
                    qcd_index,
                    qed_index,
                )
                continue
            if not evidence:
                _emit(
                    "intersection_failure",
                    "no recorded nonzero QCD-QED intersection evidence",
                    qcd_index,
                    qed_index,
                )
                continue
            qed_volume = float(normalized_prime[qed_index])
            if not qed_volume <= qed_volume_max:
                _emit(
                    "qed_volume_rejection",
                    "normalized QED divisor volume exceeds the inclusive upper bound",
                    qcd_index,
                    qed_index,
                    qed_volume=qed_volume,
                )
                continue
            serial_evidence = [_jsonable(face) for face in evidence]
            assignment = {
                # Keep short names for the helper contract and the explicit
                # divisor names used by the existing HDF5 writer together.
                "qcd_index": qcd_index,
                "qed_index": qed_index,
                "qcd_divisor_index": qcd_index,
                "qed_divisor_index": qed_index,
                "qcd_label": list(labels[qcd_index]),
                "qed_label": list(labels[qed_index]),
                "qcd_divisor_label": list(labels[qcd_index]),
                "qed_divisor_label": list(labels[qed_index]),
                "qcd_charge_hash": charge_hash(charges[qcd_index]),
                "qed_charge_hash": charge_hash(charges[qed_index]),
                "qcd_reference_volume": float(prime_reference[qcd_index]),
                "qcd_radial_scale": float(normalization["radial_scale"]),
                "qcd_volume_scale": float(normalization["volume_scale"]),
                "qcd_volume_target": float(normalization["target"]),
                "qcd_volume_tolerance": float(normalization["qcd_volume_tolerance"]),
                "divisor_volume_tolerance": float(normalization["divisor_volume_tolerance"]),
                "qcd_volume_residual": float(normalization["qcd_volume_residual"]),
                "qcd_volume": float(normalization["qcd_volume"]),
                "qcd_divisor_volume": float(normalization["qcd_volume"]),
                "qed_volume": qed_volume,
                "qed_divisor_volume": qed_volume,
                "minimum_prime_volume": float(normalization["minimum_prime_volume"]),
                "minimum_effective_volume": float(normalization["minimum_effective_volume"]),
                "normalized_volume_summary": {
                    "prime_count": len(normalized_prime),
                    "effective_count": len(normalized_effective),
                    "minimum_prime_volume": float(normalization["minimum_prime_volume"]),
                    "minimum_effective_volume": float(normalization["minimum_effective_volume"]),
                    "qcd_volume": float(normalization["qcd_volume"]),
                    "qed_volume": qed_volume,
                    "normalized_prime_volumes_sha256": _stable_hash(normalized_prime),
                    "normalized_effective_volumes_sha256": _stable_hash(
                        normalized_effective
                    ),
                },
                "normalization_data_hash": normalization_data_hash,
                "normalization_map_version": normalization_map_version,
                "qcd_volume_exact": bool(normalization["qcd_volume_exact"]),
                "all_divisor_minimums_valid": True,
                "qed_volume_filter": "less_than_or_equal_to_127.5",
                "qed_volume_upper_bound": float(qed_volume_max),
                "qed_volume_comparison": "less_than_or_equal_to",
                "qcd_qed_intersection": True,
                "intersection_evidence": serial_evidence,
                "intersection_evidence_convention": (
                    "triangulated_two_face_lattice_point_labels"
                ),
                "assignment_policy": "ordered_qcd_qed_complete_pool",
                "orientifold_invariance_convention": (
                    "explicit_validated_O3/O7_prime_divisor_image_map"
                ),
                "qcd_invariant": True,
                "qed_invariant": True,
                "terminal_status": "accepted_assignment",
                "terminal_reason": "assignment-level QCD normalization and QED filters passed",
            }
            assignment["assignment_hash"] = _stable_hash(
                {"assignment": assignment, "normalization": normalization_data}
            )
            pool.append(assignment)
            accepted_for_qcd += 1
            accepted_record = _emit(
                "accepted_assignment",
                assignment["terminal_reason"],
                qcd_index,
                qed_index,
                assignment_hash=assignment["assignment_hash"],
                normalization_data_hash=normalization_data_hash,
                qcd_radial_scale=assignment["qcd_radial_scale"],
                qcd_volume_scale=assignment["qcd_volume_scale"],
                qcd_volume=assignment["qcd_volume"],
                qed_volume=assignment["qed_volume"],
            )
            accepted_terminal_records.append((accepted_record, assignment))
        if accepted_for_qcd == 0:
            _emit(
                "no_eligible_qed_divisor",
                "no ordered QCD-QED assignment passed the invariant, intersection, charge, and volume filters",
                qcd_index,
            )

    pool.sort(
        key=lambda item: (
            tuple(item["qcd_divisor_label"]),
            tuple(item["qed_divisor_label"]),
            item["qcd_divisor_index"],
            item["qed_divisor_index"],
        )
    )
    for rank, assignment in enumerate(pool):
        assignment["pool_rank"] = rank
    for record, assignment in accepted_terminal_records:
        record["pool_rank"] = int(assignment["pool_rank"])
    assignment_pool_type = globals().get("AssignmentPool")
    if assignment_pool_type is not None:
        result = assignment_pool_type(pool, collected_terminal_records, _stable_hash(pool))
        if result:
            validate_assignment_pool(result)
        return result
    return pool


def select_qed_divisor(
    *,
    policy,
    selection_policy,
    qcd_divisor_index,
    prime_toric_divisors,
    prime_divisor_labels,
    prime_divisor_charges_array,
    prime_divisor_volumes,
    neighbors,
    intersection_evidence,
    orientifold,
    effective_seed,
    qed_divisor_index_user=None,
    qed_volume_max=None,
):
    """Select one QED divisor conditional on one fixed geometry."""
    if policy == "none":
        return None
    if policy != "intersecting_d7":
        raise QEDAssignmentFailure("no_eligible_qed_divisor", f"unsupported visible-sector policy {policy!r}")
    if selection_policy not in {"uniform_eligible", "explicit"}:
        raise ValueError(f"unsupported QED selection policy {selection_policy!r}")

    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    stable = tuple(tuple(int(value) for value in label) for label in prime_divisor_labels)
    charges = _integer_array(prime_divisor_charges_array, "prime-divisor charges")
    volumes = np.asarray(prime_divisor_volumes, dtype=float).reshape(-1)
    if charges.ndim != 2 or charges.shape[0] != len(labels) or len(stable) != len(labels) or volumes.shape != (len(labels),):
        raise QEDAssignmentFailure("invalid_charge_basis_mapping", "prime-divisor metadata have inconsistent shapes")
    if not 0 <= int(qcd_divisor_index) < len(labels):
        raise QEDAssignmentFailure("invalid_charge_basis_mapping", "the selected QCD divisor index is invalid")
    invariant, images = _orientifold_mask(orientifold, len(labels))
    qcd_index = int(qcd_divisor_index)
    record = {
        "policy": policy,
        "selection_policy": selection_policy,
        "effective_seed": int(effective_seed),
        "qcd_divisor_index": qcd_index,
        "qcd_divisor_index_base": 0,
        "qcd_divisor_index_user": qcd_index + 1,
        "qcd_divisor_index_user_base": 1,
        "qcd_divisor_label": list(stable[qcd_index]),
        "qcd_image_index": int(images[qcd_index]),
        "qcd_invariant": bool(invariant[qcd_index]),
        "qcd_charge": charges[qcd_index].copy(),
        "qcd_charge_hash": charge_hash(charges[qcd_index]),
        "qcd_divisor_volume": float(volumes[qcd_index]),
        "qem_convention": "single_prime_divisor_toy",
        "claim_boundary": "geometry_derived_integer_toy_visible_sector_only",
        "candidate_pool_ordering": "stable_lattice_point_label_lexicographic",
        "qed_volume_upper_bound": None if qed_volume_max is None else float(qed_volume_max),
        "qed_volume_filter_status": "disabled" if qed_volume_max is None else "pending",
        "orientifold_h11_plus": orientifold.get("h11_plus"),
        "orientifold_h11_minus": orientifold.get("h11_minus"),
    }
    eligible_before_charge_filter = sorted(
        {
            int(index)
            for index in neighbors[qcd_index]
            if int(index) != qcd_index and invariant[int(index)]
        },
        key=lambda index: (stable[index], index),
    )
    excluded_zero_charge = [
        index
        for index in eligible_before_charge_filter
        if np.all(charges[index] == 0)
    ]
    eligible = sorted(
        {index for index in eligible_before_charge_filter if index not in excluded_zero_charge},
        key=lambda index: (stable[index], index),
    )
    record["candidate_pool_indices"] = eligible
    record["candidate_pool_labels"] = [list(stable[index]) for index in eligible]
    record["candidate_pool_size"] = len(eligible)
    record["candidate_pool_excluded_zero_charge_indices"] = excluded_zero_charge
    record["qed_candidate_pool_pre_volume_filter"] = True
    record["candidate_pool_conditioning"] = "fixed_geometry_qcd_policy_pre_volume_filter"
    record["qcd_neighbor_count"] = int(len(neighbors[qcd_index]))
    record["qed_candidate_count"] = int(len(eligible))
    if not eligible:
        record["terminal_status"] = "no_eligible_qed_divisor"
        record["terminal_reason"] = "no invariant nonzero-charge QED divisor intersects QCD"
        raise QEDAssignmentFailure("no_eligible_qed_divisor", record["terminal_reason"], record)

    explicit = None
    if qed_divisor_index_user is not None:
        if int(qed_divisor_index_user) < 1:
            record["terminal_status"] = "invalid_explicit_index"
            record["terminal_reason"] = "QED divisor index is one-based and must be positive"
            raise QEDAssignmentFailure("invalid_explicit_index", record["terminal_reason"], record)
        explicit = int(qed_divisor_index_user) - 1
    if selection_policy == "explicit" and explicit is None:
        raise QEDAssignmentFailure("invalid_explicit_index", "explicit QED selection requires an index", record)
    if selection_policy == "uniform_eligible" and explicit is not None:
        raise QEDAssignmentFailure("invalid_explicit_index", "an explicit index requires explicit selection", record)
    if selection_policy == "explicit":
        if explicit not in eligible:
            if not 0 <= explicit < len(labels):
                category, reason = "invalid_explicit_index", "explicit QED index is outside the prime-divisor list"
            elif not invariant[explicit]:
                category, reason = "orientifold_invariance_failure", "explicit QED divisor is not invariant"
            elif explicit == qcd_index:
                category, reason = "invalid_explicit_index", "QCD and QED divisors must differ"
            else:
                category, reason = "intersection_failure", "explicit QED divisor does not intersect QCD"
            record["terminal_status"], record["terminal_reason"] = category, reason
            raise QEDAssignmentFailure(category, reason, record)
        selected, rank = explicit, eligible.index(explicit) + 1
    else:
        rank = int(np.random.default_rng(int(effective_seed)).integers(len(eligible))) + 1
        selected = eligible[rank - 1]
    record.update(
        {
            "qed_divisor_index": int(selected),
            "qed_divisor_index_base": 0,
            "qed_divisor_index_user": int(selected) + 1,
            "qed_divisor_index_user_base": 1,
            "selection_rank": int(rank),
            "qed_divisor_label": list(stable[selected]),
            "qed_image_index": int(images[selected]),
            "qed_invariant": bool(invariant[selected]),
            "qed_charge": charges[selected].copy(),
            "em_charge": charges[selected].copy(),
            "qed_charge_hash": charge_hash(charges[selected]),
            "em_charge_hash": charge_hash(charges[selected]),
            "qed_divisor_volume": float(volumes[selected]),
        }
    )
    if qed_volume_max is not None and float(volumes[selected]) > float(qed_volume_max):
        record.update({"qed_volume_filter_status": "rejected"})
        record["terminal_status"] = "qed_volume_rejection"
        record["terminal_reason"] = "selected eligible QED divisor exceeds the volume upper bound"
        raise QEDAssignmentFailure("qed_volume_rejection", record["terminal_reason"], record)
    pair = tuple(sorted((qcd_index, selected)))
    record.update(
        {
            "qed_divisor_index": int(selected),
            "qed_divisor_index_base": 0,
            "qed_divisor_index_user": int(selected) + 1,
            "qed_divisor_index_user_base": 1,
            "selection_rank": int(rank),
            "qed_divisor_label": list(stable[selected]),
            "qed_image_index": int(images[selected]),
            "qed_invariant": bool(invariant[selected]),
            "qed_charge": charges[selected].copy(),
            "em_charge": charges[selected].copy(),
            "qed_charge_hash": charge_hash(charges[selected]),
            "em_charge_hash": charge_hash(charges[selected]),
            "qed_divisor_volume": float(volumes[selected]),
            "qcd_qed_intersection": True,
            "intersection_evidence": [list(face) for face in intersection_evidence.get(pair, [])],
            "intersection_evidence_convention": "triangulated_two_face_lattice_point_labels",
            "charge_convention": "CYTools divisor_basis(as_matrix=True) column selected by prime label",
            "qed_selection": selection_policy,
            "qed_volume_filter_status": "passed" if qed_volume_max is not None else "disabled",
        }
    )
    if not record["intersection_evidence"]:
        record["terminal_status"] = "intersection_failure"
        record["terminal_reason"] = "no face-level intersection evidence was retained"
        raise QEDAssignmentFailure("intersection_failure", record["terminal_reason"], record)
    return record


def _rank(rows):
    basis = {}
    for vector in rows:
        work = [Fraction(int(value)) for value in vector]
        for pivot, row in sorted(basis.items()):
            factor = work[pivot]
            if factor:
                work = [value - factor * row[index] for index, value in enumerate(work)]
        pivot = next((index for index, value in enumerate(work) if value), None)
        if pivot is not None:
            scale = work[pivot]
            basis[pivot] = tuple(value / scale for value in work)
    return len(basis)


def record_potential_match(q, l, qed_charge, direct_count, source_index):
    charges = _integer_array(q, "potential charge matrix")
    charge = _integer_array(qed_charge, "QED charge").reshape(-1)
    scales = np.asarray(l, dtype=float)
    if charges.ndim != 2 or scales.shape != (2, charges.shape[1]) or not 0 <= int(source_index) < charges.shape[1]:
        raise QEDAssignmentFailure("potential_term_mismatch", "potential arrays have incompatible shapes")
    if not np.array_equal(charges[:, int(source_index)], charge):
        raise QEDAssignmentFailure("potential_term_mismatch", "QED potential source does not equal qed_charge")
    order = sorted(range(charges.shape[1]), key=lambda index: (-float(scales[1, index]), index))
    return {
        "qed_potential_source": "direct_effective_cone" if source_index < direct_count else "appended_prime_divisor_e3",
        "qed_unsorted_potential_index": int(source_index),
        "qed_post_sort_source_position": int(order.index(int(source_index))),
        "qed_charge_exact_match": True,
        "qed_potential_scale": float(scales[1, source_index]),
    }


def classify_qed_leading_status(charges, scales, source_index):
    """Classify leading rank using exact integer row-rank increments."""
    q = _integer_array(charges, "potential charge matrix")
    l = np.asarray(scales, dtype=float)
    if l.shape != (2, q.shape[1]) or not 0 <= int(source_index) < q.shape[1]:
        raise QEDAssignmentFailure("potential_term_mismatch", "potential arrays have incompatible shapes")
    order = sorted(range(q.shape[1]), key=lambda index: (-float(l[1, index]), index))
    selected = []
    rank = 0
    for index in order:
        next_rank = _rank([q[:, selected_index] for selected_index in selected] + [q[:, index]])
        if next_rank > rank:
            selected.append(index)
            rank = next_rank
    status = "leading" if int(source_index) in selected else "dependent"
    return {
        "status": status,
        "selected_source_indices": [int(index) for index in selected],
        "ordered_source_indices": [int(index) for index in order],
        "selected_rank": int(rank),
        "method": "exact_rational_incremental_rank",
    }


def write_visible_sector_hdf5(group, assignment):
    """Write scalar/vector assignment fields and provenance into an HDF5 group."""
    for name in (
        "effective_seed", "candidate_pool_size", "selection_rank", "qcd_divisor_index",
        "qed_divisor_index", "qcd_image_index", "qed_image_index", "qcd_divisor_volume",
        "qed_divisor_volume", "qed_volume_upper_bound", "qed_instanton_index",
        "qed_unsorted_potential_index", "qed_post_sort_source_position", "qed_potential_scale",
        "qed_log10_lambda4",
    ):
        if name in assignment and assignment[name] is not None:
            group.create_dataset(name, data=assignment[name])
    for name in (
        "qcd_charge", "qed_charge", "em_charge", "candidate_pool_indices",
        "candidate_pool_labels", "qcd_divisor_label", "qed_divisor_label",
    ):
        if name in assignment:
            group.create_dataset(name, data=np.asarray(assignment[name], dtype=np.int64))
    for name in ("qcd_invariant", "qed_invariant", "qcd_qed_intersection", "qed_charge_exact_match"):
        if name in assignment:
            group.create_dataset(name, data=int(bool(assignment[name])))
    for name, value in assignment.items():
        if name in {
            "policy", "selection_policy", "qed_selection", "candidate_pool_conditioning",
            "candidate_pool_ordering", "charge_convention", "qem_convention", "qed_potential_source",
            "qed_leading_status", "qed_volume_filter_status", "terminal_status", "terminal_reason",
            "claim_boundary", "qcd_charge_hash", "qed_charge_hash", "em_charge_hash",
            "intersection_evidence_convention",
        }:
            group.attrs[name] = str(value)
    for name in ("qcd_divisor_index_user", "qed_divisor_index_user"):
        if name in assignment:
            group.attrs[name] = int(assignment[name])
            group.attrs[f"{name}_base"] = 1
    if "intersection_evidence" in assignment:
        group.attrs["intersection_evidence_json"] = json.dumps(assignment["intersection_evidence"], sort_keys=True)
