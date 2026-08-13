"""Geometry-derived toy QED divisor selection for the CYTools generator.

This module deliberately has no CYTools dependency.  The generator supplies
the stable prime-divisor labels, divisor-basis charges, two-face intersection
evidence, volumes, and validated orientifold image map.
"""

from __future__ import annotations

from fractions import Fraction

import hashlib
import json

import numpy as np


TERMINAL_FAILURE_CATEGORIES = (
    "no_eligible_qed_divisor",
    "invalid_explicit_index",
    "orientifold_invariance_failure",
    "intersection_failure",
    "qed_volume_rejection",
    "invalid_charge_basis_mapping",
    "potential_term_mismatch",
    "numerical_geometry_failure",
    "output_collision",
    "accepted_assignment",
)


class QEDAssignmentFailure(RuntimeError):
    """A visible-sector outcome with an explicit terminal category."""

    def __init__(self, category, reason, record=None):
        if category not in TERMINAL_FAILURE_CATEGORIES:
            raise ValueError(f"unknown QED assignment failure category {category!r}")
        super().__init__(reason)
        self.category = category
        self.reason = str(reason)
        self.record = {} if record is None else record


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
    if orientifold.get("status") != "validated" or orientifold.get("involution_type") != "O3/O7":
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure", "intersecting_d7 requires a validated O3/O7 involution"
        )
    if orientifold.get("h11_minus", 0) != 0:
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure", "intersecting_d7 requires h11_minus=0"
        )
    images = np.asarray(orientifold.get("prime_divisor_image_indices", []), dtype=int).reshape(-1)
    if images.shape != (divisor_count,) or np.any((images < 0) | (images >= divisor_count)):
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure", "orientifold prime-divisor image map has an invalid shape"
        )
    return images == np.arange(divisor_count), images


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
    for name in ("qcd_charge", "qed_charge", "em_charge", "candidate_pool_indices"):
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
