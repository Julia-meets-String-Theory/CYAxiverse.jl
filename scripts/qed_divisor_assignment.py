"""Select and serialize geometry-derived toy QED divisor assignments.

Keep this module independent of CYTools so that the selection contract can be
tested with small integer fixtures.  The generator supplies CYTools' stable
prime-divisor labels, basis matrix, intersection evidence, and orientifold
image map.
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
    """Raise a visible-sector assignment failure with a terminal category."""

    def __init__(self, category, reason, record=None):
        if category not in TERMINAL_FAILURE_CATEGORIES:
            raise ValueError(f"unknown QED assignment failure category {category!r}")
        super().__init__(reason)
        self.category = category
        self.reason = str(reason)
        self.record = {} if record is None else record


def _as_integer_matrix(values, name):
    """Validate an exact integer matrix without applying a basis change."""
    array = np.asarray(values)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", f"{name} must be a finite matrix"
        )
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", f"{name} must contain integers"
        )
    return rounded.astype(np.int64, copy=False)


def charge_hash(charge):
    """Hash an integer charge vector in a stable, documented representation."""
    vector = np.asarray(charge)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "charge must be a finite vector"
        )
    if not np.array_equal(vector, np.rint(vector)):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "charge must contain integers"
        )
    payload = json.dumps(
        [int(value) for value in vector], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_divisor_labels(prime_toric_divisors, lattice_points):
    """Return prime-divisor lattice-point labels in CYTools divisor order."""
    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    points = np.asarray(lattice_points, dtype=int)
    if labels.size == 0 or np.unique(labels).size != labels.size:
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping",
            "prime toric divisor labels must be unique and non-empty",
        )
    if points.ndim != 2 or points.shape[1] != 4:
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping",
            "lattice points must be an n_points by four-dimensional array",
        )
    if np.any(labels < 0) or np.any(labels >= points.shape[0]):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping",
            "a prime toric divisor label is outside the lattice-point table",
        )
    result = tuple(tuple(int(value) for value in points[label]) for label in labels)
    if len(set(result)) != len(result):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping",
            "prime toric divisors do not have unique lattice-point identities",
        )
    return result


def prime_divisor_charges(basis_matrix, prime_toric_divisors):
    """Derive one exact divisor-basis charge per stable prime-divisor label."""
    basis = _as_integer_matrix(basis_matrix, "divisor basis matrix")
    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    if basis.shape[1] == 0 or labels.size == 0:
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "the divisor basis has no prime columns"
        )
    if np.any(labels < 0) or np.any(labels >= basis.shape[1]):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping",
            "the divisor basis cannot represent every prime divisor label",
        )
    charges = np.asarray(basis[:, labels].T, dtype=np.int64)
    if charges.ndim != 2 or charges.shape[0] != labels.size:
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "prime-divisor charge shape is invalid"
        )
    return charges


def prime_divisor_intersection_graph(prime_toric_divisors, face_simplices):
    """Build stable prime-divisor neighbors and face-level evidence."""
    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != labels.size:
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping",
            "prime toric divisor labels must be unique and non-empty",
        )
    positions = {int(label): index for index, label in enumerate(labels)}
    simplices = np.asarray(face_simplices, dtype=int)
    if simplices.size == 0:
        return tuple(() for _ in labels), {}
    if simplices.ndim != 2:
        raise QEDAssignmentFailure(
            "intersection_failure", "two-face simplices must be a two-dimensional array"
        )
    neighbors = [set() for _ in labels]
    evidence = {}
    for simplex in simplices:
        face_labels = tuple(sorted({int(label) for label in simplex if int(label) in positions}))
        face_positions = tuple(sorted(positions[label] for label in face_labels))
        for first, second in ((face_positions[i], face_positions[j])
                              for i in range(len(face_positions))
                              for j in range(i + 1, len(face_positions))):
            neighbors[first].add(second)
            neighbors[second].add(first)
            evidence.setdefault((first, second), []).append(face_labels)
    return tuple(tuple(sorted(values)) for values in neighbors), evidence


def _orientifold_invariant_mask(orientifold, divisor_count):
    """Validate and return the prime-divisor orientifold-invariance mask."""
    if not orientifold or not orientifold.get("requested", False):
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure",
            "the selected visible-sector policy requires an orientifold input",
        )
    if orientifold.get("status") != "validated":
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure",
            "the orientifold input did not pass geometry validation",
        )
    if orientifold.get("involution_type") != "O3/O7":
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure",
            "the intersecting_d7 policy requires an O3/O7 involution",
        )
    if orientifold.get("h11_minus", 0) != 0:
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure",
            "the intersecting_d7 policy requires h11_minus=0",
        )
    image_indices = np.asarray(
        orientifold.get("prime_divisor_image_indices", []), dtype=int
    ).reshape(-1)
    if image_indices.shape != (divisor_count,) or np.any(
        (image_indices < 0) | (image_indices >= divisor_count)
    ):
        raise QEDAssignmentFailure(
            "orientifold_invariance_failure",
            "the orientifold prime-divisor image map has an invalid shape",
        )
    return image_indices == np.arange(divisor_count), image_indices


def _base_assignment_record(
    policy,
    selection_policy,
    effective_seed,
    qcd_index,
    qcd_label,
    qcd_charge,
    qcd_volume,
):
    return {
        "policy": policy,
        "selection_policy": selection_policy,
        "effective_seed": int(effective_seed),
        "qcd_divisor_index": int(qcd_index),
        "qcd_divisor_index_base": 0,
        "qcd_divisor_index_user": int(qcd_index) + 1,
        "qcd_divisor_index_user_base": 1,
        "qcd_divisor_label": list(qcd_label),
        "qcd_charge": np.asarray(qcd_charge, dtype=np.int64),
        "qcd_charge_hash": charge_hash(qcd_charge),
        "qcd_divisor_volume": float(qcd_volume),
        "qem_convention": "single_prime_divisor_toy",
        "claim_boundary": "geometry_derived_integer_toy_visible_sector_only",
    }


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
    """Select one eligible QED divisor conditional on a fixed geometry."""
    if policy == "none":
        return None
    if policy != "intersecting_d7":
        raise QEDAssignmentFailure(
            "no_eligible_qed_divisor", f"unsupported visible-sector policy {policy!r}"
        )
    if selection_policy not in {"uniform_eligible", "explicit"}:
        raise ValueError(f"unsupported QED selection policy {selection_policy!r}")

    labels = np.asarray(prime_toric_divisors, dtype=int).reshape(-1)
    stable_labels = tuple(tuple(int(x) for x in label) for label in prime_divisor_labels)
    volumes = np.asarray(prime_divisor_volumes, dtype=float).reshape(-1)
    charges = np.asarray(prime_divisor_charges_array, dtype=np.int64)
    if len(stable_labels) != len(labels) or volumes.shape != (len(labels),):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "prime-divisor metadata have inconsistent lengths"
        )
    if charges.shape != (len(labels), charges.shape[1] if charges.ndim == 2 else 0):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "prime-divisor charges have an invalid shape"
        )
    if not 0 <= int(qcd_divisor_index) < len(labels):
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", "the selected QCD divisor index is invalid"
        )
    if len(neighbors) != len(labels):
        raise QEDAssignmentFailure(
            "intersection_failure", "the divisor intersection graph has an invalid size"
        )

    invariant, image_indices = _orientifold_invariant_mask(orientifold, len(labels))
    qcd_index = int(qcd_divisor_index)
    qcd_label = stable_labels[qcd_index]
    record = _base_assignment_record(
        policy,
        selection_policy,
        effective_seed,
        qcd_index,
        qcd_label,
        charges[qcd_index],
        volumes[qcd_index],
    )
    record.update(
        {
            "qcd_image_index": int(image_indices[qcd_index]),
            "qcd_invariant": bool(invariant[qcd_index]),
            "candidate_pool_ordering": "stable_lattice_point_label_lexicographic",
            "qed_volume_upper_bound": None if qed_volume_max is None else float(qed_volume_max),
            "qed_volume_filter_status": "disabled" if qed_volume_max is None else "pending",
        }
    )

    eligible_before_charge_filter = [
        index
        for index in neighbors[qcd_index]
        if int(index) != qcd_index and invariant[int(index)]
    ]
    excluded_zero_charge = sorted(
        int(index)
        for index in eligible_before_charge_filter
        if np.all(charges[int(index)] == 0)
    )
    eligible = [
        index
        for index in eligible_before_charge_filter
        if not np.all(charges[int(index)] == 0)
    ]
    eligible = sorted(eligible, key=lambda index: (stable_labels[index], int(index)))
    record["candidate_pool_indices"] = [int(index) for index in eligible]
    record["candidate_pool_labels"] = [list(stable_labels[index]) for index in eligible]
    record["candidate_pool_size"] = len(eligible)
    record["candidate_pool_excluded_zero_charge_indices"] = excluded_zero_charge
    record["candidate_pool_conditioning"] = "fixed_geometry_qcd_policy_pre_volume_filter"
    record["qed_candidate_pool_pre_volume_filter"] = True
    if not eligible:
        record["terminal_status"] = "no_eligible_qed_divisor"
        record["terminal_reason"] = "no invariant QED divisor intersects the QCD divisor"
        raise QEDAssignmentFailure(
            "no_eligible_qed_divisor", record["terminal_reason"], record
        )

    explicit_internal = None
    if qed_divisor_index_user is not None:
        try:
            user_index = int(qed_divisor_index_user)
        except (TypeError, ValueError):
            user_index = 0
        if user_index < 1:
            record["terminal_status"] = "invalid_explicit_index"
            record["terminal_reason"] = "QED divisor index must be one-based and positive"
            raise QEDAssignmentFailure(
                "invalid_explicit_index", record["terminal_reason"], record
            )
        explicit_internal = user_index - 1
        record["qed_divisor_index_user"] = user_index
        record["qed_divisor_index_user_base"] = 1
    if selection_policy == "explicit" and explicit_internal is None:
        record["terminal_status"] = "invalid_explicit_index"
        record["terminal_reason"] = "explicit QED selection requires --qed-divisor-index"
        raise QEDAssignmentFailure(
            "invalid_explicit_index", record["terminal_reason"], record
        )
    if selection_policy == "uniform_eligible" and explicit_internal is not None:
        record["terminal_status"] = "invalid_explicit_index"
        record["terminal_reason"] = (
            "--qed-divisor-index is only valid with explicit QED selection"
        )
        raise QEDAssignmentFailure(
            "invalid_explicit_index", record["terminal_reason"], record
        )

    if selection_policy == "explicit":
        if explicit_internal not in eligible:
            if not 0 <= explicit_internal < len(labels):
                category = "invalid_explicit_index"
                reason = "explicit QED divisor index is outside the prime-divisor list"
            elif not invariant[explicit_internal]:
                category = "orientifold_invariance_failure"
                reason = "explicit QED divisor is not orientifold invariant"
            elif explicit_internal == qcd_index:
                category = "invalid_explicit_index"
                reason = "the QCD divisor cannot also be the QED divisor"
            else:
                category = "intersection_failure"
                reason = "explicit QED divisor does not intersect the QCD divisor"
            record["terminal_status"] = category
            record["terminal_reason"] = reason
            raise QEDAssignmentFailure(category, reason, record)
        selected_index = explicit_internal
        selection_rank = eligible.index(selected_index) + 1
    else:
        rng = np.random.default_rng(int(effective_seed))
        selection_rank = int(rng.integers(len(eligible))) + 1
        selected_index = eligible[selection_rank - 1]

    selected_label = stable_labels[selected_index]
    if np.all(charges[selected_index] == 0):
        record["terminal_status"] = "invalid_charge_basis_mapping"
        record["terminal_reason"] = "selected QED divisor has a zero restricted charge"
        raise QEDAssignmentFailure(
            "invalid_charge_basis_mapping", record["terminal_reason"], record
        )
    record.update(
        {
            "qed_divisor_index": int(selected_index),
            "qed_divisor_index_base": 0,
            "qed_divisor_index_user": int(selected_index) + 1,
            "qed_divisor_index_user_base": 1,
            "selection_rank": int(selection_rank),
            "qed_divisor_label": list(selected_label),
            "qed_image_index": int(image_indices[selected_index]),
            "qed_invariant": bool(invariant[selected_index]),
            "qed_charge": charges[selected_index].copy(),
            "em_charge": charges[selected_index].copy(),
            "qed_charge_hash": charge_hash(charges[selected_index]),
            "em_charge_hash": charge_hash(charges[selected_index]),
            "qed_divisor_volume": float(volumes[selected_index]),
            "qcd_qed_intersection": True,
            "intersection_evidence": [
                list(face)
                for face in intersection_evidence.get(
                    tuple(sorted((qcd_index, selected_index))), []
                )
            ],
            "intersection_evidence_convention": (
                "triangulated_two_face_lattice_point_labels"
            ),
            "charge_convention": (
                "CYTools divisor_basis(as_matrix=True) column selected by prime label"
            ),
        }
    )
    if not record["intersection_evidence"]:
        record["terminal_status"] = "intersection_failure"
        record["terminal_reason"] = "no face-level intersection evidence was retained"
        raise QEDAssignmentFailure(
            "intersection_failure", record["terminal_reason"], record
        )

    if qed_volume_max is not None:
        if qed_volume_max <= 0.0 or not np.isfinite(qed_volume_max):
            raise ValueError("qed_volume_max must be finite and positive")
        if record["qed_divisor_volume"] > float(qed_volume_max):
            record["qed_volume_filter_status"] = "rejected"
            record["terminal_status"] = "qed_volume_rejection"
            record["terminal_reason"] = (
                "selected eligible QED divisor exceeds the declared volume upper bound"
            )
            raise QEDAssignmentFailure(
                "qed_volume_rejection", record["terminal_reason"], record
            )
        record["qed_volume_filter_status"] = "passed"
    return record


class _ExactRowBasis:
    """Maintain an exact rational row-echelon basis for integer vectors."""

    def __init__(self, rows=None):
        self.rows = {} if rows is None else dict(rows)

    def copy(self):
        return _ExactRowBasis(self.rows)

    def add(self, vector):
        work = [Fraction(int(value)) for value in vector]
        for pivot in sorted(self.rows):
            factor = work[pivot]
            if factor:
                row = self.rows[pivot]
                for index in range(pivot, len(work)):
                    work[index] -= factor * row[index]
        pivot = next((index for index, value in enumerate(work) if value), None)
        if pivot is None:
            return False
        scale = work[pivot]
        normalized = tuple(value / scale for value in work)
        self.rows[pivot] = normalized
        return True


def classify_qed_leading_status(charges, scales, source_index):
    """Classify a QED term using exact ordered rational rank selection."""
    Q = _as_integer_matrix(charges, "potential charge matrix")
    L = np.asarray(scales, dtype=float)
    if L.ndim != 2 or L.shape[0] != 2 or L.shape[1] != Q.shape[1]:
        raise QEDAssignmentFailure(
            "potential_term_mismatch", "potential charges and scales have incompatible shapes"
        )
    if not 0 <= int(source_index) < Q.shape[1]:
        return {
            "status": "unavailable",
            "selected_source_indices": [],
            "ordered_source_indices": [],
            "selected_rank": 0,
            "method": "exact_rational_incremental_rank",
        }
    order = sorted(range(Q.shape[1]), key=lambda index: (-float(L[1, index]), index))
    basis = _ExactRowBasis()
    selected = []
    for index in order:
        if basis.add(Q[:, index]):
            selected.append(index)
    if source_index in selected:
        status = "leading"
        in_span = True
    else:
        in_span = not basis.copy().add(Q[:, source_index])
        if not in_span:
            status = "unavailable"
        else:
            duplicate = any(
                np.array_equal(Q[:, source_index], Q[:, index])
                or np.array_equal(Q[:, source_index], -Q[:, index])
                for index in selected
            )
            status = "dependent" if duplicate else "span_leading"
    return {
        "status": status,
        "qed_in_selected_span": bool(in_span),
        "selected_source_indices": [int(index) for index in selected],
        "ordered_source_indices": [int(index) for index in order],
        "selected_rank": len(selected),
        "method": "exact_rational_incremental_rank",
    }


def record_potential_match(q, l, qed_charge, direct_count, source_index):
    """Verify exact QED charge equality and return its source metadata."""
    Q = _as_integer_matrix(q, "potential charge matrix")
    L = np.asarray(l, dtype=float)
    charge = _as_integer_matrix(np.asarray(qed_charge).reshape(1, -1), "QED charge")[0]
    if Q.shape[0] != charge.size or L.shape != (2, Q.shape[1]):
        raise QEDAssignmentFailure(
            "potential_term_mismatch", "QED potential matching received incompatible arrays"
        )
    if not 0 <= int(source_index) < Q.shape[1] or not np.array_equal(
        Q[:, source_index], charge
    ):
        raise QEDAssignmentFailure(
            "potential_term_mismatch", "the recorded QED potential source does not equal qed_charge"
        )
    direct_match = int(source_index) < int(direct_count)
    order = sorted(range(Q.shape[1]), key=lambda index: (-float(L[1, index]), index))
    return {
        "qed_potential_source": (
            "direct_effective_cone" if direct_match else "appended_prime_divisor_e3"
        ),
        "qed_unsorted_potential_index": int(source_index),
        "qed_post_sort_source_position": int(order.index(int(source_index))),
        "qed_charge_exact_match": True,
        "qed_charge_hash": charge_hash(charge),
        "qed_potential_scale": float(L[1, source_index]),
    }


def summarize_terminal_failures(records):
    """Count every assignment result under the fixed terminal taxonomy."""
    summary = {category: 0 for category in TERMINAL_FAILURE_CATEGORIES}
    for record in records:
        category = record.category if isinstance(record, QEDAssignmentFailure) else record
        if category not in summary:
            category = "numerical_geometry_failure"
        summary[category] += 1
    return summary


def write_visible_sector_hdf5(group, assignment):
    """Write a visible-sector assignment into an open HDF5 group."""
    scalar_names = (
        "effective_seed",
        "candidate_pool_size",
        "selection_rank",
        "qcd_divisor_index",
        "qed_divisor_index",
        "qcd_image_index",
        "qed_image_index",
        "qcd_divisor_volume",
        "qed_divisor_volume",
        "qed_volume_upper_bound",
        "qed_unsorted_potential_index",
        "qed_post_sort_source_position",
        "qed_potential_scale",
        "qed_log10_lambda4",
    )
    for name in scalar_names:
        if name in assignment and assignment[name] is not None:
            value = np.nan if assignment[name] is None else assignment[name]
            group.create_dataset(name, data=value)
    for name in (
        "qcd_charge",
        "qed_charge",
        "em_charge",
        "candidate_pool_indices",
    ):
        if name in assignment:
            group.create_dataset(name, data=np.asarray(assignment[name], dtype=np.int64))
    for name in (
        "qcd_invariant",
        "qed_invariant",
        "qcd_qed_intersection",
        "qed_charge_exact_match",
    ):
        if name in assignment:
            group.create_dataset(name, data=int(bool(assignment[name])))
    for name in (
        "policy",
        "selection_policy",
        "candidate_pool_conditioning",
        "candidate_pool_ordering",
        "charge_convention",
        "qem_convention",
        "qed_potential_source",
        "qed_leading_status",
        "qed_volume_filter_status",
        "terminal_status",
        "terminal_reason",
        "claim_boundary",
        "qcd_charge_hash",
        "qed_charge_hash",
        "em_charge_hash",
    ):
        if name in assignment:
            group.attrs[name] = str(assignment[name])
    group.attrs["qed_divisor_index_user"] = int(assignment["qed_divisor_index_user"])
    group.attrs["qed_divisor_index_user_base"] = 1
    group.attrs["qcd_divisor_index_user"] = int(assignment["qcd_divisor_index_user"])
    group.attrs["qcd_divisor_index_user_base"] = 1
    group.attrs["candidate_pool_labels_json"] = json.dumps(
        assignment.get("candidate_pool_labels", []), separators=(",", ":")
    )
    group.attrs["qcd_divisor_label_json"] = json.dumps(
        assignment.get("qcd_divisor_label", []), separators=(",", ":")
    )
    group.attrs["qed_divisor_label_json"] = json.dumps(
        assignment.get("qed_divisor_label", []), separators=(",", ":")
    )
    group.attrs["intersection_evidence_json"] = json.dumps(
        assignment.get("intersection_evidence", []), separators=(",", ":")
    )


def read_visible_sector_hdf5(group):
    """Read a visible-sector group into a serializable assignment mapping."""
    result = {name: group[name][()] for name in group.keys()}
    for name in group.attrs:
        value = group.attrs[name]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result[name] = value
    for name in (
        "candidate_pool_labels_json",
        "qcd_divisor_label_json",
        "qed_divisor_label_json",
        "intersection_evidence_json",
    ):
        if name in result:
            result[name[:-5] if name.endswith("_json") else name] = json.loads(result[name])
    return result
