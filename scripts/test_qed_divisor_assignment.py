"""Focused regression tests for geometry-derived QED divisor assignments."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - exercised in the base Python
    h5py = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qed_divisor_assignment import (  # noqa: E402
    QEDAssignmentFailure,
    classify_qed_leading_status,
    prime_divisor_charges,
    prime_divisor_intersection_graph,
    read_visible_sector_hdf5,
    record_potential_match,
    select_qed_divisor,
    stable_divisor_labels,
    summarize_terminal_failures,
    write_visible_sector_hdf5,
)


def fixture():
    # The stable lattice labels are deliberately not in internal divisor order.
    points = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [2, 0, 0, 0],
        ],
        dtype=int,
    )
    prime_labels = np.asarray([1, 2, 3, 4], dtype=int)
    basis = np.asarray(
        [
            [1, 2, 0, 3, 4],
            [0, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    neighbors = ((1, 2, 3), (0,), (0,), (0,))
    evidence = {
        (0, 1): [(-1, 1, 0, 0)],
        (0, 2): [(0, 1, 0, 0)],
        (0, 3): [(0, 2, 0, 0)],
    }
    orientifold = {
        "requested": True,
        "status": "validated",
        "involution_type": "O3/O7",
        "h11_minus": 0,
        "prime_divisor_image_indices": np.arange(4, dtype=int),
    }
    volumes = np.asarray([30.0, 2.0, 3.0, 130.0])
    return points, prime_labels, basis, neighbors, evidence, orientifold, volumes


def test_stable_order_and_exact_charge_derivation():
    points, prime_labels, basis, *_ = fixture()
    labels = stable_divisor_labels(prime_labels, points)
    assert labels == ((1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (2, 0, 0, 0))
    charges = prime_divisor_charges(basis, prime_labels)
    assert charges.dtype.kind == "i"
    assert np.array_equal(charges[0], [2, 0])
    assert np.array_equal(charges[3], [4, 1])


def test_uniform_selection_is_replayable_and_seeded():
    points, prime_labels, basis, neighbors, evidence, orientifold, volumes = fixture()
    labels = stable_divisor_labels(prime_labels, points)
    charges = prime_divisor_charges(basis, prime_labels)
    first = select_qed_divisor(
        policy="intersecting_d7",
        selection_policy="uniform_eligible",
        qcd_divisor_index=0,
        prime_toric_divisors=prime_labels,
        prime_divisor_labels=labels,
        prime_divisor_charges_array=charges,
        prime_divisor_volumes=volumes,
        neighbors=neighbors,
        intersection_evidence=evidence,
        orientifold=orientifold,
        effective_seed=11,
    )
    replay = select_qed_divisor(
        policy="intersecting_d7",
        selection_policy="uniform_eligible",
        qcd_divisor_index=0,
        prime_toric_divisors=prime_labels,
        prime_divisor_labels=labels,
        prime_divisor_charges_array=charges,
        prime_divisor_volumes=volumes,
        neighbors=neighbors,
        intersection_evidence=evidence,
        orientifold=orientifold,
        effective_seed=11,
    )
    assert first["qed_divisor_index"] == replay["qed_divisor_index"]
    assert first["qed_charge"].tolist() == replay["qed_charge"].tolist()
    assert first["candidate_pool_indices"] == [1, 2, 3]
    assert first["candidate_pool_size"] == 3
    assert first["selection_rank"] in (1, 2, 3)
    assert first["qed_divisor_index_user"] == first["qed_divisor_index"] + 1
    changed_seed = select_qed_divisor(
        policy="intersecting_d7",
        selection_policy="uniform_eligible",
        qcd_divisor_index=0,
        prime_toric_divisors=prime_labels,
        prime_divisor_labels=labels,
        prime_divisor_charges_array=charges,
        prime_divisor_volumes=volumes,
        neighbors=neighbors,
        intersection_evidence=evidence,
        orientifold=orientifold,
        effective_seed=1,
    )
    assert changed_seed["candidate_pool_indices"] == first["candidate_pool_indices"]
    assert changed_seed["qed_divisor_index"] != first["qed_divisor_index"]


def test_explicit_selection_and_invalid_index_do_not_fallback():
    points, prime_labels, basis, neighbors, evidence, orientifold, volumes = fixture()
    labels = stable_divisor_labels(prime_labels, points)
    charges = prime_divisor_charges(basis, prime_labels)
    selected = select_qed_divisor(
        policy="intersecting_d7",
        selection_policy="explicit",
        qed_divisor_index_user=2,
        qcd_divisor_index=0,
        prime_toric_divisors=prime_labels,
        prime_divisor_labels=labels,
        prime_divisor_charges_array=charges,
        prime_divisor_volumes=volumes,
        neighbors=neighbors,
        intersection_evidence=evidence,
        orientifold=orientifold,
        effective_seed=3,
    )
    assert selected["qed_divisor_index"] == 1
    assert selected["qed_divisor_index_user"] == 2
    try:
        select_qed_divisor(
            policy="intersecting_d7",
            selection_policy="explicit",
            qed_divisor_index_user=5,
            qcd_divisor_index=0,
            prime_toric_divisors=prime_labels,
            prime_divisor_labels=labels,
            prime_divisor_charges_array=charges,
            prime_divisor_volumes=volumes,
            neighbors=neighbors,
            intersection_evidence=evidence,
            orientifold=orientifold,
            effective_seed=3,
        )
    except QEDAssignmentFailure as failure:
        assert failure.category == "invalid_explicit_index"
    else:
        raise AssertionError("out-of-range explicit index unexpectedly succeeded")
    try:
        select_qed_divisor(
            policy="intersecting_d7",
            selection_policy="explicit",
            qed_divisor_index_user=1,
            qcd_divisor_index=0,
            prime_toric_divisors=prime_labels,
            prime_divisor_labels=labels,
            prime_divisor_charges_array=charges,
            prime_divisor_volumes=volumes,
            neighbors=neighbors,
            intersection_evidence=evidence,
            orientifold=orientifold,
            effective_seed=3,
        )
    except QEDAssignmentFailure as failure:
        assert failure.category == "invalid_explicit_index"
    else:
        raise AssertionError("explicit QCD index unexpectedly selected as QED")


def test_qed_volume_rejection_keeps_pre_filter_candidate_provenance():
    points, prime_labels, basis, neighbors, evidence, orientifold, volumes = fixture()
    labels = stable_divisor_labels(prime_labels, points)
    charges = prime_divisor_charges(basis, prime_labels)
    try:
        select_qed_divisor(
            policy="intersecting_d7",
            selection_policy="explicit",
            qed_divisor_index_user=4,
            qcd_divisor_index=0,
            prime_toric_divisors=prime_labels,
            prime_divisor_labels=labels,
            prime_divisor_charges_array=charges,
            prime_divisor_volumes=volumes,
            neighbors=neighbors,
            intersection_evidence=evidence,
            orientifold=orientifold,
            effective_seed=3,
            qed_volume_max=127.5,
        )
    except QEDAssignmentFailure as failure:
        assert failure.category == "qed_volume_rejection"
        assert failure.record["candidate_pool_indices"] == [1, 2, 3]
        assert failure.record["qed_divisor_index_user"] == 4
        assert failure.record["qed_volume_filter_status"] == "rejected"
    else:
        raise AssertionError("volume filter did not reject the selected divisor")


def test_intersection_graph_and_potential_leading_status():
    _, prime_labels, _, *_ = fixture()
    neighbors, evidence = prime_divisor_intersection_graph(
        prime_labels, np.asarray([[1, 2], [1, 3]], dtype=int)
    )
    assert neighbors[0] == (1, 2)
    assert evidence[(0, 1)] == [(1, 2)]
    Q = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=int)
    L = np.asarray([[1.0, 1.0, 1.0], [3.0, 2.0, 1.0]])
    match = record_potential_match(Q, L, [1, 1], direct_count=2, source_index=2)
    assert match["qed_potential_source"] == "appended_prime_divisor_e3"
    assert match["qed_post_sort_source_position"] == 2
    status = classify_qed_leading_status(Q, L, 2)
    assert status["status"] == "span_leading"
    duplicate_status = classify_qed_leading_status(
        np.asarray([[1, 1], [0, 0]], dtype=int),
        np.asarray([[1.0, 1.0], [2.0, 1.0]]),
        1,
    )
    assert duplicate_status["status"] == "dependent"


def test_hdf5_metadata_round_trip_and_terminal_accounting():
    if h5py is None:
        return
    points, prime_labels, basis, neighbors, evidence, orientifold, volumes = fixture()
    labels = stable_divisor_labels(prime_labels, points)
    charges = prime_divisor_charges(basis, prime_labels)
    assignment = select_qed_divisor(
        policy="intersecting_d7",
        selection_policy="explicit",
        qed_divisor_index_user=2,
        qcd_divisor_index=0,
        prime_toric_divisors=prime_labels,
        prime_divisor_labels=labels,
        prime_divisor_charges_array=charges,
        prime_divisor_volumes=volumes,
        neighbors=neighbors,
        intersection_evidence=evidence,
        orientifold=orientifold,
        effective_seed=7,
    )
    assignment.update(
        record_potential_match(
            np.asarray([[0, 2], [1, 0]], dtype=int),
            np.asarray([[1.0, 1.0], [2.0, 1.0]]),
            assignment["qed_charge"],
            direct_count=2,
            source_index=0,
        )
    )
    assignment["qed_leading_status"] = "leading"
    assignment["terminal_status"] = "accepted_assignment"
    assignment["terminal_reason"] = "test fixture"
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "assignment.h5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("visible_sector")
            write_visible_sector_hdf5(group, assignment)
        with h5py.File(path, "r") as handle:
            restored = read_visible_sector_hdf5(handle["visible_sector"])
        assert restored["qed_charge"].tolist() == assignment["qed_charge"].tolist()
        assert restored["candidate_pool_labels"] == assignment["candidate_pool_labels"]
        assert restored["terminal_status"] == "accepted_assignment"
        assert json.loads(restored["intersection_evidence_json"]) == assignment[
            "intersection_evidence"
        ]
    summary = summarize_terminal_failures(
        ["accepted_assignment", "invalid_explicit_index", "qed_volume_rejection"]
    )
    assert summary["accepted_assignment"] == 1
    assert summary["invalid_explicit_index"] == 1
    assert summary["qed_volume_rejection"] == 1


if __name__ == "__main__":
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} QED divisor assignment tests passed")
