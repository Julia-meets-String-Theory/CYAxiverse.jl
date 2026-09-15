"""Manager-facing exact comparison for the generator-2.3 G1 gate.

This harness is intentionally calibration-only.  It imports the two completed
implementations only at comparison time and compares the complete identity-
bearing state, not only counts or graph topology.  No T0--T4 label is accepted
by the command-line interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Mapping

from . import generator_independent as independent
from . import generator_primary as primary


def _flatten_cells(matrix: dict[str, dict[str, tuple[int, ...]]]) -> frozenset[tuple[str, str, int]]:
    return frozenset(
        (tier, profile, seed)
        for tier, profiles in matrix.items()
        for profile, seeds in profiles.items()
        for seed in seeds
    )


# The approved G1 calibration registry is 1 C0 + 6 C1 + 6 C2 + 3 C3 cells.
# Refuse a comparison if either implementation silently drops or adds a cell.
CALIBRATION_CELLS = _flatten_cells(primary.CALIBRATION_SEEDS)
if CALIBRATION_CELLS != _flatten_cells(independent.CALIBRATION_MATRIX) or len(CALIBRATION_CELLS) != 16:
    raise RuntimeError("primary and independent calibration registries do not contain the approved 16 C0-C3 cells")


def _canonical_serialized_bytes(value: Any) -> bytes:
    """Serialize complete output losslessly with one deterministic JSON form."""
    def normalize(item: Any) -> Any:
        if isinstance(item, bytes):
            return {"__bytes__": item.hex()}
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item
    return (json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _unprefix_checksum(value: str) -> str:
    prefix = "cyax-snapshot-sha256:"
    return value.removeprefix(prefix)


def compare_cell(tier: str, profile_id: str, seed: int) -> dict[str, Any]:
    """Compare one approved calibration cell and return JSON-ready evidence."""

    if (tier, profile_id, seed) not in CALIBRATION_CELLS:
        raise ValueError(f"cell is not registered for this harness: {tier}/{profile_id}/{seed}")

    left = primary.generate_calibration_snapshot(tier, profile_id, seed)
    right = independent.generate_calibration_snapshot(tier, profile_id, seed)
    right_records = independent.physical_records(right)
    left_output = primary.complete_output(left)
    right_output = independent.complete_output(right)
    left_bytes = _canonical_serialized_bytes(left_output)
    right_bytes = _canonical_serialized_bytes(right_output)
    checks = {
        "physical_records": left.records == right_records,
        "source_bytes": left.source_bytes == right.source_bytes,
        "construction_trace": left.construction_trace == right.construction_trace,
        "prf_choices": left.prf_choices == right.prf_trace,
        "candidate_vectors": left.candidate_vectors == right.candidate_vectors,
        "assertion_ids": left.assertion_ids == [record.assertion_id for record in right.assertions],
        "logical_snapshot_checksum": left.logical_snapshot_checksum == _unprefix_checksum(right.logical_snapshot_checksum),
        "snapshot_id": left.snapshot_id == right.snapshot_id,
        "complete_generator_manifest": left.manifest == right.manifest,
        "complete_canonical_serialized_bytes": left_bytes == right_bytes,
    }
    return {
        "tier": tier,
        "profile_id": profile_id,
        "seed": seed,
        "primary": {
            "entity_count": len(left.entities),
            "assertion_count": len(left.assertions),
            "source_revision_count": len(left.source_revisions),
            "construction_trace_count": len(left.construction_trace),
            "prf_choice_count": len(left.prf_choices),
            "candidate_vector_count": len(left.candidate_vectors),
            "logical_snapshot_checksum": left.logical_snapshot_checksum,
            "complete_canonical_sha256": hashlib.sha256(left_bytes).hexdigest(),
        },
        "independent": {
            "entity_count": len(right.entities),
            "assertion_count": len(right.assertions),
            "source_revision_count": len(right.source_revisions),
            "construction_trace_count": len(right.construction_trace),
            "prf_choice_count": len(right.prf_trace),
            "candidate_vector_count": len(right.candidate_vectors),
            "logical_snapshot_checksum": right.logical_snapshot_checksum,
            "complete_canonical_sha256": hashlib.sha256(right_bytes).hexdigest(),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="C0", choices=("C0", "C1", "C2", "C3"))
    parser.add_argument("--profile", default="P-medium", choices=("P-low", "P-medium", "P-high"))
    parser.add_argument("--seed", default=168900, type=int)
    args = parser.parse_args()
    evidence = compare_cell(args.tier, args.profile, args.seed)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
