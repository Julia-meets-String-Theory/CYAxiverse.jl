"""Manager-facing exact comparison for the generator-2.3 G1 gate.

This harness is intentionally calibration-only.  It imports the two completed
implementations only at comparison time and compares the complete identity-
bearing state, not only counts or graph topology.  No T0--T4 label is accepted
by the command-line interface.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from . import generator_independent as independent
from . import generator_primary as primary


CALIBRATION_CELLS = {
    ("C0", "P-medium", 168900),
}


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
    checks = {
        "physical_records": left.records == right_records,
        "source_bytes": left.source_bytes == right.source_bytes,
        "construction_trace": left.construction_trace == right.construction_trace,
        "prf_choices": left.prf_choices == right.prf_trace,
        "assertion_ids": left.assertion_ids == [record.assertion_id for record in right.assertions],
        "logical_snapshot_checksum": left.logical_snapshot_checksum == _unprefix_checksum(right.logical_snapshot_checksum),
        "snapshot_id": left.snapshot_id == right.snapshot_id,
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
            "logical_snapshot_checksum": left.logical_snapshot_checksum,
        },
        "independent": {
            "entity_count": len(right.entities),
            "assertion_count": len(right.assertions),
            "source_revision_count": len(right.source_revisions),
            "construction_trace_count": len(right.construction_trace),
            "prf_choice_count": len(right.prf_trace),
            "logical_snapshot_checksum": right.logical_snapshot_checksum,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="C0", choices=("C0",))
    parser.add_argument("--profile", default="P-medium", choices=("P-medium",))
    parser.add_argument("--seed", default=168900, type=int)
    args = parser.parse_args()
    evidence = compare_cell(args.tier, args.profile, args.seed)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
