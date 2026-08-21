#!/usr/bin/env python3
"""Combine per-shard ``reproduce_fuzzy_axions_h11_4.py`` outputs into one summary.

A sharded run (``--shard-count N``) partitions the favorable-polytope
population deterministically across ``N`` independent processes, each writing
its own reproduction JSON and terminal ledger. Because the shards cover
disjoint geometries, the population totals are exactly additive and the
terminal-ledger class funnels are disjoint by geometry identity
(``polytope_normal_form_id``). This tool re-assembles those pieces into a
single merged summary and re-evaluates population completeness and Table 1
claim status against the combined totals.

It intentionally imports nothing from the heavy CYTools/HDF5 stack: it reads
only the compact shard summaries, so a merge is fast and runs anywhere.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

MERGE_SCHEMA_VERSION = "cyaxiverse-orientifold-shard-merge-1.0"

# Population counts that are additive across disjoint polytope shards.
_ADDITIVE_COUNT_KEYS = (
    "favorable_polytopes",
    "raw_frsts",
    "frst_classes",
    "raw_trilayer_polytopes",
    "raw_trilayer_frst_classes",
    "nonfrozen_trilayer_frst_classes",
    "h21_plus_zero_trilayer_frst_classes",
    "identity_torus_action_count",
    "identity_torus_action_cy_count",
    "identity_valid_o3o7_action_cy_count",
    "source_evidence_inherited_orientifold_cys",
    "source_evidence_h11_minus_zero_orientifold_cys",
    "source_vertex_evidence_inherited_orientifold_cys",
    "source_vertex_evidence_h11_minus_zero_orientifold_cys",
    "kaehler_point_export_accepted_count",
    "kaehler_point_export_rejected_count",
)


class ShardMergeError(ValueError):
    """Raise when shard summaries are inconsistent or cannot be merged."""


def _sum_optional(values: list[Any]) -> Any:
    """Sum a per-shard count, preserving ``None`` when every shard is ``None``.

    A count is ``None`` for a shard when the corresponding stage was not run
    (e.g. no ``--orientifold-audit``). Mixing ``None`` and integers means the
    shards were run with different flags, which is a merge error.
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ShardMergeError(
            "a count is present in some shards but null in others; shards were "
            "run with inconsistent flags"
        )
    return int(sum(int(value) for value in present))


def merge_shard_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge loaded shard reproduction summaries into one combined summary."""
    if len(summaries) < 2:
        raise ShardMergeError("merging requires at least two shard summaries")

    shard_blocks = []
    for summary in summaries:
        shard = summary.get("input", {}).get("shard")
        if not isinstance(shard, dict) or not shard.get("is_sharded"):
            raise ShardMergeError(
                "every input must be a sharded run (--shard-count > 1); found a "
                "summary with no shard block"
            )
        shard_blocks.append(shard)

    shard_count = shard_blocks[0]["count"]
    total_favorable = shard_blocks[0]["total_favorable_polytopes"]
    indices = sorted(int(block["index"]) for block in shard_blocks)
    for block in shard_blocks:
        if block["count"] != shard_count:
            raise ShardMergeError("shards disagree on --shard-count")
        if block["total_favorable_polytopes"] != total_favorable:
            raise ShardMergeError("shards disagree on total_favorable_polytopes")
    if indices != sorted(set(indices)):
        raise ShardMergeError(f"duplicate shard indices: {indices}")
    if indices != list(range(shard_count)):
        raise ShardMergeError(
            f"expected shard indices {list(range(shard_count))}, got {indices}; "
            "cannot merge an incomplete set of shards"
        )

    requested_h11 = summaries[0].get("input", {}).get("requested_h11")
    source_commits = {
        summary.get("run_provenance", {}).get("source_commit") for summary in summaries
    }
    if len(source_commits) != 1:
        raise ShardMergeError(f"shards come from different source commits: {source_commits}")
    for summary in summaries:
        if summary.get("input", {}).get("requested_h11") != requested_h11:
            raise ShardMergeError("shards disagree on requested_h11")

    merged_counts: dict[str, Any] = {}
    for key in _ADDITIVE_COUNT_KEYS:
        merged_counts[key] = _sum_optional(
            [summary.get("counts", {}).get(key) for summary in summaries]
        )
    if merged_counts["favorable_polytopes"] != total_favorable:
        raise ShardMergeError(
            f"summed shard favorable_polytopes ({merged_counts['favorable_polytopes']}) "
            f"!= declared total_favorable_polytopes ({total_favorable}); shards overlap "
            "or are missing"
        )

    merged_ledger = _merge_terminal_ledgers(
        [summary.get("terminal_ledger") for summary in summaries]
    )

    targets = summaries[0].get("paper_targets")
    population_complete = bool(
        targets is not None
        and merged_counts["favorable_polytopes"] == targets["favorable_polytopes"]
    )
    claim_status = _merged_claim_status(merged_counts, targets, population_complete)

    return {
        "schema_version": MERGE_SCHEMA_VERSION,
        "requested_h11": requested_h11,
        "source_commit": next(iter(source_commits)),
        "shard_count": shard_count,
        "shards": [
            {
                "index": int(block["index"]),
                "shard_favorable_polytopes": int(block["shard_favorable_polytopes"]),
                "terminal_ledger": (summary.get("terminal_ledger") or {}).get("sidecar_path"),
                "terminal_ledger_sha256": (summary.get("terminal_ledger") or {}).get(
                    "sidecar_sha256"
                ),
            }
            for block, summary in sorted(
                zip(shard_blocks, summaries), key=lambda pair: int(pair[0]["index"])
            )
        ],
        "counts": merged_counts,
        "population_complete": population_complete,
        "population_completion_reason": (
            f"merged {merged_counts['favorable_polytopes']} favorable polytopes vs "
            f"target {targets['favorable_polytopes']}"
            if targets is not None
            else f"no Table 1 target recorded for h11={requested_h11}"
        ),
        "paper_targets": targets,
        "claim_status": claim_status,
        "terminal_ledger": merged_ledger,
        "table_1_accepted_class_count": merged_ledger["table_1_accepted_class_count"],
    }


def _merge_terminal_ledgers(ledgers: list[Any]) -> dict[str, Any]:
    """Union disjoint per-shard terminal-ledger summaries."""
    if any(ledger is None for ledger in ledgers):
        raise ShardMergeError(
            "a shard has no terminal_ledger summary; rerun it with --orientifold-audit"
        )
    status_counts: Counter[str] = Counter()
    record_kind_counts: Counter[str] = Counter()
    record_count = 0
    class_funnel: dict[tuple, dict[str, Any]] = {}
    for ledger in ledgers:
        record_count += int(ledger.get("record_count", 0))
        for status, count in (ledger.get("terminal_status_counts") or {}).items():
            status_counts[status] += int(count)
        for kind, count in (ledger.get("record_kind_counts") or {}).items():
            record_kind_counts[kind] += int(count)
        for record in ledger.get("class_funnel") or []:
            key = (record.get("polytope_normal_form_id"), record.get("frst_class_index"))
            if key in class_funnel:
                raise ShardMergeError(
                    f"class {key} appears in more than one shard; shards are not "
                    "disjoint by geometry"
                )
            class_funnel[key] = record
    ordered = [class_funnel[key] for key in sorted(class_funnel, key=lambda item: (str(item[0]), item[1]))]
    accepted = sum(1 for record in ordered if record.get("accepted_for_table_1"))
    return {
        "record_count": record_count,
        "record_kind_counts": dict(sorted(record_kind_counts.items())),
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "class_count": len(ordered),
        "table_1_accepted_class_count": accepted,
        "class_funnel": ordered,
    }


def _merged_claim_status(
    counts: dict[str, Any], targets: Any, population_complete: bool
) -> dict[str, Any]:
    if targets is None:
        return {"favorable_polytopes": None, "frst_classes": None, "h21_plus_zero": None}
    return {
        "favorable_polytopes": (
            "exact" if counts["favorable_polytopes"] == targets["favorable_polytopes"] else "mismatch"
        ),
        "frst_classes": (
            "exact" if counts["frst_classes"] == targets["frst_classes"] else "mismatch"
        ),
        "h21_plus_zero": (
            "benchmark_match_candidate"
            if population_complete
            and counts["h21_plus_zero_trilayer_frst_classes"]
            == targets["h11_minus_zero_h21_plus_zero_orientifold_cys"]
            else "diagnostic_only"
        ),
    }


def _load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            summaries.append(json.load(stream))
    return summaries


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shard_outputs",
        nargs="+",
        type=Path,
        help="Per-shard reproduction JSON outputs (the --output files) to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the merged summary JSON (refuses to overwrite).",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing merged output: {args.output}")
    merged = merge_shard_summaries(_load_summaries(args.shard_outputs))
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output} "
        f"(favorable_polytopes={merged['counts']['favorable_polytopes']}, "
        f"frst_classes={merged['counts']['frst_classes']}, "
        f"table_1_accepted_class_count={merged['table_1_accepted_class_count']}, "
        f"population_complete={merged['population_complete']})"
    )


if __name__ == "__main__":
    main()
