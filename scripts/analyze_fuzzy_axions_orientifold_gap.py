#!/usr/bin/env python3
"""Classify the h11=2,3 gap to Sheridan et al. Table 1.

Read the corrected audit artifacts and compare their class-level counts with
the aggregate Table 1 targets from arXiv:2412.12012.  The artifacts retain
class identifiers for the code-certified ``h11_minus_zero`` subset and
fixed-surface diagnostics, but they do not retain every candidate terminal
record.  This module therefore reports two separate results:

* the exact aggregate Table 1 deficit; and
* a mutually exclusive classification of every FRST class that the audit did
  not certify, using only candidate-linked unresolved components retained in
  the artifact.

The second result is not a reconstruction of the paper's class membership.
Table 1 supplies counts, not class identifiers.  In particular,
``candidate_linked_unavailable`` means that an unresolved component with an
actual candidate identifier remains at the proof boundary (in these runs the
reason is ``non_smooth_ambient_cone``); it does not mean that the orientifold
is singular, and it does not establish a paper error.  Generic surface-attempt
rows and partial candidate contexts are annotations only.  This analysis
intentionally accepts only h11=2 and h11=3 artifacts; the superseded h11=4
artifact is out of scope.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


SUPPORTED_H11 = (2, 3)
DEFAULT_ARTIFACTS = {
    2: Path("/private/tmp/cyax-orientifold-rerun-h11-2-20260820.json"),
    3: Path("/private/tmp/cyax-orientifold-rerun-h11-3-20260820.json"),
}
REPRODUCTION_SCHEMA_VERSION = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1"
ANALYSIS_SCHEMA_VERSION = "cyaxiverse-fuzzy-axions-orientifold-gap-analysis-1.0"

TABLE_1_TARGETS = {
    2: {
        "favorable_polytopes": 36,
        "frst_classes": 36,
        "inherited_orientifold_cys": 32,
        "h11_minus_zero_orientifold_cys": 32,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 11,
        "models": 2,
    },
    3: {
        "favorable_polytopes": 243,
        "frst_classes": 274,
        "inherited_orientifold_cys": 253,
        "h11_minus_zero_orientifold_cys": 253,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 66,
        "models": 263,
    },
}

_COUNT_ALIASES = {
    "source_evidence_inherited_orientifold_cys": (
        "source_vertex_evidence_inherited_orientifold_cys",
    ),
    "source_evidence_h11_minus_zero_orientifold_cys": (
        "source_vertex_evidence_h11_minus_zero_orientifold_cys",
    ),
}


class ArtifactError(ValueError):
    """Raise when an audit artifact cannot support this comparison."""


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{field} must be an integer, got {value!r}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count(counts: dict[str, Any], name: str) -> int | None:
    value = counts.get(name)
    if value is not None:
        return _as_int(value, f"counts.{name}")
    for alias in _COUNT_ALIASES.get(name, ()):
        value = counts.get(alias)
        if value is not None:
            return _as_int(value, f"counts.{alias}")
    return None


def _reason_codes(attempts: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(attempt["reason_code"])
        for attempt in attempts
        if attempt.get("reason_code") not in (None, "")
    }


def _class_evidence(detail: dict[str, Any], class_index: int) -> dict[str, Any]:
    """Collect annotations and candidate-linked evidence for one class."""

    diagnostics = (
        detail.get("orientifold_action_audit", {})
        .get("reason_diagnostics")
        or {}
    )
    attempts = [
        attempt
        for attempt in diagnostics.get("surface_attempts", [])
        if _as_int(attempt.get("frst_class_index"), "frst_class_index")
        == class_index
    ]
    unresolved = [
        component
        for component in diagnostics.get("unresolved_components", [])
        if _as_int(component.get("frst_class_index"), "frst_class_index")
        == class_index
    ]

    terminal_statuses: Counter[str] = Counter()
    for attempt in attempts:
        context = attempt.get("candidate_context")
        if isinstance(context, dict) and context.get("candidate_terminal_status"):
            terminal_statuses[str(context["candidate_terminal_status"])] += 1
    for component in unresolved:
        status = component.get("candidate_terminal_status")
        if status:
            terminal_statuses[str(status)] += 1

    surface_statuses = Counter(
        str(attempt.get("status", "missing")) for attempt in attempts
    )
    reason_codes = _reason_codes(attempts) | _reason_codes(unresolved)
    candidate_linked_unresolved = [
        component
        for component in unresolved
        if component.get("candidate_id") not in (None, "")
        and component.get("reason_code") not in (None, "")
    ]

    return {
        "surface_attempt_count": len(attempts),
        "unresolved_component_count": len(unresolved),
        "surface_status_counts": dict(sorted(surface_statuses.items())),
        "candidate_terminal_status_counts": dict(sorted(terminal_statuses.items())),
        "reason_codes": sorted(reason_codes),
        "candidate_linked_unavailable": bool(candidate_linked_unresolved),
        "candidate_linked_unavailable_component_count": len(
            candidate_linked_unresolved
        ),
        "candidate_linked_candidate_ids": sorted(
            {
                str(component["candidate_id"])
                for component in candidate_linked_unresolved
            }
        ),
    }


def _class_record(
    detail: dict[str, Any], class_index: int, accepted: bool
) -> dict[str, Any]:
    polytope_index = _as_int(detail.get("polytope_index"), "polytope_index")
    evidence = _class_evidence(detail, class_index)
    if accepted:
        category = "certified_inherited"
        reason = "accepted_verified_orientifold"
    elif evidence["candidate_linked_unavailable"]:
        category = "unaccepted_with_candidate_linked_unavailable"
        reason = (
            "an unresolved fixed component is linked to an actual candidate; "
            "the source-matched proof boundary is not a singularity determination "
            "or a paper-error finding"
        )
    else:
        category = "unaccepted_not_classified_by_retained_terminal_ledger"
        reason = (
            "no candidate-linked unresolved component is retained; generic surface "
            "attempts and partial candidate contexts are annotations, not an "
            "exhaustive candidate verdict"
        )

    return {
        "polytope_index": polytope_index,
        "frst_class_index": int(class_index),
        "category": category,
        "reason": reason,
        **evidence,
    }


def _validate_artifact(data: dict[str, Any], expected_h11: int, path: Path) -> None:
    if expected_h11 not in SUPPORTED_H11:
        raise ArtifactError(
            f"only corrected h11=2,3 artifacts are supported; got h11={expected_h11}"
        )
    if data.get("schema_version") != REPRODUCTION_SCHEMA_VERSION:
        raise ArtifactError(
            f"{path}: unsupported reproduction schema {data.get('schema_version')!r}"
        )
    input_data = data.get("input")
    if not isinstance(input_data, dict):
        raise ArtifactError(f"{path}: missing input metadata")
    actual_h11 = _as_int(input_data.get("requested_h11"), "input.requested_h11")
    if actual_h11 != expected_h11:
        raise ArtifactError(
            f"{path}: requested h11={actual_h11}, expected h11={expected_h11}"
        )
    if not input_data.get("population_complete"):
        raise ArtifactError(f"{path}: input population is not complete")
    targets = TABLE_1_TARGETS[expected_h11]
    details = data.get("details")
    if not isinstance(details, list):
        raise ArtifactError(f"{path}: details must be a list")
    if len(details) != targets["favorable_polytopes"]:
        raise ArtifactError(
            f"{path}: details has {len(details)} polytopes, expected "
            f"{targets['favorable_polytopes']}"
        )
    counts = data.get("counts")
    if not isinstance(counts, dict):
        raise ArtifactError(f"{path}: missing counts")
    paper_targets = data.get("paper_targets")
    if not isinstance(paper_targets, dict):
        raise ArtifactError(f"{path}: missing paper_targets")
    for field, expected in targets.items():
        if _as_int(paper_targets.get(field), f"paper_targets.{field}") != expected:
            raise ArtifactError(
                f"{path}: paper_targets.{field} does not match the expected "
                f"h11={expected_h11} Table 1 target"
            )
    favorable_count = _as_int(
        counts.get("favorable_polytopes"), "counts.favorable_polytopes"
    )
    if favorable_count != len(details):
        raise ArtifactError(
            f"{path}: counts.favorable_polytopes={favorable_count}, but details "
            f"contains {len(details)} polytopes"
        )
    total_classes = sum(
        _as_int(detail.get("frst_class_count"), "frst_class_count")
        for detail in details
    )
    reported_classes = _as_int(counts.get("frst_classes"), "counts.frst_classes")
    if total_classes != reported_classes:
        raise ArtifactError(
            f"{path}: details sum to {total_classes} FRST classes, but counts says "
            f"{reported_classes}"
        )
    diagnostics = data.get("orientifold_reason_diagnostics")
    if diagnostics is not None and _as_int(
        diagnostics.get("h11"), "diagnostic.h11"
    ) != expected_h11:
        raise ArtifactError(f"{path}: diagnostic h11 does not match requested h11")


def _class_level_audit(data: dict[str, Any], h11: int) -> dict[str, Any]:
    details = data["details"]
    accepted_ids: set[tuple[int, int]] = set()
    class_records: list[dict[str, Any]] = []
    for detail in details:
        polytope_index = _as_int(detail.get("polytope_index"), "polytope_index")
        class_count = _as_int(detail.get("frst_class_count"), "frst_class_count")
        audit = detail.get("orientifold_action_audit") or {}
        listed = audit.get("h11_minus_zero_classes") or []
        listed_ints = [_as_int(index, "h11_minus_zero_class") for index in listed]
        if len(set(listed_ints)) != len(listed_ints):
            raise ArtifactError(
                f"h11={h11}, polytope={polytope_index}: duplicate accepted class id"
            )
        if any(index < 0 or index >= class_count for index in listed_ints):
            raise ArtifactError(
                f"h11={h11}, polytope={polytope_index}: accepted class id out of range"
            )
        accepted_ids.update((polytope_index, index) for index in listed_ints)

    inherited_count = _count(
        data["counts"], "source_evidence_inherited_orientifold_cys"
    )
    h11_zero_count = _count(
        data["counts"], "source_evidence_h11_minus_zero_orientifold_cys"
    )
    if inherited_count is None or h11_zero_count is None:
        raise ArtifactError("artifact lacks source-evidence orientifold counts")
    if len(accepted_ids) != h11_zero_count:
        raise ArtifactError(
            f"h11={h11}: accepted class ids number {len(accepted_ids)}, but count is "
            f"{h11_zero_count}"
        )
    if inherited_count != h11_zero_count:
        raise ArtifactError(
            f"h11={h11}: this artifact-only class partition requires inherited "
            "and h11-minus-zero counts to agree because only "
            "h11_minus_zero_classes identifiers are retained"
        )

    for detail in details:
        class_count = _as_int(detail.get("frst_class_count"), "frst_class_count")
        polytope_index = _as_int(detail.get("polytope_index"), "polytope_index")
        for class_index in range(class_count):
            class_records.append(
                _class_record(
                    detail,
                    class_index,
                    (polytope_index, class_index) in accepted_ids,
                )
            )

    category_counts = Counter(
        record["category"]
        for record in class_records
        if record["category"] != "certified_inherited"
    )
    total_classes = len(class_records)
    if sum(category_counts.values()) + len(accepted_ids) != total_classes:
        raise ArtifactError("class-level categories are not mutually exhaustive")

    inherited_ids_available = inherited_count == len(accepted_ids)
    return {
        "total_frst_class_count": total_classes,
        "certified_class_count": len(accepted_ids),
        "certified_class_ids": (
            [
                {"polytope_index": polytope, "frst_class_index": class_index}
                for polytope, class_index in sorted(accepted_ids)
            ]
            if inherited_ids_available
            else None
        ),
        "certified_class_id_basis": (
            "h11_minus_zero_classes_equal_inherited_count"
            if inherited_ids_available
            else "inherited_class_ids_not_retained"
        ),
        "code_unaccepted_class_count": total_classes - len(accepted_ids),
        "category_counts": dict(sorted(category_counts.items())),
        "unaccepted_class_records": [
            record
            for record in class_records
            if record["category"] != "certified_inherited"
        ],
        "candidate_terminal_status_coverage": (
            "partial: artifacts retain surface attempts and some candidate "
            "contexts, not every candidate terminal record"
        ),
    }


def _comparison(target: int, code: int) -> dict[str, Any]:
    difference = int(target) - int(code)
    return {
        "table_1_target": int(target),
        "code_output": int(code),
        "target_minus_code": difference,
        "target_gap_count": max(difference, 0),
        "code_exceeds_target_count": max(-difference, 0),
        "status": (
            "matches"
            if difference == 0
            else "lower_bound_gap"
            if difference > 0
            else "code_exceeds_target"
        ),
    }


def _conditional_ceiling(
    target: int, code: int, candidate_linked_unavailable_count: int
) -> dict[str, Any]:
    """Compute the conditional upgrade ceiling without extrapolating search scope."""

    ceiling = int(code) + int(candidate_linked_unavailable_count)
    return {
        "certified_code_count": int(code),
        "candidate_linked_unavailable_class_count": int(
            candidate_linked_unavailable_count
        ),
        "conditional_ceiling_count": ceiling,
        "table_1_target": int(target),
        "conditional_ceiling_deficit": max(int(target) - ceiling, 0),
        "holding_fixed": (
            "all other retained candidate verdicts, candidate search scope, and "
            "the current evidence boundary"
        ),
        "interpretation": (
            "conditional accounting only; it is not a prediction that every "
            "candidate-linked unavailable class will be accepted"
        ),
    }


def analyze_artifact(
    data: dict[str, Any],
    h11: int,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic comparison for one corrected h11 artifact."""

    path = source_path or Path("<in-memory-artifact>")
    _validate_artifact(data, h11, path)
    targets = TABLE_1_TARGETS[h11]
    counts = data["counts"]
    class_level = _class_level_audit(data, h11)
    inherited_count = _count(counts, "source_evidence_inherited_orientifold_cys")
    h11_zero_count = _count(counts, "source_evidence_h11_minus_zero_orientifold_cys")
    h21_count = _as_int(
        counts.get("h21_plus_zero_trilayer_frst_classes"),
        "counts.h21_plus_zero_trilayer_frst_classes",
    )
    favorable_count = _as_int(
        counts.get("favorable_polytopes"), "counts.favorable_polytopes"
    )
    frst_count = _as_int(counts.get("frst_classes"), "counts.frst_classes")
    comparisons = {
        "favorable_polytopes": _comparison(
            targets["favorable_polytopes"], favorable_count
        ),
        "frst_classes": _comparison(targets["frst_classes"], frst_count),
        "inherited_orientifold_cys": _comparison(
            targets["inherited_orientifold_cys"], inherited_count
        ),
        "h11_minus_zero_orientifold_cys": _comparison(
            targets["h11_minus_zero_orientifold_cys"], h11_zero_count
        ),
        "h11_minus_zero_h21_plus_zero_orientifold_cys": _comparison(
            targets["h11_minus_zero_h21_plus_zero_orientifold_cys"],
            h21_count,
        ),
    }
    for key in ("inherited_orientifold_cys", "h11_minus_zero_orientifold_cys"):
        candidate_linked_count = class_level["category_counts"].get(
            "unaccepted_with_candidate_linked_unavailable", 0
        )
        comparisons[key].update(
            {
                "target_gap_class_ids": None,
                "target_gap_class_id_status": "aggregate_only_table_1_has_no_class_ids",
                "class_level_code_accounting": class_level["category_counts"],
                "class_level_code_unaccepted_count": class_level[
                    "code_unaccepted_class_count"
                ],
                "conditional_ceiling": _conditional_ceiling(
                    targets[key],
                    comparisons[key]["code_output"],
                    candidate_linked_count,
                ),
            }
        )

    model_stage = data.get("model_stage")
    diagnostic_data = data.get("orientifold_reason_diagnostics") or {}
    return {
        "h11": h11,
        "artifact": {
            "path": str(path),
            "sha256": _sha256(path) if path.exists() else None,
            "schema_version": data["schema_version"],
        },
        "population": {
            "population_complete": bool(data["input"]["population_complete"]),
            "favorable_polytopes": comparisons["favorable_polytopes"],
            "frst_classes": comparisons["frst_classes"],
        },
        "orientifold_comparison": comparisons,
        "class_level_audit": class_level,
        "fixed_surface_diagnostics": {
            "surface_attempt_count": diagnostic_data.get("surface_attempt_count"),
            "certified_surface_count": diagnostic_data.get("certified_surface_count"),
            "skipped_surface_count": diagnostic_data.get("skipped_surface_count"),
            "skip_reason_counts": diagnostic_data.get("reason_counts", {}),
            "unresolved_candidate_component_count": diagnostic_data.get(
                "unresolved_candidate_component_count"
            ),
            "interpretation": (
                "diagnostic rows are evidence attempts, not accepted orientifold "
                "classes; surface statuses are annotations, and only unresolved "
                "components linked to a candidate enter the class partition"
            ),
        },
        "model_count": {
            "table_1_target": targets["models"],
            "code_output": None,
            "status": (
                "not_run_in_audit_artifact"
                if model_stage is None
                else "present_but_not_classified"
            ),
        },
        "interpretation": {
            "target_gap_is_aggregate": True,
            "target_gap_class_ids_are_not_in_table_1": True,
            "candidate_linked_unavailable_is_not_singularity": True,
            "candidate_linked_unavailable_is_not_paper_error": True,
            "surface_attempts_are_not_exhaustive_candidate_verdicts": True,
            "h11_4_status": "excluded_superseded_artifact",
        },
    }


def load_artifact(path: str | Path, expected_h11: int) -> dict[str, Any]:
    """Load and validate one corrected audit JSON artifact."""

    if expected_h11 not in SUPPORTED_H11:
        raise ArtifactError("this analysis accepts h11=2 and h11=3 only")
    resolved = Path(path).expanduser().resolve()
    try:
        with resolved.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"{resolved}: top-level JSON value must be an object")
    _validate_artifact(data, expected_h11, resolved)
    return data


def analyze_paths(paths: dict[int, str | Path]) -> dict[str, Any]:
    """Analyze exactly the corrected h11=2 and h11=3 artifacts."""

    if set(paths) != set(SUPPORTED_H11):
        raise ArtifactError("paths must contain exactly h11=2 and h11=3")
    analyses = []
    for h11 in SUPPORTED_H11:
        path = Path(paths[h11]).expanduser().resolve()
        data = load_artifact(path, h11)
        analyses.append(analyze_artifact(data, h11, source_path=path))
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "scope": {
            "h11": list(SUPPORTED_H11),
            "excluded_h11": [4],
            "source": "Sheridan et al. arXiv:2412.12012 Table 1",
            "audit_mode": "corrected_artifact_only",
        },
        "analyses": analyses,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11-2", type=Path, default=DEFAULT_ARTIFACTS[2])
    parser.add_argument("--h11-3", type=Path, default=DEFAULT_ARTIFACTS[3])
    parser.add_argument("--output", type=Path, help="write the same JSON to this path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = analyze_paths({2: args.h11_2, 3: args.h11_3})
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise ArtifactError(f"refusing to overwrite existing output {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
    except (ArtifactError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
