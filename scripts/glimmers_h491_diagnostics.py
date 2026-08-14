#!/usr/bin/env python3
"""Account for bounded native h11=491 NTFE diagnostic stages.

The module accepts already-recorded candidate-stage facts.  It never imports
CYTools and never invokes a sampler.  The default fixture is a deterministic
87-candidate regression fixture with 74 Kähler-tip failures and 13 numerical
failures, matching the bounded observation supplied with the Glimmers task.
That observation is retained as a regression baseline only; this report does
not diagnose NTFE as defective or recommend changing its settings.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


H11 = 491
STAGE_NAMES = (
    "ntfe_yield",
    "frst_validity",
    "kahler_cone_construction",
    "stretched_cone_feasibility",
    "tip_convergence",
    "volume_filters",
    "numerical_residuals",
)

STAGE_OUTCOMES = {
    "ntfe_yield": ("yielded", "generation_failure", "not_yielded", "not_recorded"),
    "frst_validity": ("valid", "invalid", "not_run", "not_recorded"),
    "kahler_cone_construction": ("constructed", "failure", "not_run", "not_recorded"),
    "stretched_cone_feasibility": ("feasible", "infeasible", "not_run", "not_recorded"),
    "tip_convergence": ("converged", "failure", "not_run", "not_recorded"),
    "volume_filters": ("passed", "rejected", "not_run", "not_recorded"),
    "numerical_residuals": ("passed", "failed", "not_run", "not_recorded"),
}

TERMINAL_STATUSES = (
    "accepted_geometry",
    "proposal_budget_exhausted",
    "geometry_target_shortfall",
    "ntfe_generation_failure",
    "duplicate_ntfe_identity",
    "invalid_frst",
    "kahler_cone_failure",
    "kahler_tip_failure",
    "volume_filter_rejection",
    "numerical_geometry_failure",
    "assignment_pool_shortfall",
    "output_collision",
    "user_decision_required",
)

OBSERVED_H491_BASELINE = {
    "candidate_proposals": 87,
    "terminal_status_counts": {
        "kahler_tip_failure": 74,
        "numerical_geometry_failure": 13,
    },
    "accepted_geometry": 0,
    "source_label": "bounded native h11=491 production observation",
}


@dataclass(frozen=True)
class NativeH491SamplerSettings:
    """Preserve the approved native NTFE settings in every report."""

    sampler_name: str = "ntfe_fast"
    backend: str = "cgal"
    make_star: bool = True
    triang_method: str = "fast"
    ntfe_face_sampler: str = "fast"
    ntfe_max_face_points: int = 17
    ntfe_face_pool_size: int = 1000
    ntfe_as_generator: bool = True
    proposal_budget: int = 100
    accepted_target: int = 100
    deterministic_seed: int = 20260813
    retry_budget: int = 500

    def validate(self) -> None:
        """Reject settings that change the approved native h11=491 route."""
        if self.sampler_name != "ntfe_fast":
            raise ValueError("h11=491 diagnostics require the native ntfe_fast sampler")
        if self.backend != "cgal":
            raise ValueError("h11=491 diagnostics preserve the approved cgal backend")
        if not self.make_star:
            raise ValueError("h11=491 diagnostics require make_star=True")
        if self.triang_method != "fast" or self.ntfe_face_sampler != "fast":
            raise ValueError("native h11=491 NTFE requires the fast face sampler")
        if self.ntfe_max_face_points != 17 or self.ntfe_face_pool_size != 1000:
            raise ValueError("native h11=491 NTFE settings must retain max_npts=17 and N_face_triangs=1000")
        if not self.ntfe_as_generator:
            raise ValueError("native h11=491 NTFE requires as_generator=True")
        if self.proposal_budget < 0 or self.retry_budget < 0 or self.accepted_target < 0:
            raise ValueError(
                "proposal_budget, retry_budget, and accepted_target must be non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the sampler contract using generator-compatible names."""
        self.validate()
        payload = asdict(self)
        payload.update(
            {
                "scheme": self.sampler_name,
                "N": self.proposal_budget,
                "max_npts": self.ntfe_max_face_points,
                "N_face_triangs": self.ntfe_face_pool_size,
                "as_generator": self.ntfe_as_generator,
                "sampling_unit": "two_face_inequivalent_frst",
                "forbidden_substitutions": [
                    "random_triangulations_fast",
                    "random_triangulations_gnn",
                    "dualGNN",
                    "PyTorch",
                ],
            }
        )
        return payload


@dataclass(frozen=True)
class H491DiagnosticReport:
    """Represent stage counts and baseline comparison for one bounded report."""

    settings: NativeH491SamplerSettings
    candidate_records: tuple[dict[str, Any], ...]
    stage_counts: dict[str, dict[str, int]]
    terminal_status_counts: dict[str, int]
    accepted_count: int
    baseline_comparison: dict[str, Any]
    shortfall: dict[str, Any]

    def to_dict(self, include_records: bool = True) -> dict[str, Any]:
        """Serialize the diagnostic report without live CYTools objects."""
        report = {
            "report_schema": "glimmers-h11-491-stage-diagnostics-1",
            "h11": H11,
            "target_population": (
                "two-face-inequivalent FRST proposals of the unique favorable "
                "KS h11=491 polytope"
            ),
            "realised_sample": "bounded native NTFE diagnostic; not population-representative",
            "sampler": self.settings.to_dict(),
            "candidate_count": len(self.candidate_records),
            "accepted_count": self.accepted_count,
            "stage_counts": self.stage_counts,
            "terminal_status_counts": self.terminal_status_counts,
            "baseline": OBSERVED_H491_BASELINE,
            "baseline_comparison": self.baseline_comparison,
            "shortfall": self.shortfall,
            "interpretation_guard": (
                "The 87/74/13 counts are a regression baseline for stage accounting. "
                "They do not establish that native NTFE is defective and do not "
                "authorize changing NTFE, filters, population, or parity."
            ),
        }
        if include_records:
            report["candidate_records"] = [dict(record) for record in self.candidate_records]
        return report

    def write_json(self, path: str, include_records: bool = True) -> None:
        """Atomically write a bounded diagnostic report."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".glimmers-h491-diagnostics-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    self.to_dict(include_records=include_records),
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def validate_native_h491_settings(settings: NativeH491SamplerSettings) -> None:
    """Validate settings before accepting a native h11=491 diagnostic."""
    settings.validate()


def diagnose_h491(
    candidate_records: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    settings: NativeH491SamplerSettings | None = None,
    accepted_target: int | None = None,
) -> H491DiagnosticReport:
    """Account for explicit stage facts from a bounded h11=491 candidate set."""
    settings = settings or NativeH491SamplerSettings()
    settings.validate()
    if isinstance(candidate_records, Mapping):
        candidate_records = candidate_records.get(
            "candidate_records", candidate_records.get("candidates", [])
        )
        if not isinstance(candidate_records, Iterable) or isinstance(
            candidate_records, (str, bytes)
        ):
            raise ValueError("diagnostic mapping must contain candidate records")
    normalized_records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for ordinal, raw_record in enumerate(candidate_records, start=1):
        if not isinstance(raw_record, Mapping):
            raise ValueError("each h11=491 diagnostic record must be a mapping")
        record = _normalize_record(raw_record, ordinal)
        if record["candidate_index"] in seen_indices:
            raise ValueError(f"duplicate candidate_index {record['candidate_index']}")
        seen_indices.add(record["candidate_index"])
        normalized_records.append(record)
    if len(normalized_records) > settings.proposal_budget:
        raise ValueError(
            "diagnostic candidate count exceeds the configured proposal_budget: "
            f"{len(normalized_records)} > {settings.proposal_budget}"
        )

    stage_counts = {
        stage: {outcome: 0 for outcome in STAGE_OUTCOMES[stage]}
        for stage in STAGE_NAMES
    }
    terminal_counts: Counter[str] = Counter()
    accepted_count = 0
    for record in normalized_records:
        for stage in STAGE_NAMES:
            outcome = record["stages"][stage]
            stage_counts[stage][outcome] += 1
        terminal = record["terminal_status"]
        terminal_counts[terminal] += 1
        if terminal == "accepted_geometry":
            accepted_count += 1

    baseline_comparison = _compare_with_baseline(
        len(normalized_records), dict(terminal_counts), accepted_count
    )
    target = settings.accepted_target if accepted_target is None else int(accepted_target)
    if target < 0:
        raise ValueError("accepted_target must be non-negative")
    shortfall = {
        "accepted_target": target,
        "accepted_count": accepted_count,
        "missing": max(0, target - accepted_count),
        "terminal_status": (
            "accepted_geometry" if accepted_count >= target else "geometry_target_shortfall"
        ),
        "budget_status": (
            None
            if accepted_count >= target
            else "proposal_budget_exhausted"
            if len(normalized_records) >= settings.proposal_budget
            else "bounded_diagnostic_source_exhausted"
        ),
        "proposal_budget": settings.proposal_budget,
    }
    return H491DiagnosticReport(
        settings,
        tuple(normalized_records),
        stage_counts,
        dict(sorted(terminal_counts.items())),
        accepted_count,
        baseline_comparison,
        shortfall,
    )


def make_h491_regression_fixture() -> list[dict[str, Any]]:
    """Build the bounded 87-candidate 74/13 stage-accounting fixture."""
    records = []
    for candidate_index in range(1, 88):
        tip_failed = candidate_index <= 74
        records.append(
            {
                "h11": H11,
                "candidate_index": candidate_index,
                "proposal_seed": 20260813 + candidate_index - 1,
                "terminal_status": (
                    "kahler_tip_failure" if tip_failed else "numerical_geometry_failure"
                ),
                "stages": {
                    "ntfe_yield": "yielded",
                    "frst_validity": "valid",
                    "kahler_cone_construction": "constructed",
                    "stretched_cone_feasibility": "feasible",
                    "tip_convergence": "failure" if tip_failed else "converged",
                    "volume_filters": "not_run" if tip_failed else "passed",
                    "numerical_residuals": "not_run" if tip_failed else "failed",
                },
                "fixture": True,
            }
        )
    return records


def diagnose_h491_regression_fixture(
    settings: NativeH491SamplerSettings | None = None,
) -> H491DiagnosticReport:
    """Run the bounded fixture and require the observed 87/74/13 baseline."""
    report = diagnose_h491(
        make_h491_regression_fixture(), settings=settings, accepted_target=100
    )
    comparison = report.baseline_comparison
    if not comparison["matches_all"]:
        raise AssertionError(f"h11=491 regression baseline mismatch: {comparison}")
    return report


def _normalize_record(raw: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    h11 = raw.get("h11", H11)
    if isinstance(h11, bool) or int(h11) != H11:
        raise ValueError(f"diagnostic record {ordinal} is not h11=491")
    candidate_index = raw.get("candidate_index", ordinal)
    if isinstance(candidate_index, bool) or int(candidate_index) < 1:
        raise ValueError("candidate_index must be a positive integer")
    terminal = str(raw.get("terminal_status", "numerical_geometry_failure"))
    terminal = terminal.strip().lower().replace("-", "_").replace(" ", "_")
    # The current generator uses this spelling; the task contract uses the
    # shorter canonical spelling below.
    if terminal == "kaehler_tip_failure":
        terminal = "kahler_tip_failure"
    terminal = {
        "sampler_retry_exhausted": "proposal_budget_exhausted",
        "topology_or_cone_error": "kahler_cone_failure",
        "divisor_volume_filter_rejection": "volume_filter_rejection",
        "qcd_normalization_failure": "volume_filter_rejection",
        "no_eligible_intersecting_qed_pair": "assignment_pool_shortfall",
    }.get(terminal, terminal)
    if terminal == "assignment_pool_failure":
        terminal = "assignment_pool_shortfall"
    if terminal not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported h11=491 terminal status {terminal!r}")
    source_stages = raw.get("stages", {})
    if not isinstance(source_stages, Mapping):
        raise ValueError("stages must be a mapping")
    stages = {
        stage: _stage_value(raw, source_stages, stage, terminal)
        for stage in STAGE_NAMES
    }
    normalized = {
        "h11": H11,
        "candidate_index": int(candidate_index),
        "terminal_status": terminal,
        "stages": stages,
    }
    for key in ("proposal_seed", "two_face_hash", "triangulation_hash", "reason", "fixture"):
        if key in raw:
            normalized[key] = _json_safe(raw[key])
    return normalized


_TOP_LEVEL_STAGE_KEYS = {
    "ntfe_yield": ("ntfe_yield", "ntfe_status", "ntfe_yielded"),
    "frst_validity": ("frst_validity", "frst_status", "frst_valid"),
    "kahler_cone_construction": (
        "kahler_cone_construction",
        "kahler_cone_status",
        "kahler_cone_constructed",
    ),
    "stretched_cone_feasibility": (
        "stretched_cone_feasibility",
        "stretched_cone_status",
        "stretched_cone_feasible",
    ),
    "tip_convergence": ("tip_convergence", "tip_status", "tip_converged"),
    "volume_filters": ("volume_filters", "volume_filter_status", "volume_filters_passed"),
    "numerical_residuals": (
        "numerical_residuals",
        "numerical_residual_status",
        "numerical_residuals_passed",
    ),
}


def _stage_value(
    raw: Mapping[str, Any], stages: Mapping[str, Any], stage: str, terminal: str
) -> str:
    value = None
    found = False
    for key in (stage, *_TOP_LEVEL_STAGE_KEYS[stage]):
        if key in stages:
            value = stages[key]
            found = True
            break
        if key in raw:
            value = raw[key]
            found = True
            break
    if found:
        return _normalize_stage_outcome(stage, value)
    return _infer_stage_outcome(stage, terminal)


def _normalize_stage_outcome(stage: str, value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("status", value.get("outcome"))
    if isinstance(value, bool):
        positive = {
            "ntfe_yield": "yielded",
            "frst_validity": "valid",
            "kahler_cone_construction": "constructed",
            "stretched_cone_feasibility": "feasible",
            "tip_convergence": "converged",
            "volume_filters": "passed",
            "numerical_residuals": "passed",
        }
        negative = {
            "ntfe_yield": "generation_failure",
            "frst_validity": "invalid",
            "kahler_cone_construction": "failure",
            "stretched_cone_feasibility": "infeasible",
            "tip_convergence": "failure",
            "volume_filters": "rejected",
            "numerical_residuals": "failed",
        }
        value = positive[stage] if value else negative[stage]
    if value is None:
        return "not_recorded"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ok": {
            "ntfe_yield": "yielded",
            "frst_validity": "valid",
            "kahler_cone_construction": "constructed",
            "stretched_cone_feasibility": "feasible",
            "tip_convergence": "converged",
            "volume_filters": "passed",
            "numerical_residuals": "passed",
        },
        "success": {
            "ntfe_yield": "yielded",
            "frst_validity": "valid",
            "kahler_cone_construction": "constructed",
            "stretched_cone_feasibility": "feasible",
            "tip_convergence": "converged",
            "volume_filters": "passed",
            "numerical_residuals": "passed",
        },
    }
    normalized = aliases.get(normalized, {}).get(stage, normalized)
    if normalized not in STAGE_OUTCOMES[stage]:
        raise ValueError(f"unsupported {stage} outcome {value!r}")
    return normalized


def _infer_stage_outcome(stage: str, terminal: str) -> str:
    """Infer only direct terminal evidence; leave earlier stages unrecorded."""
    if terminal == "ntfe_generation_failure":
        return "generation_failure" if stage == "ntfe_yield" else "not_run"
    if stage == "ntfe_yield":
        return "yielded"
    if terminal == "invalid_frst":
        return "invalid" if stage == "frst_validity" else "not_run"
    if terminal == "kahler_cone_failure":
        return "failure" if stage == "kahler_cone_construction" else "not_run"
    if terminal == "kahler_tip_failure":
        return "failure" if stage == "tip_convergence" else "not_recorded"
    if terminal == "volume_filter_rejection":
        return "rejected" if stage == "volume_filters" else "not_recorded"
    if terminal == "numerical_geometry_failure":
        return "failed" if stage == "numerical_residuals" else "not_recorded"
    return "not_recorded"


def _compare_with_baseline(
    candidate_count: int, terminal_counts: dict[str, int], accepted_count: int
) -> dict[str, Any]:
    expected = OBSERVED_H491_BASELINE
    terminal_comparison = {}
    for status, expected_count in expected["terminal_status_counts"].items():
        observed_count = terminal_counts.get(status, 0)
        terminal_comparison[status] = {
            "observed": observed_count,
            "baseline": expected_count,
            "matches": observed_count == expected_count,
        }
    return {
        "candidate_count": {
            "observed": candidate_count,
            "baseline": expected["candidate_proposals"],
            "matches": candidate_count == expected["candidate_proposals"],
        },
        "accepted_geometry": {
            "observed": accepted_count,
            "baseline": expected["accepted_geometry"],
            "matches": accepted_count == expected["accepted_geometry"],
        },
        "terminal_status_counts": terminal_comparison,
        "matches_all": (
            candidate_count == expected["candidate_proposals"]
            and accepted_count == expected["accepted_geometry"]
            and all(item["matches"] for item in terminal_comparison.values())
        ),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _load_records(path: str) -> list[Mapping[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, Mapping):
        payload = payload.get("candidate_records", payload.get("candidates", payload))
    if not isinstance(payload, list):
        raise ValueError("diagnostic input must be a JSON list or an object with candidates")
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON candidate-stage records; never invokes NTFE")
    parser.add_argument("--output", help="Optional atomic JSON report path")
    parser.add_argument(
        "--fixture", action="store_true", help="Use the bounded 87-candidate regression fixture"
    )
    parser.add_argument(
        "--omit-records", action="store_true", help="Omit per-candidate records from output"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run only the bounded in-process fixture checks"
    )
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("glimmers_h491_diagnostics self-test passed")
        return
    if args.input and args.fixture:
        parser.error("choose --input or --fixture, not both")
    if args.input:
        records = _load_records(args.input)
    else:
        records = make_h491_regression_fixture()
    report = diagnose_h491(records, accepted_target=100)
    payload = report.to_dict(include_records=not args.omit_records)
    if args.output:
        report.write_json(args.output, include_records=not args.omit_records)
        print(f"Wrote {args.output}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _self_test() -> None:
    """Run bounded diagnostic checks without invoking CYTools or NTFE."""
    settings = NativeH491SamplerSettings()
    settings.validate()
    report = diagnose_h491_regression_fixture(settings)
    assert len(report.candidate_records) == 87
    assert report.terminal_status_counts == {
        "kahler_tip_failure": 74,
        "numerical_geometry_failure": 13,
    }
    assert report.stage_counts["ntfe_yield"]["yielded"] == 87
    assert report.stage_counts["frst_validity"]["valid"] == 87
    assert report.stage_counts["kahler_cone_construction"]["constructed"] == 87
    assert report.stage_counts["stretched_cone_feasibility"]["feasible"] == 87
    assert report.stage_counts["tip_convergence"]["failure"] == 74
    assert report.stage_counts["tip_convergence"]["converged"] == 13
    assert report.stage_counts["volume_filters"]["passed"] == 13
    assert report.stage_counts["numerical_residuals"]["failed"] == 13
    assert report.baseline_comparison["matches_all"]
    assert report.shortfall["terminal_status"] == "geometry_target_shortfall"
    assert report.shortfall["missing"] == 100

    invalid_settings = NativeH491SamplerSettings(ntfe_face_sampler="random")
    try:
        invalid_settings.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("non-native face sampler was not rejected")


if __name__ == "__main__":
    _main()
