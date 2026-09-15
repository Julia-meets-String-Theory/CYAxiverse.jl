#!/usr/bin/env python3
"""Deterministic CYAX-0168 resource/performance classifier.

The classifier consumes caller-supplied calibration or benchmark summaries.
It does not open fixtures or infer semantics from backend labels.  All
resource and performance outcomes are explicit, allowing G1 conformance tests
to exercise the precedence and joint-CI rules with tiny synthetic mappings.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any, Callable, Iterable, Mapping, Sequence


class ClassifierError(ValueError):
    """Raised for malformed classifier inputs."""


INCONCLUSIVE = "Inconclusive / invalid execution"
ARCHITECTURE_PROBLEM = "Architecture problem"
OPERATIONAL_FAILURE = "Operational envelope failure / owner decision required"
RETAIN_S = "Retain S / G not justified"
HYBRID = "Hybrid"
GRAPH = "G (experimental derived-index status)"


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    lower: float
    upper: float
    censored: bool = False
    finite: bool = True

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | Sequence[float] | "ConfidenceInterval") -> "ConfidenceInterval":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            point = value.get("point", value.get("point_estimate"))
            lower = value.get("lower", value.get("lower95"))
            upper = value.get("upper", value.get("upper95"))
            if point is None or lower is None or upper is None:
                raise ClassifierError("confidence interval requires point/lower/upper")
            return cls(float(point), float(lower), float(upper), bool(value.get("censored", False)), bool(value.get("finite", True)))
        seq = tuple(value)
        if len(seq) != 3:
            raise ClassifierError("confidence interval sequence must be (point, lower, upper)")
        return cls(float(seq[0]), float(seq[1]), float(seq[2]))

    @property
    def crossing(self) -> bool:
        return not self.finite or self.censored


def capped_allowances(physical_ram_bytes: int, available_volume_capacity: int) -> dict[str, int]:
    if physical_ram_bytes <= 0 or available_volume_capacity <= 0:
        raise ClassifierError("resource denominators must be positive")
    return {
        "memory_excess_allowance": int(min(2 * 1024**3, 0.05 * physical_ram_bytes)),
        "disk_excess_allowance": int(min(2 * 1024**3, 0.05 * available_volume_capacity)),
    }


def memory_rel_g(g_peak_rss: int, s_peak_rss: int, physical_ram_bytes: int) -> bool:
    allowance = capped_allowances(physical_ram_bytes, 1)["memory_excess_allowance"]
    return g_peak_rss <= 2 * s_peak_rss or g_peak_rss - s_peak_rss <= allowance


def disk_rel_g(g_allocated: int, s_allocated: int, available_volume_capacity: int) -> bool:
    allowance = capped_allowances(1, available_volume_capacity)["disk_excess_allowance"]
    return g_allocated <= 2 * s_allocated or g_allocated - s_allocated <= allowance


def margin_distance(value: float, threshold: float) -> float:
    if not math.isfinite(value) or not math.isfinite(threshold) or threshold <= 0:
        raise ClassifierError("margin inputs must be finite and threshold-positive")
    return max(0.0, (threshold - value) / threshold)


def direct_observation_margin(statistics: Mapping[str, Any], *, thresholds: Mapping[str, float]) -> bool:
    """T4 condition-3 margin test over named point/lower-CI statistics."""
    for name, threshold in thresholds.items():
        if name not in statistics:
            return False
        value = statistics[name]
        if isinstance(value, Mapping):
            interval = ConfidenceInterval.from_value(value)
            value = interval.lower if name.startswith("lower") else interval.point
            if interval.censored or not interval.finite:
                return False
        try:
            if not math.isfinite(float(value)):
                return False
            if margin_distance(float(value), float(threshold)) > 0.25:
                return False
        except (TypeError, ValueError):
            return False
    return True


def t4_condition3_cells(cells: Iterable[Mapping[str, Any]], *, thresholds: Mapping[str, float]) -> bool:
    """Return true when at least one valid T3 cell meets the direct margin."""
    return any(bool(cell.get("valid", True)) and direct_observation_margin(cell, thresholds=thresholds) for cell in cells)


def project_monotone_upper_envelope(
    observations: Mapping[int, int | float | Decimal], *, n1: int = 50_000, n2: int = 500_000,
    n3: int = 1_000_000, n4: int = 5_000_000, headroom: Decimal | float = Decimal("1.25"),
) -> Decimal:
    """Frozen T4 projection for one resource/duration quantity."""
    if set(observations) != {1, 2, 3}:
        raise ClassifierError("T4 projection requires valid T1, T2, and T3 maxima")
    if not n1 < n2 < n3 < n4:
        raise ClassifierError("assertion-count scales must be strictly increasing")
    with localcontext() as ctx:
        ctx.prec = 50
        r1, r2, r3 = (Decimal(str(observations[index])) for index in (1, 2, 3))
        if any(value < 0 for value in (r1, r2, r3)):
            raise ClassifierError("projection observations must be nonnegative")
        e1, e2, e3 = r1, max(r1, r2), max(r1, r2, r3)
        slopes = (
            e1 / n1, e2 / n2, e3 / n3,
            (e2 - e1) / (n2 - n1), (e3 - e2) / (n3 - n2),
            (e3 - e1) / (n3 - n1),
        )
        slope = max(slopes)
        return Decimal(str(headroom)) * (e3 + slope * (n4 - n3))


def t4_headroom_pass(projected: Decimal | float, limit: Decimal | float) -> bool:
    """The exact ``R <= 0.75 L`` condition (equality passes)."""
    if float(limit) <= 0:
        raise ClassifierError("T4 hard limit must be positive")
    return Decimal(str(projected)) <= Decimal("0.75") * Decimal(str(limit))


def t4_condition4(
    projected_rows: Mapping[str, tuple[Decimal | float, Decimal | float]], *,
    required_categories_present: bool = True, censored: bool = False,
    prior_hard_breach: bool = False,
) -> dict[str, Any]:
    """Evaluate all projected T4 rows and return exact failure details."""
    failures = []
    if not required_categories_present: failures.append("missing-input")
    if censored: failures.append("censored-input")
    if prior_hard_breach: failures.append("prior-hard-breach")
    rows: dict[str, bool] = {}
    for name, (projected, limit) in projected_rows.items():
        rows[name] = t4_headroom_pass(projected, limit)
        if not rows[name]: failures.append(name)
    return {"pass": not failures, "rows": rows, "failures": tuple(failures)}


def resource_disposition(
    *, host_valid: bool = True, measurement_valid: bool = True,
    combined_cache_capacity_breach: bool = False, s_hard_breach: bool = False,
    g_hard_breach: bool = False, g_relative_breach: bool = False,
    s_passes: bool = True,
) -> str | None:
    """Return the resource-only classifier outcome before performance steps."""
    if not host_valid or not measurement_valid:
        return INCONCLUSIVE
    if combined_cache_capacity_breach or s_hard_breach:
        return OPERATIONAL_FAILURE
    if s_passes and (g_hard_breach or g_relative_breach):
        return RETAIN_S
    return None


def _get(cell: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in cell:
            return cell[name]
    return default


def _cell_pass(cell: Mapping[str, Any], *, graph: bool, critical: bool = False, relational: bool = False) -> bool:
    if not cell.get("valid", True) or cell.get("censored", False):
        return False
    if critical:
        t3_speed, t3_lower, t3_save = 5.0, 3.0, 50.0
    elif relational:
        t3_speed, t3_lower, t3_save = 2.0, 1.5, 10.0
    else:
        t3_speed, t3_lower, t3_save = 2.0, 1.5, 10.0
    # Caller may supply nested ``t3``/``t2`` maps or flat keys.
    t3 = _get(cell, "t3", default=cell)
    t2 = _get(cell, "t2", default=cell)
    speed = float(_get(t3, "speedup", "speedup_g_over_s", "point_speedup", default=float("nan")))
    lower = float(_get(t3, "lower95_speedup", "lower_speedup", "lower", default=float("nan")))
    saving = float(_get(t3, "absolute_saving", "saving", "saving_g_over_s", default=float("nan")))
    t2speed = float(_get(t2, "speedup", "speedup_g_over_s", "point_speedup", default=float("nan")))
    t2lower = float(_get(t2, "lower95_speedup", "lower_speedup", "lower", default=float("nan")))
    if relational:
        # Scell uses S-over-G; callers can provide explicit relational keys.
        speed = float(_get(t3, "s_over_g", "speedup_s_over_g", default=speed))
        lower = float(_get(t3, "lower95_s_over_g", "lower_s_over_g", default=lower))
        saving = float(_get(t3, "saving_s_over_g", "absolute_saving_s_over_g", default=saving))
        t2speed = float(_get(t2, "s_over_g", "speedup_s_over_g", default=t2speed))
        t2lower = float(_get(t2, "lower95_s_over_g", "lower_s_over_g", default=t2lower))
    t3_ok = speed >= t3_speed and lower >= t3_lower and saving >= t3_save
    t2_ok = t2speed > 1.0 and t2lower >= 1.0
    return bool(t3_ok and t2_ok)


def _normalize_cells(cells: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str, Mapping[str, Any]]]:
    result = []
    if isinstance(cells, Mapping):
        for key, cell in cells.items():
            if isinstance(key, tuple) and len(key) == 3:
                family, profile, mode = (str(part) for part in key)
            else:
                family = str(cell.get("family", "")); profile = str(cell.get("profile", "")); mode = str(cell.get("mode", cell.get("cache_mode", "")))
            result.append((family, profile, mode, cell))
    else:
        for cell in cells:
            result.append((str(cell.get("family", "")), str(cell.get("profile", "")), str(cell.get("mode", cell.get("cache_mode", ""))), cell))
    return result


def _performance_outcome(cells: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> str:
    normalized = _normalize_cells(cells)
    graph_families = {"Q03", "Q06", "Q07", "Q09"}
    relational_families = {"Q01", "Q04", "Q12"}
    profiles = ("P-low", "P-medium", "P-high")
    by_key = {(family, profile, mode): cell for family, profile, mode, cell in normalized}
    def cell_for(family: str, profile: str, mode: str) -> Mapping[str, Any] | None:
        return by_key.get((family, profile, mode))
    def graph_family(family: str) -> bool:
        return sum(all(cell_for(family, profile, mode) is not None and _cell_pass(cell_for(family, profile, mode) or {}, graph=True) for mode in ("fresh-process", "warm-process")) for profile in profiles) >= 2
    graph_arm = sum(graph_family(family) for family in graph_families) >= 2
    q07_arm = sum(all(cell_for("Q07", profile, mode) is not None and _cell_pass(cell_for("Q07", profile, mode) or {}, graph=True, critical=True) for mode in ("fresh-process", "warm-process")) for profile in profiles) >= 2
    graph_arm = graph_arm or q07_arm
    def rel_family(family: str) -> bool:
        return sum(all(cell_for(family, profile, mode) is not None and _cell_pass(cell_for(family, profile, mode) or {}, graph=False, relational=True) for mode in ("fresh-process", "warm-process")) for profile in profiles) >= 2
    relational_arm = any(rel_family(family) for family in relational_families)
    if graph_arm and relational_arm:
        return HYBRID
    if graph_arm:
        return GRAPH
    return RETAIN_S


def classify_without_step5(cells: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> str:
    """Evaluate the deterministic performance Boolean with CI uncertainty omitted."""
    return _performance_outcome(cells)


def reachable_outcomes(
    predicate_intervals: Sequence[Mapping[str, Any]], evaluate: Callable[[Mapping[str, bool]], str],
) -> frozenset[str]:
    """Enumerate admissible monotone CI assignments for the joint rule."""
    options: list[tuple[str, ...]] = []
    names: list[str] = []
    for item in predicate_intervals:
        name = str(item["name"]); lower = float(item["lower"]); upper = float(item["upper"])
        thresholds = sorted({float(t) for t in item.get("thresholds", ()) if lower <= float(t) <= upper})
        if not thresholds:
            options.append(("pass" if bool(item.get("point_pass", lower >= upper)) else "fail",))
        else:
            # Intervals partition at thresholds; evaluate both endpoints and
            # every threshold (threshold itself is on the passing side).
            candidate_values = sorted({lower, upper, *thresholds})
            patterns = {
                tuple("pass" if value >= threshold else "fail" for threshold in thresholds)
                for value in candidate_values
            }
            if len(thresholds) == 1:
                options.append(tuple(pattern[0] for pattern in sorted(patterns)))
            else:
                options.append(tuple(pattern for pattern in sorted(patterns)))
        names.append(name)
    outcomes = set()
    for assignment in itertools.product(*options):
        outcomes.add(evaluate(dict(zip(names, assignment))))
    return frozenset(outcomes)


def joint_ci_reachable_outcomes(
    predicate_intervals: Sequence[Mapping[str, Any]],
    evaluate: Callable[[Mapping[str, Any]], str],
) -> frozenset[str]:
    """Public alias for the exhaustive joint-CI conformance machinery."""
    return reachable_outcomes(predicate_intervals, evaluate)


def classify(
    result: Mapping[str, Any], *, cells: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Apply frozen classifier precedence and, where supplied, joint CI rule."""
    if not result.get("host_valid", result.get("measurement_valid", True)) or result.get("pairing_broken", False) or result.get("missing_samples", False) or result.get("implementation_failure_before_contract", False):
        return INCONCLUSIVE
    if result.get("architecture_problem", False) or result.get("gold_parity_failure", False) or result.get("nondeterministic_export", False):
        return ARCHITECTURE_PROBLEM
    if result.get("combined_cache_capacity_breach", False) or result.get("s_hard_breach", False) or result.get("operational_envelope_failure", False):
        return OPERATIONAL_FAILURE
    if result.get("s_passes", True) and (result.get("g_hard_breach", False) or result.get("g_relative_breach", False)):
        return RETAIN_S
    if result.get("required_ci_crossing", False):
        outcomes = result.get("reachable_outcomes")
        if outcomes is None and result.get("ci_intervals") is not None and result.get("ci_evaluate") is not None:
            outcomes = joint_ci_reachable_outcomes(result["ci_intervals"], result["ci_evaluate"])
        if outcomes is None and result.get("ci_assignments") is not None:
            outcomes = result["ci_assignments"]
        if outcomes is not None:
            outcomes = frozenset(outcomes)
            if len(outcomes) != 1:
                return INCONCLUSIVE
            performance = next(iter(outcomes))
        else:
            return INCONCLUSIVE
    else:
        performance = _performance_outcome(cells if cells is not None else result.get("cells", ()))
    return performance


__all__ = [
    "ClassifierError", "ConfidenceInterval", "INCONCLUSIVE", "ARCHITECTURE_PROBLEM",
    "OPERATIONAL_FAILURE", "RETAIN_S", "HYBRID", "GRAPH", "capped_allowances",
    "memory_rel_g", "disk_rel_g", "margin_distance", "direct_observation_margin",
    "t4_condition3_cells", "project_monotone_upper_envelope", "t4_headroom_pass",
    "t4_condition4", "reachable_outcomes", "classify",
    "joint_ci_reachable_outcomes", "classify_result", "classify_without_step5", "resource_disposition",
]

classify_result = classify
