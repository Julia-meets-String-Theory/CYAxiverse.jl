#!/usr/bin/env python3
"""Keep accepted-geometry targets separate from proposal and retry budgets.

This module is deliberately independent of CYTools and of the geometry
generator.  The integration layer can pass it a deterministic stream of
candidate decisions, while tests and bounded probes can replay the same
decisions without constructing a live polytope.

``accepted_target`` counts accepted geometries.  ``proposal_budget`` limits
the total number of candidate proposals that may be examined.  ``retry_budget``
limits additional non-accepted attempts after the first proposal.  A rejected
candidate therefore consumes one proposal and one retry slot, but it never
reduces the accepted target.  These are intentionally separate fields in the
configuration and in the serialized report.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


TERMINAL_STATUSES = (
    "accepted_geometry",
    "proposal_budget_exhausted",
    "geometry_target_shortfall",
    "ntfe_generation_failure",
    "duplicate_ntfe_identity",
    "invalid_frst",
    "kahler_cone_failure",
    "kahler_tip_failure",
    "kaehler_point_shortfall",
    "volume_filter_rejection",
    "numerical_geometry_failure",
    "assignment_pool_shortfall",
    "output_collision",
    "user_decision_required",
)

# The generator also records full-triangulation duplicates.  Keep this status
# available to callers without making it a new scientific category.
COUNT_STATUSES = TERMINAL_STATUSES + ("duplicate_full_triangulation",)

_STATUS_ALIASES = {
    "accepted": "accepted_geometry",
    "accepted_frst": "accepted_geometry",
    "duplicate": "duplicate_ntfe_identity",
    "duplicate_ntfe": "duplicate_ntfe_identity",
    "duplicate_full": "duplicate_full_triangulation",
    "invalid": "invalid_frst",
    "invalid_frst_candidate": "invalid_frst",
    "kahler": "kahler_tip_failure",
    # Existing generator reports use the German transliteration with "ae".
    "kaehler_tip_failure": "kahler_tip_failure",
    "kaehler": "kahler_tip_failure",
    "kahler_cone": "kahler_cone_failure",
    "topology_or_cone_error": "kahler_cone_failure",
    "divisor_volume_filter_rejection": "volume_filter_rejection",
    "qcd_normalization_failure": "volume_filter_rejection",
    "no_eligible_intersecting_qed_pair": "assignment_pool_shortfall",
    "numerical": "numerical_geometry_failure",
    "assignment_pool": "assignment_pool_shortfall",
    "collision": "output_collision",
}
_APPROVED_SAMPLERS = frozenset(("fair", "fast", "ntfe_fast"))
_FORBIDDEN_SAMPLER_MARKERS = ("gnn", "dualgnn", "torch", "pytorch")


class ProposalControllerError(ValueError):
    """Raise when a proposal controller contract is malformed."""


class ProposalGenerationFailure(RuntimeError):
    """Report a recoverable sampler-generation failure to the controller."""

    def __init__(self, message: str, status: str = "ntfe_generation_failure"):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ProposalControllerConfig:
    """Describe one replayable accepted-target experiment."""

    accepted_target: int
    proposal_budget: int
    retry_budget: int
    h11: int
    sampler_name: str
    deterministic_seed: int

    def __post_init__(self) -> None:
        for name in ("accepted_target", "proposal_budget", "retry_budget", "h11"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProposalControllerError(f"{name} must be an integer")
        if self.accepted_target < 0:
            raise ProposalControllerError("accepted_target must be non-negative")
        if self.proposal_budget < 0:
            raise ProposalControllerError("proposal_budget must be non-negative")
        if self.retry_budget < 0:
            raise ProposalControllerError("retry_budget must be non-negative")
        if self.h11 < 1:
            raise ProposalControllerError("h11 must be positive")
        if isinstance(self.deterministic_seed, bool) or not isinstance(
            self.deterministic_seed, int
        ):
            raise ProposalControllerError("deterministic_seed must be an integer")
        if not isinstance(self.sampler_name, str) or not self.sampler_name.strip():
            raise ProposalControllerError("sampler_name must be a non-empty string")
        sampler = self.sampler_name.strip()
        lowered = sampler.lower()
        if any(marker in lowered for marker in _FORBIDDEN_SAMPLER_MARKERS):
            raise ProposalControllerError(
                "GNN, Torch, and PyTorch sampler routes are forbidden"
            )
        if sampler not in _APPROVED_SAMPLERS:
            raise ProposalControllerError(
                f"unsupported sampler_name {sampler!r}; expected one of "
                f"{sorted(_APPROVED_SAMPLERS)}"
            )
        if self.h11 == 491 and sampler != "ntfe_fast":
            raise ProposalControllerError(
                "h11=491 requires the native sampler_name 'ntfe_fast'"
            )

    def proposal_seed(self, proposal_index: int) -> int:
        """Derive the stable per-proposal seed used by the replay contract."""
        if proposal_index < 1:
            raise ProposalControllerError("proposal_index is one-based")
        return self.deterministic_seed + proposal_index - 1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible controller settings."""
        return asdict(self)


@dataclass(frozen=True)
class ProposalDecision:
    """Classify one proposal without coupling to a CYTools exception type."""

    status: str
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", normalize_status(self.status))
        if not isinstance(self.reason, str):
            object.__setattr__(self, "reason", str(self.reason))

    @classmethod
    def accepted(cls, reason: str = "accepted geometry") -> "ProposalDecision":
        """Create an accepted-geometry decision."""
        return cls("accepted_geometry", reason)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible decision data."""
        return {
            "status": self.status,
            "reason": self.reason,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class ProposalRecord:
    """Persist the decision and seed for one examined proposal."""

    proposal_index: int
    proposal_seed: int
    retry_index: int
    proposal_status: str
    terminal_status: str
    accepted_count_after: int
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible proposal-record data."""
        return {
            "proposal_index": self.proposal_index,
            "proposal_seed": self.proposal_seed,
            "retry_index": self.retry_index,
            "proposal_status": self.proposal_status,
            "terminal_status": self.terminal_status,
            "accepted_count_after": self.accepted_count_after,
            "reason": self.reason,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class ProposalControllerReport:
    """Summarize a proposal stream, including a complete shortfall record."""

    config: ProposalControllerConfig
    records: tuple[ProposalRecord, ...]
    terminal_status: str
    budget_status: str | None
    accepted_count: int
    retry_count: int
    source_exhausted: bool

    @property
    def proposal_count(self) -> int:
        """Return the number of candidate proposals examined."""
        return len(self.records)

    @property
    def shortfall(self) -> int:
        """Return the number of accepted geometries still missing."""
        return max(0, self.config.accepted_target - self.accepted_count)

    @property
    def status_counts(self) -> dict[str, int]:
        """Count proposal, candidate, and final shortfall statuses separately."""
        counts = {status: 0 for status in COUNT_STATUSES}
        counts.update(
            {
                "proposal": self.proposal_count,
                "acceptance": self.accepted_count,
                "retry": self.retry_count,
                "shortfall": self.shortfall,
                "retry_budget_exhausted": 0,
                "proposal_source_exhausted": 0,
            }
        )
        for record in self.records:
            counts[record.terminal_status] = counts.get(record.terminal_status, 0) + 1
        if self.terminal_status == "geometry_target_shortfall":
            counts["geometry_target_shortfall"] += 1
        if self.budget_status is not None:
            counts[self.budget_status] = counts.get(self.budget_status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete replay and terminal accounting contract."""
        return {
            "report_schema": "glimmers-proposal-controller-1",
            "config": self.config.to_dict(),
            "terminal_status": self.terminal_status,
            "budget_status": self.budget_status,
            "accepted_count": self.accepted_count,
            "accepted_target": self.config.accepted_target,
            "proposal_count": self.proposal_count,
            "proposal_budget": self.config.proposal_budget,
            "retry_count": self.retry_count,
            "retry_budget": self.config.retry_budget,
            "shortfall": self.shortfall,
            "source_exhausted": self.source_exhausted,
            "status_counts": self.status_counts,
            "records": [record.to_dict() for record in self.records],
        }

    def write_json(self, path: str) -> None:
        """Atomically write the replayable report to ``path``."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".glimmers-proposal-controller-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class ProposalController:
    """Consume proposals until the accepted target or an explicit cap wins."""

    def __init__(self, config: ProposalControllerConfig):
        self.config = config

    def run(
        self,
        proposals: Iterable[Any] | Callable[..., Any],
        evaluator: Callable[..., Any] | None = None,
    ) -> ProposalControllerReport:
        """Evaluate a deterministic proposal stream under both independent caps.

        ``proposals`` may be an iterable of decisions/candidate mappings or a
        factory accepting ``(proposal_index, proposal_seed, retry_index)``.
        An evaluator may turn a candidate into a decision; it accepts either
        ``(candidate)`` or ``(candidate, context)``.  A
        :class:`ProposalGenerationFailure` is recorded and consumes a retry
        slot, allowing native sampler failures to be diagnosed without
        changing the sampler.
        """
        if self.config.accepted_target == 0:
            return ProposalControllerReport(
                self.config, (), "accepted_geometry", None, 0, 0, False
            )

        iterator = None
        if not callable(proposals):
            iterator = iter(proposals)

        records: list[ProposalRecord] = []
        accepted_count = 0
        retry_count = 0
        source_exhausted = False
        while accepted_count < self.config.accepted_target:
            if len(records) >= self.config.proposal_budget:
                break
            if records and retry_count >= self.config.retry_budget:
                break

            proposal_index = len(records) + 1
            proposal_seed = self.config.proposal_seed(proposal_index)
            retry_index = retry_count
            try:
                if iterator is not None:
                    candidate = next(iterator)
                else:
                    candidate = _call_with_supported_arguments(
                        proposals, proposal_index, proposal_seed, retry_index
                    )
            except StopIteration:
                source_exhausted = True
                break
            except ProposalGenerationFailure as exc:
                normalized = ProposalDecision(
                    exc.status, str(exc), {"exception_type": type(exc).__name__}
                )
            except Exception as exc:
                normalized = ProposalDecision(
                    "ntfe_generation_failure",
                    f"{type(exc).__name__}: {exc}",
                    {"exception_type": type(exc).__name__},
                )
            else:
                try:
                    decision = (
                        _call_evaluator(evaluator, candidate, proposal_index, proposal_seed)
                        if evaluator is not None
                        else candidate
                    )
                    normalized = normalize_decision(decision)
                except ProposalGenerationFailure as exc:
                    normalized = ProposalDecision(
                        exc.status, str(exc), {"exception_type": type(exc).__name__}
                    )
                except Exception as exc:
                    normalized = ProposalDecision(
                        "numerical_geometry_failure",
                        f"{type(exc).__name__}: {exc}",
                        {"exception_type": type(exc).__name__},
                    )

            if normalized.status == "accepted_geometry":
                accepted_count += 1
            else:
                retry_count += 1
            records.append(
                ProposalRecord(
                    proposal_index=proposal_index,
                    proposal_seed=proposal_seed,
                    retry_index=retry_index,
                    proposal_status="proposal",
                    terminal_status=normalized.status,
                    accepted_count_after=accepted_count,
                    reason=normalized.reason,
                    metadata=normalized.metadata,
                )
            )

        if accepted_count >= self.config.accepted_target:
            terminal_status = "accepted_geometry"
            budget_status = None
        else:
            terminal_status = "geometry_target_shortfall"
            if len(records) >= self.config.proposal_budget:
                budget_status = "proposal_budget_exhausted"
            elif records and retry_count >= self.config.retry_budget:
                budget_status = "retry_budget_exhausted"
            elif source_exhausted:
                budget_status = "proposal_source_exhausted"
            else:
                budget_status = "proposal_budget_exhausted"
        return ProposalControllerReport(
            self.config,
            tuple(records),
            terminal_status,
            budget_status,
            accepted_count,
            retry_count,
            source_exhausted,
        )


def normalize_status(status: str) -> str:
    """Normalize an accepted/rejection label to the shared terminal taxonomy."""
    if not isinstance(status, str):
        raise ProposalControllerError("proposal status must be a string")
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _STATUS_ALIASES.get(normalized, normalized)
    if normalized not in COUNT_STATUSES:
        raise ProposalControllerError(
            f"unsupported proposal status {status!r}; expected one of {COUNT_STATUSES}"
        )
    return normalized


def normalize_decision(decision: Any) -> ProposalDecision:
    """Convert a simple callback result into a :class:`ProposalDecision`."""
    if isinstance(decision, ProposalDecision):
        return decision
    if isinstance(decision, str):
        return ProposalDecision(decision)
    if isinstance(decision, bool):
        return ProposalDecision("accepted_geometry" if decision else "numerical_geometry_failure")
    if isinstance(decision, Mapping):
        status = decision.get("terminal_status", decision.get("status"))
        if status is None:
            raise ProposalControllerError("decision mapping requires status or terminal_status")
        metadata = decision.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return ProposalDecision(
            normalize_status(status),
            str(decision.get("reason", decision.get("terminal_reason", ""))),
            metadata,
        )
    raise ProposalControllerError(
        "a proposal decision must be a status string, mapping, bool, or ProposalDecision"
    )


def run_proposal_controller(
    config: ProposalControllerConfig,
    proposals: Iterable[Any] | Callable[..., Any],
    evaluator: Callable[..., Any] | None = None,
) -> ProposalControllerReport:
    """Run a proposal stream with explicit accepted, proposal, and retry caps."""
    return ProposalController(config).run(proposals, evaluator=evaluator)


# Keep the shorter name convenient for bounded integration tests.
run_controller = run_proposal_controller


def deterministic_seed_for(config: ProposalControllerConfig, proposal_index: int) -> int:
    """Return the recorded deterministic seed for one proposal index."""
    return config.proposal_seed(proposal_index)


def _call_with_supported_arguments(function: Callable[..., Any], *arguments: Any) -> Any:
    """Call a factory with the documented prefix it accepts."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*arguments)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return function(*arguments)
    return function(*arguments[: len(positional)])


def _call_evaluator(
    evaluator: Callable[..., Any], candidate: Any, proposal_index: int, proposal_seed: int
) -> Any:
    context = {
        "proposal_index": proposal_index,
        "proposal_seed": proposal_seed,
    }
    return _call_with_supported_arguments(evaluator, candidate, context)


def _json_safe(value: Any) -> Any:
    """Convert common scalar/container values without requiring NumPy."""
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


def _self_test() -> None:
    """Run deterministic controller checks without importing CYTools."""
    config = ProposalControllerConfig(2, 5, 5, 491, "ntfe_fast", 17)
    stream = [
        {"status": "invalid_frst", "reason": "mock invalid FRST"},
        {"status": "duplicate_ntfe_identity"},
        {"status": "accepted_geometry"},
        {"status": "numerical_geometry_failure"},
        {"status": "accepted_geometry"},
    ]
    report = run_proposal_controller(config, stream)
    assert report.terminal_status == "accepted_geometry"
    assert report.accepted_count == 2
    assert report.proposal_count == 5
    assert report.shortfall == 0
    assert report.status_counts["invalid_frst"] == 1
    assert report.status_counts["duplicate_ntfe_identity"] == 1
    assert report.status_counts["numerical_geometry_failure"] == 1
    assert [record.proposal_seed for record in report.records] == [17, 18, 19, 20, 21]

    shortfall = run_proposal_controller(
        ProposalControllerConfig(2, 3, 9, 491, "ntfe_fast", 99),
        ["invalid_frst", "kahler_tip_failure", "assignment_pool_shortfall", "accepted"],
    )
    assert shortfall.terminal_status == "geometry_target_shortfall"
    assert shortfall.budget_status == "proposal_budget_exhausted"
    assert shortfall.accepted_count == 0
    assert shortfall.shortfall == 2
    assert shortfall.status_counts["geometry_target_shortfall"] == 1
    assert shortfall.status_counts["proposal_budget_exhausted"] == 1

    retry_shortfall = run_proposal_controller(
        ProposalControllerConfig(1, 9, 1, 491, "ntfe_fast", 4),
        ["invalid_frst", "accepted_geometry"],
    )
    assert retry_shortfall.budget_status == "retry_budget_exhausted"
    assert retry_shortfall.proposal_count == 1
    assert retry_shortfall.status_counts["retry_budget_exhausted"] == 1


if __name__ == "__main__":
    _self_test()
    print("glimmers_proposal_controller self-test passed")
