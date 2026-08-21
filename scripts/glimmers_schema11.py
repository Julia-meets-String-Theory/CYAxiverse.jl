"""Small, dependency-light helpers for the Glimmers schema 1.1 contract.

The generator deliberately keeps these operations independent of CYTools and
HDF5.  That makes the deterministic population rules, factorized charge
representation, and external accounting artifacts testable without starting a
geometry calculation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping

import numpy as np


SCHEMA_VERSION = "1.1"
CHARGE_FACTORIZED_SCHEMA_VERSION = "glimmers-charge-factorized-1.1"
NORMALIZATION_MAP_VERSION = "homogeneous-qcd-volume-40-v1"
TARGET_GEOMETRY_COUNT = 1400
MINIMUM_EFT_ROWS = 100_000
MAXIMUM_EFT_ROWS = 200_000
# Keep the historical name as an upper-bound alias for callers that have not
# yet moved to the explicit minimum/ceiling vocabulary.
TARGET_EFT_ROWS = MAXIMUM_EFT_ROWS
QCD_VOLUME_TARGET = 40.0
QCD_VOLUME_TOLERANCE = 1e-9
DIVISOR_VOLUME_TOLERANCE = 1e-8
QED_VOLUME_MAX = 127.5
STORAGE_HARD_STOP_BYTES = 2 * 1024**3
PRODUCTION_COMPLETE_DATASET_STATUS = "production_complete"
DIAGNOSTIC_PARTIAL_DATASET_STATUS = "diagnostic_partial"
MODEL_TARGET_SHORTFALL_STATUS = "model_target_shortfall"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _stable_text(value):
    """Decode persisted byte labels before applying stable string identity."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def stable_hash(value):
    """Return a deterministic SHA-256 digest for serializable input."""
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(*parts):
    """Derive a process-independent non-negative NumPy-compatible seed."""
    digest = stable_hash(parts)
    return int.from_bytes(bytes.fromhex(digest[:16]), byteorder="big") & ((1 << 63) - 1)


def reconcile_eft_capacity(
    validated_capacity,
    rows_written,
    *,
    minimum_rows=MINIMUM_EFT_ROWS,
    maximum_rows=MAXIMUM_EFT_ROWS,
):
    """Reconcile validated assignment capacity with the emitted row table."""
    capacity = int(validated_capacity)
    rows = int(rows_written)
    minimum, target = _validate_row_bounds(minimum_rows, maximum_rows)
    if capacity < 0 or rows < 0:
        raise ValueError("validated capacity and rows written must be non-negative")
    if rows > capacity:
        raise ValueError("rows written cannot exceed validated assignment capacity")
    if rows > target:
        raise ValueError("rows written cannot exceed the exact row target")
    production_complete = rows == target
    return {
        "validated_assignment_capacity": capacity,
        "rows_written": rows,
        "requested_target": target,
        "minimum_acceptable": minimum,
        "capacity_shortfall": max(0, target - capacity),
        "row_shortfall": max(0, target - rows),
        "minimum_shortfall": max(0, minimum - rows),
        "minimum_reached": rows >= minimum,
        "production_complete": production_complete,
        "diagnostic_success": True,
        "dataset_status": (
            PRODUCTION_COMPLETE_DATASET_STATUS
            if production_complete
            else DIAGNOSTIC_PARTIAL_DATASET_STATUS
        ),
        "model_target_shortfall": not production_complete,
    }


class CapacityAllocation(dict):
    """Mapping of geometry IDs to quotas with serializable global accounting.

    The object deliberately behaves like the old quota mapping for the
    generator integration (`allocation[geometry_id]` and `allocation.values()`)
    while exposing the capacity-aware contract through attributes and
    ``to_dict()``.  Metadata is kept outside the mapping entries so a caller
    cannot accidentally count accounting fields as model rows.
    """

    def __init__(self, quotas, metadata):
        super().__init__(quotas)
        self._metadata = dict(metadata)

    def __getitem__(self, key):
        if key == "quotas":
            return dict(self)
        if key in self._metadata and not dict.__contains__(self, key):
            return self._metadata[key]
        return super().__getitem__(key)

    def __contains__(self, key):
        return key in self._metadata or super().__contains__(key)

    def get(self, key, default=None):
        if key == "quotas":
            return dict(self)
        if key in self._metadata and not dict.__contains__(self, key):
            return self._metadata[key]
        return super().get(key, default)

    def __getattr__(self, name):
        try:
            return self._metadata[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self):
        """Return the complete JSON-compatible allocation record."""
        result = dict(self._metadata)
        result["quotas"] = dict(self)
        return result


def _coerce_capacity_map(identifiers, pool_sizes, maximum_rows):
    """Normalize optional assignment-pool capacities without reordering IDs."""
    if pool_sizes is None:
        # The compatibility path has no persisted pools.  Treat capacity as
        # unknown rather than fabricating a finite pool-size claim; the ceiling
        # still bounds the returned allocation.
        return {identifier: int(maximum_rows) for identifier in identifiers}, False
    if isinstance(pool_sizes, Mapping):
        missing = [identifier for identifier in identifiers if identifier not in pool_sizes]
        extra = [str(identifier) for identifier in pool_sizes if str(identifier) not in identifiers]
        if missing or extra:
            raise ValueError(
                "pool_sizes must contain exactly one capacity for every geometry ID"
            )
        capacities = {
            identifier: int(pool_sizes[identifier]) for identifier in identifiers
        }
    else:
        try:
            values = list(pool_sizes)
        except TypeError as exc:
            raise TypeError("pool_sizes must be a mapping or a sequence") from exc
        if len(values) != len(identifiers):
            raise ValueError("pool_sizes must have one entry per geometry ID")
        capacities = {
            identifier: int(value)
            for identifier, value in zip(identifiers, values)
        }
    if any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("assignment-pool capacities must be non-negative")
    return capacities, True


def _validate_row_bounds(minimum_rows, maximum_rows):
    """Validate and normalize the approved model-row bounds."""
    minimum = int(minimum_rows)
    maximum = int(maximum_rows)
    if minimum < 1:
        raise ValueError("minimum_rows must be positive")
    if maximum < minimum:
        raise ValueError("maximum_rows must be at least minimum_rows")
    return minimum, maximum


def allocate_eft_quotas(
    geometry_ids,
    pool_sizes=None,
    minimum_rows=MINIMUM_EFT_ROWS,
    maximum_rows=MAXIMUM_EFT_ROWS,
    *,
    target_rows=None,
):
    """Allocate capacity-aware quotas over stable geometry IDs.

    ``pool_sizes`` may be a mapping keyed by geometry ID or a sequence aligned
    with ``geometry_ids``.  The allocator first computes the feasible count
    from those actual capacities, caps it at ``maximum_rows``, and distributes
    rows in deterministic stable-hash round-robin order without exceeding any
    pool.  It records ``ceiling_reached``, ``capacity_exhausted``, or
    ``model_target_shortfall`` as the stop reason and never requires a fixed
    number of geometries or a fixed per-geometry quota.

    If ``pool_sizes`` is omitted, capacity is explicitly marked unknown and
    the compatibility allocation is bounded by the ceiling.  The integration
    path must provide persisted pool sizes so the result can be a capacity
    claim.

    ``target_rows`` is a compatibility alias for ``maximum_rows``.  New code
    should pass the approved ``minimum_rows`` and ``maximum_rows`` explicitly.
    """
    if target_rows is not None:
        if pool_sizes is not None:
            raise TypeError("target_rows cannot be combined with pool_sizes")
        maximum_rows = target_rows
    elif isinstance(pool_sizes, (int, np.integer)) and not isinstance(pool_sizes, bool):
        # The pre-remediation call was allocate_eft_quotas(ids, target_rows).
        maximum_rows = pool_sizes
        pool_sizes = None

    minimum, ceiling = _validate_row_bounds(minimum_rows, maximum_rows)
    identifiers = [str(identifier) for identifier in geometry_ids]
    if not identifiers:
        raise ValueError("geometry_ids must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("geometry IDs must be unique for quota allocation")
    capacities, capacity_known = _coerce_capacity_map(
        identifiers, pool_sizes, ceiling
    )
    ordering = sorted(
        identifiers, key=lambda identifier: (stable_hash(identifier), identifier)
    )
    maximum_feasible = sum(capacities.values()) if capacity_known else None
    accepted_count = (
        min(ceiling, maximum_feasible) if capacity_known else ceiling
    )
    quotas = {identifier: 0 for identifier in ordering}
    remaining = accepted_count
    # A stable round-robin fills small pools exactly and redistributes their
    # unused quota to larger pools.  It is O(accepted_count), bounded by the
    # approved 200000-row ceiling, and independent of worker completion order.
    while remaining:
        active = [identifier for identifier in ordering if quotas[identifier] < capacities[identifier]]
        if not active:
            raise AssertionError("quota allocation exhausted before accepted_count")
        for identifier in active:
            quotas[identifier] += 1
            remaining -= 1
            if not remaining:
                break

    if capacity_known and maximum_feasible < minimum:
        stop_reason = "model_target_shortfall"
        terminal_status = "model_target_shortfall"
    elif accepted_count >= ceiling:
        stop_reason = "ceiling_reached"
        terminal_status = "model_ceiling_reached"
    else:
        stop_reason = "capacity_exhausted"
        terminal_status = "model_minimum_reached"
    metadata = {
        "pool_capacity_known": capacity_known,
        "pool_sizes": {identifier: capacities[identifier] for identifier in ordering},
        "ordered_geometry_ids": ordering,
        "requested_minimum": minimum,
        "ceiling": ceiling,
        "minimum_rows": minimum,
        "maximum_rows": ceiling,
        "maximum_feasible_rows": maximum_feasible,
        "accepted_count": accepted_count,
        "minimum_reached": accepted_count >= minimum,
        "successful": accepted_count >= minimum,
        "stop_reason": stop_reason,
        "terminal_status": terminal_status,
        "claim_boundary": "adapted_finite_model_table_not_exact_200000_reproduction",
    }
    return CapacityAllocation(quotas, metadata)


def row_assignment_seed(base_seed, geometry_id, row_index):
    """Return the deterministic seed recorded for one EFT draw."""
    if int(row_index) < 0:
        raise ValueError("row_index must be non-negative")
    return stable_seed("glimmers-eft-row", int(base_seed), str(geometry_id), int(row_index))


def sample_pool_with_replacement(
    pool_size,
    quota,
    geometry_id,
    base_seed,
    *,
    assignment_hashes=None,
    callback=None,
    draw_cap=None,
    eligible_pool_ranks=None,
):
    """Draw ordered assignments with replacement and collapse accepted duplicates.

    ``quota`` is the requested number of unique accepted rows for one geometry.
    The deterministic draw cap is ``10 * quota`` unless ``draw_cap`` is supplied
    explicitly by a caller that is reproducing the same approved policy.  A
    callback may reject a draw by returning ``{"accepted": False, ...}``; the
    sampler then draws again from the same geometry.  A duplicate of an already
    accepted assignment is counted and collapsed without invoking the callback.
    """
    pool_size = int(pool_size)
    quota = int(quota)
    if pool_size < 0 or quota < 0:
        raise ValueError("pool_size and quota must be non-negative")
    if assignment_hashes is None:
        hashes = [str(index) for index in range(pool_size)]
    else:
        hashes = [_stable_text(value) for value in assignment_hashes]
        if len(hashes) != pool_size:
            raise ValueError("assignment_hashes must contain one hash per pool rank")
    if len(set(hashes)) != len(hashes):
        raise ValueError("assignment hashes must be unique within one geometry")
    if eligible_pool_ranks is None:
        eligible = list(range(pool_size))
    else:
        eligible = [int(rank) for rank in eligible_pool_ranks]
        if len(set(eligible)) != len(eligible) or any(
            rank < 0 or rank >= pool_size for rank in eligible
        ):
            raise ValueError("eligible pool ranks must be unique and in range")
    if draw_cap is None:
        draw_cap = 10 * quota
    draw_cap = int(draw_cap)
    if draw_cap < 0:
        raise ValueError("draw_cap must be non-negative")
    if quota and draw_cap == 0:
        raise ValueError("a positive quota requires a positive draw_cap")

    if not eligible:
        return {
            "geometry_id": str(geometry_id),
            "requested_unique_rows": quota,
            "draw_cap": draw_cap,
            "total_draws": 0,
            "accepted_unique_rows": 0,
            "duplicate_draws": 0,
            "failed_draws": 0,
            "cap_induced_capacity_shortfall": quota,
            "failure_status_counts": {},
            "failure_reason_counts": {},
            "eligible_assignment_capacity": 0,
            "raw_assignment_pool_capacity": pool_size,
            "rows": [],
        }

    accepted = []
    accepted_hashes = set()
    duplicate_draws = 0
    failed_draws = 0
    failure_status_counts = Counter()
    failure_reason_counts = Counter()
    total_draws = 0

    while len(accepted) < quota and total_draws < draw_cap:
        draw_index = total_draws
        draw_seed = row_assignment_seed(base_seed, geometry_id, draw_index)
        total_draws += 1
        pool_rank = eligible[
            int(np.random.default_rng(draw_seed).integers(len(eligible)))
        ]
        assignment_hash = hashes[pool_rank]
        if assignment_hash in accepted_hashes:
            duplicate_draws += 1
            continue

        result = {"accepted": True}
        if callback is not None:
            try:
                result = callback(pool_rank, draw_seed, draw_index)
            except Exception as error:  # pragma: no cover - defensive boundary
                result = {
                    "accepted": False,
                    "status": getattr(error, "terminal_status", "invalid_row_schema"),
                    "reason": f"{type(error).__name__}: {error}",
                }
            if not isinstance(result, Mapping) or "accepted" not in result:
                raise TypeError(
                    "assignment draw callback must return a mapping with an 'accepted' field"
                )
        if not bool(result["accepted"]):
            failed_draws += 1
            status = _stable_text(result.get("status", "invalid_row_schema"))
            reason = _stable_text(result.get("reason", "row construction failed"))
            failure_status_counts[status] += 1
            failure_reason_counts[reason] += 1
            continue

        accepted_hashes.add(assignment_hash)
        record = dict(result.get("record", {}))
        record.update(
            {
                "geometry_id": str(geometry_id),
                "assignment_pool_size": pool_size,
                "pool_size": pool_size,
                "assignment_pool_rank": pool_rank,
                "sampled_rank": pool_rank,
                "sampled_pool_rank": pool_rank,
                "row_index": len(accepted),
                "draw_index": draw_index,
                "row_seed": draw_seed,
                "draw_seed": draw_seed,
                "model_seed": draw_seed,
                "assignment_hash": assignment_hash,
            }
        )
        accepted.append(record)

    accepted_unique_rows = len(accepted)
    cap_induced_shortfall = max(0, quota - accepted_unique_rows) if total_draws >= draw_cap else 0
    return {
        "geometry_id": str(geometry_id),
        "requested_unique_rows": quota,
        "draw_cap": draw_cap,
        "total_draws": total_draws,
        "accepted_unique_rows": accepted_unique_rows,
        "duplicate_draws": duplicate_draws,
        "failed_draws": failed_draws,
        "cap_induced_capacity_shortfall": cap_induced_shortfall,
        "failure_status_counts": dict(sorted(failure_status_counts.items())),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "eligible_assignment_capacity": len(eligible),
        "raw_assignment_pool_capacity": pool_size,
        "rows": accepted,
    }


def sample_pool_without_replacement(
    pool_size,
    quota,
    geometry_id,
    base_seed,
    *,
    assignment_hashes=None,
    requested_minimum=MINIMUM_EFT_ROWS,
    ceiling=MAXIMUM_EFT_ROWS,
    accepted_count=None,
    stop_reason=None,
    return_records=False,
):
    """Sample distinct pool ranks and optionally emit replayable row records.

    The effective quota is capped at ``pool_size``.  This makes a small pool a
    capacity outcome rather than a false exact-quota error.  When
    ``assignment_hashes`` is supplied, hashes are validated for uniqueness
    within the complete geometry pool and copied into every emitted record.
    Production integration must provide those persisted assignment hashes;
    omitting them is supported only for rank-only unit fixtures and retains a
    null hash rather than inventing an assignment identity.
    """
    pool_size = int(pool_size)
    quota = int(quota)
    if pool_size < 0 or quota < 0:
        raise ValueError("pool_size and quota must be non-negative")
    minimum, maximum = _validate_row_bounds(requested_minimum, ceiling)
    if assignment_hashes is not None:
        hashes = [_stable_text(value) for value in assignment_hashes]
        if len(hashes) != pool_size:
            raise ValueError("assignment_hashes must contain one hash per pool rank")
        if len(set(hashes)) != len(hashes):
            raise ValueError("assignment hashes must be unique within one geometry")
    else:
        hashes = None
    effective_quota = min(quota, pool_size)
    remaining = list(range(pool_size))
    sampled = []
    for row_index in range(effective_quota):
        seed = row_assignment_seed(base_seed, geometry_id, row_index)
        selected_position = int(np.random.default_rng(seed).integers(len(remaining)))
        sampled.append((remaining.pop(selected_position), seed))
    if accepted_count is None:
        accepted = effective_quota
    else:
        accepted = int(accepted_count)
        if accepted < 0:
            raise ValueError("accepted_count must be non-negative")
    if stop_reason is None:
        if accepted < minimum:
            stop_reason = "model_target_shortfall"
        elif accepted >= maximum:
            stop_reason = "ceiling_reached"
        else:
            stop_reason = "capacity_exhausted"
    if stop_reason not in {
        "ceiling_reached",
        "capacity_exhausted",
        "model_target_shortfall",
    }:
        raise ValueError("stop_reason is not a recognized capacity terminal reason")
    if not return_records:
        return sampled
    records = []
    for row_index, (pool_rank, row_seed) in enumerate(sampled):
        records.append(
            {
                "geometry_id": str(geometry_id),
                "assignment_pool_size": pool_size,
                "pool_size": pool_size,
                "assignment_pool_rank": int(pool_rank),
                "sampled_rank": int(pool_rank),
                "sampled_pool_rank": int(pool_rank),
                "row_index": row_index,
                "row_seed": int(row_seed),
                "model_seed": int(row_seed),
                "assignment_hash": None if hashes is None else hashes[pool_rank],
                "requested_minimum": minimum,
                "ceiling": maximum,
                "maximum_rows": maximum,
                "accepted_count": accepted,
                "stop_reason": stop_reason,
            }
        )
    return records


def sample_capacity_aware_assignments(
    assignment_pools,
    base_seed,
    *,
    minimum_rows=MINIMUM_EFT_ROWS,
    maximum_rows=MAXIMUM_EFT_ROWS,
    row_callback=None,
    eligible_pool_ranks=None,
):
    """Allocate and sample complete assignment-hash pools deterministically.

    ``assignment_pools`` is a mapping from a stable geometry ID to an ordered
    sequence of assignment hashes.  The sequence order is the persisted pool
    rank order.  Production sampling is with replacement; accepted duplicate
    identities are collapsed, and an optional callback can reject a draw and
    trigger another draw from the same geometry.  Return a JSON-compatible
    record containing the global allocation, row-level replay metadata,
    per-geometry draw accounting, and terminal stop reason.  Rows are emitted
    in stable geometry-hash order, not in input or worker completion order.
    """
    if not isinstance(assignment_pools, Mapping):
        raise ValueError("assignment_pools must be a geometry mapping")
    identifiers = [str(identifier) for identifier in assignment_pools]
    pools = {}
    for identifier, raw_pool in assignment_pools.items():
        stable_identifier = str(identifier)
        if isinstance(raw_pool, Mapping):
            if "assignment_hashes" in raw_pool:
                raw_pool = raw_pool["assignment_hashes"]
            elif "assignment_hash" in raw_pool:
                raw_pool = raw_pool["assignment_hash"]
            else:
                raise ValueError(
                    "assignment-pool mappings must expose assignment_hashes"
                )
        try:
            hashes = [_stable_text(value) for value in raw_pool]
        except TypeError as exc:
            raise TypeError(
                "each assignment pool must be an ordered hash sequence"
            ) from exc
        if len(set(hashes)) != len(hashes):
            raise ValueError("assignment hashes must be unique within one geometry")
        pools[stable_identifier] = hashes
    if len(pools) != len(identifiers):
        raise ValueError("geometry IDs must be unique after string normalization")
    eligible = {}
    for identifier in identifiers:
        if eligible_pool_ranks is None:
            eligible[identifier] = list(range(len(pools[identifier])))
            continue
        if identifier not in eligible_pool_ranks:
            raise ValueError(
                "eligible_pool_ranks must contain one entry for every geometry ID"
            )
        ranks = [int(rank) for rank in eligible_pool_ranks[identifier]]
        if len(set(ranks)) != len(ranks) or any(
            rank < 0 or rank >= len(pools[identifier]) for rank in ranks
        ):
            raise ValueError("eligible pool ranks must be unique and in range")
        eligible[identifier] = ranks
    if not identifiers:
        reconciliation = reconcile_eft_capacity(
            0,
            0,
            minimum_rows=minimum_rows,
            maximum_rows=maximum_rows,
        )
        return {
            "allocation": {
                "quotas": {},
                "pool_capacity_known": True,
                "pool_sizes": {},
                "ordered_geometry_ids": [],
                "requested_minimum": int(minimum_rows),
                "ceiling": int(maximum_rows),
                "minimum_rows": int(minimum_rows),
                "maximum_rows": int(maximum_rows),
                "maximum_feasible_rows": 0,
                "accepted_count": 0,
                "minimum_reached": False,
                "successful": False,
                "stop_reason": MODEL_TARGET_SHORTFALL_STATUS,
                "terminal_status": MODEL_TARGET_SHORTFALL_STATUS,
            },
            "rows": [],
            "requested_minimum": int(minimum_rows),
            "ceiling": int(maximum_rows),
            "maximum_feasible_rows": 0,
            "planned_accepted_count": 0,
            "accepted_count": 0,
            "minimum_reached": False,
            "planned_stop_reason": MODEL_TARGET_SHORTFALL_STATUS,
            "actual_stop_reason": MODEL_TARGET_SHORTFALL_STATUS,
            "stop_reason": MODEL_TARGET_SHORTFALL_STATUS,
            "terminal_status": MODEL_TARGET_SHORTFALL_STATUS,
            "successful": False,
            "raw_assignment_capacity": 0,
            "validated_assignment_capacity": 0,
            "reconciliation": reconciliation,
            "per_geometry_sampling": {},
            "conservation": {
                "accepted_rows": 0,
                "sum_allocated_quotas": 0,
                "pool_capacity": 0,
                "validated_assignment_capacity": 0,
                "distinct_sampled_ranks": {},
            },
            "claim_boundary": "adapted_finite_model_table_not_exact_200000_reproduction",
        }
    allocation = allocate_eft_quotas(
        identifiers,
        {identifier: len(eligible[identifier]) for identifier in identifiers},
        minimum_rows=minimum_rows,
        maximum_rows=maximum_rows,
    )
    rows = []
    per_geometry_sampling = {}
    for identifier in allocation.ordered_geometry_ids:
        def geometry_callback(pool_rank, draw_seed, draw_index, *, _identifier=identifier):
            if row_callback is None:
                return {"accepted": True}
            return row_callback(_identifier, pool_rank, draw_seed, draw_index)

        sampling = sample_pool_with_replacement(
            len(pools[identifier]),
            allocation[identifier],
            identifier,
            base_seed,
            assignment_hashes=pools[identifier],
            callback=geometry_callback,
            eligible_pool_ranks=eligible[identifier],
        )
        per_geometry_sampling[identifier] = {
            key: value
            for key, value in sampling.items()
            if key != "rows"
        }
        rows.extend(sampling["rows"])
    accepted_count = len(rows)
    planned_count = allocation.accepted_count
    if accepted_count < minimum_rows:
        stop_reason = "model_target_shortfall"
        terminal_status = "model_target_shortfall"
    elif accepted_count >= maximum_rows:
        stop_reason = "ceiling_reached"
        terminal_status = "model_ceiling_reached"
    elif accepted_count < planned_count:
        stop_reason = "capacity_exhausted"
        terminal_status = "model_minimum_reached"
    else:
        stop_reason = allocation.stop_reason
        terminal_status = allocation.terminal_status
    validated_capacity = sum(len(eligible[identifier]) for identifier in identifiers)
    raw_capacity = sum(len(pools[identifier]) for identifier in identifiers)
    reconciliation = reconcile_eft_capacity(
        validated_capacity,
        accepted_count,
        minimum_rows=minimum_rows,
        maximum_rows=maximum_rows,
    )
    return {
        "allocation": allocation.to_dict(),
        "rows": rows,
        "requested_minimum": allocation.requested_minimum,
        "ceiling": allocation.ceiling,
        "maximum_feasible_rows": allocation.maximum_feasible_rows,
        "planned_accepted_count": planned_count,
        "accepted_count": accepted_count,
        "minimum_reached": accepted_count >= minimum_rows,
        "planned_stop_reason": allocation.stop_reason,
        "actual_stop_reason": stop_reason,
        "stop_reason": stop_reason,
        "terminal_status": terminal_status,
        "successful": accepted_count >= minimum_rows,
        "raw_assignment_capacity": raw_capacity,
        "validated_assignment_capacity": validated_capacity,
        "reconciliation": reconciliation,
        "per_geometry_sampling": per_geometry_sampling,
        "conservation": {
            "accepted_rows": len(rows),
            "sum_allocated_quotas": planned_count,
            "pool_capacity": raw_capacity,
            "validated_assignment_capacity": validated_capacity,
            "distinct_sampled_ranks": {
                identifier: len(
                    {
                        row["assignment_pool_rank"]
                        for row in rows
                        if row["geometry_id"] == identifier
                    }
                )
                for identifier in allocation.ordered_geometry_ids
            },
        },
        "claim_boundary": allocation.claim_boundary,
    }


def normalize_qcd_assignment(
    prime_volumes,
    effective_volumes,
    qcd_index,
    *,
    target=QCD_VOLUME_TARGET,
    min_prime=1.0,
    min_effective=1.0,
    qcd_volume_tolerance=QCD_VOLUME_TOLERANCE,
    divisor_volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
):
    """Apply and validate homogeneous QCD normalization for one assignment."""
    prime = np.asarray(prime_volumes, dtype=float).reshape(-1)
    effective = np.asarray(effective_volumes, dtype=float).reshape(-1)
    qcd_index = int(qcd_index)
    if prime.size == 0 or effective.size == 0 or not np.all(np.isfinite(prime)):
        raise ValueError("normalization reference volumes must be finite and non-empty")
    if not np.all(np.isfinite(effective)) or not 0 <= qcd_index < prime.size:
        raise ValueError("normalization reference volumes or qcd_index are invalid")
    if prime[qcd_index] <= 0.0:
        raise ValueError("QCD reference volume must be positive")
    qcd_volume_tolerance = float(qcd_volume_tolerance)
    divisor_volume_tolerance = float(divisor_volume_tolerance)
    if (
        not np.isfinite(qcd_volume_tolerance)
        or qcd_volume_tolerance < 0.0
        or not np.isfinite(divisor_volume_tolerance)
        or divisor_volume_tolerance < 0.0
    ):
        raise ValueError("normalization tolerances must be finite and non-negative")
    radial_scale = float(np.sqrt(float(target) / prime[qcd_index]))
    scale_squared = radial_scale**2
    normalized_prime = scale_squared * prime
    normalized_effective = scale_squared * effective
    qcd_volume = float(normalized_prime[qcd_index])
    minimum_prime = float(np.min(normalized_prime))
    minimum_effective = float(np.min(normalized_effective))
    if not np.all(np.isfinite(normalized_prime)) or not np.all(
        np.isfinite(normalized_effective)
    ):
        raise ValueError("post-normalization volumes are not finite")
    if not np.isclose(
        qcd_volume, float(target), rtol=0.0, atol=qcd_volume_tolerance
    ):
        raise ValueError("post-normalization QCD volume is not exactly the target")
    if minimum_prime < float(min_prime) - divisor_volume_tolerance:
        raise ValueError("post-normalization prime-divisor minimum is below one")
    if minimum_effective < float(min_effective) - divisor_volume_tolerance:
        raise ValueError("post-normalization effective-divisor minimum is below one")
    return {
        "qcd_index": qcd_index,
        "radial_scale": radial_scale,
        "volume_scale": scale_squared,
        "qcd_volume": qcd_volume,
        "prime_volumes": normalized_prime,
        "effective_volumes": normalized_effective,
        "minimum_prime_volume": minimum_prime,
        "minimum_effective_volume": minimum_effective,
        "target": float(target),
        "qcd_volume_tolerance": qcd_volume_tolerance,
        "divisor_volume_tolerance": divisor_volume_tolerance,
        "qcd_volume_residual": abs(qcd_volume - float(target)),
        "normalization_map_version": NORMALIZATION_MAP_VERSION,
        "qcd_volume_exact": bool(
            np.isclose(
                qcd_volume, float(target), rtol=0.0, atol=qcd_volume_tolerance
            )
        ),
    }


def enumerate_assignment_pool(
    *,
    prime_labels,
    prime_charges,
    prime_volumes_reference,
    effective_volumes_reference,
    neighbors,
    intersection_evidence,
    invariant_mask=None,
    qcd_volume_target=QCD_VOLUME_TARGET,
    min_prime_volume=1.0,
    min_effective_volume=1.0,
    qcd_volume_tolerance=QCD_VOLUME_TOLERANCE,
    divisor_volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
    qed_volume_max=QED_VOLUME_MAX,
):
    """Enumerate every eligible ordered ``(QCD, QED)`` pair for a geometry."""
    labels = [tuple(int(value) for value in label) for label in prime_labels]
    charges = np.asarray(prime_charges, dtype=np.int64)
    prime_reference = np.asarray(prime_volumes_reference, dtype=float).reshape(-1)
    effective_reference = np.asarray(effective_volumes_reference, dtype=float).reshape(-1)
    if charges.ndim != 2 or charges.shape[0] != len(labels):
        raise ValueError("prime labels and charges have inconsistent shapes")
    if prime_reference.shape != (len(labels),):
        raise ValueError("prime reference volumes have an inconsistent shape")
    if invariant_mask is None:
        invariant = np.ones(len(labels), dtype=bool)
    else:
        invariant = np.asarray(invariant_mask, dtype=bool).reshape(-1)
        if invariant.shape != (len(labels),):
            raise ValueError("invariant_mask has an inconsistent shape")

    pool = []
    for qcd_index in range(len(labels)):
        try:
            normalization = normalize_qcd_assignment(
                prime_reference,
                effective_reference,
                qcd_index,
                target=qcd_volume_target,
                min_prime=min_prime_volume,
                min_effective=min_effective_volume,
                qcd_volume_tolerance=qcd_volume_tolerance,
                divisor_volume_tolerance=divisor_volume_tolerance,
            )
        except ValueError:
            continue
        for qed_index in sorted(
            {int(index) for index in neighbors[qcd_index]},
            key=lambda index: (labels[index], index),
        ):
            if qed_index == qcd_index or not invariant[qed_index]:
                continue
            if not np.any(charges[qed_index]):
                continue
            pair = tuple(sorted((qcd_index, qed_index)))
            evidence = intersection_evidence.get(pair, ())
            if not evidence:
                continue
            qed_volume = float(normalization["prime_volumes"][qed_index])
            if not qed_volume <= float(qed_volume_max):
                continue
            assignment = {
                "qcd_divisor_index": qcd_index,
                "qed_divisor_index": qed_index,
                "qcd_divisor_label": list(labels[qcd_index]),
                "qed_divisor_label": list(labels[qed_index]),
                "qcd_charge_hash": stable_hash(charges[qcd_index].tolist()),
                "qed_charge_hash": stable_hash(charges[qed_index].tolist()),
                "qcd_volume": normalization["qcd_volume"],
                "qed_volume": qed_volume,
                "minimum_prime_volume": normalization["minimum_prime_volume"],
                "minimum_effective_volume": normalization["minimum_effective_volume"],
                "qcd_radial_scale": normalization["radial_scale"],
                "qcd_volume_scale": normalization["volume_scale"],
                "qcd_volume_target": normalization["target"],
                "qcd_volume_tolerance": normalization["qcd_volume_tolerance"],
                "divisor_volume_tolerance": normalization["divisor_volume_tolerance"],
                "qcd_volume_residual": normalization["qcd_volume_residual"],
                "qcd_volume_exact": normalization["qcd_volume_exact"],
                "all_divisor_minimums_valid": True,
                "qed_volume_filter": "less_than_or_equal_to_127.5",
                "qcd_qed_intersection": True,
                "intersection_evidence": [list(face) for face in evidence],
                "intersection_evidence_convention": (
                    "triangulated_two_face_lattice_point_labels"
                ),
                "assignment_policy": "ordered_qcd_qed_complete_pool",
                "normalization_map_version": NORMALIZATION_MAP_VERSION,
            }
            assignment["assignment_hash"] = stable_hash(assignment)
            pool.append(assignment)
    pool.sort(
        key=lambda item: (
            tuple(item["qcd_divisor_label"]),
            tuple(item["qed_divisor_label"]),
            item["qcd_divisor_index"],
            item["qed_divisor_index"],
        )
    )
    for rank, assignment in enumerate(pool):
        assignment["pool_rank"] = rank
    return pool


def factorized_charge_metadata(direct_charges, direct_l=None, pair_l=None):
    """Create canonical source-index metadata for ``q_j - q_i`` charges."""
    direct = np.asarray(direct_charges, dtype=np.int64)
    if direct.ndim != 2:
        raise ValueError("direct_charges must be a two-dimensional h11-by-N matrix")
    pairs = list(itertools.combinations(range(direct.shape[1]), 2))
    pair_i = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    pair_j = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    metadata = {
        "schema_version": CHARGE_FACTORIZED_SCHEMA_VERSION,
        "orientation": "h11 x N_direct; charge vectors are columns",
        "difference_convention": "q_pair[:, k] = q_direct[:, pair_j[k]] - q_direct[:, pair_i[k]]",
        "pair_ordering": "lexicographic_i_then_j_with_i_less_than_j",
        "direct_source_count": int(direct.shape[1]),
        "pair_source_count": len(pairs),
        "direct_charge_coefficients": np.ones((1, direct.shape[1]), dtype=np.int8),
        "pair_charge_coefficients": np.asarray(
            [[-1] * len(pairs), [1] * len(pairs)], dtype=np.int8
        ),
        "pair_i": pair_i,
        "pair_j": pair_j,
        "direct_l": None if direct_l is None else np.asarray(direct_l, dtype=float),
        "pair_l": None if pair_l is None else np.asarray(pair_l, dtype=float),
        "l_convention": "sign_and_log10_lambda4_rows; geometry reference only",
        "materialization": "explicit_opt_in_bounded_reconstruction",
    }
    if metadata["direct_l"] is not None and metadata["direct_l"].shape != (2, direct.shape[1]):
        raise ValueError("direct_l must have shape (2, N_direct)")
    if metadata["pair_l"] is not None and metadata["pair_l"].shape != (2, len(pairs)):
        raise ValueError("pair_l must have shape (2, N_pairs)")
    return metadata


def reconstruct_pairwise_charges(direct_charges, pair_i, pair_j):
    """Materialize only the requested logical pair block."""
    direct = np.asarray(direct_charges, dtype=np.int64)
    left = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    right = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    if left.shape != right.shape or np.any(left < 0) or np.any(right >= direct.shape[1]):
        raise ValueError("pair source indices have an invalid shape or range")
    return direct[:, right] - direct[:, left]


def _atomic_bytes(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"output collision: {path}")
    temporary = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with open(temporary, "wb") as stream:
            stream.write(payload)
        if os.path.exists(path):
            raise FileExistsError(f"output collision: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_dump(path, payload):
    _atomic_bytes(
        path,
        json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n",
    )


def atomic_jsonl_dump(path, records):
    encoded = b"".join(
        json.dumps(_jsonable(record), sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
        for record in records
    )
    _atomic_bytes(path, encoded)


def append_jsonl_record(path, record):
    """Append one serializable JSONL record and flush it to stable storage."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    encoded = (
        json.dumps(_jsonable(record), sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    with open(path, "ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def ensure_fresh_output_root(path):
    """Create an output root only when it is absent or genuinely empty."""
    path = os.path.abspath(path)
    if os.path.exists(path) and not os.path.isdir(path):
        raise FileExistsError(f"output collision: {path}")
    os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        raise FileExistsError(
            f"output root must be fresh and non-overwriting; existing entries: {path}"
        )
    return path


def summarize_terminal_records(candidate_records, model_records=()):
    """Aggregate terminal statuses while retaining the requested dimensions."""
    summary = {"candidate": {}, "model": {}}
    for key, records in (("candidate", candidate_records), ("model", model_records)):
        by_h11 = defaultdict(Counter)
        by_sampler = defaultdict(Counter)
        totals = Counter()
        for record in records:
            status = str(record.get("terminal_status", "unknown"))
            h11 = str(record.get("h11", "unknown"))
            sampler = str(record.get("sampler", "unknown"))
            by_h11[h11][status] += 1
            by_sampler[sampler][status] += 1
            totals[status] += 1
        summary[key] = {
            "totals_by_terminal_status": dict(sorted(totals.items())),
            "by_h11": {
                h11: dict(sorted(counts.items()))
                for h11, counts in sorted(by_h11.items())
            },
            "by_sampler": {
                sampler: dict(sorted(counts.items()))
                for sampler, counts in sorted(by_sampler.items())
            },
        }
    return summary


def estimate_storage(root, model_rows=0, estimated_row_bytes=256):
    """Estimate persistent bytes and enforce the schema hard-stop upstream."""
    geometry_bytes = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith((".h5", ".parquet")):
                geometry_bytes += os.path.getsize(os.path.join(directory, filename))
    model_bytes = int(model_rows) * int(estimated_row_bytes)
    estimate = {
        "geometry_bytes_observed": geometry_bytes,
        "model_rows": int(model_rows),
        "estimated_model_bytes": model_bytes,
        "estimated_manifest_bytes": 64 * 1024,
        "estimated_persistent_bytes": geometry_bytes + model_bytes + 64 * 1024,
        "hard_stop_bytes": STORAGE_HARD_STOP_BYTES,
        "hard_stop_gib": 2.0,
        "status": "within_budget"
        if geometry_bytes + model_bytes + 64 * 1024 <= STORAGE_HARD_STOP_BYTES
        else "storage_budget_exceeded",
    }
    return estimate


def write_eft_parquet(path, rows, *, metadata=None):
    """Write one compressed Parquet table atomically; require pyarrow explicitly."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "--eft requires pyarrow for its compressed Parquet columnar table; "
            "install pyarrow or do not select --eft"
        ) from exc
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise FileExistsError(f"output collision: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pylist([_jsonable(row) for row in rows])
    if metadata:
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata.update(
            {
                str(key).encode("utf-8"): str(value).encode("utf-8")
                for key, value in metadata.items()
            }
        )
        table = table.replace_schema_metadata(schema_metadata)
    temporary = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        parquet.write_table(table, temporary, compression="zstd", use_dictionary=True)
        if os.path.exists(path):
            raise FileExistsError(f"output collision: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path
