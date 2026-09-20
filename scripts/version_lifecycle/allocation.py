"""Global occupied-set and deterministic version allocation helpers.

This module consumes a static snapshot and a read-only representation of the
mutable event head.  It does not append events or mutate refs; event writers
must bind the returned snapshot/head identities in their own CAS transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from .codec import canonical_json, sha256_hex
from .static import BlockedResult, StaticSnapshot, StaticValidationError, validate_static_snapshot
from .versions import (
    Version,
    VersionLike,
    as_version,
    final_version,
    maintenance_line,
    parse_package_version,
    principal_sentinel,
)


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """A deterministic allocation decision or an exact blocked outcome."""

    status: str
    reason_code: str | None = None
    version: Version | None = None
    line: str | None = None
    static_snapshot_digest: str | None = None
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.status == "AVAILABLE" and self.version is not None

    @property
    def final_version(self) -> Version | None:
        return None if self.version is None else self.version.final


@dataclass(frozen=True, slots=True)
class GlobalAllocationView:
    """Combined static and mutable occupied namespace."""

    static_snapshot: StaticSnapshot
    allocation_event_head: Any
    event_head_commit: str
    static_occupied: frozenset[str]
    mutable_occupied: frozenset[str]
    occupied: frozenset[str]

    @property
    def status(self) -> str:
        return "READY"

    @property
    def snapshot_digest(self) -> str:
        return self.static_snapshot.snapshot_digest

    def is_available(self, value: VersionLike) -> bool:
        version = as_version(value).final
        return version.canonical not in self.occupied

    def available(self, value: VersionLike) -> bool:
        return self.is_available(value)

    def occupied_versions(self) -> tuple[str, ...]:
        return tuple(sorted(self.occupied))

    @property
    def event_head(self) -> str:
        return self.event_head_commit


def _version_identity(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value[1:] if value.startswith("v") else value
    try:
        # Mutable occupation includes active DEV reservations.  The final
        # identity is the namespace member reserved by either spelling.
        return parse_package_version(candidate).final.canonical
    except (TypeError, ValueError):
        return None


_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_EVENT_VERSION_FIELDS = frozenset(
    {
        "version", "final_version", "declared_version", "reserved_final",
        "closed_final_version",
        "intended_dev", "intended_dev_version", "dev_version", "public_tag",
        "approved_base_version", "main_at_event_version", "previous_main_version",
    }
)


def _event_occupied(events: Iterable[Mapping[str, Any]]) -> set[str]:
    """Replay event occupation with reservation terminal semantics.

    A prepared reservation remains occupied while it is prepared or opened.
    A proven pre-entry abort releases that reserved final identity.  A
    consumed reservation occupies its reserved identity and, when closure
    used a different final, the closed final as well.  Other lifecycle events
    retain the generic version-field occupation rules because their durable
    identities remain unavailable after withdrawal or abort.
    """

    occupied: set[str] = set()
    reservation_states: dict[str, str] = {}
    reservation_versions: dict[str, str] = {}
    consumed_closed_versions: dict[str, str | None] = {}
    reservation_event_types = {
        "development_reservation_prepared",
        "development_reservation_opened",
        "development_reservation_aborted",
        "development_reservation_consumed",
    }
    for event in events:
        event_type = event.get("event_type")
        if event_type in reservation_event_types:
            reservation_id = event.get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id:
                raise ValueError("reservation event requires a reservation_id")
            final_identity = _version_identity(event.get("final_version"))
            if final_identity is None:
                raise ValueError("reservation event final_version is not a canonical package identity")
            reservation_states[reservation_id] = str(event_type).removeprefix(
                "development_reservation_"
            )
            reservation_versions[reservation_id] = final_identity
            if event_type == "development_reservation_consumed":
                closed_value = event.get("closed_final_version")
                closed_identity = None
                if closed_value is not None:
                    closed_identity = _version_identity(closed_value)
                    if closed_identity is None:
                        raise ValueError(
                            "reservation event closed_final_version is not a canonical package identity"
                        )
                consumed_closed_versions[reservation_id] = closed_identity
            continue
        for key, value in event.items():
            if key not in _EVENT_VERSION_FIELDS:
                continue
            if not isinstance(value, str):
                raise ValueError(f"event version field {key} must be a string")
            identity = _version_identity(value)
            if identity is None:
                raise ValueError(f"event version field {key} is not a canonical package/tag identity")
            occupied.add(identity)
    for reservation_id, state in reservation_states.items():
        if state == "aborted":
            continue
        occupied.add(reservation_versions[reservation_id])
        closed_identity = consumed_closed_versions.get(reservation_id)
        if closed_identity is not None:
            occupied.add(closed_identity)
    return occupied


def _validated_event_head(
    head: Any,
    *,
    static_snapshot_digest: str | None = None,
) -> tuple[str, set[str]]:
    """Require an exact commit identity and a validated canonical event stream."""

    if head is None:
        raise ValueError("allocation_event_head is required")
    if isinstance(head, Mapping) and head.get("status") == "VALIDATED":
        commit = head.get("head_commit")
        occupied_values = head.get("occupied_versions")
        proof_digest = head.get("proof_digest")
        proof_snapshot_digest = head.get("static_snapshot_digest")
        if not isinstance(commit, str) or _GIT_OBJECT_RE.fullmatch(commit) is None:
            raise ValueError("validated event proof requires a full head_commit")
        if not isinstance(occupied_values, list):
            raise ValueError("validated event proof requires ordered occupied_versions")
        if static_snapshot_digest is not None and proof_snapshot_digest != static_snapshot_digest:
            raise ValueError("occupancy proof is bound to a different static snapshot")
        normalized: list[str] = []
        for value in occupied_values:
            try:
                parsed = parse_package_version(value)
            except (TypeError, ValueError) as error:
                raise ValueError("validated event proof contains a noncanonical version") from error
            identity = parsed.final.canonical
            if parsed.canonical != value:
                raise ValueError("validated event proof contains a noncanonical version")
            normalized.append(identity)
        if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("validated event proof occupied_versions must be sorted and unique")
        proof_preimage = {"head_commit": commit, "occupied_versions": normalized}
        if proof_snapshot_digest is not None:
            proof_preimage["static_snapshot_digest"] = proof_snapshot_digest
        expected = sha256_hex(canonical_json(proof_preimage))
        if proof_digest != expected:
            raise ValueError("validated event proof digest does not match its identity")
        return commit, set(normalized)
    commit = getattr(head, "commit", None)
    raw = getattr(head, "raw", None)
    provided_events = getattr(head, "events", None)
    if isinstance(head, Mapping):
        commit = head.get("commit", head.get("head_commit", commit))
        raw = head.get("raw", raw)
        provided_events = head.get("events", provided_events)
    if commit is not None or raw is not None or provided_events is not None:
        if not isinstance(commit, str) or _GIT_OBJECT_RE.fullmatch(commit) is None:
            raise ValueError("event head commit must be a full Git object ID")
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError("event head raw stream bytes are required")
        from .events import parse_stream

        parsed = tuple(parse_stream(bytes(raw)))
        if provided_events is not None:
            supplied = tuple(dict(event) for event in provided_events)
            if supplied != parsed:
                raise ValueError("event head events do not match canonical raw stream")
        return commit, _event_occupied(parsed)
    raise ValueError("allocation_event_head must be a validated LedgerHead or occupancy proof")


def validated_occupancy_proof(
    head_commit: str,
    occupied_versions: Iterable[str],
    *,
    static_snapshot_digest: str | None = None,
) -> dict[str, Any]:
    """Build the explicit proof form accepted by :func:`global_allocation_view`."""

    if not isinstance(head_commit, str) or _GIT_OBJECT_RE.fullmatch(head_commit) is None:
        raise ValueError("head_commit must be a full Git object ID")
    normalized: set[str] = set()
    for value in occupied_versions:
        try:
            parsed = parse_package_version(value)
        except (TypeError, ValueError) as error:
            raise ValueError("occupied version must be canonical") from error
        identity = parsed.final.canonical
        if parsed.canonical != value:
            raise ValueError("occupied version must be canonical")
        normalized.add(identity)
    ordered = sorted(normalized)
    proof = {"head_commit": head_commit, "occupied_versions": ordered}
    if static_snapshot_digest is not None:
        if not isinstance(static_snapshot_digest, str):
            raise ValueError("static_snapshot_digest must be a string")
        proof["static_snapshot_digest"] = static_snapshot_digest
    return {
        "status": "VALIDATED",
        **proof,
        "proof_digest": sha256_hex(canonical_json(proof)),
    }


def global_allocation_view(
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    allocation_event_head: Any = None,
) -> GlobalAllocationView | BlockedResult:
    """Combine one exact static snapshot with one exact mutable event head."""

    if isinstance(static_snapshot, BlockedResult):
        return static_snapshot
    try:
        snapshot = validate_static_snapshot(static_snapshot)
        if not snapshot.authority_verified:
            raise StaticValidationError("fresh remote static authority has not been verified")
        event_head_commit, mutable_values = _validated_event_head(
            allocation_event_head,
            static_snapshot_digest=snapshot.snapshot_digest,
        )
    except StaticValidationError as error:
        detail = str(error)
        if "source bytes" in detail:
            reason = "STATIC_SOURCE_BYTES_UNAVAILABLE"
        elif "authority" in detail:
            reason = "STATIC_AUTHORITY_SELECTOR_UNRESOLVED"
        else:
            reason = "STATIC_SNAPSHOT_INVALID"
        return BlockedResult(reason_code=reason, detail=detail)
    except (TypeError, ValueError) as error:
        return BlockedResult(reason_code="ALLOCATION_EVENT_HEAD_INVALID", detail=str(error))
    static_occupied = frozenset(snapshot.occupied_versions)
    mutable_occupied = frozenset(mutable_values)
    return GlobalAllocationView(
        static_snapshot=snapshot,
        allocation_event_head=allocation_event_head,
        event_head_commit=event_head_commit,
        static_occupied=static_occupied,
        mutable_occupied=mutable_occupied,
        occupied=static_occupied | mutable_occupied,
    )


allocation_view = global_allocation_view


def _view_or_blocked(
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    allocation_event_head: Any,
) -> GlobalAllocationView | AllocationResult:
    view = global_allocation_view(static_snapshot, allocation_event_head)
    if isinstance(view, BlockedResult):
        return AllocationResult(status="BLOCKED", reason_code=view.reason_code, detail=view.detail)
    return view


def select_principal_sentinel(
    closed_version: VersionLike,
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    allocation_event_head: Any = None,
) -> AllocationResult:
    """Select exactly ``X.Y.(Z+1)-DEV`` for a principal line reopen."""

    try:
        closed = final_version(closed_version)
    except (TypeError, ValueError) as error:
        return AllocationResult(status="BLOCKED", reason_code="INVALID_CLOSED_VERSION", detail=str(error))
    view = _view_or_blocked(static_snapshot, allocation_event_head)
    if isinstance(view, AllocationResult):
        return view
    sentinel = principal_sentinel(closed)
    if not view.is_available(sentinel):
        return AllocationResult(
            status="BLOCKED",
            reason_code="PRINCIPAL_SENTINEL_UNAVAILABLE",
            line="principal",
            static_snapshot_digest=view.snapshot_digest,
            detail=f"exact principal sentinel {sentinel} is occupied",
        )
    return AllocationResult(
        status="AVAILABLE",
        version=sentinel,
        line="principal",
        static_snapshot_digest=view.snapshot_digest,
    )


def select_principal_dev(
    closed_version: VersionLike,
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    allocation_event_head: Any = None,
) -> AllocationResult:
    """Compatibility alias for :func:`select_principal_sentinel`."""

    return select_principal_sentinel(closed_version, static_snapshot, allocation_event_head)


def select_maintenance_version(
    line: str,
    closed_version: VersionLike,
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    allocation_event_head: Any = None,
    *,
    max_search: int | None = None,
) -> AllocationResult:
    """Choose the lowest globally available patch on an ``X.Y`` line."""

    try:
        major, minor = maintenance_line(line)
        closed = final_version(closed_version)
    except (TypeError, ValueError) as error:
        return AllocationResult(status="BLOCKED", reason_code="INVALID_MAINTENANCE_LINE", detail=str(error))
    if (closed.major, closed.minor) != (major, minor):
        return AllocationResult(
            status="BLOCKED",
            reason_code="MAINTENANCE_LINE_MISMATCH",
            line=line,
            detail=f"closed version {closed} is outside {line}",
        )
    view = _view_or_blocked(static_snapshot, allocation_event_head)
    if isinstance(view, AllocationResult):
        view_line = view.line or line
        return AllocationResult(
            status=view.status,
            reason_code=view.reason_code,
            line=view_line,
            static_snapshot_digest=view.static_snapshot_digest,
            detail=view.detail,
        )
    patch = closed.patch + 1
    searched = 0
    while max_search is None or searched < max_search:
        candidate = Version(major, minor, patch, True)
        if view.is_available(candidate):
            return AllocationResult(
                status="AVAILABLE",
                version=candidate,
                line=line,
                static_snapshot_digest=view.snapshot_digest,
            )
        patch += 1
        searched += 1
    return AllocationResult(
        status="BLOCKED",
        reason_code="MAINTENANCE_PATCH_UNAVAILABLE",
        line=line,
        static_snapshot_digest=view.snapshot_digest,
        detail=f"no available patch found in first {max_search} candidates",
    )


def select_maintenance_dev(
    line: str,
    closed_version: VersionLike,
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    allocation_event_head: Any = None,
    *,
    max_search: int | None = None,
) -> AllocationResult:
    """Compatibility alias for :func:`select_maintenance_version`."""

    return select_maintenance_version(
        line,
        closed_version,
        static_snapshot,
        allocation_event_head,
        max_search=max_search,
    )


__all__ = [
    "AllocationResult",
    "GlobalAllocationView",
    "global_allocation_view",
    "allocation_view",
    "select_maintenance_dev",
    "select_maintenance_version",
    "select_principal_dev",
    "select_principal_sentinel",
    "validated_occupancy_proof",
]
