"""Global occupied-set and deterministic version allocation helpers.

Allocation consumes one verified static snapshot and one complete immutable
lifecycle-ref snapshot. No mutable head or append-only stream is a valid
allocation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .static import (
    BlockedResult,
    StaticSnapshot,
    StaticValidationError,
    _has_verified_authority,
    validate_static_snapshot,
)
from .manifests import (
    LifecycleRefSnapshot,
    ManifestError,
    _has_verified_lifecycle_authority,
    validate_lifecycle_ref_snapshot,
)
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
    """Combined static and immutable lifecycle occupied namespace."""

    static_snapshot: StaticSnapshot
    lifecycle_ref_snapshot: LifecycleRefSnapshot
    static_occupied: frozenset[str]
    lifecycle_occupied: frozenset[str]
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
    def lifecycle_snapshot_digest(self) -> str:
        return self.lifecycle_ref_snapshot.snapshot_digest

    @property
    def mutable_occupied(self) -> frozenset[str]:
        """Compatibility name; the underlying authority is immutable."""

        return self.lifecycle_occupied

    @property
    def ref_snapshot_digest(self) -> str:
        return self.lifecycle_snapshot_digest


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


def global_allocation_view(
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    lifecycle_snapshot: LifecycleRefSnapshot | Mapping[str, Any] | BlockedResult | None = None,
) -> GlobalAllocationView | BlockedResult:
    """Combine one exact static snapshot with the complete lifecycle snapshot."""

    if isinstance(static_snapshot, BlockedResult):
        return static_snapshot
    try:
        snapshot = validate_static_snapshot(static_snapshot)
        # The public ``authority_verified`` field is descriptive only.  The
        # static module carries a private authority marker after it has
        # resolved and validated the canonical remote source.  Structural
        # snapshots and caller-forged booleans must fail closed here.
        if not _has_verified_authority(snapshot):
            raise StaticValidationError("fresh remote static authority has not been verified")
        if lifecycle_snapshot is None or isinstance(lifecycle_snapshot, BlockedResult):
            if isinstance(lifecycle_snapshot, BlockedResult):
                return lifecycle_snapshot
            raise ManifestError("complete lifecycle ref snapshot is required")
        if (
            not isinstance(lifecycle_snapshot, LifecycleRefSnapshot)
            or not _has_verified_lifecycle_authority(lifecycle_snapshot)
        ):
            raise ManifestError("fresh remote lifecycle authority has not been verified")
        lifecycle = validate_lifecycle_ref_snapshot(lifecycle_snapshot)
        if (
            snapshot.source_repository != lifecycle.source_repository
            or snapshot._repository_authority
            != lifecycle._repository_authority
        ):
            raise ValueError(
                "static and lifecycle authorities belong to different repositories"
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
    except (TypeError, ValueError, ManifestError) as error:
        return BlockedResult(reason_code="LIFECYCLE_REF_SNAPSHOT_INVALID", detail=str(error))
    static_occupied = frozenset(snapshot.occupied_versions)
    lifecycle_occupied = frozenset(lifecycle.occupied_versions)
    return GlobalAllocationView(
        static_snapshot=snapshot,
        lifecycle_ref_snapshot=lifecycle,
        static_occupied=static_occupied,
        lifecycle_occupied=lifecycle_occupied,
        occupied=static_occupied | lifecycle_occupied,
    )


allocation_view = global_allocation_view


def _view_or_blocked(
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    lifecycle_snapshot: Any,
) -> GlobalAllocationView | AllocationResult:
    view = global_allocation_view(static_snapshot, lifecycle_snapshot)
    if isinstance(view, BlockedResult):
        return AllocationResult(status="BLOCKED", reason_code=view.reason_code, detail=view.detail)
    return view


def select_principal_sentinel(
    closed_version: VersionLike,
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    lifecycle_snapshot: Any = None,
) -> AllocationResult:
    """Select exactly ``X.Y.(Z+1)-DEV`` for a principal line reopen."""

    try:
        closed = final_version(closed_version)
    except (TypeError, ValueError) as error:
        return AllocationResult(status="BLOCKED", reason_code="INVALID_CLOSED_VERSION", detail=str(error))
    view = _view_or_blocked(static_snapshot, lifecycle_snapshot)
    if isinstance(view, AllocationResult):
        return view
    try:
        sentinel = principal_sentinel(closed)
    except ValueError:
        return AllocationResult(
            status="BLOCKED",
            reason_code="PRINCIPAL_VERSION_EXHAUSTED",
            line="principal",
            static_snapshot_digest=view.snapshot_digest,
        )
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
    lifecycle_snapshot: Any = None,
) -> AllocationResult:
    """Compatibility alias for :func:`select_principal_sentinel`."""

    return select_principal_sentinel(closed_version, static_snapshot, lifecycle_snapshot)


def select_maintenance_version(
    line: str,
    closed_version: VersionLike,
    static_snapshot: StaticSnapshot | Mapping[str, Any] | BlockedResult,
    lifecycle_snapshot: Any = None,
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
    view = _view_or_blocked(static_snapshot, lifecycle_snapshot)
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
        try:
            candidate = Version(major, minor, patch, True)
        except ValueError:
            return AllocationResult(
                status="BLOCKED",
                reason_code="MAINTENANCE_PATCH_EXHAUSTED",
                line=line,
                static_snapshot_digest=view.snapshot_digest,
            )
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
    lifecycle_snapshot: Any = None,
    *,
    max_search: int | None = None,
) -> AllocationResult:
    """Compatibility alias for :func:`select_maintenance_version`."""

    return select_maintenance_version(
        line,
        closed_version,
        static_snapshot,
        lifecycle_snapshot,
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
]
