"""Fail-closed Gate A orchestration over immutable lifecycle manifests.

This module is deliberately a coordinator, not a Git or GitHub client. A
typed port owns protected create-once writes and returns durable identity
evidence. The coordinator checks that evidence, keeps the relevant freeze
held across every irreversible transition, and only releases it after the
complete correspondence has been verified.

The old append-only event stream is not an authority here. The durable
transition names accepted by this module are the hyphenated manifest types
from the versioned schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Protocol

from .codec import sha256_hex
from .authorization import AuthorizationError, verify_owner_authorization
from .certification import is_safe_public_value
from .manifests import (
    canonical_manifest_bytes,
    lifecycle_ref_for_manifest,
    seal_manifest,
    validate_manifest,
)
from .versions import parse_package_version


UTC_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
PUBLIC_MANIFEST_TYPES = frozenset(
    {
        "version-claimed", "reservation-prepared", "reservation-opened",
        "reservation-aborted", "reservation-consumed", "candidate-opened",
        "candidate-withdrawn", "release-intent-prepared",
        "release-intent-aborted", "released", "publication",
    }
)


class TransactionError(RuntimeError):
    """A known fail-closed transaction outcome."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(detail or reason_code)
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class AllocationView:
    """The two immutable snapshots used by a serialized allocation."""

    static_snapshot_digest: str
    lifecycle_snapshot_digest: str
    occupied: frozenset[str] = field(default_factory=frozenset)

    @property
    def occupied_versions(self) -> frozenset[str]:
        return self.occupied


@dataclass(frozen=True, slots=True)
class ClosureIntent:
    """Inputs for the first principal closure/reopen transaction."""

    owner_line: str
    final_version: str
    closure_timestamp_utc: str
    outgoing_reserved_final: str | None = None
    transaction_id: str = ""
    expected_line_head: str = ""
    static_snapshot_digest: str = ""
    lifecycle_snapshot_digest: str = ""
    owner_authorization: str = ""
    owner_authorization_ref: str = ""
    owner_authorization_digest: str = ""
    repository: str = ""
    predecessor_refs: tuple[str, ...] = ()

    @property
    def line(self) -> str:
        return self.owner_line


@dataclass(frozen=True, slots=True)
class BootstrapIntent:
    """A maintenance bootstrap request, retained as an explicit deferral."""

    owner_line: str
    approved_base_version: str
    transaction_id: str = ""

    @property
    def line(self) -> str:
        return self.owner_line


@dataclass(frozen=True, slots=True)
class ReleaseIntent:
    """Inputs for a principal candidate-to-publication transaction."""

    owner_line: str
    final_version: str
    candidate_ref: str
    public_tag: str
    candidate_sha: str = ""
    candidate_tree: str = ""
    transaction_id: str = ""
    anchor_ref: str = ""
    anchor_sha: str = ""
    anchor_tree: str = ""
    closure_timestamp_utc: str = ""
    release_line: str = ""
    timestamp_utc: str = ""
    owner_authorization: str = ""
    owner_authorization_ref: str = ""
    owner_authorization_digest: str = ""
    repository: str = ""
    static_snapshot_digest: str = ""
    lifecycle_snapshot_digest: str = ""
    predecessor_refs: tuple[str, ...] = ()

    @property
    def line(self) -> str:
        return self.release_line or self.owner_line


@dataclass(frozen=True, slots=True)
class TransactionResult:
    """A durable result with explicit freeze and reconciliation state."""

    status: str
    reason_code: str | None = None
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    frozen: bool = False
    phase: str = ""

    @property
    def complete(self) -> bool:
        return self.status in {"PASS", "COMPLETE"}


class ImmutableLifecyclePort(Protocol):
    """Port contract used by the coordinator's create-once fixture."""

    def create_manifest(self, manifest_type: str, manifest: dict[str, Any]) -> Any: ...


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA1_RE.fullmatch(value) is not None


def _final(value: Any) -> str:
    try:
        parsed = parse_package_version(value)
    except (TypeError, ValueError) as error:
        raise TransactionError("INVALID_FINAL_VERSION") from error
    if not parsed.is_final:
        raise TransactionError("INVALID_FINAL_VERSION")
    return parsed.canonical


def _utc(value: Any) -> str:
    if not isinstance(value, str) or UTC_RE.fullmatch(value) is None:
        raise TransactionError("INVALID_CLOSURE_TIMESTAMP")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise TransactionError("INVALID_CLOSURE_TIMESTAMP") from error
    return value


def _call(port: Any, name: str, *args: Any) -> Any:
    method = getattr(port, name, None)
    if method is None:
        raise TransactionError("PORT_UNSUPPORTED", name)
    try:
        return method(*args)
    except TransactionError:
        raise
    except Exception as error:
        raise TransactionError("PORT_OPERATION_FAILED", f"{name}: {error}") from error


def _optional_call(port: Any, names: tuple[str, ...], *args: Any) -> Any:
    for name in names:
        if hasattr(port, name):
            return _call(port, name, *args)
    return None


def _freeze(port: Any, names: tuple[str, ...], intent: Any) -> Any:
    token = _optional_call(port, names, intent)
    if not token:
        raise TransactionError("LINE_FREEZE_UNAVAILABLE")
    return token


def _release_exclusion(port: Any, lease: Any) -> None:
    if lease is None:
        return
    if hasattr(port, "release_static_mutation"):
        _call(port, "release_static_mutation", lease)
    elif hasattr(port, "release_allocation_exclusion"):
        _call(port, "release_allocation_exclusion", lease)


def _acquire_exclusion(port: Any, intent: Any) -> Any:
    if hasattr(port, "acquire_static_mutation"):
        lease = _call(port, "acquire_static_mutation", intent)
    elif hasattr(port, "acquire_allocation_exclusion"):
        lease = _call(port, "acquire_allocation_exclusion", intent)
    else:
        raise TransactionError("EXCLUSION_UNAVAILABLE")
    if lease is None:
        raise TransactionError("EXCLUSION_UNAVAILABLE")
    return lease


def _view(port: Any, intent: Any) -> Any:
    if hasattr(port, "allocation_view"):
        method = getattr(port, "allocation_view")
        try:
            return method()
        except TypeError:
            return method(intent)
    if hasattr(port, "read_allocation_view"):
        return _call(port, "read_allocation_view", intent)
    return None


def _bound_view(port: Any, intent: Any) -> Any:
    view = _fresh_view(port, intent)
    static_digest, lifecycle_digest = _view_digests(view)
    if (
        static_digest != getattr(intent, "static_snapshot_digest", "")
        or lifecycle_digest != getattr(intent, "lifecycle_snapshot_digest", "")
    ):
        raise TransactionError("ALLOCATION_SNAPSHOT_STALE")
    return view


def _view_digests(view: Any) -> tuple[str, str]:
    static_digest = getattr(
        view, "static_snapshot_digest", getattr(view, "snapshot_digest", None)
    )
    lifecycle_digest = getattr(view, "lifecycle_snapshot_digest", None)
    if (
        not isinstance(static_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", static_digest) is None
        or not isinstance(lifecycle_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", lifecycle_digest) is None
    ):
        raise TransactionError("ALLOCATION_AUTHORITY_UNAVAILABLE")
    return static_digest, lifecycle_digest


def _fresh_view(port: Any, intent: Any) -> Any:
    view = _view(port, intent)
    if view is None:
        raise TransactionError("ALLOCATION_AUTHORITY_UNAVAILABLE")
    _view_digests(view)
    return view


def _successor_view(port: Any, intent: Any, previous: Any) -> Any:
    view = _fresh_view(port, intent)
    if _view_digests(view) == _view_digests(previous):
        raise TransactionError("ALLOCATION_SNAPSHOT_STALE")
    return view


def _owner_authority(port: Any) -> Any:
    configured = getattr(port, "owner_authorization_authority", None)
    if configured is not None:
        return configured
    if hasattr(port, "fetch_owner_authorization"):
        return port
    raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED")


def _authorization_reference(
    port: Any,
    intent: Any,
    action: str,
    target_ref: str,
    final_version: str,
) -> str:
    resolver = getattr(port, "owner_authorization_ref_for", None)
    if resolver is not None:
        reference = resolver(action, target_ref, final_version)
    elif final_version == getattr(intent, "final_version", ""):
        reference = getattr(intent, "owner_authorization_ref", "")
    else:
        reference = ""
    if not isinstance(reference, str) or not reference:
        raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED")
    return reference


def _authorization_now_utc(port: Any) -> str:
    clock = getattr(port, "authorization_now_utc", None)
    if not callable(clock):
        raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED")
    try:
        value = clock()
    except Exception as error:
        raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED") from error
    try:
        return _utc(value)
    except TransactionError as error:
        raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED") from error


def _authorize(
    port: Any,
    intent: Any,
    action: str,
    target_ref: str,
    *,
    final_version: str | None = None,
) -> tuple[dict[str, Any], str]:
    affected_version = final_version or getattr(intent, "final_version", "")
    reference = _authorization_reference(
        port, intent, action, target_ref, affected_version
    )
    try:
        record = verify_owner_authorization(
            _owner_authority(port),
            reference,
            repository=getattr(intent, "repository", ""),
            transaction_id=getattr(intent, "transaction_id", ""),
            action=action,
            owner_line=getattr(intent, "owner_line", ""),
            final_version=affected_version,
            target_ref=target_ref,
            now_utc=_authorization_now_utc(port),
        )
        return record, reference
    except (AuthorizationError, KeyError, TypeError, ValueError) as error:
        raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED", str(error)) from error


def _typed_manifest(
    port: Any,
    manifest_type: str,
    payload: dict[str, Any],
    authorize: Any = None,
    authorization_version: str | None = None,
) -> Any:
    """Validate, seal and create one immutable typed manifest."""

    if manifest_type not in PUBLIC_MANIFEST_TYPES:
        raise TransactionError("INVALID_MANIFEST_TYPE", manifest_type)
    if not hasattr(port, "create_manifest"):
        raise TransactionError("PORT_UNSUPPORTED", "create_manifest")
    try:
        sealed = seal_manifest(dict(payload))
        validate_manifest(sealed)
        # Re-encode to exercise the exact canonical no-LF wire contract before
        # handing bytes/identity to the protected create-once port.
        canonical_manifest_bytes(sealed)
    except (TypeError, ValueError) as error:
        raise TransactionError("MANIFEST_INVALID", str(error)) from error
    if authorize is not None:
        try:
            target = lifecycle_ref_for_manifest(sealed)
            record, reference = authorize(
                manifest_type,
                target,
                authorization_version or str(payload.get("final_version", "")),
            )
            rebound = dict(payload)
            rebound.update({
                "owner_authorization": record["owner_authorization"],
                "owner_authorization_ref": reference,
                "owner_authorization_digest": record["owner_authorization_digest"],
            })
            rebound_sealed = seal_manifest(rebound)
            validate_manifest(rebound_sealed)
            if (
                rebound_sealed["manifest_id"] != sealed["manifest_id"]
                or lifecycle_ref_for_manifest(rebound_sealed) != target
            ):
                raise TransactionError("MANIFEST_AUTHORIZATION_CYCLE")
            sealed = rebound_sealed
            canonical_manifest_bytes(sealed)
        except TransactionError:
            raise
        except Exception as error:
            raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED", str(error)) from error
    method = getattr(port, "create_manifest")
    try:
        created = method(manifest_type, sealed)
    except TransactionError:
        raise
    except Exception as error:
        raise TransactionError("PORT_OPERATION_FAILED", f"create_manifest: {error}") from error
    if not isinstance(created, dict):
        raise TransactionError("MANIFEST_CREATE_UNCERTAIN")
    try:
        validate_manifest(created)
    except (TypeError, ValueError) as error:
        raise TransactionError("MANIFEST_CREATE_UNCERTAIN", str(error)) from error
    if (created.get("manifest_type") != manifest_type
            or created.get("manifest_id") != sealed.get("manifest_id")):
        raise TransactionError("MANIFEST_CREATE_CONFLICT")
    if created != sealed:
        raise TransactionError("MANIFEST_CREATE_CONFLICT")
    return created


def _manifest_payload(
    intent: Any,
    manifest_type: str,
    fields: dict[str, Any],
    *,
    predecessor_refs: tuple[str, ...] | list[str] | None = None,
    timestamp_utc: str | None = None,
    allocation: bool = True,
    allocation_view: Any = None,
) -> dict[str, Any]:
    """Add exact common/allocation fields required by ``manifests.py``."""

    timestamp = (timestamp_utc or getattr(intent, "timestamp_utc", "")
                 or getattr(intent, "closure_timestamp_utc", ""))
    owner_authorization = getattr(intent, "owner_authorization", "")
    if not timestamp or not owner_authorization:
        raise TransactionError("MANIFEST_AUTHORITY_IDENTITY_UNPROVEN")
    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": manifest_type,
        "timestamp_utc": timestamp,
        "predecessor_refs": list(predecessor_refs if predecessor_refs is not None else getattr(intent, "predecessor_refs", ())),
        "owner_authorization": owner_authorization,
        "owner_authorization_ref": getattr(intent, "owner_authorization_ref", ""),
        "owner_authorization_digest": getattr(intent, "owner_authorization_digest", ""),
        **fields,
    }
    if allocation:
        if allocation_view is None:
            static_digest = getattr(intent, "static_snapshot_digest", "")
            lifecycle_digest = getattr(intent, "lifecycle_snapshot_digest", "")
        else:
            static_digest, lifecycle_digest = _view_digests(allocation_view)
        result.update({
            "transaction_id": getattr(intent, "transaction_id", ""),
            "static_iteration_snapshot": static_digest,
            "lifecycle_ref_snapshot": lifecycle_digest,
        })
    return result


def _as_mapping(value: Any, reason: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TransactionError(reason)
    return value


def _required_proof(port: Any, name: str, reason: str, *args: Any) -> dict[str, Any]:
    if not hasattr(port, name):
        raise TransactionError(reason)
    proof = _as_mapping(_call(port, name, *args), reason)
    if proof.get("verified") is not True:
        raise TransactionError(reason)
    return proof


def _expect_manifest(value: dict[str, Any], manifest_type: str, reason: str) -> dict[str, Any]:
    actual = value.get("manifest_type")
    if actual is not None and actual != manifest_type:
        raise TransactionError(reason)
    return value


def _next_principal(final_version: str, occupied: Any) -> str:
    major, minor, patch = (int(part) for part in final_version.split("."))
    if patch >= 2**32 - 1:
        raise TransactionError("PRINCIPAL_VERSION_EXHAUSTED")
    candidate = f"{major}.{minor}.{patch + 1}"
    occupied_set = set(occupied or ())
    if candidate in occupied_set or f"v{candidate}" in occupied_set:
        raise TransactionError("PRINCIPAL_SENTINEL_UNAVAILABLE")
    return candidate


def _line_target(intent: Any) -> str:
    return "refs/heads/vmm" if getattr(intent, "owner_line", "") == "principal" else f"refs/heads/{intent.owner_line}"


def _manifest_authorize(
    port: Any,
    intent: Any,
    _manifest_type: str,
    target: str,
    final_version: str,
) -> tuple[dict[str, Any], str]:
    """Fetch exact-target authorization for one lifecycle-ref mutation."""

    return _authorize(
        port,
        intent,
        "create-release-manifest",
        target,
        final_version=final_version,
    )


def run_maintenance_bootstrap(port: Any, intent: BootstrapIntent) -> TransactionResult:
    """Maintenance bootstrap remains explicitly deferred to approved S2."""

    return TransactionResult(
        "BLOCKED", "DEFERRED_MAINTENANCE_BOOTSTRAP",
        "Gate A does not create production maintenance-line reservations",
        evidence={"deferred": True, "manifest_types": ()}, phase="deferred",
    )


def run_rare_recovery(*args: Any, **kwargs: Any) -> TransactionResult:
    """Rare recovery is a later S2 operation, never an implicit fallback."""

    return TransactionResult(
        "BLOCKED", "DEFERRED_RARE_RECOVERY",
        "rare lifecycle recovery requires a later approved S2 gate",
        evidence={"deferred": True, "manifest_types": ()}, phase="deferred",
    )


def run_closure(port: Any, intent: ClosureIntent) -> TransactionResult:
    """Close the principal line, consume its reservation, and reopen.

    The freeze is acquired before the shared exclusion and is released only
    after terminal reservation consumption, the exact next principal
    sentinel, reopen, activation and correspondence proof all succeed.
    """

    phase = "preflight"
    evidence: dict[str, Any] = {}
    token: Any = None
    lease: Any = None
    try:
        if intent.owner_line != "principal":
            raise TransactionError("PRINCIPAL_LINE_REQUIRED")
        final_version = _final(intent.final_version)
        outgoing = _final(intent.outgoing_reserved_final or final_version)
        _utc(intent.closure_timestamp_utc)
        if (not _sha(intent.expected_line_head)
                or not re.fullmatch(r"[0-9a-f]{64}", intent.static_snapshot_digest)
                or not re.fullmatch(r"[0-9a-f]{64}", intent.lifecycle_snapshot_digest)
                or not intent.owner_authorization
                or not intent.owner_authorization_ref
                or not re.fullmatch(r"[0-9a-f]{64}", intent.owner_authorization_digest)
                or not intent.repository or not intent.predecessor_refs):
            raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED")
        token = _freeze(port, ("freeze_line",), intent)
        evidence["freeze_token"] = token
        phase = "frozen"
        lease = _acquire_exclusion(port, intent)
        phase = "serialized"
        view = _bound_view(port, intent)
        evidence["bound_view"] = view
        evidence["closure_target_proof"] = _required_proof(
            port, "verify_closure_target", "CLOSURE_TARGET_UNPROVEN", intent, view
        )
        phase = "closure_target_verified"

        if not hasattr(port, "create_closure_anchor"):
            raise TransactionError("CLOSURE_PORT_UNSUPPORTED")
        expected_anchor_ref = f"refs/tags/iterations/{final_version}"
        _authorize(port, intent, "create-tag", expected_anchor_ref)
        closure = _as_mapping(_call(port, "create_closure_anchor", intent), "CLOSURE_IDENTITY_MISMATCH")
        if (closure.get("version", closure.get("final_version")) != final_version
                or not _sha(closure.get("commit", closure.get("anchor_sha")))
                or not _sha(closure.get("tree", closure.get("anchor_tree")))):
            raise TransactionError("CLOSURE_IDENTITY_MISMATCH")
        evidence["closure"] = closure
        phase = "closed"

        if not hasattr(port, "create_anchor"):
            raise TransactionError("ANCHOR_PORT_UNSUPPORTED")
        _authorize(port, intent, "create-tag", expected_anchor_ref)
        anchor = _as_mapping(_call(port, "create_anchor", intent, closure), "ANCHOR_IDENTITY_MISMATCH")
        if (anchor.get("version", anchor.get("final_version")) != final_version
                or anchor.get("ref") != expected_anchor_ref
                or anchor.get("commit", anchor.get("anchor_sha")) != closure.get("commit", closure.get("anchor_sha"))
                or anchor.get("tree", anchor.get("anchor_tree")) != closure.get("tree", closure.get("anchor_tree"))
                or anchor.get("closure_timestamp_utc") != intent.closure_timestamp_utc
                or not _sha(anchor.get("commit", anchor.get("anchor_sha")))
                or not _sha(anchor.get("tree", anchor.get("anchor_tree")))):
            raise TransactionError("ANCHOR_IDENTITY_MISMATCH")
        evidence["anchor"] = anchor
        phase = "anchored"

        post_anchor_view = _successor_view(port, intent, view)
        evidence["post_anchor_view"] = post_anchor_view
        consumption_payload = _manifest_payload(
            intent, "reservation-consumed", {
                "owner_line": intent.owner_line, "final_version": final_version,
                "reserved_final": outgoing, "closed_final_version": final_version,
                "closure_anchor_ref": anchor["ref"],
                "terminal_disposition": ("closed" if outgoing == final_version
                                          else "CONSUMED_UNUSED_DEV_RESERVATION"),
            }, allocation_view=post_anchor_view)
        consumption = _as_mapping(_typed_manifest(
            port, "reservation-consumed", consumption_payload,
            authorize=lambda kind, target, version: _manifest_authorize(port, intent, kind, target, version)),
            "OUTGOING_RESERVATION_RECONCILIATION_FAILED")
        expected_disposition = consumption_payload["terminal_disposition"]
        if (consumption.get("reserved_final") != outgoing
                or consumption.get("closed_final_version") != final_version
                or consumption.get("terminal_disposition") != expected_disposition):
            raise TransactionError("OUTGOING_RESERVATION_RECONCILIATION_FAILED")
        evidence["outgoing_terminal_proof"] = _required_proof(
            port,
            "verify_outgoing_terminal",
            "OUTGOING_RESERVATION_RECONCILIATION_FAILED",
            intent,
            consumption,
        )
        evidence["consumption"] = _expect_manifest(consumption, "reservation-consumed", "OUTGOING_RESERVATION_RECONCILIATION_FAILED")
        phase = "outgoing_terminal"

        fresh_view = _successor_view(port, intent, post_anchor_view)
        evidence["post_consumption_view"] = fresh_view
        occupied = (getattr(fresh_view, "occupied", None)
                     if fresh_view is not None else None)
        if occupied is None and fresh_view is not None:
            occupied = getattr(fresh_view, "occupied_versions", ())
        next_final = _next_principal(final_version, occupied or ())
        evidence["next_final"] = next_final
        preparation_payload = _manifest_payload(
            intent, "reservation-prepared", {
                "owner_line": intent.owner_line, "final_version": next_final,
                "reserved_final": next_final,
                "intended_dev_version": f"{next_final}-DEV",
                "expected_line_head": intent.expected_line_head,
            }, predecessor_refs=(lifecycle_ref_for_manifest(consumption),),
            allocation_view=fresh_view)
        preparation = _as_mapping(_typed_manifest(
            port, "reservation-prepared", preparation_payload,
            authorize=lambda kind, target, version: _manifest_authorize(port, intent, kind, target, version)),
            "NEXT_RESERVATION_MISMATCH")
        if (preparation.get("version", preparation.get("final_version")) != next_final
                and preparation.get("reserved_final") != next_final):
            raise TransactionError("NEXT_RESERVATION_MISMATCH")
        if preparation.get("intended_dev", preparation.get("intended_dev_version")) != f"{next_final}-DEV":
            raise TransactionError("NEXT_RESERVATION_MISMATCH")
        evidence["preparation"] = _expect_manifest(preparation, "reservation-prepared", "NEXT_RESERVATION_MISMATCH")
        phase = "next_prepared"

        post_preparation_view = _successor_view(port, intent, fresh_view)
        evidence["post_preparation_view"] = post_preparation_view

        if not hasattr(port, "reopen_dev"):
            raise TransactionError("REOPEN_PORT_UNSUPPORTED")
        _authorize(
            port, intent, "create-release-manifest", _line_target(intent),
            final_version=next_final,
        )
        reopened = _as_mapping(_call(port, "reopen_dev", intent, preparation), "REOPEN_IDENTITY_MISMATCH")
        if reopened.get("version", reopened.get("actual_dev_version")) != f"{next_final}-DEV":
            raise TransactionError("REOPEN_IDENTITY_MISMATCH")
        evidence["reopened"] = reopened
        phase = "dev_reopened"
        if not hasattr(port, "activate_next"):
            raise TransactionError("ACTIVATION_PORT_UNSUPPORTED")
        _authorize(
            port, intent, "create-release-manifest", _line_target(intent),
            final_version=next_final,
        )
        activation = _as_mapping(_call(port, "activate_next", intent, preparation, reopened), "ACTIVATION_IDENTITY_MISMATCH")
        if (activation.get("head") is not None and reopened.get("head") is not None
                and activation.get("head") != reopened.get("head")):
            raise TransactionError("ACTIVATION_IDENTITY_MISMATCH")
        evidence["activation"] = activation
        phase = "next_active"
        post_activation_view = _successor_view(
            port, intent, post_preparation_view
        )
        evidence["post_activation_view"] = post_activation_view
        opened_payload = _manifest_payload(
            intent, "reservation-opened", {
                "owner_line": intent.owner_line, "final_version": next_final,
                "reserved_final": next_final,
                "intended_dev_version": f"{next_final}-DEV",
                "actual_dev_head": activation.get("head", reopened.get("head")),
            }, predecessor_refs=(lifecycle_ref_for_manifest(preparation),),
            allocation_view=post_activation_view)
        opened = _as_mapping(_typed_manifest(
            port, "reservation-opened", opened_payload,
            authorize=lambda kind, target, version: _manifest_authorize(port, intent, kind, target, version)),
            "RESERVATION_OPEN_MANIFEST_MISMATCH")
        evidence["reservation_opened"] = opened
        phase = "reservation_opened"
        post_opened_view = _successor_view(port, intent, post_activation_view)
        evidence["post_opened_view"] = post_opened_view
        claim_payload = _manifest_payload(
            intent, "version-claimed", {
                "owner_line": intent.owner_line,
                "final_version": next_final,
            }, predecessor_refs=(lifecycle_ref_for_manifest(opened),),
            allocation_view=post_opened_view)
        claim = _as_mapping(_typed_manifest(
            port, "version-claimed", claim_payload,
            authorize=lambda kind, target, version: _manifest_authorize(
                port, intent, kind, target, version
            )), "VERSION_CLAIM_MISMATCH")
        evidence["version_claimed"] = _expect_manifest(
            claim, "version-claimed", "VERSION_CLAIM_MISMATCH"
        )
        phase = "version_claimed"
        evidence["terminal_allocation_view"] = _successor_view(
            port, intent, post_opened_view
        )
        evidence["closure_correspondence"] = _required_proof(
            port,
            "verify_closure_correspondence",
            "CLOSURE_CORRESPONDENCE_UNPROVEN",
            intent,
            anchor,
            consumption,
            preparation,
            reopened,
            activation,
            claim,
        )
        phase = "correspondence_verified"

        lease_to_release = lease
        lease = None
        _release_exclusion(port, lease_to_release)
        _call(port, "unfreeze_line", token)
        token = None
        return TransactionResult("COMPLETE", evidence=evidence, phase="unfrozen")
    except TransactionError as error:
        try:
            if lease is not None:
                _release_exclusion(port, lease)
        except Exception:
            pass
        invalid = {"CLOSURE_IDENTITY_MISMATCH", "ANCHOR_IDENTITY_MISMATCH",
                   "NEXT_RESERVATION_MISMATCH", "ACTIVATION_IDENTITY_MISMATCH",
                   "REOPEN_IDENTITY_MISMATCH"}
        return TransactionResult("INVALID" if error.reason_code in invalid else "BLOCKED",
                                 error.reason_code, error.detail, evidence,
                                 frozen=token is not None, phase=phase)
    except Exception as error:
        try:
            if lease is not None:
                _release_exclusion(port, lease)
        except Exception:
            pass
        evidence["error_type"] = type(error).__name__
        return TransactionResult("BLOCKED", "TRANSACTION_OUTCOME_UNCERTAIN",
                                 str(error), evidence, frozen=token is not None,
                                 phase=phase)


def _validate_certification(record: Any, candidate: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        raise TransactionError("CERTIFICATION_IDENTITY_UNPROVEN")
    binding = record.get("binding", record.get("certification_binding"))
    if binding not in {"tree-bound", "commit-bound"}:
        raise TransactionError("UNSUPPORTED_CERTIFICATION_BINDING")
    subject_sha = record.get("subject_sha", record.get("certification_subject_sha"))
    subject_tree = record.get("subject_tree", record.get("certification_subject_tree"))
    if (subject_sha != candidate.get("sha") or subject_tree != candidate.get("tree")
            or not _sha(subject_sha) or not _sha(subject_tree)):
        raise TransactionError("CERTIFICATION_IDENTITY_UNPROVEN")
    for key in ("policy_revision", "harness_revision", "environment"):
        if (
            not isinstance(record.get(key), str)
            or not record[key]
            or not is_safe_public_value(record[key], key=key)
        ):
            raise TransactionError("CERTIFICATION_IDENTITY_UNPROVEN")
    if not _validated_evidence_refs(
        record.get("evidence_refs"), "CERTIFICATION_IDENTITY_UNPROVEN"
    ):
        raise TransactionError("CERTIFICATION_IDENTITY_UNPROVEN")
    return binding


def _validated_evidence_refs(value: Any, reason: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
        or not is_safe_public_value(value, key="evidence_refs")
    ):
        raise TransactionError(reason)
    return list(value)


def run_release(port: Any, intent: ReleaseIntent) -> TransactionResult:
    """Run principal candidate → intent → tag → release → publication."""

    phase = "preflight"
    evidence: dict[str, Any] = {}
    token: Any = None
    lease: Any = None
    tag_created = False
    released_created = False
    try:
        if (intent.release_line or intent.owner_line) != "principal":
            raise TransactionError("DEFERRED_MAINTENANCE_RELEASE")
        final_version = _final(intent.final_version)
        if intent.public_tag != f"v{final_version}":
            raise TransactionError("PUBLIC_TAG_IDENTITY_MISMATCH")
        _utc(intent.closure_timestamp_utc)
        _utc(intent.timestamp_utc)
        if (intent.anchor_ref != f"refs/tags/iterations/{final_version}"
                or not _sha(intent.anchor_sha) or not _sha(intent.anchor_tree)
                or not _sha(intent.candidate_sha) or not _sha(intent.candidate_tree)
                or not re.fullmatch(r"[0-9a-f]{64}", intent.static_snapshot_digest)
                or not re.fullmatch(r"[0-9a-f]{64}", intent.lifecycle_snapshot_digest)
                or not intent.owner_authorization or not intent.owner_authorization_ref
                or not re.fullmatch(r"[0-9a-f]{64}", intent.owner_authorization_digest)
                or not intent.repository or not intent.predecessor_refs):
            raise TransactionError("OWNER_AUTHORIZATION_UNVERIFIED")
        if not hasattr(port, "verify_anchor"):
            raise TransactionError("ANCHOR_PORT_UNSUPPORTED")
        anchor = _call(port, "verify_anchor", intent)
        if not isinstance(anchor, dict):
            raise TransactionError("ANCHOR_IDENTITY_MISMATCH")
        if (anchor.get("ref") != intent.anchor_ref
                or anchor.get("tree", anchor.get("anchor_tree")) != intent.anchor_tree
                or anchor.get("sha", anchor.get("anchor_sha")) != intent.anchor_sha
                or anchor.get("closure_timestamp_utc")
                != intent.closure_timestamp_utc):
            raise TransactionError("ANCHOR_IDENTITY_MISMATCH")
        evidence["anchor"] = anchor
        phase = "anchor_verified"
        lease = _acquire_exclusion(port, intent)
        phase = "serialized"
        bound_view = _bound_view(port, intent)
        evidence["bound_view"] = bound_view
        _authorize(port, intent, "create-release-manifest", intent.candidate_ref)

        if hasattr(port, "make_durable_candidate"):
            candidate = _as_mapping(_call(port, "make_durable_candidate", intent), "CANDIDATE_DURABILITY_UNPROVEN")
        elif hasattr(port, "create_candidate"):
            candidate = _as_mapping(_call(port, "create_candidate", intent), "CANDIDATE_DURABILITY_UNPROVEN")
        else:
            raise TransactionError("CANDIDATE_PORT_UNSUPPORTED")
        candidate_sha = candidate.get("sha", candidate.get("candidate_sha"))
        candidate_tree = candidate.get("tree", candidate.get("candidate_tree"))
        if (candidate.get("durable") is not True
                or candidate.get("ref", candidate.get("candidate_ref")) != intent.candidate_ref
                or candidate.get("version", candidate.get("final_version")) != final_version
                or not _sha(candidate_sha) or not _sha(candidate_tree)
                or (intent.candidate_sha and intent.candidate_sha != candidate_sha)
                or (intent.candidate_tree and intent.candidate_tree != candidate_tree)):
            raise TransactionError("CANDIDATE_DURABILITY_UNPROVEN")
        if intent.anchor_tree and candidate_tree != intent.anchor_tree:
            raise TransactionError("CANDIDATE_ANCHOR_MISMATCH")
        evidence["candidate"] = candidate
        phase = "candidate_durable"

        main_candidate_sha = candidate.get("main_at_candidate_sha")
        main_candidate_version = candidate.get("main_at_candidate_version")
        try:
            _final(main_candidate_version)
        except TransactionError as error:
            raise TransactionError("CANDIDATE_MAIN_IDENTITY_UNPROVEN") from error
        if not _sha(main_candidate_sha):
            raise TransactionError("CANDIDATE_MAIN_IDENTITY_UNPROVEN")
        opened_payload = _manifest_payload(
            intent, "candidate-opened", {
                "candidate_id": candidate.get("candidate_id", ""),
                "candidate_ref": intent.candidate_ref,
                "candidate_sha": candidate_sha,
                "candidate_tree": candidate_tree,
                "final_version": final_version,
                "release_line": "principal",
                "anchor_ref": intent.anchor_ref,
                "anchor_sha": intent.anchor_sha,
                "anchor_tree": intent.anchor_tree,
                "main_at_candidate_sha": main_candidate_sha,
                "main_at_candidate_version": main_candidate_version,
            }, predecessor_refs=intent.predecessor_refs,
            allocation_view=bound_view)
        opened = _as_mapping(_typed_manifest(
            port, "candidate-opened", opened_payload,
            authorize=lambda kind, target, version: _manifest_authorize(port, intent, kind, target, version)),
            "CANDIDATE_OPEN_MANIFEST_MISMATCH")
        if (opened.get("candidate_id") not in {None, candidate.get("candidate_id")}
                or opened.get("candidate_sha") not in {None, candidate_sha}):
            raise TransactionError("CANDIDATE_OPEN_MANIFEST_MISMATCH")
        evidence["candidate_opened"] = _expect_manifest(opened, "candidate-opened", "CANDIDATE_OPEN_MANIFEST_MISMATCH")
        phase = "candidate_opened"

        if not hasattr(port, "certify_candidate"):
            raise TransactionError("CERTIFICATION_PORT_UNSUPPORTED")
        certification = _as_mapping(_call(port, "certify_candidate", intent, candidate), "CERTIFICATION_IDENTITY_UNPROVEN")
        binding = _validate_certification(certification, {"sha": candidate_sha, "tree": candidate_tree})
        evidence["candidate_certification"] = dict(certification)
        phase = "candidate_certified"

        if not hasattr(port, "freeze_main"):
            raise TransactionError("MAIN_FREEZE_UNAVAILABLE")
        freeze = _as_mapping(_call(port, "freeze_main", intent), "MAIN_FREEZE_UNAVAILABLE")
        token = freeze.get("token")
        if not token or not _sha(freeze.get("sha")):
            raise TransactionError("MAIN_FREEZE_UNAVAILABLE")
        evidence["main_freeze"] = freeze
        phase = "main_frozen"
        if not hasattr(port, "verify_principal_interval"):
            raise TransactionError("PRINCIPAL_ANCESTRY_UNPROVEN")
        interval = _as_mapping(_call(port, "verify_principal_interval", intent, candidate, freeze), "PRINCIPAL_ANCESTRY_UNPROVEN")
        commits = interval.get("intervening_commits")
        disposition = interval.get("disposition")
        if (interval.get("verified") is not True or not isinstance(commits, list)
                or len({row.get("sha") for row in commits if isinstance(row, dict)}) != len(commits)
                or any(not isinstance(row, dict) or not _sha(row.get("sha")) or not _sha(row.get("tree")) for row in commits)
                or disposition not in {"no_drift", "tree_neutral_included"}):
            raise TransactionError("PRINCIPAL_ANCESTRY_UNPROVEN")
        evidence["principal_interval"] = interval
        phase = "ancestry_verified"
        if not hasattr(port, "promote_principal"):
            raise TransactionError("RELEASE_PORT_UNSUPPORTED")
        _authorize(port, intent, "create-release-manifest", "refs/heads/main")
        final = _as_mapping(_call(port, "promote_principal", intent, candidate, certification, freeze), "RELEASE_TREE_MISMATCH")
        final_sha = final.get("sha", final.get("final_release_sha"))
        final_tree = final.get("tree", final.get("final_release_tree"))
        if (final.get("version", final.get("final_version")) != final_version
                or not _sha(final_sha) or not _sha(final_tree) or final_tree != candidate_tree):
            raise TransactionError("RELEASE_TREE_MISMATCH")
        try:
            previous_main_version = parse_package_version(
                _final(freeze.get("version"))
            )
            main_at_release_version = _final(final.get("main_at_release_version"))
        except TransactionError as error:
            raise TransactionError("RELEASE_MAIN_IDENTITY_UNPROVEN") from error
        if final.get("main_at_release_sha") != final_sha:
            raise TransactionError("RELEASE_MAIN_IDENTITY_UNPROVEN")
        if (
            parse_package_version(final_version) <= previous_main_version
            or main_at_release_version != final_version
        ):
            raise TransactionError("PRINCIPAL_VERSION_REGRESSION")
        evidence["final"] = final
        phase = "final_verified"

        transfer_evidence: dict[str, Any] | None = None
        if binding == "commit-bound" and final_sha != candidate_sha:
            if not hasattr(port, "recertify_final"):
                raise TransactionError("RECERTIFICATION_REQUIRED")
            certification = _as_mapping(_call(port, "recertify_final", intent, final), "RECERTIFICATION_REQUIRED")
            if (certification.get("binding") != "commit-bound"
                    or certification.get("subject_sha") != final_sha
                    or certification.get("subject_tree") != final_tree):
                raise TransactionError("RECERTIFICATION_REQUIRED")
            _validate_certification(certification, {"sha": final_sha, "tree": final_tree})
            evidence["final_certification"] = dict(certification)
            phase = "final_recertified"
        elif binding == "tree-bound" and final_sha != candidate_sha:
            if not hasattr(port, "verify_tree_transfer"):
                raise TransactionError("CERTIFICATION_TRANSFER_UNPROVEN")
            transfer_evidence = _as_mapping(_call(port, "verify_tree_transfer", intent, candidate, certification, final), "CERTIFICATION_TRANSFER_UNPROVEN")
            expected_anchor_tree = intent.anchor_tree or candidate_tree
            if (transfer_evidence.get("verified") is not True
                    or transfer_evidence.get("candidate_sha") != candidate_sha
                    or transfer_evidence.get("final_release_sha") != final_sha
                    or transfer_evidence.get("candidate_tree") != candidate_tree
                    or transfer_evidence.get("final_release_tree") != final_tree
                    or transfer_evidence.get("anchor_tree") != expected_anchor_tree):
                raise TransactionError("CERTIFICATION_TRANSFER_UNPROVEN")
            certification = dict(certification)
            certification["transfer_evidence"] = dict(transfer_evidence)
            evidence["certification_transfer"] = dict(transfer_evidence)
            evidence["final_certification"] = dict(certification)
            phase = "transfer_verified"
        else:
            evidence["final_certification"] = dict(certification)

        final_evidence_refs = _validated_evidence_refs(
            final.get("evidence_refs"), "RELEASE_EVIDENCE_INVALID"
        )

        if not hasattr(port, "verify_public_tag_ruleset"):
            raise TransactionError("PUBLIC_TAG_RULESET_UNAVAILABLE")
        protection = _call(port, "verify_public_tag_ruleset", intent, f"refs/tags/{intent.public_tag}")
        if protection is None or not hasattr(protection, "require_public_tag"):
            raise TransactionError("PUBLIC_TAG_RULESET_UNAVAILABLE")
        try:
            protection.require_public_tag(f"refs/tags/{intent.public_tag}")
        except Exception as error:
            raise TransactionError("PUBLIC_TAG_RULESET_UNAVAILABLE") from error
        evidence["public_tag_ruleset"] = protection
        phase = "tag_ruleset_verified"

        intent_view = _successor_view(port, intent, bound_view)
        evidence["intent_allocation_view"] = intent_view

        intent_payload = _manifest_payload(
            intent, "release-intent-prepared", {
                "candidate_id": candidate.get("candidate_id", ""),
                "candidate_ref": intent.candidate_ref, "candidate_sha": candidate_sha,
                "candidate_tree": candidate_tree, "final_version": final_version,
                "release_line": "principal", "public_tag": intent.public_tag,
                "anchor_ref": intent.anchor_ref, "anchor_sha": intent.anchor_sha,
                "anchor_tree": intent.anchor_tree,
                "certification_binding": binding,
                "certification_subject_sha": certification.get("subject_sha"),
                "certification_subject_tree": certification.get("subject_tree"),
                "certification_policy_revision": certification["policy_revision"],
                "certification_harness_revision": certification["harness_revision"],
                "certification_environment": certification["environment"],
                "certification_evidence_refs": sorted(certification["evidence_refs"]),
                "final_release_sha": final_sha, "final_release_tree": final_tree,
            }, predecessor_refs=(lifecycle_ref_for_manifest(opened),),
            allocation_view=intent_view)
        if transfer_evidence is not None:
            intent_payload["certification_transfer_evidence"] = dict(transfer_evidence)
        prepared = _as_mapping(_typed_manifest(
            port, "release-intent-prepared", intent_payload,
            authorize=lambda kind, target, version: _manifest_authorize(port, intent, kind, target, version)),
            "RELEASE_INTENT_MISMATCH")
        for key, expected in intent_payload.items():
            if key in {
                "owner_authorization", "owner_authorization_ref",
                "owner_authorization_digest",
            }:
                continue
            if key in prepared and prepared[key] != expected:
                raise TransactionError("RELEASE_INTENT_MISMATCH")
        evidence["intent"] = _expect_manifest(prepared, "release-intent-prepared", "RELEASE_INTENT_MISMATCH")
        phase = "release_intent_durable"

        _authorize(port, intent, "create-tag", f"refs/tags/{intent.public_tag}")
        if hasattr(port, "create_protected_tag"):
            tag = _as_mapping(_call(port, "create_protected_tag", intent, prepared, final, evidence.get("public_tag_ruleset")), "PUBLIC_TAG_IDENTITY_MISMATCH")
        elif hasattr(port, "create_tag"):
            tag = _as_mapping(_call(port, "create_tag", intent, prepared, final), "PUBLIC_TAG_IDENTITY_MISMATCH")
        else:
            raise TransactionError("TAG_PORT_UNSUPPORTED")
        tag_created = True
        if (tag.get("name", tag.get("public_tag")) != intent.public_tag
                or tag.get("commit", tag.get("tag_commit")) != final_sha
                or tag.get("tree", tag.get("tag_tree")) != final_tree
                or tag.get("protected") is False):
            raise TransactionError("PUBLIC_TAG_IDENTITY_MISMATCH")
        evidence["tag"] = tag
        phase = "public_tag_created"
        released_view = _successor_view(port, intent, intent_view)
        evidence["released_allocation_view"] = released_view

        release_fields = {
            "final_version": final_version, "release_line": "principal",
            "public_tag": intent.public_tag, "candidate_ref": intent.candidate_ref,
            "candidate_sha": candidate_sha, "candidate_tree": candidate_tree,
            "anchor_ref": intent.anchor_ref, "anchor_sha": intent.anchor_sha,
            "anchor_tree": intent.anchor_tree, "final_release_sha": final_sha,
            "final_release_tree": final_tree, "certification_binding": binding,
            "certification_subject_sha": certification["subject_sha"],
            "certification_subject_tree": certification["subject_tree"],
            "certification_policy_revision": certification["policy_revision"],
            "certification_harness_revision": certification["harness_revision"],
            "certification_environment": certification["environment"],
            "certification_evidence_refs": sorted(certification["evidence_refs"]),
            "closure_timestamp_utc": intent.closure_timestamp_utc,
            "previous_main_sha": freeze["sha"],
            "previous_main_version": freeze.get("version"),
            "main_at_release_sha": final.get("main_at_release_sha"),
            "main_at_release_version": final.get("main_at_release_version"),
            "main_at_candidate_sha": main_candidate_sha,
            "main_at_candidate_version": main_candidate_version,
            "evidence_refs": sorted(final_evidence_refs),
        }
        if transfer_evidence is not None:
            release_fields["certification_transfer_evidence"] = dict(
                transfer_evidence
            )
        if not (_sha(release_fields["previous_main_sha"])
                and isinstance(release_fields["previous_main_version"], str)
                and _sha(release_fields["main_at_release_sha"])
                and isinstance(release_fields["main_at_release_version"], str)
                and release_fields["evidence_refs"]):
            raise TransactionError("RELEASE_MAIN_IDENTITY_UNPROVEN")
        released_payload = _manifest_payload(
            intent, "released", release_fields,
            predecessor_refs=(lifecycle_ref_for_manifest(prepared),),
            allocation_view=released_view)
        released = _as_mapping(_typed_manifest(
            port, "released", released_payload,
            authorize=lambda kind, target, version: _manifest_authorize(port, intent, kind, target, version)),
            "RELEASED_MANIFEST_MISMATCH")
        released_created = True
        for key, expected in release_fields.items():
            if key in released and released[key] != expected:
                raise TransactionError("RELEASED_MANIFEST_MISMATCH")
        evidence["released"] = _expect_manifest(released, "released", "RELEASED_MANIFEST_MISMATCH")
        evidence["released_proof"] = _required_proof(
            port,
            "verify_released",
            "RELEASED_STATE_UNPROVEN",
            intent,
            released,
            certification,
            final,
        )
        phase = "released_manifest_durable"
        evidence["post_released_allocation_view"] = _successor_view(
            port, intent, released_view
        )

        if not hasattr(port, "publish_github_release"):
            raise TransactionError("PUBLICATION_RECONCILIATION_REQUIRED")
        _authorize(port, intent, "create-release-manifest", f"refs/tags/{intent.public_tag}")
        publication = _as_mapping(_call(port, "publish_github_release", intent, released), "PUBLICATION_IDENTITY_MISMATCH")
        release_id = publication.get("id", publication.get("github_release_id"))
        if (publication.get("tag", publication.get("public_tag")) != intent.public_tag
                or isinstance(release_id, bool) or not isinstance(release_id, int) or release_id <= 0):
            raise TransactionError("PUBLICATION_IDENTITY_MISMATCH")
        evidence["github_release"] = publication
        phase = "github_release_published"
        if not hasattr(port, "persist_publication_evidence"):
            raise TransactionError("PUBLICATION_EVIDENCE_UNAVAILABLE")
        if not hasattr(port, "publication_evidence_target"):
            raise TransactionError("PUBLICATION_EVIDENCE_UNAVAILABLE")
        evidence_target = _call(
            port, "publication_evidence_target", intent, released, publication
        )
        if not isinstance(evidence_target, str) or not evidence_target:
            raise TransactionError("PUBLICATION_EVIDENCE_UNAVAILABLE")
        _authorize(port, intent, "create-release-manifest", evidence_target)
        publication_evidence = _as_mapping(_call(port, "persist_publication_evidence", intent, released, publication), "PUBLICATION_EVIDENCE_MISMATCH")
        if (publication_evidence.get("ref") != evidence_target
                or not isinstance(publication_evidence.get("digest"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", publication_evidence["digest"])):
            raise TransactionError("PUBLICATION_EVIDENCE_MISMATCH")
        evidence["publication_evidence"] = publication_evidence
        release_ref = lifecycle_ref_for_manifest(released)
        publication_payload = {
            "schema_version": 1, "manifest_type": "publication",
            "timestamp_utc": publication.get("published_at_utc", intent.timestamp_utc),
            "predecessor_refs": [release_ref],
            "owner_authorization": intent.owner_authorization,
            "owner_authorization_ref": intent.owner_authorization_ref,
            "owner_authorization_digest": intent.owner_authorization_digest,
            "released_manifest_ref": release_ref,
            "released_manifest_id": released["manifest_id"],
            "released_manifest_digest": sha256_hex(canonical_manifest_bytes(released)),
            "public_tag": intent.public_tag,
            "tag_commit": tag.get("commit", tag.get("tag_commit")),
            "tag_tree": tag.get("tree", tag.get("tag_tree")),
            "github_release_id": release_id,
            "github_release_url": publication.get("url", publication.get("github_release_url", "")),
            "published_at_utc": publication.get("published_at_utc", intent.timestamp_utc),
            "publication_evidence_ref": publication_evidence["ref"],
            "publication_evidence_digest": publication_evidence["digest"],
        }
        evidence["publication"] = _as_mapping(_typed_manifest(
            port, "publication", publication_payload,
            authorize=lambda kind, target, version: _manifest_authorize(port, intent, kind, target, version)),
            "PUBLICATION_MANIFEST_MISMATCH")
        evidence["terminal_proof"] = _required_proof(
            port,
            "verify_terminal",
            "TERMINAL_CONSISTENCY_UNPROVEN",
            intent,
            released,
            publication,
            publication_evidence,
        )
        phase = "terminal_verified"

        lease_to_release = lease
        lease = None
        _release_exclusion(port, lease_to_release)
        _call(port, "unfreeze_main", token)
        token = None
        return TransactionResult("COMPLETE", evidence=evidence, phase="unfrozen")
    except TransactionError as error:
        try:
            if lease is not None:
                _release_exclusion(port, lease)
        except Exception:
            pass
        if error.reason_code == "OWNER_AUTHORIZATION_UNVERIFIED":
            status = "BLOCKED"
        elif error.reason_code.endswith("_MISMATCH"):
            status = "INVALID"
        elif released_created:
            status = "publication_reconciliation_pending"
        elif tag_created:
            status = "tag_reconciliation_pending"
        else:
            status = "BLOCKED"
        return TransactionResult(status, error.reason_code, error.detail, evidence, frozen=token is not None, phase=phase)
    except Exception as error:
        try:
            if lease is not None:
                _release_exclusion(port, lease)
        except Exception:
            pass
        evidence["error_type"] = type(error).__name__
        status = ("publication_reconciliation_pending" if released_created else
                  "tag_reconciliation_pending" if tag_created else "BLOCKED")
        return TransactionResult(status, "RELEASE_OUTCOME_UNCERTAIN", str(error), evidence, frozen=token is not None, phase=phase)


__all__ = [
    "AllocationView", "BootstrapIntent", "ClosureIntent", "ImmutableLifecyclePort",
    "ReleaseIntent", "TransactionError", "TransactionResult", "run_closure",
    "run_maintenance_bootstrap", "run_rare_recovery", "run_release",
]
