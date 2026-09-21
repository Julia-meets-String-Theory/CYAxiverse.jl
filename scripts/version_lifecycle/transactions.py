"""Guarded closure and maintenance-bootstrap lifecycle orchestration.

The port performs repository mutations and returns durable proofs. This
controller checks their correspondence and never releases a freeze after an
uncertain or incomplete transaction. A production port must use verified live
protections; fixture ports are used for Gate A tests.
"""

from __future__ import annotations

import re
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .certification import is_safe_public_value
from .git_refs import GitIdentityError, ProtectionEvidence
from .versions import MAX_VERSION_COMPONENT, maintenance_line, parse_package_version


UTC = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
SHA = re.compile(r"^[0-9a-f]{40}$")


def _public_identity(value: Any, *, key: str) -> bool:
    """Return whether an identity is nonempty and safe for public evidence."""

    return (
        isinstance(value, str)
        and bool(value)
        and is_safe_public_value(value, key=key)
    )


def _git_sha(value: Any) -> bool:
    """Return whether a value is a complete lowercase Git object identity."""

    return isinstance(value, str) and SHA.fullmatch(value) is not None


class TransactionError(RuntimeError):
    def __init__(self, reason_code: str, detail: str = "") -> None:
        super().__init__(detail or reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class AllocationView:
    static_snapshot_digest: str
    event_head: str
    occupied_versions: frozenset[str]


@dataclass(frozen=True)
class ClosureIntent:
    transaction_id: str
    line: str
    final_version: str
    outgoing_reserved_final: str
    expected_line_head: str
    closure_timestamp_utc: str


@dataclass(frozen=True)
class BootstrapIntent:
    transaction_id: str
    line: str
    approved_base_sha: str
    expected_main_sha: str
    expected_main_version: str


@dataclass
class TransactionResult:
    status: str
    phase: str
    reason_code: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    frozen: bool = False


class ClosurePort(Protocol):
    def freeze_line(self, intent: ClosureIntent) -> str: ...
    def acquire_static_mutation(self, intent: ClosureIntent) -> Any: ...
    def release_static_mutation(self, lease: Any) -> None: ...
    def allocation_view(self) -> AllocationView: ...
    def verify_closure_target(self, intent: ClosureIntent, view: AllocationView) -> None: ...
    def merge_final(self, intent: ClosureIntent) -> dict[str, str]: ...
    def create_anchor(self, intent: ClosureIntent, closure: dict[str, str]) -> dict[str, str]: ...
    def consume_outgoing(
        self, intent: ClosureIntent, anchor: dict[str, str], view: AllocationView
    ) -> dict[str, str]: ...
    def verify_outgoing_terminal(self, intent: ClosureIntent, consumption: dict[str, str]) -> None: ...
    def prepare_next(
        self, intent: ClosureIntent, next_final: str, view: AllocationView
    ) -> dict[str, str]: ...
    def reopen_dev(self, intent: ClosureIntent, preparation: dict[str, str]) -> dict[str, str]: ...
    def activate_next(
        self, intent: ClosureIntent, preparation: dict[str, str], reopened: dict[str, str]
    ) -> dict[str, str]: ...
    def verify_closure_correspondence(
        self, intent: ClosureIntent, anchor: dict[str, str],
        consumption: dict[str, str], preparation: dict[str, str],
        reopened: dict[str, str], activation: dict[str, str],
    ) -> None: ...
    def unfreeze_line(self, token: str) -> None: ...


class BootstrapPort(Protocol):
    def freeze_bootstrap(self, intent: BootstrapIntent) -> str: ...
    def acquire_static_mutation(self, intent: BootstrapIntent) -> Any: ...
    def release_static_mutation(self, lease: Any) -> None: ...
    def verify_base_and_absence(self, intent: BootstrapIntent) -> None: ...
    def allocation_view(self) -> AllocationView: ...
    def prepare_reservation(
        self, intent: BootstrapIntent, final_version: str, view: AllocationView
    ) -> dict[str, str]: ...
    def create_line_if_absent(
        self, intent: BootstrapIntent, preparation: dict[str, str]
    ) -> dict[str, str]: ...
    def abort_nonentry(
        self, intent: BootstrapIntent, preparation: dict[str, str],
        branch_proof: dict[str, str],
    ) -> None: ...
    def install_dev(
        self, intent: BootstrapIntent, branch: dict[str, str],
        preparation: dict[str, str],
    ) -> dict[str, str]: ...
    def activate_reservation(
        self, intent: BootstrapIntent, preparation: dict[str, str],
        dev_head: dict[str, str],
    ) -> dict[str, str]: ...
    def record_line_opened(
        self, intent: BootstrapIntent, branch: dict[str, str],
        activation: dict[str, str],
    ) -> dict[str, str]: ...
    def verify_bootstrap_correspondence(
        self, intent: BootstrapIntent, preparation: dict[str, str],
        branch: dict[str, str], dev_head: dict[str, str],
        activation: dict[str, str], line_opened: dict[str, str],
    ) -> None: ...
    def unfreeze_bootstrap(self, token: str) -> None: ...


@dataclass(frozen=True)
class ReleaseIntent:
    transaction_id: str
    release_line: str
    final_version: str
    anchor_ref: str
    anchor_sha: str
    anchor_tree: str
    closure_timestamp_utc: str
    candidate_ref: str


class ReleasePort(Protocol):
    def acquire_static_mutation(self, intent: ReleaseIntent) -> Any: ...
    def release_static_mutation(self, lease: Any) -> None: ...
    def verify_anchor(self, intent: ReleaseIntent) -> None: ...
    def make_durable_candidate(self, intent: ReleaseIntent) -> dict[str, Any]: ...
    def append_candidate_opened(
        self, intent: ReleaseIntent, candidate: dict[str, Any]
    ) -> dict[str, Any]: ...
    def certify_candidate(
        self, intent: ReleaseIntent, candidate: dict[str, Any]
    ) -> dict[str, Any]: ...
    def freeze_main(self, intent: ReleaseIntent) -> dict[str, Any]: ...
    def verify_principal_interval(
        self, intent: ReleaseIntent, candidate: dict[str, Any],
        freeze: dict[str, Any],
    ) -> dict[str, Any]: ...
    def promote_principal(
        self, intent: ReleaseIntent, candidate: dict[str, Any],
        certification: dict[str, Any], freeze: dict[str, Any],
    ) -> dict[str, Any]: ...
    def verify_maintenance(
        self, intent: ReleaseIntent, candidate: dict[str, Any],
        certification: dict[str, Any],
    ) -> dict[str, Any]: ...
    def recertify_final(
        self, intent: ReleaseIntent, final: dict[str, Any]
    ) -> dict[str, Any]: ...
    def verify_tree_transfer(
        self, intent: ReleaseIntent, candidate: dict[str, Any],
        certification: dict[str, Any], final: dict[str, Any],
    ) -> dict[str, Any]: ...
    def append_release_intent(
        self, intent: ReleaseIntent, candidate: dict[str, Any],
        certification: dict[str, Any], final: dict[str, Any],
    ) -> dict[str, Any]: ...
    def verify_public_tag_ruleset(
        self, intent: ReleaseIntent, tag_ref: str
    ) -> ProtectionEvidence: ...
    def create_protected_tag(
        self, intent: ReleaseIntent, prepared: dict[str, Any],
        final: dict[str, Any], protection: ProtectionEvidence,
    ) -> dict[str, Any]: ...
    def append_released(
        self, intent: ReleaseIntent, candidate: dict[str, Any],
        certification: dict[str, Any], final: dict[str, Any],
        tag: dict[str, Any],
    ) -> dict[str, Any]: ...
    def verify_released(
        self, intent: ReleaseIntent, event: dict[str, Any],
        certification: dict[str, Any], final: dict[str, Any],
    ) -> None: ...
    def publish_github_release(
        self, intent: ReleaseIntent, event: dict[str, Any]
    ) -> dict[str, Any]: ...
    def persist_publication_evidence(
        self, intent: ReleaseIntent, event: dict[str, Any],
        publication: dict[str, Any],
    ) -> dict[str, Any]: ...
    def verify_terminal(
        self, intent: ReleaseIntent, event: dict[str, Any],
        publication: dict[str, Any], evidence: dict[str, Any],
    ) -> None: ...
    def unfreeze_main(self, token: str) -> None: ...


def _version(value: str) -> tuple[int, int, int]:
    try:
        version = parse_package_version(value)
    except (TypeError, ValueError) as exc:
        raise TransactionError("INVALID_FINAL_VERSION") from exc
    if not version.is_final:
        raise TransactionError("INVALID_FINAL_VERSION")
    return version.tuple


def _canonical_final_version(value: Any) -> str | None:
    """Normalize one final package identity through the shared parser."""

    try:
        parsed = parse_package_version(value)
    except (TypeError, ValueError):
        return None
    return parsed.canonical if parsed.is_final else None


def _next_final(line: str, closed: str, view: AllocationView) -> str:
    major, minor, patch = _version(closed)
    if line == "principal":
        proposed = f"{major}.{minor}.{patch + 1}"
        try:
            parse_package_version(proposed)
        except (TypeError, ValueError) as exc:
            raise TransactionError("PRINCIPAL_VERSION_EXHAUSTED") from exc
        if proposed in view.occupied_versions:
            raise TransactionError("PRINCIPAL_SENTINEL_UNAVAILABLE")
        return proposed
    expected = f"maintenance/{major}.{minor}"
    if line != expected:
        raise TransactionError("MAINTENANCE_LINE_MISMATCH")
    candidate = patch + 1
    while True:
        proposed = f"{major}.{minor}.{candidate}"
        try:
            parse_package_version(proposed)
        except (TypeError, ValueError) as exc:
            raise TransactionError("MAINTENANCE_PATCH_EXHAUSTED") from exc
        if proposed not in view.occupied_versions:
            return proposed
        candidate += 1


def _acquire_static_mutation(port: Any, intent: Any) -> Any:
    """Acquire the shared static/event mutation boundary or block."""

    try:
        lease = port.acquire_static_mutation(intent)
    except Exception as exc:
        raise TransactionError("EXCLUSION_UNAVAILABLE", str(exc)) from exc
    if lease is None:
        raise TransactionError("EXCLUSION_UNAVAILABLE")
    return lease


def _release_static_mutation(port: Any, lease: Any) -> None:
    try:
        port.release_static_mutation(lease)
    except Exception as exc:
        raise TransactionError("EXCLUSION_UNAVAILABLE", str(exc)) from exc


def _release_static_mutation_best_effort(port: Any, lease: Any) -> None:
    if lease is None:
        return
    try:
        port.release_static_mutation(lease)
    except Exception:
        # The enclosing transaction remains frozen on every error path. The
        # OS releases a process-owned file lock when the process exits.
        pass


def run_closure(port: ClosurePort, intent: ClosureIntent) -> TransactionResult:
    """Run closure through deterministic reopen; retain freeze on any error."""
    _version(intent.final_version)
    _version(intent.outgoing_reserved_final)
    if not UTC.fullmatch(intent.closure_timestamp_utc):
        raise TransactionError("INVALID_CLOSURE_TIMESTAMP")
    try:
        datetime.strptime(intent.closure_timestamp_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise TransactionError("INVALID_CLOSURE_TIMESTAMP") from exc
    phase = "preflight"
    evidence: dict[str, Any] = {}
    token: str | None = None
    mutation_lease: Any = None
    try:
        token = port.freeze_line(intent)
        if not token:
            raise TransactionError("LINE_FREEZE_UNAVAILABLE")
        evidence["freeze_token"] = token
        phase = "frozen"
        mutation_lease = _acquire_static_mutation(port, intent)
        phase = "exclusion_acquired"
        view = port.allocation_view()
        port.verify_closure_target(intent, view)
        evidence["bound_view"] = view
        phase = "view_bound"
        closure = port.merge_final(intent)
        if (
            closure.get("version") != intent.final_version
            or not _git_sha(closure.get("commit"))
            or not _git_sha(closure.get("tree"))
        ):
            raise TransactionError("CLOSURE_IDENTITY_MISMATCH")
        evidence["closure"] = closure
        phase = "closed"
        anchor = port.create_anchor(intent, closure)
        if (
            anchor.get("version") != intent.final_version
            or not _git_sha(anchor.get("commit"))
            or not _git_sha(anchor.get("tree"))
            or anchor.get("commit") != closure["commit"]
            or anchor.get("tree") != closure["tree"]
            or anchor.get("closure_timestamp_utc") != intent.closure_timestamp_utc
        ):
            raise TransactionError("ANCHOR_IDENTITY_MISMATCH")
        evidence["anchor"] = anchor
        phase = "anchored"
        # Anchor creation changes the static namespace.  Consumption must
        # bind a fresh static/event pair rather than the pre-closure view.
        post_anchor_view = port.allocation_view()
        evidence["post_anchor_view"] = post_anchor_view
        try:
            consumption = port.consume_outgoing(intent, anchor, post_anchor_view)
            port.verify_outgoing_terminal(intent, consumption)
            expected_disposition = (
                "closed"
                if intent.outgoing_reserved_final == intent.final_version
                else "CONSUMED_UNUSED_DEV_RESERVATION"
            )
            if (
                consumption.get("reserved_final") != intent.outgoing_reserved_final
                or consumption.get("closed_final_version") != intent.final_version
                or consumption.get("terminal_disposition") != expected_disposition
                or not consumption.get("event_id")
            ):
                raise TransactionError("OUTGOING_RESERVATION_RECONCILIATION_FAILED")
        except Exception as exc:
            raise TransactionError(
                "OUTGOING_RESERVATION_RECONCILIATION_FAILED", str(exc)
            ) from exc
        evidence["consumption"] = consumption
        phase = "outgoing_terminal"
        fresh_view = port.allocation_view()
        next_final = _next_final(intent.line, intent.final_version, fresh_view)
        evidence["next_final"] = next_final
        preparation = port.prepare_next(intent, next_final, fresh_view)
        if (
            preparation.get("version") != next_final
            or preparation.get("intended_dev") != f"{next_final}-DEV"
        ):
            raise TransactionError("NEXT_RESERVATION_MISMATCH")
        evidence["preparation"] = preparation
        phase = "next_prepared"
        reopened = port.reopen_dev(intent, preparation)
        if reopened.get("version") != f"{next_final}-DEV":
            raise TransactionError("REOPEN_IDENTITY_MISMATCH")
        evidence["reopened"] = reopened
        phase = "dev_reopened"
        activation = port.activate_next(intent, preparation, reopened)
        if activation.get("head") != reopened.get("head"):
            raise TransactionError("ACTIVATION_IDENTITY_MISMATCH")
        evidence["activation"] = activation
        phase = "next_active"
        port.verify_closure_correspondence(
            intent, anchor, consumption, preparation, reopened, activation
        )
        phase = "correspondence_verified"
        try:
            _release_static_mutation(port, mutation_lease)
        except TransactionError:
            # A failed external release leaves the freeze in place and must
            # not be retried with an uncertain lease state.
            mutation_lease = None
            raise
        mutation_lease = None
        port.unfreeze_line(token)
        return TransactionResult("COMPLETE", "unfrozen", evidence=evidence)
    except TransactionError as exc:
        _release_static_mutation_best_effort(port, mutation_lease)
        return TransactionResult(
            "BLOCKED", phase, exc.reason_code, evidence, frozen=token is not None
        )
    except Exception as exc:
        _release_static_mutation_best_effort(port, mutation_lease)
        evidence["error_type"] = type(exc).__name__
        return TransactionResult(
            "BLOCKED", phase, "TRANSACTION_OUTCOME_UNCERTAIN", evidence,
            frozen=token is not None,
        )


def run_maintenance_bootstrap(
    port: BootstrapPort, intent: BootstrapIntent
) -> TransactionResult:
    """Create a maintenance line only after reservation and freeze converge."""
    try:
        major, minor = maintenance_line(intent.line)
    except (TypeError, ValueError):
        raise TransactionError("MAINTENANCE_LINE_MISMATCH")
    phase = "preflight"
    evidence: dict[str, Any] = {}
    token: str | None = None
    mutation_lease: Any = None
    try:
        port.verify_base_and_absence(intent)
        token = port.freeze_bootstrap(intent)
        if not token:
            raise TransactionError("BOOTSTRAP_FREEZE_UNAVAILABLE")
        evidence["freeze_token"] = token
        phase = "frozen"
        # The first check is advisory.  Recheck under the effective freeze so
        # the approved base and branch absence cannot change between checks.
        mutation_lease = _acquire_static_mutation(port, intent)
        phase = "exclusion_acquired"
        port.verify_base_and_absence(intent)
        view = port.allocation_view()
        evidence["bound_view"] = view
        patch = 0
        while f"{major}.{minor}.{patch}" in view.occupied_versions:
            if patch == MAX_VERSION_COMPONENT:
                raise TransactionError("MAINTENANCE_PATCH_EXHAUSTED")
            patch += 1
        if patch > MAX_VERSION_COMPONENT:
            raise TransactionError("MAINTENANCE_PATCH_EXHAUSTED")
        final = f"{major}.{minor}.{patch}"
        preparation = port.prepare_reservation(intent, final, view)
        if preparation.get("version") != final or preparation.get("intended_dev") != f"{final}-DEV":
            raise TransactionError("BOOTSTRAP_RESERVATION_MISMATCH")
        evidence["preparation"] = preparation
        phase = "reservation_prepared"
        branch = port.create_line_if_absent(intent, preparation)
        evidence["branch"] = branch
        state = branch.get("state")
        if state == "not_created":
            port.abort_nonentry(intent, preparation, branch)
            try:
                _release_static_mutation(port, mutation_lease)
            except TransactionError:
                # The release boundary is now uncertain.  Do not retry a
                # possibly one-shot external lease; retain the freeze.
                mutation_lease = None
                raise
            mutation_lease = None
            return TransactionResult(
                "BLOCKED", phase, "BOOTSTRAP_BRANCH_NOT_CREATED", evidence,
                frozen=True,
            )
        if state != "created" or not branch.get("head"):
            raise TransactionError("BOOTSTRAP_CREATION_UNCERTAIN")
        phase = "branch_created"
        dev_head = port.install_dev(intent, branch, preparation)
        if dev_head.get("version") != f"{final}-DEV" or not dev_head.get("head"):
            raise TransactionError("BOOTSTRAP_DEV_MISMATCH")
        evidence["dev_head"] = dev_head
        phase = "dev_installed"
        activation = port.activate_reservation(intent, preparation, dev_head)
        if activation.get("head") != dev_head["head"]:
            raise TransactionError("BOOTSTRAP_ACTIVATION_MISMATCH")
        evidence["activation"] = activation
        phase = "reservation_active"
        line_opened = port.record_line_opened(intent, branch, activation)
        evidence["line_opened"] = line_opened
        phase = "line_opened"
        port.verify_bootstrap_correspondence(
            intent, preparation, branch, dev_head, activation, line_opened
        )
        phase = "correspondence_verified"
        try:
            _release_static_mutation(port, mutation_lease)
        except TransactionError:
            # A release failure can follow a completed append.  Retain the
            # freeze and avoid a second release attempt with an uncertain
            # external lease state.
            mutation_lease = None
            raise
        mutation_lease = None
        port.unfreeze_bootstrap(token)
        return TransactionResult("COMPLETE", "unfrozen", evidence=evidence)
    except TransactionError as exc:
        _release_static_mutation_best_effort(port, mutation_lease)
        return TransactionResult(
            "BLOCKED", phase, exc.reason_code, evidence, frozen=token is not None
        )
    except Exception as exc:
        _release_static_mutation_best_effort(port, mutation_lease)
        evidence["error_type"] = type(exc).__name__
        return TransactionResult(
            "BLOCKED", phase, "BOOTSTRAP_OUTCOME_UNCERTAIN", evidence,
            frozen=token is not None,
        )


def run_release(port: ReleasePort, intent: ReleaseIntent) -> TransactionResult:
    """Orchestrate one candidate→intent→tag→event→publication path.

    The port must verify GitHub protection/freeze and durable Git identities.
    This controller never withdraws or repoints a tag and retains the main
    freeze until terminal evidence is verified.
    """
    major, minor, _ = _version(intent.final_version)
    if intent.release_line != "principal" and intent.release_line != f"maintenance/{major}.{minor}":
        raise TransactionError("RELEASE_LINE_MISMATCH")
    if not UTC.fullmatch(intent.closure_timestamp_utc):
        raise TransactionError("INVALID_CLOSURE_TIMESTAMP")
    try:
        datetime.strptime(intent.closure_timestamp_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise TransactionError("INVALID_CLOSURE_TIMESTAMP") from exc
    if intent.anchor_ref != f"refs/tags/iterations/{intent.final_version}":
        raise TransactionError("ANCHOR_IDENTITY_MISMATCH")

    phase = "preflight"
    evidence: dict[str, Any] = {}
    main_token: str | None = None
    mutation_lease: Any = None
    tag_attempted = False
    tag_created = False
    released_appended = False
    try:
        port.verify_anchor(intent)
        phase = "anchor_verified"
        mutation_lease = _acquire_static_mutation(port, intent)
        phase = "exclusion_acquired"
        candidate = port.make_durable_candidate(intent)
        if (
            candidate.get("ref") != intent.candidate_ref
            or candidate.get("tree") != intent.anchor_tree
            or candidate.get("version") != intent.final_version
            or not SHA.fullmatch(str(candidate.get("sha", "")))
            or not SHA.fullmatch(str(candidate.get("tree", "")))
            or candidate.get("durable") is not True
        ):
            raise TransactionError("CANDIDATE_DURABILITY_UNPROVEN")
        evidence["candidate"] = candidate
        phase = "candidate_durable"
        opened = port.append_candidate_opened(intent, candidate)
        if opened.get("candidate_sha") != candidate["sha"] or not opened.get("event_id"):
            raise TransactionError("CANDIDATE_OPEN_EVENT_MISMATCH")
        evidence["candidate_opened"] = opened
        phase = "candidate_opened"
        certification = port.certify_candidate(intent, candidate)
        if certification.get("binding") not in ("tree-bound", "commit-bound"):
            raise TransactionError("UNSUPPORTED_CERTIFICATION_BINDING")
        if (
            certification.get("subject_tree") != intent.anchor_tree
            or not SHA.fullmatch(str(certification.get("subject_sha", "")))
            or certification.get("subject_sha") != candidate.get("sha")
            or not _public_identity(
                certification.get("policy_revision"), key="policy_revision"
            )
            or not _public_identity(
                certification.get("harness_revision"), key="harness_revision"
            )
            or not _public_identity(
                certification.get("environment"), key="environment"
            )
            or not certification.get("evidence_refs")
        ):
            raise TransactionError("CERTIFICATION_IDENTITY_UNPROVEN")
        evidence["certification"] = certification
        evidence["candidate_certification"] = dict(certification)
        phase = "candidate_certified"
        transfer_evidence: dict[str, Any] | None = None

        if intent.release_line == "principal":
            candidate_main_version = _canonical_final_version(
                candidate.get("main_at_candidate_version")
            )
            if (
                not _git_sha(candidate.get("main_at_candidate_sha"))
                or candidate_main_version is None
            ):
                raise TransactionError("CANDIDATE_MAIN_IDENTITY_UNPROVEN")
            freeze = port.freeze_main(intent)
            main_token = freeze.get("token")
            freeze_version = _canonical_final_version(freeze.get("version"))
            if (
                not main_token
                or not _git_sha(freeze.get("sha"))
                or freeze_version is None
            ):
                raise TransactionError("MAIN_FREEZE_UNAVAILABLE")
            evidence["main_freeze"] = freeze
            phase = "main_frozen"
            previous = _version(freeze_version)
            if previous >= (major, minor, _version(intent.final_version)[2]):
                raise TransactionError("PRINCIPAL_VERSION_REGRESSION")
            interval = port.verify_principal_interval(intent, candidate, freeze)
            commits = interval.get("intervening_commits")
            disposition = interval.get("disposition")
            if (
                interval.get("verified") is not True
                or interval.get("candidate_main_sha") != candidate["main_at_candidate_sha"]
                or interval.get("freeze_main_sha") != freeze["sha"]
                or not SHA.fullmatch(str(interval.get("candidate_main_tree", "")))
                or not SHA.fullmatch(str(interval.get("freeze_main_tree", "")))
                or not isinstance(commits, list)
                or any(
                    not isinstance(item, dict)
                    or not SHA.fullmatch(str(item.get("sha", "")))
                    or not SHA.fullmatch(str(item.get("tree", "")))
                    for item in commits
                )
                or len({item["sha"] for item in commits}) != len(commits)
                or disposition not in ("no_drift", "tree_neutral_included")
                or (disposition == "no_drift" and (
                    commits
                    or interval["candidate_main_sha"] != interval["freeze_main_sha"]
                    or interval["candidate_main_tree"] != interval["freeze_main_tree"]
                ))
                or (disposition == "tree_neutral_included" and (
                    not commits or interval["freeze_main_tree"] != intent.anchor_tree
                ))
            ):
                raise TransactionError("PRINCIPAL_ANCESTRY_UNPROVEN")
            evidence["principal_interval"] = interval
            phase = "ancestry_verified"
            final = port.promote_principal(intent, candidate, certification, freeze)
            if (
                not _git_sha(final.get("previous_main_sha"))
                or _canonical_final_version(final.get("previous_main_version")) is None
                or not _git_sha(final.get("main_at_event_sha"))
                or _canonical_final_version(final.get("main_at_event_version")) is None
                or final.get("previous_main_sha") != freeze["sha"]
                or final.get("previous_main_version") != freeze_version
                or final.get("main_at_event_sha") != final.get("sha")
                or final.get("main_at_event_version") != intent.final_version
                or final.get("ancestry_disposition") != disposition
            ):
                raise TransactionError("MAIN_RECONCILIATION_UNPROVEN")
        else:
            final = port.verify_maintenance(intent, candidate, certification)
            if (
                not _git_sha(final.get("main_before_sha"))
                or not _git_sha(final.get("main_at_event_sha"))
                or _canonical_final_version(final.get("main_before_version")) is None
                or _canonical_final_version(final.get("main_at_event_version")) is None
                or final.get("main_before_sha") != final.get("main_at_event_sha")
                or final.get("main_before_version") != final.get("main_at_event_version")
            ):
                raise TransactionError("MAINTENANCE_MAIN_CHANGED")
        if (
            final.get("tree") != intent.anchor_tree
            or final.get("version") != intent.final_version
            or not SHA.fullmatch(str(final.get("sha", "")))
            or not SHA.fullmatch(str(final.get("tree", "")))
        ):
            raise TransactionError("RELEASE_TREE_MISMATCH")
        evidence["final"] = final
        phase = "final_verified"

        if certification["binding"] == "commit-bound" and candidate["sha"] != final["sha"]:
            certification = port.recertify_final(intent, final)
            if (
                certification.get("binding") != "commit-bound"
                or certification.get("subject_sha") != final["sha"]
                or certification.get("subject_tree") != final["tree"]
                or not _public_identity(
                    certification.get("policy_revision"), key="policy_revision"
                )
                or not _public_identity(
                    certification.get("harness_revision"), key="harness_revision"
                )
                or not _public_identity(
                    certification.get("environment"), key="environment"
                )
                or not certification.get("evidence_refs")
            ):
                raise TransactionError("RECERTIFICATION_REQUIRED")
            evidence["certification"] = certification
            phase = "final_recertified"
        elif certification["binding"] == "tree-bound":
            if certification["subject_tree"] != final["tree"]:
                raise TransactionError("CERTIFICATION_TRANSFER_UNPROVEN")
            if certification["subject_sha"] != candidate["sha"]:
                raise TransactionError("CERTIFICATION_TRANSFER_UNPROVEN")
            if candidate["sha"] != final["sha"]:
                transfer = port.verify_tree_transfer(intent, candidate, certification, final)
                if (
                    transfer.get("verified") is not True
                    or transfer.get("candidate_sha") != candidate["sha"]
                    or transfer.get("final_release_sha") != final["sha"]
                    or transfer.get("candidate_tree") != intent.anchor_tree
                    or transfer.get("final_release_tree") != intent.anchor_tree
                    or transfer.get("anchor_tree") != intent.anchor_tree
                    or not any(
                        isinstance(transfer.get(name), str) and bool(transfer.get(name))
                        for name in ("evidence_ref", "evidence_digest", "digest")
                    )
                ):
                    raise TransactionError("CERTIFICATION_TRANSFER_UNPROVEN")
                transfer_evidence = dict(transfer)
                # The transfer proof is part of the certification identity
                # passed to both durable release events.  Keeping this copy
                # on the certification prevents an in-memory verification
                # from being mistaken for durable evidence.
                certification = dict(certification)
                certification["transfer_evidence"] = {
                    **dict(transfer_evidence),
                    "certified_tree": certification["subject_tree"],
                }
                transfer_evidence = dict(certification["transfer_evidence"])
                evidence["certification"] = dict(certification)
                evidence["certification_transfer"] = dict(transfer_evidence)
                phase = "transfer_verified"

        evidence["final_certification"] = dict(certification)

        public_tag = f"v{intent.final_version}"
        tag_ref = f"refs/tags/{public_tag}"
        try:
            tag_protection = port.verify_public_tag_ruleset(intent, tag_ref)
            if not isinstance(tag_protection, ProtectionEvidence):
                raise GitIdentityError("public tag ruleset proof has the wrong type")
            tag_protection.require_public_tag(tag_ref)
        except Exception as exc:
            raise TransactionError("PUBLIC_TAG_RULESET_UNAVAILABLE") from exc
        evidence["public_tag_ruleset"] = asdict(tag_protection)
        phase = "tag_ruleset_verified"

        prepared = port.append_release_intent(intent, candidate, certification, final)
        if (
            prepared.get("public_tag") != public_tag
            or prepared.get("release_sha") != final["sha"]
            or prepared.get("release_tree") != final["tree"]
            or not prepared.get("event_id")
        ):
            raise TransactionError("RELEASE_INTENT_MISMATCH")
        if transfer_evidence is not None and prepared.get(
            "certification_transfer_evidence"
        ) != transfer_evidence:
            raise TransactionError("RELEASE_INTENT_TRANSFER_EVIDENCE_UNPROVEN")
        evidence["intent"] = prepared
        phase = "release_intent_durable"
        tag_attempted = True
        tag = port.create_protected_tag(intent, prepared, final, tag_protection)
        tag_created = True
        if (
            tag.get("name") != public_tag
            or tag.get("commit") != final["sha"]
            or tag.get("tree") != final["tree"]
            or tag.get("protected") is not True
        ):
            raise TransactionError("PUBLIC_TAG_IDENTITY_MISMATCH")
        evidence["tag"] = tag
        phase = "public_tag_created"
        released = port.append_released(intent, candidate, certification, final, tag)
        released_appended = True
        port.verify_released(intent, released, certification, final)
        if released.get("public_tag") != public_tag or not released.get("event_id"):
            raise TransactionError("RELEASED_EVENT_MISMATCH")
        if transfer_evidence is not None and released.get(
            "certification_transfer_evidence"
        ) != transfer_evidence:
            raise TransactionError("RELEASED_EVENT_TRANSFER_EVIDENCE_UNPROVEN")
        evidence["released"] = released
        phase = "released_event_durable"
        publication = port.publish_github_release(intent, released)
        if not publication.get("id") or publication.get("tag") != public_tag:
            raise TransactionError("PUBLICATION_IDENTITY_MISMATCH")
        evidence["publication"] = publication
        phase = "github_release_published"
        publication_evidence = port.persist_publication_evidence(
            intent, released, publication
        )
        if (
            publication_evidence.get("event_id") != released["event_id"]
            or publication_evidence.get("public_tag") != public_tag
            or publication_evidence.get("github_release_id") != publication["id"]
        ):
            raise TransactionError("PUBLICATION_EVIDENCE_MISMATCH")
        evidence["publication_evidence"] = publication_evidence
        phase = "publication_evidence_durable"
        port.verify_terminal(intent, released, publication, publication_evidence)
        phase = "terminal_verified"
        try:
            _release_static_mutation(port, mutation_lease)
        except TransactionError:
            # A release failure can follow a complete publication.  Keep the
            # main freeze and leave reconciliation to the caller.
            mutation_lease = None
            raise
        mutation_lease = None
        if main_token is not None:
            port.unfreeze_main(main_token)
        return TransactionResult("COMPLETE", "unfrozen", evidence=evidence)
    except TransactionError as exc:
        _release_static_mutation_best_effort(port, mutation_lease)
        if exc.reason_code.endswith("_MISMATCH") or "INVALID" in exc.reason_code:
            return TransactionResult(
                "INVALID", phase, exc.reason_code, evidence,
                frozen=main_token is not None,
            )
        if released_appended:
            return TransactionResult(
                "publication_reconciliation_pending", phase, exc.reason_code,
                evidence, frozen=main_token is not None,
            )
        if tag_attempted or tag_created:
            return TransactionResult(
                "tag_reconciliation_pending", phase, exc.reason_code,
                evidence, frozen=main_token is not None,
            )
        return TransactionResult(
            "BLOCKED", phase, exc.reason_code, evidence,
            frozen=main_token is not None,
        )
    except Exception as exc:
        _release_static_mutation_best_effort(port, mutation_lease)
        evidence["error_type"] = type(exc).__name__
        if released_appended:
            status = "publication_reconciliation_pending"
        elif tag_attempted:
            status = "tag_reconciliation_pending"
        else:
            status = "BLOCKED"
        return TransactionResult(
            status, phase, "RELEASE_OUTCOME_UNCERTAIN", evidence,
            frozen=main_token is not None,
        )
