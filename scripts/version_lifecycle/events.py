"""Canonical event stream validation for the version lifecycle authority.

The event stream is deliberately small and boring.  Events are ordinary JSON
objects encoded with the lifecycle canonical JSON codec and separated by one
LF.  This module does not perform Git writes; :mod:`writer` owns the CAS
append and remote recovery protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import json
import re
from typing import Any

from .certification import is_safe_public_value
from .codec import canonical_json, sha256_hex
from .git_refs import GitIdentityError, require_candidate_ref
from .versions import maintenance_line, parse_package_version, parse_public_tag


SCHEMA_VERSION = 1
EVENT_ID_PREFIX = "EVT-"
MAX_EVENT_SEQUENCE = 999_999_999_999
EVENT_ID_RE = re.compile(r"^EVT-(\d{12})$")
UTC_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
BRANCH_REF_RE = re.compile(r"^refs/heads/[A-Za-z0-9._/-]+$")
CANDIDATE_REF_RE = re.compile(r"^refs/heads/candidates/[A-Za-z0-9._/-]+$")
ANCHOR_REF_RE = re.compile(r"^refs/tags/iterations/([^/]+)$")

EVENT_TYPES = (
    "development_reservation_prepared",
    "development_reservation_opened",
    "development_reservation_aborted",
    "development_reservation_consumed",
    "maintenance_line_opened",
    "candidate_opened",
    "candidate_withdrawn",
    "release_intent_prepared",
    "release_intent_aborted",
    "released",
)

# Identities fixed by a durable intent before the irreversible public-tag
# boundary. The later released event must reproduce every value exactly.
RELEASE_INTENT_BINDING_FIELDS = (
    "candidate_ref",
    "candidate_sha",
    "candidate_tree",
    "anchor_ref",
    "anchor_sha",
    "anchor_tree",
    "final_version",
    "release_line",
    "certification_binding",
    "certification_subject_sha",
    "certification_subject_tree",
    "certification_policy_revision",
    "certification_harness_revision",
    "certification_environment",
    "certification_evidence_refs",
    "certification_transfer_evidence",
    "final_release_sha",
    "final_release_tree",
    "public_tag",
)

# All lifecycle events participate in the serialized allocation stream.  The
# fields are therefore required even for transitions which do not allocate a
# version: the transaction is the durable idempotency key and the two heads
# are the exclusion boundary captured by the event.
COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "event_type",
        "timestamp_utc",
        "transaction_id",
        "static_iteration_snapshot",
        "expected_event_head",
    }
)

TYPE_REQUIRED_FIELDS = {
    "development_reservation_prepared": frozenset(
        {
            "owner_line",
            "final_version",
            "intended_dev_version",
            "expected_line_head",
            "reservation_id",
        }
    ),
    "development_reservation_opened": frozenset(
        {
            "owner_line",
            "final_version",
            "intended_dev_version",
            "actual_dev_head",
            "reservation_id",
        }
    ),
    "development_reservation_aborted": frozenset(
        {
            "owner_line",
            "final_version",
            "intended_dev_version",
            "reservation_id",
            "abort_reason",
            "non_entry_evidence",
        }
    ),
    "development_reservation_consumed": frozenset(
        {
            "owner_line",
            "final_version",
            "intended_dev_version",
            "reservation_id",
            "closure_anchor",
            "terminal_disposition",
        }
    ),
    "maintenance_line_opened": frozenset(
        {
            "release_line",
            "approved_base_version",
            "branch_ref",
            "branch_head",
            "reservation_id",
            "dev_version",
        }
    ),
    "candidate_opened": frozenset(
        {
            "candidate_id",
            "candidate_ref",
            "candidate_sha",
            "candidate_tree",
            "final_version",
            "release_line",
            "anchor_ref",
            "anchor_sha",
            "anchor_tree",
        }
    ),
    "candidate_withdrawn": frozenset(
        {"candidate_id", "candidate_ref", "candidate_sha", "withdrawal_evidence"}
    ),
    "release_intent_prepared": frozenset(
        {
            "intent_id",
            "candidate_id",
            "candidate_ref",
            "candidate_sha",
            "candidate_tree",
            "anchor_ref",
            "anchor_sha",
            "anchor_tree",
            "certification_binding",
            "certification_subject_sha",
            "certification_subject_tree",
            "certification_policy_revision",
            "certification_harness_revision",
            "certification_environment",
            "certification_evidence_refs",
            "final_release_sha",
            "final_release_tree",
            "final_version",
            "release_line",
            "public_tag",
        }
    ),
    "release_intent_aborted": frozenset(
        {"intent_id", "candidate_id", "public_tag", "no_public_tag_evidence"}
    ),
    "released": frozenset(
        {
            "closure_timestamp_utc",
            "final_version",
            "release_line",
            "anchor_ref",
            "anchor_sha",
            "anchor_tree",
            "candidate_ref",
            "candidate_sha",
            "candidate_tree",
            "final_release_sha",
            "final_release_tree",
            "certification_binding",
            "certification_subject_sha",
            "certification_subject_tree",
            "certification_policy_revision",
            "certification_harness_revision",
            "certification_environment",
            "certification_evidence_refs",
            "public_tag",
            "evidence_refs",
            "main_at_event_sha",
            "main_at_event_version",
        }
    ),
}

TYPE_FORBIDDEN_FIELDS = {
    "development_reservation_prepared": frozenset(
        {
            "actual_dev_head", "abort_reason", "non_entry_evidence",
            "closure_anchor", "terminal_disposition", "branch_ref", "branch_head",
            "candidate_id", "candidate_ref", "candidate_sha", "candidate_tree",
            "intent_id", "public_tag", "main_at_event_sha", "main_at_event_version",
        }
    ),
    "development_reservation_opened": frozenset(
        {
            "expected_line_head", "abort_reason", "non_entry_evidence",
            "closure_anchor", "terminal_disposition", "branch_ref", "branch_head",
            "candidate_id", "candidate_ref", "candidate_sha", "candidate_tree",
            "intent_id", "public_tag", "main_at_event_sha", "main_at_event_version",
        }
    ),
    "development_reservation_aborted": frozenset(
        {
            "expected_line_head", "actual_dev_head", "closure_anchor",
            "terminal_disposition", "branch_ref", "branch_head", "candidate_id",
            "candidate_ref", "candidate_sha", "candidate_tree", "intent_id",
            "public_tag", "main_at_event_sha", "main_at_event_version",
        }
    ),
    "development_reservation_consumed": frozenset(
        {
            "expected_line_head", "actual_dev_head", "abort_reason",
            "non_entry_evidence", "branch_ref", "branch_head", "candidate_id",
            "candidate_ref", "candidate_sha", "candidate_tree", "intent_id",
            "public_tag", "main_at_event_sha", "main_at_event_version",
        }
    ),
    "maintenance_line_opened": frozenset(
        {
            "expected_line_head", "actual_dev_head", "abort_reason",
            "non_entry_evidence", "closure_anchor", "terminal_disposition",
            "candidate_id", "candidate_ref", "candidate_sha", "candidate_tree",
            "intent_id", "public_tag", "previous_main_sha", "previous_main_version",
        }
    ),
    "candidate_opened": frozenset(
        {
            "owner_line", "reservation_id", "expected_line_head", "actual_dev_head",
            "abort_reason", "non_entry_evidence", "closure_anchor",
            "terminal_disposition", "intent_id", "public_tag", "previous_main_sha",
            "previous_main_version", "main_at_event_sha", "main_at_event_version",
        }
    ),
    "candidate_withdrawn": frozenset(
        {
            "owner_line", "reservation_id", "expected_line_head", "actual_dev_head",
            "abort_reason", "non_entry_evidence", "closure_anchor",
            "terminal_disposition", "intent_id", "public_tag", "previous_main_sha",
            "previous_main_version", "main_at_event_sha", "main_at_event_version",
        }
    ),
    "release_intent_prepared": frozenset(
        {
            "owner_line", "reservation_id", "expected_line_head", "actual_dev_head",
            "abort_reason", "non_entry_evidence", "closure_anchor",
            "terminal_disposition", "withdrawal_evidence", "previous_main_sha",
            "previous_main_version", "main_at_event_sha", "main_at_event_version",
        }
    ),
    "release_intent_aborted": frozenset(
        {
            "owner_line", "reservation_id", "expected_line_head", "actual_dev_head",
            "abort_reason", "non_entry_evidence", "closure_anchor",
            "terminal_disposition", "withdrawal_evidence", "previous_main_sha",
            "main_at_event_sha", "main_at_event_version", "release_commit",
            "release_tree", "final_release_sha", "final_release_tree",
        }
    ),
    "released": frozenset(
        {
            "owner_line", "reservation_id", "expected_line_head", "actual_dev_head",
            "abort_reason", "non_entry_evidence", "closure_anchor",
            "terminal_disposition", "withdrawal_evidence", "intent_id",
            "release_commit", "release_tree", "public_tag_commit", "public_tag_tree",
        }
    ),
}

# These are the fields currently defined by the schema.  Rejecting unknown
# keys is intentional: a silently ignored key would not be covered by the
# canonical event identity or by transition replay.
SCHEMA_FIELDS = (
    frozenset().union(COMMON_FIELDS, *TYPE_REQUIRED_FIELDS.values())
    | frozenset(
        {
            "candidate_id",
            "previous_main_sha",
            "previous_main_version",
            "closed_final_version",
            "certification_transfer_evidence",
        }
    )
)


class EventError(ValueError):
    """Base class for malformed or invalid event streams."""

    reason_code = "INVALID_EVENT"


class EventSchemaError(EventError):
    reason_code = "EVENT_SCHEMA_INVALID"


class EventCanonicalEncodingError(EventError):
    reason_code = "EVENT_ENCODING_NONCANONICAL"


class EventTransitionError(EventError):
    reason_code = "EVENT_TRANSITION_INVALID"


class EventIdExhausted(EventError):
    reason_code = "EVENT_ID_EXHAUSTED"


class UnsupportedCertificationBinding(EventError):
    reason_code = "UNSUPPORTED_CERTIFICATION_BINDING"


# Event values are durable public authority.  Keep this lexical boundary
# deliberately conservative: references and digests are useful in the ledger,
# while workstation paths, local services, credentials and secret-like values
# are not durable identities.  This is a public-value gate, not a substitute
# for review of the evidence itself.
def _reject_unsafe_public_string(value: str, *, field: str = "event value") -> None:
    if not is_safe_public_value(value, key=field):
        raise EventSchemaError(f"{field} contains a private or secret-like value")


def _require_no_public_tag_evidence(event: Mapping[str, Any]) -> None:
    """Validate the structured R-034 no-tag proof carried by an abort.

    The proof records the exact proposed tag, a timestamped absence result and
    the digest of the exclusion-boundary observation.  The writer separately
    rechecks its live exclusion callback immediately before the append.
    """

    proof = event.get("no_public_tag_evidence")
    if not isinstance(proof, Mapping):
        raise EventSchemaError("no_public_tag_evidence must be a structured object")
    required = {
        "public_tag",
        "tag_absent",
        "checked_at_utc",
        "exclusion_verified",
        "exclusion_snapshot",
    }
    missing = sorted(required - proof.keys())
    if missing:
        raise EventSchemaError(
            "no_public_tag_evidence missing required fields: " + ", ".join(missing)
        )
    if set(proof) != required:
        extra = sorted(set(proof) - required)
        raise EventSchemaError(
            "no_public_tag_evidence has undeclared fields: " + ", ".join(extra)
        )
    if proof["public_tag"] != event.get("public_tag"):
        raise EventSchemaError("no-public-tag proof must bind the event public_tag")
    if proof["tag_absent"] is not True:
        raise EventSchemaError("no-public-tag proof must state tag_absent=true")
    if proof["exclusion_verified"] is not True:
        raise EventSchemaError("no-public-tag proof must be checked under exclusion")
    if not isinstance(proof["checked_at_utc"], str):
        raise EventSchemaError("no-public-tag proof checked_at_utc must be a timestamp")
    parse_timestamp(proof["checked_at_utc"])
    if not isinstance(proof["exclusion_snapshot"], str) or SHA256_RE.fullmatch(
        proof["exclusion_snapshot"]
    ) is None:
        raise EventSchemaError("no-public-tag proof exclusion_snapshot must be SHA-256 hex")
def _canonical_bytes(value: Any) -> bytes:
    encoded = canonical_json(value)
    if isinstance(encoded, str):
        encoded = encoded.encode("utf-8")
    if not isinstance(encoded, bytes):
        raise TypeError("canonical_json must return str or bytes")
    return encoded


def _strict_json_loads(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EventCanonicalEncodingError("event is not valid UTF-8") from exc

    def duplicate_key(object_pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in object_pairs:
            if key in result:
                raise EventCanonicalEncodingError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=duplicate_key,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EventCanonicalEncodingError(f"non-finite JSON value: {value}")
            ),
        )
    except EventError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise EventCanonicalEncodingError("malformed JSON event") from exc


def _reject_floats(value: Any) -> None:
    if isinstance(value, float):
        raise EventSchemaError("floating-point values are not permitted in events")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise EventSchemaError("event object keys must be strings")
            _reject_floats(child)
    elif isinstance(value, list):
        for child in value:
            _reject_floats(child)


def _reject_unsafe_values(value: Any, *, key: str | None = None) -> None:
    """Enforce the event wire subset beyond what ``json`` accepts."""

    if value is None:
        raise EventSchemaError("null is not permitted in canonical events")
    if isinstance(value, str):
        _reject_unsafe_public_string(value, field=key or "event value")
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise EventSchemaError("event object keys must be strings")
            _reject_unsafe_public_string(child_key, field="event field name")
            _reject_unsafe_values(child, key=child_key)
    elif isinstance(value, list):
        if key not in {"certification_evidence_refs", "evidence_refs"}:
            raise EventSchemaError(f"undeclared event array: {key or '<root>'}")
        for child in value:
            _reject_unsafe_values(child, key=key)


def parse_timestamp(timestamp: Any) -> datetime:
    if not isinstance(timestamp, str) or UTC_TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise EventSchemaError("timestamp_utc must be YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise EventSchemaError("timestamp_utc is not a real UTC calendar time") from exc


def validate_event_id(event_id: Any) -> int:
    if not isinstance(event_id, str):
        raise EventSchemaError("event_id must be a string")
    match = EVENT_ID_RE.fullmatch(event_id)
    if match is None:
        raise EventSchemaError("event_id must be EVT- followed by 12 digits")
    sequence = int(match.group(1), 10)
    if sequence == 0:
        raise EventSchemaError("event_id sequence starts at one")
    return sequence


def event_id(sequence: int) -> str:
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise TypeError("event sequence must be an integer")
    if sequence < 1 or sequence > MAX_EVENT_SEQUENCE:
        raise EventIdExhausted("the 12-digit event identifier space is exhausted")
    return f"{EVENT_ID_PREFIX}{sequence:012d}"


def next_event_id(events: Iterable[Mapping[str, Any]]) -> str:
    events = list(events)
    if not events:
        return event_id(1)
    prior = validate_event_id(events[-1].get("event_id"))
    if prior >= MAX_EVENT_SEQUENCE:
        raise EventIdExhausted("the 12-digit event identifier space is exhausted")
    return event_id(prior + 1)


def _require_text(event: Mapping[str, Any], name: str) -> None:
    value = event.get(name)
    if not isinstance(value, str) or not value:
        raise EventSchemaError(f"{name} must be a non-empty string")
    _reject_unsafe_public_string(value, field=name)


def _require_digest(event: Mapping[str, Any], name: str) -> None:
    _require_text(event, name)
    if SHA256_RE.fullmatch(event[name]) is None:
        raise EventSchemaError(f"{name} must be lowercase SHA-256 hex")


def _require_git_object(event: Mapping[str, Any], name: str) -> None:
    _require_text(event, name)
    if GIT_OBJECT_RE.fullmatch(event[name]) is None:
        raise EventSchemaError(f"{name} must be a full hexadecimal Git object ID")


def _require_non_entry_evidence(event: Mapping[str, Any]) -> None:
    """Require a replayable, exclusion-bound proof of definite DEV non-entry."""

    proof = event.get("non_entry_evidence")
    if not isinstance(proof, Mapping):
        raise EventSchemaError("non_entry_evidence must be a structured proof")
    common = {
        "verified", "reservation_id", "owner_line", "final_version",
        "intended_dev_version", "line_ref", "expected_line_head",
        "line_state", "dev_not_entered", "exclusion_verified",
        "observed_at_utc", "evidence_ref", "evidence_digest",
    }
    state = proof.get("line_state")
    required = common | ({"observed_line_head"} if state == "unchanged" else set())
    if set(proof) != required:
        raise EventSchemaError("non_entry_evidence has missing or undeclared fields")
    if (
        proof["verified"] is not True
        or proof["dev_not_entered"] is not True
        or proof["exclusion_verified"] is not True
    ):
        raise EventSchemaError("non_entry_evidence is not a verified non-entry proof")
    for field in ("reservation_id", "owner_line", "final_version", "intended_dev_version"):
        if proof[field] != event[field]:
            raise EventSchemaError(f"non_entry_evidence {field} does not match reservation")
    expected_ref = (
        "refs/heads/vmm" if event["owner_line"] == "principal"
        else f"refs/heads/{event['owner_line']}"
    )
    if proof["line_ref"] != expected_ref:
        raise EventSchemaError("non_entry_evidence line_ref does not match owner line")
    _require_git_object(proof, "expected_line_head")
    if state == "unchanged":
        _require_git_object(proof, "observed_line_head")
        if proof["observed_line_head"] != proof["expected_line_head"]:
            raise EventSchemaError("non_entry_evidence line head changed")
    elif state == "absent":
        if event["owner_line"] == "principal":
            raise EventSchemaError("principal line cannot be absent for reservation abort")
    else:
        raise EventSchemaError("non_entry_evidence line_state is unsupported")
    if parse_timestamp(proof["observed_at_utc"]) > parse_timestamp(event["timestamp_utc"]):
        raise EventSchemaError("non_entry_evidence observation follows abort event")
    _require_text(proof, "evidence_ref")
    _require_digest(proof, "evidence_digest")


def _require_certification_transfer_evidence(event: Mapping[str, Any]) -> None:
    """Validate the durable proof for a tree-bound commit transfer.

    The proof is kept as one structured public value so replay can compare it
    with the certification and with the release identities.  It deliberately
    accepts one evidence reference or digest, never a workstation locator.
    """

    proof = event.get("certification_transfer_evidence")
    if not isinstance(proof, Mapping):
        raise EventSchemaError("certification_transfer_evidence must be an object")
    allowed = {
        "verified",
        "candidate_sha",
        "final_release_sha",
        "certified_tree",
        "candidate_tree",
        "final_release_tree",
        "anchor_tree",
        "evidence_ref",
        "evidence_digest",
        "digest",
    }
    if set(proof) - allowed:
        raise EventSchemaError("certification_transfer_evidence has unknown fields")
    if proof.get("verified") is not True:
        raise EventSchemaError("certification_transfer_evidence must be verified")
    for field in (
        "candidate_sha",
        "final_release_sha",
        "candidate_tree",
        "final_release_tree",
        "anchor_tree",
    ):
        _require_git_object(proof, field)
    evidence_fields = [
        field
        for field in ("evidence_ref", "evidence_digest", "digest")
        if field in proof
    ]
    if len(evidence_fields) != 1:
        raise EventSchemaError(
            "certification_transfer_evidence requires one evidence_ref or evidence_digest"
        )
    if evidence_fields[0] == "evidence_ref":
        _require_text(proof, "evidence_ref")
    elif evidence_fields[0] == "evidence_digest":
        _require_digest(proof, "evidence_digest")
    else:
        _require_text(proof, "digest")
    for field in (
        "candidate_sha",
        "final_release_sha",
        "candidate_tree",
        "final_release_tree",
        "anchor_tree",
    ):
        event_field = {
            "candidate_sha": "candidate_sha",
            "final_release_sha": "final_release_sha",
            "candidate_tree": "candidate_tree",
            "final_release_tree": "final_release_tree",
            "anchor_tree": "anchor_tree",
        }[field]
        if proof[field] != event.get(event_field):
            raise EventSchemaError(
                f"certification_transfer_evidence {field} does not match event"
            )
    if "certified_tree" in proof:
        _require_git_object(proof, "certified_tree")
        if proof["certified_tree"] != event.get("certification_subject_tree"):
            raise EventSchemaError(
                "certification_transfer_evidence certified_tree does not match event"
            )


def _require_final_version(event: Mapping[str, Any], name: str) -> None:
    _require_text(event, name)
    try:
        version = parse_package_version(event[name])
    except (TypeError, ValueError):
        version = None
    if version is None or not version.is_final:
        raise EventSchemaError(f"{name} must be canonical final SemVer")


def _require_public_tag(event: Mapping[str, Any], name: str) -> None:
    _require_text(event, name)
    try:
        parse_public_tag(event[name])
    except (TypeError, ValueError):
        raise EventSchemaError(f"{name} must be canonical vX.Y.Z")


def _require_dev_version(event: Mapping[str, Any], name: str) -> None:
    _require_text(event, name)
    try:
        version = parse_package_version(event[name])
    except (TypeError, ValueError):
        version = None
    if version is None or not version.is_dev:
        raise EventSchemaError(f"{name} must be canonical X.Y.Z-DEV")


def _require_ref(
    event: Mapping[str, Any], name: str, pattern: re.Pattern[str], description: str
) -> None:
    _require_text(event, name)
    if pattern.fullmatch(event[name]) is None:
        raise EventSchemaError(f"{name} must be a canonical {description}")


def _is_maintenance_line(value: Any) -> bool:
    try:
        maintenance_line(value)
    except (TypeError, ValueError):
        return False
    return True


def _require_candidate_ref(event: Mapping[str, Any], name: str = "candidate_ref") -> None:
    """Validate a durable candidate ref with the shared Git ref policy."""

    _require_text(event, name)
    try:
        require_candidate_ref(event[name])
    except GitIdentityError as exc:
        raise EventSchemaError(f"{name} must be a canonical candidate ref") from exc


def _require_anchor_ref(event: Mapping[str, Any], name: str = "anchor_ref") -> None:
    """Require an iteration anchor ref whose version uses the shared parser."""

    _require_text(event, name)
    value = event[name]
    match = ANCHOR_REF_RE.fullmatch(value)
    if match is None:
        raise EventSchemaError(f"{name} must be a canonical iteration anchor ref")
    try:
        version = parse_package_version(match.group(1))
    except (TypeError, ValueError) as exc:
        raise EventSchemaError(
            f"{name} must be a canonical iteration anchor ref"
        ) from exc
    if not version.is_final:
        raise EventSchemaError(f"{name} must be a canonical iteration anchor ref")


def _require_git_identity_or_anchor_ref(event: Mapping[str, Any], name: str) -> None:
    """Accept a full commit identity or the canonical iteration anchor ref."""

    _require_text(event, name)
    value = event[name]
    if GIT_OBJECT_RE.fullmatch(value) is None:
        try:
            _require_anchor_ref(event, name)
        except EventSchemaError as exc:
            raise EventSchemaError(
                f"{name} must be a full Git object ID or canonical iteration anchor ref"
            ) from exc


def _validate_type_fields(event: Mapping[str, Any]) -> None:
    event_type = event["event_type"]
    if event_type != "development_reservation_consumed" and "closed_final_version" in event:
        raise EventSchemaError("closed_final_version is only valid for reservation consumption")
    required = TYPE_REQUIRED_FIELDS[event_type]
    missing = sorted(required - event.keys())
    if missing:
        raise EventSchemaError(f"{event_type} missing required fields: {', '.join(missing)}")
    forbidden = sorted(TYPE_FORBIDDEN_FIELDS.get(event_type, frozenset()).intersection(event))
    if forbidden:
        raise EventSchemaError(
            f"{event_type} has fields from an unrelated lifecycle state: {', '.join(forbidden)}"
        )
    if event_type not in {"release_intent_prepared", "released"} and (
        "certification_transfer_evidence" in event
    ):
        raise EventSchemaError(
            "certification_transfer_evidence is only valid on release events"
        )

    if event_type in {
        "development_reservation_prepared",
        "development_reservation_opened",
        "development_reservation_aborted",
        "development_reservation_consumed",
    }:
        for field in (
            "owner_line",
            "final_version",
            "intended_dev_version",
            "reservation_id",
        ):
            _require_text(event, field)
        _require_final_version(event, "final_version")
        _require_dev_version(event, "intended_dev_version")
        if event["intended_dev_version"][0:-4] != event["final_version"]:
            raise EventSchemaError("intended_dev_version must identify final_version")
        if event["owner_line"] != "principal" and not _is_maintenance_line(
            event["owner_line"]
        ):
            raise EventSchemaError("owner_line must be principal or maintenance/X.Y")
        if event["owner_line"] != "principal" and not _line_matches_version(
            event["owner_line"], event["final_version"]
        ):
            raise EventSchemaError(
                "maintenance reservation owner_line must match final_version major/minor"
            )
        if event_type == "development_reservation_prepared":
            _require_git_object(event, "expected_line_head")
        elif event_type == "development_reservation_opened":
            _require_git_object(event, "actual_dev_head")
        elif event_type == "development_reservation_aborted":
            _require_text(event, "abort_reason")
            _require_non_entry_evidence(event)
        else:
            _require_git_identity_or_anchor_ref(event, "closure_anchor")
            closure_anchor = event["closure_anchor"]
            disposition = event["terminal_disposition"]
            if disposition == "closed":
                if "closed_final_version" in event:
                    raise EventSchemaError("closed reservation cannot declare a different final")
                closed_final = event["final_version"]
            elif disposition == "CONSUMED_UNUSED_DEV_RESERVATION":
                _require_final_version(event, "closed_final_version")
                closed_final = event["closed_final_version"]
                if closed_final == event["final_version"]:
                    raise EventSchemaError("unused reservation disposition requires a different final")
                if not _line_matches_version(event["owner_line"], closed_final):
                    raise EventSchemaError("closed final does not belong to the owner line")
            else:
                raise EventSchemaError("invalid reservation terminal_disposition")
            if (
                ANCHOR_REF_RE.fullmatch(closure_anchor)
                and closure_anchor != f"refs/tags/iterations/{closed_final}"
            ):
                raise EventSchemaError(
                    "closure_anchor ref must identify the closed final version"
                )

    elif event_type == "maintenance_line_opened":
        for field in (
            "release_line",
            "approved_base_version",
            "branch_head",
            "reservation_id",
            "dev_version",
        ):
            _require_text(event, field)
        _require_ref(event, "branch_ref", BRANCH_REF_RE, "branch ref")
        if not _is_maintenance_line(event["release_line"]):
            raise EventSchemaError("release_line must be maintenance/X.Y")
        _require_final_version(event, "approved_base_version")
        if not _line_matches_version(
            event["release_line"], event["approved_base_version"]
        ):
            raise EventSchemaError(
                "approved_base_version must match release_line maintenance/X.Y"
            )
        expected_branch_ref = f"refs/heads/{event['release_line']}"
        if event["branch_ref"] != expected_branch_ref:
            raise EventSchemaError(
                "branch_ref must match release_line maintenance/X.Y"
            )
        _require_dev_version(event, "dev_version")
        _require_git_object(event, "branch_head")

    elif event_type == "candidate_opened":
        for field in (
            "candidate_id",
            "candidate_sha",
            "candidate_tree",
            "final_version",
            "release_line",
            "anchor_sha",
            "anchor_tree",
        ):
            _require_text(event, field)
        _require_candidate_ref(event)
        _require_anchor_ref(event)
        _require_final_version(event, "final_version")
        if event["release_line"] != "principal" and not _is_maintenance_line(
            event["release_line"]
        ):
            raise EventSchemaError("release_line must be principal or maintenance/X.Y")
        if not _line_matches_version(event["release_line"], event["final_version"]):
            raise EventSchemaError("candidate release_line does not contain final_version")
        if event["anchor_ref"] != f"refs/tags/iterations/{event['final_version']}":
            raise EventSchemaError("candidate anchor_ref does not identify final_version")
        for field in ("candidate_sha", "candidate_tree", "anchor_sha", "anchor_tree"):
            _require_git_object(event, field)

    elif event_type == "candidate_withdrawn":
        for field in ("candidate_id", "candidate_sha", "withdrawal_evidence"):
            _require_text(event, field)
        _require_candidate_ref(event)
        _require_git_object(event, "candidate_sha")

    elif event_type == "release_intent_prepared":
        for field in (
            "intent_id",
            "candidate_id",
            "candidate_sha",
            "candidate_tree",
            "anchor_sha",
            "anchor_tree",
            "certification_binding",
            "certification_subject_sha",
            "certification_subject_tree",
            "certification_policy_revision",
            "certification_harness_revision",
            "certification_environment",
            "final_release_sha",
            "final_release_tree",
            "final_version",
            "release_line",
            "public_tag",
        ):
            _require_text(event, field)
        _require_candidate_ref(event)
        _require_anchor_ref(event)
        _require_final_version(event, "final_version")
        _require_public_tag(event, "public_tag")
        if event["certification_binding"] not in {"tree-bound", "commit-bound"}:
            raise UnsupportedCertificationBinding(
                "certification binding must be tree-bound or commit-bound"
            )
        if event["public_tag"][1:] != event["final_version"]:
            raise EventSchemaError("public_tag must identify final_version")
        if event["release_line"] != "principal" and not _is_maintenance_line(
            event["release_line"]
        ):
            raise EventSchemaError("release_line must be principal or maintenance/X.Y")
        if not _line_matches_version(event["release_line"], event["final_version"]):
            raise EventSchemaError("release intent release_line does not contain final_version")
        if event["anchor_ref"] != f"refs/tags/iterations/{event['final_version']}":
            raise EventSchemaError("release intent anchor_ref does not identify final_version")
        for field in (
            "certification_binding",
            "certification_subject_sha",
            "certification_subject_tree",
            "certification_policy_revision",
            "certification_harness_revision",
            "certification_environment",
        ):
            _require_text(event, field)
        if not isinstance(event["certification_evidence_refs"], list) or not event[
            "certification_evidence_refs"
        ]:
            raise EventSchemaError("certification_evidence_refs must be a non-empty array")
        for item in event["certification_evidence_refs"]:
            _require_text({"value": item}, "value")
        for field in (
            "candidate_sha", "candidate_tree", "anchor_sha", "anchor_tree",
            "certification_subject_sha", "certification_subject_tree",
            "final_release_sha", "final_release_tree",
        ):
            _require_git_object(event, field)
        if len({
            event["candidate_tree"], event["anchor_tree"],
            event["certification_subject_tree"], event["final_release_tree"],
        }) != 1:
            raise EventSchemaError("release intent trees must equal the certified anchor tree")
        if (
            event["certification_binding"] == "commit-bound"
            and event["certification_subject_sha"] != event["final_release_sha"]
        ):
            raise EventSchemaError("commit-bound intent must certify the final release commit")
        if (
            event["certification_binding"] == "tree-bound"
            and event["certification_subject_sha"] != event["candidate_sha"]
        ):
            raise EventSchemaError("tree-bound intent must certify the candidate commit")
        transfer_present = "certification_transfer_evidence" in event
        if event["certification_binding"] == "tree-bound" and (
            event["candidate_sha"] != event["final_release_sha"]
        ):
            if not transfer_present:
                raise EventSchemaError(
                    "tree-bound release intent requires certification transfer evidence"
                )
        if transfer_present:
            if event["certification_binding"] != "tree-bound":
                raise EventSchemaError(
                    "commit-bound release intent cannot carry transfer evidence"
                )
            _require_certification_transfer_evidence(event)

    elif event_type == "release_intent_aborted":
        for field in ("intent_id", "candidate_id", "public_tag"):
            _require_text(event, field)
        _require_public_tag(event, "public_tag")
        _require_no_public_tag_evidence(event)

    elif event_type == "released":
        for field in (
            "closure_timestamp_utc",
            "final_version",
            "release_line",
            "anchor_sha",
            "anchor_tree",
            "candidate_sha",
            "candidate_tree",
            "final_release_sha",
            "final_release_tree",
            "certification_binding",
            "certification_subject_sha",
            "certification_subject_tree",
            "certification_policy_revision",
            "certification_harness_revision",
            "certification_environment",
            "public_tag",
            "main_at_event_sha",
            "main_at_event_version",
        ):
            _require_text(event, field)
        _require_anchor_ref(event)
        _require_candidate_ref(event)
        parse_timestamp(event["closure_timestamp_utc"])
        _require_final_version(event, "final_version")
        _require_public_tag(event, "public_tag")
        if event["certification_binding"] not in {"tree-bound", "commit-bound"}:
            raise UnsupportedCertificationBinding(
                "certification binding must be tree-bound or commit-bound"
            )
        if event["public_tag"][1:] != event["final_version"]:
            raise EventSchemaError("public_tag must identify final_version")
        if event["release_line"] != "principal" and not _is_maintenance_line(
            event["release_line"]
        ):
            raise EventSchemaError("release_line must be principal or maintenance/X.Y")
        if not _line_matches_version(event["release_line"], event["final_version"]):
            raise EventSchemaError("released release_line does not contain final_version")
        if event["anchor_ref"] != f"refs/tags/iterations/{event['final_version']}":
            raise EventSchemaError("released anchor_ref does not identify final_version")
        for field in (
            "anchor_sha", "anchor_tree", "candidate_sha", "candidate_tree",
            "final_release_sha", "final_release_tree", "certification_subject_sha",
            "certification_subject_tree", "main_at_event_sha",
        ):
            _require_git_object(event, field)
        if len({
            event["anchor_tree"],
            event["candidate_tree"],
            event["final_release_tree"],
            event["certification_subject_tree"],
        }) != 1:
            raise EventSchemaError("released identity trees must match exactly")
        if event["release_line"] == "principal":
            _require_final_version(event, "main_at_event_version")
            if event["main_at_event_version"] != event["final_version"]:
                raise EventSchemaError("principal main_at_event_version must equal final_version")
        else:
            _require_final_version(event, "main_at_event_version")
        for field in ("certification_evidence_refs", "evidence_refs"):
            if not isinstance(event[field], list) or not event[field]:
                raise EventSchemaError(f"{field} must be a non-empty ordered array")
            for item in event[field]:
                _require_text({field: item}, field)
        if event["release_line"] == "principal":
            for field in ("previous_main_sha", "previous_main_version"):
                _require_text(event, field)
            _require_git_object(event, "previous_main_sha")
            _require_final_version(event, "previous_main_version")
        elif "previous_main_sha" in event or "previous_main_version" in event:
            raise EventSchemaError("maintenance released events cannot claim previous main")
        transfer_present = "certification_transfer_evidence" in event
        if event["certification_binding"] == "tree-bound" and (
            event["candidate_sha"] != event["final_release_sha"]
        ):
            if not transfer_present:
                raise EventSchemaError(
                    "tree-bound released event requires certification transfer evidence"
                )
        if transfer_present:
            if event["certification_binding"] != "tree-bound":
                raise EventSchemaError(
                    "commit-bound released event cannot carry transfer evidence"
                )
            _require_certification_transfer_evidence(event)


def validate_event(event: Mapping[str, Any] | bytes | bytearray | str) -> dict[str, Any]:
    """Validate one canonical event and return a detached dictionary.

    A bytes/string input is required to be the complete standalone canonical
    JSON object without a trailing LF.  Mapping input is canonicalized and
    checked through the same codec, so callers cannot bypass encoding rules by
    constructing a dictionary directly.
    """

    if isinstance(event, (bytes, bytearray)):
        raw = bytes(event)
        if raw.endswith(b"\n"):
            raise EventCanonicalEncodingError("standalone event must not have a trailing LF")
        parsed = _strict_json_loads(raw)
        if not isinstance(parsed, dict):
            raise EventSchemaError("event must be a JSON object")
        if _canonical_bytes(parsed) != raw:
            raise EventCanonicalEncodingError("event bytes are not canonical")
        value = parsed
    elif isinstance(event, str):
        return validate_event(event.encode("utf-8"))
    elif isinstance(event, Mapping):
        value = dict(event)
        _reject_floats(value)
        # Round trip through the codec.  This also rejects unsupported values.
        _canonical_bytes(value)
    else:
        raise EventSchemaError("event must be a mapping or canonical JSON bytes")

    _reject_floats(value)
    _reject_unsafe_values(value)
    unknown = sorted(set(value) - SCHEMA_FIELDS)
    if unknown:
        raise EventSchemaError(f"undeclared event fields: {', '.join(unknown)}")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise EventSchemaError("schema_version must equal 1")
    if not isinstance(value.get("event_type"), str) or value["event_type"] not in EVENT_TYPES:
        raise EventSchemaError("event_type is not in the canonical vocabulary")
    validate_event_id(value.get("event_id"))
    parse_timestamp(value.get("timestamp_utc"))
    for field in ("transaction_id", "static_iteration_snapshot", "expected_event_head"):
        _require_text(value, field)
    _require_digest(value, "static_iteration_snapshot")
    _require_git_object(value, "expected_event_head")
    _validate_type_fields(value)
    return value


def canonical_event_bytes(event: Mapping[str, Any] | bytes | bytearray | str) -> bytes:
    """Return the canonical standalone JSON bytes for *event*."""

    value = validate_event(event)
    return _canonical_bytes(value)


def event_digest(event: Mapping[str, Any] | bytes | bytearray | str) -> str:
    return sha256_hex(canonical_event_bytes(event))


def _identity(event: Mapping[str, Any]) -> str:
    return str(
        event.get("reservation_id")
        or event.get("candidate_id")
        or event.get("intent_id")
        or event.get("release_line")
        or event.get("owner_line")
        or event.get("event_id")
    )


def validate_transition(
    prior_events: Iterable[Mapping[str, Any]], event: Mapping[str, Any]
) -> None:
    """Validate the lifecycle transition represented by *event*.

    The replay is intentionally identity based.  A malformed predecessor,
    duplicate transaction or terminal-to-active reversal is rejected before a
    writer can create a descendant commit.
    """

    prior = [validate_event(item) for item in prior_events]
    for index, old in enumerate(prior, start=1):
        if validate_event_id(old["event_id"]) != index:
            raise EventTransitionError(
                "prior event IDs must be contiguous from EVT-000000000001"
            )
    current = validate_event(event)
    sequence = validate_event_id(current["event_id"])
    if sequence != len(prior) + 1:
        raise EventTransitionError("event IDs must be contiguous from EVT-000000000001")
    current_tx = current["transaction_id"]
    for old in prior:
        if old["transaction_id"] == current_tx:
            raise EventTransitionError("transaction_id already has a durable event")

    event_type = current["event_type"]
    if event_type == "development_reservation_prepared":
        key = _identity(current)
        if any(
            old["event_type"] == event_type and _identity(old) == key for old in prior
        ):
            raise EventTransitionError("reservation is already prepared")
        if _version_seen(prior, current.get("final_version")):
            raise EventTransitionError("final version is already globally occupied")
    elif event_type == "development_reservation_opened":
        _require_reservation_predecessor(prior, current, "prepared")
        _require_matching_reservation_identity(prior, current)
        if _has_reservation_state(prior, current, "opened"):
            raise EventTransitionError("reservation is already opened")
    elif event_type == "development_reservation_aborted":
        _require_reservation_predecessor(prior, current, "prepared")
        _require_matching_reservation_identity(prior, current)
        if _has_reservation_state(prior, current, "opened"):
            raise EventTransitionError("opened reservation cannot be aborted")
        prepared = next(
            old for old in prior
            if old["event_type"] == "development_reservation_prepared"
            and old["reservation_id"] == current["reservation_id"]
        )
        if (
            current["non_entry_evidence"]["expected_line_head"]
            != prepared["expected_line_head"]
        ):
            raise EventTransitionError(
                "non-entry proof does not bind the prepared line head"
            )
    elif event_type == "development_reservation_consumed":
        _require_reservation_predecessor(prior, current, "opened")
        _require_matching_reservation_identity(prior, current)
        if _has_reservation_state(prior, current, "consumed"):
            raise EventTransitionError("reservation is already terminal")
    elif event_type == "maintenance_line_opened":
        line = current["release_line"]
        if any(old.get("release_line") == line for old in prior if old["event_type"] == event_type):
            raise EventTransitionError("maintenance line is already open")
        _require_reservation_predecessor(prior, current, "opened")
        opened = next(
            old
            for old in prior
            if old["event_type"] == "development_reservation_opened"
            and old.get("reservation_id") == current.get("reservation_id")
        )
        if opened.get("owner_line") != current.get("release_line"):
            raise EventTransitionError("maintenance line does not own its reservation")
        if opened.get("intended_dev_version") != current.get("dev_version"):
            raise EventTransitionError("maintenance DEV identity does not match reservation")
    elif event_type == "candidate_opened":
        candidate = current["candidate_id"]
        if any(old.get("candidate_id") == candidate for old in prior):
            raise EventTransitionError("candidate identity already exists")
        if any(
            old["event_type"] == "candidate_opened"
            and old.get("final_version") == current.get("final_version")
            for old in prior
        ):
            raise EventTransitionError("final version already has a durable candidate")
        consumed = _find_consumed_reservation(prior, current)
        if consumed is None:
            raise EventTransitionError(
                "candidate requires the owning line's consumed reservation"
            )
        if consumed.get("closure_anchor") not in {
            current.get("anchor_ref"),
            current.get("anchor_sha"),
            current.get("anchor_tree"),
        }:
            raise EventTransitionError("candidate anchor does not match consumed closure anchor")
    elif event_type == "candidate_withdrawn":
        candidate = _find_candidate(prior, current["candidate_id"])
        if candidate is None:
            raise EventTransitionError("candidate withdrawal has no opened candidate")
        if _candidate_released(prior, current["candidate_id"]):
            raise EventTransitionError("released candidate cannot be withdrawn")
        if any(
            old["event_type"] == "release_intent_prepared"
            and old.get("candidate_id") == current["candidate_id"]
            and not _intent_is_terminal(prior, old)
            for old in prior
        ):
            raise EventTransitionError("active release intent must be aborted before withdrawal")
        _require_matching_fields(
            candidate,
            current,
            ("candidate_ref", "candidate_sha", "candidate_tree"),
        )
        if any(
            old.get("candidate_id") == current["candidate_id"]
            and old["event_type"] == "candidate_withdrawn"
            for old in prior
        ):
            raise EventTransitionError("candidate is already withdrawn")
    elif event_type == "release_intent_prepared":
        candidate = _find_candidate(prior, current["candidate_id"])
        if (
            candidate is None
            or _candidate_withdrawn(prior, current["candidate_id"])
            or _candidate_released(prior, current["candidate_id"])
        ):
            raise EventTransitionError("release intent requires an active candidate")
        _require_matching_fields(
            candidate,
            current,
            (
                "candidate_ref",
                "candidate_sha",
                "candidate_tree",
                "final_version",
                "release_line",
                "anchor_ref",
                "anchor_sha",
                "anchor_tree",
            ),
        )
        if any(old.get("intent_id") == current["intent_id"] for old in prior):
            raise EventTransitionError("release intent identity already exists")
        if any(
            old["event_type"] == "release_intent_prepared"
            and old.get("candidate_id") == current.get("candidate_id")
            and old.get("public_tag") == current.get("public_tag")
            and not _intent_is_terminal(prior, old)
            for old in prior
        ):
            raise EventTransitionError("candidate already has an active release intent")
    elif event_type == "release_intent_aborted":
        prepared = next(
            (
                old
                for old in prior
                if old["event_type"] == "release_intent_prepared"
                and old.get("intent_id") == current["intent_id"]
            ),
            None,
        )
        if prepared is None:
            raise EventTransitionError("intent abort has no prepared intent")
        _require_matching_fields(prepared, current, ("candidate_id", "public_tag"))
        if any(
            old["event_type"] == "released"
            and old.get("public_tag") == prepared.get("public_tag")
            for old in prior
        ):
            raise EventTransitionError("released intent cannot be aborted")
        if any(
            old["event_type"] == "release_intent_aborted"
            and old.get("intent_id") == current["intent_id"]
            for old in prior
        ):
            raise EventTransitionError("intent is already terminal")
    elif event_type == "released":
        matching_intents = [
            old
            for old in prior
            if old["event_type"] == "release_intent_prepared"
            and old.get("public_tag") == current.get("public_tag")
            and (
                current.get("candidate_id") is None
                or old.get("candidate_id") == current.get("candidate_id")
            )
        ]
        if not matching_intents:
            raise EventTransitionError("released event requires a matching prepared intent")
        intent = matching_intents[-1]
        if any(
            old["event_type"] == "release_intent_aborted"
            and old.get("intent_id") == intent.get("intent_id")
            for old in prior
        ):
            raise EventTransitionError("aborted release intent cannot be released")
        _require_matching_fields(
            intent,
            current,
            RELEASE_INTENT_BINDING_FIELDS,
        )
        if any(
            old["event_type"] == "released"
            and old.get("public_tag") == current.get("public_tag")
            for old in prior
        ):
            raise EventTransitionError("public release is already terminal")


def _reservation_key(event: Mapping[str, Any]) -> str:
    return str(event.get("reservation_id"))


def _has_reservation_state(
    prior: Iterable[Mapping[str, Any]], current: Mapping[str, Any], state: str
) -> bool:
    names = {
        "prepared": "development_reservation_prepared",
        "opened": "development_reservation_opened",
        "consumed": "development_reservation_consumed",
        "aborted": "development_reservation_aborted",
    }
    return any(
        old["event_type"] == names[state]
        and _reservation_key(old) == _reservation_key(current)
        for old in prior
    )


def _require_reservation_predecessor(
    prior: Iterable[Mapping[str, Any]], current: Mapping[str, Any], state: str
) -> None:
    if not _has_reservation_state(prior, current, state):
        raise EventTransitionError(
            f"reservation transition requires a prior {state} event"
        )
    if _has_reservation_state(prior, current, "aborted") or _has_reservation_state(
        prior, current, "consumed"
    ):
        raise EventTransitionError("reservation is already terminal")


def _require_matching_reservation_identity(
    prior: Iterable[Mapping[str, Any]], current: Mapping[str, Any]
) -> None:
    predecessor = next(
        (
            old
            for old in prior
            if old["event_type"] in {
                "development_reservation_prepared",
                "development_reservation_opened",
            }
            and _reservation_key(old) == _reservation_key(current)
        ),
        None,
    )
    if predecessor is None:
        raise EventTransitionError("reservation predecessor is not identifiable")
    _require_matching_fields(
        predecessor,
        current,
        ("owner_line", "final_version", "intended_dev_version"),
    )


def _require_matching_fields(
    predecessor: Mapping[str, Any],
    current: Mapping[str, Any],
    names: Iterable[str],
) -> None:
    for name in names:
        if predecessor.get(name) != current.get(name):
            raise EventTransitionError(f"lifecycle identity field changed: {name}")


def _find_candidate(
    prior: Iterable[Mapping[str, Any]], candidate_id: str
) -> Mapping[str, Any] | None:
    for old in prior:
        if old["event_type"] == "candidate_opened" and old.get("candidate_id") == candidate_id:
            return old
    return None


def _candidate_withdrawn(prior: Iterable[Mapping[str, Any]], candidate_id: str) -> bool:
    return any(
        old["event_type"] == "candidate_withdrawn" and old.get("candidate_id") == candidate_id
        for old in prior
    )


def _candidate_released(prior: Iterable[Mapping[str, Any]], candidate_id: str) -> bool:
    """Return whether a candidate already crossed the immutable release boundary."""

    prior = list(prior)
    candidate = _find_candidate(prior, candidate_id)
    return any(
        old["event_type"] == "released"
        and (
            old.get("candidate_id") == candidate_id
            or (
                candidate is not None
                and old.get("candidate_ref") == candidate.get("candidate_ref")
                and old.get("candidate_sha") == candidate.get("candidate_sha")
                and old.get("candidate_tree") == candidate.get("candidate_tree")
            )
        )
        for old in prior
    )


def _intent_is_terminal(
    prior: Iterable[Mapping[str, Any]], prepared: Mapping[str, Any]
) -> bool:
    """Return whether a prepared intent has an abort or matching release."""

    intent_id = prepared.get("intent_id")
    candidate_id = prepared.get("candidate_id")
    public_tag = prepared.get("public_tag")
    for old in prior:
        if old["event_type"] == "release_intent_aborted" and old.get("intent_id") == intent_id:
            return True
        if (
            old["event_type"] == "released"
            and old.get("public_tag") == public_tag
            and (
                old.get("candidate_id") is None
                or old.get("candidate_id") == candidate_id
            )
        ):
            return True
    return False


def _line_matches_version(line: str, version: str) -> bool:
    if line == "principal":
        return True
    try:
        major, minor = maintenance_line(line)
        parsed = parse_package_version(version)
    except (TypeError, ValueError):
        return False
    return parsed.is_final and (parsed.major, parsed.minor) == (major, minor)


def _find_consumed_reservation(
    prior: Iterable[Mapping[str, Any]], current: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    for old in prior:
        if (
            old["event_type"] == "development_reservation_consumed"
            and old.get("owner_line") == current.get("release_line")
            and old.get("closed_final_version", old.get("final_version"))
            == current.get("final_version")
        ):
            return old
    return None


def _version_seen(prior: Iterable[Mapping[str, Any]], version: Any) -> bool:
    if not isinstance(version, str):
        return False
    reservation_state: dict[str, str] = {}
    reservation_version: dict[str, str] = {}
    occupied = False
    for old in prior:
        event_type = old["event_type"]
        if (
            event_type == "development_reservation_consumed"
            and old.get("closed_final_version") == version
        ):
            occupied = True
        reservation_id = old.get("reservation_id")
        if reservation_id is not None and event_type.startswith("development_reservation_"):
            reservation_state[str(reservation_id)] = event_type.removeprefix(
                "development_reservation_"
            )
            if old.get("final_version"):
                reservation_version[str(reservation_id)] = str(old["final_version"])
            continue
        if event_type == "maintenance_line_opened":
            dev = old.get("dev_version")
            if isinstance(dev, str) and dev.removesuffix("-DEV") == version:
                occupied = True
        elif event_type in {
            "candidate_opened",
            "release_intent_prepared",
            "released",
        } and old.get("final_version") == version:
            occupied = True
        elif event_type == "candidate_withdrawn":
            candidate = _find_candidate(prior, old.get("candidate_id"))
            if candidate is not None and candidate.get("final_version") == version:
                occupied = True
    for reservation_id, state in reservation_state.items():
        if reservation_version.get(reservation_id) == version and state != "aborted":
            occupied = True
    return occupied


def parse_stream(raw: bytes | bytearray | str) -> list[dict[str, Any]]:
    """Parse and validate a complete JSONL stream."""

    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    raw = bytes(raw)
    if not raw:
        return []
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EventCanonicalEncodingError("event stream is not valid UTF-8") from exc
    lines = raw.splitlines(keepends=True)
    if any(not line.endswith(b"\n") or line.endswith(b"\r\n") for line in lines):
        raise EventCanonicalEncodingError("each event must end in exactly one LF")
    events: list[dict[str, Any]] = []
    for line in lines:
        if line == b"\n":
            raise EventCanonicalEncodingError("blank JSONL records are not permitted")
        events.append(validate_event(line[:-1]))
    validate_event_sequence(events)
    return events


def validate_event_sequence(events: Iterable[Mapping[str, Any]]) -> None:
    prior: list[dict[str, Any]] = []
    for event in events:
        validate_transition(prior, event)
        prior.append(validate_event(event))


def append_only_prefix(old_raw: bytes, new_raw: bytes) -> bytes:
    """Validate an append and return its exact newly appended bytes."""

    old_raw = bytes(old_raw)
    new_raw = bytes(new_raw)
    if not new_raw.startswith(old_raw):
        raise EventTransitionError("new event stream is not an immutable byte-prefix append")
    suffix = new_raw[len(old_raw) :]
    if not suffix:
        raise EventTransitionError("append must add one event")
    parse_stream(old_raw)
    new_events = parse_stream(new_raw)
    old_events = parse_stream(old_raw)
    if len(new_events) != len(old_events) + 1:
        raise EventTransitionError("append must add exactly one JSONL event")
    if new_raw != old_raw + canonical_event_bytes(new_events[-1]) + b"\n":
        raise EventCanonicalEncodingError("appended event has noncanonical framing")
    return suffix


def make_event(event_type: str, *, event_id_value: str, timestamp_utc: str, **fields: Any) -> dict[str, Any]:
    """Construct a schema-shaped event before validation."""

    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id_value,
        "event_type": event_type,
        "timestamp_utc": timestamp_utc,
        **fields,
    }
    return validate_event(event)


__all__ = [
    "COMMON_FIELDS",
    "EVENT_TYPES",
    "MAX_EVENT_SEQUENCE",
    "SCHEMA_VERSION",
    "EventCanonicalEncodingError",
    "EventError",
    "EventIdExhausted",
    "EventSchemaError",
    "EventTransitionError",
    "UnsupportedCertificationBinding",
    "append_only_prefix",
    "canonical_event_bytes",
    "event_digest",
    "event_id",
    "make_event",
    "next_event_id",
    "parse_stream",
    "parse_timestamp",
    "validate_event",
    "validate_event_id",
    "validate_event_sequence",
    "validate_transition",
]
