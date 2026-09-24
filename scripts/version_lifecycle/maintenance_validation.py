"""Validation-only maintenance-bootstrap contract for Gate A fixtures.

This module does not create a maintenance line, lifecycle manifest, or Git ref.
Production maintenance automation remains outside Gate A.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .git_refs import GitIdentityError, require_full_ref
from .versions import parse_package_version


SCHEMA_VERSION = 1
RECORD_TYPE = "maintenance-bootstrap-validation-v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_ID_RE = re.compile(r"^LIF-SHA256-[0-9a-f]{64}$")
BOOTSTRAP_ID_RE = re.compile(r"^BST-SHA256-[0-9a-f]{64}$")
STATUSES = frozenset(
    {"ACTIVATED", "PROVEN_NON_ENTRY", "CREATION_UNCERTAIN", "ACTIVATION_FAILED"}
)
FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "fixture_only",
        "approved_base_ref",
        "approved_base_sha",
        "approved_base_tree",
        "line_absence_proof_digest",
        "static_iteration_snapshot",
        "lifecycle_ref_snapshot",
        "owner_line",
        "selected_final_version",
        "selected_dev_version",
        "line_ref",
        "expected_branch_head",
        "observed_branch_head",
        "activated_dev_head",
        "reservation_ref",
        "reservation_manifest_id",
        "bootstrap_ref",
        "bootstrap_id",
        "activation_status",
        "line_frozen",
        "version_unavailable",
    }
)


class MaintenanceValidationError(ValueError):
    """The validation-only maintenance contract is inconsistent."""


def _full_ref(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise MaintenanceValidationError(f"{field} must be a full Git ref")
    try:
        return require_full_ref(value)
    except GitIdentityError as error:
        raise MaintenanceValidationError(f"{field} must be a full Git ref") from error


def validate_maintenance_bootstrap_fixture(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate R-030/R-031 without enabling a production writer."""

    if not isinstance(record, Mapping) or set(record) != FIELDS:
        raise MaintenanceValidationError("maintenance fixture fields are noncanonical")
    value = dict(record)
    if value["schema_version"] != SCHEMA_VERSION or isinstance(
        value["schema_version"], bool
    ):
        raise MaintenanceValidationError("unsupported maintenance fixture schema")
    if value["record_type"] != RECORD_TYPE or value["fixture_only"] is not True:
        raise MaintenanceValidationError("maintenance record must be validation-only")
    if value["activation_status"] not in STATUSES:
        raise MaintenanceValidationError("unknown maintenance activation status")
    for field in ("line_frozen", "version_unavailable"):
        if not isinstance(value[field], bool):
            raise MaintenanceValidationError(f"{field} must be boolean")
    for field in (
        "line_absence_proof_digest",
        "static_iteration_snapshot",
        "lifecycle_ref_snapshot",
    ):
        if not isinstance(value[field], str) or DIGEST_RE.fullmatch(value[field]) is None:
            raise MaintenanceValidationError(f"{field} must be a SHA-256 digest")
    for field in ("approved_base_sha", "approved_base_tree", "expected_branch_head"):
        if not isinstance(value[field], str) or SHA_RE.fullmatch(value[field]) is None:
            raise MaintenanceValidationError(f"{field} must be a full Git SHA")
    if value["expected_branch_head"] != value["approved_base_sha"]:
        raise MaintenanceValidationError(
            "maintenance expected branch head must equal the approved base"
        )

    base_ref = _full_ref(value["approved_base_ref"], "approved_base_ref")
    if not base_ref.startswith("refs/tags/"):
        raise MaintenanceValidationError("approved maintenance base must be immutable")
    final = parse_package_version(value["selected_final_version"])
    dev = parse_package_version(value["selected_dev_version"])
    if final.is_dev or not dev.is_dev or dev.final != final:
        raise MaintenanceValidationError("maintenance final/DEV versions do not correspond")
    line = f"maintenance/{final.major}.{final.minor}"
    if value["owner_line"] != line:
        raise MaintenanceValidationError("maintenance owner line does not match version")
    expected_line_ref = f"refs/heads/{line}"
    if _full_ref(value["line_ref"], "line_ref") != expected_line_ref:
        raise MaintenanceValidationError("maintenance branch ref does not match owner line")

    manifest_id = value["reservation_manifest_id"]
    bootstrap_id = value["bootstrap_id"]
    if not isinstance(manifest_id, str) or MANIFEST_ID_RE.fullmatch(manifest_id) is None:
        raise MaintenanceValidationError("reservation manifest ID is invalid")
    if not isinstance(bootstrap_id, str) or BOOTSTRAP_ID_RE.fullmatch(bootstrap_id) is None:
        raise MaintenanceValidationError("bootstrap ID is invalid")
    owner_component = f"maintenance-{final.major}.{final.minor}"
    expected_reservation_ref = (
        "refs/heads/lifecycle/v1/reservations/"
        f"{owner_component}/v{dev.canonical}/{manifest_id}"
    )
    expected_bootstrap_ref = (
        "refs/heads/lifecycle/v1/maintenance-bootstrap/"
        f"{owner_component}/v{final.canonical}/{bootstrap_id}"
    )
    if _full_ref(value["reservation_ref"], "reservation_ref") != expected_reservation_ref:
        raise MaintenanceValidationError("reservation identity does not correspond")
    if _full_ref(value["bootstrap_ref"], "bootstrap_ref") != expected_bootstrap_ref:
        raise MaintenanceValidationError("bootstrap identity does not correspond")

    status = value["activation_status"]
    observed = value["observed_branch_head"]
    dev_head = value["activated_dev_head"]
    if status == "ACTIVATED":
        if (
            not isinstance(observed, str)
            or SHA_RE.fullmatch(observed) is None
            or observed != dev_head
            or value["line_frozen"]
            or not value["version_unavailable"]
        ):
            raise MaintenanceValidationError("activated maintenance identities do not correspond")
    elif status == "PROVEN_NON_ENTRY":
        if (
            observed != "ABSENT"
            or dev_head != "ABSENT"
            or value["line_frozen"]
            or value["version_unavailable"]
        ):
            raise MaintenanceValidationError("maintenance non-entry proof is inconsistent")
    else:
        if not value["line_frozen"] or not value["version_unavailable"]:
            raise MaintenanceValidationError(
                "uncertain or failed maintenance activation must remain frozen and unavailable"
            )
        for field_value in (observed, dev_head):
            if field_value != "UNKNOWN" and (
                not isinstance(field_value, str) or SHA_RE.fullmatch(field_value) is None
            ):
                raise MaintenanceValidationError("uncertain maintenance head is invalid")
    return value


__all__ = [
    "MaintenanceValidationError",
    "RECORD_TYPE",
    "SCHEMA_VERSION",
    "validate_maintenance_bootstrap_fixture",
]
