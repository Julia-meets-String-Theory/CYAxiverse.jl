"""Validation-only maintenance-bootstrap contract tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.maintenance_validation import (  # noqa: E402
    MaintenanceValidationError,
    validate_maintenance_bootstrap_fixture,
)


def fixture(**changes: object) -> dict[str, object]:
    manifest_id = "LIF-SHA256-" + "1" * 64
    bootstrap_id = "BST-SHA256-" + "2" * 64
    result: dict[str, object] = {
        "schema_version": 1,
        "record_type": "maintenance-bootstrap-validation-v1",
        "fixture_only": True,
        "approved_base_ref": "refs/tags/v1.2.2",
        "approved_base_sha": "a" * 40,
        "approved_base_tree": "b" * 40,
        "line_absence_proof_digest": "3" * 64,
        "static_iteration_snapshot": "4" * 64,
        "lifecycle_ref_snapshot": "5" * 64,
        "owner_line": "maintenance/1.2",
        "selected_final_version": "1.2.3",
        "selected_dev_version": "1.2.3-DEV",
        "line_ref": "refs/heads/maintenance/1.2",
        "expected_branch_head": "a" * 40,
        "observed_branch_head": "c" * 40,
        "activated_dev_head": "c" * 40,
        "reservation_ref": (
            "refs/heads/lifecycle/v1/reservations/maintenance-1.2/"
            f"v1.2.3-DEV/{manifest_id}"
        ),
        "reservation_manifest_id": manifest_id,
        "bootstrap_ref": (
            "refs/heads/lifecycle/v1/maintenance-bootstrap/maintenance-1.2/"
            f"v1.2.3/{bootstrap_id}"
        ),
        "bootstrap_id": bootstrap_id,
        "activation_status": "ACTIVATED",
        "line_frozen": False,
        "version_unavailable": True,
    }
    result.update(changes)
    return result


class MaintenanceValidationTests(unittest.TestCase):
    def test_activated_correspondence_is_valid(self) -> None:
        self.assertEqual(
            validate_maintenance_bootstrap_fixture(fixture())["activation_status"],
            "ACTIVATED",
        )

    def test_proven_nonentry_is_valid_only_when_released(self) -> None:
        value = fixture(
            activation_status="PROVEN_NON_ENTRY",
            observed_branch_head="ABSENT",
            activated_dev_head="ABSENT",
            version_unavailable=False,
        )
        self.assertEqual(
            validate_maintenance_bootstrap_fixture(value)["activation_status"],
            "PROVEN_NON_ENTRY",
        )
        with self.assertRaises(MaintenanceValidationError):
            validate_maintenance_bootstrap_fixture(
                dict(value, version_unavailable=True)
            )

    def test_identity_mismatch_is_invalid(self) -> None:
        for changes in (
            {"expected_branch_head": "d" * 40},
            {"owner_line": "maintenance/1.3"},
            {"selected_dev_version": "1.2.4-DEV"},
            {"line_ref": "refs/heads/maintenance/1.3"},
            {"observed_branch_head": "d" * 40},
            {"reservation_ref": "refs/heads/lifecycle/v1/reservations/bad"},
            {"bootstrap_ref": "refs/heads/lifecycle/v1/maintenance-bootstrap/bad"},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                MaintenanceValidationError
            ):
                validate_maintenance_bootstrap_fixture(fixture(**changes))

    def test_uncertain_and_failed_activation_stay_frozen_and_unavailable(self) -> None:
        for status in ("CREATION_UNCERTAIN", "ACTIVATION_FAILED"):
            value = fixture(
                activation_status=status,
                observed_branch_head="UNKNOWN",
                activated_dev_head="UNKNOWN",
                line_frozen=True,
            )
            self.assertEqual(
                validate_maintenance_bootstrap_fixture(value)["activation_status"],
                status,
            )
            with self.assertRaises(MaintenanceValidationError):
                validate_maintenance_bootstrap_fixture(
                    dict(value, line_frozen=False)
                )
            with self.assertRaises(MaintenanceValidationError):
                validate_maintenance_bootstrap_fixture(
                    dict(value, version_unavailable=False)
                )


if __name__ == "__main__":
    unittest.main()
