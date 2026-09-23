"""Focused R-046 synthetic owner-authority coverage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.authorization import (  # noqa: E402
    AuthorizationError,
    AuthorizationResolution,
    authorization_digest,
    authorization_id,
    authorization_id_preimage,
    canonical_authorization_bytes,
    seal_authorization,
    validate_authorization,
    verify_owner_authorization,
)
from version_lifecycle.codec import canonical_json  # noqa: E402


VECTOR_PREIMAGE = (
    b'{"authority_source_ref":"owner-authority://synthetic/grant-001",'
    b'"authorized_actions":["create-release-manifest","create-tag"],'
    b'"expires_at_utc":"2026-01-03T00:00:00Z",'
    b'"final_version":"0.3.0",'
    b'"issued_at_utc":"2026-01-02T00:00:00Z",'
    b'"owner_account":"owner-123","owner_line":"principal",'
    b'"repository":"Julia-meets-String-Theory/CYAxiverse.jl",'
    b'"schema_version":1,'
    b'"target_refs":["refs/heads/lifecycle/v1/releases/v0.3.0",'
    b'"refs/tags/v0.3.0"],"transaction_id":"txn-0125-001"}'
)
VECTOR_DIGEST = "8531fbb361a9d99f67e45e82f8511447ec9f0f03b8d1fa96ac281d5dff3daa40"
VECTOR_ID = "AUTH-SHA256-" + VECTOR_DIGEST


def vector_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": "Julia-meets-String-Theory/CYAxiverse.jl",
        "owner_account": "owner-123",
        "authority_source_ref": "owner-authority://synthetic/grant-001",
        "issued_at_utc": "2026-01-02T00:00:00Z",
        "expires_at_utc": "2026-01-03T00:00:00Z",
        "transaction_id": "txn-0125-001",
        "owner_line": "principal",
        "final_version": "0.3.0",
        "authorized_actions": ["create-release-manifest", "create-tag"],
        "target_refs": [
            "refs/heads/lifecycle/v1/releases/v0.3.0",
            "refs/tags/v0.3.0",
        ],
    }


class Authority:
    def __init__(self, record: dict[str, object], *, owner: bool = True, raw: bytes | None = None):
        self.record = record
        self.owner = owner
        self.raw = raw if raw is not None else canonical_authorization_bytes(record)

    def fetch_owner_authorization(self, reference: str) -> AuthorizationResolution:
        return AuthorizationResolution(self.record, self.raw, self.owner)


class AuthorizationTests(unittest.TestCase):
    def test_frozen_r046_vector_is_exact(self):
        record = seal_authorization(vector_record())
        self.assertEqual(authorization_id_preimage(record), VECTOR_PREIMAGE)
        self.assertEqual(len(VECTOR_PREIMAGE), 465)
        self.assertEqual(authorization_digest(record), VECTOR_DIGEST)
        self.assertEqual(record["owner_authorization_digest"], VECTOR_DIGEST)
        self.assertEqual(record["owner_authorization"], VECTOR_ID)
        self.assertEqual(authorization_id(record), VECTOR_ID)
        self.assertEqual(canonical_authorization_bytes(record), canonical_json(record))
        self.assertEqual(validate_authorization(record), record)

    def test_each_identity_is_non_circular_and_tamper_is_rejected(self):
        record = seal_authorization(vector_record())
        for field in ("owner_authorization", "owner_authorization_digest"):
            tampered = dict(record)
            tampered[field] = (
                "AUTH-SHA256-" + "0" * 64
                if field == "owner_authorization" else "0" * 64
            )
            with self.assertRaises(AuthorizationError):
                validate_authorization(tampered)
        for field in ("repository", "owner_account", "transaction_id", "owner_line",
                      "final_version", "authority_source_ref", "issued_at_utc",
                      "expires_at_utc"):
            tampered = dict(record)
            tampered[field] = str(tampered[field]) + "-tampered"
            with self.assertRaises(AuthorizationError):
                validate_authorization(tampered)
        preimage_classes = {
            "schema_version": 2,
            "authorized_actions": ["create-tag"],
            "target_refs": ["refs/tags/v0.4.0"],
        }
        for field, value in preimage_classes.items():
            tampered = dict(record)
            tampered[field] = value
            with self.subTest(field=field), self.assertRaises(AuthorizationError):
                validate_authorization(tampered)

    def test_private_or_noncanonical_authority_reference_is_rejected(self):
        unsafe = (
            "file:///Users/alice/private/grant.json",
            "/Users/alice/private/grant.json",
            "https://authority.example/grant-001",
            "owner-authority://alice:password@synthetic/grant-001",
            "owner-authority://synthetic/../grant-001",
            "owner-authority://synthetic/secret-token",
            "owner-authority://synthetic/grant-001?token=value",
            "owner-authority://synthetic/grant-001?",
            "owner-authority://synthetic/grant-001#",
            "owner-authority://synthetic/grant-001#secret-locator",
            "owner-authority://synthetic/ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "owner-authority://synthetic/sk-live-abcdefghijklmnopqrstuvwxyz012345",
            "owner-authority://synthetic/grant%2Fprivate",
            "owner-authority://synthetic/prefixghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "OWNER-AUTHORITY://synthetic/grant-001",
            "owner-authority://localhost/grant-001",
            "owner-authority://127.0.0.1/grant-001",
            "owner-authority://synthetic/Users/alice/grant-001",
        )
        for reference in unsafe:
            value = vector_record()
            value["authority_source_ref"] = reference
            with self.subTest(reference=reference), self.assertRaises(
                AuthorizationError
            ):
                seal_authorization(value)

    def test_token_shaped_transaction_id_is_rejected(self):
        value = vector_record()
        value["transaction_id"] = "prefixghp_abcdefghijklmnopqrstuvwxyz0123456789"
        with self.assertRaisesRegex(AuthorizationError, "safe public value"):
            seal_authorization(value)

    def test_changed_bytes_or_owner_assertion_blocks(self):
        record = seal_authorization(vector_record())
        reference = str(record["authority_source_ref"])
        kwargs = dict(
            repository=str(record["repository"]), transaction_id=str(record["transaction_id"]),
            action="create-tag", owner_line="principal", final_version="0.3.0",
            target_ref="refs/tags/v0.3.0", now_utc="2026-01-02T12:00:00Z",
        )
        with self.assertRaises(AuthorizationError):
            verify_owner_authorization(Authority(record, raw=canonical_authorization_bytes(record) + b"x"), reference, **kwargs)
        with self.assertRaises(AuthorizationError):
            verify_owner_authorization(Authority(record, owner=False), reference, **kwargs)
        malformed = {"record": record, "canonical_bytes": canonical_authorization_bytes(record),
                     "repository_owner_verified": "true"}
        with self.assertRaises(AuthorizationError):
            verify_owner_authorization(lambda _reference: malformed, reference, **kwargs)

    def test_cross_scope_expiry_action_and_target_fail(self):
        record = seal_authorization(vector_record())
        reference = str(record["authority_source_ref"])
        base = dict(
            repository="Julia-meets-String-Theory/CYAxiverse.jl", transaction_id="txn-0125-001",
            action="create-tag", owner_line="principal", final_version="0.3.0",
            target_ref="refs/tags/v0.3.0", now_utc="2026-01-02T12:00:00Z",
        )
        cases = [
            {"repository": "other/repository"},
            {"transaction_id": "other-transaction"},
            {"action": "delete-release"},
            {"owner_line": "maintenance/1.2"},
            {"final_version": "0.4.0"},
            {"target_ref": "refs/tags/v0.4.0"},
            {"now_utc": "2026-01-03T00:00:00Z"},
        ]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(AuthorizationError):
                verify_owner_authorization(Authority(record), reference, **(base | change))


if __name__ == "__main__":
    unittest.main()
