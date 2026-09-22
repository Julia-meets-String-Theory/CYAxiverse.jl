"""Focused static/lifecycle snapshot and allocation boundary tests."""

from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.allocation import (  # noqa: E402
    global_allocation_view,
    select_maintenance_version,
    select_principal_sentinel,
)
from version_lifecycle.codec import canonical_json, sha256_hex  # noqa: E402
from version_lifecycle.manifests import (  # noqa: E402
    LifecycleRefSnapshot,
    ManifestError,
    _mark_lifecycle_authority_verified,
    validate_lifecycle_ref_snapshot,
)
from version_lifecycle.static import (  # noqa: E402
    BlockedResult,
    StaticSnapshot,
    StaticValidationError,
    _mark_authority_verified,
    _anchor_bindings,
    _annotated_anchor_payload,
    recompute_snapshot_digests,
    validate_static_snapshot,
)


SOURCE_REPOSITORY = "https://github.com/cyaxiverse/CYAxiverse.jl"


def static_snapshot(
    occupied: list[str] | None = None,
    *,
    verified: bool = True,
    iteration_bindings: list[dict[str, str]] | None = None,
) -> StaticSnapshot:
    raw = b'target_iteration = "fixture"\n'
    bindings = iteration_bindings or []
    data: dict[str, object] = {
        "snapshot_schema_version": 1,
        "canonical_static_iteration_source": "refs/heads/vmm:iterations.toml",
        "source_repository": SOURCE_REPOSITORY,
        "source_ref": "refs/heads/vmm",
        "source_path": "iterations.toml",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "iterations_toml_sha256": sha256_hex(raw),
        "iteration_ref_bindings": bindings,
        "ref_set_digest": sha256_hex(canonical_json(bindings)),
        "public_tag_bindings": [],
        "tag_set_digest": sha256_hex(canonical_json([])),
        "occupied_versions": sorted(occupied or []),
    }
    data.update(recompute_snapshot_digests(data))
    result = StaticSnapshot(data, source_bytes=raw)
    return _mark_authority_verified(result, "fixture-authority") if verified else result


def lifecycle_snapshot(
    occupied: list[str] | None = None,
    *,
    verified: bool = True,
) -> LifecycleRefSnapshot:
    data: dict[str, object] = {
        "snapshot_schema_version": 1,
        "source_repository": SOURCE_REPOSITORY,
        "lifecycle_ref_bindings": [],
        "lifecycle_ref_set_digest": sha256_hex(canonical_json([])),
        "occupied_versions": sorted(occupied or []),
    }
    data["lifecycle_snapshot_digest"] = sha256_hex(canonical_json(data))
    result = validate_lifecycle_ref_snapshot(data)
    return _mark_lifecycle_authority_verified(result) if verified else result


class StaticAuthorityTests(unittest.TestCase):
    @staticmethod
    def _tag_object(root: Path, commit: str, timestamp: str, epoch: int) -> str:
        raw = (
            f"object {commit}\n"
            "type commit\n"
            "tag iterations/1.2.3\n"
            f"tagger Fixture <fixture@example.invalid> {epoch} +0000\n"
            "\n"
            f"closure_timestamp_utc={timestamp}\n"
        ).encode("ascii")
        return subprocess.check_output(
            ["git", "-C", str(root), "hash-object", "-t", "tag", "-w", "--stdin"],
            input=raw,
        ).decode().strip()

    def test_anchor_binding_rejects_lightweight_and_binds_tag_object_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for key, value in (("user.name", "Fixture"), ("user.email", "fixture@example.invalid")):
                subprocess.run(["git", "-C", str(root), "config", key, value], check=True)
            (root / "iterations.toml").write_text(
                'schema_version = 1\ntarget_iteration = "fixture"\n', encoding="utf-8"
            )
            subprocess.run(["git", "-C", str(root), "add", "iterations.toml"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            with self.assertRaisesRegex(StaticValidationError, "annotated tag"):
                _anchor_bindings(
                    root, {"refs/tags/iterations/1.2.3": commit}
                )
            first = self._tag_object(
                root, commit, "2026-09-20T12:34:56Z", 1789907696
            )
            second = self._tag_object(
                root, commit, "2026-09-20T12:34:57Z", 1789907697
            )
            self.assertNotEqual(first, second)
            self.assertEqual(
                _annotated_anchor_payload(
                    root, first, "iterations/1.2.3", commit
                ),
                "2026-09-20T12:34:56Z",
            )
            binding = {
                "ref": "iterations/1.2.3",
                "tag_object": first,
                "object_type": "tag",
                "commit": commit,
                "tree": subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
                ).strip(),
                "closure_timestamp_utc": "2026-09-20T12:34:56Z",
            }
            original = static_snapshot(iteration_bindings=[binding]).snapshot_digest
            changed = dict(binding)
            changed["tag_object"] = second
            changed["closure_timestamp_utc"] = "2026-09-20T12:34:57Z"
            self.assertNotEqual(
                original,
                static_snapshot(iteration_bindings=[changed]).snapshot_digest,
            )

    def test_allocation_rejects_structural_but_unverified_static_snapshot(self) -> None:
        result = global_allocation_view(static_snapshot(verified=False), lifecycle_snapshot())
        self.assertIsInstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "STATIC_AUTHORITY_SELECTOR_UNRESOLVED")

    def test_allocation_requires_complete_lifecycle_snapshot(self) -> None:
        result = global_allocation_view(static_snapshot(), None)
        self.assertIsInstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "LIFECYCLE_REF_SNAPSHOT_INVALID")

    def test_allocation_rejects_structural_but_unverified_lifecycle_snapshot(self) -> None:
        result = global_allocation_view(
            static_snapshot(), lifecycle_snapshot(verified=False)
        )
        self.assertIsInstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "LIFECYCLE_REF_SNAPSHOT_INVALID")

    def test_cross_repository_snapshots_are_blocked(self) -> None:
        lifecycle = lifecycle_snapshot().to_dict()
        lifecycle["source_repository"] = "https://github.com/cyaxiverse/other"
        lifecycle["lifecycle_snapshot_digest"] = sha256_hex(
            canonical_json({k: v for k, v in lifecycle.items() if k != "lifecycle_snapshot_digest"})
        )
        result = global_allocation_view(static_snapshot(), lifecycle)
        self.assertIsInstance(result, BlockedResult)
        self.assertEqual(result.reason_code, "LIFECYCLE_REF_SNAPSHOT_INVALID")

    def test_static_raw_source_digest_is_verified(self) -> None:
        snapshot = static_snapshot()
        with self.assertRaises(StaticValidationError):
            validate_static_snapshot(snapshot, source_bytes=b"different")

    def test_static_snapshot_rejects_reversed_duplicate_and_nested_tamper(self) -> None:
        original = static_snapshot(["1.2.3", "1.2.4"]).to_dict()
        reversed_values = copy.deepcopy(original)
        reversed_values["occupied_versions"] = ["1.2.4", "1.2.3"]
        with self.assertRaises(StaticValidationError):
            recompute_snapshot_digests(reversed_values)
        duplicate = copy.deepcopy(original)
        duplicate["occupied_versions"] = ["1.2.3", "1.2.3"]
        with self.assertRaises(StaticValidationError):
            recompute_snapshot_digests(duplicate)
        tampered = copy.deepcopy(original)
        tampered["source_tree"] = "c" * 40
        with self.assertRaises(StaticValidationError):
            validate_static_snapshot(tampered, structural_only=True)

    def test_empty_snapshot_digest_preimages_are_exact(self) -> None:
        snapshot = static_snapshot()
        empty = sha256_hex(b"[]")
        self.assertEqual(snapshot.ref_set_digest, empty)
        self.assertEqual(snapshot.tag_set_digest, empty)
        self.assertEqual(lifecycle_snapshot().ref_set_digest, empty)

    def test_lifecycle_snapshot_rejects_tampered_digest_and_occupied_set(self) -> None:
        value = lifecycle_snapshot(["1.2.3"]).to_dict()
        value["occupied_versions"] = ["1.2.4"]
        with self.assertRaises(ManifestError):
            validate_lifecycle_ref_snapshot(value)
        value = lifecycle_snapshot(["1.2.3", "1.2.4"]).to_dict()
        value["occupied_versions"] = ["1.2.4", "1.2.3"]
        with self.assertRaises(ManifestError):
            validate_lifecycle_ref_snapshot(value)

    def test_combined_view_occupies_static_and_lifecycle_versions(self) -> None:
        view = global_allocation_view(
            static_snapshot(["1.2.3"]), lifecycle_snapshot(["1.2.4"])
        )
        self.assertFalse(isinstance(view, BlockedResult))
        self.assertEqual(view.occupied, frozenset({"1.2.3", "1.2.4"}))

    def test_principal_sentinel_is_exact_and_never_skips(self) -> None:
        available = select_principal_sentinel(
            "1.2.3", static_snapshot(), lifecycle_snapshot()
        )
        self.assertEqual(available.status, "AVAILABLE")
        self.assertEqual(str(available.version), "1.2.4-DEV")
        blocked = select_principal_sentinel(
            "1.2.3", static_snapshot(), lifecycle_snapshot(["1.2.4"])
        )
        self.assertEqual(blocked.reason_code, "PRINCIPAL_SENTINEL_UNAVAILABLE")
        self.assertIsNone(blocked.version)

    def test_maintenance_selection_preserves_future_line_grammar(self) -> None:
        decision = select_maintenance_version(
            "maintenance/1.2",
            "1.2.3",
            static_snapshot(),
            lifecycle_snapshot(["1.2.4"]),
            max_search=3,
        )
        self.assertEqual(decision.status, "AVAILABLE")
        self.assertEqual(str(decision.version), "1.2.5-DEV")


if __name__ == "__main__":
    unittest.main()
