"""Tiny synthetic known-truth tests for the Ladybug G1 adapter.

These tests never use a campaign fixture.  They skip on a backend-free Python
interpreter and are executed independently in the frozen wheel environment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    from .ladybug_backend import LadybugBackend, LadybugBackendError
except ImportError:  # flat same-directory test invocation
    from ladybug_backend import LadybugBackend, LadybugBackendError


HAS_LADYBUG = importlib.util.find_spec("ladybug") is not None


def _tiny_snapshot() -> dict[str, list[dict[str, object]]]:
    return {
        "entities": [
            {"id": "impl-1", "kind": "Implementation"},
            {"id": "req-1", "kind": "Requirement"},
            {"id": "ver-1", "kind": "Verification"},
        ],
        "literals": [],
        "source_revisions": [],
        "assertions": [
            {"id": "assert-1", "source_id": "impl-1", "target_id": "req-1", "predicate": "implements"},
            {"id": "assert-2", "source_id": "ver-1", "target_id": "impl-1", "predicate": "verifies"},
        ],
    }


@unittest.skipUnless(HAS_LADYBUG, "requires the frozen ladybug wheel")
class LadybugBackendTests(unittest.TestCase):
    def test_build_reopen_query_export_and_deterministic_rebuild(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyax0168-ladybug-test-") as temp:
            root = Path(temp)
            first = LadybugBackend(root / "first").build(_tiny_snapshot())
            second = LadybugBackend(root / "second").build(_tiny_snapshot())
            first_backend = LadybugBackend(first).open()
            try:
                self.assertEqual(
                    first_backend.required_traversal("ver-1"),
                    [["impl-1", "Implementation"], ["req-1", "Requirement"]],
                )
                expected = first_backend.logical_export()
            finally:
                first_backend.close()
            with LadybugBackend(second) as reopened:
                self.assertEqual(reopened.logical_export(), expected)
            self.assertEqual(
                hashlib.sha256((first / "logical.json").read_bytes()).digest(),
                hashlib.sha256((second / "logical.json").read_bytes()).digest(),
            )

    def test_tamper_and_incomplete_candidate_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyax0168-ladybug-test-") as temp:
            root = Path(temp)
            destination = root / "graph"
            LadybugBackend(destination).build(_tiny_snapshot())
            logical = destination / "logical.json"
            original = logical.read_bytes()
            logical.write_bytes(original + b" ")
            with self.assertRaises(LadybugBackendError):
                LadybugBackend(destination).open()
            logical.write_bytes(original)
            graph = destination / "graph.lbdb"
            graph_original = graph.read_bytes()
            graph.write_bytes(graph_original + b"tamper")
            with self.assertRaises(LadybugBackendError):
                LadybugBackend(destination).open()
            graph.write_bytes(graph_original)
            manifest = destination / "manifest.json"
            manifest_record = json.loads(manifest.read_bytes().decode("utf-8"))
            manifest_record["upstream_commit"] = "tampered"
            manifest.write_bytes((json.dumps(manifest_record, sort_keys=True, separators=(",", ":")) + "\n").encode())
            with self.assertRaises(LadybugBackendError):
                LadybugBackend(destination).open()

            failed = LadybugBackend(root / "failed")
            with self.assertRaises(LadybugBackendError):
                failed.build(_tiny_snapshot(), fail_stage="after_export")
            self.assertIsNotNone(failed.last_candidate)
            with self.assertRaises(LadybugBackendError):
                LadybugBackend(failed.last_candidate).open()  # type: ignore[arg-type]
            # The final destination was never published and is recoverable.
            self.assertFalse((root / "failed").exists())
            LadybugBackend(root / "failed").build(_tiny_snapshot())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
