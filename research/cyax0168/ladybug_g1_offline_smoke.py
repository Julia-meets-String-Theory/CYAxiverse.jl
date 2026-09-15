"""Network-blocked, calibration-only Ladybug 0.20.4 G1 smoke harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import tempfile

from .ladybug_backend import LadybugBackend, LadybugBackendError


SNAPSHOT = {
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


def run() -> dict[str, object]:
    original_connect = socket.socket.connect

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    socket.socket.connect = blocked
    try:
        with tempfile.TemporaryDirectory(prefix="cyax0168-g1-ladybug-smoke-") as temp:
            root = Path(temp)
            first = LadybugBackend(root / "first").build(SNAPSHOT)
            second = LadybugBackend(root / "second").build(SNAPSHOT)
            with LadybugBackend(first) as opened:
                traversal = opened.required_traversal("ver-1")
                logical = opened.logical_export()
            with LadybugBackend(second) as opened:
                deterministic = opened.logical_export() == logical

            logical_file = first / "logical.json"
            original_logical = logical_file.read_bytes()
            logical_file.write_bytes(original_logical + b" ")
            try:
                LadybugBackend(first).open()
            except LadybugBackendError:
                tamper_rejection = True
            else:
                tamper_rejection = False
            logical_file.write_bytes(original_logical)

            failed = LadybugBackend(root / "failed")
            try:
                failed.build(SNAPSHOT, fail_stage="after_export")
            except LadybugBackendError:
                partial_build_rejection = True
                assert failed.last_candidate is not None
                try:
                    LadybugBackend(failed.last_candidate).open()
                except LadybugBackendError:
                    recovery = True
                else:
                    recovery = False
            else:
                partial_build_rejection = recovery = False

            import ladybug  # type: ignore

            return {
                "candidate": "ladybug==0.20.4",
                "version": ladybug.__version__,
                "threads": 1,
                "network_blocked": True,
                "smoke_import": True,
                "clean_graph_creation": True,
                "reopen": True,
                "required_traversal": traversal == [["impl-1", "Implementation"], ["req-1", "Requirement"]],
                "complete_logical_export": bool(logical),
                "deterministic_rebuild": deterministic,
                "tamper_rejection": tamper_rejection,
                "partial_build_rejection": partial_build_rejection,
                "recovery": recovery,
                "graph_bytes": (first / "graph.lbdb").stat().st_size,
                "logical_export_sha256": hashlib.sha256(original_logical).hexdigest(),
                "traversal_rows": traversal,
            }
    finally:
        socket.socket.connect = original_connect


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
