"""One-cell, checkpointed identity runner for the isolated CYAX-0168 bundle.

This runner is intentionally scoped to the first declared calibration cell.
It validates the isolation packet and its two approved file inputs, invokes
only the supplied independent generator, computes framed and JSONL identity
digests, and atomically writes one cell evidence record followed by one
checkpoint.  It does not inspect or materialize any decision fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


# Prevent import-time bytecode writes next to the approved implementation.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "input" / "g1-pre-rerun-input-isolation-v1.json"
OUTPUT_DIR = ROOT / "output"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_packet_without_self(packet: dict[str, Any]) -> bytes:
    body = dict(packet)
    body.pop("packet_sha256", None)
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _canonical_jsonl(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write, fsync, and atomically replace one file in OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=OUTPUT_DIR)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(OUTPUT_DIR, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_packet() -> tuple[dict[str, Any], dict[str, Any]]:
    packet = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise RuntimeError("input packet is not a JSON object")
    declared_packet_hash = packet.get("packet_sha256")
    if not isinstance(declared_packet_hash, str):
        raise RuntimeError("input packet has no packet_sha256")
    actual_packet_hash = _sha256(_canonical_packet_without_self(packet))
    if actual_packet_hash != declared_packet_hash:
        raise RuntimeError(
            f"packet hash mismatch: declared={declared_packet_hash} actual={actual_packet_hash}"
        )

    if packet.get("schema_version") != "cyax-0168-input-isolation-v1":
        raise RuntimeError("unexpected input schema version")
    if packet.get("candidate_head") != "4ed8293fd5483f98eba6f4603c5b8d34e945f4bd":
        raise RuntimeError("unexpected candidate head")
    if packet.get("normative_head") != "fe31fed192bfbefa28946736cdb5d2ec134985e0":
        raise RuntimeError("unexpected normative head")

    allowed_hashes: dict[str, Any] = {}
    for entry in packet.get("allowed_inputs", []):
        if not isinstance(entry, dict):
            continue
        relative_path = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            continue
        candidate = (ROOT / relative_path).resolve()
        if candidate.parent != ROOT / Path(relative_path).parent:
            raise RuntimeError(f"allowed input escapes bundle: {relative_path}")
        if not candidate.is_file():
            raise RuntimeError(f"missing allowed input: {relative_path}")
        actual = _sha256(candidate.read_bytes())
        if actual != expected:
            raise RuntimeError(
                f"allowed input hash mismatch for {relative_path}: declared={expected} actual={actual}"
            )
        allowed_hashes[relative_path] = {
            "sha256": actual,
            "git_blob": entry.get("git_blob"),
        }

    declared_cells = packet.get("allowed_inputs", [])[-1].get("cells")
    expected_cells = [
        ["C0", "P-medium", 168900],
        ["C1", "P-low", 168911],
        ["C1", "P-low", 168912],
        ["C1", "P-medium", 168913],
        ["C1", "P-medium", 168914],
        ["C1", "P-high", 168915],
        ["C1", "P-high", 168916],
        ["C2", "P-low", 168921],
        ["C2", "P-low", 168922],
        ["C2", "P-medium", 168923],
        ["C2", "P-medium", 168924],
        ["C2", "P-high", 168925],
        ["C2", "P-high", 168926],
        ["C3", "P-low", 168931],
        ["C3", "P-medium", 168932],
        ["C3", "P-high", 168933],
    ]
    if declared_cells != expected_cells:
        raise RuntimeError("declared calibration matrix is not the approved 16-cell matrix")

    return packet, {
        "packet_sha256": declared_packet_hash,
        "packet_canonical_sha256": actual_packet_hash,
        "allowed_file_hashes": allowed_hashes,
        "runtime": {
            "implementation": "CPython",
            "version": ".".join(str(part) for part in sys.version_info[:3]),
            "stdlib_only": True,
        },
    }


def _framed_sha256(generator: Any, value: Any) -> str:
    return _sha256(generator.frame(value))


def _frame_safe_complete_value(value: Any) -> Any:
    """Make diagnostic manifest floats/raw bytes explicit before framing.

    The approved frame intentionally rejects floats, while ``complete_output``
    carries diagnostic profile rates as floats and source payloads as bytes.
    Tagged values preserve both type and exact representation without changing
    any generator record or semantic identity.
    """
    if isinstance(value, bytes):
        return {"__cyax_bytes_hex__": value.hex()}
    if isinstance(value, float):
        return {"__cyax_float_repr__": repr(value)}
    if isinstance(value, list):
        return [_frame_safe_complete_value(item) for item in value]
    if isinstance(value, tuple):
        return [_frame_safe_complete_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _frame_safe_complete_value(item) for key, item in value.items()}
    return value


def main() -> int:
    packet, validation = _validate_packet()

    # The owner pause boundary is the first declared cell.  Keep this list
    # literal and do not start any subsequent cell in this invocation.
    cell = ("C0", "P-medium", 168900)
    result_path = OUTPUT_DIR / "C0__P-medium__168900.json"
    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    if result_path.exists() and checkpoint_path.exists():
        return 0

    implementation_dir = ROOT / "research" / "cyax0168"
    sys.path.insert(0, str(implementation_dir))
    import generator_independent as generator  # noqa: PLC0415

    started = time.perf_counter()
    snapshot = generator.generate_calibration_snapshot(*cell)
    elapsed_seconds = time.perf_counter() - started
    complete = generator.complete_output(snapshot)
    reconstructed = generator.reconstruct_prf_trace(complete)

    # The framed digests use the generator's approved domain-separated
    # canonical encoding, preserving bytes as bytes and ordered arrays as
    # ordered arrays.  JSONL digests additionally expose the exact common
    # record serialization specified by the contract.
    record_values = {
        "entities": complete["entities"],
        "literals": complete["literals"],
        "source_revisions": complete["source_revisions"],
        "assertions": complete["assertions"],
    }
    framed_record_hashes = {
        name: _framed_sha256(generator, records) for name, records in record_values.items()
    }
    jsonl_record_hashes = {
        name: _sha256(_canonical_jsonl(records)) for name, records in record_values.items()
    }

    identities = {
        "complete_canonical_serialized_output_sha256": _framed_sha256(
            generator, _frame_safe_complete_value(complete)
        ),
        "complete_output_framing": (
            "sha256(frame(tagged_complete_output(snapshot))); bytes=hex-tagged, "
            "floats=repr-tagged because the approved frame rejects floats"
        ),
        "record_class_sha256": framed_record_hashes,
        "record_class_jsonl_sha256": jsonl_record_hashes,
        "source_bytes_sha256": _framed_sha256(generator, complete["source_bytes"]),
        "source_bytes_framing": "sha256(frame(source_bytes keyed by canonical_locator; raw bytes retained))",
        "construction_trace_sha256": _framed_sha256(generator, complete["construction_trace"]),
        "reconstructed_prf_trace_sha256": _framed_sha256(generator, reconstructed),
        "candidate_vector_registry_sha256": _framed_sha256(generator, complete["candidate_vectors"]),
        "assertion_id_sequence_sha256": _framed_sha256(generator, complete["assertion_ids"]),
        "logical_snapshot_checksum": complete["logical_snapshot_checksum"],
        "snapshot_id": complete["snapshot_id"],
    }

    # Preserve the complete generator manifest, plus enough deterministic
    # inventory to make every identity auditable without duplicating the large
    # physical payload and traces in this checkpoint artifact.
    result: dict[str, Any] = {
        "evidence_schema": "cyax-0168-isolated-generator-identities-v1",
        "status": "BLOCKED",
        "pause_reason": "owner_requested_pause_after_first_durable_cell",
        "cell": {"tier": cell[0], "profile_id": cell[1], "seed": cell[2]},
        "input_validation": validation,
        "generator_manifest": snapshot.manifest,
        "complete_output_inventory": {
            "entity_count": len(complete["entities"]),
            "literal_count": len(complete["literals"]),
            "source_revision_count": len(complete["source_revisions"]),
            "assertion_count": len(complete["assertions"]),
            "source_bytes_count": len(complete["source_bytes"]),
            "construction_trace_count": len(complete["construction_trace"]),
            "prf_choice_count": len(complete["prf_choices"]),
            "reconstructed_prf_trace_count": len(reconstructed),
            "candidate_vector_count": len(complete["candidate_vectors"]),
            "assertion_id_count": len(complete["assertion_ids"]),
        },
        "identity_serialization": {
            "framed": "generator_independent.frame; SHA-256 of the exact framed value",
            "record_jsonl": "UTF-8 sorted-key compact JSON, LF after each record, records in generator order",
        },
        "identities": identities,
        "diagnostic_elapsed_seconds": elapsed_seconds,
    }
    result_bytes = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    _atomic_write(result_path, result_bytes)

    checkpoint = {
        "checkpoint_schema": "cyax-0168-isolated-generator-checkpoint-v1",
        "status": "BLOCKED",
        "pause_reason": "owner_requested_pause_after_first_durable_cell",
        "bundle_root": str(ROOT),
        "input_packet_sha256": validation["packet_sha256"],
        "completed_cells": [
            {
                "tier": cell[0],
                "profile_id": cell[1],
                "seed": cell[2],
                "result_path": str(result_path),
                "result_sha256": _sha256(result_bytes),
                "logical_snapshot_checksum": identities["logical_snapshot_checksum"],
                "snapshot_id": identities["snapshot_id"],
            }
        ],
        "remaining_declared_cells": packet["allowed_inputs"][-1]["cells"][1:],
        "next_action": "STOP; do not start another cell until owner resumes",
    }
    checkpoint_bytes = (json.dumps(checkpoint, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    _atomic_write(checkpoint_path, checkpoint_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
