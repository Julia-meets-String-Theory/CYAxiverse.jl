"""Immutable pre-rerun identity packet for CYAX-0168 G1.

The packet is deliberately small.  It binds one exact candidate commit to one
frozen host-manifest hash and carries no measured result or fixture identity.
This makes it safe to prepare before a later rerun while preventing accidental
mixing of a new candidate with an older host context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from .host import HostError, HostManifest, load_frozen_host_manifest
except ImportError:  # pragma: no cover - direct unittest discovery
    from host import HostError, HostManifest, load_frozen_host_manifest


PACKET_SCHEMA_VERSION = "cyax-0168-pre-rerun-execution-packet-v1"
PACKET_GATE = "CYAX-0168-G1"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ExecutionPacketError(ValueError):
    """Raised when a pre-rerun identity packet is not exact or immutable."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _manifest_id(value: HostManifest | Mapping[str, Any]) -> str:
    if isinstance(value, HostManifest):
        return value.identity
    try:
        return load_frozen_host_manifest(value).identity
    except HostError as exc:
        raise ExecutionPacketError(str(exc)) from exc


@dataclass(frozen=True)
class PreRerunExecutionPacket:
    candidate_head: str
    host_manifest_id: str
    candidate_context_id: str
    schema_version: str = PACKET_SCHEMA_VERSION
    gate: str = PACKET_GATE
    scope: str = "calibration-only; decision fixtures prohibited"

    def __post_init__(self) -> None:
        if not _COMMIT_PATTERN.fullmatch(self.candidate_head):
            raise ExecutionPacketError("candidate_head must be a 40-character lowercase commit SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", self.host_manifest_id):
            raise ExecutionPacketError("host_manifest_id must be a 64-character lowercase SHA-256")
        expected = context_identity(self.candidate_head, self.host_manifest_id)
        if self.candidate_context_id != expected:
            raise ExecutionPacketError("candidate context hash does not match packet identities")
        if self.schema_version != PACKET_SCHEMA_VERSION or self.gate != PACKET_GATE:
            raise ExecutionPacketError("unsupported execution packet schema or gate")

    @property
    def packet_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_record(include_hash=False))).hexdigest()

    def to_record(self, *, include_hash: bool = True) -> dict[str, Any]:
        record = {
            "candidate_head": self.candidate_head,
            "candidate_context_id": self.candidate_context_id,
            "gate": self.gate,
            "host_manifest_id": self.host_manifest_id,
            "schema_version": self.schema_version,
            "scope": self.scope,
        }
        if include_hash:
            record["packet_sha256"] = self.packet_sha256
        return record


def context_identity(candidate_head: str, host_manifest_id: str) -> str:
    """Hash the exact pair, making the no-mixed-context rule explicit."""
    if not _COMMIT_PATTERN.fullmatch(candidate_head):
        raise ExecutionPacketError("candidate_head must be a 40-character lowercase commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", host_manifest_id):
        raise ExecutionPacketError("host_manifest_id must be a 64-character lowercase SHA-256")
    return hashlib.sha256(_canonical_json({
        "candidate_head": candidate_head,
        "host_manifest_id": host_manifest_id,
    })).hexdigest()


def build_execution_packet(
    candidate_head: str,
    host_manifest: HostManifest | Mapping[str, Any],
) -> PreRerunExecutionPacket:
    """Bind one exact candidate head to one frozen host-manifest identity."""
    manifest_id = _manifest_id(host_manifest)
    return PreRerunExecutionPacket(
        candidate_head=candidate_head,
        host_manifest_id=manifest_id,
        candidate_context_id=context_identity(candidate_head, manifest_id),
    )


def validate_execution_packet(
    packet: PreRerunExecutionPacket | Mapping[str, Any] | str | os.PathLike[str],
    host_manifest: HostManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Verify packet hash and that the supplied host is the packet's host."""
    if isinstance(packet, (str, os.PathLike)):
        try:
            with Path(packet).open("r", encoding="utf-8") as stream:
                packet = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionPacketError(f"cannot load execution packet: {exc}") from exc
    if isinstance(packet, PreRerunExecutionPacket):
        parsed = packet
        record = packet.to_record()
    elif isinstance(packet, Mapping):
        expected_fields = {
            "candidate_head", "candidate_context_id", "gate", "host_manifest_id",
            "schema_version", "scope", "packet_sha256",
        }
        if set(packet) != expected_fields:
            raise ExecutionPacketError("execution packet has unexpected or missing fields")
        try:
            parsed = PreRerunExecutionPacket(
                candidate_head=str(packet["candidate_head"]),
                host_manifest_id=str(packet["host_manifest_id"]),
                candidate_context_id=str(packet["candidate_context_id"]),
                schema_version=str(packet["schema_version"]),
                gate=str(packet["gate"]),
                scope=str(packet["scope"]),
            )
        except (ExecutionPacketError, TypeError, ValueError) as exc:
            raise ExecutionPacketError(str(exc)) from exc
        record = dict(packet)
        expected_packet_hash = hashlib.sha256(_canonical_json(parsed.to_record(include_hash=False))).hexdigest()
        if record["packet_sha256"] != expected_packet_hash:
            raise ExecutionPacketError("execution packet hash mismatch")
    else:
        raise ExecutionPacketError("packet must be a packet, mapping, or JSON path")
    supplied_id = _manifest_id(host_manifest)
    if supplied_id != parsed.host_manifest_id:
        raise ExecutionPacketError("execution packet host manifest does not match supplied context")
    return {
        "valid": True,
        "candidate_head": parsed.candidate_head,
        "host_manifest_id": parsed.host_manifest_id,
        "candidate_context_id": parsed.candidate_context_id,
        "packet_sha256": parsed.packet_sha256,
        "gate": parsed.gate,
        "scope": parsed.scope,
    }


def write_execution_packet(path: str | os.PathLike[str], packet: PreRerunExecutionPacket) -> None:
    """Publish one packet atomically; callers must not mutate it in place."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(packet.to_record(), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


__all__ = [
    "PACKET_SCHEMA_VERSION", "PACKET_GATE", "ExecutionPacketError",
    "PreRerunExecutionPacket", "context_identity", "build_execution_packet",
    "validate_execution_packet", "write_execution_packet",
]
