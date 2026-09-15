from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from .execution_packet import (
        ExecutionPacketError,
        build_execution_packet,
        validate_execution_packet,
        write_execution_packet,
    )
    from .host import load_frozen_host_manifest
    from .test_host import manifest
except ImportError:  # pragma: no cover
    from execution_packet import (
        ExecutionPacketError,
        build_execution_packet,
        validate_execution_packet,
        write_execution_packet,
    )
    from host import load_frozen_host_manifest
    from test_host import manifest


class ExecutionPacketTests(unittest.TestCase):
    def test_versioned_evidence_pair_is_self_consistent(self):
        root = Path(__file__).resolve().parent / "evidence"
        host = load_frozen_host_manifest(root / "g1-pre-rerun-host-manifest-v2.json")
        packet = root / "g1-pre-rerun-execution-packet-v1.json"
        self.assertEqual(
            validate_execution_packet(packet, host)["host_manifest_id"], host.identity
        )

    def test_packet_binds_exact_candidate_and_one_host_hash(self):
        host = manifest()
        head = "cd5112ed6afb41937c42ffa9354b5ecc554f4d31"
        packet = build_execution_packet(head, host)
        result = validate_execution_packet(packet.to_record(), host)
        self.assertTrue(result["valid"])
        self.assertEqual(result["candidate_head"], head)
        self.assertEqual(result["host_manifest_id"], host.identity)
        self.assertEqual(len(packet.packet_sha256), 64)

    def test_packet_rejects_mixed_or_tampered_context_and_writes_atomically(self):
        host = manifest()
        other = manifest(ordinary_available_volume_capacity_bytes=400_000)
        packet = build_execution_packet("cd5112ed6afb41937c42ffa9354b5ecc554f4d31", host)
        with self.assertRaises(ExecutionPacketError):
            validate_execution_packet(packet.to_record(), other)
        tampered = dict(packet.to_record(), candidate_head="0" * 40)
        with self.assertRaises(ExecutionPacketError):
            validate_execution_packet(tampered, host)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.json"
            write_execution_packet(path, packet)
            self.assertTrue(validate_execution_packet(path, host)["valid"])


if __name__ == "__main__":
    unittest.main()
