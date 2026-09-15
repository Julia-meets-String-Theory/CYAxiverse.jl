"""Focused conformance checks for the CYAX-0168 generator 2.2 primary."""

from __future__ import annotations

import hashlib
import unittest

from research.cyax0168.generator_primary import (
    GENERATOR_VERSION,
    GenerationError,
    canonical_frame,
    generate_snapshot,
    prf_choice,
)


class GeneratorPrimaryTests(unittest.TestCase):
    def test_small_conformance_snapshot_is_repeatable_and_exactly_50_per_block(self):
        first = generate_snapshot("conformance-tiny", "P-low", 900001, entity_count=100)
        second = generate_snapshot("conformance-tiny", "P-low", 900001, entity_count=100)

        self.assertEqual(first.manifest["generator_version"], GENERATOR_VERSION)
        self.assertEqual(len(first.entities), 100)
        self.assertEqual(len(first.assertions), 500)
        self.assertEqual(first.manifest["logical_snapshot_checksum"], second.manifest["logical_snapshot_checksum"])
        self.assertEqual(first.assertions, second.assertions)
        self.assertEqual(first.source_revisions, second.source_revisions)

    def test_claim_registry_and_source_direct_statement_bytes_are_complete(self):
        snapshot = generate_snapshot("conformance-tiny", "P-low", 900001, entity_count=100)
        self.assertEqual(snapshot.manifest["claim_key_registry"], [{
            "claim_key": "synthetic.block_claim",
            "literal_type": "text",
            "semantic_slot": "synthetic_fixture_block_statement",
        }])
        self.assertTrue(all(entity.get("claim_key") == "synthetic.block_claim" for entity in snapshot.entities if entity["entity_type"] == "Claim"))

        revisions = {row["source_revision_id"]: row for row in snapshot.source_revisions}
        for assertion in snapshot.assertions:
            revision = revisions[assertion["source_revision_id"]]
            payload = snapshot.source_objects[revision["object_sha256"]]
            self.assertTrue(payload.endswith(b"\n"))
            self.assertEqual(assertion["origin"], "source_direct")
            self.assertEqual(assertion["source_locator"], revision["canonical_locator"])
            self.assertIsNone(assertion["source_event_at"])

    def test_prf_rejects_self_or_duplicate_without_removing_candidates(self):
        candidates = ["a", "b", "c"]
        seen = []
        chosen, counter = prf_choice(
            candidates,
            seed=11,
            profile_id="P-low",
            purpose="dependency_target",
            ordinal=4,
            reject=lambda value: seen.append(value) or value != "c",
        )
        self.assertEqual(chosen, "c")
        self.assertGreaterEqual(counter, 0)
        self.assertTrue(sorted(seen))  # candidates remain available across retries

    def test_prf_uses_big_endian_digest_and_closed_purpose_registry(self):
        candidates = ["a", "b", "c", "d"]
        digest = hashlib.sha256(canonical_frame(["cyax-gen-2.2", 7, "P-low", "dependency_target", 0, 0])).digest()
        expected = candidates[int.from_bytes(digest, "big") % len(candidates)]
        selected, _ = prf_choice(candidates, seed=7, profile_id="P-low", purpose="dependency_target", ordinal=0)
        self.assertEqual(selected, expected)
        with self.assertRaises(GenerationError):
            prf_choice(candidates, seed=7, profile_id="P-low", purpose="unlisted", ordinal=0)

    def test_unregistered_tier_requires_an_explicit_tiny_size(self):
        with self.assertRaisesRegex(GenerationError, "unknown tier"):
            generate_snapshot("unregistered-tiny", "P-low", 900001)

if __name__ == "__main__":
    unittest.main()
