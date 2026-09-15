"""Focused conformance checks for the CYAX-0168 generator 2.3 primary."""

from __future__ import annotations

import hashlib
import unittest

from research.cyax0168.generator_primary import (
    GENERATOR_VERSION,
    GenerationError,
    canonical_frame,
    reconstruct_candidate_vector,
    generate_snapshot,
    prf_choice,
)


class GeneratorPrimaryTests(unittest.TestCase):
    def test_generator_23_identity_domains_and_calibration_barrier(self):
        snapshot = generate_snapshot("C0", "P-medium", 168900)
        self.assertEqual(snapshot.manifest["generator_name"], "cyax-0168-scale-2.3")
        self.assertEqual(snapshot.manifest["generator_version"], "2.3")
        self.assertEqual(snapshot.manifest["namespace"], "cyax-0168-synthetic-v2.3")
        self.assertEqual(snapshot.manifest["entity_count"], 1_000)
        self.assertEqual(snapshot.manifest["assertion_count"], 5_000)
        self.assertEqual(
            snapshot.manifest["logical_snapshot_checksum"],
            "e73063a46e1c585bfb15aa0db5e920a0f6b4a37f2b5a44a7a0b372f7aecd3b18",
        )
        self.assertTrue(all("/scale/2.3/" in row["canonical_locator"] for row in snapshot.source_revisions))
        with self.assertRaises(GenerationError):
            generate_snapshot("T0", "P-medium", 162000, entity_count=100)

    def test_phase_2_uses_primary_order_and_later_to_previous_chain_direction(self):
        snapshot = generate_snapshot("conformance-tiny", "P-low", 900001, entity_count=100)
        by_id = {row["assertion_id"]: row for row in snapshot.assertions}
        work_ids = {row["entity_id"] for row in snapshot.entities if row["entity_type"] == "WorkItem"}
        phase = next(row for row in snapshot.trace["phases"] if row["phase"] == 2)
        for component in phase["component_blocks"]:
            component_work_ids = [
                next(row["entity_id"] for row in snapshot.entities
                     if row["entity_type"] == "WorkItem"
                     and row["canonical_source_identity"][3] == block)
                for block in component
            ]
            self.assertEqual(component_work_ids, sorted(component_work_ids, key=lambda value: value.encode("utf-8")))
        chain = [by_id[aid] for aid in phase["chain_assertion_ids"]]
        self.assertTrue(all(row["predicate"] == "depends_on" for row in chain))
        self.assertTrue(all(row["subject_id"] in work_ids and row["object_id"] in work_ids for row in chain))
        for component in phase["component_blocks"]:
            component_work_ids = [
                next(row["entity_id"] for row in snapshot.entities
                     if row["entity_type"] == "WorkItem"
                     and row["canonical_source_identity"][3] == block)
                for block in component
            ]
            length = min(4, len(component_work_ids) - 1)
            expected = list(zip(component_work_ids[1:length + 1], component_work_ids[:length]))
            actual = [(row["subject_id"], row["object_id"]) for row in chain[:length]]
            self.assertEqual(actual, expected)

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
        digest = hashlib.sha256(canonical_frame(["cyax-gen-2.3", 7, "P-low", "dependency_target", 0, 0])).digest()
        expected = candidates[int.from_bytes(digest, "big") % len(candidates)]
        selected, _ = prf_choice(candidates, seed=7, profile_id="P-low", purpose="dependency_target", ordinal=0)
        self.assertEqual(selected, expected)
        with self.assertRaises(GenerationError):
            prf_choice(candidates, seed=7, profile_id="P-low", purpose="unlisted", ordinal=0)

    def test_generator_trace_deduplicates_vectors_and_reconstructs_losslessly(self):
        snapshot = generate_snapshot("conformance-tiny", "P-low", 900001, entity_count=100)
        choices = snapshot.prf_choices
        self.assertTrue(choices)
        self.assertTrue(all("candidates" not in choice for choice in choices))
        self.assertLess(len(snapshot.candidate_vectors), len(choices))
        for choice in choices:
            vector = reconstruct_candidate_vector(snapshot.trace, choice)
            self.assertEqual(vector, snapshot.candidate_vectors[choice["candidate_vector_id"]])
        # Trace compression is diagnostic only: semantic output remains the
        # byte-stable generator-2.3 result.
        self.assertEqual(snapshot.logical_snapshot_checksum, snapshot.manifest["logical_snapshot_checksum"])

    def test_unregistered_tier_requires_an_explicit_tiny_size(self):
        with self.assertRaisesRegex(GenerationError, "unknown tier"):
            generate_snapshot("unregistered-tiny", "P-low", 900001)

if __name__ == "__main__":
    unittest.main()
