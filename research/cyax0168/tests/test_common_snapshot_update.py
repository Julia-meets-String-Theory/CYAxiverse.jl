"""Focused CYAX-0168 Worker A+B conformance tests.

These tests use only synthetic records and source bytes.  They never open a
backend or materialize a T0--T4 fixture.
"""

from __future__ import annotations

import dataclasses
import hashlib
import tempfile
import unittest
from pathlib import Path

from research.cyax0168 import common as c
from research.cyax0168 import publication, snapshot as s, update


T0 = "2000-01-01T00:00:00.000000Z"
T1 = "2000-01-02T00:00:00.000000Z"
T2 = "2000-01-03T00:00:00.000000Z"


def _records(*, predicate: str = "governs", source_event_at: str | None = None, valid_from: str | None = T0):
    raw = b"synthetic source"
    digest = hashlib.sha256(raw).hexdigest()
    revision = c.SourceRevision(
        source_revision_id=c.compute_source_revision_id("synthetic_fixture", "fixture/source", digest, source_event_at),
        source_kind="synthetic_fixture",
        source_entity_id=c.compute_entity_id("fixture", "Source", "source"),
        canonical_locator="fixture/source",
        object_sha256=digest,
        object_byte_count=len(raw),
        source_event_at=source_event_at,
        observed_at=T0,
    )
    source = c.Entity(revision.source_entity_id, "Source", "fixture", "source")
    work = c.Entity(c.compute_entity_id("fixture", "WorkItem", "work"), "WorkItem", "fixture", "work")
    decision = c.Entity(c.compute_entity_id("fixture", "Decision", "decision"), "Decision", "fixture", "decision")
    fields = dict(
        subject_id=decision.entity_id,
        predicate=predicate,
        object_id=work.entity_id,
        literal_ref=None,
        source_revision_id=revision.source_revision_id,
        source_locator="fixture/source",
        source_event_at=source_event_at,
        asserted_at=source_event_at or T0,
        valid_from=valid_from,
        valid_to=None,
        validity_basis="source_event" if source_event_at else "explicit",
        authority_class="ordinary_record",
        authority_derivation_rule_id="ordinary_record",
        origin="source_direct",
        curation_state="curator_checked",
        review_state="not_required",
        epistemic_state="supported",
        dispute_state="undisputed",
    )
    assertion = c.Assertion(c.compute_assertion_id(fields), **fields)
    return raw, (source, work, decision), revision, assertion


class CommonConformanceTests(unittest.TestCase):
    def test_closed_registry_authority_temporal_and_state_rules(self):
        with self.assertRaises(c.SemanticError):
            c.validate_claim_key_registry([c.ClaimKeyRegistryEntry("z", "text", "slot"), c.ClaimKeyRegistryEntry("a", "text", "slot")])
        revision = _records()[2]
        event = c.OwnerDecisionRegistryEntry(revision.source_revision_id, "fixture/source", "entry-1")
        registry = c.validate_owner_decision_registry([event], [(revision.source_revision_id, "fixture/source")])
        owner = c.SourceEvidence("github_issue_comment", revision.source_revision_id, "fixture/source", actor_is_repository_owner=True)
        self.assertEqual(c.derive_authority_class(owner, registry), ("owner_decision", "owner_decision_registered_event"))
        ordinary = dataclasses.replace(owner, source_locator="other")
        self.assertEqual(c.derive_authority_class(ordinary, registry)[0], "ordinary_record")
        _raw, _entities, _revision, original = _records()
        future = dataclasses.replace(original, valid_from=T1)
        future = dataclasses.replace(future, assertion_id=c.compute_assertion_id(future.semantic_fields()))
        self.assertTrue(c.as_of_holds(future, T1))
        self.assertFalse(c.as_of_holds(future, T0))
        unknown = dataclasses.replace(original, valid_from=None)
        unknown = dataclasses.replace(unknown, assertion_id=c.compute_assertion_id(unknown.semantic_fields()))
        self.assertFalse(c.as_of_holds(unknown, None))
        with self.assertRaises(c.SemanticError):
            c.as_of_holds(_records()[3], "bad-time")
        c.validate_state_transition(c.ALLOWED_EPISTEMIC_TRANSITIONS, "supported", "verified")
        with self.assertRaises(c.SemanticError):
            c.validate_state_transition(c.ALLOWED_EPISTEMIC_TRANSITIONS, "verified", "supported")

    def test_dispute_supersession_and_dependency_staleness_fail_closed(self):
        _raw, (_source, work, decision), revision, assertion = _records()
        self.assertTrue(c.is_admissible(assertion))
        disputed = dataclasses.replace(assertion, dispute_state="disputed")
        disputed = dataclasses.replace(disputed, assertion_id=c.compute_assertion_id(disputed.semantic_fields()))
        self.assertFalse(c.is_admissible(disputed))
        successor_fields = dict(assertion.semantic_fields(), subject_id=work.entity_id, object_id=decision.entity_id, predicate="supersedes", valid_from=T1)
        successor = c.Assertion(c.compute_assertion_id(successor_fields), **successor_fields)
        self.assertIs(c.effective_supersession([successor], T1), successor)
        self.assertEqual(c.dependency_stale_entities({"dep": [work.entity_id], "leaf": ["dep"]}, [work.entity_id]), {"dep", "leaf"})


class SnapshotPublicationUpdateTests(unittest.TestCase):
    def setUp(self):
        raw, entities, revision, assertion = _records()
        self.raw = raw
        self.entities = entities
        self.revision = revision
        self.assertion = assertion
        self.semantic = s.compute_semantic_source_bundle_projection_checksum(
            [revision], [(revision.object_sha256, revision.object_byte_count)], [], []
        )

    def _manifest(self, assertions):
        return s.build_manifest(
            entities=self.entities,
            literals=[],
            source_revisions=[self.revision],
            assertions=assertions,
            semantic_source_bundle_projection_checksum=self.semantic,
            authority_rule_version="cyax-authority-v1",
            semantic_evaluator_rule_version="cyax-evaluator-v1",
            source_bundle_id=None,
            assertion_compiler_version="test-compiler",
            curator_version="test-curator",
            validator_version="test-validator",
            context_compiler_version="test-context",
        )

    def test_semantic_vs_physical_identity_and_tamper(self):
        manifest, payloads = self._manifest([self.assertion])
        complete = dataclasses.replace(manifest, build_complete=True)
        bundle = s.SnapshotBundle(complete, tuple(sorted(self.entities, key=lambda e: e.entity_id)), (), (self.revision,), (self.assertion,))
        self.assertNotEqual(manifest.physical_payload_checksum, "")
        label = dataclasses.replace(self.entities[0], display_label_ref=c.compute_literal_id("text", "diagnostic"))
        label_lit = c.Literal(label.display_label_ref, "text", "diagnostic")
        changed, _ = s.build_manifest(
            entities=[label, *self.entities[1:]], literals=[label_lit], source_revisions=[self.revision], assertions=[self.assertion],
            semantic_source_bundle_projection_checksum=self.semantic, authority_rule_version="cyax-authority-v1", semantic_evaluator_rule_version="cyax-evaluator-v1", source_bundle_id=None,
            assertion_compiler_version="test-compiler", curator_version="test-curator", validator_version="test-validator", context_compiler_version="test-context")
        self.assertEqual(manifest.snapshot_id, changed.snapshot_id)
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "snapshot"
            publication.publish_snapshot(destination, manifest, payloads)
            self.assertEqual(s.read_snapshot_dir(destination).manifest.snapshot_id, manifest.snapshot_id)
            target = destination / "assertions.jsonl"
            target.write_bytes(target.read_bytes().replace(b"governs", b"requires", 1))
            with self.assertRaises(s.SnapshotError):
                s.read_snapshot_dir(destination)

    def test_source_bundle_publish_and_tamper(self):
        manifest, payloads = s.build_source_bundle(source_revisions=[self.revision], objects={self.revision.object_sha256: self.raw}, observation_boundary="synthetic-observation")
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "source-bundle"
            publication.publish_source_bundle(destination, manifest, payloads)
            self.assertEqual(s.read_source_bundle_dir(destination).manifest.source_bundle_id, manifest.source_bundle_id)
            object_path = destination / "objects" / "sha256" / self.revision.object_sha256[:2] / self.revision.object_sha256[2:]
            object_path.write_bytes(b"tampered")
            with self.assertRaises(s.SourceBundleError):
                s.read_source_bundle_dir(destination)

    def test_freshness_is_separate_from_snapshot_consistency(self):
        manifest, _ = self._manifest([self.assertion])
        base = s.SnapshotBundle(
            dataclasses.replace(manifest, build_complete=True),
            tuple(sorted(self.entities, key=lambda e: e.entity_id)),
            (),
            (self.revision,),
            (self.assertion,),
        )

        class Refresher(s.SourceRefresher):
            def __init__(self, current):
                self.current = current

            def current_source_revision_id(self, source_revision):
                return self.current

        self.assertEqual(s.matches_selected_snapshot(base.manifest.snapshot_id, manifest.snapshot_id), True)
        self.assertEqual(s.fresh_against_current_sources(base, Refresher(self.revision.source_revision_id)), "fresh")
        self.assertEqual(s.fresh_against_current_sources(base, Refresher("different")), "stale")
        self.assertEqual(s.fresh_against_current_sources(base, Refresher(None)), "unknown")

    def test_delta_removal_and_no_resurrection(self):
        manifest, _ = self._manifest([self.assertion])
        base = s.SnapshotBundle(dataclasses.replace(manifest, build_complete=True), tuple(sorted(self.entities, key=lambda e: e.entity_id)), (), (self.revision,), (self.assertion,))
        target, _ = self._manifest([])
        delta = update.SnapshotDelta(manifest.snapshot_id, target.snapshot_id, remove_assertion_ids=(self.assertion.assertion_id,))
        result = update.apply_delta(base, delta)
        self.assertEqual(result.manifest.snapshot_id, target.snapshot_id)
        self.assertEqual(base.manifest.snapshot_id, manifest.snapshot_id)
        resurrect = update.SnapshotDelta(manifest.snapshot_id, target.snapshot_id, added_assertions=(self.assertion,), remove_assertion_ids=(self.assertion.assertion_id,))
        with self.assertRaises(update.DeltaError):
            update.apply_delta(base, resurrect)

    def test_crash_hook_leaves_no_openable_candidate(self):
        manifest, payloads = self._manifest([self.assertion])
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "snapshot"
            def crash(stage, _path):
                if stage == "candidate_validated":
                    raise RuntimeError("simulated interruption")
            with self.assertRaises(RuntimeError):
                publication.publish_snapshot(destination, manifest, payloads, hook=crash)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
