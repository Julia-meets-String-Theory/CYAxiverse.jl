"""Conformance tests for the standard-library SQLite materialization.

Only tiny records are created directly from the frozen common models.  This
module never imports a generator and never opens any campaign fixture.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from research.cyax0168 import common as c
from research.cyax0168 import publication, snapshot as s, sqlite_backend as b


T0 = "2000-01-01T00:00:00.000000Z"


def _bundle(*, cycle: bool = False) -> s.SnapshotBundle:
    raw = b"sqlite synthetic conformance source"
    digest = hashlib.sha256(raw).hexdigest()
    source = c.Entity(c.compute_entity_id("sqlite-test", "Source", "source"), "Source", "sqlite-test", "source")
    nodes = tuple(
        c.Entity(c.compute_entity_id("sqlite-test", "WorkItem", name), "WorkItem", "sqlite-test", name)
        for name in ("a", "b", "c")
    )
    revision = c.SourceRevision(
        c.compute_source_revision_id("synthetic_fixture", "sqlite/source", digest, None),
        "synthetic_fixture", source.entity_id, "sqlite/source", digest, len(raw), None, T0,
    )
    assertions: list[c.Assertion] = []

    def add(subject: c.Entity, obj: c.Entity) -> None:
        fields = {
            "subject_id": subject.entity_id,
            "predicate": "depends_on",
            "object_id": obj.entity_id,
            "literal_ref": None,
            "source_revision_id": revision.source_revision_id,
            "source_locator": "sqlite/source",
            "source_event_at": None,
            "asserted_at": T0,
            "valid_from": T0,
            "valid_to": None,
            "validity_basis": "explicit",
            "authority_class": "ordinary_record",
            "authority_derivation_rule_id": "ordinary_record",
            "origin": "source_direct",
            "curation_state": "curator_checked",
            "review_state": "not_required",
            "epistemic_state": "supported",
            "dispute_state": "undisputed",
        }
        assertions.append(c.Assertion(c.compute_assertion_id(fields), **fields))

    add(nodes[0], nodes[1])
    add(nodes[1], nodes[2])
    if cycle:
        add(nodes[2], nodes[0])
    semantic = s.compute_semantic_source_bundle_projection_checksum(
        [revision], [(revision.object_sha256, revision.object_byte_count)], [], []
    )
    manifest, _payloads = s.build_manifest(
        entities=(source, *nodes),
        literals=(),
        source_revisions=(revision,),
        assertions=assertions,
        semantic_source_bundle_projection_checksum=semantic,
        authority_rule_version="cyax-authority-v1",
        semantic_evaluator_rule_version="cyax-evaluator-v1",
        source_bundle_id=None,
        assertion_compiler_version="sqlite-test-compiler",
        curator_version="sqlite-test-curator",
        validator_version="sqlite-test-validator",
        context_compiler_version="sqlite-test-context",
        built_at="2000-01-01T00:00:00.000000Z",
    )
    return s.SnapshotBundle(
        dataclasses.replace(manifest, build_complete=True),
        tuple(sorted((source, *nodes), key=lambda x: x.entity_id)),
        (),
        (revision,),
        tuple(sorted(assertions, key=lambda x: x.assertion_id)),
    )


class SQLiteBackendTests(unittest.TestCase):
    def test_build_deterministic_export_reopen_and_query_only(self):
        bundle = _bundle(cycle=True)
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first"
            second = Path(td) / "second"
            one = b.SQLiteBackend.build(first, bundle)
            self.assertTrue(one.read_only)
            exported = one.export_bundle()
            self.assertEqual(exported.entities, bundle.entities)
            self.assertEqual(exported.assertions, bundle.assertions)
            self.assertEqual(set(one.logical_export()), {"entities", "literals", "source_revisions", "assertions"})
            self.assertEqual(one.snapshot_export()[3], tuple(s.assertion_json_record(x) for x in bundle.assertions))
            # Find the first WorkItem by canonical identity rather than by
            # lexical UUID ordering.
            root = next(x for x in bundle.entities if x.canonical_source_identity == "a")
            self.assertEqual(
                {(entity_id, depth) for entity_id, depth in one.reachable_ids(root.entity_id, max_depth=2)},
                {(next(x for x in bundle.entities if x.canonical_source_identity == "b").entity_id, 1),
                 (next(x for x in bundle.entities if x.canonical_source_identity == "c").entity_id, 2)},
            )
            with self.assertRaises(sqlite3.OperationalError):
                one.connection.execute("CREATE TABLE forbidden(x INTEGER)")
            one.close()
            two = b.SQLiteBackend.build(second, bundle)
            self.assertEqual((first / b.DATABASE_NAME).read_bytes(), (second / b.DATABASE_NAME).read_bytes())
            two.close()
            reopened = b.SQLiteBackend.open(first)
            self.assertEqual(reopened.export_bundle().manifest.snapshot_id, bundle.manifest.snapshot_id)
            self.assertEqual(reopened.query_assertions(predicate="depends_on"), bundle.assertions)
            reopened.close()

    def test_recursive_cte_reverse_depth_and_cycle_guard(self):
        bundle = _bundle(cycle=True)
        with tempfile.TemporaryDirectory() as td:
            backend = b.SQLiteBackend.build(Path(td) / "db", bundle)
            by_name = {x.canonical_source_identity: x for x in bundle.entities if x.entity_type == "WorkItem"}
            self.assertEqual(
                backend.reachable_ids(by_name["c"].entity_id, direction="reverse", max_depth=1),
                ((by_name["b"].entity_id, 1),),
            )
            # The cycle must not emit the root, and the depth bound is frozen.
            self.assertNotIn((by_name["a"].entity_id, 3), backend.reachable_ids(by_name["a"].entity_id, max_depth=3))
            self.assertEqual(backend.reachable_ids(by_name["a"].entity_id, max_depth=0), ())
            backend.close()

    def test_one_calling_worker_and_reject_partial_or_residual_publications(self):
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "db"
            backend = b.SQLiteBackend.build(destination, bundle)
            result: list[BaseException] = []

            def other_worker() -> None:
                try:
                    backend.query_assertions()
                except BaseException as exc:  # assert exact type below
                    result.append(exc)

            worker = threading.Thread(target=other_worker)
            worker.start()
            worker.join()
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], b.SQLiteWorkerError)
            backend.close()

            (destination / (b.DATABASE_NAME + "-wal")).write_bytes(b"residual")
            with self.assertRaises(b.SQLiteBackendError):
                b.SQLiteBackend.open(destination)

    def test_tamper_contract_marker_and_crash_are_rejected(self):
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "db"
            backend = b.SQLiteBackend.build(destination, bundle)
            backend.close()
            db = destination / b.DATABASE_NAME
            original = db.read_bytes()
            db.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(b.SQLiteBackendError):
                b.SQLiteBackend.open(destination)
            db.write_bytes(original)
            manifest_path = destination / b.MANIFEST_NAME
            record = json.loads(manifest_path.read_text())
            record["build_contract_checksum"] = "0" * 64
            manifest_path.write_bytes(s.canonical_json_line(record))
            with self.assertRaises(b.SQLiteBackendError):
                b.SQLiteBackend.open(destination)

            incomplete = Path(td) / "incomplete"
            incomplete.mkdir()
            (incomplete / b.DATABASE_NAME).write_bytes(original)
            with self.assertRaises(b.SQLiteBackendError):
                b.SQLiteBackend.open(incomplete)

            crashed = Path(td) / "crashed"

            def crash(stage: str, _path: Path) -> None:
                if stage == "candidate_validated":
                    raise RuntimeError("synthetic interruption")

            with self.assertRaises(RuntimeError):
                b.SQLiteBackend.build(crashed, bundle, hook=crash)
            self.assertFalse(crashed.exists())
            self.assertEqual(list(Path(td).glob(".crashed.tmp-*")), [])

    def test_schema_has_foreign_keys_checks_and_bidirectional_indexes(self):
        with tempfile.TemporaryDirectory() as td:
            backend = b.SQLiteBackend.build(Path(td) / "db", _bundle())
            tables = {row[0] for row in backend.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"metadata", "entities", "literals", "source_revisions", "assertions"} <= tables)
            indexes = {row[0] for row in backend.connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            self.assertIn("idx_assertions_subject_predicate", indexes)
            self.assertIn("idx_assertions_object_predicate", indexes)
            checks = backend.connection.execute("SELECT sql FROM sqlite_master WHERE name='assertions'").fetchone()[0]
            self.assertIn("object_id IS NULL", checks)
            self.assertEqual(backend.connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            backend.close()


if __name__ == "__main__":
    unittest.main()
