#!/usr/bin/env python3
"""SQLite materialization for the CYAX-0168 frozen snapshot contract.

This module is deliberately small and boring: it is the standard-library
SQLite implementation of the already validated canonical records in
``common.py``/``snapshot.py``.  It does not know how fixtures are generated
and it has no path to a generator or to a benchmark backend.  A materialized
directory contains only ``manifest.json`` and a SQLite database; the database
stores logical records, not source text or mutable semantic state.

The public object owns exactly one connection.  Query methods are tied to the
thread which opened that connection, which makes the one-process/one-
connection/one-query-worker topology an executable invariant rather than a
benchmark convention.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Same-directory test/import convention and package imports.
    import common as c
    import publication
    import snapshot as s
except ImportError:  # pragma: no cover - exercised by package importers
    from . import common as c
    from . import publication
    from . import snapshot as s


MATERIALIZATION_SCHEMA_VERSION = "cyax-sqlite-materialization-v1"
DATABASE_NAME = "materialization.db"
MANIFEST_NAME = "manifest.json"
MATERIALIZATION_FILES = frozenset({DATABASE_NAME, MANIFEST_NAME})


class SQLiteBackendError(s.SnapshotError):
    """Raised when a SQLite materialization is malformed or unsafe to open."""


class SQLiteWorkerError(SQLiteBackendError):
    """Raised when a connection is used by a second calling/query worker."""


def _json_text(value: Any) -> str:
    """Encode a nested value with the snapshot canonical JSON rules."""

    # canonical_json_line accepts mappings.  The wrapper preserves scalar and
    # null values without adding an application-visible field to the encoding.
    return s.canonical_json_line({"v": value}).decode("utf-8").rstrip("\n")


def _json_value(value: str) -> Any:
    try:
        record = json.loads(value)
        if not isinstance(record, dict) or set(record) != {"v"}:
            raise ValueError("wrapped JSON value must have only key 'v'")
        return record["v"]
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SQLiteBackendError("malformed canonical JSON value in SQLite row") from exc


def _sql_literals(values: Iterable[str]) -> str:
    # The values are closed constants from common.py, never caller input.
    return ",".join("'" + value.replace("'", "''") + "'" for value in sorted(values))


def _manifest_record(manifest: s.SnapshotManifest, db_bytes: bytes, *, complete: bool) -> dict[str, Any]:
    return {
        "backend": "sqlite",
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "snapshot_id": manifest.snapshot_id,
        "build_contract_checksum": manifest.build_contract_checksum,
        "physical_payload_checksum": manifest.physical_payload_checksum,
        "db_byte_count": len(db_bytes),
        "db_sha256": c.sha256_hex(db_bytes),
        "sqlite_version": sqlite3.sqlite_version,
        "build_complete": complete,
        "built_at": manifest.built_at,
    }


def _canonical_manifest_bytes(record: Mapping[str, Any]) -> bytes:
    return s.canonical_json_line(record)


def _schema_sql() -> str:
    entities = _sql_literals(c.ENTITY_TYPES)
    literals = _sql_literals(c.LITERAL_TYPES)
    source_kinds = _sql_literals(c.SOURCE_KINDS)
    authority = _sql_literals(c.AUTHORITY_CLASSES)
    rules = _sql_literals(c.AUTHORITY_DERIVATION_RULE_IDS)
    predicates = _sql_literals(c.PREDICATES)
    origins = _sql_literals(c.ORIGINS)
    curation = _sql_literals(c.CURATION_STATES)
    review = _sql_literals(c.REVIEW_STATES)
    epistemic = _sql_literals(c.EPISTEMIC_STATES)
    dispute = _sql_literals(c.DISPUTE_STATES)
    validity = _sql_literals(c.VALIDITY_BASES)
    return f"""
    PRAGMA foreign_keys = ON;
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY NOT NULL CHECK (key <> ''),
        value TEXT NOT NULL
    );
    CREATE TABLE literals (
        literal_id TEXT PRIMARY KEY NOT NULL,
        literal_type TEXT NOT NULL CHECK (literal_type IN ({literals})),
        value_json TEXT NOT NULL
    );
    CREATE TABLE entities (
        entity_id TEXT PRIMARY KEY NOT NULL,
        entity_type TEXT NOT NULL CHECK (entity_type IN ({entities})),
        namespace TEXT NOT NULL CHECK (namespace <> ''),
        canonical_source_identity_json TEXT NOT NULL,
        display_label_ref TEXT REFERENCES literals(literal_id),
        claim_key TEXT,
        CHECK ((entity_type = 'Claim' AND claim_key IS NOT NULL)
               OR (entity_type <> 'Claim' AND claim_key IS NULL))
    );
    CREATE TABLE source_revisions (
        source_revision_id TEXT PRIMARY KEY NOT NULL,
        source_kind TEXT NOT NULL CHECK (source_kind IN ({source_kinds})),
        source_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
        canonical_locator TEXT NOT NULL CHECK (canonical_locator <> ''),
        object_sha256 TEXT NOT NULL,
        object_byte_count INTEGER NOT NULL CHECK (object_byte_count >= 0),
        source_event_at TEXT,
        observed_at TEXT NOT NULL,
        actor_id TEXT,
        authority_class TEXT NOT NULL CHECK (authority_class IN ({authority})),
        authority_derivation_rule_id TEXT NOT NULL CHECK (authority_derivation_rule_id IN ({rules}))
    );
    CREATE TABLE assertions (
        assertion_id TEXT PRIMARY KEY NOT NULL,
        subject_id TEXT NOT NULL REFERENCES entities(entity_id),
        predicate TEXT NOT NULL CHECK (predicate IN ({predicates})),
        object_id TEXT REFERENCES entities(entity_id),
        literal_ref TEXT REFERENCES literals(literal_id),
        source_revision_id TEXT NOT NULL REFERENCES source_revisions(source_revision_id),
        source_locator TEXT NOT NULL CHECK (source_locator <> ''),
        source_event_at TEXT,
        asserted_at TEXT NOT NULL,
        valid_from TEXT,
        valid_to TEXT,
        validity_basis TEXT NOT NULL CHECK (validity_basis IN ({validity})),
        authority_class TEXT NOT NULL CHECK (authority_class IN ({authority})),
        authority_derivation_rule_id TEXT NOT NULL CHECK (authority_derivation_rule_id IN ({rules})),
        origin TEXT NOT NULL CHECK (origin IN ({origins})),
        curation_state TEXT NOT NULL CHECK (curation_state IN ({curation})),
        review_state TEXT NOT NULL CHECK (review_state IN ({review})),
        epistemic_state TEXT NOT NULL CHECK (epistemic_state IN ({epistemic})),
        dispute_state TEXT NOT NULL CHECK (dispute_state IN ({dispute})),
        CHECK ((object_id IS NULL) <> (literal_ref IS NULL))
    );
    CREATE INDEX idx_assertions_subject_predicate ON assertions(subject_id, predicate, assertion_id);
    CREATE INDEX idx_assertions_object_predicate ON assertions(object_id, predicate, assertion_id);
    CREATE INDEX idx_assertions_predicate_subject ON assertions(predicate, subject_id, assertion_id);
    CREATE INDEX idx_assertions_predicate_object ON assertions(predicate, object_id, assertion_id);
    CREATE INDEX idx_assertions_source_revision ON assertions(source_revision_id, assertion_id);
    """


def _row_assertion(row: sqlite3.Row) -> c.Assertion:
    fields = {name: row[name] for name in c._ASSERTION_ID_FIELDS}
    return c.Assertion(assertion_id=row["assertion_id"], **fields)


def _build_database_bytes(bundle: s.SnapshotBundle) -> bytes:
    """Build a deterministic SQLite database for one complete bundle."""

    if not bundle.manifest.build_complete:
        raise SQLiteBackendError("SQLite publication requires a complete snapshot bundle")
    registry = c.validate_claim_key_registry(bundle.manifest.claim_key_registry) if bundle.manifest.claim_key_registry else None
    try:
        s.validate_snapshot_records(
            bundle.entities,
            bundle.literals,
            bundle.source_revisions,
            bundle.assertions,
            claim_key_registry=registry,
        )
    except Exception as exc:
        raise SQLiteBackendError(f"bundle failed SQLite materialization validation: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="cyax-sqlite-build-") as temp:
        db_path = Path(temp) / DATABASE_NAME
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA page_size=4096")
            connection.execute("PRAGMA auto_vacuum=NONE")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(_schema_sql())
            manifest_json = _canonical_manifest_bytes(s.manifest_to_json(bundle.manifest)).decode("utf-8").rstrip("\n")
            metadata = {
                "backend": "sqlite",
                "schema_version": MATERIALIZATION_SCHEMA_VERSION,
                "snapshot_manifest_json": manifest_json,
                "snapshot_id": bundle.manifest.snapshot_id,
                "build_contract_checksum": bundle.manifest.build_contract_checksum,
                "physical_payload_checksum": bundle.manifest.physical_payload_checksum,
            }
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
            connection.executemany(
                """INSERT INTO literals(literal_id, literal_type, value_json)
                   VALUES (?, ?, ?)""",
                ((x.literal_id, x.literal_type, _json_text(x.value)) for x in s.sort_literals(bundle.literals)),
            )
            connection.executemany(
                """INSERT INTO entities(entity_id, entity_type, namespace,
                         canonical_source_identity_json, display_label_ref, claim_key)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    (
                        x.entity_id,
                        x.entity_type,
                        x.namespace,
                        _json_text(x.canonical_source_identity),
                        x.display_label_ref,
                        x.claim_key,
                    )
                    for x in s.sort_entities(bundle.entities)
                ),
            )
            connection.executemany(
                """INSERT INTO source_revisions(
                    source_revision_id, source_kind, source_entity_id,
                    canonical_locator, object_sha256, object_byte_count,
                    source_event_at, observed_at, actor_id, authority_class,
                    authority_derivation_rule_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    (
                        x.source_revision_id,
                        x.source_kind,
                        x.source_entity_id,
                        x.canonical_locator,
                        x.object_sha256,
                        x.object_byte_count,
                        x.source_event_at,
                        x.observed_at,
                        x.actor_id,
                        x.authority_class,
                        x.authority_derivation_rule_id,
                    )
                    for x in s.sort_source_revisions(bundle.source_revisions)
                ),
            )
            connection.executemany(
                """INSERT INTO assertions(
                    assertion_id, subject_id, predicate, object_id, literal_ref,
                    source_revision_id, source_locator, source_event_at,
                    asserted_at, valid_from, valid_to, validity_basis,
                    authority_class, authority_derivation_rule_id, origin,
                    curation_state, review_state, epistemic_state, dispute_state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tuple(getattr(x, name) for name in ("assertion_id",) + c._ASSERTION_ID_FIELDS)
                    for x in s.sort_assertions(bundle.assertions)
                ),
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("VACUUM")
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise SQLiteBackendError(f"failed to build SQLite materialization: {exc}") from exc
        finally:
            connection.close()
        for sidecar in (db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
            if sidecar.exists():
                raise SQLiteBackendError(f"SQLite build left sidecar {sidecar.name}")
        return db_path.read_bytes()


class SQLiteBackend:
    """One read-only SQLite connection over one published logical snapshot."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], connection: sqlite3.Connection, bundle: s.SnapshotBundle):
        self.root = root
        self._manifest = dict(manifest)
        self._connection = connection
        self._bundle = bundle
        self._worker_ident = threading.get_ident()
        self._closed = False

    @property
    def connection(self) -> sqlite3.Connection:
        self._ensure_worker()
        return self._connection

    @property
    def manifest(self) -> Mapping[str, Any]:
        return dict(self._manifest)

    @property
    def read_only(self) -> bool:
        return True

    @classmethod
    def build(cls, destination: Path, bundle: s.SnapshotBundle, *, hook: publication.PublicationHook | None = None) -> "SQLiteBackend":
        destination = Path(destination)
        db_bytes = _build_database_bytes(bundle)
        false_record = _manifest_record(bundle.manifest, db_bytes, complete=False)
        true_record = _manifest_record(bundle.manifest, db_bytes, complete=True)

        def validate(path: Path) -> object:
            staged = path / DATABASE_NAME
            if not staged.is_file() or staged.stat().st_size != len(db_bytes):
                raise SQLiteBackendError("staged SQLite database byte count mismatch")
            if c.sha256_hex(staged.read_bytes()) != c.sha256_hex(db_bytes):
                raise SQLiteBackendError("staged SQLite database checksum mismatch")
            SQLiteBackend._validate_database_file(staged, bundle.manifest)
            return True

        # Let publication hooks and publication failures retain their original
        # exception type.  In particular, a crash hook is an intentional
        # interruption point for recovery tests, and callers must be able to
        # distinguish it from a malformed completed materialization.
        publication._stage_directory(
            destination,
            {DATABASE_NAME: db_bytes},
            false_record,
            true_record,
            manifest_name=MANIFEST_NAME,
            validate=validate,
            hook=hook,
        )
        try:
            return cls.open(destination, read_only=True)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    @classmethod
    def open(cls, root: Path, *, read_only: bool = True) -> "SQLiteBackend":
        if not read_only:
            raise SQLiteBackendError("SQLite materializations support only non-mutating read-only reopen")
        root = Path(root).resolve()
        if not root.is_dir():
            raise SQLiteBackendError(f"SQLite materialization directory is missing: {root}")
        if root.name.startswith(".") and ".tmp-" in root.name:
            raise SQLiteBackendError("SQLite temporary publication candidates are never openable")
        actual_files = {str(x.relative_to(root)) for x in root.rglob("*") if x.is_file()}
        if actual_files != MATERIALIZATION_FILES:
            raise SQLiteBackendError("SQLite materialization has unknown, partial, or residual files")
        try:
            raw_manifest = (root / MANIFEST_NAME).read_bytes()
            record = json.loads(raw_manifest.decode("utf-8"))
            if raw_manifest != _canonical_manifest_bytes(record):
                raise SQLiteBackendError("SQLite materialization manifest is not canonical JSON")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SQLiteBackendError("invalid SQLite materialization manifest") from exc
        expected = {"backend", "schema_version", "snapshot_id", "build_contract_checksum", "physical_payload_checksum", "db_byte_count", "db_sha256", "sqlite_version", "build_complete", "built_at"}
        if not isinstance(record, dict) or set(record) != expected:
            raise SQLiteBackendError("SQLite materialization manifest has unknown or missing fields")
        if record["backend"] != "sqlite" or record["schema_version"] != MATERIALIZATION_SCHEMA_VERSION or record["build_complete"] is not True:
            raise SQLiteBackendError("SQLite materialization is not a complete v1 publication")
        db_path = root / DATABASE_NAME
        db_bytes = db_path.read_bytes()
        if record["db_byte_count"] != len(db_bytes) or record["db_sha256"] != c.sha256_hex(db_bytes):
            raise SQLiteBackendError("SQLite materialization database checksum mismatch")
        uri = "file:" + urllib.parse.quote(str(db_path), safe="/") + "?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(uri, uri=True, check_same_thread=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
                raise SQLiteBackendError("SQLite read-only query_only mode was not enabled")
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise SQLiteBackendError("SQLite foreign-key enforcement is disabled")
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if tables != {"metadata", "literals", "entities", "source_revisions", "assertions"}:
                raise SQLiteBackendError("SQLite normalized schema is incomplete or contains unknown tables")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise SQLiteBackendError("SQLite integrity_check failed")
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
            if metadata.get("snapshot_id") != record["snapshot_id"] or metadata.get("build_contract_checksum") != record["build_contract_checksum"]:
                raise SQLiteBackendError("SQLite metadata does not match materialization manifest")
            bundle = cls._read_bundle(connection, record)
        except SQLiteBackendError:
            raise
        except sqlite3.Error as exc:
            raise SQLiteBackendError(f"could not reopen SQLite materialization read-only: {exc}") from exc
        return cls(root, record, connection, bundle)

    @staticmethod
    def _validate_database_file(path: Path, expected_snapshot: s.SnapshotManifest | None, *, expected_manifest_record: Mapping[str, Any] | None = None) -> None:
        try:
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise SQLiteBackendError("SQLite foreign-key enforcement is disabled")
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
            if metadata.get("backend") != "sqlite" or metadata.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
                raise SQLiteBackendError("SQLite metadata build contract is missing")
            if expected_manifest_record is not None:
                if metadata.get("snapshot_id") != expected_manifest_record["snapshot_id"] or metadata.get("build_contract_checksum") != expected_manifest_record["build_contract_checksum"]:
                    raise SQLiteBackendError("SQLite metadata does not match materialization manifest")
            if expected_snapshot is not None and metadata.get("snapshot_id") != expected_snapshot.snapshot_id:
                raise SQLiteBackendError("SQLite database snapshot ID mismatch")
        except sqlite3.Error as exc:
            raise SQLiteBackendError(f"SQLite database is unreadable or incomplete: {exc}") from exc
        finally:
            try:
                connection.close()
            except (UnboundLocalError, AttributeError):
                pass

    @staticmethod
    def _read_bundle(connection: sqlite3.Connection, materialization_record: Mapping[str, Any]) -> s.SnapshotBundle:
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        if metadata.get("snapshot_id") != materialization_record["snapshot_id"] or metadata.get("build_contract_checksum") != materialization_record["build_contract_checksum"]:
            raise SQLiteBackendError("SQLite metadata disagrees with materialization manifest")
        try:
            manifest_record = json.loads(metadata["snapshot_manifest_json"])
            manifest = s.manifest_from_json(manifest_record)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SQLiteBackendError("SQLite snapshot manifest metadata is malformed") from exc
        if not manifest.build_complete or manifest.snapshot_id != materialization_record["snapshot_id"]:
            raise SQLiteBackendError("SQLite snapshot manifest is not complete or has wrong identity")
        literals = tuple(
            c.Literal(row["literal_id"], row["literal_type"], _json_value(row["value_json"]))
            for row in connection.execute("SELECT literal_id, literal_type, value_json FROM literals ORDER BY literal_id")
        )
        entities = tuple(
            c.Entity(row["entity_id"], row["entity_type"], row["namespace"], _json_value(row["canonical_source_identity_json"]), row["display_label_ref"], row["claim_key"])
            for row in connection.execute("SELECT entity_id, entity_type, namespace, canonical_source_identity_json, display_label_ref, claim_key FROM entities ORDER BY entity_id")
        )
        revisions = tuple(
            c.SourceRevision(*(row[name] for name in ("source_revision_id", "source_kind", "source_entity_id", "canonical_locator", "object_sha256", "object_byte_count", "source_event_at", "observed_at", "actor_id", "authority_class", "authority_derivation_rule_id")))
            for row in connection.execute("SELECT source_revision_id, source_kind, source_entity_id, canonical_locator, object_sha256, object_byte_count, source_event_at, observed_at, actor_id, authority_class, authority_derivation_rule_id FROM source_revisions ORDER BY source_revision_id")
        )
        assertions = tuple(_row_assertion(row) for row in connection.execute("SELECT * FROM assertions ORDER BY assertion_id"))
        registry = c.validate_claim_key_registry(manifest.claim_key_registry) if manifest.claim_key_registry else None
        try:
            s.validate_snapshot_records(entities, literals, revisions, assertions, claim_key_registry=registry)
        except Exception as exc:
            raise SQLiteBackendError(f"SQLite logical export failed validation: {exc}") from exc
        export_manifest, payloads = s.build_manifest(
            entities=entities,
            literals=literals,
            source_revisions=revisions,
            assertions=assertions,
            semantic_source_bundle_projection_checksum=manifest.semantic_source_bundle_projection_checksum,
            authority_rule_version=manifest.authority_rule_version,
            semantic_evaluator_rule_version=manifest.semantic_evaluator_rule_version,
            source_bundle_id=manifest.source_bundle_id,
            assertion_compiler_version=manifest.assertion_compiler_version,
            curator_version=manifest.curator_version,
            validator_version=manifest.validator_version,
            context_compiler_version=manifest.context_compiler_version,
            schema_version=manifest.schema_version,
            built_at=manifest.built_at,
            claim_key_registry=manifest.claim_key_registry,
        )
        export_manifest = dataclasses.replace(export_manifest, build_complete=True)
        if export_manifest.snapshot_id != manifest.snapshot_id or export_manifest.physical_payload_checksum != manifest.physical_payload_checksum or export_manifest.build_contract_checksum != manifest.build_contract_checksum:
            raise SQLiteBackendError("SQLite logical export checksum disagrees with frozen snapshot")
        if materialization_record["physical_payload_checksum"] != manifest.physical_payload_checksum:
            raise SQLiteBackendError("SQLite materialization physical checksum disagrees with snapshot")
        return s.SnapshotBundle(export_manifest, entities, literals, revisions, assertions)

    def _ensure_worker(self) -> None:
        if self._closed:
            raise SQLiteBackendError("SQLite backend is closed")
        if threading.get_ident() != self._worker_ident:
            raise SQLiteWorkerError("SQLite materialization permits exactly one calling/query worker")

    def export_bundle(self) -> s.SnapshotBundle:
        self._ensure_worker()
        return self._bundle

    def logical_export(self) -> dict[str, list[dict[str, Any]]]:
        """Return the complete canonical logical export used for parity."""

        self._ensure_worker()
        return {
            "entities": [s.entity_json_record(x) for x in self._bundle.entities],
            "literals": [s.literal_json_record(x) for x in self._bundle.literals],
            "source_revisions": [s.source_revision_json_record(x) for x in self._bundle.source_revisions],
            "assertions": [s.assertion_json_record(x) for x in self._bundle.assertions],
        }

    complete_logical_export = logical_export

    def snapshot_export(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        """Return the same tuple form as :func:`update.snapshot_export`."""

        self._ensure_worker()
        logical = self.logical_export()
        return tuple(tuple(logical[name]) for name in ("entities", "literals", "source_revisions", "assertions"))

    def query_assertions(self, *, subject_id: str | None = None, object_id: str | None = None, predicate: str | None = None) -> tuple[c.Assertion, ...]:
        self._ensure_worker()
        clauses: list[str] = []
        params: list[str] = []
        if subject_id is not None:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if object_id is not None:
            clauses.append("object_id = ?")
            params.append(object_id)
        if predicate is not None:
            if predicate not in c.PREDICATES:
                raise SQLiteBackendError(f"unknown predicate: {predicate!r}")
            clauses.append("predicate = ?")
            params.append(predicate)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return tuple(_row_assertion(row) for row in self._connection.execute("SELECT * FROM assertions" + where + " ORDER BY assertion_id", params))

    def reachable_ids(self, root_id: str, *, predicate: str = "depends_on", direction: str = "forward", max_depth: int = 32) -> tuple[tuple[str, int], ...]:
        """Return deterministic recursive-CTE reachability with cycle guards."""

        self._ensure_worker()
        if predicate not in c.PREDICATES:
            raise SQLiteBackendError(f"unknown predicate: {predicate!r}")
        if direction not in {"forward", "reverse"}:
            raise SQLiteBackendError("direction must be 'forward' or 'reverse'")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise SQLiteBackendError("max_depth must be a nonnegative integer")
        if direction == "forward":
            join = "a.subject_id = reach.node_id"
            next_column = "a.object_id"
        else:
            join = "a.object_id = reach.node_id"
            next_column = "a.subject_id"
        sql = f"""
            WITH RECURSIVE reach(node_id, depth, path) AS (
                SELECT ?, 0, '|' || ? || '|'
                UNION ALL
                SELECT {next_column}, reach.depth + 1,
                       reach.path || {next_column} || '|'
                FROM reach
                JOIN assertions AS a ON {join}
                WHERE a.predicate = ?
                  AND {next_column} IS NOT NULL
                  AND reach.depth < ?
                  AND instr(reach.path, '|' || {next_column} || '|') = 0
            )
            SELECT node_id, MIN(depth) AS depth
            FROM reach
            WHERE depth > 0
            GROUP BY node_id
            ORDER BY depth, node_id
        """
        rows = self._connection.execute(sql, (root_id, root_id, predicate, max_depth)).fetchall()
        return tuple((row["node_id"], int(row["depth"])) for row in rows)

    def close(self) -> None:
        if not self._closed:
            self._ensure_worker()
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "SQLiteBackend":
        self._ensure_worker()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


# Descriptive aliases used by generic materialization tests.
SQLiteMaterialization = SQLiteBackend
open_sqlite = SQLiteBackend.open
build_sqlite = SQLiteBackend.build
