#!/usr/bin/env python3
"""Small, fail-closed Ladybug materialization for CYAX-0168 G1.

This module is intentionally an adapter, not a second semantic evaluator.  It
accepts a tiny synthetic, already-canonical logical snapshot, stores that
snapshot beside a Ladybug graph, and exposes only read-only reopen/query
operations.  The candidate is imported lazily so the backend-independent G1
code remains runnable on hosts without the frozen wheel.

The directory protocol is deliberately simple: ``logical.json`` is the
complete logical export, ``graph.lbdb`` is the query projection, and the
manifest is written with ``build_complete=false`` before the graph is built
and rewritten to ``true`` only after close, hash, and export checks succeed.
Readers verify the exact file set and every recorded hash before opening the
database.  A failed candidate therefore cannot be mistaken for a published
one, and a tampered candidate fails closed.

G1 boundary: this file contains no campaign fixture names, no decision data,
and no backend access to T0--T4 or F-real materializations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "cyax-ladybug-g1-v1"
CANDIDATE_COMMIT = "df58ee387c4e5e9f02bb9d518636b52cd4abe5f7"
CANDIDATE_VERSION = "0.20.4"
EXPECTED_FILES = frozenset({"manifest.json", "logical.json", "graph.lbdb"})


class LadybugBackendError(RuntimeError):
    """Raised for an unsupported, incomplete, or tampered candidate."""


def _canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically and reject floating-point semantics."""

    def reject_floats(item: Any) -> None:
        if isinstance(item, float):
            raise LadybugBackendError("floating-point values are forbidden in the logical export")
        if isinstance(item, Mapping):
            for child in item.values():
                reject_floats(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                reject_floats(child)

    reject_floats(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise LadybugBackendError(f"logical record must be a mapping or dataclass, got {type(value)!r}")


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping) or dataclasses.is_dataclass(value):
        value = [value]
    try:
        return [_record(item) for item in value]
    except TypeError as exc:
        raise LadybugBackendError("logical records must be iterable") from exc


def _field(item: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in item:
            return item[name]
    raise LadybugBackendError(f"logical record is missing one of {names!r}")


def _normalise_snapshot(snapshot: Any) -> dict[str, list[dict[str, Any]]]:
    """Convert a mapping or ``SnapshotBundle``-like object to canonical data."""

    if isinstance(snapshot, Mapping):
        get = snapshot.get
    else:
        get = lambda name, default=None: getattr(snapshot, name, default)
    result = {
        "entities": _records(get("entities", ())),
        "literals": _records(get("literals", ())),
        "source_revisions": _records(get("source_revisions", ())),
        "assertions": _records(get("assertions", ())),
    }
    # The common contract uses primary IDs.  Accepting the short aliases is
    # useful for a tiny smoke fixture, but emit one frozen representation.
    for item in result["entities"]:
        item["entity_id"] = _field(item, "entity_id", "id")
        item["entity_type"] = _field(item, "entity_type", "kind")
        item.pop("id", None)
        item.pop("kind", None)
    for item in result["literals"]:
        item["literal_id"] = _field(item, "literal_id", "id")
        item["literal_type"] = _field(item, "literal_type", "kind")
        item.pop("id", None)
        item.pop("kind", None)
    for item in result["source_revisions"]:
        item["source_revision_id"] = _field(item, "source_revision_id", "id")
        item.pop("id", None)
    for item in result["assertions"]:
        item["assertion_id"] = _field(item, "assertion_id", "id")
        item["subject_id"] = _field(item, "subject_id", "source_id", "from_id")
        item["object_id"] = item.get("object_id", item.get("target_id", item.get("to_id")))
        item["predicate"] = _field(item, "predicate", "kind")
        item.pop("id", None)
        item.pop("source_id", None)
        item.pop("from_id", None)
        item.pop("target_id", None)
        item.pop("to_id", None)
        item.pop("kind", None)
    # Primary-ID byte order is part of the common contract and makes the
    # complete export independent of insertion order.
    for key, id_key in (
        ("entities", "entity_id"),
        ("literals", "literal_id"),
        ("source_revisions", "source_revision_id"),
        ("assertions", "assertion_id"),
    ):
        result[key].sort(key=lambda item: str(item[id_key]).encode("utf-8"))
        identifiers = [item[id_key] for item in result[key]]
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise LadybugBackendError(f"{id_key} values must be non-empty strings")
        if len(identifiers) != len(set(identifiers)):
            raise LadybugBackendError(f"duplicate {id_key} in logical input")
    ids = {item["entity_id"] for item in result["entities"]}
    if len(ids) != len(result["entities"]):
        raise LadybugBackendError("duplicate entity_id in logical input")
    for assertion in result["assertions"]:
        if assertion["subject_id"] not in ids:
            raise LadybugBackendError("assertion subject does not resolve to an entity")
        if assertion["object_id"] is not None and assertion["object_id"] not in ids:
            raise LadybugBackendError("assertion object does not resolve to an entity")
    return result


def _require_ladybug() -> Any:
    # Network/runtime extension installation is not part of this adapter.  A
    # normal import is the only allowed acquisition path.
    try:
        import ladybug  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on non-smoke hosts
        raise LadybugBackendError("frozen ladybug==0.20.4 is not installed") from exc
    if getattr(ladybug, "__version__", None) != CANDIDATE_VERSION:
        raise LadybugBackendError(
            f"unsupported ladybug version: {getattr(ladybug, '__version__', None)!r}"
        )
    return ladybug


class LadybugBackend:
    """Build, validate, reopen, and query one immutable Ladybug candidate."""

    def __init__(self, path: str | Path, *, threads: int = 1) -> None:
        if threads != 1:
            raise LadybugBackendError("G1 freezes one Ladybug query worker (threads=1)")
        self.path = Path(path)
        self.threads = 1
        # Ladybug documents this environment variable for its worker count.
        # Override rather than inherit an accidental parallel configuration.
        os.environ["THREADS"] = "1"
        self._database: Any | None = None
        self._connection: Any | None = None
        self.last_candidate: Path | None = None

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    def build(self, snapshot: Any, *, fail_stage: str | None = None) -> Path:
        """Materialize a tiny logical snapshot using atomic publication.

        ``fail_stage`` is test-only fault injection.  It leaves an explicitly
        incomplete candidate behind so recovery tests can prove that readers
        reject it before any database open.
        """

        if self.path.exists():
            raise LadybugBackendError(f"destination already exists: {self.path}")
        if fail_stage not in (None, "after_graph", "after_export"):
            raise LadybugBackendError(f"unknown failure stage: {fail_stage!r}")
        logical = _normalise_snapshot(snapshot)
        logical_bytes = _canonical_json(logical)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        candidate = Path(tempfile.mkdtemp(prefix=f".{self.path.name}.tmp-", dir=self.path.parent))
        self.last_candidate = candidate
        marker = {
            "schema_version": SCHEMA_VERSION,
            "build_complete": False,
            "candidate": "ladybug",
            "ladybug_version": CANDIDATE_VERSION,
            "upstream_commit": CANDIDATE_COMMIT,
            "threads": 1,
        }
        try:
            (candidate / "manifest.json").write_bytes(_canonical_json(marker))
            ladybug = _require_ladybug()
            database = ladybug.Database(
                candidate / "graph.lbdb", max_num_threads=1, read_only=False
            )
            connection = ladybug.Connection(database, num_threads=1)
            self._create_schema(connection)
            self._insert_logical(connection, logical)
            connection.close()
            database.close()
            if fail_stage == "after_graph":
                raise LadybugBackendError("injected failure after graph close")
            (candidate / "logical.json").write_bytes(logical_bytes)
            if fail_stage == "after_export":
                raise LadybugBackendError("injected failure after logical export")
            complete = dict(marker)
            complete.update(
                {
                    "build_complete": True,
                    "logical_bytes": len(logical_bytes),
                    "logical_sha256": hashlib.sha256(logical_bytes).hexdigest(),
                    "files": {
                        name: {"bytes": (candidate / name).stat().st_size, "sha256": _sha256(candidate / name)}
                        for name in ("logical.json", "graph.lbdb")
                    },
                }
            )
            (candidate / "manifest.json").write_bytes(_canonical_json(complete))
            self._fsync_files(candidate)
            self._validate_directory(candidate)
            os.replace(candidate, self.path)
            self.last_candidate = self.path
            return self.path
        except BaseException:
            self.close()
            # Keep the incomplete tree available for explicit rejection and
            # recovery tests.  It is never treated as a published root.
            raise

    @staticmethod
    def _create_schema(connection: Any) -> None:
        connection.execute(
            "CREATE NODE TABLE Entity(id STRING, entity_type STRING, PRIMARY KEY(id))"
        )
        connection.execute(
            "CREATE NODE TABLE Literal(id STRING, literal_type STRING, value_json STRING, PRIMARY KEY(id))"
        )
        connection.execute(
            "CREATE NODE TABLE SourceRevision(id STRING, source_kind STRING, PRIMARY KEY(id))"
        )
        connection.execute(
            "CREATE REL TABLE Assertion(FROM Entity TO Entity, assertion_id STRING, predicate STRING)"
        )

    @staticmethod
    def _insert_logical(connection: Any, logical: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        for entity in logical["entities"]:
            connection.execute(
                "CREATE (e:Entity {id: $id, entity_type: $entity_type})",
                {"id": entity["entity_id"], "entity_type": entity["entity_type"]},
            )
        for literal in logical["literals"]:
            value = _canonical_json(literal.get("value"))[:-1].decode("utf-8")
            connection.execute(
                "CREATE (e:Literal {id: $id, literal_type: $literal_type, value_json: $value_json})",
                {
                    "id": literal["literal_id"],
                    "literal_type": literal["literal_type"],
                    "value_json": value,
                },
            )
        for revision in logical["source_revisions"]:
            connection.execute(
                "CREATE (e:SourceRevision {id: $id, source_kind: $source_kind})",
                {
                    "id": revision["source_revision_id"],
                    "source_kind": str(revision.get("source_kind", "")),
                },
            )
        for assertion in logical["assertions"]:
            if assertion["object_id"] is None:
                continue
            connection.execute(
                "MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid}) "
                "CREATE (s)-[:Assertion {assertion_id: $aid, predicate: $predicate}]->(o)",
                {
                    "sid": assertion["subject_id"],
                    "oid": assertion["object_id"],
                    "aid": assertion["assertion_id"],
                    "predicate": assertion["predicate"],
                },
            )

    @staticmethod
    def _fsync_files(path: Path) -> None:
        for child in path.iterdir():
            if child.is_file():
                with child.open("rb") as handle:
                    os.fsync(handle.fileno())
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            # Hash and marker validation remain fail-closed on platforms where
            # directory fsync is unavailable; the smoke host supports it.
            pass

    @staticmethod
    def _validate_directory(path: Path) -> dict[str, Any]:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise LadybugBackendError("candidate has no manifest")
        try:
            raw_manifest = manifest_path.read_bytes()
            manifest = json.loads(raw_manifest.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LadybugBackendError("candidate manifest is invalid") from exc
        if raw_manifest != _canonical_json(manifest):
            raise LadybugBackendError("candidate manifest is not canonical JSON")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("build_complete") is not True
            or manifest.get("candidate") != "ladybug"
            or manifest.get("upstream_commit") != CANDIDATE_COMMIT
        ):
            raise LadybugBackendError("candidate is not marked build_complete")
        actual = {item.name for item in path.iterdir() if item.is_file()}
        if actual != EXPECTED_FILES:
            raise LadybugBackendError(f"candidate file set mismatch: {sorted(actual)}")
        for name in ("logical.json", "graph.lbdb"):
            target = path / name
            expected = manifest.get("files", {}).get(name, {})
            if expected.get("bytes") != target.stat().st_size or expected.get("sha256") != _sha256(target):
                raise LadybugBackendError(f"candidate hash mismatch: {name}")
        logical = path / "logical.json"
        if manifest.get("logical_bytes") != logical.stat().st_size or manifest.get("logical_sha256") != _sha256(logical):
            raise LadybugBackendError("logical export hash mismatch")
        if manifest.get("threads") != 1 or manifest.get("ladybug_version") != CANDIDATE_VERSION:
            raise LadybugBackendError("candidate worker/version contract mismatch")
        try:
            parsed = json.loads(logical.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LadybugBackendError("logical export is invalid JSON") from exc
        if _canonical_json(parsed) != logical.read_bytes():
            raise LadybugBackendError("logical export is not canonical JSON")
        return manifest

    def open(self) -> "LadybugBackend":
        if self.is_open:
            return self
        self._validate_directory(self.path)
        ladybug = _require_ladybug()
        self._database = ladybug.Database(
            self.path / "graph.lbdb", max_num_threads=1, read_only=True
        )
        self._connection = ladybug.Connection(self._database, num_threads=1)
        return self

    def close(self) -> None:
        connection, database = self._connection, self._database
        self._connection = self._database = None
        if connection is not None:
            connection.close()
        if database is not None:
            database.close()

    def __enter__(self) -> "LadybugBackend":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def execute(self, query: str, parameters: Mapping[str, Any] | None = None) -> list[list[Any]]:
        if self._connection is None:
            raise LadybugBackendError("backend is not open")
        result = self._connection.execute(query, dict(parameters or {}))
        return result.get_all()

    def required_traversal(self, source_id: str, *, max_depth: int = 3) -> list[list[Any]]:
        if not isinstance(source_id, str) or not source_id:
            raise LadybugBackendError("source_id must be non-empty")
        if not isinstance(max_depth, int) or not 1 <= max_depth <= 32:
            raise LadybugBackendError("max_depth must be between 1 and 32")
        # The depth is range-checked before interpolation; values remain
        # parameterized.  This is the one smoke query, not a query compiler.
        return self.execute(
            f"MATCH (s:Entity {{id: $source_id}})-[:Assertion*1..{max_depth}]->(e:Entity) "
            "RETURN DISTINCT e.id, e.entity_type ORDER BY e.id",
            {"source_id": source_id},
        )

    def logical_export(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.is_dir():
            raise LadybugBackendError("backend path does not exist")
        self._validate_directory(self.path)
        return json.loads((self.path / "logical.json").read_bytes().decode("utf-8"))


def build_ladybug_graph(path: str | Path, snapshot: Any) -> Path:
    """Convenience wrapper for the frozen one-worker candidate."""

    return LadybugBackend(path).build(snapshot)


__all__ = ["LadybugBackend", "LadybugBackendError", "build_ladybug_graph"]
