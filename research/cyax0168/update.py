#!/usr/bin/env python3
"""Canonical immutable N -> N+1 snapshot deltas for CYAX-0168.

An update is a transformation of a frozen snapshot, never an authority
record.  The validator checks the complete delta before any disposable
materialization is allowed to run.  In particular, an assertion removal is
an exact ID operation; it is not a tombstone, retraction predicate, or a
permission to resurrect an old record.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping, Sequence

try:  # Same-directory test/import convention and package imports.
    import common as c
    import snapshot as s
except ImportError:  # pragma: no cover - exercised by package importers
    from . import common as c
    from . import snapshot as s


class DeltaError(s.SnapshotError):
    """Raised when a canonical delta is malformed or cannot apply to N."""


@dataclasses.dataclass(frozen=True)
class SnapshotDelta:
    base_snapshot_id: str
    expected_target_snapshot_id: str
    added_source_objects: tuple[tuple[str, bytes], ...] = ()
    added_source_revisions: tuple[c.SourceRevision, ...] = ()
    added_entities: tuple[c.Entity, ...] = ()
    added_literals: tuple[c.Literal, ...] = ()
    added_assertions: tuple[c.Assertion, ...] = ()
    remove_assertion_ids: tuple[str, ...] = ()
    # These values are part of the target build contract when an update also
    # changes source-bundle content.  They default to the base manifest for
    # assertion-only deltas.
    target_semantic_source_bundle_projection_checksum: str | None = None
    target_source_bundle_id: str | None = None
    built_at: str | None = None


# Friendly aliases used by callers that describe this as a canonical update.
Delta = SnapshotDelta
CanonicalDelta = SnapshotDelta


def _ordered_ids(records: Sequence[Any], field: str, label: str) -> None:
    values = [getattr(record, field) for record in records]
    if values != sorted(values):
        raise DeltaError(f"{label} must be ordered by {field}")
    if len(values) != len(set(values)):
        raise DeltaError(f"{label} contains a duplicate {field}")


def _validate_added_objects(objects: Sequence[tuple[str, bytes]]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    previous = ""
    for digest, value in objects:
        if not isinstance(digest, str) or not c.SHA256_RE.match(digest):
            raise DeltaError(f"added source object digest is malformed: {digest!r}")
        if digest <= previous:
            raise DeltaError("added_source_objects must be sorted and unique by digest")
        previous = digest
        if not isinstance(value, bytes) or c.sha256_hex(value) != digest:
            raise DeltaError(f"added source object {digest!r} is not content-addressed")
        result[digest] = value
    return result


def validate_delta(base: s.SnapshotBundle, delta: SnapshotDelta) -> None:
    """Validate ``delta`` against immutable snapshot ``base``.

    Validation is intentionally strict: unsorted additions, unknown removal
    IDs, duplicate IDs, dangling references, and collisions fail before a
    candidate target is constructed.
    """

    if not base.manifest.build_complete:
        raise DeltaError("delta base must be a published build_complete snapshot")
    if delta.base_snapshot_id != base.manifest.snapshot_id:
        raise DeltaError(
            f"delta base {delta.base_snapshot_id!r} does not match snapshot N "
            f"{base.manifest.snapshot_id!r}"
        )
    if not isinstance(delta.expected_target_snapshot_id, str) or not delta.expected_target_snapshot_id:
        raise DeltaError("expected_target_snapshot_id must be non-empty")
    if delta.built_at is not None:
        try:
            # built_at is nonsemantic but must still be a canonical timestamp
            # when present in a manifest/delta record.
            c._validate_timestamp_instant(delta.built_at)
        except c.SemanticError as exc:
            raise DeltaError(str(exc)) from exc
    if not delta.remove_assertion_ids:
        remove_ids: tuple[str, ...] = ()
    else:
        remove_ids = delta.remove_assertion_ids
        if tuple(remove_ids) != tuple(sorted(remove_ids)):
            raise DeltaError("remove_assertion_ids must be sorted")
        if len(remove_ids) != len(set(remove_ids)):
            raise DeltaError("remove_assertion_ids must be unique")
    base_entities = {record.entity_id for record in base.entities}
    base_literals = {record.literal_id for record in base.literals}
    base_revisions = {record.source_revision_id for record in base.source_revisions}
    base_assertions = {record.assertion_id for record in base.assertions}
    unknown_removals = set(remove_ids) - base_assertions
    if unknown_removals:
        raise DeltaError(f"delta removes assertion(s) absent from N: {sorted(unknown_removals)}")

    _ordered_ids(delta.added_entities, "entity_id", "added_entities")
    _ordered_ids(delta.added_literals, "literal_id", "added_literals")
    _ordered_ids(delta.added_source_revisions, "source_revision_id", "added_source_revisions")
    _ordered_ids(delta.added_assertions, "assertion_id", "added_assertions")
    object_map = _validate_added_objects(delta.added_source_objects)

    for record in delta.added_entities:
        try:
            record.validated()
        except c.SemanticError as exc:
            raise DeltaError(str(exc)) from exc
        if record.entity_id in base_entities:
            raise DeltaError(f"added entity collides with N: {record.entity_id}")
    for record in delta.added_literals:
        try:
            record.validated()
        except c.SemanticError as exc:
            raise DeltaError(str(exc)) from exc
        if record.literal_id in base_literals:
            raise DeltaError(f"added literal collides with N: {record.literal_id}")
    for record in delta.added_source_revisions:
        try:
            record.validated()
        except c.SemanticError as exc:
            raise DeltaError(str(exc)) from exc
        if record.source_revision_id in base_revisions:
            raise DeltaError(f"added source revision collides with N: {record.source_revision_id}")
        if record.object_sha256 not in object_map:
            raise DeltaError(
                f"added source revision {record.source_revision_id} has no added source object"
            )
        if len(object_map[record.object_sha256]) != record.object_byte_count:
            raise DeltaError(f"added source revision {record.source_revision_id} object byte count mismatch")

    # A removed assertion can never be re-added.  This catches resurrection
    # even when the caller tries to hide it behind a byte-identical record.
    added_assertion_ids = {record.assertion_id for record in delta.added_assertions}
    if added_assertion_ids & set(remove_ids):
        raise DeltaError("delta attempts to resurrect a removed assertion ID")
    if added_assertion_ids & base_assertions:
        raise DeltaError("added assertion collides with an assertion in N")
    for record in delta.added_assertions:
        try:
            record.validated()
        except c.SemanticError as exc:
            raise DeltaError(str(exc)) from exc

    entities = tuple(base.entities) + tuple(delta.added_entities)
    literals = tuple(base.literals) + tuple(delta.added_literals)
    revisions = tuple(base.source_revisions) + tuple(delta.added_source_revisions)
    assertions = tuple(
        record for record in base.assertions if record.assertion_id not in set(remove_ids)
    ) + tuple(delta.added_assertions)
    try:
        registry = c.validate_claim_key_registry(base.manifest.claim_key_registry) if base.manifest.claim_key_registry else None
        s.validate_snapshot_records(entities, literals, revisions, assertions, claim_key_registry=registry)
    except (c.SemanticError, s.SnapshotError) as exc:
        raise DeltaError(f"delta target has invalid closure: {exc}") from exc


def apply_delta(
    base: s.SnapshotBundle,
    delta: SnapshotDelta,
    *,
    target_semantic_source_bundle_projection_checksum: str | None = None,
    target_source_bundle_id: str | None = None,
) -> s.SnapshotBundle:
    """Apply a validated delta and require the frozen target snapshot ID."""

    validate_delta(base, delta)
    removals = set(delta.remove_assertion_ids)
    entities = tuple(base.entities) + tuple(delta.added_entities)
    literals = tuple(base.literals) + tuple(delta.added_literals)
    revisions = tuple(base.source_revisions) + tuple(delta.added_source_revisions)
    assertions = tuple(a for a in base.assertions if a.assertion_id not in removals) + tuple(
        delta.added_assertions
    )
    semantic_source_checksum = (
        target_semantic_source_bundle_projection_checksum
        or delta.target_semantic_source_bundle_projection_checksum
        or base.manifest.semantic_source_bundle_projection_checksum
    )
    source_bundle_id = (
        target_source_bundle_id
        if target_source_bundle_id is not None
        else delta.target_source_bundle_id
        if delta.target_source_bundle_id is not None
        else base.manifest.source_bundle_id
    )
    manifest, _payloads = s.build_manifest(
        entities=entities,
        literals=literals,
        source_revisions=revisions,
        assertions=assertions,
        semantic_source_bundle_projection_checksum=semantic_source_checksum,
        authority_rule_version=base.manifest.authority_rule_version,
        semantic_evaluator_rule_version=base.manifest.semantic_evaluator_rule_version,
        source_bundle_id=source_bundle_id,
        assertion_compiler_version=base.manifest.assertion_compiler_version,
        curator_version=base.manifest.curator_version,
        validator_version=base.manifest.validator_version,
        context_compiler_version=base.manifest.context_compiler_version,
        schema_version=base.manifest.schema_version,
        built_at=delta.built_at,
        claim_key_registry=base.manifest.claim_key_registry,
    )
    if manifest.snapshot_id != delta.expected_target_snapshot_id:
        raise DeltaError(
            "applied delta does not reproduce expected_target_snapshot_id: "
            f"{manifest.snapshot_id} != {delta.expected_target_snapshot_id}"
        )
    return s.SnapshotBundle(
        manifest=manifest,
        entities=s.sort_entities(entities),
        literals=s.sort_literals(literals),
        source_revisions=s.sort_source_revisions(revisions),
        assertions=s.sort_assertions(assertions),
    )


def rollback(base: s.SnapshotBundle, _candidate: s.SnapshotBundle | None = None) -> s.SnapshotBundle:
    """Discard an unpublished candidate and return the unchanged snapshot N."""

    return base


def snapshot_export(bundle: s.SnapshotBundle) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Return the complete canonical logical export used for equality checks."""

    return (
        tuple(s.entity_json_record(e) for e in s.sort_entities(bundle.entities)),
        tuple(s.literal_json_record(l) for l in s.sort_literals(bundle.literals)),
        tuple(s.source_revision_json_record(r) for r in s.sort_source_revisions(bundle.source_revisions)),
        tuple(s.assertion_json_record(a) for a in s.sort_assertions(bundle.assertions)),
    )


def delta_to_record(delta: SnapshotDelta) -> dict[str, Any]:
    """Serialize a delta to canonical JSON-compatible data (bytes as hex)."""

    return {
        "base_snapshot_id": delta.base_snapshot_id,
        "expected_target_snapshot_id": delta.expected_target_snapshot_id,
        "added_source_objects": [
            {"sha256": digest, "bytes_hex": value.hex()}
            for digest, value in delta.added_source_objects
        ],
        "added_source_revisions": [s.source_revision_json_record(r) for r in delta.added_source_revisions],
        "added_entities": [s.entity_json_record(e) for e in delta.added_entities],
        "added_literals": [s.literal_json_record(l) for l in delta.added_literals],
        "added_assertions": [s.assertion_json_record(a) for a in delta.added_assertions],
        "remove_assertion_ids": list(delta.remove_assertion_ids),
        "target_semantic_source_bundle_projection_checksum": delta.target_semantic_source_bundle_projection_checksum,
        "target_source_bundle_id": delta.target_source_bundle_id,
        "built_at": delta.built_at,
    }


def canonical_delta_bytes(delta: SnapshotDelta) -> bytes:
    return s.canonical_json_line(delta_to_record(delta))


def delta_from_record(record: Mapping[str, Any]) -> SnapshotDelta:
    required = {
        "base_snapshot_id", "expected_target_snapshot_id", "added_source_objects",
        "added_source_revisions", "added_entities", "added_literals", "added_assertions",
        "remove_assertion_ids", "target_semantic_source_bundle_projection_checksum",
        "target_source_bundle_id", "built_at",
    }
    if set(record) != required:
        raise DeltaError("delta record has unknown or missing fields")
    objects_list: list[tuple[str, bytes]] = []
    try:
        for entry in record["added_source_objects"]:
            if set(entry) != {"sha256", "bytes_hex"}:
                raise DeltaError("added source object record has unknown or missing fields")
            objects_list.append((entry["sha256"], bytes.fromhex(entry["bytes_hex"])))
    except (TypeError, ValueError, KeyError) as exc:
        raise DeltaError("invalid added source object encoding") from exc
    objects = tuple(objects_list)
    return SnapshotDelta(
        base_snapshot_id=record["base_snapshot_id"],
        expected_target_snapshot_id=record["expected_target_snapshot_id"],
        added_source_objects=objects,
        added_source_revisions=tuple(
            s.source_revision_from_json_record(item) for item in record["added_source_revisions"]
        ),
        added_entities=tuple(s.entity_from_json_record(item) for item in record["added_entities"]),
        added_literals=tuple(s.literal_from_json_record(item) for item in record["added_literals"]),
        added_assertions=tuple(s.assertion_from_json_record(item) for item in record["added_assertions"]),
        remove_assertion_ids=tuple(record["remove_assertion_ids"]),
        target_semantic_source_bundle_projection_checksum=record[
            "target_semantic_source_bundle_projection_checksum"
        ],
        target_source_bundle_id=record["target_source_bundle_id"],
        built_at=record["built_at"],
    )


# Explicit names used by the G1 handoff and convenient compatibility aliases.
validate_canonical_delta = validate_delta
apply_canonical_delta = apply_delta
