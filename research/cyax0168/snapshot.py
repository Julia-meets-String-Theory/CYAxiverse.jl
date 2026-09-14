#!/usr/bin/env python3
"""CYAX-0168 immutable assertion snapshot: identity, persistence, freshness.

Implements, standard-library only, the frozen contract in
``specs/0168-structured-graph-materialization/spec.md``: "Immutable source
bundle", "Immutable assertion snapshot and identity", and "Freshness versus
consistency". Builds on ``common.py`` (the closed vocabulary, per-record
validators, and canonical framing) for the referential closure that turns a
bag of records into a validated, checksummed, on-disk snapshot.

Scope boundary (CYAX-0168 G1, Worker A+B): this module computes and
verifies identity/checksums and reads/writes the on-disk snapshot payload.
It never materializes, executes, or inspects a T0-T4 fixture through
either backend, never builds an S/G-specific index, and never enters
G2+ semantic-parity territory. Atomic publication itself (the
build-validate-fsync-rename protocol) is ``publication.py``; canonical
N-to-N+1 deltas are ``update.py``.

Import convention: this module uses a flat ``import common`` (not a
package-relative import), matching this programme's established
same-directory test convention (``sys.path.insert(0, HERE)`` before
importing any of these modules by name; see ``test_evaluator.py``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Same-directory test/import convention and package imports.
    import common as c
except ImportError:  # pragma: no cover - exercised by package importers
    from . import common as c


class SnapshotError(c.SemanticError):
    """Raised when a snapshot payload violates the frozen identity contract."""


class SourceBundleError(SnapshotError):
    """Raised when a content-addressed source bundle is incomplete or stale."""


SCHEMA_VERSION = "cyax-snapshot-v1"
SNAPSHOT_PAYLOAD_FILES = (
    "entities.jsonl",
    "literals.jsonl",
    "source_revisions.jsonl",
    "assertions.jsonl",
)
_PAYLOAD_KEY_PREFIX = {
    "entities.jsonl": "entities",
    "literals.jsonl": "literals",
    "source_revisions.jsonl": "source_revisions",
    "assertions.jsonl": "assertions",
}


# ---------------------------------------------------------------------------
# Canonical JSON encoding for JSONL payloads (spec: "Canonical encoding and
# stable IDs"; "Immutable assertion snapshot and identity")
# ---------------------------------------------------------------------------


def _reject_floats(value: Any) -> None:
    if isinstance(value, float):
        raise SnapshotError("floats are forbidden in canonical snapshot payloads")
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_floats(item)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_floats(item)


def _nfc_deep(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_deep(item) for item in value]
    if isinstance(value, tuple):
        return [_nfc_deep(item) for item in value]
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key) if isinstance(key, str) else key
            if normalized_key in result:
                raise SnapshotError(f"canonical JSON object has colliding NFC key: {normalized_key!r}")
            result[normalized_key] = _nfc_deep(item)
        return result
    return value


def canonical_json_line(record: Mapping[str, Any]) -> bytes:
    """UTF-8/NFC, sorted-key, no-insignificant-whitespace JSON plus one LF."""

    _reject_floats(record)
    normalized = _nfc_deep(record)
    text = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return text.encode("utf-8") + b"\n"


def encode_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_line(record) for record in records)


def decode_jsonl(data: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in data.split(b"\n"):
        if not line:
            continue
        records.append(json.loads(line.decode("utf-8")))
    return records


def _ensure_canonical_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    """Decode JSONL and require the exact canonical byte representation.

    A parser accepting alternate whitespace or key order would allow two
    physical source bundles to claim one identity.  Empty lines are not part
    of the v1 payload contract (the trailing LF is allowed by
    :func:`canonical_json_line`).
    """

    if not data:
        return []
    records = decode_jsonl(data)
    if encode_jsonl(records) != data:
        raise SourceBundleError(f"{label} is not canonical UTF-8/NFC JSONL")
    if any(not isinstance(record, dict) for record in records):
        raise SourceBundleError(f"{label} records must be JSON objects")
    return records


# ---------------------------------------------------------------------------
# Record <-> JSON conversion
# ---------------------------------------------------------------------------


def entity_json_record(entity: c.Entity) -> dict[str, Any]:
    record = {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "namespace": entity.namespace,
        "canonical_source_identity": entity.canonical_source_identity,
        "display_label_ref": entity.display_label_ref,
    }
    if entity.entity_type == "Claim":
        record["claim_key"] = entity.claim_key
    return record


def entity_from_json_record(record: Mapping[str, Any]) -> c.Entity:
    required = {"entity_id", "entity_type", "namespace", "canonical_source_identity", "display_label_ref"}
    if not required.issubset(record) or set(record) - required - {"claim_key"}:
        raise SnapshotError("entity record has unknown or missing fields")
    if record["entity_type"] == "Claim" and set(record) != required | {"claim_key"}:
        raise SnapshotError("Claim entity record must contain claim_key")
    if record["entity_type"] != "Claim" and "claim_key" in record:
        raise SnapshotError("non-Claim entity record must not contain claim_key")
    return c.Entity(
        entity_id=record["entity_id"],
        entity_type=record["entity_type"],
        namespace=record["namespace"],
        canonical_source_identity=record["canonical_source_identity"],
        display_label_ref=record.get("display_label_ref"),
        claim_key=record.get("claim_key"),
    )


def literal_json_record(literal: c.Literal) -> dict[str, Any]:
    return {"literal_id": literal.literal_id, "literal_type": literal.literal_type, "value": literal.value}


def literal_from_json_record(record: Mapping[str, Any]) -> c.Literal:
    if set(record) != {"literal_id", "literal_type", "value"}:
        raise SnapshotError("literal record has unknown or missing fields")
    return c.Literal(
        literal_id=record["literal_id"], literal_type=record["literal_type"], value=record["value"]
    )


def source_revision_json_record(revision: c.SourceRevision) -> dict[str, Any]:
    return {
        "source_revision_id": revision.source_revision_id,
        "source_kind": revision.source_kind,
        "source_entity_id": revision.source_entity_id,
        "canonical_locator": revision.canonical_locator,
        "object_sha256": revision.object_sha256,
        "object_byte_count": revision.object_byte_count,
        "source_event_at": revision.source_event_at,
        "observed_at": revision.observed_at,
        "actor_id": revision.actor_id,
        "authority_class": revision.authority_class,
        "authority_derivation_rule_id": revision.authority_derivation_rule_id,
    }


def source_revision_from_json_record(record: Mapping[str, Any]) -> c.SourceRevision:
    # ``byte_count`` was used by the independent synthetic generator's raw
    # records; accept it only as an input alias and emit the frozen v1 name
    # ``object_byte_count`` from this module.
    if "object_byte_count" not in record and "byte_count" in record:
        record = dict(record)
        record["object_byte_count"] = record.pop("byte_count")
    expected = {
        "source_revision_id", "source_kind", "source_entity_id", "canonical_locator",
        "object_sha256", "object_byte_count", "source_event_at", "observed_at",
        "actor_id", "authority_class", "authority_derivation_rule_id",
    }
    if set(record) != expected:
        raise SnapshotError("source revision record has unknown or missing fields")
    return c.SourceRevision(
        source_revision_id=record["source_revision_id"],
        source_kind=record["source_kind"],
        source_entity_id=record["source_entity_id"],
        canonical_locator=record["canonical_locator"],
        object_sha256=record["object_sha256"],
        object_byte_count=record["object_byte_count"],
        source_event_at=record.get("source_event_at"),
        observed_at=record["observed_at"],
        actor_id=record.get("actor_id"),
        authority_class=record.get("authority_class", "ordinary_record"),
        authority_derivation_rule_id=record.get("authority_derivation_rule_id", "ordinary_record"),
    )


def assertion_json_record(assertion: c.Assertion) -> dict[str, Any]:
    record = dict(assertion.semantic_fields())
    record["assertion_id"] = assertion.assertion_id
    return record


def assertion_from_json_record(record: Mapping[str, Any]) -> c.Assertion:
    if set(record) != set(c._ASSERTION_ID_FIELDS) | {"assertion_id"}:
        raise SnapshotError("assertion record has unknown or missing fields")
    kwargs = {name: record[name] for name in c._ASSERTION_ID_FIELDS}
    return c.Assertion(assertion_id=record["assertion_id"], **kwargs)


# ---------------------------------------------------------------------------
# Canonical ordering (spec: "primary-ID byte order")
# ---------------------------------------------------------------------------


def sort_entities(entities: Sequence[c.Entity]) -> tuple[c.Entity, ...]:
    return tuple(sorted(entities, key=lambda e: e.entity_id))


def sort_literals(literals: Sequence[c.Literal]) -> tuple[c.Literal, ...]:
    return tuple(sorted(literals, key=lambda l: l.literal_id))


def sort_source_revisions(revisions: Sequence[c.SourceRevision]) -> tuple[c.SourceRevision, ...]:
    return tuple(sorted(revisions, key=lambda r: r.source_revision_id))


def sort_assertions(assertions: Sequence[c.Assertion]) -> tuple[c.Assertion, ...]:
    return tuple(sorted(assertions, key=lambda a: a.assertion_id))


# ---------------------------------------------------------------------------
# Semantic source bundle projection and snapshot identity
# (spec: "Immutable assertion snapshot and identity")
# ---------------------------------------------------------------------------


def reachable_literal_ids(assertions: Sequence[c.Assertion]) -> set[str]:
    return {a.literal_ref for a in assertions if a.literal_ref is not None}


def reachable_source_revision_ids(assertions: Sequence[c.Assertion]) -> set[str]:
    return {a.source_revision_id for a in assertions}


def semantic_entity_projection(entity: c.Entity) -> dict[str, Any]:
    """Entity projection with ``display_label_ref`` *omitted* (not null)."""

    record: dict[str, Any] = {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "namespace": entity.namespace,
        "canonical_source_identity": entity.canonical_source_identity,
    }
    if entity.entity_type == "Claim":
        record["claim_key"] = entity.claim_key
    return record


def semantic_literal_projection(literal: c.Literal) -> dict[str, Any]:
    return {"literal_id": literal.literal_id, "literal_type": literal.literal_type, "value": literal.value}


def semantic_source_revision_projection(revision: c.SourceRevision) -> dict[str, Any]:
    return source_revision_json_record(revision)


def semantic_assertion_projection(assertion: c.Assertion) -> dict[str, Any]:
    return assertion_json_record(assertion)


def compute_semantic_source_bundle_projection_checksum(
    source_revisions: Sequence[c.SourceRevision],
    objects: Sequence[tuple[str, int]],
    actor_role_records: Sequence[Mapping[str, Any]],
    owner_decision_entries: Sequence[c.OwnerDecisionRegistryEntry],
) -> str:
    """SHA-256 of the reachable source-bundle projection.

    Callers determine "reachable from semantic assertions" (e.g. via
    :func:`reachable_source_revision_ids`) before calling this; the
    projection itself only orders and hashes what it is given.
    """

    ordered_revisions = [
        semantic_source_revision_projection(r) for r in sort_source_revisions(source_revisions)
    ]
    ordered_objects = [list(pair) for pair in sorted(objects, key=lambda pair: pair[0])]
    ordered_owner_entries = [
        {
            "source_revision_id": e.source_revision_id,
            "source_locator": e.source_locator,
            "registry_entry_id": e.registry_entry_id,
        }
        for e in sorted(
            owner_decision_entries, key=lambda e: c.frame([e.source_revision_id, e.source_locator])
        )
    ]
    payload = c.frame(
        [
            "cyax-source-bundle-projection-v1",
            ordered_revisions,
            ordered_objects,
            sorted((dict(record) for record in actor_role_records), key=c.frame),
            ordered_owner_entries,
        ]
    )
    return c.sha256_hex(payload)


def compute_snapshot_id(
    *,
    schema_version: str,
    semantic_source_bundle_projection_checksum: str,
    authority_rule_version: str,
    semantic_evaluator_rule_version: str,
    entities: Sequence[c.Entity],
    literals: Sequence[c.Literal],
    source_revisions: Sequence[c.SourceRevision],
    assertions: Sequence[c.Assertion],
) -> str:
    """The logical semantic checksum: ``cyax-snapshot-sha256:<digest>``.

    Reordered input, a different compiler with identical semantics, and a
    diagnostic-label-only change leave this unchanged, because every input
    list is independently re-sorted here and ``display_label_ref``/
    unreferenced literals are excluded from the projection.
    """

    sorted_assertions = sort_assertions(assertions)
    reachable_literals = reachable_literal_ids(sorted_assertions)
    reachable_revisions = reachable_source_revision_ids(sorted_assertions)

    semantic_entities = [semantic_entity_projection(e) for e in sort_entities(entities)]
    semantic_literals = [
        semantic_literal_projection(l)
        for l in sort_literals(literals)
        if l.literal_id in reachable_literals
    ]
    semantic_source_revisions = [
        semantic_source_revision_projection(r)
        for r in sort_source_revisions(source_revisions)
        if r.source_revision_id in reachable_revisions
    ]
    semantic_assertions = [semantic_assertion_projection(a) for a in sorted_assertions]

    payload = c.frame(
        [
            "cyax-logical-snapshot-v1",
            schema_version,
            semantic_source_bundle_projection_checksum,
            authority_rule_version,
            semantic_evaluator_rule_version,
            ["semantic_entities", semantic_entities],
            ["semantic_literals", semantic_literals],
            ["semantic_source_revisions", semantic_source_revisions],
            ["semantic_assertions", semantic_assertions],
        ]
    )
    return f"cyax-snapshot-sha256:{c.sha256_hex(payload)}"


def compute_physical_payload_checksum(
    entities_jsonl: bytes,
    literals_jsonl: bytes,
    source_revisions_jsonl: bytes,
    assertions_jsonl: bytes,
) -> str:
    """Hash of the complete encoded physical payload (display refs and
    display-only literals included).

    Implementation convention: the spec fixes the *logical* checksum's
    exact frame array but describes this one only in prose ("hashes the
    complete encoded physical payload"); this framing of the four raw
    JSONL byte blobs is this module's deterministic, documented choice.
    """

    payload = c.frame(
        ["cyax-physical-payload-v1", entities_jsonl, literals_jsonl, source_revisions_jsonl, assertions_jsonl]
    )
    return c.sha256_hex(payload)


def compute_build_contract_checksum(
    *,
    snapshot_id: str,
    physical_payload_checksum: str,
    source_bundle_id: str | None,
    assertion_compiler_version: str,
    curator_version: str,
    validator_version: str,
    context_compiler_version: str,
) -> str:
    payload = c.frame(
        [
            "cyax-build-contract-v1",
            snapshot_id,
            physical_payload_checksum,
            source_bundle_id,
            assertion_compiler_version,
            curator_version,
            validator_version,
            context_compiler_version,
        ]
    )
    return c.sha256_hex(payload)


def compute_source_bundle_id(
    *,
    source_revisions_jsonl: bytes,
    actors_jsonl: bytes,
    events_jsonl: bytes,
    owner_decision_events_jsonl: bytes,
    objects: Sequence[tuple[str, int]],
) -> str:
    """``cyax-source-bundle-sha256:<digest>`` over the complete physical
    source bundle (all four JSONL payloads plus every captured object),
    distinct from :func:`compute_semantic_source_bundle_projection_checksum`,
    which covers only the subset reachable from the current snapshot's
    semantic assertions.
    """

    ordered_objects = [list(pair) for pair in sorted(objects, key=lambda pair: pair[0])]
    payload = c.frame(
        [
            "cyax-source-bundle-v1",
            source_revisions_jsonl,
            actors_jsonl,
            events_jsonl,
            owner_decision_events_jsonl,
            ordered_objects,
        ]
    )
    return f"cyax-source-bundle-sha256:{c.sha256_hex(payload)}"


# ---------------------------------------------------------------------------
# Manifest and snapshot bundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SnapshotManifest:
    schema_version: str
    snapshot_id: str
    physical_payload_checksum: str
    build_contract_checksum: str
    semantic_source_bundle_projection_checksum: str
    authority_rule_version: str
    semantic_evaluator_rule_version: str
    source_bundle_id: str | None
    assertion_compiler_version: str
    curator_version: str
    validator_version: str
    context_compiler_version: str
    entity_count: int
    literal_count: int
    source_revision_count: int
    assertion_count: int
    entities_byte_count: int
    entities_sha256: str
    literals_byte_count: int
    literals_sha256: str
    source_revisions_byte_count: int
    source_revisions_sha256: str
    assertions_byte_count: int
    assertions_sha256: str
    ordered_source_revision_ids: tuple[str, ...]
    build_complete: bool = False
    built_at: str | None = None
    claim_key_registry: tuple[c.ClaimKeyRegistryEntry, ...] = ()


@dataclasses.dataclass(frozen=True)
class SnapshotBundle:
    manifest: SnapshotManifest
    entities: tuple[c.Entity, ...]
    literals: tuple[c.Literal, ...]
    source_revisions: tuple[c.SourceRevision, ...]
    assertions: tuple[c.Assertion, ...]


# ---------------------------------------------------------------------------
# Content-addressed source bundle
# ---------------------------------------------------------------------------


SOURCE_BUNDLE_SCHEMA_VERSION = "cyax-source-bundle-v1"
SOURCE_BUNDLE_PAYLOAD_FILES = (
    "source_revisions.jsonl",
    "actors.jsonl",
    "events.jsonl",
    "owner_decision_events.jsonl",
)


@dataclasses.dataclass(frozen=True)
class SourceBundleManifest:
    """The immutable manifest for the captured source bundle.

    Payload and object hashes are recorded explicitly instead of trusting a
    directory listing.  ``built_at`` is audit metadata only and is excluded
    from ``source_bundle_id``.
    """

    schema_version: str
    source_bundle_id: str
    observation_boundary: str
    source_revision_ids: tuple[str, ...]
    object_digests: tuple[tuple[str, int], ...]
    payload_byte_counts: Mapping[str, int]
    payload_sha256: Mapping[str, str]
    build_complete: bool = False
    built_at: str | None = None


@dataclasses.dataclass(frozen=True)
class SourceBundle:
    manifest: SourceBundleManifest
    source_revisions: tuple[c.SourceRevision, ...]
    actors_jsonl: bytes
    events_jsonl: bytes
    owner_decision_events_jsonl: bytes
    objects: Mapping[str, bytes]


def _object_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalise_objects(objects: Mapping[str, bytes] | Sequence[tuple[str, bytes]]) -> dict[str, bytes]:
    if isinstance(objects, Mapping):
        pairs = list(objects.items())
    else:
        pairs = list(objects)
    result: dict[str, bytes] = {}
    for digest, value in pairs:
        if not isinstance(digest, str) or not c.SHA256_RE.match(digest):
            raise SourceBundleError(f"source object digest is not lowercase SHA-256: {digest!r}")
        if not isinstance(value, bytes):
            raise SourceBundleError(f"source object {digest!r} must contain bytes")
        actual = _object_digest(value)
        if actual != digest:
            raise SourceBundleError(f"source object {digest!r} content hash is {actual!r}")
        if digest in result:
            raise SourceBundleError(f"duplicate source object digest: {digest}")
        result[digest] = value
    return result


def _source_payload_records(
    source_revisions: Sequence[c.SourceRevision],
    actors_jsonl: bytes,
    events_jsonl: bytes,
    owner_decision_events_jsonl: bytes,
) -> tuple[bytes, bytes, bytes, bytes]:
    ordered = sort_source_revisions(source_revisions)
    revision_payload = encode_jsonl([source_revision_json_record(r) for r in ordered])
    # Validate all auxiliary payloads now, so publication cannot create a
    # bundle that a reader would later reject.  Empty payloads are valid for a
    # synthetic fixture; real F-real callers must populate them.
    _ensure_canonical_jsonl(actors_jsonl, "actors.jsonl")
    _ensure_canonical_jsonl(events_jsonl, "events.jsonl")
    owner_records = _ensure_canonical_jsonl(
        owner_decision_events_jsonl, "owner_decision_events.jsonl"
    )
    owner_entries: list[c.OwnerDecisionRegistryEntry] = []
    for record in owner_records:
        expected = {"source_revision_id", "source_locator", "registry_entry_id"}
        if set(record) != expected:
            raise SourceBundleError("owner_decision_events records have unknown or missing fields")
        owner_entries.append(c.OwnerDecisionRegistryEntry(**record))
    if owner_entries:
        try:
            c.validate_owner_decision_registry(
                owner_entries,
                known_source_locators=[(r.source_revision_id, r.canonical_locator) for r in ordered],
            )
        except c.SemanticError as exc:
            raise SourceBundleError(str(exc)) from exc
    return revision_payload, actors_jsonl, events_jsonl, owner_decision_events_jsonl


def build_source_bundle(
    *,
    source_revisions: Sequence[c.SourceRevision],
    objects: Mapping[str, bytes] | Sequence[tuple[str, bytes]],
    observation_boundary: str,
    actors_jsonl: bytes = b"",
    events_jsonl: bytes = b"",
    owner_decision_events_jsonl: bytes = b"",
    schema_version: str = SOURCE_BUNDLE_SCHEMA_VERSION,
    built_at: str | None = None,
) -> tuple[SourceBundleManifest, dict[str, bytes]]:
    """Build a source-bundle manifest and payload map without publishing.

    Every revision's captured object must be present and byte-addressed.  The
    returned payload map contains the four JSONL files and object bytes keyed
    by ``objects/sha256/<two>/<remaining>``; :mod:`publication` performs the
    atomic filesystem operation.
    """

    if schema_version != SOURCE_BUNDLE_SCHEMA_VERSION:
        raise SourceBundleError(f"unsupported source bundle schema: {schema_version!r}")
    if not isinstance(observation_boundary, str) or not observation_boundary:
        raise SourceBundleError("observation_boundary must be a non-empty string")
    ordered_revisions = sort_source_revisions(source_revisions)
    seen: set[str] = set()
    for revision in ordered_revisions:
        try:
            revision.validated()
        except c.SemanticError as exc:
            raise SourceBundleError(str(exc)) from exc
        if revision.source_revision_id in seen:
            raise SourceBundleError(f"duplicate source_revision_id: {revision.source_revision_id}")
        seen.add(revision.source_revision_id)
    object_map = _normalise_objects(objects)
    for revision in ordered_revisions:
        if revision.object_sha256 not in object_map:
            raise SourceBundleError(
                f"source revision {revision.source_revision_id} object is unavailable"
            )
        if len(object_map[revision.object_sha256]) != revision.object_byte_count:
            raise SourceBundleError(
                f"source revision {revision.source_revision_id} object byte count mismatch"
            )
    payloads_tuple = _source_payload_records(
        ordered_revisions, actors_jsonl, events_jsonl, owner_decision_events_jsonl
    )
    payloads = dict(zip(SOURCE_BUNDLE_PAYLOAD_FILES, payloads_tuple))
    object_pairs = tuple(sorted((digest, len(value)) for digest, value in object_map.items()))
    bundle_id = compute_source_bundle_id(
        source_revisions_jsonl=payloads["source_revisions.jsonl"],
        actors_jsonl=payloads["actors.jsonl"],
        events_jsonl=payloads["events.jsonl"],
        owner_decision_events_jsonl=payloads["owner_decision_events.jsonl"],
        objects=object_pairs,
    )
    manifest = SourceBundleManifest(
        schema_version=schema_version,
        source_bundle_id=bundle_id,
        observation_boundary=observation_boundary,
        source_revision_ids=tuple(r.source_revision_id for r in ordered_revisions),
        object_digests=object_pairs,
        payload_byte_counts={name: len(payloads[name]) for name in SOURCE_BUNDLE_PAYLOAD_FILES},
        payload_sha256={name: c.sha256_hex(payloads[name]) for name in SOURCE_BUNDLE_PAYLOAD_FILES},
        build_complete=False,
        built_at=built_at,
    )
    for digest, value in object_map.items():
        payloads[f"objects/sha256/{digest[:2]}/{digest[2:]}"] = value
    return manifest, payloads


def source_bundle_manifest_to_json(manifest: SourceBundleManifest) -> dict[str, Any]:
    record = dataclasses.asdict(manifest)
    record["source_revision_ids"] = list(manifest.source_revision_ids)
    record["object_digests"] = [list(pair) for pair in manifest.object_digests]
    record["payload_byte_counts"] = dict(manifest.payload_byte_counts)
    record["payload_sha256"] = dict(manifest.payload_sha256)
    return record


def source_bundle_manifest_from_json(record: Mapping[str, Any]) -> SourceBundleManifest:
    required = {
        "schema_version", "source_bundle_id", "observation_boundary",
        "source_revision_ids", "object_digests", "payload_byte_counts",
        "payload_sha256", "build_complete", "built_at",
    }
    if set(record) != required:
        raise SourceBundleError("source bundle manifest has unknown or missing fields")
    return SourceBundleManifest(
        schema_version=record["schema_version"],
        source_bundle_id=record["source_bundle_id"],
        observation_boundary=record["observation_boundary"],
        source_revision_ids=tuple(record["source_revision_ids"]),
        object_digests=tuple((pair[0], pair[1]) for pair in record["object_digests"]),
        payload_byte_counts=dict(record["payload_byte_counts"]),
        payload_sha256=dict(record["payload_sha256"]),
        build_complete=record["build_complete"],
        built_at=record["built_at"],
    )


def source_bundle_path(root: Path, source_bundle_id: str) -> Path:
    """Return the content-addressed publication path for a bundle ID."""

    if not source_bundle_id.startswith("cyax-source-bundle-sha256:"):
        raise SourceBundleError("malformed source_bundle_id")
    return root / source_bundle_id


def validate_source_bundle(
    manifest: SourceBundleManifest,
    payloads: Mapping[str, bytes],
) -> SourceBundle:
    """Validate source-bundle bytes and return immutable in-memory content."""

    if manifest.schema_version != SOURCE_BUNDLE_SCHEMA_VERSION:
        raise SourceBundleError("unsupported source bundle schema")
    if not isinstance(manifest.source_bundle_id, str) or not manifest.source_bundle_id.startswith(
        "cyax-source-bundle-sha256:"
    ):
        raise SourceBundleError("malformed source_bundle_id")
    if not isinstance(manifest.build_complete, bool) or not manifest.build_complete:
        raise SourceBundleError("source bundle is not marked build_complete")
    if tuple(manifest.object_digests) != tuple(sorted(manifest.object_digests)):
        raise SourceBundleError("source bundle object_digests must be sorted")
    if len(manifest.object_digests) != len(set(manifest.object_digests)):
        raise SourceBundleError("source bundle object_digests must be unique")
    for digest, byte_count in manifest.object_digests:
        if not c.SHA256_RE.match(digest) or not isinstance(byte_count, int) or byte_count < 0:
            raise SourceBundleError("source bundle object manifest entry is malformed")
    expected_names = set(SOURCE_BUNDLE_PAYLOAD_FILES)
    if {name for name in payloads if name in expected_names} != expected_names:
        raise SourceBundleError("source bundle is missing one or more JSONL payloads")
    expected_object_names = {
        f"objects/sha256/{digest[:2]}/{digest[2:]}" for digest, _ in manifest.object_digests
    }
    if set(payloads) != expected_names | expected_object_names:
        raise SourceBundleError("source bundle contains unknown payloads")
    if set(manifest.payload_byte_counts) != expected_names or set(manifest.payload_sha256) != expected_names:
        raise SourceBundleError("source bundle payload manifest is incomplete")
    for name in SOURCE_BUNDLE_PAYLOAD_FILES:
        value = payloads[name]
        if len(value) != manifest.payload_byte_counts.get(name):
            raise SourceBundleError(f"{name} byte count mismatch")
        if c.sha256_hex(value) != manifest.payload_sha256.get(name):
            raise SourceBundleError(f"{name} checksum mismatch")
    revision_records = _ensure_canonical_jsonl(payloads["source_revisions.jsonl"], "source_revisions.jsonl")
    revisions = tuple(
        sort_source_revisions([source_revision_from_json_record(record) for record in revision_records])
    )
    canonical_revision_payload = encode_jsonl([source_revision_json_record(r) for r in revisions])
    if payloads["source_revisions.jsonl"] != canonical_revision_payload:
        raise SourceBundleError("source_revisions.jsonl is not ordered by source_revision_id")
    if tuple(r.source_revision_id for r in revisions) != manifest.source_revision_ids:
        raise SourceBundleError("source_revision_ids do not match manifest")
    objects: dict[str, bytes] = {}
    for digest, byte_count in manifest.object_digests:
        key = f"objects/sha256/{digest[:2]}/{digest[2:]}"
        if key not in payloads:
            raise SourceBundleError(f"missing source object {digest}")
        value = payloads[key]
        if len(value) != byte_count or _object_digest(value) != digest:
            raise SourceBundleError(f"source object {digest} is tampered")
        objects[digest] = value
    for revision in revisions:
        revision.validated()
        if revision.object_sha256 not in objects:
            raise SourceBundleError(f"source revision {revision.source_revision_id} has no object")
        if len(objects[revision.object_sha256]) != revision.object_byte_count:
            raise SourceBundleError(f"source revision {revision.source_revision_id} byte count mismatch")
    owner_payload = payloads["owner_decision_events.jsonl"]
    _source_payload_records(revisions, payloads["actors.jsonl"], payloads["events.jsonl"], owner_payload)
    object_pairs = tuple(sorted((digest, len(value)) for digest, value in objects.items()))
    expected_id = compute_source_bundle_id(
        source_revisions_jsonl=payloads["source_revisions.jsonl"],
        actors_jsonl=payloads["actors.jsonl"],
        events_jsonl=payloads["events.jsonl"],
        owner_decision_events_jsonl=owner_payload,
        objects=object_pairs,
    )
    if expected_id != manifest.source_bundle_id:
        raise SourceBundleError("source_bundle_id does not match canonical bundle bytes")
    return SourceBundle(
        manifest=manifest,
        source_revisions=revisions,
        actors_jsonl=payloads["actors.jsonl"],
        events_jsonl=payloads["events.jsonl"],
        owner_decision_events_jsonl=owner_payload,
        objects=objects,
    )


def read_source_bundle_dir(path: Path) -> SourceBundle:
    """Open a published source bundle and reject tampering or partial state."""

    if path.name.startswith(".") or ".tmp-" in path.name:
        raise SourceBundleError("temporary source-bundle paths are not openable")
    manifest_path = path / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise SourceBundleError(f"missing bundle_manifest.json in {path}")
    try:
        raw_manifest = manifest_path.read_bytes()
        record = json.loads(raw_manifest.decode("utf-8"))
        manifest = source_bundle_manifest_from_json(record)
        if raw_manifest != canonical_json_line(record):
            raise SourceBundleError("source bundle manifest is not canonical JSON")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise SourceBundleError("invalid source bundle manifest") from exc
    expected_files = {"bundle_manifest.json", *SOURCE_BUNDLE_PAYLOAD_FILES}
    expected_files.update(
        f"objects/sha256/{digest[:2]}/{digest[2:]}" for digest, _ in manifest.object_digests
    )
    actual_files = {
        str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()
    }
    if actual_files != expected_files:
        raise SourceBundleError(
            f"source bundle contains unmanifested or residual files: "
            f"{sorted(actual_files ^ expected_files)}"
        )
    payloads: dict[str, bytes] = {}
    for name in SOURCE_BUNDLE_PAYLOAD_FILES:
        file_path = path / name
        if not file_path.is_file():
            raise SourceBundleError(f"missing {name} in {path}")
        payloads[name] = file_path.read_bytes()
    for digest, _ in manifest.object_digests:
        key = f"objects/sha256/{digest[:2]}/{digest[2:]}"
        file_path = path / key
        if not file_path.is_file():
            raise SourceBundleError(f"missing source object {digest}")
        payloads[key] = file_path.read_bytes()
    return validate_source_bundle(manifest, payloads)


def manifest_to_json(manifest: SnapshotManifest) -> dict[str, Any]:
    record = dataclasses.asdict(manifest)
    record["ordered_source_revision_ids"] = list(manifest.ordered_source_revision_ids)
    record["claim_key_registry"] = [dataclasses.asdict(entry) for entry in manifest.claim_key_registry]
    return record


def manifest_from_json(record: Mapping[str, Any]) -> SnapshotManifest:
    expected = {field.name for field in dataclasses.fields(SnapshotManifest)}
    if set(record) != expected:
        raise SnapshotError("snapshot manifest has unknown or missing fields")
    kwargs = dict(record)
    kwargs["ordered_source_revision_ids"] = tuple(record.get("ordered_source_revision_ids", ()))
    kwargs["claim_key_registry"] = tuple(
        c.ClaimKeyRegistryEntry(**entry) for entry in record["claim_key_registry"]
    )
    return SnapshotManifest(**kwargs)


def snapshot_path(root: Path, snapshot_id: str) -> Path:
    """Return the content-addressed publication path for a snapshot ID."""

    if not snapshot_id.startswith("cyax-snapshot-sha256:"):
        raise SnapshotError("malformed snapshot_id")
    return root / snapshot_id


def validate_snapshot_records(
    entities: Sequence[c.Entity],
    literals: Sequence[c.Literal],
    source_revisions: Sequence[c.SourceRevision],
    assertions: Sequence[c.Assertion],
    *,
    claim_key_registry: Mapping[str, c.ClaimKeyRegistryEntry] | None = None,
) -> None:
    """Full closed-vocabulary and referential validation of one candidate
    record set: per-record structural validity (via ``common.py``),
    uniqueness of every primary ID, predicate-signature conformance against
    actual entity types, Claim-registry literal-type conformance, and that
    every ``subject_id``/``object_id``/``literal_ref``/``source_revision_id``
    resolves within the set.
    """

    entity_by_id: dict[str, c.Entity] = {}
    for entity in entities:
        entity.validated(claim_key_registry)
        if entity.entity_id in entity_by_id:
            raise SnapshotError(f"duplicate entity_id: {entity.entity_id}")
        entity_by_id[entity.entity_id] = entity

    literal_by_id: dict[str, c.Literal] = {}
    for literal in literals:
        literal.validated()
        if literal.literal_id in literal_by_id:
            raise SnapshotError(f"duplicate literal_id: {literal.literal_id}")
        literal_by_id[literal.literal_id] = literal
    for entity in entities:
        if entity.display_label_ref is not None:
            label = literal_by_id.get(entity.display_label_ref)
            if label is None:
                raise SnapshotError(
                    f"entity {entity.entity_id} display_label_ref {entity.display_label_ref} not found"
                )
            if label.literal_type != "text":
                raise SnapshotError("display_label_ref must point to a text literal")

    revision_by_id: dict[str, c.SourceRevision] = {}
    for revision in source_revisions:
        revision.validated()
        if revision.source_revision_id in revision_by_id:
            raise SnapshotError(f"duplicate source_revision_id: {revision.source_revision_id}")
        if revision.source_entity_id not in entity_by_id:
            raise SnapshotError(
                f"source revision {revision.source_revision_id} source_entity_id "
                f"{revision.source_entity_id} not found"
            )
        revision_by_id[revision.source_revision_id] = revision

    assertion_by_id: dict[str, c.Assertion] = {}
    if any(entity.entity_type == "Claim" for entity in entity_by_id.values()) and claim_key_registry is None:
        raise SnapshotError("Claim records require a frozen claim_key_registry")
    for assertion in assertions:
        assertion.validated()
        if assertion.assertion_id in assertion_by_id:
            raise SnapshotError(f"duplicate assertion_id: {assertion.assertion_id}")
        assertion_by_id[assertion.assertion_id] = assertion

        if assertion.subject_id not in entity_by_id:
            raise SnapshotError(
                f"assertion {assertion.assertion_id} subject {assertion.subject_id} not found"
            )
        subject_type = entity_by_id[assertion.subject_id].entity_type

        object_type: str | None = None
        if assertion.object_id is not None:
            if assertion.object_id not in entity_by_id:
                raise SnapshotError(
                    f"assertion {assertion.assertion_id} object {assertion.object_id} not found"
                )
            object_type = entity_by_id[assertion.object_id].entity_type

        has_literal_ref = assertion.literal_ref is not None
        if has_literal_ref and assertion.literal_ref not in literal_by_id:
            raise SnapshotError(
                f"assertion {assertion.assertion_id} literal_ref {assertion.literal_ref} not found"
            )

        c.validate_predicate_signature(assertion.predicate, subject_type, object_type, has_literal_ref)

        if has_literal_ref and claim_key_registry is not None:
            subject_entity = entity_by_id[assertion.subject_id]
            if subject_entity.entity_type == "Claim" and subject_entity.claim_key is not None:
                registered = claim_key_registry.get(subject_entity.claim_key)
                literal = literal_by_id[assertion.literal_ref]
                if registered is not None and registered.literal_type != literal.literal_type:
                    raise SnapshotError(
                        f"Claim {subject_entity.entity_id} literal_type "
                        f"{literal.literal_type!r} does not match registry "
                        f"{registered.literal_type!r} for claim_key {subject_entity.claim_key!r}"
                    )

        if assertion.source_revision_id not in revision_by_id:
            raise SnapshotError(
                f"assertion {assertion.assertion_id} source_revision "
                f"{assertion.source_revision_id} not found"
            )
        if assertion.origin == "source_direct":
            try:
                source_revision = revision_by_id[assertion.source_revision_id]
                if (
                    source_revision.source_event_at is not None
                    and assertion.source_event_at != source_revision.source_event_at
                ):
                    raise c.SemanticError(
                        f"assertion {assertion.assertion_id} source_event_at disagrees with source revision"
                    )
                c.validate_asserted_at(assertion, source_revision)
            except c.SemanticError as exc:
                raise SnapshotError(str(exc)) from exc


def build_manifest(
    *,
    entities: Sequence[c.Entity],
    literals: Sequence[c.Literal],
    source_revisions: Sequence[c.SourceRevision],
    assertions: Sequence[c.Assertion],
    semantic_source_bundle_projection_checksum: str,
    authority_rule_version: str,
    semantic_evaluator_rule_version: str,
    source_bundle_id: str | None,
    assertion_compiler_version: str,
    curator_version: str,
    validator_version: str,
    context_compiler_version: str,
    schema_version: str = SCHEMA_VERSION,
    built_at: str | None = None,
    claim_key_registry: Sequence[c.ClaimKeyRegistryEntry] = (),
) -> tuple[SnapshotManifest, dict[str, bytes]]:
    """Build the manifest and encoded JSONL payloads for one candidate
    record set. Does not validate referential closure or write anything to
    disk; call :func:`validate_snapshot_records` first and
    ``publication.publish_snapshot`` to persist the result.
    """

    sorted_entities = sort_entities(entities)
    sorted_literals = sort_literals(literals)
    sorted_revisions = sort_source_revisions(source_revisions)
    sorted_assertions = sort_assertions(assertions)
    registry = tuple(claim_key_registry)
    if registry:
        try:
            c.validate_claim_key_registry(registry)
        except c.SemanticError as exc:
            raise SnapshotError(str(exc)) from exc

    entities_jsonl = encode_jsonl([entity_json_record(e) for e in sorted_entities])
    literals_jsonl = encode_jsonl([literal_json_record(l) for l in sorted_literals])
    source_revisions_jsonl = encode_jsonl([source_revision_json_record(r) for r in sorted_revisions])
    assertions_jsonl = encode_jsonl([assertion_json_record(a) for a in sorted_assertions])

    snapshot_id = compute_snapshot_id(
        schema_version=schema_version,
        semantic_source_bundle_projection_checksum=semantic_source_bundle_projection_checksum,
        authority_rule_version=authority_rule_version,
        semantic_evaluator_rule_version=semantic_evaluator_rule_version,
        entities=sorted_entities,
        literals=sorted_literals,
        source_revisions=sorted_revisions,
        assertions=sorted_assertions,
    )
    physical_payload_checksum = compute_physical_payload_checksum(
        entities_jsonl, literals_jsonl, source_revisions_jsonl, assertions_jsonl
    )
    build_contract_checksum = compute_build_contract_checksum(
        snapshot_id=snapshot_id,
        physical_payload_checksum=physical_payload_checksum,
        source_bundle_id=source_bundle_id,
        assertion_compiler_version=assertion_compiler_version,
        curator_version=curator_version,
        validator_version=validator_version,
        context_compiler_version=context_compiler_version,
    )

    manifest = SnapshotManifest(
        schema_version=schema_version,
        snapshot_id=snapshot_id,
        physical_payload_checksum=physical_payload_checksum,
        build_contract_checksum=build_contract_checksum,
        semantic_source_bundle_projection_checksum=semantic_source_bundle_projection_checksum,
        authority_rule_version=authority_rule_version,
        semantic_evaluator_rule_version=semantic_evaluator_rule_version,
        source_bundle_id=source_bundle_id,
        assertion_compiler_version=assertion_compiler_version,
        curator_version=curator_version,
        validator_version=validator_version,
        context_compiler_version=context_compiler_version,
        entity_count=len(sorted_entities),
        literal_count=len(sorted_literals),
        source_revision_count=len(sorted_revisions),
        assertion_count=len(sorted_assertions),
        entities_byte_count=len(entities_jsonl),
        entities_sha256=c.sha256_hex(entities_jsonl),
        literals_byte_count=len(literals_jsonl),
        literals_sha256=c.sha256_hex(literals_jsonl),
        source_revisions_byte_count=len(source_revisions_jsonl),
        source_revisions_sha256=c.sha256_hex(source_revisions_jsonl),
        assertions_byte_count=len(assertions_jsonl),
        assertions_sha256=c.sha256_hex(assertions_jsonl),
        ordered_source_revision_ids=tuple(r.source_revision_id for r in sorted_revisions),
        build_complete=False,
        built_at=built_at,
        claim_key_registry=registry,
    )
    payloads = {
        "entities.jsonl": entities_jsonl,
        "literals.jsonl": literals_jsonl,
        "source_revisions.jsonl": source_revisions_jsonl,
        "assertions.jsonl": assertions_jsonl,
    }
    return manifest, payloads


def read_snapshot_dir(path: Path) -> SnapshotBundle:
    """Open and fully validate a published snapshot directory: recomputes
    and checks every manifest-derived value, ID, foreign key, enum,
    predicate signature, literal constraint, canonical order, payload hash,
    semantic projection, logical checksum, physical payload checksum, and
    build-contract checksum. Fails closed on any mismatch.
    """

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SnapshotError(f"missing manifest.json in {path}")
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest_record = json.loads(raw_manifest.decode("utf-8"))
        manifest = manifest_from_json(manifest_record)
        if raw_manifest != canonical_json_line(manifest_record):
            raise SnapshotError("snapshot manifest is not canonical JSON")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SnapshotError("invalid snapshot manifest") from exc
    if not isinstance(manifest_record.get("build_complete"), bool) or manifest_record.get("build_complete") is not True:
        raise SnapshotError(f"snapshot at {path} was never published (build_complete is not true)")
    if not str(manifest_record.get("snapshot_id", "")).startswith("cyax-snapshot-sha256:"):
        raise SnapshotError("malformed snapshot_id")

    expected_files = set(SNAPSHOT_PAYLOAD_FILES) | {"manifest.json"}
    actual_files = {
        str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()
    }
    if actual_files != expected_files:
        raise SnapshotError(
            f"snapshot contains unmanifested or residual files: {sorted(actual_files ^ expected_files)}"
        )

    raw_payloads: dict[str, bytes] = {}
    for name in SNAPSHOT_PAYLOAD_FILES:
        file_path = path / name
        if not file_path.is_file():
            raise SnapshotError(f"missing {name} in {path}")
        raw_payloads[name] = file_path.read_bytes()

    for name in SNAPSHOT_PAYLOAD_FILES:
        prefix = _PAYLOAD_KEY_PREFIX[name]
        raw = raw_payloads[name]
        expected_bytes = manifest_record.get(f"{prefix}_byte_count")
        expected_sha = manifest_record.get(f"{prefix}_sha256")
        if len(raw) != expected_bytes:
            raise SnapshotError(f"{name} byte count does not match manifest ({len(raw)} != {expected_bytes})")
        if c.sha256_hex(raw) != expected_sha:
            raise SnapshotError(f"{name} checksum does not match manifest")

    entity_records = _ensure_canonical_jsonl(raw_payloads["entities.jsonl"], "entities.jsonl")
    literal_records = _ensure_canonical_jsonl(raw_payloads["literals.jsonl"], "literals.jsonl")
    revision_records = _ensure_canonical_jsonl(
        raw_payloads["source_revisions.jsonl"], "source_revisions.jsonl"
    )
    assertion_records = _ensure_canonical_jsonl(raw_payloads["assertions.jsonl"], "assertions.jsonl")
    if [record["entity_id"] for record in entity_records] != sorted(record["entity_id"] for record in entity_records):
        raise SnapshotError("entities.jsonl is not ordered by entity_id")
    if [record["literal_id"] for record in literal_records] != sorted(record["literal_id"] for record in literal_records):
        raise SnapshotError("literals.jsonl is not ordered by literal_id")
    if [record["source_revision_id"] for record in revision_records] != sorted(record["source_revision_id"] for record in revision_records):
        raise SnapshotError("source_revisions.jsonl is not ordered by source_revision_id")
    if [record["assertion_id"] for record in assertion_records] != sorted(record["assertion_id"] for record in assertion_records):
        raise SnapshotError("assertions.jsonl is not ordered by assertion_id")
    entities = [entity_from_json_record(r) for r in entity_records]
    literals = [literal_from_json_record(r) for r in literal_records]
    source_revisions = [
        source_revision_from_json_record(r) for r in revision_records
    ]
    assertions = [assertion_from_json_record(r) for r in assertion_records]

    registry_entries = tuple(manifest.claim_key_registry)
    registry = c.validate_claim_key_registry(registry_entries) if registry_entries else None
    validate_snapshot_records(
        entities,
        literals,
        source_revisions,
        assertions,
        claim_key_registry=registry,
    )

    recomputed_snapshot_id = compute_snapshot_id(
        schema_version=manifest_record["schema_version"],
        semantic_source_bundle_projection_checksum=manifest_record[
            "semantic_source_bundle_projection_checksum"
        ],
        authority_rule_version=manifest_record["authority_rule_version"],
        semantic_evaluator_rule_version=manifest_record["semantic_evaluator_rule_version"],
        entities=entities,
        literals=literals,
        source_revisions=source_revisions,
        assertions=assertions,
    )
    if recomputed_snapshot_id != manifest_record["snapshot_id"]:
        raise SnapshotError(
            "recomputed snapshot_id does not match manifest "
            "(on a collision with an unequal preimage, both preimages must be preserved as failure "
            "evidence, never rehashed or suffixed)"
        )

    recomputed_physical = compute_physical_payload_checksum(
        raw_payloads["entities.jsonl"],
        raw_payloads["literals.jsonl"],
        raw_payloads["source_revisions.jsonl"],
        raw_payloads["assertions.jsonl"],
    )
    if recomputed_physical != manifest_record["physical_payload_checksum"]:
        raise SnapshotError("recomputed physical_payload_checksum does not match manifest")

    recomputed_build_contract = compute_build_contract_checksum(
        snapshot_id=recomputed_snapshot_id,
        physical_payload_checksum=recomputed_physical,
        source_bundle_id=manifest_record.get("source_bundle_id"),
        assertion_compiler_version=manifest_record["assertion_compiler_version"],
        curator_version=manifest_record["curator_version"],
        validator_version=manifest_record["validator_version"],
        context_compiler_version=manifest_record["context_compiler_version"],
    )
    if recomputed_build_contract != manifest_record["build_contract_checksum"]:
        raise SnapshotError("recomputed build_contract_checksum does not match manifest")

    return SnapshotBundle(
        manifest=manifest,
        entities=sort_entities(entities),
        literals=sort_literals(literals),
        source_revisions=sort_source_revisions(source_revisions),
        assertions=sort_assertions(assertions),
    )


# ---------------------------------------------------------------------------
# Freshness versus consistency (spec: "Freshness versus consistency")
# ---------------------------------------------------------------------------


def matches_selected_snapshot(candidate_snapshot_id: str, expected_snapshot_id: str) -> bool:
    """A materialization "matches" only when its recomputed snapshot_id is
    byte-identical to the explicit frozen snapshot ID it was selected
    against."""

    return candidate_snapshot_id == expected_snapshot_id


class SourceRefresher:
    """Caller-supplied current-state reacquisition contract.

    ``fresh_against_current_sources`` requires reacquiring every canonical
    locator at a new observation boundary, verifying actor/role evidence,
    and comparing captured revision IDs, state/event records, and object
    fingerprints with the selected bundle -- live network/GitHub access
    that CYAX-0168 G1 core machinery does not perform. This module defines
    only the fail-closed decision contract; a caller injects the actual
    reacquisition mechanism (and in G1 test/CI contexts, a fake).
    """

    def current_source_revision_id(self, source_revision: c.SourceRevision) -> str | None:
        """Return the source's current ``source_revision_id`` if it can be
        determined, or ``None`` if the source is unavailable/unverifiable."""

        raise NotImplementedError


def assess_freshness(
    bundle: SnapshotBundle,
    refresher: SourceRefresher,
) -> str:
    """``'fresh'``, ``'stale'``, or ``'unknown'``.

    Any change, unavailable source, or unverifiable authority record fails
    closed to ``'stale'``/``'unknown'``; only a clean reacquisition of every
    referenced source revision, matching exactly, is ``'fresh'``.
    """

    referenced_ids = reachable_source_revision_ids(bundle.assertions)
    revision_ids = {revision.source_revision_id for revision in bundle.source_revisions}
    if referenced_ids - revision_ids:
        raise SnapshotError(
            "freshness cannot be assessed for a bundle with missing source revisions"
        )
    by_id = {r.source_revision_id: r for r in bundle.source_revisions if r.source_revision_id in referenced_ids}
    saw_unknown = False
    for source_revision_id, revision in by_id.items():
        current_id = refresher.current_source_revision_id(revision)
        if current_id is None:
            saw_unknown = True
            continue
        if current_id != source_revision_id:
            return "stale"
    if saw_unknown:
        return "unknown"
    return "fresh"


def fresh_against_current_sources(bundle: SnapshotBundle, refresher: SourceRefresher) -> str:
    """Named contract alias for :func:`assess_freshness`."""

    return assess_freshness(bundle, refresher)
