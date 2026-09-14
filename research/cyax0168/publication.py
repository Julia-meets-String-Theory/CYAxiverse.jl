#!/usr/bin/env python3
"""Fail-closed atomic publication for CYAX-0168 snapshots and source bundles.

The publisher is intentionally backend-neutral.  It writes a complete
candidate beside the destination, validates the candidate, writes the final
``build_complete`` marker last, and atomically renames it into place.  A
failure leaves an already-published destination untouched and a temporary
candidate that :mod:`snapshot` refuses to open.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Mapping

try:  # Same-directory test/import convention and package imports.
    import snapshot as s
except ImportError:  # pragma: no cover - exercised by package importers
    from . import snapshot as s


class PublicationError(s.SnapshotError):
    """Raised when an atomic publication cannot be completed safely."""


PublicationHook = Callable[[str, Path], None]


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    # macOS and POSIX support directory fsync; a platform that does not
    # support it is not silently treated as a durable publication.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:  # pragma: no cover - platform-specific
        raise PublicationError(f"cannot open directory for fsync: {path}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:  # pragma: no cover - platform-specific
        raise PublicationError(f"directory fsync is unsupported: {path}") from exc
    finally:
        os.close(fd)


def _canonical_manifest_bytes(record: Mapping[str, object]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _stage_directory(
    destination: Path,
    payloads: Mapping[str, bytes],
    manifest_false: Mapping[str, object],
    manifest_true: Mapping[str, object],
    *,
    manifest_name: str,
    validate: Callable[[Path], object],
    hook: PublicationHook | None,
) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PublicationError(f"publication destination already exists: {destination}")
    temp_path = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        if hook:
            hook("temporary_directory_created", temp_path)
        for name, value in payloads.items():
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise PublicationError(f"invalid publication payload path: {name!r}")
            target = temp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes(target, value)
        marker = temp_path / manifest_name
        _write_bytes(marker, _canonical_manifest_bytes(manifest_false))
        if hook:
            hook("payloads_fsynced", temp_path)
        # Object-store payloads live below nested directories.  Fsync each
        # directory before the root so the rename cannot publish a directory
        # entry whose child metadata was not durable.
        nested_dirs = {
            target.parent
            for target in (temp_path / name for name in payloads)
            if target.parent != temp_path
        }
        for directory in sorted(nested_dirs, key=lambda value: len(value.parts), reverse=True):
            _fsync_dir(directory)
        _fsync_dir(temp_path)
        validate(temp_path)
        if hook:
            hook("candidate_validated", temp_path)
        # Marker is deliberately rewritten only after every payload has been
        # validated.  A crash before this point leaves an unopenable candidate.
        marker.unlink()
        _write_bytes(marker, _canonical_manifest_bytes(manifest_true))
        if hook:
            hook("final_manifest_fsynced", temp_path)
        _fsync_dir(temp_path)
        if destination.exists():
            raise PublicationError(f"publication destination appeared during build: {destination}")
        os.replace(temp_path, destination)
        _fsync_dir(destination.parent)
        if hook:
            hook("renamed", destination)
        return destination
    except BaseException:
        # Cleanup is best effort.  Keeping an interrupted temp tree is safe
        # (readers reject it) but removing it avoids accumulating stale data.
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def publish_snapshot(
    destination: Path,
    manifest: s.SnapshotManifest,
    payloads: Mapping[str, bytes],
    *,
    hook: PublicationHook | None = None,
) -> Path:
    """Publish a validated immutable snapshot atomically.

    ``destination`` is the content-addressed final directory.  The supplied
    manifest must have ``build_complete=False``; callers cannot publish a
    pre-marked or mismatched candidate.  The function returns the final path
    only after rename and parent-directory fsync succeed.
    """

    if manifest.build_complete:
        raise PublicationError("candidate snapshot manifest must not be pre-marked complete")
    required = set(s.SNAPSHOT_PAYLOAD_FILES)
    if set(payloads) != required:
        raise PublicationError("snapshot payload set is incomplete or contains unknown files")
    try:
        registry = (
            s.c.validate_claim_key_registry(manifest.claim_key_registry)
            if manifest.claim_key_registry
            else None
        )
        s.validate_snapshot_records(
            [s.entity_from_json_record(r) for r in s.decode_jsonl(payloads["entities.jsonl"])],
            [s.literal_from_json_record(r) for r in s.decode_jsonl(payloads["literals.jsonl"])],
            [s.source_revision_from_json_record(r) for r in s.decode_jsonl(payloads["source_revisions.jsonl"])],
            [s.assertion_from_json_record(r) for r in s.decode_jsonl(payloads["assertions.jsonl"])],
            claim_key_registry=registry,
        )
    except Exception as exc:
        raise PublicationError(f"snapshot candidate failed record validation: {exc}") from exc
    expected_record = s.manifest_to_json(manifest)
    complete_manifest = dataclasses.replace(manifest, build_complete=True)
    final_record = s.manifest_to_json(complete_manifest)
    # Validate the candidate from its staged files, not merely the in-memory
    # records, so the atomic protocol catches encoding/checksum drift.
    def validate(path: Path) -> object:
        # ``read_snapshot_dir`` requires the final marker.  For candidate
        # validation temporarily use the complete marker in memory by checking
        # all fields directly; the semantic and physical checks are repeated
        # after publication by the caller/open path.
        for name in s.SNAPSHOT_PAYLOAD_FILES:
            raw = (path / name).read_bytes()
            prefix = s._PAYLOAD_KEY_PREFIX[name]
            if len(raw) != expected_record[f"{prefix}_byte_count"]:
                raise PublicationError(f"staged {name} byte count mismatch")
            if s.c.sha256_hex(raw) != expected_record[f"{prefix}_sha256"]:
                raise PublicationError(f"staged {name} checksum mismatch")
        return True

    result = _stage_directory(
        destination,
        payloads,
        expected_record,
        final_record,
        manifest_name="manifest.json",
        validate=validate,
        hook=hook,
    )
    try:
        s.read_snapshot_dir(result)
    except Exception as exc:
        shutil.rmtree(result, ignore_errors=True)
        raise PublicationError(f"published snapshot failed reopen validation: {exc}") from exc
    return result


def publish_source_bundle(
    destination: Path,
    manifest: s.SourceBundleManifest,
    payloads: Mapping[str, bytes],
    *,
    hook: PublicationHook | None = None,
) -> Path:
    """Publish a content-addressed source bundle with the same marker order."""

    if manifest.build_complete:
        raise PublicationError("candidate source bundle manifest must not be pre-marked complete")
    required = set(s.SOURCE_BUNDLE_PAYLOAD_FILES)
    object_names = {
        name for name in payloads if name.startswith("objects/sha256/")
    }
    if {name for name in payloads if name in required} != required:
        raise PublicationError("source bundle payload set is incomplete")
    expected_objects = {
        f"objects/sha256/{digest[:2]}/{digest[2:]}" for digest, _ in manifest.object_digests
    }
    if object_names != expected_objects:
        raise PublicationError("source bundle object set does not match manifest")
    try:
        # build_source_bundle returns a false marker and its payloads.  Reuse
        # the in-memory validator after flipping only the marker state.
        complete_manifest = dataclasses.replace(manifest, build_complete=True)
        s.validate_source_bundle(complete_manifest, payloads)
    except Exception as exc:
        raise PublicationError(f"source bundle candidate failed validation: {exc}") from exc
    false_record = s.source_bundle_manifest_to_json(manifest)
    true_record = s.source_bundle_manifest_to_json(dataclasses.replace(manifest, build_complete=True))

    def validate(path: Path) -> object:
        staged = {name: (path / name).read_bytes() for name in required}
        staged.update(
            {
                name: (path / name).read_bytes()
                for name in expected_objects
            }
        )
        complete = dataclasses.replace(manifest, build_complete=True)
        s.validate_source_bundle(complete, staged)
        return True

    result = _stage_directory(
        destination,
        payloads,
        false_record,
        true_record,
        manifest_name="bundle_manifest.json",
        validate=validate,
        hook=hook,
    )
    try:
        s.read_source_bundle_dir(result)
    except Exception as exc:
        shutil.rmtree(result, ignore_errors=True)
        raise PublicationError(f"published source bundle failed reopen validation: {exc}") from exc
    return result


def publish_delta(
    destination: Path,
    base: s.SnapshotBundle,
    delta: object,
    *,
    hook: PublicationHook | None = None,
) -> Path:
    """Apply and publish a canonical delta using :mod:`update`.

    Importing lazily avoids a module cycle while keeping the public helper
    convenient for crash/recovery tests.
    """

    try:
        import update
    except ImportError:  # pragma: no cover - exercised by package imports
        from . import update

    candidate = update.apply_delta(base, delta)
    # Reconstruct canonical payloads from the candidate bundle.  The manifest
    # is already complete only as an in-memory semantic result; clear the bit
    # before entering publication.
    manifest, payloads = s.build_manifest(
        entities=candidate.entities,
        literals=candidate.literals,
        source_revisions=candidate.source_revisions,
        assertions=candidate.assertions,
        semantic_source_bundle_projection_checksum=candidate.manifest.semantic_source_bundle_projection_checksum,
        authority_rule_version=candidate.manifest.authority_rule_version,
        semantic_evaluator_rule_version=candidate.manifest.semantic_evaluator_rule_version,
        source_bundle_id=candidate.manifest.source_bundle_id,
        assertion_compiler_version=candidate.manifest.assertion_compiler_version,
        curator_version=candidate.manifest.curator_version,
        validator_version=candidate.manifest.validator_version,
        context_compiler_version=candidate.manifest.context_compiler_version,
        schema_version=candidate.manifest.schema_version,
        built_at=candidate.manifest.built_at,
        claim_key_registry=candidate.manifest.claim_key_registry,
    )
    return publish_snapshot(destination, manifest, payloads, hook=hook)


# The protocol is intentionally exposed under descriptive aliases so callers
# do not need to know whether the published object is a snapshot or source
# bundle when writing generic crash/recovery tests.
atomic_publish_snapshot = publish_snapshot
atomic_publish_source_bundle = publish_source_bundle
open_snapshot = s.read_snapshot_dir
open_source_bundle = s.read_source_bundle_dir
