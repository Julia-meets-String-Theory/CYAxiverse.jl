"""Generate immutable general-``L`` source witnesses from KS Parquet rows.

Read the frozen KS mirror, enumerate paper-equivalent FRST classes, evaluate
the exact ``(L, t, lambda_f)`` path, and publish matching source and terminal
ledger JSONL inputs for the bounded witness driver. This module does not
select a population, write the production database, or promote the result
beyond ``production_gate=not_validated``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

import inherited_orientifold_candidates as orientifold
from build_orientifold_axion_database import exact_action_h21_diagnostic
from generate_geometric_data_multitriangulation import (
    extract_topology,
    load_mirror_polytopes,
)
from glimmers_raw_frst import (
    compute_polytope_id,
    compute_polytope_normal_form_id,
    compute_triangulation_hash,
)
from orientifold_terminal_ledger import TerminalLedgerWriter, read_terminal_ledger
from reproduce_fuzzy_axions_h11_4 import _frst_classes
from run_general_l_action_replacement_bounded import (
    CAPS,
    CANDIDATE_SCHEMA,
    GLOBAL_LIMITS,
    REQUIRED_SOURCE_FILES,
    action_digest,
    build_witness_record,
    canonical_bytes,
    input_manifest_digest,
    load_json,
    sha256_json,
)


SOURCE_GENERATOR_SCHEMA = "cyaxiverse-general-l-action-source-generation-1.0"
SOURCE_INPUT_SCHEMA = "cyaxiverse-general-l-action-replacement-input-1.0"
H11_VALUES = (2, 3, 4, 5)
SOURCE_PARTITIONS = tuple(range(5, 14))
SELECTION_ROUTE = "paper_equivalent_frst_class_general_l_exact_action"
COUNTING_UNIT = "favorable CY FRST class keyed by polytope_id::frst_hash"
EXACT_H21_UNAVAILABLE_STATUS = "exact_action_h21_evidence_unavailable"
CHECKPOINT_SCHEMA = "cyaxiverse-general-l-action-source-checkpoint-1.0"
EXPECTED_COMPLETE_COUNTS = {
    2: {"favorable_polytopes": 36, "frst_classes": 36},
    3: {"favorable_polytopes": 243, "frst_classes": 274},
    4: {"favorable_polytopes": 1185, "frst_classes": 1760},
    5: {"favorable_polytopes": 4897, "frst_classes": 11713},
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zstd_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Compress JSONL rows incrementally with zstd level 19."""
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd is required for source JSONL artifacts")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite source artifact: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        process = subprocess.Popen(
            [zstd, "-19", "-q", "-c"],
            stdin=subprocess.PIPE,
            stdout=stream,
        )
        try:
            assert process.stdin is not None
            for row in rows:
                process.stdin.write(canonical_bytes(_jsonable(row)) + b"\n")
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("zstd failed while compressing source JSONL")
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            process.kill()
            process.wait()
            raise
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite source artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _zstd_json(path: Path, value: Mapping[str, Any]) -> None:
    _zstd_jsonl(path, [value])


def _zstd_file(path: Path, output: Path) -> None:
    """Compress an existing byte stream without buffering it in Python."""
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd is required for source artifacts")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite source artifact: {output}")
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            with path.open("rb") as source:
                subprocess.run(
                    [zstd, "-19", "-q", "-c"],
                    stdin=source,
                    stdout=stream,
                    check=True,
                )
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite source artifact: {output}")
    finally:
        temporary.unlink(missing_ok=True)


def _partition_manifest(parquet_dir: str | os.PathLike[str]) -> dict[str, Any]:
    directory = Path(parquet_dir).expanduser().resolve()
    if not directory.is_dir():
        raise RuntimeError(f"Parquet mirror directory does not exist: {directory}")
    paths = []
    for partition in SOURCE_PARTITIONS:
        path = directory / f"polytopes-4d-{partition:02d}-vertices.parquet"
        if not path.is_file():
            raise RuntimeError(f"missing frozen source partition: {path}")
        paths.append(path)
    return {
        "schema_version": "cyaxiverse-parquet-source-partition-manifest-1.0",
        "status": "complete",
        "directory": str(directory),
        "pattern": "polytopes-4d-*-vertices.parquet",
        "partitions": [
            {
                "partition": partition,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for partition, path in zip(SOURCE_PARTITIONS, paths)
        ],
    }


def _frst_sort_key(triangulation: Any) -> tuple[str, list[list[int]]]:
    simplices = np.asarray(triangulation.simplices(), dtype=int)
    return compute_triangulation_hash(simplices), simplices.tolist()


def ordered_frst_classes(poly: Any) -> list[Any]:
    """Return paper-equivalent FRST representatives in deterministic order."""
    _, representatives = _frst_classes(poly)
    return sorted(representatives, key=_frst_sort_key)


def _topology_for_class(poly: Any, triangulation: Any) -> dict[str, Any]:
    cy = triangulation.get_cy()
    topology = dict(extract_topology(cy, triangulation))
    triangulation_cones = orientifold._triangulation_cones(poly, triangulation)
    topology["fixed_surface_n_s"] = orientifold.identity_fixed_surface_n_s_table(
        triangulation_cones, triangulation
    )
    topology["compute_general_fixed_surface_n_s"] = True
    topology["non_smooth_facet_dual_vertices"] = (
        orientifold.facets_with_non_smooth_cones(poly, triangulation)
    )
    return topology


def _has_complete_action(record: Mapping[str, Any]) -> bool:
    return all(
        record.get(field) is not None
        for field in ("lattice_matrix", "torus_shift", "lambda_f")
    )


def _ensure_terminal_evidence(record: dict[str, Any]) -> None:
    if "h11_parity" not in record:
        if record.get("h11_plus") is not None and record.get("h11_minus") is not None:
            record["h11_parity"] = {
                "h11_plus": int(record["h11_plus"]),
                "h11_minus": int(record["h11_minus"]),
            }
        else:
            record["h11_parity"] = {
                "status": "unavailable",
                "reason": "terminal record did not reach H2 parity extraction",
            }
    if "fixed_component_evidence" not in record:
        legacy = {
            field: record.get(field)
            for field in (
                "fixed_point_components",
                "fixed_point_set",
                "smoothness",
            )
        }
        if all(value is not None for value in legacy.values()):
            legacy["fixed_surface_n_s_evidence"] = record.get(
                "fixed_surface_n_s_evidence"
            )
            record["fixed_component_evidence"] = legacy
        else:
            record["fixed_component_evidence"] = {
                "status": "not_evaluated",
                "reason": "terminal record did not reach fixed-component evaluation",
            }


def _structural_failure_record(status: str, reason: str, *, stage: str) -> dict[str, Any]:
    return {
        "record_kind": "lattice_matrix_search_summary",
        "terminal_status": status,
        "terminal_reason": reason,
        "terminal_reason_code": "source_generation_failure",
        "source_failure_stage": stage,
        "candidate_id": None,
        "matrix_id": None,
        "lambda_f": None,
        "torus_shift": None,
    }


def _terminal_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    evidence = {
        "terminal_status": record.get("terminal_status"),
        "terminal_reason": record.get("terminal_reason"),
        "terminal_reason_code": record.get("terminal_reason_code"),
        "record_kind": record.get("record_kind"),
        "h11_parity": record.get("h11_parity"),
        "fixed_component_evidence": record.get("fixed_component_evidence"),
    }
    if "exact_action_h21_evidence" in record:
        evidence["exact_action_h21_evidence"] = record[
            "exact_action_h21_evidence"
        ]
    return evidence


def _decorate_record(
    raw_record: Mapping[str, Any],
    *,
    poly: Any,
    triangulation: Any,
    h11: int,
    source_row: int,
    source_provenance: Mapping[str, Any],
    polytope_id: str,
    polytope_normal_form_id: str,
    frst_hash: str,
    frst_class_index: int,
    exact_diagnostic: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    record = _jsonable(dict(raw_record))
    if record.get("polytope_id") not in (None, polytope_id):
        raise RuntimeError("enumerator polytope_id disagrees with source identity")
    if record.get("frst_hash") not in (None, frst_hash):
        raise RuntimeError("enumerator frst_hash disagrees with source identity")
    record.update(
        {
            "schema_version": CANDIDATE_SCHEMA,
            "h11": int(h11),
            "source_row": int(source_row),
            "source_partition": source_provenance["parquet_partition"],
            "source_provenance": dict(source_provenance),
            "polytope_id": polytope_id,
            "polytope_normal_form_id": polytope_normal_form_id,
            "frst_hash": frst_hash,
            "frst_class_index": int(frst_class_index),
        }
    )
    if record.get("record_kind") == "candidate" and not _has_complete_action(record):
        record["source_record_kind"] = "candidate"
        record["source_terminal_identity"] = {
            "candidate_id": record.get("candidate_id"),
            "matrix_id": record.get("matrix_id"),
            "attempt_kind": record.get("attempt_kind"),
        }
        record["record_kind"] = "lattice_matrix_search_summary"
        record["candidate_id"] = None
        record["matrix_id"] = None
        record["lambda_f"] = None
        record["torus_shift"] = None
    if (
        record.get("record_kind") == "candidate"
        and _has_complete_action(record)
        and record.get("terminal_status") == "accepted_verified_orientifold"
        and record.get("lambda_f") == 1
    ):
        action = {
            key: record[key]
            for key in ("lattice_matrix", "torus_shift", "lambda_f")
        }
        action["action_digest"] = action_digest(action)
        record["action_digest"] = action["action_digest"]
        try:
            evidence = _jsonable(exact_diagnostic(poly, triangulation, record))
        except Exception as exc:
            evidence = {
                "schema_version": "cyaxiverse-exact-action-hodge-evidence-3.0",
                "status": "unavailable",
                "action_digest": None,
                "reason": f"exact diagnostic failed: {type(exc).__name__}: {exc}",
            }
        if evidence.get("action_digest") != action["action_digest"]:
            evidence = {
                **evidence,
                "status": "unavailable",
                "reason": "exact-action diagnostic digest disagrees with source action",
                "expected_action_digest": action["action_digest"],
            }
        record["exact_action_h21_evidence"] = evidence
        if evidence.get("status") == "validated" and evidence.get("h21_plus") != 0:
            record["source_terminal_status"] = record["terminal_status"]
            record["source_terminal_reason"] = record.get("terminal_reason")
            record["terminal_status"] = "h21_plus_nonzero"
            record["terminal_reason_code"] = "h21_plus_nonzero"
            record["terminal_reason"] = (
                "exact action Hodge diagnostic returned "
                f"h21_plus={evidence.get('h21_plus')}"
            )
        elif evidence.get("status") != "validated":
            record["source_terminal_status"] = record["terminal_status"]
            record["source_terminal_reason"] = record.get("terminal_reason")
            record["terminal_status"] = EXACT_H21_UNAVAILABLE_STATUS
            record["terminal_reason_code"] = EXACT_H21_UNAVAILABLE_STATUS
            record["terminal_reason"] = evidence.get(
                "reason", "exact action Hodge evidence is unavailable"
            )
    _ensure_terminal_evidence(record)
    action = None
    if record.get("record_kind") == "candidate":
        action = {
            key: record[key]
            for key in ("lattice_matrix", "torus_shift", "lambda_f")
        }
        if record.get("action_digest") is not None:
            action["action_digest"] = record["action_digest"]
        action["action_digest"] = action_digest(action)
        if record.get("action_digest") not in (None, action["action_digest"]):
            raise RuntimeError("enumerator action_digest disagrees with exact action")
        record["action_digest"] = action["action_digest"]
    if record.get("terminal_status") == "accepted_verified_orientifold":
        record["accepted_witness"] = {
            "candidate_id": record["candidate_id"],
            "matrix_id": record["matrix_id"],
            "lambda_f": record["lambda_f"],
            "torus_shift": record["torus_shift"],
            "fixed_component_evidence": record["fixed_component_evidence"],
        }
    else:
        record.setdefault("accepted_witness", None)
    witness = build_witness_record(record, action, _terminal_evidence(record))
    return witness


class CandidateSinkContractError(RuntimeError):
    """Raise when an enumerator returns a candidate outside its record sink."""


def _canonical_record_counter(records: Iterable[Mapping[str, Any]]) -> Counter[bytes]:
    counter: Counter[bytes] = Counter()
    for record in records:
        if not isinstance(record, Mapping):
            raise CandidateSinkContractError(
                "candidate enumerator returned a non-object record"
            )
        counter[canonical_bytes(_jsonable(record))] += 1
    return counter


def _records_from_enumerator(
    returned: Iterable[Mapping[str, Any]] | None,
    emitted: list[dict[str, Any]],
) -> list[Mapping[str, Any]]:
    """Retain all terminal rows and reject candidates bypassing ``record_sink``."""
    returned_records = list(returned or [])
    remaining_emitted = _canonical_record_counter(emitted)
    raw_records: list[Mapping[str, Any]] = list(emitted)
    for record in returned_records:
        if not isinstance(record, Mapping):
            raise CandidateSinkContractError(
                "candidate enumerator returned a non-object record"
            )
        digest = canonical_bytes(_jsonable(record))
        if record.get("record_kind") == "candidate":
            if remaining_emitted[digest] <= 0:
                raise CandidateSinkContractError(
                    "candidate enumerator returned a candidate not emitted "
                    "through record_sink"
                )
            remaining_emitted[digest] -= 1
        elif remaining_emitted[digest] > 0:
            # The default enumerator returns matrix-success rows as well as
            # candidates, while the sink already emitted those rows.
            remaining_emitted[digest] -= 1
        else:
            # Search summaries and terminal matrix failures are returned but
            # are not necessarily sent through the candidate sink. Preserve
            # them so terminal accounting remains complete.
            raw_records.append(record)
    return raw_records


def _iter_class_records(
    poly: Any,
    *,
    h11: int,
    source_row: int,
    source_provenance: Mapping[str, Any],
    exact_diagnostic: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]],
    frst_classifier: Callable[[Any], list[Any]],
    candidate_enumerator: Callable[..., list[dict[str, Any]]],
    before_class: Callable[[int, str, str], bool] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield one decorated, sorted terminal batch for each FRST class."""
    polytope_id = compute_polytope_id(np.asarray(poly.points(), dtype=int))
    polytope_normal_form_id = compute_polytope_normal_form_id(
        np.asarray(poly.normal_form(), dtype=int)
    )
    try:
        classes = sorted(frst_classifier(poly), key=_frst_sort_key)
    except Exception as exc:
        failed_hash = (
            "frst-classification-failure:"
            f"{sha256_json({'source_row': source_row, 'polytope_id': polytope_id})}"
        )
        raw = _structural_failure_record(
            "numerical_geometry_failure",
            f"FRST class enumeration failed: {type(exc).__name__}: {exc}",
            stage="frst_classification",
        )
        yield [
            _decorate_record(
                raw,
                poly=poly,
                triangulation=None,
                h11=h11,
                source_row=source_row,
                source_provenance=source_provenance,
                polytope_id=polytope_id,
                polytope_normal_form_id=polytope_normal_form_id,
                frst_hash=failed_hash,
                frst_class_index=0,
                exact_diagnostic=exact_diagnostic,
            )
        ]
        return
    for class_index, triangulation in enumerate(classes):
        frst_hash = compute_triangulation_hash(
            np.asarray(triangulation.simplices(), dtype=int)
        )
        if before_class is not None and before_class(
            class_index, frst_hash, polytope_id
        ):
            continue
        sink: list[dict[str, Any]] = []
        try:
            topology = _topology_for_class(poly, triangulation)
            returned = candidate_enumerator(
                poly,
                triangulation,
                topology,
                record_sink=sink.append,
            )
            raw_records = _records_from_enumerator(returned, sink)
        except CandidateSinkContractError:
            raise
        except Exception as exc:
            raw_records = list(sink)
            raw_records.append(
                _structural_failure_record(
                    "numerical_geometry_failure",
                    f"general-L class evaluation failed: {type(exc).__name__}: {exc}",
                    stage="class_evaluation",
                )
            )
        decorated = [
            _decorate_record(
                raw_record,
                poly=poly,
                triangulation=triangulation,
                h11=h11,
                source_row=source_row,
                source_provenance=source_provenance,
                polytope_id=polytope_id,
                polytope_normal_form_id=polytope_normal_form_id,
                frst_hash=frst_hash,
                frst_class_index=class_index,
                exact_diagnostic=exact_diagnostic,
            )
            for raw_record in raw_records
        ]
        decorated.sort(
            key=lambda row: (
                row.get("action_digest") or "",
                row["terminal_record_identity"],
                row["terminal_record_digest"],
            )
        )
        yield decorated


def _class_records(
    poly: Any,
    *,
    h11: int,
    source_row: int,
    source_provenance: Mapping[str, Any],
    exact_diagnostic: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]],
    frst_classifier: Callable[[Any], list[Any]],
    candidate_enumerator: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for class_records in _iter_class_records(
        poly,
        h11=h11,
        source_row=source_row,
        source_provenance=source_provenance,
        exact_diagnostic=exact_diagnostic,
        frst_classifier=frst_classifier,
        candidate_enumerator=candidate_enumerator,
    ):
        result.extend(class_records)
    return result


def _repository_revision(root: Path) -> dict[str, str | None]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "source_commit": command("rev-parse", "HEAD"),
        "tree_sha256": command("rev-parse", "HEAD^{tree}"),
        "working_tree_diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        ).hexdigest(),
    }


def _input_entry(
    path: Path,
    *,
    h11: int,
    role: str,
    published_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "h11": int(h11),
        "role": role,
        "path": str((published_path or path).resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "file_type": "jsonl.zst",
        "source_row_or_partition_identity": (
            f"h11={int(h11)}::{role}::polytope_id::frst_hash"
        ),
        "selection_route": SELECTION_ROUTE,
        "counting_unit": COUNTING_UNIT,
    }


def _runtime_versions() -> dict[str, str]:
    def distribution(names: tuple[str, ...]) -> str:
        for name in names:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return "unavailable"

    julia_version = "unavailable"
    try:
        julia_version = subprocess.run(
            ["julia", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "python_version": platform.python_version(),
        "julia_version": julia_version,
        "cytools_version": distribution(("cytools",)),
        "numpy_version": distribution(("numpy",)),
        "pyarrow_version": distribution(("pyarrow",)),
    }


def _source_file_digests(root: Path) -> dict[str, str]:
    names = set(REQUIRED_SOURCE_FILES)
    names.add("scripts/generate_general_l_action_source.py")
    result = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"required source file is missing: {path}")
        result[name] = _sha256_file(path)
    return result


def _write_plain_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Create one canonical JSONL file without retaining its rows."""
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite checkpoint segment: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_bytes(_jsonable(row)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _link_create_only(source: Path, destination: Path) -> None:
    """Link one completed temporary file without replacing an existing file."""
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite checkpoint segment: {destination}")
    try:
        os.link(source, destination)
    finally:
        source.unlink(missing_ok=True)


def _files_equal(left: Path, right: Path) -> bool:
    """Compare two files incrementally."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_block = left_stream.read(1024 * 1024)
            right_block = right_stream.read(1024 * 1024)
            if left_block != right_block:
                return False
            if not left_block:
                return True


def _checkpoint_segment_paths(
    checkpoint_root: Path,
    h11: int,
    source_row: int,
    frst_class_index: int,
    frst_hash: str,
) -> tuple[Path, Path, Path]:
    class_id = sha256_json({"frst_hash": frst_hash})[:16]
    stem = f"class-{source_row:08d}-{frst_class_index:05d}-{class_id}"
    directory = checkpoint_root / f"h11-{h11:03d}"
    return (
        directory / f"{stem}.source.jsonl",
        directory / f"{stem}.ledger.jsonl",
        directory / f"{stem}.metadata.json.zst",
    )


def _write_checkpoint_segment(
    source_path: Path, ledger_path: Path, metadata_path: Path,
    rows: list[Mapping[str, Any]], *, partition_manifest: Mapping[str, Any],
    source_commit: str, h11: int, source_row: int, frst_class_index: int,
    frst_hash: str,
) -> None:
    """Persist one complete class as a resumable source/ledger pair."""
    if any(os.path.lexists(path) for path in (source_path, ledger_path, metadata_path)):
        raise FileExistsError("checkpoint class segment is already present")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_fd, source_name = tempfile.mkstemp(
        prefix=f".{source_path.name}.", dir=source_path.parent
    )
    os.close(source_fd)
    source_temporary = Path(source_name)
    source_temporary.unlink()
    ledger_temporary = source_path.with_name(
        f".{ledger_path.name}.tmp-{os.getpid()}"
    )
    try:
        _write_plain_jsonl(source_temporary, rows)
        writer = TerminalLedgerWriter(
            ledger_temporary,
            provenance={
                "source_commit": source_commit,
                "git_dirty": True,
                "working_tree_identity": {"diff_sha256": "checkpoint-segment"},
                "runtime_versions": {"python": "source-generator"},
                "input_partition_manifest": dict(partition_manifest),
            },
            metadata={"checkpoint_segment": True},
        )
        try:
            for row in rows:
                writer.write(dict(row))
            writer.close()
        except Exception:
            writer.abort()
            raise
        if not _files_equal(source_temporary, ledger_temporary):
            raise RuntimeError("terminal ledger normalization changed source rows")
        metadata = {
            "schema": CHECKPOINT_SCHEMA,
            "h11": int(h11),
            "source_row": int(source_row),
            "frst_class_index": int(frst_class_index),
            "frst_hash": str(frst_hash),
            "polytope_id": str(rows[0]["polytope_id"]),
            "source_commit": source_commit,
            "source": {
                "path": str(source_path),
                "size_bytes": source_temporary.stat().st_size,
                "sha256": _sha256_file(source_temporary),
            },
            "ledger": {
                "path": str(ledger_path),
                "size_bytes": ledger_temporary.stat().st_size,
                "sha256": _sha256_file(ledger_temporary),
            },
            "terminal_row_count": len(rows),
        }
        _link_create_only(source_temporary, source_path)
        _link_create_only(ledger_temporary, ledger_path)
        Path(f"{ledger_temporary}.summary.json").unlink(missing_ok=True)
        _zstd_json(metadata_path, metadata)
    finally:
        source_temporary.unlink(missing_ok=True)
        ledger_temporary.unlink(missing_ok=True)
        Path(f"{ledger_temporary}.summary.json").unlink(missing_ok=True)


def _verify_checkpoint_segment(
    source_path: Path,
    ledger_path: Path,
    metadata_path: Path,
    *,
    h11: int,
    source_row: int,
    frst_class_index: int,
    frst_hash: str,
    polytope_id: str,
    source_commit: str,
) -> None:
    """Verify a completed class segment and every immutable binding."""
    if not all(path.is_file() for path in (source_path, ledger_path, metadata_path)):
        raise RuntimeError("checkpoint class segment is incomplete")
    metadata = load_json(metadata_path)
    expected_identity = {
        "schema": CHECKPOINT_SCHEMA,
        "h11": int(h11),
        "source_row": int(source_row),
        "frst_class_index": int(frst_class_index),
        "frst_hash": str(frst_hash),
        "polytope_id": str(polytope_id),
        "source_commit": str(source_commit),
        "terminal_row_count": metadata.get("terminal_row_count"),
    }
    for key, value in expected_identity.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"checkpoint segment metadata mismatch: {key}")
    for key, path in (("source", source_path), ("ledger", ledger_path)):
        fingerprint = metadata.get(key)
        if not isinstance(fingerprint, Mapping) or fingerprint.get("path") != str(path):
            raise RuntimeError("checkpoint segment metadata mismatch")
        if (
            fingerprint.get("size_bytes") != path.stat().st_size
            or fingerprint.get("sha256") != _sha256_file(path)
        ):
            raise RuntimeError("checkpoint segment fingerprint mismatch")
    if not _files_equal(source_path, ledger_path):
        raise RuntimeError("checkpoint source and terminal ledger segments differ")


def _stream_row_summary(path: Path) -> dict[str, Any]:
    """Summarize one canonical JSONL file with bounded row memory."""
    source_rows: set[int] = set()
    polytopes: set[str] = set()
    classes: set[tuple[str, str]] = set()
    pseudo_failure_classes: set[tuple[str, str]] = set()
    record_kind_counts: Counter[str] = Counter()
    terminal_status_counts: Counter[str] = Counter()
    row_count = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise RuntimeError(f"blank checkpoint row at {path}:{line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"malformed checkpoint row at {path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise RuntimeError(f"checkpoint row is not an object: {path}:{line_number}")
            row_count += 1
            if isinstance(row.get("source_row"), int) and not isinstance(row["source_row"], bool):
                source_rows.add(int(row["source_row"]))
            polytopes.add(str(row["polytope_id"]))
            classes.add((str(row["polytope_id"]), str(row["frst_hash"])))
            if (
                str(row["frst_hash"]).startswith("frst-classification-failure:")
                or row.get("source_failure_stage") == "frst_classification"
            ):
                pseudo_failure_classes.add(
                    (str(row["polytope_id"]), str(row["frst_hash"]))
                )
            record_kind_counts[str(row["record_kind"])] += 1
            terminal_status_counts[str(row["terminal_status"])] += 1
    return {
        "source_row_count": len(source_rows),
        "terminal_row_count": row_count,
        "favorable_polytopes_seen": len(polytopes),
        "frst_classes_seen": len(classes),
        "pseudo_failure_classes": len(pseudo_failure_classes),
        "record_kind_counts": dict(sorted(record_kind_counts.items())),
        "terminal_status_counts": dict(sorted(terminal_status_counts.items())),
    }


def _concatenate_files(paths: Iterable[Path], destination: Path) -> None:
    """Concatenate checkpoint segments into one create-only raw JSONL file."""
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite source artifact: {destination}")
    with destination.open("xb") as output:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())


def _prepare_checkpoint_root(
    root: Path, binding: Mapping[str, Any]
) -> None:
    """Create or verify a resumable checkpoint root without overwriting it."""
    if os.path.lexists(root) and not root.is_dir():
        raise FileExistsError(f"checkpoint root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "checkpoint-manifest.json.zst"
    if marker.exists():
        observed = load_json(marker)
        if canonical_bytes(_jsonable(observed)) != canonical_bytes(_jsonable(binding)):
            differing = next(
                (
                    key
                    for key in sorted(set(observed) | set(binding))
                    if canonical_bytes(_jsonable(observed.get(key)))
                    != canonical_bytes(_jsonable(binding.get(key)))
                ),
                "unknown",
            )
            raise RuntimeError(f"checkpoint binding mismatch: {differing}")
        return
    unexpected = [path for path in root.iterdir() if path.name != marker.name]
    if unexpected:
        raise RuntimeError("checkpoint root contains unbound files")
    _zstd_json(marker, dict(binding))


def _current_rss_bytes() -> int:
    """Return peak resident memory in bytes on supported host platforms."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if os.uname().sysname == "Darwin" else usage * 1024)


def _tree_bytes(root: Path) -> int:
    """Return the size of regular files below one bounded work root."""
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _resource_guard(
    h11: int,
    checkpoint_root: Path,
    stage: Path,
    *,
    projected_bytes: int = 0,
) -> None:
    """Fail closed before memory or temporary/output ceilings are exceeded."""
    if _current_rss_bytes() > int(GLOBAL_LIMITS["max_rss_bytes"]):
        raise RuntimeError("resource_cap_exceeded: RSS ceiling")
    ceiling = int(CAPS[h11]["max_new_output_bytes"])
    if _tree_bytes(checkpoint_root) + _tree_bytes(stage) + int(projected_bytes) > ceiling:
        raise RuntimeError("resource_cap_exceeded: temporary/output ceiling")


def generate_source_rows(
    parquet_dir: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    h11_values: Iterable[int] = H11_VALUES,
    limit: int | None = None,
    repository_root: str | os.PathLike[str] | None = None,
    checkpoint_root: str | os.PathLike[str] | None = None,
    loader: Callable[..., list[tuple[Any, Mapping[str, Any]]]] = load_mirror_polytopes,
    frst_classifier: Callable[[Any], list[Any]] = ordered_frst_classes,
    candidate_enumerator: Callable[..., list[dict[str, Any]]] | None = None,
    exact_diagnostic: Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]] = exact_action_h21_diagnostic,
) -> dict[str, Any]:
    """Generate create-only source and terminal-ledger inputs."""
    h11_values = tuple(sorted({int(value) for value in h11_values}))
    if h11_values != H11_VALUES:
        raise ValueError("source generation requires h11 values 2, 3, 4, and 5")
    if limit is not None and (isinstance(limit, bool) or int(limit) < 1):
        raise ValueError("limit must be a positive integer when supplied")
    selected_enumerator = candidate_enumerator or orientifold.enumerate_orientifold_candidates
    output = Path(output_root).expanduser().resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite source output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partition_manifest = _partition_manifest(parquet_dir)
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parent.parent
    )
    revision = _repository_revision(root)
    if any(revision.get(key) is None for key in revision):
        raise RuntimeError("could not establish source repository revision")
    runtime_versions = _runtime_versions()
    environment = {
        name: os.environ.get(name)
        for name in (
            "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    run_scope = "complete" if limit is None else "pilot"
    checkpoint = (
        Path(checkpoint_root).expanduser().resolve()
        if checkpoint_root is not None
        else output.parent / f"{output.name}-checkpoints"
    )
    if checkpoint == output:
        raise ValueError("checkpoint root must differ from source output root")
    configuration = {
        "source_generator_schema": SOURCE_GENERATOR_SCHEMA,
        "parquet_dir": str(Path(parquet_dir).expanduser().resolve()),
        "partitions": list(SOURCE_PARTITIONS),
        "h11_values": list(H11_VALUES),
        "limit": limit,
        "run_scope": run_scope,
        "selection_route": SELECTION_ROUTE,
        "counting_unit": COUNTING_UNIT,
        "output_root": str(output),
        "checkpoint_root": str(checkpoint),
    }
    source_file_digests = _source_file_digests(root)
    configuration_digest = sha256_json(configuration)
    checkpoint_binding = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": revision["source_commit"],
        "tree_sha256": revision["tree_sha256"],
        "working_tree_diff_sha256": revision["working_tree_diff_sha256"],
        "configuration_digest": configuration_digest,
        "configuration": configuration,
        "source_file_digests": source_file_digests,
        "source_partition_manifest": partition_manifest,
        "environment_revision": sha256_json(
            {"runtime_versions": runtime_versions, "environment": environment}
        ),
        "seed": 0,
        "limits": CAPS,
        "global_limits": GLOBAL_LIMITS,
        "output_root": str(output),
        "checkpoint_root": str(checkpoint),
        "run_scope": run_scope,
    }
    _prepare_checkpoint_root(checkpoint, checkpoint_binding)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    summaries: dict[int, Any] = {}
    input_entries = []
    try:
        for h11 in H11_VALUES:
            _resource_guard(h11, checkpoint, stage)
            loader_kwargs = {
                "h11": h11,
                "limit": 10**9 if limit is None else int(limit),
                "favorable": True,
                "partitions": SOURCE_PARTITIONS,
            }
            if loader is load_mirror_polytopes:
                loader_kwargs["stream"] = True
            loaded = loader(parquet_dir, **loader_kwargs)
            source_row_count = 0
            segment_pairs: list[tuple[Path, Path]] = []
            segments_by_class: dict[
                tuple[int, int, str], tuple[Path, Path, Path]
            ] = {}
            for source_row, (poly, provenance) in enumerate(loaded, start=1):
                source_row_count = source_row
                source_provenance = dict(provenance)
                source_provenance["parquet_partition"] = Path(
                    source_provenance["parquet_file"]
                ).name
                source_provenance["parquet_row_index"] = int(
                    source_provenance["row_index"]
                )
                required_mapping = (
                    "physical_h11", "physical_h21", "mirror_h11", "mirror_h12"
                )
                if any(
                    field not in source_provenance
                    or isinstance(source_provenance[field], bool)
                    or not isinstance(source_provenance[field], (int, np.integer))
                    for field in required_mapping
                ) or (
                    int(source_provenance["physical_h11"]) != h11
                    or int(source_provenance["physical_h11"])
                    != int(source_provenance["mirror_h12"])
                    or int(source_provenance["physical_h21"])
                    != int(source_provenance["mirror_h11"])
                ):
                    raise RuntimeError(
                        "Parquet-to-physical Hodge-label mapping failed"
                    )

                def before_class(
                    class_index: int, frst_hash: str, polytope_id: str
                ) -> bool:
                    paths = _checkpoint_segment_paths(
                        checkpoint, h11, source_row, class_index, frst_hash
                    )
                    source_segment, ledger_segment, metadata_segment = paths
                    segments_by_class[(source_row, class_index, frst_hash)] = paths
                    source_exists = os.path.lexists(source_segment)
                    ledger_exists = os.path.lexists(ledger_segment)
                    metadata_exists = os.path.lexists(metadata_segment)
                    if len({source_exists, ledger_exists, metadata_exists}) != 1:
                        raise RuntimeError("checkpoint class segment is incomplete")
                    if source_exists:
                        _verify_checkpoint_segment(
                            source_segment,
                            ledger_segment,
                            metadata_segment,
                            h11=h11,
                            source_row=source_row,
                            frst_class_index=class_index,
                            frst_hash=frst_hash,
                            polytope_id=polytope_id,
                            source_commit=str(revision["source_commit"]),
                        )
                        segment_pairs.append((source_segment, ledger_segment))
                        return True
                    _resource_guard(h11, checkpoint, stage)
                    return False

                for class_records in _iter_class_records(
                    poly,
                    h11=h11,
                    source_row=source_row,
                    source_provenance=source_provenance,
                    exact_diagnostic=exact_diagnostic,
                    frst_classifier=frst_classifier,
                    candidate_enumerator=selected_enumerator,
                    before_class=before_class,
                ):
                    if not class_records:
                        raise RuntimeError(
                            f"FRST class emitted no terminal rows for h11={h11}, "
                            f"source_row={source_row}"
                        )
                    first = class_records[0]
                    class_key = (
                        source_row,
                        int(first["frst_class_index"]),
                        str(first["frst_hash"]),
                    )
                    segment_paths = segments_by_class.get(class_key)
                    if segment_paths is None:
                        segment_paths = _checkpoint_segment_paths(
                            checkpoint,
                            h11,
                            source_row,
                            int(first["frst_class_index"]),
                            str(first["frst_hash"]),
                        )
                    source_segment, ledger_segment, metadata_segment = segment_paths
                    if not os.path.lexists(source_segment):
                        projected = 4 * sum(
                            len(canonical_bytes(_jsonable(row))) + 1
                            for row in class_records
                        )
                        _resource_guard(
                            h11, checkpoint, stage, projected_bytes=projected
                        )
                        _write_checkpoint_segment(
                            source_segment,
                            ledger_segment,
                            metadata_segment,
                            class_records,
                            partition_manifest=partition_manifest,
                            source_commit=str(revision["source_commit"]),
                            h11=h11,
                            source_row=source_row,
                            frst_class_index=int(first["frst_class_index"]),
                            frst_hash=str(first["frst_hash"]),
                        )
                        segment_pairs.append((source_segment, ledger_segment))
                    _resource_guard(h11, checkpoint, stage)
            segment_pairs.sort(key=lambda pair: pair[0].name)
            if not segment_pairs:
                raise RuntimeError(f"no FRST classes were emitted for h11={h11}")
            source_raw = stage / f".h11-{h11:03d}.source-rows.jsonl"
            ledger_raw = stage / f".h11-{h11:03d}.terminal-ledger.jsonl"
            h11_checkpoint_bytes = _tree_bytes(checkpoint / f"h11-{h11:03d}")
            _resource_guard(
                h11,
                checkpoint,
                stage,
                projected_bytes=2 * h11_checkpoint_bytes,
            )
            _concatenate_files((pair[0] for pair in segment_pairs), source_raw)
            _concatenate_files((pair[1] for pair in segment_pairs), ledger_raw)
            # Keep both raw streams and both compressed streams inside the
            # declared per-h11 temporary/output ceiling before compression
            # starts.  The zstd process may retain an output frame while the
            # raw input files are still present.
            _resource_guard(
                h11,
                checkpoint,
                stage,
                projected_bytes=source_raw.stat().st_size + ledger_raw.stat().st_size,
            )
            source_name = f"h11-{h11:03d}.source-rows.jsonl.zst"
            ledger_name = f"h11-{h11:03d}.terminal-ledger.jsonl.zst"
            source_stage = stage / source_name
            ledger_stage = stage / ledger_name
            _zstd_file(source_raw, source_stage)
            _zstd_file(ledger_raw, ledger_stage)
            if source_stage.stat().st_size + ledger_stage.stat().st_size > int(
                CAPS[h11]["max_new_output_bytes"]
            ):
                raise RuntimeError("resource_cap_exceeded: published output ceiling")
            if not _files_equal(source_raw, ledger_raw):
                raise RuntimeError(
                    f"terminal ledger normalization changed source rows for h11={h11}"
                )
            source_summary = _stream_row_summary(source_raw)
            ledger_summary = _stream_row_summary(ledger_raw)
            if source_summary != ledger_summary:
                raise RuntimeError(f"source and ledger summaries differ for h11={h11}")
            expected = EXPECTED_COMPLETE_COUNTS[h11]
            completion = {
                "status": "passed" if limit is not None or (
                    source_summary["favorable_polytopes_seen"]
                    == expected["favorable_polytopes"]
                    and source_summary["frst_classes_seen"]
                    == expected["frst_classes"]
                    and source_summary["pseudo_failure_classes"] == 0
                ) else "blocked_on_evidence",
                "expected": expected if limit is None else None,
                "observed": {
                    "favorable_polytopes": source_summary["favorable_polytopes_seen"],
                    "frst_classes": source_summary["frst_classes_seen"],
                    "pseudo_failure_classes": source_summary["pseudo_failure_classes"],
                },
            }
            if completion["status"] != "passed":
                raise RuntimeError(
                    f"complete source count mismatch for h11={h11}: "
                    f"expected {expected}, observed {completion['observed']}"
                )
            input_entries.extend(
                (
                    _input_entry(
                        source_stage,
                        h11=h11,
                        role="source_rows",
                        published_path=output / source_name,
                    ),
                    _input_entry(
                        ledger_stage,
                        h11=h11,
                        role="terminal_ledger",
                        published_path=output / ledger_name,
                    ),
                )
            )
            summaries[h11] = {
                **source_summary,
                "source_row_count": source_row_count,
                "run_scope": run_scope,
                "population_completion": completion,
            }
            source_raw.unlink()
            ledger_raw.unlink()
        input_manifest = {
            "schema": SOURCE_INPUT_SCHEMA,
            "task_id": "general-l-action-replacement-bounded-run-h11-2-5",
            "program": "source-compatible inherited-orientifold general-L action validation",
            "h11_values": list(H11_VALUES),
            "selection_route": SELECTION_ROUTE,
            "counting_unit": COUNTING_UNIT,
            "action_conventions": "exact (L,t,lambda_f), contragredient L, reduced common-denominator torus shifts",
            "terminal_conventions": "schema-1.2 terminal identity and complete-record digest",
            "limits": CAPS,
            "global_limits": GLOBAL_LIMITS,
            "seed": 0,
            "dependency_manifest_sha256": _sha256_file(root / "Manifest.toml"),
            "project_toml_sha256": _sha256_file(root / "Project.toml"),
            "manifest_toml_sha256": _sha256_file(root / "Manifest.toml"),
            "runtime_versions": runtime_versions,
            "relevant_environment_variables": environment,
            "environment_revision": sha256_json(
                {"runtime_versions": runtime_versions, "environment": environment}
            ),
            "source_file_digests": source_file_digests,
            "source_commit": revision["source_commit"],
            "tree_sha256": revision["tree_sha256"],
            "working_tree_diff_sha256": revision["working_tree_diff_sha256"],
            "configuration_digest": configuration_digest,
            "configuration": configuration,
            "run_scope": run_scope,
            "population_completion": {
                "status": "passed" if limit is None else "not_required_for_pilot",
                "expected": EXPECTED_COMPLETE_COUNTS if limit is None else None,
                "observed": {
                    str(h11): summaries[h11]["population_completion"]["observed"]
                    for h11 in H11_VALUES
                },
            },
            "output_root": str(output),
            "checkpoint_root": str(checkpoint),
            "source_partition_manifest": partition_manifest,
            "production_gate": "not_validated",
            "scale_status": "not_applicable",
            "no_overwrite": True,
            "inputs": input_entries,
        }
        entry_bindings = {
            "source_commit": revision["source_commit"],
            "tree_sha256": revision["tree_sha256"],
            "working_tree_diff_sha256": revision["working_tree_diff_sha256"],
            "environment_revision": input_manifest["environment_revision"],
            "configuration_digest": configuration_digest,
            "seed": 0,
            "limits": CAPS,
            "global_limits": GLOBAL_LIMITS,
            "output_root": str(output),
            "selection_route": SELECTION_ROUTE,
            "counting_unit": COUNTING_UNIT,
        }
        for entry in input_entries:
            entry.update(entry_bindings)
        input_manifest["input_manifest_sha256"] = input_manifest_digest(input_manifest)
        generation_manifest = {
            "schema": SOURCE_GENERATOR_SCHEMA,
            "source_input_manifest": input_manifest,
            "source_partition_manifest": partition_manifest,
            "h11_summaries": summaries,
            "output_root": str(output),
            "checkpoint_root": str(checkpoint),
            "run_scope": run_scope,
            "population_completion": {
                "status": "passed" if limit is None else "not_required_for_pilot",
                "expected": EXPECTED_COMPLETE_COUNTS if limit is None else None,
            },
            "production_gate": "not_validated",
            "scale_status": "not_applicable",
        }
        generation_manifest["artifact_count"] = len(H11_VALUES) * 2 + 3
        _zstd_json(stage / "input-manifest.json.zst", input_manifest)
        _zstd_json(stage / "source-generation-manifest.json.zst", generation_manifest)
        artifacts = sorted(
            path for path in stage.iterdir() if path.name != "SHA256SUMS.txt"
        )
        (stage / "SHA256SUMS.txt").write_text(
            "".join(f"{_sha256_file(path)}  {path.name}\n" for path in artifacts),
            encoding="ascii",
        )
        try:
            # The final stage contains every published h11 artifact plus the
            # manifests and checksum file.  Check the largest declared
            # ceiling before making the output directory visible.
            _resource_guard(5, checkpoint, stage)
            os.mkdir(output)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite source output root: {output}")
        for artifact in stage.iterdir():
            os.link(artifact, output / artifact.name)
        shutil.rmtree(stage)
        generation_manifest["output_root"] = str(output)
        generation_manifest["artifact_count"] = len(artifacts) + 1
        return generation_manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = generate_source_rows(
        args.parquet_dir,
        args.output_root,
        limit=args.limit,
        checkpoint_root=args.checkpoint_root,
    )
    print(json.dumps(_jsonable(result), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
