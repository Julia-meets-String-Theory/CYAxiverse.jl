#!/usr/bin/env python3
"""Enrich authoritative orientifold ledger rows with immutable source joins.

The merged h11=4/5 ledgers retain an accepted witness, but not the source
partition row, full lattice-point configuration, selected FRST, or the
matrix represented by ``matrix_id``.  This bounded runner reconstructs those
identities from the durable KS mirror and CYTools.  It never manufactures a
population certificate: missing formula or refined-GLSM evidence is retained
as a terminal ``mpcp_certificate_unavailable`` row.

The checkpoint and optional output are atomic zstd level-19 JSONL artifacts.
No HDF5/database writer is imported or called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from glimmers_raw_frst import compute_polytope_id, compute_triangulation_hash, stable_hash
from inherited_orientifold_candidates import (
    enumerate_polytope_involutions,
    enumerate_projected_lattice_representatives,
)


SCHEMA_VERSION = "cyaxiverse-orientifold-population-enrichment-1.0"
CHECKPOINT_SCHEMA_VERSION = "cyaxiverse-orientifold-population-enrichment-checkpoint-1.0"
INPUT_CERTIFICATE_SCHEMA_VERSION = "cyaxiverse-population-input-certificate-1.0"
POPULATION_CERTIFICATE_SCHEMA_VERSION = "cyaxiverse-population-mpcp-certificate-1.0"
CYTOOLS_VERSION = "1.4.12"
REQUIRED_PARTITIONS = tuple(f"polytopes-4d-{value:02d}-vertices.parquet" for value in range(5, 11))
HISTORICAL_SOURCE_DIGEST = "11c4f2bb6f4412d3d0c453690ab2d212249e3e0e4a08e9d785fb42b8f9f8b765"


class EnrichmentError(RuntimeError):
    """Raise when the frozen enrichment contract cannot be satisfied."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zstd_json(path: Path) -> Any:
    completed = subprocess.run(["zstd", "-dcq", str(path)], capture_output=True, check=False)
    if completed.returncode:
        raise EnrichmentError(f"cannot decode zstd JSON {path}: {completed.stderr.decode(errors='replace')}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EnrichmentError(f"invalid JSON in {path}") from exc


def zstd_jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    completed = subprocess.run(["zstd", "-dcq", str(path)], capture_output=True, check=False)
    if completed.returncode:
        raise EnrichmentError(f"cannot decode checkpoint {path}")
    rows = []
    for number, line in enumerate(completed.stdout.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EnrichmentError(f"invalid checkpoint JSON at line {number}") from exc
        if not isinstance(row, dict):
            raise EnrichmentError(f"checkpoint line {number} is not an object")
        rows.append(row)
    return rows


def locate_raw_candidate(raw_ledger_dir: Path, candidate_id: str) -> dict[str, Any]:
    """Locate exactly one accepted candidate in immutable raw ledgers."""
    indexed = index_raw_candidates(raw_ledger_dir, {candidate_id})
    try:
        return indexed[candidate_id]
    except KeyError as exc:
        raise EnrichmentError(f"raw terminal ledger has no candidate_id {candidate_id}") from exc


_CANDIDATE_ID_RE = re.compile(rb'"candidate_id"\s*:\s*"([0-9a-f]{64})"')


def index_raw_candidates(
    raw_ledger_dir: Path, candidate_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Index requested raw witnesses in one streaming pass over each shard.

    Raw h11=5 shards expand to several GiB each.  Stream zstd output instead
    of capturing a complete decompressed shard, and parse only lines whose
    witness identifier is requested.  Every requested identifier must occur
    exactly once; ambiguity remains a hard failure.
    """
    if not candidate_ids:
        return {}
    paths = sorted(raw_ledger_dir.glob("*.terminal-ledger.jsonl.zst"))
    if not paths:
        raise EnrichmentError(f"raw terminal-ledger directory has no shard ledgers: {raw_ledger_dir}")
    encoded_ids = {value.encode("ascii") for value in candidate_ids}
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            process = subprocess.Popen(
                ["zstd", "-dcq", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise EnrichmentError(f"cannot decode raw terminal ledger: {path}") from exc
        assert process.stdout is not None
        try:
            for line in process.stdout:
                match = _CANDIDATE_ID_RE.search(line)
                if match is None or match.group(1) not in encoded_ids:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EnrichmentError(f"raw terminal ledger has invalid JSON: {path}") from exc
                witness = row.get("accepted_witness")
                candidate_id = witness.get("candidate_id") if isinstance(witness, Mapping) else None
                if candidate_id not in candidate_ids:
                    continue
                if candidate_id in indexed:
                    raise EnrichmentError(f"raw terminal ledger has ambiguous candidate_id {candidate_id}")
                indexed[candidate_id] = {"path": str(path.resolve()), "record": row}
        finally:
            process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        returncode = process.wait()
        if returncode:
            raise EnrichmentError(
                f"cannot decode raw terminal ledger: {path}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
    missing = sorted(candidate_ids - set(indexed))
    if missing:
        raise EnrichmentError(f"raw terminal ledger has no candidate_id {missing[0]}")
    return indexed


def validate_binary_shift(entry: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, list[int]]:
    """Bind the raw binary representative and projected candidate numerator."""
    witness = entry.get("accepted_witness")
    raw_record = raw.get("record") if isinstance(raw, Mapping) else None
    raw_witness = raw_record.get("accepted_witness") if isinstance(raw_record, Mapping) else None
    if not isinstance(witness, Mapping) or not isinstance(raw_witness, Mapping):
        raise EnrichmentError("raw and merged accepted witnesses are required")
    binary = raw_record.get("torus_shift_binary_source")
    if binary is None:
        binary = raw_witness.get("torus_shift_binary_source")
    if not isinstance(binary, list) or len(binary) != 4 or not all(isinstance(v, int) and not isinstance(v, bool) for v in binary):
        raise EnrichmentError("raw terminal ledger has no exact torus_shift_binary_source")
    matrix = np.asarray(raw_record.get("lattice_matrix"), dtype=np.int64)
    if matrix.shape != (4, 4):
        raise EnrichmentError("raw terminal ledger lattice_matrix is malformed")
    shift = witness.get("torus_shift")
    if not isinstance(shift, Mapping) or not isinstance(shift.get("numerator"), list):
        raise EnrichmentError("merged accepted witness torus shift is incomplete")
    try:
        observed_rational = tuple(
            Fraction(int(value), int(shift["denominator"]))
            for value in shift["numerator"]
        )
    except (TypeError, ValueError, ZeroDivisionError):
        raise EnrichmentError("merged accepted witness torus shift is not exact")
    representatives = enumerate_projected_lattice_representatives(matrix, +1)
    matching = [
        representative
        for representative in representatives
        if representative.get("binary_source") == binary
        # The source writer stores representative["vector"] = numerator / 2
        # and then constructs torus_shift by dividing that vector by 2 once
        # more. Preserve that historical source convention exactly here.
        and tuple(value / 2 for value in representative.get("vector", ())) == observed_rational
    ]
    if len(matching) != 1:
        raise EnrichmentError(
            "raw binary shift has no unique projected representative matching the merged rational torus shift"
        )
    projected = [int(value) for value in matching[0]["numerator"]]
    if stable_hash([witness["matrix_id"], tuple(projected), int(witness["lambda_f"])]) != witness.get("candidate_id"):
        raise EnrichmentError("candidate_id does not bind matrix, raw binary shift, and lambda_f")
    return {"binary": list(binary), "projected": [int(value) for value in projected]}


def zstd_jsonl_write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL with zstd -19 and replace the destination atomically."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_name(f".{path.name}.raw-{os.getpid()}-{time.time_ns()}")
    compressed = path.with_name(f".{path.name}.zst-{os.getpid()}-{time.time_ns()}")
    try:
        with raw.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
        completed = subprocess.run(
            ["zstd", "-19", "-q", "-f", "-o", str(compressed), str(raw)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode:
            raise EnrichmentError(f"zstd compression failed: {completed.stderr.strip()}")
        os.replace(compressed, path)
    finally:
        raw.unlink(missing_ok=True)
        compressed.unlink(missing_ok=True)


def source_contract(source_dir: Path) -> dict[str, Any]:
    """Verify the durable byte-level source contract before reading rows."""
    manifest_path = source_dir / "source-manifest.json.zst"
    manifest = zstd_json(manifest_path)
    if manifest.get("historical_handoff_directory_digest") != HISTORICAL_SOURCE_DIGEST:
        raise EnrichmentError("durable source manifest does not bind the approved handoff directory digest")
    entries = []
    expected = {item["filename"]: item for item in manifest.get("partitions", [])}
    if set(expected) != set(REQUIRED_PARTITIONS):
        raise EnrichmentError("durable source manifest must contain exactly partitions 05..10")
    for name in REQUIRED_PARTITIONS:
        path = source_dir / name
        if not path.is_file():
            raise EnrichmentError(f"durable source partition is missing: {path}")
        observed = file_digest(path)
        if observed != expected[name]["sha256"]:
            raise EnrichmentError(f"source partition hash mismatch: {name}")
        entries.append({
            "partition": name,
            "path": str(path.resolve()),
            "sha256": observed,
            "size_bytes": path.stat().st_size,
        })
    return {
        "dataset": manifest.get("dataset"),
        "dataset_revision": manifest.get("dataset_revision"),
        "directory": str(source_dir.resolve()),
        "directory_sha256": HISTORICAL_SOURCE_DIGEST,
        "partitions": entries,
        "physical_h11_contract": manifest.get("physical_h11_contract"),
    }


def candidate_identity(entry: Mapping[str, Any]) -> str:
    witness = entry.get("accepted_witness")
    if not isinstance(witness, Mapping):
        raise EnrichmentError("authoritative candidate has no accepted_witness")
    return digest({
        "polytope_id": entry.get("polytope_id"),
        "frst_hash": entry.get("frst_hash"),
        "frst_class_index": entry.get("frst_class_index"),
        "candidate_id": witness.get("candidate_id"),
        "matrix_id": witness.get("matrix_id"),
        "torus_shift": witness.get("torus_shift"),
        "lambda_f": witness.get("lambda_f"),
    })


def replay_candidate_identity(entry: Mapping[str, Any], source_digest: str) -> str:
    """Match the exact replay driver's canonical identity key."""
    witness = entry.get("accepted_witness")
    if not isinstance(witness, Mapping):
        raise EnrichmentError("authoritative candidate has no accepted_witness")
    witness_digest = digest(witness)
    action_digest = entry.get("action_digest") or witness.get("action_digest") or witness_digest
    return digest({
        "declared_source_digest": source_digest,
        "canonical_polytope_id": entry.get("polytope_id"),
        "global_coordinates": entry.get("global_points") or witness.get("global_points"),
        "frst_hash": entry.get("frst_hash"),
        "action_digest": action_digest,
        "witness_digest": witness_digest,
    })


def authoritative_candidates(merged: Mapping[str, Any], h11: int) -> list[tuple[str, dict[str, Any]]]:
    if merged.get("requested_h11") != int(h11):
        raise EnrichmentError("merged ledger physical h11 does not match request")
    funnel = merged.get("terminal_ledger", {}).get("class_funnel")
    if not isinstance(funnel, list):
        raise EnrichmentError("merged ledger has no class funnel")
    rows = []
    identities = set()
    for entry in funnel:
        if not isinstance(entry, dict) or entry.get("accepted_for_table_1") is not True:
            continue
        if not isinstance(entry.get("accepted_witness"), dict):
            continue
        identity = replay_candidate_identity(entry, HISTORICAL_SOURCE_DIGEST)
        if identity in identities:
            raise EnrichmentError(f"duplicate authoritative candidate identity: {identity}")
        identities.add(identity)
        rows.append((identity, dict(entry)))
    rows.sort(key=lambda item: item[0])
    return rows


def _iter_source_rows(source: Mapping[str, Any], h11: int):
    try:
        import pyarrow.parquet as parquet
        from cytools import Polytope
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise EnrichmentError("pyarrow and CYTools are required for exact source joins") from exc
    source_dir = Path(source["directory"])
    favorable_index = 0
    for partition in source["partitions"]:
        path = Path(partition["path"])
        row_index = 0
        for batch in parquet.ParquetFile(path).iter_batches(
            columns=["vertices", "vertex_count", "facet_count", "point_count", "h11", "h12"],
            batch_size=512,
        ):
            for row in batch.to_pylist():
                current_row = row_index
                row_index += 1
                if int(row["h12"]) != int(h11):
                    continue
                poly = Polytope(row["vertices"], deterministic_glsm_basis=True)
                actual_h11 = int(poly.h11())
                if actual_h11 != int(h11):
                    yield {
                        "terminal_status": "physical_h11_mismatch",
                        "reason": f"CYTools Polytope.h11()={actual_h11}, requested={h11}",
                        "partition": partition["partition"],
                        "source_row": current_row,
                    }
                    continue
                if not bool(poly.is_favorable(lattice="N")):
                    continue
                yield {
                    "partition": partition["partition"],
                    "partition_sha256": partition["sha256"],
                    "source_row": current_row,
                    "row_metadata": row,
                    "poly": poly,
                    "physical_h11": actual_h11,
                    "global_points": np.asarray(poly.points(), dtype=np.int64).tolist(),
                    "polytope_id": compute_polytope_id(poly.points()),
                    "favorable": True,
                    "favorable_index": favorable_index,
                }
                favorable_index += 1


def locate_source(
    source: Mapping[str, Any],
    h11: int,
    target_id: str,
    expected_favorable_index: int | None = None,
) -> dict[str, Any]:
    if expected_favorable_index is not None:
        for row in _iter_source_rows(source, h11):
            if row.get("favorable_index") != int(expected_favorable_index):
                continue
            if row.get("polytope_id") != target_id:
                raise EnrichmentError(
                    "source favorable-index route does not match canonical polytope identity"
                )
            return row
        raise EnrichmentError(f"source favorable index was not found: {expected_favorable_index}")
    matches = []
    for row in _iter_source_rows(source, h11):
        if row.get("terminal_status"):
            continue
        if row.get("polytope_id") == target_id:
            matches.append(row)
    if not matches:
        raise EnrichmentError(f"source polytope identity was not found: {target_id}")
    if len(matches) != 1:
        locations = [(row.get("partition"), row.get("source_row")) for row in matches]
        raise EnrichmentError(f"source polytope identity is ambiguous: {target_id}: {locations}")
    return matches[0]


def index_source_rows(
    source: Mapping[str, Any],
    h11: int,
    expected_routes: Mapping[str, int | None],
) -> dict[str, dict[str, Any]]:
    """Build a one-pass source index while retaining exact route checks."""
    if not expected_routes:
        return {}
    target_ids = set(expected_routes)
    expected_indices = {
        int(index) for index in expected_routes.values() if index is not None
    }
    indexed: dict[str, dict[str, Any]] = {}
    indexed_indices: dict[int, str] = {}
    for row in _iter_source_rows(source, h11):
        if row.get("terminal_status"):
            continue
        polytope_id = str(row["polytope_id"])
        favorable_index = int(row["favorable_index"])
        if polytope_id not in target_ids and favorable_index not in expected_indices:
            continue
        if polytope_id in indexed:
            raise EnrichmentError(f"source polytope identity is ambiguous: {polytope_id}")
        previous = indexed_indices.get(favorable_index)
        if previous is not None and previous != polytope_id:
            raise EnrichmentError(
                f"source favorable index is ambiguous: {favorable_index}"
            )
        indexed[polytope_id] = row
        indexed_indices[favorable_index] = polytope_id
    for polytope_id in target_ids:
        row = indexed.get(polytope_id)
        if row is None:
            raise EnrichmentError(f"source polytope identity was not found: {polytope_id}")
        expected_index = expected_routes[polytope_id]
        if expected_index is not None and int(row["favorable_index"]) != int(expected_index):
            raise EnrichmentError("source favorable-index route does not match canonical polytope identity")
    return indexed


def match_frst(poly: Any, expected_hash: str, raw_candidate: Mapping[str, Any]) -> tuple[Any, int]:
    """Reconstruct the selected FRST from raw retained ambient maximal cones."""
    raw_record = raw_candidate.get("record") if isinstance(raw_candidate, Mapping) else None
    auxiliary = raw_record.get("auxiliary_fan") if isinstance(raw_record, Mapping) else None
    if not isinstance(auxiliary, list):
        raise EnrichmentError("raw terminal ledger has no retained auxiliary fan")
    maximal = {
        tuple(sorted(tuple(int(value) for value in ray) for ray in cone))
        for component in auxiliary
        if isinstance(component, Mapping)
        for cone in component.get("ambient_cones", ())
        if isinstance(cone, list) and len(cone) == 4
    }
    if not maximal:
        raise EnrichmentError("raw terminal ledger has no retained ambient maximal cones")
    global_points = {
        tuple(int(value) for value in point): index
        for index, point in enumerate(np.asarray(poly.points(), dtype=np.int64).tolist())
    }
    origin = (0, 0, 0, 0)
    if origin not in global_points:
        raise EnrichmentError("source polytope has no lattice origin for star-FRST reconstruction")
    origin_index = global_points[origin]
    simplices = []
    for cone in sorted(maximal):
        try:
            # The retained ambient cones are 4-ray fan cones.  CYTools expects
            # the corresponding 4-dimensional star simplices, so retain the
            # exact lattice origin as their fifth vertex.
            simplices.append([origin_index] + [global_points[ray] for ray in cone])
        except KeyError as exc:
            raise EnrichmentError("retained ambient cone contains a point absent from the source polytope") from exc
    try:
        triangulation = poly.triangulate(
            simplices=np.asarray(simplices, dtype=np.int64),
            check_input_simplices=True,
            include_points_interior_to_facets=False,
        )
    except Exception as exc:
        raise EnrichmentError(f"selected FRST reconstruction from retained ambient cones failed: {exc}") from exc
    observed = compute_triangulation_hash(np.asarray(triangulation.simplices(), dtype=np.int64))
    if observed != expected_hash:
        raise EnrichmentError(
            f"selected FRST hash mismatch after retained-cone reconstruction: expected {expected_hash}, got {observed}"
        )
    return triangulation, 0


def match_matrix(poly: Any, polytope_id: str, frst_hash: str, matrix_id: str) -> tuple[np.ndarray, str]:
    for matrix in enumerate_polytope_involutions(np.asarray(poly.points(), dtype=np.int64)):
        observed = stable_hash([polytope_id, frst_hash, tuple(int(value) for value in matrix.flatten())])
        if observed == matrix_id:
            return np.asarray(matrix, dtype=np.int64), observed
    raise EnrichmentError(f"matrix_id was not reproducible from exact polytope and FRST: {matrix_id}")


def _cytools_api_contract() -> dict[str, Any]:
    try:
        import cytools
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": "unavailable", "expected": CYTOOLS_VERSION, "observed": None, "reason": f"{type(exc).__name__}: {exc}"}
    observed = str(getattr(cytools, "__version__", getattr(cytools, "version", "unknown")))
    return {
        "status": "verified" if observed == CYTOOLS_VERSION else "unsupported",
        "expected": CYTOOLS_VERSION,
        "observed": observed,
        "contract": f"cytools-public-api=={CYTOOLS_VERSION}",
        "reason": None if observed == CYTOOLS_VERSION else "CYTools version is outside the audited input contract",
    }


def build_input_certificate(
    source: Mapping[str, Any],
    source_record: Mapping[str, Any],
    entry: Mapping[str, Any],
    binary_shift: dict[str, list[int]],
    raw_ledger_path: str,
) -> dict[str, Any]:
    """Bind source, selected FRST, and exact action without output evidence."""
    source_part = source_record["source"]
    selected = source_record["selected_frst"]
    action = source_record["action_witness"]
    action_witness = {
        **action,
        "torus_shift_binary_source": binary_shift["binary"],
        "torus_shift_projected_numerator": binary_shift["projected"],
    }
    certificate: dict[str, Any] = {
        "certificate_schema_version": INPUT_CERTIFICATE_SCHEMA_VERSION,
        "source": {
            "source_sha256": source_part["source_sha256"],
            "partition": source_part["partition"],
            "partition_sha256": source_part["partition_sha256"],
            "source_row": source_part["source_row"],
            "population_polytope_index": source_part.get("population_polytope_index"),
            "polytope_id": source_part["polytope_id"],
            "global_points": source_part["global_points"],
        },
        "frst": {
            "frst_hash": selected["frst_hash"],
            "points": selected["points"],
            "simplices": selected["simplices"],
            "simplices_index_space": selected["simplices_index_space"],
            "point_digest": selected["point_digest"],
            "simplex_digest": selected["simplex_digest"],
        },
        "action": {
            "witness": action_witness,
            "digest": action["action_digest"],
            "candidate_witness_digest": digest(entry["accepted_witness"]),
            "raw_terminal_ledger": raw_ledger_path,
        },
        "cytools_api_contract": _cytools_api_contract(),
        "geometry": {
            "physical_h11": source_part["physical_h11"],
            "favorable_lattice": source_part["favorable_lattice"],
            "reconstruction": "CYTools Polytope from global points; selected FRST from exact local points and simplices",
        },
        "provenance": {
            "source_directory": source_part["source_directory"],
            "source_directory_sha256": source_part["source_directory_sha256"],
            "selection_route": source_part["selection_route"],
        },
    }
    key = {
        "source_sha256": certificate["source"]["source_sha256"],
        "partition": certificate["source"]["partition"],
        "partition_sha256": certificate["source"]["partition_sha256"],
        "source_row": certificate["source"]["source_row"],
        "population_polytope_index": certificate["source"].get("population_polytope_index"),
        "polytope_id": certificate["source"]["polytope_id"],
        "global_points": certificate["source"]["global_points"],
        "frst_hash": certificate["frst"]["frst_hash"],
        "frst_points": certificate["frst"]["points"],
        "frst_simplices": certificate["frst"]["simplices"],
        "action_witness": action_witness,
        "action_digest": action["action_digest"],
        "cytools_api_contract": certificate["cytools_api_contract"],
        "physical_h11": certificate["geometry"]["physical_h11"],
    }
    certificate["certificate_key"] = key
    certificate["certificate_key_digest"] = digest(key)
    certificate["certificate_digest"] = digest({
        name: value
        for name, value in certificate.items()
        if name not in {"certificate_digest", "certificate_key_digest"}
    })
    return certificate


def build_source_record(
    source: Mapping[str, Any],
    row: Mapping[str, Any],
    triangulation: Any,
    matrix: np.ndarray,
    entry: Mapping[str, Any],
    binary_shift: dict[str, list[int]],
    raw_ledger_path: str,
) -> dict[str, Any]:
    witness = entry["accepted_witness"]
    selected = {
        "frst_hash": entry["frst_hash"],
        "points": np.asarray(triangulation.points(), dtype=np.int64).tolist(),
        "simplices": np.asarray(triangulation.simplices(as_indices=True), dtype=np.int64).tolist(),
        "simplices_index_space": "triangulation_local",
        "point_digest": digest(sorted(tuple(int(v) for v in point) for point in np.asarray(triangulation.points(), dtype=np.int64).tolist())),
        "simplex_digest": digest(sorted(tuple(sorted(int(v) for v in simplex)) for simplex in np.asarray(triangulation.simplices(as_indices=True), dtype=np.int64).tolist())),
        "cytools_triangulation_index": row["triangulation_index"],
    }
    action = {
        "lattice_matrix": matrix.tolist(),
        "torus_shift": witness["torus_shift"],
        "lambda_f": int(witness["lambda_f"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": source["dataset"],
            "dataset_revision": source["dataset_revision"],
            "source_directory": source["directory"],
            "source_directory_sha256": source["directory_sha256"],
            "source_sha256": row["partition_sha256"],
            "partition": row["partition"],
            "partition_sha256": row["partition_sha256"],
            "source_row": int(row["source_row"]),
            "population_polytope_index": entry.get("polytope_index"),
            "polytope_id": row["polytope_id"],
            "global_points": row["global_points"],
            "physical_h11": int(row["physical_h11"]),
            "favorable_lattice": "N",
            "selection_route": "mirror_h12_then_CYTools_Polytope.h11_then_is_favorable_N_then_canonical_polytope_id",
            "row_metadata": row["row_metadata"],
        },
        "selected_frst": selected,
        "action_witness": {
            **action,
            "matrix_id": witness["matrix_id"],
            "candidate_id": witness["candidate_id"],
            "torus_shift_binary_source": binary_shift["binary"],
            "torus_shift_projected_numerator": binary_shift["projected"],
            "action_digest": digest(action),
        },
        "raw_terminal_ledger": raw_ledger_path,
        "ledger_witness": witness,
    }


def enrich_candidate(
    source: Mapping[str, Any],
    h11: int,
    entry: dict[str, Any],
    raw_candidate: Mapping[str, Any] | None = None,
    source_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = replay_candidate_identity(entry, HISTORICAL_SOURCE_DIGEST)
    row: dict[str, Any] = {
        "record_type": "row",
        "row_identity": identity,
        "h11": int(h11),
        "frst_class_index": entry.get("frst_class_index"),
        "candidate": entry,
        "terminal_status": "mpcp_certificate_unavailable",
        "failure_categories": [],
        "source_record": None,
        "mpcp_certificate": None,
    }
    try:
        if source_row is None:
            source_row = locate_source(
                source,
                h11,
                str(entry["polytope_id"]),
                expected_favorable_index=entry.get("polytope_index"),
            )
        else:
            if source_row.get("polytope_id") != str(entry["polytope_id"]):
                raise EnrichmentError(
                    "source favorable-index route does not match canonical polytope identity"
                )
            expected_index = entry.get("polytope_index")
            if expected_index is not None and source_row.get("favorable_index") != int(expected_index):
                raise EnrichmentError(
                    "source favorable-index route does not match canonical polytope identity"
                )
        if raw_candidate is None:
            raise EnrichmentError("raw terminal-ledger candidate join is required")
        binary_shift = validate_binary_shift(entry, raw_candidate)
        tri, tri_index = match_frst(
            source_row["poly"], str(entry["frst_hash"]), raw_candidate
        )
        source_row = dict(source_row)
        source_row["triangulation_index"] = tri_index
        witness = entry["accepted_witness"]
        matrix, _ = match_matrix(source_row["poly"], str(entry["polytope_id"]), str(entry["frst_hash"]), str(witness["matrix_id"]))
        shift = witness.get("torus_shift")
        if not isinstance(shift, Mapping) or not isinstance(shift.get("numerator"), list):
            raise EnrichmentError("accepted witness torus shift is incomplete")
        expected_candidate_id = stable_hash(
            [witness["matrix_id"], tuple(binary_shift["projected"]), int(witness["lambda_f"])]
        )
        if expected_candidate_id != witness.get("candidate_id"):
            raise EnrichmentError("accepted witness candidate_id does not bind matrix, shift, and lambda_f")
        source_record = build_source_record(
            source, source_row, tri, matrix, entry, binary_shift, str(raw_candidate["path"])
        )
        row["source_record"] = source_record
        row["mpcp_certificate"] = build_input_certificate(
            source, source_record, entry, binary_shift, str(raw_candidate["path"])
        )
        row["terminal_status"] = "input_identity_verified"
        row["failure_categories"] = []
    except EnrichmentError as exc:
        text = str(exc)
        status = "source_join_failed"
        for token, candidate_status in (
            ("FRST hash", "frst_identity_mismatch"),
            ("matrix_id", "matrix_identity_mismatch"),
            ("physical h11", "physical_h11_mismatch"),
            ("source polytope identity", "source_identity_unavailable"),
        ):
            if token in text:
                status = candidate_status
                break
        row["terminal_status"] = status
        row["failure_categories"] = [{"category": status, "reason": text}]
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.max_rows) < 1:
        raise EnrichmentError("--max-rows must be positive for an enrichment run")
    if int(args.max_rows) > 2 and args.pilot_only:
        raise EnrichmentError("--pilot-only permits at most two candidates")
    checkpoint_interval = int(getattr(args, "checkpoint_interval", 32))
    if checkpoint_interval < 1:
        raise EnrichmentError("--checkpoint-interval must be positive")
    source_dir = Path(args.source_dir).expanduser().resolve()
    source = source_contract(source_dir)
    raw_ledger_dir = Path(args.raw_ledger_dir).expanduser().resolve()
    merged_path = Path(args.merged).expanduser().resolve()
    expected_merged = args.merged_sha256 or file_digest(merged_path)
    if file_digest(merged_path) != expected_merged:
        raise EnrichmentError("merged ledger changed before enrichment")
    merged = zstd_json(merged_path)
    candidates = authoritative_candidates(merged, int(args.h11))[: int(args.max_rows)]
    config = {
        "schema_version": SCHEMA_VERSION,
        "h11": int(args.h11),
        "source": source,
        "merged": str(merged_path),
        "merged_sha256": expected_merged,
        "raw_ledger_dir": str(raw_ledger_dir),
        "max_rows": int(args.max_rows),
        "checkpoint_interval": checkpoint_interval,
        "workers": 1,
        "cytools_version": CYTOOLS_VERSION,
        "selection": "accepted_for_table_1 with accepted_witness, sorted by immutable candidate identity",
        "database_writes": 0,
    }
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    existing = zstd_jsonl_read(checkpoint)
    if existing:
        header = existing[0]
        if header.get("record_type") != "header" or header.get("config_sha256") != digest(config):
            raise EnrichmentError("checkpoint frozen configuration does not match")
    else:
        existing = [{"record_type": "header", "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION, "config": config, "config_sha256": digest(config)}]
    identities = {str(row["row_identity"]) for row in existing[1:] if row.get("record_type") == "row"}
    rows = list(existing)
    pending = [
        (identity, entry)
        for identity, entry in candidates
        if identity not in identities
    ]
    raw_index = index_raw_candidates(
        raw_ledger_dir,
        {
            str(entry["accepted_witness"]["candidate_id"])
            for _, entry in pending
        },
    )
    source_index = index_source_rows(
        source,
        int(args.h11),
        {
            str(entry["polytope_id"]): (
                int(entry["polytope_index"]) if entry.get("polytope_index") is not None else None
            )
            for _, entry in pending
        },
    )
    since_checkpoint = 0
    for identity, entry in pending:
        candidate_id = str(entry["accepted_witness"]["candidate_id"])
        raw_candidate = raw_index[candidate_id]
        row = enrich_candidate(
            source,
            int(args.h11),
            entry,
            raw_candidate,
            source_index[str(entry["polytope_id"])],
        )
        rows.append(row)
        identities.add(identity)
        since_checkpoint += 1
        if since_checkpoint >= checkpoint_interval:
            zstd_jsonl_write_atomic(checkpoint, rows)
            since_checkpoint = 0
    if since_checkpoint or not checkpoint.exists():
        zstd_jsonl_write_atomic(checkpoint, rows)
    summary = {
        "record_type": "summary",
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "config": config,
        "rows_evaluated": len(rows) - 1,
        "terminal_status_counts": {
            status: sum(1 for row in rows[1:] if row.get("terminal_status") == status)
            for status in sorted({str(row.get("terminal_status")) for row in rows[1:]})
        },
        "duplicate_count": len(rows[1:]) - len(identities),
        "database_writes": 0,
        "scientific_result": "no_scientific_result",
        "runtime": {"python": platform.python_version(), "cytools": CYTOOLS_VERSION},
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise EnrichmentError(f"refusing to overwrite output: {output}")
        zstd_jsonl_write_atomic(output, rows + [summary])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11", type=int, choices=(4, 5), required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--merged-sha256")
    parser.add_argument("--raw-ledger-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=2)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=32,
        help="Atomically rewrite the zstd checkpoint after this many new rows.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    try:
        print(json.dumps(run(build_parser().parse_args()), sort_keys=True))
    except EnrichmentError as exc:
        raise SystemExit(f"enrichment blocked: {exc}")
