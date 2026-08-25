"""Run a bounded, provenance-first MPCP/refinement replay.

The driver is deliberately independent of the production HDF5 writers.  It
accepts explicit replay objects (or a JSON manifest containing points and the
selected FRST) and evaluates only indices 26, 31, and 33.  CYTools public APIs
are feature-detected and used first.  Exact project kernels are used only at
the boundaries where CYTools has no orientifold implementation.

This module does not choose a triangulation by a Hodge result or a population
target.  Every yielded candidate and every terminal reason is retained in the
returned report.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
import math
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    # Keep the bounded source hash in one immutable fixture module.  The
    # import is optional for standalone analytic callers, but real bounded
    # source records are rejected unless they carry this audited hash.
    from mpcp_immutable_source import SOURCE_DATASET, SOURCE_PARQUET_SHA256
except ImportError:  # pragma: no cover - direct standalone import fallback
    SOURCE_DATASET = None
    SOURCE_PARQUET_SHA256 = None


TARGET_INDICES = (26, 31, 33)
# These are the immutable source-row anchors for the bounded replay.  Keeping
# the row key here prevents a manifest from silently relabelling a valid
# coordinate configuration as a different catalog row.
IMMUTABLE_SOURCE_ROWS = {26: 21, 31: 27, 33: 29}
SCHEMA_VERSION = "cyaxiverse-bounded-mpcp-replay-1.3"
CERTIFICATE_SCHEMA_VERSION = "cyaxiverse-bounded-mpcp-certificate-1.1"
# Population certificates are deliberately separate from the h11=2 fixture
# contract above.  The fixture validator is immutable and must not be widened
# to accept a different source population by changing its index allow-list.
POPULATION_CERTIFICATE_SCHEMA_VERSION = "cyaxiverse-population-mpcp-certificate-1.0"
# h11=4/5 input identity is intentionally separate from the h11=2 bounded
# output certificate.  It binds only source geometry and the exact action;
# fixed-component Euler and refined-GLSM evidence is produced by the live
# replay and must not be required here.
POPULATION_INPUT_CERTIFICATE_SCHEMA_VERSION = "cyaxiverse-population-input-certificate-1.0"
# Keep this separate from the replay schema: changing a formula, evidence
# interpretation, or persisted certificate field invalidates old certificates.
FORMULA_SCHEMA_VERSION = "cyaxiverse-mpcp-formula-ledger-20260823-2"
SUPPORTED_CYTOOLS_API_VERSION = "1.4.12"
DEFAULT_CAPS = {
    "max_triangulations": 64,
    "max_refinement_cells": 20_000,
    "max_seconds_per_index": 120.0,
}


def _package_version(name: str) -> str | None:
    """Return a package version without making optional dependencies required."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cytools_version_guard(module: Any = None) -> dict[str, Any]:
    """Require the audited CYTools API before constructing new geometry."""

    if module is None:
        try:
            module = importlib.import_module("cytools")
        except Exception as exc:  # pragma: no cover - optional environment
            return {
                "status": "unavailable",
                "expected": SUPPORTED_CYTOOLS_API_VERSION,
                "observed": None,
                "reason": f"CYTools import failed: {type(exc).__name__}: {exc}",
            }
    observed = getattr(module, "version", None) or getattr(module, "__version__", None)
    if observed is None:
        observed = _package_version("cytools")
    observed = None if observed is None else str(observed)
    if observed != SUPPORTED_CYTOOLS_API_VERSION:
        return {
            "status": "unsupported",
            "expected": SUPPORTED_CYTOOLS_API_VERSION,
            "observed": observed,
            "reason": (
                "the replay contract is audited only for CYTools "
                f"{SUPPORTED_CYTOOLS_API_VERSION}; observed {observed!r}"
            ),
        }
    return {
        "status": "verified",
        "expected": SUPPORTED_CYTOOLS_API_VERSION,
        "observed": observed,
        "reason": None,
    }


def runtime_provenance() -> dict[str, Any]:
    """Record runtime and optional CYTools provenance."""

    provenance: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "sympy", "cytools", "regfans", "triangulumancer")
        },
        "cytools_api_version": None,
        "cytools_import": "not_attempted",
    }
    try:
        module = importlib.import_module("cytools")
    except Exception as exc:  # pragma: no cover - depends on optional environment
        provenance["cytools_import"] = "unavailable"
        provenance["cytools_import_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        provenance["cytools_import"] = "available"
        provenance["cytools_api_version"] = (
            getattr(module, "version", None)
            or getattr(module, "__version__", None)
            or _package_version("cytools")
        )
        provenance["cytools_module"] = getattr(module, "__file__", None)
        provenance["cytools_version_guard"] = _cytools_version_guard(module)
    return provenance


def _callable(obj: Any, name: str) -> bool:
    """Return whether an object exposes a callable public method."""

    return callable(getattr(obj, name, None))


def feature_detection(obj: Any) -> dict[str, dict[str, Any]]:
    """Describe the public CYTools operations available on ``obj``."""

    names = (
        "points", "vertices", "facets", "faces", "dual", "all_triangulations",
        "triangulate", "automorphisms", "fan", "get_toric_variety", "get_cy",
        "heights", "secondary_cone", "glsm_charge_matrix", "glsm_linear_relations",
        "sr_ideal", "intersection_numbers", "divisor_basis", "is_smooth",
        "h11", "h21", "chi", "prime_toric_divisors", "toric_mori_cone",
        "toric_kahler_cone", "is_fine", "is_regular", "is_star", "is_valid",
        "simplices", "vc",
    )
    detected: dict[str, dict[str, Any]] = {}
    for name in names:
        method = getattr(obj, name, None)
        entry: dict[str, Any] = {"present": method is not None, "callable": callable(method)}
        if callable(method):
            try:
                entry["signature"] = str(inspect.signature(method))
            except (TypeError, ValueError):
                entry["signature"] = None
        detected[name] = entry
    return detected


def _jsonable(value: Any) -> Any:
    """Convert exact/numpy values into a JSON-safe diagnostic value."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_digest(value: Any) -> str:
    """Hash a JSON-safe value with one deterministic canonical encoding."""

    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _action_digest(action: Mapping[str, Any]) -> str:
    """Hash the exact source-gauge action witness ``(L,t,lambda_f)``."""

    try:
        payload = {
            "lattice_matrix": _jsonable(action["lattice_matrix"]),
            "torus_shift": _jsonable(action["torus_shift"]),
            "lambda_f": int(action["lambda_f"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"action witness is incomplete: {exc}") from exc
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def component_h2_evidence_digest(action_record: Mapping[str, Any]) -> str | None:
    """Hash the complete fixed-component, Euler, and refined-H2 evidence.

    The full records are retained in the certificate.  This digest prevents a
    compact result field (for example only ``chi_F_I``) from being replayed as
    if it were the underlying component or GLSM proof.
    """

    fixed = action_record.get("fixed_locus_euler")
    h2 = action_record.get("refined_glsm")
    if not isinstance(fixed, Mapping) or not isinstance(h2, Mapping):
        return None
    return _canonical_digest({
        "fixed_locus_euler": fixed,
        "refined_glsm": h2,
    })


def _certificate_key(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable identity portion of one replay certificate."""

    source = certificate.get("source")
    frst = certificate.get("frst")
    action = certificate.get("action")
    api = certificate.get("cytools_api_contract")
    evidence = certificate.get("evidence")
    if not all(isinstance(item, Mapping) for item in (source, frst, action, api, evidence)):
        raise ValueError("certificate identity sections must be mappings")
    return {
        "source_sha256": source.get("source_sha256"),
        "source_row": source.get("source_row"),
        "polytope_id": source.get("polytope_id"),
        "global_points": source.get("global_points"),
        "frst_hash": frst.get("frst_hash"),
        "action_witness": action.get("witness"),
        "action_digest": action.get("digest"),
        "cytools_api_contract": api,
        "formula_schema_version": certificate.get("formula_schema_version"),
        "certificate_schema_version": certificate.get("certificate_schema_version"),
        "component_h2_evidence_digest": evidence.get("component_h2_evidence_digest"),
    }


def _fallback_reasons(value: Any, *, path: str = "") -> list[dict[str, str]]:
    """Collect every explicitly recorded fallback reason for replay provenance."""

    found: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in {"fallback_reason", "reason"} and item not in (None, ""):
                found.append({"path": child_path, "reason": str(item)})
            found.extend(_fallback_reasons(item, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_fallback_reasons(item, path=f"{path}[{index}]"))
    return found


def _certificate_api_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and require the audited CYTools API contract from a report."""

    provenance = report.get("runtime_provenance")
    guard = provenance.get("cytools_version_guard") if isinstance(provenance, Mapping) else None
    if not isinstance(guard, Mapping):
        return {
            "status": "unavailable",
            "expected": SUPPORTED_CYTOOLS_API_VERSION,
            "observed": None,
            "reason": "bounded report has no CYTools version guard",
        }
    return {
        "status": str(guard.get("status")),
        "expected": str(guard.get("expected")),
        "observed": None if guard.get("observed") is None else str(guard.get("observed")),
        "reason": guard.get("reason"),
        "contract": f"cytools-public-api=={SUPPORTED_CYTOOLS_API_VERSION}",
    }


def build_replay_certificate(
    index: int,
    record: Mapping[str, Any],
    report: Mapping[str, Any],
    action_record: Mapping[str, Any],
    *,
    action_witness: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a source-keyed certificate only from a fully computed action.

    Return ``None`` for a terminal or incomplete action.  The function does
    not infer a class, FRST, action, or result from an index or target value.
    """

    index = int(index)
    if index not in TARGET_INDICES:
        return None
    record_index = record.get("index")
    if record_index is not None:
        try:
            if isinstance(record_index, bool) or int(record_index) != index:
                return None
        except (TypeError, ValueError):
            return None
    if action_record.get("terminal_status") != "refined_action_evaluated":
        return None
    source = report.get("source_identity")
    fixed = action_record.get("fixed_locus_euler")
    h2 = action_record.get("refined_glsm")
    split = action_record.get("hodge_split")
    if not isinstance(source, Mapping) or not isinstance(fixed, Mapping):
        return None
    if not isinstance(h2, Mapping) or not isinstance(split, Mapping):
        return None
    if fixed.get("status") != "computed" or h2.get("status") != "refined_h2_action_verified":
        return None
    source_sha = source.get("source_sha256")
    source_row = source.get("source_row")
    polytope_id = source.get("polytope_id")
    global_points = source.get("global_points")
    frst_hash = action_record.get("frst_hash") or action_record.get("triangulation_identity")
    witness = action_witness if action_witness is not None else action_record.get("action")
    api = _certificate_api_contract(report)
    evidence_digest = component_h2_evidence_digest(action_record)
    required = (source_sha, source_row, polytope_id, global_points, frst_hash, witness, evidence_digest)
    if any(value is None for value in required):
        return None
    try:
        if isinstance(source_row, bool) or int(source_row) != IMMUTABLE_SOURCE_ROWS[index]:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if source.get("parquet_sha256", source_sha) != source_sha:
        return None
    if SOURCE_DATASET is not None and source.get("source_dataset") == SOURCE_DATASET:
        if SOURCE_PARQUET_SHA256 is None or source_sha != SOURCE_PARQUET_SHA256:
            return None
    report_source = {
        "source_sha256": source_sha,
        "source_row": source_row,
        "polytope_id": polytope_id,
        "global_points": _jsonable(global_points),
    }
    record_source = record.get("source")
    if isinstance(record_source, Mapping):
        record_source_normalized = {
            "source_sha256": record_source.get(
                "source_sha256", record_source.get("parquet_sha256")
            ),
            "source_row": record_source.get("source_row", record_source.get("row_index")),
            "polytope_id": record_source.get("polytope_id"),
            "global_points": _jsonable(record_source.get("global_points")),
        }
        if record_source_normalized != report_source:
            return None
    for field, expected in report_source.items():
        observed = action_record.get(field)
        if observed is not None and _jsonable(observed) != expected:
            return None
    if action_witness is not None and action_record.get("action") is not None:
        try:
            if _action_digest(action_witness) != _action_digest(action_record["action"]):
                return None
        except (TypeError, ValueError):
            return None
    if action_record.get("action_digest") is not None:
        try:
            if action_record["action_digest"] != _action_digest(witness):
                return None
        except (TypeError, ValueError):
            return None
    refinement_records = report.get("refinement_records")
    candidate_index = action_record.get("candidate_index")
    if isinstance(refinement_records, Sequence) and not isinstance(refinement_records, (str, bytes)):
        matching_refinements = [
            row for row in refinement_records
            if isinstance(row, Mapping) and row.get("candidate_index") == candidate_index
        ]
        if not matching_refinements or all(
            row.get("triangulation_identity") != frst_hash
            for row in matching_refinements
        ):
            return None
    if api.get("status") != "verified" or api.get("observed") != SUPPORTED_CYTOOLS_API_VERSION:
        return None
    action = {
        "witness": _jsonable(witness),
        "digest": _action_digest(witness),
    }
    certificate: dict[str, Any] = {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "formula_schema_version": FORMULA_SCHEMA_VERSION,
        "replay_schema_version": report.get("schema_version", SCHEMA_VERSION),
        "index": int(index),
        "source": {
            "dataset": source.get("source_dataset"),
            "source_sha256": source_sha,
            "parquet_sha256": source_sha,
            "source_row": int(source_row),
            "polytope_id": str(polytope_id),
            "global_points": _jsonable(global_points),
        },
        "frst": {
            "frst_hash": str(frst_hash),
            "selected_source_frst_hash": report.get("selected_frst", {}).get("identity"),
            "candidate_index": action_record.get("candidate_index"),
            "index_space": "triangulation_local",
        },
        "action": action,
        "cytools_api_contract": api,
        "evidence": {
            "fixed_locus_euler": _jsonable(fixed),
            "refined_glsm": _jsonable(h2),
            "component_h2_evidence_digest": evidence_digest,
        },
        "result": {
            "terminal_status": action_record.get("terminal_status"),
            "chi_F_I": fixed.get("chi_F_I"),
            "hodge_split": _jsonable(split),
        },
        "provenance": {
            "runtime": _jsonable(report.get("runtime_provenance")),
            "caps": _jsonable(report.get("caps")),
            "source_identity": _jsonable(source),
            "selected_frst": _jsonable(report.get("selected_frst")),
            "fallback_reasons": _fallback_reasons(action_record),
        },
    }
    certificate["certificate_key"] = _certificate_key(certificate)
    certificate["certificate_key_digest"] = _canonical_digest(certificate["certificate_key"])
    certificate["certificate_digest"] = _canonical_digest({
        key: value for key, value in certificate.items()
        if key not in {"certificate_digest", "certificate_key_digest"}
    })
    return certificate


def validate_replay_certificate(
    certificate: Mapping[str, Any] | None,
    *,
    report: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    frst_hash: str | None = None,
    action: Mapping[str, Any] | None = None,
    action_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate certificate identity and evidence without trusting its result."""

    if not isinstance(certificate, Mapping):
        return {"status": "missing", "terminal": True, "reasons": ["certificate is absent"]}
    reasons: list[str] = []
    if certificate.get("replay_schema_version") != SCHEMA_VERSION:
        reasons.append("replay schema version is unsupported")
    if certificate.get("certificate_schema_version") != CERTIFICATE_SCHEMA_VERSION:
        reasons.append("certificate schema version is unsupported")
    if certificate.get("formula_schema_version") != FORMULA_SCHEMA_VERSION:
        reasons.append("formula/schema version is stale or mismatched")
    cert_source = certificate.get("source")
    cert_frst = certificate.get("frst")
    cert_action = certificate.get("action")
    cert_api = certificate.get("cytools_api_contract")
    evidence = certificate.get("evidence")
    if not all(isinstance(item, Mapping) for item in (cert_source, cert_frst, cert_action, cert_api, evidence)):
        return {"status": "invalid", "terminal": True, "reasons": ["certificate identity/evidence sections are incomplete"]}
    cert_index = certificate.get("index")
    try:
        if isinstance(cert_index, bool) or int(cert_index) not in TARGET_INDICES:
            reasons.append("certificate index is outside the bounded replay scope")
        elif int(cert_source.get("source_row")) != IMMUTABLE_SOURCE_ROWS[int(cert_index)]:
            reasons.append("certificate source row does not match its immutable index")
    except (KeyError, TypeError, ValueError):
        reasons.append("certificate index/source row binding is invalid")
    source_sha = cert_source.get("source_sha256")
    parquet_sha = cert_source.get("parquet_sha256")
    if not isinstance(source_sha, str) or not source_sha:
        reasons.append("certificate source SHA-256 is missing")
    if parquet_sha != source_sha:
        reasons.append("certificate parquet/source SHA-256 fields disagree")
    try:
        global_points = cert_source.get("global_points")
        point_array = _integer_rows(global_points, name="certificate global points", dimension=4)
        # The immutable bounded source has eight global points.  Keep that
        # strict gate for source-keyed production certificates; lightweight
        # connector fixtures may use a synthetic coordinate set, provided
        # their own source key and digest are otherwise internally valid.
        if point_array.shape[0] != 8 and source_sha == SOURCE_PARQUET_SHA256:
            reasons.append("certificate global point count is not the immutable eight-point source")
        if len({tuple(int(value) for value in row) for row in point_array.tolist()}) != point_array.shape[0]:
            reasons.append("certificate global coordinates are not unique")
        canonical_id = f"lattice-points-sha256:{point_identity(global_points)}"
        if cert_source.get("polytope_id") != canonical_id:
            reasons.append("certificate polytope_id does not match canonical global coordinates")
    except (TypeError, ValueError):
        reasons.append("certificate global coordinates are invalid")
    if SOURCE_DATASET is not None and cert_source.get("dataset") == SOURCE_DATASET:
        if source_sha != SOURCE_PARQUET_SHA256:
            reasons.append("certificate source SHA-256 does not match the immutable bounded source")
    if cert_api.get("expected") != SUPPORTED_CYTOOLS_API_VERSION or cert_api.get("status") != "verified":
        reasons.append("CYTools/API version contract is not verified")
    if cert_api.get("observed") != cert_api.get("expected"):
        reasons.append("observed CYTools/API version does not match the contract")
    witness = cert_action.get("witness")
    try:
        if cert_action.get("digest") != _action_digest(witness):
            reasons.append("exact action witness digest mismatch")
    except (TypeError, ValueError):
        reasons.append("exact action witness is invalid")
    try:
        actual_evidence_digest = _canonical_digest({
            "fixed_locus_euler": evidence.get("fixed_locus_euler"),
            "refined_glsm": evidence.get("refined_glsm"),
        })
        if evidence.get("component_h2_evidence_digest") != actual_evidence_digest:
            reasons.append("component/H2 evidence digest mismatch")
    except (TypeError, ValueError):
        reasons.append("component/H2 evidence is not JSON-canonical")
    try:
        key = _certificate_key(certificate)
        if certificate.get("certificate_key") != key:
            reasons.append("certificate key does not match bound fields")
        if certificate.get("certificate_key_digest") != _canonical_digest(key):
            reasons.append("certificate key digest mismatch")
        expected_certificate_digest = _canonical_digest({
            key_name: value for key_name, value in certificate.items()
            if key_name not in {"certificate_digest", "certificate_key_digest"}
        })
        if certificate.get("certificate_digest") != expected_certificate_digest:
            reasons.append("certificate digest mismatch")
    except (TypeError, ValueError):
        reasons.append("certificate key is invalid")

    expected_source = source
    if expected_source is None and isinstance(report, Mapping):
        expected_source = report.get("source_identity")
    if isinstance(expected_source, Mapping):
        source_pairs = {
            "source_sha256": expected_source.get("source_sha256"),
            "source_row": expected_source.get("source_row"),
            "polytope_id": expected_source.get("polytope_id"),
            "global_points": expected_source.get("global_points"),
        }
        for name, expected in source_pairs.items():
            if expected is not None and cert_source.get(name) != expected:
                reasons.append(f"source {name} mismatch")
    if frst_hash is not None and cert_frst.get("frst_hash") != str(frst_hash):
        reasons.append("FRST hash mismatch")
    if action is not None:
        try:
            if cert_action.get("digest") != _action_digest(action):
                reasons.append("action digest does not match the live action")
        except (TypeError, ValueError):
            reasons.append("live action is invalid")
    if isinstance(action_record, Mapping):
        actual_digest = component_h2_evidence_digest(action_record)
        if actual_digest is None or actual_digest != evidence.get("component_h2_evidence_digest"):
            reasons.append("certificate evidence does not match the live component/H2 record")
        actual_frst = action_record.get("frst_hash") or action_record.get("triangulation_identity")
        if actual_frst is not None and cert_frst.get("frst_hash") != str(actual_frst):
            reasons.append("certificate FRST does not match the live action record")
    return {
        "status": "valid" if not reasons else "mismatch",
        "terminal": bool(reasons),
        "reasons": reasons,
        "certificate_digest": certificate.get("certificate_digest"),
        "certificate_key_digest": certificate.get("certificate_key_digest"),
    }


def _population_simplex_digest(simplices: Any) -> str:
    """Hash exact FRST simplices independently of ordering."""

    array = _integer_rows(simplices, name="population FRST simplices")
    canonical = sorted(
        tuple(sorted(int(value) for value in row)) for row in array.tolist()
    )
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_population_replay_certificate(
    certificate: Mapping[str, Any] | None,
    *,
    source: Mapping[str, Any] | None = None,
    frst_hash: str | None = None,
    action: Mapping[str, Any] | None = None,
    requested_h11: int | None = None,
) -> dict[str, Any]:
    """Validate one h11=4/5 source/action certificate.

    Keep this contract distinct from :func:`validate_replay_certificate`,
    whose immutable scope is the three h11=2 fixture rows.  This validator
    checks identity and completeness only; the live CYTools geometry remains
    the source of exact replay results.
    """

    if not isinstance(certificate, Mapping):
        return {"status": "missing", "terminal": True, "reasons": ["certificate is absent"]}
    reasons: list[str] = []
    if certificate.get("certificate_schema_version") != POPULATION_CERTIFICATE_SCHEMA_VERSION:
        reasons.append("population certificate schema version is unsupported")
    for name in ("source", "frst", "action", "cytools_api_contract", "evidence", "geometry"):
        if not isinstance(certificate.get(name), Mapping):
            reasons.append(f"population certificate section is missing: {name}")
    cert_source = certificate.get("source")
    cert_frst = certificate.get("frst")
    cert_action = certificate.get("action")
    cert_api = certificate.get("cytools_api_contract")
    evidence = certificate.get("evidence")
    geometry = certificate.get("geometry")
    if reasons:
        return {"status": "invalid", "terminal": True, "reasons": reasons}

    source_sha = cert_source.get("source_sha256")
    partition = cert_source.get("partition")
    partition_sha = cert_source.get("partition_sha256", source_sha)
    source_row = cert_source.get("source_row")
    global_points = cert_source.get("global_points")
    polytope_id = cert_source.get("polytope_id")
    if not isinstance(source_sha, str) or not source_sha:
        reasons.append("population certificate source SHA-256 is missing")
    if not isinstance(partition, str) or not partition:
        reasons.append("population certificate source partition is missing")
    if not isinstance(partition_sha, str) or len(partition_sha) != 64:
        reasons.append("population certificate partition SHA-256 is missing")
    if isinstance(source_row, bool) or not isinstance(source_row, int):
        reasons.append("population certificate source row is not an integer")
    try:
        point_array = _integer_rows(global_points, name="population certificate global points", dimension=4)
        canonical_id = f"lattice-points-sha256:{point_identity(point_array)}"
        if polytope_id != canonical_id:
            reasons.append("population certificate polytope_id does not match global coordinates")
    except (TypeError, ValueError):
        reasons.append("population certificate global coordinates are invalid")

    frst_points = cert_frst.get("points")
    frst_simplices = cert_frst.get("simplices")
    if cert_frst.get("simplices_index_space") != "triangulation_local":
        reasons.append("population certificate FRST must declare triangulation_local indices")
    try:
        local_points = _integer_rows(frst_points, name="population certificate FRST points", dimension=4)
        simplex_array = _integer_rows(frst_simplices, name="population certificate FRST simplices")
        if np.any(simplex_array < 0) or np.any(simplex_array >= len(local_points)):
            reasons.append("population certificate FRST simplex index is outside local points")
        if cert_frst.get("point_digest") != point_identity(local_points):
            reasons.append("population certificate FRST point digest mismatch")
        if cert_frst.get("simplex_digest") != _population_simplex_digest(simplex_array):
            reasons.append("population certificate FRST simplex digest mismatch")
    except (TypeError, ValueError):
        reasons.append("population certificate FRST points or simplices are invalid")
    if not isinstance(cert_frst.get("frst_hash"), str) or not cert_frst.get("frst_hash"):
        reasons.append("population certificate FRST hash is missing")
    if frst_hash is not None and cert_frst.get("frst_hash") != str(frst_hash):
        reasons.append("population certificate FRST hash mismatch")

    witness = cert_action.get("witness")
    try:
        if cert_action.get("digest") != _action_digest(witness):
            reasons.append("population certificate action witness digest mismatch")
    except (TypeError, ValueError):
        reasons.append("population certificate action witness is invalid")
    if action is not None:
        try:
            if cert_action.get("digest") != _action_digest(action):
                reasons.append("population certificate action does not match the live action")
        except (TypeError, ValueError):
            reasons.append("live population action is invalid")

    if cert_api.get("expected") != SUPPORTED_CYTOOLS_API_VERSION or cert_api.get("status") != "verified":
        reasons.append("population certificate CYTools/API contract is not verified")
    if cert_api.get("observed") != cert_api.get("expected"):
        reasons.append("population certificate observed CYTools version does not match")
    if not isinstance(certificate.get("formula_schema_version"), str):
        reasons.append("population certificate formula schema version is missing")
    fixed = evidence.get("fixed_locus_euler")
    h2 = evidence.get("refined_glsm")
    if not isinstance(fixed, Mapping) or fixed.get("status") != "computed" or not isinstance(fixed.get("components"), list):
        reasons.append("population certificate fixed-component Euler evidence is incomplete")
    if not isinstance(h2, Mapping) or h2.get("status") != "refined_h2_action_verified" or not isinstance(h2.get("h2_matrix"), list):
        reasons.append("population certificate refined H2 evidence is incomplete")
    try:
        evidence_digest = _canonical_digest({"fixed_locus_euler": fixed, "refined_glsm": h2})
        if evidence.get("component_h2_evidence_digest") != evidence_digest:
            reasons.append("population certificate component/H2 evidence digest mismatch")
    except (TypeError, ValueError):
        reasons.append("population certificate component/H2 evidence is not canonical")
    if requested_h11 is not None and geometry.get("physical_h11") != int(requested_h11):
        reasons.append("population certificate physical h11 does not match the run")

    if isinstance(source, Mapping):
        expected_pairs = {
            "source_sha256": source.get("source_sha256", source.get("parquet_sha256")),
            "source_row": source.get("source_row"),
            "population_polytope_index": source.get("population_polytope_index"),
            "polytope_id": source.get("polytope_id"),
            "global_points": source.get("global_points"),
            "partition": source.get("partition"),
            "partition_sha256": source.get("partition_sha256"),
        }
        for name, expected in expected_pairs.items():
            if expected is not None and cert_source.get(name) != expected:
                reasons.append(f"population certificate source {name} mismatch")

    key = {
        "source_sha256": source_sha,
        "partition": partition,
        "partition_sha256": partition_sha,
        "source_row": source_row,
        "polytope_id": polytope_id,
        "global_points": global_points,
        "frst_hash": cert_frst.get("frst_hash"),
        "frst_points": frst_points,
        "frst_simplices": frst_simplices,
        "action_witness": witness,
        "action_digest": cert_action.get("digest"),
        "formula_schema_version": certificate.get("formula_schema_version"),
        "cytools_api_contract": cert_api,
        "component_h2_evidence_digest": evidence.get("component_h2_evidence_digest"),
        "physical_h11": geometry.get("physical_h11"),
    }
    if certificate.get("certificate_key") != key:
        reasons.append("population certificate key does not match bound fields")
    if certificate.get("certificate_key_digest") != _canonical_digest(key):
        reasons.append("population certificate key digest mismatch")
    expected_digest = _canonical_digest({
        key_name: value
        for key_name, value in certificate.items()
        if key_name not in {"certificate_digest", "certificate_key_digest"}
    })
    if certificate.get("certificate_digest") != expected_digest:
        reasons.append("population certificate digest mismatch")
    return {
        "status": "valid" if not reasons else "mismatch",
        "terminal": bool(reasons),
        "reasons": reasons,
        "certificate_digest": certificate.get("certificate_digest"),
        "certificate_key_digest": certificate.get("certificate_key_digest"),
    }


def validate_population_input_certificate(
    certificate: Mapping[str, Any] | None,
    *,
    source: Mapping[str, Any] | None = None,
    frst_hash: str | None = None,
    action: Mapping[str, Any] | None = None,
    requested_h11: int | None = None,
) -> dict[str, Any]:
    """Validate h11=4/5 source and action identity before live replay.

    This validator deliberately does not inspect fixed-component Euler,
    refined H2, or formula results.  Those are output evidence and must be
    computed against the reconstructed CYTools geometry during replay.
    """

    if not isinstance(certificate, Mapping):
        return {"status": "missing", "terminal": True, "reasons": ["input certificate is absent"]}
    reasons: list[str] = []
    if certificate.get("certificate_schema_version") != POPULATION_INPUT_CERTIFICATE_SCHEMA_VERSION:
        reasons.append("population input certificate schema version is unsupported")
    for name in ("source", "frst", "action", "cytools_api_contract", "geometry"):
        if not isinstance(certificate.get(name), Mapping):
            reasons.append(f"population input certificate section is missing: {name}")
    if reasons:
        return {"status": "invalid", "terminal": True, "reasons": reasons}

    cert_source = certificate["source"]
    cert_frst = certificate["frst"]
    cert_action = certificate["action"]
    cert_api = certificate["cytools_api_contract"]
    geometry = certificate["geometry"]
    source_sha = cert_source.get("source_sha256")
    partition_sha = cert_source.get("partition_sha256", source_sha)
    source_row = cert_source.get("source_row")
    global_points = cert_source.get("global_points")
    polytope_id = cert_source.get("polytope_id")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        reasons.append("population input source SHA-256 is missing or malformed")
    if not isinstance(partition_sha, str) or len(partition_sha) != 64:
        reasons.append("population input partition SHA-256 is missing or malformed")
    if isinstance(source_row, bool) or not isinstance(source_row, int) or source_row < 0:
        reasons.append("population input source row is not a nonnegative integer")
    try:
        point_array = _integer_rows(global_points, name="population input global points", dimension=4)
        canonical_id = f"lattice-points-sha256:{point_identity(point_array)}"
        if polytope_id != canonical_id:
            reasons.append("population input polytope_id does not match global coordinates")
    except (TypeError, ValueError):
        reasons.append("population input global coordinates are invalid")
    if geometry.get("physical_h11") is not None and requested_h11 is not None:
        if geometry.get("physical_h11") != int(requested_h11):
            reasons.append("population input physical h11 does not match the run")
    if cert_frst.get("simplices_index_space") != "triangulation_local":
        reasons.append("population input FRST must declare triangulation_local simplices")
    try:
        local_points = _integer_rows(cert_frst.get("points"), name="population input FRST points", dimension=4)
        simplices = _integer_rows(cert_frst.get("simplices"), name="population input FRST simplices")
        if np.any(simplices < 0) or np.any(simplices >= len(local_points)):
            reasons.append("population input FRST simplex index is outside local points")
        if cert_frst.get("point_digest") != point_identity(local_points):
            reasons.append("population input FRST point digest mismatch")
        if cert_frst.get("simplex_digest") != _population_simplex_digest(simplices):
            reasons.append("population input FRST simplex digest mismatch")
    except (TypeError, ValueError):
        reasons.append("population input FRST points or simplices are invalid")
    if not isinstance(cert_frst.get("frst_hash"), str) or not cert_frst.get("frst_hash"):
        reasons.append("population input FRST hash is missing")
    if frst_hash is not None and cert_frst.get("frst_hash") != str(frst_hash):
        reasons.append("population input FRST hash mismatch")
    witness = cert_action.get("witness")
    try:
        expected_action_digest = _action_digest(witness)
        if cert_action.get("digest") != expected_action_digest:
            reasons.append("population input action witness digest mismatch")
        binary_shift = witness.get("torus_shift_binary_source")
        if (
            not isinstance(binary_shift, list)
            or len(binary_shift) != 4
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in binary_shift)
        ):
            reasons.append("population input binary torus shift is missing or malformed")
        else:
            matrix = _integer_rows(witness.get("lattice_matrix"), name="population input lattice matrix")
            if matrix.shape != (4, 4):
                reasons.append("population input lattice matrix has the wrong shape")
            projected = (np.eye(4, dtype=np.int64) + matrix) @ np.asarray(binary_shift, dtype=np.int64)
            projected = [int(value) for value in projected.tolist()]
            if witness.get("torus_shift_projected_numerator") != projected:
                reasons.append("population input projected torus numerator mismatch")
            candidate_payload = [witness.get("matrix_id"), projected, int(witness["lambda_f"])]
            candidate_digest = hashlib.sha256(
                json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if witness.get("candidate_id") != candidate_digest:
                reasons.append("population input candidate_id does not bind raw binary shift")
            # The source writer stores ``vector / 2`` as the rational shift,
            # while ``vector`` is itself ``projected / 2``.  Therefore the
            # normalized witness shift is projected / 4; compare exact
            # rationals rather than the unnormalized JSON representation.
            observed_shift = witness.get("torus_shift")
            observed_rational = tuple(
                Fraction(int(value), int(observed_shift["denominator"]))
                for value in observed_shift["numerator"]
            ) if isinstance(observed_shift, Mapping) else None
            expected_rational = tuple(Fraction(int(value), 4) for value in projected)
            if observed_rational != expected_rational:
                reasons.append("population input projected and rational torus shifts disagree")
        if action is not None and cert_action.get("digest") != _action_digest(action):
            reasons.append("population input action does not match the live action")
    except (TypeError, ValueError):
        reasons.append("population input action witness is invalid")
    if cert_api.get("expected") != SUPPORTED_CYTOOLS_API_VERSION or cert_api.get("status") != "verified":
        reasons.append("population input CYTools/API version is not verified")
    if cert_api.get("observed") != cert_api.get("expected"):
        reasons.append("population input observed CYTools version does not match")
    if isinstance(source, Mapping):
        for name, expected in {
            "source_sha256": source.get("source_sha256", source.get("parquet_sha256")),
            "source_row": source.get("source_row"),
            "polytope_id": source.get("polytope_id"),
            "global_points": source.get("global_points"),
            "partition": source.get("partition"),
            "partition_sha256": source.get("partition_sha256"),
        }.items():
            if expected is not None and cert_source.get(name) != expected:
                reasons.append(f"population input source {name} mismatch")
    key = {
        "source_sha256": source_sha,
        "partition": cert_source.get("partition"),
        "partition_sha256": partition_sha,
        "source_row": source_row,
        "population_polytope_index": cert_source.get("population_polytope_index"),
        "polytope_id": polytope_id,
        "global_points": global_points,
        "frst_hash": cert_frst.get("frst_hash"),
        "frst_points": cert_frst.get("points"),
        "frst_simplices": cert_frst.get("simplices"),
        "action_witness": witness,
        "action_digest": cert_action.get("digest"),
        "cytools_api_contract": cert_api,
        "physical_h11": geometry.get("physical_h11"),
    }
    if certificate.get("certificate_key") != key:
        reasons.append("population input certificate key does not match bound fields")
    if certificate.get("certificate_key_digest") != _canonical_digest(key):
        reasons.append("population input certificate key digest mismatch")
    expected_digest = _canonical_digest({
        key_name: value
        for key_name, value in certificate.items()
        if key_name not in {"certificate_digest", "certificate_key_digest"}
    })
    if certificate.get("certificate_digest") != expected_digest:
        reasons.append("population input certificate digest mismatch")
    return {
        "status": "valid" if not reasons else "mismatch",
        "terminal": bool(reasons),
        "reasons": reasons,
        "certificate_digest": certificate.get("certificate_digest"),
        "certificate_key_digest": certificate.get("certificate_key_digest"),
    }


# Short aliases make the bridge discoverable to callers that use the ledger's
# vocabulary while preserving the explicit names above for new code.
make_mpcp_certificate = build_replay_certificate
verify_mpcp_certificate = validate_replay_certificate


def _integer_rows(values: Any, *, name: str, dimension: int | None = None) -> np.ndarray:
    """Decode an exact integer row array and reject silent coercions."""

    array = np.asarray(values)
    if array.ndim != 2 or (dimension is not None and array.shape[1] != dimension):
        raise ValueError(f"{name} must be a two-dimensional array")
    try:
        integers = np.asarray(array, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not np.array_equal(array, integers):
        raise ValueError(f"{name} must contain exact integers")
    return integers


def triangulation_identity(triangulation: Any) -> str:
    """Hash a triangulation independently of simplex and cell ordering."""

    if not _callable(triangulation, "simplices"):
        raise ValueError("triangulation does not expose public simplices()")
    simplices = np.asarray(triangulation.simplices(as_indices=True), dtype=np.int64)
    if simplices.ndim != 2:
        raise ValueError("triangulation simplices must be a two-dimensional array")
    canonical = sorted(tuple(sorted(int(value) for value in row)) for row in simplices.tolist())
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def point_identity(points: Any) -> str:
    """Hash a lattice point configuration without relying on catalog labels."""

    rows = _integer_rows(points, name="point configuration")
    canonical = sorted(tuple(int(value) for value in row) for row in rows.tolist())
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def _normalise_polytope_identity(value: Any) -> str:
    """Normalize the persisted lattice-point identity prefix."""

    text = str(value)
    prefix = "lattice-points-sha256:"
    return text[len(prefix):] if text.startswith(prefix) else text


def _coordinate_index_map(parent_points: Any, local_points: Any, *, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Map triangulation-local coordinates to one frozen parent point order."""

    parent = _integer_rows(parent_points, name="global polytope points", dimension=4)
    local = _integer_rows(local_points, name=name, dimension=4)
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(parent)}
    if len(lookup) != len(parent):
        raise ValueError("global polytope point coordinates are not unique")
    if len({tuple(int(value) for value in row) for row in local.tolist()}) != len(local):
        raise ValueError(f"{name} contains duplicate point coordinates")
    mapped: list[int] = []
    missing: list[list[int]] = []
    for row in local.tolist():
        key = tuple(int(value) for value in row)
        if key not in lookup:
            missing.append(list(key))
        else:
            mapped.append(int(lookup[key]))
    if missing:
        raise ValueError(f"{name} contains coordinates absent from the global polytope: {missing}")
    mapped_array = np.asarray(mapped, dtype=np.int64)
    return mapped_array, {
        "status": "local_to_global_coordinates_verified",
        "index_space": "triangulation_local",
        "local_count": int(len(local)),
        "global_count": int(len(parent)),
        "local_to_global": mapped_array.tolist(),
        "coordinates_exact": True,
    }


def _source_identity_evidence(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Require a source-row key, exact global coordinates, and expected Hodge data."""

    source = record.get("source")
    if not isinstance(source, Mapping):
        return {
            "status": "source_identity_unavailable",
            "reason": "replay manifests require a source mapping keyed by polytope_id and source_row",
            "terminal": True,
        }
    source_id = source.get("polytope_id")
    source_row = source.get("source_row", source.get("row_index"))
    source_sha_field = source.get("source_sha256")
    parquet_sha_field = source.get("parquet_sha256")
    source_sha = parquet_sha_field or source_sha_field
    global_points = source.get("global_points", record.get("global_points"))
    missing = [
        name for name, value in (
            ("polytope_id", source_id),
            ("source_row", source_row),
            ("source_sha256", source_sha),
            ("global_points", global_points),
            ("expected_hodge", source.get("expected_hodge", record.get("expected_hodge"))),
        ) if value is None
    ]
    if missing:
        return {
            "status": "source_identity_incomplete",
            "reason": f"source mapping is missing required fields: {', '.join(missing)}",
            "missing": missing,
            "terminal": True,
        }
    try:
        points = _integer_rows(global_points, name="source global points", dimension=4)
        if isinstance(source_row, bool) or int(source_row) != source_row:
            raise ValueError("source_row must be an integer")
        expected_hodge = source.get("expected_hodge", record.get("expected_hodge"))
        if not isinstance(expected_hodge, Mapping):
            raise ValueError("expected_hodge must be a mapping")
        expected_hodge_values = {}
        for key in ("h11", "h21", "chi"):
            value = expected_hodge[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"expected_hodge[{key}] must be an integer")
            expected_hodge_values[key] = int(value)
        expected_hodge = expected_hodge_values
        source_row_expected = IMMUTABLE_SOURCE_ROWS.get(index)
        point_count_expected = source.get("expected_point_count", 8)
        boundary_count_expected = source.get("expected_boundary_point_count", 7)
        for name, value in (
            ("expected_point_count", point_count_expected),
            ("expected_boundary_point_count", boundary_count_expected),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        point_count_expected = int(point_count_expected)
        boundary_count_expected = int(boundary_count_expected)
    except (TypeError, ValueError, KeyError) as exc:
        return {
            "status": "source_identity_invalid",
            "reason": str(exc),
            "terminal": True,
        }
    expected_id = _normalise_polytope_identity(source_id)
    actual_id = point_identity(points)
    result: dict[str, Any] = {
        "status": "source_identity_ready",
        "index": int(index),
        "polytope_id": str(source_id),
        "source_row": int(source_row),
        "expected_source_row": source_row_expected,
        "source_row_match": source_row_expected is None or int(source_row) == source_row_expected,
        "source_dataset": source.get("dataset"),
        "source_path": source.get("parquet_file", source.get("source_path")),
        "source_sha256": source.get("parquet_sha256", source.get("source_sha256")),
        "global_points": points.tolist(),
        "global_point_count": int(len(points)),
        "expected_point_count": int(point_count_expected),
        "expected_boundary_point_count": int(boundary_count_expected),
        "expected_hodge": expected_hodge,
        "actual_polytope_id": f"lattice-points-sha256:{actual_id}",
        "polytope_id_match": expected_id == actual_id,
        "terminal": False,
    }
    if source_row_expected is not None and int(source_row) != source_row_expected:
        result.update({
            "status": "source_row_mismatch",
            "reason": (
                f"source_row {source_row} does not match immutable class {index} row "
                f"{source_row_expected}"
            ),
            "terminal": True,
        })
    elif (
        source_sha_field is not None
        and parquet_sha_field is not None
        and source_sha_field != parquet_sha_field
    ):
        result.update({
            "status": "source_sha256_mismatch",
            "reason": "source parquet/source SHA-256 fields disagree",
            "terminal": True,
        })
    elif SOURCE_DATASET is not None and source.get("dataset") == SOURCE_DATASET and source_sha != SOURCE_PARQUET_SHA256:
        result.update({
            "status": "source_sha256_mismatch",
            "reason": "source SHA-256 does not match the immutable bounded source",
            "terminal": True,
        })
    elif expected_id != actual_id:
        result.update({
            "status": "source_polytope_id_mismatch",
            "reason": "source polytope_id does not match the canonical global coordinate identity",
            "terminal": True,
        })
    elif len(points) != int(point_count_expected):
        result.update({
            "status": "source_point_count_mismatch",
            "reason": "source global point count does not match its immutable source certificate",
            "terminal": True,
        })
    supplied_points = record.get("points", record.get("polytope_points"))
    if supplied_points is not None:
        try:
            supplied = _integer_rows(supplied_points, name="replay global points", dimension=4)
            if point_identity(supplied) != actual_id:
                result.update({
                    "status": "source_global_coordinates_mismatch",
                    "reason": "record points disagree with source.global_points",
                    "terminal": True,
                })
        except (TypeError, ValueError) as exc:
            result.update({
                "status": "source_global_coordinates_invalid",
                "reason": str(exc),
                "terminal": True,
            })
    return result


def _triangulation_flags(triangulation: Any) -> dict[str, Any]:
    """Read public fine/regular/star/valid flags with terminal evidence."""

    result: dict[str, Any] = {}
    for name in ("is_fine", "is_regular", "is_star", "is_valid"):
        method = getattr(triangulation, name, None)
        if not callable(method):
            result[name] = None
            result[f"{name}_reason"] = "public API is absent"
            continue
        try:
            result[name] = bool(method())
        except Exception as exc:
            result[name] = None
            result[f"{name}_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _hodge_values(obj: Any) -> dict[str, int]:
    """Read Hodge data using the public object API without changing conventions."""

    values: dict[str, int] = {}
    for name in ("h11", "h21", "chi"):
        method = getattr(obj, name, None)
        if not callable(method):
            raise ValueError(f"object does not expose public {name}()")
        last_error: Exception | None = None
        for kwargs in ({}, {"lattice": "N"}):
            try:
                values[name] = int(method(**kwargs))
                last_error = None
                break
            except (TypeError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            raise ValueError(f"public {name}() failed: {last_error}")
    return values


def _hodge_match_evidence(actual: Mapping[str, Any], expected: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Compare all three Hodge/Euler values and fail closed on any mismatch."""

    observed = {key: int(actual[key]) for key in ("h11", "h21", "chi")}
    wanted = {key: int(expected[key]) for key in ("h11", "h21", "chi")}
    mismatch = {
        key: {"expected": wanted[key], "observed": observed[key]}
        for key in wanted if wanted[key] != observed[key]
    }
    return {
        "status": "matched" if not mismatch else "mismatch",
        "name": name,
        "expected": wanted,
        "observed": observed,
        "mismatch": mismatch,
        "terminal": bool(mismatch),
    }


def _points_from_object(obj: Any, *, name: str = "points") -> np.ndarray:
    """Read point coordinates from a CYTools-like object."""

    method = getattr(obj, "points", None)
    if not callable(method):
        raise ValueError(f"{name} object does not expose points()")
    return _integer_rows(method(), name=name, dimension=4)


def _point_permutation(points: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the exact point permutation induced by ``matrix``."""

    if matrix.shape != (4, 4):
        return np.asarray([], dtype=np.int64), {
            "status": "lattice_action_invalid",
            "reason": f"lattice action shape {matrix.shape} is not (4, 4)",
        }
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(points)}
    mapped: list[int] = []
    for point in points:
        image = tuple(int(value) for value in matrix @ point)
        if image not in lookup:
            return np.asarray([], dtype=np.int64), {
                "status": "point_set_not_preserved",
                "mapped_point": list(image),
            }
        mapped.append(lookup[image])
    return np.asarray(mapped, dtype=np.int64), {"status": "point_set_preserved"}


def fan_action_evidence(poly: Any, triangulation: Any, matrix: Any) -> dict[str, Any]:
    """Check exact cell-complex preservation for a selected/refined fan."""

    points = _points_from_object(poly)
    matrix_array = _integer_rows(matrix, name="lattice action", dimension=4)
    if matrix_array.shape != (4, 4):
        raise ValueError("lattice action must have shape (4, 4)")
    mapped_indices, point_evidence = _point_permutation(points, matrix_array)
    if point_evidence["status"] != "point_set_preserved":
        return {"status": "fan_not_preserved", **point_evidence}
    tri_points = _points_from_object(triangulation, name="triangulation points")
    point_lookup = {tuple(int(value) for value in row): index for index, row in enumerate(points)}
    try:
        tri_global = np.asarray([point_lookup[tuple(int(value) for value in row)] for row in tri_points], dtype=np.int64)
        simplices = np.asarray(triangulation.simplices(as_indices=True), dtype=np.int64)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return {"status": "fan_evidence_unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    cells = {tuple(sorted(int(tri_global[int(index)]) for index in cell)) for cell in simplices}
    mapped_cells = {tuple(sorted(int(mapped_indices[index]) for index in cell)) for cell in cells}
    return {
        "status": "fan_preserved" if cells == mapped_cells else "fan_not_preserved",
        "cell_count": len(cells),
        "point_count": len(points),
        "reason": None if cells == mapped_cells else "action maps cells outside the cell complex",
    }


def _fraction_value(value: Any) -> Fraction:
    """Convert an API height to an exact rational without binary rounding."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        return Fraction(str(float(value)))
    return Fraction(value)


def symmetric_heights(poly: Any, triangulation: Any, matrix: Any) -> dict[str, Any]:
    """Build Moritz's exact equivariant height average from a selected FRST."""

    matrix_array = _integer_rows(matrix, name="lattice action", dimension=4)
    if not _callable(triangulation, "heights"):
        return {
            "status": "symmetric_heights_unavailable",
            "reason": "Triangulation.heights() is absent; no height vector is invented",
            "fallback": None,
        }
    try:
        raw_heights = np.asarray(triangulation.heights()).reshape(-1)
        tri_points = _points_from_object(triangulation, name="triangulation points")
        poly_points = _points_from_object(poly)
    except Exception as exc:
        return {
            "status": "symmetric_heights_unavailable",
            "reason": f"CYTools height/point API failed: {type(exc).__name__}: {exc}",
            "fallback": None,
        }
    if raw_heights.size != tri_points.shape[0]:
        return {
            "status": "symmetric_heights_unavailable",
            "reason": f"height length {raw_heights.size} != triangulation point count {tri_points.shape[0]}",
            "fallback": None,
        }
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(poly_points)}
    poly_perm, point_evidence = _point_permutation(poly_points, matrix_array)
    if point_evidence["status"] != "point_set_preserved":
        return {"status": "symmetric_heights_unavailable", "reason": point_evidence, "fallback": None}
    try:
        tri_global = [lookup[tuple(int(value) for value in row)] for row in tri_points]
        global_to_tri = {global_index: tri_index for tri_index, global_index in enumerate(tri_global)}
        image_tri = [global_to_tri[int(poly_perm[global_index])] for global_index in tri_global]
    except KeyError as exc:
        return {
            "status": "symmetric_heights_unavailable",
            "reason": f"selected triangulation omits an action image: {exc}",
            "fallback": None,
        }
    heights = tuple(_fraction_value(value) for value in raw_heights.tolist())
    averaged = tuple((heights[index] + heights[image_tri[index]]) / 2 for index in range(len(heights)))
    invariant = all(averaged[index] == averaged[image_tri[index]] for index in range(len(averaged)))
    origin_indices = [index for index, point in enumerate(tri_points) if not np.any(point)]
    return {
        "status": "symmetric_heights_ready" if invariant else "symmetric_heights_failure",
        "heights": [_jsonable(value) for value in averaged],
        "height_count": len(averaged),
        "origin_height": _jsonable(averaged[origin_indices[0]]) if origin_indices else None,
        "origin_height_indices": origin_indices,
        "invariant_exact": invariant,
        "formula": "h[p]=(h_prime[p]+h_prime[L(p)])/2",
        "source_anchor": "Moritz KS_orientifolds.tex lines 471-489",
        "fallback": None,
    }


def _exact_h2_action_from_glsm(glsm: Any, image_indices: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the full refined relation ``M Q = Q P`` exactly."""

    q = _integer_rows(glsm, name="refined GLSM matrix")
    image = np.asarray(image_indices, dtype=np.int64).reshape(-1)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
        raise ValueError("refined GLSM matrix must be non-empty")
    if image.size != q.shape[1] or sorted(int(value) for value in image) != list(range(image.size)):
        raise ValueError("refined divisor image is not a column permutation")
    rows, columns = q.shape
    q_matrix = [[Fraction(int(value)) for value in row] for row in q.tolist()]
    transformed = [[Fraction(0) for _ in range(columns)] for _ in range(rows)]
    for source, target in enumerate(image):
        for row in range(rows):
            transformed[row][int(target)] += q_matrix[row][source]

    def rank(matrix: list[list[Fraction]]) -> int:
        reduced = [row[:] for row in matrix]
        row_count = len(reduced)
        column_count = len(reduced[0]) if reduced else 0
        pivot_row = 0
        rank_value = 0
        for column in range(column_count):
            pivot = next((r for r in range(pivot_row, row_count) if reduced[r][column]), None)
            if pivot is None:
                continue
            reduced[pivot_row], reduced[pivot] = reduced[pivot], reduced[pivot_row]
            pivot_value = reduced[pivot_row][column]
            reduced[pivot_row] = [value / pivot_value for value in reduced[pivot_row]]
            for r in range(row_count):
                if r == pivot_row or not reduced[r][column]:
                    continue
                factor = reduced[r][column]
                reduced[r] = [a - factor * b for a, b in zip(reduced[r], reduced[pivot_row])]
            pivot_row += 1
            rank_value += 1
            if pivot_row == row_count:
                break
        return rank_value

    rank_value = rank(q_matrix)
    if rank_value != rows:
        raise ValueError("refined GLSM matrix does not have full row rank")

    def inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
        size = len(matrix)
        augmented = [
            row[:] + [Fraction(int(row_index == column_index)) for column_index in range(size)]
            for row_index, row in enumerate(matrix)
        ]
        for column in range(size):
            pivot = next((r for r in range(column, size) if augmented[r][column]), None)
            if pivot is None:
                raise ValueError("selected GLSM minor is singular")
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            pivot_value = augmented[column][column]
            augmented[column] = [value / pivot_value for value in augmented[column]]
            for r in range(size):
                if r == column or not augmented[r][column]:
                    continue
                factor = augmented[r][column]
                augmented[r] = [a - factor * b for a, b in zip(augmented[r], augmented[column])]
        return [row[size:] for row in augmented]

    def multiply(left: list[list[Fraction]], right: list[list[Fraction]]) -> list[list[Fraction]]:
        return [
            [sum(left[row][inner] * right[inner][column] for inner in range(len(right)))
             for column in range(len(right[0]))]
            for row in range(len(left))
        ]

    def exact_determinant(matrix: list[list[Fraction]]) -> Fraction:
        reduced = [row[:] for row in matrix]
        size = len(reduced)
        answer = Fraction(1)
        sign = 1
        for column in range(size):
            pivot = next((r for r in range(column, size) if reduced[r][column]), None)
            if pivot is None:
                return Fraction(0)
            if pivot != column:
                reduced[column], reduced[pivot] = reduced[pivot], reduced[column]
                sign *= -1
            pivot_value = reduced[column][column]
            answer *= pivot_value
            for r in range(column + 1, size):
                if not reduced[r][column]:
                    continue
                factor = reduced[r][column] / pivot_value
                reduced[r] = [a - factor * b for a, b in zip(reduced[r], reduced[column])]
        return answer * sign

    selected = None
    determinant = None
    for candidate_columns in itertools.combinations(range(columns), rows):
        minor = [[q_matrix[row][column] for column in candidate_columns] for row in range(rows)]
        det = exact_determinant(minor)
        if det:
            if det.denominator != 1:
                raise ValueError("selected GLSM minor determinant is nonintegral")
            selected, determinant = tuple(int(column) for column in candidate_columns), int(det)
            break
    if selected is None:
        raise ValueError("no nonzero full-row-rank minor in refined GLSM matrix")
    basis_minor = [[q_matrix[row][column] for column in selected] for row in range(rows)]
    transformed_minor = [[transformed[row][column] for column in selected] for row in range(rows)]
    candidate = multiply(transformed_minor, inverse(basis_minor))
    if any(value.denominator != 1 for row in candidate for value in row):
        raise ValueError("refined GLSM action is not integral")
    residual = multiply(candidate, q_matrix)
    if any(residual[row][column] != transformed[row][column] for row in range(rows) for column in range(columns)):
        raise ValueError("refined GLSM relation residual is nonzero")
    matrix = np.asarray([[int(value) for value in row] for row in candidate], dtype=np.int64)
    if not np.array_equal(matrix @ matrix, np.eye(q.shape[0], dtype=np.int64)):
        raise ValueError("refined H2 action is not an involution")
    proof = {
        "method": "exact_full_refined_glsm_quotient_relation",
        "equation": "M Q_prime = Q_prime P",
        "Q_shape": [int(value) for value in q.shape],
        "P_shape": [int(image.size), int(image.size)],
        "Q_rank": rank_value,
        "selected_column_indices": list(selected),
        "selected_minor_determinant": determinant,
        "exact_rational_solution": True,
        "integral_solution": True,
        "exact_residual_zero": True,
        "involution_verified_exactly": True,
        "fallback": "project exact solver; CYTools exposes Q_prime but no orientifold H2 action",
        "fallback_reason": "CYTools exposes the refined charge matrix but does not expose an orientifold H2-action solver",
    }
    return matrix, proof


def refined_glsm_evidence(
    cy: Any,
    matrix: Any,
    points: Any,
    *,
    original_prime_labels: Sequence[int] | None = None,
    original_points: Any = None,
) -> dict[str, Any]:
    """Build the full refined divisor permutation and exact ``H^2`` action."""

    if not _callable(cy, "glsm_charge_matrix") or not _callable(cy, "prime_toric_divisors"):
        return {
            "status": "incomplete_refined_glsm",
            "reason": "CYTools refined GLSM or prime-divisor API is absent",
            "fallback": "exact solver cannot construct Q_prime without full divisor columns",
            "fallback_reason": "the exact solver requires both the full refined charge matrix and prime divisor columns",
        }
    try:
        q = np.asarray(cy.glsm_charge_matrix(include_origin=False))
        labels = np.asarray(cy.prime_toric_divisors(), dtype=np.int64).reshape(-1)
        points_array = _integer_rows(points, name="refined point configuration", dimension=4)
        action = _integer_rows(matrix, name="lattice action", dimension=4)
        point_perm, point_status = _point_permutation(points_array, action)
    except Exception as exc:
        return {"status": "incomplete_refined_glsm", "reason": f"{type(exc).__name__}: {exc}"}
    if point_status["status"] != "point_set_preserved":
        return {"status": "incomplete_refined_glsm", "reason": point_status}
    if labels.size == 0 or q.ndim != 2 or q.shape[1] != labels.size:
        return {
            "status": "incomplete_refined_glsm",
            "reason": f"Q_prime shape {q.shape} does not match prime labels {labels.shape}",
        }
    prime_positions = {int(label): position for position, label in enumerate(labels)}
    try:
        image = np.asarray([prime_positions[int(point_perm[int(label)])] for label in labels], dtype=np.int64)
    except (KeyError, IndexError) as exc:
        return {
            "status": "refined_prime_divisors_not_preserved",
            "reason": f"action does not preserve the refined prime-divisor set: {exc}",
            "prime_labels": labels.tolist(),
        }
    try:
        h2, proof = _exact_h2_action_from_glsm(q, image)
    except (ArithmeticError, TypeError, ValueError) as exc:
        return {
            "status": "nonintegral_refined_h2_action",
            "reason": str(exc),
            "prime_labels": labels.tolist(),
            "Q_prime_shape": list(q.shape),
            "fallback": "project exact full refined GLSM solver",
            "fallback_reason": "CYTools supplies Q_prime and divisor permutation data, while the exact quotient solve is project code",
        }
    original_set = set(int(value) for value in (original_prime_labels or ()))
    exceptional_reason = "label comparison; original point coordinates were not supplied"
    if original_points is not None and original_prime_labels is not None:
        try:
            original_points_array = _integer_rows(
                original_points, name="original refined point configuration", dimension=4
            )
            original_coordinates = {
                tuple(int(value) for value in original_points_array[int(label)])
                for label in original_set
            }
            exceptional = [
                int(label)
                for label in labels
                if tuple(int(value) for value in points_array[int(label)])
                not in original_coordinates
            ]
            exceptional_reason = "coordinate comparison against selected FRST prime rays"
        except (IndexError, TypeError, ValueError) as exc:
            return {
                "status": "incomplete_refined_glsm",
                "reason": f"exceptional-ray coordinate comparison failed: {type(exc).__name__}: {exc}",
            }
    else:
        exceptional = [int(label) for label in labels if int(label) not in original_set]
    return {
        "status": "refined_h2_action_verified",
        "Q_prime_shape": list(q.shape),
        "prime_labels": labels.tolist(),
        "prime_image_indices": image.tolist(),
        "exceptional_prime_labels": exceptional,
        "exceptional_rays_included": True,
        "exceptional_ray_comparison": exceptional_reason,
        "prime_divisor_semantics": (
            "CYTools prime_toric_divisors() labels the divisor columns used by the "
            "CalabiYau hypersurface; all returned refined columns, including "
            "exceptional height-one rays, enter Q_prime"
        ),
        "h2_matrix": h2.tolist(),
        "proof": proof,
    }


def resolved_hodge_evidence(cy: Any) -> dict[str, Any]:
    """Read resolved Hodge/Euler invariants without assuming ``h11=2``."""

    try:
        values = _hodge_values(cy)
    except (TypeError, ValueError, OverflowError) as exc:
        return {"status": "resolved_hodge_unavailable", "reason": str(exc)}
    return {"status": "resolved_hodge_read", **values, "h11_assumption": None}


def _public_geometry_evidence(poly: Any, triangulation: Any, cy: Any = None) -> dict[str, Any]:
    """Call public lattice/fan/toric/CY methods and retain their API outcomes."""

    evidence: dict[str, Any] = {"fallbacks": []}

    def call(label: str, obj: Any, name: str, *args: Any, **kwargs: Any) -> None:
        method = getattr(obj, name, None)
        if not callable(method):
            evidence[label] = {"status": "api_absent", "method": name}
            evidence["fallbacks"].append({
                "operation": label,
                "reason": f"public {name}() is absent",
                "fallback": "no geometric outcome inferred",
                "fallback_reason": f"public {name}() is absent; no geometric outcome is inferred",
            })
            return
        try:
            value = method(*args, **kwargs)
            shape = list(np.asarray(value).shape) if isinstance(value, np.ndarray) else None
            evidence[label] = {"status": "cytools_builtin", "method": name, "shape": shape}
        except Exception as exc:
            evidence[label] = {
                "status": "api_failed",
                "method": name,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            evidence["fallbacks"].append({
                "operation": label,
                "reason": evidence[label]["reason"],
                "fallback": "exact boundary checks only; result remains unavailable",
                "fallback_reason": "the CYTools public call failed; exact boundary checks do not infer the missing geometric result",
            })

    call("lattice_points", poly, "points")
    call("lattice_vertices", poly, "vertices")
    call("lattice_facets", poly, "facets")
    call("dual_polytope", poly, "dual")
    call("polytope_automorphisms", poly, "automorphisms")
    call("selected_fan", triangulation, "fan")
    call("selected_sr_ideal", triangulation, "sr_ideal")
    call("toric_variety", triangulation, "get_toric_variety")
    toric = None
    if _callable(triangulation, "get_toric_variety"):
        try:
            toric = triangulation.get_toric_variety()
        except Exception:
            toric = None
    if toric is not None:
        call("toric_fan_cones", toric, "fan_cones")
        call("toric_smoothness", toric, "is_smooth")
        call("toric_glsm", toric, "glsm_charge_matrix", include_origin=False)
        call("toric_sr_ideal", toric, "sr_ideal")
        call("toric_intersections", toric, "intersection_numbers", exact_arithmetic=True)
    else:
        evidence["toric_variety_dependent_apis"] = {
            "status": "unavailable",
            "reason": "get_toric_variety() did not return an object",
        }
    if cy is not None:
        call("cy_glsm", cy, "glsm_charge_matrix", include_origin=False)
        call("cy_intersections", cy, "intersection_numbers", exact_arithmetic=True)
        call("cy_divisor_basis", cy, "divisor_basis", as_matrix=True)
        call("cy_mori_cone", cy, "toric_mori_cone", in_basis=True)
        call("cy_kahler_cone", cy, "toric_kahler_cone")
        call("cy_smoothness", cy, "is_smooth")
    return evidence


def height_one_point_evidence(poly: Any) -> dict[str, Any]:
    """Read the boundary lattice points that lie on height-one facets."""

    if not _callable(poly, "points") or not _callable(poly, "facets"):
        return {
            "status": "height_one_points_unavailable",
            "reason": "public Polytope.points() and facets() are both required",
            "fallback": None,
            "fallback_reason": "Batyrev boundary-ray data is not inferred without the public facet API",
        }
    try:
        points = _integer_rows(poly.points(), name="polytope lattice points", dimension=4)
        facets = tuple(poly.facets())
        facet_points = []
        for facet_index, facet in enumerate(facets):
            if not _callable(facet, "points"):
                return {
                    "status": "height_one_points_unavailable",
                    "reason": f"facet {facet_index} does not expose public points()",
                    "fallback": None,
                    "fallback_reason": "facet lattice points are not inferred",
                }
            facet_points.append(
                _integer_rows(
                    facet.points(),
                    name=f"facet {facet_index} lattice points",
                    dimension=4,
                )
            )
    except Exception as exc:
        return {
            "status": "height_one_points_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "fallback": None,
            "fallback_reason": "Batyrev boundary-ray data is unavailable from CYTools",
        }
    point_set = {tuple(int(value) for value in row) for row in points.tolist()}
    origin = (0, 0, 0, 0)
    boundary_set = {
        tuple(int(value) for value in row)
        for rows in facet_points
        for row in rows.tolist()
    }
    expected_boundary = point_set - {origin}
    if origin in boundary_set or boundary_set != expected_boundary:
        return {
            "status": "height_one_points_inconsistent",
            "point_count": len(point_set),
            "facet_count": len(facet_points),
            "height_one_point_count": len(boundary_set),
            "expected_boundary_point_count": len(expected_boundary),
            "reason": "facet point union does not equal all non-origin lattice points",
            "fallback": None,
            "fallback_reason": "the reflexive height-one boundary set is not certified",
        }
    return {
        "status": "height_one_points_read",
        "point_count": len(point_set),
        "facet_count": len(facet_points),
        "height_one_point_count": len(boundary_set),
        "height_one_points": [list(row) for row in sorted(boundary_set)],
        "origin_excluded": origin not in boundary_set,
        "facet_point_counts": [len(rows) for rows in facet_points],
        "source_anchor": "Batyrev alg-geom/9310003v1 Def. 4.1.5 and Theorem 4.2.2",
        "fallback": None,
        "fallback_reason": None,
    }


def omitted_point_facet_evidence(poly: Any, local_points: Any) -> dict[str, Any]:
    """Certify points omitted from a boundary-only triangulation."""

    try:
        global_points = _points_from_object(poly, name="global polytope points")
        local = _integer_rows(local_points, name="triangulation-local points", dimension=4)
        facets = tuple(poly.facets())
    except Exception as exc:
        return {
            "status": "omitted_point_certificate_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "terminal": True,
        }
    local_set = {tuple(int(value) for value in row) for row in local.tolist()}
    omitted = [
        tuple(int(value) for value in row)
        for row in global_points.tolist()
        if tuple(int(value) for value in row) not in local_set
    ]
    certificates = []
    for point in omitted:
        containing_facets = []
        for facet_index, facet in enumerate(facets):
            try:
                facet_points = _integer_rows(
                    facet.points(), name=f"facet {facet_index} points", dimension=4
                )
                facet_vertices = _integer_rows(
                    facet.vertices(), name=f"facet {facet_index} vertices", dimension=4
                )
            except Exception as exc:
                return {
                    "status": "omitted_point_certificate_unavailable",
                    "reason": f"facet {facet_index} API failed: {type(exc).__name__}: {exc}",
                    "terminal": True,
                }
            if point in {tuple(int(value) for value in row) for row in facet_points.tolist()}:
                is_vertex = point in {
                    tuple(int(value) for value in row) for row in facet_vertices.tolist()
                }
                containing_facets.append({
                    "facet_index": int(facet_index),
                    "facet_point_count": int(len(facet_points)),
                    "facet_vertex_count": int(len(facet_vertices)),
                    "facet_interior": not is_vertex,
                })
        certificates.append({
            "point": list(point),
            "facet_memberships": containing_facets,
            "facet_interior": any(item["facet_interior"] for item in containing_facets),
        })
    status = "omitted_facet_interior_points_certified"
    if not omitted or any(not item["facet_interior"] for item in certificates):
        status = "omitted_point_certificate_mismatch"
    return {
        "status": status,
        "global_point_count": int(len(global_points)),
        "triangulation_point_count": int(len(local)),
        "omitted_point_count": int(len(omitted)),
        "omitted_points": [list(point) for point in omitted],
        "certificates": certificates,
        "include_points_interior_to_facets": False,
        "terminal": status != "omitted_facet_interior_points_certified",
        "source_anchor": "Batyrev anticanonical MPCP restriction; derived facet-interior omission certificate",
    }


def dual_action_evidence(poly: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    """Check the exact dual-lattice parity witness for the inherited action."""

    if not _callable(poly, "dual"):
        return {
            "status": "dual_check_unavailable",
            "reason": "Polytope.dual() is absent",
            "terminal": True,
        }
    try:
        dual = poly.dual()
        vertices_method = getattr(dual, "vertices", None)
        if not callable(vertices_method):
            raise ValueError("dual polytope does not expose vertices()")
        dual_vertices = _integer_rows(vertices_method(), name="dual vertices", dimension=4)
        matrix = _integer_rows(action["lattice_matrix"], name="lattice action", dimension=4)
        shift = _decode_vector(action["torus_shift"])
        lambda_f = int(action["lambda_f"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "dual_check_unavailable", "reason": str(exc), "terminal": True}
    fixed_vertices = []
    parity_failures = []
    for vertex in dual_vertices.tolist():
        q = np.asarray(vertex, dtype=np.int64)
        image = matrix.T @ q
        if np.array_equal(image, q):
            pairing = sum(2 * shift[index] * int(q[index]) for index in range(4))
            parity = pairing + lambda_f
            if parity.denominator != 1 or int(parity) % 2:
                parity_failures.append({"vertex": vertex, "parity": _jsonable(parity)})
            fixed_vertices.append({"vertex": vertex, "two_t_pairing": _jsonable(pairing)})
    return {
        "status": "dual_check_verified" if not parity_failures else "dual_check_mismatch",
        "dual_vertex_count": int(len(dual_vertices)),
        "fixed_dual_vertex_count": int(len(fixed_vertices)),
        "fixed_dual_vertices": fixed_vertices,
        "parity_failures": parity_failures,
        "equation": "2*<t,q> + lambda_f = 0 mod 2",
        "terminal": bool(parity_failures),
    }


def _decode_vector(value: Any) -> tuple[Fraction, ...]:
    """Decode a JSON-friendly exact rational vector."""

    if isinstance(value, Mapping) and "numerator" in value:
        denominator = int(value["denominator"])
        if denominator <= 0:
            raise ValueError("rational vector denominator must be positive")
        return tuple(Fraction(int(item), denominator) for item in value["numerator"])
    return tuple(_fraction_value(item) for item in value)


def lower_subdivision_evidence(
    poly: Any,
    height_evidence: Mapping[str, Any],
    *,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Retain the CYTools lower subdivision without tie-breaking cells."""

    if height_evidence.get("status") != "symmetric_heights_ready":
        return {
            "status": "lower_subdivision_unavailable",
            "reason": "an exact symmetric height witness is unavailable",
            "fallback": "no numerical heights are invented",
            "fallback_reason": "no numerical heights are invented",
        }
    vc_method = getattr(poly, "vc", None)
    if not callable(vc_method):
        return {
            "status": "lower_subdivision_unavailable",
            "reason": "Polytope.vc() is absent",
            "fallback": "no tie-broken triangulation is substituted",
            "fallback_reason": "no tie-broken triangulation is substituted",
        }
    try:
        vector_configuration = vc_method()
        subdivide = getattr(vector_configuration, "subdivide", None)
        if not callable(subdivide):
            raise AttributeError("vector configuration does not expose subdivide()")
        heights = [
            Fraction(int(value["numerator"]), int(value["denominator"]))
            if isinstance(value, Mapping)
            else _fraction_value(value)
            for value in height_evidence["heights"]
        ]
        # PPL retains cells on a degenerate lower hull. Feature-detect optional
        # keyword arguments so an older backend fails closed with provenance.
        kwargs = {
            "heights": heights,
            "backend": "ppl",
            "make_fine": False,
            "check_heights": False,
            "cure_heights": True,
        }
        try:
            signature = inspect.signature(subdivide)
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        except (TypeError, ValueError):
            pass
        subdivision = subdivide(**kwargs)
        cells = None
        for name in ("simplices", "cells", "maximal_cells"):
            method = getattr(subdivision, name, None)
            if callable(method):
                try:
                    raw_cells = method()
                    try:
                        cells = np.asarray(raw_cells, dtype=np.int64).tolist()
                    except (TypeError, ValueError):
                        # A genuine lower subdivision may contain cells with
                        # different numbers of vertices; keep that ragged
                        # cell complex instead of silently dropping it.
                        cells = [
                            np.asarray(cell, dtype=np.int64).reshape(-1).tolist()
                            for cell in raw_cells
                        ]
                    break
                except Exception:
                    continue
        if cells is None:
            cells = []
        cell_count = len(cells)
        cells_truncated = False
        if max_cells is not None and max_cells < 0:
            raise ValueError("max_cells must be nonnegative or None")
        if max_cells is not None and cell_count > max_cells:
            cells = cells[:max_cells]
            cells_truncated = True
        return {
            "status": "resource_capped_cells" if cells_truncated else "lower_subdivision_retained",
            "backend": "ppl" if "backend" in kwargs else "cytools_default",
            "kwargs": {key: _jsonable(value) for key, value in kwargs.items() if key != "heights"},
            "cell_count": cell_count,
            "cell_cap": max_cells,
            "cells": cells,
            "cells_truncated": cells_truncated,
            "simplicial": all(len(cell) <= 4 for cell in cells) if cells else None,
            "tie_breaking": False,
            "fallback": None,
            "fallback_reason": (
                "cell list truncated at declared max_refinement_cells cap"
                if cells_truncated else None
            ),
        }
    except Exception as exc:
        return {
            "status": "lower_subdivision_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "fallback": "CYTools lower-subdivision backend failed; non-simplicial cells are not inferred",
            "fallback_reason": "CYTools lower-subdivision backend failed; non-simplicial cells are not inferred",
        }


def hodge_split_from_euler(*, h11: int, h21: int, h11_minus: int, chi_fixed: int, chi_x: int) -> dict[str, Any]:
    """Apply the resolved Moritz Eq. (4.51) split with exact integer checks."""

    values = (h11, h21, h11_minus, chi_fixed, chi_x)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("resolved Hodge/Euler inputs must be integers")
    delta = int(chi_fixed) - int(chi_x)
    if delta % 4:
        raise ValueError("resolved fixed-locus Euler difference is not divisible by four")
    h21_minus = int(h11_minus) + delta // 4 - 1
    split = {
        "h11_plus": int(h11) - int(h11_minus),
        "h11_minus": int(h11_minus),
        "h21_plus": int(h21) - h21_minus,
        "h21_minus": h21_minus,
        "chi_fixed_locus": int(chi_fixed),
        "chi_x": int(chi_x),
    }
    eigenspace_values = [split[name] for name in ("h11_plus", "h11_minus", "h21_plus", "h21_minus")]
    if min(eigenspace_values) < 0:
        raise ValueError("resolved Eq. (4.51) produced a negative eigenspace")
    return split


def _all_triangulations(poly: Any, *, cap: int, deadline: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate every CYTools candidate until the declared resource cap."""

    method = getattr(poly, "all_triangulations", None)
    if not callable(method):
        return [], {
            "status": "all_triangulations_unavailable",
            "reason": "Polytope.all_triangulations is absent; no single triangulate fallback is permitted",
            "fallback": None,
        }
    kwargs = {
        "only_fine": True,
        "only_regular": True,
        "only_star": True,
        "include_points_interior_to_facets": False,
    }
    try:
        signature = inspect.signature(method)
        if "as_list" in signature.parameters:
            kwargs["as_list"] = False
        iterator = method(**kwargs)
    except Exception as exc:
        return [], {
            "status": "all_triangulations_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "kwargs": kwargs,
            "fallback": None,
        }
    records: list[dict[str, Any]] = []
    capped = False
    try:
        for candidate_index, triangulation in enumerate(iterator):
            if time.monotonic() > deadline:
                capped = True
                break
            if candidate_index >= cap:
                capped = True
                break
            try:
                identity = triangulation_identity(triangulation)
            except (AttributeError, TypeError, ValueError) as exc:
                records.append({
                    "candidate_index": candidate_index,
                    "terminal_status": "triangulation_identity_unavailable",
                    "reason": str(exc),
                })
                continue
            point_count = None
            point_count_reason = None
            try:
                point_count = int(
                    _points_from_object(
                        triangulation, name="candidate triangulation points"
                    ).shape[0]
                )
            except (AttributeError, TypeError, ValueError) as exc:
                point_count_reason = str(exc)
            records.append({
                "candidate_index": candidate_index,
                "triangulation": triangulation,
                "triangulation_identity": identity,
                "flags": _triangulation_flags(triangulation),
                "point_count": point_count,
                "point_count_reason": point_count_reason,
            })
    except Exception as exc:
        return records, {
            "status": "all_triangulations_iteration_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "kwargs": kwargs,
            "yielded": len(records),
            "capped": capped,
            "fallback": None,
        }
    return records, {
        "status": "resource_capped" if capped else "all_triangulations_exhausted",
        "kwargs": kwargs,
        "yielded": len(records),
        "cap": cap,
        "capped": capped,
        "terminal_accounting": "all yielded candidates retained; cap is explicit",
        "fallback": None,
    }


def _construct_polytope(
    record: Mapping[str, Any],
    source_evidence: Mapping[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Construct one frozen CYTools Polytope from the source global coordinates."""

    source_evidence = source_evidence or _source_identity_evidence(
        int(record.get("index", -1)), record
    )
    if source_evidence.get("terminal"):
        return None, dict(source_evidence)
    points = source_evidence.get("global_points")
    if points is None:
        return None, {
            "status": "source_global_coordinates_unavailable",
            "reason": "source identity did not supply global coordinates",
            "terminal": True,
        }
    if record.get("polytope") is not None:
        try:
            supplied_points = _points_from_object(record["polytope"], name="supplied global polytope points")
            expected_points = _integer_rows(points, name="source global points", dimension=4)
            if point_identity(supplied_points) != point_identity(expected_points):
                return None, {
                    "status": "source_global_coordinates_mismatch",
                    "reason": "supplied Polytope points disagree with source.global_points",
                    "terminal": True,
                }
        except (TypeError, ValueError) as exc:
            return None, {
                "status": "source_global_coordinates_invalid",
                "reason": str(exc),
                "terminal": True,
            }
        return record["polytope"], {
            "status": "frozen_global_object_supplied",
            "polytope_id": source_evidence["polytope_id"],
            "source_row": source_evidence["source_row"],
            "global_point_count": source_evidence["global_point_count"],
        }
    try:
        module = importlib.import_module("cytools")
        version_guard = _cytools_version_guard(module)
        if version_guard["status"] != "verified":
            return None, {
                "status": "cytools_version_unsupported",
                "reason": version_guard["reason"],
                "version_guard": version_guard,
                "fallback": None,
                "fallback_reason": "no geometry is constructed against an unaudited CYTools API",
            }
        # Construct exactly once from the immutable source configuration.  In
        # particular, never construct Polytope(triangulation.points()), since
        # CYTools intentionally omits facet-interior points from a boundary FRST.
        poly = module.Polytope(_integer_rows(points, name="source global points", dimension=4))
    except Exception as exc:
        return None, {
            "status": "polytope_input_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "fallback": "CYTools Polytope construction is built-in-first; no geometric outcome is inferred",
            "fallback_reason": "explicit replay coordinates could not be constructed by CYTools; no geometric outcome is inferred",
        }
    return poly, {
        "status": "constructed_from_frozen_global_coordinates",
        "point_identity": point_identity(points),
        "polytope_id": source_evidence["polytope_id"],
        "source_row": source_evidence["source_row"],
        "global_point_count": source_evidence["global_point_count"],
    }


def _construct_selected_triangulation(
    poly: Any,
    record: Mapping[str, Any],
    *,
    selected: Mapping[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    """Reconstruct a selected FRST on the frozen global Polytope."""

    selected = selected if selected is not None else record.get("selected_frst")
    if record.get("selected_triangulation") is not None:
        if not isinstance(selected, Mapping):
            return None, {
                "status": "simplices_index_space_mismatch",
                "reason": "object-supplied selected FRST still requires an explicit selected_frst index-space record",
                "terminal": True,
            }
        index_space = selected.get("simplices_index_space")
        if index_space not in {"triangulation_local", "polytope_global"}:
            return None, {
                "status": "simplices_index_space_mismatch",
                "reason": "object-supplied selected FRST must declare simplices_index_space",
                "observed": index_space,
                "terminal": True,
            }
        try:
            actual_points = _points_from_object(
                record["selected_triangulation"], name="object-supplied selected FRST points"
            )
            global_points = _points_from_object(poly, name="frozen global polytope points")
            if index_space == "triangulation_local":
                expected_points = _integer_rows(
                    selected.get("points"), name="selected FRST local points", dimension=4
                )
                mapping = _coordinate_index_map(global_points, actual_points, name="object selected FRST points")[1]
                if point_identity(actual_points) != point_identity(expected_points):
                    raise ValueError("object selected FRST points disagree with selected_frst.points")
            else:
                mapping = _coordinate_index_map(global_points, actual_points, name="object selected FRST points")[1]
        except (TypeError, ValueError) as exc:
            return None, {
                "status": "simplices_index_space_mismatch",
                "reason": str(exc),
                "index_space": index_space,
                "terminal": True,
            }
        return record["selected_triangulation"], {
            "status": "frozen_global_object_supplied",
            "index_space": index_space,
            "mapping": mapping,
        }
    if selected is None:
        return None, {"status": "input_missing", "reason": "selected FRST is required and is never chosen implicitly"}
    if not isinstance(selected, Mapping) or selected.get("simplices") is None:
        return None, {"status": "selected_frst_unavailable", "reason": "selected_frst.simplices is required"}
    index_space = selected.get("simplices_index_space")
    if index_space not in {"triangulation_local", "polytope_global"}:
        return None, {
            "status": "simplices_index_space_mismatch",
            "reason": "selected FRST must declare simplices_index_space as triangulation_local or polytope_global",
            "observed": index_space,
            "terminal": True,
        }
    if not _callable(poly, "triangulate"):
        return None, {"status": "selected_frst_unavailable", "reason": "Polytope.triangulate is absent"}
    try:
        simplices = np.asarray(selected["simplices"], dtype=np.int64)
        if simplices.ndim != 2:
            raise ValueError("selected FRST simplices must be a two-dimensional integer array")
        global_points = _points_from_object(poly, name="frozen global polytope points")
        if index_space == "triangulation_local":
            local_points = selected.get("points")
            if local_points is None:
                raise ValueError("triangulation_local simplices require selected_frst.points")
            local_to_global, mapping = _coordinate_index_map(
                global_points, local_points, name="selected FRST local points"
            )
            if np.any(simplices < 0) or np.any(simplices >= len(local_to_global)):
                raise ValueError("selected FRST local simplex index is outside selected_frst.points")
            global_simplices = local_to_global[simplices]
        else:
            local_to_global = np.arange(len(global_points), dtype=np.int64)
            mapping = {
                "status": "global_indices_verified",
                "index_space": "polytope_global",
                "local_count": int(len(global_points)),
                "global_count": int(len(global_points)),
                "local_to_global": local_to_global.tolist(),
                "coordinates_exact": True,
            }
            global_simplices = simplices
            if np.any(global_simplices < 0) or np.any(global_simplices >= len(global_points)):
                raise ValueError("global simplex index is outside source global points")
    except (TypeError, ValueError) as exc:
        return None, {
            "status": "simplices_index_space_mismatch",
            "reason": str(exc),
            "index_space": index_space,
            "terminal": True,
        }
    kwargs = {
        "simplices": global_simplices,
        "check_input_simplices": True,
        "include_points_interior_to_facets": False,
    }
    try:
        signature = inspect.signature(poly.triangulate)
        if "include_points_interior_to_facets" not in signature.parameters:
            return None, {
                "status": "selected_frst_api_mismatch",
                "reason": "Polytope.triangulate does not expose the required explicit facet-interior flag",
                "index_space": index_space,
                "terminal": True,
            }
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        triangulation = poly.triangulate(**kwargs)
    except Exception as exc:
        return None, {
            "status": "selected_frst_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "kwargs": list(kwargs),
            "index_space": index_space,
            "terminal": True,
        }
    try:
        actual_points = _points_from_object(triangulation, name="reconstructed selected FRST points")
        expected_local_points = (
            _integer_rows(selected["points"], name="selected FRST local points", dimension=4)
            if index_space == "triangulation_local"
            else _integer_rows(global_points, name="global points", dimension=4)
        )
        if point_identity(actual_points) != point_identity(expected_local_points):
            return None, {
                "status": "selected_frst_point_count_mismatch",
                "reason": "reconstructed selected FRST coordinates differ from the manifest index space",
                "expected_point_count": int(len(expected_local_points)),
                "actual_point_count": int(len(actual_points)),
                "terminal": True,
            }
    except (TypeError, ValueError) as exc:
        return None, {"status": "selected_frst_point_count_mismatch", "reason": str(exc), "terminal": True}
    return triangulation, {
        "status": "constructed_on_frozen_global_polytope",
        "index_space": index_space,
        "mapping": mapping,
        "include_points_interior_to_facets": False,
    }


def _normalise_triangulation_identity(value: Any) -> str:
    """Normalize common persisted FRST identity prefixes for exact comparison."""

    text = str(value)
    for prefix in ("frst-sha256:", "triangulation-sha256:"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _selected_identity_evidence(record: Mapping[str, Any], actual: str) -> dict[str, Any]:
    """Verify a supplied selected-FRST identity when the manifest carries one."""

    selected = record.get("selected_frst")
    if not isinstance(selected, Mapping):
        return {"status": "not_supplied", "actual": actual}
    expected = next(
        (
            selected.get(name)
            for name in ("identity", "triangulation_identity", "frst_hash", "triangulation_hash")
            if selected.get(name) is not None
        ),
        None,
    )
    if expected is None:
        return {"status": "not_supplied", "actual": actual}
    expected_normalized = _normalise_triangulation_identity(expected)
    status = "matched" if expected_normalized == actual else "mismatch"
    return {
        "status": status,
        "expected": str(expected),
        "expected_normalized": expected_normalized,
        "actual": actual,
        "reason": None if status == "matched" else "selected FRST identity changed during reconstruction",
    }


def _exact_public_intersection_tensor(triangulation: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read the public fan intersection tensor with an exactness audit.

    CYTools exposes the requested exact-arithmetic keyword, but the audited
    1.4.12 installation rejects it unless experimental features are enabled.
    The bounded fixture therefore falls back to the public dense tensor only
    after checking that every returned value has an unambiguous small rational
    reconstruction.  This is an API/provenance fallback, not a replacement
    for missing smooth Cartier data.
    """

    fan_method = getattr(triangulation, "fan", None)
    if not callable(fan_method):
        return None, {
            "status": "unavailable",
            "method": "Triangulation.fan()",
            "reason": "public fan() is absent",
            "fallback": None,
        }
    try:
        fan = fan_method()
    except Exception as exc:
        return None, {
            "status": "unavailable",
            "method": "Triangulation.fan()",
            "reason": f"{type(exc).__name__}: {exc}",
            "fallback": None,
        }
    intersection_method = getattr(fan, "intersection_numbers", None)
    if not callable(intersection_method):
        return None, {
            "status": "unavailable",
            "method": "fan.intersection_numbers()",
            "reason": "public intersection_numbers() is absent",
            "fallback": None,
        }
    exact_method = "fan.intersection_numbers(exact_arithmetic=True, as_np_array=True)"
    try:
        raw = intersection_method(exact_arithmetic=True, as_np_array=True)
        array = np.asarray(raw, dtype=object)
        if array.ndim != 4 or len(set(array.shape)) != 1:
            raise ValueError(f"unexpected exact tensor shape {array.shape}")
        tensor = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            tensor[index] = Fraction(array[index])
        return tensor, {
            "status": "cytools_exact",
            "method": exact_method,
            "shape": list(array.shape),
            "fallback": None,
            "exactness": "public exact-arithmetic result",
        }
    except Exception as exact_exc:
        fallback_reason = f"{type(exact_exc).__name__}: {exact_exc}"
    try:
        raw = intersection_method(as_np_array=True)
        array = np.asarray(raw)
        if array.ndim != 4 or len(set(array.shape)) != 1:
            raise ValueError(f"unexpected public tensor shape {array.shape}")
        tensor = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            value = array[index]
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"non-finite public intersection at {index}: {value!r}")
            rational = Fraction(str(numeric)).limit_denominator(1_000_000)
            if abs(float(rational) - numeric) > 1e-10:
                raise ValueError(
                    f"public intersection at {index} is not exactly reconstructible: {value!r}"
                )
            tensor[index] = rational
        return tensor, {
            "status": "public_float_rationalized",
            "method": "fan.intersection_numbers(as_np_array=True)",
            "shape": list(array.shape),
            "fallback": "exact project formula with rationalized public entries",
            "fallback_reason": fallback_reason,
            "exactness": "finite-fixture rational reconstruction checked at 1e-10",
        }
    except Exception as fallback_exc:
        return None, {
            "status": "unavailable",
            "method": "fan.intersection_numbers(as_np_array=True)",
            "reason": f"{type(fallback_exc).__name__}: {fallback_exc}",
            "fallback": "none",
            "fallback_reason": fallback_reason,
        }


def _identity_conormal_n_s_terms(
    tensor: np.ndarray, p_index: int, q_index: int, n_rays: int
) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
    """Evaluate the identity-case conormal expansion over exact entries."""

    l2 = sum(
        (Fraction(tensor[p_index, q_index, r, s])
         for r in range(1, n_rays + 1)
         for s in range(1, n_rays + 1)),
        Fraction(0),
    )
    l_dp = sum(
        (Fraction(tensor[p_index, q_index, r, p_index])
         for r in range(1, n_rays + 1)),
        Fraction(0),
    )
    l_dq = sum(
        (Fraction(tensor[p_index, q_index, r, q_index])
         for r in range(1, n_rays + 1)),
        Fraction(0),
    )
    dpdq = Fraction(tensor[p_index, q_index, p_index, q_index])
    return l2, l_dp, l_dq, dpdq, l2 - l_dp - l_dq + dpdq


def _identity_surface_n_s_diagnostics(
    triangulation: Any,
    triangulation_cones: Sequence[Sequence[Sequence[int]]],
    components: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain exact class-fixture evidence for identity contained surfaces.

    The direct divisor contraction is the identity reduction of Moritz
    eq. (4.50), with the conormal sign.  The result is reported even when the
    ambient star is not smooth; in that case it is a formal/orbifold Chow
    contraction and is not promoted to an ordinary smooth Eq. (4.50)
    certificate.
    """

    general_l = importlib.import_module("orientifold_general_l_geometry")
    surfaces = [
        component
        for component in components
        if bool(component.get("f_vanishes_identically"))
        and int(component.get("fixed_toric_dimension", -1)) == 2
    ]
    if not surfaces:
        return []
    tensor, intersection_api = _exact_public_intersection_tensor(triangulation)
    if tensor is None:
        return [
            {
                "status": "unavailable",
                "reason_code": "intersection_api_unavailable",
                "reason": intersection_api.get("reason"),
                "intersection_api": intersection_api,
            }
        ]
    try:
        fan = triangulation.fan()
        vectors = np.asarray(fan.vectors(), dtype=int)
    except Exception as exc:
        return [
            {
                "status": "unavailable",
                "reason_code": "fan_vector_api_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
                "intersection_api": intersection_api,
            }
        ]
    if tensor.shape[0] != vectors.shape[0] + 1:
        return [
            {
                "status": "unavailable",
                "reason_code": "intersection_tensor_ray_order_mismatch",
                "reason": (
                    f"tensor has {tensor.shape[0] - 1} ray slots but fan has "
                    f"{vectors.shape[0]} vectors"
                ),
                "intersection_api": intersection_api,
            }
        ]

    def fraction_json(value: Fraction) -> Any:
        return _jsonable(Fraction(value))

    def primitive_quotient_ray(annihilator: np.ndarray, ray: Sequence[int]) -> tuple[int, ...]:
        quotient = np.asarray(annihilator @ np.asarray(ray, dtype=int), dtype=int)
        divisor = math.gcd(*[abs(int(value)) for value in quotient.tolist()])
        if divisor == 0:
            raise ValueError("quotient ray vanishes")
        return tuple(int(value // divisor) for value in quotient)

    diagnostics: list[dict[str, Any]] = []
    for component in surfaces:
        sigma_rays = tuple(tuple(int(value) for value in ray) for ray in component["sigma_rays"])
        diagnostic: dict[str, Any] = {
            "status": "formal_only",
            "sigma_rays": [list(ray) for ray in sigma_rays],
            "nu": component.get("nu"),
            "component_containment": "f|S identically zero",
            "formula": (
                "n_S = integral_S c2(O(K_V^-1)|S tensor N*S) = "
                "D_p D_q (K_V^-1-D_p)(K_V^-1-D_q)"
            ),
            "conormal_sign": "N*S (conormal), hence +D_p D_q in the expanded identity formula",
            "intersection_api": intersection_api,
        }
        matches = []
        for ray in sigma_rays:
            found = np.flatnonzero(np.all(vectors == np.asarray(ray, dtype=int), axis=1))
            if found.size != 1:
                matches = []
                break
            matches.append(int(found[0]) + 1)
        if len(matches) != 2:
            diagnostic.update({
                "status": "unavailable",
                "reason_code": "sigma_ray_order_unavailable",
                "reason": "original-Sigma surface rays do not match a unique fan-vector slot",
            })
            diagnostics.append(diagnostic)
            continue
        p_index, q_index = matches
        n_rays = vectors.shape[0]
        l2, l_dp, l_dq, dpdq, n_s = _identity_conormal_n_s_terms(
            tensor, p_index, q_index, n_rays
        )
        diagnostic.update({
            "fan_vector_indices_one_based": [p_index, q_index],
            "chow_intersection_terms": {
                "integral_Dp_Dq_Kinv_squared": fraction_json(l2),
                "integral_Dp_Dq_Kinv_Dp": fraction_json(l_dp),
                "integral_Dp_Dq_Kinv_Dq": fraction_json(l_dq),
                "integral_Dp_Dq_Dp_Dq": fraction_json(dpdq),
            },
            "formal_n_s": fraction_json(n_s),
        })

        containing_cones = [
            tuple(tuple(int(value) for value in ray) for ray in cone)
            for cone in triangulation_cones
            if all(ray in cone for ray in sigma_rays)
        ]
        sigma_matrix = np.column_stack(np.asarray(sigma_rays, dtype=int))
        saturated = bool(general_l._sublattice_is_saturated(sigma_matrix))
        annihilator = general_l._integer_kernel_basis(sigma_matrix.T).T
        quotient_cones: set[tuple[tuple[int, ...], ...]] = set()
        for cone in containing_cones:
            quotient_rays = tuple(
                primitive_quotient_ray(annihilator, ray)
                for ray in cone
                if ray not in sigma_rays
            )
            if len(quotient_rays) == 2:
                quotient_cones.add(tuple(sorted(quotient_rays)))
        quotient_rays = sorted({ray for cone in quotient_cones for ray in cone})
        quotient_smooth = all(
            abs(general_l._exact_determinant(np.column_stack(cone))) == 1
            for cone in quotient_cones
        )
        quotient_complete = bool(
            quotient_cones
            and general_l._complete_simplicial_fan(sorted(quotient_cones), 2)
        )
        quotient_c1_squared = None
        if quotient_smooth and quotient_complete:
            try:
                unit = {ray: Fraction(1) for ray in quotient_rays}
                quotient_c1_squared = general_l._surface_divisor_intersection(
                    unit, unit, quotient_rays, sorted(quotient_cones)
                )
            except (TypeError, ValueError, ZeroDivisionError):
                quotient_c1_squared = None
        ambient_determinants = [
            int(general_l._exact_determinant(np.asarray(cone, dtype=int)))
            for cone in containing_cones
        ]
        local_cartier = []
        for cone in containing_cones:
            divisor_data = {
                str([int(value) for value in ray]): general_l._ambient_cartier_data(cone, ray) is not None
                for ray in vectors
            }
            local_cartier.append({
                "ambient_cone": [list(ray) for ray in cone],
                "determinant": int(general_l._exact_determinant(np.asarray(cone, dtype=int))),
                "all_integral_divisor_cartier": all(divisor_data.values()),
                "divisor_cartier_status": divisor_data,
                "anticanonical_cartier_data": _jsonable(
                    general_l._ambient_anticanonical_cartier_data(cone)
                ),
            })
        toric_smooth = None
        toric_api_status = "api_absent"
        try:
            toric = triangulation.get_toric_variety()
            if toric is not None and callable(getattr(toric, "is_smooth", None)):
                toric_smooth = bool(toric.is_smooth())
                toric_api_status = "cytools_builtin"
        except Exception as exc:
            toric_api_status = f"api_failed: {type(exc).__name__}: {exc}"
        diagnostic.update({
            "original_sigma_containing_ambient_cones": [
                [list(ray) for ray in cone] for cone in containing_cones
            ],
            "ambient_containing_cone_determinants": ambient_determinants,
            "ambient_toric_smoothness": {
                "status": toric_api_status,
                "is_smooth": toric_smooth,
                "method": "triangulation.get_toric_variety().is_smooth()",
            },
            "orbifold_issue": bool(
                not saturated
                or any(abs(value) != 1 for value in ambient_determinants)
                or toric_smooth is False
            ),
            "fixed_cone_lattice": {
                "saturated": saturated,
                "index": int(math.gcd(*[
                    abs(int(general_l._exact_determinant(sigma_matrix[np.ix_(rows, range(2))])))
                    for rows in itertools.combinations(range(4), 2)
                ])),
                "quotient_annihilator": np.asarray(annihilator, dtype=int).tolist(),
            },
            "quotient_star_fan": {
                "rays": [list(ray) for ray in quotient_rays],
                "maximal_cones": [[list(ray) for ray in cone] for cone in sorted(quotient_cones)],
                "complete": quotient_complete,
                "smooth_unimodular": quotient_smooth,
                "coarse_toric_orbit_euler": len(quotient_cones) if quotient_complete else None,
                "c1_squared_of_sum_boundary_divisors": (
                    None if quotient_c1_squared is None else fraction_json(quotient_c1_squared)
                ),
            },
            "restricted_cartier_data": local_cartier,
            "certificate_status": (
                "unavailable_non_smooth_ambient_or_nonsaturated_sigma"
                if not saturated or toric_smooth is False or not all(abs(value) == 1 for value in ambient_determinants)
                else "not_checked"
            ),
            "reason_code": (
                "nonsaturated_fixed_cone_lattice"
                if not saturated
                else "non_smooth_ambient_cone"
                if toric_smooth is False or not all(abs(value) == 1 for value in ambient_determinants)
                else None
            ),
        })
        diagnostics.append(diagnostic)
    return diagnostics


def _fixed_locus_fallback(poly: Any, triangulation: Any, action: Mapping[str, Any], topology: Any = None) -> dict[str, Any]:
    """Run the project exact fixed-locus kernel with explicit fallback provenance."""

    try:
        general_l = importlib.import_module("orientifold_general_l_geometry")
        ioc = importlib.import_module("inherited_orientifold_candidates")
        euler = importlib.import_module("toric_fixed_component_euler")
    except Exception as exc:
        return {
            "status": "fixed_locus_euler_unavailable",
            "reason": f"project exact fixed-locus kernel unavailable: {type(exc).__name__}: {exc}",
            "fallback": "project exact fallback not importable",
            "fallback_reason": "CYTools has no public orientifold fixed-locus Euler API; project exact modules were not importable",
        }
    try:
        matrix = _integer_rows(action["lattice_matrix"], name="lattice action", dimension=4)
        shift = _decode_vector(action["torus_shift"])
        lambda_f = int(action["lambda_f"])
        cones = ioc._triangulation_cones(poly, triangulation)
        auxiliary = general_l.build_auxiliary_fan(cones, matrix)
        fixed_keys = general_l._pointwise_invariant_cone_keys(cones, matrix)
        dual_polytope = poly.dual()
        dual_points_method = getattr(dual_polytope, "points", None)
        if not callable(dual_points_method):
            dual_points_method = getattr(dual_polytope, "vertices", None)
        dual_points = None if not callable(dual_points_method) else np.asarray(
            dual_points_method(), dtype=np.int64
        )
        ambient_rays = sorted({ray for cone in cones for ray in cone})
        components = general_l._fixed_component_records(
            auxiliary,
            matrix,
            shift,
            lambda_f,
            fixed_cone_keys=fixed_keys,
            dual_points=dual_points,
            ambient_rays=ambient_rays,
            fan_cones=cones,
        )
        fixed_surface_n_s_evidence = {}
        surface_diagnostics = []
        has_contained_surface = any(
            item.get("f_vanishes_identically") is True
            and int(item.get("fixed_toric_dimension", -1)) == 2
            for item in components
        )
        if np.array_equal(matrix, ioc.IDENTITY) and has_contained_surface:
            # Identity actions reduce eq. (4.50) to the direct original-Sigma
            # divisor contraction already audited in the inherited kernel. The
            # bounded exact public-API contraction is the value supplied to
            # the Euler gate; a formal nonzero result must remain unavailable
            # when its star is not smooth/unimodular.
            surface_diagnostics = _identity_surface_n_s_diagnostics(
                triangulation, cones, components
            )
            for diagnostic in surface_diagnostics:
                formal_n_s = diagnostic.get("formal_n_s")
                if formal_n_s is None or diagnostic.get("sigma_rays") is None:
                    continue
                value = Fraction(
                    int(formal_n_s["numerator"]),
                    int(formal_n_s["denominator"]),
                )
                if value.denominator == 1:
                    fixed_surface_n_s_evidence[
                        general_l._component_key({
                            "sigma_rays": diagnostic["sigma_rays"],
                            "nu": diagnostic["nu"],
                        })
                    ] = int(value)
        fixed = euler.exact_fixed_locus_euler(
            auxiliary,
            matrix,
            components,
            fixed_surface_n_s_evidence=fixed_surface_n_s_evidence,
        )
    except Exception as exc:
        return {
            "status": "fixed_locus_euler_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "fallback": "project exact Moritz component and ordinary-Euler fallback",
            "fallback_reason": "CYTools has no public orientifold fixed-locus Euler API; project Moritz components and ordinary Euler are used",
        }
    fixed = dict(fixed)
    fixed["fallback"] = "project exact Moritz component and ordinary-Euler fallback"
    fixed["fallback_reason"] = "CYTools has no public orientifold fixed-locus Euler API; project Moritz components and ordinary Euler are used"
    fixed["auxiliary_fan_distinct_from_selected_frst"] = True
    fixed["component_count"] = len(components)
    fixed["component_count_basis"] = "enumerated_source_components"
    fixed["fixed_surface_n_s_evidence"] = _jsonable(fixed_surface_n_s_evidence)
    if surface_diagnostics:
        fixed["fixed_surface_n_s_diagnostics"] = _jsonable(surface_diagnostics)
    if surface_diagnostics:
        certified_euler = sum(
            int(item["chi"])
            for item in fixed.get("components", [])
            if item.get("euler_status") == "computed" and item.get("chi") is not None
        )
        coarse_surface_euler = sum(
            int(item["quotient_star_fan"]["coarse_toric_orbit_euler"])
            for item in surface_diagnostics
            if item.get("quotient_star_fan", {}).get("coarse_toric_orbit_euler") is not None
        )
        chi_x = None
        try:
            cy = triangulation.get_cy()
            chi_x = int(cy.chi())
        except (AttributeError, TypeError, ValueError):
            pass
        coarse_chi_f = certified_euler + coarse_surface_euler
        fixed["chi_F_I_diagnostic"] = {
            "status": "not_certified",
            "certified_component_sum": certified_euler,
            "coarse_quotient_surface_sum": coarse_surface_euler,
            "coarse_euler_sum": coarse_chi_f,
            "chi_X": chi_x,
            "eq_4_51_status": "unavailable_fixed_locus_not_certified",
            "eq_4_51_delta_chi_coarse": (
                None if chi_x is None else coarse_chi_f - chi_x
            ),
            "eq_4_51_delta_divisible_by_four": (
                None if chi_x is None else (coarse_chi_f - chi_x) % 4 == 0
            ),
            "hodge_split": None,
        }
    fixed["certified_component_count"] = sum(
        item.get("euler_status") == "computed"
        for item in fixed.get("components", [])
    )
    return fixed


def evaluate_action_on_refinement(
    poly: Any,
    triangulation: Any,
    action: Mapping[str, Any],
    *,
    original_cy: Any = None,
    original_points: Any = None,
    expected_hodge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one action/refinement without selecting on its result."""

    result: dict[str, Any] = {
        "action": _jsonable(action),
        "terminal_status": None,
        "selected_frst_replaced": False,
        "auxiliary_quotient_fan": True,
    }
    try:
        matrix = _integer_rows(action["lattice_matrix"], name="lattice action", dimension=4)
        if not np.array_equal(matrix @ matrix, np.eye(4, dtype=np.int64)):
            raise ValueError("lattice action is not an involution")
        shift = _decode_vector(action["torus_shift"])
        if len(shift) != 4 or any((2 * value).denominator != 1 for value in shift):
            raise ValueError("2*t is not integral")
    except (KeyError, TypeError, ValueError) as exc:
        result["terminal_status"] = "action_not_involution"
        result["reason"] = str(exc)
        return result
    result["action_involution"] = {"status": "passed", "fallback": "exact rational action check"}
    result["fan_preservation"] = fan_action_evidence(poly, triangulation, matrix)
    result["symmetric_heights"] = symmetric_heights(poly, triangulation, matrix)
    cy = None
    if _callable(triangulation, "get_cy"):
        try:
            cy = triangulation.get_cy()
        except Exception as exc:
            result["cytools_invariants_reason"] = f"{type(exc).__name__}: {exc}"
    if cy is None:
        result["terminal_status"] = "resolved_cy_unavailable"
        result["reason"] = "refined Triangulation.get_cy() did not return a CalabiYau object"
        return result
    original_labels = None
    if original_cy is not None and _callable(original_cy, "prime_toric_divisors"):
        try:
            original_labels = np.asarray(original_cy.prime_toric_divisors(), dtype=np.int64).tolist()
        except Exception:
            original_labels = None
    # CYTools prime_toric_divisors() labels the refined triangulation point
    # order, which can differ from the full polytope point order.
    points = _points_from_object(triangulation, name="refined triangulation points")
    result["refined_glsm"] = refined_glsm_evidence(
        cy,
        matrix,
        points,
        original_prime_labels=original_labels,
        original_points=original_points,
    )
    result["resolved_hodge"] = resolved_hodge_evidence(cy)
    if expected_hodge is not None and result["resolved_hodge"].get("status") == "resolved_hodge_read":
        result["hodge_source_identity"] = _hodge_match_evidence(
            result["resolved_hodge"], expected_hodge, name="refined candidate"
        )
        if result["hodge_source_identity"]["terminal"]:
            result["terminal_status"] = "source_hodge_mismatch"
            result["reason"] = "refined CalabiYau Hodge/Euler data disagree with immutable source"
            return result
    result["refined_geometry_apis"] = _public_geometry_evidence(poly, triangulation, cy)
    result["fixed_locus_euler"] = _fixed_locus_fallback(poly, triangulation, action)
    fixed = result["fixed_locus_euler"]
    hodge = result["resolved_hodge"]
    h2 = result["refined_glsm"]
    if h2.get("status") != "refined_h2_action_verified":
        result["terminal_status"] = h2.get("status", "nonintegral_refined_h2_action")
        result["reason"] = h2.get("reason")
        return result
    if hodge.get("status") != "resolved_hodge_read":
        result["terminal_status"] = hodge.get("status", "resolved_hodge_unavailable")
        result["reason"] = hodge.get("reason")
        return result
    if fixed.get("status") != "computed":
        result["terminal_status"] = "fixed_locus_euler_unavailable"
        result["reason"] = fixed.get("reason", "ordinary fixed-locus Euler evidence is unavailable")
        return result
    try:
        h2_matrix = np.asarray(h2["h2_matrix"], dtype=np.int64)
        h11_minus_numerator = h2_matrix.shape[0] - int(np.trace(h2_matrix))
        if h11_minus_numerator % 2:
            raise ValueError("refined H2 involution has nonintegral minus eigenspace rank")
        h11_minus = h11_minus_numerator // 2
        result["hodge_split"] = hodge_split_from_euler(
            h11=hodge["h11"],
            h21=hodge["h21"],
            h11_minus=h11_minus,
            chi_fixed=int(fixed["chi_F_I"]),
            chi_x=hodge["chi"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        result["terminal_status"] = "resolved_hodge_split_unavailable"
        result["reason"] = str(exc)
        return result
    result["terminal_status"] = "refined_action_evaluated"
    return result


def analyze_replay_index(
    index: int,
    record: Mapping[str, Any],
    *,
    caps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one explicit replay index and retain all terminal accounting."""

    caps = {**DEFAULT_CAPS, **(caps or {})}
    index = int(index)
    report: dict[str, Any] = {
        "index": index,
        "schema_version": SCHEMA_VERSION,
        "scope_status": "in_scope" if index in TARGET_INDICES else "index_out_of_scope",
        "runtime_provenance": runtime_provenance(),
        "caps": _jsonable(caps),
        "terminal_records": [],
        "refinement_records": [],
        "action_records": [],
        "replay_certificates": [],
        "selected_frst": None,
        "table1_status": "not_comparable",
        "same_table1_status": "not_comparable",
    }
    if index not in TARGET_INDICES:
        report["terminal_records"].append({"terminal_status": "index_out_of_scope", "reason": f"only {TARGET_INDICES} are permitted"})
        return report
    start = time.monotonic()
    source_identity = _source_identity_evidence(index, record)
    report["source_identity"] = _jsonable(source_identity)
    if source_identity.get("terminal"):
        report["terminal_records"].append({
            "terminal_status": source_identity.get("status", "source_identity_unavailable"),
            "reason": source_identity.get("reason"),
        })
        return report
    poly, poly_status = _construct_polytope(record, source_identity)
    report["polytope_input"] = _jsonable(poly_status)
    if poly is None:
        terminal_status = poly_status.get("status", "polytope_input_unavailable")
        if terminal_status == "input_missing":
            terminal_status = "polytope_input_unavailable"
        report["terminal_records"].append({
            "terminal_status": terminal_status,
            "reason": poly_status.get("reason", "replay record has no polytope input"),
            "version_guard": poly_status.get("version_guard"),
        })
        return report
    report["api_features_polytope"] = feature_detection(poly)
    try:
        poly_hodge = _hodge_values(poly)
        report["polytope_hodge"] = _hodge_match_evidence(
            poly_hodge, source_identity["expected_hodge"], name="frozen global Polytope"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        report["polytope_hodge"] = {
            "status": "unavailable",
            "reason": str(exc),
            "terminal": True,
        }
    if report["polytope_hodge"].get("terminal"):
        report["terminal_records"].append({
            "terminal_status": "source_hodge_mismatch" if report["polytope_hodge"].get("status") == "mismatch" else "source_hodge_unavailable",
            "reason": "frozen global Polytope Hodge/Euler data do not match the immutable source",
            "hodge": report["polytope_hodge"],
        })
        return report
    report["height_one_points"] = height_one_point_evidence(poly)
    boundary_expected = source_identity["expected_boundary_point_count"]
    observed_boundary = report["height_one_points"].get("height_one_point_count")
    if observed_boundary != boundary_expected:
        report["terminal_records"].append({
            "terminal_status": "source_boundary_point_count_mismatch",
            "reason": "global Polytope boundary point count differs from immutable source",
            "expected": boundary_expected,
            "observed": observed_boundary,
        })
        return report
    selected, selected_status = _construct_selected_triangulation(poly, record)
    report["selected_frst_input"] = _jsonable(selected_status)
    if selected is None:
        report["terminal_records"].append({
            "terminal_status": selected_status.get("status", "selected_frst_unavailable"),
            "reason": selected_status.get("reason"),
        })
        return report
    selected_identity = triangulation_identity(selected)
    identity_evidence = _selected_identity_evidence(record, selected_identity)
    report["selected_frst"] = {
        "identity": selected_identity,
        "identity_evidence": identity_evidence,
        "flags": _triangulation_flags(selected),
        "source_index_space": selected_status.get("index_space"),
        "source_local_to_global": selected_status.get("mapping"),
        "preserved_as_source_geometry": True,
        "used_as_auxiliary_quotient_fan": False,
    }
    if identity_evidence["status"] == "mismatch":
        report["terminal_records"].append({
            "terminal_status": "selected_frst_identity_mismatch",
            "reason": identity_evidence["reason"],
            "identity_evidence": identity_evidence,
        })
        return report
    original_cy = None
    original_points = None
    if _callable(selected, "get_cy"):
        try:
            original_cy = selected.get_cy()
            report["selected_frst"]["resolved_hodge"] = resolved_hodge_evidence(original_cy)
            report["selected_frst"]["hodge_source_identity"] = _hodge_match_evidence(
                report["selected_frst"]["resolved_hodge"],
                source_identity["expected_hodge"],
                name="selected FRST",
            )
            if report["selected_frst"]["hodge_source_identity"]["terminal"]:
                report["terminal_records"].append({
                    "terminal_status": "source_hodge_mismatch",
                    "reason": "selected FRST Hodge/Euler data disagree with immutable source",
                    "hodge": report["selected_frst"]["hodge_source_identity"],
                })
                return report
        except Exception as exc:
            report["selected_frst"]["cytools_reason"] = f"{type(exc).__name__}: {exc}"
    try:
        original_points = _points_from_object(selected, name="selected FRST points")
    except (TypeError, ValueError) as exc:
        report["selected_frst"]["point_order_reason"] = f"{type(exc).__name__}: {exc}"
    if original_points is None:
        report["terminal_records"].append({
            "terminal_status": "selected_frst_point_count_mismatch",
            "reason": "selected FRST local coordinates are unavailable",
        })
        return report
    report["selected_frst"]["omitted_point_facet_certificate"] = omitted_point_facet_evidence(
        poly, original_points
    )
    omission = report["selected_frst"]["omitted_point_facet_certificate"]
    if omission.get("status") != "omitted_facet_interior_points_certified":
        report["terminal_records"].append({
            "terminal_status": "omitted_point_certificate_mismatch",
            "reason": omission.get("reason", "selected FRST omission certificate failed"),
            "certificate": omission,
        })
        return report
    if omission.get("triangulation_point_count") != boundary_expected:
        report["terminal_records"].append({
            "terminal_status": "selected_frst_point_count_mismatch",
            "reason": "selected FRST point count differs from immutable boundary-point count",
            "expected": boundary_expected,
            "observed": omission.get("triangulation_point_count"),
        })
        return report
    report["selected_geometry_apis"] = _public_geometry_evidence(poly, selected, original_cy)
    matrix_actions = record.get("actions")
    if matrix_actions is None and record.get("action") is not None:
        matrix_actions = [record["action"]]
    if matrix_actions is None:
        report["terminal_records"].append({
            "terminal_status": "action_input_missing",
            "reason": "explicit action records are required for bounded replay; no outcome-derived action is invented",
            "fallback": None,
        })
        return report
    try:
        actions = list(matrix_actions)
    except TypeError:
        actions = []
    if not actions:
        report["terminal_records"].append({"terminal_status": "action_input_missing", "reason": "action list is empty"})
        return report
    report["dual_action_checks"] = []
    for action_index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            report["dual_action_checks"].append({
                "action_index": action_index,
                "status": "dual_check_unavailable",
                "reason": "action record is not a mapping",
                "terminal": True,
            })
            continue
        dual_check = dual_action_evidence(poly, action)
        dual_check["action_index"] = action_index
        report["dual_action_checks"].append(dual_check)
        if dual_check.get("terminal"):
            report["terminal_records"].append({
                "terminal_status": dual_check.get("status", "dual_check_unavailable"),
                "reason": dual_check.get("reason", "dual-lattice parity check failed"),
                "action_index": action_index,
            })
    if any(check.get("terminal") for check in report["dual_action_checks"]):
        return report
    raw_refinements, enumeration = _all_triangulations(
        poly,
        cap=int(caps["max_triangulations"]),
        deadline=start + float(caps["max_seconds_per_index"]),
    )
    report["triangulation_enumeration"] = _jsonable({key: value for key, value in enumeration.items() if key != "triangulation"})
    candidate_point_counts = sorted({
        int(row["point_count"])
        for row in raw_refinements
        if row.get("point_count") is not None
    })
    boundary_count = report.get("height_one_points", {}).get("height_one_point_count")
    scope_differs = bool(candidate_point_counts) and boundary_count not in candidate_point_counts
    report["height_one_refinement_scope"] = {
        "global_polytope_point_count": source_identity["global_point_count"],
        "expected_global_polytope_point_count": source_identity["expected_point_count"],
        "height_one_point_count": boundary_count,
        "expected_height_one_point_count": source_identity["expected_boundary_point_count"],
        "candidate_point_counts": candidate_point_counts,
        "include_points_interior_to_facets": enumeration.get("kwargs", {}).get(
            "include_points_interior_to_facets"
        ),
        "status": (
            "candidate_scope_differs_from_boundary"
            if scope_differs
            else "candidate_scope_matches_boundary"
        ),
        "reason": (
            "CYTools enumeration point scope differs from the Batyrev boundary set; "
            "the facet-interior-ray convention requires scientific-owner direction"
            if scope_differs
            else None
        ),
    }
    if scope_differs:
        report["terminal_records"].append({
            "terminal_status": "mpcp_scope_unresolved",
            "reason": report["height_one_refinement_scope"]["reason"],
            "height_one_point_count": boundary_count,
            "candidate_point_counts": candidate_point_counts,
            "fallback": None,
        })
    if (
        report["height_one_refinement_scope"]["include_points_interior_to_facets"] is not False
        or any(count != boundary_count for count in candidate_point_counts)
    ):
        report["terminal_records"].append({
            "terminal_status": "refinement_point_count_mismatch",
            "reason": "boundary-only MPCP enumeration did not use the immutable facet-interior omission scope",
            "candidate_point_counts": candidate_point_counts,
            "expected_boundary_point_count": boundary_count,
            "include_points_interior_to_facets": report["height_one_refinement_scope"]["include_points_interior_to_facets"],
        })
        return report
    expected_frsts = record.get("selected_frsts")
    if expected_frsts is not None:
        if not isinstance(expected_frsts, Sequence) or isinstance(expected_frsts, (str, bytes)):
            report["terminal_records"].append({
                "terminal_status": "source_frst_identity_mismatch",
                "reason": "selected_frsts must be a sequence of explicit source FRST records",
            })
            return report
        expected_ids = []
        missing_fields = []
        for selected_record in expected_frsts:
            if not isinstance(selected_record, Mapping):
                missing_fields.append("mapping")
                continue
            value = next(
                (
                    selected_record.get(name)
                    for name in ("identity", "triangulation_identity", "frst_hash", "triangulation_hash")
                    if selected_record.get(name) is not None
                ),
                None,
            )
            if value is None:
                missing_fields.append("identity")
            else:
                expected_ids.append(_normalise_triangulation_identity(value))
        observed_ids = sorted(
            row.get("triangulation_identity")
            for row in raw_refinements
            if row.get("triangulation_identity") is not None
        )
        report["source_frst_catalog"] = {
            "expected_count": len(expected_ids),
            "observed_count": len(observed_ids),
            "expected_identities": sorted(expected_ids),
            "observed_identities": observed_ids,
            "index_space": [row.get("simplices_index_space") for row in expected_frsts],
            "status": "matched" if not missing_fields and sorted(expected_ids) == observed_ids else "mismatch",
        }
        if missing_fields or report["source_frst_catalog"]["status"] != "matched":
            report["terminal_records"].append({
                "terminal_status": "source_frst_identity_mismatch",
                "reason": "enumerated FRST identities do not match the immutable source FRST catalog",
                "source_frst_catalog": report["source_frst_catalog"],
                "missing_fields": missing_fields,
            })
            return report
    original_prime_labels = None
    if original_cy is not None and _callable(original_cy, "prime_toric_divisors"):
        try:
            original_prime_labels = np.asarray(original_cy.prime_toric_divisors(), dtype=np.int64).tolist()
        except Exception:
            original_prime_labels = None
    seen_refinements: set[str] = set()
    for refinement in raw_refinements:
        candidate = refinement.get("triangulation")
        if candidate is None:
            report["refinement_records"].append(_jsonable(refinement))
            continue
        identity = refinement["triangulation_identity"]
        if identity in seen_refinements:
            report["refinement_records"].append({
                "triangulation_identity": identity,
                "terminal_status": "duplicate_refinement",
            })
            continue
        seen_refinements.add(identity)
        candidate_record = {
            "candidate_index": refinement["candidate_index"],
            "triangulation_identity": identity,
            "point_count": refinement.get("point_count"),
            "point_count_reason": refinement.get("point_count_reason"),
            "flags": refinement["flags"],
            "selected_frst_identity": selected_identity,
            "selected_frst_replaced": False,
        }
        try:
            candidate_points = _points_from_object(candidate, name="candidate triangulation points")
            candidate_record["local_to_global"] = _coordinate_index_map(
                _points_from_object(poly, name="frozen global polytope points"),
                candidate_points,
                name="candidate triangulation local points",
            )[1]
            candidate_record["omitted_point_facet_certificate"] = omitted_point_facet_evidence(
                poly, candidate_points
            )
        except (TypeError, ValueError) as exc:
            candidate_record["terminal_status"] = "candidate_global_coordinate_mismatch"
            candidate_record["reason"] = str(exc)
            report["refinement_records"].append(candidate_record)
            continue
        if candidate_record["point_count"] != boundary_count:
            candidate_record["terminal_status"] = "candidate_source_point_count_mismatch"
            candidate_record["reason"] = "candidate local point count does not equal the source boundary-point count"
            report["refinement_records"].append(candidate_record)
            continue
        if candidate_record["omitted_point_facet_certificate"].get("status") != "omitted_facet_interior_points_certified":
            candidate_record["terminal_status"] = "candidate_omitted_point_certificate_mismatch"
            report["refinement_records"].append(candidate_record)
            continue
        required_flags = ("is_fine", "is_regular", "is_star")
        if any(refinement["flags"].get(name) is False for name in required_flags):
            candidate_record["terminal_status"] = "candidate_not_fine_regular_star"
            report["refinement_records"].append(candidate_record)
            continue
        if any(refinement["flags"].get(name) is not True for name in required_flags):
            candidate_record["terminal_status"] = "candidate_mpcp_flags_unavailable"
            candidate_record["reason"] = {
                name: refinement["flags"].get(f"{name}_reason", "public flag unavailable")
                for name in required_flags
                if refinement["flags"].get(name) is not True
            }
            report["refinement_records"].append(candidate_record)
            continue
        candidate_record["action_compatibility"] = []
        candidate_record["equivariant_refinement"] = []
        for action_index, action in enumerate(actions):
            if time.monotonic() > start + float(caps["max_seconds_per_index"]):
                report["terminal_records"].append({
                    "terminal_status": "resource_cap_seconds",
                    "reason": "per-index time cap reached before action evaluation",
                    "candidate_index": refinement["candidate_index"],
                    "action_index": action_index,
                })
                break
            if not isinstance(action, Mapping):
                candidate_record["action_compatibility"].append({
                    "action_index": action_index,
                    "status": "action_record_invalid",
                    "reason": "each action must be a mapping with lattice_matrix and torus_shift",
                })
                report["action_records"].append({
                    "candidate_index": refinement["candidate_index"],
                    "action_index": action_index,
                    "terminal_status": "action_record_invalid",
                    "reason": "each action must be a mapping with lattice_matrix and torus_shift",
                })
                continue
            try:
                compatibility = fan_action_evidence(poly, candidate, action.get("lattice_matrix"))
            except (TypeError, ValueError, KeyError) as exc:
                compatibility = {
                    "status": "fan_evidence_unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            compatibility["action_index"] = action_index
            candidate_record["action_compatibility"].append(compatibility)
            if action.get("lattice_matrix") is not None:
                height_evidence = symmetric_heights(poly, selected, action["lattice_matrix"])
                candidate_record["equivariant_refinement"].append({
                    "action_index": action_index,
                    "symmetric_heights": height_evidence,
                    "lower_subdivision": lower_subdivision_evidence(
                        poly,
                        height_evidence,
                        max_cells=int(caps["max_refinement_cells"]),
                    ),
                    "selected_frst_preserved": selected_identity == identity,
                })
            if compatibility.get("status") != "fan_preserved":
                report["action_records"].append({
                    "candidate_index": refinement["candidate_index"],
                    "action_index": action_index,
                    "terminal_status": "refinement_action_incompatible",
                    "reason": compatibility.get("reason", compatibility.get("status")),
                })
                continue
            evaluated = evaluate_action_on_refinement(
                poly,
                candidate,
                action,
                original_cy=original_cy,
                original_points=original_points,
                expected_hodge=source_identity["expected_hodge"],
            )
            evaluated["candidate_index"] = refinement["candidate_index"]
            evaluated["action_index"] = action_index
            report["action_records"].append(_jsonable(evaluated))
        candidate_record["terminal_status"] = "refinement_enumerated"
        report["refinement_records"].append(candidate_record)
        if time.monotonic() > start + float(caps["max_seconds_per_index"]):
            report["terminal_records"].append({"terminal_status": "resource_cap_seconds", "reason": "per-index time cap reached"})
            break
    # Attach the candidate FRST identity to every action record after the
    # complete refinement loop.  This keeps incompatible/terminal actions in
    # the accounting while giving computed certificates an immutable key.
    candidate_identities = {
        int(row["candidate_index"]): row.get("triangulation_identity")
        for row in report["refinement_records"]
        if row.get("candidate_index") is not None
    }
    for action_record in report["action_records"]:
        candidate_index = action_record.get("candidate_index")
        action_record["frst_hash"] = candidate_identities.get(candidate_index)
        action_record["selected_source_frst_hash"] = report["selected_frst"].get("identity")
        action_record["source_sha256"] = report["source_identity"].get("source_sha256")
        action_record["source_row"] = report["source_identity"].get("source_row")
        action_record["polytope_id"] = report["source_identity"].get("polytope_id")
        action_record["global_points"] = report["source_identity"].get("global_points")
    for action_record in report["action_records"]:
        certificate = build_replay_certificate(
            index,
            record,
            report,
            action_record,
        )
        if certificate is not None:
            report["replay_certificates"].append(certificate)
    report["counts"] = {
        "triangulations_yielded": len(raw_refinements),
        "refinements_retained": len(report["refinement_records"]),
        "action_evaluations": len(report["action_records"]),
        "action_terminal_records": sum(
            action.get("terminal_status") is not None
            for action in report["action_records"]
        ),
        "terminal_records": len(report["terminal_records"]),
        "replay_certificates": len(report["replay_certificates"]),
    }
    report["table1_status"] = _table1_status(record, report)
    report["same_table1_status"] = report["table1_status"]
    return report


def _table1_status(record: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    """Compare to an explicitly supplied target only; never embed geometry outcomes."""

    expected = record.get("table1_expected")
    observed = report.get("table1_observed")
    if expected is None or observed is None:
        return "not_comparable"
    return "match" if expected == observed else "mismatch"


def run_bounded_analysis(
    records: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    caps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze exactly indices 26, 31, and 33 with terminal accounting."""

    if isinstance(records, Mapping):
        lookup = {int(index): record for index, record in records.items()}
    else:
        lookup = {int(record["index"]): record for record in records}
    reports = {}
    for index in TARGET_INDICES:
        reports[str(index)] = analyze_replay_index(index, lookup.get(index, {}), caps=caps)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {"indices": list(TARGET_INDICES), "complete_h11_scans": False, "production_hdf5_writes": False},
        "runtime_provenance": runtime_provenance(),
        "reports": reports,
        "counts": {
            "indices_requested": len(TARGET_INDICES),
            "indices_with_terminal_records": sum(bool(report["terminal_records"]) for report in reports.values()),
        "indices_with_replay_data": sum(
            report.get("polytope_input", {}).get("status") in {
                "frozen_global_object_supplied",
                "constructed_from_frozen_global_coordinates",
            }
            for report in reports.values()
        ),
            "action_terminal_records": sum(
                report.get("counts", {}).get("action_terminal_records", 0)
                for report in reports.values()
            ),
        },
    }


def load_manifest(path: str | Path) -> dict[int, Mapping[str, Any]]:
    """Load a JSON replay manifest keyed by explicit index."""

    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, Mapping) and "records" in value:
        value = value["records"]
    if isinstance(value, Mapping):
        return {int(key): record for key, record in value.items()}
    if isinstance(value, list):
        return {int(record["index"]): record for record in value}
    raise ValueError("replay manifest must be a list or object keyed by index")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded replay and print JSON; do not write production data."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="explicit JSON replay manifest")
    parser.add_argument("--max-triangulations", type=int, default=DEFAULT_CAPS["max_triangulations"])
    parser.add_argument("--max-refinement-cells", type=int, default=DEFAULT_CAPS["max_refinement_cells"])
    parser.add_argument("--max-seconds-per-index", type=float, default=DEFAULT_CAPS["max_seconds_per_index"])
    args = parser.parse_args(argv)
    result = run_bounded_analysis(
        load_manifest(args.manifest),
        caps={
            "max_triangulations": args.max_triangulations,
            "max_refinement_cells": args.max_refinement_cells,
            "max_seconds_per_index": args.max_seconds_per_index,
        },
    )
    print(json.dumps(_jsonable(result), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
