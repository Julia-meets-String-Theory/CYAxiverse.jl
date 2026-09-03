"""Run or validate a bounded general-``L`` action replacement.

The module is intentionally provenance-first.  It does not select a population,
write the production database, or claim that the exact-action gate is valid.
The public helpers are also useful for validating small, immutable fixtures.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping


RUN_SCHEMA = "cyaxiverse-general-l-action-replacement-run-1.0"
WITNESS_SCHEMA = "cyaxiverse-exact-action-witness-manifest-1.1"
TERMINAL_SCHEMA = "cyaxiverse-orientifold-terminal-ledger-1.2"
VALIDATION_SCHEMA = "cyaxiverse-general-l-action-replacement-validation-1.0"
CANDIDATE_SCHEMA = "cyaxiverse-inherited-orientifold-candidate-3.0"
CHECKPOINT_SCHEMA = "cyaxiverse-general-l-action-replacement-checkpoint-1.0"
APPROVAL_SCHEMA = "cyaxiverse-general-l-action-replacement-approval-1.0"
TERMINAL_KINDS = {"matrix_validation", "candidate", "lattice_matrix_search_summary"}
STATUS_VOCABULARY = {
    "matrix_validation_passed", "numerical_geometry_failure", "polytope_not_preserved",
    "frst_not_preserved", "prime_divisor_set_not_preserved", "nonintegral_h2_action",
    "h2_action_not_involution", "torus_shift_not_involution",
    "orientifold_h11_minus_filter_rejection", "torus_shift_search_exhausted",
    "fixed_point_set_non_smooth", "smoothness_verification_unavailable",
    "accepted_verified_orientifold", "accepted_inherited_candidate",
    "h21_plus_nonzero", "exact_action_h21_evidence_unavailable",
}
CAPS = {
    2: {"max_favorable_polytopes": 36, "max_frst_classes": 36, "max_actions_per_class": 128, "max_action_attempts": 4608, "max_terminal_rows": 100_000, "max_wall_seconds": 7200, "max_new_output_bytes": 536870912},
    3: {"max_favorable_polytopes": 243, "max_frst_classes": 274, "max_actions_per_class": 128, "max_action_attempts": 35_072, "max_terminal_rows": 500_000, "max_wall_seconds": 14400, "max_new_output_bytes": 1073741824},
    4: {"max_favorable_polytopes": 1185, "max_frst_classes": 1760, "max_actions_per_class": 128, "max_action_attempts": 225_280, "max_terminal_rows": 2_000_000, "max_wall_seconds": 28800, "max_new_output_bytes": 2147483648},
    5: {"max_favorable_polytopes": 4897, "max_frst_classes": 11713, "max_actions_per_class": 128, "max_action_attempts": 1_499_264, "max_terminal_rows": 8_000_000, "max_wall_seconds": 57600, "max_new_output_bytes": 4294967296},
}
GLOBAL_LIMITS = {
    "max_rss_bytes": 2_147_483_648,
    "max_temporary_bytes": "equal_to_h11_output_ceiling",
    "workers": 1,
    "threads": {"OMP_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "MKL_NUM_THREADS": 1},
    "database_writes": 0,
}
REQUIRED_SOURCE_FILES = {
    "scripts/build_orientifold_axion_database.py",
    "scripts/inherited_orientifold_candidates.py",
    "scripts/trilayer_involutions.py",
    "scripts/orientifold_general_l_geometry.py",
    "scripts/orientifold_terminal_ledger.py",
    "scripts/toric_fixed_component_euler.py",
    "scripts/generate_geometric_data_multitriangulation.py",
    "scripts/glimmers_raw_frst.py",
    "scripts/reproduce_fuzzy_axions_h11_4.py",
    "validation/orientifold_exact_action_formula_ledger_20260822.md",
    "scripts/run_general_l_action_replacement_bounded.py",
}
APPROVAL_BINDING_FIELDS = (
    "task_id", "program", "h11_values", "counting_unit", "selection_route",
    "action_conventions", "terminal_conventions", "limits", "global_limits", "seed",
    "dependency_manifest_sha256", "project_toml_sha256", "manifest_toml_sha256",
    "runtime_versions", "relevant_environment_variables", "environment_revision",
    "source_file_digests", "source_commit", "tree_sha256",
    "working_tree_diff_sha256", "configuration_digest", "output_root",
    "checkpoint_root", "source_generation_output_root",
    "source_generation_checkpoint_root", "production_gate", "scale_status",
    "no_overwrite",
    "input_manifest_sha256",
)
COUNTER_NAMES = (
    "source_rows_seen", "favorable_polytopes_seen", "frst_classes_seen",
    "source_action_candidates", "matrix_validation_attempts", "candidate_action_attempts",
    "search_summary_rows", "terminal_rows", "accepted_action_count", "selected_class_count",
    "rejected_action_count", "malformed_rows", "blank_rows", "duplicate_class_count",
    "duplicate_action_count", "duplicate_terminal_identity_count", "orphan_class_count",
    "missing_action_digest_count", "live_minus_ledger", "ledger_minus_live", "status_by_reason",
)


class ContractError(ValueError):
    """Raise when a fail-closed contract is not satisfied."""


def _path_exists(path: Path) -> bool:
    """Include dangling symlinks in no-overwrite checks."""
    return os.path.lexists(os.fspath(path))


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON canonically, rejecting non-finite numbers."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def sha256_json(value: Any) -> str:
    """Hash canonical JSON using UTF-8 SHA-256."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def input_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Hash the input manifest without self- or approval-binding fields.

    The owner approval is created after this digest is known.  Its file
    fingerprint is then added to a copy of the manifest, so including that
    fingerprint here would make the approval binding circular.
    """
    payload = dict(manifest)
    payload.pop("input_manifest_sha256", None)
    payload.pop("approval_fingerprint", None)
    return sha256_json(payload)


# Keep descriptive names available to fixture and audit callers.
canonical_digest = sha256_json


def _rational_vector(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("numerator"), list):
        raise ContractError("torus_shift must be a common-denominator object")
    numerator = value["numerator"]
    denominator = value.get("denominator")
    if len(numerator) != 4 or isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ContractError("torus_shift requires four numerators and d>0")
    if any(isinstance(n, bool) or not isinstance(n, int) for n in numerator):
        raise ContractError("torus_shift numerators must be exact integers")
    try:
        nums = [int(n) for n in numerator]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError("torus_shift numerators must be integers") from exc
    divisor = abs(math.gcd(*nums, denominator))
    if divisor != 1:
        raise ContractError("torus_shift must be reduced to a common denominator")
    return {"numerator": nums, "denominator": denominator}


def _matrix(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise ContractError("lattice_matrix must be a 4 by 4 array")
    result = []
    for row in value:
        out = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ContractError("lattice_matrix entries must be exact integers")
            out.append(item)
        result.append(out)
    return result


def action_digest(action: Mapping[str, Any]) -> str:
    """Hash exact ``(L,t,lambda_f)`` without a floating-point conversion."""
    payload = {"lattice_matrix": _matrix(action["lattice_matrix"]),
               "torus_shift": _rational_vector(action["torus_shift"]),
               "lambda_f": action["lambda_f"]}
    if isinstance(payload["lambda_f"], bool) or not isinstance(payload["lambda_f"], int) or payload["lambda_f"] not in (0, 1):
        raise ContractError("lambda_f must be 0 or 1")
    return sha256_json(payload)


def matrix_digest(matrix: Any) -> str:
    """Hash the exact integer lattice matrix independently of its shift."""
    # Match build_orientifold_axion_database.lattice_matrix_digest: the
    # matrix digest hashes an object containing only the exact matrix.  It is
    # intentionally distinct from the full (L,t,lambda_f) action digest.
    return sha256_json({"lattice_matrix": _matrix(matrix)})


def validate_action(row: Mapping[str, Any], *, require_digest: bool = True) -> dict[str, Any]:
    """Normalize and validate an action, including its independently recomputed digest."""
    for key in ("lattice_matrix", "torus_shift", "lambda_f"):
        if key not in row:
            raise ContractError(f"action is missing {key}")
    normalized = {"lattice_matrix": _matrix(row["lattice_matrix"]),
                  "torus_shift": _rational_vector(row["torus_shift"]),
                  "lambda_f": row["lambda_f"]}
    digest = action_digest(normalized)
    if require_digest and not row.get("action_digest"):
        raise ContractError("missing action_digest")
    if row.get("action_digest") is not None and row["action_digest"] != digest:
        raise ContractError("action_digest mismatch")
    normalized["action_digest"] = digest
    normalized["matrix_digest"] = matrix_digest(normalized["lattice_matrix"])
    return normalized


def validate_action_evaluation(action: Mapping[str, Any]) -> None:
    """Require exact half-lattice shift and involution conditions."""
    normalized = validate_action(action, require_digest=False)
    denominator = normalized["torus_shift"]["denominator"]
    if any((2 * numerator) % denominator for numerator in normalized["torus_shift"]["numerator"]):
        raise ContractError("torus shift is not in the half-lattice")
    matrix = normalized["lattice_matrix"]
    square = [[sum(matrix[i][k] * matrix[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    identity = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    if square != identity:
        raise ContractError("L is not an involution")


def _accepted_hodge_evidence(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the exact-action Hodge evidence carried by an accepted row."""
    for candidate in (
        row.get("exact_action_h21_evidence"),
        row.get("hodge_split"),
        (row.get("terminal_evidence") or {}).get("exact_action_h21_evidence")
        if isinstance(row.get("terminal_evidence"), Mapping) else None,
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return None


def acceptance_contract_ok(row: Mapping[str, Any]) -> bool:
    """Check the complete O3/O7, h11-minus, and exact h21-plus boundary."""
    if row.get("terminal_status") != "accepted_verified_orientifold":
        return False
    evidence = _accepted_hodge_evidence(row)
    required_hodge = ("h11_plus", "h11_minus", "h21_plus", "h21_minus", "chi_fixed_locus", "chi_x")
    if not isinstance(evidence, Mapping) or any(
        isinstance(evidence.get(name), bool) or not isinstance(evidence.get(name), int)
        for name in required_hodge
    ):
        return False
    try:
        recomputed = hodge_split_from_euler(
            h11=evidence["h11_plus"] + evidence["h11_minus"],
            h21=evidence["h21_plus"] + evidence["h21_minus"],
            h11_minus=evidence["h11_minus"],
            chi_fixed_locus=evidence["chi_fixed_locus"],
            chi_x=evidence["chi_x"],
        )
    except ContractError:
        return False
    exact_hodge = all(recomputed[name] == evidence[name] for name in required_hodge)
    terminal_evidence = row.get("terminal_evidence")
    smoothness = (
        terminal_evidence.get("smoothness")
        if isinstance(terminal_evidence, Mapping)
        else row.get("smoothness")
    )
    has_smoothness = (
        isinstance(smoothness, Mapping) and smoothness.get("status") == "smooth"
    ) or (
        isinstance(terminal_evidence, Mapping)
        and terminal_evidence.get("smoothness_status") == "smooth"
    ) or (
        isinstance(row.get("smoothness"), Mapping)
        and row["smoothness"].get("status") == "smooth"
    )
    fixed_evidence = next(
        (
            row.get(name)
            for name in ("fixed_component_evidence", "fixed_point_components", "fixed_point_set")
            if row.get(name) is not None
        ),
        None,
    )
    if fixed_evidence is None and isinstance(terminal_evidence, Mapping):
        fixed_evidence = next(
            (
                terminal_evidence.get(name)
                for name in ("fixed_component_evidence", "fixed_point_components", "fixed_point_set")
                if terminal_evidence.get(name) is not None
            ),
            None,
        )
    has_fixed_evidence = isinstance(fixed_evidence, Mapping) and fixed_evidence.get("status") not in {
        "unavailable", "not_evaluated", "unknown"
    }
    return (
        isinstance(row.get("lambda_f"), int)
        and not isinstance(row.get("lambda_f"), bool)
        and row.get("lambda_f") == 1
        and isinstance(row.get("h11_minus"), int)
        and not isinstance(row.get("h11_minus"), bool)
        and row.get("h11_minus") == 0
        and evidence.get("status") == "validated"
        and evidence["h11_minus"] == 0
        and evidence["h21_plus"] == 0
        and exact_hodge
        and has_smoothness
        and has_fixed_evidence
    )


def hodge_split_from_euler(*, h11: int, h21: int, h11_minus: int,
                           chi_fixed_locus: int, chi_x: int) -> dict[str, int]:
    """Apply Moritz Eq. (4.51) with exact divisibility and sign checks."""
    values = (h11, h21, h11_minus, chi_fixed_locus, chi_x)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ContractError("Hodge and Euler inputs must be exact integers")
    h11, h21, h11_minus, chi_fixed_locus, chi_x = map(int, values)
    delta = chi_fixed_locus - chi_x
    if delta % 4:
        raise ContractError("chi(F_I)-chi(X) is not divisible by four")
    h21_minus = h11_minus + delta // 4 - 1
    result = {"h11_plus": h11 - h11_minus, "h11_minus": h11_minus,
              "h21_plus": h21 - h21_minus, "h21_minus": h21_minus,
              "chi_fixed_locus": chi_fixed_locus, "chi_x": chi_x}
    if min(result[name] for name in ("h11_plus", "h11_minus", "h21_plus", "h21_minus")) < 0:
        raise ContractError("negative Hodge eigenspace dimension")
    return result


def build_witness_record(source_candidate: Mapping[str, Any], action: Mapping[str, Any] | None,
                         terminal_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Build one lossless live/ledger witness record from its three payloads."""
    if not isinstance(source_candidate, Mapping) or not isinstance(terminal_evidence, Mapping):
        raise ContractError("source candidate and terminal evidence must be objects")
    required = ("polytope_id", "frst_hash")
    if any(key not in source_candidate for key in required):
        raise ContractError("source candidate is missing class identity")
    record = json.loads(json.dumps(source_candidate, allow_nan=False))
    record["source_candidate"] = json.loads(json.dumps(source_candidate, allow_nan=False))
    record["terminal_evidence"] = json.loads(json.dumps(terminal_evidence, allow_nan=False))
    record.setdefault("candidate_id", source_candidate.get("candidate_id"))
    record.setdefault("record_kind", "candidate")
    record.setdefault("terminal_status", terminal_evidence.get("terminal_status", "smoothness_verification_unavailable"))
    record.setdefault("terminal_reason_code", terminal_evidence.get("terminal_reason_code", record["terminal_status"]))
    if action is None:
        # Matrix/search rows carry a source matrix identifier in some legacy
        # enumerator outputs.  Keep it inside the lossless source payload but
        # use the contract's structural terminal identity at the top level.
        if record.get("record_kind") != "candidate":
            record["source_trilayer_candidate"] = record["source_candidate"]
            record["candidate_id"] = None
        record.update({"action_digest": None, "lattice_matrix": None, "torus_shift": None, "lambda_f": None})
    else:
        action_with_digest = dict(action)
        if not action_with_digest.get("action_digest"):
            action_with_digest["action_digest"] = action_digest(action_with_digest)
        normalized = validate_action(action_with_digest)
        record.update(normalized)
        if (
            normalized["lambda_f"] == 0
            and record.get("terminal_status") == "accepted_verified_orientifold"
        ):
            record["terminal_reason_code"] = "lambda_f_zero_out_of_branch"
            record["selection_status"] = "excluded_o3_o7_lambda_f_zero"
        if terminal_evidence.get("action_evaluation_enabled", True):
            try:
                validate_action_evaluation(record)
            except ContractError:
                if record.get("terminal_status") == "accepted_verified_orientifold":
                    raise
    return finalize_terminal(record)


def terminal_identity(row: Mapping[str, Any]) -> str:
    """Compute the stable terminal identity from class, candidate, and action."""
    required = ("polytope_id", "frst_hash")
    if any(key not in row or row[key] is None for key in required):
        raise ContractError("terminal identity fields are incomplete")
    payload = {
        "polytope_id": row["polytope_id"],
        "frst_hash": row["frst_hash"],
        "candidate_id": row.get("candidate_id"),
        "action_digest": row.get("action_digest"),
    }
    if payload["candidate_id"] is None and payload["action_digest"] is None:
        fallback = row.get("source_trilayer_candidate")
        if fallback is None:
            raise ContractError("structural terminal identity fallback is missing")
        payload["source_trilayer_candidate"] = fallback
    return sha256_json(payload)


terminal_record_identity = terminal_identity


def terminal_digest(row: Mapping[str, Any]) -> str:
    """Hash the complete terminal record, excluding only its two digests."""
    payload = dict(row)
    payload.pop("terminal_record_identity", None)
    payload.pop("terminal_record_digest", None)
    return sha256_json(payload)


terminal_record_digest = terminal_digest


def finalize_terminal(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and add terminal identity and complete-record digest."""
    result = json.loads(json.dumps(row, allow_nan=False))
    if result.get("ledger_schema_version", TERMINAL_SCHEMA) != TERMINAL_SCHEMA:
        raise ContractError("terminal schema mismatch")
    if result.get("record_kind") not in TERMINAL_KINDS:
        raise ContractError("unknown terminal record kind")
    if not isinstance(result.get("terminal_status"), str) or not result["terminal_status"]:
        raise ContractError("terminal_status is required")
    if result.get("record_kind") != "candidate" and any(
        result.get(field) is not None
        for field in ("candidate_id", "action_digest", "lattice_matrix", "torus_shift", "lambda_f")
    ):
        raise ContractError("matrix/search terminal rows must not carry an action triple")
    if result.get("terminal_status") not in STATUS_VOCABULARY:
        result["original_terminal_status"] = result.get("terminal_status")
        result["validation_failure_reason"] = "unknown_terminal_status"
    result["ledger_schema_version"] = TERMINAL_SCHEMA
    result.setdefault("action_digest", None)
    result["terminal_record_identity"] = terminal_identity(result)
    result["terminal_record_digest"] = terminal_digest(result)
    return result


def verify_terminal(row: Mapping[str, Any]) -> None:
    """Fail if a terminal row's identity or complete-record digest is tampered."""
    if row.get("record_kind") == "candidate" and row.get("action_digest") is not None:
        validate_action(row, require_digest=True)
    if row.get("terminal_record_identity") != terminal_identity(row):
        raise ContractError("terminal_record_identity mismatch")
    if row.get("terminal_record_digest") != terminal_digest(row):
        raise ContractError("terminal_record_digest mismatch")


def class_key(row: Mapping[str, Any]) -> str:
    return f"{row['polytope_id']}::{row['frst_hash']}"


def action_key(row: Mapping[str, Any]) -> str:
    if not row.get("action_digest"):
        raise ContractError("missing action_digest")
    return f"{class_key(row)}::{row['action_digest']}"


def compare_witnesses(live: Iterable[Mapping[str, Any]], ledger: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare class, action, and terminal sets in both directions."""
    live, ledger = list(live), list(ledger)
    for row in live + ledger:
        verify_terminal(row)
    def sets(rows: list[Mapping[str, Any]], key):
        values = [key(row) for row in rows]
        return set(values), sorted(Counter(values).items())
    result = {}
    for name, key in (("class", class_key), ("action", action_key),
                      ("terminal", lambda row: (row["terminal_record_identity"], row["terminal_record_digest"]))):
        left_rows = [row for row in live if name != "action" or row.get("action_digest") is not None]
        right_rows = [row for row in ledger if name != "action" or row.get("action_digest") is not None]
        left, left_dupes = sets(left_rows, key); right, right_dupes = sets(right_rows, key)
        result[name] = {"live_minus_ledger": sorted(left - right), "ledger_minus_live": sorted(right - left),
                        "live_duplicates": [k for k, n in left_dupes if n > 1],
                        "ledger_duplicates": [k for k, n in right_dupes if n > 1],
                        "equal": left == right and not any(n > 1 for _, n in left_dupes + right_dupes)}
    result["equal"] = all(result[name]["equal"] for name in ("class", "action", "terminal"))
    result["missing_action_digest_count"] = sum(
        row.get("action_digest") is None
        for row in live + ledger
        if row.get("record_kind") == "candidate"
    )
    if result["missing_action_digest_count"]:
        result["action"]["equal"] = False
        result["equal"] = False
    return result


def account_terminal_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Count every terminal kind and reason, including search summaries."""
    rows = list(rows)
    counts: dict[str, Any] = {name: 0 for name in COUNTER_NAMES}
    counts["status_by_reason"] = {}
    counts["unknown_status_count"] = 0
    counts["acceptance_contract_failure_count"] = 0
    counts["terminal_rows"] = len(rows)
    representatives: dict[str, str] = {}
    record_kind_counts: Counter[str] = Counter()
    terminal_status_counts: Counter[str] = Counter()
    class_seen, class_rows, action_seen, terminal_seen = set(), [], [], []
    for row in rows:
        verify_terminal(row)
        kind = row["record_kind"]
        record_kind_counts[kind] += 1
        terminal_status_counts[str(row.get("terminal_status"))] += 1
        if kind == "matrix_validation": counts["matrix_validation_attempts"] += 1
        elif kind == "candidate": counts["candidate_action_attempts"] += 1
        elif kind == "lattice_matrix_search_summary": counts["search_summary_rows"] += 1
        if kind == "candidate":
            if acceptance_contract_ok(row):
                counts["accepted_action_count"] += 1
                key = class_key(row)
                digest = row["action_digest"]
                if key not in representatives or digest < representatives[key]:
                    representatives[key] = digest
            else:
                counts["rejected_action_count"] += 1
                if (
                    row.get("terminal_status") == "accepted_verified_orientifold"
                    and row.get("lambda_f") != 0
                ):
                    counts["acceptance_contract_failure_count"] += 1
        class_value = class_key(row); class_seen.add(class_value); class_rows.append(class_value); terminal_seen.append(row["terminal_record_identity"])
        if kind == "candidate":
            if row.get("action_digest") is None: counts["missing_action_digest_count"] += 1
            else: action_seen.append(action_key(row))
        reason = str(row.get("terminal_reason_code") or row["terminal_status"])
        counts["status_by_reason"][reason] = counts["status_by_reason"].get(reason, 0) + 1
        if row.get("terminal_status") not in STATUS_VOCABULARY: counts["unknown_status_count"] += 1
    counts["source_rows_seen"] = len(rows)
    counts["favorable_polytopes_seen"] = len({row["polytope_id"] for row in rows})
    counts["source_action_candidates"] = sum(
        row.get("record_kind") == "candidate" and row.get("action_digest") is not None
        for row in rows
    )
    counts["selected_class_count"] = len({
        class_key(row)
        for row in rows
        if acceptance_contract_ok(row)
    })
    counts["record_kind_counts"] = dict(sorted(record_kind_counts.items()))
    counts["terminal_status_counts"] = dict(sorted(terminal_status_counts.items()))
    counts["frst_classes_seen"] = len(class_seen)
    counts["duplicate_class_count"] = len(class_rows) - len(set(class_rows))
    counts["duplicate_action_count"] = len(action_seen) - len(set(action_seen))
    counts["duplicate_terminal_identity_count"] = len(terminal_seen) - len(set(terminal_seen))
    counts["representative_action_digest_by_class"] = dict(sorted(representatives.items()))
    if counts["terminal_rows"] != counts["matrix_validation_attempts"] + counts["candidate_action_attempts"] + counts["search_summary_rows"]:
        raise ContractError("terminal accounting identity failed")
    return counts


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _approval_file_fingerprint(path: Path) -> dict[str, Any]:
    """Return the exact immutable identity used by the execution CLI."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ContractError("blocked_on_evidence: owner approval file is missing")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _canonical_absolute_path(value: Any, *, name: str) -> Path:
    """Return a canonical absolute path and reject relative aliases."""
    if not isinstance(value, (str, os.PathLike)):
        raise ContractError(f"path_mismatch: {name} must be an absolute path")
    raw = Path(os.fspath(value))
    if not raw.is_absolute():
        raise ContractError(f"path_mismatch: {name} must be an absolute path")
    if any(part in {".", ".."} for part in raw.parts):
        raise ContractError(f"path_mismatch: {name} must be a canonical path")
    if raw.is_symlink():
        raise ContractError(f"path_mismatch: {name} must not be a symlink alias")
    resolved = raw.resolve(strict=False)
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two canonical paths are equal or nested."""
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _validate_source_manifest_for_preparation(
    manifest: Mapping[str, Any], *, repo_root: Path | None = None
) -> tuple[Path, Path]:
    """Validate an unbound source manifest before assigning bounded roots."""
    if not isinstance(manifest, Mapping):
        raise ContractError("schema_mismatch: source manifest must be an object")
    if manifest.get("schema") != "cyaxiverse-general-l-action-replacement-input-1.0":
        raise ContractError("schema_mismatch")
    if "approval_fingerprint" in manifest and manifest["approval_fingerprint"] is not None:
        raise ContractError("provenance_mismatch: source manifest is already bound")
    if manifest.get("input_manifest_sha256") != input_manifest_digest(manifest):
        raise ContractError("input_fingerprint_mismatch: stale source manifest")
    if manifest.get("run_scope") != "pilot":
        raise ContractError("provenance_mismatch: source manifest is not a pilot")
    required_fields = set(APPROVAL_BINDING_FIELDS) - {
        "source_generation_output_root",
        "source_generation_checkpoint_root",
        "input_manifest_sha256",
    }
    if any(field not in manifest for field in required_fields):
        raise ContractError("input_fingerprint_mismatch: incomplete source bindings")
    if (
        manifest.get("h11_values") != [2, 3, 4, 5]
        or manifest.get("seed") != 0
        or _normalized_limits(manifest.get("limits")) != CAPS
        or manifest.get("global_limits") != GLOBAL_LIMITS
        or manifest.get("production_gate") != "not_validated"
        or manifest.get("scale_status") != "not_applicable"
        or manifest.get("no_overwrite") is not True
        or not isinstance(manifest.get("source_file_digests"), Mapping)
    ):
        raise ContractError("provenance_mismatch: source bindings are not bounded")
    source_output = _canonical_absolute_path(
        manifest.get("output_root"), name="source-generation output_root"
    )
    source_checkpoint = _canonical_absolute_path(
        manifest.get("checkpoint_root"), name="source-generation checkpoint_root"
    )
    if not _path_exists(source_output) or not source_output.is_dir():
        raise ContractError("input_fingerprint_mismatch: stale source output root")
    if not _path_exists(source_checkpoint) or not source_checkpoint.is_dir():
        raise ContractError("input_fingerprint_mismatch: stale source checkpoint root")
    if source_output == source_checkpoint or _paths_overlap(source_output, source_checkpoint):
        raise ContractError("provenance_mismatch: source roots must be distinct")
    entries = manifest.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ContractError("input_fingerprint_mismatch: source inputs are missing")
    seen: set[tuple[int, str]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ContractError("input_fingerprint_mismatch: malformed source input")
        role = entry.get("role")
        h11 = entry.get("h11")
        if role not in {"source_rows", "terminal_ledger"} or h11 not in {2, 3, 4, 5}:
            raise ContractError("schema_mismatch: malformed source input")
        key = (int(h11), str(role))
        if key in seen:
            raise ContractError("input_fingerprint_mismatch: duplicate source input")
        seen.add(key)
        if entry.get("output_root") != manifest.get("output_root"):
            raise ContractError("input_fingerprint_mismatch: stale source input binding")
    if seen != {(h11, role) for h11 in (2, 3, 4, 5) for role in ("source_rows", "terminal_ledger")}:
        raise ContractError("input_fingerprint_mismatch: incomplete source inputs")
    refingerprint_manifest(manifest, repo_root=repo_root)
    if repo_root is not None:
        revision = repository_revision(repo_root.resolve())
        if any(
            manifest.get(key) != revision.get(key)
            for key in ("source_commit", "tree_sha256", "working_tree_diff_sha256")
        ):
            raise ContractError("provenance_mismatch: stale source revision")
    return source_output, source_checkpoint


def prepare_bounded_manifest(
    manifest: Mapping[str, Any],
    *,
    output_root: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    output_manifest_path: str | os.PathLike[str],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Create a bounded manifest from an immutable source manifest.

    Preserve the source-generation bindings and source-file fingerprints while
    assigning two fresh, canonical roots for a later bounded execution. Do
    not modify the source manifest or overwrite any output.
    """
    source_output, source_checkpoint = _validate_source_manifest_for_preparation(
        manifest, repo_root=repo_root
    )
    if "source_generation_output_root" in manifest or "source_generation_checkpoint_root" in manifest:
        raise ContractError("provenance_mismatch: manifest is already prepared")
    bounded_output = _canonical_absolute_path(output_root, name="bounded output_root")
    bounded_checkpoint = _canonical_absolute_path(
        checkpoint_root, name="bounded checkpoint_root"
    )
    output_manifest = _canonical_absolute_path(
        output_manifest_path, name="output manifest"
    )
    if _path_exists(bounded_output):
        raise FileExistsError(f"refusing to overwrite bounded output root: {bounded_output}")
    if _path_exists(bounded_checkpoint):
        raise FileExistsError(
            f"refusing to overwrite bounded checkpoint root: {bounded_checkpoint}"
        )
    if bounded_output == bounded_checkpoint or _paths_overlap(bounded_output, bounded_checkpoint):
        raise ContractError("provenance_mismatch: bounded roots must be distinct")
    if _paths_overlap(bounded_output, source_output) or _paths_overlap(
        bounded_output, source_checkpoint
    ) or _paths_overlap(bounded_checkpoint, source_output) or _paths_overlap(
        bounded_checkpoint, source_checkpoint
    ):
        raise ContractError("provenance_mismatch: bounded roots alias source roots")
    if _path_exists(output_manifest):
        raise FileExistsError(f"refusing to overwrite prepared manifest: {output_manifest}")
    if any(
        _paths_overlap(output_manifest, root)
        for root in (
            source_output,
            source_checkpoint,
            bounded_output,
            bounded_checkpoint,
        )
    ):
        raise ContractError("provenance_mismatch: manifest path aliases a run root")

    prepared = copy.deepcopy(dict(manifest))
    # Retain the exact source-manifest spellings as immutable provenance. The
    # validation helpers canonicalize them only for comparison and safety.
    prepared["source_generation_output_root"] = manifest["output_root"]
    prepared["source_generation_checkpoint_root"] = manifest["checkpoint_root"]
    prepared["output_root"] = str(bounded_output)
    prepared["checkpoint_root"] = str(bounded_checkpoint)
    prepared.pop("approval_fingerprint", None)
    for entry in prepared["inputs"]:
        entry["output_root"] = str(bounded_output)
        entry["checkpoint_root"] = str(bounded_checkpoint)
    # Match the JSON representation returned by ``load_json``. In particular,
    # JSON object keys such as the h11 limits are strings after serialization.
    prepared = json.loads(canonical_bytes(prepared))
    prepared["input_manifest_sha256"] = input_manifest_digest(prepared)
    write_json_zst(output_manifest, prepared)
    return prepared


def prepare_bounded_manifest_from_files(
    manifest_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    output_manifest_path: str | os.PathLike[str],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load an unbound source manifest and create a prepared copy."""
    source_path = _canonical_absolute_path(manifest_path, name="source manifest")
    destination = _canonical_absolute_path(output_manifest_path, name="output manifest")
    if source_path == destination:
        raise ContractError("provenance_mismatch: source and output manifests must differ")
    return prepare_bounded_manifest(
        load_json(source_path),
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        output_manifest_path=destination,
        repo_root=repo_root,
    )


def _validate_prepared_roots(manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    """Validate the source and fresh bounded roots in a prepared manifest."""
    source_output = _canonical_absolute_path(
        manifest.get("source_generation_output_root"),
        name="source-generation output_root",
    )
    source_checkpoint = _canonical_absolute_path(
        manifest.get("source_generation_checkpoint_root"),
        name="source-generation checkpoint_root",
    )
    bounded_output = _canonical_absolute_path(
        manifest.get("output_root"), name="bounded output_root"
    )
    bounded_checkpoint = _canonical_absolute_path(
        manifest.get("checkpoint_root"), name="bounded checkpoint_root"
    )
    if not source_output.is_dir() or not source_checkpoint.is_dir():
        raise ContractError("input_fingerprint_mismatch: stale source-generation root")
    if (
        source_output == source_checkpoint
        or _paths_overlap(source_output, source_checkpoint)
        or bounded_output == bounded_checkpoint
        or _paths_overlap(bounded_output, bounded_checkpoint)
        or _paths_overlap(bounded_output, source_output)
        or _paths_overlap(bounded_output, source_checkpoint)
        or _paths_overlap(bounded_checkpoint, source_output)
        or _paths_overlap(bounded_checkpoint, source_checkpoint)
    ):
        raise ContractError("provenance_mismatch: prepared roots are aliased")
    if _path_exists(bounded_output):
        raise FileExistsError(f"refusing to overwrite bounded output root: {bounded_output}")
    if _path_exists(bounded_checkpoint):
        raise FileExistsError(
            f"refusing to overwrite bounded checkpoint root: {bounded_checkpoint}"
        )
    return bounded_output, bounded_checkpoint


def create_approval_bound_manifest(
    manifest: Mapping[str, Any],
    approval: Mapping[str, Any],
    *,
    approval_path: str | os.PathLike[str],
    output_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create an approval-bound manifest without modifying either input.

    The input manifest digest excludes only ``input_manifest_sha256`` and the
    later ``approval_fingerprint`` field.  The owner first approves that
    digest, then this create-only operation binds the approval file bytes to a
    new manifest.  The execution CLI can therefore verify the approval file
    exactly without a self-referential hash.
    """
    if manifest.get("schema") != "cyaxiverse-general-l-action-replacement-input-1.0":
        raise ContractError("schema_mismatch")
    if approval.get("schema") != APPROVAL_SCHEMA:
        raise ContractError("schema_mismatch")
    if manifest.get("approval_fingerprint") is not None:
        raise ContractError("provenance_mismatch: manifest is already approval-bound")
    if not all(
        isinstance(manifest.get(key), str)
        for key in (
            "source_generation_output_root",
            "source_generation_checkpoint_root",
        )
    ):
        raise ContractError("provenance_mismatch: manifest is not prepared")
    if approval.get("status") not in {"approved", "owner_approved"}:
        raise ContractError("blocked_on_evidence: new owner approval is required")
    if any(
        key not in manifest or key not in approval or approval[key] != manifest[key]
        for key in APPROVAL_BINDING_FIELDS
    ):
        raise ContractError("provenance_mismatch")
    if any(
        key not in approval
        for key in ("approval_id", "approval_date", "new_bounded_run_authorized")
    ):
        raise ContractError("blocked_on_evidence: new owner approval is required")
    if (
        approval["new_bounded_run_authorized"] is not True
        or not isinstance(approval["approval_id"], str)
        or not approval["approval_id"]
        or not isinstance(approval["approval_date"], str)
        or not approval["approval_date"]
        or approval["input_manifest_sha256"] != input_manifest_digest(manifest)
    ):
        raise ContractError("provenance_mismatch")
    _validate_prepared_roots(manifest)
    output = _canonical_absolute_path(output_manifest_path, name="output manifest")
    approval_file = _canonical_absolute_path(approval_path, name="approval file")
    if output == approval_file:
        raise ContractError("provenance_mismatch: approval and output paths must differ")
    source_roots = (
        _canonical_absolute_path(
            manifest["source_generation_output_root"],
            name="source-generation output_root",
        ),
        _canonical_absolute_path(
            manifest["source_generation_checkpoint_root"],
            name="source-generation checkpoint_root",
        ),
    )
    bounded_roots = (
        _canonical_absolute_path(manifest["output_root"], name="bounded output_root"),
        _canonical_absolute_path(
            manifest["checkpoint_root"], name="bounded checkpoint_root"
        ),
    )
    if any(_paths_overlap(output, root) for root in (*source_roots, *bounded_roots)):
        raise ContractError("provenance_mismatch: output manifest aliases a run root")
    if any(_paths_overlap(approval_file, root) for root in (*source_roots, *bounded_roots)):
        raise ContractError("provenance_mismatch: approval file aliases a run root")
    try:
        approval_on_disk = load_json(approval_file)
    except (ContractError, OSError) as exc:
        raise ContractError("input_fingerprint_mismatch: malformed approval") from exc
    if canonical_bytes(approval_on_disk) != canonical_bytes(approval):
        raise ContractError("input_fingerprint_mismatch: approval content mismatch")
    bound = dict(manifest)
    bound["approval_fingerprint"] = _approval_file_fingerprint(approval_file)
    if _path_exists(output):
        raise FileExistsError(f"refusing to overwrite approval-bound manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # ``write_json_zst`` and ``atomic_create`` are defined below and resolved
    # when this function is called after module initialization.
    write_json_zst(output, bound)
    return bound


def create_approval_bound_manifest_from_files(
    manifest_path: str | os.PathLike[str],
    approval_path: str | os.PathLike[str],
    output_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load owner inputs and create an immutable approval-bound manifest."""
    return create_approval_bound_manifest(
        load_json(Path(manifest_path).expanduser().resolve()),
        load_json(Path(approval_path).expanduser().resolve()),
        approval_path=approval_path,
        output_manifest_path=output_manifest_path,
    )


def _verify_file_fingerprint(path: Path, size_bytes: Any, expected_sha256: Any) -> None:
    """Verify one immutable file identity without accepting a path fallback."""
    if (
        not path.is_absolute()
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not path.is_file()
        or path.stat().st_size != size_bytes
        or _file_sha256(path) != expected_sha256
    ):
        raise ContractError("input_fingerprint_mismatch")


def refingerprint_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path | None = None
) -> None:
    """Recompute every immutable input and source-code fingerprint."""
    if manifest.get("schema") != "cyaxiverse-general-l-action-replacement-input-1.0":
        raise ContractError("schema_mismatch")
    if manifest.get("input_manifest_sha256") != input_manifest_digest(manifest):
        raise ContractError("input_fingerprint_mismatch")
    entries = manifest.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ContractError("input_fingerprint_mismatch")
    seen_paths = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or not all(
            entry.get(field) is not None
            for field in (
                "h11", "role", "file_type", "source_row_or_partition_identity",
                "selection_route", "counting_unit", "path", "size_bytes", "sha256",
            )
        ):
            raise ContractError("input_fingerprint_mismatch")
        path = Path(str(entry["path"])).expanduser()
        if not path.is_absolute():
            raise ContractError("input_fingerprint_mismatch")
        path = path.resolve()
        if path in seen_paths:
            raise ContractError("input_fingerprint_mismatch")
        seen_paths.add(path)
        _verify_file_fingerprint(path, entry["size_bytes"], entry["sha256"])

    if repo_root is not None:
        repo_root = repo_root.resolve()
        for name, expected in (
            ("Project.toml", manifest.get("project_toml_sha256")),
            ("Manifest.toml", manifest.get("manifest_toml_sha256")),
        ):
            path = repo_root / name
            if expected is None or not path.is_file():
                raise ContractError("input_fingerprint_mismatch")
            _verify_file_fingerprint(path, path.stat().st_size, expected)
        source_digests = manifest.get("source_file_digests")
        if not isinstance(source_digests, Mapping) or not source_digests:
            raise ContractError("input_fingerprint_mismatch")
        if not REQUIRED_SOURCE_FILES.issubset({str(name) for name in source_digests}):
            raise ContractError("input_fingerprint_mismatch")
        for name, expected in source_digests.items():
            path = Path(str(name))
            if not path.is_absolute():
                path = repo_root / path
            path = path.resolve()
            if not path.is_file():
                raise ContractError("input_fingerprint_mismatch")
            _verify_file_fingerprint(path, path.stat().st_size, expected)


def _git(command: list[str], root: Path) -> str:
    try:
        return subprocess.run(command, cwd=root, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"provenance_mismatch: cannot obtain {' '.join(command)}") from exc


def repository_revision(root: Path) -> dict[str, str]:
    """Return the exact repository revision and working-tree fingerprints."""
    head = _git(["git", "rev-parse", "HEAD"], root)
    tree = _git(["git", "rev-parse", "HEAD^{tree}"], root)
    diff = hashlib.sha256(subprocess.run(["git", "diff", "--no-ext-diff", "--binary", "HEAD"], cwd=root,
                                         check=True, stdout=subprocess.PIPE).stdout).hexdigest()
    return {"source_commit": head, "tree_sha256": tree, "working_tree_diff_sha256": diff}


def _read_compressed(path: Path) -> bytes:
    if path.suffix != ".zst": return path.read_bytes()
    zstd = shutil.which("zstd")
    if zstd is None: raise ContractError("zstd is required to read compressed inputs")
    try:
        return subprocess.run([zstd, "-dc", "-q", str(path)], check=True, stdout=subprocess.PIPE).stdout
    except subprocess.CalledProcessError as exc:
        raise ContractError("input_fingerprint_mismatch: invalid zstd stream") from exc


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Load JSONL and return rows plus blank and malformed counts."""
    rows, blank, malformed = [], 0, 0
    for line_number, raw in enumerate(_read_compressed(path).splitlines(), 1):
        if not raw.strip(): blank += 1; continue
        try: value = json.loads(raw)
        except json.JSONDecodeError: malformed += 1; continue
        if not isinstance(value, dict): malformed += 1; continue
        rows.append(value)
    return rows, blank, malformed


def _normalized_limits(value: Any) -> dict[int, dict[str, Any]]:
    """Normalize JSON object keys before comparing the h11 limit contract."""
    if not isinstance(value, Mapping):
        raise ContractError("provenance_mismatch")
    try:
        normalized = {int(key): dict(item) for key, item in value.items()}
    except (TypeError, ValueError):
        raise ContractError("provenance_mismatch") from None
    if any(not isinstance(item, Mapping) for item in value.values()):
        raise ContractError("provenance_mismatch")
    return normalized


def _validate_binding(approval: Mapping[str, Any], manifest: Mapping[str, Any], output: Path) -> None:
    """Require approval and input bindings to agree exactly."""
    if (
        approval.get("schema") != "cyaxiverse-general-l-action-replacement-approval-1.0"
        or manifest.get("schema") != "cyaxiverse-general-l-action-replacement-input-1.0"
    ):
        raise ContractError("schema_mismatch")
    if approval.get("status") not in {"approved", "owner_approved"}:
        raise ContractError("blocked_on_evidence: new owner approval is required")
    for key in APPROVAL_BINDING_FIELDS:
        if key not in approval or key not in manifest or approval[key] != manifest[key]:
            raise ContractError("provenance_mismatch")
    approval_fingerprint = manifest.get("approval_fingerprint")
    if not isinstance(approval_fingerprint, Mapping):
        raise ContractError("input_fingerprint_mismatch: approval fingerprint is required")
    if any(key not in approval for key in ("approval_id", "approval_date", "new_bounded_run_authorized")):
        raise ContractError("blocked_on_evidence: new owner approval is required")
    if (
        approval["new_bounded_run_authorized"] is not True
        or not isinstance(approval["approval_id"], str)
        or not approval["approval_id"]
        or not isinstance(approval["approval_date"], str)
        or not approval["approval_date"]
        or sorted(approval["h11_values"]) != [2, 3, 4, 5]
        or approval["seed"] != 0
        or _normalized_limits(approval["limits"]) != CAPS
        or approval["global_limits"] != GLOBAL_LIMITS
        or approval["production_gate"] != "not_validated"
        or approval["scale_status"] != "not_applicable"
        or approval["no_overwrite"] is not True
        or approval["input_manifest_sha256"] != input_manifest_digest(manifest)
    ):
        raise ContractError("provenance_mismatch")
    runtime_versions = approval["runtime_versions"]
    if not isinstance(runtime_versions, Mapping) or any(
        not isinstance(runtime_versions.get(name), str) or not runtime_versions[name]
        for name in ("python_version", "julia_version", "cytools_version")
    ):
        raise ContractError("provenance_mismatch")
    if runtime_versions["python_version"] != platform.python_version():
        raise ContractError("provenance_mismatch")
    environment = approval["relevant_environment_variables"]
    if not isinstance(environment, Mapping):
        raise ContractError("provenance_mismatch")
    for name, expected in environment.items():
        if expected is None:
            if name in os.environ:
                raise ContractError("provenance_mismatch")
        elif os.environ.get(str(name)) != str(expected):
            raise ContractError("provenance_mismatch")
    bounded_output, bounded_checkpoint = _validate_prepared_roots(manifest)
    if bounded_output != output.resolve():
        raise ContractError("provenance_mismatch")
    if bounded_checkpoint != _canonical_absolute_path(
        approval["checkpoint_root"], name="bounded checkpoint_root"
    ):
        raise ContractError("provenance_mismatch")


def _source_entries(manifest: Mapping[str, Any]) -> dict[int, dict[str, dict[str, Any]]]:
    entries: dict[int, dict[str, dict[str, Any]]] = {h11: {} for h11 in (2, 3, 4, 5)}
    for entry in manifest.get("inputs", []):
        for field in (
            "source_commit", "tree_sha256", "working_tree_diff_sha256",
            "environment_revision", "configuration_digest", "seed", "limits",
            "global_limits", "output_root", "checkpoint_root", "selection_route",
            "counting_unit",
        ):
            if entry.get(field) != manifest.get(field): raise ContractError("input_fingerprint_mismatch")
        if entry.get("role") not in {"source_rows", "terminal_ledger"} or entry.get("h11") not in entries:
            raise ContractError("schema_mismatch")
        role = entry["role"]
        if role in entries[entry["h11"]]: raise ContractError("duplicate input partition")
        entries[entry["h11"]][role] = entry
    if any(set(value) != {"source_rows", "terminal_ledger"} for value in entries.values()): raise ContractError("missing source partition")
    return entries


def _witness_rows(rows: list[dict[str, Any]], h11: int) -> list[dict[str, Any]]:
    witnesses = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContractError("malformed source JSON")
        if "h11" not in row or row["h11"] != h11:
            raise ContractError("source row h11 mismatch")
        if "source_candidate" in row:
            candidate = row["source_candidate"]
            evidence = row.get("terminal_evidence", {})
            action = {key: row[key] for key in ("lattice_matrix", "torus_shift", "lambda_f") if key in row}
            if "action_digest" in row:
                action["action_digest"] = row["action_digest"]
            if not all(key in action for key in ("lattice_matrix", "torus_shift", "lambda_f")):
                action = None
        else:
            candidate = row
            evidence = row.get("terminal_evidence")
            if evidence is None:
                evidence = {
                    key: value
                    for key, value in row.items()
                    if key not in {"terminal_record_identity", "terminal_record_digest"}
                }
            action = ({key: row[key] for key in ("lattice_matrix", "torus_shift", "lambda_f", "action_digest") if key in row}
                      if row.get("record_kind", "candidate") == "candidate" else None)
            if len(action) < 3: action = None
        if not isinstance(candidate, Mapping) or not isinstance(evidence, Mapping):
            raise ContractError("source candidate and terminal evidence must be objects")
        candidate = dict(candidate)
        candidate_schema = candidate.get("schema_version", candidate.get("candidate_schema_version"))
        if candidate_schema is None:
            raise ContractError("schema_mismatch: candidate schema is missing")
        if candidate_schema != CANDIDATE_SCHEMA:
            raise ContractError("schema_mismatch: unsupported candidate schema")
        if candidate.get("candidate_id") is None and row.get("candidate_id") is not None:
            candidate["candidate_id"] = row["candidate_id"]
        if action is None and candidate.get("candidate_id") is None:
            candidate.setdefault("source_trilayer_candidate", dict(candidate))
        witness = build_witness_record(candidate, action, evidence)
        if "terminal_record_identity" in row and (row["terminal_record_identity"] != witness["terminal_record_identity"] or row.get("terminal_record_digest") != witness["terminal_record_digest"]):
            raise ContractError("terminal_record_digest_mismatch")
        witness["source_row"] = row.get("source_row")
        if isinstance(witness["source_row"], bool) or not isinstance(witness["source_row"], int):
            raise ContractError("unknown source row order")
        if (
            isinstance(witness.get("frst_class_index"), bool)
            or not isinstance(witness.get("frst_class_index"), int)
        ):
            raise ContractError("unknown FRST class order")
        witnesses.append(witness)
    return sorted(
        witnesses,
        key=lambda row: (
            row["source_row"], row["polytope_id"], row["frst_hash"],
            row.get("frst_class_index", 0), row.get("action_digest") or "",
            row["terminal_record_identity"], row["terminal_record_digest"],
        ),
    )


def _publish_tree(stage: Path, output: Path) -> None:
    if _path_exists(output): raise FileExistsError(f"refusing to overwrite output root {output}")
    os.rename(stage, output)


def _compress_zstd(raw: bytes) -> bytes:
    zstd = shutil.which("zstd")
    if zstd is None: raise ContractError("zstd is required for published artifacts")
    return subprocess.run(
        [zstd, "-19", "-q", "-c"], input=raw, stdout=subprocess.PIPE, check=True
    ).stdout


def _json_zst_payload(value: Any) -> bytes:
    return _compress_zstd(canonical_bytes(value))


def _jsonl_zst_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    return _compress_zstd(raw)


def _write_jsonl_zst(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_create(path, _jsonl_zst_payload(rows))


def _current_rss_bytes() -> int:
    """Return the process peak RSS in bytes on the supported host platforms."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def execute_bounded(approval: Mapping[str, Any], manifest: Mapping[str, Any], output: Path,
                    *, repo_root: Path, resume: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute the deterministic source-fixture adapter and publish complete artifacts."""
    _validate_binding(approval, manifest, output)
    revision = repository_revision(repo_root)
    if any(manifest.get(key) != revision.get(key) for key in ("source_commit", "tree_sha256", "working_tree_diff_sha256")):
        raise ContractError("provenance_mismatch")
    refingerprint_manifest(manifest, repo_root=repo_root)
    approval_fingerprint = manifest.get("approval_fingerprint")
    if not isinstance(approval_fingerprint, Mapping):
        raise ContractError("input_fingerprint_mismatch: approval fingerprint is required")
    try:
        observed_approval_fingerprint = _approval_file_fingerprint(
            Path(str(approval_fingerprint["path"]))
        )
    except (KeyError, TypeError, ContractError) as exc:
        raise ContractError("input_fingerprint_mismatch: approval fingerprint") from exc
    if dict(approval_fingerprint) != observed_approval_fingerprint:
        raise ContractError("input_fingerprint_mismatch: approval fingerprint")
    if resume is not None: validate_resume(resume, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    partitions = _source_entries(manifest)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    summaries, all_files = {}, []
    try:
        for h11 in (2, 3, 4, 5):
            h11_started = time.monotonic()
            if _current_rss_bytes() > GLOBAL_LIMITS["max_rss_bytes"]:
                raise ContractError("resource_cap_exceeded")
            source_rows, source_blank, source_bad = load_jsonl(Path(partitions[h11]["source_rows"]["path"]))
            ledger_rows, ledger_blank, ledger_bad = load_jsonl(Path(partitions[h11]["terminal_ledger"]["path"]))
            live = _witness_rows(source_rows, h11); ledger = _witness_rows(ledger_rows, h11)
            comparison = compare_witnesses(live, ledger)
            counters = account_terminal_rows(live)
            ledger_counters = account_terminal_rows(ledger)
            counters["source_rows_seen"] = len(source_rows)
            counters["blank_rows"] = source_blank + ledger_blank
            counters["malformed_rows"] = source_bad + ledger_bad
            counters["source_action_candidates"] = sum(
                row.get("record_kind") == "candidate" and row.get("action_digest") is not None
                for row in live
            )
            counters["favorable_polytopes_seen"] = len({row["polytope_id"] for row in live})
            for name in (
                "matrix_validation_attempts", "candidate_action_attempts", "search_summary_rows",
                "terminal_rows", "accepted_action_count", "selected_class_count",
                "rejected_action_count", "status_by_reason", "representative_action_digest_by_class",
                "acceptance_contract_failure_count",
            ):
                counters[name] = ledger_counters[name]
            for name in (
                "duplicate_class_count", "duplicate_action_count",
                "duplicate_terminal_identity_count", "missing_action_digest_count",
                "unknown_status_count",
            ):
                counters[name] = counters.get(name, 0) + ledger_counters.get(name, 0)
            counters["orphan_class_count"] = len(
                set(comparison["class"]["live_minus_ledger"])
                | set(comparison["class"]["ledger_minus_live"])
            )
            counters["live_minus_ledger"] = len(comparison["class"]["live_minus_ledger"]) + len(comparison["action"]["live_minus_ledger"])
            counters["ledger_minus_live"] = len(comparison["class"]["ledger_minus_live"]) + len(comparison["action"]["ledger_minus_live"])
            summary = bounded_gate(h11, counters, fingerprints_match=True); summary["comparison"] = comparison
            if source_bad or ledger_bad or source_blank or ledger_blank:
                summary["status"] = "blocked_on_evidence"
                summary["failure_reasons"].append("malformed_or_blank_source_rows") if source_bad or ledger_bad else None
                summary["failure_reasons"].append("blank_source_rows") if source_blank or ledger_blank else None
            if not comparison["equal"]: summary["status"] = "blocked_on_evidence"; summary["failure_reasons"].append("witness_mismatch")
            per_class_counts = (
                Counter(class_key(row) for row in rows if row.get("record_kind") == "candidate")
                for rows in (live, ledger)
            )
            if any(
                count > CAPS[h11]["max_actions_per_class"]
                for per_class in per_class_counts
                for count in per_class.values()
            ):
                summary["status"] = "blocked_on_evidence"
                summary["failure_reasons"].append("resource_cap_exceeded")
            if counters["matrix_validation_attempts"] + counters["candidate_action_attempts"] > CAPS[h11]["max_action_attempts"]:
                summary["status"] = "blocked_on_evidence"
                summary["failure_reasons"].append("resource_cap_exceeded")
            if counters.get("acceptance_contract_failure_count", 0):
                summary["status"] = "blocked_on_evidence"
                summary["failure_reasons"].append("acceptance_contract_failure")
            summary["failure_reasons"] = sorted(set(summary["failure_reasons"]))
            summary.update({
                "selection_route": approval["selection_route"],
                "counting_unit": approval["counting_unit"],
                "action_conventions": approval["action_conventions"],
                "terminal_conventions": approval["terminal_conventions"],
                "provenance": {
                    "approval_id": approval["approval_id"],
                    "source_commit": revision["source_commit"],
                    "tree_sha256": revision["tree_sha256"],
                    "working_tree_diff_sha256": revision["working_tree_diff_sha256"],
                    "environment_revision": approval["environment_revision"],
                    "configuration_digest": approval["configuration_digest"],
                },
                "representative_action_digest_by_class": ledger_counters["representative_action_digest_by_class"],
            })
            if time.monotonic() - h11_started > CAPS[h11]["max_wall_seconds"]:
                summary["status"] = "blocked_on_evidence"
                summary["failure_reasons"] = sorted(set(summary["failure_reasons"] + ["resource_cap_exceeded"]))
            summaries[h11] = summary
            prefix = f"h11-{h11:03d}"
            provenance = {"approval_id": approval["approval_id"], "source_commit": revision["source_commit"],
                          "tree_sha256": revision["tree_sha256"], "working_tree_diff_sha256": revision["working_tree_diff_sha256"],
                          "environment_revision": approval["environment_revision"], "configuration_digest": approval["configuration_digest"]}
            witness_metadata = {"selection_route": approval["selection_route"], "counting_unit": approval["counting_unit"],
                                "limits": CAPS[h11], "seed": approval["seed"], "status": summary["status"]}
            outputs = {f"{prefix}.live-action-witness-manifest.json.zst": {"schema": WITNESS_SCHEMA, "h11": h11, "provenance": provenance, "metadata": witness_metadata, "record_count": len(live), "records": live},
                       f"{prefix}.ledger-action-witness-manifest.json.zst": {"schema": WITNESS_SCHEMA, "h11": h11, "provenance": provenance, "metadata": witness_metadata, "records": ledger},
                       f"{prefix}.terminal-ledger.summary.json.zst": {"schema": TERMINAL_SCHEMA, "h11": h11, "provenance": provenance, "metadata": witness_metadata, "record_count": len(ledger), "record_kind_counts": dict(ledger_counters.get("record_kind_counts", {})), "terminal_status_counts": dict(ledger_counters.get("terminal_status_counts", {})), "representative_action_digest_by_class": ledger_counters["representative_action_digest_by_class"], "attempt_accounting": {"matrix_validation_attempts": ledger_counters["matrix_validation_attempts"], "candidate_attempts": ledger_counters["candidate_action_attempts"], "search_summary_rows": ledger_counters["search_summary_rows"], "terminal_records": ledger_counters["terminal_rows"]}, "counters": counters},
                       f"{prefix}.validation-summary.json.zst": summary}
            serialized = {name: _json_zst_payload(value) for name, value in outputs.items()}
            serialized[f"{prefix}.terminal-ledger.jsonl.zst"] = _jsonl_zst_payload(ledger)
            if sum(len(payload) for payload in serialized.values()) > CAPS[h11]["max_new_output_bytes"]:
                summary["status"] = "blocked_on_evidence"
                summary["failure_reasons"] = sorted(set(summary["failure_reasons"] + ["resource_cap_exceeded"]))
                witness_metadata["status"] = summary["status"]
                outputs[f"{prefix}.terminal-ledger.summary.json.zst"]["metadata"] = witness_metadata
                outputs[f"{prefix}.validation-summary.json.zst"] = summary
                serialized = {name: _json_zst_payload(value) for name, value in outputs.items()}
                serialized[f"{prefix}.terminal-ledger.jsonl.zst"] = _jsonl_zst_payload(ledger)
                if sum(len(payload) for payload in serialized.values()) > CAPS[h11]["max_new_output_bytes"]:
                    raise ContractError("resource_cap_exceeded: output ceiling")
            for name, payload in serialized.items():
                path = stage / name
                atomic_create(path, payload)
                all_files.append(path)
        run = {"schema": RUN_SCHEMA, "status": "passed" if all(item["status"] == "passed" for item in summaries.values()) else "blocked_on_evidence",
               "production_gate": "not_validated", "scale_status": "not_applicable", "h11_values": [2, 3, 4, 5], "summaries": summaries,
               "approval": approval, "input_manifest": manifest, "repository_revision": revision,
               "checkpoint_root": approval["checkpoint_root"],
               "resume": {"used": resume is not None, "last_class": resume.get("last_class") if resume is not None else None},
               "execution_adapter": "immutable-jsonl-source-witness-validation"}
        write_json_zst(stage / "run-manifest.json.zst", run); all_files.append(stage / "run-manifest.json.zst")
        sums = "".join(f"{_file_sha256(path)}  {path.name}\n" for path in sorted(all_files, key=lambda path: path.name)).encode()
        atomic_create(stage / "SHA256SUMS.txt", sums); all_files.append(stage / "SHA256SUMS.txt")
        _publish_tree(stage, output)
        return run
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def atomic_create(path: Path, payload: bytes) -> None:
    """Publish bytes atomically without replacing an existing artifact."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(path): raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name); stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_json_zst(path: Path, value: Any) -> None:
    """Write compact JSON with zstd level 19, atomically and create-only."""
    atomic_create(Path(path), _json_zst_payload(value))


def load_json(path: Path) -> Any:
    """Read JSON or zstd-compressed JSON and reject malformed documents."""
    try:
        return json.loads(_read_compressed(Path(path)))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError("malformed JSON") from exc


def validate_resume(checkpoint: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Reject a checkpoint unless every binding field matches exactly."""
    fields = ("input_manifest_sha256", "code_revision", "environment_revision", "configuration_digest", "seed", "limits", "output_root")
    expected_code_revision = expected.get("code_revision", expected.get("source_commit"))
    boundary = checkpoint.get("last_class")
    boundary_shape_ok = (
        isinstance(boundary, list)
        and len(boundary) == 4
        and isinstance(boundary[0], int)
        and not isinstance(boundary[0], bool)
        and isinstance(boundary[1], str)
        and isinstance(boundary[2], str)
        and isinstance(boundary[3], int)
        and not isinstance(boundary[3], bool)
    )
    if checkpoint.get("schema") != "cyaxiverse-general-l-action-replacement-checkpoint-1.0" or any(
        checkpoint.get(field) != (expected_code_revision if field == "code_revision" else expected.get(field))
        for field in fields
    ) or checkpoint.get("last_class_complete") is not True or not boundary_shape_ok:
        raise ContractError("resume_mismatch")


def next_class(
    rows: Iterable[Mapping[str, Any]], checkpoint: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return rows strictly after a verified completed class boundary."""
    validate_resume(checkpoint, expected if expected is not None else checkpoint)
    boundary = tuple(checkpoint["last_class"])
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            row["source_row"], row["polytope_id"], row["frst_hash"],
            row.get("frst_class_index", 0), row.get("action_digest") or "",
        ),
    )
    return [
        row
        for row in ordered
        if (
            row["source_row"], row["polytope_id"], row["frst_hash"],
            row.get("frst_class_index", 0),
        ) > boundary
    ]


def bounded_gate(h11: int, counters: Mapping[str, Any], *, fingerprints_match: bool = True) -> dict[str, Any]:
    """Return a fail-closed validation summary for one bounded h11 value."""
    cap = CAPS.get(int(h11))
    if cap is None: raise ContractError("unsupported h11")
    failures = []
    if not fingerprints_match: failures.append("input_fingerprint_mismatch")
    if counters.get("unknown_status_count", 0): failures.append("schema_mismatch")
    for name, limit_name in (
        ("favorable_polytopes_seen", "max_favorable_polytopes"),
        ("frst_classes_seen", "max_frst_classes"),
        ("terminal_rows", "max_terminal_rows"),
        ("matrix_validation_attempts", "max_action_attempts"),
        ("candidate_action_attempts", "max_action_attempts"),
    ):
        if counters.get(name, 0) > cap[limit_name]:
            failures.append("resource_cap_exceeded")
    if counters.get("missing_action_digest_count", 0):
        failures.append("missing_action_digest")
    if counters.get("acceptance_contract_failure_count", 0):
        failures.append("acceptance_contract_failure")
    if counters.get("malformed_rows", 0):
        failures.append("malformed_rows")
    if counters.get("blank_rows", 0):
        failures.append("blank_rows")
    if any(
        counters.get(name, 0)
        for name in (
            "duplicate_class_count", "duplicate_action_count",
            "duplicate_terminal_identity_count", "orphan_class_count",
            "live_minus_ledger", "ledger_minus_live",
        )
    ):
        failures.append("witness_mismatch")
    return {"schema": VALIDATION_SCHEMA, "h11": int(h11), "status": "passed" if not failures else "blocked_on_evidence",
            "production_gate": "not_validated", "scale_status": "not_applicable", "failure_reasons": sorted(set(failures)),
            "counters": dict(counters), "limits": cap}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval")
    parser.add_argument("--input-manifest")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--checkpoint-root",
        help="Bounded checkpoint root for --prepare-bounded-manifest.",
    )
    parser.add_argument(
        "--bounded-output-root",
        help="Fresh bounded output root for --prepare-bounded-manifest.",
    )
    parser.add_argument(
        "--bounded-checkpoint-root",
        help="Fresh bounded checkpoint root for --prepare-bounded-manifest.",
    )
    parser.add_argument(
        "--output-manifest",
        help="Create-only output path for preparation or approval binding.",
    )
    parser.add_argument(
        "--prepare-bounded-manifest",
        action="store_true",
        help=(
            "Create a fresh bounded manifest from an unbound source manifest; "
            "run this before --bind-approval-manifest."
        ),
    )
    parser.add_argument(
        "--bind-approval-manifest",
        action="store_true",
        help="Bind an owner approval file to a new manifest and exit.",
    )
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    try:
        if args.prepare_bounded_manifest and args.bind_approval_manifest:
            raise ContractError(
                "--prepare-bounded-manifest and --bind-approval-manifest "
                "are mutually exclusive"
            )
        if args.output_root is not None and args.bounded_output_root is not None:
            raise ContractError(
                "--output-root and --bounded-output-root are mutually exclusive"
            )
        if args.checkpoint_root is not None and args.bounded_checkpoint_root is not None:
            raise ContractError(
                "--checkpoint-root and --bounded-checkpoint-root are mutually exclusive"
            )
        if args.bind_approval_manifest and any(
            value is not None
            for value in (
                args.checkpoint_root,
                args.bounded_output_root,
                args.bounded_checkpoint_root,
            )
        ):
            raise ContractError(
                "bounded preparation roots require --prepare-bounded-manifest"
            )
        if args.bind_approval_manifest and (
            args.output_root is not None or args.resume is not None
        ):
            raise ContractError(
                "--output-root and --resume are execution-only and are not "
                "valid with --bind-approval-manifest"
            )
        if args.prepare_bounded_manifest:
            if args.approval or args.resume or not args.input_manifest or not args.output_manifest:
                raise ContractError(
                    "--prepare-bounded-manifest requires --input-manifest, "
                    "--output-manifest, and fresh bounded roots only"
                )
            bounded_output = args.bounded_output_root or args.output_root
            bounded_checkpoint = args.bounded_checkpoint_root or args.checkpoint_root
            if not bounded_output or not bounded_checkpoint:
                raise ContractError(
                    "--prepare-bounded-manifest requires --output-root and "
                    "--checkpoint-root (or the --bounded-* aliases)"
                )
            prepared = prepare_bounded_manifest_from_files(
                args.input_manifest,
                bounded_output,
                bounded_checkpoint,
                args.output_manifest,
                repo_root=Path(__file__).resolve().parent.parent,
            )
            print(json.dumps(prepared, sort_keys=True, indent=2))
            return 0
        if args.bind_approval_manifest:
            if not args.approval or not args.input_manifest or not args.output_manifest:
                raise ContractError(
                    "--bind-approval-manifest requires --approval, --input-manifest, "
                    "and --output-manifest"
                )
            bound = create_approval_bound_manifest_from_files(
                args.input_manifest, args.approval, args.output_manifest
            )
            print(json.dumps(bound, sort_keys=True, indent=2))
            return 0
        if any(
            value is not None
            for value in (
                args.checkpoint_root,
                args.bounded_output_root,
                args.bounded_checkpoint_root,
            )
        ):
            raise ContractError(
                "bounded preparation roots require --prepare-bounded-manifest"
            )
        if args.output_manifest is not None:
            raise ContractError(
                "--output-manifest requires --prepare-bounded-manifest or "
                "--bind-approval-manifest"
            )
        if not args.approval or not args.input_manifest or not args.output_root:
            raise ContractError(
                "execution requires --approval, --input-manifest, and --output-root"
            )
        approval_path = Path(args.approval).expanduser().resolve()
        manifest_path = Path(args.input_manifest).expanduser().resolve()
        approval = load_json(approval_path)
        manifest = load_json(manifest_path)
        approval_fingerprint = manifest.get("approval_fingerprint")
        observed_approval_fingerprint = {
            "path": str(approval_path),
            "size_bytes": approval_path.stat().st_size,
            "sha256": _file_sha256(approval_path),
        }
        if approval_fingerprint != observed_approval_fingerprint:
            raise ContractError("input_fingerprint_mismatch: approval fingerprint")
        output = Path(args.output_root).expanduser().resolve()
        if _path_exists(output):
            raise FileExistsError(f"refusing to overwrite output root {output}")
        checkpoint = load_json(Path(args.resume)) if args.resume else None
        execute_bounded(approval, manifest, output, repo_root=Path(__file__).resolve().parent.parent, resume=checkpoint)
    except (ContractError, FileExistsError, OSError) as exc:
        raise SystemExit(f"blocked_on_evidence: {exc}") from exc
    return 0


if __name__ == "__main__":
    main()
