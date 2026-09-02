#!/usr/bin/env python3
"""Validate legacy and canonical physical-scale checkpoint bundles.

The validator is intentionally read-only.  It rejects a missing, malformed,
mutated, or incompletely bound bundle before a resume can treat it as
terminal.  The legacy format is accepted only for the one explicitly supplied
geometry-1 adoption input; all newly written checkpoints use the canonical
JSON format.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


SCALES = ("0.9", "0.95", "0.99", "1.0", "1.01", "1.05", "1.1")
COVERAGE = "partial_index_range"
LEGACY_SCHEMA = "physical-scale-pilot-geometry-checkpoint-v1"
CANONICAL_SCHEMA = "physical-scale-inflation-geometry-checkpoint-v6"
DOMAIN_CERTIFICATE_VERSION = "physical-domain-certificate-3"
CONVERSION_POLICY_VERSION = "kinv-mixed-tolerance-v1"
KINV_CONVERSION_RULE = "max_absolute_error <= 1e-12 OR max_relative_error <= 1e-12"
QUARANTINE_SCHEMA = "physical-scale-inflation-nonterminal-partial-quarantine-v1"
QUARANTINE_VERSION = "nonterminal-partials-v1"
OLD_CONVERSION_POLICY_VERSION = "kinv-absolute-tolerance-v1"
OLD_KINV_CONVERSION_RULE = "max_absolute_error <= 1e-12"
OLD_QUARANTINE_FAILURE_REASON = (
    "uncheckpointed header-only partial from old Kinv absolute-only policy"
)
CURRENT_QUARANTINE_FAILURE_REASON = (
    "uncheckpointed header-only partial from lazy-transpose array-hash failure"
)
EXECUTION_SOURCE_FILES = (
    "scripts/run_physical_scale_inflation_pilot_20260825.jl",
    "scripts/validate_physical_scale_checkpoint_20260825.py",
    "scripts/validate_physical_scaling_sidecars_20260825.py",
    "scripts/generate_physical_scaling_sidecars_20260825.jl",
    "scripts/preflight_physical_scaling_20260825.jl",
    "scripts/inflation_scale_continuation.jl",
    "scripts/inflation_scan_shards_common.jl",
    "scripts/inflation_diagnostics_common.jl",
    "scripts/inflation_refinement_common.jl",
    "scripts/audit_physical_certificate.jl",
    "src/CYAxiverse.jl",
    "src/read.jl",
    "src/generate.jl",
    "src/structs.jl",
    "Project.toml",
    "../../CYAxiverse.jl/Manifest.toml",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def output_tree_bytes(root: pathlib.Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _root_relative(root: pathlib.Path, relative: str, label: str) -> pathlib.Path:
    require(relative and not pathlib.PurePath(relative).is_absolute(),
            f"{label} must be a relative path")
    path = (root / relative).resolve()
    root_resolved = root.resolve()
    require(path == root_resolved or root_resolved in path.parents,
            f"{label} escapes output root")
    return path


def _validate_reappeared_terminal_bundle(
    output_root: pathlib.Path,
    original_root_path: pathlib.Path,
    quarantine: dict[str, Any],
    *,
    expected_policy: str = "",
    expected_common: str = "",
    expected_selection: str = "",
    expected_diff: str = "",
    expected_source_manifest: str = "",
) -> dict[str, Any]:
    """Prove that a later original path is a terminal bundle, not a partial.

    The quarantine inventory remains the immutable historical header-only
    bundle.  A later calculation may recreate the canonical geometry output,
    but only a matching current v6 checkpoint may make that path acceptable.
    """
    geometry = quarantine["geometry"]
    h11, polytope, frst = (int(geometry[key]) for key in ("h11", "polytope", "frst"))
    checkpoint = output_root / "checkpoints" / (
        f"h11_{h11:03d}_np_{polytope:07d}_cy_{frst:07d}.checkpoint-v6.json"
    )
    require(checkpoint.is_file(),
            "reappeared quarantine original requires a matching terminal checkpoint")
    checksum_path = pathlib.Path(str(checkpoint) + ".sha256")
    require(checksum_path.is_file(),
            "reappeared quarantine original terminal checkpoint checksum is missing")
    summary_path = original_root_path / "summary.csv"
    shard_dir = original_root_path / "shards"
    require(summary_path.is_file() and shard_dir.is_dir(),
            "reappeared quarantine original terminal outputs are incomplete")
    shard_paths = sorted(path for path in shard_dir.iterdir()
                         if path.is_file() and path.suffix == ".csv")
    require(len(shard_paths) == 1,
            "reappeared quarantine original requires exactly one terminal shard")
    actual_files = sorted(path for path in original_root_path.rglob("*") if path.is_file())
    require(all(not path.is_symlink() for path in actual_files),
            "reappeared quarantine original contains a symlink")
    require(actual_files == sorted([summary_path, shard_paths[0]]),
            "reappeared quarantine original contains unexpected files")
    checkpoint_obj = json.loads(checkpoint.read_text(encoding="utf-8"))
    require(checkpoint_obj.get("schema") == CANONICAL_SCHEMA and
            checkpoint_obj.get("checkpoint_version") == 6,
            "reappeared quarantine original checkpoint schema is not v6")
    require(checkpoint_obj.get("geometry_index") == int(quarantine["geometry_index"]),
            "reappeared quarantine original checkpoint geometry mismatch")
    require(checkpoint_obj.get("geometry_artifact_sha256") ==
            quarantine["geometry_artifact_sha256"],
            "reappeared quarantine original artifact mismatch")
    for key, expected_value in (
        ("policy_sha256", expected_policy),
        ("common_configuration_digest_sha256", expected_common),
        ("selection_manifest_sha256", expected_selection),
        ("current_complete_git_diff_sha256", expected_diff),
        ("execution_source_manifest_sha256", expected_source_manifest),
    ):
        if expected_value:
            require(checkpoint_obj.get(key) == expected_value,
                    f"reappeared quarantine original identity mismatch: {key}")
    checkpoint_summary = checkpoint_obj.get("summary", {})
    checkpoint_shards = checkpoint_obj.get("shards", {}).get("files", [])
    require(checkpoint_summary.get("path") == str(summary_path) and
            isinstance(checkpoint_shards, list) and len(checkpoint_shards) == 1 and
            checkpoint_shards[0].get("path") == str(shard_paths[0]),
            "reappeared quarantine original checkpoint does not bind its outputs")
    identity_keys = (
        "geometry_artifact_sha256", "geometry_sidecar_sha256", "policy_sha256",
        "common_configuration_digest_sha256", "selection_manifest_sha256",
        "continuation_source_sha256", "current_complete_git_diff_sha256",
        "project_sha256", "manifest_sha256", "run_configuration_digest_sha256",
        "execution_source_manifest_sha256",
    )
    expected = {key: checkpoint_obj[key] for key in identity_keys}
    expected.update({
        "geometry_h11": str(h11),
        "geometry_polytope": str(polytope),
        "geometry_frst": str(frst),
        "geometry_index": str(quarantine["geometry_index"]),
    })
    validate_canonical_checkpoint(
        checkpoint, expected=expected, summary_path=summary_path,
        shard_path=shard_paths[0], output_root=output_root,
    )
    return checkpoint_obj


def validate_quarantine_manifest(
    path: pathlib.Path,
    *,
    output_root: pathlib.Path,
    expected_policy: str = "",
    expected_common: str = "",
    expected_selection: str = "",
    expected_diff: str = "",
    expected_source_manifest: str = "",
    expected_geometry_index: str = "",
) -> dict[str, Any]:
    require(path.is_file(), f"quarantine manifest is missing: {path}")
    checksum_path = pathlib.Path(str(path) + ".sha256")
    require(checksum_path.is_file(), "quarantine manifest checksum is missing")
    checksum_fields = checksum_path.read_text(encoding="utf-8").split()
    require(len(checksum_fields) == 2 and checksum_fields[1] == path.name,
            "quarantine manifest checksum line is malformed")
    actual = sha256_path(path)
    require(checksum_fields[0] == actual, "quarantine manifest checksum mismatch")
    obj = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "manifest_version", "quarantine_version", "status", "run_id",
        "geometry_index", "geometry", "geometry_artifact_sha256",
        "selection_manifest_sha256", "policy_sha256", "common_configuration_digest_sha256",
        "current_complete_git_diff_sha256", "execution_source_manifest_sha256",
        "execution_source_hashes", "old_conversion_policy_version",
        "old_kinv_conversion_acceptance", "failure_reason", "coverage_label",
        "max_negative_modes", "original_geometry_output_root",
        "quarantine_geometry_output_root", "files", "terminal_geometry_count_before",
        "terminal_geometry_indices_before", "atomic_move", "source_files_preserved",
        "prohibitions",
    }
    require(required.issubset(obj), "quarantine manifest is missing a required field")
    require(obj["schema"] == QUARANTINE_SCHEMA and obj["manifest_version"] == 1,
            "quarantine manifest schema mismatch")
    require(obj["quarantine_version"] == QUARANTINE_VERSION and obj["status"] == "quarantined",
            "quarantine manifest status/version mismatch")
    partial_policy = obj["old_conversion_policy_version"]
    partial_rule = obj["old_kinv_conversion_acceptance"]
    partial_reason = obj["failure_reason"]
    accepted_partial_identities = {
        (OLD_CONVERSION_POLICY_VERSION, OLD_KINV_CONVERSION_RULE,
         OLD_QUARANTINE_FAILURE_REASON),
        (CONVERSION_POLICY_VERSION, KINV_CONVERSION_RULE,
         CURRENT_QUARANTINE_FAILURE_REASON),
    }
    require((partial_policy, partial_rule, partial_reason) in accepted_partial_identities,
            "quarantine manifest partial-policy identity mismatch")
    require(obj["coverage_label"] == COVERAGE and obj["max_negative_modes"] == 1,
            "quarantine manifest coverage/policy mismatch")
    require(obj["atomic_move"] == "directory_rename" and obj["source_files_preserved"] is True,
            "quarantine manifest move/provenance identity mismatch")
    require(obj["prohibitions"] == expected_prohibitions(),
            "quarantine manifest prohibitions are incomplete or violated")
    geometry = obj["geometry"]
    require(isinstance(geometry, dict), "quarantine manifest geometry identity is malformed")
    h11, polytope, frst = (int(geometry.get(key, -1)) for key in ("h11", "polytope", "frst"))
    require(5 <= h11 <= 10 and 1 <= polytope <= 3 and frst == 1,
            "quarantine manifest geometry is outside the fixed 18")
    geometry_index = h11 * 1_000_000 + polytope * 1_000 + frst
    require(obj["geometry_index"] == geometry_index, "quarantine geometry index mismatch")
    if expected_geometry_index:
        require(geometry_index == int(expected_geometry_index),
                "quarantine manifest geometry index is not the requested next geometry")
    for key, expected in (
        ("policy_sha256", expected_policy),
        ("common_configuration_digest_sha256", expected_common),
        ("selection_manifest_sha256", expected_selection),
        ("current_complete_git_diff_sha256", expected_diff),
        ("execution_source_manifest_sha256", expected_source_manifest),
    ):
        if expected:
            require(obj.get(key) == expected, f"quarantine manifest identity mismatch: {key}")
    require(len(obj["geometry_artifact_sha256"]) == 64,
            "quarantine geometry artifact identity is malformed")
    require(isinstance(obj["execution_source_hashes"], dict) and obj["execution_source_hashes"],
            "quarantine execution-source map is missing")
    root = output_root.resolve()
    original_root = obj["original_geometry_output_root"]
    quarantine_root = obj["quarantine_geometry_output_root"]
    original_root_path = _root_relative(root, original_root, "original geometry output root")
    quarantine_root_path = _root_relative(root, quarantine_root, "quarantine geometry output root")
    require(original_root == f"geometries/h11_{h11:03d}_np_{polytope:07d}_cy_{frst:07d}",
            "quarantine original geometry root is not canonical")
    require(quarantine_root == f"quarantine/{QUARANTINE_VERSION}/h11_{h11:03d}_np_{polytope:07d}_cy_{frst:07d}",
            "quarantine destination root is not canonical")
    require(quarantine_root_path.is_dir(), "quarantine destination directory is missing")
    reappeared_terminal = None
    if original_root_path.exists():
        reappeared_terminal = _validate_reappeared_terminal_bundle(
            output_root, original_root_path, obj,
            expected_policy=expected_policy,
            expected_common=expected_common,
            expected_selection=expected_selection,
            expected_diff=expected_diff,
            expected_source_manifest=expected_source_manifest,
        )
    files = obj["files"]
    require(isinstance(files, list) and len(files) == 2,
            "quarantine file inventory must contain summary and shard")
    roles = {item.get("role") for item in files if isinstance(item, dict)}
    require(roles == {"summary", "shard"}, "quarantine file roles are incomplete or unexpected")
    seen_original: set[str] = set()
    seen_destination: set[str] = set()
    for item in files:
        require(isinstance(item, dict), "quarantine file inventory entry is malformed")
        for key in ("role", "original_relative_path", "quarantine_relative_path", "sha256",
                    "bytes", "data_rows", "header_sha256"):
            require(key in item, f"quarantine file inventory is missing {key}")
        original_relative = item["original_relative_path"]
        destination_relative = item["quarantine_relative_path"]
        require(original_relative not in seen_original and destination_relative not in seen_destination,
                "quarantine file inventory contains duplicate paths")
        seen_original.add(original_relative)
        seen_destination.add(destination_relative)
        original_path = _root_relative(root, original_relative, "quarantine original file")
        destination_path = _root_relative(root, destination_relative, "quarantine destination file")
        if reappeared_terminal is not None:
            require(original_path.is_file() and not original_path.is_symlink(),
                    "quarantine original terminal file is missing or is a symlink")
            require(sha256_path(original_path) != item["sha256"],
                    "quarantine historical hash is conflated with the terminal output")
        else:
            require(not original_path.exists(), "quarantine original file still exists")
        require(destination_path.is_file() and not destination_path.is_symlink(),
                "quarantine destination file is missing or is a symlink")
        require(destination_path.parent == quarantine_root_path or
                quarantine_root_path in destination_path.parents,
                "quarantine destination file is outside its geometry directory")
        data = destination_path.read_bytes()
        require(item["sha256"] == sha256_bytes(data),
                "quarantine destination file hash mismatch")
        require(item["header_sha256"] == item["sha256"],
                "quarantine header hash does not bind the preserved file")
        require(item["bytes"] == len(data) and item["data_rows"] == 0,
                "quarantine file byte/row accounting mismatch")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("quarantine file is not UTF-8") from error
        require(data.endswith(b"\n") and len(list(csv.reader(io.StringIO(text)))) == 1,
                "quarantine file is not header-only")
    require(len(seen_original) == 2 and len(seen_destination) == 2,
            "quarantine file inventory is incomplete")
    destination_files = {
        str(path.relative_to(root))
        for path in quarantine_root_path.rglob("*") if path.is_file()
    }
    require(destination_files == seen_destination,
            "quarantine destination contains an unexpected file")
    require(obj["terminal_geometry_count_before"] >= 0 and
            isinstance(obj["terminal_geometry_indices_before"], list),
            "quarantine terminal-count provenance is malformed")
    return obj


def rebind_quarantine_manifest(
    path: pathlib.Path,
    *,
    output_root: pathlib.Path,
    source_manifest_path: pathlib.Path,
    expected_policy: str,
    expected_common: str,
    expected_selection: str,
    expected_diff: str,
    expected_source_manifest: str,
    expected_geometry_index: str = "",
) -> tuple[dict[str, Any], str]:
    previous_digest = sha256_path(path)
    obj = validate_quarantine_manifest(path, output_root=output_root)
    source = validate_execution_source_manifest(source_manifest_path, expected_source_manifest)
    rebound_values = {
        "policy_sha256": expected_policy,
        "common_configuration_digest_sha256": expected_common,
        "selection_manifest_sha256": expected_selection,
        "current_complete_git_diff_sha256": expected_diff,
        "execution_source_manifest_sha256": expected_source_manifest,
    }
    require(all(rebound_values.values()), "quarantine rebind expected identity is incomplete")
    if (all(obj.get(key) == value for key, value in rebound_values.items()) and
            obj.get("execution_source_hashes") == source["source_file_hashes"]):
        current = validate_quarantine_manifest(
            path,
            output_root=output_root,
            expected_policy=expected_policy,
            expected_common=expected_common,
            expected_selection=expected_selection,
            expected_diff=expected_diff,
            expected_source_manifest=expected_source_manifest,
            expected_geometry_index=expected_geometry_index,
        )
        return current, previous_digest
    for key, value in (
        ("policy_sha256", expected_policy),
        ("common_configuration_digest_sha256", expected_common),
        ("selection_manifest_sha256", expected_selection),
        ("current_complete_git_diff_sha256", expected_diff),
        ("execution_source_manifest_sha256", expected_source_manifest),
    ):
        require(value, f"quarantine rebind expected identity is missing: {key}")
        obj[key] = value
    obj["execution_source_hashes"] = source["source_file_hashes"]
    obj["quarantine_identity_migration"] = {
        "previous_manifest_sha256": previous_digest,
        "rebound_without_recomputation": True,
        "reason": "approved source/policy identity regeneration; preserved quarantine files unchanged",
    }
    if expected_geometry_index:
        require(obj["geometry_index"] == int(expected_geometry_index),
                "quarantine manifest geometry index is not the requested next geometry")
    data = canonical(obj) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    checksum_temporary = pathlib.Path(str(temporary) + ".sha256")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
        digest = sha256_path(path)
        checksum_temporary.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        checksum_temporary.replace(pathlib.Path(str(path) + ".sha256"))
    finally:
        temporary.unlink(missing_ok=True)
        checksum_temporary.unlink(missing_ok=True)
    return validate_quarantine_manifest(
        path,
        output_root=output_root,
        expected_policy=expected_policy,
        expected_common=expected_common,
        expected_selection=expected_selection,
        expected_diff=expected_diff,
        expected_source_manifest=expected_source_manifest,
        expected_geometry_index=expected_geometry_index,
    ), previous_digest


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_legacy_checkpoint(path: pathlib.Path) -> dict[str, str]:
    require(path.is_file(), f"legacy checkpoint is missing: {path}")
    fields: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require("=" in line, f"legacy checkpoint line {line_number} is malformed")
        key, value = line.split("=", 1)
        require(key and key not in fields, f"legacy checkpoint key is duplicated or empty: {key!r}")
        fields[key] = value
    required = {
        "schema", "h11", "polytope", "frst", "status", "scale_strings",
        "terminal_scale_count", "physical_scaling_gate_status",
        "physical_control_gate_status", "summary_report_sha256",
        "branch_shard_sha256", "scale_calculation_started", "inflation_evaluated",
        "orientifold_computed", "geometry_generated", "database_written",
    }
    require(required.issubset(fields), "legacy checkpoint is missing required fields")
    require(fields["schema"] == LEGACY_SCHEMA, "legacy checkpoint schema mismatch")
    require((fields["h11"], fields["polytope"], fields["frst"]) == ("5", "1", "1"),
            "only the fixed geometry-1 legacy checkpoint may be adopted")
    require(fields["status"] == "completed", "legacy checkpoint is not terminal")
    require(fields["scale_strings"] == ";".join(SCALES), "legacy scale grid mismatch")
    require(fields["terminal_scale_count"] == "7", "legacy terminal scale count mismatch")
    require(fields["physical_scaling_gate_status"] == "passed", "legacy scaling gate is not passed")
    require(fields["physical_control_gate_status"] == "not_established", "legacy control gate changed")
    for key in ("orientifold_computed", "geometry_generated", "database_written"):
        require(fields[key] == "false", f"legacy prohibition was violated: {key}")
    require(fields["scale_calculation_started"] == "true", "legacy scale rows were not calculated")
    require(fields["inflation_evaluated"] == "true", "legacy scale rows were not evaluated")
    for key in ("summary_report_sha256", "branch_shard_sha256"):
        require(len(fields[key]) == 64 and all(c in "0123456789abcdef" for c in fields[key]),
                f"legacy {key} is not a SHA-256 digest")
    return fields


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    require(path.is_file(), f"CSV output is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, f"CSV output has no rows: {path}")
    return rows


def row_digest(row: dict[str, str]) -> str:
    return sha256_bytes(canonical(row))


def validate_rows(
    rows: list[dict[str, str]],
    *,
    row_type: str,
    h11: str,
    polytope: str,
    frst: str,
) -> None:
    require(all(row.get("row_type") == row_type for row in rows), f"unexpected row type in {row_type} output")
    require(all((row.get("h11"), row.get("polytope"), row.get("frst")) == (h11, polytope, frst)
                for row in rows), "CSV geometry identity mismatch")
    require(all(row.get("physical_scaling_gate_status") == "passed" for row in rows),
            "CSV scaling gate status is not passed")
    require(all(row.get("physical_control_gate_status") == "not_established" for row in rows),
            "CSV control gate status changed")
    require(all(row.get("branch_coverage_status") == COVERAGE for row in rows),
            "CSV coverage is not partial_index_range")


def validate_legacy_bundle(root: pathlib.Path, expected_geometry_sha256: str) -> dict[str, Any]:
    checkpoint = root / "checkpoints" / "h11_005_np_0000001_cy_0000001.checkpoint"
    summary = root / "summary.csv"
    shard = root / "shards" / "inflation_scale_continuation_shard_0001_of_0001.csv"
    legacy = parse_legacy_checkpoint(checkpoint)
    require(sha256_path(summary) == legacy["summary_report_sha256"], "legacy summary hash mismatch")
    require(sha256_path(shard) == legacy["branch_shard_sha256"], "legacy shard hash mismatch")
    summary_rows = read_csv(summary)
    shard_rows = read_csv(shard)
    require(len(summary_rows) == 7, "legacy summary must contain 7 scale rows")
    require([row.get("sampled_scale") for row in summary_rows] == list(SCALES),
            "legacy summary scale order mismatch")
    validate_rows(summary_rows, row_type="scale", h11=str(expected_h11),
                  polytope=str(expected_polytope), frst=str(expected_frst))
    require({row.get("sampled_scale") for row in shard_rows} == set(SCALES),
            "legacy shard scale coverage mismatch")
    validate_rows(shard_rows, row_type="branch", h11=str(expected_h11),
                  polytope=str(expected_polytope), frst=str(expected_frst))
    require(expected_geometry_sha256, "geometry SHA-256 is required")
    resources = {
        "summary_bytes": summary.stat().st_size,
        "shard_bytes": shard.stat().st_size,
        "summary_allocated_bytes_max": max(int(row.get("allocated_bytes") or 0) for row in summary_rows),
        "summary_output_bytes_max": max(int(row.get("output_bytes") or 0) for row in summary_rows),
        "summary_estimated_stage_allocated_bytes_max": max(
            int(row.get("estimated_stage_allocated_bytes") or 0) for row in summary_rows
        ),
        "summary_max_stage_allocated_bytes": max(
            int(row.get("max_stage_allocated_bytes") or 0) for row in summary_rows
        ),
    }
    return {
        "legacy_checkpoint_sha256": sha256_path(checkpoint),
        "summary_path": str(summary),
        "summary_sha256": sha256_path(summary),
        "summary_rows": summary_rows,
        "summary_row_digests": [row_digest(row) for row in summary_rows],
        "shard_path": str(shard),
        "shard_sha256": sha256_path(shard),
        "shard_rows": len(shard_rows),
        "resources": resources,
    }


def validate_legacy_canonical_checkpoint(
    checkpoint_path: pathlib.Path,
    summary_path: pathlib.Path,
    shard_path: pathlib.Path,
    *,
    expected_geometry_sha256: str,
    expected_h11: int,
    expected_polytope: int,
    expected_frst: int,
) -> dict[str, Any]:
    """Validate a v5 checkpoint only as an immutable migration source.

    This mode does not accept the old identity as a current run identity. It
    proves that the existing summary and shard match the old checkpoint and
    that both outputs are terminal before the current policy rebinds them.
    """
    require(checkpoint_path.is_file(), f"legacy canonical checkpoint is missing: {checkpoint_path}")
    checksum_path = pathlib.Path(str(checkpoint_path) + ".sha256")
    require(checksum_path.is_file(), "legacy canonical checkpoint checksum is missing")
    checksum_fields = checksum_path.read_text(encoding="utf-8").split()
    require(len(checksum_fields) == 2 and checksum_fields[1] == checkpoint_path.name,
            "legacy canonical checkpoint checksum line is malformed")
    checkpoint_sha256 = sha256_path(checkpoint_path)
    require(checksum_fields[0] == checkpoint_sha256,
            "legacy canonical checkpoint checksum mismatch")
    obj = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    require(obj.get("schema") == "physical-scale-inflation-geometry-checkpoint-v5" and
            obj.get("checkpoint_version") == 5,
            "legacy canonical checkpoint schema mismatch")
    inp = obj.get("input", {})
    require((inp.get("h11"), inp.get("polytope"), inp.get("frst")) ==
            (expected_h11, expected_polytope, expected_frst),
            "legacy canonical checkpoint geometry identity mismatch")
    geometry_index = expected_h11 * 1_000_000 + expected_polytope * 1_000 + expected_frst
    require(obj.get("geometry_index") == geometry_index,
            "legacy canonical checkpoint geometry index mismatch")
    require(inp.get("artifact_sha256") == expected_geometry_sha256 and
            obj.get("geometry_artifact_sha256") == expected_geometry_sha256,
            "legacy canonical checkpoint artifact identity mismatch")
    require(obj.get("terminal_status") == "completed", "legacy canonical checkpoint is not terminal")
    require(obj.get("scale_strings") == list(SCALES) and obj.get("max_negative_modes") == 1,
            "legacy canonical checkpoint scale policy mismatch")
    require(obj.get("coverage_label") == COVERAGE,
            "legacy canonical checkpoint coverage mismatch")
    counts = obj.get("terminal_counts", {})
    require(counts.get("terminal_geometry_count") == 1 and
            counts.get("terminal_geometry_indices") == [geometry_index] and
            counts.get("terminal_scale_point_count") == 7,
            "legacy canonical checkpoint terminal count mismatch")
    summary = obj.get("summary", {})
    shards = obj.get("shards", {})
    files = shards.get("files", [])
    require(len(files) == 1, "legacy canonical checkpoint shard inventory mismatch")
    shard = files[0]
    require(summary.get("path") == str(summary_path) and
            shard.get("path") == str(shard_path),
            "legacy canonical checkpoint output path mismatch")
    require(summary.get("sha256") == sha256_path(summary_path) and
            shard.get("sha256") == sha256_path(shard_path),
            "legacy canonical checkpoint output hash mismatch")
    summary_rows = read_csv(summary_path)
    shard_rows = read_csv(shard_path)
    require(len(summary_rows) == 7 and
            [row.get("sampled_scale") for row in summary_rows] == list(SCALES),
            "legacy canonical checkpoint summary rows are incomplete")
    validate_rows(summary_rows, row_type="scale", h11=str(expected_h11),
                  polytope=str(expected_polytope), frst=str(expected_frst))
    require({row.get("sampled_scale") for row in shard_rows} == set(SCALES),
            "legacy canonical checkpoint shard scales are incomplete")
    validate_rows(shard_rows, row_type="branch", h11=str(expected_h11),
                  polytope=str(expected_polytope), frst=str(expected_frst))
    require(summary.get("row_count") == len(summary_rows) and
            shard.get("row_count") == len(shard_rows),
            "legacy canonical checkpoint row counts do not reproduce")
    summary_row_digests = summary.get("row_digests", [])
    if not summary_row_digests:
        summary_row_digests = [row_digest(row) for row in summary_rows]
    require(summary_row_digests == [row_digest(row) for row in summary_rows],
            "legacy canonical checkpoint summary row digests do not reproduce")
    require(obj.get("prohibitions", {}).get("orientifold_computed") is False and
            obj.get("prohibitions", {}).get("geometry_generated") is False and
            obj.get("prohibitions", {}).get("database_written") is False,
            "legacy canonical checkpoint contains an unauthorized action")
    resources = obj.get("resource_accounting", {})
    return {
        "legacy_checkpoint_sha256": checkpoint_sha256,
        "summary_sha256": sha256_path(summary_path),
        "shard_sha256": sha256_path(shard_path),
        "summary_rows": len(summary_rows),
        "branch_rows": len(shard_rows),
        "summary_row_digests": summary_row_digests,
        "summary_bytes": summary_path.stat().st_size,
        "shard_bytes": shard_path.stat().st_size,
        "max_allocated_bytes": int(resources.get("evaluator_allocated_bytes", 0)),
        "max_output_bytes": int(resources.get("evaluator_output_bytes", 0)),
        "max_estimated_stage_bytes": int(resources.get("summary_estimated_stage_allocated_bytes_max", 0)),
        "max_stage_bytes": int(resources.get("summary_max_stage_allocated_bytes", 0)),
        "summary_path": str(summary_path),
        "shard_path": str(shard_path),
    }


def expected_prohibitions() -> dict[str, bool]:
    return {
        "orientifold_computed": False,
        "geometry_generated": False,
        "population_expanded": False,
        "replacement_or_silent_skip": False,
        "database_written": False,
        "dependency_specification_changed": False,
        "commit_created": False,
        "exhaustive_coverage_claim": False,
        "physical_viability_claim": False,
        "production_claim": False,
        "validated_candidate_claim": False,
    }


def validate_execution_source_manifest(path: pathlib.Path, expected_sha256: str = "") -> dict[str, Any]:
    require(path.is_file(), "execution-source manifest is missing")
    actual_sha256 = sha256_path(path)
    if expected_sha256:
        require(actual_sha256 == expected_sha256, "execution-source manifest hash mismatch")
    source_manifest = json.loads(path.read_text(encoding="utf-8"))
    require(source_manifest.get("schema") == "physical-scale-inflation-execution-source-manifest-v2" and
            source_manifest.get("manifest_version") == 2,
            "execution-source manifest schema mismatch")
    source_entries = source_manifest.get("source_file_hashes")
    require(isinstance(source_entries, dict) and source_entries,
            "execution-source manifest has no source hashes")
    require(set(source_entries) == set(EXECUTION_SOURCE_FILES),
            "execution-source manifest source inventory is incomplete or unexpected")
    require(source_manifest.get("source_file_inventory") == list(EXECUTION_SOURCE_FILES),
            "execution-source manifest inventory is not canonical")
    worktree = pathlib.Path(source_manifest.get("worktree", ""))
    require(worktree.is_dir(), "execution-source manifest worktree is missing")
    for relative, source_hash in source_entries.items():
        source_path = worktree / relative
        require(source_path.is_file() and sha256_path(source_path) == source_hash,
                f"live execution source hash mismatch: {relative}")
    try:
        live_diff = subprocess.check_output(
            ["git", "-C", str(worktree), "diff", "--binary", "HEAD"]
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot read complete Git diff: {error}") from error
    require(sha256_bytes(live_diff) == source_manifest.get("complete_git_diff_sha256"),
            "live complete Git diff hash mismatch")
    payload = dict(source_manifest)
    payload_hash = payload.pop("payload_sha256", None)
    require(payload_hash == sha256_bytes(canonical(payload)),
            "execution-source manifest payload identity mismatch")
    return source_manifest


def validate_identity_manifest(
    path: pathlib.Path,
    *,
    expected_schema: str,
    expected_policy: str,
    expected_common: str,
    expected_selection: str,
    expected_diff: str,
    expected_project: str,
    expected_manifest: str,
    expected_source_manifest: str,
    output_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    require(path.is_file(), f"identity manifest is missing: {path}")
    checksum_path = pathlib.Path(str(path) + ".sha256")
    require(checksum_path.is_file(), "identity manifest checksum is missing")
    checksum_fields = checksum_path.read_text(encoding="utf-8").split()
    require(len(checksum_fields) == 2 and checksum_fields[1] == path.name,
            "identity manifest checksum line is malformed")
    actual = sha256_path(path)
    require(checksum_fields[0] == actual, "identity manifest checksum mismatch")
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(obj.get("schema") == expected_schema, "identity manifest schema mismatch")
    for key, value in (
        ("policy_sha256", expected_policy),
        ("common_configuration_digest_sha256", expected_common),
        ("selection_manifest_sha256", expected_selection),
        ("current_complete_git_diff_sha256", expected_diff),
        ("project_sha256", expected_project),
        ("manifest_sha256", expected_manifest),
        ("execution_source_manifest_sha256", expected_source_manifest),
    ):
        require(obj.get(key) == value, f"identity manifest mismatch: {key}")
    require(obj.get("domain_certificate_version") == DOMAIN_CERTIFICATE_VERSION,
            "identity manifest domain certificate version mismatch")
    require(obj.get("conversion_policy_version") == CONVERSION_POLICY_VERSION,
            "identity manifest conversion policy version mismatch")
    require(obj.get("kinv_conversion_acceptance") == KINV_CONVERSION_RULE,
            "identity manifest Kinv conversion rule mismatch")
    source_path = pathlib.Path(obj.get("execution_source_manifest_path", ""))
    source = validate_execution_source_manifest(source_path, expected_source_manifest)
    require(obj.get("execution_source_hashes") == source.get("source_file_hashes"),
            "identity manifest execution-source map mismatch")
    quarantine_status = obj.get("quarantine_manifest_status", "none")
    require(quarantine_status in ("none", "quarantined"),
            "identity manifest quarantine status is invalid")
    quarantine_path_value = obj.get("quarantine_manifest_path")
    quarantine_sha_value = obj.get("quarantine_manifest_sha256")
    if quarantine_status == "quarantined":
        require(output_root is not None and quarantine_path_value and quarantine_sha_value,
                "identity manifest quarantine binding is incomplete")
        quarantine_path = pathlib.Path(quarantine_path_value)
        require(quarantine_path.is_file() and sha256_path(quarantine_path) == quarantine_sha_value,
                "identity manifest quarantine manifest hash mismatch")
        quarantine = validate_quarantine_manifest(
            quarantine_path,
            output_root=output_root,
            expected_policy=expected_policy,
            expected_common=expected_common,
            expected_selection=expected_selection,
            expected_diff=expected_diff,
            expected_source_manifest=expected_source_manifest,
        )
        require(obj.get("quarantine_geometry_index") == quarantine.get("geometry_index"),
                "identity manifest quarantine geometry mismatch")
    else:
        require(quarantine_path_value is None and quarantine_sha_value is None and
                obj.get("quarantine_geometry_index") is None,
                "identity manifest has quarantine fields with status none")
    require(obj.get("scale_strings") == list(SCALES), "identity manifest scale grid mismatch")
    require(obj.get("max_negative_modes") == 1, "identity manifest max-negative-modes mismatch")
    require(obj.get("coverage_label") == COVERAGE, "identity manifest coverage mismatch")
    require(obj.get("prohibitions") == expected_prohibitions(),
            "identity manifest prohibitions are incomplete or violated")
    if output_root is not None and "output_bytes_under_root" in obj:
        require(output_tree_bytes(output_root) <= 2_000_000_000,
                "identity manifest output tree exceeds the cap")
    if "preexisting_output_bytes_under_root" in obj:
        require(isinstance(obj["preexisting_output_bytes_under_root"], int) and
                obj["preexisting_output_bytes_under_root"] >= 0,
                "identity manifest pre-existing output count is invalid")
    if "new_output_bytes" in obj:
        require(isinstance(obj["new_output_bytes"], int) and obj["new_output_bytes"] >= 0,
                "identity manifest new output count is invalid")
    if "preexisting_output_bytes_under_root" in obj and "new_output_bytes" in obj and \
            "output_bytes_under_root" in obj:
        require(obj["output_bytes_under_root"] >=
                obj["preexisting_output_bytes_under_root"] + obj["new_output_bytes"],
                "identity manifest output counters are not monotonic")
    return obj


def validate_resource_evidence_root(output_root: pathlib.Path) -> dict[str, Any]:
    """Recover the measured calculation peak from checksum-bound checkpoints.

    Provenance migration can replace the active checkpoint with an adoption
    record.  The original calculation checkpoint remains beside it under a
    checksum-bound ``.superseded-<digest>.json`` name.  Only records emitted by
    the calculation path (not adoption records) qualify as peak-RSS evidence.
    """
    checkpoint_root = output_root / "checkpoints"
    require(checkpoint_root.is_dir(), "resource-evidence checkpoint directory is missing")
    candidates: list[dict[str, Any]] = []
    paths = sorted(checkpoint_root.glob("*.checkpoint-v6.json"))
    paths += sorted(checkpoint_root.glob("*.checkpoint-v6.json.superseded-*.json"))
    for path in paths:
        checksum_path = pathlib.Path(str(path) + ".sha256")
        if not checksum_path.is_file():
            continue
        fields = checksum_path.read_text(encoding="utf-8").split()
        require(len(fields) == 2 and fields[1] == path.name,
                "resource-evidence checksum line is malformed")
        digest = sha256_path(path)
        require(fields[0] == digest, "resource-evidence checkpoint checksum mismatch")
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("schema") != CANONICAL_SCHEMA or obj.get("terminal_status") != "completed":
            continue
        resources = obj.get("resource_accounting")
        if not isinstance(resources, dict) or "evaluator_allocated_bytes" not in resources:
            continue
        maxrss = resources.get("maxrss_bytes")
        rss_cap = resources.get("rss_cap_bytes")
        output_cap = resources.get("output_cap_bytes")
        require(isinstance(maxrss, int) and maxrss >= 0,
                "resource-evidence max RSS is invalid")
        require(rss_cap == 2_000_000_000 and output_cap == 2_000_000_000,
                "resource-evidence caps are invalid")
        require(maxrss <= rss_cap, "resource-evidence RSS exceeds the cap")
        for key in ("evaluator_allocated_bytes", "evaluator_output_bytes",
                    "new_output_bytes", "output_bytes_under_root_at_checkpoint"):
            require(isinstance(resources.get(key), int) and resources[key] >= 0,
                    f"resource-evidence field is invalid: {key}")
        require(resources["new_output_bytes"] <= output_cap,
                "resource-evidence new output exceeds the cap")
        require(resources["output_bytes_under_root_at_checkpoint"] <= output_cap,
                "resource-evidence output tree exceeds the cap")
        candidates.append({
            "maxrss_bytes": maxrss,
            "geometry_index": int(obj["geometry_index"]),
            "source_checkpoint_path": str(path),
            "source_checkpoint_sha256": digest,
            "rss_cap_bytes": rss_cap,
            "output_cap_bytes": output_cap,
        })
    require(candidates, "no checksum-bound calculation resource evidence was found")
    peak = max(candidates, key=lambda item: (item["maxrss_bytes"], item["geometry_index"]))
    peak["calculation_checkpoint_count"] = len(candidates)
    return peak


def validate_canonical_checkpoint(
    path: pathlib.Path,
    *,
    expected: dict[str, str],
    summary_path: pathlib.Path,
    shard_path: pathlib.Path,
    output_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    require(path.is_file(), f"canonical checkpoint is missing: {path}")
    checksum_path = pathlib.Path(str(path) + ".sha256")
    require(checksum_path.is_file(), "canonical checkpoint checksum is missing")
    checksum_fields = checksum_path.read_text(encoding="utf-8").split()
    require(len(checksum_fields) == 2 and checksum_fields[1] == path.name, "canonical checksum line is malformed")
    require(checksum_fields[0] == sha256_path(path), "canonical checkpoint checksum mismatch")
    obj = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema", "checkpoint_version", "run_id", "geometry_index", "input",
        "geometry_artifact_sha256", "geometry_sidecar_sha256", "policy_sha256",
        "common_configuration_digest_sha256", "selection_manifest_sha256",
        "continuation_source_sha256", "current_complete_git_diff_sha256",
        "project_sha256", "manifest_sha256", "julia_executable", "julia_version",
        "execution_source_manifest_path", "execution_source_manifest_sha256", "execution_source_hashes",
        "domain_certificate_version", "conversion_policy_version", "kinv_conversion_acceptance",
        "run_configuration", "run_configuration_digest_sha256", "summary", "shards",
        "resource_accounting", "terminal_status", "terminal_counts", "gates", "coverage_label",
        "scale_strings", "max_negative_modes", "prohibitions", "scale_records",
    }
    require(required.issubset(obj), "canonical checkpoint is missing a required identity field")
    require(obj["schema"] == CANONICAL_SCHEMA and obj["checkpoint_version"] == 6,
            "canonical checkpoint schema mismatch")
    for key in (
        "geometry_artifact_sha256", "geometry_sidecar_sha256", "policy_sha256",
        "common_configuration_digest_sha256", "selection_manifest_sha256",
        "continuation_source_sha256", "current_complete_git_diff_sha256",
        "project_sha256", "manifest_sha256", "run_configuration_digest_sha256",
    ):
        if key in expected:
            require(obj[key] == expected[key], f"canonical checkpoint identity mismatch: {key}")
    source_manifest_path = pathlib.Path(obj.get("execution_source_manifest_path", ""))
    source_manifest_sha = obj.get("execution_source_manifest_sha256", "")
    require(len(source_manifest_sha) == 64, "execution-source manifest is missing")
    if "execution_source_manifest_sha256" in expected:
        require(source_manifest_sha == expected["execution_source_manifest_sha256"],
                "execution-source manifest identity mismatch")
    source_manifest = validate_execution_source_manifest(source_manifest_path, source_manifest_sha)
    require(obj["execution_source_hashes"] == source_manifest["source_file_hashes"],
            "checkpoint execution-source hash map mismatch")
    require(obj["current_complete_git_diff_sha256"] == source_manifest.get("complete_git_diff_sha256"),
            "checkpoint complete Git diff binding mismatch")
    require(obj["julia_version"] == "1.12.6" and obj["julia_executable"], "Julia identity is incomplete")
    require(obj["domain_certificate_version"] == DOMAIN_CERTIFICATE_VERSION,
            "checkpoint domain certificate version mismatch")
    require(obj["conversion_policy_version"] == CONVERSION_POLICY_VERSION,
            "checkpoint conversion policy version mismatch")
    require(obj["kinv_conversion_acceptance"] == KINV_CONVERSION_RULE,
            "checkpoint Kinv conversion rule mismatch")
    inp = obj["input"]
    geometry_fields = ("h11", "polytope", "frst")
    for field in geometry_fields:
        require(field in inp, f"canonical checkpoint input is missing {field}")
        expected_key = f"geometry_{field}"
        require(expected_key in expected, f"expected {expected_key} is required")
        require(int(inp[field]) == int(expected[expected_key]),
                f"canonical checkpoint geometry identity mismatch: {field}")
    h11, polytope, frst = (int(inp[field]) for field in geometry_fields)
    require(5 <= h11 <= 10 and 1 <= polytope <= 3 and frst == 1,
            "canonical checkpoint geometry is outside the fixed 18")
    geometry_index = h11 * 1_000_000 + polytope * 1_000 + frst
    require(obj["geometry_index"] == geometry_index,
            "canonical checkpoint geometry index mismatch")
    require(int(expected["geometry_index"]) == geometry_index,
            "expected geometry index mismatch")
    require(inp.get("artifact_sha256") == obj["geometry_artifact_sha256"], "input artifact binding mismatch")
    require(inp.get("orientifold_requested") is False, "canonical checkpoint requested an orientifold")
    config = obj["run_configuration"]
    require(sha256_bytes(canonical(config)) == obj["run_configuration_digest_sha256"],
            "run configuration digest does not reproduce")
    config_identity = {
        "policy_sha256": "policy_sha256",
        "common_configuration_digest_sha256": "common_configuration_digest_sha256",
        "selection_manifest_sha256": "selection_manifest_sha256",
        "continuation_source_sha256": "continuation_source_sha256",
        "current_complete_git_diff_sha256": "current_complete_git_diff_sha256",
        "execution_source_manifest_sha256": "execution_source_manifest_sha256",
    }
    for config_key, checkpoint_key in config_identity.items():
        require(config.get(config_key) == obj[checkpoint_key],
                f"run configuration identity mismatch: {config_key}")
    require(config.get("execution_source_hashes") == obj["execution_source_hashes"],
            "run configuration execution-source map mismatch")
    fixed_input = config.get("fixed_input", {})
    for field in ("relative_path", "artifact_sha256", "h11", "polytope", "frst",
                  "polytope_id", "triangulation_id", "orientifold_requested"):
        if field in inp:
            require(fixed_input.get(field) == inp[field],
                    f"run configuration fixed-input mismatch: {field}")
    require(config.get("scale_strings") == list(SCALES), "run configuration scales mismatch")
    require(config.get("max_negative_modes") == 1, "run configuration max-negative-modes mismatch")
    require(config.get("domain_certificate_version") == DOMAIN_CERTIFICATE_VERSION,
            "run configuration domain certificate version mismatch")
    require(config.get("conversion_policy_version") == CONVERSION_POLICY_VERSION,
            "run configuration conversion policy version mismatch")
    require(config.get("kinv_conversion_acceptance") == KINV_CONVERSION_RULE,
            "run configuration Kinv conversion rule mismatch")
    require(obj["scale_strings"] == list(SCALES), "checkpoint scale grid mismatch")
    require(obj["max_negative_modes"] == 1, "checkpoint max-negative-modes mismatch")
    require(obj["terminal_status"] == "completed", "canonical checkpoint is not terminal")
    require(obj["coverage_label"] == COVERAGE, "checkpoint coverage is not partial_index_range")
    gates = obj["gates"]
    require(gates.get("physical_scaling_gate") == "passed", "checkpoint scaling gate is not passed")
    require(gates.get("physical_control_gate") == "not_established", "checkpoint control gate changed")
    require(gates.get("physical_viability_claim") is False, "checkpoint made an unauthorized claim")
    require(obj["prohibitions"] == expected_prohibitions(), "checkpoint prohibitions are incomplete or violated")
    scale_records = obj["scale_records"]
    require(isinstance(scale_records, list) and len(scale_records) == len(SCALES),
            "checkpoint scale record count mismatch")
    for record, scale in zip(scale_records, SCALES):
        require(record.get("scale_string") == scale,
                "checkpoint scale record identity mismatch")
        require(record.get("physical_scaling_gate_status") == "passed",
                "checkpoint scale record scaling gate is not passed")
        require(record.get("physical_control_gate_status") == "not_established",
                "checkpoint scale record control gate changed")
        require(record.get("coverage_label") == COVERAGE,
                "checkpoint scale record coverage changed")
    summary = obj["summary"]
    require(summary.get("path") == str(summary_path), "checkpoint summary path mismatch")
    require(summary.get("sha256") == sha256_path(summary_path), "checkpoint summary hash mismatch")
    require(summary.get("bytes") == summary_path.stat().st_size, "checkpoint summary byte count mismatch")
    require(summary.get("row_count") == 7, "checkpoint summary row count mismatch")
    summary_rows = read_csv(summary_path)
    require(len(summary_rows) == 7 and [row.get("sampled_scale") for row in summary_rows] == list(SCALES),
            "checkpoint summary scale rows are incomplete")
    validate_rows(summary_rows, row_type="scale", h11=str(h11),
                  polytope=str(polytope), frst=str(frst))
    require(summary.get("row_digests") in ([], [row_digest(row) for row in summary_rows]),
            "checkpoint summary row digest mismatch")
    shards = obj["shards"]
    files = shards.get("files")
    require(isinstance(files, list) and len(files) == 1, "checkpoint shard inventory mismatch")
    shard = files[0]
    require(shard.get("path") == str(shard_path), "checkpoint shard path mismatch")
    require(shard.get("sha256") == sha256_path(shard_path), "checkpoint shard hash mismatch")
    require(shards.get("sha256") == shard["sha256"], "checkpoint shard aggregate hash mismatch")
    require(shard.get("bytes") == shard_path.stat().st_size, "checkpoint shard byte count mismatch")
    shard_rows = read_csv(shard_path)
    require({row.get("sampled_scale") for row in shard_rows} == set(SCALES),
            "checkpoint shard scale coverage is incomplete")
    validate_rows(shard_rows, row_type="branch", h11=str(h11),
                  polytope=str(polytope), frst=str(frst))
    require(shard.get("row_count") == len(shard_rows), "checkpoint shard row count mismatch")
    counts = obj["terminal_counts"]
    require(counts.get("terminal_geometry_count") == 1 and counts.get("terminal_scale_point_count") == 7,
            "checkpoint terminal geometry/scale count mismatch")
    require(counts.get("terminal_geometry_indices") == [geometry_index],
            "checkpoint terminal geometry index mismatch")
    require(counts.get("summary_row_count") == 7 and counts.get("branch_row_count") == shard.get("row_count"),
            "checkpoint terminal output count mismatch")
    resources = obj["resource_accounting"]
    for key in (
        "maxrss_bytes", "new_output_bytes", "summary_bytes", "shard_bytes", "checkpoint_bytes",
        "preexisting_output_bytes_under_root", "output_bytes_under_root_before_checkpoint",
        "output_bytes_under_root_at_checkpoint", "output_cap_bytes",
    ):
        require(isinstance(resources.get(key), int) and resources[key] >= 0, f"resource counter is invalid: {key}")
    require(resources["summary_bytes"] == summary_path.stat().st_size, "summary resource count mismatch")
    require(resources["shard_bytes"] == shard_path.stat().st_size, "shard resource count mismatch")
    require(resources["output_bytes_under_root_at_checkpoint"] >=
            resources["output_bytes_under_root_before_checkpoint"],
            "output-tree accounting is not monotonic")
    require(resources["output_bytes_under_root_at_checkpoint"] <= resources["output_cap_bytes"],
            "output-tree accounting exceeds the cap")
    if output_root is not None:
        require(output_tree_bytes(output_root) <= resources["output_cap_bytes"],
                "current output tree exceeds the cap")
    return obj


def validate_canonical_migration_source(
    path: pathlib.Path,
    summary_path: pathlib.Path,
    shard_path: pathlib.Path,
    *,
    expected_geometry_sha256: str,
    expected_h11: int,
    expected_polytope: int,
    expected_frst: int,
) -> dict[str, Any]:
    """Validate a terminal v6 bundle as an immutable migration source.

    Migration must prove the old checkpoint, its geometry identity, and its
    per-geometry outputs before rebinding current policy/source identities.
    The old execution-source map is retained as provenance, but is not used
    as the live source of the current run.
    """
    require(path.is_file(), f"migration checkpoint is missing: {path}")
    checksum_path = pathlib.Path(str(path) + ".sha256")
    require(checksum_path.is_file(), "migration checkpoint checksum is missing")
    checksum_fields = checksum_path.read_text(encoding="utf-8").split()
    require(len(checksum_fields) == 2 and checksum_fields[1] == path.name,
            "migration checkpoint checksum line is malformed")
    checkpoint_sha256 = sha256_path(path)
    require(checksum_fields[0] == checkpoint_sha256,
            "migration checkpoint checksum mismatch")
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(obj.get("schema") == CANONICAL_SCHEMA and obj.get("checkpoint_version") == 6,
            "migration checkpoint schema mismatch")
    inp = obj.get("input", {})
    require((inp.get("h11"), inp.get("polytope"), inp.get("frst")) ==
            (expected_h11, expected_polytope, expected_frst),
            "migration checkpoint geometry identity mismatch")
    geometry_index = expected_h11 * 1_000_000 + expected_polytope * 1_000 + expected_frst
    require(obj.get("geometry_index") == geometry_index,
            "migration checkpoint geometry index mismatch")
    require(inp.get("artifact_sha256") == expected_geometry_sha256 and
            obj.get("geometry_artifact_sha256") == expected_geometry_sha256,
            "migration checkpoint artifact identity mismatch")
    require(obj.get("terminal_status") == "completed",
            "migration checkpoint is not terminal")
    require(obj.get("scale_strings") == list(SCALES) and
            obj.get("max_negative_modes") == 1 and obj.get("coverage_label") == COVERAGE,
            "migration checkpoint scale policy mismatch")
    gates = obj.get("gates", {})
    require(gates.get("physical_scaling_gate") == "passed" and
            gates.get("physical_control_gate") == "not_established" and
            gates.get("physical_viability_claim") is False,
            "migration checkpoint gate/claim identity mismatch")
    require(obj.get("prohibitions") == expected_prohibitions(),
            "migration checkpoint prohibitions are incomplete or violated")
    for key in (
        "policy_sha256", "common_configuration_digest_sha256",
        "selection_manifest_sha256", "continuation_source_sha256",
        "current_complete_git_diff_sha256", "project_sha256", "manifest_sha256",
        "execution_source_manifest_sha256",
    ):
        value = obj.get(key)
        require(isinstance(value, str) and len(value) == 64 and
                all(char in "0123456789abcdef" for char in value),
                f"migration checkpoint identity is malformed: {key}")
    require(isinstance(obj.get("execution_source_hashes"), dict) and
            obj["execution_source_hashes"],
            "migration checkpoint execution-source provenance is missing")
    config = obj.get("run_configuration", {})
    require(sha256_bytes(canonical(config)) == obj.get("run_configuration_digest_sha256"),
            "migration run configuration digest does not reproduce")
    require(config.get("scale_strings") == list(SCALES) and
            config.get("max_negative_modes") == 1 and
            config.get("fixed_input", {}).get("h11") == expected_h11 and
            config.get("fixed_input", {}).get("polytope") == expected_polytope and
            config.get("fixed_input", {}).get("frst") == expected_frst,
            "migration run configuration geometry/policy mismatch")
    scale_records = obj.get("scale_records")
    require(isinstance(scale_records, list) and len(scale_records) == len(SCALES),
            "migration checkpoint scale record count mismatch")
    for record, scale in zip(scale_records, SCALES):
        summary_record = record.get("summary", {})
        coverage_ok = record.get("coverage_label") == COVERAGE or (
            summary_record.get("coverage_status") == COVERAGE and
            summary_record.get("branch_coverage_status") == COVERAGE
        )
        require(record.get("scale_string") == scale and
                record.get("physical_scaling_gate_status") == "passed" and
                record.get("physical_control_gate_status") == "not_established" and
                record.get("physical_viability_status") == "not_established" and
                coverage_ok,
                "migration checkpoint scale record identity mismatch")
    summary = obj.get("summary", {})
    shards = obj.get("shards", {})
    files = shards.get("files", [])
    require(summary.get("path") == str(summary_path) and len(files) == 1 and
            files[0].get("path") == str(shard_path),
            "migration checkpoint output path mismatch")
    require(summary.get("sha256") == sha256_path(summary_path) and
            files[0].get("sha256") == sha256_path(shard_path) and
            shards.get("sha256") == files[0].get("sha256"),
            "migration checkpoint output hash mismatch")
    summary_rows = read_csv(summary_path)
    shard_rows = read_csv(shard_path)
    require(len(summary_rows) == len(SCALES) and
            [row.get("sampled_scale") for row in summary_rows] == list(SCALES),
            "migration checkpoint summary rows are incomplete")
    validate_rows(summary_rows, row_type="scale", h11=str(expected_h11),
                  polytope=str(expected_polytope), frst=str(expected_frst))
    require({row.get("sampled_scale") for row in shard_rows} == set(SCALES),
            "migration checkpoint shard scales are incomplete")
    validate_rows(shard_rows, row_type="branch", h11=str(expected_h11),
                  polytope=str(expected_polytope), frst=str(expected_frst))
    require(summary.get("bytes") == summary_path.stat().st_size and
            summary.get("row_count") == len(summary_rows) and
            files[0].get("bytes") == shard_path.stat().st_size and
            files[0].get("row_count") == len(shard_rows),
            "migration checkpoint output accounting mismatch")
    summary_row_digests = summary.get("row_digests", [])
    require(summary_row_digests in ([], [row_digest(row) for row in summary_rows]),
            "migration checkpoint summary row digests do not reproduce")
    resources = obj.get("resource_accounting", {})
    return {
        "legacy_checkpoint_sha256": checkpoint_sha256,
        "summary_sha256": sha256_path(summary_path),
        "shard_sha256": sha256_path(shard_path),
        "summary_rows": len(summary_rows),
        "branch_rows": len(shard_rows),
        "summary_row_digests": [row_digest(row) for row in summary_rows],
        "summary_bytes": summary_path.stat().st_size,
        "shard_bytes": shard_path.stat().st_size,
        "max_allocated_bytes": int(resources.get("evaluator_allocated_bytes", 0)),
        "max_output_bytes": int(resources.get("evaluator_output_bytes", 0)),
        "max_estimated_stage_bytes": int(resources.get("summary_estimated_stage_allocated_bytes_max", 0)),
        "max_stage_bytes": int(resources.get("summary_max_stage_allocated_bytes", 0)),
        "summary_path": str(summary_path),
        "shard_path": str(shard_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=pathlib.Path)
    parser.add_argument("--canonical", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--shard", type=pathlib.Path)
    parser.add_argument("--geometry-sha256", default="")
    parser.add_argument("--sidecar-sha256", default="")
    parser.add_argument("--policy-sha256", default="")
    parser.add_argument("--common-digest", default="")
    parser.add_argument("--selection-sha256", default="")
    parser.add_argument("--continuation-sha256", default="")
    parser.add_argument("--diff-sha256", default="")
    parser.add_argument("--project-sha256", default="")
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--run-config-sha256", default="")
    parser.add_argument("--source-manifest-sha256", default="")
    parser.add_argument("--expected-h11", default="")
    parser.add_argument("--expected-polytope", default="")
    parser.add_argument("--expected-frst", default="")
    parser.add_argument("--expected-geometry-index", default="")
    parser.add_argument("--source-only", type=pathlib.Path)
    parser.add_argument("--quarantine-only", type=pathlib.Path)
    parser.add_argument("--quarantine-rebind", type=pathlib.Path)
    parser.add_argument("--source-manifest-path", type=pathlib.Path)
    parser.add_argument("--manifest-only", type=pathlib.Path)
    parser.add_argument("--legacy-canonical", type=pathlib.Path)
    parser.add_argument("--migration-source", type=pathlib.Path)
    parser.add_argument("--resource-evidence-root", type=pathlib.Path)
    parser.add_argument("--expected-schema", default="")
    parser.add_argument("--expected-policy", default="")
    parser.add_argument("--expected-common", default="")
    parser.add_argument("--expected-selection", default="")
    parser.add_argument("--expected-diff", default="")
    parser.add_argument("--expected-project", default="")
    parser.add_argument("--expected-manifest", default="")
    parser.add_argument("--output-root", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.resource_evidence_root is not None:
        evidence = validate_resource_evidence_root(args.resource_evidence_root)
        print("RESOURCE_EVIDENCE\t" + "\t".join((
            str(evidence["maxrss_bytes"]), str(evidence["geometry_index"]),
            evidence["source_checkpoint_path"], evidence["source_checkpoint_sha256"],
            str(evidence["calculation_checkpoint_count"]),
        )))
        return 0
    if args.source_only is not None:
        manifest = validate_execution_source_manifest(args.source_only, args.source_manifest_sha256)
        print("SOURCE\t" + sha256_path(args.source_only) + "\t" + str(len(manifest["source_file_hashes"])))
        return 0
    if args.quarantine_only is not None:
        require(args.output_root is not None, "--quarantine-only requires --output-root")
        quarantine = validate_quarantine_manifest(
            args.quarantine_only,
            output_root=args.output_root,
            expected_policy=args.expected_policy,
            expected_common=args.expected_common,
            expected_selection=args.expected_selection,
            expected_diff=args.expected_diff,
            expected_source_manifest=args.source_manifest_sha256,
            expected_geometry_index=args.expected_geometry_index,
        )
        print("QUARANTINE\t" + "\t".join((
            sha256_path(args.quarantine_only), str(quarantine["geometry_index"]),
            str(len(quarantine["files"])),
        )))
        return 0
    if args.quarantine_rebind is not None:
        require(args.output_root is not None and args.source_manifest_path is not None,
                "--quarantine-rebind requires --output-root and --source-manifest-path")
        quarantine, previous_digest = rebind_quarantine_manifest(
            args.quarantine_rebind,
            output_root=args.output_root,
            source_manifest_path=args.source_manifest_path,
            expected_policy=args.expected_policy,
            expected_common=args.expected_common,
            expected_selection=args.expected_selection,
            expected_diff=args.expected_diff,
            expected_source_manifest=args.source_manifest_sha256,
            expected_geometry_index=args.expected_geometry_index,
        )
        print("QUARANTINE_REBOUND\t" + "\t".join((
            sha256_path(args.quarantine_rebind), str(quarantine["geometry_index"]),
            str(len(quarantine["files"])), previous_digest,
        )))
        return 0
    if args.manifest_only is not None:
        manifest = validate_identity_manifest(
            args.manifest_only,
            expected_schema=args.expected_schema,
            expected_policy=args.expected_policy,
            expected_common=args.expected_common,
            expected_selection=args.expected_selection,
            expected_diff=args.expected_diff,
            expected_project=args.expected_project,
            expected_manifest=args.expected_manifest,
            expected_source_manifest=args.source_manifest_sha256,
            output_root=args.output_root,
        )
        print("MANIFEST\t" + sha256_path(args.manifest_only) + "\t" + str(manifest.get("terminal_geometry_count", 0)))
        return 0
    if args.legacy_canonical is not None:
        require(args.summary is not None and args.shard is not None,
                "--legacy-canonical requires --summary and --shard")
        require(args.expected_h11 and args.expected_polytope and args.expected_frst,
                "--legacy-canonical requires expected geometry identity")
        info = validate_legacy_canonical_checkpoint(
            args.legacy_canonical, args.summary, args.shard,
            expected_geometry_sha256=args.geometry_sha256,
            expected_h11=int(args.expected_h11),
            expected_polytope=int(args.expected_polytope),
            expected_frst=int(args.expected_frst),
        )
        print("LEGACY_CANONICAL\t" + "\t".join((
            info["legacy_checkpoint_sha256"], info["summary_sha256"],
            info["shard_sha256"], str(info["summary_rows"]),
            str(info["branch_rows"]), str(info["max_allocated_bytes"]),
            str(info["max_output_bytes"]), str(info["max_estimated_stage_bytes"]),
            str(info["max_stage_bytes"]),
            ",".join(info["summary_row_digests"]),
        )))
        return 0
    if args.migration_source is not None:
        require(args.summary is not None and args.shard is not None,
                "--migration-source requires --summary and --shard")
        require(args.expected_h11 and args.expected_polytope and args.expected_frst,
                "--migration-source requires expected geometry identity")
        info = validate_canonical_migration_source(
            args.migration_source, args.summary, args.shard,
            expected_geometry_sha256=args.geometry_sha256,
            expected_h11=int(args.expected_h11),
            expected_polytope=int(args.expected_polytope),
            expected_frst=int(args.expected_frst),
        )
        print("MIGRATION_SOURCE\t" + "\t".join((
            info["legacy_checkpoint_sha256"], info["summary_sha256"],
            info["shard_sha256"], str(info["summary_rows"]),
            str(info["branch_rows"]), str(info["max_allocated_bytes"]),
            str(info["max_output_bytes"]), str(info["max_estimated_stage_bytes"]),
            str(info["max_stage_bytes"]), ",".join(info["summary_row_digests"]),
        )))
        return 0
    if args.legacy_root is not None:
        info = validate_legacy_bundle(args.legacy_root, args.geometry_sha256)
        print("LEGACY\t" + "\t".join((
            info["legacy_checkpoint_sha256"], info["summary_sha256"], info["shard_sha256"],
            str(len(info["summary_rows"])), str(info["shard_rows"]),
            str(info["resources"]["summary_allocated_bytes_max"]),
            str(info["resources"]["summary_output_bytes_max"]),
            str(info["resources"]["summary_estimated_stage_allocated_bytes_max"]),
            str(info["resources"]["summary_max_stage_allocated_bytes"]),
            ",".join(info["summary_row_digests"]),
        )))
        return 0
    require(args.canonical is not None, "--canonical or --legacy-root is required")
    require(args.summary is not None and args.shard is not None,
            "--canonical requires --summary and --shard")
    expected = {
        "geometry_artifact_sha256": args.geometry_sha256,
        "geometry_sidecar_sha256": args.sidecar_sha256,
        "policy_sha256": args.policy_sha256,
        "common_configuration_digest_sha256": args.common_digest,
        "selection_manifest_sha256": args.selection_sha256,
        "continuation_source_sha256": args.continuation_sha256,
        "current_complete_git_diff_sha256": args.diff_sha256,
        "project_sha256": args.project_sha256,
        "manifest_sha256": args.manifest_sha256,
        "run_configuration_digest_sha256": args.run_config_sha256,
        "execution_source_manifest_sha256": args.source_manifest_sha256,
        "geometry_h11": args.expected_h11,
        "geometry_polytope": args.expected_polytope,
        "geometry_frst": args.expected_frst,
        "geometry_index": args.expected_geometry_index,
    }
    validate_canonical_checkpoint(args.canonical, expected=expected, summary_path=args.summary,
                                  shard_path=args.shard, output_root=args.output_root)
    print("CANONICAL\t" + sha256_path(args.canonical))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL_CLOSED\t{error}", file=sys.stderr)
        raise SystemExit(1)
