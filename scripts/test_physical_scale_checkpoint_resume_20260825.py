#!/usr/bin/env python3
"""Run fail-closed mutation and zero-write resume tests."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

from validate_physical_scale_checkpoint_20260825 import (
    CANONICAL_SCHEMA,
    canonical,
    sha256_path,
    validate_canonical_checkpoint,
    validate_quarantine_manifest,
    validate_resource_evidence_root,
)


WORKTREE = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ROOT = pathlib.Path("/private/tmp/cyax-inflation-physical-scale-pilot-20260825/first-geometry")
CHECKPOINT = SOURCE_ROOT / "checkpoints" / "h11_005_np_0000001_cy_0000001.checkpoint-v6.json"
CHECKSUM = pathlib.Path(str(CHECKPOINT) + ".sha256")
QUARANTINE_MANIFEST = SOURCE_ROOT / "quarantine" / "nonterminal_partial_quarantine_manifest-v1.json"
JULIA = pathlib.Path("/Users/vmehta/.juliaup/bin/julia")
PROJECT = pathlib.Path("/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl")
DATA_ROOT = "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data"
EXPECTED_GEOMETRY11_EFFECTIVE_HASH = \
    "9f0c830901a6079c945cb5215b6cdcac55b4d249e1b9398e4b7cd6adad0913ae"


def checkpoint_object(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_from_checkpoint(obj: dict) -> dict[str, str]:
    return {
        "geometry_artifact_sha256": obj["geometry_artifact_sha256"],
        "geometry_sidecar_sha256": obj["geometry_sidecar_sha256"],
        "policy_sha256": obj["policy_sha256"],
        "common_configuration_digest_sha256": obj["common_configuration_digest_sha256"],
        "selection_manifest_sha256": obj["selection_manifest_sha256"],
        "continuation_source_sha256": obj["continuation_source_sha256"],
        "current_complete_git_diff_sha256": obj["current_complete_git_diff_sha256"],
        "project_sha256": obj["project_sha256"],
        "manifest_sha256": obj["manifest_sha256"],
        "run_configuration_digest_sha256": obj["run_configuration_digest_sha256"],
        "execution_source_manifest_sha256": obj["execution_source_manifest_sha256"],
        "geometry_h11": str(obj["input"]["h11"]),
        "geometry_polytope": str(obj["input"]["polytope"]),
        "geometry_frst": str(obj["input"]["frst"]),
        "geometry_index": str(obj["geometry_index"]),
    }


def write_hashed_json(path: pathlib.Path, obj: dict) -> None:
    path.write_bytes(canonical(obj) + b"\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pathlib.Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def copy_quarantine_bundle(temp_root: pathlib.Path) -> pathlib.Path:
    if not QUARANTINE_MANIFEST.is_file():
        raise AssertionError("quarantine manifest is missing")
    quarantine_obj = json.loads(QUARANTINE_MANIFEST.read_text(encoding="utf-8"))
    if quarantine_obj.get("geometry_index") != 8002001:
        raise AssertionError("quarantine manifest does not bind geometry 8002001")
    if quarantine_obj.get("old_conversion_policy_version") != "kinv-mixed-tolerance-v1" or \
            quarantine_obj.get("old_kinv_conversion_acceptance") != \
            "max_absolute_error <= 1e-12 OR max_relative_error <= 1e-12" or \
            quarantine_obj.get("failure_reason") != \
            "uncheckpointed header-only partial from lazy-transpose array-hash failure":
        raise AssertionError("quarantine manifest does not bind the current partial policy")
    quarantine_relative = pathlib.PurePath(quarantine_obj["quarantine_geometry_output_root"])
    if quarantine_relative.is_absolute() or quarantine_relative.parts[:2] != \
            ("quarantine", "nonterminal-partials-v1"):
        raise AssertionError("quarantine destination is not canonical")
    quarantine_root = SOURCE_ROOT / quarantine_relative
    destination = temp_root / quarantine_relative
    destination.parent.mkdir(parents=True)
    shutil.copytree(
        quarantine_root,
        destination,
    )
    manifest = temp_root / "quarantine" / QUARANTINE_MANIFEST.name
    shutil.copy2(QUARANTINE_MANIFEST, manifest)
    shutil.copy2(pathlib.Path(str(QUARANTINE_MANIFEST) + ".sha256"),
                 pathlib.Path(str(manifest) + ".sha256"))
    return manifest


def copy_post_resume_quarantine_bundle(temp_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Copy the historical quarantine and its later terminal replacement."""
    manifest = copy_quarantine_bundle(temp_root)
    quarantine_obj = json.loads(manifest.read_text(encoding="utf-8"))
    geometry_name = pathlib.PurePath(
        quarantine_obj["original_geometry_output_root"]
    ).name
    source_checkpoint = SOURCE_ROOT / "checkpoints" / (
        f"{geometry_name}.checkpoint-v6.json"
    )
    if not source_checkpoint.is_file():
        raise AssertionError("post-resume terminal checkpoint is missing")
    checkpoint_obj = checkpoint_object(source_checkpoint)
    destination_checkpoint = temp_root / "checkpoints" / source_checkpoint.name
    destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, destination_checkpoint)
    shutil.copy2(pathlib.Path(str(source_checkpoint) + ".sha256"),
                 pathlib.Path(str(destination_checkpoint) + ".sha256"))
    source_summary = pathlib.Path(checkpoint_obj["summary"]["path"])
    source_shard = pathlib.Path(checkpoint_obj["shards"]["files"][0]["path"])
    destination_root = temp_root / "geometries" / geometry_name
    # The validator resolves output-root-relative paths. Store canonical
    # resolved paths in the copied checkpoint so macOS /var -> /private/var
    # aliases cannot create a false path mismatch in this fixture.
    destination_summary = (destination_root / "summary.csv").resolve()
    destination_shard = (destination_root / "shards" / source_shard.name).resolve()
    destination_shard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_summary, destination_summary)
    shutil.copy2(source_shard, destination_shard)
    checkpoint_obj["summary"]["path"] = str(destination_summary)
    checkpoint_obj["shards"]["files"][0]["path"] = str(destination_shard)
    write_hashed_json(destination_checkpoint, checkpoint_obj)
    return manifest, destination_summary


def run_materialization_regression() -> None:
    """Exercise the runner's normalized-matrix path on geometry 11.

    The transposed HDF5 array is an Adjoint when it is not materialized.
    BLAS can then accumulate E*tau differently from the generator's concrete
    Matrix{Float64} path. The runner must use the latter and reproduce the
    pinned sidecar hash.
    """
    source = WORKTREE / "scripts" / "run_physical_scale_inflation_pilot_20260825.jl"
    code = f'''
include({json.dumps(str(source))})
entries = fixed_inputs()
entry = only(filter(x -> x.h11 == 8 && x.polytope == 2 && x.frst == 1, entries))
meta = geometry_sidecar(entry)
h5open(entry.path, "r") do file
    read_data(name) = read(file[name])
    qprime = Float64.(read_data("cytools/geometric/effective_cone"))
    hyperplanes = Float64.(read_data("cytools/geometric/kahler_hyperplanes"))
    tau = Float64.(read_data("cytools/geometric/divisor_volumes"))
    tip = Float64.(read_data("cytools/geometric/tip"))
    lazy_e = size(qprime, 2) == entry.h11 ? qprime : qprime'
    concrete_e = size(qprime, 2) == entry.h11 ? Matrix{{Float64}}(qprime) : Matrix{{Float64}}(qprime')
    lazy_h = size(hyperplanes, 2) == length(tip) ? hyperplanes : hyperplanes'
    concrete_h = size(hyperplanes, 2) == length(tip) ? Matrix{{Float64}}(hyperplanes) : Matrix{{Float64}}(hyperplanes')
    @assert concrete_e isa Matrix{{Float64}}
    @assert concrete_h isa Matrix{{Float64}}
    @assert canonical_array_sha256(concrete_e * tau) == "{EXPECTED_GEOMETRY11_EFFECTIVE_HASH}"
    @assert canonical_array_sha256(concrete_e * tau) == meta.hashes["effective_divisor_volumes"]
    @assert maximum(abs.(lazy_e * tau - concrete_e * tau)) > 0
end
evidence = load_geometry_evidence(entry, meta)
@assert evidence.evidence.E isa Matrix{{Float64}}
@assert evidence.evidence.H isa Matrix{{Float64}}
@assert canonical_array_sha256(evidence.evidence.effective) == "{EXPECTED_GEOMETRY11_EFFECTIVE_HASH}"
println("geometry11_materialization_regression=passed")
'''
    environment = dict(os.environ)
    environment.pop("PYTHON", None)
    environment.pop("newARGS", None)
    environment["CYAXIVERSE_DATA_DIR"] = DATA_ROOT
    command = [str(JULIA), "--startup-file=no", f"--project={PROJECT}", "-e", code]
    result = subprocess.run(command, cwd=WORKTREE, env=environment, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "geometry11_materialization_regression=passed" not in result.stdout:
        raise AssertionError("geometry-11 materialization regression did not pass")


def expect_quarantine_fail(temp_root: pathlib.Path, mutation: str) -> None:
    manifest = copy_quarantine_bundle(temp_root)
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {
        "expected_policy": obj["policy_sha256"],
        "expected_common": obj["common_configuration_digest_sha256"],
        "expected_selection": obj["selection_manifest_sha256"],
        "expected_diff": obj["current_complete_git_diff_sha256"],
        "expected_source_manifest": obj["execution_source_manifest_sha256"],
        "expected_geometry_index": str(obj["geometry_index"]),
    }
    destination_summary = temp_root / obj["files"][0]["quarantine_relative_path"]
    if mutation == "manifest":
        obj["failure_reason"] = "tampered quarantine record"
        write_hashed_json(manifest, obj)
    elif mutation == "file":
        destination_summary.write_bytes(destination_summary.read_bytes() + b"tamper")
    elif mutation == "data_row":
        data = destination_summary.read_bytes()
        destination_summary.write_bytes(data + b"x,y\n")
        item = next(item for item in obj["files"] if item["role"] == "summary")
        item["sha256"] = sha256_path(destination_summary)
        item["header_sha256"] = item["sha256"]
        item["bytes"] = destination_summary.stat().st_size
        write_hashed_json(manifest, obj)
    else:
        raise AssertionError(f"unknown quarantine mutation: {mutation}")
    try:
        validate_quarantine_manifest(
            manifest,
            output_root=temp_root,
            **expected,
        )
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    raise AssertionError(f"{mutation} quarantine mutation was accepted")


def run_post_resume_quarantine_regressions() -> dict[str, str]:
    """Check terminal replacement acceptance and partial rejection."""
    quarantine_obj = json.loads(QUARANTINE_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "expected_policy": quarantine_obj["policy_sha256"],
        "expected_common": quarantine_obj["common_configuration_digest_sha256"],
        "expected_selection": quarantine_obj["selection_manifest_sha256"],
        "expected_diff": quarantine_obj["current_complete_git_diff_sha256"],
        "expected_source_manifest": quarantine_obj["execution_source_manifest_sha256"],
        "expected_geometry_index": str(quarantine_obj["geometry_index"]),
    }
    with tempfile.TemporaryDirectory(prefix="cyax-quarantine-reappeared-") as temp:
        temp_root = pathlib.Path(temp)
        manifest = copy_quarantine_bundle(temp_root)
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        original_root = temp_root / obj["original_geometry_output_root"]
        original_root.mkdir(parents=True)
        historical_root = temp_root / obj["quarantine_geometry_output_root"]
        shutil.copy2(historical_root / "summary.csv", original_root / "summary.csv")
        (original_root / "shards").mkdir()
        historical_shard = next((historical_root / "shards").glob("*.csv"))
        shutil.copy2(historical_shard, original_root / "shards" / historical_shard.name)
        try:
            validate_quarantine_manifest(manifest, output_root=temp_root, **expected)
        except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            arbitrary_reappeared = "passed"
        else:
            raise AssertionError("arbitrary reappeared original was accepted")
    with tempfile.TemporaryDirectory(prefix="cyax-quarantine-terminal-") as temp:
        temp_root = pathlib.Path(temp)
        manifest, terminal_summary = copy_post_resume_quarantine_bundle(temp_root)
        validate_quarantine_manifest(manifest, output_root=temp_root, **expected)
        historical_summary = temp_root / quarantine_obj["files"][0]["quarantine_relative_path"]
        terminal_summary.write_bytes(historical_summary.read_bytes())
        try:
            validate_quarantine_manifest(manifest, output_root=temp_root, **expected)
        except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            mutated_replacement = "passed"
        else:
            raise AssertionError("mutated terminal replacement was accepted")
    return {
        "arbitrary_reappeared_original_fail_closed": arbitrary_reappeared,
        "mutated_terminal_replacement_fail_closed": mutated_replacement,
        "validated_terminal_replacement": "passed",
    }


def copy_bundle(temp_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    source_obj = checkpoint_object(CHECKPOINT)
    temp_checkpoint = temp_root / "checkpoints" / CHECKPOINT.name
    temp_summary = temp_root / "geometries" / "h11_005_np_0000001_cy_0000001" / "summary.csv"
    temp_shard = temp_root / "geometries" / "h11_005_np_0000001_cy_0000001" / "shards" / \
        "inflation_scale_continuation_shard_0001_of_0001.csv"
    source_path = pathlib.Path(source_obj["execution_source_manifest_path"])
    temp_source = temp_root / source_path.name
    temp_checkpoint.parent.mkdir(parents=True)
    temp_summary.parent.mkdir(parents=True)
    temp_shard.parent.mkdir(parents=True)
    shutil.copy2(CHECKPOINT, temp_checkpoint)
    shutil.copy2(CHECKSUM, pathlib.Path(str(temp_checkpoint) + ".sha256"))
    shutil.copy2(pathlib.Path(source_obj["summary"]["path"]), temp_summary)
    shutil.copy2(pathlib.Path(source_obj["shards"]["files"][0]["path"]), temp_shard)
    shutil.copy2(source_path, temp_source)
    shutil.copy2(pathlib.Path(str(source_path) + ".sha256"), pathlib.Path(str(temp_source) + ".sha256"))
    if QUARANTINE_MANIFEST.is_file():
        copy_quarantine_bundle(temp_root)
    obj = checkpoint_object(temp_checkpoint)
    obj["summary"]["path"] = str(temp_summary)
    obj["shards"]["files"][0]["path"] = str(temp_shard)
    obj["execution_source_manifest_path"] = str(temp_source)
    write_hashed_json(temp_checkpoint, obj)
    return temp_checkpoint, temp_summary, temp_shard, temp_source


def expect_fail(temp_root: pathlib.Path, mutation: str) -> None:
    checkpoint = temp_root / "checkpoints" / CHECKPOINT.name
    summary = temp_root / "geometries" / "h11_005_np_0000001_cy_0000001" / "summary.csv"
    shard = temp_root / "geometries" / "h11_005_np_0000001_cy_0000001" / "shards" / \
        "inflation_scale_continuation_shard_0001_of_0001.csv"
    obj = checkpoint_object(checkpoint)
    expected = expected_from_checkpoint(obj)
    if mutation in {"checkpoint", "resume_config"}:
        obj["run_configuration"]["max_negative_modes"] = 2
        write_hashed_json(checkpoint, obj)
    elif mutation == "summary":
        summary.write_bytes(summary.read_bytes().replace(b",0.9,", b",0.91,", 1))
    elif mutation == "shard":
        shard.write_bytes(shard.read_bytes().replace(b",0.9,", b",0.91,", 1))
    elif mutation == "source_manifest":
        source = pathlib.Path(obj["execution_source_manifest_path"])
        manifest = json.loads(source.read_text(encoding="utf-8"))
        runner = "scripts/run_physical_scale_inflation_pilot_20260825.jl"
        manifest["source_file_hashes"][runner] = "0" * 64
        payload = dict(manifest)
        payload.pop("payload_sha256", None)
        manifest["payload_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
        write_hashed_json(source, manifest)
        obj["execution_source_manifest_sha256"] = sha256_path(source)
        obj["execution_source_hashes"] = manifest["source_file_hashes"]
        write_hashed_json(checkpoint, obj)
        expected = expected_from_checkpoint(obj)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    try:
        validate_canonical_checkpoint(
            checkpoint, expected=expected, summary_path=summary, shard_path=shard, output_root=temp_root
        )
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    raise AssertionError(f"{mutation} mutation was accepted")


def tree_snapshot(root: pathlib.Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, sha256_path(path))
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def main() -> int:
    if not (CHECKPOINT.is_file() and CHECKSUM.is_file()):
        raise AssertionError("v5 adopted geometry-1 checkpoint is missing")
    obj = checkpoint_object(CHECKPOINT)
    if obj.get("schema") != CANONICAL_SCHEMA:
        raise AssertionError("unexpected canonical checkpoint schema")
    summary = pathlib.Path(obj["summary"]["path"])
    shard = pathlib.Path(obj["shards"]["files"][0]["path"])
    expected = expected_from_checkpoint(obj)
    validate_canonical_checkpoint(CHECKPOINT, expected=expected, summary_path=summary, shard_path=shard, output_root=SOURCE_ROOT)
    run_materialization_regression()
    if not QUARANTINE_MANIFEST.is_file():
        raise AssertionError("quarantine manifest is missing")
    quarantine_obj = json.loads(QUARANTINE_MANIFEST.read_text(encoding="utf-8"))
    quarantine_expected = {
        "expected_policy": quarantine_obj["policy_sha256"],
        "expected_common": quarantine_obj["common_configuration_digest_sha256"],
        "expected_selection": quarantine_obj["selection_manifest_sha256"],
        "expected_diff": quarantine_obj["current_complete_git_diff_sha256"],
        "expected_source_manifest": quarantine_obj["execution_source_manifest_sha256"],
        "expected_geometry_index": str(quarantine_obj["geometry_index"]),
    }
    validate_quarantine_manifest(
        QUARANTINE_MANIFEST,
        output_root=SOURCE_ROOT,
        **quarantine_expected,
    )
    post_resume = (SOURCE_ROOT / "checkpoints" /
                   "h11_008_np_0000002_cy_0000001.checkpoint-v6.json").is_file()
    post_resume_quarantine_tests = {}
    if post_resume:
        post_resume_quarantine_tests = run_post_resume_quarantine_regressions()
    quarantine_mutations = {}
    for mutation in ("manifest", "file", "data_row"):
        with tempfile.TemporaryDirectory(prefix=f"cyax-quarantine-{mutation}-") as temp:
            expect_quarantine_fail(pathlib.Path(temp), mutation)
            quarantine_mutations[mutation] = "passed"
    quarantine_before = tree_snapshot(SOURCE_ROOT / "quarantine")
    validate_quarantine_manifest(
        QUARANTINE_MANIFEST,
        output_root=SOURCE_ROOT,
        **quarantine_expected,
    )
    validate_quarantine_manifest(
        QUARANTINE_MANIFEST,
        output_root=SOURCE_ROOT,
        **quarantine_expected,
    )
    quarantine_after = tree_snapshot(SOURCE_ROOT / "quarantine")
    if quarantine_before != quarantine_after:
        raise AssertionError("idempotent quarantine validation changed output")
    mutations = {}
    for mutation in ("checkpoint", "summary", "shard", "source_manifest", "resume_config"):
        with tempfile.TemporaryDirectory(prefix=f"cyax-{mutation}-") as temp:
            temp_root = pathlib.Path(temp)
            copy_bundle(temp_root)
            expect_fail(temp_root, mutation)
            mutations[mutation] = "passed"

    # Exercise the validation-only resume on a temporary one-geometry bundle.
    # The live output root also contains the intentionally preserved header-only
    # next-geometry failure, which must remain outside the terminal bundle.
    with tempfile.TemporaryDirectory(prefix="cyax-idempotent-resume-") as temp:
        resume_root = pathlib.Path(temp)
        copy_bundle(resume_root)
        for name in ("environment_manifest-v1.json", "run_manifest-v1.json"):
            shutil.copy2(SOURCE_ROOT / name, resume_root / name)
            shutil.copy2(pathlib.Path(str(SOURCE_ROOT / name) + ".sha256"),
                         pathlib.Path(str(resume_root / name) + ".sha256"))
            manifest_path = resume_root / name
            manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_obj.get("quarantine_manifest_status") == "quarantined":
                manifest_obj["quarantine_manifest_path"] = str(
                    resume_root / "quarantine" / QUARANTINE_MANIFEST.name
                )
                write_hashed_json(manifest_path, manifest_obj)
        before = tree_snapshot(resume_root)
        environment = dict(os.environ)
        environment.pop("PYTHON", None)
        environment.pop("newARGS", None)
        environment["CYAXIVERSE_DATA_DIR"] = DATA_ROOT
        command = [
            str(JULIA), "--startup-file=no", f"--project={PROJECT}",
            str(WORKTREE / "scripts/run_physical_scale_inflation_pilot_20260825.jl"),
            "--resume-existing-only", "--output-root", str(resume_root),
        ]
        subprocess.run(command, cwd=WORKTREE, env=environment, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        after = tree_snapshot(resume_root)
        if before != after:
            raise AssertionError("idempotent resume changed output bytes or timestamps")

    environment = dict(os.environ)
    environment.pop("PYTHON", None)
    environment.pop("newARGS", None)
    environment["CYAXIVERSE_DATA_DIR"] = DATA_ROOT
    selection_command = [
        str(JULIA), "--startup-file=no", f"--project={PROJECT}",
        str(WORKTREE / "scripts/run_physical_scale_inflation_pilot_20260825.jl"),
        "--resume", "--resume-selection-only", "--max-geometries", "1",
        "--output-root", str(SOURCE_ROOT),
    ]
    if not post_resume:
        selection_command.insert(5, "--quarantine-nonterminal-partials")
    selection_result = subprocess.run(
        selection_command, cwd=WORKTREE, env=environment, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    quarantine_geometry_index = quarantine_obj["geometry_index"]
    if post_resume:
        if "next_geometry_index=none" not in selection_result.stdout or \
                "runner_targets=\n" not in selection_result.stdout:
            raise AssertionError("post-resume selection-only did not prove no target remains")
    else:
        quarantine_h11 = int(quarantine_obj["geometry"]["h11"])
        quarantine_polytope = int(quarantine_obj["geometry"]["polytope"])
        expected_target_number = (quarantine_h11 - 5) * 3 + quarantine_polytope
        if f"next_geometry_index={quarantine_geometry_index}" not in selection_result.stdout or \
                f"runner_targets={expected_target_number}" not in selection_result.stdout:
            raise AssertionError("resume-selection-only did not prove the quarantined geometry is next")

    # Validate every current terminal checkpoint against its own immutable
    # per-geometry summary and shard. This catches accidental reuse of a
    # shared root CSV when the completed set is resumed.
    checkpoint_count = 0
    checkpoint_indices = []
    for checkpoint_path in sorted((SOURCE_ROOT / "checkpoints").glob(
            "h11_*_np_*_cy_*.checkpoint-v6.json")):
        checkpoint_obj = checkpoint_object(checkpoint_path)
        checkpoint_summary = pathlib.Path(checkpoint_obj["summary"]["path"])
        checkpoint_shard = pathlib.Path(checkpoint_obj["shards"]["files"][0]["path"])
        checkpoint_expected = expected_from_checkpoint(checkpoint_obj)
        validate_canonical_checkpoint(
            checkpoint_path, expected=checkpoint_expected,
            summary_path=checkpoint_summary, shard_path=checkpoint_shard,
            output_root=SOURCE_ROOT,
        )
        checkpoint_count += 1
        checkpoint_indices.append(checkpoint_obj["geometry_index"])
    if checkpoint_count < 10:
        raise AssertionError(f"expected at least 10 terminal checkpoints, got {checkpoint_count}")

    resource_evidence = validate_resource_evidence_root(SOURCE_ROOT)
    accounting_path = SOURCE_ROOT / "terminal_accounting_manifest-v1.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    terminal_resources = accounting.get("resource_accounting")
    if post_resume:
        if accounting.get("status") != "completed" or checkpoint_count != 18:
            raise AssertionError("post-resume terminal accounting is not complete")
        if not isinstance(terminal_resources, dict):
            raise AssertionError("completed terminal accounting has no resource evidence")
        for key in ("maxrss_bytes", "rss_cap_bytes", "output_cap_bytes",
                    "source_checkpoint_path", "source_checkpoint_sha256",
                    "calculation_checkpoint_count"):
            if terminal_resources.get(key) != resource_evidence.get(key):
                raise AssertionError(f"terminal resource evidence mismatch: {key}")
        if terminal_resources["maxrss_bytes"] > terminal_resources["rss_cap_bytes"]:
            raise AssertionError("terminal resource evidence exceeds the RSS cap")
        if terminal_resources["output_bytes_under_root"] > terminal_resources["output_cap_bytes"]:
            raise AssertionError("terminal resource evidence exceeds the output cap")

    report = {
        "schema": "physical-scale-checkpoint-test-report-v2",
        "status": "passed",
        "tests": {
            "checkpoint_mutation_fail_closed": mutations["checkpoint"],
            "per_geometry_summary_mutation_fail_closed": mutations["summary"],
            "per_geometry_shard_mutation_fail_closed": mutations["shard"],
            "runner_source_manifest_mutation_fail_closed": mutations["source_manifest"],
            "resume_config_mutation_fail_closed": mutations["resume_config"],
            "quarantine_manifest_mutation_fail_closed": quarantine_mutations["manifest"],
            "quarantined_file_mutation_fail_closed": quarantine_mutations["file"],
            "quarantined_data_row_fail_closed": quarantine_mutations["data_row"],
            "quarantine_geometry11_current_policy": "passed",
            "quarantine_validation_idempotent_zero_output_mutation": "passed",
            "idempotent_resume_zero_output_mutation": "passed",
            "resume_selection_only_next_geometry": "passed",
            "runner_materialization_geometry11_hash": "passed",
            "all_terminal_checkpoint_bundles": "passed",
            "checksum_bound_calculation_resource_evidence": "passed",
            "completed_terminal_resource_caps": "passed" if post_resume else "not_applicable",
        },
        "checkpoint_sha256": sha256_path(CHECKPOINT),
        "summary_sha256": sha256_path(summary),
        "shard_sha256": sha256_path(shard),
        "execution_source_manifest_sha256": obj["execution_source_manifest_sha256"],
        "run_configuration_digest_sha256": obj["run_configuration_digest_sha256"],
        "quarantine_manifest_sha256": sha256_path(QUARANTINE_MANIFEST),
        "quarantine_geometry_index": quarantine_obj["geometry_index"],
        "coverage_label": obj["coverage_label"],
        "terminal_geometry_count": obj["terminal_counts"]["terminal_geometry_count"],
        "terminal_geometry_indices": obj["terminal_counts"]["terminal_geometry_indices"],
        "terminal_scale_point_count": obj["terminal_counts"]["terminal_scale_point_count"],
        "validated_terminal_checkpoint_count": checkpoint_count,
        "validated_terminal_checkpoint_indices": checkpoint_indices,
    }
    report["tests"].update(post_resume_quarantine_tests)
    report_path = WORKTREE / "validation" / "physical_scale_checkpoint_tests_20260825.json"
    temporary = pathlib.Path(str(report_path) + f".tmp-{os.getpid()}")
    temporary.write_bytes(canonical(report) + b"\n")
    temporary.replace(report_path)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
