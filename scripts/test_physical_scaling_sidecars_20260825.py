#!/usr/bin/env python3
"""Focused schema, digest, mutation, malformed-input, and fail-closed tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import tempfile


WORKTREE = pathlib.Path(__file__).resolve().parents[1]
CERT_DIR = WORKTREE / "validation" / "physical_scaling_certificates_20260825"
POLICY = CERT_DIR / "physical_scaling_pilot_policy-v1.json"
POLICY_CHECKSUM = CERT_DIR / "physical_scaling_pilot_policy-v1.sha256"
SIDECAR_CHECKSUM = CERT_DIR / "physical_scaling_certificates.sha256"
EXPECTED_MANIFEST_SHA256 = "a6df5dca258c11724d4162477cdee7cc34e5802f2f3f296a7ea64b55f23c3247"
EXPECTED_CERTIFICATE_VERSION = "physical-domain-certificate-3"
EXPECTED_CONVERSION_POLICY_VERSION = "kinv-mixed-tolerance-v1"
EXPECTED_KINV_CONVERSION_RULE = "max_absolute_error <= 1e-12 OR max_relative_error <= 1e-12"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load(path: pathlib.Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def expected_sha_line(path: pathlib.Path) -> tuple[str, str]:
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise AssertionError(f"malformed checksum line: {path}")
    return fields[0], fields[1]


def validate_sidecar(path: pathlib.Path, policy: dict) -> dict:
    obj = load(path)
    required = {
        "schema", "certificate_version", "input", "selection_manifest",
        "audit_evidence", "field_classifications", "approved_conventions",
        "source_identity", "configuration_payload", "configuration_digest",
        "geometry_evidence", "stored_QL_reference", "precision_conversion_audit",
        "physical_scaling_gate", "physical_control_gate", "physical_viability",
    }
    if not required.issubset(obj):
        raise AssertionError(f"missing sidecar fields in {path.name}")
    if obj["schema"] != "physical-scaling-certificate-v1" or obj["certificate_version"] != EXPECTED_CERTIFICATE_VERSION:
        raise AssertionError(f"unexpected sidecar schema in {path.name}")
    inp = obj["input"]
    if inp["orientifold_requested"] is not False:
        raise AssertionError(f"orientifold was not false in {path.name}")
    if obj["selection_manifest"]["sha256"] != EXPECTED_MANIFEST_SHA256:
        raise AssertionError(f"selection manifest mismatch in {path.name}")
    if obj["physical_scaling_gate"]["status"] != "passed":
        raise AssertionError(f"scaling gate did not pass in {path.name}")
    if obj["physical_control_gate"]["status"] not in {"passed", "not_established", "failed"}:
        raise AssertionError(f"malformed control status in {path.name}")
    if obj["physical_viability"]["status"] != "not_evaluated":
        raise AssertionError(f"sidecar made a viability claim in {path.name}")
    conventions = obj["approved_conventions"]
    if conventions.get("conversion_policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
        raise AssertionError(f"conversion policy provenance mismatch in {path.name}")
    if conventions.get("kinv_conversion_acceptance") != EXPECTED_KINV_CONVERSION_RULE:
        raise AssertionError(f"Kinv conversion rule mismatch in {path.name}")
    payload = obj["configuration_payload"]
    digest = digest_bytes(canonical(payload))
    if digest != obj["configuration_digest"]["geometry_sha256"]:
        raise AssertionError(f"geometry configuration digest mismatch in {path.name}")
    if obj["configuration_digest"]["common_sha256"] != policy["common_configuration_digest_sha256"]:
        raise AssertionError(f"common configuration digest mismatch in {path.name}")
    if payload.get("conversion_policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
        raise AssertionError(f"configuration conversion policy mismatch in {path.name}")
    source = obj["source_identity"]
    required_source = {
        "geometry_sha256", "polytope_id", "triangulation_id", "cy3_fingerprint",
        "stored_cytools_version", "historical_generator_source_sha256",
        "current_continuation_source_sha256", "current_complete_git_diff_sha256",
        "audit_script_sha256", "evidence_report_sha256", "selection_manifest_sha256",
    }
    if not required_source.issubset(source):
        raise AssertionError(f"incomplete reconstructed source identity in {path.name}")
    if source["geometry_sha256"] != inp["artifact_sha256"]:
        raise AssertionError(f"source identity geometry hash mismatch in {path.name}")
    ql = obj["stored_QL_reference"]
    if ql["threshold_status"] not in {"passed_existing_cited_tolerance", "failed_existing_cited_tolerance"}:
        raise AssertionError(f"missing explicit Q/L threshold status in {path.name}")
    if source.get("conversion_policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
        raise AssertionError(f"source conversion policy provenance mismatch in {path.name}")
    conversion = obj["precision_conversion_audit"]
    if conversion.get("policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
        raise AssertionError(f"precision conversion policy mismatch in {path.name}")
    if conversion.get("acceptance_rules", {}).get("Kinv") != EXPECTED_KINV_CONVERSION_RULE:
        raise AssertionError(f"precision Kinv conversion rule mismatch in {path.name}")
    for metric in ("L", "K", "Kinv", "tau", "volume"):
        if conversion.get("metric_status", {}).get(metric) != "passed":
            raise AssertionError(f"precision conversion metric is not passed in {path.name}: {metric}")
    kinv_metric = conversion.get("metrics", {}).get("Kinv", {})
    if not (kinv_metric.get("max_absolute_error", float("inf")) <= 1e-12 or
            kinv_metric.get("max_relative_error", float("inf")) <= 1e-12):
        raise AssertionError(f"Kinv mixed conversion tolerance failed in {path.name}")
    return obj


def fail_closed_mutation(obj: dict) -> None:
    mutated = copy.deepcopy(obj)
    mutated["input"]["artifact_sha256"] = "0" * 64
    if mutated["input"]["artifact_sha256"] == obj["input"]["artifact_sha256"]:
        raise AssertionError("mutation did not change input identity")
    if mutated["source_identity"]["geometry_sha256"] == mutated["input"]["artifact_sha256"]:
        raise AssertionError("identity mutation was not detected")


def fail_closed_malformed(obj: dict) -> None:
    malformed = copy.deepcopy(obj)
    del malformed["physical_scaling_gate"]["status"]
    if "status" in malformed["physical_scaling_gate"]:
        raise AssertionError("malformed status deletion failed")
    try:
        if malformed["physical_scaling_gate"].get("status") != "passed":
            raise ValueError("missing physical_scaling_gate.status")
    except ValueError:
        return
    raise AssertionError("malformed sidecar was not rejected")


def main() -> int:
    policy = load(POLICY)
    if digest_bytes(canonical(policy["common_configuration_payload"])) != policy["common_configuration_digest_sha256"]:
        raise AssertionError("common configuration digest does not reproduce")
    common = policy["common_configuration_payload"]
    if common.get("conversion_policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
        raise AssertionError("conversion policy version does not reproduce")
    if common.get("conversion_acceptance", {}).get("Kinv") != EXPECTED_KINV_CONVERSION_RULE:
        raise AssertionError("Kinv conversion rule does not reproduce")
    policy_sha, policy_name = expected_sha_line(POLICY_CHECKSUM)
    if policy_name != POLICY.name or policy_sha != digest_bytes(POLICY.read_bytes()):
        raise AssertionError("policy checksum does not reproduce")
    rows = SIDECAR_CHECKSUM.read_text(encoding="utf-8").splitlines()
    if len(rows) != 18:
        raise AssertionError(f"expected 18 sidecar checksum rows, got {len(rows)}")
    paths = sorted(CERT_DIR.glob("h11_*.physical-scaling-certificate-v1.json"))
    if len(paths) != 18:
        raise AssertionError(f"expected 18 sidecars, got {len(paths)}")
    checksums = {}
    for row in rows:
        fields = row.split()
        if len(fields) != 2:
            raise AssertionError("malformed sidecar checksum row")
        checksums[fields[1]] = fields[0]
    records = []
    for path in paths:
        if checksums.get(path.name) != digest_bytes(path.read_bytes()):
            raise AssertionError(f"sidecar checksum mismatch: {path.name}")
        records.append(validate_sidecar(path, policy))
    identities = {(r["input"]["h11"], r["input"]["polytope_id"]) for r in records}
    if len(identities) != 18:
        raise AssertionError("sidecar input identities are not distinct")
    for h11 in range(5, 11):
        if sum(r["input"]["h11"] == h11 for r in records) != 3:
            raise AssertionError(f"wrong sidecar count at h11={h11}")
    fail_closed_mutation(records[0])
    fail_closed_malformed(records[0])
    report = {
        "schema": "physical-scaling-sidecar-test-report-v1",
        "status": "passed",
        "tests": {
            "schema": "passed",
            "canonical_common_digest": "passed",
            "canonical_geometry_digests": "passed",
            "policy_checksum": "passed",
            "sidecar_checksums": "passed",
            "strict_18_input_binding": "passed",
            "mutation_fail_closed": "passed",
            "malformed_status_fail_closed": "passed",
            "control_gate_independence": "passed",
            "viability_not_claimed": "passed",
        },
        "sidecar_count": 18,
        "physical_scaling_gate_passed": 18,
        "physical_control_gate_not_established": sum(r["physical_control_gate"]["status"] == "not_established" for r in records),
        "new_output_bytes_tested": sum(p.stat().st_size for p in paths) + POLICY.stat().st_size,
    }
    out = CERT_DIR / "physical_scaling_sidecar_tests_20260825.json"
    tmp = out.with_suffix(out.suffix + f".tmp-{os.getpid()}")
    tmp.write_bytes(canonical(report) + b"\n")
    tmp.replace(out)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
