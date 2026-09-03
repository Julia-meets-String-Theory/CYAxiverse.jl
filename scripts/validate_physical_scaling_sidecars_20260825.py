#!/usr/bin/env python3
"""Strict JSON-sidecar and fixed-selection validation for preflight."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys


EXPECTED_MANIFEST_SHA256 = "a6df5dca258c11724d4162477cdee7cc34e5802f2f3f296a7ea64b55f23c3247"
EXPECTED_COMMIT = "9f31d716eaab8d63d3f76826a40de5ae38c7015d"
EXPECTED_BRANCH = "agents/physical-scale-inflation-20260825"
EXPECTED_CERTIFICATE_VERSION = "physical-domain-certificate-3"
EXPECTED_CONVERSION_POLICY_VERSION = "kinv-mixed-tolerance-v1"
EXPECTED_KINV_CONVERSION_RULE = "max_absolute_error <= 1e-12 OR max_relative_error <= 1e-12"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: pathlib.Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def main() -> int:
    if len(sys.argv) != 6:
        fail("usage: validate... CERT_DIR POLICY MANIFEST DATA_ROOT WORKTREE")
    cert_dir, policy_path, manifest_path, data_root, worktree = map(pathlib.Path, sys.argv[1:])
    policy = load(policy_path)
    if policy.get("schema") != "physical-scaling-pilot-policy-v1":
        fail("policy schema mismatch")
    if digest(canonical(policy["common_configuration_payload"])) != policy["common_configuration_digest_sha256"]:
        fail("common policy digest mismatch")
    common = policy["common_configuration_payload"]
    if common.get("conversion_policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
        fail("conversion policy version mismatch")
    if common.get("conversion_acceptance", {}).get("Kinv") != EXPECTED_KINV_CONVERSION_RULE:
        fail("Kinv conversion acceptance rule mismatch")
    policy_checksum = policy_path.with_name("physical_scaling_pilot_policy-v1.sha256").read_text(encoding="utf-8").split()
    if len(policy_checksum) != 2 or policy_checksum[0] != digest(policy_path.read_bytes()) or policy_checksum[1] != policy_path.name:
        fail("policy checksum mismatch")
    manifest = load(manifest_path)
    if digest(manifest_path.read_bytes()) != EXPECTED_MANIFEST_SHA256:
        fail("selection manifest hash mismatch")
    if manifest.get("selected_count") != 18 or len(manifest.get("selected_inputs", [])) != 18:
        fail("selection manifest count mismatch")
    if manifest.get("git_commit") != EXPECTED_COMMIT or manifest.get("git_branch") != EXPECTED_BRANCH:
        fail("selection manifest source identity mismatch")
    data_root_text = str(data_root).rstrip("/")
    manifest_by_relative: dict[str, dict] = {}
    for item in manifest["selected_inputs"]:
        path = item.get("path", "")
        prefix = data_root_text + "/"
        if not path.startswith(prefix):
            fail("manifest path outside data root")
        relative = path[len(prefix):]
        if relative in manifest_by_relative:
            fail("duplicate selection path")
        if item.get("orientifold", {}).get("requested") is not False:
            fail("selection entry is not explicitly non-orientifold")
        manifest_by_relative[relative] = item
    if subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip() != EXPECTED_COMMIT:
        fail("repository HEAD mismatch")
    if subprocess.check_output(["git", "-C", str(worktree), "branch", "--show-current"], text=True).strip() != EXPECTED_BRANCH:
        fail("repository branch mismatch")
    source_files = policy["source_identity"]["source_file_hashes"]
    for relative, expected in source_files.items():
        path = worktree / relative
        if not path.is_file() or digest(path.read_bytes()) != expected:
            fail(f"source hash mismatch: {relative}")
    complete_diff = subprocess.check_output(["git", "-C", str(worktree), "diff", "--binary", "HEAD"])
    if digest(complete_diff) != policy["source_identity"]["current_complete_git_diff_sha256"]:
        fail("complete Git diff hash mismatch")

    checksum_rows = (cert_dir / "physical_scaling_certificates.sha256").read_text(encoding="utf-8").splitlines()
    checksums: dict[str, str] = {}
    for row in checksum_rows:
        fields = row.split()
        if len(fields) != 2:
            fail("malformed sidecar checksum row")
        checksums[fields[1]] = fields[0]
    paths = sorted(cert_dir.glob("h11_*.physical-scaling-certificate-v1.json"))
    if len(paths) != 18:
        fail("expected exactly 18 sidecars")
    expected_classifications = {
        "basis_identity": "stored_alias", "charge_orientation": "exactly_reconstructable",
        "configuration_digest": "genuinely_unavailable", "curve_volumes": "stored_alias",
        "cy_volume": "stored_alias", "divisor_volumes": "stored_alias",
        "effective_divisor_volumes": "exactly_reconstructable", "instanton_control": "genuinely_unavailable",
        "kahler_margin": "exactly_reconstructable", "kinv": "stored_alias",
        "moduli_status": "source_fixed_convention", "normalization": "genuinely_unavailable",
        "perturbative_control": "genuinely_unavailable", "phase_convention": "genuinely_unavailable",
        "potent_curve_volumes": "genuinely_unavailable", "precision_bits": "exactly_reconstructable",
        "prime_divisor_volumes": "stored_alias", "source_identity": "genuinely_unavailable",
        "spd_tolerance": "genuinely_unavailable", "units": "genuinely_unavailable",
        "visible_sector_status": "source_fixed_convention",
    }
    rows: list[tuple[str, ...]] = []
    for path in paths:
        obj = load(path)
        if checksums.get(path.name) != digest(path.read_bytes()):
            fail(f"sidecar checksum mismatch: {path.name}")
        if obj.get("schema") != "physical-scaling-certificate-v1" or obj.get("certificate_version") != EXPECTED_CERTIFICATE_VERSION:
            fail(f"sidecar schema mismatch: {path.name}")
        inp = obj.get("input", {})
        relative = inp.get("relative_path")
        if relative not in manifest_by_relative:
            fail(f"sidecar path is not exactly in manifest: {path.name}")
        item = manifest_by_relative[relative]
        for field, expected in (("artifact_sha256", item["sha256"]), ("h11", item["h11"]), ("polytope", item["polytope"]), ("polytope_id", item["polytope_id"]), ("triangulation_id", item["triangulation_id"])):
            if inp.get(field) != expected:
                fail(f"sidecar/manifest binding mismatch: {path.name}/{field}")
        if inp.get("orientifold_requested") is not False:
            fail(f"sidecar orientifold request is not false: {path.name}")
        artifact = data_root / relative
        if digest(artifact.read_bytes()) != item["sha256"]:
            fail(f"artifact hash mismatch: {relative}")
        expected_name = f"h11_{item['h11']:03d}_np_{item['polytope']:07d}_cy_0000001.physical-scaling-certificate-v1.json"
        if path.name != expected_name:
            fail(f"sidecar filename identity mismatch: {path.name}")
        if obj.get("selection_manifest", {}).get("sha256") != EXPECTED_MANIFEST_SHA256:
            fail(f"sidecar manifest hash mismatch: {path.name}")
        if obj.get("field_classifications") != expected_classifications:
            fail(f"field classifications changed: {path.name}")
        conventions = obj.get("approved_conventions", {})
        if conventions.get("conversion_policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
            fail(f"conversion policy provenance is missing: {path.name}")
        if conventions.get("kinv_conversion_acceptance") != EXPECTED_KINV_CONVERSION_RULE:
            fail(f"Kinv conversion rule provenance is missing: {path.name}")
        if obj.get("physical_scaling_gate", {}).get("status") != "passed":
            fail(f"physical scaling gate is not passed: {path.name}")
        control_status = obj.get("physical_control_gate", {}).get("status")
        if control_status not in {"passed", "not_established", "failed"}:
            fail(f"malformed physical control gate: {path.name}")
        if obj.get("physical_viability", {}).get("status") != "not_evaluated":
            fail(f"sidecar contains viability claim: {path.name}")
        payload = obj.get("configuration_payload")
        if digest(canonical(payload)) != obj.get("configuration_digest", {}).get("geometry_sha256"):
            fail(f"geometry configuration digest mismatch: {path.name}")
        if obj.get("configuration_digest", {}).get("common_sha256") != policy["common_configuration_digest_sha256"]:
            fail(f"common configuration digest mismatch: {path.name}")
        if obj.get("configuration_payload", {}).get("conversion_policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
            fail(f"configuration conversion policy mismatch: {path.name}")
        source = obj.get("source_identity", {})
        required_source = ("geometry_sha256", "polytope_id", "triangulation_id", "cy3_fingerprint", "stored_cytools_version", "historical_generator_source_sha256", "current_continuation_source_sha256", "current_complete_git_diff_sha256", "audit_script_sha256", "evidence_report_sha256", "selection_manifest_sha256", "conversion_policy_version")
        if any(key not in source for key in required_source):
            fail(f"source identity is incomplete: {path.name}")
        if source["geometry_sha256"] != item["sha256"] or source["current_complete_git_diff_sha256"] != policy["source_identity"]["current_complete_git_diff_sha256"] or source["selection_manifest_sha256"] != EXPECTED_MANIFEST_SHA256 or source["conversion_policy_version"] != EXPECTED_CONVERSION_POLICY_VERSION:
            fail(f"source identity value mismatch: {path.name}")
        ql = obj.get("stored_QL_reference", {})
        if ql.get("threshold_status") not in {"passed_existing_cited_tolerance", "failed_existing_cited_tolerance"}:
            fail(f"Q/L threshold status missing: {path.name}")
        required_status = obj.get("physical_scaling_gate", {}).get("required_field_statuses", {})
        if not required_status or any(str(value).startswith("failed") for value in required_status.values()):
            fail(f"scaling required field failed: {path.name}")
        conversion = obj.get("precision_conversion_audit", {})
        if conversion.get("policy_version") != EXPECTED_CONVERSION_POLICY_VERSION:
            fail(f"precision conversion policy mismatch: {path.name}")
        if conversion.get("acceptance_rules", {}).get("Kinv") != EXPECTED_KINV_CONVERSION_RULE:
            fail(f"precision Kinv conversion rule mismatch: {path.name}")
        metrics = conversion.get("metrics", {})
        metric_status = conversion.get("metric_status", {})
        for metric in ("L", "K", "Kinv", "tau", "volume"):
            if metric not in metrics or metric_status.get(metric) != "passed":
                fail(f"precision conversion metric is not passed: {path.name}/{metric}")
        kinv_metric = metrics["Kinv"]
        if not (kinv_metric.get("max_absolute_error", float("inf")) <= 1e-12 or
                kinv_metric.get("max_relative_error", float("inf")) <= 1e-12):
            fail(f"Kinv mixed conversion tolerance failed: {path.name}")
        basis = obj.get("basis_identity", {})
        rows.append((path.name, relative, item["sha256"], str(item["h11"]), str(item["polytope"]), item["polytope_id"], item["triangulation_id"], checksums[path.name], obj["physical_scaling_gate"]["status"], control_status, str(ql["max_log10_error"]), str(ql["sign_mismatches"]), basis.get("basis_sha256", ""), basis.get("basis_matrix_sha256", "")))
    print("META\t" + "\t".join((policy["common_configuration_digest_sha256"], policy["source_identity"]["current_complete_git_diff_sha256"])))
    for row in rows:
        print("ENTRY\t" + "\t".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
