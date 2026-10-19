#!/usr/bin/env python3
"""Offline verifier for the clean exact-input rerun contract.

The default invocation verifies the pre-dispatch state.  ``--phase captured``
additionally requires every response to be present and hash-consistent;
``--phase scored`` checks opaque response/scorecard completeness and metadata
blindness.  No result or score is inferred by this verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
RUN_MANIFEST = HERE / "runs" / "manifest.json"
RUNS_DIR = HERE / "runs"
MAPPING = HERE / "scoring" / "condition_mapping.json"
BLIND_MANIFEST = HERE / "scoring" / "blind_materialization_manifest.json"
LEAKAGE_REPORT = HERE / "scoring" / "blind_content_leakage.json"
SCORER_INPUT = HERE / "scoring" / "scorer_input.input"
SCORER_INPUT_MANIFEST = HERE / "scoring" / "scorer_input_manifest.json"
SCORER_EVENTS = HERE / "scoring" / "scorer_events.jsonl"
SCORER_RUN = HERE / "scoring" / "scorer_run.json"
OPAQUE = tuple(f"S{i}" for i in range(1, 7))
RUNS = ("A1", "B1", "B2", "A2", "A3", "B3")
CRITICAL = ("K6", "K7", "K8", "K9", "K11")
SCORE_KEYS = tuple(f"K{i}" for i in range(1, 13))
SOURCE_IDS = {
    "issue-155-current", "issue-155-checkpoint", "issue-155-closure-event", "pr-156",
    "spec-0155", "agents-contract-at-merge", "sdd-contract-at-merge",
    "checkpoint-guide-at-merge", "pr-156-merge-commit", "closure-search",
}


def load_dispatch_module():
    spec = importlib.util.spec_from_file_location("exact_dispatch", HERE / "exact_dispatch.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load exact dispatch module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_launcher_module():
    spec = importlib.util.spec_from_file_location("launch_subject", HERE / "launch_subject.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load launch_subject module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_identity_module():
    spec = importlib.util.spec_from_file_location("sanitize_identity", HERE / "sanitize_identity.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load identity sanitization module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def verify_source_identity(dispatch: object) -> None:
    manifest = dispatch.load_manifest()
    source = load_json(PILOT / "source_snapshot.json")
    if source.get("snapshot_id") != manifest["source_snapshot"]:
        fail("source snapshot identity mismatch")
    if source.get("source_count") != len(source.get("sources", [])):
        fail("source snapshot count mismatch")
    if {item.get("id") for item in source["sources"]} != SOURCE_IDS:
        fail("source snapshot IDs mismatch frozen inventory")
    for name in ("preregistration_commit", "reviewed_context_revision"):
        if subprocess.check_output(
            ["git", "cat-file", "-t", manifest[name]],
            cwd=dispatch.ROOT,
            text=True,
        ).strip() != "commit":
            fail(f"{name} is not a commit")


def verify_launcher_manifest(dispatch: object) -> None:
    launcher = load_launcher_module()
    contract = dispatch.load_manifest().get("launcher")
    if not isinstance(contract, dict):
        fail("launcher contract is missing")
    if contract.get("argv") != launcher.build_invocation("codex"):
        fail("launcher argv is not the pinned direct codex invocation")
    if contract.get("shell") is not False or contract.get("tool_use") is not False or contract.get("source_reopening") is not False:
        fail("launcher contract permits shell/tool/source access")


def load_runs(dispatch: object) -> tuple[dict, list[dict]]:
    dispatch.verify_artifacts()
    manifest = load_json(RUN_MANIFEST)
    if manifest.get("schema") != "cyax-0163-clean-rerun-runs-v1":
        fail("unsupported run manifest schema")
    if tuple(manifest.get("launch_order", [])) != RUNS:
        fail("launch order differs from frozen A B B A A B order")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or tuple(item.get("run_id") for item in entries) != RUNS:
        fail("run manifest entries/order mismatch")
    dispatch_manifest = dispatch.load_manifest()
    expected_hashes = {condition: dispatch_manifest["inputs"][condition]["sha256"] for condition in ("A", "B")}
    for item in entries:
        run_id = item.get("run_id")
        condition = item.get("condition")
        if condition not in expected_hashes or run_id[0] != condition:
            fail(f"condition mapping mismatch for {run_id}")
        if item.get("input_sha256") != expected_hashes[condition]:
            fail(f"preregistered input hash mismatch in run {run_id}")
        path = RUNS_DIR / item["input_artifact"]
        if dispatch.sha256(path.read_bytes()) != item["input_sha256"]:
            fail(f"actual input hash mismatch in run {run_id}")
    for condition in ("A", "B"):
        hashes = {item["input_sha256"] for item in entries if item["condition"] == condition}
        if len(hashes) != 1:
            fail(f"replicate input hashes differ for condition {condition}")
    return manifest, entries


def verify_responses(manifest: dict, entries: list[dict]) -> None:
    for item in entries:
        status = item.get("status")
        response_meta = item.get("response")
        path = RUNS_DIR / item["response_artifact"]
        if status in {"NOT_LAUNCHED", "DISPATCHED"}:
            if response_meta is not None or path.exists():
                fail(f"response present before capture for {item['run_id']}")
            continue
        if status != "CAPTURED" or not isinstance(response_meta, dict):
            fail(f"invalid response status/metadata for {item['run_id']}")
        if not path.is_file():
            fail(f"missing response artifact for {item['run_id']}")
        data = path.read_bytes()
        if not data:
            fail(f"empty response artifact for {item['run_id']}")
        observed = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        expected = {"bytes": response_meta.get("bytes"), "sha256": response_meta.get("sha256")}
        if observed != expected:
            fail(f"response hash/byte count mismatch for {item['run_id']}")
        if response_meta.get("completion_status") != "COMPLETE":
            fail(f"incomplete response status for {item['run_id']}")
        if response_meta.get("source_reopenings") != 0:
            fail(f"source reopening recorded for {item['run_id']}")


def stable_identity(result: dict, audit: dict) -> str | None:
    """Select the stable fresh-process identity actually present in events."""
    for value in (
        result.get("thread_id"),
        result.get("session_id"),
        audit.get("thread_id"),
        audit.get("session_id"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def require_unique_fresh_identities(results: list[tuple[dict, dict]]) -> list[str]:
    identities = []
    identity_module = load_identity_module()
    for result, audit in results:
        identity = stable_identity(result, audit)
        if identity is None:
            fail("captured run has no fresh thread/session identity")
        if not identity_module.is_pseudonym(identity):
            fail("captured run identity is not a SHA-256 pseudonym")
        identities.append(identity)
    if len(identities) != 6 or len(set(identities)) != 6:
        fail("captured runs do not have six unique fresh thread/session identities")
    return identities


def verify_launch_contract(dispatch: object, entries: list[dict]) -> None:
    """Verify direct invocation, sanitized event capture, and no-tool audit."""
    launcher = load_launcher_module()
    identity_module = load_identity_module()
    expected_argv = dispatch.load_manifest()["launcher"]["argv"]
    captured_identities: list[tuple[dict, dict]] = []
    for item in entries:
        if item.get("status") != "CAPTURED":
            continue
        invocation = item.get("invocation")
        if not isinstance(invocation, dict):
            fail(f"missing launcher invocation for {item['run_id']}")
        if invocation.get("argv") != expected_argv or invocation.get("shell") is not False:
            fail(f"launcher invocation changed for {item['run_id']}")
        required = {
            "model": "gpt-5.6-sol",
            "reasoning": "high",
            "sandbox": "read-only",
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "skip_git_repo_check": True,
        }
        if any(invocation.get(key) != value for key, value in required.items()):
            fail(f"launcher isolation/model settings changed for {item['run_id']}")
        encoded = json.dumps(invocation, sort_keys=True, separators=(",", ":")).encode()
        if item.get("invocation_sha256") != hashlib.sha256(encoded).hexdigest():
            fail(f"launcher invocation hash mismatch for {item['run_id']}")
        result = item.get("launch_result")
        if not isinstance(result, dict) or result.get("returncode") != 0:
            fail(f"codex process did not complete successfully for {item['run_id']}")
        event_path = RUNS_DIR / item["event_artifact"]
        if not event_path.is_file():
            fail(f"event capture missing for {item['run_id']}")
        event_bytes = event_path.read_bytes()
        try:
            identity_module.verify_sanitized_event_stream(event_bytes)
        except ValueError as exc:
            fail(f"unsanitized identity in event capture for {item['run_id']}: {exc}")
        event_meta = result.get("event")
        if event_meta != {"bytes": len(event_bytes), "sha256": hashlib.sha256(event_bytes).hexdigest()}:
            fail(f"event capture hash mismatch for {item['run_id']}")
        original_meta = result.get("event_original")
        if not isinstance(original_meta, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(original_meta.get("sha256", ""))):
            fail(f"original event provenance missing for {item['run_id']}")
        privacy = result.get("identity_sanitization")
        if not isinstance(privacy, dict) or privacy.get("schema") != identity_module.SCHEMA:
            fail(f"identity sanitization record missing for {item['run_id']}")
        if privacy.get("sanitized_sha256") != event_meta["sha256"] or privacy.get("original_sha256") != original_meta["sha256"]:
            fail(f"identity sanitization hash provenance mismatch for {item['run_id']}")
        stderr_meta = result.get("stderr")
        if not isinstance(stderr_meta, dict) or set(stderr_meta) != {"bytes", "sha256"}:
            fail(f"stderr metadata missing for {item['run_id']}")
        audit = launcher.audit_events(event_bytes)
        if not audit["valid_jsonl"] or audit["forbidden_events"] or audit["source_reopenings"] != 0:
            fail(f"tool/source-reopening event detected for {item['run_id']}")
        if result.get("event_audit") != audit:
            fail(f"event audit record mismatch for {item['run_id']}")
        identity = stable_identity(result, audit)
        if identity is None:
            fail(f"fresh thread/session identity missing for {item['run_id']}")
        if not identity_module.is_pseudonym(identity):
            fail(f"fresh identity is not a privacy pseudonym for {item['run_id']}")
        captured_identities.append((result, audit))
        if result.get("response"):
            response_meta = result["response"]
            if response_meta.get("sha256") != item["response"].get("sha256") or response_meta.get("bytes") != item["response"].get("bytes"):
                fail(f"final response capture mismatch for {item['run_id']}")
    require_unique_fresh_identities(captured_identities)


def verify_blind_mapping(entries: list[dict]) -> dict[str, str]:
    mapping_doc = load_json(MAPPING)
    if mapping_doc.get("schema") != "cyax-0163-clean-rerun-blind-mapping-v1":
        fail("unsupported blind mapping schema")
    mapping = mapping_doc.get("mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(OPAQUE) or set(mapping.values()) != set(RUNS):
        fail("blind mapping is not a one-to-one S1-S6 mapping")
    conditions = {item["run_id"]: item["condition"] for item in entries}
    for opaque, run_id in mapping.items():
        if conditions[run_id] not in {"A", "B"}:
            fail(f"blind mapping points to unknown run {run_id}")
        if re.search(r"(?:condition|relational|non[-_ ]?graph|(?:^|[^A-Z])[AB][1-3](?:[^0-9]|$))", opaque, re.I):
            fail(f"condition-bearing opaque ID {opaque}")
    return mapping


def validate_scorecard(card: dict, opaque: str) -> None:
    expected_fields = {
        "opaque_id", "scores", "raw_total", "conflict_critical", "automatic_failure",
        "automatic_failure_reasons", "unsupported_assertions", "authority_errors",
        "temporal_errors", "supersession_errors", "unjustified_inferences",
        "correct_abstentions", "incorrect_abstentions", "next_action_correct", "source_reopenings",
    }
    if set(card) != expected_fields:
        fail(f"scorecard {opaque} metadata/field set is not frozen")
    if card["opaque_id"] != opaque or set(card["scores"]) != set(SCORE_KEYS):
        fail(f"scorecard {opaque} identity/keys mismatch")
    if any(type(card["scores"][key]) is not int or card["scores"][key] not in (0, 1) for key in SCORE_KEYS):
        fail(f"scorecard {opaque} contains non-binary score")
    if card["raw_total"] != sum(card["scores"].values()):
        fail(f"scorecard {opaque} total mismatch")
    if card["conflict_critical"] != sum(card["scores"][key] for key in CRITICAL):
        fail(f"scorecard {opaque} conflict-critical total mismatch")
    for key in ("raw_total", "conflict_critical", "unsupported_assertions", "authority_errors", "temporal_errors", "supersession_errors", "unjustified_inferences", "correct_abstentions", "incorrect_abstentions", "source_reopenings"):
        if type(card[key]) is not int or card[key] < 0:
            fail(f"scorecard {opaque} has invalid {key}")
    if type(card["automatic_failure"]) is not bool or type(card["next_action_correct"]) is not bool:
        fail(f"scorecard {opaque} has invalid boolean")
    if not isinstance(card["automatic_failure_reasons"], list) or any(not isinstance(x, str) for x in card["automatic_failure_reasons"]):
        fail(f"scorecard {opaque} has invalid failure reasons")
    if card["automatic_failure"] != bool(card["automatic_failure_reasons"]):
        fail(f"scorecard {opaque} failure flag mismatch")


def verify_scoring(mapping: dict[str, str], require_complete: bool) -> None:
    responses_dir = HERE / "scoring" / "blind_responses"
    scorecards_dir = HERE / "scoring" / "blind_scorecards"
    response_paths = list(responses_dir.glob("*")) if responses_dir.exists() else []
    scorecard_paths = list(scorecards_dir.glob("*")) if scorecards_dir.exists() else []
    if any(path.suffix != ".response" or path.stem not in OPAQUE for path in response_paths):
        fail("condition-bearing or unknown scorer-facing response filename detected")
    if any(path.suffix != ".json" or path.stem not in OPAQUE for path in scorecard_paths):
        fail("condition-bearing or unknown scorecard filename detected")
    response_files = {path.stem for path in response_paths}
    scorecard_files = {path.stem for path in scorecard_paths}
    if response_files - set(OPAQUE) or scorecard_files - set(OPAQUE):
        fail("condition-bearing or unknown scorer-facing filename detected")
    if require_complete and (response_files != set(OPAQUE) or scorecard_files != set(OPAQUE)):
        fail("blind responses or scorecards are incomplete")
    for opaque in sorted(scorecard_files):
        card = load_json(scorecards_dir / f"{opaque}.json")
        validate_scorecard(card, opaque)


def verify_blind_materialization(entries: list[dict]) -> None:
    """Verify opaque copies and leakage metadata without exposing the mapping."""
    responses_dir = HERE / "scoring" / "blind_responses"
    if not responses_dir.exists():
        return
    if not BLIND_MANIFEST.is_file() or not LEAKAGE_REPORT.is_file():
        fail("blind response directory requires immutable manifest and leakage report")
    manifest = load_json(BLIND_MANIFEST)
    if manifest.get("schema") != "cyax-0163-blind-materialization-v1":
        fail("unsupported blind materialization schema")
    items = manifest.get("items")
    if not isinstance(items, dict) or set(items) != set(OPAQUE):
        fail("blind materialization manifest is incomplete")
    runs = {item["run_id"]: item for item in entries}
    leakage = load_json(LEAKAGE_REPORT)
    if leakage.get("schema") != "cyax-0163-blind-content-leakage-v1":
        fail("unsupported blind leakage schema")
    for opaque in OPAQUE:
        item = items[opaque]
        run_id = item.get("source_run")
        if run_id not in runs:
            fail(f"blind materialization points to unknown run: {opaque}")
        source = RUNS_DIR / runs[run_id]["response_artifact"]
        target = HERE / item.get("opaque_artifact", "")
        if not source.is_file() or not target.is_file():
            fail(f"blind response source/target missing: {opaque}")
        source_bytes = source.read_bytes()
        target_bytes = target.read_bytes()
        if source_bytes != target_bytes:
            fail(f"blind response was changed: {opaque}")
        for prefix, data in (("source", source_bytes), ("opaque", target_bytes)):
            if item.get(f"{prefix}_bytes") != len(data) or item.get(f"{prefix}_sha256") != hashlib.sha256(data).hexdigest():
                fail(f"blind {prefix} hash mismatch: {opaque}")
        if item.get("byte_identical") is not True:
            fail(f"blind byte-identity flag missing: {opaque}")
        leak = leakage.get("items", {}).get(opaque)
        if not isinstance(leak, dict) or leak.get("sha256") != hashlib.sha256(target_bytes).hexdigest():
            fail(f"blind leakage hash mismatch: {opaque}")


def verify_scorer_input() -> None:
    """Verify a materialized scorer stdin artifact has no control metadata."""
    if not SCORER_INPUT.exists():
        return
    if not SCORER_INPUT_MANIFEST.is_file():
        fail("scorer input requires its immutable manifest")
    manifest = load_json(SCORER_INPUT_MANIFEST)
    if manifest.get("schema") != "cyax-0163-blind-scorer-input-v1":
        fail("unsupported scorer input schema")
    data = SCORER_INPUT.read_bytes()
    if manifest.get("input") != {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}:
        fail("scorer input hash/byte count mismatch")
    if manifest.get("opaque_ids") != list(OPAQUE):
        fail("scorer input opaque order changed")
    for forbidden in (b"condition_mapping.json", b"runs/manifest.json", b"dispatch_manifest.json", b"source_snapshot.json"):
        if forbidden in data:
            fail(f"control metadata leaked into scorer input: {forbidden!r}")
    for opaque in OPAQUE:
        if b"--- BEGIN BLIND RESPONSE " + opaque.encode() + b" ---" not in data:
            fail(f"scorer input is missing opaque response section: {opaque}")


def verify_scorer_capture_privacy() -> None:
    """Verify a captured scorer stream contains only hashed identities."""
    if not SCORER_EVENTS.exists() and not SCORER_RUN.exists():
        return
    if not SCORER_EVENTS.is_file() or not SCORER_RUN.is_file():
        fail("scorer event/run privacy artifacts must be paired")
    identity = load_identity_module()
    launcher = load_launcher_module()
    event_bytes = SCORER_EVENTS.read_bytes()
    try:
        identity.verify_sanitized_event_stream(event_bytes)
    except ValueError as exc:
        fail(f"unsanitized identity in scorer events: {exc}")
    record = load_json(SCORER_RUN)
    result = record.get("result")
    if not isinstance(result, dict):
        fail("scorer run has no result")
    event_meta = result.get("event")
    if event_meta != {"bytes": len(event_bytes), "sha256": hashlib.sha256(event_bytes).hexdigest()}:
        fail("scorer event hash mismatch")
    original_meta = result.get("event_original")
    privacy = result.get("identity_sanitization")
    if not isinstance(original_meta, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(original_meta.get("sha256", ""))):
        fail("scorer original event provenance missing")
    if not isinstance(privacy, dict) or privacy.get("schema") != identity.SCHEMA:
        fail("scorer identity sanitization record missing")
    if privacy.get("sanitized_sha256") != event_meta["sha256"] or privacy.get("original_sha256") != original_meta["sha256"]:
        fail("scorer identity sanitization hash provenance mismatch")
    audit = launcher.audit_events(event_bytes)
    if result.get("event_audit") != audit or audit["forbidden_events"] or not audit["valid_jsonl"]:
        fail("scorer event audit mismatch or tool/source reopening detected")
    scorer_identity = stable_identity(result, audit)
    if scorer_identity is None or not identity.is_pseudonym(scorer_identity):
        fail("scorer fresh identity is missing or not a privacy pseudonym")


def verify_final_immutability() -> None:
    result = HERE / "final_result.json"
    record = HERE / "final_result.sha256"
    if not result.exists() and not record.exists():
        return
    if not result.is_file() or not record.is_file():
        fail("final result must have both result and immutable hash record")
    expected = record.read_text(encoding="ascii").strip()
    observed = hashlib.sha256(result.read_bytes()).hexdigest()
    if expected != observed or not re.fullmatch(r"[0-9a-f]{64}", expected):
        fail("final result hash record mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre-dispatch", "captured", "scored"), default="pre-dispatch")
    args = parser.parse_args()
    dispatch = load_dispatch_module()
    verify_source_identity(dispatch)
    verify_launcher_manifest(dispatch)
    manifest, entries = load_runs(dispatch)
    if args.phase == "pre-dispatch" and any(item["status"] != "NOT_LAUNCHED" for item in entries):
        fail("pre-dispatch verification requires all runs to remain NOT_LAUNCHED")
    if args.phase in {"captured", "scored"} and any(item["status"] != "CAPTURED" for item in entries):
        fail(f"{args.phase} verification requires all six responses to be CAPTURED")
    verify_responses(manifest, entries)
    if args.phase in {"captured", "scored"}:
        verify_launch_contract(dispatch, entries)
    mapping = verify_blind_mapping(entries)
    verify_blind_materialization(entries)
    verify_scorer_input()
    verify_scorer_capture_privacy()
    verify_scoring(mapping, require_complete=args.phase == "scored")
    verify_final_immutability()
    print(f"PASS: {args.phase} exact-dispatch verification; six frozen runs, replicate hashes, response/scoring contracts checked")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
