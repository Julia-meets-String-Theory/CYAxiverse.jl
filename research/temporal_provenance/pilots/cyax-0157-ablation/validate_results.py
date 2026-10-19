#!/usr/bin/env python3
"""Validate the frozen CYAX-0157 held-out result without network access.

This validator reads the frozen run, scorecard, mapping, context-accounting,
and result narrative artifacts. It never edits them and treats the scorecard
as blind evidence until the explicit mapping file is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


PILOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PILOT_DIR.parents[3]
RUNS_MANIFEST_PATH = PILOT_DIR / "runs" / "manifest.json"
SCORECARD_PATH = PILOT_DIR / "scoring" / "blind_scorecard.json"
MAPPING_PATH = PILOT_DIR / "scoring" / "condition_mapping.json"
RESULTS_PATH = PILOT_DIR / "results.md"
METHODOLOGY_REVIEW_PATH = PILOT_DIR / "methodology_review.md"
PHASE2_MANIFEST_PATH = PILOT_DIR / "phase2_manifest.json"
EXPECTED_ORDER = ["A1", "B1", "B2", "A2"]
EXPECTED_RUNS = set(EXPECTED_ORDER)
EXPECTED_FACTS = [f"K{i}" for i in range(1, 13)]
EXPECTED_MODEL = "gpt-5.6-sol"
EXPECTED_REASONING = "high"
SCORE_FREEZE = "c52c4e0"

PRIVATE_PATTERNS = (
    (re.compile(r"/(?:Users|home)/[^\s`'\"]+"), "absolute home path"),
    (re.compile(r"(?:^|[\s'\"`])~/(?:[^\s`'\"]*)"), "home-relative path"),
    (re.compile(r"(?:[A-Za-z]:\\|\\\\[^\\]+\\)"), "Windows path"),
    (re.compile(r"file://", re.IGNORECASE), "file URL"),
    (re.compile(r"(?:gh[pousr]_|sk-)[A-Za-z0-9_\-]{20,}"), "credential pattern"),
)


class ValidationError(ValueError):
    """Raised when a frozen result contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load {path.relative_to(PILOT_DIR)}: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def git_bytes(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )


def resolve_commit(ref: str) -> str:
    result = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    require(result.returncode == 0, f"Git commit is unavailable: {ref}")
    return result.stdout.strip()


def require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = git("merge-base", "--is-ancestor", ancestor, descendant)
    require(result.returncode == 0, f"Git chronology failed for {label}")


def verify_frozen_paths(ref: str, paths: list[Path], label: str) -> None:
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        exists = git_bytes("cat-file", "-e", f"{ref}:{relative}")
        require(exists.returncode == 0, f"{label}: {relative} is absent at freeze {ref}")
        frozen = git_bytes("show", f"{ref}:{relative}")
        require(frozen.returncode == 0, f"{label}: cannot read {relative} at freeze {ref}")
        require(path.read_bytes() == frozen.stdout, f"{label}: {relative} changed since freeze {ref}")


def verify_freeze_boundaries(
    prereg_commit: str,
    context_commit: str,
    response_commit: str,
    score_commit: str,
    runs_manifest: dict[str, Any],
) -> None:
    prereg_paths = [
        PILOT_DIR / "preregistration.md",
        PILOT_DIR / "source_snapshot.json",
        PILOT_DIR / "answer_key.md",
    ]
    context_paths = [
        PILOT_DIR / "condition_a_context.md",
        PILOT_DIR / "condition_b_context.md",
        PILOT_DIR / "condition_a_ledger.jsonl",
        PILOT_DIR / "common_subject_prompt.md",
        PHASE2_MANIFEST_PATH,
    ]
    response_paths = [
        RUNS_MANIFEST_PATH,
        *[PILOT_DIR / runs_manifest["runs"][run_name]["response"] for run_name in EXPECTED_ORDER],
    ]
    score_paths = [SCORECARD_PATH, MAPPING_PATH]
    verify_frozen_paths(prereg_commit, prereg_paths, "preregistration freeze")
    verify_frozen_paths(context_commit, context_paths, "context freeze")
    verify_frozen_paths(response_commit, response_paths, "response freeze")
    verify_frozen_paths(score_commit, score_paths, "score freeze")


def validate_response_entry(run_name: str, entry: dict[str, Any]) -> None:
    required = {"condition", "context", "response", "response_bytes", "response_sha256", "response_words", "source_reopenings"}
    require(required <= set(entry), f"{run_name}: response manifest fields are incomplete")
    response_path = PILOT_DIR / entry["response"]
    require(response_path.is_file(), f"{run_name}: response file is missing")
    data = response_path.read_bytes()
    text = data.decode("utf-8")
    require(len(data) == entry["response_bytes"], f"{run_name}: response byte count mismatch")
    require(sha256_bytes(data) == entry["response_sha256"], f"{run_name}: response SHA-256 mismatch")
    require(len(text.split()) == entry["response_words"], f"{run_name}: response word count mismatch")
    require(entry["source_reopenings"] == 0, f"{run_name}: source reopening count is not zero")
    require(entry["condition"] in {"A", "B"}, f"{run_name}: invalid condition identity")
    require(entry["context"] in {"condition_a_context.md", "condition_b_context.md"}, f"{run_name}: invalid context identity")
    expected_context = "condition_a_context.md" if entry["condition"] == "A" else "condition_b_context.md"
    require(entry["context"] == expected_context, f"{run_name}: condition/context mismatch")


def validate_run_manifest(manifest: dict[str, Any]) -> None:
    require(manifest.get("launch_order") == EXPECTED_ORDER, "run launch order is not ABBA")
    require(manifest.get("model") == EXPECTED_MODEL, "run model is not the frozen gpt-5.6-sol setting")
    require(manifest.get("reasoning_effort") == EXPECTED_REASONING, "run reasoning setting is not the frozen high setting")
    require(manifest.get("responses_frozen_before_scoring") is True, "responses were not frozen before scoring")
    runs = manifest.get("runs")
    require(isinstance(runs, dict) and set(runs) == EXPECTED_RUNS, "run manifest must contain exactly A1, B1, B2, A2")
    for run_name in EXPECTED_ORDER:
        validate_response_entry(run_name, runs[run_name])
    require([runs[name]["condition"] for name in EXPECTED_ORDER] == ["A", "B", "B", "A"], "ABBA condition identities are inconsistent")


def validate_context_accounting(runs_manifest: dict[str, Any], phase2_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contexts = phase2_manifest.get("contexts")
    require(isinstance(contexts, dict) and set(contexts) == {"condition_a", "condition_b"}, "Phase 2 context accounting is incomplete")
    for condition_name, condition_letter in (("condition_a", "A"), ("condition_b", "B")):
        record = contexts[condition_name]
        path = PILOT_DIR / record["path"]
        data = path.read_bytes()
        text = data.decode("utf-8")
        require(record["byte_count"] == len(data), f"{condition_name}: byte accounting mismatch")
        require(record["sha256"] == sha256_bytes(data), f"{condition_name}: context SHA-256 mismatch")
        require(record["word_count"] == len(text.split()), f"{condition_name}: word accounting mismatch")
        require(record["fact_coverage"] == EXPECTED_FACTS, f"{condition_name}: fact coverage is not K1-K12")
        for run_name, run in runs_manifest["runs"].items():
            if run["condition"] == condition_letter:
                require(run["context"] == record["path"], f"{run_name}: run context does not match Phase 2 accounting")
    require(contexts["condition_a"]["source_references"] == contexts["condition_b"]["source_references"], "condition source accounting is asymmetric")
    require(contexts["condition_a"]["evidence_items"] == contexts["condition_b"]["evidence_items"], "condition evidence accounting is asymmetric")
    require(phase2_manifest.get("source_reopening_count") == 0, "Phase 2 source reopening count is not zero")
    return contexts


def validate_scorecard(scorecard: dict[str, Any]) -> None:
    require(scorecard.get("automatic_failure_checks_applied_first") is True, "automatic failures were not checked first")
    require(scorecard.get("blind_scoring_complete_before_mapping_reveal") is True, "blind scorecard was not frozen before mapping reveal")
    require(scorecard.get("rubric_items") == EXPECTED_FACTS, "scorecard rubric is not K1-K12")
    require(scorecard.get("scorer_model") == EXPECTED_MODEL, "scorecard model is not frozen")
    require(scorecard.get("scorer_reasoning_effort") == "xhigh", "scorecard reasoning setting is not frozen xhigh")
    opaque = scorecard.get("opaque_responses")
    require(isinstance(opaque, dict) and set(opaque) == {"R1", "R2", "R3", "R4"}, "scorecard must contain exactly four opaque responses")
    for response_name, record in opaque.items():
        require(isinstance(record, dict), f"{response_name}: scorecard record is not an object")
        require(record.get("automatic_failures") == [], f"{response_name}: automatic failure recorded")
        scores = record.get("k_scores")
        require(isinstance(scores, list) and len(scores) == 12, f"{response_name}: score vector must contain 12 values")
        require(all(value in (0, 1) for value in scores), f"{response_name}: score vector is not binary")
        require(isinstance(record.get("score"), int) and 0 <= record["score"] <= 12, f"{response_name}: invalid total score")
        require(sum(scores) == record["score"], f"{response_name}: score does not equal binary K sum")
        require(isinstance(record.get("opaque_word_count"), int) and record["opaque_word_count"] >= 0, f"{response_name}: invalid opaque word count")
        for field in ("authority_mistakes", "correct_abstentions", "incorrect_abstentions", "material_provenance_errors", "next_action_mistakes", "supersession_mistakes", "unsupported_assertions"):
            require(isinstance(record.get(field), list), f"{response_name}: scorecard field {field} is not a list")


def validate_mapping(mapping: dict[str, Any], scorecard: dict[str, Any], runs_manifest: dict[str, Any]) -> dict[str, str]:
    require(mapping.get("mapping_revealed_after_blind_scorecard_frozen") is True, "mapping was revealed before blind scorecard freeze")
    opaque_to_run = mapping.get("opaque_to_run")
    require(isinstance(opaque_to_run, dict) and set(opaque_to_run) == {"R1", "R2", "R3", "R4"}, "opaque-to-run mapping is incomplete")
    require(set(opaque_to_run.values()) == EXPECTED_RUNS, "opaque-to-run mapping is not bijective")
    hashes = mapping.get("response_sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(opaque_to_run), "mapping response hashes are incomplete")
    for opaque_name, run_name in opaque_to_run.items():
        expected_hash = runs_manifest["runs"][run_name]["response_sha256"]
        require(hashes[opaque_name] == expected_hash, f"{opaque_name}: mapped response SHA-256 mismatch")
        require(scorecard["opaque_responses"][opaque_name]["score"] == sum(scorecard["opaque_responses"][opaque_name]["k_scores"]), f"{opaque_name}: mapped score is not self-consistent")
    return opaque_to_run


def validate_means(opaque_to_run: dict[str, str], scorecard: dict[str, Any], runs_manifest: dict[str, Any]) -> tuple[float, float, float]:
    scores_by_run = {
        run_name: scorecard["opaque_responses"][opaque_name]["score"]
        for opaque_name, run_name in opaque_to_run.items()
    }
    a_runs = [run_name for run_name, record in runs_manifest["runs"].items() if record["condition"] == "A"]
    b_runs = [run_name for run_name, record in runs_manifest["runs"].items() if record["condition"] == "B"]
    a_mean = sum(scores_by_run[run_name] for run_name in a_runs) / len(a_runs)
    b_mean = sum(scores_by_run[run_name] for run_name in b_runs) / len(b_runs)
    difference = abs(a_mean - b_mean)
    require(a_mean == 12.0 and b_mean == 12.0 and difference == 0.0, f"mapped means are not 12.0/12 and 0.0: {a_mean}, {b_mean}, {difference}")
    return a_mean, b_mean, difference


def response_freeze_from_results(text: str) -> str:
    match = re.search(r"^- Response freeze: `([0-9a-f]{40})`$", text, re.MULTILINE)
    require(match is not None, "results.md does not record a full response-freeze commit")
    return match.group(1)


def freeze_refs_from_results(results_text: str) -> dict[str, str]:
    patterns = {
        "prereg": r"^- Preregistration freeze: `([0-9a-f]{40})`$",
        "context": r"^- Released context revision: `([0-9a-f]{40})`$",
        "response": r"^- Response freeze: `([0-9a-f]{40})`$",
        "score": r"score freeze\s+`([0-9a-f]{40})`",
    }
    refs: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, results_text, re.IGNORECASE | re.MULTILINE)
        require(match is not None, f"results.md does not record a full {name} freeze commit")
        refs[name] = match.group(1)
    return refs


def validate_chronology(runs_manifest: dict[str, Any], results_text: str) -> dict[str, str]:
    refs = freeze_refs_from_results(results_text)
    prereg_commit = resolve_commit(refs["prereg"])
    context_commit = resolve_commit(refs["context"])
    response_commit = resolve_commit(refs["response"])
    score_commit = resolve_commit(refs["score"])
    expected_score_commit = resolve_commit(SCORE_FREEZE)
    head_commit = resolve_commit("HEAD")
    require(prereg_commit == resolve_commit("b1eea67"), "results.md preregistration freeze is not b1eea67")
    require(context_commit == resolve_commit(runs_manifest["context_commit"]), "results.md context freeze disagrees with run manifest")
    require(score_commit == expected_score_commit, "results.md score freeze is not c52c4e0")
    require(prereg_commit, "preregistration freeze commit is empty")
    require_ancestor(prereg_commit, context_commit, "preregistration freeze before context freeze")
    require_ancestor(context_commit, response_commit, "context freeze before response freeze")
    require_ancestor(response_commit, score_commit, "response freeze before score freeze")
    require_ancestor(score_commit, head_commit, "score freeze before current checkout")
    verify_freeze_boundaries(prereg_commit, context_commit, response_commit, score_commit, runs_manifest)
    return {"prereg": prereg_commit, "context": context_commit, "response": response_commit, "score": score_commit}


def validate_results_text(results_text: str, runs_manifest: dict[str, Any], contexts: dict[str, dict[str, Any]], means: tuple[float, float, float]) -> None:
    required_phrases = (
        "Independent pre-run semantic-parity decision: PASS",
        "Model and reasoning for all subjects: GPT-5.6 Sol / high",
        "Launch order: A1, B1, B2, A2",
        "Source reopenings: zero for every run",
        "Condition B matched Condition A",
        "not specifically to graph-shaped representation",
        "No graph backend is selected.",
    )
    for phrase in required_phrases:
        require(phrase in results_text, f"results.md is missing required claim: {phrase}")
    require("A mean: 12.0/12" in results_text and "B mean: 12.0/12" in results_text, "results.md means are inconsistent")
    require("Difference: 0.0 points" in results_text, "results.md condition difference is inconsistent")
    require("Automatic failures" in results_text, "results.md omits automatic-failure accounting")
    for run_name in EXPECTED_ORDER:
        record = runs_manifest["runs"][run_name]
        condition = record["condition"]
        row = f"| {run_name} | {condition} | 12/12 | None | 0 | {record['response_words']:,} |"
        require(row in results_text, f"results.md row is inconsistent for {run_name}")
    for condition_name in ("condition_a", "condition_b"):
        record = contexts[condition_name]
        require(f"| {'A' if condition_name == 'condition_a' else 'B'} |" in results_text, f"results.md omits {condition_name} accounting")
        require(f"{record['word_count']:,}" in results_text, f"results.md word accounting is inconsistent for {condition_name}")
        require(f"{record['byte_count']:,}" in results_text, f"results.md byte accounting is inconsistent for {condition_name}")
        require(record["sha256"] in results_text, f"results.md SHA-256 accounting is inconsistent for {condition_name}")
    require(means == (12.0, 12.0, 0.0), "computed means do not support the frozen result narrative")


def privacy_targets(runs_manifest: dict[str, Any]) -> list[Path]:
    paths = [RESULTS_PATH, METHODOLOGY_REVIEW_PATH, RUNS_MANIFEST_PATH, SCORECARD_PATH, MAPPING_PATH, PHASE2_MANIFEST_PATH]
    paths.extend(PILOT_DIR / runs_manifest["runs"][run_name]["response"] for run_name in EXPECTED_ORDER)
    return paths


def validate_privacy(runs_manifest: dict[str, Any]) -> None:
    for path in privacy_targets(runs_manifest):
        text = path.read_text(encoding="utf-8")
        for pattern, label in PRIVATE_PATTERNS:
            match = pattern.search(text)
            require(match is None, f"{path.relative_to(PILOT_DIR)} contains {label}")


def validate_all() -> None:
    runs_manifest = load_json(RUNS_MANIFEST_PATH)
    scorecard = load_json(SCORECARD_PATH)
    mapping_file = load_json(MAPPING_PATH)
    phase2_manifest = load_json(PHASE2_MANIFEST_PATH)
    results_text = RESULTS_PATH.read_text(encoding="utf-8")
    validate_run_manifest(runs_manifest)
    contexts = validate_context_accounting(runs_manifest, phase2_manifest)
    validate_scorecard(scorecard)
    opaque_to_run = validate_mapping(mapping_file, scorecard, runs_manifest)
    means = validate_means(opaque_to_run, scorecard, runs_manifest)
    chronology = validate_chronology(runs_manifest, results_text)
    validate_results_text(results_text, runs_manifest, contexts, means)
    validate_privacy(runs_manifest)
    print("results validate: PASS")
    print(f"runs=4 order={'/'.join(EXPECTED_ORDER)} model={EXPECTED_MODEL} reasoning={EXPECTED_REASONING}")
    print("scores=A1 12/12, A2 12/12, B1 12/12, B2 12/12; automatic_failures=0; reopenings=0")
    print(f"mapped_means=A {means[0]:.1f}/12, B {means[1]:.1f}/12, difference={means[2]:.1f}")
    print(f"chronology=context {chronology['context']}, responses {chronology['response']}, score {chronology['score']}")


def expect_failure(function: Callable[[], Any], label: str) -> None:
    try:
        function()
    except ValidationError:
        return
    raise ValidationError(f"self-test did not reject {label}")


def self_test() -> None:
    validate_all()
    expect_failure(
        lambda: require_response_metrics_for_test(b"wrong", {"response_bytes": 99, "response_sha256": "0" * 64, "response_words": 1}),
        "response byte/hash mismatch",
    )
    expect_failure(lambda: require_binary_score_for_test([1] * 11), "short score vector")
    expect_failure(lambda: require_bijective_mapping_for_test({"R1": "A1", "R2": "A1"}), "non-bijective mapping")
    print("results self-test: PASS")


def require_response_metrics_for_test(data: bytes, record: dict[str, Any]) -> None:
    require(len(data) == record["response_bytes"], "test byte mismatch")
    require(sha256_bytes(data) == record["response_sha256"], "test hash mismatch")
    require(len(data.decode("utf-8").split()) == record["response_words"], "test word mismatch")


def require_binary_score_for_test(scores: list[int]) -> None:
    require(len(scores) == 12 and all(value in (0, 1) for value in scores), "test score vector invalid")


def require_bijective_mapping_for_test(mapping: dict[str, str]) -> None:
    require(len(mapping) == len(set(mapping.values())), "test mapping is not bijective")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("validate", "privacy", "self-test"), default="validate")
    command = parser.parse_args(argv).command if argv is not None else parser.parse_args().command
    try:
        runs_manifest = load_json(RUNS_MANIFEST_PATH)
        if command == "validate":
            validate_all()
        elif command == "privacy":
            validate_run_manifest(runs_manifest)
            validate_privacy(runs_manifest)
            print("results privacy: PASS")
        else:
            self_test()
    except (ValidationError, OSError, UnicodeError) as exc:
        print(f"results {command}: FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
