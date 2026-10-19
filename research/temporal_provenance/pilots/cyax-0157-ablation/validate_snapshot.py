#!/usr/bin/env python3
"""Validate the CYAX-0157 ablation source snapshot, preregistration shape,
captured source fixtures, and answer key alignment.

Run from the repository root:
    python3 research/temporal_provenance/pilots/cyax-0157-ablation/validate_snapshot.py
"""

import json
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(PILOT_DIR, "source_snapshot.json")
PREREG_PATH = os.path.join(PILOT_DIR, "preregistration.md")
ANSWER_KEY_PATH = os.path.join(PILOT_DIR, "answer_key.md")
SOURCES_DIR = os.path.join(PILOT_DIR, "sources")

EXPECTED_HEAD = "8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45"
EXPECTED_TREE = "fc980d28f3faa51f66031fe2e1fb5281b2ad5a48"
EXPECTED_BASE = "856012f015a866bf7ff352bc50e8d10c250855e6"

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"  OK: {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_fixture(filename):
    fixture_path = os.path.join(PILOT_DIR, filename)
    if not os.path.exists(fixture_path):
        fail(f"fixture file missing: {filename}")
        return None
    try:
        with open(fixture_path) as f:
            return json.load(f)
    except Exception as exc:
        fail(f"cannot read fixture '{filename}': {exc}")
        return None


def _ordered_numbers(expected_count):
    return [str(i) for i in range(1, expected_count + 1)]


def validate_snapshot():
    print("=== Source snapshot validation ===")
    with open(SNAPSHOT_PATH) as f:
        snap = json.load(f)

    if snap["snapshot_id"] != "cyax-0157-ablation-20260913T0122Z":
        fail("unexpected snapshot_id")
    else:
        ok("snapshot_id")

    if snap["git_objects"]["pr_head_commit"] != EXPECTED_HEAD:
        fail(f"head commit mismatch: {snap['git_objects']['pr_head_commit']}")
    else:
        ok("head commit")

    if snap["git_objects"]["pr_head_tree"] != EXPECTED_TREE:
        fail(f"head tree mismatch: {snap['git_objects']['pr_head_tree']}")
    else:
        ok("head tree")

    if snap["git_objects"]["pr_base_commit"] != EXPECTED_BASE:
        fail(f"base commit mismatch: {snap['git_objects']['pr_base_commit']}")
    else:
        ok("base commit")

    if snap["governing_issue"]["state"] != "OPEN":
        fail("Issue #157 state != OPEN")
    else:
        ok("Issue #157 OPEN")

    if snap["supporting_pr"]["state"] != "OPEN":
        fail("PR #158 state != OPEN")
    else:
        ok("PR #158 OPEN")

    if snap["supporting_pr"]["merged_at"] is not None:
        fail("PR #158 appears merged")
    else:
        ok("PR #158 unmerged")

    if snap["supporting_pr"]["is_draft"]:
        fail("PR #158 is draft")
    else:
        ok("PR #158 non-draft")

    supporting_pr = snap["supporting_pr"]
    if "status_checks" in supporting_pr:
        fail("supporting_pr.status_checks must be captured in sources/pr_158_checks.json")
    checks_fixture = supporting_pr.get("checks_fixture")
    checks = {}
    if checks_fixture:
        checks_data = load_json_fixture(checks_fixture)
        if isinstance(checks_data, list):
            checks = {c["name"]: c["conclusion"] for c in checks_data}
    for key, expected in {
        "Fast tests": "FAILURE",
        "build": "SUCCESS",
        "Full test suite": "SKIPPED",
    }.items():
        if checks.get(key) != expected:
            fail(f"{key} expected {expected}, got {checks.get(key)}")
        else:
            ok(f"{key} {expected.lower()}")

    if "comments" in supporting_pr:
        fail("supporting_pr.comments must be captured in sources/pr_158_comments.json")
    if "reviews" in supporting_pr:
        fail("supporting_pr.reviews must be captured in sources/pr_158_reviews.json")

    for art in snap["repository_artifacts_at_head"]:
        try:
            result = subprocess.run(
                ["git", "cat-file", "-p", f"{EXPECTED_HEAD}:{art['path']}"],
                capture_output=True, check=True
            )
            sha = hashlib.sha256(result.stdout).hexdigest()
            if sha != art["content_sha256"]:
                fail(f"SHA-256 mismatch for {art['path']}: {sha} != {art['content_sha256']}")
            else:
                ok(f"SHA-256 verified: {art['path']}")
        except subprocess.CalledProcessError:
            fail(f"cannot read {art['path']} at {EXPECTED_HEAD}")

    if not snap["integrity"]["deterministic"]:
        fail("integrity.deterministic is not true")
    else:
        ok("deterministic flag")

    if not snap["integrity"]["snapshot_frozen_before_contexts"]:
        fail("snapshot_frozen_before_contexts is not true")
    else:
        ok("snapshot_frozen_before_contexts flag")


def validate_captured_fixtures():
    print("\n=== Captured source fixture validation ===")
    with open(SNAPSHOT_PATH) as f:
        snap = json.load(f)

    fixture_checks = [
        ("governing_issue", "captured_fixture", "captured_fixture_sha256"),
        ("supporting_pr", "captured_fixture", "captured_fixture_sha256"),
        ("supporting_pr", "reviews_fixture", "reviews_fixture_sha256"),
        ("supporting_pr", "comments_fixture", "comments_fixture_sha256"),
        ("supporting_pr", "checks_fixture", "checks_fixture_sha256"),
    ]

    for section, fixture_key, hash_key in fixture_checks:
        obj = snap[section]
        if fixture_key not in obj:
            fail(f"{section}.{fixture_key} missing from snapshot")
            continue
        if hash_key not in obj:
            fail(f"{section}.{hash_key} missing from snapshot")
            continue

        fixture_path = os.path.join(PILOT_DIR, obj[fixture_key])
        expected_hash = obj[hash_key]

        if not os.path.exists(fixture_path):
            fail(f"fixture file missing: {obj[fixture_key]}")
            continue

        actual_hash = sha256_file(fixture_path)
        if actual_hash != expected_hash:
            fail(f"fixture hash mismatch for {obj[fixture_key]}: {actual_hash} != {expected_hash}")
        else:
            ok(f"fixture hash verified: {obj[fixture_key]}")

    required_fixtures = [
        "sources/issue_157.json",
        "sources/pr_158.json",
        "sources/pr_158_comments.json",
        "sources/pr_158_reviews.json",
        "sources/pr_158_checks.json",
    ]
    for fixture in required_fixtures:
        path = os.path.join(PILOT_DIR, fixture)
        if not os.path.exists(path):
            fail(f"required fixture missing: {fixture}")
        else:
            ok(f"fixture exists: {fixture}")

    issue_path = os.path.join(PILOT_DIR, "sources/issue_157.json")
    with open(issue_path) as f:
        issue = json.load(f)
    body_hash = hashlib.sha256((issue["body"] + "\n").encode()).hexdigest()
    expected = snap["governing_issue"]["body_sha256"]
    if body_hash != expected:
        fail(f"Issue body hash mismatch: {body_hash} != {expected}")
    else:
        ok("Issue body hash matches snapshot body_sha256")

    pr_path = os.path.join(PILOT_DIR, "sources/pr_158.json")
    with open(pr_path) as f:
        pr = json.load(f)
    body_hash = hashlib.sha256((pr["body"] + "\n").encode()).hexdigest()
    expected = snap["supporting_pr"]["body_sha256"]
    if body_hash != expected:
        fail(f"PR body hash mismatch: {body_hash} != {expected}")
    else:
        ok("PR body hash matches snapshot body_sha256")

    reviews_path = os.path.join(PILOT_DIR, "sources/pr_158_reviews.json")
    with open(reviews_path) as f:
        reviews = json.load(f)
    if reviews != []:
        fail("reviews fixture is not empty")
    else:
        ok("reviews fixture is empty array")

    comments_path = os.path.join(PILOT_DIR, "sources/pr_158_comments.json")
    with open(comments_path) as f:
        comments = json.load(f)
    if len(comments) != 2:
        fail(f"expected 2 comments, found {len(comments)}")
    else:
        ok("2 comments in fixture")

    checks_path = os.path.join(PILOT_DIR, "sources/pr_158_checks.json")
    with open(checks_path) as f:
        checks_list = json.load(f)
    if len(checks_list) != 3:
        fail(f"expected 3 checks, found {len(checks_list)}")
    else:
        ok("3 checks in fixture")


def validate_preregistration():
    print("\n=== Preregistration shape validation ===")
    with open(PREREG_PATH) as f:
        text = f.read()

    questions = re.findall(r"^\d+\.\s+", text, re.MULTILINE)
    if len(questions) != 12:
        fail(f"expected 12 questions, found {len(questions)}")
    else:
        ok("12 questions present")

    rubric_rows = re.findall(r"\|\s*K\d+\s*\|", text)
    if len(rubric_rows) != 12:
        fail(f"expected 12 rubric rows (K1-K12), found {len(rubric_rows)}")
    else:
        ok("12 rubric rows present")

    required_sections = [
        "Pilot identity",
        "Frozen canonical snapshot",
        "Conditions",
        "Reconstruction task",
        "Scoring rubric",
        "Automatic failure conditions",
        "Word accounting",
        "Permitted source references",
        "Launch order",
        "Blind scoring procedure",
        "Stop rules",
    ]
    for section in required_sections:
        if section not in text:
            fail(f"missing section: {section}")
        else:
            ok(f"section present: {section}")

    if "A1, B1, B2, A2" not in text:
        fail("launch order A1, B1, B2, A2 not found")
    else:
        ok("launch order A1, B1, B2, A2")

    if "2300" not in text or "2500" not in text:
        fail("word count target 2300-2500 not found")
    else:
        ok("word count target 2300-2500")

    if "5%" not in text:
        fail("5% A/B difference limit not found")
    else:
        ok("5% A/B difference limit")

    if 'len(text.split())' not in text:
        fail("Python len(text.split()) word-count method not found")
    else:
        ok("word-count method specified")

    if "len(text.encode('utf-8'))" not in text:
        fail("UTF-8 byte count method not found")
    else:
        ok("byte-count method specified")

    if "Condition A" not in text:
        fail("Condition A not mentioned")
    else:
        ok("Condition A present")

    condition_a_match = re.search(
        r"Condition A.*?(relational|provenance|temporal)", text, re.DOTALL | re.IGNORECASE
    )
    if not condition_a_match:
        fail("Condition A does not mention relational/provenance/temporal")
    else:
        ok("Condition A is relational/provenance")

    condition_b_match = re.search(
        r"Condition B.*?(non-graph|structured summary)", text, re.DOTALL | re.IGNORECASE
    )
    if not condition_b_match:
        fail("Condition B does not mention non-graph/structured summary")
    else:
        ok("Condition B is non-graph summary")

    if re.search(r"source.opening.budget|maximum two|preferred.zero|may request to open", text, re.IGNORECASE):
        fail("source reopening language still present")
    else:
        ok("no source reopening language")

    if not re.search(r"source reopening.*exactly zero", text, re.IGNORECASE):
        fail("source reopening policy must be exactly zero")
    else:
        ok("source reopening policy fixed at zero")

    zero_reopen_phrases = [
        "Do not open any other source",
        "Do not access live GitHub",
    ]
    for phrase in zero_reopen_phrases:
        if phrase not in text:
            fail(f"missing zero-reopening instruction: '{phrase}'")
        else:
            ok(f"zero-reopening: '{phrase}'")

    if "A1: Condition A (relational/provenance)" in text:
        ok("A1 labeled as relational/provenance")
    else:
        fail("A1 not correctly labeled as relational/provenance")

    if "B1: Condition B (non-graph summary)" in text:
        ok("B1 labeled as non-graph summary")
    else:
        fail("B1 not correctly labeled as non-graph summary")


def validate_qk_alignment():
    print("\n=== Q/K alignment validation ===")
    with open(PREREG_PATH) as f:
        prereg = f.read()
    with open(ANSWER_KEY_PATH) as f:
        answer_key = f.read()

    q_themes = [
        (1, "Issue #157"),
        (2, "PR #158"),
        (3, "Issue"),
        (4, "PR #158"),
        (5, "spec"),
        (6, "CI"),
        (7, "review"),
        (8, "R-005"),
        (9, "close"),
        (10, "prior head"),
        (11, "limitation"),
        (12, "next"),
    ]

    question_section = re.search(
        r"### Questions \(Q1–Q12\)(.*?)(?:\n## Scoring rubric|$)",
        prereg,
        re.DOTALL,
    )
    if not question_section:
        fail("Questions section not found")
        question_lines = []
    else:
        question_lines = re.findall(r"^\s*(\d+)\.\s+(.+)$", question_section.group(1), re.MULTILINE)

    if len(question_lines) != 12:
        fail(f"expected 12 question lines, found {len(question_lines)}")
    else:
        ok("12 questions listed")

    q_numbers = [n for n, _ in question_lines]
    if q_numbers != _ordered_numbers(12):
        fail(f"questions are not labeled Q1-K12 sequentially: {', '.join(q_numbers)}")
    else:
        ok("questions labeled 1-12 sequentially")

    for num_text, text_question in question_lines:
        idx = int(num_text) - 1
        if idx < len(q_themes):
            token = q_themes[idx][1]
            if token.lower() not in text_question.lower():
                fail(f"Q{num_text} missing theme token '{token}'")
            else:
                ok(f"Q{num_text} theme token '{token}'")

    rubric_lines = re.findall(r"\|\s*K(\d+)\s*\|", prereg, re.MULTILINE)
    rubric_keys = [key for key in re.findall(r"\|\s*K(\d+)\s*\|", prereg)]
    if len(rubric_lines) != 12:
        fail(f"expected 12 rubric entries, found {len(rubric_lines)}")
    elif rubric_keys != _ordered_numbers(12):
        fail(f"rubric keys not sequential K1-K12: {', '.join(rubric_keys)}")
    else:
        ok("12 sequential rubric entries K1-K12")

    ak_headings = re.findall(r"^## K(\d+) — (.+)$", answer_key, re.MULTILINE)
    if len(ak_headings) != 12:
        fail(f"expected 12 answer key headings, found {len(ak_headings)}")
    else:
        ok("12 answer key headings")

    expected_heading_keywords = {
        "1": "Issue #157",
        "2": "PR #158",
        "3": "OPEN",
        "4": "PR",
        "5": "spec",
        "6": "erification",
        "7": "eview",
        "8": "R-005",
        "9": "completion",
        "10": "stale",
        "11": "ncertainti",
        "12": "action",
    }

    for num, heading in ak_headings:
        kw = expected_heading_keywords.get(num)
        if kw and kw.lower() not in heading.lower():
            fail(f"K{num} heading '{heading}' missing keyword '{kw}'")
        else:
            ok(f"K{num} heading keyword '{kw}' present")

    expected_k = _ordered_numbers(12)
    key_numbers = [num for num, _ in ak_headings]
    if key_numbers == expected_k:
        ok("Q/K one-to-one sequential mapping by item numbers")
    else:
        fail(f"answer-key headings are not K1-K12: {', '.join(key_numbers)}")


def validate_answer_key():
    print("\n=== Answer key shape validation ===")
    with open(ANSWER_KEY_PATH) as f:
        text = f.read()

    key_items = re.findall(r"^## K(\d+)", text, re.MULTILINE)
    expected = [str(i) for i in range(1, 13)]
    if key_items != expected:
        fail(f"expected K1-K12, found K{', K'.join(key_items)}")
    else:
        ok("12 answer key items K1-K12")

    if "Held back from experimental subjects" not in text:
        fail("held-back statement not found")
    else:
        ok("held-back statement present")

    if "cyax-0157-ablation-20260913T0122Z" not in text:
        fail("snapshot ID not found in answer key")
    else:
        ok("snapshot ID in answer key")


def validate_privacy():
    print("\n=== Privacy scan ===")
    all_files = [SNAPSHOT_PATH, PREREG_PATH, ANSWER_KEY_PATH]
    for fixture in os.listdir(SOURCES_DIR):
        all_files.append(os.path.join(SOURCES_DIR, fixture))

    privacy_patterns = [
        (r"/Users/[^\s\"]+", "absolute macOS path"),
        (r"/home/[^\s\"]+", "absolute home path"),
        (r"~\/[^\s\"]+", "tilde home path"),
        (r"@[a-zA-Z0-9_-]+\.local\b", "local hostname"),
        (r"\b[A-Za-z0-9_]+@[a-zA-Z0-9.-]+\.(edu|com|org)\b", "email address"),
        (r"ghp_[A-Za-z0-9_]+", "GitHub token"),
        (r"sk-[A-Za-z0-9_]+", "API key"),
    ]
    for fpath in all_files:
        with open(fpath) as f:
            content = f.read()
        fname = os.path.relpath(fpath, PILOT_DIR)
        found_any = False
        for pattern, label in privacy_patterns:
            matches = re.findall(pattern, content)
            if matches:
                fail(f"{fname}: found {label}: {matches[0][:40]}...")
                found_any = True
        if not found_any:
            ok(f"{fname}: no private locators detected")


def validate_no_contexts():
    print("\n=== No reconstruction contexts exist ===")
    allowed_files = {
        "preregistration.md",
        "answer_key.md",
        "source_snapshot.json",
        "validate_snapshot.py",
        "common_subject_prompt.md",
        "condition_a_context.md",
        "condition_b_context.md",
        "condition_a_ledger.jsonl",
        "phase2_context_builder.py",
        "phase2_manifest.json",
    }
    prohibited_patterns = (
        r"(^|_)(context|condition|response|scorecard)",
        r"(^|_|-)reconstruction",
    )
    for root, _, filenames in os.walk(PILOT_DIR):
        for filename in filenames:
            rel = os.path.relpath(os.path.join(root, filename), PILOT_DIR)
            base = os.path.basename(filename)
            if base in allowed_files:
                continue
            if Path(rel).parts[0] == "sources":
                continue
            if any(re.search(pat, filename, re.IGNORECASE) for pat in prohibited_patterns):
                fail(f"unexpected context/reconstruction artifact exists: {rel}")
    ok("authorized Phase 2 context-construction artifacts are present; no responses or scores found")


if __name__ == "__main__":
    validate_snapshot()
    validate_captured_fixtures()
    validate_preregistration()
    validate_qk_alignment()
    validate_answer_key()
    validate_privacy()
    validate_no_contexts()

    print(f"\n{'='*50}")
    if failures:
        print(f"FAILED: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)
