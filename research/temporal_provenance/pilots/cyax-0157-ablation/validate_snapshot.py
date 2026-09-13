#!/usr/bin/env python3
"""Validate the CYAX-0157 ablation source snapshot and preregistration shape.

Run from the repository root:
    python3 research/temporal_provenance/pilots/cyax-0157-ablation/validate_snapshot.py
"""

import json
import hashlib
import os
import re
import subprocess
import sys

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.path.join(PILOT_DIR, "source_snapshot.json")
PREREG_PATH = os.path.join(PILOT_DIR, "preregistration.md")
ANSWER_KEY_PATH = os.path.join(PILOT_DIR, "answer_key.md")

EXPECTED_HEAD = "8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45"
EXPECTED_TREE = "fc980d28f3faa51f66031fe2e1fb5281b2ad5a48"
EXPECTED_BASE = "856012f015a866bf7ff352bc50e8d10c250855e6"

failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def ok(msg):
    print(f"  OK: {msg}")


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

    if len(snap["supporting_pr"]["reviews"]) != 0:
        fail("PR #158 reviews not empty")
    else:
        ok("PR #158 reviews empty")

    checks = {c["name"]: c["conclusion"] for c in snap["supporting_pr"]["status_checks"]}
    if checks.get("Fast tests") != "FAILURE":
        fail(f"Fast tests expected FAILURE, got {checks.get('Fast tests')}")
    else:
        ok("Fast tests FAILURE")
    if checks.get("build") != "SUCCESS":
        fail(f"Documentation expected SUCCESS, got {checks.get('build')}")
    else:
        ok("Documentation SUCCESS")
    if checks.get("Full test suite") != "SKIPPED":
        fail(f"Full test suite expected SKIPPED, got {checks.get('Full test suite')}")
    else:
        ok("Full test suite SKIPPED")

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
        fname = os.path.basename(fpath)
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
    for name in ["context_a.md", "context_b.md", "context.md",
                  "condition_a.md", "condition_b.md"]:
        path = os.path.join(PILOT_DIR, name)
        if os.path.exists(path):
            fail(f"context file already exists: {name}")
    ok("no reconstruction context files found")


if __name__ == "__main__":
    validate_snapshot()
    validate_preregistration()
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
