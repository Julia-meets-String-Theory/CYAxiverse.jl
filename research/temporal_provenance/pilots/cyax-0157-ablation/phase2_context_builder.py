#!/usr/bin/env python3
"""Build and validate deterministic phase-2 held-out ablation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent
SNAPSHOT_PATH = ROOT / "source_snapshot.json"
PROMPT_PATH = ROOT / "common_subject_prompt.md"
CONTEXT_A_PATH = ROOT / "condition_a_context.md"
CONTEXT_B_PATH = ROOT / "condition_b_context.md"
LEDGER_A_PATH = ROOT / "condition_a_ledger.jsonl"
MANIFEST_PATH = ROOT / "phase2_manifest.json"

WORD_MIN = 2300
WORD_MAX = 2500
MAX_AB_RATIO = 0.05

EXPECTED_FACTS = [f"F{i:02d}" for i in range(1, 13)]

PRIVACY_PATTERNS: Sequence = (
    (re.compile(r"/Users/[^\s\"]+"), "absolute-macos-path"),
    (re.compile(r"/home/[^\s\"]+"), "absolute-unix-path"),
    (re.compile(r"~/(?:[^\s\"]+)?"), "tilde-path"),
    (re.compile(r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b"), "email"),
    (re.compile(r"ghp_[A-Za-z0-9_]+"), "github-token"),
    (re.compile(r"sk-[A-Za-z0-9_]+"), "api-token"),
)


@dataclass(frozen=True)
class Fact:
    fact_id: str
    q: str
    statement: str
    sources: Sequence[str]
    evidence: Sequence[str]
    uncertainty: str


PROMPT_TEXT = """You are a fresh agent with no prior CYAxiverse context. Using only the
provisioned context document, reconstruct the current authoritative state of Issue
#157 and PR #158.

Do not open any other source.
Source reopening is exactly zero.
Do not access live GitHub.
Do not browse the repository.
Do not request additional materials.

Answer all 12 questions and cite the supporting context reference for each.

1. What is Issue #157 and what is its scope?
2. What is PR #158 and what is its relationship to Issue #157?
3. What is the current state of Issue #157 (open/closed, project status)?
4. What is the current state of PR #158 (open/closed, draft status, merge status, head commit, base branch/commit, project status)?
5. What is the approved specification governing this work, what is its approval status, and what evidence exists that the current-surface remediation is implemented at the exact PR head without claiming merge or Issue completion?
6. What do the CI status checks and focused test results show at the current PR head?
7. What evidence exists for independent review of the PR, and what are the limitations of that evidence on the GitHub review surface?
8. What is R-005 and why does it remain open?
9. Why must merging PR #158 not close Issue #157, and what distinguishes PR completion from Issue completion?
10. What is the relationship between the prior head 1ed97e18... and the current head 8f6a9c28...?
11. What specific evidence limitations, uncertainties, or required abstentions apply to the current verification surface?
12. What is the next valid action for this work item?"""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _words(text: str) -> int:
    return len(text.split())


def _bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def _bytes_payload(text: str) -> bytes:
    payload = text + "\n"
    return payload.encode("utf-8")


def _bytes_payload_count(text: str) -> int:
    return len(_bytes_payload(text))


def _bytes_payload_hash(text: str) -> str:
    return hashlib.sha256(_bytes_payload(text)).hexdigest()


def _load_inputs() -> Dict:
    snap = _load_json(SNAPSHOT_PATH)
    issue = _load_json(ROOT / snap["governing_issue"]["captured_fixture"])
    pr = _load_json(ROOT / snap["supporting_pr"]["captured_fixture"])
    comments = _load_json(ROOT / snap["supporting_pr"]["comments_fixture"])
    reviews = _load_json(ROOT / snap["supporting_pr"]["reviews_fixture"])
    checks = _load_json(ROOT / snap["supporting_pr"]["checks_fixture"])
    return {
        "snapshot": snap,
        "issue": issue,
        "pr": pr,
        "comments": comments,
        "reviews": reviews,
        "checks": checks,
    }


def _validate_sources(snapshot: Dict, issue: Dict, pr: Dict, comments: List[Dict], reviews: List[Dict], checks: List[Dict]) -> List[str]:
    issues = []

    expected = {
        snapshot["governing_issue"]["captured_fixture"]: snapshot["governing_issue"]["captured_fixture_sha256"],
        snapshot["supporting_pr"]["captured_fixture"]: snapshot["supporting_pr"]["captured_fixture_sha256"],
        snapshot["supporting_pr"]["comments_fixture"]: snapshot["supporting_pr"]["comments_fixture_sha256"],
        snapshot["supporting_pr"]["reviews_fixture"]: snapshot["supporting_pr"]["reviews_fixture_sha256"],
        snapshot["supporting_pr"]["checks_fixture"]: snapshot["supporting_pr"]["checks_fixture_sha256"],
    }
    for rel, expected_hash in expected.items():
        path = ROOT / rel
        if not path.exists():
            issues.append(f"missing fixture: {rel}")
            continue
        actual = _sha256_file(path)
        if actual != expected_hash:
            issues.append(f"fixture hash mismatch {rel}: {actual} != {expected_hash}")

    issue_body = hashlib.sha256((issue["body"] + "\n").encode("utf-8")).hexdigest()
    if issue_body != snapshot["governing_issue"]["body_sha256"]:
        issues.append("issue body hash mismatch")
    pr_body = hashlib.sha256((pr["body"] + "\n").encode("utf-8")).hexdigest()
    if pr_body != snapshot["supporting_pr"]["body_sha256"]:
        issues.append("pr body hash mismatch")

    if snapshot["governing_issue"]["state"] != "OPEN":
        issues.append("governing issue is not OPEN")
    if snapshot["supporting_pr"]["state"] != "OPEN":
        issues.append("supporting pr is not OPEN")
    if snapshot["supporting_pr"]["is_draft"]:
        issues.append("supporting pr is draft")
    if snapshot["supporting_pr"]["merged_at"] is not None:
        issues.append("supporting pr appears merged")

    if len(comments) != 2:
        issues.append(f"expected 2 comments, found {len(comments)}")
    if len(reviews) != 0:
        issues.append(f"expected empty reviews, found {len(reviews)}")
    if len(checks) != 3:
        issues.append(f"expected 3 check entries, found {len(checks)}")

    by_name = {c["name"]: c for c in checks}
    expected_checks = {
        "Fast tests": "FAILURE",
        "build": "SUCCESS",
        "Full test suite": "SKIPPED",
    }
    for name, expected_conclusion in expected_checks.items():
        item = by_name.get(name)
        if item is None:
            issues.append(f"missing check record {name}")
        elif item.get("conclusion") != expected_conclusion:
            issues.append(f"check {name} conclusion {item.get('conclusion')} != {expected_conclusion}")

    return issues


def _build_facts(snapshot: Dict, issue: Dict, pr: Dict, comments: List[Dict], checks: List[Dict]) -> List[Fact]:
    prior = snapshot["git_objects"]["prior_head_stale"]
    head = snapshot["git_objects"]["pr_head_commit"]
    base = snapshot["git_objects"]["pr_base_commit"]
    tree = snapshot["git_objects"]["pr_head_tree"]

    head_status = next((c for c in checks if c["name"] == "Fast tests"), {})
    build_status = next((c for c in checks if c["name"] == "build"), {})
    full_status = next((c for c in checks if c["name"] == "Full test suite"), {})
    first_comment = comments[0]["body"] if len(comments) > 0 else ""
    second_comment = comments[1]["body"] if len(comments) > 1 else ""

    return [
        Fact(
            "F01",
            "Q1",
            "Issue #157 is a public-surface remediation request titled ‘Remove current machine-local path disclosures’. The scope is to eliminate confirmed machine-local filesystem details while keeping scientifically useful identifiers, preserve scientific algorithms and schema behavior, and add focused path-resolution tests.",
            ["github:Julia-meets-String-Theory/CYAxiverse.jl#157"],
            ["issue body Problem and Scope sections"],
            "No broader scope change is asserted in the frozen source snapshot.",
        ),
        Fact(
            "F02",
            "Q2",
            "PR #158 is titled ‘Sanitize machine-local path configuration’ and implements Issue #157’s requested remediation by moving personal and Slurm paths to explicit configuration and repository-relative resolution while preserving algorithms and schema behavior.",
            ["github:Julia-meets-String-Theory/CYAxiverse.jl#158"],
            ["PR #158 summary lines"],
            "The body is explicit that Issue #157 stays open and that R-005 is not finished.",
        ),
        Fact(
            "F03",
            "Q3",
            "Issue #157 is OPEN and its project status is Verification at the observed freeze point.",
            ["github:Julia-meets-String-Theory/CYAxiverse.jl#157"],
            ["snapshot.governing_issue.state", "snapshot.governing_issue.project_status"],
            "Project status can change after observation time.",
        ),
        Fact(
            "F04",
            "Q4",
            f"PR #158 is OPEN, non-draft, unmerged, mergeable, with head commit {head}, tree {tree}, and base branch vmm/{base}.",
            ["github:Julia-meets-String-Theory/CYAxiverse.jl#158"],
            ["snapshot.supporting_pr state, mergeable, head_oid, head_tree, base_oid, is_draft"],
            "Merge decision is not present in the frozen observation surface.",
        ),
        Fact(
            "F05",
            "Q5",
            "The captured artifacts include the CYAX-0157 and CYAX-0155 specification files and the PR body documents current-surface remediation at head state while explicitly separating completion from PR merge or issue closure.",
            [
                "specs/0157-public-path-remediation/spec.md",
                "specs/0155-private-safe-chat-checkpoints/spec.md",
                "github:Julia-meets-String-Theory/CYAxiverse.jl#158",
                "github:Julia-meets-String-Theory/CYAxiverse.jl#157",
            ],
            ["snapshot.repository_artifacts_at_head", "snapshot.open_work_items", "PR #158 body and Issue #157 body"],
            "Approval text is recorded in the frozen artifact set only, not inferred by this generator.",
        ),
        Fact(
            "F06",
            "Q6",
            f"Captured CI/check status is Fast tests: {head_status.get('conclusion')} ({head_status.get('completedAt')}), build: {build_status.get('conclusion')}, Full test suite: {full_status.get('conclusion')}; PR body records focused suites passing and then encountering the unchanged mismatch at test/runtests.jl:1317.",
            [
                "sources/pr_158_checks.json",
                "github:Julia-meets-String-Theory/CYAxiverse.jl#158",
            ],
            [
                f"check {head_status.get('name', 'Fast tests')}",
                f"check {build_status.get('name', 'build')}",
                f"check {full_status.get('name', 'Full test suite')}",
                "PR #158 verification section",
            ],
            "Focused pass/fail counts in the body are self-reported; no independent rerun result is embedded in the frozen snapshot.",
        ),
        Fact(
            "F07",
            "Q7",
            "PR #158 self-reports independent re-review approve/no findings, yet the `sources/pr_158_reviews.json` surface is empty, which creates a review-recording limitation.",
            ["sources/pr_158_reviews.json", "github:Julia-meets-String-Theory/CYAxiverse.jl#158"],
            ["first review section", "reviews fixture len 0"],
            "This is a source-surface boundary, not independent contradiction of the PR body claim.",
        ),
        Fact(
            "F08",
            "Q8",
            "R-005 remains open and requires a surviving-public-branch read-only privacy scan plus preservation/reachability classification; no public branch has been deleted, force-pushed, or rewritten at the snapshot time.",
            ["snapshot.open_work_items.R-005", "github:Julia-meets-String-Theory/CYAxiverse.jl#157", "github:Julia-meets-String-Theory/CYAxiverse.jl#158"],
            ["issue body R-005 deferred", "pr body R-005 deferred", "snapshot open_work_items"],
            "R-005 completion is forward-looking relative to this artifact.",
        ),
        Fact(
            "F09",
            "Q9",
            "PR #158 must not be treated as closing Issue #157 because R-005 and remaining branch-safety work are outstanding; PR and Issue completion remain distinct by the current rules in snapshot bodies.",
            ["github:Julia-meets-String-Theory/CYAxiverse.jl#158", "github:Julia-meets-String-Theory/CYAxiverse.jl#157"],
            ["issue/PR explicit statements", "R-005 text"],
            "The distinction is a policy constraint and should not be inferred as already executed.",
        ),
        Fact(
            "F10",
            "Q10",
            f"The prior head `{prior}` is stale because later comments and verification target current head `{head}` in PR #158 context, including explicit incorporation of base `{base}`.",
            ["sources/pr_158_comments.json", "snapshot.git_objects.prior_head_stale", "snapshot.git_objects.pr_head_commit", "snapshot.git_objects.pr_base_commit"],
            ["first comment references prior head", "second comment references current head", "snapshot git_objects"],
            "Earlier comment context is historical and should not be used as current PR state.",
        ),
        Fact(
            "F11",
            "Q11",
            "Uncertainties include: no independent full CI rerun evidence in this frozen surface, snapshot-timestamp scope, empty formal review list, and no production-scale run claims.",
            [
                "sources/pr_158_checks.json",
                "sources/pr_158_comments.json",
                "sources/pr_158_reviews.json",
                "snapshot.observation_time",
            ],
            ["PR body, snapshot fields", "fixtures for checks/comments/reviews"],
            "Abstain from claims requiring state after 2026-09-13T01:22:00Z.",
        ),
        Fact(
            "F12",
            "Q12",
            f"Next valid action is to capture an exact-head review/owner decision for `{head}`, keep Issue #157 open after any PR merge unless R-005 is complete, and execute the branch scan/classification before history rewrite or branch deletion operations.",
            ["github:Julia-meets-String-Theory/CYAxiverse.jl#157", "github:Julia-meets-String-Theory/CYAxiverse.jl#158", "snapshot.open_work_items.R-005"],
            ["Issue #157 and PR #158 body language", "snapshot R-005 requirement"],
            "Only the owner can complete this action; this artifact cannot do it directly.",
        ),
    ]


def _fact_lines(facts: Sequence[Fact]) -> List[Dict[str, str]]:
    return [
        {
            "record": "fact",
            "fact_id": fact.fact_id,
            "question": fact.q,
            "statement": fact.statement,
            "sources": list(fact.sources),
            "evidence": list(fact.evidence),
            "uncertainty": fact.uncertainty,
        }
        for fact in facts
    ]


def _evidence_rows(facts: Sequence[Fact]) -> List[Dict[str, str]]:
    rows = []
    for idx, fact in enumerate(facts, start=1):
        for eidx, evidence in enumerate(fact.evidence, start=1):
            src = fact.sources[(eidx - 1) % len(fact.sources)] if fact.sources else "unknown"
            rows.append(
                {
                    "evidence_id": f"E{idx:02d}{eidx}",
                    "fact_id": fact.fact_id,
                    "question": fact.q,
                    "source": src,
                    "summary": evidence,
                }
            )
    return rows


def _pad(text: str, min_words: int = WORD_MIN, max_words: int = WORD_MAX) -> str:
    words = text.split()
    if len(words) >= min_words:
        return text

    pad_pool = [
        "This model is built only from frozen offline artifacts and does not call remote systems.",
        "Each claim is accompanied by source provenance and an explicit uncertainty marker where the surface is underspecified.",
        "The subject context inherits no hidden memory and receives no additional authority from prompt order, confidence levels, or interface recency.",
        "The artifact set records the branch lineage and the distinction between current and stale references for safe reconstruction.",
        "When evidence is self-reported, it is labeled as self-reported and separated from independently captured surface records.",
        "No path for branch deletion, force-push, or rewrite is encoded as complete because those steps are explicitly outside the captured state.",
        "The output format is deterministic and stable under re-execution of this builder from the same snapshot and fixtures.",
        "All 12 items in this section correspond to the 12 canonical questions in the shared prompt.",
    ]

    i = 0
    while len(words) < min_words and i < 100:
        candidate = pad_pool[i % len(pad_pool)]
        candidate_words = candidate.split()
        if len(words) + len(candidate_words) <= max_words:
            words.extend(candidate_words)
        i += 1

    if len(words) < min_words:
        raise RuntimeError("unable to pad deterministic text to required minimum without violating max")

    return " ".join(words)


def _compose_context_a(snapshot: Dict, facts: Sequence[Fact], evidence_rows: Sequence[Dict[str, str]]) -> str:
    source_refs = [
        "github:Julia-meets-String-Theory/CYAxiverse.jl#157",
        "github:Julia-meets-String-Theory/CYAxiverse.jl#158",
        "specs/0157-public-path-remediation/spec.md",
        "specs/0155-private-safe-chat-checkpoints/spec.md",
        "AGENTS.md",
        "sources/issue_157.json",
        "sources/pr_158.json",
        "sources/pr_158_checks.json",
        "sources/pr_158_comments.json",
        "sources/pr_158_reviews.json",
    ]

    sections = [
        "# Condition A — relational/provenance temporal context",
        "",
        "Built from the frozen snapshot and captured fixtures only.",
        "",
        "## Governing instructions",
        PROMPT_TEXT,
        "",
        "## Canonical source index",
        *(f"- {x}" for x in source_refs),
        "",
        "## Prose facts linked by temporal and authority provenance",
    ]

    fact_map = {"Q1": "F01", "Q2": "F02", "Q3": "F03", "Q4": "F04", "Q5": "F05", "Q6": "F06", "Q7": "F07", "Q8": "F08", "Q9": "F09", "Q10": "F10", "Q11": "F11", "Q12": "F12"}
    for fact in facts:
        sections.extend(
            [
                f"### {fact_map[fact.q]} / {fact.q}",
                fact.statement,
                f"Sources: {'; '.join(fact.sources)}",
                f"Evidence: {'; '.join(fact.evidence)}",
                f"Uncertainty: {fact.uncertainty}",
            ]
        )
        sections.append("")

    sections.extend(
        [
            "## Provenance-ledger (JSONL)",
            "```jsonl",
            json.dumps({
                "record_type": "snapshot",
                "snapshot_id": snapshot["snapshot_id"],
                "observation_time": snapshot["observation_time"],
                "deterministic": snapshot["integrity"]["deterministic"],
                "frozen_before_contexts": snapshot["integrity"]["snapshot_frozen_before_contexts"],
            }, sort_keys=True),
            json.dumps(
                {
                    "record_type": "governance",
                    "issue": snapshot["governing_issue"],
                    "pr": snapshot["supporting_pr"],
                    "open_work_items": snapshot["open_work_items"],
                },
                sort_keys=True,
            ),
            json.dumps(
                {
                    "record_type": "checks",
                    "git_objects": snapshot["git_objects"],
                    "checks_fixture": snapshot["supporting_pr"]["checks_fixture"],
                },
                sort_keys=True,
            ),
            "```",
            "",
            "## Coverage accounting",
            f"Distinct source references: {len(set(source_refs))}",
            f"Evidence items: {len(evidence_rows)}",
            "Fact coverage map: " + _render_fact_coverage(facts),
            "",
        ]
    )

    text = "\n".join(sections)
    return _pad(text)


def _compose_context_b(snapshot: Dict, facts: Sequence[Fact], evidence_rows: Sequence[Dict[str, str]]) -> str:
    source_refs = [
        "github:Julia-meets-String-Theory/CYAxiverse.jl#157",
        "github:Julia-meets-String-Theory/CYAxiverse.jl#158",
        "snapshot:cyax-0157-ablation-20260913T0122Z",
        "specs/0157-public-path-remediation/spec.md",
        "specs/0155-private-safe-chat-checkpoints/spec.md",
        "AGENTS.md",
        "sources/issue_157.json",
        "sources/pr_158.json",
        "sources/pr_158_checks.json",
        "sources/pr_158_comments.json",
        "sources/pr_158_reviews.json",
    ]

    sections = [
        "# Condition B — structured summary",
        "",
        "This is a structured summary with no reference-link notation, no pairwise topology wording, and no tuple-style notation.",
        "",
        "## Governing instructions",
        PROMPT_TEXT,
        "",
        "## Source index",
        *(f"- {x}" for x in source_refs),
        "",
        "## Question-by-question summary",
    ]

    fact_map = {"Q1": "F01", "Q2": "F02", "Q3": "F03", "Q4": "F04", "Q5": "F05", "Q6": "F06", "Q7": "F07", "Q8": "F08", "Q9": "F09", "Q10": "F10", "Q11": "F11", "Q12": "F12"}
    for fact in facts:
        sections.extend(
            [
                f"### {fact_map[fact.q]} / {fact.q}",
                fact.statement,
                "Evidence anchors:",
                *(f"- {s}" for s in fact.sources),
                f"Uncertainty: {fact.uncertainty}",
            ]
        )
        sections.append("")

    sections.extend(
        [
            "## Evidence ledger",
            *(f"{row['evidence_id']} {row['question']} {row['fact_id']} | {row['summary']}" for row in evidence_rows),
            "",
            "## Stale versus current state",
            f"stale: `1ed97e181e3203524d5435c43f58c1a3ff03d813`, current: `{snapshot['git_objects']['pr_head_commit']}`",
            "If a previous head appears, it is historical and not current.",
            "",
            "## Next action",
            f"Take an exact-head review and owner decision for `{snapshot['git_objects']['pr_head_commit']}` first.",
            "Do not close Issue #157 until R-005 is complete and branch preservation/reachability is recorded.",
            "",
            "## Coverage accounting",
            f"Distinct source references: {len(set(source_refs))}",
            f"Evidence items: {len(evidence_rows)}",
            "Fact coverage map: " + _render_fact_coverage(facts),
        ]
    )

    text = "\n".join(sections)
    return _pad(text)


def _render_fact_coverage(facts: Sequence[Fact]) -> str:
    return ", ".join(f"{f.q}:{f.fact_id}" for f in facts)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ledger_lines(snapshot: Dict, facts: Sequence[Fact]) -> List[Dict[str, object]]:
    return [
        {
            "record_type": "snapshot",
            "snapshot_id": snapshot["snapshot_id"],
            "observation_time": snapshot["observation_time"],
            "git_head": snapshot["git_objects"]["pr_head_commit"],
        },
        *_fact_lines(facts),
        {
            "record_type": "checks",
            "head": snapshot["git_objects"]["pr_head_commit"],
            "tree": snapshot["git_objects"]["pr_head_tree"],
            "base": snapshot["git_objects"]["pr_base_commit"],
        },
    ]


def build(emit: bool = True) -> Dict:
    data = _load_inputs()
    snapshot = data["snapshot"]
    issue = data["issue"]
    pr = data["pr"]
    comments = data["comments"]
    reviews = data["reviews"]
    checks = data["checks"]

    failures = _validate_sources(snapshot, issue, pr, comments, reviews, checks)
    if failures:
        raise RuntimeError("input validation failed:\n" + "\n".join(failures))

    facts = _build_facts(snapshot, issue, pr, comments, checks)
    evidence = _evidence_rows(facts)
    ledger = _ledger_lines(snapshot, facts)

    context_a = _compose_context_a(snapshot, facts, evidence)
    context_b = _compose_context_b(snapshot, facts, evidence)

    w_a, w_b = _words(context_a), _words(context_b)
    if not (WORD_MIN <= w_a <= WORD_MAX):
        raise RuntimeError(f"Condition A word count out of range: {w_a}")
    if not (WORD_MIN <= w_b <= WORD_MAX):
        raise RuntimeError(f"Condition B word count out of range: {w_b}")
    if abs(w_a - w_b) / max(w_a, w_b) > MAX_AB_RATIO:
        raise RuntimeError("A/B word-count difference above 5%")

    prompt_payload = "\n".join(PROMPT_TEXT.splitlines())
    context_a_payload = context_a
    context_b_payload = context_b
    ledger_payload = "\n".join(json.dumps(item, sort_keys=True) for item in ledger)

    manifest = {
        "pilot": "CYAX-0157",
        "phase": 2,
        "snapshot_id": snapshot["snapshot_id"],
        "observation_time": snapshot["observation_time"],
        "snapshot_sha256": _sha256_file(SNAPSHOT_PATH),
        "question_to_fact": {str(i): f"F{i:02d}" for i in range(1, 13)},
        "fact_ids": [f.fact_id for f in facts],
        "source_reopening": "exactly_zero",
        "artifact_contract": {
            "word_min": WORD_MIN,
            "word_max": WORD_MAX,
            "max_ab_ratio": MAX_AB_RATIO,
            "required_questions": 12,
            "required_facts": 12,
        },
        "fixtures": {
            "issue_fixture": snapshot["governing_issue"]["captured_fixture"],
            "pr_fixture": snapshot["supporting_pr"]["captured_fixture"],
            "comments_fixture": snapshot["supporting_pr"]["comments_fixture"],
            "reviews_fixture": snapshot["supporting_pr"]["reviews_fixture"],
            "checks_fixture": snapshot["supporting_pr"]["checks_fixture"],
            "hashes": {
                snapshot["governing_issue"]["captured_fixture"]: snapshot["governing_issue"]["captured_fixture_sha256"],
                snapshot["supporting_pr"]["captured_fixture"]: snapshot["supporting_pr"]["captured_fixture_sha256"],
                snapshot["supporting_pr"]["comments_fixture"]: snapshot["supporting_pr"]["comments_fixture_sha256"],
                snapshot["supporting_pr"]["reviews_fixture"]: snapshot["supporting_pr"]["reviews_fixture_sha256"],
                snapshot["supporting_pr"]["checks_fixture"]: snapshot["supporting_pr"]["checks_fixture_sha256"],
            },
        },
        "outputs": {
            "prompt": {
                "path": PROMPT_PATH.name,
                "word_count": _words(PROMPT_TEXT),
                "byte_count": _bytes_payload_count(prompt_payload),
                "sha256": _bytes_payload_hash(prompt_payload),
            },
            "condition_a_context": {
                "path": CONTEXT_A_PATH.name,
                "word_count": w_a,
                "byte_count": _bytes_payload_count(context_a_payload),
                "sha256": _bytes_payload_hash(context_a_payload),
            },
            "condition_b_context": {
                "path": CONTEXT_B_PATH.name,
                "word_count": w_b,
                "byte_count": _bytes_payload_count(context_b_payload),
                "sha256": _bytes_payload_hash(context_b_payload),
            },
            "condition_a_ledger": {
                "path": LEDGER_A_PATH.name,
                "line_count": len(ledger),
                "byte_count": _bytes_payload_count(ledger_payload),
                "sha256": _bytes_payload_hash(ledger_payload),
            },
        },
        "coverage": {
            "evidence_items": len(evidence),
            "distinct_source_references": len({src for row in evidence for src in [row["source"]] if row["source"] != "unknown"}),
            "fact_map": _render_fact_coverage(facts),
        },
    }

    if emit:
        _write_lines(PROMPT_PATH, [prompt_payload])
        _write_lines(CONTEXT_A_PATH, [context_a_payload])
        _write_lines(CONTEXT_B_PATH, [context_b_payload])
        _write_lines(LEDGER_A_PATH, [ledger_payload])
        MANIFEST_PATH.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    return manifest


def _privacy_scan() -> List[str]:
    paths = [
        SNAPSHOT_PATH,
        ROOT / "source_snapshot.json",
        ROOT / "answer_key.md",
        ROOT / "preregistration.md",
        PROMPT_PATH,
        CONTEXT_A_PATH,
        CONTEXT_B_PATH,
        MANIFEST_PATH,
        LEDGER_A_PATH,
        *[ROOT / f for f in [
            "sources/issue_157.json",
            "sources/pr_158.json",
            "sources/pr_158_comments.json",
            "sources/pr_158_checks.json",
            "sources/pr_158_reviews.json",
        ]],
    ]
    found = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        for pattern, label in PRIVACY_PATTERNS:
            if pattern.search(text):
                found.append(f"{path}: {label}")
    return found


def validate() -> None:
    manifest = _load_json(MANIFEST_PATH)
    errors: List[str] = []

    if not (manifest.get("question_to_fact") and len(manifest["question_to_fact"]) == 12):
        errors.append("Q/F mapping missing or not 12 items")

    if manifest.get("fact_ids") != EXPECTED_FACTS:
        errors.append("fact ids are not exact F01-F12")

    if manifest.get("source_reopening") != "exactly_zero":
        errors.append("source reopening policy is not exactly zero")

    inputs = _load_inputs()
    snap = inputs["snapshot"]
    issue = inputs["issue"]
    pr = inputs["pr"]
    comments = inputs["comments"]
    reviews = inputs["reviews"]
    checks = inputs["checks"]
    errors.extend(_validate_sources(snap, issue, pr, comments, reviews, checks))

    prompt_text = PROMPT_PATH.read_text()
    q_count = len(re.findall(r"^\d+\. ", prompt_text, re.MULTILINE))
    if q_count != 12:
        errors.append("prompt does not contain exactly 12 questions")

    if "exactly zero" not in prompt_text:
        errors.append("prompt lacks exact zero source reopening policy")

    built = build(emit=False)
    expected_outputs = built["outputs"]
    manifest_outputs = manifest.get("outputs", {})
    if _words(CONTEXT_A_PATH.read_text(errors="ignore")) != built["outputs"]["condition_a_context"]["word_count"]:
        errors.append("condition A regeneration count mismatch")
    if _words(CONTEXT_B_PATH.read_text(errors="ignore")) != built["outputs"]["condition_b_context"]["word_count"]:
        errors.append("condition B regeneration count mismatch")

    # A/B word-count constraints
    a_words = manifest["outputs"]["condition_a_context"]["word_count"]
    b_words = manifest["outputs"]["condition_b_context"]["word_count"]
    if not (WORD_MIN <= a_words <= WORD_MAX):
        errors.append("condition A word count out of range")
    if not (WORD_MIN <= b_words <= WORD_MAX):
        errors.append("condition B word count out of range")
    if abs(a_words - b_words) / max(a_words, b_words) > MAX_AB_RATIO:
        errors.append("A/B difference over 5%")

    forbidden = [
        re.compile(r"\bedge\b", re.IGNORECASE),
        re.compile(r"\bedges\b", re.IGNORECASE),
        re.compile(r"\badjacency\b", re.IGNORECASE),
        re.compile(r"\badjacency list\b", re.IGNORECASE),
        re.compile(r"\btriple\b", re.IGNORECASE),
        re.compile(r"\bassertion\b", re.IGNORECASE),
        re.compile(r"\bnode\b", re.IGNORECASE),
    ]
    b_text = CONTEXT_B_PATH.read_text()
    for token in forbidden:
        if token.search(b_text):
            errors.append(f"condition B contains forbidden token pattern '{token.pattern}'")

    if "K1" in CONTEXT_A_PATH.read_text() or "K1" in CONTEXT_B_PATH.read_text():
        errors.append("answer key leakage marker detected")

    artifact_paths = {
        "prompt": PROMPT_PATH,
        "condition_a_context": CONTEXT_A_PATH,
        "condition_b_context": CONTEXT_B_PATH,
        "condition_a_ledger": LEDGER_A_PATH,
    }

    for key in ["prompt", "condition_a_context", "condition_b_context", "condition_a_ledger"]:
        actual_path = artifact_paths.get(key)
        manifest_meta = manifest_outputs.get(key, {})
        expected_meta = expected_outputs.get(key, {})
        if not actual_path:
            errors.append(f"missing expected artifact path for {key}")
            continue
        if not actual_path.exists():
            errors.append(f"missing artifact file for {key}")
            continue
        actual_bytes = actual_path.read_bytes()
        actual_hash = hashlib.sha256(actual_bytes).hexdigest()

        if manifest_meta.get("byte_count") != len(actual_bytes):
            errors.append(f"manifest byte_count mismatch for {key}")
        if expected_meta.get("byte_count") != len(actual_bytes):
            errors.append(f"regenerated artifact byte_count mismatch for {key}")
        if manifest_meta.get("sha256") != actual_hash:
            errors.append(f"manifest sha256 mismatch for {key}")
        if expected_meta.get("sha256") != actual_hash:
            errors.append(f"regenerated artifact sha256 mismatch for {key}")

        if key == "condition_a_ledger":
            manifest_line_count = manifest_meta.get("line_count")
            actual_line_count = len(actual_bytes.decode("utf-8", errors="ignore").splitlines())
            if manifest_line_count != actual_line_count:
                errors.append(f"ledger line_count mismatch for {key}")

    if issues := _privacy_scan():
        errors.extend(f"privacy pattern: {x}" for x in issues)

    if errors:
        raise RuntimeError("validation failed:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build", "validate", "privacy"], help="run mode")
    args = parser.parse_args()

    if args.mode == "build":
        build(emit=True)
        print("Phase-2 artifacts built")
    elif args.mode == "privacy":
        issues = _privacy_scan()
        if issues:
            raise RuntimeError("privacy scan failed:\n" + "\n".join(issues))
        print("Privacy scan passed")
    else:
        validate()
        print("Phase-2 validation passed")


if __name__ == "__main__":
    main()
