#!/usr/bin/env python3
"""Deterministically build and validate the CYAX-0155 A/B contexts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FREEZE = "44937a83cc039805d072bd0263517d8dfef03a10"
A_PATH = HERE / "condition_a_context.md"
B_PATH = HERE / "condition_b_context.md"
MANIFEST = HERE / "context_manifest.json"

SOURCES = [
    ("issue-155-current", "GitHub Issue observation", "current work-state observation"),
    ("issue-155-checkpoint", "Issue comment at 2026-09-11T00:01:47Z", "durable historical workflow statement"),
    ("issue-155-closure-event", "Issue timeline event at 2026-09-11T01:09:18Z", "later work-state event"),
    ("pr-156", "GitHub pull-request observation", "implementation/integration evidence"),
    ("pr-156-merge-commit", "Git commit f0013552cd69a93221464e9c8ccfd56339f39052", "repository implementation evidence"),
    ("spec-0155", "Approved CYAX-0155 S1 specification", "feature intent and acceptance contract"),
    ("agents-contract-at-merge", "AGENTS.md at merge revision", "repository policy authority"),
    ("sdd-contract-at-merge", "cyaxiverse-sdd at merge revision", "workflow/authority semantics"),
    ("checkpoint-guide-at-merge", "Human guide at merge revision", "implementation artifact"),
    ("closure-search", "Finite public-artifact search through 2026-09-13T04:35:55Z", "mechanical absence record"),
]

# Same 18 evidence items feed both renderers. Items state source evidence, not the
# scored synthesis that follows from combining them.
ITEMS = [
    ("issue-155-current", "The governing Issue is #155, titled ‘Add private-safe conversation checkpoints to the CYAxiverse Project.’ Its objective is to surface durable, sanitized outcomes from exploratory AI/chat work in the existing Project without turning GitHub into an archive of private conversations."),
    ("issue-155-current", "At the snapshot observation, GitHub reports Issue #155 as CLOSED with state reason COMPLETED. Its item in ‘CYAxiverse Research & Development’ has Project Status Done."),
    ("pr-156", "PR #156 names Issue #155 as the governing Issue and the CYAX-0155 spec as canonical. It presents itself as implementing R-001 through R-005."),
    ("pr-156", "GitHub reports PR #156 MERGED to vmm at 2026-09-11T00:01:20Z with merge commit f0013552cd69a93221464e9c8ccfd56339f39052."),
    ("pr-156-merge-commit", "The single-parent squash commit changes four files with 177 insertions and zero deletions: AGENTS.md, cyaxiverse-sdd, the human checkpoint guide, and the approved CYAX-0155 specification."),
    ("spec-0155", "The spec is ‘Approved for S1 implementation.’ R-001 requires publication sanitization; R-002 defines checkpoints as summaries rather than transcripts; R-003 preserves the existing Kanban lifecycle; R-004 defines Research & Chats as a saved view rather than an authority layer; R-005 requires failing closed when publication safety is uncertain."),
    ("spec-0155", "The acceptance text says R-001–R-005 are represented in the repository control plane, privacy-sensitive data is not introduced, Project Status semantics are unchanged, and ‘a saved Research & Chats view can be configured without introducing a second ledger.’"),
    ("agents-contract-at-merge", "The repository contract says ordinary durable state belongs in Git and assigns the main agent responsibility for scope, interpretation, integration, final diff review, PR state, and handoff. It also contains the privacy/publication rule added by PR #156."),
    ("sdd-contract-at-merge", "The SDD contract says the governing Issue is the primary Project work item and linked PRs normally provide implementation, integration, or evidence. It says GitHub Issues and Projects track live work state, while an approved spec records feature intent."),
    ("checkpoint-guide-at-merge", "The human guide defines Research & Chats as another saved view over the same Project items using the normal Status field and sanitized metadata. It does not report a concrete saved view or filter instance."),
    ("issue-155-checkpoint", "The checkpoint reports that PR #156 merged and that the repository-side privacy boundary is active on vmm."),
    ("issue-155-checkpoint", "Under ‘Remaining Project action,’ the checkpoint says: ‘Keep #155 open until the Research & Chats saved Project view/filter is configured.’"),
    ("issue-155-checkpoint", "The same checkpoint says the available connector exposed neither Projects-v2 view/field mutations nor direct verification, found no alternative Projects-specific connector, and did not claim that UI configuration was complete."),
    ("pr-156", "The PR body likewise describes creation/configuration of the saved view as separate from the repository diff, requiring Projects-v2 write capability, and says no Project mutation is claimed."),
    ("issue-155-closure-event", "Sixty-seven minutes after the checkpoint, the timeline records that vmmhep closed #155 at 2026-09-11T01:09:18Z. The event has no commit identifier and was not performed through a GitHub App."),
    ("issue-155-closure-event", "The captured closure event contains the event type, actor, and time but no reason or explanatory text."),
    ("pr-156-merge-commit", "The PR #156 squash-merge message contains no Closes, Fixes, or Resolves keyword for #155."),
    ("closure-search", "The frozen search found no later public Issue comment, PR material, repository artifact, or search result that documents saved-view configuration or explicitly supersedes the checkpoint’s keep-open condition. This finding is limited to the captured search scope and observation time."),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words(text: str) -> int:
    return len(text.split())


def header(title: str) -> list[str]:
    return [f"# {title}", "", "Frozen snapshot: `cyax-0155-adversarial-20260913T043555Z`", "Observation time: `2026-09-13T04:35:55Z`", "", "This is a derived context, not an authority source. Source scope and authority are stated below. Use only this document during reconstruction.", ""]


def render_a() -> str:
    lines = header("Condition A — relational/provenance evidence")
    lines += ["## Source records", "", "| Record | Identity | Authority class |", "| --- | --- | --- |"]
    for sid, label, authority in SOURCES:
        lines.append(f"| `{sid}` | {label} | {authority} |")
    lines += ["", "## Provenance-bearing assertions", ""]
    for index, (sid, text) in enumerate(ITEMS, 1):
        lines += [f"- `A{index:02d}` — {text}", f"  Provenance: `{sid}`."]
    lines += ["## Explicit relationship assertions", "", "- `issue-155-current` governs the work represented by `pr-156`.", "- `pr-156-merge-commit` supplies implementation evidence for `spec-0155`.", "- `issue-155-checkpoint` follows the merge and precedes `issue-155-closure-event`.", "- `closure-search` describes only the captured record set.", "- No captured assertion links saved-view configuration to a confirming artifact.", "- No captured assertion links closure to an explanation or explicit supersession statement.", "", "## Reconstruction boundary", "", "Determine the combined meaning and evidence scope without assigning authority to this derived representation.", ""]
    return "\n".join(lines)


def render_b() -> str:
    lines = header("Condition B — structured evidence summary")
    lines += ["## Source index", "", "| Source identity | Captured material | Authority scope |", "| --- | --- | --- |"]
    for sid, label, authority in SOURCES:
        lines.append(f"| `{sid}` | {label} | {authority} |")
    lines += ["", "## Evidence cards", ""]
    headings = [
        "Governing work item", "Current Issue observation", "PR-to-Issue scope", "Current PR observation", "Repository diff", "Approved requirements", "Spec acceptance text", "Repository contract", "SDD semantics", "Human guide", "Implementation checkpoint", "Remaining Project action", "Checkpoint tooling limit", "PR Project boundary", "Later closure event", "Closure-event contents", "Merge-message check", "Later-artifact search",
    ]
    for heading, (sid, text) in zip(headings, ITEMS):
        lines += [f"### {heading}", "", text, "", f"Source: `{sid}`.", ""]
    lines += ["## Chronology", "", "| Time | Captured event |", "| --- | --- |", "| 2026-09-11T00:01:20Z | GitHub records PR #156 as merged. |", "| 2026-09-11T00:01:47Z | The durable checkpoint records repository completion and a remaining Project action. |", "| 2026-09-11T01:09:18Z | The timeline records closure of #155 by vmmhep without an attached reason. |", "| 2026-09-13T04:35:55Z | The snapshot observes #155 closed/completed and Project status Done; the finite resolving-evidence search ends. |", "", "## Reading boundary", "", "The source index distinguishes live work-state observations, feature intent, repository implementation evidence, historical workflow text, and a bounded search result. The summary does not add a conclusion that any source does not state.", ""]
    return "\n".join(lines)


def metrics(text: str) -> dict[str, int | str]:
    data = text.encode("utf-8")
    return {"words": words(text), "bytes": len(data), "sha256": digest(data)}


def build(write: bool = True) -> tuple[str, str, dict]:
    a, b = render_a(), render_b()
    prompt = (HERE / "common_subject_prompt.md").read_text()
    manifest = {
        "snapshot_id": "cyax-0155-adversarial-20260913T043555Z",
        "preregistration_commit": FREEZE,
        "generator": "context_builder.py",
        "answer_bearing_facts_each": 12,
        "evidence_items_each": len(ITEMS),
        "source_identities_each": len(SOURCES),
        "condition_a": metrics(a),
        "condition_b": metrics(b),
        "common_prompt": metrics(prompt),
    }
    manifest["word_difference"] = abs(manifest["condition_a"]["words"] - manifest["condition_b"]["words"])
    manifest["word_difference_percent_of_larger"] = round(100 * manifest["word_difference"] / max(manifest["condition_a"]["words"], manifest["condition_b"]["words"]), 3)
    manifest["input_words"] = {
        "condition_a_plus_prompt": manifest["condition_a"]["words"] + manifest["common_prompt"]["words"],
        "condition_b_plus_prompt": manifest["condition_b"]["words"] + manifest["common_prompt"]["words"],
    }
    if write:
        A_PATH.write_text(a, encoding="utf-8")
        B_PATH.write_text(b, encoding="utf-8")
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return a, b, manifest


def validate() -> None:
    a, b, manifest = build(write=False)
    assert subprocess.check_output(["git", "cat-file", "-t", FREEZE], cwd=ROOT, text=True).strip() == "commit"
    assert json.loads((HERE / "source_snapshot.json").read_text())["snapshot_id"] == manifest["snapshot_id"]
    assert manifest["word_difference_percent_of_larger"] <= 5.0
    assert manifest["evidence_items_each"] == 18
    assert manifest["source_identities_each"] == 10
    assert all(a.count(f"Provenance: `{sid}`") == sum(1 for source, _ in ITEMS if source == sid) for sid, _, _ in SOURCES)
    assert all(b.count(f"Source: `{sid}`.") == sum(1 for source, _ in ITEMS if source == sid) for sid, _, _ in SOURCES)
    assert A_PATH.read_text() == a and B_PATH.read_text() == b
    stored = json.loads(MANIFEST.read_text())
    assert stored == manifest
    banned_b = re.compile(r"\b(edge list|subject-predicate-object|adjacency|traversal hint|node identifier|graph topology)\b", re.I)
    assert not banned_b.search(b)
    leakage = ["the correct answer", "must abstain", "closure does not prove", "remains uncertain", "next valid action is"]
    assert not any(phrase in a.lower() or phrase in b.lower() for phrase in leakage)
    private = re.compile(r"/(Users|home)/|(?:^|\s)~[/\\]|gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}")
    assert not private.search(a + b + (HERE / "common_subject_prompt.md").read_text())
    print(json.dumps(manifest, indent=2))
    print("PASS: deterministic generation, parity inventory, <=5% word budget, B structure boundary, leakage scan, privacy scan")


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if command == "build":
        _, _, manifest = build(write=True)
        print(json.dumps(manifest, indent=2))
    elif command == "validate":
        validate()
    elif command == "determinism":
        before = (A_PATH.read_bytes(), B_PATH.read_bytes(), MANIFEST.read_bytes())
        build(write=True)
        after = (A_PATH.read_bytes(), B_PATH.read_bytes(), MANIFEST.read_bytes())
        assert before == after
        print("PASS: byte-identical regeneration")
    else:
        raise SystemExit("usage: context_builder.py [build|validate|determinism]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
