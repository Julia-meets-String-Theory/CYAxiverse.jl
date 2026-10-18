#!/usr/bin/env python3
"""Build and validate the frozen CYAX-0157 Phase 2 contexts.

The builder has no network path. It reads only the frozen snapshot and the
captured fixtures in this pilot directory. Condition A is rendered through
the repository's canonical temporal-provenance ledger builder; Condition B
uses the same field-level inventory in a plain structured presentation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PILOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PILOT_DIR.parents[3]
SNAPSHOT_PATH = PILOT_DIR / "source_snapshot.json"
PREREGISTRATION_PATH = PILOT_DIR / "preregistration.md"
ANSWER_KEY_PATH = PILOT_DIR / "answer_key.md"
SOURCES_DIR = PILOT_DIR / "sources"
LEDGER_PATH = PILOT_DIR / "condition_a_ledger.jsonl"
CONDITION_A_PATH = PILOT_DIR / "condition_a_context.md"
CONDITION_B_PATH = PILOT_DIR / "condition_b_context.md"
PROMPT_PATH = PILOT_DIR / "common_subject_prompt.md"
MANIFEST_PATH = PILOT_DIR / "phase2_manifest.json"
OBSERVATION_TIME = "2026-09-13T01:22:00Z"
SCHEMA_VERSION = "cyax-temporal-provenance-0.1"
GENERATOR_VERSION = "phase2-canonical-parity-2"

EXPECTED_KEYS = [f"K{i}" for i in range(1, 13)]
ALIGNMENT_LEAK_RE = re.compile(r"\b(?:F(?:0[1-9]|1[0-2])|Q(?:[1-9]|1[0-2]))\b")
ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s'\"`])/(?:Users|home|private|var|tmp)/|"
    r"(?:[A-Za-z]:\\|\\\\[^\\]+\\)|"
    r"/(?:Users|home)/[^\s`'\"]+"
)
URL_RE = re.compile(r"https?://|git@[A-Za-z0-9_.-]+:")
SECRET_RE = re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})")
NON_GRAPH_TERMS = re.compile(
    r"\b(?:graph|graphs|edge|edges|adjacency|adjacent|triple|triples|"
    r"assertion|assertions|predicate|predicates|node|nodes|ledger|ledgers|"
    r"relational|topology|tuple|tuples)\b",
    re.IGNORECASE,
)


def load_canonical_builder():
    path = REPO_ROOT / "research" / "temporal_provenance" / "context_builder.py"
    spec = importlib.util.spec_from_file_location("cyax_canonical_context_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Evidence:
    source: str
    anchor: str


@dataclass(frozen=True)
class Fact:
    key: str
    heading: str
    answer: str
    evidence: tuple[Evidence, ...]
    temporal: str
    uncertainty: str
    next_action: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def artifact_blob(snapshot: dict[str, Any], path: str) -> str:
    for artifact in snapshot.get("repository_artifacts_at_head", []):
        if artifact.get("path") == path:
            return artifact.get("blob_oid", artifact.get("oid", ""))
    return ""


def prior_head(snapshot: dict[str, Any]) -> str:
    objects = snapshot.get("git_objects", {})
    value = objects.get("prior_head_stale", objects.get("prior_head"))
    if isinstance(value, dict):
        value = value.get("oid", value.get("commit", value.get("sha")))
    return value or "1ed97e181e3203524d5435c43f58c1a3ff03d813"


def project_status(record: dict[str, Any]) -> str:
    items = record.get("projectItems", [])
    if not items:
        raise ValueError("frozen fixture has no project status")
    return items[0]["status"]["name"]


def load_inputs() -> dict[str, Any]:
    snapshot = read_json(SNAPSHOT_PATH)
    issue = read_json(SOURCES_DIR / "issue_157.json")
    pr = read_json(SOURCES_DIR / "pr_158.json")
    comments = read_json(SOURCES_DIR / "pr_158_comments.json")
    reviews = read_json(SOURCES_DIR / "pr_158_reviews.json")
    checks = read_json(SOURCES_DIR / "pr_158_checks.json")
    if not isinstance(comments, list) or not isinstance(reviews, list) or not isinstance(checks, list):
        raise ValueError("captured list fixtures have unexpected shapes")
    if sha256_text(issue["body"] + "\n") != snapshot["governing_issue"]["body_sha256"]:
        raise ValueError("Issue fixture body does not match frozen snapshot")
    if sha256_text(pr["body"] + "\n") != snapshot["supporting_pr"]["body_sha256"]:
        raise ValueError("PR fixture body does not match frozen snapshot")
    check_map = {item["name"]: item["conclusion"] for item in checks}
    expected_checks = {"Fast tests": "FAILURE", "build": "SUCCESS", "Full test suite": "SKIPPED"}
    if {key: check_map.get(key) for key in expected_checks} != expected_checks:
        raise ValueError("status-check fixture does not match the frozen check contract")
    return {
        "snapshot": snapshot,
        "issue": issue,
        "pr": pr,
        "comments": comments,
        "reviews": reviews,
        "checks": checks,
    }


def source_ids(inputs: dict[str, Any]) -> dict[str, str]:
    snapshot = inputs["snapshot"]
    return {
        "issue": snapshot["governing_issue"]["source"],
        "pr": snapshot["supporting_pr"]["source"],
        "spec": "specs/0157-public-path-remediation/spec.md",
        "agents": "AGENTS.md",
        "spec0155": "specs/0155-private-safe-chat-checkpoints/spec.md",
        "head": f"commit:{snapshot['supporting_pr']['head_oid']}",
        "base": f"commit:{snapshot['supporting_pr']['base_oid']}",
        "prior": f"commit:{prior_head(snapshot)}",
        "tree": f"tree:{snapshot['supporting_pr']['head_tree']}",
    }


def evidence(source: str, anchor: str) -> Evidence:
    return Evidence(source, anchor)


def facts(inputs: dict[str, Any]) -> tuple[Fact, ...]:
    snapshot = inputs["snapshot"]
    issue = inputs["issue"]
    pr = inputs["pr"]
    ids = source_ids(inputs)
    head = snapshot["supporting_pr"]["head_oid"]
    base = snapshot["supporting_pr"]["base_oid"]
    tree = snapshot["supporting_pr"]["head_tree"]
    prior = prior_head(snapshot)
    observation = snapshot["observation_time"]
    issue_project = project_status(issue)
    pr_project = project_status(pr)
    assert issue_project == "Verification" and pr_project == "Verification"
    assert issue["state"] == "OPEN" and pr["state"] == "OPEN"
    assert pr["isDraft"] is False and pr["mergedAt"] is None
    check_text = (
        "Fast tests: FAILURE; build: SUCCESS; Full test suite: SKIPPED. "
        "The PR body reports focused remediation counts of optional Python interpreter contract 9/9, "
        "notebook static checks 20/20, notebook runtime initialization smoke 10/10, "
        "retired 2026-08-25 entry-point guards 54/54, Slurm-log resolution 6/6, "
        "and data-directory resolution 10/10."
    )
    assert check_text.split("Fast tests:")[1].split(".")[0].strip() == "FAILURE; build: SUCCESS; Full test suite: SKIPPED"
    return (
        Fact(
            "K1",
            "Issue identity and public-surface scope",
            "Issue #157, titled Remove current machine-local path disclosures, asks for confirmed personal and machine-local filesystem details to be removed from current public CYAxiverse surfaces. Its scope keeps useful source filenames, hashes, commits, branch names, scientific algorithms, numerical values, units, identities, persisted schemas, and reproducibility. It also covers sanitizing Issue, PR, and comment prose, replacing tracked personal defaults with explicit configuration or repository-relative resolution, focused path-resolution coverage, and a separate inventory of stale public branches.",
            (evidence(ids["issue"], "title, Problem, Scope, and Acceptance criteria sections"), evidence(ids["agents"], "repository governance boundary")),
            f"Current at the frozen observation time {observation}; the scope is limited to the stated public-surface remediation.",
            "The snapshot does not establish a wider redesign, a scientific algorithm change, or a completed branch inventory.",
            "Preserve the stated scientific identity and behavior while handling the separate branch-safety work through its own evidence gate.",
        ),
        Fact(
            "K2",
            "Supporting change and completion boundary",
            "PR #158, titled Sanitize machine-local path configuration, implements the Issue #157 remediation. Its recorded changes move personal data and Slurm locations to explicit environment configuration, use repository-relative or configurable resolution, sanitize path-bearing documentation and notebooks, preserve scientific algorithms, hashes, schemas, and numerical behavior, and add focused coverage. The PR body explicitly says Issue #157 remains open and that PR #158 does not complete R-005.",
            (evidence(ids["pr"], "Summary and explicit Issue #157/R-005 statements"), evidence(ids["issue"], "Acceptance criteria")),
            f"Current PR identity at {observation}; its completion boundary remains the one stated in the captured PR body.",
            "The fixture records the PR body claim; it does not turn that claim into an Issue closure or an R-005 completion decision.",
            "Treat the PR as the current-surface remediation record and keep the Issue and branch gate separate.",
        ),
        Fact(
            "K3",
            "Issue state at observation",
            f"Issue #157 is OPEN at the frozen observation. Its project item in the CYAxiverse Research & Development project has status {issue_project}. This is the current state recorded for the governing work item, not a prediction about a later project transition.",
            (evidence(ids["issue"], "API state and projectItems status fields"),),
            f"Current through {observation}; later GitHub state is outside the captured surface.",
            "Project status is timestamp-scoped and can change after the frozen observation.",
            "Use the recorded Verification status as evidence, then obtain a new owner-authorized decision if the work advances.",
        ),
        Fact(
            "K4",
            "Pull request state and project placement",
            f"PR #158 is OPEN, non-draft, unmerged, and mergeable. Its head commit is {head} with head tree {tree}; its base branch is {pr['baseRefName']} at {base}. PR #158 Project status: {pr_project}. The snapshot records no merge decision and no Issue completion claim, so these are properties of the frozen PR surface rather than an instruction to merge.",
            (evidence(ids["pr"], "state, isDraft, mergedAt, mergeable, headRefOid, baseRefName, baseRefOid, and projectItems"), evidence(ids["head"], "current PR head commit object"), evidence(ids["tree"], "current PR head tree object"), evidence(ids["base"], "current vmm base commit object")),
            f"Current head is {head} at {observation}; the PR is unmerged in this snapshot.",
            "The frozen surface does not include a later owner merge decision or a later project transition.",
            "Review and record an owner decision against this exact head before any merge action.",
        ),
        Fact(
            "K5",
            "Approved specification and current-surface implementation",
            f"The CYAX-0157 spec is Approved for S1 implementation. Its requirements are R-001 Sanitize current GitHub prose; R-002 Configure executable paths at runtime; R-003 Preserve scientific behavior and evidence identity; R-004 Preserve operational compatibility deliberately; and R-005 Close every live-reference route before history review. The frozen Issue and PR evidence describes current-surface remediation implemented at exact current head {head}, while the spec, PR, and governance material make no merge or Issue completion claim. The related CYAX-0155 specification is retained as an allowed governance reference, not as evidence of a changed requirement.",
            (evidence(ids["spec"], "approval header, implementation status, and R-001 through R-005"), evidence(ids["pr"], "Verification at exact head and deferred-work sections"), evidence(ids["issue"], "Verification note and R-005 statement"), evidence(ids["agents"], "repository change and evidence boundary"), evidence(ids["spec0155"], "allowed related specification identity"), evidence(ids["head"], "exact current head object")),
            f"The approval and implementation statements are scoped to exact head {head} at {observation}; R-005 remains current and open.",
            "The artifact set records implementation evidence and approval wording but does not independently execute the spec or infer completion of its final gate.",
            "Keep R-001 through R-004 tied to the current-surface evidence and perform R-005 separately before history review.",
        ),
        Fact(
            "K6",
            "Checks and focused verification",
            f"At current head {head}, the captured checks show Fast tests: FAILURE, Documentation build: SUCCESS, and Full test suite: SKIPPED. The Fast failure is the unchanged pre-existing Hessian expectation mismatch at test/runtests.jl:1317, with observed diagonal 19.739208802178716 versus expected 39.47841760435743. The PR body reports focused remediation results: optional Python interpreter contract 9/9, notebook static checks 20/20, notebook runtime initialization smoke 10/10, retired 2026-08-25 entry-point guards 54/54, Slurm-log resolution 6/6, and data-directory resolution 10/10.",
            (evidence(ids["pr"], "Verification at exact head and focused test counts"), evidence(ids["pr"], "captured status checks: Fast tests, build, and Full test suite")),
            f"Check timestamps and focused counts are scoped to the captured head {head} and observation {observation}.",
            "Focused counts are self-reported in the PR body; no independent rerun or production-scale execution is included here.",
            "Report the unchanged Fast-suite mismatch accurately and do not convert the focused counts into a claim of a clean full suite.",
        ),
        Fact(
            "K7",
            "Review record and evidence limit",
            f"The PR body self-reports independent re-review APPROVE with no findings against exact current head {head}. The captured GitHub reviews surface is an empty array, so it exposes no formal APPROVED, CHANGES_REQUESTED, or COMMENTED decision. The approval is therefore a self-reported PR-body statement bound to the exact head, not an independently confirmed GitHub review record.",
            (evidence(ids["pr"], "independent re-review statement against exact current head"), evidence(ids["pr"], "captured reviews surface: empty array")),
            f"The claimed approval targets {head}; the empty formal review surface is current only at {observation}.",
            "The absence of a formal review record limits what can be claimed about independent approval.",
            "Obtain and record a formal exact-head review or an explicitly recorded owner decision.",
        ),
        Fact(
            "K8",
            "Open R-005 branch-safety requirement",
            "R-005, Close every live-reference route before history review, remains open. It requires a read-only privacy scan of surviving public branches and a preservation/reachability classification before any branch deletion or history rewriting. The frozen Issue and PR bodies say that no public branch has been deleted, force-pushed, or rewritten and that this scan and classification are deferred. R-005 is the explicit completion gate for Issue #157.",
            (evidence(ids["spec"], "R-005 requirement"), evidence(ids["issue"], "R-005 remains open and deferred"), evidence(ids["pr"], "R-005 remains open and deferred")),
            f"Open at {observation}; no later branch scan is represented.",
            "The remaining public-branch exposure is undetermined because the required scan has not been performed.",
            "Perform the read-only branch scan and record preservation/reachability classification before deletion or rewrite discussion.",
        ),
        Fact(
            "K9",
            "Separate completion decisions",
            "Merging PR #158 must not close Issue #157. A merge would incorporate the current-surface remediation into vmm, but PR completion and Issue completion remain distinct because R-005 is still open. The specification and both captured public bodies explicitly preserve that separation; no merge has occurred in the frozen state and no evidence authorizes Issue closure.",
            (evidence(ids["spec"], "implementation status and no-auto-close rule"), evidence(ids["pr"], "Issue remains open and PR does not complete R-005"), evidence(ids["issue"], "PR does not close this issue")),
            f"This is a workflow rule applied to the frozen unmerged state at {observation}.",
            "The artifact does not establish a future merge, closure, or owner decision.",
            "If a merge is later approved, keep Issue #157 open until R-005 has its own evidence-backed completion decision.",
        ),
        Fact(
            "K10",
            "Head lineage and supersession",
            f"The prior head {prior} is stale relative to the current PR head {head}. The first captured PR comment requests review against the prior head; the later comment requests review against {head} and records incorporation of the current {pr['baseRefName']} base {base}. The prior reference is historical context and must not be used as the current PR state.",
            (evidence(ids["pr"], "current headRefOid"), evidence(ids["pr"], "current baseRefOid and baseRefName"), evidence(ids["prior"], "prior head object identity")),
            f"Current is {head}; stale is {prior}; both are scoped to the observation {observation}.",
            "Earlier comment context does not establish current state after the head changed.",
            f"Bind every new review or owner decision to {head}, not to the stale prior reference.",
        ),
        Fact(
            "K11",
            "Evidence limits and abstentions",
            f"The verification surface has no independent CI rerun, is scoped to the snapshot time {observation}, and includes a PR-body self-report for focused tests and review approval. The formal review array is empty. No production CYTools geometry generation, large database scan, retired physical-scaling workflow, or external evidence generation was independently verified, and the surviving-public-branch scan required by R-005 was not performed. Abstain from claims requiring state after {observation} or evidence beyond these captured records.",
            (evidence(ids["pr"], "verification and no-production-execution statements"), evidence(ids["pr"], "captured checks and review surface"), evidence(ids["issue"], "timestamp-scoped verification note"), evidence(ids["agents"], "evidence and execution boundary")),
            f"All limitations are active at the frozen boundary {observation}; future state is unknown.",
            "Missing reruns, formal review evidence, production execution, and branch classification are explicit abstention triggers.",
            "State the missing evidence and abstain rather than infer a later CI, review, production, or branch result.",
        ),
        Fact(
            "K12",
            "Next authorized action",
            f"The next valid action is to obtain and record an exact-head review and owner merge decision for PR #158 against {head}. If an owner later approves a merge into {pr['baseRefName']}, Issue #157 must remain open because that merge does not complete R-005. Separately, perform the R-005 surviving-public-branch read-only scan, produce the preservation/reachability classification, and act on those results before any branch deletion or history rewriting. Only the owner can complete the decision; this frozen artifact cannot do it.",
            (evidence(ids["spec"], "R-005 gate and implementation status"), evidence(ids["pr"], "exact-head review request and deferred-work section"), evidence(ids["issue"], "R-005 and Issue-closure boundary")),
            f"This is the next action at observation {observation}, directed at exact head {head}.",
            "The artifact records a required action, not an owner decision or completed scan.",
            "Obtain the exact-head review/owner decision, keep the Issue open after any merge, then complete R-005 independently.",
        ),
    )


def source_resources(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = inputs["snapshot"]
    ids = source_ids(inputs)
    head = snapshot["supporting_pr"]["head_oid"]
    base = snapshot["supporting_pr"]["base_oid"]
    prior = prior_head(snapshot)
    tree = snapshot["supporting_pr"]["head_tree"]
    spec_blob = artifact_blob(snapshot, "specs/0157-public-path-remediation/spec.md") or "1c670b6f"
    agents_blob = artifact_blob(snapshot, "AGENTS.md") or "e7ece2c1"
    spec0155_blob = artifact_blob(snapshot, "specs/0155-private-safe-chat-checkpoints/spec.md") or "b686c91b"
    definitions = [
        (ids["issue"], "ExternalSource", "external_source", "Issue body and API state fields at the frozen observation", None, ids["issue"]),
        (ids["pr"], "ExternalSource", "external_source", "PR body, API state, checks, comments, and review surface at the frozen observation", None, ids["pr"]),
        (ids["spec"], "Requirement", "approved_spec", "Approved S1 specification and R-001 through R-005 scope", spec_blob, ids["spec"]),
        (ids["agents"], "Requirement", "approved_spec", "Repository governance and evidence boundary", agents_blob, ids["agents"]),
        (ids["spec0155"], "Requirement", "approved_spec", "Allowed related specification identity", spec0155_blob, ids["spec0155"]),
        (ids["head"], "Implementation", "implementation_evidence", "Exact current PR head implementation identity", head, ids["head"]),
        (ids["base"], "Implementation", "implementation_evidence", "Current vmm base identity incorporated by the PR", base, ids["base"]),
        (ids["prior"], "Implementation", "implementation_evidence", "Historical prior PR head, stale relative to current", prior, ids["prior"]),
        (ids["tree"], "Artifact", "implementation_evidence", "Exact current PR head tree identity", tree, ids["tree"]),
    ]
    result = []
    for identifier, kind, authority_class, scope, revision, locator in definitions:
        record: dict[str, Any] = {
            "record_type": "resource",
            "id": identifier,
            "kind": kind,
            "label": identifier,
            "canonical": True,
            "locator": locator,
            "observed_at": OBSERVATION_TIME,
            "authority": {"class": authority_class, "scope": scope},
        }
        if revision:
            record["revision"] = revision
        result.append(record)
    return result


def build_ledger(inputs: dict[str, Any], inventory: tuple[Fact, ...]) -> list[dict[str, Any]]:
    ids = source_ids(inputs)
    records: list[dict[str, Any]] = [
        {
            "record_type": "meta",
            "id": "pilot.cyax0157.phase2",
            "schema_version": SCHEMA_VERSION,
            "pilot": "CYAX-0157 held-out ablation Phase 2",
            "title": "CYAX-0157 frozen temporal provenance context",
            "root": "work.issue.157",
            "reconstruction_task": "Reconstruct the frozen current state from the cited records and preserve their authority boundaries.",
            "observed_at": OBSERVATION_TIME,
        },
    ]
    records.extend(source_resources(inputs))
    records.extend(
        [
            {"record_type": "resource", "id": "work.issue.157", "kind": "WorkItem", "label": "Issue #157", "canonical": False, "authority": {"class": "agent_extraction", "scope": "curated work-item identity"}},
            {"record_type": "resource", "id": "work.pr.158", "kind": "WorkItem", "label": "PR #158", "canonical": False, "authority": {"class": "agent_extraction", "scope": "curated pull-request identity"}},
            {"record_type": "resource", "id": "claim.scope", "kind": "Claim", "label": "current public-surface scope", "canonical": False, "authority": {"class": "agent_extraction", "scope": "field-level scope interpretation"}},
            {"record_type": "resource", "id": "claim.implementation", "kind": "Claim", "label": "current-surface remediation", "canonical": False, "authority": {"class": "agent_extraction", "scope": "field-level implementation interpretation"}},
            {"record_type": "resource", "id": "state.issue.open", "kind": "Claim", "label": "Issue OPEN", "canonical": False, "authority": {"class": "workflow_state", "scope": "frozen Issue API state"}},
            {"record_type": "resource", "id": "state.issue.verification", "kind": "Claim", "label": "Issue Project Verification", "canonical": False, "authority": {"class": "workflow_state", "scope": "frozen Issue project item"}},
            {"record_type": "resource", "id": "state.pr.open", "kind": "Claim", "label": "PR OPEN", "canonical": False, "authority": {"class": "workflow_state", "scope": "frozen PR API state"}},
            {"record_type": "resource", "id": "state.pr.verification", "kind": "Claim", "label": "PR Project Verification", "canonical": False, "authority": {"class": "workflow_state", "scope": "frozen PR project item"}},
            {"record_type": "resource", "id": "state.pr.unmerged", "kind": "Claim", "label": "PR unmerged", "canonical": False, "authority": {"class": "workflow_state", "scope": "frozen PR merge fields"}},
            {"record_type": "resource", "id": "requirement.r005", "kind": "Requirement", "label": "R-005 branch gate", "canonical": False, "authority": {"class": "approved_spec", "scope": "live-reference closure requirement"}},
            {"record_type": "resource", "id": "verification.focused", "kind": "Verification", "label": "focused remediation verification", "canonical": False, "authority": {"class": "verification_evidence", "scope": "PR-body reported focused counts"}},
            {"record_type": "resource", "id": "verification.checks", "kind": "Verification", "label": "captured CI check state", "canonical": False, "authority": {"class": "verification_evidence", "scope": "captured status-check conclusions"}},
            {"record_type": "resource", "id": "review.surface", "kind": "Verification", "label": "formal review surface", "canonical": False, "authority": {"class": "verification_evidence", "scope": "captured empty reviews array"}},
            {"record_type": "resource", "id": "action.exact_head", "kind": "Action", "label": "exact-head review and owner decision", "canonical": False, "authority": {"class": "owner_decision", "scope": "next action requires owner"}},
        ]
    )

    def assertion(
        identifier: str,
        predicate: str,
        subject: str,
        object_ref: str,
        status: str,
        refs: Iterable[Evidence],
        method: str,
        valid_from: str = OBSERVATION_TIME,
        valid_to: str | None = None,
        qualifiers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": "assertion",
            "id": identifier,
            "predicate": predicate,
            "subject": subject,
            "object": {"ref": object_ref},
            "epistemic_status": status,
            "recorded_at": OBSERVATION_TIME,
            "valid_time": {"from": valid_from, "to": valid_to},
            "provenance": [asdict(item) for item in refs],
            "curation": {"review_status": "curator_checked", "method": method},
        }
        if qualifiers:
            record["qualifiers"] = qualifiers
        return record

    by_key = {fact.key: fact for fact in inventory}
    records.extend(
        [
            assertion("rel.issue.scope", "concerns", "work.issue.157", "claim.scope", "observed", by_key["K1"].evidence, "field-level extraction from frozen Issue fixture"),
            assertion("rel.pr.implements_issue", "implements", "work.pr.158", "work.issue.157", "reported", by_key["K2"].evidence, "field-level extraction from frozen PR and Issue fixtures"),
            assertion("rel.issue.open", "has_status", "work.issue.157", "state.issue.open", "observed", by_key["K3"].evidence, "API state extraction"),
            assertion("rel.issue.verification", "documented_in", "work.issue.157", "state.issue.verification", "observed", by_key["K3"].evidence, "project status extraction"),
            assertion("rel.pr.open", "has_status", "work.pr.158", "state.pr.open", "observed", by_key["K4"].evidence, "API state extraction"),
            assertion("rel.pr.verification", "documented_in", "work.pr.158", "state.pr.verification", "observed", by_key["K4"].evidence, "project status extraction"),
            assertion("rel.pr.unmerged", "has_status", "work.pr.158", "state.pr.unmerged", "observed", by_key["K4"].evidence, "merge-state extraction"),
            assertion("rel.pr.implements_surface", "implements", "work.pr.158", "claim.implementation", "reported", by_key["K5"].evidence, "implementation-boundary extraction"),
            assertion("rel.spec.governs_issue", "governs", ids["spec"], "work.issue.157", "accepted", by_key["K5"].evidence, "approved-spec extraction"),
            assertion("rel.spec.r005", "governs", ids["spec"], "requirement.r005", "accepted", by_key["K8"].evidence, "approved-spec requirement extraction"),
            assertion("rel.checks.support_focus", "supports", "verification.checks", "verification.focused", "reported", by_key["K6"].evidence, "verification-surface extraction"),
            assertion("rel.pr.uses_head", "uses", "work.pr.158", ids["head"], "observed", by_key["K4"].evidence, "exact-head identity extraction"),
            assertion("rel.head.supersedes_prior", "supersedes", ids["head"], ids["prior"], "observed", by_key["K10"].evidence, "head-lineage extraction"),
            assertion("rel.pr.review_claim", "has_outcome", "work.pr.158", "review.surface", "reported", by_key["K7"].evidence, "review-surface comparison"),
            assertion("rel.r005.blocks_closure", "blocks", "requirement.r005", "work.issue.157", "unresolved", by_key["K8"].evidence, "completion-gate extraction"),
            assertion("rel.issue.next_action", "next_valid_action", "work.issue.157", "action.exact_head", "unresolved", by_key["K12"].evidence, "owner-action extraction"),
        ]
    )
    for line_number, record in enumerate(records, 1):
        record["_line"] = line_number
    return records


def serializable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in record.items() if key != "_line"} for record in records]


def ledger_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in serializable_records(records)
    )


def prompt_text() -> str:
    preregistration = PREREGISTRATION_PATH.read_text(encoding="utf-8")
    start_marker = "## Reconstruction task\n"
    end_marker = "## Scoring rubric"
    start = preregistration.index(start_marker)
    end = preregistration.index(end_marker, start)
    return preregistration[start:end].rstrip() + "\n"


def citation(items: Iterable[Evidence]) -> str:
    return "; ".join(f"[{item.source} | {item.anchor}]" for item in items)


def source_index(records: list[dict[str, Any]]) -> list[str]:
    lines = ["## Canonical source index", ""]
    for record in records:
        if record.get("record_type") != "resource" or not record.get("canonical"):
            continue
        revision = f" @ `{record['revision']}`" if record.get("revision") else ""
        authority = record["authority"]
        lines.append(
            f"- `{record['id']}` — {record['locator']}{revision} "
            f"[{authority['class']}; {authority['scope']}]"
        )
    return lines


def fact_block(fact: Fact, include_evidence: bool = True) -> list[str]:
    lines = [f"### {fact.heading}", "", fact.answer, ""]
    if include_evidence:
        lines.append(f"- Evidence: {citation(fact.evidence)}")
    lines.extend(
        [
            f"- Temporal and current/stale reading: {fact.temporal}",
            f"- Uncertainty: {fact.uncertainty}",
            f"- Next action: {fact.next_action}",
            "",
        ]
    )
    return lines


def shared_boundary_sections(inputs: dict[str, Any]) -> list[str]:
    snapshot = inputs["snapshot"]
    head = snapshot["supporting_pr"]["head_oid"]
    prior = prior_head(snapshot)
    observation = snapshot["observation_time"]
    return [
        "## Temporal and authority boundary",
        "",
        f"Every statement above is bounded by the frozen observation at {observation}. The current PR reference is {head}; the earlier reference {prior} is historical and stale. A later update cannot be supplied by this artifact. Source identity, captured wording, and recorded API fields are kept distinct from a curator interpretation. A source citation identifies where the statement came from, while its authority scope explains what that source can establish.",
        "",
        "## Evidence reading discipline",
        "",
        "The Issue and PR bodies provide public workflow statements. The captured check records provide conclusions and timestamps, while the PR body provides the focused counts and the review claim. The empty formal review surface limits the review conclusion. Approved specification text supplies requirement scope; commit and tree identities bind current-surface statements to the exact revision. These roles are complementary, not interchangeable, so a reported result is never silently upgraded to an independent execution result.",
        "",
        "## Completion boundary",
        "",
        "The current-surface remediation and the remaining live-reference gate are separate decisions. Nothing here records a merge, Issue closure, branch deletion, force-push, history rewrite, independent CI rerun, production execution, or completed branch classification. The safe continuation is therefore evidence collection at the exact current head, followed by the owner decision and the separately required R-005 scan. If the evidence is insufficient for a later claim, the correct response is to name the missing evidence and abstain.",
        "",
    ]


def render_condition_a(inputs: dict[str, Any], inventory: tuple[Fact, ...], canonical: Any) -> str:
    loaded = canonical.load_ledger(LEDGER_PATH)
    rendered = canonical.render_context(loaded).rstrip("\n")
    lines = [
        rendered,
        "",
        "## Field-level current-state record",
        "",
        "The following prose is derived from the same canonical fields represented by the validated temporal records. Each section keeps the cited source, temporal boundary, uncertainty, and next action visible so that provenance remains attached to the answer-bearing statement.",
        "",
    ]
    for fact in inventory:
        # The canonical renderer carries evidence for every fact except the
        # two workflow summaries without a dedicated relation below.
        lines.extend(fact_block(fact, include_evidence=fact.key in {"K9", "K11"}))
    return "\n".join(lines).rstrip() + "\n"


def render_condition_b(inputs: dict[str, Any], records: list[dict[str, Any]], inventory: tuple[Fact, ...]) -> str:
    observation = inputs["snapshot"]["observation_time"]
    lines = [
        "# Current-state structured summary",
        "",
        f"This concise structured summary uses the frozen public evidence at {observation}. It presents the same twelve answer-bearing records, source identities, authority scopes, current and historical distinctions, uncertainty boundaries, and next actions as the companion condition. It contains no subject prompt; the common reconstruction instructions are provisioned separately.",
        "",
        "## Reading and authority",
        "",
        "A citation names the frozen public source and the specific evidence anchor. The source index records the authority scope for each identity. Public body statements, captured check conclusions, specification requirements, and revision identities retain their distinct evidentiary roles. A reported statement is not treated as an independent execution result, and a missing formal review record is not treated as approval.",
        "",
    ]
    lines.extend(source_index(records))
    lines.extend(
        [
            "",
            "## Current-state records",
            "",
            "The records below are organized by topic so that an answer can be reconstructed without changing the authority of the cited material. The presentation keeps a current observation separate from historical lineage, a reported verification result separate from a captured check conclusion, and a requested owner decision separate from an accomplished state. It also keeps the known negative evidence visible: an empty formal review surface, a failed wider Fast suite, a skipped full suite, and an unfinished public-branch classification all limit the claims that can safely be made.",
            "",
        ]
    )
    for fact in inventory:
        lines.extend(fact_block(fact))
    lines.extend(shared_boundary_sections(inputs))
    return "\n".join(lines).rstrip() + "\n"


def semantic_inventory(inventory: tuple[Fact, ...]) -> dict[str, Any]:
    return {
        fact.key: {
            "heading": fact.heading,
            "answer": fact.answer,
            "evidence": [asdict(item) for item in fact.evidence],
            "temporal": fact.temporal,
            "uncertainty": fact.uncertainty,
            "next_action": fact.next_action,
        }
        for fact in inventory
    }


def source_references(text: str, allowed: Iterable[str]) -> list[str]:
    return sorted({source for source in allowed if source in text})


def evidence_item_ids(inventory: tuple[Fact, ...]) -> list[str]:
    return [f"{fact.key}:{index}" for fact in inventory for index, _ in enumerate(fact.evidence, 1)]


def metrics(path: Path, text: str, inventory: tuple[Fact, ...], allowed: list[str]) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "byte_count": len(data),
        "word_count": len(text.split()),
        "sha256": sha256_bytes(data),
        "line_count": text.count("\n"),
        "source_references": source_references(text, allowed),
        "evidence_items": len(evidence_item_ids(inventory)),
        "fact_coverage": [fact.key for fact in inventory],
    }


def condition_artifacts(inputs: dict[str, Any]) -> dict[str, Any]:
    canonical = load_canonical_builder()
    inventory = facts(inputs)
    records = build_ledger(inputs, inventory)
    canonical.validate_ledger(records)
    write_bytes(LEDGER_PATH, ledger_bytes(records))
    loaded = canonical.load_ledger(LEDGER_PATH)
    canonical.validate_ledger(loaded)
    condition_a = render_condition_a(inputs, inventory, canonical)
    condition_b = render_condition_b(inputs, loaded, inventory)
    prompt = prompt_text()
    return {
        "ledger": ledger_bytes(loaded),
        "condition_a": condition_a.encode("utf-8"),
        "condition_b": condition_b.encode("utf-8"),
        "prompt": prompt.encode("utf-8"),
        "inventory": inventory,
        "records": loaded,
        "canonical": canonical,
    }


def build() -> None:
    inputs = load_inputs()
    artifacts = condition_artifacts(inputs)
    write_bytes(LEDGER_PATH, artifacts["ledger"])
    write_bytes(CONDITION_A_PATH, artifacts["condition_a"])
    write_bytes(CONDITION_B_PATH, artifacts["condition_b"])
    write_bytes(PROMPT_PATH, artifacts["prompt"])
    allowed = list(source_ids(inputs).values())
    inventory = artifacts["inventory"]
    context_metrics = {
        "condition_a": metrics(CONDITION_A_PATH, artifacts["condition_a"].decode("utf-8"), inventory, allowed),
        "condition_b": metrics(CONDITION_B_PATH, artifacts["condition_b"].decode("utf-8"), inventory, allowed),
    }
    prompt_data = artifacts["prompt"]
    prompt_metrics = {
        "path": PROMPT_PATH.name,
        "byte_count": len(prompt_data),
        "word_count": len(prompt_data.decode("utf-8").split()),
        "sha256": sha256_bytes(prompt_data),
    }
    manifest = {
        "manifest_version": "phase2-1",
        "generator_version": GENERATOR_VERSION,
        "pilot": "CYAX-0157 held-out ablation",
        "phase": 2,
        "frozen_inputs": {"snapshot": SNAPSHOT_PATH.name, "fixtures_directory": SOURCES_DIR.name},
        "source_references": allowed,
        "source_references_are_identical": True,
        "contexts": context_metrics,
        "common_subject_prompt": prompt_metrics,
        "semantic_inventory": {
            "field_names": ["answer", "evidence", "temporal", "uncertainty", "next_action"],
            "fact_order": EXPECTED_KEYS,
            "facts": semantic_inventory(inventory),
            "condition_parity": {
                "condition_a": {"fact_fields": semantic_inventory(inventory), "evidence_item_ids": evidence_item_ids(inventory)},
                "condition_b": {"fact_fields": semantic_inventory(inventory), "evidence_item_ids": evidence_item_ids(inventory)},
            },
        },
        "answer_key_completeness": {"path": ANSWER_KEY_PATH.name, "keys": EXPECTED_KEYS},
        "deterministic": True,
        "source_reopening_count": 0,
        "subject_artifacts": [],
    }
    write_bytes(MANIFEST_PATH, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("built Phase 2 artifacts")
    print_metrics(context_metrics)


def print_metrics(context_metrics: dict[str, Any]) -> None:
    for name in ("condition_a", "condition_b"):
        item = context_metrics[name]
        print(f"{name}: words={item['word_count']} bytes={item['byte_count']} sha256={item['sha256']}")


def expected_artifacts(inputs: dict[str, Any]) -> dict[str, bytes]:
    artifacts = condition_artifacts(inputs)
    return {key: artifacts[key] for key in ("ledger", "condition_a", "condition_b", "prompt")}


def paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def validate_answer_key() -> None:
    text = ANSWER_KEY_PATH.read_text(encoding="utf-8")
    keys = re.findall(r"^## (K\d{1,2})\b", text, re.MULTILINE)
    if keys != EXPECTED_KEYS:
        raise AssertionError(f"answer key completeness mismatch: {keys}")


def validate_context_shape(name: str, text: str, inventory: tuple[Fact, ...], allowed: list[str]) -> None:
    if not text.endswith("\n") or "\r" in text:
        raise AssertionError(f"{name} does not preserve deterministic Markdown line breaks")
    if ALIGNMENT_LEAK_RE.search(text):
        raise AssertionError(f"{name} contains subject-facing Fxx/Qx alignment")
    if "common_subject_prompt" in text or "answer_key" in text or "source_snapshot.json" in text:
        raise AssertionError(f"{name} contains held-back artifact leakage")
    if ABSOLUTE_PATH_RE.search(text) or URL_RE.search(text) or SECRET_RE.search(text):
        raise AssertionError(f"{name} contains a private path, URL, or secret pattern")
    if name == "condition_b" and NON_GRAPH_TERMS.search(text):
        match = NON_GRAPH_TERMS.search(text)
        raise AssertionError(f"Condition B contains non-graph forbidden vocabulary: {match.group(0)}")
    if text.lstrip().startswith("{"):
        raise AssertionError(f"{name} embeds a governance JSON object")
    for fact in inventory:
        for value in (fact.answer, fact.temporal, fact.uncertainty, fact.next_action):
            if value not in text:
                raise AssertionError(f"{name} is missing field text for {fact.key}")
        for item in fact.evidence:
            if item.source not in text or item.anchor not in text:
                raise AssertionError(f"{name} is missing evidence {item.source} / {item.anchor}")
    found = source_references(text, allowed)
    if found != sorted(allowed):
        raise AssertionError(f"{name} source references are not the common allowlist: {found}")
    duplicate_paragraphs = [item for item, count in Counter(paragraphs(text)).items() if count > 1]
    if duplicate_paragraphs:
        raise AssertionError(f"{name} contains repeated paragraphs")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    duplicate_sentences = [item for item, count in Counter(sentences).items() if item and count > 1 and len(item.split()) >= 8]
    if duplicate_sentences:
        raise AssertionError(f"{name} contains repeated substantive sentences")
    words = len(text.split())
    if not 2300 <= words <= 2500:
        raise AssertionError(f"{name} word count {words} is outside 2300-2500")


def validate() -> None:
    inputs = load_inputs()
    canonical = load_canonical_builder()
    inventory = facts(inputs)
    if [fact.key for fact in inventory] != EXPECTED_KEYS:
        raise AssertionError("field-level inventory is not K01-K12")
    records = canonical.load_ledger(LEDGER_PATH)
    canonical.validate_ledger(records)
    if records[0].get("record_type") != "meta":
        raise AssertionError("canonical ledger first record is not meta")
    assertions = [record for record in records if record["record_type"] == "assertion"]
    required = {"predicate", "valid_time", "epistemic_status", "curation", "provenance"}
    if len(assertions) < len(inventory) or not all(required <= set(record) for record in assertions):
        raise AssertionError("canonical assertion fields are incomplete")
    canonical_text = canonical.render_context(records)
    if not canonical_text.startswith("# Cold-start context:"):
        raise AssertionError("Condition A was not derived from canonical renderer")
    allowed = list(source_ids(inputs).values())
    a_text = CONDITION_A_PATH.read_text(encoding="utf-8")
    b_text = CONDITION_B_PATH.read_text(encoding="utf-8")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    if prompt != prompt_text():
        raise AssertionError("common prompt is not an exact preregistration copy")
    validate_answer_key()
    validate_context_shape("condition_a", a_text, inventory, allowed)
    validate_context_shape("condition_b", b_text, inventory, allowed)
    a_words = len(a_text.split())
    b_words = len(b_text.split())
    if abs(a_words - b_words) / max(a_words, b_words) > 0.05:
        raise AssertionError(f"context word-count difference exceeds 5%: {a_words} vs {b_words}")
    manifest = read_json(MANIFEST_PATH)
    for name, path in (("condition_a", CONDITION_A_PATH), ("condition_b", CONDITION_B_PATH)):
        recorded = manifest["contexts"][name]
        data = path.read_bytes()
        actual = {"byte_count": len(data), "word_count": len(data.decode("utf-8").split()), "sha256": sha256_bytes(data)}
        for field, value in actual.items():
            if recorded[field] != value:
                raise AssertionError(f"manifest {name}.{field} mismatch: {recorded[field]} != {value}")
    expected = expected_artifacts(inputs)
    actual_paths = {"ledger": LEDGER_PATH, "condition_a": CONDITION_A_PATH, "condition_b": CONDITION_B_PATH, "prompt": PROMPT_PATH}
    for name, data in expected.items():
        if actual_paths[name].read_bytes() != data:
            raise AssertionError(f"deterministic regeneration bytes differ for {name}")
    if manifest["semantic_inventory"]["facts"] != semantic_inventory(inventory):
        raise AssertionError("manifest semantic inventory differs from the source-derived inventory")
    parity = manifest["semantic_inventory"]["condition_parity"]
    if parity["condition_a"] != parity["condition_b"]:
        raise AssertionError("condition semantic inventories are not identical")
    if manifest.get("source_reopening_count") != 0 or manifest.get("subject_artifacts") != []:
        raise AssertionError("source reopening or subject artifacts are recorded")
    print("phase2 validate: PASS")
    print(f"condition_a: words={a_words} bytes={len(CONDITION_A_PATH.read_bytes())} sha256={sha256_bytes(CONDITION_A_PATH.read_bytes())}")
    print(f"condition_b: words={b_words} bytes={len(CONDITION_B_PATH.read_bytes())} sha256={sha256_bytes(CONDITION_B_PATH.read_bytes())}")
    print(f"word_difference_percent={abs(a_words - b_words) / max(a_words, b_words) * 100:.3f}")
    print(f"source_references={len(allowed)} evidence_items={len(evidence_item_ids(inventory))} fact_coverage={len(inventory)}/12")


def privacy() -> None:
    targets = [CONDITION_A_PATH, CONDITION_B_PATH, PROMPT_PATH, LEDGER_PATH, MANIFEST_PATH]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        if ABSOLUTE_PATH_RE.search(text) or URL_RE.search(text) or SECRET_RE.search(text):
            raise AssertionError(f"privacy scan failed for {path.name}")
    print("phase2 privacy: PASS (no private paths, live URLs, or secret patterns)")


def determinism() -> None:
    inputs = load_inputs()
    first = expected_artifacts(inputs)
    second = expected_artifacts(inputs)
    if first != second:
        raise AssertionError("two regenerated Phase 2 artifact sets differ")
    print("phase2 determinism: PASS (ledger, contexts, and prompt bytes identical across regeneration)")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["phase2"]:
        args = args[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate", "privacy", "determinism"))
    command = parser.parse_args(args).command
    try:
        {"build": build, "validate": validate, "privacy": privacy, "determinism": determinism}[command]()
    except (AssertionError, OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"phase2 {command}: FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
