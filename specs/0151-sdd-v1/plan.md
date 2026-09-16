# Implementation Plan — CYAX-0151

## 1. Governing specification

Canonical path: `specs/0151-sdd-v1/spec.md`

Status: **Approved**

Owner approval: 2026-09-10, durably recorded on GitHub Issue #151.

Approved peer-review clarification: `specs/0151-sdd-v1/clarification-a.md`, with approval provenance on Issue #151 comment `5624347708`. Where that clarification conflicts with earlier wording in `spec.md`, the clarification controls while earlier wording remains historical context.

The first implementation change materialized the approved specification as a tracked repository file. No repository changes preceded the approved planning stage.

## 2. Implementation strategy

Implement CYAX-0151 as independently reviewable gates:

```text
G0  Approve SDD contract              PASS
 |
 v
G1  Repository SDD layer              PASS via PR #152
 |
 v
G2  GitHub Project                    project configuration + active-work reconciliation
 |
 v
G3  #148/#149 migration               separate process-only PR
 |
 v
G4  Six-week development backfill     Project metadata/reconciliation
 |
 v
G5  Usage evaluation                  after 6–8 weeks + sufficient real samples
```

This keeps the control-plane change small and prevents the SDD pilot or historical backfill from becoming prerequisites for landing the core templates.

## 3. Requirement coverage

| Requirement | Planned realization | Verification |
| --- | --- | --- |
| R-001 Constitution precedence | Minimal SDD section in `AGENTS.md`; SDD skill explicitly defers to it | Diff review for duplicated/conflicting policy |
| R-002 Spec precedence | `AGENTS.md`, SDD skill, spec template and PR convergence checklist | Cross-artifact review |
| R-003 No competing ledger | No active-work `specs/README`; Project is live map | Repository/file review |
| R-004 Risk beats line count | S0–S3 rules in skill + intake form; classification ambiguity affecting scientific/contract meaning fails closed to provisional S2 | Template/skill inspection |
| R-005 Intent changes are spec changes | Agent stop/re-review rules | Skill + AGENTS review |
| R-006 Tasks not automatically Issues | Explicit task/sub-issue policy; tasks are not live merge/status state | Skill/template review |
| R-007 Scientific claim boundary | Required S2/S3 spec-template section | Template inspection |
| R-008 Normative ambiguity | Explicit escalation rule | AGENTS + skill review |
| R-009 Negative results | Scientific completion guidance | Spec template + skill |
| R-010 Partial gate PRs | PR template supports requirement/gate subsets and S0 N/A/omission path | PR-template inspection |
| R-011 Unmet requirements visible | Convergence checklist/process | PR template + skill |
| R-012 Minimal AGENTS amendment | Small normative section only | Diff-size/content review |
| R-013 On-demand SDD skill | New canonical skill + existing adapter pattern | Symlink/content verification |
| R-014 Preserve #148 contract | Dedicated migration after G1/G2 using chronological decision/supersession record | Manual scientific diff/convergence |
| R-015 Preserve gate history | G0/G1 represented retrospectively, not rewritten; new IDs not projected backward | #148/#149 migration review |
| R-016 No pilot science changes | Process-only #148 migration | Changed-file + semantic review |

## 4. G1 — Repository SDD layer

G1 is complete via PR #152 and remains accepted. The peer-review clarification does not reopen or invalidate it.

The merged G1 layer established:

- `specs/0151-sdd-v1/{spec.md,plan.md,tasks.md}`;
- canonical `.agents/skills/cyaxiverse-sdd` plus tracked Claude/Codex symlink mirrors;
- minimal SDD language in `AGENTS.md`;
- `.github/ISSUE_TEMPLATE/sdd_work.yml` while retaining bug-report intake;
- `.github/pull_request_template.md`;
- concise SDD guidance in `docs/AI_AGENT_AND_GIT_WORKFLOW.md`.

G1 verification and the transparently unavailable local `git diff --check`/wrapper execution remain recorded in the historical `tasks.md` and Issue/PR evidence. No retrospective rerun is required by Clarification A.

## 5. G2 — GitHub Project

Create **CYAxiverse Research & Development** and configure the approved Project semantics.

### 5.1 Primary item model

For S1–S3, the governing Issue is the primary Project item. Linked PRs provide implementation/integration/evidence and should not normally become duplicate WIP cards on the main workflow board.

For S0, an Issue remains optional and a PR-only change may itself be the Project item.

A dedicated PR / Integration view may expose linked PRs without counting them as duplicate spec-level WIP.

### 5.2 Workflow

Use:

```text
Backlog
→ Specifying
→ Spec Review
→ Ready
→ Implementing
→ Verification
→ Done
```

Do not force every S0/S1 item through every state. S2 normally follows the full specification/review path. Keep blocked work in its true workflow state and use native GitHub dependency/blocking semantics rather than a permanent Blocked column.

### 5.3 Fields

Configure the approved small field set, with **Next review** replacing the earlier `Review gate` wording:

- Workstream
- Spec class
- Priority
- Next review
- Work cycle
- Target release

Prefer native GitHub metadata/relationships over duplicate custom fields.

### 5.4 WIP policy

Start with advisory/experimental WIP limits:

- Specifying = 2
- Implementing = 2
- Verification = 2

The WIP unit is the governing spec-level Issue, not subagents or linked implementation PRs.

### 5.5 Views and automation

Create the approved views:

- Current Flow
- Research Map
- Now / Next
- Verification Queue
- Development History
- PR / Integration View

Automate only low-risk mechanical state. Do not automate scientific-owner approval or other consequential review transitions.

### 5.6 Active-work reconciliation

Before declaring G2 complete, add/reconcile enough **current active work** that the board is immediately useful. This may identify active Issues/PRs, obvious parent/child relationships, blockers, duplicates and superseded items.

This is not the historical six-week backfill. Do not turn G2 into G4.

### 5.7 G2 convergence

Confirm that:

- one S1–S3 work item has one obvious primary main-board card;
- linked PRs do not double-count WIP;
- `Next review` represents the next required review rather than the full review lifecycle;
- the workflow remains useful for S0/S1 without unnecessary ceremony;
- Project state is navigational and does not become a competing authority.

## 6. G3 — #148/#149 pilot migration

After G2, use a separate focused branch/PR to create `specs/0148-catastrophe-continuation/{spec.md,plan.md,tasks.md}`.

Before drafting the canonical migrated spec, construct a concise chronological decision/supersession record from the original Issue, later owner comments, accepted gate records, PRs and durable scientific decision artifacts.

Canonical migrated intent means the **latest durably owner-approved, non-superseded normative intent**. Preserve older wording as history. Do not rewrite history or imply newly introduced requirement IDs governed historical work.

For #148, preserve at minimum historical G0/G1 evidence, later precision/tolerance clarification, the owner-approved P96 N=8 metric contract, revised outcome-neutral G3 semantics, and any later durable owner decisions present at migration time.

Any ambiguity that could change scientific meaning is a stop condition for owner clarification.

## 7. G4 — Six-week Project backfill

After the #148 pilot, backfill approximately 2026-07-30 through 2026-09-10. Classify recent merged PRs for history, reconcile remaining historical relationships/duplicates, and create only strategically useful migration specs.

Do not reverse-spec routine completed maintenance merely for completeness.

## 8. G5 — Evaluation

Evaluate after approximately 6–8 weeks **and** enough real SDD usage to make the assessment meaningful.

Where practical, the sample should include several completed S1/S2 deliverables, at least one completed or substantially progressed S2 scientific/contract deliverable, at least one modest newly started S1 deliverable, and enough routine S0 work to test whether the lightweight path stayed lightweight.

Assess traceability, Project usability, agent continuation burden, PR evidence mapping, stale/duplicate work, owner-review bottlenecks, specification overhead and S0 burden. If elapsed time passes without adequate samples, defer the evaluation rather than drawing conclusions from the calendar alone. Only then consider additional SDD automation.

## 9. Compatibility

No changes are planned to scientific behaviour, numerical behaviour, package APIs, persisted scientific schemas, dependencies, Julia support or `Project.toml` version.

## 10. Implementation stop conditions

Return to CYAX-0151 before continuing if implementation would require changing the authority hierarchy, scientific-owner approval semantics, the Project's non-authoritative role, adopting Spec Kit wholesale, or introducing scientific/package behaviour changes.

Classification ambiguity that might alter scientific/durable contract meaning fails closed to provisional S2 pending clarification.

Ordinary formatting, template wording and technical layout choices that preserve the approved contract and Clarification A do not require re-approval.
