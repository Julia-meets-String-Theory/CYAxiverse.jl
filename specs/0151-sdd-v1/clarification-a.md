# CYAX-0151 Clarification A — Peer-review corrections

**Status:** Approved
**Approved:** 2026-09-10
**Approval provenance:** GitHub Issue #151 comment `5624347708`
**Effect on G1:** None. G1 remains PASS and PR #152 remains accepted.

This is a normative amendment to CYAX-0151. Where this clarification conflicts with earlier wording in `spec.md`, this clarification supersedes that wording. Earlier wording remains historical context and is not silently rewritten.

## 1. Durable approval provenance

Approved S2/S3 specifications SHALL identify durable approval provenance. At minimum, the specification metadata SHALL support:

```yaml
review_required: <owner/reviewer role>
approval_ref: <durable GitHub Issue comment or PR review reference>
```

The approval record SHALL identify the specification and the reviewed revision or normative content sufficiently to distinguish it from earlier and later drafts.

Preferred workflow:

```text
draft spec
→ commit draft
→ owner reviews identified revision
→ durable approval record
→ mark spec approved + record approval_ref
```

This avoids requiring an approved spec commit to refer recursively to its own SHA.

If classification ambiguity could affect scientific meaning or a durable scientific/contract boundary, classify the work as provisional **S2** pending clarification. Ordinary engineering uncertainty alone does not force S2. Once clarification establishes that no such contract risk exists, the work may be reclassified S1.

## 2. GitHub Project item semantics

For S1–S3 work, the **governing GitHub Issue is the primary Project work item**. Linked PRs normally provide implementation, integration and evidence for that Issue and SHALL NOT normally become duplicate WIP cards on the main workflow board.

For S0 work, an Issue remains optional; a PR-only change MAY itself be the Project item where tracking it is useful.

A dedicated PR/integration view MAY expose linked PRs without treating them as duplicate spec-level WIP.

Replace the proposed `Review gate` Project field with **`Next review`**, with values such as:

- Engineering
- Scientific owner
- Independent scientific
- Release
- blank / none

The field describes the next required review rather than every review the item may eventually require.

## 3. Brownfield supersession semantics

For brownfield migration, the original Issue body SHALL NOT automatically be treated as the current canonical contract.

Before migration, reconstruct a concise chronological decision/supersession record from durable sources such as:

- original Issue wording;
- owner decisions/comments;
- accepted gate records;
- PRs;
- durable validation or decision artifacts.

The canonical migrated intent is the **latest durably owner-approved, non-superseded normative intent**.

Earlier wording remains historical evidence. It SHALL NOT be silently rewritten or presented as though later interpretations had existed from the beginning. If durable records appear inconsistent and precedence cannot be established safely, migration SHALL stop for owner clarification.

For #148 specifically, migration SHALL preserve at minimum:

- historical G0/G1 evidence;
- subsequent precision/tolerance clarification;
- the owner-approved P96 N=8 metric contract;
- revised outcome-neutral G3 semantics;
- any further durable decisions made before migration begins.

New requirement IDs MAY describe the migrated current contract but SHALL NOT be represented as identifiers under which historical work was originally executed.

## 4. Lightweight S0/S1 path

### S0

S0 requires no specification. An Issue is optional. Ordinary focused PR/tests are used as appropriate. Irrelevant SDD sections in the PR template may be omitted or marked `N/A`.

### S1

When S1 work changes bounded intended engineering behaviour, a compact `spec.md` is required. `plan.md` and `tasks.md` are optional and are created only when complexity warrants them.

An S1 spec directory MAY therefore contain only `spec.md`.

### S2

S2 normally requires `spec.md`, `plan.md`, and `tasks.md`, unless an explicitly justified simplification preserves equivalent traceability.

### S3

S3 uses a programme roadmap plus independently specified S1/S2 slices.

The objective is proportional ceremony, not uniform artifact production.

## 5. `tasks.md` semantics

`tasks.md` is an **execution decomposition**, not the authoritative live workflow ledger.

Tasks SHOULD normally end at an observable deliverable/evidence-readiness boundary, for example implementation complete, required focused verification complete, evidence produced, convergence complete, or PR ready for required review.

The following state belongs primarily to GitHub Issue/PR/Project:

- PR draft/ready state;
- merge state;
- Issue open/closed state;
- current Kanban state;
- current review ownership.

A task list SHALL NOT need a self-referential “merge this PR and then update this same task file as merged” step in order to be considered complete.

The stale CYAX-0151 T113 is retained as historical evidence motivating this clarification; it is not evidence that G1 failed.

## 6. G2 clarification

G2 remains the next gate. Retain:

```text
Backlog
→ Specifying
→ Spec Review
→ Ready
→ Implementing
→ Verification
→ Done
```

Not every item must traverse every state. S0 may use only relevant states; simple S1 work may skip unnecessary intermediate states; S2 normally follows the full specification/review path.

Do not add a permanent `Blocked` column.

The initial `2 / 2 / 2` WIP limits for Specifying / Implementing / Verification are experimental/advisory rather than hard scientific-development constraints.

Before considering the board established, perform a **small active-work reconciliation** so the Project begins as a useful current-state map. This MAY identify active Issues/PRs, obvious parent/child work, blockers, duplicates and superseded items. It SHALL NOT become the six-week historical backfill.

The gate sequence remains:

```text
G2 — establish Project + reconcile current active work
↓
G3 — migrate #148/#149 as first scientific SDD pilot
↓
G4 — reconstruct historical six-week development view
↓
G5 — evaluate real SDD use
```

## 7. G5 sample requirement

The approximately 6–8 week observation period remains useful, but elapsed time alone is insufficient.

G5 SHOULD occur only after enough actual SDD usage exists to evaluate the workflow meaningfully. Where practical, the evidence set should include:

- several completed S1/S2 deliverables;
- at least one completed or substantially progressed S2 scientific/contract deliverable;
- at least one modest S1 deliverable newly started under SDD rather than retrospectively migrated;
- routine S0 work sufficient to assess whether the lightweight path remained lightweight.

If the calendar window expires without adequate samples, defer evaluation rather than drawing conclusions from elapsed time alone.

## 8. Unchanged decisions

This clarification does not alter these decisions:

- G1 remains PASS.
- PR #152 remains accepted and is not reopened.
- No retrospective rerun of the unavailable G1 local checks is required.
- `.agents/skills/` remains the canonical skill source.
- `AGENTS.md` remains the project constitution.
- GitHub Spec Kit remains excluded from v1.
- #148/#149 remains the first SDD scientific pilot.
- G2 precedes G3.
- CYAX-0151 authorizes no scientific/package behaviour change.
