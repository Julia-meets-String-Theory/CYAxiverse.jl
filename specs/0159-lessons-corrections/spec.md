---
spec_id: CYAX-0159
title: Lessons & Corrections
issue: 159
class: S1
status: draft
workstream: Infrastructure
parent: null
depends_on:
  - specs/0151-sdd-v1/clarification-a.md
  - specs/0155-private-safe-chat-checkpoints/spec.md
created: 2026-09-11
last_reviewed: 2026-09-11
review_required: repository-owner review and independent Sol/High adversarial review before merge
approval_ref: N/A while draft
---

# CYAX-0159 — Lessons & Corrections

## Objective

Issue #159 introduces a lightweight repository-wide memory of reusable failure
patterns and preventive checks. It helps future agents learn from owner
redirections, independent-review findings, failed material assumptions, and
semantic, scope, state, tooling, workflow, or privacy failures without creating
a diary, transcript archive, or second authority system.

## Current baseline and authority

`AGENTS.md` is the repository constitution. CYAX-0151 and its approved
amendments define the SDD contract, while CYAX-0155 defines the privacy boundary
for durable material derived from private or local context. CYAX-0151's
authority order remains unchanged:

```text
AGENTS.md + relevant CYAxiverse skill
    ↓
approved feature spec.md
    ↓
plan.md
    ↓
tasks.md
    ↓
GitHub Issue / Project
    ↓
implementation
```

Applicable validated lessons are cross-cutting advisory inputs consulted after
the governing sources are known and before planning or implementation; they are
not another authority tier. Candidate lessons are advisory proposals. No lesson
can override any artifact in the approved hierarchy, and the registry does not
own GitHub merge, Issue, Project, review, or scientific-acceptance state. This
draft does not assert an approval reference; Issue #159 is the governing intake
object.

## Scope

This bounded S1 change adds or updates only:

1. `.agents/LESSONS.md`, a compact registry with six sanitized candidate seed
   lessons and a candidate → validated → superseded knowledge lifecycle;
2. lightweight task-start, during-work, and convergence hooks in the canonical
   `.agents/skills/cyaxiverse-sdd/SKILL.md`;
3. a compact Lessons / corrections section in
   `.github/pull_request_template.md`;
4. this compact governing specification.

No `plan.md` or `tasks.md` is required for this bounded S1 slice.

## Requirements and acceptance mapping

| Requirement | Implementation | Acceptance evidence |
| --- | --- | --- |
| R-001 Authority and purpose | Registry preserves the CYAX-0151 precedence chain, treats validated lessons as advisory inputs, states candidate non-authority, and keeps the GitHub status and no-diary/no-archive/no-second-ledger boundaries. | Read `.agents/LESSONS.md`; verify lessons form no authority tier and override no artifact in the approved hierarchy. |
| R-002 Durable schema and lifecycle | Each entry uses ID, status, date, type, scope/tags, failure, correction, root cause, preventive check, applicability, evidence, supersession, and promotion fields; lifecycle is candidate → validated → superseded. | Inspect the schema and all six entries; verify candidates are not presented as validated policy. |
| R-003 Seed lessons | Add exactly L-0001 through L-0006, all `candidate`, dated 2026-09-11, using only durable public repository/GitHub references. | Count six registry entries and review their evidence references and sanitized wording. |
| R-004 SDD hooks | Applicable S1–S3 work reads applicable validated lessons at start; during work considers sanitized candidates for reusable corrections; close records one of four outcomes while keeping S0 lightweight. | Inspect the canonical skill diff and confirm `.codex/skills/cyaxiverse-sdd` remains its existing symlink mirror. |
| R-005 PR convergence prompt | PR template asks the four compact Lessons / corrections questions and permits S0 omission/`N/A`. | Inspect `.github/pull_request_template.md`. |
| R-006 Privacy and owner redirections | Redirections are abstracted, not copied; no private URLs, transcripts, resolver/local identifiers, or machine-local paths are stored; uncertain publication fails closed. | Diff review and repository-relative-reference scan of the registry/spec. |
| R-007 Proportionality and boundary | Process-only S1 documentation change; no second status/control plane, automatic promotion, scientific/package behavior, API, dependency, persisted schema, or version change. | Complete changed-file scope, diff review, `agent_verify.py diff-check`, and `git diff --check`. |

## Registry lifecycle and promotion

The registry stores advisory knowledge only. A candidate becomes validated only
after a review confirms generalizability, durable evidence, privacy safety, and
no conflict with the approved hierarchy. A superseded lesson remains for
historical traceability and names its replacement when applicable. Stable or
high-impact validated lessons may be separately promoted through normal review
into `AGENTS.md`, the SDD skill, or another normative source; promotion is never
automatic. When promotion is approved, the lesson becomes `superseded`, records
the repository-relative normative destination plus its reviewed revision or
approval reference, and is no longer applied independently. A narrower lesson
may replace it if only part of its guidance was promoted.

## Privacy boundary

Before durable publication, owner redirections and local observations are
sanitized into an abstract failure and prevention rule. The registry must not
contain private conversation/share URLs, raw transcripts, resolving private
references, absolute machine-local paths, local identity or session details,
private attachment locations, secrets, credentials, or tokens. Repository
content uses repository-relative paths and durable public Issue, PR, revision,
or hash references. If safety is uncertain, omit the datum and ask the owner.

## Seed set and durable provenance

The six initial candidates are deliberately limited to evidence already
available in durable public context:

- L-0001: candidate/repaired evidence versus scientific acceptance — Issue #148
  / PR #149 gate history.
- L-0002: explicit namespaces for nested process/scientific gates — sanitized
  correction pattern checked against Issue #151, PR #154, and Issue #148.
- L-0003: chronological brownfield supersession —
  `specs/0151-sdd-v1/clarification-a.md`, section 3, and Issue #151 / PR #154.
- L-0004: `tasks.md` is not live status — clarification A, section 5, and
  Issue #151.
- L-0005: advisory WIP state must not be falsified — Issue #151 G2 PASS comment
  `5625939343` and clarification A, section 6.
- L-0006: machine-local paths are not scientific provenance — `AGENTS.md`,
  CYAX-0155 R-001/R-005, and Issue #157 / PR #158.

No Fuzzy/Table-1-specific lesson is included.

## S0 proportionality

S0 work does not require reading the complete registry, creating a lesson, or
completing irrelevant SDD fields. The PR template may omit the Lessons /
corrections section or mark it `N/A`. The convergence hook asks the lesson
question only when relevant and does not create ceremony for ordinary S0 work.

## Non-scope

This specification does not change `AGENTS.md`; scientific, numerical,
package, API, dependency, persisted-schema, or version behavior; GitHub Issue,
Project, merge, or review state; the approved SDD status lifecycle; privacy
requirements; scientific acceptance criteria; or the authority of governing
specifications. It does not publish private conversation material, create a
second workflow/status ledger, automatically promote lessons, add a complex
taxonomy or automation, add plan/tasks artifacts, or add a Fuzzy/Table-1
lesson. Any normative rule promotion or additional historical lesson is
follow-up work subject to its own review.

## Compatibility and version impact

The change is documentation/control-plane-only. It changes no Julia or Python
package behavior, public API, dependency, persisted scientific schema, or
package version.

## Verification requirements

The implementation must report the complete changed-file scope, confirm that
only the four bounded files above changed, inspect the existing symlink mirror,
run `python3 scripts/agent_verify.py diff-check` when available, and run
`git diff --check`. A fresh independent Sol/High adversarial review is required
before merge and must assess authority precedence, lifecycle/promotion,
privacy, owner-redirection handling, S0/S1 proportionality, seed fidelity, and
spec ↔ skill ↔ PR-template ↔ registry convergence.

## Completion criterion

CYAX-0159 is complete for this slice when the four bounded artifacts converge
with this specification, all six candidates are present and sanitized, the
requested checks pass, and the independent review result and any unresolved
owner decision are recorded in the PR. Normative promotion and future lesson
validation remain separate reviewed work.
