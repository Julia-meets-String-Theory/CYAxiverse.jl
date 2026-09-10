---
spec_id: CYAX-0151
title: Adopt CYAxiverse Spec-Driven Development v1
issue: 151
class: S1
status: approved
workstream: Infrastructure
parent: null
depends_on: []
created: 2026-09-10
last_reviewed: 2026-09-10
review_required: project-owner
version_impact: none
---

# Adopt CYAxiverse Spec-Driven Development v1

## 1. Objective

Adopt a lightweight, spec-anchored and risk-tiered Spec-Driven Development (SDD) workflow for `CYAxiverse.jl`.

The workflow SHALL make feature and scientific intent durable and traceable across specifications, implementation, evidence, GitHub Issues, pull requests and project status without creating a second competing source of project truth or turning process documentation into another codebase.

This is a process/control-plane change only.

It SHALL NOT alter scientific behaviour, numerical conventions, package APIs, persisted scientific schemas, runtime dependencies or package version.

## 2. Motivation

CYAxiverse already has a strong repository-wide development and scientific contract in `AGENTS.md`.

Recent scientific work has also evolved toward specification-like GitHub issues. In particular, work such as Issue #148 is organized around:

- a scientific objective;
- explicit scope and non-scope;
- claim boundaries;
- staged validation gates;
- acceptance criteria;
- stop conditions;
- reproducibility requirements;
- completion criteria.

The remaining problem is feature-level traceability.

Intent and current state can presently be distributed among:

- GitHub Issues;
- PR descriptions and discussions;
- validation artifacts;
- repository documentation;
- commit history;
- agent handoffs;
- historical conversations.

As the volume of agent-assisted development has increased, reconstructing how the package evolved and why particular decisions were made has become unnecessarily difficult.

SDD v1 SHALL provide a durable feature-level contract while preserving GitHub and Git as the normal work-coordination and historical record.

## 3. Current baseline

### 3.1 Existing project constitution

`AGENTS.md` is the canonical repository-wide agent contract.

It already governs, among other things:

- safe worktree/Git practice;
- Julia environment and verification;
- numerical precision and performance;
- persisted-data invariants;
- scientific claim boundaries;
- progressive validation;
- compatibility and version impact;
- issue/branch/PR workflow;
- delegated-agent responsibilities;
- independent verification where appropriate.

SDD SHALL build on this structure rather than duplicate it.

### 3.2 Existing durable workflow

Ordinary durable development state is already intended to live primarily in:

```text
issue -> branch -> commits -> PR -> tests/evidence -> merge
```

Long handoffs are exceptional rather than the normal project ledger.

SDD SHALL preserve this principle.

### 3.3 Existing proto-SDD practice

Some recent scientific issues already carry sufficiently rich contracts that they can serve as prototypes for SDD.

Issue #148 / PR #149 SHALL be the first migration pilot because the existing work already distinguishes:

- overall scientific objective;
- sequential gates;
- accepted versus future gates;
- explicit scientific non-claims;
- implementation evidence;
- independent review;
- partial PR completion versus overall issue completion.

The migration SHALL preserve that scientific contract rather than reinterpret it.

## 4. Scope

SDD v1 includes:

1. a risk-tiered SDD classification;
2. a feature-specification authority model;
3. a standard `spec.md` contract;
4. subordinate `plan.md` and `tasks.md` artifacts;
5. an on-demand `cyaxiverse-sdd` agent skill;
6. a minimal normative SDD amendment to `AGENTS.md`;
7. a lightweight SDD GitHub Issue intake form;
8. a traceability-oriented PR template;
9. one GitHub Project as the visual development control plane;
10. an initial Kanban workflow and WIP policy;
11. a small set of Project views and low-risk automations;
12. migration of #148/#149 as the first pilot;
13. a controlled backfill of approximately the previous six weeks of development;
14. an evaluation of SDD v1 after a real usage period.

## 5. Non-scope

SDD v1 SHALL NOT:

- make specifications an executable source-code generator;
- prohibit direct human editing of source code;
- install GitHub Spec Kit wholesale;
- replace `AGENTS.md`;
- duplicate `AGENTS.md` inside every feature specification;
- replace GitHub Issues;
- replace pull requests;
- create a GitHub Issue for every implementation task;
- require user-story syntax for mathematical or scientific requirements;
- force the full SDD process onto trivial maintenance work;
- reverse-specify the entire historical repository;
- treat natural-language requirements as a substitute for tests or numerical evidence;
- turn research work into sprint commitments;
- require a desired scientific hypothesis to succeed for work to count as complete;
- introduce Jira, Linear, Trello, Notion or another parallel project database in v1;
- treat an implementation agent's completion statement as independent verification;
- change the scientific meaning of #148/#149 during its SDD migration.

## 6. Authority model

The project SHALL use the following authority hierarchy:

1. **`AGENTS.md` + relevant CYAxiverse skill** — repository-wide constitution and domain rules.
2. **Approved feature `spec.md`** — canonical feature/scientific intent.
3. **`plan.md`** — current technical/numerical realization of the approved spec.
4. **`tasks.md`** — execution decomposition derived from the spec and plan.
5. **GitHub Issue / GitHub Project** — intake, coordination, discussion, dependencies and current status.
6. **Implementation** — actual current code behaviour.

### R-001 — Constitution precedence

An individual specification SHALL NOT override a normative project-wide requirement in `AGENTS.md`.

### R-002 — Spec precedence over accidental implementation behaviour

IF implementation behaviour conflicts with an approved specification, the conflict SHALL be resolved explicitly.

The implementation SHALL NOT silently become the new contract merely because it exists in code.

The resolution SHALL be either:

- change the implementation to satisfy the approved spec; or
- revise and re-review the spec because intended behaviour has deliberately changed.

### R-003 — No competing project ledger

GitHub Projects SHALL provide the live visual work map.

The repository SHALL NOT introduce a separately maintained active-work ledger that must be manually synchronized with Issues/PRs/Projects.

## 7. Risk-tiered SDD classification

SDD depth SHALL be based primarily on semantic, scientific and contract risk rather than code size.

### S0 — Trivial

Examples: typo; obvious documentation correction; mechanical low-risk maintenance; narrowly obvious CI repair.

Expected process: issue optional; specification not required; PR/tests as appropriate.

### S1 — Engineering

Examples: bounded new engineering behaviour; CLI capability; internal API change; performance improvement; development-process change.

Expected process: compact `spec.md`; plan/tasks when complexity warrants them; normal engineering review.

### S2 — Scientific / Contract

Examples: changed scientific interpretation; numerical algorithm with scientific consequences; basis or normalization convention; sampling/population contract; persisted scientific schema; scientific classifier; physical acceptance criterion; durable public scientific API.

Expected process:

```text
specification
    -> clarification/owner review
    -> approval
    -> plan
    -> tasks
    -> implementation
    -> convergence
    -> required verification/review
```

### S3 — Research Programme

Examples: catastrophic-inflation programme; orientifold programme; large-h11 ensemble programme.

An S3 programme SHALL normally be represented by a shallow roadmap of independently testable S1/S2 slices rather than one indefinitely expanding feature specification.

### R-004 — Risk beats line count

A small implementation change SHALL be classified S2 when it changes scientific or durable contract meaning.

A large mechanical change need not become S2 solely because of line count.

## 8. Specification lifecycle

Feature specifications SHALL use:

```text
Draft -> Approved -> Accepted -> Superseded
```

### Draft

The feature contract remains under formulation.

### Approved

The feature intent is sufficiently resolved for implementation to proceed.

For S2 work, unresolved normative scientific choices SHALL be explicitly approved by the scientific owner before this state is reached.

### Accepted

The implementation and required evidence satisfy the approved specification.

### Superseded

A later specification deliberately replaces the contract.

### R-005 — Intent changes are spec changes

WHEN implementation or investigation discovers that intended behaviour must change, the affected specification SHALL be updated and re-reviewed before implementation proceeds under the new intent.

Technical implementation details MAY evolve without re-approval when the approved intent and contract remain unchanged.

## 9. Standard SDD artifacts

For a normal S1/S2 specification the repository structure SHALL be:

```text
specs/
└── NNNN-feature-name/
    ├── spec.md
    ├── plan.md
    └── tasks.md
```

S3 programmes MAY additionally use:

```text
specs/
└── programme-name/
    └── roadmap.md
```

### 9.1 `spec.md`

The specification template SHALL support, as applicable:

- objective/scientific question;
- motivation;
- current baseline;
- source facts;
- implementation facts;
- empirical evidence;
- owner-approved conventions;
- explicit inference/hypothesis;
- scope;
- non-scope;
- scientific claim boundary;
- relevant fixed conventions/invariants;
- immutable requirement identifiers once referenced by implementation;
- testable natural-language requirements;
- EARS-style forms where useful;
- mathematical/domain-specific requirements where clearer;
- staged acceptance gates;
- stop conditions;
- verification/evidence requirements;
- API/schema/compatibility impact;
- version impact;
- dependencies and blockers;
- unresolved owner decisions;
- falsifiable completion criterion.

### 9.2 `plan.md`

The implementation plan SHALL be subordinate to the spec.

It SHALL include an explicit coverage mapping between:

```text
requirement/gate
    <-> planned implementation
    <-> planned verification
```

where applicable.

Technical architecture, algorithms, data structures, precision strategy, migration mechanics and other realization details SHOULD normally live here rather than in the normative specification.

### 9.3 `tasks.md`

Execution tasks SHALL:

- have one bounded observable outcome;
- identify the requirement/gate they advance;
- include their verification expectations;
- retain normal diagnose -> edit -> test -> correct -> retest responsibility with the implementation worker;
- identify material escalation conditions.

### R-006 — Tasks are not automatically Issues

A `tasks.md` item SHALL NOT automatically become a GitHub Issue.

A task SHOULD be promoted to a GitHub sub-issue only when it is independently durable, blocked, mergeable/reviewable, owner-decision-bearing, or otherwise valuable as a standalone work object.

## 10. Scientific requirements

### R-007 — Scientific claim boundary

S2 scientific specifications SHALL explicitly distinguish what the work may establish and must not establish.

### R-008 — Normative scientific ambiguity

IF implementation requires choosing or changing a physical normalization, basis convention, population/counting definition, scientific acceptance criterion, physical interpretation, scientific schema, or other normative scientific convention not already fixed by an approved contract, THEN the implementing agent SHALL return the decision to the appropriate owner rather than infer it silently.

### R-009 — Negative results

A scientifically well-evidenced negative result MAY satisfy a research specification when the specification's completion criterion permits it.

Failure of the desired physical hypothesis SHALL NOT automatically be classified as failure of the software/research task.

## 11. GitHub Issue policy

GitHub Issues SHALL remain the durable work/intake objects.

For substantial new SDD work, the repository SHALL provide a lightweight SDD intake form capturing at least:

- proposed SDD class;
- workstream;
- problem or research question;
- motivation / why now;
- known sources and prior work;
- known constraints and non-scope;
- known dependencies;
- whether the work may affect scientific interpretation, physical normalization or basis, sampling/population definition, persisted schema, or public API.

The Issue SHALL NOT duplicate a completed `spec.md`.

Once a specification exists, the Issue SHOULD act primarily as the coordination/index surface and identify the canonical spec.

## 12. Pull-request policy

The standard PR surface for SDD work SHALL support:

- governing Issue;
- canonical spec path;
- reviewed spec revision where useful;
- requirement/gate IDs implemented;
- concise implementation summary;
- explicit non-scope;
- scientific/API/schema/compatibility/version impact;
- requirement/gate -> evidence mapping;
- convergence status;
- remaining work.

### R-010 — Partial gate PRs are valid

A PR MAY implement only part of a larger specification.

Such a PR SHALL state clearly which requirements/gates it completes and SHALL NOT imply completion of later gates or the parent specification.

## 13. Final convergence

Before S2 work is declared complete, the responsible agent/reviewer SHALL reconcile:

```text
approved spec
    <-> plan
    <-> tasks
    <-> implementation
    <-> tests / scientific evidence
    <-> PR scope
```

### R-011 — Unmet requirements remain visible

IF convergence identifies a requirement that is not satisfied, that requirement SHALL remain explicitly incomplete or be converted into additional work.

It SHALL NOT be considered complete merely because the original task list has been checked off.

## 14. Agent-policy integration

SDD SHALL integrate into the existing agent control plane rather than create a parallel one.

### R-012 — Minimal `AGENTS.md` amendment

`AGENTS.md` SHALL receive only the normative rules necessary to establish SDD authority and stop conditions.

Detailed workflow guidance and templates SHALL live in an on-demand `cyaxiverse-sdd` skill.

### R-013 — On-demand SDD skill

The `cyaxiverse-sdd` skill SHALL cover:

- S0–S3 classification;
- drafting/revising specs;
- clarification;
- distinguishing scientific contracts from engineering requirements;
- deriving plans/tasks;
- requirement traceability;
- consistency analysis;
- final convergence;
- GitHub coordination without duplicate sources of truth.

SDD v1 SHALL NOT require installation of GitHub Spec Kit.

Selected Spec Kit concepts or automation MAY be evaluated in a later revision after the native workflow has been piloted.

## 15. GitHub Project / Kanban contract

One organization-level GitHub Project SHALL provide the visual development map.

Provisional name: **CYAxiverse Research & Development**.

### 15.1 Workflow states

The initial Status workflow SHALL be:

```text
Backlog
-> Specifying
-> Spec Review
-> Ready
-> Implementing
-> Verification
-> Done
```

Blocked work SHALL normally remain in its true workflow stage and use GitHub's native dependency/blocking relationship rather than moving to a permanent `Blocked` column.

### 15.2 Core custom fields

V1 SHALL begin with a deliberately small field set:

- **Workstream:** Inflation; Orientifolds; Sampling / Ensembles; Spectrum / BHSR; Axion-photon; Bayesian; Infrastructure; Release.
- **Spec class:** S0; S1; S2; S3.
- **Priority:** Now; Next; Later.
- **Review gate:** Engineering; Scientific owner; Independent scientific; Release.
- **Work cycle:** rolling two-week organizational interval.
- **Target release:** small set of relevant release/research destinations.

Native GitHub metadata SHOULD be used instead of duplicating it in custom fields where possible.

### 15.3 Initial WIP policy

Initial WIP limits SHALL be:

- Specifying: 2
- Implementing: 2
- Verification: 2

These values are experimental and SHALL be reviewed from observed use.

The WIP unit SHALL be the spec-level deliverable, not an individual subagent.

The purpose of WIP control is to protect integration/review capacity, especially scientific-owner attention.

### 15.4 Research scheduling principle

Work-cycle metadata SHALL be organizational rather than a scientific sprint promise.

Agents and reviewers SHALL NOT treat iteration boundaries as justification for weakening a scientific gate or forcing a positive result.

## 16. Required Project views

The initial Project SHALL provide:

- **Current Flow:** everyday Kanban board organized by Status.
- **Research Map:** cross-workstream table exposing current state, priority, class, relationships and review needs.
- **Now / Next:** focused strategic queue excluding low-priority backlog noise.
- **Verification Queue:** view optimized around work awaiting owner or independent review.
- **Development History:** completed work grouped by time period and workstream so package evolution is visible retrospectively.
- **PR / Integration View:** open integration state, including draft/ready PRs and their relationship to governing work.

## 17. Project automation boundary

V1 SHOULD automate only low-risk mechanical state.

Suitable automation includes:

- automatically add relevant CYAxiverse Issues to the Project;
- initialize new items in Backlog;
- move closed Issues to Done;
- reflect merged PR completion where appropriate.

V1 SHALL NOT initially automate scientific approval, `Specifying -> Spec Review`, `Spec Review -> Ready`, or consequential owner-review gates.

Further automation MAY be added after observing the actual workflow.

## 18. Pilot: #148 / #149

Issue #148 and PR #149 SHALL be the first real SDD migration.

The pilot SHALL create:

```text
specs/0148-catastrophe-continuation/
├── spec.md
├── plan.md
└── tasks.md
```

### R-014 — Preserve existing scientific contract

The migration SHALL preserve rather than reinterpret the objective, scientific motivation, claim boundary, G0–G4 structure, geometric requirements, stop conditions, completion criterion, and existing accepted evidence.

### R-015 — Preserve historical gate boundaries

The migration SHALL represent only what durable evidence supports.

In particular, historically accepted and future gates SHALL remain distinguishable rather than being retrospectively rewritten as though the SDD artifacts had existed from the beginning.

### R-016 — No pilot science changes

No catastrophe-continuation scientific behaviour, convention or acceptance criterion SHALL be altered merely to migrate #148/#149 into the SDD structure.

## 19. Historical backfill

One controlled historical backfill SHALL cover approximately **2026-07-30 through 2026-09-10**.

The purpose is project visibility, not retrospective bureaucracy.

### Pass A — Development history

Recent merged PRs SHOULD be classified sufficiently to make completed development visible by workstream, time/work cycle, and associated durable work item where clear.

### Pass B — Current-state reconciliation

Open development/scientific work SHOULD be reconciled sufficiently to identify active work, genuine blockers/dependencies, parent/sub-work relationships, duplicates, superseded work, and stale work that should not appear as active backlog.

### Pass C — Selective specification

Living specifications SHALL be backfilled only for active/high-value S2 work, ongoing S3 research programmes, foundational scientific contracts likely to evolve, and work where future continuation materially benefits from a canonical intent record.

Routine completed maintenance SHALL NOT require retrospective specification.

## 20. Rollout gates

### G0 — Approve the SDD v1 contract

**Objective:** Agree on the normative SDD model before changing repository process.

**Acceptance:** authority hierarchy, S0–S3 classification, specification lifecycle, Project/Kanban role, #148/#149 pilot boundary and non-goals are agreed, with no unresolved normative governance decision.

**Result:** PASS. Project-owner approval recorded on Issue #151 on 2026-09-10.

### G1 — Implement the repository SDD layer

**Objective:** Introduce the minimum repository artifacts needed to use the approved workflow.

**Acceptance:** repository contains approved versions of SDD templates, `cyaxiverse-sdd` skill, minimal `AGENTS.md` amendment, SDD Issue intake and SDD-aware PR template. Changes remain process-only.

**Stop condition:** If implementation requires broadening or contradicting this approved specification, return to G0/review before continuing.

### G2 — Establish the GitHub Project

**Objective:** Create the visual control plane without creating an independent project database.

**Acceptance:** Project contains the approved workflow states, core fields, WIP policy, saved views and low-risk automation.

### G3 — Migrate #148/#149

**Objective:** Test SDD against an active, real scientific workflow.

**Acceptance:** canonical #148 spec exists; plan/tasks reflect durable evidence and remaining work; PR evidence maps to relevant requirements/gates; manual convergence identifies no scientific-contract drift; ongoing #148 development remains possible.

**Stop condition:** Any migration ambiguity that could change the scientific meaning of #148 returns to the owner rather than being normalized silently.

### G4 — Six-week backfill

**Objective:** Make recent development and current work understandable from the Project.

**Acceptance:** recent merged work is visible by time/workstream; active work is visible; obvious duplicate/superseded/stale state is reconciled; only strategically useful historical specifications are backfilled.

### G5 — V1 evaluation

**Objective:** Determine whether SDD improves CYAxiverse development in practice rather than merely adding process.

Evaluate after approximately 6–8 weeks of actual use.

**Acceptance:** assess whether current work is easier to understand from one Project; scientific intent is easier to recover; substantial S2/S3 work begins from clearer approved contracts; PRs trace claims to requirements/evidence more effectively; agent continuation requires less repeated contextual reconstruction; duplicate/stale work is reduced; small work remains lightweight; specification maintenance overhead is acceptable; and no competing source of project truth has emerged.

A negative evaluation SHALL result in simplification or revision of SDD rather than automatic expansion of process tooling.

## 21. Compatibility and version impact

- Scientific behaviour: none.
- Numerical behaviour: none.
- Package API: none.
- Persisted scientific schemas: none.
- Runtime dependencies: none.
- Supported Julia environment: none.
- Package version: no version bump.

This specification governs development process/control-plane behaviour only.

## 22. Open owner decisions

There are no intentionally unresolved design questions in this approved specification.

Approval confirms these normative choices:

1. CYAxiverse uses spec-anchored, not spec-as-source, SDD.
2. The risk model is S0–S3 as defined here.
3. Issue #151 itself is S1 Engineering, with explicit project-owner approval because it changes repository governance.
4. `AGENTS.md` remains the highest project-level development authority.
5. Feature `spec.md` is canonical feature/scientific intent below `AGENTS.md`.
6. Plans and tasks are subordinate implementation artifacts.
7. GitHub Issues remain work objects; GitHub Projects is the visual control plane.
8. Tasks do not automatically become Issues.
9. Initial board states are `Backlog -> Specifying -> Spec Review -> Ready -> Implementing -> Verification -> Done`.
10. Initial WIP limits are 2 / 2 / 2 for Specifying / Implementing / Verification.
11. Work cycles are organizational rather than sprint commitments.
12. GitHub Spec Kit is not installed in v1.
13. #148/#149 is the first migration pilot.
14. Approximately six weeks of history are backfilled for visibility, not comprehensively reverse-specified.
15. V1 is evaluated after approximately 6–8 weeks before further process expansion.

## 23. Completion criterion

CYAX-0151 is complete when:

1. G0–G4 have passed;
2. the workflow has been used on real development long enough to reach G5;
3. G5 concludes either SDD v1 is accepted for continuing use or a documented revision/simplification is required;
4. all process changes remain within the scientific/API/version non-impact boundary of this specification.

A successful implementation of SDD tooling alone is not sufficient. The process must also demonstrate that it improves traceability at acceptable overhead.

## 24. Non-normative design references

The v1 design was informed by the external material identified during Issue #151 preparation, including:

- BCMS — *Spec-Driven Development: A Practical Guide*
- GitHub Spec Kit — agentic SDD reference
- GitHub Spec Kit — brownfield/existing-project guidance
- GitHub Spec Kit — spec-of-specs guidance
- GitHub Spec Kit — evolving-spec guidance
- Birgitta Böckeler / Martin Fowler — observations on SDD tooling and brownfield use
- Thoughtworks Technology Podcast — discussion of spec-driven development
- Alistair Mavin — EARS requirements syntax
- GitHub Projects documentation
- The Kanban Guide

These references inform the workflow design but are not themselves normative CYAxiverse project policy.

`AGENTS.md` and this approved specification remain the relevant in-repository authorities.
