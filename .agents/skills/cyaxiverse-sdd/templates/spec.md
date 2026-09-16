---
spec_id: CYAX-NNNN
title: <title>
issue: <GitHub issue number>
class: S1 | S2 | S3
status: draft | approved | accepted | superseded
workstream: <workstream>
parent: null
depends_on: []
created: YYYY-MM-DD
last_reviewed: YYYY-MM-DD
review_required: <role or N/A>
approval_ref: <durable approval reference or N/A while draft>
---

# <Title>

## Objective

State the capability, research question, reproduction, or correction in terms that can ultimately be accepted or rejected.

## Motivation

Why is this work useful now? For scientific work, separate research motivation from the claims the implementation may establish.

## Current baseline

Distinguish where relevant:

- Source facts
- Current implementation facts
- Existing empirical evidence
- Owner-approved conventions
- Open inference / hypothesis

For brownfield work, also record a concise chronological decision/supersession history when later durable approvals modify earlier Issue wording. Canonical migrated intent is the latest durably owner-approved, non-superseded normative intent.

Link source papers, prior specs, Issues, PRs, and durable validation artifacts.

## Scope

What this specification includes.

## Non-scope

What it deliberately does not attempt.

## Scientific claim boundary

Required for S2/S3 scientific work.

### This specification may establish

...

### This specification must not establish

...

## Fixed conventions and invariants

Record only feature-relevant conventions. Do not duplicate `AGENTS.md`.

## Requirements

Requirement IDs are immutable once referenced by implementation. For brownfield migration, do not imply newly introduced IDs governed historical work.

### R-001 — <name>

<One independently testable requirement.>

Where useful:

> WHEN <event>, CYAxiverse SHALL <observable response>.

### R-002 — <failure boundary>

> IF <invalid/unsupported condition>, CYAxiverse SHALL <fail-closed behaviour>.

## Acceptance gates

Use progressive gates when appropriate.

### G0 — <gate>

**Objective:** ...

**Acceptance:** ...

**Stop condition:** ...

## Verification requirements

Define required evidence, e.g. analytic/synthetic fixture, named source fixture, arbitrary precision, bounded replay, population execution, independent review, package/audit/CI.

## Interfaces and compatibility

State intended public API, persisted schema, source-data contract, numerical semantics, environment, compatibility and package-version impact. Technical realization belongs primarily in `plan.md`.

## Dependencies and blockers

Use native GitHub dependencies for actual work blocking where appropriate.

## Open owner decisions

List unresolved normative decisions. S2/S3 implementation must not guess them.

If classification ambiguity could affect scientific meaning or a durable scientific/contract boundary, treat the work as provisional S2 until clarified.

## Completion criterion

Give a falsifiable definition of complete. A documented negative scientific result may count when explicitly allowed here.
