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

Requirement IDs are immutable once referenced by implementation.

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

## Completion criterion

Give a falsifiable definition of complete. A documented negative scientific result may count when explicitly allowed here.
