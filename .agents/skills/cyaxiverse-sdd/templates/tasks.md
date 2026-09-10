# Tasks — CYAX-NNNN

## Rules

- Every task names the requirements/gates it advances.
- A task has one observable outcome.
- Verification is part of the task, not a later cleanup phase.
- Routine diagnose -> edit -> test -> correct -> retest stays with the implementation worker.
- Escalate only material conditions named by the task/spec.
- Tasks do not automatically become GitHub Issues.

## Phase 1 — Baseline / prerequisites

- [ ] **T001 [R-001, G0] <objective>**
  - Inputs: ...
  - Expected output: ...
  - Verify: ...
  - Escalate if: ...

## Phase 2 — Implementation

- [ ] **T002 [R-002, G1] <objective>**
  - Expected output: ...
  - Verify: ...
  - Escalate if: ...

## Phase 3 — Independent verification

- [ ] **T003 [G1] Independently verify <claim/deliverable>**
  - Must not rely solely on the implementation worker's reported result when independent review is required.

## Convergence

Add correction tasks when final spec/code/evidence review identifies an unmet requirement.

- [ ] **TC01 [R-...] <remaining work>**
