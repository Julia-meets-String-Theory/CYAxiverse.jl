# Tasks — CYAX-0172

## Rules

- This is S2 scientific/numerical corrective work.
- No production implementation begins before CYAX-0172 G0 approval.
- Each implementation task includes its own verification.
- Historical P0 evidence is read-only.
- `tasks.md` is not authoritative live workflow status.

## Phase 0 — Governing contract

- [ ] **T001 [CYAX-0172 G0] Freeze and independently review the S2 contract**
  - Inputs: Issue #172, CYAX-0170 P0, current helper/test source.
  - Expected output: exact reviewed spec/plan/tasks candidate.
  - Verify: independent Spec Review and Scientific/Numerical Review have no
    blocking finding.
  - Escalate if: normalization, phase/charge convention, physical claim
    boundary, or oracle remains ambiguous.

- [ ] **T002 [CYAX-0172 G0] Record durable owner approval**
  - Expected output: spec status `approved` with exact `approval_ref`.
  - Verify: approval binds the reviewed normative revision/content.
  - Escalate if: owner requests a scientific-contract change; changed normative
    bytes require fresh review.

## Phase 1 — Focused implementation

- [ ] **T101 [R-001, R-002, R-003, CYAX-0172 G1] Correct and verify Hessian**
  - Expected output: corrected helper plus focused analytic regression tests.
  - Verify:
    - identity fixture returns `4π² I₂`;
    - nontrivial direct analytic matrix agrees;
    - historical formula relation is factor two.
  - Escalate if: any fixed convention must change.

- [ ] **T102 [R-004, R-005, CYAX-0172 G1] Verify zero-mode invariance**
  - Expected output: analytic-root and scaling/sign evidence.
  - Verify:
    `k_c = 0.5*log10((1+sqrt(5))/2)` within specified tolerance and
    historical/corrected sign-change bracket agreement.
  - Escalate if: the root or bracket moves beyond tolerance.

## Phase 2 — Bounded replay

- [ ] **T201 [R-006, R-007, CYAX-0172 G2] Replay representative catastrophe**
  - Expected output: before/after comparison of bracket, refined `k_c`,
    catastrophe type, and side signatures.
  - Verify: only expected Hessian/eigenvalue magnitude scaling changes.
  - Escalate if: location, branch classification, or another observable changes
    for a reason not explained by the positive global scaling.

- [ ] **T202 [R-008, CYAX-0172 G1, CYAX-0172 G2] Freeze candidate evidence**
  - Expected output: exact commit/tree, changed paths, commands/results, replay
    artifact and evidence summary.
  - Verify: diff remains within approved scope; P0 bytes unchanged.

## Phase 3 — Independent verification

- [ ] **T301 [R-008, CYAX-0172 G3] Independent Scientific/Numerical Review**
  - Reviewer must be independent of the implementation worker.
  - Review binds the exact frozen candidate and all G1/G2 evidence.
  - Changed implementation/evidence bytes after review require fresh review.

- [ ] **T302 [CYAX-0172 G3] Return exact reviewed candidate to Control Desk**
  - Status may be PASS / REQUEST_CHANGES / BLOCKED.
  - This return does not authorize merge or Issue closure.

## Convergence

- [ ] **TC01 [R-001-R-008] Repair exact review findings if required**
  - Bounded implementation defects may be repaired and re-reviewed.
  - New scientific choices return to the specification/owner.
