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

- [ ] **T102 [R-004, R-005, CYAX-0172 G1] Verify typed-input zero-mode invariance**
  - Expected output: analytic-root and scaling/sign evidence.
  - Verify:
    - freeze `p64 = 3602879701896397/9007199254740992`;
    - compare the 256-bit refined root to
      `k64 = 0.5*log10(-2*cos(2*pi*p64))` at a tolerance justified by the
      final bisection interval;
    - compare the same refined root to the exact-decimal
      `k_phi = 0.5*log10((1+sqrt(5))/2)` only at absolute `1e-12`;
    - historical/corrected sign-change bracket remains identical.
  - Escalate if: the typed-input root, exact-decimal cross-check, or bracket
    fails its own predeclared tolerance.

## Phase 2 — Bounded replay

- [ ] **T201 [R-006, R-007, CYAX-0172 G2] Replay the exact golden-ratio witness**
  - Frozen inputs: source `vmm@995163f0058488ea183ac645045ed8b1636bef4a`;
    helper blob `cd861aa8a257ae36c9dadef6486449a0497c35a4`;
    `test/runtests.jl` blob `67a2c0d6d4ead46fb6acf58a0b6e3acfc41e22d0`;
    `Q=reshape([1.0,1.0],2,1)`; `L=[[2.0,-1.0],[1.0,1.0]]`;
    `theta=[0.0]`; `phase=[0.4,0.0]`;
    `k_grid=range(0.05,0.20; length=4)`; `precision_bits=256`;
    bisection/refinement stopping tolerance `1e-20`;
    exact-decimal cross-check acceptance `1e-12`;
    `p64=3602879701896397/9007199254740992`.
  - Expected output: identical coarse bracket `[0.10,0.15]` before/after,
    refined root agreeing with
    `k64=0.5*log10(-2*cos(2*pi*p64))` at refinement-justified tolerance,
    agreement with exact-decimal `k_phi=0.5*log10((1+sqrt(5))/2)` only at
    absolute `1e-12`, unchanged catastrophe type/side signatures, and only
    the expected factor-two Hessian/eigenvalue magnitude scaling.
  - Verify: no post-hoc case selection is permitted; this fixture is the sole
    CYAX-0172 G2 replay witness.
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
