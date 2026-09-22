# Tasks — CYAX-0176

## Rules

- No source implementation before CYAX-0176 G0 passes.
- PR #158 head is an exact dependency; material drift requires rebind.
- Core CYAxiverse must remain Python-free.
- No task may make MOSEK universally mandatory without owner/spec re-review.
- `tasks.md` is not live Issue/PR state.

## Phase 0 — Contract and base

- [ ] **T001 [CYAX-0176 G0] Review the S1 contract**
  - Expected output: exact spec/plan/tasks with independent Spec and Standards
    reviews having no blocking finding.
  - Verify: reviewed content binds PR #158 head and the Issue #176 repaired
    design decisions.
  - Escalate if: a reviewed r5 decision cannot be represented consistently.

- [ ] **T002 [R-013] Rebind privacy/interpreter base**
  - Input: PR #158 head
    `8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45`.
  - Verify exact interpreter helper, extension integration, and privacy
    guarantees before mutation.
  - Escalate on material PR #158 drift.

## Phase 1 — Initialization boundary

- [ ] **T101 [R-001-R-005, CYAX-0176 G1] Consolidate explicit CYTools enable**
  - Remove eager duplicate MOSEK side effects.
  - Preserve interpreter check and no-auto-build rule.
  - Add focused load/enable/idempotence/basic-call tests.
  - Escalate if core import would become Python-dependent.

- [ ] **T102 [R-010, CYAX-0176 G1] Enforce `hilbert_save` pre-enable policy**
  - Verify rejection before enable performs zero HDF5 writes and zero
    optimizer/license checks.
  - Verify enabled persistence does not require active MOSEK.

## Phase 2 — Solver/license state

- [ ] **T201 [R-004-R-007, R-011-R-012, CYAX-0176 G2] Implement solver-state contract**
  - Remove HOME path assumption.
  - Invoke activation API correctly.
  - Implement active/inactive/restart-required outcomes.
  - Verify supported fallback and explicit mandatory-MOSEK failure.
  - Escalate if upstream refresh semantics cannot support truthful in-process
    recovery.

## Phase 3 — Capability matrix

- [ ] **T301 [R-008-R-009, CYAX-0176 G3] Trace and verify optimizer consultation**
  - Cover fast/fair triangulation, stored-simplices reconstruction, standard
    geometry generation, Hilbert basis/save/generation.
  - Record consultation, selected backend, mandatory capability, fallback, and
    failure separately.
  - Escalate if observed upstream behavior conflicts with the governing spec.

## Phase 4 — Convergence

- [ ] **T401 [R-001-R-013, CYAX-0176 G1-G3] Freeze exact candidate evidence**
  - Record exact base/head/tree, changed paths, commands/results, capability
    matrix, privacy scan, and residual limitations.
  - Run focused tests and applicable broader checks.

- [ ] **T402 [CYAX-0176 G4] Independent Spec and Standards review**
  - Reviewers are independent of implementation workers.
  - Bind verdicts to exact candidate bytes/commit.
  - Changed affected bytes require fresh review.

- [ ] **T403 [CYAX-0176 G4] Return to Control Desk**
  - No merge or Issue closure authority is implied.

## Convergence

- [ ] **TC01 [R-001-R-013] Repair bounded review findings if required**
  - New normative/scientific choices return to owner/spec review.
