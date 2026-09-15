# Tasks — CYAX-0170

## Rules

- Every task names the requirements/gates it advances.
- Each worker owns diagnose -> investigate/build evidence -> check -> correct ->
  recheck within its exclusive validation write set.
- Production algorithms, dependencies, schemas, and later phases are out of scope.
- `tasks.md` records evidence readiness, not live Issue/PR/Project state.
- Material scientific ambiguity returns to the owner; ordinary failed commands
  or historical defects are recorded and investigated.

## Phase 0 — Contract and topology

- [ ] **T001 [G0] Obtain durable owner approval of the S2 draft**
  - Input: `specs/0170-p0-numerical-equivalence/spec.md`, Issue #170.
  - Expected output: explicit durable approval reference and approved metadata.
  - Verify: approval references the reviewed normative revision/content.
  - Escalate if: requested changes alter the scientific reference or P0 boundary.

- [x] **T002 [R-015, G1] Declare safe P0 topology and worktree**
  - Output: pinned focused worktree and Issue #170 topology record.
  - Verify: unrelated dirty checkout untouched; worker write sets disjoint.

## Phase 1 — Parallel evidence construction

- [ ] **T101 [R-001, R-002, R-015, G1] Freeze environment and B1 load identity**
  - Owner: Worker A.
  - Output: `validation/p0_numerical_equivalence/environment_and_load.md` and
    optional validation-only helper scripts.
  - Verify: exact commands/outcomes and separate warm/cold measurements.
  - Escalate if: reconstructing the environment requires changing dependencies.

- [ ] **T102 [R-003, R-015, G1] Inventory compatibility and consumers**
  - Owner: Worker B.
  - Output: `validation/p0_numerical_equivalence/compatibility_inventory.md`.
  - Verify: named surfaces, repository-wide consumers, domains, failures, and
    persistence/conditional behavior are accounted for.
  - Escalate if: a surface cannot be classified without an owner API decision.

- [ ] **T103 [R-004-R-010, R-015, G1] Freeze semantics and F1-F13**
  - Owner: Worker C.
  - Output: semantics report, deterministic fixtures/results, and hashes.
  - Verify: focused execution at the pinned source; historical defects retained.
  - Escalate if: routes implement irreconcilable scientific contracts or fixture
    construction would require production changes.

- [ ] **T104 [R-010, R-012, R-015, G1] Capture B2-B7 baseline**
  - Owner: Worker D.
  - Output: benchmark report, bounded scripts/results, and identities.
  - Verify: exact commands, warmup/sample/stat/allocation method, results/gaps.
  - Escalate if: a benchmark would make a high-`h11`/population claim or alter
    production behavior.

## Phase 2 — Manager integration

- [ ] **T201 [R-011, R-013, R-014, G1] Integrate equivalence contract v1**
  - Output: `validation/p0_numerical_equivalence/numerical_equivalence_contract-v1.md`.
  - Verify: every comparison and tolerance is traceable and precommitted.
  - Escalate if: a tolerance cannot be fixed without seeing later output.

- [ ] **T202 [R-001-R-015, G1] Converge evidence index and exact candidate**
  - Output: evidence index/hash manifest, completed requirement mapping, clean
    artifact-only diff, and exact candidate commit.
  - Verify: focused checks, applicable package gates, diff-check, privacy scan,
    spec-plan-task-evidence reconciliation.
  - Escalate if: any production/dependency/schema change appears.

## Phase 3 — Independent verification

- [ ] **T301 [R-016, G2] Independently review the exact P0 candidate**
  - Owner: fresh Independent Numerical Reviewer who did not define fixtures or
    contract.
  - Output: durable review identifying candidate revision and PASS/revise verdict.
  - Verify: source parity, precommitted tolerances, F1-F13 failure modes,
    support/zero/order/threshold/compatibility coverage, and objective P2 gates.
  - Escalate if: any material claim changes; correct then review the new state.

- [ ] **T302 [R-016, G3] Return reviewed P0 checkpoint and stop**
  - Owner: Manager.
  - Output: required 13-part Manager return packet and recommendation.
  - Verify: reviewed revision distinguished from later mechanical synchronization;
    no P1/P2/P3 work started.

## Convergence

- [ ] **TC01 [R-001-R-016] Resolve findings from Manager convergence or G2 review**
  - Add bounded correction details only when a finding exists.

