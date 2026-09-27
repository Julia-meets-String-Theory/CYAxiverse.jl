# Tasks — CYAX-0191

This file decomposes the S3 programme into observable outputs. It does not
report live Issue, Project, PR, merge, review-owner, or scientific-acceptance
state. Later scientific tasks are planning only until the exact SDD candidate
is independently reviewed, the owner records final SDD approval, and a
separate implementation handoff authorizes them.

## Rules

- Every task names the requirements/gates it advances.
- Each task has one observable outcome and its verification.
- Keep routine diagnose, edit, check, correction, and recheck within the
  authorized task; stop at scientific-owner or scope boundaries.
- Tasks do not automatically become GitHub Issues.
- This checklist is not the live source for Issue, PR, Project, merge, or
  review state.
- Use `CYAX-0191 Gate A` through `CYAX-0191 Gate E` to distinguish scientific
  gates from the separate SDD review and approval boundary.

## Phase 1 — SDD preparation and approval

- [x] **T001 [R-001–R-020; CYAX-0191 Gate A entry contract] Draft the canonical S3 specification, plan, and tasks.**
  - Inputs: pinned Issue #191 contract, production-planning note, repository
    contract, applicable project skills, and review rubric.
  - Expected output: exactly `spec.md`, `plan.md`, and `tasks.md` under
    `specs/0191-kahler-moduli-stabilization/`, with no scientific code or
    evidence execution.
  - Verify: requirement/gate crosswalk covers the Issue #191 contract;
    scientific facts, conventions, evidence, inference, and exclusions are
    separately labeled; working-tree paths remain in the authorized scope.
  - Escalate if: the source contract is ambiguous, materially changed, or
    requires any additional public path.

- [x] **T002 [R-001–R-020] Check document structure, traceability, and whitespace.**
  - Expected output: all three documents are internally consistent and
    reviewable as one exact candidate.
  - Verify: applicable document checks, requirement/gate-to-plan/task mapping,
    and `git diff --check`; record the exact commands and outcomes in the
    manager return.
  - Escalate if: a material scientific discrepancy or untraceable normative
    requirement remains.

- [ ] **T003 [R-020] Obtain a fresh comprehensive Spec review of the frozen candidate.**
  - Expected output: independent Spec disposition of all required rubric
    dimensions, findings, and limits for exact commit/tree and all three file
    identities.
  - Verify: reviewer is fresh, independent, and uses the capability pin in
    the governing Spec review request; exact files and execution base match
    the candidate submitted.
  - Escalate if: the pinned reviewer capability is unavailable or the review
    finds a blocking fidelity, scope, evidence, or authority defect.

- [ ] **T004 [R-020] Obtain a fresh comprehensive Standards review of the same candidate.**
  - Expected output: independent Standards disposition of all required rubric
    dimensions, findings, and limits for the same exact commit/tree and file
    identities used by T003.
  - Verify: reviewer is fresh, independent, and uses the capability pin in
    the governing Standards review request; retain its verdict separately
    from the Spec verdict.
  - Escalate if: the pinned reviewer capability is unavailable or a concrete
    mandatory-standard, correctness, maintainability, privacy, or proportionality
    defect blocks acceptance.

- [ ] **T005 [R-020] Correct review findings and re-review changed normative bytes.**
  - Expected output: a corrected candidate with every blocking finding
    addressed and each nonblocking disposition explicit.
  - Verify: changed files remain the three authorized docs; material normative
    changes receive fresh affected reviews on identical final bytes; no prior
    verdict silently transfers.
  - Escalate if: a correction requires a scientific choice, scope expansion,
    or an unauthorized path.

- [ ] **T006 [R-020] Return the exact reviewed SDD candidate for owner approval.**
  - Expected output: exact branch commit/tree, file paths and byte identities,
    independent Spec and Standards returns, checks, limits, and the owner
    decision requested.
  - Verify: both reviews bind the same final normative bytes and pass without
    blocking findings; owner final approval is separately recorded in
    `approval_ref` after Control Desk reconciliation.
  - Escalate if: candidate or governing inputs drift, either review is blocked,
    or separate owner approval is absent.

## Phase 2 — Geometry boundary and model foundations

**Entry:** T006 has completed, the owner has approved the exact SDD revision,
and a separate implementation handoff authorizes scientific work.

- [ ] **T101 [R-002, R-008, R-010; CYAX-0191 Gate D] Build the versioned native geometry record.**
  - Expected output: geometry fixture with intersection tensor, Euler
    characteristic, ordered divisors, basis maps, dual curve basis, imported
    inequalities, cone construction/provenance, precision/exactness, and
    immutable source identity.
  - Verify: exact geometry identity and CYTools revision; tensor index and
    permutation checks; divisor/curve duality; basis-map checks; native
    fixture comparison; toric outputs retain the `torically inferred` status
    unless completeness is independently established.
  - Escalate if: source, basis, tensor, domain, precision, or provenance is
    missing or contradictory.

- [ ] **T102 [R-002–R-006 (including R-005), R-009] Implement the declared 2020 potential model and switches.**
  - Expected output: independently testable BBHL, linear non-perturbative,
    quadratic non-perturbative, and optional-uplift contributions; full source
    heavy-field reduction; explicit phase and model conventions.
  - Verify: exact convention record; distinct BBHL values; B1 switch behavior;
    phase term agrees with direct complex evaluation; parent metric and
    retained kinetic metric are reported separately.
  - Escalate if: implementing the model needs a changed normalization, basis,
    phase, physical domain, or heavy-field prescription.

- [ ] **T103 [R-007, R-009, R-011] Implement the generic-charge extension and result types.**
  - Expected output: `Q`-based model input, critical-point state, structured
    reports, and explicit exact/near-zero/lifted mode dispositions.
  - Verify: source-basis reduction and independent low-dimensional `Q` oracle;
    critical point retains `(t,tau,rho)`; no universal `stable` or `controlled`
    boolean; result fields distinguish source from physical mass data.
  - Escalate if: the generic extension changes or is presented as literal
    source-paper behavior.

- [ ] **T104 [R-009–R-012] Add replaceable numerical and replay interfaces.**
  - Expected output: generic potential/differentiation/search/fluctuation
    interfaces with replaceable AD and solver adapters and structured run
    manifests.
  - Verify: preserve supported precision and sparse/exact geometry where
    practical; type-stable numerical hot paths; no CYTools/Python call in
    potential, AD, Hessian, root finding, or optimization; manifest records
    backend versions, code/configuration identity, scales and budgets.
  - Escalate if: a required backend forces precision loss, optional-runtime
    coupling, or a schema choice absent from the approved spec.

## Phase 3 — Independent local validation

- [ ] **T201 [R-003, R-004, R-016; CYAX-0191 Gate B] Verify limits and the inverse-metric discriminator.**
  - Expected output: no-scale/correction-off limits and a one-modulus
    discriminator that distinguishes `C_full` from `C_frozen`.
  - Verify: BBHL and non-perturbative limiting cases; derived formula and
    normalization accompany the fixture; no frozen-block substitution passes
    silently.
  - Escalate if: analytic results disagree or the metric domain fails.

- [ ] **T202 [R-004–R-007, R-016; CYAX-0191 Gate B] Verify derivatives, phase, charge, and basis oracles.**
  - Expected output: independent derivative oracle under the declared
    heavy-field reduction, unequal-phase discriminator, generic-`Q` oracle,
    identity-charge reduction, and nontrivial basis covariance result.
  - Verify: covariance covers intersections, divisor/curve bases, Kähler
    coordinates, `tau`, `Q`, and derivatives; scalar potential remains
    invariant; optional Schachner comparison supplements only a fully matched
    limit with WSI/GV/uplift off.
  - Escalate if: any oracle requires altering the scientific convention.

- [ ] **T203 [R-010, R-012, R-016; CYAX-0191 Gate B] Freeze numerical scaling and acceptance rules.**
  - Expected output: benchmark-independent declared precision, field and
    potential scales, residual tolerance, solver criteria, and spectral
    thresholds.
  - Verify: manifest review occurs before Gate C execution; scaled residual
    uses a maximum of individually scaled components; cancellation-suppressed
    energy is not the sole potential scale.
  - Escalate if: policy is tuned after observing the named benchmark result.

## Phase 4 — Named source reproduction

**Entry:** all CYAX-0191 Gate B tasks pass.

- [ ] **T301 [R-010, R-013, R-017; CYAX-0191 Gate C] Reproduce P0-B1 standard KKLT.**
  - Expected output: source manifest, refined point, potential, residuals, and
    correction-off control evidence.
  - Verify: JHEP PDF Eqs. (4.1)–(4.2), Table 2 `M_{1,1}`, Table 3 first row
    non-uplifted columns, Appendix A.1; compare the named values at source
    precision; retain geometry-derived BBHL metadata while the active
    correction is off.
  - Escalate if: source-precision comparison fails or correction toggles
    modify stored geometry facts.

- [ ] **T302 [R-010, R-014, R-017; CYAX-0191 Gate C] Reproduce P0-B2 and retain its source anomaly.**
  - Expected output: separate source-table observations, equation-derived
    quantities, provenance-bearing independent refinement, and anomaly note.
  - Verify: JHEP PDF Eq. (4.5), Table 4 `M_{1,2}`; reproduce stationary
    location at source precision; derive `xihat` from its definition; reproduce
    equation-defined energy; preserve the approximately factor-100 table
    energy difference without parameter or normalization tuning.
  - Escalate if: the discrepancy's cause is not established or a proposed fix
    would alter the approved source convention.

- [ ] **T303 [R-010, R-011, R-015, R-017; CYAX-0191 Gate C] Reproduce P0-B3 publisher-PDF E5.**
  - Expected output: source manifest, refined critical point, domain evidence,
    source/equation observables, and exact `rho_3` flat-direction disposition.
  - Verify: publisher PDF Eqs. (4.13)–(4.14), Tables 6–7 E5; exact geometry
    polynomial/derivatives and inequalities; compare rounded observations at
    source precision and refine before deriving related quantities.
  - Escalate if: geometry, basis, E5 mapping, or exact-flat-direction
    semantics conflict.

- [ ] **T304 [R-017; CYAX-0191 Gate C] Independently review the named-reproduction evidence bundle.**
  - Expected output: candidate-specific source locator, manifest, residual,
    and comparison review for all three benchmarks.
  - Verify: independent identity and requirements review; a rounded count or
    matching values alone do not establish population identity or completeness.
  - Escalate if: any artifact cannot be tied to exact source, geometry, model,
    convention, code, precision, and search identity.

## Phase 5 — Geometry validation and result classification

- [ ] **T401 [R-008, R-018; CYAX-0191 Gate D] Complete the independent CYTools boundary validation.**
  - Expected output: exact geometry provenance plus tensor, permutation,
    divisor/curve, basis, cone, and native-fixture evidence.
  - Verify: full required Gate D evidence is present and matches the named
    geometry manifests.
  - Escalate if: imported and native fixtures disagree or cone completeness is
    claimed without evidence.

- [ ] **T501 [R-001, R-006, R-011, R-012, R-019; CYAX-0191 Gate E] Classify each accepted named result.**
  - Expected output: scaled stationarity, retained-sector fluctuation and
    generalized mass analysis, flat-mode classification, BF status where
    applicable, both metric/domain reports, and structured controls.
  - Verify: distinguish numerical, fluctuation, control, and global questions;
    heavy-sector stability and global compactification consistency remain
    `NOT_ASSESSED` unless separately established; no certified-string-vacuum
    wording.
  - Escalate if: evidence does not support the requested class or any report
    crosses the claim boundary.

## Convergence

- [ ] **TC01 [R-001–R-020; CYAX-0191 Gate A–E] Reconcile approved spec, plan, tasks, implementation, and evidence.**
  - Expected output: every requirement has an evidence disposition and each
    gate's PASS, FAIL, or unassessed state is supported by the exact evidence.
  - Verify: independent scientific and engineering review at the required
    boundaries; no task checkmark substitutes for results or approval.
  - Escalate if: implementation or evidence changes scientific intent; amend
    and re-review the spec before continuing the affected work.

The task list ends at evidence readiness for the authorized scientific
programme. Later Issue/PR/Project state, merge, closure, API promotion, and
follow-on research remain with their governing workflows and require separate
authority.
