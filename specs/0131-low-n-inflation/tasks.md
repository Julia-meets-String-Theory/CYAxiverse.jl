# Tasks — CYAX-0131

## Rules

- This is S2 scientific-contract work.
- The current authorization covers document drafting and review preparation
  only.
- No scientific search, catastrophe refinement, trajectory integration,
  candidate refinement, observational optimization, or production data
  generation may begin until exact final S2 owner approval and a new,
  separately reviewed implementation handoff authorize it.
- Every implementation-stage task below is conditional on that later handoff.
- `tasks.md` is subordinate execution/evidence planning, not live Issue, PR,
  Project, merge, or review status.
- Do not invent a scientific choice. Return unresolved normalization, basis,
  phase mapping, acceptance, population, physical-interpretation, or schema
  choices to the scientific owner.

## Phase 0 — Prepare and review the governing contract

- [ ] **T001 [R-001–R-011, CYAX-0131 S2 contract] Create and reconcile the three normative documents**
  - Inputs: owner-approved scientific rebind and S2 contract approval;
    current source snapshot and handoff-bound source identities.
  - Expected output: new `spec.md`, `plan.md`, and `tasks.md`, with a stable
    requirement-to-plan-to-task map.
  - Verify: all eleven requirements and all specified model, phase, k,
    diagnostic, observational, claim, and correction boundaries agree across
    the three files; public diff contains only those three new paths.
  - Escalate if: faithful drafting needs a new scientific convention, a path
    outside the three files, or a different source/model choice.

- [ ] **T002 [R-011, CYAX-0131 S2 contract] Validate and freeze the exact document candidate**
  - Expected output: exact candidate identity for all three files and the
    combined commit/tree reviewed.
  - Verify: `git diff --check` passes; documents are internally consistent;
    no absolute local path or unsupported execution authorization appears.
  - Escalate if: validation exposes a normative ambiguity or a public path
    outside scope.

- [ ] **T003 [R-011, CYAX-0131 S2 contract] Obtain independent SPEC and SCIENTIFIC review**
  - Inputs: one frozen exact candidate and the applicable canonical review
    rubrics.
  - Expected output: both independent reviews bind the same exact candidate
    and have no blocking findings.
  - Verify: reviewer role, exact candidate identity, verdict, findings, and
    limitations are recorded; historical reviews do not substitute.
  - Escalate if: a finding requires changing an owner-approved scientific
    choice or reviewer capability is unavailable. Any normative-byte repair
    requires both axes to review the same revised candidate again.

- [ ] **T004 [R-011, CYAX-0131 S2 contract] Reconcile and obtain exact-content owner approval**
  - Inputs: final specialist-reviewed candidate and reconciliation evidence.
  - Expected output: separate owner approval bound to the exact normative
    content.
  - Verify: approval explicitly authorizes the final S2 contract; it does not
    implicitly authorize numerical execution.
  - Escalate if: owner approval is absent, stale, or bound to different bytes.

## Phase 1 — Rebind the later implementation handoff

- [ ] **T101 [R-011, CYAX-0131 implementation handoff] Prepare the current-schema implementation packet**
  - Precondition: T004 is accepted and exact owner approval is recorded.
  - Expected output: a new handoff bound to current source, issue/dependency
    state, schemas, authorized execution surface, model identities, and scope.
  - Verify: handoff explicitly carries implementation limits, allowed
    commands, required identity/evidence fields, and numerical acceptance
    criteria approved for execution. It pins which source field or interval
    defines reported `N_e` and its window, plus the sample/index identity used.
    The source distinguishes total `efolds`, `slow_roll_efolds`, and sample
    `n`; the task selects none of them.
  - Escalate if: current source contradicts the approved contract, a required
    scientific threshold or model convention is missing, or owner rebind is
    needed.

- [ ] **T102 [R-011, CYAX-0131 implementation handoff] Obtain separate handoff review and dispatch**
  - Precondition: T101 has a complete exact candidate.
  - Expected output: fresh review and explicit dispatch for the implementation
    stage.
  - Verify: authorization names the precise allowed execution and outputs.
  - Escalate if: review is blocked or the authorization scope is broader than
    the owner-approved specification.

## Phase 2 — Conditional calibration and bounded discovery

All Phase 2 tasks remain unstarted and unauthorized until T102 explicitly
dispatches them.

- [ ] **T201 [R-001, CYAX-0131 calibration] Establish the zero-phase 10-row N8 author-model identity**
  - Expected output: source-consistent zero-phase author-model calibration
    record using the concrete `author_inflation.n8_author_trajectory(...)`
    route where trajectory samples are required.
  - Verify: model, 10-row source, revision, zero phase vector, coordinate and
    metric/basis conventions, witness identity, and numerical status are
    recorded.
  - Escalate if: route identity or source mapping differs from the approved
    author model.

- [ ] **T202 [R-002, CYAX-0131 calibration] Establish the row-2 N8 phase benchmark and refined catastrophe**
  - Expected output: same-model evidence for `phase[2] = 0.04` radians, all
    other phases zero, source-consistent critical-point identities on both
    relevant sides, and a refined catastrophe location.
  - Verify: exact row/phase convention, the two point identities, refined
    location, solver status, and handoff-approved tolerances are recorded.
  - Escalate if: either critical point cannot be identified source
    consistently, the catastrophe does not refine, or a new phase/model
    convention is required.

- [ ] **T203 [R-003, CYAX-0131 cross-check] Record the separate 12-row P96/Table-1 continuation**
  - Expected output: clearly labeled P96/Table-1 reconstruction cross-check.
  - Verify: 12-row model identity and source route are separate from the
    10-row author model; no certification transfer or merged population is
    asserted.
  - Escalate if: reporting the cross-check would require redefining author
    model authority.

- [ ] **T204 [R-004, CYAX-0131 N5 replay] Re-solve the reduced two-cosine phase-shifted fold**
  - Expected output: reduced light-direction result with `delta = π/4` on the
    second cosine and a re-solved/refined shifted fold.
  - Verify: input model, phase assignment, source route, refined fold identity,
    and solver status are recorded.
  - Escalate if: the replay needs an unsupported full eight-row phase map; in
    that case mark that mapping `NOT_VERIFIABLE` and do not construct it.

- [ ] **T205 [R-005, CYAX-0131 physical-k gate] Independently re-establish physical `k`**
  - Expected output: physical-k evidence tied to the reviewed physical
    convention and distinguishable from all homotopy-k values.
  - Verify: evidence satisfies the exact handoff criteria before any physical
    trajectory diagnostics are reported.
  - Escalate if: physical meaning, normalization, or evidence criteria are
    ambiguous. Homotopy evidence remains calibration/discovery only.

- [ ] **T206 [R-006, R-007, CYAX-0131 diagnostics] Report the permitted sample diagnostics after the physical-k gate**
  - Precondition: T205 passes.
  - Expected output: only `N_e`, `n_s`, `paper_delta_H` scalar amplitude, and
    cumulative turning from the concrete N8 author trajectory/sample route,
    labeled as the 10-row author trajectory with the raw-radian coordinate
    convention and the rounded/reconstructed author metric identified where
    applicable; transverse-Hessian eigenvalues, if reported, are attached to a
    named catastrophe point.
  - Verify: the implementation handoff pins the source field/interval and
    window for `N_e` plus the sample/index identity; this task does not choose
    among total `efolds`, `slow_roll_efolds`, or sample `n`. Each quantity has
    sample/source identity, units, scale category, and the required coordinate
    and metric labels; point identity and metric/basis are recorded where
    relevant. State that transverse eigenvalues are not pivot-scale or
    along-trajectory spectra.
  - Escalate if: a new diagnostic, acceptance window, or physical
    interpretation is needed.

- [ ] **T207 [R-008, R-009, R-010, CYAX-0131 convergence] Verify scope and correction boundaries**
  - Expected output: final evidence and summary preserve the five
    `NOT_REACHED` observational items, fixed-saxion claim boundary, integrated
    Eq. 19 repair, and Issue #172 `4π²` ancestry.
  - Verify: no population/full-KS, dynamical stabilization, fully stabilized
    string-cosmology, or observational acceptance claim has entered the result.
  - Escalate if: current code/source drift alters one of these boundaries.

## Deferred observational work

The following remain `NOT_REACHED` and need a later, separately governed
owner-approved stage before any execution: a new `A_s` conversion/acceptance
window; a new `n_s` acceptance window; tensor-to-scalar ratio `r`; a full
physical N=5 trajectory; and the CYAX-0131 observational stretch goal. Their
presence in this task map is a deferral record, not execution authority.
