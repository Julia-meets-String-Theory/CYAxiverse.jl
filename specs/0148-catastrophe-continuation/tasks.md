# Tasks — CYAX-0148

## Rules

- Every task advances the migrated CYAX-0148 specification or an acceptance
  gate.
- New requirement IDs are migration-era identifiers. Checked historical work
  is mapped to them for evidence traceability; the IDs did not govern the
  historical execution.
- A checked task means its bounded execution/evidence outcome is durably
  supported. It does **not** by itself mean the parent scientific gate, PR, or
  Issue is accepted/merged/closed.
- Verification is part of the task, not a later cleanup phase.
- Routine diagnose -> edit -> test -> correct -> retest remains with the
  implementation worker.
- Tasks do not automatically become GitHub Issues.
- `tasks.md` is execution/evidence-readiness decomposition, not authoritative
  PR merge state, Issue closure state, Project/Kanban state, or current review
  ownership.
- Rejected evidence remains explicitly rejected even when the task that
  produced/reviewed it is historically complete.
- G2 PASS has been recorded; G3 proceeded and is now also PASS. The earlier
  G3-blocked-on-G2 constraint is satisfied.
- Exact higher-precision bit depths in current tasks are implementation/evidence
  choices; the governing spec requires a genuine target-constructed
  higher-precision stability path without canonically fixing those bit depths.

## Migration-cutoff evidence state

Brownfield reconstruction cutoff:

- `vmm`: `6434e133af4b91db81babfafa024fc3cbada901a`
- PR #149 frozen head: `e991495f1bb58edd7a7043dfc90771b6666c717c`
- G0: PASS
- G1: PASS
- P96 N8 metric decision: approved
- G2: not accepted at the original cutoff; first candidate `9fe32eb...`
  rejected `091cc9c...`; repair candidate `5b8daff...` awaiting fresh
  independent scientific review at that cutoff
- G3: not started at the original cutoff
- optional G4: not started

Post-cutoff accepted evidence (from PR #149 branch):

- G2: second candidate `5b8daff...` rejected `3c33205...`; further repair
  `cc8ac73...` **PASS** after independent review, durable acceptance
  `f9b04ed...`
- G3: **PASS** — implementation `4abd9c3...`, evidence `c6d2291...`,
  independent review, durable acceptance `f33ba76...`
- optional G4: not started; applicable because G3 yielded a suitable off-ray
  catastrophe locus

Current scientific gate state:
**G0 PASS · G1 PASS · G2 PASS · G3 PASS**; optional G4 not started.

Live state after this cutoff belongs in GitHub Issue #148 / PR #149 / Project.

## G0 — Baseline and scientific-contract audit

- [x] **T001 [R-002, R-003, R-012, R-013, G0] Audit the pre-continuation baseline**
  - Outcome: identify source/revision, radial and homotopy scale meanings,
    coordinate/phase/metric conventions, existing matcher/continuation and
    augmented-solver behavior, and the N=5 discrepancy.
  - Evidence: `docs/src/issue_148_g0_baseline_audit.md`,
    `scripts/audit_issue_148_g0.jl`, accepted audit commit `62c5a61...`, Issue
    #148 comment `5608088958`.
  - Verified result: G0 PASS; the N=5 discrepancy is a source-fixed
    implementation/validation defect, not an unresolved scientific convention.
  - Scientific boundary: no convention was changed by this task.

## G1 — Analytic N=5 continuation validation

- [x] **T101 [R-003, R-004, G1] Repair the source N=5 benchmark defect**
  - Outcome: remove the incorrect reuse of the N=8 catastrophe scale and use
    the source N=5 critical scale `(4/pi) log(1024/255)`.
  - Evidence: tested G1 code `792a02fb77d22cefdfec8fe77969d9c719e7b6d0` and
    G1 repair evidence.
  - Verify: analytic source oracle and focused regression must distinguish the
    repaired value from the historical defective fixture.

- [x] **T102 [R-004, G1] Implement genuine N=5 branch continuation**
  - Outcome: follow the known regular zero-phase source-reduced branch toward
    the analytic catastrophe with intrinsic branch identity and numerical event
    localization rather than independent grid solves plus post-hoc matching.
  - Evidence: `src/paper_benchmarks/poly102_inflation.jl`, focused G1 replay
    and regression scripts at the accepted code revision.
  - Verify: accepted default event error `4.768e-11`, gradient
    `4.586e-27`, Hessian magnitude `3.745e-11`, with literal event/failure
    tolerances and branch probes.

- [x] **T103 [R-004, R-012, G1] Demonstrate target-constructed high-precision N=5 stability**
  - Outcome: source quantities constructed at target precision; continuation
    preserves numeric type without silent Float64 narrowing.
  - Evidence: genuine 128/256-bit reruns in the accepted G1 evidence.
  - Verify: event-scale error improved from approximately `2.91e-31` to
    `3.39e-61`, with stable critical point and independent residual/Hessian
    agreement.

- [x] **T104 [R-002, R-004, R-012, R-014, G1] Complete fresh independent G1 verification**
  - Outcome: independent scientific reviewer checks the corrected code,
    analytic oracle, precision path, branch identity and failure boundaries.
  - Evidence: `docs/src/issue_148_g1_final_independent_acceptance_792a02f.md`
    and manager decision `1eef936e1908959db50a969e717b16a9c9e414bf`.
  - Verified result: 63/63 focused regression assertions passed; G1 accepted
    PASS. Broader pre-existing CI/audit limitations remain separate and were
    not silently converted into G1 claims.

## G2 prerequisite — N=8 metric authority

- [x] **T201 [R-003, R-005, R-013, G2] Audit the N=8 canonical metric boundary**
  - Outcome: expose the period-one/raw-radian coordinate mismatch and the
    Eq.96/CYTools versus literal-displayed-equation factor-two matrix conflict.
  - Evidence: `docs/src/issue_148_n8_metric_boundary_audit.md` and
    `scripts/audit_issue_148_n8_metric_boundary.jl` at `dd12209...`.
  - Escalation outcome: correctly stopped for owner choice before scientific
    canonical outputs were implemented.

- [x] **T202 [R-005, R-013, G2] Obtain durable scientific-owner P96 approval**
  - Outcome: scientific N8 outputs use period-one
    `K_theta=M96/k^2`; equivalent raw-radian metric is
    `M96/[k^2(2pi)^2]`; `M96` is the precise reconstructed Eq.96/CYTools
    reference metric in the relevant GLSM basis; its reconstruction/source
    identity must be verified at working precision; any representation in
    another basis must use the corresponding explicit metric congruence
    transformation; Eq.96/CYTools matrix authority is selected; unconverted
    author raw-radian `M96/k^2` remains A96 reproduction-only.
  - Evidence: Issue #148 comment `5623902670` and
    `docs/src/issue_148_n8_approved_metric_contract.md` at `d1a0b70...`.
  - Verify: no worker is left to infer coordinate, matrix, reconstruction,
    source-identity, or basis-transformation authority.

## G2 — N=8 radial multifield validation

### Historical rejected candidate

- [x] **T210 [R-006, R-007, R-014, G2] Independently evaluate the first G2 candidate**
  - Input candidate: `9fe32eb0bebc4d9ad99fb761357c604674e3681a`.
  - Outcome: fresh independent review established that the advertised passing
    checks did not prove the required continuation/validation mechanism.
  - Evidence: `docs/src/issue_148_g2_independent_review_9fe32eb.md`, manager
    decision `091cc9c...`, Issue #148 comment `5624861152`.
  - Verified result: **candidate rejected / G2 FAIL**. Disabling the bordered
    corrector reproduced the accepted path, matcher evidence copied
    continuation identities, the precision stages held the event scale fixed,
    canonical comparisons mixed term sets, and status/tolerance/failure
    coverage was inadequate.
  - Preserve: useful exact source12 radial degeneracy evidence, 256-bit
    augmented convergence, preserved G1, and the honest `:unresolved`
    higher-derivative classification.

### First repair candidate and second rejection

- [x] **T211 [R-003, R-005, R-006, R-007, R-012, G2] Produce a bounded G2 repair candidate and revision-specific replay evidence**
  - Implementation candidate: `5b8daffd732ba307ed1980615b9f049a95b92c2c`.
  - Evidence state at original migration cutoff continued through
    `e991495f1bb58edd7a7043dfc90771b6666c717c`.
  - Expected correction scope from the first FAIL review: bordered-coordinate
    consistency and provenance, actual old-matcher exercise, genuine
    target-constructed event precision ladder, source-precision disclosure,
    like-for-like P96/A96 checks, and stronger status/failure/tolerance
    evidence.
  - Observed candidate evidence: replay exit 0; 321/321 named assertions plus
    additional top-level checks; eight accepted bordered-corrector steps on the
    selected trace with explicit fallback provenance; disabled-corrector trace
    differs; forced failure records `:step_failed`; actual
    `pilot_match_records!` comparison reports five intrinsic comparisons with
    zero observed disagreements; 128/256-bit target-constructed augmented event
    scales agree below `1e-30`; source12/author10 and P96/A96 paths are labelled
    separately; focused G1 replay remains 63/63.
  - Boundary: this checkbox means a reviewable repair candidate/evidence package
    existed. This candidate was subsequently rejected (see T211a) and a further
    repair at `cc8ac73...` was the revision ultimately accepted (see T212).

- [x] **T211a [R-006, R-007, R-014, G2] Independently evaluate the `5b8daff` repair candidate**
  - Reviewed revision: `5b8daffd732ba307ed1980615b9f049a95b92c2c`, evidence
    through `e991495f1bb58edd7a7043dfc90771b6666c717c`.
  - Evidence: `docs/src/issue_148_g2_repair_independent_review_5b8daff.md` at
    `3c332053d8feaecd6f0bad9836444a1efd26ecff`.
  - Verified result: **candidate rejected / G2 FAIL**. The review identified:
    - the asserted branch merger was a duplicate-root artifact (two
      representations of one root separated by `1.18e-6`, just above the `1e-6`
      deduplication cutoff; strict 256-bit correction collapses them to
      `4.20e-48` apart);
    - the 256-bit event stage was a no-op (inherited residual below the
      precision-independent `1e-40` tolerance, exiting on iteration 1
      unchanged; the displayed extra digits are widening, not recovered source
      precision);
    - the default continuation trace accepted materially off-branch bordered
      points (`4.08e-4` displacement under strict correction at the same `k`).
  - Preserve: the independent review identified viable routes to genuine
    merger evidence (minimum/saddle pair tracking) and genuine precision
    refinement (strict tolerance requiring nontrivial iterations).
  - These are implementation/validation failures, not scientific-owner blocks.

### Further repair and acceptance

- [x] **T212 [R-003, R-005, R-006, R-007, R-012, R-014, G2] Freshly and independently review the further-repaired G2 candidate**
  - Reviewed revision: `cc8ac73668ac488a492dd16008c0b790a4e4ef3b` (further
    repair beyond the rejected `5b8daff...` candidate, correcting branch-
    fidelity with strict minimum/saddle pair tracking, genuine higher-precision
    event refinement with nontrivial iterations, and scale-aware continuation
    tolerance).
  - Evidence: `docs/src/issue_148_g2_final_independent_review_cc8ac73.md`,
    `docs/src/issue_148_g2_final_repair_evidence.md`.
  - Verified independently: bordered/pseudo-arclength mechanism genuinely
    exercised near singularity; halving radial step changes event by `7.57e-13`;
    a strictly distinct minimum/index-one saddle pair retains inertias 0/1 as
    separation shrinks from `2.021e-3` to `1.733e-6`; fallback explicit and
    bounded; old matcher independently exercised with five comparisons and zero
    disagreements; 128/256-bit event solves construct source quantities at
    target precision with agreement below `1e-30` and nontrivial iterations;
    `M96` reconstruction and source identity verified at working precision;
    P96/A96 like-for-like; source12/author10 distinct;
    conditioning/status/failure/tolerance justified; projected higher-derivative
    output reported without forcing a cusp label (`:unresolved` retained); G1
    preserved; no G3/API/schema expansion.
  - Verified result: **PASS recommendation** from fresh independent reviewer.
    Manager acceptance at `f9b04ed74bc30cc8e0071fcbe18179a4f85016e2`.

- [x] **T213 [R-006, R-007, R-012, R-014, G2] Record G2 gate acceptance**
  - Outcome: the T212 PASS review and the manager decision at `f9b04ed...`
    accepted the `cc8ac73...` implementation. This supersedes both earlier G2
    FAILs (`091cc9c...` for `9fe32eb...`; `3c33205...` for `5b8daff...`) for
    their explicitly identified older revisions.
  - Verified result: **G2 PASS** — accepted mechanism/evidence mapped to every
    G2 acceptance item. Durable decision at `f9b04ed...`.
  - Accepted scientific boundary: bounded zero-phase, fixed-saxion,
    source-twelve radial N8 one-null degeneracy under P96; positive transverse
    modes; projected classifier `:unresolved`; no stronger catastrophe
    classification.

## G3 — First off-ray Kahler discriminant investigation

G2 PASS unblocked G3. All G3 tasks below are now complete with accepted
evidence.

- [x] **T301 [R-003, R-008, R-009, R-012, G3] Freeze the accepted radial witness and Kahler-coordinate geometry**
  - Inputs: accepted G2 catastrophe witness at `cc8ac73...`, benchmark
    geometry/source identity, cone-adapted Kahler/two-cycle representation.
  - Evidence: `docs/src/issue_148_g3_control_audit.md`,
    `scripts/audit_issue_148_g3_controls.jl` at `b562780...`.
  - Verified result: exact source geometry reconstructed with 64 intersection
    entries, 39 Mori generators, exact two-cycle tip
    `t_ref=(1,4,4,-2,4,3,3,3)`, source actions
    `(14,29/2,29/2,31/2,31/2,31/2,31/2,16,17,17,25,45)`; paper radial `k` and
    Kahler coordinates explicit; no unconstrained four-cycle perturbation used.

- [x] **T302 [R-008, R-009, R-012, G3] Compute local Kahler-to-potential sensitivity/rank and choose one independent non-radial direction**
  - Outcome: rank two for both `[log(k),alpha]` action controls and
    row-centered full-log-amplitude controls, with singular values
    `(56.0275,8.14452)` and `(158.439,42.2188)`. Off-ray action derivative has
    relative residual `0.317` after best radial fit.
  - Evidence: `issue_148_g3_control_audit.md`, G3 independent review.
  - Verified result: direction `u=(0,1,2,-1,1,1,1,1)` supplies genuinely
    independent non-radial control; `MORI*u >= 0` verified; positive volume
    and metric over the certified slice.

- [x] **T303 [R-008, R-012, G3] Implement/assemble the bounded cone-valid deformation map**
  - Outcome: `t(k,alpha)=sqrt(k)(t_ref+alpha*u)` with `0 <= alpha <= 1/20`;
    divisor volumes, total volume, kinetic metric, instanton actions, potential
    coefficients, and control diagnostics recomputed consistently from exact
    integer/rational source data at every geometry evaluation.
  - Evidence: `issue_148_g3_control_audit.md`, `issue_148_g3_repair_evidence.md`.
  - Verified result: every accepted point satisfies Kahler-cone and
    physical-domain checks; deformation is not radial rescaling.

- [x] **T304 [R-001, R-010, R-012, R-015, G3] Determine the local discriminant fate without imposing persistence**
  - Input: accepted radial catastrophe and validated deformation from T301-T303.
  - Outcome: the source-twelve, zero-phase, fixed-saxion P96 degeneracy
    persists from the radial event onto one positive-alpha branch; 46 stored
    states (one seed + 45 accepted predictor/corrector transitions) reaching
    approximately `alpha=1.435e-4`, `k=0.5007`; stops at imposed scale guard
    with `:k_bounds_reached`.
  - Evidence: `issue_148_g3_repair_evidence.md`,
    `issue_148_g3_final_independent_review_4abd9c3.md`.
  - Verified result: persistence to a numerical boundary, not physical
    termination; opposing-seed failure does not establish absence of another
    branch; no curve was forced.

- [x] **T305 [R-011, R-012, G3] Test zero-phase cubic symmetry behavior and higher-derivative/transverse diagnostics**
  - Outcome: zero-phase cubic cancellation tested and not protected along this
    control direction. Projected D3/D4, transverse Hessian, nullity diagnostics
    reported at representative points.
  - Evidence: `issue_148_g3_repair_evidence.md`, G3 independent review.
  - Verified result: all stored states retain one near-null canonical mode and
    positive transverse spectrum; minimum transverse eigenvalue `0.0105284`;
    segment minimum P96 metric eigenvalue `1.2876e-4`.

- [x] **T306 [R-008, R-010, R-011, R-012, G3] Independently verify representative off-ray points or fate witnesses**
  - Outcome: independent/chained 128/256-bit solves verified four exact rational
    off-ray controls; independent probes verified derivative signs, full Hessian
    factor, normalization, continuity, accounting, and invalid-state statuses.
  - Evidence: G3 independent review (373 independent-probe assertions passed),
    prerequisite audit (79 assertions passed), G1 replay and diff checks passed.
  - Verified result: representative results replayable from recorded identity.

- [x] **T307 [R-002, R-008-R-012, R-014, R-015, G3] Complete fresh independent scientific G3 review**
  - Outcome: fresh independent adversarial review of the claimed local
    discriminant result against the approved spec, including claim boundary,
    outcome neutrality, geometry, precision and evidence sufficiency.
  - Evidence: `docs/src/issue_148_g3_final_independent_review_4abd9c3.md`,
    manager decision `docs/src/issue_148_g3_manager_decision.md` at
    `f33ba768f53fee749352a82c04aedc09919d1053`.
  - Verified result: **G3 PASS**. The earlier G3 FAIL review is
    revision-specific and superseded by this decision. No owner decision
    required.
  - Accepted scientific boundary: persistence only along the accepted audited
    positive-alpha off-radial direction and bounded segment; no global
    continuation, physical termination, exclusion of other branches; opposing-
    seed failure remains inconclusive.

## Optional G4 — Bounded physical probe

G3 PASS yielded a bounded off-ray catastrophe locus (46 stored states along the
positive-alpha branch). G4 is therefore applicable in principle, provided it
does not require changing the agreed physical model. G4 is not started.

- [ ] **T401 [R-002, R-016, G4] Select a small bounded set of points along the validated off-ray catastrophe locus for an exploratory physical probe**
  - Verify: selection is derived from the G3 locus and does not imply a
    probability/population claim.

- [ ] **T402 [R-002, R-012, R-016, G4] Apply existing detuning/inflationary diagnostics along the selected locus points and record exploratory evidence**
  - Verify: the probe is explicitly separated from the validated G3
    continuation/discriminant claim and makes no unsupported stabilization or
    prevalence claim.
  - Escalate if: the probe requires changing the agreed physical model,
    normalization or reported observable.

## Convergence

When a fresh independent review identifies a concrete unmet requirement, add a
bounded `TC##` correction task mapped to that requirement/gate rather than
rewriting an earlier completed task or marking the parent gate complete by
fiat.

Before eventual S2 completion, reconcile:

```text
approved spec
<-> plan
<-> tasks/evidence readiness
<-> implementation
<-> tests/scientific evidence
<-> PR scope
```

Any unmet requirement remains visible. Once the relevant deliverable/evidence
boundary is satisfied, merge/close/current-Kanban state remains GitHub state and
does not require a self-referential task-file update.
