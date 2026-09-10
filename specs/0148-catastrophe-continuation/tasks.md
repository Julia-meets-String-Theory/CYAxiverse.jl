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
- Scientific G3 tasks are blocked until GitHub Issue #148 records an explicit
  G2 PASS under the approved migrated contract.
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
- G2: not accepted; first candidate rejected, later repair candidate awaiting
  fresh independent scientific review at the cutoff
- G3: not started
- optional G4: not started; applicable only if G3 PASS yields a suitable
  off-ray catastrophe locus and the existing physical model remains usable
  unchanged

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

### Post-FAIL repair candidate at migration cutoff

- [x] **T211 [R-003, R-005, R-006, R-007, R-012, G2] Produce a bounded G2 repair candidate and revision-specific replay evidence**
  - Implementation candidate: `5b8daffd732ba307ed1980615b9f049a95b92c2c`.
  - Evidence state continues through migration-frozen PR head
    `e991495f1bb58edd7a7043dfc90771b6666c717c`.
  - Expected correction scope from the failed review: bordered-coordinate
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
    exists. The 128/256-bit choice records the current implementation/evidence
    plan; it is not a canonical requirement. This task does **not** mean G2 is
    accepted.

- [ ] **T212 [R-003, R-005, R-006, R-007, R-012, R-014, G2] Freshly and independently review the repaired G2 candidate**
  - Input: freeze the exact repaired code/evidence revision to be reviewed; do
    not review a moving branch head.
  - Verify independently:
    - the bordered/pseudo-arclength mechanism is genuinely exercised and
      adequate near singularity;
    - fallback is explicit, bounded and does not invalidate the scientific
      branch-continuation claim;
    - the old matcher is independently exercised rather than fed continuation
      identities;
    - the current 128/256-bit event solves construct source quantities at target
      precision and genuinely establish event stability;
    - reconstructed `M96` and its source identity are verified at working
      precision in the relevant GLSM basis;
    - any P96 metric represented in another basis is obtained through the
      corresponding explicit congruence transformation;
    - P96/A96 comparisons are like-for-like and source12/author10 remain
      distinct;
    - conditioning/status/failure/tolerance claims are justified by the replay;
    - projected higher-derivative output is reported without changing cutoffs
      or forcing a cusp label;
    - G1 remains preserved and no G3/API/schema expansion occurred.
  - Expected output: a new revision-specific independent scientific review with
    explicit PASS/FAIL recommendation and exact commands/observations.
  - Escalate if: a remaining question requires changing P96, another physical
    convention, catastrophe acceptance semantics, or the scientific claim
    boundary. Concrete implementation defects stay with the worker correction
    loop.

- [ ] **T213 [R-006, R-007, R-012, R-014, G2] Reach G2 evidence readiness after independent review**
  - Outcome: all concrete findings from T212 are either corrected and
    independently rechecked, or explicitly block G2; the evidence package is
    sufficient for manager gate adjudication.
  - Verify: accepted mechanism/evidence is mapped to every G2 acceptance item,
    with no reliance on worker DONE status or raw test count.
  - Boundary: the eventual PASS/FAIL gate decision itself is recorded on
    GitHub Issue #148, not treated as authoritative state in this task file.

## G3 — First off-ray Kahler discriminant investigation

All G3 tasks below are blocked until Issue #148 records explicit scientific
G2 PASS under the approved specification.

- [ ] **T301 [R-003, R-008, R-009, R-012, G3] Freeze the accepted radial witness and Kahler-coordinate geometry**
  - Inputs: accepted G2 catastrophe witness/revision, benchmark geometry/source
    identity, cone-adapted Kahler/two-cycle representation and control checks.
  - Expected output: replayable geometry/control packet defining the radial
    origin of the non-radial investigation.
  - Verify: paper radial `k`, Kahler coordinates and all basis/metric mappings
    are explicit; no unconstrained independent four-cycle perturbation is used.
  - Escalate if: coordinate interpretation or physical normalization is
    ambiguous.

- [ ] **T302 [R-008, R-009, R-012, G3] Compute local Kahler-to-potential sensitivity/rank and choose one independent non-radial direction**
  - Outcome: evaluate an appropriate local map such as
    `partial log|Lambda_I^4| / partial t^a` and demonstrate that the chosen
    direction supplies control independent of radial rescaling.
  - Verify: rank/sensitivity, direction `u`, normalization and allowed bounded
    neighborhood are recorded with geometry/source identity.
  - Escalate if: no usable independent direction exists within the validated
    local cone/control region.

- [ ] **T303 [R-008, R-012, G3] Implement/assemble the bounded cone-valid deformation map**
  - Outcome: vary the selected cone-adapted Kahler coordinate and consistently
    recompute dependent divisor volumes, total volume, kinetic metric,
    instanton actions, potential coefficients and applicable control
    diagnostics.
  - Verify: every accepted point satisfies the applicable Kahler-cone and
    physical-domain checks; deformation is not radial rescaling in disguise.
  - Escalate if: implementation would require a new scientific normalization,
    schema or physical-model decision.

- [ ] **T304 [R-001, R-010, R-012, R-015, G3] Determine the local discriminant fate without imposing persistence**
  - Input: accepted radial catastrophe and validated deformation from T301-T303.
  - Outcome: reproducibly determine whether the relevant structure persists,
    terminates, splits/unfolds, changes class, develops additional nullity, or
    exhibits another evidence-supported local discriminant outcome.
  - Verify: use appropriate augmented/local continuation analysis and retain
    truthful termination/failure states rather than forcing a curve.
  - Escalate if: observed behavior requires changing the agreed acceptance
    semantics rather than only technical realization.

- [ ] **T305 [R-011, R-012, G3] Test zero-phase cubic symmetry behavior and higher-derivative/transverse diagnostics**
  - Outcome: test rather than assume whether the symmetry-protected cubic
    normal-form coefficient remains zero along the legitimate Kahler
    deformation; report projected D3/D4, transverse Hessian, nullity and other
    relevant catastrophe diagnostics.
  - Verify: calculations are like-for-like in source/model/metric convention and
    retain declared precision/source-precision boundaries.
  - Escalate if: classification requires a normative cutoff/interpretation not
    fixed by the spec.

- [ ] **T306 [R-008, R-010, R-011, R-012, G3] Independently verify representative off-ray points or fate witnesses**
  - Outcome: independent checks of stationarity/degeneracy or the relevant
    termination/splitting witness, plus cone/geometric/EFT controls.
  - Verify: representative results can be replayed from recorded
    source/geometry/code/environment/precision identity.

- [ ] **T307 [R-002, R-008-R-012, R-014, R-015, G3] Complete fresh independent scientific G3 review**
  - Outcome: adversarial review of the claimed local discriminant result against
    the approved spec, including claim boundary, outcome neutrality, geometry,
    precision and evidence sufficiency.
  - Verify: a negative persistence result is evaluated as a legitimate
    scientific outcome rather than treated as failure solely because it is
    negative.
  - Boundary: gate adjudication/current workflow state remains on GitHub Issue
    #148 / Project.

## Optional G4 — Bounded physical probe

Run only after G3 PASS, only if G3 yields a suitable off-ray catastrophe locus,
and only if the existing physical model can be used without a new normative
choice. If G3 yields no such locus, G4 is N/A.

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
