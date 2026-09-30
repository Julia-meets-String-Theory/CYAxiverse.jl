---
spec_id: CYAX-0131
title: Low-N inflation calibration and bounded discovery contract
issue: 131
class: S2
status: draft
workstream: low-N inflation
parent: null
depends_on: [130, 172]
created: 2026-09-30
last_reviewed: null
review_required: independent Spec Reviewer + independent Scientific Reviewer
drafting_authority_ref: CYAxiverse-agent-exchange@main:handoff-reviews/cyax-0131-low-n-inflation-manager-handoff/r3/evidence/owner-s2-scientific-contract-approval-v1.json (SHA-256 75e1f0a3c9ec89052dde98e0b30a9386425b68ee080ebfdcb27407be7cf22da4)
approval_ref: N/A while draft; final owner S2 approval must bind the exact reviewed normative content
---

# Low-N inflation calibration and bounded discovery contract

## Objective

Define the scientific contract for CYAX-0131's low-N inflation calibration and
bounded discovery work. The contract fixes which N=8 model is authoritative,
how the approved phase benchmarks are defined, how homotopy and physical
scales are distinguished, and when trajectory diagnostics may be reported.
Any later scientific execution requires this specification to be approved and
a separate, current implementation handoff to authorize the exact work.

## Motivation and authority

CYAX-0131 needs a reproducible calibration route that keeps the paper-author
trajectory model distinct from the later P96/Table-1 reconstruction. Earlier
contract review identified ambiguity in model authority, phase assignment,
physical interpretation of `k`, and the point at which trajectory diagnostics
become eligible. The owner resolved those choices in the records below.

The approved scientific contract authorizes preparation and review of this
specification. It does not constitute final approval of these specification
bytes or authorize numerical execution.

### Owner-approved scientific decisions

The governing rebind decision is
`CYAxiverse-agent-exchange@main:handoff-reviews/cyax-0131-low-n-inflation-manager-handoff/r3/evidence/owner-scientific-rebind-decision-v1.json`
(SHA-256
`a9d76863dacee3aa88612c1bac4c4d5cf24a1dc2ac725f6a77c69f2d5979c77a`). The
owner's S2 contract approval is
`CYAxiverse-agent-exchange@main:handoff-reviews/cyax-0131-low-n-inflation-manager-handoff/r3/evidence/owner-s2-scientific-contract-approval-v1.json`
(SHA-256
`75e1f0a3c9ec89052dde98e0b30a9386425b68ee080ebfdcb27407be7cf22da4`). These
records are authority for the scientific choices below; they do not establish
results from a search or trajectory replay.

### Source and implementation baseline

The contract is drafted against `CYAxiverse.jl` source snapshot
`a5eacd8cd4a5905161cab239a461ba252c64e8e0`, tree
`d419416514a44d943ddd20290885ac8ba4090999`. Bound source identities recorded
for that snapshot are:

- `src/paper_benchmarks/poly102_inflation.jl`, blob
  `9ad74dc6adeeff6fc92c1e52f7b113df29e30b1d`: current corrected N=8 author
  and full-potential source, including the integrated Eq. 19 repair.
- `src/paper_benchmarks/n8_continuation.jl`, blob
  `ff486c0fc982de0006553fd2d963365f07820e6d`: P96/Table-1 continuation
  cross-check source.
- `src/paper_benchmarks/catastrophe_diagnostics.jl`, blob
  `df119eb8899de593300e9a4db8ad7cfc1962b77f`: diagnostic definitions.
- `scripts/phase_volume_detuning_scan.jl`, blob
  `91c66cdf15d6fe0bf9bd759b66898e920c70cd3e`: low-N phase and volume scan
  surface.
- `scripts/inflation_candidate_refinement.jl`, blob
  `60bc2c493fa002cd5c906330cd3267d8315d036d`: candidate refinement surface.

The integrated N=8 Eq. 19 repair is present from commit
`e47413c8d2185c1886a58237e5fa4e1ce0594682`. Preserve the corrected
`q·tau`/cross-coefficient convention; do not restore the superseded extra-pi
form. The current source also descends from the Issue #172 Hessian
normalization correction, merge commit
`eda8f21512913c147090d819ed3cf04d74d701ae`; preserve its `4π²` normalization
ancestry. These are baseline source facts, not choices reopened by CYAX-0131.

### Evidence categories

- **Source facts:** repository paths, source revisions, source-declared model
  routes, and integrated correction ancestry listed above.
- **Owner-approved conventions:** model authority, phase vectors, k hierarchy,
  diagnostic eligibility, deferred observational items, and claim boundaries
  stated in this specification.
- **Implementation facts:** no new implementation or execution fact is
  established by this document-preparation tranche.
- **Empirical results:** none are asserted here. A future result requires
  exact source/code identity, selected model route, row count and phase
  identity, k category, coordinate/metric identity, named witness, units,
  numerical status, and relevant environment details.
- **Inference:** a reconstruction or interpretation not directly established
  by the bound source or owner decision remains labeled as inference and must
  not override an approved convention.

## Scope

This contract governs later calibration and bounded discovery for CYAX-0131:

- establish the author-authoritative N=8 catastrophe calibration in the
  10-row author trajectory model;
- apply the owner-specified N=8 nonzero-phase benchmark and establish its
  source-consistent two-sided critical-point identities and refined
  catastrophe location before treating its trajectory observables as
  calibrated physical diagnostics;
- retain the separate 12-row P96/Table-1 continuation as a cross-check;
- define the N=5 nonzero-phase replay only for the documented reduced
  two-cosine light-direction model;
- require independent physical-k re-establishment before reporting physical
  trajectory diagnostics; and
- preserve the fixed-saxion effective-theory claim boundary.

## Non-scope

This specification does not authorize or establish:

- a population scan, population prevalence, or a full-KS claim;
- dynamical saxion stabilization, fully stabilized string cosmology, or a
  claim beyond the fixed-saxion effective theory;
- an observational acceptance window or observational optimization;
- a new scalar-amplitude conversion or acceptance rule;
- a new `n_s` acceptance window or a tensor-to-scalar ratio result;
- a full physical N=5 trajectory or an invented eight-row phase assignment;
- execution before final owner approval and a new reviewed implementation
  handoff; or
- implementation, merge, release, or Issue #131 closure by this contract
  preparation.

## Scientific claim boundary

This work may establish bounded calibration/discovery facts for the specified
fixed-saxion effective-theory models, with each evidence item labeled by model
and scale convention. It may report the owner-approved N=8 trajectory
diagnostics only after physical `k` has been independently re-established.

It must not promote homotopy evidence into physical evidence, a bounded search
into a population claim, fixed-saxion results into dynamical stabilization, or
catastrophe-point Hessian eigenvalues into pivot-scale or along-trajectory
spectra. No claim of observational viability follows from the permitted
diagnostics alone.

## Fixed conventions and invariants

1. **Authoritative N=8 trajectory model:** use the 10-row author trajectory
   model identified by the paper-accompanying Mathematica source and mirrored
   by the `author_inflation` trajectory route. In the repository source, name
   the concrete route `author_inflation.n8_author_trajectory(...)` and its
   returned `samples`; do not assume an unverified `trajectory_observables`
   API is the route. Label this model as the 10-row author trajectory model.
2. **N=8 phase benchmark:** the nonzero phase vector has length ten,
   `phase[2] = 0.04` radians, and zero in every other row. Do not interpret a
   scalar phase value without its row assignment.
3. **N=8 calibration prerequisite:** establish source-consistent identities
   for the two critical points on the relevant sides and refine the
   catastrophe location in this same 10-row model before its trajectory
   observables are used as calibrated physical diagnostics.
4. **P96 separation:** the 12-row P96/Table-1 continuation is a separate
   reconstruction/extended benchmark and cross-check. It cannot certify or
   substitute for the 10-row author-model calibration or its diagnostics.
5. **N=5 replay:** use the documented reduced two-cosine light-direction
   model, with `delta = π/4` on the second cosine. Re-solve/refine the
   phase-shifted fold. The full eight-row phase mapping is not required for
   this calibration; if no explicit source mapping is established, label it
   `NOT_VERIFIABLE` and do not invent one.
6. **Scale hierarchy:** a homotopy `k` supports discovery/calibration only.
   Independently re-establish a physical `k` before reporting any physical
   trajectory observable. Record the scale category with every result.
7. **Diagnostic eligibility:** after physical-k re-establishment, the only
   permitted N=8 trajectory diagnostics in this contract are `N_e`, `n_s`,
   scalar amplitude under the `paper_delta_H` convention, and cumulative
   turning. Use the concrete N=8 trajectory and its sample records as the
   data route.
8. **Transverse Hessian:** bind transverse-Hessian eigenvalues to a named
   catastrophe diagnostic point, including model, phase vector, coordinates,
   scale category, metric/basis, and source/code identity. Do not describe
   these values as pivot-scale or along-trajectory spectra.
9. **Observational stage:** all five items below remain `NOT_REACHED` in this
   contract tranche:
   - new `A_s` conversion/acceptance window;
   - new `n_s` acceptance window;
   - tensor-to-scalar ratio `r`;
   - full physical N=5 trajectory; and
   - CYAX-0131 observational stretch goal.
10. **Model scope:** conclusions remain fixed-saxion effective-theory
    calibration/discovery. No population prevalence, dynamical saxion
    stabilization, fully stabilized string cosmology, or full-KS claim is
    permitted.
11. **Integrated source corrections:** retain the corrected Eq. 19
    `q·tau`/cross-coefficient form and the Issue #172 `4π²` Hessian
    normalization ancestry. Neither is reopened by this work.
12. **Execution gate:** these documents and their review do not authorize
    scientific execution. Execution requires final owner approval of the exact
    S2 normative content and a new, separately reviewed implementation
    handoff that binds current source and execution scope.

## Requirements

### R-001 — Use the source-authoritative 10-row N=8 model

All CYAX-0131 N=8 catastrophe calibration SHALL use the 10-row author
trajectory model. Evidence SHALL identify the selected model and the concrete
`author_inflation.n8_author_trajectory(...)`/returned-sample route. A result
from another row set SHALL NOT be represented as this author-model result.

### R-002 — Bind and calibrate the N=8 nonzero-phase benchmark

The nonzero-phase benchmark SHALL set only row 2 to `0.04` radians and all
other phases to zero. Before its trajectory diagnostics are called calibrated
physical diagnostics, the evidence SHALL establish source-consistent identities
for both critical points on the relevant sides and a refined catastrophe
location in the same 10-row model. The required identity/replay tolerances and
solver acceptance criteria must come from the reviewed implementation
handoff; this specification introduces no numerical threshold.

### R-003 — Keep the P96/Table-1 continuation separate

The 12-row P96/Table-1 continuation MAY be reported as a separate
reconstruction/extended-benchmark cross-check. Its result SHALL identify the
12-row route and SHALL NOT certify, replace, or be merged with the 10-row
author-model calibration or diagnostics.

### R-004 — Re-solve the reduced N=5 phase-shifted fold

The N=5 phase replay SHALL use the documented reduced two-cosine
light-direction model with `delta = π/4` on the second cosine and SHALL
re-solve/refine the shifted fold. A full eight-row phase map is not required
for this calibration. Without explicit source authority for that map, its
status SHALL be `NOT_VERIFIABLE`; no map may be inferred or invented.

### R-005 — Separate homotopy and physical `k`

Evidence using homotopy `k` SHALL be labeled discovery/calibration evidence
only. Before any physical N=8 trajectory diagnostic is reported, physical `k`
SHALL be independently re-established and identified separately from the
homotopy value. Failure to establish physical `k` blocks physical diagnostic
claims but does not turn homotopy evidence into physical evidence.

### R-006 — Restrict and route permitted N=8 diagnostics

Only after R-005 is satisfied, the N=8 trajectory diagnostic set MAY include
`N_e`, `n_s`, `paper_delta_H` scalar amplitude, and cumulative turning. The
evidence SHALL identify the `author_inflation.n8_author_trajectory(...)`
execution and the returned sample records from which each quantity is
reported. This requirement does not create an observational acceptance
criterion.

### R-007 — Bind transverse-Hessian values to a named catastrophe point

Any transverse-Hessian eigenvalues SHALL be attached to a named catastrophe
diagnostic point whose model, phase assignment, coordinate, scale category,
metric/basis, and code/source identity are recorded. They SHALL NOT be
interpreted as pivot-scale or along-trajectory spectra.

### R-008 — Keep the five observational items unreached

The new `A_s` conversion/acceptance window, new `n_s` acceptance window,
tensor-to-scalar ratio `r`, full physical N=5 trajectory, and CYAX-0131
observational stretch goal SHALL each remain `NOT_REACHED` under this
specification. None is a success criterion for the bounded calibration
contract.

### R-009 — Preserve the fixed-saxion claim boundary

Every conclusion SHALL be stated as fixed-saxion effective-theory
calibration/discovery. It SHALL NOT claim population prevalence, dynamical
saxion stabilization, fully stabilized string cosmology, or full-KS coverage.

### R-010 — Preserve integrated Eq. 19 and Issue #172 corrections

CYAX-0131 work SHALL preserve the integrated Eq. 19 `q·tau`/cross-coefficient
repair and the current Issue #172 `4π²` Hessian normalization ancestry. It
SHALL NOT restore the superseded extra-pi form or silently reopen either
correction.

### R-011 — Enforce approval and handoff gates

No scientific search, catastrophe/trajectory numerical execution, candidate
refinement, or observational optimization is authorized until (a) independent
SPEC and SCIENTIFIC reviews accept the same exact S2 candidate, (b) Control
Desk reconciliation is complete, (c) the owner approves the exact normative
content, and (d) a new current-schema implementation handoff is separately
reviewed and authorizes that execution. Any material source or scientific
contract drift returns to the owner before execution.

## Acceptance and review

This candidate is complete for specification preparation when the three
requested documents exist, all R-001–R-011 requirements map consistently
through plan and tasks, and document-only validation reports no path or
formatting defect. S2 contract acceptance requires fresh independent SPEC and
SCIENTIFIC reviews bound to the same exact final document candidate, Control
Desk reconciliation, and a separate owner final approval bound to that exact
normative content. A normative-byte change after review requires both reviews
to be repeated.

No numerical, physical, observational, population, or implementation result
is an acceptance criterion for this specification-preparation tranche.
