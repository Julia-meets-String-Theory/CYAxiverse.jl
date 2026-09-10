---
spec_id: CYAX-0148
title: Continue catastrophic-inflation structures away from the radial Kahler ray
issue: 148
class: S2
status: draft
workstream: Inflation
parent: null
depends_on: []
created: 2026-09-10
last_reviewed: 2026-09-10
review_required: scientific-owner-after-independent-scientific-review
approval_ref: N/A while draft
version_impact: no feature-branch bump
---

# Continue catastrophic-inflation structures away from the radial Kahler ray

## Objective

Develop and scientifically validate genuine numerical continuation of critical
points and catastrophe/discriminant structure in `CYAxiverse.jl`, progressing
from analytic and published radial benchmarks to one controlled non-radial
deformation inside the Kahler cone.

The first new-science milestone is to determine the **fate of the known N=8
radial catastrophe under one physically valid non-radial Kahler deformation**.
A persistent local catastrophe locus is one admissible outcome, but persistence
is not required. Termination, splitting/unfolding, a change of catastrophe
class, or development of additional null directions are also admissible if
established reproducibly with geometric/EFT control.

Generic active or agentic exploration of the full Kahler cone is outside this
specification and remains a possible follow-on goal.

This is a brownfield SDD migration of GitHub Issue #148 and PR #149. New
requirement identifiers below describe the migrated current contract; they were
not the identifiers under which historical G0/G1 work was originally executed.

## Motivation

The published catastrophic-inflation construction varies Kahler data along a
one-dimensional radial scaling trajectory. The underlying Kahler moduli provide
a higher-dimensional control-parameter space. Rather than search directly for
inflation with a global phenomenological merit function, this pilot treats the
problem as a bifurcation/discriminant problem,

```math
F(\theta,t) \equiv \nabla_\theta V(\theta;t)=0,
```

with degenerate critical points characterized by an augmented system such as

```math
\nabla_\theta V = 0,\qquad
H_\theta v = 0,\qquad
v^\mathsf{T}v=1.
```

The scientific question is whether and how the radial catastrophe embeds into
nearby physically valid Kahler control directions. The software question is
whether CYAxiverse can follow the relevant branches and diagnose that local
structure with source-faithful conventions, controlled precision, intrinsic
branch identity, and replayable evidence.

## Current baseline

This section records the brownfield migration cutoff; live workflow state after
that cutoff belongs in GitHub Issue/PR/Project rather than being inferred from
this document.

### Migration cutoff

- CYAX-0151 G2 Project/Kanban prerequisite: PASS, recorded on Issue #151 in
  comment `5625939343`.
- `vmm` migration base: `6434e133af4b91db81babfafa024fc3cbada901a`.
- Issue #148: open at migration inspection.
- PR #149: draft and unmerged at migration inspection.
- PR #149 migration-frozen head:
  `e991495f1bb58edd7a7043dfc90771b6666c717c`.
- Scientific gate state at that cutoff: G0 PASS; G1 PASS; G2 not accepted after
  a recorded FAIL, with a later repair candidate/evidence awaiting fresh
  independent scientific review; G3 not started; optional G4 not started.

### Source facts

The durable G0 audit identifies the scientific source as *Catastrophic
Inflation in the Axiverse*, arXiv:2608.14780v1, with source PDF SHA-256
`b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
Relevant source facts preserved by this specification include:

- the radial control `k` scales divisor/four-cycle volumes as
  `tau^i(k) = k tau^i(k=1)`;
- source axions are period-one coordinates with Fourier argument
  `2pi Q theta + delta`, where `delta` is in radians;
- the analytic N=5 catastrophe scale is
  `(4/pi) log(1024/255) = 1.770068132610995629...`;
- the published N=8 Table-1 benchmark contains twelve displayed action terms
  and has zero-phase catastrophe scale approximately
  `0.674506370003365`;
- the separate executable-author trajectory retains ten leading terms and must
  remain distinguishable from the twelve-term source benchmark;
- the radial benchmark family is a fixed-saxion construction and does not by
  itself establish saxion/moduli stabilization along the family.

### Implementation facts at migration cutoff

The Issue #148 branch/PR introduces and/or changes the principal paths:

- `src/paper_benchmarks/poly102_inflation.jl`;
- `src/paper_benchmarks/n8_continuation.jl`;
- `src/paper_benchmarks.jl`;
- `scripts/inflation_scale_continuation.jl`;
- `scripts/issue_148_g1_n5_regression_tests.jl`;
- `scripts/issue_148_g1_replay_checks.jl`;
- `scripts/issue_148_g2_continuation_evidence.jl`;
- corresponding focused tests and durable review/evidence documents under
  `docs/src/`.

G1 repaired a pre-existing N=5 fixture/validation defect that reused the N=8
critical scale and added genuine N=5 branch continuation and source-constructed
high-precision validation.

The first G2 candidate was rejected after independent review. A later repair
candidate exists at `5b8daffd732ba307ed1980615b9f049a95b92c2c`, with subsequent
revision-specific evidence recorded through the migration-frozen PR head. The
repair evidence reports corrected bordered-coordinate semantics, actual
post-hoc matcher exercise, target-constructed 128/256-bit augmented event
solves, explicit fallback/failure provenance, and like-for-like normalization
diagnostics. At the migration cutoff this repair had **not** received durable
fresh independent scientific acceptance, so it does not change the recorded
G2 gate state.

### Empirical evidence already accepted

Historical accepted evidence includes:

- G0 baseline/scientific-contract audit at `62c5a61...`;
- G1 tested code `792a02fb77d22cefdfec8fe77969d9c719e7b6d0`;
- G1 manager decision `1eef936e1908959db50a969e717b16a9c9e414bf`;
- 63/63 focused G1 regression assertions and genuine 128/256-bit source
  construction/continuation stability recorded in the G1 evidence;
- the owner-approved P96 N=8 metric contract recorded durably at
  `d1a0b709b69aa680b9bca4223739366c3bfdf8b6` and Issue #148 comment
  `5623902670`.

The first G2 candidate evidence is retained as revision-specific historical
evidence but not as acceptance evidence. The independent rejection and manager
decision at `091cc9c...` control the status of that candidate.

### Chronological decision / supersession record

1. **Original Issue #148 contract (2026-09-09).** The Issue introduced G0-G4,
   with G3 originally requiring a nontrivial local catastrophe curve/locus.
2. **G0 PASS — Issue comment `5608088958`.** The apparent N=5 discrepancy was
   classified as a source-fixed implementation/validation defect rather than
   an unresolved scale convention. No scientific convention changed.
3. **Owner clarification — Issue comment `5609698767`.** This added the
   target-precision/tolerance requirements for G1, strengthened G2
   near-singular/precision/higher-derivative requirements, and **superseded the
   original G3 persisted-curve requirement** with outcome-neutral local
   discriminant semantics. It also required a local sensitivity/rank
   diagnostic before expensive off-ray continuation and a test, rather than an
   assumption, of the zero-phase cubic symmetry constraint.
4. **G1 PASS — Issue comment `5610658451` and manager decision `1eef936...`.**
   Source N=5 scale, continuation, failure boundaries, and high-precision
   replay were accepted after fresh independent review.
5. **N=8 metric-boundary audit — `dd12209...`.** A source/author normalization
   conflict and a factor-two displayed-equation versus Eq.96/CYTools matrix
   conflict were identified; implementation stopped for owner choice.
6. **P96 owner approval — Issue comment `5623902670`, durable contract
   `d1a0b70...`.** Scientific N=8 outputs use period-one coordinates with
   `K_theta=M96/k^2`; equivalent raw-radian coordinates use
   `G_x=M96/[k^2(2pi)^2]`. Eq.96/CYTools matrix authority is selected over the
   literal `2M96` result from displayed Eqs.17/23. Unconverted author
   raw-radian `M96/k^2` is reproduction-only A96.
7. **First G2 candidate rejected — Issue comment `5624861152`, manager
   decision `091cc9c...`.** Passing checks did not establish genuine
   near-singular continuation, an independent matcher comparison, a genuine
   event precision ladder, or like-for-like canonical comparisons. The
   projected higher-derivative classification remained honestly `:unresolved`
   and was not promoted to a cusp claim.
8. **Post-FAIL G2 repair candidate — migration-frozen PR head `e991495...`.**
   Repair implementation/evidence exists, but no later durable independent
   review or manager PASS was present at the migration cutoff. Therefore G2
   remains not accepted in this migrated baseline.

Later durable owner decisions supersede earlier wording only where their scope
actually conflicts. Historical wording and failed-review evidence remain part
of the record rather than being rewritten.

## Scope

This specification includes:

- exact recovery of the N=5 analytic catastrophe benchmark;
- genuine N=5 critical-branch continuation and failure-boundary validation;
- N=8 radial multifield continuation against the published twelve-term
  benchmark;
- independent augmented catastrophe validation;
- the approved P96 canonical metric convention for scientific N=8 outputs;
- intrinsic branch identity plus comparison with the existing post-hoc matcher;
- source-constructed arbitrary-precision validation where required;
- higher-derivative catastrophe/discriminant diagnostics;
- one controlled, physically valid non-radial Kahler deformation;
- determination of the local fate of the radial catastrophe under that
  deformation;
- an optional, bounded physical probe after G3 passes only if G3 yields a
  suitable off-ray catastrophe locus and no model change is required.

## Non-scope

This specification does not include:

- generic adaptive or agentic search of the full Kahler cone;
- a probability measure on the Kahler cone;
- population-level prevalence of catastrophic inflation;
- completeness of catastrophe discovery;
- a requirement that inflation succeed away from the radial construction;
- a requirement that a catastrophe locus persist under the non-radial
  deformation;
- stabilization of saxions/moduli beyond the fixed-saxion benchmark
  assumptions;
- changing a physical normalization, basis, acceptance criterion, reported
  observable, population definition, or scientific schema without explicit
  owner review;
- a persisted-data schema migration or package release/version bump as part of
  the scientific gates.

## Scientific claim boundary

### This specification may establish

- numerical continuation of named critical-point branches;
- reproduction of the analytic N=5 and published N=8 radial catastrophe
  benchmarks under explicitly stated conventions;
- a source-faithful canonical N=8 diagnostic under the approved P96 contract;
- the local fate of the radial catastrophe/discriminant under one specified,
  physically valid non-radial Kahler deformation;
- local catastrophe diagnostics and their variation within the validated
  deformation segment;
- a scientifically well-evidenced negative local result, including failure of
  persistence, when that is the observed outcome.

### This specification must not establish

- generic frequency or probability of catastrophic inflation in the
  Kreuzer-Skarke ensemble;
- completeness of catastrophe discovery in Kahler moduli space;
- successful inflation away from the radial benchmark solely from catastrophe
  persistence or degeneracy;
- population-level prevalence;
- stabilized saxion/moduli dynamics beyond the fixed-saxion assumptions;
- validity outside explicitly checked Kahler-cone, geometric, numerical and
  EFT-control conditions;
- a cusp or other catastrophe class when the approved diagnostic evidence is
  `:unresolved`.

## Fixed conventions and invariants

- Preserve `AGENTS.md` and relevant CYAxiverse scientific/numerical policy.
- Paper radial `k` means `tau_four-cycle(k)=k tau(1)` for the identified
  benchmark geometry. Any two-cycle/Kahler-coordinate realization must state
  the corresponding mapping consistently.
- Source axions use period-one `theta` and Fourier argument
  `2pi Q theta + delta`.
- N=5 uses the source critical scale
  `(4/pi) log(1024/255)`, not the historical N=8-value reuse.
- Arbitrary-precision validation constructs source quantities at target
  precision. Float64 values must not be widened and then represented as though
  source precision had been recovered.
- Numerical continuation code must preserve the intended input numeric type and
  must not silently narrow a high-precision path to Float64.
- Scientific N=8 canonical outputs use P96:
  `K_theta(k)=M96/k^2` in period-one GLSM coordinates. In raw radians
  `x=2pi theta`, use `G_x(k)=M96/[k^2(2pi)^2]`.
- `M96` is the precise reconstructed Eq.96/CYTools reference metric in the
  relevant GLSM basis. Its reconstruction and source identity must be verified
  at working precision.
- Any representation of the P96 metric in another basis must carry the metric
  through the corresponding explicit congruence transformation.
- The documented factor-two discrepancy with literal displayed Eqs.17/23
  remains a source inconsistency; do not describe M96 as derived from that
  displayed normalization.
- The author raw-radian `M96/k^2` convention is A96 reproduction-only and must
  remain explicitly labelled.
- The twelve-term published N=8 potential and ten-term author truncation are
  distinct calculation paths and must not be mixed in like-for-like
  normalization claims.
- A non-radial deformation must be built in Kahler-form/two-cycle coordinates
  or another explicitly validated cone-adapted parameterization. Four-cycle
  volumes must not be perturbed as unconstrained independent coordinates.
- All dependent physical quantities affected by the deformation must be
  recomputed consistently, including as applicable divisor volumes,
  Calabi-Yau volume, kinetic metric, instanton actions, potential coefficients,
  and control diagnostics.

## Requirements

### R-001 — Outcome-neutral scientific objective

CYAxiverse SHALL determine the local fate of the validated N=8 radial
catastrophe under one physically valid non-radial Kahler deformation without
requiring persistence of a catastrophe curve as the successful outcome.

### R-002 — Claim-boundary preservation

All gate evidence and final conclusions SHALL remain within the scientific
claim boundary above and SHALL distinguish local/replayable evidence from
population, inflationary, or moduli-stabilization claims.

### R-003 — Source and representation identity

Every named benchmark result SHALL identify the source/revision, term set,
coordinate representation, radial/control parameter meaning, and relevant
basis/metric convention sufficiently to prevent source12/P96, author10/A96,
or homotopy quantities from being conflated.

### R-004 — N=5 source-faithful continuation and precision

N=5 validation SHALL use the analytic source critical scale, genuine branch
continuation, scale-justified event/fold tolerances, preserved numeric types,
and at least one genuinely target-constructed high-precision rerun whose
critical scale, critical point, residual, and Hessian behavior are stable with
increased precision.

### R-005 — P96 N=8 scientific metric contract

Scientific N=8 canonical diagnostics SHALL use the owner-approved P96
coordinate/matrix contract. `M96` SHALL be the precise reconstructed Eq.96/
CYTools reference metric in the relevant GLSM basis, with its reconstruction
and source identity verified at working precision. Any representation in
another basis SHALL apply the corresponding explicit metric congruence
transformation. A96 MAY be retained only as explicitly labelled author
reproduction.

### R-006 — Well-posed N=8 near-singular continuation

The N=8 radial benchmark SHALL be followed by continuation that remains
well-posed as the Hessian approaches singularity. Pseudo-arclength is the
preferred baseline; another method is acceptable only if its equivalent
adequacy near the catastrophe is demonstrated rather than assumed.

### R-007 — Independent N=8 validation and precision ladder

N=8 G2 evidence SHALL include an independent augmented catastrophe comparison,
intrinsic branch identity, an actual comparison with the existing post-hoc
matching machinery, explicit failure/status/tolerance/conditioning evidence,
and a genuine Float64-discovery to target-constructed higher-precision
refinement/stability path sufficient to establish numerical stability.
Projected higher-derivative diagnostics SHALL be reported without forcing a
classification.

### R-008 — Physically valid non-radial deformation

The G3 deformation SHALL be constructed in Kahler-form/two-cycle coordinates or
another validated cone-adapted representation, with all dependent quantities
recomputed consistently and applicable Kahler-cone/physical-domain checks
satisfied at every accepted point.

A schematic pilot may use

```math
t(s,\alpha)=s\left(t_{\rm ref}+\alpha u\right),
```

provided the meaning of `s`, direction `u`, allowed `alpha` interval, relation
to the source radial coordinate, and cone/control conditions are validated.

### R-009 — Independent-control sensitivity before expensive G3 continuation

Before expensive off-ray continuation, compute a local sensitivity/rank
diagnostic for the map from independent Kahler directions to the relevant
instanton actions/amplitudes, for example an appropriately defined
`partial log|Lambda_I^4| / partial t^a`, and establish that the chosen
non-radial direction supplies genuinely independent control rather than radial
rescaling in another parameterization.

### R-010 — Determine the local discriminant fate

Starting from an independently accepted radial N=8 catastrophe, G3 SHALL
determine reproducibly what happens under the selected non-radial deformation.
Persistence, termination, splitting/unfolding, changed catastrophe class,
additional null directions, or another evidence-supported local discriminant
outcome are admissible.

### R-011 — Symmetry and higher-derivative diagnostics

At zero phase, G3 SHALL test rather than assume whether the
symmetry-protected cubic normal-form coefficient remains zero along legitimate
Kahler directions. Representative points SHALL report projected third/fourth
derivatives, transverse Hessian information, nullity/classification evidence,
and relevant geometric/EFT controls.

### R-012 — Replayable evidence and identity

Scientific gate evidence SHALL record enough identity to replay the result,
including source/revision, code revision, geometry/witness identity where
applicable, coordinate/metric convention, term selection, precision, exact
commands, observed values/tolerances, and relevant environment/tool versions.

### R-013 — Scientific ambiguity fails closed

IF implementation requires choosing or changing a physical normalization,
basis convention, catastrophe acceptance criterion, Kahler-coordinate
interpretation, reported observable, or other unresolved normative scientific
choice, THEN work SHALL stop for the scientific owner rather than infer the
choice silently.

### R-014 — Gate sequencing and evidence status

Gates SHALL advance only from evidence accepted under their current contract.
Rejected candidate evidence remains historical evidence and SHALL NOT be
promoted to gate completion merely because tests passed or a repair candidate
exists. G3 SHALL NOT begin scientifically until G2 is explicitly accepted.

### R-015 — Negative scientific outcomes are valid outcomes

A reproducible negative scientific result MAY satisfy G3 when it determines the
fate of the radial catastrophe under the chosen deformation within the agreed
claim/control boundary. Failure of the desired inflationary or persistence
hypothesis is not automatically failure of the research/software task.

### R-016 — Optional physical probe remains separate and locus-conditional

Any post-G3 bounded physical probe SHALL remain explicitly exploratory and
separate from the validated continuation/discriminant claim. It SHALL run only
if G3 yields a suitable off-ray catastrophe locus from which well-defined
points can be selected, and only if the probe does not require changing the
agreed physical model. If G3 yields no such locus, optional G4 is not
applicable.

## Acceptance gates

The gate definitions below are the migrated canonical contract. Historical
PASS/FAIL statements describe only the migration-cutoff evidence and do not
project these new requirement IDs backward onto past execution.

### G0 — Baseline and scientific-contract audit

**Objective:** Establish the exact implementation and benchmark conventions
before adding continuation machinery.

**Acceptance:** Durable evidence identifies existing radial/scale behavior,
post-hoc branch matching, augmented catastrophe solvers, N=5/N=8 benchmark
sources and scale meanings, the N=5 discrepancy classification, and applicable
precision/basis/phase/metric/physical-scaling conventions.

**Stop condition:** If the N=5 discrepancy cannot be resolved from source and
implementation facts, return the smallest required owner decision before G1.

**Migration-cutoff historical result:** PASS. The N=5 discrepancy was
classified as a source-fixed implementation/validation defect.

### G1 — Analytic N=5 continuation validation

**Objective:** Implement and validate genuine numerical continuation for the
analytically controlled N=5 catastrophe.

**Acceptance:** Demonstrate, to documented and scale-justified tolerances, that
one known regular branch is genuinely continued to the analytic catastrophe;
branch identity is intrinsic rather than post-hoc; gradient/Hessian/event
conditions agree with the analytic oracle; failure boundaries are covered; and
target-constructed high-precision reruns preserve type/precision and show
stable critical quantities as precision increases.

**Stop condition:** Do not begin N=8 scientific G2 unless unexplained numerical
or convention-dependent N=5 failures are resolved.

**Migration-cutoff historical result:** PASS after fresh independent scientific
review.

### G2 — N=8 radial multifield validation

**Objective:** Apply the validated continuation method to the published N=8
catastrophic-inflation construction along the source radial control direction.

**Acceptance:** Starting from regular roots away from the catastrophe:

1. identify and genuinely continue the relevant branch/branches using a
   well-posed near-singular method;
2. reproduce the expected radial branch-merger/degeneracy structure without
   substituting post-hoc matching for continuation;
3. recover an event consistent with the independent twelve-term augmented
   catastrophe solve;
4. use the approved P96 scientific metric contract for canonical outputs,
   verify the reconstructed `M96` metric/source identity at working precision,
   apply explicit congruence transformation for any other-basis representation,
   and keep A96/source12/author10 comparisons explicitly separated;
5. verify gradient, null-vector, normalization, canonical Hessian,
   conditioning/status/failure behavior, and justified numerical tolerances;
6. run a genuine target-constructed higher-precision event/refinement
   stability ladder from Float64 discovery, with no hidden Float64 narrowing or
   widening-as-source-precision, sufficient to demonstrate stability as
   precision increases;
7. compare intrinsic continuation identity with the actual existing post-hoc
   periodic-distance matcher and report any disagreements;
8. report projected higher-derivative classification honestly, including
   `:unresolved` if that is what the evidence supports.

**Stop condition:** Do not release or scientifically execute a non-radial G3
Kahler direction until G2 is explicitly accepted with an agreed scientific
interpretation.

**Migration-cutoff historical state:** The first candidate was FAIL after fresh
independent review. A post-FAIL repair candidate/evidence exists at the frozen
PR head but was not yet independently accepted; therefore G2 remains not
accepted at the migration cutoff.

### G3 — First off-ray Kahler discriminant continuation

**Objective:** Determine the local discriminant/catastrophe structure under one
physically valid non-radial Kahler deformation through the validated N=8 radial
construction.

**Acceptance:** After G2 PASS:

1. establish by local sensitivity/rank analysis that the chosen deformation
   supplies genuinely independent Kahler control;
2. construct and validate one non-radial cone-adapted deformation direction;
3. starting from the accepted radial catastrophe, follow or otherwise resolve
   the nearby augmented critical/degenerate structure sufficiently to determine
   its local fate;
4. treat persistence, termination, splitting/unfolding, changed class, or
   additional null directions as admissible outcomes rather than forcing a
   persisted curve;
5. independently verify the relevant stationarity/degeneracy conditions at
   representative points or termination/splitting witnesses;
6. record Kahler-cone, geometric and EFT-control diagnostics throughout the
   validated segment/region;
7. demonstrate that the deformation is not radial rescaling in another
   parameterization;
8. test the zero-phase cubic normal-form coefficient rather than assuming its
   symmetry protection persists, and report projected third/fourth derivatives,
   transverse Hessian information and other existing diagnostics;
9. preserve replayable geometry/source/code/environment identity.

Successful inflation along the deformation is not required for G3.

**Stop condition:** Any ambiguity in deformation coordinates, normalization,
physical interpretation, or acceptance criteria that could change scientific
meaning returns to the scientific owner. A failure of catastrophe persistence
is an admissible scientific result, not by itself a stop/failure condition.

### Optional G4 — Bounded physical probe

G4 is applicable only if G3 passes **and** yields a suitable off-ray
catastrophe locus. If G3 establishes termination, unfolding, or another outcome
without such a locus, G4 is not applicable.

Where applicable, run G4 only if it does not require changing the agreed
physical model. Select a small number of well-defined points along the validated
off-ray catastrophe locus, detune them using the existing catastrophic-inflation
construction, and evaluate existing inflationary diagnostics. Keep this
exploratory probe separate from the validated continuation claim.

## Verification requirements

Verification SHALL be progressive and evidence-oriented:

1. analytic/source-controlled N=5 fixture;
2. named published N=8 radial fixture;
3. bounded continuation and failure-boundary replay;
4. independent augmented-solve comparison;
5. target-constructed arbitrary-precision stability checks where required;
6. actual post-hoc matcher comparison where branch-assignment equivalence is
   claimed;
7. geometric/Kahler/EFT checks before and during non-radial G3 work;
8. focused regression tests plus broader package/audit/docs/CI checks required
   by the changed layer;
9. fresh independent scientific review before scientific gate acceptance where
   required.

A test count alone is not scientific acceptance. Evidence must exercise the
claimed mechanism and use like-for-like source/model/metric conventions.

## Interfaces and compatibility

- G1 contains an additive N=5 continuation capability and correction of the
  defective N=5 benchmark fixture/validation behavior.
- No intentional breaking package API change is part of this specification.
- Preserve existing public and persisted contracts unless an explicitly
  reviewed change becomes necessary.
- No persisted scientific schema change is currently authorized.
- Preserve Julia 1.12 project support and intentional numerical precision.
- Feature branches state version impact but do not bump `Project.toml`; any
  reviewed release bump belongs at the normal `vmm -> main` integration
  boundary.
- Source scientific artifact identities/versions remain distinct from package
  versioning.

## Dependencies and blockers

- Scientific G3 is blocked by explicit acceptance of scientific G2.
- At the migration cutoff, G2 has a repair candidate awaiting fresh independent
  scientific review; the earlier FAIL remains the controlling gate decision
  until superseded by a durable later adjudication.
- Generic adaptive Kahler-cone exploration is a separate follow-on goal and is
  not a dependency for this pilot.

Use GitHub Issue/Project relationships for live blocking/current-state tracking.

## Open owner decisions

At the migration cutoff, the earlier N=8 metric-normalization boundary has been
resolved by the P96 approval. No new owner decision is required merely to
review the current G2 repair candidate.

If a scientifically valid G2 repair still leaves catastrophe classification or
another normative interpretation ambiguous, return the smallest remaining
question to the scientific owner before changing a cutoff, normalization,
claim, or gate criterion.

## Completion criterion

This Goal is complete when G0-G3 have passed with durable evidence and
independent scientific review, or when a documented failed gate establishes
that the proposed continuation route is not currently viable.

Generic adaptive Kahler-cone exploration is a separate follow-on Goal.
