---
spec_id: CYAX-0170
title: Freeze the P0 numerical-equivalence and performance baseline
issue: 170
class: S2
status: approved
workstream: numerical architecture
parent: null
depends_on: []
created: 2026-09-15
last_reviewed: 2026-09-15
review_required: project-owner and independent numerical reviewer
approval_ref: https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/170#issuecomment-5685945127
---

# Freeze the P0 numerical-equivalence and performance baseline

## Objective

Produce a reviewed, reproducible description of what CYAxiverse computes and
exposes at scientific reference revision
`7a40285bb5c313f7e8746b90644d5f45bb67be44`. The result must be precise enough
to judge a later numerical architecture by historical equivalence rather than
by plausible output.

## Motivation

Later modularisation and performance work can change floating-point behavior,
dispatch domains, compatibility surfaces, term order, root discovery, and
classification without producing an obvious failure. P0 freezes those
boundaries before any such implementation begins.

## Current baseline

### Source facts

- The scientific reference is branch `vmm` at revision
  `7a40285bb5c313f7e8746b90644d5f45bb67be44`.
- Repository policy classifies a durable numerical-equivalence contract as S2.
- Issue #170 is the governing work item.

### Current implementation facts

Implementation behavior, API surfaces, environment identity, and benchmark
results are recorded in the frozen P0 evidence set. Source observations do not
become normative scientific interpretations merely because they are historical.
The independently reviewed evidence candidate is
`ddc304f25040ff36bc84e7897d5b6dc5d16344f6`; synchronization commit
`6f2acaef2a181937f63d8f085c4317e91a08267b` is mechanical bookkeeping only.

### Existing empirical evidence

Existing tests and `validation/` artifacts may be cited after their exact source,
fixture, environment, and applicability are verified. Expected output in source
comments is not executed evidence.

### Owner-approved conventions

The owner approved the P0 numerical-equivalence and compatibility baseline and
`cyaxiverse-numerical-equivalence-v1` in Issue #170 comment
`5685945127`, subject only to mechanical synchronization of the dispositions
below. The scientific reference remains
`7a40285bb5c313f7e8746b90644d5f45bb67be44`; no P1/P2/P3A/P3B authorization is
granted.

The approved route-specific dispositions are:

1. F13 preserves historical route differences: validator routes retain
   historical non-finite rejection, while direct critical-point non-finite
   propagation/failure remains a historical parity defect. P0 performs no
   harmonization.
2. The governed/source-faithful top-level N=5 route is authoritative for future
   modularisation and equivalence gates. Stale nested legacy N=5 behavior is
   historical only and is not the scientific oracle.
3. Float64 and high-precision B6 baselines remain separately preserved.
   Numerical/physical authority is unresolved; no future migration may silently
   choose one route as an oracle.
4. Observed `2pi^2 I` is retained as historical defect evidence, not intended
   behavior. The `4pi^2 I` correction is separate Issue #172 work. Issue #173
   remains separate.

These dispositions describe the approved P0 contract; they do not repair,
reinterpret, regenerate, or relabel the empirical evidence.

### Open inference

Whether current edge behavior is scientifically desirable is outside P0. P0 may
document a defect or undesirable result but must not repair it.

## Scope

- Exact source, environment, dependency, fixture, and benchmark identity.
- Separate warm/precompiled load and cold environment/build measurements.
- Compatibility, dispatch, persistence, and repository-consumer inventory.
- Historical numerical semantics and pathological fixtures F1-F13.
- Baselines B1-B7 where applicable.
- A versioned comparison contract and precommitted future acceptance gates.
- Independent numerical review of the exact integrated candidate.

## Non-scope

P0 does not authorize production `PotentialNumerics`; consumer migration;
derivative replacement; source decomposition; dependency changes; performance
refactoring; zero-term pruning; coefficient/log canonicalization; altered
reduction order; `@fastmath`; threading; BigFloat redesign; changes to
`critical_points`, inflation outputs, SLURM behavior, HDF5 schemas, or scientific
ownership governed by Issues #113, #124, or #127; or high-`h11` claims.

## Scientific claim boundary

### This specification may establish

- Historical source and implementation behavior at the pinned revision.
- Reproducible empirical measurements for identified fixtures/environments.
- A frozen compatibility and numerical comparison contract for later work.
- Evidence that historical behavior is defective or undesirable.

### This specification must not establish

- That historical behavior is the uniquely correct physical model.
- Population-level or high-`h11` conclusions.
- Scientific acceptance without the required independent review and durable
  owner approval.
- Permission to change or repair historical behavior.

## Fixed conventions and invariants

- Preserve coefficient factor `c_a` and stored base-10 log exponent `ell_a`
  separately; do not normalize them to sign plus adjusted logarithm.
- Map term order route by route from persistence through canonicalization to
  the scientific consumer. Raw HDF5 order is not assumed universally normative.
- Treat representation validity separately from policy admissibility for
  `c_a = 0` and `ell_a = -Inf`.
- Freeze stationarity support after the historical basis transformation,
  Float64 materialization, and exact transformed-zero test.
- Preserve the historical empty-support row rule and distinguish physical
  Hessian, row-scaled Newton Jacobian, and congruence-scaled classification
  matrix.
- Freeze seed-displacement and final-inertia thresholds separately.
- Preserve distinct argument-offset, coordinate-displacement, unit-torus, and
  legacy-radian evaluation orders.
- Do not claim concurrent thread safety for paths that alter global/default
  BigFloat precision without separate evidence.

## Requirements

### R-001 — Exact replay identity

P0 SHALL record repository revision/ref, Julia and build identity, resolved
dependencies and Manifest identity, CPU/architecture/OS, BLAS and thread state,
relevant environment, precompilation state, RNG seeds, fixture hashes, warm-up,
sample count, statistic, and allocation method.

### R-002 — Distinct package-load baselines

P0 SHALL report warmed/precompiled `using CYAxiverse` separately from cold
environment establishment and compile/precompile cost.

### R-003 — Compatibility inventory

P0 SHALL inventory exported and observable non-exported bindings, accepted
argument domains and methods, keyword defaults, return/status/failure behavior,
aliases/extensions, scripts/notebooks, conditional SLURM behavior, and persisted
HDF5 paths/orientation/units/schema. The named module surfaces in Issue #170 are
mandatory inventory targets.

### R-004 — Potential encoding and order boundaries

P0 SHALL freeze separate coefficient/log semantics, zero representations,
phase/coordinate/charge conventions, and route-specific raw versus
oriented/canonicalized term-order boundaries.

### R-005 — Zero and normalization semantics

P0 SHALL record producer/consumer behavior for `c_a = 0` and `ell_a = -Inf`,
including global physical normalization, stationarity row scaling, zero-only
support, all-`-Inf` support, and any resulting NaN/failure without repairing it.

### R-006 — Stationarity support and empty rows

P0 SHALL freeze transformed-Float64 exact-zero support and the historical
empty-support-row `row_logscale` rule.

### R-007 — Critical-point matrix and threshold distinctions

P0 SHALL distinguish H1 physical Hessian, H2 row-scaled Newton Jacobian, H3
congruence-scaled classification matrix, the meaning of returned
`hessian_eigenvalues`, the seed-displacement threshold, and final-inertia
threshold.

### R-008 — Argument and displacement semantics

P0 SHALL freeze argument offsets, coordinate displacement, unit-torus and
legacy-radian operation order without algebraically reordering floating-point
evaluation.

### R-009 — Fixture corpus

P0 SHALL provide F1-F13 with content identity/hash, construction provenance,
purpose, expected historical behavior, precision/environment, or an
evidence-backed reason that a fixture is impossible or inapplicable.

### R-010 — Physical and precision baseline

P0 SHALL record representative value, gradient, physical Hessian, mass or
generalized-Hessian quantities, epsilon, eta-related quantities, stationary
correction, gradient flow, and exit classification for current Float64 and
supported BigFloat routes.

### R-011 — Ownership and dispatch acceptance contract

P0 SHALL define future borrowed-versus-copied workspace result lifetimes and a
P2 Julia gate covering concrete hot-path objects, kernel selection before hot
loops, function barriers, inference/JET, ambiguity/Aqua checks, and avoidance of
unbounded runtime-value specialization.

### R-012 — Performance baseline

P0 SHALL record B1-B7 where applicable: load; dense derivatives; structured
evaluation; `critical_points`; inflation flow; stationary correction;
representative spectra; and geometry/filesystem behavior. Unavailable or
inapplicable measurements SHALL be explicit.

### R-013 — Versioned equivalence contract

P0 SHALL version comparison semantics for stored input, argument realization,
physical derivatives, stationarity scaling/vector/Jacobian/classification,
seed/final modes, periodic root-set matching, residuals/failures, precision,
inflation/correction behavior, and protected spectra.

### R-014 — Precommitted tolerances

Comparison tolerances SHALL be derived before later implementation output from
historical/scientific tolerances, repeated historical behavior, analytic
fixtures, and high-precision diagnostics where appropriate. Higher precision
is not automatically the historical-parity oracle.

### R-015 — Evidence identity and publication safety

Every substantive result and review SHALL identify its exact source, candidate,
fixture, environment, and hashes as applicable, while durable files use
repository-relative/public-safe provenance rather than machine-local paths.

### R-016 — Independent review and phase stop

A reviewer who did not define the contract or fixtures SHALL review the exact
integrated candidate against R-001-R-015. After that verdict the Manager SHALL
return the P0 checkpoint and SHALL NOT continue into P1, P2, P3A, or P3B without
a new owner decision.

## Acceptance gates

### CYAX-0170 G0 — Governing contract approval

**Objective:** Establish durable owner-approved P0 intent and review provenance.

**Acceptance:** Met. This specification records the owner approval in Issue #170
comment `5685945127`.

**Stop condition:** Any consequential scientific choice outside the four approved
dispositions remains owner-controlled; unresolved intent must not be guessed.

### CYAX-0170 G1 — Evidence candidate

**Objective:** Produce reconstructible identity, inventories, F1-F13, B1-B7,
and the versioned equivalence contract without production changes.

**Acceptance:** R-001-R-015 have mapped artifacts and executed evidence or an
explicit, evidence-backed inapplicability/gap.

**Stop condition:** Stop for the owner on any ambiguity named in Issue #170 or
if evidence would require a production algorithm/schema/API change.

### CYAX-0170 G2 — Independent numerical review

**Objective:** Review the exact candidate revision independently.

**Acceptance:** A durable verdict identifies the candidate revision, evidence,
coverage, tolerance defensibility, unresolved findings, and clear PASS or revise.

**Stop condition:** Any material contract/evidence change after review requires
review of the changed state.

### CYAX-0170 G3 — Manager return

**Objective:** Return the reviewed P0 checkpoint for owner decision.

**Acceptance:** The return identifies evidence revision, Issue/spec status,
topology, environment, inventories, benchmarks, fixtures, contract version,
reviewer/verdict, ambiguities/deviations, and a recommendation to authorize later
work, revise P0, or reopen design.

**Stop condition:** No later phase begins automatically.

## Verification requirements

Verification proceeds through analytic/synthetic fixtures, named governed
fixtures, bounded replay, applicable package/audit/import checks, exact-diff
review, and independent numerical review. Every report gives commands actually
run and observed outcomes; expected output is never presented as execution.

## Interfaces and compatibility

P0 changes no public API, dependency, persisted schema, source data, numerical
algorithm, or package version. It records current interfaces and creates
validation/specification artifacts only.

## Dependencies and blockers

- The exact synchronized candidate requires the independent normative-fidelity
  review authorized by the P0 handoff; that review does not authorize merge.
- Optional Python/CYTools or external datasets may limit applicable measurements;
  such limits are evidence, not permission to make them mandatory.
- Issues #113, #124, and #127 retain their scientific ownership boundaries.
- Issue #172 owns the separate `4pi^2 I` correction. Issue #173 owns any
  numerical/physical authority decision for the B6 routes.

## Open owner decisions

1. After the exact synchronization candidate receives normative-fidelity review,
   decide whether to merge PR #171.
2. Separately decide whether to authorize P1/P2/P3A/P3B; this P0 approval grants
   no such authorization.

## Completion criterion

P0 is complete only when the independently reviewed exact candidate provides a
reproducible, versioned description of current numerical and compatibility
behavior sufficient to judge later implementations, and the Manager returns it
without crossing the P0 boundary.
