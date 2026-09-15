---
spec_id: CYAX-0170
title: Freeze the P0 numerical-equivalence and performance baseline
issue: 170
class: S2
status: draft
workstream: numerical architecture
parent: null
depends_on: []
created: 2026-09-15
last_reviewed: null
review_required: project-owner and independent numerical reviewer
approval_ref: N/A while draft; owner dispatch is recorded in Issue #170 but durable normative approval remains pending
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
results remain to be recorded by P0 evidence. Source observations do not become
normative scientific interpretations merely because they are historical.

### Existing empirical evidence

Existing tests and `validation/` artifacts may be cited after their exact source,
fixture, environment, and applicability are verified. Expected output in source
comments is not executed evidence.

### Owner-approved conventions

The owner dispatched P0 execution on 2026-09-15 with the scientific reference
revision and phase boundary above. This draft does not claim durable approval of
its normative contract until an explicit approval reference is recorded.

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

**Acceptance:** This specification records an explicit durable approval reference.

**Stop condition:** Consequential scientific choices remain draft; historical
evidence may be gathered, but unresolved intent must not be guessed.

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

- Durable CYAX-0170 G0 approval is required before this draft becomes normative.
- Optional Python/CYTools or external datasets may limit applicable measurements;
  such limits are evidence, not permission to make them mandatory.
- Issues #113, #124, and #127 retain their scientific ownership boundaries.

## Open owner decisions

1. Approve, amend, or reject this draft as the durable CYAX-0170 P0 contract.
2. After CYAX-0170 G2, decide whether to authorize P1/P2, revise P0, or reopen
   design.

## Completion criterion

P0 is complete only when the independently reviewed exact candidate provides a
reproducible, versioned description of current numerical and compatibility
behavior sufficient to judge later implementations, and the Manager returns it
without crossing the P0 boundary.
