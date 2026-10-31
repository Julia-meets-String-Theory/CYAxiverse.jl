---
spec_id: CYAX-0215
title: Legacy minimizer acceptance criteria and batch vacua default
issue: 218
pr: 215
class: S2
status: draft
workstream: legacy vacua / minima counting
parent: null
depends_on: []
created: 2026-10-31
last_reviewed: null
review_required: repository-owner approval after independent Scientific/Numerical Review
approval_ref: N/A while draft
---

# Legacy minimizer acceptance criteria and batch vacua default

## Objective

Record, as an S2 contract, the scientific decisions made in PR #215: which
numerical endpoints of the legacy minimizer count as minima, when the origin is
counted, how two legacy vacua branches report their counts, and which search
method batch vacua runs use by default.

## Status of this record (brownfield)

PR #215 began as a test-suite repair that follows an internal test-suite audit
of `vmm` @ `f12fbb8`. It was first classified as PR-only S0. Review found that
it also changes scientific acceptance criteria and a batch default, which are
S2 decisions under `AGENTS.md` and the `cyaxiverse-sdd` skill.

This specification was written after the implementation. Its requirement IDs
did not govern the execution of PR #215. They record the decisions as they
stand at the implementation candidate (see *Candidate identity*), so that the
owner can approve, amend or reject each one before merge. If approval changes
a decision, the implementation changes before merge and the evidence is
re-run.

Issue #218 is the governing work item. This specification is the normative
scientific contract. PR #215 holds the implementation and its evidence.

## Scope

This specification governs only these changes in PR #215:

- the objective and Hessian that `minimize` passes to Newton's method;
- the stopping tolerances and the `"stationary"` acceptance criterion of
  `minimize`;
- the endpoint filter and the origin rule of `subspace_minimize`;
- the zero-column `α` case of `vacua_id_basis`;
- the integral count stored by `_legacy_vacua_search`;
- rejection of non-finite input by `spectrum_mode_counts`;
- the default `--method` of `scripts/batch_vacua_pipeline.jl`.

The rest of PR #215 (test-suite structure, Python test portability, CI
triggers, documentation layout, test runtime) is S0/S1 engineering. It does not
change scientific meaning and is not governed here.

## Non-scope

This specification does not cover or authorize:

- legacy vacuum de-duplication in `vacua_id` and `vacua_full` (#216);
- the legacy subspace `phase` contract (#217);
- known issues A and B recorded in PR #215 (`@test_broken` cases);
- any change to the single-geometry API default (`:legacy`);
- any change to the reduced-JLM, leading-branch or exact-determinant searches;
- recomputation of persisted results (an operational follow-up outside the
  repository);
- promotion from `vmm` to `main`, version adoption, tags or publication.

## Fixed conventions (unchanged by this work)

- Potential: `V(x) = Σ_a Λ_a (1 - cos(Q_a · x))`, with `x` in radians, `Λ_a`
  the linear instanton scales (`LV`) and `Q` the fields × instantons charge
  matrix (`QV`).
- `minimize` reports `"xmin"` as a fraction of 2π, reduced mod 1. Components
  within `eps()/1e-2` of 0 or 1 are set to 0.
- `subspace_minimize` returns exact-unique endpoints. It does not merge
  approximate duplicates.

## Current baseline

### Source facts

- At `f12fbb8`, the objective and Hessian in `minimize(LV::Vector, QV, x0::Vector)`
  broadcast the 1×N row `cos.(x' * QV)` against `LV` as an N×N matrix. The
  gradient was correct. Newton's method therefore received an objective and
  Hessian that were not those of `V`.
- At `f12fbb8`, the step tolerance was `x_tol = minimum(abs.(LV))` and the
  gradient tolerance was the absolute `g_tol = eps()/1e-2`.
- At `f12fbb8`, `subspace_minimize` kept every Newton endpoint and always added
  the origin. `batch_vacua_pipeline.jl` defaulted to `--method legacy`.

### Owner direction recorded in PR #215

- The strict origin rule (R-005) was chosen by the owner after review, as
  recorded in the PR #215 description.
- The other decisions below were made during PR #215 implementation and review.
  They become approved only through this specification's `approval_ref`.

## Requirements

### R-001 — Exact objective and Hessian

`minimize` SHALL pass Newton's method `V`, its gradient and its exact Hessian
`H_ij = Σ_a Λ_a Q_ia Q_ja cos(Q_a · x)`. The Hessian buffer SHALL be zeroed
before each evaluation. There is no alternative convention to choose: this is
the derivative of the fixed potential.

### R-002 — Stopping tolerances

`minimize` SHALL stop with `x_abstol = sqrt(eps())` (radians) and
`g_tol = (eps()/1e-2) · max_i s_i`, where `s_i = Σ_a |Λ_a| |Q_ia|` bounds
`|∂_i V|` everywhere. If every `s_i` is zero (constant `V`), `g_tol = 0`.

Basis: `sqrt(eps())` is a chosen Float64 stopping scale. It is motivated by
local numerical behaviour: once a step is about `sqrt(eps())`, the change in the
objective is O(eps), and Newton's quadratic convergence leaves an error near
eps after such a step. It is not a universal smallest resolvable step, and not
a certified bound on position error. `eps()/1e-2` is the existing constant, now
made relative so the run does not change when all `Λ_a` are rescaled together.

### R-003 — Stationarity criterion

`minimize` SHALL report `"stationary" = true` only when all of these hold:

1. `max_i s_i > 0`;
2. `|∂_i V| ≤ (eps()/1e-2) · s_i` for every component `i`;
3. the Hessian at the endpoint is nonsingular and the Newton step satisfies
   `‖H⁻¹ ∇V‖_∞ ≤ sqrt(eps())`, computed in the Hessian eigenbasis.

Basis: (2) is 100 times above the roundoff of a sum of terms bounded by
`|Λ_a Q_ia|`. (3) is the step tolerance of R-002. It is needed because (2) is
not basis invariant: a weak direction that mixes the fields can have small
gradient components while the point is displaced along it.

`"converged"` SHALL report Optim's convergence flag without change.

### R-004 — Endpoint acceptance in `subspace_minimize`

`subspace_minimize` SHALL keep a Newton endpoint only if it is converged,
stationary (R-003) and has `Hsign == 1` (smallest Hessian eigenvalue strictly
positive). Saddles, maxima, unresolved endpoints and non-converged runs SHALL be
discarded.

### R-005 — Origin rule

`subspace_minimize` SHALL add the origin only when the smallest eigenvalue of
`Q · Diagonal(L) · Qᵀ` is strictly positive. A degenerate origin (a zero
eigenvalue) SHALL NOT be returned. This includes rank-deficient flat
directions, even where the origin is a non-strict minimum.

### R-006 — Zero-column `α` in `vacua_id_basis`

When `α` has no columns (no dependent charges), `vacua_id_basis` SHALL take the
trivial branch and return `|det(Qtilde)|` vacua, the same as the existing
all-zero placeholder case. It previously raised `BoundsError`.

### R-007 — Integral legacy count

`_legacy_vacua_search` SHALL store the `vacua_id` count as `Int`, and SHALL
raise an error on a non-integral count instead of rounding.

### R-008 — Non-finite spectrum input

`spectrum_mode_counts` SHALL raise `ArgumentError` when any eigenvalue is not
finite, instead of classifying it.

### R-009 — Batch default method

`scripts/batch_vacua_pipeline.jl` SHALL default to `--method auto`. Its help
text SHALL state the previous default and how to match or recompute earlier
results. The single-geometry API (`compute_vacua_data`, `compute_axion_data`
and the positional CLI) SHALL keep `:legacy`.

`auto` follows the existing decision tree (exact determinant → selected branch
set → reduced JLM → legacy). The selected method is recorded in
`search_method` and `auto_selected_method`. Because the method is part of the
configuration digest, results from an earlier default run report a
configuration mismatch instead of being reused.

### R-010 — Records

The scientific behaviour change SHALL be documented in the `minimize` and
`subspace_minimize` docstrings, the batch help text and the PR #215
description. Legacy vacua results produced before PR #215 SHALL be treated as
stale and recomputed, not compared with new counts.

## Scientific claim boundary

### This specification may establish

- that `minimize` iterates on `V` and its exact derivatives;
- that accepted endpoints satisfy the stated numerical criteria, which are
  local convergence criteria;
- that the origin is counted only where the second-order test classifies it as
  a strict minimum;
- that the fixtures in *Verification* behave as derived by hand.

### This specification must not establish

- that an accepted endpoint is a certified minimum, or a bound on its distance
  from an exact minimum;
- that a legacy count is complete or a rigorous lower bound. Legacy results
  retain the operational classification `finite_multistart_minimum_lower_bound`,
  but this specification does not establish that reported counts are rigorous
  lower bounds, particularly while the known legacy counting defects in #216
  and #217 remain unresolved;
- that Float64 eigenvalues resolve the sign of strongly hierarchical directions
  (more than about 16 decades in a mixing basis);
- that `auto` and `legacy` counts agree, or that results from different
  `auto_selected_method` values are the same counting unit;
- any correction to de-duplication (#216) or phases (#217).

## Acceptance gates

### G0 — Specification approval

- An independent Scientific/Numerical Reviewer reviews this specification
  against the implementation candidate. The review record names the reviewer
  role and the exact draft revision reviewed (the commit SHA at which
  `spec.md` was reviewed).
- Blocking findings are repaired, and the changed revision is re-reviewed.
- The owner approves that exact reviewed revision in a durable GitHub
  comment on PR #215 that names its commit SHA. That comment is the approval
  event.

Stop condition: the owner amends a decision. The implementation and evidence
then change before G1, and the implementation candidate is updated.

### Approval recording

A later commit sets `status: approved`, `last_reviewed` and `approval_ref`
(the approval comment from G0). It changes only that metadata. It changes no
requirement and no implementation. The approval refers to the reviewed draft
revision, not to this commit, so the reference is not circular.

### G1 — Implementation and evidence reconciliation

- R-001 to R-010 have the evidence listed under *Verification* on the
  implementation candidate. Each regression test was confirmed to fail against
  a deliberately broken version of the code it guards.
- The diff from the implementation candidate to the final merge candidate
  touches only `specs/0215-minimizer-acceptance-contract/`, and the only change
  to `spec.md` after the reviewed revision is the approval metadata.
- CI passes on the final merge candidate.

### G2 — Independent review of the final merge candidate

Independent review of the entire final merge candidate has no blocking
finding. This covers the whole PR #215 diff against `vmm`, including the
test-suite restructuring, CI qualification and retired entrypoints, not only
R-001 to R-010. This specification governs the scientific behaviour; the
engineering changes are reviewed against their own evidence in the PR #215
description.

A passing review is evidence for the owner's merge decision; it is not merge
authority. The merge target is `vmm`.

## Verification

| Requirement | Evidence (`test/runtests.jl` unless stated) |
| --- | --- |
| R-001 | End-to-end minimum: position, gradient and `V` within `sqrt(eps())`-derived bounds; phase-Hessian fixture |
| R-002 | `V = 1e-20 (1 - cos x)` from `x = 0.3` reaches the minimum at 0 instead of stopping at the start; no Optim deprecation warnings |
| R-003 | Hierarchical `L = (1, 1e-20)`; constant potential; mixed weak direction `Q = [1 1; 1 -1]`, `L = (1, 1e-12)`, start `(0.01, -0.01)`, with phases within the hand-derived `2 sqrt(eps())` |
| R-004 | Saddle and maximum endpoints rejected; mixed-sign saddle `Q = [1 0 1; 0 1 1]`, `L = (1, 1, -0.9)` with minima at `x1 = x2 = ±acos(5/9)` |
| R-005 | `-2 sin⁴(x/2)` maximum (`L = (-1, 1/4)`, `Q = [1 2]`), `V ≡ 0`, and rank-deficient `Q = [1; 0]` exclude the origin |
| R-006 | Non-unimodular `Qtilde = [2 0; 0 1]` gives 2 vacua |
| R-007 | Integration evidence only: the legacy batch run on geometry `2,2,1` completes and writes its result (`test/full.jl`). Not directly verified: no test asserts that the stored count is an `Int`, or that a non-integral count raises (Julia's `Int` conversion raises `InexactError`) |
| R-008 | `spectrum_mode_counts([NaN, 1.0])` raises `ArgumentError` |
| R-009 | Parsed default is `:auto` and help-text assertion (`test/full.jl`). The single-geometry `:legacy` default has no source change in PR #215 |
| R-010 | Docstrings and help text present; docs build passes |

Suites: fast and full Julia suites, Python tests, scientific qualification and
documentation, locally and in CI, on the implementation candidate. Results are recorded in the
PR #215 description. Unobserved checks are not PASS.

## Candidate identity

Two identities are tracked:

- **Implementation candidate:** `ab48726` on `codex/test-suite-repairs`. The
  scientific and test evidence in the PR #215 description was produced on it.
- **Final merge candidate:** the PR #215 head after the approval-recording
  commit. G1 checks that it differs from the implementation candidate only in
  `specs/0215-minimizer-acceptance-contract/`. G2 reviews it, and CI must pass
  on it before merge.

## Interfaces and compatibility

- `minimize` results gain the `"converged"` and `"stationary"` keys.
- `subspace_minimize` gains the `initial_points` keyword.
- The batch default changes as in R-009.
- Persisted values: R-007 normalizes the legacy count to `Int` before it is
  written to `vacua_pipeline/verified`, so the stored numeric type can differ
  from earlier legacy results. Dataset paths and the overall HDF5 structure are
  unchanged. The datatype of historically stored legacy counts has not been
  verified.
- Package-version impact: patch, no version bump.

## Open owner decisions

None are intentionally left open. Approval of this specification is the owner's
decision on R-001 to R-010 as written.
