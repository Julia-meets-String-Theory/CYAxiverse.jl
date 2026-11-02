---
spec_id: CYAX-0215
title: Legacy minimizer acceptance criteria and batch vacua default
issue: 218
class: S2
status: draft
workstream: legacy vacua / minima counting
parent: null
depends_on: []
created: 2026-10-31
last_reviewed: null
review_required: repository-owner approval after independent Scientific/Numerical Review
approval_ref: N/A while draft
version_impact: patch, no version bump
---

# Legacy minimizer acceptance criteria and batch vacua default

## Objective

Record, as an S2 contract, the scientific decisions made in PR #215: which
numerical endpoints of the legacy minimizer pass its acceptance filter, when
the origin is added, how two legacy vacua branches report their counts, and
which search method batch vacua runs use by default.

## Motivation

At `f12fbb8`, the legacy minimizer passed Newton's method an objective and a
Hessian that were not those of the potential, used a step tolerance tied to the
smallest instanton scale, and returned every Newton endpoint and the origin.
Direct calls of `minimize` and `subspace_minimize` therefore had no defined
acceptance criterion. PR #215 repairs the objective and Hessian and states a
numerical acceptance filter. This specification records that filter, its known
limitations, and what it does not establish, so that the owner can approve it
before merge. The filter is a heuristic. It is not a certificate of a minimum.

The legacy vacua pipeline never reached the minimizer (see *Current
baseline*), so these repairs do not change any stored legacy count. PR #215
also fixes an error that stopped some legacy geometries from completing
(R-006), and makes the count that those geometries now reach writable (R-007). A proof-backed classifier that replaces R-003 to R-005 is proposed
separately in #220.

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

The identifier `CYAX-0215` follows PR #215, not the governing Issue #218,
because the Issue was opened after the specification directory had been named
and cited. This is a deliberate exception to the convention that the
identifier matches the Issue number.

### Review and amendment history

- Draft `444fcc3` (implementation candidate `ab48726`): independent review did
  not pass. Its blocking findings concerned the stale-result wording and the
  Float64 origin rule.
- Draft `6b033b5` (implementation candidate `dc42633`): independent review did
  not pass. Its blocking findings were:
  1. The draft and the help text called legacy results stored before PR #215
     stale, but the legacy pipeline never reaches the minimizer.
  2. R-003 condition (2) rejects genuine minima at any phase, not only at large
     phases, and the draft did not say so.
- Owner scope amendment, made after the review of `6b033b5`:
  1. Legacy counts stored before PR #215 are not stale because of the minimizer
     defect, and are retained (R-009, R-010).
  2. Condition (2) is retained in PR #215 as a documented legacy limitation.
     PR #215 does not claim complete or certified classification of stationary
     points; that is deferred to #220 (R-003).

  These supersede the earlier R-003, R-009 and R-010 wording. Item 2 rested
  on the incorrect premise that condition (2) predated PR #215. It is
  superseded (see below).
- Draft `0442ead` (implementation candidate `b309a55`): independent review did
  not pass. Its blocking findings were:
  1. R-004 said that rank-deficient charges usually give an empty result. They
     can give many points from one flat family (R-004).
  2. R-003 called condition (2) an existing legacy heuristic. It was
     introduced in PR #215 (`5316fc0`). `minimize` at `f12fbb8` had no
     stationarity filter, and `subspace_minimize` kept every endpoint.
- Owner decisions made after the review of `0442ead`, on the corrected facts:
  1. Condition (2) is removed in PR #215 (`a508769`). This supersedes item 2
     of the earlier amendment. The Newton-step and Hessian conditions stay. The
     acceptance is still described as heuristic, and #220 still owns validated
     nearby-root certification, curvature classification and explicit
     unresolved outcomes (R-003).
  2. Over-counting of flat families for rank-deficient charges is accepted as a
     documented limitation of PR #215. #220 must establish exact structural-rank
     handling and isolated-minimum count eligibility before #217 reconnects the
     affected pipeline. This is a classification and counting-unit defect, not
     a de-duplication defect (#216) (R-004).

- Draft `92f4c71` (implementation candidate `dbd7b62`): independent review did
  not pass. Its blocking findings were text only:
  1. *Current baseline* said that `Rational` counts from the `|det|` branch
     failed to save at `f12fbb8`. That branch was unreachable there; R-006
     makes it reachable, and R-007 makes its count writable.
  2. The proposed R-009 rationale named the `TypeError` of #217 as the reason
     legacy cannot complete. The `BoundsError` at the `Leff` mask or the
     column strip comes first, by construction.

The failed reviews are not relabelled as passed. The records of these four
reviews are not published on GitHub; this list is their only record in the
repository. The amendments become durable when the owner approves the revision
that contains them at G0.

## Scope

This specification governs only these changes in PR #215:

- the objective and Hessian that `minimize` passes to Newton's method, and the
  shared Hessian helper `_phase_hessian!`, which also changes the Hessian of
  `id_minimize(::Vector, ::Matrix)`;
- the stopping tolerances and the `"stationary"` acceptance filter of
  `minimize`, including the removal of condition (2);
- the endpoint filter and the origin rule of `subspace_minimize`;
- the zero-column `α` case of `vacua_id_basis`;
- the integral count stored by `_legacy_vacua_search`;
- rejection of non-finite input by `spectrum_mode_counts`;
- the default `--method` of `scripts/batch_vacua_pipeline.jl` and its help
  text.

The rest of PR #215 is S0/S1 engineering and is not governed here: test-suite
structure, Python test portability, CI triggers, documentation layout, test
runtime, and the retirement of the 2026-08-25 physical-scale entrypoints
(`bd6807e`, including `scripts/audit_physical_certificate.jl` and the
physical-scale inflation pilot). It does not change scientific meaning. The
retired scripts stay available at earlier revisions, and their removal does
not affect the replay of any evidence here.

## Non-scope

This specification does not cover or authorize:

- legacy vacuum de-duplication in `vacua_id` and `vacua_full` (#216);
- the legacy subspace `phase` contract, including the unreachable minimizer
  call in `vacua_id` and `vacua_full` (#217);
- a proof-backed classification of minima, non-minima and unresolved critical
  points, structural-rank handling, and isolated-minimum count eligibility
  (#220);
- `id_minima` and `id_minimize(::Vector, ::Vector)`. Both are non-functional
  before and after PR #215: they return `nothing` or raise (`AssertionError`,
  or `MethodError` in `denominator` or in the shared Hessian helper, depending
  on the input);
- known issues A and B (see *Known issues*);
- any change to the single-geometry API default (`:legacy`);
- any change to the reduced-JLM, leading-branch or exact-determinant searches;
- recomputation of persisted results;
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
  objective was `(Σ_a Λ_a) · Σ_b (1 - cos θ_b)`. The gradient was correct.
  Newton's method therefore received an objective and Hessian that were not
  those of `V`.
- At `f12fbb8`, the step tolerance was `x_tol = minimum(abs.(LV))` and the
  gradient tolerance was the absolute `g_tol = eps()/1e-2`.
- At `f12fbb8`, `subspace_minimize` kept every Newton endpoint and always added
  the origin. `batch_vacua_pipeline.jl` defaulted to `--method legacy`.

### Reachability of the minimizer from the legacy pipeline

- The only route from `_legacy_vacua_search` (`scripts/vacua_pipeline.jl`) to
  the minimizer is `vacua_id(L, Q)`. It also calls `vacua_estimate`, which
  does not minimize. The only minimizer call in `vacua_id`
  (`src/generate.jl`) is `subspace_minimize(...; phase=phase[i])`. It passes a
  scalar to the keyword `phase`, so it would raise `TypeError` if it were
  reached. `phase=phase[i]` has been in the call since `7b72fb3` (2022-10-24);
  `6c46788` (2026-08-04) added `runs=runs`. The keyword was untyped at
  `7b72fb3`. From `f6c71e5` (2022-11-02) until `1bd2cd1` (2023-06-07) several
  keyword-only methods coexisted, and the last defined, `phase::Vector`, was
  active; since `1bd2cd1` the only method takes `phase::Matrix`. Both typed
  forms reject a scalar. The pipeline was created on 2026-07-30 (`9fce046`).
- The call is not reached, by construction. In the nonzero-`α` branch,
  `vacua_id_basis` strips one more column from `αeff` than `LQtildebar`
  appended to `Lhat` (the extra strip of known issue A). So `Qeff` always has
  one column fewer than `Leff`, and the `Leff` column mask in `vacua_id`
  raises `BoundsError` on every row that would need the minimizer. With a
  single dependent column, `vacua_id_basis` itself raises `BoundsError`
  (known issue A). Examples: `Q = [2 0 1 1; 0 1 1 0]`,
  `L = [1 1 1 1; 0 -0.1 -0.2 -0.3]` fails at the mask; the known issue A case
  fails inside `vacua_id_basis`.
- The minimizer has therefore never run inside a legacy pipeline count. Stored
  legacy counts come only from the branches that do not minimize. The extra
  `αeff` strip, the `Leff` mask, the `phase=phase[i]` call and the single
  `phase::Matrix` method are present at every revision in
  `git rev-list 9fce046^..f12fbb8` (607 commits), so this holds for the whole
  life of the pipeline before PR #215.
- At `f12fbb8`, the trivial `|det|` branch of `vacua_id_basis` was
  unreachable. `LQtildebar` returns `αeff[:, 2:end]`, whose columns are all
  nonzero, so the test `αeff == zeros(h11, 1)` never held. A zero-column `α`
  fell through to the nonzero branch and raised `BoundsError`. Such
  geometries failed to complete, and no verified count was produced for them.
  Every legacy count that did complete was an `Int`.
- R-006 makes the `|det|` branch reachable for the first time. It returns a
  `Rational` count, which HDF5 cannot write (`MethodError`); R-007 converts it
  to `Int`. Reaching this branch also exposes known issue B.

### Known issues

These are recorded as `@test_broken` cases in `test/runtests.jl`, with
hand-derived expectations. PR #215 does not fix them.

- **A.** The nonzero-`α` branch of `vacua_id_basis` strips a placeholder
  column that `LQtildebar` already removed. With one dependent instanton it
  raises `BoundsError` (`Q = [2 0 1; 0 1 1]`, `L = [1 1 1; 0 -1 -1.1]`,
  expected `Qeff = [1 0 0; 0 1 1]`).
- **B.** For `Q = [2 0 1; 0 1 1]` with the third instanton below threshold, the
  trivial branch of `vacua_id` reports 2 vacua but lists only θ/2π = (0, 0),
  not (1/2, 0).

An earlier candidate recorded the false rejection by condition (2) as a
`@test_broken` known issue. That condition is removed (R-003), and its test is
now an ordinary test.

### Owner direction

- The strict origin rule (R-005) was chosen by the owner after review, as
  recorded in the PR #215 description.
- The amendments under *Review and amendment history*: stored legacy counts
  retained; condition (2) removed; flat-family over-counting accepted as a
  limitation deferred to #220. Until G0 they are recorded only in this
  specification, not on PR #215.
- The other decisions below were made during PR #215 implementation and review.
  They become approved only through this specification's `approval_ref`.

## Requirements

### R-001 — Exact objective and Hessian

`minimize` SHALL pass Newton's method `V`, its gradient and its exact Hessian
`H_ij = Σ_a Λ_a Q_ia Q_ja cos(Q_a · x)`. The Hessian buffer SHALL be zeroed
before each evaluation. There is no alternative convention to choose: this is
the derivative of the fixed potential.

The same Hessian helper serves `id_minimize(::Vector, ::Matrix)`, where it is
the `nlsolve` Jacobian and enters its eigenvalue sign test. Its Hessian changes
in the same way. Its own acceptance rule is unchanged.

### R-002 — Stopping tolerances

`minimize` SHALL stop with `x_abstol = sqrt(eps())` (radians) and
`g_tol = (eps()/1e-2) · max_i s_i`, where `s_i = Σ_a |Λ_a| |Q_ia|` bounds
`|∂_i V|` everywhere. If every `s_i` is zero (constant `V`), `g_tol = 0`.

Basis: `sqrt(eps())` is a chosen Float64 stopping scale. It is motivated by
local numerical behaviour: once a step is about `sqrt(eps())`, the change in the
objective is O(eps), and Newton's quadratic convergence leaves an error near
eps after such a step. It is not a universal smallest resolvable step, and not
a certified bound on position error. `eps()/1e-2` is the existing constant, now
made relative so that the run is invariant under a common rescaling of all
`Λ_a` in exact arithmetic. In Float64 the run is bitwise unchanged for
power-of-two factors, provided no value overflows or underflows (measured for
two factors on the fixtures below). For other factors rounding changes the run, and this can
change the accept/reject decision of R-003 and R-004.

- **Measured** (at `dbd7b62`): four fixtures (`Q = I` with `L = (1, 1e-6)`;
  `Q = [1 0 1; 0 1 1]` with `L = (1, 1, -0.9)`; `Q = [1 1; 1 -1]` with
  `L = (1, 1e-12)`; `Q = [1 0 2 1; 0 1 1 -1; 1 1 0 2]` with
  `L = (1.3, 0.7, 0.2, 0.05)`), 10 starts each, `2π·rand` from
  `MersenneTwister(7)` in that order. Factors `2^-40` and `2^30`: in 80 of 80
  runs `"xmin"`, `"stationary"`, `"converged"` and `"Hsign"` were bitwise
  identical (`"logV"`, `"Heigs"` and `"gradlog"` shift with the factor by
  design). Factor 3: `"xmin"` or `"stationary"` differed in 17 of 40
  runs, and `_is_converged_minimum` changed in 2 of 40.

### R-003 — Stationarity filter

`minimize` SHALL report `"stationary" = true` only when both of these hold:

- **(1)** `max_i s_i > 0`;
- **(3)** no computed eigenvalue of the Hessian at the endpoint is exactly 0,
  and the Newton step satisfies `‖H⁻¹ ∇V‖_∞ ≤ sqrt(eps())`, computed in the
  Hessian eigenbasis.

The labels keep the numbering of earlier drafts; condition (2) is removed (see
below). `"converged"` SHALL report Optim's convergence flag without change.

**Meaning.** The `"stationary"` field reports whether a Newton endpoint
satisfies a heuristic numerical acceptance test in Float64. It is not a
certificate of an exact or nearby stationary point. `"stationary" = false`
means that the endpoint failed this test. It does not establish that no
minimum lies nearby. Validated nearby-root certification, curvature
classification and explicit unresolved outcomes are deferred to #220.

**Basis of (3).** Optim stops on a gradient below `g_tol`, which is relative to
the largest scale `max_i s_i` (R-002). Along a direction whose scale is far
below it (a strongly suppressed `Λ`, or a weak direction that mixes the
fields), the gradient is small away from the minimum too, so a small gradient
does not locate the point. The Newton step estimates the remaining
displacement in every direction, including directions that mix the fields,
and (3) requires it to be within the step tolerance of R-002. Optim can stop
before that: its gradient check runs before the first step, and its `f` test
fires once `1 - cos` underflows near a minimum.

**Condition (1).** It is implied by (3) for finite input. If every `s_i` is 0,
every product `Λ_a Q_ia` is 0, so the computed Hessian is exactly 0 and (3)
fails. It is kept as an explicit statement for the constant potential.

**Removed condition (2).** Earlier drafts (`444fcc3`, `6b033b5`, `0442ead`)
also required `|∂_i V| ≤ (eps()/1e-2) · s_i` for every component `i`. This
condition was introduced in PR #215 (`5316fc0`). It is not inherited from
`f12fbb8`, which had no stationarity filter. By owner decision it is removed
(`a508769`):

- **Mechanism (derived by hand).** For `Q = I`, condition (2) on component `i`
  reads `L_i |sin x_i| ≤ 100 eps() L_i`, that is `|x_i| ≤ 100 eps() ≈ 2.2e-14`
  rad, whatever `L_i` is. Condition (3) allows a displacement of about
  `sqrt(eps()) ≈ 1.5e-8` rad, and Optim can stop anywhere within it.
- **Measured.** `Q = I`, `L = (1, 1e-6)`, 400 starts uniform in `[0, 2π)²`
  from `MersenneTwister(4242)`. All 400 endpoints were converged, had
  `Hsign = 1`, and lay within `sqrt(eps())` rad of a minimum (displacement
  recovered as `|∂_i V| / L_i`, at most 4.9e-9 rad). With condition (2), at
  `b309a55`, 208 of them were rejected, one only 2.25e-14 rad away. At
  `dbd7b62` all 400 are accepted.
- **No compensating benefit was shown.** With (2) removed, no test changes
  result except the `L = (1, 1e-6)` regression, which then passes. The
  hierarchical case that motivated (2) is rejected by (3).

**Remaining limitations.** The filter can still miss or reject genuine minima.
Rounding of large phases `Q_a · x`, which grows with `Σ_i |Q_ia x_i|`, enters
the gradient and so the Newton step. The sign of the smallest eigenvalue is
taken in Float64 (`Hsign`, see R-005). A run can stop before it resolves a
direction about 16 decades below the largest scale (see *Scientific claim
boundary*). Rescaling can change the decision (R-002).

**Isolation.** The regression `L = (1, 1e-6)`, `x = (0, 1e-9)` fails with
condition (2) present. With (3) disabled, the hierarchical and mixed weak
direction cases fail (5 assertions). Condition (1) is not isolated, because
(3) implies it.

### R-004 — Endpoint acceptance in `subspace_minimize`

`subspace_minimize` SHALL keep a Newton endpoint only if it is converged,
passes the R-003 filter and has `Hsign == 1` (computed smallest Hessian
eigenvalue strictly positive). Saddles, maxima, endpoints that fail the filter
and non-converged runs SHALL be discarded. A discarded endpoint is not thereby
established to be a non-minimum (R-003).

**Limitation: flat families of rank-deficient charges (owner-accepted,
deferred to #220).** If the charges have rank `r < n` (`n` fields), every
critical point lies on an exactly flat family of dimension `n - r`, and the
exact Hessian always has an eigenvalue 0 (the smallest one at a minimum of the
family; saddles and maxima also have negative eigenvalues). Whether an endpoint is accepted then
depends on rounding:

- When the computed smallest eigenvalue rounds positive and the Newton step
  passes, endpoints at different points of the same flat family are each
  accepted, and `subspace_minimize` returns each as a separate exact-unique
  point. The number of points returned grows with the number of starts, so it
  does not count minima.
- When the computed eigenvalue is 0 or negative, the endpoints are rejected.
  An empty result does not mean that the potential has no minima. R-005
  rejects the origin under the same condition.

Measured at `dbd7b62`:

- `Q = [1 1; 2 2]`, `L = (0.7, 0.4)`, so `V = 1.1 (1 - cos(x1 + 2 x2))` with
  the flat direction `(2, -1)`. 40 starts `2π·rand(MersenneTwister(11), 2)`:
  39 endpoints accepted (computed `λ_min` about 4e-18), and 40 points returned:
  the 39 endpoints and the origin, which R-005 adds because its computed
  smallest eigenvalue, 4.4e-18, is positive (a degenerate origin returned).
  All lie on the family `x1 + 2 x2 ≡ 0 (mod 2π)` to within 3e-14 in phase (the maximum of
  `|rem2pi(2π (x1 + 2 x2), RoundNearest)|` over the returned fractions `x`). At
  `b309a55`, with condition (2), 38 were accepted and 39 returned.
- `Q = u vᵀ` with `u = (1, 2, -1)`, `v = (1, -1, 2)`, `L = (0.3, 0.9, 0.5)`,
  40 starts `2π·rand(MersenneTwister(11), 3)` from a fresh generator: 0
  accepted, empty result.
- Random 3×4 charges from one `MersenneTwister(1)` stream. First 30 rank-1
  draws: `u = rand(rng, -3:3, 3)`, `v = rand(rng, -3:3, 4)` (a draw with
  `u = 0` or `v = 0` would be skipped; none occurred), `Q = u vᵀ`, then
  `L = rand(rng, Uniform(0.1, 1.1), 4)`, then 20 starts `2π·rand(rng, 3)`.
  16 of 30 gave a non-empty result, with 217 points in total. Then 30 rank-2
  draws: `A = rand(rng, -3:3, 3, 2)`, `B = rand(rng, -3:3, 2, 4)`, `Q = A B`
  (a draw with rank other than 2 would be skipped; none occurred), then `L`
  and 20 starts as before. 26 of 30 gave a non-empty result, with 215 points.

This is a classification and counting-unit defect. #220 must establish exact
structural-rank handling and isolated-minimum count eligibility before #217
reconnects the affected pipeline. De-duplication (#216) is not expected to
repair it.

Exposure: `vacua_id` passes each subspace to `subspace_minimize` as a 1×k
single-field charge row, which has full rank when it is nonzero. `vacua_full`
passes the h11×k matrix `Qsub`, which can be rank deficient. Neither currently
reaches `subspace_minimize` (#217), so no stored legacy count is affected. The
limitation applies to direct callers of `subspace_minimize`.

### R-005 — Origin rule

`subspace_minimize` SHALL add the origin only when the computed Float64
smallest eigenvalue of `Q · Diagonal(L) · Qᵀ` is strictly positive. This is a
sign test with no tolerance. A degenerate origin whose computed eigenvalue is
0 or negative SHALL NOT be returned, including rank-deficient flat directions
where the origin is a non-strict minimum.

The test does not decide the sign of the exact eigenvalue:

- **A degenerate origin can be returned.** If its computed eigenvalue rounds
  positive, the origin is returned. The sample has 2000 rank-1 charge matrices
  `Q = u vᵀ` with 3 fields and 4 instantons: `u = rand(rng, -3:3, 3)`,
  `v = rand(rng, -3:3, 4)` (so `Q` has entries up to ±9), draws with `u = 0`
  or `v = 0` are skipped, and `L = rand(rng, Uniform(0.1, 1.1), 4)`, with
  `rng = MersenneTwister(20261031)`. #220 records 131 of 2000 with a positive computed smallest
  eigenvalue, at `ab48726`, with the origin returned each time. A replay at
  `dbd7b62` that makes those calls in that order for each sample (redrawing
  `u` and `v` if either is 0) found 132 of 2000, with the origin returned in
  all 132. The count depends on the draw order; the effect does not.
- **A genuine minimum can be dropped.** If its computed eigenvalue rounds to 0,
  the origin is dropped. An example is `Q = [1 1; 1 -1]`, `L = (1, 1e-20)`.
  Its Float64 Hessian `[1 1; 1 1]` equals that of the saddle
  `L = (1, -1e-20)`.

#220 tracks the replacement.

### R-006 — Zero-column `α` in `vacua_id_basis`

When `α` has no columns (no dependent charges), `vacua_id_basis` SHALL take the
trivial branch and return `|det(Qtilde)|` vacua. The existing test for an
all-zero placeholder column could never hold (*Current baseline*), so this is
the first route into that branch. The zero-column case previously raised
`BoundsError`, so such geometries failed to complete. This adds completions;
it changes no stored count.

### R-007 — Integral legacy count

`_legacy_vacua_search` SHALL store the `vacua_id` count as `Int`. A
non-integral count SHALL raise (`InexactError` from the `Int` conversion). The
`|det|` branch that R-006 makes reachable returns a `Rational`, which HDF5
cannot write. At `f12fbb8` that branch was unreachable, so no geometry failed
at save time. This makes the new completions writable; it changes no stored
count.

### R-008 — Non-finite spectrum input

`spectrum_mode_counts` SHALL raise `ArgumentError` when any eigenvalue is not
finite, instead of classifying it.

At `f12fbb8`, one non-finite eigenvalue made the whole spectrum count as
zeroish (derived from the code). With a `NaN`, the scale
`max(maximum(abs, eigenvalues), 1.0)` is `NaN`, so every comparison is false.
With an `Inf`, the tolerance is `Inf`. `[NaN, 1, -2]` and `[Inf, 1, -2]` both
gave `zeroish = 3`. The callers are inflation scripts
(`scripts/inflation_scan_common.jl`, `scripts/inflation_scale_continuation.jl`,
`scripts/analyze_inflation_candidates.jl`). It is not established whether any
stored output of those scripts was computed from a non-finite spectrum.

### R-009 — Batch default method

`scripts/batch_vacua_pipeline.jl` SHALL default to `--method auto`. Its help
text SHALL:

- state that the previous default was legacy (before PR #215), without tying
  the change to the minimizer fix, and that the method is part of the
  configuration digest;
- point to `--method legacy` to reproduce the legacy counting method used for
  previously stored results;
- state that PR #215 fixes errors that prevented some geometries from
  completing or saving, that it does not invalidate legacy counts already
  stored, and that rerunning may produce additional results for geometries
  that previously failed;
- not call earlier results stale.

The single-geometry API (`compute_vacua_data`, `compute_axion_data` and the
positional CLI) SHALL keep `:legacy`.

`auto` follows the existing decision tree:

1. `jlm_reduced.prepare` builds the reduced problem. Its errors propagate.
2. Exact determinant, only when `square_vacua !== nothing`. Its errors
   propagate.
3. Otherwise the selected branch set. It falls through only on the
   leading-branch enumeration limit (`ArgumentError`). Any other error
   propagates.
4. Reduced JLM. On any error it falls through to legacy, and records the error
   in `auto_fallback_reason`.
5. Legacy.

The selected method is recorded in `search_method` and `auto_selected_method`.
Because the method is part of the configuration digest, results from an
earlier default run (legacy) report a configuration mismatch under `auto`
instead of being reused. The batch run then blocks that geometry unless
`--force` is given. With `--force` it recomputes the geometry and replaces the
stored result, including a stored legacy result. Retention under R-010 means
that PR #215 does not invalidate stored legacy counts; it does not prevent a
user from replacing them with `--force`.

**Rationale (proposed; owner confirmation required).** The reason first given
for the change, that legacy results stored before PR #215 were stale, is
withdrawn (R-010). The proposed rationale is that the legacy method cannot
complete any geometry whose `vacua_id_basis` yields a row that needs the
minimizer, or that has a single dependent column. In the first case `vacua_id`
raises `BoundsError` at the `Leff` mask; in the second, `vacua_id_basis`
raises `BoundsError` after its extra column strip (known issue A). Both come
before the minimizer call, which would itself raise `TypeError` (#217).
`auto` tries the exact-determinant, leading-branch and reduced-JLM searches
first. This rationale does not claim that `auto` and `legacy` counts agree, or that they
are the same counting unit (see *Scientific claim boundary*). If the owner does
not adopt a rationale, the alternative is to keep `--method legacy` as the
batch default, which is an implementation change. Approving this revision as
written confirms the proposed rationale. Another rationale needs a new
revision and a new T002, because the approval-recording commit may change
metadata only.

### R-010 — Records

The scientific behaviour change SHALL be documented in the `minimize`,
`subspace_minimize` and `id_minimize(::Vector, ::Matrix)` docstrings, the batch
help text and the PR #215 description. Those records SHALL limit the change to
direct calls of the minimizer.

Legacy counts stored before PR #215 SHALL be retained. They SHALL NOT be
labelled stale, invalidated, migrated or recomputed because of the minimizer
defect: the legacy pipeline never reached the minimizer. Rerunning the legacy
method may add results for geometries that previously failed (R-006, R-007).
Results of earlier direct or non-pipeline calls of `vacua_id`, `vacua_full`,
`minimize` or `subspace_minimize` are outside this statement. Whether they were
affected is not established.

## Scientific claim boundary

### This specification may establish

- that `minimize` iterates on `V` and its exact derivatives;
- that accepted endpoints satisfy the stated numerical filter, which is a
  heuristic set of local convergence criteria;
- that the origin is added only when the computed Float64 smallest eigenvalue
  of `Q · Diagonal(L) · Qᵀ` is positive;
- that the minimizer acceptance rules in PR #215 do not affect the legacy
  pipeline's stored vacuum counts, because the inspected pipeline cannot reach
  `subspace_minimize`;
- that R-006 and R-007 let previously failing geometries complete, without
  changing counts already written;
- that the fixtures in *Verification* behave as derived by hand.

### This specification must not establish

- that an accepted endpoint is a certified minimum, or a bound on its distance
  from an exact minimum;
- that `"stationary" = false`, or rejection by `subspace_minimize`,
  establishes that no minimum lies nearby;
- that every genuine minimum is accepted: phase rounding, the Float64 sign of
  the smallest eigenvalue, and runs that stop before a weak direction is
  resolved can each lose one (R-003);
- that returned points are isolated minima, or that their number counts
  minima: for rank-deficient charges, points of one flat family are returned
  separately, and their number grows with the number of starts (R-004, #220);
- that removing condition (2) makes `"stationary"` a certificate: the
  remaining conditions are heuristic (R-003, #220);
- that the origin rule decides the sign of the exact smallest eigenvalue: a
  degenerate origin whose computed eigenvalue rounds positive may be returned,
  and a genuine minimum whose computed eigenvalue rounds to 0 may be dropped
  (#220);
- that an empty `subspace_minimize` result means the potential has no minima,
  in particular for rank-deficient charges;
- that Float64 resolves directions about 16 decades below the largest scale,
  in a mixing basis or not. With `Q = I`, `L = (1, -1e-16)`,
  `subspace_minimize(L, Q; runs=100)` returns an empty vector, and none of 100 starts
  `rand(Uniform(0, 2π), 2) .* rand(2)` after `Random.seed!(9876543210)` is
  accepted, although by hand `(0, π)` is a minimum;
- that a legacy count is complete or a rigorous lower bound. Legacy results
  retain the operational classification `finite_multistart_minimum_lower_bound`,
  but this specification does not establish that reported counts are rigorous
  lower bounds, particularly while the known legacy counting defects in #216,
  #217 and #220 remain unresolved;
- that historical legacy results are correct with respect to defects unrelated
  to the minimizer;
- that `auto` and `legacy` counts agree, or that results from different
  `auto_selected_method` values are the same counting unit;
- any correction to de-duplication (#216), phases (#217) or the classification
  of critical points (#220).

## Acceptance gates

### G0 — Specification approval

- An independent Scientific/Numerical Reviewer reviews this specification
  against the implementation candidate. The review record is a comment on
  PR #215 that names the reviewer role and the exact draft revision reviewed
  (the commit SHA at which `spec.md` was reviewed).
- Blocking findings are repaired, and the changed revision is re-reviewed.
- The owner approves that exact reviewed revision in a durable GitHub
  comment on PR #215 that names its commit SHA. That comment is the approval
  event.

Stop condition: the owner amends a decision, or a blocking finding requires an
implementation change. The implementation and evidence then change before G1,
and the implementation candidate is updated. Because *Candidate identity* then
changes, T001, T002, T003 and T005 are repeated for the new revision.

### Approval recording

A later commit sets `status: approved`, `last_reviewed` and `approval_ref`
(the approval comment from G0). It changes only that metadata. It changes no
requirement and no implementation. The approval refers to the reviewed draft
revision, not to this commit, so the reference is not circular.

### G1 — Implementation and evidence reconciliation

- R-001 to R-010 have the evidence listed under *Verification* on the
  implementation candidate. Each regression test listed there was confirmed to
  fail against a deliberately broken version of the code it guards, except
  where the table states that a condition is not isolated or not directly
  verified.
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
| R-001 | **Objective test:** at the mixed-sign minimum (`Q = [1 0 1; 0 1 1]`, `L = (1, 1, -0.9)`, from `(1, 0.95)`), `sign(V) = -1` and `log10\|V\| = log10(16/45)` to a derived `12 eps()`. The derivation assumes a position error of at most `sqrt(eps())`, the step tolerance, which is not a proven bound. It fails with the f12fbb8 objective. **End-to-end minimum** (uniform `L = (1, 1, 1)`): position, gradient and `V` within `sqrt(eps())`-derived bounds. It guards `x_abstol` and the buffer zeroing, but not the objective, because with uniform `L` the f12fbb8 objective is `3V`. **Phase-Hessian fixture:** tests the helper that `id_minimize(::Vector, ::Matrix)` shares |
| R-002 | `V = 1e-20 (1 - cos x)` from `x = 0.3` reaches the minimum at 0 instead of stopping at the start. The absence of Optim deprecation warnings was checked by a manual log search on Optim 1.13.3 (only `x_tol` warns there), not by a test |
| R-003 | **Cases:** hierarchical `L = (1, 1e-20)` and mixed weak direction `Q = [1 1; 1 -1]`, `L = (1, 1e-12)` from `(0.01, -0.01)`, with phases within the hand-derived `2 sqrt(eps())`; these fail with (3) disabled. Constant potential: guarded jointly by (1) and (3), so it fails only if both are disabled. **Regression for the removal of (2):** `L = (1, 1e-6)`, `x = (0, 1e-9)` is accepted; it fails with (2) present. **Not isolated:** condition (1), which (3) implies. **Measured rates:** the 400-start probe (recipe in R-003) is recorded, not a test |
| R-004 | **Cases:** saddle and maximum endpoints rejected; mixed-sign saddle `Q = [1 0 1; 0 1 1]`, `L = (1, 1, -0.9)` with minima at `x1 = x2 = ±acos(5/9)`. **Not isolated:** the `converged` conjunct. **Not tested:** the rank-deficient behaviour, which is a recorded probe (recipes in R-004) |
| R-005 | `-2 sin⁴(x/2)` maximum (`L = (-1, 1/4)`, `Q = [1 2]`), `V ≡ 0`, and rank-deficient `Q = [1; 0]` exclude the origin. The last passes because its zero eigenvalue is computed exactly. The rounding cases in R-005 are documented, not tested |
| R-006 | Non-unimodular `Qtilde = [2 0; 0 1]` gives 2 vacua |
| R-007 | **Integration evidence only:** the legacy batch run on geometry `2,2,1` completes and writes its result (`test/full.jl`). **Not directly verified:** no test asserts that the stored count is an `Int`, or that a non-integral count raises |
| R-008 | `spectrum_mode_counts([NaN, 1.0])` raises `ArgumentError`. The `f12fbb8` behaviour is derived from its code |
| R-009 | `test/full.jl`: the parsed default is `:auto`. The help text names the previous default ("was legacy before PR #215", and not "minimizer fix") and the configuration digest, points to `--method legacy` to reproduce stored results, states that stored legacy counts are not invalidated, and does not contain "stale". The auto-chain conditions are existing behaviour, not changed or newly tested here. The single-geometry `:legacy` default has no source change in PR #215 |
| R-010 | The docstrings and help text are present and limit the change to direct calls; the docs build passes. The reachability facts were established by reading the call graph and running `vacua_id` on the cases in *Current baseline* |

Suites: fast and full Julia suites, Python tests, scientific qualification and
documentation, locally and in CI, on the implementation candidate. Results are
recorded in the PR #215 description. Unobserved checks are not PASS.

## Candidate identity

Two identities are tracked:

- **Implementation candidate:** `dbd7b62` on `codex/test-suite-repairs`. The
  scientific and test evidence in the PR #215 description is produced on it.
  Earlier candidates and their reviews:
  - `ab48726`: reviewed with draft `444fcc3`; not passed.
  - `dc42633`: reviewed with draft `6b033b5`; not passed.
  - `b309a55`: reviewed with draft `0442ead`; not passed.
- **Final merge candidate:** the PR #215 head after the approval-recording
  commit. G1 checks that it differs from the implementation candidate only in
  `specs/0215-minimizer-acceptance-contract/`. G2 reviews it, and CI must pass
  on it before merge.

## Interfaces and compatibility

- `minimize` results gain the `"converged"` and `"stationary"` keys
  (`"stationary"` as in R-003).
- `subspace_minimize` gains the `initial_points` keyword.
- `id_minimize(::Vector, ::Matrix)` returns results computed with the exact
  Hessian (R-001). Its keys and signature are unchanged.
- `pseudo_Q` and `pseudo_L` gain an `rng` keyword for reproducible test data.
  It has no scientific effect.
- The batch default and help text change as in R-009.
- Persisted values: R-007 writes the legacy count to `vacua_pipeline/verified`
  as `Int`. Dataset paths and the overall HDF5 structure are unchanged.
  Before PR #215, no legacy route produced a non-`Int` count (*Current
  baseline*). So no stored legacy count is expected to have a different
  numeric type. This has not been checked against historical files.
- Package-version impact: patch, no version bump.

## Open owner decisions

- **R-009 rationale.** The owner confirms the proposed rationale for the
  `--method auto` batch default, states another, or keeps `--method legacy`
  as the default (an implementation change).

Approval of this specification is the owner's decision on R-001 to R-010 and
on the amendments, as written. The replacement of R-003 to R-005 by a
proof-backed classifier is a separate owner decision under #220.
