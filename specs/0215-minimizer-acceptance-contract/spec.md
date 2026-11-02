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
before merge.

The legacy vacua pipeline never reached the minimizer (see *Current
baseline*), so these repairs do not change any stored legacy count. PR #215
also fixes two errors that stopped some legacy geometries from completing or
saving. A proof-backed classifier that replaces R-003 to R-005 is proposed
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

  These supersede the earlier R-003, R-009 and R-010 wording. The failed
  reviews are not relabelled as passed. The amendment becomes durable when the
  owner approves the revision that contains it at G0.

## Scope

This specification governs only these changes in PR #215:

- the objective and Hessian that `minimize` passes to Newton's method, and the
  shared Hessian helper `_phase_hessian!`, which also changes the Hessian of
  `id_minimize(::Vector, ::Matrix)`;
- the stopping tolerances and the `"stationary"` acceptance filter of
  `minimize`;
- the endpoint filter and the origin rule of `subspace_minimize`;
- the zero-column `α` case of `vacua_id_basis`;
- the integral count stored by `_legacy_vacua_search`;
- rejection of non-finite input by `spectrum_mode_counts`;
- the default `--method` of `scripts/batch_vacua_pipeline.jl` and its help
  text.

The rest of PR #215 (test-suite structure, Python test portability, CI
triggers, documentation layout, test runtime) is S0/S1 engineering. It does not
change scientific meaning and is not governed here.

## Non-scope

This specification does not cover or authorize:

- legacy vacuum de-duplication in `vacua_id` and `vacua_full` (#216);
- the legacy subspace `phase` contract, including the unreachable minimizer
  call in `vacua_id` and `vacua_full` (#217);
- a proof-backed classification of minima, non-minima and unresolved critical
  points, including the replacement of R-003 condition (2) (#220);
- `id_minima`, which returns `nothing`, and `id_minimize(::Vector, ::Vector)`,
  which raises `MethodError` on the shared helper. Both are non-functional
  before and after PR #215;
- known issues A and B recorded in PR #215 (`@test_broken` cases);
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

- `_legacy_vacua_search` (`scripts/vacua_pipeline.jl`) calls only
  `vacua_id(L, Q)`. The only minimizer call in `vacua_id`
  (`src/generate.jl`) is `subspace_minimize(...; phase=phase[i])`. It passes a
  scalar to the keyword `phase::Matrix`, so it raises `TypeError` whenever it
  is reached. That call has been unchanged since 2022-12; the pipeline was
  created on 2026-07-30.
- In the nontrivial cases tested (`Q = [2 0 1 1; 0 1 1 0]`,
  `L = [1 1 1 1; 0 -0.1 -0.2 -0.3]`, and known issue A), `vacua_id` raises
  `BoundsError` even earlier, at the `Leff` column mask.
- The minimizer has therefore never run inside a legacy pipeline count. Stored
  legacy counts come only from the branches that do not minimize.
- At `f12fbb8`, the `|det|` branch returned its count as a `Rational`, which
  HDF5 cannot write (`MethodError`), so those geometries failed to save. The
  zero-column `α` case raised `BoundsError`. R-007 and R-006 fix these. Before
  PR #215, such geometries failed to complete; no stored count was produced
  for them.

### Owner direction recorded in PR #215

- The strict origin rule (R-005) was chosen by the owner after review, as
  recorded in the PR #215 description.
- The scope amendment under *Review and amendment history* (stored legacy
  counts retained; condition (2) kept as a documented limitation, with its
  replacement deferred to #220).
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
power-of-two factors, and otherwise changes only by rounding.

### R-003 — Legacy stationarity filter and its limitation

`minimize` SHALL report `"stationary" = true` only when all of these hold:

1. `max_i s_i > 0`;
2. `|∂_i V| ≤ (eps()/1e-2) · s_i` for every component `i`;
3. no computed eigenvalue of the Hessian at the endpoint is exactly 0, and the
   Newton step satisfies `‖H⁻¹ ∇V‖_∞ ≤ sqrt(eps())`, computed in the Hessian
   eigenbasis.

`"converged"` SHALL report Optim's convergence flag without change.

**Meaning.** The `"stationary"` field reports whether a Newton endpoint
satisfies the legacy numerical acceptance conditions. It is not a certificate
of an exact or nearby stationary point. `"stationary" = false` means that the
endpoint failed this filter. It does not establish that no minimum lies nearby.

**Limitation: condition (2) rejects genuine minima.** Condition (2) can reject
endpoints of genuine local minima even when condition (3), the Newton-step
condition, passes. This is a documented limitation, not evidence that those
candidates are not minima. PR #215 retains condition (2) as an existing legacy
heuristic, and does not claim complete or certified classification of
stationary points. Its replacement by validated nearby-root certification,
with explicit unresolved outcomes, is deferred to #220.

- **Mechanism (derived by hand).** For `Q = I`, condition (2) on component `i`
  reads `L_i |sin x_i| ≤ 100 eps() L_i`, that is `|x_i| ≤ 100 eps() ≈ 2.2e-14`
  rad, whatever `L_i` is. Condition (3) allows a displacement of about
  `sqrt(eps()) ≈ 1.5e-8` rad. Optim stops on a gradient relative to the
  largest scale, `max_i s_i`. Along a weak direction it can therefore stop at
  a displacement that condition (3) accepts and condition (2) rejects.
  Rounding of large phases `Q_a · x` adds a second route to the same failure.
- **Measured.** `Q = I`, `L = (1, 1e-6)`, 400 starts uniform in `[0, 2π)²`
  from `MersenneTwister(4242)`. All 400 endpoints were converged, had
  `Hsign = 1`, and lay within `sqrt(eps())` rad of a minimum (displacement
  recovered as `|∂_i V| / L_i`). Condition (2) rejected 208 of them, one only
  2.3e-14 rad away.
- **Further basis.** (2) allows 100 times `eps()` relative to `s_i`, the bound
  on `|∂_i V|`. It does not account for the rounding of the phases, which grows
  with `Σ_i |Q_ia x_i|`, or for accumulation over the N terms. (3) is the step
  tolerance of R-002. It is needed because (2) is not basis invariant: a weak
  direction that mixes the fields can have small gradient components while the
  point is displaced along it.
- **Isolation.** The known-issue test for `L = (1, 1e-6)`, `x = (0, 1e-9)`
  isolates condition (2): removing (2) turns its `@test_broken` into an
  unexpected pass. Condition (1) is not isolated by any test. For an exactly
  constant `V` the Hessian is exactly 0, so condition (3) also rejects it.

### R-004 — Endpoint acceptance in `subspace_minimize`

`subspace_minimize` SHALL keep a Newton endpoint only if it is converged,
passes the R-003 filter and has `Hsign == 1` (computed smallest Hessian
eigenvalue strictly positive). Saddles, maxima, endpoints that fail the filter
and non-converged runs SHALL be discarded. A discarded endpoint is not thereby
established to be a non-minimum (R-003).

If the charges have rank `r < n` (`n` fields), every critical point lies on an
exactly flat family, and the exact smallest Hessian eigenvalue is 0. R-003 and
R-004 then reject the endpoints unless rounding makes the computed eigenvalue
positive, and R-005 rejects the origin under the same condition. The result is
usually empty. An empty result does not mean that the potential has no minima.
`vacua_id` and `vacua_full` pass such rank-deficient charge matrices, but
neither currently reaches `subspace_minimize` (#217).

### R-005 — Origin rule

`subspace_minimize` SHALL add the origin only when the computed Float64
smallest eigenvalue of `Q · Diagonal(L) · Qᵀ` is strictly positive. This is a
sign test with no tolerance. A degenerate origin whose computed eigenvalue is
0 or negative SHALL NOT be returned, including rank-deficient flat directions
where the origin is a non-strict minimum.

The test does not decide the sign of the exact eigenvalue:

- **A degenerate origin can be returned.** If its computed eigenvalue rounds
  positive, the origin is returned. In the seeded sample of #220, the computed
  smallest eigenvalue was above 0 in 131 of 2000 random rank-1 `Q`, and the
  origin was returned each time. The sample is: seed `20261031`; 3 fields and
  4 instantons; integer entries in `[-3, 3]`; `L_a` uniform in `[0.1, 1.1]`.
  Independent review found 142 of 1993.
- **A genuine minimum can be dropped.** If its computed eigenvalue rounds to 0,
  the origin is dropped. An example is `Q = [1 1; 1 -1]`, `L = (1, 1e-20)`.
  Its Float64 Hessian `[1 1; 1 1]` equals that of the saddle
  `L = (1, -1e-20)`.

#220 tracks the replacement.

### R-006 — Zero-column `α` in `vacua_id_basis`

When `α` has no columns (no dependent charges), `vacua_id_basis` SHALL take the
trivial branch and return `|det(Qtilde)|` vacua, the same as the existing
all-zero placeholder case. It previously raised `BoundsError`, so such
geometries failed to complete. This adds completions; it changes no stored
count.

### R-007 — Integral legacy count

`_legacy_vacua_search` SHALL store the `vacua_id` count as `Int`. A
non-integral count SHALL raise (`InexactError` from the `Int` conversion). At
`f12fbb8`, the `|det|` branch returned a `Rational` that HDF5 could not write,
so those geometries failed to save. This makes them writable; it changes no
stored count.

### R-008 — Non-finite spectrum input

`spectrum_mode_counts` SHALL raise `ArgumentError` when any eigenvalue is not
finite, instead of classifying it.

### R-009 — Batch default method

`scripts/batch_vacua_pipeline.jl` SHALL default to `--method auto`. Its help
text SHALL:

- state that the previous default was legacy and that the method is part of
  the configuration digest;
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
instead of being reused.

### R-010 — Records

The scientific behaviour change SHALL be documented in the `minimize` and
`subspace_minimize` docstrings, the batch help text and the PR #215
description. Those records SHALL limit the change to direct calls of the
minimizer.

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
- that accepted endpoints satisfy the stated numerical filter, which is a set
  of local convergence criteria;
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
- that every genuine minimum is accepted: condition (2) rejects genuine minima
  at any phase (R-003), and large phases add rounding;
- that the origin rule decides the sign of the exact smallest eigenvalue: a
  degenerate origin whose computed eigenvalue rounds positive may be returned,
  and a genuine minimum whose computed eigenvalue rounds to 0 may be dropped
  (#220);
- that an empty `subspace_minimize` result means the potential has no minima,
  in particular for rank-deficient charges;
- that Float64 resolves directions about 16 decades below the largest scale,
  in a mixing basis or not. With `Q = I`, `L = (1, -1e-16)`, no endpoint of
  100 seeded starts is accepted, and `subspace_minimize` returns nothing,
  although by hand `(0, π)` is a minimum;
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
| R-001 | **Objective test:** at the mixed-sign minimum (`Q = [1 0 1; 0 1 1]`, `L = (1, 1, -0.9)`, from `(1, 0.95)`), `sign(V) = -1` and `log10\|V\| = log10(16/45)` to a derived `12 eps()`. It fails with the f12fbb8 objective. **End-to-end minimum** (uniform `L = (1, 1, 1)`): position, gradient and `V` within `sqrt(eps())`-derived bounds. It guards `x_abstol` and the buffer zeroing, but not the objective, because with uniform `L` the f12fbb8 objective is `3V`. **Phase-Hessian fixture:** tests the helper that `id_minimize(::Vector, ::Matrix)` shares |
| R-002 | `V = 1e-20 (1 - cos x)` from `x = 0.3` reaches the minimum at 0 instead of stopping at the start. The absence of Optim deprecation warnings was checked by a manual log search on Optim 1.13.3 (only `x_tol` warns there), not by a test |
| R-003 | **Cases:** hierarchical `L = (1, 1e-20)`; constant potential; mixed weak direction `Q = [1 1; 1 -1]`, `L = (1, 1e-12)` from `(0.01, -0.01)`, with phases within the hand-derived `2 sqrt(eps())`. **Known-issue test:** `@test_broken` for `L = (1, 1e-6)`, `x = (0, 1e-9)`, which isolates condition (2). **Not isolated:** condition (1). **Measured rate:** the 208-of-400 rejection rate is a recorded probe (recipe in R-003), not a test |
| R-004 | **Cases:** saddle and maximum endpoints rejected; mixed-sign saddle `Q = [1 0 1; 0 1 1]`, `L = (1, 1, -0.9)` with minima at `x1 = x2 = ±acos(5/9)`. **Not isolated:** the `converged` conjunct. **Not tested:** the empty result for rank-deficient `Q` |
| R-005 | `-2 sin⁴(x/2)` maximum (`L = (-1, 1/4)`, `Q = [1 2]`), `V ≡ 0`, and rank-deficient `Q = [1; 0]` exclude the origin. The last passes because its zero eigenvalue is computed exactly. The rounding cases in R-005 are documented, not tested |
| R-006 | Non-unimodular `Qtilde = [2 0; 0 1]` gives 2 vacua |
| R-007 | **Integration evidence only:** the legacy batch run on geometry `2,2,1` completes and writes its result (`test/full.jl`). **Not directly verified:** no test asserts that the stored count is an `Int`, or that a non-integral count raises |
| R-008 | `spectrum_mode_counts([NaN, 1.0])` raises `ArgumentError` |
| R-009 | `test/full.jl`: the parsed default is `:auto`. The help text names the previous default and the configuration digest, points to `--method legacy` to reproduce stored results, states that stored legacy counts are not invalidated, and does not contain "stale". The auto-chain conditions are existing behaviour, not changed or newly tested here. The single-geometry `:legacy` default has no source change in PR #215 |
| R-010 | The docstrings and help text are present and limit the change to direct calls; the docs build passes. The reachability facts were established by reading the call graph and running `vacua_id` on the cases in *Current baseline* |

Suites: fast and full Julia suites, Python tests, scientific qualification and
documentation, locally and in CI, on the implementation candidate. Results are
recorded in the PR #215 description. Unobserved checks are not PASS.

## Candidate identity

Two identities are tracked:

- **Implementation candidate:** `b309a55` on `codex/test-suite-repairs`. The
  scientific and test evidence in the PR #215 description is produced on it.
  Earlier candidates and their reviews:
  - `ab48726`: reviewed with draft `444fcc3`; not passed.
  - `dc42633`: reviewed with draft `6b033b5`; not passed.
- **Final merge candidate:** the PR #215 head after the approval-recording
  commit. G1 checks that it differs from the implementation candidate only in
  `specs/0215-minimizer-acceptance-contract/`. G2 reviews it, and CI must pass
  on it before merge.

## Interfaces and compatibility

- `minimize` results gain the `"converged"` and `"stationary"` keys.
- `subspace_minimize` gains the `initial_points` keyword.
- `id_minimize(::Vector, ::Matrix)` returns results computed with the exact
  Hessian (R-001). Its keys and signature are unchanged.
- `pseudo_Q` and `pseudo_L` gain an `rng` keyword for reproducible test data.
  It has no scientific effect.
- The batch default and help text change as in R-009.
- Persisted values: R-007 writes the legacy count to `vacua_pipeline/verified`
  as `Int`. Dataset paths and the overall HDF5 structure are unchanged.
  Before PR #215, a `Rational` count could not be written. So no stored legacy
  count is expected to have a different numeric type. This has not been
  checked against historical files.
- Package-version impact: patch, no version bump.

## Open owner decisions

None are intentionally left open. Approval of this specification is the owner's
decision on R-001 to R-010 and on the scope amendment, as written. The
replacement of R-003 to R-005 by a proof-backed classifier is a separate owner
decision under #220.
