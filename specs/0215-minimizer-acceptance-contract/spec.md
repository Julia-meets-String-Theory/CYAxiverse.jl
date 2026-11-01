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
numerical endpoints of the legacy minimizer count as minima, when the origin is
counted, how two legacy vacua branches report their counts, and which search
method batch vacua runs use by default.

## Motivation

At `f12fbb8`, the legacy minimizer passed Newton's method an objective and a
Hessian that were not those of the potential, used a step tolerance tied to the
smallest instanton scale, and counted every Newton endpoint and the origin as
minima. Legacy vacuum counts therefore had no defined acceptance criterion.
PR #215 repairs the objective and Hessian and states acceptance criteria. This
specification records those criteria and what they do not establish, so that
the owner can approve them before merge. A proof-backed classifier that
replaces R-003 to R-005 is proposed separately in #220.

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

## Scope

This specification governs only these changes in PR #215:

- the objective and Hessian that `minimize` passes to Newton's method, and the
  shared Hessian helper `_phase_hessian!`, which also changes the Hessians of
  `id_minimize` (both methods) and so of `id_minima`;
- the stopping tolerances and the `"stationary"` acceptance criterion of
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
- the legacy subspace `phase` contract (#217);
- a proof-backed classification of minima, non-minima and unresolved critical
  points (#220);
- detection or rejection of stale legacy results by the pipeline (#216);
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
  objective was `(Σ_a Λ_a) · Σ_b (1 - cos θ_b)`. The gradient was correct.
  Newton's method therefore received an objective and Hessian that were not
  those of `V`.
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

The same Hessian helper serves `id_minimize` and `id_minima`, where it is the
`nlsolve` Jacobian and enters their eigenvalue sign test. Their Hessians change
in the same way. Their own acceptance rules are unchanged.

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
3. no computed eigenvalue of the Hessian at the endpoint is exactly 0, and the
   Newton step satisfies `‖H⁻¹ ∇V‖_∞ ≤ sqrt(eps())`, computed in the Hessian
   eigenbasis.

Basis: (2) allows 100 times `eps()` relative to `s_i`, the bound on `|∂_i V|`.
It does not account for the rounding of the phases `Q_a · x`, which grows with
`Σ_i |Q_ia x_i|`, or for accumulation over the N terms. At phases of order
100 rad or more, the computed gradient at an exact minimum can exceed it, so
(2) can reject a genuine minimum. (3) is the step tolerance of R-002. It is
needed because (2) is not basis invariant: a weak direction that mixes the
fields can have small gradient components while the point is displaced along
it.

Condition (2) is not isolated by any test: in the fixtures, (1) or (3) also
rejects every case that (2) rejects. Whether (2) catches anything that (3)
misses is an open question. #220 proposes removing it as an acceptance gate.
It is therefore deliberately not pinned by a new test here.

`"converged"` SHALL report Optim's convergence flag without change.

### R-004 — Endpoint acceptance in `subspace_minimize`

`subspace_minimize` SHALL keep a Newton endpoint only if it is converged,
stationary (R-003) and has `Hsign == 1` (computed smallest Hessian eigenvalue
strictly positive). Saddles, maxima, unresolved endpoints and non-converged
runs SHALL be discarded.

If the charges have rank `r < n` (`n` fields), every critical point lies on an
exactly flat family, and the exact smallest Hessian eigenvalue is 0. R-003 and
R-004 then reject the endpoints unless rounding makes the computed eigenvalue
positive, and R-005 rejects the origin under the same condition. The result is
usually empty. In independent review, three rank-deficient `Q` with 200 starts
each kept no endpoint. An empty result does not mean that the potential has no
minima. `vacua_full` passes such rank-deficient charge matrices; it is
currently unreachable because of the `phase` `TypeError` in #217.

### R-005 — Origin rule

`subspace_minimize` SHALL add the origin only when the computed Float64
smallest eigenvalue of `Q · Diagonal(L) · Qᵀ` is strictly positive. This is a
sign test with no tolerance. A degenerate origin whose computed eigenvalue is
0 or negative SHALL NOT be returned, including rank-deficient flat directions
where the origin is a non-strict minimum.

The test does not decide the sign of the exact eigenvalue. A degenerate origin
whose computed eigenvalue rounds positive is returned. In a seeded sample of
2000 random rank-1 `Q`, 131 had a computed smallest eigenvalue above 0, and
the origin was returned each time (independent review: 142 of 1993). A
genuine minimum whose computed eigenvalue rounds to 0 is dropped. An example is
`Q = [1 1; 1 -1]`, `L = (1, 1e-20)`, whose Float64 Hessian `[1 1; 1 1]` equals
that of the saddle `L = (1, -1e-20)`. #220 tracks the replacement.

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
text SHALL state the previous default, SHALL state that legacy results written
before the minimizer fix are stale and that the configuration digest cannot
detect them, and SHALL point to `--force` to recompute. It SHALL NOT advise
`--method legacy` to reuse them. The single-geometry API (`compute_vacua_data`,
`compute_axion_data` and the positional CLI) SHALL keep `:legacy`.

`auto` follows the existing decision tree:

1. exact determinant, only when the reduced problem has `square_vacua !== nothing`;
2. otherwise the selected branch set. It falls through only on the
   leading-branch enumeration limit (`ArgumentError`); any other error
   propagates;
3. reduced JLM. On any error it falls through to legacy and records the error
   in `auto_fallback_reason`;
4. legacy.

The selected method is recorded in `search_method` and `auto_selected_method`.
Because the method is part of the configuration digest, results from an
earlier default run (legacy) report a configuration mismatch under `auto`
instead of being reused.

### R-010 — Records

The scientific behaviour change SHALL be documented in the `minimize` and
`subspace_minimize` docstrings, the batch help text and the PR #215
description. Legacy vacua results produced before PR #215 are stale and SHALL
be recomputed, not compared with new counts.

This is documented, not enforced. `VACUA_PIPELINE_VERSION` is unchanged and the
configuration digest records no minimizer identity, so the pipeline does not
detect pre-fix legacy results: a run with `--method legacy` treats them as
matching and reuses them unless `--force` is passed. Enforcement is follow-up
work in #216.

## Scientific claim boundary

### This specification may establish

- that `minimize` iterates on `V` and its exact derivatives;
- that accepted endpoints satisfy the stated numerical criteria, which are
  local convergence criteria;
- that the origin is added only when the computed Float64 smallest eigenvalue
  of `Q · Diagonal(L) · Qᵀ` is positive;
- that the fixtures in *Verification* behave as derived by hand.

### This specification must not establish

- that an accepted endpoint is a certified minimum, or a bound on its distance
  from an exact minimum;
- that every nondegenerate minimum is accepted: large phases can fail R-003
  condition (2);
- that the origin rule decides the sign of the exact smallest eigenvalue: a
  degenerate origin whose computed eigenvalue rounds positive may be returned,
  and a genuine minimum whose computed eigenvalue rounds to 0 may be dropped
  (#220);
- that an empty `subspace_minimize` result means the potential has no minima,
  in particular for rank-deficient charges;
- that a legacy count is complete or a rigorous lower bound. Legacy results
  retain the operational classification `finite_multistart_minimum_lower_bound`,
  but this specification does not establish that reported counts are rigorous
  lower bounds, particularly while the known legacy counting defects in #216,
  #217 and #220 remain unresolved;
- that Float64 eigenvalues resolve the sign of strongly hierarchical directions
  (more than about 16 decades in a mixing basis);
- that the pipeline detects or rejects legacy results produced before PR #215;
- that `auto` and `legacy` counts agree, or that results from different
  `auto_selected_method` values are the same counting unit;
- any correction to de-duplication (#216), phases (#217) or the classification
  of critical points (#220).

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

Stop condition: the owner amends a decision, or a blocking finding requires an
implementation change. The implementation and evidence then change before G1,
the implementation candidate is updated, and T001 and T005 are repeated for it.

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
| R-001 | Mixed-sign minimum `Q = [1 0 1; 0 1 1]`, `L = (1, 1, -0.9)` from `(1, 0.95)`: `sign(V) = -1` and `log10|V| = log10(16/45)` to a derived `11 eps()` (fails with the f12fbb8 objective). End-to-end minimum with uniform `L = (1, 1, 1)`: position, gradient and `V` within `sqrt(eps())`-derived bounds; this guards `x_abstol` and the buffer zeroing, but not the objective, because with uniform `L` the f12fbb8 objective is `3V`. Phase-Hessian fixture. The `id_minimize` Hessians share the helper and its fixture |
| R-002 | `V = 1e-20 (1 - cos x)` from `x = 0.3` reaches the minimum at 0 instead of stopping at the start. The absence of Optim deprecation warnings was checked by a manual log search on Optim 1.13.3 (only `x_tol` warns there), not by a test |
| R-003 | Hierarchical `L = (1, 1e-20)`; constant potential; mixed weak direction `Q = [1 1; 1 -1]`, `L = (1, 1e-12)`, start `(0.01, -0.01)`, with phases within the hand-derived `2 sqrt(eps())`. Condition (2) is not isolated (see R-003) |
| R-004 | Saddle and maximum endpoints rejected; mixed-sign saddle `Q = [1 0 1; 0 1 1]`, `L = (1, 1, -0.9)` with minima at `x1 = x2 = ±acos(5/9)`. The `converged` conjunct is not isolated by any test. The empty result for rank-deficient `Q` is not tested |
| R-005 | `-2 sin⁴(x/2)` maximum (`L = (-1, 1/4)`, `Q = [1 2]`), `V ≡ 0`, and rank-deficient `Q = [1; 0]` exclude the origin. The last passes because its zero eigenvalue is computed exactly; the rounding cases in R-005 are documented, not tested |
| R-006 | Non-unimodular `Qtilde = [2 0; 0 1]` gives 2 vacua |
| R-007 | Integration evidence only: the legacy batch run on geometry `2,2,1` completes and writes its result (`test/full.jl`). Not directly verified: no test asserts that the stored count is an `Int`, or that a non-integral count raises (Julia's `Int` conversion raises `InexactError`) |
| R-008 | `spectrum_mode_counts([NaN, 1.0])` raises `ArgumentError` |
| R-009 | Parsed default is `:auto`; help text names the configuration digest, does not advise `--method legacy` to match old results, states that the digest does not record the minimizer version, and points to `--force` (`test/full.jl`). The auto-chain fall-through conditions are existing behaviour, not changed or newly tested here. The single-geometry `:legacy` default has no source change in PR #215 |
| R-010 | Docstrings and help text present; docs build passes. Enforcement is not implemented (see R-010) |

Suites: fast and full Julia suites, Python tests, scientific qualification and
documentation, locally and in CI, on the implementation candidate. Results are recorded in the
PR #215 description. Unobserved checks are not PASS.

## Candidate identity

Two identities are tracked:

- **Implementation candidate:** `dc42633` on `codex/test-suite-repairs`. The
  scientific and test evidence in the PR #215 description is produced on it.
  It replaces `ab48726`, after the G0 review of `444fcc3` required a help-text
  change (R-009) and an objective-value test (R-001).
- **Final merge candidate:** the PR #215 head after the approval-recording
  commit. G1 checks that it differs from the implementation candidate only in
  `specs/0215-minimizer-acceptance-contract/`. G2 reviews it, and CI must pass
  on it before merge.

## Interfaces and compatibility

- `minimize` results gain the `"converged"` and `"stationary"` keys.
- `subspace_minimize` gains the `initial_points` keyword.
- `id_minimize` and `id_minima` return results computed with the exact Hessian
  (R-001). Their keys and signatures are unchanged.
- `pseudo_Q` and `pseudo_L` gain an `rng` keyword for reproducible test data.
  It has no scientific effect.
- The batch default and help text change as in R-009.
- Persisted values: R-007 normalizes the legacy count to `Int` before it is
  written to `vacua_pipeline/verified`, so the stored numeric type can differ
  from earlier legacy results. Dataset paths and the overall HDF5 structure are
  unchanged. The datatype of historically stored legacy counts has not been
  verified.
- Package-version impact: patch, no version bump.

## Open owner decisions

None are intentionally left open. Approval of this specification is the owner's
decision on R-001 to R-010 as written. The replacement of R-003 to R-005 by a
proof-backed classifier is a separate owner decision under #220.
