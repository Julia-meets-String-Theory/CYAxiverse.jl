---
spec_id: CYAX-0234
title: Required solver tolerances for the n8 author trajectory
issue: 234
class: S1
status: approved
workstream: A96 poly-102 inflation reproduction path
parent: null
depends_on: [CYAX-0232]
created: 2026-11-14
last_reviewed: 2026-11-14
review_required: repository-owner approval
approval_ref: "Issue #234 comment 6880625727 — owner decision option (b), 2026-11-14: require explicit tolerances, no new default numbers"
version_impact: none adopted on this branch (public keyword defaults removed; see Compatibility)
---

# Required solver tolerances for the n8 author trajectory

## Objective

Make the solver tolerances of `n8_author_trajectory` and of the script-level
refinement configuration required, so that a call without them fails at once
instead of running a solve that cannot converge.

## Current baseline

Facts at `vmm` `563c2d4` (after CYAX-0232, PR #237).

- `CYAxiverse.paper_benchmarks.author_inflation.n8_author_trajectory`
  (`src/paper_benchmarks/poly102_inflation.jl`) takes `reltol=nothing` and
  `abstol=nothing` and, when they are `nothing`, uses
  `10^-(precision_bits ÷ 2)` and `10^-(2 precision_bits ÷ 3)`.
- For every allowed `precision_bits ≥ 64` that `reltol` is below the BigFloat
  working precision `2^-precision_bits` (64 bits: 1e-32 against 5.4e-20;
  100 bits, the default: 1e-50 against 7.9e-31). A fast-suite test at 64 bits
  with these defaults did not finish (Issue #234). Whether the 100-bit default
  call also fails to converge was inferred, not tested.
- `n8_physical_gradient_flow` is the same function (alias in the same file,
  re-exported by `src/paper_benchmarks/compatibility.jl`).
- `scripts/inflation_refinement_common.jl`: `inflation_refinement_config`
  passes `reltol=nothing`/`abstol=nothing` through, and
  `refine_inflation_candidate(candidate; config=inflation_refinement_config())`
  uses that config by default. Its `inflation_stage_measure(...;
  capture_errors=true)` would turn a solver error into a `:failed` summary.
- Every call site in the repository passes both tolerances:
  `test/kinetic_normalization.jl` (CYAX-0232 reproduction labels),
  `test/runtests.jl` (refinement test), and
  `scripts/benchmark_inflation_scalability.jl`.

## Scope

- `n8_author_trajectory` / `n8_physical_gradient_flow`.
- `inflation_refinement_config` and, through its default argument,
  `refine_inflation_candidate`.
- Their docstrings, `docs/src/api.md` and
  `validation/inflation_scan_call_contract.md`.
- Regression tests.

## Non-scope

- Choosing new default tolerance values (owner decision: none).
- Any numerical change for callers that pass tolerances.
- Other trajectory routes (`n8_slow_roll_trajectory`, `n8_hilltop_probe`).
- The signature of `refine_inflation_candidate`.

## Requirements

### R-001 — The trajectory requires both tolerances

When `reltol` or `abstol` is `nothing`, `n8_author_trajectory` (and so its
alias) throws an `ArgumentError` before any BigFloat setup or solving. The
message names the missing keyword(s), tells the caller to pass both
explicitly, and says why: the former precision-derived defaults were below the
working precision. The existing argument checks keep their order before this
check.

### R-002 — The refinement configuration requires both tolerances

`inflation_refinement_config` throws the same kind of `ArgumentError` when
`reltol` or `abstol` is missing, so the failure happens when the config is
built. Positive tolerances are still required. `refine_inflation_candidate`
keeps its signature; called without `config`, it therefore throws (fails
closed) instead of returning a `:failed` summary.

### R-003 — No numerical change for explicit callers

With explicit tolerances, the solver receives `BigFloat(reltol)` and
`BigFloat(abstol)` at the working precision, exactly as before, and the
recorded `solver.reltol`/`solver.abstol` are unchanged. The refinement config
stores `Float64` tolerances as before.

### R-004 — Documentation

The docstrings of `n8_author_trajectory`/`n8_physical_gradient_flow` and
`inflation_refinement_config`, `docs/src/api.md`, and
`validation/inflation_scan_call_contract.md` state that the tolerances are
required and why.

### R-005 — Regression tests

Fast-suite tests show that the default call of each name (module and
compatibility namespace) throws `ArgumentError` within a wall-clock bound far
below the former hang; that passing only one tolerance throws; that
`inflation_refinement_config()` and a one-tolerance config throw; and that
`refine_inflation_candidate(candidate)` with its default config throws. The
existing tests pass unchanged.

## Compatibility

Intentional public-behaviour break: a call to `n8_author_trajectory`,
`n8_physical_gradient_flow` or `inflation_refinement_config` without both
tolerances now throws `ArgumentError` instead of attempting a solve that does
not converge. No repository caller relied on the defaults.
