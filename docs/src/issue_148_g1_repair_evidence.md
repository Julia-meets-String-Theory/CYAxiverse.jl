# Issue 148 G1 repair evidence

Scope: N5 zero-phase source-reduced continuation (issue-only, no N8 conventions changed).
Source convention kept from `docs/src/issue_148_g0_baseline_audit.md`:
`a(k) = (32/(255/8)) * exp(-2πk(32-255/8))`,
`k_c = (4/π) log(1024/255)`.
Source PDF SHA-256 (tracked in independent review doc): `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.

Environment used for checks:
- Julia: `julia-1.12.6+0`
- Command environment: `JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia`
- Working tree: `f9765003119f3fd7bde216eb31a5dbffbcc268c7` (parenting manager commit).
- Date: `2026-09-09`.

## Focused checks run

- `python3 scripts/agent_verify.py snapshot`
  - status: passed (0)
- `python3 scripts/agent_verify.py diff-check`
  - status: passed (0)
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/inflation_scale_continuation.jl --benchmarks-only`
  - status: passed (0)
  - key output: `n5_continuation` has length 4, `n5_catastrophe_index=2`, `n5_catastrophe_k=1.7700681326502656`, gradient magnitudes at all 4 steps <= `1e-18`, and returned catastrophe theta at `π`.
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia julia --startup-file=no --project=. /tmp/g1_n5_regression_check.jl`
  - status: passed (0)
  - exact output excerpt: `n5_kc_path_catastrophe_index=2`, `n5_kc_path_catastrophe_k=1.7700681483189589` (with perturbed grid),
    `n5_kc_path_catastrophe_theta=3.141592653589793`, final single-step residual check `iterations=1`.
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia julia --startup-file=no --project=. /tmp/g1_benchmark_check.jl`
  - status: passed (0)
  - key output:
    - `passed=true`
    - `n5_catastrophe_index=2`, `n5_catastrophe_k=1.7700681326502656`
    - `n5_zero_curvature=true`, `n5_minima_below=2`, `n5_minima_above=1`

## Targeted regression boundary checks now enforced in `test/runtests.jl`

- Event-localized catastrophe continuity with perturbed seed (`seed_theta = π + 1e-3`) and `k` values not containing exact `k_c`:
  - `catastrophe_k` is finite and within `5e-7` of source `k_c`.
  - catastrophe is flagged on adjacent continuation steps.
- Branch identity remains intrinsic (`branch == :pi` for this continuation path).
- Final-update convergence is revalidated after the Newton loop.
- Max-iteration `1` run and `catastrophe`/`hessian` sign checks added.

## Full package test status

- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'`
  - failed in pre-existing `CYAxiverse.jl` testset `phase and volume detuning scan` at line `@test S.hessian([0.0, 0.0], Q, L, phases[1]) ≈ 4π^2 * I`.
  - Reproduction confirms `S.hessian(...) = 2π^2 I`.
  - this failure is independent of the Issue 148 N5 continuation code path.

## Tolerance choices

- `gradient_tolerance=1e-10`, `hessian_tolerance=1e-10` selected as continuation stopping and fold-detection controls.
- `n5_ratio_formula` and critical-scale checks use `atol=1e-12...5e-13` in witnessed ranges matching observed `1e-15`–`1e-18` residual agreement in the checks above.
