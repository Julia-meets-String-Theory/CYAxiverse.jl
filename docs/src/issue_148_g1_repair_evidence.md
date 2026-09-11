# Issue 148 G1 repair evidence

Scope: N5 zero-phase source-reduced continuation only; no N8 changes and no
physical-normalization, scientific-schema, or persisted-data updates. The change
adds the N5 continuation function and result type (additive N5 API); it does not
remove or change existing public interfaces. Persisted schema: N/A.

## Replay identity

- **Source:** arXiv:2608.14780v1, PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- **Witness:** accepted-G0 N=5 source fixture, `h11=5`, polytope 2787; one
  analytically selected regular zero-phase branch (`branch=:pi`), not a branch
  enumeration.
- **Selection route:** single known regular branch; no off-ray, exhaustive, or
  through-cusp enumeration.
- **Counting unit:** N/A (single-branch continuation, not a population count).
- **Coordinate/action convention:** reduced radian angle `vartheta`, with
  `2pi theta_light = (0,0,1,2,0) vartheta`, and
  `a(k) = (32/(255/8)) exp[-2pi k (32-255/8)]`.
- **Scale:** dimensionless source radial four-cycle multiplier `k`.
  `scale_status = source_reduced` (not homotopy, not physical observable).
- **Critical scale:** `k_c = (4/pi) log(1024/255) = 1.7700681326109957` (Float64).
- **Units:** dimensionless throughout; `g` and `H` have order-one natural scales.

## Tested revision and environment

- `CODE`: `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/issue-148-catastrophe-continuation`
- Tested code SHA: `792a02f`
- Accepted G0 base: `62c5a61`
- Julia: `1.12.6` (commit `15346901f00`), macOS arm64
- `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`
- Date: `2026-09-09`

## Focused checks at `792a02f`

### Regression tests (63/63 pass)

```
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. scripts/issue_148_g1_n5_regression_tests.jl
```

Status: passed (0). 63/63 assertions pass.

Key assertions:
- Near-cusp satellite `acos(-1/(4a(k)))` at `k=kc-1e-5` is rejected as `:pi`
  (`@test_throws ArgumentError`).
- Perturbed pi seed `pi+1e-3` at `k=kc-1e-5` converges to `:pi` branch,
  closer to pi than half the satellite distance.
- 256-bit `n5_reduced_critical_points(kc-1e-20)` returns 4 points and 2 minima.
- 256-bit `kc-1e-40` coordinate symmetry: both analytic satellites found with
  offsets `±1.253e-20` from pi, symmetry error `0`, and all 4 returned
  coordinates have gradient residual `< 1e-70`.
- `Rational{Int}` input to `n5_reduced_ratio`, `n5_reduced_exponent`, and
  `n5_reduced_critical_points` returns `BigFloat`, agreeing with Float64 to
  `1e-14`. `Rational{BigInt}` also returns `BigFloat`.
- All 49 previously passing assertions (event/tolerance/nonconvergence/
  rejection/BigFloat/branch identity) continue to pass.

### Replay checks

```
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl
```

Status: passed (0). Key observed values:

- `catastrophe_detected=true`, `catastrophe_index=2`
- `catastrophe_k=1.77006813256331207995`
- `catastrophe_theta=3.141592653589793116`
- `catastrophe_residual=4.586e-27`
- `catastrophe_hessian=3.745e-11`
- `catastrophe_scale_error=4.768e-11`
- `default_path_tight_event_detected=false` (one-iteration coarse path)
- `nonconverged_first_step_converged=false`
- `near_event_nonconv_catastrophe=false`
- `endpoint_scale_catastrophe_detected=false`
- `invalid_near_satellite_seed_rejected=true`
- `invalid_farsatellite_seed_rejected=true`
- `near_cusp_satellite_rejected=true` (`near_cusp_satellite_err=ArgumentError`)
- `near_cusp_satellite_theta=3.13762933148014999674`
- `near_cusp_satellite_distance_from_pi=0.00396332210964311926`
- `pi_near_cusp_branch=pi`, `pi_near_cusp_converged=true`
- `pi_near_cusp_theta=3.14159315229535307523`
- Precision ladder event errors: Float64 `4.768e-11`, 128-bit `4.768e-11`,
  256-bit `4.768e-11` (stable)
- Precision ladder event gradients: Float64 `4.586e-27`, 128-bit `7.052e-50`,
  256-bit `4.108e-88` (improving)
- `hp_256_kc_minus_1e20_points=4`, `hp_256_kc_minus_1e20_minima=2`
- `hp_256_satellite_offset=1.253314e-10`
- `hp_256_kc_minus_1e40_points=4`, `hp_256_kc_minus_1e40_minima=2`
- `hp_256_kc_minus_1e40_lower_offset=-1.253314e-20`
- `hp_256_kc_minus_1e40_upper_offset=1.253314e-20`
- `hp_256_kc_minus_1e40_symmetry_error=0.000000e+00`
- `hp_256_kc_minus_1e40_upper_found=true`, `hp_256_kc_minus_1e40_lower_found=true`
- `hp_256_kc_minus_1e40_max_gradient=1.170419e-97`
- `rational_int_ratio_type=BigFloat`, `rational_int_agrees=true`
- `rational_int_crit_type=BigFloat`
- `validation_checks_complete=true`

### Whitespace and diff-check

```
git diff --check HEAD
```

Status: passed (clean).

## Implementation deltas at `792a02f`

Cumulative changes from G0 (`62c5a61`):

- `src/paper_benchmarks/poly102_inflation.jl`
  - **Target-precision satellite construction:** `n5_reduced_critical_points`
    constructs `two_pi = convert(T, 2) * convert(T, π)` at working type instead
    of using bare `2π` (Float64). At 256 bits and `kc-1e-40`, both analytic
    satellites are found with symmetric offsets `±1.253e-20` from pi and gradient
    residuals `≤1.17e-97`. Previously the Float64 `2π` error (`~2.45e-16`)
    displaced the upper satellite and produced a nonstationary coordinate.
  - **Rational-input handling:** `_n5_numeric_type` helper maps `Rational` types
    to `BigFloat` (current precision) instead of narrowing to `Float64`. Applied
    to `n5_reduced_ratio`, `n5_reduced_exponent`, `n5_reduced_critical_points`,
    `_n5_type_of_collection`, `_n5_validate_zero_phase_seed`, and internal
    gradient/hessian/predictor functions. Float64 and BigFloat behavior unchanged.
  - **High-precision atol default:** `n5_reduced_critical_points` default
    tolerance computed as `64eps(_n5_numeric_type(typeof(k)))`, matching
    working precision. Float64 behavior identical.
  - **Branch validation fix:** replaced vacuous
    `closest_index == expected_index || closest_distance <= tolerance` with
    `expected_distance <= tolerance`, measuring distance from converged theta to
    the expected branch's critical point. The satellite at `k=kc-1e-5` (distance
    0.00396 from pi, tolerance 0.00198) is correctly rejected.
  - Prior `0cf16ca` deltas remain: exact source-critical formula, target-precision
    construction, continuation entry validation, converged-neighbor event
    requirement, scale/residual localization, seed validation, and intrinsic
    branch identity.

- `scripts/issue_148_g1_n5_regression_tests.jl`
  - 63 assertions (was 40 at G0). Adds near-cusp satellite rejection, perturbed
    pi seed acceptance, 256-bit kc-1e-20 classification, 256-bit kc-1e-40
    coordinate symmetry and stationarity, and Rational{Int}/Rational{BigInt}
    type checks.

- `scripts/issue_148_g1_replay_checks.jl`
  - Adds near-cusp satellite, perturbed pi seed, 256-bit kc-1e-20 and kc-1e-40
    coordinate/stationarity output, and rational-input type checks.

## Tolerance rationale

Along the exact `pi` branch, `H(pi,k) = -1 + 4a(k)` and `dH/dk|_{kc} = -pi/4`.
An event bracket of `1e-10` implies an order-`7.85e-11` Hessian localization
scale, consistent with the observed `3.745e-11`.

- `gradient_tolerance=1e-10` and `hessian_tolerance=1e-10` are continuation
  residual criteria against absolute gradient and fold detection magnitudes.
  These are dimensionless quantities with order-one natural scales; Float64
  roundoff (`2.22e-16`) is about `4.5e5` times smaller than `1e-10`.
- `event_scale_tolerance=1e-10` (strict `1e-14` in adversarial boundary probes)
  is an explicit bracket-width criterion preventing coarse single-step false
  positives.
- The precision ladder is built at target precision for each run, with source
  values constructed at working precision inside `setprecision` blocks.
  128- and 256-bit runs show stable event width and Hessian with improving
  gradient (`4.586e-27` → `7.052e-50` → `4.108e-88`).

## Known unrelated limitations (not changed in this task)

- `bin/audit.jl` failure: N8 JET reports and Revise `EMFILE`; outside N5 scope.
- `Pkg.test()` phase-volume detuning scan assertion: pre-existing baseline
  failure (`test/runtests.jl:1267`).
