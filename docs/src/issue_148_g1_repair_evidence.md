# Issue 148 G1 repair evidence

Scope: N5 zero-phase source-reduced continuation only; no N8 changes and no
physical-normalization, scientific-schema, or persisted-data updates. The change
adds the N5 continuation function and result type (additive N5 API); it does not
remove or change existing public interfaces.

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
- Tested code SHA: `c6345b0`
- Accepted G0 base: `62c5a61`
- Julia: `1.12.6` (commit `15346901f00`), macOS arm64
- `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`
- Date: `2026-09-09`

## Focused checks at `c6345b0`

### Regression tests (49/49 pass)

```
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. scripts/issue_148_g1_n5_regression_tests.jl
```

Status: passed (0). 49/49 assertions pass.

Key new assertions:
- Near-cusp satellite `acos(-1/(4a(k)))` at `k=kc-1e-5` is rejected as `:pi`
  (`@test_throws ArgumentError`).
- Perturbed pi seed `pi+1e-3` at `k=kc-1e-5` converges to `:pi` branch,
  closer to pi than half the satellite distance.
- 256-bit `n5_reduced_critical_points(kc-1e-20)` returns 4 points and 2 minima
  (previously returned 2 points/1 minimum with fixed Float64 default atol).
- All 40 previously passing assertions (event/tolerance/nonconvergence/
  rejection/BigFloat) continue to pass.

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
- `near_cusp_satellite_theta=3.13762933148014999674`
- `near_cusp_satellite_distance_from_pi=0.00396332210964311926`
- `near_cusp_satellite_rejected=true`
- `near_cusp_satellite_err=ArgumentError`
- `pi_near_cusp_branch=pi`, `pi_near_cusp_converged=true`
- `pi_near_cusp_theta=3.14159315229535307523`
- Precision ladder event errors: Float64 `4.768e-11`, 128-bit `4.768e-11`,
  256-bit `4.768e-11` (stable)
- Precision ladder event gradients: Float64 `4.586e-27`, 128-bit `7.052e-50`,
  256-bit `4.108e-88` (improving)
- `hp_256_kc_minus_1e20_points=4`, `hp_256_kc_minus_1e20_minima=2`
- `hp_256_satellite_offset=1.253314e-10`
- `validation_checks_complete=true`

### Whitespace and diff-check

```
git diff --check HEAD
```

Status: passed (clean).

### Package tests

```
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

Status: 10/11 pass, 1 fail (pre-existing). The single failure is the
phase-volume detuning scan Hessian assertion at `test/runtests.jl:1267`, a
known baseline limitation unrelated to the N5 continuation scope.

## Implementation deltas at `c6345b0`

- `src/paper_benchmarks/poly102_inflation.jl`
  - **Branch validation fix:** replaced vacuous
    `closest_index == expected_index || closest_distance <= tolerance` with
    `expected_distance <= tolerance`, where `expected_distance` is the periodic
    distance from the converged theta to the expected branch's critical point.
    The satellite at `k=kc-1e-5` (distance 0.00396 from pi, tolerance 0.00198)
    is now correctly rejected; the pi seed (distance ~5e-7) is accepted.
  - **High-precision atol default:** `n5_reduced_critical_points` default
    tolerance changed from `64eps(Float64)` to `64eps(typeof(float(k)))`.
    Float64 behavior is preserved identically. BigFloat inputs now use
    precision-appropriate tolerance, resolving the 256-bit classification
    boundary at `kc-1e-20`.
  - Prior `0cf16ca` deltas remain: exact source-critical formula, target-precision
    construction, continuation entry validation, converged-neighbor event
    requirement, scale/residual localization, seed validation, and intrinsic
    branch identity.

- `scripts/issue_148_g1_n5_regression_tests.jl`
  - Adds true near-cusp satellite rejection assertions, perturbed pi seed
    acceptance, and 256-bit critical-point classification checks.
    Total: 49 assertions (was 40).

- `scripts/issue_148_g1_replay_checks.jl`
  - Adds near-cusp satellite computed from `acos(-1/(4a(k)))` at `k=kc-1e-5`,
    perturbed pi seed at same scale, and 256-bit `kc-1e-20` classification
    output.

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
