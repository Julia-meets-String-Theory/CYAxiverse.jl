# Issue 148 G1 repair evidence

Scope: N5 zero-phase source-reduced continuation only; no N8 changes, no public API/schema edits, and no physical-normalization updates.

## Source conventions and tested revision

Source fixture remains from arXiv:2608.14780v1:

- Exact reduced ratio: `a(k) = (32/(255/8)) * exp[-2πk(32-255/8)]`
- Critical scale: `k_c = (4/π) log(1024/255) = 1.7700681326109957`
- Source PDF SHA-256: `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`

Git and environment for checks:

- `CODE`: `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/issue-148-catastrophe-continuation`
- Tested code SHA: `6014c4d9a909baa36f9b8e0e6e9a1930b4cb89b8` (current task branch)
- Julia: `1.12.6` (`1.12.6`, commit `15346901f00`), macOS arm64
- Environment: `JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia`
- Date: `2026-09-09`

## Focused checks run

- `python3 scripts/agent_verify.py snapshot`
  - status: `passed` (0)
  - log: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_28cq3670`
- `python3 scripts/agent_verify.py diff-check`
  - status: `passed` (0)
  - log: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_0cgl3tpa/diff_check.stdout.log`
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl`
  - status: `passed` (0)
  - stdout: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_0m63vnfr/run.stdout.log`
  - key observed values:
    - `catastrophe_detected=true`
    - `catastrophe_index=2`
    - `catastrophe_k=1.7700681325633121`
    - `|catastrophe_k-k_c|=4.79e-11`
    - `catastrophe_residual=4.586e-27`
    - `catastrophe_hessian=3.745e-11`
    - `catastrophe_scale_error=4.768e-11`
    - `default_path_2: branch=pi, hessian signs ±, theta=π`
    - `default_path_tight_event_detected=false` for `max_iterations=1` and `event_scale_tolerance=1e-14`
    - `nonconverged_first_step_converged=false`, `nonconverged_first_step_iterations=1`
    - `invalid_satellite_seed_rejected=true` for `seed_theta=3.101964568393247`
    - `mixed_type_path_eltype=BigFloat`
    - `ratio_mixed_max_error_256=0.00000e+00`
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_n5_regression_tests.jl`
  - status: `passed` (0)
  - stdout: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_w7ifvh8k/run.stdout.log`
  - result: `24/24` regression assertions passed.

## Implementation deltas

- `src/paper_benchmarks/poly102_inflation.jl`
  - Restores source-exact reduced-ratio formula to `32/(255/8)` prefactor and non-anchored exponent.
  - `_n5_reduced_zero_phase_catastrophe_event` now performs bounded bisection refinement with explicit `gradient_tolerance`, `hessian_tolerance`, `event_scale_tolerance`, and `max_iterations` controls.
  - Final-event metadata now checks refined catastrophe convergence against all declared tolerances.
  - Branch identity now comes only from validated seeds (`:pi` or `:zero`) and propagates through continuation; unsupported seeds throw.
  - `max_iterations` and tolerances are validated as positive finite values.
  - Mixed precision is preserved in working type promotion across `k`, seed, and tolerances.
- `test/runtests.jl` boundary testset adds focused N5 checks for:
  - continued source formula and minima-count change around `k_c`
  - catastrophe continuity and localized event metadata (`catastrophe_hessian`, `catastrophe_residual`, and adjacent-step propagation)
  - false-positive guard on coarse bracket refinement
  - nonconvergence and control-value failures
  - zero-branch path behavior and unsupported satellite seed rejection.
- Added tracked artifacts:
  - `scripts/issue_148_g1_replay_checks.jl`
  - `scripts/issue_148_g1_n5_regression_tests.jl`

## Tolerance rationale

- `gradient_tolerance=1e-10` and `hessian_tolerance=1e-10` are continuation residual controls for local Newton correctness and catastrophe classification.
- `event_scale_tolerance=1e-10` is required for a localization certificate; the replay shows event localization to `4.8e-11` in `k` with hessian `3.7e-11`.
- The observed `1e-14` one-step coarse bracket test is intentionally expected to return no detected catastrophe before refinement, showing non-exhaustive brackets are rejected by tolerance contract.
- `max_iterations=64` was used for the default pass; failure cases explicitly exercise `max_iterations=1` to confirm nonconvergence handling.

## Known unrelated limitation (not changed in this task)

The pre-existing full-package failure in the phase-volume detuning scan is unchanged: the local population scan expected `S.hessian(...) ≈ 4π^2 * I` but observed `2π^2 * I` at the recorded path. This was already identified before G1 and is outside N5 source-reduced continuation scope.
