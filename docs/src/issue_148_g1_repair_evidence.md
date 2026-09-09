# Issue 148 G1 repair evidence

Scope: N5 zero-phase source-reduced continuation only; no N8 changes, no public API/schema edits, and no physical-normalization updates.

## Source conventions and tested revision

Source fixture remains from arXiv:2608.14780v1:

- Exact reduced ratio: `a(k) = (32/(255/8)) * exp[-2πk(32-255/8)]`
- Critical scale: `k_c = (4/π) log(1024/255) = 1.7700681326109957`
- Source PDF SHA-256: `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`

Git and environment for checks:

- `CODE`: `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/issue-148-catastrophe-continuation`
- Tested code SHA: `0cf16ca`
- Julia: `1.12.6` (`1.12.6`, commit `15346901f00`), macOS arm64
- `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`
- Date: `2026-09-09`

## Focused checks run

- `python3 scripts/agent_verify.py snapshot`
  - status: `passed` (0)
  - log dir: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_a53r8by2`
- `python3 scripts/agent_verify.py diff-check`
  - status: `passed` (0)
  - log: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_jqx02uxm/diff_check.stdout.log`
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl`
  - status: `passed` (0)
  - stdout: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_vtpe_xg0/run.stdout.log`
  - key observed values:
    - `catastrophe_detected=true`
    - `catastrophe_index=2`
    - `catastrophe_k=1.77006813256331207995`
    - `catastrophe_theta=3.141592653589793116`
    - `catastrophe_residual=4.586e-27`
    - `catastrophe_hessian=3.745e-11`
    - `catastrophe_scale_error=4.768e-11`
    - `default_path_tight_event_detected=false` for one-iteration coarse path
    - `nonconverged_first_step_converged=false`, `nonconverged_second_step_converged=false`
    - `near_event_nonconv_catastrophe=false`
    - `endpoint_scale_catastrophe_detected=false`
    - `invalid_near_satellite_seed_rejected=true`
    - `invalid_farsatellite_seed_rejected=true`
    - `precision_ladder_0_catastrophe=2`
    - `precision_ladder_0_event_scale_error=4.768e-11`
    - `precision_ladder_128_event_scale_error=4.768e-11`
    - `precision_ladder_256_event_scale_error=4.768e-11`
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_n5_regression_tests.jl`
  - status: `passed` (0)
  - stdout: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_898yf3zy/run.stdout.log`
  - result: `40/40` regression assertions passed
- Python-free import check:
  - `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. -e 'using CYAxiverse; println("python_free_import_ok")'`
  - status: `passed` (0)
  - stdout: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_0lqrgr2g/run.stdout.log`
- `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. bin/audit.jl`
  - status: `failed` (1)
  - stdout: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_cjldqwrd/run.stdout.log`
  - stderr: `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_cjldqwrd/run.stderr.log`
  - cause: existing `JET found 2 report(s) in CYAxiverse` plus repeated `FileWatching: too many open files (EMFILE)` from `Revise`; unrelated to `N5` fix scope

## Implementation deltas

- `src/paper_benchmarks/poly102_inflation.jl`
  - Restores exact source-critical formula with target-precision construction and removes the `N5_REDUCED_CATASROPHE_REFINE_ITERATIONS` typo in favor of `N5_REDUCED_CATASTROPHE_REFINE_ITERATIONS` while preserving the backward-compatible alias used elsewhere.
  - `n5_reduced_critical_points` now builds critical point containers in a promoted target type, including branch-anchor theta points.
  - Catastrophe detection now:
    - enforces `max_iterations > 0`, `gradient_tolerance > 0`, `hessian_tolerance > 0`, and `event_scale_tolerance > 0` in continuation entry
    - runs finite bracketing refinement only when neighboring steps are converged,
    - requires both residual and scale to satisfy `event_scale_tolerance` to mark any candidate event converged,
    - propagates failures (`catastrophe_detected=false`) for nonconverged localization paths.
  - Branch validation is now intrinsic and stateful:
    - seed must be near `0` or `π` only,
    - non-`π`/`zero` continuation support raises `ArgumentError`,
    - unsupported near-cusp seed (`3.101964568393247`) is rejected in both near- and far-window probes.

- `scripts/issue_148_g1_replay_checks.jl`
  - Adds direct checks for seed rejection, nonconvergence boundaries, endpoint-scale rejection, real catastrophe booleans, and precision-ladder tracking at `0`, `128`, and `256` bits.
- `scripts/issue_148_g1_n5_regression_tests.jl`
  - Adds hard assertions for the rejection and nonconvergence cases above plus closed-form `k_c` ladder checks at `Float64`, `BigFloat(128)`, and `BigFloat(256)`.

## Tolerance rationale

- `gradient_tolerance=1e-10` and `hessian_tolerance=1e-10` are continuation residual criteria against absolute gradient and fold detection magnitudes.
- `event_scale_tolerance=1e-10` (strict 1e-14 in the adversarial boundary probe) is an explicit bracket-width criterion; it prevents coarse single-step false positives from being accepted.
- Observed adversarial ladder values show stable event width (`~4.768e-11`) and hessian (`3.745e-11`) across bit precision, while endpoint-only and one-iteration coarse probes remain unaccepted.
- The precision ladder is built at target precision for each run (`Float64`, `BigFloat(128)`, `BigFloat(256)`), avoiding `Float64` pre-computation promotion.

## Known unrelated limitation (not changed in this task)

- `bin/audit.jl` failure remains unchanged outside the current G1 scope (N8 JET/`Revise` issues above).
- The manager previously reported full suite phase-volume assertion mismatch as a pre-existing baseline failure; full-suite rerun is intentionally not repeated here.
