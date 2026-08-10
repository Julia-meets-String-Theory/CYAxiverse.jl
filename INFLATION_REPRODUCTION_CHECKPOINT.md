# Inflation reproduction checkpoint

Date: 2026-08-06

This checkpoint records the state after pausing the inflation-reproduction work. It is intentionally stored in the requested branch worktree so the work can be resumed even if the surrounding session state is unavailable.

## Scope and authoritative instructions

- Branch: `agents/inflation-reproduction-instructions`
- Worktree: `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/inflation-reproduction-instructions`
- Session plan read from: `/Users/vmehta/.copilot/session-state/b5f1b531-fbe4-46ad-a196-2c65998660a3/plan.md`
- Plan status: items 1, 2, and 4 are marked done; item 3 is in progress.
- Remaining plan item 3: complete the corrected canonical/mass-basis trajectory comparison, validate the long high-precision Julia runs, and derive the remaining N=5 benchmark quantities. The plan specifically says that the short bounded run succeeds while the long runs and package suite remain pending.

The branch is already checked out in the sibling worktree above. The main `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl` worktree is on branch `vmm` and was not used for implementation changes during this continuation.

## Worktree state at pause

`git status --short --branch` in the requested worktree reported:

```text
## agents/inflation-reproduction-instructions
 M Project.toml
 M src/paper_benchmarks.jl
```

No commit was made during this continuation. These modifications include the other agent's pending solver work and the step-size patch added during this continuation. Do not discard them or reset the worktree.

The main worktree initially had an unrelated existing untracked `scripts/__pycache__/`; it was not touched.

## Existing pending implementation changes

`Project.toml` has an uncommitted `LinearSolve` dependency and compatibility entry.

`src/paper_benchmarks.jl` has the following pending implementation changes in the nested `author_inflation` module:

- imports `LinearAlgebra` and `LinearSolve`;
- slices the nine-component dense ODE state to its first eight components before passing it to the eight-field diagnostic helper;
- removes the premature callback-based termination that could stop at the first transient slow-roll crossing;
- supplies a `9 x 9` BigFloat Jacobian prototype to `ODEFunction`;
- explicitly selects `GenericLUFactorization()` for `Rodas5P`;
- makes the selected tolerances explicit in solver metadata;
- records accepted/rejected steps, RHS evaluations, and Jacobian evaluations;
- handles `sample_count == 1` without constructing a degenerate range.

The additional uncommitted patch made during this continuation adds the keyword `max_step::Real=100` to `n8_author_trajectory`, validates that it is positive, passes `min(tmax, max_step)` as `dtmax`, and records `max_step` in solver metadata. The purpose is to prevent a full-horizon dense solution from taking a giant first step through the stiff initial transient. This patch has not yet been scientifically validated.

## What was verified

### Environment

- Julia: `1.12.6`
- The branch package resolves `OrdinaryDiffEq 7.3.0` and `LinearSolve 5.5.0`.
- The branch's independent reference uses the miniforge/CYTools environment when Python is needed, but the Julia trajectory checks below did not require Python.

### Initial-condition agreement

For the 60-efold target detuning

```text
delta_k = 1.5320548620798324e-3
```

the Julia author trajectory's initial diagnostics at 64-bit BigFloat precision were:

```text
epsilon       = 0.0387533589227068442034
eta_parallel  = 26775.8619825140195871
potential     = 0.668452161273865690431
```

The independent reference JSON reports the same values to the displayed precision, and its initial theta agrees with the Julia theta. This rules out an obvious initial-coordinate, sign, or normalization mismatch.

### Bounded solver probe

Command:

```sh
julia --project=. --startup-file=no -e 'using CYAxiverse; B=CYAxiverse.paper_benchmarks; d=1.5320548620798324e-3; elapsed=@elapsed result=B.n8_author_trajectory(d; precision_bits=64, max_time=100, scan_step=5, sample_count=1, reltol=1e-8, abstol=1e-10, maxiters=1_000_000); println("elapsed=", elapsed); println("entered=", result.entered_slow_roll, " event=", result.end_event, " efolds=", result.efolds); println("solver=", result.solver)'
```

Before the `max_step` patch, this completed in about 6.1 seconds and returned:

```text
entered=false event=no_slow_roll_window efolds=0.0
accepted_steps=222 rejected_steps=70 rhs_evaluations=2560 jacobian_evaluations=222
```

This is expected for a physical-time horizon of only 100 and confirms that the solver returns successfully and uses the supplied Jacobian (`jacobian_evaluations` equals the accepted-step count in this run).

### Short-horizon physical behavior

A direct 10-unit ODE probe with the original full-horizon `dtmax=tmax` setting produced:

```text
t=0   V=0.668452161273865690431  eta=26775.8619825140195871  N=0.0
t=1   V=0.668451977411907527916  eta=26775.869554401686246   N=0.6684511935835843083
t=10  V=0.668451193580280127974  eta=-0.149087078980645594   N=6.6845119358358187412
```

Thus the correctly resolved short trajectory does enter the slow-roll window by approximately physical time 10. The problem is associated with resolving/scanning that initial stiff transient in a full-horizon solution, not with the basic RHS or initial state.

### Full-horizon run with an unrestricted solver step

Command parameters:

```text
delta_k=1.5320548620798324e-3
precision_bits=64
max_time=1_000_000
reltol=1e-10
abstol=1e-12
```

With the previous `dtmax=tmax` implementation, the run took about 61 seconds:

```text
entered=false event=no_slow_roll_window efolds=0.0
accepted_steps=620 rejected_steps=100 rhs_evaluations=6382 jacobian_evaluations=620
```

A direct sample of that solution showed an invalid dense interpolation across the giant initial step:

```text
t=1       N=6.1518482988124712e-05
t=10      N=0.0006151848298812471
t=100     N=0.006151848298812471
t=1000    N=0.06151848298812471
t=1e6     N=61.51848298812471
```

This contradicted the short-horizon result (`N(1)≈0.668`) and established that `dtmax=tmax` cannot be used with the dense-output event scan.

### Full-horizon run with `max_step=100`

After adding `max_step=100`, the same controlled-tolerance full run took about 96 seconds:

```text
entered=false event=no_slow_roll_window efolds=0.0
accepted_steps=10613 rejected_steps=100 rhs_evaluations=96319 jacobian_evaluations=10613
```

It still did not detect entry. The leading hypothesis is that the first 100-unit step remains too coarse for the very early stiff transient, so the dense scan at the author scan interval of 5 does not see the short negative crossing. A test with `max_step=5` was requested but was interrupted before execution.

## Independent numerical references

`validation/author_poly102_reference_miniforge.json` contains the independent machine-precision transcription results:

```text
delta_k=1e-7:              N_e=464213.5708051051
delta_k=0.00153205486208:  N_e=59.690642055250756
```

Both canonical-Hessian and mass-eigenbasis entries in that JSON agree for the two detunings. The supplied paper/reference targets are approximately `463115` and `60`, respectively. The independent result is explicitly machine-precision and uses a final-finite-exit policy because transient crossings occur in Float64; it is not a definitive high-precision replacement.

## Package test result at pause

The first attempt to run the suite was blocked by the sandbox when Julia tried to create:

```text
/Users/vmehta/.julia/logs/manifest_usage.toml.pid
```

The same command was then rerun with the required permission:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Precompilation completed successfully. The suite then reported 48 passed, 7 failed, and 2 errors. The failures are in the inflation reproduction contract block:

- `n8_degenerate_point()` returns `k=0.6745063700033533`, while the later contract expects the supplied constant `0.674506370003365` at `atol=1e-15`.
- The top-level `n8_inflation_initial_condition` implementation returns no `canonical_norm` for its default canonical-basis path, causing two errors.
- Its mass-basis path reports the old draft hilltop/basis coordinates and catastrophe `k`, rather than the later contract's `N8_BEST_X` and detuned `basis_k`.
- The top-level `n8_hilltop_efolds` returns the old local-form value `6818.662123768784` at `delta_k=1e-7`, while the later contract expects the author-normal-form value near `463115`.
- The same function returns `1.2924063410124793` at the 60-efold detuning, while the later contract expects approximately 60.
- The nested `n8_hilltop_probe` uses the author-normal-form implementation, so its e-fold result does not match the still-old top-level `n8_hilltop_efolds`.
- The top-level `n5_critical_scale()` returns the printed closed-form value `1.7700681326109957`, while the later author benchmark contract expects `0.674506370003365`.

The source contains two overlapping APIs: an older top-level draft benchmark implementation and a newer nested `author_inflation` implementation. The test file also contains expectations for both conventions. Do not blindly change constants to make the tests pass; first decide whether the public API should preserve both namespaces, rename the legacy functions, or update the conflicting tests/documentation to distinguish the draft local model from the author trajectory model.

Other test groups passed, including geometry, package loading, vacua persistence, spectrum, paper reproduction, and most benchmark tests. The complete suite is not green at this checkpoint.

## Exact next steps when resuming

1. Check for a surviving Julia process before launching anything expensive. The sandbox denied `ps` during the pause, so use an approved process listing or the session's process controls. Do not kill an unrelated Julia/CYTools process.
2. Run the bounded author trajectory with the newly added step control:

   ```sh
   julia --project=. --startup-file=no -e 'using CYAxiverse; B=CYAxiverse.paper_benchmarks; d=1.5320548620798324e-3; result=B.n8_author_trajectory(d; precision_bits=64, max_time=1_000, scan_step=5, max_step=5, sample_count=2, reltol=1e-8, abstol=1e-10, maxiters=10_000_000); println(result.entered_slow_roll, " ", result.end_event, " ", result.efolds); println(result.solver)'
   ```

   This command was attempted but rejected/interrupted before it ran; no result is available.
3. If `max_step=5` detects entry, inspect whether the full run can use a two-phase strategy: a small step cap through the initial transient followed by a larger cap after entry, while retaining the scan-step event policy and dense-output bisection.
4. Validate the 60-efold target first, then the tuned `delta_k=1e-7` target. Start with moderate precision/tolerances and only then run the requested 100-bit defaults.
5. Compare entry/exit event type, e-fold count, initial diagnostics, and samples against `validation/author_poly102_reference_miniforge.json` before changing checked-in fixtures or `validation/inflation_comparison.csv`.
6. Resolve the old-vs-author public API/test conflict deliberately. The old test block expects the legacy local normal form (`~6819` e-folds for the tuned detuning); the later inflation reproduction block expects the author normal form (`~463115`). These are different scientific quantities and should be named/documented distinctly rather than silently conflated.
7. Run the complete package suite again after the API decision and solver validation. Only update the session plan and commit after the results are reproducible.

## Important safety notes

- Do not reset or clean the requested worktree; it contains the other agent's uncommitted implementation.
- Do not overwrite the existing validation fixture or comparison CSV with unvalidated nonlinear results.
- The main `vmm` worktree is separate from the requested branch worktree.
- The checkpoint itself is currently uncommitted by design so it remains visible as part of the paused worktree state.

## Continuation update after resuming

The requested worktree was resumed on 2026-08-06. The following changes and
checks supersede the pending items above.

### Solver changes now present

`n8_author_trajectory` now has the following behavior:

- `Rodas5P` is the only supported method. The experimental `FBDF` branch was
  removed because its dense output was not reliable for this autonomous
  BigFloat trajectory; passing `method=:FBDF` now raises an `ArgumentError`.
- `max_step` defaults to `100` and `initial_step` defaults to `1e-5`. Both
  must be positive and are recorded in solver metadata.
- The ODE includes an explicit zero `tgrad` callback. The equations are
  autonomous, so this prevents OrdinaryDiffEq from estimating an unnecessary
  finite-difference time derivative at every Rosenbrock step.
- The analytic 9-by-9 Jacobian, `jac_prototype`, and
  `GenericLUFactorization()` are retained. The nine-component state is sliced
  to eight fields whenever the physical diagnostic helper is called.
- Saved accepted solver states are used for slow-roll sign scanning. Dense
  interpolation is used only for bisection after a bracket is found. This
  avoids the former fixed-`scan_step` cost of hundreds of thousands of
  BigFloat Hessian evaluations on the tuned long run.
- All bracketed slow-roll windows are collected and the longest completed
  window is selected. This avoids returning the short transient window found
  first for `delta_k=1e-7` and matches the independent final-finite-exit
  reference policy.
- For a successful trajectory, `efolds` is the total e-fold coordinate at the
  selected exit (`end_n`). `slow_roll_efolds` is the duration of that selected
  window (`end_n-entry_n`). The no-window return includes both fields as zero.

### Numerical validation completed

The following 60-efold target was used repeatedly:

```text
delta_k = 1.5320548620798324e-3
```

At 64-bit BigFloat precision with `max_time=100`,
`reltol=1e-10`, `abstol=1e-12`, `max_step=100`, and `initial_step=1e-5`,
the result is:

```text
entered_slow_roll = true
entry_n           = 0.45447190566306507227
end_n / efolds    = 60.0034473518028820954
slow_roll_efolds  = 59.5489754461398170249
end_event         = :eta_parallel
accepted_steps    = 576
rejected_steps    = 96
rhs_evaluations   = 5376
jacobian_evals    = 576
```

The corresponding 80-bit run completed in about 7 seconds and returned
`efolds=59.548976808777578...` when the pre-convention-change interval value
was reported, agreeing with the 64-bit trajectory to practical tolerance.
The initial diagnostics remain consistent with the independent reference:
epsilon `0.038753358922706844...`, eta-parallel
`26775.861982514019...`, and potential `0.668452161273865690...`.

For the tuned case

```text
delta_k = 1e-7
```

at 64-bit precision with `max_time=1_000_000` and the same tolerances, the
solver completed in about 92 seconds. The final-window scan returned:

```text
entry_n          = 412759.9736084036265
end_n / efolds   = 464281.2900479779730
slow_roll_efolds = 51521.3164395743465
end_event        = :eta_parallel
accepted_steps   = 10586
rejected_steps   = 113
rhs_evaluations  = 85592
jacobian_evals   = 10586
```

The absolute endpoint is close to the independent machine-precision value
`464213.5708051051` and the supplied paper target `463115` (within 1%). The
first-window policy incorrectly returned only `160.447...` efolds; the
longest-window/total-endpoint policy fixes that selection.

Two 100-bit attempts with the default or very tight tolerances were
intentionally interrupted after several minutes because BigFloat Hessian and
matrix arithmetic remained CPU-bound. No numerical result from those runs is
being treated as validated. The practical arbitrary-precision check is the
successful 80-bit run above.

### N=5 and API decision

The author namespace checks now return:

```text
author_inflation.n5_critical_scale()                 = 0.674506370003365
author_inflation.n5_reduced_ratio(k_c)                = 0.25
author_inflation.n5_hilltop_efolds(1e-7).efolds       = 27349.0
author_inflation.n5_hilltop_efolds(6.65e-5).efolds    = 60.00000000000001
```

The source intentionally preserves the older top-level draft/local-form API
and the newer author-normal-form API under `paper_benchmarks.author_inflation`.
The two models have incompatible definitions and produce different N=8 and
N=5 critical scales, so the inflation reproduction contract tests now use the
explicit author namespace; the earlier draft tests continue using the legacy
top-level functions.

### Test and worktree state

After the namespace update, the complete suite passed. The final successful
run reported all test groups passing, including the formerly conflicting
inflation reproduction contract and the full package/pipeline/spectrum groups.
Pipeline tests still print the pre-existing duplicate-include warnings and a
deliberately exercised temporary `KeyError` failure path, but their assertions
pass.

Current expected worktree status is:

```text
 M Project.toml
 M docs/src/api.md
 M scripts/inflation_reproduction.jl
 M scripts/reproduce_2023_n8.jl
 M src/CYAxiverse.jl
 M src/paper_benchmarks.jl
 M test/runtests.jl
?? INFLATION_REPRODUCTION_CHECKPOINT.md
```

No commit has been made. The validation fixture and comparison CSV were not
modified. The next safe action is a review of the naming diff followed by the
physical source-file split; do not reset or clean this worktree.

## Scientific naming migration

A non-breaking scientific naming layer was added. The canonical entry points
are now:

```text
CYAxiverse.axion_benchmarks
CYAxiverse.axion_benchmarks.poly102_inflation
poly102_inflation.n8_physical_gradient_flow
poly102_inflation.n8_hilltop_normal_form_efolds
poly102_inflation.n8_hilltop_normal_form
poly102_inflation.n8_efold_gradient_flow
poly102_inflation.n5_hilltop_normal_form_efolds
```

At the parent benchmark namespace, `n8_local_hilltop_efolds` identifies the
legacy reduced N=8 local model, while
`n8_hilltop_normal_form_efolds` identifies the poly-102 normal-form quantity.
The historical `paper_benchmarks`, `author_inflation`, `n8_author_trajectory`,
and related names remain available as compatibility aliases. This avoids a
silent scientific API break while removing ambiguity for new callers.

The inflation reproduction script, API documentation, and author-contract
tests now use the scientific names. The complete package suite was rerun after
this migration and passed. The physical source split into separate files is
still the next structural refactor; the aliases were deliberately added first
so that the file move can be tested independently.

## Physical source split completed

The requested physical split was completed on 2026-08-06. The public
CYAxiverse.axion_benchmarks binding and legacy paper_benchmarks module name
are unchanged; src/paper_benchmarks.jl is now a 19-line façade that imports
the component files in dependency order:

    src/paper_benchmarks.jl                         façade/module definition
    src/paper_benchmarks/reduced_models.jl          legacy reduced-model functions
    src/paper_benchmarks/poly102_inflation.jl       Poly102/author inflation module
    src/paper_benchmarks/compatibility.jl           public compatibility aliases/helpers

The reduced-model file supplies the parent-module functions and shared
_LOG10E constant. The Poly102 file retains the nested implementation module
for compatibility while exposing the scientific poly102_inflation alias. The
compatibility file contains the parent-level aliases and helper wrappers. The
include order is intentional: the component definitions are loaded before the
aliases that refer to them.

Structural checks:

- git diff --check passed.
- Component line counts are 747 (reduced_models.jl), 819
  (poly102_inflation.jl), and 120 (compatibility.jl); the façade is 19
  lines.
- The main worktree's unrelated scripts/__pycache__/ remained untouched.
- No commit was made, and the validation fixture/comparison CSV remain
  unchanged.

Smoke command:

    julia --project=. --startup-file=no -e 'using CYAxiverse; B = CYAxiverse.axion_benchmarks; P = B.poly102_inflation; @assert isdefined(B, :n8_physical_gradient_flow); @assert isdefined(P, :n8_physical_gradient_flow); @assert B.n8_hilltop_efolds(1e-7).efolds > 0; @assert P.n5_hilltop_normal_form_efolds(1e-7).efolds > 0; println("split smoke test passed")'

Result: split smoke test passed.

Full verification:

    julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'

Result: Testing CYAxiverse tests passed. All reported test groups passed,
including 76 paper-reproduction benchmark assertions, 57 pipeline persistence
assertions, and the inflation reproduction contract. The existing pipeline
duplicate-include warnings and intentionally exercised temporary KeyError
failure path still appear, but their assertions pass.

The physical split is therefore complete. Any follow-up work should focus on
reviewing the uncommitted diff or committing this branch; do not reset or clean
the worktree.

## Discrepancy investigation update (2026-08-07)

The N=8/poly-102 mismatch has been narrowed down. For
`delta_k=1.5320548620798324e-3`, the corrected Julia physical-time BigFloat
flow converges as follows (`Rodas5P`, 64-bit BigFloat, `reltol=1e-10`,
`abstol=1e-12`):

```text
max_step=10   total N=60.00344735   slow-roll duration=59.54897545
max_step=5    total N=60.00337404   slow-roll duration=59.54890213
max_step=2    total N=60.00336060   slow-roll duration=59.54888869
max_step=1    total N=60.00335883   slow-roll duration=59.54888693
max_step=0.5  total N=60.00335817   slow-roll duration=59.54888627
```

The independent SciPy value `59.690642055250756` was generated with
Float64 BDF, `max_step=100`, and dense-output scanning across the stiff initial
transient. Re-running the independent transcription with smaller step caps
moves the total e-fold count toward `60.002–60.003`, matching Julia and the
paper's approximately-60 target. The published `59.69064` value is therefore
a solver/interpolation artifact, not a coordinate or mass-basis disagreement.

The nearly-zero hilltop eigenvalue makes Float64 eigenvector components
ill-conditioned. Nevertheless, passing the SciPy-derived direction and the
BigFloat-derived direction separately through the same Julia physical solver
gave `60.0033581647` and `60.0033581718`, respectively. This rules out the
basis-vector choice as the source of the discrepancy.

The remaining meaningful difference is between the local normal-form estimate
(`63.85738654`) and the full nonlinear physical flow (`60.00335817`). This is
an approximation/model-scope difference: the local normal form omits the
resolved heavy-mode transient. Keep both values, with their methods named
explicitly, and do not use the SciPy `59.69064` number as a high-precision
reference.
