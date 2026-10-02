# CYAX-0131 trajectory diagnostic report

Status: bounded diagnostics only. The owner-terminated full replay remains
`OWNER_TERMINATED` with numerical outcome `UNESTABLISHED`. These probes do not
promote a trajectory, establish an e-fold result, or satisfy trajectory gates.

## Run identity and exact command

The probe script was reviewed against the current author API and then updated
only to flush a runtime record, tolerance record, calibration start/result,
and each probe start/result. No package source, model, solver, production
tolerance, or packet acceptance setting changed. Probe script SHA-256:
`b95a8aabeb22713fc9ec8ae2c44632a78e548552594f4168db8ac126c71ba725`.
The imported trajectory source SHA-256 was
`578a6b44877f82082f50699dedc2830bc2d34f4efedc93394e3b2b255ac9a564`.

Executed from repository root:

```sh
repo_root="$(git rev-parse --show-toplevel)"
p0_env="${repo_root}/validation/p0_numerical_equivalence/environment"
JULIA_LOAD_PATH="@:${p0_env}:@stdlib" \
  julia --startup-file=no --project=. \
  validation/cyax_0131_low_n_inflation/trajectory_default_probe.jl \
  2>&1 | tee validation/cyax_0131_low_n_inflation/trajectory-default-probe.raw.log
```

Julia reported version 1.12.6, default compilation, and one runtime thread.
The command did not set a thread override. It set only the displayed
`JULIA_LOAD_PATH` stack: candidate project, existing pinned P0 environment,
then standard libraries. Both probes completed inside one process; total
wall time was 26.8 seconds and exit status was 0. No full replay or other
probe process remains active. The raw log SHA-256 is
`3bc615177740d9ad0ee1d454460723b5231e64858fe732805e62eb3d24b57694`; do not
stage it because the raw solver warning includes a host-local package path.
The sanitized record is `trajectory-default-probe.sanitized.log`.

## Frozen diagnostic configuration

The route recomputed `author_inflation.n8_row2_phase_catastrophe()` once and
used its refined `critical_k`, refined basis point, phase vector, and author10
model for both calls. The shared calibration completed at 128 bits in
9.014349333 seconds and allocated 6,791,387,072 bytes. Its phase vector was
`[0, 0.04, 0, 0, 0, 0, 0, 0, 0, 0]`; refined `critical_k` was
`0.5080234603138255193960987179021912891477`.

The source uses two representations of the nominal row-2 phase: continuation
and returned `phase_vector` use `Float64(0.04)`, while the BigFloat refinement
sets the target with `BigFloat("0.04")` inside its 128-bit precision block.
The stationary branch and trajectory probes receive the returned Float64
vector, which the trajectory converts to BigFloat; those phase bits are not
identical to the decimal BigFloat refinement input. This report preserves that
distinction and makes no exact same-phase catastrophe-to-trajectory identity
claim. It does not change the normative phase convention (0.04 radians) or
invent a tolerance. Since the run/probes produced no eligible physical
trajectory and no physical diagnostics are claimed, this encoding detail does
not promote or invalidate a physical result. Any later physical diagnostic
must first use a consistently recorded phase encoding through refinement and
trajectory evaluation.

Each trajectory used `delta_k=1.5320548620798324e-3`, displacement `1e-8`,
sign `-1`, `basis=:canonical_hessian`, the refined `basis_theta`, Rodas5P,
100-bit BigFloat precision, source-default tolerances, simulated horizon 10,
`maxiters=1000`, `max_step=100`, `initial_step=1e-5`, `scan_step=5`,
`sample_count=2`, `save_everystep=true`, and `dense=true`. No parameter was
changed between first and repeated probes.

## Measurements

At 100-bit precision the source formula evaluated to `reltol=1e-50` and
`abstol=1.0000000000000000000000000000002e-66`. Runtime `eps(BigFloat(1))`
was `1.5777218104420236108234571305656e-30`; tolerance-to-unit-spacing
ratios were `6.3382530011411470074835160268802e-21` and
`6.3382530011411470074835160268813e-37`. These ratios compare requested
tolerances with spacing at unit scale. They are not a solver acceptance rule
or proof that the inherited tolerances are invalid.

| Measurement | First probe | Same-process repeat |
|---|---:|---:|
| Wall time | 6.024014 s | 2.666163875 s |
| GC time | 0.246801541 s | 0.175834336 s |
| Allocated bytes | 4,590,881,560 | 3,791,734,040 |
| Returned summary object | 15,752 bytes | 15,752 bytes |
| Retcode | `MaxIters` | `MaxIters` |
| Accepted / rejected steps | 922 / 78 | 922 / 78 |
| RHS / Jacobian evaluations | 8,000 / 922 | 8,000 / 922 |
| Entered slow roll | no | no |
| E-fold fields | zero; no window | zero; no window |

The configured simulated horizon was 10, but each solve stopped at its
1,000-iteration cap. The API returns no terminal simulation time on this
path: the returned summary has no `t`/endpoint field, and the local ODE
solution is not retained in the result. The function returns solver retcode
and counters; because no slow-roll window was found before the cap, it returns
an empty sample list and no `entry_n` or `end_n`. Therefore actual simulated
time is unavailable. The returned zeros are the function's no-window summary,
not measured physical e-folds.

The repeat retained a large allocation volume after the first same-process
call. Its shorter elapsed time does not isolate compilation as the reason;
the probe does not separately measure compilation. The return object is small,
so total allocations should not be confused with retained memory. Both runs
have the same cap and solver counters and cannot establish convergence or
slow-roll behavior.

## Interpretation and next validation

The full-run interrupt traceback independently placed the active ODE in
Rosenbrock `perform_step!`, LinearSolve generic LU, BigFloat arithmetic, and
GC. Process-wide exit totals were 112,801,083,728 allocations and 3,137 GCs;
short main-thread samples were dominated by GC and process footprint reached
14.2 GB. Together with 3.79–4.59 GB allocated by each capped 1,000-iteration
probe, the evidence supports substantial transient allocation and GC pressure
in this BigFloat/Generic-LU route. It does not quantify which solver operation
caused the full-run total, retained-state memory, or a completed trajectory's
runtime. The full run was interrupted by owner decision after 101m51s elapsed
and 99m51s CPU; that is not a solver verdict and provides no grounded ETA.

The recovered pre-trajectory replay stages completed: zero-phase N8
calibration; all 400 row-2 phase steps; all 12 stationary points; N5 shifted
fold checks; and all seven P96 cross-check points. The full physical
trajectory has no recovered time, solver counters, final `N_e`, or outcome.
Later gates remain `NOT_REACHED` or `UNESTABLISHED` as applicable. No
observational quantity was produced from either bounded probe.

Practical next validation: retain the packet defaults until the owner decides
otherwise. The bounded calibration/null evidence can be reviewed without a
completed physical trajectory when no physical diagnostics or observables are
claimed. If physical diagnostics are pursued later, first authorize a finite
operational wall-time/memory budget and a staged convergence protocol that
exposes progress without changing scientific acceptance criteria; then
require the existing completed/eligible trajectory gates before reporting
those diagnostics. The current short run only shows that the default route
reaches the 1,000-step diagnostic cap quickly while allocating several
gigabytes. Do not substitute Float64 or call 64-bit BigFloat Float64; this
author route requires BigFloat (`precision_bits >= 64`), so a same-route
Float64 comparison is unavailable without building a different solver path,
which is outside this authorization.

No replacement tolerance or solver setting is supported by these data. If
the next step requires changing the frozen r5 tolerances, solver method,
output retention, or accepted operational budget, the minimum owner decision
is explicit approval of the exact changed setting(s) and whether they are
diagnostic-only or accepted for scientific verification, together with the
unchanged scientific convergence gates. Until then, inherited settings and
criteria remain unchanged and the physical result remains unestablished.
