# CYAX-0131 bounded execution record

## Exact inputs and runtime

- Dispatched r5 handoff: `CYAX-0131 low-N inflation manager handoff r5`,
  revision 5, SHA-256
  `ac2fcce3eae410c64e21d203a46fbcb9782eccddbae08cf5f8c5d7341e24392f`.
- Approved S2 input: `CYAX-0131-S2-NORMATIVE-R2`, commit
  `691ffd1757cf314f8ae9a095fc4b4d665b128512`, tree
  `b7f1df0fc7d44cf3b568313cff66c26364b591ea`.
- Execution base: commit `a5eacd8cd4a5905161cab239a461ba252c64e8e0`, tree
  `d419416514a44d943ddd20290885ac8ba4090999`.
- Julia: 1.12.6. The active package resolves to the candidate `src/CYAxiverse.jl`.
- Existing pinned environment (unchanged): Project SHA-256
  `ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9`;
  Manifest SHA-256
  `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf`.
- After verification setup, candidate `Project.toml` remains SHA-256
  `38275eaf04c8f9ae28541542326f6c791c3a6a40d2a9ca3916088ca0652c368c` and
  candidate `Manifest.toml` remains absent. The tracked P0 Project and
  Manifest hashes above remain unchanged.

## Frozen bounded configuration

`replay.jl` runs the following deterministic sequence once. A failure at any
required calibration or trajectory gate stops later dependent work; no search
is widened to force a pass.

| Stage | Fixed budget and convention | Coverage/stop rule |
|---|---|---|
| N8 zero-phase author10 witness | `author_inflation.n8_degenerate_point()`, exact 10-row `trajectory=true` potential, ten zero phases; existing source checks `k≈0.674506370003365` at `1e-15`, gradient/null residuals `<1e-10` | One named source witness; failure stops N8 phase work |
| N8 row-2 phase calibration | `phase[2]=0.04` radians, all other phases zero; 400 warm phase increments from the zero-phase source witness; Float64 corrector tolerance `1e-11`, maximum 1,000 iterations per phase; 128-bit BigFloat refinement at `1e-30`, maximum 1,000 iterations | 400 attempted increments retained; all must converge; refined gradient/null residual `<1e-10`; seven transverse modes positive |
| N8 critical identities | Continue from the refined author10 point at `k_c+10×10^-5`; fixed-k steps in `10^-5` increments through `k_c-10^-5`; trust-region tolerance `1e-13`, maximum 2,000 iterations per point | Twelve retained points including start, each convergence recorded; endpoint minimum-Hessian eigenvalues must have opposite signs; no displaced seed is silently substituted |
| N5 reduced shifted fold | Reduced source two-cosine light-direction model; `π/4` applied only to the second cosine; 256-bit NLsolve trust region, `ftol=xtol=10^-70`, maximum 1,000 iterations | Re-solve the fold; require gradient `≤10^-9`, absolute Hessian `≤10^-10`, and residual `≤10^-70` |
| N5 two-sided root coverage | At `k_c±10^-4`, scan 10,000 angular intervals on `[0,2π]` (10,001 nodes per side); each detected sign-change bracket receives at most 256 BigFloat bisections; locally refine detected lower-side roots at `10^-70` | Preserve both root lists, all local identities and residuals; record roots near the fold on each side and full-grid counts; no eight-row phase mapping |
| P96 cross-check | Immutable 12-row Table-1 route; zero-phase start `theta=zeros(8)`, `k=0.68`; pseudo-arclength `ds=10^-5`, six continuation steps, tolerance `10^-10`, bounds `(0.67,0.69)` | Initial point plus six accepted steps; label only as P96 cross-check; never counts toward author10 gates |
| N8 physical author trajectory | One existing reference detuning `delta_k=1.5320548620798324e-3`; `critical_k` is the refined row-2 author10 catastrophe; row-2 phase vector and refined theta passed to concrete `author_inflation.n8_author_trajectory`; Rodas5P; 100-bit precision; default source-derived tolerances; `max_time=10^6`, `maxiters=10^8`, `max_step=100`, `scan_step=5`, `initial_step=10^-5`; 20 returned samples | Promotion requires `refinement_status=:completed`, entered slow-roll, positive accepted steps, successful retcode, and exact source-precision equality `physical_k=critical_k+delta_k`. Report source `efolds=end_n` separately from `N_e=slow_roll_efolds=end_n-entry_n`, then emit permitted diagnostics for each explicitly indexed sample. Compare `N_e` with the source `reference_efolds().n8[2]` value 60.0 without a pass/fail threshold; that reference is the original default-zero-phase author-model anchor at this delta, not an expected value for the shifted row-2 phase. |

The earlier displaced-seed outcomes and their exact residuals remain in
`n8_shifted_fold_audit.md`. They are retained as rejected identities; the
successful central stationary continuation is separately labeled.

## Commands and results

The run uses the regular Julia 1.12.6 host executable, the candidate project,
and the already pinned P0 environment on `JULIA_LOAD_PATH`. From repository
root, the exact replay command is:

```sh
repo_root="$(git rev-parse --show-toplevel)"
JULIA_LOAD_PATH="@:${repo_root}/validation/p0_numerical_equivalence/environment:@stdlib" \
  julia --startup-file=no --project=. \
  validation/cyax_0131_low_n_inflation/replay.jl \
  > validation/cyax_0131_low_n_inflation/replay.log 2>&1
```

The initial short phased
physical-route probe produced `refinement_status=:completed`,
`entered_slow_roll=true`, 41 accepted steps, and successful Rodas5P return at
64-bit precision. That probe used `max_time=10`, two samples,
`reltol=1e-8`, and `abstol=1e-10`; it is an eligibility-gate check, not the
reported 100-bit calibration run. Its first diagnostics gate attempt exposed a
precision mismatch in the independent `k` check; the gate now recomputes the
same `critical_k + delta_k` arithmetic at the configured trajectory precision.

An earlier replay checkpoint recorded 40 minutes 32 seconds of active
execution. That checkpoint is historical and is superseded by the final
owner-termination status below. The frozen configuration and acceptance rules
were not changed.

Verification and final-state results are recorded below. Commands are
recorded only with sanitized, repository-relative paths. Do not copy raw
private or host-local logs into this subtree.

Independent static verification while the replay continued:

- Command: `git diff --check` — passed with no whitespace errors.
- Command: `python3 scripts/agent_verify.py diff-check` — passed, exit code 0;
  summary: no whitespace errors; no warnings.
- A first direct full-suite attempt with `--compiled-modules=no` stopped during
  package loading because `AbstractTrees` source was unavailable in that
  no-cache mode. No tests ran. A regular cache-path run then passed the initial
  8/8 optionality assertions but stopped before the remaining suite because
  the pinned P0 manifest did not contain the declared test extra `CairoMakie`.
- Focused regression command — passed:

  ```sh
  repo_root="$(git rev-parse --show-toplevel)"
  p0_env="${repo_root}/validation/p0_numerical_equivalence/environment"
  JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 \
    JULIA_LOAD_PATH="@:${p0_env}:@stdlib" \
    julia --threads=1 --startup-file=no --project=. \
    validation/cyax_0131_low_n_inflation/focused_regressions.jl
  ```

  It passed;
  30/30 assertions in 16.8 seconds. It exercises the shifted N8 calibration,
  physical-k eligibility, selected-sample and N_e contract, and rejected
  refinement/solver/zero-step/scale cases. One initial run had 29/30 because
  the supplemental assertion recomputed the e-fold window at default BigFloat
  precision; the corrected check uses the recorded 64-bit trajectory
  precision and passes.
- Python-free import and optional-extension boundary — passed. The repository
  defines `PyCall` as a weak dependency and `CYAxiversePyCallExt` as its
  conditional extension. The command asserted the candidate source path,
  absence of `PYTHON`, `PYTHONPATH`, and `PYTHONHOME`, no loaded `PyCall`
  module, and no loaded `CYAxiversePyCallExt`:

  ```sh
  repo_root="$(git rev-parse --show-toplevel)"
  p0_env="${repo_root}/validation/p0_numerical_equivalence/environment"
  env -u PYTHON -u PYTHONPATH -u PYTHONHOME JULIA_NUM_THREADS=1 \
    JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_LOAD_PATH="@:${p0_env}:@stdlib" \
    julia --threads=1 --startup-file=no --history-file=no --project=. \
    -e 'using CYAxiverse; @assert realpath(pathof(CYAxiverse)) == realpath(joinpath(pwd(), "src", "CYAxiverse.jl")); @assert !haskey(ENV, "PYTHON"); @assert !haskey(ENV, "PYTHONPATH"); @assert !haskey(ENV, "PYTHONHOME"); @assert Base.get_extension(CYAxiverse, :CYAxiversePyCallExt) === nothing; @assert !any(pkgid -> pkgid.name == "PyCall", keys(Base.loaded_modules)); println((candidate_source=:passed, core_import=:passed, python_environment=:unset, pycall_module_loaded=false, pycall_extension_loaded=false))'
  ```

  Output: `(candidate_source = :passed, core_import = :passed,
  python_environment = :unset, pycall_module_loaded = false,
  pycall_extension_loaded = false)`.
- Exact audit command: `julia --project=. bin/audit.jl` — passed, exit code 0.
  JET analyzed 1,263 top-level definitions with no errors; Aqua passed all
  reported groups; potential, derivative, symmetry, and positive-definiteness
  checks passed. The repository entry point created an external cache with
  `Aqua 0.8.18` and `JET 0.12.2`. Its external Project SHA-256 is
  `ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9` and
  Manifest SHA-256 is
  `75be439157e143f111261868819fb99ca60a79bb4225c899f401e1094dfc4d51`.
  Five direct core versions were newer than P0 in this isolated audit cache:
  HDF5 `0.17.4` vs `0.17.3`, LinearSolve `5.18.2` vs `5.17.4`, Nemo `0.56.2`
  vs `0.56.1`, StaticArrays `1.9.22` vs `1.9.20`, and TimerOutputs `1.2.2`
  vs `1.2.1`; all other direct core versions matched. Candidate dependency
  files and tracked P0 environment files remain unchanged.
- The test environment at an external temporary path started from byte copies
  of the approved P0 Project/Manifest, rebound only the external CYAxiverse
  manifest path to this candidate, and added only declared test extras with
  `Pkg.PRESERVE_ALL`: CairoMakie `0.15.15`, ColorSchemes `3.31.0`. Its Project
  SHA-256 is `8e26eb84a345c68f887f719dfb6c99645ce30a70aa8a438a4aa6a6aea7331bf6`;
  Manifest SHA-256 is
  `b515317bcb0d126c9070f761135073ee1c6ff2463194ca3600aed3a53a997f43`.
  Every direct core dependency version matches the P0 manifest. Full
  `Pkg.test` passed in its external sandbox. Its exact Julia invocation was:

  ```sh
  repo_root="$(git rev-parse --show-toplevel)"
  test_env="${CYAX0131_TEST_ENV:?set to the external test environment identified by the Project and Manifest hashes above}"
  julia --threads=1 --startup-file=no --project="${test_env}" \
    -e 'using Pkg; Pkg.test(Pkg.PackageSpec(name="CYAxiverse", uuid=Base.UUID("e5e45d93-5055-4eab-878b-2e484be3f951"), path=ARGS[1]); allow_reresolve=false)' \
    "${repo_root}"
  ```

  The external Project/Manifest identities are the hashes above; the path
  itself is omitted. The sanitized output
  is `full-tests.sanitized.log`: all 55 testsets passed (2,081/2,081
  assertions), with four existing warnings and two informational skips for
  unavailable real-data round-trip fixtures. An initial invocation using an
  unsupported `preserve` keyword and a second missing the package name/UUID
  both stopped before tests; neither changed an environment. The final test
  run used `--startup-file=no --threads=1`, `allow_reresolve=false`, and only
  the declared test extras above. Candidate and tracked P0 dependency files
  remained byte-identical.

- Final whitespace checks at this source state: `git diff --check` passed;
  `python3 scripts/agent_verify.py diff-check` passed with exit code 0 and
  summary `no whitespace errors`, no warnings.
- Read-only replay liveness sample at elapsed time `59:59`: the Julia
  main-thread stack showed active BigFloat/MPFR arithmetic (`BigFloat`, `fma`,
  `cos`, `sum`, and BigFloat dictionary lookup) during top-level evaluation.
  No high-level replay symbol was available to distinguish the N5 scan from
  the ODE stage. This is liveness evidence only, not a scientific result. The
  replay process was using 98.9% CPU and its redirected log remained buffered;
  the configuration and stop rule above were unchanged.
- Historical source-grounded stage audit before the interrupt traceback: the exact launched script is
  `validation/cyax_0131_low_n_inflation/replay.jl`; its top-level imports and
  constants are followed by `main()` at line 292. The package/module
  top-level constants are static source tables, metadata, simple path maps,
  and aliases; the only `log10(exp(1.0))` constant is a small Float64 value.
  BigFloat precision blocks and numerical loops occur inside functions, not
  during module initialization. The three plausible BigFloat-heavy stages
  inside `main` are the early N8 refinement, the middle N5 256-bit fold/root
  scan, and the final 100-bit author trajectory. The interpreter-to-BigFloat
-  stack sample did not expose a high-level call or progress counter. The later
  interrupt traceback below confirmed the trajectory ODE stage. No process
  restart or run-configuration change was made.
- Historical pre-termination host health checkpoint at elapsed `90:42`: the process remained in
  running state with `99.4%` CPU and CPU time `89:07.15`; the redirected log
  was still buffered. The packet supplies the configured simulation horizon
  (`max_time=10^6`) and iteration limit (`maxiters=10^8`) but no wall-clock
  limit. No simulation-time or step counter is exposed, so a grounded ETA is
  unavailable. This predates the owner termination.

## Final disposition after owner termination and bounded diagnostics

The owner ordered interruption of the full replay after diagnostics were
preserved. Peer process evidence confirms the Julia replay was terminated by
SIGINT and had exited afterward. This worker did not restart the replay. The
process ran 101m51s elapsed and 99m51s CPU before interruption. Its traceback
located the active stage at
`n8_author_trajectory` -> OrdinaryDiffEq Rosenbrock `perform_step!` ->
LinearSolve generic LU -> BigFloat arithmetic/allocation -> GC. Process-wide
exit totals were 112,801,083,728 allocations (112,801,082,254 pool; 1,474 big)
and 3,137 GCs. Main-thread GC samples and 14.2 GB peak footprint are sampling
and process observations, not a full-run allocation attribution. Full-run
physical trajectory status is `OWNER_TERMINATED`, numerical outcome
`UNESTABLISHED`: no terminal solver result, actual simulation time,
accepted/rejected counts, `N_e`, samples, or physical observable was recovered.

The flushed completed records are preserved without local runtime paths in
`replay-partial.sanitized.log` (the source log was SHA-256
`a2b63069f15753a643f0b8b6c7f1b981d725500abb6939241b4bddb148680c1816`). The
zero-phase author10 witness converged. Row-2 phase continuation completed
400/400 increments and 12/12 stationary points, with 128-bit refinement and
a two-sided minimum-eigenvalue sign change. The N5 reduced shifted fold/root
coverage completed. Seven P96/Table-1 points converged and remain cross-check
only. No full physical N5 trajectory or five observational items were
reached.

The separately authorized diagnostic command and same-process 100-bit probe
measurements are recorded in `trajectory-diagnostics-report.md` and
`trajectory-default-probe.sanitized.log`. The first and repeated probes used
the author's unchanged default tolerances; each hit the diagnostic
`maxiters=1000` cap with 922 accepted, 78 rejected, 8,000 RHS and 922 Jacobian
evaluations, `entered_slow_roll=false`, and `retcode=MaxIters`. They allocated
4,590,881,560 and 3,791,734,040 bytes, respectively. Because the author API
does not return terminal time on this no-window path, simulated time is
unavailable. This is a bounded null, not a scientific solver failure or
eligible trajectory. No physical diagnostics were emitted.

At 100 bits the evaluated source defaults are `reltol=1e-50` and
`abstol≈1e-66`; runtime unit-scale epsilon and ratios are in the diagnostic
report. These ratios do not establish a tolerance defect. Calibration source
uses Float64 `0.04` for continuation/branch and `BigFloat("0.04")` for the
128-bit refinement; the representations are recorded separately, with no
exact same-phase trajectory claim. Any physical diagnostic in future must
preserve a consistent phase encoding through refinement and trajectory.

R-001 through R-004 calibration/cross-check gates are evidenced as described
in `traceability.md`. R-005/R-006 remain a bounded null with no physical
claim or diagnostics promoted. The five deferred observational items are
explicitly `NOT_REACHED`; the full physical N5 trajectory is also
`NOT_REACHED`. This satisfies the packet's bounded-null/no-observables route;
successful physical trajectory output is not required solely to retain the
calibration slice. Source/evidence content hashes are listed in
`EVIDENCE_SHA256SUMS.txt`; the exact frozen Git commit/tree are returned to
the Manager with this record.

The exact bounded-diagnostic invocation, environment stack, trajectory
configuration, output hashes, and measured results are in
`trajectory-diagnostics-report.md`. It exited 0 after 26.8 seconds. Both
100-bit source-default probes reached their explicit `maxiters=1000` cap with
`retcode=MaxIters`, 922 accepted and 78 rejected steps, 8,000 RHS evaluations,
922 Jacobian evaluations, and no slow-roll window. The first/repeat allocated
4,590,881,560/3,791,734,040 bytes in 6.024014/2.666163875 seconds; measured GC
time was 0.246801541/0.175834336 seconds. The API exposed no terminal
simulation-time value on this return path. These are bounded null diagnostics,
not scientific failure or eligible physical observations.

The 100-bit defaults evaluate to `reltol=1e-50`,
`abstol=1.0000000000000000000000000000002e-66`; `eps(BigFloat(1))` was
`1.5777218104420236108234571305656e-30`. The ratios and limitations are in
the diagnostic report. No replacement tolerance was selected. The nominal
row-2 phase uses Float64 `0.04` during continuation/branch and
`BigFloat("0.04")` in the 128-bit refinement. Their numeric encodings are not
bit-identical; this is recorded without claiming exact same-phase physical
trajectory identity. No physical diagnostic was promoted.
