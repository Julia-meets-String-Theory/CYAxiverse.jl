# CYAX-0131 r2 execution evidence

## Authority and candidate scope

- Handoff: `cyax-0131-low-n-inflation-manager-handoff`, revision 2,
  SHA-256 `6d76d0e43d005f63a40f10c8e3c7d3b8635e031df15bd8e8633ce7ee6d126b9e`.
- Detached Control Desk state: `READY_TO_DISPATCH`; review result
  `PASS_WITH_NONBLOCKING_FINDINGS / CURRENT`, SHA-256
  `da89c0ace19bbfcb995d87355d9ca53092ff25f5ab42d87cae6fdf52cdbbd2c9`.
- P0 source identity, Issue #131, #130 foundation, merged/post-merge-verified
  #172, private policy, and role capability checks were completed by the
  Manager before mutation. The implementation base is
  `74fea608684eae746f25ae45b18512f89b862fb0`.
- The immutable benchmark source blobs remained unchanged. The helper's
  Hessian normalization remains `4*pi^2`; tests compare it with the prior
  formula and confirm identical eigenvalue signs and the expected factor.
- Writable implementation is limited to `scripts/phase_volume_detuning_scan.jl`
  and `test/runtests.jl`, plus this `validation/cyax_0131_low_n_inflation/`
  subtree. No `src/**`, fixture, or package metadata path was changed.

## Implemented bounded evidence

- The deterministic N5 mathematical homotopy used `k_homotopy` in
  `[0.5, 1.5]`, two phase inputs, five grid points, and eight interval
  attempts. All intervals were covered: one refined fixed-configuration
  zero-mode crossing, seven rejected intervals, zero failures. The result is
  `homotopy_only`; it is not a catastrophe classification or a physical
  observable source.
- The deterministic N8 mathematical homotopy used `k_homotopy` in
  `[0.5, 1.0]`, two phase inputs, six grid points, and ten interval attempts.
  All intervals were covered: zero crossings, ten rejected intervals, zero
  failures. No homotopy candidate was promoted.
- N5 reduced-model replay used the current eight-row appendix-B fixture and
  source-defined critical scale `1.770068132610995629125239519150256394278`.
  Four `pi`-branch continuation points converged; two reported catastrophe
  events straddle `kc` at physical `delta_k=-2e-4` and `+2e-4`. The 120-bit
  local diagnostic classified the named point as `cusp`. Existing reduced
  anchors replayed at `delta_k=1e-7` and `6.65e-5`; a full physical N5
  trajectory remains `NOT_REACHED`.
- The read-only `validation/inflation_reproduction_results.md` text assigns
  `0.674506370003365` to the N5 reduced critical scale. Current
  `poly102_inflation.jl`, `reduced_models.jl`, and the accepted Issue #148 G1
  correction (commit `2a4e495ccdd838cc5b1e884fbac136115a7d433f`) establish
  N5 `kc=1.7700681326109957`; `0.674506370003365` is the N8 scale. The older
  document is left unchanged and this historical discrepancy is recorded in
  `scan_manifest.toml`.
- The N5 `0.04`-radian single-instanton probe is explicitly not a paper-supplied
  full phase vector. At the named zero-theta, `k_homotopy=1` point, the
  eight-row homotopy potential evaluated to `0` for zero phase and
  `2.03548114864806348473318060537900610308e-19` for the probe. The raw-radian
  input is converted to cycles for the helper. This diagnostic uses no
  acceptance tolerance and supplies no physical observables.
- The N8 twelve-row Table 1 augmented replay converged in three iterations
  (`gradient_residual=2.4873142859848615e-15`,
  `null_residual=4.7779859664277034e-15`). Its separate 120-bit local
  catastrophe diagnostic is `unresolved` at the named twelve-row point. This
  Table 1 solve is kept separate from the physical trajectory model.
- N8 physical-k re-establishment used the existing ten-row zero-phase
  `n8_degenerate_point` route. It returned `k=0.674506370003365`, with
  gradient residual `2.021453022468334e-15` and null residual
  `6.258470461730165e-13`. The existing point solver and `kc` input are
  Float64 at its `1e-11` source tolerance (53-bit input). The separately
  reported 120-bit diagnostic and 100-bit arithmetic trajectory do not refine
  that `kc`; no homotopy crossing required promotion-specific refinement.
- The named ten-row catastrophe-point diagnostic remains `unresolved` at
  120-bit diagnostic precision and tolerance `1e-8`. Its transverse Hessian
  eigenvalues are reported only at that named point in `scan_manifest.toml`.
- The N8 trajectory used the existing physical flow with the requested
  `delta_k=0.0015320548620798324`, `precision_bits=100`,
  `reltol=1e-8`, `abstol=1e-10`, `max_time=1e4`, `max_step=100`,
  `initial_step=1e-5`, `scan_step=5`, 20 stored samples, and `maxiters=1,000,000`.
  These are 100-bit BigFloat arithmetic with explicit solver tolerances; they
  do not imply 100-bit observable accuracy. The flow terminated at
  `eta_parallel` after 343 accepted and 65 rejected steps in 8.099734 seconds.
  `N_e=60.006354132181679315584233375102`; the existing tight-step physical
  reference is approximately `60.00336`, a reported difference of
  `0.002994132181679315584233375102...` with no defined acceptance tolerance.
  The older `59.690642055250756` Float64 BDF result is retained as coarse
  provenance only.
- The requested and measured detuning from the re-established ten-row
  `kc` agree within `1e-11` (measured binding residual `0`). The exact
  `trajectory_observables` sample is stored index 20 (one-based), coordinate
  `n=60.006354132181679315584233375102`. Its `n_s`, `paper_delta_H`, and
  cumulative turning values are exit-sample diagnostics only; they do not
  define an observational pivot or establish viability.
- The N8 deterministic single-instanton probe also records distinct zero-phase
  and probe potential values at its named ten-row diagnostic point. Its phase
  vector is not claimed to be source-matched, and it supplies no trajectory
  observables.

## Trajectory attempts and corrected run settings

| Attempt | Horizon / settings | Observed outcome |
| --- | --- | --- |
| 1 | `1e6`, precision 100, API-derived default tolerances | Interrupted after about 23 minutes. No solver result or generated trajectory artifact was observed: `INTERRUPTED_UNOBSERVED`. |
| 2 | Exact `1e6` restart, same inputs and defaults | No output or progress channel appeared during about 45 minutes; interrupted: `INTERRUPTED_UNOBSERVED`. |
| 3 | `1e4`, precision 100, API-derived default tolerances (`reltol` about `1e-50`, `abstol` about `1e-66`) | Interrupted after 546.654604 seconds with `InterruptException`, before a solver result: `INTERRUPTED_UNOBSERVED`. |
| 4 | `1e4`, precision 100, existing benchmark tolerances `reltol=1e-8`, `abstol=1e-10`, `maxiters=1,000,000` | Finite `eta_parallel` exit in 8.099734 seconds; completed trajectory evidence is recorded above and in `n8_trajectory_attempts.csv`. The `1e5` fallback was not run because the 1e4 flow terminated. |

The looser integration tolerances reuse settings already present in the
repository's bounded N8 benchmark and tests. Arithmetic precision and solver
error tolerances are reported separately.

## Verification record

Runtime was Julia 1.12.6 on `aarch64`, one Julia thread. Package tests used the
retained P0 numerical-equivalence environment with cached dependencies and
offline package resolution. This avoids making test results depend on a live
registry or network.

| Check | Command / environment | Result |
| --- | --- | --- |
| Focused package test pass | `CYAXIVERSE_TEST_FULL=0 JULIA_PKG_OFFLINE=true julia --project=validation/p0_numerical_equivalence/environment --startup-file=no -e 'using Pkg; Pkg.test("CYAxiverse")'` (cached task and user depots) | Exit 0; `CYAxiverse.jl` testset 65/65; package tests passed. |
| Bounded pipeline plus package tests | `CYAXIVERSE_TEST_FULL=0 CYAX131_RUN_BOUNDED_PIPELINE=1 JULIA_PKG_OFFLINE=true julia --project=validation/p0_numerical_equivalence/environment --startup-file=no -e 'using Pkg; Pkg.test("CYAxiverse")'` (cached task and user depots) | Exit 0; `CYAxiverse.jl` testset 86/86; package tests passed. The bounded pipeline stages and trajectory result are recorded above. |
| Python-free package import | Julia was run with an empty `PATH`, `PYTHON` and `JULIA_PYTHONCALL_EXE` set to unavailable paths, and `PYTHONPATH` unset; `using CYAxiverse` in the P0 environment | Exit 0; `python-free package import passed`. |
| Repository audit | `julia --project=. bin/audit.jl` (cached audit environment, offline package resolution) | Exit 1. JET reported two pre-existing undefined-`i` findings: `n8_full_potential` at `reduced_models.jl:106`, and the N8 full-derivative call chain ending at `poly102_inflation.jl:649`. Bound source blobs are unchanged. Revise also emitted repeated `FolderMonitor: too many open files (EMFILE)` background errors. The audit did not pass; no `src/**` correction was in scope. |
| `agent_verify.py` whitespace check | `python3 scripts/agent_verify.py diff-check` | Exit 0; `status=passed`, `no whitespace errors`. |
| Git whitespace check | `git diff --check` | Exit 0; no output. |

Package-test output also contained four existing warnings that log-domain
linear-boundary truncation is used only for the Float64 seed while the
arbitrary-precision Hessian retains all instantons. The optional orientifold
database round-trip test was skipped because its local dataset was absent.
Package tests used offline dependency resolution and emitted no DNS warning.

An intermediate package-test attempt exited 1 on a Julia parse error in the
driver's direct-execution `@__DIR__` ternary. The syntax was corrected; the two
passing package runs above include the corrected driver and test assertions.

## Claim boundary

`k_homotopy` remains mathematical, `homotopy_only` evidence. It does not
populate physical observables and is not assumed equivalent to physical `k`.
N5 remains benchmark/reduced-model evidence only. N8 physical observables use
only the existing ten-row/poly-102 flow after physical-k re-establishment.
`paper_delta_H` is retained; new `A_s` conversion/acceptance windows, new
`n_s` acceptance windows, `r`, observational viability, the Issue #131 stretch
goal, population scanning, dynamical saxion stabilisation, merge, and Issue
#131 closure remain `NOT_REACHED` or unauthorized as specified in
`scan_manifest.toml`.
