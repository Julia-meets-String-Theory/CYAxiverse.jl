# CYAX-0131 implementation evidence map

This evidence belongs to the exact owner-dispatched r5 implementation packet
(SHA-256 `ac2fcce3eae410c64e21d203a46fbcb9782eccddbae08cf5f8c5d7341e24392f`).
The governing approved S2 revision is `CYAX-0131-S2-NORMATIVE-R2` at commit
`691ffd1757cf314f8ae9a095fc4b4d665b128512`, tree
`b7f1df0fc7d44cf3b568313cff66c26364b591ea`. Its `spec.md`, `plan.md`, and
`tasks.md` are immutable. The current execution candidate is based on
`a5eacd8cd4a5905161cab239a461ba252c64e8e0`, tree
`d419416514a44d943ddd20290885ac8ba4090999`; the final candidate identity is
recorded after freeze.

| Requirement | Implementation/evidence route | Status |
|---|---|---|
| R-001 | `src/paper_benchmarks/poly102_inflation.jl`; zero-phase source replay in `replay.jl`; 10 retained author rows and zero phases | **PASS** — source witness converged at `k=0.674506370003365`; gradient residual `2.02e-15`, null residual `6.26e-13`; `replay-partial.sanitized.log` |
| R-002 | `n8_row2_phase_catastrophe`; focused test in `test/runtests.jl`; 400-step author10 phase ramp, BigFloat refinement, same-branch two-sided stationary bracket | **PASS for bounded calibration** — 400/400 phase steps and 12/12 stationary points converged; repaired refinement gradient residual `1.93e-37`, null residual `6.08e-34`, and the minimum absolute Hessian eigenvalue is `1.65e-34`; the seven transverse eigenvalues are positive and the recorded stationary-branch minimum-Hessian signs are opposite. Refinement lifts the exact continuation Float64 phase vector to 128-bit BigFloat and returns it as witness provenance. Historical `BigFloat("0.04")` calibration numbers remain identified with the prior candidate in the diagnostic report. |
| R-003 | Immutable `src/paper_benchmarks/n8_continuation.jl`; bounded `n8_pseudo_arclength_continuation` output separately labeled P96/Table-1 cross-check | **PASS as cross-check only** — seven retained points converged; contributes nothing to R-001/R-002; `replay-partial.sanitized.log` |
| R-004 | Reduced N5 two-cosine phase gradient/Hessian/fold helpers; second cosine receives `π/4`; 256-bit fold and two-sided finite-grid root evidence; no 8-row phase vector | **PASS for reduced model** — gradient `1.73e-77`, Hessian `0`, residual `1.73e-77`; lower-side 4 and upper-side 2 scanned sign-change roots; full eight-row mapping `NOT_VERIFIABLE`; `replay-partial.sanitized.log` |
| R-005 | Refinement calls concrete `author_inflation.n8_author_trajectory`; source uses `physical_k=critical_k + delta_k`; diagnostic gate binds exact phase encoding, refined coordinates, basis/metric convention, and critical `k` to the completed author10 catastrophe witness | **BOUNDED NULL / NO PHYSICAL CLAIM** — the full trajectory run was owner-terminated before a result. Bounded source probes reached the `:tmax` horizon with an open slow-roll interval; no completed-window diagnostics or physical observables are emitted. The focused regression verifies matching witness acceptance and rejects absent/altered phase, coordinate, metric/basis, and critical-`k` identities. |
| R-006 | Diagnostics gate requires completed refinement and finite-exit window, successful solver return, entered slow-roll, positive accepted steps, exact returned sample indices/`samples[i].n`, and a matching source witness; `N_e = slow_roll_efolds = end_n - entry_n`; no pivot | **BOUNDED NULL / NO DIAGNOSTICS PROMOTED** — the full run remains owner-terminated. Open `:tmax` windows are retained only as explicitly censored duration, with no completed `N_e`, end coordinates, or samples. A synthetic completed-window fixture exercises positive gate behavior; null/open, failed status, zero-step, and mismatched-witness cases are rejected. |
| R-007 | Shifted author10 catastrophe point is named by model, phase vector, raw-radian coordinates, author metric/basis, radial calibration scale, refined `theta` and `k`; only its transverse Hessian values are reported | **PASS for bounded calibration** — repaired exact Float64 phase vector `[0.0, 0.04, 0, 0, 0, 0, 0, 0, 0, 0]`, its 128-bit lifted representation, refined coordinates and `k` are recorded in `focused-regressions-review-repair.sanitized.log`; the historical decimal-BigFloat identity remains separately in `replay-partial.sanitized.log`. No physical trajectory observable is claimed. |
| R-008 | Five deferred categories are recorded with no acceptance window evaluated | **NOT_REACHED, explicitly recorded** — (1) `A_s` conversion and acceptance window, (2) `n_s` acceptance window, (3) tensor-to-scalar `r`, (4) observational stretch goal, and (5) full physical N5 trajectory. No window evaluated. |
| R-009 | Result boundary remains fixed-saxion effective-theory calibration/discovery; no population, stabilization, string-cosmology, or full-KS claim | **PASS boundary** — report claims only bounded calibrations and cross-check; population prevalence, stabilization, fully stabilized string cosmology, full-KS, and observational viability remain `NOT_ESTABLISHED`. |
| R-010 | Source audit confirms Eq. 19 `q·tau`/cross-coefficient repair and Issue #172 `4π²` normalization ancestry; protected source paths stay unchanged | **PASS** — exact protected source blobs and ancestry were revalidated at resume; final whitespace checks recorded after evidence updates. |
| R-011 | Exact approved S2/P0 evidence and current r5 handoff authorize this bounded implementation; specialist SPEC/SCIENTIFIC reviews are post-freeze gates | **P0 PASS; candidate ready for fresh reviews after freeze** — exact handoff, approved S2 and Manager resume currentness evidence remain bound; no merge/release/issue closure is authorized. |

## Execution boundaries

- Source evidence in `replay.jl` labels author10 catastrophe-calibration
  scales separately from the physical author radial scale used by the
  trajectory. No phase-volume or scale homotopy value is used for physical
  observables.
- The N5 replay is only the reduced two-cosine light-direction model. The
  full eight-row phase mapping is `NOT_VERIFIABLE` and was not constructed.
- The P96/Table-1 continuation remains an independent 12-row cross-check.
  It cannot satisfy either author10 calibration gate.
- The short initial displaced-seed probes and their rejected/failed outcomes
  are retained in `n8_shifted_fold_audit.md`. The refined central stationary
  branch is a separate successful identity, not a reinterpretation of those
  probes.
- The implementation gate permits trajectory sample rows only after the
  required eligibility checks; any permitted `n_s`, `paper_delta_H`, or
  cumulative-turning value carries its exact returned index and sample `n`.
  No sample row was emitted in this owner-terminated replay or the capped
  probes. `N_e` is defined as the window aggregate
  `slow_roll_efolds`; no `N_e` was measured. The existing 60-e-fold anchor is
  a discrepancy reference, not an acceptance threshold.
- The protected Table-1 continuation source and catastrophe-diagnostics source
  were currentness-checked against the approved r5 baseline and remain
  unchanged: `src/paper_benchmarks/n8_continuation.jl` Git blob
  `ff486c0fc982de0006553fd2d963365f07820e6d` and
  `src/paper_benchmarks/catastrophe_diagnostics.jl` Git blob
  `df119eb8899de593300e9a4db8ad7cfc1962b77f`.
- Full physical N5 trajectory, new `A_s` conversion/acceptance, new `n_s`
  acceptance, tensor-to-scalar `r`, observational stretch goal, population
  prevalence, dynamical saxion stabilization, and full-KS claims remain
  `NOT_REACHED` or `NOT_ESTABLISHED`.

Machine-readable historical completed scientific records are retained in
`replay-partial.sanitized.log`; the repaired row-2 calibration and bounded
gate regression are in `focused-regressions-review-repair.sanitized.log`.
Full-run termination and the separate
maxiters-censored diagnostics are recorded in
`trajectory-diagnostics-report.md` and
`trajectory-default-probe.sanitized.log`. The raw `replay.log` and raw probe
log contain local runtime paths and are not candidate evidence. Exact
commands, bounded budgets, verification results, and the final candidate
identity are recorded in `execution_record.md` after final checks.
