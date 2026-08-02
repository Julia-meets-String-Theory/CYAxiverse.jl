# Checkpoint — 2026-08-02: 2023-paper overhaul begins

## Completed before this checkpoint

- Reconstructed the Appendix C (h^{1,1}=8) geometry and its 12 Table-1 charges.
- Reproduced the five-to-one minimum transition and
  (k_c=0.6745063700033533), agreeing with the draft value to the quoted
  precision.
- Added the complete 78-term equation-(19) potential as an opt-in diagnostic.
- Corrected the benchmark trajectory layer to use the paper's 12-term
  equation-(25) truncation by default and to convert the catastrophe solution
  from the leading-charge basis to GLSM coordinates.
- Extracted the local canonical normal form.  At Δk=10^-7 and
  Δφ=10^-8 M_Pl, the analytic Eq. (13) estimate gives approximately
  6,819 e-folds and the numerical valley flow gives approximately 6,822.

## Open collaborator question

The draft reports approximately 463,115 e-folds for the same nominal
trajectory.  The references expose a factor-of-two convention ambiguity in
the real axion metric versus (K_{T\bar T}), but that accounts for only a
factor of two, not the observed factor of approximately 67.9.  No cited
string-theory normalization produces the remaining factor.  Do not calibrate
the code to this number until the collaborator's trajectory conventions are
clarified.

## 2023-paper code overhaul — first slice

The overhaul starts from the validated truncated-potential path:

1. Keep `paper_benchmarks.n8_potential` and the deterministic
   `reduced_critical_points` solver as the reference implementation.
2. Add a reusable 2023 reproduction entry point for geometry data, with
   explicit truncation, threshold, fundamental-domain, and minima-count
   settings.
3. Validate that entry point first on the N=8 five-to-one benchmark, then on
   generated (h^{1,1}=8) HDF5 data before scaling to the 1000-geometry run.

The current test suite passes after the normalization diagnostics.

## JLM optimizer overhaul — first implementation slice

- Added `CYAxiverse.jlm_reduced`, a Julia-native reduced JLM preparation and
  solve module that does not require the PyCall extension.
- The new `ReducedJLMProblem` stores the reduced charge matrix as sparse data
  and separates preparation from solving, so a large scan can reuse selected
  charges, phases, determinants, and threshold masks without rebuilding them.
- `jlm_reduced.minimize` and `jlm_reduced.minimize_save` return/write the
  existing `Min_JLM_*` result shape, preserving the legacy statistics format.
- Fixed an inherited `αmatrix` indexing bug: retained perturbation columns now
  carry aligned `Qbar/Lbar/α` data, and row masking is anchored to the
  strongest retained perturbing instanton rather than an out-of-bounds
  leading-scale column.
- Patched the old PyCall-backed `jlm_minimizer` scale and phase lookup to use
  the aligned retained perturbation data.

Caveat: the legacy alpha-reduced JLM problem is not identical to the full
Table-1 N=8 reduced critical-point problem used for paper reproduction. Keep
the five-to-one N=8 benchmark attached to `generate.reduced_critical_points`;
use `jlm_reduced` as the scalable replacement for the old JLM statistics
optimizer path.

Validation: `julia --project=. test/runtests.jl` passes.

## JLM reduced batch runner

- Added `scripts/batch_jlm_reduced.jl`, a local/scheduler-friendly batch
  runner for `CYAxiverse.jlm_reduced.minimize_save`.
- The runner accepts explicit geometries or `paths_cy()` selection, supports
  `--h11`, `--limit`, `--offset`, `--threshold`, `--starts`, `--force`,
  `--hilbert`, `--data-dir`, and writes a CSV summary.
- The new geometry-data entry point is orientation-aware: it accepts potential
  matrices saved in either Python-facing or Julia-facing orientation and
  normalizes them to axion-by-instanton `Q` and two-by-instanton `L`.
- Smoke run completed on the Appendix C HDF5 geometry:
  `julia --project=. scripts/batch_jlm_reduced.jl --data-dir paper_benchmarks/appendix_c --geometry 8,1,1 --starts 2048 --force --summary paper_benchmarks/appendix_c/jlm_reduced_batch_smoke.csv`

Smoke result: `Nvac=1`, `issquare=0`, `extra_rows=2`, `det_QTilde=1`, runtime
approximately 5.9 seconds. This validates loading, orientation normalization,
legacy HDF5 output, and CSV logging for the new batch path.

## h11=10 reduced-JLM 1000-geometry batch

Ran the new batch runner on the local 1000-geometry `h11=10` dataset:

`julia --project=. scripts/batch_jlm_reduced.jl --data-dir /Users/vmehta/Documents/CYAxiverse/cyaxiverse/data --h11 10 --limit 1000 --starts 2048 --force --summary /Users/vmehta/Documents/CYAxiverse/cyaxiverse/data/logs/jlm_reduced_h11_010_1000_corrected.csv`

The first pass exposed a square-case bug in `jlm_reduced`: it incorrectly
multiplied the legacy square count by an extra determinant factor, producing
counts such as 64, which violate the paper's reported global maximum of 54.
The corrected square case uses the legacy convention `N_min = det_QTilde`.
A focused test now covers this (`det_QTilde=4` must report `N_min=4`, not 64).

Corrected h11=10 batch statistics:

- completed: 1000 / 1000
- failed: 0
- wall time reported by runner: 8.55 seconds
- sum of per-geometry times: 7.57 seconds
- mean per geometry: 0.0076 seconds
- mean Nvac: 1.622
- median Nvac: 1
- maximum Nvac: 5
- fraction with one minimum: 0.559
- histogram: 1 -> 559, 2 -> 290, 3 -> 122, 4 -> 28, 5 -> 1

Only six of the 1000 geometries were non-square reduced JLM cases. These were
rerun with `--starts 100000`, matching the old Python solver's sampling scale,
and all six returned the same `Nvac` values as the 2048-start batch.

This matches the qualitative Figure 1 story in arXiv:2309.01831: most
geometries have one minimum, the median is one, and counts are small. The local
arXiv source contains only `N_vac_KS_scatter_all.pdf`, not a raw h11=10 data
table, so exact h11=10 point-by-point agreement cannot yet be checked from the
source archive alone.

## Digitized Figure 1 reference data

Digitized the arXiv:2309.01831 Figure 1 source PDF
`/tmp/arxivsrc/minima/N_vac_KS_scatter_all.pdf`. The plot is a binned scatter:
each square is an `(h11, Nvac)` bin and the colorbar gives the approximate
number of geometries in that bin.

Added `scripts/digitize_2023_minima_figure.py`, which renders the figure at
150 DPI when needed, calibrates the axes from gridlines, samples each integer
`(h11, Nvac)` bin, converts marker colors through the colorbar, and writes:

- `paper_benchmarks/2023_minima/figure1_digitized_bins.csv`
- `paper_benchmarks/2023_minima/figure1_digitized_summary_by_h11.csv`
- `paper_benchmarks/2023_minima/figure1_digitized_h11_004_030_summary.csv`
- `paper_benchmarks/2023_minima/figure1_digitized_overlay.png`

The bin CSV includes both the raw colorbar-derived count estimate and a
`count_normalized` column. For `h11 >= 4`, normalized counts rescale each h11
slice to the paper's stated 1000 geometries. The raw colorbar totals for
`h11=4:30` range from approximately 1013 to 1047, which is a reasonable
digitization error for a rasterized colorbar.

Digitized `h11=4:30` summary:

| h11 | mean Nvac | median | max digitized |
|---:|---:|---:|---:|
| 4 | 1.1960 | 1 | 10 |
| 5 | 1.2986 | 1 | 10 |
| 6 | 1.3738 | 1 | 10 |
| 7 | 1.4239 | 1 | 4 |
| 8 | 1.5849 | 1 | 5 |
| 9 | 1.6585 | 1 | 5 |
| 10 | 1.7384 | 1 | 6 |
| 11 | 1.7930 | 1 | 6 |
| 12 | 1.8667 | 2 | 6 |
| 13 | 1.9233 | 2 | 8 |
| 14 | 1.9913 | 2 | 8 |
| 15 | 2.0163 | 2 | 9 |
| 16 | 2.1081 | 2 | 9 |
| 17 | 1.9967 | 2 | 9 |
| 18 | 2.0283 | 2 | 8 |
| 19 | 2.1673 | 2 | 25 |
| 20 | 2.2245 | 2 | 25 |
| 21 | 2.3050 | 2 | 25 |
| 22 | 2.1548 | 2 | 25 |
| 23 | 2.1653 | 2 | 9 |
| 24 | 1.9964 | 2 | 9 |
| 25 | 2.0650 | 2 | 9 |
| 26 | 2.2675 | 2 | 16 |
| 27 | 2.2820 | 2 | 16 |
| 28 | 2.1513 | 2 | 16 |
| 29 | 2.3159 | 2 | 16 |
| 30 | 2.3302 | 2 | 16 |

## Current generated-data comparison: h11=4:11

Ran `scripts/batch_jlm_reduced.jl` on the currently available generated data
for `h11=4:11` and compared against the digitized Figure 1 reference. The
available sample sizes are 998, 998, 999, then 1000 geometries for h11=7:11.

While doing this comparison, fixed two batch-runner ergonomics:

- If both `paths_cy()` and filesystem scanning are available, the runner now
  uses whichever finds more geometries. This avoids stale/small path-index
  files hiding newly generated geometry directories.
- Summary CSV files are now replaced by default. Use `--append-summary` only
  when intentionally accumulating rows.

Comparison output:
`paper_benchmarks/2023_minima/h11_004_011_reduced_jlm_vs_digitized.csv`

| h11 | n | computed mean | digitized mean | delta | computed median | digitized median | computed max | digitized max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 998 | 1.0281 | 1.1960 | -0.1679 | 1 | 1 | 3 | 10 |
| 5 | 998 | 1.0982 | 1.2986 | -0.2004 | 1 | 1 | 5 | 10 |
| 6 | 999 | 1.1982 | 1.3738 | -0.1756 | 1 | 1 | 3 | 10 |
| 7 | 1000 | 1.2860 | 1.4239 | -0.1379 | 1 | 1 | 4 | 4 |
| 8 | 1000 | 1.3980 | 1.5849 | -0.1869 | 1 | 1 | 4 | 5 |
| 9 | 1000 | 1.5410 | 1.6585 | -0.1175 | 1 | 1 | 5 | 5 |
| 10 | 1000 | 1.6220 | 1.7384 | -0.1164 | 1 | 1 | 5 | 6 |
| 11 | 1000 | 1.6860 | 1.7930 | -0.1070 | 1 | 1 | 6 | 6 |

The trend and medians match the digitized plot well, while the means are
systematically lower by roughly 0.1--0.2 for this regenerated sample. This may
reflect digitization/colorbar bias and/or the fact that the generated
geometries are not guaranteed to be the exact random sample used in the
published plot.

## Inflation-candidate prefilter and branch-enumeration fold-in

From the current `h11=4:11` generated-data comparison, the simple high-vacua
proxy identified six geometries with `Nvac >= 5`:

| h11 | polytope | frst | Nvac | det(Qtilde) |
|---:|---:|---:|---:|---:|
| 5 | 1 | 1 | 5 | 5 |
| 9 | 1 | 1 | 5 | 5 |
| 10 | 1 | 1 | 5 | 5 |
| 11 | 1 | 1 | 5 | 5 |
| 11 | 2 | 1 | 5 | 5 |
| 11 | 7 | 1 | 6 | 6 |

An initial full all-instanton Newton critical-point search was not the right
scaling target for these cases. The selected leading instantons are extremely
hierarchical, with leading `log10(Λ)` spans ranging from roughly 44 to 2718,
so the direct stationarity solve became allocation-heavy and stalled on the
larger candidates.

The useful efficiency observation is that the leading selected lattice gives a
cheap deterministic branch prefilter. Added reusable branch enumeration to
`src/generate.jl`:

- `leading_lattice_offsets(selected; tolerance=1e-8)`
- `leading_critical_branches(selected; tolerance=1e-8, max_branches=1_000_000)`
- `leading_critical_branches(Q, L; kwargs...)`

This enumerates
`Qtilde' * θ ∈ {0, 1/2}^h11 mod 1`, including the determinant-lattice copies
from `abs(det(Qtilde))`, and reports the leading selected-potential inertia.
Downstream scans can then evaluate the full potential, gradient, and Hessian
only on retained branches. A guard prevents accidentally enumerating enormous
branch sets in large scans unless `max_branches` is raised explicitly.

Also updated `reduced_critical_points` to clamp hierarchy equation scales to a
positive Float64 floor, avoiding zero scales when leading amplitudes differ by
hundreds of orders of magnitude.

Added `scripts/analyze_inflation_candidates.jl`, now wired to the shared
branch API. It evaluates the full potential/Hessian diagnostics on the
enumerated branches and writes:

- `paper_benchmarks/2023_minima/inflation_screen/candidate_summary.csv`
- `paper_benchmarks/2023_minima/inflation_screen/candidate_critical_points.csv`

Six-candidate screen result:

| geometry | branches | leading minima | saddles | least-tachyonic eta_min | slow-roll proxy hits |
|---|---:|---:|---:|---:|---:|
| h11=5, p=1, f=1 | 160 | 5 | 159 | -6.019e4 | 0 |
| h11=9, p=1, f=1 | 2,560 | 5 | 2,550 | -1.361e7 | 0 |
| h11=10, p=1, f=1 | 5,120 | 5 | 5,100 | -1.897e6 | 0 |
| h11=11, p=1, f=1 | 10,240 | 5 | 7,680 | -4.416e8 | 0 |
| h11=11, p=2, f=1 | 10,240 | 5 | 10,235 | -5.846e6 | 0 |
| h11=11, p=7, f=1 | 12,288 | 6 | 12,282 | -1.131e7 | 0 |

Interpretation: these six determinant-rich geometries do not look like robust
inflationary-trajectory candidates under the current quick slow-roll proxy.
They do, however, validate the branch-enumeration prefilter as the right
pipeline primitive: 40,608 leading branches were checked in about twelve
seconds via the shared API, while the naive full Newton scan was the wrong
scaling battleground.

Regression coverage added in `test/runtests.jl`:

- determinant-lattice offset count;
- half-integer branch count;
- leading-minima count;
- sign-aware leading negative-mode count.

Validation after the fold-in:

- `julia --project=. test/runtests.jl` passed.
- `scripts/analyze_inflation_candidates.jl` reproduced the six-candidate
  summary through the shared `CYAxiverse.generate.leading_critical_branches`
  API.
- `git diff --check` clean.
