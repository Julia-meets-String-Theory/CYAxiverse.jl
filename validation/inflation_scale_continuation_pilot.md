# Scale-continuation and catastrophe pilot

Date: 2026-08-07
Script: `scripts/inflation_scale_continuation.jl`
Status: bounded diagnostic pilot; `physical + full` is the default, with
explicit author-comparison (`physical + fixed`) and legacy
`homotopy_only` modes.

## Scientific boundary

The generic HDF5 files retain the geometric metadata needed by the draft
author's coefficient construction: divisor volumes, `Kinv`, and `CY_volume`.
The pilot now provides two explicitly labelled paths:

```text
physical: tau -> k*tau, Kinv -> k^2*Kinv, Q unchanged,
          V -> V                 (fixed)
          V -> k^(3/2)*V         (full)
homotopy: L[2, :] -> k*L[2, :], K unchanged
```

The physical path reconstructs the leading and pair/cross coefficients and
rejects files whose stored `Q/L` do not match the author ordering or
normalization. The legacy mathematical homotopy is available explicitly:

```text
L[2, :] -> scale * L[2, :]
```

with `reference_scale=1.0`. This stretches the stored base-10 instanton
exponents and is not a physical volume continuation. The reconstructed path is
the default; select the author's convention with
`--volume-normalization fixed` or the legacy path with
`--scale-status homotopy_only`. The N=5/N=8 benchmark scale laws remain
separate regression evidence.

The tested grid was `0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10`. It is a
diagnostic window, not a completeness claim. The canonical package
orientation and period-one coordinate convention are used throughout.

## Implementation and output contract

The pilot streams leading-branch seeds at the reference potential using the
existing exact branch-count check. Seeds are corrected independently at each
scale with a bounded Float64 trust-region solve. Corrected points are matched
between adjacent scales by periodic `L∞` distance, with deterministic
`(distance, previous index, current index)` tie-breaking. Correction statuses
are `converged`, `residual_failed`, `inertia_failed`, `duplicate`, or `failed`.

The generalized Hessian is formed as `F⁻¹ H_theta F⁻ᵀ`, where `K=F Fᵀ`, and
its eigenvalues are reported as canonical/generalized eigenvalues. The
existing screen is applied only after correction: `value > 0`, at least one
negative mode, `epsilon < 1`, and `abs(min_eta) < 1`.

The optional augmented diagnostic solves `gradient=0`, `H*v=0`, and
`dot(v,v)=1`; it reports gradient, null-vector, and normalization residuals.
It is not run automatically for generic geometry rows.

Both the summary report and one-writer shard use the same append-only CSV
schema. `row_type=scale` contains per-geometry/per-scale coverage, resource,
candidate, and bracket aggregates. `row_type=branch` contains the seed and
corrected coordinates, branch provenance/matching IDs, correction status and
residual, inertia, value, gradient norm, epsilon, eta extrema, zeroish-mode
count, candidate status, and failure text. Vector fields use semicolons.

`--resume` skips only completed `(h11, polytope, frst, sampled_scale)` summary
keys after validating the exact header. Existing fixed-point scan CSVs are not
read or overwritten. Defaults for bulk output are under
`/private/tmp/inflation-scale-continuation`.

The default preflight guard is the handoff's 750,000,000-byte nominal stage
allocation policy. It uses the conservative estimate
`branch_count * h11^2 * 4000` before correction. A preflight rejection is
recorded as `branch_coverage_status=resource_cap` with the exact branch
estimate; it is not reported as complete enumeration.

## Benchmark gate

`pilot_benchmark_regression()` passed in the repository environment:

| check | result |
|---|---:|
| N=5 critical scale | `0.674506370003365` |
| N=5 reduced ratio | `0.25` |
| N=5 zero reduced curvature | passed |
| N=8 critical scale | `0.674506370003365` |
| N=8 gradient residual | `2.1e-15` |
| N=8 null residual | `6.3e-13` |
| N=8 zero mode and positive heavy modes | passed |
| N=8 detuned negative-mode count | `1` |

The focused one-axion augmented test also converged with zero gradient, null,
and normalization residuals at the exact cusp.

## Historical ten-geometry calibration

The calibration below predates the physical default and was explicitly run
with `scale_status=homotopy_only`. Its crossings and screen hits must retain
that interpretation; they are not physical-volume results.

The full-enumeration invocation used `--max-branches 400000`, the seven-point
grid above, and the ten audited rows from the handoff. Bulk files were kept
outside Git under `/private/tmp/inflation-scale-continuation/`.

| search | geometries | scales/geometry | complete | resource-capped | max corrected branches | screen-passing corrected branches | corrected candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| all leading branches | 10 | 7 | 2 (`h11=8`) | 8 | 768 | 0 | 0 |
| leading index `1:1` | 10 | 7 | 0 (partial-index search) | 0 | 80 | 8 | 8 |
| leading index `1:2` | 10 | 7 | 0 (partial-index search) | 0 | 680 | 80 | 80 |

For complete enumeration, both h11=8 rows completed within the stage policy.
The h11=11, 13, 14, and 16 rows were conservatively resource-capped before
correction; their exact all-branch estimates were retained in the report.
The low-index comparisons completed all ten geometries. All low-index screen
hits were on detuned homotopy scales, not at the reference scale, and none is
a physical candidate.

## Near-catastrophe table

The only full-enumeration crossing signal was the strong-hierarchy row
`(8,544,1)`. The smallest generalized Hessian eigenvalue changed sign across
each adjacent grid bracket below. Each bracket had 384 matched branch hits;
no branch passed the existing screen.

| geometry | scale bracket | matched branch hits | candidate status | interpretation |
|---|---|---:|---|---|
| `(8,544,1)` | `0.90:0.95` | 384 | near-catastrophe only | homotopy-only diagnostic |
| `(8,544,1)` | `0.95:0.99` | 384 | near-catastrophe only | homotopy-only diagnostic |
| `(8,544,1)` | `0.99:1.00` | 384 | near-catastrophe only | homotopy-only diagnostic |
| `(8,544,1)` | `1.00:1.01` | 384 | near-catastrophe only | homotopy-only diagnostic |
| `(8,544,1)` | `1.01:1.05` | 384 | near-catastrophe only | homotopy-only diagnostic |
| `(8,544,1)` | `1.05:1.10` | 384 | near-catastrophe only | homotopy-only diagnostic |

The weak `(8,1,1)` row had no crossing signal. The low-index runs reproduced
the same strong-row pattern with 21 (`1:1`) or 84 (`1:2`) matched branch hits
per bracket. The weak `(13,1,1)` row produced corrected screen rows only at
homotopy scales `1.01` and `1.10`; it had no reference-scale candidate and no
crossing bracket.

These are not `refined_candidate` or `trajectory_candidate` results. No
trajectory refinement was run because all generic scale results are
homotopy-only and the existing arbitrary-precision refinement boundary is
model-specific to the N=8 benchmark.

## Resource accounting and limitations

The completed all-branch h11=8 rows measured maximum per-scale allocations of
approximately 331 MB for `(8,1,1)` and 128 MB for `(8,544,1)`. The low-index
comparison remained below 161 MB per scale, with maximum wall times below one
second except for the h11=8 rows. An initial unconstrained h11=11 all-branch
attempt allocated multiple gigabytes; the preflight guard was added before
rerunning the calibration. The bounded rerun did not launch those oversized
corrections.

The existing fixed-point calibration remains `0/0`; this pilot does not create
a candidate-recall estimate. A crossing means only that a corrected branch
approaches a generalized-Hessian zero under the documented homotopy. It does
not establish a physical scale, inflationary trajectory, or generic corpus
coverage.

## Recommendation

Do not expand to a production high-h11 scan yet. The pilot recovers the fixed
benchmarks and stays within the bounded rerun after resource preflight, but the
generic crossing signal is homotopy-only and the all-branch resource envelope
already caps h11 >= 11. Proceed only after a documented physical scale model is
available, or treat the next phase explicitly as a mathematical homotopy study
with a more efficient corrector and a separately approved resource budget.
