# User guide

!!! warning
    Under construction

## Optional plotting

Core package loading does not import the plotting stack. To use the plotting
namespace, load its optional dependencies explicitly:

```julia
using CYAxiverse
using CairoMakie, ColorSchemes

CYAxiverse.plotting.vacua_db_jlm()
```

## Data directory selection

The geometry database is rooted at a directory containing entries such as
`h11_004/np_0000084/cy_0000001/cyax.h5`. When using a checkout of
`CYAxiverse.jl`, the package's general default is `../data`: the `data`
directory one level above the repository directory. This default is anchored
to the package checkout and does not depend on the current working directory.

Data-directory selection follows this order:

1. an explicit `--data-dir` option or function argument;
2. the `CYAXIVERSE_DATA_DIR` environment variable;
3. a recognized legacy deployment alias such as `newARGS=docker`;
4. the checkout-relative default `../data`.

Set `CYAXIVERSE_DATA_DIR` when the database is stored elsewhere:

```sh
export CYAXIVERSE_DATA_DIR=/path/to/data
```

The resolver does not create a missing data directory or silently fall back to
the current working directory. Use `--data-dir` for a different database or
for an isolated output copy.

## Vacua minima pipeline

The vacua pipeline keeps the legacy estimate and the numerical minima search
separate. The search decision tree is:

1. Use the square, determinant-certified leading problem when its branch guard
    applies. This count is exact for that algebraic problem.
2. Use `CYAxiverse.generate.leading_critical_branches` to enumerate the
    selected leading determinant-lattice branches. These branches are a
    deterministic prefilter; they are not a completeness claim for the full
    potential.
3. Use `CYAxiverse.jlm_reduced.prepare` once and repeated
    `CYAxiverse.jlm_reduced.minimize` calls when reduced-JLM data can be reused.
    Finite starts and iteration limits make the result a search-budget result.
4. Use a bounded finite-start critical-point search only when the preceding
    paths do not apply.

Every persisted result must identify its method, threshold, start budget,
residual and merge tolerances, iteration bound, solver status, and runtime.
The single-geometry API defaults to the legacy path and refuses to replace an
existing `vacua_pipeline` group unless `force=true` is explicit. Existing
`spectrum` and legacy vacua groups are preserved.

For resumable production work, use the batch runner with a bounded selection
first:

```sh
julia --project=. scripts/batch_vacua_pipeline.jl \
     --data-dir /path/to/data --geometry 4,50,1 --dry-run
```

The runner skips only completed results whose stored configuration matches the
requested configuration. Mismatched or incomplete results are reported and
require `--force`; each completed geometry is flushed immediately to the CSV
summary. For HPC runs, use `--workers N` to dispatch geometries through
Julia's `Distributed` worker pool. Results are streamed back to the parent one
geometry at a time, while `--batch-size B` limits the number of geometries
held in one dispatch group. Each worker owns its geometry HDF5 operation and
the parent owns summary writes. Set `--blas-threads M` so that `N * M` does not
exceed the allocated physical cores:

```sh
julia --project=. scripts/batch_vacua_pipeline.jl \
    --data-dir /path/to/data --h11 4 --workers 4 --blas-threads 2 \
    --batch-size 16 \
    --starts 2048 --method reduced_jlm \
    --summary /path/to/logs/vacua_h11_004.csv
```

The default is one worker, preserving sequential behavior. Workers never
share HDF5 handles, and failures are returned to the parent for per-geometry
CSV logging.

## Physical PQ spectra

`pq_spectrum` provides the fast leading-Hessian spectrum intended for broad
surveys. For a threshold-certified physical sector, use the separate
high-precision APIs:

```julia
potential = CYAxiverse.read.potential(geom_idx)

count = CYAxiverse.generate.pq_physical_mode_count(
    potential.K, potential.L, potential.Q; prec=200)

spectrum = CYAxiverse.generate.pq_hybrid_physical_spectrum(
    potential.K, potential.L, potential.Q;
    prec=200,
    quartics=false,
    label="my geometry",
)
```

The mode count is confirmed at increasing arbitrary precision. If the hybrid
refinement does not stabilize, it falls back to the full high-precision
eigensystem; a provisional warning is emitted only if that fallback also fails
validation. Such a result should be rerun at higher precision or checked with
`pq_physical_spectrum`.

Set `quartics=false` for large mass-only scans. When quartics are requested,
every input instanton contributes to their contraction even though only the
physical mass eigenvectors are returned. This option is available on both
`pq_physical_spectrum`, `pq_hybrid_physical_spectrum`, and
`pq_window_spectrum`; it leaves the mass-only results unchanged while avoiding
quartic contractions.

For resumable ensemble runs, use `scripts/batch_physical_spectrum.jl`. It
discovers indexed or on-disk geometries, flushes one CSV row per geometry, and
writes compact results into each source `cyax.h5` file under
`spectrum/physical`:

```sh
julia --project=. scripts/batch_physical_spectrum.jl \
    --data-dir /path/to/data --h11 200 --limit 100 --mass-only \
    --summary /path/to/data/logs/physical_spectrum_h11_200.csv

julia --project=. scripts/batch_physical_spectrum.jl \
    --data-dir /path/to/data --geometry 10,1,1 --quartics --force
```

The quartic batch mode computes only diagonal self-couplings and derives
`fpert_log10 = m - 0.5 * lambda_self_log10`; mixed quartic arrays and
eigenvectors are not persisted. Existing output is skipped unless `--force` is
given, and per-geometry failures are recorded without stopping the batch.

The hybrid solver accepts `quartic_backend=:auto` (the default). It retains
the dense path for small or dense charge matrices and switches to sparse charge
transforms for large sparse matrices. Use `quartic_backend=:dense` or
`:sparse` to force a backend while benchmarking. With `quartics=true` and
`mixed_quartics=false`, diagonal contractions stream over instantons and do
not allocate the full physical-mode charge matrix.

`pq_spectrum` returns `m` together with the aligned Hessian-eigenvalue signs in
`msign`. Its sequential PQ decay quantity is `f`; the legacy spectrum HDF5
schema stores this array under `decay/fpert` for compatibility with existing
readers.

### Hierarchy blocks and mass windows

`instanton_scale_blocks(L; gap_log10=1.0, min_block_size=1)` groups contiguous
log-scale entries without materializing their potentially enormous amplitudes.
Each block preserves the zero-based input instanton indices and the result
records every inter-block gap. For a charge-aware perturbative screening,
`instanton_hierarchy_diagnostics(K, L, Q)` also reports canonical charge
coupling, separation-to-coupling ratios, and conservative `certified_safe`
flags. A failed certificate leaves the numerical block/subspace path in use;
scale separation alone is never treated as a proof of decoupling.

Use `pq_window_spectrum` when only a mass interval is needed:

```julia
window = CYAxiverse.generate.pq_window_spectrum(
    potential.K, potential.L, potential.Q;
    min_log10_mass=12.0,
    max_log10_mass=18.0,
    prec=200,
    quartics=false,
)
window.mode_indices
window.diagnostics
```

Both window boundaries are counted with arbitrary-precision inertia, and only
an oversampled band around the requested modes is refined. The inclusive
boundary guard is configurable with `boundary_margin_log10` (default `1e-10`)
and is recorded in the diagnostics. The diagnostics also record
precision-count stabilization, residual convergence, boundary gaps, and
whether the reference eigensystem fallback was needed. A lower-threshold query
is obtained with `max_log10_mass=Inf` and remains compatible with the existing
`pq_hybrid_physical_spectrum` path.

### Large-geometry scaling checkpoint

The mass-only hybrid solver was validated on ten geometries each at
`h11=150`, `180`, and `200`. Physical-mode counts agreed at 150 and 200 digits
for all 30 inputs, and every 200-digit hybrid solve converged without a
provisional warning.

| `h11` | Instantons per input | Physical modes observed | Mean selector time | Mean hybrid time |
|---:|---:|---:|---:|---:|
| 150 | 11,935 | 4–14 | 0.204 s | 5.83 s |
| 180 | 17,020 | 3–9 | 0.437 s | 9.97 s |
| 200 | 20,910 | 5–8 | 0.649 s | 14.05 s |

The selector uses an allocation-free candidate scan and was checked for exact
ordered-column agreement with the previous repeated-rank implementation on
all 1,000 available `h11=10` inputs and all ten `h11=20` inputs. The
high-precision leading Hessian accumulates only over nonzero charge support.
The hybrid solver also reuses its selected data and requested-precision Hessian
between inertia counting and eigenvector refinement.

At `h11=200`, the selected charge matrix in the profiled input was 0.99% dense.
Sparse Hessian accumulation took 0.078 seconds, while canonical whitening took
3.77 seconds. The whitening makes canonical charges dense; a direct
canonical-charge rank-one construction was tested and was slower, so sparse
storage is intentionally confined to the pre-whitening accumulation stage.

## Inflation screening

`CYAxiverse.read.oriented_potential(geom_idx)` is the package-owned
normalization boundary for screening inputs. It returns canonical `Q`, `L`,
and `K` orientations and validates dimensions and finite numerical data; scan
thresholds and resource policy remain script-level decisions.

## Publication plotting

Plotting is an optional package extension, so importing `CYAxiverse` does not
load CairoMakie or a graphical backend. Load the renderer explicitly when a
plot is needed:

```julia
using CYAxiverse
using CairoMakie
using ColorSchemes

const plotting = CYAxiverse.plotting
style = plotting.paper_style(resolution = (900, 650))
```

The default style provides the serif/STIX typography, pale lavender-gray
background, white gridlines, dark spines, and accent palette used by the
reference figures. Pass one `style` object to every panel in a figure set.
The renderers return a `PlotResult` containing the created `figure`, the
rendered `axis`, and the primary `plot` object.

Statistical and function plots use the same axis preparation path:

```julia
x = collect(range(1, 491; length = 100))
fraction_excluded = 0.45 .* exp.(-((x .- 150) ./ 120) .^ 2)
uncertainty = plotting.band(x, fraction_excluded .- 0.02,
    fraction_excluded .+ 0.02; label = "95% C.L.")

exclusion = plotting.exclusionplot(x, fraction_excluded;
    style, bands = (uncertainty,),
    xlabel = raw"Hodge number $h^{1,1}$",
    ylabel = "Fraction excluded",
    label = raw"CY$_3$ manifolds")
plotting.save_plot("exclusion.pdf", exclusion)
```

Use `boxplot` for a vector of grouped samples, `scatterplot` for sampled
geometry data, `minima_plot` for a sampled potential with minimum and critical
point markers, and `trajectoryplot` for columns of an inflationary trajectory
matrix. For multi-panel figures, create a Makie `Figure` and use
`plotting.styled_axis(fig[row, column]; style)` before passing each axis through
`axis = ...`. Use `axis_kwargs` and `plot_kwargs` for renderer-specific
customization without changing the shared style.
