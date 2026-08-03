# CYAxiverse Spectrum Run Guide

This guide explains how to run the scalable axion spectrum computation and produce the spectrum distributions used for the arXiv:2103.06812 comparison.

## 1. Before starting

Run all commands from the repository root, the directory containing
`Project.toml` and `scripts/`.

The package uses Julia 1.12:

```sh
julia --version
```

Python is **not required** for the spectrum batch or plotting commands in this
guide. They use Julia, HDF5, and the stored geometry data directly. Leave
`PYTHON` unset unless you are using the optional CYTools/PyCall integration.

Set the local geometry database path. This assumes the database is in a `data/`
directory below the repository root; replace it if the database is elsewhere:

```sh
export DATA="${DATA:-$PWD/data}"
```

For CYTools-backed workflows only, set `PYTHON` to the executable in the
environment that contains CYTools before starting Julia:

```sh
export PYTHON="/path/to/cytools-environment/bin/python"
```

The legacy `src/init_python.jl` file and the optional PyCall extension use this
setting to select or rebuild PyCall. It is not part of the normal spectrum
runner path.

## 2. Run a one-geometry pilot

Start with one geometry. This checks the environment, database path, solver, and HDF5 output without committing to a large run.

```sh
julia --project=. scripts/batch_physical_spectrum.jl \
  --data-dir "$DATA" \
  --h11 4 \
  --limit 1 \
  --quartics \
  --prec 200 \
  --force \
  --summary /tmp/physical-spectrum-pilot.csv
```

A successful run prints a line similar to:

```text
success physical=4 ...s
```

The result is written below:

```text
$DATA/h11_004/np_0000001/cy_0000001/cyax.h5
```

The batch runner amends this existing `cyax.h5` file in place under
`spectrum/physical`; it does not create a separate spectrum HDF5 file. Read
the stored result with `CYAxiverse.read.physical_spectrum(geom_idx)`.

The compact result contains:

- physical masses `m`;
- physical mode indices;
- `fK_log10` values;
- diagonal quartic signs and logarithms;
- `fpert_log10` when `--quartics` is enabled.

## 3. Choose the computation mode

### Mass-only mode

Use this when only masses and physical-mode counts are needed. It is faster and uses less memory.

```sh
julia --project=. scripts/batch_physical_spectrum.jl \
  --data-dir "$DATA" \
  --h11 10 \
  --limit 100 \
  --mass-only \
  --prec 200 \
  --summary "$DATA/logs/h11_010_mass_only.csv"
```

### Quartic mode

Use this when reproducing the `fpert` distributions. The production path computes only diagonal quartics for physical modes; it does not build the full quartic tensor.

```sh
julia --project=. scripts/batch_physical_spectrum.jl \
  --data-dir "$DATA" \
  --h11 10 \
  --limit 100 \
  --quartics \
  --prec 200 \
  --summary "$DATA/logs/h11_010_quartics.csv"
```

The perturbative decay constant is computed using

```text
log10(fpert) = log10(m) - 0.5 * log10(abs(lambda_iiii))
```

The stored values use the same units and conventions as the existing high-precision spectrum routines.

## 4. Run the full h11=4..30 range

The database currently contains approximately 26,995 geometries for `h11=4..30`. Run one `h11` value at a time so that progress can be monitored and resumed:

```sh
SUMMARY="$DATA/logs/physical_spectrum_h11_004_030.csv"

for h in $(seq 4 30); do
  julia --project=. scripts/batch_physical_spectrum.jl \
    --data-dir "$DATA" \
    --h11 "$h" \
    --quartics \
    --prec 200 \
    --summary "$SUMMARY" \
    --append-summary
done
```

Do not use `--force` for a resumed run. Existing HDF5 files are skipped, so an interrupted run can be restarted with the same command.

### Run in parallel

The batch runner is process-parallel: run different `h11` values in separate
Julia processes. For an eight-core machine, use eight workers and keep each
worker single-threaded so Julia and BLAS do not oversubscribe the CPU:

```sh
WORKERS=8
PARALLEL_DIR="$DATA/logs/physical_spectrum_h11_parallel"
mkdir -p "$PARALLEL_DIR"
export DATA PARALLEL_DIR
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

seq 4 30 | xargs -P "$WORKERS" -n 1 sh -c '
  h="$1"
  tag=$(printf "%03d" "$h")
  summary="$PARALLEL_DIR/h11_${tag}.csv"
  log="$PARALLEL_DIR/h11_${tag}.log"
  julia --project=. scripts/batch_physical_spectrum.jl \
    --data-dir "$DATA" \
    --h11 "$h" \
    --quartics \
    --prec 200 \
    --summary "$summary" \
    >"$log" 2>&1
' sh
```

Do not use `--append-summary` in the parallel command. Each worker writes a
different summary and log, avoiding concurrent writes to one CSV. After all
workers finish, merge the per-`h11` summaries:

```sh
SUMMARY="$DATA/logs/physical_spectrum_h11_004_030.csv"
first=$(find "$PARALLEL_DIR" -maxdepth 1 -name 'h11_*.csv' -print | sort | head -n 1)
{
  head -n 1 "$first"
  for summary in $(find "$PARALLEL_DIR" -maxdepth 1 -name 'h11_*.csv' -print | sort); do
    tail -n +2 "$summary"
  done
} > "$SUMMARY"
```

Set `WORKERS` to the number of cores you want to use. For example, use
`WORKERS=4` on a four-core machine. The run remains resumable because existing
HDF5 outputs are skipped; rerunning the command refreshes each worker's CSV
and log while continuing from the existing geometry outputs.

Useful controls:

```text
--limit N       process only N selected geometries
--offset N      skip the first N selected geometries
--geometry H,P,F process one explicit geometry
--force         recompute an existing output
--mass-only     skip quartics and fpert
--quartics      compute diagonal quartics and fpert
```

For a controlled chunk, for example:

```sh
julia --project=. scripts/batch_physical_spectrum.jl \
  --data-dir "$DATA" \
  --h11 20 \
  --offset 500 \
  --limit 100 \
  --quartics \
  --prec 200 \
  --summary "$DATA/logs/h11_020_chunk.csv"
```

Each geometry failure is recorded in the CSV summary and does not stop the remaining geometries.

## 5. Make the Appendix B plots

After batch outputs are available, aggregate them by `h11`:

```sh
julia --project=. scripts/reproduce_appendix_b_spectra.jl \
  --data-dir "$DATA" \
  --output-dir "$DATA/physical_spectrum/appendix_b" \
  --h11-min 4 \
  --h11-max 30
```

The plotting script applies Makie's built-in LaTeX-compatible font theme
automatically, so no separate LaTeX installation or font configuration is
needed.

This writes:

```text
$DATA/physical_spectrum/appendix_b/masses_by_h11.pdf
$DATA/physical_spectrum/appendix_b/fpert_by_h11.pdf
$DATA/physical_spectrum/appendix_b/fK_by_h11.pdf
$DATA/physical_spectrum/appendix_b/appendix_b_quantiles.csv
```

The mass plot uses the paper's physical range from approximately `H0` to `M_Pl`. The `fpert` plot uses the diagonal quartic magnitude, and the `fK` plot uses the Kähler metric eigenvalues.

Missing `h11` values appear as empty rows in the quantile CSV. This is expected if the batch is still running.

## 6. Validate the run

Run the package tests from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

The current suite should pass all tests.

For a numerical spot check, compare the hybrid result with the high-precision physical reference on a small geometry:

```sh
env -u PYTHON julia --project=. -e '
using CYAxiverse
ENV["CYAXIVERSE_DATA_DIR"] = ENV["DATA"]
g = CYAxiverse.structs.GeometryIndex(4, 1, 1)
hybrid = CYAxiverse.generate.pq_hybrid_physical_spectrum(g; prec=80, quartics=true, mixed_quartics=false)
reference = CYAxiverse.generate.pq_physical_spectrum(g; prec=80)
println("mass max difference = ", maximum(abs.(hybrid.m .- reference.m)))
println("quartic max difference = ", maximum(abs.(hybrid.λself .- reference.λself)))
'
```

The expected differences should be zero or at the level of floating-point rounding. If the solver emits a provisional warning, record it and do not treat that geometry as a precision validation point without comparing it to the high-precision reference.

## 7. Output and safety notes

- Do not delete existing `cyax.h5` files when restarting a run. Spectrum results
  are stored inside those files under `spectrum/physical`.
- Use a new summary path for a new parameter choice, especially if changing precision, threshold, or quartic mode.
- The batch runner updates the geometry database in place; preserve the
  existing `cytools` groups and datasets in every `cyax.h5` file.
- Do not compare distributions until the CSV shows that the relevant `h11` values have sufficient successful geometry counts.
- The production solver targets physical modes above the Hubble threshold. It does not attempt a full arbitrary-precision eigendecomposition for every mode.
- The full sweep is computationally expensive. Run a pilot and a small high-`h11` slice first, then estimate the total runtime from the recorded `runtime_seconds` values.

## 8. Troubleshooting

### `nothing/cyax.h5` or a missing database file

The data directory was not selected. Set `DATA` and pass `--data-dir "$DATA"` explicitly.

### Python or PyCall initialization errors

Set `PYTHON` before starting Julia:

```sh
export PYTHON="/full/path/to/python"
```

Then rerun the command in a fresh Julia process.

### Existing files are skipped

This is normal for resumable runs. Use `--force` only for the specific geometries that must be recomputed.

### The plot script reports no physical spectrum outputs

Check that the batch runner has written files below:

```text
$DATA/physical_spectrum/
```

Also check that the data directory passed to the plot script is the same one passed to the batch runner.
