# Plan: scalable ensemble spectrum run for arXiv:2103.06812-style reproduction

## Goal

Build a production-ready spectrum pipeline that can run over `O(200k+)`
Calabi-Yau geometries and generate the axion quantities needed to reproduce
the ensemble-level statistics in arXiv:2103.06812:

- axion mass spectra;
- physical-mode counts above/below the Hubble threshold;
- Kähler decay quantities `fK`;
- diagonal quartic self-interactions `λiiii`;
- perturbative decay quantities `fpert`;
- per-geometry diagnostic/provisional flags;
- per-`h11` summary tables suitable for reproducing mass, quartic, `fpert`,
  mass-window, massless/tachyonic, and later BHSR-exclusion plots.

This task is about the spectrum-generation layer first. Do not attempt the
black-hole-superradiance exclusion-probability machinery until the spectrum
outputs are validated.

## Current repo capabilities to reuse

The relevant existing code paths are:

- `CYAxiverse.generate.pq_spectrum`
  - fast leading-Hessian spectrum;
  - returns masses, signs, quartic-derived `f`, and diagnostics when requested.
- `CYAxiverse.generate.pq_physical_mode_count`
  - high-precision physical-sector count above the Hubble threshold.
- `CYAxiverse.generate.pq_physical_spectrum`
  - high-precision reference physical-sector routine.
- `CYAxiverse.generate.pq_hybrid_physical_spectrum`
  - production candidate for large scans;
  - computes only physical modes above threshold;
  - supports `quartics=false` for mass-only scans;
  - with `quartics=true`, returns diagonal and selected mixed quartics for the
    retained physical modes.
- `scripts/benchmark_hybrid_scaling.jl`
  - current benchmark harness for large-geometry scaling.
- `scripts/batch_jlm_reduced.jl`
  - useful pattern for CLI design, geometry selection, HDF5 output, CSV logging,
    `--limit`, `--offset`, `--force`, and explicit geometry arguments.
- `scripts/vacua_pipeline.jl`
  - useful HDF5 layout precedent for storing spectrum quantities.

The current scaling checkpoint in `docs/src/userguide.md` reports mass-only
hybrid timings:

| h11 | instantons per input | physical modes observed | mean selector time | mean hybrid time |
|---:|---:|---:|---:|---:|
| 150 | 11,935 | 4–14 | 0.204 s | 5.83 s |
| 180 | 17,020 | 3–9 | 0.437 s | 9.97 s |
| 200 | 20,910 | 5–8 | 0.649 s | 14.05 s |

These benchmarks suggest that the core spectrum machinery is likely efficient
enough for `O(200k+)` geometries, provided the production runner stores compact
outputs and avoids full quartic tensors.

## Deliverable 1: production batch spectrum runner

Add a new script:

```text
scripts/batch_physical_spectrum.jl
```

It should follow the structure and ergonomics of `scripts/batch_jlm_reduced.jl`.

Required CLI options:

- `--data-dir PATH`
- `--h11 H`
- `--limit N`
- `--offset N`
- `--geometry H,P,F`
- `--prec N`, default initially `200`
- `--threshold-log10 X`, default `log10(CYAxiverse.generate.constants()["Hubble"])`
- `--quartics`
- `--mass-only`
- `--force`
- `--summary PATH`
- `--append-summary`
- `--hilbert` if existing loaders support this consistently

Recommended behavior:

- Default to mass-only unless `--quartics` is passed.
- Resolve geometries by comparing indexed discovery with filesystem scanning,
  as `batch_jlm_reduced.jl` now does.
- Write one compact HDF5 result per geometry.
- Write or replace one CSV summary by default; append only with
  `--append-summary`.
- Continue on per-geometry failure, recording the error in the summary.
- Flush the summary incrementally so long runs survive interruptions.

## Deliverable 2: output schema

Use an HDF5 group such as:

```text
cytools/spectrum/physical/
```

or, if preserving the existing spectrum location is safer:

```text
spectrum/physical/
```

Store:

```text
metadata/h11
metadata/polytope
metadata/frst
metadata/threshold_log10
metadata/prec
metadata/quartics
metadata/provisional
metadata/runtime_seconds

spectrum/physical/m
spectrum/physical/mode_indices
spectrum/physical/eigenvectors              # optional; can be large
spectrum/physical/lambda_self_sign          # only with quartics
spectrum/physical/lambda_self_log10         # only with quartics
spectrum/physical/fpert_log10               # only with quartics
spectrum/physical/fK_log10                  # if implemented
spectrum/physical/mass_signs_or_inertia     # if available/relevant
```

The CSV summary should include at least:

```text
h11,polytope,frst,status,error,runtime_seconds,prec,threshold_log10,
instantons,physical_count,massless_count,
min_mass_log10,max_mass_log10,median_mass_log10,
quartics,negative_lambda_count,positive_lambda_count,
min_fpert_log10,max_fpert_log10,median_fpert_log10,
provisional
```

If `fK` is not immediately available from `pq_hybrid_physical_spectrum`, add it
as a follow-up after the basic runner works. The 2021 paper uses `fK` as an
important statistical comparison and as a possible proxy for `fpert`.

## Deliverable 3: `fpert` calculation

For each physical mode with mass `m_i` and diagonal quartic `λiiii`, compute:

```text
log10(fpert_i) = log10(m_i) - 0.5 * log10(abs(λiiii))
```

Use the same unit conventions already used by `pq_spectrum` / `hp_spectrum`.
Before trusting a new implementation, compare against an existing
`pq_spectrum` or `pq_physical_spectrum` result on small diagonal test cases in
`test/runtests.jl`.

Important: never compute or store the full quartic tensor for large `h11`.
Only diagonal `λiiii` is required for the main ensemble reproduction.

## Deliverable 4: validation tests

Add tests that cover:

1. The batch runner can process one tiny synthetic or existing test geometry.
2. Mass-only mode returns a valid HDF5 file and CSV row.
3. Quartic mode stores diagonal `λiiii` and `fpert`.
4. `fpert` agrees with the existing small analytic tests in `test/runtests.jl`.
5. A failed geometry records a failed CSV row without killing the batch.

Do not make tests depend on the full local generated-data directory. Use
minimal fixtures or existing deterministic benchmark data.

## Deliverable 5: pilot runs

Run pilots in increasing order:

### Pilot A: smoke test

One explicit geometry.

```bash
julia --project=. scripts/batch_physical_spectrum.jl \
  --data-dir /Users/vmehta/Documents/CYAxiverse/cyaxiverse/data \
  --geometry 10,1,1 \
  --mass-only \
  --force \
  --summary /tmp/cyax_spectrum_smoke.csv
```

Expected result:

- one success row;
- physical count recorded;
- HDF5 output present;
- no provisional warning unless justified.

### Pilot B: small h11 batch

Run mass-only over 100 geometries at `h11=10`.

```bash
julia --project=. scripts/batch_physical_spectrum.jl \
  --data-dir /Users/vmehta/Documents/CYAxiverse/cyaxiverse/data \
  --h11 10 \
  --limit 100 \
  --mass-only \
  --force \
  --summary /Users/vmehta/Documents/CYAxiverse/cyaxiverse/data/logs/physical_spectrum_h11_010_100_mass_only.csv
```

Expected result:

- zero or near-zero failures;
- stable runtime estimate;
- physical-count and mass-window summaries are plausible.

### Pilot C: quartic/fpert benchmark

Run with quartics enabled on representative slices:

- `h11=10`, 100 geometries;
- `h11=50`, 20 geometries if available;
- `h11=150`, 10 geometries if available;
- `h11=200`, 10 geometries if available.

Record:

- mass-only runtime;
- quartic-enabled runtime;
- physical mode counts;
- memory behavior;
- provisional warnings.

This is the decisive benchmark for whether `fpert` can be computed directly
for all `200k+` geometries or whether high-`h11` should use `fK` as a proxy
except on validation subsamples.

## Deliverable 6: per-h11 aggregation script

Add:

```text
scripts/summarize_physical_spectrum.jl
```

or a Python script if plotting/CSV handling is easier.

Inputs:

- one or more batch summary CSVs;
- optional HDF5 result directories.

Outputs:

```text
paper_benchmarks/2021_bhsr/spectrum_summary_by_h11.csv
paper_benchmarks/2021_bhsr/mass_windows_by_h11.csv
paper_benchmarks/2021_bhsr/fpert_summary_by_h11.csv
paper_benchmarks/2021_bhsr/fK_summary_by_h11.csv
```

Mass windows to include:

- below Hubble / effectively massless;
- birefringence window: `10^-33 eV <= m <= 10^-28 eV`;
- stellar black-hole-superradiance-relevant windows;
- supermassive black-hole-superradiance-relevant windows.

Use the exact window definitions from `KSAxiverseBHSR.tex` before finalizing.

## Deliverable 7: reproducibility notes

Update a checkpoint/log file after implementation with:

- commands run;
- runtime tables;
- failure counts;
- output file paths;
- whether quartics are production-ready;
- whether `fK` proxy is needed at high `h11`;
- remaining blockers before BHSR exclusion reproduction.

## Guardrails

- Do not rewrite core spectrum algorithms unless a benchmark demonstrates a
  concrete scaling or correctness problem.
- Do not attempt full mixed quartic tensors at large `h11`.
- Do not silently overwrite existing spectrum outputs unless `--force` is set.
- Do not make long-running pilots part of the default test suite.
- Treat provisional hybrid results as warnings in the CSV summary.
- Keep output compact: a 200k-geometry run can become storage-bound if every
  eigenvector and mixed quartic is written unnecessarily.

## Acceptance criteria

The task is complete when:

1. `scripts/batch_physical_spectrum.jl` exists and supports mass-only and
   quartic/fpert modes.
2. A one-geometry smoke run succeeds.
3. A 100-geometry mass-only pilot succeeds.
4. A quartic/fpert pilot on at least one representative batch succeeds.
5. The summary CSV contains enough information to aggregate by `h11`.
6. Tests pass with:

   ```bash
   julia --project=. test/runtests.jl
   ```

7. `git diff --check` is clean.

## Suggested next task after this plan

Once the spectrum batch runner is validated, implement the BHSR
post-processing layer separately. That layer should consume the compact
spectrum summaries and reproduce:

- mass-distribution plots;
- `fpert` and `fK` distributions;
- mass-window counts;
- eventually the exclusion-probability curves and Table 1 peak exclusion
  fractions from arXiv:2103.06812.
