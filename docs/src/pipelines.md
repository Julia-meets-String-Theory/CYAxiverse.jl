# Pipelines

This page is the end-to-end user guide for the two scientific pipelines the
package provides, and for the orientifolded axion database built on top of
them. For the auto-generated function reference, see the
[API](@ref "Available functions") page; every function named here is
documented there from its docstring.

The two pipelines share a front half (CYTools generates geometry and writes an
HDF5 file) and then branch:

1. **Axion spectra** — CYTools geometry → masses `m`, perturbative decay
   constants `fpert`, and quartic self-couplings `λ`.
2. **Vacua and inflation** — CYTools geometry → number of vacua,
   leading-branch fixed-geometry saddle diagnostics, and the number of
   e-folds along candidate inflationary trajectories.

Both pipelines run on the standard Calabi–Yau geometry and on orientifolded
geometry. The orientifolded database in this repository is built for the
inherited-orientifold trilayer sector, `h11₋ = 0` and `h21₊ = 0`.

## 1. Prerequisites and setup

- Julia 1.12 (matches `Project.toml` and CI). Run every command locally, never
  in a sandbox or container:

  ```sh
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

- Loading `CYAxiverse` for the Julia spectrum, vacua, and inflation paths does
  **not** require Python. Set `ENV["PYTHON"]` to a CYTools-enabled interpreter
  only for the optional CYTools/PyCall geometry-generation front half. On a
  typical macOS install the CYTools interpreter is a dedicated conda
  environment, for example
  `/opt/homebrew/Caskroom/miniforge/base/envs/cytools/bin/python`.

- Select the geometry database with `CYAXIVERSE_DATA_DIR` (or the `--data-dir`
  option that the batch scripts accept). See the
  [User guide](@ref "Data directory selection") for the full resolution order.

## 2. The `cyax.h5` data model

Every geometry is one HDF5 file at
`h11_XXX/np_YYYYYYY/cy_ZZZZZZZ/cyax.h5` beneath the selected database root.
Preserve the zero padding and use the [`CYAxiverse.filestructure`](@ref
"CYAxiverse.filestructure") path helpers rather than constructing paths by
hand. All datasets are written with maximum compression (`deflate=9` in Julia,
`gzip` level 9 in the Python writers).

| Group | Written by | Contents |
|---|---|---|
| `cytools/geometric/` | geometry generation | `points`, `simplices`, `h21`, `glsm`, `basis`, `tip`, `tip_prefactor`, `CY_volume`, `divisor_volumes`, `Kinv` |
| `cytools/geometric/visible_sector/` | visible-sector assignment | `qcd_divisor_index`, `qed_divisor_index`, `qcd_divisor_volume`, `qed_divisor_volume`, `qed_volume_upper_bound` |
| `cytools/potential/` | geometry generation | `L` (instanton scales), `Q` (charge matrix) |
| `spectrum/physical/` | pipeline 1 | `m`, `mode_indices`, `fK_log10`, `fpert_log10`, `lambda_self_sign`, `lambda_self_log10`, `metadata/` |
| `vacua_pipeline/` | pipeline 2 | `estimate`, `issquare`, `metadata/` (`auto_selected_method`, `det_Qtilde`, `branch_count`, `search_status`, …) |
| `inflation/` | pipeline 2 | legacy-compatible `catastrophes/` group containing `leading_branch_saddle_count`, `leading_branch_saddles_present`, `leading_minima_count`, saddle and negative-mass counts, plus e-fold flow results; `scale_status=homotopy_only` |
| `orientifold/` | orientifold bridge | involution (`h2_involution_matrix`, `lattice_matrix`, `torus_shift_*`), `lambda_f`, `h11_minus`, `h11_plus`, `h21_plus`, and reproduction provenance |
| `construction_metadata/` | geometry generation | canonical lattice points, face-restriction data |

### Matrix orientation (read this before touching a reader or writer)

The canonical Julia orientation is `Q :: (h11, N)` and `L :: (2, N)`, where `N`
is the number of instantons and `L`'s two rows are `[sign; log10|Λ|]`. HDF5.jl
reads datasets with **reversed axes** relative to `h5py` (column-major vs
row-major), so the Python writers store `Q` and `L` transposed on disk —
`(N, h11)` and `(N, 2)` — so that the raw Julia read yields the canonical
orientation.

- [`CYAxiverse.read.potential`](@ref) and
  [`CYAxiverse.read.potential_factored`](@ref) return the **raw** on-disk
  orientation. The spectrum and vacua engines use this path and require the
  canonical `(h11, N)`/`(2, N)` shapes.
- [`CYAxiverse.read.oriented_potential`](@ref) additionally **repairs**
  orientation, tolerating either stored layout. The inflation path uses it.

Convert an instanton-scale column to a full value only where needed with
`L[:, i][1] * 10 ^ L[:, i][2]`; otherwise preserve the `[sign, log10]`
representation in persisted and inter-module data.

## 3. Pipeline 1 — axion spectra

Compute the physical axion spectrum for every geometry under a database root:

```sh
julia --project=. --startup-file=no scripts/batch_physical_spectrum.jl \
    --data-dir /path/to/database --h11 4 --quartics --prec 200
```

- The engine is [`CYAxiverse.generate.pq_hybrid_physical_spectrum`](@ref): a
  PQ-efficient hybrid that falls back to the full high-precision eigensystem
  when the residual tolerance is not met. For the direct high-precision
  spectrum use [`CYAxiverse.generate.hp_spectrum`](@ref); for the reduced
  physical basis use [`CYAxiverse.generate.pq_spectrum`](@ref).
- `--quartics` computes the quartic couplings; omit it (or pass `--mass-only`)
  for masses and decay constants alone.
- `--prec` sets the arbitrary-precision digit count for the Hessian
  diagonalization and quartic contractions.

### Outputs (`spectrum/physical/`)

| Dataset | Meaning |
|---|---|
| `m` | `log10` of each physical axion mass in eV |
| `mode_indices` | index of each retained physical mode |
| `fK_log10` | `log10` Kähler decay constant, in GeV |
| `fpert_log10` | `log10(fpert/GeV) = log10(m/eV) - 9 - ½·log10(abs(lambda_self))` |
| `lambda_self_log10`, `lambda_self_sign` | signed `log10` of the quartic self-coupling |

Masses, decay constants, and instanton scales span 30+ orders of magnitude, so
these are stored on a `log10` scale. A batch also writes a CSV summary (one row
per geometry) with per-geometry status and spectrum statistics.

The full quartic tensor (the cross-couplings `λ31` and `λ22`) is available from
[`CYAxiverse.generate.hp_spectrum`](@ref) but is not persisted by the batch
driver, which stores the self-coupling `λself`.

## 4. Pipeline 2 — vacua and inflation

Compute vacua, catastrophe presence, and e-folds for one geometry:

```sh
julia --project=. --startup-file=no scripts/vacua_pipeline.jl \
    4 84 1 /path/to/database
```

The three physical quantities come from three established components:

- **Number of vacua** — the hierarchy-truncated potential search, driven
  through `compute_axion_data`/`compute_vacua_data`. **Always select
  `method = :auto`** (determinant count → selected leading-branch count →
  finite multistart lower bound, with a legacy fallback). The algorithm
  identifier is retained only as internal compatibility metadata.
  The `:legacy` default of those entry points uses
  [`CYAxiverse.generate.vacua_id`](@ref), whose reduced charge basis can come
  back empty on geometries with deeply suppressed instantons (for example the
  orientifolded QCD-volume-40 point, where `log10|Λ|` reaches several hundred
  negative), raising a `BoundsError`. The finite-search engine handles those
  geometries; the legacy path is retained only for backward compatibility.
- **Leading-branch saddles** — the generalized-Hessian classification from
  [`CYAxiverse.inflation_points.diagnose`](@ref): a stationary point with one
  or more negative mass-squared modes is a saddle/tachyonic direction. The
  inflation scan records `leading_branch_saddles_present`, a saddle count,
  and `leading_minima_count` (stationary points with zero negative modes).
  This is a bounded fixed-geometry diagnostic, not a catastrophe-continuation
  result; `scale_status=homotopy_only` is persisted.
- **E-folds** — [`CYAxiverse.inflation_points.gradient_flow`](@ref) integrates
  a bounded slow-roll flow from a physical mass-mode displacement and records
  the total and slow-roll e-folds, the exit event, and a `:max_efolds` status
  when the flow reaches its horizon without exiting.

### Outputs

- `vacua_pipeline/` — `estimate` (the vacua count), `issquare`, and a
  `metadata` group recording `auto_selected_method`, `det_Qtilde`,
  `branch_count`, `search_status`, the Git revision, and timing.
- `inflation/catastrophes/` (legacy-compatible group) — `leading_branch_saddles_present`,
  `leading_branch_saddle_count`, `leading_minima_count`, saddle and
  negative-mass counts, and mass bounds. Legacy v1 names are retained only
  under `legacy_v1/` with an explicit migration note.
- `inflation/` e-fold results — per-candidate flow status, total and slow-roll
  e-folds, and the qualified-trajectory count.

Writes are recoverable: the pipeline refuses to overwrite an existing vacua
group unless `force` is set, and finalizes atomically. Distinguish the
estimated, verified, failed, unavailable, and timeout states rather than
inferring a skip from file existence.

## 5. The orientifolded axion database

The orientifold bridge builds a `cyax.h5` database restricted to the inherited
orientifold **trilayer** sector (`h11₋ = 0` and `h21₊ = 0`) of the
Kreuzer–Skarke database, then runs pipelines 1 and 2 on it. It reproduces the
sector counts of Sheridan et al. (arXiv:2412.12012), Table 1: 11, 66, 267, and
1033 accepted classes for `h11 = 2, 3, 4, 5`.

```sh
# Step A–C: build the geometry files for one physical h11
/opt/homebrew/Caskroom/miniforge/base/envs/cytools/bin/python \
    scripts/build_orientifold_axion_database.py --h11 3 \
    --parquet-dir /path/to/ks-mirror \
    --ledger-population-dir /path/to/preserved_population \
    --ledger-name <hXX-cartier-nf.terminal-ledger...zst> \
    --db-root /path/to/orientifold_database --stage full

# then run pipelines 1 and 2 on the new files (Sections 3 and 4)
```

### How the population is selected

1. The accepted `h11₋ = 0` classes come from `terminal_ledger.class_funnel`
   inside the preserved, compressed population ledger (`accepted_for_table_1 ==
   true` entries), each carrying `polytope_id`, `frst_class_index`,
   `frst_hash`, and the accepting O3/O7 involution witness.
2. Each class is re-instantiated in CYTools; the rebuilt FRST is verified
   against the ledger's `frst_hash` (a hard gate — any mismatch stops the run).
3. The `h21₊ = 0` sub-gate is re-derived per class with the paper's own
   `χ(F_I)` Hodge identity (`_h21_plus_zero_diagnostic`, eq. 4.51). The
   resulting trilayer class count must match Table 1 exactly.

### The QCD-viable evaluation point

Each geometry is evaluated at the Kähler point where a candidate QCD divisor
has volume **40**, the established `homogeneous-qcd-volume-40-v1` normalization
(`scripts/qed_divisor_assignment.py`). One `cyax.h5` is written per viable QCD
divisor (so a class contributes several files, one per divisor whose
homogeneous dilation to volume 40 keeps every other divisor volume ≥ 1). The
visible-sector assignment records the QCD divisor and flags a compatible QED
divisor when one is present; QED presence is informational, not a gate.

Because the QCD-volume-40 point makes every instanton action large, the
instanton scales `L` are deeply suppressed (`log10|Λ|` of order −100 to −650).
This is the regime that requires `method = :auto` in pipeline 2 (Section 4).

## Reference: the three "Stage N" vocabularies

There are **three unrelated "Stage N" vocabularies** in this codebase: one for
geometry generation, one for the vacua pipeline, and one for the inflation
scan. The same numbers (`Stage 4`, `Stage 5`) mean different things depending
on which pipeline is being discussed. Each is named explicitly —
`geometry-stage-N`, `vacua-stage-N`, `inflation-stage-N` — so a stage number is
never read against the wrong pipeline.

| Vocabulary | Where | Meaning |
|---|---|---|
| `geometry-stage-N` | `scripts/*.py` | Stage 1 = raw FRST collection; Stage 2 = EFT reference |
| `vacua-stage-N` | `scripts/validate_vacua_stage4_5.jl` | Stage 4 = anchor/regression; Stage 5 = resource benchmarking |
| `inflation-stage-N` | `validation/inflation_scan_call_contract.md` | Stage 3 = HP refinement; 4 = diagnostics; 5 = shards; 6 = stratified pilot; 7 = candidate pilot |

### Geometry pipeline (`geometry-stage-N`)

Python, driven by CYTools. `geometry-stage-1` collects and freezes a fixed
population of fine, star, regular triangulations (FRSTs); `geometry-stage-2`
reads only that frozen population and never resamples an FRST.

```
KS database (cytools.fetch_polytopes) or scripts/manifests/h11_491_11_ks.json
        │
        ▼  scripts/generate_stage1_raw_frsts.py            [geometry-stage-1]
           emits  frst_candidates/*.h5  (lattice pts, simplices, tri hash,
                    optional topology_cache: h11, h21, basis, glsm, kappa, c2, …)
                  frst_terminal_statuses.jsonl, polytope_manifest.json,
                  run_manifest.json
           population is FROZEN at 1400 geometries
        │
        ▼  scripts/generate_stage2_eft_reference.py         [geometry-stage-2]
           reads only Stage-1 artifacts; NEVER resamples an FRST
           topology/cone checks → orientifold → Kähler point
             (--moduli-policy canonical_qcd) → divisor cuts →
             visible sector (intersecting_d7) → assignment pool → --eft rows
           emits  h11_XXX/np_YYYYYYY/cy_ZZZZZZZ/cyax.h5
                  stage2_input_ledger.jsonl, stage2_terminal_statuses.jsonl,
                  eft_models.parquet
```

### Vacua pipeline (`vacua-stage-N`)

Julia, validated by `scripts/validate_vacua_stage4_5.jl`
(`validation/vacua_stage4_5.md`). Both stages are read-only with respect to
geometry HDF5 data.

- **`vacua-stage-4`** — anchor/regression checks: the N=5/N=8 critical and
  minima anchors, the six initial inflation-screen geometries, reduced
  JLM-method `h11=4:11` summaries against the digitized 2023 aggregate, and
  deterministic selected-column ordering at `h11=10` and `h11=20`.
- **`vacua-stage-5`** — resource benchmarking: one available geometry at
  each requested large dimension, calling `compute_axion_data(...; save=false)`
  so no vacua group is created or replaced.

### Inflation pipeline (`inflation-stage-N`)

Julia, contracted by `validation/inflation_scan_call_contract.md`. Stages 6
and 7 both run `scripts/inflation_scan_pilot.jl`; they differ by pilot
configuration (geometry sample and `h11` range), not by driver.

| Stage | Name | Driver |
|---|---|---|
| `inflation-stage-3` | optional high-precision refinement | `scripts/inflation_refinement_common.jl` |
| `inflation-stage-4` | coherent diagnostics | `scripts/inflation_diagnostics_common.jl` |
| `inflation-stage-5` | append-only result and checkpoint shards | `scripts/inflation_scan_shards_common.jl` (via `scripts/inflation_scan_prep.jl`) |
| `inflation-stage-6` | stratified real-geometry pilot | `scripts/inflation_scan_pilot.jl` |
| `inflation-stage-7` | candidate-focused pilot | `scripts/inflation_scan_pilot.jl` (different sample/range) |

`inflation-stage-3` is intentionally script-level and model-specific: the
validated arbitrary-precision solver is `n8_poly102`, and it does not yet
accept an arbitrary scan-prep `Q/L/K` geometry.

### From HDF5 to each pipeline

Once a `cyax.h5` file exists, the Julia-side consumers branch from the same
read:

```
cyax.h5 ──read.potential / read.oriented_potential──► (Q, L, K)
   ├─► generate.pq_spectrum / hp_spectrum
   │      / pq_hybrid_physical_spectrum / pq_window_spectrum
   │        └─► scripts/batch_physical_spectrum.jl → spectrum/physical + CSV
   ├─► generate.LQtilde → αmatrix → jlm_reduced.prepare/minimize
   │        └─► scripts/vacua_pipeline.jl / batch_vacua_pipeline.jl   [vacua-stage-N]
   ├─► inflation_points.prepare_geometry_context → diagnose/correct/gradient_flow
   │        └─► scripts/inflation_scan_prep.jl → shards → merge → pilot reports  [inflation-stage-N]
   └─► axion_photon.run_local_scan → CSV   (no HDF5 persistence yet)
```

Adapted from `validation/2026-08_package_and_performance_review.md`, section 2.
