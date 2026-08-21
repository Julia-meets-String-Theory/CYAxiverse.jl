# Pipelines

!!! warning
    Under construction

There are **three unrelated "Stage N" vocabularies** in this codebase: one
for geometry generation, one for the vacua pipeline, and one for the
inflation scan. The same numbers (`Stage 4`, `Stage 5`) mean different
things depending on which pipeline is being discussed. This page names each
vocabulary explicitly — `geometry-stage-N`, `vacua-stage-N`,
`inflation-stage-N` — so a stage number is never read against the wrong
pipeline.

| Vocabulary | Where | Meaning |
|---|---|---|
| `geometry-stage-N` | `scripts/*.py` | Stage 1 = raw FRST collection; Stage 2 = EFT reference |
| `vacua-stage-N` | `scripts/validate_vacua_stage4_5.jl` | Stage 4 = anchor/regression; Stage 5 = resource benchmarking |
| `inflation-stage-N` | `validation/inflation_scan_call_contract.md` | Stage 3 = HP refinement; 4 = diagnostics; 5 = shards; 6 = stratified pilot; 7 = candidate pilot |

## Geometry pipeline (`geometry-stage-N`)

Python, driven by CYTools. `geometry-stage-1` collects and freezes a fixed
population of fine, star, regular triangulations (FRSTs); `geometry-stage-2`
reads only that frozen population and never resamples an FRST.

```
KS database (cytools.fetch_polytopes) or scripts/manifests/h11_491_11_ks.json
        │
        ▼  scripts/generate_stage1_raw_frsts.py            [geometry-stage-1]
           emits  frst_candidates/*.h5  (lattice pts, simplices, tri hash,
                    optional topology_cache: h11, h21, basis, kappa, c2, …)
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

## Vacua pipeline (`vacua-stage-N`)

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

## Inflation pipeline (`inflation-stage-N`)

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

## From HDF5 to each pipeline

Once `geometry-stage-2` has written `cyax.h5`, the three Julia-side
consumers branch from the same read:

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
