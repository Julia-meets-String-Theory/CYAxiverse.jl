# Shoroq reference case (frozen 2026-09-30 / 2026-10-01; full reproduction verified 2026-10-29)

The thesis dataset: 4700 Kreuzer–Skarke polytopes (h11 = 4–50, 100 per h11) -> 4491 accepted geometries ->
16,313 physical modes; relic abundance at T_reh = 5 MeV, 10⁴ GeV, 10¹⁰ GeV; common-angle analysis on 0 < θ ≤ 0.5.
It is a reference case of the general engine: its sample, expected counts and frozen checksums live only here.

| file | role |
|---|---|
| `data/polytope_manifest_h11_004_050.json` | the 4700 polytopes (vertices, unified positions, normal-form hashes, seeds 20260930 + position, historical tranche) |
| `configs/run.json` | the three reheating scenarios (default theta methodology) |
| `configs/expectations.json` | expected population counts and frozen checksums; fills the `@expectation` slots of the methodology files |
| `configs/relic_production_config.frozen.json`, `configs/theta_analysis_config.frozen.json` | the frozen production configurations; the composer must rebuild them byte for byte from methodology v1 + `run.json` + `expectations.json` (checked by `regression_relic_theta.sh`) |
| `sample/` | optional stage 1: ordered scan of the KS mirror and the legacy selection rule; must reproduce the manifest |
| `run_reference.sh` | producers + engine with the Shoroq configuration (about 1.5 h with 4 workers) |
| `tests/` | release-blocking regressions against the frozen data (below) |

```bash
references/shoroq/run_reference.sh                                   # WORK from configs/paths.env
references/shoroq/tests/regression_relic_theta.sh FROZEN_ROOT WORK   # stages 4-5 on the frozen relic input (48 checks)
references/shoroq/tests/regression_subset.sh FROZEN_ROOT WORK 11 12  # producers on 12 polytopes at h11 = 11 (repaired saddle included)
references/shoroq/tests/regression_full.sh FROZEN_ROOT WORK          # all 4700 polytopes, stages 2-5, both directions, workspace untouched
```

What the regressions require:
- relic and theta: byte-identical A1 canonical tables and relic tables; the theta **sensitivity** population (all 4491
  geometries, the frozen population) reproduces every frozen theta file (CSV byte for byte, summary JSON identical apart
  from its `population` label); the **primary** population (4162 geometries without inflation-excluded mode, 2026-10-31
  decision) is pinned by its denominators and N_max; independent verification and benchmark identical to the frozen run
  (benchmark: one documented NA → 0 mapping); theta input verification and Python cross-checks of both populations;
- subset and full: every raw, materialized and spectrum dataset identical; every vacuum-repair record and relic-input
  manifest field identical; population complete in both directions; frozen checksums intact before and after; no file
  of the frozen workspace modified.

Accepted metadata-only differences (exact rules in the tests): tranche-2 files carry their tranche-local `np_` index in
the frozen data and the unified position in a unified run (`metadata/polytope`, path components, label columns); HDF5
files carry run-time metadata (file checksums); run metadata records script/config locations. The frozen
`qcd_volume_exact_rational` cross-check of the 6 beyond-tolerance QCD flags is not recomputed by the producer.

## Accepted dataset (reference values)

| quantity | value |
|---|---|
| input polytopes | 4700 (h11 = 4–50, 100 per h11) |
| accepted geometries | **4491** (209 polytopes rejected by the adaptive prescription; no replacement) |
| axion directions Σh11 | 118,425 |
| modes stored by the spectrum solver (m > H0 at θ = 0) | 16,250 |
| θ = 0 saddles repaired at validated minima | 46 geometries (305 → 368 modes) |
| **physical modes, relic input** | **16,313** |
| QCD-edge flags (kept) | 539, of which 6 beyond the 1e-5 gate tolerance |
| FROZEN / LATE / POST / PRE (5 MeV; 1e4 GeV; 1e10 GeV) | 83/930/3132/12168; 83/930/9607/5693; 83/930/12250/3050 |
| inflation-excluded (m > 3H_I,max) | 365 modes in 329 geometries (all PRE) |
| modes in the abundance sum | 14,935 per scenario |
| max N_TIGHT at a common θ (5 MeV; 1e4 GeV; 1e10 GeV), frozen population = sensitivity population (4491 geometries) | 32 at 0.447 (boundary-limited); 41 at 0.035; 33 at 2.5e-6 (no robust preferred angle) |
| the same, primary population (4162 geometries; the 329 PARTIAL_VALIDITY geometries excluded, decision of 2026-10-31) | 25 at 0.448 (boundary-limited); 40 at 0.035; 29 at 2.53e-6 (no robust preferred angle) |

## Datasets obtained separately (not in Git)

| data | needed for | size | where |
|---|---|---|---|
| Kreuzer–Skarke 4d polytopes, HuggingFace `calabi-yau-data/polytopes-4d` rev `60c0e119…` | stage 1 only (optional) | about 1 GB of HTTP range reads | public |
| **Frozen reference** (`FROZEN_ROOT`), the accepted dataset and results of 2026-09-30/10-01, in the layout documented in `references/shoroq/tests/verify_frozen_reference.py` | regression tests only | about 3.6 GB in total (see below) | the author's archive. It must be deposited with its SHA256SUMS before the code is published. |

The frozen reference consists of:

| directory | size |
|---|---|
| `relic_abundance_input/` (spectra view resolved) | 0.9 GB |
| `relic_abundance_production/` | 25 MB |
| `relic_abundance_production_prep/` | 14 MB |
| `relic_abundance_final_analysis/` | 48 MB |
| `thesis_g2_adaptive/` (both tranches and the unified view) | 2.6 GB |
| `spectrum_count_audit/results/accounting_all.jsonl` | 5 MB |
| `tachyonic_vacua_audit/results` and `corrected_spectra/` | 12 MB |

Everything else is either committed (`references/shoroq/data/`, `data/`) or regenerated under `$WORK`.

