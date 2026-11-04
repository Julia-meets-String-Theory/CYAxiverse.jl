# Methodology v2 examples: constant pre-reheating equation of state w_R

Same inputs as `examples/minimal/` (two non-Shoroq geometries); only the run file differs. Use a new `WORK` for each run (stages refuse to overwrite).

| run file | scenarios | shows |
|---|---|---|
| `run_v2_equation_of_state_scan.json` | T_reh = 10⁴ GeV, w_R = −1/2, −1/3, 0, 0.17, 1/3 | the effect of w_R on the PRE abundances at fixed T_reh |
| `run_v2_mixed_scenarios.json` | (10⁴ GeV, 0.17), (10¹⁰ GeV, −1/3), (5 MeV, −1/2), (10⁴ GeV, default) | different (T_reh, w_R) pairs in one run; a scenario without `w_R` uses 0.0 |
| `run_v2_w0_reproduces_v1.json` | the three reference temperatures at w_R = 0 | v2 at w_R = 0 equals the v1 example apart from the `w_R` column |

```bash
engine/run_prepare_from_spectra.sh examples/minimal/spectra_list.json
RELIC_RUN=examples/wR_general/run_v2_equation_of_state_scan.json engine/run_relic.sh
RELIC_RUN=examples/wR_general/run_v2_equation_of_state_scan.json engine/run_theta.sh
```

Read `relic/tables/master_modes_<label>.csv` (columns `w_R`, `abundance_branch` = `fuzzy_wR_general_eq2.10_2.15` for PRE modes with w_R ≠ 0, `Omega_h2_at_theta_ref`), the per-geometry
`geometry_summary_<label>.csv`, `relic/run_config.json` (`methodology_version`, `reheating.scenarios[].w_R`) and `final_analysis/theta/theta_summary_final.json` (one entry per scenario, with `w_R`).
Ω(w_R)/Ω(0) = (m/3H_R)^(2w_R/(1+w_R)) for PRE modes; POST, LATE, FROZEN modes and all classes are unchanged. Accepted: −1 < w_R ≤ 1. Warned: w_R < −1/3 (accelerating epoch).
The 10⁴ GeV scan is quick; strongly negative w_R at low T_reh can stop with a "float64 normal range" message (nothing is clamped): see `METHODOLOGY.md`.
