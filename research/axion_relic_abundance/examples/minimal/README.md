# Minimal example: two geometries, three reheating temperatures

**Inputs.**
- Two genuine CYAxiverse vmm geometries (h¹¹ = 9 and h¹¹ = 15, outside the Shoroq sample), shipped as test fixtures in `tests/external_acceptance/data/`. Their vacuum evidence is in `provenance.json` there.
- `spectra_list.json` lists them with state `DOCUMENTED_EXTERNAL`, so no CYAxiverse or Julia 1.12 environment is needed.
  - Change the state to `"VERIFY"` (and drop `method`, `reference`, `evidence`) to re-run the θ = 0 inertia scan with the producer toolchain.
- `run.json` asks for T_reh = 5 MeV, 10⁴ GeV and 10¹⁰ GeV.

**Run** from `research/axion_relic_abundance/`, with `configs/paths.env` set as in the top-level README. Use a new `WORK` for each run, because stages refuse to overwrite:
```bash
engine/run_prepare_from_spectra.sh examples/minimal/spectra_list.json
RELIC_RUN=examples/minimal/run.json engine/run_relic.sh
RELIC_RUN=examples/minimal/run.json engine/run_theta.sh
```

**What to look at**, all in `$WORK`:

| file | content |
|---|---|
| `relic/tables/master_modes_<label>.csv` | one row per mode: `category` (FROZEN / LATE / POST_REHEATING / PRE_REHEATING), `abundance_state`, `Omega_h2_at_theta_ref` (θ_ref = π/2), `inflation_valid` |
| `relic/tables/geometry_abundance_validity_<label>.csv` | one row per geometry: validity class and θ population. Both geometries are `EVALUATED_POSITIVE`, `PRIMARY` |
| `relic/tables/output_validation.json`, `relic/validation/production_verification.json` | `"all_pass": true` (schema, invariants, independent implementations C and B) |
| `final_analysis/theta/theta_population_summary.json` | primary and sensitivity angle analysis per scenario: N_tight_max, maximal intervals, descriptor |
| `final_analysis/verification/crosscheck_python*.json` | the independent Python cross-check of the interval analysis agrees |

**Expected values.** Ω h² at θ_ref = π/2 is about:

| T_reh | h¹¹ = 9 | h¹¹ = 15 |
|---|---|---|
| 5 MeV | 19.7 | 7.6·10⁻⁴ |
| 10⁴ GeV | 2.16·10⁷ | 1.24 |
| 10¹⁰ GeV | 2.16·10¹³ | 1.24 |

With two geometries, at most one reaches the dark-matter band at any common angle (N_tight_max = 1). The example demonstrates the workflow; it is not a population statement.
