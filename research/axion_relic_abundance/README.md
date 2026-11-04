# Axion relic abundance for CYAxiverse (vmm) samples

This directory (`research/axion_relic_abundance/` of CYAxiverse.jl) computes the misalignment relic abundance of the
physical axion modes of Calabi–Yau compactifications produced with CYAxiverse.jl (branch `vmm`), with the validated
methodology of the Shoroq master's thesis study (v1), optionally with a **constant pre-reheating equation of state w_R per
reheating scenario (methodology v2**, section below). A researcher can:

- start from **existing CYAxiverse physical spectra** (no geometry generation, no spectrum recomputation), or
- start from a **polytope manifest** and let the producers generate geometries, spectra and validated vacua first;

then choose **reheating temperatures**, compute **mode-level and geometry-level** abundances, and optionally run the
**common-misalignment-angle analysis**.

The Shoroq dataset (4,491 geometries, 16,313 physical modes) is a separate, fully reproducible **reference case** under
`references/shoroq/`. Nothing in the engine depends on its manifests, checksums or population counts.

- What comes from CYAxiverse vmm and what is added here; adopted conventions: **`SCIENTIFIC_SCOPE_AND_CONVENTIONS.md`**
- Methodology, validity range, assumptions: **`METHODOLOGY.md`**
- Input/output formats and units: **`DATA_CONTRACTS.md`**
- Shoroq reference case and its regression tests: **`references/shoroq/README.md`**
- Origin of every file and every change made during the generalisation: `PROVENANCE.md`; integration architecture: `INTEGRATION_DESIGN.md`

## Scientific scope in brief

The **physics of the spectra is CYAxiverse vmm's**, used unchanged:
- the geometry generator and the axion potential (instanton charges, amplitudes and the pair-term prescription), following
  the framework of N. Gendler et al., *Axion Minima in String Theory*, arXiv:2309.01831;
- the Kähler metric, and the physical-spectrum solver with its mass, quartic and decay-constant conventions.

**This contribution** adds the analysis on top of those spectra:
- vacuum-validity bookkeeping of the inputs;
- reheating-dependent misalignment abundances, mode classes and inflation-validity exclusions;
- per-geometry summaries with explicit zero/NA semantics;
- the common-initial-angle analysis;
- independent numerical verification.

**Conventions.**
- The vmm kinetic metric differs by a factor of two from the literal N = 1 supergravity normalisation. This is an open
  upstream convention question.
- The pair-term signs are a modelling prescription, not first-principles results.
- Both are inherited without modification and no quantity is rescaled (`SCIENTIFIC_SCOPE_AND_CONVENTIONS.md` §3).
- Results are therefore results in the vmm conventions.

## Quick start (about one minute, no Shoroq data)

Engine-only toolchain: Python 3.11 with `environment/python_engine_requirements.txt`, and Julia 1.8.4 with
`environment/julia_analysis` (see Requirements). Set `PYTHON`, `JULIA_ANALYSIS` and `WORK` in `configs/paths.env`, then:
```bash
engine/run_prepare_from_spectra.sh examples/minimal/spectra_list.json
RELIC_RUN=examples/minimal/run.json engine/run_relic.sh
RELIC_RUN=examples/minimal/run.json engine/run_theta.sh
```
`examples/minimal/README.md` explains the inputs and how to read the outputs.

## Choosing the pre-reheating equation of state w_R (methodology v2)

Methodology v1 fixes w_R = 0 (modulus-dominated epoch before reheating). With a **v2 run file** (`"schema": "cyaxiverse_relic_run/2"`) every scenario may state
its own constant `w_R` with **-1 < w_R <= 1** (default 0.0), together with its `T_reh_GeV`; several (T_reh, w_R) pairs can be compared in one run. Same three commands, other run file:
```bash
engine/run_prepare_from_spectra.sh examples/minimal/spectra_list.json
RELIC_RUN=examples/wR_general/run_v2_equation_of_state_scan.json engine/run_relic.sh      # w_R = -1/2, -1/3, 0, 0.17, 1/3 at T_reh = 10^4 GeV
RELIC_RUN=examples/wR_general/run_v2_equation_of_state_scan.json engine/run_theta.sh      # common-angle analysis for every (T_reh, w_R)
```
Other examples: `run_v2_mixed_scenarios.json` (different T_reh and w_R in one run, one scenario using the default), `run_v2_w0_reproduces_v1.json` (v2 at w_R = 0
reproduces the v1 example bit for bit apart from the extra `w_R` column). In a run file, a scenario is `{"label": "T1e4-w0p17", "T_reh_GeV": 1e4, "w_R": 0.17}`.

- What changes: only the PRE_REHEATING abundances, Ω(w_R) = Ω(0) (m/3H_R)^(2w_R/(1+w_R)). Classes, inflation validity, POST/LATE/FROZEN values, the θ analysis rules do not.
- Refused: w_R <= -1 or > 1, non-numbers, a repeated (T_reh, w_R), a `w_R` key under methodology v1. Warned: w_R < -1/3 (accelerating epoch, not conventional reheating),
  w_R < -0.9. The run stops (nothing clamped) if a mode's abundance leaves the float64 range, which happens for strongly negative w_R at low T_reh.
- Every output table has a `w_R` column; `relic/run_config.json` records `methodology_version` and each scenario's `w_R`.
- Physics, derivation, domain, numerical range and the growing sudden-onset systematic at negative w_R: `METHODOLOGY.md` ("Methodology v2"). Validation: `python tests/test_wR_general.py WORK`.

## Questions a new user asks

| question | answer |
|---|---|
| What is calculated? | The misalignment relic abundance Ω h² of each physical axion mode of a CYAxiverse geometry, per reheating temperature, the per-geometry sum, and which geometries reach the dark-matter band at a common initial angle. |
| Relation to CYAxiverse? | It reads the physical spectra written by `scripts/batch_physical_spectrum.jl`; the spectra, potential and conventions are vmm's, unchanged. Nothing here changes `src/`. |
| Inherited assumptions? | vmm pair-term prescription (arXiv:2309.01831 framework) and kinetic normalization (factor-two question open upstream); methodology v1 cosmology (`METHODOLOGY.md`). |
| Inputs? | One `cyax.h5` per geometry plus a vacuum statement, or a polytope manifest for the producers. |
| Own spectra / own (m, f) tables? | Own vmm spectra: yes (route A). Direct mass and decay-constant tables: **not supported**; write a contract-valid `cyax.h5` (`DATA_CONTRACTS.md` §3a). |
| Multiple triangulations? | `GEOMETRY_SELECTION=all` (default) or `first_frst`; see `DATA_CONTRACTS.md` §4. |
| Reheating scenarios? | The run file (labels and T_reh in GeV, and w_R with a v2 run file); hard limits, BBN notes and the supported w_R domain in `METHODOLOGY.md`. |
| Output tables? | See Outputs below and `DATA_CONTRACTS.md` §6; zero, partial and undefined results are distinct classes. |
| Reproduce the frozen reference? | `references/shoroq/` (needs the external frozen archive). |
| Limits of the approximations? | `METHODOLOGY.md`, "Accuracy and systematic uncertainty": the sudden-onset estimate is accurate to an order-one factor only, which the 10% TIGHT band does not cover. |

Values in `configs/paths.env` containing spaces must be quoted (`PYTHON="/path with spaces/bin/python"`).

## Layout

```
engine/                     general computation (no reference data)
  input/                    existing spectra -> relic input (vacuum-validation states)
  relic/                    run-configuration composer, A1 mode table, relic driver (physics block frozen), validators C and B
  theta/                    optional common-angle analysis and its independent cross-check
  run_prepare_from_spectra.sh, run_relic.sh, run_theta.sh, common.sh
producers/                  optional: compute what the engine needs from a polytope manifest
  geometry/, run_geometry.sh          G2 geometries, materialization, audits, physical spectra (upstream vmm code)
  vacuum/,   run_vacuum_repair.sh     theta = 0 inertia scan, validated minima of saddles, relic input
configs/                    methodology_v1.json (frozen) and methodology_v2.json (w_R per scenario), theta_methodology_v1.json, reheating_policy_v1/v2.json,
                            relic_output_schema.json and relic_output_schema_v2.json,
                            run.example.json, spectra_list.example.json, paths.env.example
data/standardmodel2018.dat  g_rho(T), g_s(T) (Saikawa & Shirai, arXiv:1803.01038), sha256-pinned
references/shoroq/          frozen reference case: configs (run, expectations, frozen configurations), polytope manifest,
                            sample reconstruction (stage 1), run_reference.sh, regression tests
examples/minimal/           two shipped non-Shoroq geometries, three T_reh: relic + theta in about a minute
examples/wR_general/        v2 run files: w_R scan, mixed (T_reh, w_R) scenarios, w_R = 0 reproducing v1
tests/                      test_static.py, test_external_acceptance.py (+ fixtures), test_general.sh (+ general_helpers.py),
                            test_wR_general.py (methodology v2 validation), ode_onset_check_wR.py
environment/                Julia projects (vmm 1.12.7; analysis 1.8.4) and Python requirements
```

## Requirements and installation

| component | version | needed for |
|---|---|---|
| Python | 3.11 with `environment/python_engine_requirements.txt` (numpy, pandas, h5py, mpmath, **scipy**; from PyPI) | every stage (engine) |
| Julia | **1.8.4** with `environment/julia_analysis/` (CSV, DataFrames, JSON, HDF5; resolved for 1.8.4, does not precompile under 1.12) | relic and theta stages |
| CYAxiverse.jl | branch `vmm`. Default: the enclosing checkout. Reproduction pin `a5eacd8`; other commits need `ALLOW_OTHER_CYAX_COMMIT=1`, then re-validate | producers; `VERIFY` entries of existing spectra only |
| Julia | 1.12.7, with `environment/julia_vmm/Manifest.toml` copied to the CYAxiverse root and instantiated | producers; `VERIFY` entries only |
| Python | full `environment/python_requirements.txt` (CYTools 1.4.12 from https://cy.tools; MOSEK must not be active) | producers only |

**First run (engine only, existing spectra with documented vacuum evidence):**
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r environment/python_engine_requirements.txt
```
```bash
cp configs/paths.env.example configs/paths.env      # set PYTHON, JULIA_ANALYSIS, WORK (JULIA_VMM, CYAX_REPO only for producers/VERIFY)
```
Julia 1.8.4: official binaries at https://julialang.org/downloads/oldreleases/ (macOS, Linux, Windows). Instantiate the
analysis project once (needs network) and check that it loads; `Pkg.instantiate()` alone can succeed although a package
fails to precompile:
```bash
julia --project=environment/julia_analysis -e 'using Pkg; Pkg.instantiate(); using CSV, DataFrames, JSON, HDF5; println("OK ", VERSION)'
```
Then run `tests/test_external_acceptance.py` (below): it needs no other data. `engine/common.sh` checks the toolchain a
stage needs and stops with an explicit `ERROR:` when something is missing. Every stage reads only `$WORK`, refuses to
overwrite earlier outputs, and stops when a previous gate did not pass. Outputs are not relocatable: a moved or copied
`$WORK` is detected and refused (rerun the relic stage in the new location).

## Use

**A. Existing spectra** (one `cyax.h5` per geometry, CYAxiverse layout `h11_XXX/np_XXXXXXX/cy_XXXXXXX/cyax.h5`):
```bash
engine/run_prepare_from_spectra.sh my_spectra.json      # see configs/spectra_list.example.json
RELIC_RUN=my_run.json engine/run_relic.sh               # see configs/run.example.json
RELIC_RUN=my_run.json engine/run_theta.sh               # optional common-angle analysis
```
Every geometry needs an explicit vacuum-validation state: `VERIFY` (the θ = 0 inertia scan runs on the potential
stored in the file) or `DOCUMENTED_EXTERNAL` (method, reference and checksummed evidence). Unverified or tachyonic
geometries stop the build or are excluded, as the list's `vacuum_policy` says; they never enter silently.

**B. Polytope manifest** (custom sample, any size; layout in `DATA_CONTRACTS.md`):
```bash
POLYTOPE_MANIFEST=my_manifest.json producers/run_geometry.sh
POLYTOPE_MANIFEST=my_manifest.json producers/run_vacuum_repair.sh
RELIC_RUN=my_run.json engine/run_relic.sh && RELIC_RUN=my_run.json engine/run_theta.sh
```

**C. Shoroq reference**: `references/shoroq/run_reference.sh` (about 1.5 h with 4 workers); comparison with the
frozen data: `references/shoroq/tests/regression_full.sh`.

The run file sets the reheating scenarios (labels and T_reh in GeV), the approval flag for runs of more than 1000
modes, and optional theta-analysis overrides. Hard limits on T_reh stop the run; BBN warnings are recorded in
`relic/scenario_checks.json` (`METHODOLOGY.md`).

## Outputs (`$WORK`)

| path | content |
|---|---|
| `relic_input/` | relic-input manifest (one record per geometry, with vacuum validation), CSV, `SHA256SUMS.txt`, `spectra/` view |
| `vacuum_check/` | inertia-scan records of the VERIFY entries (direct-spectra input) |
| `relic/template.json`, `relic/run_config.json`, `relic/scenario_checks.json` | composed configuration, run-time locations and checksums, T_reh warnings |
| `relic/a1/` | canonical per-mode and per-geometry tables, `vacuum_validation.csv`, all input checks |
| `relic/tables/` | `master_modes_<label>.csv` (per mode), `geometry_summary_<label>.csv` (per geometry, every usable geometry, also without mode above H0), `geometry_abundance_validity_<label>.csv` (class EVALUATED_POSITIVE / EVALUATED_ZERO / PARTIAL_VALIDITY / INFLATION_INVALID_ALL and theta population), `population_validity_summary.json`, `h11_summary_<label>.csv`, `global_summary.json`, `run_metadata.json` |
| `relic/validation/` | schema validation, independent verification (C on every row, 50-digit B on a stratified sample) |
| `final_analysis/` | `theta_config.json`, `theta/` (primary population: complete abundances; per-geometry thresholds, exact N_TIGHT maxima, robustness, h11 tables, curves, `theta_population.csv` and `theta_population_summary.json`), `theta_sensitivity_partial_validity/` (labelled sensitivity population including partially valid geometries), `verification/` (cross-check of both) |

## Tests

```bash
python tests/test_static.py                                        # seconds, no data
python tests/test_external_acceptance.py WORK                       # no external data (shipped non-Shoroq fixtures), about 10 min
python tests/test_producer_selection.py [WORK]                      # seed precedence and FRST selection: unit part in seconds; with WORK and the producer toolchain, end to end (about 8 min)
python tests/ode_onset_check.py                                    # exact field evolution vs the m = 3H sudden-onset estimate (METHODOLOGY, accuracy section)
tests/test_general.sh FROZEN_ROOT WORK                              # general samples from the frozen Shoroq spectra, about 2 min
references/shoroq/tests/regression_relic_theta.sh FROZEN_ROOT WORK  # Shoroq stages 4-5, byte-identical, about 2 min
references/shoroq/tests/regression_subset.sh FROZEN_ROOT WORK [H11] [N]   # Shoroq producers on a subset, about 10 min
references/shoroq/tests/regression_full.sh FROZEN_ROOT WORK         # full Shoroq reproduction, about 1.5 h
```
`test_external_acceptance.py` uses two genuine non-Shoroq geometries shipped in `tests/external_acceptance/data/` and
SYNTHETIC variants built from them: zero-mode geometries, integer-valued logarithms, no geometry in the TIGHT band
(NO_MATCH), geometries without CDM-eligible mode, class-threshold masses, tampered tables, strict run files, engine-only
toolchain, moved/copied WORK, invalid HDF5 inputs, and a byte-identical repeat; each case states the required behaviour.
`test_general.sh`, `regression_*.sh` need the frozen Shoroq reference (about 3.6 GB, `references/shoroq/README.md`).
`test_general.sh` uses no population count:
- one geometry and a 25-geometry subset read directly from existing spectra, exercising every vacuum state;
- the Shoroq reheating temperatures plus custom ones (20 MeV, 1e7 GeV, 1e14 GeV);
- refusals of invalid configurations and inputs.

Every row of a Shoroq scenario must equal its frozen row, and the independent verifications must pass. The Shoroq
regressions are release-blocking: every Shoroq table and theta result must stay byte-identical.
