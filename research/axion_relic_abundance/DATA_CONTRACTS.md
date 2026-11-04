# Data contracts: inputs, outputs, units

All JSON files carry a `schema` string; a file with another schema is refused. Paths inside a list file may be
absolute or relative to that file. Geometry identifiers follow the CYAxiverse data layout:
`geometry_id = h11_XXX/np_XXXXXXX/cy_XXXXXXX` (h11, polytope position, FRST index; zero-padded to 3, 7, 7 digits).
A mode is identified by `(geometry_id, mode_index)`.

## 1. Run file (`cyaxiverse_relic_run/1`; example `configs/run.example.json`)

| key | type | meaning |
|---|---|---|
| `scenarios` | list of `{label, T_reh_GeV}` | reheating scenarios; label 1–32 characters of `[A-Za-z0-9.+-]`, unique (used in file names); T_reh in **GeV** |
| `production_run_approved` | bool (default false) | must be true for an abundance run on more than 1000 modes |
| `theta_analysis` | object (optional) | overrides of `configs/theta_methodology_v1.json`: `theta.theta_max`, `dark_matter.{Omega_DM_h2, TIGHT_DM}`, `optimisation.{smoothing_half_widths_dex, primary_smoothing_half_width_dex, plateau_fraction, bootstrap_resamples, bootstrap_seed, robust_criteria}` (ranges in `METHODOLOGY.md`) |
| `description` | string | free text |

Unknown keys are refused at every level (run file, scenario objects — only `label`, `T_reh_GeV`, `description` — and
theta sections `theta`, `dark_matter`, `optimisation`); labels must be unique also ignoring case; booleans are not numbers.

### 1b. Run file of methodology v2 (`cyaxiverse_relic_run/2`; examples in `examples/wR_general/`)

Same keys as 1, with `w_R` allowed in every scenario object. `engine/run_relic.sh` reads the run file's `schema` and selects the methodology:
`cyaxiverse_relic_run/1` -> `configs/methodology_v1.json` (w_R = 0 fixed; a `w_R` key is refused), `cyaxiverse_relic_run/2` -> `configs/methodology_v2.json`.
The two must match (a v1 run file with the v2 methodology file, or the reverse, is refused by the composer). `engine/run_theta.sh` uses the same run file
(the theta analysis is the same in both versions and takes `w_R` from the relic run).

| key | type | meaning |
|---|---|---|
| `scenarios[].w_R` | number, **-1 < w_R <= 1**, optional (default **0.0**) | constant equation of state p = w_R ρ of the pre-reheating epoch of this scenario (boolean, string, null, NaN refused) |

- A (`T_reh_GeV`, `w_R`) pair may appear once per run. Labels are file names: choose self-explanatory ones (for example `T1e4-w0p17`); a label never carries the physics, the
  `w_R` column does.
- Accepted with a recorded warning: w_R < -1/3 (accelerating epoch), w_R < -0.9 (steep; the run stops for a mode whose abundance leaves the float64 range). Noted: w_R = 1/3.
- The effective `w_R` of each scenario is written to `relic/run_config.json` (`methodology_version: 2`, `reheating.scenarios[].w_R`, with `PRE_mass_exponent_2w_over_1pw` and
  `Omega_pre_h2_per_fGeV2_theta2_at_m_eq_3H_R`), to `relic/scenario_checks.json`, `run_metadata.json`, `global_summary.json`, and as column `w_R` of every output table.

## 2. Existing spectra list (`cyaxiverse_relic_spectra/1`; example `configs/spectra_list.example.json`)

| key | meaning |
|---|---|
| `vacuum_policy.on_unusable` | `refuse` (default): any unusable geometry stops the build; `exclude`: it is listed with status `EXCLUDED_<state>` and kept out of every table |
| `spectra[].path` | one `cyax.h5` per geometry |
| `spectra[].geometry_id` | optional; default: the last three directories of `path`. Its `np_` index must equal the file's `spectrum/physical/metadata/polytope` (or `provenance.local_index`) |
| `spectra[].vacuum_validation.state` | `VERIFY` or `DOCUMENTED_EXTERNAL` (anything else is `UNVERIFIED`) |
| `…method`, `…reference`, `…evidence[] {path, sha256}` | required for `DOCUMENTED_EXTERNAL`; every evidence file is re-hashed (hex, case-insensitive) when the input is built and again by A1 |
| `spectra[].provenance` | optional pass-through: `tranche`, `local_index`, `normal_form_sha256`, `qcd_edge_flag`, `qcd_edge_beyond_gate_tolerance` (default `NA`; `local_index` defaults to the position) |

`VERIFY` needs the potential in the file (`potential/{Q,L}`, `Kinv`, as written by the CYAxiverse materializer and kept
by `scripts/batch_physical_spectrum.jl`) and applies only to θ = 0 spectra.

## 3. Spectrum file (`cyax.h5`, CYAxiverse vmm physical spectrum; checked by A1 for every file)

| dataset | unit / content | requirement |
|---|---|---|
| `spectrum/physical/m` | log10(m / eV), numeric dataset (float or integer; strings refused) | finite; every value ≥ log10 H0 = −32.826522356547; may be empty (validated spectrum without mode above H0) |
| `spectrum/physical/fpert_log10` | log10(f_pert / GeV) | finite; equals m − 9 − ½ `lambda_self_log10` to 1e-9 |
| `spectrum/physical/lambda_self_log10`, `lambda_self_sign` | log10 \|λ_iiii\| (GeV⁴ normalisation of the writer), sign ∈ {−1, +1} | same length as `m` |
| `spectrum/physical/mode_indices` | 0-based indices in the ascending eigenmode list | unique, exactly the top block h11−n … h11−1 |
| `spectrum/physical/metadata/*` | `h11`, `polytope`, `frst`, `prec`, `quartics`, `threshold_log10`, `status`, `terminal_status`, `provisional`, `schema_version`, `evaluation_point` | prec = 500, quartics = 1, threshold = log10 H0 (±1e-12), status and terminal_status `completed`, provisional = 0, schema `cyaxiverse-physical-spectrum-3`, h11 and frst equal to the geometry_id |
| `cytools/geometric/CY_volume`, `cytools/geometric/h11` | string-frame CY volume; Hodge number | volume finite and positive |
| `vacuum_correction/` group | present only for spectra about a validated minimum | iff `evaluation_point = validated_leading_branch_minimum` (status VACUUM_CORRECTED) |

### 3a. Supplying your own spectra (manual input)

There is **no direct mass / decay-constant table input**: the engine reads one vmm-layout `cyax.h5` per geometry, so provenance and
the contract checks (A1) stay uniform. A researcher with only masses and decay constants can write such a file with h5py, with:
- 1-D numeric datasets (a 2-D or scalar dataset is refused with `dataset_shape`): `spectrum/physical/m` = log10(m/eV), `fpert_log10` =
  log10(f/GeV), `lambda_self_log10` = 2 (m − 9 − fpert_log10), `lambda_self_sign` = ±1, `mode_indices` = h11−n … h11−1, all of the same length n;
- the metadata as **datasets** (not attributes) in `spectrum/physical/metadata/`: `h11`, `polytope` (equal to the `np_` index), `frst`, `prec = 500`,
  `quartics = 1`, `threshold_log10 = -32.826522356547`, `status = terminal_status = completed`, `provisional = 0`, `schema_version =
  cyaxiverse-physical-spectrum-3`; `evaluation_point` must be **absent** (spectrum about θ = 0, as written by vmm) or exactly
  `validated_leading_branch_minimum` (repaired spectrum, which also needs the `vacuum_correction/` group); any other value is refused;
- `cytools/geometric/CY_volume` (> 0, finite) and `cytools/geometric/h11`;
- a vacuum statement in the spectra list: `DOCUMENTED_EXTERNAL` with method, reference and checksummed evidence (a spectrum that is not
  a vmm θ = 0 spectrum cannot use `VERIFY`, which needs the stored potential).
f is not independent of m: `fpert_log10 = m − 9 − ½ lambda_self_log10` must hold. θ_i is not an input (see the angle analysis), and
the cosmology is fixed by methodology v1. Every refusal names the file, the check and, where relevant, the dataset.

## 4. Polytope manifest (producers; `references/shoroq/data/polytope_manifest_h11_004_050.json` is the Shoroq one)

Seeds (producers/geometry/resolve_seed.py): the G2 base seed is, in order of precedence, the environment variable `G2_SEED`, the manifest
`base_seed` (or the base implied by the per-entry `seed` fields), then 20260930 (the Shoroq recipe). The polytope at 1-based position p
of its h11 block gets seed + p. An entry `seed` that disagrees with `base_seed + position`, a `unified_position` that is not the list
position, or a non-integer or out-of-range seed is refused. The effective seed and its source are recorded in
`provenance/seed_resolution.json` and checked against the G2 ledger. `H11_MIN`, `H11_MAX`, `N_PER_H11` default to the manifest.

Top level: `polytopes` (list), `base_seed`, `seed_rule`, `h11_range`, `per_h11`, `source`, `revision`. Each entry:
`h11`, `vertices` (4d integer vertices), `unified_position` (1-based position within its h11), `seed`
(= base_seed + position); optional `normal_form_sha256` and `historical_tranche` (default `NA`). The G2 generator reads the polytopes of
each h11 in list order; `H11_MIN`, `H11_MAX`, `N_PER_H11` select the range and the number per h11.

Multiple triangulations: G2 accepts more than one FRST of a polytope when `N_PER_H11` exceeds the number of polytopes of an h11 block
(`cy_0000001`, `cy_0000002`, …; each geometry_id is unique). `GEOMETRY_SELECTION=all` (default) sends every accepted geometry to the
vacuum scan and the relic input; `first_frst` keeps only `cy_0000001` of each polytope (the single-triangulation recipe of the Shoroq
reference, which has one geometry per polytope). `vacuum_repair/05_selection_report.json` lists accepted, selected, not selected and failed
polytopes with reasons; the manifest key `geometry_selection` appears when a policy left accepted geometries out. A θ = 0 tachyonic
geometry with FRST > 1 is excluded as `EXCLUDED_VACUUM_UNRESOLVED` (vacuum repair covers the first FRST only).

## 5. Relic-input directory (written by `engine/input` or `producers/vacuum`; read by the engine)

`relic_abundance_input_manifest.json` (`name = relic_abundance_input_manifest`): `summary` (`final_usable_geometries`,
`final_usable_modes`, …), `per_h11` (`modes_final` per h11), `spectrum_source_base` (relative paths resolve against it),
optional `vacuum_policy`, and `geometries[]` with `geometry_id, h11, position, frst, tranche, local_index, status,
spectrum_source, spectrum_source_sha256, n_modes_above_H0, validation_status, exclusion_reason, qcd_edge_flag,
qcd_edge_beyond_gate_tolerance` and (direct-spectra input) `vacuum_validation {state, method, reference, evidence[]}`.
Statuses: `ORIGINAL_SPECTRUM`, `VACUUM_CORRECTED`, `EXCLUDED_*`. Producer manifests carry no `vacuum_validation`: their
state follows from the status (`METHODOLOGY.md`). Also: `relic_abundance_input_geometries.csv`, `SHA256SUMS.txt` (of the
two files), `spectra/<geometry_id>/cyax.h5` (symlinks).

## 6. Outputs

Column names, types, units and NA rules of every table: `configs/relic_output_schema.json` (methodology v1) or `configs/relic_output_schema_v2.json` (v2: the same tables plus the
column `w_R` appended last, and the branch label `fuzzy_wR_general_eq2.10_2.15` for PRE modes with w_R != 0; `fuzzy_wR0_eq2.10_2.15` for w_R = 0). v2 theta outputs carry `w_R` in
every per-scenario table and summary (v1 theta outputs are unchanged). (null token `NA`, booleans
`true`/`false`, floats written with `%.17g`, so round-trip exact). Read the tables with the schema types: an integral
float such as 1.1873334118689998e16 is written without a decimal point, and type inference on a small table can
otherwise yield an integer column. Main units: masses eV, decay constants GeV, temperatures GeV, H in GeV
(`three_H_R_eV` in eV), abundances Ω h² at θ_ref = π/2.

| file | rows |
|---|---|
| `master_modes_<label>.csv` | one per mode: category, abundance state and branch, T_osc, 1+z, Ω h² (NA when not evaluated), flags |
| `geometry_summary_<label>.csv` | one per usable geometry, also without mode above H0: counts per class, Ω_geometry h² (sum of included modes; 0 for an evaluated empty sum; NA when nothing is included and a mode is above 3H_I,max; a partial sum for PARTIAL_VALIDITY), PRE/POST parts (NA = no included mode of that class), dominant mode |
| `geometry_abundance_validity_<label>.csv` | one per geometry: class `EVALUATED_POSITIVE` / `EVALUATED_ZERO` / `PARTIAL_VALIDITY` / `INFLATION_INVALID_ALL`, reason, complete abundance (evaluated classes), partial sum (PARTIAL_VALIDITY), theta population `PRIMARY` / `SENSITIVITY_ONLY` / `EXCLUDED_NOT_EVALUATED` |
| `population_validity_summary.json` | per scenario: geometries per class and theta denominators |
| `h11_summary_<label>.csv`, `global_summary.json` | bookkeeping (each geometry is a separate universe; cross-geometry sums are not physical) |
| `final_analysis/theta/theta_geometry_final.csv` | per geometry and scenario: θ at R = 0.1, 0.8, 0.9, 0.95, 1 (`Inf` for Ω = 0); class at the representative angle (empty for NO_MATCH) |
| `final_analysis/theta/theta_summary_final.{csv,json}` | primary population: exact N_TIGHT maximum and maximising intervals, status (`INTERIOR`, `BOUNDARY_LIMITED`, `NO_MATCH`), robustness, bootstrap; undefined quantities are `null` (JSON) or empty (CSV); JSON key `population` |
| `final_analysis/theta_sensitivity_partial_validity/` | the same file set for the sensitivity population (primary + PARTIAL_VALIDITY partial sums), `population = SENSITIVITY_INCLUDING_PARTIAL_VALIDITY` |
| `final_analysis/theta/theta_population.csv`, `theta_population_summary.json` | every geometry with class and population membership; denominators, N_max, intervals and status of both populations; geometries not evaluated |
| `final_analysis/verification/crosscheck_python{,_sensitivity_partial_validity}.json` | independent Python cross-check of each population |

## 7. Expectations of a reference case (`cyaxiverse_relic_expectations/1`)

`relic` and `theta` trees filling the `@expectation` slots of the methodology files (expected counts, frozen checksums).
They can only fill slots; an attempt to change a methodology value is refused. Example:
`references/shoroq/configs/expectations.json`.
