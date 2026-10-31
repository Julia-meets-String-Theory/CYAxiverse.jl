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
by `batch_physical_spectrum.jl`) and applies only to θ = 0 spectra.

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

## 4. Polytope manifest (producers; `references/shoroq/data/polytope_manifest_h11_004_050.json` is the Shoroq one)

Top level: `polytopes` (list), `base_seed`, `seed_rule`, `h11_range`, `per_h11`, `source`, `revision`. Each entry:
`h11`, `vertices` (4d integer vertices), `unified_position` (1-based position within its h11), `seed`
(= base_seed + position); optional `normal_form_sha256` and `historical_tranche` (default `NA`). The G2 generator reads the polytopes of
each h11 in list order; `H11_MIN`, `H11_MAX`, `N_PER_H11` select the range and the number per h11.

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

Column names, types, units and NA rules of every table: `configs/relic_output_schema.json` (null token `NA`, booleans
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
