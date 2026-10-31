# Provenance of every file

Every script in this repository was taken from the research workspace that produced the accepted dataset
(2026-09-30 geometry and spectra; 2026-10-01 relic abundance and θ analysis). The table lists, for each file,
its origin (a path relative to that workspace), the first 12 hex digits of the original sha256, and what changed.

**Change classes**
- **identical**: byte-for-byte copy.
- **paths**: only machine-specific locations were replaced by command-line arguments or `$WORK`. Usage comments were updated to match. No computation, criterion or tolerance changed.
- **trimmed**: as *paths*, and a check that depended on material outside this repository was removed. The removed part is named below.
- **new**: written for this repository. It contains no physics.

## Stage 1: sample (`references/shoroq/sample/`)

| file | origin | sha256 | change |
|---|---|---|---|
| `hf_range.py` | `thesis_g2_adaptive/tranche2/tools/hf_range.py` | b6ea89015baf | identical |
| `ks_ordered_scan.py` | `thesis_g2_adaptive/tranche2/tools/ks_ordered_scan_v2.py` | 9abbd6ee83d6 | docstring only (renamed from `_v2`) |
| `select_polytopes.py` | `thesis_g2_adaptive/tranche2/tools/select_polytopes.py` | 129d78d658d3 | identical; the reference manifest is now the committed sample (it was the tranche-1 list) |
| `build_g2_manifest.py` | — | — | new; replaces `build_tranche2_manifest.py` (one manifest instead of two tranches) |

## Stage 2: geometry and spectra (`producers/geometry/`)

| file | origin | sha256 | change |
|---|---|---|---|
| `materialize_cyax_for_julia.py` | `thesis_g2_adaptive/tools/` | 24606c2c902e | identical |
| `audit_geometry.py` | `thesis_g2_adaptive/tools/` | 79d2b4db2466 | identical |
| `julia_crosscheck.jl` | `thesis_g2_adaptive/tools/` | 0448081c03f5 | identical |
| `compare_julia.py` | `thesis_g2_adaptive/tools/` | 0b9747f15f3e | identical |
| `validate_stage.py` | `thesis_g2_adaptive/tools/` | bda0d1973230 | identical |
| `qkq_scale_aware.py` | `thesis_g2_adaptive/tranche2/tools/` | 0b96e5d83d1d | identical |
| `status_ledgers.py` | `thesis_g2_adaptive/tools/` | 2806d1d48586 | identical |
| `capture_provenance.py` | `thesis_g2_adaptive/tools/` | c9c86a230a21 | identical |
| `integrity_sample.py` | `thesis_g2_adaptive/tranche2/tools/integrity_sample_t2.py` | 1a25c1371dc9 | identical |
| `integrity_spectra.jl` | `thesis_g2_adaptive/tools/` | a2da9ecbdd2a | identical |
| `flag_qcd_edge.py` | — | — | new; writes down the QCD-edge flag rule of 2026-09-30, which was applied inline at the time. It reproduces the frozen 539 flags (6 beyond tolerance) exactly. |
| `final_gate.py` | — | — | new; encodes the two documented gate resolutions of 2026-09-30 (scale-aware qKq criterion; QCD-edge geometries kept and flagged) |

The generator, the Python→Julia bridge and the spectrum CLI are **not copied**. They are used from CYAxiverse.jl `vmm` a5eacd8:
- `scripts/generate_geometric_data_multitriangulation.py`
- `scripts/build_orientifold_axion_database.py`
- `scripts/batch_physical_spectrum.jl`
- `scripts/batch_vacua_pipeline.jl`

## Stage 3: vacuum repair (`producers/vacuum/`)

| file | origin | sha256 | change |
|---|---|---|---|
| `common.jl` | `tachyonic_vacua_audit/scripts/` | 3706db5ed05b | header comment only |
| `00_theta0_inertia_scan.jl` | `spectrum_count_audit/scripts/mode_accounting.jl` | d8964ce50de3 | usage comment only |
| `01_identify_theta0.jl` | `tachyonic_vacua_audit/scripts/` | d82bd9dbe8b9 | comments only |
| `02b_read_vacua_groups.jl` | `tachyonic_vacua_audit/scripts/` | 9376ecf22662 | identical |
| `03_validate_minima.jl` | `tachyonic_vacua_audit/scripts/` | e0dddf251c2a | usage comment only |
| `04_vacuum_spectra.jl` | `tachyonic_vacua_audit/scripts/` | b46b151e3c8c | usage comment; the provenance string written into `metadata/vacuum_correction_script` names the new path |
| `05_build_relic_input.py` | `tachyonic_vacua_audit/scripts/` | 6d0258f6b03a | trimmed (see below) |
| `06_verify_corrected.jl` | `tachyonic_vacua_audit/scripts/` | 0b07b7bfd917 | paths: θ=0 reference spectra as an argument |
| `07_verify_manifest.py` | `tachyonic_vacua_audit/scripts/` | 0d0cf2c5ca55 | paths: relative sources; source-location rule rewritten for the WORK layout |

**`05_build_relic_input.py` in detail.**
- The classification checks and the manifest records are unchanged.
- Accepted geometries now come from the G2 status ledger joined with the committed sample. This replaces the unified-view manifest of the two tranches.
- `local_index` equals the position, because a single run has no tranche-local numbering.
- `spectrum_source` is written relative to `WORK` (`spectrum_source_base = ".."`), and the spectra view uses relative symlinks.
- The record of failures of the repository vacua CLI with its *default* `--method legacy` was dropped. It was informational only: 46/46 failures, and the method is not used.
- An `SHA256SUMS.txt` for the input directory is now written.
- The 46-geometry count became a command-line gate (`--expected-tachyonic`).

## Stage 4: relic abundance (`engine/relic/`)

| file | origin | sha256 | change |
|---|---|---|---|
| `compute_relic_abundance.jl` | `relic_abundance_production/run_record/compute_relic_abundance_production.jl` | 4483390956fa | **header comment only**; everything from `using` onward is byte-identical (`tests/test_static.py`) |
| `01_build_mode_table.py` | `relic_abundance_production_prep/scripts/01_a1_input_adapter.py` | 793020229b8a | trimmed (see below) |
| `02_write_run_config.py` | — | — | new; replaces the location/pin part of `00_build_config.py`. All constants are in the committed template. |
| `03_validate_outputs.py` | `relic_abundance_production_prep/scripts/06_validate_outputs.py` | f3c7fadc7580 | paths |
| `04_verify_relic_outputs.py` | `relic_abundance_production_prep/scripts/11_production_verify.py` | 6d7f05432c4a | trimmed: part (c), the comparison with the pre-production benchmark directory, is removed; parts (a) and (b) are unchanged |
| `common.py` | `relic_abundance_production_prep/scripts/common.py` | e76729307544 | paths |
| `reference_mp.py` | `relic_abundance_production_prep/scripts/` | 8463cac407ba | path of the g* table |
| `independent_relic.py` | `relic_abundance_method_audit/scripts/` | 8cf1a57382ed | path of the g* table |

**`01_build_mode_table.py` in detail.**
- Inputs and outputs are now arguments; the manifest pin is optional (`--expect-manifest-sha256`).
- Relative `spectrum_source` paths are resolved.
- The cross-check against the descriptive spectrum-characterisation table (`relic_spectrum_analysis/`) is removed. That table is not part of this repository, and A1 reads the HDF5 values directly.
- The label-anomaly check now accepts zero anomalies. It required exactly 19, which is a property of the two-tranche layout.

## Stage 5: θ analysis (`engine/theta/`)

| file | origin | sha256 | change |
|---|---|---|---|
| `01_theta_final.jl` | `relic_abundance_final_analysis/scripts/` | 3fffc5bde05c | identical |
| `theta_core.py` | `relic_abundance_theta_analysis/scripts/` | 698fcacc2ba1 | identical |
| `common.jl` | `relic_abundance_final_analysis/scripts/` | 27d5fb92e1c9 | paths: inputs from `WORK/relic/run_config.json`. `spectrum_master()` (presentation stage) is removed. The exact-interval machinery is unchanged. |
| `00_verify_inputs.jl` | `relic_abundance_final_analysis/scripts/00_verify_frozen_inputs.jl` | c9b30450e1fe | trimmed (see below) |
| `02_crosscheck_theta.py` | `relic_abundance_final_analysis/scripts/05_crosscheck.py` | 5dda517ffd31 | trimmed: the representative-geometry ranking is removed; the census is compared with the approved counts instead of a figure-data table |

**`00_verify_inputs.jl` in detail.**
- The checksum checks of workspace products outside this repository are removed:
  - the production `SHA256SUMS`;
  - the unified-view and spectrum-characterisation `SHA256SUMS`.
- The source-location checks are rewritten for the `WORK` layout.
- All population, inflation-cut, census and C = 1 physics checks are unchanged.

## Configuration, data, environment

| file | origin | change |
|---|---|---|
| `references/shoroq/configs/relic_production_config.frozen.json` | `relic_abundance_production_prep/config/production_run_config.json` (2ef6566ae399) | Locations and pins replaced by run-time placeholders, `code` section made repository-relative. **No numeric or boolean value changed** (checked by `02_write_run_config.py` at every run). |
| `configs/relic_output_schema.json` | `relic_abundance_production_prep/config/output_schema.json` | identical |
| `references/shoroq/configs/theta_analysis_config.frozen.json` | `relic_abundance_final_analysis/config/final_analysis_config.json` (bb941a4013e1) | `frozen_inputs` (absolute paths) replaced by `inputs` (resolution rule + frozen checksums for identification); every rule and constant unchanged |
| `references/shoroq/data/polytope_manifest_h11_004_050.json` | `thesis_g2_adaptive/tranche2/manifests/unified_polytope_list.json` | reformatted as one G2 manifest; vertices, order, normal forms and seeds unchanged |
| `data/standardmodel2018.dat` | legacy `correct_glimmers_pipeline/standardmodel2018.dat` | identical (sha256 a4bf5f18…) |
| `environment/julia_analysis/` | `relic_abundance_final_analysis/julia_env/` | CairoMakie and Serialization removed with `Pkg.rm` (offline); none of the 87 remaining packages changed version |
| `environment/julia_vmm/Manifest.toml` | `CYAxiverse_vmm_production/Manifest.toml` | identical (sha256 bc566a33…) |

## Not included, and why

- **Presentation.** Figure scripts, figure data, representative-geometry selection, spectrum characterisation.
- **Historical audits.** Geometry/Kähler prescription audits, legacy (1600-geometry) datasets and their repairs, the relic method audit, dry runs and benchmarks, θ-domain audits, provenance audits.
- **Unfinished work.** Decay/stability and self-interaction analyses.
- **Diagnostics not used as acceptance criteria.**
  - `precision_check.py`, `kaehler_path_stats.py`, `summarize_audit.py`, `spectra_table.py`, `write_run_manifest.py`.
  - The vacuum-audit report tables (`08_min_resolved_eigen.jl`, `09_report_tables.py`).
  - The `--method legacy` run of the vacua CLI.

## Removed checks after review (2026-10-28)

Every check removed during cleanup was reviewed. Checks that the frozen data can support are back as regression tests, without changing the pipeline scripts.

| removed from | check | status |
|---|---|---|
| `11_production_verify.py` part (c) | Production rows equal the pre-production benchmark; benchmark A-vs-B agreement at full precision | **restored and strengthened** in `references/shoroq/tests/compare_relic_benchmark.py`: the relic script is rerun on the frozen benchmark table (real and synthetic boundary modes) and must reproduce the frozen benchmark output byte for byte; then the original (c) comparisons are applied |
| `00_verify_frozen_inputs.jl` | Checksums of the frozen production, relic-input, unified-view and repair products | **restored** in `references/shoroq/tests/verify_frozen_reference.py`. Every regression run first verifies its reference against the reference's own SHA256SUMS, plus pinned checksums for files without a list. |
| `05_build_relic_input.py` and the original manifest | The relic-input manifest content | **added** in `references/shoroq/tests/compare_vacuum_repair.py`: every scientific field of every rebuilt record, and the vacuum classification, must equal the frozen ones |
| 2026-09-30 inline QCD-edge flagging | The flag rule on the full population | **added** in `references/shoroq/tests/check_qcd_flags_frozen.py`: `flag_qcd_edge.py` on the 4491 frozen audits must reproduce the frozen `qcd_edge_flags.json` |
| `05_crosscheck.py` | Python θ cross-check | **compared with the frozen result** in `regression_relic_theta.sh` (θ intervals, maxima and the PRE constant must be identical) |
| `01_a1_input_adapter.py` | Comparison with the spectrum-characterisation table | not restored. A1 reads the HDF5 values directly, and the byte-identity of `canonical_modes.csv` with the frozen table is a stronger check. |
| `05_crosscheck.py` | Representative-geometry ranking | not restored (presentation) |
| `10_production_sanity.py` | Census and invariants | already covered by `00_verify_inputs.jl` (census, inflation cut, inclusion rule) and `03_validate_outputs.py` |
| `08_min_resolved_eigen.jl` | Smallest resolved Hessian eigenvalue at θ_min | covered by `03_validate_minima.jl` (no resolved negative eigenvalue at about 500 and about 1000 digits) and `06_verify_corrected.jl` |

## Execution hardening (2026-10-28)

**`engine/common.sh`** now:
- checks that the interpreters exist and that `CYAX_REPO` is the pinned commit `a5eacd8` (override with `ALLOW_OTHER_CYAX_COMMIT=1`);
- runs Julia with `--startup-file=no`;
- runs stages 4–5 with `--project=environment/julia_analysis`. Stage 4 previously ran without a project. SHA, the only standard library the relic script loads that was missing, was added offline with no version change.

**Stage drivers** now:
- stop on missing inputs, on a failed previous gate, or on an existing output directory.

## Full-reproduction validation (2026-10-29)

- `references/shoroq/tests/regression_full.sh`: stages 2–5 on all 4700 polytopes with bidirectional comparisons and checks of the frozen data and the workspace before and after the run. `references/shoroq/tests/check_full_population.py`: population completeness and totals. `references/shoroq/tests/compare_relic_theta_full.py`: stages 4–5 on a regenerated relic input.
- **Integrity acceptance** (`references/shoroq/tests/check_integrity_sample.py`): the production criteria, meaning the exact-rational chain plus an identical spectrum recompute. L dex is a diagnostic only, because production accepted up to 6.7e-4 dex in the leading terms.
- **Label rule:** the frozen tranche-2 spectra carry tranche-local `np_` indices, while the unified run uses the unified position. A difference is accepted only through `unified_polytopes.json` (`local_index`), and it covers both the `metadata/polytope` field and the path components.
- **`qcd_volume_exact_rational`:** this field is not recomputed by `flag_qcd_edge.py`. The test requires it to be null and lists the 6 frozen values as not reproduced.
- **`GIT_OPTIONAL_LOCKS=0`** in `common.sh`. Without it, `git status` creates a transient `index.lock` that changes the checkout's `.git` directory mtime.
- **Empty-tachyonic guard** in `run_03_vacuum_repair.sh` (subsets only). The relic-input summary now records the actual number of tachyonic geometries attempted.
- Result: full numerical reproduction, with no unresolved discrepancy. See `Shoroq_clean_validation/VALIDATION_SUMMARY.md`.

## Generalisation (2026-10-29)

Copy of `Shoroq_clean_pipeline` (left unchanged; its 72 files are checksum-verified) reorganised into a general engine,
optional producers and the Shoroq reference case. File-by-file mapping and full diff: `CODE_CHANGES.{tsv,diff}` in the
validation evidence directory `generalization_20261029/`.

**Unchanged science.** The physics block of `engine/relic/compute_relic_abundance.jl` (checksum `a0c613f1…`), reference
implementations B and C, the theta interval algorithm and all upstream CYAxiverse code are unchanged.

**Code changes.**
- Relic driver, one line (G1): `is_population` only when `expected_modes` is configured. Reverting G1 gives the frozen
  body hash `9f73b9c6…` (`tests/test_static.py`).
- `engine/relic/00_compose_run_config.py` (new): methodology files with `@run`/`@expectation` slots + run file +
  optional expectations → configuration. T_reh hard constraints and BBN warnings; scenario quantities from reference B;
  whitelisted theta overrides. The composed Shoroq configurations are byte-identical to the frozen ones.
- `engine/relic/01_build_mode_table.py` (A1): population checks only if expected; general checks (≥ 1 mode, unique keys,
  exclusions only by declared policy); vacuum-validation state per geometry (`a1/vacuum_validation.csv`); FRST from the
  record instead of 1.
- `engine/relic/04_verify_relic_outputs.py`: guards for small samples (sampler size, empty strata, empty maxima).
- Theta stage: configuration and scenario labels from the composed files; schema-typed CSV reading (integral floats
  written by `%.17g` otherwise become Int64 on small tables); empty h11 bins skipped; configurable primary smoothing and
  robustness criteria; clear stop when a geometry has no included mode. The Shoroq theta outputs are byte-identical.
- `engine/input/build_input_from_spectra.py`, `engine/run_prepare_from_spectra.sh` (new): existing spectra → relic input
  with explicit vacuum-validation states (`METHODOLOGY.md`).
- `producers/vacuum/00_theta0_inertia_scan.jl`: optional FRST column in the geometry list.
- `producers/vacuum/05_build_relic_input.py`: `--expected-tachyonic` optional; `historical_tranche` and
  `normal_form_sha256` optional in the polytope manifest (default `NA`).
- Producer drivers require `POLYTOPE_MANIFEST` (no Shoroq default); `references/shoroq/run_reference.sh` sets it.
- `producers/vacuum/04_vacuum_spectra.jl`: only the stored metadata string `vacuum_correction_script` now names
  `producers/vacuum/04_vacuum_spectra.jl` (metadata; the comparisons skip `metadata/`, as for the earlier path).

**Tests.** After every phase `references/shoroq/tests/regression_relic_theta.sh` passed (release-blocking); at the end
`regression_subset.sh` (h11 = 11 and 45) and `regression_full.sh` (14/14) passed, and `tests/test_general.sh` 22/22.

## Acceptance fixes (2026-10-30)

These fixes follow the external acceptance tests of 2026-10-30 (Claude and Codex reports). The physics block,
implementations B and C, all tolerances, the Kähler normalisation and the pair-term convention are unchanged. Every Shoroq
relic table, theta output, verification result and composed configuration is byte-identical (`regression_relic_theta.sh`
45/45). Full change list and reasons: `FIX_IMPLEMENTATION_REPORT.md` in the validation folder `fix_implementation_20261030/`.

**Code changes.**
- **Zero-mode / zero-Omega geometries (F1; approved decision).** A validated geometry is never dropped:
  - A1 keeps geometries without mode above H0.
  - The relic driver takes the geometry population from the A1 geometry table ([G2], [G4]).
  - Ω_geometry = 0 when no mode is included ([G4]).
  - New table `geometry_cdm_eligibility_<label>.csv` ([G6]).
  - Census key semantics updated ([G5]).
  - Validators 03 and 04 updated; theta input checks updated.
  - Theta: Ω = 0 gives thresholds +∞, and the geometry stays in every count.
  - Every log10 Ω and 1/Ω site was audited.
- **Schema-typed table reading in Python (F2):** `engine/relic/common.py:read_table`.
- **NO_MATCH (F6):** theta analysis and Python cross-check.
- **Smaller fixes:**
  - canonical checksum verified in every run ([G2], F4);
  - T_osc/T_eq tolerance ([G3], F5) and explicit threshold ties in verifiers C and B;
  - strict run files (F7, F14);
  - producer toolchain required only where used (F8);
  - engine requirements file and docs (F9, F11);
  - moved or copied WORK refused (F10);
  - sub-H0 vacuum diagnostics for VERIFY entries (F3);
  - named failures instead of tracebacks, and atomic JSON writes (F12);
  - dtype and sign contract (F13); case-insensitive evidence sha (F15); small-sample bootstrap note (F16);
  - f > M_pl diagnostic count (F18); `.DS_Store` removed.
- The driver changes are listed as text blocks in `tests/driver_changes.json`. `tests/test_static.py` reverts them and
  checks the frozen body hash.

**Reference benchmark.** The frozen pre-production benchmark contains 5 synthetic pseudo-geometries per scenario whose only
mode is FROZEN or LATE. Their Ω_geometry changes from NA to 0 by the approved decision.
`references/shoroq/tests/compare_relic_benchmark.py` applies exactly this cell mapping to the frozen file and requires every
other byte to be identical.

**New test.** `tests/test_external_acceptance.py` (14 cases, no external data) uses two genuine non-Shoroq geometries in
`tests/external_acceptance/data/` (see `provenance.json`) and SYNTHETIC variants built from them.

## Abundance validity classes and theta populations (2026-10-31)

**Decision (user, 2026-10-31).** Geometries with an inflation-excluded mode are no longer treated as having a complete
abundance:
- `PARTIAL_VALIDITY` geometries are analysed only in a labelled sensitivity population.
- `INFLATION_INVALID_ALL` geometries have an abundance that is not evaluated (NA) and are in neither population.
- `EVALUATED_ZERO` geometries (no mode above H0, or all FROZEN/LATE) stay in the primary population with Ω = 0.

**Conflict with the frozen methodology.** The frozen 2026-10-01 methodology kept the 329 partially valid Shoroq geometries
(per-mode cut) in its single theta population. They are now excluded from the primary population. The conflict check, the
full field-by-field before/after comparison and the scientific consequences are in
`Shoroq_clean_validation/population_decision_20261031/`.

**Changes.**
- Driver `[G4]`: NA for an empty CDM set with a not-evaluated mode. Driver `[G6]`: table
  `geometry_abundance_validity_<label>.csv` and `population_validity_summary.json`.
- Schema, validators 03/04, theta input checks.
- `01_theta_final.jl` runs one analysis per population, writing `theta/` and `theta_sensitivity_partial_validity/`, plus
  `theta_population.csv` and `theta_population_summary.json`.
- `02_crosscheck_theta.py` takes `THETA_POPULATION=PRIMARY|SENSITIVITY`.
- Regression: the sensitivity population must reproduce every frozen theta output, and the primary population is pinned.
- The benchmark mapping is now restricted to geometries without an inflation-excluded mode.
- New tests: `populations` and `all_invalid` cases in `tests/test_external_acceptance.py`.

## Integration into CYAxiverse.jl (2026-10-31)

The package was copied unchanged to `research/axion_relic_abundance/` on branch `feature/axion-relic-abundance` (base `vmm`
`f12fbb8`), except:
- **Fixtures** (`tests/external_acceptance/data/*/cyax.h5`): the absolute local path prefix in three provenance-only
  attributes (`construction_metadata_json`, `julia_materialization/driver_path`, `julia_materialization/source_file`) was
  replaced by `<acceptance-root>/`, and the files were repacked with H5Ocopy so that no old bytes remain. Every dataset,
  every other attribute and every storage filter was verified identical. `SHA256SUMS.txt` was updated. Nothing reads these
  attributes.
- **`engine/common.sh`**: an empty `CYAX_REPO` defaults to the enclosing CYAxiverse.jl checkout. The revision pin is
  unchanged. `configs/paths.env.example` documents this.
- **New documentation**: `SCIENTIFIC_SCOPE_AND_CONVENTIONS.md` and `INTEGRATION_DESIGN.md` (replacing the earlier
  `INTEGRATION.md` proposal). The README has a scope summary and a quick start. The "Known limitations" entry on
  conventions in `METHODOLOGY.md` was reworded.
- **New example**: `examples/minimal/`.

No computation, criterion, tolerance or physics block changed.
