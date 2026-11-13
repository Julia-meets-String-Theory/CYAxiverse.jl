# Historical inflation CSV entries (CYAX-0232 T041, Clarification A3)

Source: merged code, i.e. commits reachable from `10aa3dc`. Every header below is
resolved from the field tuples in the historical blobs (read with `git cat-file -p
<commit>:<path>`) by a script, not transcribed by hand; the script also asserts
that the tuple is identical in every blob of the entry's period. Commit hashes are
abbreviated to 7 characters. Line numbers are those of the named commit's blob.

Rules applied (Clarification A, A3.1–A3.3): one entry per (exact header,
`row_type` values, version-column values). Version columns are
`schema_version`, `shard_schema_version`, `contract_version` and
`diagnostic_schema_version`; the pilot columns `domain_certificate_version`
and `conversion_policy_version` are not listed. CSV cells hold the values as
plain text (for example `contract_version` cell `1`, `row_type` cell `point`).

## Summary

| Entry | Header id (columns) | `row_type` | Version values | Commits | Writers |
|---|---|---|---|---|---|
| INF-PREP-1 | `PREP_A` (35) | — | contract_version=1 | 2664648–201f0ac | physical |
| INF-PREP-2 | `PREP_A` (35) | — | contract_version=2 | fdf75f7 | physical |
| INF-PREP-3 | `PREP_B` (44) | — | contract_version=3, diagnostic_schema_version=1 | 814988e–64ac09b | physical |
| INF-PREP-4 | `PREP_C` (53) | — | contract_version=4, diagnostic_schema_version=1 | dd2c18c–10aa3dc | physical |
| INF-SHARD-1 | `SHARD_1` (51) | — | shard_schema_version=1, contract_version=3, diagnostic_schema_version=1 | 5015b7d–64ac09b | physical |
| INF-SHARD-2 | `SHARD_2` (60) | — | shard_schema_version=2, contract_version=4, diagnostic_schema_version=1 | dd2c18c–10aa3dc | physical |
| INF-REF-3 | `REF` (64) | point, summary | schema_version=3 | c7482f5–10aa3dc | physical |
| INF-FLOW-1 | `FLOW` (36) | flow | schema_version=1 | c7482f5–10aa3dc | physical |
| INF-PILOT-3-REPORT | `PILOT_3` (72) | scale | schema_version=3 | 10aa058–900cd31 | physical |
| INF-PILOT-3-SHARD | `PILOT_3` (72) | branch | schema_version=3 | 10aa058–900cd31 | physical |
| INF-PILOT-4-REPORT | `PILOT_4` (92) | scale | schema_version=4 | 9f31d71 | physical |
| INF-PILOT-4-SHARD | `PILOT_4` (92) | branch | schema_version=4 | 9f31d71 | physical |
| INF-PILOT-6-REPORT | `PILOT_6` (102) | scale | schema_version=6 | b3719e3–10aa3dc | physical |
| INF-PILOT-6-SHARD | `PILOT_6` (102) | branch | schema_version=6 | b3719e3–10aa3dc | physical |

Entries per family (an entry with several version columns is listed under each):

- diagnostic (`INFLATION_DIAGNOSTIC_SCHEMA_VERSION`, only ever `"1"` before the change): INF-PREP-3, INF-PREP-4, INF-SHARD-1, INF-SHARD-2. No merged non-test code wrote a stand-alone diagnostic-row file (see *Not entries*).
- scan contract (`INFLATION_SCAN_CONTRACT_VERSION` `"1"`, `"2"`, `"3"`, `"4"`): INF-PREP-1 to INF-PREP-4, INF-SHARD-1, INF-SHARD-2.
- shard (`INFLATION_SHARD_SCHEMA_VERSION` `"1"`, `"2"`; includes the merged CSV): INF-SHARD-1, INF-SHARD-2.
- refinement (`REFINEMENT_SCHEMA_VERSION` `"3"`): INF-REF-3.
- flow (`FLOW_SCHEMA_VERSION` `"1"`): INF-FLOW-1.
- pilot (`PILOT_SCHEMA_VERSION` `"3"`, `"4"`, `"6"`; report and shard): INF-PILOT-3-REPORT/-SHARD, INF-PILOT-4-REPORT/-SHARD, INF-PILOT-6-REPORT/-SHARD.

Version history checked against every blob (full history, merges included):

| Constant | Values (first commit) |
|---|---|
| `INFLATION_SCAN_CONTRACT_VERSION` (`scripts/inflation_scan_common.jl`) | "1" (`2664648`), "2" (`fdf75f7`), "3" (`814988e`), "4" (`dd2c18c`) |
| `INFLATION_DIAGNOSTIC_SCHEMA_VERSION` (`scripts/inflation_diagnostics_common.jl:3`) | "1" (`814988e`; single blob) |
| `INFLATION_SHARD_SCHEMA_VERSION` (`scripts/inflation_scan_shards_common.jl:11`) | "1" (`5015b7d`), "2" (`dd2c18c`) |
| `REFINEMENT_SCHEMA_VERSION` / `FLOW_SCHEMA_VERSION` (`scripts/inflation_candidate_refinement.jl:17–18`) | "3" / "1" (`c7482f5`; unchanged) |
| `PILOT_SCHEMA_VERSION` (`scripts/inflation_scale_continuation.jl`) | "3" (`10aa058`), "4" (`9f31d71`), "6" (`b3719e3`); no "5" in any merged commit |

Header changes and version changes coincide everywhere except `fdf75f7`
(contract "1" → "2" with the scan-prep header unchanged), which is why
INF-PREP-1 and INF-PREP-2 share a header. No header changed without a version
change.

## Entries

### INF-PREP-1

- **Family:** scan contract (`INFLATION_SCAN_CONTRACT_VERSION`); predates the diagnostic family
- **File kind:** scan-prep summary CSV (`--summary PATH`, default `<data_dir>/logs/inflation_scan_prep.csv`), written by `scripts/inflation_scan_prep.jl`
- **Header id:** `PREP_A` (35 columns; SHA-256 of the header line without newline, first 16 hex: `1c2eaf411ec16883`)
- **`row_type` values:** none (the header has no `row_type` column)
- **Version columns and values:** `contract_version` = `"1"`
- **Commit range:** first `2664648`, last `201f0ac`; superseded by `fdf75f7` (contract `"2"`, same header → INF-PREP-2)
- **Commits (full-history log of the writer files while this layout was in effect):** `2664648`, `201f0ac`
- **Writers:**
  - `2664648:scripts/inflation_scan_prep.jl:186` `_scan_prep_write_header` (writes `SCAN_PREP_HEADER`, :198)
  - `2664648:scripts/inflation_scan_prep.jl:234` `_scan_prep_append`
  - `2664648:scripts/inflation_scan_prep.jl:256` `run_scan_prep` (header call :266; failure row stamps `contract_version` :293; appends :299, :306)
  - `2664648:scripts/inflation_scan_common.jl:102` `run_geometry` (stamps `contract_version` :125; constant `scripts/inflation_scan_common.jl:14` = "1")
  - `201f0ac:scripts/inflation_scan_common.jl:152` `run_geometry` (stamps :175; constant :14 = "1"); `scripts/inflation_scan_prep.jl` blob unchanged from 2664648
- **R-008 classification of every writer:** physical (all). `run_geometry` takes K from `read.oriented_potential` (chain B′).
- **Notes:** Every row carries `contract_version`: success rows from `run_geometry`, failure rows from the failure tuple in `run_scan_prep`. `--append-summary`/`--resume` accept any file whose first line equals the header, so a file started at `2664648`/`201f0ac` and appended at `fdf75f7` holds rows of INF-PREP-1 and INF-PREP-2 under one header.

Exact header:

```text
contract_version,data_dir,max_branches,h11,polytope,frst,status,error,instantons,selected_instantons,qtilde_det,leading_log_gap,log_scale_span,strong_hierarchy,branch_count,leading_minima_count,saddle_count,candidate_slowroll_saddles,least_tachyonic_min_eta,least_tachyonic_epsilon,best_min_eta,best_epsilon,best_abs_min_eta,mass_min,mass_max,negative_mass_count,stage_load_s,stage_select_s,stage_hierarchy_s,stage_factor_s,stage_mass_basis_s,stage_branches_s,stage_classify_s,stage_allocated_bytes,total_seconds
```

### INF-PREP-2

- **Family:** scan contract (`INFLATION_SCAN_CONTRACT_VERSION`)
- **File kind:** scan-prep summary CSV
- **Header id:** `PREP_A` (35 columns; SHA-256 of the header line without newline, first 16 hex: `1c2eaf411ec16883`)
- **`row_type` values:** none (the header has no `row_type` column)
- **Version columns and values:** `contract_version` = `"2"`
- **Commit range:** first and last `fdf75f7` (one commit); superseded by `814988e` (header and contract change → INF-PREP-3)
- **Commits (full-history log of the writer files while this layout was in effect):** `fdf75f7`
- **Writers:**
  - `fdf75f7:scripts/inflation_scan_prep.jl:186` `_scan_prep_write_header`, `:234` `_scan_prep_append`, `:256` `run_scan_prep` (:266, failure row :293, appends :299, :306)
  - `fdf75f7:scripts/inflation_scan_common.jl:193` `run_geometry` (stamps :214; constant :14 = "2")
- **R-008 classification of every writer:** physical (all).
- **Notes:** Header identical to INF-PREP-1; the two entries differ only in `contract_version`. `fdf75f7` changed the contract constant to "2" without a header change.

Exact header:

```text
contract_version,data_dir,max_branches,h11,polytope,frst,status,error,instantons,selected_instantons,qtilde_det,leading_log_gap,log_scale_span,strong_hierarchy,branch_count,leading_minima_count,saddle_count,candidate_slowroll_saddles,least_tachyonic_min_eta,least_tachyonic_epsilon,best_min_eta,best_epsilon,best_abs_min_eta,mass_min,mass_max,negative_mass_count,stage_load_s,stage_select_s,stage_hierarchy_s,stage_factor_s,stage_mass_basis_s,stage_branches_s,stage_classify_s,stage_allocated_bytes,total_seconds
```

### INF-PREP-3

- **Family:** diagnostic (`INFLATION_DIAGNOSTIC_SCHEMA_VERSION`) + scan contract
- **File kind:** scan-prep summary CSV (summary mode; from `5015b7d` the same driver has a shard mode, see INF-SHARD-1)
- **Header id:** `PREP_B` (44 columns; SHA-256 of the header line without newline, first 16 hex: `23affeed32ba8d54`)
- **`row_type` values:** none (the header has no `row_type` column)
- **Version columns and values:** `contract_version` = `"3"`; `diagnostic_schema_version` = `"1"`
- **Commit range:** first `814988e`, last `64ac09b`; superseded by `dd2c18c` (header and contract change → INF-PREP-4)
- **Commits (full-history log of the writer files while this layout was in effect):** `814988e`, `5015b7d`, `b09d44b`, `64ac09b`
- **Writers:**
  - `814988e:scripts/inflation_scan_prep.jl:190` `_scan_prep_write_header`, `:238` `_scan_prep_append`, `:260` `run_scan_prep` (:270; failure row :299 stamps `contract_version` and `diagnostic_schema_version`; appends :307, :314)
  - `5015b7d:scripts/inflation_scan_prep.jl:222` `_scan_prep_write_header`, `:270` `_scan_prep_append`, `:292` `run_scan_prep` (summary header :319; failure row :356; append :376)
  - `814988e:scripts/inflation_scan_common.jl:195` `run_geometry` (stamps both at :218–219; contract constant :18 = "3"); same blob at `5015b7d`
  - `b09d44b:scripts/inflation_scan_common.jl:202` `run_geometry` (:225–226); `64ac09b:scripts/inflation_scan_common.jl:219` `run_geometry` (:243–244)
  - `814988e:scripts/inflation_diagnostics_common.jl:3` `INFLATION_DIAGNOSTIC_SCHEMA_VERSION = "1"` (one blob for the whole history)
- **R-008 classification of every writer:** physical (all).

Exact header:

```text
contract_version,diagnostic_schema_version,measurement_scope,data_dir,max_branches,h11,polytope,frst,status,error,instantons,selected_instantons,qtilde_det,leading_log_gap,log_scale_span,strong_hierarchy,branch_count,leading_minima_count,saddle_count,candidate_slowroll_saddles,least_tachyonic_min_eta,least_tachyonic_epsilon,best_min_eta,best_epsilon,best_abs_min_eta,mass_min,mass_max,negative_mass_count,stage_load_s,stage_select_s,stage_hierarchy_s,stage_factor_s,stage_mass_basis_s,stage_branches_s,stage_classify_s,stage_load_output_bytes,stage_select_output_bytes,stage_hierarchy_output_bytes,stage_factor_output_bytes,stage_mass_basis_output_bytes,stage_classify_output_bytes,stage_allocated_bytes,stage_output_bytes,total_seconds
```

### INF-PREP-4

- **Family:** diagnostic + scan contract
- **File kind:** scan-prep summary CSV
- **Header id:** `PREP_C` (53 columns; SHA-256 of the header line without newline, first 16 hex: `505f1c577382504c`)
- **`row_type` values:** none (the header has no `row_type` column)
- **Version columns and values:** `contract_version` = `"4"`; `diagnostic_schema_version` = `"1"`
- **Commit range:** first `dd2c18c`, last `10aa3dc` (current at the base commit)
- **Commits (full-history log of the writer files while this layout was in effect):** `dd2c18c`, `8dffef1`, `560b901`, `4c0e1ad`, `d802686` (merge), `304a22c`, `a4c352e` (merge), `81e43c4` (merge), `b4d2ffb`, `900cd31` (merge)
- **Writers:**
  - `dd2c18c:scripts/inflation_scan_prep.jl:246` `_scan_prep_write_header`, `:301` `_scan_prep_append`, `:328` `run_scan_prep` (:357; failure row :401; append :424)
  - `560b901:scripts/inflation_scan_prep.jl:251`, `:306`, `:333` (:362; :406; :429)
  - `304a22c:scripts/inflation_scan_prep.jl:250` `_scan_prep_write_header`, `:305` `_scan_prep_append`, `:332` `run_scan_prep` (:361; failure row :405; append :428) — same blob at `10aa3dc`
  - `run_geometry`: `dd2c18c:scripts/inflation_scan_common.jl:239` (:266–267), `8dffef1:scripts/inflation_scan_common.jl:324` (:351–352), `560b901`/`4c0e1ad`/`b4d2ffb:scripts/inflation_scan_common.jl:329` (:356–357; = `10aa3dc`); contract constant "4" (`:18`, from `560b901` `:23`)
- **R-008 classification of every writer:** physical (all). This is plan.md's `inflation_scan_prep.jl` `_scan_prep_write_header`/`_scan_prep_append` (250/305) row.

Exact header:

```text
contract_version,diagnostic_schema_version,measurement_scope,data_dir,max_branches,h11,polytope,frst,status,error,instantons,selected_instantons,qtilde_det,negative_mode_range,structured_charge_validated,structured_fallback_reason,search_classification,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,leading_log_gap,log_scale_span,strong_hierarchy,branch_count,leading_minima_count,saddle_count,candidate_slowroll_saddles,least_tachyonic_min_eta,least_tachyonic_epsilon,best_min_eta,best_epsilon,best_abs_min_eta,mass_min,mass_max,negative_mass_count,stage_load_s,stage_select_s,stage_hierarchy_s,stage_factor_s,stage_mass_basis_s,stage_branches_s,stage_classify_s,stage_load_output_bytes,stage_select_output_bytes,stage_hierarchy_output_bytes,stage_factor_output_bytes,stage_mass_basis_output_bytes,stage_classify_output_bytes,stage_allocated_bytes,stage_output_bytes,total_seconds
```

### INF-SHARD-1

- **Family:** shard (`INFLATION_SHARD_SCHEMA_VERSION`) + diagnostic + scan contract
- **File kind:** scan-prep shard CSV (`<shard_dir>/inflation_scan_prep_shard_NNNN_of_NNNN.csv`) **and** the merged shard CSV of `inflation_merge_shards` (identical header, rows copied verbatim)
- **Header id:** `SHARD_1` (51 columns; SHA-256 of the header line without newline, first 16 hex: `92a31accd1ebcbee`)
- **`row_type` values:** none (the header has no `row_type` column)
- **Version columns and values:** `shard_schema_version` = `"1"`; `contract_version` = `"3"`; `diagnostic_schema_version` = `"1"`
- **Commit range:** first `5015b7d`, last `64ac09b`; superseded by `dd2c18c` (header, shard and contract change → INF-SHARD-2)
- **Commits (full-history log of the writer files while this layout was in effect):** `5015b7d`, `b09d44b`, `64ac09b`
- **Writers:**
  - `5015b7d:scripts/inflation_scan_shards_common.jl:61` `inflation_prepare_shard` (writes `INFLATION_SHARD_HEADER`, :71)
  - `5015b7d:scripts/inflation_scan_shards_common.jl:83` `inflation_append_shard_row` (stamps `shard_schema_version` from :11 = "1"; other columns from the summary)
  - `5015b7d:scripts/inflation_scan_shards_common.jl:170` `inflation_merge_shards` (writes the header :182, copies rows of shards whose first line equals the header)
  - `5015b7d:scripts/inflation_scan_merge_shards.jl:51` `main` (calls the merge at :55; one blob for the whole history)
  - `5015b7d:scripts/inflation_scan_prep.jl:292` `run_scan_prep` in shard mode (prepare :311, append :370)
  - `b09d44b:scripts/inflation_scan_pilot.jl:314` `run_inflation_pilot` (calls `run_scan_prep` in shard mode, :326); `64ac09b:scripts/inflation_scan_pilot.jl:344` (:357)
  - `run_geometry` as in INF-PREP-3 (`5015b7d`, `b09d44b`, `64ac09b`)
- **R-008 classification of every writer:** physical (all); `inflation_merge_shards` is a physical derived combiner (plan.md: "from shards").
- **Notes:** Header = `INFLATION_SHARD_METADATA_FIELDS` followed by `SCAN_PREP_FIELDS` at the same commit (the INF-PREP-3 header).

Exact header:

```text
shard_schema_version,run_id,shard_index,shard_count,attempt,started_unix_s,finished_unix_s,contract_version,diagnostic_schema_version,measurement_scope,data_dir,max_branches,h11,polytope,frst,status,error,instantons,selected_instantons,qtilde_det,leading_log_gap,log_scale_span,strong_hierarchy,branch_count,leading_minima_count,saddle_count,candidate_slowroll_saddles,least_tachyonic_min_eta,least_tachyonic_epsilon,best_min_eta,best_epsilon,best_abs_min_eta,mass_min,mass_max,negative_mass_count,stage_load_s,stage_select_s,stage_hierarchy_s,stage_factor_s,stage_mass_basis_s,stage_branches_s,stage_classify_s,stage_load_output_bytes,stage_select_output_bytes,stage_hierarchy_output_bytes,stage_factor_output_bytes,stage_mass_basis_output_bytes,stage_classify_output_bytes,stage_allocated_bytes,stage_output_bytes,total_seconds
```

### INF-SHARD-2

- **Family:** shard + diagnostic + scan contract
- **File kind:** scan-prep shard CSV and merged shard CSV
- **Header id:** `SHARD_2` (60 columns; SHA-256 of the header line without newline, first 16 hex: `38f4d22330cd475a`)
- **`row_type` values:** none (the header has no `row_type` column)
- **Version columns and values:** `shard_schema_version` = `"2"`; `contract_version` = `"4"`; `diagnostic_schema_version` = `"1"`
- **Commit range:** first `dd2c18c`, last `10aa3dc` (current at the base commit)
- **Commits (full-history log of the writer files while this layout was in effect):** `dd2c18c`, `8dffef1`, `560b901`, `4c0e1ad`, `d802686` (merge), `304a22c`, `a4c352e` (merge), `81e43c4` (merge), `b4d2ffb`, `900cd31` (merge)
- **Writers:**
  - `dd2c18c:scripts/inflation_scan_shards_common.jl:61` `inflation_prepare_shard`, `:83` `inflation_append_shard_row` (shard constant :11 = "2"), `:183` `inflation_merge_shards` — same blob at `10aa3dc`
  - `5015b7d:scripts/inflation_scan_merge_shards.jl:51` `main` (:55), unchanged
  - `run_scan_prep` shard mode: `dd2c18c:scripts/inflation_scan_prep.jl:328` (:347, :415); `560b901:scripts/inflation_scan_prep.jl:333` (:352, :420); `304a22c:scripts/inflation_scan_prep.jl:332` (:351, :419) = `10aa3dc`
  - `run_inflation_pilot`: `dd2c18c:scripts/inflation_scan_pilot.jl:365` (:378); `304a22c:scripts/inflation_scan_pilot.jl:364` (:377) = `10aa3dc`
  - `run_geometry` as in INF-PREP-4
- **R-008 classification of every writer:** physical (all).
- **Notes:** Header = `INFLATION_SHARD_METADATA_FIELDS` followed by the INF-PREP-4 `SCAN_PREP_FIELDS`.

Exact header:

```text
shard_schema_version,run_id,shard_index,shard_count,attempt,started_unix_s,finished_unix_s,contract_version,diagnostic_schema_version,measurement_scope,data_dir,max_branches,h11,polytope,frst,status,error,instantons,selected_instantons,qtilde_det,negative_mode_range,structured_charge_validated,structured_fallback_reason,search_classification,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,leading_log_gap,log_scale_span,strong_hierarchy,branch_count,leading_minima_count,saddle_count,candidate_slowroll_saddles,least_tachyonic_min_eta,least_tachyonic_epsilon,best_min_eta,best_epsilon,best_abs_min_eta,mass_min,mass_max,negative_mass_count,stage_load_s,stage_select_s,stage_hierarchy_s,stage_factor_s,stage_mass_basis_s,stage_branches_s,stage_classify_s,stage_load_output_bytes,stage_select_output_bytes,stage_hierarchy_output_bytes,stage_factor_output_bytes,stage_mass_basis_output_bytes,stage_classify_output_bytes,stage_allocated_bytes,stage_output_bytes,total_seconds
```

### INF-REF-3

- **Family:** refinement (`REFINEMENT_SCHEMA_VERSION`)
- **File kind:** refinement CSV (`--output`), written once with mode `"w"`
- **Header id:** `REF` (64 columns; SHA-256 of the header line without newline, first 16 hex: `4dda9b7a3206c99b`)
- **`row_type` values:** `point`, `summary`
- **Version columns and values:** `schema_version` = `"3"`
- **Commit range:** first `c7482f5`, last `10aa3dc` (current at the base commit)
- **Commits (full-history log of the writer files while this layout was in effect):** `c7482f5`, `304a22c`, `a4c352e` (merge), `81e43c4` (merge)
- **Writers:**
  - `c7482f5:scripts/inflation_candidate_refinement.jl:61` `_write_csv` (header :63)
  - `c7482f5:scripts/inflation_candidate_refinement.jl:150` `_point_row` (`row_type=:point`, `schema_version=REFINEMENT_SCHEMA_VERSION` :168)
  - `c7482f5:scripts/inflation_candidate_refinement.jl:387` `_summary_row` (`row_type=:summary` :391)
  - `c7482f5:scripts/inflation_candidate_refinement.jl:509` `main` (`_write_csv(options[:output], rows)` :550); `304a22c:scripts/inflation_candidate_refinement.jl:508` (:549) = `10aa3dc`
  - `c7482f5:scripts/inflation_candidate_refinement.jl:364` `scan_geometry_for_inflation`, `:214` `_refine_geometry` (K from `read.oriented_potential` :223); same lines at `304a22c`
- **R-008 classification of every writer:** physical (all).
- **Notes:** Constant `REFINEMENT_SCHEMA_VERSION = "3"` at `:17` from the first commit; no merged version "1" or "2" exists.

Exact header:

```text
row_type,schema_version,run_id,data_dir,code_commit,max_branches,negative_mode_range,random_starts,random_seed,max_points,float_tolerance,high_tolerance,duplicate_tolerance,working_basis,physical_basis,physical_vectors,scale_status,h11,polytope,frst,search_status,coverage_status,refinement_status,search_mode,branch_source,branch_estimate,branch_count,mask_count,masks_visited,branch_seed_index,leading_negative_modes,seed_theta,float_correction_status,float_correction_residual,float_correction_iterations,float_value,float_gradient_residual,float_gradient_norm,float_epsilon,float_min_eta,float_negative_modes,float_zeroish_modes,float_positive_modes,high_precision_bits,high_correction_status,high_correction_residual,high_correction_iterations,high_value,high_gradient_residual,high_gradient_norm,high_epsilon,high_min_eta,high_negative_modes,high_zeroish_modes,high_positive_modes,residual_agreement,inertia_agreement,accepted,candidate_status,candidate_reason,wall_seconds,allocated_bytes,output_bytes,failure
```

### INF-FLOW-1

- **Family:** flow (`FLOW_SCHEMA_VERSION`)
- **File kind:** flow CSV (`--flow-output`, default `<output base>.flows<ext>`), written once with mode `"w"`
- **Header id:** `FLOW` (36 columns; SHA-256 of the header line without newline, first 16 hex: `78e1ff39e5f8ecc1`)
- **`row_type` values:** `flow`
- **Version columns and values:** `schema_version` = `"1"`
- **Commit range:** first `c7482f5`, last `10aa3dc` (current at the base commit)
- **Commits (full-history log of the writer files while this layout was in effect):** `c7482f5`, `304a22c`, `a4c352e` (merge), `81e43c4` (merge)
- **Writers:**
  - `c7482f5:scripts/inflation_candidate_refinement.jl:61` `_write_csv` with `FLOW_FIELDS`
  - `c7482f5:scripts/inflation_candidate_refinement.jl:281` `_flow_row` (`row_type=:flow`, `schema_version=FLOW_SCHEMA_VERSION` :285)
  - `c7482f5:scripts/inflation_candidate_refinement.jl:313` `_flow_geometry`
  - `c7482f5:scripts/inflation_candidate_refinement.jl:509` `main` (:552); `304a22c:scripts/inflation_candidate_refinement.jl:508` (:551) = `10aa3dc`
- **R-008 classification of every writer:** physical (all).
- **Notes:** Constant `FLOW_SCHEMA_VERSION = "1"` at `:18`.

Exact header:

```text
row_type,schema_version,run_id,data_dir,code_commit,h11,polytope,frst,candidate_index,branch_seed_index,leading_negative_modes,seed_theta,float_theta,high_theta,precision_bits,working_basis,physical_basis,scale_status,coordinate_chart,mass_mode_index,displacement_sign,mass_eigenvalue,flow_theta_initial,flow_theta_final,flow_status,flow_end_event,flow_efolds,flow_slow_roll_efolds,flow_entry_efolds,flow_exit_efolds,flow_max_efolds,flow_step,flow_accepted,flow_reason,flow_error,flow_steps
```

### INF-PILOT-3-REPORT

- **Family:** pilot (`PILOT_SCHEMA_VERSION`)
- **File kind:** pilot report CSV (`--report`; one `scale` row per geometry and sampled scale; the deleted runner's per-geometry `summary.csv`)
- **Header id:** `PILOT_3` (72 columns; SHA-256 of the header line without newline, first 16 hex: `dad0cb3295e9bee4`)
- **`row_type` values:** `scale`
- **Version columns and values:** `schema_version` = `"3"`
- **Commit range:** first `10aa058`, last `b4d2ffb`/`900cd31`; superseded by `9f31d71` (header and version change → v4)
- **Commits (full-history log of the writer files while this layout was in effect):** `10aa058`, `2f05f6d`, `8f20b17`, `4c0e1ad`, `d802686` (merge), `304a22c`, `a4c352e` (merge), `81e43c4` (merge), `b4d2ffb`, `900cd31` (merge)
- **Writers:**
  - `pilot_prepare_csv` / `pilot_append_csv` / `_pilot_context` (schema stamp) / `_pilot_branch_row` / `_pilot_summary_row` / `run_scale_continuation` (fallback failure row) in `scripts/inflation_scale_continuation.jl`:
  -   `10aa058` and `2f05f6d`: :582 / :602 / :704 (:707) / :642 / :671 / :985 (:1076)
  -   `8f20b17`: :589 / :609 / :711 (:714) / :649 / :678 / :993 (:1084)
  -   `4c0e1ad`: :595 / :615 / :717 (:720) / :655 / :684 / :1001 (:1092)
  -   `304a22c` and `b4d2ffb`: :595 / :615 / :717 (:720) / :655 / :684 / :1001 (:1093)
  - constant `PILOT_SCHEMA_VERSION = "3"` at `scripts/inflation_scale_continuation.jl:23`
- **R-008 classification of every writer:** physical. `run_scale_continuation` loads K from `read.oriented_potential`; its A96 helper `pilot_benchmark_regression` only prints (`--benchmarks-only`) and never reaches the CSV writers.
- **Notes:** Same header as the matching SHARD entry; rows are told apart by `row_type` (`scale` here). `_pilot_summary_row` and the fallback failure row both write `row_type=scale` to the report.

Exact header:

```text
row_type,schema_version,run_id,data_root,geometry_path,h11,polytope,frst,reference_scale,sampled_scale,scale_grid,scale_source,scale_status,volume_normalization,stored_reference_max_log10_error,stored_reference_sign_mismatches,leading_log_gap,log_scale_span,strong_hierarchy,search_mode,branch_coverage_status,branch_estimate,branch_count,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,estimated_stage_allocated_bytes,max_stage_allocated_bytes,reference_uncorrected_screen_candidates,branch_seed_index,leading_negative_modes,leading_seed_theta,corrected_theta,correction_status,correction_residual,correction_iterations,branch_provenance_id,branch_match_id,matching_status,value,gradient_norm,epsilon,min_eta,max_eta,abs_min_eta,hessian_min,hessian_max,zeroish_modes,negative_modes,positive_modes,near_catastrophe,catastrophe_bracket,catastrophe_reason,candidate_status,candidate_reason,seed_count,corrected_count,matched_count,lost_count,new_count,duplicate_count,correction_failed_count,near_catastrophe_brackets,screen_candidate_count,corrected_candidate_count,minima_count,wall_seconds,allocated_bytes,output_bytes,failure
```

### INF-PILOT-3-SHARD

- **Family:** pilot (`PILOT_SCHEMA_VERSION`)
- **File kind:** pilot shard CSV (`<shard_dir>/inflation_scale_continuation_shard_NNNN_of_NNNN.csv`; one `branch` row per branch record)
- **Header id:** `PILOT_3` (72 columns; SHA-256 of the header line without newline, first 16 hex: `dad0cb3295e9bee4`)
- **`row_type` values:** `branch`
- **Version columns and values:** `schema_version` = `"3"`
- **Commit range:** first `10aa058`, last `b4d2ffb`/`900cd31`; superseded by `9f31d71` (header and version change → v4)
- **Commits (full-history log of the writer files while this layout was in effect):** `10aa058`, `2f05f6d`, `8f20b17`, `4c0e1ad`, `d802686` (merge), `304a22c`, `a4c352e` (merge), `81e43c4` (merge), `b4d2ffb`, `900cd31` (merge)
- **Writers:**
  - `pilot_prepare_csv` / `pilot_append_csv` / `_pilot_context` (schema stamp) / `_pilot_branch_row` / `_pilot_summary_row` / `run_scale_continuation` (fallback failure row) in `scripts/inflation_scale_continuation.jl`:
  -   `10aa058` and `2f05f6d`: :582 / :602 / :704 (:707) / :642 / :671 / :985 (:1076)
  -   `8f20b17`: :589 / :609 / :711 (:714) / :649 / :678 / :993 (:1084)
  -   `4c0e1ad`: :595 / :615 / :717 (:720) / :655 / :684 / :1001 (:1092)
  -   `304a22c` and `b4d2ffb`: :595 / :615 / :717 (:720) / :655 / :684 / :1001 (:1093)
  - constant `PILOT_SCHEMA_VERSION = "3"` at `scripts/inflation_scale_continuation.jl:23`
- **R-008 classification of every writer:** physical (as the REPORT entry).
- **Notes:** Same header as the matching REPORT entry; rows are told apart by `row_type` (`branch` here).

Exact header:

```text
row_type,schema_version,run_id,data_root,geometry_path,h11,polytope,frst,reference_scale,sampled_scale,scale_grid,scale_source,scale_status,volume_normalization,stored_reference_max_log10_error,stored_reference_sign_mismatches,leading_log_gap,log_scale_span,strong_hierarchy,search_mode,branch_coverage_status,branch_estimate,branch_count,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,estimated_stage_allocated_bytes,max_stage_allocated_bytes,reference_uncorrected_screen_candidates,branch_seed_index,leading_negative_modes,leading_seed_theta,corrected_theta,correction_status,correction_residual,correction_iterations,branch_provenance_id,branch_match_id,matching_status,value,gradient_norm,epsilon,min_eta,max_eta,abs_min_eta,hessian_min,hessian_max,zeroish_modes,negative_modes,positive_modes,near_catastrophe,catastrophe_bracket,catastrophe_reason,candidate_status,candidate_reason,seed_count,corrected_count,matched_count,lost_count,new_count,duplicate_count,correction_failed_count,near_catastrophe_brackets,screen_candidate_count,corrected_candidate_count,minima_count,wall_seconds,allocated_bytes,output_bytes,failure
```

### INF-PILOT-4-REPORT

- **Family:** pilot (`PILOT_SCHEMA_VERSION`)
- **File kind:** pilot report CSV (`--report`; one `scale` row per geometry and sampled scale; the deleted runner's per-geometry `summary.csv`)
- **Header id:** `PILOT_4` (92 columns; SHA-256 of the header line without newline, first 16 hex: `db90b15ce325ea22`)
- **`row_type` values:** `scale`
- **Version columns and values:** `schema_version` = `"4"`
- **Commit range:** first and last `9f31d71` (one commit); superseded by `b3719e3` (header change, version "4" → "6"; no merged "5")
- **Commits (full-history log of the writer files while this layout was in effect):** `9f31d71`
- **Writers:**
  - `9f31d71:scripts/inflation_scale_continuation.jl`: `pilot_prepare_csv` :1233, `pilot_append_csv` :1253, `_pilot_context` :1355 (schema stamp :1398), `_pilot_branch_row` :1293, `_pilot_summary_row` :1322, `run_scale_continuation` :1703 (prepare :1719–1720; appends :1770, :1782, :1784; fallback row :1809, append :1854)
  - constant `PILOT_SCHEMA_VERSION = "4"` at `scripts/inflation_scale_continuation.jl:23`
- **R-008 classification of every writer:** physical. `run_scale_continuation` loads K from `read.oriented_potential`; its A96 helper `pilot_benchmark_regression` only prints (`--benchmarks-only`) and never reaches the CSV writers.
- **Notes:** Same header as the matching SHARD entry; rows are told apart by `row_type` (`scale` here). `_pilot_summary_row` and the fallback failure row both write `row_type=scale` to the report.

Exact header:

```text
row_type,schema_version,run_id,data_root,geometry_path,h11,polytope,frst,reference_scale,sampled_scale,scale_grid,scale_source,scale_status,volume_normalization,domain_certificate_version,domain_status,domain_reason,fixed_point_status,trajectory_status,coverage_status,moduli_status,phase_convention,units,normalization,source_identity,precision_bits,source_numeric_type,source_precision_bits,target_numeric_type,target_precision_bits,conversion_status,conversion_error_bound,conversion_tolerance,conversion_comparison,stored_reference_max_log10_error,stored_reference_sign_mismatches,leading_log_gap,log_scale_span,strong_hierarchy,search_mode,branch_coverage_status,branch_estimate,branch_count,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,estimated_stage_allocated_bytes,max_stage_allocated_bytes,reference_uncorrected_screen_candidates,branch_seed_index,leading_negative_modes,leading_seed_theta,corrected_theta,correction_status,correction_residual,correction_iterations,branch_provenance_id,branch_match_id,matching_status,value,gradient_norm,epsilon,min_eta,max_eta,abs_min_eta,hessian_min,hessian_max,zeroish_modes,negative_modes,positive_modes,near_catastrophe,catastrophe_bracket,catastrophe_reason,candidate_status,candidate_reason,seed_count,corrected_count,matched_count,lost_count,new_count,duplicate_count,correction_failed_count,near_catastrophe_brackets,screen_candidate_count,corrected_candidate_count,minima_count,wall_seconds,allocated_bytes,output_bytes,failure
```

### INF-PILOT-4-SHARD

- **Family:** pilot (`PILOT_SCHEMA_VERSION`)
- **File kind:** pilot shard CSV (`<shard_dir>/inflation_scale_continuation_shard_NNNN_of_NNNN.csv`; one `branch` row per branch record)
- **Header id:** `PILOT_4` (92 columns; SHA-256 of the header line without newline, first 16 hex: `db90b15ce325ea22`)
- **`row_type` values:** `branch`
- **Version columns and values:** `schema_version` = `"4"`
- **Commit range:** first and last `9f31d71` (one commit); superseded by `b3719e3` (header change, version "4" → "6"; no merged "5")
- **Commits (full-history log of the writer files while this layout was in effect):** `9f31d71`
- **Writers:**
  - `9f31d71:scripts/inflation_scale_continuation.jl`: `pilot_prepare_csv` :1233, `pilot_append_csv` :1253, `_pilot_context` :1355 (schema stamp :1398), `_pilot_branch_row` :1293, `_pilot_summary_row` :1322, `run_scale_continuation` :1703 (prepare :1719–1720; appends :1770, :1782, :1784; fallback row :1809, append :1854)
  - constant `PILOT_SCHEMA_VERSION = "4"` at `scripts/inflation_scale_continuation.jl:23`
- **R-008 classification of every writer:** physical (as the REPORT entry).
- **Notes:** Same header as the matching REPORT entry; rows are told apart by `row_type` (`branch` here).

Exact header:

```text
row_type,schema_version,run_id,data_root,geometry_path,h11,polytope,frst,reference_scale,sampled_scale,scale_grid,scale_source,scale_status,volume_normalization,domain_certificate_version,domain_status,domain_reason,fixed_point_status,trajectory_status,coverage_status,moduli_status,phase_convention,units,normalization,source_identity,precision_bits,source_numeric_type,source_precision_bits,target_numeric_type,target_precision_bits,conversion_status,conversion_error_bound,conversion_tolerance,conversion_comparison,stored_reference_max_log10_error,stored_reference_sign_mismatches,leading_log_gap,log_scale_span,strong_hierarchy,search_mode,branch_coverage_status,branch_estimate,branch_count,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,estimated_stage_allocated_bytes,max_stage_allocated_bytes,reference_uncorrected_screen_candidates,branch_seed_index,leading_negative_modes,leading_seed_theta,corrected_theta,correction_status,correction_residual,correction_iterations,branch_provenance_id,branch_match_id,matching_status,value,gradient_norm,epsilon,min_eta,max_eta,abs_min_eta,hessian_min,hessian_max,zeroish_modes,negative_modes,positive_modes,near_catastrophe,catastrophe_bracket,catastrophe_reason,candidate_status,candidate_reason,seed_count,corrected_count,matched_count,lost_count,new_count,duplicate_count,correction_failed_count,near_catastrophe_brackets,screen_candidate_count,corrected_candidate_count,minima_count,wall_seconds,allocated_bytes,output_bytes,failure
```

### INF-PILOT-6-REPORT

- **Family:** pilot (`PILOT_SCHEMA_VERSION`)
- **File kind:** pilot report CSV (`--report`; one `scale` row per geometry and sampled scale; the deleted runner's per-geometry `summary.csv`)
- **Header id:** `PILOT_6` (102 columns; SHA-256 of the header line without newline, first 16 hex: `5a3553bc1b844207`)
- **`row_type` values:** `scale`
- **Version columns and values:** `schema_version` = `"6"`
- **Commit range:** first `b3719e3`, last `10aa3dc` (current at the base commit)
- **Commits (full-history log of the writer files while this layout was in effect):** `b3719e3`, `1ea31f5` (merge), `78cb603` (merge), `07552d0` (merge), `d72ec45` (merge), `2a4e495`, `b35cb74`, `eceb003`, `34f4fd3` (merge), `995163f` (merge), `5dd236a`, `01b6123` (merge), `be9ed87`, `7b6b36a` (merge), `f011838` (merge), `fb8139c` (merge); runner file also `1ed97e1`; runner deleted by `bd6807e` (merged by `8b21216`)
- **Writers:**
  - `scripts/inflation_scale_continuation.jl` `pilot_prepare_csv` / `pilot_append_csv` / `_pilot_context` (schema stamp) / `_pilot_branch_row` / `_pilot_summary_row` / `run_scale_continuation` (fallback failure row):
  -   `b3719e3`, `2a4e495`, `b35cb74`, `eceb003`: :1446 / :1466 / :1597 (:1654) / :1506 / :1537 / :1979, :1988, :2011, :2024 respectively (fallback :2125, :2134, :2157, :2170)
  -   `5dd236a`: :1445 / :1465 / :1596 (:1653) / :1505 / :1536 / :1978 (fallback :2124)
  -   `7b6b36a` (= `10aa3dc`): :1445 / :1465 / :1596 (:1653) / :1505 / :1536 / :2023 (fallback :2169)
  - constant `PILOT_SCHEMA_VERSION = "6"` at `scripts/inflation_scale_continuation.jl:25`
  - deleted runner `scripts/run_physical_scale_inflation_pilot_20260825.jl` `run_one_geometry`: `b3719e3:scripts/run_physical_scale_inflation_pilot_20260825.jl:1379` (prepare :1386–1387; appends :1413, :1415), `5dd236a:scripts/run_physical_scale_inflation_pilot_20260825.jl:1388` (:1395–1396; :1422, :1424), `1ed97e1:scripts/run_physical_scale_inflation_pilot_20260825.jl:1396` (:1403–1404; :1430, :1432). It writes `PILOT_SUMMARY_FIELDS`/`PILOT_BRANCH_FIELDS` of the `scripts/inflation_scale_continuation.jl` it includes (see Open points)
- **R-008 classification of every writer:** physical. `run_scale_continuation` loads K from `read.oriented_potential`; its A96 helper `pilot_benchmark_regression` only prints (`--benchmarks-only`) and never reaches the CSV writers.
- **Notes:** Same header as the matching SHARD entry; rows are told apart by `row_type` (`scale` here). `_pilot_summary_row` and the fallback failure row both write `row_type=scale` to the report.

Exact header:

```text
row_type,schema_version,run_id,data_root,geometry_path,h11,polytope,frst,reference_scale,sampled_scale,scale_grid,scale_source,scale_status,volume_normalization,domain_certificate_version,domain_status,domain_reason,physical_scaling_gate_status,physical_scaling_gate_reason,physical_scaling_gate_provenance,physical_control_gate_status,physical_control_gate_reason,physical_control_gate_provenance,physical_viability_status,physical_viability_reason,fixed_point_status,trajectory_status,coverage_status,moduli_status,phase_convention,units,normalization,source_identity,precision_bits,source_numeric_type,source_precision_bits,target_numeric_type,target_precision_bits,conversion_status,conversion_error_bound,conversion_tolerance,conversion_comparison,conversion_policy_version,stored_reference_max_log10_error,stored_reference_sign_mismatches,leading_log_gap,log_scale_span,strong_hierarchy,search_mode,branch_coverage_status,branch_estimate,branch_count,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,estimated_stage_allocated_bytes,max_stage_allocated_bytes,reference_uncorrected_screen_candidates,branch_seed_index,leading_negative_modes,leading_seed_theta,corrected_theta,correction_status,correction_residual,correction_iterations,branch_provenance_id,branch_match_id,matching_status,value,gradient_norm,epsilon,min_eta,max_eta,abs_min_eta,hessian_min,hessian_max,zeroish_modes,negative_modes,positive_modes,near_catastrophe,catastrophe_bracket,catastrophe_reason,candidate_status,candidate_reason,seed_count,corrected_count,matched_count,lost_count,new_count,duplicate_count,correction_failed_count,near_catastrophe_brackets,screen_candidate_count,corrected_candidate_count,physical_eligible_count,minima_count,wall_seconds,allocated_bytes,output_bytes,failure
```

### INF-PILOT-6-SHARD

- **Family:** pilot (`PILOT_SCHEMA_VERSION`)
- **File kind:** pilot shard CSV (`<shard_dir>/inflation_scale_continuation_shard_NNNN_of_NNNN.csv`; one `branch` row per branch record)
- **Header id:** `PILOT_6` (102 columns; SHA-256 of the header line without newline, first 16 hex: `5a3553bc1b844207`)
- **`row_type` values:** `branch`
- **Version columns and values:** `schema_version` = `"6"`
- **Commit range:** first `b3719e3`, last `10aa3dc` (current at the base commit)
- **Commits (full-history log of the writer files while this layout was in effect):** `b3719e3`, `1ea31f5` (merge), `78cb603` (merge), `07552d0` (merge), `d72ec45` (merge), `2a4e495`, `b35cb74`, `eceb003`, `34f4fd3` (merge), `995163f` (merge), `5dd236a`, `01b6123` (merge), `be9ed87`, `7b6b36a` (merge), `f011838` (merge), `fb8139c` (merge); runner file also `1ed97e1`; runner deleted by `bd6807e` (merged by `8b21216`)
- **Writers:**
  - `scripts/inflation_scale_continuation.jl` `pilot_prepare_csv` / `pilot_append_csv` / `_pilot_context` (schema stamp) / `_pilot_branch_row` / `_pilot_summary_row` / `run_scale_continuation` (fallback failure row):
  -   `b3719e3`, `2a4e495`, `b35cb74`, `eceb003`: :1446 / :1466 / :1597 (:1654) / :1506 / :1537 / :1979, :1988, :2011, :2024 respectively (fallback :2125, :2134, :2157, :2170)
  -   `5dd236a`: :1445 / :1465 / :1596 (:1653) / :1505 / :1536 / :1978 (fallback :2124)
  -   `7b6b36a` (= `10aa3dc`): :1445 / :1465 / :1596 (:1653) / :1505 / :1536 / :2023 (fallback :2169)
  - constant `PILOT_SCHEMA_VERSION = "6"` at `scripts/inflation_scale_continuation.jl:25`
  - deleted runner `scripts/run_physical_scale_inflation_pilot_20260825.jl` `run_one_geometry`: `b3719e3:scripts/run_physical_scale_inflation_pilot_20260825.jl:1379` (prepare :1386–1387; appends :1413, :1415), `5dd236a:scripts/run_physical_scale_inflation_pilot_20260825.jl:1388` (:1395–1396; :1422, :1424), `1ed97e1:scripts/run_physical_scale_inflation_pilot_20260825.jl:1396` (:1403–1404; :1430, :1432). It writes `PILOT_SUMMARY_FIELDS`/`PILOT_BRANCH_FIELDS` of the `scripts/inflation_scale_continuation.jl` it includes (see Open points)
- **R-008 classification of every writer:** physical (as the REPORT entry).
- **Notes:** Same header as the matching REPORT entry; rows are told apart by `row_type` (`branch` here).

Exact header:

```text
row_type,schema_version,run_id,data_root,geometry_path,h11,polytope,frst,reference_scale,sampled_scale,scale_grid,scale_source,scale_status,volume_normalization,domain_certificate_version,domain_status,domain_reason,physical_scaling_gate_status,physical_scaling_gate_reason,physical_scaling_gate_provenance,physical_control_gate_status,physical_control_gate_reason,physical_control_gate_provenance,physical_viability_status,physical_viability_reason,fixed_point_status,trajectory_status,coverage_status,moduli_status,phase_convention,units,normalization,source_identity,precision_bits,source_numeric_type,source_precision_bits,target_numeric_type,target_precision_bits,conversion_status,conversion_error_bound,conversion_tolerance,conversion_comparison,conversion_policy_version,stored_reference_max_log10_error,stored_reference_sign_mismatches,leading_log_gap,log_scale_span,strong_hierarchy,search_mode,branch_coverage_status,branch_estimate,branch_count,mask_count,masks_visited,masks_skipped,lattice_copy_count,lattice_copies_visited,estimated_stage_allocated_bytes,max_stage_allocated_bytes,reference_uncorrected_screen_candidates,branch_seed_index,leading_negative_modes,leading_seed_theta,corrected_theta,correction_status,correction_residual,correction_iterations,branch_provenance_id,branch_match_id,matching_status,value,gradient_norm,epsilon,min_eta,max_eta,abs_min_eta,hessian_min,hessian_max,zeroish_modes,negative_modes,positive_modes,near_catastrophe,catastrophe_bracket,catastrophe_reason,candidate_status,candidate_reason,seed_count,corrected_count,matched_count,lost_count,new_count,duplicate_count,correction_failed_count,near_catastrophe_brackets,screen_candidate_count,corrected_candidate_count,physical_eligible_count,minima_count,wall_seconds,allocated_bytes,output_bytes,failure
```

## Not entries (checked and excluded)

- **A96 diagnostic candidate row.** `inflation_refinement_diagnostic_row`
  (`814988e:scripts/inflation_diagnostics_common.jl:39`) flattens the A96 summary built by
  `_refinement_summary` (`814988e:scripts/inflation_refinement_common.jl:67`,
  stamp at :74; the trajectory comes from `poly102_inflation.n8_physical_gradient_flow`).
  The only merged caller of `inflation_append_diagnostic_row`
  (`814988e:scripts/inflation_diagnostics_common.jl:108`) and `inflation_diagnostic_csv_line` is the test at
  `10aa3dc:test/runtests.jl:3463–3483` (temporary directory). Per A3.3 this
  does not make the writer a writer of a recorded entry. For reference, its header
  (38 columns, `diagnostic_schema_version` = "1", no `row_type`) is
  `diagnostic_schema_version,candidate_id,model,screen_status,screen_accepted,screen_measurement_scope,screen_value,screen_epsilon,screen_min_eta,screen_negative_modes,screen_wall_seconds,screen_allocated_bytes,screen_output_bytes,refinement_status,refinement_error,refinement_measurement_status,refinement_measurement_scope,refinement_precision_bits,refinement_solver_method,refinement_solver_retcode,refinement_event_policy,refinement_entered_slow_roll,refinement_end_event,refinement_terminated,refinement_efolds,refinement_slow_roll_efolds,refinement_accepted_steps,refinement_rejected_steps,refinement_rhs_evaluations,refinement_jacobian_evaluations,refinement_wall_seconds,refinement_allocated_bytes,refinement_output_bytes,serialization_status,serialization_measurement_scope,serialization_wall_seconds,serialization_allocated_bytes,serialization_output_bytes`.
  It equals no entry header, so such a row is rejected as unknown.
  `scripts/inflation_refinement_common.jl` itself has no file write in either blob
  (`851d488`, `814988e`).
- **`scripts/inflation_scan_contract.jl`** stamps `contract_version` on rows it
  only prints (`println(summary)`), at every blob (`8b28553`, `2664648`,
  `fdf75f7`, `dd2c18c`, `304a22c`).
- **`scripts/inflation_scan_pilot.jl` `_inflation_pilot_write_report`**
  (`b09d44b:scripts/inflation_scan_pilot.jl:286`, `304a22c:scripts/inflation_scan_pilot.jl:330`): `inflation_scan_pilot_report.csv`
  has no version column and no `row_type` (fields `INFLATION_PILOT_REPORT_FIELDS`,
  17 columns at `b09d44b`, 20 from `64ac09b`). It is not in these families
  (plan.md lists it as an unversioned derived CSV). Its shard output is INF-SHARD-1/-2.
- **`scripts/build_orientifold_vacua_inflation.jl`** includes
  `inflation_scan_common.jl` and `inflation_candidate_refinement.jl` but writes
  only the F6 HDF5 group; no CSV, and no `contract_version` or
  `diagnostic_schema_version` in any blob.
- **`scripts/compare_reduction_inflation_scan.jl`**, **`scripts/issue_148_g2_continuation_evidence.jl`**,
  **`validation/inflation_author_code_vs_julia_*.jl`** include
  `inflation_scale_continuation.jl` but never call `pilot_prepare_csv`/`pilot_append_csv`;
  their CSVs (if any) have no family version column.
- **Tests.** `test/runtests.jl` also calls `_write_csv` with `FLOW_FIELDS`
  (`10aa3dc:test/runtests.jl:699`), `run_scan_prep` and `inflation_merge_shards`;
  test-persisted files are not merged-writer outputs.
- **Search record.** `git log 10aa3dc -S<constant>` for the six constants;
  `-S`/`-G` for `SCAN_PREP_FIELDS`, `INFLATION_SHARD_FIELDS`, `REFINEMENT_FIELDS`,
  `FLOW_FIELDS`, `PILOT_COMMON_FIELDS`, `PILOT_SUMMARY_FIELDS`, `PILOT_BRANCH_FIELDS`,
  `pilot_prepare_csv`, `pilot_append_csv`, `inflation_prepare_shard`,
  `inflation_append_shard_row`, `inflation_merge_shards`, `_scan_prep_append`,
  `run_scan_prep`, `_write_csv(`, `contract_version`, `diagnostic_schema_version`,
  `shard_schema_version`, `row_type`, `inflation_append_diagnostic_row`; every
  `include` of the inflation scripts; `--follow` on each writer file (no renames:
  `inflation_scan_common.jl` was created by copy from `inflation_scan_contract.jl`
  at `2664648`); all deleted/renamed code files in history. The only additional
  merged writer found is the deleted `scripts/run_physical_scale_inflation_pilot_20260825.jl`.

## Distinguishability

All ten header ids are pairwise different (checked by the script), including
across families: refinement (`row_type,schema_version,run_id,data_dir,…`) and
pilot (`row_type,schema_version,run_id,data_root,…`) differ from the fourth
column on, and both differ from the flow header (`…,code_commit,h11,…`). So the
refinement "3" (INF-REF-3) and the pilot "3" (INF-PILOT-3-*) cannot be confused,
and the shared column name `schema_version` never makes two families collide.
Every header also differs from the current (post-change) headers, which carry
`kinetic_normalization`.

Entries that share a header (A3.2 check: each pair lists the same version
columns):

| Entry pair | Shared header | Version columns | Told apart by |
|---|---|---|---|
| INF-PREP-1 / INF-PREP-2 | `PREP_A` | `contract_version` (both) | `contract_version` "1" vs "2" |
| INF-PILOT-3-REPORT / INF-PILOT-3-SHARD | `PILOT_3` | `schema_version` (both) | `row_type` `scale` vs `branch` |
| INF-PILOT-4-REPORT / INF-PILOT-4-SHARD | `PILOT_4` | `schema_version` (both) | `row_type` `scale` vs `branch` |
| INF-PILOT-6-REPORT / INF-PILOT-6-SHARD | `PILOT_6` | `schema_version` (both) | `row_type` `scale` vs `branch` |

No pair of entries has the same header, an overlapping `row_type` set and the
same version values, so **no row written by a merged writer can match two
entries**. The scan-prep shard file and the merged shard file are one layout and
are recorded as one entry each (INF-SHARD-1, INF-SHARD-2), not two.

A row matches no entry (and is rejected under A3.3) when, for example, its
header is a recorded one but its version value differs (e.g. `PREP_B` with
`contract_version` "4"), or its `row_type` is outside the entry's set (e.g.
`REF` with `row_type` `flow`).

## Open points

1. **Header-only files.** A3 decides rows, not files. A file with only a header
   is ambiguous at file level where two entries share the header: a `PREP_A`
   file (INF-PREP-1 or -2) and a pilot report vs pilot shard of the same version
   (identical bytes). Merged writers can produce such files (a flow CSV with no flow rows, a
   pilot shard with no branch rows, a summary or shard after its header write only). A reader that must return a label for an
   empty file needs a rule; A3 does not give one (no row means no row-level match).
2. **Files mixing two entries.** `--append-summary`/`--resume` in `run_scan_prep`
   check only the header, so a summary started at `2664648`/`201f0ac` and
   continued at `fdf75f7` holds INF-PREP-1 and INF-PREP-2 rows. Each row still
   matches exactly one entry; both entries have only physical writers, so both
   infer `:legacy_c_half`. No other version change kept the header, so no other
   mix is possible from merged code (the pilot report/shard and the shard/merged
   files each hold a single entry).
3. **Deleted runner `scripts/run_physical_scale_inflation_pilot_20260825.jl` (present `b3719e3` → deleted by `bd6807e`).**
   It `include`s `scripts/inflation_scale_continuation.jl` from a configured
   worktree path (`CONTINUATION`, an absolute machine-local path at `b3719e3`,
   environment-configurable from `5dd236a`), not from its own commit. The layout
   of the `summary.csv` and shard files it writes is therefore set by the included
   file, not by the runner's commit. With the merged continuation of its period
   (v6) these are INF-PILOT-6-REPORT/-SHARD, and it is recorded as such; its
   `REQUIRED_COMMIT` (`9f31d71`) plus an uncommitted diff suggests the runs
   used a working copy, so files from those runs may hold other layouts.
   Classification: physical (`load_geometry_evidence` takes K from
   `read.oriented_potential`).
4. **Runner copies of legacy files.** `adopt_existing_geometry`
   (`b3719e3:scripts/run_physical_scale_inflation_pilot_20260825.jl:1054`, copy at :1070; `5dd236a` :1063/:1079; `1ed97e1`
   :1071/:1087) copies a pre-existing summary and shard byte-for-byte; the Python
   validator (`scripts/validate_physical_scale_checkpoint_20260825.py`) checks
   `row_type`, geometry and gate columns but not the header or `schema_version`.
   `validate_old_header_only_csv` (:604) also accepts header-only files whose
   header is the v6 header **without** `conversion_policy_version` (101 columns),
   a layout no merged `inflation_scale_continuation.jl` blob defines. These are
   copies of layouts from unmerged code; they are not recorded as entries, and
   under A3.3 such rows (if any) match no entry and are rejected. Whether the
   copy step is a "writer" of an entry is not decided by A3.
5. **`inflation_merge_shards` has no row check.** It validates only each
   shard's header line and copies rows verbatim, so a merged file contains exactly
   the rows of its shards (single entry per merged file, as above). It is
   recorded as a physical writer of INF-SHARD-1/-2 together with the shard
   writers; plan.md lists it as "physical (not named in spec)".
6. **Privacy.** The deleted runner's source contains machine-local absolute
   paths; none are reproduced here. Keep it that way when this record is copied
   into `plan.md`.
