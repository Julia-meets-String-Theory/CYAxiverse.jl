# Implementation Plan — CYAX-0232

## Governing specification

Canonical specification: `specs/0232-kinetic-normalization/spec.md`

Spec revision approved for this plan: `f097536` (blob `0a454b3`), approved on
Issue #232 (`approval_ref` in `spec.md`). The approval recorded seven G1 notes;
this plan resolves each (see *Approval notes*).

Implementation base: `origin/vmm` at `10aa3dc`. All file:line citations below
refer to `10aa3dc`. If the implementation branch is based on a later `vmm`,
the inventories are re-run and the R-002 baseline is regenerated (spec R-002).

## Owner decisions before the affected tasks

The inventory found points that the approved spec does not decide (spec stop
condition). Each holds only the tasks named; all other tasks proceed.

**Status (2026-11-12).** The owner adopted an independent review of this plan.
The decisions are drafted as `clarification-a.md` (A1–A7), which supersedes the
recommendations below where they differ:
- OD-1 adds the `pq_spectrum_save` phase transformation (A1);
- OD-2 requires a recorded historical layout before inference (A2);
- OD-3 drops the column priority, requires a consistent version tuple, and
  rejects ambiguous rows (A3);
- OD-6 states that the script's outputs are not certified (A6);
- OD-7 uses field-level labels instead of one A96 label (A7).

Focused G0 round 1 of `f6c75bd` returned REQUEST_CHANGES (F5 versions span
commit ranges; inflation family identity needs the header; the A7 trigger).
The revision resolves all findings and nits.

The held tasks stay held until Clarification A passes a focused G0 review and
the owner approves it.

| ID | Question | Original recommendation | Held tasks |
|---|---|---|---|
| OD-1 | `generate.cubic` (generate.jl:567) expects `L` as N×2 and `Q` as N×h11. `pq_spectrum_save` (3013–3015) passes `Ltilde` as 2×h11, and `hp_spectrum_save` (1073) passes 2×N and h11×N. The stored F2 cubic tensor is mis-indexed at h11 = 2 and the call throws otherwise. Neither function has a test. Is the orientation fix in scope? | Fix it in this work (F2 gets a new schema anyway, and legacy F2 already reads as `cubic_basis = :unknown`); record legacy F2 cubic tensors as unreliable in the conventions page. | T030 |
| OD-2 | F5 legacy inference lists only `reproduction-1.2` and `kaehler-production-1.0`. Versions `1.0` (written by `ec73c15`…`e7a8f51`; `model_stage` from `aee4c1a` on) and `1.1` (`b191af1`, `31cf465`, `f45ea31`, `68df2a2`, `239a3a7`) exist; `analyze_fuzzy_axions_orientifold_gap.py` accepts only `1.1` (lines 56, 656, 997). | Amend R-007 to infer F5 `1.0` and `1.1` as `:legacy_c_half`, as for F6 2.0–2.2. | T060, T061 |
| OD-3 | Approval note 5 limits inflation CSV inference to the pre-change constants "unless history shows earlier versions in use". History shows contract `1`–`3`, shard `1`, and pilot `3`/`4` written by merged code. Rows also carry several version columns at once. | Infer every historical version that merged code wrote; family membership is decided by `contract_version`, then `diagnostic_schema_version`, then the file's own schema column. | T063, T064 |
| OD-4 | Reproduction functions that use no kinetic matrix (for example `validate_vacua_stage4_5.jl` `_stage4_n5_n8`, `catastrophic_inflation_population_study.jl` `main`, `reproduce_2023_n8.jl`, the K-free `reduced_models`/`author_inflation` functions) cannot be assigned P96 or A96 by R-008. | No label duty for functions and outputs that consume no kinetic matrix (they are not K-dependent under R-006); record them as K-free in the search record. | T072 |
| OD-5 | `inflation_scale_continuation.jl` `run_scale_continuation` (2023) is a resumable physical runner not named in R-007 *Batch resume*; its `_pilot_completed_scales` (1474) swallows parse errors (1488–1494) and has no overwrite option. | Treat as `inflation_scan_prep.jl`: legacy rows are not complete, `--resume` on a legacy report refuses, remedy is a new report path; parse errors propagate. | T066 |
| OD-6 | `scan_fresh_relaxed_qcd_candidates.py` passes `kinv` to the untracked `cytools_catastrophe_scan.solve_at_k`; its outputs cannot be classified from the repository. | Out of scope; document that the script is not covered and refuse to label its outputs. | T040 (inventory note only) |
| OD-7 | `pilot_benchmark_regression` (inflation_scale_continuation.jl:1829) mixes A96 and P96 (`n8_hilltop`, 1862) and returns a printed `NamedTuple`. | Label it `:a96_reproduction_raw_radian` and document the P96 sub-call in its docstring. | T072 |

`read.K` is decided here without the owner: it is kept and documented as
unlabelled and not physical (R-001(a) allows this, and it avoids a signature
change that the Interfaces section excludes). It has no caller and no writer at
`10aa3dc`. It reads `cytools/potential/K`, not the stored
`cytools/geometric/Kinv`. T020 adds a docstring warning that it is not the
supported physical kinetic-matrix accessor and points to `kinetic_matrix`.

## Approval notes (G1 obligations from the approval comment)

| Note | Resolution in this plan |
|---|---|
| 1. R-001(b) covers every K-taking API | K-argument inventory below includes `pq_window_spectrum` (2741), `physical_mode_inertia_count` (2804), `confirm_physical_mode_count` (2825), `pq_physical_mode_count` (2848); T020 adds their docstrings. |
| 2. Scalar fixture is new | T011 adds it. |
| 3. Pin fixture and test per R-005 row | *R-005 row map* below. |
| 4. `run_batch_axion_photon` rejection is per-geometry `:failed` | T065 moves the existence check and configuration read inside the per-geometry `try` (now 2101–2109, outside the `try` at 2122). |
| 5. Inflation CSV legacy versions | OD-3. |
| 6. Eq. 2.15 verified by R-004 | T011 asserts it. |
| 7. `notebooks/` derived writers follow R-006 | Writer inventory includes `notebooks/orientifold_axiverse_statistics.jl`. |

## Coverage

| Requirement / Gate | Planned implementation | Planned verification |
| --- | --- | --- |
| R-001 | `kinetic_factor`, `kinetic_matrix` in `src/read.jl`, exported; route `potential_factored` (295), `_kinetic_matrix` (180–182), `_canonical_frame` (axion_photon.jl:1037), the QED norm (1152), `leading_axion_reference_data` (fuzzy_axions_model_stage.jl:85) and `benchmark_hybrid_scaling.jl` through them; Python evaluator (`reproduce_fuzzy_axions_h11_4.py:1102`) equivalent; K-argument docstrings. | T010, T020, T021: unit tests, grep audit, Python equivalence test. |
| R-002 | Baseline artifacts at the base before any change; sha256 recorded in *Baseline*. | T002, T090. |
| R-003 | Testset on the h11 = 2 literals (runtests.jl:2017–2050), QCD first. | T012. |
| R-004 | Non-diagonal fixture with the pins below; orientation guard; scalar fixture; unchanged "HP spectrum: one-axion analytic mass" (runtests.jl:2052). | T011. |
| R-005 | c = ½ comparison harness passes `2·Kinv` where `Kinv` is expected (the F5 evaluator, model stage and HDF5 routes use `Kinv` only for the factorization). One test per row of the *R-005 row map*. | T013, T014, T015, T016. |
| R-006 | Labels and versions per the writer inventory; `KINETIC_NORMALIZATION`; reader fields; `stored_kinetic_normalization`. | T030–T045 round trips. |
| R-007 | Reader inventory duties; batch-resume behaviour; error propagation; writer refusals; layout migrations. | T060–T067. |
| R-008 | Labels on reproduction structs and `NamedTuple`s; docstrings for bare arrays; search record below. | T070–T072. |
| R-009 | `docs/src/conventions/kinetic_normalization.md`; README and axion–photon page pointers. | T080, docs build. |
| R-010 | Certification audit (done below: no `Kinv`-derived quantity). | T081 records it in the PR. |
| R-011 | Reciprocity check expects `K·Kinv = 2I`. | T050. |
| G1 | Fast and full suites, Python regressions, docs, `scripts/agent_verify.py` at the exact SHA; changed-test list; R-002 hashes; R-005 count report. | T090–T092. |
| G2 | Fresh-context review at the exact SHA. | T093. |

## Existing architecture

- Two `src/read.jl` conversions supply c = ½ to every physical path:
  `potential_factored` (`C = chol(Kinv)`, 295) and `_kinetic_matrix`
  (`K = inv(Kinv)`, 180–182, used by `potential` 255/264 and
  `oriented_potential` 409/421).
- Direct `Kinv` factorizations: `_canonical_frame` (axion_photon.jl:1037), the
  QED norm (1151–1152), `leading_axion_reference_data`
  (fuzzy_axions_model_stage.jl:85), the Python evaluator
  (reproduce_fuzzy_axions_h11_4.py:1102), `benchmark_hybrid_scaling.jl`.
- K-taking APIs inherit their convention from the caller (K-argument contract,
  confirmed in the approval).

## Inventories (at `10aa3dc`)

### Writer inventory (R-006 general rule)

K-dependence chains: **A** `potential_factored().C`; **B** `read.potential().K`;
**B′** `read.oriented_potential().K`; **C** `_canonical_frame` and QED norm;
**D** `leading_axion_reference_data`; **E** Python `_fuzzy_axion_mass_log10`.

| Writer (file:line) | Output | Family | Version now → after | Chain | R-008 |
|---|---|---|---|---|---|
| `generate.jl:1064` `hp_spectrum_save` | `spectrum/{quartdiag,decay,quart31,quart22,masses,cubic}` | F2 | none → `spectrum-groups-1`, `cubic_basis="charge_basis"` | A | physical (no caller) |
| `generate.jl:3006` `pq_spectrum_save` (driver `scripts/spectra.jl:23`) | `spectrum/{decay,masses,cubic}` | F2 | none → `spectrum-groups-1`, `cubic_basis="canonical_mass_basis"` | A | physical |
| `batch_physical_spectrum.jl` `_write_result`/`_write_result_inplace` (294/226) | `spectrum/physical/*` | F1 | `-3` (:12) → `-4` | A, B (`_fK_log10`, 206–210) | physical |
| same, `_write_summary_header`/`_append_summary` (319/333) | summary CSV | F1 summary | none; label column | from F1 | physical |
| `axion_photon.jl:1860` `write_axion_photon_result` | `spectrum/axion_photon/*` | F3 | identity 1 (:201) → 2 | C | physical (derived on read-back) |
| `axion_photon.jl:1337` `write_scan_csv` (driver `run_axion_photon_scan.jl:97`) | scan CSV | F3 CSV | none; label column | C | physical (derived on read-back) |
| `fuzzy_axion_model_stage_driver.jl` `main` (116–132, opens `"w"` at 119) | root datasets | F4 | none → `fuzzy-axion-model-stage-1` | D | physical |
| `reproduce_fuzzy_axions_h11_4.py` `--output` JSON (`_legacy_reproduce`, stamped 3551) | reproduction JSON incl. `model_stage` (3647) | F5 | `reproduction-1.2` (139) → `1.3` | D, E | physical |
| same, nested `kaehler_production_search` (stamped 1422, 1797, 3804) | inside the JSON | F5 | `kaehler-production-1.0` (1049) → `1.1` | E | physical |
| same, `_write_model_stage_input` (2829) | model-stage input HDF5 (transient, `TemporaryDirectory` 2997–3001) | F5 | `kaehler-point-provenance-1` (2844) → `-2` | E (production-mode `tau`, `cy_volume`, `inverse_metric`) | physical |
| `build_orientifold_vacua_inflation.jl:147` `write_inflation_group!` | `inflation/*` | F6 | `2.3` (:51) → `2.4` | B′ | physical |
| `inflation_scan_prep.jl` `_scan_prep_write_header`/`_scan_prep_append` (250/305) | scan-prep summary CSV | diagnostic + contract | diagnostic `1` → `2`; contract `4` → `5` | B′ | physical |
| `inflation_scan_shards_common.jl` `inflation_prepare_shard`/`inflation_append_shard_row` (61/83) | shard CSVs | shard | `2` (:11) → `3` | B′ | physical |
| `inflation_scan_shards_common.jl:183` `inflation_merge_shards` (driver `inflation_scan_merge_shards.jl:55`) | merged CSV | shard (derived, combiner) | `2` → `3` | from shards | physical (not named in spec) |
| `inflation_scan_pilot.jl:330` `_inflation_pilot_write_report` | pilot report CSV | other CSV (derived) | none; label column | from shards | physical (not named in spec) |
| `inflation_candidate_refinement.jl:61` `_write_csv` (549, 551) | refinement CSV, flow CSV | inflation CSV | refinement `3` → `4`; flow `1` → `2` | B′ | physical |
| `inflation_scale_continuation.jl` `pilot_prepare_csv`/`pilot_append_csv` (1445/1465) | report and shard CSV | pilot | `6` (:25) → `7` | B′ (`cholesky(scaled.K)`, 1731) | physical |
| `inflation_diagnostics_common.jl:108` `inflation_append_diagnostic_row` | diagnostic row | diagnostic | `1` → `2` | per row | generic (test caller only) |
| `analyze_inflation_candidates.jl:179` `write_namedtuple_csv` | `candidate_summary.csv`, `candidate_critical_points.csv` | other CSV | none; label column | B′ | physical |
| `compare_reduction_inflation_scan.jl:147` `write_csv` | `scan_rows.csv`, `scan_rows.checkpoint.csv`, `geometry_summary.csv` | other CSV | none; label column | B′ | physical |
| `reproduce_appendix_b_spectra.jl` `_write_summary` (148) | `appendix_b_quantiles.csv` | derived from F1 | none; label column | from F1 | physical |
| `inflation_reproduction.jl` `write_fixture`/`write_comparison` (113/136) | `.jls` fixture, comparison CSV | reproduction | none; label as a dictionary key / column | `N8_K_RAW`, `N5_K_RAW` | A96 |
| `author_poly102_reference.py:242` | reference JSON | reproduction | none; top-level key | `K_RAW` | A96 |

Label form for non-HDF5, non-CSV reproduction outputs: a `:kinetic_normalization`
key in the serialized dictionary (`.jls`) and a top-level JSON key. Figures
(PDF) are not data outputs; their source CSV or HDF5 carries the label.

**Notebooks.** `notebooks/orientifold_axiverse_statistics.jl` writes only PDF
figures (531–907) from F1/F6 inputs; it is a reader (below), not a data writer.

**Layout migrations (R-007 exception).** `migrate_legacy_spectra.jl`
(`migrate_file!` 15–33; copies `cytools/spectrum/physical` to
`spectrum/physical`, skips files with `spectrum/physical/m`) and
`migrate_quartic_index_ordering.jl` (`_quartic_rewrite!` 376–387; F2 quartic
index labels only). They add no label or version. They refuse a file whose
target group carries a current-version label (T067).

**Examined and excluded (not K-dependent).** `vacua_save`/`vacua_save_TB`
(generate.jl:4121/4147, L and Q only); `vacua_estimate_save` (4803); minimizers
(`minimizer.jl` 380/408, `jlm_minimizer.jl` 125–211, `jlm_reduced.jl` 406–418);
`scan_relaxed_qcd_candidates.jl` (CSV from Q and L only, 208–231; spec
confirmed); `validate_vacua_stage4_5.jl` stage 4 (Q, L only, 157–190, write
345–353) and stage 5 (`stage5_resource.csv` stores only the mode count, set by
`LQtilde(Q, L)`); `scripts/vacua_pipeline.jl` (K validated for shape only,
70–81); certificates (`certificates.jl:1338`, caller-supplied K); geometry
generators that write the stored `Kinv` and `L` (unchanged by the spec);
stdout-only scripts (`benchmark_*`, `inflation_scan_contract.jl`,
`reproduce_2023_n8.jl`, `audit_issue_148_*`, `issue_148_*`); `ext/` and `bin/`.

### Reader and combiner inventory (R-007)

| Reader / combiner (file:line) | Family | Version check today | Combines | Swallows errors | Planned duty |
|---|---|---|---|---|---|
| `read.jl:614` `physical_spectrum` | F1 | none (returns `schema_version`) | no | no | inference (`-3`, unversioned complete layout); rejections; label fields |
| `read.jl:599` `pq_spectrum`, `:655` `hp_spectrum` | F2 | none | no | no | F2 inference; rejections; label fields |
| `read.jl:487` `cubic_tensor` | F2 | none | no | no | as above plus `cubic_basis` |
| `read.jl:448` `read.K` | unlabelled | none | no | no | documented as unlabelled, not physical |
| `axion_photon.jl:1491` `_read_axion_photon_identity` | F3 | `== 1` (1514) | no | no | identity 1 → legacy; 2 → label required; keep stored version |
| `axion_photon.jl:1580` `_validate_axion_photon_group` | F3 | via 1491 | no | yes, `catch → false` (1695–1697) | let R-007 errors propagate |
| `axion_photon.jl:1898` `read_axion_photon_result` | F3 | via above | no | via above | rejections; `stored_kinetic_normalization` |
| `axion_photon.jl:2016`, `:2049` `_has_axion_photon` (file, path) | F3 | via above | no | yes (2039–2041; 2051–2054, 2071–2074) | let R-007 errors propagate |
| `axion_photon.jl:1337` `write_scan_csv` | F3 results | none | yes, across results | no | combination rule; label and `_inferred` columns |
| `batch_physical_spectrum.jl:190` `_has_physical_spectrum` | F1 | digest only (176–183) | no | no | batch resume |
| `reproduce_appendix_b_spectra.jl:82` `_read_values` → `_write_summary` | F1 | none (direct HDF5) | yes, across files | yes, `try … @warn` (83–101) | propagate; combination; label |
| `notebooks/orientifold_axiverse_statistics.jl:176` `read_geometry_row` | F1, F6 | F1 `== -3` (190–192); F6 `== 2.0` (196–198) | yes, pooled | yes, `catch → excluded` (247–249) | accept current and inferred legacy; propagate; combination |
| `build_orientifold_vacua_inflation.jl:100` `_has_inflation_group`, `:131` `_inflation_write_blocked` | F6 | presence; digest includes the group version | no | no | batch resume; blocked message names label |
| `inflation_scan_prep.jl:267` `_scan_prep_completed` | diagnostic + contract | contract `== "4"` (283) | across rows | yes (285, 298–299) | resume; per-row label; propagate |
| `inflation_scan_prep.jl:250` header check; `inflation_scan_shards_common.jl:51` header check; `inflation_scale_continuation.jl:1445` header check | append targets | exact header | — | no | append rule (fails closed already) |
| `inflation_scan_shards_common.jl:139` `inflation_completed_shard_geometries` | shard | header, contract (151, 159) | across files | yes (161, 173–174) | resume; combination; propagate |
| `inflation_scan_shards_common.jl:183` `inflation_merge_shards` | shard | header (198) | across files | no (rethrows, 212) | combination |
| `inflation_scan_pilot.jl:224` `_inflation_pilot_terminal_rows` → report | shard | header (227) | across files, rows | yes (217–220, 235) | combination; derived writer; propagate |
| `inflation_scale_continuation.jl:1474` `_pilot_completed_scales` | pilot | none | across rows | yes (1488–1494) | OD-5 |
| `reproduce_fuzzy_axions_h11_4.py:2894` `_run_model_stage` (reads 3019–3044) | F4 | none | no | no | F4 current/legacy checks |
| `fuzzy_axion_model_stage_driver.jl:45` `main` | F5 model-stage input | none | no | no | F5 checks on `record_metadata_schema` |
| `analyze_fuzzy_axions_orientifold_gap.py:651`, `:971`, `:1091` | F5 JSON | `== 1.1` (56, 656, 997) | yes, across shards (971) | no | OD-2 |
| `trace_lost_h11_3_fixed_components.py:208` | F5 JSON | none | no | no | F5 checks (OD-2) |
| `merge_orientifold_shards.py:225` → `merge_shard_summaries` (70) | F5 JSON → merge JSON (`shard-merge-1.0`) | none | yes, across files | no | combination; derived writer (sums `kaehler_point_export_accepted_count`, evaluator-dependent in production mode); version `shard-merge-1.1` |
| `orientifold_population_preflight.py:314` | merge JSON | none | no | no | accept only labelled merge output |
| `migrate_legacy_spectra.jl:15`, `migrate_quartic_index_ordering.jl:308` | F1 legacy, F2 quartics | none | no | per-file failure count | layout migration (T067) |

No Python reader of F1, F2, F3 or F6 exists. No reader exists for the F1
summary CSV, the F3 CSV, `appendix_b_quantiles.csv`, the refinement and flow
CSVs, or the `analyze_inflation_candidates.jl`/`compare_reduction_inflation_scan.jl`
CSVs.

**Mixed legacy files.** Both persisted legacy inflation layouts (scan-prep
summary, shard CSV) are written only by physical functions
(`run_geometry`, inflation_scan_common.jl:329). The A96 candidate-diagnostic row
(`inflation_refinement_diagnostic_row`, `inflation_append_diagnostic_row`) is
persisted only by a test (runtests.jl:3463–3483). Physical readers therefore
infer these legacy layouts; no mixed legacy layout exists.

### Batch-resume runners (R-007)

| Runner | Change |
|---|---|
| `batch_physical_spectrum.jl:359` `run_batch` | Legacy (`-3`, unversioned) or non-`sugra_c1` output is not complete; without `--force` record `"failed"` in the summary CSV with the stored label; with `--force` delete the group, then rewrite. Status stays the CSV string `"failed"`. The existence check (376) moves inside the per-geometry `try` (382–410). |
| `axion_photon.jl:2078` `run_batch_axion_photon` | Existence check and configuration read move inside the per-geometry `try`; an R-007 rejection or legacy label records `:failed` with the message and the batch continues (approval note 4). |
| `build_orientifold_vacua_inflation.jl:409` `run_pipeline2` | Keeps `:blocked` (447–460); `vacua_error` names the stored F6 label. Add `--force` to the CLI (539), which the docstring (28) already describes. |
| `inflation_scan_prep.jl:332` `run_scan_prep` (summary and shard modes) | Legacy rows are not complete; `--append-summary` on a legacy summary refuses; remedy is a new path. |
| `inflation_scale_continuation.jl:2023` `run_scale_continuation` | OD-5. |

### R-008 search record

Command at `10aa3dc` (matching lines per directory):

```
for t in M96 N5_K_RAW N8_K_RAW K_RAW reduced_models n8_kinetic_matrix n5_kinetic_matrix n5_potential n8_geometry author_inflation poly102_inflation N8_BEST_X EXPECTED_K_EIGENVALUES; do for d in src ext scripts bin test specs docs validation; do git grep -n -F "$t" 10aa3dc -- $d | wc -l; done; done
```

| term | src | ext | scripts | bin | test | specs | docs | validation |
|---|---|---|---|---|---|---|---|---|
| M96 | 3 | 0 | 4 | 0 | 0 | 20 | 49 | 0 |
| N5_K_RAW | 4 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| N8_K_RAW | 6 | 0 | 2 | 0 | 1 | 0 | 1 | 0 |
| K_RAW | 10 | 0 | 7 | 0 | 1 | 0 | 1 | 0 |
| reduced_models | 1 | 0 | 0 | 0 | 0 | 4 | 11 | 9 |
| n8_kinetic_matrix | 12 | 0 | 2 | 0 | 5 | 0 | 3 | 0 |
| n5_kinetic_matrix | 3 | 0 | 1 | 0 | 1 | 0 | 0 | 2 |
| n5_potential | 3 | 0 | 4 | 0 | 2 | 0 | 1 | 7 |
| n8_geometry | 9 | 0 | 8 | 0 | 3 | 0 | 4 | 2 |
| author_inflation | 53 | 0 | 2 | 0 | 2 | 0 | 6 | 8 |
| poly102_inflation | 2 | 0 | 8 | 0 | 2 | 7 | 42 | 6 |
| N8_BEST_X | 15 | 0 | 2 | 0 | 5 | 0 | 1 | 1 |
| EXPECTED_K_EIGENVALUES | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

Classification (`[S]` struct, `[NT]` `NamedTuple`, `[A]` bare array or scalar):

- **A96, `author_inflation` (`poly102_inflation.jl`).** `n5_geometry` [NT] 152,
  `n5_kinetic_matrix` [A] 161, `n5_light_direction` [NT] 164, `n8_geometry` [NT]
  187, `n8_kinetic_matrix` [A] 197, `n8_full_potential` [NT] 624,
  `n8_coordinate_maps` [NT] 676, `n8_degenerate_point` [NT] 700,
  `n8_mass_eigenbasis` [NT] 755, `n8_unstable_direction` [NT] 781,
  `n8_inflation_initial_condition` [NT] 805, `n8_hilltop_probe` [NT] 904,
  `n8_slow_roll_trajectory` [NT] 970, `n8_author_trajectory` [NT] 1111 (alias
  `n8_physical_gradient_flow` 1321). Compatibility aliases (compatibility.jl:1–37).
- **P96, `reduced_models.jl`.** `n8_geometry` [NT] 122, `n8_kinetic_matrix` [A]
  283, `n8_hilltop` [NT] 286, `n8_inflation_initial_condition` [NT] 331,
  `n8_local_hilltop_coefficients` [NT] 417, `n8_gradient_flow` [NT] 523,
  `n8_valley_flow` [NT] (via `_n8_valley_setup` 601), `n8_stiff_gradient_flow`
  [NT] 735, `n8_full_potential` [NT] 76; compatibility.jl `n8_coordinate_maps`
  47, `n8_mass_eigenbasis` 53, `n8_unstable_direction` 75, `n8_basis_directions`
  96 [NT].
- **P96, `n8_continuation.jl`.** `n8_bigfloat_p96_diagnostic` [NT] 823,
  `n8_continuation_classify` [NT] 853, `n8_pseudo_arclength_continuation` →
  `N8ContinuationResult` [S].
- **P96, `n8_g3_control.jl`** (no search-term hit; spec 0148 contract,
  `metric_contract = :P96_CYTools`). `n8_g3_predictor_corrector` →
  `G3ContinuationResult` [S] (already has `metric_contract`);
  `n8_g3_seed_from_radial`, `n8_g3_bigfloat_augmented_solve`,
  `n8_g3_local_diagnostics` [NT].
- **A96, `catastrophe_diagnostics.jl`.** `benchmark_manifest` [NT] 38,
  `n5_catastrophe_diagnostic` [NT] 260, `n8_catastrophe_diagnostic` [NT] 275.
  `local_catastrophe_diagnostic` (220/247) is a K-argument API: physical.
- **Scripts.** A96: `author_poly102_reference.py`, `benchmark_inflation_scalability.jl`,
  `inflation_refinement_common.jl` (all four functions), `inflation_reproduction.jl`
  (`n8_fixture` 15, `n5_fixture` 82), `audit_issue_148_g0.jl`. P96:
  `audit_issue_148_g3_controls.jl`, `issue_148_g2_continuation_evidence.jl` (with a
  labelled A96 comparison), `generate_appendix_c_geometry.py` (validation; keeps
  `inv(kinv)`). Both: `audit_issue_148_n8_metric_boundary.jl` (prints only),
  `pilot_benchmark_regression` (OD-7).
- **K-free reproduction (OD-4).** `n5_potential`, `n8_potential`,
  `n5_reduced_critical_points`, `n5_reduced_zero_phase_continuation`
  (`N5ReducedZeroPhaseContinuationStep` [S]), `n5_critical_scale`,
  `n5_reduced_ratio`, `instanton_scales`, `n8_minima_scan`;
  `validate_vacua_stage4_5.jl` `_stage4_n5_n8` (report lines 342–343);
  `catastrophic_inflation_population_study.jl` `main`;
  `reproduce_2023_n8.jl`; `issue_148_g1_n5_regression_tests.jl`;
  `issue_148_g1_replay_checks.jl`.
- **test/, docs/, validation/, specs/.** runtests.jl testsets 3091 and 3383
  (A96 and P96 reproduction), 1636 (P96 Eq. 96 check), 298 and 1359 (K-free
  `n5_potential` as physical-test input), 3565 (A96 K-free); docs `api.md`
  autodocs and the `issue_148_*` pages (spec 0148 evidence); validation
  `p0_numerical_equivalence` replay scripts and inventories (reproduction
  routes); specs 0148 (defines P96/A96) and 0125 (file-name references only).

Reproduction functions not named in the spec: `n8_g3_control.jl` (P96),
`catastrophe_diagnostics.jl` (A96), and the K-free set (OD-4).

## Fixtures

### R-004 pins

- `Kinv = [3 1; 1 2]`; `Q = [1 1 0; 0 1 1]` (columns q₁ = (1,0), q₂ = (1,1) and a
  subleading q₃ = (0,1), which `LQtilde` needs because it asserts N > h11,
  generate.jl:3155).
- Model stage (`leading_axion_reference_data(Q, tau, cy_volume, prefactor_P,
  gravitino_mass, inverse_metric)`, fuzzy_axions_model_stage.jl:76):
  `tau = [1.0, 20.0, 30.0]`, `cy_volume = 50.0`, `g_s = 0.5029733`, `W_0 = 1`.
  Expected: selected columns [1, 2]; Λ₂⁴/Λ₁⁴ = 2.85e-51; G = [1.5 2; 2 3.5];
  Gram–Schmidt norms (1.224744871, 0.912870929); log10 m/eV =
  (24.258982913, −1.141289065) at c = 1 and (24.409497911, −0.990774067) at
  c = ½ (difference ½·log10 2). Values from a replica of the source functions;
  T011 recomputes them in the package.
- `leading_hierarchy`: `InstantonData(Q, log10_lambda4 from the model stage,
  ones(Int, 3), [1, 2, 3])` with `kinv = [3 1; 1 2]`; asserts f₁, f₂, mᵢ and
  Eq. 2.15. Orientation guard: `BQ` gives norms (1.29099, 0.86603).

### R-005 row map

| Rows | Test route | Extension needed |
|---|---|---|
| Spectrum f, m, quartics | HDF5 fixture with the R-004 pins through `run_batch` (F1) and `hp_spectrum_save`/`pq_spectrum_save` (F2); template runtests.jl:2469–2500 | new fixture file; F1 runner has no test today |
| Cubic, both bases | same, with ϑ₀ = [0.1, 0.05]; both writers take ϑ₀ and `pq_spectrum_save` transforms it to x₀ = basisᵀC⁻¹ϑ₀ internally (A1.2); plus an h11 = 3 fixture for `pq_spectrum_save` (ϑ₀ = [0.1, 0.05, 0.02], non-degenerate leading masses) and an `:hp_effective` case with N > h11; each tensor checked against central differences of the Hessian of the writer's own `(L, Q)` potential, step 1e-5, tolerance 1e-6·max\|tensor\| (A1.4) | A1 |
| F3 (all rows, both policies) | test/axion_photon.jl templates (11–87, 171–221, 223, 314) with `Kinv = [3 1; 1 2]` and the R-004 Q/L; modes at 10^24.26 eV and 10^−1.14 eV lie either side of `ELECTRON_MASS_EV`; `qed_charge = [1, 1]` (norm 7 vs 3.5), `em_charge`, `qed_log10_lambda4` | fixture metadata |
| Inflation ε, η, eig(K⁻¹H), mass_min/max | runtests.jl:645–704 and full.jl:784–840 with `Kinv = [3 1; 1 2]`; non-stationary evaluation points for `diagnose` | evaluation points |
| F4, F5 masses | runtests.jl:1692 and 2017; new direct Python test of `_fuzzy_axion_mass_log10` with the R-004 pins (its superpotential adds Σe^{−2πτ}, line 1107) | Python test |
| F6 | full.jl:784–840 through `write_inflation_group!` | as inflation |

### R-010 audit

`test/fixtures/scientific_certification` (`backend_probe.jl`, `oracles.jl`,
`public.toml`, `run_fixture.jl`) supplies every K as a literal (T01–T17;
`run_fixture.jl:63, 74`). No `Kinv`-derived quantity. No stop.

### R-011 site

`_pilot_physical_domain_certificate_big` (inflation_scale_continuation.jl:496)
checks `base_K*kinv − I` (730). It becomes `base_K*kinv − 2I`. Test geometry:
runtests.jl:131–146 with `base_K = 2I` (passes) and `base_K = I`
(`:domain_failure`).

## Tests expected to change

| Test | Change | Reason |
|---|---|---|
| runtests.jl:2495 `C*C' ≈ Kinv` | `≈ Kinv/2` | R-001 |
| runtests.jl:1723 `Kmetric = inv(Kinv)` | `kinetic_matrix(Kinv)` | R-001 |
| runtests.jl:780 testset comment "K = 1/2 g" | K = g | R-009 |
| runtests.jl:154 `base_K = I` | `2I` | R-011 |
| runtests.jl:645–704, full.jl:784 (stored `Kinv = I`) | expected values follow R-005 factors where asserted | R-005 |

G1 lists every other changed test with its reason.

## Baseline (R-002)

To be recorded by T002 before any code change: base commit, Julia patch
release, OS image, BLAS, artifact paths and sha256.

## Proposed approach

1. Baseline (T002), then the core switch (T010): every K-taking API inherits
   c = 1.
2. Labels and versions, family by family (F1 → F6 → CSVs), with reader rules and
   round-trip tests.
3. Batch-resume and error-propagation changes.
4. Reproduction labels.
5. Documentation, evidence bundle, G2.

## Verification strategy

- Fast: R-003, R-004, R-005 analytic and source fixtures.
- Mid: per-family round trips and the R-007 matrix.
- Full: suite, docs, Python regressions, R-002 bitwise job.
- Review: G2 fresh-context review at the exact SHA (CI green is not sign-off).

Environment note: the local depot is not instantiated (`julia --project=.`
cannot load the package); T001 instantiates it before T002.

## Migration / compatibility

No regeneration. Legacy outputs read as `:legacy_c_half` (inferred) or are
rejected where the spec says so. PR #219's reader follows separately.

## Risk and stop conditions

Stop for the owner if:
- an OD above is answered differently from the recommendation in a way that
  changes other tasks;
- R-002 is not bitwise equal;
- a new inventory hit falls outside the spec's rules;
- the F5 evaluator dependence of another output is unclear;
- implementation would need a normative change.

## Task decomposition

T001–T002 gate everything. T010 precedes all label work. Family tasks
(T030–T045) are independent of one another after T010. OD-held tasks wait for
the approval of Clarification A; nothing else does.
