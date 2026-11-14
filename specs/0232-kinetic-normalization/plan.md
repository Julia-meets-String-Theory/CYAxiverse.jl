# Implementation Plan — CYAX-0232

## Governing specification

Canonical specification: `specs/0232-kinetic-normalization/spec.md`

Spec revision approved for this plan: `f097536` (blob `0a454b3`), approved on
Issue #232 (`approval_ref` in `spec.md`). The approval recorded seven G1 notes;
this plan resolves each (see *Approval notes*).

Implementation base: `origin/vmm` at `10aa3dc`. All file:line citations below
refer to `10aa3dc`. If the implementation branch is based on a later `vmm`,
the inventories are re-run and the R-002 baseline is regenerated (spec R-002).

## Owner decisions on the points the spec did not decide

The inventory found seven points that the approved spec does not decide
(OD-1..OD-7). The owner approved their resolution as `clarification-a.md`
(A1–A7) at `1ada61d` on Issue #232 (comment `6835402073`), after five focused G0
rounds. The tasks they held (T030, T041, T060 for F5, T061, T063, T064, T066,
T072) are released.

| ID | Point | Approved decision |
|---|---|---|
| OD-1 | `generate.cubic` argument orientation in both F2 writers; `pq_spectrum_save` phase coordinates | A1: N×2 `L`, N×n `Q`; x₀ = basisᵀ·C⁻¹·ϑ₀; ϑ₀ stored; fixtures C2, C2-hp, C3 |
| OD-2 | F5 `1.0`/`1.1` legacy inference | A2: infer files matching a recorded layout from merged commits (normalized path sets, R ⊆ paths ⊆ R ∪ O, at least one match); file-provenance meaning of the F5 label |
| OD-3 | Historical inflation CSV versions | A3: recorded file-layout entries (header, `row_type`, version values); no column priority; ambiguous or unknown rows rejected; physical writers only |
| OD-4 | Reproduction functions with no kinetic matrix | A4: no label duty; listed as K-free |
| OD-5 | `run_scale_continuation` resume | A5: scan-prep rule; label check before the header check; all three silent skips raise |
| OD-6 | `scan_fresh_relaxed_qcd_candidates.py` (untracked dependency) | A6: out of scope, exempt from R-006, outputs not certified |
| OD-7 | `pilot_benchmark_regression` mixes A96 and P96 | A7: `kinetic_normalization_by_field` by provenance; physical consumers reject it |

**G1 obligations from the round-5 nits** (approval comment):
1. A2 K-dependent fields are the values the Julia driver computes under
   `model_stage` and fields derived from them (for example
   `claim_status.model_count`, `aee4c1a:786–795`). K-free contents are the
   population counts under `counts`, the other `claim_status` entries,
   geometry and the stored `inverse_metric`. T041 and T080 use this wording.
2. A2 normalization: only object keys create paths; each array nesting level
   adds one `[]`; element keys of an array that can be empty go in O, with the
   line that allows the empty case cited (T041).
3. A1.4 scale spread is tested in log space,
   `maximum(log10 column) − minimum(log10 column) ≤ 1.5` (T030).
4. A3.3: the test-only A96 diagnostic row is built by
   `inflation_refinement_diagnostic_row` (`inflation_diagnostics_common.jl:39`)
   and appended by `inflation_append_diagnostic_row` (`:108`).

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
| R-006 | Labels and versions per the writer inventory; `KINETIC_NORMALIZATION`; reader fields; `stored_kinetic_normalization`. | T030–T041 round trips. |
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
| `scan_fresh_relaxed_qcd_candidates.py` | candidate CSV and JSON | not certified (A6) | unchanged; no label | `kinv` passed to an untracked module | out of scope: CYAX-0232 does not certify its outputs under either convention |

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

**Inventory completeness check (T040, at `d7db30e`).** Search: every tracked
`.jl` or `.py` file under `src/`, `ext/`, `scripts/`, `bin/` and `notebooks/`
that matches both a K-source pattern (`kinetic_factor`, `kinetic_matrix`,
`potential_factored`, `read.potential(`, `.potential(`, `oriented_potential`,
`_canonical_frame`, `leading_axion_reference_data`, `_fuzzy_axion_mass_log10`,
`[Kk]inv`, `inverse_metric`, `N8_K_RAW`, `N5_K_RAW`, `K_RAW`, `M96`,
`run_geometry(`, `scan_geometry_for_inflation`, `pq_*spectrum`, `hp_spectrum`,
`read_axion_photon_result`, `_run_local_scan`, `physical_spectrum`,
`prepare_geometry_context`) and a persistence pattern (HDF5 or h5py open for
writing, `open(…, "w"/"a")`, `json.dump`, `write_text(`, `to_csv`,
`serialize(`, `CSV.write`, `writedlm`, `_write_csv`, `write_csv`). It found 37
files. 23 are named in this plan, the spec or Clarification A. The other 14 are
not K-dependent writers:
- geometry generators that write the stored `Kinv`, `Q` and `L` (the excluded
  category above): `build_orientifold_axion_database.py`,
  `generate_geometric_data.py`, `generate_geometric_data_multitriangulation.py`
  (its `minimum_metric_eigenvalue` is an eigenvalue of the stored `Kinv`, not of
  K_kin), `generate_h11_491_frsts.py`, `generate_relaxed_qcd_sample.py`,
  `generate_fresh_relaxed_qcd_sample.py`;
- `read.potential` used for `Q` and `L` only: `geometries_hilbert.jl`,
  `hilbert.jl`, `hilbert_minima_compare.jl`, `glsm_v_qprime.jl`;
- no metric: `glimmers_eft_row_schema.py` (lists `Kinv` among fields that are
  never persisted) and `mpcp_bounded_analysis.py` (`Kinv` there names a Chow
  intersection term);
- tests: `test_reproduce_fuzzy_axions_h11_4.py`, `test_stage2_stage_boundary.py`.

A second pass, files that write and read a labelled family's data (the F1–F6
paths, the reproduction schema, the shard, contract and diagnostic version
columns, or mass, `fpert`, coupling, ε or η fields), found no file outside the
inventory. The R-002 and R-005 harnesses (`r002_invariance_artifacts.jl`,
`r005_scaling_harness.jl`) are this plan's verification tools (T002, T013):
their dumps compare conventions on purpose, must stay byte-identical across
checkouts, and no reader consumes them, so they carry no label.
`scan_fresh_relaxed_qcd_candidates.py` is recorded in the writer table as not
certified (A6).

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

**T070–T072 implementation.** Every [NT] above appends
`kinetic_normalization` at each return point: A96 (`A96_KINETIC_NORMALIZATION`
in `author_inflation`, aliased in `paper_benchmarks`) for the
`poly102_inflation.jl` set, its compatibility aliases, `benchmark_manifest`,
`n5_catastrophe_diagnostic` and `n8_catastrophe_diagnostic`; P96
(`P96_KINETIC_NORMALIZATION`) for the `reduced_models.jl` set (`n8_valley_flow`
on its result, not on the internal `_n8_valley_setup`), the compatibility.jl
P96 functions, `n8_continuation.jl` and `n8_g3_control.jl`. The public structs
`N8ContinuationResult` and `G3ContinuationResult` keep their fields and
positional constructors (R-006): `Base.getproperty` returns the P96 label for
`:kinetic_normalization`, `Base.propertynames` lists it, and the struct
docstrings state it. Nested intermediates (trajectory `samples`,
`n8_basis_directions().directions`, which callers iterate) are not labelled.
`inflation_reproduction.jl` `n5_fixture` and `n8_fixture` carry the A96 label,
and the serialized fixture has the key `:kinetic_normalization =>
"a96_reproduction_raw_radian"`; `n8_fixture` now takes its divisor volumes from
the A96 `n8_geometry()` instead of the P96 one (equal values, so that its only
contract is A96). `author_poly102_reference.py` writes the top-level JSON key.
`benchmark_inflation_scalability.jl` only prints a labelled
`n8_physical_gradient_flow` result. `n5_kinetic_matrix` and both
`n8_kinetic_matrix` docstrings state their contract.
`pilot_benchmark_regression` carries `kinetic_normalization_by_field` (A7) and
no top-level label; no physical consumer of it exists. Left unlabelled as
K-free (A4): `n5_potential`, `n8_potential`, `n5_reduced_critical_points`,
`n5_reduced_zero_phase_continuation`, `n5_critical_scale`, `n5_reduced_ratio`,
`instanton_scales`, `n8_minima_scan`, the K-free scripts above, and the
`local_catastrophe_diagnostic` K-argument API. Functions the record missed:
P96 `reduced_models.jl` `n8_hilltop_efolds` (alias `n8_local_hilltop_efolds`;
it uses `n8_local_hilltop_coefficients`), `n8_continuation_catastrophe_localization`
and `n8_find_regular_branches` (each element) in `n8_continuation.jl`, and the
`full=true` path of `reduced_models.jl` `n8_potential_derivatives` (through
`n8_full_potential`); A96 `poly102_inflation.jl` `n8_basis_directions` and
`n8_full_derivatives` (the `full=true` path of its `n8_potential_derivatives`).
K-free and unlabelled (A4): the default path of both `n8_potential_derivatives`,
`reduced_models.jl` `n8_degenerate_point`, `n8_bigfloat_continuation_refine`,
`n8_bigfloat_augmented_solve`, `n8_continuation_compare_matcher`,
`phase_fixture`, `cumulative_turning`, and the hard-coded normal-form functions
`n5_hilltop_efolds`, `poly102_inflation.n8_hilltop_efolds` and
`reference_efolds`. `trajectory_observables` takes a caller-supplied trajectory
and is not labelled. Open: the comparison CSV of `inflation_reproduction.jl`
(`write_comparison`) is not labelled, because its N8 rows hold the P96
`n8_local_hilltop_efolds` next to A96 values: one R-006 row label cannot
state both contracts, and A7 covers only in-memory results. This waits for an
owner decision.

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
| Cubic, both bases | dedicated fixtures in *Cubic fixtures (A1.4)* below, not the R-004 pins | A1 |
| F3 (all rows, both policies) | test/axion_photon.jl templates (11–87, 171–221, 223, 314) with `Kinv = [3 1; 1 2]` and the R-004 Q/L; modes at 10^24.26 eV and 10^−1.14 eV lie either side of `ELECTRON_MASS_EV`; `qed_charge = [1, 1]` (norm 7 vs 3.5), `em_charge`, `qed_log10_lambda4` | fixture metadata |
| Inflation ε, η, eig(K⁻¹H), mass_min/max | runtests.jl:645–704 and full.jl:784–840 with `Kinv = [3 1; 1 2]`; non-stationary evaluation points for `diagnose` | evaluation points |
| F4, F5 masses | runtests.jl:1692 and 2017; new direct Python test of `_fuzzy_axion_mass_log10` with the R-004 pins (its superpotential adds Σe^{−2πτ}, line 1107) | Python test |
| F6 | full.jl:784–840 through `write_inflation_group!` | as inflation |

**Harness (T013).** `scripts/r005_scaling_routes.jl` evaluates every route
above with the stored inverse metric `kinv_scale · Kinv`;
`scripts/r005_scaling_harness.jl` dumps the outputs bitwise (`dump`) or writes
the R-005 count report (`report`). Fixtures: `r004` and `r003` (spectrum, F3,
model stage) and `infl` (`Q = [1 0 1; 0 1 1]`, `L = (−1, −2, −3)`,
`Kinv = [3 1; 1 2]`; spectrum, inflation, F6). The R-004 fixture is not used
for inflation: its leading scales differ by 50 decades, and the Float64 eigen
route of `run_geometry` returns a roundoff light mass in both conventions.
`oriented_potential` rejects the R-003 charge layout.

T013 verification (macOS arm64, Julia 1.12, one machine): `dump` at the base
`10aa3dc` with `kinv_scale = 1` and at the candidate `9058e4f` + harness with
`kinv_scale = 2` are byte-identical (233 outputs, sha256
`bb53463d754e78d609d21044f37e018e4748d9d76a07b23d164aebcdad4eef51`). The
candidate dump with `kinv_scale = 1` differs in 136 of the 233 outputs; the
other 97 are K-independent. The one recorded route error is the A1 cubic
assertion of `hp_spectrum_save` on the R-003 charges, identical in both
conventions. T030 changes the F2 cubic routes (A1 orientation fix, under
which the R-003 route no longer fails) and adds the F2 `metadata` keys, so
base and candidate dumps are no longer comparable for those keys.

### Cubic fixtures (A1.4)

Every potential coefficient |Λₐ| (10 to the instanton's log10 column) lies
within 10^1.5 of the others. `L` columns are (sign, log10 Λ), all
signs +1. Leading masses were checked as non-degenerate with a replica of the
c = 1 leading Hessian `Cᵀ(Σ Λ q qᵀ)C`, `C = chol(Kinv/2)`.

| Fixture | `Kinv` | `Q` (columns are charges) | log10 Λ | ϑ₀ | Use |
|---|---|---|---|---|---|
| C2 | `[3 1; 1 2]` | `[1 1 0; 0 1 1]` | `(0, −0.5, −1.0)` | `[0.1, 0.05]` | both writers, h11 = 2; leading m² ratios differ by > 90% |
| C2-hp | `[3 1; 1 2]` | `[1 1 0; 0 1 1]` | `(0, −0.5, −0.7)` | `[0.1, 0.05]` | `hp_spectrum_save` `:hp_effective`: q₃ passes the `valid_cols` filter (−0.7 ≥ −0.5 + log10 0.5, generate.jl:3602–3609) and the keep test (−0.7 − (−0.5) = −0.2 > log10 0.5, :3621–3631), so `Canonicalα` and N = 3 > h11. C2 under `:hp_effective` drops q₃ (N = 2) |
| C3 | `[3 1 0; 1 2 0.5; 0 0.5 1.5]` | `[1 0 0 1; 0 1 0 1; 0 0 1 1]` | `(0, −0.4, −0.9, −1.5)` | `[0.1, 0.05, 0.02]` | `pq_spectrum_save`, h11 = 3; leading m² gaps > 70% |

The finite-difference check uses central differences of the Hessian of the
writer's own `(L, Q)` potential, step 1e-5, tolerance 1e-6·max|tensor|
(truncation ≈ 1e-11·|Q|², roundoff ≈ 2e-10 relative). The R-005 cubic
comparison is normwise (Frobenius) after per-mode sign alignment, relative
tolerance 1e-10. Non-degeneracy is asserted as (λ_{k+1} − λ_k)/λ_{k+1} ≥ 0.5 for
adjacent ascending leading Hessian eigenvalues. T030 asserts the `Canonicalα` branch
for C2-hp.

T030 measurements (macOS arm64, Julia 1.12): finite-difference deviation,
relative to max|tensor|, 1.5e-11 (C2 and C2-hp `hp_spectrum_save`, N = 2 and
N = 3), 4.2e-11 (C2 `pq_spectrum_save`) and 9.2e-11 (C3); normwise R-005
deviation from 2^{-3/2} after sign alignment 3.8e-16 (C2) and 3.2e-16 (C3);
minimum adjacent leading-eigenvalue gap 0.93 (C2) and 0.75 (C3).

### F1 and F3 labels (T031, T032)

**F1.** `read.f1_label` applies the R-007 F1 rules; `physical_spectrum` and
`stored_kinetic_normalization(path; family=:f1)` call it.
`batch_physical_spectrum.jl` takes its schema version from
`read.F1_SCHEMA_VERSION`. `_write_result_inplace` deletes the whole
`spectrum/physical` group before every write, so the forced rewrite and the
digest-mismatch rewrite of R-007 both start from an empty group. The summary
`kinetic_normalization` column is the last column. With `--append-summary`,
`_write_summary_header` refuses any existing header other than
`SUMMARY_HEADER`, not only one without the label column, because any other
header would misalign the appended columns.

**F3.** `read.f3_label` applies the R-007 F3 rules and returns the stored
identity version; `read.f3_identity_label` maps a version to its label for
in-memory results. `_read_axion_photon_identity` keeps the stored version.
`read_axion_photon_result` applies `f3_label` before the catch-all group
validation, so its rejections name the label problem. `write_scan_csv` appends
`kinetic_normalization` and `kinetic_normalization_inferred` and rejects mixed
labels (the T064 rule for this combiner).

**Left to later tasks.**
- T062: `_validate_axion_photon_group` and both `_has_axion_photon` methods
  still return `false` on an R-007 error. The notebook reader
  (`orientifold_axiverse_statistics.jl:190–192`) still requires F1 `-3`, so it
  rejects `-4` outputs until T062.
- T065: `_has_physical_spectrum` does not yet report a legacy output as
  `:failed`; a `-3` output has a different configuration digest and is
  recomputed, as before this change. `run_batch_axion_photon` still skips a
  version-1 result whose configuration matches.

### F4 and F5 labels (T033, T034)

**F4.** `fuzzy_axion_model_stage_driver.jl` writes root `schema_version` and
`kinetic_normalization`. Its work is now in `run_model_stage_driver`, and the
command line runs only when the file is the program (`PROGRAM_FILE`), so the
test suite can include it. Before computing, and again before opening the
output, it refuses an existing output that is not a current `"sugra_c1"` F4
file (legacy, unlabelled or not HDF5). It also refuses an F5 model-stage input
that is not `kaehler-point-provenance-2` with label `"sugra_c1"`: a primary
writer does not compute from legacy inputs (R-007). `read.f4_label` and
`read.f5_model_stage_input_label` hold the rules;
`stored_kinetic_normalization` accepts `:f4`.

**F5.** `REPRODUCTION_SCHEMA_VERSION` is `…-reproduction-1.3` with a top-level
`kinetic_normalization`. `KAEHLER_PRODUCTION_SCHEMA_VERSION` is
`…-kaehler-production-1.1`, and each of its three stamped objects carries the
label next to its `schema_version`. `_write_model_stage_input` writes
`record_metadata_schema = "kaehler-point-provenance-2"` and a root
`kinetic_normalization` dataset. `_run_model_stage` checks the F4 output with
`model_stage_output_label` (the Python F4 rules) and requires `"sugra_c1"`.
Python tests cover the summary, two of the three kaehler-production objects
(the optimizer result and the production search summary; the third is the
literal in `reproduce`), the input file, and the F4 rules. The bridge test
`test_fuzzy_axion_model_stage_driver.py` runs the Julia driver on a `-2` input.

**Left to later tasks.** `analyze_fuzzy_axions_orientifold_gap.py` still
requires `…-reproduction-1.1` (T061); `merge_orientifold_shards.py` has no
label yet (T037).

### F6 and inflation CSV labels (T035, T036)

**F6.** `write_inflation_group!` writes `inflation/kinetic_normalization`;
`INFLATION_GROUP_SCHEMA_VERSION` comes from `read.F6_SCHEMA_VERSION`.
`read.f6_label` reads only `inflation/schema_version`, so the nested
`catastrophes/legacy_v1/schema_version` marker (still `…-1.0`) is never taken
as the group version; `stored_kinetic_normalization` accepts `:f6`.

**Inflation CSVs.** Versions: diagnostic `2`, contract `5`, shard `3`,
refinement `4`, flow `2`, pilot `7`. The column `kinetic_normalization`
follows `schema_version` in the refinement, flow and pilot field lists and
`diagnostic_schema_version` in the scan-prep fields, which the shard fields
include. Physical rows take `INFLATION_PHYSICAL_KINETIC_NORMALIZATION` (from
`CYAxiverse.KINETIC_NORMALIZATION`). The A96 refinement summary in
`inflation_refinement_common.jl` carries `"a96_reproduction_raw_radian"`, and
`inflation_refinement_diagnostic_row` copies the summary's own label and
refuses a summary without one. `inflation_scan_contract.jl` prints its failure
rows with the label too (stdout only). `analyze_inflation_candidates.jl`
(both CSVs) and `compare_reduction_inflation_scan.jl` (`scan_rows.csv`, its
checkpoint and `geometry_summary.csv`) add the column last; they have no
version. The existing exact-header checks of `_scan_prep_write_header`,
`inflation_prepare_shard` and `pilot_prepare_csv` already refuse appending to
a legacy file.

**Left to later tasks.** T037: the merged shard CSV gets the column from its
inputs, but `inflation_merge_shards` has no mixed-label check, and the pilot
report has no label yet. T063: CSV reader inference from the T041 entries.
T065, T066: resume checks (scan-prep, shards, scale continuation). T062: the
notebook reader still requires F6 `2.0`.

### Derived-writer labels (T037)

- `reproduce_appendix_b_spectra.jl`: `_read_values` checks each F1 input with
  `read.f1_label` outside the existing unreadable-file skip, so R-007 errors
  propagate (this part of T062 is done here); inputs whose labels differ raise
  an `ArgumentError` naming both; no readable input is an error.
  `appendix_b_quantiles.csv` gains `kinetic_normalization` and
  `kinetic_normalization_inferred` columns.
- `inflation_merge_shards`: every row must carry the same non-empty label.
- Pilot report (`inflation_scan_pilot.jl`): terminal rows keep the shard label;
  `_inflation_pilot_report_rows` rejects mixed or empty labels; the report adds
  a last `kinetic_normalization` column. Its shard inputs pass the exact-header
  check, so they are never inferred, and no inferred column is needed.
- `merge_orientifold_shards.py`: `shard-merge-1.1` with
  `kinetic_normalization` and `kinetic_normalization_inferred`, taken from the
  inputs through `scripts/kinetic_normalization_labels.py` (standard library
  only). `…-reproduction-1.2` inputs are inferred `legacy_c_half`; `…-1.0` and
  `…-1.1` inputs are refused until T061 implements the A2 layout matching
  (sharding started at `f45ea31`, a 1.1 commit, so such merges exist).

### Historical layout records (T041)

The records are three appendices in `layouts/`, because they run to about
3,300 lines; this section states their scope, conventions and results.
- `layouts/f5-reproduction-1.0.md`: F5 `…-reproduction-1.0` (A2).
- `layouts/f5-reproduction-1.1.md`: F5 `…-reproduction-1.1` (A2).
- `layouts/inflation-csv-entries.md`: inflation CSV entries (A3).

Each record was built by reading the writer and every local module it
imports at each commit (`git cat-file -p <commit>:<path>`), with source
citations `commit:path:line` on each block. Commits that differ only in
imported modules were grouped by the blob ids of the writer's transitive
local-import closure and diffed against an analysed commit; each such group
either has its own layout group or a recorded reason why no path changes.

**F5 conventions (all F5 records).**
- (a) An array key `k` gives the path `k`, and its object elements give
  `k[].<key>`; no bare `k[]` path is listed. A map node at `m` gives
  `m.<map>…` paths for its values only; no bare `m.<map>`. An array of
  scalars adds no path below `k`; a list of lists normalizes its inner lists
  under a further `[]` (a list of lists of objects gives `k[][].<key>`; a
  matrix or ray list adds nothing). `scripts/f5_legacy_layout_api.py`
  `normalize` is the reference implementation, and the encoded layouts
  (`scripts/f5_legacy_layouts_v10a.py`, `_v10b.py`, `_v11.py`) are checked
  against it (T061).
- (b) In `--keep-details` layouts, the `details[]` element keys that every
  element carries are required. A file with `details: []` has the path set of
  the layout with `--keep-details` off and the same other flags, and is
  matched there. Keys below values that can be `null` or `[]` are optional.
- (c) `diagnostic.checks` (the Kähler-export diagnostic) is fixed fields, not
  a map node. `terminal_status_counts` and
  `schema_metadata.deprecated_aliases` (1.1) are map nodes.
- (d) An `--h11` value without a Table 1 row writes `paper_targets: null`
  and drops its children; from `239a3a7`, `population_preflight` is `null`
  outside h11 ∈ {4, 5}. These give separate layout families.
- Flags compose: each group records a base R and O and per-flag deltas and
  interactions, and each flag combination with a distinct path set is one
  layout, generated mechanically. Combinations no command line can produce
  (for example `--orientifold-reason-diagnostics` without
  `--orientifold-audit`, which every parser in the range rejects) are not
  recorded.

**F5 coverage and counts.**

| Version | Commits | Groups | Layouts per group |
|---|---|---|---|
| 1.0 | 9 that change the writer (`ec73c15`–`e7a8f51`), plus 9 import-only commits; all listed | 6 | 5, 8, 8, 16 (8 new; the other 8 equal group 3), 16, 36 |
| 1.1 | 57 (5 change the writer); all listed | 5 (G1, G2, G2b, G3, G4) | 26, 30, 30, 30, 45 (total 161) |

Within a version, some layouts overlap: a file can match more than one,
for example when no class was exported or every reason list is empty. A2
infers a file that matches at least one layout of its version, and all 1.0 and
1.1 layouts give `:legacy_c_half`, so overlaps do not change a result. Layouts of
different groups do not overlap except G2 and G2b (one optional path apart).

**F5 K-dependent fields (A2 label meaning, G1 note 1).** At every commit
with a model stage (from `aee4c1a`):
- computed by the Julia driver: `model_stage.total_model_count`,
  `model_stage.model_count_per_record` and every `model_stage.models[].*` field
  (the set of models, and so their indices, depends on the mass evaluator);
- derived from them: `claim_status.model_count`, and
  `model_stage.diagnostic_reason` when there is at least one input record.

`model_stage.convention.prefactor_P` comes from the driver but depends only
on `gs`; it is K-free. Everything else, including the stored
`inverse_metric`, is K-free. `ec73c15` has no K-dependent field; per A2 its
files are still inferred `:legacy_c_half`.

**Inflation CSV entries (A3).** 14 entries, every writer physical:
scan-prep summary INF-PREP-1 to 4 (contract `1`–`4`; diagnostic `1` from
PREP-3), shard INF-SHARD-1 and 2 (shard `1`, `2`; the merged CSV of
`inflation_merge_shards` has the shard header), refinement INF-REF-3 (`point`
and `summary` rows), flow INF-FLOW-1, and pilot report and shard pairs at
versions `3`, `4` and `6` (no merged `5`). One further merged writer was found:
`scripts/run_physical_scale_inflation_pilot_20260825.jl` (`b3719e3`, deleted
by `bd6807e`) writes the PILOT-6 layouts. The ten distinct headers differ.
Four pairs share a header: PREP-1 and PREP-2, told apart by `contract_version`,
and each pilot report and shard pair, told apart by `row_type`. Each pair
lists the same version columns (A3.2), so no row matches two entries. The A96
diagnostic row (`inflation_refinement_common.jl`) was persisted only by a test
(`runtests.jl:3463–3483`) and is not an entry (A3.3).

**Decisions on the records' open points.**
- JSON with `Infinity` or `NaN` (written by `json.dumps` defaults) is valid
  input for the T061 readers.
- A header-only inflation CSV has no rows to infer. Where two entries share
  its header, both give the same inferred label, so the file-level label is
  still `:legacy_c_half`.
- Rows from unmerged code (the deleted runner's copies of earlier files, and
  the 101-column pilot header that no merged commit defines) match no entry and
  are rejected (A3.3).
- PREP-1 and PREP-2 rows mixed in one file (by `--append-summary`) each match
  one entry; both are physical, so the file is inferred.

T041 has no escalation. Every F5 version has at least one recorded layout for
each file a merged writer can produce. Every inflation entry has physical
writers only, and no row can match two entries.

### Readers: inference matrix and F5 readers (T060, T061)

**T060.** `test/kinetic_normalization.jl` (*R-007 inference and rejection
matrix*) runs every R-007 case for F1–F4 and F6 on minimal synthetic files
through `stored_kinetic_normalization`, one `@test` per case: each inference,
and the current version with a missing, unknown, legacy or reproduction
label, a legacy or unversioned file with a label, a missing version, an
unknown version and an unrecognized layout (plus F2 `cubic_basis` and the F6
nested marker). F5 is covered by the Python tests below.

**T061, encoded layouts.** The T041 records are encoded in
`scripts/f5_legacy_layouts_v10a.py`, `_v10b.py` and `_v11.py` (250 layouts:
37, 52 and 161), built by composition from each record's named path sets.
`scripts/f5_legacy_layout_api.py` holds the A2 normalizer and a synthesizer;
each module's self-check confirms that a file holding exactly R, and one
holding R ∪ O, normalize back to those sets. `kinetic_normalization_labels.py`
matches a 1.0 or 1.1 file against the layouts of its version (R ⊆ paths ⊆
R ∪ O, normalized against each layout's map nodes) and infers
`legacy_c_half` on any match; otherwise it raises, naming the unrecorded
paths. `scripts/test_f5_legacy_layouts.py` has one subtest per layout (R-only
and R ∪ O files) and the rejections.

**T061, readers.**
- `analyze_fuzzy_axions_orientifold_gap.py`: keeps its 1.1-only scope;
  single and sharded artifacts pass the F5 checks, and shard labels are
  combined with the R-007 rule. Its outputs hold no K-dependent field, so they
  carry no label. Its shard test fixtures are now built on recorded layout
  `1.1-G3-L03` with the 1.1 provenance shape.
- `trace_lost_h11_3_fixed_components.py`: `_load_pre_summary` applies the F5
  checks; it reads only K-free candidate details.
- `orientifold_population_preflight.py`: applies `shard_merge_label` to the
  merged artifact. This departs from the plan's wording "accept only labelled
  merge output": the preflight reads only population counts, which are
  K-free, and the existing h11 = 4/5 merges are `shard-merge-1.0`, which R-007
  reads as an inferred legacy derived output. Requiring a label would block
  every h11 = 4/5 run on its existing evidence. The label is returned in the
  preflight result.
- `scripts/run_python_regression_tests.py` now also runs
  `test_f5_legacy_layouts.py` and `test_merge_orientifold_shards.py`.

### Error propagation and inflation CSV rows (T062, T063)

**T062.** R-007 label errors are no longer swallowed:
- `axion_photon.jl`: `_check_axion_photon_label` runs `f3_label` outside the
  catch-all of `_validate_axion_photon_group` and both `_has_axion_photon`
  methods. A legacy identity-1 group reads as present (so the writer refusal
  applies); an invalid label raises; other defects still read as absent or
  incomplete. `run_batch_axion_photon` calls `_has_axion_photon` outside its
  per-geometry `try`, so an invalid group currently aborts the batch; T065
  moves the check inside the `try` and reports `:failed` (approval note 4).
- `notebooks/orientifold_axiverse_statistics.jl` `read_geometry_row`: the F1
  and F6 label checks run before the exclusion handler; the stale exact
  version tests (F1 `-3`, F6 `2.0`) are replaced by the label rules, so
  current and inferred legacy outputs are read; each row carries both labels,
  and the pooled rows must share one label per family (T064 rule for this
  combiner).
- `reproduce_appendix_b_spectra.jl` was done in T037.
- Inflation resume readers: below.

**T063.** `scripts/inflation_csv_labels.jl` (included by
`inflation_scan_common.jl`) holds the 14 recorded legacy entries, generated
from the historical writer sources of the T041 record, and
`inflation_csv_row_label(header, fields; path, row)`: a labelled row must
carry the current version values and an allowed label (`"sugra_c1"`, or
`"a96_reproduction_raw_radian"` in the per-row-labelled diagnostic family); a
legacy row must match exactly one recorded entry (A3.3). `_scan_prep_completed`,
`inflation_completed_shard_geometries` and `_inflation_pilot_terminal_rows`
call it on every row outside their parse-error handling, so its errors
propagate, and only `"sugra_c1"` rows count as complete. The scale-continuation
reader `_pilot_completed_scales` is T066 (A5).

### Combination, resume and migrations (T064–T067)

**T064.** Every combiner in the reader inventory rejects mixed labels with an
`ArgumentError` naming both: `write_scan_csv` (T032), Appendix B,
`inflation_merge_shards`, the pilot report and `merge_orientifold_shards.py`
(T037), the gap analysis (T061), and the notebook's pooled rows, now
`check_pooled_labels` (tested by extracting it from the notebook). The
inflation resume readers do not combine values: after T063 their rows are
`sugra_c1` or filtered out.

**T065.** Batch resume, one test per runner with a legacy output with and
without the overwrite option:
- `batch_physical_spectrum.jl` `run_batch`: the existence and label checks
  run inside the per-geometry `try`; a legacy or non-`sugra_c1` output records
  `"failed"` with the stored label unless `--force`, which rewrites it;
  `_has_physical_spectrum` is false for such outputs.
- `run_batch_axion_photon`: existence, label and configuration checks inside
  the per-geometry `try`; a legacy result or an R-007 rejection records
  `:failed` and the batch continues (approval note 4); `force=true` recomputes.
- `build_orientifold_vacua_inflation.jl`: `_has_inflation_group` is false for
  a non-`sugra_c1` group; `run_pipeline2` keeps `:blocked` and its
  `vacua_error` names the stored F6 label; the CLI accepts `--force`.
- `inflation_scan_prep.jl`: legacy rows are not complete (T063), and
  `--append-summary` and shard append refuse a legacy header (existing exact
  header checks).

**T066 (A5).** `_pilot_completed_scales` raises on a missing required
column, a field-count mismatch and a parse error, checks every row's label,
and counts only `sugra_c1` rows. `_pilot_resume_label_check` runs on the
report and the shard file before either `pilot_prepare_csv` header check and
refuses `--resume` with the stored or inferred label (a header-only legacy
file is named by its header).

**T067.** `migrate_legacy_spectra.jl` (now guarded by `PROGRAM_FILE`) adds no
label or version, keeps a copied version, skips a legacy target and refuses a
current-version target; `migrate_quartic_index_ordering.jl` refuses a
current-version F2 group.

### Conventions page (T080)

`docs/src/kinetic_normalization.md` (manual page *Kinetic normalization*)
states the definition, units, factor and K-argument contracts, the
source-facts table with versions and the arXiv:2103.06812 numbering map, the
open inferences, the P96 and A96 contracts, the full R-005 scaling tables, the
label values, every family's label location and version history, the R-007
inference, rejection, combination, batch-resume and migration rules, the
unreliable legacy F2 cubic note (A1.5), the meaning of an inferred F5 label
(A2), and the non-certified `scan_fresh_relaxed_qcd_candidates.py` (A6). It is
pointed to from the README, the axion–photon page (whose `K = Kinv⁻¹` sentence
is corrected) and the docstrings of `KINETIC_NORMALIZATION`, `kinetic_factor`,
`kinetic_matrix`, `_kinetic_matrix`, `_validate_kinetic_matrix` and
`stored_kinetic_normalization`; the `_kinetic_matrix` "`K = (1/2) g`" text was
replaced in T010.

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
| full.jl:551 `schema_version == "…-3"` | `"…-4"`, plus the label fields | R-006 (T031) |
| full.jl:451 `inflation/schema_version == "…-2.3"` | `"…-2.4"`, plus the label | R-006 (T035) |
| runtests.jl:405 pilot report row | the appended `scale` row carries `schema_version` and `kinetic_normalization` | A5: unlabelled rows now raise (T066) |
| axion_photon.jl:38 one-axion canonical charge | `3/2` → `3·sqrt(0.25/2)` | R-001 (factor `sqrt(Kinv/2)`) |
| axion_photon.jl:83 QED threshold | `+ 0.5·log10(0.5)` | R-001 (QED norm `qᵀ(Kinv/2)q`) |
| axion_photon.jl:128 rank-certificate hand case | `q`, `theta_from_canonical` divided by √2 | R-001 |
| runtests.jl:739 `oriented.K` | `I` → `2I` | R-001 |
| full.jl:140 inflation pilot rows | rows carry `kinetic_normalization` | R-006 (T037) |
| runtests.jl:705–711, 3489–3505 | label and version assertions added; no expected value changed | R-006 (T036) |

G1 lists every other changed test with its reason.

## Baseline (R-002)

**Design.** `scripts/r002_invariance_artifacts.jl` writes a canonical text dump
of the R-002 quantities for seven pinned fixture geometries, built as HDF5
files in a temporary data directory:
- `r004_nondiagonal` and `c3_h11_3`: the R-004 and C3 inputs;
- `geometry_2_2_1` and `non_unimodular_qtilde`: the runtests.jl:2278 cases;
- `ill_conditioned_kinv`: the runtests.jl:2469 case;
- `merged_subleading_charge`: the C2-hp charges;
- `h11_3_det_6_relevant_subleading`: 6 vacua, 48 leading branches.

For each fixture it records:
- (a) `Q`, `L` and `Kinv` read back through `read.potential_factored` and
  `read.potential`;
- (b, c) `vacua_estimate`, `vacua_id`, and the `vacua_pipeline` datasets and
  count metadata for each method (`:legacy`, `:leading_branches`,
  `:reduced_jlm`, `:auto`), with seeds fixed and run-specific metadata
  excluded;
- (d) the ϑ-Hessian at each stored coordinate column.

Floats are written as IEEE-754 hex. Errors are recorded by type and first
line: `vacua_id` already throws a `BoundsError` at the base for the last two
fixtures, so the candidate must reproduce that error unchanged.

`:auto` selects the exact-determinant path for every fixture, so its
reduced-JLM non-square and legacy fallbacks are not exercised. Both read only
`(Q, L)`.

**Comparison.** `.github/workflows/R002Invariance.yml` runs the script at the
merge base with `vmm` and at the candidate in the same job (ubuntu-latest,
Julia 1.12 x64, `JULIA_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`). The job
summary records the Julia version, BLAS configuration and runner, and the job
fails on any difference. This satisfies the R-002 requirement that both runs
share the Julia patch release, OS image and BLAS.

**Local check at `10aa3dc`** (macOS aarch64, Julia 1.12.6; not the pinned
environment): two runs give identical hashes. Against CI, only
`ill_conditioned_kinv` differs, by one ulp in two ϑ-Hessian entries of order
1e-19 to 1e-18. This is why base and candidate run in the same job.

**Pinned hashes.** Base `10aa3dce4a39816846c4da34a295367ef0dcaf63`, workflow run
43305737956 (runner image ubuntu-26.04, image version 20261111.82.6). The
candidate `b149791` gave identical hashes.

| sha256 | Artifact |
|---|---|
| `8855cf5b82b413941c8ad884e4217e50d7f0df0df201d86394ed5cb8e02d9763` | `r004_nondiagonal` |
| `27dd39599f8b23d02b9a77287d1b328f3234fb3220d83e918b3d6b1b3c7a2eed` | `c3_h11_3` |
| `fcff3044b3ce4cfb2a656e69517ea0d27011bcbf8d0b67544d2da7fa758088a4` | `geometry_2_2_1` |
| `03894bbf861305f024d4f365f79d1c77afebbaf1b4d4f0d540e9a50a14dc740d` | `non_unimodular_qtilde` |
| `680e3fb56da050a1c9117dcd4069e22b914fe83b3ab47a8f8980f73b29097a18` | `ill_conditioned_kinv` |
| `5ae88afab45c8853ff4e341fbb790a285d4a0e498a89fcbafb6d1b138bcc5b1a` | `merged_subleading_charge` |
| `db5025462dcbf4d039cff704df1a2f2af81f4476407b3cec5b629c87eccbe58b` | `h11_3_det_6_relevant_subleading` |
| `1c878cd1531409e29cd585cc9ee097ed5e068b356196ac6b984ab34b4b9a59e4` | `r002_artifacts.txt` |

Environment (run 43307736586, which printed it to the job log; same hashes,
candidate `e0cc584`): Julia 1.12.7, BLAS `LBTConfig([ILP64]
libopenblas64_.so)`, Ubuntu 26.04.1 LTS, x64. T090 repeats the comparison at
the implementation SHA; if CI's Julia 1.12 patch release changes in the
meantime, the base is regenerated in the same job, as R-002 requires.

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

## Open owner decisions at G1

These points are not decided by the spec, Clarification A or this plan's
recorded decisions. Each is left as implemented and is not declared complete.

| # | Point | State at the candidate | Where recorded |
|---|---|---|---|
| G1-1 | `orientifold_population_preflight.py` accepts an inferred legacy `shard-merge-1.0` merge | accepted; the label is returned in the preflight result | *Readers* (T061) |
| G1-2 | R-002 and R-005 harness dumps | unlabelled verification tools | *Writer inventory*, second pass |
| G1-3 | `inflation_reproduction.jl` comparison CSV (`write_comparison`) mixes P96 and A96 rows | unlabelled | *R-008 search record* |
| G1-4 | Committed `validation/inflation_fixtures.jls`, `validation/inflation_comparison.csv`, `validation/author_poly102_reference_miniforge.json` | not regenerated, so they carry no label although their writers now write one (T071) | this table |
| G1-5 | `trajectory_observables` (caller-supplied trajectory) | unlabelled | *R-008 search record* |
| G1-6 | Hard-coded normal-form functions (`n5_hilltop_efolds`, `poly102_inflation.n8_hilltop_efolds`, `reference_efolds`) | unlabelled as K-free under A4, although their constants were derived under a kinetic convention | *R-008 search record* |
| G1-7 | `n8_author_trajectory` default tolerances | see below | this section |

**Owner direction (2026-11-14), approved as Clarification B
(`clarification-b.md`; Issue #232 comment `6871292308`):**
- G1-1, G1-2, G1-5: ratified as implemented.
- G1-3: extend A7 to the comparison CSV with per-field labels (Clarification B2, implemented).
- G1-4: regenerate the three `validation/` files on the owner's machine and
  check that only the label changes.
- G1-6: one docstring line per function naming its source contract (P96/A96);
  no API or result change.
- G1-7: a separate follow-up Issue; no change in this PR.
- The `infl` fixture substitution for the R-005 inflation and F6 rows is
  ratified (Clarification B).
- Before G2: add a test for the A5 shard-file refusal, assert the label columns of
  the two analysis scripts, and move the R-001 F5 evaluator equivalence
  test out of the CYTools-gated module.
- `impl/0232-kinetic-normalization` is fast-forwarded to the `-xaimz1` head.

**G1-7.** `n8_author_trajectory` (`poly102_inflation.jl:1166`) derives its
default tolerances from `precision_bits`: `reltol = 10^-(precision_bits ÷ 2)`,
`abstol = 10^-(2·precision_bits ÷ 3)` (line 1237). For every allowed
`precision_bits ≥ 64` this `reltol` is below the BigFloat working precision
`2^-precision_bits` (64 bits: 1e-32 against 5e-20; the default 100 bits:
1e-50 against 8e-31). The behaviour is unchanged from the base `10aa3dc`. Its
only caller in the repository is the T070 label test, which hung at 64 bits
with the defaults and now passes explicit `reltol`, `abstol` and `maxiters`
(`5b5c4c0`). That the default call does not converge is inferred from that
hang and is untested at the default 100 bits. Changing the defaults changes a
reproduction path's numerics, which the spec's non-scope excludes, so it is
left for the owner.

## G1 evidence (T090–T092)

Code candidate: `90f6f43`. Later commits on the branch change only
`specs/0232-kinetic-normalization/`. Base: `10aa3dc`.

| Check | Command or run | Result at `90f6f43` |
|---|---|---|
| R-002 bitwise (T090) | `R002Invariance.yml` run 43519251792 (push), base and candidate in one job | PASS: all 8 hashes equal between base and candidate and equal to the pinned hashes in *Baseline* |
| Full suite, including fast (`CYAXIVERSE_TEST_FULL=1`) | `CI.yml` run 43519257038 (workflow_dispatch), job *Full test suite* | PASS ("CYAxiverse tests passed"); one existing skip: Pipeline 2 real-data round trip (data not on the runner) |
| Certification (R-010) | same run, *Process-isolated scientific qualification* (watchdog self-test, fast and public suites) | PASS |
| Portable Python tests | same run, *Python tests*; and `python3 scripts/run_python_regression_tests.py` in a cloud container (41 tests, 1 module skipped for CYTools) | PASS |
| `agent_verify.py` | `snapshot`, `diff-check` (clean tree), `python-regressions`; `package` is covered by the CI full suite | PASS |
| CYTools Python tests | `python scripts/run_python_regression_tests.py test_reproduce_fuzzy_axions_h11_4.py test_fuzzy_axion_model_stage_driver.py test_trace_lost_h11_3_fixed_components.py test_build_orientifold_axion_database.py` in the owner's CYTools environment | PASS (owner-local, macOS, at `0a22a56`): 88 tests, 0 skipped, which equals the 41 + 4 + 2 + 41 test methods in those modules. The default set gave 41 passed, 1 skipped (CYTools module, run separately). The earlier record of "105" was a different module set |
| Docs build | `julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`, then `julia --project=docs/ docs/make.jl` | PASS (owner-local, macOS, at `0a22a56`), including `checkdocs = :exports`. Warnings: `edit_link` could not query the git remote (local environment); search index 540.76 KiB above the 500 KiB warning threshold (warning only; not compared with the base) |
| R-005 count report | `julia --startup-file=no --project=. scripts/r005_scaling_harness.jl report <out_dir>` | owner-local at `0a22a56`: `evidence/r005_report.md`. Every gated factor matches; every reported count is equal. Not covered by the harness: e-folds, F6 `n_candidates`/`n_qualified`/flow e-folds, F4 `model_lambda` and model counts, F5 optimizer outputs |

Changed tests: *Tests expected to change* (all rows, including the G1
additions). The R-005 rows at runtests.jl:645–704 and full.jl:784–840 did
not change: their assertions do not depend on K, and the full suite passes.

**Convergence (T092).** A requirement-by-requirement audit at `5b5c4c0`
found two gaps, both fixed in `90f6f43`: `stored_kinetic_normalization` was
not exported (spec *Interfaces*), and T050's `base_K = I` case had no test.
Remaining partial or unmet items, none declared complete:
- R-008 / T071: the committed `validation/` files carry no label until the
  B4 regeneration (G1-4). The `write_comparison` CSV is labelled per field
  under Clarification B2 (tested in the T071 testset).
- R-005: the inflation and F6 rows use the `infl` fixture instead of the
  R-004/R-003 pins (*R-005 row map*). This deviation was made in the plan,
  not by a spec amendment.
- R-005 F4: the persisted `model_mass_reference_log10_ev` is checked only
  through the in-memory route.
- No test: the A7 physical-consumer rejection (no physical consumer exists).
  Closed after `90f6f43` on owner direction: the A5 shard-file refusal
  (T066 testset), the label columns of `analyze_inflation_candidates.jl` and
  `compare_reduction_inflation_scan.jl` (*analysis-script label columns*
  testset), and the R-001 F5 evaluator equivalence test, now in
  `scripts/test_fuzzy_axion_mass_evaluator.py` on the CYTools-free
  `scripts/fuzzy_axion_mass_evaluator.py` (function moved verbatim and
  re-exported by `reproduce_fuzzy_axions_h11_4.py`), so it runs in CI.
- G1-6 docstrings added to `n8_hilltop_efolds`, `n5_hilltop_efolds` and
  `reference_efolds` (`poly102_inflation.jl`).
- R-005 "reported" rows missing from the count report: e-folds, F6
  `n_candidates`, `n_qualified`, flow e-folds, F4 `model_lambda` and model
  counts, F5 optimizer outputs.
- `run_pipeline2` catches the R-007 `ArgumentError` from `f6_note` into
  `vacua_error`; the geometry stays `:blocked`, so nothing is skipped silently.
- The *Left to later tasks* notes under T030–T037 record the state at those
  tasks; T037 and T061–T066 closed them.

## Task decomposition

T001–T002 gate everything. T010 precedes all label work. Family tasks
(T030–T041) are independent of one another after T010. T041 precedes T061
and T063.
