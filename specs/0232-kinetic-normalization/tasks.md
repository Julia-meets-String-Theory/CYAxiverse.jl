# Tasks — CYAX-0232

`tasks.md` records evidence readiness, not live PR, Issue or Project state.
File:line references are at `10aa3dc` (see `plan.md`). Tasks marked
**Held: OD-n** wait for that owner decision in `plan.md`; all others proceed.

## Phase 1 — Baseline

- [ ] **T001 [G1] Instantiate the environment**
  - Output: `julia --project=.` loads CYAxiverse on Julia 1.12; Python test
    environment imports the F5 script.
  - Verify: `Pkg.test()` fast suite passes at the base.
  - Escalate if: the base suite fails.

- [ ] **T002 [R-002] Generate and pin the bitwise baseline**
  - Input: base commit; the R-002 artifact list (vacua datasets, ϑ-Hessians).
  - Output: artifacts and their sha256, Julia patch, OS image and BLAS recorded
    in `plan.md` *Baseline*.
  - Verify: regeneration on the same environment reproduces the hashes.
  - Escalate if: the hashes are not reproducible at the base.

## Phase 2 — Core switch

- [ ] **T010 [R-001] Shared kinetic functions and routing**
  - Output: exported `kinetic_factor`, `kinetic_matrix`, `KINETIC_NORMALIZATION`;
    `potential_factored`, `_kinetic_matrix`, `_canonical_frame`, the QED norm,
    `leading_axion_reference_data`, `benchmark_hybrid_scaling.jl` routed through
    them; `metric_residual` from the same factor.
  - Verify: factor identities (`BBᵀ = Kinv/2`, `BᵀK B = I`); grep audit that no
    physical path factors or inverts `Kinv` directly.
  - Escalate if: a direct factorization outside the plan's list is found.

- [ ] **T011 [R-004, R-005] Analytic fixtures**
  - Output: non-diagonal fixture with the `plan.md` pins through
    `leading_hierarchy` (f, m, Eq. 2.15) and the model stage (m); orientation
    guard; new scalar fixture through both functions; "HP spectrum: one-axion
    analytic mass" unchanged.
  - Verify: assertions to 1e-10 relative; guard changes a value by > 1e-3.
  - Escalate if: the package values differ from the pinned replica values.

- [ ] **T012 [R-003] Source fixture**
  - Output: testset on runtests.jl:2017–2050 literals, QCD first.
  - Verify: c = 1 within 5% of arXiv:2412.12012 Eq. 4.3; c = ½ misses f by > 20%.
  - Escalate if: either gate fails.

- [ ] **T013 [R-005] Comparison harness**
  - Output: helper that evaluates each route with `2·Kinv` in place of `Kinv`.
  - Verify: harness reproduces current (c = ½) outputs bitwise on the base.

- [ ] **T014 [R-005] Spectrum and F4/F5 rows**
  - Output: tests for f, m, quartics (F1, F2), F4 and F5 masses, including a
    direct Python test of `_fuzzy_axion_mass_log10`.
  - Verify: factors per the spec table; threshold-filtered mode sets reported.

- [ ] **T015 [R-005] Axion–photon rows**
  - Output: F3 tests under both QED policies with the `plan.md` fixture metadata.
  - Verify: each F3 row; `light_mode_count` behaviour per policy.

- [ ] **T016 [R-005] Inflation and F6 rows**
  - Output: tests at non-stationary points and through `write_inflation_group!`.
  - Verify: ε, η, eig ½; mass_min/max −½log₁₀2.

- [ ] **T020 [R-001] K-argument docstrings**
  - Output: contract text on every K-taking API in the plan's list (approval
    note 1).
  - Verify: docs build; doctest-free review of the list.

- [ ] **T021 [R-001] Python evaluator equivalence**
  - Output: Python route uses the c = 1 factor.
  - Verify: Python and Julia masses agree on the R-004 pins.

## Phase 3 — Labels and schemas (R-006)

- [ ] **T030 [R-006, R-005] F2 writers and readers** — **Held: OD-1**
  - Output: `spectrum-groups-1`, `cubic_basis`, label; F2 replace keyword;
    readers report `cubic_basis` (`:unknown` for legacy).
  - Verify: round trip; F2 replace leaves F1 and F3 intact; cubic rows of R-005.

- [ ] **T031 [R-006] F1 writer, summary CSV and reader**
  - Output: `-4`, label, summary column; `physical_spectrum` label fields.
  - Verify: round trip; unversioned F1 without `fpert_log10` inference case.

- [ ] **T032 [R-006] F3 writer, CSV and reader**
  - Output: identity 2, label, CSV columns; `stored_kinetic_normalization`.
  - Verify: round trip; stored identity version kept.

- [ ] **T033 [R-006] F4 driver**
  - Output: `fuzzy-axion-model-stage-1`, label; refuses an existing unlabelled or
    legacy output file (it opens `"w"` today, line 119).
  - Verify: round trip; refusal test.

- [ ] **T034 [R-006] F5 outputs**
  - Output: `reproduction-1.3`, `kaehler-production-1.1`,
    `kaehler-point-provenance-2`, labels.
  - Verify: round trip of each versioned output listed in `plan.md`.

- [ ] **T035 [R-006] F6 group**
  - Output: `orientifold-inflation-2.4`, label.
  - Verify: round trip; nested `legacy_v1` marker not treated as group version.

- [ ] **T036 [R-006] Inflation CSV families and other inflation CSVs**
  - Output: each schema version +1; contract `"5"`; per-row labels; label
    columns on the derived and other CSVs in the writer inventory.
  - Verify: round trip per family; A96 refinement rows carry
    `"a96_reproduction_raw_radian"`.

- [ ] **T037 [R-006] Derived writers**
  - Output: labels on `appendix_b_quantiles.csv`, `inflation_merge_shards`,
    the pilot report and `merge_orientifold_shards.py` (`shard-merge-1.1`).
  - Verify: derived label equals the inputs' label; inferred inputs give
    `"legacy_c_half"` with the inferred flag.

- [ ] **T040 [R-006] Inventory completeness check**
  - Output: rerun the writer search at the implementation SHA; note
    `scan_fresh_relaxed_qcd_candidates.py` per OD-6.
  - Verify: no writer outside the inventory.
  - Escalate if: a new writer falls outside the spec's rules.

## Phase 4 — Readers, resume and errors (R-007)

- [ ] **T050 [R-011] Reciprocity check**
  - Verify: `base_K = 2I` passes; `base_K = I` returns `:domain_failure`.

- [ ] **T060 [R-007] Inference and rejection matrix** — **Held: OD-2** for F5 only
  - Output: every inference and rejection case in R-007 for F1–F6.
  - Verify: one test per case.

- [ ] **T061 [R-007] F5 readers** — **Held: OD-2**
  - Output: `analyze_fuzzy_axions_orientifold_gap.py`,
    `trace_lost_h11_3_fixed_components.py`, `orientifold_population_preflight.py`.
  - Verify: Python tests per accepted and rejected version.

- [ ] **T062 [R-007] Error propagation**
  - Output: `_validate_axion_photon_group`, both `_has_axion_photon` methods,
    `reproduce_appendix_b_spectra.jl`, the notebook reader, and the inflation
    resume readers let R-007 errors propagate.
  - Verify: a legacy or invalid input raises at each site.

- [ ] **T063 [R-007] Inflation CSV inference** — **Held: OD-3**
  - Verify: one test per accepted historical version.

- [ ] **T064 [R-007] Combination rule** — **Held: OD-3** for the version-column choice
  - Output: mixed-label rejection in every combiner in the reader inventory.
  - Verify: `ArgumentError` naming both labels.

- [ ] **T065 [R-007] Batch resume: F1, F3, F6, scan-prep**
  - Output: behaviours in `plan.md` *Batch-resume runners*; `--force` for the F6
    CLI.
  - Verify: one resume test per runner, including a legacy output without and
    with the overwrite option.

- [ ] **T066 [R-007] Scale-continuation resume** — **Held: OD-5**

- [ ] **T067 [R-007] Layout migrations**
  - Output: migrations add no label or version and refuse a current-version
    target.
  - Verify: migration tests.

## Phase 5 — Reproduction paths (R-008)

- [ ] **T070 [R-008] Reproduction labels on structs and `NamedTuple`s**
  - Output: `kinetic_normalization` on every [S]/[NT] reproduction result in the
    search record.
  - Verify: reproduction tests numerically unchanged; label assertions.

- [ ] **T071 [R-008] Bare-array contracts and persisted reproduction outputs**
  - Output: docstrings for `n5_kinetic_matrix`, `n8_kinetic_matrix`; labels in
    the `.jls` fixture, the A96 JSON and the comparison CSV.
  - Verify: label assertions; physical readers reject reproduction labels.

- [ ] **T072 [R-008] K-free and mixed reproduction functions** — **Held: OD-4, OD-7**

## Phase 6 — Documentation and evidence

- [ ] **T080 [R-009] Conventions page and pointers**
  - Verify: docs build; links from docstrings, README and the axion–photon page.

- [ ] **T081 [R-010] Certification audit note**
  - Output: the `plan.md` R-010 audit restated in the PR.

- [ ] **T090 [R-002, G1] Bitwise comparison at the implementation SHA**
  - Escalate if: any artifact differs.

- [ ] **T091 [G1] Evidence bundle**
  - Output: fast and full suites, Python regressions, docs, `agent_verify.py` at
    the exact SHA; changed-test list with reasons; R-005 count report.

- [ ] **T092 [G1] Spec ↔ plan ↔ tasks ↔ code convergence**
  - Verify: every R-001..R-011 has evidence on the candidate SHA.

## Phase 7 — Independent verification

- [ ] **T093 [G2] Fresh-context review of the whole PR at the exact SHA**
  - Must not rely on the implementation worker's report.
  - Escalate if: a blocking finding needs an implementation change (new
    candidate; repeat T090–T093).
