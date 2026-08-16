# CYAxiverse.jl — package analysis and performance/precision review

**Date:** 2026-08-16
**Base commit:** `f6bb850` (branch `codex/stage2-independent-runs`); all `src/` findings verified to apply unchanged on `vmm` (`8948877`).
**Method:** full read of `README`, `Project.toml`, `docs/`, `src/`, `scripts/`, `test/`, plus `git log`. All performance numbers were measured on the author's machine under Julia 1.12.6, not estimated.

This document has two parts:

- **Part I — what the package does**: domain, pipelines, module map, analyses under development, and proposed extensions.
- **Part II — performance and numerical accuracy**: prioritised, measured findings with `file:line` references.

Status of the three correctness defects in Part II §0: **fixed** in PR #83 (`fix/spectrum-basis-correctness` → `vmm`).

---
---

# Part I — What the package does

## 1. Domain

CYAxiverse computes the **4d effective axion/ALP sector of type IIB Calabi–Yau orientifold compactifications**, from Kreuzer–Skarke reflexive 4-polytopes → fine regular star triangulations (FRSTs) → CY₃ hypersurfaces.

Everything reduces to one Lagrangian triple:

| Object | Shape | Meaning |
|---|---|---|
| `K` | `h11 × h11` | axion kinetic (Kähler) metric; stored on disk as `Kinv` |
| `Q` | `h11 × N_inst` | integer instanton charge matrix, **one charge vector per column** |
| `L` | `2 × N_inst` | instanton scales as `(sign/mantissa, log10 Λ⁴)` per column |

with

```
V(θ) = Σ_a Λ_a⁴ (1 − cos 2π Q_a·θ),      Λ_a⁴ = L[1,a] · 10^L[2,a]
```

The `(sign, log10)` representation is load-bearing: instanton scales span 30+ orders of magnitude, so essentially every routine works in log space or in `ArbFloat`/`BigFloat` to avoid underflow. `.copilot/AGENTS.md` §1 makes this an explicit coding rule.

### Core computed objects

| Object | Where | Meaning |
|---|---|---|
| Axion masses `m` | `generate.hp_spectrum`, `pq_spectrum` | log₁₀(m/eV) from eigenvalues of `K^{-1/2} H K^{-T/2}`, `H = Σ Λ_a⁴ q_a q_aᵀ` |
| Decay constants `fK`, `fpert` | same | kinetic-eigenvalue and perturbative (quartic-derived) decay constants |
| Quartics `λself`, `λ31`, `λ22` | same | signed log₁₀ of the three families, with cancellation diagnostics |
| Vacua count `N_vac` | `generate.vacua*`, `jlm_reduced` | distinct minima on the axion torus |
| Critical branches | `generate.foreach_leading_critical_branch` | exact `2^h11 × |det Q̃|` leading-order enumeration |
| Axion–photon `C_γ`, `g_aγγ` | `axion_photon` | leading hierarchy → mixing → photon coupling |
| Inflation candidates | `inflation_points`, `paper_benchmarks` | ε/η diagnostics, hilltop/catastrophe flows, e-folds |

Physical constants live in `generate.constants()` (`src/generate.jl:49`): `M_Pl = 2.435e18 GeV`, `H₀ = 2.13×0.7×10⁻³³`, `log10(2π)`.

## 2. Pipelines — and a naming hazard

There are **three unrelated "Stage N" vocabularies**. This required disambiguation three separate times during review and is the single most likely source of misreading for a new collaborator.

| Vocabulary | Where | Meaning |
|---|---|---|
| **Geometry** | `scripts/*.py` | Stage 1 = raw FRST collection; Stage 2 = EFT reference |
| **Vacua** | `scripts/vacua_pipeline.jl`, `validate_vacua_stage4_5.jl` | Stage 4 = anchor/regression; Stage 5 = resource benchmarking |
| **Inflation** | `validation/inflation_scan_call_contract.md` | Stage 3 = HP refinement; 4 = diagnostics; 5 = shards; 6 = stratified pilot; 7 = candidate pilot |

> **Recommendation.** A one-page `docs/src/pipelines.md` naming these `geometry-stage-N` / `vacua-stage-N` / `inflation-stage-N`, with the dataflow diagram, costs an hour and prevents a class of error a reviewer would not catch.

### 2.1 Geometry pipeline (Python + CYTools)

```
KS database (cytools.fetch_polytopes) or scripts/manifests/h11_491_11_ks.json
        │
        ▼  scripts/generate_stage1_raw_frsts.py                    [Stage 1]
           APPROVED_PLAN   {50:500, 100:500, 200:300, 491:100}
           polytopes       {50:50,  100:50,  200:30,  491:1}
           samplers        fair | fast | ntfe_fast | gnn_ntfe
           emits  frst_candidates/*.h5  (lattice pts, simplices, tri hash,
                    optional gzip-9 topology_cache: h11, h21, basis,
                    basis_matrix CSR, kappa COO, c2, mori_cone,
                    kahler hyperplanes, face_restriction_dim2)
                  frst_terminal_statuses.jsonl, polytope_manifest.json,
                  run_manifest.json
           population is FROZEN at 1400 geometries
        │
        ▼  scripts/generate_stage2_eft_reference.py                [Stage 2]
           reads only Stage-1 artifacts; NEVER resamples an FRST
           topology/cone checks → orientifold → Kähler point
             (--moduli-policy canonical_qcd) → divisor cuts →
             visible sector (intersecting_d7) → assignment pool → --eft rows
           emits  h11_XXX/np_YYYYYYY/cy_ZZZZZZZ/cyax.h5
                  stage2_input_ledger.jsonl, stage2_terminal_statuses.jsonl,
                  stage2_topology_diagnostics.jsonl,
                  stage2_kaehler_point_diagnostics.jsonl,
                  stage2_assignment_pool_rejections.jsonl,
                  charge_factorized_manifest.json, eft_models.parquet
```

**Supporting modules:** `glimmers_schema11.py` (population rules; `QCD_VOLUME_TARGET=40.0`, `QED_VOLUME_MAX=127.5`, `TARGET_GEOMETRY_COUNT=1400`, `MIN/MAX_EFT_ROWS=100k/200k`), `glimmers_raw_frst.py` (Stage-1↔2 HDF5 interchange contract + topology cache codec), `qed_divisor_assignment.py`, `glimmers_eft_row_schema.py`, `glimmers_provenance.py`, `glimmers_proposal_controller.py` (separates accepted target / proposal budget / retry budget), `geometry_charge_conventions.py`, `migrate_stage1_topology_cache.py`, `replenish_stage1_raw_frsts.py`.

**Specialist entry points:** `generate_h11_491_frsts.py`, `probe_h11_491_sampler.py`, `diagnose_h11_491_cytools.py`, `generate_appendix_c_geometry.py` (exact N=8 benchmark), `generate_relaxed_qcd_sample.py`.

### 2.2 The physics of the acceptance filters

Two moduli policies (`scripts/README_generate_geometric_data_multitriangulation.md:681`):

- **`adaptive`** (legacy): randomised angular Kähler points; every prime toric divisor ≥ 1; at least one in `[25, 40]`; Kähler-cone slack ≥ 1; plus the stretched-cone instanton-control inequality from arXiv:2309.01831.
- **`canonical_qcd`** (production, Glimmers-style): `J_tip = tip_of_stretched_cone(1.0)` is an **angular direction only**. Pick a QCD prime divisor, set `m = sqrt(40 / Vol(D_QCD)_tip)`, evaluate at `J_final = m·J_tip`. Divisor volumes scale as `m²`, CY volume as `m³` — deliberately **not** normalised to 1.

**Literature basis** (`README:167`): hep-th/0002240 (KS), arXiv:2008.01730 (fair FRST sampling), arXiv:2309.10855 (NTFE), arXiv:2309.01831 (axion minima / stretched cone), arXiv:2103.06812 (superradiance), arXiv:2305.06363 (orientifolding KS), arXiv:2412.12012 (fuzzy axions — source of the `[25,40]` window), arXiv:2309.13145 (Glimmers — source of `canonical_qcd`), arXiv:2512.00144 (Bayesian inference — source of the immutable-metadata requirement).

### 2.3 Julia consumers

```
cyax.h5 ──read.potential / read.oriented_potential──► (Q, L, K)
   ├─► generate.pq_spectrum / hp_spectrum
   │      / pq_hybrid_physical_spectrum / pq_window_spectrum
   │        └─► scripts/batch_physical_spectrum.jl → spectrum/physical + CSV
   ├─► generate.LQtilde → αmatrix → jlm_reduced.prepare/minimize
   │        └─► scripts/vacua_pipeline.jl / batch_vacua_pipeline.jl
   ├─► inflation_points.prepare_geometry_context → diagnose/correct/gradient_flow
   │        └─► scripts/inflation_scan_prep.jl → shards → merge → pilot reports
   └─► axion_photon.run_local_scan → CSV   (no HDF5 persistence yet)
```

## 3. Data layout

Path convention enforced by `src/filestructure.jl`:

```
$DATA/h11_XXX/np_YYYYYYY/cy_ZZZZZZZ/cyax.h5
                                    /minima.h5
                                    /qshape.h5
```

Verified live schema (`h11_015/np_0000001/cy_0000001/cyax.h5`, `schema_version = cyaxiverse-ks-cy3-v5`):

```
construction_metadata/canonical_lattice_points (4,20) Int64
construction_metadata/face_restriction_dim2    (3,55) Int64
cytools/geometric/  CY_volume () ; Kinv (15,15) ; basis (15,) ; basis_matrix (20,15)
                    c2 (15,) ; curve_volumes (79,) ; divisor_volumes (15,)
                    effective_cone (15,19) Float64   ← direct divisor charges, columns
                    glsm (19,15) ; h11 ; h21 ; kahler_hyperplanes (15,79)
                    kappa (4,90) Float64             ← COO [i,j,k,value], 0-based
                    mori_cone (15,79) ; points (4,20) ; prime_divisor_volumes (19,)
                    prime_toric_divisors (19,) ; simplices (5,68)
                    tip (15,) ; tip_prefactor (2,) ; triangulation_points (4,20)
cytools/potential/  L (2,190) Float64 ; Q (15,190) Float64
root attrs: construction_metadata_json, schema_version
```

`Q` is stored as **Float64**; `axion_photon._read_integer_float_dataset` (`src/axion_photon.jl:241`) exists to convert it to `Matrix{Int}` in 4096-column chunks without materialising the whole array.

Optional groups in newer generator modes: `visible_sector/`, `orientifold/`, `standard_model/`, `assignment_pool/`, `divisor_volume_evidence/`.

Result groups written by Julia: `spectrum/masses`, `spectrum/decay`, `spectrum/quartdiag|quart31|quart22`, `spectrum/cubic`, `spectrum/physical` (+`metadata`), `vacua/`, `vacua_TB/`, `vacua_pipeline/` (+ metadata: method, threshold, starts, tolerances, solver status, Julia version, git revision, runtime), `hilbert/`.

> **Forward-compatibility fork.** Production schema-1.1 Stage-2 HDF5 stores **no dense `Q/L/Kinv`** — only reconstruction references (source datasets, indices, conventions, hashes, replay tolerances). The existing `../data` corpus (30,975 geometries) is the older dense `v5` schema. The Julia reader story currently assumes `v5`.

### Environment variables

| Variable | Consumer | Role |
|---|---|---|
| `CYAXIVERSE_DATA_DIR` | `filestructure.resolve_data_dir` (`src/filestructure.jl:85`) | preferred data root (28 call sites) |
| `newARGS` | same, `_LEGACY_DATA_DIRS` | legacy deployment aliases (`docker`, `vacua_0323`, …) |
| `PYTHON` | `src/init_python.jl`, PyCall ext | interpreter containing CYTools |
| `SLURM_JOB_ID` / `SLURM_ARRAY_TASK_ID` / `SLURM_NPROCS` / `MAX_JOB` | `src/slurm.jl` | HPC dispatch |
| `MOSEKLM_LICENSE_FILE` | Python generator | QP solver for stretched-cone tips (path only, never contents) |
| `CYAXIVERSE_AUDIT_CACHE` | `bin/audit.jl` | Aqua/JET audit env cache |

Resolution precedence (`docs/src/userguide.md:26`): explicit arg → `CYAXIVERSE_DATA_DIR` → `newARGS` alias → checkout-relative `../data`. Never creates the directory; never silently falls back to `pwd()`.

### External tools

- **CYTools ≥ 1.4.0** is the only geometry engine. Julia access is via the weakdep extension `ext/CYAxiversePyCallExt.jl`, which must be explicitly armed (`CYTools.enable_cytools!()`); `ensure_cytools!()` guards every call and refuses to auto-`Pkg.build`.
- `add_functions/cytools_wrapper.jl` wraps `fetch_polytopes`, `Polytope`, `Cone`, `hilbert_basis`, and implements the Julia-side generator `geometries_generate` (`:333`). Handles the CYTools ≥0.8.0 rename `compute_Kinv` → `compute_inverse_kahler_metric`.
- **PALP / TOPCOM are not used directly** — only via CYTools' `cgal`/`qhull` backends.
- **MOSEK** (via `qpsolvers`) preferred for stretched-cone QPs; HiGHS fallback. Solver choice recorded per artifact.
- **Nemo** supplies exact integer linear algebra: Smith normal form, exact rank, exact determinants.
- **MPI/MPIClusterManagers** in legacy HPC scripts — *not* in `Project.toml`, so those run against a different environment.
- **Docker** builds on the CYTools image but pins Julia 1.8.4, while `Project.toml` requires 1.12 — stale.

## 4. Module map

Load order fixed in `src/CYAxiverse.jl:14-22`. No re-export; everything is namespaced (`CYAxiverse.generate.…`). Only `greet_CYAxiverse` is exported.

```
structs ──► filestructure ──► read ──► minimizer ──► generate ──► jlm_reduced
                                                        │             ▲
                                                        ├──► paper_benchmarks
   read ──► inflation_points                            │
   filestructure + structs ──► axion_photon             │
   plotting (stubs) ◄── ext/CYAxiverseCairoMakieExt
   ext/CYAxiversePyCallExt ──► jlm_python, jlm_minimizer, cytools_wrapper
```

### `structs` — `src/structs.jl` (416 lines)

Pure type definitions. `GeometryIndex{T}(h11, polytope, frst)` is the universal key. Geometry: `TopologicalData`, `GeometricData`, `AxionPotential(L, Q, K::Hermitian)`. Spectrum: `AxionSpectrum`, `PhysicalAxionSpectrum`, `IndexedAxionSpectrum`.

Notably rich diagnostics: `QuarticComponentDiagnostics`/`QuarticDiagnostics` (`orders_lost`, `digits_remaining`, `reliable`), `MassBasisDiagnostics` (eigenpair residuals, nearest relative gaps, orthogonality error), `PerturbativeSplitDiagnostics` (`certified_safe`), `InstantonHierarchyDiagnostics`, `SpectrumWindowDiagnostics` (`counts_by_precision`, boundary gaps, `provisional`).

Vacua/basis: `LQLinearlyIndependent`, `CanonicalQBasis`, `Canonicalα`, `Projector`, `ProjectedQ`, `RationalQSNF`, `BasisSNF`, `ReducedPotential`, `Min_JLM_Square/1D/ND`, `Solver1D`, `SolverND`. Tree: `MyTree{D}` + `ParentTrack`.

### `filestructure` — `src/filestructure.jl` (541 lines)

`resolve_data_dir` / `present_dir` (modern boundary), `_LEGACY_DATA_DIRS` (hard-coded cluster paths), `np_path_generate`, `paths_cy()` (caches `paths_cy.h5`), `h11lst`, `count_geometries`, `isgeometry`, `geom_dir`, `geom_dir_read`, `cyax_file`, `minfile`, `logfile`, `logcreate`, `plots_dir`.

> **Inconsistency.** `geom_dir`/`geom_dir_read` (`:382-446`) still branch on `localARGS()` and the legacy `h11 >= 238` flat layout, and `geom_dir` calls `mkdir` as a read-path side effect. Only `present_dir` was migrated to `resolve_data_dir`, so the modern boundary is half-applied.

### `read` — `src/read.jl` (584 lines)

`topology`, `geometry(; hilbert)`, `hilbert_basis`, `visible_sector` (`:86`), `construction_metadata_json`.

`potential(geom_idx)` → `AxionPotential(L, Q, Hermitian(inv(Kinv)))`. **Inverts `Kinv` on every call** — see Part II §B1.

`oriented_potential(geom_idx; canonicalize_charge_rows=true)` (`:279`) is the **package-owned normalisation boundary**: accepts both historical orientations, validates dimensions/finiteness, and applies `_canonicalize_generated_potential`, which detects the triangular `n + n(n−1)/2` term count and collapses duplicate leading charge rows.

Spectrum readers: `pq_spectrum`, `hp_spectrum`, `physical_spectrum`, `cubic_tensor`. Vacua readers: `qshape`, `vacua`, `vacua_TB`, `vacua_jlm`, `pipeline_vacua`.

### `minimizer` — `src/minimizer.jl` (677 lines)

`critical_points(L, Q; …)` (`:91`) is the modern deterministic solver: Halton starts, `NLsolve` Newton, roots folded to `[0,1)^N`, deduplicated by **periodic** sup-distance, classified by Hessian inertia. Each stationarity equation is rescaled by the largest log amplitude touching that row so hierarchically suppressed directions survive Float64. Supports `initial_points` seeds displaced along negative Hessian modes.

Legacy `minimize`/`minimize_save`/`grad_std`/`id_minimize`/`subspace_minimize`/`minima_lattice` are ArbFloat-based; several are marked broken in their docstrings (`id_minimize`: "Currently cannot locate local minima").

### `generate` — `src/generate.jl` (4253 lines) — the core

**(a) Pseudo-data:** `pseudo_Q`, `pseudo_K`, `pseudo_L`.

**(b) Derivative evaluators:** `V`, `jacobian`, `hessian`, `cubic` (rank-3 `∂ᵢ∂ⱼ∂ₖV`), `hessian_norm`. Plus reusable-workspace evaluators promoted from the inflation scripts: `LogShiftedDerivativeWorkspace` + `logshifted_derivatives!`, and `StructuredChargeRepresentation` + `structured_charge_evaluator` — a *validated* base-plus-pairwise factorisation (`Q_pair[:,k] = Q_j − Q_i`) with hash-based detection and an explicit generic fallback carrying `fallback_reason`.

**(c) Spectra:**
- `hp_spectrum(K,L,Q; prec=1000, …)` — full arbitrary precision; builds `H` in `ArbFloat`, Cholesky-whitens, `eigen`, then a fused instanton loop accumulating `λself`/`λ31`/`λ22` simultaneously.
- `pq_spectrum(K,L,Q; mixing_correction, …)` — PQ selection (`LQtilde`) then sequential PQ frame, optional leading-Hessian mass-basis correction. Quartics contracted in log space via `pq_contracted_log!`.
- `pq_hybrid_physical_spectrum` — production high-h11 path: arbitrary-precision inertia counting (`positive_inertia` on a `BunchKaufman`) + Float64-seeded Schur refinement, oversampling, `quartic_backend=:auto|:dense|:sparse`. Falls back to the full HP eigensystem and reports `provisional`.
- `pq_window_spectrum` — two-sided mass windows with `boundary_margin_log10` and precision-stabilisation records.
- `instanton_scale_blocks` / `instanton_hierarchy_diagnostics` — `certified_safe` requires `separation_gap ≥ 6` **and** canonical charge coupling `≤ 1e-6`; explicitly *not* a theorem of block-diagonality.
- Measured scaling (`docs/src/userguide.md:194`): h11=200 with 20,910 instantons → 0.65 s selector, 14 s hybrid solve, 5–8 physical modes.

**(d) Basis / vacua:**
- `LQtilde(Q,L)` (`:2617`) — sorts by `L[2,:]` descending, picks the first `h11` independent columns via allocation-free doubly-reorthogonalised Gram–Schmidt. Verified against the previous `rank`-based implementation on all 1000 h11=10 geometries.
- `αmatrix` → `Canonicalα` with `α = Q̃⁻¹Q̄` in exact `Rational`.
- `vacua_SNF` (Nemo SNF → `θ∥`), `basis_snf`, `vacua`, `vacua_TB`, `vacua_id_basis`, `vacua_MK`, `vacua_projector`, `vacuaΩ`, `vacuaΠ`, `vacua_full`, `vacua_no_optim`, `vacua_estimate`.
- `foreach_leading_critical_branch` / `leading_critical_branches` (`:2908`) — streams all `2^h11 × |det Q̃|` branches without materialising them; supports `negative_mode_range`/`max_negative_modes`; mask counts in `BigInt` (fixes an Int overflow at h11≥64); returns exact mask counts, visited/skipped, lattice copies, search classification.
- `_exact_integer_determinant` (`:2683`) — fraction-free Bareiss with `BigInt` fallback.

### `jlm_reduced` — `src/jlm_reduced.jl` (420 lines)

Julia-native reimplementation of the collaborator ("JLM") reduced-minima method. `prepare(Q,L; threshold, reduction=:alphamatrix|:catastrophe)` separates the expensive reduction from the solve so scans can reuse it. `critical_ensemble` seeds from the leading sign pattern plus ±0.125 per-axion displacements.

### `paper_benchmarks` — `src/paper_benchmarks.jl` + 3 files (1753 lines)

Deterministic reproduction targets, aliased as `CYAxiverse.axion_benchmarks`. `n5_potential`, `n8_potential`/`n8_full_potential` (12 diagonal + 66 cross terms), `n8_geometry` (published Appendix-C vertices, hard-coded 8×8 kinetic matrix, `V=126`, `h21=28`, `χ=−40`), `n5_critical_scale() = (4/π)log(1024/255)` (exact cusp catastrophe), `n8_hilltop`, `n8_gradient_flow`, `n8_stiff_gradient_flow` (backward Euler). `poly102_inflation.jl` (857 lines) holds the fixed poly-102 model with `N8_KC = 0.674506370003365`.

### `inflation_points` — `src/inflation_points.jl` (500 lines)

The **generic**, geometry-agnostic inflation boundary; explicitly refuses to choose population, scale path, or threshold. `basis_policy()` declares `:periodic_string` as working basis and `:mass_eigenbasis` as physical. `gradient_flow` is fixed-step RK4 in **e-folds** in the canonical chart `χ = Lᵀθ`, tracking slow-roll windows via `indicator = max(ε,|η∥|) − 1`. `compare_precision` gates Float64 against BigFloat on residual + inertia agreement.

### `axion_photon` (alias `glimmers`) — `src/axion_photon.jl` (1073 lines)

Self-contained implementation of the Glimmers leading chain, deliberately **not** routed through `read`/`generate`.

`_select_independent_terms` (`:605-725`) is the most interesting numerical kernel: a **modular rank screen mod p = 1_000_003** proves independence cheaply; modularly-dependent columns are buffered and resolved as one exact batch via `Nemo.rank` binary search, after which it switches permanently to exact `Rational{BigInt}` elimination. Emits a replayable `RationalRankCertificate` with ordered source indices, prefix ranks, and the exact selected determinant.

`_canonical_frame` (`:727-748`) — QR of `Lᵀ Q_reduced` with `K = LLᵀ` Cholesky, sign-fixed so `diag(q) > 0`. **This is the one place in the package that gets the `Kinv` handling right** (see Part II §B1) and should be the template.

`leading_hierarchy` — `log10 f = log10 M_Pl − log10 2π − log10 q_aa`; `log10 m/eV = ½ log10Λ_a⁴ + log10 M_Pl + 9 + log10 2π + log10 q_aa`. `mixing_matrix`, `photon_observables` (`n_EM = Q_reduced \ Q_EM`, `C_γ,a = Σ_b n_EM^b Θ_ba`, `g = α_EM C_γ/(2π f)`, below-threshold suppression in log space, `Γ_γγ = m³g²/(64π)`), `qed_instanton_log10_threshold_eV`.

### `plotting` + extension

Weak-dependency split: `src/plotting.jl` (188) declares types and *empty function stubs*; `ext/CYAxiverseCairoMakieExt.jl` (644) supplies methods only when CairoMakie + ColorSchemes are loaded. `paper_style()` fixes serif/STIX typography, pale lavender-gray background, white gridlines.

## 5. Analyses and their state

| Analysis | State |
|---|---|
| **Axion–photon coupling** | Kernels + tests + docs + CLI complete. **CSV only — no HDF5 persistence, no batch runner.** |
| **Axion spectra** | Complete and validated. Scaling verified at h11 ∈ {150,180,200}; 150 vs 200 digits agree on all 30 geometries |
| **Vacua counting** | Mature, with an explicit 4-tier honesty layer |
| **Inflation** | Generic *screening* works; generic *refinement* is the acknowledged gap |
| **Kähler / geometric sampling** | The active WIP area |
| **Topological statistics** | Infrastructure present, analysis not written. 30,975 geometries on disk |

### Detail

**Axion–photon.** Open questions the docs state it targets (`docs/src/axion_photon.md:257`): stability of the local canonical frame; variation of `f, m, C_γ, g` across h11 slices; sparsity of leading photon couplings; which width estimate dominates. Explicitly deferred (`:266`): the 200,000-model ensemble, the h11=491 slice, D7 tadpole cancellation, matter spectrum, flux choices, E3 zero-mode proof, QCD mass normalisation, GUT vs non-GUT, X-ray/helioscope limits, birefringence, reheating/freeze-in/dark-radiation/DM composition, CP phases, resonance analysis.

> **Largest stated approximation.** The reference EFT "treats the full CYTools `h11` basis as the `C4`-axion sector under a declared paper-style all-`C4` assumption. The assumption is recorded as metadata and is not inferred from, or enforced by, the computed orientifold parity split."

**Vacua.** Decision tree (`docs/src/userguide.md:43`): (1) square determinant-certified leading problem — exact for *that* algebraic problem; (2) `leading_critical_branches` — deterministic prefilter, **not** a completeness claim; (3) `jlm_reduced.prepare` + repeated `minimize` — a search-budget result; (4) bounded finite-start search. Comparison target `paper_benchmarks/2023_minima/` is a **digitised** version of the 2023 paper's Fig. 1, explicitly labelled "a different estimand, not a pass/fail equality target."

**Inflation resource wall.** h11=150 and h11=300 have exact `det Q̃` = 2 and 4, so branch counts are `2·2^150` and `4·2^300`; both hit `max_branches=100000`. The h11=300 sample produced ~589 MB of stage allocations. Policy tiers: normal ≤50, middle 51–100, high-memory ≥101; refinement requires ≤750 MB cumulative allocation and ≤300 MB output.

**Not present / stubbed.** `docs/src/examples.md` is **0 bytes** while `docs/make.jl:20` lists it as a page. `id_minimize` (`src/minimizer.jl:563`) cannot locate local minima per its own docstring. `id_minima` (`:605`) is a bare stub. `TBW` docstrings on `h11lst(Vector)`, `omega`, `norm2`, `norm2minus1`, `phase`, `vacua_id`, `minima_lattice`. `mathematica/Quartics_MD.nb` is not integrated.

## 6. Proposed extensions

Ranked by value/effort, assuming the author rather than a cold start.

### Tier 1 — high value, low effort (kernels exist; only orchestration missing)

1. **Axion–photon batch runner + HDF5 persistence.** `run_local_scan` is capped at 4 h11 slices × 2 files and writes CSV only. Model `scripts/batch_axion_photon.jl` on `batch_physical_spectrum.jl` (discovery, `--force`, skip-if-complete, per-geometry CSV flush, `Distributed` workers, per-geometry HDF5 ownership), persisting to `spectrum/axion_photon`: `log10_f_GeV`, `log10_mass_eV`, `Cgamma`, `log10_g_GeVinv`, `log10_g_effective_GeVinv`, widths, residuals, `rank_certificate_payload`. **Single highest-leverage change in the repo** — converts a validated kernel into a population result across 30,975 geometries.
2. **The exclusion figure the plotting API was built for.** `docs/src/userguide.md:250` already demos `plotting.exclusionplot` with "Fraction excluded" vs h11 and a 95% C.L. band — on *synthetic* data. Wire real `g_aγγ(m_a)` against CAST, ADMX, SN1987A, Chandra/H1821+643.
3. **Black-hole superradiance exclusion.** The acknowledgements name arXiv:2103.06812 as the package's origin, and `m_a`, `f_a` per mode already exist. Add `α = G M m_a` for measured BH mass/spin pairs, the superradiance rate condition, and the `f_a`-dependent self-interaction quench. Reuses the same exclusion machinery as (2).
4. **Fill `docs/src/examples.md`** with three end-to-end recipes (geometry → physical spectrum; → vacua pipeline; → axion-photon result). Currently only findable by reading validation notes.
5. **Fix the docs deploy branch.** `.github/workflows/Documentation.yml` triggers on push to `dev` and `docs/make.jl` sets `devbranch = "dev"`, but no `dev` branch exists. Docs build on PRs and never deploy.
6. **Add Python tests to CI.** Seven `scripts/test_*.py` files (`test_stage2_stage_boundary.py` alone is 47 KB) with **zero** CI coverage — `.github/workflows/CI.yml` is Julia-only. The CYTools-free subset runs today; the Stage-1 tests already mock CYTools entirely via `FakeCalabiYau`. Add a `conftest.py` so imports don't depend on cwd.

### Tier 2 — medium effort

7. **Generic inflation refinement** — the named blocker. `validation/inflation_scan_call_contract.md:290` states the requirement precisely. `gradient_flow` and `compare_precision` are already generic; missing are an adaptive/arbitrary-precision ODE path (`OrdinaryDiffEq` is a declared dep used only by the benchmark models), an explicit event policy matching `n8_physical_gradient_flow`'s `end_event`, and a candidate-metadata struct. Converts Stage 3 from `:n8_poly102`-only to corpus-wide.
8. **Kill the `read.potential` inverse** — see Part II §B1.
9. **Unify the `geom_dir` path family with `resolve_data_dir`**, with the legacy layout as an explicit opt-in flag and directory creation split into `ensure_geom_dir!`.
10. **Split `test/runtests.jl`** (1758 lines, ~30 testsets, no filtering) by domain with an `ARGS` selector.
11. **Wire `bin/audit.jl` (Aqua + JET) into CI** as a non-blocking job; it already handles its own cached environment.
12. **Arrow/Parquet output on the Julia side** to match `eft_models.parquet`, so the two halves join without a text round-trip.

### Tier 3 — high scientific value, higher effort

13. **Parity-resolved (`h11_plus`) EFT.** The generator already stores `orientifold/h2_involution_matrix`, `invariant_kahler_basis`, `anti_invariant_h2_basis`, `prime_divisor_invariant_indices`, and the `h11_plus`/`h11_minus` split. The work is to project `Q`, `L`, `K` onto the invariant subspace and re-run the hierarchy. Changes physical conclusions (C₂/C₄ axion content, the effective `h11` entering every count), so it should be a labelled parallel track, not a replacement.
14. **Relic abundance / dark-radiation layer.** Misalignment relic abundance and ALP decay history are self-contained given `m_a`, `f_a`. Turns spectrum data into a cosmological statement; natural companion to (2) and (3).
15. **Bayesian conditioning on the sampler.** Every artifact records sampler scheme, seed, controls, favorability filter, acceptance outcomes, and terminal-status counts by h11 — precisely so this is possible. The missing layer is importance reweighting. The README is explicit that it does not exist: *"Provenance makes future Bayesian conditioning possible; it does not itself specify a prior or correct for all selection effects."* The only path from "31k geometries" to a defensible population statement, and every other analysis inherits the correction.
16. **Vacua counting beyond exponential enumeration.** Two partially-scaffolded escapes: (a) use the implemented `negative_mode_range`/`max_negative_modes` to count *minima* (index 0) directly, with Morse-theoretic bounds on the total; (b) an unbiased sampling estimator over sign masks with reported variance, so h11 ≥ 101 yields a number with error bars instead of a `branch_cap` status. The high-h11 tier currently produces no count at all.
17. **QCD-axion identification.** `standard_model/qcd_divisor_index` and `visible_sector/qcd_charge` are stored, and the `[25,40]` window comes from the fuzzy-axion paper, but nothing ties `Λ_QCD` to the selected divisor or checks where `m_a(QCD)` falls. The QED analogue (`qed_instanton_log10_threshold_eV`) is implemented and largely symmetric.

---
---

# Part II — Performance and numerical accuracy

Every measurement below was taken on the author's machine, Julia 1.12.6.

## 0. Correctness defects — FIXED in PR #83

Recorded here for the record; all three are resolved on `fix/spectrum-basis-correctness`.

### 0.1 `hp_spectrum` mislabelled every `λ31` and `λ22`

`src/generate.jl:873-874` built the index lists with a **two-dimensional** comprehension (column-major, first index fastest) while the fused accumulation loop at `:904-917` filled the value arrays with the first index **slowest**.

Measured against an independent contraction at h11 = 4:

- `λ31`: **all 12 components** mislabelled — the value at label `(i,j)` was the true value of `(j,i)`. Wrong for all h11 ≥ 2.
- `λ22`: **2 of 6** mislabelled (the `(4,1)`/`(3,2)` swap). Wrong for all h11 ≥ 4; h11 = 3 coincidentally agrees.
- Maximum discrepancy **2.78 in log₁₀**. `λself` unaffected.

`spectrum/quart31/index` and `spectrum/quart22/index` written by `hp_spectrum_save` carry the old labelling. **Values are correct — only the index arrays need relabelling**, so affected results can be migrated in place rather than recomputed.

It went unnoticed because `test/runtests.jl:1088-1119` exercised `hp_spectrum` only at h11 = 1, where both index arrays are empty (`:1118-1119` asserts exactly that).

### 0.2 `basis_snf` returned the wrong `id_coords`

`src/generate.jl:3437`/`:3442` applied `inv` **inside** a `@.` broadcast, where it acts elementwise, and both `ifelse` branches returned the original matrix. `BasisSNF.id_coords` was a thresholded copy of `basis`. It also raised `DivideError` whenever thresholding produced an exact zero — the common case.

### 0.3 `read.L_arb` iterated the wrong axis

`src/read.jl:339-345` allocated `Ltemp` of length `size(L,2)` but looped `axes(L,1)` (= `1:2`), reading `L[i,1]`/`L[i,2]`. `L` is `2 × N` everywhere else, so only two entries were filled, from the wrong values.

## 1. Priority A — high impact, low risk

### A1. `Matrix{ArbFloat}` has an abstract element type

Unqualified `ArbFloat` is the `UnionAll` `ArbFloat{P}`, so `zeros(ArbFloat, n, n)` has `isconcretetype == false` — every element boxed, every operation a dynamic dispatch.

Affected: `src/generate.jl:824, 833, 851, 870-872, 881-885, 516`; `src/read.jl:339`; `src/minimizer.jl:45, 54, 63, 439`.

```
rank-one accumulation kernel, h11=20, 200 instantons
digits=100 : abstract 57.5 ms  vs  concrete 15.3 ms   → 3.75x
digits=1000: abstract 50.3 ms  vs  concrete 53.9 ms   → ~1x (Arb arithmetic dominates)
```

Production defaults are `prec=200` (`scripts/batch_physical_spectrum.jl:52`) and `prec=100`, so this is a straight 3–4× on the dominant kernel with zero accuracy impact. `high_precision_leading_hessian` already has the right pattern at `:1220`: `T = typeof(ArbFloat(0))`.

Also `src/minimizer.jl:65-66` converts `ArbFloat(LV[c])`, `ArbFloat(QV[c,i])`, `ArbFloat(QV[c,j])` **inside** the `O(p·h11²)` triple loop — hoist to typed arrays once.

### A2. `hcat(collect.(q)...)` splats ~10⁴ arguments

`src/generate.jl:1548, 1549, 1634, 1635, 1911, 1912, 1989, 1991`. For h11=100 that splats 9,900 arguments.

```
n = 9900 columns of length 4
hcat(splat)   25.42 ms
reduce(hcat)  22.16 ms
preallocated   0.01 ms      → 2542x
```

Same pattern at `src/generate.jl:3377, 4085, 4106, 2668-2669`; `src/jlm_reduced.jl:336`; `src/filestructure.jl:226, 261, 330, 360`.

### A3. `cos.(x' * QV)` recomputed inside the `(i,j)` Hessian loop

`src/minimizer.jl:465-481`, `:524-532`, `:569-577`; and `generate.hessian_norm` at `:593, 600, 609`. An `O(h11·p)` matvec plus `p` transcendentals evaluated `O(h11²)` times per Hessian. At h11=50, p=1500 that is ~1.9×10⁸ flops instead of ~4×10⁶. Hoist `c = cos.(x'*QV)`, form `W = vec(LV) .* vec(c)`, then a single `mul!`. `generate.hessian` (`:538-548`) already does this.

### A4. `pq_contracted_log!` recomputes `log(abs(charge))` `O(h11²·n)` times

`src/generate.jl:1147-1166`. Called ~`1.5 h11²` times (`:1515-1523`); for h11=100, n=5000 that is ≈3×10⁸ `log` calls versus `n·h11 = 5×10⁵` to cache `logQ`/`sgnQ` once.

```
50 components, 5000 instantons: recomputed 3.10 ms  vs  cached 0.15 ms   → 21x
```

Numerically identical.

### A5. `logsum_sorted!` sorts once per component

`src/generate.jl:1133-1141` — `O(h11²·n log n)`. Max-shifted log-sum-exp is `O(n)` and at least as accurate as the sequential `gauss_sum` chain.

> ⚠️ **Do not merge the sign groups before the log-sum.** They are summed separately and combined via `gauss_diff` (`:1175-1177`) precisely to avoid catastrophic cancellation.

### A6. `eigen` where `eigvals` suffices

- `src/generate.jl:1469` — eigenvectors discarded. `n=300`: `eigen` 6.00 ms vs `eigvals` 2.39 ms (**2.5×**).
- `src/minimizer.jl:239, 286-287, 495-496, 542-544, 587-589` — `minimum(eigen(hess(xmin)).values)`; use `eigmin(Hermitian(...))`. `eigen` is called **two or three times on the same matrix** at `:286-287`, `:495-496`, `:542-544`.
- `src/generate.jl:848` — a full generic-precision symmetric eigensolve of a Float64 matrix; see §B1.

### A7. `constants()` re-parses ArbFloat strings on every call

`src/generate.jl:49-54`. Measured **2.26 µs and 1,856 B per call**, at ~25 call sites including inside the precision-doubling loops (`:2295-2305`, `:2036-2050`). Every consumer immediately does `Float64(log10(constants()["MPlanck"]))`. Make them `const` Float64 plus one `T`-generic helper; keep `constants()` as a compatibility wrapper.

### A8. `det(inv(A))` instead of `1/det(A)` over `Rational`

`src/generate.jl:2542, 2548, 3504, 3514`.

```
n=60 integer matrix:
det(Matrix{BigInt})                14.4 ms
inv(Matrix{Rational{BigInt}})     604 ms     → 42x
inv(Matrix{Rational{Int64}})      OverflowError from n = 20
```

`_exact_integer_determinant` (`:2683-2728`) already exists in the same module.

### A9. `opnorm(A*B')` on the full instanton set

`src/generate.jl:1330`. For h11=100, n≈5000 this forms a 2500×2500 dense matrix and takes its full SVD. Since `rank(AB') ≤ h11`, thin QR gives identical singular values:

```
opnorm(A*B')     = 3189.1304935223866
opnorm(R_A*R_B') = 3189.1304935223890     rel diff 7.1e-16
1.057 s  vs  4.06 ms                      → 260x
```

Runs once per block boundary (`:1323`), up to 31 under the `≤ 32` guard at `:1820`/`:2104`. `left_norm`/`right_norm` (`:1328-1329`) should reuse the same `R` factors; hoist `Qcanonical` and use prefix accumulation instead of per-boundary `reduce(vcat, …)` (`:1324-1325`, currently `O(B·n)`).

### A10. `positive_inertia` allocates a `Hermitian` and calls `eigen` per 2×2 block

`src/generate.jl:2257-2258`. For a real symmetric 2×2 the inertia is exact from trace and determinant:

```julia
a, b, c = D[i,i], D[i+1,i], D[i+1,i+1]
det2 = a*c - b*b
positive += det2 < 0 ? 1 : (a + c > 0 ? 2 : 0)
```

On the critical path of every hybrid/window spectrum (`:2285, 2006, 2362, 1835`).

### A11. `LQtildebar` is ~900× slower than `LQtilde` and quadratic in memory

`src/generate.jl:3206-3267` calls `Nemo.nullspace` on a growing matrix for **every** instanton (`:3220-3222`) and grows four arrays with `hcat` in-loop (`:3224-3228`).

```
h11=20 (n=300):  LQtilde 0.026 ms /  72 KB    LQtildebar  4.48 ms /  15.6 MB   (171x, 215x)
h11=40 (n=990):  LQtilde 0.070 ms / 388 KB    LQtildebar 63.50 ms / 277.3 MB   (901x, 715x)
```

On the default `hp_spectrum(h11,tri,cy)` path (`:773`, default `selection=:hp_effective` at `:957`, `:965`) plus `vacua` (`:2538`), `vacua_TB` (`:3496`), `vacua_id_basis` (`:3312`), `vacua_MK` (`:3611`), `vacua_projector` (`:3701`), `vacuaΠ` (`:4015`), `vacua_full` (`:4044`), `vacua_no_optim` (`:4129`).

> ⚠️ **Semantics differ.** `LQtildebar` appends retained `Qbar` columns onto `Qhat` (`:3259-3261`); `αmatrix` returns `Qhat == Qtilde` and carries the extras in `Qbar_eff`. `vacua_id_basis:3314` slices `[:, 1:h11]` so it is compatible; `hp_spectrum`'s `:hp_effective` path uses the full `Qhat` and needs `hcat(Qtilde, Qbar_eff)`.

### A12. `inflation_points` builds Hessians it discards

- `src/inflation_points.jl:440` — the line search calls `derivatives` (`:188-219`), which builds the full `h11 × h11` Hessian with an `O(h11²·n)` inner loop. With `max_line_search = 12` (`:408`), up to 12 discarded Hessians per Newton step.
- `src/inflation_points.jl:261-266` — `_flow_rhs` uses only `value` and `gradient`, but `_flow_state` (`:243-259`) computes `lower \ hessian / lower'`: two dense triangular solves, `O(h11³)`, plus the Hessian assembly. `gradient_flow` runs 4 `_flow_rhs` + 1 `_flow_state` per RK4 step (`:338-344`); with defaults `max_efolds=60`, `step=1e-3` that is 60,000 steps → **~240,000 wasted Hessian builds and ~480,000 wasted triangular solves**. This almost certainly dominates the whole inflation module.

Split `derivatives` into `value_gradient!` and `derivatives!` writing into preallocated buffers. `_flow_state`'s `eta_parallel` (`:255-256`) only needs `tangent' H_c tangent`; use `w = transpose(lower) \ tangent` once and form `dot(w, data.hessian * w)` — `O(h11²)` instead of `O(h11³)`.

### A13. `grad_std` recomputed inside every `minimize`

`src/minimizer.jl:486` performs 100 gradient evaluations (`:428-433`) and does not depend on `x0`. `subspace_minimize` (`:620-634`) calls `minimize` `runs` times, with `runs = 10_000` by default and `runs = 100_000` from `vacua_full` (`src/generate.jl:4037`) → 10⁶–10⁷ wasted gradient evaluations. Hoist `gradσ` and `x_tol` into the caller.

### A14. Dead and harmful GC calls

`src/minimizer.jl:254, 297, 337, 672` are `GC.gc()` **after** `return` (unreachable). `:238, 284, 324, 366, 391` are live forced *full* collections inside the minimisation loop, defeating the generational collector. `scripts/batch_physical_spectrum.jl:312` uses `GC.gc(false)` — the right call if one is needed at all.

### A15. Unused heavy dependencies

`IntervalArithmetic`, `Tullio`, `LoopVectorization`, `LinearSolve`, `OrdinaryDiffEq`, `TimerOutputs` are `using`-ed in `src/generate.jl:10-14` and `src/minimizer.jl:10-11` but **never used** in `src/`. `StaticArrays` appears once (`:534`) in a function whose result is discarded (§C4). Pure win for every `pmap` worker process.

## 2. Priority B — high impact, needs care

### B1. `read.potential` forms `inv(Kinv)` — the worst numerical decision in the package

`src/read.jl:164, 171`: `AxionPotential(L, Q, Hermitian(inv(Kinv)))`. Everything downstream then re-factorises: `cholesky(K)` at `src/generate.jl:847, 1189, 1208, 1233, 1320, 1470, 1814, 1948, 2392` and `src/inflation_points.jl:112`.

Measured, `n = 40`, against an exact `BigFloat` reference:

```
cond(Kinv)=1e6    inv-path max rel err 4.5e-8      chol-path 2.3e-10
cond(Kinv)=1e10   inv-path: cholesky(inv(Kinv)) THROWS PosDefException    chol-path 8.7e-8
cond(Kinv)=1e14   inv-path: THROWS PosDefException                        chol-path 1.0e-2
```

For Kähler metrics from stretched-cone tips at large h11, `cond(Kinv) ≥ 10¹⁰` is routine — this is not merely ~200× less accurate, it is a **hard failure mode**.

The whitening never needs `K`. With `Kinv = C Cᵀ`, the generalised problem `H v = m² K v` becomes

```
whitened = Symmetric(Cᵀ H C)      # same spectrum, no inverse, no triangular solve
Tls      = C * eigenvectors       # replaces  Kfactor.L' \ eigenvectors
```

Two further points:

- **`fK` should come from the Cholesky factor's singular values, not `eigvals(K)`.** `f_K,i = ½ log₁₀ λ_i(K) = −log₁₀ σ_i(C)`. One-sided Jacobi / `svdvals` on a triangular factor gives singular values with **relative** accuracy independent of `cond(K)` (Demmel–Veselić), whereas `eigvals(inv(Kinv))` has absolute accuracy `O(ε‖K‖)` — which destroys exactly the small decay constants the physics cares about.
- `src/generate.jl:848` computes a **full generic-precision symmetric eigendecomposition of a Float64 matrix** at `prec` digits. The input carries at most 16 digits, so the extra precision is spurious `O(h11³)` waste at 1000-digit arithmetic.

`src/axion_photon.jl:727-748` `_canonical_frame` **already does this correctly** (`cholesky(Symmetric(Kinv))`, then `mul!(charge_columns, transpose(lower), Q_float)`) — use it as the template. Implementation route that avoids breaking `AxionPotential` (`src/structs.jl:35-39`): keep the `K` field, add `read.potential_factored` returning `(; L, Q, Kinv, C)`, and switch the spectrum entry points to it.

### B2. High-precision Hessians rebuilt from scratch at each precision level

`confirm_physical_mode_count` (`:2295-2305`) → `physical_mode_inertia_count` (`:2289-2292`) → `high_precision_leading_hessian` (`:1218-1235`). From `prec=1000` with `max_prec=4000` that is three full `O(h11³)` builds plus three Cholesky and three Bunch–Kaufman factorisations at 1000/2000/4000 digits — and the `W` already computed at `prec` by the caller (`:1801`) is discarded. `_confirm_window_counts` (`:2025-2056`) has the identical structure.

Three improvements, increasing in care:

1. `_confirm_window_counts:2030` recomputes at `prec`; stop that.
2. **Certified Float64 pre-screen.** Compute the Float64 leading Hessian (`:1194-1210`), take `eigvals`, and use a backward-error bound `‖ΔW‖₂ ≤ γ‖W‖₂` to certify the inertia count when no eigenvalue lies within the bound of the shifted threshold. Escalate to Arb only for the ambiguous band. `IntervalArithmetic` is already a declared, unused dependency — this is what it is for.
3. Increment precision by 1.5×, not 2× (`:2298`, `:2039`).

> ⚠️ **Do not lower the default `prec` blindly.** The threshold `10^(2(threshold_log10 − mass_offset))` sits ~600 orders of magnitude below `‖W‖`, so the shifted matrix genuinely needs the working precision. The safe optimisation is the certified screen, not a smaller `prec`.

### B3. Instanton Hessian assembly ignores charge sparsity

Dense `h11²`-per-instanton loops at `src/generate.jl:834-842, 1186-1188, 1205-1207, 410-428`; `src/inflation_points.jl:209-216`; `src/minimizer.jl:131-139, 155-160`. For h11=491 with n ≈ 1.2×10⁵ the dense form is ~2.9×10¹³ operations — infeasible.

`high_precision_leading_hessian` (`:1224-1232`) already builds a per-instanton support list; `select_quartic_backend` (`:1710-1716`) already measures density and switches at 10%; `diagonal_quartics` (`:1733-1746`) already has the `nzrange` pattern. Precompute a `SparseMatrixCSC` view of `Q` once and iterate `nzrange`.

### B4. `Matrix{Rational}` is abstract-eltype, and exact `inv` overflows

`Rational` unqualified is a `UnionAll`. Occurrences: `src/structs.jl:260, 283-285, 291, 307-308`; `src/generate.jl:1034, 3051, 3065, 3068, 3097, 3231, 3243-3244, 3314, 3316-3317, 3432, 3435, 3454, 3457, 4048, 4133, 4145`; `src/jlm_reduced.jl:52, 82, 187-188`.

More seriously, `αmatrix` (`:3065`) and `_catastrophe_alpha` (`src/jlm_reduced.jl:81-82`) call `inv` on them:

```
n=20  Matrix{Rational} inv → OverflowError    Matrix{Rational{BigInt}} inv →  11.2 ms
n=40  → OverflowError                          → 111 ms
n=60  → OverflowError                          → 604 ms
```

What is needed is `α = (Qhat⁻¹ Qbar)'` — a **solve**, not an inverse. `Nemo.solve` over `QQ` is fraction-free (Hadamard-bounded coefficient growth); `hnf_with_transform` over `ZZ` is faster still when only the lattice matters.

> Note that `:3066, 3069-3070, 3240, 3317, 4049, 4134` apply `abs(x) < 1e-4` thresholds to **exact rationals** — a numerical hack inside an exact computation, and a plausible source of wrong lattices. A fraction-free solve makes those thresholds unnecessary. `vacua_SNF` (`:3448-3467`) and `basis_snf` (`:3426-3446`) also form `inv(QᵀQ)Qᵀ·Tparallel` — normal equations in exact arithmetic square the coefficient bit-length.

### B5. `minimizer.critical_points` inner loops should be BLAS

`src/minimizer.jl:128-161`. Three closures, each allocating `arguments = twoπ .* (q' * θ) .+ phase` fresh (`:130, 144, 154`) and running scalar `O(n²p)`/`O(np)` loops. Driven by `nlsolve` with `starts = 4096` (`:92`), or `100_000` from `jlm_reduced.critical_ensemble` (`src/jlm_reduced.jl:314`).

`q[i,j] * scaled_amplitudes[i,j]` is θ-independent. Precompute `QA = q .* scaled_amplitudes`, then the gradient is one `gemv` and the Hessian one `gemm`, with `args`/`sinbuf`/`cosbuf` preallocated in a closure-captured workspace. Also makes the loop threadable.

Two more in the same function:

- `:191` — the duplicate check is `O(R²·n)` with a fresh broadcast allocation per comparison. At `starts = 100_000` this becomes the bottleneck; use a torus grid hash at cell size `merge_tolerance`.
- `:184` — allocates a new `n`-vector per start; reuse a buffer.

### B6. `schur_physical_basis` is `O(h11⁴)` per iteration in Arb

`src/generate.jl:1671-1691` builds `S = Hermitian(Matrix(A) - coupling * ((Matrix(C) - eigenvalues[i]*I) \ coupling'))` — a fresh `O(h11³)` generic solve at arbitrary precision — plus a full `eigen(S)`, for each eigenvalue, with `maxiter = 100` (`:1798`).

`Matrix(C) - eigenvalues[i]*I` differs between `i` only by a shift. Eigendecompose `C` once as `V Λ V'`; every resolvent becomes two matmuls with a diagonal between. Also `:1690`/`:1851` perform `physical_count` separate matvecs where one gemm suffices, and `:1846` re-materialises `Matrix(W)` every iteration.

### B7. Redundant recomputation across the spectrum entry points

| Duplicate | Locations |
|---|---|
| `cholesky(K)` | `:1470`, then again inside `:1189`/`:1233`; `:1814` after `:1801` |
| `Matrix(Q')/Kls'` and `Matrix(Qtilde')/Kls'` | `:1476-1477` — `Qtilde`'s columns are a subset of `Q`'s, so `Qleading` is a **row slice** of `Qcanonical`; have `LQtilde` return the `selected_indices` it already computes at `:2636` |
| `instanton_scale_blocks` | `:1360` and `:1373` — three `sortperm`s of the full instanton list |
| `instanton_hierarchy_diagnostics` | `:1818` then `:1823`; `:2102` then `:2105` |
| `LQtilde` | `pq_spectrum:1472`, then again inside `pq_spectrum_save:2474` |
| `_leading_det_qtilde` | `:3016`, `:3036`, and once more inside `foreach_leading_critical_branch:2919` |

### B8. `leading_lattice_offsets` is quadratic in `|det(Qtilde)|`

`src/generate.jl:2766-2778`: `_contains_torus_point` (`:2731-2737`) linearly scans all accepted points per candidate → `O(det² · h11)`, with `det` routinely 10³–10⁵.

Better: the offsets are exactly the coset representatives of `ℤⁿ / Q̃ᵀℤⁿ`, so compute them from the **Smith normal form** (`Nemo.snf_with_transform`, already used at `:3430`/`:3452`). No search, no floating-point `mod`, and it removes the `tolerance = 1e-8` heuristic and the `@warn` at `:2780`.

### B9. `read._canonicalize_generated_potential` is `O(k³)`

`src/read.jl:240-242` and `:254-256` evaluate a prefix sum **inside** the double loop over `(first, second)`. Precompute the offsets once. The duplicate-column search at `:196-221` is `O(k²·h11)` — hash the columns (`_charge_vector_hash`, `src/generate.jl:153-160`).

## 3. Priority C — nice to have

- **C1.** `filestructure` path helpers do filesystem I/O and ENV lookups per call: `cyax_file` (`:521`) → `geom_dir_read` (`:414-436`) → `present_dir()` → `resolve_data_dir()` (`:85-114`), plus 1–3 `localARGS()` calls and an `isdir` stat. `localARGS` is type-unstable — `@code_warntype` gives `Body::UNION{STRING, VECTOR{STRING}}` (`:21-27`). Cache the resolved root in a `Ref{String}`.
- **C2.** `paths_cy()` re-reads an HDF5 file on every call (`:285-303`); `jlm_vacua_db` calls it **four times** (`src/generate.jl:4180-4184`), `h11lst` once per invocation (`:312`). Memoise.
- **C3.** `pseudo_L` (`:507-509`) builds matrices by splatted `vcat` of one-row matrices; `L4` (`:510`) is computed and never used.
- **C4.** `generate.jacobian` (`:526-536`) returns a `Matrix` on one branch and a dynamically-constructed `SVector` on the other (`size(grad,1)` is not a compile-time constant).
- **C5.** `if @isdefined h11 else h11::Int = size(Q,2) end` at `src/generate.jl:3308, 3348, 3697, 3773, 3849, 3880, 3902, 3931, 3953, 4011, 4040, 4125` and `src/minimizer.jl:420, 452, 606`. `@isdefined` on a local always returns `false` here, so this is a no-op that additionally forces a `Union{Int,Nothing}` boxed slot.
- **C6.** `norm2`/`norm2minus1` (`:3885-3887, 3906-3909, 3935-3938, 3957-3961`) index sparse matrices column-by-column with `Ω[:, i]`, allocating each time; use `nzrange`/`nonzeros`. The `ifelse(column == true, axes(Ω,2), axes(Ω,1))` evaluates both branches and is type-unstable when the axes differ.
- **C7.** `ωnorm2` (`:3128-3132`) computes `Qhat[:, i]` three times per column.
- **C8.** `_exact_rank_columns` (`src/axion_photon.jl:533-536`) copies the charge matrix and builds a fresh `Nemo.matrix` per binary-search probe (`:538-555`). Build the `ZZMatrix` once and slice.
- **C9.** Abstract struct fields in `src/structs.jl`: `GeometricData.h21::Integer` (`:26`), untyped `Solver1D`/`SolverND` (`:294-304, 316-326`), `BasisSNF.volume::Number`/`.basis::Matrix`/`.id_coords::Matrix` (`:310-314`), unparameterised `ProjectedQ.Ωperp` (`:266-268`), `Canonicalα.α::Matrix{Rational}` (`:283-284`), `Min_JLM_*.N_min::Integer` (`:328-346`).
- **C10.** `jlm_reduced` stores `Q_reduced` sparse (`:298`) then densifies it at `:333`. `_symmetry_multiplicity` (`:52`) builds `Matrix{Integer}`.

## 4. I/O and data layout

**Julia.** The access pattern is sound — `h5open(...) do file ... end` with a single read, and `_read_dataset` (`src/read.jl:13-15`) type-asserts the object. Three issues:

1. **`deflate=9` on every write** (`src/generate.jl:997-1019, 2482-2489, 3564-3596, 4237-4243`; `src/jlm_reduced.jl:389-396`; and 30+ datasets in the Python generator). gzip-9 is roughly 3–5× slower to write than gzip-4 for typically <10% extra compression on float arrays. **shuffle + deflate-4** usually beats deflate-9 alone on both ratio and speed for `Float64`.
2. **No explicit chunking.** With compression on, h5py auto-chunks for a generic access pattern, but the Julia readers always read whole datasets (`src/read.jl:21-22, 33-58, 157-172`). A single chunk covering the dataset (or an `(h11, all)` column slab for `Q`) minimises decompression overhead. `src/axion_photon.jl:241-263` already does a deliberate 4096-column chunked read with a reused buffer — a good template.
3. `scripts/batch_physical_spectrum.jl:253-257` opens the summary CSV with `open(path,"a")` and flushes **per geometry**. Open once per run.

**Python.** `stable_hash` (e.g. `scripts/glimmers_raw_frst.py:82-87`) does `json.dumps(array.tolist())` before SHA-256. For `kappa` at h11=491 that is a multi-hundred-thousand-element Python list round-trip per hash. `np.ascontiguousarray(a).tobytes()` with an explicit dtype/shape prefix is 50–200× faster — but **it changes the digest**, so it is a provenance-schema change requiring a version bump.

## 5. Parallelism

**Current state.** All parallelism is `Distributed`/`pmap` at script level (`scripts/optim_with_phases.jl:50,83`; `Qeff.jl:101,130`; `hilbert.jl:98,130`; `top_geom_missing_h11.jl:112,199,212`; `geometries_hilbert.jl:78,109`). There is **no** `Threads.@threads`/`@spawn` anywhere in `src/`. `scripts/validate_vacua_stage4_5.jl:291-292` is the only place touching `BLAS.set_num_threads`.

**Hazards.**

1. **BLAS oversubscription.** `pmap` over N workers, each with default OpenBLAS threading, gives `N × nthreads` compute threads. Add `exeflags=["--project=…", "-t1"]` and `@everywhere BLAS.set_num_threads(1)` in every `addprocs` script (`hilbert.jl:20`, `geometries_hilbert.jl:20`, `top_geom_missing_h11.jl:20`). On the Python side, `ProcessPoolExecutor` (`generate_geometric_data_multitriangulation.py:3867, 3958`) needs an `initializer=` setting `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`; several scripts *record* these variables (`generate_h11_491_frsts.py:517-519`) but nothing *sets* them.
2. **HDF5 is not thread-safe.** All existing use is per-process, which is correct; keep it that way.
3. **`setprecision(ArbFloat; digits=prec)` is a global.** Used at `src/generate.jl:822, 1219, 1808, 2126, 2352, 3609` and `src/minimizer.jl:217, 259, 303`. Unlike `BigFloat` (task-local since Julia 1.9), ArbNumerics' `setprecision` mutates a package-level `Ref`. **Threading any Arb path is currently unsafe.** Fixing A1 (explicit `ArbFloat{bits}`) removes the problem.
4. **`pq_contracted_log!` shares mutable buffers** allocated once at `:1503-1504`.

**Where to add parallelism**, highest value first:

- `src/generate.jl:1515-1523` — the ~`1.5 h11²` independent `pq_contracted_log!` calls. Embarrassingly parallel; needs per-thread buffers only.
- `:1617-1629`, `:1897-1908`, `:1972-1986` — the `signed_quartic` component loops, the dominant cost of those functions.
- `:895-918` — the fused instanton loop in `hp_spectrum`; parallelise over `i` so each thread owns a disjoint slice.
- `src/minimizer.jl:182-195` — the `starts`-fold `nlsolve` restart loop. 4,096–100,000 fully independent Newton solves; the only shared state is `roots`/`residuals`. **The single largest available speedup in the vacua pipeline.**
- `src/inflation_points.jl:464-498` — `compare_precision` over a candidate list.

**Load balancing.** `pmap` over geometries has a long-tailed runtime distribution (cost grows like `h11³`–`h11⁴`). Sort the geometry list by **descending h11** before `pmap` so expensive tasks start first — a one-line change per driver, typically 20–40% wall clock on a heterogeneous batch.

## 6. Python / CYTools scripts

- **P1.** `_reconstruct_intersection_geometry` (`scripts/generate_geometric_data_multitriangulation.py:4168-4181`) is a pure-Python loop over every nonzero intersection number, allocating a `set` of up to 6 permutation tuples per row. At h11=491 the COO table has 10⁵–10⁶ rows, and this runs inside `reconstruct_potential_from_reference` (`:4236`), i.e. **per EFT row**. Vectorise the S₃ orbit with `np.add.at` over the six index permutations, masking duplicates once. Expect 50–500×.
- **P2.** `np.einsum` without `optimize=True` (`:4271-4273`) runs a naive C nested loop with no BLAS. Rewrite as `tmp = effective_cone[pair_i] @ inverse_metric` followed by `np.einsum("ai,ai->a", tmp, effective_cone[pair_j])`. The fancy-index gathers also materialise `(P, h11)` copies — for `nq ≈ 500`, h11=491 that is ~490 MB each; chunk the pair loop.
- **P3.** `pair_i`/`pair_j` built with a Python double loop (`:4257-4262`, `:4746-4749`); use `np.triu_indices(direct_count, k=1)`. In `factorized_manifest_for_paths` the loop exists only to compute a length that is `n(n-1)//2` in closed form.
- **P4.** `ProcessPoolExecutor` submits every task up front (`:3868`), so all polytope payloads are pickled and held in the parent before any work starts. The `run_batches` path (`:3958-4060`) already uses bounded `wait(FIRST_COMPLETED)` refill — apply the same to `_run_tasks`.
- **P5.** `generate_and_save_geometry` calls `cy.compute_divisor_volumes` at `:1253, 1256, 1920, 1932, 2093, 2094`, `cy.compute_cy_volume` at `:1250, 1943, 2085`, and `cy.compute_inverse_kahler_metric` at `:1262, 1925` — several at the same Kähler point. CYTools does not memoise; cache per `(cy, point.tobytes())`.
- **P6.** `generate_stage1_raw_frsts.py:679`, `generate_stage2_eft_reference.py:700`, `replenish_stage1_raw_frsts.py:550` each spawn a subprocess per unit of work. A CYTools import costs seconds; batch if these are per-geometry.

## 7. Benchmarking and precision-regression tests

`BenchmarkTools` is **not** a dependency. Add it to `[extras]` and the `test` target. Promote the existing ad-hoc harnesses (`scripts/benchmark_spectrum_windows.jl`, `benchmark_hybrid_scaling.jl`, `benchmark_inflation_scalability.jl`) to a `benchmark/benchmarks.jl` `BenchmarkGroup` so `PkgBenchmark.judge` can gate PRs.

Tests to add first — each would have caught a real defect above:

1. **Quartic index/value alignment.** ✅ Added in PR #83.
2. **Ill-conditioned `K` round-trip.** Construct `Kinv` with `cond = 10^{6,10,14}` and assert the spectrum matches a `BigFloat(400)` reference to a stated relative tolerance. On current code the `10^10` case throws `PosDefException`; after B1 it should pass.
3. **`LQtilde` vs `LQtildebar` equivalence** over a stored corpus — the safety net for A11. ⚠️ This will also expose that `LQtilde` uses a **floating-point** rank test (`:2589`) while `LQtildebar` uses **exact** `Nemo.nullspace`; they can genuinely disagree on ill-conditioned integer charge matrices, and the fast one can accept a rationally-dependent column. `src/axion_photon.jl:605-725` already solves this properly — reuse it in `generate.LQtilde` for exactness *and* speed.
4. **Inertia-count stability** across `prec = 200/400/800` as a golden file — the guard for any B2 change.
5. **Log-sum-exp equivalence** to <4 ulp on adversarial inputs (log-scales spanning `[-1500, 0]`, mixed signs, near-exact cancellation).
6. **`gradient_flow` bit-identity** before/after the A12 refactor.
7. **Allocation budgets** so A1/A2 do not regress.

## 8. Summary of expected gains

| # | Change | Measured / estimated | Risk |
|---|---|---|---|
| 0.1 | Fix `λ31`/`λ22` index ordering | correctness | — |
| A1 | Concrete `ArbFloat{P}` containers | **3.75×** on the Arb kernel at prec=100 | none |
| A2 | Preallocate index matrices | **2542×** on that step | none |
| A3 | Hoist `cos.(x'Q)` out of the Hessian loop | ~`h11²` fewer matvecs | none |
| A4 | Cache `log|Q|` in the quartic contraction | **21×** on that loop | none |
| A5 | Max-shifted LSE instead of sort | `O(n log n)` → `O(n)` | low |
| A9 | Thin-QR `opnorm(AB')` | **260×**, 7.1e-16 agreement | none |
| A11 | `LQtildebar` → `LQtilde`-based selection | **901×** time, **715×** memory at h11=40 | check `Qhat` semantics |
| A12 | Gradient-only line search / RK4 RHS | removes ~240k Hessians per `gradient_flow` | none |
| A13 | Hoist `grad_std` out of `minimize` | ~100× fewer gradient evaluations | none |
| B1 | Cholesky-of-`Kinv` whitening | **200×** more accurate; removes a hard `PosDefException` at cond ≥ 1e10 | medium |
| B2 | Certified-Float64 inertia pre-screen | avoids 2–3 full Arb rebuilds per geometry | medium |
| B4 | Fraction-free exact solve | removes the `OverflowError` cliff at n≈20 | medium |
| B5 | BLAS-ify `critical_points` | `O(n²p)` scalar → one gemm | low |
| P1 | Vectorise the COO intersection reconstruction | 50–500× | low |

**Recommended order.** §0.1 first (it invalidates saved `quart31`/`quart22` labels) — done. Then **B1**, the largest accuracy defect and a latent crash at large h11. Then **A1 + A2 + A4 + A9 + A11** together: a large multiplier on the spectrum path for a few hours of mechanical work with no accuracy exposure.

---

## Cross-cutting observation

The repo's greatest asset is its **epistemic discipline** — nearly every docstring and README distinguishes what was computed from what was assumed, and terminal statuses are never collapsed. The greatest structural risk is the **three independent "Stage N" vocabularies** (Part I §2), which required disambiguation three times during this review.
