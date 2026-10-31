# Scientific scope and conventions

This document says which physics this contribution takes from CYAxiverse (branch `vmm`) unchanged and which analysis it adds. It also lists the conventions that every result inherits.

## 1. What is contributed

The relic-abundance analysis of physical axion spectra that CYAxiverse vmm has already computed:

| capability | where |
|---|---|
| Validated relic input from existing vmm HDF5 spectra, with an explicit vacuum-validity state per geometry | `engine/input/` |
| Canonical per-mode tables (A1 contract checks) | `engine/relic/01_build_mode_table.py` |
| Reheating-dependent misalignment abundance per mode (configurable T_reh). The classes are POST_REHEATING and PRE_REHEATING (CDM-eligible, summed), FROZEN and LATE (census only), and inflation-excluded | `engine/relic/compute_relic_abundance.jl` |
| Per-geometry and per-h¹¹ summaries, with the abundance-validity class of each geometry | same, `relic/tables/` |
| Schema validation, an independent float64 implementation (C) on every row, and a 50-digit reference (B) on a stratified sample | `engine/relic/03_*`, `04_*`, `independent_relic.py`, `reference_mp.py` |
| Common-initial-misalignment-angle analysis (exact interval overlap, TIGHT band) with an independent Python cross-check | `engine/theta/` |
| Optional producers that call the upstream vmm scripts to generate geometries, spectra and validated vacua from a polytope manifest | `producers/` |
| The Shoroq thesis sample as a reproducible reference case | `references/shoroq/` |

**Not contributed, and not changed:**
- the geometry generator (G2), the axion potential (charges Q, amplitudes and signs L, pair terms);
- the Kähler metric and the physical-spectrum solver (`pq_hybrid_physical_spectrum`), with its mass, quartic and decay-constant definitions;
- the vacuum-finding code.

This contribution adds no file under `src/` and changes no upstream file.

## 2. Physics taken from CYAxiverse vmm unchanged

**Spectra.**
- The inputs are taken as written by `scripts/batch_physical_spectrum.jl`: masses `spectrum/physical/m`, quartics `lambda_self_log10` / `lambda_self_sign`, and `fpert_log10`.
- **Nothing is rescaled.** The relic stage uses f = f_pert = m/√|λ_self| (C = 1), as stored.

**Theoretical framework.** The axion potential and its vacua follow the CYAxiverse vmm implementation of the framework of N. Gendler et al., *Axion Minima in String Theory*, arXiv:2309.01831:
- V(θ) = Σ_a Λ_a [1 − cos(2π q_a·θ)], with single-instanton terms and pair (cross) terms;
- amplitudes and signs are computed upstream in `scripts/generate_geometric_data_multitriangulation.py`.

## 3. Adopted conventions

### 3.1 Pair-instanton terms (adopted modelling choice)

- The analysis uses the vmm pair-term amplitudes, signs and phase prescription **exactly as implemented upstream**. This is an intentional modelling choice for this study.
- The sign of a pair term depends on instanton phases (for example Pfaffian and W₀ phases) that this analysis does not determine from first principles. **No claim is made that these phases are uniquely fixed.**
- A consequence in the reference sample is that 46 of 4491 Shoroq geometries have tachyonic directions at θ = 0.
  - Their spectra are evaluated at validated minima of the full potential (`producers/vacuum`, `METHODOLOGY.md`).
  - They are not excluded and not altered.

### 3.2 Kinetic normalization (open convention question, documented)

**The vmm convention.** The physical-spectrum solver uses the kinetic term

  L_kin = −½ M_pl² θ̇ᵀ K θ̇, with K = (stored `Kinv`)⁻¹ = ∂_T∂_T̄𝒦, T = τ + iθ, 𝒦 = −2 log 𝒱.

**The literal N = 1 supergravity form.** The canonical kinetic term −M_pl² 𝒦_{i\bar j} ∂Tⁱ∂T̄ʲ corresponds, in the same ½-normalized form, to a metric 2K. The two conventions differ by a factor of two in the kinetic metric. The vmm convention is therefore **not** identical to the literal supergravity normalization.

**Status.**
- This is an existing upstream convention question, awaiting clarification from the CYAxiverse author (V. M. Mehta).
- This contribution **retains the vmm normalization without modification**: no mass, quartic, decay constant or abundance is rescaled, and no alternative-normalization result is part of this branch.
- For orientation only: if the metric were 2K, every mass would scale by 1/√2, every quartic by 1/4, and f_pert and f_K by √2. The abundances, the inflation exclusions and the angle analysis would then shift accordingly.

**Interpretation.** Results produced with this code are results **in the vmm convention**.

### 3.3 Other conventions

| item | convention |
|---|---|
| units | masses in eV (log₁₀), decay constants in GeV (log₁₀), M_pl = 2.435·10¹⁸ GeV (reduced) |
| physical modes | eigenmodes with m ≥ H₀ = 2.13·0.7·10⁻³³ eV, as stored by the vmm solver. The solver returns only positive-curvature directions, so θ = 0 stability is checked separately (vacuum-validation states) |
| decay constant | f_relic = f_pert (C = 1) |
| reheating | before T_reh: modulus (matter) domination, w_R = 0. PRE_REHEATING modes follow Fuzzy Axions, arXiv:2412.12012, eqs. (2.10) + (2.15). After T_reh: radiation domination with full H(T). g\*(T) is from Saikawa & Shirai, arXiv:1803.01038 |
| inflation validity | modes with m > 3H_I,max ≈ 1.4·10¹⁴ GeV (arXiv:2412.12012, footnote 16) are not evaluated (`INFLATION_VALIDITY_EXCLUDED`). This is not a zero abundance |
| reference angle | θ_ref = π/2 is only the internal normalization of stored abundances, not a physical angle |
| DM target | Ω_DM h² = 0.12; TIGHT band 0.9 ≤ Ω/Ω_DM ≤ 1 |

All values are in `configs/methodology_v1.json` and `configs/theta_methodology_v1.json`, with their sources in `METHODOLOGY.md`.

## 4. Validity semantics (zero versus undefined)

| class | meaning | Ω_geometry | primary θ population | sensitivity population |
|---|---|---|---|---|
| `EVALUATED_POSITIVE` | every mode evaluated, CDM sum > 0 | number > 0 | yes | yes |
| `EVALUATED_ZERO` | every mode evaluated, no CDM-eligible mode (zero modes above H₀, or only FROZEN/LATE) | 0 | yes | yes |
| `PARTIAL_VALIDITY` | some modes inflation-excluded; the sum covers the valid modes only | NA (the partial sum is reported in a separate column) | **no** | yes, labelled partial |
| `INFLATION_INVALID_ALL` | every mode inflation-excluded | NA, **not zero** | no | no (listed as not evaluated) |

**Rules:**
- NA and 0 are never interchanged.
- The default (primary) angle analysis uses only fully evaluated geometries.
- The sensitivity population reproduces the original Shoroq angle analysis, including partially valid geometries with clearly labelled partial sums.
- When no geometry reaches the TIGHT band (N_max = 0), the result is `NO_MATCH`, with no representative angle.
- An empty primary population gives `EMPTY_POPULATION`.

## 5. Reference results

The Shoroq reference case covers:
- 4,700 processed polytopes, 4,491 accepted geometries and 16,313 physical modes;
- T_reh = 5 MeV, 10⁴ GeV and 10¹⁰ GeV.

It is reproduced byte for byte by `references/shoroq/tests/regression_relic_theta.sh`, given the frozen reference archive, which is held outside the repository. These results are in the vmm conventions of §3.
