# Methodology v1 and v2: scope, assumptions, validity range

The engine applies one fixed, audited methodology (approved 2026-10-01 for the Shoroq study). It is stored in
`configs/methodology_v1.json` and `configs/theta_methodology_v1.json` and enforced by the relic driver, which refuses
any configuration whose constants or switches differ. **v1 is not a general cosmology code**: it does not support
other cosmological models, other reheating equations of state, or other abundance prescriptions. What a run may choose
is listed under "Configurable" below.

## Fixed (v1)

**Spectrum input.**
- CYAxiverse vmm physical spectrum (`pq_hybrid_physical_spectrum`): eigenvalues of the leading-instanton Hessian above
  λ(H0), counted by inertia; precision 500 digits; quartics stored; schema `cyaxiverse-physical-spectrum-3`.
- Threshold H0 = 2.13·0.7·10⁻³³ eV = 1.491·10⁻³³ eV. Sub-H0 modes are out of scope.
- Decay constant: f = f_pert = m/√|λ_iiii| (C = 1). f_K and f_PQ = √(2π) f_pert are not used.

**Cosmology** (physics block of `engine/relic/compute_relic_abundance.jl`, byte-identical to the production script):
- H(T) from radiation (g_ρ, g_s of `data/standardmodel2018.dat`), matter and Λ; flat closure.
- H0 = 1.491·10⁻³³ eV (h = 0.69898, the CYAxiverse spectrum threshold); ω_m = 0.315·0.674² fixed; Ω_r0 from
  T_CMB = 2.7255 K; reduced M_pl = 2.435·10¹⁸ GeV; T_eq = 0.8032 eV.
- Changing any of these requires a new methodology version (physics block, reference B, validators), not a setting.

**Oscillation and classes** (onset at m = 3H):

| class | condition | treatment |
|---|---|---|
| FROZEN | m ≤ 3H0 | census only |
| LATE | m ≤ 3H(T_eq) | census only (formula value kept separately) |
| POST_REHEATING | 3H(T_eq) < m ≤ 3H_R | summed |
| PRE_REHEATING | m > 3H_R | summed |

- POST: Ω h² = ½ m² f² θ² [s0/s(T_osc)] / (3 M_pl² H100²).
- PRE: Sheridan et al., arXiv:2412.12012, scenario (2) (modulus domination, w_R = 0), Eqs. (2.1), (2.2), (2.10),
  (2.15); the printed closed form (2.16) is not used (2.503× its own (2.10)+(2.15)). Ω is independent of m and ∝ f²θ²T_R.

**Validity.**
- Per mode: m > 3H_I,max = 1.4·10¹⁴ GeV is excluded from the abundance (Fuzzy footnote 16), recorded as NA, never zero.
- m > m_KK (= M_pl/V^{2/3}, M_pl = 2.4·10¹⁸ GeV) and m > f_pert are flags only. No 100 eV window.

**Angles.**
- θ_ref = π/2 is only the internal normalisation of the stored abundances, not a physical, preferred or allowed angle.
- Harmonic scaling Ω(θ) = Ω(θ_ref)(θ/θ_ref)²; one common θ for every mode of every geometry.
- No lower angle cutoff (a lower cutoff was audited as unjustified); no isocurvature, H_I or anharmonic correction;
  no temperature-dependent masses.

**Dark-matter criterion (theta analysis).** R = Ω_geometry/Ω_DM; TIGHT_DM for TIGHT_DM[0] ≤ R ≤ TIGHT_DM[1],
ALLOWED_UNDERABUNDANT below, OVERABUNDANT above (never accepted). The geometry is the statistical unit.

**Abundance validity of a geometry** (approved 2026-10-30 and 2026-10-31; `geometry_abundance_validity_<label>.csv`). No
validated geometry is dropped; each has exactly one class:

| class | definition | Ω_geometry h² in `geometry_summary` | theta population |
|---|---|---|---|
| `EVALUATED_POSITIVE` | no mode above 3H_I,max, ≥ 1 mode in the CDM sum | complete abundance > 0 | primary |
| `EVALUATED_ZERO` | no mode above 3H_I,max, none in the CDM sum: no mode above H0 (`NO_MODES_ABOVE_THRESHOLD`) or all FROZEN/LATE (`ALL_MODES_FROZEN_OR_LATE`) | 0 (an evaluated empty sum) | primary, underabundant at every angle |
| `PARTIAL_VALIDITY` | some, not all, modes above 3H_I,max | partial sum over the valid modes, flagged `omega_total_excludes_invalid_modes` (NA if none of the valid modes is included); **not the complete abundance** | sensitivity population only |
| `INFLATION_INVALID_ALL` | every mode above 3H_I,max | NA (not evaluated) | neither; listed |

- `NO_MODES_ABOVE_THRESHOLD` does not mean there are no axions: lighter ones lie below the population threshold H0,
  outside the scope.
- Missing, invalid, incomplete or unverified spectra are a different case. They stop the build, or are listed as
  `EXCLUDED_*`; they never receive an abundance.
- **Primary theta population** (`final_analysis/theta/`): EVALUATED_POSITIVE + EVALUATED_ZERO, using complete abundances.
- **Sensitivity population** (`final_analysis/theta_sensitivity_partial_validity/`, `population =
  SENSITIVITY_INCLUDING_PARTIAL_VALIDITY`): primary + PARTIAL_VALIDITY with their partial sums.
- Every geometry, its class and its membership are listed in `theta/theta_population.csv`. The denominators and headline
  results of both populations are in `theta/theta_population_summary.json`.
- An Ω = 0 geometry never reaches the band: its θ thresholds are +∞ (`Inf` in CSV, `null` in JSON).
- log10 Ω quantiles in `h11_summary` and `global_summary` are taken over geometries with an included mode. These relic
  summaries are bookkeeping and include the partial sums; the population-resolved statements are those of the theta
  populations.
- **Change from the frozen 2026-10-01 methodology.** The frozen methodology kept the 329 partially valid Shoroq geometries
  (per-mode cut, partial sums) in the single theta population. The frozen theta results equal the present sensitivity
  population. The primary results without them are new (2026-10-31 decision).

**No match.** If no geometry reaches the TIGHT band anywhere on 0 < θ ≤ θ_max (N_TIGHT maximum = 0), the status is
`NO_MATCH`: no maximising interval, no representative angle, no smoothed maximum, plateau or excess (all `null`/empty).
Bootstrap resamples without match are counted (`fraction_NO_MATCH`) and carry no angle.

**Class thresholds.** A mode exactly at 3H0, 3H(T_eq), 3H_R or 3H_I,max (within 1e-12 dex) is a tie: the side is set by
the last bit of the threshold value. The relic driver accepts a solved T_osc within its bisection tolerance (|Δ ln T| ≤ 1e-10)
of T_eq for the class given by the exact threshold, and the independent verifications list such ties (`threshold_ties`)
instead of counting them as disagreements. Every other category must agree exactly.

## Configurable (run file, `DATA_CONTRACTS.md`)

| setting | default | admissible | enforced by |
|---|---|---|---|
| reheating scenarios: labels, T_reh [GeV] | (none; required) | see below | `engine/relic/00_compose_run_config.py` and the driver |
| `production_run_approved` | false | true required for more than 1000 modes | relic driver |
| `theta.theta_max` | 0.5 | 0 < θ_max ≤ 1; warning above 0.5 (harmonic error grows beyond ~2–3 %) | composer |
| `dark_matter.Omega_DM_h2`, `TIGHT_DM` | 0.12, [0.9, 1.0] | Ω_DM > 0, 0 < lo < hi, hi > 0.95 | composer |
| `optimisation.smoothing_half_widths_dex`, `primary_smoothing_half_width_dex` | [0.05, 0.1, 0.2], 0.1 | positive; primary in the list | composer |
| `optimisation.plateau_fraction`, `bootstrap_resamples`, `bootstrap_seed` | 0.8, 1000, 20261001 | (0, 1), ≥ 1, integer | composer |
| `optimisation.robust_criteria` | shift 0.3 dex, bootstrap 0.5 dex, plateau 1.0 dex | positive | composer |

Not configurable: θ_ref, θ_min (always none), the upper-boundary rule, the diagnostic TIGHT lower edges 0.8 and 0.95,
the R levels 0.1/0.8/0.95 of the reported thresholds. Output columns named `..._0p5` refer to θ_max.

## Reheating temperature: validity range (`configs/reheating_policy_v1.json`)

**Hard constraints** (the run stops):
- T_reh inside the g* table, 2.0·10⁻⁶ – 1.0·10¹⁷ GeV (no extrapolation). This also implies T_reh > T_eq = 8.0·10⁻¹⁰ GeV,
  which the driver checks independently.
- 3H_R < 3H_I,max (reheating after inflation; T_reh ≲ 6·10¹⁵ GeV).
- T_reh < T_search_max = 10¹⁸ GeV.

**Warnings** (recorded in `relic/scenario_checks.json`, the computation is unchanged):
- T_reh < 5 MeV: at or below the BBN / neutrino-decoupling lower bounds of a few MeV quoted in the literature.
- 5 MeV ≤ T_reh < 10 MeV: a note (within a factor of 2 of those bounds). The Shoroq 5 MeV scenario carries this note.

The derived scenario quantities (H_R, 1+z_R, g(T_R), the PRE constant) are computed by the 50-digit reference
implementation B; for the three Shoroq temperatures they reproduce the frozen configuration bit for bit.

**Convention note (v1, inherited).** The "today" entropy and energy degrees of freedom g_s0, g_ρ0 are the values of the g* table clamped at its lower edge
(T_CMB = 2.35·10⁻¹³ GeV lies far below the first table entry, 2·10⁻⁶ GeV): g_s0 = 3.9309, g_ρ0 = 3.3831. Other conventions differ by about 0.5 % in g_s0 and enter the
normalisation of every POST and PRE abundance equally (Ω ∝ g_s0). It is part of the frozen v1 convention and is kept unchanged in v2.

## Methodology v2: constant pre-reheating equation of state w_R

Methodology v2 (`configs/methodology_v2.json`, run files `cyaxiverse_relic_run/2`, output schema `configs/relic_output_schema_v2.json`,
policy `configs/reheating_policy_v2.json`) lets each reheating scenario carry its own constant equation of state `w_R` of the epoch
before reheating. **Everything else is methodology v1, unchanged**: the spectra, f = f_pert, the cosmology constants, the onset
3H = m, the POST_REHEATING calculation, the classes and class thresholds (3H0, 3H(T_eq), 3H_R), the inflation cut, the PRIMARY and
PARTIAL_VALIDITY populations, θ_ref = π/2 as a normalisation, Ω ∝ θ², the common-angle analysis. At `w_R = 0` the engine calls the v1
function itself, so a v2 run at `w_R = 0` reproduces v1 bit for bit apart from the extra `w_R` column (tested, `tests/test_wR_general.py` D2–D4).
Methodology v1 stays fixed at w_R = 0: a v1 run file with `w_R` is refused, and a hand-edited v1 configuration with a per-scenario `w_R` is
refused by the engine.

**Model.** p_R = w_R ρ_R before the instantaneous transition to radiation at T_R, so ρ_R ∝ a^(−3(1+w_R)) and
H(z) = H_R [(1+z)/(1+z_R)]^(3(1+w_R)/2) for z ≥ z_R (Fuzzy Axions, arXiv:2412.12012, eq. (2.2)); after T_R the v1 cosmology holds.
H_R is defined by ρ_r(T_R) = 3 H_R² M_pl² (radiation only), (1+z_R) by entropy conservation, (2.1); both as in v1.

**Derivation.** For a mode with m > 3H_R the onset 3H(z_a) = m lies in the pre-reheating epoch:

  (1+z_a) = (1+z_R) (m / 3H_R)^(2/(3(1+w_R)))                                   (2.15)
  Ω h² = ρ_a(today)/(ρ_c/h²), ρ_a(onset) = ½ m² f² θ², ρ_a ∝ a⁻³ after onset (the number density is conserved through the instantaneous
  transition and the entropy-conserving radiation era):
  Ω h² = m² f² θ² / [2 (1+z_a)³ · 3 M_pl² H100²]                                 (2.10)

Dividing by the same expression at w_R = 0 gives the **relative law** (full normalisation retained; both factors have the same dimension
and the same (1+z_R)³ matching):

  Ω(w_R) / Ω(0) = (m / 3H_R)^(2 w_R/(1+w_R)),   with Ω(0) = (9/2) f² θ² H_R² [s0/s(T_R)] / (3 M_pl² H100²), independent of m.

Mass dependence at fixed f, θ, T_R, d ln Ω / d ln m = 2w_R/(1+w_R): 0 (w_R = 0), −1 (−1/3), −2 (−1/2), +1/2 (+1/3). These are examples; every
w_R in the domain is supported (e.g. 0.17, −0.42, 0.77 are in the validation matrix). At m = 3H_R the onset is at reheating for every w_R: (1+z_a) = (1+z_R),
Ω is w_R-independent and continuous with the POST value (they differ only by the matter and Λ terms that the POST branch's full H contains and
the radiation-only H_R of (2.2) does not, relative O(T_eq/T_R)). The class boundary 3H_R does not depend on w_R, so **classes, inflation validity,
inclusion and all POST, LATE and FROZEN values are independent of w_R at fixed T_R** (tested).

**Supported domain (enforced).** `-1 < w_R <= 1`, a finite number (booleans, strings and null are refused; a missing `w_R` means the documented default 0.0).
- w_R > −1 is required: H ∝ a^(−3(1+w_R)/2) must fall as the universe expands so that 3H = m is reached before reheating for m > 3H_R. w_R = −1 is a
  constant-H (de Sitter) phase with no onset; w_R < −1 has H growing with a, so the "epoch before H_R" would have smaller H.
- w_R ≤ 1: a causal fluid (sound speed ≤ c); w_R = 1 is kination.
- A (T_reh, w_R) pair may appear once per run (two labels for identical physics would be ambiguous). T_reh limits are those of v1.

**Warnings** (recorded in `relic/scenario_checks.json`, printed by the composer, no number changes):
- w_R < −1/3: the pre-reheating epoch is **accelerating** (ä > 0; w_R = −1/3 itself is the coasting borderline). Mathematically executable, but it is
  an inflation-like continuation, not a conventional reheating epoch; the instantaneous switch to radiation then needs a microphysical interpretation that a
  constant-w_R prescription does not give. Conventional reheating epochs have w_R ≥ 0 (matter-like, w_R = 0) up to 1/3 (radiation-like, noted) and kination (1).
- w_R < −0.9: 1+z_a grows very steeply (exponent 2/(3(1+w_R)) up to 6.7 and more); see numerical range.
- A constant effective w_R is not a complete reheating model. The sudden-onset systematic grows with the expansion rate of the background (below).

**Numerical range.** For w_R ≠ 0 the engine evaluates ln Ω = ln[m² f² θ²/(2 ρ_c/h²)] − 3 ln(1+z_R) − [2/(1+w_R)] ln(m/3H_R) in logarithms (no overflow of
(1+z_a)³). An abundance or (1+z_a) outside the float64 normal range (|ln Ω| > 708) **stops the run with a message naming the mode; nothing is
clamped, replaced by zero, or written as a bound.** For T_R = 5 MeV, modes up to 3H_I,max are in the PRE branch with m/3H_R up to about 10³⁶, so
the heaviest valid mode (m = 1.4·10¹⁴ GeV, f = 10¹⁶ GeV) has ln Ω below −708 for w_R below about −0.82 (T_R = 5 MeV), −0.86 (10⁴ GeV) and −0.92 (10¹⁰ GeV); lighter modes reach the limit later (tested: w_R = −0.99 at 5 MeV stops; w_R = −0.5 at 5 MeV runs). Remedy: use a w_R closer to 0, a higher T_R, or
a smaller set of heavy modes. The log-form engine result and the w_R → 0 limit agree with the exact ratio to about 10⁻¹⁴ (w = 10⁻¹², 10⁻⁶ tested).

**Verification (independent of the engine's closed form).** `tests/test_wR_general.py`: the onset is found by bisection on H(z) of (2.2) in 50 digits and Ω from (2.10)
(`omega_by_root_finding`), for every PRE row of a 21-scenario run (3 T_R × w_R ∈ {−1/2, −1/3, 0, 1/3, 0.17, −0.42, 0.77}) to 7·10⁻¹⁴;
mass exponents, the normalisation at m = 3H_R, w_R → 0 and PRE/POST continuity at 50 digits; the independent verifiers C (float64, every row) and B (50-digit
samples and whole geometries) are driven by the scenario's w_R read from the run configuration; theta thresholds and N_max are recomputed from the geometry totals.

**Reading w_R results.** The scenario's `w_R` is a column of every table and of `relic/run_config.json`; `w_R_defaulted` there says whether it was stated or defaulted. Floats are written with
`%.17g` (−0.3 appears as −0.29999999999999999). A negative-w_R scenario also records a note on the systematic below in `relic/scenario_checks.json`. Partial sums of `PARTIAL_VALIDITY` geometries
are sensitivity-only quantities and change by many orders of magnitude with w_R; they are not abundances.

**Sudden-onset systematic of the pre-reheating prescription** (`python tests/ode_onset_check_wR.py`; idealised constant-mass, pure power-law setups, harmonic, small θ_i;
not applied, not universal): exact/sudden = 0.403 (w_R = 1/3, a ∝ t^(1/2)), 0.330 (0.17), 0.250 (0), 0.131 (−1/4), 0.094 (−1/3), 0.061 (−0.42), 0.035 (−1/2); i.e. the
sudden-onset value is 2.5, 3.0, 4.0, 7.6, 10.6, 16.5, 28 times larger. The approximation error therefore **grows rapidly as w_R decreases**, and the v1 statement
that the TIGHT band does not cover the approximation uncertainty holds a fortiori. Results at negative w_R are results of the stated prescription, not precision predictions.

## Vacuum validity of the input spectra

A spectrum is valid only at a minimum of the axion potential. At θ = 0 the production mass matrix can have tachyonic
directions (46 of 4491 Shoroq geometries: negative-sign pair terms). States, per geometry:

| state | meaning | usable |
|---|---|---|
| `VERIFIED_THETA0_INERTIA` | θ = 0 inertia scan of the potential in the file: no tachyonic direction with \|m\| ≥ H0, and its count above H0 equals the stored mode count | yes (ORIGINAL_SPECTRUM) |
| `VERIFIED_VACUUM_REPAIR` | spectrum about a leading-branch minimum validated by `producers/vacuum` on the full potential at ~500 and ~1000 digits | yes (VACUUM_CORRECTED) |
| `DOCUMENTED_EXTERNAL` | vacuum established elsewhere; method, reference and evidence files (sha256 re-checked) given; not re-verified here | yes, reported as such |
| `TACHYONIC_AT_THETA0` | the scan finds tachyonic directions with \|m\| ≥ H0 | no: repair with `producers/vacuum` |
| `UNVERIFIED` | no potential data for the scan, scan failure, count mismatch, or no information | no |

Unusable geometries stop the build (`vacuum_policy.on_unusable = "refuse"`, default) or are excluded and listed
(`"exclude"`). Sub-H0 tachyonic directions are out of scope, as in the spectrum. For VERIFY entries the record carries
diagnostics (`a1/vacuum_validation.csv`): full-Hessian negative inertia, resolved PQ eigenvalues and the leading-scale
span; `subH0_inertia_resolved = false` when the span exceeds the 500-digit working precision or not every eigenvalue was
resolved (the minimum property below H0 is then not established; the certificate above H0 is unaffected).

## Scientific validation that runs for every sample

- A1: per-file spectrum contract (settings, schema, completeness, m ≥ H0, f_pert = m/√λ, mode indices), unique keys,
  counts equal to the manifest, vacuum states, round-trip-exact canonical tables.
- Relic driver: constants and derived cosmology against the configuration; T_osc consistency with the class.
- Output validation: schema, NA rules, state/inclusion invariants, aggregates recomputed from the per-mode table,
  identical mode sets across scenarios.
- Independent verification: implementation C (float64, separate code) on every row; implementation B (50 digits) on a
  stratified sample and on complete geometries.
- Theta stage: input checksums, every mode re-read from its HDF5 file, every included Ω recomputed by the physics block,
  PRE constant per scenario, and an independent Python cross-check of the exact interval analysis.

Expected population counts (geometries, modes, statuses, census, flag counts) are checked only for a reference case
(`references/shoroq/configs/expectations.json`).

## Accuracy and systematic uncertainty of the abundance prescription

Three kinds of uncertainty must not be mixed:
- **Numerical precision** (negligible). The Julia driver agrees with an independent 30-digit implementation to about 5e-7 relative
  (the spread between g* interpolation schemes) on per-mode and per-geometry abundances; class assignments agree exactly.
- **Model assumptions** (fixed by v1, listed above): ΛCDM with the vmm H0, f = f_pert, instantaneous reheating, w_R = 0 for the PRE
  branch, no temperature-dependent mass, no isocurvature, no anharmonic or H_I correction. They are choices, not errors.
- **Approximation (systematic) error of the sudden-onset estimate.** The abundance is the comoving number density at the instant
  3H(T_osc) = m, with the harmonic Ω ∝ θ_i². A direct integration of θ'' + 3Hθ' + m²θ = 0 gives a different late-time number density:

  | test setup (constant m, pure power-law background, harmonic potential, small θ_i) | exact / sudden-onset |
  |---|---|
  | radiation domination (a ∝ t^(1/2)) | 0.403 (the sudden-onset value is 2.48 times larger) |
  | modulus domination, w = 0 (a ∝ t^(2/3)) | 0.250 (the sudden-onset value is 4 times larger) |

  These two numbers belong to those specific idealised setups (reproduced by `python tests/ode_onset_check.py` (scipy only) and by
  any ODE solver). They are **not universal correction factors**: they change with the background
  (g*(T) variation, the matter-radiation transition, the transition at T_R), and the PRE and POST branches differ. The sudden-onset estimate
  is an order-one-accurate approximation, standard in the literature; it is appropriate for comparing geometries and for scalings
  (Ω ∝ θ_i² f² and the m-dependence), not for percent-level predictions.
  The anharmonic (cosine) correction to the harmonic scaling is small only for small angles: 0.1% at θ = 0.1, 2–3% at 0.5, about 10% at 1
  (the θ_max warning above 0.5 reflects this).
- **Consequence for the dark-matter criterion.** The TIGHT band 0.9 ≤ Ω/Ω_DM ≤ 1 is a 10% window. It does **not** cover an approximation
  uncertainty of a factor 2.5–4. In the common-angle analysis a factor c in Ω is equivalent to a factor 1/√c in the angle θ
  (Ω ∝ θ²): the selected angle and the interval widths in dex move by 0.5 log10 c, while the *count* of geometries in the band at a
  given angle shifts accordingly. The outputs of this package are therefore results of the stated prescription; **no output is rescaled
  by these factors**, and the established method has not been replaced by an ODE-based one.

## Known limitations

- **Scope.** Fixed cosmology. Methodology v1: w_R = 0 only; run files are strict, an unknown key (for instance `w_R` in a v1 run file) is refused, not ignored.
  Methodology v2: a constant w_R per scenario in −1 < w_R ≤ 1 (section above); no time-dependent w, no gradual reheating transition, no temperature-dependent mass.
- **Spectra.** Only the leading h11 instantons enter the mass matrix; f_pert is basis-dependent by √2 for near-degenerate
  modes; a few modes have λ_self < 0. Two vmm conventions are inherited unchanged (`SCIENTIFIC_SCOPE_AND_CONVENTIONS.md`):
  - the pair-instanton amplitude, sign and phase prescription, adopted as a modelling choice (framework of arXiv:2309.01831);
  - the kinetic metric K = ∂_T∂_T̄𝒦 in L = −½M²θ̇ᵀKθ̇. This differs by a factor of two from the literal N = 1 supergravity
    normalisation, an open upstream convention question awaiting the author's clarification. No quantity is rescaled.
- **Producers.** G2 v1 recipe (fast FRST sampler, adaptive Kähler point, QCD window [25, 40], base seed 20260930); the
  `fast` sampler is biased; without MOSEK the Kähler-cone tip is solver-dependent.
- **Reproducibility.** Byte-identical results are demonstrated with the pinned environment (CYAxiverse `a5eacd8`, Julia
  1.12.7 and 1.8.4). With other versions, agreement is scientific (validators C and B), not bitwise.
