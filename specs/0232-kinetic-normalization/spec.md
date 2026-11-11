---
spec_id: CYAX-0232
title: Adopt the supergravity axion kinetic normalization
issue: 232
class: S2
status: draft
workstream: axion spectrum physical conventions
parent: null
depends_on: []
created: 2026-11-10
last_reviewed: null
review_required: Spec Reviewer + Scientific/Numerical Reviewer
approval_ref: N/A while draft
---

# Adopt the supergravity axion kinetic normalization

## Objective

Make every **physical** axion kinetic path in CYAxiverse use the kinetic matrix

$$K_{\rm kin} = 2\,\partial_T\partial_{\bar T}\mathcal{K} = 2\,\mathtt{Kinv}^{-1}$$

in $\mathcal{L}_{\rm kin} = -\tfrac12 M_{\rm pl}^2 K_{\rm kin}\,\partial\vartheta\,\partial\vartheta$,
where $\vartheta$ has period one. This is the literal $\mathcal{N}=1$
supergravity normalization, $c = 1$.

"Physical" includes the spectrum, axion–photon, Fuzzy-axion model-stage
(Julia driver and Python optimizer) and generic inflation-scan paths.

The named **reproduction paths** for arXiv:2608.14780 (Issue #148, spec 0148)
are excluded. They keep their approved contracts: P96 at $c = \tfrac12$, and
A96 as the author's raw-radian matrix (R-008).

Every persisted output is labelled with its normalization, and readers fail
closed on ambiguous labels. The convention is documented. The potential, the
stored `Kinv`, and the vacuum locations and counts do not change.

## Motivation

Issue #232 shows that the physical kinetic paths in `vmm` use $c = \tfrac12$.
Three sources define the axion kinetic term completely and consistently with
period-one axions, and all three display $c = 1$:

- arXiv:2412.12012 (Eqs. 3.1, 3.11);
- arXiv:2309.13145 (Eqs. 3.6, 3.7, 3.9);
- arXiv:2608.14780 (Eqs. 22–23).

arXiv:2412.12012 also uses $c = 1$ in its published numbers.

The other sources are not part of the consensus:

- **arXiv:2103.06812** has two possible readings (see the source-facts table).
- **arXiv:2512.00144** is internally inconsistent.
- **arXiv:2112.04503 and arXiv:1808.01282** are undetermined.

Compared with the consensus sources, `vmm` currently gives:

| Quantity | Current `vmm` relative to $c = 1$ |
|---|---|
| decay constants | low by √2 |
| masses | high by √2 |
| quartics | high by 4 |
| canonical-basis cubics | high by $2^{3/2}$ |
| $g_{a\gamma\gamma}$ | high by √2 |
| slow-roll ε and η | high by 2 |

PR #219 consumes these spectra and waits on this specification.

## Current baseline

### Source facts

The common measure is
$\mathcal{L}_{\rm kin} = -c\,M_{\rm pl}^2 g\,\dot\vartheta\dot\vartheta$, with
$g = \partial_T\partial_{\bar T}\mathcal{K}$, $\mathcal{K} = -2\log\mathcal{V}$,
$\vartheta$ of period one, and $\tau_i = \tfrac12\kappa_{ijk}t^jt^k$.

| Source | Displayed normalization | $c$ |
|---|---|---|
| arXiv:2412.12012 | Eq. 3.1: $-\tfrac12M^2K$. Eq. 3.11: $K = 2\partial_T\partial_{\bar T}\mathcal{K}$. Eq. 3.13 gives the F-term inverse metric | 1 |
| arXiv:2309.13145 | Eq. 3.6: $-\tfrac{M^2}{2}\mathcal{K}_{ab}$. Eq. 3.9: $\mathcal{K}_{ab} = 2\partial_T\partial_{\bar T}\mathcal{K}$. Eq. 3.7: inverse $= 2(\tau\tau^T - \mathcal{V}\kappa t)$. Eq. 2.2: $\cos 2\pi\mathcal{Q}\theta$ | 1 |
| arXiv:2608.14780 | Eq. 22: $\tfrac12K$. Eq. 23: $K = \tfrac12\partial_\tau\partial_\tau\mathcal{K}$. Period one is stated | 1 |
| arXiv:2103.06812 | Eq. (1): $-\tfrac{1}{8\pi^2}M^2K$ with $\cos(\mathcal{Q}\theta+\delta)$, which implies period 2π. $K = \partial_\tau\partial_\tau\mathcal{K}$ (text following Eq. (5)). Eq. (6): $W \supset e^{-2\pi q\cdot T}$, which implies period one | 2 under period 2π; $1/(2\pi^2)$ under period one |
| arXiv:2512.00144 | Eq. C.18: $[K^{-1}] = 2(\tau\tau^T - \mathcal{V}\kappa t)$, which implies $c = 1$. Eq. A.3 gives $K_{ij} = \partial_T\partial_{\bar T}\mathcal{K}$, and Eq. II.4 writes the τ kinetic term $K^{ij}d\tau_i\wedge\star d\tau_j$, both with no ½; Eq. A.6 uses $\cos(q\phi)$. These are inconsistent with each other | inconsistent |
| arXiv:2112.04503 | Eq. (41): $-\tfrac12K_{AB}$; $K$ is not defined precisely | undetermined |
| arXiv:1808.01282 | Eq. 4.3: $-\tfrac{M^2}{2}K$; $K$ is not defined precisely | undetermined |
| arXiv:2309.01831 | No kinetic term. Eq. 9 writes the pair term as $\pi q^TK^{-1}q'$ | n/a |

Equation numbers for arXiv:2103.06812 and arXiv:2112.04503 are those of the
v2 HTML rendering; the other entries use the latest arXiv version as of
2026-11-10. The conventions page (R-009) records the version for each entry,
and maps the arXiv:2103.06812 v2 numbers (1), (5), (6) to the 1.1, 2.4, 2.5
numbering used in the Issue #232 correction comment.

The arXiv:2512.00144 entries II.4, A.3 and A.6 follow the CYAX-0232 G0
review's reading of the paper. The author verified C.18 directly.

### Open inference

These conclusions are derived, not stated by the sources:

- **arXiv:2608.14780 numerics.** The published numbers use $c = \tfrac12$.
  Basis: its Eq. 96 eigenvalues equal eig($g$), within 0.07%, on the
  reconstructed N = 8 geometry.
- **arXiv:2309.01831 Eq. 9.** The $K^{-1}$ there is $g^{-1}$, which is the
  CYTools `Kinv`. Basis: the F-term cross term derived with
  $K^{a\bar b}\mathcal{K}_{\bar b} = -2\tau^a$ reproduces Eq. 9 only with this
  identification.
- **arXiv:2412.12012 numerics.** These use $c = 1$. Basis: R-003's evidence.

### Implementation facts (`vmm` `10aa3dc`)

Every path cited here is unchanged between `34187c5`, where the facts were
first verified, and `10aa3dc`.

**CYTools and storage.**
- CYTools `compute_inverse_kahler_metric` returns
  $g^{-1} = 4(\tau\tau^T - \mathcal{V}\kappa t)$, and every generator stores
  it as `Kinv`.

**Where $c = \tfrac12$ enters.** In `src/read.jl`, exactly two Kinv→kinetic
conversions do this (the direct factorizations listed next also give
$c = \tfrac12$):
- `src/read.jl` `potential_factored`, which forms `C = chol(Kinv).L`. Its
  docstring promises `Kinv ≈ C * C'`;
- `src/read.jl` `_kinetic_matrix`, which forms `K = inv(Hermitian(Kinv))`.
  Its docstring calls this "`K = (1/2) g`" without defining $g$.

**Paths that factor `Kinv` directly** for kinetic purposes:
- `src/axion_photon.jl` `_canonical_frame`, which uses `chol(Kinv)`;
- `src/axion_photon.jl` QED kinetic norm, $q^T$`Kinv`$q$ (line 1152);
- `src/paper_benchmarks/fuzzy_axions_model_stage.jl`, which uses `chol(Kinv)`;
- `scripts/reproduce_fuzzy_axions_h11_4.py`, the optimizer mass evaluator,
  which uses `cholesky(inverse_metric)`;
- `scripts/benchmark_hybrid_scaling.jl`, which forms `inv(Kinv)`.

**Paths that take K as an argument.**
- `_canonical_factor_from_K(K)` returns $C$ with $CC^T = K^{-1}$. It treats
  its argument as the matrix in $-\tfrac12M^2K\,\partial\vartheta\partial\vartheta$,
  so it is convention-neutral.
- The callers that take K are `hp_spectrum(K,…)`, `pq_spectrum(K,…)`,
  `pq_physical_spectrum(K,…)`, `pq_hybrid_physical_spectrum(K,…)`, and the
  leading-Hessian, `mass_basis_accuracy` and `instanton_hierarchy_diagnostics`
  families.
- They inherit their convention from whoever supplies K. In the geometry
  routes, that is `_kinetic_matrix` through `read.potential` and
  `read.oriented_potential`.

**Inflation consumers of `read.potential` / `read.oriented_potential` K.**
- `src/inflation_points.jl`: `prepare_geometry_context` →
  `prepare_context(Q,L,K)`, which computes ε and η.
- `scripts/inflation_scan_common.jl`
- `scripts/inflation_scan_contract.jl`
- `scripts/inflation_candidate_refinement.jl`
- `scripts/inflation_scale_continuation.jl`. Its reciprocity check requires
  `K*Kinv ≈ I`.
- `scripts/analyze_inflation_candidates.jl` (writes a CSV)
- `scripts/compare_reduction_inflation_scan.jl` (writes a CSV)
- `scripts/inflation_scan_prep.jl` (streamed diagnostic-family CSV, rows from
  line 406; resume check `_scan_prep_completed`, line 267)
- `scripts/build_orientifold_vacua_inflation.jl`. It writes an HDF5
  `inflation` group into each geometry file, with
  `INFLATION_GROUP_SCHEMA_VERSION = "cyaxiverse-phase3-orientifold-inflation-2.3"`
  (legacy `…-1.0`), from `inflation_scan_common.run_geometry` and
  `inflation_candidate_refinement.scan_geometry_for_inflation`. Its
  `inflation/catastrophes` and `inflation/efolds` contents
  (`candidate_slowroll_saddles`, `n_candidates`, `n_qualified`,
  `flow_efolds`, `flow_slow_roll_efolds`) depend on ε and η.
  `_has_inflation_group` (line 100) decides resume; without `force=true`
  an existing group is refused (line 155), and with it the group is deleted
  and rewritten (line 168).

`inflation_scan_common.jl` also stamps rows with
`INFLATION_SCAN_CONTRACT_VERSION = "4"`.

Their persisted CSV schemas:

| Constant | Value | File |
|---|---|---|
| `INFLATION_DIAGNOSTIC_SCHEMA_VERSION` | "1" | `inflation_diagnostics_common.jl` |
| `REFINEMENT_SCHEMA_VERSION` | "3" | `inflation_candidate_refinement.jl` |
| `FLOW_SCHEMA_VERSION` | "1" | `inflation_candidate_refinement.jl` |
| `PILOT_SCHEMA_VERSION` | "6" | `inflation_scale_continuation.jl` |
| `INFLATION_SHARD_SCHEMA_VERSION` | "2" | `inflation_scan_shards_common.jl` |

**Reproduction paths** (Issue #148 / spec 0148 contracts, and the
arXiv:2608.14780 N = 5 and N = 8 source metrics). These consume $M_{96}$,
`N5_K_RAW`, `N8_K_RAW`, `K_RAW` or the `reduced_models` source kinetic
matrices, or check against them:
- `src/paper_benchmarks/{n8_g3_control, n8_continuation, poly102_inflation, reduced_models, compatibility, catastrophe_diagnostics}.jl`.
  The module file `src/paper_benchmarks.jl` only includes these and the
  physical `fuzzy_axions_*` files; it has no functions of its own to
  classify
- the function `pilot_benchmark_regression` in
  `scripts/inflation_scale_continuation.jl` (line 1829; A96 through
  `poly102_inflation`, printed only). The rest of that file is physical.
- `scripts/audit_issue_148_*.jl`
- `scripts/issue_148_*.jl`
- `scripts/inflation_refinement_common.jl`. It runs
  `poly102_inflation.n8_physical_gradient_flow`, which uses `N8_K_RAW/k²`
  (A96), and its summary rows carry
  `diagnostic_schema_version = INFLATION_DIAGNOSTIC_SCHEMA_VERSION` (line 74),
  the same family the physical scan writes (`inflation_scan_common.jl:357`).
- `scripts/inflation_reproduction.jl` (`n8_kinetic_matrix`, A96).
- `scripts/benchmark_inflation_scalability.jl` (`n8_physical_gradient_flow`,
  A96).
- `scripts/author_poly102_reference.py` (`K_RAW / k**2`, A96).
- `scripts/generate_appendix_c_geometry.py`. It forms `inv(kinv)` (line 105)
  and checks its eigenvalues against arXiv:2608.14780 Eq. 96 (line 75). This
  is a P96 reproduction validation: Eq. 96 equals eig($g$).

**Fuzzy-axion persisted outputs.**
- `scripts/fuzzy_axion_model_stage_driver.jl` writes an unversioned HDF5 file
  with root datasets including `total_model_count`,
  `model_count_per_record` and `model_mass_reference_log10_ev`
  (lines 116–132).
- `scripts/reproduce_fuzzy_axions_h11_4.py` writes versioned outputs with
  `REPRODUCTION_SCHEMA_VERSION = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.2"`
  (line 139) and
  `KAEHLER_PRODUCTION_SCHEMA_VERSION = "cyaxiverse-fuzzy-axions-kaehler-production-1.0"`
  (line 1049). Both contain `mass_log10_ev` (lines 1341, 1656). This script
  reproduces arXiv:2412.12012, whose own numerics are $c = 1$; it is therefore
  a physical path, not a reproduction path in the R-008 sense.

**Validation-only factorizations.** These Cholesky-factor `Kinv` only to check
that it is positive definite, as in `load_geometry_inputs`.

**Vacuum search.** It uses only $Q$ and $L$: `vacua_id`, `vacua_estimate`
and `scripts/vacua_pipeline.jl`. The pipeline checks K only for shape and
finiteness.

**Persisted layouts:**
- F1: datasets in `spectrum/physical` and scalar datasets in
  `spectrum/physical/metadata`.
  - `physical_spectrum` (`read.jl`) requires `spectrum/physical/{m, mode_indices, fK_log10}`
    and `spectrum/physical/metadata/{threshold_log10, prec, provisional, runtime_seconds}`.
    Every other dataset, including `fpert_log10` and `schema_version`, is
    optional.
  - `schema_version = "cyaxiverse-physical-spectrum-3"` since `239a3a7`.
  - Earlier and migrated outputs (`scripts/migrate_legacy_spectra.jl`, which
    copies old metadata as it is) have no `schema_version` unless their
    source carried one.
- F1 batch summary CSV: `SUMMARY_HEADER` in `scripts/batch_physical_spectrum.jl`.
  With `--append-summary`, `_write_summary_header` keeps an existing file and
  its header unchanged.
- F2: unversioned child groups `spectrum/{masses, decay, quartdiag, quart31, quart22, cubic}`,
  written directly into the shared `spectrum` group (`generate.jl:1076`,
  `:3017`) by two functions:
  - `hp_spectrum_save`: cubic in the ϑ/charge basis;
  - `pq_spectrum_save`: cubic in the canonical mass basis.
  - `scripts/migrate_quartic_index_ordering.jl` rewrites existing F2 groups.
  - The same `spectrum` group also holds F1 (`spectrum/physical`) and F3
    (`spectrum/axion_photon`).
- F3: `spectrum/axion_photon`, identity group `identity` with integer
  `schema_version = 1`, plus a CSV summary. `run_batch_axion_photon` skips
  geometries that already hold a result with a matching configuration. For a
  stored result whose configuration differs, it reports `:failed` and asks for
  `force=true` (`axion_photon.jl:2101–2120`).

### Owner-approved conventions

- P96 and A96 (Issue #148, 2026-09-10; spec 0148).
- Owner decisions of 2026-11-10, recorded on Issue #232 and its comments:
  1. adopt $c = 1$ for physical paths and document it;
  2. new schema plus label, no regeneration;
  3. keep the reproduction contracts;
  4. version impact `minor`;
  5. generic inflation scans are physical and adopt $c = 1$;
  6. unversioned legacy physical spectra are read as $c = \tfrac12$, by
     inference;
  7. the label is stored as a scalar string dataset, which is the deliberate
     realization of "attribute".

**Supersession note.** The Issue #232 body states decision 3 as keeping P96
at $c = \tfrac12$ "as for A96". The owner's correction comment of 2026-11-10
on Issue #232 supersedes that wording: A96 is the raw-radian contract, not
$c = \tfrac12$. This specification follows the corrected wording.

## Scope

- One shared factor (`kinetic_factor`) and one shared kinetic matrix
  (`kinetic_matrix`), both built from `Kinv`. Every physical Kinv→kinetic
  conversion uses them (R-001).
- A normative contract for APIs that take K as an argument (R-001).
- Labelled persisted outputs: every K-dependent output under the R-006
  general rule, including F1, F2, F3, the Fuzzy-axion outputs (F4, F5), the
  orientifold inflation group (F6) and the inflation CSV families.
- Fail-closed readers and writers, and a rule for combining outputs (R-007).
- Labels on the reproduction paths (R-008).
- Updates to the physical Python path and the benchmark script.
- Tests, docstrings, a conventions documentation page, and a README note.

## Non-scope

- Regenerating persisted spectra, axion–photon scans or inflation scans.
- Changing the potential, `Kinv` storage, charges, phases or vacuum algorithms.
- Changing the numerics of any reproduction path, or any Issue #148 gate.
- Conversion utilities between normalizations.
- Determining the arXiv:2103.06812 numerics.
- Re-interpreting published results.
- Releasing or adopting a package version.
- Updating PR #219's readers. That is the consumer's own follow-up.

## Scientific claim boundary

### This specification may establish

- That the CYAxiverse physical kinetic outputs follow the convention displayed
  by arXiv:2412.12012, arXiv:2309.13145 and arXiv:2608.14780 (Eqs. 22–23).
- That those outputs reproduce arXiv:2412.12012 Eq. 4.3 within the R-003
  tolerance, while the $c = \tfrac12$ evaluation does not.

### This specification must not establish

- Population-level or physical claims from legacy outputs that are relabelled
  but not recomputed.
- Any claim that past papers or past CYAxiverse results are wrong.
- Any statement about the arXiv:2103.06812 or arXiv:2512.00144 numerics.
- Any statement that changed inflation-candidate or Fuzzy-axion model counts
  are physical results. Changed counts are reported as consequences of the
  convention only.

## Fixed conventions and invariants

- **Physical kinetic matrix.** $K_{\rm kin} = 2\,\mathtt{Kinv}^{-1}$, with the
  ½ prefactor and period-one $\vartheta$.
- **Canonical factor.** $B$ is lower triangular with positive diagonal and
  $BB^T = \mathtt{Kinv}/2 = K_{\rm kin}^{-1}$.
- **Coordinate map.** $\vartheta = B\psi$, with
  $\mathcal{L}_{\rm kin} = -\tfrac12M_{\rm pl}^2\dot\psi^T\dot\psi$.
- **Transformation rules.**
  - A charge column $q$ maps to $\hat q = B^Tq$.
  - A ϑ-Hessian $H$ maps to $B^THB$.
  - A kinetic norm is $q^T(\mathtt{Kinv}/2)q$.
- **K-argument contract.** Wherever a public or internal API takes a matrix
  argument named K (or documented as the kinetic matrix), that argument is the
  matrix in $-\tfrac12M_{\rm pl}^2K\,\partial\vartheta\partial\vartheta$.
  These APIs never rescale it.
- **Potential.** The potential uses $K^{a\bar b} = \mathtt{Kinv}$, unchanged.
- **Inverse metrics stay $g^{-1}$.** Every persisted or exported inverse
  metric (`Kinv`, the Python `inverse_metric` passed from F5 to the Julia
  model stage, `kinv` in scripts) stays $g^{-1}$, as CYTools returns it. The
  factor ½ is applied only inside `kinetic_factor`, `kinetic_matrix`, the
  R-001 norm and their Python equivalents, never to a stored or exported
  matrix.
- **Reproduction contracts.**
  - P96 is $K_\theta = M_{96}/k^2$ with period-one θ, which is $c = \tfrac12$
    because $M_{96} \approx g$.
  - A96 is the author's raw-radian $M_{96}/k^2$. In period-one coordinates
    this is $K_\theta = (2\pi)^2M_{96}/k^2$, about $c = 2\pi^2$ in the common
    measure.
  - The label `"a96_reproduction_raw_radian"` covers the author source
    matrices divided by $k^2$ *when used in raw-radian coordinates*: for
    N = 8, $M_{96}$ (`N8_K_RAW`; the same matrix in period-one θ is P96,
    `n8_continuation.jl:6–8`); for
    N = 5, `N5_K_RAW` (`n5_geometry().metric_convention = :raw_angles_radians`,
    `poly102_inflation.jl:150–161`). Spec 0148 names only the N = 8 matrix
    A96; extending the label to the N = 5 author matrix records the same
    raw-radian convention class and changes no numerics.

## Requirements

### R-001 — Single physical kinetic convention and the K-argument contract

**(a) Shared functions.** The package SHALL provide two functions:
- `kinetic_factor(Kinv)` returns $B = $ `cholesky(Symmetric(Kinv ./ 2)).L`;
- `kinetic_matrix(Kinv)` returns $2\,$`inv(Hermitian(Kinv))`.

Every physical path that converts a stored or supplied `Kinv` into kinetic
data SHALL use one of these two functions or the norm
$q^T(\mathtt{Kinv}/2)q$.

These paths include:
- `potential_factored`;
- `_kinetic_matrix`, and therefore `read.potential` and
  `read.oriented_potential`;
- `_canonical_frame`;
- the QED kinetic norm;
- `fuzzy_axions_model_stage.jl`;
- `scripts/benchmark_hybrid_scaling.jl`.

The Python path in `scripts/reproduce_fuzzy_axions_h11_4.py` SHALL implement
the same contract and have an equivalent test.

`potential_factored` SHALL return `C = kinetic_factor(Kinv)`, so that
`C * C' ≈ Kinv / 2`, and its docstring SHALL state this. The `Kinv` it returns
stays as now: the stored matrix symmetrized as `(Kinv + Kinv')/2`
(`read.jl:294`), without the factor ½.

`leading_hierarchy`'s `metric_residual` (`axion_photon.jl:1093`) SHALL be
computed from the same factorization that `_canonical_frame` obtains from
`kinetic_factor`, so that the residual
tests $\Theta^T(BB^T)^{-1}\Theta = I$ for the $B$ actually applied.

The unexported `read.K` (`read.jl:448`) reads `cytools/potential/K`, which no
repository writer produces and whose normalization is unknown. `plan.md`
SHALL classify it (remove, or document it as unlabelled and not physical).

**(b) APIs that take K.** APIs that take K SHALL follow the K-argument
contract and SHALL NOT call `kinetic_factor` on their argument.
`_canonical_factor_from_K` is unchanged. Docstrings of every K-taking public
API SHALL state the contract. The same holds for APIs that take a canonical
factor C directly (for example `leading_hessian_matrix_float64(C, …)`,
`generate.jl:1319`, its `_scaled` variant at 1440,
`high_precision_leading_hessian` at 1470 and
`leading_hessian_mass_basis_float64` at 1778): an argument documented as a
canonical factor satisfies $CC^T = K^{-1}$ for the K-argument $K$ and is never
rescaled.

**(c) Exemptions.** These are exempt from (a), and the reproduction paths
are also exempt from (b): a reproduction API that takes a kinetic matrix takes
it in the coordinates of its own contract (R-008):
- the reproduction paths, including reproduction validations such as the
  `generate_appendix_c_geometry.py` Eq. 96 eigenvalue check, which are
  governed by R-008;
- validation-only factorizations that are not used to compute outputs.

**(d) No other conversions.** No other physical path SHALL invert or factor
`Kinv` for kinetic purposes.

### R-002 — Invariance of potential and vacua

The following SHALL be unchanged:

(a) the potential data $(Q, L)$, including the pair-term amplitudes, and the
stored `Kinv`;

(b) the vacuum locations, in period-one $\vartheta$ coordinates of the GLSM
basis, as returned by `vacua_id` and `vacua_estimate` and as stored in the
`vacua_pipeline` group datasets (`estimate`, `issquare`, `verified`,
`threshold`, and the coordinate datasets);

(c) the vacuum counts;

(d) the ϑ-Hessian $\partial^2V/\partial\vartheta\,\partial\vartheta$ at each
baseline vacuum. This array does not depend on K, and the R-002 baseline
artifact stores it.

Mode counts that use a K-dependent spectrum (for example
`inflation_points._diagnose`, whose zero threshold is
`zero_tolerance·max(max|eig|, 1)`) are not invariants; R-005 covers them.

**Bitwise comparison.** These are compared bitwise against baseline artifacts
generated at the `vmm` commit the implementation branch is based on. That is
`10aa3dc` at drafting; `plan.md` records the commit actually used. The
baseline is pinned by its sha256 in `plan.md`. It is regenerated if the
implementation base changes. It uses the same Julia 1.12 patch release, OS
image and BLAS as the CI job that runs the fixtures.

**Tolerance comparison.** A tolerance comparison is allowed only where the PR
lists a changed floating-point schedule in a K-independent path. It uses the
periodic torus distance at most $\max(10^{-12},$ `merge_tolerance`$)$.

**Quantities that are not invariant.** These change by R-005 and SHALL be
reported:
- canonical-coordinate quantities;
- mass-threshold counts (for example, modes with $m \ge H_0$);
- slow-roll classifications;
- Fuzzy-axion model counts.

### R-003 — Named source reproduction

**Fixture.** The literal inputs of the `test/runtests.jl` testset "Fuzzy-axion
h11=2 source-corrected role validation":
- `Q`, `tau`, `cy_volume` and `inverse_metric`;
- $g_s = 0.5029733$ and $W_0 = 1$;
- instanton scales from Eq. 3.18, using the formula implemented in
  `fuzzy_axions_model_stage.jl:87–96` (which `leading_axion_reference_data`
  uses internally; that function returns only masses);
- $M_{\rm pl} = $ `constants()["MPlanck"]`.

**Evaluation.** The leading directions are taken in the order QCD first
(divisor D6) then fuzzy (D2), through `_canonical_frame` with
`Q_reduced = [q_{D6} q_{D2}]`. Then:
- $f_i = M_{\rm pl}/(2\pi\hat q_{ii})$;
- $m_{\rm fuzzy} = \Lambda_{\rm fuzzy}^2/f_{\rm fuzzy}$;
- $m_{\rm QCD}$ from Eq. 2.6 of the source, evaluated with the computed
  $f_{\rm QCD}$. It depends on the convention only through $f_{\rm QCD}$.

**Acceptance.** Each of $f_{\rm QCD}$, $f_{\rm fuzzy}$, $m_{\rm QCD}$ and
$m_{\rm fuzzy}$ SHALL lie within **5% relative** of Eq. 4.3:
- $f_a = (3.2\cdot10^{15}, 1.2\cdot10^{16})$ GeV;
- $m_a = (1.8\cdot10^{-9}, 5.0\cdot10^{-20})$ eV.

**Discrimination.** The same evaluation with $c = \tfrac12$ (using `Kinv` in
place of `Kinv/2`) SHALL miss each $f_a$ by more than 20%. This shows the test
can tell the two conventions apart. (At $c = \tfrac12$, $m_{\rm QCD}$ is
about 2.5e-9 eV, also outside the 5% gate.)

The stringy-only selector `leading_axion_reference_data` is not this gate. Its
scaling is covered by R-005.

### R-004 — Factor contract and orientation

**Factor identities.** On the R-004 fixtures, `kinetic_factor` SHALL satisfy:
- $\lVert BB^T - \mathtt{Kinv}/2\rVert \le 10^{-14}\lVert\mathtt{Kinv}\rVert$;
- $B^TK_{\rm kin}B = I$ to $10^{-12}$;
- $B$ is lower triangular with positive diagonal.

For general geometries, the bounds scale with $n\,\kappa(\mathtt{Kinv})\,\epsilon$.

**Non-diagonal fixture.**
- Inputs: `Kinv` $= \begin{pmatrix}3&1\\1&2\end{pmatrix}$, charges
  $q_1 = (1,0)^T$ and $q_2 = (1,1)^T$, and a hierarchy
  $\Lambda_2^4/\Lambda_1^4 \le 10^{-20}$.
- API: `axion_photon.leading_hierarchy` for $f$ and $m$; the Julia model
  stage for $m$ only (it exposes no decay constants). The hierarchy is set
  through the fixture's `tau`. `plan.md` SHALL pin the model-stage inputs:
  per-divisor `Q`, `tau` with $\tau_1 < \tau_2$ so that $q_1$ leads, and
  `cy_volume`.
- Assertions, to $10^{-10}$ relative:
  - $f_1 = M_{\rm pl}/(2\pi\sqrt{G_{11}})$;
  - $f_2 = M_{\rm pl}/(2\pi\sqrt{\det G/G_{11}})$, where $G = Q^T(\mathtt{Kinv}/2)Q$;
  - $m_i^2 = \Lambda_i^4/f_i^2$.

**Orientation guard.** The test SHALL assert that using $BQ$ in place of
$B^TQ$ changes at least one asserted value by more than $10^{-3}$ relative.

**Scalar fixture.** The one-axion fixture
$f = M_{\rm pl}\sqrt{2/(q^2k)}/(2\pi)$, where $k$ is the scalar `Kinv`, is
retained. It is evaluated through both `kinetic_factor` and `kinetic_matrix`.

**Existing K-argument test.** The test "HP spectrum: one-axion analytic mass"
(`test/runtests.jl:2052`) passes K = 4 to `hp_spectrum(K,…)`. Under the
K-argument contract it SHALL stay unchanged and pass.

### R-005 — Scaling contract

On the R-004 non-diagonal fixture and on the R-003 fixture inputs, every
listed output, compared with a $c = \tfrac12$ evaluation of the same inputs,
SHALL equal the factor below to $10^{-10}$ relative (for log10 fields: the
corresponding offset, e.g. $-\tfrac12\log_{10}2$, to $10^{-10}$ absolute;
for the canonical-mass-basis cubic tensor: up to one sign per mass-basis
mode, since eigenvector signs are arbitrary). The exception is counts,
which SHALL be reported.

Some outputs keep only modes above a mass threshold (the $H_0$ cut in
`pq_physical_spectrum`, the F3 QED threshold). For these, the factor
comparison covers the modes present in both evaluations, and the difference
in the mode sets SHALL be reported.

**Spectrum (F1, F2, in memory):**

| Output | Factor |
|---|---|
| $f_K$, $f_{\rm pert}$ | √2 |
| masses | 1/√2 |
| $\lambda_{\rm self}$, $\lambda_{31}$, $\lambda_{22}$ | 1/4 |
| cubic tensor, `pq_spectrum_save` (canonical mass basis), with the evaluation phase held fixed in ϑ (the mass-basis phase is $x_0 = \mathrm{basis}^TB^{-1}\vartheta_0$ in each convention, since `cubic` takes `Qmass = Q̃ᵀ·C·basis`, `generate.jl:3013–3015`) and a nonzero ϑ-phase fixture | $2^{-3/2}$ |
| cubic tensor, `hp_spectrum_save` (ϑ/charge basis) | 1 |

**Axion–photon (F3):**

| Output | Factor |
|---|---|
| $f$ | √2 |
| mixing Θ from `mixing_matrix` (ratios of $q$) | 1 |
| persisted `hierarchy/q` and `hierarchy/theta_from_canonical` (`axion_photon.jl:1382–1383`) | 1/√2 |
| $g_{a\gamma\gamma}$, `log10_g_effective` under `:divisor_instanton` | 1/√2 |
| `log10_g_effective` for light modes under `:electron_proxy` | $2^{-3/2}$ |
| persisted `hierarchy/log10_f_GeV` (`axion_photon.jl:1384`) | $+\tfrac12\log_{10}2$ (×√2) |
| persisted `hierarchy/log10_mass_eV` (`axion_photon.jl:1385`) | $-\tfrac12\log_{10}2$ (×1/√2) |
| `log10_g_effective` for heavy modes under `:electron_proxy` | 1/√2 |
| QED threshold mass | 1/√2 |
| photon width, all modes under `:divisor_instanton` | $2^{-5/2}$ |
| photon width, heavy modes under `:electron_proxy` | $2^{-5/2}$ |
| photon width, light modes under `:electron_proxy` | $2^{-9/2}$ |
| quartic width | $2^{-9/2}$ |
| `light_mode_count` under `:divisor_instanton` | unchanged |
| `light_mode_count` under `:electron_proxy` | may change; reported |

Under `:electron_proxy`, the light-mode and heavy-mode factors apply to modes
on the same side of the QED threshold in both evaluations. Modes that cross
the threshold are reported.

The axion–photon decay constants SHALL equal arXiv:2309.13145 Eq. 2.15, with
the Gram–Schmidt step taken in the Eq. 3.7 inverse metric.

**Inflation (in memory and CSV):**

| Output | Factor |
|---|---|
| ε, η, every eigenvalue of $K^{-1}H$ | 1/2 |
| `mass_min`, `mass_max` (log10 eV; diagnostic CSV rows, `inflation_scan_common.jl:343–345, 384`, and F6 `inflation/catastrophes`, `build_orientifold_vacua_inflation.jl:246–247`) | $-\tfrac12\log_{10}2$ |
| e-fold counts along a fixed trajectory | about 2 in the slow-roll approximation; reported, not gated |
| candidate classifications | may change; reported |
| mode counts from `inflation_points._diagnose` | may change; reported |
| F6 `candidate_slowroll_saddles`, `n_candidates`, `n_qualified`, `flow_efolds`, `flow_slow_roll_efolds` | may change; reported |

**Fuzzy-axion model stage (F4, F5):**

| Output | Factor |
|---|---|
| model-stage masses at fixed inputs, including F4 `model_mass_reference_log10_ev` | 1/√2 ($-\tfrac12\log_{10}2$ in log10 fields) |
| the F5 mass evaluator `_fuzzy_axion_mass_log10` at a fixed Kähler point | 1/√2 |
| F4 `model_lambda` and model counts | may change; reported |
| F5 optimizer outputs: converged points, `mass_log10_ev` at the solution (constrained to the 1e-18 eV target), residuals and feasibility | may change; reported |

### R-006 — Persisted labelling contract

**Label.** The label is `kinetic_normalization`, stored as a scalar UTF-8
string HDF5 dataset, or as a CSV column of the same name on every row. This
form is owner-approved decision 7. Allowed values:

| Value | Meaning | Written by |
|---|---|---|
| `"sugra_c1"` | $K_{\rm kin} = 2\,\mathtt{Kinv}^{-1}$ | every physical writer that computes from `Kinv` |
| `"p96_reproduction_c_half"` | the P96 contract | P96 reproduction writers only |
| `"a96_reproduction_raw_radian"` | the A96 contract | A96 reproduction writers only |
| `"legacy_c_half"` | the pre-change physical convention | derived writers only (below) |

Primary writers, which compute K-dependent values from `Kinv`, never write
`"legacy_c_half"`. Readers produce it by inference (R-007).

**Derived writers.** A *derived writer* writes K-dependent values that it takes
from persisted outputs or from results returned by readers, rather than
computing them from `Kinv` (for example
`scripts/reproduce_appendix_b_spectra.jl`, which aggregates F1 into
`appendix_b_quantiles.csv`, and `axion_photon.write_scan_csv` applied to
results from `read_axion_photon_result`). A derived writer SHALL apply the
R-007 checks to its inputs, SHALL reject inputs whose labels differ (R-007
combination), and SHALL write the common input label. When that label was
inferred it SHALL also write `kinetic_normalization_inferred = true` (a CSV
column or scalar dataset next to the label). A derived writer SHALL NOT write
`"sugra_c1"` for inputs labelled or inferred otherwise, and SHALL NOT write
into the F1–F6 HDF5 families (R-007, primary HDF5 writers).

**General rule.** A *K-dependent output* is any persisted value computed from
a kinetic matrix or factor derived from `Kinv`: through `kinetic_factor`,
`kinetic_matrix`, `potential_factored().C`, `read.potential().K`,
`read.oriented_potential().K`, the R-001 norm, or a reproduction matrix, or
taken from another K-dependent output. Every writer of a K-dependent output in
`src/`, `ext/`, `scripts/` or `bin/`, other than the layout migrations named
in R-007, SHALL label it, and SHALL increment the
output's schema or contract version if it has one. The table below lists the
writers known at `10aa3dc`. It is not the definition: a writer found later is
governed by this rule, is added to `plan.md`'s inventory, and does not need a
spec amendment. A known writer not in the table is `spectra.jl` (an F2
driver). Two scripts that read K are not K-dependent writers:
`scan_relaxed_qcd_candidates.jl` loads `read.oriented_potential` K but its
CSV comes from `jlm_reduced.prepare(Q, L)` and `critical_ensemble` only
(`solve_reduced`, lines 208–231); and `validate_vacua_stage4_5.jl` writes
`stage4_inflation.csv` from `LQtilde(Q, L)` and `leading_critical_branches`
only (lines 157–190, 344–352); it uses no K, so it is not a K-dependent
output. Its N5/N8 anchor results (`_stage4_n5_n8`, lines 138–151) are
written only to the stage-4 text report (lines 342–343); `plan.md`
classifies that report under R-008.

**Output families:**

| Family | Label location | Version change |
|---|---|---|
| F1 | `spectrum/physical/metadata/kinetic_normalization` | `schema_version = "cyaxiverse-physical-spectrum-4"` |
| F1 summary CSV | `kinetic_normalization` column | — |
| F2 (`hp_spectrum_save`, `pq_spectrum_save`) | `spectrum/metadata/kinetic_normalization` | new `spectrum/metadata/schema_version = "cyaxiverse-spectrum-groups-1"`, plus `spectrum/metadata/cubic_basis` ∈ {`"charge_basis"`, `"canonical_mass_basis"`} |
| F3 | `spectrum/axion_photon/identity/kinetic_normalization` | identity `schema_version = 2` |
| F3 CSV summary | `kinetic_normalization` column | — |
| F4: `fuzzy_axion_model_stage_driver.jl` HDF5 | root dataset `kinetic_normalization` | new root `schema_version = "cyaxiverse-fuzzy-axion-model-stage-1"` |
| F5: `reproduce_fuzzy_axions_h11_4.py` versioned outputs | top-level `kinetic_normalization` key (JSON) or root scalar string dataset (HDF5) | `REPRODUCTION_SCHEMA_VERSION` → `…-reproduction-1.3`; `KAEHLER_PRODUCTION_SCHEMA_VERSION` → `…-kaehler-production-1.1` |
| F6: `build_orientifold_vacua_inflation.jl` HDF5 `inflation` group | `inflation/kinetic_normalization` | `INFLATION_GROUP_SCHEMA_VERSION` → `…-orientifold-inflation-2.4` |
| Inflation CSV families (the five schema constants above, and rows stamped with `INFLATION_SCAN_CONTRACT_VERSION`) | `kinetic_normalization` column | each schema version incremented by one; contract version `"4"` → `"5"` |
| Other CSVs written by the inflation scripts (`analyze_inflation_candidates.jl`, `compare_reduction_inflation_scan.jl`) | `kinetic_normalization` column | version incremented if the file has one |

**F5 completeness.** Any other versioned output of
`reproduce_fuzzy_axions_h11_4.py` whose content depends on the mass evaluator
SHALL also carry the label (a root scalar string dataset
`kinetic_normalization` in HDF5, a top-level key in JSON) and a version
increment (`x.y` → `x.(y+1)`; an integer suffix `-n` → `-(n+1)`). `plan.md` SHALL
list every such output, with the evidence that the others do not depend on
the evaluator. This includes the model-stage input HDF5 written by
`_write_model_stage_input` (attribute
`record_metadata_schema = "kaehler-point-provenance-1"`, which becomes
`"kaehler-point-provenance-2"`), whose
production-mode `point`s are optimizer-selected and so depend on the
evaluator.

**Per-row labels in the inflation diagnostic family.** The physical writers
(`inflation_scan_common.jl`, `inflation_scan_prep.jl`) and the A96
reproduction refinement (`inflation_refinement_common.jl`) produce rows of the
`INFLATION_DIAGNOSTIC_SCHEMA_VERSION` family. At `10aa3dc` the refinement rows
stay in memory (the file is included only by `test/runtests.jl`). The label is
nevertheless per row, in memory and in any file: physical rows carry
`"sugra_c1"` and refinement rows carry `"a96_reproduction_raw_radian"`. The
other four inflation families have only physical writers at `10aa3dc`, and
their rows carry `"sugra_c1"`.

**In memory.**
- The package SHALL export the constant
  `CYAxiverse.KINETIC_NORMALIZATION == :sugra_c1`, which describes every
  physical result it computes from `Kinv`.
- Reproduction results carry their reproduction label through the
  `kinetic_normalization` field of their own result types (R-008).
- Readers that return a `NamedTuple` (for example `physical_spectrum`,
  `read.pq_spectrum`, `read.hp_spectrum`) SHALL add the fields
  `kinetic_normalization::Symbol` and `kinetic_normalization_inferred::Bool`.
- Readers that return a public struct (for example
  `read_axion_photon_result`, which returns `axion_photon.AxionPhotonResult`)
  SHALL keep the struct layout unchanged. The label is available from a new
  public accessor,
  `stored_kinetic_normalization(path; family) -> (label::Symbol, inferred::Bool)`,
  with `family ∈ (:f1, :f2, :f3, :f4, :f6)` for HDF5 outputs, which applies
  the same R-007 checks. Struct-returning readers SHALL still enforce R-007
  before returning.
- **F3 in memory.** `AxionPhotonIdentity.schema_version` SHALL hold the
  *stored* identity version; `_read_axion_photon_identity` currently replaces
  it with the current constant (`axion_photon.jl:1566`), and that SHALL
  change. Version 2 means `:sugra_c1`; version 1 means `:legacy_c_half`,
  inferred. `write_scan_csv` and every other consumer of `AxionPhotonResult`
  SHALL derive the label from this field. This mapping is sound because a
  version-2 F3 group can only hold `"sugra_c1"`: `write_axion_photon_result`
  SHALL keep refusing any result whose `identity.schema_version` is not the
  current version (the existing `_assert_result_identity_matches!` check,
  `axion_photon.jl:1737`), and F3 readers reject a stored `"legacy_c_half"`
  (R-007).
- Python readers expose the same pair as a `(label, inferred)` tuple or as
  dict keys of the same names.
- Field layouts and positional constructors of public structs (exported from
  `CYAxiverse` or from any of its submodules) SHALL NOT change. In particular,
  `AxionPotential.K` is $K_{\rm kin}$ and carries no field.
- Results computed from a caller-supplied K are not labelled by the package.

### R-007 — Readers, writers and combination (fail closed)

**Inference is limited to recognized legacy layouts:**
- F1 with `schema_version == "cyaxiverse-physical-spectrum-3"` and no label;
- F1 with no `schema_version` and no label, whose `spectrum/physical` group
  holds every dataset that `physical_spectrum` requires: `m`, `mode_indices`
  and `fK_log10`, and `metadata/{threshold_log10, prec, provisional, runtime_seconds}`.
  This includes data migrated by `migrate_legacy_spectra.jl`. Optional
  datasets such as `fpert_log10` do not affect recognition;
- F2 with `spectrum/masses/log10` present and no `spectrum/metadata` group;
- F3 with identity `schema_version == 1` and no label;
- F4 with no `schema_version` and no label, whose root holds
  `total_model_count` and `model_mass_reference_log10_ev`;
- F5 at `…-reproduction-1.2` or `…-kaehler-production-1.0` with no label;
- F6 at `…-orientifold-inflation-2.3` or `…-1.0` with no label. The F6
  version is `inflation/schema_version`; the nested compatibility marker
  `inflation/catastrophes/legacy_v1/schema_version`
  (`build_orientifold_vacua_inflation.jl:268–269`) is not the group version
  and does not trigger the "legacy version that carries a label" rejection;
- any other output covered by the R-006 general rule that is written by a
  *physical* function (R-008 classification), at its pre-change version (or
  with no version), with no label, as recorded in `plan.md`'s inventory;
- the F1 and F3 summary CSVs, and the inflation CSV families at their
  pre-change versions, with no `kinetic_normalization` column.

A legacy F2 layout records neither its writer (`hp_spectrum_save` or
`pq_spectrum_save`) nor its cubic basis. Its readers (at `10aa3dc`,
`read.cubic_tensor`, `read.jl:487`) SHALL report the basis in a
`cubic_basis::Symbol` field, `:unknown` for legacy F2 and `:charge_basis` or
`:canonical_mass_basis` for current F2; any consumer that needs the basis (for example to compare
or rescale cubic tensors) SHALL throw for a legacy F2 cubic tensor.

These SHALL be read as `:legacy_c_half`, and the result SHALL record that the
value was inferred.

**Legacy reproduction outputs.** An unlabelled pre-change output written by a
reproduction function is never inferred as `:legacy_c_half`. If `plan.md`'s
inventory identifies its layout as written only by a reproduction function,
it is read with that function's contract label (`:p96_reproduction_c_half` or
`:a96_reproduction_raw_radian`), recorded as inferred. Otherwise the reader
SHALL throw. At `10aa3dc` no repository reader consumes such outputs.

**Mixed legacy files.** A legacy file in which physical and reproduction rows
share one layout and cannot be told apart (possible for the inflation
diagnostic family) SHALL be rejected by physical
readers unless every writer of that legacy layout is recorded in `plan.md` as
purely physical.

**Readers SHALL throw** an `ArgumentError` (Julia) or a `ValueError`
(Python), naming the file and family, for:
- a current-version output (F1 `-4`, F2 `spectrum-groups-1`, F3 identity 2,
  F4 `model-stage-1`, F5 `reproduction-1.3` or `kaehler-production-1.1`, F6
  `2.4`, or an incremented inflation schema or contract version) whose label
  is missing, or a CSV row of such a family with an empty label;
- a label value outside R-006;
- a legacy version that carries a label;
- a stored `"legacy_c_half"` without `kinetic_normalization_inferred = true`
  next to it;
- a stored `"legacy_c_half"` in an F1–F6 HDF5 output (these primary families
  never hold derived values);
- a physical family whose label is neither `"sugra_c1"` nor, for a derived
  output only, a stored `"legacy_c_half"` with
  `kinetic_normalization_inferred = true`, except the per-row
  reproduction-labelled rows of per-row-labelled families. A physical reader of such a family SHALL return
  each row's label, and a physical aggregate SHALL filter out reproduction
  rows *before* applying the combination check to the remaining rows;
- a current-version F2 (`spectrum-groups-1`) whose `cubic_basis` is missing or
  unknown;
- an unknown version or layout.

**Definitions.** A *physical family* is any R-006 output family whose primary
writers are physical functions under the R-008 classification: F1–F6, the F1
and F3 summary CSVs, the inflation CSV families and the other outputs the
general rule covers. A *derived output* is a CSV or other file written by a
derived writer (R-006), for example `appendix_b_quantiles.csv` or an F3 CSV
written by `write_scan_csv` from read-back results; F1–F6 HDF5 outputs are
never derived outputs. A current-version derived output carrying a stored
`"legacy_c_half"` with `kinetic_normalization_inferred = true` is a *derived
legacy output*. Its readers SHALL return `(:legacy_c_half, true)`, exactly as
for an inferred legacy layout. It is never treated as complete by a batch run
(batch resume below) and is combined only with inputs of the same label.

**Primary HDF5 writers.** The F1–F6 HDF5 writers SHALL write only values they
compute from `Kinv` at the current convention, and SHALL refuse results or
inputs read back from a legacy or derived output. Layout migrations (below)
are not primary writers: they are exempt from this refusal and from the F2
writer refusal.

**Errors are not skipped.** A reader, aggregator, combiner or existence check
SHALL NOT catch and suppress an R-007 rejection or a combination error.
Existing catch-all handlers SHALL let these errors propagate, in particular:
- the `try … @warn` skip in `scripts/reproduce_appendix_b_spectra.jl:83–101`;
- both `_has_axion_photon` methods, the file method
  (`axion_photon.jl:2016–2041`) and the path method (`:2049–2075`), each of
  which returns `false` on any error. It is used by `run_batch_axion_photon` (line 2101) and by the
  `write_axion_photon_result` overwrite guard. A legacy or unlabelled F3 group
  SHALL NOT read as absent, so that the batch-resume `:failed` rule and the
  writer refusal apply to it.

**Writers and migration.**
- F2 writers SHALL refuse to write into a `spectrum` group that holds legacy
  or unlabelled F2 child groups, unless the caller passes an explicit replace
  option. That option removes only the F2 child groups
  `{masses, decay, quartdiag, quart31, quart22, cubic}` and `spectrum/metadata`
  before writing. F1 (`spectrum/physical`), F3 (`spectrum/axion_photon`) and
  any other child of `spectrum` SHALL NOT be touched.
- The F4 and F5 writers create new files; they SHALL refuse to overwrite an
  existing unlabelled or legacy file at the output path.
- CSV writers SHALL refuse to append rows to an existing file whose header
  lacks a `kinetic_normalization` column or whose schema version differs (for
  example `batch_physical_spectrum.jl --append-summary` onto a legacy summary).
- A forced F1 rewrite (`--force`, or the batch-resume overwrite below) SHALL
  delete the whole `spectrum/physical` group, including its `metadata`, before
  writing, so that no dataset from the earlier output survives. F2, F3 and
  other children of `spectrum` SHALL NOT be touched.
- **Layout migrations.** `scripts/migrate_quartic_index_ordering.jl` and
  `scripts/migrate_legacy_spectra.jl` move or reorder existing values without
  recomputing them. They are layout migrations, not derived writers, and the
  R-006 labelling duty does not apply to them. Each SHALL preserve any
  existing label and `schema_version` it copies, and SHALL NOT add a label or
  a `schema_version`. When its source is unversioned, the output of
  `migrate_legacy_spectra.jl` is therefore the unversioned F1 layout, read by
  inference as `:legacy_c_half`; a versioned source keeps its version and is
  read under that version's rule.

**Batch resume.** A physical batch or resumable run SHALL NOT treat an output
labelled or inferred as anything other than `:sugra_c1` (including a stored
derived `"legacy_c_half"`) as complete. This
covers `batch_physical_spectrum`, `run_batch_axion_photon`,
`build_orientifold_vacua_inflation.jl` (`_has_inflation_group`) and
`inflation_scan_prep.jl`'s resume check. Such an output is recomputed and
replaced only under the runner's explicit overwrite option: `force=true` for
`run_batch_axion_photon` and `build_orientifold_vacua_inflation.jl`, and
`--force` for `batch_physical_spectrum`. Otherwise it is reported as
`:failed`, with a message naming the stored label, as
`run_batch_axion_photon` already does for a configuration mismatch. For
`batch_physical_spectrum` this `:failed` rule applies only to outputs
labelled or inferred other than `:sugra_c1`. A `"sugra_c1"` output whose
configuration digest differs (the digest hashes `schema_version` and the git
`source_revision`, `batch_physical_spectrum.jl:176–183`) is silently
recomputed as today, and that rewrite also deletes the whole
`spectrum/physical` group first (forced-rewrite rule above).
`inflation_scan_prep.jl` has no overwrite option: with
`--append-summary` the R-007 append rule refuses a legacy summary, and the
remedy is a new summary path.

**Reader and combiner inventory.** `plan.md` SHALL list every reader and every
cross-file combiner of the R-006 families in `src/`, `ext/`, `scripts/` and
`bin/`, including those that bypass the package readers (for example
`scripts/reproduce_appendix_b_spectra.jl:83–101`, which reads
`spectrum/physical` directly and aggregates across files). Each SHALL apply
R-007.

**Combination.** Any CYAxiverse function or script that combines outputs
across files, calls or CSV rows SHALL reject inputs whose labels differ, with
an `ArgumentError` naming both values. No implicit conversion is provided.

### R-008 — Reproduction paths retained and distinguishable

The reproduction paths named under *Implementation facts* SHALL be numerically
unchanged, and so SHALL their tests. Their results SHALL carry
`kinetic_normalization` = `:p96_reproduction_c_half` or
`:a96_reproduction_raw_radian`, according to their contract. Any persisted
reproduction output SHALL carry the matching R-006 string. In particular:
- `inflation_refinement_common.jl` rows carry
  `"a96_reproduction_raw_radian"` (R-006 per-row labels);
- `inflation_reproduction.jl`, `benchmark_inflation_scalability.jl` and
  `author_poly102_reference.py` are A96 (including their N = 5 author-matrix
  functions, such as the call to `n5_kinetic_matrix`
  (defined at `poly102_inflation.jl:161`) at `inflation_reproduction.jl:100`);
- the `generate_appendix_c_geometry.py` Eq. 96 eigenvalue check is a P96
  reproduction validation and keeps `inv(kinv)`.

Physical readers SHALL reject reproduction labels, as R-007 requires.

**Classification rule.** Classification is per function (or per top-level
script entry point where there is no function). A function, existing or new,
is a reproduction function if and only if it consumes $M_{96}$, `N5_K_RAW`,
`N8_K_RAW`, `K_RAW` or the `reduced_models` source kinetic matrices, directly
or through `n8_kinetic_matrix`, `poly102_inflation`, `N8_BEST_X` or the other
`paper_benchmarks` reproduction APIs; checks against them (for example
`EXPECTED_K_EIGENVALUES`); or implements a spec 0148 contract. A file can
hold both physical and reproduction functions, as
`inflation_scale_continuation.jl` does.

**Search record.** `plan.md` SHALL record a search at the implementation base
for `M96`, `N5_K_RAW`, `N8_K_RAW`, `K_RAW`, `reduced_models`,
`n8_kinetic_matrix`, `n5_kinetic_matrix`, `n5_potential`, `n8_geometry`,
`author_inflation`, `poly102_inflation`, `N8_BEST_X` and
`EXPECTED_K_EIGENVALUES` over `src/`, `ext/`, `scripts/`, `bin/`, `test/`, `specs/`,
`docs/` and `validation/`, and SHALL classify every hit.
- Hits in `test/`, `docs/`, `validation/` and `specs/` are classified without
  stopping the work.
- A function in `src/`, `ext/`, `scripts/` or `bin/` that the rule makes a
  reproduction function and that is not in the list above is added to
  `plan.md` with its contract (P96 or A96); this needs no spec amendment.
- The work stops for a spec amendment only if such a function writes a
  persisted output that the R-006 tables do not already cover, or if the
  rule does not decide between P96 and A96.

### R-009 — Documentation

A documentation page on conventions SHALL state:
- the definition, units, factor contract and K-argument contract;
- the source-facts table, including the arXiv:2103.06812 readings and the
  arXiv:2512.00144 inconsistency;
- the open inferences;
- the reproduction contracts: P96 at $c = \tfrac12$, and A96 as raw-radian;
- the full R-005 scaling tables;
- the label and schema history, and the inference rules.

These SHALL point to it:
- the affected docstrings, including a replacement for the `_kinetic_matrix`
  "`K = (1/2) g`" text and for `_validate_kinetic_matrix`;
- the README;
- the axion–photon documentation page.

### R-010 — Certification boundary

IF a quantity in `test/fixtures/scientific_certification` is built from
`Kinv` through a path covered by R-001, the work SHALL stop for owner
approval before that fixture is re-certified. Fixtures that supply $K$
directly follow the K-argument contract. They SHALL be shown to be unchanged.

### R-011 — Inflation reciprocity

Every reciprocity check between a kinetic matrix and `Kinv` SHALL use the
R-001 relation $K_{\rm kin}\cdot\mathtt{Kinv} = 2I$ (as in
`inflation_scale_continuation.jl`). A test SHALL show that a physical-mode
geometry passes the check.

## Acceptance gates

### CYAX-0232 G0 — Specification approval

**Objective:** an approved specification revision.

**Acceptance:** an independent specification review in a fresh context,
followed by an owner approval comment on Issue #232 that names the reviewed
revision.

**Stop condition:** a normative gap is found.

### CYAX-0232 G1 — Implementation candidate

**Objective:** R-001 to R-011 implemented.

**Acceptance:**
- the fast and full suites, the Python regressions, the docs build and
  `agent_verify.py` all pass at the exact SHA;
- every changed test is listed with its reason;
- the R-002 baseline hashes are recorded;
- the changed counts required by R-005 are reported.

**Stop condition:** R-002 fails, or R-010 requires re-certification.

### CYAX-0232 G2 — Independent review

**Objective:** an independent judgement of the whole PR.

**Acceptance:** a fresh-context scientific and numerical review of the whole
PR at the exact SHA, plus CI green.

**Stop condition:** a blocking finding, which produces a new candidate.

## Verification requirements

| Evidence | Requirement |
|---|---|
| Factor identity, orientation guard and analytic fixtures | R-004 |
| Named source fixture and discrimination check | R-003 |
| Scaling tables, including inflation and the policy-dependent axion–photon outputs | R-005 |
| Pinned bitwise baseline | R-002 |
| Round-trip per family (F1–F6, CSVs); each inference and rejection case, including unversioned F1 without `fpert_log10`; F2 replace leaving F1 and F3 intact; append refusal; resume behaviour for each named runner; mixed-label combination; per-row inflation labels; `stored_kinetic_normalization` for F3 | R-006, R-007 |
| Writer, reader and combiner inventory in `plan.md` under the R-006 general rule | R-006, R-007 |
| Reproduction paths unchanged and labelled; classification search recorded | R-008 |
| Docs build | R-009 |
| Certification audit | R-010 |
| Reciprocity test | R-011 |
| Equivalent test for the Python path | R-001 |

## Interfaces and compatibility

**Public API.**
- Signatures and exported struct layouts are unchanged.
- New exports: `KINETIC_NORMALIZATION`, `kinetic_factor`, `kinetic_matrix`
  and `stored_kinetic_normalization`.
- `NamedTuple`-returning readers gain two fields (R-006).
- APIs that take K keep their meaning under the K-argument contract.
- Physical outputs computed from `Kinv` change by the R-005 factors.

**Persisted schemas.**
- F1 moves to `-4`.
- F2 gains `spectrum-groups-1` metadata.
- F3 identity moves to version 2.
- F4 gains `fuzzy-axion-model-stage-1`; F5 moves to `reproduction-1.3` and
  `kaehler-production-1.1`; F6 moves to `orientifold-inflation-2.4`.
- The inflation CSV schema versions are each incremented by one, and the scan
  contract version moves to `"5"`.
- Legacy outputs, including unversioned F1 and F4, remain readable as
  `:legacy_c_half`.

**Downstream.** PR #219's readers, which accept `-3` only, need their own
update.

**Package-version impact:** `minor`. The version is not bumped on the feature
branch.

## Dependencies and blockers

- PR #219 depends on this specification.

## Open owner decisions

Decisions 1–7 are recorded on Issue #232. One interpretation needs the
owner's explicit confirmation in the approval comment:

- **K-argument contract.** The Issue #232 body lists `_canonical_factor_from_K`
  and its callers as $c = \tfrac12$ paths, and its risks section names the
  test "HP spectrum: one-axion analytic mass" as one that may change. This
  specification instead makes the K-taking APIs convention-neutral (R-001(b)):
  they inherit $c = 1$ from `read.potential`, which now supplies
  $K_{\rm kin} = 2\,\mathtt{Kinv}^{-1}$, and that test stays unchanged
  (R-004). The approval comment SHALL name this contract.

The durable `approval_ref` will be the owner comment on Issue #232 that
approves the reviewed revision.

## Completion criterion

R-001 to R-011 pass at an approved, reviewed candidate merged into `vmm` by the
owner.
