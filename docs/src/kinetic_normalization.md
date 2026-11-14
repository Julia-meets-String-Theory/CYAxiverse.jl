# Kinetic normalization

This page states the axion kinetic normalization that CYAxiverse uses, how
persisted outputs record it, and how outputs written under the earlier
convention are read. The governing specification is CYAX-0232
(`specs/0232-kinetic-normalization/spec.md`, Issue #232).

## The convention

Every physical axion path uses the kinetic matrix

```math
K_{\rm kin} = 2\,\partial_T\partial_{\bar T}\mathcal{K} = 2\,\mathtt{Kinv}^{-1}
```

in

```math
\mathcal{L}_{\rm kin} = -\tfrac12 M_{\rm pl}^2 K_{\rm kin}\,\partial\vartheta\,\partial\vartheta ,
```

where ``\vartheta`` has period one and `Kinv` is the inverse Kähler metric
``g^{-1}`` stored by CYTools. This is the literal ``\mathcal N = 1``
supergravity normalization, written ``c = 1`` in the common measure
``\mathcal L_{\rm kin} = -c\,M_{\rm pl}^2 g\,\dot\vartheta\dot\vartheta`` with
``g = \partial_T\partial_{\bar T}\mathcal K`` and ``\mathcal K = -2\log\mathcal V``.
`CYAxiverse.KINETIC_NORMALIZATION == :sugra_c1` names it. Before CYAX-0232 the
physical paths used ``c = \tfrac12``.

Units: ``M_{\rm pl}`` is the reduced Planck mass; masses are reported as
`log10(m/eV)`, decay constants as `log10(f/GeV)`.

**Shared functions.**
- `kinetic_factor(Kinv)` returns the canonical factor
  ``B = \mathrm{cholesky}(\mathtt{Kinv}/2).L``, lower triangular with positive
  diagonal and ``BB^T = \mathtt{Kinv}/2 = K_{\rm kin}^{-1}``.
- `kinetic_matrix(Kinv)` returns ``K_{\rm kin} = 2\,\mathrm{inv}(\mathtt{Kinv})``.
- The kinetic norm of a charge column ``q`` is ``q^T(\mathtt{Kinv}/2)q``.

Every physical conversion of `Kinv` into kinetic data uses one of these. The
factor ½ is applied only there. Every stored or exported inverse metric
(`Kinv`, the Python `inverse_metric`, `kinv` in scripts) stays ``g^{-1}``, and
the potential uses ``K^{a\bar b} = \mathtt{Kinv}`` unchanged.

**Coordinates.** ``\vartheta = B\psi`` gives
``\mathcal L_{\rm kin} = -\tfrac12 M_{\rm pl}^2\dot\psi^T\dot\psi``. A charge
column ``q`` maps to ``\hat q = B^Tq``, and a ϑ-Hessian ``H`` to ``B^THB``.

**K-argument contract.** An API that takes a matrix argument named `K` (or
documented as the kinetic matrix) takes the matrix in
``-\tfrac12M_{\rm pl}^2K\,\partial\vartheta\partial\vartheta`` for period-one
ϑ, and never rescales it. For stored CYTools data, pass
`CYAxiverse.kinetic_matrix(Kinv)`. An API that takes a canonical factor `C`
takes one with ``CC^T = K^{-1}``; `CYAxiverse.kinetic_factor(Kinv)` is such a
factor. Results computed from a caller-supplied `K` are not labelled by the
package.

`read.K` reads `cytools/potential/K`, which no repository writer produces. It
is unlabelled and is not the physical kinetic-matrix accessor.

## Sources

The common measure above is the reference. Equation numbers for
arXiv:2103.06812 and arXiv:2112.04503 are those of the v2 HTML rendering; the
other entries use the latest arXiv version as of 2026-11-10.

| Source (version) | Displayed normalization | ``c`` |
|---|---|---|
| arXiv:2412.12012 (latest as of 2026-11-10) | Eq. 3.1: ``-\tfrac12M^2K``. Eq. 3.11: ``K = 2\partial_T\partial_{\bar T}\mathcal K``. Eq. 3.13 gives the F-term inverse metric | 1 |
| arXiv:2309.13145 (latest as of 2026-11-10) | Eq. 3.6: ``-\tfrac{M^2}{2}\mathcal K_{ab}``. Eq. 3.9: ``\mathcal K_{ab} = 2\partial_T\partial_{\bar T}\mathcal K``. Eq. 3.7: inverse ``= 2(\tau\tau^T - \mathcal V\kappa t)``. Eq. 2.2: ``\cos 2\pi\mathcal Q\theta`` | 1 |
| arXiv:2608.14780 (latest as of 2026-11-10) | Eq. 22: ``\tfrac12K``. Eq. 23: ``K = \tfrac12\partial_\tau\partial_\tau\mathcal K``. Period one is stated | 1 |
| arXiv:2103.06812 (v2) | Eq. (1): ``-\tfrac{1}{8\pi^2}M^2K`` with ``\cos(\mathcal Q\theta+\delta)``, which implies period 2π. ``K = \partial_\tau\partial_\tau\mathcal K`` (text after Eq. (5)). Eq. (6): ``W \supset e^{-2\pi q\cdot T}``, which implies period one | 2 under period 2π; ``1/(2\pi^2)`` under period one |
| arXiv:2512.00144 (latest as of 2026-11-10) | Eq. C.18: ``[K^{-1}] = 2(\tau\tau^T - \mathcal V\kappa t)``, which implies ``c = 1``. Eq. A.3 gives ``K_{ij} = \partial_T\partial_{\bar T}\mathcal K`` and Eq. II.4 writes the τ kinetic term ``K^{ij}d\tau_i\wedge\star d\tau_j``, both with no ½; Eq. A.6 uses ``\cos(q\phi)``. These are inconsistent with each other | inconsistent |
| arXiv:2112.04503 (v2) | Eq. (41): ``-\tfrac12K_{AB}``; ``K`` is not defined precisely | undetermined |
| arXiv:1808.01282 (latest as of 2026-11-10) | Eq. 4.3: ``-\tfrac{M^2}{2}K``; ``K`` is not defined precisely | undetermined |
| arXiv:2309.01831 (latest as of 2026-11-10) | No kinetic term. Eq. 9 writes the pair term as ``\pi q^TK^{-1}q'`` | n/a |

The arXiv:2103.06812 v2 equations (1), (5) and (6) are numbered 1.1, 2.4 and
2.5 in the Issue #232 correction comment. The arXiv:2512.00144 entries II.4,
A.3 and A.6 follow the CYAX-0232 G0 review's reading of the paper; C.18 was
verified directly.

**Open inferences** (derived, not stated by the sources):
- The published arXiv:2608.14780 numbers use ``c = \tfrac12``: its Eq. 96
  eigenvalues equal eig(``g``), within 0.07 %, on the reconstructed N = 8
  geometry.
- The ``K^{-1}`` of arXiv:2309.01831 Eq. 9 is ``g^{-1}``, the CYTools `Kinv`:
  the F-term cross term derived with ``K^{a\bar b}\mathcal K_{\bar b} = -2\tau^a``
  reproduces Eq. 9 only with this identification.
- The arXiv:2412.12012 numerics use ``c = 1``: CYAX-0232 R-003 reproduces its
  Eq. 4.3 decay constant within 5 % at ``c = 1`` and misses it by more than
  20 % at ``c = \tfrac12``.

## Reproduction contracts

The reproduction paths for arXiv:2608.14780 (Issue #148, spec 0148) keep their
own contracts and are numerically unchanged:

- **P96** (`:p96_reproduction_c_half`): ``K_\theta = M_{96}/k^2`` with period-one
  θ, which is ``c = \tfrac12`` because ``M_{96} \approx g``. Used by
  `paper_benchmarks` `reduced_models.jl`, `n8_continuation.jl` and
  `n8_g3_control.jl`.
- **A96** (`:a96_reproduction_raw_radian`): the author's raw-radian
  ``M_{96}/k^2`` (`N8_K_RAW`); in period-one coordinates
  ``K_\theta = (2\pi)^2M_{96}/k^2``, about ``c = 2\pi^2``. The same label covers
  the N = 5 author matrix `N5_K_RAW`. Used by `poly102_inflation.jl`,
  `catastrophe_diagnostics.jl` and the reproduction scripts.

Reproduction results that are structs or `NamedTuple`s carry their label in
`kinetic_normalization`; a result that mixes the two contracts carries
`kinetic_normalization_by_field` instead. The persisted comparison CSV of
`scripts/inflation_reproduction.jl` mixes them too: it has one
`<column>_kinetic_normalization` label column for each value column and no
row-level label. Functions that return a bare matrix,
such as `n5_kinetic_matrix` and `n8_kinetic_matrix`, state their contract in
their docstrings. Reproduction functions that use no kinetic matrix carry no
label. Physical readers reject reproduction labels.

## Scaling relative to the earlier convention

Each output at ``c = 1``, compared with a ``c = \tfrac12`` evaluation of the
same inputs (equivalently, with `2 Kinv` in place of `Kinv`), changes by the
factor below. Outputs that keep only modes above a mass threshold compare the
modes present in both evaluations.

**Spectrum (F1, F2, in memory)**

| Output | Factor |
|---|---|
| ``f_K``, ``f_{\rm pert}`` | √2 |
| masses | 1/√2 |
| ``\lambda_{\rm self}``, ``\lambda_{31}``, ``\lambda_{22}`` | 1/4 |
| cubic tensor, `pq_spectrum_save` (canonical mass basis, evaluation phase fixed in ϑ) | ``2^{-3/2}`` (up to one sign per mode) |
| cubic tensor, `hp_spectrum_save` (ϑ/charge basis) | 1 |

**Axion–photon (F3)**

| Output | Factor |
|---|---|
| ``f`` | √2 |
| mixing Θ from `mixing_matrix` | 1 |
| persisted `hierarchy/q` and `hierarchy/theta_from_canonical` | 1/√2 |
| ``g_{a\gamma\gamma}``, `log10_g_effective` under `:divisor_instanton` | 1/√2 |
| `log10_g_effective` for light modes under `:electron_proxy` | ``2^{-3/2}`` |
| persisted `hierarchy/log10_f_GeV` | ``+\tfrac12\log_{10}2`` |
| persisted `hierarchy/log10_mass_eV` | ``-\tfrac12\log_{10}2`` |
| `log10_g_effective` for heavy modes under `:electron_proxy` | 1/√2 |
| QED threshold under `:divisor_instanton` | 1/√2 |
| QED threshold under `:electron_proxy` (by default the electron mass) | 1 (fixed) |
| photon width, all modes under `:divisor_instanton` | ``2^{-5/2}`` |
| photon width, heavy modes under `:electron_proxy` | ``2^{-5/2}`` |
| photon width, light modes under `:electron_proxy` | ``2^{-9/2}`` |
| quartic width | ``2^{-9/2}`` |
| `light_mode_count` under `:divisor_instanton` | unchanged |
| `light_mode_count` under `:electron_proxy` | may change |

The axion–photon decay constants equal arXiv:2309.13145 Eq. 2.15, with the
Gram–Schmidt step taken in the Eq. 3.7 inverse metric.

**Inflation (in memory and CSV)**

| Output | Factor |
|---|---|
| ε, η, every eigenvalue of ``K^{-1}H`` | 1/2 |
| `mass_min`, `mass_max` (log10 eV) | ``-\tfrac12\log_{10}2`` |
| e-fold counts along a fixed trajectory | about 2 in the slow-roll approximation |
| candidate classifications, mode counts, F6 candidate and e-fold counts | may change |

**Fuzzy-axion model stage (F4, F5)**

| Output | Factor |
|---|---|
| model-stage masses at fixed inputs, including F4 `model_mass_reference_log10_ev` | 1/√2 (``-\tfrac12\log_{10}2`` in log10 fields) |
| the F5 mass evaluator `_fuzzy_axion_mass_log10` at a fixed Kähler point | 1/√2 |
| F4 `model_lambda` and model counts; F5 optimizer outputs | may change |

## Labels in persisted outputs

The label is `kinetic_normalization`, a scalar UTF-8 string dataset in HDF5,
a CSV column on every row, or a top-level JSON key. Its values:

| Value | Meaning | Written by |
|---|---|---|
| `"sugra_c1"` | ``K_{\rm kin} = 2\,\mathtt{Kinv}^{-1}`` | physical writers |
| `"p96_reproduction_c_half"` | the P96 contract | P96 reproduction writers |
| `"a96_reproduction_raw_radian"` | the A96 contract | A96 reproduction writers |
| `"legacy_c_half"` | the earlier physical convention | derived writers only, with `kinetic_normalization_inferred = true` |

| Family | Label location | Current version (previous) |
|---|---|---|
| F1 physical spectrum | `spectrum/physical/metadata/kinetic_normalization` | `cyaxiverse-physical-spectrum-4` (`-3`, or unversioned) |
| F1 batch summary CSV | `kinetic_normalization` column | unversioned |
| F2 spectrum groups (`hp_spectrum_save`, `pq_spectrum_save`) | `spectrum/metadata/kinetic_normalization`, plus `cubic_basis` | `cyaxiverse-spectrum-groups-1` (unversioned) |
| F3 axion–photon result | `spectrum/axion_photon/identity/kinetic_normalization` | identity version 2 (1) |
| F3 scan CSV | `kinetic_normalization` and `kinetic_normalization_inferred` columns | unversioned |
| F4 model-stage output | root `kinetic_normalization` | `cyaxiverse-fuzzy-axion-model-stage-1` (unversioned) |
| F5 reproduction JSON | top-level `kinetic_normalization`; also in each `kaehler-production-1.1` object | `…-reproduction-1.3` (1.2, 1.1, 1.0) |
| F5 model-stage input HDF5 | root `kinetic_normalization` | `kaehler-point-provenance-2` (`-1`) |
| F6 orientifold inflation group | `inflation/kinetic_normalization` | `…-orientifold-inflation-2.4` (2.3, 2.2, 2.1, 2.0, 1.0) |
| Inflation CSV families | `kinetic_normalization` column, per row | diagnostic 2, contract 5, shard 3, refinement 4, flow 2, pilot 7 (each one lower before) |
| Derived outputs (`appendix_b_quantiles.csv`, merged shard CSVs, the pilot report, `merge_orientifold_shards.py` merges) | the inputs' common label, with `kinetic_normalization_inferred` where the input label was inferred | `shard-merge-1.1` (1.0) for the orientifold merge |

The exported `stored_kinetic_normalization(path; family)` (defined as
`CYAxiverse.read.stored_kinetic_normalization`) returns
`(label, inferred)` for the HDF5 families F1, F2, F3, F4 and F6; readers that
return a `NamedTuple` (`physical_spectrum`, `read.pq_spectrum`,
`read.hp_spectrum`, `read.cubic_tensor`) carry the fields
`kinetic_normalization` and `kinetic_normalization_inferred`. An in-memory
`AxionPhotonResult` keeps its stored identity version; map it with
`CYAxiverse.read.f3_identity_label`. The Python readers use
`scripts/kinetic_normalization_labels.py`.

## Reading earlier outputs

Outputs written before the change carry no label. They are read as
`:legacy_c_half` with `inferred = true` only when their layout is recognized:

- F1 at `-3` without a label, or unversioned and holding every dataset
  `physical_spectrum` requires (optional datasets such as `fpert_log10` do not
  matter; this includes data copied by `migrate_legacy_spectra.jl`);
- F2 with `spectrum/masses/log10` and no `spectrum/metadata`;
- F3 at identity version 1 without a label;
- F4 unversioned with `total_model_count` and `model_mass_reference_log10_ev`;
- F5 at `…-reproduction-1.2` without a label, or at 1.0 or 1.1 when the file
  matches one of the recorded historical layouts
  (`specs/0232-kinetic-normalization/layouts/`,
  `scripts/f5_legacy_layouts_v10a.py`, `_v10b.py`, `_v11.py`);
- F6 at 2.3, 2.2, 2.1, 2.0 or 1.0 without a label (the group version is
  `inflation/schema_version`; the nested
  `catastrophes/legacy_v1/schema_version` marker is not it);
- inflation CSV rows without the label column that match exactly one recorded
  historical entry (header, `row_type`, version values; see
  `scripts/inflation_csv_labels.jl`);
- the unversioned F1 summary and F3 scan CSVs, and the earlier
  `shard-merge-1.0` merges.

Readers raise an `ArgumentError` (Julia) or `ValueError` (Python) naming the
file for a current version without a label, a label outside the table, a
legacy version that carries a label, `"legacy_c_half"` or a reproduction label
in a primary physical family, an F2 `cubic_basis` that is missing or unknown,
and an unknown version or layout. These errors are never suppressed: existence
checks and batch readers let them propagate.

**Combining outputs.** Any function that combines outputs across files, calls
or rows rejects inputs whose labels differ, naming both. No conversion between
conventions is provided.

**Batch resume.** A batch run never treats an output labelled or inferred as
anything other than `:sugra_c1` as complete. It is recomputed only under the
runner's overwrite option (`--force` for `batch_physical_spectrum.jl`,
`force=true` for `run_batch_axion_photon` and Pipeline 2); otherwise it is
reported as failed (Pipeline 2: blocked) with the stored label. CSV writers
refuse to append to a file whose header lacks the label column, and
`run_scale_continuation --resume` refuses a report or shard file that holds
earlier rows; the remedy is a new output path.

**Migrations.** `migrate_legacy_spectra.jl` and
`migrate_quartic_index_ordering.jl` move values without recomputing them. They
add no label or version and refuse a current-version target.

## Notes on specific outputs

- **Earlier F2 cubic tensors are unreliable.** Before CYAX-0232 both F2
  writers called the cubic evaluator with mis-oriented instanton data, and an
  earlier F2 file records neither its writer nor its cubic basis. Readers report
  `cubic_basis = :unknown` for such files, and
  `read.require_cubic_basis` refuses them. Do not compare or rescale an earlier
  cubic tensor; recompute it.
- **Meaning of an inferred F5 label.** For F5 files at 1.0 or 1.1 the label
  states the convention of the file's kinetic-matrix-dependent fields: the
  values the Julia model stage computes (`model_stage.total_model_count`,
  `model_count_per_record`, `models[]`) and fields derived from them, such as
  `claim_status.model_count`. Population counts, statuses, geometry and the
  stored `inverse_metric` do not depend on the convention. A file with no such
  field (no model stage) is still read as inferred `legacy_c_half`, so that
  combining it with current outputs fails closed: this records the file's
  provenance, not a statement that those values carry the earlier
  normalization.
- **Not certified.** `scripts/scan_fresh_relaxed_qcd_candidates.py` passes
  `kinv` to a module that is not in the repository. CYAX-0232 does not certify
  its outputs under either convention, and it writes no label; physical
  readers treat its outputs as an unknown layout.
