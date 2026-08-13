# Local axion--photon scan

This page documents the bounded axion--photon scan implemented in
`CYAxiverse.axion_photon`. It is designed to run against the complete geometry
files already present in
`/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data`.

The formulas are the leading EFT hierarchy, mixing, photon-coupling, and
width relations described in *Glimmers from the Axiverse*,
[arXiv:2309.13145](https://arxiv.org/abs/2309.13145), applied to the local
CYAxiverse file convention. The paper is retained as a scientific reference;
the API names describe the calculations performed here.

The scan implements the paper's core hierarchy and photon-coupling routines:

- sort the stored instanton terms by their package log scale and retain the
  first independent charge columns;
- construct the canonical upper-triangular charge matrix in package layout;
- compute log decay constants and log masses;
- expand an explicitly chosen EM charge in the reduced charge basis;
- compute the leading mixing matrix, `Cgamma`, and photon couplings;
- apply the below-QED-threshold coupling proxy;
- compute the two leading width estimates in log units; and
- aggregate compact per-geometry diagnostics into a CSV file.

The geometry generator also has a pilot visible-sector mode,
`--visible-sector-policy intersecting_d7`. Given an explicit validated O3/O7
involution with `h11_minus=0`, it selects an invariant QED divisor intersecting
the selected QCD divisor, stores both divisor charges and their metadata, and adds the QED
Euclidean D3 term needed for the paper-style light threshold. This is a
geometry-level compatibility filter and scale assignment; it is not a full
D7 tadpole, matter-spectrum, flux, or E3 zero-mode construction.

For the geometry-generation track itself, use
`--moduli-policy canonical_qcd`. It samples a pairwise-intersecting triangle of
prime toric divisors before finding the stretched-cone tip, chooses one member
as QCD (or uses `--qcd-divisor-index`), and applies the homogeneous scale
`m = sqrt(40 / vol(D_QCD)_tip)`. The selected triple and QCD member are stored
under `cytools/geometric/standard_model`; this is the paper-level geometric
assignment, not a complete brane construction.

This is an adapted local scan, not a reproduction of the paper's 200,000-model
ensemble. In particular, the local charge convention and local potential
generator are recorded explicitly by the reader and result status.

## Matrix layout and performance

The input arrays keep the package layout exactly: `Q` is `h11 × N` and `L` is
`2 × N`. The hierarchy matrix `q` is different from the input `Q`: its rows
label canonical axions and its columns label selected instantons, so the
column-oriented implementation uses

```text
q = X' * Q_reduced
```

The result is upper triangular. The implementation does not transpose `Q` or
materialize a transposed `Q` array. The stored `Q_reduced` remains
column-oriented.

For the local Float64-backed charge datasets, the loader converts to the
package's `Matrix{Int}` convention in reusable 4096-column chunks. The scan
also opens each selected HDF5 file once, reuses modular-rank work buffers,
avoids forming `K = Kinv⁻¹`, and solves the EM expansion with a Float-valued
factorization. These choices reduce both temporary memory and generic integer
factorization costs at larger h11.

## Run the scan

From the package checkout:

```bash
cd /Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl
julia --project=. scripts/run_axion_photon_scan.jl \
  --data-dir /Users/vmehta/Documents/CYAxiverse/cyaxiverse/data \
  --h11 15,100,200,300 \
  --limit 2 \
  --output /private/tmp/cyaxiverse-axion-photon.csv
```

The default slices are `h11 = 15, 100, 200, 300`, with two deterministic
files per slice. The script does not write into the repository. Use
`--em-divisor-index N` to use a fixed one-based column of `effective_cone` for
the EM charge. If it is omitted, the scan uses the direct divisor with the
smallest stored `prime_divisor_volumes` value, unless the file contains a
`visible_sector` group, in which case it uses the stored QED divisor charge.
That fallback is a reproducible technical choice, not an identification of the
Standard Model divisor or the QCD divisor.

For a generated visible-sector file, use the stored QED instanton threshold
instead of the electron-mass proxy with:

```bash
julia --project=. scripts/run_axion_photon_scan.jl \
  --data-dir /path/to/generated/data \
  --qed-threshold divisor_instanton
```

The derived mode intentionally refuses legacy files without visible-sector
metadata.

The output is a compact CSV summary. The complete arrays remain available in
the returned Julia objects:

```julia
using CYAxiverse
const A = CYAxiverse.axion_photon

results = A.run_local_scan(
    data_dir="/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data",
    h11s=(15, 100, 200, 300),
    limit_per_h11=2,
)

r = results[1]
r.geometry.index
r.hierarchy.log10_mass_eV
r.hierarchy.log10_f_GeV
r.photons.Cgamma
r.photons.log10_g_GeVinv
r.photons.log10_g_effective_GeVinv
r.photons.log10_photon_width_GeV
r.photons.log10_quartic_width_GeV
```

All mode arrays are ordered by decreasing stored `log10(Λ⁴)` scale. Indices
reported by `selected_indices`, `dependent_indices`, and
`source_indices` are one-based Julia column indices into the original `Q`.

## Local input convention

`load_instanton_data(path)` reads the package-shaped datasets without reorienting
them:

```text
cytools/potential/Q  : h11 × N charge matrix
cytools/potential/L  : 2 × N package scale matrix
```

The package convention is `L[1, i] = sign/mantissa` and
`L[2, i] = log10 scale`. The scan retains the sign from the first row and
orders by the second row, matching the existing CYAxiverse spectrum routines.
The local files normally use `±1` for the first row.

`load_geometry_inputs(path)` requires the following fields:

```text
cytools/geometric/tip
cytools/geometric/divisor_volumes
cytools/geometric/CY_volume
cytools/geometric/Kinv
cytools/geometric/effective_cone
cytools/geometric/prime_divisor_volumes
cytools/geometric/prime_toric_divisors
```

Visible-sector files additionally expose `geometry.visible_sector`, including
the QCD/QED divisor indices, divisor charges, involution images, and the
zero-based source column of the stored QED instanton term. The loader converts
that source column to Julia's one-based indexing.

For this scan, each column of `effective_cone` is treated as a direct divisor
charge vector. The mapping is exposed as
`geometry.direct_charges`, `geometry.direct_divisor_volumes`, and
`geometry.direct_labels`. This is the local-file convention used by the
scan; it is not asserted to be the paper's exact SM/QCD divisor construction.
Legacy files missing these fields are skipped when
`require_complete=true` (the default).

## What the numerical result means

For the selected charge columns `Q_reduced`, the routine constructs `X` and `q` such
that

```text
theta = X * phi,
X' * K * X = 1,
q = X' * Q_reduced,
```

with `q` upper triangular up to numerical residuals. The result stores `X` as
`theta_from_canonical`, `q` as `q`, and reports
`metric_residual` and `triangular_residual`.

The log observables use the paper's leading hierarchy expressions:

```text
f_a = M_Pl / (2π q_aa)
m_a² = Λ_a⁴ / f_a²
```

`log10_f_GeV` and `log10_mass_eV` are stored instead of exponentiating the
very wide local scale range. The photon routine solves, in the same
column-oriented convention,

```text
Q_EM = Q_reduced * n_EM
C_gamma,a = Σ_b n_EM^b Θ_ba
g_aγγ = α_EM C_gamma,a / (2π f_a)
```

and stores the solve residual as `charge_residual`. `log10_g_GeVinv` is the
unsuppressed EFT coupling. For modes below the configurable
`light_threshold_eV` (default: the electron mass),
`log10_g_effective_GeVinv` applies the paper's parametric
`m_a² / m_QED²` suppression. The default `qed_threshold_policy` is
`electron_proxy`, with `m_QED = m_e`. The `divisor_instanton` policy instead
uses the stored QED E3 scale and charge norm:

```text
log10(m_QED/eV) = 1/2 log10(Λ_QED⁴)
  + log10(M_Pl/GeV) + 9 + log10(2π)
  + 1/2 log10(Q_QED K⁻¹ Q_QEDᵀ)
```

The latter requires `visible_sector` metadata and keeps the calculation in
log space to avoid underflow.

`log10_photon_width_GeV` uses `m³ g²/(64π)`. The quartic estimate uses the
nearest hierarchy neighbor `b = a + 1` and the paper's
`λ_abbb² m_a/(128π³)` scaling. The final mode has no next neighbor and is
reported as `-Inf` for that estimate. These are leading estimates, not a
full decay-channel calculation.

## Signed-scale policy

The hierarchy routine defaults to `signed_scale_policy=:require_positive`.
This is the strict-positive leading-instanton guard for the selected terms. The local
representative files at the default slices pass this guard after ordering by
the stored `L[2, :]` values, although later dependent terms can be negative.

If a future local subset selects a non-positive independent term, an explicitly
adapted run can be requested with:

```bash
julia --project=. scripts/run_axion_photon_scan.jl \
  --h11 15,100 --limit 1 --absolute-scales
```

Such rows are marked `adapted_absolute_scale`. This option takes absolute
values only for the scale used in the hierarchy; the original coefficient
signs remain in `InstantonData.coefficient_signs`.

## Scope boundary

The current scan is intended to answer bounded numerical questions such as:

- does the selected local hierarchy produce a stable canonical frame?
- how do `f`, `m`, `Cgamma`, and `g` vary across the chosen local slices?
- how sparse are the leading photon couplings in this local sample?
- which of the leading photon and quartic width estimates dominates?

It does not support claims about the paper's population-level results. The
following remain deferred:

- population-level reconstruction of the paper's divisor/SM/QCD sampling procedure;
- the missing `h11=491` slice and the paper's full multiplicities;
- global D7 tadpole cancellation, matter-spectrum engineering, flux choices,
  and a proof of the E3 instanton's required zero modes;
- QCD mass normalization, GUT versus non-GUT population lines, or external
  X-ray/helioscope limits;
- birefringence predictions;
- reheating, freeze-in/out, decay-history, dark-radiation, and DM-composition
  calculations; and
- CP-breaking phases and the appendix resonance/channel analysis.

Use the historical handoff
`/Users/vmehta/Documents/CYAxiverse/cyaxiverse/HANDOFF_AXION_PHOTON_LOCAL_SCAN.md`
for the original reproducibility contract and next-agent checklist. It is
retained for provenance; new code and documentation use the science-based
axion/photon names above.
