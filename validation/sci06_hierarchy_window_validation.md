# SCI-06 hierarchy/window spectrum validation

Status: scientifically validated on `science/spectrum-validation`; ready for
the separate CODE-03 integration review after this branch is reviewed.

Validation date: 2026-08-11

Starting feature: `faxiverse-spectrum` at `f2b8085`

## Scope and boundary

This validation covers the leading-Hessian axion spectrum only. The convention
is

\[
  W = K_L^{-1} H K_L^{-T}, \qquad
  H = \sum_a \Lambda_a^4 q_a q_a^T,
\]

with `L` storing signed mantissas and base-10 logarithms. The hierarchy and
window code does not select CYTools instantons, define a KS ensemble, infer a
Bayesian geometry distribution, or make an inflation claim.

The exact high-precision eigensystem in `pq_physical_spectrum` remains the
reference oracle for the bounded tests. The new API is
`pq_window_spectrum`; its result keeps zero-based full-spectrum mode indices,
inclusive mass-window bounds, quartic semantics, and visible fallback status.

## Certification meaning

`instanton_scale_blocks` performs a configurable descending log-scale
clustering without materializing `10^log_scale`. For each adjacent block split,
`instanton_hierarchy_diagnostics(K, L, Q)` reports:

- `off_block_norm`: canonical charge cross-coupling norm;
- `separation_gap`: the inter-block base-10 log-scale gap;
- `coupling_to_gap_ratio`: the coupling divided by the scale ratio, evaluated
  with an overflow-safe large-gap branch;
- `certified_safe`: true only for the conservative current screen
  `separation_gap >= 6` and canonical coupling `<= 1e-6`.

This is a numerical screening certificate, not a theorem that the physical
Hamiltonian is exactly block diagonal. If any proposed split fails it,
`pq_window_spectrum` uses the complete arbitrary-precision eigensystem rather
than reporting a targeted hierarchy result.

For a targeted result, the window diagnostics are certified only when all of
the following hold:

1. arbitrary-precision inertia gives a stable lower and upper count (when
   `confirm=true` is used);
2. refined eigenvectors satisfy the requested relative residual tolerance;
3. every returned mass lies in the inclusive requested interval; and
4. the returned mode count equals the two-boundary inertia count.

If targeted refinement or the hierarchy screen fails, the full high-precision
eigensystem is used. If that reference result still fails residual or interval
validation, `provisional=true` remains visible. The configurable
`boundary_margin_log10` is the symmetric guard used when Float64 mass bounds
are converted to arbitrary-precision eigenvalue bounds; the default is
`1e-10`.

## Validation matrix

The focused tests in `test/runtests.jl` cover:

| Case | Evidence | Result |
| --- | --- | --- |
| Full-window reference equality | Three positive modes, exact masses and zero-based indices | Pass |
| Lower-threshold compatibility | Existing hybrid API comparison | Pass |
| Interior window | Single exact interior mode | Pass |
| Exact upper boundary | Boundary mode retained inclusively | Pass |
| Boundary-margin sensitivity | Margins `0` and `1e-8`, plus negative-input rejection | Pass |
| Permutation invariance | Instanton-column permutation preserves masses and self-couplings | Pass |
| Nearly degenerate hierarchy | Three entries cluster into one block; no sequential split within it | Pass |
| Strongly mixed charges | Both proposed splits fail the coupling screen; window falls back | Pass |
| Empty/reversed windows | Empty result with certified diagnostics | Pass |
| Extreme log-scale span | Million-decade input produces finite masses/eigenvectors | Pass |
| Existing feature regression | Full original `faxiverse-spectrum` tests | Pass |

The full Julia package suite passes with the feature-tip environment resolved to
the historical feature `Project.toml`. The existing suite still emits its
pre-existing duplicate-helper warnings from the vacua pipeline test includes;
those are outside SCI-06 and remain CODE-05 scope.

## Bounded performance evidence

Run:

```sh
julia --project=. scripts/benchmark_spectrum_windows.jl
```

The deterministic benchmark uses `h11=100`, an identity sparse charge basis,
five requested modes, precision 100, `confirm=false`, and `quartics=false` for
the window call. On the validation host (Julia 1.12.6, Apple arm64), one
post-warmup measurement produced:

| Measurement | Full reference | Window |
| --- | ---: | ---: |
| Wall time (s) | 1.232 | 0.624 |
| Allocated bytes | 1,792,010,064 | 750,916,064 |
| Returned modes | 100 | 5 |
| Fallback | n/a | false |

This is approximately a 2.0x wall-time reduction and 58% fewer allocated bytes
for this bounded mass-only case. It is not a production high-`h11` scaling
claim: the benchmark is synthetic, the reference API computes quartics, and
the window path still constructs dense Hessian/inertia workspaces. A production
pilot must measure real sparse CY geometries and report fallback frequency.

## Scientific decision

The hierarchy metadata and two-sided window API preserve the tested spectrum
conventions, expose conservative coupling/error diagnostics, and provide a
measurable bounded improvement. The feature is accepted as a validated,
diagnostic-capable spectrum implementation for CODE-03 review.

The following claims remain unauthorized:

- uniform or representative KS-geometry coverage;
- a population-level spectrum distribution;
- a theorem-level perturbative error bound from scale separation alone;
- production high-`h11` performance or a broad black-hole-superradiance result;
- any inflationary or Bayesian conclusion.

CODE-03 must rebase this tip onto current `vmm`, preserve existing constructors
and readers, and separately review persistence of the new diagnostics. Changes
to the split thresholds, boundary policy, mass normalization, or eigenvalue
sign semantics require a new SCI-06 decision.
