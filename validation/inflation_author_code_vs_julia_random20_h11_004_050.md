# Actual author-code comparison: 20 random geometries, h11=4..50

Date: 2026-08-09

## Scope and selection

This is a deterministic 20-geometry pilot. The sampler draws `h11`
uniformly from `4:50`, then draws a geometry uniformly from that h11 group,
rejecting duplicate `(h11, polytope, FRST)` identities. The RNG is Julia's
`MersenneTwister` with seed `20260820`.

For each geometry and each `k` in `0.9, 1.0, 1.1`, the package exports both
physical normalization choices:

- `:fixed`: `CY_vol` held fixed, matching the archived author path;
- `:full`: `CY_vol -> k^(3/2) CY_vol`.

The Python bridge imports and calls the archived
`cytools_catastrophe_scan.geometric_quantities` and
`Camcode_full_2.dim_reductor` routines. CYTools, SciPy, solver, and HSNF
modules are dependency shims only; the archived coefficient and reduction
functions are executed unchanged. The HSNF shim was exercised by the
`(22,492,1)` case and independently checked on random integer matrices for
`L*A*R == D`.

## Aggregate result

All 20 geometries completed. Therefore there were 60 `:fixed` versus author
comparisons and 60 `:full - :fixed` volume-law checks.

- `:fixed`: 60/60 key matches, 60/60 zero count mismatches, 60/60 zero sign
  mismatches.
- Largest finite `:fixed` log10 residual: `6.548e-11` dex, at `(48,29,1)`.
- `:full - :fixed`: 60/60 zero sign mismatches.
- Largest residual from `-3 log10(k)`: `9.988e-12` dex, at `(48,29,1)`.
- At `k=1`, `:full` and `:fixed` coincide, so both match the author path.

The larger residual at h11=48 is still Float64-level numerical agreement for
the high-dimensional reduction; it is not a structural mismatch.

## Selected geometries and residuals

Residuals below are maxima over `k=0.9,1.0,1.1` for that geometry.

| geometry `(h11,polytope,FRST)` | max `:fixed` vs author error (dex) | max full-minus-fixed residual (dex) |
|---|---:|---:|
| `(22,912,1)` | `4.547e-13` | `2.105e-13` |
| `(23,573,1)` | `2.046e-12` | `4.378e-13` |
| `(9,804,1)` | `2.274e-13` | `9.678e-14` |
| `(14,319,1)` | `4.547e-13` | `2.105e-13` |
| `(40,67,1)` | `1.819e-12` | `3.988e-13` |
| `(18,697,1)` | `2.728e-12` | `4.378e-13` |
| `(23,18,1)` | `3.183e-12` | `8.926e-13` |
| `(14,377,1)` | `4.547e-13` | `4.378e-13` |
| `(28,784,1)` | `1.819e-12` | `4.378e-13` |
| `(22,492,1)` | `3.411e-13` | `9.678e-14` |
| `(24,494,1)` | `2.728e-12` | `3.988e-13` |
| `(48,29,1)` | `6.548e-11` | `9.988e-12` |
| `(35,1,1)` | `2.984e-12` | `9.264e-13` |
| `(38,71,1)` | `6.366e-12` | `9.264e-13` |
| `(22,821,1)` | `3.411e-13` | `9.678e-14` |
| `(12,761,1)` | `1.137e-13` | `9.678e-14` |
| `(11,971,1)` | `2.274e-13` | `9.678e-14` |
| `(29,394,1)` | `4.093e-12` | `8.926e-13` |
| `(28,384,1)` | `4.547e-12` | `1.763e-12` |
| `(27,282,1)` | `1.137e-12` | `4.378e-13` |

The corrected raw JSON result and selection CSV were written to
`/private/tmp/cyaxiverse-author-bridge-random20-h11-004-050/` as
`author_results_corrected.json` and `selection.csv`.

Replay from the package root:

```sh
CYAXIVERSE_DATA_DIR=/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data \
CYAXIVERSE_AUTHOR_BRIDGE_OUTPUT=/private/tmp/cyaxiverse-author-bridge-random20-h11-004-050 \
julia --project=. validation/inflation_author_code_vs_julia_random20.jl
```
