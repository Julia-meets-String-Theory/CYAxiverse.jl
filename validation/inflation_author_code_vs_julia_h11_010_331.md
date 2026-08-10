# Actual author-code comparison: h11=10 geometry `(10,331,1)`

Date: 2026-08-09

## Scope

This is a second deterministic h11=10 check, selected from the 1,000 local
`h11_010` geometries with Python RNG seed `20260810`. It uses the package's
physical `fixed` and `full` routines and imports/calls the archived author
implementations:

- `cytools_catastrophe_scan.geometric_quantities`
- `Camcode_full_2.dim_reductor`

The replay command is:

```sh
CYAXIVERSE_DATA_DIR=/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data \
julia --project=. validation/inflation_author_code_vs_julia_h11_010_331.jl
```

The local machine does not have CYTools/SciPy installed. The bridge therefore
stubs only import-only CYTools and solver modules; the archived coefficient and
reduction routines themselves execute unchanged against the stored geometry
arrays. Charge keys are compared modulo `q ~ -q`, since the author sorts the
leading charges before forming differences and the cosine potential is even.

## Result

The package `fixed` routine now matches the active author reducer at this
geometry to Float64 precision. The `full - fixed` relation also passes
precisely:

| scale | package fixed vs actual author: max finite log10 error | full-fixed expected shift | measured shift error | sign mismatches |
|---:|---:|---:|---:|---:|
| 0.9 | `2.27e-13` | `+0.1372724717` | `1.71e-13` | 0 |
| 1.0 | `4.55e-13` | `0` | `0` | 0 |
| 1.1 | `2.27e-13` | `-0.1241780555` | `2.10e-13` | 0 |

Thus `:fixed` reproduces the actual author code and `:full` differs only by
the verified homogeneous volume factor.

## Source discrepancy

The active author code in `Camcode_full_2.py:138` evaluates the cross term as

```text
(8*pi^2/V^2) * [q_i^T Kinv q_j + (q_i+q_j).tau]
```

The package path now uses the same expression in the continuation helper and
the active generic geometry exporters (`add_functions/cytools_wrapper.jl`,
`scripts/generate_geometric_data.py`, and
`scripts/generate_geometric_data_multitriangulation.py`):

```text
(8*pi^2/V^2) * [q_i^T Kinv q_j + (q_i+q_j).tau]
```

The pre-existing HDF5 `L` values retain the older convention on this geometry;
the physical pilot now reconstructs the author potential at `k=1` before
seeding, and records the stored-reference discrepancy in the CSV provenance
fields rather than silently using the stale coefficients.
