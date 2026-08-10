# Author-versus-Julia coefficient check: random (h^{1,1}=10)

Date: 2026-08-09

## Scope

This is a bounded coefficient-level check, not a physical inflation result. It
compares the draft-author coefficient formula with one stored low-dimensional
geometry and with the current generic Julia continuation.

The replayable command is:

```sh
julia --project=. validation/inflation_author_vs_julia_h11_010.jl
```

The script uses seed `20260809` to select one file from the 1,000 available
`h11=10` geometries. The selected geometry was:

```text
h11_010/np_0000401/cy_0000001/cyax.h5
```

It has 14 leading effective-cone charge rows and 105 total stored terms.

## Results

| comparison | max finite log-coefficient error | sign mismatches |
|---|---:|---:|
| author formula vs stored (L), (k=1) | (1.71\times10^{-13}) | 0 |
| author fixed-volume path vs current homotopy, (k=0.9) | (4.00\times10^{-1}) | 0 |
| author full-volume path vs current homotopy, (k=0.9) | (2.63\times10^{-1}) | 0 |
| author fixed-volume path vs current homotopy, (k=1.1) | (3.96\times10^{-1}) | 0 |
| author full-volume path vs current homotopy, (k=1.1) | (2.72\times10^{-1}) | 0 |

## Interpretation

The base implementation agrees: the author formula reproduces the stored
potential at (k=1) to Float64-level accuracy. The scaled paths do not agree
because the current generic pilot performs only

```text
L[2, :] <- k * L[2, :]
```

whereas the author path jointly applies

```text
tau  <- k * tau
Kinv <- k^2 * Kinv
V_CY <- V_CY              (:fixed)
V_CY <- k^(3/2) * V_CY    (:full)
```

and recomputes all coefficient prefactors. Therefore the current generic
pilot cannot be said to reproduce the author’s physical scale algorithm away
from (k=1); it remains `scale_status=homotopy_only`.

The check reconstructs the author formula directly in Julia from the stored
geometry fields. The original Python/CYTools runtime was not required for
this coefficient comparison, and no claim is made here about catastrophe
finding, trajectories, or population prevalence.
