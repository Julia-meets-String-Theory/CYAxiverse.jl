# Inflation reproduction validation

Date: 2026-08-07

The reproduction path uses raw angular coordinates `theta`, canonical
coordinates `chi = G^(1/2) theta`, and a physical tangent normalized with the
raw-coordinate metric. The N=8 mass basis solves

```text
H_theta v = m^2 G v
```

once at the supplied hilltop. The eigenvectors are not recomputed along a
trajectory. The physical potential uses the ten retained trajectory rows,
zero phases, and `V = sum(A_i * (1 - cos(Q_i * theta)) )`. Slow-roll entry and
exit use `max(epsilon, abs(eta_parallel)) < 1`; a trajectory that reaches the
solver horizon while still inflating is reported as `end_event=:tmax` and
`terminated=false`.

## N=8 discrepancy

The two reported N=8 quantities have different scopes:

| quantity | delta | result | scope |
| --- | ---: | ---: | --- |
| `n8_hilltop_normal_form_efolds` | `1e-7` | `464970.4928` | one-dimensional local normal form |
| physical gradient flow | `1e-7` | `464281.2900` | full ten-row nonlinear flow; 64-bit `BigFloat` |
| `n8_hilltop_normal_form_efolds` | `1.5320548620798324e-3` | `63.85738654` | one-dimensional local normal form |
| physical gradient flow | `1.5320548620798324e-3` | `60.00336` | full ten-row nonlinear flow; tight-step convergence |

The independent SciPy value `59.690642055250756` in
`author_poly102_reference_miniforge.json` used Float64 BDF with `max_step=100`
and dense-output sampling through the stiff initial transient. Smaller step
caps move that calculation to approximately `60.002--60.003`, matching the
Julia physical flow. It is retained as provenance, not as a high-precision
physical reference. The remaining normal-form/full-flow difference is a model
scope difference caused by resolving the heavy-mode transient.

## N=5 benchmark scope

The N=5 e-fold anchors are for the explicit one-dimensional reduced potential

```text
V/V0 = 1 - cos(theta) + a(k) * (1 - cos(2theta)),
a(k_c) = 1/4,  k_c = 0.674506370003365.
```

The resulting normal-form values are:

| delta | e-folds |
| ---: | ---: |
| `1e-7` | `27349.0` |
| `6.65e-5` | `60.00000000000001` |

The package also preserves the reconstructed eight-term N=5 charge data,
geometry, metric, and K-normalized light direction. No independent author
eight-field physical N=5 trajectory is present, so these e-fold numbers must
not be described as a full eight-term trajectory reproduction.

## Bounded scalability pilot

Command:

```sh
julia --project=. --startup-file=no \
  scripts/benchmark_inflation_scalability.jl 1 10
julia --project=. --startup-file=no \
  scripts/benchmark_inflation_scalability.jl 10 10
```

The pilot was run with Julia 1.12.6, 64-bit `BigFloat`, `reltol=1e-8`,
`abstol=1e-10`, `max_step=1`, `initial_step=1e-5`, and `max_time=10`. Warmup
and compilation were excluded from the per-run timings.

| count | completed | failures | mean wall/run | allocated/run | result size | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | `0.149429 s` | `180,988,048 B` | `15,936 B` | `2,153,922,560 B` |
| 10 | 10 | 0 | `0.148687 s` | `180,988,048 B` | `15,936 B` | `2,205,663,232 B` |

The one-row peak-RSS value was `2,153,922,560` bytes in the recorded run;
the ten-row run reported `2,205,663,232` bytes. The apparent per-call
allocation is dominated by arbitrary-precision solver work, not serialized
geometry data. The output object is small because solver states are not
retained in the returned result.

This is a warm-process repetition benchmark for one hard-coded poly-102
fixture, not a representative geometry corpus. The current path has no
geometry/HDF5 loading benchmark, streaming writer, worker isolation,
checkpoint/resume logic, timeout/retry policy, or scan result sharding. No
O(10^5) scalability claim is justified; a production scan needs those pieces
and a separate pilot over real geometries before sizing the run.
