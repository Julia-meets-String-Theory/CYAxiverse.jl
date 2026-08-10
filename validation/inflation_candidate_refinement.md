# SCI-02: corrected generic inflation candidates and precision boundary

Date: 2026-08-10
Program: KS axiverse inflation only
Branch: `science/inflation-candidate-refinement`
Integrated base: `vmm` at `d802686`
Status: **complete** for the bounded SCI-02 numerical boundary
Scale status: reference-potential correction only; no scale continuation or
scientific trajectory study was performed.

## Scope

This first SCI-02 slice promotes the generic numerical boundary needed after
Float64 branch selection:

- periodic integer/string-basis `Q`, `L`, and `K` inputs are dimension-checked and the kinetic
  matrix is symmetrized and required to be positive definite;
- a reusable log-shifted potential context evaluates the value, gradient, and
  Hessian at either `Float64` or requested `BigFloat` precision;
- a bounded damped Newton corrector labels a point `:converged` only when its
  coordinate-gradient infinity norm meets the declared tolerance;
- generalized-Hessian inertia, gradient norm, epsilon, and eta diagnostics are
  recomputed after correction and labelled in the physical mass eigenbasis;
- retained points are re-corrected and compared at Float64 and arbitrary
  precision, with residual and positive/zero/negative inertia agreement
  recorded;
- precision-approved hilltops can be passed through a bounded flow scan that
  tests every negative physical mass mode on both displacement sides and
  records slow-roll e-fold acceptance;
- the driver preserves complete, deterministic low-index, stochastic, and
  resource-capped search/refinement status in each CSV row.

The package boundary is in `src/inflation_points.jl`; selection thresholds,
branch caps, stochastic starts, torus-distance deduplication, CSV output, and
experiment orchestration remain in
`scripts/inflation_candidate_refinement.jl`.

The generic package boundary now includes a bounded gradient-flow diagnostic,
but the model-specific N=8 trajectory code remains outside this boundary.

## Basis contract

SCI-02 uses the periodic integer/string basis as its working representation:
`Q`, `L`, branch seeds, derivative evaluation, and damped Newton correction
remain there because stationary points and the scalar potential are unchanged
by a basis description. This is also the lower-allocation choice for the
large-`h11` path: no dense charge rotation is introduced into the hot branch
loop.

When a physical mode interpretation is needed, SCI-02 uses the mass
eigenbasis, defined by the generalized Hessian problem

```text
H_theta v = m^2 K v,       v_i^T K v_j = delta_ij.
```

The implementation uses a Cholesky reduction internally to solve this
problem; that is a numerical eigensolver representation, not a claim that a
kinetic-normalized coordinate basis is itself physical. The default scalar
diagnostics retain only the mass eigenvalues, inertia, epsilon, and eta
values. Raw-coordinate, `K`-orthonormal mass eigenvectors are available via
`CYAxiverse.inflation_points.mass_eigenbasis(...; vectors=true)` only when
mode directions or couplings require them, since materializing them costs
`O(h11^2)` storage.

## Generic geometry call

The low-level geometry entry point loads one `GeometryIndex` through
`read.oriented_potential`, preserving the periodic/string-basis provenance:

```julia
using CYAxiverse

geom = CYAxiverse.structs.GeometryIndex(h11, polytope, frst)
prepared = CYAxiverse.inflation_points.prepare_geometry_context(geom)
result = CYAxiverse.inflation_points.gradient_flow(
    geom, hilltop;
    displacement=1e-8,
    mode=:most_negative,
    max_efolds=60,
    step=1e-3,
)
```

`hilltop` is deliberately caller-supplied; this call does not infer a
stationary point or a candidate population. Branch seeds and derivatives stay
in the periodic/string basis. The most-negative physical mass mode supplies
the initial displacement and mode diagnostics, while the flow itself uses a
canonical Cholesky chart only as a lower-allocation numerical integration
chart. E-folds are the independent variable, so the arbitrary log-potential
shift does not affect the flow equation.

The returned named tuple records `geometry`, `input_basis`, `source`,
`basis`, `coordinate_chart`, selected mode information, slow-roll windows,
exit status, and the requested horizon. The fixed-step RK4 implementation is
bounded and intended for candidate-level comparison; it does not establish a
physical time normalization, stabilization, Kähler-cone/EFT validity, or a
large-`h11` production scan. Dense mass-mode vectors remain opt-in, and the
full dense Hessian currently makes this a validation path rather than the
final large-`h11` throughput kernel.

The candidate-discovery path is now available in
`scripts/inflation_candidate_refinement.jl`:

```julia
scan = scan_geometry_for_inflation(
    geom;
    precision_bits=128,
    min_efolds=50,
    max_efolds=60,
    flow_step=1e-3,
)
```

It enumerates the certified finite branch set, precision-checks each corrected
point, and flows only `:refined_candidate` points. For every such point it
computes the physical eigensystem once, tests all negative mass modes with both
displacement signs, and returns `scan.flow.rows`. A flow is accepted when its
slow-roll window reaches `min_efolds`; `:max_efolds` is recorded as a lower
bound unless `require_finite_exit=true` is requested. The command-line driver
accepts repeated `--geometry` arguments and writes the refinement and flow
records separately with `--output` and `--flow-output`. These records carry
`scale_status=:unknown`: no physical divisor-volume continuation is evaluated
by this call.

## Bounded comparison matrix

The temporary CSV outputs were written under
`/private/tmp/inflation-candidate-refinement/` with `precision_bits=128`,
`float_tolerance=1e-9`, `high_tolerance=1e-18`, no stochastic starts, and
`max_points=1000`. The exact branch estimates were computed before execution.

| geometry | search | exact branches | corrected | high-precision corrected | inertia agreement | screen/refined candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(8,1,1)` weak | complete | 768 | 768 | 768 | 768 | 0 / 0 |
| `(8,544,1)` strong | complete | 768 | 768 | 768 | 768 | 0 / 0 |
| `(8,1,1)` weak | `1:1` | 24 | 24 | 24 | 24 | 0 / 0 |
| `(8,544,1)` strong | `1:1` | 24 | 24 | 24 | 24 | 0 / 0 |
| `(8,1,1)` weak | `1:2` | 108 | 108 | 108 | 108 | 0 / 0 |
| `(8,544,1)` strong | `1:2` | 108 | 108 | 108 | 108 | 0 / 0 |

Total: 1,800 exact branches; all 1,800 Float64 corrections converged, all
1,800 arbitrary-precision corrections converged, and all 1,800 residual and
inertia comparisons agreed. No point passed the existing screen
(`value > 0`, at least one negative mode, `epsilon < 1`, and
`abs(min_eta) < 1`).

The separate stochastic probe used `(8,1,1)`, `1:1`, two starts, seed
`20260810`, and `max_points=100`. Its 24 deterministic branches all agreed;
both stochastic starts returned explicit `:singular_hessian` correction and
high-precision statuses. They were not labelled stationary or candidates.

## Reproduction commands

The comparison matrix was run with:

```sh
julia --project=. --startup-file=no \
  -e 'include("scripts/inflation_candidate_refinement.jl"); ...'
```

where the six calls used the local data root, geometries `(8,1,1)` and
`(8,544,1)`, searches complete/`1:1`/`1:2`, `max_branches=100000`,
`precision_bits=128`, `float_tolerance=1e-9`, `high_tolerance=1e-18`, and
`max_points=1000`. The CLI also supports an explicit `--data-dir`; every row
records the resolved data root and the Git commit.

Package validation on this branch:

- the full `Pkg.test()` suite passed;
- the generic correction/precision testset passed 25/25;
- the geometry-level mass-basis gradient-flow testset passed 20/20;
- a bounded two-geometry CLI run completed for `(8,1,1)` and `(8,544,1)`;
- that pilot produced no precision-approved candidates, so its flow report
  contained only the schema header.
- a bounded `(20,143,1)` `1:1` run visited all 20 estimated branches and
  produced 20 point rows, but no precision-approved candidates or flow rows.
- bounded `1:1` runs for `(22,492,1)` and `(22,912,1)` each visited all 44
  estimated branches and likewise produced no precision-approved candidates
  or flow rows.
- `git diff --check` passed.

## Interpretation and limitations

These are corrected finite-search results on two `h11=8` reference geometries.
The complete rows establish only the enumerated 768-branch coverage for those
inputs. The low-index rows are deterministic subsets, and the stochastic probe
has an explicit start budget. The zero candidate count is therefore a bounded
null result; candidate recall remains unmeasured because the reference rows
contain no known positive generic candidates.

The calculation corrects the supplied reference potential. It does not vary
the SCI-01 physical divisor-volume path, certify Kähler-cone or EFT validity,
or establish a stabilized compactification. A flow row is therefore a bounded
reference-potential trajectory diagnostic, not a physical trajectory claim.
No production scan or `h11=491` run was launched.

Follow-on work should validate the same boundary on the remaining approved
lower-`h11` strong/weak rows, retain all explicit correction failures, and
define the resource policy for larger rows before any population-level
interpretation. Those extensions are not required for completion of this
bounded SCI-02 deliverable.
