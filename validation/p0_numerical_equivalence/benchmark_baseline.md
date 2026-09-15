# P0 B2–B7 benchmark and baseline evidence

Status: complete for the bounded Worker-D scope.  This artifact records the
current numerical behavior at the reviewed source reference.  It does not
authorize or implement P1/P2 changes, and it makes no high-`h11` or population
claim.

## Identity and replay boundary

- Numerical reference source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`.
- Execution revision: `99d702c8b28e674d34e29566f8ecc483d76c7f7f` on
  `research/p0-numerical-equivalence-20260915`.  The execution revision adds
  P0 contract documents on top of the numerical reference; production source
  and `Project.toml` were unchanged relative to the reference at execution.
- Julia: 1.12.6, build `15346901f0039751c5488744f1f62de7d87510a8`,
  `arm64-apple-darwin24.0.0`.
- Host: Apple M4 Pro, Darwin/aarch64; BLAS vendor `:lbt`, BLAS threads 8;
  `JULIA_NUM_THREADS=1`; `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OMP_NUM_THREADS` unset.  Dependencies were precompiled in the temporary
  resolved environment used for the run.
- Resolved environment identity: Manifest SHA-256
  `67e67bf7edb48d3b9a750b31b2639b774ba8d4a0f6ba4f509e28172033712828`.
  The full UUID/name/version inventory is in
  [`b2_b7_metadata.txt`](benchmark_results/b2_b7_metadata.txt).
- A privacy-safe normalized copy is retained in `environment_benchmarks/`.
  Its only Manifest change replaces the machine-local CYAxiverse path with
  repository-relative `../../..`; its Project and Manifest SHA-256 values are
  `f42fcc77faac53ac2ce271605b59ef3e677c018e7510dfe306f04ebb5fa95cec`
  and `5370a09f21ad09488d6cf81aab67bb648e016d5fbb850881786ab145609c8b84`.
- The root checkout has no package Manifest.  A temporary resolved environment
  was required because package/depot writes are restricted in the execution
  environment; registry download was unavailable, but all required package
  sources and artifacts were present in the local depot caches.

The exact successful replay command was:

```text
JULIA_DEPOT_PATH=<writable-temporary-depot>:<existing-local-depot> \
  JULIA_NUM_THREADS=1 julia \
  --project=validation/p0_numerical_equivalence/environment_benchmarks \
  --startup-file=no \
  validation/p0_numerical_equivalence/scripts/benchmark_baseline.jl
```

The script itself records the source SHA, execution revision, environment
identity, fixture digests, and the benchmark protocol.  It writes:

- [`b2_b7.tsv`](benchmark_results/b2_b7.tsv), SHA-256
  `e6b276317e9016325cb2b8898122aa37ff47d025ec8056337833adab18e7aaf7`;
- [`b2_b7_metadata.txt`](benchmark_results/b2_b7_metadata.txt), SHA-256
  `1b79b1a01b95512d95b05d964d9abd41c1a2100831f57f601c45cfa440420c31`.

## Measurement protocol

For each timed route, the harness discards one warm-up call, runs the declared
sample count, calls `GC.gc()` before each sample, and reports median, minimum,
and maximum wall time.  `@timed.bytes` is the primary allocation statistic;
one post-warm-up `@allocated` call is retained as a cross-check.  The default
sample count is 5; route-specific counts are shown below.  Results are
single-process, single Julia thread, and are not a concurrency claim.

Potential inputs use the package's stored signed/log10 encoding.  The dense
`generate.V/jacobian/hessian` routes use the historical `Q :: h11 × N` and
`L :: N × (sign, log10 scale)` layout.  The fixed paper benchmark derivative
route is reported separately because it uses the paper's `2π Q⋅θ` convention.

## B2 — dense physical derivatives

Fixture: deterministic Appendix-C N=8 table potential at `k=1.0`, eight
coordinates, digest `0a6aa22b93f8c8676936beb78698e979adabbba1be8edda872da1ee0b8b1ce8d`.
Values are normalized only by the historical stored encoding; tiny values are
therefore expected.

| Route | Samples | Median seconds | Median bytes | Observed result |
|---|---:|---:|---:|---|
| `generate.V` | 5 | 5.0e-6 | 800 | value `4.304396002053623e-40` |
| `generate.jacobian` | 5 | 1.725e-5 | 3056 | 8-vector; first `8.592527067086707e-39` |
| value + gradient | 5 | 1.9458e-5 | 3856 | value above; 8-vector gradient |
| gradient + Hessian | 5 | 2.3375e-5 | 23792 | 8-vector + 8×8 Hessian |
| value + gradient + Hessian | 5 | 2.8042e-5 | 24592 | scalar + 8-vector + 8×8 Hessian |
| `paper_benchmarks.n8_potential_derivatives` | 5 | 1.4916e-5 | 8496 | paper `2π` route; value `1.6483060377866048e-38` |

## B3 — current structured evaluator

Fixtures are deterministic `pseudo_Q` base-plus-pairwise charge sets with
signed log scales `L[2,j] = -0.4(j-1)`.  Their digests are:

- h11=4, base count 8, 36 instantons:
  `3f9afcb36d54b69875846169455b6f6afb6473d55f9f088b8e9f24d575e0a330`;
- h11=8, base count 12, 78 instantons:
  `736c8ed1a1b8805a314406470bd8119eb989f0816ce956e37b29d3393b539341`.

The structure proof reports `validated=true`, `fallback=false` in both cases.
Structured and generic log-shifted derivative values/gradients/Hessians agree
within the predeclared Float64 `rtol=atol=1e-13` envelope (parity `true` for
both h11 values).  This tolerance allows the distinct summation orders; it is
not a P2 acceptance tolerance.

| Route | Fixture | Samples | Median seconds | Median bytes | Observed result |
|---|---|---:|---:|---:|---|
| structure proof/preparation | h11=4, N=36 | 3 | 1.1834e-5 | 16880 | validated; base count 8 |
| repeated structured evaluation | h11=4, N=36 | 5 | 8.375e-6 | 48 | value `0.013370758697706065`; 4-gradient/4×4 Hessian |
| repeated generic evaluation | h11=4, N=36 | 5 | 2.917e-6 | 48 | same value within envelope |
| structure proof/preparation | h11=8, N=78 | 3 | 1.6667e-5 | 30672 | validated; base count 12 |
| repeated structured evaluation | h11=8, N=78 | 5 | 9.584e-6 | 48 | value `0.057074324357425564`; 8-gradient/8×8 Hessian |
| repeated generic evaluation | h11=8, N=78 | 5 | 4.875e-6 | 48 | same value within envelope |

The structured route is evidence of the current implementation and parity only;
it is not an optimization result or a claim that structure is valid for an
unproven geometry population.

## B4 — `critical_points`

Fixture digest `2d5b66eda4b8a8b29dc119d237a6583adfffac4741cadb33828f4f966f5f8c4d`:
two axions, three signed instantons, `starts=24`, residual tolerance
`1e-10`, merge tolerance `1e-7`, and maximum 100 Newton iterations.

| Samples | Median seconds | Median bytes | Starts | Critical roots | Minima | Residuals | Inertia |
|---:|---:|---:|---:|---:|---:|---|---|
| 3 | 0.000128916 | 159504 | 24 | 4 | 1 | `[0.0, 1.256122746310878e-15, 5.261410430776349e-16, 6.159391718249908e-16]` | `[(0,0,2),(2,0,0),(1,0,1),(1,0,1)]` |

The current API reports converged roots and residuals but does not expose
counts of discarded failed starts.  Failure-state accounting is therefore
unavailable at this route boundary and is not inferred from root count.

## B5a — inflation flow

The bounded N=8 e-fold flow uses `delta_k=1e-3`, displacement `1e-6`,
`max_efolds=0.25`, `max_step=0.1`, `initial_step=1e-3`, and four saved samples.
Fixture digest: `06f83cef908810195a6cf410a4f11f6acff49e09b7ef602dfc37fcec26bcf5c7`.

| Route | Samples | Median seconds | Median bytes | Steps | Slow-roll result | Exit event |
|---|---:|---:|---:|---:|---|---|
| `n8_efold_gradient_flow` (bounded RK4) | 2 | 0.004579229000000001 | 20273808 | 250 | `entered=false`, `efolds=0.0` | `tmax` |
| `n8_hilltop_probe` (local normal form diagnostic) | 3 | 0.000168 | 250304 | 4 | `entered=true`, `efolds=46.49704928459734` | `local_normal_form` |

The current flow result exposes accepted step count and event, but not internal
RHS or Hessian evaluation counters; those counters are unavailable, not
estimated.  The local normal-form diagnostic is a distinct scientific route
and must not be conflated with the full nonlinear flow.

## B5b — stationary correction

Fixture digest `4edc707dced4714e2201eac62e869001ba88a6d2ca4d88697c1561c8e6faf076`:
Appendix-B N=5 potential/kinetic fixture, seed
`[0.02,-0.01,0.03,-0.02,0.01]`, periodic/string working basis.

| Route | Samples | Median seconds | Median bytes | Status | Iterations | Residual | Precision agreement |
|---|---:|---:|---:|---|---:|---:|---|
| Float64 correction | 3 | 1.8708e-5 | 13728 | converged | 3 | `4.853797830701449e-69` | — |
| Float64 + BigFloat-128 replay | 1 | 0.000352209 | 313256 | both converged | Float 3 / high 3 | high `4.85379783070144820208939166754302125091e-69` | accepted; residual and inertia agreement true |

The current correction API exposes iterations, terminal status, residual, and
elapsed seconds.  Hessian-evaluation and line-search-trial counts are not
exposed and are unavailable.  BigFloat behavior was measured serially; this is
not evidence of safe concurrent global-precision mutation.

## B6 — representative spectra

Fixture digest `ac7f529c38af00b5abc1d28cae5e3271571e524e74ed450ff043ae22cc0217e6`:
Appendix-B N=5 potential and kinetic matrix at `k=1.0`.

| Route | Samples | Median seconds | Median bytes | Mass logs | Signs | Component counts |
|---|---:|---:|---:|---|---|---|
| `pq_spectrum`, Float64 mass correction | 2 | 0.000418458 | 102048 | `[13.8609125391,14.0735852771,14.2304345900,21.9489597034,22.3524826756]` | `[-1,1,1,1,1]` | self 5; λ31 20; λ22 10 |
| `pq_spectrum`, high precision 80 | 1 | 0.000657334 | 501280 | `[-12.9185584434,-4.5978009625,-1.9630253110,21.9489597034,22.3524826756]` | `[1,1,1,1,1]` | self 5; λ31 20; λ22 10 |

The light-mode Float64 and high-precision outputs differ materially in this
current route.  This is a baseline observation requiring numerical-contract
review; no production repair or physical interpretation is made here.
The Float64 mass-basis diagnostics were also recorded: eigenpair residuals
`[0.0227315659,0.2592611110,0.1535304731,1.8039858746e-16,2.2070838395e-16]`,
nearest relative gaps `[0.0381365736,0.0268989685,0.0268989685,0.8440612475,0.8440612475]`,
and orthogonality error `3.9085625896e-16`.

## B7 — geometry/filesystem behavior

No checked-out geometry database was assumed.  A synthetic HDF5 fixture was
created under the canonical `h11_002/np_0000001/cy_0000001/cyax.h5` layout,
with Q/L/Kinv plus minimal geometric enrichment fields.  Fixture digest:
`335489dfcc0f5a617906008aaea627761db1d9743e826a51005f47bceba253b8`.

| Route | Samples | Median seconds | Median bytes | Observed result |
|---|---:|---:|---:|---|
| filesystem scan `np_path_generate(2)` | 3 | 0.00048125 | 26592 | one path; index shape `(3,1)` |
| HDF5 query `oriented_potential` | 3 | 0.000413583 | 15776 | Q `(2,3)`, L `(2,3)`, K `(2,2)`, eigenvalues `[1.0,1.0]` |
| HDF5 geometry enrichment `geometry` | 3 | 0.000514333 | 16272 | h21 1; CY volume 1.0; GLSM `(2,2)`; divisor volumes `[1.0,1.0]` |
| HDF5-backed `pq_spectrum` | 2 | 0.0005198959999999999 | 39888 | mass logs `[28.0846788339,28.1846788339]`; signs `[1,1]` |

This is a filesystem/schema behavior probe, not a geometry catalogue or
population result.

## Claim boundary and unavailable fields

Source/implementation facts are separated from empirical timings above.  The
timings are machine-, Julia-build-, dependency-, thread-, and fixture-specific
empirical evidence.  B1 package-load measurements are out of this worker's
scope and remain owned by Worker A.  B4 discarded-start failure states, B5a
RHS/Hessian counters, and B5b Hessian/line-search counters are unavailable from
the current APIs.  B7 uses only the bounded synthetic fixture.  No production,
test, dependency, package, or scientific-schema file was modified.
