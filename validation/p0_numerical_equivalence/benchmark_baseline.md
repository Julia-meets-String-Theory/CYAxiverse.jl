# P0 B2–B7 benchmark and baseline evidence

Status: complete for the bounded Worker-D scope.  This artifact records the
current numerical behavior at the reviewed source reference.  It does not
authorize or implement P1/P2 changes, and it makes no high-`h11` or population
claim.

## Identity and replay boundary

- Numerical reference source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`.
- Execution revision: `8dab6e6185867c62d962b44ca4e664f749df55db` on
  `research/p0-numerical-equivalence-20260915`.  The execution revision adds
  P0 contract documents on top of the numerical reference; production source
  and `Project.toml` were unchanged relative to the reference at execution.
- Julia: 1.12.6, build `15346901f0039751c5488744f1f62de7d87510a8`,
  `arm64-apple-darwin24.0.0`.
- Host: Apple M4 Pro, Darwin/aarch64; BLAS vendor `:lbt`, BLAS threads 8;
  `JULIA_NUM_THREADS=1`; `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OMP_NUM_THREADS` unset.  Dependencies were precompiled in the temporary
  resolved environment used for the run.
- Resolved environment identity: retained benchmark Manifest SHA-256
  `5370a09f21ad09488d6cf81aab67bb648e016d5fbb850881786ab145609c8b84`.
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
  JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 julia \
  --project=validation/p0_numerical_equivalence/environment_benchmarks \
  --startup-file=no \
  validation/p0_numerical_equivalence/scripts/benchmark_baseline.jl
```

The script itself records the source SHA, execution revision, environment
identity, fixture digests, and the benchmark protocol.  It writes:

- [`b2_b7.tsv`](benchmark_results/b2_b7.tsv), SHA-256
  `274ba44db46e394f53cdf47c07155048c9772891d348ecded7b0ff0ce3d7692a`;
- [`b2_b7_metadata.txt`](benchmark_results/b2_b7_metadata.txt), SHA-256
  `81541ae468698e900f25b422ae527504054b23767fa47cfbdf11a271fe53a22a`.

## Measurement protocol

For each timed route, the harness discards one warm-up call, runs the declared
sample count, calls `GC.gc()` before each sample, and reports median, minimum,
and maximum wall time.  `@timed.bytes` is the primary allocation statistic;
one post-warm-up `@allocated` call is retained as a cross-check.  The default
sample count is 5; route-specific counts are shown below.  Results are
single-process, single Julia thread, and are not a concurrency claim.  The
B5b high-precision replay uses the contract value
`high_residual_tolerance=1e-40`.

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
| `generate.V` | 5 | 4.5e-6 | 800 | value `4.304396002053623e-40` |
| `generate.jacobian` | 5 | 2.0041e-5 | 3056 | 8-vector; first `8.592527067086707e-39` |
| value + gradient | 5 | 2.1834e-5 | 3856 | value above; 8-vector gradient |
| gradient + Hessian | 5 | 2.4958e-5 | 23792 | 8-vector + 8×8 Hessian |
| value + gradient + Hessian | 5 | 2.9375e-5 | 24592 | scalar + 8-vector + 8×8 Hessian |
| `paper_benchmarks.n8_potential_derivatives` | 5 | 2.0333e-5 | 8496 | paper `2π` route; value `1.6483060377866048e-38` |

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
| structure proof/preparation | h11=4, N=36 | 3 | 1.6375e-5 | 16880 | validated; base count 8 |
| repeated structured evaluation | h11=4, N=36 | 5 | 1.0042e-5 | 48 | value `0.013370758697706065`; 4-gradient/4×4 Hessian |
| repeated generic evaluation | h11=4, N=36 | 5 | 3.209e-6 | 48 | same value within envelope |
| structure proof/preparation | h11=8, N=78 | 3 | 2.125e-5 | 30672 | validated; base count 12 |
| repeated structured evaluation | h11=8, N=78 | 5 | 1.0125e-5 | 48 | value `0.057074324357425564`; 8-gradient/8×8 Hessian |
| repeated generic evaluation | h11=8, N=78 | 5 | 4.959e-6 | 48 | same value within envelope |

The structured route is evidence of the current implementation and parity only;
it is not an optimization result or a claim that structure is valid for an
unproven geometry population.

## B4 — `critical_points`

Fixture digest `50f29554921e16473fd3cce03d24330e9bf15142a2655fd03a45a6822e3da86d`:
two axions, three signed instantons, `starts=24`, residual tolerance
`1e-10`, merge tolerance `1e-7`, and maximum 100 Newton iterations.  The
deterministic start set is the zero point followed by the 23 Halton points
with bases 2 and 3, in the following column order:

```text
[[0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625, 0.3125, 0.8125, 0.1875, 0.6875, 0.4375, 0.9375, 0.03125, 0.53125, 0.28125, 0.78125, 0.15625, 0.65625, 0.40625, 0.90625],
 [0.0, 0.3333333333333333, 0.6666666666666666, 0.1111111111111111, 0.4444444444444444, 0.7777777777777777, 0.2222222222222222, 0.5555555555555556, 0.8888888888888888, 0.037037037037037035, 0.37037037037037035, 0.7037037037037037, 0.14814814814814814, 0.48148148148148145, 0.8148148148148147, 0.25925925925925924, 0.5925925925925926, 0.9259259259259258, 0.07407407407407407, 0.4074074074074074, 0.7407407407407407, 0.18518518518518517, 0.5185185185185185, 0.8518518518518517]]
```

| Samples | Median seconds | Median bytes | Starts | Critical roots | Minima | Residuals | Inertia |
|---:|---:|---:|---:|---:|---:|---|---|
| 3 | 0.00013475 | 159504 | 24 | 4 | 1 | `[0.0, 1.256122746310878e-15, 5.261410430776349e-16, 6.159391718249908e-16]` | `[(0,0,2),(2,0,0),(1,0,1),(1,0,1)]` |

The returned root coordinates (columns, in API order) were
`[[0.0,0.5,1.0,0.5]; [0.0,0.5,0.5,9.740137739595606e-18]]`.  A second
identical replay was matched one-to-one with periodic distances
`[(left=1,right=1,distance=0.0),(left=2,right=2,distance=0.0),
(left=3,right=3,distance=0.0),(left=4,right=4,distance=0.0)]`.  The API exposes
converged roots, residuals, and inertia, but does not expose per-start
converged/discarded statuses; this status field is unavailable and is not
inferred from root count.  Replay status was identical.

## B5a — inflation flow

The bounded N=8 e-fold flow uses `delta_k=1e-3`, displacement `1e-6`,
`max_efolds=0.25`, `max_step=0.1`, `initial_step=1e-3`, and four saved samples.
Fixture digest: `3a311d6932671d867581f772372344299857f6a94f43fc56ca1c792c041f5854`.

| Route | Samples | Median seconds | Median bytes | Steps | Slow-roll result | Exit event |
|---|---:|---:|---:|---:|---|---|
| `n8_efold_gradient_flow` (bounded RK4) | 2 | 0.004716291500000001 | 20273808 | 250 | `entered=false`, `efolds=0.0` | `tmax` |
| `n8_hilltop_probe` (local normal form diagnostic) | 3 | 0.000240541 | 250304 | 4 | `entered=true`, `efolds=46.49704928459734` | `local_normal_form` |

The flow initial diagnostics were `theta=[-1.4923750075970572e-17,
1.5399581730410878,4.743227134138578,0.030838153753630058,
6.252347153425956,4.774065287892169,4.712432868804226,
-8.164119782005513e-14]`, `epsilon=0.01643103336811026`,
`eta_parallel=26645.874759895047`, and gradient norm
`3.3880222480384505e-5`.  The four saved flow samples have `n=`
`[0.0,0.001,0.002,0.003]`, epsilon
`[0.016431033366086884,1209.555220455177,10087.060081332997,24.648039785734355]`,
and eta-parallel
`[26645.874759898354,310360.5064502586,9163.351817676801,-1236.061135978546]`;
the TSV retains their complete theta, tangent, potential, `n_s`, and
`delta_H` records.  The current API does not return the final integrated state,
so final-state diagnostics are unavailable for this route.  The local
normal-form probe has saved `n=` `[0.0,15.499016428199113,30.998032856398225,
46.49704928455084]` and its endpoint sample is retained in the TSV.  It is a
distinct scientific route and must not be conflated with the full nonlinear
flow.

## B5b — stationary correction

Fixture digest `fe0989bebd9ee1f7bd5c8f020c9ca96f27d2fb97c8d0f1741c3ba414454dae30`:
Appendix-B N=5 potential/kinetic fixture, seed
`[0.02,-0.01,0.03,-0.02,0.01]`, periodic/string working basis.

| Route | Samples | Median seconds | Median bytes | Status | Iterations | Residual | Precision agreement |
|---|---:|---:|---:|---|---:|---:|---|
| Float64 correction | 3 | 2.375e-5 | 13728 | converged | 3 | `4.853797830701449e-69` | — |
| Float64 + BigFloat-128 replay | 1 | 0.000469542 | 313320 | both converged | Float 3 / high 0 | high `4.85379783070144820208939166754302125091e-69` | accepted; residual and inertia agreement true |

The Float64 final coordinate is `[0.0,0.0,0.00364021949874719,
0.007280438997494433,0.0]`; its diagnostic is value
`9.733679990838844e-74`, gradient norm `2.5351541631498092e-67`,
`epsilon=3.391756173758798e12`, eta values
`[-1.0235767073133454e62,-6.751444912815211e59,1.6843211068206007e62,
5.382814422652798e77,3.4518773147313846e78]`, and inertia `(0,3,2)`.
The BigFloat-128 replay final coordinate is
`[0.0,0.0,0.003640219498747189908610666364552344020922,
0.00728043899749443272628735002172106760554,0.0]`; its diagnostic residual
is `4.85379783070144820208939166754302125091e-69`, eta values
`[-3.261146534747821835493747769724395141888e39,
2.160294491679440758724967345699925363793e39,
6.047413739668620870477719102918490447617e39,
5.382814422649071361319690567715949075234e77,
3.45187731472899172759089514058812569457e78]`, and inertia `(0,3,2)`.
The high residual gate was exactly `1e-40`; high iterations are zero because
the Float64 solution already satisfies that gate.  The current correction API
exposes terminal status, residual, coordinate, and iterations, but not
Hessian-evaluation or line-search-trial counts.  BigFloat behavior was
measured serially; this is not evidence of safe concurrent global-precision
mutation.

## B6 — representative spectra

Fixture digest `68261b5571df88e6afe75f985eafac27a144390779306d3e4c416a6fc1caf6ef`:
Appendix-B N=5 potential and kinetic matrix at `k=1.0`.

| Route | Samples | Median seconds | Median bytes | Mass logs | Signs | Component counts |
|---|---:|---:|---:|---|---|---|
| `pq_spectrum`, Float64 mass correction | 2 | 0.000321979 | 105408 | `[13.860912539123262,14.073585277123616,14.230434590975188,21.948959703416346,22.352482675621758]` | `[-1,1,1,1,1]` | self 5; λ31 20; λ22 10 |
| `pq_spectrum`, high precision 80 | 1 | 0.000781625 | 569872 | `[-12.91855844342438,-4.5978009624626,-1.9630253110345888,21.948959703416346,22.352482675621758]` | `[1,1,1,1,1]` | self 5; λ31 20; λ22 10 |

**Material numerical disagreement:** the Float64 and high-precision mass logs
remain materially different in the three light modes.  This is a
contract-critical baseline observation requiring numerical review; it is not
reinterpreted as physics and no production repair is made here.

Both routes returned an `AxionSpectrum`; that type has no explicit status
field, so status is recorded as `returned_AxionSpectrum`.  The API does not
place eigenvectors in that return object, but the public
`leading_hessian_mass_basis_float64` and `leading_hessian_mass_basis` helpers
expose the corresponding basis.  The exact eigenvector matrices used for the
diagnostic replay are retained in the TSV; they are, respectively:

```text
Float64:
[0.39222574598110505 0.8597694790943079 0.2140280495892792 -0.006083260852806413 0.24720516769249368;
 0.6217534489546949 -0.29005123184303205 0.5609699525620624 0.011396408500575234 -0.46311528350410425;
 0.6777984298415944 -0.23814031881542608 -0.6289113004298099 -0.007312205502053051 0.29714573007387735;
 0.013072926480957743 0.3463465234107736 -0.4939385412206783 0.019617385855001883 -0.7971907299909466;
 0.0 0.0 0.0 0.9996973570204117 0.024600698494220714]

High precision 80:
[0.7962368571425973 -0.5450753470544687 0.08804733025283883 0.0060832608528064115 -0.24720516769249368;
 -0.1909342566763616 -0.3621084576507912 0.7860126783859225 -0.01139640850057523 0.46311528350410425;
 0.3195177831857669 0.6940766035108754 0.5725530555646512 0.007312205502053049 -0.29714573007387735;
 0.47692699984712306 0.3000466620211309 -0.21590445946185874 -0.01961738585500187 0.7971907299909464;
 1.4937152236141752e-70 1.418326169643478e-54 5.5489376766446104e-49 -0.9996973570204116 -0.024600698494220703]
```

The signed quartic log arrays (`λself`, `λ31`, `λ22`) are exposed and are
retained in full in the TSV.  Their exact values are:

```text
Float64 λself = [-53.454956053973945,-55.04665451413669,-56.028489331615916,-5.491414659353435,-4.541782163446751]
Float64 λ31 = [-53.852881360180554,-54.098712395265274,-53.846618991001556,-53.849435999348074,-54.64873242428423,-55.29250134987386,-55.06940272022329,-52.336804533452295,-55.38616594620724,-55.78445826059139,-55.857102799948045,-54.13057164125298,-29.071344360855008,-27.72164151319292,-28.32254936791434,-7.100344646381931,-21.82344321078392,-20.47374036312184,-21.074648217843254,-6.9579386620167885]
Float64 λ22 = [-54.25080652417646,-54.742466559619906,-55.538256237055926,-43.93703834026267,-41.2376326449602,-42.439448354403,-39.10510424023698,-36.405698544912816,-37.60751425435564,-8.62413004080992]
High precision 80 λself = [-72.28012202403208,-58.404602890433566,-53.288243585756454,-5.491414659353435,-4.541782163446751]
High precision 80 λ31 = [-71.9345040823567,-71.77187506181936,-57.76243567880104,-55.346468628777345,-71.24144110636625,-59.05819525959592,-56.72182753232511,-54.306706952615166,-70.76327120376847,-59.24297283088475,-53.7229151299447,-53.468571391750714,-28.724578813884115,-28.377995823099702,-28.215252541435685,-7.100344646381931,-21.476677663813025,-21.13009467302862,-20.9673513913646,-6.95793866201679]
High precision 80 λ22 = [-71.58810311510496,-71.26268948187499,-59.71178598801153,-43.243507246342574,-42.55034126477377,-42.22485470144521,-38.41157314629519,-37.71840716472638,-37.39292060139834,-8.62413004080992]
```

Mass-basis diagnostics were available for both routes.  Float64 reported
eigenpair residuals
`[0.022731565901217474,0.25926111094949195,0.15353047307110929,1.803985874641314e-16,2.2070838395042235e-16]`, nearest relative gaps
`[0.038136573596770106,0.02689896852220496,0.02689896852220496,0.8440612474933319,0.8440612474933319]`, and orthogonality error
`3.9085625896275094e-16`.  High precision 80 reported residuals
`[0.13880469798367817,0.06769877611749449,0.04922746846121679,1.0016073232804368e-17,1.1947650250507238e-16]`, gaps
`[0.006484504478078751,0.03857935931341675,0.006484504478078751,0.8440612474933319,0.8440612474933319]`, and orthogonality error
`1.1343812690346643e-16`.  Quartic and instanton-hierarchy diagnostics were
also non-`nothing` for both routes; all exact component diagnostics and
statuses are retained in the TSV.

## B7 — geometry/filesystem behavior

No checked-out geometry database was assumed.  A synthetic HDF5 fixture was
created under the canonical `h11_002/np_0000001/cy_0000001/cyax.h5` layout,
with Q/L/Kinv plus minimal geometric enrichment fields.  Fixture digest:
`335489dfcc0f5a617906008aaea627761db1d9743e826a51005f47bceba253b8`.

| Route | Samples | Median seconds | Median bytes | Observed result |
|---|---:|---:|---:|---|
| filesystem scan `np_path_generate(2)` | 3 | 0.00044225 | 26592 | one path; index shape `(3,1)` |
| HDF5 query `oriented_potential` | 3 | 0.000462792 | 15776 | Q `(2,3)`, L `(2,3)`, K `(2,2)`, eigenvalues `[1.0,1.0]` |
| HDF5 geometry enrichment `geometry` | 3 | 0.000722041 | 16272 | h21 1; CY volume 1.0; GLSM `(2,2)`; divisor volumes `[1.0,1.0]` |
| HDF5-backed `pq_spectrum` | 2 | 0.0005605005 | 39888 | mass logs `[28.0846788339,28.1846788339]`; signs `[1,1]` |

This is a filesystem/schema behavior probe, not a geometry catalogue or
population result.

## Claim boundary and unavailable fields

Source/implementation facts are separated from empirical timings above.  The
timings are machine-, Julia-build-, dependency-, thread-, and fixture-specific
empirical evidence.  B1 package-load measurements are out of this worker's
scope and remain owned by Worker A.  B4 per-start converged/discarded statuses,
B5a RHS/Hessian counters and the final integrated flow state, and B5b
Hessian/line-search counters are unavailable from the current APIs.  B6 has no
explicit status or eigenvector field on `AxionSpectrum`; status and eigenvectors
were obtained through the returned-value boundary and the documented leading-
Hessian helper, respectively.  B7 uses only the bounded synthetic fixture.  No
production, test, dependency, package, or scientific-schema file was modified.
