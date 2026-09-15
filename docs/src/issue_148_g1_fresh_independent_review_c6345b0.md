# Issue 148 G1 fresh independent review at `c6345b0`

Recommendation: **FAIL** for G1 at code commit
`c6345b03281ec560366c6aea21600bb3b998b0c5`, with evidence commit
`cc0cdb2792458bd43789cf37f4bdf0e563f2bf7d`, against accepted G0 commit
`62c5a6135de9ddc8208a1530f50640d666f33cfd`.

The bounded branch-identity repair works. The exact source satellite at
`k=k_c-1e-5` is rejected rather than labelled `:pi`, a perturbed regular `pi`
seed is accepted and continued to a localized event, and even substantially
closer 128/256-bit satellites were not mislabelled. The regular `pi` branch also
passes a genuinely constructed, decreasing-tolerance precision ladder.

One material target-precision defect remains in the critical-point helper used
by branch validation. Its upper satellite is constructed as `2pi-acos(...)`,
where `2pi` is evaluated in `Float64` before the subtraction. At 256 bits and
`k=k_c-1e-40`, the helper omits the analytic upper satellite and reports a
spurious nonstationary point on the lower side of `pi`. The existing 256-bit
test checks only four points and two minima at `k_c-1e-20`, so it does not detect
this coordinate failure. This is an ordinary implementation defect, not an
owner/source ambiguity.

## Contract, identity, and scope

- Authoritative contract: [Issue 148](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148),
  Gate G1, and the [scientific-owner precision clarification](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148#issuecomment-5609698767).
- Accepted baseline: [G0 audit at `62c5a61`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/62c5a6135de9ddc8208a1530f50640d666f33cfd/docs/src/issue_148_g0_baseline_audit.md).
- Prior defect report: [fresh review at `0cf16ca`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/13050466ef2bb0c91e2ee015bf2ea4fb683974e7/docs/src/issue_148_g1_fresh_independent_review_0cf16ca.md).
- Repair evidence reviewed: [`cc0cdb2` evidence](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/cc0cdb2792458bd43789cf37f4bdf0e563f2bf7d/docs/src/issue_148_g1_repair_evidence.md).
- Source: [arXiv:2608.14780v1](https://arxiv.org/pdf/2608.14780v1), local
  PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- Witness: accepted-G0 N=5 source fixture, `h11=5`, polytope 2787; one
  analytically selected regular zero-phase branch. Selection route is not an
  enumeration and the counting unit is N/A.
- Coordinate/action convention: reduced radian angle `vartheta`,
  `2pi theta_light=(0,0,1,2,0) vartheta`.
- Scale and units: dimensionless source radial four-cycle multiplier `k`, with
  `scale_status=source_reduced`; `g` and `H` are dimensionless. This is neither
  repository homotopy scale and is not a physical observable.
- Persisted schema: N/A. No persisted artifact or schema is produced.
- Environment: Julia 1.12.6, `arm64-apple-darwin24.0.0`, Darwin,
  `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`.
- Review scope is N5 zero-phase source-reduced behavior only. It makes no N8,
  off-ray, exhaustive-branch, through-cusp, metric, phase, physical-observable,
  population, or persisted-schema claim. G2 pseudo-arclength/metric requirements
  and G3 reframing remain future gates.

## G1 acceptance matrix

| Requirement | Result | Independent evidence |
|---|---|---|
| 1. Continue the known branch from a regular point | **PASS** | The four-point path starts from `pi+1e-3`, uses the previous corrected state in the next predictor, and returns four converged `:pi` records. The first correction takes three Newton iterations and subsequent records take one. |
| 2. Recover the expected catastrophe from continuation | **PASS** | A grid omitting `k_c` brackets the Hessian sign change and returns `k=1.77006813256331207995`, `abs(k-k_c)=4.768e-11`. A separate two-point near-cusp path returns `abs(k-k_c)=3.815e-11`. |
| 3. Gradient residual within tolerance | **PASS** | Default event residual is `4.586e-27`, below the declared `1e-10`; stored results equal the independent analytic gradient oracle on the strict precision ladder. |
| 4. Hessian mode approaches/attains zero | **PASS** | The regular branch changes from positive to negative Hessian across the event. Default event `abs(H)=3.745e-11`, below `1e-10`; the source predicts `H(pi,k)=-1+4a(k)`. |
| 5. Maintain branch identity without post-hoc assignment | **PASS** | The corrected validator measures distance to the expected branch point. The exact same-scale satellite is rejected, while the perturbed `pi` solution is closer to `pi` than half the satellite separation. High-precision satellite and seed-sweep probes found no `:pi` mislabels. |
| 6. Agree with the analytic N5 benchmark | **FAIL** | The source ratio, `k_c`, regular branch, event, and two-to-one minima change agree. However, `n5_reduced_critical_points` loses the analytic upper satellite at sufficiently close target-precision scales and emits a nonstationary coordinate instead. |
| 7. Focused regression and failure-boundary coverage | **PARTIAL** | The tracked suite passes 49/49 and covers failed correctors, nonconverged neighbors, coarse localization, endpoint scale, invalid controls, same-scale satellite rejection, and BigFloat results. Its 256-bit check asserts only count/minima and loose satellite bounds; it does not assert coordinate symmetry or stationarity, so the remaining precision defect passes. |

The owner clarification is otherwise met on the regular branch. Source
quantities and tolerances were constructed inside each `setprecision` block,
`BigFloat` result fields remain `BigFloat`, and the strict 53/128/256-bit ladder
records `k_c`, `theta_c`, `g`, and `H`. No Arb check is needed because the
analytic source solution is the stronger oracle.

## Branch identity and event recovery

The repair at
[`poly102_inflation.jl` lines 268--286](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/c6345b03281ec560366c6aea21600bb3b998b0c5/src/paper_benchmarks/poly102_inflation.jl#L268-L286)
replaces the vacuous nearest-critical-point residual fallback with distance to
the expected branch point. At `k=k_c-1e-5`, the analytic satellite is
`3.13762933148015`, a distance `0.003963322109643119` from `pi`. Direct replay
returned `ArgumentError` for that exact satellite. Starting from `pi+1e-3` at
the same scale instead returned:

```text
branch=:pi
converged=true
theta=3.141593152295353
abs(theta-pi)=4.987055599592338e-7
gradient=3.916839627038023e-12
hessian=7.85401210356973e-6
```

Continuing that perturbed state across `[k_c-1e-5,k_c+1e-5]` kept both records
on `:pi` and localized the event to source-oracle error `3.815e-11`, event
gradient `3.669e-27`, and event Hessian `2.996e-11`. Thus the supported
perturbed branch is a real predictor/corrector path with event recovery, rather
than an independent grid solve or post-hoc label.

At 128 bits (`k_c-1e-20`) and 256 bits (`k_c-1e-20`, `k_c-1e-40`), both exact
analytic satellites were rejected. Twenty-five seeds spanning each close pair
gave respectively 9, 9, and 11 accepted corrections; none of those accepted
records was closer to a satellite than to `pi`. Default-tolerance 256-bit
probes also rejected exact satellites at `k_c-1e-20`, `k_c-1e-40`, and
`k_c-1e-70`. This independently confirms that the repaired branch predicate is
not the remaining failure.

The tracked regression additions exercise the exact same-scale satellite and
perturbed `pi` seed at
[`issue_148_g1_n5_regression_tests.jl` lines 70--115](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/c6345b03281ec560366c6aea21600bb3b998b0c5/scripts/issue_148_g1_n5_regression_tests.jl#L70-L115).

## Material target-precision defect

The source gives

```text
a(k)=(32/(255/8)) exp[-2pi k (32-255/8)]
k_c=(4/pi) log(1024/255)
theta_satellite,± = pi ± delta
```

The implementation correctly constructs the lower satellite with a
target-precision `acos`, but
[`poly102_inflation.jl` line 228](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/c6345b03281ec560366c6aea21600bb3b998b0c5/src/paper_benchmarks/poly102_inflation.jl#L228)
constructs the other as `2pi-acos(...)`. In Julia 1.12.6, `typeof(2pi)` is
`Float64`. At 256-bit precision,

```text
BigFloat(2pi) - 2*BigFloat(pi)
    = -2.44929359829470635445213186455e-16
```

At `k=k_c-BigFloat("1e-40")`, the analytic satellites have offsets
`±1.25331413731550025120788264241e-20` from `pi`. Even with
`atol=zero(BigFloat)`, the helper reports offsets

```text
-2.44916826688097480442701107629e-16
-1.25331413731550025120788264241e-20
 0
```

after the unrelated zero point. The positive satellite is absent. Its nearest
reported-coordinate error is `1.2533141373155e-20`; the inserted coordinate has
`abs(g)=7.346e-48`, while a 256-bit evaluation at the true upper satellite has
`abs(g)=1.170e-97`. Counts and signs remain superficially plausible
(`4` points, signs `[1,-1,-1,1]`, `2` minima), which is why the tracked
count-only regression passes.

The fix is bounded: construct `two_pi` in `T` and derive the upper satellite
from it, then assert both analytic coordinates, their symmetry about `T(pi)`,
and their gradient residuals at a close 256-bit scale. Exhaustive branch
enumeration is not required; this is correctness of the helper already used by
the intrinsic branch validator.

## Source formulas, tolerances, and precision ladder

The source PDF hash was independently reproduced. Equations 31--36 give the
reduced coordinate, source coefficients, normalized ratio, and critical scale.
The implementation at
[`poly102_inflation.jl` lines 195--235](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/c6345b03281ec560366c6aea21600bb3b998b0c5/src/paper_benchmarks/poly102_inflation.jl#L195-L235)
matches those formulas. Along `pi`,
`g=sin(theta)+2a sin(2theta)=0`, `H=-1+4a`, and
`dH/dk|k_c=-pi/4`. Therefore an event bracket of `1e-10` implies a natural
Hessian localization scale of about `7.85e-11`, consistent with the observed
`3.745e-11`. The default `1e-10` gradient/Hessian/scale tolerances are absolute
on dimensionless order-one quantities; Float64 roundoff `2.22e-16` is about
`4.5e5` times smaller.

The tracked fixed-tolerance replay at
[`issue_148_g1_replay_checks.jl` lines 198--243](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/c6345b03281ec560366c6aea21600bb3b998b0c5/scripts/issue_148_g1_replay_checks.jl#L198-L243)
constructs its BigFloat source values inside `setprecision`. A separate strict
ladder gave:

| Precision | Requested tolerance | `k_c` | `abs(k-k_c)` | `theta_c` | `abs(theta-pi)` | `abs(g)` | `abs(H)` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 bits | `1e-12` | `1.7700681326109957` | `3.72591e-13` | `3.141592653589793` | `0` | `3.58439e-29` | `2.92655e-13` |
| 128 bits | `1e-30` | `1.770068132610995629125239519150256394278` | `3.23117e-31` | `3.141592653589793238462643383279502884195` | `0` | `4.77870e-70` | `2.53776e-31` |
| 256 bits | `1e-60` | `1.770068132610995629125239519150256394269282054995653154938614774330771515119501` | `2.54895e-61` | `3.141592653589793238462643383279502884197169399375105820974944592307816406286198` | `0` | `2.19596e-138` | `2.00194e-61` |

At every precision, the stored gradient and Hessian exactly equalled a separate
evaluation of the source analytic formulas at the returned event. This confirms
stable/improving `k_c`, `theta_c`, `g`, and `H` for the regular branch. It does
not excuse the separate upper-satellite construction defect.

The prior failure cases also behave correctly: one-iteration localization at
`event_scale_tolerance=1e-14`, a nonconverged neighbor, and an event endpoint
whose bracket exceeds `1e-14` all produce no catastrophe. Invalid nonpositive
or nonfinite tolerances throw. Event creation requires two converged adjacent
states and literal gradient, Hessian, and bracket-width gates at
[`poly102_inflation.jl` lines 333--392](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/c6345b03281ec560366c6aea21600bb3b998b0c5/src/paper_benchmarks/poly102_inflation.jl#L333-L392)
and
[`poly102_inflation.jl` lines 501--537](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/c6345b03281ec560366c6aea21600bb3b998b0c5/src/paper_benchmarks/poly102_inflation.jl#L501-L537).

## Conventions and compatibility

The functional diff from G0 is confined to the N5 reduced formula,
continuation, directly affected pilot expectations, and focused tests. The
source period-one GLSM and reduced-radian mapping, radial `k`, geometry, basis,
metric, phase, units, physical claims, dependencies, package version, and
persisted contracts are unchanged. N8 production formulas are unchanged; the
pilot now compares N8 to its own `N8_KC` after decoupling the formerly incorrect
N5 reuse. The new continuation result and function are additive interfaces and
do not remove or reshape an existing interface. The evidence commit changes
only the repair document; its `src`, `scripts`, and `test` trees equal the code
commit.

## Standards

The independent Standards reviewer reported two hard findings:

1. `n5_reduced_ratio`, `n5_reduced_exponent`, and the continuation collection
   type promotion start from `Float64`. A `Rational{Int}` input therefore
   returns Float64 `ratio`, `k`, and `theta`, contrary to the repository rule
   against forcing exact-rational paths through Float64. The reviewed BigFloat
   and BigInt paths are not narrowed by this mechanism.
2. The repair evidence does not spell out a schema identity field. This review
   closes that evidence omission above with `Persisted schema: N/A`.

The reviewer also recorded judgment-call smells: duplicated N5 ratio/critical
logic between `reduced_models.jl` and `poly102_inflation.jl`; Symbol branch
identity and repeated branch switches; the three tolerance values as a data
clump; the misspelled compatibility alias; and the now-unused
`_n5_nearest_critical_distance` helper. These are secondary to the scientific
precision defect and do not require an interface redesign within G1.

## Spec

The independent Spec reviewer found that all required behavior passes at the
tested scale and precision: source `k_c`/ratio, predictor/corrector
continuation, event localization, stationarity and Hessian residuals,
two-to-one minima change, failed-step suppression, same-scale satellite
rejection, the perturbed `pi+1e-3` seed, and the 128/256-bit tracked ladder.

The reviewer independently found the deeper precision-stability defect above:
at 256 bits and `k_c-1e-40`, even with zero classification tolerance, the
helper's Float64 `2pi` loses the upper satellite and inserts a point displaced
about `-2.449e-16` from `pi`. This contradicts the owner requirement that
arbitrary-precision source quantities be constructed at target precision and
G1's analytic-agreement requirement. No material scope creep was found.

## Exact checks and outcomes

1. `python3 scripts/agent_verify.py snapshot` passed on a clean worktree before
   this review document was added.
2. `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_n5_regression_tests.jl`
   passed 49/49 assertions. Raw stdout:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_fy9tcu9j/run.stdout.log`.
3. `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl`
   exited 0 with the event, failure-boundary, branch, and fixed-tolerance ladder
   values above. Raw stdout:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_j24fkrrn/run.stdout.log`.
4. Direct Float64 replay computed the satellite from `acos(-1/(4a(k)))` at the
   same `k_c-1e-5`, proved its rejection, accepted the perturbed `pi` seed, and
   replayed the two-point event path.
5. Direct target-precision replays constructed `k_c`, grid, seed, tolerances,
   and analytic oracles inside 128/256-bit blocks. The strict ladder, exact
   satellite rejections, seed sweeps, and close-scale coordinate reproduction
   reported above all exited 0.
6. `git diff --check 62c5a61...c6345b0` and
   `git diff --check 0cf16ca..c6345b0` passed.
7. The known full-suite phase-volume failure and N8 JET/Revise audit limitations
   were accepted as proven baseline limitations and were not rerun.

The exact command that reproduces the material defect is:

```sh
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. -e 'using CYAxiverse; P=CYAxiverse.paper_benchmarks.poly102_inflation; setprecision(BigFloat,256) do; kc=BigFloat(4)/BigFloat(pi)*log(BigFloat(1024)/BigFloat(255)); k=kc-BigFloat("1e-40"); a=P.n5_reduced_ratio(k); lower=acos(-one(BigFloat)/(BigFloat(4)*a)); upper=BigFloat(2)*BigFloat(pi)-lower; cp=P.n5_reduced_critical_points(k;atol=zero(BigFloat)); grad(t)=sin(t)+BigFloat(2)*a*sin(BigFloat(2)*t); println((k_delta=kc-k,true_lower_offset=lower-BigFloat(pi),true_upper_offset=upper-BigFloat(pi),reported_offsets=cp.theta.-BigFloat(pi),nearest_upper_error=minimum(abs.(cp.theta.-upper)),spurious_gradient=abs(grad(cp.theta[2])),true_upper_gradient=abs(grad(upper)),two_pi_float_error=BigFloat(2pi)-BigFloat(2)*BigFloat(pi),signs=cp.hessian_sign,minima=cp.minima)); end'
```

Return only the bounded target-precision satellite-construction correction and
a coordinate/stationarity regression to the G1 implementation worker. Address
or explicitly narrow the Rational-input contract in the same bounded pass. Do
not begin G2 before another fresh G1 review passes.

Review totals: Standards reported 2 hard findings and 6 judgment-call smell
labels; the worst is exact-rational narrowing. Spec reported 1 wrong behavior,
no otherwise missing G1 item, and no scope creep; the worst is the target-
precision upper-satellite construction. The scientific gate fails on the Spec
axis.
