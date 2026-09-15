# Issue 148 G1 final independent acceptance at `792a02f`

Recommendation: **PASS** G1 at code commit
`792a02fb77d22cefdfec8fe77969d9c719e7b6d0`, with repair evidence commit
`668ef258dc8ad03046277a5973fe6ed1db1cd0e6`, against accepted G0 commit
`62c5a6135de9ddc8208a1530f50640d666f33cfd`.

All seven [Issue 148 G1 requirements](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148)
pass. The two defects isolated by the
[fresh review of `c6345b0`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/456c793a539a27aec483c9c17385c3237bca77db/docs/src/issue_148_g1_fresh_independent_review_c6345b0.md)
are corrected: the upper satellite is constructed with a target-type `2pi`,
and exact rational inputs no longer narrow through `Float64`. Independent
replay reproduces both 256-bit satellites at `k=k_c-1e-40`, exact symmetry,
stationarity, exact-rational and mixed-precision type preservation, and a
strict 128/256-bit continuation ladder whose errors improve with precision.

This recommendation is limited to N=5, zero phase, the known regular `pi`
branch, and the dimensionless source-reduced radial scale. It makes no G2,
N=8, off-ray, enumeration, through-cusp, metric, physical-observable,
population, or persisted-schema claim. Arb/interval certification is not a G1
requirement under the
[scientific-owner precision clarification](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148#issuecomment-5609698767).

## Contract and replay identity

- Authoritative contract: Issue 148 Gate G1 and owner comment
  `issuecomment-5609698767`.
- Accepted baseline: [G0 audit at `62c5a61`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/62c5a6135de9ddc8208a1530f50640d666f33cfd/docs/src/issue_148_g0_baseline_audit.md).
- Repair evidence: [evidence at `668ef25`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/668ef258dc8ad03046277a5973fe6ed1db1cd0e6/docs/src/issue_148_g1_repair_evidence.md).
- Source: [arXiv:2608.14780v1](https://arxiv.org/pdf/2608.14780v1);
  accepted-G0 local PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- Witness: accepted-G0 N=5 source fixture, `h11=5`, polytope 2787; one
  analytically selected regular zero-phase branch, `branch=:pi`.
- Selection route: continuation of the known branch; not branch enumeration.
  Counting unit: N/A.
- Coordinate/action convention: reduced radian angle `vartheta`, with
  `2pi theta_light=(0,0,1,2,0) vartheta`.
- Scale and units: dimensionless source radial four-cycle multiplier `k`,
  `scale_status=source_reduced`; gradient `g` and Hessian `H` are
  dimensionless. This is not the repository homotopy scale or a physical
  observable.
- Persisted schema: N/A; this gate creates no persisted scientific artifact.
- Environment: Julia 1.12.6, Darwin arm64,
  `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`.
- Review date: 2026-09-09. Worktree HEAD was evidence commit `668ef25` and was
  clean before this report. `git diff --exit-code 792a02f 668ef25 -- src scripts test`
  exited 0, so the tested source, regression, and replay trees equal code
  commit `792a02f`.

## Independent source oracle

The reduced source equations are

```text
a(k) = (32/(255/8)) exp[-2pi k (32-255/8)]
g(theta,k) = sin(theta) + 2a(k) sin(2theta)
H(theta,k) = cos(theta) + 4a(k) cos(2theta).
```

On the regular branch `theta=pi`, `g=0` and `H=-1+4a`. Setting `H=0`
gives

```text
k_c = (4/pi) log(1024/255)
    = 1.770068132610995629125239519150256394269...
```

For `k<k_c`, the two additional analytic critical points are
`acos[-1/(4a)]` and `2pi-acos[-1/(4a)]`; they merge with `pi` at the
catastrophe. The implementation's source construction is visible in
[`poly102_inflation.jl` lines 199--239](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/src/paper_benchmarks/poly102_inflation.jl#L199-L239).
An independent reviewer oracle evaluated the literal formulas above rather
than calling the implementation's critical-scale or critical-point helpers.

At 256 bits and `k=k_c-1e-40`, the implementation and independent oracle gave:

| Quantity | Result |
|---|---:|
| reported critical points | 4 |
| Hessian signs in coordinate order | `[1,-1,1,-1]` |
| minima | 2 |
| lower offset from `pi` | `-1.253314137315500220e-20` |
| upper offset from `pi` | `+1.253314137315500220e-20` |
| lower-coordinate error | `0` at working precision |
| upper-coordinate error | `0` at working precision |
| symmetry error | `0` at working precision |
| maximum `abs(g)` over all four points | `1.17041909e-97` |
| ratio error versus literal source formula | `0` at working precision |
| Hessian error versus literal source formula | `0` at working precision |

This directly closes the prior `Float64(2pi)` failure. The repair constructs
`two_pi = convert(T,2)*convert(T,pi)` before forming the upper satellite at
[`poly102_inflation.jl` lines 228--234](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/src/paper_benchmarks/poly102_inflation.jl#L228-L234).

## G1 acceptance matrix

| Requirement | Result | Independent evidence |
|---|---|---|
| 1. Continue the known branch from a regular point | **PASS** | The four-point path starts at `k_c-1e-3` with `H=7.857e-4` and seed `pi+1e-3`. The first corrector takes three Newton iterations; later steps take one and use the previous corrected state in the predictor. All four records converge with `branch=:pi`. |
| 2. Recover the expected catastrophe from continuation | **PASS** | The path brackets the sign change between converged adjacent records and returns `k=1.77006813256331207995`, `abs(k-k_c)=4.768e-11`. It does not insert `k_c` into the input grid. |
| 3. Gradient residual within tolerance | **PASS** | Default event residual is `4.586e-27`, below `1e-10`. Strict 128/256-bit event residuals are `4.304e-70` and `2.919e-138`, and agree with the independent formula exactly at working precision. |
| 4. Hessian mode approaches/attains zero | **PASS** | The regular-branch Hessian changes from positive to negative across the event. Default `abs(H)=3.745e-11 < 1e-10`; strict 128/256-bit values improve to `2.286e-31` and `2.661e-61`. |
| 5. Maintain branch identity without post-hoc assignment | **PASS** | Branch state is selected from the seed and carried forward. The corrected state feeds the next predictor; the validator checks it against the branch expected from the prior state and does not reassign a label. The exact same-scale satellite is rejected, while a perturbed regular `pi` seed is accepted. See [`poly102_inflation.jl` lines 273--317](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/src/paper_benchmarks/poly102_inflation.jl#L273-L317) and [lines 458--497](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/src/paper_benchmarks/poly102_inflation.jl#L458-L497). |
| 6. Agree with independent analytic N=5 benchmark | **PASS** | The ratio, closed-form `k_c`, regular branch, two-to-one minima change, event, both close 256-bit satellites, gradient, and Hessian agree with the literal source oracle. |
| 7. Focused regression and failure-boundary coverage | **PASS** | The tracked suite passes 63/63. It covers the prior wrong fixture, continuation/event, nonconvergence, coarse localization, endpoint scale, invalid tolerances, branch/satellite rejection, 128/256-bit paths, close satellite coordinates/stationarity, and rational types. The tracked replay completes and records the numerical evidence. |

## Precision and mixed-type adjudication

The owner required source quantities to be created at target precision, input
types to survive the high-precision path, and a genuine high-precision
stability rerun. The type routing at
[`poly102_inflation.jl` lines 98--107](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/src/paper_benchmarks/poly102_inflation.jl#L98-L107)
maps `Rational` to `BigFloat` before transcendental evaluation and promotes a
mixed collection to `BigFloat`. Target-precision constants are built inside
each `setprecision` block.

The independent strict ladder began at the regular point `k_c-1e-5`, included
opposite sides of the event at `k_c +/- 1e-20` or `k_c +/- 1e-40`, and did not
put `k_c` in the continuation grid:

| Precision | Literal tolerances | `abs(k_event-k_c)` | `abs(theta_event-pi)` | `abs(g)` | `abs(H)` |
|---:|---:|---:|---:|---:|---:|
| 128 bits | `1e-30` | `2.91038301e-31` | `0` | `4.30427314e-70` | `2.28580948e-31` |
| 256 bits | `1e-60` | `3.38813179e-61` | `0` | `2.91893294e-138` | `2.66103248e-61` |

Every returned event field was `BigFloat`; all four corrected path records
converged. The difference between the 128-bit and 256-bit constructions of
`k_c` was `8.77696157e-39`, consistent with 128-bit rounding. Stored event
gradient and Hessian values equalled separate literal-formula evaluations at
both precisions.

For `177//100` and `big(177)//big(100)`, ratio outputs were `BigFloat` and
both differed from a target-precision literal source oracle by zero at 256-bit
working precision. The exponent and critical-point coordinates were also
`BigFloat`. A deliberately abstract mixed input vector containing
`Rational{Int}` and `BigFloat` returned `BigFloat` event records throughout,
converged, detected the event, and had maximum path gradient
`7.10891435e-15 < 1e-10`. Thus neither the exact-rational nor mixed
high-precision routes silently passes through `Float64`.

The tracked precision and rational assertions are at
[`issue_148_g1_n5_regression_tests.jl` lines 86--145](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/scripts/issue_148_g1_n5_regression_tests.jl#L86-L145),
and tracked numerical replay is at
[`issue_148_g1_replay_checks.jl` lines 198--268](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/scripts/issue_148_g1_replay_checks.jl#L198-L268).

## Tolerance and failure contracts

The default event returned:

```text
event_converged=true
bracket_width=9.53674916814861717e-11
source_error=4.76836348184406233e-11
residual=4.58636334724509817e-27
abs_hessian=3.74504871558656305e-11
```

All three literal acceptance quantities are below their declared `1e-10`
tolerances. Event creation requires converged neighbors and the conjunction of
gradient, Hessian, and bracket-width tests at
[`poly102_inflation.jl` lines 338--397](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/src/paper_benchmarks/poly102_inflation.jl#L338-L397)
and [lines 506--542](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/792a02fb77d22cefdfec8fe77969d9c719e7b6d0/src/paper_benchmarks/poly102_inflation.jl#L506-L542).

The tolerances have a source-scale rationale. Along `pi`,
`dH/dk|k_c=-pi/4`; a `1e-10` event bracket therefore implies an order
`7.85e-11` Hessian localization scale. The observed `3.745e-11` is
consistent with that derivative, while Float64 roundoff is about `4.5e5`
times smaller than `1e-10`. The values were not tuned to force the analytic
answer.

The advertised failure boundaries also replayed literally:

- one event-refinement iteration with `event_scale_tolerance=1e-14` reports no
  event;
- a nonconverged adjacent correction cannot produce an event;
- an endpoint with small Hessian but a `1e-11` bracket fails an
  `event_scale_tolerance=1e-14` contract;
- nonpositive or nonfinite tolerances throw `ArgumentError`;
- exact analytic satellites are rejected as the regular `pi` branch, while a
  perturbed `pi` seed is accepted.

## Exact checks and observed outcomes

```sh
python3 scripts/agent_verify.py snapshot
```

Passed; worktree was clean at `668ef25`.

```sh
python3 scripts/agent_verify.py run -- env \
  JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. \
  scripts/issue_148_g1_n5_regression_tests.jl
```

Passed, exit 0: `63/63` assertions.

```sh
python3 scripts/agent_verify.py run -- env \
  JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. \
  scripts/issue_148_g1_replay_checks.jl
```

Passed, exit 0. It reproduced the default event, failure boundaries,
branch checks, stable fixed-tolerance 128/256-bit ladder, both close 256-bit
satellites with symmetry error 0 and maximum gradient `1.170419e-97`, and
`BigFloat` rational outputs.

The independent satellite/rational/mixed/strict-ladder probe was run with:

```sh
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. -e '
using CYAxiverse, Printf
P = CYAxiverse.paper_benchmarks.poly102_inflation
ratio(k) = (BigFloat(32)/(BigFloat(255)/BigFloat(8))) *
    exp(-BigFloat(2)*BigFloat(pi)*k*(BigFloat(32)-BigFloat(255)/BigFloat(8)))
setprecision(BigFloat,256) do
    kc=BigFloat(4)/BigFloat(pi)*log(BigFloat(1024)/BigFloat(255)); k=kc-BigFloat("1e-40")
    a=ratio(k); cp=P.n5_reduced_critical_points(k;atol=zero(BigFloat))
    lo=acos(-one(BigFloat)/(BigFloat(4)*a)); hi=BigFloat(2)*BigFloat(pi)-lo
    g(t)=sin(t)+BigFloat(2)*a*sin(BigFloat(2)*t)
    @show length(cp.theta) cp.minima cp.hessian_sign minimum(abs.(cp.theta.-lo))
    @show minimum(abs.(cp.theta.-hi)) abs((lo-BigFloat(pi))+(hi-BigFloat(pi)))
    @show maximum(abs(g(t)) for t in cp.theta) abs(cp.ratio-a)
    q=177//100; qb=big(177)//big(100)
    @show typeof(P.n5_reduced_ratio(q)) typeof(P.n5_reduced_ratio(qb))
    @show typeof(P.n5_reduced_exponent(q)) eltype(P.n5_reduced_critical_points(q).theta)
    mixed=Real[q,kc+BigFloat("1e-5")]
    path=P.n5_reduced_zero_phase_continuation(mixed;
        seed_theta=BigFloat(pi)+BigFloat("1e-3"),
        gradient_tolerance=BigFloat("1e-10"),
        hessian_tolerance=BigFloat("1e-10"),
        event_scale_tolerance=BigFloat("1e-10"),max_iterations=128)
    @show eltype(path).parameters[1] all(s.converged for s in path)
    @show any(s.catastrophe_detected for s in path) maximum(s.gradient for s in path)
end
for (bits,toltext,innertext) in ((128,"1e-30","1e-20"),(256,"1e-60","1e-40"))
    setprecision(BigFloat,bits) do
        kc=BigFloat(4)/BigFloat(pi)*log(BigFloat(1024)/BigFloat(255))
        tol=BigFloat(toltext); d=BigFloat(innertext)
        ks=[kc-BigFloat("1e-5"),kc-d,kc+d,kc+BigFloat("1e-5")]
        path=P.n5_reduced_zero_phase_continuation(ks;
            seed_theta=BigFloat(pi)+BigFloat("1e-3"),
            gradient_tolerance=tol,hessian_tolerance=tol,
            event_scale_tolerance=tol,max_iterations=192)
        s=path[findfirst(x->x.catastrophe_detected,path)]; a=ratio(s.catastrophe_k)
        go=abs(sin(s.catastrophe_theta)+BigFloat(2)*a*sin(BigFloat(2)*s.catastrophe_theta))
        ho=cos(s.catastrophe_theta)+BigFloat(4)*a*cos(BigFloat(2)*s.catastrophe_theta)
        @show bits abs(s.catastrophe_k-kc) abs(s.catastrophe_theta-BigFloat(pi))
        @show s.catastrophe_residual abs(s.catastrophe_hessian)
        @show abs(s.catastrophe_residual-go) abs(s.catastrophe_hessian-ho)
    end
end'
```

It exited 0 with the satellite, rational, mixed-type, and strict-ladder values
reported above. The literal default event bracket was checked with:

```sh
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. -e '
using CYAxiverse, Printf
P=CYAxiverse.paper_benchmarks.poly102_inflation
kc=P.n5_critical_scale(); ks=[kc-1e-3,kc-2e-4,kc+2e-4,kc+1e-3]
p=P.n5_reduced_zero_phase_continuation(ks;seed_theta=pi+1e-3,
    gradient_tolerance=1e-10,hessian_tolerance=1e-10,
    event_scale_tolerance=1e-10,max_iterations=64)
e=P._n5_zero_phase_catastrophe_event(p[2].k,p[2].hessian,p[3].k,
    p[3].hessian,Float64(pi);gradient_tolerance=1e-10,
    hessian_tolerance=1e-10,event_scale_tolerance=1e-10,max_iterations=64)
@printf("event_converged=%s bracket_width=%.17e source_error=%.17e residual=%.17e abs_hessian=%.17e\n",
    e.converged,e.scale_error,abs(e.k-kc),e.residual,abs(e.hessian))'
```

It exited 0 and produced the literal default bracket-width record above.

```sh
git diff --exit-code 792a02f 668ef25 -- src scripts test
git diff --check 62c5a61..792a02f
git diff --check 792a02f..668ef25
```

All exited 0.

## Scope and baseline limitations

The final repair from review evidence commit `456c793` to code commit
`792a02f` changes only the N5 implementation and its two focused tracked
scripts: 25 changed implementation lines, 30 added regression lines, and 27
added replay lines. It does not modify N8, G2, off-ray geometry, metric or
phase conventions, physical normalization, package version, dependencies, or
persisted schema.

The known full-suite phase-volume detuning failure and the N8 JET/Revise audit
limitations are accepted baseline limitations from G0 and were intentionally
not rerun. They limit repository-wide CI claims but do not contradict the
focused scientific G1 acceptance established here. G2 remains blocked until
the manager explicitly accepts this G1 recommendation and separately respects
the owner boundary on the N8 metric normalization.
