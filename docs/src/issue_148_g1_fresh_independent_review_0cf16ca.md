# Issue 148 G1 fresh independent review at `0cf16ca`

Recommendation: **FAIL** for G1 at code commit
`0cf16ca3d15b7a0823a9f315d7637dd718a77de0`, with evidence commit
`c391d3075f9b41ef6efb3407f0495d45b214b9cd`, against accepted G0 commit
`62c5a6135de9ddc8208a1530f50640d666f33cfd`.

The source formula, catastrophe localization, literal gradient/Hessian/event
tolerance gates, failed-step rejection, and target-precision continuation now
work on the tested regular path. One material scientific defect remains: the
new intrinsic branch validator accepts a neighboring analytic stationary
branch as `:pi`. Therefore branch identity and its regression boundary do not
yet satisfy G1.

## Scope and replay identity

- Authoritative contract: [Issue 148](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148),
  G1, with the [scientific-owner precision clarification](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148#issuecomment-5609698767).
- Source: arXiv:2608.14780v1, PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- Witness: the accepted-G0 N=5 source fixture, `h11=5`, polytope 2787; one
  analytically selected regular zero-phase branch, not a branch enumeration.
- Coordinate/action convention: reduced radian angle `vartheta`, with
  `2pi theta_light=(0,0,1,2,0) vartheta`, and
  `a(k)=(32/(255/8)) exp[-2pi k (32-255/8)]`.
- Scale: dimensionless source radial four-cycle multiplier `k`; this is not
  either repository homotopy scale. The review makes no N=8, physical
  observable, off-ray, population, metric, phase, or persisted-schema claim.
- Environment: Julia 1.12.6, Darwin arm64,
  `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`.

## Material finding

### FAIL: branch validation is vacuous for an exact neighboring critical point

G0 requires the previous corrected state to carry the known regular branch
identity, without post-hoc nearest-neighbor assignment
([G0 audit, lines 297--301](issue_148_g0_baseline_audit.md)). The implementation
does carry a seed label and the previous state, but its acceptance predicate is
incorrect:

```julia
expected_index = _n5_periodic_index(T, points, anchor)
closest_index, closest_distance = _n5_nearest_critical_distance(points, theta)
tolerance = _n5_branch_tolerance(points, expected_index)
return closest_index == expected_index || closest_distance <= tolerance
```

At [`poly102_inflation.jl` lines 283--286](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/0cf16ca3d15b7a0823a9f315d7637dd718a77de0/src/paper_benchmarks/poly102_inflation.jl),
`closest_distance` is the distance to whichever stationary point is nearest.
It is zero for every exact stationary point, so the second disjunct accepts a
point even when `closest_index != expected_index`.

The source satellite at `k=k_c-1e-5` is

```text
theta_satellite = acos(-1/(4a(k))) = 3.13762933148015
abs(pi-theta_satellite)            = 0.003963322109643119
```

It lies inside the advertised `0.01` `:pi` seed window. A direct replay at the
reviewed SHA returned:

```text
accepted=true
branch=pi
theta=3.13762933148015
theta_minus_sat=0.0
gradient=8.673617379884035e-19
hessian=-1.5707963268329905e-5
converged=true
```

The committed rejection at
[`issue_148_g1_n5_regression_tests.jl` lines 67--68](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/0cf16ca3d15b7a0823a9f315d7637dd718a77de0/scripts/issue_148_g1_n5_regression_tests.jl)
uses `3.101964568393247`, the satellite at `k_c-1e-3`, while its first scale is
`k_c-1e-5`. That value is about `0.0396` from `pi` and is rejected by the fixed
seed window before the corrected stationary state is tested. The replay and
repair evidence therefore do not exercise the reported near-cusp branch
boundary.

This is a bounded implementation defect, not a scientific-owner ambiguity.
Validate the corrected point against the selected expected point itself, then
add a regression that computes the satellite from `a(k)` at the same near-cusp
scale and proves that it cannot acquire `branch=:pi`.

## Acceptance checks that pass

1. **Source formula and scale.** The implementation constructs the reduced
   ratio from the source prefactor and exponent and returns
   `k_c=1.77006813261099571477` in Float64. Target-precision integer and `pi`
   construction is present for `BigFloat` paths.
2. **One-step continuation on the intended path.** Starting at
   `k_c-1e-3` from `pi+1e-3`, the four-point path uses each corrected state as
   the next predictor input. It returned four converged `:pi` records; the
   first required three Newton iterations and later records one each.
3. **Event recovery without supplying `k_c` as a grid point.** The grid omits
   the analytic event. Bisection returned
   `k=1.77006813256331207995`, source-oracle error `4.768e-11`,
   `theta=3.141592653589793116`, gradient `4.586e-27`, and Hessian
   `3.745e-11`.
4. **Literal event contracts on the reviewed paths.** Event creation now
   requires two converged adjacent continuation states. Endpoint candidates
   use their actual bracket width, and corrector convergence uses absolute
   `|g|`. The one-iteration `1e-14` localization, near-event failed-step, and
   endpoint-width adversarial paths all returned no catastrophe.
5. **Failure regression.** The tracked focused suite passed 40/40 assertions,
   including invalid controls, failed correctors, strict endpoint/event
   localization, and mixed/BigFloat result types. It does not cover the true
   near-cusp satellite above.
6. **No hidden convention/schema change.** The functional diff is confined to
   N=5 reduced behavior and directly affected benchmark/test code. No N=8,
   phase, basis, metric, units, physical claim, dependency, package version, or
   persisted-schema change was found. The change adds the continuation function
   and result type, so the repair evidence should describe this as an additive
   N=5 API rather than “no public API edits.”

## Independent precision and tolerance adjudication

The tracked replay genuinely constructs 128- and 256-bit `BigFloat` source
values inside each `setprecision` block and preserves `BigFloat` result types.
Its fixed `1e-10` ladder is stable: event-scale error and Hessian are both
unchanged at about `4.768e-11` and `3.745e-11`; the event gradient improves
from `4.586e-27` in Float64 to `7.052e-50` and `4.108e-88` at 128 and 256 bits.
The replay casts printed results to Float64 and does not print per-precision
`k_c` or `theta_c`, so this review also ran a decreasing-tolerance ladder:

| Precision | requested tolerance | `abs(k-k_c)` | `abs(theta-pi)` | gradient | Hessian |
|---:|---:|---:|---:|---:|---:|
| 128 bits | `1e-30` | `3.23117e-31` | `0` | `4.77870e-70` | `2.53776e-31` |
| 256 bits | `1e-60` | `2.54895e-61` | `0` | `2.19596e-138` | `2.00194e-61` |

This confirms target-precision construction and stable, improving continuation
results independently of the Float64 oracle value.

For the default absolute tolerances, `g` and `H` are dimensionless and have
order-one natural scales. Along the exact `pi` branch,
`H(pi,k)=-1+4a(k)` and `dH/dk|kc=-pi/4`. Thus an event bracket of `1e-10`
implies an order-`7.85e-11` Hessian localization scale, consistent with the
observed `3.745e-11`. Float64 roundoff on these unit-scale quantities is
`2.22e-16`, about `4.5e5` times smaller than `1e-10`; 128- and 256-bit
roundoff is smaller still. This supplies the scale/roundoff relation missing
from the repair evidence's tolerance section.

One additional precision boundary should be corrected or documented. At 256
bits, `n5_reduced_critical_points(k_c-BigFloat("1e-20"))` uses the default
`atol=64eps(Float64)` and returns two points with one minimum. With
`atol=zero(BigFloat)`, it returns the analytically expected four points and two
minima, with the satellite `1.25331e-10` from `pi`. The fixed Float64 default at
[`poly102_inflation.jl` lines 224--234](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/0cf16ca3d15b7a0823a9f315d7637dd718a77de0/src/paper_benchmarks/poly102_inflation.jl)
limits the high-precision classification used by branch validation.

## Standards axis

The independent standards review reported three hard findings and three
judgment calls:

1. **Hard: high precision is narrowed in changed N=5 helpers.** The
   `64eps(Float64)` default produces the classification boundary above.
   Rational input is also promoted through Float64 in `n5_reduced_ratio` and
   continuation. This conflicts with the repository's intentional-precision
   rule.
2. **Hard: the replay report is not itself an executable gate.** Its `capture`
   helper catches errors, and its result checks print values rather than assert
   them. The separate 40/40 regression script is executable, but it misses the
   true near-cusp satellite.
3. **Hard: replay identity is incomplete in the repair evidence.** The repair
   evidence records source hash, code SHA, Julia, and date, but omits the
   geometry/witness, selection route/counting unit, coordinate/action
   convention, units, and scale status required by `AGENTS.md` section 4 and
   the scientific-reproduction skill. This review records those fields above.
4. **Judgment: Mysterious Name.** The misspelled
   `N5_REDUCED_CATASROPHE_REFINE_ITERATIONS` alias remains, although no
   repository use was found.
5. **Judgment: Duplicated Code.** Production N=5 ratio/critical-point logic
   remains duplicated between `reduced_models.jl` and
   `poly102_inflation.jl`; separate replay-oracle duplication is justified.
6. **Judgment: Repeated Switches.** Branch handling is repeated across branch
   validation, anchor selection, and event creation.

## Spec axis

The independent spec review reported one missing/partial requirement and one
wrong behavior:

1. **Missing/partial:** the regression does not exercise the near-cusp branch
   identity boundary required by “Carry branch identity in continuation state;
   do not assign the result by post-hoc nearest-neighbor matching.”
2. **Wrong:** the validator's `closest_distance <= tolerance` fallback accepts
   the true neighboring stationary point as `branch=:pi`, producing the direct
   replay above and contradicting the repair evidence's rejection claim.

No scope creep was found.

## Exact checks and outcomes

1. `python3 scripts/agent_verify.py snapshot` passed on a clean worktree before
   this review document was added.
2. `python3 scripts/agent_verify.py diff-check` and
   `git diff --check 62c5a61...c391d30` passed.
3. `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl`
   passed as a process. The observed results are recorded above; raw output was
   written to
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_s3yk80c0/run.stdout.log`.
4. `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g1_n5_regression_tests.jl`
   passed 40/40 assertions; raw output was written to
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_an9oz5l9/run.stdout.log`.
5. A direct Julia replay computed `theta_satellite=acos(-1/(4a(k)))` at the
   same `k=k_c-1e-5` and supplied it as the one-point seed. It exited 0 with the
   incorrect accepted identity quoted in the material finding.
6. A direct Julia precision replay constructed `k_c`, grid points, seed, and
   tolerances inside 128- and 256-bit blocks, with `max_iterations=256`. It
   exited 0 with the ladder in the precision table.

The exact direct adversarial command was:

```sh
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. -e 'using CYAxiverse; P=CYAxiverse.paper_benchmarks.poly102_inflation; kc=P.n5_critical_scale(); k0=kc-1e-5; a=P.n5_reduced_ratio(k0); sat=acos(-1/(4a)); println("kc=",repr(kc)); println("k0=",repr(k0)); println("satellite=",repr(sat)); println("distance_from_pi=",repr(pi-sat)); try p=P.n5_reduced_zero_phase_continuation([k0]; seed_theta=sat, gradient_tolerance=1e-12); s=p[1]; println("accepted=true"); println("branch=",s.branch); println("theta=",repr(s.theta)); println("theta_minus_sat=",repr(s.theta-sat)); println("gradient=",repr(s.gradient)); println("hessian=",repr(s.hessian)); println("converged=",s.converged); catch e; println("accepted=false"); println("error=",typeof(e),": ",e); end'
```

The exact precision-boundary and decreasing-tolerance command was:

```sh
env JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. -e 'using CYAxiverse; P=CYAxiverse.paper_benchmarks.poly102_inflation; setprecision(BigFloat,256) do; kc=BigFloat(4)/BigFloat(pi)*log(BigFloat(1024)/BigFloat(255)); k=kc-BigFloat("1e-20"); d=P.n5_reduced_critical_points(k); d0=P.n5_reduced_critical_points(k;atol=zero(BigFloat)); println("precision_boundary_bits=",precision(BigFloat)); println("precision_boundary_delta=",kc-k); println("default_points=",length(d.theta)); println("default_minima=",d.minima); println("zero_atol_points=",length(d0.theta)); println("zero_atol_minima=",d0.minima); println("zero_atol_satellite_offset=",pi-d0.theta[2]); end; for (bits,tolstr) in ((128,"1e-30"),(256,"1e-60")); setprecision(BigFloat,bits) do; kc=BigFloat(4)/BigFloat(pi)*log(BigFloat(1024)/BigFloat(255)); t=BigFloat(tolstr); grid=[kc-BigFloat("1e-3"),kc-BigFloat("2e-4"),kc+BigFloat("2e-4"),kc+BigFloat("1e-3")]; p=P.n5_reduced_zero_phase_continuation(grid;seed_theta=BigFloat(pi)+BigFloat("1e-3"),gradient_tolerance=t,hessian_tolerance=t,event_scale_tolerance=t,max_iterations=256); idx=findfirst(s->s.catastrophe_detected,p); s=p[idx]; println("ladder_bits=",bits); println("ladder_type=",typeof(s.catastrophe_k)); println("ladder_k=",s.catastrophe_k); println("ladder_k_error=",abs(s.catastrophe_k-kc)); println("ladder_theta_error=",abs(s.catastrophe_theta-BigFloat(pi))); println("ladder_gradient=",s.catastrophe_residual); println("ladder_hessian=",s.catastrophe_hessian); end; end'
```

The known full-suite phase-volume mismatch and N8 JET/Revise audit failures are
pre-existing and outside this G1 verdict; they were not rerun.

Return only the bounded branch-identity correction and its true near-cusp
failure regression to the G1 implementation worker. Correct or explicitly
document the high-precision critical-point classification boundary and complete
the repair evidence identity/tolerance wording in the same bounded pass. Do
not begin G2 before a fresh G1 review passes.

Review totals: Standards 6 findings, worst is high-precision/branch-validation
contract narrowing; Spec 2 findings, worst is the incorrect intrinsic branch
identity. The scientific gate fails on the Spec axis.
