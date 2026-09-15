# Issue 148 G1 independent review

Independent-review recommendation: **FAIL** at implementation commit
`2a4e495ccdd838cc5b1e884fbac136115a7d433f` against accepted G0 base
`62c5a6135de9ddc8208a1530f50640d666f33cfd`.

The N=5 source value is repaired correctly, and the supplied benchmark command
passes, but the change does not recover the catastrophe location from
continuation. It evaluates a caller-supplied grid containing the analytic
critical scale. Branch identity and failure-boundary requirements are also not
met. These are bounded implementation and regression gaps; no scientific-owner
decision is required, and G2 must not begin.

## Review identity and conventions

- Scope: [GitHub Issue 148](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148),
  Gate G1, plus the accepted G1 packet in
  [`issue_148_g0_baseline_audit.md`](issue_148_g0_baseline_audit.md).
- Source: *Catastrophic Inflation in the Axiverse*, arXiv:2608.14780v1;
  source PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- Source-fixed zero-phase reduced model:
  `a(k)=(32/(255/8)) exp[-2pi k (32-255/8)]` and
  `k_c=(4/pi)log(1024/255)=1.7700681326109957`.
- Claim boundary: N=5 zero-phase source-reduced radial benchmark only. No N=8,
  off-ray, exhaustive enumeration, through-cusp, normalization, metric, basis,
  phase, schema, or physical-claim change was accepted or found in this diff.
- Julia 1.12.6, Darwin arm64. The new numerical path runs in `Float64` even for
  `BigFloat` inputs.

## Independent gate findings

1. **FAIL — the catastrophe is supplied, not recovered.**
   `pilot_benchmark_regression` constructs its grid around
   `benchmark.n5_critical_scale()` and inserts that value as the middle sample
   ([`scripts/inflation_scale_continuation.jl`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/2a4e495ccdd838cc5b1e884fbac136115a7d433f/scripts/inflation_scale_continuation.jl),
   lines 1832--1836). The continuation routine returns results only at the
   caller's `k_values`; it has no Hessian-zero event location, interpolation,
   augmented solve, or returned continuation estimate of `k_c`
   ([`poly102_inflation.jl`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/2a4e495ccdd838cc5b1e884fbac136115a7d433f/src/paper_benchmarks/poly102_inflation.jl),
   lines 225--277). A replay whose grid stopped at `k_c-1e-5` returned the
   same three input scales and no `near_catastrophe` point. This does not meet
   Issue 148 G1 acceptance 2, “the expected catastrophe location is recovered
   from continuation.” The benchmark's `n5_kc_residual` compares the source
   constant to the same closed form, independently of the continued path.

2. **FAIL — branch identity metadata is false outside the default seed.**
   The docstring says `seed_theta` selects the branch (lines 213--216), while
   every record is assigned `branch=:pi` (line 272). With `seed_theta=0`, the
   replay returned `theta=0` at all three scales but `branch=:pi` at all three.
   Previous corrected `theta` is propagated, so the implementation is not
   post-hoc nearest-neighbour matching; however, it does not carry truthful
   intrinsic branch identity as required. Either restrict and validate this
   API to the known `pi` branch or represent the selected branch in state and
   test it.

3. **FAIL — no documented failure boundary is tested, and convergence status
   has an off-by-one defect.** The new tests cover only successful default
   paths ([`test/runtests.jl`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/2a4e495ccdd838cc5b1e884fbac136115a7d433f/test/runtests.jl), lines 336--352 and
   3387--3396). They do not exercise invalid input, solver non-convergence, or
   a continuation boundary. The corrector tests the residual before an update
   but does not test it after the final permitted update (implementation lines
   245--259). Independently, `seed_theta=3.14`, `k=k_c-1e-3`, and
   `max_iterations=2` returned gradient `1.3861899133062302e-16` but
   `converged=false`. Add a documented failure contract and focused tests,
   including the final-update case.

4. **FAIL — validation is circular and incomplete at the repaired boundary.**
   The tested path uses `theta=pi`, an analytic stationary point for every
   `k`; all five default records exited on their first residual check. The
   ratio is tested only at the supplied `k_c`, where the anchored
   implementation returns `1/4` by construction. The independently correct
   root reduced model and the raw action formula are not compared to the
   repaired `poly102_inflation` ratio away from `k_c`, and the added tests do
   not establish the two-to-one minima change in the repaired namespace.
   A corrected gate should start from a perturbed regular seed, show actual
   corrector work, locate the Hessian-zero event, compare the resulting `k`
   against the independent source formula, and test minima on both sides.

5. **FAIL — the required reproducibility evidence is not tracked.** Issue 148
   requires exact commands, observed values, tolerances, source identity, and
   code revision. Commit `2a4e495` changes only the implementation, benchmark,
   and tests. The local `validation/issue_148_g1_repair_evidence.md` is ignored
   by `.gitignore`, omits the source fingerprint and Julia environment, and
   does not justify the absolute `1e-10` gradient/Hessian tolerances. A tracked
   corrected evidence record should state the precision boundary and explain
   the residual, Hessian, event-location, and comparison tolerances from
   roundoff and observed side-of-cusp separation.

## Standards axis

Five findings:

1. **Hard — precision narrowing.** The new continuation API accepts generic
   `Real` inputs, then converts scales, state, and tolerances to `Float64`
   (implementation lines 225--273). This violates `AGENTS.md` section 3 and
   `cyaxiverse-julia-quality` section 3. A replay confirmed that a `BigFloat`
   scale produces a `Float64` ratio.
2. **Hard — abstract result container.** `Vector{NamedTuple}` at line 232 is an
   abstract element container in a numerical loop, contrary to the Julia
   quality instruction to prefer concrete internal containers.
3. **Hard — missing failure-boundary regression.** The input guards and solver
   controls have no failing-path test, contrary to the Julia quality
   requirement for regression coverage at the changed boundary.
4. **Judgement — Duplicated Code.** The N=5 constants and reduced-model logic
   duplicate the existing implementation in `reduced_models.jl` lines
   150--175. A shared implementation would make the comparison independent
   only if the test retains a separate source-derived oracle.
5. **Judgement — Speculative Generality.** `seed_theta` claims general branch
   selection, but the result model represents only `:pi`.

## Spec axis

Four findings:

1. Required failure-boundary coverage is missing. G0 acceptance 4 requires
   checks that “exercise a documented continuation failure boundary”; only
   successful paths are documented and tested.
2. Branch identity is hard-coded. G0 acceptance 2 requires branch identity in
   continuation state, but `seed_theta=0` still reports `branch=:pi`.
3. `converged` can remain false even after the final update reaches the stated
   residual tolerance. This violates the stationarity/convergence contract.
4. The required replay record is absent from the commit: no tracked G1 artifact
   records command, observed values, source identity, revision, and tolerances.

No scope creep was found. The N=8 benchmark remains on its prior convention and
is only decoupled from the repaired N=5 value.

## Exact checks and observed outcomes

1. `python3 scripts/agent_verify.py run -- julia --startup-file=no --project=. scripts/inflation_scale_continuation.jl --benchmarks-only`
   exited 0. It reported `k_c=1.7700681326109957`, ratio `0.25`, zero middle
   Hessian, gradients from `0` to `9.32e-18`, and `passed=true`. All five N=5
   records reported one iteration, meaning each accepted its initial/predicted
   state before a Newton correction.
2. `python3 scripts/agent_verify.py run -- julia --startup-file=no --project=. /tmp/issue148_g1_regression_check.jl`
   exited 0 with the worker-reported `theta=[pi,pi,pi]`, gradients near zero,
   and Hessians `[7.857066692962267e-4,0,-7.850898189895039e-4]`.
3. `python3 scripts/agent_verify.py run -- julia --startup-file=no --project=. /tmp/issue148_g1_independent_checks.jl`
   exited 0. It reproduced the source ratio to at worst
   `5.551115123125783e-17` on the checked Float64 points and exposed the
   supplied-`k_c`, branch-label, last-update convergence, precision, and
   abstract-container results above. Raw output:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_wqvtl2ad/run.stdout.log`.
4. `python3 scripts/agent_verify.py snapshot`,
   `python3 scripts/agent_verify.py diff-check`, and
   `git diff --check 62c5a6135de9ddc8208a1530f50640d666f33cfd...2a4e495ccdd838cc5b1e884fbac136115a7d433f`
   passed before this review artifact was added.
5. The full local package test was initially unavailable because its optional
   CairoMakie test dependency was absent. This routine environment limitation
   did not affect the first FAIL recommendation. Draft PR 149's first
   documentation build failed on G0 audit links; commit `f976500` later fixed
   those links. Fast tests were pending and the Full test suite was skipped at
   the time of the first review.

## Bounded correction target

Return this gate to the same implementation worker. Keep the source-fixed
formula and N=5-only scope. Add continuation-derived catastrophe localization
from a regular starting point; truthful intrinsic branch state; a documented
and tested failure boundary; a final-update convergence recheck; independent
off-critical source-ratio and minima-change tests; concrete result typing; and
a tracked replay record with source/revision/environment identity and justified
tolerances. Re-run focused checks and applicable CI before fresh independent
review.

## Re-review at corrected commit b35cb74

Re-review recommendation: **FAIL** at
`b35cb74781513601cc4079e37060bf75a3e39e0e` against the same accepted G0 base
`62c5a6135de9ddc8208a1530f50640d666f33cfd`.

The correction now estimates a catastrophe from a sign change between
continued samples whose grid omits the analytic `k_c`. It also corrects the
last-update convergence check, labels the zero seed as `:zero`, uses a concrete
continuation-step vector, adds off-critical ratio and minima checks, and tracks
the repair evidence. Four acceptance gaps remain; they are ordinary bounded
implementation/evidence corrections and require no scientific-owner decision.

### Re-review material findings

1. **FAIL — a detected event need not satisfy the declared Hessian
   tolerance.** `_n5_catastrophe_root` performs one linear interpolation of
   the endpoint Hessians (`src/paper_benchmarks/poly102_inflation.jl`, lines
   265--269), and lines 350--369 mark the result as detected without refining
   it or checking `abs(catastrophe_hessian) <= hessian_tolerance`. On the
   committed regression grid at `test/runtests.jl` lines 3394--3400, the event
   is at `1.7700681483189589`, only `1.57e-8` from the Float64 oracle, but its
   Hessian is `-1.2337005306228832e-8`, more than 100 times the declared
   `1e-10` Hessian tolerance. A coarser valid bracket still reported
   `catastrophe_detected=true` with `k` error `2.3565e-6` and Hessian
   `-1.8508e-6`. The tests assert only a `5e-7` `k` comparison and the analytic
   anchor theta; no event tolerance exists in the API, and no test checks the
   returned event Hessian. Recover the Hessian zero by a bounded root refiner
   to a declared event/bracket tolerance, and require its gradient and Hessian
   residuals in acceptance.

2. **FAIL — branch identity remains a seed-region label rather than the
   corrected branch's identity.** `_n5_validate_zero_phase_seed` labels every
   seed within `pi/3` of `pi` as `:pi` (lines 253--263). At
   `k=k_c-1e-3`, the distinct analytic satellite critical point
   `theta=acos(-1/(4a))=3.101964568393247` lies only `0.0396281` from `pi`.
   Supplying that exact stationary seed returns the same satellite point with
   `branch=:pi`. Exhaustive branch support is not required, but unsupported
   seeds must fail rather than acquire the source-fold identity. Validate the
   corrected state against the selected branch or restrict the seed contract
   enough to exclude the neighboring branches, then test this boundary.

3. **FAIL — the advertised precision path is incomplete and has a mixed-type
   error.** Homogeneous BigFloat `k_values` now produce a concrete
   `N5ReducedZeroPhaseContinuationStep{BigFloat}`, but
   `N5_REDUCED_CRITICAL_SCALE` remains a Float64 constant and the ratio is
   anchored to it (lines 87--89 and 195--206). At 256-bit precision the exact
   source formula gives
   `k_c=1.770068132610995629125239519150256394...`; the stored constant differs
   by `8.56e-17`, and `n5_reduced_ratio(k_c)` differs from the direct source
   ratio by `1.68e-17`, far above BigFloat roundoff. Also, Float64 `k_values`
   with a BigFloat `seed_theta` raise a `MethodError` constructing the typed
   step because the result type is selected from `k_values` alone (lines
   300--311). Compute the source expression in the promoted working type and
   include the seed/tolerance types in promotion, or explicitly restrict and
   document the accepted input types.

4. **FAIL — failure and reproducibility evidence remain partial, and the
   tolerance rationale does not describe the implemented gate.** The tracked
   `issue_148_g1_repair_evidence.md` records parent `f976500`, not corrected
   `b35cb74`, and two commands depend on untracked `/tmp` scripts (lines
   9--28). Its tolerance section says the `1e-10` controls were “selected” but
   supplies no roundoff, grid-spacing, or event-error argument; it omits the
   actual `5e-7` test and `5e-6` benchmark event tolerances and does not report
   event Hessians (lines 51--54). The tests cover invalid arguments and a
   successful one-iteration correction, but neither the docstring nor tests
   state and exercise a non-converged corrector result. Non-finite tolerances
   are also accepted: setting both tolerances to `Inf` marks the first regular
   sample as a detected catastrophe. Record the final revision and replayable
   commands, justify all four numerical tolerances, document non-convergence,
   and reject non-finite controls.

### Initial-finding resolution

| Initial finding at 2a4e495 | Re-review at b35cb74 |
|---|---|
| Analytic `k_c` supplied in grid | Partially resolved: sign-change event is estimated, but not refined to Hessian/event tolerance |
| Hard-coded `branch=:pi` | Partially resolved: zero branch is labeled, but nearby satellite branches are still mislabeled `:pi` |
| No failure test; final-update status bug | Final-update bug and basic input guards resolved; numerical non-convergence and non-finite tolerance contract remain |
| Exact-`pi`, at-`k_c` circular validation | Resolved for the Float64 path with a perturbed seed, off-critical ratio check, and two-to-one minima check |
| Evidence untracked/incomplete | Tracked, but final revision, replayable scripts, event residuals, and tolerance justification remain incomplete |
| Float64 narrowing and abstract result vector | Concrete vector resolved; homogeneous BigFloat type retained, but source precision and mixed-type behavior remain defective |

### Re-review standards axis

Three findings (two hard, one judgement call):

1. **Hard — reproducibility evidence identity/tolerances.** The evidence records
   `f976500`, two untracked `/tmp` scripts, and selected rather than justified
   tolerances. This violates `AGENTS.md` section 4 and
   `cyaxiverse-scientific-reproduction` sections 3 and 6.
2. **Hard — incomplete solver failure regression.** Tests cover invalid
   arguments and a successful one-iteration case, but not a non-converged
   corrector or its documented result contract. This violates
   `cyaxiverse-julia-quality` section 6 and `AGENTS.md` section 5.
3. **Judgement — Duplicated Code.** N=5 reduced-model helpers remain duplicated
   between `reduced_models.jl` and `poly102_inflation.jl`. Repetition of the
   source expression in a validation oracle is justified; duplicate production
   implementations still risk drift.

The original abstract-container finding is resolved. Homogeneous BigFloat
inputs retain their outer type, although the independent precision checks above
show that the underlying source constant is still Float64 and mixed inputs
fail. No N=8, units, phase, metric, persisted schema, or implicit physical
convention change was found.

### Re-review spec axis

Two findings:

1. **Partial — Hessian-zero tolerance is not enforced.** The committed
   regression event has Hessian about `-1.23e-8` while the requested
   `hessian_tolerance` is `1e-10`; event detection neither refines nor checks
   that residual.
2. **Partial — replay identity and tolerances are incomplete.** The evidence
   names the parent revision, depends on untracked scripts, and omits a
   justification for event-location and Hessian tolerances and a statement of
   effective arithmetic precision.

The supplied-`k_c`, final-update, Float64 off-critical oracle/minima, and
abstract-container findings are resolved. Seed labels now persist in state,
but the independent satellite-seed replay above shows that the branch-selection
boundary is not yet truthful. No scope creep was found.

### Re-review exact checks and outcomes

1. The benchmarks-only command from the tracked repair evidence exited 0 with
   `passed=true`, event `k=1.7700681326502656`, error `3.93e-11`, gradient
   `3.78e-27`, and Hessian `-3.08e-11`. Its fine symmetric bracket happens to
   meet `1e-10`; the implementation does not enforce that result.
2. `python3 scripts/agent_verify.py run -- env JULIA_DEPOT_PATH=/tmp/julia-depot:/Users/vmehta/.julia julia --startup-file=no --project=. /tmp/issue148_g1_rereview_checks.jl`
   exited 0 and produced the event, branch, precision, type, and failure results
   above. Raw output:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_183jehj6/run.stdout.log`.
3. The reported full-suite failure is pre-existing at the accepted base. The
   `phase_volume_detuning_scan.jl` blob is identical at base and corrected SHA
   (`cd861aa8a257ae36c9dadef6486449a0497c35a4`), and the base test contains the
   same `4pi^2 I` assertion. A direct replay returned `2pi^2 I`, not `4pi^2 I`.
   Raw output:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_dwjdviy8/run.stdout.log`.
   This unrelated existing failure does not change the G1 verdict.
4. `git diff --check 62c5a6135de9ddc8208a1530f50640d666f33cfd...b35cb74781513601cc4079e37060bf75a3e39e0e`
   passed. PR 149's documentation build passed; Fast tests were pending and the
   Full test suite was skipped when checked.

Return the same implementation worker a bounded correction: refine the
sign-changing Hessian event to declared tolerances; make branch selection
truthful at neighboring source branches; preserve real source precision and
mixed input types; and complete the failure/tolerance/replay evidence. Keep the
N=5 zero-phase scope and all established conventions. Do not begin G2 before a
fresh independent G1 review passes.

Re-review totals: Standards 3 findings, worst is incomplete reproducibility
identity/tolerance evidence; Spec 2 findings, worst is the unenforced
Hessian-zero tolerance.

## Final re-review at code commit 6014c4d

Final re-review recommendation: **FAIL** at code commit
`6014c4d9a909baa36f9b8e0e6e9a1930b4cb89b8`, with tracked evidence commit
`e880fd22ff46711826967191da90c69aaafcf6b6`, against accepted G0 base
`62c5a6135de9ddc8208a1530f50640d666f33cfd`.

The source formula, ordinary sign-change event refinement, mixed precision,
concrete result type, non-finite control checks, and tracked replay identity
are repaired. The replay grid omits the analytic critical scale and recovers
`k=1.7700681325633121`, with source-oracle error `4.768e-11`, gradient
`4.586e-27`, and Hessian `3.745e-11`. Independent 256-bit checks gave zero
observed error for both the direct source ratio and the typed source critical
scale. Three adversarial acceptance failures remain.

### Final material findings

1. **FAIL — the `:pi` branch identity remains a fixed seed-window label.**
   `_n5_validate_zero_phase_seed` assigns `:pi` to every seed within `0.01`
   radians of `pi` without checking which stationary branch the corrected
   state occupies (`src/paper_benchmarks/poly102_inflation.jl`, lines
   268--277). At `k=k_c-1e-5`, the analytic satellite
   `acos(-1/(4a(k)))` is only `0.0039633221` from `pi`. Supplying that exact
   satellite returns the same theta, `gradient=8.67e-19`, negative Hessian
   `-1.5708e-5`, `converged=true`, and `branch=:pi`. The farther satellite at
   `k_c-1e-3` is rejected only because it lies outside the chosen window;
   that regression (`test/runtests.jl`, lines 3441--3446) cannot establish an
   intrinsic identity at the branch collision. Canonicalize the selected
   regular branch or validate the corrected stationary state itself, then
   cover satellites arbitrarily close to the cusp.

2. **FAIL — a failed continuation can still certify a catastrophe.** Event
   detection uses the analytic branch anchor and endpoint Hessians but never
   requires either the preceding or current continuation step to have
   converged (`poly102_inflation.jl`, lines 437--468). For
   `[k_c-1e-3,k_c]`, `seed_theta=pi+1e-3`, `max_iterations=1`, and
   `gradient_tolerance=1e-16`, the two continuation statuses were
   `[false,true]`, the first gradient was `1.003e-9`, yet both records were
   marked `catastrophe_detected=true` through adjacent-record propagation.
   The committed non-convergence test uses `[0.1,1.0]` and therefore misses
   this near-event failure boundary (`test/runtests.jl`, lines 3432--3438).
   Require the continuation states that establish an event to be converged,
   and add this near-cusp failure regression.

3. **FAIL — two declared tolerance contracts have bypasses.** First, the
   endpoint paths in both the refiner and public continuation assign
   `scale_error=0` whenever an endpoint Hessian is within
   `hessian_tolerance`, without enforcing `event_scale_tolerance`
   (`poly102_inflation.jl`, lines 299--307 and 439--451). With an endpoint at
   `k_c+1e-11`, `hessian_tolerance=1e-10`, and
   `event_scale_tolerance=1e-14`, the implementation reports an event at that
   endpoint although its source-oracle scale error is `1.0000e-11`. Second,
   the docstring says `gradient_tolerance` applies to `|g|`, but convergence
   multiplies it by a Hessian/amplitude scale (lines 410--428). At `k=0.1`,
   a returned step was `converged=true` with gradient `1.3562e-10` for an
   advertised `1e-10` tolerance. Enforce the literal residual and event-scale
   contracts in every exit path, or explicitly define and test different
   tolerance semantics before accepting them.

The tracked regression suite does not contain those adversarial cases. Its
standalone replay also prints `invalid_satellite_seed_rejected=true`
unconditionally: the closure returns `false` both after an unexpected success
and after the catch path (`scripts/issue_148_g1_replay_checks.jl`, lines
17--25 and 108--113). The `@test_throws` regression remains genuine for the
single farther satellite, but the replay line is not independent evidence.

### Final Standards axis

Five findings (three hard, two judgement calls):

1. **Hard — branch identity remains heuristic near the cusp.** The fixed
   `0.01` seed region admits source satellites that approach `pi`; the test
   covers only a farther satellite. This conflicts with the scientific
   identity and reproducibility requirements.
2. **Hard — the replay's invalid-seed claim is vacuous.** The replay emits
   success for both rejection and unexpected acceptance, although the
   separate regression assertion is genuine.
3. **Hard — `n5_reduced_critical_points` still narrows generic high-precision
   inputs.** It constructs `Float64[0,pi]` at lines 224--233; a BigFloat input
   therefore returns `Vector{Float64}` theta values alongside a BigFloat
   ratio. The continuation and source-formula paths themselves now preserve
   mixed and high precision.
4. **Judgement — dead residue and a misspelled name remain.** The unused
   `N5_REDUCED_CRITICAL_SCALE`, unused `_n5_catastrophe_root`, and
   `N5_REDUCED_CATASROPHE_REFINE_ITERATIONS` remain at lines 88, 280--285,
   and 93/290.
5. **Judgement — production N5 reduced-model logic remains duplicated**
   between `reduced_models.jl` and `poly102_inflation.jl`. Independent oracle
   duplication in replay code is appropriate.

The evidence statement that there were “no public API edits” is imprecise:
G1 changes qualified N5 behavior and adds the continuation entry point. No
persisted schema, N8 behavior, units, phase, coordinate, metric, or physical
convention change was found.

### Final Spec axis

Two findings:

1. **FAIL — near-cusp satellite seeds still receive the regular `:pi`
   identity.** This violates G0's requirement to carry known regular branch
   identity in continuation state (`issue_148_g0_baseline_audit.md`, lines
   297--301).
2. **FAIL — catastrophe detection can succeed despite a failed continuation
   step.** This violates the required documented continuation failure boundary
   (`issue_148_g0_baseline_audit.md`, lines 305--306).

The source `k_c`, independent ratio oracle, ordinary bisection event,
off-critical minima change, final-update status, homogeneous and mixed
precision continuation, concrete typing, and tracked source/revision evidence
are resolved. No scope expansion was found.

### Final exact checks and outcomes

1. The tracked replay script exited 0 and reproduced the reported four-record
   continuation and event. Raw output:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_nefpd_f0/run.stdout.log`.
2. The tracked focused regression script exited 0 with 24/24 assertions.
   Raw output:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_olmcx1fv/run.stdout.log`.
3. Independent adversarial replay exited 0 and produced the branch, failed
   continuation event, endpoint-scale, absolute-gradient, precision, and
   non-finite-control results above. Script:
   `/tmp/issue148_g1_final_rereview_checks.jl`; raw output:
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_934vulah/run.stdout.log`.
4. `git diff --check
   62c5a6135de9ddc8208a1530f50640d666f33cfd...e880fd22ff46711826967191da90c69aaafcf6b6`
   passed. The worktree was clean before this review artifact was appended.
5. The broader audit's JET `i` findings are pre-existing N8 limitations:
   `reduced_models.jl` has no diff from G0, and the `poly102_inflation.jl`
   diff has no N8 functional hunk. The audit log is
   `/var/folders/jd/gst5c4ys0313mjw6knrpxy0m0000gp/T/agent_verify_e5hy0bcu/run.stdout.log`;
   its unrelated N8 result and subsequent EMFILE/Revise limitations do not
   alter this G1 verdict.

Return only these bounded N5 corrections to the implementation worker. Do not
begin G2 until a fresh independent G1 review passes.

Final re-review totals: Standards 5 findings; Spec 2 findings. The scientific
gate fails on intrinsic branch identity, failed-step event certification, and
literal tolerance enforcement.
