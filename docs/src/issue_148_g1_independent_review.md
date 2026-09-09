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
5. The full local package test remains unavailable because its optional
   CairoMakie test dependency is absent. This routine environment limitation
   does not affect the FAIL recommendation. Draft PR 149 had Fast tests and
   build pending, with the Full test suite skipped, when checked.

## Bounded correction target

Return this gate to the same implementation worker. Keep the source-fixed
formula and N=5-only scope. Add continuation-derived catastrophe localization
from a regular starting point; truthful intrinsic branch state; a documented
and tested failure boundary; a final-update convergence recheck; independent
off-critical source-ratio and minima-change tests; concrete result typing; and
a tracked replay record with source/revision/environment identity and justified
tolerances. Re-run focused checks and applicable CI before fresh independent
review.
