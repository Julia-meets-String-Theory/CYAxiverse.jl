# Issue 148 G2 independent scientific review of `9fe32eb`

## Recommendation

**FAIL.** The implementation reproduces the known radial scale numerically and
preserves the twelve-term source data, but the complete G2 acceptance contract
is not met. In particular, the accepted event trace does not exercise the
bordered pseudo-arclength corrector, the matcher comparison does not invoke the
existing post-hoc matcher, and the stated precision ladder does not refine the
continuation-derived event scale. Several focused checks accept these gaps or
mask their failure paths.

This is an implementation and validation failure, not presently a scientific
owner block. The `:unresolved` higher-derivative result is discussed separately
below because it may expose a later claim-boundary decision, but it does not
prevent correcting the concrete G2 defects first.

## Reviewed state and claim boundary

- Review worktree HEAD: `0706b46df9e95b4cd0e53ae9718d2047ccc28dac`.
- Tested implementation and replay SHA: `9fe32eb0bebc4d9ad99fb761357c604674e3681a`.
  The only `9fe32eb..0706b46` change is
  `docs/src/issue_148_g2_continuation_evidence.md`; production and replay code
  are identical.
- G2 base: `d1a0b709b69aa680b9bca4223739366c3bfdf8b6`.
- Source: arXiv `2608.14780v1`, local PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- Owner-approved convention: P96, period-one GLSM `theta`, argument
  `2pi Q theta`, and `K_theta=M96/k^2`; raw-radian P96 uses
  `M96/[k^2(2pi)^2]`. `M96` is the Eq.96/CYTools matrix, deliberately selected
  over literal Eq.23's `2M96`. A96 remains a separately labeled author
  reproduction.
- Scientific scope: the twelve-term, zero-phase, fixed-saxion radial N8
  benchmark only. No exhaustive population, off-ray G3, physical inflation,
  schema, or release claim was reviewed.

The source text independently confirms Appendix D, the twelve Table-1 actions
and charge rows, five minima for `k<kc`, one minimum for `k>kc`,
`kc=0.674506370003365`, and the paper's description of the N8 result as the
same local quartic-cusp structure as N5. The implementation's exact-data helper
matches the repository twelve-term `Q` and action vector exactly. The author
trajectory is the first ten of those twelve terms, as documented.

## Acceptance assessment

| G2 requirement | Result | Evidence |
|---|---|---|
| Source 12 vs author 10 identity | **PASS** | `_n8_exact_source_data(Float64)` equals `_n8_potential().Q/qdotτ`; the author path is the ten-term prefix. The PDF hash and Table 1 were independently checked. |
| P96 metric and coordinate contract | **PARTIAL/PASS at Float64** | The metric audit exits zero and reproduces the P96/A96 tensor factors. Float64 continuation canonical Hessians use the precise repository `M96/k^2`. The arbitrary-precision classification wrapper still widens Float64 potential/metric data. |
| Genuine intrinsic continuation and near-singular well-posedness | **FAIL** | The accepted event path is bit-for-bit unchanged when `max_corrector_iterations=0`. Every observed `k` increment is the fixed-`k` fallback step `5e-5`; the pseudo predictor proposes about `3.12e-10`. No accepted point is shown to come from the bordered corrector. |
| Bordered rank/conditioning and failure semantics | **FAIL** | Result records contain no bordered rank, singular value, condition number, or fallback-used field. The replay prints canonical Hessian values under a “bordered conditioning” heading. Several below-side runs exhaust 300 attempts with only 291 accepted states and return `:completed` at `k=0.6745`, without recording the rejected attempts. |
| Continuation-derived event vs independent augmented solve | **PARTIAL** | Fixed-`k` seeded correction plus Hessian-sign bisection gives `k=0.6745063700038947`, within `5.413e-13` of the existing augmented solve. This is good numerical degeneracy evidence, but it is not the claimed pseudo-arclength recovery and its localization residual is the scalar-normalized residual unless separately labeled. |
| Branch identity and merger behavior | **FAIL** | IDs are caller-supplied integers copied into every state; the regression checks only that the copied integer stays constant. No fallback/jump diagnostic exists. The reported `3.886e-16` event-chain separation comes from a pair already only `1.18e-6` apart at the start, just above the `1e-6` deduplication threshold, while other event chains remain about `0.49-0.50` apart. This does not establish the expected branch merger. |
| Existing post-hoc matcher comparison | **FAIL** | The replay builds “matcher records” by copying 25 continuation points and assigning the same branch ID. It never runs `pilot_match_records!`. The comparison reads `seed_index`, whereas the old matcher propagates `branch_match_id`, and it does not restrict comparisons to matching scale slices. Zero disagreements are therefore predetermined. |
| Float64 to BigFloat precision/stability ladder | **FAIL** | `n8_bigfloat_continuation_refine` holds `k` fixed. Both 128- and 256-bit results retain the Float64-localized `k`, have identical `5.413e-13` scale error, identical `6.185e-45` gradient, and identical Hessian minimum. The independent 256-bit augmented solve is valuable and recovers new digits, but no 128/256-bit augmented/event stability comparison is made. Its convergence checks are inside `if aug_big.converged`, so a returned nonconvergence does not fail the replay. |
| Hidden Float64 narrowing audit | **FAIL** | Charges, rational actions, zero phases, and typed pi are sound in the two BigFloat solvers. However, `n8_continuation_classify(...; precision_bits>53)` first calls Float64-narrowing `_n8_potential` and `n8_kinetic_matrix`, then widens those values. No target-precision exact reconstruction or explicit 53-bit source bound is carried by this result. |
| P96 higher derivatives and catastrophe classification | **PARTIAL / unresolved claim boundary** | The replay reports one null mode and positive transverse modes, but returns `:unresolved`. It classifies the Float64 independent augmented point, not the continuation-localized point. Its test permits any of `:cusp`, `:fold`, or `:unresolved`. A direct 256-bit exact-potential evaluation also remains `:unresolved` under the unchanged `1e-8` cutoff, so the worker is correct not to promote a cusp/fold label. This does not reproduce the paper's quartic-cusp interpretation under the current classifier. |
| Canonical comparison and normalization labels | **FAIL** | The replay labels P96/A96 second/fourth ratios as expected `(2pi)^2/(2pi)^4`, but compares the twelve-term P96 point with the ten-term author A96 diagnostic. The observed ratios are `-67.967699` and `572.482884`, not `39.478418` and `1558.545457`; only finiteness of the fourth ratio is asserted. The evidence document omits this disagreement. |
| Declared, scale-aware tolerances | **FAIL** | Numerical thresholds appear in code (`1e-10`, `1e-12`, `1e-8`, `1e-40`, `1e-6`, `1e-4`) without a scale/conditioning derivation in the evidence. The `<1e-8` raw-gradient check is extremely loose relative to the approximately `1e-27` potential scale. Preconditioned and raw residuals are not consistently distinguished in the narrative. |
| Focused regressions and compatibility | **FAIL for G2 coverage; PASS for preserved G1/API/schema scope** | The G2 checks live only in the replay script, not `test/runtests.jl`. The reported 321 checks include 300 repetitions of the constant branch-ID assertion. There is no genuine corrector, matcher, nonconvergence, fallback, invalid-input, or false-status boundary regression. G1's 63 checks still pass; the diff is additive, changes no persisted schema/version, and contains no G3 work. |

## Major findings

### 1. The accepted path is not produced by the bordered corrector

At the selected above-side branch, the initial tangent has
`tangent_k=6.234480288307823e-7`. With `ds=-5e-4`, the pseudo-arclength
predictor therefore proposes `delta_k=-3.1172401441539117e-10`. The next
accepted state instead has `delta_k=-4.999999999999449e-5`, exactly the
fallback cap in `n8_continuation.jl:316`.

More decisively, a 120-attempt replay with the normal corrector and with
`max_corrector_iterations=0` produced identical lengths, all coordinates,
all `k` values, all monitored eigenvalues, status `:catastrophe_detected`, and
bracket `(110,111)`. Thus the focused event evidence is entirely insensitive
to the pseudo-arclength corrector.

The bordered equations also use the scaled radial coordinate inconsistently:
the top-right Jacobian column is `F_k*k_arc_scale`, but the returned last
correction is added directly to `k`. The arc row is likewise written with a
different scaling convention. At the first predictor the independently formed
matrix is full rank (`rank=9`) but has `cond=1.116e11`; the evidence neither
records this nor distinguishes an ill-conditioned bordered step from fallback.

Sequential fixed-`k` correction can be a legitimate simpler continuation
method if its adequacy and limits are demonstrated. Here it is described as a
near-fold switch even though it is active from the first step, and it has no
reported method provenance or jump/failure record. It therefore does not meet
the approved “pseudo-arclength or demonstrated equivalent well-posedness”
condition.

### 2. The post-hoc comparison is synthetic and cannot detect disagreement

The existing machinery performs independent correction at adjacent slices and
then calls `pilot_match_records!`, propagating string-valued
`branch_match_id`s through greedy periodic-distance matches. The G2 replay
instead creates each matcher record from a continuation state itself and sets
`seed_index=best.branch_id`. The helper compares those copied seed integers to
the same continuation integer. It neither evaluates independently corrected
slice populations nor uses the old matcher's assigned identity. The claimed
zero-disagreement result is consequently not evidence for requirement 7.

### 3. The precision ladder certifies a fixed-`k` stationary correction, not
the event

The exact source construction in the BigFloat functions is a real improvement:
integer charges, rational actions, exact zero phases, and target-typed pi are
used. The new 256-bit augmented solve converges to
`k=0.6745063700033669`, with raw gradient `8.435e-71` and null residual
`3.449e-68`, independently supporting the radial degeneracy.

However, the function named `n8_bigfloat_continuation_refine` never updates
`k`. The replay's 128- and 256-bit stages therefore repeat the same
Float64-derived event scale and report identical results. A genuine event
ladder needs target-constructed solves at both precisions, comparison of `k`,
the periodic point, raw/preconditioned stationarity, null residual, and the
relevant canonical diagnostics, with the metric's 53-bit source bound or an
exact reconstruction stated explicitly.

### 4. `:unresolved` is honest, but it does not establish a cusp class

The Float64 P96 diagnostic reports `D2=-8.91e-38`, `D3=-6.16e-29`,
`D4=9.29e-34`, one near-null mode, and positive transverse modes. Under its
single cutoff, all projected higher derivatives remain below threshold, so
`:unresolved` is the truthful existing-classifier result.

An independent 256-bit evaluation using the exact twelve-term potential and
the new augmented point also returns `:unresolved`: canonical gradient
`1.87e-69`, null eigenvalue `-3.70e-65`, `D3=-2.25e-40`,
`D4=2.48e-59`, and derivative cutoff `1.69e-28` at the unchanged `1e-8`
relative criterion. This shows that the label is not merely Float64 noise.

Therefore `:unresolved` can satisfy the narrow instruction to run and report
the current diagnostic, but it cannot support a positive claim that the G2
implementation classified or reproduced the source's quartic-cusp normal
form. The paper explicitly calls the N8 structure a local quartic cusp. After
the concrete continuation and validation failures are corrected, the manager
should either keep G2's claim at “radial degeneracy with unresolved projected
class” or obtain owner direction on whether effective transverse relaxation or
another already-approved diagnostic is required for the source cusp claim.

## Commands and observed outcomes

All Julia commands used Julia `1.12.6` with
`JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`.

1. `gh issue view 148 --comments --json ...` — exit 0 with escalated network;
   current issue body and owner comments through P96 approval were inspected.
2. `shasum -a 256 /private/tmp/2608.14780v1.pdf` — exit 0; hash matched
   `b0f5539...d2e6`.
3. `julia --startup-file=no --project=. scripts/issue_148_g2_continuation_evidence.jl`
   — exit 0, reported 321/321; reproduced `k=0.6745063700038947`, but also
   reproduced the false matcher comparison, identical fixed-`k` precision
   stages, mismatched P96/A96 derivative ratios, and `:unresolved` class.
4. `julia --startup-file=no --project=. scripts/audit_issue_148_n8_metric_boundary.jl`
   — exit 0; distance ratio `2pi`, Hessian ratio `(2pi)^2`, and projected
   derivative ratios `(2pi)^3/(2pi)^4` passed for like-for-like ten-term data.
5. `julia --startup-file=no --project=. scripts/issue_148_g1_n5_regression_tests.jl`
   — exit 0, 63/63.
6. Independent exact-source probe — exit 0;
   `source12_q_exact=true`, `actions_exact=true`, author term count 10 and
   source term count 12.
7. Normal versus disabled-corrector probes — exit 0. For 10 attempts all
   states were identical; for 120 attempts status, bracket, all coordinates,
   scales, and eigenvalues were exactly identical.
8. Independent branch-separation probe — exit 0. Event branches 3 and 5 had
   periodic separation `1.1815e-6` at `k=0.68` and roundoff separation at the
   end; other event-chain pairs remained `0.49-0.50` apart.
9. Independent bordered diagnostic — exit 0. At the first predictor,
   `rank(J)=8`, `rank(B)=9`, `cond(B)=1.116e11`, and
   `sigma_min(B)=3.537e-10`.
10. Independent exact-potential 256-bit classification probe — exit 0;
    one null mode, positive transverse modes, and classification
    `:unresolved` under the existing cutoff.
11. `python3 scripts/agent_verify.py diff-check` and
    `git diff --check d1a0b70..0706b46` — exit 0; no whitespace errors. The
    implementation worktree was clean before this untracked review artifact
    was added.

## Minimum correction before re-review

1. Make the bordered system use one consistent radial coordinate and prove the
   accepted event path actually uses it, or explicitly present fixed-`k`
   continuation as the baseline and demonstrate its well-posedness, branch
   fidelity, and stopping/failure boundary near the singularity.
2. Record corrector/fallback provenance, bordered rank/conditioning, rejected
   steps, and terminal reason. Correct `:completed` so exhausted or stalled
   attempts cannot masquerade as completion.
3. Run the actual old adjacent-slice periodic matcher on an independently
   solved bounded sample at the same scales, compare `branch_match_id` against
   intrinsic continuation identities, and report every disagreement.
4. Replace the fixed-`k` 128/256 “event ladder” with genuine target-precision
   event or augmented solves at both precisions; assert nonconvergence and
   stability outcomes unconditionally. State or reconstruct the P96 metric
   precision boundary for canonical results.
5. Compare P96 and A96 only with identical term sets and corresponding
   coordinates, assert the transformation factors, and preserve the separate
   source12-versus-author10 comparison without calling it a normalization
   ratio.
6. Add focused regressions for the real corrector, actual matcher,
   nonconvergence/fallback/status boundaries, and scale-aware residual and
   derivative tolerances. Keep the final `:unresolved` claim conservative or
   return its scientific interpretation to the owner after the mechanical
   gates pass.
