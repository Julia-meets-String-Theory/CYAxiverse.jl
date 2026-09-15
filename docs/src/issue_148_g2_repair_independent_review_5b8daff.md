# Issue 148 G2 repair independent review of `5b8daff`

## Recommendation

**FAIL.** The repair fixes several concrete defects from the rejected
`9fe32eb` candidate, and the radial degeneracy itself is numerically real.
However, two required scientific gates still do not hold: the asserted branch
merger is a duplicate-root artifact, and the reported 256-bit event stage does
not refine the 128-bit event. The replay's loose continuation tolerance also
accepts materially off-branch bordered points and is not protected by a
scale-aware branch-fidelity regression.

These are implementation and validation failures, not a current scientific
owner block. The unchanged classifier's `:unresolved` result is honest and is
acceptable for the explicit requirement to run and report the existing
projected diagnostic, provided G2 does not claim that it has numerically
established the paper's quartic-cusp label.

## Reviewed state and contract

- G2 base: `d1a0b709b69aa680b9bca4223739366c3bfdf8b6`.
- Rejected implementation used for the repair comparison:
  `9fe32eb0bebc4d9ad99fb761357c604674e3681a`.
- Repaired production/replay code:
  `5b8daffd732ba307ed1980615b9f049a95b92c2c`.
- Evidence HEAD inspected and tested:
  `e991495f1bb58edd7a7043dfc90771b6666c717c`.
- Current Issue 148 body and owner comments were read on 2026-09-10. G0 and
  G1 remain PASS; G3 is not authorized.
- Source identity: arXiv `2608.14780v1`, Appendix D Table 1, PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- Owner-approved P96: period-one GLSM `theta`, argument `2pi Q theta`, and
  `K_theta=M96/k^2`, where `M96` is the reconstructed Eq.96/CYTools benchmark
  matrix. Raw-radian P96 uses `M96/[k^2(2pi)^2]`; unconverted author A96 stays
  separately labeled. This review does not revisit that decision.
- Scope is the zero-phase, fixed-saxion, twelve-term radial N8 benchmark. No
  exhaustive branch, population, off-ray, inflation-success, or release claim
  was assessed.

## Acceptance assessment

| Requirement | Result | Independent evidence |
|---|---|---|
| Source twelve-term vs author ten-term identity | **PASS** | `_n8_exact_source_data(BigFloat)` equals the repository source `Q` and action vector exactly; it has 12 terms, while `author_inflation.N8_Q_TRAJECTORY` has 10. |
| P96 metric, coordinates, and provenance | **PASS with declared 53-bit metric boundary** | The metric audit exits zero and reproduces the P96/A96 tensor relations. `n8_bigfloat_p96_diagnostic` constructs the potential from integer/rational source data and explicitly reports `metric_source_precision_bits=53` and `:float64_reconstructed`; it does not claim a high-precision physical matrix. |
| Consistent bordered scaling and diagnostics | **PASS mechanically** | The repaired bordered column, correction, arclength row, residual, and trust norm consistently use `y=k/k_arc_scale`. Steps record method, rank, condition, and rejected line-search steps. The tested bordered matrices were rank 9. |
| Genuine continuation / fallback adequacy | **PARTIAL, insufficient candidate validation** | The advertised trace contains 8 bordered and 292 fixed-k fallback steps. Independent stricter fallback-only runs from a strictly corrected regular seed recover `k=0.6745063700036618` and `0.6745063700037428` with radial steps `5e-5` and `2.5e-5`, respectively, only `3.08e-13` and `3.89e-13` from the Float64 augmented result. This demonstrates that the local fixed-k method can be adequate. The submitted `1e-8` trace does not establish branch fidelity: its first accepted bordered point moves by `4.082e-4` when strictly corrected at the same `k`, despite being labeled converged. |
| Continuation event vs independent augmented solve | **PASS for existence/location** | The submitted trace localizes `k=0.6745063618037688`, `8.200e-9` from the independent Float64 augmented solve, with raw gradient `2.101e-37`. The stricter independent fallback probe improves the scale agreement to about `4e-13`. The augmented solve was not used to generate those continuation brackets. |
| Intrinsic branch identity and expected merger | **FAIL** | The replay's claimed merging pair begins only `1.1815e-6` apart, just above its `1e-6` deduplication cutoff. Exact-source 256-bit fixed-scale correction at the regular `k=0.68` seed collapses those two points to a separation of `4.20e-48`; they are the same root before continuation. Thus `1.576e-10` later separation is not merger evidence. Four other strictly distinct symmetry images remain about `0.49-0.50` apart. |
| Existing post-hoc matcher comparison | **PASS for the declared finite sample** | The replay actually invokes `_pilot_records`, `_pilot_init_branch_ids!`, and `pilot_match_records!` on fixed-scale solves at `k=0.68` and `0.6795`. Independent inspection reproduced 5 matches and compared all 5 propagated `branch_match_id` values with same-scale continuation identities; all agreed. Distances and residuals were finite, and no continuation record was copied into the adjacent population. |
| Float64 -> 128 -> 256 event ladder | **FAIL** | The 128-bit exact-source augmented solve is genuine. The 256-bit call starts from that result and exits on iteration 1 because the inherited residual is already below the precision-independent `1e-40` tolerance. Its `k` is exactly equal to the 128-bit `k`; displayed extra digits are widening/padding, not recovered source digits. A strict 256-bit solve changes `k` by `5.47e-26` and reduces gradient/null residuals from about `2.23e-62`/`9.11e-65` to about `1.18e-100`/`4.55e-103`. |
| Hidden narrowing / precision provenance | **FAIL for the event certificate; PASS for source construction and metric label** | Integer charges, rational actions, exact phases, and target-typed `pi` are sound. The no-op 256-bit event is nevertheless labeled `precision_bits=256` without recording that the recovered state came unchanged from the 128-bit seed. The canonical diagnostic correctly discloses its 53-bit metric source. |
| Projected higher-derivative diagnostic | **PASS at the stated claim boundary** | A strict exact-source 256-bit probe gives one near-null mode, positive transverse spectrum, and `:unresolved` under the unchanged cutoff: `D3=8.70e-46`, `D4=3.30e-70`, while the derivative cutoff is `1.69e-28`. The repair neither changes the cutoffs nor promotes this to a cusp. |
| Same-model P96/A96 tensor factors | **PASS** | The like-for-like ten-term nondegenerate probe uses the same point/model/metric witness and reproduces `(2pi)^2` for `D2` and `(2pi)^4` for `D4`. The source12 diagnostic remains separate. |
| Scale-aware tolerances and regression coverage | **FAIL** | The `1e-8` normalized continuation tolerance is justified in the evidence using a supposed event-near bordered condition of `1.1e10`, but the replay reads `best_cat.steps[end-1]`, far past the event; the actual bracket-step conditions are about `4.86e6` and `4.94e6`. More importantly, residual acceptance permits `4.08e-4` fixed-scale branch error and duplicate roots. The precision check accepts an exactly unchanged 256-bit state. The 321-test count still contains 300 caller-supplied constant-ID checks, while the new method/status/precision checks remain top-level in a standalone replay and do not catch either defect. |
| Failure/status semantics | **PASS mechanically, incomplete scientifically** | Disabled-corrector and zero-tolerance probes now distinguish `:fixed_k_fallback`, `:step_failed`, `:max_attempts`, and `:max_steps`; no tested result falsely reports `:completed`. They do not test strict branch correction/deduplication or reject a no-op higher-precision stage. |
| G1, API/schema, optional Python, and no G3 | **PASS** | The G1 replay reaches `validation_checks_complete=true`; the G2 diff is additive, changes no persisted schema/version, and contains no off-ray G3 work. Core import succeeds in the G2 replay without CYTools. |

## Decisive findings

### The merger check certifies a duplicate root

`n8_find_regular_branches` accepts a normalized gradient residual of `1e-10`
and deduplicates only below periodic distance `1e-6`. The merger regression
then requires a starting distance merely greater than that same cutoff and a
later distance below it. For the selected pair (branch IDs 3 and 5), strict
target-constructed correction at `k=0.68` moves one candidate by
`1.176e-6`, the other by `5.35e-9`, and leaves their corrected roots only
`4.20e-48` apart. This is one root represented twice because residual error
exceeds the deduplication margin.

The bounded data contain a viable route to genuine merger evidence. Four
below-side minimum chains have nearby index-one saddle chains: associated
minimum/saddle separations shrink from about `2.021e-3` at `k=0.66` to
`6.78e-5` at `k=0.67449095` after strict common-scale correction. The repair
must follow and identify such a genuinely distinct minimum/saddle pair to the
event, rather than thresholding duplicate above-side seeds.

### The default trace understates its branch error

The first four reported bordered points have normalized residuals from
`1.61e-9` to `8.61e-9`, so the replay accepts them under `1e-8`. Strict
fixed-scale correction moves the first of those points by `4.082e-4`, equal
to its reported step-scale angular displacement. The early trace is therefore
not a resolved stationary branch at the accuracy used to argue intrinsic
identity. Near the event, fallback points are much better: strict correction
moves them by about `2.42e-10`, and stricter fallback-only event recovery is
stable under a factor-two radial-step change. This supports retaining the
fixed-k fallback, but its tolerance and fidelity checks must be based on
state/conditioning error, not only the normalized residual.

The evidence also labels a `1.105e10` condition estimate as the accepted event
step by reading the penultimate trace element. The catastrophe bracket is at
indices 130-131, while the trace continues to 301 elements; the quoted step is
near `k=0.666`, not the event. The actual recorded bracket conditions are of
order `4.9e6`.

### The 256-bit result is a no-op widening of the 128-bit event

The repaired replay solves the exact augmented system at 128 bits, then passes
that state to the 256-bit call with the same absolute `1e-40` stopping rule.
The second call exits at iteration 1 and returns `k_256-k_128=0`. Reevaluating
the exact model at 256 bits is useful residual evidence, but it is not a
256-bit recovered event. With `1e-70`, the same solver performs a real
refinement; independent and chained strict 256-bit solves agree in `k` to
about `4.2e-51` and expose the missing change at order `1e-26`.

## `:unresolved` owner boundary

The current G2 clarification requires use of the existing projected
higher-derivative classifier and forbids changing its acceptance criteria to
obtain an expected label. The repaired code satisfies that instruction and
correctly returns `:unresolved`. On the present contract, that result does not
itself require owner action; it means the accepted claim must remain a
one-null radial degeneracy with positive transverse modes, not a reproduced
quartic-cusp classification.

If final G2 acceptance is intended to include the paper's quartic-cusp label,
then, after the mechanical failures above are corrected, the smallest owner
question is: **does numerical reproduction of the one-null radial degeneracy
with `:unresolved` under the unchanged repository classifier satisfy G2, or
must a separately approved scale/diagnostic establish the source cusp class?**
No cutoff or physical normalization should change without that decision.

## Minimum correction before fresh review

1. Replace the duplicate-root merger check with an associated, strictly
   corrected minimum/saddle pair. Record their identities and same-scale
   separation from regular seeds to the localized event.
2. Make continuation acceptance scale-aware enough that a converged state is
   demonstrably close to the fixed-scale root. Add a regression that would
   reject the observed `4.08e-4` branch error and duplicate seed population.
3. Run a genuine 256-bit augmented refinement, either independently from the
   Float64 event or with a precision-appropriate stopping rule. Record
   iteration count, state provenance, `k`, periodic point stability,
   gradient, null residual, null normalization, and Hessian diagnostics.
4. Assert that the higher-precision stage actually refines or independently
   reproduces the event; an unchanged widened seed must not pass as recovered
   256-bit digits.
5. Correct the event-step conditioning report and preserve the now-useful
   method/rank/status provenance, actual matcher comparison, metric boundary,
   same-model tensor checks, and unchanged `:unresolved` classification.
6. Put the decisive branch, precision, and failure checks in the maintained
   focused regression path rather than relying on the 300 constant-ID
   assertions as the reported test count.

## Commands and observed outcomes

- `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia
  --startup-file=no --project=. scripts/issue_148_g2_continuation_evidence.jl`
  exited 0 and reported 321/321 in the named testset, 8 bordered steps, 292
  fixed-k fallbacks, 5 matcher comparisons with 0 disagreements, and
  `:unresolved` classification.
- Reviewer probes at `/private/tmp/issue148_g2_reviewer_probe.jl`,
  `/private/tmp/issue148_g2_reviewer_probe2.jl`,
  `/private/tmp/issue148_g2_reviewer_probe3.jl`, and
  `/private/tmp/issue148_g2_reviewer_matcher_probe.jl` exited 0 after one
  corrected probe-script error. They produced the duplicate-root,
  strict-fallback, true-256-bit, below-side pair, and matcher observations
  quoted above. These scripts are reviewer-local; all decisive numerical
  outputs are recorded in this review.
- `julia --startup-file=no --project=.
  scripts/audit_issue_148_n8_metric_boundary.jl` exited 0 and reproduced the
  P96/A96 coordinate, metric, Hessian, and derivative factors.
- `julia --startup-file=no --project=.
  scripts/issue_148_g1_replay_checks.jl` exited 0 with
  `validation_checks_complete=true`.
- `python3 scripts/agent_verify.py diff-check` and `git diff --check` exited 0.

No broad package-suite result or release readiness is claimed; the known
unrelated baseline failures were not rerun.
