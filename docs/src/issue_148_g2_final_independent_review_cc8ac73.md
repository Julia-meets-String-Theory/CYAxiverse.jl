# Issue 148 G2 final independent review of `cc8ac73`

## Recommendation

**PASS.** The final repair closes the three defects that remained after review
of `5b8daff`: the reported merger now uses a strictly distinct minimum and
index-one saddle, the 256-bit event stage performs a genuine solve, and the
loose bordered point with approximately `4.08e-4` fixed-scale state error is
rejected. Independent probes also show that fixed-`k` continuation is adequate
for this radial gate near the singular Hessian: a factor-two step-size change
leaves the localized event stable at the `1e-12` scale, and strict exact-source
polishing keeps the event-bracket states on their fixed-scale roots.

This passes G2 at the deliberately narrow claim boundary of a source-twelve,
fixed-saxion, radial one-null degeneracy under P96. The unchanged projected
classifier remains `:unresolved`; this review does not promote the result to a
numerically classified quartic cusp. G0/G1 remain PASS, and G3 remains
unauthorized.

## Reviewed state and authority

- G2 base: `d1a0b709b69aa680b9bca4223739366c3bfdf8b6`.
- Earlier repaired candidate: `5b8daffd732ba307ed1980615b9f049a95b92c2c`.
- Final production/replay candidate:
  `cc8ac73668ac488a492dd16008c0b790a4e4ef3b`.
- Evidence HEAD: `62a88caf0b16a031b1732d574694def55bbb518e`.
  The only `cc8ac73..62a88ca` change is the final repair evidence document.
- Source identity: arXiv `2608.14780v1`, Appendix D, twelve-term Table 1;
  recorded PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
- Owner contract: P96, period-one GLSM `theta`, argument `2pi Q theta`, and
  `K_theta=M96/k^2`, with Eq.96/CYTools `M96` selected over the literal
  Eq.23 `2M96`. Raw-radian P96 uses `M96/[k^2(2pi)^2]`; unconverted A96 is
  retained only as a labeled author reproduction.
- Current Issue 148 body and owner comments through the recorded first G2 FAIL
  decision were read directly. They introduce no later change to the approved
  P96 or G2 acceptance contract.

## Acceptance assessment

| Requirement | Result | Independent evidence |
|---|---|---|
| Genuine intrinsic continuation near the singular Hessian | **PASS via demonstrated fixed-`k` adequacy** | The submitted trace honestly records 300 accepted `:fixed_k_fallback` steps. Independent runs with radial steps `5e-5` and `2.5e-5` both detected the event and gave `k=0.6745063700046396` and `0.6745063700038825`, respectively, only `1.286e-12` and `5.291e-13` from the independent Float64 augmented result. The two localizations differ by `7.57e-13`. No claim that a bordered point was accepted is needed. |
| Coordinates, bordered scaling, conditioning, and state fidelity | **PASS** | The bordered system consistently uses `y=k/k_arc_scale`. Submitted event-bracket matrices are rank 9 with conditions `5.539e6` and `5.023e6`. Independent factor-two step probes also found rank 9 and conditions from `4.93e6` to `2.45e7`. The old loose bordered state still has normalized residual `1.607e-9` but strict fixed-scale displacement `4.08248e-4`; default acceptance rejects it and returns a fixed-`k` state whose independent 256-bit polish displacement is `3.13e-15`. Event-bracket polish displacements were `1.56e-12`/`9.59e-12` for the `5e-5` run and approximately `6.05e-11` for the half-step run. |
| Strictly distinct minimum/saddle merger | **PASS** | Deterministic branches 105/113 start at common `k=0.66` with periodic separation `2.020937e-3`. Exact-source 256-bit correction at common `k=0.6745` leaves separation `4.364196e-5`, with Hessian inertias 0 and 1; they are not duplicate roots. Continuing the independently corrected pair to `k=0.674505`, `0.674506`, `0.6745063`, and `0.67450636` shrinks the separation through `2.0263e-5`, `1.0536e-5`, `4.5840e-6`, and `1.7331e-6`, while both solves converge and retain inertias 0/1. |
| Event versus independent augmented solve | **PASS** | The continuation localization is `0.6745063700046396`; the repository augmented solve is `0.6745063700033533`. The exact-source 256-bit augmented endpoint is `0.6745063700033668455...`, with raw stationarity and null residuals below `8e-100` in the submitted replay. The continuation bracket, rather than the augmented event, is used to seed the precision ladder. |
| Genuine 128/256 precision ladder | **PASS** | The submitted independent/chained 256-bit stages require 93/66 iterations and move `k` from the 128-bit result by `8.978e-26`. A separate replay from the existing augmented seed required 24/95/77 iterations at 128/independent-256/chained-256, moved `k` by `6.097e-26`, and gave independent/chained 256-bit agreement of `8.79e-53` in `k` and `2.11e-29` in periodic point distance. Its 256-bit gradient/null residuals were at or below `2.1e-100`. This excludes no-op widening. |
| Exact source construction and metric precision boundary | **PASS** | The BigFloat event path constructs integer charges, rational actions, exact zero phases, and target-typed `pi`. An independent check found exact equality with the repository source-twelve charge/action data. The P96 canonical diagnostic explicitly records `metric_source_precision_bits=53` and `metric_precision_boundary=:float64_reconstructed`; it does not claim recovered metric digits. |
| Source 12 versus author 10 and canonical tensor comparison | **PASS** | The source path contains 12 terms and the author path 10. Their scientific roles remain separate. The like-for-like ten-term P96/A96 comparison uses the same model, point, and metric witness and reproduces `(2pi)^2` for the projected second derivative and `(2pi)^4` for the fourth derivative. |
| Existing post-hoc matcher comparison | **PASS for the bounded sample** | A fresh replay invoked `_pilot_records`, `_pilot_init_branch_ids!`, and `pilot_match_records!` on independently corrected populations at `k=0.68` and `k=0.6795`. It produced 5 matches; all 5 propagated `branch_match_id` values agreed with the continuation identities at the corresponding scales. The legacy distances ranged from zero to `9.58e-3`, within its declared `0.05` tolerance. Matcher output is not used to assign continuation IDs. |
| Projected classification and claim boundary | **PASS** | The unchanged diagnostic reports one near-null direction, positive transverse eigenvalues, and `:unresolved` at both Float64 and exact-source 256-bit potential evaluation with the declared 53-bit metric boundary. The implementation does not alter cutoffs or claim that it established the paper's quartic-cusp classification. |
| Failure status, deduplication, and regression coverage | **PASS** | Disabled-corrector and zero-tolerance probes return `:fixed_k_fallback` and `:step_failed` rather than false completion. Regular seeds are polished to scaled residual below `1e-12` and deduplicated at periodic distance `1e-5`. The durable replay directly exercises off-branch rejection, distinct-pair correction, the real 128/256 event solves, matcher calls, normalization, and failure status. The reported 327 count includes 300 constant branch-ID checks, so this recommendation relies on the decisive numerical assertions and independent probes, not on that count. |
| G1, API/schema, optional dependencies, and scope | **PASS** | The G1 replay ends with `validation_checks_complete=true`. The G2 diff is additive, changes no persisted schema or package version, and does not add G3/off-ray work. Core loading in the focused replay does not require CYTools/Python. |

## Commands and observed outcomes

All Julia commands used Julia `1.12.6` with
`JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`.

1. `gh issue view 148 --repo Julia-meets-String-Theory/CYAxiverse.jl --comments --json body,comments,title,url`
   exited zero and supplied the current issue body and owner comments.
2. `julia --startup-file=no --project=. scripts/issue_148_g2_continuation_evidence.jl`
   exited zero. It reproduced 1/5 sampled minima above/below, rank-9 event
   brackets, continuation/augmented agreement `1.286e-12`, a real 128/256
   ladder, 5/5 matcher agreement, the strict minimum/saddle witness, and
   327/327 assertions.
3. Reviewer-local `/private/tmp/issue148_g2_final_reviewer_probe.jl` exited
   zero and produced the independent inertia/separation ladder, off-branch
   rejection measurement, and independent/chained precision results recorded
   above.
4. Reviewer-local `/private/tmp/issue148_g2_final_reviewer_probe2.jl` exited
   zero and reproduced the event with two fixed-`k` step sizes, exact-source
   event-bracket polishing, rank/condition diagnostics, and inertia crossing.
5. Reviewer-local `/private/tmp/issue148_g2_reviewer_matcher_probe.jl` exited
   zero against the final candidate and reported 5 matches with zero identity
   disagreements.
6. `julia --startup-file=no --project=. scripts/audit_issue_148_n8_metric_boundary.jl`
   exited zero and reproduced the precise/rounded metric distinction, `2pi`
   distance factor, `(2pi)^2` Hessian factor, and `(2pi)^3/(2pi)^4` projected
   derivative factors.
7. `julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl`
   exited zero with `validation_checks_complete=true`.
8. `python3 scripts/agent_verify.py diff-check` exited zero for the clean
   working tree. `git diff --check d1a0b70..62a88ca` identified one trailing
   blank line at EOF in `issue_148_g2_final_repair_evidence.md`. This is a
   trivial documentation-hygiene correction and does not alter the scientific
   or implementation acceptance recommendation.

No broad package-suite or release-readiness claim is made. The known unrelated
baseline package/audit failures were not rerun.

## Final claim boundary

G2 may claim a replayable radial source-twelve N8 one-null catastrophe under
P96, reached by intrinsic fixed-`k` branch chains with independently verified
minimum/saddle convergence and agreement with exact-source augmented solves.
It may not claim an exhaustive stationary-point population, a resolved
quartic-cusp label, stabilized moduli, successful inflation, off-ray validity,
or any G3 result.
