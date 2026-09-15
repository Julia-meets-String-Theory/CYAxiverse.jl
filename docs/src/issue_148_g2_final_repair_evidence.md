# Issue 148 G2 final repair evidence

This record supersedes the rejected repair evidence in
`issue_148_g2_repair_evidence.md` and the pause record
`issue_148_g2_repair_pause.md`. It records the bounded implementation repair
after independent review `5b8daff`.

## Tested revision and command

- Branch: `issue-148-catastrophe-continuation`
- Tested commit: `cc8ac73` (`fix: enforce strict G2 branch fidelity and precision ladder`)
- Julia: 1.12.6, arm64-apple-darwin24.0.0
- Replay:
  `JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia julia --startup-file=no --project=. scripts/issue_148_g2_continuation_evidence.jl`
- Outcome: exit 0; focused regression testset passed 327/327.

The replay also ran `scripts/issue_148_g1_replay_checks.jl` and
`scripts/audit_issue_148_n8_metric_boundary.jl`; both exited 0. The known broad
baseline suite was not used as a G2 gate.

## Corrections verified

`N8ContinuationStep` now records the fixed-k branch state error in addition to
corrector method, bordered rank and condition, rejected line searches, and
failure status. A bordered candidate is accepted only after strict fixed-k
polishing at the same radial scale and a `1e-6` state-error bound. The replay
records 300 strict fixed-k fallback steps, while the bordered systems at the
actual event bracket (steps 110 and 111) have rank 9 and condition estimates
`5.539e6` and `5.023e6`. A bounded loose-candidate regression reproduces the
old first bordered displacement `4.082e-4` and rejects it; accepted steps have
state error at most `1e-6`. Disabled-corrector and zero-tolerance probes report
`:fixed_k_fallback` and `:step_failed` respectively.

Regular seeds are strictly polished before classification and deduplicated at
periodic distance `1e-5`. The focused regression checks strict residuals and
the minimum pairwise distance on an independently searched sample.

The below-side merger witness is selected from independently continued
minimum and index-one saddle chains at a common scale, then corrected
independently with exact-source 256-bit arithmetic. The selected pair is branch
IDs 105 and 113 at `k=0.67450000`: separation is `2.021e-3` at the regular
seeds and `4.364e-5` after the common-scale correction. This replaces the
previous duplicate-root witness.

The event ladder uses exact integer/rational source data at 128 bits, then
independent and chained 256-bit augmented solves with `1e-70` stopping. The
256-bit stages take 93 and 66 iterations, respectively; the chained stage
changes `k` from the 128-bit result by `8.978e-26`, and independent/chained
256-bit `k` values agree within `1e-45`. The endpoint residuals are
`|gradient|=2.373e-100` and `|Hv|=7.986e-100`. The P96 canonical diagnostic
continues to disclose its 53-bit reconstructed metric boundary, and the
projected classifier remains conservatively `:unresolved` under unchanged
cutoffs.

The actual post-hoc matcher is run on fresh same-scale bounded samples: 5
records are compared and all 5 `branch_match_id` identities agree with the
continuation identities. The source12 and author10 paths remain separate, and
the like-for-like P96/A96 tensor factors are `(2pi)^2` and `(2pi)^4`.
