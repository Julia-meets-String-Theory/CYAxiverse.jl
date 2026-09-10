# Issue 148 G2 continuation repair evidence

This revision addresses the independent review of the rejected candidate
`9fe32eb`. The repaired implementation and replay script were tested at
commit `5b8daffd732ba307ed1980615b9f049a95b92c2c` with Julia `1.12.6`
(`arm64-apple-darwin24.0.0`). The source is arXiv `2608.14780v1`,
Appendix D Table 1, PDF SHA-256
`b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
The earlier rejected evidence remains historical; this document is the
revision-specific record.

## Replay command

```text
JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. \
  scripts/issue_148_g2_continuation_evidence.jl
```

Exit status was `0`. Output is retained at
`/private/tmp/issue148-g2-repair-5b8daff.log`. The named regression testsets
reported `321/321`, with additional top-level assertions for diagnostic,
provenance, matcher, precision, tensor-factor, and failure-boundary checks.

## Continuation and failure provenance

The bordered equations now use one transformed radial coordinate consistently:
the Jacobian column, correction update, arc row, arc residual, and trust norm
all use `k/k_arc_scale`. Each step records `corrector_method`,
`bordered_rank`, `bordered_condition`, and `rejected_steps`.

For the selected above-side event chain, the accepted trace contains 8
bordered-corrector steps and 292 fixed-k fallback steps after the initial
state. The recorded bordered rank is 9 and its event-near condition estimate
is approximately `1.105e10`. The fallback is therefore explicit and bounded,
rather than hidden. A replay with `max_corrector_iterations=0` takes a
different first accepted scale (`0.67995` versus the normal trace), and the
step is recorded as `:fixed_k_fallback`. A zero-tolerance bounded probe
records `:step_failed` and a nonconverged step. Exhausted below-side attempts
record `:max_attempts`; no run reports `:running` or falsely reports
`:completed`.

The continuation starts from regular fixed-scale roots at `k=0.68` and
`k=0.66`. Five above-side and four below-side chains detect a sign-changing
event bracket. The selected repaired trace localizes
`k_cont=0.6745063618037688`, with bracket width `7.451e-13`, gradient
residual `7.134e-13`, and canonical minimum `1.765e-36`. The independent
Float64 augmented solve gives `0.6745063700033533`; the difference is
`8.200e-9`. The target-precision augmented event solve below gives the
higher-precision event scale. This Float64 conditioned-bracket error is
reported rather than hidden.

Two distinct sampled event chains have periodic separation
`1.576e-10` at the event and `1.182e-6` at their regular `k=0.68`
seeds. This is finite branch-merger evidence; no exhaustive population claim
is made.

## Actual post-hoc matcher comparison

The replay includes the existing script-local
`pilot_match_records!`. It independently solves bounded records from 13
regular roots at `k=0.68` and 24 fresh regular roots at `k=0.6795`, then
runs the old greedy periodic matcher with `matching_tolerance=0.05`.
It found 5 matches and 5 intrinsic-identity comparisons, with 0 disagreements.
The comparison uses the matcher's propagated `branch_match_id` and maps each
independent corrected record to the nearest continuation chain at the same
scale. Every observed disagreement would be printed; none occurred.

## Target-constructed precision ladder

The 128-bit and 256-bit event stages use the exact integer charge matrix,
rational action list, exact zero phases, and target-typed `pi` inside the
augmented stationarity/null-vector solve. The 128-bit event gives
`k=0.6745063700033668455292817484356833211648`,
gradient residual `2.764e-62`, null residual `6.243e-64`, and unit-null
residual below `1e-35`. The 256-bit event gives the same scale through the
128-bit digits and adds target digits,
`k=0.6745063700033668455292817484356833211647994685850690651965...`,
gradient residual `2.229e-62`, and null residual `9.112e-65`; the scale
difference is below `1e-30`. Both convergence results are asserted
unconditionally.

The exact-source 256-bit P96 diagnostic reports `:unresolved` and stationary
true. Its canonical metric records `metric_source_precision_bits=53` and
`metric_precision_boundary=:float64_reconstructed`; no high-precision
canonical metric claim is made. The source potential and derivatives retain
target-precision exact data.

## Higher derivatives and normalization

The source12 P96 diagnostic reports one near-null eigenvalue, positive
transverse eigenvalues, and `:unresolved` under the unchanged existing
derivative cutoff. The A96 path remains separately labeled. A like-for-like
ten-term probe at the same author point and common metric witness verifies
`D2_P96/D2_A96=(2pi)^2=39.478418` and
`D4_P96/D4_A96=(2pi)^4=1558.545457`. The source12 twelve-term and author10
ten-term paths are asserted as distinct.

The approved P96 metric is `M96/k^2` in period-one GLSM coordinates with
argument `2pi Q theta`. The reconstructed metric agrees with
`n8_kinetic_matrix(k)` to zero at Float64 replay precision, and its published
Eq.96 spectrum agrees within 2%. The metric's 53-bit source boundary is
carried into the target-precision diagnostic result.

## Scope and preserved boundaries

This revision covers radial N8 G2 continuation only. It does not change the
classifier cutoff, promote the result to a cusp claim, add G3 work, alter N5,
change public schemas, or repair unrelated baseline failures. G1 remains
preserved.
