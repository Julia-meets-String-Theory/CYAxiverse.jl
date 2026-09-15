# Issue 148 G1 manager decision

**G1: PASS.** Recorded 2026-09-10 00:16 UTC (2026-09-09 America/New_York).
G0 remains PASS. G2 has not started. This is scientific gate acceptance, not a
claim that repository-wide CI or release checks are green.

## Acceptance and evidence

The manager accepts the fresh Sol/xhigh
[independent acceptance report](issue_148_g1_final_independent_acceptance_792a02f.md)
against [Issue 148](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148)
and its [owner clarification](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148#issuecomment-5609698767).

- Accepted G0: `62c5a6135de9ddc8208a1530f50640d666f33cfd`.
- Tested G1 code: `792a02fb77d22cefdfec8fe77969d9c719e7b6d0`.
- [Repair evidence](issue_148_g1_repair_evidence.md) at
  `668ef258dc8ad03046277a5973fe6ed1db1cd0e6`; source, scripts, and tests are
  identical to the tested code commit. Integration adds only documentation.
- Source N5 `k_c = (4/pi) log(1024/255) = 1.770068132610995629...`.
  The incorrect N8-value reuse is repaired.
- Predictor/corrector continuation carries the known regular zero-phase pi
  branch, validates identity, and localizes its Hessian-zero event without
  supplying the analytic event as a grid point. Default event error is
  `4.768e-11`, gradient `4.586e-27`, Hessian magnitude `3.745e-11`.
  Literal gradient, Hessian, and event-bracket tolerances are enforced.
- Fresh independent focused regression: **63/63 passed**. Tracked replay,
  adversarial branch/failure probes, source-oracle checks, and whitespace
  checks passed. Both 256-bit satellites at `k_c-1e-40` have zero coordinate
  and symmetry error at working precision; maximum gradient `1.17041909e-97`.
- Genuine strict 128/256-bit continuation reruns improved event scale error
  from `2.91038301e-31` to `3.38813179e-61`, with stable theta and independent
  gradient/Hessian agreement. Target-precision and rational/mixed inputs are
  verified. The analytic oracle makes interval certification unnecessary for G1.

Scope remains N5, zero phase, source-reduced, dimensionless radial control and
one known regular branch. No N8 production behavior, physical convention,
persisted schema, dependency, or package version changed. Version impact is
an additive N5 continuation capability plus a benchmark bug fix; no feature-
branch version bump. Historical failed reviews remain as revision-specific
evidence and are superseded for this code by the final acceptance report.

## Integration limitations

[Draft PR 149](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/pull/149)
remains draft and unmerged. Documentation CI passed at evidence commit
`668ef25`; fast CI run `34419409879` was pending at adjudication and the full
suite was skipped by the vmm-PR workflow. Earlier CI and local package tests
stop at the independently proven pre-existing phase-volume Hessian assertion
(`2pi^2 I` observed versus `4pi^2 I` expected). These are not 63 failed/passed
G1 assertions: the standalone G1 suite was run separately and passed.

The broader Julia audit reports existing N8 undefined-`i` JET findings and
Revise file-watcher limits. Source equivalence to G0 was independently checked.
Python-free import passed. These limitations are not silently waived for a
future merge/release and were not repaired outside G1's scope.

## Future owner constraints — not dispatched

The linked owner comment is authoritative for later gates:

- G2: resolve period-one versus raw-radian N8 metric normalization before
  canonical-distance, canonically normalized Hessian, or observable claims;
  use well-posed near-singular continuation (pseudo-arclength preferred, or
  demonstrate simpler-method adequacy); Float64 discovery then genuinely
  constructed BigFloat refinement; classify degeneracy with projected higher
  derivatives. Interval certification is optional and targeted.
- G3: determine the local discriminant's fate under a geometrically valid
  non-radial direction. Persistence, termination, splitting/unfolding, changed
  class, or additional null directions are admissible outcomes. First establish
  independent control with a local sensitivity/rank diagnostic, and test
  rather than assume persistence of the zero-phase cubic symmetry constraint.
  A persistent curve is not required for success.

Manager remains Astra/low. Implementation worker preference after Spark's
completed task is standard-context Opus4.6 through Claude Code **--safe-mode**
with an explicit bounded packet; then Luna/xhigh at Opus's usage limit. The
1M Opus option required usage credits and was not enabled. Fresh Sol/xhigh
retains independent scientific-review ownership.
