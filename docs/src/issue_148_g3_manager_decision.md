# Issue 148 G3 manager decision

**G3: PASS.** The manager accepts fresh Sol/xhigh independent review of code
`4abd9c31aba1d387bac27769730c4e0dc7d0c4a0`, evidence revision
`c6d2291e5c90c8890f48a9574343c14808a5a507`.

Authority: Issue #148 and owner clarification `5609698767`; approved P96
contract; prerequisite scientific audit `b562780`. The owner clarification
permits determining local fate rather than requiring a persisted cusp curve.

## Accepted result

One source-twelve, zero-phase, fixed-saxion N8 one-null degeneracy persists
away from the radial ray on the audited positive-alpha two-cycle slice
`t=sqrt(k)(t_ref+alpha*u)`, with
`u=(0,1,2,-1,1,1,1,1)`. Exact geometric reconstruction and the local rank
diagnostic establish independent non-radial control. The zero-phase cubic
cancellation was tested and is not protected along this control.

The traced segment contains 46 stored states: the seed and 45 accepted
predictor/corrector transitions. It reaches approximately
`alpha=1.434567377e-4`, `k=0.5006853945`, then stops at the imposed scale
guard. This is persistence to a numerical boundary, not physical termination.
The opposing seed failure does not establish absence of another branch.

All stored states retain one near-null canonical mode and positive transverse
spectrum; the minimum transverse eigenvalue is `0.0105283777`. The segment
minimum P96 metric eigenvalue is `1.2875523e-4`. Source geometry, curve/divisor
controls, volume, actions, coefficients, and metric are recomputed consistently.
Independent/chained 128/256-bit solves verify four exact rational off-ray
controls, and independent probes verify derivative signs, the full Hessian
factor, normalization, continuity, accounting, and invalid-state statuses.

## Evidence and acceptance basis

- `issue_148_g3_control_audit.md` and `audit_issue_148_g3_controls.jl`:
  exact source geometry, valid direction/interval, rank and cubic sensitivity.
- `issue_148_g3_repair_evidence.md`: final replay, exact commands, numerical
  outcomes, precision and controls at representative points.
- `issue_148_g3_final_independent_review_4abd9c3.md`: fresh independent PASS;
  G3 replay passed, prerequisite audit passed 79 assertions, independent probe
  passed 373 assertions, G1 replay and diff checks passed.

The earlier G3 FAIL review is revision-specific and superseded by this
decision. A documentation-only endpoint-versus-segment metric correction was
made after review; no reviewed implementation changed. The unused tangent
calculation noted by review is non-blocking and left outside this acceptance
change.

## Claim and integration boundaries

No scientific-owner decision is required. G0–G3 now have accepted evidence
and independent scientific review. The scientific gate objective of Issue #148
is satisfied at this bounded claim level. No G4, inflation, stabilized-moduli,
population, global-cone, negative-alpha, exhaustive-branch, or resolved G2 cusp
claim is authorized or established.

PR #149 remains draft and unmerged pending repository integration. Known
unrelated baseline package/audit failures remain separate from focused gate
acceptance; no broad green-CI or release-readiness claim is made. No package
version or persisted schema changed. G4 is not started.
