# Issue 148 G2 manager decision

Historical FAIL for the candidate identified below. Superseded for the final
repair by `issue_148_g2_final_manager_decision.md`; retained for traceability.

**G2: FAIL** on 2026-09-10. G0 and G1 remain accepted PASS. G3 must not
start. The approved P96 metric contract remains authoritative and unchanged.

The reviewed implementation is `9fe32eb0bebc4d9ad99fb761357c604674e3681a`;
candidate evidence HEAD is `0706b46df9e95b4cd0e53ae9718d2047ccc28dac`.
Fresh Sol/xhigh independent review is recorded in
`issue_148_g2_independent_review_9fe32eb.md`. Its evidence supersedes the
candidate's implied acceptance claims in `issue_148_g2_continuation_evidence.md`.

## Reasons for rejection

- Disabling the bordered corrector yields bit-for-bit identical accepted
  paths and event brackets. The demonstrated path uses fixed-k fallback;
  equivalent near-singular well-posedness and branch fidelity are not proven.
- The supposed post-hoc matcher comparison copies continuation points and
  identities instead of exercising the existing matcher.
- Both BigFloat continuation refinements hold k fixed. The independent
  256-bit augmented solve is useful, but does not supply the required
  128/256-bit event stability ladder.
- Canonical derivative comparisons mix twelve-term and ten-term models.
  The arbitrary-precision classification wrapper widens Float64 data.
- Failure/status semantics, conditioning records, tolerance justification,
  branch-merger evidence, and meaningful regression coverage are insufficient.

The reviewer independently reproduced the advertised 321 passing checks;
their count does not cure these gaps (300 repeat a copied branch-ID check).
The exact twelve-term source data, radial degeneracy agreement, 256-bit
augmented convergence, preserved G1 63/63, and no API/schema/G3 expansion
are useful passing evidence to retain.

## Scientific boundary and next bounded work

The existing projected diagnostic honestly reports `:unresolved`, including
the reviewer's independent exact-potential 256-bit probe. This is not evidence
for a quartic-cusp claim. No scientific-owner decision is needed to fix the
concrete implementation and validation defects first. Do not change derivative
cutoffs, normalization, or the claim boundary to force a label. If source cusp
classification still requires a different diagnostic or interpretation after
repair, return the smallest remaining question to the owner.

The next bounded task is **G2 IMPLEMENTATION repair**, using the same Luna/xhigh
worker (Opus reached its usage limit). Its acceptance is the six-item minimum
correction list in the independent review, plus the original G2 contract.
Preserve approved P96, source12/author10 identity, exact precision construction,
G1, and shared branch/worktree. Own normal correction/retest loops. Escalate
only genuine scientific/interface boundaries. Use a 25-minute initial lease
and the one-checkpoint/one-extension policy. Fresh Sol/xhigh independent review
is required again before any PASS decision. No G3 is authorized by this record.

## Orchestration and validation record

Opus standard-context CLI ran in safe mode and exhausted its session allowance
before completion. Luna/xhigh inherited the preserved edits and completed the
candidate. A fresh Sol/xhigh reviewer had no implementation ownership. The
manager accepts the evidence-based FAIL recommendation, not the worker's DONE
status as a gate decision.

Exact replay commands and observations are in the independent review. No
repository-wide green CI, release, population, or inflation claim is made.
PR #149 remains draft and unmerged; Issue #148 remains open.
