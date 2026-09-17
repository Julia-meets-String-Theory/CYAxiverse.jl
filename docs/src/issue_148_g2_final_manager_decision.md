# Issue 148 G2 final manager decision

**G2: PASS**, following fresh Sol/xhigh independent review of implementation
`cc8ac73668ac488a492dd16008c0b790a4e4ef3b`, evidence HEAD
`62a88caf0b16a031b1732d574694def55bbb518e`.

The acceptance source remains Issue #148 and its owner clarification. The
approved P96 metric contract is unchanged. This decision supersedes the
earlier G2 FAIL decisions for their explicitly identified older revisions.
G0/G1 remain PASS. **Do not begin G3 in this continuation.**

## Accepted evidence

The authoritative independent assessment is
`issue_148_g2_final_independent_review_cc8ac73.md`; the implementation replay
is documented in `issue_148_g2_final_repair_evidence.md`.

- Intrinsic fixed-k continuation is accepted through independently demonstrated
  adequacy near the singular Hessian, not a claim of successful pseudo-arclength
  steps. Halving the radial step changes the localized event by `7.57e-13`.
  Event-bracket bordered matrices have rank nine and recorded conditioning;
  independent exact-source polishing verifies branch fidelity.
- A strictly distinct minimum/index-one saddle pair retains inertias 0/1 as
  its separation shrinks from `2.021e-3` to `1.733e-6` toward the event.
  The former duplicate-root and off-branch acceptance defects are corrected.
- Continuation gives `k=0.6745063700046396`; the existing Float64 augmented
  solve gives `0.6745063700033533`. Genuine source-constructed 128/256-bit
  augmented refinements recover `0.6745063700033668455...`, with nontrivial
  iterations, independent/chained agreement, and residuals below `8e-100`
  in the submitted replay. The ladder no longer passes a widened seed as a
  newly refined event.
- The actual legacy matcher agrees in all five bounded comparisons and does
  not determine intrinsic continuation identity. Same-model P96/A96 tensor
  factors, source12/author10 separation, failure statuses, and G1 replay pass.
- The canonical metric's 53-bit source boundary is explicit. BigFloat
  potential/event precision is not represented as recovered metric precision.

The reported 327 checks include repeated identity assertions. Acceptance rests
on the decisive regressions and independent numerical probes, not that count.
Exact commands and observations are preserved in the independent review.

## Scientific claim boundary

The accepted result is the bounded zero-phase, fixed-saxion, source-twelve
radial N8 one-null degeneracy under P96. The existing projected diagnostic
reports positive transverse modes and `:unresolved`; its acceptance criteria
were not changed. Reporting that outcome satisfies the current diagnostic
requirement. **No numerically resolved quartic-cusp label is accepted.**

No owner decision is currently required. Any future proposal to change the
diagnostic/cutoff or promote a cusp, observable, inflation, or off-ray claim
must respect the existing scientific-owner boundary. This result does not
establish an exhaustive population, stabilized moduli, or G3 validity.

## Integration and verification scope

Luna/xhigh retained implementation ownership through ordinary repair loops.
Fresh Sol/xhigh reviews rejected earlier candidates and independently accepted
the final repair. The manager accepts that evidence-based recommendation.
Only documentation hygiene was adjusted afterward; no reviewed code changed.

PR #149 remains draft and unmerged, and Issue #148 remains open. Known unrelated
baseline package/audit failures are not repaired or concealed by this gate.
No broad green-CI or release claim is made. G3 requires a separate continuation.
