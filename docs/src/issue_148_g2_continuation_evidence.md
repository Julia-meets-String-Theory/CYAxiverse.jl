# Issue 148 G2 continuation evidence

This replay records the bounded N=8 radial continuation run for the approved
P96 contract. The implementation and replay script were tested at commit
`9fe32eb0bebc4d9ad99fb761357c604674e3681a` on Julia `1.12.6`
(`arm64-apple-darwin24.0.0`). The source is arXiv `2608.14780v1`, Appendix D,
Table 1, with source PDF SHA-256
`b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.

## Contract and source identity

The continuation uses period-one GLSM coordinates `theta`, arguments
`2pi Q theta`, and the owner-approved `K_theta=M96/k^2` metric. `M96` is the
reconstructed `n8_geometry().kinetic` matrix with `V=126`, the Appendix-D
vertices, and the published Eq.96 spectrum. The existing augmented solve
reports its stationary point in leading-charge coordinates; the replay maps it
explicitly to GLSM coordinates before comparison. The twelve source terms and
their rational actions are retained in the reordered `(Q,L)` term identity.
The ten-term author raw-angle path remains the separately labeled A96 check.

## Exact replay

```text
JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. \
  scripts/issue_148_g2_continuation_evidence.jl
```

Observed exit status: `0`. The run used `Random.seed!(148)` for the bounded
regular-seed discovery. The tested output is retained at
`/private/tmp/issue148-g2-run-9fe32eb.log`.

## Observed results

The independent existing twelve-term augmented solve gave
`k=0.6745063700033533`, gradient infinity norm `2.487e-15`, and null residual
`4.778e-15`. Continuation started at `k=0.68` and `k=0.66`, where the bounded
discovery found respectively `(1 minimum, 12 index-1 saddles)` and
`(5 minima, 14 index-1 saddles)` among its finite stationary sample.

Five above-side chains and four below-side chains reported the catastrophe
event. The first selected above-side event bracket was
`[0.67455000, 0.67450000]`; bisection gave
`k=0.6745063700038947`, bracket width `7.451e-13`, gradient residual
`8.286e-14`, and canonical Hessian minimum `1.109e-36`. Agreement with the
independent augmented scale was `5.413e-13`. The continuation branch ID stayed
`3` for all `301` converged points. The bounded post-hoc periodic matcher
comparison used 25 of those points and recorded zero disagreements; its output
was not used to define continuation identity. A second event-chain comparison
found a periodic separation of `3.886e-16` on the finite sampled event set.

The bordered tangent was normalized to unit `(theta,k)` Euclidean norm. Near
the event its radial tangent component was `1e-6` scale, and the bordered
corrector switched to short fixed-k Newton steps when the radial projection
became ill-conditioned. That switch preserves the previous corrected point as
the branch seed and reports the radial failure boundary through the event
bracket. This is the documented localization strategy for the singular fold.

The source-constructed BigFloat ladder converged at 128 and 256 bits. The
stationary gradient residual was `6.185e-45` at both reported refinements. An
independent 256-bit augmented solve gave `k=0.6745063700033669`, gradient
residual `8.435e-71`, null residual `3.449e-68`, and periodic coordinate
distance `2.002e-10` from the continuation-localized point. Charges are exact
integers, actions are exact integer/rational values, and `pi` is typed inside
the BigFloat solve; no Float64 seed is used for the 256-bit refinement.

The P96 diagnostic found one near-null eigenvalue, a stationary gradient, a
vanishing projected second derivative, and positive transverse eigenvalues.
Its classification is `:unresolved` under the existing derivative cutoff,
which is retained unchanged; the A96 diagnostic is separately labeled and also
reports `:unresolved`. This result supports a verified radial degeneracy and
nullity classification without promoting the current cutoff to a cusp/fold
claim. The Eq.96 metric spectrum matched all eight published values within 2%.

The focused replay contains 321 checks in its named regression testsets plus
explicit top-level assertions for the observed diagnostics. These checks cover
source and metric conventions, regular seeds, continuation event recovery,
precision, intrinsic branch identity, and the independent augmented reference.
Existing baseline package failures (the unrelated phase-volume normalization mismatch
and the known N8 JET/ReviseEMFILE environment issue) were not modified or
used as G2 evidence.

## Scope boundary

This artifact covers G2 radial continuation only. It does not make an off-ray
G3 claim, change the N5 design, alter public schemas, or claim physical
inflation success. The metric normalization follows the approved P96 contract;
the source factor-two discrepancy between literal displayed Eq.23 and the
Eq.96 benchmark matrix remains documented in
`issue_148_n8_approved_metric_contract.md`.
