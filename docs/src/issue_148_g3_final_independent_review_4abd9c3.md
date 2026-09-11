# Issue 148 G3 final independent review of `4abd9c3`

## Recommendation

**PASS.**  Repaired code
`4abd9c31aba1d387bac27769730c4e0dc7d0c4a0` and evidence revision
`c6d2291e5c90c8890f48a9574343c14808a5a507` satisfy the current G3 owner
contract for one bounded positive-alpha branch.  Fresh source inspection,
focused replay, exact-rational reconstruction, independent generalized-
eigenvalue calculations, finite-difference derivative checks, and separate
128/256-bit solves support the submitted result.

The accepted scientific result is deliberately limited: the source-twelve,
zero-phase, fixed-saxion P96 degeneracy persists from the radial event onto one
positive-alpha branch through the traced segment.  The computation stores 46
states (one seed plus 45 accepted predictor/corrector transitions) and stops at
the imposed lower-`k` guard with `:k_bounds_reached`.  This is bounded
persistence to a numerical limit, not physical termination, exhaustive branch
enumeration, a G2 cusp reclassification, or a G4/inflation claim.

No scientific-owner decision is required.  The earlier failures were concrete
implementation and validation defects and are repaired at this revision.

## Reviewed state and authority

- Current Issue 148 body and owner comment
  [`5609698767`](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148#issuecomment-5609698767),
  fetched afresh on 2026-09-10.  The owner permits persistence, termination,
  splitting/unfolding, class change, or extra null directions, while requiring
  independent control rank, a test of the zero-phase cubic, reproducible
  geometry/EFT control, and a determined local fate.
- G3 prerequisite contract: `b562780`; audit and replay at
  `docs/src/issue_148_g3_control_audit.md` and
  `scripts/audit_issue_148_g3_controls.jl`.
- Accepted G2 implementation and decision:
  `cc8ac73668ac488a492dd16008c0b790a4e4ef3b` and
  `f9b04ed74bc30cc8e0071fcbe18179a4f85016e2`.
- Approved period-one P96/CYTools metric contract:
  `d1a0b709b69aa680b9bca4223739366c3bfdf8b6`.
- Rejected G3 candidate and prior review: `1c57ed2` and
  `docs/src/issue_148_g3_independent_review_1c57ed2.md`.
- Repaired implementation and evidence: `4abd9c3` and `c6d2291`.
- Review environment: Julia 1.12.6, Darwin arm64, host project and existing
  depot.  The worktree was clean before this review document was added.

## Spec

### Source geometry, physical slice, and independent control

The prerequisite replay passed and independently reconstructs the source
height geometry, 64 exact intersection entries, 39 Mori generators, the exact
two-cycle tip

```text
t_ref = (1,4,4,-2,4,3,3,3)
```

and source actions

```text
(14,29/2,29/2,31/2,31/2,31/2,31/2,16,17,17,25,45).
```

The repaired implementation uses the audited non-radial direction

```text
u = (0,1,2,-1,1,1,1,1),
t(k,alpha) = sqrt(k) (t_ref + alpha u),
0 <= alpha <= 1/20,
```

and reconstructs `tau`, volume, P96 metric, source actions, and all twelve
Eq. (19) coefficients from exact integer/rational source data at every
geometry evaluation.  The audit reproduced `MORI*u >= 0`, positive volume and
metric over the certified slice, and the paper's potent-curve transfer.  It
also reproduced rank two for both `[log(k),alpha]` action controls and the
row-centered full-log-amplitude controls, with singular values
`(56.0275,8.14452)` and `(158.439,42.2188)`.  The off-ray action derivative has
relative residual `0.316508` after its best radial fit.

A fresh reviewer contraction at exact `alpha=1//10000` and `k=1//2` rebuilt
the intersection tensor and compared independently evaluated `tau`, volume,
actions, `K=inv(4(tau*tau'-V*A))/k^2`, and
`(8*pi/V^2) S exp(-2*pi*S)` with `_g3_geometry`.  The geometric quantities and
actions agreed within `1e-70`; full amplitudes agreed within `1e-90`.  The
action and centered-amplitude control matrices retained rank two at that
interior control.  This rules out unconstrained four-cycle perturbation,
radial reparameterization, term reselection, and hidden Float64 source
construction in the high-precision path.

### Genuine continuation and branch identity

The replay reconstructs the accepted 256-bit G2 event in source coordinates,
uses its null direction to form signed local seeds at `alpha=1e-8`, and then
uses an analytic `d(theta,v,k)/dalpha` predictor with a damped fixed-alpha
Newton corrector for the 17 augmented equations.  Every accepted transition
therefore depends on the preceding state and a nontrivial corrector; it is not
an independently solved grid or a post-hoc branch match.

Fresh continuation reproduced:

```text
stored states                         46
accepted transitions                  45
attempted loop iterations             49
rejected backtracking trials          39
final alpha                  1.434567377e-4
final k                         0.5006853945
status / reason        k_bounds_reached
```

The attempt count includes the final bound inspection; the accepted count is
exactly `length(steps)-1`.  Alpha increases and `k` decreases monotonically.
Maximum successive periodic state motion is `0.0021087021`, minimum absolute
adjacent null-vector overlap is `0.999753296`, and the maximum fixed-alpha
corrector-Jacobian condition is `1.56019e9`.  These values support continuous
identity of the submitted branch over the bounded segment.

Every stored point has positive volume, positive curve volumes, source actions
above one, and a positive P96 metric.  Fresh segment minima are:

```text
minimum toric curve volume       0.7075912623
minimum source action            7.0098828294
minimum P96 metric eigenvalue    1.2875523019e-4
minimum transverse eigenvalue    0.01052837772
```

The repair evidence calls `2.328173e-4` the segment-wide minimum metric
eigenvalue.  That number is the final-state value; the true segment minimum is
`1.2875523e-4` near the initial state.  This is a non-blocking documentation
error because the corrected minimum remains positive with a clear margin and
does not alter any acceptance or physical-control conclusion.

The failed opposing signed seed remains only a diagnostic.  Neither code nor
evidence promotes it to absence of another branch, and no negative-alpha or
exhaustive-search claim is made.

### Residuals, nullity, transverse spectrum, and higher derivatives

The repair separates normalized gradient/null/augmented residuals from full
Eq. (19) gradient and Hessian-null residuals.  Fresh direct evaluation over all
46 states found maximum normalized augmented residual `5.69410e-12` and
maximum full gradient/null equation residual `2.64603e-33`.

Each stored state contains the complete eight-value canonical spectrum, one
reported near-null mode, and a positive transverse minimum.  A reviewer formed
the generalized eigenproblem `H x = lambda K x` independently at every state.
Its sorted eigenvalues agreed with the stored Cholesky-whitened spectrum to a
maximum absolute discrepancy of `1.16e-10`.  The reported transverse values
use the normalized augmented-equation Hessian; the full Eq. (19) factor is
positive and separately retained, so it changes the spectrum's common scale
but not nullity or inertia.

The two prior formula defects are closed:

1. `projected_d4` now uses
   `-(2*pi)^4 sum(a*cos(2*pi*Q'*theta).*(Q'*v)^4)`.  At exact
   `alpha=1//10000`, a centered finite difference of the directional Hessian
   gave `D3=1.5296203631e-16` and `D4=1.4104278811e-15`; both agreed with the
   stored diagnostics to relative tolerance `1e-10`, including the positive D4
   sign that the rejected revision inverted.
2. The full Hessian now uses `(2*pi)^2`.  Centered differentiation of the full
   gradient at the same far-segment point agreed with the direct Hessian action
   to relative error below `1e-30`.  At the 256-bit `alpha=1e-8` point, the
   replay also gives direct full null residual `2.05063e-104`; the rejected
   half-factor formula gives exactly `1.02532e-104` and is explicitly excluded
   by regression.

The audited fixed-state cubic-control derivative remains
`-1.3384279422e6`; its one-sided exact-geometry finite difference agrees to
`1.24e-9` relative at the repaired replay's smaller step.  It remains correctly
described as a partial control derivative, not a total derivative along the
branch or a G2 cusp classification.

### Independent precision and status boundaries

The submitted replay independently solves exact rational controls
`1e-8,1e-6,1e-5,1e-4` at 128 and 256 bits and chains each 128-bit result into
a second 256-bit solve.  All twelve solves perform nontrivial iterations,
converge, retain one null mode, and have a positive transverse spectrum.  The
independent/chained 256-bit `k` differences range from `4.56e-72` to
`1.77e-74`.

A separate reviewer run repeated independent 128/256 construction at two
separated controls:

| exact alpha | iterations 128/256 | absolute 128/256 `k` difference | transverse minimum 128/256 |
|---:|---:|---:|---:|
| `1//1000000` | `218/219` | `5.51e-35` | `0.0158760203193 / 0.0158760203193` |
| `1//10000` | `7/9` | `4.22e-34` | `0.246219827763 / 0.246219827763` |

The traced Float64 states are initial guesses only.  Each refinement rebuilds
the integer charges, rational intersection data, fixed rational alpha, P96
metric, actions, and coefficients at the requested precision.

Fresh adversarial calls reproduced the truthful status tuple

```text
(invalid_initial_state, max_steps, step_failed,
 invalid_initial_state, invalid_initial_state)
```

for a perturbed non-solution with zero steps, a valid zero-step request, a
disabled corrector, negative initial alpha, and a zero null vector.  The
perturbed state is retained with `converged=false` and augmented residual
`0.787586`.  No resource or invalid-input boundary returns `:completed`.
Method labels now say fixed-alpha predictor/corrector and corrector-Jacobian
condition; the rejected pseudo-arclength alias and bordered-condition label are
absent.

## Standards

No blocking repository-standard violation was found.  The source preserves
exact/rational inputs, constructs BigFloat source quantities at target
precision, leaves Python optional, does not change a persisted schema or public
normalization contract, and records scientific identity and failure states.
The focused evidence script supplies meaningful regression checks for the
changed numerical and status boundaries.  `using CYAxiverse` succeeded in all
review runs, and the preserved G1 replay also passed.

One non-blocking cleanup remains at
`src/paper_benchmarks/n8_g3_control.jl:355`: `tangent_z` is computed and signed
but never used.  The active predictor correctly recomputes `dz_dalpha` at each
step, so the dead local does not affect the result.  Removing it later would
avoid an unnecessary linear solve and make the method easier to read.

## Exact checks and observed outcomes

All Julia commands used the host Julia 1.12.6 project with
`JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`.

1. `julia --startup-file=no --project=.
   scripts/issue_148_g3_control_evidence.jl` exited zero in 17.5 seconds.  It
   reproduced the 46-state branch, four-control precision ladder, corrected
   D4/Hessian regressions, status boundaries, and `PASS` marker.
2. `julia --startup-file=no --project=.
   scripts/audit_issue_148_g3_controls.jl` exited zero in 12.7 seconds.  Its
   `20/20`, `49/49`, and `10/10` testsets reproduced source geometry, the full
   positive-alpha interval, P96 agreement, independent control ranks, and the
   cubic-control finite difference.
3. A fresh independent Julia probe, separate from the evidence script,
   reconstructed the branch, exact interior geometry, generalized spectra,
   direct full residuals, finite-difference D3/D4 and Hessian action, two
   exact-control precision comparisons, and five status boundaries.  It passed
   `373/373` assertions in 15.3 seconds with the numerical values recorded
   above.
4. `julia --startup-file=no --project=.
   scripts/issue_148_g1_replay_checks.jl` exited zero and printed
   `validation_checks_complete=true`.
5. `git diff --check 1c57ed2...HEAD` exited zero.  The reviewed production
   changes are confined to the G3 source/evidence path; no accepted G2
   classifier or P96 contract code changed.

## Claim boundary and remaining limitations

This PASS establishes one locally persistent positive-alpha degeneracy branch
only through the submitted bounded segment.  The `k_bounds_reached` status is
an imposed computational guard at `k=0.5006853945`, not a physical endpoint.
The review does not establish an opposing branch's absence, negative-alpha
control, exhaustive branches, global-cone behavior, a G2 cusp, saxion
stabilization, inflation, observables, a population result, or G4.  Those
questions remain outside this gate.

Standards findings: zero blocking, one non-blocking dead-local cleanup.  Spec
findings: zero blocking, one non-blocking evidence-number correction.  The
worst issue in either axis does not alter the scientific result or its stated
boundary.
