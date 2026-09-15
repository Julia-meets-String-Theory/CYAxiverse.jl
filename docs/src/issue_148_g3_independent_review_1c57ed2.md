# Issue 148 G3 independent review of `1c57ed2`

## Recommendation

**FAIL.** The candidate contains a real off-ray augmented
predictor/corrector trace, and independent probes support a smooth one-null
branch over the submitted bounded segment. The exact geometric map, source
coefficients, P96 extension, positive-cone controls, and independent-control
rank audit also reproduce. The gate cannot pass at this revision because two
required numerical diagnostics are implemented incorrectly, status handling
can report an unvalidated state as completed, the evidence calls an imposed
`k` bound a branch termination, and the submitted replay omits the required
transverse-Hessian and representative along-segment precision evidence.

These are implementation and validation defects, not an unresolved owner
boundary. The scientific result that survives review is narrower: one local
positive-`alpha` degeneracy branch persists from `alpha=1e-8` through at least
`alpha=1.434567377e-4`, where the computation stops at its declared lower
`k` bound. Nothing in this review establishes physical termination there or
absence of an opposing branch.

## Reviewed state and authority

- Candidate code: `1c57ed2be15741ffcdf0227f0210cbf664d989f6`.
- Candidate evidence: `c40e1e88dee778c55be6877eee71a5c88814b7a7`.
- G3 prerequisite contract: `b562780`; audit document
  `docs/src/issue_148_g3_control_audit.md` and replay
  `scripts/audit_issue_148_g3_controls.jl`.
- Accepted G2 implementation and decision: `cc8ac73668ac488a492dd16008c0b790a4e4ef3b`
  and `f9b04ed74bc30cc8e0071fcbe18179a4f85016e2`.
- Owner-approved P96 contract: `d1a0b709b69aa680b9bca4223739366c3bfdf8b6`.
- Current Issue 148 body and owner clarification comment `5609698767` were
  read directly on 2026-09-10. The clarification allows persistence,
  termination, splitting/unfolding, class change, or additional null
  directions, and does not require a persisted curve. It requires a real fate
  determination with reproducible geometric/EFT evidence. A solver failure or
  imposed scale bound remains a numerical boundary, not physical termination.
- Scientific identity remains `arXiv:2608.14780v1`, PDF SHA-256
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`,
  tracked N=8 HDF5 SHA-256
  `8fc483bc71a7e3f356b512adcfd44555f86a86bb765566c3f650984bf0ef1f4a`,
  source-twelve Table-1 terms, equal fixed one-loop magnitudes, exact zero
  phases, fixed saxions, period-one GLSM axions, and the P96/CYTools metric.

The review used Julia 1.12.6 on `arm64-apple-darwin24.0.0` with
`JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`. The working
tree was clean before this review document was added.

## Blocking findings

### 1. The reported projected fourth derivative has the wrong sign

For the implemented potential

```text
V(theta) = sum_I a_I [1 - cos(2 pi q_I.theta)],
```

the fourth directional derivative is

```text
D_v^4 V = -(2 pi)^4 sum_I a_I cos(2 pi q_I.theta) (q_I.v)^4.
```

`_g3_step_diagnostics` instead uses a positive sign at
`src/paper_benchmarks/n8_g3_control.jl:259`. This disagrees with direct
differentiation and with the established implementation in
`src/paper_benchmarks/catastrophe_diagnostics.jl:164`.

At the independently refined `alpha=1/100000000` point, the candidate returns

```text
projected_d4 = -4.6829387491932273e-24
```

while direct evaluation gives

```text
D_v^4 V      = +4.6829387491932273e-24.
```

Their sum is exactly zero at 256-bit working precision. Projected D3 is
implemented with the correct sign. Because projected D3/D4 are explicit G3
acceptance diagnostics, this is gate-blocking even though the D3 result is
already nonzero on the traced branch.

### 2. The full-factor Hessian/null residual is off by a factor of two, and
residual labels conflate different quantities

The BigFloat result constructs `full_hessian` as `2*pi^2` at
`src/paper_benchmarks/n8_g3_control.jl:424`. The period-one Hessian coefficient
is `(2*pi)^2 = 4*pi^2`, as used by the normalized system at line 226. At the
same 256-bit off-ray point,

```text
reported full_null_residual = 5.8296342186166574e-92
direct full_null_residual   = 1.1659268437233315e-91
ratio                       = 0.5.
```

In addition, the successful BigFloat return assigns
`gradient_residual=norm(system.residual, Inf)` at lines 418 and 428. That is
the maximum residual of the combined gradient, null, and normalization
equations, not the gradient residual. At the exact-rational `alpha=1e-8`
review point the returned value was `3.5915e-62`, while the normalized
gradient and null residuals were `1.6531e-66` and `1.2593e-64`. The evidence
table labels the combined value as a normalized gradient residual. Full and
normalized residuals are not retained for every traced state.

The absolute residuals are numerically small, so this does not erase the
observed solution. It does fail the required exact raw/scaled residual
contract.

### 3. Invalid initial states can be reported as converged and completed

`n8_g3_predictor_corrector` pushes its input state with `converged=true`
without testing the augmented residual at
`src/paper_benchmarks/n8_g3_control.jl:293-301`. When `n_steps=0`,
`:max_steps` is then mapped to `status=:completed` at lines 390-392.

An independent probe perturbed the accepted radial point by `0.01` in every
axion coordinate and requested zero steps. It returned

```text
status                 = completed
termination_reason     = max_steps
stored states          = 1
step.converged          = true
gradient residual      = 0.7875858918
null residual          = 0.0047600977
normalization residual = 0
```

This is a concrete false-success boundary. The disabled-corrector test in the
submitted replay correctly returns `:step_failed` for its valid initial seed,
but it does not cover initial-state validation or resource exhaustion.

### 4. The submitted fate statement turns a scale bound into physical
termination

The code returns the distinct status `:k_bounds_reached`, which is useful.
The evidence then states that the branch “terminates within the declared `k`
bounds” and prints “branch terminates at k lower bound.” The final state is
`k=0.5006853945`, and the next attempt is stopped by the explicit guard
`k <= 0.501` at lines 314-316. No singularity, loss of cone/EFT control,
additional null direction, branch collision, or corrector failure occurs
there.

The supported result is **persistence on a bounded segment, followed by an
imposed scale-bound stop**. That is an owner-admissible G3 fate result for the
tested segment. It is not branch termination, and behavior beyond the bound
is untested.

### 5. Required along-segment controls are absent from the submitted durable
evidence

`G3ContinuationStep` stores the metric minimum, full D3/D4, cone quantities,
residuals, and a condition number, but it stores no canonical Hessian spectrum,
inertia, transverse minimum, or additional-null count. The submitted high-
precision replay refines only the first off-ray seed at `alpha=1e-8`; it does
not refine representative interior or far-segment states. Its `alpha` is also
passed as a Float64 control rather than an exact rational or target-constructed
BigFloat, without declaring that 53-bit control boundary.

Independent review probes show that the missing facts are likely repairable:

- all 46 stored states had one normalized canonical near-null eigenvalue with
  maximum magnitude `2.15043e-9`, while the minimum next transverse
  eigenvalue was positive at `0.0105284`;
- representative second transverse eigenvalues at exact rational
  `alpha=1e-8, 1e-6, 1e-5, 1e-4` were respectively `0.0105284`, `0.0158760`,
  `0.0412276`, and `0.246220`;
- independent 128/256-bit solves converged at all four points from traced
  seeds with exact-rational controls; their `k` changes from 128 to 256 bits
  ranged from `5.96e-36` to `4.22e-34`;
- at `alpha=1e-4`, independent 128-bit, independent 256-bit, and chained
  256-bit solves took 7, 8, and 2 iterations. Independent/chained 256-bit
  agreement was `1.29e-67` in `k` and `2.66e-67` in maximum periodic axion
  distance.

Those reviewer observations support the branch, but the production result and
focused regression need to retain and assert the required controls throughout.
They must also use the corrected D4 and residual formulas.

## What independently passes

### Geometry, source coefficients, and control rank

The prerequisite replay passed `20/20`, `49/49`, and `10/10` checks. It
reproduced the exact two-cycle map

```text
t(k,alpha) = sqrt(k) [(1,4,4,-2,4,3,3,3)
                      + alpha (0,1,2,-1,1,1,1,1)]
```

with all 39 toric curve volumes nondecreasing along the selected shape
direction. The exact source actions at the tip are

```text
(14,29/2,29/2,31/2,31/2,31/2,31/2,16,17,17,25,45).
```

The action and row-centered full-amplitude controls paired with `log(k)` both
have rank two. Their singular values reproduce as `(56.0275,8.14452)` and
`(158.439,42.2188)`. The best radial fit leaves relative action-control
residual `0.316508`. The cubic partial-control derivative reproduces as
`-1.338427942e6`; this is correctly interpreted as a fixed-state partial
derivative, not a total derivative along the later branch and not a G2 cusp
classification.

Direct comparisons between the candidate geometry and the audit's independent
exact-rational contraction were made at `alpha=0,1e-4,1/20` and at both
`k=1/2` and a rational representation of the accepted event scale. The
two-cycle-derived `tau`, volume, curve volumes, actions, P96 metric, and full
Eq. (19) coefficients agreed exactly at rational points or to 256-bit rounding
(`metric` maximum discrepancy at most `1.69e-79`; relative coefficient
discrepancy at most `7.04e-75`). This validates recomputation of
`tau,V,K,S,Lambda` from the source geometry rather than independent divisor
perturbations or term reselection.

The common `8*pi/V^2` factor is present in `source_coefficients` and diagnostics.
Removing it together with the largest term for the augmented solve is a
positive common scaling and does not change stationarity or nullity. The
candidate remains source-twelve, zero-phase, fixed-saxion, and fixed-one-loop.
No persisted schema, package version, source population, or G4/inflation path
is changed.

### Actual continuation and branch identity

The submitted evidence replay exited zero. It performs a fixed-`alpha`
17-equation augmented corrector in `(theta,v,k)` after an analytic
`d(theta,v,k)/dalpha` predictor. Every accepted transition required three or
four corrector iterations; the result is not an independently solved grid or
a copied branch label.

The result contains 46 stored states: one converged input seed and 45 accepted
predictor/corrector transitions. `accepted_steps` correctly reports 45, while
the prose incorrectly calls all 46 “accepted steps.” There were 49 attempted
loop iterations and 39 rejected backtracking trials. The final accepted state
has step index 48; attempt 49 stops at the bound.

Independent continuity checks found maximum successive periodic Euclidean
point motion `0.00210870` and minimum absolute adjacent null-vector overlap
`0.999753`. Both `alpha` and decreasing `k` are monotone. The maximum fixed-
`alpha` augmented-Jacobian condition number is `1.56019e9`. The field named
`bordered_condition` is not a pseudo-arclength bordered condition; it is
`cond(jacobian[:,1:17])`. The alias
`n8_g3_pseudoarclength_continuation = n8_g3_predictor_corrector` at line 398 is
therefore inaccurate and should not remain as an implicit method/API claim.

Across the stored segment, the maximum normalized gradient, null, and
normalization residuals independently reproduce as `5.69410e-12`,
`3.89631e-12`, and `2.22045e-16`. Direct full-factor gradient/null evaluation
gives a maximum equation residual of `2.64603e-33`. Minimum curve volume,
minimum source action, and minimum metric eigenvalue are `0.707591`,
`7.009883`, and `1.28755e-4`, so cone, divisor/EFT-transfer, volume, and P96
positivity remain controlled throughout the actual traced segment.

The side `+1` seed failure does not undermine this one-branch persistence
claim. A bounded sensitivity probe over `alpha=1e-10,1e-9,1e-8,1e-7` and
displacements `1e-5,1e-4,1e-3` repeatedly recovered the side `-1` branch for
appropriate scale-matched displacements and did not recover the side `+1`
branch. This is useful failure evidence, but it is not proof that a second
branch is absent; the sign of a null vector is arbitrary and no exhaustive
enumeration is required for the accepted one-segment claim.

### Preserved gates and scope

`scripts/issue_148_g1_replay_checks.jl` exited zero with
`validation_checks_complete=true`. The accepted G2 classification remains
`:unresolved`; the candidate does not alter the existing classifier or claim
a G2 cusp. The legitimate G3 cubic variation is a new off-ray observation and
does not retroactively reclassify G2.

## Minimum correction and re-review boundary

1. Correct the full D4 sign and the full Hessian coefficient, then add direct-
   formula regressions tied to the existing catastrophe diagnostic convention.
2. Separate and accurately name augmented, normalized gradient/null, and full
   gradient/null residuals; retain the required values for representative
   points and preferably for every stored step.
3. Validate the initial augmented state before recording it. Make `max_steps`,
   invalid/nonconverged input, corrector failure, alpha bound, and `k` bound
   truthful non-success states. Do not map a resource limit to `:completed`.
4. Retain and assert canonical one-null count, transverse Hessian spectrum or
   at least transverse minimum/inertia, D3/D4, cone/EFT controls, and condition
   information throughout the accepted segment.
5. Add independent and chained 128/256-bit checks at multiple exact-rational
   off-ray controls, including an interior and far-segment point. Seeds may be
   Float64 initial guesses; source geometry and fixed controls must be
   constructed at target precision or their source boundary must be explicit.
6. State the fate as bounded persistence followed by `k_bounds_reached`.
   Correct the 45-transition/46-stored-state count. Remove the pseudo-arclength
   alias and `bordered_condition` label unless that method is actually
   implemented.
7. Keep the opposing-seed failure as a diagnostic. Do not promote it to branch
   absence, and do not require exhaustive enumeration to accept the repaired
   bounded persistence claim.

No scientific-owner decision is needed for these corrections. Fresh review
should focus on the corrected diagnostics and status boundaries; it need not
repeat unrelated known package failures or expand into G4/population work.

## Commands and observed outcomes

All focused Julia commands used:

```text
JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. <command>
```

1. `scripts/issue_148_g3_control_evidence.jl` exited zero and printed the
   submitted 46-state trace, `k_bounds_reached`, one failed signed seed, and
   the submitted 128/256 seed refinement.
2. `scripts/audit_issue_148_g3_controls.jl` exited zero with `79/79` total
   checks and the exact geometry, cone, rank, P96, and cubic-control values
   recorded above.
3. `scripts/issue_148_g1_replay_checks.jl` exited zero with
   `validation_checks_complete=true`.
4. Reviewer inline Julia probes reconstructed the accepted G2 event with
   `n8_bigfloat_augmented_solve`, called
   `n8_g3_seed_from_radial` and `n8_g3_predictor_corrector` with the evidence
   arguments, and independently recomputed periodic step distances,
   null-vector overlaps, exact full residuals, canonical spectra via
   `factor.L \ H / factor.U` for `factor=cholesky(Symmetric(K))`, and direct
   D3/D4 formulas. They produced
   the continuation, residual, transverse, and sign/factor observations above.
5. A target-precision probe called `n8_g3_bigfloat_augmented_solve` independently
   at exact rational controls `1//100000000`, `1//1000000`, `1//100000`, and
   `1//10000`, and chained the `1//10000` 128-bit result into a 256-bit solve.
   All converged with the values recorded above.
6. A bounded seed-sensitivity probe called `n8_g3_seed_from_radial` for both
   signs at four positive `alpha` seeds and three displacement scales. It
   supported the limited opposing-seed interpretation recorded above.
7. `git diff --check 1c57ed2^..c40e1e8` exited zero. The candidate tree was
   clean before this review document.

No broad package-suite or release-readiness claim is made. Known unrelated
baseline failures were outside this focused scientific review.
