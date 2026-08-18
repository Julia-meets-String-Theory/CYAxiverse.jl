# `classify_point` whitening — reproducible accuracy sweep

**Date:** 2026-08-18
**Change under test:** PR #87 (`efeb971`, "Whiten inflation-candidate classification and hoist the Cholesky factor")
**Script:** `scripts/validate_classify_point_whitening_accuracy.jl`
**Julia:** 1.12.6, run unsandboxed on the local host per `.copilot/AGENTS.md` §0

---

## 1. Why this exists

PR #87 replaced `analyze_inflation_candidates.jl::classify_point`'s two
explicit inverses (`inv(factor')` for the Hessian rotation, `inv(K)` for the
gradient norm) with triangular solves against a caller-owned `Cholesky`
factor — the same quantity, `L⁻¹HL⁻ᵀ` and `‖L⁻¹g‖`, computed a different way.
The commit message reports measuring both forms against a 512-bit `BigFloat`
reference over h11 = 10, 20, 40 and cond(K) = 1e6...1e14, finding the two
forms comparable — both scaling like `eps·cond(K)`, neither systematically
better — and shipping the change on cost/consistency grounds (a modest
1.12–1.32× speedup, and matching the `Kfactor` convention already used
elsewhere) rather than on the accuracy improvement the whitening approach
might suggest.

That sweep was never checked in. Unlike B1
(`validation/2026-08_package_and_performance_review.md` §"B1", with a scripted,
reproducible `BigFloat` comparison), the specific numbers backing Item 3's
"neither is systematically better" claim existed only as commit-message
prose. This script reproduces the sweep so the claim is checked-in and
re-runnable rather than resting on an unlogged one-off computation.

---

## 2. Method

For each `(h11, cond(K))` cell, 20 independent draws vary both `K`'s
eigenbasis rotation and the evaluation point `theta` (`MersenneTwister`
seeded from `(seed_base, h11, cond, draw)`, fully reproducible). `Q`/`L` come
from the package's own `pseudo_Q`/`pseudo_L` fixtures — the same synthetic
generator `test/runtests.jl`'s "whitened point classification" testset
already pins its equivalence tests against.

Per draw, three computations of the same formula:

1. **Pre-#87 form** (`classify_point_explicit_inverse`, byte-for-byte the
   implementation `classify_point` had before this PR): `to_theta = inv(factor')`,
   `canonical_hessian = to_theta' * H * to_theta`; `gradnorm = sqrt(max(dot(g, inv(K)*g), 0.0))`.
2. **Current form** (`classify_point` as shipped): `Kfactor.L \ H / Kfactor.L'`;
   `gradnorm = norm(Kfactor.L \ g)`.
3. **Reference**: the identical formula evaluated at 512-bit `BigFloat`
   precision from the same `Q`/`L`/`theta`/`K`, promoted losslessly (`Float64`→`BigFloat`
   is exact) before any Cholesky factor or solve is formed.

Error is reported relative to the reference's own scale
(`max(maximum(abs, reference_eigs), 1.0)` for the Hessian; `max(abs(reference_gradnorm), 1.0)`
for the gradient norm), matching the tolerance convention already used in
`test/runtests.jl`'s equivalence pins. "Win rate" is the fraction of draws
where the current (whitened) form's error against the reference is strictly
smaller than the pre-#87 form's.

---

## 3. Results

`julia --project=. scripts/validate_classify_point_whitening_accuracy.jl` (defaults: h11 = 10,20,40; cond(K) = 1e6,1e8,1e10,1e12,1e14; 20 draws/cell; 512-bit reference):

| h11 | cond(K) | old max err (Hessian) | new max err (Hessian) | new closer % | old max err (gradnorm) | new max err (gradnorm) | new closer % |
|---|---|---|---|---|---|---|---|
| 10 | 1e6  | 1.013e-11 | 1.013e-11 | 55% | 3.149e-12 | 5.065e-12 | 30% |
| 20 | 1e6  | 1.432e-11 | 1.432e-11 | 50% | 7.993e-12 | 7.160e-12 | 45% |
| 40 | 1e6  | 6.821e-12 | 6.821e-12 | 45% | 2.169e-12 | 3.410e-12 | 45% |
| 10 | 1e8  | 1.643e-09 | 1.643e-09 | 60% | 4.244e-10 | 8.217e-10 | 25% |
| 20 | 1e8  | 1.031e-09 | 1.031e-09 | 55% | 2.993e-10 | 5.153e-10 | 40% |
| 40 | 1e8  | 4.030e-10 | 4.030e-10 | 30% | 2.525e-10 | 2.015e-10 | 60% |
| 10 | 1e10 | 2.412e-07 | 2.412e-07 | 50% | 5.616e-08 | 1.206e-07 | 15% |
| 20 | 1e10 | 1.418e-07 | 1.418e-07 | 25% | 3.256e-08 | 7.090e-08 | 40% |
| 40 | 1e10 | 2.880e-08 | 2.880e-08 | 55% | 3.483e-08 | 1.440e-08 | 35% |
| 10 | 1e12 | 1.581e-05 | 1.581e-05 | 40% | 9.225e-06 | 7.903e-06 | 35% |
| 20 | 1e12 | 1.029e-05 | 1.029e-05 | 55% | 3.038e-06 | 5.145e-06 | 40% |
| 40 | 1e12 | 5.327e-06 | 5.327e-06 | 45% | 2.299e-06 | 2.663e-06 | 35% |
| 10 | 1e14 | 2.613e-03 | 2.613e-03 | 55% | 8.489e-04 | 1.306e-03 | 30% |
| 20 | 1e14 | 1.564e-03 | 1.564e-03 | 45% | 3.259e-04 | 7.816e-04 | 35% |
| 40 | 1e14 | 7.185e-04 | 7.185e-04 | 30% | 4.050e-04 | 3.593e-04 | 45% |

**Totals (300 draws):** whitened form closer on Hessian eigenvalues in 139/300
(46.3%); closer on gradient norm in 111/300 (37.0%). Wall time: 34s for the
full sweep.

---

## 4. Reading the two halves differently

**Hessian eigenvalues: the commit's claim reproduces almost exactly, and
more strongly than stated.** At every cell the old and new max relative
errors agree to the three displayed significant figures — they are not
merely "comparable," they are numerically nearly the same computation. A
higher-precision spot check (h11=20, cond=1e10, draw 1) confirms this isn't
a display-rounding artifact: old and new disagree from each other by
9.5e-7 absolute, while each disagrees from the reference by ≈35.1875 —
i.e. the two forms differ from *each other* by about three-millionths of
how far either is from the truth. This has a mechanical explanation: `factor
= cholesky(K).L`, and old's `to_theta' * H * to_theta` reduces to
`inv(L) * H * inv(L)'` — Julia's `inv` on a `Triangular` matrix is itself
implemented via forward/backward substitution against the identity, not a
general LU-based inverse. So "form the explicit inverse" and "solve
directly" were never two different algorithms here; both bottom out in the
same triangular substitution arithmetic, just reordering when the products
happen. The accuracy motivation for this half of the change was never
going to show up, because there was no meaningfully different computation to
compare.

**Gradient norm: closer to a real trade than a coin flip, and tilted the
other way.** Here old used a genuine dense `inv(K)` (LU-based, since the
`K` argument reaching this line is a plain `Matrix`, not `Hermitian`-wrapped
— it does not get to exploit symmetry or positive-definiteness) against a
`Cholesky`-triangular-solve in the new form — the two algorithms folklore
says should favor the solve. This sweep instead finds the *old* form closer
to the reference in 63% of draws overall, most consistently at h11=10 (only
15–30% new-closer across all five condition numbers). The likely reading:
both LU-with-pivoting and Cholesky-triangular-solve are backward-stable to
the same asymptotic order for a matrix this well short of numerically
singular (cond(K)·eps ≤ ~2e-2 even at the top of this sweep), so which one
lands closer to the reference on a given draw is closer to a realization of
rounding noise than a systematic algorithmic edge — but the realized tilt in
this sweep favors old, not new, which the commit's "neither is
systematically better" framing did not distinguish from a true 50/50.

**Practical bearing.** Both readings stay well inside the same conclusion
the commit shipped on: the change was justified by cost and API consistency,
not accuracy, and no accuracy edge should be claimed for it — if anything,
this sweep's gradnorm numbers argue mildly against one being present. All
measured errors here (≤2.6e-3 relative, even at cond(K)=1e14) are far below
anything that would move a `saddle`/`negative_modes` classification; PR #87's
own documented risk on that front — eigenvalues numerically at zero flipping
sign between forms — sits seven orders of magnitude below these, at
`|eig|/scale ≤ 1.3e-17`, and was independently closed by PR #88's
tolerance-aware mode counts.

---

## 5. Reproduction

```
julia --startup-file=no --project=. scripts/validate_classify_point_whitening_accuracy.jl
```

Deterministic for fixed `--seed-base` (default `20260818`); pass
`--output-csv PATH` for a per-draw CSV. `--h11`, `--cond`, `--seeds`, and
`--precision-bits` widen or narrow the sweep. No HDF5 geometry is read or
written; `Q`/`L` come from `pseudo_Q`/`pseudo_L`, `K` is a synthetic
random-rotation fixture at a requested condition number.

---

## 6. Decision: keep the whitened form (2026-08-18)

The gradnorm win rate (§3–4) raised the question of reverting `classify_point`'s
gradient-norm computation to the pre-#87 `sqrt(max(dot(g, inv(K)*g), 0.0))`
form. **Declined**, for three reasons:

1. **Magnitude.** The measured gap is ~1e-8 relative at low cond(K), rising
   to at most ~2.6e-3 at cond(K)=1e14 — far below anything that would move
   `epsilon = 0.5*(gradnorm/value)^2` across a slow-roll threshold, and small
   next to the 9–13-order-of-magnitude Float64 mass errors §B1 in
   `2026-08_package_and_performance_review.md` already documents as inherent
   to this precision.
2. **No clean partial revert exists.** `classify_point` takes `Kfactor::Cholesky`,
   not `K`. Restoring `inv(K)` for gradnorm alone means either threading both
   `K` and `Kfactor` through every call site (undoing the single-shared-factorization
   consistency PR #87 established) or reconstructing `K` from `Kfactor.L * Kfactor.L'`
   and inverting that — more arithmetic than what ships today, and not even
   guaranteed to reproduce the old numbers exactly.
3. **It would reintroduce a real fix.** `norm(Kfactor.L \ g)` is nonnegative
   by construction; the old form needed the `max(..., 0.0)` clamp specifically
   because a dense `inv(K)` could return a slightly negative quadratic form.
   Reverting brings that failure mode back for a sub-part-in-a-million trade.

**Open caveat, not acted on:** the 63% tilt is measured on synthetic
`pseudo_Q`/`pseudo_L` fixtures, not the real geometry corpus. Confirming it
holds (or doesn't) on stored geometries via `CYAxiverse.read.potential` would
be the cheap next step if this is ever revisited, but given the magnitude
above it is not expected to change the conclusion.
