---
spec_id: CYAX-0173-XFV
title: Preregistered cross-fixture precision validation
issue: 173
class: S2
status: draft_pending_independent_review_and_owner_approval
workstream: numerical architecture
parent: CYAX-0170
depends_on: [CYAX-0173-B6]
approval_ref: null
---

# Preregistered cross-fixture precision validation

## Purpose and authority

This is a proposed S2 scientific contract for a bounded study of the conditions
under which Float64 canonical-Hessian computation fails to preserve the
three-lowest-mode diagnostic spectrum or subspace. Issue #173 and the accepted
N=5 B6 evidence motivate the study. The existing Float64 and high-precision
routes remain separately authoritative for historical parity. Neither route is
promoted as the physical oracle. A supported research association or scalar
boundary cannot change production precision, route selection, source behavior,
historical data, or spectrum migration acceptance criteria.

This draft is not approval to run the study. Before implementation or fixture
outcome inspection, fresh independent Spec and Scientific reviews of the same
exact `spec.md`, `plan.md`, and `tasks.md` bytes must satisfy the
`CYAX-REVIEW-AXES/1` successful-review semantics. The Spec review must satisfy those `CYAX-REVIEW-AXES/1` successful-review semantics; the Scientific verdict may be `PASS` or a limitations-only
`PASS_WITH_NONBLOCKING_FINDINGS` with no blocking finding. Preserve each exact
reviewer verdict and every stated limitation. `REQUEST_CHANGES`, `BLOCKED`, or
any blocking Scientific finding fails this prerequisite. A separate final S2
owner approval tied to the reviewed revision by `approval_ref` remains
required; review success does not grant owner approval or study authority. A
later execution handoff must bind those exact documents and that approval. The
present contract preparation stops at the owner approval gate.

Source facts, frozen B6 empirical evidence, proposed study rules, later
observations, and physical inference must be labelled separately. The fixed
`k=min(3, canonical Hessian dimension)` sector is a B6-derived diagnostic
choice, not a universal definition of the physical light sector.

## Study population and preregistration

**XFV-01 — Inputs and eligibility.** The later study may read only configured
canonical CYAxiverse geometry/potential data through the package read and file
structure APIs. The counting unit is one uniquely identified canonical
`GeometryIndex(h11, polytope, frst)`. Before any route outcome, require
parseable authoritative metric and potential inputs for both existing routes,
canonical Hessian dimension at least three, and complete selection metadata.
Known results, disagreement labels, route outputs, and scientific outcome data
must not affect eligibility, ordering, selection, replacement, or split. B6 is
excluded from the new sample and evaluated only as a separate positive control.
Fewer than 30 eligible inputs stops the study before outcomes.

**XFV-02 — Deterministic sample.** Sort all eligible IDs lexicographically by
`(h11, polytope, frst)`. For inventory size `N`, set `q=floor(N/5)` and
`r_N=N mod 5`; partition the sorted list into five ordered contiguous blocks,
with the first `r_N` blocks of size `q+1` and the remaining blocks of size `q`.
This quotient/remainder allocation occurs before hash ranking. Within each
block rank IDs by ascending SHA-256 of the UTF-8 string
`CYAX0173-XFV1|h11|polytope|frst` (using the canonical integer strings); use
canonical ID order to break a hypothetical hash tie. Select six per block.
The first four selected by rank per block are discovery (20 total); the next
two are sealed holdout (10 total). Freeze the full eligible-inventory digest,
IDs, block sizes, ranks, selections, split, rule/seed, code and `vmm`/data
identities in `fixture_manifest.json` and its SHA-256 before any route
comparison. The manifest must also bind each selected consumed input object by
SHA-256, byte length, schema/version identity where applicable, and immutable
dataset/repository identity; a local path or GeometryIndex alone is insufficient.

**XFV-03 — Replacement and exposure.** Replace a selected ID only when missing
or corrupt authoritative input is established before either route produces a
numerical outcome. Use the next unused hash-ranked ID in the same block and
record the reason and both identities. After any outcome, retain the selected
fixture and classify failure or unresolved status. Freeze a prior-outcome
exposure register before any new route output. For each known Float64 or HP
outcome, cite its durable, publication-safe source identity in that register;
prior exposure never changes selection. A holdout fixture exposed to relevant
prior Float64 or HP outcomes
can be reported descriptively but makes scalar-boundary support inconclusive.
All 30 selected IDs remain in every disposition denominator.

## B6 positive-control gate

**XFV-04 — B6-XFV2.** Before interpreting any new-fixture result, verify the
exact accepted frozen B6 input/artifact identities and require the accepted P1
replay to exit `0` with `P1_REPLAY_PASS`. With no tolerance retuning, require:

The frozen B6 evidence is content-bound by these SHA-256 identities:

| Artifact identity | SHA-256 | Public scientific provenance |
| --- | --- | --- |
| `b6-matching-declaration.json` | `ef40308b2e1241bda488b7ba7e7c3fceb17d92b69155087d8782eb6fd9ea56ee` | Pre-result matching, cluster, and subspace-comparison declaration for the frozen N=5 B6 diagnostic. |
| `b6-diagnostic-synthesis.json` | `34ad8a944408637cea79a1c95033b42f4a2292f6909c58088a9e31135332beea` | Machine-readable synthesis of the bounded frozen N=5 B6 diagnostic. |
| `b6-precision-ladder.toml` | `48cc2419d453b911b5decdf77299727c24330031796492a086c5f382fefa48e5` | Frozen B6 precision-ladder readbacks and matrix diagnostics. |
| `b6-report.md` | `ca73dea10d4f9e93cb24d7e725dc82e2c92d34ea44abe0529a65fc3bd6396c95` | Descriptive report of the bounded frozen N=5 B6 diagnostic. |
| `src/generate.jl` | `d55d53a1bd1201bfd4783d59b854f53d99bfef37ec0f080a93a972e1f7292e58` | Package source module at the B6 reference source state; defines `spectrum_mode_counts` and the leading-Hessian functions. |

These names and descriptions identify the frozen scientific evidence without
publishing private exchange locators. The identities bind historical artifacts;
they do not grant authority beyond this positive-control contract.

1. Standard Float64 first-three log10 masses within `1e-12` of
   `[13.860912539123262, 14.073585277123616, 14.230434590975188]`, with raw
   historical replay signs `[-1,+1,+1]`.
2. HP first-three raw historical replay signs `[+1,+1,+1]` at 128,
   256 and 512 decimal digits; 128/256 and 256/512 first-three sorted-mass
   maximum deltas at most `1e-8` dex. Replay this item on the frozen B6
   diagnostic route with `C_HP=promote(C64)`, using the same assembly as
   `W_HP80`.
3. HP solve of the promoted Float64 assembled matrix: first-three masses
   within `1e-12` dex of
   `[13.238179298077366, 13.843308526910718, 13.998657518779794]`, with raw
   historical replay signs `[+1,+1,-1]`.
The full-spectrum toleranced partition in XFV-05 remains normative for sign
stability.
4. Float64/HP three-lowest-mode projector Frobenius distance at most `1e-12`.
   For the study's Float64 route, construct and freeze each fixture's kinetic
   matrix from its stored Float64 geometry-level `Kinv` using the K-based
   `potential()` reader route in package module `read.jl`,
   `K64=Hermitian(inv(Hermitian(Matrix{Float64}(Kinv))))`;
   `Kinv` is input to this inversion, not the kinetic matrix itself; this is not
   the `potential_factored()` route that factors `Kinv` directly. Compute
   `C64=inv(cholesky(K64).U)` using the upper-Cholesky convention, for which
   `C64' K64 C64=I` in exact arithmetic.
   For the B6 positive-control diagnostic, assemble the frozen 80-digit
   `W_HP80` from promoted inputs with `C_HP=promote(C64)` and the exact
   integer `Qtilde`; use that same matrix for the `E_inf` comparison with
   `promote(W64)`. This B6 assembly factor does not define the HP reference
   route for new fixtures: XFV-05 recomputes `C_K` from `K_HP` at each HP
   ladder precision. Define route-specific `W64` and `W_HP` for each fixture
   in the common scale specified by XFV-05; do not reuse B6 matrix objects
   across fixtures. The separate HP solve of `promote(W64)` in item 3 uses that
   exact promoted matrix. For this B6 positive-control replay, use the frozen
   diagnostic `W_HP80` for the HP projector, `T_HP80`/`Delta_T`, and `kappa_2`; the projector and tensor
   use its frozen diagnostic eigensystem. For new fixtures, the XFV-05 HP reference route remains the separate
   recomputed-`C_K` route for eigensolves, residuals,
   signs, masses, and partition.
   The `E_inf<=1e-15` and `kappa_2>=1e70` checks below apply only to the
   B6 positive control; they are not acceptance gates for new fixtures. For
   assembly, compute
   `E_inf = ||W_HP80 - promote(W64)||_inf / ||W_HP80||_inf`, promoting `W64`
   to the same 80-decimal-digit (268-bit readback) precision, in the same
   frozen canonical whitened coordinates and common
   scaling; `||.||_inf` is the induced infinity norm (maximum absolute row
   sum). Any nonfinite matrix input fails the gate. If numerator and denominator
   are both zero, `E_inf=0`; if the denominator is zero and the numerator is
   nonzero, `E_inf=+Inf`, which fails the upper-bound comparison. Require
   `E_inf <= 1e-15`. Compute the
   condition gate for this B6 positive control on the exact frozen
   80-decimal-digit (268-bit readback) diagnostic `W_HP80` assembled with
   `C_HP=promote(C64)` and used for `E_inf`, with
   `kappa_2(W_HP80)=sigma_max(W_HP80)/sigma_min(W_HP80) >= 1e70`.
5. For the basis-invariant three-lowest-mode aggregate quartic tensor in the
   frozen canonical metric/mapping, compute `T_HP80` from the frozen B6
   diagnostic eigensystem of `W_HP80` built with `C_HP=promote(C64)`; `T64`
   uses the Float64 route. Then
   compute
   `Delta_T = abs(log10(||T64||_F) - log10(||T_HP80||_F))`. If both Frobenius
   norms are zero, `Delta_T=0`; if exactly one is zero, `Delta_T=+Inf`; any
   other nonfinite input fails the B6 gate. Require `Delta_T <= 1e-12`. The
   frozen fixed-input reduction/product-order
   variants remain bitwise equal to the source Float64 scaled canonical matrix
   and keep first-three signs `[-1,+1,+1]`.

Failure stops interpretation and returns `BLOCKED_FOR_REBIND` with exact failed
checks. Frozen historical B6 artifacts remain unchanged. Their prose estimate
`3.48e73` for the condition number is a historical quantitative error: the
frozen eigenvalues imply approximately `3.484032825071285e70`. The correction
does not weaken the `>=1e70` positive-control gate and grants no new scientific
or production authority.

## Precision and light-sector diagnostics

**XFV-05 — Ladder and reference.** Run the fixed decimal-digit ladder
`64, 80, 128, 256`, then 512 only when needed for the reference. The 256-digit
result is the HP reference only if 128/256 comparisons pass: sorted first-`k`
mass-multiset maximum absolute delta `<=1e-8` dex; metric-consistent
first-`k` subspace projector Frobenius distance `<=1e-10`; finite/defined
aggregate light-tensor log10 Frobenius norm difference using the XFV-04
`Delta_T` construct and its zero-norm handling `<=1e-8`; and signed
classification stability. For the HP convergence projector and aggregate light-tensor comparisons at
128/256 and, after fallback, 256/512, build deterministic XFV-06 clusters
separately at each compared point. Number modes by positions in ascending-mass
order, which selects the first `k`. Let `S` be the smallest set of positions
containing `1..k` such that no cluster at either point has one position in
`S` and one outside `S`. If `S={1..k}`, the projector comparison is
unchanged. If `S` is larger, compare at both points the projectors onto modes
at positions in `S` and compute the aggregate light-tensor convergence
comparison on `span(S)` at both points, not on the first-`k` span. Apply the
existing `<=1e-10` projector and `<=1e-8` light-tensor gates to those
comparisons. Record `S` and its rank with each convergence result. If `S`
contains all `n` positions, both projectors are the identity, so their
distance is `0` and the existing `<=1e-10` projector gate passes; record
the projector comparison as vacuous. When XFV-05 `S` in the status-deciding comparison (128/256 without fallback
or 256/512 after fallback) is larger than `1..k`, mark the fixture
cutoff-crossing even if the HP-reference point in that comparison alone has
no cutoff-spanning cluster; its endpoint remains `UNRESOLVED`, only its
subspace-discrepancy response is `UNRESOLVED` and excluded from its Spearman
pair, and the mass response stays available under its existing filters. The
cutoff-cluster span is allowed in this convergence gate in addition to its
diagnostic use; it does not restore a resolved first-`k` endpoint.
At every executed ladder point—the Float64 route and
HP 64, 80, 128 and 256 digits, plus HP 512 when the fallback runs—the required residual set is
all `n` eigenpairs of each route's full spectrum, including zeroish modes.
Record both residuals for every pair; the first-`k` light-mass/subspace
selection does not narrow this set. For each
whitened pair `(lambda,u)`, record
`rho_W = ||W*u-lambda*u||_2 / (||W||_2*||u||_2 + abs(lambda)*||u||_2)`.
For its paired generalized vector `v`, record
`rho_G = ||H_theta_common*v-lambda*K*v||_2 / (||H_theta_common||_2*||v||_2 + abs(lambda)*||K||_2*||v||_2)`.
On Float64, evaluate the pair with `W64`, `H_theta_common_64`, `K64`, and
`v=C64*u`. On the HP reference route at precision `p`, evaluate it with
`W_HP[p]`, `H_theta_common_HP[p]`, `K_HP[p]`, and `v=C_K[p]*u`, pairing each
`lambda` with the eigenvector from that same `W`; the generalized residual is
against that same-precision `K_HP[p]` and HP common-scaled Hessian. At every executed point, require Float64 residuals `<=1e-12` and HP
residuals `<=1e-30`. The eigenpairs and both residuals must be finite and defined at
every point. When the 128/256 comparison passes, record `HP_REFERENCE_CONVERGED` and use
the 256-digit result as the HP reference. Any 128/256 comparison failure
triggers the 512-digit run. Once 512 runs, the 128/256 comparison failure is
superseded for comparison status, and apply the same checks to 256/512. If
256/512 passes, record `HP_REFERENCE_CONVERGED` and use the 512-digit result
as the HP reference. If its only failure is sign stability, with all required
HP quantities finite and defined, record `HP_SIGN_UNRESOLVED` and define the
mass and subspace responses against 512. Any other 256/512 comparison failure
yields `HP_REFERENCE_UNRESOLVED`. The all-executed-point eigenpair, residual,
finite, and defined requirements below remain independent and apply at every
executed precision. For the comparison used to establish the reference, every
required mass, projector/subspace metric, and aggregate tensor metric must be
finite and defined; those comparison metrics retain their current scope. The all-`n`
eigenpairs and residuals above must be finite and defined at every executed
point. Any nonfinite or undefined HP eigenpair or residual, or HP residual
above `1e-30`, at any executed point yields `HP_REFERENCE_UNRESOLVED`; any
corresponding Float64 failure, including residual above `1e-12`, yields
`UNRESOLVED` and cannot satisfy an endpoint. These are validity gates, not
physical-discrepancy thresholds.

For signed stability and endpoint negative-eigenvalue counts, use the full
spectrum of each route's common-scaled canonical whitened matrix
`W=C' H_theta_common C`. Let `L` and `Q` denote the fixture's frozen
`Ltilde` and `Qtilde` inputs, and let `q_a` be column `a` of `Q`. The source
native leading Hessian is
`H_theta,native=sum_a L[1,a]*10^(L[2,a])*q_a*q_a'`. The source uses `L[1,a]`
directly, without applying `sign`. The frozen B6 ladder's
`sign(Ltilde[1,:])` shorthand has the same values only for that witness,
because all five first-row entries are `+1`; the source rule above governs
other fixtures.

For each fixture, compute `r_scale=max_a Ltilde[2,a]` as the binary64 value in its frozen Float64 input.
At each HP precision, promote that value exactly before forming
`10^(-r_scale)`; a printed decimal is not an HP literal. The pinned Float64
common-log scaling weights each term by `L[1,a]*10^(L[2,a]-r_scale)`; a
Float64 term with `L[2,a]-r_scale < log10(floatmin(Float64))` is set exactly
to zero. Thus `W_common=10^(-r_scale)*W_native`, and the eigenvalues use that
same factor. Define `H_theta_common=10^(-r_scale)*H_theta,native`. Apply the
same exact `10^(-r_scale)` to each native HP matrix built from the directly
promoted fixture inputs; do not discard HP terms to imitate Float64 underflow.
For each fixture, set `K_HP[p]=promote(K64)` exactly at each requested HP
ladder precision `p`, then recompute `C_K[p]=inv(cholesky(K_HP[p]).U)` at that
precision. Do not reconstruct `K_HP` from rounded decimal text or invert
`Kinv` again at HP. The Float64 route uses `W64=C64' H_theta_common_64 C64`;
the HP reference route uses `W_HP[p]=C_K[p]' H_theta_common_HP[p] C_K[p]` for
its eigensolve, residuals, signs, masses, and full-spectrum partition. This
HP reference object is separate from the B6-only `E_inf` assembly using
`C_HP=promote(C64)`.

The native eigenvalue is dimensionless in the source's reduced-Planck
convention, with `2*pi` applied separately in the mass conversion. If
`lambda_scaled` is an eigenvalue of the common-scaled matrix, then
`lambda_native=10^(r_scale)*lambda_scaled`. The source mass conversion is
`m/eV=2*pi*(2.435e18)*1e9*sqrt(abs(lambda_native))`; `lambda_native` is not
a bare physical `m^2/M_Pl^2` quantity. The package module `generate.jl`
defines `spectrum_mode_counts` and the leading-Hessian functions. Apply the
pinned `spectrum_mode_counts(relative_tolerance=1e-10)` once to each full
scaled spectrum at each route and ladder point. It computes
`s=max(Float64(maximum(abs,lambda)),1.0)` and
`tau=Float64(relative_tolerance)*s` in Float64, including for HP eigenvalue
vectors; the eigenvalues retain their route precision. Do not divide a route's
spectrum by its own spectral norm. Light-sector and light-cluster
negative counts are read from the full-spectrum partition, with no new
tolerance computed on a light subset. Use the returned Float64 `s` and `tau`,
with `s=max(Float64(maximum(abs,lambda)),1.0)` and
`tau=Float64(1e-10)*s`; classify `lambda < -tau` as negative,
`-tau <= lambda <= tau` as zeroish, and `lambda > tau` as positive.
Zeroish modes do not count as negative. Any nonfinite eigenvalue makes that
sign-partition comparison unresolved. Stability requires equal negative counts
and equal negative/zeroish/positive cardinalities between the compared ladder
points. Raw B6 solver signs remain historical replay outputs; this toleranced
partition is normative for sign stability and endpoint negative-count
comparisons. In the bound `b6-precision-ladder.toml`, the HP native-unit fields
`precision_ladder[].eigenvalues_in_mass_order` (main promoted-frozen-`C64`
route) and `precision_ladder[].arbitrary_precision_whitening_eigenvalues`
(independent `C_K` route) are multiplied by `10^(-r_scale)` before
`spectrum_mode_counts`, with B6 `r_scale=-15.59436499571845`; this printed decimal identifies that
binary64 value and is not an HP literal. The Float64
field `float64_vs_reference_high_precision80.float64_solver_eigenvalues_paired_by_reported_mass_order`
and `precision_ladder[].high_precision_solver_on_float64_matrix_eigenvalues`
(HP solve of `promote(W64)`) are already common-scaled and use factor `1`.
The full partition is `(0,3,2)` on Float64 and both HP routes at
64/80/128/256/512; the three existing light labels inherit `(0,3,0)`. This arithmetic control does not assert
cross-route eigenvalue equality. Any 128/256 comparison failure triggers 512. Without fallback, a passing
128/256 comparison records `HP_REFERENCE_CONVERGED` with the 256-digit
reference. Once 512 runs, the 256/512 comparison decides comparison status
and supersedes the earlier 128/256 failure. A passing 256/512 comparison
records `HP_REFERENCE_CONVERGED` with the 512-digit reference. If the only
256/512 comparison failure is sign stability, with all required HP quantities
finite and defined, record `HP_SIGN_UNRESOLVED`; any other 256/512 comparison
failure gives `HP_REFERENCE_UNRESOLVED`. These comparison decisions do not
override the independent all-executed-point HP finite/residual gates above:
any nonfinite or undefined HP eigenpair or residual, or HP residual above
`1e-30`, at any executed point still yields `HP_REFERENCE_UNRESOLVED`.
Any nonfinite or undefined required Float64 quantity, or Float64 residual above
`1e-12`, yields `UNRESOLVED` and cannot satisfy an endpoint. A cluster
crossing the first-`k` boundary yields endpoint `UNRESOLVED` under XFV-06–07.
A failed route is separately recorded.
Read back requested/actual precision at every ladder point and restore global
ArbFloat precision after each fixture, including failures. No tachyon/stability
or route sign-disagreement claim may use an unresolved HP reference.

**XFV-06 — Basis-invariant comparison.** Compare the sorted multiset of the
`k=min(3,dimension)` lowest physical masses in log10 units. This is a set
diagnostic, not one-to-one mode identity. For the Float64-versus-HP-reference
comparison, map each route's eigenvectors to θ coordinates, `v=C·u`, using
its own factor (`C64` for Float64 and `C_K[p]` for HP). Let `V` collect the
selected mapped vectors for each route. Use the single metric
`K=K_HP[p]=promote(K64)` for both routes and form each K-orthogonal projector
as `P=V(Vᵀ K V)⁻¹VᵀK`. Take principal angles in the K inner product and
define the projector Frobenius distance as `||R(P_F64-P_HP)R⁻¹||_F`, with
`K=RᵀR` and `R=cholesky(K).U`. On B6, where both routes use `C64`, this
equals the existing canonical comparison. This cross-route definition leaves
the same-route HP convergence comparisons in XFV-05 unchanged. Use the frozen
B6 deterministic cluster rule: sort eigenvalues
numerically and join adjacent values when `g_i<=1e-8`, where
`g_i=abs(lambda_(i+1)-lambda_i)/max(||A||_2,abs(lambda_i),abs(lambda_(i+1)))`
and `A` is the matrix whose eigenvalues are being clustered;
connected components form clusters. If a cluster crosses the
first-`k`/complement boundary, the first-`k` aggregate is `UNRESOLVED` for
endpoint classification. Do not select only part of that cluster or expand
`k` for endpoint classification. If XFV-05 `S` in the status-deciding comparison (128/256 without fallback
or 256/512 after fallback) is larger than `1..k`, apply the same
cutoff-response rule even when the HP-reference point in that comparison
alone has no cutoff-spanning cluster. In
either case, the endpoint remains `UNRESOLVED`, even if the widened HP
convergence comparison passes. Mark only the associated basis-dependent
subspace-discrepancy response `UNRESOLVED` and exclude it from its Spearman
pair; the mass response remains available under its existing availability and
finite-response filters. Keep the fixture in all disposition accounting, and
evaluate its other response pairs under their own preregistered availability
and finite-response rules. Use of the cutoff-cluster span in the XFV-05
convergence gate is allowed in addition to its diagnostic use; it does not
resolve the first-`k` endpoint.
Apply the frozen B6 matching method only when its preconditions hold. For new
fixtures, matching sign groups are the full-spectrum partition labels
`negative`, `zeroish`, and `positive` defined in XFV-05; do not apply per-route
spectral-norm normalization. Cluster gaps are scale-invariant, so the cluster
rule above is unchanged. The B6 raw-sign assignment remains a historical B6
record. Withhold individual labels on cluster, cardinality, or sign ambiguity.
Record the light-cluster negative-eigenvalue count by reading the full-spectrum
partition above; do not recompute its tolerance on the light subset. Only
basis/gauge-invariant aggregate light-sector tensor measures enter the
cross-fixture synthesis; raw individual `lambda31` signs are not physical
evidence. Record dimensionless scale-aware eigenpair/generalized residuals,
conditioning, units, basis, metric, normalization, order, and conversion rules.
These metadata must identify the fixed canonical common-scaling convention
above; recording them does not permit a route-specific normalization.

**XFV-07 — Endpoint is total.** Every selected new fixture gets exactly one of
`DIVERGENT`, `NON_DIVERGENT`, or `UNRESOLVED`. `UNRESOLVED` applies when the HP
reference or signs remain unresolved or a required route fails before endpoint
metrics exist, any required endpoint quantity is nonfinite or undefined, a
required residual fails its bound, or a deterministic cluster crosses the
first-`k` boundary. An XFV-05 `S` larger than `1..k` in the status-deciding comparison
(128/256 without fallback or 256/512 after fallback) is also cutoff-crossing,
even if the HP-reference point in that comparison alone has no cutoff-spanning
cluster. In either cutoff-crossing case, the
endpoint remains `UNRESOLVED` even if the widened HP convergence comparison
passes. Exclude only the associated basis-dependent subspace-discrepancy
response from its Spearman pair; the mass response stays available under its
existing availability and finite-response rules. No unresolved comparison can
satisfy either resolved
endpoint. For a converged reference with defined finite quantities and no
cutoff-spanning cluster, mark `DIVERGENT` if any of these predeclared diagnostic
conditions holds: maximum sorted light-mass delta `>1e-6` dex; light-subspace
K-induced projector Frobenius distance
`||R(P_F64-P_HP)R⁻¹||_F >1e-8` at HP-reference precision `p`, with
`K=K_HP[p]=promote(K64)=RᵀR` and `R=cholesky(K).U`; or Float64 and
sign-stable HP light-sector negative-eigenvalue counts differ, using the light counts
inherited from the full-spectrum partition above. Otherwise mark
`NON_DIVERGENT`. Report the three components separately. These thresholds are
study labels, not production tolerances. No unresolved fixture is dropped.

## Explanatory measurements and causal limits

**XFV-08 — Preregistered predictors.** Record the following Float64-available
candidate predictors before consulting HP results:

| Name | Definition and unavailable rule |
| --- | --- |
| `log10_condition_number` | `log10(cond2(W64))` for finite positive 2-norm condition; retain `+Inf`; otherwise `UNAVAILABLE`. |
| `log10_cancellation_index` | For each Float64 assembled entry, `N=sum(abs(term_j))` over its exact rank-one terms and `D=abs(final_entry)`. Ratio `R_entry=1` for `N=D=0`, `+Inf` for `N>0,D=0`, otherwise `N/D` for finite `N,D` and `D>0`; all other cases `UNAVAILABLE`. If any entry unavailable, fixture predictor unavailable; else `log10(max(1,max_entry R_entry))`, retaining `+Inf`. |
| `log10_heavy_light_hierarchy` | `log10(m_(k+1)/max_(i<=k)m_i)` when both masses are finite, positive and the first mass above `k` exists; retain a `+Inf` ratio; otherwise `UNAVAILABLE`. |
| `dimension` | Exact positive canonical Hessian dimension. |

HP-dependent assembly differences, HP solves and residuals are explanatory
measurements only. A later production escalation rule would need predictors
available before consulting HP and a separate owner decision.

**XFV-09 — Assembly versus solve.** On every converged fixture, compare (A)
the standard Float64 route with an HP eigensolve of the *exact promoted
Float64-assembled canonical matrix*, without reassembly, and (B) that
promoted-matrix HP solve with HP assembly/solve from the same exact frozen
Float64 physical inputs promoted directly, holding the whitening factor fixed
at `C_HP=promote(C64)`. Use directly promoted `Ltilde` and exact `Qtilde` at
each executed HP ladder point. This uses the same fixed-`C` assembly
construction as the B6 `E_inf` assembly; it does not use the `C_K` HP
reference route and isolates assembly precision alone. Report both continuous
light-sector discrepancy legs separately; they are descriptive only and are
not assumed additive.
For divergence-flagged fixtures also perform the frozen B6 fixed-input
accumulation/product-order sensitivity checks. Do not assign
`assembly-dominant`, `solver-amplified`, or other categorical causal counts
without a separately preregistered successor.

## Discovery, scalar boundary, and sealed holdout

**XFV-10 — Discovery.** Report all 20 fixture metrics, endpoint/convergence
states, predictor availability and prior exposure. Report descriptive Spearman
rank correlations between each available predictor and continuous mass and
subspace discrepancies, with exact contributing `n`; report both continuous
assembly/solve legs per fixture. For `HP_SIGN_UNRESOLVED`, any 128/256 comparison failure triggers the 512
run, so the status-deciding comparison is 256/512; define mass and subspace
responses against 512. If XFV-05 `S` in that comparison is larger than
`1..k`, the endpoint remains `UNRESOLVED`, only the subspace response is
`UNRESOLVED` and excluded from its Spearman pair, and the mass response
remains available under its existing availability and finite-response filters.
The sign/negative-count response is `UNRESOLVED` and excluded from Spearman. For
`HP_REFERENCE_UNRESOLVED`, all responses are unavailable and excluded from
Spearman. Retain both fixture types in disposition accounting. For each
predictor/response pair, include only
observations where the predictor is available under its preregistered rule and
the response discrepancy is finite. Retain `+Inf` only for predictors whose
definition explicitly allows it, ranked above every finite value. Exclude
`UNAVAILABLE`, `NaN`, `-Inf` unless a future preregistered predictor explicitly
retains it, and every nonfinite response. Assign midranks to ties and compute
Spearman rho as the Pearson correlation of the included ranks. Report the exact
pairwise-filtered `n` for each coefficient. If `n<3`, report
`UNDEFINED_N_LT_3`; if the predictor has fewer than two distinct ranked values,
report `UNDEFINED_CONSTANT_PREDICTOR`; and if the response has fewer than two
distinct ranked values, report `UNDEFINED_CONSTANT_RESPONSE`. Evaluate these
rules independently and report every applicable status when conditions
overlap. Do not impute. For a cutoff-crossing fixture under XFV-06 or the XFV-05
`S` rule, take `S` from the status-deciding comparison (128/256 without
fallback or 256/512 after fallback); this includes `S` larger than `1..k` at
256/512 for `HP_SIGN_UNRESOLVED`. Exclude only its `UNRESOLVED` basis-dependent
subspace-discrepancy response from that Spearman pair. Retain the mass response under
its existing availability and finite-response filters; evaluate the fixture's
other responses under their existing pairwise availability and finite-response
filters. These correlations are descriptive
only and confer no production or causal authority.
 Boundary fitting
requires all 20 binary resolved endpoints, both endpoint classes nonempty, and
a candidate predictor available for all 20. Otherwise freeze
`NO_SIMPLE_SCALAR_BOUNDARY_INCONCLUSIVE` with reason. A minimum of 24/30
converged HP references permits descriptive cross-fixture synthesis only and
does not waive this complete-discovery prerequisite.

**XFV-11 — Polarity-aware adjacent partitions.** For each eligible predictor,
sort its 20 extended-real values and enumerate adjacent *distinct* values
`vi<vj`. Each pair yields two candidates: `LE` predicts `DIVERGENT` for
`x<=vi`; `GE` predicts `DIVERGENT` for `x>=vj`. Equality is on the divergent
side. These observed-value partitions work for `-Inf`, `+Inf`, and adjacent
binary64 values; an arithmetic midpoint is not the partition identity.
Compute sensitivity `TP/(TP+FN)` and specificity `TN/(TN+FP)`, with the
nonempty class prerequisites. Qualify a candidate only at exact sensitivity
`1.0` and specificity `>=0.80`. Choose highest specificity, then
lexicographically smallest predictor name, then polarity `GE` before `LE`,
then numerically smallest threshold in extended-real order. Here the
threshold is the selected observed `vi` for `LE` or `vj` for `GE`. Freeze the
exact predictor, transform, polarity, threshold, pair/partition, confusion
matrix, class denominators, endpoint counts, and analysis code SHA-256 before
unsealing holdout. If none qualifies, freeze `NO_SIMPLE_SCALAR_BOUNDARY`.

**XFV-12 — Holdout.** Keep route outcomes sealed from discovery/model selection
until the discovery summary and exact rule or negative status are frozen. Apply
the frozen rule to all ten without retuning. Report all endpoint states,
predictor values/availability, exposure flags, confusion matrix, sensitivity,
specificity, positive/negative denominators, and 95% Wilson binomial intervals
where defined. `SUPPORTED_ON_FIXED_PREREGISTERED_HOLDOUT` requires all ten
binary resolved endpoints, predictor available on all ten, no relevant prior
holdout outcome exposure, both classes nonempty, sensitivity exactly `1.0`,
and specificity `>=0.80`. Otherwise return
`NO_VALIDATED_SIMPLE_SCALAR_BOUNDARY_INCONCLUSIVE` with exact reason (including
`HOLDOUT_PRIOR_EXPOSURE_INCONCLUSIVE` where applicable). Retain all ten in the
table and denominator accounting. Even support on this fixed holdout is not a
general regime boundary or production policy.

## Evidence and stopping gates

**XFV-13 — Synthesis and reproducibility.** At least 24 of the 30 new fixtures
must reach `HP_REFERENCE_CONVERGED` for bounded descriptive synthesis. Below
that, return `INSUFFICIENT_REFERENCE_CONVERGENCE` with all 30 dispositions; do
not replace outcome-bearing fixtures or claim a boundary. Preserve raw
per-fixture inputs/metrics/statuses, formulas, thresholds, selection/analysis
code identities, exact source and environment/dependency identities, precision
readbacks, seed/rule, chronology and digests of manifest/discovery/holdout
freezes, deterministic rebuild evidence, and separate B6 control evidence.
Public evidence must use repository-relative references and omit credentials,
machine-local paths and private context. Configured source paths alone are not
provenance. Any source/input drift, manifest failure, B6 control failure,
holdout peek, outcome-based replacement, precision-state restoration failure,
or need for out-of-scope mutation stops the later study at its stated gate.

**XFV-14 — Final authority.** A later study candidate must be frozen at an
exact commit/tree and receive fresh independent Spec and Scientific review of
the same exact bytes. Review findings and unresolved claims remain visible.
Any proposed production route or precision policy, physical interpretation,
source correction, migration, merge, or Issue #173 closure requires a separate
owner decision and its own governing authority. This contract preparation
changes no package version, API, scientific data schema, or production code.
