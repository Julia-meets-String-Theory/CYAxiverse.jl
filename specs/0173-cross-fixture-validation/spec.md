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
outcome inspection, the exact `spec.md`, `plan.md`, and `tasks.md` bytes must
receive fresh independent Spec and Scientific `PASS`, followed by a
separate final S2 owner approval tied to the reviewed revision by `approval_ref`.
A later execution handoff must bind those exact documents and that approval.
The present contract preparation stops at the owner approval gate.

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
`r=N mod 5`; partition the sorted list into five ordered contiguous blocks,
with the first `r` blocks of size `q+1` and the remaining blocks of size `q`.
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

1. Standard Float64 first-three log10 masses within `1e-12` of
   `[13.860912539123262, 14.073585277123616, 14.230434590975188]`, signs
   `[-1,+1,+1]`.
2. HP first-three signs `[+1,+1,+1]` at 128, 256 and 512 decimal digits;
   128/256 and 256/512 first-three sorted-mass maximum deltas at most `1e-8`
   dex.
3. HP solve of the promoted Float64 assembled matrix: first-three masses
   within `1e-12` dex of
   `[13.238179298077366, 13.843308526910718, 13.998657518779794]`, signs
   `[+1,+1,-1]`.
4. Float64/HP three-lowest-mode projector Frobenius distance at most `1e-12`;
   HP versus Float64 assembly relative *infinity-norm* matrix error, as in
   the frozen B6 diagnostic, at most `1e-15`; HP
   canonical-Hessian 2-norm condition number at least `1e70`.
5. Aggregate three-lowest-mode quartic-tensor log10 Frobenius absolute delta
   at most `1e-12` where defined. The frozen fixed-input reduction/product-order
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
aggregate light-tensor log10 Frobenius delta `<=1e-8`; and signed
classification stability. Signed stability requires equal negative-eigenvalue
counts and equal positive/negative/zeroish sign-group cardinalities under the
frozen B6 matching declaration; ambiguous clusters do not acquire individual
identity. If these fail, run 512 and apply the same 256/512 checks. If only
sign stability fails, record `HP_SIGN_UNRESOLVED`; if any other convergence
check fails, record `HP_REFERENCE_UNRESOLVED`. A failed route is separately
recorded. Read back requested/actual precision at every ladder point and
restore global ArbFloat precision after each fixture, including failures.
No tachyon/stability or route sign-disagreement claim may use an unresolved HP
reference.

**XFV-06 — Basis-invariant comparison.** Compare the sorted multiset of the
`k=min(3,dimension)` lowest physical masses in log10 units. This is a set
diagnostic, not one-to-one mode identity. Compare their metric-consistent
canonical light subspaces using projector Frobenius distance and principal
angles. Apply the frozen B6 deterministic cluster/matching method only when
its preconditions hold; withhold individual labels on cluster, cardinality, or
sign ambiguity. Record the light-cluster negative-eigenvalue count separately.
Only basis/gauge-invariant aggregate light-sector tensor measures enter the
cross-fixture synthesis; raw individual `lambda31` signs are not physical
evidence. Record dimensionless scale-aware eigenpair/generalized residuals,
conditioning, units, basis, metric, normalization, order, and conversion rules.

**XFV-07 — Endpoint is total.** Every selected new fixture gets exactly one of
`DIVERGENT`, `NON_DIVERGENT`, or `UNRESOLVED`. `UNRESOLVED` applies when the HP
reference or signs remain unresolved or a required route fails before endpoint
metrics exist. For a converged reference, mark `DIVERGENT` if any of these
predeclared diagnostic conditions holds: maximum sorted light-mass delta
`>1e-6` dex; light-subspace projector Frobenius distance `>1e-8`; or Float64
and sign-stable HP light-sector negative-eigenvalue counts differ. Otherwise
mark `NON_DIVERGENT`. Report the three components separately. These thresholds
are study labels, not production tolerances. No unresolved fixture is dropped.

## Explanatory measurements and causal limits

**XFV-08 — Preregistered predictors.** Record the following Float64-available
candidate predictors before consulting HP results:

| Name | Definition and unavailable rule |
| --- | --- |
| `log10_condition_number` | `log10(cond2(W64))` for finite positive 2-norm condition; retain `+Inf`; otherwise `UNAVAILABLE`. |
| `log10_cancellation_index` | For each Float64 assembled entry, `N=sum(abs(term_j))` over its exact rank-one terms and `D=abs(final_entry)`. Ratio `r=1` for `N=D=0`, `+Inf` for `N>0,D=0`, otherwise `N/D` for finite `N,D` and `D>0`; all other cases `UNAVAILABLE`. If any entry unavailable, fixture predictor unavailable; else `log10(max(1,max_entry r))`, retaining `+Inf`. |
| `log10_heavy_light_hierarchy` | `log10(m_(k+1)/max_(i<=k)m_i)` when both masses are finite, positive and the first mass above `k` exists; retain a `+Inf` ratio; otherwise `UNAVAILABLE`. |
| `dimension` | Exact positive canonical Hessian dimension. |

HP-dependent assembly differences, HP solves and residuals are explanatory
measurements only. A later production escalation rule would need predictors
available before consulting HP and a separate owner decision.

**XFV-09 — Assembly versus solve.** On every converged fixture, compare (A)
the standard Float64 route with an HP eigensolve of the *exact promoted
Float64-assembled canonical matrix*, without reassembly, and (B) that
promoted-matrix HP solve with HP assembly/solve from the *same exact frozen
Float64 physical inputs promoted directly*. Report both continuous
light-sector discrepancy legs separately; they are not assumed additive.
For divergence-flagged fixtures also perform the frozen B6 fixed-input
accumulation/product-order sensitivity checks. Do not assign
`assembly-dominant`, `solver-amplified`, or other categorical causal counts
without a separately preregistered successor.

## Discovery, scalar boundary, and sealed holdout

**XFV-10 — Discovery.** Report all 20 fixture metrics, endpoint/convergence
states, predictor availability and prior exposure. Report descriptive Spearman
rank correlations between each available predictor and continuous mass and
subspace discrepancies, with exact contributing `n`; report both continuous
assembly/solve legs per fixture. Boundary fitting requires all 20 binary
resolved endpoints, both endpoint classes nonempty, and a candidate predictor
available for all 20. Otherwise freeze
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
