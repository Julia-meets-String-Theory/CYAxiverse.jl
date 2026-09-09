# Issue 148 G0 baseline and scientific-contract audit

Independent-review recommendation: **G0 audit completeness PASS; existing N=5
validation FAIL**.  The apparent N=5 critical-scale discrepancy is a
source-fixed, reproducible implementation/validation defect, not an
unresolved scientific convention.  Passing G0 records and classifies that
defect; it does not validate or repair the affected N=5 implementation.  This
audit changes no scientific or physical convention.

Manager decision (2026-09-09): **G0 PASS**, following a Sol/xhigh scientific
audit and a fresh Sol/xhigh independent review. All Issue 148 G0 acceptance
items are covered below. Existing N=5 validation remains defective and is not
certified by this gate. No production implementation or scientific convention
was changed. G1 is proposed only, not started. Version impact: none for this
audit-only change; no package release readiness is claimed.

The replay script intentionally detects the defect at the audited baseline
SHA. It is historical audit evidence, not a regression requirement to retain
the defect. After G1 repairs it, replay this audit at its recorded revision;
G1 must supply source-faithful regression coverage for the corrected behavior.

## Reproducibility envelope

- Repository state: `3a5b034fbbabce6638607373e682cbd6f66f6aaa`
  (`vmm` at the start of the audit).
- Scope and acceptance source: GitHub Issue 148, retrieved in full on
  2026-09-09.
- Paper source: *Catastrophic Inflation in the Axiverse*, arXiv:2608.14780v1.
- Source PDF SHA-256:
  `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
  This equals `PAPER_SOURCE_IDENTITY.source_sha256` in
  [`catastrophe_diagnostics.jl`](../../src/paper_benchmarks/catastrophe_diagnostics.jl).
- Julia: 1.12.6, Darwin arm64.  The project was instantiated before the
  focused checks; the generated `Manifest.toml` is ignored and is not part of
  this evidence.

## Source facts

These statements come from arXiv:2608.14780v1 rather than from repository
behavior.

1. The paper's radial control is
   `tau^i(k) = k tau^i(k=1)` (Eq. 25), where the `tau^i` are divisor/four-cycle
   volumes at the stretched-cone tip.  Thus each geometry has its own `k=1`
   reference point.
2. The source axions `theta` have period one and enter the potential as
   `cos(2pi Q theta + delta)` (Eqs. 20 and 75).  The additive phases `delta`
   are radians.  The kinetic term is `K_ij dtheta^i dtheta^j / 2` (Eqs. 22 and
   23).
3. For N=5, the instanton actions divided by `2pi` are
   `[6, 6.25, 24, 26, 31.875, 32, 36.125, 162.125]` (Sec. 4.1).  The four
   dominant terms leave the light direction
   `2pi theta_light = (0,0,1,2,0) vartheta` (Eq. 28).  On that direction the
   leading reduced potential is
   `1-cos(vartheta) + a(k)(1-cos(2vartheta))`, with
   `a(k) = (32/(255/8)) exp[-2pi k (32-255/8)]` (Eqs. 31--34).
   Its catastrophe is therefore
   `a(k_c)=1/4` and
   `k_c=(4/pi) log(1024/255)=1.7700681326109957` (Eq. 36).  There are two
   minima below and one above the cusp.  The N=5 phase-detuned source example
   applies `delta=pi/4` to the second reduced cosine.
4. For N=8, Table 1 has 12 displayed actions
   `[14,14.5,14.5,15.5,15.5,15.5,15.5,16,17,17,25,45]`.
   The zero-phase catastrophe is `k_c=0.674506370003365` (Fig. 10), with five
   minima below and one above.  The phase-detuned source example assigns
   `delta=0.04` to the second row of Table 1 and shifts the catastrophe to about
   0.5.  The separate downloaded author trajectory code, rather than the paper,
   retains the 10 leading rows.
5. In the current paper version, the N=5 and N=8 geometries are Appendices C
   and D, respectively.  Several repository labels still say B and C.
6. The source treats the radial scan as a fixed-saxion benchmark family.  It
   does not establish moduli stabilization along the family.  Its stated
   perturbative-control checks include prime-divisor and potent-curve volumes.

## Implementation facts

The following are facts about repository state at the audited commit.

### Benchmarks and augmented solvers

- [`reduced_models.jl`](../../src/paper_benchmarks/reduced_models.jl) implements
  the source N=5 ratio and closed-form critical scale.  Its N=8 augmented
  solve uses the 12 Table-1 terms, leading-charge period-one coordinates, and
  solves `grad V=0`, `H v=0`, and `v'v=1` with a hierarchy-scaled residual.
  It reproduces the N=8 source value.
- [`poly102_inflation.jl`](../../src/paper_benchmarks/poly102_inflation.jl)
  implements the executable-author-code N=8 convention: 10 retained terms,
  raw GLSM angles in radians, `K(k)=K(1)/k^2`, and the canonical Hessian.  Its
  augmented N=8 solve also reproduces the source value.
- The same file incorrectly defines `n5_critical_scale()` as the N=8 constant
  and defines the N=5 ratio as
  `0.25 exp[-2pi(k-k_c)(32-255/8)]`.  That reanchoring creates an artificial
  N=5 cusp at the N=8 value and is not the ratio obtained from its own N=5
  actions.
- [`compatibility.jl`](../../src/paper_benchmarks/compatibility.jl) does not alias
  the root N=5 critical-scale and ratio functions to the `poly102_inflation`
  versions.  Consequently, two public namespaces return different values:
  the root namespace returns the paper value and `poly102_inflation` returns
  the N=8 value.
- [`inflation_scale_continuation.jl`](../../scripts/inflation_scale_continuation.jl)
  selects `poly102_inflation` in `pilot_benchmark_regression` and declares the
  N=5 check successful only when its critical scale equals the N=8 value.
  [`inflation_reproduction.jl`](../../scripts/inflation_reproduction.jl),
  [`catastrophe_diagnostics.jl`](../../src/paper_benchmarks/catastrophe_diagnostics.jl),
  and [`catastrophic_inflation_population_study.jl`](../../scripts/catastrophic_inflation_population_study.jl)
  inherit this synthetic N=5 fixture.  The existing reproduction result
  therefore reports 0.674506 for N=5.

### Current continuation and matching behavior

- The generic physical scale path in
  [`inflation_scale_continuation.jl`](../../scripts/inflation_scale_continuation.jl)
  applies `tau -> k tau`, `Kinv -> k^2 Kinv`, full-normalization
  `V_CY -> k^(3/2) V_CY`, and recomputes coefficients.  The fixed-volume path
  is explicitly diagnostic/unsupported.  A scaling certificate is required.
- Its separate `pilot_homotopy_scale` multiplies the complete `log10`
  amplitude row by a scalar while leaving `K` fixed.  This sends
  `|A_i| -> |A_i|^scale`, including prefactors.  It is correctly labelled
  `homotopy_only` and is not the paper's radial `k`.
- At each sampled scale, the branch sampler corrects every original reference
  seed independently.  It does not predict from the previously corrected
  point.  Branch identities are then assigned post hoc between adjacent
  slices by greedy one-to-one matching of periodic `L-infinity` distance
  (default threshold 0.1).  Unmatched records become `lost` or `new`.
- Candidate crossings are identified from a sign change or near-zero smallest
  generalized-Hessian eigenvalue, optionally combined with a minima-count
  change.  This is a candidate detector, not a catastrophe certificate.
- The optional generic augmented solver uses state `[theta,v,log(k)]` and
  solves the package-coordinate gradient, a scaled raw `H v`, and Euclidean
  `v'v=1`.  It reports generalized eigenvalues with `K`.  It is not integrated
  into automatic branch continuation.
- [`phase_volume_detuning_scan.jl`](../../scripts/phase_volume_detuning_scan.jl)
  has another homotopy parameter:
  `A_i(k)=L[i,1] 10^(k L[i,2])`.  Its phases are cycles inside
  `2pi(Q theta+phase)`.  Its `refine_catastrophe` bisects a Hessian zero at a
  fixed supplied `theta`; it does not jointly solve stationarity and
  degeneracy and is not an augmented catastrophe solver.

## Scale and coordinate dictionary

| Name/context | Precise meaning | Scientific status |
|---|---|---|
| Paper radial `k` | `tau_four-cycle(k)=k tau(1)` for that geometry | Source physical benchmark coordinate |
| Generic `reference_scale` | Fixed label `1.0` for the supplied reference data | Equals the paper's `k=1` only for a source-identified paper fixture |
| Generic `sampled_scale` | Multiplier applied to the supplied reference data | Physical only after certificate/full map |
| `pilot_homotopy_scale` | Exponent on the full absolute amplitude | Numerical homotopy only |
| Phase-volume scan `k` | Multiplier of the exponent row in `L` | Numerical homotopy only |
| Augmented `log(k)` | Positivity-preserving variable for the generic certified scale path | Same `k` after exponentiation |
| N=5 reduced `vartheta` | Radian angle of period `2pi` on the source light direction | Source reduced coordinate |
| Generic package `theta` | Period-one coordinates; charges stored as axions by instantons | Package convention |
| Source charge tables | Instantons by axions | Transposed on package ingestion |
| Benchmark phases | Additive radians in `2pi Q theta + delta` | Source convention |
| Detuning-scan phases | Cycles in `2pi(Q theta + phase)` | Different documented implementation convention |

The homogeneous four-cycle map implies two-cycle Kahler coordinates scale as
`sqrt(k)`, total Calabi--Yau volume as `k^(3/2)`, `Kinv` as `k^2`, and `K` as
`k^-2`.  This implication and the current generic implementation agree.

The main N=8 solver uses period-one coordinates.  The executable-author-code
N=8 fixture uses raw radian angles but retains the paper's numerical metric
matrix and explicitly labels this `metric_convention=:raw_angles_radians`.
A literal coordinate conversion from the paper's period-one Eq. 22 would also
rescale the metric by `(2pi)^-2`.  This mismatch does not alter the location of
the N=8 degeneracy under a constant nonsingular metric scaling.  No convention
is changed here; the choice must be made explicitly before later physical
canonical-field or observable claims.  It does not block a source-scalar N=5
G1 task.

Numerical precision is also two-layered.  Generic branch correction currently
uses Float64 arithmetic (53 bits) after a scaling-domain audit may evaluate
inputs at at least 128-bit precision.  Catastrophe diagnostics can rerun at 120
bits, but they cast the stored Float64 source arrays upward; this improves
arithmetic precision without restoring extra source digits.

## N=5 discrepancy: reproduced classification

At the audited commit:

```text
root/source n5 critical scale       1.7700681326109957
poly102 n5 critical scale           0.674506370003365
difference                          1.0955617626076308
source ratio at source scale        0.25
source ratio at poly102 scale       0.5910573869337175
source curvature at theta=pi there  1.3642295477348698
poly102 ratio at source scale       0.105742693318219
poly102 curvature there            -0.577029226727124
```

The paper formula has zero residual at 1.7700681326109957 and changes from two
minima below to one above.  The 0.674506 value is the N=8 constant.  Although
one can algebraically call the reanchored variable a shifted coordinate, no
implementation consistently shifts the divisor volumes, raw N=5 potential,
metric, or geometry reference point.  Both APIs instead call it the same
radial scale.  The source and repository data therefore rule out two
established scale conventions.  The discrepancy is a **source-fixed
implementation/validation defect**.

The default N=5 catastrophe diagnostic returns `:cusp` at the artificial
0.674506 anchor.  At the source scale it returns `:unresolved` with projected
near-null curvature about `-0.577`, because it consumes the reanchored ratio.
This confirms that the affected validation is self-consistent with the
fixture, not with the raw/source N=5 model.

## Exact checks and observed outcomes

1. `julia --startup-file=no --project=. scripts/inflation_scale_continuation.jl --benchmarks-only`
   exited 0.  It reported N=5 and N=8 at 0.674506 and `passed=true`; the N=8
   augmented residuals were approximately `2.02e-15` (gradient) and
   `6.26e-13` (null equation).  This is evidence of the defective acceptance
   check, not source validation for N=5.
2. `python3 scripts/agent_verify.py run -- julia --startup-file=no --project=. scripts/audit_issue_148_g0.jl`
   evaluated both namespaces and the raw reduced curvature, producing the
   discrepancy table above.  It also found the root 12-term N=8 augmented
   solution at
   `k=0.6745063700033533`, gradient residual `2.49e-15`, null residual
   `4.78e-15`, and first two eigenvalues of its hierarchy-preconditioned
   Hessian approximately `6.09e-16` and `1.4233`.  These are not the kinetic-
   metric generalized eigenvalues.
3. The executable-author-code 10-term N=8 solution returned
   `k=0.674506370003365`, gradient residual `2.02e-15`, null residual
   `6.26e-13`, and a positive first heavy eigenvalue.
4. The stored N=5 and N=8 metric eigenvalues were positive and matched the
   source tables at the stored precision.
5. `python3 scripts/agent_verify.py run -- julia --startup-file=no --project=. scripts/inflation_scale_continuation.jl --benchmarks-only`
   passed.  `agent_verify snapshot` and `agent_verify diff-check` passed; a
   direct trailing-whitespace scan also covered the two untracked audit files.
6. Downloaded author artifacts matched the repository hashes: CYTools scan
   `d820dd3e...f4`, `poly102_settings.wl`
   `2c49a27a...9fcc`, and `poly102_core.wl` `558df6893...ffd0`.  The settings
   identify 0.674506 as the N=8 catastrophe, retain 10 terms, use raw radian
   angles, and scale the metric as `k^-2`.  Separate collaborator artifacts
   `analytic_catastrophe_checks.py` (`bdd8ce2f...707c`) and
   `classify_verified_catastrophes.py` (`be82736f...8849`) identify the N=5
   geometry `(h11=5, polytope=2787)` with
   `k_c=1.7700681326109957`; no N=5 trajectory code analogous to the poly-102
   Mathematica program was identified.

## Independent-review adjudication

The independent review read the full Issue 148 text, rehashed and inspected the
35-page source PDF, traced both N=5 namespaces to their raw action data, and
replayed both N=8 augmented solvers.  The N=5 defect classification does not
depend on choosing between coordinate conventions: the raw `q dot tau`
prefactors and exponential actions in `poly102_inflation.n5_potential` produce
the paper ratio, while the neighboring `n5_reduced_ratio` helper produces a
different ratio at the same named `k`.  The paper, the root implementation, and
two collaborator diagnostics all independently select the 1.770068 value.

Review corrections to the initial audit draft were substantive but did not
change its verdict: the source-hash constant lives in
`catastrophe_diagnostics.jl`, the 10-row N=8 convention comes from downloaded
author code rather than the paper, and the 12-term solver's reported spectrum
is hierarchy-preconditioned rather than kinetic-metric generalized.  The G1
packet was narrowed to Issue 148's known regular branch; exhaustive stationary-
branch coverage, continuation through the cusp, and unrelated label cleanup
are not G1 requirements.

The remaining metric-coordinate mismatch is explicitly bounded away from G0
and source-scalar N=5 G1.  It must be resolved before a later claim about
canonically normalized N=8 distance or an observable.  The audit establishes
no moduli stabilization, complete catastrophe population, off-ray result, or
physical inflation claim.

## G0 acceptance adjudication

| Issue 148 G0 item | Evidence/result |
|---|---|
| Radial/scale continuation | Physical certified map and both nonphysical homotopies distinguished above |
| Post-hoc matching | Reference-seed correction plus adjacent-slice greedy matcher identified |
| Augmented catastrophe solver | Generic, 12-term N=8, 10-term author N=8, and fixed-theta pseudo-refiner distinguished |
| N=5 and N=8 sources | Paper formulas/tables and 12-term versus author 10-term N=8 conventions recorded |
| Meanings of scale | Dictionary records all relevant scale variables |
| N=5 critical-scale inconsistency | Reproduced and classified as implementation/validation defect |
| Precision/basis/phase/metric | Explicitly recorded, including the later metric decision boundary |
| Physical scaling and claims | Homogeneous map recorded; fixed-saxion/no-stabilization claim boundary retained |

No scientific-owner decision is needed to pass G0 or to begin a zero-phase,
source-reduced N=5 G1.  The known regular branch must be continued toward and
used to recover the cusp; Issue 148 does not require exhaustive continuation
of every stationary branch or continuation through the cusp.  A later task
that claims canonically normalized N=8
distances or observables must decide whether the literal paper period-one
metric transformation or the executable-author-code raw-angle convention is
authoritative.

## Proposed bounded G1 worker packet

**Objective:** Repair the N=5 benchmark defect and implement source-faithful,
zero-phase N=5 continuation of the known regular critical branch toward and to
the analytic cusp.  Do no N=8 or off-ray work.

**Acceptance:**

1. Establish one canonical N=5 radial `k`, ratio, and critical scale consistent
   with the raw actions and Eq. 36; reject the old 0.674506 fixture.
2. Continue the known regular zero-phase N=5 critical branch using the previous
   corrected state with a genuine predictor/corrector or pseudo-arclength
   method.  Carry branch identity in continuation state; do not assign the
   result by post-hoc nearest-neighbor matching.  Exhaustive stationary-branch
   coverage and continuation through the cusp are outside this gate.
3. Recover `k_c=1.7700681326109957`, stationarity, the vanishing Hessian mode,
   and the two-to-one minima change to stated tolerances.  Compare against the
   closed form.
4. Add focused regression checks that fail for the old N=8 reuse and exercise
   a documented continuation failure boundary.  Update only directly affected
   N=5 validation evidence; unrelated stale source labels are outside scope.

**Inputs:** Issue 148; arXiv:2608.14780v1 and its hash above; this G0 audit;
`reduced_models.jl`, `poly102_inflation.jl`,
`inflation_scale_continuation.jl`, its focused tests, and directly affected N=5
validation artifacts.

**Constraints:** Zero phase and N=5 only.  Preserve the source period-one GLSM
coordinates, the reduced `vartheta` mapping, and existing physical
normalization.  Do not change basis, metric, phase, observable, or physical
claim conventions.  Do not conflate either homotopy parameter with radial
`k`.  State arithmetic precision and tolerances.  Preserve the public API
shape while applying the required source-value correction.

**Worker ownership:** Own the ordinary investigate, diagnose, correct,
implement, verify, and recheck loop, including routine numerical and test
failures.

**Escalation:** Correcting the known-wrong N=5 benchmark value is required even
though it changes that public result.  Escalate a proposed interface-shape
break or a change to radial scale, Kahler coordinates, basis, metric, phase,
observable, catastrophe acceptance, or physical claim boundary.  Do not tune a
scientific convention to make a benchmark pass.

**Lease:** 20 minutes; one short checkpoint only at actual expiry, followed by
one evidence-based extension when making progress.

Recommended worker: bounded **IMPLEMENTATION**, Luna/max or a comparably strong
Julia coding specialist.  Follow with fresh Sol/high--xhigh
**INDEPENDENT_REVIEW** because the benchmark correction is scientifically
material.
