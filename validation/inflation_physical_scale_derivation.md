# SCI-01: physical-scale derivation and release decision

Date: 2026-08-25
Status: owner-selected convention and independent fails-closed physical scaling
and physical control gates implemented; physical-scale production authorization
remains pending
Program: KS axiverse inflation only

## Authoritative source identity

The current primary paper is
`/Users/vmehta/Documents/CYAxiverse/cyaxiverse/catastrophicKS.pdf`,
SHA-256
`b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6`.
The PDF metadata identifies *Catastrophic Inflation in the Axiverse* by Naomi
Gendler, Oliver Janssen, Matthew Kleban, and Cameron Norton; it has 35 pages.
All page references below give the printed paper page and, where useful, the
corresponding PDF page rendered during the read-only review.

The generic Julia continuation inspected for this decision is
`scripts/inflation_scale_continuation.jl`, whose pre-edit SHA-256 was
`b144b377d7f2246e25bb516a908c5d81ddb3ef0d6e491ae62e0d463a3774fc97`.
The author implementation inspected for the coefficient map is
`/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CN_Axiverse_code/ks_axiverse_python_collaborator/src/cytools_catastrophe_scan.py`,
SHA-256
`d820dd3e19d2833bac0691d74c2f99d2461c8eb0ef1620062f70d3daffd3bcf4`.
The other archived author paths named by the earlier ledger were not present
at their recorded locations during this review, so no claim below depends on
those files.

## Current scientific decision

The owner selected the homogeneous overall Kähler convention

```text
V_CY(k) = k^(3/2) V_CY(1),
tau(k)  = k tau(1),
K(k)    = k^(-2) K(1),
Kinv(k) = k^2 Kinv(1),
Q(k)    = Q(1).
```

This resolves the earlier fixed-volume versus full-volume choice for the
physical path. The exponent `3/2` is the owner's homogeneous geometric
selection: scaling the four-cycle volumes by `k` corresponds to scaling the
two-cycle coordinates by `sqrt(k)` and the cubic Calabi--Yau volume by
`k^(3/2)`. The paper explicitly supplies the divisor-volume path in Eq. (25)
but does not print this exponent as a separate equation. Therefore the
homogeneous law is recorded as an owner decision, not misquoted as a direct
paper equation.

The owner also selected the exact scientific unit contract

    M_s=M_Pl;k=dimensionless

The typed certificate requires this exact two-clause string: the first clause
sets the string scale equal to the Planck scale, and the second keeps the
continuation parameter k dimensionless.
An arbitrary nonempty units label is out_of_model; missing or empty units
remain missing_evidence. The string records the implementation convention
without assigning Planck units to k. It does not remove the paper's
distinction between geometric string-unit control cuts and the M_Pl factors
printed in the inflation equations.

The fixed-`V_CY` path remains an author-code comparison diagnostic only. It
must not emit `scale_status=physical`. Historical logarithmic stretching
remains `scale_status=homotopy_only`; it must not be relabelled by this
decision.

This decision does not establish moduli stabilization. The paper states that
it does not undertake moduli stabilization, uses fixed saxion values, and
leaves dynamical stabilization at those points open (printed p. 6, PDF page 7,
after Eq. (16); printed p. 25, PDF page 26, Conclusions). The implementation
must retain `moduli_status=not_established` unless separate evidence proves
otherwise.

## Independent physical gates

The implementation now records two separate gates in every scale result,
branch row, summary row, and persisted inflation group. The gate statuses are
not inferred from one another, and only the literal status `passed` is a pass.

`physical_scaling_gate` must be `passed` before any physical scale calculation.
It covers complete geometry and domain evidence, the exact unit contract,
normalization, phase convention, basis and charge orientation, replayable
source/configuration provenance, numeric precision and conversion audit,
declared symmetry/inverse/SPD policy, positive volumes, and Kähler-domain
checks. A failed or missing scaling gate raises the existing fail-closed
physical-domain error; it cannot fall back to a diagnostic calculation.

`physical_control_gate` is an independent post/domain qualification gate. It
records potent-ray evidence, instanton control, perturbative control, moduli
control or stabilization, and visible-sector applicability. The diagnostic
pilot may record `not_established` and may calculate after a passed scaling
gate. This does not make the output physically viable, production-qualified,
or a validated candidate. A control status other than `passed` blocks all such
labels. A screen hit with both gates passed is recorded only as
`eligible_not_validated`; it is not a candidate proof.

Historical `homotopy_only` and fixed-volume comparison paths retain their
nonphysical statuses. Their two physical gates are explicitly
`not_applicable`, with reason and provenance, and those paths are not changed
into physical calculations by the new schema.

## Full-paper equations and qualifications

The source evidence reviewed for the normalization and domain boundary is:

| source location | evidence used | implementation consequence |
| --- | --- | --- |
| printed p. 6 (PDF p. 7), Eqs. (16)--(19) | `J` is in the Kähler cone; the Mori/effective divisor cones determine curve and instanton data; Eq. (19) contains `V_CY^(-2)`, divisor-volume exponentials, `Kinv` cross terms, integer effective-divisor charges, and phases from `W0`/`A_alpha` | Reconstruct coefficients from `tau`, `Kinv`, `V_CY`, `Q`, and phases. Do not scale composed logarithms. Require explicit cone/effective-divisor evidence. |
| printed p. 7 (PDF p. 8), Eqs. (20)--(21) and perturbative-control text | Prime toric divisor volumes must exceed one for the instanton expansion; potent Mori-cone curve volumes must exceed one as the stated proxy; additional corrections may remain | Record divisor/curve control evidence and fail closed when it is absent. |
| printed p. 8 (PDF p. 9), Eqs. (22)--(24) | `K_ij` is the axion kinetic metric derived from the Kähler potential; canonical fields depend on `K^(1/2)`; generic slow-roll parameters are large without cancellations | Compute the generalized Hessian in one declared basis and preserve the kinetic-metric convention. |
| printed p. 9 (PDF p. 10), Eq. (25) | The control path is `tau_i(k)=k tau_i(k=1)` from the stretched-cone tip; a change in critical-point count is a catastrophe diagnostic only after the path/domain checks | Keep scale, fixed-point, trajectory, and coverage statuses independent. |
| printed p. 10 (PDF p. 11), Eq. (26) and assumptions 1--7 | Slow-roll flow is `d phi_i/dN ~= -M_Pl^2 V^(-1) dV/dphi_i`; the examples assume heavy saxions, adjustable phases, prime-toric instantons, approximate de Sitter, negligible unknown corrections, and freely chosen initial conditions | Preserve assumptions as metadata. Do not turn them into a proof of stabilization, a measure, or a physical population claim. |
| printed pp. 13--14 (PDF pp. 14--15), Eqs. (31)--(37) | The N=5 example has `V_CY^(-2)` restored in its amplitude; its quoted density amplitudes are far below observation for zero phases, while nonzero phases unfold the cusp | Keep phase convention and absolute normalization explicit; benchmark numbers are not a generic physical validation. |
| printed pp. 18--23 (PDF pp. 19--24), Eqs. (38)--(40), Figs. 9--17 | The N=8 example quotes control checks, multifield slow-roll trajectories, turning, tilt, and `delta_H`; a phase `delta=0.04` changes the catastrophe scale and amplitude | A trajectory result requires its own status, tolerances, units, phase, and normalization; it is not implied by a fixed-point result. |
| printed p. 26 (PDF p. 27), Eqs. (49)--(51) | `delta_H ~= V^(3/2)/(5 sqrt(3 pi) M_Pl^3 |V'|)` and the quartic form explicitly depend on the absolute potential scale | A common potential factor `c` leaves normalized stationary/Hessian signs unchanged but changes `delta_H` by `c^(1/2)` when the shape and kinetic metric are held fixed. |
| printed pp. 29--31 (PDF pp. 30--32), Eqs. (86)--(89), (94)--(97), Table 1 | N=5 and N=8 benchmark geometries give explicit volumes, divisor data, curve ranges, charges, and kinetic eigenvalues | Use these as named benchmark evidence only; do not claim generic geometry validation from them. |

## Transformation table for the selected convention

The following table separates paper equations from the owner-selected
homogeneous law:

| quantity | selected physical path | source or qualification |
| --- | --- | --- |
| two-cycle coordinates `t` | `t(k)=sqrt(k) t(1)` | homogeneous consequence of the owner choice; not printed separately in the paper |
| divisor volumes `tau` | `k tau(1)` | paper Eq. (25) |
| `V_CY` | `k^(3/2) V_CY(1)` | owner-selected homogeneous convention; Eq. (19) supplies the explicit `V_CY^(-2)` factor |
| `K` and `Kinv` | `K -> k^(-2) K`, `Kinv -> k^2 Kinv` | selected metric convention; Eq. (23) defines the metric, and the scale law must be verified by the implementation |
| charges `Q` | unchanged integer columns in the fixed basis | Eq. (19) identifies integer effective-divisor charges |
| leading coefficient | reconstruct `V_CY^(-2) (q dot tau) exp(-2 pi q dot tau)` | Eq. (19); the exponential changes with `k`, so it is not a simple log stretch |
| pair/cross coefficient | reconstruct `V_CY^(-2) [pi q_i^T Kinv q_j + (q_i+q_j) dot tau] exp[-2 pi (q_i+q_j) dot tau]` | Eq. (19); the two terms in the bracket have different homogeneous powers before the common volume factor |
| phases | unchanged only when the selected phase convention says so; record zero or supplied nonzero phases | Eq. (19), assumptions on printed p. 10, and Appendix B Eq. (75) |
| Planck/string conversion | exact metadata contract is `M_s=M_Pl;k=dimensionless`; keep k dimensionless and do not infer an additional conversion from the paper's dimensionless fields | Eq. (26) and Eqs. (49)--(51) use `M_Pl`, while the geometric control cuts are in string units |
| scalar amplitude | recompute from the scaled potential and kinetic metric | Eqs. (49)--(51); absolute amplitude is not fixed by shape-only diagnostics |

At the same non-unit `k`, the selected full-volume and author fixed-volume
coefficient maps use the same `tau`, `Kinv`, `Q`, and phase data. Because every
term in Eq. (19) has the common `V_CY^(-2)` prefactor, their potential,
gradient, and Hessian amplitudes differ by `k^(-3)` when only the volume law is
changed. Dimensionless stationary-point locations and Hessian inertia are
unchanged by that common factor, but Eq. (49) gives a scalar-amplitude ratio
`delta_H(full)/delta_H(fixed)=k^(-3/2)` at the same shape and kinetic metric.
This comparison does not remove the separate changes from the `tau` and
`Kinv` path relative to `k=1`.

## Numeric representation audit

The bounded existing-data sample
/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data/h11_015/np_0000001/cy_0000001/cyax.h5
has SHA-256
fa29501f512a1d9c00437b26d6d4f5acd42c8e9b638edb5f6c48be0d35d9fc9e.
Read-only HDF5 inspection reports float64 for
cytools/geometric/CY_volume, cytools/geometric/divisor_volumes,
cytools/geometric/Kinv, and cytools/potential/L; the stored Q dataset
in this artifact is also float64, although charges are semantically
integer. The Julia reader assigns the geometric fields to Float64 at
src/read.jl:32-59, reads potential L/Kinv through Float64 bindings at
src/read.jl:240-258, and returns Float64 L/K at
src/read.jl:385-424; the corresponding typed package structs are
Float64 at src/structs.jl:23-39. The downstream structured evaluator and
its workspaces are explicitly Float64 at src/generate.jl:100-142 and
src/generate.jl:245-273.
The inflation persistence writer separately stores BigFloat flow values as
decimal strings with precision-bit sidecars at
scripts/build_orientifold_vacua_inflation.jl:258-287; that path does not
introduce a new Float64 narrowing.

The certificate still evaluates the homogeneous law in BigFloat. The only
conversion to the existing evaluator is now an explicit audited boundary in
scripts/inflation_scale_continuation.jl:218-334 and
scripts/inflation_scale_continuation.jl:762-816. It records source type and
bits, target Float64 and 53 bits, per-field round-trip error bounds, the
declared reference/metric/relative tolerances, and field-by-field comparison.
It fails closed as numerical_failure/scale_status=unsupported when the
conversion is unsafe. For the inspected source representation, Float64
(53 bits) is the actual stored precision; no generalization of the
downstream evaluator was required. The boundary does not claim that
arbitrary-precision source values are lossless: it permits conversion only
when the measured bound is within the recorded tolerance.

## Physical-domain and claim boundary

An output may receive `scale_status=physical` only after the same-`k`
certificate records and passes the `physical_scaling_gate`. Its required
items are:

- finite positive divisor and Calabi--Yau volumes;
- Kähler-cone interior membership, positive effective curve/divisor volumes,
  and a numerical margin;
- symmetric positive-definite `K` and `Kinv`, reciprocal residual, and the
  declared SPD tolerance policy;
- fixed basis and charge orientation, including a verified `Q/L` reference at
  `k=1`;
- the exact phase convention, units, normalization, source identity,
  configuration identity, precision metadata, and the audited conversion to
  the evaluator's numeric type.

The independent `physical_control_gate` records, but does not silently fold
into the scaling gate:

- potent-ray evidence and instanton control;
- perturbative-control evidence;
- moduli control or stabilization; and
- visible-sector/QCD applicability.

The control gate may be `not_established` for a diagnostic calculation. It
must be `passed` before any physical viability, production qualification, or
validated-candidate statement. `moduli_status=not_established` remains the
default absent separate stabilization evidence.

Terminal domain results must distinguish `passed`, `domain_failure`,
`missing_evidence`, `numerical_failure`, and `out_of_model`. Missing data is
never a pass and never a fallback to `homotopy_only`.

The records must also retain independent `fixed_point_status`,
`trajectory_status`, and `coverage_status`. A saddle is not a catastrophe; a
Hessian sign change is not stabilized physical continuation; a screen hit is
not a refined candidate; a refined stationary point is not a trajectory; and
zero found rows is not a population exclusion.

## Historical diagnostic evidence

The earlier bounded pilot and its existing output rows remain mathematical
diagnostics. Its historical logarithmic path is explicitly
`L[2,:] -> k L[2,:]` with `K` and other inputs held fixed; this is not a
divisor-volume continuation. Any earlier screen/crossing rows therefore stay
`scale_status=homotopy_only` and cannot support a physical candidate, rate, or
null claim. The repaired persistence driver
`scripts/build_orientifold_vacua_inflation.jl` currently writes this status.
The fails-closed certificate boundary and the independent gate contract are
implemented in `scripts/inflation_scale_continuation.jl` and covered by narrow
synthetic checks; this does not relabel historical rows or authorize a
production run.

## Release decision and stop state

The owner-selected homogeneous normalization resolves the prior convention
choice. It does not authorize a physical-scale production calculation. The
current release claim is:

> The project has a primary-source-backed owner decision for the homogeneous
> volume law, separate fail-closed scaling and control gates, and a documented
> distinction between fixed-volume comparison and historical homotopy
> diagnostics. No physical inflation candidate, trajectory, stabilized moduli
> point, population rate, or physical null has been established.

Production remains blocked pending complete target-geometry scaling
certificates, independent control evidence, the named benchmark ladder, and
bounded resource gates. The diagnostic permission for a passed scaling gate
with `physical_control_gate=not_established` does not relax that boundary.
No scale, trajectory, geometry, population, database, or production run is
part of this ledger update.

Owner decision: Viraf M. Mehta — homogeneous convention selected in the
2026-08-25 project instruction. Version impact: scientific behavior and
physical-status contract change; package-version bump is deferred to the
reviewed release boundary.
