---
spec_id: CYAX-0191
title: Reproducible Kähler-Moduli Stabilisation from CYTools Geometry
issue: 191
class: S3
status: draft
workstream: Kähler-moduli stabilisation research arm
parent: null
depends_on:
  - "Issue #191 Phase-0 contract and recorded Gate A PASS"
created: 2026-09-26
last_reviewed: 2026-09-26
review_required: fresh independent Spec and Standards reviews, followed by owner final SDD approval
approval_ref: N/A while draft
---

# Reproducible Kähler-Moduli Stabilisation from CYTools Geometry

## Objective

Establish a production-shaped, independently testable research programme in
CYAxiverse.jl for numerical Kähler-sector stabilisation in the O3/O7 type-IIB
setting. Phase 0 is a bounded reproduction of the two-cycle scalar potential
and named critical points in AbdusSalam, Abel, Cicoli, Quevedo and Shukla,
“A Systematic Approach to Kähler Moduli Stabilisation,” JHEP 08 (2020) 047,
arXiv:2005.11329. Geometry is version-pinned, model data and conventions are
explicit, and non-perturbative inputs are externally supplied with provenance.

This is a research arm, isolated from the stable CYAxiverse.jl API. This
document is a **draft** translation of the owner-approved scientific contract
in Issue #191. It does not approve itself. Consequential scientific
implementation may start only after fresh exact-candidate Spec and Standards
reviews pass and the owner separately records final SDD approval. That approval
must identify the reviewed normative revision or content in `approval_ref`.

## Motivation

The programme makes Kähler-sector critical-point and fluctuation calculations
reproducible from explicit geometry, a declared truncated EFT, conventions,
and numerical-search evidence. The source reproductions are conditional
results for that EFT. A production-shaped boundary from the first Phase-0
slice keeps geometry import, model physics, differentiation, search, and
provenance independently reviewable as later work proceeds.

## Current baseline

### Source facts

- Issue #191 contains the complete 29-section Phase-0 scientific contract,
  including equations, source locators, benchmark data, acceptance rules, and
  explicit claim exclusions. Its published JHEP version-of-record PDF is the
  canonical locator for equations and tables.
- Issue #191 records Gate A as owner-confirmed after inspection, hashing, and
  GitHub rendering of the exact raw issue Markdown. This document carries that
  recorded entry state; it does not repeat or recertify the rendering check.
- Issue #191's production-planning comment is implementation guidance. It
  requires production-shaped architecture and does not amend the scientific
  contract or broaden Phase-0 physics.
- The handoff identifies the exact source snapshot, source identities, and
  separate scientific and architecture authority. The current source contract
  is Issue #191 as pinned for this preparation; material issue or source drift
  requires rebind and review.

### Owner-approved conventions

- Phase 0 uses the O3/O7 source setting with `h^{1,1}_{-}=0`, flux-fixed
  complex-structure moduli and axio-dilaton, the source heavy-field reduction,
  and the correction switches and externally supplied inputs stated below.
- The Kähler potential includes the leading BBHL `O(alpha'^3)` correction
  when enabled. The superpotential contains `W_0` and explicitly specified
  single-instanton/gaugino-condensation contributions. Loops,
  higher-derivative effects, additional instantons, and explicit uplift are
  absent unless an explicit model switch enables them.
- `W_0`, `g_s`, `K_cs`, `A_a`, `a_a`, phases, and other external inputs are
  assumptions with provenance. An integer divisor charge says only that a
  non-perturbative term is included in the assumed EFT. It proves no
  effectivity, rigidity, zero-mode condition, nonzero Pfaffian, orientifold
  compatibility, tadpole cancellation, or global consistency.
- The adopted quadratic non-perturbative interference phase is
  `a_i rho_i - a_j rho_j - phi_i + phi_j`. The conflicting printed master
  formula is a review-diagnosed internal source inconsistency, not an
  author-confirmed erratum.
- The B2 table anomaly is retained as source evidence. Its cause is unresolved;
  parameters and normalization must not be adjusted to force the table's
  potential value to agree with the equation-defined value.
- Generic divisor charges are a CYAxiverse extension. They are not represented
  as a literal claim of the 2020 source and require independent validation.
- CYTools is a geometry importer and fixture-generation boundary. It is
  optional and absent from potential, AD, Hessian, root-finding, and optimizer
  hot paths.

### Existing implementation and empirical evidence

This SDD preparation makes no claim about the state or performance of a
scientific implementation. No Gate B, Gate C, Gate D, or Gate E execution or
benchmark result is produced by this document set. Gate A is the source
contract's recorded entry evidence only.

### Inference and unresolved questions

No new physical convention is inferred here. If an implementation cannot
resolve a normalization, basis, phase, heavy-field prescription, domain,
benchmark acceptance, physical interpretation, or scientific schema from the
owner-approved Issue #191 contract, work stops for owner direction. A technical
choice may be made without renewed scientific approval only when it preserves
that contract.

## Scope

- Translate the complete Issue #191 scientific contract into testable
  requirements and Phase-0 acceptance gates.
- Plan a canonical, versioned Julia geometry representation; a 2020 potential
  model separated from numerical methods; replaceable AD and solver backends;
  generic-precision and sparse-friendly computation; and structured
  provenance/replay identity.
- Plan independent local validation (CYAX-0191 Gate B), the three named source
  reproductions (Gate C), geometry-boundary validation (Gate D), and result
  classification (Gate E).
- Keep geometry, EFT assumptions, source observations, equation-derived
  quantities, numerical evidence, and physical-control assessments distinct.

## Non-scope

Phase 0 does not require or claim:

- a globally consistent compactification, certified string vacuum, stable
  omitted complex-structure or axio-dilaton sector, realized instanton/gauge
  sector, negligible omitted corrections, or proof of non-perturbative
  stability;
- automatic rigid-divisor or zero-mode analysis, Pfaffians, orientifold
  construction, tadpole cancellation, quantized flux construction, derivation
  of `W_0`, full gauge-sector construction, heavy-sector mass analysis,
  exhaustive vacuum enumeration, proof of toric-cone completeness, or complete
  loop/higher-derivative corrections;
- reimplementation of CYTools, runtime CYTools coupling to the numerical hot
  path, or stable public-API promotion;
- importing WSI, GV, or other additional correction physics into Phase 0
  without an approved extension, or enabling optional explicit uplift without
  its declared model switch;
- treating the planned, non-public KahlerJAX package/API as an available
  dependency;
- direct axion-pipeline integration or follow-on programme validation.

The 2026 JAX framework is methodological context; its additional corrections
are not imported automatically. The higher-dimensional caveat in Hebecker,
Schachner and Sifo, arXiv:2609.16117, remains an unresolved caveat, not a
replacement potential or settled disproof of LVS. Follow-on work receives no
automatic inheritance of Phase-0 validation.

## Scientific claim boundary

### This specification may establish

If all applicable Phase-0 gates pass, the work may establish that a critical
point of a specifically declared retained Kähler-sector potential was found,
reproduced against the named source data or equation-defined model, and
classified with the reported retained-field numerical and geometric evidence.
These are conditional results for the specified truncated EFT.

### This specification must not establish

It must not label a Phase-0 point a certified string vacuum or infer global
compactification consistency, admissible instanton realization, omitted-sector
stability, omitted-correction control, population completeness, physical
probability from optimizer hit rates, or any claim outside the retained model
and evidence. Unless separately demonstrated, both `heavy-sector stability`
and `global compactification consistency` remain `NOT_ASSESSED`.

### Epistemic categories

Reports and manifests distinguish source facts; implementation facts;
empirical evidence; owner-approved conventions; and inference. Rounded source
coordinates are observations, not exact internally redundant data. Review
refinements and equation-derived values are labeled as such and carry their
own replay provenance.

### Later axion-pipeline handoff

A later handoff retains `t_star`, `tau_star`, `rho_star`, and `Vcal_star`,
together with EFT parameters, normalizations, retained/frozen fields, and
reduction assumptions. An axion-only reduction distinguishes derivatives at
fixed saxions from derivatives of an EFT in which responsive saxions were
actually integrated out.

## Fixed conventions and invariants

### Geometry and coordinates

For triple intersections `kappa_ijk` and two-cycle coordinates `t^i`,

```math
Vcal(t) = (1/6) kappa_ijk t^i t^j t^k,
tau_i(t) = d Vcal / d t^i = (1/2) kappa_ijk t^j t^k,
kappa_ij(t) = kappa_ijk t^k.
```

These identities are checked against geometry and basis conventions. There is
no generic `t^i > 0` condition. The imported geometric inequalities define the
allowed region.

The source complex coordinate is `T_i^paper = rho_i - i tau_i` and

```math
W = W_0 + sum_i A_i exp(-i a_i T_i^paper),
W_0 = |W_0| exp(i theta_0),   A_i = |A_i| exp(i phi_i).
```

Julia may use `T_i^Julia = tau_i + i rho_i = i T_i^paper`, giving
`W = W_0 + sum_i A_i exp(-a_i T_i^Julia)`. The model identity records the
coordinate convention; axion periodicity/condensate branch; phases; Einstein
or string frame; length and `alpha'` conventions; four-dimensional Planck
normalization; treatment of `K_cs`; and active/frozen fields.

### BBHL and heavy-field reduction

```math
xi = -zeta(3) chi(X) / (2(2 pi)^3),
xihat = xi g_s^(-3/2),
Y = Vcal + xihat/2,
K = K_cs - ln(2s) - 2 ln(Y),  s = g_s^(-1).
```

`xi`, `xihat`, `xihat/2`, and `Y` remain distinct values. `K_cs` remains in
absolute potential values and physical mass scales.

The prescribed flux assumptions are `D_alpha W_0 = D_S W_0 = 0`. With
`K_full` partitioned into heavy `S` and retained `T` blocks, the retained block
of the inverse full metric is

```math
(K_full^(-1))^(T Tbar) =
  (K_(T Tbar) - K_(T Sbar) K_(S Sbar)^(-1) K_(S Tbar))^(-1).
```

It is generally not `(K_(T Tbar))^(-1)`. The selected source potential omits
explicit heavy-sector F-term contributions under this reduction, retains the
full inverse-metric effects encoded by the Schur complement, and is evaluated
without another expansion in `xihat/Vcal`. This prescription is not a claim of
a consistently integrated-out complete EFT.

For the one-modulus discriminator, `x = xihat/Vcal` and

```math
C_full = 3x(1+7x+x^2)/((1-x)(2+x)^2),
C_frozen = 3x/(4-x),
C_full - C_frozen = 81x^2/((4-x)(1-x)(2+x)^2).
```

This is a derived regression fixture, not a source-table result; its derivation
and normalization stay attached to its validation artifact.

### Potential and non-perturbative phases

For the source basis-divisor model,
`V = V_alpha'^3 + V_np1 + V_np2 + V_optional_uplift`. Each term is independently
testable. The BBHL correction switch must apply consistently to the active
Kähler potential, metric/inverse metric, and potential while stored geometric
metadata remains unchanged. Optional explicit uplift is a potential
contribution, off by default and enabled only by its explicit model switch.

The published master formula's printed quadratic phase is equivalent to
`a_i rho_i - a_j rho_j - phi_i + phi_i`, while Appendix A.2 Eq. (A.3) and direct
complex evaluation give the adopted Phase-0 phase

```math
a_i rho_i - a_j rho_j - phi_i + phi_j,
cos(a_i rho_i - a_j rho_j - phi_i + phi_j).
```

This is a review-diagnosed internal source inconsistency, not an
author-confirmed erratum. Gate B uses unequal nonzero prefactor phases (for
example `phi_1=0.37`, `phi_2=-0.23`), nonzero amplitudes and nontrivial axions;
the closed-form term agrees with direct complex evaluation and rejects the
printed equal-index-phase alternative.

### Generic divisor charges

For CYAxiverse extension charges `D_a = Q_ai D_i`,
`tau_a = Q_ai tau_i` and `rho_a = Q_ai rho_i`. The quadratic structure
transforms consistently, including contractions `Q_ai kappa_ijk t^k Q_bj`.
This is not a literal 2020-source claim. It passes only after source-basis
reduction, an independently derived low-dimensional oracle, a nontrivial
charge matrix, and nontrivial basis covariance.

### Geometry import and domain semantics

CYTools remains responsible for computational algebraic geometry. Preferred
flow is CYTools to a versioned geometry artifact to native Julia-side data.
Direct PythonCall.jl may be used at the import/research/fixture boundary only.
No Python call occurs in potential evaluation, AD, gradient/Hessian
construction, root finding, or optimization.

Every geometry artifact records CYTools revision; polytope and triangulation
identities; divisor order and basis map; dual curve-basis map; intersection
tensor; Euler characteristic; toric Mori/Kähler-cone construction and relevant
normalization; precision or exact representation; and provenance. Toric cone
outputs are called *torically inferred geometric data* unless completeness is
established independently.

Cone membership uses imported inequalities, not componentwise positivity of
`t`. A cone inequality margin is not a physical curve volume because positive
rescaling of a defining charge changes the margin without changing the cone.
A physical curve volume identifies the curve set, integral charge
normalization, frame, and length units; otherwise the value is called a cone
margin.

### Critical points, fluctuations, and controls

A critical-point result retains `(t_star, tau_star, rho_star)`. If an axion is
eliminated analytically, record the imposed phase condition, representative,
omitted stationarity check, and fluctuation condition. Do not use a universal
`stable::Bool` or `controlled::Bool`.

Use

```math
L_kin = -(1/2) G_AB partial_mu x^A partial^mu x^B,
H_AB = nabla_A nabla_B V at the critical point,
H_AB v^B = m^2 G_AB v^B.
```

At an exact critical point the connection term in the covariant Hessian
vanishes. Raw coordinate-Hessian eigenvalues are not physical masses. Report
stationarity, coordinate Hessian when useful, retained kinetic metric, and
kinetic-normalized generalized mass eigenvalues separately.

For `W = W_0 + sum_a A_a exp(-a_a Q_ai T_i)`, each `u` in
`ker(Q_active)` gives an exact axionic shift symmetry in this ungauged
truncation. Distinguish a symmetry-protected exact zero, a numerically
unresolved near-zero mode, and a genuinely lifted light mode.

For AdS4, report `m^2 L_AdS^2 >= -9/4` as the Breitenlohner–Freedman
assessment. Passing it is not a strict local minimum or proof of
non-perturbative stability.

Every benchmark predeclares working precision, field normalizations and
characteristic scales, potential scale, scaled stationarity tolerance,
root/optimizer criteria, and spectral thresholds. One acceptable scaled
stationarity diagnostic is

```math
r_stat = max_A |s_A partial_A V / V_scale|.
```

There is no sum over `A` inside an individual component before taking the
maximum. `s_A` is a predeclared scale for field `x^A`. Do not choose `V_scale`
solely as `|V_star|` when vacuum energy is cancellation-suppressed. Freeze
scales and tolerances before benchmark execution.

A bounded numerical search records method, backend/versions, domain,
initialization rule, random seed when relevant, budget, convergence criteria,
failures, duplicate rule, and refinement. A search is not proof of complete
enumeration; optimizer hit frequency is not physical probability.

Use structured states `PASS`, `FAIL`, `NOT_ASSESSED`, and `NOT_APPLICABLE`.
Report full-parent-metric domain/finiteness/invertibility/positivity;
retained-kinetic-metric finiteness/nonsingularity/positivity; imported-domain
membership; weak coupling; declared `alpha'` and retained exponential
diagnostics; omitted instantons only where data exist; loops/higher derivatives
only where calculated or bounded; scale hierarchy with assumptions or
`NOT_ASSESSED`; heavy-sector stability normally `NOT_ASSESSED`; and global
consistency `NOT_ASSESSED`. Reproduction can pass while controls are unknown or
marginal.

### Replay and numerical architecture

The canonical geometry boundary is versioned Julia-side data containing
intersection data, Euler characteristic, basis maps, toric-domain data,
precision/exactness, and provenance. CYTools is an importer/fixture boundary,
not a numerical hot-path object. The 2020 potential is one validated model
implementation consumed by generic potential/differentiation/search
interfaces. AD and solver backends remain replaceable behind stable
interfaces. Preserve generic precision, exact/sparse structure where
practical, and type stability in hot paths.

Replay identity includes source/revision and locator; geometry, basis, model,
and convention identities; units and schema; selection route and counting
unit for each reported benchmark/search result; code revision; relevant
software/backend versions; numeric precision; solver/search configuration,
budget, and seeds; and benchmark manifest. For Phase 0 the unit is the named
benchmark case and its refined critical point, not a population count. Future
pointwise comparison of `V`, gradient, and Hessian/mass data with KahlerJAX or
another implementation uses serialized inputs and does not couple runtime
packages.

The public Schachner implementation at commit
`83c2eb2e24fddad696424cf851fb130c67321ffa`,
`code/kahler_stabilisation.py` blob
`a9a6aa55c07d73f9e3ebdfa36da13cd6db4ac107`, is a supplementary, non-governing
overlapping-limit/analytic-gradient comparator. Where conventions match
exactly, its gradient may add a Gate B cross-check with WSI/GV corrections and
uplift disabled; it does not replace the independent derivative oracle or
import those corrections. StringForge's pinned documentation commit
`fd6fbfea906d39ed763175cb486ef7bcd35e7ac9`,
`documentation/source/packages/kahlerjax.md` blob
`de5f2764c6a6ab39404c346f73e71e437a9af1fa`, is only a planned, non-public
future architecture comparator with an unstable API. No unreleased package or
assumed API is a dependency. The pinned historical C++ `kklt.cpp` example,
blob `339452be1ed845d5d72a3fdfde0ae461780374bf`, is secondary corroboration;
its hard-coded objectives do not make command-line `theta` a general parameter
interface.

Other reference roles remain distinct: Becker, Becker, Haack and Louis, JHEP
06 (2002) 060, is the BBHL-correction provenance; Kachru, Kallosh, Linde and
Trivedi, Phys. Rev. D 68 (2003) 046005, is KKLT background; Balasubramanian,
Berglund, Conlon and Quevedo, JHEP 03 (2005) 007, is LVS background;
Breitenlohner and Freedman, Annals Phys. 144 (1982) 249–281, is the AdS
criterion; and version-pinned CYTools identifies geometry/API provenance.
AbdusSalam, Hughes, Quevedo and Schachner, “Coexisting Flux String Vacua from
Numerical Kähler Moduli Stabilisation,” JHEP 01 (2026) 056,
arXiv:2507.00615, is numerical-method context only; its additional correction
assumptions are not imported automatically. These roles do not replace the
2020 source as the Phase-0 model authority.

### Future research, after Phase 0

Only after Phase 0 closes may later work consider general `K,W` differentiable
potentials; continuation in `W_0`, `g_s`, or other parameters; multiple
critical-point searches; omitted-correction robustness; semi-automatic
non-perturbative divisor selection; large CYTools/CYAxiverse scans;
heavy-sector or global-consistency integration; an axion-pipeline handoff; or
topology/stabilisation/axion correlations. None inherits Phase-0 validation
automatically.

## Requirements

The IDs below define prospective implementation/evidence requirements. They
do not claim that these IDs governed earlier work.

### R-001 — Conditional scientific scope and claim boundary

Implement only the specified O3/O7 retained Kähler-sector EFT and report
conditional Phase-0 results. Preserve all explicit assumptions and exclusions
in “Scientific claim boundary.” Never infer global consistency, omitted-sector
stability, realization of assumed non-perturbative terms, negligible omitted
corrections, exhaustive search, or a certified string vacuum.

### R-002 — Geometry and convention identity

Validate the volume, divisor-volume and intersection identities against the
imported basis and inequalities. Record the paper/Julia coordinate map and all
model-identity fields listed above. Do not impose generic componentwise
positivity on two-cycle coordinates.

### R-003 — BBHL values and switches

Compute and retain `xi`, `xihat`, `xihat/2`, and `Y` as distinct quantities.
`K_cs` remains in absolute potential and physical-mass scales. A correction
switch changes active model evaluation consistently without redefining or
discarding geometry-derived metadata. The B1 fixture uses
`bbhl_correction_enabled=false` in its Kähler potential, metric/inverse metric,
and potential, while retaining computed `xi_geom` and `xihat_geom` metadata.

### R-004 — Source potential and heavy-field prescription

Represent the source potential as independently testable `alpha'^3`, linear
non-perturbative, quadratic non-perturbative, and optional-uplift terms, with
uplift disabled by default. Under the declared flux assumptions, omit explicit
heavy-sector F-terms while retaining the full inverse-metric Schur-complement
effects. Do not substitute a frozen `T Tbar`-block inverse or an additional
`xihat/Vcal` expansion. Keep any later general `K,W` construction out of
Phase-0 acceptance.

### R-005 — Governing unequal-phase convention

Use `a_i rho_i - a_j rho_j - phi_i + phi_j` in the quadratic cross term.
Preserve the master-formula discrepancy as review-diagnosed and not
author-confirmed. Pass the unequal-phase/direct-complex discriminator in Gate
B; no test may silently normalize away the unequal prefactor phase.

### R-006 — Distinct metric-domain checks

Check the full parent Hermitian metric used by the source reduction for domain
validity, finiteness, invertibility, and positive definiteness. Separately
check the retained real kinetic metric for finiteness, nonsingularity, and
positive definiteness. A positive retained block does not pass the parent check
or establish heavy-sector stability.

### R-007 — CYAxiverse generic-charge extension

Implement `D_a=Q_ai D_i`, `tau_a=Q_ai tau_i`, and `rho_a=Q_ai rho_i` with
consistent quadratic contractions. Treat this as a CYAxiverse extension. Gate
B evidence includes basis-divisor reduction, an independent low-dimensional
oracle, a nontrivial `Q`, and nontrivial basis covariance.

### R-008 — Geometry import boundary

Import CYTools output into canonical versioned Julia-side geometry data with
the required identity, basis, intersection, domain, precision, and provenance
fields. Keep CYTools/Python out of all numerical hot paths. Unless separately
proven, label cone data torically inferred. Preserve the distinction between
cone margin and unit-normalized physical curve volume.

### R-009 — Production-shaped model and numerical interfaces

Keep the 2020 model independent of numerical search, use replaceable AD and
solver backends, support scientific precision beyond `Float64` when permitted,
preserve sparse/exact structure where practical, and maintain type-stable hot
paths. Define interfaces without freezing one backend into the scientific
model contract or committing to stable public-API promotion.

### R-010 — Replay identity and manifests

Every geometry artifact and benchmark result has sufficient provenance for
replay: source/revision/locator; geometry and witness identity; basis, model,
and conventions; units; schema; code revision; software/backend versions;
precision; solver/search configuration; seed and budget where applicable; and
benchmark manifest. A benchmark manifest also includes canonical artifact,
version, equation/table/row locators, labelled alternate-version
cross-references, geometry/basis, active/frozen fields, phases/representative,
EFT parameters, correction switches, charges, exact-flat-direction
expectations, source observations, equation-derived values, source/numeric
precision, tolerances, and source discrepancies.

### R-011 — Critical points and physical fluctuations

Retain `(t_star,tau_star,rho_star)` and record analytic axion elimination
conditions and omitted checks. Report the real kinetic convention, covariant
Hessian, generalized mass eigenproblem, and kinetic-normalized masses; raw
coordinate-Hessian eigenvalues are not called physical masses. Classify exact
symmetry zeros separately from unresolved near-zero and lifted modes. Report
the AdS4 BF comparison where applicable without equating it to strict
minimality or non-perturbative stability.

### R-012 — Scaling, bounded search, and structured controls

Freeze manifest scales, stationarity and optimizer tolerances, and spectral
thresholds before benchmark execution. Report the scaled stationarity
diagnostic without a component sum before its maximum. A bounded search reports
its full method/budget/failure/replay context and makes no completeness or
probability claim. Return the required structured control categories and
`PASS`/`FAIL`/`NOT_ASSESSED`/`NOT_APPLICABLE` states.

### R-013 — P0-B1 standard KKLT fixture

Use the published JHEP PDF Eqs. (4.1)–(4.2), Table 2 `M_{1,1}`, Table 3 first
row non-uplifted columns, and Appendix A.1. Use `chi=-40`, `kappa_111=1`,
`tau_1=t_1^2/2`, `Vcal=t_1^3/6`, `t_1>0`, `W_0=-1e-4`, `a_1=0.1`,
`g_s=0.1`, `K_cs=0.1`, `A_1=1`, `theta_0=pi`, `phi_1=rho_1=0`. Keep
geometry-derived `xi_geom` and `xihat_geom` metadata. Disable BBHL consistently
in the active Kähler potential, metric/inverse metric, and potential; do not
enable anti-brane uplift. Compare to source observations
`t_*=15.0724`, `Vcal_*=570.688`, and `V_*=-3.97181e-15` at source precision.

### R-014 — P0-B2 finite-`xihat` source anomaly fixture

Use published JHEP PDF Eq. (4.5), Table 4 `M_{1,2}`; `chi=-200`,
`kappa_111=5`, `tau_1=5t_1^2/2`, `Vcal=5t_1^3/6`, `g_s=0.2`, `W_0=-0.68`,
`a_1=pi/16`, `K_cs=A_1=1`, `theta_0=pi`, `phi_1=rho_1=0`, no separate
anti-brane uplift. Preserve source observations `t_table=2.43598`,
`tau_table=14.8350`, `Vcal_table=12.0459`, table `xihat` label `2.70901`, and
`V_table=0.404328e-7`. The equation-defined `xihat` is derived from its
definition; the table's labeled value numerically corresponds to `xihat/2`.
Keep review refinements labeled non-source digits, with approximately
`t_*=2.435980787125177`, `tau_*=14.83500598810749`,
`Vcal_*=12.04592985463893`, and equation-defined `V_*=4.043279072909e-6`.
Acceptance requires source-precision location reproduction, independent
equation-defined refinement/potential reproduction, and preservation of the
approximately factor-100 source-table energy anomaly. Do not tune other
parameters or normalization to erase it; its cause remains unresolved unless
independently established.

The extra digits are review calculations, not values printed by the source.
Their benchmark evidence manifest identifies the review script
(`benchmark_checks.py` or the corresponding independent check), exact script
hash, working precision, environment, and heavy-field reduction prescription.

### R-015 — P0-B3 structureless LVS E5 fixture

Use the publisher PDF Eqs. (4.13)–(4.14), Tables 6–7, row E5; label any
alternative-version numbering only as cross-reference metadata. Geometry has
`chi=-126`, `kappa_111=1`, `kappa_222=8`, `kappa_223=-5`, `kappa_233=3`,
`kappa_333=0`, and

```math
Vcal=(t_1^3+8t_2^3-15t_2^2t_3+9t_2t_3^2)/6,
tau_1=t_1^2/2,
tau_2=4t_2^2-5t_2t_3+3t_3^2/2,
tau_3=3t_2t_3-5t_2^2/2.
```

Enforce source inequalities `t_1<0`, `t_3-2t_2>0`, `t_1+t_3>0`,
`t_1+3t_2>0` and check `t_i tau_i=3 Vcal`. Use `K_cs=1`, `W_0=-1`,
`A_1=A_2=1`, `A_3=0`, `a_1=pi`, `a_2=pi/2`, `g_s=0.10`,
`Q_active=[[1,0,0],[0,1,0]]`, `theta_0=pi`, `phi_1=phi_2=0`,
`rho_1=rho_2=0`; `rho_3=0` is only a representative of the exact flat
direction from `A_3=0`. Preserve rounded source observations
`t=(-2.92776,40.3554,80.9498)`, `Vcal=154711.0`, `xihat=9.65442`,
`V=-5.94805e-18`, `tau=(4.28590,9.73469,5728.89)`. Refine the stationary point
before deriving internally related quantities.

### R-016 — CYAX-0191 Gate B independent local validation

Before Gate C, produce no-scale and limits with BBHL and non-perturbative
amplitudes disabled as appropriate; the inverse-metric discriminator;
independent low-dimensional derivative oracle under the same heavy-field
prescription; unequal-phase/direct-complex discriminator; generic-charge
oracle and identity-charge reduction; nontrivial basis covariance over
intersection tensor, divisor/curve bases, Kähler coordinates, `tau`, `Q`, and
derivatives with invariant scalar potential; and frozen numerical scaling.
The public Schachner gradient may only add a convention-matched overlapping
limit check with WSI/GV and uplift disabled.

### R-017 — CYAX-0191 Gate C named-source reproduction

Produce and independently verify B1, B2, and B3 manifests/results with all
R-010 metadata, source-precision-aware comparisons, axion representatives,
exact flat-direction expectations, and independent residuals. Preserve
source-reported and equation-derived values separately. A failed benchmark
stops the progression; do not proceed to population execution or infer a
source correction.

### R-018 — CYAX-0191 Gate D geometry-boundary validation

Record exact geometry identity, CYTools revision and cone provenance; validate
tensor index/permutation behavior, divisor/curve duality, basis imports, and
native Julia fixture comparisons. Do not label inferred toric cones complete
without independent proof.

### R-019 — CYAX-0191 Gate E result classification

For accepted named results, report scaled stationarity, retained-sector
fluctuations, exact/numerical flat-direction classification, BF assessment
where applicable, parent-metric domain, retained kinetic metric, structured
controls, and explicit global claim boundary. Never promote an unsupported
global-vacuum or heavy-sector claim.

### R-020 — Progressive gates and authority boundary

Complete and review each required gate before advancing to the next. Any
failed earlier gate stops later execution. The SDD candidate must first pass
fresh, independent, comprehensive Spec and Standards review on the same frozen
normative bytes, using the reviewer capability pins in the governing review
request. If a pinned capability is unavailable, block and report; do not
substitute. The candidate must then receive separately recorded owner final
SDD approval.
Those conditions are prerequisites to a later consequential implementation
handoff, not authorization in this draft. This document set authorizes no
scientific code, Potential2020, Gate B/C execution, benchmark, API promotion,
package merge, or Issue #191 closure.

## Acceptance gates

Issue gates are always written with their namespace `CYAX-0191 Gate A` through
`CYAX-0191 Gate E`. They are scientific/evidence gates and remain distinct from
the SDD review and owner-approval gate below.

### CYAX-0191 Gate A — mathematical contract and artifact integrity

**Recorded entry status:** PASS in Issue #191, with owner-confirmed inspection,
hashing, and successful GitHub rendering of the exact raw issue Markdown before
the issue was frozen. This draft does not recertify that event.

**Acceptance for the Phase-0 source contract:** exact source equations pinned;
all governing equations present in raw Markdown; no missing left sides, terms,
signs, labels, or operators; exact Markdown rendered and candidate hash
recorded; coordinate and phase conventions fixed; quadratic phase discrepancy
recorded and resolved as specified; heavy-field reduction fixed; full-parent
and retained-metric checks distinguished; B1 correction-off semantics
explicit; B2 anomaly disposition fixed; B3 publisher-PDF E5 mapping fixed.

**Stop:** any failure in this entry evidence stops implementation. Do not
reconstruct missing mathematics ad hoc. Any material change to the source
contract requires rebind and renewed review.

### CYAX-0191 Gate B — independent local validation

**Acceptance:** all R-016 limiting, inverse-metric, derivative, unequal-phase,
generic-charge, identity-charge, basis-covariance, and predeclared-scaling
evidence passes. Optional Schachner comparison is supplementary and cannot
replace any required oracle.

**Stop:** any discriminator or oracle failure stops before Gate C. Resolve a
technical defect within the approved contract; return any semantic ambiguity
or proposed scientific change to the owner and re-review the SDD first.

### CYAX-0191 Gate C — named source reproduction

**Acceptance:** B1, B2, and B3 named fixtures meet R-013 through R-015, including
source-precision comparisons, independent residuals, complete manifests, and
the B2 source/equation anomaly treatment.

**Stop:** stop on any failed or untraceable fixture. Do not alter global
normalization, source convention, or acceptance tolerance to match a printed
count/value. No population scan is included.

### CYAX-0191 Gate D — geometry-boundary validation

**Acceptance:** Gate D has R-018 identity/provenance, tensor, duality, basis,
and native-fixture evidence.

**Stop:** stop on a geometry identity, basis, tensor, cone, or provenance
contradiction; do not repair scientific geometry by inference.

### CYAX-0191 Gate E — scientific result classification

**Acceptance:** Gate E reports each R-019 category for every accepted named
result and retains unsupported categories as `NOT_ASSESSED`.

**Stop:** stop or narrow claims if metrics, domains, residuals, fluctuations,
or controls do not support the proposed classification. No universal stable
or controlled boolean and no unsupported global-vacuum certification.

### SDD candidate review and owner approval

This S3 specification remains draft until the exact frozen `spec.md`,
`plan.md`, and `tasks.md` candidate receives fresh, independent, comprehensive
Spec and Standards reviews with no blocking finding. Both reviews bind the
same candidate commit/tree and exact normative file identities. Any material
normative byte change requires fresh affected review; no earlier PASS silently
carries forward. After review, return exact candidate identities to the
Control Desk for reconciliation and separate owner final SDD approval. Only
that approval is recorded in `approval_ref` and permits a future
implementation-dispatch request. No reviewer or Manager can supply it.

## Verification requirements

- Review each scientific requirement against the pinned Issue #191 contract;
  separate source values, equation-derived values, review calculations, and
  inference.
- Gate B uses analytic/synthetic and independent low-dimensional evidence
  before named-source fixtures.
- Gate C includes the exact B1/B2/B3 source manifests and independent
  stationarity residuals.
- Gate D uses native geometry fixtures and exact basis/geometry provenance.
- Gate E validates interpretation from the produced retained-sector evidence;
  classification does not substitute for global compactification evidence.
- Freeze precision, scales, tolerances, search configuration, and source
  locators before numerical benchmark execution.
- Require independent Spec and Standards review for this S3 candidate and
  preserve exact byte identities. The scientific contract itself is not
  reopened by those implementation-quality reviews.

## Interfaces and compatibility

The intended architecture has a versioned Julia-side geometry boundary; a
model interface for the 2020 potential; separate differentiation, Hessian,
critical-point search, and fluctuation interfaces; replaceable AD and solver
backends; and structured result, control, and provenance objects. Names and
exact module paths remain implementation choices subject to this contract.

CYTools/Python may create or import geometry fixtures, but must remain optional
and outside numerical hot paths after geometry import. Numeric methods preserve
precision and sparse/exact geometry where practical. Future interoperability
uses serialized inputs and pointwise outputs without runtime coupling.

This draft changes no Julia API, scientific behavior, persisted schema,
supported environment, dependency, package version, or stable public-API
commitment. The eventual implementation remains in the research arm and
requires its own authorized work packet after SDD approval.

## Dependencies and blockers

- The governing intake and scientific authority is Issue #191. The
  implementation-planning comment informs production shape but is
  non-normative for scientific acceptance.
- Gate A's owner-confirmed source evidence is an entry dependency. Gate B
  precedes Gate C; Gates D and E require source and geometry identity/evidence
  as specified above.
- Any absent or ambiguous scientific convention, material source/issue drift,
  failed upstream gate, or requirement to change an unauthorized path is a
  stop condition and requires owner direction or rebind.
- External comparison sources remain supplementary as stated under
  “Replay and numerical architecture.” They do not supersede the publisher
  source, the independent Gate B oracle, or Issue #191.

## Open owner decisions

No new scientific decision is proposed by this draft. The Phase-0 values and
conventions above are fixed by Issue #191. If a future implementation exposes
an ambiguity in that contract, record the exact ambiguity and affected
requirement, stop the affected work, and request owner direction before
changing this draft or implementing a substitute.

## Completion criterion

The Phase-0 research programme is complete only when Gates B, C, D, and E each
pass with the evidence and claim limits defined here; each accepted result has
replayable source, geometry, convention, code, numeric, and search identity;
independent review accepts the SDD and subsequent scientific evidence at the
required boundaries; and the reports make no unsupported physical or global
claim. Gate A remains its separately recorded source-entry evidence. Stable
public-API promotion, Issue closure, population claims, and follow-on research
are not implied by this criterion.
