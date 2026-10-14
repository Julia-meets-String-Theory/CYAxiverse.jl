---
spec_id: CYAX-INF-01A
title: Catastrophic-inflation scientific reference readiness
issue: 201
class: S2
status: draft
workstream: Inflation
parent_programme: Catastrophic Inflation Research Programme (owner-adopted, private canonical source)
approval_ref: pending exact normative-revision review and owner approval
version_impact: no package-version adoption in this preparation
---

# CYAX-INF-01A — Scientific Reference Readiness

## Authority and separation
This is the existing CYAX-0151 S2 scientific contract for [Issue #201](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/201), bounded by the Owner's adopted M0–M6 programme and revision reconciliation. Exact private source identities, not their storage URLs, are pinned in the private Control Desk handoff. Owner approval of the programme does **not** by itself constitute approval of this exact subsequently drafted specification; retain `draft` until exact revision review and owner approval are recorded. No implementation begins from this draft.

INF-01A establishes trustworthy N=5/N=8 benchmark and reproducibility reference fixtures. INF-01B separately verifies the full covariant multifield N=8 inflationary trajectory, physical-Hubble e-fold count, exit, and quantum diffusion conditioned on absolute normalization. INF-01C separately performs bounded geometry discovery after INF-01B. M0–M6 remain programme-level milestones; no scope expansion or roadmap reopening.

## Normative scope and acceptance
- **INF-A-01 source/replay identity**: For every fixture record immutable source/commit/tree or input artifact checksum; geometry, triangulation/witness and selection route; complete potential/charge term set and truncation; units; phase vector/coordinate convention; metric/basis and P96 authority; precision/solver/tolerances; exact commands/environment/versions; hashes for fixture inputs and outputs. Source-proven and derived fields must remain distinct. Preserve the assignment-independent common axion potential; do not introduce QCD/QED divisor assignment.
- **INF-A-02 benchmark classification**: Freeze established and unresolved N=5 reduced analytic, N=5 full-potential, and N=8 physical radial/catastrophe/continuation observations separately. Reuse already reviewed #130 and #148 work where revision/conditions apply. A passing historical test or closed Issue alone is not acceptance. Include named positive and negative/control cases and non-success/failure status.
- **INF-A-03 physical scaling**: Audit/reconstruct `tau_i(k)=k*tau_i(1)`, `Vcal(k)=k^(3/2)*Vcal(1)`, `K(k)=k^(-2)*K(1)`, `Q(k)=Q(1)` and corresponding potential coefficients from physical inputs; do not treat amplitude homotopy, fixed-volume comparison, or reused logarithmic coefficients as a physical scaling calculation. Record domain and independent `physical_scaling_gate` versus `physical_control_gate` statuses.
- **INF-A-04 #119 reproducibility**: Reconcile six required scaling evidence fields (`configuration_digest`, `normalization`, `phase_convention`, `source_identity`, `spd_tolerance`, `units`), checkpoint/shard manifest recovery status and bounded BigFloat-to-Float64 boundary. Record separately missing instanton/perturbative/moduli-stabilization control; absence of the latter does not block fixed-saxion reference, but prohibits microscopic physical-viability claims. Do not declare the 18-geometry historical observation newly replayed.
- **INF-A-05 precision, truncation, phase**: Audit source-constructed high precision versus Float64 promotion; coefficient provenance and sensitivity of catastrophe/near-null data to omitted instantons and term-set choice; phase coordinates, field redefinition redundancies and microscopic phase correlations. A numerical zero eigenvalue alone cannot certify a cusp/fold. Record negative/unresolved outcomes; do not silently select a scientific convention.
- **INF-A-06 N8 metric**: Preserve owner-approved P96 Eq.96 benchmark-matrix authority with period-one `K_theta=M96/k^2` and equivalent raw-radian `G_x=M96/[k^2(2pi)^2]`; author unconverted raw-radian metric is author-reproduction-only; the literal Eq.17/23 factor-of-two discrepancy remains explicit. Do not mix conventions when comparing distances, Hessians, observables or source term sets.
- **INF-A-07 evidence reuse and bounded tests**: Build a requirements-to-evidence matrix against #119/#130/#131/#148, focusing on gaps, source drift and replay necessity. Focused tests on exact candidate plus all prescribed integration/CI/scientific gates; reusable test evidence requires identical relevant code, input, assumptions, numerical precision, environment, scope and source identity. Never represent reuse as fresh execution.
- **INF-A-08 deliverables**: Produce a frozen, versioned reference corpus (fixtures, input/output checksums, deterministic commands, source/revision/environment identities, tolerances, result classifications) and concise evidence-and-gap report. Every open item names evidence locator, scientific consequence, severity/claim boundary and a bounded resolution or an Owner decision if a new convention is required. Unavailable verification is recorded as such.
- **INF-A-09 review**: Apply existing SPEC, STANDARDS, SCIENTIFIC axes where declared in the packet with fresh independent reviewers and exact capability/rubric/candidate bindings. Scientific Reviewer evaluates evidence/claim validity independently and must not inherit Manager conclusions. No redundant reviews, no acceptance-by-LLM-agreement, and no new governance system.

## Explicit exclusions
No new inflation engine; no unbounded parameter or phase search; no new multi-geometry production run; no population inference; no observational optimization; no moduli-stabilization or orientifold investigation; no INF-01B multifield integration or stochastic/diffusion analysis; no INF-01C geometry survey; no source/convention/acceptance mutation without governed approval. This is reference readiness, not a claim of fully realized string cosmology.

## Exit and authority
A scientifically adequate INF-01A handback contains exact reproducible corpus, source-proven evidence and limitations, independently checkable regression outcomes and complete disposition of requirements. Handoff readiness, Manager dispatch, implementation, integration, PR merge, Issue closure and future scientific stage progression are distinct owner-controlled acts. Mark nothing ready on the strength of this draft alone.
