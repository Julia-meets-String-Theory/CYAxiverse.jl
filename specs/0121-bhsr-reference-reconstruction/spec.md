---
id: CYAX-0121
title: Governed BHSR reference reconstruction and migration boundary
status: reviewed_pending_owner_approval
work_class: S2
canonical_issue: https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/121
---

# Governed BHSR reference reconstruction and migration boundary

## 1. Purpose

Define the governing scientific contract for CYAxiverse black-hole-superradiance (BHSR) reference reconstruction and post-processing.

This specification codifies the owner-approved CYAX-0121 scientific rebind. It does not introduce a new scientific choice.

The immediate programme is to build and validate an interim, provenance-explicit BHSR oracle layer that:

1. reconstructs the governing historical/reference scalar-superradiance calculations under explicit model identities;
2. preserves the source-defined multi-axion probability composition and self-interaction-aware decision structure;
3. handles incomplete, provisional, censored, or route-ambiguous inputs fail-closed;
4. provides a separately named modern interim likelihood route where historical source uncertainty information is incomplete;
5. exposes a clean future migration/interface boundary to the independently governed Hoof et al. Bayesian backend work in Issue #192.

It does not authorize production deployment of Hoof code, current-2026 BHSR constraint claims, population studies, route promotion, package release, or Issue #121 closure.

## 2. Authority and scientific provenance

The owner-approved scientific direction is the durable decision RESOLVE_BHSR_AMBIGUITIES_AND_TARGET_HOOF_BAYESIAN_PRODUCTION.

The owner-approved S2 contract decision is APPROVE_SCIENTIFIC_CONTRACT_FOR_GOVERNING_S2_SPEC.

This specification must remain faithful to those decisions.

### 2.1 Governing/reference sources

- Stott & Marsh, arXiv:1805.02016v2 — scalar Kerr/Regge numerical method and Appendix-B likelihood construction.
- Stott, arXiv:2009.07206v1 — analytic-rate black-hole-spin bounds and observational ensemble.
- Mehta et al., arXiv:2103.06812v2 / JCAP 2021(07)033 — CY axion application, self-interaction-aware constraints, and multi-axion probability composition.

Hoof et al., arXiv:2406.10337 is the intended future production-statistics architecture, but its code and production inference are governed independently by Issue #192.

Historical code or grids, if recovered, are regression evidence only. Byte equality to historical floating-point arrays is not the scientific acceptance criterion.

## 3. Model identities

Every BHSR artifact must carry an explicit model identity. The following identities remain distinct.

### 3.1 REFERENCE_2021_ANALYTIC

This is the governing source-equivalent/reference analytic superradiance route.

Requirements:

- use the source analytic scalar-rate conventions;
- include the source-governed efficient mode envelope with l=m=1,...,5;
- preserve explicit N=n_r+l+1 notation;
- include the |211> benchmark;
- permit an initial bounded n_r=0,...,4 exploration per l, but do not promote that bounded exploration into permanent production semantics;
- retain the full union of efficient spin intervals;
- collapse to a single Regge onset only when the complete union is exactly one interval extending to a*=1.

Published Figure-3 digitisation is non-gating validation only.

### 3.2 CF_2018_VALIDATION

This is an independent numerical validation route, not a silent replacement for the reference analytic route.

Requirements:

- adaptive continued-fraction order refinement/doubling;
- default relative convergence target 1e-8;
- deterministic hard maximum order 16384;
- independent arithmetic-precision, angular, and root-residual checks;
- failure to converge by the cap fails closed.

Disagreement between a converged CF route and a low-alpha analytic approximation is diagnostic evidence, not automatically a failure of the CF route.

Any future promotion of a CF route to production authority requires a separately governed scientific decision.

### 3.3 REFERENCE_2018_APPENDIX_B_FORMULA

This identity reproduces the historical Appendix-B formula route where source inputs are sufficiently defined.

It preserves:

- the source Appendix-B equations labelled (B1)-(B4), corresponding to the formula slots called Eqs. 95-98 in the owner-approved r6 contract; these are two numbering conventions, and Eqs. 95-98 are not the printed Appendix-B labels in arXiv:1805.02016v2;
- (B1) / r6 Eq. 95: P_ex = 1 - P_allowed;
- (B2) / r6 Eq. 96: product over black-hole data points;
- (B3)-(B4) / r6 Eqs. 97-98: the projected-error formulas;
- zero BH mass-spin covariance as in the source approximation;
- the source effective one-dimensional projected-error calculation with the standard error function;
- for a contour y=f(x), Sigma_y^2 = sigma_y^2 + f'(xbar)^2 sigma_x^2;
- for an inverse contour x=g(y), Sigma_x^2 = sigma_x^2 + g'(ybar)^2 sigma_y^2; when g(y) is multivalued, select the inverse branch nearest xbar to evaluate the derivative, then evaluate the error function between g1 and g2;
- the source branch selection and cusp rules described in Appendix-B prose;
- source cusp approximation.

A generic two-dimensional Mahalanobis, contour-integral, or other two-dimensional surrogate must not replace the source one-dimensional projected-error and error-function calculation under this identity.

It must not claim an exactly source-equivalent 24-BH probability where the historical source does not provide a complete unique Gaussian-sigma convention.

### 3.4 CYAX_BHSR_LIKELIHOOD_V1

This is a separately named interim modern likelihood identity.

Requirements:

- handle asymmetric and censored measurements explicitly;
- never invent symmetric Gaussian sigmas for one-sided bounds;
- propagate censored measurements as probability bounds/intervals or an equivalent censor-aware likelihood;
- classify excluded at threshold 0.9545 only when the lower bound on exclusion probability exceeds the threshold;
- classify allowed only when the upper bound does not exceed the threshold;
- otherwise report unresolved;
- never extrapolate a published stellar contour outside its source/domain support.

This identity is not the Hoof Bayesian production backend.

### 3.5 REFERENCE_2021_BOSENOVA

This is the bounded formula-level self-interaction route.

It preserves the source equations and omissions, including:

- N_max source scaling;
- N_Bose source scaling;
- f_pert = sqrt(m^2/abs(lambda_iiii));
- the Eq. 26 criterion from Stott, arXiv:2009.07206v1, Section II.2: Gamma_SR * tau_BH * (N_Bose/N_max) > ln(N_Bose);
- per-mode source analytic Gamma_SR rates and the source timescale when deriving each self-interaction-modified Regge boundary;
- the reference Salpeter default tau_BH = tau_Sal approximately 4.5e7 yr where the source reference path uses it; any target-specific alternative timescale must be source-bound in the method manifest and carry a distinct target identity;
- equality at the criterion is the transition boundary and must be tested on both just-satisfied and just-unsatisfied sides, without tolerance-dependent reclassification;
- no ad hoc multiplicative suppression factor;
- source use of abs(lambda_iiii) in the action-magnitude criterion while retaining interaction sign as provenance;
- omission of off-diagonal flavor-changing processes and cubic interactions under the reference identity.

This route is an interim/reference model and must not be relabelled as the final production self-interaction model.

### 3.6 HOOF_2024_BAYESIAN_BHSR

This is the intended future production-statistics identity.

Under CYAX-0121, implementation may define only the migration/interface contract:

- route/provenance interfaces;
- data adapters;
- posterior/likelihood handoff expectations;
- nuisance-parameter hooks;
- compatibility requirements for future hierarchical population modelling.

Issue #192 is independently governed and must define its own physics and inference branch. That branch may consume the common provenance-bound spectrum interface, but it must not inherit the CYAX-0121 scientific BHSR calculation. CYAX-0121 does not authorize Hoof code vendoring/import as production runtime, backend deployment, production inference, or current-constraint publication.

## 4. Physical-spectrum input authority

### 4.1 Required provenance

CYAxiverse physical-spectrum inputs must retain:

- exact schema identity or an explicitly reviewed compatible successor;
- completed terminal status;
- mass units;
- mode indices;
- configuration digest;
- precision identity;
- provisional status;
- content-bound route/source provenance.

For ordinary current physical-spectrum-v3 outputs, source revision is authoritative only when supplied by exact content-bound provenance or verified by recomputing the persisted configuration digest against the candidate producer revision/configuration.

### 4.2 CYAX-0173 route-authority boundary

The bounded N=5 investigation supports the diagnosis that Float64 canonical-Hessian assembly can lose disputed low-mode information under extreme conditioning. That diagnosis does not promote a universal spectrum route.

Therefore:

- no Float64/high-precision route may be silently unified or promoted;
- disputed inputs require exact content-bound route provenance;
- route-specific outputs remain distinct unless a later owner-approved scientific decision changes authority;
- absent or ambiguous route provenance fails closed as INDETERMINATE_SPECTRUM_AUTHORITY / AUTHORITY_UNRESOLVED.

## 5. Source probability composition

A source-equivalent geometry decision must preserve the source probability tree:

P_allowed(A | {d_i}) = product_i P_allowed(A | d_i)

P_allowed(M | {d_i}) = product_a P_allowed(A_a | {d_i})

P_ex(M) = 1 - P_allowed(M)

A binary any-mode OR is forbidden.

Only after P_ex is computed may a reference classification be derived using the source threshold P_ex > 0.9545, unless exact source validation establishes a different version-bound value; such a difference requires rebind rather than silent mutation.

## 6. Complete-model and indeterminate semantics

An authoritative geometry probability requires an authoritative source-defined P_allowed for every source-relevant axion, except where the frozen source itself defines an outside-support neutral contribution.

A source-relevant axion must not be omitted or assigned an invented neutral probability.

If any required axion is missing required self-interaction data, provisional, spectrum-authority unresolved, failed/unavailable, or otherwise missing authoritative source-defined probability, then authoritative geometry probability is unavailable and the geometry decision is INDETERMINATE_INCOMPLETE_MODEL.

Partial or evaluable-subset products may be retained only as explicitly non-authoritative diagnostics that list included and excluded mode identities.

## 7. Result and denominator contract

Per-mode output must retain at least:

- source constraint/model identity;
- mode index;
- mass;
- self-interaction inputs when required;
- per-black-hole likelihood components;
- per-axion allowed/exclusion probability;
- spectrum route and precision/provenance;
- provisional flag;
- decision status.

Per-geometry output must retain at least:

- geometry identity;
- source/model identity;
- selection route;
- attempted, completed, evaluable, provisional, and indeterminate counts;
- authoritative geometry probability when available;
- exclusion probability when available;
- threshold identity;
- geometry decision;
- spectrum-authority status.

No aggregate fraction may be reported without its exact numerator, denominator, and selection route.

## 8. Population boundary — Issue #115

Issue #115 remains the authority gate for authoritative large-h11 ensemble/population claims.

CYAX-0121 may implement and validate source fixtures, deterministic reference kernels, bounded explicit-denominator adapter tests, and bounded non-population smoke tests.

It may not make new CYAxiverse population fractions, prevalence claims, or authoritative large-h11 ensemble summaries until the applicable #115 contract is separately satisfied.

## 9. Implementation scope

The implementation successor derived from this specification remains a post-processing/reconstruction layer.

Expected create-new implementation paths are:

- scripts/bhsr_regge_reconstruction.jl;
- scripts/bhsr_likelihood_reconstruction.jl;
- focused BHSR tests;
- validation/cyax_0121_bhsr_reconstruction/**.

An exact current test/runtests.jl may be modified only to include focused tests.

Forbidden unless separately re-governed:

- src/** production mutation;
- scripts/batch_physical_spectrum.jl mutation;
- package/dependency/workflow/version changes;
- production database replacement;
- population-scale execution;
- silent #173 route promotion;
- Hoof production deployment;
- current-2026 constraints claims;
- merge or Issue closure authority.

## 10. Numerical and scientific acceptance

Before comparison or tuning, freeze:

- arithmetic precision;
- CF order/convergence rule;
- root tolerance;
- contour grid/interpolation resolution;
- likelihood quadrature/interpolation settings;
- route identities;
- censoring semantics;
- support rules;
- uncertainty treatment.

Predeclare comparison observables such as critical spin versus mass, contour turning points, selected P_ex values, published exclusion interval endpoints, and selected 2021 summary quantities.

Do not tune hidden constants to force agreement.

A route passes when it is numerically converged and agrees with source observables within a justified uncertainty budget that separates numerical convergence error, source-data uncertainty, and graphical digitisation uncertainty.

Material unexplained disagreement returns SCIENTIFIC_CONTRACT_AMBIGUITY.

## 11. Required validation stages

### S1 — Source-governed analytic envelope

Validate l=m=1,...,5, explicit overtone manifest, bounded initial n_r=0,...,4 exploration, |211> benchmark, higher-l materiality, and full-union topology.

### S2 — Adaptive CF validation

Validate adaptive order doubling, 1e-8 default relative target, hard cap 16384, independent precision/angular/root-residual checks, and source/published spot checks.

### S3A — Historical Appendix-B formula route

Validate exact historical equations/branch/cusp logic where source uncertainty inputs are defined. Do not invent missing source sigmas.

### S3B — CYAX_BHSR_LIKELIHOOD_V1

Validate asymmetric measurements, censored measurements, explicit probability bounds, excluded/allowed/unresolved threshold semantics, and support-boundary failures.

### S4 — Reference bosenova route

Validate source N_max, N_Bose, f_pert, spin-down criterion, and deliberate omission boundaries.

### S5 — Hoof migration interface

Validate route/provenance identity, adapter contracts, nuisance-parameter hooks, and future migration compatibility without deploying Hoof production code.

## 12. Verification requirements

The implementation successor must require, on one exact final candidate:

- focused BHSR tests;
- full package tests under the repository full-test mode;
- Julia quality audit;
- Python-free package import;
- repository diff checks;
- exact changed-path scope check;
- source/method manifest integrity;
- numerical convergence evidence;
- reference probability-tree tests;
- incomplete-model and censoring tests;
- route-provenance fail-closed tests;
- exact independent SPEC and SCIENTIFIC final reviews under current reviewer capability policy.

Remote CI is corroborative only and does not replace mandatory local verification.

## 13. Stop conditions

Stop before scientific implementation or before accepting a result if:

- the governing source contract is ambiguous;
- source/version/route provenance is missing or inconsistent;
- correct work requires mutation outside reviewed scope;
- a new physical/statistical model choice is required;
- Hoof production deployment would be required;
- current-2026 claims would be implied;
- the #173 authority boundary would be silently changed;
- population claims would require unsatisfied #115 authority;
- mandatory numerical convergence fails;
- required reviewer capability is unavailable.

## 14. Claim boundaries

Successful CYAX-0121 implementation may claim only what its exact route identities and validation support.

It may not by itself claim current-2026 BHSR constraints, authoritative population prevalence, universal promotion of one spectrum precision route, production Hoof deployment, package release, or Issue #121 completion.

## 15. Approval and successor rule

The frontmatter status is lifecycle metadata. `reviewed_pending_owner_approval` does not itself record or imply final owner approval. Exact normative bytes still require final owner approval before Control Desk constructs a current-schema implementation handoff bound to the approved exact spec identity and submits it for a fresh Handoff Review.

No historical READY_TO_DISPATCH record for the pre-spec r3 baseline packet remains sufficient after current-source/schema/spec-policy drift.
