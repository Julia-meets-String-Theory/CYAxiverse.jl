---
id: CYAX-0121
title: BHSR reference reconstruction tasks
status: reviewed_pending_owner_approval
governing_spec: specs/0121-bhsr-reference-reconstruction/spec.md
---

# BHSR reference reconstruction tasks

## T001 — Freeze currentness and source manifests

Outcome: exact execution base, approved spec identity, source versions, route identities, unit conventions, convergence parameters, censoring/support semantics, and authority boundaries are frozen before scientific code mutation.

Verify: all identities reproduce; material drift returns BLOCKED_FOR_REBIND.

## T101 — Implement source-governed analytic Regge envelope

Outcome: deterministic REFERENCE_2021_ANALYTIC contour engine with l=m=1,...,5, explicit overtone metadata, |211> benchmark, and full-union topology.

Verify: source-derived benchmarks, higher-l materiality case, interval-topology tests.

## T102 — Implement adaptive CF validation route

Outcome: CF_2018_VALIDATION with adaptive order doubling, 1e-8 target, cap 16384, and independent residual/precision checks.

Verify: convergence ladder, cap failure behavior, source/published spot checks.

## T201 — Implement historical Appendix-B formula likelihood

Outcome: REFERENCE_2018_APPENDIX_B_FORMULA reproduces source Appendix-B equations labelled (B1)-(B4), corresponding to Eqs. 95-98 in the owner-approved r6 contract, and source branch/cusp logic where source uncertainty inputs are defined. The two equation numberings are identified explicitly; Eqs. 95-98 are not the printed Appendix-B labels in arXiv:1805.02016v2.

Verify: source effective one-dimensional projected-error and standard error-function fixtures; f(x) and multivalued inverse g(y) cases with the nearest-xbar derivative branch and erf evaluated between g1 and g2; no generic two-dimensional surrogate; equation-level fixtures and bounded limitation where source sigmas are not uniquely defined.

## T202 — Implement censor-aware interim likelihood

Outcome: CYAX_BHSR_LIKELIHOOD_V1 supports asymmetric/censored measurements without invented symmetric sigmas and returns excluded/allowed/unresolved consistently.

Verify: censored-boundary fixtures around threshold 0.9545 and source-support failure cases.

## T301 — Implement reference bosenova route

Outcome: REFERENCE_2021_BOSENOVA implements source N_max, N_Bose, f_pert, and the per-mode analytic-rate criterion Gamma_SR * tau_BH * (N_Bose/N_max) > ln(N_Bose), from Stott arXiv:2009.07206v1 Eq. 26, with documented omissions and provenance.

Verify: formula fixtures use the source Salpeter default tau_Sal approximately 4.5e7 yr; any target-specific alternative is source-bound and has a distinct target identity; boundaries use each mode's source analytic rate without an ad hoc multiplicative suppression factor; equality is the transition boundary with just-satisfied and just-unsatisfied tests that do not use tolerance-dependent reclassification; sign/magnitude provenance checks.

## T401 — Implement physical-spectrum post-processing adapter

Outcome: eligible physical-spectrum outputs are processed without losing mode identity, units, provenance, provisional state, self-interaction data, or #173 route identity.

Verify: exact provenance fixtures, ambiguous-route fail-closed fixture, provisional and missing-self-interaction fixtures.

## T402 — Implement complete-model probability aggregation

Outcome: authoritative geometry probability exists only when every source-relevant axion is authoritative or explicitly source-defined outside support; no any-mode OR or omission surrogate exists.

Verify: multi-axion product fixtures, indeterminate required-mode fixture, partial-diagnostic non-authority tests.

## T403 — Implement denominator/accounting contract

Outcome: aggregate outputs carry exact attempted/completed/evaluable/provisional/indeterminate counts and selection route.

Verify: failed/provisional/mixed fixtures with exact numerator/denominator assertions.

## T501 — Define Hoof migration/interface contract

Outcome: independently governed Issue #192 has its own physics and inference branch consuming the common provenance-bound spectrum interface, with explicit route/provenance, posterior/likelihood adapter, nuisance-parameter, and hierarchical-population extension interfaces. Issue #192 never inherits the CYAX-0121 scientific BHSR calculation.

Verify: common-interface provenance and migration fixtures only; #192 physics and inference remain an independent branch; no Hoof production code deployment.

## T601 — Run bounded reconstruction/smoke evidence

Outcome: source/reference fixtures and bounded eligible CYAxiverse examples exercise validated routes without population claims.

Verify: route/model identities and claim boundaries are explicit; no updated-2026 claim is emitted.

## T701 — Full repository verification

Outcome: focused tests, full package test, Julia audit, Python-free import, diff check, changed-path scope, and manifest integrity all pass on one frozen final candidate.

Verify: exact commands, exit statuses, counts, environment identity, and candidate commit/tree are retained.

## T702 — Final independent reviews

Outcome: fresh independent SPEC and SCIENTIFIC reviews bind the same frozen final implementation candidate under current capability/rubric policy.

Verify: exact artifact identities, model/reasoning, freshness/independence, verdicts, and findings are retained.

## T703 — Manager handback

Outcome: governed handback returns READY_FOR_RECONCILIATION or a reason-coded bounded blocker.

Must state: no merge, current-2026 constraint, population study, spectrum-route promotion, Hoof deployment, release, or Issue #121 closure is authorized by the handback.
