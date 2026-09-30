---
id: CYAX-0121
title: BHSR reference reconstruction implementation plan
status: draft
governing_spec: specs/0121-bhsr-reference-reconstruction/spec.md
---

# BHSR reference reconstruction implementation plan

## Objective

Implement the owner-approved CYAX-0121 BHSR scientific contract as an interim, source-faithful and provenance-explicit oracle layer, preserving the independently governed #115 population boundary, #173 spectrum-authority boundary, and #192 Hoof production boundary.

## P0 — Currentness and source freeze

Before code mutation:

1. verify current vmm, exact writable/read-only blobs, Issues #121/#115/#173/#192, repository policy, private capability policy, and approved spec identity;
2. retrieve and freeze exact governing source versions and numerical-method manifests;
3. freeze route identities, convergence settings, unit conventions, censoring/support rules, thresholds, and uncertainty treatment;
4. stop for rebind on material scientific, scope, authority, dependency, or source drift.

## P1 — Reference analytic envelope

Implement REFERENCE_2021_ANALYTIC with source analytic scalar rates, l=m=1,...,5, explicit overtone metadata, bounded initial n_r=0,...,4 exploration, the |211> benchmark, and full-union topology.

## P2 — Independent CF validation

Implement CF_2018_VALIDATION with adaptive order doubling, relative target 1e-8, cap 16384, and independent precision/angular/root-residual checks. Do not promote CF to reference production authority.

## P3 — Likelihood routes

### P3A — REFERENCE_2018_APPENDIX_B_FORMULA

Implement historical equations, branch logic, projected-error treatment, and cusp approximation only where source inputs are defined. Missing unique source sigmas remain an explicit reproduction limitation.

### P3B — CYAX_BHSR_LIKELIHOOD_V1

Implement asymmetric and censored-data handling, probability bounds/intervals, excluded/allowed/unresolved threshold semantics, and strict source-support enforcement.

## P4 — Reference bosenova route

Implement REFERENCE_2021_BOSENOVA using source N_max, N_Bose, f_pert, spin-down efficiency, and deliberate omission/provenance rules.

## P5 — Physical-spectrum adapter and authority rules

Implement a thin adapter that preserves exact mode indices, units, configuration digest, precision, provisional status, self-interaction data, and content-bound route provenance.

Keep #173-disputed outputs route-specific. Fail closed on absent/ambiguous authority. Preserve complete-model and explicit denominator semantics. Do not mutate the upstream spectrum producer.

## P6 — Hoof migration interface

Define route/provenance interfaces, posterior/likelihood data adapters, nuisance-parameter hooks, and future hierarchical-population extension points. No Hoof production code deployment occurs.

## P7 — Verification and exact-candidate review

On one frozen final candidate, run focused tests, full package tests, Julia audit, Python-free import, diff/scope checks, manifest integrity checks, numerical convergence evidence, probability-tree tests, fail-closed authority/censoring tests, and fresh independent SPEC and SCIENTIFIC reviews.

Return only READY_FOR_RECONCILIATION or a bounded reason-coded blocker.

## Scope discipline

Expected implementation scope remains a post-processing/reconstruction layer plus focused tests/evidence. Any need for src/**, upstream producer mutation, new dependency, Hoof deployment, population execution, or new scientific choices requires rebind before mutation.

## Non-goals

This plan does not produce current-2026 BHSR constraints, establish population prevalence, promote a spectrum precision route, deploy Hoof, or authorize merge/release/Issue closure.
