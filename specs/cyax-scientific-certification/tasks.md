# Unified S2 tasks — complete requirement-to-file-and-evidence map

> **Status:** Frozen detached pre-edit review candidate. This document is not yet installed in the target repository, independently approved, implementation-ready, merged, or a standalone dispatch grant. The governing task has one exact authorised S2 scope; private control-plane authority and actor configuration are separately bound.
>
> **Construction:** Scientific and engineering requirements below are verbatim public-safe extracts of the approved complete R2. The headings and ordering are reorganised into one umbrella feature and a subordinate instruction annex, without changing the scientific methods, options, thresholds, schema meanings, fixtures, required evidence or allowed file set. If any transformation introduces a conflict, the exact governing R2 controls and independent reviewers must block acceptance pending correction.

This is a requirement index, **not** a state ledger, extra task, permission, authorization or evidence receipt. Preserve §3's exact 33-file allowlist. Actual satisfaction requires independent content inspection and observed tests, never completion marks in this file.

## Production file bindings (R2 §3, unchanged)

| Number | Mutation | Path | Source requirement |
|---:|---|---|---|
| 1 | Append | `.agents/skills/cyaxiverse-julia-quality/SKILL.md` | R2 §5, §10, §11 |
| 2 | Append | `.agents/skills/cyaxiverse-sdd/SKILL.md` | R2 §5, §10, §11 |
| 3 | Append | `.agents/skills/cyaxiverse-agent-orchestration/SKILL.md` | R2 §5, §10, §11 |
| 4 | Create | `.agents/skills/cyaxiverse-julia-quality/references/diagnostic-test-loop.md` | R2 §5, §10, §11 |
| 5 | Create | `.agents/skills/cyaxiverse-julia-quality/references/implementation-self-review.md` | R2 §5, §10, §11 |
| 6 | Create | `.agents/skills/cyaxiverse-sdd/references/architecture-assessment.md` | R2 §5, §10, §11 |
| 7 | Create | `.agents/skills/cyaxiverse-sdd/references/agent-writing-domain-modeling.md` | R2 §5, §10, §11 |
| 8 | Create | `.agents/skills/cyaxiverse-sdd/references/retrospective.md` | R2 §5, §10, §11 |
| 9 | Create | `specs/cyax-skills-2026-10-09/spec.md` | R2 §5, §10, §11 |
| 10 | Create | `specs/cyax-skills-2026-10-09/plan.md` | R2 §5, §10, §11 |
| 11 | Create | `specs/cyax-skills-2026-10-09/tasks.md` | R2 §5, §10, §11 |
| 12 | Create | `specs/cyax-skills-2026-10-09/sources.md` | R2 §5, §10, §11 |
| 13 | Create | `specs/cyax-skills-2026-10-09/cases.json` | R2 §5, §10, §11 |
| 14 | Create | `specs/cyax-skills-2026-10-09/verify.py` | R2 §5, §10, §11 |
| 15 | Create | `specs/cyax-scientific-certification/spec.md` | R2 §§6–9 |
| 16 | Create | `specs/cyax-scientific-certification/plan.md` | R2 §§6–9 |
| 17 | Create | `specs/cyax-scientific-certification/tasks.md` | R2 §§6–9 |
| 18 | Narrow edit | `src/CYAxiverse.jl` | R2 §§6–9 |
| 19 | Create | `src/scientific_certification.jl` | R2 §§6–9 |
| 20 | Create | `src/scientific_certification/intervals.jl` | R2 §§6–9 |
| 21 | Create | `src/scientific_certification/stationary.jl` | R2 §§6–9 |
| 22 | Create | `src/scientific_certification/spectra.jl` | R2 §§6–9 |
| 23 | Create | `src/scientific_certification/certificates.jl` | R2 §§6–9 |
| 24 | Create | `test/scientific_certification.jl` | R2 §§6–9 |
| 25 | Create | `test/fixtures/scientific_certification/public.toml` | R2 §§6–9 |
| 26 | Create | `test/fixtures/scientific_certification/oracles.jl` | R2 §§6–9 |
| 27 | Create | `test/fixtures/scientific_certification/backend_probe.jl` | R2 §§6–9 |
| 28 | Create | `test/fixtures/scientific_certification/run_fixture.jl` | R2 §§6–9 |
| 29 | Narrow edit | `test/runtests.jl` | R2 §§6–9 |
| 30 | Create | `scripts/verify_scientific_certificate.jl` | R2 §§6–9 |
| 31 | Create | `docs/src/scientific_certification.md` | R2 §§7–9, §12.4 |
| 32 | Narrow edit | `docs/make.jl` | R2 §§7–9, §12.4 |
| 33 | Narrow edit | `Project.toml` | R2 §3, §6.5, §12.2 |

## Closed scientific requirement clusters

| Cluster | Exact source and obligations | Evidence source |
|---|---|---|
| Input/model and interpretation | R2 §6.1, §7, §8.1–§8.2; exact scalar, orientation, phase, signed-amplitude and log semantics | T01–T04, T06, T09–T10, T12, T16 |
| Local root proof | R2 §6.2, §6.5–§6.6, §8.3, §8.5; unwrapped box, strict inclusion and contraction | T01–T04, T07, T09, T11–T12 |
| Shared interval inertia | R2 §6.3; LDL pivots excluding zero, metric SPD proof, classified Hessian evidence | T03–T05, T07–T08, T17–T18 |
| Generalized spectral coverage | R2 §6.4, §6.6, §7, §8.3/§8.5; diagonal and inertia-count routes, exact endpoint witnesses, required relative widths, ordered modes and clusters | T01–T06, T14–T18 |
| Arithmetic and backend isolation | R2 §6.1, §6.5, §6.6, §12.2; explicitly rounded primitives, ambient-state preservation, version source guard | T06, T10–T11, T13, T18, MX-05–MX-06 |
| Closed scientific schemas | R2 §8.1–§8.6; exact quoted rationals, strict keys, resource bounds, deterministic atomic writes, complete proof replay | T11, T13, T14, T15, MX-03–MX-05 |
| Independent numerical oracles and held-out | R2 §9, §9.1–§9.4; named benchmark, independent exact algebra, T15 private prior commitment and post-freeze evaluation | T05, T06, T14–T16 |
| Verification and lawful integration | R2 §12.2–§12.6 and §13; full local/CI, frozen candidate, independent reviews, separate acceptance/integration statuses | T01–T18, SK-01–SK-12, MX-01–MX-06 |

## Mandatory scientific test families (all 18; verbatim criteria in umbrella spec §9)

- [ ] T01 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T02 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T03 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T04 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T05 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T06 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T07 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T08 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T09 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T10 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T11 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T12 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T13 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T14 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T15 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T16 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T17 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.
- [ ] T18 — perform every original R2 §9 subcase and record independent oracle/proof/negative disposition against the exact candidate and required precision profile.

T06 has distinct `T06_hierarchy` and `T06_fractional_log` subcases; T16 has distinct `T16_phase_unit`, `T16_common_shift`, `T16_signed_charge` subcases. The required exact producer/replay and relative-bound obligations attach to all designated positives. T15 input and oracle must not be placed in public source.

## Cross-component controls (all six; exact criteria in umbrella spec §12.6)

- [ ] MX-01 — execute/assess the complete R2 §12.6 obligation with candidate-specific detached evidence.
- [ ] MX-02 — execute/assess the complete R2 §12.6 obligation with candidate-specific detached evidence.
- [ ] MX-03 — execute/assess the complete R2 §12.6 obligation with candidate-specific detached evidence.
- [ ] MX-04 — execute/assess the complete R2 §12.6 obligation with candidate-specific detached evidence.
- [ ] MX-05 — execute/assess the complete R2 §12.6 obligation with candidate-specific detached evidence.
- [ ] MX-06 — execute/assess the complete R2 §12.6 obligation with candidate-specific detached evidence.

## Evidence and independent review

- Match the same 33 paths, exact existing-file mutation classes, and no additional production path; exercise checker mutants in disposable checkouts only (R2 §§3, 11).
- Bind project head, test environment, exact backend source, current source, full-suite retention, audit, docs, existing full CI and clean-package import (R2 §§12.2–12.5).
- Preserve one coherent root/Hessian/spectrum witness; no proof assembled from mismatched attempts or revisions (R2 §§6.6, 8.5).
- Complete eligible pre-edit instruction reviews, later frozen-candidate SPEC/STANDARDS/SCIENTIFIC, Handoff Review where required, and independent T15 assessment without inheriting old verdicts (R2 §§2.2, 9.4, 13).
- Separate `IMPLEMENTATION_COMPLETE`, `ACCEPTED_FOR_INTEGRATION`, and `INTEGRATED` criteria as in R2 §13. No task file entry declares any of these outcomes.
