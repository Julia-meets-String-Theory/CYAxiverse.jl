# Subordinate instruction tasks and evidence coverage

> **Status:** Frozen detached pre-edit review candidate. This document is not yet installed in the target repository, independently approved, implementation-ready, merged, or a standalone dispatch grant. The governing task has one exact authorised S2 scope; private control-plane authority and actor configuration are separately bound.
>
> **Construction:** Scientific and engineering requirements below are verbatim public-safe extracts of the approved complete R2. The headings and ordering are reorganised into one umbrella feature and a subordinate instruction annex, without changing the scientific methods, options, thresholds, schema meanings, fixtures, required evidence or allowed file set. If any transformation introduces a conflict, the exact governing R2 controls and independent reviewers must block acceptance pending correction.

This file is a map of the fourteen instruction paths in R2 §3. It does not carry task-level scientific acceptance or independent authority.

| Number | Required mutation | Path |
|---:|---|---|
| 1 | Append | `.agents/skills/cyaxiverse-julia-quality/SKILL.md` |
| 2 | Append | `.agents/skills/cyaxiverse-sdd/SKILL.md` |
| 3 | Append | `.agents/skills/cyaxiverse-agent-orchestration/SKILL.md` |
| 4 | Create | `.agents/skills/cyaxiverse-julia-quality/references/diagnostic-test-loop.md` |
| 5 | Create | `.agents/skills/cyaxiverse-julia-quality/references/implementation-self-review.md` |
| 6 | Create | `.agents/skills/cyaxiverse-sdd/references/architecture-assessment.md` |
| 7 | Create | `.agents/skills/cyaxiverse-sdd/references/agent-writing-domain-modeling.md` |
| 8 | Create | `.agents/skills/cyaxiverse-sdd/references/retrospective.md` |
| 9 | Create | `specs/cyax-skills-2026-10-09/spec.md` |
| 10 | Create | `specs/cyax-skills-2026-10-09/plan.md` |
| 11 | Create | `specs/cyax-skills-2026-10-09/tasks.md` |
| 12 | Create | `specs/cyax-skills-2026-10-09/sources.md` |
| 13 | Create | `specs/cyax-skills-2026-10-09/cases.json` |
| 14 | Create | `specs/cyax-skills-2026-10-09/verify.py` |

## Eight methods → five bounded local procedures

| Selected upstream method(s) | Required local procedure |
|---|---|
| diagnosing-bugs; tdd | diagnostic-test-loop.md |
| code-review | implementation-self-review.md (implementer self-check, not independent review) |
| codebase-design; improve-codebase-architecture | architecture-assessment.md |
| writing-for-agents; domain-modeling | agent-writing-domain-modeling.md |
| retro | retrospective.md |

## Twelve mandatory acceptance cases

- [ ] SK-01 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-02 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-03 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-04 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-05 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-06 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-07 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-08 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-09 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-10 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-11 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.
- [ ] SK-12 — R2 §10 expected/forbidden observations and prescribed evidence kind, assessed against the exact frozen procedure bytes and candidate.

SK-01 requires real red/green Julia execution; SK-09 an observed payload manifest; SK-11 a candidate-identity negative control; other cases require source-bound reasoning observations. Do not confuse a simulated `BLOCKED` success in SK-12 with actual implementation readiness.

## Whole-candidate/receipt/negative checks

R2 §11 and §11.1–§11.2 are normative in this annex's `spec.md`: exact fourteen-path instruction projection within all 33 files, three frozen append suffixes, five exact procedure bodies, link reachability, length budgets, symlink preservation, twelve case IDs, detached receipt hash/bytes, and the required mutants `BROKEN_LINK`, `OUT_OF_SCOPE`, `PARENT_MODIFIED`, `CASE_ID_DUPLICATE`, and `EVIDENCE_CANDIDATE_MISMATCH` plus wrong-type/duplicate-key and MX checks. Fail-closed status is not an independent review verdict.
