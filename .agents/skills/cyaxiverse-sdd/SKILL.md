---
name: cyaxiverse-sdd
description: Draft, review, plan, task, migrate, or converge CYAxiverse specification-driven work. Use for S1-S3 feature/spec work, especially scientific contracts that need durable intent and requirement-to-evidence traceability.
---

# CYAxiverse Specification-Driven Development

1. Read `AGENTS.md`, the governing Issue, any existing feature spec, and only the domain skills/material relevant to the task. `AGENTS.md` remains the repository constitution; this skill does not override it.
2. Classify work by semantic risk: `S0` trivial (no spec required), `S1` bounded engineering, `S2` scientific/durable contract, `S3` research programme composed of independently testable S1/S2 slices. A small change is S2 when it changes scientific meaning.
3. For S1-S3 work, locate the governing `specs/<id>-<name>/spec.md`. If none exists, draft it before changing intended behaviour. An Issue is the intake/coordination object, not a substitute for the feature contract.
4. Draft requirements around observable behaviour and evidence. For S2 scientific work, separate source facts, implementation facts, empirical evidence, owner-approved conventions and inference; state what the work may and must not establish. Use EARS-style requirements only when they improve precision.
5. Keep a spec `draft` while normative intent is unresolved. For S2/S3 work, do not implement consequential behaviour until the relevant scientific choices are owner-approved. Stop rather than infer a normalization, basis, population/counting definition, acceptance criterion, physical interpretation or scientific schema.
6. After approval, derive `plan.md` and `tasks.md`. Map every normative requirement/gate to planned implementation and verification. Give each task one observable outcome, include its verification, and name escalation conditions. Do not create a GitHub Issue for every task; promote only independently durable, blocked, mergeable/reviewable or owner-decision-bearing work.
7. Implementation may change technical details without re-approval when intent is unchanged. If investigation changes intended behaviour, update and re-review the spec before continuing the affected work.
8. Partial-gate PRs are valid. State exactly which requirements/gates the PR implements, explicit non-scope, scientific/API/schema/version impact, evidence, and remaining work. Do not imply completion of a parent spec from a partial deliverable.
9. Before S2 completion, converge approved spec ↔ plan ↔ tasks ↔ implementation ↔ tests/scientific evidence ↔ PR scope. Any unmet requirement remains explicitly incomplete or becomes additional work; checked-off tasks are not proof of spec satisfaction.
10. GitHub Issues and Projects track work state and relationships. The Project is the live visual map, not a competing authority or prose ledger. Preserve historical Issues/PRs during brownfield migration rather than rewriting them as though SDD had existed earlier.

Finish with the governing spec/status, requirements/gates advanced, verification evidence, remaining work, and any owner decision required. Use the relevant domain skill as well when the work changes Julia/scientific/sampling/vacua/release behaviour.
