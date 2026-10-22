---
spec_id: CYAX-0209
title: Cross-provider agent coordination and verification
issue: 209
class: S3
status: draft
workstream: Infrastructure
parent: null
depends_on: []
created: 2026-10-22
last_reviewed: 2026-10-22
review_required: "SPEC and STANDARDS; applicable architecture/handoff review; SCIENTIFIC where scientific claims require it"
approval_ref: N/A
---

# Cross-provider agent coordination and verification

## Objective

Establish and evaluate a governed workflow using Claude Code CLI and OpenAI Codex CLI, combining independent role-policy assurance with bounded coordination and harness experiments. Determine whether it improves verified engineering outcomes, execution reliability, and total effort under an approved budget while preserving scientific semantics and authority boundaries.

This is an S3 programme composed of separately classified S1/S2 slices. Durable role, authority, execution, provenance, and surface-policy contracts are S2. Bounded adapter implementation can be S1 only where it implements an already approved stable contract without altering its meaning.

**Current state:** the Owner has accepted the architecture direction and authorized Issue/draft-specification preparation. This file is a draft proposal, not an approved normative specification or an execution grant. `last_reviewed` records preparation review only. Consequential implementation, native pilot execution, integration, policy adoption, and activation retain separate authorization requirements.

## Motivation

Cross-provider policy support and reliable CLI execution address related but independently verifiable problems. The existing PR #74 audit tests the role-policy foundation. A subsequent CLI pilot must establish its own permission enforcement, evidence integrity, recovery, and context independence. Controlled evaluation must then determine whether operational success translates into useful engineering improvement.

One programme coordinates these dependencies without transferring an acceptance result from one subject to another. Conversational Control Desk, scientific discussion, specifications, and applicable independent reviews retain their existing roles.

## Current baseline

### Source and implementation facts

- Repository baseline inspected for preparation: `Julia-meets-String-Theory/CYAxiverse.jl`, `vmm`, commit `fb8139ca18cc5a91c59c6b7d693e5174abb3bc27`, tree `0126b886db7dbdd1654e8122b767290c958343ef`.
- `AGENTS.md` governs repository work; approved specifications govern feature intent below higher-order project policy. Issues/Projects coordinate work and do not supersede the feature contract.
- `.agents/POLICIES/SURFACE_ALLOCATION.md` governs dispatch, surface overrides, permissions, and independent context. A task-specific override does not amend unrelated authority or scientific requirements.
- `.agents/skills/cyaxiverse-agent-orchestration/SKILL.md` preserves worker continuity and explicitly avoids hard-coding a particular model into repository-wide policy.
- Exchange Issue #16 excludes actor launch from its existing deterministic control-plane scope. Ordinary-Chat V0 review is an integration boundary, not an approved CLI transport.

No source/test audit of PR #74, native capability proof, CLI pilot result, or harness benefit is claimed by this preparation artifact. Revalidate current governing sources before formal review or execution.

### Owner direction and clarification chronology

The [governing Issue #209](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/209) records the sanitized preparation decision and **OC-001, equivalent role options**, dated 2026-10-22. Earlier mapping language was susceptible to a fixed provider/configuration interpretation. OC-001 resolves the intended architecture as follows:

1. The stated OpenAI/Anthropic models are equivalent eligible options at the corresponding governed role/capability step.
2. The Manager may select either eligible primary; the selected Manager provider does not fix other actors' providers.
3. The applicable governed fallback sequence follows primary selection. There is no requirement to use or exhaust OpenAI before choosing Anthropic, or vice versa.
4. Exact configuration identity is recorded for each actual attempt; it is not a programme-wide selection lock.
5. The Owner accepted architecture-direction and preparation work only. Operational policy adoption and execution remain separate decisions.

OC-001 is recorded here as intended-behavior clarification. It does not rewrite an older policy document, claim that new requirement IDs governed historical episodes, or independently update the private runtime-policy authority. Workstream A must record the requirements identity and effective scope used for every assessment.

### Existing evidence and hypotheses

Official CLI documentation describes structured programmatic execution; that does not establish the capabilities available to a particular account or prove execution correctness. Harness² motivates bounded proposal/probe/lesson experiments. Neither source establishes CYAxiverse scientific correctness or the benefit of this proposed workflow. Those are questions for the separately authorized pilot and evaluation.

## Scope

- **Workstream A:** preserve and independently assess the exact exchange PR #74 role-policy candidate, with the clarified intended selection behavior transparently distinguished from historical requirements.
- **Workstream B:** define the provider-neutral contract, bounded CLI pilot, controlled harness evaluation, and evidence-led adoption proposal.
- Shared role-level options, task decomposition, evidence identity, privacy, failure handling, independent-review separation, and authority mapping.
- Reuse or link existing memory, review, skills, and verification work without creating a second project-status authority.

## Non-scope

Preparation does not implement adapters, run native agents or a pilot, amend policy, merge or integrate changes, activate capabilities, release/publish scientific results, or change package versions. The programme does not replace SDD, scientific-owner approval, or independent review; change accepted scientific baselines/tolerances; require Harness² as a runtime; automatically launch browser Chat agents; or authorize unrestricted production credentials.

It does not reopen exchange Issue #16, change PR #74's frozen source candidate, or inherit its historical acceptance evidence without proving applicability.

## Scientific claim boundary

### This specification may establish

- A reviewed contract for governed agent selection, execution, and evidence.
- Measured workflow behavior on named, reproducible engineering/numerical cases under recorded conditions.
- Bounded evidence of benefits, regressions, uncertainty, cost, and lesson transfer.

### This specification must not establish

- Scientific correctness from model agreement, benchmark scores, or passing visible tests alone.
- A change in physical normalization, interpretation, accepted tolerance, population definition, schema, or scientific approval authority.
- Population-level/general effectiveness from a finite convenience sample without the applicable evidence.

## Fixed conventions and invariants

- Apply `AGENTS.md`, the relevant approved specifications, surface policy, and applicable handoff/handback protocols. Do not duplicate or silently override their authority order.
- The private Control Desk ledger remains authoritative for private policy. GitHub remains authoritative for engineering work state. CLI histories and local recovery journals are working evidence, not new normative authorities.
- Preserve the Owner's initial dispatch boundary. Manager delegation may occur only within an authorized envelope.
- Role eligibility, native capability, execution surface, permission, context independence, and authority are distinct dimensions.
- Preserve existing scientific, precision, physical-domain, API, persisted-schema, privacy, and environment invariants. Preserve regular-host Julia verification and the repository CI gate where applicable.
- Namespace programme gates as `CYAX-0209 G...`; keep exchange and other programmes' gates distinct.

## Requirements

These IDs describe this draft's prospective requirements. Once referenced by implementation they are immutable identifiers. They do not retroactively govern historical work.

### R-001 — Preparation authority and separate decisions

The programme shall record preparation, specification approval, exact-candidate review, native capability, pilot execution, integration, policy adoption, and activation as distinct decisions. Evidence in one category shall not manufacture another. A draft or a completed task checklist shall not be treated as operational authority.

### R-002 — Equivalent role options without provider lock

The architecture shall represent authorized **role/capability option sets**, with primary eligibility and fallback eligibility explicit. The Manager may select any eligible primary within the approved task envelope, including either declared equivalent even when both are available. Selecting an already authorized option shall not require a new Owner approval merely because its provider differs from a previous task.

Initial declared alternatives are:

| Role/capability step | Declared alternatives | Selection meaning |
| --- | --- | --- |
| Corresponding `gpt-6.1-sol` step | `gpt-6.1-sol` or `claude-opus-5-5` | Either eligible option may be selected for the applicable role. |
| Corresponding `gpt-6-luna` step | `gpt-6-luna` or `claude-haiku-5-5` | Either eligible option may be selected for the applicable step. |
| Additional Implementer fallback | `claude-sonnet-5-5` | Available in its governed fallback position; primary eligibility requires an applicable rule. |

This table is a declaration of selection options, not native availability proof, a one-to-one replacement instruction, or a permanent model-name lock. The architecture shall consume a governed, versioned option catalogue rather than embed these identifiers into provider-specific branching throughout orchestration. Catalogue changes follow the applicable policy mechanism; the Manager cannot declare an unapproved model equivalent on its own.

Per-role selections shall be independent where authorized. A Manager's provider shall not force all Implementers or reviewers into a homogeneous provider profile. Both CLI providers shall be assessed for Manager eligibility under the same governing criteria; actual use depends on the approved role, capability, surface, and permission combination.

### R-003 — Exact attempt provenance and supported configuration

Each attempt shall record the selected and observed provider, exact model, execution mode, native reasoning configuration, CLI/adapter versions, surface, permission profile, and governing option/authority identity. Receipts for one equivalent shall not count as receipts for another. A selected option must satisfy the required capability using supported native configuration; labels such as `low`, `high`, `xhigh`, and `max` shall not be silently translated across providers.

Pre-launch validation shall reject an unavailable, unsupported, unauthorized, or unverifiable invocation. Within its envelope, the Manager may select another eligible primary before any attempt starts, recording the choice. Once an attempt starts, subsequent changes follow R-004. Unavailability of one option shall not be reported as unavailability of all eligible options. Missing native evidence blocks the affected claim; it shall not be reported as PASS.

### R-004 — Primary, fallback, continuity, and concurrency

For each task, the Manager shall select one eligible primary and declare the applicable ordered fallback sequence and transition conditions. Transitions shall be sequential, preserve task state/evidence, and record why the previous attempt stopped. A retry shall not be relabeled as a new task or primary to bypass the ladder. Replanning requires authority within the existing envelope or escalation under that envelope.

The assigned worker normally owns diagnosis, edits, testing, correction, and retesting. Ordinary failures do not justify replacement. Preserve existing bounded checkpoint/stall recovery rules.

Concurrent primary workers are permitted only for distinct authorized tasks with allocated scopes/dependencies. Concurrent substitutes for one task are not sequential fallback. Combining changes produces a new integrated candidate requiring its own applicable verification and review. Worktrees alone do not satisfy all configuration, credential, service, and tool-channel isolation requirements.

### R-005 — Manager and trusted execution boundary

One designated Manager shall own each deliverable's decomposition, scope, integration, candidate freeze, evidence collection, and handback. A trusted host-side execution boundary shall enforce authority, allowed operations, paths, permissions, delegation depth, concurrency, resource budgets, and process lifecycle. Model-generated requests shall not widen that envelope.

Provider adapters shall remain thin translations and observers. Preserve `cyaxctl`'s existing Issue #16 no-launch scope. Any launcher must have separately approved ownership and contract. Reuse existing helpers where appropriate; the first pilot does not require a new general scheduler or memory service.

### R-006 — Versioned task contract

Each task packet shall bind: programme/deliverable/task/episode/attempt identities; contract version and canonical packet identity; governing specification and authority references/validity; role and objective; acceptance criteria; exact baseline and dependencies; allowed scope/worktree/operations/prohibitions; applicable option set, selected primary and fallback rule; effective execution/environment/budget profile; invariants; required checks and review axes; checkpoint/escalation conditions; and approved evidence destinations.

Each required field shall have a producer, verifier, and acceptance rule. A CLI envelope must be separately reviewed; ordinary-Chat V0 transport shall not be assumed to authorize execution.

### R-007 — Checked result and evidence contract

Results shall distinguish worker-reported outcome, observed process state, independently checked verification, and approval state. `DONE`, `BLOCKED`, and `FAILED` remain worker terminal outcomes; one predeclared `CHECKPOINT` is nonterminal. Exit zero, schema-valid output, or model-authored completion text alone shall not establish acceptance.

Record commands/checks actually run, exit statuses, observed results, missing checks, changed paths, candidate/artifact identities, assumptions, uncertainty, findings, blockers, usage/cost, and sanitized evidence. Bind provider events to the actual invocation and attempt. Reject missing, contradictory, tampered, or unsupported authority-bearing or terminal-result data. Streaming fragments remain incomplete until terminal requirements are satisfied. Informational extensions may be ignored only under an explicit safe version-compatibility rule.

### R-008 — Failure, interruption, and recovery

Define timeout, cancellation, crash, partial-write, checkpoint, retry, and Manager-loss behavior. Before retrying uncertain execution, reconcile descendants, partial artifacts, and completed effects. Stable task identity, unique attempts, and duplicate-action protection shall prevent accidental repeated actions. Uncertain effects remain blocked pending reconciliation.

Budgets cover all Managers, workers, reviewers, probes, and retries. A bounded local recovery journal may record execution facts but cannot create approval, policy, or durable project-status authority.

### R-009 — Independent review

Independent reviewers shall receive the exact immutable candidate, approved requirements/rubric, and a fresh isolated context with an auditable invocation/configuration record. Source and approved specifications are permitted common inputs; implementation conversations and unauthorized private memory are not. A new process or different provider alone does not establish independence.

Implementers and material integrators shall not certify their own candidate as independent. Preserve distinct applicable SPEC, STANDARDS, and SCIENTIFIC axes. One report may satisfy multiple formal sign-offs only if the governing protocol permits it. Candidate or requirements changes require explicit applicability analysis before reusing evidence or conclusions.

### R-010 — Privacy and authority evidence

Define protected data classes, approved processing, evidence destinations, and tool-channel boundaries before collection. Reject disallowed material before transport, hashing, logging, or publication where the governing contract prohibits those operations. Test with synthetic sentinels rather than private material.

Keep private-ledger contents, private locators, credentials, and conversation data out of repository artifacts. Use sanitized engineering evidence and a restricted annex only at an independently approved destination. A digest establishes content identity, not authorship or authorization. Inventory inherited hooks, plugins, instructions, memory, MCP settings, and credentials; validate effective controls across all enabled channels, not only the shell.

### R-011 — Scientific and test integrity

Keep scientific acceptance, evaluator-only checks, accepted tolerances, authority, and permission controls outside the experiment's editable scope. Use independent mathematical checks and appropriate authoritative references. Visible tests may be extended within the task contract, but weakening acceptance criteria shall not improve evaluation results.

Preserve the established Julia/environment verification route. Isolated adapter tests shall not be presented as equivalent package/scientific evidence when their environment changes relevant semantics. Report unavailable checks explicitly.

### R-012 — Exact PR #74 subject and versioned requirements

Workstream A shall preserve:

- Repository: `vmmhep/CYAxiverse-agent-exchange`.
- PR: [#74](https://github.com/vmmhep/CYAxiverse-agent-exchange/pull/74).
- Head branch: `control-desk/issue16-cross-provider-role-policy-r1`.
- Frozen candidate: `e959d56d4071df81ed2d7aca7380d0bf11c8a072`.
- Original base: `45ee9f4d94f01ff7507d42ff8c6724a218e2f8b3`.
- Declared changed files: `tools/CYAXControl/src/RolePolicy.jl`, `tools/CYAXControl/src/PrivacyBoundary.jl`, and `tools/CYAXControl/test/runtests.jl`.

Independently verify remote identity, ancestry, tree, complete diff, and actual changed paths. Distinguish any later head. The original source candidate remains unchanged by OC-001.

Before a formal verdict, freeze the review handoff, references, and requirements identity. Record OC-001 as a versioned clarification of intended selection behavior, including its relationship to the original owner-directed policy contract. Assess historical-contract compliance and clarified intended behavior separately where they differ. Do not retroactively edit the historical contract or inherit a verdict across changed requirements. The new programme's requirement IDs shall not be projected onto historical execution.

### R-013 — Preserve the adversarial review obligations

The read-only review shall establish its authoritative baseline independently, including Issue #16, the repository-relative owner-directed policy contract, applicable runtime/handoff/scoped-policy decisions, expiry/supersession, and `EFFICIENCY_MEASURES.md` Version 2. Retrieve canonical private policy through the authorized Google Drive connector or an independently approved successor route. Do not reconstruct unavailable authority from assertions.

Mandatory axes are:

1. **Policy:** all required roles; equivalent role options and authorized mixed-provider selection; correct primary/fallback semantics; adoption/digest/scope/time boundaries; preservation of OpenAI defaults and V1 historical bindings; protection against retroactive reinterpretation and V2-to-V1 downgrade acceptance.
2. **Validators and callers:** inspect `validate_policy_document`, `match_packet_roles`, `selected_roles`, `adopted_policy_from_source`, `require_resolved_policy!`, direct `policy["roles"]` consumers, and handoff/preflight/readiness/dispatch/P0/handback/reviewer-capability paths, including callers outside the diff. Attack malformed, duplicate, unknown, missing, mismatched, and stale inputs.
3. **Privacy/provenance:** strict scoped token grammar; aliases and provider/model mismatch; mode/effort confusion; caller-authored native proof; wrong thread/turn/context; replay/cross-episode reuse; prohibited propagation before privacy rejection.
4. **Independence:** self-review, inherited implementation context, omitted axes, wrong candidate/rubric/capability, historical PASS reuse, and Handoff Review PASS misused as dispatch/scientific acceptance.
5. **Tests:** focused RolePolicy/PrivacyBoundary coverage, applicable CYAXControl regression coverage, and required authorized native checks. Synthetic fixtures do not establish native capability. Record actual commands, versions, exits/results, and missing tests with minimal reproduction specifications. The reviewer makes no implementation changes.

The report shall provide subject identity, requirement coverage, P0-P3 findings with precise code or evidence locations, mechanisms/consequences, smallest corrections, closure verification, actual tests/gaps, and bounded next work. Use `PASS`, `PASS_WITH_NONBLOCKING_FINDINGS`, `REQUEST_CHANGES`, or `BLOCKED_INSUFFICIENT_EVIDENCE` for the declared acceptance target. A demonstrated mandatory defect takes `REQUEST_CHANGES` precedence; missing evidence remains explicit. Without a demonstrated blocker, missing mandatory acceptance evidence yields `BLOCKED_INSUFFICIENT_EVIDENCE`.

Report code acceptance, native test completeness, provider-native capability, private-policy adoption readiness, and dispatch/activation readiness separately. The formal charter determines required roles and native evidence; this programme does not dispatch that review or grant its verdict. Preserve the original review priority/capability only where still applicable under its independently verified authority.

### R-014 — Bounded pilot conformance

After specification approval and separate execution authorization, demonstrate both Claude-Manager/Codex-worker and Codex-Manager/Claude-worker arrangements; independent per-role choices; a fresh read-only reviewer of a frozen candidate; and reconstruction of its evidence. Exercise separately scoped concurrency and aggregate-candidate verification.

Also exercise invalid authority/capability, forbidden writes, scope escape, contaminated review context, malformed/partial output, false completion, interruption, retry after uncertain completion, and Manager loss. Use synthetic or replayable tasks with no production-side effects and explicit resource/credential limits. Native option availability and adapter support shall be reported per tested configuration rather than generalized to every option.

### R-015 — Controlled evaluation

Predeclare task snapshots, environment, role-to-model assignments, budgets, outcomes, repeat/run-order policy, stopping rules, uncertainty reporting, and minimum benefit required for adoption. Separate development cases from held-out evaluation cases, preferably by related defect family. Hide reference fixes, evaluator-only checks, answer-bearing later discussion, and held-out lessons. Include clean cases and plausible incorrect repairs.

Where budget permits compare:

| Condition | Instructions | Verification |
| --- | --- | --- |
| A | Current approved baseline | Current approved baseline |
| B | Mechanism-first variant | Current approved baseline |
| C | Current approved baseline | Enhanced verification |
| D | Mechanism-first variant | Enhanced verification |

Keep model/configuration fixed within a matched comparison to isolate the intervention. **Experimental control is not an operational provider lock.** Evaluate coordination separately against an approved single-worker baseline; record all role/model assignments. An A-versus-D-only budget supports a conclusion about the combined intervention, not attribution to individual components.

Measure verified correction, introduced regressions, detection, false-positive blocking, false acceptance, test integrity, provenance, recovery, latency, human intervention, and full cost including failed probes and all actor calls. Small-sample conclusions remain exploratory. Mandatory authority/scientific gates cannot be traded away for an aggregate score.

### R-016 — Lessons and research inputs

The first evaluation shall use frozen predeclared instruction variants. Any recursive or online self-evolution experiment requires separately approved scope and controls. Extract lessons from development cases, freeze the lesson bundle before transfer evaluation, and follow the existing candidate/validated/promotion lifecycle. No automatic normative promotion is permitted.

Use Harness² as methodological input only. Its implementation's documented unrestricted networking/disabled sandbox defaults are not adopted. Identify the exact title/version/source and intended contribution of the separately mentioned AI-agent engineering benchmark before claiming benchmark alignment.

### R-017 — Evidence-led adoption and historical continuity

Adoption proposals shall be limited to supported surface-default changes, orchestration procedures, evidence helpers, instructions, and validated lessons. Document regressions, negative results, uncertainty, support cost, ownership, and rollback/revocation conditions. Normative adoption requires the applicable reviews, exact identity, authority, and effective interval.

Preserve historical episode interpretation and explicit expiry, supersession, and revocation handling for future actions, including in-flight episodes at a boundary. Pilot acceptance alone shall not enact policy, merge code, or activate actors.

## Acceptance gates

### CYAX-0209 G0 — Definition prepared

**Objective:** produce the governing Issue and draft specification incorporating OC-001.

**Acceptance:** canonical draft location and reviewable preparation PR; requirements, audit identity, authority boundaries, pilot/evaluation plan, related work, and unresolved decisions stated; no claim of normative approval.

**Stop condition:** unresolved publication safety or scope that would require execution/implementation. Continue unaffected drafting where possible.

### CYAX-0209 G1A — Independent policy audit

**Objective:** obtain the separately governed exact-candidate review.

**Acceptance:** frozen source/requirements/handoff identities, mandatory coverage, actual evidence and scoped verdict, and the five separate readiness judgments from R-013.

**Stop condition:** mandatory authority/identity evidence unavailable, review-independence violation, or defect requiring correction. Missing evidence must not erase feasible static findings. This gate does not grant CLI execution authority.

### CYAX-0209 G1B — Architecture/specification approval

**Objective:** approve the contract and resolve normative decisions for the intended first slice.

**Acceptance:** required review roles and durable approval reference bound to the exact reviewed revision; approved ownership, option-selection/fallback rules, enforcement boundary, privacy/review criteria, and evaluation design.

**Stop condition:** unresolved normative intent or missing required review. Drafting may proceed alongside G1A; dependencies are determined by behavior actually invoked.

### CYAX-0209 G2 — Implementation and pilot authorization

**Objective:** authorize a bounded slice with explicit actions and budget.

**Acceptance:** separate implementation and native-execution grants as applicable; current operational policy dependencies; approved actor/options/surface/permission envelope; failure limits, environment route, evidence destinations, and test plan. Authorization may name an option set without permanently fixing one provider.

**Stop condition:** missing scope/authority, unsupported configuration, or inability to enforce the envelope. An authorized alternative may be selected under R-002/R-004; this is not permission to enlarge the option set.

### CYAX-0209 G3 — Operational conformance

**Objective:** demonstrate pilot-applicable R-002 through R-011 and R-014 for the approved configuration. R-012/R-013 remain in G1A; G2 checks any policy-audit dependencies of the behavior actually invoked.

**Acceptance:** both coordination directions, selected-option provenance, independent review, bounded concurrency/integration, prohibited-operation rejection, and recovery evidence pass their predeclared criteria.

**Stop condition:** authority/permission escape, unreconciled side effects, invalid acceptance evidence, or required verification missing. Partial success remains scoped.

### CYAX-0209 G4 — Evaluation concluded

**Objective:** determine verified benefit and cost under R-015/R-016.

**Acceptance:** matched comparisons and holdouts, per-case results, full cost, integrity checks, negative results, uncertainty, and a supported recommendation against predeclared thresholds.

**Stop condition:** evaluator leakage, changed acceptance criteria, invalid scientific oracle, or uncontrolled intervention differences that prevent the proposed inference. A sound negative result may satisfy this gate.

### CYAX-0209 G5 — Adoption decision and authorized activation

**Objective:** make and execute only the separately approved adoption decision.

**Acceptance:** explicit adopt/narrow/revise/do-not-adopt decision; any integration, policy update, and activation individually authorized and evidenced; rollback/revocation responsibilities and historical continuity recorded.

**Stop condition:** missing required decision, failed adoption criteria, or activation conditions unsatisfied. A documented decision not to adopt can close the programme without claiming implementation/activation success.

## Verification requirements

Verification must be proportional to each slice and must report checks actually run. Preparation uses content/structure, reference, privacy, and exact-file scope checks; it does not claim Julia/native tests or a formal independent architecture verdict.

Minimum subsequent scenarios include:

| Scenario | Expected evidence |
| --- | --- |
| Either declared equivalent selected while both are eligible/available | Both choices accepted without a provider priority or new Owner approval inside the same envelope. |
| OpenAI Manager with Anthropic Implementer and independently eligible reviewer; reverse direction | Role-level authorization without a provider-wide profile lock. |
| Different tasks choose different eligible primaries | No inherited global configuration constraint. |
| Primary unavailable before start | That invocation blocked; permitted reselection recorded without claiming every alternative unavailable. |
| Failure after start | Declared sequential transition, continuity, reconciled effects, and new attempt identity. |
| Retry disguised as new primary, or concurrent fallback for same task | Rejection unless separately authorized replanning actually permits it. |
| Receipt names an equivalent different from actual/selected execution | Discrepancy rejected or reconciled; no identity substitution. |
| Manager invents an equivalent, native effort mapping, or permission | Rejection outside the governed option/configuration envelope. |
| Sonnet appears only as fallback but is used as primary | Rejection unless applicable primary eligibility exists. |
| Requirements change while PR #74 bytes stay fixed | New requirements identity; explicit historical/current-intent distinction and evidence reuse analysis. |
| V1 history, malformed V2, expired/tampered scope, replayed native proof | Correct preserved historical binding or rejection, as applicable. |
| Scientific defect and plausible wrong repair | Independent oracle, immutable tolerances, regression/test-integrity evidence. |
| Crash, cancellation, Manager loss, uncertain completion | No unaccounted descendant or duplicate action; truthful blocked/failed/partial reporting. |

Required native checks need an authorized route/operator. A read-only review can inspect existing verifiable receipts without being granted launch authority. Candidate freeze/review evidence must identify code and governing requirements separately.

## Interfaces and compatibility

- Draft a versioned provider-neutral task/result/evidence envelope and thin adapters. The final wire schema, compatibility rules, subprocess implementation, and native configuration mapping belong to the approved slice's subordinate plan.
- Preserve existing V1 bindings and strict parsing. Do not silently reinterpret a malformed V2 record through V1 or widen accepted token grammar outside its scope.
- Keep ordinary-Chat V0, private policy authority, deterministic control-plane functions, and new execution transport distinct.
- Preparation changes no public Julia API, scientific behavior, persisted scientific schema, supported package environment, or package version. Package-infrastructure SemVer impact: none for preparation. Implementation slices must declare their actual impact separately.
- Agent tooling must not become a runtime dependency of `using CYAxiverse`.

## Dependencies and blockers

### Required governing sources

`AGENTS.md`; `.agents/POLICIES/SURFACE_ALLOCATION.md`; relevant orchestration, SDD, and Julia-quality skills; applicable validated lessons; `scripts/agent_verify.py`; `CLAUDE.md` and tool adapters; applicable exchange handoff/handback protocols; independently retrieved private authority where required.

For Workstream A also verify `handoff-reviews/control-desk-cross-provider-eligibility/r1/owner-directed-policy-contract-v1.json`, current `CD-RUNTIME-0001`, `CD-HANDOFF-CAPABILITY-0001`, and scoped `CD-ISSUE16-ROLE-POLICY-0001` with expiry/supersession. Their presence as references is not proof of authenticity, currentness, adoption, or access.

### Related programmes

- [Exchange Issue #16](https://github.com/vmmhep/CYAxiverse-agent-exchange/issues/16) and [PR #74](https://github.com/vmmhep/CYAxiverse-agent-exchange/pull/74): control-plane/role-policy lineage and independent audit.
- [#162](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/162): durable memory and cold-start reconstruction.
- [#180](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/180): procedural memory/lesson transfer; no automatic self-evolution permission.
- [#182](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/182): ordinary-Chat V0 Handoff Review runtime.
- [#198](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/198): engineering skills/scientific certification; coordinate reuse and preserve separate acceptance.

Related work is not automatically a blocking dependency or adopted implementation. Add native dependency relationships only for a concrete blocked outcome. Architecture preparation does not require every policy audit conclusion; execution must satisfy the dependencies of invoked behavior.

### Primary research/interface references

- [Codex non-interactive execution](https://developers.openai.com/codex/non-interactive-mode).
- [Codex approvals/security](https://developers.openai.com/codex/agent-approvals-security).
- [Claude Code programmatic execution](https://code.claude.com/docs/en/headless).
- [Claude Code shell sandbox scope](https://code.claude.com/docs/en/sandboxing).
- [Harness² paper v1](https://arxiv.org/html/2610.14224v1) and [reference implementation](https://github.com/google-research/harness2).

Pin actual execution versions for evidence; those versions are observations/experimental controls, not permanent architecture selections.

## Open owner decisions

The equivalent-option principle and preparation authorization are settled by OC-001. Do not ask the Owner to approve them again.

Before the relevant later gate, resolve:

1. Launcher/adapter repository ownership and first independently reviewable S1/S2 slice.
2. Authoritative option catalogue, per-role primary/fallback eligibility, native effort mappings, and trusted capability evidence source. Primary provider choice within an approved set remains delegated to the Manager.
3. Any operational policy adoption dependencies, scope validity, and revocation/in-flight handling not already settled by governing sources.
4. Pilot envelope: actor/delegation limits, allowed operations, runtime/verification route, credentials, budgets, and recovery policy.
5. Frozen evaluation set, exact engineering benchmark reference, independent scientific oracles, repeats/budgets, and adoption thresholds.
6. Applicable formal review roles, review-context controls, evidence/report destinations, and exact review candidates.

No unresolved decision here blocks the authorized preparation deliverable. None may be guessed to authorize later consequential work.

## Completion criterion

Preparation is complete when CYAX-0209 G0 is evidenced by this governing draft and its linked preparation PR. The specification remains draft and the programme remains open.

The programme is complete only when the relevant audit, contract approval, authorized pilot, and controlled evaluation have produced their scoped evidence and the Owner has recorded an explicit adoption decision. If adoption is approved, the necessary integration/policy/activation actions require their separate authorization and evidence. A defensible negative result and an explicit do-not-adopt decision can complete the research objective. A successful demonstration, merged draft, or collection of agent PASS messages cannot by itself satisfy completion.
