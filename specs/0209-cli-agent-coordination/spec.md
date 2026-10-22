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

**Working classification: provisional S3 programme**, composed of independently reviewable S1/S2 slices. Durable role, authority, execution, provenance, and surface-policy contracts, including the trusted launcher, require S2 treatment. Thin adapters can be S1 only when they implement an approved stable contract without changing its meaning. If review finds an S3 umbrella unnecessary, reclassify or divide it into linked S2 specifications; preserve requirement/evidence identities and every applicable security review and authorization gate.

**Current state:** the Owner accepted the architecture direction and authorized Issue/draft-specification preparation. `status: draft` and `approval_ref: N/A` remain accurate; `last_reviewed` records preparation review only. The acceptance-gate table consolidates the separate decision boundaries.

## Motivation

Cross-provider policy support and reliable CLI execution address related but independently verifiable problems. The existing PR #74 audit tests the role-policy foundation. A subsequent CLI pilot MUST establish its own permission enforcement, evidence integrity, recovery, and context independence. Controlled evaluation MUST then determine whether operational success translates into useful engineering improvement.

One programme coordinates these dependencies without transferring an acceptance result from one subject to another. Conversational Control Desk, scientific discussion, specifications, and applicable independent reviews retain their existing roles.

## Current baseline

### Source and implementation facts

- **Governance baseline:** project-policy claims bind to `Julia-meets-String-Theory/CYAxiverse.jl`, `vmm`, commit `fb8139ca18cc5a91c59c6b7d693e5174abb3bc27`, tree `0126b886db7dbdd1654e8122b767290c958343ef`.
- **Proposal candidate:** the initial draft commit `1c174adc1f906fe17151bafe16f8b801ca1a3ddf` is a direct child of that baseline and added only this specification. It changed no governing policy files. PR #210 records each subsequent draft candidate separately; a later proposal commit does not silently change the governance snapshot. Formal review/execution MUST revalidate applicable sources and record supersession.
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

OC-001 is recorded here as intended-behavior clarification. It does not rewrite an older policy document, claim that new requirement IDs governed historical episodes, or independently update the private runtime-policy authority. Workstream A MUST record the requirements identity and effective scope used for every assessment.

### Existing evidence and hypotheses

Official CLI documentation describes structured programmatic execution; that does not establish the capabilities available to a particular account or prove execution correctness. Harness² motivates bounded proposal/probe/lesson experiments. Neither source establishes CYAxiverse scientific correctness or the benefit of this proposed workflow. Those are questions for the separately authorized pilot and evaluation.

## Scope

- **Workstream A:** preserve and independently assess the exact exchange PR #74 role-policy candidate, with the clarified intended selection behavior transparently distinguished from historical requirements.
- **Workstream B:** define the provider-neutral contract, bounded CLI pilot, controlled harness evaluation, and evidence-led adoption proposal.
- Shared role-level options, task decomposition, evidence identity, privacy, failure handling, independent-review separation, and authority mapping.
- Reuse or link existing memory, review, skills, and verification work without creating a second project-status authority.

## Non-scope

The programme preserves SDD, scientific-owner approval, independent review, accepted scientific baselines/tolerances, Chat roles, and existing durable authorities. It excludes a mandatory Harness² runtime, automatic browser-agent launch, unrestricted production credentials, package-version changes, and reopening exchange Issue #16. R-012/R-013 preserve the independent PR #74 subject and charter.

## Scientific claim boundary

### This specification may establish

- A reviewed contract for governed agent selection, execution, and evidence.
- Measured workflow behavior on named, reproducible engineering/numerical cases under recorded conditions.
- Bounded evidence of benefits, regressions, uncertainty, cost, and lesson transfer.

### This specification MUST NOT establish

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

Uppercase MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY use the [RFC 2119/8174 convention](https://www.rfc-editor.org/info/rfc8174/). They identify prospective requirements; only the applicable specification approval makes them governing. Descriptive facts and proposed decision values confer no authority. R-001–R-017 remain the canonical IDs; dotted child IDs refine coverage without creating a second requirement system. They do not retroactively govern historical work.

### R-001 — Preparation authority and separate decisions

The programme MUST record preparation, specification approval, exact-candidate review, native capability, pilot execution, integration, policy adoption, and activation as distinct decisions. Evidence in one category MUST NOT manufacture another. A draft or a completed task checklist MUST NOT be treated as operational authority.

### R-002 — Equivalent role options without provider lock

The architecture MUST represent versioned, authorized role/capability option sets with primary and fallback eligibility. The Manager MAY choose either eligible primary even when both are available; its provider MUST NOT determine the provider of Implementers or reviewers. Choosing an already authorized option within the task envelope requires no fresh Owner approval. Both CLI providers MUST be assessed for Manager eligibility under the same criteria.

OC-001 establishes **selection eligibility**, not empirical performance equality, universal suitability, or interchangeable native effort parameters. The catalogue MUST record role/task scope, permitted surface/permissions, qualification evidence, validity, and review triggers. Model identifiers in R-003 are initial catalogue entries, not a permanent architecture lock.

**R-002.1 — Evidence can change eligibility.** Before evaluation, the protocol MUST predeclare role/task quality floors and review triggers. A mandatory integrity failure or a predeclared quality-floor failure MUST generate a scoped suitability finding and restriction/requalification proposal. The launcher MUST apply preapproved pilot stop/quarantine rules. Temporary quarantine, permanent revocation, and statistical uncertainty MUST remain distinct. Durable restriction, reinstatement, and in-flight handling require the applicable policy decision; evaluators MUST NOT rewrite the operational catalogue or historical baseline. An inconclusive comparison proves neither equality nor inferiority.

### R-003 — Explicit options, native effort, and attempt identity

The following is the **draft catalogue for qualification**, based on primary documentation inspected on 2026-10-22. Each row has its own native values; identical labels do not imply equal computational effort or performance.

| Option | Provider / interface | Role/capability membership | Documented effort values | Proposed pilot effort values |
| --- | --- | --- | --- | --- |
| `gpt-6.1-sol` | OpenAI / model API; Codex use requires client qualification | Sol/Opus option step; applicable roles require catalogue authorization | `low`, `medium`, `high`, `xhigh`, `max` | `low`, `medium`, `high`, `xhigh`, `max` |
| `claude-opus-5-5` | Anthropic / Claude Code | Sol/Opus option step; applicable roles require catalogue authorization | `low`, `medium`, `high`, `xhigh`, `max` | `low`, `medium`, `high`, `xhigh`, `max` |
| `gpt-6-luna` | OpenAI / model API; Codex use requires client qualification | Luna/Haiku option step; applicable roles require catalogue authorization | `none`, `low`, `medium`, `high`, `xhigh`, `max` | `low`, `medium`, `high`, `xhigh`, `max` |
| `claude-haiku-5-5` | Anthropic / Claude Code | Luna/Haiku option step; applicable roles require catalogue authorization | `low`, `medium`, `high`, `xhigh`, `max` | `low`, `medium`, `high`, `xhigh`, `max` |
| `claude-sonnet-5-5` | Anthropic / Claude Code | Additional Implementer fallback; no automatic primary/other-role eligibility | `low`, `medium`, `high`, `xhigh`, `max` | `low`, `medium`, `high`, `xhigh`, `max` |

Sources: [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [Claude Code model/effort configuration](https://code.claude.com/docs/en/model-config), and [Codex model selection](https://developers.openai.com/codex/models).

The exact supplied identifier `gpt-6-luna` is retained; no `6.1` rename is inferred. `medium` is explicitly included; its omission from earlier example lists was not a justified exclusion. API documentation, client availability, account entitlement, and governed role permission MUST remain separate evidence. Every row is pending qualification of its pinned CLI/client, transport, effective effort/mode, and role-specific allowed subset. An approved task MAY use a narrower qualified subset; it MUST NOT silently substitute labels.

**R-003.1 — Effective configuration.** Each attempt MUST distinguish requested configuration, client-resolved/transmitted configuration, and provider-reported effective metadata where available, including provider/model and native effort/mode. The recorder MUST identify each observation's trusted source and limitations, CLI/adapter versions, permission/surface profile, and governing catalogue/authority identities. The qualification plan MUST define sufficient evidence for each claim. Missing provider metadata MUST remain missing; copying a requested value into an observed field is not proof of effective behavior or computation spent. Preflight MUST reject unsupported, unavailable, unauthorized, or unverifiable invocations. Any native effort downgrade or model rerouting MUST be rejected or handled as an explicit authorized transition under R-004, never accepted under the requested identity. The catalogue MUST separately govern orchestration modes that create subagents; scalar effort authorization alone does not authorize delegation. Initial serial trials disable such native delegation.

Before launch, the Manager MAY reselect another qualified eligible primary without enlarging authority. After launch, R-004 governs transitions. Receipts for one equivalent MUST NOT count as receipts for another. Missing capability evidence blocks the affected claim, not every alternative.

### R-004 — Explicit sequential fallback and continuity

For each task, the Manager MUST select one eligible primary and bind an ordered finite fallback list, transition conditions, capability floor, attempt limit, and remaining budget to the packet. Logical task identity persists across attempts. Relabeling a retry as a new primary MUST NOT bypass the ladder.

**R-004.1 — Proposed availability ladder (approval input DEC-02).** This order is a concrete proposal, not an assertion of an already adopted ranking:

| Starting role / step | Proposed sequence after the selected primary |
| --- | --- |
| Implementer at Sol/Opus step | Untried eligible Sol/Opus counterpart → Sonnet → selected eligible Luna/Haiku option → untried eligible Luna/Haiku counterpart → STOP |
| Implementer at Luna/Haiku step | Untried eligible Luna/Haiku counterpart → STOP |
| Manager or independent reviewer | Untried eligible counterpart at the authorized step → STOP; no inherited Implementer downgrade ladder |

The Manager chooses either provider first within a pair. In this proposed order, an eligible same-step counterpart is tried before a lower stage; changing that rule is a DEC-02 decision, not a runtime shortcut. Sonnet precedes the Luna/Haiku stage in the proposed Sol/Opus ladder; moving from a Luna/Haiku primary to Sonnet is replanning, not a late upward fallback. Each entry MUST independently satisfy the task's approved capability floor and role restrictions; ineligible entries are skipped without lowering the floor. Cross-step use requires explicit catalogue qualification. The first serial pilot SHOULD qualify same-step failover first; later stages remain inactive until approved and qualified.

**R-004.2 — Transition conditions.**

| Observed condition | Required action |
| --- | --- |
| Provider unavailable, rate limit beyond the packet's wait allowance, or unrecoverable transport/process failure | Reconcile the old attempt; advance to the next eligible entry if budget/attempt limits permit. |
| Concrete stall after the existing bounded checkpoint/recovery procedure | Reconcile and replace within the approved floor; silence alone is insufficient. |
| Ordinary failed test or repairable implementation error | Same worker diagnoses, corrects, and retests; no automatic fallback. |
| Evidence that the task exceeds the chosen capability or has ambiguous scientific intent | Block and replan/escalate within existing authority; do not downgrade as a response to difficulty. |
| Budget exhausted, authority invalid, safety boundary failed, or effects uncertain | Stop; resume only after the applicable reconciliation/authorization. A cheaper model does not reset the budget or cure an authority failure. |

Distinct authorized tasks MAY later have concurrent primaries under R-014. Parallel alternatives for one task MUST NOT be presented as sequential fallback. Any aggregate candidate requires its own identity and applicable checks.

### R-005 — S2 launcher trust model

One designated Manager owns decomposition, scope, integration requests, candidate freeze, evidence collection, and handback. **The launcher, authority verifier, credential broker, reviewer-packet assembler, and any publication finalizer form a separately reviewed S2 trust boundary.** Thin provider adapters translate requests and observe events; they do not own authority.

**R-005.1 — Custody and bypass prevention.** Worker-provider credentials MUST be held only in the trusted launcher/transport domain and MUST NOT be reachable by the Manager's model-accessible environment or tools. The Manager's own legitimate inference transport MAY authenticate through a protected runtime/broker, but tools and descendants MUST NOT obtain reusable credentials or unrestricted provider-call capability from environment variables, files, keychains, helpers, process inspection, descriptors, sockets, or equivalent channels. A shared credential-bearing environment with general model-controlled shell access fails this requirement. These custody, transport-isolation, and bypass-prevention rules apply to every model-accessible Manager, Implementer, and reviewer context and all tools/descendants, regardless of process ancestry.

Only the trusted boundary MAY create an execution attempt. It MUST validate current authority, role/option, scope, permissions, resource budgets, delegation depth, and concurrency. Manager tools MUST be unable to invoke provider CLIs directly or bypass the launcher using renamed binaries, interpreters/SDKs, direct APIs, native subagents, MCP, hooks, plugins, browser integrations, or local services. Candidate-controlled configuration MUST NOT expand these channels. The supported-channel inventory, effective controls, trusted components, and residual assumptions MUST be reviewable. An executable deny-list or prompt instruction alone is insufficient.

**R-005.2 — Independent qualification.** Before native orchestration, the S2 launcher review MUST establish an enforceable separation between protected transport and untrusted tools for the selected CLI/host configurations. Denial/reconciliation scenarios in the verification table MUST be exercised under a bounded G2 qualification-test grant. Passing those checks is a prerequisite for a later autonomous-orchestration grant, not for obtaining the narrow qualification-test grant. Current provider docs describe narrower shell/proxy boundaries, so native flags MUST NOT be assumed to cover every channel. If the separation is infeasible, the configuration is ineligible for autonomous coordination; a separately scoped planning-only/operator-mediated path MAY be evaluated and labeled accordingly.

**R-005.3 — Local-only pilot effects.** Model-accessible Managers, workers, and reviewers MUST have no remote repository publication credentials or write-capable Git/API/MCP/browser channel. Workers return patches or local commits; reviewers write only designated reports/build outputs. A separately trusted finalizer MAY publish or integrate only an exact artifact/destination under the applicable grant. Its existence does not grant that authority.

**Placement proposal (DEC-01):** a separate `tools/CYAXLauncher/` component in `vmmhep/CYAxiverse-agent-exchange`, with its own S2 issue/specification and named maintainer. This path is proposed, not created or approved. Preserve `CYAXControl`/`cyaxctl`'s Issue #16 no-launch contract; orchestration MUST NOT become a Julia package runtime dependency. The first pilot needs this reviewed boundary, not a general scheduler or memory service.

### R-006 — Versioned task contract

Each task packet MUST bind: programme/deliverable/task/episode/attempt identities; contract version and canonical packet identity; governing specification and authority references/validity; role and objective; acceptance criteria; exact baseline and dependencies; allowed scope/worktree/operations/prohibitions; applicable option set, selected primary and fallback rule; effective execution/environment/budget profile; invariants; required checks and review axes; checkpoint/escalation conditions; and approved evidence destinations.

Each required field MUST have a producer, verifier, and acceptance rule. A CLI envelope MUST be separately reviewed; ordinary-Chat V0 transport MUST NOT be assumed to authorize execution.

### R-007 — Checked result and evidence contract

Results MUST distinguish worker-reported outcome, observed process state, independently checked verification, and approval state. `DONE`, `BLOCKED`, and `FAILED` remain worker terminal outcomes; one predeclared `CHECKPOINT` is nonterminal. Exit zero, schema-valid output, or model-authored completion text alone MUST NOT establish acceptance.

The result recorder MUST record commands/checks actually run, exit statuses, observed results, missing checks, changed paths, candidate/artifact identities, assumptions, uncertainty, findings, blockers, usage/cost, and sanitized evidence. The adapter MUST bind provider events to the actual invocation and attempt. The verifier MUST reject missing, contradictory, tampered, or unsupported authority-bearing or terminal-result data. Streaming fragments MUST remain incomplete until terminal requirements are satisfied. Informational extensions MAY be ignored only under an explicit safe version-compatibility rule.

### R-008 — Failure, interruption, and recovery

The execution contract MUST define timeout, cancellation, crash, partial-write, checkpoint, retry, and Manager-loss behavior. Before retrying uncertain execution, the host MUST reconcile descendants, partial artifacts, and completed effects. Stable task identity, unique attempts, and duplicate-action protection MUST prevent accidental repeated actions. Uncertain effects MUST remain blocked pending reconciliation.

Budgets MUST cover all Managers, workers, reviewers, probes, and retries. A bounded local recovery journal may record execution facts but cannot create approval, policy, or durable project-status authority.

### R-009 — Host-authored independent review

Independent reviewers MUST receive an exact frozen candidate, independently established requirements/rubric, and a fresh context with an auditable invocation/configuration. Different providers or fresh processes alone do not establish independence. Implementers and material integrators MUST NOT certify their own candidate as independent. SPEC, STANDARDS, and SCIENTIFIC axes retain their governing sign-off rules.

**R-009.1 — Packet authorship.** The trusted launcher MUST deterministically assemble the complete review packet from a versioned host-controlled template, verified candidate/base identities and file manifest, authoritative requirements, approved rubric/required axes, reviewer assignment, and required host-recorded evidence including failures. The host MUST record template, input, and materialized packet identities. The Manager MAY request review of a candidate; it MUST NOT author normative reviewer instructions, choose a weaker rubric, omit required inputs, or substitute its narrative for evidence.

Manager-authored claims MAY appear only where the protocol permits them, segregated as untrusted advisory material. Candidate source/logs are review data, not authority to change instructions. The host MUST control resumed sessions, discovered memory, hooks, plugins, and tool settings; supply read-only candidate access and separate writable report/build locations; and independently verify the freeze at startup. A changed candidate, packet, or requirements identity requires a new bound attempt and evidence-applicability analysis. Deterministic packet assembly is a control against Manager contamination, not proof that all prompt-injection risk is eliminated.

### R-010 — Privacy and authority evidence

The privacy contract MUST define protected data classes, approved processing, evidence destinations, and tool-channel boundaries before collection. The boundary MUST reject disallowed material before transport, hashing, logging, or publication where the governing contract prohibits those operations. Privacy tests MUST use synthetic sentinels rather than private material.

Repository artifacts MUST exclude private-ledger contents, private locators, credentials, and conversation data. Reports MUST use sanitized engineering evidence; any restricted annex MUST use an independently approved destination. A digest establishes content identity, not authorship or authorization. The launcher MUST inventory inherited hooks, plugins, instructions, memory, MCP settings, and credentials, and MUST validate effective controls across all enabled channels.

### R-011 — Scientific and test integrity

The evaluation MUST keep scientific acceptance, evaluator-only checks, accepted tolerances, authority, and permission controls outside its editable scope. It MUST use independent mathematical checks and appropriate authoritative references. Visible tests MAY be extended within the task contract; acceptance criteria MUST NOT be weakened to improve evaluation results.

Verification MUST preserve the established Julia/environment verification route. Isolated adapter tests MUST NOT be presented as equivalent package/scientific evidence when their environment changes relevant semantics. Operators MUST report unavailable checks explicitly.

### R-012 — Exact PR #74 subject and versioned requirements

Workstream A MUST preserve:

- Repository: `vmmhep/CYAxiverse-agent-exchange`.
- PR: [#74](https://github.com/vmmhep/CYAxiverse-agent-exchange/pull/74).
- Head branch: `control-desk/issue16-cross-provider-role-policy-r1`.
- Frozen candidate: `e959d56d4071df81ed2d7aca7380d0bf11c8a072`.
- Original base: `45ee9f4d94f01ff7507d42ff8c6724a218e2f8b3`.
- Declared changed files: `tools/CYAXControl/src/RolePolicy.jl`, `tools/CYAXControl/src/PrivacyBoundary.jl`, and `tools/CYAXControl/test/runtests.jl`.

The reviewer MUST independently verify remote identity, ancestry, tree, complete diff, and actual changed paths, and MUST distinguish any later head. The original source candidate remains unchanged by OC-001.

**Formal verdict target:** the frozen candidate against the independently established original requirements applicable to that candidate. The reviewer MUST freeze that baseline, handoff bytes, and references before a verdict. **OC-001 compatibility:** report prospective option-set gaps separately to inform a successor candidate. Every finding/test MUST be tagged original-baseline, OC-001, or both, with an applicability reason. A newly clarified gap MUST NOT become a fabricated historical violation.

Before commissioning expensive native checks, Workstream A's governing authority SHOULD record whether to retain the complete original audit or approve a successor charter focused on retained V1/validator/privacy/provenance infrastructure. Until explicitly revised, the complete existing obligations remain. This programme MUST NOT silently narrow, cancel, or supersede the independent review.

### R-013 — Preserve the adversarial review obligations

The read-only review MUST establish its authoritative baseline independently, including Issue #16, the repository-relative owner-directed policy contract, applicable runtime/handoff/scoped-policy decisions, expiry/supersession, and `EFFICIENCY_MEASURES.md` Version 2. The reviewer MUST retrieve canonical private policy through the authorized Google Drive connector or an independently approved successor route and MUST NOT reconstruct unavailable authority from assertions.

The reviewer MUST cover all of the following axes and actions under the requirement applicability labels from R-012:

1. **Policy:** all required roles; equivalent role options and authorized mixed-provider selection; correct primary/fallback semantics; adoption/digest/scope/time boundaries; preservation of OpenAI defaults and V1 historical bindings; protection against retroactive reinterpretation and V2-to-V1 downgrade acceptance.
2. **Validators and callers:** inspect `validate_policy_document`, `match_packet_roles`, `selected_roles`, `adopted_policy_from_source`, `require_resolved_policy!`, direct `policy["roles"]` consumers, and handoff/preflight/readiness/dispatch/P0/handback/reviewer-capability paths, including callers outside the diff. Attack malformed, duplicate, unknown, missing, mismatched, and stale inputs.
3. **Privacy/provenance:** strict scoped token grammar; aliases and provider/model mismatch; mode/effort confusion; caller-authored native proof; wrong thread/turn/context; replay/cross-episode reuse; prohibited propagation before privacy rejection.
4. **Independence:** self-review, inherited implementation context, omitted axes, wrong candidate/rubric/capability, historical PASS reuse, and Handoff Review PASS misused as dispatch/scientific acceptance.
5. **Tests:** focused RolePolicy/PrivacyBoundary coverage, applicable CYAXControl regression coverage, and required authorized native checks. Synthetic fixtures do not establish native capability. The reviewer MUST record actual commands, versions, exits/results, and missing tests with minimal reproduction specifications, and MUST NOT make implementation changes.

The report MUST provide subject identity, requirement coverage, P0-P3 findings with precise code or evidence locations, mechanisms/consequences, smallest corrections, closure verification, actual tests/gaps, and bounded next work. The reviewer MUST use `PASS`, `PASS_WITH_NONBLOCKING_FINDINGS`, `REQUEST_CHANGES`, or `BLOCKED_INSUFFICIENT_EVIDENCE` for the **original-baseline verdict target in R-012**. The reviewer MUST report OC-001 compatibility gaps separately. A demonstrated mandatory defect MUST take `REQUEST_CHANGES` precedence; missing evidence MUST remain explicit. Without a demonstrated blocker, missing mandatory acceptance evidence yields `BLOCKED_INSUFFICIENT_EVIDENCE`.

The report MUST distinguish code acceptance, native test completeness, provider-native capability, private-policy adoption readiness, and dispatch/activation readiness. The formal charter determines required roles and native evidence; this programme does not dispatch that review or grant its verdict. The reviewer MUST preserve the original review priority/capability only where still applicable under its independently verified authority.

### R-014 — Serial pilot, then optional concurrency

The initial native pilot MUST run serially: one designated Manager and at most one active worker or reviewer, with native child delegation disabled or mediated. Test Claude-Manager/Codex-worker and Codex-Manager/Claude-worker directions in separate trials, then freeze each candidate and request host-authored read-only review. The host MUST confirm termination/reconciliation of the previous attempt and descendants before replacement.

Conformance includes option/effort identity, independent provider choice, packet/result validity, prohibited-operation rejection, false completion, partial output, interruption, recovery, and Manager loss. The pilot MUST use synthetic/replayable tasks, local artifacts, and bounded resources and MUST report support per tested configuration.

Concurrent work is a later optional milestone requiring separate authorization and evidence that shared Git metadata, hooks/configuration, Julia depots/precompile caches, ports, credentials, and services cannot violate task scope or corrupt results. Worktrees alone are insufficient. The verification route MUST preserve regular-host Julia/environment semantics and the repository CI gate under R-011; an operator-mediated route MAY be used where approved. Serial success does not claim concurrency or aggregate-integration success.

### R-015 — Revised Phase C evaluation plan

**R-015.1 — Primary comparisons.** The initial evaluation MUST use three arms and exactly two primary paired contrasts: D-single versus A-single, and D-coordinated versus D-single.

| Arm | Instructions | Verification | Topology |
| --- | --- | --- | --- |
| A-single | Current approved baseline | Current approved baseline | Single worker |
| D-single | Mechanism-first | Enhanced | Single worker |
| D-coordinated | Mechanism-first | Enhanced | Manager plus bounded worker |

Every arm MUST retain the same applicable mandatory authority, privacy, scientific, test-integrity, and independent-review gates. Enhanced verification is the experimental addition; baseline A is not exempt from mandatory safeguards.

B (mechanism-first only), C (verification only), provider rankings, extra role/model combinations, and concurrency MAY be secondary, budgeted experiments. They are not required factorial dimensions. Two-way provider conformance is an operational requirement, not a demand for every performance cross-product.

Matched cases MUST hold repository/environment, worker configuration, independent acceptance oracle, and declared total resource caps fixed and MUST record all role assignments and intervention differences. The topology contrast MUST include every additional Manager/reviewer cost. The shared D-single comparator MUST be identified in analysis. A-versus-D estimates the combined intervention; component attribution requires separate evidence. Experimental control creates no operational provider lock.

**R-015.2 — Exposure and holdouts.** The case register MUST record defect/fix provenance and first known public exposure, exact model/version, any reliable cutoff source, and exposure risk. The design SHOULD prefer post-cutoff cases where evidence supports that label, but MUST NOT infer absence from model weights merely from dates or withheld prompts. Unknown training exposure remains unknown.

The set MUST include historical defects and independently authored mutation-injected/synthetic cases with independently checked, realistic ground truth. Reports MUST separate public/high-risk, unknown-risk, and newly authored/lower-known-risk cases, preserving any uncertainty. No case is training-data-clean merely because the worker cannot see its reference fix.

The protocol MUST split development and holdouts by defect family before tuning and MUST protect hidden tests, reference repairs, later answer-bearing Git history/discussion, and held-out lessons from all worker tool channels. The oracle MUST accept valid alternative repairs rather than demand reference-patch matching. The set MUST include no-defect cases and plausible incorrect repairs.

**R-015.3 — Sample size and inference.** The protocol MUST freeze case/family counts, outcomes, run order, repeats, matching, analysis, stopping rules, and dispositions of failures/blocked/over-budget attempts before execution. Repeats and related mutations MUST NOT count as independent defects. Reports MUST show paired per-case results, denominators, effects, uncertainty, integrity failures, and missing evidence. Confirmatory claims MUST first establish that independent cases can resolve the minimum meaningful effect. Insufficient precision is exploratory. The intervention MUST NOT be tuned, and sampling MUST NOT be extended in response to observed results, merely to obtain a desired conclusion.

**R-015.4 — Total effort and governance cost.**

| Dimension | Required accounting |
| --- | --- |
| Human active person-time | Owner approvals/review, handoff/packet preparation, review dispatch, intervention, recovery, evidence reconciliation, and in-scope integration/adoption preparation. |
| Agent/infrastructure expenditure | All Manager, worker, reviewer, verifier, optimizer/probe, failed and retried calls; provider charges and execution resources. |
| Elapsed time | End-to-end completion and critical-path gate waiting, separating queueing/external delays. |
| Setup/support | One-time harness/governance setup and recurring maintenance, separate from per-task costs. |

Reports MUST include this cost vector unless an approved aggregation exists. Active minutes and waiting time MUST NOT be conflated; parallel durations MUST NOT be double-counted as elapsed time. Any money/time conversion or setup amortization MUST state valuation and task-volume assumptions. Lower tokens or worker latency alone cannot establish lower total effort.

**R-015.5 — Decision thresholds and exploratory scope.** Before Phase C specification approval, DEC-04 MUST fix numerical minimum benefit, primary quality endpoint/margin, human-effort and monetary/resource ceilings, elapsed-time bound, uncertainty rule, maximum rounds, and finite total budget/stop condition. These remain unresolved approval inputs; this draft does not invent numbers or claim confirmatory readiness.

Authority, privacy, scientific, and test-integrity requirements are hard gates. Exploratory results MAY support a separately reviewed bounded safety/verification improvement with its own mechanism, conformance, and regression evidence. Exploratory results alone MUST NOT justify a CLI-first default, general coordination superiority, or empirical equality of model options.

### R-016 — Lessons and research inputs

The first evaluation MUST use frozen predeclared instruction variants. Any recursive or online self-evolution experiment requires separately approved scope and controls. The programme MUST extract lessons from development cases, freeze the lesson bundle before transfer evaluation, and follow the existing candidate/validated/promotion lifecycle. No automatic normative promotion is permitted.

The programme MUST treat Harness² as methodological input only; its execution defaults are not adopted. The unnamed AI-agent engineering benchmark is explicitly excluded from the first slice and its claimed evidence. Adding it later requires its exact title, version, primary source, intended contribution, and scope review; do not substitute another benchmark.

### R-017 — Evidence-led adoption and historical continuity

Adoption proposals MUST be limited to supported surface-default changes, orchestration procedures, evidence helpers, instructions, and validated lessons. Proposals MUST document regressions, negative results, uncertainty, support cost, ownership, and rollback/revocation conditions. Normative adoption requires the applicable reviews, exact identity, authority, and effective interval.

Adoption MUST preserve historical episode interpretation and explicit expiry, supersession, and revocation handling for future actions, including in-flight episodes at a boundary.

**R-017.1 — Programme exit.** At the predeclared stopping/budget boundary, the programme MUST record adopt, narrow, revise-with-new-authorization, or do-not-adopt. If coordination fails to meet the approved quality, benefit, and cost criteria against D-single, the programme MUST close its adoption branch with a do-not-adopt recommendation. An inconclusive result justifies declining a default change without proving ineffectiveness. Further experiments require a new bounded question and authorization. Independently useful safety/verification improvements MAY continue as separately scoped proposals; the independently governed PR #74 audit retains its own disposition.

## Acceptance gates

R-001 requires these decisions and their evidence to remain distinct. Only G0 preparation is authorized/completed by the present deliverable. All later grants apply to their exact scope and candidate, not automatically to the whole programme.

| Gate | Acceptance / decision | Stop or boundary |
| --- | --- | --- |
| CYAX-0209 G0 — Definition prepared | Governing Issue, reviewable draft PR, canonical IDs, source/candidate identities, unresolved inputs, and sanitized preparation checks | No operational authority follows from preparation. |
| CYAX-0209 G1A — Independent policy audit | R-012 original-baseline verdict, separate OC-001 gaps, R-013 coverage/tests, five readiness judgments | Missing mandatory authority/evidence blocks the affected conclusion; retain feasible static findings. Charter revision belongs to Workstream A. |
| CYAX-0209 G1B — Architecture/spec approval | Exact revision, required SPEC/STANDARDS and applicable reviews, durable approval reference; resolve DEC-01–DEC-03 for the first launcher slice | Unresolved normative intent blocks that slice. Phase C is unapproved until DEC-04 is resolved; a narrower approval MUST state exclusions. |
| CYAX-0209 G2 — Implementation and native pilot grants | Record consequential implementation and native execution as separate grants: actor/surface/permission envelope, finite budget, environment, evidence destination, test plan, current policy dependencies | A narrow native qualification-test grant can precede a separate orchestration grant; R-005 qualification is prerequisite for the latter. Implementation authority grants neither. Authorized alternatives may be chosen without widening scope. |
| CYAX-0209 G3 — Serial operational conformance | Pilot-applicable R-002–R-011 and R-014: both directions, option provenance, host-authored review, rejection and recovery | Authority/permission escape, unreconciled effects, or missing required evidence stop the affected trial. No concurrency claim. |
| CYAX-0209 G3C — Optional later concurrency | Separate authorized scope, shared-resource isolation, aggregate candidate checks and applicable review | Serial conformance is prerequisite; concurrency is not required to conclude the initial evaluation. |
| CYAX-0209 G4 — Evaluation concluded | R-015/R-016 paired outcomes, exposure strata, full cost, uncertainty, negative results, recommendation against frozen thresholds | Leakage, invalid oracle, changed criteria, or uncontrolled differences invalidate the affected inference. Negative/inconclusive evidence can complete the bounded study. |
| CYAX-0209 G5 — Adoption/disposition | Explicit adopt/narrow/revise/do-not-adopt decision; integration, policy adoption, and activation each have separate current authorization and evidence | Missing grants or failed criteria prevent the affected action. R-017.1 permits closure without adoption. |

Each gate MUST report satisfied, failed, blocked, or not applicable, with evidence and reason. Candidate/requirements changes require a new identity and explicit evidence-applicability analysis.

## Verification requirements

Verification MUST be proportional to each slice and MUST report checks actually run. Preparation uses content/structure, reference, privacy, and exact-file scope checks; it does not claim Julia/native tests or a formal independent architecture verdict.

Minimum subsequent scenarios include:

| Requirement | Scenario | Expected evidence |
| --- | --- | --- |
| R-002 | Either declared equivalent selected while both are eligible/available | Both choices accepted without a provider priority or new Owner approval inside the same envelope. |
| R-002, R-014 | OpenAI Manager with Anthropic Implementer and independently eligible reviewer; reverse direction | Role-level authorization without a provider-wide profile lock. |
| R-002 | Different tasks choose different eligible primaries | No inherited global configuration constraint. |
| R-003–R-004 | Primary unavailable before start | That invocation blocked; permitted reselection recorded without claiming every alternative unavailable. |
| R-004, R-008 | Failure after start | Declared sequential transition, continuity, reconciled effects, and new attempt identity. |
| R-004 | Retry disguised as new primary, or concurrent fallback for same task | Rejection unless separately authorized replanning actually permits it. |
| R-003, R-007 | Receipt names an equivalent different from actual/selected execution | Discrepancy rejected or reconciled; no identity substitution. |
| R-002–R-005 | Manager invents an equivalent, native effort mapping, or permission | Rejection outside the governed option/configuration envelope. |
| R-004 | Sonnet appears only as fallback but is used as primary | Rejection unless applicable primary eligibility exists. |
| R-012–R-013 | Requirements change while PR #74 bytes stay fixed | New requirements identity; explicit historical/current-intent distinction and evidence reuse analysis. |
| R-013 | V1 history, malformed V2, expired/tampered scope, replayed native proof | Correct preserved historical binding or rejection, as applicable. |
| R-011, R-015 | Scientific defect and plausible wrong repair | Independent oracle, immutable tolerances, regression/test-integrity evidence. |
| R-008 | Crash, cancellation, Manager loss, uncertain completion | No unaccounted descendant or duplicate action; truthful blocked/failed/partial reporting. |
| R-005.1 | Manager tools attempt provider CLI/API/SDK/MCP/native delegation or credential extraction | Legitimate inference transport works; unauthorized launch/credential reuse is prevented across enabled channels. |
| R-005.3 | Worker/Manager attempts Git push or repository API/browser/MCP write | Remote publication denied; local artifact identity retained. |
| R-009.1 | Manager omits an axis, edits reviewer prompt, or candidate injects configuration | Host packet retains required inputs/axes; candidate cannot alter execution permissions or review authority. |
| R-009.1 | Candidate changes after freeze | Identity mismatch blocks review startup. |
| R-014 | Second active worker or surviving descendant in initial pilot | Launch denied until serial lifecycle reconciliation; later concurrency needs its own evidence. |
| R-002.1 | Role-specific quality-floor/integrity trigger fires | Scoped stop/quarantine and eligibility review; no automatic permanent catalogue rewrite. |
| R-015.2–R-015.5 | Public fix exposure, repeated mutations, approval delay, or small sample | Risk-stratified results, independent units, governance costs, and exploratory limits reported. |
| R-017.1 | No qualifying coordination benefit by budget boundary | Explicit close/do-not-adopt recommendation; no automatic extension. |

Required native checks need an authorized route/operator. A read-only review can inspect existing verifiable receipts without being granted launch authority. Candidate freeze/review evidence MUST identify code and governing requirements separately.

## Interfaces and compatibility

- Draft a versioned provider-neutral task/result/evidence envelope and thin adapters. The final wire schema, compatibility rules, subprocess implementation, and native configuration mapping belong to the approved slice's subordinate plan.
- Preserve existing V1 bindings and strict parsing. Do not silently reinterpret a malformed V2 record through V1 or widen accepted token grammar outside its scope.
- Keep ordinary-Chat V0, private policy authority, deterministic control-plane functions, and new execution transport distinct.
- Preparation changes no public Julia API, scientific behavior, persisted scientific schema, supported package environment, or package version. Package-infrastructure SemVer impact: none for preparation. Implementation slices MUST declare their actual impact separately.
- Agent tooling MUST NOT become a runtime dependency of `using CYAxiverse`.

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

Related work is not automatically a blocking dependency or adopted implementation. Add native dependency relationships only for a concrete blocked outcome. Architecture preparation does not require every policy audit conclusion; execution MUST satisfy the dependencies of invoked behavior.

### Primary research/interface references

- [Codex non-interactive execution](https://developers.openai.com/codex/non-interactive-mode).
- [Codex approvals/security](https://developers.openai.com/codex/agent-approvals-security).
- [Claude Code programmatic execution](https://code.claude.com/docs/en/headless).
- [Claude Code shell sandbox scope](https://code.claude.com/docs/en/sandboxing).
- [GPT-6.1 Sol native API effort](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [GPT-6 Luna native API effort](https://developers.openai.com/api/docs/models/gpt-6-luna), [Codex models](https://developers.openai.com/codex/models), and [Claude Code model/effort configuration](https://code.claude.com/docs/en/model-config).
- [Harness² paper v1](https://arxiv.org/html/2610.14224v1) and [reference implementation](https://github.com/google-research/harness2).

Pin actual execution versions for evidence; those versions are observations/experimental controls, not permanent architecture selections.

## Open approval inputs

OC-001's equivalent-option principle and preparation authorization are settled. The following are concrete remaining decisions, not requests to repeat those approvals.

| ID | Proposed disposition / missing value | Required boundary |
| --- | --- | --- |
| DEC-01 | Separate S2 launcher at `vmmhep/CYAxiverse-agent-exchange/tools/CYAXLauncher/`; approve owning specification, maintainer, trust model, and feasibility evidence plan. Path is proposed, not approved. | G1B launcher slice; qualification evidence before G2 native orchestration. |
| DEC-02 | Approve or amend R-004.1's ordered ladder; complete per-role/task capability floors, primary/fallback membership, pinned-client native effort/mode subsets, and trusted qualification sources. | G1B first execution contract; availability/permission receipts before each native attempt. |
| DEC-03 | Freeze serial pilot scope, permitted tool channels, credentials/transport isolation, task-specific surface override, regular-host verification route, resource limits, stop/quarantine and recovery rules, reviewer packet template/rubric, evidence destinations. | Relevant G1B contract and separate G2 grants. |
| DEC-04 | Numerical minimum benefit and endpoint, quality margin, human/money/resource/time ceilings, independent case/family register, analysis precision, maximum rounds, total budget, finite stop rule. | Phase C specification approval and native evaluation authorization. Unresolved: no confirmatory/adoption readiness claim. |
| DEC-05 | Unnamed benchmark excluded from first slice; exact source/version/contribution required only if a later amendment includes it. | Any later claim of benchmark alignment. |
| DEC-06 | Workstream A authority decides whether full original-candidate audit remains useful before expensive native checks; existing complete charter remains until explicitly revised. | Independent G1A commissioning; no programme-imposed narrowing. |

Confirm provisional S3 classification during G1B; any reclassification preserves S2 trust-boundary review and all grants. Architecture drafting and scoped review can proceed while later evaluation inputs remain unresolved.

## Completion criterion

Preparation is complete when CYAX-0209 G0 is evidenced by this governing draft and its linked preparation PR. The specification remains draft and the programme remains open.

The bounded coordination programme concludes with its scoped evidence, unresolved limitations, and explicit adopt/narrow/revise/do-not-adopt decision under R-017.1. A negative or inconclusive result can end it without changing defaults. Closure MUST identify any independently governed work, including PR #74 review, that continues under its own authority. Adoption-dependent actions use the separate G5 grants. A merged draft or agent agreement is not programme completion.
