# Lessons & Corrections

This is a small repository-wide registry of reusable failure patterns and
preventive checks. It is knowledge memory, not a diary, transcript archive,
personal scorecard, workflow ledger, or second authority system. Entries must
preserve the generalizable correction rather than the private event that led
to it.

## Authority and use

CYAX-0151's authority order remains unchanged:

```text
AGENTS.md + relevant CYAxiverse skill
    ↓
approved feature spec.md
    ↓
plan.md
    ↓
tasks.md
    ↓
GitHub Issue / Project
    ↓
implementation
```

Validated lessons are cross-cutting advisory inputs consulted after the
governing sources are known and before planning or implementation. They are not
an additional authority tier. Lessons never override `AGENTS.md`, a relevant
normative skill, an approved governing specification, or another artifact in
the approved hierarchy. If a lesson conflicts with an authoritative artifact,
do not silently follow the lesson: reconcile the conflict against the higher
sources and correct or supersede the lesson or update the appropriate artifact
through its normal review. Candidate lessons are proposals, not authority, and
are not a source of merge, Issue, Project, review, or scientific-acceptance
state. GitHub Issues, pull requests, and Projects retain their existing roles
for live work and status.

Read only applicable `validated` lessons for the task. Do not require every
agent to read the complete historical registry.

## Lifecycle and promotion

The knowledge lifecycle is:

```text
candidate → validated → superseded
```

- **candidate** records a sanitized, potentially reusable observation. It is
  advisory and must not be used as a substitute for a governing contract.
- **validated** means a reviewer found the pattern generalizable, accurately
  represented, privacy-safe, and supported by durable evidence. Validation does
  not promote the lesson into a normative policy.
- **superseded** retains historical traceability after a reviewed correction,
  narrower lesson, changed context, or normative promotion replaces the
  lesson. Set `Superseded by` to the replacement lesson or normative source.

The promotion ladder is local observation or correction → candidate → reviewed
validated lesson → repeated, stable, or high-impact rule → separately
reviewed promotion to `AGENTS.md`, the SDD skill, or another normative source.
Promotion requires the normal review and approval for the destination artifact;
it is never automatic. When a normative source adopts the rule, mark the lesson
`superseded`, record that repository-relative source and its reviewed revision
or approval reference in `Superseded by` and `Promotion status`, and stop
applying the lesson independently. If only part is promoted, replace the
original with a narrower candidate or validated lesson for any remaining
guidance. The historical entry remains traceable but inactive.

Owner redirections, private corrections, and local observations are first
sanitized and abstracted before becoming a candidate. Do not copy raw chat
text, private conversation or share URLs, transcript excerpts, local session
identifiers, machine-specific context, or resolving private references. If a
correction does not yield a generalizable prevention rule, do not add it.

## Compact lesson schema

Every entry uses these fields:

```text
ID: L-xxxx
Status: candidate | validated | superseded
Date: YYYY-MM-DD
Type: owner-redirection | peer-review | scientific-error | semantic-drift |
      scope-drift | state/provenance | tooling | testing | privacy | workflow
Scope / tags: ...
Observed failure: ...
Correction: ...
Root cause: ... (when known)
Preventive rule / check: ...
Applicability / exceptions: ...
Evidence / durable reference: ...
Superseded by: L-xxxx | <repository-relative normative source + section> | N/A
Promotion status: not promoted | promoted to <source> at <revision/approval ref>
```

## Seed lessons

The six initial entries were validated through the independent review and
repository-owner approval recorded in PR #160 comment `5636316397`. They are
deliberately limited to patterns supported by durable repository or public
GitHub history; they do not assert private conversation details.

## L-0001 — Candidate or repaired evidence is not scientific acceptance

ID: L-0001
Status: superseded
Date: 2026-09-11
Type: scientific-error
Scope / tags: scientific acceptance, candidate evidence, gate review, Issue #148

Observed failure: Candidate or repaired evidence can be treated as though it
were an accepted scientific result before the required independent review and
owner approval have occurred.

Correction: Keep candidate, repaired, rejected, independently reviewed, and
accepted states distinct. Claim a scientific gate only from its durable review
and approval record, not from the existence of a repaired artifact or a
matching intermediate result.

Root cause: Evidence production was conflated with scientific gate acceptance.

Preventive rule / check: Before writing PASS, accepted, or population-level
language, identify the governing scientific gate and its required review,
approval, and evidence record. If that record is absent, use candidate or
provisional language and stop at the applicable gate.

Applicability / exceptions: Applies to scientific claims and gate records.
Process lessons may discuss candidate evidence, but they do not establish
scientific acceptance.

Evidence / durable reference: Issue #148 / PR #149 gate history, including the
durable candidate/rejection and later fresh-independent-review acceptance
records in repository history. Stronger current coverage is in `AGENTS.md`
section 4 and the SDD Approved/Accepted lifecycle.

Superseded by: `AGENTS.md` section 4; `specs/0151-sdd-v1/spec.md` section 8;
canonical Scientific Reviewer verdict/authority semantics in
`protocol/review-rubrics/scientific-v1.md` in the CYAxiverse agent-exchange
control plane.
Promotion status: not promoted from this lesson; stronger normative
scientific-claim, review, and acceptance rules were adopted independently.

## L-0002 — Namespace nested gates explicitly

ID: L-0002
Status: validated
Date: 2026-09-11
Type: owner-redirection
Scope / tags: gate namespacing, semantic drift, SDD, scientific review, Issue #151

Observed failure: Nested workflows use overlapping labels such as G2 or G3,
so a bare gate label can be read as a CYAX-0151 process gate or an Issue #148
scientific gate.

Correction: Always qualify a gate with its workflow or Issue, for example
`CYAX-0151 G3`, `#148 scientific G2`, or `#148 scientific G3`.

Root cause: Cross-workflow references omitted the namespace from a label that
was locally meaningful in more than one workflow.

Preventive rule / check: In specs, Issues, PRs, reviews, and handoffs, write
the process/Issue namespace with every gate status. Do not use an ambiguous
bare `G2` or `G3` when more than one gate vocabulary is in scope.

Applicability / exceptions: A single-workflow local note may abbreviate after
defining its namespace, but cross-workflow or durable status references retain
the explicit namespace.

Evidence / durable reference: Sanitized owner-redirection-derived pattern
checked against the durable SDD and pilot context in Issue #151, PR #154, and
Issue #148; no private correction or resolving private reference is retained.

Superseded by: N/A
Promotion status: not promoted

## L-0003 — Brownfield migration preserves supersession chronology

ID: L-0003
Status: superseded
Date: 2026-09-11
Type: workflow
Scope / tags: brownfield migration, supersession, SDD, historical traceability

Observed failure: A brownfield migration can treat the original Issue body as
the final contract and silently rewrite historical work to match a later
interpretation.

Correction: Reconstruct a concise chronological decision and supersession
record from durable Issue wording, owner decisions, accepted gate records,
PRs, and validation artifacts. Use the latest durably owner-approved,
non-superseded normative intent as the migrated contract while preserving
earlier wording as history.

Root cause: Historical evidence and current normative intent were collapsed
into one retrospective narrative.

Preventive rule / check: Before migration, identify which durable decision
superseded each earlier statement; do not project newly introduced requirement
IDs backward onto historical execution. Stop for owner clarification when
precedence cannot be established safely.

Applicability / exceptions: Applies to brownfield migration and backfill.
Greenfield work starts from its current approved specification and does not
need a fabricated historical chronology.

Evidence / durable reference: `specs/0151-sdd-v1/clarification-a.md`, section 3,
and its durable SDD migration history in Issue #151 / PR #154.

Superseded by: `specs/0151-sdd-v1/clarification-a.md` section 3 and
`.agents/skills/cyaxiverse-sdd/SKILL.md` step 13.
Promotion status: not promoted from this lesson; the brownfield supersession
rule was adopted independently into the normative SDD workflow.

## L-0004 — `tasks.md` is not authoritative live status

ID: L-0004
Status: superseded
Date: 2026-09-11
Type: state/provenance
Scope / tags: tasks, GitHub status, Project, merge state, SDD

Observed failure: An execution checklist in `tasks.md` is treated as the live
source for PR draft/ready state, merge state, Issue closure, Kanban state, or
review ownership.

Correction: Use `tasks.md` for subordinate execution decomposition and
evidence-readiness. Read current merge, Issue, Project, and review state from
GitHub's Issue, PR, and Project surfaces.

Root cause: The purpose of an execution artifact was confused with the live
workflow control plane.

Preventive rule / check: Before reporting current status, check the governing
Issue, linked PR, Project state, and current review owner. Do not mark a task
complete merely to imply that a PR merged or an Issue closed.

Applicability / exceptions: A task file may record planned or completed work
and evidence, but it must not become a second live status ledger.

Evidence / durable reference: `specs/0151-sdd-v1/clarification-a.md`, section 5,
and the corresponding Issue #151 clarification.

Superseded by: `specs/0151-sdd-v1/clarification-a.md` section 5 and
`specs/0151-sdd-v1/spec.md` section 6, R-003.
Promotion status: not promoted from this lesson; the same rule was adopted
independently into the normative SDD contract.

## L-0005 — Do not falsify workflow state for advisory WIP limits

ID: L-0005
Status: superseded
Date: 2026-09-11
Type: workflow
Scope / tags: WIP, GitHub Project, state integrity, SDD G2

Observed failure: Advisory WIP counters can create pressure to move or relabel
work so the displayed limits look satisfied even though the underlying work
has not changed state.

Correction: WIP counts and workflow states must reflect the actual governing
Issue state. If an advisory limit is saturated, report the saturation or seek
an explicit workflow decision; never fabricate a state transition to make a
dashboard green.

Root cause: A monitoring aid was treated as a hard objective rather than an
advisory view of reality.

Preventive rule / check: Reconcile the Project state with active Issues before
reporting WIP. Keep the experimental/advisory qualification visible and make
state changes only for real workflow transitions.

Applicability / exceptions: Applies to advisory Project WIP limits and status
reporting. It does not change the approved status lifecycle or scientific gate
criteria.

Evidence / durable reference: Issue #151 G2 PASS comment `5625939343` and
`specs/0151-sdd-v1/clarification-a.md`, section 6. Deterministic
`LifecycleProjection` / `ProjectStatus` machinery in CYAXControl provides
additional enforcement evidence but is not the normative replacement.

Superseded by: `specs/0151-sdd-v1/clarification-a.md` section 6 and
`specs/0151-sdd-v1/spec.md` sections 15.1 and 15.3.
Promotion status: not promoted from this lesson; true-state/advisory-WIP
semantics were adopted independently into the normative SDD contract.

## L-0006 — Machine-local paths are not scientific provenance

ID: L-0006
Status: superseded
Date: 2026-09-11
Type: privacy
Scope / tags: scientific provenance, privacy, reproducibility, public paths

Observed failure: A machine-local path or execution-context locator is used
as public scientific provenance, even though it is private, non-portable, and
does not identify the source or artifact for independent replay.

Correction: Identify public scientific material with durable source/revision,
artifact or content hash, repository-relative paths, selection or witness
identity, and sanitized wording. Omit private or machine-specific context;
if safety is uncertain, stop for owner direction.

Root cause: A local execution locator was confused with a durable scientific
identity and publication-safe provenance record.

Preventive rule / check: Before a scientific or workflow write, scan paths and
provenance for absolute filesystem locations, usernames, hostnames, device or
session identifiers, and private attachment/context locators. Replace them
with durable public references or omit them under the privacy boundary.

Applicability / exceptions: Repository-relative paths and public revision/hash
references are allowed. Private local context may remain local; it is not
made public by being useful for debugging. This lesson contains no
Fuzzy/Table-1-specific claim.

Evidence / durable reference: `AGENTS.md` sections 1 and 4;
`specs/0155-private-safe-chat-checkpoints/spec.md`, R-001 and R-005; Issue #157
/ PR #158 public-path remediation history. CYAXControl privacy preflight
provides additional deterministic enforcement on generated exchange artifacts.

Superseded by: `AGENTS.md` section 1 privacy/publication boundary and section 4
Scientific claim boundary.
Promotion status: not promoted from this lesson; stronger repository-wide
privacy and provenance rules were adopted independently.

## L-0007 — Handoff draft is not launch authorization

ID: L-0007
Status: superseded
Date: 2026-09-11
Type: owner-redirection
Scope / tags: delegation launch, authorization semantics, state/provenance

Observed failure: Discussion or refinement of a proposed delegation packet was
treated as launch authorization, and work/state advanced before explicit owner
authorization.

Correction: Treat discussion and refinement of a proposed packet as advisory, not
authorization. Present the final packet and obtain explicit owner authorization
before launch. After approval, ordinary already-authorized worker
diagnose/edit/check/correct/recheck work remains covered by that authorization.
Only a material post-approval change to authorized scope or terms requires
renewed approval.

Root cause: Plan updates were conflated with owner-governed launch authorization.

Preventive rule / check: Require a distinct pre-launch authorization checkpoint.
Treat packet edits as planning detail unless they materially change authorized
scope or terms; require renewed approval before applying such a change.

Applicability / exceptions: Applies to owner-redirection and delegated workflows
with explicit owner-governed execution. It does not replace scientific or merge
gates already governed by higher sources.

Evidence / durable reference: Historical candidate retained for traceability.
The current surface-allocation policy now gives the stronger owner-manual
dispatch, pre-dispatch review, and Manager downstream-dispatch contract.

Superseded by: `.agents/POLICIES/SURFACE_ALLOCATION.md` sections 6.1, 8, 9,
and 15.
Promotion status: not promoted from this lesson; owner-dispatch semantics were
adopted independently into the normative surface-allocation policy.

## L-0008 — Bound delegated topology by role, not aggregate agent count

ID: L-0008
Status: candidate
Date: 2026-09-11
Type: workflow
Scope / tags: delegation topology, role maxima, implementation continuity,
independent review

Observed failure: A delegated task can leave downstream role cardinality implicit
or rely on one aggregate concurrency/agent-count limit. That can accidentally
permit multiple implementers, allow undeclared roles merely because aggregate
capacity remains, or conflict with the need for a fresh reviewer on a successor
candidate.

Correction: Before delegated execution, declare every permitted downstream role
and its maximum simultaneous active cardinality. Undeclared roles have
cardinality zero. Manager and Control Desk contexts are excluded from downstream
role maxima. Preserve one implementation worker across ordinary
diagnose/edit/test/correct/retest and review-driven repair unless governed
replacement is required. Each required independent review axis may activate one
fresh reviewer per review round or successor candidate; that fresh activation
does not increase the role's simultaneous cardinality.

Root cause: Delegated topology was bounded by aggregate counts rather than by
role-specific admissible cardinality and role lifecycle.

Preventive rule / check: Require structured role maxima equivalent to
`implementer: 1`, `spec_reviewer: 1`, `standards_reviewer: 1`, and
`scientific_reviewer: 1`; undeclared downstream roles default to zero. Do not
use a global maximum-concurrency value or a lifetime activation cap as a
substitute. Verify implementation-worker continuity through ordinary repair
loops, and verify that each required successor review uses a fresh reviewer
without concurrent duplication of that review role.

Applicability / exceptions: Applies to delegated multi-agent work. The Manager
and Control Desk are not counted against downstream role maxima. A maximum is a
permission ceiling, not a requirement to activate the role; non-required review
axes remain inactive. A governed replacement may replace the implementation
worker when continuity is impossible or explicitly disallowed.

Evidence / durable reference: `.agents/POLICIES/SURFACE_ALLOCATION.md` and
`.agents/skills/cyaxiverse-agent-orchestration/SKILL.md` provide adjacent
role/topology, independence, delegation, and worker-continuity mechanisms. The
role-specific-maxima residue remains a candidate pending fresh exact review.

Superseded by: N/A
Promotion status: not promoted

## L-0009 — Handoff state follows observed execution outcome

ID: L-0009
Status: superseded
Date: 2026-09-11
Type: state/provenance
Scope / tags: handoff state, status integrity, orchestration

Observed failure: Handoff bookkeeping conflated the immediate operation result
with the later worker terminal return, so launch state could be inferred from the
wrong phase.

Correction: Use the immediate handoff operation result as the phase-1 launch
decision. Only success creates or attaches new execution task/worker state.
Decline, cancellation, or failure creates no new task/worker state, leaves prior
state unchanged, and is not silently retried. After successful launch, later worker
terminal states (`DONE`, `BLOCKED`, `FAILED`) are a separate phase-2 downstream
contract.

Root cause: State bookkeeping mixed handoff operation status with downstream return
state.

Preventive rule / check: Record phase 1 from the operation result and phase 2 from
the later worker terminal return. Do not make phase-2 evidence a prerequisite for
the phase-1 success transition; apply phase-2 checks only after successful launch.

Applicability / exceptions: Applies to workflows using explicit worker return
contracts and observed execution results. Internal planning notes may be prepared
before execution and do not by themselves move execution state.

Evidence / durable reference: The CYAXControl Issue #16 state machine now binds
review currentness, readiness, explicit owner dispatch receipt, handback
validation, and Control Desk reconciliation to observed exact evidence rather
than inferred intent. `.agents/POLICIES/SURFACE_ALLOCATION.md` sections 8 and 9
provide the corresponding public lifecycle/dispatch context.

Superseded by: CYAXControl deterministic state-transition contract in
`vmmhep/CYAxiverse-agent-exchange/tools/CYAXControl/src/ControlPlane.jl` and
its documented review/readiness/dispatch/handback flow, together with
`.agents/POLICIES/SURFACE_ALLOCATION.md` sections 8 and 9.
Promotion status: not promoted from this lesson; stronger deterministic
state-transition semantics were adopted independently.

