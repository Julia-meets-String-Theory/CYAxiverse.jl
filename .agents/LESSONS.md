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
Status: validated
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
records in repository history.

Superseded by: N/A
Promotion status: not promoted

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
Status: validated
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

Superseded by: N/A
Promotion status: not promoted

## L-0004 — `tasks.md` is not authoritative live status

ID: L-0004
Status: validated
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

Superseded by: N/A
Promotion status: not promoted

## L-0005 — Do not falsify workflow state for advisory WIP limits

ID: L-0005
Status: validated
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
`specs/0151-sdd-v1/clarification-a.md`, section 6.

Superseded by: N/A
Promotion status: not promoted

## L-0006 — Machine-local paths are not scientific provenance

ID: L-0006
Status: validated
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

Evidence / durable reference: `AGENTS.md` privacy and scientific-claim
boundaries; `specs/0155-private-safe-chat-checkpoints/spec.md`, R-001 and R-005;
Issue #157 / PR #158 public-path remediation history.

Superseded by: N/A
Promotion status: not promoted

## L-0007 — Handoff draft is not launch authorization

ID: L-0007
Status: candidate
Date: 2026-09-11
Type: owner-redirection
Scope / tags: delegation launch, authorization semantics, state/provenance

Observed failure: A delegated packet that only refined the execution plan was treated as permission to execute; work and state were advanced before explicit owner authorization.

Correction: Treat packet refinement as advisory only until a final, explicit owner authorization is given. Before each material post-authorization change, request renewed authorization and re-handoff terms.

Root cause: Ownership and authorization checkpoints were conflated with planning updates.

Preventive rule / check: Require a distinct authorization checkpoint before launch, and require that execution state references that checkpoint.

Applicability / exceptions: Applies to owner-redirection and delegated workflows with explicit owner-governed execution. Does not replace existing scientific or merge gates already governed by higher authority.

Evidence / durable reference: `AGENTS.md` delegated-workflow guidance in section 6; `.agents/skills/cyaxiverse-agent-orchestration/SKILL.md` sections 2 and 3 (delegation packet and worker return states); and issue/PR provenance context (`Issue #159` / `PR #160`) for owner-redirection sequencing.

Superseded by: N/A
Promotion status: not promoted

## L-0008 — Delegation contract must name capability and budget

ID: L-0008
Status: candidate
Date: 2026-09-11
Type: workflow
Scope / tags: delegation packet, capacity planning, escalation policy

Observed failure: A delegated task was launched with implicit model, reasoning, and cost/effort expectations, which reduced convergence quality and increased review churn.

Correction: State role, capability target, fallback policy, effort budget/lease, escalation conditions, and worker-reuse expectations in the packet before launch.

Root cause: The delegation contract lacked explicit operational constraints needed for bounded execution.

Preventive rule / check: Reject a delegated packet that omits any required contract field; require updates before each reuse of the same worker in a new bounded phase.

Applicability / exceptions: Applies to multi-agent or handoff-based implementation work; local single-agent edits with no downstream worker handoff may use a lighter form.

Evidence / durable reference: `AGENTS.md` delegated-task lifecycle in section 6; `.agents/skills/cyaxiverse-agent-orchestration/SKILL.md` sections 2 and 5 (delegation packet fields, task class and worker reuse).

Superseded by: N/A
Promotion status: not promoted

## L-0009 — Handoff state follows observed execution outcome

ID: L-0009
Status: candidate
Date: 2026-09-11
Type: state/provenance
Scope / tags: handoff state, status integrity, orchestration

Observed failure: A handoff outcome was recorded as if downstream work had started, despite delegation being declined, failed, or not yet returned.

Correction: Create or change execution state only after a successful handoff return state or accepted worker output; failed or declined handoff leaves prior state unchanged until manager re-issues direction.

Root cause: State bookkeeping used intent rather than observed execution result.

Preventive rule / check: Map worker returns to explicit states (`DONE`, `BLOCKED`, `FAILED`) and require a durable return artifact or result before recording downstream work-state updates.

Applicability / exceptions: Applies to workflows with explicit worker return contracts and observed execution results. Does not stop internal planning notes from being prepared before execution.

Evidence / durable reference: `.agents/skills/cyaxiverse-agent-orchestration/SKILL.md` sections 3 and 4 (worker return states, managed state progression); `AGENTS.md` section 6 (delegated-task lifecycle and manager handoff ownership).

Superseded by: N/A
Promotion status: not promoted
