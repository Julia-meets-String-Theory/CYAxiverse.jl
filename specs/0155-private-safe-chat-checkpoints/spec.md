# CYAX-0155 — Private-safe conversation checkpoints

Status: Approved for S1 implementation
Issue: #155
Spec class: S1
Version impact: none

## Objective

Allow useful outcomes from exploratory AI/chat investigations to become visible in the existing CYAxiverse GitHub Project without publishing the private conversation itself or machine-local context.

## Boundary

GitHub is a durable coordination/publication surface, not a private-chat archive. Content copied, summarized, or generated from private chats, local agent sessions, local files, connected applications, or private attachments is non-public by default and must be deliberately sanitized before any durable GitHub write.

### R-001 — Publication sanitization

Before creating or updating a commit, Issue, PR, Project item, spec, validation artifact, or comment from private/local context, the writer must remove private or machine-specific metadata unless the owner has explicitly approved the exact datum for publication.

This includes private conversation/share URLs or transcript dumps; absolute local filesystem/home-directory paths; local usernames, hostnames, machine/device identifiers; local Codex/agent/workspace/session paths; private attachment or connector-local locations; and secrets, credentials, or tokens.

Repository files must be referenced using repository-relative paths. If sanitization is uncertain, the write must stop for owner direction.

### R-002 — Conversation checkpoints are summaries, not transcripts

A conversation-derived Project item is created only when the discussion produces a durable project outcome worth tracking: a research question, decision, blocker, proposed specification, implementation follow-up, or evidence-bearing investigation.

The durable checkpoint may contain only deliberately public-safe project information such as:

- title / research question;
- workstream and normal Project status;
- concise current conclusion or decision;
- next action or owner decision needed;
- related Issue/spec/PR/repository-relative references;
- optionally, an opaque non-resolving private reference.

It must not contain a resolving private-chat URL, raw transcript, or private-machine locator.

### R-003 — Preserve the existing Kanban semantics

Conversation-derived work uses the existing Project status workflow:

`Backlog → Specifying → Spec Review → Ready → Implementing → Verification → Done`

Do not add a second status lifecycle, a permanent conversation-only status column, or a second project ledger. For S1–S3 work, the governing Issue remains the primary Project item. Linked PRs remain implementation/integration/evidence surfaces rather than duplicate WIP cards.

### R-004 — Research & Chats is a view, not an authority layer

A `Research & Chats` saved Project view may surface durable conversation/checkpoint Issues using the normal Project Status and fields. It must not contain private-chat links or become a separate source of truth.

The initial useful card fields are:

- title;
- Status;
- Workstream;
- Priority;
- Next review where applicable;
- concise public-safe conclusion/decision;
- next action;
- related governing Issue/spec/PR.

A lightweight marker/filter for checkpoint-originated Issues may be added when Project configuration supports it, provided that marker does not encode private information.

### R-005 — Fail closed

If an agent cannot determine whether a datum is safe to publish, it must omit the datum and ask the owner before publishing it. Convenience or traceability does not override the privacy boundary.

## Non-scope

- Bulk conversion of historical conversations into Issues.
- Publishing ChatGPT/shared-conversation URLs.
- Remediating previously published local paths or other historical leakage; that requires a separate privacy audit/remediation decision.
- Changing scientific behavior, numerical conventions, APIs, persisted schemas, dependencies, or package version.
- Altering the approved SDD status lifecycle.

## Implementation plan

This compact S1 change is implemented directly from the requirements above:

| Requirement | Implementation | Verification |
| --- | --- | --- |
| R-001, R-005 | Add concise normative rule to `AGENTS.md`; add operational check to `cyaxiverse-sdd` | Diff review for explicit prohibited categories, repo-relative-path rule, and fail-closed behavior |
| R-002 | Add checkpoint promotion/content rules to `cyaxiverse-sdd` and human guide | Confirm guide contains only sanitized metadata template and no private-link mechanism |
| R-003, R-004 | Add human guide describing the `Research & Chats` view/layer while preserving current Project Status | Compare against CYAX-0151 G2 Project contract; no status change |

## Acceptance

- R-001–R-005 are represented in the repository control plane.
- No private conversation URL, absolute local path, local username/hostname, local workspace path, private attachment locator, credential, or token is introduced by this implementation.
- Existing Project Status semantics are unchanged.
- Issue #155 provides the first durable Project-facing item for this change; a saved `Research & Chats` view can be configured without introducing a second ledger.
