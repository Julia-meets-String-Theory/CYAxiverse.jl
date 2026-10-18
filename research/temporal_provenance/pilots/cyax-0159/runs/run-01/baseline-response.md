# Run 01 — unstructured baseline response

## 1. Authoritative state and exact boundary

As of the manifest anchor, the authoritative repository state is:

- Branch/ref boundary: `vmm@856012f015a866bf7ff352bc50e8d10c250855e6`
- Commit: merge commit `856012f015a866bf7ff352bc50e8d10c250855e6`
- Subject: `Merge pull request #160 ... Add Lessons and Corrections`
- Parents:
  - base: `f0013552cd69a93221464e9c8ccfd56339f39052`
  - feature tip: `bd3e866537b0115a4cb76cbae31b3768da3de724`
- Merge time: 2026-09-11 12:37:54 −04:00

The merge contains exactly four scoped file changes:

- `.agents/LESSONS.md`
- `.agents/skills/cyaxiverse-sdd/SKILL.md`
- `.github/pull_request_template.md`
- `specs/0159-lessons-corrections/spec.md`

The current CYAX-0159 specification is approved, class S1, with approval
reference `PR #160 comment 5636316397`. Its six seed lessons, L-0001 through
L-0006, are all `validated`, dated 2026-09-11, and marked
`Promotion status: not promoted`.

## 2. Why this state is authoritative

The approved specification preserves the CYAX-0151 authority order:

`AGENTS.md + relevant skill → approved spec.md → plan.md → tasks.md → GitHub Issue/Project → implementation`

The lessons registry explicitly says validated lessons are advisory inputs, not
an additional authority tier. Candidate lessons cannot override governing
artifacts or establish merge, Issue, Project, review, or
scientific-acceptance state.

Applicable scopes are:

- CYAX-0159: bounded documentation/control-plane S1 change.
- CYAX-0151 clarification A: SDD authority, approval provenance, brownfield
  supersession, task/live-status separation, and Project semantics.
- CYAX-0155: privacy boundary for material derived from private/local context.
- Canonical SDD skill: validated lessons are read only when applicable;
  candidates remain advisory; convergence and privacy hooks apply.
- GitHub Issue/PR/Project: live coordination and status surfaces, not the
  specification authority.

No `plan.md` or `tasks.md` is required for this bounded S1 slice. The change
claims no scientific, numerical, API, dependency, persisted-schema, or version
impact.

The `.codex/skills/cyaxiverse-sdd` entry is an existing symlink to
`../../.agents/skills/cyaxiverse-sdd`; `.agents/skills/` remains canonical.

## 3. Superseded, rejected, disputed, and candidate state

Historical revisions:

- `2b9b056`: introduced six seed entries as candidates and initially placed
  validated lessons in the authority chain.
- `f965dc6`: corrected authority/promotion semantics but retained draft status
  and candidate entries.
- `bd3e866`: changed the specification to approved, recorded PR #160 approval
  provenance, and validated L-0001–L-0006.
- `856012f`: merged the approved result through PR #160.

Therefore, the earlier draft/candidate states are superseded by the
approved/validated state at the anchor.

At the anchor, all L-0001–L-0006 entries have `Superseded by: N/A`; none has
been normatively promoted.

A later descendant branch, apparently corresponding to PR #161, adds
L-0007–L-0009 as `candidate` entries. Its commits begin after the anchor
(`4f30149` → `71d43a0` → `071d345` → `e1288cf`) and are not included in the
anchor merge. Those entries explicitly state that they lack cited public
incident proof. They must not be treated as validated or authoritative in the
baseline reconstruction.

## 4. Material evidence

- `specs/0159-lessons-corrections/spec.md:1-16`: approved metadata and approval
  reference.
- `spec.md:28-55`: authority order and advisory lesson scope.
- `spec.md:57-81`: bounded four-file scope and requirements.
- `spec.md:83-104`: lifecycle, promotion, and privacy rules.
- `spec.md:106-125`: six-seed provenance.
- `spec.md:152-168`: verification and completion criteria.
- `.agents/LESSONS.md:9-40`: authority and live-status boundaries.
- `.agents/LESSONS.md:42-74`: candidate → validated → superseded lifecycle and
  privacy sanitization.
- `.agents/LESSONS.md:104-308`: L-0001–L-0006 content and evidence.
- `.agents/skills/cyaxiverse-sdd/SKILL.md:8-25`: SDD hooks, authority, candidate
  handling, convergence, and privacy.
- `.github/pull_request_template.md:31-59`: verification,
  Lessons/corrections, convergence, and remaining-work prompts.
- `specs/0151-sdd-v1/clarification-a.md:3-19, 35-77, 79-117`: approved SDD
  authority, Project semantics, brownfield supersession, and task/live-status
  semantics.
- `specs/0155-private-safe-chat-checkpoints/spec.md:1-23, 24-50, 64-91`:
  privacy boundary, sanitized checkpoints, preserved Project lifecycle,
  fail-closed behavior, and non-scope.
- Git history shows PR #160's merge parents and the three-step feature evolution
  above.

## 5. Unresolved questions and evidence limitations

I could not retrieve the four GitHub URLs because the browsing source returned
cache misses and the local Git remote could not resolve GitHub. Therefore I
abstain from asserting:

- the live status of Issue #159;
- the live Project/Kanban state;
- the exact review/check results recorded on PR #160;
- whether PR #161 is currently open, closed, merged, or otherwise resolved.

The anchor merge proves that PR #160's code was merged at that revision, but it
does not independently prove every completion-criterion check or current GitHub
workflow state. No owner decision should be inferred from retrieval frequency
or branch presence.

## 6. Next valid action

Use the GitHub Issue/PR/Project surfaces to verify current live status and
inspect PR #160's independent Sol/High review and owner-approval records. If
GitHub evidence remains unavailable, report CYAX-0159 as approved and merged at
the anchor, but leave live-status and completion-evidence claims explicitly
unverified.

## Inputs and consumption

Input artifacts opened:

- `research/temporal_provenance/pilots/cyax-0159/reconstruction_prompt.md`
- `research/temporal_provenance/pilots/cyax-0159/baseline_manifest.json`
- `specs/0159-lessons-corrections/spec.md` at the anchor
- `.agents/LESSONS.md` at the anchor
- `.agents/skills/cyaxiverse-sdd/SKILL.md` at the anchor
- `.github/pull_request_template.md` at the anchor
- `specs/0151-sdd-v1/clarification-a.md` at the anchor
- `specs/0155-private-safe-chat-checkpoints/spec.md` at the anchor
- Git objects for the four manifest history hashes
- Local descendant branch metadata relevant to PR #161

GitHub URLs from the manifest were attempted but returned no readable content.

Approximate context consumed: 6,215 words from the prompt, manifest, and six
listed repository artifacts; approximately 8,000–9,000 words including Git
history and diff excerpts.
