# Run 01 — derived temporal-provenance response

## 1. Authoritative current state and exact boundary

- CYAX-0159 is complete on `vmm`, with R-001–R-007 implemented.
- The approved implementation head is
  `bd3e866537b0115a4cb76cbae31b3768da3de724`.
- The integration boundary is merge commit
  `856012f015a866bf7ff352bc50e8d10c250855e6` (PR #160 into `vmm`, merged
  2026-09-11T16:37:54Z).
- In this checkout, `origin/vmm` is now
  `e619da813951db261a2a885fde791b50a8940046`, whose first parent is
  `856012f`; the anchor is therefore contained in `vmm`.
- `main` is a separate, divergent boundary. Local `main` is
  `f0e6e7eb30015aff78790cf07119639ba0f6163c`; merge base with `856012f` is
  `9e7609a9b0a769b15f9a6b6e78bf89534740577c`. Neither branch contains the
  other, and `main` does not contain the CYAX-0159 merge.

## 2. Why it is authoritative

Authority follows the approved CYAX-0159 specification:

`AGENTS.md` + relevant skill → approved spec → plan → tasks → GitHub Issue/Project → implementation.

The approved spec is `specs/0159-lessons-corrections/spec.md` at `bd3e866`,
with `status: approved` and approval reference PR #160 comment `5636316397`.
Issue #159 is the governing intake object; PR #160 supplies implementation,
integration, and evidence. Validated lessons are advisory only. Candidate
lessons cannot override the hierarchy or establish merge, Issue, Project,
review, or scientific-acceptance state.

Dependencies are approved CYAX-0151 Clarification A at corrected revision
`6434e133af4b91db81babfafa024fc3cbada901a` and CYAX-0155 at
`f0013552cd69a93221464e9c8ccfd56339f39052`.

## 3. Non-current, superseded, or disputed state

- `2b9b056` was the initial implementation and was superseded after an
  adversarial review requested changes to authority-tier and
  terminal-promotion semantics.
- `f965dc6` repaired those issues and was then superseded by final
  status/provenance correction `bd3e866`.
- PR #161 at `e1288cfe6a259d1f3f3c2d2c90e20bcf8d483b56` remains an open,
  unmerged candidate-only change for L-0007–L-0009. It is not part of
  CYAX-0159's accepted six-lesson state.
- The accepted registry contains exactly six validated lessons: L-0001 through
  L-0006.
- The initial “ready for acceptance” position is rejected by the reported
  initial REQUEST_CHANGES review.

## 4. Supporting evidence

- Approved spec: R-001–R-007, authority chain, lifecycle, privacy boundary,
  non-scope, verification, and completion criterion.
- Accepted registry: six validated entries, candidate → validated → superseded
  lifecycle, advisory/non-authoritative semantics.
- Accepted SDD skill: lesson hooks at task start, during work, and convergence.
- Accepted PR template: four Lessons/corrections questions with S0 `N/A`
  handling.
- Git evidence: `856012f` is an ancestor of `origin/vmm`; it is not an ancestor
  of `main`.
- Commit history confirms `2b9b056 → f965dc6 → bd3e866 → 856012f`.
- Clarification A confirms approved authority, brownfield supersession,
  `tasks.md` non-authority, and advisory WIP semantics.
- CYAX-0155 confirms publication sanitization and fail-closed privacy handling.

## 5. Unresolved risks and evidence limitations

- Release boundary remains unresolved: CYAX-0159 is integrated into `vmm`, but
  not into divergent `main`. Do not describe it as released on `main`.
- The curated context classifies final independent reviews and focused checks as
  `reported`, not `verified`: PR #160 describes PASS results, but no formal
  GitHub review/check records were available on the PR surface.
- Live GitHub page fetches returned cache misses during this audit. Issue
  closure, PR #161 open status, and compare observations therefore rely on the
  regenerated curated context's cited workflow-state records.
- No owner decision remains for CYAX-0159 itself. Normative promotion or
  additional lesson validation requires separate reviewed work.

## 6. Next valid action

No action is required for CYAX-0159. Treat any L-0007–L-0009 promotion,
additional lesson validation, or normative rule promotion as separate reviewed
work. If release to `main` is required, perform an explicit vmm-to-main
integration/release action; do not infer it from vmm completion.

## Inputs and consumption

Artifacts opened:

- `research/temporal_provenance/pilots/cyax-0159/reconstruction_prompt.md`
- `research/temporal_provenance/pilots/cyax-0159/graph_manifest.json`
- Regenerated `research/temporal_provenance/pilots/cyax-0159/context.md`
- `specs/0159-lessons-corrections/spec.md` at `bd3e866`
- `.agents/LESSONS.md` at `bd3e866`
- `.agents/skills/cyaxiverse-sdd/SKILL.md` at `bd3e866`
- `.github/pull_request_template.md` at `bd3e866`
- `specs/0151-sdd-v1/clarification-a.md` at `6434e133`
- `specs/0155-private-safe-chat-checkpoints/spec.md` at `f0013552`
- Git metadata for commits `2b9b056`, `f965dc6`, `bd3e866`, `856012f`,
  `e619da8`, `e1288cf`, and refs `vmm`/`main`

Approximate consumed text: 2,106 words of prompt/manifest/context plus about
6,011 words of canonical artifact excerpts/full files, approximately 8,100
words total excluding terse Git metadata. I did not read
`baseline_manifest.json`, `ledger.jsonl`, or
`research/temporal_provenance/README.md`.
