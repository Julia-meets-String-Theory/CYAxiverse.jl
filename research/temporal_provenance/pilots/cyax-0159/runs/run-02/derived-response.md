# Run 02 — derived temporal-provenance response

CYAX-0159 is complete on `vmm` with R-001–R-007 implemented. The durable
integration boundary is:

- `vmm@856012f015a866bf7ff352bc50e8d10c250855e6` — PR #160 merge commit
  (`[workflow_state]`).
- Accepted PR #160 implementation head:
  `bd3e866537b0115a4cb76cbae31b3768da3de724` (`[implementation_evidence]`).
- Default `main` has diverged and does not contain the accepted merge.

Authority comes from Issue #159 for the work item's objective and requirements,
the approved CYAX-0159 specification for R-001–R-007 and non-scope, PR #160
owner-approval comment `5636316397` for the owner decision, Git commits for
implementation state, and Issue/PR/Project surfaces for workflow state.
Validated lessons are advisory only and do not create another authority tier.

PR implementation revision history is separate from lesson lifecycle:

- `2b9b056` was the initial PR implementation. A reported adversarial review
  rejected its ready-for-acceptance claim.
- `f965dc6` superseded that implementation with authority and lifecycle repairs.
- `bd3e866` superseded `f965dc6` with the approved-status and provenance update.
- The accepted L-0001–L-0006 are `validated`, `not promoted`, and each has
  `Superseded by: N/A`; the PR commit sequence does not semantically supersede
  those lessons.
- PR #161 at `e1288cfe6a259d1f3f3c2d2c90e20bcf8d483b56` is open and unmerged. Its
  L-0007–L-0009 remain candidates and are not part of CYAX-0159's accepted
  six-lesson state.

The material durable anchors are the approved specification for R-001–R-007 and
the completion criterion; `.agents/LESSONS.md` at `bd3e866` for the six lesson
states; the PR #160 owner-approval comment for approval and validation; commits
`2b9b056`, `f965dc6`, `bd3e866`, and merge `856012f` for revision history and
integration; the Issue #159 close comment for completion on `vmm`; PR #161 for
the candidate boundary; and the `856012f...main` comparison for branch
divergence.

Two evidence boundaries remain. First, CYAX-0159 is integrated on `vmm`, not on
divergent `main`; a release to `main` is separate work. Second, PR #160 reports
independent review and focused-check PASS results, but the observed public PR
surface exposed no formal review or check records. Treat those results as
reported, not formally verified. No implementation, review, or owner decision
remains open for #159 itself.

No action is required for #159. Any new lesson, validation, or normative
promotion needs separate reviewed work. If release to `main` is required, use a
separate explicit integration action.

Inputs opened: `reconstruction_prompt.md`, `graph_manifest.json`, and
`context.md`. Canonical artifacts opened: none—the context resolved all material
ambiguities. Approximate words consumed: 2,359.
