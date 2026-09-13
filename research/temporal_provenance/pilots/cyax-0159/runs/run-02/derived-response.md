# Run 02 — derived temporal-provenance response

CYAX-0159 is complete on `vmm` with R-001–R-007 implemented. The durable
integration boundary is:

- `vmm@856012f015a866bf7ff352bc50e8d10c250855e6` — PR #160 merge commit
  (`[workflow_state]`).
- Accepted PR #160 implementation head:
  `bd3e866537b0115a4cb76cbae31b3768da3de724` (`[implementation_evidence]`).
- Default `main` has diverged and does not contain merge `856012f`
  (`github:compare:856012f-main`, `[workflow_state]`). Do not treat `main` as
  containing CYAX-0159.

Authority comes from the approved specification `repo:spec:0159`
(`[approved_spec]`) and owner approval `github:pr:160:owner-approval`
(`[owner_decision]`). Implementation and merge claims come from the cited
repository and workflow artifacts; the supplied `context.md` is only a curated,
non-authoritative view.

L-0001–L-0006 are each `validated`, `unpromoted`, and have `Superseded: N/A`.
This lesson lifecycle is distinct from PR implementation revision history. The
code revisions are: initial `2b9b056` → review-repair `f965dc6` → accepted
status/provenance head `bd3e866`. That supersession chain does not supersede or
alter the six lessons.

PR #161 remains open at `e1288cfe...`; L-0007–L-0009 are unmerged candidates
with no owner merge or validation. They are not part of CYAX-0159's current
state.

PR #160 reports independent adversarial, standards/spec, and focused checks as
passing. However, the observed GitHub PR surface exposed no formal review or
check records (`github:pr:160:observed-surface`, observed
2026-09-13T01:20Z). Treat the PASS claims as reported workflow evidence, not
independently visible formal records; absence is not proof that checks failed.

Unresolved items are the separate `vmm`-to-`main` release boundary and the
formal-review/check evidence limitation. The next valid action is none for
#159; any lesson promotion or new lesson requires separate reviewed work. If
release to `main` is desired, handle that as a separate integration action.

Inputs opened: `reconstruction_prompt.md`, `graph_manifest.json`, and
`context.md`. Canonical artifacts opened: none—context resolved all material
ambiguities. Approximate words consumed: 2,359. No files edited.
