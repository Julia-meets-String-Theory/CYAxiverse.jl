# CYAX-0157 Held-Out Ablation — Preregistration

## Pilot identity

- **Pilot:** CYAX-0157 held-out ablation (CYAX-0163 R-008 gate)
- **Snapshot:** `cyax-0157-ablation-20260913T0122Z`
- **Source manifest:** `source_snapshot.json` in this directory
- **Captured sources:** `sources/` subdirectory (exact frozen GitHub fixtures)
- **Branch:** `research/issue-163-temporal-provenance`
- **Observation time:** 2026-09-13 approximately 01:22 UTC
- **Purpose:** Determine whether a concise provenance-aware relational context
  produces materially better reconstruction than an equally concise non-graph
  structured summary, both built from the same frozen canonical snapshot.

This preregistration is committed before either experimental context exists.

## Frozen canonical snapshot

Both conditions use only the sources frozen in `source_snapshot.json` and the
captured fixtures under `sources/`. No live GitHub reads, ledger access, or
source reopening is permitted at any stage. The snapshot contains:

| Source class | Identity |
| --- | --- |
| Governing Issue #157 | OPEN, Project Verification, fixture `sources/issue_157.json` |
| Supporting PR #158 | OPEN, non-draft, unmerged, head `8f6a9c28...`, base `856012f...`, fixture `sources/pr_158.json` |
| PR #158 comments | 2 comments, fixture `sources/pr_158_comments.json` |
| PR #158 reviews | Empty, fixture `sources/pr_158_reviews.json` |
| PR #158 checks | Fast FAILURE, Documentation SUCCESS, Full SKIPPED, fixture `sources/pr_158_checks.json` |
| CYAX-0157 spec | blob `1c670b6f...`, content SHA-256 `81c8ad7b...` |
| AGENTS.md | blob `e7ece2c1...`, content SHA-256 `b213b788...` |
| CYAX-0155 spec | blob `b686c91b...`, content SHA-256 `4e010cd6...` |
| PR #158 head commit | `8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45` |
| PR #158 head tree | `fc980d28f3faa51f66031fe2e1fb5281b2ad5a48` |
| PR #158 base | `856012f015a866bf7ff352bc50e8d10c250855e6` |
| R-005 status | Open; public branch scan deferred |
| Prior head | `1ed97e18...` stale relative to current `8f6a9c28...` |

All captured fixtures are hashed in `source_snapshot.json` and validated
offline by `validate_snapshot.py`.

## Conditions

### Condition A — Derived temporal-provenance context (relational)

A concise provenance-aware context generated from a JSONL temporal ledger and
`context_builder.py`, using the same frozen snapshot. Source citations,
temporal annotations, authority scoping, and explicit relational structure
follow the existing prototype conventions from the CYAX-0159 pilot.

### Condition B — Structured summary (non-graph)

A concise structured summary of Issue #157/PR #158 current state, assembled
from the frozen snapshot by the experiment curator before the reconstruction
agent sees it. No temporal graph, ledger, or relational assertion vocabulary.
Plain structured prose with explicit source citations.

### Shared properties

- Both contexts are built from the same frozen `source_snapshot.json` and
  captured `sources/` fixtures.
- Both are provided to genuinely fresh agents with no prior CYAxiverse context.
- Neither agent may open live GitHub pages, browse the repository, request any
  source, or reopen any source. Subjects receive only the assigned frozen
  context document, common instructions, and the 12 questions.
- Source reopening is exactly zero (no source may be reopened for any reason).
- Both receive the same reconstruction prompt and the same 12 questions.
- Word count target: 2300–2500 words per context (Python `len(text.split())`).
- Maximum A/B word-count difference: 5%.
- Both agents use the same model and reasoning configuration.

## Reconstruction task

Each subject agent receives one context document and the following
instructions.

### Common instructions

You are a fresh agent with no prior CYAxiverse context. Using only the
provided context document, reconstruct the current authoritative state of
Issue #157 and PR #158 in the CYAxiverse repository. Answer all 12 questions
below. For each answer, cite the specific source from the context that
supports it. If the context does not contain sufficient evidence for a
confident answer, state what is missing and abstain rather than guess.

Do not open any other source. Do not access live GitHub, browse the
repository, or request additional materials. Answer using only the provided
context document.

### Questions (Q1–Q12)

1. What is Issue #157 and what is its scope?
2. What is PR #158 and what is its relationship to Issue #157?
3. What is the current state of Issue #157 (open/closed, project status)?
4. What is the current state of PR #158 (open/closed, draft status, merge
   status, head commit, base branch/commit, project status)?
5. What is the approved specification governing this work, what is its
   approval status, and what evidence exists that the current-surface
   remediation is implemented at the exact PR head without claiming merge
   or Issue completion?
6. What do the CI status checks and focused test results show at the
   current PR head?
7. What evidence exists for independent review of the PR, and what are the
   limitations of that evidence on the GitHub review surface?
8. What is R-005 and why does it remain open?
9. Why must merging PR #158 not close Issue #157, and what distinguishes
   PR completion from Issue completion?
10. What is the relationship between the prior head `1ed97e18...` and the
    current head `8f6a9c28...`?
11. What specific evidence limitations, uncertainties, or required
    abstentions apply to the current verification surface?
12. What is the next valid action for this work item?

## Scoring rubric (12 points)

Each question maps to exactly one answer-key item. Each item scores 1 if the
response contains the key fact with correct scope, 0 otherwise. Partial
credit is not awarded. The maximum score is 12.

| Item | Key fact (abbreviated) | Score |
| --- | --- | --- |
| K1 | Issue #157: remove machine-local path disclosures from public surfaces | 0 or 1 |
| K2 | PR #158: implements #157 remediation; does not complete #157 | 0 or 1 |
| K3 | Issue #157 OPEN, Project status Verification at observation | 0 or 1 |
| K4 | PR #158 OPEN, non-draft, unmerged, head `8f6a9c28...`, base vmm at `856012f...`, Project Verification | 0 or 1 |
| K5 | CYAX-0157 spec approved S1; current-surface remediation implemented at exact head with R-001–R-005 scope; no claim of merge/Issue completion | 0 or 1 |
| K6 | Fast FAILURE (unchanged Hessian mismatch), Documentation SUCCESS, Full SKIPPED; focused remediation tests passed | 0 or 1 |
| K7 | PR body reports independent re-review APPROVE at exact head; GitHub `reviews[]` empty; no formal review decision exposed | 0 or 1 |
| K8 | R-005: surviving public branches need read-only scan and preservation/reachability classification; no delete/rewrite yet | 0 or 1 |
| K9 | PR #158 must not close Issue #157; merge and Issue completion are distinct because R-005 remains open | 0 or 1 |
| K10 | Prior head `1ed97e18...` is stale relative to current `8f6a9c28...`; current re-review request targets the current head | 0 or 1 |
| K11 | No independent CI rerun; timestamp-scoped GitHub surface; PR body is self-reported verification; evidence limitation on review surface | 0 or 1 |
| K12 | Next action: obtain/record exact-head review/owner merge decision for PR #158; if merged keep #157 open; separately perform R-005 | 0 or 1 |

## Automatic failure conditions

A response receives an automatic failure (score 0 regardless of item count)
if it contains any of:

- Invented or inferred owner decision not present in the frozen sources.
- Superseded claim (e.g., prior head) treated as current without noting
  supersession.
- Material claim without a durable source citation from the provided context.
- Silent resolution of a dispute or ambiguity that the sources leave open.
- Reliance on private context, live GitHub access, or sources not in the
  frozen snapshot.
- Salience, retrieval order, or agent confidence treated as authority.
- Claim that PR #158 is merged, Issue #157 is closed, or R-005 is complete
  when the snapshot shows otherwise.

## Word accounting

For each condition's context and each response, record:

| Metric | Method |
| --- | --- |
| Word count | Python `len(text.split())` on the UTF-8 text |
| Byte count | Python `len(text.encode('utf-8'))` on the UTF-8 text |
| Distinct source references | Count of unique canonical source identifiers cited |
| Evidence items | Count of distinct factual claims with explicit source anchors |
| Fact inventory | Fixed 12-item inventory mapped to K1–K12 |

## Permitted source references

Both conditions may cite only:

- `github:Julia-meets-String-Theory/CYAxiverse.jl#157` (Issue)
- `github:Julia-meets-String-Theory/CYAxiverse.jl#158` (PR)
- `specs/0157-public-path-remediation/spec.md` at blob `1c670b6f...`
- `AGENTS.md` at blob `e7ece2c1...`
- `specs/0155-private-safe-chat-checkpoints/spec.md` at blob `b686c91b...`
- Git objects: commits `8f6a9c28...`, `856012f...`, `1ed97e18...`;
  tree `fc980d28...`
- PR #158 status checks and comments as captured in `sources/`
- R-005 status as stated in Issue #157 and PR #158 bodies

No other sources are permitted. No live reads. No ledger or prior pilot data.

## Launch order

The precommitted launch order is: **A1, B1, B2, A2**.

- A1: Condition A (relational/provenance), first fresh agent.
- B1: Condition B (non-graph summary), first fresh agent.
- B2: Condition B (non-graph summary), second fresh agent (replication).
- A2: Condition A (relational/provenance), second fresh agent (replication).

The alternating order (ABBA counterbalancing) prevents systematic position
effects. Each launch uses a genuinely fresh agent session with no CYAxiverse
context.

## Blind scoring procedure

- The scorer receives all four responses with condition labels removed and
  responses identified only by opaque run identifiers.
- The scorer applies the 12-point rubric independently to each response.
- After all four are scored, condition labels are revealed and the paired
  comparisons computed.
- Automatic failure checks are applied before point totals.

## Stop rules

- If both A1 and B1 receive automatic failures, stop and diagnose context
  quality before proceeding to B2 and A2.
- If any context exceeds the 2500-word limit or the A/B difference exceeds
  5%, rebuild the out-of-spec context before launching its runs.
- Complete all four runs unless a contamination or automatic-failure incident
  invalidates the experiment.

## Preregistration integrity

This document is committed before either Condition A or Condition B context
exists. The commit that adds the final preregistration must not also add any
reconstruction context, response, or scorecard. The source snapshot and
captured fixtures are frozen at the same commit.

Snapshot identity: `cyax-0157-ablation-20260913T0122Z`
