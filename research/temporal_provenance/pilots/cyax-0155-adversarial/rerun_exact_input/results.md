# CYAX-0163 clean exact-input rerun results

Final result: **B MATCHES A**.

This clean rerun repairs only the dispatch contamination recorded in
`../contamination.md`. The earlier six nominal outputs remain inadmissible and
were not used as evidence, prior information, or a reason to change the frozen
contexts, answer key, rubric, or launch order.

## Frozen identities

- Source snapshot: `cyax-0155-adversarial-20260913T043555Z`.
- Preregistration and answer-key freeze:
  `44937a83cc039805d072bd0263517d8dfef03a10`.
- Reviewed context revision:
  `cc9545ff7bdecb44f3c5bf39d9eed3405c7ddc42`.
- Exact-dispatch harness freeze:
  `7419b755d4901e3475368dcec03c84b4f553b7b6`.
- Common prompt: 899 bytes, SHA-256
  `cf9b6aec4c7e232a9c2462e2f100dcefb6e49ea67422ce7670c4f85e66f3c52a`.
- Delimiter: 30 bytes, SHA-256
  `aa23570e27878cc6f2b1871145b5cca8a53128ee24ecaa12af68aca8f6f3f2f3`.
- Condition A context: 6,429 bytes, SHA-256
  `aedd1350bef5389be90d2608318f7b9675f94aad0a50a8d5d59499fb017bc780`.
- Condition B context: 6,702 bytes, SHA-256
  `33c61ad16a5228a557ae59f0ac80834674e40aae5b77a0a9e0044e82c1b2237c`.
- Complete A input: 7,358 bytes, SHA-256
  `d33243fcfd5c17698ebd6d4e64e3dce8633e889c88935c2e11ccf8f3f6a1354f`.
- Complete B input: 7,631 bytes, SHA-256
  `523d3f90ced130219eb4a06c7658d6ec4c40530d0b5e1999113476beca6c1f9f`.
- Launch order: A1, B1, B2, A2, A3, B3.
- Subjects: six fresh GPT-5.6 Sol/high processes, each in a fresh empty
  ephemeral read-only workspace with no inherited project rules or user
  configuration.

## Run evidence and blind scores

| Run | Input SHA-256 | Output bytes | Output SHA-256 | Total | Conflict-critical |
| --- | --- | ---: | --- | ---: | ---: |
| A1 | `d33243fcfd5c17698ebd6d4e64e3dce8633e889c88935c2e11ccf8f3f6a1354f` | 4,482 | `8adf61d6cc26c76eec28bb049609ba6c383320940cbd75b41fa402080549fef0` | 12/12 | 5/5 |
| A2 | `d33243fcfd5c17698ebd6d4e64e3dce8633e889c88935c2e11ccf8f3f6a1354f` | 3,764 | `456ba1fe60a14465c5794f910ba9a74a04529592ac6a3a891cbb928e4e33bbe2` | 12/12 | 5/5 |
| A3 | `d33243fcfd5c17698ebd6d4e64e3dce8633e889c88935c2e11ccf8f3f6a1354f` | 4,247 | `e9272dcf24d6b208b001098c62bf641a20f766dfeb1580d3ca54429aa2330c3d` | 12/12 | 5/5 |
| B1 | `523d3f90ced130219eb4a06c7658d6ec4c40530d0b5e1999113476beca6c1f9f` | 4,115 | `1f518c93e978176e1a22fd6a700756c09f19833eb178a558e1eb375acc8488ab` | 12/12 | 5/5 |
| B2 | `523d3f90ced130219eb4a06c7658d6ec4c40530d0b5e1999113476beca6c1f9f` | 4,071 | `178aada6be7c756f88d01914ff0a45d17c538a92e3670d219abede697036260c` | 12/12 | 5/5 |
| B3 | `523d3f90ced130219eb4a06c7658d6ec4c40530d0b5e1999113476beca6c1f9f` | 4,760 | `fd24565fb69cd51cb6410c8637ffc04e205f6ccf2d23ac7d088cb2910cb055cf` | 12/12 | 5/5 |

All runs completed, used distinct privacy-safe identity pseudonyms, and had
zero source reopenings or tool events. All A input hashes match each other; all
B input hashes match each other. The blinded response files are byte-identical
to the captured outputs. The pre-scoring content-leakage check found no A/B,
graph/non-graph, or relational labels. Scorecards were frozen before the
condition mapping was revealed.

Every scorecard records zero automatic failures, unsupported assertions,
authority errors, temporal errors, supersession errors, unjustified
inferences, incorrect abstentions, next-action errors, and source reopenings.
Every response records one correct abstention.

## Interpretation

Condition B matches Condition A on total accuracy, conflict-critical accuracy,
authority handling, temporal and supersession handling, abstention, and next
action. Under the preregistered decision rule, the result supports structured
provenance/context assembly as the default Issue #163 direction. Explicit graph
infrastructure remains behind a new evidence threshold.

This is evidence from one historical work item and three runs per condition.
It does not establish general representation equivalence, statistical
significance, or a backend choice. The finite frozen source search establishes
absence only within its recorded scope and observation time, and the paired
contexts were manually curated before deterministic rendering.

The next gate is owner/Control Desk review. Do not begin another relational
experiment, fuzzy/Table-1 testing, backend evaluation, PR merge, Issue #163
approval, or private-memory infrastructure changes from this result alone.

Independent methodology/evidence review passed with no required repair. See
`methodology_review.md`.
