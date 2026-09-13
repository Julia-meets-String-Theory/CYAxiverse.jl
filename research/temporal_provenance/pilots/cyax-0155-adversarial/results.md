# CYAX-0155 adversarial reconstruction results

Result: **INCONCLUSIVE — dispatch contamination**.

## Frozen identities

- Source snapshot: `cyax-0155-adversarial-20260913T043555Z`.
- Preregistration and answer-key freeze: `44937a83cc039805d072bd0263517d8dfef03a10`.
- Reviewed context revision: `cc9545ff7bdecb44f3c5bf39d9eed3405c7ddc42`.
- Condition A: 852 words, 6,429 bytes, SHA-256 `aedd1350bef5389be90d2608318f7b9675f94aad0a50a8d5d59499fb017bc780`.
- Condition B: 895 words, 6,702 bytes, SHA-256 `33c61ad16a5228a557ae59f0ac80834674e40aae5b77a0a9e0044e82c1b2237c`.
- Common prompt: 122 words, 899 bytes, SHA-256 `cf9b6aec4c7e232a9c2462e2f100dcefb6e49ea67422ce7670c4f85e66f3c52a`.
- Parity: PASS before launch; 12 answer-bearing facts, 18 evidence items, 10 source identities each; 43-word difference, 4.804% of the larger context.

## Nominal response scores

These scores describe the frozen responses but are not admissible for a
condition comparison because the dispatched inputs were contaminated.

| Run | Raw total | Conflict-critical | Automatic failure | Source reopenings |
| --- | ---: | ---: | --- | ---: |
| A1 | 12/12 | 5/5 | none | 0 |
| A2 | 12/12 | 5/5 | none | 0 |
| A3 | 12/12 | 5/5 | none | 0 |
| B1 | 12/12 | 5/5 | none | 0 |
| B2 | 12/12 | 5/5 | none | 0 |
| B3 | 12/12 | 5/5 | none | 0 |

All nominal scorecards record zero unsupported assertions, authority errors,
temporal errors, supersession errors, unjustified inferences, and incorrect
abstentions; one correct abstention and a correct next action per run.

## Error-pattern comparison

At the response layer, no condition-specific error pattern appeared: all six
responses preserved current-versus-historical state, implementation-versus-
Project completion, the missing resolution/supersession evidence, authority,
abstention, and a verification-first next action. Several responses mentioned
reopening only conditionally after future verification; none asserted that #155
should be reopened from the frozen evidence alone.

This apparent parity cannot support the preregistered “evidence against
near-term graph investment” conclusion because A and B were not dispatched at
their frozen byte identities and replicate inputs varied. No graph/non-graph
winner is declared.

## Limitations

- The exact-input dispatch contract failed; planned input word counts are not actual dispatched counts.
- One historical work item and three attempted runs per condition cannot establish equivalence.
- The source search proves only absence within its frozen scope and time.
- Contexts were manually curated before deterministic rendering.
- Blind artifacts used opaque IDs, but the manager who launched runs was not epistemically blind to the mapping.
- No statistical-significance claim is made.

## Conservative interpretation and next gate

The only valid result is INCONCLUSIVE. Existing prior evidence still supports
curated structured context plus provenance and still does not justify a graph
backend, but this contaminated pilot neither strengthens nor weakens that
evidence.

Next gate: separately preregister a repaired execution using the already
reviewed source/context identities or a newly frozen equivalent, dispatch exact
prompt-plus-context bytes to six fresh GPT-5.6 Sol/high subjects, record input
hashes at launch, and repeat blind scoring. Do not select a backend or begin the
fuzzy/Table-1 stress test.
