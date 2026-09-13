# Temporal-provenance memory experiment

This directory is the backend-independent research prototype for Issue
[#163](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/163),
under the durable-memory programme in Issue
[#162](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/162).
The draft research contract is
[`specs/0163-temporal-provenance-memory/spec.md`](../../specs/0163-temporal-provenance-memory/spec.md).

The prototype tests one question: does deterministic provenance-aware
materialization plus concise structured context assembly help a fresh agent
reconstruct authoritative CYAxiverse state more reliably and with less context
than direct reconstruction from unstructured artifacts? The current default
direction is authoritative sources → deterministic extraction/materialization
→ structured context → fresh agent. Graph-shaped representation is optional,
downstream, and evidence-gated.

The JSONL ledger and generated context are disposable. GitHub Issues, pull
requests, owner decisions, approved specifications, repository revisions, and
verification artifacts remain authoritative.

## Phase 0 validation

The preliminary Issue #163 synthesis is now narrowed by the completed pilots.
The systems below are design references and comparator patterns, not evidence
that a graph will improve CYAxiverse reconstruction or selected dependencies.

| System | Useful evidence | CYAxiverse limitation / decision |
| --- | --- | --- |
| [Flow](https://github.com/samyakkkk/flow/tree/53b9b45daf657c2ef45d561ed2b596383ff76c20) | An integrated coding-memory reference: one Brain per project, source-at-commit reads, typed graph writes with provenance, an append-only [mutation journal](https://github.com/samyakkkk/flow/blob/53b9b45daf657c2ef45d561ed2b596383ff76c20/flow-t3/shared/graph-gateway/src/journal.ts), evidence cards, revisioned living documents, and compare-and-swap updates. Its orientation workflow directly targets restart and compaction recovery. | The graph ontology is code-centric. Entity and relation upserts replace properties in place; entity merging deletes the duplicate. Memory refinement replaces the canonical claim, memory deletion cascades to evidence, and recency/strength can sink active memories. These are useful retrieval and operations mechanisms, not a durable scientific-authority model. Borrow the source verification, journal, evidence display, isolation, and writer-safety patterns; do not adopt its ontology or memory-strength policy as the authoritative layer. |
| [MnemoBrain / MnemeBrain Lite](https://github.com/mnemebrain/mnemebrain-lite/tree/b20be4f4c513b3e15e6fb5bc8718e24b0eceac34) | Public source fact at pinned revision: an evidence/provenance belief graph with explicit truth states, an `EvidenceLedger`, evidence-driven revision, temporal decay, and a `WorkingMemoryFrame`; the public architecture lists consolidation as a separate planned/in-progress layer. | Architectural/operational interpretation for #163: keep reflex/episodic recall separate from deliberate durable knowledge. This interpretation does not make either tier authoritative; canonical evidence and owner-scoped review remain required. No backend is adopted. |
| [Mnemosyne](https://github.com/mnemosyne-oss/mnemosyne/blob/5f3d7df84b6eea1c127a448aa4eedb600f0cec8e/docs/architecture.md) | Public source fact at pinned `docs/architecture.md`: local-first SQLite storage with working, episodic, and scratchpad tiers; automatic prompt injection, temporal retrieval, and decaying/importance-weighted recall are described. | #162-style comparator for cross-session episodic/reflex memory, automatic recall/context injection, local SQLite-first operation, temporal retrieval, and temporary/decaying memory. Recency/importance and decay can guide retrieval only; they must not determine authority, supersession, or scientific currentness. |
| [GBrain](https://github.com/garrytan/gbrain/blob/a6be012a3bcfac42e279630aedec5cda4a450e29/docs/protocol/MEMORY_VERBS_v1.md) | Public source fact at pinned `MEMORY_VERBS_v1.md`: a frozen seven-verb MCP protocol with mandatory provenance/evidence fields, additive versioning, `context_pack`/`delta`, budget metadata, and conformance checks. | First-class #163 pattern source for evidence-bearing retrieval, audit-preserving withdrawal, versioned/frozen memory protocol, deterministic token-budgeted context assembly, explicit gap/unknown reporting, and a narrow MCP surface. Semantic similarity may propose candidate supersession; durable CYAxiverse evidence determines authority. |
| [Graphiti/Zep](https://github.com/getzep/graphiti) | Episodes, edge provenance, valid time versus system time, fact invalidation, and hybrid semantic/keyword/graph retrieval. | Extraction, entity resolution, and contradiction handling are model-driven. Current issues document unsafe invalidation scope, stale-fact ranking, backfill gaps, and inconsistent invalidation ([#1489](https://github.com/getzep/graphiti/issues/1489), [#1645](https://github.com/getzep/graphiti/issues/1645), [#1728](https://github.com/getzep/graphiti/issues/1728), [#1841](https://github.com/getzep/graphiti/issues/1841)). Borrow the temporal mechanics; do not use it as the authority layer. |
| [NornicDB canonical ledger](https://github.com/orneryd/NornicDB/blob/main/docs/user-guides/canonical-graph-ledger.md) and [Roynard](https://arxiv.org/abs/2604.11364) | Separating durable knowledge, decaying episodes, and evidence-gated guidance is useful. `FactKey`/`FactVersion`, validity windows, mutation records, and as-of reads are relevant. | Roynard is a conceptual proposal, not independent validation. NornicDB's example closes the old version by mutation, a single `CURRENT` value cannot represent unresolved alternatives, and persistence/safety depend on backend configuration. Keep the pilot backend-independent and allow disputes. |
| [Mem0](https://arxiv.org/abs/2504.19413) | Practical extraction and memory update/retrieval baseline. | Paper, current graph product, and implementation history differ. A reported hard-delete path loses temporal history ([mem0 #4187](https://github.com/mem0ai/mem0/issues/4187)). It lacks CYAxiverse authority and durable episode-to-assertion evidence semantics. Use only as a later comparison baseline. |
| [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) | Entity/relation/claim extraction, graph-plus-text retrieval, hierarchical community summaries, and Local/Global/DRIFT context assembly. Optional covariates carry status and time bounds. | It is an indexing and retrieval system, not an append-only temporal authority ledger. Claim extraction is optional and generated. Consider it only as a disposable read/context view after the pilot. |
| [Hindsight](https://arxiv.org/abs/2512.12818) | Separates world facts, experiences, observations, and beliefs; combines semantic, lexical, graph, and temporal retrieval. | Occurrence and mention times do not establish full transaction-time history. Synthesized observations and confidence are not source authority, and immutable supersession is not a demonstrated contract. Borrow evidence-oriented retrieval ideas, not authority semantics. |

Important corrections to the preliminary synthesis:

- Product claims about bitemporality must be verified with actual backfill,
  invalidation, current-state, and as-of queries.
- GraphRAG has limited claim status, time-bound, and source-text provenance; the
  missing pieces are immutable assertion lineage, authority, and safe
  supersession, not all temporal metadata.
- Evidence-grounded memory objects remain generated interpretations unless
  they retain and correctly scope canonical source authority.
- Vendor benchmark results do not establish benefit for CYAxiverse.

### Pinned comparator source identities

The comparator rows above separate public source facts from CYAxiverse
interpretation. The exact public primary documents used for the three added
comparators are pinned by repository revision:

| Comparator | Public document | Revision |
| --- | --- | --- |
| MnemoBrain / MnemeBrain Lite | [`README.md`](https://github.com/mnemebrain/mnemebrain-lite/blob/b20be4f4c513b3e15e6fb5bc8718e24b0eceac34/README.md) and [`docs/architecture.md`](https://github.com/mnemebrain/mnemebrain-lite/blob/b20be4f4c513b3e15e6fb5bc8718e24b0eceac34/docs/architecture.md) | `b20be4f4c513b3e15e6fb5bc8718e24b0eceac34` |
| Mnemosyne | [`docs/architecture.md`](https://github.com/mnemosyne-oss/mnemosyne/blob/5f3d7df84b6eea1c127a448aa4eedb600f0cec8e/docs/architecture.md) | `5f3d7df84b6eea1c127a448aa4eedb600f0cec8e` |
| GBrain | [`docs/protocol/MEMORY_VERBS_v1.md`](https://github.com/garrytan/gbrain/blob/a6be012a3bcfac42e279630aedec5cda4a450e29/docs/protocol/MEMORY_VERBS_v1.md) | `a6be012a3bcfac42e279630aedec5cda4a450e29` |

These external revisions are source identities only. They are not dependency
pins, integrations, or authority for CYAxiverse artifacts.

### Comparator synthesis

The comparator roles are intentionally different. Flow is a revision-pinned
coding-memory reference: retain source-at-commit reads, evidence cards,
mutation journaling, project isolation, and compare-and-swap updates, while
rejecting historical erasure, hard deletion, and recency-driven authority.
The pinned MnemoBrain / MnemeBrain Lite sources identify an evidence/provenance belief graph,
explicit truth states, an EvidenceLedger, evidence-driven revision, and a
WorkingMemoryFrame; our
architectural interpretation is to separate reflex/episodic recall from
deliberate durable knowledge. The pinned Mnemosyne architecture is a #162-style
reference for cross-session episodic/reflex recall, automatic context
injection, local SQLite-first operation, temporal retrieval, and temporary or
decaying memory; its recency/importance machinery cannot establish scientific
authority. The pinned GBrain protocol is a first-class #163 pattern source for
provenance-bearing durable memories, evidence retrieval, audit-preserving
withdrawal, versioned/frozen memory, deterministic token-budgeted context
assembly, explicit gap/unknown reporting, and a narrow MCP memory surface.
GBrain may propose candidate supersession by semantic similarity, but durable
CYAxiverse evidence must decide whether supersession is authoritative.

None of these systems is integrated or selected as a backend. These patterns
inform the broader layer of structured provenance and deterministic context
assembly, which may later be implemented with JSONL assertions, relational
tables, SQLite/FTS, evidence cards, typed records, or optional graph
projections.

### Flow decision

Flow is more directly relevant than a generic memory library because it joins
agent-session evidence, a code graph, retained notes, living documents, and
orientation in one coding workflow. The comparison above is pinned to
`53b9b45daf657c2ef45d561ed2b596383ff76c20` (2026-09-12) because Flow declares
itself early and subject to frequent change.

The mechanisms worth testing or adapting are:

- verified committed-source reads with a full revision and normalized path;
- an append-only mutation journal separate from the current graph projection;
- evidence cards that expose source, branch, session, and observation details;
- one Brain per project, one long-lived writer, transactional updates, and
  compare-and-swap document revisions;
- an orientation view with a strict context budget.

The following mechanisms conflict with Issue #163's invariants:

- [`upsert_entity` and `upsert_relation`](https://github.com/samyakkkk/flow/blob/53b9b45daf657c2ef45d561ed2b596383ff76c20/flow-t3/shared/graph-gateway/src/verbs.ts)
  update the current graph in place, and entity consolidation deletes the
  duplicate after rewiring its edges;
- memory [`refines`](https://github.com/samyakkkk/flow/blob/53b9b45daf657c2ef45d561ed2b596383ff76c20/flow-t3/shared/orchestrator/src/memory/consolidate.ts)
  replaces the canonical claim, while contradiction changes counters and
  strength rather than retaining an explicit authoritative alternative;
- memory-card deletion can
  [cascade through observations and anchors](https://github.com/samyakkkk/flow/blob/53b9b45daf657c2ef45d561ed2b596383ff76c20/flow-t3/shared/orchestrator/src/memory/knowledge.ts);
- recency, evidence count, source weight, and contradiction penalties feed a
  [`strength` score](https://github.com/samyakkkk/flow/blob/53b9b45daf657c2ef45d561ed2b596383ff76c20/flow-t3/shared/orchestrator/src/memory/strength.ts),
  and a maintenance sweep can mark a memory `sunk`. That policy is appropriate
  for retrieval but cannot determine scientific authority or durable truth;
- its fixed [graph schema](https://github.com/samyakkkk/flow/blob/53b9b45daf657c2ef45d561ed2b596383ff76c20/flow-t3/shared/graph-gateway/src/schema.ts)
  represents software entities and dependencies, not project decisions,
  scientific claims, valid-time intervals, or explicit supersession.

Flow also strengthens the privacy requirement. It retains agent activity and
derived documents locally for later extraction and retry. CYAxiverse must keep
the existing fail-closed publication boundary even if a local Brain is used as
a disposable input or future comparison system.

The most relevant benchmark pressures are:

- [RECON](https://openreview.net/pdf?id=T3S5Blz7jM): multi-hop evidence chains,
  cascading invalidation, conflicts, counterfactuals, and temporal constraints;
- [AuthMem-Bench](https://arxiv.org/abs/2608.01679): source-authority collapse
  across memory consolidation;
- [LongMemEval-V2](https://arxiv.org/abs/2605.12493): dynamic state, workflow
  knowledge, environment gotchas, and premise awareness;
- [MemoryAgentBench](https://arxiv.org/abs/2507.05257): retrieval, test-time
  learning, long-range understanding, and selective forgetting;
- [MemBench](https://arxiv.org/abs/2506.21605): factual/reflective memory plus
  effectiveness, efficiency, and capacity;
- [LOCOMO-CONV](https://arxiv.org/abs/2609.03467): implicit and composed
  conversational retrieval rather than only explicit QA prompts.

None directly tests CYAxiverse's authority hierarchy, so the repository needs a
small task-specific benchmark rather than a borrowed aggregate score.

## Phase 1 prototype and default layer

The experiment implements one inspectable backend-independent form of the
default layer. It is evidence about provenance-aware materialization and
context assembly, not a normative requirement to store production memory as
Markdown or JSONL. A future implementation may use relational tables,
SQLite/FTS, evidence cards, typed records, or optional graph projections while
preserving the same identifiers and authority semantics.

The current prototype uses one JSONL stream with three record types:

1. `meta` defines the pilot root and reconstruction task.
2. `resource` gives a stable identity to canonical artifacts and derived
   semantic objects such as requirements, implementation versions, claims,
   verifications, and actions.
3. `assertion` represents every material relationship, including provenance,
   valid time, system recording time, epistemic status, and curation status.

One stream is smaller and easier to inspect than separate artifact, claim, and
link stores for this pilot. The record types can be split into tables later
without changing their identifiers or semantics.

### Load-bearing invariants

- Canonical resources have a public GitHub or repository-relative locator and
  a narrow authority class/scope.
- Every assertion cites at least one canonical resource and a source anchor.
- Authority belongs to the cited source and its scope. An extracted resource
  has only `agent_extraction` authority.
- `valid_time` states when a relationship held in the project/domain;
  `recorded_at` states when the ledger recorded it; source `observed_at` states
  when the prototype observed that artifact state.
- Currentness is derived. A current `supersedes` relation makes its target
  historical without declaring the historical state false. A `contradicts`
  assertion explicitly says whether it disputes or refutes the target.
- Only reviewed, non-candidate relationship assertions affect dispositions.
  `curator_checked` or `independently_reviewed` curation is required, and the
  epistemic state must no longer be `extracted`, `unresolved`, or `rejected`.
  Unreviewed or unresolved extractions remain inspectable candidates but are
  inert. Relations that are themselves superseded, disputed, contradicted,
  resolved, or stale through a required dependency are also ignored. Activity
  is resolved to a fixed point so an inactive relation cannot change its target.
- Required dependency staleness propagates only through explicit `depends_on`
  assertions. It is not inferred from chronology or semantic similarity.
- Supersession cycles, missing references, non-canonical provenance, invalid
  timestamps, duplicate IDs, and unsupported predicates fail validation.
- Salience is deliberately forbidden in ledger records. Ordering/filtering is
  a disposable retrieval concern and cannot become an authority-adjacent
  stored score.
- Corrections append new resources/assertions and explicit supersession or
  contradiction. Git preserves the ledger revision history.
- The prototype ingests no private transcript, resolving private reference, or
  machine-local locator.

The controlled vocabulary is intentionally narrow. It is not a repository-wide
ontology. The JSON Schema is descriptive; `context_builder.py` is the
executable validator for cross-record invariants.

### Known limitations

- Git review/merge serialization is the only multi-writer safety mechanism.
  There is no transactional service or compare-and-swap API.
- The pilot has only required/supporting/context dependencies; it does not yet
  model grouped `all`/`any` evidential alternatives.
- Currentness is evaluated over the frozen ledger, not through a general
  point-in-time query API.
- Curation remains a human/agent interpretation. The context therefore cites
  canonical anchors and tells the reconstruction agent to verify ambiguity and
  abstain.
- The prototype does not automate extraction, embeddings, lexical search,
  semantic search, graph databases, or source refresh.

These are deliberate Phase 1/2 boundaries, not production claims.

## Phase 2 prototype

Validate the pilot and regenerate the materialized context:

```bash
python3 research/temporal_provenance/context_builder.py \
  research/temporal_provenance/pilots/cyax-0159/ledger.jsonl \
  --output research/temporal_provenance/pilots/cyax-0159/context.md
```

Run focused tests:

```bash
python3 -m unittest -v \
  research/temporal_provenance/test_context_builder.py
```

The generated view groups the selected assertions into the six reconstruction
dimensions from Issue #162 and emits a canonical source index. It does not copy
source text or assign truth from retrieval order.

## Phase 3 pilot selection

CYAX-0159 is the clean pilot, frozen at
`vmm@856012f015a866bf7ff352bc50e8d10c250855e6`.

It provides a compact but non-trivial chain:

```text
owner-scoped Issue #159
  → approved CYAX-0159 specification and R-001–R-007
  → PR #160 implementation revisions
  → failed review claim and repair
  → owner approval and final provenance update
  → merge on vmm and Issue/Project completion
```

It also contains two useful authority boundaries:

- PR #161's L-0007–L-0009 are open, unmerged candidates, not additions to the
  accepted six validated lessons.
- `vmm` contains the accepted slice, while default branch `main` is currently
  divergent and does not contain that merge. A reconstruction must state its
  branch anchor rather than silently saying "the repository".

The pilot preserves an evidence caveat: PR #160 reports independent review and
focused verification in its body, but GitHub exposes no formal review or check
records. The prototype records this as an unresolved evidence limitation rather
than silently upgrading the PR narrative.

Issue #155 was later used as the bounded adversarial conflict/supersession
follow-up. Its repository-side privacy controls are part of the frozen source
snapshot, while the snapshot retains the unresolved closure/Projects evidence
gap. The first dispatch was contaminated and remains **INCONCLUSIVE**; the
exact-input repair is a separate valid rerun recorded below. The fuzzy/Table-1
history remains out of scope.

## Phase 4 benchmark protocol

Use two genuinely fresh agents with the same model, reasoning level, task text,
and response budget:

- **Baseline:** give `reconstruction_prompt.md` and the raw sources in
  `baseline_manifest.json`; prohibit access to the ledger/context.
- **Derived context:** give `reconstruction_prompt.md` and `context.md`; permit
  opening a cited canonical source only to resolve a material ambiguity that
  `context.md` cannot answer. For run 02, prefer zero source openings and allow
  at most two; record every opened source and its ambiguity-resolution reason.
  Prohibit access to the baseline manifest and ledger before answering.

Freeze the GitHub observations or record their retrieval time. Record every
artifact each agent opens and the approximate words/tokens consumed. Score each
dimension from 0–2:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Authority | wrong/invented | partly scoped | correct hierarchy and scopes |
| Current state | materially wrong | incomplete | correct, branch-anchored state |
| History | misses/misuses old state | partial | correct review/revision and candidate boundaries |
| Evidence | unsupported | some traceability | material claims trace to durable anchors |
| Uncertainty | silently resolves | vague caveat | exact missing/ambiguous evidence and abstention |
| Action | unsafe/skips gate | incomplete | valid action with required boundary |

Automatic failure follows Issue #162: invented owner decision, superseded claim
treated as current, material claim without durable provenance, silent dispute
resolution, reliance on inaccessible private context, or salience treated as
authority.

The initial bounded pilot passes only if the derived context improves
reliability or context efficiency materially. A graph-specific proposal has a
stricter binding gate: it must demonstrate a representation-specific need and
material benefit that concise structured provenance cannot satisfy. A
reasonable pre-registered threshold for the bounded pilot is:

- no automatic failure;
- at least 10/12 overall and no dimension below 1;
- at least two points better than the unstructured baseline, **or** equal score
  with at least 40% less consumed context;
- all current-state, candidate/non-current, branch-boundary, and evidence-caveat
  gold facts present.

These thresholds are provisional experimental design, not an Issue #163 owner
decision. Record each paired run's dimension scores, automatic-failure
classification, and reliability/efficiency result in a concise scorecard (for
example, `pilots/cyax-0159/runs/run-01/scorecard.md`). The first paired run
should report sensitivity rather than tune the ledger to the test agent's
answer.

### Pilot results

Run 01 scored the unstructured baseline at 10/12 and the derived response at
12/12, with no automatic failure. It showed a reliability improvement but no
efficiency improvement because the derived agent reopened most canonical
sources. The run therefore failed the efficiency classification.

After the run-01 scorecard identified exact omissions, the ledger was repaired
to make the accepted lesson lifecycle, PR revision lifecycle, and review/check
evidence boundary explicit. Run 02 used the pre-registered strict source budget.
The fresh agent opened no canonical source, scored 12/12, and consumed 2,359
measured words. The frozen baseline consumed 6,215 measured words before its
additional Git excerpts. This is a 62.0% measured-context reduction and a
two-point reliability improvement.

The CYAX-0159 pilot therefore provisionally passes the first meaningful success
condition: a fresh agent reconstructed authoritative current state,
supersession history, evidence, unresolved boundaries, and the next valid
action more reliably and with less context than the unstructured baseline. The
result supports a provenance-aware derived context layer. It does not yet show
that graph structure caused the benefit; manual curation and concise context
assembly remain confounds. Run 02 also followed a run-01-informed ledger repair,
so it is not a blind held-out result. The baseline could not retrieve live
GitHub pages, while the derived condition received timestamped GitHub
observations from the ledger. That is useful durable-cache behavior, but it is
also a source-availability asymmetry. A held-out pilot must use the same frozen
canonical snapshot for both conditions.

## Go/no-go boundary

Do not choose Graphiti, NornicDB, Neo4j, RDF/OWL, or another backend from this
prototype. The current default is structured provenance plus deterministic
context assembly. Graph-shaped representation is optional and downstream; it
requires a new preregistered use case demonstrating a representation-specific
need and material benefit that concise structured provenance cannot satisfy.
Examples include traversal that cannot be represented adequately in the context
budget, large-scale cascading dependency invalidation where graph operations
materially outperform simpler structures, multi-hop reconstruction whose
correctness or efficiency degrades without relational topology, or another
explicit representation-specific advantage. The completed held-out ablation
and exact-input adversarial rerun do not select a backend or approve the draft
specification.

## Frozen CYAX-0157 held-out ablation result

The frozen evidence is in
`research/temporal_provenance/pilots/cyax-0157-ablation/`, including the
preregistration, contexts, responses, scorecard, result, and methodology
review.

The pre-registered CYAX-0157 ablation tested whether the reliability signal
could be attributed to relational graph shape. Four fresh runs used the ABBA
order A1, B1, B2, A2, the same `gpt-5.6-sol` / `high` subject setting, a frozen
context revision, and zero source reopening. Condition A was the
provenance-aware temporal relational context; Condition B was an equally
capable concise structured summary without graph-shaped representation. Every
run scored 12/12 on the twelve binary K items, with no automatic failures.
After the blind scorecard was frozen, mapping gave A a mean of 12.0/12 and B a
mean of 12.0/12, for a difference of 0.0 points. B matched A.

This finite result does not show that graph-shaped representations have no
value, representation equivalence, or general ineffectiveness. It shows no
observed incremental reliability benefit from graph-shaped representation
beyond the curated structured context, provenance, authority scopes,
uncertainty, and current/stale distinctions shared by both conditions. The
conservative attribution is to structured context and provenance, not graph
shape. No backend is selected from this result.

## CYAX-0155 adversarial pilot and repair

The frozen evidence is in
`research/temporal_provenance/pilots/cyax-0155-adversarial/`, with the failed
dispatch under the pilot root and the repaired exact-input execution under
`rerun_exact_input/`.

The progression is explicit and the two executions must not be pooled:

1. The preregistered first execution dispatched inline transcriptions instead
   of the frozen input bytes, and replicate inputs differed. Its nominal six
   responses scored 12/12 and 5/5 conflict-critical, but the execution is
   **INCONCLUSIVE — dispatch contamination**. It is not admissible evidence
   for score, efficiency, representation, or backend claims.
2. The infrastructure was repaired and six new fresh subjects received exact,
   byte-identical per-condition inputs. Launch order was A1, B1, B2, A2, A3,
   B3; all runs had zero tool/source reopenings and blind scorecards. All six
   admissible runs scored 12/12 and 5/5 conflict-critical with zero automatic,
   authority, temporal, supersession, abstention, or next-action errors.
   The final result is **B MATCHES A**.

The exact-input rerun's independent methodology/evidence review returned
**PASS**. This is one historical work item and three runs per condition. It
does not establish general representation equivalence, statistical
significance, or a backend choice. Its source search is bounded by its recorded
scope and observation time, and contexts were manually curated before
deterministic rendering.

## Current conclusion and remaining gate

Existing experiments support concise structured provenance/context assembly as
the default Issue #163 direction. They do not currently demonstrate an
incremental benefit from graph-shaped representation. Graph infrastructure is
optional, downstream, and evidence-gated. No backend is selected, and the
CYAX-0163 specification remains draft and unapproved.

The next gate is owner/Control Desk review of the consolidated evidence and
draft specification. Do not run another representation experiment, start
fuzzy/Table-1 testing, evaluate a backend, alter #162 private-memory
infrastructure, or merge either research PR from this result alone.
