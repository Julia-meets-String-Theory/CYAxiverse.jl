---
spec_id: CYAX-0163
title: Temporal provenance for durable agent memory
issue: 163
class: S3
status: draft
workstream: Infrastructure
parent: 162
depends_on: []
created: 2026-09-12
last_reviewed: 2026-09-13
review_required: repository-owner review and independent architecture/evidence review before production adoption
approval_ref: null
---

# CYAX-0163 — Temporal provenance for durable agent memory

## Objective

Determine whether authoritative, provenance-aware materialization and concise
structured context assembly help a genuinely fresh agent reconstruct current
CYAxiverse state more reliably and with less context than direct reconstruction
from unstructured historical artifacts.

The working architecture under this draft is:

```text
authoritative durable sources
  → deterministic provenance-aware extraction/materialization
  → concise structured context assembly
  → fresh agent
```

The materialized records and context are derived and rebuildable. Canonical
GitHub and repository artifacts, owner decisions, approved specifications, and
their durable revision history remain authoritative.

## Status and approval boundary

This is a draft research specification. It does not approve a graph backend,
production memory service, automatic extraction pipeline, scientific ontology,
or change to repository authority. The backend-independent prototypes may
collect evidence while this specification is draft because they change no
package, scientific, or production-memory behavior. Production adoption needs
an explicitly reviewed specification revision and repository-owner approval;
`approval_ref` remains `null`.

The current direction supersedes the initial graph-oriented hypothesis without
rewriting its history:

1. The initial Phase-0 comparison and CYAX-0159 clean pilot established the
   question and a useful provenance-aware context signal.
2. The held-out CYAX-0157 ablation found no observed incremental reliability
   benefit from graph-shaped representation over a matched concise structured
   summary.
3. The first CYAX-0155 execution was contaminated by non-exact dispatch and is
   retained as **INCONCLUSIVE** and inadmissible for a representation claim.
4. The exact-input CYAX-0155 rerun repaired that dispatch failure and returned
   **B MATCHES A**, with independent methodology/evidence review **PASS**.

The contaminated execution is one failed execution followed by one valid
rerun, not two successful experiments. Its nominal scores remain diagnostic
history only.

## Requirements and evidence gates

| Requirement | Required behavior | Initial evidence gate |
| --- | --- | --- |
| R-001 Authority boundary | Canonical artifacts keep their existing authority and scope. Every derived material relationship cites a canonical source and anchor. Retrieval order, frequency, recency, and confidence do not increase authority. | Validator rejects assertions without canonical provenance; generated context labels itself disposable. |
| R-002 Minimal semantic model | Use the smallest backend-independent structured provenance/materialization vocabulary that can represent a selected pilot. Extracted assertions/resources have stable identities, explicit source anchors, recording time, and epistemic/curation state. Materialization is deterministic and rebuildable; do not create a repository-wide or scientific ontology. | Versioned records, source manifests, schema/validators, hash checks, and generated view/regeneration tests. |
| R-003 Time and state | Keep domain-validity time, observation/recording time, epistemic status, source authority, supersession, contradiction/dispute, and retrieval salience distinct. Preserve historical state rather than overwrite it. | Focused tests cover time shape, supersession, disputes, required dependency staleness, and forbidden stored salience. |
| R-004 Safety and privacy | Do not ingest private transcripts or machine-local locators. Fail closed when publication safety is uncertain. State the multi-writer safety boundary. | Diff/privacy checks and explicit prototype limitations. |
| R-005 Clean pilot | Use a bounded work item with a known answer and a decision/work-item/requirement/implementation/verification/outcome chain; do not begin with fuzzy/Table-1 history. | Frozen CYAX-0159 ledger and canonical source manifest. |
| R-006 Cold-start comparison | Give comparable fresh agents the same reconstruction task and frozen source snapshot. Compare an unstructured source condition with a concise structured provenance/context condition. Record sources opened, context consumed, dimension scores, and automatic failures. | CYAX-0159 pilot and frozen responses/scorecards. |
| R-007 Adversarial repair | Classify failed executions, repair the dispatch/infrastructure defect rather than coach the same agent, and preserve the failed run as inadmissible. A repaired execution must record exact inputs, identities, hashes, and independent review. | CYAX-0155 contamination record, exact-input rerun, final result, and methodology review. |
| R-008 Decision and graph evidence gate | Use the reliability/efficiency threshold and a held-out comparison against an equally concise non-graph structured context. Introduce graph infrastructure only after a future preregistered use case demonstrates material benefit from graph-specific structure that bounded structured provenance/context cannot satisfy. Backend enthusiasm or storage convenience is not evidence. | CYAX-0157 ablation, CYAX-0155 exact-input rerun, and future owner-reviewed representation-specific evidence; no backend is selected in this draft. |
| R-009 Scientific stress boundary | Start a complex scientific pilot only after the clean pilot, ablation, and adversarial gates pass. Never allow one extractor interpretation to become authoritative. | Separate reviewed slice with frozen sources, conflicting interpretations, negative evidence, explicit abstention, and a defined approval boundary. |

## Preserved semantic principles

- Authoritative sources remain authoritative; derived memory is rebuildable.
- Salience is a retrieval concern, not an authority signal.
- Owner decisions cannot be inferred from extracted or retrieved text.
- Accepted, rejected, superseded, disputed, and provisional states remain
  distinct.
- Provenance and source anchors survive context compression.
- Abstention is a valid result when evidence is insufficient.
- Temporal validity and observation/recording time remain distinct where
  materially relevant.
- Corrections and supersession append durable history; they do not silently
  erase prior claims.
- Graph, embedding, or LLM-derived relationships cannot become authoritative
  solely because they were stored.
- Required dependency staleness propagates only through explicit dependencies,
  not chronology or semantic similarity.

## Initial success criterion

The initial investigation succeeds only when a fresh agent reconstructs the
selected pilot's authoritative current state, authority basis, supersession
history, evidence, unresolved questions, and next valid action with materially
better reliability or materially less context than the unstructured baseline,
without an automatic authority/provenance failure.

A provisional threshold for the bounded pilot is:

- no automatic failure;
- at least 10/12 and no dimension below 1;
- at least two points above the baseline, or the same score with at least 40%
  less measured context;
- all current-state, candidate, branch-boundary, and evidence-limit facts.

Passing this threshold supports further investigation. It does not by itself
show that graph structure caused the benefit.

## Phase-0 comparison decision

Flow, MnemoBrain, Mnemosyne, GBrain, Graphiti/Zep, NornicDB/Roynard, Mem0,
GraphRAG, and Hindsight are reference systems, not authority sources or
selected dependencies. Their useful patterns inform structured provenance and
context assembly; none is adopted as a backend. Flow's code-centric ontology,
in-place graph/memory updates, hard deletion, and recency-based memory sinking
do not satisfy this specification's durable-authority boundary. MnemoBrain and
Mnemosyne are architectural/operational comparators, and GBrain is a pattern
source for evidence-bearing, frozen context assembly; all remain subject to the
same authority, privacy, and owner-approval boundaries.

## Non-scope

The initial investigation does not:

- graph the full repository or define a full scientific ontology;
- select or integrate a graph database;
- change Julia/Python package behavior, APIs, dependencies, or persisted
  scientific schemas;
- create a second task/status system;
- ingest or publish raw private conversations;
- infer owner decisions or automatically arbitrate scientific disputes;
- make any derived graph, index, ledger, or context irreplaceable.

## Evidence to date

### CYAX-0159 clean pilot

The clean pilot used the frozen `vmm@856012f015a866bf7ff352bc50e8d10c250855e6`
slice. The unstructured baseline scored 10/12. The repaired derived-context
run scored 12/12, opened zero sources, and used 2,359 measured words versus
6,215 measured baseline words: a 62.0% reduction. The pilot is a provisional
structured-context result, not a blind graph-attribution result; the repair was
run-01-informed and the conditions had a source-availability asymmetry.

### CYAX-0157 held-out representation ablation

The frozen ABBA experiment used four fresh `gpt-5.6-sol`/high subjects, with
zero source reopenings. Condition A (temporal relational/provenance context)
and Condition B (equally concise non-graph structured summary) both scored
12/12 on all four runs. A and B both had mean 12.0/12, a difference of 0.0
points, and independent methodology/evidence review **PASS**.

This finite result supports attribution to structured context/provenance shared
by both conditions, not to graph shape. It does not establish representation
equivalence, general ineffectiveness, statistical significance, or a backend
choice.

### CYAX-0155 adversarial conflict/supersession pilot

The initial six nominal responses scored 12/12 and 5/5 conflict-critical, but
the dispatch used inline transcriptions rather than exact frozen bytes and
replicates differed. The run is therefore **INCONCLUSIVE — dispatch
contamination**. No score, error rate, efficiency, representation, or backend
inference is drawn from it.

The repaired exact-input rerun used one byte-identical input per condition,
three fresh subjects per condition, launch order A1, B1, B2, A2, A3, B3, zero
tool/source reopenings, and blind scorecards. All six admissible runs scored
12/12 and 5/5 conflict-critical with zero automatic, authority, temporal,
supersession, abstention, or next-action errors. The final result is
**B MATCHES A**. Independent methodology/evidence review returned **PASS**.

This is evidence from one historical work item and three runs per condition.
It does not establish general representation equivalence or statistical
significance. Its finite source search is bounded by its recorded scope and
observation time, and the contexts were manually curated before deterministic
rendering.

## Graph evidence gate and non-scope

Graph-shaped representation is optional and downstream of authoritative
structured provenance/context assembly. A future graph proposal must first
preregister a use case and show material benefit that a bounded structured
context cannot explain, such as:

- traversal that cannot be represented adequately in bounded structured
  context;
- cascading dependency invalidation at a scale where explicit graph operations
  materially outperform simpler structures;
- multi-hop reconstruction whose correctness or efficiency degrades without
  relational topology; or
- another representation-specific advantage with an explicit acceptance test.

The initial investigation does not:

- select, install, or integrate a graph backend or memory system;
- graph the full repository or define a full scientific ontology;
- change Julia/Python package behavior, APIs, dependencies, persisted
  scientific schemas, or package version;
- create a second task/status system;
- ingest or publish raw private conversations;
- infer owner decisions or automatically arbitrate scientific disputes;
- make any derived graph, index, ledger, or context irreplaceable; or
- start fuzzy/Table-1 testing or modify CYAX-0152 private-memory infrastructure.

## Compatibility, version impact, and remaining approval work

Research/prototype compatibility impact is **none**. No package behavior,
dependency, public API, persisted schema, or package version changes under this
draft.

Before production adoption, obtain repository-owner and independent
architecture/evidence review of a revised specification and decide whether a
separate implementation work item under #162 should begin. The next owner gate
is not a backend selection or implementation start.
