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
last_reviewed: 2026-09-12
review_required: repository-owner review and independent architecture/evidence review before production adoption
approval_ref: null
---

# CYAX-0163 — Temporal provenance for durable agent memory

## Objective

Determine whether a small provenance-aware temporal layer helps a genuinely
fresh agent reconstruct authoritative CYAxiverse state from durable sources
more reliably and with less context than direct reconstruction from
unstructured historical artifacts.

The graph, index, ledger, and generated context are derived and disposable.
Canonical GitHub and repository artifacts remain authoritative.

## Status and approval boundary

This draft formalizes the research contract already recorded in Issue #163. It
does not approve a graph backend, production memory service, automatic
extraction pipeline, scientific ontology, or change to repository authority.
The backend-independent research prototype may collect evidence while this
specification is draft because it changes no package, scientific, or production
memory behavior. Production adoption requires a reviewed specification update
and repository-owner approval.

## Requirements and evidence gates

| Requirement | Required behavior | Initial evidence gate |
| --- | --- | --- |
| R-001 Authority boundary | Canonical artifacts keep their existing authority and scope. Every derived material relationship cites a canonical source and anchor. Retrieval order, frequency, recency, and confidence do not increase authority. | Validator rejects assertions without canonical provenance; generated context labels itself disposable. |
| R-002 Minimal semantic model | Use the smallest backend-independent entity and assertion vocabulary that can represent a selected pilot. Do not create a repository-wide or scientific ontology. | Inspectable versioned records, schema, validator, and generated view. |
| R-003 Time and state | Keep domain validity, observation/recording time, epistemic status, source authority, supersession, contradiction/dispute, and retrieval salience distinct. Preserve historical state rather than overwrite it. | Focused tests cover time shape, supersession, disputes, required dependency staleness, and forbidden stored salience. |
| R-004 Safety and privacy | Do not ingest private transcripts or machine-local locators. State the multi-writer safety boundary. Fail closed on uncertain publication. | Diff inspection and validator/tests; explicit prototype limitations. |
| R-005 Clean pilot | Select one bounded work item with a known answer and a decision/work-item/requirement/implementation/verification/outcome chain. Do not use the fuzzy/Table-1 history first. | A frozen pilot ledger and canonical source manifest. |
| R-006 Cold-start comparison | Give comparable fresh agents the same reconstruction task. Compare an unstructured source condition with a derived-context condition. Record sources opened, context consumed, dimension scores, and automatic failures. | Frozen responses and scorecards against a pre-registered rubric. |
| R-007 Adversarial repair | Classify the first run's failures, repair the infrastructure rather than coach the same agent, and run a second fresh reconstruction. | Run-01 failure classification, repaired ledger/tests, and run-02 response. |
| R-008 Decision gate | Select no graph backend unless evidence shows material reconstruction benefit that a concise non-graph summary cannot explain. | Reliability/efficiency threshold followed by a structured-summary ablation. |
| R-009 Scientific stress boundary | Start a complex scientific pilot only after the clean pilot and ablation gates pass. Do not allow one extractor interpretation to become authoritative. | Separate reviewed slice with frozen sources, conflicting interpretations, negative evidence, and explicit abstention. |

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

Graphiti/Zep, NornicDB/Roynard, Mem0, GraphRAG, Hindsight, and Flow are reference
systems, not authority sources or selected dependencies. Flow is directly
relevant as an integrated coding-memory implementation. Its revision-pinned
source reads, evidence cards, mutation journal, project isolation, and safe
document revisions merit further comparison. Its code-centric ontology,
in-place graph/memory updates, hard deletion, and recency-based memory sinking
do not satisfy this specification's durable-authority boundary.

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

## Compatibility and version impact

None for the research/prototype stage.

## Remaining approval and evidence work

- Obtain repository-owner and independent review of this draft before any
  production adoption.
- Run an equally concise non-graph structured-summary ablation on a held-out
  frozen source snapshot.
- If that gate passes, define a separate adversarial or scientific pilot and
  its approval boundary.
- Evaluate storage backends only after these experiments.
