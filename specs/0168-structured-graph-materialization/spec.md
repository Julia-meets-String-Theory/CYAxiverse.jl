---
spec_id: CYAX-0168
title: Structured versus graph provenance materialization benchmark
issue: 168
class: S2
status: draft
workstream: Infrastructure
parent: 162
depends_on: [163, 117]
created: 2026-09-13
last_reviewed: 2026-09-13
review_required: independent architecture/methodology review and repository-owner approval before benchmark execution
approval_ref: null
---

# CYAX-0168 — Structured versus graph provenance materialization benchmark

## Objective

Determine whether one embedded property-graph materialization provides a
material operational advantage over indexed SQLite for graph-native
CYAxiverse provenance retrieval when both views derive from the same immutable,
authority-safe assertion snapshot.

This specification freezes a systems/retrieval experiment. It does not
authorize benchmark execution, fresh-agent experiments, or production adoption.
The design branch starts from integrated `vmm` revision
`7a40285bb5c313f7e8746b90644d5f45bb67be44`.

## Motivation and claim boundary

CYAX-0163 established that canonical sources and reviewed typed assertions must
remain upstream of disposable retrieval structures. Its controlled comparisons
did not establish that a graph database is smaller, faster, more context
efficient, or more correct than an equally structured relational baseline.

This benchmark may establish backend-specific correctness and operational
measurements for the frozen fixtures, workload, software versions, and host
class. It must not establish that graph storage is authoritative, that graph
topology creates truth, that one backend improves fresh-agent correctness, or
that a production backend should be adopted.

## Authority and architecture boundary

The approved CYAX-0163 architecture governs:

```text
canonical sources
  → source revisions and fingerprints
  → extracted and curated typed assertions
  → immutable assertion snapshot
  → SQLite view S and property-graph view G
  → shared semantic evaluator and RetrievalBundle
  → one context compiler
```

Canonical GitHub Issues and PRs, approved specifications, owner decisions, code
revisions, and verification artifacts retain their existing authority. The
snapshot is the semantic comparison boundary. S and G are derived, disposable,
and rebuildable. Neither backend may independently store an editable
`authoritative`, `current`, `verified`, `superseded`, or `disputed` property.

Graph paths, centrality, similarity, communities, embeddings, optimizer scores,
and inferred edges are retrieval diagnostics only. They cannot create or alter
an assertion. Large source text remains in canonical upstream artifacts; the
views store stable IDs, compact relations, literals needed by the workload, and
source references.

## Common typed model

### Entity

The closed initial `entity_type` vocabulary is:

`WorkItem`, `Decision`, `Specification`, `Requirement`, `Implementation`,
`Verification`, `Claim`, `Artifact`, and `Source`.

Each entity has:

- `entity_id`: stable non-empty string;
- `entity_type`: one value from the closed vocabulary;
- `display_label_ref`: optional reference to a snapshot literal for diagnostics,
  never an authority signal.

No general scientific ontology is introduced.

### SourceRevision

Each source revision has exactly:

- `source_revision_id`;
- `source_id` referencing a `Source` entity;
- `source_kind` from a frozen allowlist;
- `canonical_locator`, using a public durable reference or repository-relative
  path and never a machine-local locator;
- `fingerprint_algorithm = "sha256"`;
- `fingerprint`;
- `observed_at`, an RFC 3339 UTC observation time;
- `authority_class`, from the CYAX-0163 authority vocabulary.

### Assertion

Each assertion has exactly:

- `assertion_id`;
- `subject_id`;
- `predicate`;
- exactly one of `object_id` or `literal_ref`;
- `source_revision_id`;
- `source_locator`, an anchor within that revision;
- `recorded_at`, an RFC 3339 UTC time;
- optional `valid_from` and `valid_to` domain-validity times;
- `origin`;
- `curation_state`;
- `review_state`;
- `epistemic_state`;
- `dispute_state`.

The closed predicate vocabulary is `governs`, `requires`, `implements`,
`verifies`, `supports`, `contradicts`, `supersedes`, `depends_on`,
`derived_from`, `concerns`, and `documented_in`. Direction is fixed by the
snapshot schema; adapters may traverse in reverse but may not mint inverse
assertions. Vocabulary expansion requires an amended reviewed design.

The approved CYAX-0163 admissibility rule applies: a relationship can affect
derived disposition only when its curation state is `curator_checked` or
`independently_reviewed` and its epistemic state is not `extracted`,
`unresolved`, or `rejected`. Candidate, superseded, disputed, resolved,
contradicted, and dependency-stale records remain inspectable but cannot alter
their targets. Required-dependency staleness follows explicit `depends_on`
assertions only.

### Shared semantic evaluator

Backend queries return candidate assertion IDs and path membership. One frozen
backend-independent evaluator then loads the common assertion objects, applies
the CYAX-0163 authority, admissibility, validity, supersession, dispute, and
staleness rules, and emits the semantic payload. Backend-native flags or scores
may appear only under `diagnostics`. This prevents a traversal implementation
from becoming a second truth engine.

Natural-language entity resolution is not measured. Query parameters are frozen
entity IDs. Any later lexical resolver must be shared and executed before the S
or G adapter is selected.

## Immutable assertion snapshot

A snapshot is a read-only directory with:

```text
snapshot/
  manifest.json
  entities.jsonl
  literals.jsonl
  source_revisions.jsonl
  assertions.jsonl
```

JSON uses UTF-8, Unicode NFC, sorted object keys, no insignificant whitespace,
one LF-terminated object per line, lexicographic record order by primary ID,
RFC 3339 UTC timestamps, and lowercase hexadecimal SHA-256 values. Arrays whose
order has no semantic meaning are sorted. Floating-point values are forbidden
in identity-bearing fields.

`manifest.json` records:

- `schema_version`;
- `snapshot_id`;
- `source_bundle_id`;
- each payload file's byte count and SHA-256;
- the ordered source-revision IDs and fingerprints;
- `authority_rule_version`;
- `assertion_compiler_version` and `curator_version`;
- entity/assertion/source/literal counts;
- `semantic_checksum_algorithm` and `semantic_checksum`.

The semantic checksum is SHA-256 over the length-prefixed canonical bytes of
the four payload files in the order above plus the schema, authority-rule,
compiler, and curator versions. `snapshot_id` is
`cyax-snapshot-sha256:<semantic_checksum>`.

Every materialization embeds the full `snapshot_id`, semantic checksum, schema
version, authority-rule version, materializer name/version, and build-complete
marker. Open and retrieval fail closed if any identity differs, a source payload
hash fails, or the build marker is absent. Rebuilding either view from the same
snapshot must reproduce the same canonical semantic export checksum; database
file byte equality is not required.

## Structured baseline S

S uses the Python standard-library SQLite binding and records the exact SQLite
library version and compile options. It uses normalized `entity`, `literal`,
`source_revision`, `assertion`, and `materialization_metadata` tables with
foreign keys and CHECK constraints matching the snapshot schema.

B-tree indexes cover at least:

- entity type and entity ID;
- assertion subject/predicate/object;
- assertion object/predicate/subject for reverse traversal;
- source revision and source locator;
- predicate plus validity interval;
- supersession, dispute, and dependency lookup paths used by the workload.

Recursive CTEs implement bounded path and transitive-closure queries with
explicit cycle prevention and a preregistered maximum depth. FTS5 is permitted
only for a separately labelled lexical lookup step; it is not used in the
frozen ID-addressed workload. JSONL scanning is not an admissible S control.
Journal/checkpoint configuration is frozen, and steady-state WAL is measured
after an explicit checkpoint.

## Graph benchmark G and smoke gate

The primary candidate is LadybugDB Python package `ladybug==0.20.4`, upstream
tag `v0.20.4` at abbreviated release commit `df58ee3`, released 2026-09-10 under
the MIT license. The official project describes it as embedded/serverless and
documents the Python API. Sources: [release](https://github.com/LadybugDB/ladybug/releases/tag/v0.20.4),
[repository](https://github.com/LadybugDB/ladybug), and
[installation guide](https://docs.ladybugdb.com/installation/).

Preliminary bounded smoke evidence on 2026-09-13:

- environment: macOS 26.6.2 ARM64, CPython 3.14.6;
- wheel: `ladybug-0.20.4-cp314-cp314-macosx_15_0_arm64.whl`;
- wheel SHA-256:
  `7a36d5b051ddc954d7ee5d5fa6165fb49d48a4785a132bafdd311723897fa649`;
- the wheel and optional result-format dependencies installed successfully from
  a local wheelhouse with `PIP_NO_INDEX=1`;
- the Python API created an on-disk graph and executed a three-hop path query;
- two clean builds produced identical canonical result checksum
  `1484e01107687c0d9e10fdf7b97fc49938b0fd85264ac42fa4810f65d311e0da`.

This confirms the requested local smoke gate only for that environment. Before
benchmark execution, the execution host must repeat the pin/hash, import,
offline-install, clean-build, reopen, query, and deterministic semantic-export
checks. Extensions that download artifacts at runtime are disabled.

G uses typed entity nodes and assertion relationships/properties sufficient to
recover the original assertion IDs. It must not duplicate large source text or
store mutable semantic disposition flags.

FalkorDBLite is the sole predeclared fallback. It may replace Ladybug only if
the repeated frozen smoke gate fails. Substitution requires recording the exact
failure, pin, artifact hash, license, offline behavior, and an amended design
review before measurement. Neo4j, TypeDB, Oxigraph, archived Kùzu upstream, and
other graph systems are outside this experiment.

## Fixtures

### F-real — Issue #117 semantic regression

Use Issue #117 as a real CYAxiverse semantic regression fixture. It is not
fully held out. Freeze the same canonical evidence boundary identified by the
CYAX-0166 draft at PR #167 head
`a5f53ad1146298ab645ff3a6ecca63742f63afb4`: Issue #117 and owner decision,
merged PRs #101 and #104, closed/unmerged PR #103 and its supersession record,
Issue #112, and named repository artifacts at one integrated revision.

Retain the K1–K12 semantics unless source review finds an error:

1. #117 remains an open bounded follow-up.
2. Its approved evidence boundary is narrow ordinary-Euler evidence from
   smooth, unimodular local cones, not a population claim.
3. Issue #117 plus its durable owner decision governs; no standalone approved
   CYAX-0117 feature spec exists in the frozen sources.
4. Bounded h11=4/h11=5 replay is permitted; broader orbifold/stringy-Euler or
   non-simplicial mathematics is not approved.
5. PR #101 merged the projection ledger and independently reproduced its digest
   without changing scientific code.
6. PR #103 is closed/unmerged and superseded by PR #104; its reused work is not
   the #117 point-certificate task.
7. The ledger contains 1,146 rows with projection SHA-256
   `ebe14b02d312993fd85ce98b2aa882701b1c14f6aff25065e5d566a2f2ad504b`.
8. `_derive_zero_dimensional_local_evidence` exists and fails closed, but lacks
   a focused direct test and completed bounded repaired replay in the freeze.
9. Broader non-simplicial/orbifold certificate mathematics remains separate and
   unresolved in #112.
10. CYTools availability, direct producer coverage, and bounded replay evidence
    remain execution risks; unavailable cases cannot be promoted.
11. The next permissible action is a focused direct producer test followed by
    bounded h11=4/h11=5 replay under the narrow contract and normal gates.
12. The result must abstain from unestablished population,
    orbifold/stringy-Euler, or broader scientific conclusions.

Each item must be compiled into expected entities/assertions and exact source
anchors, then independently reviewed. Any discovered error amends both this
fixture and CYAX-0166's copy before execution.

### F-scale — deterministic inert generator

Generator version `cyax-0168-scale-1.0` creates only project-like records. A
work item contains decision → requirement → implementation → verification →
claim motifs plus controlled dependencies, reverse dependencies, supersession,
disputes, cross-work-item relations, branching evidence, and source revisions.
It creates no physics or scientific conclusion.

Frozen tiers are:

| Tier | Entities | Assertions | Seeds |
| --- | ---: | ---: | --- |
| T0 | 1,000 | 5,000 | `162000` |
| T1 | 10,000 | 50,000 | `162011`, `162012`, `162013` |
| T2 | 100,000 | 500,000 | `162021`, `162022`, `162023` |
| T3 | 200,000 | 1,000,000 | `162031`, `162032`, `162033` |
| T4 | 1,000,000 | 5,000,000 | `162041`, `162042`, `162043` |

For the three decision-relevant seeds at each tier, freeze respectively
`(path depth, dependency fan-out, supersession length, cross-link rate)` as
`(4,2,2,0.01)`, `(8,8,8,0.05)`, and `(16,32,32,0.20)`. The generator must hit
the exact counts or fail, emit its parameters and checksum, and avoid
uncontrolled wall-clock or random identifiers. T4 runs only if both backends
complete every T3 correctness and resource gate.

## Frozen query workload and RetrievalBundle

The twelve semantic queries are:

| ID | Class | Question / required operation |
| --- | --- | --- |
| Q01 | point lookup | What governs entity X? |
| Q02 | neighborhood | Why is X authoritative? |
| Q03 | path | Reconstruct decision → requirement → implementation → verification for X. |
| Q04 | temporal/as-of | Which assertions/entities for X were superseded as of T? |
| Q05 | neighborhood | What directly depends on X? |
| Q06 | transitive closure | What transitively depends on X within frozen depth D? |
| Q07 | transitive closure | What becomes stale if X is superseded? |
| Q08 | neighborhood | Which disputes concern X? |
| Q09 | path | Trace claim X to verification/source evidence. |
| Q10 | context selection | Return the deterministic minimal evidence set for query Q. |
| Q11 | temporal/as-of | What is the next permissible action for work item X as of T? |
| Q12 | neighborhood | Which explicit artifacts implement or verify requirement R? |

The fixture manifest freezes every query's entity IDs, `as_of`, maximum depth,
tie-break order, expected entity/assertion/source-revision sets, and minimal-set
rule. Q10 minimizes assertion count first, then source-revision count, then uses
lexicographic assertion IDs; equal alternatives remain reported explicitly.
Parser-derived Julia symbol topology is excluded.

Both adapters return:

```text
RetrievalBundle
  contract_version
  query_id
  snapshot_id
  as_of
  assertions[]
  source_revisions[]
  path_refs[]
  diagnostics
```

Assertions and source revisions are the canonical snapshot objects sorted by
ID. A `path_ref` contains ordered assertion IDs only. Diagnostics may include
backend name/version, timing, depth, backend score, candidate count, path cost,
and truncation, but have no authority semantics. Contract serialization uses
the same canonical JSON rules as the snapshot.

## Correctness gate

Correctness precedes performance. For every deterministic query and fixture,
require exact expected entity/assertion sets, provenance completeness,
authority correctness, supersession correctness, dispute correctness, temporal
correctness, stable ordering, and snapshot identity. Compare the semantic
portion of S and G bundles byte-for-byte after diagnostics are removed.

Any mismatch is an architecture failure. Stop that tier, preserve the minimal
failing fixture, and do not report a backend as faster for the mismatched query.
Performance results are admissible only after both backends pass the full
semantic gate for that fixture.

## Measurement protocol

Freeze the host hardware, OS, filesystem, Python, SQLite, Ladybug, dependency,
CPU-governor/power, and process-affinity facts that can be controlled. Run with
no network access. Use one process per backend trial, alternate the backend that
runs first using a seeded balanced schedule, and report raw samples plus the
aggregation script. Disable diagnostics logging that is not common to both.

### Storage and build

For each snapshot/backend measure:

- compressed and uncompressed common snapshot size;
- database data, indexes, and steady-state WAL/checkpoint bytes separately;
- total system footprint and incremental materialization footprint;
- temporary peak build disk;
- clean build and second clean rebuild wall/CPU time;
- peak build RSS;
- canonical semantic-export checksum after each rebuild.

### Updates

Freeze deterministic batches for one assertion, 100 assertions, 1% of the
corpus, source-revision replacement, supersession, and dependency
insertion/removal. Each batch starts from the same clean base. Measure wall/CPU
time, peak RSS, changed disk bytes, whether full rebuild is required, and the
resulting semantic checksum. Run ten measured repetitions per batch after two
unmeasured warmups where the backend supports incremental update; otherwise
measure and label the required rebuild.

### Runtime memory and query latency

Measure idle/open RSS, peak build RSS, cold first-query RSS, warm-query RSS, and
peak deep-query RSS as process deltas and absolute peaks. For every admissible
query/fixture pair run ten cold trials in new processes and fifty warm measured
trials after five warmups. Record wall and CPU time, warm p50 and nearest-rank
p95, result-set size, candidate count, and adapter-boundary bytes. Do not pool
different tiers, seeds, topology profiles, or query families.

### Context cost

Pass each semantic bundle through the exact same frozen context compiler and
tokenizer. Record assertions selected, source revisions selected, backend
candidate bytes, canonical `RetrievalBundle` bytes, rendered UTF-8 context
bytes, prompt tokens, and truncation events. Database latency and token savings
are separate results.

The design proposes a T3 envelope of 30 minutes clean build time, 16 GiB peak
RSS, and 50 GiB temporary disk per backend/seed. The owner must approve or amend
this envelope before execution. T4 additionally requires both systems to pass
all T3 runs within the envelope; its final envelope is an open owner decision.

## Proposed decision thresholds

These thresholds are preregistered proposals, not owner-approved decisions.
All comparisons use paired fixture/query samples and report absolute values,
ratios, and bootstrap 95% confidence intervals; a threshold is met only when
the point estimate and confidence interval support the stated direction.

- **S wins:** G does not meet the G or Hybrid threshold and S is comparable or
  better across the real workload.
- **G wins:** G achieves at least 2× lower warm p95 latency on at least two of
  the preregistered graph-native families (`path`, `transitive closure`,
  `context selection`) at T2 and T3, including at least one family on F-real
  where applicable, while maintaining semantic parity, no more than 2× S's
  incremental materialization footprint, no more than 2× S's peak query RSS,
  and no pathological build/update cost under the approved envelope.
- **Hybrid wins:** G meets that material threshold for at least two
  graph-native families while S is at least 2× better on at least one of
  `point lookup`, `temporal/as-of`, or any separately admitted lexical family,
  with the same correctness and resource gates. Recommend explicit static
  query-class routing only.
- **Architecture problem:** semantic parity cannot be maintained, snapshot
  identity cannot fail closed, deterministic semantic rebuild fails, or the
  shared assertion/context contract proves defective.

A reduction in retrieved candidates or canonical bundle/context bytes may
substitute for latency only if it is at least 2× on two graph-native families,
preserves the exact minimal semantic payload, and does not increase prompt
tokens or truncation. The final report must also show results that miss the
threshold; it may not reduce the decision to a winner label.

## Requirements

| Requirement | Required behavior | Verification gate |
| --- | --- | --- |
| R-001 Authority-safe common model | Both views derive only from one typed assertion snapshot; semantic disposition is computed by the shared CYAX-0163 evaluator. | Schema/validator tests and adversarial candidate/supersession/dispute fixtures. |
| R-002 Immutable identity | Snapshot and materializations carry exact identities and fail closed on mismatch or incomplete build. | Tamper, stale-view, partial-build, and deterministic-export tests. |
| R-003 Fair S baseline | Use normalized indexed SQLite, recursive CTEs, and FTS5 only when lexical lookup is required. | Schema/index/query-plan review and prohibition of JSONL scan control. |
| R-004 Bounded G candidate | Use the pinned embedded Ladybug candidate after the repeated smoke gate, or the reviewed predeclared fallback only. | Release/hash/license/offline/import/rebuild/reopen/query evidence. |
| R-005 Semantic fixture | Compile #117 and reviewed K1–K12 semantics into expected common records. | Independent source/answer-key review and exact S/G semantic parity. |
| R-006 Deterministic scale | Generate exact inert tiers and frozen topology profiles from versioned seeds. | Repeat-generation byte/checksum equality and invariant tests. |
| R-007 Shared retrieval | Execute Q01–Q12 with frozen IDs/parameters and return one canonical `RetrievalBundle`. | Contract validation, stable ordering, and diagnostics-stripped byte equality. |
| R-008 Correctness first | Admit performance only after authority, provenance, temporal, supersession, dispute, and expected-set gates pass. | Machine-checked golden results and mismatch stop test. |
| R-009 Reproducible measurements | Measure storage, rebuild, updates, RSS, latency, adapter bytes, and context cost under randomized paired execution. | Raw-sample manifest, environment record, deterministic analysis, independent methodology review. |
| R-010 Decision discipline | Apply only the approved S/G/Hybrid/Architecture thresholds without automatic production adoption. | Final report maps every conclusion to frozen thresholds and returns to owner. |

## Acceptance and stop gates

### G0 — Design approval

**Acceptance:** Independent architecture/methodology review passes and repository
owner approval cites the exact design head, resource envelope, backend pin,
workload, repetitions, and decision thresholds.

**Stop:** Any unresolved semantic or measurement choice, `approval_ref: null`,
or a source/backend release change.

### G1 — Frozen inputs and backend smoke

**Acceptance:** F-real, F-scale generator, queries, answer keys, snapshot,
environment, dependencies, and hashes are frozen; S and G pass offline clean
build/rebuild/reopen/tamper smoke tests.

**Stop:** Source asymmetry, private locator, runtime download, nondeterministic
semantic export, or materialization identity failure.

### G2 — Semantic parity

**Acceptance:** S and G pass every correctness criterion on F-real and each
scale fixture before timing for that fixture is admitted.

**Stop:** Any semantic, authority, provenance, temporal, supersession, dispute,
or expected-set mismatch. Classify as Architecture problem.

### G3 — T0 through T3 systems benchmark

**Acceptance:** The randomized protocol completes and all preregistered raw and
aggregate metrics validate within the approved envelope.

**Stop:** Resource breach, corrupted/missing sample, uncontrolled environment,
or analysis deviation. Do not proceed to T4.

### G4 — Conditional T4 and decision

**Acceptance:** Only if both backends passed all T3 gates, run T4 under its
approved envelope, then apply the frozen classifier and report limitations.

**Stop:** Any earlier failure or unapproved threshold/envelope amendment.

## Interfaces, compatibility, privacy, and version impact

All benchmark code and databases are experiment-only. No package import path,
public API, scientific behavior, persisted scientific schema, or package
dependency changes are authorized. Optional Python environments and generated
databases remain outside the Julia package runtime. The package version impact
is none.

Public artifacts use repository-relative paths and durable public references.
They contain no private transcripts, attachment locations, local usernames,
home paths, hostnames, machine identifiers, credentials, or session/task IDs.
Environment data is reported at a non-identifying hardware/OS/tool granularity.

## Open owner decisions

Before execution, the repository owner must approve or amend:

1. the proposed 2× performance/context threshold and two-family rule;
2. the 2× footprint and query-RSS ceilings and the meaning of “pathological”
   build/update cost;
3. the proposed T3 resource envelope and a separate conditional T4 envelope;
4. the exact execution host class and whether CPU affinity/power controls are
   required;
5. Ladybug 0.20.4 as the benchmark candidate after the execution-host smoke
   rerun;
6. the repetitions, confidence-interval method, and F-real applicability rule;
7. whether the CYAX-0166 K1–K12 draft is stable enough to share as this
   fixture's known-answer boundary.

## Non-scope

- fresh-agent cohorts, agent-correctness claims, or transcript ingestion;
- production graph integration, automatic backend selection, or a
  backend-selection epic;
- vector search, embeddings, or parser-derived Julia code topology;
- a general project or scientific ontology;
- inference of owner decisions or automatic dispute arbitration;
- weakening provenance/history, replacing GitHub/approved specifications, or
  making a materialization authoritative;
- scientific/package behavior, API, dependency, schema, or version changes.

## Completion criterion

This S2 work item completes only after a separately authorized execution passes
G0–G3 (and G4 if eligible), publishes replayable semantic and systems evidence,
classifies the outcome as S wins, G wins, Hybrid wins, or Architecture problem,
and returns the result to #162 without adopting a production backend. A valid
negative result is complete.
