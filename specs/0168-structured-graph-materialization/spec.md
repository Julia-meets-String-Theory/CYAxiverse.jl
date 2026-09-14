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
review_required: independent architecture/methodology review and repository-owner approval before benchmark implementation or execution
approval_ref: null
---

# CYAX-0168 — Structured versus graph provenance materialization benchmark

## Objective and authorization boundary

Determine whether an embedded property-graph materialization provides a
material operational advantage over indexed SQLite for graph-shaped CYAxiverse
provenance retrieval when both views derive from the same immutable,
authority-safe assertion snapshot.

The prior exact-head rereview returned **PASS WITH REQUIRED REVISIONS**; this
revision makes those bounded R1–R9 repairs. The independent #117 K1–K12 source
audit passed and its chronology is preserved. This revision does not approve
the specification, authorize benchmark implementation or execution, revise
CYAX-0166/CYAX-0167, or select a production backend. The design branch starts
from integrated `vmm` revision
`7a40285bb5c313f7e8746b90644d5f45bb67be44`.

The benchmark may establish correctness and operational measurements only for
the frozen fixtures, workload, software versions, and host class. It cannot
establish that graph storage is authoritative, that topology creates truth,
that one backend improves fresh-agent correctness, or that a production
backend should be adopted.

## Governing architecture

The approved CYAX-0163 boundary is invariant:

```text
canonical sources
        ↓
immutable source revisions / fingerprints
        ↓
typed provenance assertions
        ↓
immutable assertion snapshot
        ↓
   ┌────┴────┐
   ↓         ↓
 SQLite    graph
   ↓         ↓
   └────┬────┘
        ↓
 shared semantic evaluator
        ↓
  RetrievalBundle
        ↓
 one bounded context compiler
```

Canonical GitHub and repository sources keep their existing authority. SQLite
and graph materializations are disposable and rebuildable. Neither backend may
independently define or edit authority, currentness, verification,
supersession, dispute, validity, or epistemic state. Backend-native paths,
scores, communities, embeddings, and inferred edges are diagnostics only.

The only later interface exposed to CYAX-0166 is the selected `snapshot_id`,
canonical `RetrievalBundle`, shared semantic evaluator, and bounded context
compiler. CYAX-0166 and its PR are not changed by this repair.

## Common typed model

### Entity and Claim content

The closed `entity_type` vocabulary is `WorkItem`, `Decision`,
`Specification`, `Requirement`, `Implementation`, `Verification`, `Claim`,
`Artifact`, and `Source`.

Every entity has `entity_id`, `entity_type`, and nullable
`display_label_ref`. The display label is diagnostic and is excluded from
semantic comparison. It must never contain the proposition represented by a
`Claim`.

A `Claim` additionally has `claim_key`, a **registered fixture-local semantic
key** such as `K07.total_rows`. Before either backend is implemented, each
fixture manifest freezes a `claim_key_registry` sorted by `claim_key`. Every
entry contains exactly `claim_key`, `literal_type`, and `semantic_slot`, where
`semantic_slot` is a stable identifier/meaning, not prose to parse. Duplicate
or unregistered keys fail validation, and the Claim literal's type must equal
the registered `literal_type`.

This field is absent on every other entity type. Exactly one Claim
`documented_in` assertion with a `literal_ref` states the content, and at least
one `concerns` assertion identifies its subject. Relationships to evidence also
use typed assertions. This narrow mechanism represents work-item state, exact
scalars, digests, and K1–K12 proposition content without placing semantics in a
label or inventing backend-specific properties.

### Literal

Each literal has `literal_id`, `literal_type`, and `value`. The closed literal
types and canonical values are:

| Type | Canonical value | Comparison |
| --- | --- | --- |
| `text` | NFC Unicode string with no CR or NUL | exact Unicode scalar sequence |
| `integer` | arbitrary-precision base-10 integer; JSON number, no leading zero | mathematical integer equality |
| `boolean` | JSON `true` or `false` | exact Boolean equality |
| `timestamp` | UTC timestamp in the format below | instant equality |
| `sha256` | 64 lowercase hexadecimal characters | exact digest equality |
| `work_item_state` | `open` or `closed` | exact token equality |
| `pull_request_state` | `draft_open`, `open`, `closed_unmerged`, or `merged` | exact token equality |
| `review_verdict` | `pending`, `pass`, `changes_required`, or `rejected` | exact token equality |

Floating point, binary blobs, maps, arrays, backend values, and unregistered
enum strings are forbidden. `text` is permitted only when the Claim registry
explicitly assigns `text`, or for diagnostic display labels. Text is never
parsed to derive authority, state, predicate behavior, ranking, or any other
ontology semantics.
Every semantic literal is stored canonically in the snapshot and linked through
its Claim and source-provenanced assertion to exact captured source evidence.
`literal_ref` in an assertion is not a generic escape hatch: it is permitted
only where the predicate-signature table says so, and neither backend may
replace an entity relation with a literal property.

### Frozen vocabularies

All values are lowercase ASCII tokens. Unknown values fail validation.

| Field | Allowed values and semantics | Provenance and transitions |
| --- | --- | --- |
| `source_kind` | `github_issue`, `github_issue_comment`, `github_pull_request`, `github_pull_request_comment`, `github_review`, `git_commit`, `repository_file`, `verification_artifact`, `external_document`, `synthetic_fixture` | Source-derived from the captured object type; `synthetic_fixture` is reserved for generator-owned canonical bytes and may never classify real evidence. Immutable for a source revision. |
| `authority_class` | `owner_decision`, `approved_specification`, `canonical_work_item`, `merged_implementation`, `verification_evidence`, `external_reference`, `ordinary_record`, `agent_proposal` | Curator-supplied by the mechanical derivation table below and independently reviewable; immutable in a snapshot. Reclassification creates a successor snapshot. |
| `origin` | `source_direct`, `curator_interpretation`, `rule_derived` | Assertion compiler supplies the value. `source_direct` is explicitly stated; `curator_interpretation` is bounded human interpretation; `rule_derived` names a frozen evaluator rule and supporting assertions. Immutable; correction creates a new assertion in a successor snapshot. |
| `curation_state` | `unreviewed`, `curator_checked`, `independently_reviewed` | Curator-supplied. Allowed forward transitions are `unreviewed → curator_checked → independently_reviewed`; a failed check creates a corrected/rejected successor assertion, never a backward mutation. |
| `review_state` | `not_required`, `pending`, `passed`, `changes_required`, `rejected` | Source-derived when a canonical review record exists, otherwise curator-supplied. `pending → passed|changes_required|rejected`; `changes_required → pending` only on a new source revision. `not_required` is terminal for that assertion revision. |
| `epistemic_state` | `extracted`, `unresolved`, `supported`, `verified`, `accepted`, `rejected` | Curator-supplied from source evidence; `accepted` requires explicit governing approval. Allowed forward transitions are `extracted → unresolved|supported|rejected`, `unresolved → supported|rejected`, `supported → verified|rejected`, and `verified → accepted|rejected`. Each transition is a successor assertion, not mutation. |
| `dispute_state` | `undisputed`, `disputed`, `resolved_upheld`, `resolved_rejected` | Evaluator-derived from admissible `contradicts`, `supports`, and resolution evidence. `undisputed → disputed → resolved_upheld|resolved_rejected`; new contrary evidence creates a successor assertion and may yield `disputed` again in the successor snapshot. |

Only `curator_checked` or `independently_reviewed` assertions whose epistemic
state is `supported`, `verified`, or `accepted` can affect semantic disposition.
`review_state=passed` is additionally required when review is required by the
governing source. Candidate, rejected, superseded, disputed, and
dependency-stale assertions remain inspectable but cannot alter their targets.
Required-dependency staleness follows explicit `depends_on` assertions only.

### Predicate signatures

Direction is the stored subject-to-object direction. Reverse traversal returns
the same assertion ID with `direction=reverse`; it never creates an inverse
assertion. `literal_ref` is allowed only for the single `documented_in` content
assertion from each Claim.

| Predicate | Allowed subject types | Allowed object types | Literal | Fixed meaning; reverse traversal |
| --- | --- | --- | --- | --- |
| `governs` | Decision, Specification, WorkItem | WorkItem, Specification, Requirement | no | subject sets the governing boundary for object; reverse asks what governs the object |
| `requires` | WorkItem, Decision, Specification, Requirement | Requirement, Implementation, Verification | no | subject makes object necessary; reverse asks what requires the object |
| `implements` | Implementation, Artifact | Requirement, Specification | no | subject realizes object; reverse asks implementations of object |
| `verifies` | Verification, Artifact | Claim, Requirement, Implementation | no | subject tests/evidences object; reverse asks verification for object |
| `supports` | Claim, Verification, Artifact, Source | Claim, Decision, Requirement | no | subject provides positive evidence for object; reverse asks support for object |
| `contradicts` | Claim, Verification, Artifact, Source | Claim, Decision, Requirement | no | subject provides contrary evidence for object; reverse asks contradictions of object |
| `supersedes` | WorkItem, Decision, Specification, Requirement, Implementation, Verification, Claim, Artifact, Source | same type as subject | no | subject replaces object from its effective time; reverse asks successors of object |
| `depends_on` | WorkItem, Requirement, Implementation, Verification, Claim, Artifact | WorkItem, Requirement, Implementation, Verification, Claim, Artifact | no | subject requires object to remain admissible/current; reverse asks dependants of object |
| `derived_from` | Decision, Requirement, Implementation, Verification, Claim, Artifact, Source | Decision, Requirement, Implementation, Verification, Claim, Artifact, Source | no | subject was produced from object; reverse asks derivatives of object |
| `concerns` | WorkItem, Decision, Specification, Requirement, Implementation, Verification, Claim, Artifact | WorkItem, Decision, Specification, Requirement, Implementation, Verification, Claim, Artifact | no | subject is explicitly about object; reverse asks records concerning object |
| `documented_in` | WorkItem, Decision, Specification, Requirement, Implementation, Verification, Claim, Artifact | Source | no | subject is documented by the source entity; reverse asks content documented by source |
| `documented_in` | Claim | — | yes | the Claim's exact proposition is the referenced literal; literal traversal has no inverse entity |

F-real must use only this vocabulary. Any required extra predicate stops the
design and requires an amended independent review.

### Assertion

Each assertion has exactly:

- `assertion_id`, `subject_id`, and one frozen `predicate`;
- exactly one of `object_id` or `literal_ref`;
- `source_revision_id` and `source_locator` anchored within captured bytes;
- `source_event_at`, nullable when the source provides no event time;
- `asserted_at`, the deterministic provenance time for creation of this
  immutable assertion revision;
- nullable `valid_from` and `valid_to` domain/effective bounds;
- `validity_basis = explicit | source_event | unknown`;
- `authority_class` and `authority_derivation_rule_id`, copied from the
  validated mechanical authority derivation for this assertion;
- `origin`, `curation_state`, `review_state`, `epistemic_state`, and
  `dispute_state` from the frozen vocabularies.

`asserted_at` equals `source_event_at` for `source_direct` when present and
otherwise the source revision's captured `observed_at`; the captured curator
action time for `curator_interpretation`; and the maximum `asserted_at` of the
explicit supporting assertion IDs for `rule_derived`. Empty rule support is
invalid. It is never compiler/build wall time. Actual wall time is nullable
`built_at` in nonsemantic manifest/build metadata and never affects semantic
evaluation or a stable ID.

`source_locator` uses a source-kind-specific stable anchor: GitHub comment ID,
review ID, Issue/PR body plus captured revision, Git commit/path/blob/line
anchor, or verification artifact record key. Line numbers alone are
insufficient unless bound to immutable captured bytes.

## Authority derivation

`authority_class` is not inferred from prose. The source-bundle compiler applies
this ordered table and records the matched rule ID:

| Source evidence | Authority class |
| --- | --- |
| GitHub Issue/PR comment or review whose immutable actor ID and captured role evidence establish `repository_owner` **and** whose exact `(source_revision_id, source_locator)` is in the fixture's frozen owner-decision-event registry | `owner_decision` |
| `spec.md` with `status: approved` and non-null approval provenance that resolves to the captured approving owner/reviewer records | `approved_specification` |
| Captured Issue body/state or PR body/state | `canonical_work_item` |
| Commit reachable from the captured base branch whose associated PR is captured as merged | `merged_implementation` |
| Immutable test, ledger, report, or certificate with its producer/source fingerprints | `verification_evidence` |
| External paper/document with immutable edition/revision identity | `external_reference` |
| Ordinary Issue/PR comment, review, repository file, or commit not matching a higher rule | `ordinary_record` |
| Agent-authored proposal or generated design lacking the required approval record | `agent_proposal` |

Owner identity and decision-event identity are independent predicates. The
immutable `owner_decision_events.jsonl` registry is fixture-bound, sorted by
the framed `(source_revision_id, source_locator)`, and contains exactly those
two fields plus `registry_entry_id`; duplicates and references outside the
source bundle fail validation. The registry is hashed into the source bundle
and is independently reviewable. No display name, login, author association,
generic owner authorship, free-text parsing, or proximity to another event can
establish decision status.

For F-real the owner allowlist contains GitHub actor ID `102535039` for the
captured observation boundary, and comment `5556641428` is an explicit registry
entry bound to its exact captured comment revision and locator. The bundle
captures actor ID, login-at-observation, author association, and repository-role
evidence, but an ordinary acknowledgement, question, or proposal by that owner
remains `ordinary_record` unless its exact event is registered. The same text
from a non-owner cannot become an owner decision even if registered in error;
validation rejects it. Issue and PR state are source facts, not owner decisions.
A merged PR establishes merged implementation, not scientific acceptance.
Verification evidence establishes only what its frozen gate says. External
sources retain external authority. Agent proposals remain proposals until a
separately captured approval changes their class in a successor snapshot.

Authority conformance fixtures include an unregistered owner acknowledgement,
an unregistered owner question, an unregistered owner proposal, the exact
registered owner decision, and the same decision text from a non-owner. Only
the exact registered event with valid owner actor/role evidence may classify as
`owner_decision`.

## Temporal model

Four times remain distinct:

- domain/effective time: `valid_from`/`valid_to`, when the proposition holds;
- source event time: `source_event_at`, when the canonical source event occurred;
- observation time: `observed_at`, when source bytes/state were captured;
- assertion-revision provenance time: `asserted_at`, determined by the rule
  above.

These four temporal roles are conceptually distinct even when two or more
values coincide. `built_at` is a fifth, nonsemantic build-wall-time field.
Timestamps are UTC Gregorian instants serialized as
`YYYY-MM-DDTHH:MM:SS.ffffffZ`, exactly six fractional digits, with no leap
second. Intervals are half-open `[valid_from, valid_to)`. A null `valid_from`
means the domain start is unknown, not negative infinity; a null `valid_to`
means no captured end is known. `validity_basis=source_event` permits the
compiler to copy a captured event time into `valid_from` only for event-defined
facts such as Issue closure or PR merge. Rebuild/observation wall time is never
semantic time.

An as-of query at `T` includes a proposition only when the evaluator can prove
`valid_from ≤ T` and (`valid_to` is null or `T < valid_to`). An assertion with
unknown domain start is returned as provenance with temporal state `unknown`
but cannot satisfy a query requiring known state. Supersession becomes effective
at the admissible `supersedes` assertion's `valid_from`; before that instant the
predecessor remains current. Equal-time competing successors yield disputed or
ambiguous state and fail closed until an admissible ordering/resolution source
exists.

## Immutable source bundle

A content-addressed source bundle precedes the assertion snapshot:

```text
source-bundle/
  bundle_manifest.json
  source_revisions.jsonl
  actors.jsonl
  events.jsonl
  owner_decision_events.jsonl
  objects/sha256/<two-hex>/<remaining-hex>
```

The object store contains the exact captured bytes for every Issue/PR body,
comment/review, repository blob, verification artifact, external source
revision, and authority-relevant actor/role record used by F-real. A canonical
URL without captured bytes is invalid. `source_revisions.jsonl` freezes
`source_revision_id`, source entity, kind, canonical public locator, object
digest and byte count, source event time, observed time, source metadata,
anchors, actor ID, event/state fields, authority class, and derivation rule ID.
The manifest freezes schema/rule versions, ordered payload hashes/counts, the
repository observation boundary, every selected source revision, immutable
owner actor/role evidence, and the owner-decision-event registry.

`source_bundle_id` is `cyax-source-bundle-sha256:<digest>` where `digest` is the
canonical framed hash defined below over the four JSONL payloads and the
ordered object `(sha256, byte_count)` pairs. If a canonical source changes, the
old bundle remains valid as historical evidence and a fresh capture creates a
new bundle/snapshot. If a source is unavailable, historical replay may use the
captured object, but any mode claiming current freshness fails closed.

## Canonical encoding and stable IDs

All identity material uses a domain-separated canonical frame. Values are
encoded as: null `N || u64be(0)`; byte string
`X || u64be(n) || raw-bytes`; string `S || u64be(n) || NFC-UTF8`; integer
`I || u64be(n) || minimal-base10-ASCII`; false/true
`B || u64be(1) || 00|01`; array
`A || u64be(item_count) || frame(item)...`; object
`O || u64be(pair_count) || frame(key) || frame(value)...` with NFC keys sorted
by UTF-8 bytes. Lengths and counts are unsigned 64-bit big-endian. Absent fields
are omitted; present null fields use the null frame. Floats are forbidden.

Stable IDs are lowercase SHA-256 of these tuples:

| ID | Canonical hash input |
| --- | --- |
| `entity_id` | `['cyax-entity-v1', namespace, entity_type, canonical_source_identity]` |
| `source_revision_id` | `['cyax-source-revision-v1', source_kind, canonical_locator, object_sha256, source_event_at]` |
| `literal_id` | `['cyax-literal-v1', literal_type, canonical_value]` |
| `assertion_id` | `['cyax-assertion-v2', <complete canonical semantic assertion revision excluding assertion_id>]` |

The rendered forms are respectively `cyax-entity-sha256:`,
`cyax-source-revision-sha256:`, `cyax-literal-sha256:`, and
`cyax-assertion-sha256:` plus the digest. Namespace is the captured repository
identity or registered external namespace. The assertion preimage contains
`subject_id`, `predicate`, `object_id`, `literal_ref`, `source_revision_id`,
`source_locator`, `source_event_at`, `asserted_at`, `valid_from`, `valid_to`,
`validity_basis`, `authority_class`, `authority_derivation_rule_id`, `origin`,
`curation_state`, `review_state`,
`epistemic_state`, and `dispute_state`, with required nulls present. These are
identity-bearing semantic fields. Any change creates a successor assertion
with a different ID. Nonsemantic audit/build metadata is limited to `built_at`,
compiler/validator/curator implementation versions, process/host records,
diagnostics, and physical payload checksums; none may appear in the assertion
record or ID preimage. Rebuild wall time is always excluded.
On an ID collision with unequal canonical preimages, validation stops and
preserves both preimages as failure evidence; no suffix or rehash is allowed.

## Immutable assertion snapshot and identity

```text
snapshot/
  manifest.json
  entities.jsonl
  literals.jsonl
  source_revisions.jsonl
  assertions.jsonl
```

JSONL is UTF-8/NFC with sorted keys, no insignificant whitespace, LF after each
record, and primary-ID byte order. Timestamps and integers use the canonical
forms above. Semantically unordered arrays are sorted by their canonical framed
bytes; semantically ordered arrays retain order.

The **logical semantic checksum** hashes an explicit semantic projection, not
the complete physical JSONL payload or whole physical source bundle. The
`semantic_source_bundle_projection_checksum` covers only source revisions,
objects, actor/role evidence, and owner-decision registry entries reachable
from semantic assertions; display-only sources are excluded. The snapshot
projection contains entities ordered
by `entity_id` with `display_label_ref` removed; literals referenced by
semantic assertions (and no literal reachable only from a display label),
ordered by `literal_id`; complete source revisions referenced by semantic
assertions, ordered by `source_revision_id`; and complete assertions ordered by
`assertion_id`. It is SHA-256 of:

```text
frame([
  'cyax-logical-snapshot-v1', schema_version,
  semantic_source_bundle_projection_checksum,
  authority_rule_version, semantic_evaluator_rule_version,
  ['semantic_entities', framed_records],
  ['semantic_literals', framed_records],
  ['semantic_source_revisions', framed_records],
  ['semantic_assertions', framed_records]
])
```

`snapshot_id = cyax-snapshot-sha256:<logical_semantic_checksum>`. Reordered
input, a different compiler with identical semantics, and a diagnostic-label-
only change leave it unchanged. Assertion/provenance changes and authority or
evaluator rule-version changes alter it. The separate
`physical_payload_checksum` hashes the complete encoded physical payload,
including display references and display-only literals. The
`build_contract_checksum` hashes `snapshot_id`, `physical_payload_checksum`,
the complete physical `source_bundle_id`, and exact assertion-compiler, curator,
validator, and context-compiler versions.

`manifest.json` records all named versions and checksums, every payload byte
count/SHA-256/record count, ordered source-revision IDs/fingerprints, and total
entity/assertion/source/literal counts. On open, validators recompute and check
every manifest-derived value, ID, foreign key, enum, predicate signature,
literal constraint, canonical order, payload hash, source-bundle link, semantic
source-bundle/snapshot projections, logical checksum, physical payload checksum,
and build-contract checksum. A display literal referenced by an assertion is semantic and cannot
be excluded merely because it is also a label. Unknown fields fail under v1.

Identity conformance fixtures freeze these results:

| Change | Logical `snapshot_id` |
| --- | --- |
| reordered input only | unchanged |
| different compiler, same semantic projection | unchanged |
| diagnostic `display_label_ref`/display-only literal only | unchanged |
| assertion or provenance field | changed |
| authority/evaluator rule version | changed |

### Freshness versus consistency

`matches_selected_snapshot` means a materialization validates against the
explicit frozen snapshot ID. `fresh_against_current_sources` is separate. A
current-state mode reacquires every canonical locator at a new observation
boundary, verifies actor/role evidence, and compares captured revision IDs,
state/event records, and object fingerprints with the selected bundle. Any
change, unavailable source, incomplete pagination, or unverifiable authority
record returns `freshness=unknown|stale` and the mode fails closed. Historical
benchmark replay remains valid when labelled with its snapshot and observation
boundary; it cannot claim currentness.

### Atomic publication

Snapshots and each materialization use the same publication protocol:

```text
build in a new sibling temporary directory
→ validate complete logical export and metadata
→ commit final database transaction / checkpoint WAL
→ fsync payload and database files as supported
→ write and fsync final manifest with build_complete=true
→ fsync temporary directory
→ atomic rename to the content-addressed final name
→ fsync parent directory
```

The destination must not exist. SQLite performs its final transaction and WAL
checkpoint before the marker; G uses its documented checkpoint/close primitive.
Unsupported fsync semantics are recorded and invalidate primary-host approval.
Open rejects temporary names, missing markers, residual uncommitted WAL/shadow
state, checksum mismatch, or a marker not written as the final state. Crash
tests interrupt every stage and prove that the previous published snapshot/view
opens unchanged while the partial candidate never opens.

## Shared semantic evaluator and `RetrievalBundle` v1

Adapters retrieve candidate assertion IDs and traversal steps only. The same
backend-independent evaluator loads canonical snapshot records and applies
authority, admissibility, time, supersession, dispute, and dependency-staleness
rules. The output has every key present:

```text
RetrievalBundle
  contract_version = 'cyax-retrieval-bundle-1.0'
  query_instance_id
  snapshot_id
  as_of                         # timestamp or null
  entities[]
  literals[]
  assertions[]
  source_revisions[]
  paths[]
    path_id
    steps[]
      assertion_id
      direction                 # forward | reverse
  diagnostics                   # object or null
```

Semantic arrays are empty rather than null and are sorted by primary ID;
`literals[]` is ordered by `literal_id`.
Records are complete canonical snapshot objects, not backend projections.
`paths` are ordered by the framed sequence of `(assertion_id,direction)`;
steps retain traversal order. `path_id` hashes that sequence. Parallel edges
remain distinct because they have distinct assertion IDs. Exact duplicate step
sequences are invalid adapter output, not silently deduplicated. Thus semantic
path multiplicity is assertion-distinct and every exact path has multiplicity
one. All referenced entities, assertions, literals, and source revisions are
included exactly once. Every `literal_ref` in a returned assertion and every
non-null `display_label_ref` in a returned entity resolves to exactly one
complete canonical Literal in `literals[]`; other literals are omitted,
duplicate literals and dangling references invalidate the bundle, and a bundle
with no literal reference contains `literals: []`. Label references/literals
are retained in complete serialization but removed with the diagnostic-only
projection for semantic gold/S/G comparison. Assertion-literal closure remains
mandatory in every projection. The same closure applies before gold comparison,
diagnostics-stripped S/G parity, complete-bundle serialization, and
bounded-context compilation. The
context compiler accepts only a validated, reference-closed bundle and never
loads a missing literal from a backend. An absent optional value is forbidden
in v1; it is represented as null.

Diagnostics may contain backend/version, candidates examined, candidate bytes,
adapter-boundary bytes, native traversal work, page faults, timings, truncation,
or duplicate-candidate counts. Diagnostics never affect authority or the
context compiler and are removed before semantic comparison.

## Fixtures and frozen gold

### F-real — Issue #117 source audit

F-real captures Issue #117 and owner comment `5556641428`; merged PR #101 at
head `563a367a8a2336522c3f284747d23744ac82edc9` and merge
`2162e81fb77700753549c839c6208daecafa325f`; closed/unmerged PR #103 and
supersession comment `5515545835`; merged PR #104 at head
`2ea2e10c772475d89979d2e0f7a1b02a3b63133d` and merge
`07552d0c1615b3e8ed047d3480f64fd96912c74e`; Issue #112; and named files at
`vmm@7a40285bb5c313f7e8746b90644d5f45bb67be44`.

The audited K1–K12 answer key is:

| Item | Required fact and anchor |
| --- | --- |
| K1 | Issue #117 is open and remains a bounded follow-up; its body `Remaining work` retains direct-test and post-approval replay work. |
| K2 | Owner comment `5556641428` approves only ordinary-Euler evidence from smooth, unimodular local cones and explicitly rejects population promotion. |
| K3 | Issue #117 plus owner comment `5556641428` govern; no standalone approved CYAX-0117 spec exists in the captured repository tree. |
| K4 | Owner comment `5556641428` authorizes resuming bounded h11=4/h11=5 replay under the narrow contract and defers broader mathematics to #112. |
| K5 | PR #101 merged only the projection ledger/ignore exception, independently reproduced the projection digest, and changed no scientific code. |
| K6 | PR #103 is closed/unmerged and superseded by PR #104 via comment `5515545835`; the reused branch/content is the general-L driver, not #117's point-certificate task. |
| K7 | The baseline artifact digest is `b7cb293ea369d52fa22aa01db6b487303b40279f87e40d14a1509bb1a062dfa3` and contains 1,146 rows: 1,060 `h21_plus_nonzero`, 28 `accepted_exact_trilayer_action`, and 58 `smoothness_verification_unavailable`. The ledger is `validation/orientifold_unaffected_projection_ledger_20260824.md`. Its comparison selects the 1,088 unaffected rows. The canonical projection digest—not the baseline artifact or ledger digest—is `ebe14b02d312993fd85ce98b2aa882701b1c14f6aff25065e5d566a2f2ad504b`. |
| K8 | The ledger's `Observed bounded verification` records a completed 1,146-row historical repaired h11=4 replay (artifact digest `4227a9244e43e719df437126db09520ec7c29fa78937b9906532190791e9e076`) and matching 1,088-row projection. It predates the 2026-09-05 owner decision and is not the durable post-approval admissible h11=4/h11=5 replay required for the present gate. |
| K9 | Issue #112 remains the separate unresolved non-simplicial/orbifold research line and does not block the narrow path. |
| K10 | Historical replay and projection evidence exist. The remaining limitations are no focused direct test of `_derive_zero_dimensional_local_evidence` in the captured tests; CYTools is required for current execution; and no captured post-owner-approval h11=4/h11=5 replay has passed the later narrow admissibility/review gates. Unavailable cases remain unpromoted. |
| K11 | Preserve the historical replay. The next admissible work is the focused direct producer test and bounded h11=4/h11=5 replay under owner comment `5556641428`, with immutable inputs, CYTools availability, and normal independent evidence gates. |
| K12 | Abstain from population, orbifold/stringy-Euler, non-simplicial, or broader scientific conclusions absent separate evidence and approval. |

Each item is compiled into exact Claims/assertions/source anchors. The
independent K1–K12 source audit passed at the starting PR head and this bounded
repair does not reopen its chronology. CYAX-0166 remains unchanged and cannot
consume the still-unstable common interface before the new exact-head
architecture/methodology rereview. F-real's manifest registers every used K1–K12
`claim_key`, its exact literal type, and its stable semantic-slot description;
the registry is part of frozen gold and may not be inferred from Claim text.

### F-scale generator

Generator version `cyax-0168-scale-2.1` is a fully specified, scientifically
inert project-record generator. It uses no runtime RNG. For each random choice,
it evaluates `SHA256(frame(['cyax-gen-2.1', seed, profile_id, purpose,
ordinal, counter]))`; unsigned big-endian digest values feed rejection sampling
to avoid modulo bias. Endpoint candidates are always primary-ID sorted.

The generator uses namespace `cyax-0168-synthetic-v2.1`. Its manifest registers
`synthetic.block_claim` as literal type `text` with semantic slot
`synthetic_fixture_block_statement`. For block ordinal `b` and role ordinal
`r`, every Entity canonical source identity is the framed array
`[tier, profile_id, seed, b, r]`; the Claim's `claim_key` is
`synthetic.block_claim`; its literal value is the exact ASCII string
`block=<b>;profile=<profile_id>;seed=<seed>` with base-10 ordinals and no
padding. Every generated Source has kind `synthetic_fixture`, locator
`cyax://0168/scale/2.1/<tier>/<profile_id>/<seed>/block/<b>/source/<r>`, and
exact source bytes equal to canonical JSON of the sorted-key object
`{"block":b,"generator":"cyax-0168-scale-2.1","profile":profile_id,"role":r,"seed":seed,"tier":tier}`
followed by one LF. Its object digest/byte count and `source_revision_id` follow
the common rules. Generated `source_event_at` is null and `observed_at` and
`asserted_at` are exactly `2000-01-01T00:00:00.000000Z`; ordinary assertions
have null `valid_from`/`valid_to` and `validity_basis=unknown`. A `supersedes`
assertion instead has `validity_basis=explicit` and `valid_from` equal to
`2000-01-02T00:00:SS.000000Z`, where `SS` is the zero-padded successor position
within its chain (00 through 31), with null `valid_to`. All assertions use
`origin=source_direct`, `curation_state=independently_reviewed`,
`review_state=not_required`, `epistemic_state=supported`, and
`dispute_state=undisputed`, except phase-6 `contradicts` assertions use
`dispute_state=disputed`. Each assertion uses the referenced Source revision
and locator plus anchor `json-object`; parallel assertions use a second Source
whose role ordinal is the original role plus `1000000`. Generated authority is
`ordinary_record`; owner allowlists and decision registries are empty. No field
may depend on iteration order, locale, wall clock, runtime version, or
unspecified randomness.

Within block `b`, role ordinals and types are fixed as: 0 WorkItem, 1 Decision,
2 Requirement A, 3 Requirement B, 4 Implementation A, 5 Implementation B,
6 Verification, 7 Claim, 8 Artifact, and 9 Source. Every generated
`display_label_ref` is null. The 13 base assertions, in this exact construction
order, are `1 governs 2`; `0 requires 2`; `0 requires 3`; `2 requires 4`;
`3 requires 5`; `4 implements 2`; `5 implements 3`; `6 verifies 4`;
`8 verifies 3`; `8 supports 7`; `7 concerns 0`; `7 documented_in <block
literal>`; and `8 documented_in 9`. An ordinary assertion uses the Source and
Source revision of its subject's block; a cross-block/supersession assertion
uses the subject block's Source; and a phase-6 contradiction uses the second
Claim's block Source. Fill assertions follow the same subject-block rule.

Synthetic source-revision records use null actor ID, login, author association,
event/state fields, and role evidence; `authority_class=ordinary_record`,
`authority_derivation_rule_id=synthetic_fixture_v2.1`, anchor `json-object`, and
no additional metadata. The Source entity's canonical source identity is still
the common framed entity identity above; its revision is the digest-bound
record for the exact synthetic bytes. Parallel evidence creates one additional
revision for that Source with role ordinal `1000000 + original assertion
construction ordinal` in both locator and canonical JSON; it does not create a
new Source entity. These rules, the common canonical schemas, and the phases
below determine every field in every Entity, Literal, Source revision, and
Assertion record.

Every tier entity count is divisible by ten. Each consecutive ten-entity block
contains exactly one WorkItem, one Decision, two Requirements, two
Implementations, one Verification, one Claim, one Artifact, and one Source,
with one source revision. IDs use `(tier, block ordinal, role ordinal)` as the
canonical source identity. Each block emits a fixed 13-assertion base motif:
one `governs`, four `requires`, two `implements`, two `verifies`, one
`supports`, one Claim `concerns`, one Claim-to-literal `documented_in`, and one
Artifact-to-Source `documented_in`. It then
applies exact profile quotas for dependencies, cross-links, disputes,
supersession chains, cycles, parallel evidence, and disconnected components.
Supersession targets only an earlier entity of the same type. Disputes create
paired supported/contradicting Claims with independent sources. Cycles are
explicit `depends_on` cycles of length three and never enter supersession.
Parallel edges are allowed only when assertion IDs differ by source revision.
Cross-links connect distinct work items. Quotas use floor(rate × eligible
population), sorted eligible ordinals, and PRF-selected endpoints; collisions
advance `counter`. Remaining assertion slots are filled with profile-neutral
`concerns` evidence in ordinal order. Exact tier counts and all invariants must
hold or generation fails.

For `B = entity_count / 10`, the ordered post-motif phases are executable:

1. Mark the highest-ID `floor(isolated_rate × B)` blocks isolated; they may
   contain internal motif edges but are excluded from cross-block phases.
2. Partition the other blocks into consecutive components of
   `max(path_depth + 1, 64)` blocks, with a shorter final component. Within each
   component, first add a WorkItem `depends_on` chain of exactly
   `min(path_depth, component_size - 1)` edges.
3. Add further cross-block `depends_on` assertions until the dependency phase
   contains exactly `min(dependency_fanout × connected_blocks, 30 × B)` edges.
   Source blocks cycle in ID order; each target is selected by the PRF from
   other blocks in the same component. The lowest-ID
   `floor(cross_link_rate × dependency_quota)` phase positions instead select a
   target in a different component. Self-edges and duplicate subject/object
   pairs are rejected.
4. Form candidate triples from consecutive primary-ID-sorted connected
   WorkItems. A triple is eligible only when each member has at least one
   non-chain outgoing dependency edge and none of the three proposed cycle
   pairs already exists. Take the lowest-ID
   `floor(cycle_rate × connected_blocks)` disjoint eligible triples. For each
   `(a,b,c)`, remove the lexicographically smallest `(assertion_id, subject_id,
   object_id)` non-chain outgoing dependency assertion for each member, then
   add exactly `a→b`, `b→c`, and `c→a` using the corresponding removed
   assertion's Source revision and provenance fields. If the quota cannot be
   met, generation fails. The exact removed assertion IDs are recorded in the
   generator trace; no chain edge is replaced. A cycle never enters
   supersession.
5. Partition the lowest-ID still-eligible WorkItems into as many disjoint
   chains of exactly `supersession_depth` as fit; add one `supersedes` assertion
   between consecutive members. A remainder shorter than the declared depth is
   unused. The new member always supersedes the previous member.
6. Pair the lowest-ID `2 × floor(dispute_rate × B)` Claims; add one
   `contradicts` assertion from the second Claim in each pair to the first. The
   base evidence continues to support both, so the evaluator sees a real
   dispute rather than a fabricated resolution.
7. For `floor(parallel_evidence_rate × 13 × B)` lowest-ID base assertions,
   create a semantically parallel assertion with a different Source revision.
   Exact subject, predicate, and object are retained; provenance and assertion
   ID differ.
8. Fill the remaining assertion budget to exactly `50 × B` with `concerns`
   assertions selected by the same PRF over signature-valid, nonduplicate pairs.

Every phase uses the profile, seed, purpose, ordinal, and rejection counter in
the PRF. The validator checks exact entity-type counts, phase quotas, intended
fan-out and shortest-path/depth strata, same-type supersession, dispute pairs,
cycle count, parallel provenance, isolated blocks, predicate signatures,
referential integrity, and total counts.

Generator acceptance is byte-level: two independent implementations supplied
only with generator version 2.1, tier, profile, and seed must produce identical
complete canonical snapshot records and the same logical snapshot checksum.
Topology/count checks alone do not pass. The version is bumped from 2.0 because
complete generated bytes and cycle-replacement identities are now frozen.

Topology is separate from seed. Each generated snapshot also contains the
low/median/high query-selectivity strata frozen below, so selectivity is not
silently encoded by the topology seed:

| Profile | dependency fan-out | path depth | supersession depth | dispute rate | cross-link rate | cycle rate | disconnected components | parallel-evidence rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P-low` | 2 | 4 | 2 | 0.005 | 0.01 | 0.000 | 20% isolated work-item blocks | 0.01 |
| `P-medium` | 8 | 8 | 8 | 0.020 | 0.05 | 0.005 | 5% isolated work-item blocks | 0.03 |
| `P-high` | 32 | 16 | 32 | 0.100 | 0.20 | 0.020 | 1% isolated work-item blocks | 0.10 |

The smallest decision matrix is:

| Tier | Entities | Assertions | Profiles and seeds |
| --- | ---: | ---: | --- |
| T0 | 1,000 | 5,000 | smoke projection of `P-medium`: `162000` |
| T1 | 10,000 | 50,000 | `P-low`: `162011,162012`; `P-medium`: `162013,162014`; `P-high`: `162015,162016` |
| T2 | 100,000 | 500,000 | `P-low`: `162021,162022`; `P-medium`: `162023,162024`; `P-high`: `162025,162026` |
| T3 | 200,000 | 1,000,000 | `P-low`: `162031`; `P-medium`: `162032`; `P-high`: `162033` |
| T4 | 1,000,000 | 5,000,000 | conditional `P-medium`: `162041` |

### Frozen query instances and gold

Instances are selected from generated snapshots before backend implementation
with the reference in-memory evaluator. For a population sorted by
`(metric, primary_id)`, nearest-rank percentile `p` selects rank
`ceil(p × population_size)` (one-based); pair populations use framed
`(source_id,target_id)` as the primary tie key. Each row below yields exactly
one preregistered decision instance per tier/profile/seed. Diagnostic 10th and
90th percentile instances may also be frozen, but they are never pooled into or
substituted for the decision statistic.

| Query | Eligible population and metric | Decision selector | Parameters | Tie break | Invalid-fixture rule | Use |
| --- | --- | --- | --- | --- | --- | --- |
| Q01 | entities with at least one admissible current `governs` result; result count | 50th percentile | `as_of=2000-01-03T00:00:00.000000Z`, depth null | smallest entity ID | empty population | decision |
| Q02 | entities with at least one admissible authority-evidence result; result count | 90th percentile | same `as_of`, depth null | smallest entity ID | empty population | decision |
| Q03 | reachable Decision/Verification pairs matching the required typed path; shortest distance | maximum distance not exceeding profile `path_depth` | `as_of` null, depth=`path_depth` | smallest framed pair | no eligible pair | decision |
| Q04 | supersession assertions; `valid_from` instant | 50th percentile | `as_of` equals the selected assertion's `valid_from`, depth null | assertion ID then subject ID | no supersession | decision |
| Q05 | entities with nonzero reverse direct `depends_on` degree; degree | 50th percentile | `as_of` null, depth=1 | smallest entity ID | empty population | decision |
| Q06 | entities with nonzero bounded reverse dependant reachability; result count | 90th percentile | `as_of` null, depth=`path_depth` | smallest entity ID | empty population | decision |
| Q07 | superseded entities with nonempty dependency-stale reachability; result count | 90th percentile | `as_of=2000-01-03T00:00:00.000000Z`, depth=`path_depth` | smallest entity ID | empty population | decision |
| Q08 | entities concerned by at least one dispute record; total inspectable dispute count | 50th percentile | same `as_of`, depth null | smallest entity ID | empty population | decision |
| Q09 | Claims with a typed evidence path to Source; shortest distance | maximum attainable distance | `as_of` null, depth=`path_depth` | smallest Claim ID, then Source ID | empty population | decision |
| Q10 | unordered pairs of current Claims for which each Claim has admissible Source evidence; minimal evidence-cover assertion count | 90th percentile | same `as_of`, depth=`path_depth` | smallest framed sorted Claim-ID pair | fewer than two eligible Claims | decision |
| Q11 | WorkItems concerned by a current Claim registered to semantic slot `synthetic_fixture_block_statement`; eligible-result count | 50th percentile | same `as_of`, depth null | smallest WorkItem ID | empty population | decision |
| Q12 | Requirements with at least one implementing and one verifying subject; combined result count | 50th percentile | same `as_of`, depth null | smallest Requirement ID | empty population | decision |

Any invalid population invalidates that entire tier/profile/seed fixture before
backend work; the generator must be corrected by a versioned design amendment,
never by post hoc reselection. F-real instances are named directly by audited
Claim/entity IDs and are diagnostic correctness cases, not performance-decision
instances. The manifest freezes all target IDs/pairs, parameters, role,
population checksum/count, selector result, gold bundle, and gold checksum.

Semantic answers are:

| ID | Category | Frozen answer object |
| --- | --- | --- |
| Q01 | naturally relational | admissible current governors of X: exact entity/assertion/source sets |
| Q02 | semantic/ranking | minimal admissible authority evidence for X, ordered by authority class then assertion ID |
| Q03 | naturally graph-shaped | one shortest Decision→Requirement→Implementation→Verification witness; lexicographically smallest step sequence breaks ties |
| Q04 | naturally relational | exact entities/assertions superseded at `as_of=T` under half-open time rules |
| Q05 | neutral | exact direct reverse `depends_on` neighborhood of X |
| Q06 | naturally graph-shaped | bounded transitive dependant reachability set plus one lexicographically smallest shortest witness per reached entity |
| Q07 | naturally graph-shaped | exact dependency-stale reachability set after X's admissible supersession plus one deterministic witness per entity |
| Q08 | neutral | exact admissible and inspectable dispute records concerning X, with derived dispute state |
| Q09 | naturally graph-shaped | one shortest Claim→Verification/Artifact→Source evidence witness, lexicographically tied |
| Q10 | semantic/ranking | deterministic minimal evidence set: cover all required query Claims; minimize assertion count, then source-revision count, then framed sorted assertion-ID list; return every exact tie |
| Q11 | semantic/ranking | next permissible action Claims: filter current unsuperseded action Claims concerning X; reject those with unmet admissible `depends_on` requirements or unresolved dispute; rank governing `accepted`, then `verified`, then `supported`, and return all Claims tied at the best rank in Claim-ID order |
| Q12 | naturally relational | exact Implementation/Artifact and Verification/Artifact subjects that implement or verify R |

Path-like queries return the precise deterministic witness objects above, not
backend-native path enumeration. Reachability queries compare both the full
reachability set and their deterministic witnesses.

## Materializations, parity, and updates

S uses the Python standard-library SQLite binding, normalized entity/literal/
source_revision/assertion/metadata tables, foreign keys and CHECK constraints,
B-tree indexes in both assertion directions, and recursive CTEs with explicit
cycle prevention and frozen depth. G uses typed entity nodes and assertion
records/edges sufficient to recover every original assertion ID. Neither stores
mutable semantic disposition flags or large source text.

Correctness has three gates for every snapshot:

1. independently generated frozen gold versus S canonical bundles;
2. the same frozen gold versus G canonical bundles;
3. diagnostics-stripped canonical S versus G bundles.

The comparison covers entities, assertions, complete provenance, direction,
assertion-distinct path multiplicity, time, authority, disputes,
supersession, and dependency staleness. Rebuild validation also compares a
complete canonical logical export of every record, including records untouched
by Q01–Q12. Agreement between two backends is never sufficient by itself.

Updates do not mutate snapshot N. Removal is a benchmark delta transformation,
not an authority source, assertion, tombstone, retraction predicate, or claim of
truth. Each frozen canonical delta contains `base_snapshot_id`,
`expected_target_snapshot_id`, primary-ID-ordered complete additions for source
objects/revisions/entities/literals/assertions, and primary-ID-ordered unique
`remove_assertion_ids[]`. Every removal ID must exist in N; every addition must
be a complete canonical record; duplicate, missing, dangling, or colliding IDs
fail before either backend runs:

```text
immutable snapshot N + frozen delta → immutable snapshot N+1
```

S and G receive the same N, delta, and target identity. They may incrementally
update disposable indexes, but their complete logical export must equal N+1.
N+1 is independently frozen from a full canonical build, and applying the delta
must reproduce its exact logical export and `snapshot_id`; historical N remains
immutable and available. Dependency insertion adds a complete `depends_on`
assertion; dependency removal names its exact assertion ID. Source-revision
replacement adds the new revision and successor assertions and removes only
the explicitly listed old assertion IDs. Supersession adds a complete
`supersedes` assertion and does not delete history. Rollback discards the
unpublished candidate and reopens N. Crash interruption leaves N published and
causes any incomplete N+1 to fail open; recovery either discards the candidate
or completes it from the same delta and must match the frozen N+1.
Every measured repetition begins from an independent reflink/copy verified to
match N; it never reuses a prior repetition. Frozen batches are one assertion,
100 assertions, 1%, source-revision replacement, supersession, and dependency
insert/remove. Crash interruption at each transaction/publication phase must
leave N readable, reject the partial N+1, and prove rollback or recovery before
timing is admissible.

## Candidate pin and stop-only fallback

The primary candidate is `ladybug==0.20.4`, upstream immutable tag `v0.20.4`
at full commit `df58ee387c4e5e9f02bb9d518636b52cd4abe5f7`, independently confirmed from
the official tag. It is MIT licensed. The existing macOS ARM64 smoke used
`ladybug-0.20.4-cp314-cp314-macosx_15_0_arm64.whl`, SHA-256
`7a36d5b051ddc954d7ee5d5fa6165fb49d48a4785a132bafdd311723897fa649`,
with CPython 3.14.6 and offline installation. That evidence is candidate-host
smoke only.

Before implementation on the approved Linux x86-64 host, freeze the exact
compatible wheel filename and SHA-256, every dependency wheel/hash, Python and
platform tags, license files, offline wheelhouse manifest, and proof of
`PIP_NO_INDEX=1` import/build/reopen/query/export. Disable network and runtime
extension installation. If no supported approved-host wheel exists or any
frozen smoke check fails, stop.

FalkorDBLite is not an automatic fallback. Its use would require an amended
design with exact engine/module/client pins, dependency hashes, transitive
license review, offline artifact proof, process-tree accounting, independent
rereview, and owner approval before any implementation or measurement.

## Calibration and fairness controls

A separate non-decision corpus uses generator version 2.1: 25,000 assertions
with `P-low/P-medium/P-high` seeds `168901/168902/168903`, and 125,000
assertions with seeds `168904/168905/168906` respectively. It is
never reused in F-real or T0–T4. Each backend receives at most eight person-hours
of tuning and 40 executed plan/config trials. Permitted changes are SQLite
index selection/order, documented pragmas, and CTE formulation; or documented
Ladybug index/configuration and equivalent query formulation. Schema semantics,
evaluator, output, query instances, hardware, data, and resource envelopes may
not change. Stop at the earlier of budget exhaustion or five consecutive trials
without ≥2% improvement in the preregistered geometric mean of calibration
latencies while passing correctness. Final schemas, queries, indexes,
pragmas/configuration, dependencies, plans, evaluator, and tuning decisions are
hashed and committed before **any F-real or T0–T4 decision fixture is
materialized, executed, profiled, explained, or inspected through either
backend**. Public seeds provide no blinding and are never described as hidden
or unrevealed. Calibration uses only its dedicated corpus; no decision-result-
driven tuning follows. A future claim of blinding would require a separately
reviewed commitment/reveal protocol.

The controlled primary comparison uses one pinned CPU core/thread per query,
one process and one connection, no concurrency, identical snapshot parser,
semantic evaluator, entity resolution, and context compiler, no vector search,
no graph-native FTS, all indexes built before timing, disabled network, fixed
affinity/thread environment variables, and checkpointed steady state. SQLite
FTS is excluded from Q01–Q12. A separately labelled native-defaults secondary
run may be reported but cannot replace the controlled result.

Cache states are named precisely. `application-cold` is a fresh Python process;
`database-open-cold` is the first query after opening the connection in that
process; `warm-process` reuses that process/connection. The primary query
campaign deliberately preconditions and retains the OS filesystem page cache,
so both cold categories are `filesystem-cache-warm`; it never calls them OS
cold. Backend order is paired and balanced. Any optional page-cache-dropped run
is secondary and requires the same privileged procedure for both. Record minor
and major page faults for every process where Linux tooling permits.

Immediately before each paired primary measurement, a fresh, version-hashed
`cache_warm` helper process performs one buffered sequential read from byte zero
through EOF of every regular materialization file included by the frozen
backend manifest: data, index, checkpoint, and steady-state auxiliary files;
WAL/shadow/spill files must be empty/absent at steady state or are included.
It warms S and G separately, verifies each ordered `(relative_path, byte_count,
sha256)` against the manifest, uses a 8 MiB read buffer, records bytes read,
start/end monotonic times, exit status, and major/minor faults, closes all file
descriptors, then memory-maps each file read-only and uses Linux `mincore` to
record and require `resident_pages=total_pages` before unmapping and exiting.
The first-read
backend alternates by the same balanced pair schedule as measured execution;
within each pair the helper reads the first backend, then the second, while the
measured order is first then second. Both complete materializations must fit
within the preregistered cache-residency budget of 50% of physical RAM combined;
otherwise `filesystem-cache-warm` is invalid. No other file-touching process
runs between helper exit and the pair. Missing files, digest/byte mismatch,
short reads, nonzero helper exit, incomplete residency, or monitor gaps
invalidate the pair. Measured major faults remain reported evidence but do not
alone invalidate a pair whose full pre-measurement residency proof passed. Each
new pair repeats the full helper procedure, so prior pair state is not used as
evidence that warming occurred.

## Primary host and measurement protocol

The proposed authority host is dedicated bare-metal Linux x86-64 with at least
64 GiB RAM, local NVMe, ext4 (or a preregistered recorded local filesystem), no
swap, no competing workload, isolated fixed CPU affinity, one benchmark thread,
and frozen governor/turbo/power policy where controllable. macOS ARM64 remains a
smoke host only. Freeze CPU model, microcode, core/SMT layout, RAM, Linux
distribution/kernel, filesystem/mounts, storage device/firmware, Python,
SQLite version and compile options, Ladybug wheel/commit/dependencies, libc and
native runtimes, and measurement-tool versions.

Primary repetition candidates, frozen for precision validation, are:

- clean build/rebuild: 12 independently cloned paired repetitions;
- each snapshot transition batch: 30 independent paired repetitions;
- application-cold/database-open-cold queries: 200 paired fresh-process
  repetitions per decision query instance;
- warm queries: 100 independent paired process blocks per decision query
  instance, each with 5 unmeasured warmups and 50 measured calls.

The exact p95 estimator is nearest rank: for sorted measurements
`x_(1)..x_(n)`, `p95=x_(ceil(0.95n))`. Thus 200 fresh processes place the
endpoint at order statistic 190 rather than estimating a tail from 20 units;
100 warm blocks preserve 100 independent resampling units while the 50 calls
within each block characterize that process only. Build and transition gates
use all-sample hard-envelope breach plus median/maximum reporting, not a p95
classifier endpoint; their 12/30 counts test paired repeatability within the
48-hour campaign budget.

Before these counts become executable, a preregistered precision report using
only the calibration corpus must simulate 10,000 campaigns at the independent-
unit level over the observed distributions and lognormal/two-component-tail
stress variants. It must demonstrate at least 95% interval coverage, at least
90% correct classification when the true speedup is 25% beyond each threshold,
and at most 5% false-positive classification at the null, separately for fresh
and warm modes. Failure stops CYAX-0168 G1 before any decision-fixture access
and requires an amended repetition design and rereview; counts may not be tuned
after any decision fixture access.
This analytical order-statistic basis plus the calibration-only simulation—not
bootstrap resample count—is the required tail-precision justification.

The independent unit is the paired build, transition, fresh process, or warm
process block—not an individual warm call. Alternate first backend with a
frozen balanced schedule. Report raw values, paired differences and ratios,
nearest-rank p50/p95 as defined above, then paired BCa 95%
bootstrap intervals with 10,000 deterministic resamples at the independent
block level. Seeds for bootstrap/order live in the preregistration manifest.
Do not pool tiers, profiles, seeds, query families, or repeated calls as
independent observations. Missing or corrupt samples invoke invalid execution;
valid timeouts and resource breaches remain censored-at-limit observations and
enter the resource classifier rather than being replaced or called invalid.

Process-tree accounting is primary. Record absolute peak RSS, incremental RSS,
PSS/USS where `/proc` permits, CPU/wall time, minor/major faults, logical and
allocated bytes for data/index/WAL/checkpoint/shadow/spill files, and temporary
peak disk. Apply one frozen `gc.collect()` immediately before every measured
Python block, never between timed calls, for both backends. Builds and queries
run in a cgroup v2 scope or equivalently validated process-tree monitor; child
processes cannot escape accounting.

The same semantic bundle passes through one compiler/tokenizer. Final selected
assertions, bundle semantics, rendered context bytes, and prompt tokens must be
identical. Differences are parity/compiler failures, never backend context
savings. Candidates examined, candidate bytes, adapter-boundary bytes, and
native traversal work remain diagnostics only.

## Hard resource envelope and conditional T4

For each T3 backend/profile: clean build ≤30 minutes; rebuild or 1% transition
≤30 minutes; peak process-tree memory ≤16 GiB; temporary disk ≤50 GiB;
individual query ≤120 seconds; no swap. The total decision campaign is ≤48
hours excluding setup and explicitly invalid reruns.

T4 runs only when all are true:

1. both backends pass all T3 semantic and resource gates;
2. no measurement invalidation remains;
3. T3 projects a classification-relevant crossover by 5 million assertions or
   lies within 25% of a decision threshold;
4. both systems have at least 25% projected memory, disk, and time headroom;
5. the T4 fixture/workload/hash was frozen before T3 results were examined.

T4 alone cannot justify graph adoption for current CYAxiverse scale.

## Proposed thresholds and deterministic classifier

These thresholds remain proposals pending owner approval. A **query family** is
exactly one frozen Q ID; graph-shaped families are Q03/Q06/Q07/Q09,
relational families are Q01/Q04/Q12, semantic/ranking families are
Q02/Q10/Q11, and neutral families are Q05/Q08. Q07 is the preregistered
critical impact-analysis family.

For a fresh decision instance/backend, the query-instance statistic is the
nearest-rank p95 of its 200 independent process measurements. For warm mode it
is the nearest-rank p95 of all 5,000 measured calls, preserving each 50-call
process block intact during inference; calls are observations, not independent
units. For one tier/family/profile, the **family/profile latency index** is the
geometric mean of the query-instance p95 values across every frozen decision
instance/seed in that cell, with equal weight per instance. It is not itself a
pooled p95. The **family/profile speedup of A over B** is B latency index divided
by A latency index; unqualified “speedup” below means G over S. **Absolute
saving of A over B** is the arithmetic mean across the same instances of
`(B query-instance p95 - A query-instance p95)`; unqualified “absolute saving”
below means G over S.

Each of 10,000 deterministic paired bootstrap resamples samples whole paired
fresh processes, or whole paired warm-process blocks, within each instance;
all calls in a selected warm block travel together. It recomputes instance
p95s, then the equal-weight family/profile index, speedup, and absolute saving.
For statistic `theta`, let `z0=Phi^-1((count(theta* < theta_hat) +
0.5 count(theta* = theta_hat))/10000)`. Let `a` be the standard delete-one
independent-unit jackknife acceleration
`sum((mean(theta_j)-theta_j)^3)/(6*(sum((mean(theta_j)-theta_j)^2))^(3/2))`.
For `alpha` in `{0.025,0.975}`, use adjusted probability
`Phi(z0 + (z0 + Phi^-1(alpha))/(1 - a*(z0 + Phi^-1(alpha))))`; the BCa endpoints
are linearly interpolated empirical bootstrap quantiles at those two adjusted
probabilities, using quantile type 7 (`h=1+(B-1)p`, linear interpolation between
the surrounding one-based order statistics). Undefined denominator/acceleration or an adjusted probability
outside `[0,1]` makes the cell Inconclusive. No measurement is pooled across Q ID,
tier, profile, seed, cache mode, or process type, and no mean/median/worst-case
choice remains open. “p95” means only the query-instance estimator;
“family p95” is forbidden shorthand for the family/profile latency index.

**G — experimental derived-index status:** At T3, on at least two preregistered
graph-shaped families and at least two of three topology profiles, require
speedup point estimate ≥2.0, lower 95% CI ≥1.5, and absolute end-to-end saving
of G over S ≥10 ms. The same family/profile direction at T2 requires point estimate >1.0
and lower 95% CI ≥1.0. All semantic and resource gates must pass.

**Hybrid:** G either meets the preceding rule on two graph-shaped families, or
one preregistered critical impact-analysis family has point estimate ≥5.0,
lower 95% CI ≥3.0, and absolute saving ≥50 ms. Under either alternative, S must
simultaneously have a material relational-family advantage: G latency index / S
latency index point estimate ≥2.0, lower 95% CI ≥1.5, and ≥10 ms absolute
saving of S over G. Routing
remains explicit and static. All semantic and resource gates must pass.

**Resource acceptance:** G materialization footprint (data, indexes, and
steady-state auxiliary files) and peak query memory must each be ≤2× S or have
absolute excess ≤2 GiB. G build/rebuild/transition time must be ≤3× S and
always inside the hard envelope. Final context/prompt values must match exactly.

Resource/control disposition is frozen separately from measurement validity:

| Observed case | Disposition |
| --- | --- |
| host controls or process-tree/resource monitoring invalid | Inconclusive / invalid execution |
| measurement evidence missing, corrupt, or pairing broken | Inconclusive / invalid execution |
| G alone validly exceeds a hard T3 envelope while S passes | Retain S / G not justified |
| G violates relative resource acceptance with otherwise valid measurements | Retain S / G not justified |
| S alone validly exceeds a hard envelope while G passes | Operational envelope failure / owner decision required; do not infer G adoption |
| both backends validly exceed a hard envelope | Operational envelope failure / owner decision required |
| query reaches the frozen 120-second limit with valid monitor/control evidence | measured resource breach, censored at 120 seconds; apply the corresponding backend-breach row above |
| crash or implementation failure before the common contract is exercised | Inconclusive / invalid execution |
| demonstrated defect in the shared semantic contract | Architecture problem |

`Operational envelope failure / owner decision required` is a sixth outcome,
added because a valid S breach cannot honestly be called invalid or used to
infer G. It authorizes neither production adoption nor threshold relaxation.

Classifier precedence is:

1. **Inconclusive / invalid execution** for failed host controls, missing or
   corrupted samples, broken pairing, or implementation failure before the
   common contract is exercised. A valid resource breach is not in this class.
2. **Architecture problem** only for a demonstrated common-contract defect:
   frozen-gold/parity failure, non-failing snapshot identity, or nondeterministic
   complete logical export.
3. **Retain S / G not justified** when S passes and G has any valid hard or
   relative resource breach.
4. **Operational envelope failure / owner decision required** when S has a
   valid hard breach, whether or not G also breaches.
5. **Inconclusive / invalid execution** when a confidence interval required to
   distinguish otherwise eligible performance classes overlaps its threshold.
6. **Hybrid** when its performance and resource rule passes.
7. **G (experimental derived-index status)** when its rule passes and no material
   S relational advantage requires Hybrid.
8. **Retain S / G not justified** for every other valid execution.

Examples: G 4× faster but 3× larger with >2 GiB excess retains S; a win only at
T4 retains S; a 20× critical-query win yields Hybrid only if its lower CI,
absolute saving, S relational advantage, and resource gates pass; candidate-byte
reduction without end-to-end latency never changes the outcome; and a required
CI overlapping a decision-bearing threshold is Inconclusive. G timing out at
120 seconds while S succeeds retains S; S timing out while G succeeds produces
Operational envelope failure, not G adoption; both timing out produce the same
operational-envelope outcome. A timeout is preserved as a measured breach, not
silently rerun into a favorable class.

### Required adversarial conformance cases

Before CYAX-0168 G1, contract tests must prove: an ordinary owner comment is not
a decision without exact registration; an epistemic-state-only successor gets
a new assertion ID; a display-label-only edit preserves logical snapshot ID;
dependency removal deletes only the named assertion in N+1; independent
generator implementations produce the same complete checksum; multiple
instances in a Q family receive equal frozen weighting; a G-only timeout while
S passes retains S; an S-only timeout while G passes yields Operational envelope
failure; every filesystem-cache-warm pair uses a fresh successful helper; and a
decision-bearing CI that overlaps its threshold yields Inconclusive. These are
normative examples of the contracts above, not substitute benchmark results.

## Requirements and gates

| Requirement | Required behavior | Verification |
| --- | --- | --- |
| R-001 Common semantics | Registered Claim keys/types, closed literals/enums/predicates, exact owner-decision events, and distinct provenance times feed one evaluator. | Unregistered-key/type, ordinary-owner/question/proposal/non-owner authority, temporal, and dispute fixtures. |
| R-002 Immutable evidence/identity | Complete semantic assertion IDs, content-addressed source bundle, semantic snapshot projection, separate physical/build checksums, freshness, and atomic publication fail closed. | State-change ID, label-only stability, rule-version change, rebuild, collision, tamper, stale, unavailable-source, and crash tests. |
| R-003 Fair materializations | Normalized indexed S and pinned G derive only from the same snapshot. | DDL/index/config/query-plan review and complete-export equality. |
| R-004 Source-correct F-real | Audited K1–K12 distinguish historical evidence from current gates. | Preserve the passed independent source/anchor audit; no chronology rewrite. |
| R-005 Deterministic scale | Byte-complete generator 2.1 produces the frozen profile/seed matrix exactly. | Two independent implementations reproduce complete records/checksum; exact cycle-removal trace and invariants. |
| R-006 Frozen workload | The Q01–Q12 population/selector/parameter/role table, gold, and exact aggregation precede backend implementation. | Invalid-stratum/tie tests, manifest/gold review, and raw-to-classifier statistic tests. |
| R-007 RetrievalBundle v1 | Both adapters return complete, literal-reference-closed canonical objects and deterministic directional paths. | Empty/order/unique/no-dangling tests, complete serialization/compiler validation, and gold/S/G equality. |
| R-008 Successor updates | N + a non-authoritative delta with exact removals yields independently frozen immutable N+1. | Dependency insert/remove, replacement, supersession, complete export, crash, rollback, and isolation tests. |
| R-009 Preregistered measurement | Calibration non-access, host, fairness, cache helper, precision validation, repetitions, exact statistics, confidence, and resources are frozen. | Pre-access hash, cache evidence, precision thresholds, host/control audit, and paired block-level analysis validation. |
| R-010 Deterministic decision | Exact outcomes, valid resource-breach table, T4 gate, and ambiguous cases are encoded. | G/S/both breach/timeout, CI overlap, classifier boundary/table tests, and owner approval. |

### CYAX-0168 G0 — repaired design approval

Acceptance requires independent architecture/methodology rereview and owner
approval citing the exact head, passed K1–K12 audit, backend pin, host,
workload, repetition candidates/ratification rule, envelope, and thresholds.
`approval_ref: null` or
any unresolved normative choice stops implementation.

### CYAX-0168 G1 — frozen inputs and smoke

Acceptance requires frozen source/gold/generator/query/calibration/dependency
manifests, a passing calibration-only precision report that ratifies the frozen
repetition counts, and offline S/G clean-build/rebuild/reopen/tamper/crash smoke
checks.
Source asymmetry, runtime download, or nondeterministic export stops.

### CYAX-0168 G2 — semantic correctness

Each backend must pass independent gold, pairwise parity, and complete logical
export before timings for that fixture are admissible. A common-contract defect
is Architecture problem; a backend implementation failure before the contract
is tested is Inconclusive/invalid.

### CYAX-0168 G3 — T0 through T3 systems benchmark

The paired protocol must complete within controls and envelopes. Missing or
corrupt evidence is Inconclusive/invalid and blocks T4.

### CYAX-0168 G4 — conditional T4 and classification

Run only under the five-part T4 gate, then apply the approved deterministic
classifier. Production adoption remains separate.

## Open owner decisions

Before benchmark implementation or execution, the repository owner must:

1. approve or amend the closed enums, predicate signatures, Claim/literal
   mechanism, authority derivation, time rules, and `RetrievalBundle` v1;
2. approve the source-bundle/stable-ID/snapshot/freshness/publication contracts;
3. acknowledge the passed corrected K1–K12 audit and decide when the separate
   CYAX-0166 copy may be repaired after this interface stabilizes;
4. approve generator 2.1, profile/seed matrix, query-selection/gold rules, and
   successor-snapshot update contract;
5. approve the calibration budget, controlled comparison, cache state,
   repetition counts, BCa method, and process-tree resource accounting;
6. approve the Linux host class and provide the exact qualifying host manifest;
7. approve Ladybug 0.20.4 only after the approved-host wheel/dependency hashes,
   license, offline installation, and smoke evidence exist;
8. approve the hard resource envelope, conditional T4 gate, proposed thresholds,
   workload categories, and classifier.

## Non-scope and completion

No benchmark code, generated fixture/database, timing/resource run, package
dependency, fresh-agent cohort, transcript ingestion, vector search, Julia code
graph, general scientific ontology, scientific/package behavior, public API,
persisted scientific schema, version change, CYAX-0166 revision, or production
backend adoption is authorized here.

This S2 work completes only after a later approved implementation/execution
passes CYAX-0168 G0–G3 (and G4 if eligible), publishes replayable evidence, and
returns the classification to #162. This repair stops before independent
rereview and approval.
