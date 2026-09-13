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

This revision repairs the design after independent review. It does not approve
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

A `Claim` additionally has:

- `claim_key`: a closed fixture-local semantic slot such as `K07.total_rows`;

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
enum strings are forbidden. `text` is permitted only for frozen Claim content
or diagnostic display labels; it is not parsed to infer authority or state.
Every semantic literal is stored canonically in the snapshot and linked through
its Claim and source-provenanced assertion to exact captured source evidence.
`literal_ref` in an assertion is not a generic escape hatch: it is permitted
only where the predicate-signature table says so, and neither backend may
replace an entity relation with a literal property.

### Frozen vocabularies

All values are lowercase ASCII tokens. Unknown values fail validation.

| Field | Allowed values and semantics | Provenance and transitions |
| --- | --- | --- |
| `source_kind` | `github_issue`, `github_issue_comment`, `github_pull_request`, `github_pull_request_comment`, `github_review`, `git_commit`, `repository_file`, `verification_artifact`, `external_document` | Source-derived from the captured object type; immutable for a source revision. |
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
- `recorded_at`, the assertion-recording time;
- nullable `valid_from` and `valid_to` domain/effective bounds;
- `validity_basis = explicit | source_event | unknown`;
- `origin`, `curation_state`, `review_state`, `epistemic_state`, and
  `dispute_state` from the frozen vocabularies.

`recorded_at` is deterministic: it equals the captured source observation time
for `source_direct`, the captured curator action time for
`curator_interpretation`, and the latest supporting assertion's `recorded_at`
for `rule_derived`. It is never rebuild wall time.

`source_locator` uses a source-kind-specific stable anchor: GitHub comment ID,
review ID, Issue/PR body plus captured revision, Git commit/path/blob/line
anchor, or verification artifact record key. Line numbers alone are
insufficient unless bound to immutable captured bytes.

## Authority derivation

`authority_class` is not inferred from prose. The source-bundle compiler applies
this ordered table and records the matched rule ID:

| Source evidence | Authority class |
| --- | --- |
| GitHub Issue/PR comment or review whose immutable actor ID is in the fixture's owner allowlist and whose role is `repository_owner` for the observation boundary | `owner_decision` |
| `spec.md` with `status: approved` and non-null approval provenance that resolves to the captured approving owner/reviewer records | `approved_specification` |
| Captured Issue body/state or PR body/state | `canonical_work_item` |
| Commit reachable from the captured base branch whose associated PR is captured as merged | `merged_implementation` |
| Immutable test, ledger, report, or certificate with its producer/source fingerprints | `verification_evidence` |
| External paper/document with immutable edition/revision identity | `external_reference` |
| Ordinary Issue/PR comment, review, repository file, or commit not matching a higher rule | `ordinary_record` |
| Agent-authored proposal or generated design lacking the required approval record | `agent_proposal` |

For F-real the owner allowlist contains GitHub actor ID `102535039` for the
captured observation boundary. Login, display name, author association, or
textual implication alone is insufficient. The bundle captures actor ID,
login-at-observation, author association, and the repository-role evidence used
to establish the allowlist. Issue and PR state are source facts, not owner
decisions. A merged PR establishes merged implementation, not scientific
acceptance. Verification evidence establishes only what its frozen gate says.
External sources retain external authority. Agent proposals remain proposals
until a separately captured approval changes their class in a successor
snapshot.

## Temporal model

Four times remain distinct:

- domain/effective time: `valid_from`/`valid_to`, when the proposition holds;
- source event time: `source_event_at`, when the canonical source event occurred;
- observation time: `observed_at`, when source bytes/state were captured;
- assertion recording time: `recorded_at`, when the assertion was compiled.

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
repository observation boundary, and every selected source revision.

`source_bundle_id` is `cyax-source-bundle-sha256:<digest>` where `digest` is the
canonical framed hash defined below over the three JSONL payloads and the
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
| `assertion_id` | `['cyax-assertion-v1', subject_id, predicate, object_id-or-null, literal_ref-or-null, source_revision_id, source_locator, valid_from, valid_to, origin]` |

The rendered forms are respectively `cyax-entity-sha256:`,
`cyax-source-revision-sha256:`, `cyax-literal-sha256:`, and
`cyax-assertion-sha256:` plus the digest. Namespace is the captured repository
identity or registered external namespace. Rebuild, observation, and recording
times are excluded except where a source event is part of revision identity.
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

The **logical semantic checksum** is SHA-256 of:

```text
frame([
  'cyax-logical-snapshot-v1', schema_version, source_bundle_id,
  authority_rule_version, semantic_evaluator_rule_version,
  ['entities.jsonl', bytes], ['literals.jsonl', bytes],
  ['source_revisions.jsonl', bytes], ['assertions.jsonl', bytes]
])
```

`snapshot_id = cyax-snapshot-sha256:<logical_semantic_checksum>`. Compiler,
curator, and backend implementation versions do not alter logical identity.
The separate `build_contract_checksum` hashes the snapshot ID plus exact
assertion-compiler, curator, validator, and context-compiler versions. This
replaces the ambiguous old `semantic_checksum` name.

`manifest.json` records all named versions and checksums, every payload byte
count/SHA-256/record count, ordered source-revision IDs/fingerprints, and total
entity/assertion/source/literal counts. On open, validators recompute and check
every manifest-derived value, ID, foreign key, enum, predicate signature,
literal constraint, canonical order, payload hash, source-bundle link, logical
checksum, and build-contract checksum. Unknown fields fail under v1.

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
  assertions[]
  source_revisions[]
  paths[]
    path_id
    steps[]
      assertion_id
      direction                 # forward | reverse
  diagnostics                   # object or null
```

Semantic arrays are empty rather than null and are sorted by primary ID.
Records are complete canonical snapshot objects, not backend projections.
`paths` are ordered by the framed sequence of `(assertion_id,direction)`;
steps retain traversal order. `path_id` hashes that sequence. Parallel edges
remain distinct because they have distinct assertion IDs. Exact duplicate step
sequences are invalid adapter output, not silently deduplicated. Thus semantic
path multiplicity is assertion-distinct and every exact path has multiplicity
one. All referenced entities, assertions, and source revisions are included
once. An absent optional value is forbidden in v1; it is represented as null.

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

Each item is compiled into exact Claims/assertions/source anchors and must pass
independent source review before reuse by CYAX-0168 or CYAX-0166. This repair
does not modify CYAX-0166.

### F-scale generator

Generator version `cyax-0168-scale-2.0` is a fully specified, scientifically
inert project-record generator. It uses no runtime RNG. For each random choice,
it evaluates `SHA256(frame(['cyax-gen-2.0', seed, profile_id, purpose,
ordinal, counter]))`; unsigned big-endian digest values feed rejection sampling
to avoid modulo bias. Endpoint candidates are always primary-ID sorted.

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
4. For the lowest-ID `floor(cycle_rate × connected_blocks)` disjoint triples,
   replace three non-chain dependency edges with a directed three-cycle. A
   cycle never changes reachability depth limits and is never interpreted as
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

Instances are selected from generated snapshots before backend implementation.
For each tier/profile/seed, compute degree, reachability, selectivity, and
shortest-path depth with the reference in-memory specification evaluator, not a
backend. Select the smallest entity ID at the nearest-rank 10th, 50th, and 90th
degree percentiles; the smallest IDs at the 10th and 90th result-set-size
percentiles; and smallest-ID pairs with shortest distances `≤4`, `5..8`, and
`≥9` (bounded by profile depth). Ties use canonical ID order. Missing strata
are recorded before implementation and invalidate that fixture rather than
triggering post-result reselection. F-real instances are named directly by the
audited Claim/entity IDs. Query parameters, `as_of`, depth, selected stratum,
gold bundle, and gold checksum are frozen in the fixture manifest.

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

Updates do not mutate snapshot N. Each frozen delta contains base snapshot ID,
ordered added source objects/revisions/entities/literals/assertions, explicit
supersession/tombstone assertions, and expected target snapshot ID:

```text
immutable snapshot N + frozen delta → immutable snapshot N+1
```

S and G receive the same N, delta, and target identity. They may incrementally
update disposable indexes, but their complete logical export must equal N+1.
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

A separate non-decision corpus uses generator version 2.0: 25,000 assertions
with `P-low/P-medium/P-high` seeds `168901/168902/168903`, and 125,000
assertions with seeds `168904/168905/168906` respectively. It is
never reused in F-real or T0–T4. Each backend receives at most eight person-hours
of tuning and 40 executed plan/config trials. Permitted changes are SQLite
index selection/order, documented pragmas, and CTE formulation; or documented
Ladybug index/configuration and equivalent query formulation. Schema semantics,
evaluator, output, query instances, hardware, data, and resource envelopes may
not change. Stop at the earlier of budget exhaustion or five consecutive trials
without ≥2% improvement in the preregistered geometric mean of calibration
latencies while passing correctness. Before decision fixtures are revealed,
hash and commit final schemas, queries, indexes, pragmas/configs, dependency
lock, and explain/query plans where stable. No result-driven tuning follows.

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

## Primary host and measurement protocol

The proposed authority host is dedicated bare-metal Linux x86-64 with at least
64 GiB RAM, local NVMe, ext4 (or a preregistered recorded local filesystem), no
swap, no competing workload, isolated fixed CPU affinity, one benchmark thread,
and frozen governor/turbo/power policy where controllable. macOS ARM64 remains a
smoke host only. Freeze CPU model, microcode, core/SMT layout, RAM, Linux
distribution/kernel, filesystem/mounts, storage device/firmware, Python,
SQLite version and compile options, Ladybug wheel/commit/dependencies, libc and
native runtimes, and measurement-tool versions.

Primary repetitions are:

- clean build/rebuild: 7 independently cloned paired repetitions;
- each snapshot transition batch: 15 independent paired repetitions;
- application-cold/database-open-cold queries: 20 paired fresh-process
  repetitions per query instance;
- warm queries: 10 independent process blocks, each with 5 unmeasured warmups
  and 50 measured calls.

The independent unit is the paired build, transition, fresh process, or warm
process block—not an individual warm call. Alternate first backend with a
frozen balanced schedule. Report raw values, paired differences and ratios,
nearest-rank p50/p95 within blocks, then topology-stratified paired BCa 95%
bootstrap intervals with 10,000 deterministic resamples at the independent
block level. Seeds for bootstrap/order live in the preregistration manifest.
Do not pool tiers, profiles, seeds, query families, or repeated calls as
independent observations. Missing/timeout/resource-breach samples remain in the
evidence and invoke invalid/inconclusive classification; they are not replaced.

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
5. the T4 fixture/workload/hash was frozen before T3 results were revealed.

T4 alone cannot justify graph adoption for current CYAxiverse scale.

## Proposed thresholds and deterministic classifier

These thresholds remain proposals pending owner approval. Speedup is paired
`S p95 / G p95`; absolute saving is `S p95 - G p95`.

**G — experimental derived-index status:** At T3, on at least two preregistered
graph-shaped families and at least two of three topology profiles, require
speedup point estimate ≥2.0, lower 95% CI ≥1.5, and absolute end-to-end saving
≥10 ms. The same family/profile direction at T2 requires point estimate >1.0
and lower 95% CI ≥1.0. All semantic and resource gates must pass.

**Hybrid:** G either meets the preceding rule on two graph-shaped families, or
one preregistered critical impact-analysis family has point estimate ≥5.0,
lower 95% CI ≥3.0, and absolute saving ≥50 ms. Under either alternative, S must
simultaneously have a material relational-family advantage: `G p95 / S p95`
point estimate ≥2.0, lower 95% CI ≥1.5, and ≥10 ms absolute saving. Routing
remains explicit and static. All semantic and resource gates must pass.

**Resource acceptance:** G materialization footprint (data, indexes, and
steady-state auxiliary files) and peak query memory must each be ≤2× S or have
absolute excess ≤2 GiB. G build/rebuild/transition time must be ≤3× S and
always inside the hard envelope. Final context/prompt values must match exactly.

Classifier precedence is:

1. **Inconclusive / invalid execution** for failed host controls, missing or
   corrupted samples, threshold-straddling CIs needed for classification, or
   implementation failure before the common contract is exercised.
2. **Architecture problem** only for a demonstrated common-contract defect:
   frozen-gold/parity failure, non-failing snapshot identity, or nondeterministic
   complete logical export.
3. **Hybrid** when its performance and resource rule passes.
4. **G (experimental derived-index status)** when its rule passes and no material
   S relational advantage requires Hybrid.
5. **Retain S / G not justified** for every other valid execution.

Examples: G 4× faster but 3× larger with >2 GiB excess retains S; a win only at
T4 retains S; a 20× critical-query win yields Hybrid only if its lower CI,
absolute saving, S relational advantage, and resource gates pass; candidate-byte
reduction without end-to-end latency never changes the outcome; and a required
CI overlapping a threshold is Inconclusive. A timeout is preserved as a breach,
not silently rerun into a favorable class.

## Requirements and gates

| Requirement | Required behavior | Verification |
| --- | --- | --- |
| R-001 Common semantics | Closed entities, Claims/literals, enums, predicate signatures, authority and time rules feed one evaluator. | Schema tests and adversarial authority/time/dispute fixtures. |
| R-002 Immutable evidence/identity | Content-addressed source bundle, stable IDs, logical snapshot identity, freshness, and atomic publication fail closed. | Rebuild, collision fixture, tamper, stale, unavailable-source, and crash tests. |
| R-003 Fair materializations | Normalized indexed S and pinned G derive only from the same snapshot. | DDL/index/config/query-plan review and complete-export equality. |
| R-004 Source-correct F-real | Audited K1–K12 distinguish historical evidence from current gates. | Independent source/anchor review before reuse. |
| R-005 Deterministic scale | Generator 2.0 produces the frozen profile/seed matrix exactly. | Invariants and repeated byte/checksum equality. |
| R-006 Frozen workload | Instances and gold precede implementation; query answer/path semantics are exact. | Manifest/gold review and query validator. |
| R-007 RetrievalBundle v1 | Both adapters return complete canonical objects and deterministic directional paths. | Contract/order/dedup/reference tests and gold/S/G equality. |
| R-008 Successor updates | N + delta yields immutable N+1 from independent clones with atomic recovery. | Complete export, crash, rollback, and repetition-isolation tests. |
| R-009 Preregistered measurement | Calibration, host, fairness, cache, repetitions, confidence, and resources are frozen. | Hash manifest, synthetic analysis validation, and independent methodology review. |
| R-010 Deterministic decision | Exact outcomes, resource gates, T4 gate, and ambiguous cases are encoded. | Classifier boundary/table tests and owner approval. |

### CYAX-0168 G0 — repaired design approval

Acceptance requires independent architecture/methodology rereview, independent
K1–K12 source review, and owner approval citing the exact head, backend pin,
host, workload, repetitions, envelope, and thresholds. `approval_ref: null` or
any unresolved normative choice stops implementation.

### CYAX-0168 G1 — frozen inputs and smoke

Acceptance requires frozen source/gold/generator/query/calibration/dependency
manifests and offline S/G clean-build/rebuild/reopen/tamper/crash smoke checks.
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
3. accept the corrected K1–K12 key after independent source review and decide
   when the separate CYAX-0166 copy may be repaired;
4. approve generator 2.0, profile/seed matrix, query-selection/gold rules, and
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
