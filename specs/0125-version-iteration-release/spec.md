---
spec_id: CYAX-0125-Gate-A
title: Package iteration and release lifecycle
issue: 125
class: S2
status: draft
review_state: "prospective review — pending fresh independent SPEC and STANDARDS review"
target_iteration: version-lifecycle-retrofit-2026-09
version_bearing: true
package_infrastructure_impact: patch
package_version_adoption: none
review_required: independent SPEC and STANDARDS
prior_approved_revision: cf0f9c39256b7a58e179af92a46fbf1cb651ff76
prior_approval_refs:
  - "Issue #125 comment 5746793852"
  - "Issue #125 comment 5752268640"
amendment_basis:
  - "manager handoff packet 19fa93e2b4b1439b1cc4217cf7ff9e42846980d46df66793156f72789894e40a"
  - "handoff-review result a88d7dde2f19b867152ad0bd1b858bf1aaa6008abddae61df1a316fafc3da0bf"
  - "owner-dispatch receipt 140d1fe1c7c09db3d40a04c66c1b258f18e8c42025a3453c2472ed682124637c"
  - "target PR head 20b3935ace0e01fcee2808681c56595e3afa7667"
approval_refs: []
review_records: []
review_rubric_sha256: 418f2d5a276cbdb74b8ad331b532d59219ef33b4e9fabc2b3d4a21c55bc06c72
---

# CYAX-0125 Gate A — package iteration and release lifecycle (prospective amendment)

## Objective and authority

This prospective amendment reduces the mutable lifecycle authority before a
new review. Gate A establishes the forward machinery for package iterations,
global version allocation, exact-tree certification and public release using
protected create-once Git refs and small immutable evidence manifests. It does
not use that machinery for a real closure, development-version transition,
public release or historical reconstruction. The actual `Project.toml` version
remains `0.2.0`.

The previously approved revision and its implementation/review evidence are
historical predecessor records. They describe the earlier append-only
`release-events` design and must not be read as evidence that this reduced
candidate was implemented, reviewed, approved or historically operative. If
this amendment is approved, it supersedes the earlier mutable-authority rules
prospectively from the adoption boundary; it does not rewrite predecessor
history or introduce the new requirement IDs into that history.

The owner-approved [base lifecycle decision](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/125#issuecomment-5746793852)
and [reservation decision](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/125#issuecomment-5752268640)
govern this retrofit. The later base decision prospectively supersedes earlier
Issue #125 statements that confined package bumps and documentation source
changes to a `vmm → main` release PR or predetermined `0.3.0` as the next
final. Those older comments remain historical evidence. `AGENTS.md` and
applicable normative skills remain repository authority until the replacement
rules merge. This candidate is not Approved and has no new approval
provenance; fresh independent SPEC and STANDARDS review under the canonical
rubric is required before it can govern implementation.

The target iteration is `version-lifecycle-retrofit-2026-09`. This is
version-bearing package infrastructure with **patch** impact: it changes
installation, documentation and release guarantees without intentionally
changing a public Julia API, scientific result, persisted scientific schema or
supported scientific convention. The impact is a declaration for later
aggregation; Gate A makes no package-version adoption bump.

## Scope and limits

Gate A may update the repository policy, SDD/release workflow, release-neutral
README and documentation, version checker, static iteration schema, protected
create-once lifecycle refs, immutable evidence-manifest validators, focused
fixtures and tests, and minimal workflow checks. Each changed path must map to
a requirement below. Normative policy promotion takes effect only after this
amendment and the implementation receive the required fresh review and merge.

Gate A excludes historical version-ladder selection, real retrospective
entries, Gate B reconstruction or adoption, any `Project.toml` change from
`0.2.0`, a production `-DEV` transition, current `vmm → main` reconciliation,
an actual public tag or release, scientific/numerical reinterpretation, changes
to persisted scientific schemas, rewriting Issue #172 evidence, and any change
to the legacy `v-0.1` tag. Existing work retains historical declarations;
Issue #172 remains grandfathered. Gate A may create isolated temporary refs and
synthetic manifests in fixture repositories only; it creates no production
lifecycle ref, claim, closure, candidate, tag, release or publication.

The prospective principal model is: `vmm` carries principal package-development
iterations; `main` carries the latest certified principal release; certified
maintenance releases live on `maintenance/X.Y` lineage with immutable public
SemVer tags and do not move a newer `main` backward. A target iteration has a
stable ID before its final version is assigned. Closing an iteration and
publicly releasing it are independent operations.

Gate A requires automation for the first principal lifecycle path:
deterministic principal allocation/reservation, closure and immutable anchor
correspondence, candidate and exact-tree certification, protected canonical tag
creation, and release/publication evidence reconciliation. Maintenance-line
bootstrap and release automation, and rare recovery automation for ambiguous
or exceptional states, are deferred to a later approved S2 gate. Deferral does
not weaken the invariants: unsupported or unproven operations fail closed and
preserve the affected identity as unavailable.

## Normative requirements

The numbered requirements below are the acceptance contract. Later plan,
tasks, implementation, tests and review evidence must account for every one.

### R-001 — Authority promotion

The two durable Issue #125 decisions authorize normative promotion. Before
merge, this specification cannot override checked-in `AGENTS.md` or skills.

### R-002 — Principal roles

`vmm` carries principal development iterations. `main` carries only the
latest certified principal release; it never carries `-DEV`.

### R-003 — Maintenance releases

Maintenance releases are first-class exact-tree certified releases on their
declared `X.Y` lineage. They do not roll a newer `main` backward.

### R-004 — Target iteration

Work records a stable target-iteration identity before final SemVer assignment.

### R-005 — SemVer and prerelease parsing

Version validation uses Julia `VersionNumber` semantics or a demonstrated
equivalent. It distinguishes final `X.Y.Z` from approved `X.Y.Z-DEV`;
the prerelease suffix alone does not make a valid development version invalid.
The governed package grammar admits only those two exact forms, with three
canonical nonnegative decimal components (`0|[1-9][0-9]*`) and no build
metadata. The raw version string must equal its canonical rendering; reject
leading-zero aliases even when Julia normalizes them. Other prereleases,
including `-alpha` and `-rc1`, are rejected even if Julia can parse them.
Canonical future public tags use exactly `vX.Y.Z` with the same component
grammar and raw-string equality.

### R-006 — Deterministic principal development identity

After principal `X.Y.Z` closes, the exact next sentinel is
`X.Y.(Z+1)-DEV`. It reserves final `X.Y.(Z+1)` but does not predetermine
the next closure's reviewed aggregate impact. If that exact sentinel is
unavailable, return `PRINCIPAL_SENTINEL_UNAVAILABLE`; do not skip it.
The machine-visible result is `status = BLOCKED` with
`reason_code = PRINCIPAL_SENTINEL_UNAVAILABLE`.

### R-007 — Deterministic maintenance development identity

After `maintenance/X.Y` closes `X.Y.Z`, use the lowest globally AVAILABLE
`X.Y.N` with `N>Z` and reopen as `X.Y.N-DEV`. Do not leave the `X.Y` line.

### R-008 — Public release from DEV forbidden

A governed `-DEV` state cannot be a closed iteration, `main`, a canonical
public `vX.Y.Z` tag, or a public release.

### R-009 — Principal regression forbidden

Principal final versions and the package version on `main` never decrease.

### R-010 — Global DEV reservation

Active `X.Y.Z-DEV` reserves final `X.Y.Z` globally for its owner line.
Competing lines cannot close, candidate or publish it.

### R-011 — Reservation consumption and no reuse

Closing the reserved final consumes its reservation into that final. If its
owner closes a different final, the unused reserved final becomes permanently
`CONSUMED_UNUSED_DEV_RESERVATION`. No closed, candidate, withdrawn, released
or consumed version may be allocated again. The existing narrow exception is
preserved: a reservation that is durably proven to have aborted before the
`-DEV` state was entered may release its not-yet-claimed final version, but
the reservation manifest/ref identity itself is permanently consumed and
cannot be reused; an uncertain outcome remains unavailable.

### R-012 — Static iteration authority

`iterations.toml` and validated immutable iteration identities describe
static version occupation and designation.

### R-013 — Protected create-once lifecycle authority

The canonical lifecycle authority is the union of the freshly resolved static
`iterations.toml` source, immutable `iterations/X.Y.Z` anchors, protected
canonical public tags, and a namespace of protected create-once lifecycle refs.
Each lifecycle ref points to a commit containing one small immutable canonical
evidence manifest. The canonical namespaces are claims/reservations,
candidate and intent identities, terminal release identities and publication
identities; the exact ref and manifest schemas are
versioned and validated. A ref is created at most once, cannot be deleted or
repointed, and is never force-updated. There is no mutable `release-events`
branch or append-only release-events stream in the prospective model, and no
second mutable ledger or unverified ref set is valid. The canonical object
types are distinct: `claim`, `reservation` (prepared/opened or terminal
abort/consume), `candidate`, `intent`, `release`, and `publication`
manifests. Each type has its own protected create-once ref namespace and
predecessor binding; one type cannot stand in for another. Publication
evidence is immutable and content-addressed, and its protected publication ref
is required and likewise create-once after GitHub Release publication.

### R-014 — Combined allocation view

`GLOBAL_ALLOCATION_VIEW` combines one exact `static_iteration_snapshot` with
one exact `lifecycle_ref_snapshot` over all protected lifecycle refs and their
manifest bytes. Only versions unoccupied in both views and permitted by line
rules are AVAILABLE. The lifecycle snapshot is replayable from protected refs,
their resolved commits/trees and manifest digests; a partial, stale, or
cross-repository snapshot is rejected.

### R-015 — Static namespace occupancy

Validated retrospective designations, closed prospective designations,
protected `iterations/*` anchors and canonical public SemVer tags occupy
their versions, including closed but unreleased iterations.

### R-016 — Lifecycle namespace occupancy

Prepared/open DEV reservations, terminal reservation consumption, maintenance
line state, candidates, withdrawals, intents and releases determine mutable
occupation through immutable manifests and their protected refs. A prepared
reservation temporarily makes its intended final unavailable. A claim ref for
a final version is created when the reservation enters `-DEV` or a candidate/
release path claims the version and permanently occupies it, even when a later
manifest records withdrawal or `CONSUMED_UNUSED_DEV_RESERVATION`. A proven
pre-entry abort has no claim ref and may release only that final version under
the recovery rule in R-011. No deleted, repointed or unreferenced local copy
can make a claimed version AVAILABLE again.

### R-017 — Create-once allocation serialization

An allocation-changing transaction binds a transaction ID, exact static
snapshot, exact lifecycle-ref snapshot, manifest ID, canonical manifest bytes,
deterministic target ref name and expected absence of that ref. It revalidates
both snapshots and creates a descendant commit plus the protected ref with a
non-force create-if-absent/CAS operation. Stale decisions refresh and
recompute. If the ref already contains the exact transaction and manifest,
the operation is idempotent success; if it contains any other bytes or
identity, the result is INVALID and the claimed version remains unavailable.
After an uncertain remote create, the writer re-reads the protected ref and
manifest. It may retry only the same canonical create against the same absent
ref; if remote evidence cannot classify the result, return `status = BLOCKED`
with `reason_code = CREATE_ONCE_OUTCOME_UNCERTAIN` and retain the affected
line/version freeze. It must never append, repoint, delete, or release a
reservation to resolve uncertainty. Opening a prepared reservation and
creating its permanent version claim, when required, occur under the same
exclusion boundary; a proven pre-entry abort creates only its terminal abort
manifest and does not create the claim ref.

### R-018 — Static/create-once mutation serialization

Prospective closure/materialization, future retrospective designation,
iteration-anchor creation, lifecycle-ref creation and canonical public-tag
creation participate in the same governed allocation exclusion boundary or
prove equivalent exclusion. The boundary protects the static snapshot and the
complete lifecycle-ref snapshot together; a check that is not held through
ref creation is insufficient. If the required exclusion or protection cannot
be established, the operation fails closed.

### R-019 — Immutable prospective closure identity

Every prospective closed iteration has one protected annotated
`iterations/X.Y.Z` ref identifying the exact closure commit/tree.

### R-020 — Prospective static metadata

Prospective metadata contains a stable iteration ID, final version,
`kind=prospective`, release line, aggregate impact, anchor ref and
contributing issue/spec/PR identities with roles. It contains no self SHA/tree,
mutable release state, future release identity or guessed closure time.

### R-021 — Retrospective historical truth

The schema supports later Gate B designations with historical declared
version, whether `Project.toml` actually carried it, historical release
status, and historical anchor SHA/tree. Gate A validates synthetic fixtures
only and populates no real retrospective entries.

### R-022 — Release-neutral source

Tracked content remains truthful on `vmm`, iteration/candidate refs,
maintenance lines, `main` and public tags. Installation docs distinguish
immutable release-tag use from active `vmm` use. Documenter source requires no
release-time edit; deployment derives from verified ref/manifest/environment while
preserving generated API, development and stable/versioned docs.
The `vmm` ref deploys the development channel. A canonical principal public
tag deploys its immutable versioned channel and may advance `stable` only
when its certified release commit equals the current principal `main`.
A canonical maintenance public tag deploys its immutable versioned channel
without changing `stable`. Deployments run only for those verified refs;
the channel is selected by ref, canonical manifest and environment, not by a
tracked source edit. The documentation verifier resolves the selected release
or publication manifest from the protected ref set and independently checks
its tag, commit/tree, package version and canonical evidence digest; missing,
ambiguous or mismatched evidence blocks deployment.

### R-023 — Exact-tree certification

The separately pinned certification harness tests an immutable package
checkout and proves its tracked tree is unchanged and matches the closed
iteration tree. Candidate and release tree equality is verified.

### R-024 — Separate certification identity

Certification evidence records independently pinned package, policy, harness,
environment and evidence identities, and states whether the certified subject
is tree-bound or commit-bound. Gate A accepts only those two binding models.
Any other value returns `status = BLOCKED` with
`reason_code = UNSUPPORTED_CERTIFICATION_BINDING` before public-tag creation;
a future reviewed spec amendment must define an additional model's subject,
equality proof, transfer rule and release gate before it can be used.

### R-025 — Single candidate lifecycle

Principal and maintenance use one sequence: remotely durable exact candidate
ref and candidate manifest, equality check, candidate-opened lifecycle ref,
certification, line-specific pre-tag checks, immutable public tag, released
manifest/ref, and GitHub Release. No duplicate candidate opening or
certification path is allowed. Before the tag, create one durable intent
manifest/ref binding candidate, certification, final commit/tree, version and
proposed tag. After tag creation, that intent and the protected tag establish
the nonterminal `tag_reconciliation_pending` state until the matching released
manifest exists. All refs are create-once and all manifests are immutable.

### R-026 — Principal exact-tree promotion

A principal release records candidate-time and freeze-time `main` identities,
mechanically freezes `main`, dispositions intervening ancestry, reconciles to
the certified closed tree, checks final version greater than prior `main`
and no `-DEV`, then applies certification-transfer rules before tagging.
The frozen candidate-to-main interval enumerates every intervening commit.
Allowed dispositions are `no_drift`, `tree_neutral_included` (the intervening
commits' net tree equals the certified tree), and `content_drift_blocked`.
Content drift blocks promotion until a new reviewed closure/candidate and
certification make the exact final tree valid. The evidence records the
intervening SHAs, tree comparisons and disposition; no commit is silently
dropped from the release decision.

### R-027 — Maintenance exact-tree release

Maintenance candidate, certification and tag remain in the declared lineage.
The `released` manifest records contemporaneous unchanged principal `main`
SHA/version. Maintenance metadata is not forward-ported to `vmm`.

### R-028 — Protected closure transaction

Freeze the owner line; bind static and lifecycle-ref snapshots; verify and merge
the reviewed final closure; create/verify the annotated iteration anchor and
explicit closure time; create the immutable closure/consumption manifests and
refs; then select, prepare, reopen and activate the deterministic next DEV
identity through a new create-once reservation ref. No unrelated merge
interleaves. Unfreeze only after full correspondence.

### R-029 — Outgoing reservation terminality

If closure/anchor succeeds but the immutable outgoing-consumption manifest/ref
cannot be created or verified, keep the line frozen and reconcile forward. Do
not allocate, prepare or reopen next DEV. If reconciliation cannot complete,
return `OUTGOING_RESERVATION_RECONCILIATION_FAILED` and retain the affected
version as unavailable.

### R-030 — Initial maintenance bootstrap (deferred S2 automation)

The later approved S2 maintenance gate shall resolve an approved immutable
`X.Y` base; prove line absence; bind the static and lifecycle-ref snapshots;
select the lowest AVAILABLE same-line patch; and create the maintenance-line,
reservation and bootstrap manifests/refs with create-if-absent/CAS and
immediate protection before installing `-DEV`. Gate A preserves and validates
this contract but does not claim to implement production maintenance bootstrap
automation.

### R-031 — Maintenance bootstrap correspondence (deferred S2 automation)

No maintenance work begins until base, branch, DEV head, owner line,
reservation ref and bootstrap manifest correspond exactly. Uncertain creation
or failed activation keeps the version unavailable and the line frozen under
the later gate's recovery rules. Gate A does not introduce a mutable
line-open event or a production maintenance recovery path.

### R-032 — Candidate durability

Before the candidate-opened manifest/ref, the exact candidate ref/SHA/tree is
remotely durable. Active and withdrawn candidates remain auditable; a released
candidate remains resolvable through the release evidence chain. Do not
delete its sole durable terminal reference.

### R-033 — Withdrawal and no reuse

Pre-tag failure may withdraw a candidate. Withdrawn version and candidate
identity remain permanently consumed and retained for audit.

### R-034 — Irreversible public-tag boundary

After protected canonical public tag creation, do not withdraw, delete,
repoint or reuse. Reconcile missing manifest/publication forward.
If the tag exists with a matching durable intent but no `released` manifest,
validation returns `tag_reconciliation_pending`; the writer creates only the
matching release ref/manifest after rechecking tag, commit, tree and
certification.
If any identity disagrees, validation is INVALID and no replacement tag is
created.

### R-035 — Public-tag immutability

Mechanically protect future canonical `vMAJOR.MINOR.PATCH` tags against
deletion, repointing and unauthorized creation without changing legacy
`v-0.1`. Return `PUBLIC_TAG_RULESET_UNAVAILABLE` if safe enforcement fails.

### R-036 — Transition and grandfathering

Pre-retrofit work keeps its historical declarations/evidence. Issue #172
remains grandfathered; the first prospective pilot needs new meaningful work
opened after Gate A becomes authoritative.

### R-037 — Package-infrastructure SemVer boundary

Assess package-facing installation/support, dependencies/extensions,
build/test/audit/release guarantees, reader/writer compatibility,
reproducibility and necessary public documentation for version impact.
Agent orchestration, SDD/project mechanics, privacy/lessons and internal
governance are normally neutral absent package delivery/behavior change.
Do not infer impact from paths alone.

### R-038 — Gate A/Gate B separation

Gate A implements and tests machinery in isolated fixtures without actual
adoption, historical population, real `-DEV`, closure, release or public tag,
and creates no production lifecycle refs/manifests. Its required automation
scope is the first principal lifecycle path; maintenance-line bootstrap/
release and rare recovery automation remain later approved S2 work.

### R-039 — Exact final-state review

After authorized merge and live protection convergence, fresh independent
review examines merged source, actual refs/rulesets/freeze controls,
create-once lifecycle-ref authority and final verification. Gate A is not
complete before reconciliation of that review.

### R-040 — Canonical immutable-manifest identity

Every lifecycle manifest has a content-derived `manifest_id`, `manifest_type`,
explicit schema version and schema-valid RFC3339 UTC `timestamp_utc`, plus a
transaction ID where applicable. `manifest_id` is
`LIF-SHA256-<64-lowercase-hex>`, where the suffix is SHA-256 of the canonical
manifest identity preimage containing every manifest field except
`manifest_id`. Readers recompute it before trusting the manifest. There is no
global sequence counter, next-ID allocator, or ordering authority hidden in
manifest identity. A successful protected create permanently consumes its ref
and identity; an unpublished failed create does not. Reject missing,
malformed, duplicate or conflicting IDs and manifests. A manifest is accepted
only when its exact bytes, commit/tree, ref name and referenced predecessor
identities agree.

### R-041 — Canonical closure UTC timestamp

Closure uses one explicit schema-valid RFC3339 UTC
`closure_timestamp_utc` bound to the annotated iteration ref and durable
closure evidence. It is never inferred from UI, commit, filesystem or local
prose time. Repeated canonical copies must agree exactly.

### R-042 — Durable release-manifest identity

A principal `released` manifest contains manifest ID/type/UTC time, final
version, principal line, anchor ref/SHA/tree, durable candidate ref/SHA/tree,
final release SHA/tree, frozen pre-integration `previous_main_sha/version`,
verified post-reconciliation `main_at_release_sha/version`, certification
binding/subject SHA/tree/policy revision/harness revision/environment/evidence
refs, closure UTC time, public tag and evidence refs. Maintenance contains the
same applicable identities and contemporaneous unchanged principal
`main_at_release_sha/version`, but no principal previous-main claim. Larger
immutable evidence may be content-addressed and referenced by the manifest.
Terminal output alone is insufficient.

The canonical `released` manifest/ref precedes GitHub Release publication and
is never mutated to add its later ID. A separate immutable publication
manifest/ref, keyed by the released manifest ID and public tag, records the
GitHub Release identity, publication time and evidence after publication. The
publication manifest has `manifest_type = publication`, its own
content-derived `manifest_id`, the exact `released_manifest_ref/id/digest`,
canonical `public_tag` and resolved tag commit/tree, `github_release_id`,
sanitized `github_release_url`, `published_at_utc`,
`publication_evidence_ref/digest`, and owner-authorization identity. Its
`publication_id` is deterministic: `pub-` followed by the lowercase SHA-256
of canonical compact JSON with exactly the sorted keys
`{\"public_tag\":\"<canonical-tag>\",\"released_manifest_id\":\"<id>\"}`;
the protected ref is exactly
`refs/heads/lifecycle/v1/publications/vX.Y.Z/<publication_id>`. A
publication is valid only when this key, released manifest, canonical tag,
GitHub Release identity and evidence digest agree. Its
content-addressed bytes are independently verified. All
durable public manifest and evidence field values, including certification
environment data and references, are sanitized public values, approved
identities, repository-relative paths or content-addressed digests. They must
not contain private conversation or local-machine locators, credentials, or
raw private environment values. The publication boundary in `AGENTS.md`
applies before any manifest or evidence artifact becomes durable on GitHub.

### R-043 — Bidirectional release consistency

Terminal validation requires every canonical public release tag to resolve to
exactly one matching released manifest/version/line/commit/tree/certified
iteration tree; every GitHub Release to resolve through its canonical tag to
that manifest and its matching publication manifest; and every released
manifest to resolve back to its immutable tag, release commit/tree and
certified anchor/tree. Mismatch is INVALID.
This bidirectional contract applies to post-retrofit canonical releases;
legacy `v-0.1` and its historical publication remain grandfathered under
R-036 rather than being reinterpreted as a future canonical release.
`publication_reconciliation_pending` is an explicit nonterminal forward
recovery state, never silently accepted as `terminal_consistent`.
`tag_reconciliation_pending` is likewise nonterminal and requires durable
intent plus a matching immutable tag. Neither state authorizes another
candidate, version allocation or public release for the affected identity.
Exactly one publication manifest/ref may bind a given pair
(`released_manifest_id`, `public_tag`) and each GitHub Release identity may
bind exactly one such pair. A second publication ref for the same pair, a
publication with a different released manifest, tag, tag target, GitHub
Release identity or evidence digest, or a duplicate key under a different
ref is INVALID. If the expected publication ref is absent after release
publication, the state is `publication_reconciliation_pending`; a matching
create-once retry is idempotent, while a conflicting or uncertain create
remains BLOCKED and frozen.

### R-044 — Certification binding and transfer

Tree-bound certification may transfer from candidate to different final
commit only with durable evidence that certified subject, candidate, final
release and anchor trees are equal. Commit-bound certification requires a
new certification of a different final release commit before tag creation.
No implicit transfer is permitted. Unrecognized binding models are blocked
under R-024; they cannot inherit either transfer rule.

### R-045 — Control-plane tooling and boundary preservation

Python is bounded repository/CI control-plane tooling for canonical version
parsing, protected-ref/manifest inspection, documentation routing and release
evidence checks. It is not a runtime dependency of the Julia package, and
`using CYAxiverse` must remain operable when Python, PyCall/CYTools and their
scientific environments are unavailable. This amendment does not change the
Julia API, scientific behavior, persisted scientific schemas, package-version
grammar, version-source boundary, or any Gate A/Gate B boundary. Any such
change requires a separately reviewed specification decision.

## Concrete static and immutable lifecycle authorities

`canonical_static_iteration_source` is the freshly resolved
`refs/heads/vmm:iterations.toml` in this repository. Every allocator resolves
that ref to an exact commit/tree and hashes the file bytes; no caller may
substitute an arbitrary historical copy. The root registry carries principal
prospective entries and, after Gate B only, reviewed retrospective entries.
If that exact selector cannot be resolved and validated, return
`status = BLOCKED` with
`reason_code = STATIC_AUTHORITY_SELECTOR_UNRESOLVED`.
For maintenance, the protected `iterations/X.Y.Z` anchor tree carries its
own static entry. The snapshot validates all protected iteration refs and
their exact anchor-tree entries; maintenance version metadata is not copied
into `vmm`. The snapshot also includes sorted canonical public SemVer tags,
their commit bindings and digests, the derived occupied set and a digest over
the canonical serialization. The legacy `v-0.1` tag is preserved and excluded
from canonical future tag parsing. Existing principal `main` and
pre-adoption `vmm` package versions remain unavailable as historical/current
identities until Gate B provides fuller designations.

Every `static_iteration_snapshot` stores the canonical source repository/ref,
resolved source commit and tree, exact `iterations.toml` SHA-256 content digest,
sorted validated `iterations/*` ref-to-commit/tree bindings and their
canonical ref-set digest, sorted canonical public tag-to-commit bindings and
their canonical tag-set digest, derived occupied version set and overall
snapshot digest. The snapshot is replayable from these identities. If any
source, anchor or tag changes before a mutation commits, the bound snapshot
is stale and the transaction refreshes or blocks.

Digest preimages are exact, noncircular canonical bytes. The file digest is
SHA-256 of raw `iterations.toml` bytes. `ref_set_digest` is SHA-256 of the
standalone canonical JSON array of objects `{ref, commit, tree}`, sorted by
the `ref` string's ascending ASCII bytes, for every validated `iterations/*`
binding. `tag_set_digest` is SHA-256 of the standalone canonical JSON array
of objects `{tag, commit, tree}`, sorted by the `tag` string's ascending ASCII
bytes, for every canonical future public tag. Duplicate `ref` or `tag`
identities are rejected. Empty sets hash canonical `[]`.
The `snapshot_digest` is SHA-256 of standalone canonical JSON for a snapshot
object containing `snapshot_schema_version = 1`, source repository/ref/commit/
tree, file digest, both complete sorted binding arrays, both verified set
digests, and the sorted occupied version array. That object excludes
`snapshot_digest` itself. The occupied version array is sorted by canonical
version string's ascending ASCII bytes and rejects duplicates. Set digests
exclude their own digest fields from their preimages but are included in the
overall snapshot preimage. Every reader recomputes and compares all four
digests: raw file, ref set, tag set and overall snapshot. Any mismatch is
rejected before relying on an allocation view.

Every `lifecycle_ref_snapshot` stores the canonical source repository, sorted
validated `refs/heads/lifecycle/*` ref-to-commit/tree/manifest-digest bindings,
the complete `lifecycle_ref_set_digest`, derived occupied version set and an
overall lifecycle snapshot digest. Its canonical preimage is independent of
the static snapshot and excludes its own digest fields. Readers re-resolve
every protected lifecycle ref, verify the manifest bytes and predecessor
identities, recompute both lifecycle digests, and reject a missing, duplicate,
repointed, deleted or cross-repository ref. The combined allocation view binds
both snapshot digests and is stale if either source changes before create-once
commit.

Protected lifecycle refs are created in the canonical repository under these
version-1 namespaces: `refs/heads/lifecycle/v1/claims/vX.Y.Z`,
`refs/heads/lifecycle/v1/reservations/<owner-line-id>/vX.Y.Z-DEV/<manifest-id>`,
`refs/heads/lifecycle/v1/candidates/vX.Y.Z/<candidate-id>`,
`refs/heads/lifecycle/v1/intents/vX.Y.Z/<candidate-id>/<manifest-id>`,
`refs/heads/lifecycle/v1/releases/vX.Y.Z`, and
`refs/heads/lifecycle/v1/publications/vX.Y.Z/<publication-id>`.
`owner-line-id`, `candidate-id`, and `publication-id` have explicit canonical
ASCII grammars in the versioned manifest schema and cannot contain `/`, `..`,
Git ref metacharacters, aliases, or noncanonical encodings. A transition that
needs a new reservation or intent record creates a new ref distinguished by
its content-derived manifest ID; terminal release and claim refs remain
singletons keyed by final version. Each ref points to a commit whose tree
contains exactly one file, `manifest.json`, holding the canonical manifest.
A manifest
contains only the transition identity, canonical version/line, transaction
and predecessor refs, exact Git SHA/tree identities, required timestamps,
owner authorization identity and content-addressed evidence references.
State progression is the immutable graph formed by each manifest's typed
predecessor refs and exact content digests; a later manifest records a new
state without editing its predecessors. State is derived by replaying the
complete protected ref set; no mutable head, append-only stream or local cache
is authoritative.

The manifest vocabulary includes version-claimed, reservation-prepared,
reservation-opened, reservation-aborted, reservation-consumed,
candidate-opened, candidate-withdrawn, release-intent-prepared,
release-intent-aborted, released and publication manifests. A later manifest
can record a transition but cannot amend an earlier manifest. An intent binds
the exact candidate, certified subject, final version/commit/tree and proposed
public tag before tag creation. A withdrawal or intent abort is allowed only
after proving under the serialized exclusion boundary that the public tag does
not exist. Type-specific schemas require and forbid fields according to their
transition. One owner-line DEV identity corresponds to one active reservation
and actual line head. A prepared reservation is globally unavailable. Abort is
allowed only with proof the matching DEV state was never entered; uncertain
creates remain unavailable and frozen.

Every manifest has `schema_version = 1` and the common fields in R-040.
Allocation manifests also require `transaction_id`,
`static_iteration_snapshot` digest and `lifecycle_ref_snapshot` digest.
Version-claimed manifests bind the final version and exact predecessor that
first made it permanently unavailable. Reservation manifests bind owner line,
final version and intended DEV;
prepared additionally binds expected line head, opened binds actual DEV head,
aborted binds definite non-entry evidence, and consumed binds closure anchor
and terminal disposition. Candidate-opened binds durable candidate ref/SHA/tree,
final version, release line and anchor; candidate-withdrawn binds its
predecessor and withdrawal evidence. Release-intent-prepared binds candidate,
certified subject/evidence, final release SHA/tree, version, line and proposed
tag; release-intent-aborted binds its predecessor and proof that no public tag
exists. Released requires the R-042 identity set and matching intent/tag.
Type-specific manifests reject unrelated fields, duplicate or nonexistent
predecessors and terminal-to-active reversals. A publication manifest
additionally requires exactly `publication_id`, `released_manifest_ref`,
`released_manifest_id`, `released_manifest_digest`, `public_tag`, `tag_commit`,
`tag_tree`, `github_release_id`, sanitized `github_release_url`,
`published_at_utc`, `publication_evidence_ref`,
`publication_evidence_digest`, and `owner_authorization`; its predecessor set
is exactly one released-manifest ref. It forbids reservation, candidate,
intent, closure, certification and `previous_main_*` fields, and it is accepted
only after the matching released manifest and canonical tag exist and the
GitHub Release identity/evidence is independently observed. It is accepted
only when its deterministic publication ID/ref key and every bound identity
agree. A missing required field, extra forbidden field, second predecessor,
duplicate pair or conflicting identity is INVALID. The validator publishes a
versioned schema with these exact required/forbidden fields and transition
predicates; the protected ref set records the schema for replay.

Canonical snapshots, manifests and evidence identities use UTF-8 with ASCII
printable wire strings, lexicographically sorted object keys, compact JSON
without insignificant whitespace, decimal integers without leading zeroes,
and no floating-point values. Strings escape only JSON-required quote and
backslash characters; slash is not escaped. Each standalone manifest has no
trailing LF.
UTC timestamps use `YYYY-MM-DDTHH:MM:SSZ` and are validated as real calendar
times. Digests are lowercase hexadecimal SHA-256 of these exact bytes.
Duplicate keys, malformed UTF-8, noncanonical encodings and nonconforming
timestamps are rejected. Snapshot ref bindings, tag bindings and occupied
versions use their explicit ASCII identity sort keys above.
Every array is declared by its schema as ordered or set-valued: ordered arrays
retain their stated lifecycle/evidence order; set-valued arrays sort by each
element's complete canonical JSON bytes and reject duplicates. No undeclared
array is accepted. The three named snapshot arrays use their explicit sort
keys instead of the generic set-array rule. Optional fields are omitted when
inapplicable; explicit
`null` is rejected unless the field schema requires it. A standalone canonical
JSON artifact has no trailing LF. For non-JSON evidence, the digest covers
exact raw bytes, while the evidence identity separately records its media type;
no normalization or inferred framing is permitted.

## Protected transactions and recovery

Closure and reopen follow R-028 in order. The outgoing reservation must become
terminal before any next allocation; if an owner closes a different final,
its old reserved final remains permanently consumed. The required principal
sentinel is exact `X.Y.(Z+1)`; maintenance searches only upward within its
`X.Y` patch line. A prepared next reservation binds static snapshot,
lifecycle-ref snapshot, transaction, manifest ID, reserved final, intended DEV,
owner line and expected line head. Reopen changes only the governed line's
`Project.toml` to that DEV identity; activation binds the actual reopened
head. On definite non-entry, create an abort manifest with evidence. On
uncertainty or activation failure, keep the version unavailable and reconcile
forward without deleting or repointing a ref.

Initial maintenance bootstrap follows R-030 before permitting work. A
create-if-absent maintenance/bootstrap ref transition from the approved
immutable base and an effective bootstrap freeze prevent unrelated pushes
before DEV/reservation/line convergence. Definite ref non-creation permits a
proven abort manifest; uncertain creation blocks. A created ref with failed
DEV install remains restricted; failed activation requires the later S2
recovery path.

A candidate is created and made remotely durable before its candidate-opened
manifest/ref.
Certification uses a separately pinned reviewed policy and harness and
verifies the package tree unchanged. Principal integration freezes `main`,
dispositions ancestry drift, yields the certified anchor tree, verifies
monotonic final version, and applies R-044. Maintenance verifies lineage and
preserves contemporaneous `main`. Public tag creation is irreversible;
subsequent failures reconcile manifests and publication forward. The released
manifest's `previous_main_*` means frozen pre-integration principal `main`;
`main_at_release_*` means verified post-reconciliation principal release
`main`, or contemporaneous unchanged principal `main` for maintenance.
For `kind=prospective` records, at closure, candidate, final release and
certified anchor trees, parsed
`Project.toml` must equal the recorded final `X.Y.Z` exactly. Principal
`main` after promotion must carry the same final version; a maintenance
public tag must resolve to the matching package version without moving
principal `main`. A mismatch is INVALID before tag creation and during
terminal validation.
Gate B retrospective records instead validate the recorded historical
declared version and the separate fact of what `Project.toml` actually
carried. Historical mismatch is preserved as truth, not rewritten to pass
prospective equality.

For each create-once operation, the writer first checks whether the exact
transaction ID and manifest ID already resolve through the same protected ref
with the same canonical payload. A match is idempotent success; the same ID
with different bytes or a different ref is INVALID. After an uncertain create,
it re-reads the protected ref, commit/tree and manifest bytes. An absent ref
permits retry only of the same canonical create; a conflicting or unclassifiable
result keeps the transaction blocked and frozen under R-017.

## Gate order, evidence and stop rules

1. Verify fresh remote refs, the owner-dispatch receipt, applicable
   skills/lessons and the pinned canonical Spec/Standards rubric.
2. Freeze the exact combined prospective authority set: `AGENTS.md`, the
   integration-release skill, `spec.md`, `plan.md`, and `tasks.md`. Obtain
   independent SPEC and STANDARDS review of that exact set, including
   authority-order consistency and requirement-to-implementation/test
   coverage. Re-review any changed normative bytes.
3. After both axes pass, verify the durable owner-dispatch receipt and add
   approval metadata to `spec.md` that pins the exact reviewed normative
   content identities and both review records. This metadata-only change may
   not alter normative text; any substantive change returns to step 2.
4. Implement A1–A6 on one branch, then converge spec, plan, tasks, source,
   tests and PR evidence. Keep actual `Project.toml=0.2.0`.
5. Obtain a fresh independent read-only exact-candidate pre-merge review.
   Its verdict does not authorize merge. Merge requires an owner decision.
6. After authorized merge, establish and verify actual branch/ref/tag
   protections, freeze controls and create-once lifecycle-ref authority without weakening
   existing `main` protection. Record ruleset/protection IDs, target patterns,
   UTC retrieval times, canonical snapshots, cryptographic digests and
   observed enforcement.
7. Obtain fresh independent closure review of merged source, live protections,
   refs, lifecycle manifests and final evidence. Gate A completes only after that
   review is reconciled.

Tests must cover final/prerelease parsing, the principal sentinel, maintenance
line/version grammar needed to preserve future compatibility, no reuse,
static/ref races, concurrent create-once writers, content-derived manifest
identity/timestamps, principal reservation/closure recovery, durable candidate
lifecycle, certification transfer, complete principal and publication evidence,
publication-key derivation, bidirectional tag/manifest/tree consistency and
post-tag forward recovery. Maintenance
bootstrap/release automation and rare recovery are not Gate A PASS criteria.
Run focused tests before package, audit, docs, Python-free
import, workflow and remote CI checks. Record exact commands, results and
unavailable checks; unobserved checks are not PASS.
Negative cases include unsupported Julia prereleases/build forms, final
package-version mismatch at each release tree, uncertain create response,
duplicate transaction or manifest ID with changed payload, intent without tag,
tag without released manifest, publication without exactly one released
predecessor, missing publication required fields, a wrong publication key,
duplicate publication pair/ref, conflicting tag target or GitHub Release
identity, mismatched evidence digest, and mismatched tag/intent/release
evidence.
Include leading-zero package and public-tag aliases, static snapshot field
omissions/digest mismatch and changed anchor/tag/source under a bound snapshot.
Test empty/nonempty ref/tag digest preimages, tampered nested digests,
tampered occupied sets, reversed ref/tag/occupied ordering, raw file digest
mismatch and rejection of a self-including snapshot digest.
Include unsupported certification bindings, missing/wrong manifest schema
version, undeclared or unsorted set arrays, duplicate set members, and
noncanonical optional-field encoding.

Fail closed when a pinned source/approval or reviewer authority cannot be
verified, the static selector or serialized writer cannot be established, an
exact principal sentinel is unavailable, an operation requires deferred
maintenance or rare-recovery automation, Julia prerelease semantics conflict,
outgoing reservation state is uncertain, durable candidate retention, certification subject,
exact-tree promotion, main freeze, or required protections cannot be proven,
or implementation would require Gate B. Do not silently weaken an invariant.
The exact static-selector and principal-sentinel blocked outcomes are stated
in R-006 and the concrete authority section. No generic fail-closed result
may replace their required status and reason codes.

The governing Issue is the live work-state record. `tasks.md` tracks
execution/evidence readiness only. A review verdict is evidence, never
readiness, merge authorization, public-release authority or issue closure.
