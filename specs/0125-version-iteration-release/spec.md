---
spec_id: CYAX-0125-Gate-A
title: Package iteration and release lifecycle
issue: 125
class: S2
status: draft
target_iteration: version-lifecycle-retrofit-2026-09
version_bearing: true
package_infrastructure_impact: patch
package_version_adoption: none
review_required: independent SPEC and STANDARDS
approval_refs:
  - "Issue #125 comment 5746793852"
  - "Issue #125 comment 5752268640"
reviewed_spec_revision: pending
---

# CYAX-0125 Gate A — package iteration and release lifecycle

## Objective and authority

Gate A establishes the forward machinery for package iterations, global version
allocation, exact-tree certification and public release. It does not use that
machinery for a real closure, development-version transition, public release or
historical reconstruction. The actual `Project.toml` version remains `0.2.0`.

The owner-approved [base lifecycle decision](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/125#issuecomment-5746793852)
and [reservation decision](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/125#issuecomment-5752268640)
govern this retrofit. The later base decision prospectively supersedes earlier
Issue #125 statements that confined package bumps and documentation source
changes to a `vmm → main` release PR or predetermined `0.3.0` as the next
final. Those older comments remain historical evidence. `AGENTS.md` and
applicable normative skills remain repository authority until Gate A's
replacement rules merge.

The target iteration is `version-lifecycle-retrofit-2026-09`. This is
version-bearing package infrastructure with **patch** impact: it changes
installation, documentation and release guarantees without intentionally
changing a public Julia API, scientific result, persisted scientific schema or
supported scientific convention. The impact is a declaration for later
aggregation; Gate A makes no package-version adoption bump.

## Scope and limits

Gate A may update the repository policy, SDD/release workflow, release-neutral
README and documentation, version checker, static iteration schema, allocation
and lifecycle validators, transaction helpers, focused fixtures and tests, and
minimal workflow checks. Each changed path must map to a requirement below.
Normative policy promotion takes effect only after the reviewed merge.

Gate A excludes historical version-ladder selection, real retrospective
entries, Gate B reconstruction or adoption, any `Project.toml` change from
`0.2.0`, a production `-DEV` transition, current `vmm → main` reconciliation,
an actual public tag or release, scientific/numerical reinterpretation, changes
to persisted scientific schemas, rewriting Issue #172 evidence, and any change
to the legacy `v-0.1` tag. Existing work retains historical declarations;
Issue #172 remains grandfathered.

The approved principal model is: `vmm` carries principal package-development
iterations; `main` carries the latest certified principal release; certified
maintenance releases live on `maintenance/X.Y` lineage with immutable public
SemVer tags and do not move a newer `main` backward. A target iteration has a
stable ID before its final version is assigned. Closing an iteration and
publicly releasing it are independent operations.

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

### R-006 — Deterministic principal development identity

After principal `X.Y.Z` closes, the exact next sentinel is
`X.Y.(Z+1)-DEV`. It reserves final `X.Y.(Z+1)` but does not predetermine
the next closure's reviewed aggregate impact. If that exact sentinel is
unavailable, return `PRINCIPAL_SENTINEL_UNAVAILABLE`; do not skip it.

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
or consumed version may be allocated again.

### R-012 — Static iteration authority

`iterations.toml` and validated immutable iteration identities describe
static version occupation and designation.

### R-013 — Mutable authority

One protected `release-events` branch and append-only `release-events.jsonl`
stream is canonical for mutable reservation, allocation and release lifecycle.
It is never merged into `vmm` or `main`; no second mutable ledger is valid.

### R-014 — Combined allocation view

`GLOBAL_ALLOCATION_VIEW` combines one exact
`static_iteration_snapshot` with one exact `allocation_event_head`.
Only versions unoccupied in both views and permitted by line rules are AVAILABLE.

### R-015 — Static namespace occupancy

Validated retrospective designations, closed prospective designations,
protected `iterations/*` anchors and canonical public SemVer tags occupy
their versions, including closed but unreleased iterations.

### R-016 — Mutable namespace occupancy

Prepared/open DEV reservations, terminal reservation consumption, maintenance
line state, candidates, withdrawals and releases determine mutable occupation.

### R-017 — Allocation serialization

An allocation-changing transaction binds a transaction ID, exact static
snapshot and expected event head. It revalidates both, derives the next event
ID, appends a descendant commit and performs a non-force linear update.
Stale decisions must refresh and recompute.

### R-018 — Static mutation serialization

Prospective closure/materialization, future retrospective designation,
iteration-anchor creation and canonical public-tag creation participate in
the same governed allocation exclusion boundary or prove equivalent exclusion.

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
release-time edit; deployment derives from ref/event/environment while
preserving generated API, development and stable/versioned docs.

### R-023 — Exact-tree certification

The separately pinned certification harness tests an immutable package
checkout and proves its tracked tree is unchanged and matches the closed
iteration tree. Candidate and release tree equality is verified.

### R-024 — Separate certification identity

Certification evidence records independently pinned package, policy, harness,
environment and evidence identities, and states whether the certified subject
is tree-bound, commit-bound or another reviewed exact model.

### R-025 — Single candidate lifecycle

Principal and maintenance use one sequence: durable exact candidate, equality
check, `candidate_opened`, certification, line-specific pre-tag checks,
immutable public tag, `released` event, GitHub Release. No duplicate
candidate opening or certification path is allowed.

### R-026 — Principal exact-tree promotion

A principal release records candidate-time and freeze-time `main` identities,
mechanically freezes `main`, dispositions intervening ancestry, reconciles to
the certified closed tree, checks final version greater than prior `main`
and no `-DEV`, then applies certification-transfer rules before tagging.

### R-027 — Maintenance exact-tree release

Maintenance candidate, certification and tag remain in the declared lineage.
The `released` event records contemporaneous unchanged principal `main`
SHA/version. Maintenance metadata is not forward-ported to `vmm`.

### R-028 — Protected closure transaction

Freeze the owner line; bind static/event heads; verify and merge reviewed final
closure; create/verify the annotated iteration anchor and explicit closure
time; refresh static snapshot; consume and verify outgoing DEV reservation;
then select, prepare, reopen and activate the deterministic next DEV identity.
No unrelated merge interleaves. Unfreeze only after full correspondence.

### R-029 — Outgoing reservation terminality

If closure/anchor succeeds but outgoing consumption fails, keep the line
frozen and reconcile forward. Do not allocate, prepare or reopen next DEV.
If reconciliation cannot complete, return
`OUTGOING_RESERVATION_RECONCILIATION_FAILED`.

### R-030 — Initial maintenance bootstrap

Resolve an approved immutable `X.Y` base; prove line absence; bind allocation
view; select lowest AVAILABLE same-line patch; prepare reservation; create the
maintenance ref with create-if-absent/CAS and immediate bootstrap protection;
install `-DEV`; activate reservation; append `maintenance_line_opened`.

### R-031 — Maintenance bootstrap correspondence

No maintenance work begins until base, branch, DEV head, owner line, reservation
and line-open event correspond exactly. Uncertain creation or failed activation
keeps the version unavailable and the line frozen under the stated recovery
rules.

### R-032 — Candidate durability

Before `candidate_opened`, the exact candidate ref/SHA/tree is remotely
durable. Active and withdrawn candidates remain auditable; a released
candidate remains resolvable through the release evidence chain. Do not
delete its sole durable terminal reference.

### R-033 — Withdrawal and no reuse

Pre-tag failure may withdraw a candidate. Withdrawn version and candidate
identity remain permanently consumed and retained for audit.

### R-034 — Irreversible public-tag boundary

After protected canonical public tag creation, do not withdraw, delete,
repoint or reuse. Reconcile missing event/publication forward.

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

Gate A implements and tests machinery without actual adoption, historical
population, real `-DEV`, closure, release or public tag.

### R-039 — Exact final-state review

After authorized merge and live protection convergence, fresh independent
review examines merged source, actual refs/rulesets/freeze controls, event
authority and final verification. Gate A is not complete before reconciliation
of that review.

### R-040 — Canonical event identity

Every event has `event_id`, `event_type` and explicit schema-valid RFC3339 UTC
`timestamp_utc`, plus transaction ID where applicable. IDs are
`EVT-` followed by exactly 12 positive decimal digits, first
`EVT-000000000001`, next prior + 1. Successful appends permanently consume
their IDs; failed unpublished attempts do not. Reject missing, malformed,
duplicate, decreasing or noncontiguous IDs. After
`EVT-999999999999`, fail closed with `EVENT_ID_EXHAUSTED`; never wrap or
silently widen the format.

### R-041 — Canonical closure UTC timestamp

Closure uses one explicit schema-valid RFC3339 UTC
`closure_timestamp_utc` bound to the annotated iteration ref and durable
closure evidence. It is never inferred from UI, commit, filesystem or local
prose time. Repeated canonical copies must agree exactly.

### R-042 — Durable release-evidence identity

A principal `released` event contains event ID/type/UTC time, final version,
principal line, anchor ref/SHA/tree, durable candidate ref/SHA/tree, final
release SHA/tree, frozen pre-integration `previous_main_sha/version`,
verified post-reconciliation `main_at_event_sha/version`, certification
binding/subject SHA/tree/policy revision/harness revision/environment/evidence
refs, closure UTC time, public tag and evidence refs. Maintenance contains the
same applicable identities and contemporaneous unchanged principal
`main_at_event_sha/version`, but no principal previous-main claim. Larger
immutable evidence may be content-addressed and referenced by the event.
Terminal output alone is insufficient.

The canonical `released` event precedes GitHub Release publication and is
never mutated to add its later ID. A separate immutable/content-addressed
publication-evidence artifact keyed by event ID and public tag records GitHub
Release identity, publication time and evidence after publication.

### R-043 — Bidirectional release consistency

Terminal validation requires every canonical public release tag to resolve to
exactly one matching released event/version/line/commit/tree/certified
iteration tree; every GitHub Release to resolve through its canonical tag to
that event; and every released event to resolve back to its immutable tag,
release commit/tree and certified anchor/tree. Mismatch is INVALID.
`publication_reconciliation_pending` is an explicit nonterminal forward
recovery state, never silently accepted as `terminal_consistent`.

### R-044 — Certification binding and transfer

Tree-bound certification may transfer from candidate to different final
commit only with durable evidence that certified subject, candidate, final
release and anchor trees are equal. Commit-bound certification requires a
new certification of a different final release commit before tag creation.
No implicit transfer is permitted.

## Concrete static and mutable authorities

`canonical_static_iteration_source` is the freshly resolved
`refs/heads/vmm:iterations.toml` in this repository. Every allocator resolves
that ref to an exact commit/tree and hashes the file bytes; no caller may
substitute an arbitrary historical copy. The root registry carries principal
prospective entries and, after Gate B only, reviewed retrospective entries.
For maintenance, the protected `iterations/X.Y.Z` anchor tree carries its
own static entry. The snapshot validates all protected iteration refs and
their exact anchor-tree entries; maintenance version metadata is not copied
into `vmm`. The snapshot also includes sorted canonical public SemVer tags,
their commit bindings and digests, the derived occupied set and a digest over
the canonical serialization. The legacy `v-0.1` tag is preserved and excluded
from canonical future tag parsing. Existing principal `main` and
pre-adoption `vmm` package versions remain unavailable as historical/current
identities until Gate B provides fuller designations.

The `release-events` branch is a minimal orphan non-package-source branch
containing one canonical `release-events.jsonl` stream, bootstrapped empty in
Gate A. Every mutation is made by a controlled writer under protected linear
history with old bytes as an exact prefix and Git non-force expected-head
update. It revalidates the static snapshot at the update boundary; static
mutation, event mutation, iteration-anchor creation and public-tag creation
share a controlled exclusion mechanism. If exclusion/protection cannot be
established, the operation blocks rather than relying on a non-atomic check.
No production lifecycle event is appended in Gate A; synthetic event fixtures
and isolated temporary Git refs exercise the machinery.

The event vocabulary includes `development_reservation_prepared`,
`development_reservation_opened`, `development_reservation_aborted`,
`development_reservation_consumed`, `maintenance_line_opened`,
`candidate_opened`, `candidate_withdrawn` and `released`.
Type-specific schemas require and forbid fields according to their transition.
The validator checks byte-prefix preservation, strict JSONL, contiguous IDs,
UTC time and lifecycle transitions. One owner-line DEV identity corresponds
to one active reservation and actual line head. A prepared reservation is
globally unavailable. Abort is allowed only with proof the matching DEV state
was never entered; uncertain outcomes remain unavailable and frozen.

## Protected transactions and recovery

Closure and reopen follow R-028 in order. The outgoing reservation must become
terminal before any next allocation; if an owner closes a different final,
its old reserved final remains permanently consumed. The required principal
sentinel is exact `X.Y.(Z+1)`; maintenance searches only upward within its
`X.Y` patch line. A prepared next reservation binds static snapshot, event
head, transaction, reserved final, intended DEV, owner line and expected line
head. Reopen changes only the governed line's `Project.toml` to that DEV
identity; activation binds the actual reopened head. On definite non-entry,
abort with evidence. On uncertainty or activation failure, keep the version
unavailable and reconcile forward.

Initial maintenance bootstrap follows R-030 before permitting work. A
create-if-absent ref transition from the approved immutable base and an
effective bootstrap freeze prevent unrelated pushes before DEV/reservation/
line-open convergence. Definite branch non-creation permits proven abort;
uncertain creation blocks. A created ref with failed DEV install remains
restricted; failed activation or line-open event reconciles forward.

A candidate is created and made remotely durable before `candidate_opened`.
Certification uses a separately pinned reviewed policy and harness and
verifies the package tree unchanged. Principal integration freezes `main`,
dispositions ancestry drift, yields the certified anchor tree, verifies
monotonic final version, and applies R-044. Maintenance verifies lineage and
preserves contemporaneous `main`. Public tag creation is irreversible;
subsequent failures reconcile event and publication forward. The released
event's `previous_main_*` means frozen pre-integration principal `main`;
`main_at_event_*` means verified post-reconciliation principal release
`main`, or contemporaneous unchanged principal `main` for maintenance.

## Gate order, evidence and stop rules

1. Verify fresh remote refs, Issue approvals, applicable skills/lessons and the
   pinned canonical Spec/Standards rubric.
2. Freeze this exact draft and obtain independent SPEC and STANDARDS review.
   Re-review any changed spec bytes. Record approval of the exact revision
   with both Issue comment refs before deriving `plan.md` or `tasks.md`.
3. Derive plan/tasks and check requirement-to-implementation/test coverage.
4. Implement A1–A6 on one branch, then converge spec, plan, tasks, source,
   tests and PR evidence. Keep actual `Project.toml=0.2.0`.
5. Obtain a fresh independent read-only exact-candidate pre-merge review.
   Its verdict does not authorize merge. Merge requires an owner decision.
6. After authorized merge, establish and verify actual branch/ref/tag
   protections, freeze controls and event authority without weakening
   existing `main` protection. Record ruleset/protection IDs, target patterns,
   UTC retrieval times, canonical snapshots, cryptographic digests and
   observed enforcement.
7. Obtain fresh independent closure review of merged source, live protections,
   refs, event authority and final evidence. Gate A completes only after that
   review is reconciled.

Tests must cover final/prerelease parsing, principal and maintenance sentinels,
no reuse, static/event races, concurrent expected-head writers, event
identity/timestamps/exhaustion, reservation/closure/bootstrap recovery,
durable candidate lifecycle, certification transfer, complete principal and
maintenance evidence, bidirectional tag/event/tree consistency and post-tag
forward recovery. Run focused tests before package, audit, docs, Python-free
import, workflow and remote CI checks. Record exact commands, results and
unavailable checks; unobserved checks are not PASS.

Fail closed when a pinned source/approval or reviewer authority cannot be
verified, the static selector or serialized writer cannot be established, an
exact principal sentinel is unavailable, maintenance cannot find a permitted
patch, Julia prerelease semantics conflict, outgoing reservation or bootstrap
state is uncertain, durable candidate retention, certification subject,
exact-tree promotion, main freeze, or required protections cannot be proven,
or implementation would require Gate B. Do not silently weaken an invariant.

The governing Issue is the live work-state record. `tasks.md` tracks
execution/evidence readiness only. A review verdict is evidence, never
readiness, merge authorization, public-release authority or issue closure.
