# CYAX-0125 Gate A prospective-amendment tasks

This checklist tracks execution and evidence readiness only. Issue #125 and
GitHub remain the live work-state record. The companion `spec.md` is a
prospective amendment, not Approved; fresh independent SPEC and STANDARDS
review is required before implementation can claim conformance.

The prior approved revision was
`cf0f9c39256b7a58e179af92a46fbf1cb651ff76`, approved under Issue #125 comment
refs `5746793852` and `5752268640`. Prior implementation/review/evidence for
the append-only `release-events` ledger is historical predecessor evidence,
not completion evidence for this reduced candidate. The checked items below
therefore do not retroactively mark that predecessor work as satisfying the
new requirements.

## Preconditions and review

- [x] Record exact handoff identity
  `19fa93e2b4b1439b1cc4217cf7ff9e42846980d46df66793156f72789894e40a`, target
  PR head `20b3935ace0e01fcee2808681c56595e3afa7667`, and owner approval
  receipt SHA `140d1fe1c7c09db3d40a04c66c1b258f18e8c42025a3453c2472ed682124637c`.
  This establishes dispatch context only.
- [x] Bind canonical Spec/Standards rubric SHA-256
  `418f2d5a276cbdb74b8ad331b532d59219ef33b4e9fabc2b3d4a21c55bc06c72`; no
  successor is adopted.
- [x] Preserve predecessor approval/review chronology and mark this spec
  prospective. No new approval metadata or PASS claim is present.
- [ ] Obtain fresh independent SPEC and STANDARDS review of this exact
  amendment. Changed bytes require fresh review; prior PASS evidence is not
  reusable as approval for the reduced candidate.

## A1 — Policy and release-neutral source

- [x] Reconcile `AGENTS.md` with create-once Git refs, immutable manifests,
  owner authorization, fail-closed controls, strict versions, permanent
  no-reuse, exact-tree certification, release-neutral docs and Python's
  bounded control-plane role.
- [x] Reconcile `.agents/skills/cyaxiverse-integration-release/SKILL.md` with
  the same authority and explicitly defer maintenance/rare-recovery
  automation to the later approved S2 gate.
- [ ] Verify any required SDD package-lifecycle wording remains aligned before
  spec approval; do not change the SDD skill unless a contradiction is found.
- [ ] Verify docs routing uses verified ref/manifest/environment context and
  requires no tracked release-time source edit.

## A2 — Static authority and immutable anchors

- [ ] Retain and verify exact `refs/heads/vmm:iterations.toml` selection,
  immutable `iterations/X.Y.Z` anchor equality, canonical SemVer grammar and
  Gate B retrospective declared-versus-actual truth.
- [ ] Define/review `static_iteration_snapshot` and complete
  `lifecycle_ref_snapshot` digests, sorting, replay and stale detection.
- [ ] Prove static/public-tag/lifecycle-ref occupied-set collisions and
  permanent no-reuse, including closed, withdrawn and consumed identities.

## A3 — Create-once lifecycle authority

- [ ] Define exact versioned protected `refs/heads/lifecycle/v1/*` namespaces for
  claims/reservations, candidates, intents,
  releases and publications; keep those object types distinct and forbid
  deletion, repointing and force updates.
- [ ] Define one small canonical manifest per lifecycle ref, including
  content-derived `manifest_id` with no global sequence allocator, schema
  version/type, transaction, owner authorization,
  predecessor refs, exact Git SHA/tree identities, timestamps and evidence
  digests. Represent progression only through immutable predecessor refs and
  validate required/forbidden fields and no terminal reversal.
- [ ] Implement complete-ref replay and create-if-absent/CAS semantics.
  Prove exact retry idempotence, conflicting identity INVALID, uncertain
  remote create BLOCKED/frozen, and no mutable ledger fallback.
- [ ] Verify the serialized exclusion boundary covers static snapshots,
  complete lifecycle-ref snapshots, anchor creation and canonical tag
  creation. Stop if protection is unavailable.

## A4 — First-principal lifecycle automation

- [ ] Implement deterministic principal sentinel reservation and permanent
  reservation consumption/no-reuse. Preserve the narrow proven pre-entry abort
  rule: the reservation identity is consumed, while an unclaimed final may be
  released only after durable non-entry proof; uncertainty remains unavailable.
- [ ] Implement principal closure, explicit UTC timestamp, immutable anchor,
  candidate durability, exact-tree certification, main freeze/ancestry
  disposition, immutable intent, protected canonical tag, released manifest
  and publication-evidence reconciliation.
- [ ] Preserve tree-bound versus commit-bound certification transfer and
  principal/maintenance SemVer, line and `main` checks.
- [ ] Record maintenance bootstrap/release and rare recovery as deferred
  later-S2 automation; do not mark their production paths complete in Gate A.

## A5 — Verification and convergence

- [ ] Run focused canonical-version, static-snapshot, lifecycle-ref,
  create-once, principal closure, certification, tag/manifest/tree and docs
  routing tests. Assert observable refs, bytes, digests, identities and
  blocked/INVALID reasons.
- [ ] Run `git diff --check`, snapshot/diff checks, package tests, audit,
  docs build, Python-free `using CYAxiverse`, bounded Python control-plane
  checks and applicable CI; record exact failures and unavailable checks.
- [ ] Verify `Project.toml=0.2.0`, no production DEV/adoption/closure/tag/
  publication/historical designation, no Gate B mutation, and no scientific,
  API, persisted-schema or version-boundary drift.
- [ ] Verify documentation lookup independently resolves the selected release
  or publication manifest's tag, commit/tree, package version and digest;
  missing or conflicting evidence blocks deployment.
- [ ] Reconcile spec ↔ plan ↔ tasks ↔ implementation ↔ evidence. Treat
  `specs/0125-version-iteration-release/evidence.md` and predecessor ledger
  review logs as historical unless fresh candidate evidence explicitly binds
  them to this amendment.

## A6 — Owner decision and live closure

- [ ] Present the exact reviewed candidate for explicit owner merge decision;
  review PASS is not merge authorization.
- [ ] After authorized merge, establish and verify actual create-once
  lifecycle-ref, immutable anchor, canonical-tag and principal freeze
  protections; record ruleset identities, patterns, UTC retrieval and
  observed enforcement.
- [ ] Obtain fresh independent post-settings closure review of merged source,
  refs, manifests, protections and verification. Mark Gate A complete only
  after reconciliation; keep Gate B and deferred maintenance/recovery S2 work
  separate.

## Current execution state

The policy/spec/plan/task candidate has been drafted in this worktree. The
reduced lifecycle implementation, fresh reviews, owner merge decision, live
protection evidence and closure review remain incomplete. No checkbox above
may be changed to imply those external states without observed evidence.
