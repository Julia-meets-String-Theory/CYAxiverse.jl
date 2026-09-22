# CYAX-0125 Gate A implementation tasks

This checklist tracks execution and evidence readiness only. Issue #125 and
GitHub remain the live work-state record. The companion `spec.md` successor is
not effective until exact SPEC and STANDARDS review passes and the external
approval record binds that frozen five-file revision. The final implementation
candidate also requires convergence review before any owner merge decision.

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
- [x] Preserve predecessor approval/review chronology and mark that evidence
  historical. Approval/review records remain outside the frozen five-file
  authority set and cannot exempt changed normative bytes from rereview.
- [ ] Obtain fresh independent SPEC and STANDARDS review of the exact
  normative successor. The completed reviews of `dd7825d3...` returned
  `REQUEST_CHANGES`; those verdicts are historical and authorize no readiness
  claim.

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
- [x] Verify docs routing uses verified ref/manifest/environment context and
  requires no tracked release-time source edit in the current implementation;
  the final exact-candidate check remains part of A5.

## A2 — Static authority and immutable anchors

- [ ] Retain and verify exact `refs/heads/vmm:iterations.toml` selection,
  immutable annotated `iterations/X.Y.Z` tag-object/payload equality,
  canonical SemVer grammar and
  Gate B retrospective declared-versus-actual truth.
- [x] Define/review `static_iteration_snapshot` and complete
  `lifecycle_ref_snapshot` digests, sorting, replay and stale detection.
- [ ] Prove static/public-tag/lifecycle-ref occupied-set collisions and
  permanent no-reuse, including closed, withdrawn and consumed identities.

## A3 — Create-once lifecycle authority

- [x] Define exact versioned protected `refs/heads/lifecycle/v1/*` namespaces for
  claims/reservations, candidates, intents,
  releases and publications; keep those object types distinct and forbid
  deletion, repointing and force updates.
- [x] Define one small canonical manifest per lifecycle ref, including
  content-derived `manifest_id` with no global sequence allocator, schema
  version/type, transaction, owner authorization,
  predecessor refs, exact Git SHA/tree identities, timestamps and evidence
  digests. For `publication`, require exactly one released-manifest
  predecessor, the released manifest ID/digest, canonical tag commit/tree,
  GitHub Release identity and publication-evidence digest; derive its `pub-`
  ID/ref deterministically from the released-manifest ID/tag pair. Represent
  progression only through immutable predecessor refs and validate
  required/forbidden fields and no terminal reversal.
- [x] Implement complete-ref replay and create-if-absent/CAS semantics in the
  bounded control-plane implementation; exact-candidate test evidence remains
  pending.
  Prove exact retry idempotence, conflicting identity INVALID, uncertain
  remote create BLOCKED/frozen, and no mutable ledger fallback.
- [x] Verify the serialized exclusion boundary covers static snapshots,
  complete lifecycle-ref snapshots, anchor creation and canonical tag
  creation. Stop if protection is unavailable.
- [ ] Verify immutable owner authorization before each affected mutation. Bind
  the exact grant ID/ref/digest in manifests and cover missing, malformed,
  changed, expired, wrong-owner and cross-operation grants before any write.
  Derive the ID and equal digest from the one canonical preimage that omits
  both identity fields. Reproduce the specification's fixed bytes, byte count,
  digest and ID, and reject independent identity, preimage-field and fetched-
  byte tampering before mutation.

## A4 — First-principal lifecycle automation

- [ ] Implement deterministic principal sentinel reservation and permanent
  reservation consumption/no-reuse. Preserve the narrow proven pre-entry abort
  rule: the reservation identity is consumed, while an unclaimed final may be
  released only after durable non-entry proof; uncertainty remains unavailable.
- [ ] Implement principal closure, explicit UTC timestamp, immutable anchor,
  candidate durability, exact-tree certification, main freeze/ancestry
  disposition, immutable intent, protected canonical tag, released manifest
  and exactly one deterministic publication manifest/ref with GitHub Release
  identity/evidence reconciliation.
- [ ] Preserve tree-bound versus commit-bound certification transfer and
  principal/maintenance SemVer, line and `main` checks.
- [x] Record maintenance bootstrap/release and rare recovery as deferred
  later-S2 automation; do not mark their production paths complete in Gate A.
- [ ] Implement and test the validation-only
  `maintenance-bootstrap-validation-v1` schema. Cover exact activated
  correspondence, proven non-entry, mismatched identities, and uncertain or
  failed activation with the version unavailable and line frozen. Do not add a
  production maintenance writer.

## A5 — Verification and convergence

- [ ] Run focused canonical-version, static-snapshot, lifecycle-ref,
  create-once, principal closure, certification, tag/manifest/tree and docs
  routing tests. Include publication positive binding and negative pre-release,
  duplicate, wrong-predecessor, tag-target, GitHub-identity and evidence-digest
  cases, plus the specification's fixed expected publication hash/ref and
  owner-authorization fixtures.
  Include lightweight/same-target anchor substitution and the complete
  owner-authorization negative matrix.
  Assert observable refs, bytes, digests, identities and blocked/INVALID
  reasons.
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

The prior N1 candidate received blocking independent-review findings. This
successor closes those findings but is not effective until its exact review and
external approval record converge. Existing implementation work remains a
provisional worktree candidate and cannot claim specification conformance.
Owner merge decision, live protection evidence, production lifecycle state,
Gate B, and closure review remain incomplete. No checkbox above implies any of
those external states.
