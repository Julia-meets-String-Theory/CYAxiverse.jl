---
name: cyaxiverse-integration-release
description: Consolidate, rebase, integrate, verify, or prepare approved CYAxiverse work for release from `vmm` to `main`. Use for branch/worktree cleanup, conflict resolution, integration PRs, release evidence, or sequencing already-approved work.
---

# CYAxiverse Integration and Release

1. Read `AGENTS.md` and the relevant PR/issue/scientific status material. Inspect
   branch/worktree state before changing anything; preserve user-owned work.
2. Classify branches/worktrees before pruning. Delete only when commits are
   merged, superseded, or explicitly abandoned. Do not use cleanup to discard
   unresolved scientific evidence or unmerged user work.
3. Rebase/merge only a reviewable deliverable onto current `vmm`. Resolve
   conflicts deliberately; do not take either side wholesale when that would
   change APIs, dependencies, persisted schemas, or scientific behavior.
4. Stop and escalate if conflict resolution would choose a normalization,
   physical cut, candidate classification, mass convention, benchmark
   interpretation, population definition, or scientific schema. Integration
   preserves approved science; it does not decide science.
5. Keep `vmm` as the integration branch and `vmm -> main` as the deliberate
   principal promotion boundary. A reviewed closure adopts the final package
   version on its owner line; the aggregate release PR verifies that the
   certified principal tree and version reach `main`. Feature branches state
   version impact, and Gate A does not adopt a package version.
6. Preserve the package iteration contract while integrating: an active
   `X.Y.Z-DEV` reservation is global to its owner line. Closing its reserved
   final consumes it; closing a different final records the unused identity as
   `CONSUMED_UNUSED_DEV_RESERVATION`. Closed, candidate, withdrawn, released,
   and consumed identities are never reused. A reservation with proven
   pre-entry abort may be made available again under the recovery rules;
   uncertain outcomes remain unavailable. Lifecycle state is represented by
   protected create-once Git refs and small immutable canonical manifests;
   refs are never deleted, repointed, or force-updated, and no mutable
   `release-events` ledger is canonical. Claim, reservation, candidate, intent,
   release and publication object types remain distinct and progress only via
   immutable predecessor-ref bindings. Certify release candidates against an
   immutable exact tree and retain candidate, tag-intent, release, and
   publication evidence identities in those refs/manifests.
7. Keep tracked source release-neutral. Documentation deployment selects the
   development, principal-versioned, maintenance-versioned, or stable channel
   from a verified ref and canonical release manifest. Do not edit documentation
   source as part of a release.
8. Gate A requires and tests automation for the first principal lifecycle path
   (principal allocation/reservation, closure/anchor, candidate,
   certification, canonical tag and release evidence), but does not adopt a
   package version, designate historical releases, create a production `-DEV`,
   close an iteration, create a public tag, publish a release, or reconcile
   `vmm -> main`. Maintenance-line bootstrap/release and rare recovery
   automation are deferred to a later approved S2 gate. All unsupported or
   uncertain operations fail closed under owner authorization.
9. Run the applicable release gates on the integrated commit: `git diff --check`,
   package tests, `bin/audit.jl`, docs build, Python-free import, bounded
   repository/CI control-plane checks, and CI as required. Python lifecycle
   tooling is not a runtime dependency of `using CYAxiverse`. Record exact
   results and unavailable optional integrations.
10. After merge, prune superseded branches/worktrees as a separate, verified
   cleanup step rather than leaving historical agent worktrees indefinitely.

End with the branch/commit integrated, verification evidence, remaining
blockers, version impact, and the next independently reviewable deliverable. Do
not claim release readiness while a required gate is unobserved.
