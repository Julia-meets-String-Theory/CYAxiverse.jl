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
   release boundary. Feature branches state version impact; the aggregate
   release PR applies the reviewed package-version bump.
6. Run the applicable release gates on the integrated commit: `git diff --check`,
   package tests, `bin/audit.jl`, docs build, Python-free import, and CI as
   required. Record exact results and unavailable optional integrations.
7. After merge, prune superseded branches/worktrees as a separate, verified
   cleanup step rather than leaving historical agent worktrees indefinitely.

End with the branch/commit integrated, verification evidence, remaining
blockers, version impact, and the next independently reviewable deliverable. Do
not claim release readiness while a required gate is unobserved.
