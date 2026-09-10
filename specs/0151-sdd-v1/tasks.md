# Tasks — CYAX-0151

## Task rules

- Every task advances an approved CYAX-0151 requirement or gate.
- Verification belongs inside the task.
- This task list does not automatically create GitHub Issues.
- Separate branches/PRs are used where the rollout gate is an independently reviewable deliverable.
- No task may reinterpret the approved SDD contract or scientific work being migrated.

## G0 — Contract approval

- [x] **T001 [G0] Approve CYAX-0151**
  - Outcome: project-owner approval obtained.
  - Evidence: durable approval recorded on Issue #151 on 2026-09-10.
  - Result: **PASS**.

## G1 — Repository SDD layer

- [x] **T101 [R-001–R-016, G1] Freeze implementation baseline**
  - `vmm` baseline: `3a5b034fbbabce6638607373e682cbd6f66f6aaa`.
  - Base tree: `f3ab269aec9ff56f5347273a863bab424a38e49c`.
  - Scope inspected before editing; #148 remains on a separate branch.

- [x] **T102 [R-001–R-016, G1] Create focused SDD-v1 branch**
  - Branch: `chore/sdd-v1-151`.
  - Based on the recorded `vmm` baseline.

- [x] **T103 [R-001–R-016, G1] Materialize approved CYAX-0151 artifacts**
  - Add `spec.md`, `plan.md`, and `tasks.md` under `specs/0151-sdd-v1/`.
  - Set spec status to `approved`.

- [x] **T104 [R-004–R-009, R-011–R-013, G1] Implement `cyaxiverse-sdd` skill**
  - Add canonical skill under `.agents/skills/` with reusable templates.
  - Avoid duplication of specialist scientific/Julia/release policy.

- [x] **T105 [R-012, R-013, G1] Add Claude/Codex skill mirrors**
  - Follow the consolidated tracked-symlink architecture.

- [x] **T106 [R-001, R-002, R-005, R-008, R-011, R-012, G1] Amend `AGENTS.md` minimally**
  - Add only normative SDD rules; detailed procedure remains in the skill.

- [x] **T107 [R-004, R-006–R-009, G1] Add SDD Issue intake form**
  - Add `.github/ISSUE_TEMPLATE/sdd_work.yml`.
  - Keep bug-report intake.
  - Retire the generic feature-request form because the SDD form supersedes it.

- [x] **T108 [R-002, R-007, R-010, R-011, G1] Add SDD-aware PR template**
  - Include spec, requirement/gate scope, evidence matrix, non-scope and convergence.
  - Support partial-gate PRs.

- [x] **T109 [R-003, R-012, G1] Update human workflow guide**
  - Add concise SDD layer and links without creating another handbook/ledger.
  - Convergence review restored unrelated pre-existing local-migration guidance rather than broadening that section.

- [x] **T110 [G1] Run focused process verification**
  - Full PR patch inspected for whitespace/conflict-marker problems: none observed.
  - `sdd_work.yml` parsed successfully as YAML with the expected intake fields.
  - Skill mirrors verified as mode-`120000` symlinks to the canonical `.agents/skills/cyaxiverse-sdd` source.
  - Changed-file scope contains only control-plane/process files; no `src/`, dependency, package-version, scientific-schema, or scientific-data changes.
  - No pull-request CI workflow was triggered for the process-only path set.
  - Exact local `git diff --check` was unavailable because the GitHub connector has no local worktree and the execution environment cannot reach GitHub to clone the branch.
  - `python3 scripts/agent_verify.py diff-check` was likewise unavailable; source inspection confirms this subcommand is a wrapper around `git diff --check`, so no additional verification semantics were unobserved.

- [x] **T111 [R-001–R-013, G1] Perform convergence review**
  - Compared approved CYAX-0151 against the tracked spec, skill, templates, `AGENTS.md`, Issue form, PR template, and human guide.
  - R-001–R-013 are represented without contradiction or a second policy source.
  - G2–G5 remain explicitly outside this PR.
  - Result: **PASS**, subject only to the transparently unavailable local command invocation recorded in T110.

- [x] **T112 [G1] Open focused process-only PR to `vmm`**
  - Draft PR: #152.
  - References #151 and states G1 scope plus G2–G5 non-scope.

- [ ] **T113 [G1] Final review and merge G1**
  - Review final diff and mergeability.
  - Mark ready only after G1 acceptance.
  - Merge only the process-layer deliverable and record resulting `vmm` SHA.

## G2 — GitHub Project

- [ ] **T201 [R-003, G2] Create organization Project** — `CYAxiverse Research & Development`.
- [ ] **T202 [G2] Configure workflow states** — Backlog, Specifying, Spec Review, Ready, Implementing, Verification, Done; no permanent Blocked column.
- [ ] **T203 [G2] Configure core fields** — Workstream, Spec class, Priority, Review gate, Work cycle, Target release; prefer native GitHub relationships where available.
- [ ] **T204 [G2] Configure initial WIP policy** — Specifying = 2; Implementing = 2; Verification = 2; spec-level deliverables are the WIP unit.
- [ ] **T205 [G2] Create saved views** — Current Flow, Research Map, Now / Next, Verification Queue, Development History, PR / Integration View.
- [ ] **T206 [G2] Configure low-risk automation** — auto-add relevant Issues, Backlog initialization, closed Issue → Done, merged PR completion where useful; no automated scientific approval.
- [ ] **T207 [R-003, G2] Project convergence check** — verify Project is navigational state over Issues/PRs, not duplicate prose/authority.

## G3 — #148/#149 pilot

- [ ] **T301 [R-014–R-016, G3] Freeze current #148/#149 state** — capture migration-time Issue, PRs, accepted gates and durable evidence.
- [ ] **T302 [R-014, G3] Create canonical #148 `spec.md`** — preserve existing scientific contract; introduce requirement IDs without semantic expansion.
- [ ] **T303 [R-014, R-015, G3] Create #148 `plan.md`** — reconstruct only durable architecture/decisions and separate historical from future/current work.
- [ ] **T304 [R-015, G3] Create #148 `tasks.md`** — mark only evidenced work complete and represent later gates according to actual migration-time state.
- [ ] **T305 [R-010, R-014–R-016, G3] Map PR evidence** — map #149 and any later #148 PRs separately without changing historical claims.
- [ ] **T306 [R-003, R-014, G3] Add canonical pointers** — link #148 to its spec and add concise migration notes where useful; preserve historical bodies.
- [ ] **T307 [R-014–R-016, G3] Independent scientific migration review** — compare original Issue/PR/evidence to migrated artifacts; ambiguity blocks for owner decision.
- [ ] **T308 [G3] Merge pilot migration** — process/documentation-only PR and record G3 result.

## G4 — Six-week backfill

- [ ] **T401 [G4] Build recent development history** — cover approximately 2026-07-30 through 2026-09-10 and classify relevant merged PRs by workstream/work cycle.
- [ ] **T402 [G4] Reconcile current active work** — identify duplicates, superseded/stale work, parent/sub-issues and genuine blockers without closing scientific ambiguity for neatness.
- [ ] **T403 [R-006, G4] Normalize durable work granularity** — keep implementation minutiae below Issue level and use sub-issues only when independently durable/blocked/mergeable/decision-bearing.
- [ ] **T404 [G4] Identify selective spec-backfill candidates** — prioritize active S2/S3 and foundational continuing contracts.
- [ ] **T405 [G4] Create only justified migration specs** — preserve owner-approved meaning; unresolved work stays draft.
- [ ] **T406 [R-003, G4] Development-history convergence check** — verify the Project answers what changed, when, workstream, active/blocked state and canonical intent location.

## G5 — Evaluate SDD v1

- [ ] **T501 [G5] Collect real usage evidence for approximately 6–8 weeks**.
- [ ] **T502 [G5] Review traceability and overhead** — assess current-state clarity, intent recoverability, agent context burden, PR traceability, duplicate/stale work, review bottlenecks, spec maintenance cost and S0 burden.
- [ ] **T503 [G5] Decide v1 disposition** — ACCEPT, REVISE or SIMPLIFY; do not automatically expand tooling.
- [ ] **T504 [G5] Evaluate optional SDD tooling** — only after usage evidence exists, decide whether selected clarification/analysis/convergence automation adds enough value.
