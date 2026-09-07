# CYAxiverse.jl: AI-agent and Git workflow after control-plane consolidation

This document is a human-facing guide. It is **not mandatory context for every
agent run**. The canonical agent contract is `AGENTS.md`; project-specific
skills are loaded only when relevant.

## 1. Why this cleanup exists

The repository accumulated several overlapping instruction layers during an
intense period of agent-assisted development: root and Copilot agent files,
Copilot prose, an AI policy, Claude settings, personal response preferences,
generic skills, and large continuation handoffs. Many repeated the same rules
with slightly different wording. That increased prompt/context cost, created
contradictions, and made it hard to know which file was authoritative.

The consolidation principle is now:

1. **One canonical project contract:** `AGENTS.md`.
2. **One source of truth for reusable project skills:** `.agents/skills/`.
3. **Thin tool adapters:** `CLAUDE.md` and `.github/copilot-instructions.md`.
4. **Executable verification over repeated prose:** `scripts/agent_verify.py`,
   package tests, audits, and CI.
5. **Git/PR state over chat memory:** issue/branch/commits/PR for ordinary work;
   long handoffs only when a long scientific investigation actually needs one.
6. **Personal tooling stays personal:** plugins, permissions, ADHD formatting,
   generic refactoring/documentation skills, and experimental agent tools are
   local configuration.

This is the control-plane part of the wider repository consolidation plan: make
active state explicit, reduce parallel sources of truth, prune stale
branches/worktrees carefully, and keep `vmm -> main` as the reviewed release
boundary.

## 2. What changed

### Kept and consolidated

- `AGENTS.md` is the only normative repository-wide agent contract.
- Five CYAxiverse-specific skills are tracked:
  - `cyaxiverse-julia-quality`
  - `cyaxiverse-scientific-reproduction`
  - `cyaxiverse-ks-geometry-sampling`
  - `cyaxiverse-vacua-pipeline`
  - `cyaxiverse-integration-release`
- Claude and Codex receive those skills through tracked symlinks pointing to
  `.agents/skills/`, so there is only one editable copy.
- `scripts/agent_verify.py` remains the preferred compact verification entry
  point.

### Removed from repository policy/state

- `AI_POLICY.md`: the small number of durable contribution rules are now in
  `AGENTS.md`.
- `.copilot/AGENTS.md`: Julia/scientific invariants were merged into the
  canonical contract and domain skills.
- `.copilot/session-progress.md`: stale session state should not be repository
  truth.
- `.claude/settings.json`: plugin choices are user-specific.
- the tracked `.codex/skills/i-have-adhd` skill: presentation preferences are
  personal and should be installed locally if desired.
- generic project skills such as documentation writing, general refactoring,
  complexity reduction, and autoresearch are no longer repository policy. They
  may still be installed globally/locally when useful.

### Important correction

The old instructions stated that axion mass-squared eigenvalues must always be
positive. That is too broad for code intentionally studying saddles and
negative-curvature/tachyonic directions. The consolidated contract applies
positivity only where the mathematical/physical object requires it and
preserves diagnostic critical-point semantics.

## 3. Day-to-day Git workflow refresher

There is no single rule that every feature needs both an issue and an early PR.
Use the lightest process that still gives you a durable, reviewable unit.

### Small bug or maintenance fix

Typical flow:

```text
vmm
  -> branch
  -> agent/human implementation
  -> focused tests
  -> PR to vmm
  -> review/CI
  -> merge
```

An issue is optional when the problem is obvious and the PR itself explains it.
Open the PR once there is a coherent first commit; make it a draft if work is
still in progress.

### Planned feature or scientific investigation

Prefer:

```text
issue / scoped problem statement
  -> branch from vmm
  -> (optional) draft PR early
  -> implementation + evidence
  -> PR ready for review
  -> merge to vmm
```

Create an issue first when it helps record motivation, acceptance criteria,
scientific questions, dependencies, or backlog status independently of a
particular implementation. This is especially useful if the work may be
paused, delegated, or split into multiple PRs.

### When to open a draft PR early

A draft PR is useful when you want any of the following before the work is
finished:

- CI on the evolving branch;
- visibility into the current diff;
- review comments on design/direction;
- a stable URL for the agent/user to coordinate around;
- stacked or dependent work that needs an explicit base.

Do **not** create an empty PR merely so an agent has somewhere to work. The
branch is the implementation workspace; the PR is the review/integration view.
A good default is to open a draft PR after the first coherent commit, not before
any code exists.

### When the PR should be the last step

For a tiny, self-contained change where CI is not needed during development,
it is perfectly reasonable to implement and test on the feature branch first,
then open the PR when the diff is ready for review. The important point is that
**merging** is the last step; PR creation may be early or late depending on
whether it helps coordination.

## 4. Recommended agent workflow

### Main agent

Use the more capable/main agent as owner of:

- interpreting the request;
- deciding scientific scope;
- choosing the branch/PR boundary;
- reviewing the repository state before edits;
- final diff review and integration decisions;
- PR description and handoff;
- deciding whether a subagent is useful.

### Subagent

Delegate only a bounded job, for example:

- implement one clearly specified component;
- investigate a reproducible failure;
- resolve a conflict under explicit constraints;
- independently reproduce a scientific/numerical result;
- perform an adversarial review of a completed diff.

Do not spawn a subagent automatically for every task. The extra context and
coordination are worthwhile only when parallelism, isolation, or independent
verification adds value.

### Branch/worktree ownership

Use **one branch/worktree per deliverable**, not one per agent. A subagent
normally contributes toward the main deliverable unless it is genuinely
producing an independently mergeable change. This reduces the branch/worktree
sprawl seen during the August/September development burst.

## 5. Issues, branches, PRs, and handoffs: what each is for

| Object | Use it for | Do not use it as |
| --- | --- | --- |
| Issue | problem statement, acceptance criteria, backlog, scientific question | a transcript of every agent action |
| Branch | isolated implementation state for one deliverable | permanent project memory |
| Draft PR | evolving review/CI/coordination surface | a substitute for a scoped branch |
| Ready PR | coherent, reviewable integration proposal with evidence | a scratchpad |
| PR description/comments | concise implementation and verification handoff | a huge machine-state dump |
| Long handoff/checkpoint | context compaction or genuinely long scientific investigation | mandatory ceremony for small changes |

For ordinary work, the PR plus commit history and tests should be enough to
continue later. If a continuation record is necessary, keep it concise unless
there is genuinely machine-readable state that cannot be reconstructed from
Git/artifacts.

## 6. How to use the project skills

Skills are **on demand**, not prerequisites for every run.

- Use `cyaxiverse-julia-quality` for Julia numerical kernels, HDF5 readers,
  package regressions, type/precision issues, optional Python boundaries, and
  audit/test hygiene.
- Use `cyaxiverse-scientific-reproduction` for source-paper reconstruction,
  benchmarks, physical-scale claims, basis/coordinate validation, and
  population conclusions.
- Use `cyaxiverse-ks-geometry-sampling` for Kreuzer-Skarke/CYTools sampling,
  target-population definitions, triangulation generation, and bias/coverage
  accounting.
- Use `cyaxiverse-vacua-pipeline` for minima/vacuum batch pipelines, resumable
  HDF5-backed jobs, no-overwrite behavior, and persistence contracts.
- Use `cyaxiverse-integration-release` for conflict resolution, branch
  sequencing, integration review, worktree consolidation, and the `vmm -> main`
  release boundary.

Generic tools such as Code Foundations, second-opinion agents, ADHD formatting,
or autoresearch may still be valuable. Keep them local and invoke them
explicitly when the task benefits from them rather than imposing them on every
repository interaction.

## 7. Relationship to the repository consolidation plan

Treat this AI cleanup as the **control-plane** track, not a separate management
system.

The broader consolidation should continue to use these principles:

- maintain one canonical view of active work;
- classify branches/worktrees before deleting them;
- prune stale/prunable worktrees and superseded branches only after confirming
  their commits are merged or intentionally abandoned;
- let issues represent outstanding work, not obsolete historical states;
- use PRs as the durable integration record;
- validate current `vmm` before declaring cleanup complete;
- keep scientific work gated separately from mechanical repository cleanup;
- retain `vmm -> main` as the deliberate release/integration boundary.

Do not create another always-on ledger unless the current GitHub state cannot
answer the question. If a lightweight active-work index is introduced, it
should summarize GitHub state rather than become a competing source of truth.

## 8. Local migration after this PR merges

Your uploaded working tree had local modifications. Before pulling the merged
cleanup, inspect them rather than blindly updating:

```sh
git switch vmm
git status --short --branch
git diff --stat
git diff
```

If the local edits are valuable, either commit them to their proper feature
branch or save a reversible patch/stash according to your normal workflow.
Then update `vmm`:

```sh
git fetch origin
git pull --ff-only origin vmm
```

If you want to keep personal settings that this PR removes from tracking, copy
or recreate them **after** the pull. They will remain ignored by Git.

Example local Claude plugin settings:

```json
{
  "enabledPlugins": {
    "code-foundations@rtd": true,
    "what@rtd": true
  }
}
```

Place that in `.claude/settings.json` locally if you still want those plugins.
Likewise, install or copy the ADHD skill into a local/global Codex skill area if
you want that response style; it is no longer an invariant of CYAxiverse.jl.

After updating, verify the tracked project skill links:

```sh
ls -l .codex/skills/cyaxiverse-*
ls -l .claude/skills/cyaxiverse-*
git status --short
```

Your personal settings should be ignored and the tracked project files should
be clean.

## 9. Suggested working rhythm

For a normal piece of work:

1. Check `vmm` is current and clean enough for the intended branch point.
2. Create an issue if the work benefits from durable scope/acceptance criteria.
3. Create one feature/fix branch from `vmm`.
4. Ask the main agent to inspect, scope, and implement; delegate only bounded
   subproblems.
5. Make coherent commits with evidence.
6. Open a draft PR after the first coherent commit when CI/review visibility is
   useful; otherwise open it when ready.
7. Run focused verification, then applicable package/audit/docs/CI gates.
8. Review the final diff and scientific claim boundary.
9. Mark the PR ready and merge to `vmm`.
10. Periodically consolidate/prune stale branches and worktrees; do not leave
    every historical agent branch alive forever.
11. When a coherent release is ready, review the aggregate version impact and
    open the `vmm -> main` release PR.

This gives agents enough structure to work safely without turning process files
into a second codebase.
