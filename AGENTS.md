# CYAxiverse.jl Agent Contract

This file is the canonical repository instruction set for AI coding agents.
Tool-specific adapters may point here, but must not redefine project policy.
Read additional skills or handoffs only when they are relevant to the task.

## 1. Work safely

- Inspect the current branch and working tree before editing. Preserve unrelated
  user changes; do not reset, clean, stash, switch branches, or overwrite work
  unless the user explicitly authorizes it.
- Keep changes scoped to the requested deliverable. Do not silently expand a
  bug fix into a redesign or a scientific investigation into a new model.
- Ordinary durable state belongs in Git: issue (when useful), branch, commits,
  PR, tests, and concise PR notes. Use long continuation handoffs only for
  genuinely long investigations or context-compaction boundaries.
- Prefer one branch/worktree per deliverable. Subagents normally work toward
  the same deliverable; do not create a new branch/worktree for every reasoning
  step.

## 2. Environment and verification

- Target Julia 1.12 as declared by the project and CI.
- For local package development, run Julia in the regular host environment,
  not in Docker or an isolated sandbox that changes filesystem, Python, BLAS,
  or database behavior. Repository CI remains the clean-checkout gate.
- Use `scripts/agent_verify.py` as the compact verification entry point when
  applicable:
  - `python3 scripts/agent_verify.py snapshot`
  - `python3 scripts/agent_verify.py diff-check`
  - `python3 scripts/agent_verify.py run -- <focused-command>`
  - `python3 scripts/agent_verify.py package`
- Run focused checks first, then the broader package/audit/docs checks required
  by the change. Report commands actually run, exit status, observed result,
  warnings, and unavailable checks. Never claim a check from expected output or
  source comments alone.

## 3. Julia and persisted-data invariants

- Preserve precision intentionally. Do not force `Float64` through existing
  high-precision, exact-rational, `BigInt`, or `ArbFloat` paths.
- Keep numerical hot paths type-stable and avoid unnecessary allocations;
  preserve sparse handling of large intersection data.
- Validate reader/writer boundaries for shape, orientation, units, identity,
  and schema. Preserve established HDF5 paths and compression (`deflate=9`)
  unless a reviewed schema change requires otherwise.
- Keep Python/CYTools optional for core Julia package import. Do not make
  `using CYAxiverse` depend on a live Python object or an optional scientific
  environment.
- Physical-domain checks apply where the mathematical object is required to be
  physical. In particular, Kähler volumes and kinetic metrics must satisfy the
  applicable domain constraints. Do not impose positivity on Hessian/mass
  directions when the code is intentionally studying saddles or tachyonic
  modes; preserve the distinction between physical minima and diagnostic
  critical points.

## 4. Scientific claim boundary

For scientific, numerical, sampling, benchmark, or persisted-data changes:

- Distinguish source facts, implementation facts, empirical verification,
  owner-approved extensions, and inference.
- Record enough identity to replay the result: source/revision, selection
  route, counting unit, geometry/witness identity, units, schema, code revision,
  and relevant environment/tool versions.
- Validate progressively: analytic/synthetic fixture -> named source fixture ->
  bounded replay -> population execution. Stop when an earlier gate fails.
- Expected aggregate counts are gates only when source-verified. Matching
  counts do not establish matching populations; compare identities/witnesses
  when equality of populations matters.
- Do not promote finite, filtered, provisional, homotopy-only, or structurally
  complete results into population-level or physical claims without the
  required evidence and scientific-owner approval.
- Stop for owner direction before changing a physical normalization, scientific
  acceptance criterion, mass/tachyon interpretation, basis convention,
  reported observable, population definition, or scientific schema when the
  intended convention is ambiguous.

## 5. Tests, compatibility, and version impact

- Add regression coverage appropriate to the changed layer. Julia changes often
  belong in `test/`; Python/CYTools tooling may require its existing Python test
  harness instead. Do not alter or bypass tests merely to force a pass.
- Preserve compatibility unless the task intentionally changes a public API,
  reader/writer contract, persisted schema, supported environment, or scientific
  behavior. State any intentional compatibility break explicitly.
- `Project.toml` is the package version source of truth. Feature branches state
  version impact but normally do not bump it. Apply the reviewed release bump at
  the `vmm -> main` integration boundary. Keep scientific artifact/schema
  versions separate from the package version.

## 6. Git, PRs, and agent delegation

- Prefer an issue for work that benefits from a durable problem statement,
  acceptance criteria, or backlog visibility. Tiny fixes do not require one.
- Create a focused branch from the intended base before implementation.
- Opening a draft PR relatively early is useful when you want CI, a stable
  review URL, or visibility into the evolving diff. Otherwise open the PR once
  the first coherent change exists. Mark it ready only after the deliverable
  and required verification are complete.
- The main agent owns scope, scientific interpretation, integration, final diff
  review, PR state, and handoff. Delegate to a subagent only when bounded
  implementation, parallel investigation, or independent verification has real
  value.
- AI-assisted commits should retain an appropriate `Co-Authored-By:` trailer
  when the tool authors the commit. The human submitter remains responsible for
  reviewing the contribution.

### Delegated-task lifecycle

Give each delegated worker one bounded objective, observable acceptance
criteria, relevant files/artifacts/inputs, material constraints, and explicit
escalation conditions. A delegated implementation worker normally owns the
diagnose -> edit -> test -> correct -> retest loop. Do not replace a worker for
an initial failure, a failed test, or an ordinary implementation correction.

Normal worker terminal returns are `DONE`, `BLOCKED`, and `FAILED`. A
long-running task may use one predeclared health `CHECKPOINT`. Prefer
completion-driven waiting over repeated progress polling; elapsed silence alone
does not establish a stall. Stall recovery is bounded: inspect one checkpoint;
if concrete progress exists, extend once; otherwise issue one recovery
instruction; if recovery also fails, interrupt or replace the worker from
durable state.

Manager-visible handoffs should be concise and evidence-oriented: result/status;
changed files or artifacts; exact checks and observed outcomes; relevant
scientific assumptions, conventions, or decisions; unresolved blockers; and
durable Git/SHA/path/artifact references. Keep raw logs, full transcripts, and
large derivations in files or artifacts unless they are needed to resolve a
specific contradiction.

## 7. Project skills

Project-specific reusable workflows live under `.agents/skills/` and are
mirrored to tool-specific skill directories by tracked symlinks. Use only the
skill relevant to the task; they are not mandatory pre-reading for every run:

- `cyaxiverse-agent-orchestration` (delegated multi-agent work only)
- `cyaxiverse-julia-quality`
- `cyaxiverse-scientific-reproduction`
- `cyaxiverse-ks-geometry-sampling`
- `cyaxiverse-vacua-pipeline`
- `cyaxiverse-integration-release`

Generic/personal skills, plugin settings, permissions, response-style
preferences, and experimental agent tooling are local user configuration, not
repository policy.
