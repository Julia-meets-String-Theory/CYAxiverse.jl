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
- Treat private chats, local agent sessions, local filesystem context, connected
  applications, and private attachments as non-public by default. Before any
  durable GitHub write derived from them, deliberately sanitize the material.
  Unless the owner explicitly approves the exact datum for publication, do not
  publish private conversation/share URLs or transcript dumps; absolute local
  filesystem/home-directory paths; local usernames, hostnames, machine/device
  identifiers; local Codex/agent/workspace/session paths; private attachment or
  connector-local locations; secrets; credentials; or tokens. Refer to
  repository content with repository-relative paths. If publication safety is
  uncertain, omit the datum and stop for owner direction.
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
  version impact but normally do not bump it. A reviewed closure adopts the
  final package version on its owner line; principal promotion then verifies
  that the certified release tree and version reach `main` at the deliberate
  `vmm -> main` integration boundary. Keep scientific artifact/schema versions
  separate from the package version. Pre-retrofit work keeps its historical
  declaration and evidence under the approved transition rules.

### Package iteration and release lifecycle

- `vmm` is the principal development line. Under the adopted lifecycle, `main`
  is intended to carry the latest certified principal release after reviewed
  reconciliation; pre-retrofit `main` history is not retroactively certified
  by this policy and never carries a `-DEV` package version. A
  `maintenance/X.Y` line may carry exact-tree certified maintenance releases
  on that `X.Y` lineage without moving the principal `main` line backward.
- Record a stable target-iteration identity before assigning a final SemVer.
  For CYAX-0125 Gate A the target is `version-lifecycle-retrofit-2026-09`, the
  package infrastructure impact is `patch`, and the package remains at
  `0.2.0`; this gate does not adopt a development version or publish a release.
- An active `X.Y.Z-DEV` reserves final `X.Y.Z` globally for its owner line.
  Other lines cannot close, candidate, or publish that version. Closing the
  reserved final consumes it; closing a different final records the reserved
  identity as `CONSUMED_UNUSED_DEV_RESERVATION`. Closed, candidate, withdrawn,
  released, and consumed versions are never reused. A reservation with proven
  pre-entry abort may be made available again under the recovery rules;
  uncertain outcomes remain unavailable.
- Lifecycle state is represented by protected create-once Git refs and small
  immutable canonical evidence manifests, together with `iterations.toml`,
  immutable iteration anchors, and protected public tags. A lifecycle ref is
  never deleted, repointed, force-updated, or treated as mutable ledger state;
  no `release-events` branch/stream is canonical. Claim, reservation,
  candidate, intent, release and publication object types remain distinct;
  progression is an immutable predecessor-ref graph. Allocation and transition writers use
  create-if-absent/CAS, exact snapshot/ref verification, owner authorization,
  and fail closed on an uncertain remote result. A proven pre-entry reservation
  abort consumes its reservation identity but may release only an unclaimed
  final version; uncertainty keeps that version unavailable.
- A release candidate is certified against an immutable exact tree and retains
  its candidate, commit, tree, version, tag intent, certification, and release
  evidence identities in those immutable refs/manifests. Public canonical tags
  are irreversible and are checked against the corresponding release manifest
  and publication evidence.
- Gate A requires and tests automation for the first principal lifecycle path
  (principal allocation/reservation, closure/anchor, candidate, certification,
  canonical tag and release evidence). It does not perform historical
  designation, package-version adoption, a production `-DEV` transition,
  closure, public-tag creation, publication, or `vmm -> main` reconciliation.
  Maintenance-line bootstrap/release and rare recovery automation are deferred
  to a later approved S2 gate. Gate B and the deliberate `vmm -> main` release
  boundary handle those actions.
- Tracked installation and documentation source remains release-neutral. The
  verified ref and release manifest select development, principal-versioned,
  maintenance-versioned, or stable documentation channels; a public release
  must not require editing tracked source files.
- Python used by lifecycle scripts is bounded repository/CI control-plane
  tooling only. It is not a runtime dependency of the Julia package, and
  `using CYAxiverse` remains operable without Python, PyCall/CYTools, or their
  scientific environments. These rules do not change Julia APIs, scientific
  behavior, persisted scientific schemas, package-version grammar, or the
  Gate A/Gate B boundary.

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

### Specification-driven work

Classify substantial work as S0–S3 using `cyaxiverse-sdd`. S0 work remains
lightweight; for S1–S3 work, locate the governing feature specification before
changing intended behavior, or draft one if none exists.

For S2/S3 work, do not implement consequential behavior while the governing
spec is still draft. Do not infer unresolved scientific normalization, basis,
population/counting, acceptance, physical-interpretation, or schema choices;
return them to the scientific owner. If investigation changes intended behavior,
update and re-review the spec before continuing the affected work.

Treat an approved `spec.md` as feature intent below this repository contract;
`plan.md` and `tasks.md` are subordinate implementation artifacts. Before
claiming S2 completion, reconcile spec, plan, tasks, implementation,
tests/evidence, and PR scope. GitHub Issues/Projects track work state but do not
supersede the governing spec. Do not create a GitHub issue for every task; use
sub-issues only for independently durable, blocked, mergeable/reviewable, or
owner-decision-bearing work.

## 7. Project skills

Project-specific reusable workflows live under `.agents/skills/` and are
mirrored to tool-specific skill directories by tracked symlinks. Use only the
skill relevant to the task; they are not mandatory pre-reading for every run:

- `cyaxiverse-agent-orchestration` (delegated multi-agent work only)
- `cyaxiverse-sdd` (S1–S3 specification/planning/convergence work)
- `cyaxiverse-julia-quality`
- `cyaxiverse-scientific-reproduction`
- `cyaxiverse-ks-geometry-sampling`
- `cyaxiverse-vacua-pipeline`
- `cyaxiverse-integration-release`

Generic/personal skills, plugin settings, permissions, response-style
preferences, and experimental agent tooling are local user configuration, not
repository policy.
