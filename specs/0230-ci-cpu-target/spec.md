---
spec_id: CYAX-0230
title: CPU-dependent package images in the CI depot caches
issue: 230
class: S1
status: draft
workstream: CI workflows, their depot caches and their tests
parent: null
depends_on: []
created: 2026-11-09
last_reviewed: 2026-11-09
review_required: repository-owner approval
approval_ref: null
version_impact: none (CI only)
---

# CPU-dependent package images in the CI depot caches

## Objective

Measure, on real CI runs, whether restored package images are rejected
because the runner's CPU differs from the CPU that built them. If they are,
choose and implement a fix by an amendment to this specification, so that a
job that restores a depot cache uses the package images in it whatever CPU
model the runner has.

## Current baseline

Facts at `vmm` `10aa3dc`.

- Six jobs restore a Julia depot cache with `julia-actions/cache@v2`:
  `lifecycle`, `fast`, `fast-cache`, `qualification` and `full` in `CI.yml`,
  and `build` in `Documentation.yml`. No workflow sets `JULIA_CPU_TARGET`.
  Every job uses `julia-actions/setup-julia@v2` with version `1.12`; the runs
  below used 1.12.7.
- Julia 1.12.7:
  - `src/init.c` sets the CPU target to `native` when none is given, and
    `base/loading.jl` compiles package images for that target when
    `JULIA_CPU_TARGET` is not set.
  - A restored image whose recorded CPU targets fail `check_clone_targets`
    on the current CPU is rejected (reason "target mismatch") and the package
    is compiled again. Every cache file of a package is a candidate: a
    different CPU target string gives a different file name, but an older
    file that passes the check still loads.
  - The manual (*Environment variables*) says that `JULIA_CPU_TARGET` sets
    the CPU targets of system and package images written to disk, and does
    not change just-in-time compilation.
- Julia's official x86-64 binaries use the list
  `generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)`
  (`JuliaCI/julia-buildkite`, `utilities/build_envs.sh`, unchanged from
  `fe8b26e7bed8`, 2024-03-08, through `261605ebc204`).
- `src/minimizer.jl` and `src/generate.jl` load `LoopVectorization`, but no
  file under `src/` calls `@turbo` or `@tullio`. Whether a dependency runs
  LoopVectorization code on the tested or qualified paths has not been
  checked. `HostCPUFeatures` (a LoopVectorization dependency, v0.1.18 in CI):
  when the precompile CPU target does not contain `native`, it fixes its CPU
  features at precompile time. With default preferences every feature is then
  false, whatever the first entry of the list (`setfeaturetrue` only warns
  unless the `allow_runtime_invalidation` preference is set), so there is no
  AVX and no FMA on any runner.
- Runs of the Fast tests and Fast tests (cache producer) jobs (about 523
  dependencies):
  - The Fast tests job of PR #227 runs 42328257352, 42530695871, 42536176418
    and 42540123090, and the producer job of push run 42664830366, all
    restored `…job=fast;os=Linux;run_id=42326043079`.
  - The PR runs precompiled 0, 138, 139 and 139; the non-zero counts start
    with packages that had new releases (`StaticArraysCore`, `Hwloc_jll`).
  - The push run precompiled 479, starting with packages that have no
    dependencies (`RangeArrays`, `LaTeXStrings`); its Python tests and
    qualification jobs precompiled 297 and 276. Julia version, runner image
    and the `Pkg.gc()` result were the same as in the previous push (#223,
    evidence 5).
  - Earlier counts are not explained by a CPU difference: the 7 on PR #225
    and push run 42255426058 (the same packages, `FFTA` and `KernelDensity`
    first, in both runs) and the 57 on push run 42326043079 (starting with
    the released `DifferentiationInterface`). The root `Manifest.toml` is not
    tracked, so each run resolves new releases.
  - Job times in the runs above: 628 s (42250911884) to 1708 s
    (42664830366).
- The Fast tests jobs precompile under `--check-bounds=yes` (the
  `julia-runtest` "Precompiling for configuration" line); the lifecycle and
  qualification jobs precompile with the default flags. Julia checks a cache
  file's header, then its flags, then its CPU targets.
- The job logs do not record the runner's CPU model.

## Scope

- A diagnostic step in every job that restores a Julia depot cache, and its
  tests in `scripts/test_ci_cache_workflow.py`.
- The fix, only as defined by an approved amendment (R-004).

## Non-scope

- Julia and package versions, and the test suite.
- `WikiRefresh.yml` (it sets up Julia but keeps no cache) and the other
  workflows.
- Local development environments.

## Requirements

### R-001 — CPU diagnostic in every cached job

In each of the six jobs, a step named `Report CPU and restored package
images`, directly after `julia-actions/cache`, prints:

- the runner's CPU model name and `Sys.CPU_NAME`;
- the value of `JULIA_CPU_TARGET`, or that it is not set;
- for the restored cache files, per recorded CPU target and cache flags: the
  number of files; how many have a header this Julia cannot read; how many
  have no object file; and how many `Base.check_clone_targets` rejects on
  this runner (called only when the file records CPU targets);
- inside a collapsed log group, for each package and flags value: the package
  name, the flags, the number of restored files, the number rejected by
  target, the number that pass the header, target and object-file checks, and
  whether the header's required modules include a package other than `Core`,
  `Base`, `Main` and the packages in `Sys.STDLIB`;
- the number of depot files newer than a marker file the step writes outside
  the depot before it starts (expected 0).

The step has the same text in all six jobs, does not contain the word `gc`
(the CYAX-0223 placement test counts it), must not fail the job
(`continue-on-error: true`), and stays after this work, so that later
evidence records the CPU.

### R-002 — Confirm the cause

The first PR CI run is on the R-001 and R-003 commit, which changes no other
workflow line. The hypothesis is:

- confirmed if a job recompiles a package whose required modules R-001
  reports as only `Core`, `Base`, `Main` and standard-library packages, whose
  version was already in the restored depot (the job log, after colour codes
  are removed, has no `Installed` line for it), and for which R-001 counts at
  least one restored file for the flags the job precompiled with, and counts
  every such file as rejected by target;
- refuted if such a package is recompiled although R-001 counts, for those
  flags, a restored file of it that passes the header, target and object-file
  checks. A package with no restored file for those flags is not evidence.

If no job of the run recompiles such a package, the cached jobs may be re-run
up to the number set by owner decision D1. If the hypothesis is refuted or
still undecided after those runs, the work stops and the owner decides the
next step, including whether R-001 merges alone.

### R-003 — Tests for R-001

`scripts/test_ci_cache_workflow.py` checks, in both `CI.yml` and
`Documentation.yml`, that every job that uses `julia-actions/cache` has the
step named `Report CPU and restored package images` directly after the cache
step, with `continue-on-error: true`, and that the step text is the same in
all six jobs. The PR records the
test's failing output against the parent commit.

### R-004 — Fix (by amendment)

If R-002 confirms the cause, an amendment to this specification, reviewed and
approved like this one, defines the fix before it is implemented. It must
address:

- the CPU target list and the source that supports it;
- the effect on `HostCPUFeatures` and on any LoopVectorization code that the
  tests and qualification run, with evidence of what runs and whether results
  change;
- the alternatives considered, including cache entries keyed by CPU model;
- the `HostCPUFeatures` preferences `cpu_target`, `freeze_cpu_target` and
  `allow_runtime_invalidation`;
- the cache entries built before the fix, within CYAX-0223 R-004 (only `fast`
  and `fast-cache` set a `cache-name`) or by amending it; GitHub removes cache
  entries not accessed for 7 days;
- compile time and entry size, measured, against the 10 GB repository cache
  limit (4065 MiB in use after run 42664830366);
- its own G1 and G2 rows;
- if the fix changes any qualification result or tolerance, the work is
  reclassified (L-0012) before it continues.

## Accepted limitations

- The recompilation continues until the fix merges.
- The step reads cache file headers with internal Julia functions
  (`Base.parse_cache_header`, `Base.check_clone_targets`); a Julia update may
  break it. Because it cannot fail the job, a break shows as a missing table,
  not a failed run.

## Acceptance gates

### CYAX-0230 G0 — Specification approval

The owner approves this specification, including decisions D1 and D2, in a durable
comment that names its commit SHA. The R-004 amendment needs its own approval
comment.

### CYAX-0230 G1 — Implementation (pre-merge)

- R-001 and R-003 in one commit; the PR records the R-003 test failing
  against the parent commit.
- R-002 evidence in the PR description: per job of the PR run (lifecycle,
  fast, qualification and the Documentation build), the CPU
  model, the target table, the per-package table and the precompile count.
  `fast-cache` and `full` do not run on this PR; they are recorded at G2.
  The Documentation cache entry was last used on 2026-11-03, so GitHub will
  probably have removed it (7 days unused) and that job's table may be empty;
  an empty table is not a failure.
- If confirmed: the approved R-004 amendment, its implementation and tests,
  and a passing PR CI run.
- All Python tests pass.
- An independent review of the exact candidate SHA, reported separately from
  CI.

### CYAX-0230 G2 — Post-merge evidence (on #230, not a merge gate)

| Observation | Required record |
| --- | --- |
| First push to `vmm` after merge | Per cached job: CPU model, target table, precompile count, job time, entry size |
| Later PR runs, until two different CPU models have been seen or the number of runs set by owner decision D2 is reached | Per cached job: CPU model, target table and precompile count |

## Interfaces and compatibility

CI only. R-001 changes no package, API, schema, scientific or version
behaviour. CI results are already CPU-dependent today, because images are
built for `native` on whichever runner builds them. The R-004 amendment must
state its own numerical effect.

## Owner decisions

The owner chose these values before this spec was committed; the G0 approval
comment confirms them.

- D1: how many re-runs of the PR's cached jobs R-002 may use if the first run
  decides nothing. Each run samples four runners (lifecycle, fast,
  qualification, and the Documentation build, which runs because the PR
  changes `Documentation.yml`); Fast tests jobs took 628 s to 1708 s and
  qualification jobs up to 2953 s. The share of runners
  with each CPU model is unknown, so no number of runs guarantees a different
  CPU; the bound is a cost choice. Decision: up to 2 re-runs.
- D2: how many PR runs G2 records if two CPU models have not been seen.
  Decision: 3, a cost choice on the same basis.

## Completion criterion

CYAX-0230 G0 and G1 pass and the PR is merged into `vmm`; CYAX-0230 G2 is
recorded on #230. If R-002 does not confirm the cause, the owner decides
under R-002 whether R-001 merges alone; the evidence is recorded on #230.
If the owner decides that nothing merges, the work is complete when that
decision and the evidence are recorded on #230.
