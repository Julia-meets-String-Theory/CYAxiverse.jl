---
spec_id: CYAX-0230
title: CPU-dependent package images in the CI depot caches
issue: 230
class: S1
status: approved
workstream: CI workflows, their depot caches and their tests
parent: null
depends_on: []
created: 2026-11-09
last_reviewed: 2026-11-11
review_required: repository-owner approval
approval_ref: "PR #231 comment 6781024991 — approves spec revision 47a32b1, with owner decisions D1 (2 re-runs) and D2 (3 PR runs); PR #231 comment 6797717210 — approves Amendment 1 at 5bb0750, with the R-007 deletions done by the owner"
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

## Amendment 1 — the R-004 fix

Status of this amendment: approved at `5bb0750` by the owner in PR #231
comment 6797717210. The deletions of R-007 steps 2, 4 and 5 are done by the
owner.

R-002 confirmed the cause on CI run 42925415808 and Documentation run
42925415827 (`ba2364b`; evidence in the PR #231 description). This amendment
defines the fix. R-005 to R-008 are its requirements.

### Facts for the amendment

Julia 1.12.7 source:

- `base/loading.jl` (`compilecache_path`, `create_expr_cache`): when
  `JULIA_CPU_TARGET` is set, the precompile process compiles the package
  image for that target. The value is also part of the cache file name.
  Code compiled at run time stays for the runner's own CPU
  (`src/init.c`: `native` when no `-C` option is given).
- `src/processor.cpp` (`match_sysimg_targets`) and `src/processor_x86.cpp`
  (`pkgimg_init_cb`): a restored image is rejected if its target uses a CPU
  feature that the process's code-generation target lacks. The CPU name only
  ranks the candidates. In the test or qualification process that target is
  `native`, the runner's CPU. In a precompile process started by
  `create_expr_cache` (`-C` set to `JULIA_CPU_TARGET`), it is `haswell`,
  limited to the runner's features. A precompile process therefore rejects
  every image built for a CPU with features beyond `haswell`, which includes
  every native image built on the runners seen, and compiles those
  dependencies again inside it. Pkg's "dependencies successfully precompiled"
  count does not include these nested compilations. The reviewer reproduced
  this on Julia 1.12.6 with `JULIA_CPU_TARGET=generic` against native images.
- `src/processor_x86.cpp`: `haswell` is a known target name. Its features
  are the Ivy Bridge set plus `avx2`, `bmi`, `bmi2`, `fma`, `lzcnt` and
  `movbe`, without AVX-512. `rdrnd` and `xsaveopt` are always removed from
  image targets. `x86-64-v3` is not a known name.
- Older files of a package stay in the depot beside new ones (up to 10 per
  package, `MAX_NUM_PRECOMPILE_FILES`). Any file that passes the checks loads,
  whatever target it records.

`HostCPUFeatures` v0.1.18 (`src/HostCPUFeatures.jl`, `src/cpu_info_x86.jl`):

- When the precompile target does not contain `native`, the package fixes
  its CPU features at precompile time (`freeze_cpu_target` becomes true).
- For a target name containing `well`, `make_generic_x86` sets `avx`,
  `avx2`, `bmi2` and `fma`. With the default preference
  `allow_runtime_invalidation = false`, it only warns, and every feature
  stays false.
- The preference `cpu_target` replaces the target name it reads.
  `freeze_cpu_target` is already true here.

Base reads a package's preferences only from projects that list the
package in `[deps]`, `[extras]` or `[weakdeps]` (`get_uuid_name`).
`Pkg.test` copies `Base.get_preferences()` into its sandbox.

LoopVectorization code in CI, at `ba2364b`:

- No file under `src/`, `ext/`, `test/`, `bin/` or `scripts/` calls
  `@turbo`, `@tullio` or `@avx`, or builds a `TensorProductOperator`.
- `Tullio` v0.3.9, a direct dependency, loads LoopVectorization support
  through a Requires.jl hook, and `@tullio` emits LoopVectorization code;
  nothing calls `@tullio`.
- In the resolved CI dependency tree (job logs, General registry
  dependency ranges for the versions used), the only other package code that
  uses LoopVectorization is LoopVectorization itself and
  `SciMLOperatorsLoopVectorizationExt` (SciMLOperators v1.30.4). That
  extension uses `@turbo` only in `tensor_outer_mul_fast!`, for tensor-product
  operators. `RecursiveFactorization`, `TriangularSolve` and `Octavian` are not
  in the tree.

Measurements on the unmerged branch `experiment/cyax-0230-cpu-target-haswell`
(`7858274`, the head commit of each run below): `ba2364b` plus `JULIA_CPU_TARGET: 'haswell'`, running only Fast
tests and qualification, each with its own experiment cache. The
Documentation build and the lifecycle job were not run; they use the same
mechanism, and the G1 PR run covers them:

| Run | Job | CPU (`Sys.CPU_NAME`) | Restored images (all `haswell`) | Rejected by target | Precompiled (time) | Job time | Entry saved | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42950743815 | Fast tests | `znver4` | none (cold) | — | 479 (1081 s) | 26.6 min | 710 MiB | tests pass |
| 42950743815 | Qualification | `znver5` | none (cold) | — | 276 (461 s) | 34.2 min | 265 MiB | PASS 6/6, 28/28 |
| 42954635686 | Fast tests | `znver4` | 485 (from run 42950743815) | 0 | 0 | 10.3 min | — | tests pass |
| 42954635686 | Qualification | `icelake-server` | 276 (from run 42950743815) | 0 | 0 | 35.8 min | — | PASS 6/6, 28/28 |
| 42958611390 | Fast tests | `znver4` | 485 (from run 42954635686) | 0 | 0 | 8.4 min | — | tests pass |
| 42958611390 | Qualification | `znver3` | 276 (from run 42954635686) | 0 | 0 | 37.9 min | — | PASS 6/6, 28/28 |
| 42962732225 | Fast tests | `znver5` | 485 (from run 42958611390) | 0 | 0 | 7.6 min | — | tests pass |
| 42962732225 | Qualification | `znver5` | 276 (from run 42958611390) | 0 | 0 | 24.9 min | — | PASS 6/6, 28/28 |

Runs 2 to 4 saved entries of the same sizes. In run 42950743815, both jobs
printed the 4 `HostCPUFeatures` warnings (`avx`, `avx2`, `bmi2`, `fma`) while
it was precompiled; with `native` images it printed none. Each experiment entry was deleted
after the next run had restored it, and the last ones after run 42962732225.
The branch was deleted after the runs.

For comparison, before the fix: earlier `vmm` pushes saved native Fast tests
entries of 711 to 816 MiB (CYAX-0223 evidence 3 and 4), and 1159 MiB for
run 42664830366, which held images for two CPU targets. The push run
42664830366 recompiled 479 packages in 1153 s on `graniterapids`. Qualification
entries were 267 to 419 MiB.

Cache use against the 10 GB limit: 4849 MiB on 2026-11-11 (7 `vmm` entries
and 3 PR #231 entries). One `vmm` generation of `haswell` entries is about
1.7 GB (Fast tests 710 MiB and qualification 265 MiB measured; lifecycle
and Documentation build about 355 and 365 MiB, as their native entries).
Cache maintenance runs at the start of a push, before the push saves its
entries (CYAX-0223 R-007), so between pushes `vmm` holds two generations:
about 3.4 GB. Each open PR adds up to about 1 GB (lifecycle, qualification,
Documentation build).

Nothing removes stale image files from a depot: `Pkg.gc` does not touch
`compiled/`, and Julia removes a package's oldest file only when it has 10.
The R-002 Fast tests entry (1159 MiB) held 479 `graniterapids` and 549
`znver3` files. If the native entries stayed, every new entry would hold the
native images plus the `haswell` ones, about 3.9 GB per generation
(1159 + 710, 416 + 265, 354 + 355 and 351 + 365 MiB), about 7.8 GB on `vmm`
and about 2.1 GB per PR: one or two open PRs would pass 10 GB. Packages whose
only image is native, and which no precompile process needed on the push
runner, would also still be compiled again on every PR runner that rejects
them, because PR Fast tests jobs save nothing. R-007 therefore deletes the
old entries.

### Alternatives considered

- **Julia's official x86-64 list**
  (`generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)`).
  It compiles every function at least twice, and some three or four times,
  so compile time and entry size grow. CI gains nothing from the extra
  targets. `HostCPUFeatures` reads the first entry, `generic`, so its features
  are all false here too.
- **Cache entries keyed by CPU model.** Every job would need a `cache-name`
  that contains the model. That needs a CYAX-0223 R-004 amendment and a
  step to read the model before the cache step. Each model needs its own
  entry, so 6 models seen so far, in 4 cached jobs, would hold about six times
  today's entries against the 10 GB limit. A PR would hit only if a `vmm`
  push had run on the same model.
- **A target list that contains `native`.** `HostCPUFeatures` would then take
  the features of the CPU that built it, and on an older runner LoopVectorization
  could emit instructions that the runner lacks.
- **No change.** Every run on a CPU without a feature of the building CPU
  recompiles, as R-002 showed.

### R-005 — One CPU target for package images

`CI.yml` and `Documentation.yml` each set, at workflow level,
`env: JULIA_CPU_TARGET: 'haswell'`, with a comment that cites this
specification and does not contain the text `JULIA_CPU_TARGET`. No job or step in them sets `JULIA_CPU_TARGET`. `WikiRefresh.yml`
keeps no cache and is not changed.

### R-006 — HostCPUFeatures

No `HostCPUFeatures` preference is set. `Project.toml` and
`LocalPreferences.toml` do not change. Under R-005, `HostCPUFeatures` reports
no AVX, AVX2, BMI2 or FMA, and prints 4 warnings when it is precompiled.

Reason: the only LoopVectorization code in the dependency tree is not on the
tested or qualified paths (*Facts*). Making the preference effective would add
`HostCPUFeatures` to the package's `[extras]` and test target, a change to
package metadata for code CI does not run. If LoopVectorization code ran, it
would choose 16-byte vectors, and VectorizationBase would choose its
algorithms without fused multiply-add where it asks `fma_fast()` (for example
`dmul`, `ddiv` and `dsqrt` in `src/special/double.jl`, and `exp`). That is
valid code on every runner, but slower, and its results can differ in the
last bits. None of this code is on the tested or qualified paths.

### R-007 — Package images built before the fix

R-005 does not change any `cache-name`, so CYAX-0223 R-004 is unchanged.
CYAX-0223 R-007 is unchanged too: its item 7 binds only the cache maintenance
job, and the deletion below is a single manual step outside it. Every existing Julia depot entry holds native images. If they stayed, the
test process would still load them where the runner accepts them, but every
precompile process would reject them (*Facts*) and compile those dependencies
again, uncounted, while the native files stayed beside the new `haswell` ones
for good, past the 10 GB limit (*Facts*, cache use).

So every cache entry whose key starts with `julia-cache;`, on every ref
(`vmm`, `main` and pull-request refs), is deleted once, by cache ID, after
the owner approves the merge and immediately before it. The owner's approval
comment for this amendment names this deletion; the merge approval sets its
time. The deletion is done with the GitHub API by an identity with
`actions: write` on the repository; the owner's approval comment names who
does the deletions of steps 2, 4 and 5. The steps, each recorded on the PR:

1. Before the deletion, `gh run list` shows no CI or Documentation run in
   progress or queued, and the owner agrees to push nothing else to `vmm` and
   to merge nothing else until step 4 is complete.
2. The deletion, with the deleted cache IDs, refs and keys, and a listing
   right after it that shows no `julia-cache;` entry.
3. The merge.
4. After every job of the merge push (CI and Documentation) has finished,
   and `gh run list` shows no CI or Documentation run in progress or queued,
   a new listing. Every `julia-cache;` entry whose key's `run_id` is not that
   of a merge-push run (CI or Documentation) is deleted; PR entries are cheap
   to lose, because later PR runs restore the `vmm` base. If the merge push
   restored any entry (its cache step log), its saved entries are deleted as
   well and its workflows are re-run, so that they build from empty; step 4
   is then repeated for the re-run. After the deletions, `gh run list` shows
   that no CI or Documentation run started after the check, and a final
   listing shows only merge-push entries; otherwise step 4 is repeated.
5. Later, immediately before the push that brings R-005 to `main`, the
   `refs/heads/main` `julia-cache;` entries are deleted, because pushes to
   `main` before then save native entries there. This is recorded on #230.
   Completion of this work does not wait for step 5; until it is done, it is
   an open item on #230.

The merge push then builds every entry from empty, as in run 42950743815.
PR runs before the merge push has saved its entries start cold; later PR runs
restore the new `vmm` entries.

### R-008 — Tests

`scripts/test_ci_cache_workflow.py` checks that `CI.yml` and
`Documentation.yml` set `JULIA_CPU_TARGET: 'haswell'` in their workflow-level
`env`, and that `JULIA_CPU_TARGET` appears nowhere else in them: not in a job
or step `env:`, a `run:` script (an `export` or a write to `$GITHUB_ENV`) or a
`with:` input. Its tests that
alter the workflow text show that the check fails in each case: the variable
missing, another value, a job `env:`, a step `env:`, a `run:` export, a
`run:` write to `$GITHUB_ENV`, and a `with:` input. The PR records
the tests failing against the parent commit.

The same commit fixes the two non-blocking findings of the review of
`ba2364b`:

- `scripts/ci_report_package_images.jl` opens each file inside its `try`, so
  a file that cannot be opened counts as unreadable.
- The R-003 check finds a cache step whether its `uses:` value is plain,
  single-quoted or double-quoted, and with or without a trailing comment;
  its tests cover each form.

The first fix is shown by running the script, before and after, on a local
depot that holds a file it cannot read. The output goes in the PR.

### Numerical effect

Package image code is compiled for `haswell` on every runner: AVX2 and FMA,
no AVX-512. Before, it was compiled for whichever CPU built the image. Code
compiled at run time does not change. In all four runs, on `znver5`,
`icelake-server`, `znver3` and `znver5`, qualification gave the same verdicts and counts as before the fix:
`fast_qualification=PASS` with 6 producers and 6 replays, and
`public_qualification=PASS` with 28 and 28. No tolerance changes, so the work
is not reclassified under L-0012.

## Accepted limitations

- The recompilation continues until the fix merges.
- The step reads cache file headers with internal Julia functions
  (`Base.parse_cache_header`, `Base.check_clone_targets`); a Julia update may
  break it. Because it cannot fail the job, a break shows as a missing table,
  not a failed run.
- Amendment 1: a runner that lacks a `haswell` feature would reject the
  images and compile them again, as before the fix. `znver3`, `znver4`,
  `znver5` and `icelake-server` runners loaded `haswell` images in the
  measurement runs. For `sapphirerapids` and `graniterapids` runners this is
  inferred from their CPU names, whose Julia feature sets contain `haswell`'s;
  no run has shown it.
- Amendment 1: `HostCPUFeatures` reports no AVX, AVX2, BMI2 or FMA, and warns
  4 times when it is precompiled (R-006).
- Amendment 1: until R-007 step 5, a `vmm` or PR run that restores a
  `refs/heads/main` entry (the default branch, used when no closer entry
  matches) gets native images. Such a restore shows in the run's cache step
  log as a restored key whose `run_id` is a run on `main` (`gh run view
  <run_id>` shows its branch, and `gh cache list --ref refs/heads/main` lists
  the key); any entry that run saves is deleted, by the
  identity named for R-007, and recorded on #230.
- Amendment 1: PR runs between the R-007 deletion and the merge push's saves
  have no `vmm` base and build cold: about 26.6 min for Fast tests and 34 min
  for qualification (run 42950743815).

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
  and a passing PR CI run. For amendment 1, also:
  - the owner's approval comment that names the amendment's commit SHA and
    the R-007 deletion;
  - R-005 and the R-008 changes in one commit, with the R-008 tests failing
    against the parent commit, and the before-and-after script output, in the
    PR;
  - on the PR's CI run, in every cached job that runs on the PR (lifecycle,
    fast, qualification, Documentation build), the R-001 step's own output
    has the line `JULIA_CPU_TARGET: haswell` after its `Sys.CPU_NAME:` line,
    followed by its target table, and the tests and qualification pass. That run restores
    native entries, so its precompile counts are not evidence (R-007);
  - R-007 steps 1 and 2 are recorded on the PR before the merge.
- All Python tests pass.
- An independent review of the exact candidate SHA, reported separately from
  CI.

### CYAX-0230 G2 — Post-merge evidence (on #230, not a merge gate)

| Observation | Required record |
| --- | --- |
| First push to `vmm` after merge | Per cached job: CPU model, target table, precompile count, job time, entry size |
| Later PR runs, until two different CPU models have been seen or the number of runs set by owner decision D2 is reached | Per cached job: CPU model, target table and precompile count |
| Amendment 1: first push to `vmm` after the merge | R-007 steps 3 and 4 (listing, deleted IDs, any re-run). Per cached job: no entry restored; precompile count, job time and entry size |
| Amendment 1: the push that brings R-005 to `main` | R-007 step 5: the deleted `main` entries |
| Amendment 1: the first run that restores each entry of that push, and the later PR runs above whose restored entry was saved after the merge | Per cached job: only `haswell` images restored, none rejected by target; packages compiled again only for new releases (`Installed` lines) |

## Interfaces and compatibility

CI only. R-001 changes no package, API, schema, scientific or version
behaviour. CI results are already CPU-dependent today, because images are
built for `native` on whichever runner builds them. The R-004 amendment must
state its own numerical effect.

## Owner decisions

The owner chose D1 and D2 before this spec was committed; the G0 approval
comment confirms them. Amendment 1 adds no owner decision: its approval
covers the R-007 deletion.

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
recorded on #230, except the R-007 step 5 row, which stays an open item on
#230. If R-002 does not confirm the cause, the owner decides
under R-002 whether R-001 merges alone; the evidence is recorded on #230.
If the owner decides that nothing merges, the work is complete when that
decision and the evidence are recorded on #230.
