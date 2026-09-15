# P0-A/B environment and package-load baseline

Status: **DONE (environment/load baseline; evidence-only)**

This record covers the P0-A exact runtime/dependency identity and P0-B load
split for the historical reference revision. It does not change production
code, package metadata, or the numerical algorithms.

## Scope and source identity

| Field | Observed value |
|---|---|
| Scientific reference SHA | `7a40285bb5c313f7e8746b90644d5f45bb67be44` |
| Evidence-run SHA | `99d702c8b28e674d34e29566f8ecc483d76c7f7f` |
| Evidence-run ref | `research/p0-numerical-equivalence-20260915` |
| Reference source check | `true` for `Project.toml` and `src/` (`git diff --quiet <reference> HEAD -- Project.toml src`) |
| Root Project.toml SHA-256 | `38275eaf04c8f9ae28541542326f6c791c3a6a40d2a9ca3916088ca0652c368c` |
| Root Manifest.toml | **absent** at the reference revision and in the evidence checkout |

The evidence-run SHA contains only the later P0 specification/planning
documents relative to the scientific reference; the package project and
`src/` tree are unchanged. The reference SHA is therefore the numerical source
identity. The later evidence SHA is retained to identify the checkout in which
the measurements and this record were produced.

## Resolved dependency environment

The exact resolved environment used for import/precompile was the Julia 1.12
audit environment whose manifest has the following identity:

| Field | Observed value |
|---|---|
| Source Manifest basename | `Manifest.toml` (external audit environment used for the recorded timing run) |
| Manifest SHA-256 | `f51abc728b461d42c1bd22fd0be638193fe5f1bd627a096d717ec2ec09d44f15` |
| Manifest format | `2.0` |
| Manifest Julia version | `1.12.6` |
| Manifest project hash | `7a59ac8c18a1d9be6e676b7f53542136e568d44e` |
| Audit Project.toml SHA-256 | `ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9` |
| Manifest package records | `242` total; `241` versioned; `1` Julia stdlib record |
| Path package records | `CYAxiverse` only; temporary copy redirected its path to this checkout |

A privacy-safe normalized copy of that exact resolved environment is retained
in `validation/p0_numerical_equivalence/environment/`.  The only Manifest
change replaces the machine-local CYAxiverse path with repository-relative
`../../..`; package versions and git-tree identities are unchanged.

| Retained artifact | SHA-256 |
|---|---|
| `environment/Project.toml` | `ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9` |
| `environment/Manifest.toml` | `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf` |

The normalized environment was independently loaded with a writable depot
prefix using:

```text
JULIA_DEPOT_PATH=<writable-depot>:<existing-local-depot> \
  JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no --history-file=no \
  --project=validation/p0_numerical_equivalence/environment \
  -e 'using CYAxiverse; using Pkg; println(Base.pkgversion(CYAxiverse)); Pkg.status(; mode=Pkg.PKGMODE_PROJECT)'
```

Observed exit `0`, CYAxiverse version `0.2.0`, with Aqua `0.8.16` and JET
`0.12.1` present in the audit environment.  This replay confirms the normalized
path is functional; it is not a second load-timing sample.

The manifest is identified by its full SHA-256 because the repository has no
root lockfile. The source manifest's only path entry pointed to the original
temporary checkout of the pinned revision. For this run, a temporary copy
changed only that path entry to the current checkout; all registry package
versions and git-tree hashes remained unchanged. No dependency resolution,
registry update, or package metadata write was performed in the repository.

The root package's direct resolved versions in this manifest were:

| Package | Version |
|---|---:|
| AbstractTrees | `0.4.5` |
| ArbNumerics | `1.6.3` |
| Dates | `1.11.0` (stdlib) |
| Distributed | `1.11.0` (stdlib) |
| Distributions | `0.25.131` |
| GenericLinearAlgebra | `0.4.1` |
| HDF5 | `0.17.3` |
| IntervalArithmetic | `0.22.36` |
| LineSearches | `7.5.1` |
| LinearAlgebra | `1.12.0` (stdlib) |
| LinearSolve | `5.17.4` |
| LoopVectorization | `0.12.174` |
| NLsolve | `4.5.1` |
| Nemo | `0.56.1` |
| NormalForms | `0.1.10` |
| Optim | `1.13.3` |
| OrdinaryDiffEq | `7.8.1` |
| Random | `1.11.0` (stdlib) |
| SHA | `0.7.0` |
| SparseArrays | `1.12.0` (stdlib) |
| StaticArrays | `1.9.20` |
| TimerOutputs | `1.2.1` |
| Tullio | `0.3.9` |

The temporary audit harness project also declared `Aqua` `0.8.16` and `JET`
`0.12.1` so that Julia could consume the existing resolved manifest. The load
measurement executed only `using CYAxiverse`; it did not import PyCall,
CYTools, Aqua, or JET. The full manifest identity above is the authoritative
identity for transitive versions and git-tree hashes.

## Runtime identity

Command run from the repository root:

```text
julia --startup-file=no --history-file=no --compiled-modules=no --project=.
  validation/p0_numerical_equivalence/scripts/environment_and_load.jl metadata
```

Observed identity (paths and private machine locators intentionally omitted):

```text
julia_version=1.12.6
julia_build_commit=15346901f0039751c5488744f1f62de7d87510a8
julia_build_branch=v1.12.6
julia_build_number=0
julia_build_date=2026-04-09 19:20 UTC
julia_build_tagged=true
julia_sysimage=sys.dylib
julia_build=julia
word_size=64
architecture=arm64-apple-darwin24.0.0
os=Darwin aarch64
cpu_name=apple-m4
cpu_threads=8
julia_threads=1
blas_vendor=lbt
blas_config=LBTConfig([ILP64] libopenblas64_.dylib)
blas_threads=8
libblastrampoline=libblastrampoline.5.dylib
```

Independent hardware query:

```text
system_profiler SPHardwareDataType
Model Name: MacBook Pro
Chip: Apple M4 Pro
Total Number of Cores: 12 (8 Performance and 4 Efficiency)
Memory: 24 GB
```

Effective environment variables for the measurements:

| Variable | Value |
|---|---|
| `JULIA_NUM_THREADS` | unset; effective Julia thread count `1` |
| `JULIA_THREAD_SLEEP_THRESHOLD` | unset |
| `JULIA_CPU_TARGET` | unset |
| `JULIA_PROJECT` | unset; project passed explicitly on the command line |
| `JULIA_DEPOT_PATH` | temporary writable depot first, host depot read-only fallback; exact locators omitted |
| `JULIA_PKG_PRECOMPILE_AUTO` | `0` |
| `JULIA_PROGRESS` | `0` |
| `OPENBLAS_NUM_THREADS` | unset |
| `OMP_NUM_THREADS` | unset |
| `MKL_NUM_THREADS` | unset |
| `VECLIB_MAXIMUM_THREADS` | unset |
| `BLAS_NUM_THREADS` | unset |
| `LANG` / `LC_ALL` | `C.UTF-8` / `C.UTF-8` |
| `TZ` | unset |

The default BLAS runtime reported eight threads even though Julia itself ran
with one thread. This is an observed host baseline, not a recommendation for
future multi-process numerical runs; later benchmark work must either preserve
this setting or record an explicit change.

RNG identity recorded by the harness:

```text
rng_algorithm=Random.MersenneTwister
rng_seed=0x5eed
rng_probe=11400839645683421532
```

No numerical fixture was evaluated by this worker, so no fixture hash is
claimed here. The package source identity is the reference SHA above, and the
load harness itself is `validation/p0_numerical_equivalence/scripts/environment_and_load.jl`.

## P0-B1a warmed/precompiled package load

Warm means: dependencies were already installed and the project-local compiled
cache had been produced by the cold-build run below. Each sample started a new
Julia process and executed one `using CYAxiverse`; no precompile output was
observed in these warm samples.

Command (temporary paths shown as privacy-safe placeholders):

```text
for sample in 1 2 3 4 5; do
  env JULIA_DEPOT_PATH=<fresh-project-cache>:<host-cache-read-only> \
      JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PROGRESS=0 \
      P0_BASE_MANIFEST=<resolved-audit-manifest> \
      /usr/bin/time -p julia --startup-file=no --history-file=no \
        --project=<temporary-audit-project> \
        validation/p0_numerical_equivalence/scripts/environment_and_load.jl load
done
```

The harness runs `GC.gc()` immediately before timing and measures
`@allocated(@eval Main using CYAxiverse)`. The process-level `real/user/sys`
values come from `/usr/bin/time -p`; the harness elapsed value starts after
the helper process has initialized its own metadata imports.

| Sample | Import elapsed (s) | Import allocated bytes | Process real (s) | Exit |
|---:|---:|---:|---:|---:|
| 1 | 5.205572250 | 516067920 | 6.31 | 0 |
| 2 | 4.463404500 | 516066128 | 5.52 | 0 |
| 3 | 3.829197833 | 516068592 | 4.84 | 0 |
| 4 | 3.745119500 | 516070016 | 4.74 | 0 |
| 5 | 3.745686292 | 516066960 | 4.72 | 0 |
| **Median** | **3.829197833** | **516067920** | **4.84** | — |
| Mean | 4.197796075 | 516067923.2 | 5.226 | — |
| Min / max | 3.745119500 / 5.205572250 | 516066128 / 516070016 | 4.72 / 6.31 | — |

The import completed successfully in all five samples and reported package
version `0.2.0`. Import allocation is process/JIT-loader allocation for the
single package import, not a steady-state numerical-kernel allocation claim.

## P0-B1b cold environment/build baseline

Cold means: a fresh writable first depot and temporary audit project were used;
the host depot remained a read-only fallback for already-installed package
sources/artifacts. Therefore this is a **fresh project-local compiled-cache
and build baseline**, not a network/download-from-zero baseline. The command
used the same resolved manifest identity and `Pkg.instantiate()` followed by
`Pkg.precompile(; strict=true)` before the timed import.

Command:

```text
env JULIA_DEPOT_PATH=<fresh-project-cache>:<host-cache-read-only> \
    JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PROGRESS=0 \
    P0_BASE_MANIFEST=<resolved-audit-manifest> \
    /usr/bin/time -p julia --startup-file=no --history-file=no \
      --project=<temporary-audit-project> \
      validation/p0_numerical_equivalence/scripts/environment_and_load.jl cold
```

Observed outcome, exit `0`:

| Stage | Elapsed |
|---|---:|
| `Pkg.instantiate()` | `0.958732625 s` |
| `Pkg.precompile(; strict=true)` | `189.847255291 s` |
| `using CYAxiverse` after precompile | `4.393729166 s`; `511190320` bytes |
| Harness total after Julia/Pkg startup | `195.327656916 s` |
| `/usr/bin/time -p` process `real` | `232.39 s` |
| `/usr/bin/time -p` process `user` | `806.46 s` |
| `/usr/bin/time -p` process `sys` | `37.97 s` |

The process also precompiled Julia/Pkg support before the harness timer:
three dependencies in about two seconds and 23 Pkg/stdlib dependencies in
about 33 seconds. The project precompile then reported **229 dependencies
successfully precompiled in 189 seconds; 92 already precompiled** (the latter
came from the read-only fallback depot). This split explains why the full
process wall time is larger than the harness's post-startup total.

## Checks and limitations

Executed checks:

```text
git rev-parse HEAD
git diff --quiet 7a40285bb5c313f7e8746b90644d5f45bb67be44 HEAD -- Project.toml src
julia --version
system_profiler SPHardwareDataType
julia ... environment_and_load.jl metadata
five warm `... environment_and_load.jl load` processes
one cold `... environment_and_load.jl cold` process
git diff --check
```

All scoped checks completed with exit `0` after the harness corrections. The
initial exploratory runs are retained as ordinary diagnostic evidence: direct
package loading from the checkout failed because the root lockfile is absent,
the host depot is not writable in this sandbox, and the first temporary
manifest copy used the package root as an active project rather than as a
consumer environment. None of these failures modified repository files.

The repository root does not contain a Manifest.  P0 therefore retains the
privacy-safe normalized audit Project/Manifest beside this record, together
with both original and normalized hashes.  Replay still requires registry
package sources/artifacts or network access in the usual Julia manner; no root
dependency metadata was added or changed.
