# Handoff: finish the package audit

## Mission

Make `bin/audit.jl` a reliable package-health gate for the `vmm` architecture:

1. Resolve the actionable JET findings without masking them through broad
   configuration changes.
2. Resolve the Aqua findings, including the stale dependency report.
3. Preserve the optional PyCall/CYTools boundary.
4. Keep the numerical sanity checks and make the complete audit exit
   successfully.

The intended recipient is an experienced Julia/CYAxiverse developer. This is
an implementation handoff, not a report-only triage.

## Starting state

The worktree is based directly on `vmm` commit `d831b52` (the two revisions
are currently identical):

```text
HEAD = d831b5288b91cd2120db1362b463354d42e84545
vmm  = d831b5288b91cd2120db1362b463354d42e84545
```

The current uncommitted changes are:

- [`bin/audit.jl`](bin/audit.jl), the new audit entry point.
- [`src/generate.jl`](src/generate.jl), including the coupled Hessian fix
  used by the physical checks.

Do not reset these changes or rebase away from `vmm`. The `vmm` branch already
provides the intended optional-extension design:

- Core `CYAxiverse` loading does not require `PYTHON`.
- CYTools/PyCall functionality is in the optional
  `CYAxiversePyCallExt` extension.
- The audit must run with no Python environment configured.

## Reproduce the baseline

Run from the repository root with Julia 1.12:

```sh
julia --version
julia --startup-file=no --project=. -e 'using CYAxiverse; println(CYAxiverse.greet_CYAxiverse())'
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=. bin/audit.jl
```

The package test suite passes on the starting state. The standalone audit
currently fails in the JET phase with 91 reports, so it does not reach the
later Aqua and physical phases because the script is fail-fast.

The physical checks themselves have been exercised separately and pass for
both one-field and multi-field charge layouts. They verify:

```text
V(x) = sum(Λ .* (1 - cos(Q' * x)))
∇V   = Q * (Λ .* sin(Q' * x))
H    = Q * Diagonal(Λ .* cos(Q' * x)) * Q'
```

## Audit harness contract

[`bin/audit.jl`](bin/audit.jl) creates a temporary Julia environment, develops
the local package into it, and installs `Aqua` 0.8 and `JET` 0.12 there. Do
not add these tools to the package's runtime dependencies merely to make the
script work.

The script must continue to:

- activate the repository by path rather than relying on the caller's
  working directory;
- run without `PYTHON`;
- analyze the local `CYAxiverse` module with JET;
- run Aqua package checks;
- exercise the scalar and multi-field potential derivatives;
- fail with a nonzero exit code when any audit check fails.

If the agent changes fail-fast behavior to aggregate all phases, preserve a
nonzero final status and print each failure clearly. Never convert a failed
check into a success-shaped fallback.

## Findings to prioritize

### P0: verify the audit boundary

Before fixing individual reports, confirm that the audit is analyzing the
local checkout and not a cached package. Confirm that:

```julia
Base.pkgdir(CYAxiverse)
```

points into this worktree, and that `Base.get_extension(CYAxiverse,
:CYAxiversePyCallExt)` is `nothing` when PyCall is not loaded.

Do not reintroduce `src/init_python.jl` into core package loading and do not
make the audit depend on CYTools.

### P1: fix real JET reports by subsystem

The 91 reports are distributed across several parts of the package. Triage
each report by root cause rather than treating every report as an independent
annotation task.

#### Type and constructor contracts

Start with:

- `structs.jl` around `IndexedAxionSpectrum`.
- `read.jl` methods such as `L_arb`, `qshape`, `physical_spectrum`, and
  `pipeline_vacua`.
- `generate.jl` methods that construct typed domain structs.

Observed examples include:

- an `IndexedAxionSpectrum` constructor call inferred with `Integer` fields
  rather than the declared `Int` fields;
- HDF5 reads inferred as unions of `Dataset`, `Datatype`, and `Group`;
- concrete constructors receiving `Any` values.

Use explicit, correct local types and narrow guards where the HDF5 API
requires them. Do not add `as any`-style casts or broad `Any` annotations that
only silence JET.

#### Numerical and minimizer code

Reports occur in:

- `generate.jl` (`pseudo_K`, `hessian_norm`, persistence, vacua, projector,
  and theta-minimum paths);
- `minimizer.jl`, including Tullio-generated code;
- `jlm_reduced.jl`;
- `add_functions/profiling.jl`.

First establish the intended matrix orientation and numeric types from the
existing tests and structs. Preserve the repository's use of `Float64`,
`Rational`, `BigInt`, and `ArbFloat`; do not narrow types just for JET.

Several reports point into Tullio-generated macro code. Do not modify the
installed Tullio package. Trace the generated report back to the calling
function and make the input/output types or macro invocation precise there.

#### Undefined names and conditional control flow

Observed examples include:

- `MatrixSpace` reported as undefined in `profiling.jl`;
- `optimize` reported as undefined in `profiling.jl`;
- potentially undefined `ax1` through `ax4` in `plotting.jl`.

Determine whether each is a missing import, a stale code path, or a variable
that is conditionally initialized. Fix the actual code path or remove dead
code; do not suppress the report globally.

#### Paper benchmarks

`paper_benchmarks.jl` includes an inference issue around
`Matrix{Float64}(Vector{Any})`. Preserve the benchmark API and numerical
results while making the benchmark data construction concretely typed.

### P1: resolve Aqua stale dependencies

Aqua currently reports these direct dependencies as stale:

- `Pluto`
- `PlutoUI`

The package source does not use these dependencies at runtime; they are
primarily associated with notebooks and scripts. Decide whether each should
be:

1. removed from the package's `[deps]` and retained only where the notebook
   environment needs it; or
2. kept and used by a real package source path.

Prefer removing stale runtime dependencies rather than disabling the Aqua
check. If package metadata changes, update the lockfile through the normal
Julia package workflow and rerun the full tests.

After the stale-dependency fix, rerun Aqua completely. The current audit
stops at JET, so additional Aqua failures may appear only after JET is clean.

## Recommended implementation sequence

1. Reproduce the baseline and save the complete JET output outside the
   repository.
2. Group reports by source file and root cause.
3. Fix constructor/type-contract issues in `structs.jl` and `read.jl`.
4. Fix undefined names and conditional variables in profiling, plotting, and
   benchmark code.
5. Fix numerical/minimizer inference issues without changing matrix
   orientation or persisted HDF5 schemas.
6. Resolve stale `Pluto`/`PlutoUI` dependencies.
7. Add focused regression tests to `test/runtests.jl` for every behavior
   changed, especially:
   - HDF5 read paths;
   - scalar and multi-field derivatives;
   - minimizer and benchmark results;
   - plotting/profiling paths if they are retained.
8. Rerun the package tests and the complete audit.

Use one targeted test invocation per coherent change, then run the full
`Pkg.test()` suite before declaring success.

## Guardrails

- Do not add a Python requirement to core package loading.
- Do not enable the CYTools extension as part of the audit.
- Do not suppress JET reports by broadening everything to `Any`.
- Do not disable Aqua checks merely to obtain a green result.
- Do not change HDF5 group names, dataset names, or matrix orientation
  without a compatibility migration.
- Do not alter unrelated `vmm` work.
- Keep generated reports, temporary audit environments, and profiling output
  out of the repository.

## Definition of done

The handoff is complete when all of the following are true:

- `HEAD` remains based on `vmm` and the optional PyCall extension still works
  when explicitly enabled.
- `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'` passes.
- `julia --startup-file=no --project=. bin/audit.jl` exits with status 0.
- JET reports zero findings for the configured `CYAxiverse` target.
- Aqua passes without disabling stale-dependency checks.
- The scalar, multi-field, symmetry, finiteness, and positive-definiteness
  checks pass.
- Any changed numerical or persistence behavior has focused regression
  coverage.
- The final handoff names remaining known warnings, if any, and explains why
  they are non-actionable.
