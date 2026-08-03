# CYAxiverse.jl Copilot Instructions

## Environment and commands

- Use Julia 1.12 as declared in `Project.toml`. Instantiate the package environment with:
  ```sh
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```
- Loading `CYAxiverse` for the Julia spectrum and test paths does not require `ENV["PYTHON"]`. Set it to the Python executable that PyCall should use only for optional CYTools/PyCall workflows. `src/init_python.jl` rebuilds PyCall when its configured interpreter differs from this value.
- Run the full test suite (the same `Pkg.test()` path used by CI) with:
  ```sh
  julia --project=. -e 'using Pkg; Pkg.test()'
  ```
- The only independently runnable regression test is the CYTools-wrapper layout repro:
  ```sh
  julia --project=. scripts/cytools_wrapper_repro.jl
  ```
  `test/runtests.jl` has no test-name filtering; add focused tests there when adding new behavior.
- Build documentation using the workflow's setup, from the repository root:
  ```sh
  julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
  julia --project=docs docs/make.jl
  ```
- No formatter or linter is configured. GitHub Actions runs package build/tests and documentation builds; its Julia 1.7 entries predate the current `Project.toml` Julia 1.12 compatibility requirement.

## Architecture

- `src/CYAxiverse.jl` is the package entry point. It composes internal submodules rather than re-exporting their APIs; call functionality through module namespaces such as `CYAxiverse.generate`, `CYAxiverse.read`, and `CYAxiverse.filestructure`. Only `greet_CYAxiverse` is exported.
- `structs.jl` defines the shared domain types. Use `GeometryIndex(h11, polytope, frst)` to identify a geometry and the typed `TopologicalData`, `GeometricData`, `AxionPotential`, and minimizer result structs to move data between layers.
- The normal data flow is: `cytools_wrapper` generates CYTools geometry and writes HDF5; `filestructure` maps a `GeometryIndex` to the on-disk database layout; `read` loads topology, geometric data, potentials, spectra, and vacuum results; `generate` computes potentials, bases, spectra, and vacuum quantities; `minimizer` and `jlm_minimizer` solve for minima; `plotting` produces CairoMakie outputs.
- Execution chooses its backend during package load. If `ENV["PYTHON"]` contains `"cytools"`, `add_functions/cytools_wrapper.jl` provides CYTools access through PyCall. Otherwise, the entry point includes the `jlm_python` solver bridge and test-local JLM minimizer modules. Do not assume both paths are loaded in one session.
- `ENV["newARGS"]` selects a named database root through `filestructure.ol_DB`; without it, command-line `ARGS` is used and unknown values fall back to the working directory. The Docker image sets `newARGS=docker`, mounts its database at `/scratch/database/`, and starts Pluto.

## Data and code conventions

- Geometry data is persisted beneath the selected database directory as `h11_XXX/np_YYYYYYY/cy_ZZZZZZZ/cyax.h5`; preserve zero padding and use `filestructure` path helpers rather than constructing alternate paths.
- HDF5 writes use existing dataset names and compression (`deflate=9`). Keep the established `cytools/geometric`, `cytools/potential`, `vacua`, and `hilbert` group layout compatible with readers in `src/read.jl`.
- Instanton-scale matrices conventionally store each value as `[sign_or_mantissa, base-10 exponent]`. Convert with `L[:, 1] .* 10 .^ L[:, 2]` only where full values are required; preserve this representation in persisted and inter-module data.
- Functions commonly provide both scalar geometry arguments `(h11, tri, cy=1)` and a `GeometryIndex` overload. Maintain both forms when extending geometry-facing APIs.
- Numerical code mixes `Float64`, exact `Rational`, and `ArbFloat` values. Avoid narrowing the existing declared types, particularly in basis and minimization code that deliberately switches to `Rational{BigInt}` for large denominators.
