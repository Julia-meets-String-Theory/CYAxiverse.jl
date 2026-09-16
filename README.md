# CYAxiverse.jl

CYAxiverse.jl is a Julia package for calculations in the axion/axion-like
particle sector of string-motivated Calabi–Yau compactifications. It consumes
geometry and instanton data, then provides axion spectra, vacua searches,
inflation diagnostics, benchmark models, and local axion–photon observables.

The package is under active development on the
[`vmm` branch](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/tree/vmm).
This README is a provisional development front door. `main` is still the
temporary default branch while [Issue #125](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/125)
governs branch-history and release reconciliation. The current branch
arrangement is transitional; it does not establish a permanent canonical
branch or stable release.

## Installation and setup

CYAxiverse.jl targets Julia 1.12 and is not registered in the Julia General
registry. Install the development branch in a Julia project:

```julia
using Pkg
Pkg.add(url = "https://github.com/Julia-meets-String-Theory/CYAxiverse.jl.git",
        rev = "vmm")
```

The core package does not require Python, CYTools, Docker, a geometry database,
or a graphical backend. Existing-data workflows require a separately supplied
geometry database. Optional CYTools/PyCall workflows include geometry generation
or inspection and the existing Python-backed `jlm_minimizer` path; plotting with
CairoMakie and ColorSchemes is a separate optional workflow.

Use the live [installation guide](https://julia-meets-string-theory.github.io/CYAxiverse.jl/dev/installation/)
for the complete setup, data-directory, optional-integration, and current
platform-support contract. Continuous integration currently tests Julia 1.12
on Ubuntu x64; other platforms may require local validation.

## Quick start: a core-only deterministic benchmark

After setting up the core package, run this self-contained N=5 reduced-model
check. It uses no geometry database, CYTools/PyCall, plotting backend, or
external data file.

```julia
using CYAxiverse

const Bench = CYAxiverse.paper_benchmarks
k = Bench.n5_critical_scale() - 1e-4
result = Bench.n5_reduced_critical_points(k)
@assert result.minima == 2
println("N=5 reduced-model minima: ", result.minima)
```

Expected output is `N=5 reduced-model minima: 2`. The test exercises the
source-faithful reduced potential on the subcritical side of its cusp.

## What the package provides

- **Geometry and potential readers.** Read existing `cyax.h5` files with
  `CYAxiverse.read` and use `CYAxiverse.read.oriented_potential` when a
  canonical `(Q, L, K)` tuple is required.
- **Axion spectra.** `CYAxiverse.generate` provides fast PQ, high-precision,
  hybrid, and mass-window paths. Results include log-scaled masses and decay
  constants. Optional quartic calculations include cancellation diagnostics.
- **Vacua and inflation diagnostics.** The vacua pipeline combines
  determinant/leading-branch paths with bounded numerical searches.
  `CYAxiverse.inflation_points` provides stationary-point correction,
  generalized-Hessian diagnostics, precision comparison, and bounded
  gradient flow.
- **Local axion–photon calculations.**
  `CYAxiverse.axion_photon.run_local_scan` and
  `run_batch_axion_photon` read complete local geometry files and compute the
  leading charge hierarchy, mixing, photon couplings, and leading width
  estimates.
- **Deterministic benchmark models.**
  `CYAxiverse.paper_benchmarks` contains reduced N=5 and N=8 models, fixed
  poly-102 inflation inputs, and fuzzy-axion model-stage helpers used by the
  repository's reproduction tests.
- **Optional extensions.** CYTools/PyCall supports optional Python-bridge
  workflows, including geometry generation or inspection and the existing
  Python-backed `jlm_minimizer` path. CairoMakie and ColorSchemes provide
  plotting methods through an optional extension. Core package loading keeps
  these integrations optional.

These are implementation surfaces, not claims that every research workflow is
complete or production-qualified. Read the method and validation notes before
interpreting a result.

## From geometry data to physics

CYAxiverse separates geometry generation from Julia-side analysis:

```text
geometry producer (CYTools is optional)
                 │
                 ▼
        cyax.h5: geometry, Q, L, and metadata
                 │
                 ▼
       CYAxiverse.read / oriented_potential
                 │
       ┌─────────┼──────────┬──────────────┐
       ▼         ▼          ▼              ▼
    spectra    vacua     inflation     axion–photon
```

The package-facing potential convention is:

| Object | Shape | Meaning |
| --- | --- | --- |
| `K` | `h11 × h11` | Axion kinetic metric. Geometry files normally store `Kinv`. |
| `Q` | `h11 × N` | Integer instanton charges, one charge vector per column. |
| `L` | `2 × N` | Signed log-scale data: sign/mantissa in row 1 and `log10` scale in row 2. |

The log-scale representation preserves information when instanton scales span
many orders of magnitude. `oriented_potential` validates dimensions and finite
values, accepts the stored matrix layouts used by the repository, and returns
the canonical package orientation.

For existing-data workflows, follow the data-directory and input-file
instructions in the
[User guide](https://julia-meets-string-theory.github.io/CYAxiverse.jl/dev/userguide/).
Python-bridge geometry generation or inspection and the existing Python-backed
`jlm_minimizer` path are separate optional workflows; importing the core Julia
package does not require a live Python object.

## Main workflows

### Axion spectra

Use the [Pipelines guide](https://julia-meets-string-theory.github.io/CYAxiverse.jl/dev/pipelines/)
for the HDF5 contract and batch drivers. The main Julia entry points are:

- `CYAxiverse.generate.pq_spectrum` for a fast leading-Hessian spectrum;
- `CYAxiverse.generate.hp_spectrum` for arbitrary-precision diagonalization;
- `CYAxiverse.generate.pq_physical_spectrum` and
  `pq_hybrid_physical_spectrum` for threshold-aware physical-mode paths; and
- `CYAxiverse.generate.pq_window_spectrum` for a bounded mass interval.

The package stores very wide mass and coupling ranges in `log10` form. The
high-precision paths retain explicit precision, residual, and fallback
diagnostics. A spectrum result is not, by itself, evidence for moduli
stabilization or a physical population-level conclusion.

### Vacua and inflation

Use `scripts/vacua_pipeline.jl` or
`scripts/batch_vacua_pipeline.jl` for bounded vacua work. The automatic method
selection can use determinant counts, selected leading branches, and finite
multistart searches. A finite search budget produces a search result, not a
completeness proof.

Use `CYAxiverse.inflation_points` for generic fixed-geometry diagnostics and
bounded gradient flows. A stationary point with a negative Hessian direction
is a saddle diagnostic. It is not automatically a catastrophe, a stabilized
vacuum, or a validated inflationary population. A flow that reaches its
configured horizon remains explicitly bounded by that horizon.

### Local axion–photon scan

The
[local axion–photon guide](https://julia-meets-string-theory.github.io/CYAxiverse.jl/dev/axion_photon/)
documents the complete local-file fields, charge orientation, leading
hierarchy, mixing, photon-coupling, and width conventions. The implementation
uses existing geometry files and writes compact CSV summaries; it does not
claim to reproduce the full paper ensemble. Visible-sector metadata and the
QED-threshold policy are explicit inputs, not inferred Standard Model
assignments.

### Optional plotting

Load `CairoMakie` and `ColorSchemes` explicitly when a plot is needed. The
plotting namespace is documented in the
[API reference](https://julia-meets-string-theory.github.io/CYAxiverse.jl/dev/api/).
Plotting is an optional extension and is not required for core numerical
analysis.

## Methodology and validation

The repository keeps scientific evidence separate from user-facing summaries.
The following notes define the current boundaries:

- [Inflation reproduction results](validation/inflation_reproduction_results.md)
  record the frozen N=5/N=8 benchmark inputs, source identity, phase and
  normalization conventions, and the distinction between reduced normal-form
  values and full nonlinear flow values.
- [P0 numerical-equivalence evidence](validation/p0_numerical_equivalence/evidence_index.md)
  records route-specific numerical behavior, deterministic fixtures, precision
  boundaries, and replay provenance. It preserves observed Float64 versus
  high-precision differences instead of silently selecting a physical oracle.
- [Orientifold audit summary](validation/fuzzy_axions_2412_12012_orientifold_audit_summary_20260820.md)
  records the inherited-orientifold and trilayer evidence boundary. Exact
  aggregate benchmark matches and conservative code-certified lower bounds
  are reported separately.
- The [API reference](https://julia-meets-string-theory.github.io/CYAxiverse.jl/dev/api/)
  documents public modules and function contracts from their source docstrings.

Use the source revision, input identity, method, units, basis, tolerances, and
result status when you report a scientific calculation. Do not promote a
bounded, filtered, diagnostic, or provisional result to a general population
or physical claim without the required evidence and scientific-owner review.

## Research references and benchmark classification

The repository uses related papers in different ways. The classification below
prevents a methodological reference from being presented as direct software
use or as a complete reproduction.

| Reference | Relationship to this package |
| --- | --- |
| [Catastrophic Inflation in the Axiverse](https://arxiv.org/abs/2608.14780) | Source and benchmark lineage for the fixed N=5/N=8 inflation inputs. The repository's validation is bounded and model-specific; it is not a generic physical validation. |
| [Glimmers from the Axiverse](https://arxiv.org/abs/2309.13145) | Methodological source for the local axion–photon hierarchy, coupling, threshold, and width relations. The package applies these relations to its local file convention; it does not claim the paper's full ensemble reproduction. |
| [Orientifolding Kreuzer–Skarke](https://arxiv.org/abs/2305.06363) | Source for orientifold geometry and fixed-locus methods used by the orientifold bridge. Current acceptance and smoothness limits remain explicit in the validation records. |
| [Fuzzy Axions and Associated Relics](https://arxiv.org/abs/2412.12012) | Source reference for the orientifold population and trilayer comparison. The repository reports exact source-verified aggregate trilayer matches and keeps inherited counts as conservative lower bounds. |

The [CYTools project](https://cytools.liammcallistergroup.com/) is an input
geometry producer for the applicable workflows. It is optional for core Julia
package loading.

## Citation

If you use CYAxiverse.jl in research, cite this repository at the exact commit
used and cite the relevant scientific paper or benchmark source. The repository
does not declare a software DOI or an unestablished preferred citation. Use the
validation notes above to identify the source and scope of a benchmark. Do not
describe a bounded local scan or benchmark helper as a complete reproduction of
a paper population unless its evidence says so.

## Contributing and support

Open a [GitHub issue](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues)
for a defect, question, or contribution. Include a small reproducer, the exact
repository revision, Julia version, input identity/schema, and the observed
status or error. Keep credentials and machine-local paths out of public issue
reports. Read `AGENTS.md` before making repository changes.

## License

CYAxiverse.jl is released under the [MIT License](LICENSE).

## Acknowledgements

The project was initiated after
[Superradiance in String Theory](https://iopscience.iop.org/article/10.1088/1475-7516/2021/07/033).
We thank the CYTools developers and project collaborators for their scientific
and software contributions. The project author is
[Viraf M. Mehta](https://inspirehep.net/authors/1228975).
