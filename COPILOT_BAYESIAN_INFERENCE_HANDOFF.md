# Copilot implementation brief: Bayesian inference on the Kähler cone

## Mission

Implement, on branch `bayesian-inference` (based on `vmm`), the reusable Bayesian-inference functionality described in arXiv:2512.00144v1, *Bayesian inference on Calabi--Yau moduli spaces and the axiverse*, and demonstrated by [AndreasSchachner/kahler_cone_sampler](https://github.com/AndreasSchachner/kahler_cone_sampler), inside `CYAxiverse.jl`.

The deliverable is a tested Julia API for geometry-independent sampling/inference plus a small, isolated Python/CYTools adapter. The package must remain usable without Python/CYTools. Do not port CYTools itself to Julia. Do not put CYTools imports, Python objects, or CYTools-dependent geometry discovery in the Julia core.

## Repository and working rules

- Repository: `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl`.
- Starting point: `vmm`; target branch: `bayesian-inference`.
- Preserve unrelated work already present on the branch.
- Follow the package’s existing module layout, naming, formatting, optional `PyCall` extension, HDF5 conventions, and Julia 1.12 compatibility.
- Before changing code, read `src/structs.jl`, `src/read.jl`, `src/generate.jl`, `src/paper_benchmarks.jl`, `ext/CYAxiversePyCallExt.jl`, `add_functions/cytools_wrapper.jl`, and `test/runtests.jl`.
- Never make `PyCall` a hard dependency. `using CYAxiverse` must work in a clean Julia environment with no Python installation.
- Keep large geometry/sample databases out of git unless explicitly requested; tests must use small deterministic fixtures.

## Scientific scope

The implementation should support the following pipeline:

1. A CYTools/Python preprocessing step obtains numerical geometry data: triple-intersection tensor, Kähler-cone generators, prime-toric-divisor charge/volume data, Hodge data, and any orientifold-even projection.
2. Julia constructs the classical geometry from that immutable data.
3. Julia samples Kähler parameters `t` in the cone using either random-walk MCMC or an interchangeable normalizing-flow backend.
4. Julia evaluates the Weil--Petersson prior and stretched-cone/volume constraints.
5. Julia computes divisor volumes, axion kinetic/decay-constant data, masses, and user-supplied observables.
6. Julia applies composable likelihoods and returns weighted samples/posteriors, evidence estimates where supported, diagnostics, and reproducible summaries.

The first implementation target is a robust MCMC path. Reuse the Julia ecosystem rather than reimplementing generic chain mechanics: use `AdvancedMH.jl` for random-walk/adaptive Metropolis where its target-density contract fits, `AbstractMCMC.jl` for the common sampler interface, and `MCMCChains.jl`/the relevant diagnostics packages for chain storage and diagnostics. Only implement CYAxiverse-specific target-density, cone-coordinate transformation, geometry validation, and result/post-processing layers. A normalizing-flow interface is required, but a complete flow trainer is a later milestone unless an existing Julia dependency is demonstrably appropriate. Do not silently substitute a Euclidean prior for the WP measure.

Do not make `Turing.jl` a prerequisite: this problem has a bespoke deterministic geometry state and does not need a probabilistic-programming DSL. Evaluate `BAT.jl` as an alternative only if its posterior-measure API, output types, and dependency footprint materially simplify the package; do not introduce both BAT and the Turing sampler stack for the same baseline path. Pin compatible versions after checking Julia 1.12 support and license/transitive-dependency impact.

## Mathematical contract

Use a documented convention and test it against hand calculations. Let `d = h11`, `kappa[i,j,k]` be the symmetric triple-intersection tensor, and `t` be Kähler parameters.

Define

```text
V(t)       = (1/6) κᵢⱼₖ tᵢ tⱼ tₖ
tau[i](t)  = (1/2) κᵢⱼₖ tⱼ tₖ
K[i,j](t)  = (1/4V) κᵢⱼₖ tₖ - (1/4V²) tau[i] tau[j]
WP(t)      = det(K)
logprior   = logdet(K) + optional user Jacobian/measure terms
```

If the codebase uses a different normalization, expose the conversion explicitly and test scale behavior rather than mixing conventions. Under `t -> λt`, `V -> λ³V`, `tau -> λ²tau`, and the WP measure is scale invariant in the expected project convention.

Represent a non-simplicial cone as `t = G * α`, with `α > 0`; use `α = exp(θ)` for unconstrained sampling. If `G` has redundant generators, retain the full map but account for the coordinate Jacobian consistently, or provide a reduced-coordinate option. The sampler must reject NaN, non-positive volume, non-positive-definite metric, and failed physical constraints with a finite sentinel log probability (not an exception inside a hot loop).

The stretched Kähler cone condition is a configurable set of lower bounds on all required holomorphic submanifold volumes. The core must not hard-code “all submanifolds” when only a divisor list is available: encode the available constraints explicitly and report which constraints were checked. An upper volume/KK bound is likewise a likelihood/constraint configuration, not a hidden global constant.

For type-IIB axion observables, implement the algebra that depends only on exported numerical geometry data in Julia. Geometry discovery, divisor classification, triangulation interrogation, and orientifold projection remain Python-only. Keep physical normalization constants in a named configuration object and use log-space arithmetic for masses, likelihoods, and weights.

## Proposed Julia API and data model

## Julia packages to evaluate before implementing infrastructure

Use the smallest interoperable stack that fits the package’s existing Julia 1.12 and optional-PyCall design. The following are recommended candidates, not a mandate to add every package:

| Need | Preferred candidate | Guidance |
|---|---|---|
| MCMC algorithm | `AdvancedMH.jl` | Prefer for random-walk/adaptive Metropolis; wrap through `AbstractMCMC.jl`. |
| Common sampler protocol | `AbstractMCMC.jl` | Gives standard transitions, warm-up, callbacks, and multi-chain integration. |
| Target-density protocol | `LogDensityProblems.jl`, `LogDensityProblemsAD.jl` | Use if compatible with the selected sampler; keeps geometry target code independent of the sampler. |
| Constrained/unconstrained transforms | `Bijectors.jl` or `TransformVariables.jl` | Evaluate for `α=exp(θ)`, bounded volumes, and transform log-Jacobians; do not duplicate a mature transform that already handles edge cases. |
| Automatic differentiation | `ForwardDiff.jl` first; `Enzyme.jl` only after benchmarking | Forward mode is a natural first choice for `d≤30`; preserve a no-AD fallback for invalid/non-differentiable constraints. |
| HMC/NUTS later | `AdvancedHMC.jl` | Add only after WP gradients are validated and the MCMC target is differentiable enough. |
| Diagnostics | `MCMCDiagnosticTools.jl` | Prefer for ESS, R-hat, MCSE, and autocorrelation diagnostics; use `ArviZ.jl` only if the richer interchange/plotting ecosystem is needed. |
| Chain container | `MCMCChains.jl` or `Chains.jl`-compatible output | Avoid locking core geometry types to a plotting-oriented representation; provide conversion methods. |
| Normalizing flows | `Bijectors.jl` plus `Lux.jl` | Evaluate for RealNVP/coupling-flow prototypes; `InvertibleNetworks.jl` is a specialist option if it proves maintained and compatible. |
| Optimization/flow training | existing `Optim.jl`, optionally `Optimization.jl` | Reuse current `Optim` dependency for simple objectives before adding another abstraction. |
| Columnar sample export | `Arrow.jl` and `Tables.jl` | Useful for large tabular sample summaries and Python interoperability; retain HDF5 for tensor-rich geometry artifacts. |
| Julia-native checkpointing | existing `HDF5.jl`; optionally `JLD2.jl` | HDF5 is the cross-language contract; JLD2 is convenient but should not define the Python-facing schema. |
| Parallel chains | `Distributed`, `Threads`, and `AbstractMCMC` support first | Add `FLoops.jl`/`Polyester.jl` only if benchmarks show a real need. |

Do not add `Turing.jl` merely to obtain MCMC, and do not add both `BAT.jl` and the Turing/AbstractMCMC stack without a concrete feature comparison. BAT is worth a short prototype if its posterior-measure abstraction and built-in algorithms reduce code, but it should remain an alternative architecture rather than a second mandatory framework. Check current Julia-version compatibility, direct dependency count, precompilation cost, and licenses before editing `Project.toml`.

Add focused files under `src/` (names may be adjusted to match local conventions):

- `src/bayesian_types.jl`: `CYGeometryData`, `KahlerGeometry`, `ConeParameterization`, `SamplerConfig`, `SampleBatch`, `Likelihood`, `PosteriorResult`, and diagnostics types. Prefer concrete, typed fields and `Vector{Float64}`/`Matrix{Float64}` at the numerical boundary; allow `T<:AbstractFloat` in pure kernels where practical.
- `src/kahler_geometry.jl`: constructors and kernels for `volume`, `divisor_volumes`, `wp_metric`, `log_wp_density`, cone maps, constraint checks, gradients where available, and stable Cholesky/log-determinant evaluation.
- `src/mcmc.jl`: a thin CYAxiverse adapter around `AbstractMCMC` plus `AdvancedMH` (or a documented BAT alternative), exposing unconstrained-coordinate Metropolis-Hastings with configurable proposal covariance, burn-in, thinning, multiple chains, adaptation, seeded RNG, progress-independent operation, and resumable output. Add CYAxiverse-specific acceptance/rejection accounting and transform-aware state handling; do not copy a generic Metropolis loop into the package unless the selected dependency cannot represent the required target. Use `MCMCChains` or an equivalent established result type where practical, and add only the minimum direct dependencies.
- `src/likelihoods.jl`: composable log-likelihood functions, `logprior`, `logposterior`, Gaussian/log-normal helpers, hard/soft divisor-volume constraints, QCD-like `(m_a, tau)` likelihoods, and a generic callable interface for downstream cosmology likelihoods. A bad point returns `-Inf`/the project sentinel consistently.
- `src/axion_inference.jl`: pure Julia post-processing from `SampleBatch` plus `CYGeometryData`; compute axion-facing quantities already supported by existing CYAxiverse code and expose hooks for additional observables. Reuse existing spectrum/minimizer kernels instead of duplicating them.
- `src/normalizing_flows.jl`: backend interface (`fit_flow`, `sample_flow`, `log_density`) and a clear `NotImplemented`/feature error until a validated implementation exists. Do not add a dependency merely to create a placeholder.
- `src/io_bayesian.jl`: versioned HDF5/JSON-compatible serialization of geometry metadata, sampler configuration, RNG seed, chains, weights, diagnostics, and schema version. Never serialize live Python objects.

Include these files from `src/CYAxiverse.jl` and export only stable public names. Keep low-level helpers private unless tests or existing conventions require otherwise.

Suggested minimal usage:

```julia
geom = KahlerGeometry(CYGeometryData(kappa, cone_generators, divisor_basis; metadata))
cfg = SamplerConfig(n_samples=10_000, burnin=2_000, seed=1234,
                    parameterization=:log_cone, volume_bounds=(Vmin, Vmax))
prior = WPStretchedPrior(geom; submanifold_bounds=..., kk_scale=...)
result = sample_mcmc(geom, prior; config=cfg)
posterior = condition(result, loglikelihood)
```

The exact names may differ, but the separation between geometry, prior, sampler, likelihood, and result must remain.

## Python/CYTools boundary

Create a Python adapter in `python/` or `scripts/` only for operations that truly require CYTools. It should:

- accept a geometry identifier/path and explicit options;
- load the CYTools geometry, obtain the Kähler cone generators and intersection data, and extract only serializable arrays/scalars;
- identify/export prime toric divisors and any CYTools-derived topology/classification labels;
- write a versioned `.npz`/JSON/HDF5 interchange artifact with basis ordering, units, orientifold sector, source geometry, and a schema version;
- fail loudly on missing fields or ambiguous basis conventions;
- contain no MCMC, posterior weighting, Julia calls, or cosmology likelihood logic.

Prefer calling the adapter as a subprocess or using the existing optional PyCall extension only as a thin bridge. The Julia side should be able to ingest the produced artifact without PyCall. Add a fixture artifact generated once from a tiny supported geometry for CI.

Document the exact CYTools API calls in the adapter and pin/record the CYTools version used to create an artifact. Never infer divisor ordering from filenames or dictionary iteration order.

## MCMC requirements

Configure an established random-walk/adaptive Metropolis implementation (prefer `AdvancedMH` through `AbstractMCMC`) in `θ = log(α)` and make proposal scale/covariance configurable. The target log density must include every coordinate transformation term required by the chosen parameterization. Provide:

- deterministic seeded runs;
- multiple independent chains;
- warm-up/adaptation separated from retained samples;
- optional parallel chains without shared mutable RNG state;
- rejection counters by cause;
- finite-value and positive-definiteness guards;
- a documented convention for whether samples are prior draws, likelihood-weighted draws, or posterior draws;
- effective sample size/autocorrelation diagnostics that do not pretend a thinned chain is independent;
- checkpoint/resume support after the basic sampler is correct.

Use stable linear algebra (`cholesky`, `logdet`) and avoid explicit matrix inverses. Verify the external sampler’s warm-up, adaptation, initial-state, multi-chain, callback, and serialization behavior against its documented API. Implement only missing glue and CYAxiverse diagnostics. Consider gradients only after the reference sampler is validated. The sampler must work for `d=1`, simplicial cones, and a small non-simplicial cone fixture before being optimized for `d≈30`.

## Likelihood and inference requirements

Build likelihoods as additive log terms with a small protocol such as `loglikelihood(like, state)`. Include:

- generic Gaussian and log-normal likelihoods;
- a divisor-volume window/measurement centered near `tau≈40` as a configurable example;
- a QCD-like mass plus divisor-volume likelihood example;
- an optional cosmology callback receiving `(m_lightest, f_lightest, cosmology_parameters)` and returning a log likelihood, with no AxiCLASS/CLASS dependency in the core;
- posterior conditioning and normalized importance weights in log space;
- marginal histograms/KDE hooks that return data, not plots;
- posterior summaries for geometry IDs, divisor labels, Hodge number, volume, divisor volumes, masses, and decay constants.

The cosmology bridge, if later added, must be an optional external executable/Python adapter with a serializable contract. Do not embed a fragile Boltzmann solver into the package core.

## Verification plan

Add unit tests and small integration tests in `test/`:

1. Analytic one-modulus geometry: verify `V`, `tau`, metric, scaling, and log-WP density.
2. Symmetric-tensor permutation invariance and dimension checks.
3. Cone parameterization: every accepted point satisfies positivity; invalid points are rejected deterministically.
4. Stretched and volume constraints at boundary/interior points.
5. MCMC on a known Gaussian target through the selected external sampler: mean/covariance, acceptance range, reproducibility, and diagnostics; include a small direct test of the CYAxiverse adapter’s transformed target density.
6. WP prior smoke test on a fixture geometry: finite densities, positive-definite metrics, no NaNs, and expected scale-invariant behavior before cutoffs.
7. Likelihood composition: hand-computed Gaussian/log-normal values and posterior weight normalization.
8. Serialization round trip, including schema/version metadata and RNG/configuration fields.
9. Python adapter fixture: exported arrays round-trip into Julia with basis order and metadata preserved; tests must skip cleanly when CYTools is unavailable.
10. Existing package test suite and `Pkg.test()` remain green without PyCall/CYTools.

Add benchmark scripts for dimensions 3, 7, 10, and 30, reporting evaluations/sec, acceptance, ESS/sec, memory, and rejection causes. Use the paper/repository’s diagnostic quantities for comparison, but do not assert exact scientific plots until conventions and fixture geometry match.

## Documentation and reproducibility

Add a concise user guide covering installation without Python, optional CYTools setup, artifact generation, a Julia-only fixture run, MCMC configuration, likelihood composition, and HDF5 schema. Record:

- paper and repository commit/version;
- CYTools version and geometry source;
- basis and normalization conventions;
- RNG algorithm/seed;
- all physical cutoff and likelihood parameters;
- whether samples are prior or posterior samples.

Use the paper’s results as validation targets, not as hidden constants. Cite arXiv:2512.00144v1 and the upstream repository in docs and module docstrings; preserve the upstream Apache-2.0 notice if code is adapted rather than reimplemented.

## Delivery sequence

Work in this order and keep each step reviewable:

1. Reconnaissance and convention note; identify reusable CYAxiverse spectrum kernels.
2. Typed geometry/interchange schema and Python adapter fixture.
3. Julia geometry/WP/constraint kernels with analytic tests.
4. Baseline MCMC and diagnostics with known-target tests.
5. Likelihood composition, posterior summaries, and serialization.
6. Axion post-processing integration using existing Julia code.
7. Normalizing-flow interface and optional implementation decision.
8. Documentation, benchmarks, full test run, and branch handoff.

At each step run the narrowest relevant tests, then the full suite before handoff. Do not claim parity with the paper until at least one fixture reproduces the expected qualitative WP volume/divisor behavior and the conventions are explicitly documented.

## Completion checklist

- [ ] Changes are on `bayesian-inference` and based on `vmm`.
- [ ] Julia-only import and tests work without Python/CYTools.
- [ ] CYTools-dependent functionality is isolated to the Python adapter/optional bridge.
- [ ] WP metric/prior, stretched-cone constraints, MCMC, likelihoods, diagnostics, and serialization are implemented and tested.
- [ ] Existing axion spectrum functionality is reused and remains backward compatible.
- [ ] Deterministic fixtures and benchmarks are included; large upstream data is not vendored.
- [ ] User-facing API and interchange schema are documented.
- [ ] `Pkg.test()` passes, and any unavailable optional integrations are clearly reported rather than silently skipped.
