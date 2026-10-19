# CYAxiverse P0-C compatibility and repository-consumer inventory

Status: `DONE` (inventory evidence only; no production, test, or specification file was changed).

## Scope and identity

This inventory records observable Julia dispatch, data, status, and persistence
surfaces at source revision
`7a40285bb5c313f7e8746b90644d5f45bb67be44`.  The repository handoff is treated
as the request context.  Its instructions were not copied into production code;
repository instructions in `AGENTS.md` and the scientific-reproduction and
Julia-quality skills were applied first.  This file is an implementation and
consumer inventory, not a claim that every numerical route is scientifically
correct or interchangeable.

Evidence labels used below:

* **implementation** means the behavior is read directly from the pinned source;
* **consumer** means a repository script, notebook, test, extension, or document
  uses the binding or persisted field;
* **inference** means a compatibility risk inferred from those two observations
  and requiring a future fixture or owner decision.

## Search and checks performed

The following searches were run from the repository root at the pinned revision.
The output was reviewed rather than treating an export list as a complete API:

```text
git rev-parse HEAD
git status --short --branch
rg --files src ext test scripts bin add_functions paper_benchmarks notebooks docs specs validation | sort
rg -n '^\s*(export|function|struct|mutable struct|abstract type|const [A-Za-z_])' src ext add_functions
rg -n --glob '*.jl' --glob '*.md' --glob '*.toml' 'CYAxiverse\.(generate|read|minimizer|filestructure|jlm_reduced|inflation_points|axion_photon|paper_benchmarks|plotting|glimmers|axion_benchmarks|profiling|slurm)' scripts notebooks docs test src ext
rg -l --glob '*.jl' --glob '*.md' 'CYAxiverse\.profiling' scripts notebooks docs test src ext
rg -n --glob '*.jl' --glob '*.md' 'h5open|create_group|spectrum/|cytools/' src scripts notebooks docs test
rg -n --glob '*.jl' 'ENV\[|haskey\(ENV|ARGS|newARGS|SLURM' src scripts notebooks test
```

The exact source revision was `7a40285bb5c313f7e8746b90644d5f45bb67be44`; the
worktree was clean before this artifact was created.  During this worker turn
the shared branch advanced to manager commit `99d702c` (P0 specification files
only).  `git diff --name-only 7a40285bb5c313f7e8746b90644d5f45bb67be44..HEAD`
contains only `specs/0170-p0-numerical-equivalence/{plan,spec,tasks}.md`; no
source file in this inventory changed.  `git diff --check` and the
repository-relative-path check for this file were run after writing it.

## Namespace topology, exports, aliases, and extensions

`src/CYAxiverse.jl` includes `structs`, `filestructure`, `read`, `minimizer`,
`generate`, `jlm_reduced`, `paper_benchmarks`, `inflation_points`, and
`axion_photon`, then includes `add_functions/profiling.jl` and `plotting.jl`.
The root module exports only `greet_CYAxiverse()`, which returns the literal
string `"Hello CYAxiverse!"`.  A binding being reachable as
`CYAxiverse.generate.foo` therefore does not mean that `foo` is exported from
either the root or its submodule.

Root bindings that are intentionally observable:

| Binding | Identity and compatibility note |
|---|---|
| `CYAxiverse.glimmers` | `const glimmers = axion_photon`; narrow historical alias to the axion/photon module. No exact `CYAxiverse.glimmers` call was found in repository consumers; documentation and scripts use the canonical module or the `axion_benchmarks` name. |
| `CYAxiverse.axion_benchmarks` | `const axion_benchmarks = paper_benchmarks`; benchmark namespace alias. Several inflation scripts use it. |
| `CYAxiverse.profiling` | Included unconditionally from `add_functions/profiling.jl`; no exports. It remains directly used by `notebooks/tree_testing.jl`. |
| `CYAxiverse.slurm` | Included only when `ENV["SLURM_JOB_ID"]` exists while `CYAxiverse` is loaded. It is absent in an ordinary non-SLURM process. |
| `CYAxiverse.plotting` | Always declares data types and renderer function names. Renderer methods are supplied only by the CairoMakie extension. |

Only these submodules declare exports in the pinned source:

* `jlm_reduced`: `ReducedJLMProblem`, `prepare`, `minimize`, `minimize_save`,
  `critical_ensemble`.
* `inflation_points`: `PointContext`, `PointDerivatives`, `PointDiagnostics`,
  `CorrectionResult`, `PrecisionComparison`, `prepare_context`, `derivatives`,
  `diagnose`, `correct_stationary_point`, `compare_precision`,
  `mass_eigenbasis`, `basis_policy`, `prepare_geometry_context`,
  `gradient_flow`.
* `axion_photon`: all public types and functions listed in its API section below.
* `plotting`: all plotting data types, constructors, renderer names, and legacy
  plotters listed below.

`structs`, `filestructure`, `read`, `minimizer`, `generate`, `paper_benchmarks`,
`profiling`, and `slurm` do not declare exports.  Their module-qualified names
and direct bindings imported by repository code are compatibility surfaces.

`Project.toml` declares weak dependencies and extensions:

* `CYAxiverseCairoMakieExt = ["CairoMakie", "ColorSchemes"]`;
* `CYAxiversePyCallExt = "PyCall"`.

The CairoMakie extension imports plotting declarations and adds renderer
methods only when both optional packages are loaded.  The PyCall extension
exposes `enable_cytools!`, `cytools_wrapper`, `jlm_python`, and the Python-backed
`jlm_minimizer`; it requires explicit initialisation and a PyCall Python that
contains CYTools/MOSEK.  Failure to initialise raises an error with setup
instructions; it does not silently fall back.

## Potential encoding and route boundaries

The historical package representation is route-dependent but has a common
stored convention: one scale row is a signed coefficient or sign/mantissa and
the other row is `log10` scale.  The usual oriented package input is `L::Matrix`
with shape `2 × N`, `L[1,a]` the signed coefficient (normally `±1`) and
`L[2,a]` the stored base-10 exponent, paired with `Q::Matrix{Int}` of shape
`h11 × N`.  Raw `read.potential` preserves on-disk orientation and does not
canonicalize it.  `read.oriented_potential` is the explicit orientation and
validation boundary for workflows that request it.

The following transformations are not globally interchangeable:

* `generate.LQtilde(Q,L)` sorts instantons by descending `L[2,:]`, then greedily
  selects exact independent charge columns.  `generate.LQtildebar(L,Q)` performs
  the historical leading/subleading decomposition and threshold merge.
* `read.oriented_potential` accepts a transposed `L` or `Q` where it can infer
  the orientation, requires `L` to be `2 × N`, `Q` columns to match, `K` to be
  square and finite, and optionally canonicalizes triangular generated
  potentials.  For a triangular term count it removes duplicate leading charge
  columns and duplicate pairwise differences, keeps the first representative,
  and rejects inconsistent duplicate coefficients.  Non-triangular term counts
  are left unchanged because a safe boundary cannot be inferred.
* `axion_photon.load_instanton_data` reads `Q` and `L` without reorientation,
  sorts by descending `L[2,:]`, retains only signs from `L[1,:]`, and stores the
  source column order.  Its hierarchy formula therefore treats the first row as
  a sign under the normal local-file contract; non-unit mantissas are not a
  drop-in replacement for the general `generate` evaluator (inference).
* `generate` derivative and spectrum routines multiply `L[1,a] * 10^L[2,a]`
  and thus retain a non-unit coefficient factor.  `inflation_points` retains the
  same factor in its shifted amplitudes.  `paper_benchmarks` uses both fixed
  full coefficients and sign/log scale matrices according to the model route.
* Legacy matrix formulas in `generate.V`, `generate.jacobian`, and
  `generate.hessian` use the older instanton-row matrix layout in places,
  whereas `generate.cubic`, modern spectrum paths, `minimizer.critical_points`,
  `jlm_reduced`, and `inflation_points` use charge columns.  Do not reorder or
  transpose raw arrays globally.

Term order is observable: `generate` PQ and HP arrays are ordered after their
  route-specific selection/sorting; `axion_photon` reports one-based
  `source_indices` into the original input after its descending scale sort;
  quartic index matrices intentionally contain zero-based mode indices.  The
  future numerical input boundary must be selected separately for each route.

### Exact-zero and `-Inf` behavior

This is an implementation inventory, not a repair:

* `read.oriented_potential`, `inflation_points.prepare_context`,
  `generate.logshifted_derivative_workspace`, hierarchy diagnostics, and
  `axion_photon._normalise_potential` reject non-finite `L`; `axion_photon`
  additionally rejects zero coefficients.  They raise `ArgumentError` (or
  `DimensionMismatch` for shape errors) before numerical work.
* `generate` accepts a zero first-row coefficient when `L` is otherwise finite;
  derivative terms then contribute exactly zero.  A row with no nonzero charge
  support is skipped by `minimizer.critical_points`, but its equation scaling
  and classification still depend on the global/available log-scale path.
* An all-`-Inf` log-scale vector makes `maximum(L[2,:]) == -Inf` and the shifted
  expression `L[2,:] - maximum(L[2,:])` evaluate as `-Inf - (-Inf) == NaN`.
  This can produce NaN amplitudes and downstream failure; no normalization away
  of this state is performed.  Mixed finite/`-Inf` inputs are rejected by the
  current finite-value guards in the modern paths.  `generate.logsum_sorted!`
  itself returns `-Inf` for an all-`-Inf` prefix and propagates `NaN` for a
  NaN maximum.  `generate.pq_contracted_log!` reports a zero contraction with a
  signed-zero/`-Inf` log according to its log-domain path.
* `inflation_points` requires at least one axion and instanton, finite `L` and
  `K`, and a successful Cholesky factorization.  Zero potential gives `Inf`
  epsilon/eta; gradient flow requires positive potential and otherwise returns
  a `:failed` result with the captured `DomainError` text.

These states require hostile fixtures before P2, especially a zero-only-support
row and an all-`-Inf` log row.  They were inventoried but no fixture was added in
P0 because this worker's write set is limited to this report.

## Public and non-exported result types (`structs`)

All types below are reachable as `CYAxiverse.structs.<name>` and are not root
exports.  Their fields are observable by scripts and tests.

| Type | Fields/shape contract |
|---|---|
| `GeometryIndex{T<:Integer}` | keyword fields `h11`, `polytope`, `frst=1`; used by all indexed overloads. |
| `TopologicalData` | `points::Matrix{Int}`, `simplices::Matrix{Int}`. |
| `GeometricData` | `tip_prefactor::Vector{Float64}`, `τ_volumes::Vector{Float64}`, `h21::Integer`, `cy_volume::Float64`, `glsm_charges::Matrix{Int}`, `basis::Vector{Int}`, `tip::Vector{Float64}`, `kinv::Matrix{Float64}`, `hilbert_basis::Matrix{Int}`. |
| `AxionPotential` | raw `L::Matrix{Float64}`, `Q::Matrix{Int}`, `K::Hermitian{Float64,Matrix{Float64}}`. |
| `QuarticComponentDiagnostics` / `QuarticDiagnostics` | cancellation vectors `orders_lost`, `digits_remaining`, `reliable`, `exact_zero`; aggregate fields `self`, `three_one`, `two_two`. |
| `MassBasisDiagnostics` | `eigenpair_residuals`, `nearest_relative_gaps`, `orthogonality_error`. |
| `InstantonScaleBlock` | `indices` are zero-based original `L` columns; `sorted_positions` are one-based positions; `log10_scales`. |
| `PerturbativeSplitDiagnostics` | `off_block_norm`, `separation_gap`, `coupling_to_gap_ratio`, `certified_safe`. |
| `InstantonHierarchyDiagnostics` | leading gap/span, heuristic flag, blocks, inter-block gaps, perturbative split records, `gap_log10`, `min_block_size`; 3-argument compatibility constructor fills empty diagnostics. |
| `SpectrumWindowDiagnostics` | counts by precision, boundary counts/gaps/margin, residual, `converged`, `fallback_used`, `certified`, `provisional`, and hierarchy. |
| `PhysicalAxionSpectrum` | masses, mode indices/eigenvectors, signed/log self/31/22 quartics, `threshold_log10`, `prec`, window bounds, optional diagnostics; legacy 13-argument constructor sets window min to threshold, max `Inf`, diagnostics `nothing`. |
| `AxionSpectrum` | `m`, `msign`, `f`, `fK`, signed/log self/31/22 fields, and optional quartic, mass-basis, hierarchy diagnostics. |
| `IndexedAxionSpectrum{T<:Float64}` | `h11`, `polytope`, `frst`, and `m`, `f`, `fK` vectors. |
| `LQLinearlyIndependent` | `Qtilde`, `Qbar` integer matrices; `Lbar`, `Ltilde` Float64 matrices. |
| `Projector`, `ProjectedQ`, `CanonicalQBasis`, `Canonicalα`, `ReducedPotential` | rational/Float projectors and the leading/bar charge, scale, alpha, mask, and reduced potential fields used by old profiling and JLM routes. |
| `Solver1D`, `SolverND`, `RationalQSNF`, `BasisSNF` | legacy JLM solver state and exact-SNF state; field types include untyped legacy fields, rational matrices, and `volume::Number`. |
| `Min_JLM_1D`, `Min_JLM_ND`, `Min_JLM_Square` | `N_min`, plus coordinates/`extra_rows` for 1D/ND or `det_QTilde` for square. |
| `MyTree{D}`, `ParentTrack{T}` | AbstractTrees-compatible parent/children tree and custom path iterator; constructors mutate parent `subtrees`. |

The `MyTree` parent-push side effect and zero/one-based conventions in
`InstantonScaleBlock` and quartic indices are compatibility details.

## `filestructure` (non-exported module-qualified surface)

### Methods and routing

* `localARGS()` returns `ENV["newARGS"]` when present, otherwise `ARGS`.
* `ol_DB(args)` maps recognized legacy aliases (`KU_Fair`, `inKC`, `home_Large`,
  `vacua_test`, `vacua_stretchtest`, `vacua_new`, `vacua_0323`, `vacua_0822`,
  `vacua_stretch`, `docker`) and `"pwd"`; unknown aliases throw `ArgumentError`.
* `default_data_dir()` returns the checkout sibling `../data` only when the
  package root looks like a checkout; otherwise `nothing`.
* `resolve_data_dir(data_dir=nothing)` precedence is explicit nonempty argument,
  `CYAXIVERSE_DATA_DIR`, recognized `newARGS`, then default checkout data.
  It normalizes to an absolute expanded path, strips a trailing separator except
  at root, does not create directories, and throws `ArgumentError` for an empty,
  unknown, or unresolved route.
* `present_dir()` and `present_dir(data_dir::AbstractString)` return the
  resolved absolute directory with a trailing separator. `data_dir()` is a
  deprecated compatibility alias that calls `present_dir` and does not append an
  extra `/data` component.
* `plots_dir()` creates and returns `<present_dir>/plots` using `mkpath`.
  `log_dir()` creates `<present_dir>/logs` using `mkdir` only if absent; races or
  an invalid parent can raise the underlying filesystem exception.
  `logfile()` joins `log_dir` with a current `DateTime` and `log.out`.
  `logcreate(path::String)` opens in `"w"` and overwrites with a timestamp line.
* `np_path_generate(h11::Int; geometric_data=false)` and the keyword-only
  all-`h11` overload traverse `h11_###/np_#######/cy_#######/cyax.h5` and
  return `(hcat(relative_path_bytes...), hcat(indexes...))`; `geometric_data`
  retains only cells passing `isgeometry`. Empty traversals can fail in `hcat`.
* `np_path()` reads/generates paths and creates `paths_cy.h5` only when neither
  `paths.h5` nor `paths_cy.h5` exists; it writes datasets `paths` and `pathinds`
  with deflate 9. `paths_cy()` reads `paths.h5` for `localARGS()=="in_KC"`,
  otherwise `paths_cy.h5`, and returns `(Vector{String}, pathinds)`. Missing
  files fall back to `np_path()`.
* `h11lst(h11min=0,h11max=100)` filters the first row of path indices by the
  half-open interval `(h11min,h11max]`. `h11lst(h11list::Vector;
  geometric_data=false)` returns generated file lists/counts; the
  `geometric_data` loop contains a local `col=zero(col)` assignment and does
  not mutate the file-list array (observable legacy quirk).
* `count_geometries(n=nothing)` returns a two-row `hcat` of unique `h11` and
  counts, with optional numeric cutoff; unsupported `n` values produce no rows.
  `isgeometry(h11,tri,cy)` checks the `cytools/geometric/h21` dataset.
* `geom_dir`, `geom_dir_read` (and `GeometryIndex` overloads) implement legacy
  path layouts; `geom_dir` creates missing directories, while `_read` returns
  nothing if absent. `inKC` omits the `cy` directory; `home_Large`/`KV1` and
  `h11>=238` have special nested `cy` behavior. `Kfile`, `Qfile`, `Lfile` are
  deprecated geometry-local path aliases. `cyax_file` and `minfile` resolve
  `cyax.h5` and `minima.h5` respectively.

## `read` surface and persisted input/output contracts

All indexed methods accept `(h11::Int, tri::Int, cy::Int=1)` or a
`GeometryIndex` overload where shown. Errors from absent files/datasets are the
underlying HDF5 errors unless an explicit shape/metric check below intervenes.

* `topology` returns `TopologicalData` from
  `cytools/geometric/points` and `cytools/geometric/simplices`.
* `geometry(...; hilbert=false)` returns `GeometricData`. Common fields come
  from `cytools/geometric/{h21,glsm,basis,tip,CY_volume,divisor_volumes,Kinv}`;
  with `hilbert=true`, metric/volume/tip fields move to
  `cytools/hilbert/geometric`. Missing `tip_prefactor` defaults to two ones;
  missing `hilbert_basis` defaults to a zero `h11×h11` matrix.
* `hilbert_basis` returns the stored geometric basis or a zero matrix when the
  dataset is absent. `visible_sector` returns `nothing` when its group is absent,
  otherwise a NamedTuple containing effective seed/pool/rank, QCD/QED indices
  and volumes, source positions/scales, integer charge vectors, invariant and
  intersection booleans, and policy/selection/terminal metadata strings.
  `construction_metadata_json` returns the root attribute string or `nothing`.
* `potential(...; hilbert=false, validate=true)` returns `AxionPotential` using
  `cytools/potential/{L,Q}` and `cytools/geometric/Kinv`; `hilbert=true` reads
  the corresponding `cytools/hilbert/potential/{L,Q}` and hilbert `Kinv`.
  K is built as `Hermitian(inv(Hermitian(Kinv)))`; exact singularity is converted
  to a geometry-named `DomainError`, and `validate=true` rejects non-SPD K with
  a `DomainError` containing the minimum eigenvalues. `validate=false` skips
  only the SPD check.
* `potential_factored` returns `(;L,Q,Kinv,C)` after symmetrizing Kinv and
  Cholesky-factorizing it; non-square Kinv raises `DimensionMismatch`, and a
  failed Cholesky raises its matrix-factorization exception. `oriented_potential`
  returns `(;Q,L,K)` with optional `canonicalize_charge_rows=true`; it accepts
  inferable transposed input and rejects mismatched dimensions/non-finite data.
  `Q`, `K`, `L_log`, and `L_arb` are legacy direct readers. `Q/K/L_log` ignore
  `hilbert` and read the root potential datasets; `K` symmetrizes raw K;
  `L_arb` converts each signed log term to an `ArbFloat` coefficient.
* `cubic_tensor` returns `(;tensor, phase)` from `spectrum/cubic/{tensor,phase}`.
  `qshape` returns `(issquare,vacua_estimate,lengthα,ωnorm2)` from local
  `qshape.h5`, with optional fields defaulting to `nothing`; absent file raises
  `ArgumentError`.
* `vacua` and `vacua_TB` return `(;vacua,θ_parallel,Qtilde)` for `h11<=50`
  using numerator/denominator rational datasets, while larger h11 values use
  decimal theta data (or no theta for TB) and still return `vacua/Qtilde`.
  `vacua_jlm(...;hilbert=false)` returns one of `Min_JLM_Square`, `Min_JLM_1D`,
  or `Min_JLM_ND` by `extra_rows` and coordinate rank, reading root or
  `hilbert/{Nvac,vac_coords,extra_rows,det_QTilde,issquare}`.
* `pq_spectrum` reads `spectrum/masses/log10`, `spectrum/decay/{fK,fpert}` and
  returns `(;m,fK,fpert)`. `physical_spectrum` reads
  `spectrum/physical`, including signed/log self and mixed quartics, mode
  indices, `fK_log10`, and metadata (`threshold_log10`, `prec`,
  `provisional`, runtime, units/formula/convention/log policy/truncation).
  `hp_spectrum` reads legacy masses/decay/quartic groups and returns
  `(;m,fK,fpert,λself,λ31_i,λ31,λ22_i,λ22)`; its reader intentionally omits
  stored signs from the returned NamedTuple.
* `pipeline_vacua` returns `(;threshold,estimate,issquare,extrarows,verified,
  theta_min,theta_parallel,metadata)` from `vacua_pipeline`. Rational theta
  groups have numerator/denominator; metadata preserves solver/search status,
  tolerances, method, branch counts, determinant, classification schema,
  model scope, full-potential status, configuration digest, Julia/revision,
  runtime, completion, and error fields. Missing optional fields become
  `nothing`; absent root group propagates HDF5 access failure.

## `generate` surface (large non-exported module)

The complete definition inventory is in `src/generate.jl`; the following groups
freeze the dispatch domains, shape conventions, keyword defaults, and return
contracts that consumers use.

### Inputs, derivatives, and helpers

* `constants()` returns `Dict{String,ArbFloat}` with `MPlanck`, `Hubble`, and
  `log2π`. `pseudo_Q(h11::Int,tri::Int,cy::Int=1)` returns a random
  `(h11+4+binomial(h11+4,2)) × h11` integer matrix; `tri` and `cy` are accepted
  but do not seed or otherwise change generation. `pseudo_K(h11::Int,tri::Int,
  cy::Int=1)` returns a random SPD `Hermitian`; `pseudo_L(...;log::Bool=true)`
  returns either a `N×2` sign/log Float64 matrix or an `ArbFloat` vector when
  `log=false` (the keyword is tested by `==1`).
* `V(x,L,Q)`, `jacobian(x,L,Q)`, and `hessian(x,L,Q)` are legacy matrix methods
  with `Matrix{Float64}` L and Matrix Q, preserving their historical orientation
  and return types (scalar/SVector/Hermitian). `V(x;L,Q)` is a separate keyword
  method for charge-column input. `cubic(x,L,Q)` accepts `AbstractVector` and
  real matrices, returns an `n×n×n` tensor; `cubic(x,phase,L,Q)` evaluates at
  `x+phase`. `hessian_norm(x,Q)` has shape-dependent matrix/tensor returns.
  These methods use `@assert` or dimension errors rather than a common validator.
* `gauss_sum(::Float64)`, `gauss_diff(::Float64)`, `gauss_log_split(::Vector{Int},
  ::Vector{Float64})`, and `gauss_log` operate in signed logarithmic space;
  extreme arguments are handled by branch thresholds, while cancellation can
  return sign zero and `-Inf` log. `logsum_sorted!(logs::Vector{Float64},n::Int)`
  returns `-Inf` for `n=0` and checks the prefix bound.
* `LogShiftedDerivativeWorkspace`, `StructuredChargeRepresentation`,
  `StructuredLogShiftedDerivativeWorkspace`, and `StructuredChargeEvaluator`
  are public-by-binding structs. `structured_charge_representation(Q::AbstractMatrix
  {Int},L::AbstractMatrix{Float64};base_count=size(Q,1)+4)` validates the
  base-plus-pairwise corpus and returns `validated=false` plus a fallback reason
  on shape/charge failure. `structured_charge_evaluator` then selects the
  structured or generic evaluator. `logshifted_derivative_workspace` requires
  `L` 2×N and matching Q columns; `logshifted_derivatives!` mutates/borrows
  workspace buffers and returns `(;value,gradient,hessian,log_shift)`. Callers
  must consume arrays before the next call. Wrong dimensions/non-finite L raise
  `DimensionMismatch`/`ArgumentError`.

### Spectrum and hierarchy APIs

* `hp_spectrum(K::Hermitian{Float64,Matrix{Float64}},L,Q; prec=1000,
  quartics=true,selection=:raw)`, and indexed/GeometryIndex overloads with
  default `selection=:hp_effective`, return a legacy `Dict` with keys `msign`,
  `m`, `fK`, `fpert`, `λselfsign`, `λself`, `λ31_i`, `λ31sign`, `λ31`,
  `λ22_i`, `λ22sign`, `λ22`. `quartics=false` returns empty mixed quartics.
  Arbitrary precision is global ArbFloat state and logs are finally Float64.
  `hp_spectrum_save(...;phase=zeros(Float64,h11))` writes masses, decay,
  quartic, and cubic groups; indexed path errors propagate.
* `pq_spectrum(K,L,Q;mixing_correction=:float64,prec=1000,
  quartic_diagnostics=false,mass_basis_diagnostics=false,
  hierarchy_diagnostics=false)` and indexed overloads return `AxionSpectrum`.
  K must be Hermitian Float64, L 2×N, Q integer h11×N. `mixing_correction`
  accepts `false`/`:none`, `true`/`:high_precision`, `:float64`, or
  `:high_precision`; diagnostics with no mass basis raise `ArgumentError`.
* `pq_physical_spectrum` returns `PhysicalAxionSpectrum`; keywords include
  `threshold_log10` (default log Hubble), `prec=1000`, and `quartics=true`.
  `pq_hybrid_physical_spectrum` adds `maxiter`, residual tolerance, Schur
  acceleration, oversampling, `quartic_backend=:auto`, mixed quartics, hierarchy
  gap/min block sizes, and a label. It uses sparse/dense selection and can warn
  or fall back to the high-precision eigensystem. `pq_window_spectrum` accepts
  `min_log10_mass`, `max_log10_mass`, precision, boundary margin, confirmation,
  and quartic options and returns only requested modes plus
  `SpectrumWindowDiagnostics`; reversed/non-finite finite bounds raise
  `ArgumentError`.
* `pq_physical_mode_count`, `pq_schur_admissible`, `pq_hp_alignment`,
  `physical_mode_inertia_count`, `leading_hessian_mass_basis`, and
  `leading_hessian_mass_basis_float64` expose count/admissibility/alignment or
  mass/sign/basis tuples. Confirmation may increase precision up to
  `max_prec`; unresolved precision is recorded as provisional/fallback rather
  than silently certified.
* `instanton_scale_blocks(L::AbstractMatrix{<:Real};gap_log10=1.0,
  min_block_size=1)` requires at least two rows, finite nonnegative gap and
  positive block size; it returns a NamedTuple with blocks (zero-based source
  indices), sorted arrays, inter-block gaps, and diagnostics. The two
  `instanton_hierarchy_diagnostics` overload families accept L alone or
  `(C|K,L,Q)`; `spectrum_mode_counts(eigenvalues::AbstractVector{<:Real};
  relative_tolerance=1e-10)` returns partitioning counts
  `negative+zeroish+positive == length` and tolerance/scale.

### Reduction, vacua, and branch APIs

* `LQtilde(Q::AbstractMatrix{Int},L::AbstractMatrix{Float64})` plus indexed
  overloads (`hilbert=false`) returns `LQLinearlyIndependent`; it requires
  charge columns and scale columns to match. `αmatrix` accepts an LQ object,
  `(Q,L)`, or indexed geometry, with `threshold::Float64=0.5` and optional
  `hilbert`; it returns `Canonicalα` when effective alpha rows remain and
  `CanonicalQBasis` otherwise. `ωnorm2` accepts canonical objects or a
  GeometryIndex and returns a scalar. `LQtildebar(L::Matrix{Float64},Q::Matrix{Int};
  threshold=0.5)` returns a legacy Dict with `Qhat`, `Qbar`, `Lhat`, `Lbar`,
  `α`; indexed overloads load the potential.
* `vacua(L::Matrix{Float64},Q::Matrix{Int};threshold=0.5)`, `vacua_TB`,
  `vacua_id_basis`, `vacua_id`, `vacua_MK`, `vacua_projector`, `vacuaΩ`,
  `vacuaΠ`, `vacua_full`, and `vacua_no_optim` are distinct historical routes.
  They return Dicts or tuples with keys such as `vacua`, `Qtilde`, `θ∥`,
  `θ̃min`, `vac`, `xmin`, and solver diagnostics depending on the route; all
  use explicit `threshold`, optional phase vectors, and (for `vacua_id`) runs.
  `vacua_SNF(Q::AbstractMatrix{<:Integer})` and `basis_snf(rays::Matrix{Int})`
  return exact-SNF structures. Thresholds are linear scale ratios, compared in
  log10 space. Do not replace these routes with `jlm_reduced` without preserving
  their Dict keys and rational/Float64 conversion points.
* `foreach_leading_critical_branch(callback,selected; tolerance=1e-8,
  max_branches=1_000_000,negative_mode_range=nothing,max_negative_modes=nothing)`
  streams leading half-integer branches and returns a report; the reversed
  `(selected,callback)` overload is supported. `leading_critical_branches`
  returns `(;coordinates,leading_negative_modes,branch_count,
  leading_minima_count,det_Qtilde,stream_report)`. Non-positive max limits,
  singular Qtilde, or branch counts over the cap raise `ArgumentError`.
* `reduced_critical_points(L::AbstractMatrix{Float64},Q::AbstractMatrix{Int};
  kwargs...)` is a wrapper around LQtilde and critical_points using selected
  charge coordinates. `jlm_vacua_db(;n=size(paths_cy()[2],2),h11=nothing)` returns
  a NamedTuple of `square`, `one_dim`, `n_dim`, and errors. `vacua_estimate` returns
  `(;vac,issquare[,extrarows])`; its save method writes minima-file estimates.
  Save methods mutate existing HDF5 files and may skip an already-present group.

The generate module also defines many underscore-prefixed helpers used directly
by tests and scan scripts (for example `_leading_mask_count`,
`_leading_det_qtilde`, `_hcat_columns`, `_quartic_index_matrix`,
`_hp_selected_potential`, `_exact_integer_determinant`, and
`leading_independent_mask!`).  They are non-exported but must be treated as
compatibility surfaces because repository consumers call them directly.

## `minimizer`

* `critical_points(L::AbstractMatrix{<:Real},Q::AbstractMatrix{<:Real};
  phases=zeros(size(Q,2)),starts=4096,residual_tolerance=1e-10,
  merge_tolerance=1e-7,max_iterations=200,coordinate_basis=nothing,
  equation_scales=nothing,initial_points=nothing)` accepts broad AbstractMatrix
  real domains, validates 2×P L/Q column agreement and positive starts, and
  returns `(;coordinates,minima,inertia,hessian_eigenvalues,residuals,
  critical_count,minima_count,starts)`. It folds coordinates on the unit torus,
  deduplicates with periodic distance, and classifies the symmetrically scaled
  physical Hessian. Empty row support is skipped; row scaling and the
  `-100eps(Float64)` negative-mode seed threshold are distinct from final
  inertia classification.
* Three indexed legacy `minimize` methods take
  `(h11,tri,cy,LV,QV,x0,gradσ,θparalleltest,Qtilde,algo,prec)`,
  `(h11,tri,cy,LV,QV,x0,gradσ,algo,prec)`, or
  `(h11,tri,cy,LV,QV,x0,gradσ,Qtilde,algo,prec)`. They use Optim and ArbFloat,
  return Dict keys `±V`, `logV`, `±x`, `logx`, and optional `±a`/`loga`,
  `±ã`/`logã`, `Heigs`, `Hsign`, `gradsum`; a failed convergence gate returns
  `nothing`. `minimize_save` writes `runs/<run>/V`, `x`, `a`, and `atilde`
  signed/log datasets and returns `nothing`.
* `grad_std` has indexed, `(LV,QV)`, and geometry-loading overloads and samples
  100 random points; `minimize(LV::Vector,QV,x0::Vector)` returns an Optim Dict
  with `±V`, `logV`, `xmin`, `Heigs`, `Hsign`, `gradlog`. `id_minimize` accepts
  Matrix or Vector QV and returns `xinit`, `xmin`, `Heigs`, `Hsign`, `gradlog` or
  `nothing`; `id_minima` is an assertion-only stub and returns `nothing`.
  `subspace_minimize(L,Q;runs=10000,phase=...)` fixes a random seed and returns
  unique minima; `minima_lattice(v::Matrix{Float64})` returns
  `Dict("lattice_vectors"=>basis)` after a Gram-rank test.
* Direct consumers include notebooks' old Optim workflow and tests of private
  `_legacy_gradient`, `_legacy_hessian`, and `_phase_hessian!`. Preserve those
  bindings even if a future typed solver replaces the implementation.

## `jlm_reduced`

`ReducedJLMProblem` fields are
`Q_reduced::SparseMatrixCSC{Float64,Int}`, `L_reduced::Matrix{Float64}`,
`phases::Vector{Float64}`, `det_QTilde::Int`, `multiplicity::Float64`,
`integer_charges::Bool`, optional `square_vacua`, `extra_rows`, `reduction`,
`coordinate_scale::Vector{Int}`, and `lift_matrix::Matrix{Float64}`.

`prepare(Q::AbstractMatrix{Int},L::AbstractMatrix{Float64};threshold=0.01,
reduction=:alphamatrix)` accepts only `:alphamatrix` or `:catastrophe`; it
selects LQtilde/alpha data and may return a square problem with no numerical
columns, or a sparse non-square problem. The GeometryIndex overload loads and
orients a potential, requires more instantons than axions, and supports
`hilbert=false`. `critical_ensemble(problem;starts=100000,
residual_tolerance=1e-9,merge_tolerance=1e-6,max_iterations=300)` returns the
critical-points NamedTuple plus `coordinates` and `reduced_coordinates`; square
problems return empty coordinate arrays and exact count metadata. `minimize`
returns `Min_JLM_Square`, `Min_JLM_1D`, or `Min_JLM_ND`, with minima count scaled
by `multiplicity` and coordinates converted to `2π` for legacy results.
`minimize_save(geom_idx;threshold=0.01,hilbert=false,kwargs...)` writes root or
`hilbert` minima fields `Nvac`, `det_QTilde`, `issquare`, and nonsquare
`vac_coords`/`extra_rows`; replacement is destructive to that group and returns
the result struct.

## `inflation_points` exports, public types, and statuses

`basis_policy()` returns the contract NamedTuple
`(;working_basis=:periodic_string,physical_basis=:mass_eigenbasis,
physical_vectors=:deferred,dense_charge_rotation=:deferred)`.

Public structs and fields:

* `PointContext{T,F}`: `Q`, `L`, `K`, Cholesky `factor`, shifted `amplitudes`,
  `log_shift`, `precision_bits`.
* `PointDerivatives{T}`: shifted `value`, `gradient`, `hessian`, `log_shift`.
* `PointDiagnostics{T}`: value, metric gradient norm, raw gradient residual,
  epsilon, eta values, generalized-Hessian eigenvalues, negative/zeroish/positive
  counts, zero tolerance, `physical_basis`.
* `CorrectionResult{T}`: folded `theta`, `status`, residual, iteration/time,
  error string, `working_basis`.
* `PrecisionComparison{F,H}`: float/high corrections and diagnostics, residual
  and inertia agreement booleans, `accepted`.

`prepare_context(Q::AbstractMatrix{Int},L::AbstractMatrix{<:Real},K::AbstractMatrix
{<:Real};precision_bits=nothing)` requires h11,N>0, L 2×N, K h11×h11,
finite data, and SPD K. `nothing` means Float64/53 bits; BigFloat requires at
least 64 bits and changes global BigFloat precision during evaluation.
`prepare_geometry_context(geom_idx;precision_bits=nothing)` loads
`read.oriented_potential` and returns `(;geometry,input_basis=:periodic_string,
source=:oriented_potential,context)`.

`derivatives(context,theta)` validates theta length and returns
`PointDerivatives` without mutating context. `mass_eigenbasis(context,data;
vectors=false)` and the theta overload return generalized mass eigenvalues and
basis metadata; `vectors=true` adds raw eigenvectors, metric and generalized
residuals. `diagnose(context,theta;zero_tolerance=1e-10)` reports generalized
Hessian classification, with counts partitioned using the tolerance-scaled
largest eigenvalue. Dimension/finite/SPD failures raise `DimensionMismatch`,
`ArgumentError`, or Cholesky exceptions.

`correct_stationary_point(context,seed;residual_tolerance=1e-10,
max_iterations=100,max_line_search=12)` folds the seed and returns statuses
`:converged`, `:singular_hessian`, `:nonfinite_step`, `:line_search_failed`, or
`:max_iterations`; it does not throw for an ordinary failed Newton path.
`compare_precision(seed,Q,L,K;precision_bits=256,float_residual_tolerance=1e-10,
high_residual_tolerance=1e-40,zero_tolerance=1e-10,max_iterations=100,
max_line_search=12)` runs both corrections and accepts only residual and inertia
agreement.

`gradient_flow(context, hilltop;displacement=1e-8,displacement_sign=-1,
mode=:most_negative,mode_index=nothing,mass_basis=nothing,max_efolds=60,
step=1e-3)` integrates RK4 in canonical Cholesky coordinates. It returns a
NamedTuple with status, error, basis/chart, mode, direction/eigenvalue, initial
and final theta, epsilon/eta, e-fold totals, entry/exit, end event, windows,
steps, horizon, and step. Statuses are `:no_slow_roll_window`, `:completed`,
`:max_efolds` (when an open window reaches the horizon), and `:failed`; the
GeometryIndex overload adds geometry/input/source fields. It requires a positive
potential during flow and records failures instead of changing basis semantics.

## `axion_photon` public module, aliases, and persistence

Exported constants are not declared; module constants are
`M_PLANCK_GEV=2.435e18`, `ALPHA_EM=1/137.035999084`, and
`ELECTRON_MASS_EV=0.511e6`. Exported structs:

* `VisibleSectorAssignment{T}`: divisor/image indices, volumes, QCD/QED/EM
  charge vectors, QED source index/scale, invariant/intersection booleans, policy.
* `GeometryInputs{T}`: path/index, tip/divisor/CY/metric data, direct charges,
  direct volumes/labels, optional visible assignment.
* `InstantonData{T}`: `Q`, descending `log10_lambda4`, coefficient signs, and
  one-based original `source_indices`.
* `RationalRankCertificate`: algorithm, matrix shape, ordered/selected/
  dependent source indices, prefix ranks, exact BigInt determinant.
* `LeadingAxionHierarchy{T}`: selected/dependent terms, certificate, reduced Q,
  scales/signs, `q`, `theta_from_canonical`, log f/mass, Planck value, triangular
  and metric residuals.
* `AxionPhotonObservables{T}`: EM coefficients, Cgamma, log couplings/widths,
  threshold and light-mode count, charge residual.
* `AxionPhotonConfiguration`, `AxionPhotonIdentity`, `AxionPhotonResult{T}`:
  configuration/policy/precision, geometry/potential/snapshot digests, and the
  complete geometry/potential/hierarchy/photon result plus status.

Methods and domains:

* `load_instanton_data(path::AbstractString;T=Float64)` requires a file and
  `cytools/potential/Q,L`; Q is integer, L is 2×N, finite and nonzero in first
  row. It sorts descending stored exponent and preserves source indices.
  `load_geometry_inputs(path;T=Float64,index=nothing)` requires geometric
  fields `tip`, `divisor_volumes`, `CY_volume`, `Kinv`, `effective_cone`,
  `prime_divisor_volumes`, `prime_toric_divisors`; wrong/missing dimensions,
  nonpositive volumes, non-SPD metric, or missing file raise `ArgumentError`,
  `DimensionMismatch`, or Cholesky errors. GeometryIndex/data-dir overloads
  resolve the canonical `h11_###/np_#######/cy_#######/cyax.h5` path.
* `local_geometry_indices(data_dir;h11s=(15,100,200,300),limit_per_h11=2,
  require_complete=true)` returns deterministic sorted complete indices; missing
  root, nonpositive limit, or no complete match raises `ArgumentError`. The
  keyword-only overload resolves `data_dir` first.
* `rank_certificate_payload(certificate)` returns JSON-ready fields with matrix
  shape vector and determinant string. `leading_hierarchy(potential,kinv;
  T=Float64,signed_scale_policy=:require_positive,m_planck_GeV=M_PLANCK_GEV)`
  performs exact rank selection and a canonical frame; Kinv must be square and
  SPD. `signed_scale_policy` accepts `:require_positive` or `:absolute`; the
  former rejects non-positive selected terms, the latter adapts and is marked.
  `mixing_matrix` returns the reduced mixing matrix.
* `qed_instanton_log10_threshold_eV(geometry; m_planck_GeV=...)` requires a
  visible sector and positive charge norm; linear threshold may underflow to
  zero in `qed_instanton_threshold_eV`. `photon_observables(result,em_charge;
  alpha_em=ALPHA_EM,light_threshold_eV=ELECTRON_MASS_EV,
  light_threshold_log10_eV=nothing)` solves the EM charge map, validates charge
  and positive/finite threshold, and returns log widths; the last mode has
  quartic width `-Inf` where no neighbor exists.
* `run_local_scan(;data_dir=nothing,h11s=(15,100,200,300),limit_per_h11=2,
  require_complete=true,T=Float64,em_divisor_index=nothing,
  light_threshold_eV=ELECTRON_MASS_EV,qed_threshold_policy=:electron_proxy,
  signed_scale_policy=:require_positive)` returns a vector of results and
  raises `ArgumentError` if no matches. QED policy accepts `:electron_proxy` or
  `:divisor_instanton`; signed absolute results have status
  `:adapted_absolute_scale`, divisor threshold status is
  `:visible_sector_instanton_threshold`, and the ordinary route is
  `:adapted_local_geometry`.
* `write_scan_csv(path,results)` requires an existing parent directory and
  writes fixed columns for path/index/status/counts/divisor/light thresholds,
  couplings/width summaries and residuals. It returns an absolute normalized
  path. `run_batch_axion_photon` returns one NamedTuple per geometry with
  `index,path,status,error`; statuses are `:skipped`, `:written`, and `:failed`.
  `skip_complete=true` uses identity/configuration checks; per-geometry failures
  are captured rather than stopping the batch.

The persisted result group is `spectrum/axion_photon`. It has scalar top-level
datasets `em_divisor_index`, `em_divisor_volume`, `em_charge_source`,
`light_threshold_policy`, and `status`; `identity` with schema/index/digests and
`configuration`; `hierarchy` with selected/dependent indices, reduced Q/scales,
signs, q, canonical transform, log f/mass, Planck and residuals plus a
`rank_certificate`; and `photons` with n_em, Cgamma, log couplings/widths,
threshold, light count, and charge residual. Arrays are compressed at deflate 9.
`write_axion_photon_result(path,result;force=false)` stages and validates an
identity-preserving update, rejects an existing valid group unless `force=true`,
and returns path. `read_axion_photon_result` validates schema, source index,
input/snapshot digests, and configuration; malformed/tampered/missing groups
raise `ArgumentError`. Compatibility aliases are `GlimmersGeometry`,
`GlimmersPotential`, `GlimmersHierarchy`, `GlimmersPhotonObservables`,
`GlimmersPilotResult`, `load_geometry`, `load_potential`, `hierarchy`,
`run_local_pilot`, and `write_pilot_csv`.

## `paper_benchmarks` and `axion_benchmarks`

`paper_benchmarks` includes reduced models, the nested `author_inflation`
(`poly102_inflation` alias), compatibility aliases, fuzzy-axion mass helpers,
model-stage helpers, and catastrophe diagnostics. It has no export declaration,
so both parent and nested module-qualified names are observable.

### Reduced and author model domains

Both routes define overlapping names such as `instanton_scales`, `n5_potential`,
`n8_potential`, `n5_geometry`, `n8_geometry`, `n5_reduced_ratio`,
`n5_reduced_critical_points`, `n8_potential_derivatives`,
`n8_full_potential`, `n8_degenerate_point`, and inflation helpers. The parent
module's `reduced_models.jl` definitions are not identical to the nested
`author_inflation` definitions. `n8_potential(;k=1.0,trajectory=false,phases=nothing)`
dispatches to the trajectory model when requested; full potentials accept
`volume_normalization=:full` or `:fixed`. Invalid positive-scale, branch,
trajectory, phase-length, or normalization inputs raise `ArgumentError`,
`DimensionMismatch`, or solver errors according to the route.

Nested `author_inflation` exposes fixed constants for N=5/N=8 charge matrices,
actions, volumes, vertices, raw metric, critical scale, and best hilltop. Its
key methods are `n8_mass_eigenbasis(k=N8_KC;theta=...)`,
`n8_unstable_direction(k=N8_KC;mode=:most_negative,basis=:canonical_hessian,
basis_theta=...,basis_k=...)`, `n8_inflation_initial_condition` with
displacement/sign/direction/basis controls, `n8_hilltop_probe`,
`n8_slow_roll_trajectory`, and stiff `n8_author_trajectory`. The latter accepts
`precision_bits>=64`, `method=:Rodas5P`, `basis=:canonical_hessian` or
`:mass_eigenbasis`, time/step/sample/maxiters/tolerance controls, and returns
e-fold/slow-roll samples, initial data, precision, solver stats, termination and
end-event fields. Its event/status values include `:tmax`, `:eta_parallel`,
`:epsilon`, and `:no_slow_roll_window`; `entered_slow_roll` and `terminated`
are separate booleans. `n8_slow_roll_trajectory` is the bounded e-fold route and
reports `:tmax`, `:eta_parallel`, `:epsilon`, or `:step_limit`.

`paper_benchmarks/compatibility.jl` aliases the author constants and methods into
the parent namespace, including `poly102_inflation`,
`n8_physical_gradient_flow`, `n8_hilltop_normal_form_efolds`,
`n8_hilltop_normal_form`, `n8_efold_gradient_flow`,
`n5_hilltop_normal_form_efolds`, and `benchmark_efold_targets`. It also adds
`n8_coordinate_maps`, `n8_mass_eigenbasis`, `n8_unstable_direction`, and
`n8_basis_directions`; invalid basis/mode symbols raise `ArgumentError`.

### Fuzzy and catastrophe helpers

The model-stage constants `FUZZY_AXION_MASS_TARGET_EV`,
`FUZZY_AXION_QCD_VOLUME_MIN`, `FUZZY_AXION_QCD_VOLUME_MAX`, and
`FUZZY_AXION_QCD_DIVISOR_DOMAINS` are also module bindings.
`fuzzy_axion_kahler_potential(cy_volume::Real)`,
`fuzzy_axion_prefactor_P(gs::Real)`, `fuzzy_axion_flux_superpotential(w0,
instanton_terms=nothing)`, and `fuzzy_axion_gravitino_mass(prefactor,kahler,
superpotential;mplanck_ev=...)` enforce positive volume/couplings where needed
and return Float64/Complex values. `leading_axion_reference_data(Q,tau,
cy_volume,prefactor_P,gravitino_mass_planck_units,inverse_metric)` returns
reduced Q/tau/reference divisor and log mass; `fuzzy_axion_dilation_root` may
return `nothing` for no positive root. `fuzzy_axion_criterion_one(tau,lambda)`
and `fuzzy_axion_criterion_two(tau_qcd,lambda;volume_min=25,volume_max=40)`
return booleans. `enumerate_fuzzy_axion_models(...;mass_target_ev=...,qcd_volume_min=25,
qcd_volume_max=40,qcd_divisor_domain=:all_prime)` returns a vector of model
NamedTuples; the domain accepts `:all_prime` and `:leading_nonself`.

Catastrophe diagnostics expose constants `CATASTROPHE_BENCHMARK_SCHEMA`,
`PAPER_SOURCE_IDENTITY`, and `VOLUME_SCALING_CONVENTION`; `benchmark_manifest()`
returns replay metadata. `local_catastrophe_diagnostic(theta,Q,amplitudes,metric;
phases=nothing,argument_scale=1,precision_bits=53,tolerance=1e-8,
gradient_tolerance=tolerance,hessian_tolerance=tolerance,
derivative_tolerance=tolerance,null_direction=nothing)` returns classification
`:unresolved`, `:fold`, or `:cusp` plus raw/canonical derivatives, null data,
residuals, precision, and cutoffs. `n5_catastrophe_diagnostic`,
`n8_catastrophe_diagnostic`, `phase_fixture`, `cumulative_turning`, and
`trajectory_observables` add benchmark-specific NamedTuples and error on missing
samples/out-of-range indexes. `catastrophe_diagnostic` and
`classify_catastrophe` are aliases.

## `plotting` and optional renderer

Core data types are exported from `CYAxiverse.plotting`:

* `PlotStyle(;background="#E8E8F0",foreground="#222222",gridcolor="#FFFFFF",
  palette=:viridis,accent_colors=(five defaults),font="STIX",fontsize=20,
  resolution=(900,650),figure_padding=(12,16,12,16))`; resolution/padding
  lengths are validated and fields are cast to String/Float64/Int.
* `curve(x,y;label="")` returns `Curve` and requires equal lengths.
  `band(x,lower,upper;label="",color="#D1495B",alpha=0.22)` returns `Band`,
  requires equal lengths and `0<=alpha<=1`. `reference_line(value;
  orientation=:vertical,label="",color="#222222",linestyle=:dash,linewidth=1.5)`
  validates vertical/horizontal and returns `ReferenceLine`.
* `PlotResult(figure,axis,plot)` always exposes those three fields. Core plotting
  declares (but does not implement) `styled_axis`, `boxplot`, `scatterplot`,
  `exclusionplot`, `functionplot`, `minima_plot`, `trajectoryplot`,
  `save_plot`, and legacy `vacua_db_jlm`, `vacua_db_jlm_single`,
  `total_geometries`, `vacua_db_jlm_box`. Calling renderers before the extension
  is loaded is a method error.

The CairoMakie methods retain broad AbstractVector dispatch. They return
`PlotResult`; passing an existing axis returns `figure=nothing`. `boxplot`
requires nonempty groups and valid orientation/position/label lengths.
`scatterplot` checks equal x/y lengths and supports color vectors/colorbars.
`exclusionplot` accepts `Band` or NamedTuple bands and `ReferenceLine` or
NamedTuple reference lines. `functionplot` accepts x/y, `(Function,x)`, a vector
of `Curve`, or a nonempty tuple of `Curve`; `minima_plot` accepts point indices,
`(x,y)` tuples, or x/y NamedTuples. `trajectoryplot` validates row/time and
label lengths and supports event indices; vector overload delegates to
`functionplot`. `save_plot(path,result)` requires a non-`nothing` figure and
returns the path; reverse argument order and `(path,figure)` overloads exist.
Invalid renderer input raises `ArgumentError`/`DimensionMismatch`.

## `profiling` and conditional `slurm`

`CYAxiverse.profiling` has no exports and is directly consumed by
`notebooks/tree_testing.jl` (the only exact `CYAxiverse.profiling` match found).
Important methods are:

* `LQtilde(Q,L)` accepts the legacy instanton-row shape (`size(Q,1)<size(Q,2)`)
  and returns `LQLinearlyIndependent`; it sorts scales descending and uses
  floating rank tests.
* `αmatrix(LQ;threshold=0.5)` returns `CanonicalQBasis` and applies rounded
  inverse-alpha and scale filtering. `project_out` accepts rational/integer or
  Float64 vectors and returns `Projector`; `omega(Ω::Matrix{Int})` returns
  `ProjectedQ` with sparse perpendicular/parallel fields.
* `vacua(L::Matrix{Float64},Q::Matrix{Int})` returns Dict keys `vacua`, `θ∥`,
  `Qtilde` for h11<50 and only `vacua`, `Qtilde` for larger h11; it prints
  timer/output diagnostics and can fail on rank, determinant, or empty data.
  `minimiser(h11,tri,cy,LV,QV,x0,gradσ,θparalleltest,Qtilde,algo,prec)` uses
  Optim/ArbFloat and returns the signed/log Dict with `eig(H)` and `grad`, or
  `nothing` if its convergence gate fails.

`CYAxiverse.slurm` is included only if `SLURM_JOB_ID` exists at package load.
It defines `writeslurm(id::Int,s::String)` and `writeslurm(id::String,s::String)`
which append to the source's hard-coded `slurmlog/slurm-<id>.out` deployment
destination (the external absolute prefix is intentionally not copied into this
repository-relative artifact).
At load it parses `SLURM_JOB_ID`; with `SLURM_ARRAY_TASK_ID` it creates a
`"jobid_taskid"` string. Missing or non-integer variables fail during module
load. A script that references `CYAxiverse.slurm` without loading CYAxiverse
under a SLURM environment sees an absent binding.

## Repository consumers

The following lists are exact file-level search results, grouped to avoid
mistaking documentation for executable use.

### Direct aliases and internal-looking bindings

* Profiling: `notebooks/tree_testing.jl`.
* `axion_benchmarks`: `scripts/benchmark_inflation_scalability.jl`,
  `scripts/inflation_refinement_common.jl`, `scripts/inflation_reproduction.jl`,
  `scripts/reproduce_2023_n8.jl`.
* `inflation_points`: `scripts/compare_reduction_inflation_scan.jl`,
  `scripts/inflation_candidate_refinement.jl`, `test/runtests.jl`.
* `slurm`: `scripts/Qeff.jl`, `scripts/geometries_hilbert.jl`,
  `scripts/glsm_v_qprime.jl`, `scripts/hilbert.jl`,
  `scripts/hilbert_minima.jl`, `scripts/hilbert_minima_compare.jl`,
  `scripts/optim_with_phases.jl`, `scripts/optim_with_phases_491.jl`,
  `scripts/optimize.jl`, `scripts/optimize_491.jl`, `scripts/spectra.jl`,
  `scripts/top_geom.jl`, `scripts/top_geom_missing_h11.jl`.
* `paper_benchmarks`: `scripts/catastrophic_inflation_population_study.jl`,
  `scripts/fuzzy_axion_model_stage_driver.jl`,
  `scripts/inflation_scale_continuation.jl`,
  `scripts/validate_vacua_stage4_5.jl`, and `test/runtests.jl`.
* `plotting`: `scripts/plot_vacua_jlm.jl`,
  `notebooks/orientifold_axiverse_statistics.jl`, `test/optional_plotting.jl`,
  and `ext/CYAxiverseCairoMakieExt.jl`.

### Module-qualified scripts/notebooks

`generate` is consumed by the spectrum, minima, inflation, vacua, validation,
and persistence workflows: `scripts/analyze_inflation_candidates.jl`,
`scripts/batch_physical_spectrum.jl`, `scripts/benchmark_hybrid_scaling.jl`,
`scripts/benchmark_spectrum_windows.jl`,
`scripts/inflation_candidate_refinement.jl`,
`scripts/inflation_scale_continuation.jl`, `scripts/inflation_scan_common.jl`,
`scripts/migrate_quartic_index_ordering.jl`, `scripts/plot_vacua_jlm.jl`,
`scripts/run_physical_scale_inflation_pilot_20260825.jl`,
`scripts/save_vacua_db.jl`, `scripts/spectra.jl`, `scripts/vacua_pipeline.jl`,
`scripts/vacua_with_phases.jl`, `scripts/validate_classify_point_whitening_accuracy.jl`,
`scripts/validate_vacua_stage4_5.jl`, notebooks `axion_self-interactions.jl`,
`optim_testing.jl`, `optim_testing backup 1.jl`, `tree_testing.jl`,
`vacua_pipeline.jl`, and `test/runtests.jl`.

`read` is consumed by `scripts/analyze_inflation_candidates.jl`,
`batch_physical_spectrum.jl`, `compare_reduction_inflation_scan.jl`,
`geometries_hilbert.jl`, `glsm_v_qprime.jl`, `hilbert_minima_compare.jl`,
`inflation_candidate_refinement.jl`, `inflation_scale_continuation.jl`,
`inflation_scan_common.jl`, `run_physical_scale_inflation_pilot_20260825.jl`,
`scan_relaxed_qcd_candidates.jl`, `vacua_pipeline.jl`,
`validate_vacua_stage4_5.jl`, notebooks `cytools_wrapper_repro.jl`,
`optim_testing.jl`, `optim_testing backup 1.jl`, `orientifold_axiverse_statistics.jl`,
`tree_testing.jl`, `vacua_pipeline.jl`, and `test/runtests.jl`.

`filestructure` is consumed by the full geometry, batch, migration, and
legacy/SWEEP script set, including `scripts/batch_jlm_reduced.jl`,
`batch_physical_spectrum.jl`, `batch_vacua_pipeline.jl`,
`build_orientifold_vacua_inflation.jl`, `inflation_scan_{contract,pilot,prep}.jl`,
`migrate_legacy_spectra.jl`, `migrate_quartic_index_ordering.jl`,
`reproduce_appendix_b_spectra.jl`, `testing/filestructure_db.jl`, all legacy
SLURM runners listed above, notebooks `axion_self-interactions.jl`,
`optim_testing*.jl`, `orientifold_axiverse_statistics.jl`, `tree_testing.jl`,
and tests.

`jlm_reduced` is used by `scripts/analyze_inflation_candidates.jl`,
`scripts/batch_jlm_reduced.jl`, `scripts/compare_reduction_inflation_scan.jl`,
`scripts/scan_relaxed_qcd_candidates.jl`, `scripts/vacua_pipeline.jl`, docs,
and tests. `inflation_points` also has public types directly referenced by
`test/runtests.jl` and is described in `docs/src/api.md`/`docs/src/pipelines.md`.

`axion_photon` is consumed by `scripts/run_axion_photon_scan.jl`, its dedicated
`test/axion_photon.jl`, `docs/src/axion_photon.md`, and API docs. No exact source
call to `CYAxiverse.glimmers.<name>` was found; `glimmers` remains a binding-level
alias that must be preserved.

## Persisted HDF5 schemas and producer/consumer edges

The canonical geometry artifact is `h11_###/np_#######/cy_#######/cyax.h5`.
The established groups/datasets are:

| Path | Producer(s) | Consumer(s)/contract |
|---|---|---|
| `cytools/geometric` | CYTools and geometry scripts | `read.geometry`, `topology`, `visible_sector`, axion-photon loader; fields include h21/glsm/basis, points/simplices, tip/volumes/CY_volume/Kinv, optional hilbert basis, visible-sector policy. |
| `cytools/potential` | CYTools/geometry producers | `read.potential`, `oriented_potential`, `generate`, profiling, axion-photon; `L` and `Q` are signed/log-scale and charge matrices. |
| `cytools/hilbert/geometric`, `cytools/hilbert/potential` | Hilbert scripts | `read.geometry/potential(...;hilbert=true)`, JLM save/compare routes. |
| `paths_cy.h5` (`paths`, `pathinds`) and legacy `paths.h5` | `filestructure.np_path` | `paths_cy`, h11-list consumers; `inKC` selects legacy paths. |
| `spectrum/masses`, `spectrum/decay`, `spectrum/quartdiag`, `spectrum/quart31`, `spectrum/quart22`, `spectrum/cubic` | `generate.pq_spectrum_save`, `hp_spectrum_save`, migrations | `read.pq_spectrum`, `hp_spectrum`, cubic readers; quartic indices are zero-based in generated output. |
| `spectrum/physical` | batch physical spectrum and migration scripts | `read.physical_spectrum`, physical-scale notebooks/scripts; metadata carries units, formulas, thresholds, precision, provisional/convergence state. |
| `vacua`, `vacua_TB` | `generate.vacua_save*` | `read.vacua*`; small h11 stores rational theta numerator/denominator, large h11 omits/rounds theta. |
| `minima.h5` root or `hilbert` | `jlm_reduced.minimize_save`, legacy `jlm_minimizer`, `minimizer.minimize_save` | `read.vacua_jlm`, optimization/minima scripts; root and hilbert fields use `Nvac`, `det_QTilde`, `issquare`, optional `vac_coords`, `extra_rows`. |
| `qshape.h5` (`square`, `vacua_estimate`, optional `extra_rows`, `ωnorm2`) | Q-shape scripts | `read.qshape`, tree/notebook workflows. |
| `vacua_pipeline` | `scripts/vacua_pipeline.jl` and batch runner | `read.pipeline_vacua`; `threshold`, `estimate`, `issquare`, optional `extrarows`/`verified`, rational theta subgroups, rich metadata/status/config digest. |
| `spectrum/axion_photon` | `axion_photon.write_axion_photon_result` | `read_axion_photon_result`, batch skip/identity checks, tests; schema and digest rules are listed in the axion-photon section. |

Additional repository-level artifacts are not consumed by `read` but are
compatibility consumers for scripts:

* `scripts/save_vacua_db.jl` writes `vacua_jlm_db.h5/{square,one_dim,n_dim}`;
  `scripts/plot_vacua_jlm.jl` writes/reads `vacua_db.h5/all_data` and passes
  four-row matrices to the legacy plotting path.
* `scripts/fuzzy_axion_model_stage_driver.jl` reads a root `record_count` and
  per-record input, then writes root counts/model index/lambda/mass/tau arrays
  and `gs`, `w0_real`, `w0_imag`, `prefactor_P`, `qcd_divisor_domain`.
* `scripts/build_orientifold_vacua_inflation.jl` writes geometry-local
  `inflation`, nested `catastrophes`, and `efolds` groups with physical-units,
  scaling/control/viability gate fields and status metadata. These are script
  contracts, not `read` methods.
* `scripts/migrate_legacy_spectra.jl` reads `cytools/spectrum/physical` and
  writes package `spectrum/physical`; migration preserves legacy metadata and
  is not equivalent to recomputation.

## Conditional environment behavior and launch assumptions

`resolve_data_dir` uses explicit `data_dir`, then `CYAXIVERSE_DATA_DIR`, then
`newARGS`, then a checkout sibling. Several legacy notebooks/scripts mutate
`newARGS` (`vacua_0323`, `vacua_stretch`, `vacua_stretchtest`) before loading or
calling package code. Newer scripts set `CYAXIVERSE_DATA_DIR` explicitly and
reject root-like or missing directories.

The `slurm` binding is load-time conditional on `SLURM_JOB_ID`; legacy scripts
also read `SLURM_ARRAY_TASK_ID`, `SLURM_ARRAY_TASK_COUNT`, `SLURM_NPROCS`, and
`MAX_JOB`. An array task changes the slurm log id. A missing variable is an
immediate parse/key error rather than a fallback.

`PyCall` integration is conditional on the extension and then on explicit
`enable_cytools!`; several notebooks set `ENV["PYTHON"]` to a machine-local
CYTools environment. The plotting extension is conditional on both CairoMakie
and ColorSchemes. `Project.toml` alone does not prove the benchmark environment
or optional extension availability.

## Compatibility gaps and P2 handoff risks

1. No package-wide exported API exists beyond the root greeting; root/module
   qualified bindings used by scripts are the real compatibility surface.
2. Potential orientation, term ordering, coefficient factor, and canonicalized
   term boundaries are route-specific. A universal `PotentialDefinition` must
   preserve raw, oriented, canonicalized, PQ-selected, and axion-photon-selected
   boundaries separately.
3. The historical zero-only and all-`-Inf` support behavior needs hostile,
   reproducible fixtures. Modern validators reject many such inputs, while
   lower-level log helpers can propagate `NaN`/`-Inf`; a future common validator
   must not silently change this behavior.
4. `critical_points` uses transformed Float64 equation scales and a physical
   Hessian classification matrix; Optim's solver Hessian/Jacobian, seed
   displacement threshold, and final inertia threshold are distinct observables.
5. Workspace derivative returns borrow mutable arrays; aliasing/copy expectations
   must be explicit before a refactor.
6. ArbFloat/BigFloat precision is process-global in several routes; concurrent
   calls can race and optional extension load state changes dispatch.
7. Legacy HDF5 groups coexist with newer `physical`, `pipeline`, and
   `axion_photon` schemas. Readers often return optional `nothing` fields and
   propagate absent required datasets; writers may skip or replace existing
   groups based on force/configuration policy.
8. `CYAxiverse.slurm` contains a hard-coded external log destination and is not
   available outside a SLURM load environment. This is a deployment constraint,
   not a reason to remove the binding in a compatibility-preserving refactor.
9. The exact consumer search found one direct profiling notebook consumer and no
   exact `glimmers` calls; absence of a textual call is not evidence that users
   do not rely on either binding interactively.
10. This inventory does not add the hostile fixtures or run full package tests;
    those belong to the numerical contract/evidence owner. It records the
    fixture gaps so P2 acceptance cannot assume they are already covered.
