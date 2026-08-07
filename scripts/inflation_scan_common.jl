"""Script-only implementation of the locked inflation scan-prep calls.

This file is intentionally not included by the package.  It keeps the
geometry loading, screening diagnostics, and branch classification sequence
shared by the one-geometry contract probe and the scan-prep driver without
introducing a package-level scan API.
"""

if !isdefined(@__MODULE__, :INFLATION_DIAGNOSTIC_SCHEMA_VERSION)
    include(joinpath(@__DIR__, "inflation_diagnostics_common.jl"))
end

using CYAxiverse
using LinearAlgebra
using Statistics

const GeometryIndex = CYAxiverse.structs.GeometryIndex
const INFLATION_SCAN_CONTRACT_VERSION = "4"

"""Parse `K` or `K:L` into an inclusive leading-index range."""
function inflation_parse_negative_mode_range(value::AbstractString)
    text = strip(value)
    lowercase(text) == "all" && return nothing
    parts = split(text, ':')
    length(parts) in (1, 2) ||
        throw(ArgumentError("negative-mode range must be K, K:K, or all"))
    first_index = parse(Int, parts[1])
    last_index = length(parts) == 1 ? first_index : parse(Int, parts[2])
    first_index:last_index
end

"""Use a stable CSV/resume label for a leading-index search range."""
inflation_negative_mode_range_label(range::Union{Nothing,UnitRange{Int}}) =
    range === nothing ? "all" : string(first(range), ":", last(range))

"""Script-level resource and refinement policy for bounded screening."""
const INFLATION_SCREENING_POLICY = (
    normal_h11_max=50,
    middle_h11_max=100,
    high_memory_h11_min=101,
    max_stage_allocated_bytes=750_000_000,
    max_stage_output_bytes=300_000_000,
)

function inflation_screening_tier(h11::Integer;
        policy=INFLATION_SCREENING_POLICY)
    h11 > 0 || throw(ArgumentError("h11 must be positive"))
    h11 <= policy.normal_h11_max ? :normal :
        h11 <= policy.middle_h11_max && h11 < policy.high_memory_h11_min ?
        :middle : :high_memory_queue
end

"""Exact lower bound on the half-integer branch multiplicity."""
inflation_branch_estimate_lower_bound(h11::Integer) = BigInt(2)^h11

"""Return whether a completed screen is eligible for Stage 3 refinement."""
function inflation_refinement_eligible(summary;
        policy=INFLATION_SCREENING_POLICY)
    status = hasproperty(summary, :status) ? summary.status : :failed
    candidates = hasproperty(summary, :candidate_slowroll_saddles) ?
        summary.candidate_slowroll_saddles :
        (hasproperty(summary, :candidate_count) ? summary.candidate_count : 0)
    allocated = hasproperty(summary, :stage_allocated_bytes) ?
        summary.stage_allocated_bytes :
        (hasproperty(summary, :allocated_bytes) ? summary.allocated_bytes : 0)
    output = hasproperty(summary, :stage_output_bytes) ?
        summary.stage_output_bytes :
        (hasproperty(summary, :output_bytes) ? summary.output_bytes : 0)
    h11 = hasproperty(summary, :h11) ? summary.h11 : typemax(Int)
    status == :success && candidates > 0 &&
        inflation_screening_tier(h11; policy) != :high_memory_queue &&
        allocated <= policy.max_stage_allocated_bytes &&
        output <= policy.max_stage_output_bytes
end

function _timed_call(f; measurement_scope::Symbol=:unspecified)
    inflation_stage_measure(f; measurement_scope)
end

"""Reusable workspace for the LAPACK symmetric eigensolver used by screening.

`LinearAlgebra.LAPACK.syevd!` allocates its work arrays on every call.  A
screening pass can call the eigensolver once per branch, so keep the arrays and
the scalar arguments at the scan-workspace boundary instead.  DSYEVD destroys
the input matrix in the same way as the public wrapper; callers therefore pass
the already disposable canonical Hessian buffer.
"""
mutable struct _SymmetricEigenWorkspace
    eigenvalues::Vector{Float64}
    work::Vector{Float64}
    iwork::Vector{LinearAlgebra.BlasInt}
    jobz::Ref{UInt8}
    uplo::Ref{UInt8}
    n::Ref{LinearAlgebra.BlasInt}
    lda::Ref{LinearAlgebra.BlasInt}
    lwork::Ref{LinearAlgebra.BlasInt}
    liwork::Ref{LinearAlgebra.BlasInt}
    info::Ref{LinearAlgebra.BlasInt}
end

function _dsyevd_call!(jobz::Ref{UInt8}, uplo::Ref{UInt8},
        n::Ref{LinearAlgebra.BlasInt}, matrix::AbstractArray{Float64},
        lda::Ref{LinearAlgebra.BlasInt}, eigenvalues::Vector{Float64},
        work::AbstractVector{Float64}, lwork::Ref{LinearAlgebra.BlasInt},
        iwork::AbstractVector{LinearAlgebra.BlasInt},
        liwork::Ref{LinearAlgebra.BlasInt}, info::Ref{LinearAlgebra.BlasInt})
    ccall((LinearAlgebra.BLAS.@blasfunc(dsyevd_),
            LinearAlgebra.libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64},
         Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ptr{Float64},
         Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt},
         Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Clong, Clong),
        jobz, uplo, n, matrix, lda, eigenvalues, work, lwork, iwork,
        liwork, info, Clong(1), Clong(1))
    nothing
end

function _symmetric_eigen_workspace(n::Integer)
    n > 0 || throw(ArgumentError("symmetric eigensolver dimension must be positive"))
    dimension = LinearAlgebra.BlasInt(n)
    query_matrix = Vector{Float64}(undef, n * n)
    query_eigenvalues = Vector{Float64}(undef, n)
    query_work = Vector{Float64}(undef, 1)
    query_iwork = Vector{LinearAlgebra.BlasInt}(undef, 1)
    query_lwork = Ref{LinearAlgebra.BlasInt}(-1)
    query_liwork = Ref{LinearAlgebra.BlasInt}(-1)
    query_info = Ref{LinearAlgebra.BlasInt}(0)
    _dsyevd_call!(Ref(UInt8('N')), Ref(UInt8('U')), Ref(dimension),
        query_matrix, Ref(dimension), query_eigenvalues, query_work,
        query_lwork, query_iwork, query_liwork, query_info)
    query_info[] == 0 ||
        throw(ErrorException("DSYEVD workspace query failed with INFO=$(query_info[])"))
    lwork = max(1, Int(round(query_work[1])))
    liwork = max(1, Int(query_iwork[1]))
    _SymmetricEigenWorkspace(
        zeros(Float64, n),
        zeros(Float64, lwork),
        zeros(LinearAlgebra.BlasInt, liwork),
        Ref(UInt8('N')), Ref(UInt8('U')), Ref(dimension), Ref(dimension),
        Ref(LinearAlgebra.BlasInt(lwork)), Ref(LinearAlgebra.BlasInt(liwork)),
        Ref(LinearAlgebra.BlasInt(0)))
end

function _symmetric_eigenvalues!(workspace::_SymmetricEigenWorkspace,
        matrix::StridedMatrix{Float64})
    size(matrix, 1) == size(matrix, 2) == length(workspace.eigenvalues) ||
        throw(DimensionMismatch("matrix dimension does not match eigensolver workspace"))
    workspace.info[] = 0
    _dsyevd_call!(workspace.jobz, workspace.uplo, workspace.n, matrix,
        workspace.lda, workspace.eigenvalues, workspace.work, workspace.lwork,
        workspace.iwork, workspace.liwork, workspace.info)
    workspace.info[] == 0 ||
        throw(ErrorException("DSYEVD failed with INFO=$(workspace.info[])"))
    workspace.eigenvalues
end

function _normalized_derivatives(theta::AbstractVector{<:Real},
        Q::Matrix{Int}, L::Matrix{Float64})
    workspace = CYAxiverse.generate.logshifted_derivative_workspace(Q, L)
    derivatives = CYAxiverse.generate.logshifted_derivatives!(
        workspace, theta, Q)
    (; value=derivatives.value, gradient=derivatives.gradient,
       hessian=derivatives.hessian, amplitudes=workspace.amplitudes,
       arguments=2π .* (Q' * theta), log_shift=derivatives.log_shift)
end

function _classify_point(theta, Q, L, Kfactor)
    derivatives = _normalized_derivatives(theta, Q, L)
    canonical_hessian = Kfactor.L \ derivatives.hessian / Kfactor.L'
    eigenvalues = eigvals(Symmetric(canonical_hessian))
    inverse_metric_gradient = Kfactor.L' \ (Kfactor.L \ derivatives.gradient)
    gradient_norm = sqrt(max(dot(derivatives.gradient, inverse_metric_gradient), 0.0))
    value = derivatives.value
    epsilon = value == 0 ? Inf : 0.5 * (gradient_norm / abs(value))^2
    eta_values = value == 0 ? fill(Inf, length(eigenvalues)) : eigenvalues ./ value
    scale = max(maximum(abs, eigenvalues), 1.0)
    (; value, gradient_norm, epsilon,
       min_eta=minimum(eta_values), max_eta=maximum(eta_values),
       abs_min_eta=minimum(abs.(eta_values)),
       negative_modes=count(<(0), eigenvalues),
       zeroish_modes=count(x -> abs(x) <= 1e-10 * scale, eigenvalues),
       positive_modes=count(>(0), eigenvalues))
end

mutable struct _ClassificationWorkspace
    derivative_evaluator::CYAxiverse.generate.StructuredChargeEvaluator
    canonical_hessian::Matrix{Float64}
    inverse_metric_gradient::Vector{Float64}
    factor_lower::Matrix{Float64}
    eigen_workspace::_SymmetricEigenWorkspace
end

function _classification_workspace(Q::Matrix{Int}, L::Matrix{Float64}, Kfactor)
    h11 = size(Q, 1)
    derivative_evaluator = CYAxiverse.generate.structured_charge_evaluator(Q, L)
    # `cholesky` defaults to an upper-factor representation; materialize the
    # lower view once so both BLAS calls below see the same factor as the
    # previous per-branch `parent(Kfactor.L)` path.
    factor_lower = Matrix(Kfactor.L)
    _ClassificationWorkspace(derivative_evaluator, zeros(h11, h11), zeros(h11),
        factor_lower, _symmetric_eigen_workspace(h11))
end

"""Classify one branch while reusing all geometry- and loop-sized buffers."""
function _classify_point!(workspace::_ClassificationWorkspace,
        theta::AbstractVector{<:Real}, Q::Matrix{Int}, Kfactor)
    derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
        workspace.derivative_evaluator, theta, Q)
    gradient = derivatives.gradient
    hessian = derivatives.hessian
    value = derivatives.value

    canonical_hessian = workspace.canonical_hessian
    copyto!(canonical_hessian, hessian)
    factor_lower = workspace.factor_lower
    LinearAlgebra.BLAS.trsm!('L', 'L', 'N', 'N', 1.0,
        factor_lower, canonical_hessian)
    LinearAlgebra.BLAS.trsm!('R', 'L', 'T', 'N', 1.0,
        factor_lower, canonical_hessian)
    eigenvalues = _symmetric_eigenvalues!(workspace.eigen_workspace,
        canonical_hessian)

    inverse_metric_gradient = workspace.inverse_metric_gradient
    copyto!(inverse_metric_gradient, gradient)
    LinearAlgebra.BLAS.trsv!('L', 'N', 'N', factor_lower, inverse_metric_gradient)
    LinearAlgebra.BLAS.trsv!('L', 'T', 'N', factor_lower, inverse_metric_gradient)
    gradient_norm = sqrt(max(dot(gradient, inverse_metric_gradient), 0.0))
    epsilon = value == 0 ? Inf : 0.5 * (gradient_norm / abs(value))^2
    if value == 0
        min_eta = max_eta = abs_min_eta = Inf
    else
        min_eta = Inf
        max_eta = -Inf
        abs_min_eta = Inf
        for eigenvalue in eigenvalues
            eta = eigenvalue / value
            min_eta = min(min_eta, eta)
            max_eta = max(max_eta, eta)
            abs_min_eta = min(abs_min_eta, abs(eta))
        end
    end
    scale = max(maximum(abs, eigenvalues), 1.0)
    (; value, gradient_norm, epsilon, min_eta, max_eta, abs_min_eta,
       negative_modes=count(<(0), eigenvalues),
       zeroish_modes=count(x -> abs(x) <= 1e-10 * scale, eigenvalues),
       positive_modes=count(>(0), eigenvalues))
end

mutable struct _ClassificationAccumulator
    saddle_count::Int
    candidate_count::Int
    least_tachyonic::Any
    best::Any
    flattest::Any
end

_ClassificationAccumulator() = _ClassificationAccumulator(0, 0, nothing, nothing, nothing)

function _record_classification!(accumulator::_ClassificationAccumulator, classification)
    classification.negative_modes > 0 || return nothing
    classification.value > 0 || return nothing
    accumulator.saddle_count += 1
    classification.epsilon < 1 && abs(classification.min_eta) < 1 &&
        (accumulator.candidate_count += 1)
    accumulator.least_tachyonic = accumulator.least_tachyonic === nothing ||
        classification.min_eta > accumulator.least_tachyonic.min_eta ?
        classification : accumulator.least_tachyonic
    accumulator.best = accumulator.best === nothing ||
        abs(classification.min_eta + 1) < abs(accumulator.best.min_eta + 1) ?
        classification : accumulator.best
    accumulator.flattest = accumulator.flattest === nothing ||
        classification.abs_min_eta < accumulator.flattest.abs_min_eta ?
        classification : accumulator.flattest
    nothing
end

function _finish_classification(accumulator::_ClassificationAccumulator)
    (; saddle_count=accumulator.saddle_count,
       candidate_count=accumulator.candidate_count,
       least_tachyonic=accumulator.least_tachyonic,
       best=accumulator.best, flattest=accumulator.flattest)
end

function _leading_branch_det_qtilde(branch_count::Int, qtilde_dimension::Int)
    qtilde_dimension >= 0 ||
        throw(ArgumentError("qtilde_dimension must be nonnegative"))
    branch_count ÷ (BigInt(2)^qtilde_dimension)
end

function _classify_branches(branches, Q, L, Kfactor)
    accumulator = _ClassificationAccumulator()
    workspace = _classification_workspace(Q, L, Kfactor)
    for index in axes(branches.coordinates, 2)
        classification = _classify_point!(workspace,
            @view(branches.coordinates[:, index]), Q, Kfactor)
        _record_classification!(accumulator, classification)
    end
    _finish_classification(accumulator)
end

function _classify_leading_branches(selected, Q, L, Kfactor;
        max_branches::Int, negative_mode_range::Union{Nothing,UnitRange{Int}}=nothing,
        max_negative_modes::Union{Nothing,Int}=nothing)
    accumulator = _ClassificationAccumulator()
    workspace = _classification_workspace(Q, L, Kfactor)
    branch_count = 0
    leading_minima_count = 0
    stream_report = CYAxiverse.generate.foreach_leading_critical_branch(
            selected; max_branches, negative_mode_range, max_negative_modes) do theta, leading_negative_modes
        branch_count += 1
        leading_negative_modes == 0 && (leading_minima_count += 1)
        classification = _classify_point!(workspace, theta, Q, Kfactor)
        _record_classification!(accumulator, classification)
    end
    representation = workspace.derivative_evaluator.representation
    (; classification=_finish_classification(accumulator), branch_count,
       leading_minima_count, stream_report,
       structured_charge_validated=representation.validated,
       structured_fallback_reason=representation.fallback_reason,
       det_Qtilde=stream_report.lattice_copy_count)
end

"""Run the locked, bounded scan-prep sequence for one geometry."""
function run_geometry(geom_idx::GeometryIndex; max_branches::Int=1_000_000,
        measurement_scope::Symbol=:unspecified,
        negative_mode_range::Union{Nothing,UnitRange{Int}}=nothing,
        max_negative_modes::Union{Nothing,Int}=nothing)
    max_branches > 0 || throw(ArgumentError("max_branches must be positive"))
    started = time_ns()
    loaded = _timed_call(() -> CYAxiverse.read.oriented_potential(geom_idx);
        measurement_scope)
    Q, L, K = loaded.value.Q, loaded.value.L, loaded.value.K

    selected = _timed_call(() -> CYAxiverse.generate.LQtilde(Q, L);
        measurement_scope)
    hierarchy = _timed_call(() ->
        CYAxiverse.generate.instanton_hierarchy_diagnostics(L); measurement_scope)
    factor = _timed_call(() -> cholesky(K); measurement_scope)
    mass_basis = _timed_call(() ->
        CYAxiverse.generate.leading_hessian_mass_basis_float64(
            K, selected.value.Ltilde, selected.value.Qtilde); measurement_scope)
    classified = _timed_call(() -> _classify_leading_branches(
        selected.value, Q, L, factor.value; max_branches,
        negative_mode_range, max_negative_modes); measurement_scope)

    masses, mass_signs, _ = mass_basis.value
    branch_classification = classified.value.classification
    least_tachyonic = branch_classification.least_tachyonic
    best = branch_classification.best
    flattest = branch_classification.flattest
    (; contract_version=INFLATION_SCAN_CONTRACT_VERSION,
       diagnostic_schema_version=INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
       measurement_scope=classified.measurement_scope,
       h11=geom_idx.h11, polytope=geom_idx.polytope, frst=geom_idx.frst,
       status=:success, instantons=size(Q, 2), selected_instantons=size(
           selected.value.Qtilde, 2), qtilde_det=classified.value.det_Qtilde,
       leading_log_gap=hierarchy.value.leading_log_gap,
       log_scale_span=hierarchy.value.log_scale_span,
       strong_hierarchy=hierarchy.value.heuristic_strong_hierarchy,
       branch_count=classified.value.branch_count,
       negative_mode_range=inflation_negative_mode_range_label(
           classified.value.stream_report.negative_mode_range),
       structured_charge_validated=classified.value.structured_charge_validated,
       structured_fallback_reason=classified.value.structured_fallback_reason,
       search_classification=classified.value.stream_report.search_classification,
       mask_count=classified.value.stream_report.mask_count,
       masks_visited=classified.value.stream_report.masks_visited,
       masks_skipped=classified.value.stream_report.masks_skipped,
       lattice_copy_count=classified.value.stream_report.lattice_copy_count,
       lattice_copies_visited=classified.value.stream_report.lattice_copies_visited,
       leading_minima_count=classified.value.leading_minima_count,
       saddle_count=branch_classification.saddle_count,
       candidate_slowroll_saddles=branch_classification.candidate_count,
       least_tachyonic_min_eta=least_tachyonic === nothing ? NaN : least_tachyonic.min_eta,
       least_tachyonic_epsilon=least_tachyonic === nothing ? NaN : least_tachyonic.epsilon,
       best_min_eta=best === nothing ? NaN : best.min_eta,
       best_epsilon=best === nothing ? NaN : best.epsilon,
       best_abs_min_eta=flattest === nothing ? NaN : flattest.abs_min_eta,
       mass_min=minimum(masses), mass_max=maximum(masses),
       negative_mass_count=count(<(0), mass_signs),
       stage_load_s=loaded.seconds, stage_select_s=selected.seconds,
       stage_hierarchy_s=hierarchy.seconds, stage_factor_s=factor.seconds,
       stage_mass_basis_s=mass_basis.seconds, stage_branches_s=0.0,
       stage_classify_s=classified.seconds,
       stage_load_output_bytes=loaded.output_bytes,
       stage_select_output_bytes=selected.output_bytes,
       stage_hierarchy_output_bytes=hierarchy.output_bytes,
       stage_factor_output_bytes=factor.output_bytes,
       stage_mass_basis_output_bytes=mass_basis.output_bytes,
       stage_classify_output_bytes=classified.output_bytes,
       stage_allocated_bytes=loaded.bytes + selected.bytes + hierarchy.bytes +
           factor.bytes + mass_basis.bytes + classified.bytes,
       stage_output_bytes=loaded.output_bytes + selected.output_bytes +
           hierarchy.output_bytes + factor.output_bytes + mass_basis.output_bytes +
           classified.output_bytes,
       total_seconds=(time_ns() - started) / 1e9)
end

function _scan_prep_error_status(error)
    message = sprint(showerror, error)
    status = error isa ArgumentError &&
        occursin("leading branch enumeration would create", message) ?
        :branch_cap : :failed
    (; status, message)
end
