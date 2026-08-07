"""Script-only implementation of the locked inflation scan-prep calls.

This file is intentionally not included by the package.  It keeps the
geometry loading, screening diagnostics, and branch classification sequence
shared by the one-geometry contract probe and the scan-prep driver without
introducing a package-level scan API.
"""

using CYAxiverse
using LinearAlgebra
using Statistics

const GeometryIndex = CYAxiverse.structs.GeometryIndex
const INFLATION_SCAN_CONTRACT_VERSION = "1"

function _timed_call(f)
    GC.gc(false)
    measured = @timed f()
    (; value=measured.value, seconds=measured.time, bytes=measured.bytes)
end

function _oriented_potential(geom_idx::GeometryIndex)
    potential = CYAxiverse.read.potential(geom_idx)
    Q = Matrix{Int}(potential.Q)
    L = Matrix{Float64}(potential.L)
    if size(L, 1) != 2 && size(L, 2) == 2
        L = Matrix(L')
    end
    if size(Q, 2) != size(L, 2) && size(Q, 1) == size(L, 2)
        Q = Matrix(Q')
    end
    size(L, 1) == 2 || throw(DimensionMismatch("L must have two rows"))
    size(Q, 2) == size(L, 2) ||
        throw(DimensionMismatch("Q and L must have the same instanton count"))
    size(Q, 1) == size(potential.K, 1) ||
        throw(DimensionMismatch("Q and K must have the same axion count"))
    size(Q, 2) > size(Q, 1) ||
        throw(DimensionMismatch("Q must contain more instantons than axions"))
    all(isfinite, L) || throw(ArgumentError("L contains non-finite values"))
    K = Hermitian(Matrix{Float64}(potential.K))
    all(isfinite, Matrix(K)) || throw(ArgumentError("K contains non-finite values"))
    Q, L, K
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
    derivatives::CYAxiverse.generate.LogShiftedDerivativeWorkspace
    canonical_hessian::Matrix{Float64}
    inverse_metric_gradient::Vector{Float64}
end

function _classification_workspace(Q::Matrix{Int}, L::Matrix{Float64})
    h11 = size(Q, 1)
    derivatives = CYAxiverse.generate.logshifted_derivative_workspace(Q, L)
    _ClassificationWorkspace(derivatives, zeros(h11, h11), zeros(h11))
end

"""Classify one branch while reusing all geometry- and loop-sized buffers."""
function _classify_point!(workspace::_ClassificationWorkspace,
        theta::AbstractVector{<:Real}, Q::Matrix{Int}, Kfactor)
    derivatives = CYAxiverse.generate.logshifted_derivatives!(
        workspace.derivatives, theta, Q)
    gradient = derivatives.gradient
    hessian = derivatives.hessian
    value = derivatives.value

    canonical_hessian = workspace.canonical_hessian
    copyto!(canonical_hessian, hessian)
    ldiv!(LowerTriangular(Kfactor.L), canonical_hessian)
    rdiv!(canonical_hessian, UpperTriangular(Kfactor.L'))
    eigenvalues = eigvals!(Symmetric(canonical_hessian))

    inverse_metric_gradient = workspace.inverse_metric_gradient
    copyto!(inverse_metric_gradient, gradient)
    ldiv!(LowerTriangular(Kfactor.L), inverse_metric_gradient)
    ldiv!(UpperTriangular(Kfactor.L'), inverse_metric_gradient)
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

function _classify_branches(branches, Q, L, Kfactor)
    saddle_count = 0
    candidate_count = 0
    least_tachyonic = nothing
    best = nothing
    flattest = nothing
    workspace = _classification_workspace(Q, L)
    for index in axes(branches.coordinates, 2)
        classification = _classify_point!(workspace,
            @view(branches.coordinates[:, index]), Q, Kfactor)
        classification.negative_modes > 0 || continue
        classification.value > 0 || continue
        saddle_count += 1
        classification.epsilon < 1 && abs(classification.min_eta) < 1 &&
            (candidate_count += 1)
        least_tachyonic = least_tachyonic === nothing ||
            classification.min_eta > least_tachyonic.min_eta ? classification : least_tachyonic
        best = best === nothing || abs(classification.min_eta + 1) < abs(best.min_eta + 1) ?
            classification : best
        flattest = flattest === nothing || classification.abs_min_eta < flattest.abs_min_eta ?
            classification : flattest
    end
    (; saddle_count, candidate_count, least_tachyonic, best, flattest)
end

"""Run the locked, bounded scan-prep sequence for one geometry."""
function run_geometry(geom_idx::GeometryIndex; max_branches::Int=1_000_000)
    max_branches > 0 || throw(ArgumentError("max_branches must be positive"))
    started = time_ns()
    loaded = _timed_call(() -> _oriented_potential(geom_idx))
    Q, L, K = loaded.value

    selected = _timed_call(() -> CYAxiverse.generate.LQtilde(Q, L))
    hierarchy = _timed_call(() ->
        CYAxiverse.generate.instanton_hierarchy_diagnostics(L))
    factor = _timed_call(() -> cholesky(K))
    mass_basis = _timed_call(() ->
        CYAxiverse.generate.leading_hessian_mass_basis_float64(
            K, selected.value.Ltilde, selected.value.Qtilde))
    branches = _timed_call(() ->
        CYAxiverse.generate.leading_critical_branches(
            selected.value; max_branches))
    classified = _timed_call(() -> _classify_branches(
        branches.value, Q, L, factor.value))

    masses, mass_signs, _ = mass_basis.value
    least_tachyonic = classified.value.least_tachyonic
    best = classified.value.best
    flattest = classified.value.flattest
    (; contract_version=INFLATION_SCAN_CONTRACT_VERSION,
       h11=geom_idx.h11, polytope=geom_idx.polytope, frst=geom_idx.frst,
       status=:success, instantons=size(Q, 2), selected_instantons=size(
           selected.value.Qtilde, 2), qtilde_det=abs(det(Float64.(selected.value.Qtilde))),
       leading_log_gap=hierarchy.value.leading_log_gap,
       log_scale_span=hierarchy.value.log_scale_span,
       strong_hierarchy=hierarchy.value.heuristic_strong_hierarchy,
       branch_count=branches.value.branch_count,
       leading_minima_count=branches.value.leading_minima_count,
       saddle_count=classified.value.saddle_count,
       candidate_slowroll_saddles=classified.value.candidate_count,
       least_tachyonic_min_eta=least_tachyonic === nothing ? NaN : least_tachyonic.min_eta,
       least_tachyonic_epsilon=least_tachyonic === nothing ? NaN : least_tachyonic.epsilon,
       best_min_eta=best === nothing ? NaN : best.min_eta,
       best_epsilon=best === nothing ? NaN : best.epsilon,
       best_abs_min_eta=flattest === nothing ? NaN : flattest.abs_min_eta,
       mass_min=minimum(masses), mass_max=maximum(masses),
       negative_mass_count=count(<(0), mass_signs),
       stage_load_s=loaded.seconds, stage_select_s=selected.seconds,
       stage_hierarchy_s=hierarchy.seconds, stage_factor_s=factor.seconds,
       stage_mass_basis_s=mass_basis.seconds, stage_branches_s=branches.seconds,
       stage_classify_s=classified.seconds,
       stage_allocated_bytes=loaded.bytes + selected.bytes + hierarchy.bytes +
           factor.bytes + mass_basis.bytes + branches.bytes + classified.bytes,
       total_seconds=(time_ns() - started) / 1e9)
end

function _scan_prep_error_status(error)
    message = sprint(showerror, error)
    status = error isa ArgumentError &&
        occursin("leading branch enumeration would create", message) ?
        :branch_cap : :failed
    (; status, message)
end
