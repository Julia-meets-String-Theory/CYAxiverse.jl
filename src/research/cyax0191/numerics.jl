abstract type DifferentiationBackend end

"""Centered finite differences; the step is derived from the input precision."""
struct CentralDifferenceBackend <: DifferentiationBackend end

function _difference_step(::CentralDifferenceBackend, x::T, scale::T, order::Int) where {T<:AbstractFloat}
    characteristic = max(abs(x), abs(scale), one(T))
    characteristic * (order == 1 ? cbrt(eps(T)) : sqrt(sqrt(eps(T))))
end

function gradient!(out::AbstractVector{T}, ::CentralDifferenceBackend, f,
        x::AbstractVector{T}; scales=ones(T, length(x))) where {T<:AbstractFloat}
    length(out) == length(x) == length(scales) || throw(DimensionMismatch("gradient dimensions do not match"))
    xp, xm = copy(x), copy(x)
    for i in eachindex(x)
        h = _difference_step(CentralDifferenceBackend(), x[i], T(scales[i]), 1)
        xp[i], xm[i] = x[i] + h, x[i] - h
        out[i] = (f(xp) - f(xm)) / (T(2) * h)
        xp[i] = xm[i] = x[i]
    end
    out
end

function hessian!(out::AbstractMatrix{T}, ::CentralDifferenceBackend, f,
        x::AbstractVector{T}; scales=ones(T, length(x))) where {T<:AbstractFloat}
    n = length(x)
    size(out) == (n, n) && length(scales) == n || throw(DimensionMismatch("Hessian dimensions do not match"))
    xp, xm = copy(x), copy(x)
    f0 = f(x)
    steps = [_difference_step(CentralDifferenceBackend(), x[i], T(scales[i]), 2) for i in 1:n]
    for i in 1:n
        xp[i], xm[i] = x[i] + steps[i], x[i] - steps[i]
        out[i, i] = (f(xp) - T(2) * f0 + f(xm)) / steps[i]^2
        xp[i] = xm[i] = x[i]
        for j in (i + 1):n
            xp[i], xp[j] = x[i] + steps[i], x[j] + steps[j]
            xm[i], xm[j] = x[i] - steps[i], x[j] - steps[j]
            fp = f(xp)
            fm = f(xm)
            xp[i], xp[j] = x[i] + steps[i], x[j] - steps[j]
            xm[i], xm[j] = x[i] - steps[i], x[j] + steps[j]
            fpm = f(xp)
            fmp = f(xm)
            value = (fp + fm - fpm - fmp) / (T(4) * steps[i] * steps[j])
            out[i, j] = out[j, i] = value
            xp[i] = xm[i] = x[i]
            xp[j] = xm[j] = x[j]
        end
    end
    out
end

function finite_difference_gradient(backend::DifferentiationBackend, f,
        x::AbstractVector{T}; scales=ones(T, length(x))) where {T<:AbstractFloat}
    out = zeros(T, length(x))
    gradient!(out, backend, f, x; scales)
end

function finite_difference_hessian(backend::DifferentiationBackend, f,
        x::AbstractVector{T}; scales=ones(T, length(x))) where {T<:AbstractFloat}
    out = zeros(T, length(x), length(x))
    hessian!(out, backend, f, x; scales)
end

"""Maximum of individually scaled stationarity components."""
function scaled_stationarity_residual(gradient::AbstractVector,
        field_scales::AbstractVector, potential_scale)
    length(gradient) == length(field_scales) || throw(DimensionMismatch("stationarity scales do not match"))
    potential_scale > zero(potential_scale) || throw(ArgumentError("potential scale must be positive"))
    maximum(abs(field_scales[i] * gradient[i] / potential_scale) for i in eachindex(gradient))
end

abstract type SearchBackend end

struct SearchCriteria{T<:AbstractFloat}
    field_scales::Vector{T}
    potential_scale::T
    stationarity_tolerance::T
    max_iterations::Int
    backtracking_factor::T
    minimum_step::T
end

function SearchCriteria(field_scales::AbstractVector{T}, potential_scale::T,
        stationarity_tolerance::T; max_iterations=1_000,
        backtracking_factor=T(0.5), minimum_step=T(2.0)^(-20)) where {T<:AbstractFloat}
    all(>(zero(T)), field_scales) || throw(ArgumentError("field scales must be positive"))
    potential_scale > zero(T) || throw(ArgumentError("potential scale must be positive"))
    stationarity_tolerance > zero(T) || throw(ArgumentError("stationarity tolerance must be positive"))
    0 < backtracking_factor < 1 || throw(ArgumentError("backtracking factor must be between zero and one"))
    minimum_step > zero(T) || throw(ArgumentError("minimum line-search step must be positive"))
    max_iterations > 0 || throw(ArgumentError("iteration budget must be positive"))
    SearchCriteria{T}(collect(field_scales), potential_scale, stationarity_tolerance,
        max_iterations, backtracking_factor, minimum_step)
end

struct DampedNewtonSearch{D<:DifferentiationBackend} <: SearchBackend
    differentiation::D
end

DampedNewtonSearch(; differentiation=CentralDifferenceBackend()) =
    DampedNewtonSearch{typeof(differentiation)}(differentiation)

struct SearchResult{T<:AbstractFloat}
    status::Symbol
    point::Vector{T}
    value::T
    scaled_residual::T
    iterations::Int
    method::String
    failures::Vector{String}
end

function search_stationary(backend::DampedNewtonSearch, f, initial::AbstractVector{T},
        criteria::SearchCriteria{T}; differentiation=backend.differentiation) where {T<:AbstractFloat}
    length(initial) == length(criteria.field_scales) || throw(DimensionMismatch("search scale dimension does not match"))
    x = copy(initial)
    failures = String[]
    g = zeros(T, length(x))
    H = zeros(T, length(x), length(x))
    for iteration in 0:criteria.max_iterations
        gradient!(g, differentiation, f, x; scales=criteria.field_scales)
        residual = T(scaled_stationarity_residual(g, criteria.field_scales, criteria.potential_scale))
        value = T(f(x))
        if residual <= criteria.stationarity_tolerance
            return SearchResult(:converged, x, value, residual, iteration,
                "damped-newton-gradient-residual", failures)
        end
        iteration == criteria.max_iterations && break
        hessian!(H, differentiation, f, x; scales=criteria.field_scales)
        step = try
            -(H \ g)
        catch error
            push!(failures, "Newton solve failed at iteration $iteration: $(typeof(error))")
            return SearchResult(:failed, x, value, residual, iteration,
                "damped-newton-gradient-residual", failures)
        end
        alpha = one(T)
        accepted = false
        trial = similar(x)
        trial_gradient = similar(g)
        while alpha >= criteria.minimum_step
            trial .= x .+ alpha .* step
            gradient!(trial_gradient, differentiation, f, trial; scales=criteria.field_scales)
            trial_residual = T(scaled_stationarity_residual(trial_gradient,
                criteria.field_scales, criteria.potential_scale))
            if isfinite(trial_residual) && trial_residual < residual
                x .= trial
                accepted = true
                break
            end
            alpha *= criteria.backtracking_factor
        end
        if !accepted
            push!(failures, "gradient-residual line search failed at iteration $iteration")
            return SearchResult(:failed, x, value, residual, iteration,
                "damped-newton-gradient-residual", failures)
        end
    end
    gradient!(g, differentiation, f, x; scales=criteria.field_scales)
    residual = T(scaled_stationarity_residual(g, criteria.field_scales, criteria.potential_scale))
    SearchResult(:failed, x, T(f(x)), residual, criteria.max_iterations,
        "damped-newton-gradient-residual", failures)
end

struct Assessment
    status::Symbol
    evidence::String
    function Assessment(status::Symbol, evidence::AbstractString)
        status in (:PASS, :FAIL, :NOT_ASSESSED, :NOT_APPLICABLE) ||
            throw(ArgumentError("assessment status must be PASS, FAIL, NOT_ASSESSED, or NOT_APPLICABLE"))
        new(status, String(evidence))
    end
end

struct CriticalPointState{T<:Real}
    t::Vector{T}
    tau::Vector{T}
    rho::Vector{T}
    geometry_artifact_sha256::String
    axion_elimination::Union{Nothing,NamedTuple}
end

function critical_point_state(geometry::GeometryRecord, t::AbstractVector{T},
        rho::AbstractVector{T}; axion_elimination=nothing) where {T<:Real}
    length(t) == length(rho) == geometry.intersections.n ||
        throw(DimensionMismatch("critical-point coordinates must match geometry dimension"))
    tau = divisor_volumes(geometry, t)
    R = promote_type(eltype(t), eltype(rho), eltype(tau))
    CriticalPointState(R.(t), R.(tau), R.(rho), geometry.artifact_sha256,
        axion_elimination)
end

struct ModeDisposition{T<:Real}
    eigenvalue::T
    disposition::Symbol
    sign_status::Symbol
end

struct FluctuationReport{T<:Real}
    coordinate_hessian::Matrix{T}
    covariant_hessian::Matrix{T}
    retained_kinetic_metric::Matrix{T}
    generalized_mass_eigenvalues::Vector{T}
    mode_dispositions::Vector{ModeDisposition{T}}
    symmetry_protected_directions::Matrix{Rational{BigInt}}
    physical_mass_assessment::Assessment
    bf_assessment::Assessment
end

abstract type FluctuationBackend end
struct GeneralizedEigenBackend <: FluctuationBackend end

function fluctuation_analysis(::GeneralizedEigenBackend, hessian::AbstractMatrix{T},
        kinetic_metric::AbstractMatrix{T}; at_critical_point=true,
        covariant_hessian=nothing, exact_zero_mode_indices=Int[],
        absolute_zero_threshold=T(1e-12), relative_zero_threshold=T(1e-10),
        active_charges=nothing, ads_radius_squared=nothing) where {T<:AbstractFloat}
    size(hessian) == size(kinetic_metric) || throw(DimensionMismatch("Hessian and kinetic metric dimensions differ"))
    size(hessian, 1) == size(hessian, 2) || throw(DimensionMismatch("fluctuation matrices must be square"))
    Hcoord = Matrix(Symmetric((hessian + hessian') / T(2)))
    Hcov = if at_critical_point
        Hcoord
    elseif covariant_hessian !== nothing
        Matrix(Symmetric((covariant_hessian + covariant_hessian') / T(2)))
    else
        throw(ArgumentError("a covariant Hessian is required away from an exact critical point"))
    end
    G = Matrix(Symmetric((kinetic_metric + kinetic_metric') / T(2)))
    isposdef(Symmetric(G)) || throw(DomainError(G, "retained kinetic metric must be positive definite"))
    L = cholesky(Symmetric(G)).L
    normalized = L \ Hcov / L'
    values = eigvals(Symmetric((normalized + normalized') / T(2)))
    scale = max(maximum(abs, values), one(T))
    near_limit = max(absolute_zero_threshold, relative_zero_threshold * scale)
    dispositions = ModeDisposition{T}[]
    exact = Set(Int.(exact_zero_mode_indices))
    all(i -> 1 <= i <= length(values), exact) || throw(BoundsError(values, collect(exact)))
    symmetry_directions = if active_charges === nothing
        zeros(Rational{BigInt}, size(G, 1), 0)
    else
        size(G, 1) == 2 * size(active_charges, 2) ||
            throw(DimensionMismatch("active charges need one retained axion block matching the kinetic metric"))
        kernel = exact_axionic_shift_basis(active_charges)
        directions = zeros(Rational{BigInt}, size(G, 1), size(kernel, 2))
        directions[(size(G, 1) ÷ 2 + 1):end, :] .= kernel
        directions
    end
    for i in eachindex(values)
        disposition = if i in exact && abs(values[i]) <= near_limit
            :symmetry_protected_exact_zero
        elseif abs(values[i]) <= near_limit
            :numerically_unresolved_near_zero
        elseif values[i] < zero(T)
            :tachyonic
        else
            :lifted
        end
        sign_status = values[i] < -near_limit ? :negative :
            (values[i] > near_limit ? :positive : :zero_within_threshold)
        push!(dispositions, ModeDisposition(values[i], disposition, sign_status))
    end
    bf = if ads_radius_squared === nothing
        Assessment(:NOT_APPLICABLE, "no AdS radius was supplied")
    else
        ads_radius_squared > zero(T) || throw(ArgumentError("AdS radius squared must be positive"))
        all(value -> value * ads_radius_squared >= -T(9) / T(4), values) ?
            Assessment(:PASS, "all retained generalized masses satisfy m^2 L_AdS^2 >= -9/4") :
            Assessment(:FAIL, "at least one retained generalized mass violates the AdS4 BF bound")
    end
    FluctuationReport(Hcoord, Hcov, G, collect(values), dispositions,
        symmetry_directions,
        Assessment(:PASS, "generalized eigenproblem H v = m^2 G v was solved"), bf)
end

"""Exact rational basis for the axionic shifts in ker(Q_active)."""
function exact_axionic_shift_basis(charges::AbstractMatrix{<:Integer})
    nrows, ncols = size(charges)
    A = Rational{BigInt}.(charges)
    pivot_columns = Int[]
    pivot_row = 1
    for column in 1:ncols
        pivot_row > nrows && break
        pivot = pivot_row
        while pivot <= nrows && iszero(A[pivot, column])
            pivot += 1
        end
        pivot > nrows && continue
        if pivot != pivot_row
            A[pivot_row, :], A[pivot, :] = copy(A[pivot, :]), copy(A[pivot_row, :])
        end
        A[pivot_row, :] ./= A[pivot_row, column]
        for row in 1:nrows
            row == pivot_row && continue
            factor = A[row, column]
            iszero(factor) || (A[row, :] .-= factor .* A[pivot_row, :])
        end
        push!(pivot_columns, column)
        pivot_row += 1
    end
    free_columns = setdiff(collect(1:ncols), pivot_columns)
    basis = zeros(Rational{BigInt}, ncols, length(free_columns))
    for (basis_column, free_column) in enumerate(free_columns)
        basis[free_column, basis_column] = 1 // 1
        for (row, pivot_column) in enumerate(pivot_columns)
            basis[pivot_column, basis_column] = -A[row, free_column]
        end
    end
    basis
end

"""Separate source, numerical, physical-mass, and control assessments."""
struct ControlReport
    stationarity::Assessment
    parent_metric_domain::Assessment
    retained_metric_domain::Assessment
    imported_domain::Assessment
    weak_coupling::Assessment
    alpha_prime_control::Assessment
    retained_exponential_control::Assessment
    omitted_instantons::Assessment
    loops_and_higher_derivatives::Assessment
    scale_hierarchy::Assessment
    heavy_sector_stability::Assessment
    global_compactification_consistency::Assessment
end

function unassessed_controls()
    unknown = Assessment(:NOT_ASSESSED, "not evaluated in the authorized Phase-0 tranche")
    ControlReport(unknown, unknown, unknown, unknown, unknown, unknown, unknown,
        unknown, unknown, unknown, unknown, unknown)
end

struct ResearchResult{T<:Real}
    critical_point::CriticalPointState{T}
    potential::T
    source_reported_observables::NamedTuple
    equation_derived_observables::NamedTuple
    fluctuation::Union{Nothing,FluctuationReport{T}}
    controls::ControlReport
end

abstract type ReplayBackend end
struct NativeReplayBackend <: ReplayBackend end

function replay_manifest(::NativeReplayBackend, model::KahlerModel,
        geometry::GeometryRecord; code_revision::AbstractString,
        selected_source_route::AbstractString, counting_unit::AbstractString,
        numeric_type::AbstractString, precision_bits::Integer,
        backend_versions::NamedTuple, solver_configuration::NamedTuple,
        scales::NamedTuple, seed=nothing, budget=nothing)
    all(!isempty, (code_revision, selected_source_route, counting_unit,
        numeric_type)) || throw(ArgumentError("replay identity strings must be explicit"))
    precision_bits > 0 || throw(ArgumentError("precision_bits must be positive"))
    uplift_identity = model.uplift === nothing ? nothing :
        (; identity=model.uplift.identity, provenance=model.uplift.provenance)
    model_parameters = (; w0_magnitude=string(model.w0_magnitude),
        theta0=string(model.theta0), gs=string(model.gs), kcs=string(model.kcs),
        amplitudes=Tuple(string.(model.amplitudes)),
        actions=Tuple(string.(model.actions)), phases=Tuple(string.(model.phases)),
        charges=Tuple(Tuple(row) for row in eachrow(model.charges)),
        uplift=uplift_identity)
    (; schema_version="cyax0191-replay-v1",
       code_revision=String(code_revision),
       source=geometry_identity(geometry),
       geometry_artifact_sha256=geometry.artifact_sha256,
       model_identity=model.identity,
       model_parameters,
       model_convention=model.convention,
       model_switches=model.switches,
       selected_source_route=String(selected_source_route),
       counting_unit=String(counting_unit),
       units=geometry.units, numeric_type=String(numeric_type),
       precision_bits=Int(precision_bits), backend_versions,
       solver_configuration, scales, seed, budget)
end

"""Rebase an integer charge matrix under D_new = B*D_old."""
function change_charge_basis(charges::AbstractMatrix{<:Integer}, B::AbstractMatrix{<:Integer})
    size(charges, 2) == size(B, 1) == size(B, 2) ||
        throw(DimensionMismatch("charge matrix and basis transform dimensions differ"))
    Matrix{Int}(charges) * _integer_inverse(B)'
end

function change_model_basis(model::KahlerModel, B::AbstractMatrix{<:Integer})
    Qnew = change_charge_basis(model.charges, B)
    KahlerModel(; w0_magnitude=model.w0_magnitude, theta0=model.theta0,
        gs=model.gs, kcs=model.kcs, amplitudes=model.amplitudes,
        actions=model.actions, phases=model.phases, charges=Qnew,
        convention=model.convention, switches=model.switches,
        uplift=model.uplift, identity="$(model.identity)|basis=$(repr(Matrix{Int}(B)))")
end

function change_coordinate_basis(t::AbstractVector, rho::AbstractVector,
        B::AbstractMatrix{<:Integer})
    size(B, 1) == size(B, 2) == length(t) == length(rho) ||
        throw(DimensionMismatch("coordinate and basis dimensions differ"))
    Binv = _integer_inverse(B)
    (Binv * t, B' * rho)
end
