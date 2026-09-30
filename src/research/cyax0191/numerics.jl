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
    isfinite(potential_scale) && potential_scale > zero(potential_scale) ||
        throw(ArgumentError("potential scale must be finite and positive"))
    maximum(abs(field_scales[i] * gradient[i] / potential_scale) for i in eachindex(gradient))
end

abstract type SearchBackend end

struct SearchCriteria{T<:AbstractFloat}
    field_scales::FrozenArray{T,1}
    potential_scale::FrozenScalar{T}
    stationarity_tolerance::FrozenScalar{T}
    max_iterations::Int
    backtracking_factor::FrozenScalar{T}
    minimum_step::FrozenScalar{T}
    potential_scale_rule::Any
    potential_scale_rule_identity::String

    function SearchCriteria{T}(field_scales::AbstractVector{T}, potential_scale::T,
            stationarity_tolerance::T, max_iterations::Int, backtracking_factor::T,
            minimum_step::T, potential_scale_rule,
            potential_scale_rule_identity::String) where {T<:AbstractFloat}
        !isempty(field_scales) || throw(ArgumentError("at least one field scale is required"))
        all(value -> isfinite(value) && value > zero(T), field_scales) ||
            throw(ArgumentError("field scales must be finite and positive"))
        isfinite(potential_scale) && potential_scale > zero(T) ||
            throw(ArgumentError("potential scale must be finite and positive"))
        isfinite(stationarity_tolerance) && stationarity_tolerance > zero(T) ||
            throw(ArgumentError("stationarity tolerance must be finite and positive"))
        isfinite(backtracking_factor) && 0 < backtracking_factor < 1 ||
            throw(ArgumentError("backtracking factor must be finite and between zero and one"))
        isfinite(minimum_step) && minimum_step > zero(T) ||
            throw(ArgumentError("minimum line-search step must be finite and positive"))
        max_iterations > 0 || throw(ArgumentError("iteration budget must be positive"))
        (potential_scale_rule === nothing) == isempty(potential_scale_rule_identity) ||
            throw(ArgumentError("a pointwise potential scale rule and its identity must be supplied together"))
        potential_scale_rule === nothing || _manifest_value_is_immutable(potential_scale_rule) ||
            throw(ArgumentError("pointwise scale rules must not retain mutable state"))
        scales = FrozenArray(collect(field_scales))
        new{T}(scales, FrozenScalar(potential_scale),
            FrozenScalar(stationarity_tolerance), max_iterations,
            FrozenScalar(backtracking_factor), FrozenScalar(minimum_step),
            potential_scale_rule, potential_scale_rule_identity)
    end
end

function Base.getproperty(criteria::SearchCriteria, name::Symbol)
    value = getfield(criteria, name)
    value isa FrozenScalar && return _thaw_scalar(value)
    value
end

function SearchCriteria(field_scales::AbstractVector{T}, potential_scale::T,
        stationarity_tolerance::T; max_iterations=1_000,
        backtracking_factor=T(0.5), minimum_step=T(2.0)^(-20),
        potential_scale_rule=nothing, potential_scale_rule_identity="") where {T<:AbstractFloat}
    SearchCriteria{T}(collect(field_scales), potential_scale, stationarity_tolerance,
        Int(max_iterations), T(backtracking_factor), T(minimum_step), potential_scale_rule,
        String(potential_scale_rule_identity))
end

function _search_potential_scale(criteria::SearchCriteria{T}, x::AbstractVector) where {T}
    scale = criteria.potential_scale_rule === nothing ? criteria.potential_scale :
        criteria.potential_scale_rule(x)
    value = T(scale)
    isfinite(value) && value > zero(T) ||
        throw(DomainError(value, "the search potential scale must be finite and positive"))
    value
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
    last_value = T(NaN)
    last_residual = T(Inf)
    for iteration in 0:criteria.max_iterations
        value = try
            T(f(x))
        catch error
            if error isa DomainError
                push!(failures, "objective rejected the current iterate at iteration $iteration")
                return SearchResult(:failed, x, T(NaN), T(Inf), iteration,
                    "damped-newton-gradient-residual", failures)
            end
            rethrow()
        end
        if !isfinite(value)
            push!(failures, "objective returned a nonfinite value at iteration $iteration")
            return SearchResult(:failed, x, value, T(Inf), iteration,
                "damped-newton-gradient-residual", failures)
        end
        last_value = value
        residual = try
            gradient!(g, differentiation, f, x; scales=criteria.field_scales)
            T(scaled_stationarity_residual(g, criteria.field_scales,
                _search_potential_scale(criteria, x)))
        catch error
            if error isa DomainError
                push!(failures, "gradient or pointwise scale rejected the current iterate at iteration $iteration")
                return SearchResult(:failed, x, value, T(Inf), iteration,
                    "damped-newton-gradient-residual", failures)
            end
            rethrow()
        end
        last_residual = residual
        if residual <= criteria.stationarity_tolerance
            return SearchResult(:converged, x, value, residual, iteration,
                "damped-newton-gradient-residual", failures)
        end
        if iteration == criteria.max_iterations
            push!(failures, "iteration budget exhausted at iteration $iteration")
            return SearchResult(:failed, x, value, residual, iteration,
                "damped-newton-gradient-residual", failures)
        end
        try
            hessian!(H, differentiation, f, x; scales=criteria.field_scales)
        catch error
            if error isa DomainError
                push!(failures, "Hessian evaluation rejected the current iterate at iteration $iteration")
                return SearchResult(:failed, x, value, residual, iteration,
                    "damped-newton-gradient-residual", failures)
            end
            rethrow()
        end
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
            trial_residual = try
                gradient!(trial_gradient, differentiation, f, trial; scales=criteria.field_scales)
                T(scaled_stationarity_residual(trial_gradient, criteria.field_scales,
                    _search_potential_scale(criteria, trial)))
            catch error
                if error isa DomainError
                    push!(failures, "domain rejected a line-search trial: objective, gradient, or pointwise scale rejected it at iteration $iteration")
                    alpha *= criteria.backtracking_factor
                    continue
                end
                rethrow()
            end
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
    SearchResult(:failed, x, last_value, last_residual, criteria.max_iterations,
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
    normalized_eigenvalue::T
    disposition::Symbol
    sign_status::Symbol
end

struct FluctuationReport{T<:Real}
    coordinate_hessian::Matrix{T}
    covariant_hessian::Matrix{T}
    retained_kinetic_metric::Matrix{T}
    generalized_mass_eigenvalues::Vector{T}
    normalized_generalized_mass_eigenvalues::Vector{T}
    mode_dispositions::Vector{ModeDisposition{T}}
    active_axionic_shift_directions::Matrix{Rational{BigInt}}
    symmetry_kernel_assessment::Assessment
    symmetry_mode_assessment::Assessment
    spectral_potential_scale::T
    spectral_field_scales::Vector{T}
    physical_mass_assessment::Assessment
    bf_assessment::Assessment
end

abstract type FluctuationBackend end
struct GeneralizedEigenBackend <: FluctuationBackend end

function fluctuation_analysis(::GeneralizedEigenBackend, hessian::AbstractMatrix{T},
        kinetic_metric::AbstractMatrix{T}; at_critical_point=true,
        covariant_hessian=nothing,
        absolute_zero_threshold=T(1e-12), relative_zero_threshold=T(1e-10),
        potential_scale=one(T), field_scales=ones(T, size(hessian, 1)),
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
    absolute_threshold = T(absolute_zero_threshold)
    relative_threshold = T(relative_zero_threshold)
    mass_scale = T(potential_scale)
    isfinite(absolute_threshold) && absolute_threshold >= zero(T) ||
        throw(ArgumentError("absolute spectral threshold must be finite and nonnegative"))
    isfinite(relative_threshold) && relative_threshold >= zero(T) ||
        throw(ArgumentError("relative spectral threshold must be finite and nonnegative"))
    isfinite(mass_scale) && mass_scale > zero(T) ||
        throw(ArgumentError("spectral potential scale must be finite and positive"))
    length(field_scales) == size(Hcov, 1) ||
        throw(DimensionMismatch("spectral field scales must match the retained fields"))
    scales = collect(T.(field_scales))
    all(value -> isfinite(value) && value > zero(T), scales) ||
        throw(ArgumentError("spectral field scales must be finite and positive"))
    L = cholesky(Symmetric(G)).L
    normalized = L \ Hcov / L'
    decomposition = eigen(Symmetric((normalized + normalized') / T(2)))
    values = decomposition.values
    normalized_values = values ./ mass_scale
    all(isfinite, normalized_values) ||
        throw(DomainError(normalized_values, "normalized generalized masses must be finite"))
    scale = max(maximum(abs, normalized_values), one(T))
    relative_limit = relative_threshold * scale
    isfinite(relative_limit) ||
        throw(DomainError(relative_limit, "relative spectral threshold overflowed"))
    near_limit = max(absolute_threshold, relative_limit)
    dispositions = ModeDisposition{T}[]
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
    exact_symmetry_annihilation = false
    symmetry_status = if active_charges === nothing
        Assessment(:NOT_ASSESSED, "active charges were not supplied")
    elseif size(symmetry_directions, 2) == 0
        Assessment(:PASS, "the active charge matrix has no axionic shift kernel")
    else
        numeric_directions = T.(symmetry_directions)
        residual = Hcov * numeric_directions
        exact_symmetry_annihilation = all(iszero, residual)
        residual_scale = opnorm(Hcov, Inf) * opnorm(numeric_directions, Inf)
        residual_norm = norm(residual, Inf)
        relative_residual = if !isfinite(residual_norm) || !isfinite(residual_scale)
            T(Inf)
        elseif iszero(residual_norm)
            zero(T)
        elseif iszero(residual_scale)
            T(Inf)
        else
            residual_norm / residual_scale
        end
        kernel_tolerance = T(4) * eps(T)
        isfinite(relative_residual) && relative_residual <= kernel_tolerance ?
            Assessment(:PASS, "the exact charge kernel is annihilated within a precision-scaled residual tolerance; exact-zero labels still require exact Hessian annihilation") :
            Assessment(:FAIL, "the covariant Hessian residual on the exact charge kernel exceeds the precision-scaled tolerance")
    end
    symmetry_subspace = zeros(T, size(G, 1), 0)
    if symmetry_status.status === :PASS && size(symmetry_directions, 2) > 0
        transformed_directions = L' * T.(symmetry_directions)
        symmetry_subspace = svd(transformed_directions).U[:, 1:size(symmetry_directions, 2)]
    end
    near_indices = findall(value -> abs(value) <= near_limit, normalized_values)
    symmetry_mode_status = if symmetry_status.status !== :PASS
        Assessment(:NOT_ASSESSED, "exact symmetry-mode attribution requires a validated charge/Hessian kernel")
    elseif size(symmetry_directions, 2) == 0
        Assessment(:PASS, "no exact active-charge shift modes are present")
    elseif isempty(near_indices)
        Assessment(:FAIL, "the validated exact shift kernel is absent from the near-zero generalized eigenspace")
    else
        zero_subspace = decomposition.vectors[:, near_indices]
        projection_residual = norm(symmetry_subspace -
            zero_subspace * (zero_subspace' * symmetry_subspace))
        subspace_tolerance = T(100) * sqrt(eps(T)) *
            max(norm(symmetry_subspace), one(T))
        projection_residual <= subspace_tolerance ?
            Assessment(:PASS, "the exact charge kernel lies in the near-zero eigenspace; per-mode labels remain conservative under degenerate mixing") :
            Assessment(:FAIL, "the near-zero eigenspace does not contain the validated exact charge kernel")
    end
    for i in eachindex(values)
        symmetry_overlap = isempty(symmetry_subspace) ? zero(T) :
            sum(abs2, symmetry_subspace' * decomposition.vectors[:, i])
        entire_near_space_is_kernel = length(near_indices) == size(symmetry_directions, 2)
        pure_symmetry_mode = symmetry_overlap >= one(T) - T(100) * sqrt(eps(T))
        disposition = if symmetry_mode_status.status === :PASS &&
                exact_symmetry_annihilation && i in near_indices &&
                (entire_near_space_is_kernel || pure_symmetry_mode)
            :symmetry_protected_exact_zero
        elseif abs(normalized_values[i]) <= near_limit
            :numerically_unresolved_near_zero
        elseif values[i] < zero(T)
            :tachyonic
        else
            :lifted
        end
        sign_status = normalized_values[i] < -near_limit ? :negative :
            (normalized_values[i] > near_limit ? :positive : :zero_within_threshold)
        push!(dispositions, ModeDisposition(values[i], normalized_values[i], disposition, sign_status))
    end
    bf = if ads_radius_squared === nothing
        Assessment(:NOT_APPLICABLE, "no AdS radius was supplied")
    else
        ads_radius_squared > zero(T) || throw(ArgumentError("AdS radius squared must be positive"))
        all(value -> value * ads_radius_squared >= -T(9) / T(4), values) ?
            Assessment(:PASS, "all retained generalized masses satisfy m^2 L_AdS^2 >= -9/4") :
            Assessment(:FAIL, "at least one retained generalized mass violates the AdS4 BF bound")
    end
    FluctuationReport(Hcoord, Hcov, G, collect(values), collect(normalized_values),
        dispositions, symmetry_directions, symmetry_status, symmetry_mode_status,
        mass_scale, scales,
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

"""Read-only snapshot of caller-supplied array data in a replay manifest."""
struct FrozenManifestArray{N,L} <: AbstractArray{Any,N}
    values::NTuple{L,Any}
    dimensions::NTuple{N,Int}

    function FrozenManifestArray{N,L}(values::NTuple{L,Any}, dimensions::NTuple{N,Int},
            ::Val{:frozen}) where {N,L}
        prod(dimensions) == L || throw(DimensionMismatch("manifest array shape does not match its data"))
        all(_manifest_value_is_immutable, values) ||
            throw(ArgumentError("manifest array storage must be recursively immutable"))
        new{N,L}(values, dimensions)
    end
end

Base.size(array::FrozenManifestArray) = getfield(array, :dimensions)
Base.IndexStyle(::Type{<:FrozenManifestArray}) = IndexLinear()
Base.getindex(array::FrozenManifestArray, index::Int) = getfield(array, :values)[index]
function Base.getindex(array::FrozenManifestArray{N}, indices::Vararg{Int,N}) where {N}
    getfield(array, :values)[LinearIndices(array)[indices...]]
end

_manifest_value_is_immutable(value) = isbitstype(typeof(value))
_manifest_value_is_immutable(value::AbstractString) = true
_manifest_value_is_immutable(value::Symbol) = true
_manifest_value_is_immutable(::Nothing) = true
_manifest_value_is_immutable(::Missing) = true
_manifest_value_is_immutable(value::Tuple) = all(_manifest_value_is_immutable, value)
_manifest_value_is_immutable(value::NamedTuple) =
    all(_manifest_value_is_immutable, values(value))
function _manifest_value_is_immutable(value)
    ismutabletype(typeof(value)) && return false
    all(index -> _manifest_value_is_immutable(getfield(value, index)),
        1:fieldcount(typeof(value)))
end

_snapshot_manifest(value::BigInt) =
    (; numeric_type="BigInt", decimal=string(value))
_snapshot_manifest(value::BigFloat) =
    (; numeric_type="BigFloat", precision_bits=precision(value), decimal=string(value))
_snapshot_manifest(value::Rational{BigInt}) =
    (; numeric_type="Rational{BigInt}", numerator=string(numerator(value)),
       denominator=string(denominator(value)))
_snapshot_manifest(value::NamedTuple) =
    NamedTuple{keys(value)}(map(_snapshot_manifest, values(value)))
_snapshot_manifest(value::Tuple) = map(_snapshot_manifest, value)
function _snapshot_manifest(value::AbstractArray{T,N}) where {T,N}
    try
        return FrozenArray(value)
    catch error
        error isa ArgumentError || rethrow()
    end
    values = Tuple(_snapshot_manifest(item) for item in value)
    FrozenManifestArray{N,length(values)}(values, size(value), Val(:frozen))
end
function _snapshot_manifest(value::AbstractDict)
    entries = sort!(collect(pairs(value)); by=pair -> repr(pair.first))
    (; entries=Tuple((key=_snapshot_manifest(pair.first),
        value=_snapshot_manifest(pair.second)) for pair in entries))
end
_snapshot_manifest(value::Pair) =
    (; first=_snapshot_manifest(first(value)), second=_snapshot_manifest(last(value)))
function _snapshot_manifest(value)
    _manifest_value_is_immutable(value) && return value
    ismutabletype(typeof(value)) &&
        throw(ArgumentError("mutable replay manifest value $(typeof(value)) has no immutable snapshot encoding"))
    names = fieldnames(typeof(value))
    fields = NamedTuple{names}(Tuple(_snapshot_manifest(getfield(value, name)) for name in names))
    (; type=string(typeof(value)), fields)
end

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
    manifest = (; schema_version="cyax0191-replay-v2",
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
    _snapshot_manifest(manifest)
end

"""Rebase an integer charge matrix under D_new = B*D_old."""
function change_charge_basis(charges::AbstractMatrix{<:Integer}, B::AbstractMatrix{<:Integer})
    size(charges, 2) == size(B, 1) == size(B, 2) ||
        throw(DimensionMismatch("charge matrix and basis transform dimensions differ"))
    BigInt.(charges) * _integer_inverse(B)
end

function change_model_basis(model::KahlerModel, B::AbstractMatrix{<:Integer})
    Qnew = change_charge_basis(model.charges, B)
    KahlerModel(; w0_magnitude=model.w0_magnitude, theta0=model.theta0,
        gs=model.gs, kcs=model.kcs, amplitudes=model.amplitudes,
        actions=model.actions, phases=model.phases, charges=Qnew,
        convention=model.convention, switches=model.switches,
        uplift=model.uplift, identity="$(model.identity)|basis=$(repr(Matrix{BigInt}(B)))")
end

function change_coordinate_basis(t::AbstractVector, rho::AbstractVector,
        B::AbstractMatrix{<:Integer})
    size(B, 1) == size(B, 2) == length(t) == length(rho) ||
        throw(DimensionMismatch("coordinate and basis dimensions differ"))
    Binv = _integer_inverse(B)
    exact_basis = BigInt.(B)
    (Binv' * _widen_exact_array(t), exact_basis * _widen_exact_array(rho))
end
