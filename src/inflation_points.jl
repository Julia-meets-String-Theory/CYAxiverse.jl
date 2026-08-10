"""
    CYAxiverse.inflation_points

Numerical boundaries for correcting, diagnosing, and flowing generic
axion-potential critical points.  This module deliberately does not choose a
geometry population, a scale path, or a candidate threshold.
"""
module inflation_points

using LinearAlgebra
using ..read: oriented_potential
using ..structs: GeometryIndex

export PointContext, PointDerivatives, PointDiagnostics, CorrectionResult,
    PrecisionComparison, prepare_context, derivatives, diagnose,
    correct_stationary_point, compare_precision, mass_eigenbasis,
    basis_policy, prepare_geometry_context, gradient_flow

"""
    basis_policy()

Return the SCI-02 basis contract.  `:periodic_string` is the working basis
for branch enumeration, derivative evaluation, and stationary-point
correction.  `:mass_eigenbasis` is the physical basis for mode-resolved
quantities.  The latter is materialized only when mode vectors are requested;
scalar diagnostics use the equivalent generalized-Hessian eigenvalues.
"""
basis_policy() = (; working_basis=:periodic_string,
    physical_basis=:mass_eigenbasis, physical_vectors=:deferred,
    dense_charge_rotation=:deferred)

"""Prepared potential inputs at one arithmetic precision."""
struct PointContext{T<:AbstractFloat, F}
    Q::Matrix{Int}
    L::Matrix{T}
    K::Matrix{T}
    factor::F
    amplitudes::Vector{T}
    log_shift::T
    precision_bits::Int
end

"""Value, gradient, and Hessian of the log-shifted potential."""
struct PointDerivatives{T<:AbstractFloat}
    value::T
    gradient::Vector{T}
    hessian::Matrix{T}
    log_shift::T
end

"""Generalized-Hessian and slow-roll diagnostics at a point."""
struct PointDiagnostics{T<:AbstractFloat}
    value::T
    gradient_norm::T
    gradient_residual::T
    epsilon::T
    eta_values::Vector{T}
    hessian_eigenvalues::Vector{T}
    negative_modes::Int
    zeroish_modes::Int
    positive_modes::Int
    zero_tolerance::T
    physical_basis::Symbol
end

"""Result of bounded stationary-point correction."""
struct CorrectionResult{T<:AbstractFloat}
    theta::Vector{T}
    status::Symbol
    residual::T
    iterations::Int
    seconds::Float64
    error::String
    working_basis::Symbol
end

"""Float64/arbitrary-precision re-evaluation of one retained point."""
struct PrecisionComparison{F<:AbstractFloat, H<:AbstractFloat}
    float_correction::CorrectionResult{F}
    high_correction::CorrectionResult{H}
    float_diagnostics::Union{Nothing, PointDiagnostics{F}}
    high_diagnostics::Union{Nothing, PointDiagnostics{H}}
    residual_agreement::Bool
    inertia_agreement::Bool
    accepted::Bool
end

function _validate_inputs(Q::AbstractMatrix{Int}, L::AbstractMatrix{<:Real},
        K::AbstractMatrix{<:Real})
    h11, instantons = size(Q)
    size(L) == (2, instantons) ||
        throw(DimensionMismatch("Q and L dimensions disagree"))
    size(K) == (h11, h11) ||
        throw(DimensionMismatch("Q and K dimensions disagree"))
    all(isfinite, L) || throw(ArgumentError("L contains non-finite values"))
    all(isfinite, K) || throw(ArgumentError("K contains non-finite values"))
    h11 > 0 && instantons > 0 ||
        throw(ArgumentError("Q must contain at least one axion and instanton"))
    nothing
end

function _context(Q::AbstractMatrix{Int}, L::AbstractMatrix{<:Real},
        K::AbstractMatrix{<:Real}, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    _validate_inputs(Q, L, K)
    q = Matrix{Int}(Q)
    l = Matrix{T}(L)
    k = Matrix{T}(K)
    k .= (k .+ transpose(k)) ./ T(2)
    factor = cholesky(Hermitian(k))
    log_shift = maximum(@view l[2, :])
    amplitudes = @view(l[1, :]) .* T(10) .^ (@view(l[2, :]) .- log_shift)
    PointContext(q, l, k, factor, collect(amplitudes), log_shift, precision_bits)
end

"""
    prepare_context(Q, L, K; precision_bits=nothing)

Validate oriented periodic/string-basis potential inputs and prepare a reusable derivative
context.  With `precision_bits=nothing`, the context uses `Float64`; otherwise
it uses `BigFloat` at the requested precision.  The kinetic matrix must be
finite, symmetric after symmetrization, and positive definite.
"""
function prepare_context(Q::AbstractMatrix{Int}, L::AbstractMatrix{<:Real},
        K::AbstractMatrix{<:Real}; precision_bits::Union{Nothing, Int}=nothing)
    if precision_bits === nothing
        return _context(Q, L, K, Float64, 53)
    end
    precision_bits >= 64 ||
        throw(ArgumentError("arbitrary precision requires at least 64 bits"))
    setprecision(BigFloat, precision_bits) do
        _context(Q, L, K, BigFloat, precision_bits)
    end
end

function _periodic(theta::AbstractVector{T}) where {T<:AbstractFloat}
    mod.(theta, one(T))
end

function _mass_hessian(context::PointContext{T}, hessian::AbstractMatrix{<:Real}) where {T}
    lower = context.factor.L
    # This is only a numerical reduction of H*v = m^2*K*v.  It is not
    # exposed as the physical coordinate basis: the physical basis is the
    # mass eigenbasis obtained by diagonalizing this representation.
    lower \ T.(hessian) / transpose(lower)
end

"""
    mass_eigenbasis(context, data; vectors=false)

Solve the physical generalized-Hessian problem at a corrected point.  The
returned eigenvalues are ordered increasingly and are the dimensionless
mass-squared eigenvalues of `H*v = m²*K*v`.  With `vectors=true`, also return
raw-coordinate eigenvectors that are `K`-orthonormal.  Vector materialization
is deliberately opt-in because it costs `O(h11^2)` storage.
"""
function mass_eigenbasis(context::PointContext{T}, data::PointDerivatives{T};
        vectors::Bool=false) where {T}
    mass_hessian = _mass_hessian(context, data.hessian)
    if !vectors
        return (; basis=:mass_eigenbasis,
            eigenvalues=eigvals(Symmetric(mass_hessian)))
    end
    eigensystem = eigen(Symmetric(mass_hessian))
    raw_eigenvectors = transpose(context.factor.L) \ eigensystem.vectors
    (; basis=:mass_eigenbasis, eigenvalues=eigensystem.values,
       raw_eigenvectors,
       metric_residual=norm(raw_eigenvectors' * context.K * raw_eigenvectors -
           I, Inf),
       generalized_residual=opnorm(data.hessian * raw_eigenvectors -
           context.K * raw_eigenvectors * Diagonal(eigensystem.values)))
end

function mass_eigenbasis(context::PointContext, theta::AbstractVector{<:Real};
        vectors::Bool=false)
    mass_eigenbasis(context, derivatives(context, theta); vectors=vectors)
end

"""Load one geometry into the SCI-02 periodic/string-basis context."""
function prepare_geometry_context(geom_idx::GeometryIndex;
        precision_bits::Union{Nothing, Int}=nothing)
    loaded = oriented_potential(geom_idx)
    (; geometry=geom_idx, input_basis=:periodic_string,
       source=:oriented_potential,
       context=prepare_context(loaded.Q, loaded.L, loaded.K; precision_bits))
end

"""Evaluate a prepared log-shifted potential without mutating the context."""
function derivatives(context::PointContext{T}, theta::AbstractVector{<:Real}) where {T}
    length(theta) == size(context.Q, 1) ||
        throw(DimensionMismatch("theta must have one entry per axion"))
    x = T.(theta)
    h11, instantons = size(context.Q)
    gradient = zeros(T, h11)
    hessian = zeros(T, h11, h11)
    value = zero(T)
    two_pi = T(2) * T(π)
    four_pi_squared = two_pi^2
    @inbounds for instanton in 1:instantons
        phase = zero(T)
        for axion in 1:h11
            phase += T(context.Q[axion, instanton]) * x[axion]
        end
        amplitude = context.amplitudes[instanton]
        sine = sin(two_pi * phase)
        cosine = cos(two_pi * phase)
        value += amplitude * (one(T) - cosine)
        gradient_scale = two_pi * amplitude * sine
        hessian_scale = four_pi_squared * amplitude * cosine
        for row in 1:h11
            charge_row = T(context.Q[row, instanton])
            gradient[row] += gradient_scale * charge_row
            for column in 1:h11
                hessian[row, column] += hessian_scale * charge_row *
                    T(context.Q[column, instanton])
            end
        end
    end
    PointDerivatives(value, gradient, hessian, context.log_shift)
end

"""Diagnose a point using the generalized Hessian `H v = m² K v`."""
function diagnose(context::PointContext{T}, theta::AbstractVector{<:Real};
        zero_tolerance::Real=1e-10) where {T}
    data = derivatives(context, theta)
    eigenvalues = mass_eigenbasis(context, data).eigenvalues
    lower = context.factor.L
    inverse_metric_gradient = transpose(lower) \ (lower \ data.gradient)
    gradient_norm = sqrt(max(dot(data.gradient, inverse_metric_gradient), zero(T)))
    value = data.value
    epsilon = iszero(value) ? T(Inf) : T(0.5) * (gradient_norm / abs(value))^2
    eta_values = iszero(value) ? fill(T(Inf), length(eigenvalues)) :
        eigenvalues ./ value
    eigen_scale = max(maximum(abs, eigenvalues), one(T))
    threshold = T(zero_tolerance) * eigen_scale
    negative_modes = count(x -> x < -threshold, eigenvalues)
    zeroish_modes = count(x -> abs(x) <= threshold, eigenvalues)
    positive_modes = count(x -> x > threshold, eigenvalues)
    PointDiagnostics(value, gradient_norm, norm(data.gradient, Inf), epsilon,
        eta_values, eigenvalues, negative_modes, zeroish_modes, positive_modes,
        threshold, :mass_eigenbasis)
end

function _flow_state(context::PointContext{T}, chi::AbstractVector{<:Real}) where {T}
    lower = context.factor.L
    theta = transpose(lower) \ T.(chi)
    data = derivatives(context, theta)
    gradient = lower \ data.gradient
    hessian = lower \ data.hessian / transpose(lower)
    value = data.value
    gradient_squared = dot(gradient, gradient)
    gradient_norm = sqrt(max(gradient_squared, zero(T)))
    epsilon = iszero(value) ? T(Inf) : gradient_squared / (T(2) * value^2)
    tangent = iszero(gradient_norm) ? zeros(T, length(gradient)) :
        -gradient / gradient_norm
    eta_parallel = iszero(gradient_norm) || iszero(value) ? T(Inf) :
        dot(tangent, hessian * tangent) / value
    (; theta, value, gradient, hessian, epsilon, eta_parallel,
       indicator=max(epsilon, abs(eta_parallel)) - one(T))
end

function _flow_rhs(context::PointContext{T}, chi::AbstractVector{T}) where {T}
    state = _flow_state(context, chi)
    state.value > zero(T) ||
        throw(DomainError(state.value, "gradient flow requires positive potential"))
    -state.gradient / state.value
end

function _flow_exit_event(state)
    state.epsilon >= abs(state.eta_parallel) ? :epsilon : :eta_parallel
end

function _flow_mode_index(eigenvalues::AbstractVector, mode::Symbol)
    mode === :most_negative && return argmin(eigenvalues)
    mode === :smallest_abs && return argmin(abs.(eigenvalues))
    throw(ArgumentError("unsupported mass mode: $mode"))
end

"""
    gradient_flow(context, hilltop; kwargs...)

Run a bounded generic slow-roll flow from a physical mass-mode displacement.
The potential is evaluated in the periodic/string basis, while the state is
integrated in the reusable canonical numerical chart `chi = L' * theta` for
`K = L * L'`.  The selected initial direction is a raw-coordinate,
`K`-normalized mass eigenvector.  E-folds are the independent variable, so
the result does not depend on the arbitrary overall normalization of the
stored log-shifted potential.  Pass `mass_basis` from
`mass_eigenbasis(...; vectors=true)` and `mode_index` when evaluating multiple
physical modes; this reuses the eigensystem instead of allocating it for every
displacement direction.

This is a candidate-level diagnostic, not a claim of stabilized geometry,
Kähler-cone validity, or a production population scan.  The fixed-step RK4
integrator is intentionally bounded and records a `:max_efolds` status when
the requested horizon is reached without a finite exit.
"""
function gradient_flow(context::PointContext{T}, hilltop::AbstractVector{<:Real};
        displacement::Real=1e-8, displacement_sign::Real=-1,
        mode::Symbol=:most_negative, mode_index::Union{Nothing, Int}=nothing,
        mass_basis::Union{Nothing, NamedTuple}=nothing,
        max_efolds::Real=60, step::Real=1e-3) where {T}
    length(hilltop) == size(context.Q, 1) ||
        throw(DimensionMismatch("hilltop must have one entry per axion"))
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    displacement_sign != 0 || throw(ArgumentError("displacement_sign must be nonzero"))
    max_efolds > 0 || throw(ArgumentError("max_efolds must be positive"))
    step > 0 || throw(ArgumentError("step must be positive"))
    all(isfinite, hilltop) || throw(ArgumentError("hilltop contains non-finite values"))
    mass = mass_basis === nothing ?
        mass_eigenbasis(context, hilltop; vectors=true) : mass_basis
    hasproperty(mass, :raw_eigenvectors) ||
        throw(ArgumentError("mass_basis must include raw_eigenvectors"))
    length(mass.eigenvalues) == length(hilltop) ||
        throw(DimensionMismatch("mass_basis has the wrong mode count"))
    index = mode_index === nothing ? _flow_mode_index(mass.eigenvalues, mode) :
        mode_index
    1 <= index <= length(mass.eigenvalues) ||
        throw(ArgumentError("mode_index is outside the mass spectrum"))
    direction = @view mass.raw_eigenvectors[:, index]
    theta_initial = _periodic(T.(hilltop) .+
        T(displacement_sign * displacement) .* direction)
    lower = context.factor.L
    chi = transpose(lower) * theta_initial
    initial = _flow_state(context, chi)
    current_efolds = zero(T)
    step_size = T(step)
    horizon = T(max_efolds)
    windows = Tuple{T, T, Symbol}[]
    entered_at = nothing
    status = :max_efolds
    error_message = ""
    try
        initial.indicator <= zero(T) && (entered_at = current_efolds)
        max_steps = ceil(Int, horizon / step_size)
        for _ in 1:max_steps
            current_efolds >= horizon && break
            h = min(step_size, horizon - current_efolds)
            k1 = _flow_rhs(context, chi)
            k2 = _flow_rhs(context, chi .+ (h / T(2)) .* k1)
            k3 = _flow_rhs(context, chi .+ (h / T(2)) .* k2)
            k4 = _flow_rhs(context, chi .+ h .* k3)
            chi .+= (h / T(6)) .* (k1 .+ T(2) .* k2 .+ T(2) .* k3 .+ k4)
            current_efolds += h
            state = _flow_state(context, chi)
            if entered_at === nothing && state.indicator <= zero(T)
                entered_at = current_efolds
            elseif entered_at !== nothing && state.indicator >= zero(T)
                push!(windows, (entered_at, current_efolds,
                    _flow_exit_event(state)))
                entered_at = nothing
            end
        end
        entered_at !== nothing && push!(windows,
            (entered_at, current_efolds, :max_efolds))
        status = isempty(windows) ? :no_slow_roll_window : :completed
    catch error
        status = :failed
        error_message = sprint(showerror, error)
    end
    selected = isempty(windows) ? nothing :
        windows[argmax(getindex.(windows, 2) .- getindex.(windows, 1))]
    final = _flow_state(context, chi)
    (; status, error=error_message, basis=:mass_eigenbasis,
       coordinate_chart=:canonical_cholesky, mode, mode_index=index,
       displacement_sign,
       mass_eigenvalue=mass.eigenvalues[index],
       mass_direction=collect(direction), theta_initial,
       theta_final=_periodic(final.theta), initial_epsilon=initial.epsilon,
       initial_eta_parallel=initial.eta_parallel,
       efolds=selected === nothing ? zero(T) : selected[2] - selected[1],
       slow_roll_efolds=selected === nothing ? zero(T) : selected[2] - selected[1],
       entry_efolds=selected === nothing ? nothing : selected[1],
       exit_efolds=selected === nothing ? nothing : selected[2],
       end_event=selected === nothing ? :no_slow_roll_window : selected[3],
       steps=ceil(Int, current_efolds / step_size), windows,
       max_efolds=horizon, step=step_size)
end

"""Run [`gradient_flow`](@ref) after loading one `GeometryIndex`."""
function gradient_flow(geom_idx::GeometryIndex, hilltop::AbstractVector{<:Real};
        precision_bits::Union{Nothing, Int}=nothing, kwargs...)
    prepared = prepare_geometry_context(geom_idx; precision_bits)
    result = gradient_flow(prepared.context, hilltop; kwargs...)
    merge(result, (; geometry=geom_idx, input_basis=prepared.input_basis,
        source=prepared.source))
end

function _newton_step(context::PointContext{T}, theta::Vector{T}) where {T}
    data = derivatives(context, theta)
    step = try
        -(Symmetric(data.hessian) \ data.gradient)
    catch error
        throw(ErrorException(string("stationary Newton solve failed: ",
            sprint(showerror, error))))
    end
    data, step
end

"""
    correct_stationary_point(context, seed; kwargs...)

Run a bounded damped Newton correction of a retained branch seed.  The result
is labelled `:converged` only when the coordinate-gradient infinity norm meets
`residual_tolerance`; otherwise it records an explicit failure status.
"""
function correct_stationary_point(context::PointContext{T}, seed::AbstractVector{<:Real};
        residual_tolerance::Real=1e-10, max_iterations::Int=100,
        max_line_search::Int=12) where {T}
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    max_line_search >= 0 || throw(ArgumentError("max_line_search must be non-negative"))
    tolerance = T(residual_tolerance)
    tolerance > zero(T) || throw(ArgumentError("residual_tolerance must be positive"))
    theta = _periodic(T.(seed))
    started = time_ns()
    residual = T(Inf)
    for iteration in 0:max_iterations
        data = derivatives(context, theta)
        residual = norm(data.gradient, Inf)
        if isfinite(residual) && residual <= tolerance
            return CorrectionResult(theta, :converged, residual, iteration,
                (time_ns() - started) / 1e9, "", :periodic_string)
        end
        iteration == max_iterations && break
        step = try
            -(Symmetric(data.hessian) \ data.gradient)
        catch error
            return CorrectionResult(theta, :singular_hessian, residual, iteration,
                (time_ns() - started) / 1e9, sprint(showerror, error),
                :periodic_string)
        end
        isfinite(norm(step, Inf)) ||
            return CorrectionResult(theta, :nonfinite_step, residual, iteration,
                (time_ns() - started) / 1e9, "Newton step is non-finite",
                :periodic_string)
        current = residual
        accepted = false
        damping = one(T)
        for _ in 0:max_line_search
            trial = _periodic(theta .+ damping .* step)
            trial_residual = norm(derivatives(context, trial).gradient, Inf)
            if isfinite(trial_residual) && trial_residual < current
                theta = trial
                accepted = true
                break
            end
            damping /= T(2)
        end
        accepted || return CorrectionResult(theta, :line_search_failed,
            residual, iteration, (time_ns() - started) / 1e9,
            "damped Newton step did not reduce the gradient residual",
            :periodic_string)
    end
    CorrectionResult(theta, :max_iterations, residual, max_iterations,
        (time_ns() - started) / 1e9, "stationary correction did not converge",
        :periodic_string)
end

function _screen_pass(diagnostic::PointDiagnostics)
    diagnostic.value > 0 && diagnostic.negative_modes > 0 &&
        diagnostic.epsilon < 1 && abs(minimum(diagnostic.eta_values)) < 1
end

"""Re-correct and compare one retained point at Float64 and BigFloat precision."""
function compare_precision(seed::AbstractVector{<:Real}, Q::AbstractMatrix{Int},
        L::AbstractMatrix{<:Real}, K::AbstractMatrix{<:Real};
        precision_bits::Int=256, float_residual_tolerance::Real=1e-10,
        high_residual_tolerance::Real=1e-40, zero_tolerance::Real=1e-10,
        max_iterations::Int=100, max_line_search::Int=12)
    float_context = prepare_context(Q, L, K)
    float_correction = correct_stationary_point(float_context, seed;
        residual_tolerance=float_residual_tolerance, max_iterations,
        max_line_search)
    float_diagnostics = float_correction.status == :converged ?
        diagnose(float_context, float_correction.theta; zero_tolerance) : nothing

    high_correction, high_diagnostics = setprecision(BigFloat, precision_bits) do
        high_context = prepare_context(Q, L, K; precision_bits)
        correction = correct_stationary_point(high_context,
            float_correction.theta; residual_tolerance=high_residual_tolerance,
            max_iterations, max_line_search)
        diagnostics = correction.status == :converged ?
            diagnose(high_context, correction.theta; zero_tolerance) : nothing
        correction, diagnostics
    end

    residual_agreement = float_diagnostics !== nothing &&
        high_diagnostics !== nothing &&
        float_correction.residual <= float_residual_tolerance &&
        high_correction.residual <= high_residual_tolerance
    inertia_agreement = float_diagnostics !== nothing && high_diagnostics !== nothing &&
        (float_diagnostics.negative_modes, float_diagnostics.zeroish_modes,
         float_diagnostics.positive_modes) ==
        (high_diagnostics.negative_modes, high_diagnostics.zeroish_modes,
         high_diagnostics.positive_modes)
    PrecisionComparison(float_correction, high_correction, float_diagnostics,
        high_diagnostics, residual_agreement, inertia_agreement,
        residual_agreement && inertia_agreement)
end

end
