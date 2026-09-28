#!/usr/bin/env julia

"""Bounded phase and volume-detuning search for low-dimensional potentials.

The volume parameter is a mathematical amplitude homotopy.  It is deliberately
labelled `homotopy_only`: this script does not transform divisor volumes,
kinetic data, or instanton units and therefore cannot produce a physical
inflation candidate.  The reported result is a zero-mode crossing at the
specified fixed configuration.  It does not establish stationarity, branch
identity on either side, or a catastrophe classification.
"""
module PhaseVolumeDetuningScan

using LinearAlgebra
using GenericLinearAlgebra
using Printf

export ScanCandidate, phase_vectors, potential, hessian, refine_zero_mode_crossing,
    scan, scan_detailed

struct ScanCandidate{T}
    phase::Vector{T}
    k_homotopy_low::T
    k_homotopy_high::T
    k_homotopy_c::T
    eigenvalue_at_k_homotopy_low::T
    eigenvalue_at_k_homotopy_high::T
    crossing_type::Symbol
    scale_status::Symbol
    n_e::Missing
    n_s::Missing
    scalar_amplitude::Missing
    r::Missing
end

function _validate(Q, L, phase)
    ndims(Q) == 2 || throw(ArgumentError("Q must be a matrix"))
    ndims(L) == 2 && size(L, 2) == 2 ||
        throw(DimensionMismatch("L must have two columns"))
    size(Q, 1) == size(L, 1) ||
        throw(DimensionMismatch("Q and L must have one row per instanton"))
    size(phase, 1) == size(Q, 1) ||
        throw(DimensionMismatch("phase must have one entry per instanton"))
    all(isfinite, Q) && all(isfinite, L) && all(isfinite, phase) ||
        throw(ArgumentError("Q, L, and phase must be finite"))
    all(x -> x > 0, L[:, 1]) ||
        throw(ArgumentError("L mantissas must be positive"))
    nothing
end

"""Return deterministic zero, one-phase, and selected two-phase perturbations."""
function phase_vectors(n::Integer; values=(-0.25, 0.25), pair_limit::Integer=typemax(Int))
    n > 0 || throw(ArgumentError("number of instantons must be positive"))
    pair_limit >= 0 || throw(ArgumentError("pair_limit must be nonnegative"))
    phase_values = Float64.(collect(values))
    isempty(phase_values) && throw(ArgumentError("values must not be empty"))
    all(isfinite, phase_values) ||
        throw(ArgumentError("phase values must be finite"))
    result = Vector{Vector{Float64}}()
    push!(result, zeros(Float64, n))
    for i in 1:n, value in phase_values
        phase = zeros(Float64, n)
        phase[i] = value
        push!(result, phase)
    end
    pairs = 0
    for i in 1:n-1, j in i+1:n, value_i in phase_values, value_j in phase_values
        pairs >= pair_limit && return result
        phase = zeros(Float64, n)
        phase[i], phase[j] = value_i, value_j
        push!(result, phase)
        pairs += 1
    end
    result
end

function _amplitudes(L, k, ::Type{T}) where {T<:AbstractFloat}
    k > zero(T) || throw(ArgumentError("k must be positive"))
    # This is a controlled hierarchy homotopy, not a physical divisor-volume map.
    T.(L[:, 1]) .* T(10) .^ (T(k) .* T.(L[:, 2]))
end

"""Evaluate the phase-shifted potential, with phases measured in cycles."""
function potential(theta, Q, L, phase; k=1, precision_bits=nothing)
    _validate(Q, L, phase)
    if precision_bits === nothing
        T = promote_type(Float64, eltype(theta), eltype(L))
        return _potential(T.(theta), Q, L, T.(phase), k)
    end
    precision_bits >= 64 || throw(ArgumentError("precision_bits must be at least 64"))
    setprecision(BigFloat, precision_bits) do
        _potential(BigFloat.(theta), Q, L, BigFloat.(phase), BigFloat(k))
    end
end

function _potential(theta, Q, L, phase, k)
    amplitudes = _amplitudes(L, k, eltype(theta))
    sum(amplitudes .* (one(eltype(theta)) .-
        cos.(2 * eltype(theta)(π) .* (Q * theta .+ phase))))
end

"""Return the Hessian in the periodic coordinate basis."""
function hessian(theta, Q, L, phase; k=1, precision_bits=nothing)
    _validate(Q, L, phase)
    if precision_bits === nothing
        T = promote_type(Float64, eltype(theta), eltype(L))
        return _hessian(T.(theta), Q, L, T.(phase), k)
    end
    precision_bits >= 64 || throw(ArgumentError("precision_bits must be at least 64"))
    setprecision(BigFloat, precision_bits) do
        _hessian(BigFloat.(theta), Q, L, BigFloat.(phase), BigFloat(k))
    end
end

function _hessian(theta, Q, L, phase, k)
    T = eltype(theta)
    amplitudes = _amplitudes(L, k, T)
    angles = T(2) * T(π) .* (Q * theta .+ phase)
    weighted = amplitudes .* cos.(angles)
    T(4) * T(π)^2 .* (transpose(Q) * (weighted .* Q))
end

function _zero_mode(h)
    values = eigvals(Symmetric(h))
    index = argmin(abs.(values))
    values[index]
end

function _refine_zero_mode_crossing(theta, Q, L, phase, k_low, k_high;
        precision_bits::Integer, tolerance::Real, max_iterations::Integer)
    precision_bits >= 64 || throw(ArgumentError("precision_bits must be at least 64"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    tolerance > 0 && isfinite(tolerance) ||
        throw(ArgumentError("tolerance must be finite and positive"))
    setprecision(BigFloat, precision_bits) do
        lo, hi = BigFloat(k_low), BigFloat(k_high)
        lo < hi || throw(ArgumentError("zero-mode crossing bracket must be increasing"))
        flo = _zero_mode(hessian(theta, Q, L, phase; k=lo,
            precision_bits=precision_bits))
        fhi = _zero_mode(hessian(theta, Q, L, phase; k=hi,
            precision_bits=precision_bits))
        tol = BigFloat(tolerance)
        if iszero(flo)
            return (; k_homotopy_c=lo, bracket_width=zero(BigFloat), residual=abs(flo),
                iterations=0, converged=true)
        elseif iszero(fhi)
            return (; k_homotopy_c=hi, bracket_width=zero(BigFloat), residual=abs(fhi),
                iterations=0, converged=true)
        end
        signbit(flo) == signbit(fhi) &&
            throw(ArgumentError("bracket must straddle a zero mode"))
        for iteration in 1:max_iterations
            if abs(hi - lo) <= tol
                mid = (lo + hi) / 2
                residual = abs(_zero_mode(hessian(theta, Q, L, phase; k=mid,
                    precision_bits=precision_bits)))
                return (; k_homotopy_c=mid, bracket_width=abs(hi - lo), residual,
                    iterations=iteration - 1, converged=true)
            end
            mid = (lo + hi) / 2
            fmid = _zero_mode(hessian(theta, Q, L, phase; k=mid,
                precision_bits=precision_bits))
            if iszero(fmid)
                return (; k_homotopy_c=mid, bracket_width=zero(BigFloat),
                    residual=abs(fmid), iterations=iteration, converged=true)
            elseif signbit(flo) == signbit(fmid)
                lo, flo = mid, fmid
            else
                hi, fhi = mid, fmid
            end
        end
        mid = (lo + hi) / 2
        residual = abs(_zero_mode(hessian(theta, Q, L, phase; k=mid,
            precision_bits=precision_bits)))
        (; k_homotopy_c=mid, bracket_width=abs(hi - lo), residual,
            iterations=max_iterations, converged=abs(hi - lo) <= tol)
    end
end

"""Refine a fixed-configuration homotopy Hessian zero-mode crossing by bisection."""
function refine_zero_mode_crossing(theta, Q, L, phase, k_homotopy_low,
        k_homotopy_high;
        precision_bits::Integer=256, tolerance::Real=1e-30,
        max_iterations::Integer=160)
    _refine_zero_mode_crossing(theta, Q, L, phase, k_homotopy_low,
        k_homotopy_high;
        precision_bits, tolerance, max_iterations).k_homotopy_c
end

"""Scan fixed-configuration homotopy zero modes; retain hits, rejects, and errors."""
function scan_detailed(theta, Q, L;
        k_homotopy_grid=range(0.5, 1.5; length=101),
        phases=phase_vectors(size(Q, 1)), precision_bits::Integer=256,
        tolerance::Real=1e-12, max_iterations::Integer=160)
    precision_bits >= 64 || throw(ArgumentError("precision_bits must be at least 64"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    tolerance > 0 && isfinite(tolerance) ||
        throw(ArgumentError("tolerance must be finite and positive"))
    ks = collect(k_homotopy_grid)
    length(ks) >= 2 || throw(ArgumentError("k_homotopy_grid must contain at least two values"))
    all(isfinite, ks) || throw(ArgumentError("k_homotopy_grid must be finite"))
    all(diff(ks) .> 0) || throw(ArgumentError("k_homotopy_grid must be strictly increasing"))
    all(>(0), ks) || throw(ArgumentError("k_homotopy_grid must contain positive values"))
    phase_list = collect(phases)
    isempty(phase_list) && throw(ArgumentError("phases must not be empty"))
    candidates = ScanCandidate{BigFloat}[]
    attempts = NamedTuple[]
    interval_denominator = length(phase_list) * (length(ks) - 1)
    for (phase_index, phase) in enumerate(phase_list)
        values = Vector{Union{Nothing,BigFloat}}(undef, length(ks))
        errors = fill("", length(ks))
        for (point_index, k) in enumerate(ks)
            try
                values[point_index] = BigFloat(_zero_mode(hessian(theta, Q, L,
                    phase; k, precision_bits)))
            catch error
                values[point_index] = nothing
                errors[point_index] = sprint(showerror, error)
            end
        end
        for interval_index in 1:(length(ks) - 1)
            low, high = BigFloat(ks[interval_index]), BigFloat(ks[interval_index + 1])
            eigen_low, eigen_high = values[interval_index], values[interval_index + 1]
            if eigen_low === nothing || eigen_high === nothing
                error_parts = [message for message in
                    (errors[interval_index], errors[interval_index + 1]) if !isempty(message)]
                push!(attempts, (; phase_index, phase=Float64.(phase),
                    interval_index, k_homotopy_low=low, k_homotopy_high=high,
                    eigenvalue_at_k_homotopy_low=eigen_low,
                    eigenvalue_at_k_homotopy_high=eigen_high,
                    status=:evaluation_failed, k_homotopy_c=nothing,
                    refined_bracket_width=nothing, refined_residual=nothing,
                    refinement_iterations=0, precision_bits,
                    error=join(error_parts, " | ")))
            elseif signbit(eigen_low) == signbit(eigen_high)
                push!(attempts, (; phase_index, phase=Float64.(phase),
                    interval_index, k_homotopy_low=low, k_homotopy_high=high,
                    eigenvalue_at_k_homotopy_low=eigen_low,
                    eigenvalue_at_k_homotopy_high=eigen_high,
                    status=:rejected_no_crossing, k_homotopy_c=nothing,
                    refined_bracket_width=nothing, refined_residual=nothing,
                    refinement_iterations=0, precision_bits, error=""))
            else
                try
                    refined = _refine_zero_mode_crossing(theta, Q, L, phase,
                        low, high;
                        precision_bits, tolerance, max_iterations)
                    if refined.converged
                        direction = eigen_low < 0 ? :negative_to_positive :
                            :positive_to_negative
                        push!(candidates, ScanCandidate(BigFloat.(phase), low, high,
                            refined.k_homotopy_c, eigen_low, eigen_high, direction,
                            :homotopy_only, missing, missing, missing, missing))
                        status = :zero_mode_crossing_refined
                        error_message = ""
                    else
                        status = :crossing_refinement_incomplete
                        error_message = "bisection exhausted its iteration budget"
                    end
                    push!(attempts, (; phase_index, phase=Float64.(phase),
                        interval_index, k_homotopy_low=low,
                        k_homotopy_high=high,
                        eigenvalue_at_k_homotopy_low=eigen_low,
                        eigenvalue_at_k_homotopy_high=eigen_high,
                        status, k_homotopy_c=refined.k_homotopy_c,
                        refined_bracket_width=refined.bracket_width,
                        refined_residual=refined.residual,
                        refinement_iterations=refined.iterations,
                        precision_bits, error=error_message))
                catch error
                    push!(attempts, (; phase_index, phase=Float64.(phase),
                        interval_index, k_homotopy_low=low,
                        k_homotopy_high=high,
                        eigenvalue_at_k_homotopy_low=eigen_low,
                        eigenvalue_at_k_homotopy_high=eigen_high,
                        status=:crossing_refinement_failed,
                        k_homotopy_c=nothing,
                        refined_bracket_width=nothing, refined_residual=nothing,
                        refinement_iterations=0, precision_bits,
                        error=sprint(showerror, error)))
                end
            end
        end
    end
    success_count = count(attempt ->
        attempt.status === :zero_mode_crossing_refined, attempts)
    rejected_count = count(attempt -> attempt.status === :rejected_no_crossing, attempts)
    failure_count = length(attempts) - success_count - rejected_count
    (; candidates, attempts, phase_count=length(phase_list),
       k_point_count=length(ks), interval_denominator,
       success_count, rejected_count, failure_count,
       coverage_status=length(attempts) == interval_denominator ? :complete : :incomplete,
       scale_status=:homotopy_only)
end

function scan(theta, Q, L; k_grid=range(0.5, 1.5; length=101), kwargs...)
    scan_detailed(theta, Q, L; k_homotopy_grid=k_grid, kwargs...).candidates
end

end
