#!/usr/bin/env julia

"""Bounded phase and volume-detuning search for low-dimensional potentials.

The volume parameter is a mathematical amplitude homotopy.  It is deliberately
labelled `homotopy_only`: this script does not transform divisor volumes,
kinetic data, or instanton units and therefore cannot produce a physical
inflation candidate.
"""
module PhaseVolumeDetuningScan

using LinearAlgebra
using GenericLinearAlgebra
using Printf

export ScanCandidate, phase_vectors, potential, hessian, refine_catastrophe,
    scan

struct ScanCandidate{T}
    phase::Vector{T}
    k_low::T
    k_high::T
    k_c::T
    eigenvalue_low::T
    eigenvalue_high::T
    catastrophe_type::Symbol
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
    result = Vector{Vector{Float64}}()
    push!(result, zeros(Float64, n))
    for i in 1:n, value in values
        phase = zeros(Float64, n)
        phase[i] = value
        push!(result, phase)
    end
    pairs = 0
    for i in 1:n-1, j in i+1:n, value_i in values, value_j in values
        pairs == pair_limit && return result
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
    T(2) * T(π)^2 .* (transpose(Q) * (weighted .* Q))
end

function _zero_mode(h)
    values = eigvals(Symmetric(h))
    index = argmin(abs.(values))
    values[index]
end

"""Refine a sign-changing zero mode by bounded bisection at arbitrary precision."""
function refine_catastrophe(theta, Q, L, phase, k_low, k_high;
        precision_bits::Integer=256, tolerance::Real=1e-30, max_iterations::Integer=160)
    lo, hi = BigFloat(k_low), BigFloat(k_high)
    flo = BigFloat(_zero_mode(hessian(theta, Q, L, phase; k=lo,
        precision_bits=precision_bits)))
    fhi = BigFloat(_zero_mode(hessian(theta, Q, L, phase; k=hi,
        precision_bits=precision_bits)))
    signbit(flo) == signbit(fhi) &&
        throw(ArgumentError("catastrophe bracket must straddle a zero mode"))
    for _ in 1:max_iterations
        mid = (lo + hi) / 2
        fmid = BigFloat(_zero_mode(hessian(theta, Q, L, phase; k=mid,
            precision_bits=precision_bits)))
        if abs(hi - lo) <= BigFloat(tolerance)
            return mid
        elseif signbit(flo) == signbit(fmid)
            lo, flo = mid, fmid
        else
            hi = mid
        end
    end
    (lo + hi) / 2
end

function scan(theta, Q, L; k_grid=range(0.5, 1.5; length=101),
        phases=phase_vectors(size(Q, 1)), precision_bits::Integer=256,
        tolerance::Real=1e-12)
    ks = collect(k_grid)
    isempty(ks) || all(diff(ks) .> 0) ||
        throw(ArgumentError("k_grid must be strictly increasing"))
    all(>(0), ks) || throw(ArgumentError("k_grid must contain positive values"))
    candidates = ScanCandidate{BigFloat}[]
    for phase in phases
        values = [_zero_mode(hessian(theta, Q, L, phase; k,
            precision_bits=precision_bits)) for k in ks]
        for index in 1:length(ks)-1
            signbit(values[index]) == signbit(values[index + 1]) && continue
            low, high = BigFloat(ks[index]), BigFloat(ks[index + 1])
            critical = refine_catastrophe(theta, Q, L, phase, low, high;
                precision_bits, tolerance)
            kind = values[index] < 0 ? :fold : :reverse_fold
            push!(candidates, ScanCandidate(BigFloat.(phase), low, high, critical,
                BigFloat(values[index]), BigFloat(values[index + 1]), kind,
                :homotopy_only, missing, missing, missing, missing))
        end
    end
    candidates
end

end
