"""
    CYAxiverse.generate
This is where most of the functions are defined.

"""
module generate

using HDF5
using LinearAlgebra
using ArbNumerics, Tullio, LoopVectorization, Nemo, SparseArrays, NormalForms, IntervalArithmetic, StaticArrays
using GenericLinearAlgebra
using Distributions
using Random: rand!
using TimerOutputs

using ..filestructure: cyax_file, minfile, present_dir, geom_dir_read, paths_cy
using ..read: potential, vacua_jlm
using ..minimizer: minimize, subspace_minimize, critical_points, minima_lattice

using ..structs: GeometryIndex, LQLinearlyIndependent, Projector, CanonicalQBasis, ProjectedQ, AxionPotential, MyTree, AxionSpectrum, PhysicalAxionSpectrum, QuarticComponentDiagnostics, QuarticDiagnostics, MassBasisDiagnostics, InstantonScaleBlock, PerturbativeSplitDiagnostics, InstantonHierarchyDiagnostics, SpectrumWindowDiagnostics, Canonicalα, RationalQSNF, Min_JLM_1D, Min_JLM_ND, Min_JLM_Square, BasisSNF

"""Validated subset of a minimizer result used by the legacy MK workflow."""
struct _MKMinimizeResult{X, V}
    xmin::X
    Vmin_log::V
end

#################
### Constant ####
#################

"""
    constants()

Loads constants:\n
- Reduced Planck Mass = 2.435 × 10^18
- Hubble = 2.13 × 0.7 × 10^-33
- log2π = log10(2π)
as `Dict{String,ArbFloat}`\n
#Examples
```julia-repl
julia> const_data = CYAxiverse.generate.constants()
Dict{String, ArbNumerics.ArbFloat{128}} with 3 entries:
  "MPlanck" => 2435000000000000000.0
  "log2π"   => 0.7981798683581150521959557408991
  "Hubble"  => 1.490999999999999999287243983194e-33
```
"""
function constants()
    mplanck_r = ArbFloat("2.435e18")
    hubble = ArbFloat("2.13") * ArbFloat("0.7") * ArbFloat("1e-33")
    log2pi = ArbFloat(log10(2π))
    return Dict("MPlanck" => mplanck_r, "Hubble" => hubble, "log2π" => log2pi)
end


###############################
##### Pseudo-Geometric data ###
###############################

"""
    pseudo_Q(h11,tri,cy=1)

Randomly generates an instanton charge matrix that takes the same form as those found in the KS Axiverse, namely `I(h11)` with 4 randomly filled rows and the cross-terms, i.e. an h11+4+C(h11+4,2) × h11 integer matrix.\n
#Examples
```julia-repl
julia> CYAxiverse.generate.pseudo_Q(4,10,1)
36×4 Matrix{Int64}:
  1   0   0   0
  0   1   0   0
  0   0   1   0
  0   0   0   1
  1   4  -3   5
 -5  -4  -2   4
  4   5   3  -2
 -5   2  -3  -3
  ⋮
 -9  -9  -5   6
  0  -6   1   7
  9   3   6   1
```
"""
function pseudo_Q(h11::Int, tri::Int, cy::Int=1)
    Q = vcat(Matrix{Int}(I(h11)), rand(-5:5, 4, h11))
    nrows = size(Q, 1)
    # Pre-allocate cross-terms to avoid intermediate vcat/hcat allocations
    ncross = binomial(nrows, 2)
    cross_terms = Matrix{Int}(undef, ncross, h11)
    idx = 1
    @inbounds for i in 1:nrows-1, j in i+1:nrows
        @views cross_terms[idx, :] .= Q[i, :] .- Q[j, :]
        idx += 1
    end
    return vcat(Q, cross_terms)
end

"""
    LogShiftedDerivativeWorkspace

Reusable Float64 buffers for evaluating a log-shifted axion potential. The
stored amplitudes represent the potential divided by `10^log_shift`; this
keeps hierarchically suppressed instantons finite during screening.
"""
mutable struct LogShiftedDerivativeWorkspace
    amplitudes::Vector{Float64}
    gradient::Vector{Float64}
    hessian::Matrix{Float64}
    log_shift::Float64
end

"""
    StructuredChargeRepresentation

Validated base-plus-pairwise representation of a canonical charge matrix.
The first `base_count` columns are retained as base charges; each later
column is recorded as an oriented difference of two base columns in the
original instanton order.  `validated=false` carries an explicit fallback
reason and must be evaluated with the generic charge path.
"""
struct StructuredChargeRepresentation
    base_Q::Matrix{Int}
    pair_i::Vector{Int}
    pair_j::Vector{Int}
    pair_orientation::Vector{Int}
    L::Matrix{Float64}
    base_count::Int
    validated::Bool
    fallback_reason::String
end

"""Reusable buffers for a validated structured charge evaluator."""
mutable struct StructuredLogShiftedDerivativeWorkspace
    representation::StructuredChargeRepresentation
    amplitudes::Vector{Float64}
    gradient::Vector{Float64}
    hessian::Matrix{Float64}
    base_phases::Vector{Float64}
    base_Q_float::Matrix{Float64}
    pair_gradient_coeff::Vector{Float64}
    pair_hessian_coeff::Matrix{Float64}
    pair_hessian_product::Matrix{Float64}
    log_shift::Float64
end

"""Prepared structured evaluator with an explicit generic fallback."""
struct StructuredChargeEvaluator{W}
    representation::StructuredChargeRepresentation
    workspace::W
    uses_generic_fallback::Bool
end

"""Hash one integer charge vector for collision-filtered structure lookup."""
function _charge_vector_hash(Q::AbstractMatrix{Int}, column::Int)
    value = UInt(0xcbf29ce484222325)
    @inbounds for row in axes(Q, 1)
        value ⊻= reinterpret(UInt, Int64(Q[row, column]))
        value *= UInt(0x100000001b3)
    end
    value
end

"""Hash an oriented difference of two base charge columns."""
function _charge_difference_hash(Q::AbstractMatrix{Int}, i::Int, j::Int,
        orientation::Int)
    value = UInt(0xcbf29ce484222325)
    @inbounds for row in axes(Q, 1)
        difference = orientation * (Q[row, j] - Q[row, i])
        value ⊻= reinterpret(UInt, Int64(difference))
        value *= UInt(0x100000001b3)
    end
    value
end

"""Build and validate the observed base-plus-pairwise charge structure."""
function structured_charge_representation(Q::AbstractMatrix{Int},
        L::AbstractMatrix{Float64}; base_count::Int=size(Q, 1) + 4)
    q = Q isa Matrix{Int} ? Q : Matrix{Int}(Q)
    l = L isa Matrix{Float64} ? L : Matrix{Float64}(L)
    h11 = size(q, 1)
    n_instantons = size(q, 2)
    empty_i = Int[]
    empty_j = Int[]
    empty_orientation = Int[]
    fallback = reason -> StructuredChargeRepresentation(
        h11 == 0 ? zeros(Int, 0, 0) : q[:, 1:min(base_count, n_instantons)],
        empty_i, empty_j, empty_orientation, l, min(base_count, n_instantons),
        false, reason)

    size(l, 1) == 2 || return fallback("L must have two rows")
    size(l, 2) == n_instantons || return fallback("Q and L have different instanton counts")
    all(isfinite, l) || return fallback("L contains non-finite values")
    base_count >= 1 || return fallback("base_count must be positive")
    base_count <= n_instantons || return fallback("not enough columns for the base set")
    expected = base_count + binomial(base_count, 2)
    n_instantons == expected ||
        return fallback("instanton count is not base plus all pairwise differences")

    base_Q = q[:, 1:base_count]
    buckets = Dict{UInt,Vector{NTuple{3,Int}}}()
    for i in 1:(base_count - 1), j in (i + 1):base_count
        for orientation in (-1, 1)
            key = _charge_difference_hash(base_Q, i, j, orientation)
            push!(get!(buckets, key, NTuple{3,Int}[]),
                (i, j, orientation))
        end
    end

    pair_i = Vector{Int}(undef, n_instantons - base_count)
    pair_j = Vector{Int}(undef, n_instantons - base_count)
    pair_orientation = Vector{Int}(undef, n_instantons - base_count)
    for column in (base_count + 1):n_instantons
        candidates = get(buckets, _charge_vector_hash(q, column), nothing)
        matched = false
        if candidates !== nothing
            for (i, j, orientation) in candidates
                matches = true
                @inbounds for row in axes(q, 1)
                    matches &= q[row, column] ==
                        orientation * (base_Q[row, j] - base_Q[row, i])
                end
                if matches
                    offset = column - base_count
                    pair_i[offset] = i
                    pair_j[offset] = j
                    pair_orientation[offset] = orientation
                    matched = true
                    break
                end
            end
        end
        matched || return fallback(string("charge column ", column,
            " is not a signed base-column difference"))
    end
    StructuredChargeRepresentation(base_Q, pair_i, pair_j,
        pair_orientation, l, base_count, true, "")
end

"""
    structured_charge_evaluator(Q, L; base_count=size(Q, 1) + 4)

Prepare a validated structured evaluator for the observed charge corpus.  If
the shape or charge proof fails, the returned evaluator retains the failed
representation and uses the generic log-shifted derivative workspace.
"""
function structured_charge_evaluator(Q::AbstractMatrix{Int},
        L::AbstractMatrix{Float64}; base_count::Int=size(Q, 1) + 4)
    representation = structured_charge_representation(Q, L; base_count)
    if representation.validated
        workspace = structured_logshifted_derivative_workspace(representation)
        return StructuredChargeEvaluator(representation, workspace, false)
    end
    StructuredChargeEvaluator(representation,
        logshifted_derivative_workspace(Q, L), true)
end

"""Prepare reusable buffers for a validated structured charge representation."""
function structured_logshifted_derivative_workspace(
        representation::StructuredChargeRepresentation)
    representation.validated || throw(ArgumentError(string(
        "cannot prepare an unvalidated structured representation: ",
        representation.fallback_reason)))
    l = representation.L
    log_shift = maximum(@view l[2, :])
    amplitudes = @view(l[1, :]) .* 10.0 .^ (@view(l[2, :]) .- log_shift)
    h11 = size(representation.base_Q, 1)
    StructuredLogShiftedDerivativeWorkspace(representation, amplitudes,
        zeros(Float64, h11), zeros(Float64, h11, h11),
        zeros(Float64, representation.base_count),
        Matrix{Float64}(representation.base_Q),
        zeros(Float64, representation.base_count),
        zeros(Float64, representation.base_count, representation.base_count),
        zeros(Float64, h11, representation.base_count), log_shift)
end

"""Evaluate a validated structured charge potential in place."""
function structured_logshifted_derivatives!(
        workspace::StructuredLogShiftedDerivativeWorkspace,
        theta::AbstractVector{<:Real})
    representation = workspace.representation
    h11 = size(representation.base_Q, 1)
    length(theta) == h11 || throw(DimensionMismatch(
        "theta must have one entry per axion"))
    gradient = workspace.gradient
    hessian = workspace.hessian
    fill!(gradient, 0.0)
    fill!(hessian, 0.0)
    pair_gradient_coeff = workspace.pair_gradient_coeff
    pair_hessian_coeff = workspace.pair_hessian_coeff
    fill!(pair_gradient_coeff, 0.0)
    fill!(pair_hessian_coeff, 0.0)
    value = 0.0
    two_pi = 2π
    four_pi_squared = (2π)^2
    base_Q = representation.base_Q
    amplitudes = workspace.amplitudes
    base_phases = workspace.base_phases

    @inbounds for base_index in 1:representation.base_count
        phase = 0.0
        for row in 1:h11
            phase += base_Q[row, base_index] * theta[row]
        end
        base_phases[base_index] = phase
        amplitude = amplitudes[base_index]
        sine = sin(two_pi * phase)
        cosine = cos(two_pi * phase)
        value += amplitude * (1.0 - cosine)
        gradient_scale = two_pi * amplitude * sine
        hessian_scale = four_pi_squared * amplitude * cosine
        for row in 1:h11
            charge_row = base_Q[row, base_index]
            gradient[row] += gradient_scale * charge_row
            for column in 1:h11
                hessian[row, column] += hessian_scale * charge_row *
                    base_Q[column, base_index]
            end
        end
    end

    @inbounds for pair_index in eachindex(representation.pair_i)
        i = representation.pair_i[pair_index]
        j = representation.pair_j[pair_index]
        orientation = representation.pair_orientation[pair_index]
        phase = orientation * (base_phases[j] - base_phases[i])
        amplitude = amplitudes[representation.base_count + pair_index]
        sine = sin(two_pi * phase)
        cosine = cos(two_pi * phase)
        value += amplitude * (1.0 - cosine)
        gradient_scale = two_pi * amplitude * sine
        hessian_scale = four_pi_squared * amplitude * cosine
        pair_gradient_coeff[j] += orientation * gradient_scale
        pair_gradient_coeff[i] -= orientation * gradient_scale
        pair_hessian_coeff[i, i] += hessian_scale
        pair_hessian_coeff[j, j] += hessian_scale
        pair_hessian_coeff[i, j] -= hessian_scale
        pair_hessian_coeff[j, i] -= hessian_scale
    end
    base_Q_float = workspace.base_Q_float
    LinearAlgebra.mul!(gradient, base_Q_float, pair_gradient_coeff, 1.0, 1.0)
    LinearAlgebra.mul!(workspace.pair_hessian_product, base_Q_float,
        pair_hessian_coeff)
    LinearAlgebra.mul!(hessian, workspace.pair_hessian_product,
        transpose(base_Q_float), 1.0, 1.0)
    (; value, gradient, hessian, log_shift=workspace.log_shift)
end

"""Use the generic evaluator when charge-structure validation fails."""
function structured_logshifted_derivatives!(
        workspace::LogShiftedDerivativeWorkspace, theta::AbstractVector{<:Real},
        Q::AbstractMatrix{Int})
    logshifted_derivatives!(workspace, theta, Q)
end

"""Evaluate a prepared structured evaluator, including its generic fallback."""
function structured_logshifted_derivatives!(
        evaluator::StructuredChargeEvaluator{LogShiftedDerivativeWorkspace},
        theta::AbstractVector{<:Real}, Q::AbstractMatrix{Int})
    logshifted_derivatives!(evaluator.workspace, theta, Q)
end

function structured_logshifted_derivatives!(
        evaluator::StructuredChargeEvaluator{StructuredLogShiftedDerivativeWorkspace},
        theta::AbstractVector{<:Real}, Q::AbstractMatrix{Int})
    structured_logshifted_derivatives!(evaluator.workspace, theta)
end

"""
    logshifted_derivative_workspace(Q, L)

Prepare reusable derivative buffers for `Q :: h11 × n` and
`L :: 2 × n`. `L[2, :]` is interpreted as base-10 log scale and `L[1, :]`
as the signed coefficient.
"""
function logshifted_derivative_workspace(Q::AbstractMatrix{Int},
        L::AbstractMatrix{Float64})
    size(L, 1) == 2 || throw(DimensionMismatch("L must have two rows"))
    size(Q, 2) == size(L, 2) ||
        throw(DimensionMismatch("Q and L must have the same instanton count"))
    all(isfinite, L) || throw(ArgumentError("L contains non-finite values"))
    log_shift = maximum(@view L[2, :])
    amplitudes = @view(L[1, :]) .* 10.0 .^ (@view(L[2, :]) .- log_shift)
    h11 = size(Q, 1)
    LogShiftedDerivativeWorkspace(amplitudes, zeros(Float64, h11),
        zeros(Float64, h11, h11), log_shift)
end

"""
    logshifted_derivatives!(workspace, theta, Q)

Evaluate the log-shifted value, gradient, and Hessian in place. The returned
named tuple borrows `workspace.gradient` and `workspace.hessian`; callers must
consume or copy them before the next call. The normalized value is related to
the physical potential by `V = 10^workspace.log_shift * value`.
"""
function logshifted_derivatives!(workspace::LogShiftedDerivativeWorkspace,
        theta::AbstractVector{<:Real}, Q::AbstractMatrix{Int})
    h11 = length(workspace.gradient)
    length(theta) == h11 ||
        throw(DimensionMismatch("theta must have one entry per axion"))
    size(Q, 1) == h11 || throw(DimensionMismatch("Q has the wrong axion count"))
    size(Q, 2) == length(workspace.amplitudes) ||
        throw(DimensionMismatch("Q has the wrong instanton count"))
    gradient = workspace.gradient
    hessian = workspace.hessian
    fill!(gradient, 0.0)
    fill!(hessian, 0.0)
    value = 0.0
    two_pi = 2π
    four_pi_squared = (2π)^2
    @inbounds for a in axes(Q, 2)
        phase = 0.0
        for i in axes(Q, 1)
            phase += Q[i, a] * theta[i]
        end
        amplitude = workspace.amplitudes[a]
        sine = sin(two_pi * phase)
        cosine = cos(two_pi * phase)
        value += amplitude * (1.0 - cosine)
        gradient_scale = two_pi * amplitude * sine
        hessian_scale = four_pi_squared * amplitude * cosine
        for i in axes(Q, 1)
            charge_i = Q[i, a]
            gradient[i] += gradient_scale * charge_i
            for j in axes(Q, 1)
                hessian[i, j] += hessian_scale * charge_i * Q[j, a]
            end
        end
    end
    (; value, gradient, hessian, log_shift=workspace.log_shift)
end

"""
    pseudo_K(h11,tri,cy=1)

Randomly generates an h11 × h11 Hermitian matrix with positive definite eigenvalues. \n
#Examples
```julia-repl
julia> K = CYAxiverse.generate.pseudo_K(4,10,1)
4×4 Hermitian{Float64, Matrix{Float64}}:
 2.64578  2.61012  0.91203  2.27339
 2.61012  3.89684  2.22451  1.93356
 0.91203  2.22451  2.94717  1.58126
 2.27339  1.93356  1.58126  4.85208

julia> eigen(K).values
4-element Vector{Float64}:
 0.17629073145135896
 1.8632009739875723
 2.7425362219513487
 9.559840749713599
```
"""
function pseudo_K(h11::Int, tri::Int, cy::Int=1)
    K = Matrix{Float64}(undef, h11, h11)
    while true
        rand!(K)
        K .= 2.0 .* (K .+ K') .+ 2.0 .* I(h11)
        # Fast check for positive definiteness via Cholesky instead of full eigen decomposition
        cholfact = cholesky(Hermitian(K); check = false)
        if issuccess(cholfact)
            return Hermitian(K)
        end
    end
end

"""
    pseudo_L(h11,tri,cy=1;log=true)

Randomly generates a h11+4+C(h11+4,2)-length hierarchical list of instanton scales, similar to those found in the KS Axiverse.  Option for (sign,log10) or full precision.\n
#Examples
```julia-repl
julia> CYAxiverse.generate.pseudo_L(4,10)
36×2 Matrix{Float64}:
  1.0     0.0
  1.0    -4.0
  1.0    -8.0
  1.0   -12.0
  1.0   -16.0
  1.0   -20.0
  1.0   -24.0
  1.0   -28.0
 -1.0   -29.4916
  1.0   -33.8515
  ⋮
  1.0  -133.665
 -1.0  -138.951

julia> CYAxiverse.generate.pseudo_L(4,10,log=false)
36-element Vector{ArbNumerics.ArbFloat}:
  1.0
  0.0001
  1.0e-8
  1.0e-12
  1.0e-16
  1.0e-20
  1.0e-24
  1.0e-28
 -1.462574279422558833057690597964e-31
 -2.381595397961591074099629406235e-34
  ⋮
  3.796809523142314798130344022481e-134
 -3.173000613781491329619833894919e-138
```
"""
function pseudo_L(h11::Int,tri::Int,cy::Int=1;log::Bool=true)
    L1::Matrix{Float64} = [1. 0.]
    L2::Matrix{Float64} = vcat([[1. -(4. *(j-1.))] for j=2.:h11+4.]...)
    L3::Matrix{Float64} = vcat([[sign(rand(Uniform(-100. *h11,100. *h11))) -(4*(j-1))+log10(abs(rand(Uniform(-100. *h11,100. *h11))))]
     for j=h11+5:h11+4+binomial(h11+4,2)]...)
    L4::Matrix{Float64} = @.(log10(abs(L3)))
    L::Matrix{Float64} = vcat(L1,L2,L3)
    L = hcat(sign.(L[:,1]), log10.(abs.(L[:,1])) .+ L[:,2])
    if log == 1
        return L
    else
        Ltemp::Vector{ArbFloat} = ArbFloat.(L[:,1]) .* ArbFloat(10.) .^ ArbFloat.(L[:,2])
        return Ltemp
    end
end

function V(x, L::Matrix{Float64}, Q::Matrix)
    @assert size(L, 2) == 2
    Λ = L[:, 1] .* 10. .^ L[:, 2]
    sum(Λ' * (1. .- cos.(Q' * x)))
end
function jacobian(x, L::Matrix{Float64}, Q::Matrix)
    Λ = L[:, 1] .* 10. .^ L[:, 2]
    if size(Q, 1) == 1
        grad_temp = Λ' .* (Q .* sin.(x' * Q))
        grad = sum(grad_temp, dims = 2)
    else
        grad_temp = Λ' .* (Q .* sin.(sum(x .* Q, dims=1)))
        grad = sum(grad_temp, dims = 2)
        SVector{size(grad, 1)}(grad)
    end
end

function hessian(x, L::Matrix{Float64}, Q::Matrix)
    Λ = L[:, 1] .* 10. .^ L[:, 2]
    phases = vec(sum(x .* Q, dims=1))
    hessian = zeros(size(Q, 1), size(Q, 1))
    for i in axes(Q, 1), j in axes(Q, 1)
        if i >= j
            hessian[i, j] = sum(Λ .* Q[i, :] .* Q[j, :] .* cos.(phases))
        end
    end
    Hermitian(hessian + hessian' - Diagonal(hessian))
end

"""
    cubic(x, L, Q)

Return the rank-3 field-space derivative tensor
`∂ᵢ∂ⱼ∂ₖ V(x)` for
`V(x) = Σₐ Λₐ (1 - cos(Qₐ⋅x))`, where `Λ` is encoded by the signed
`(sign, log10(abs(Λ)))` columns of `L`.

For a phased potential, evaluate this function at `x + phase`.  In
particular, the tensor vanishes at the unphased origin but is generally
nonzero for arbitrary phases.
"""
function cubic(x::AbstractVector{<:Real}, L::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real})
    @assert size(L, 2) == 2
    @assert size(Q, 1) == size(L, 1) && size(Q, 2) == length(x)
    Λ = L[:, 1] .* 10.0 .^ L[:, 2]
    phases = Q * x
    n = size(Q, 2)
    C = zeros(promote_type(eltype(Λ), eltype(x), Float64), n, n, n)
    @inbounds for a in axes(Q, 1)
        coefficient = -Λ[a] * sin(phases[a])
        for i in 1:n, j in 1:n, k in 1:n
            C[i, j, k] += coefficient * Q[a, i] * Q[a, j] * Q[a, k]
        end
    end
    C
end

"""
    cubic(x, phase, L, Q)

Evaluate the same cubic tensor at `x + phase`.  This is a convenience for
phase-shifted cosine potentials; the returned tensor is in the coordinate
basis represented by `Q`.
"""
cubic(x::AbstractVector{<:Real}, phase::AbstractVector{<:Real}, L::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real}) =
    cubic(x .+ phase, L, Q)

function hessian_norm(x, Q::Matrix)
    hessian = zeros(size(Q, 1), size(Q, 1))
    if size(Q, 1) == 1
        for i in axes(Q, 1), j in axes(Q, 1)
            if i>=j
                hessian[i, j] = (transpose(@view(Q[i, :])) * @view(Q[j, :])) * cos.(x' * Q)[i]
            end
        end
        hessian = hessian + hessian' - Diagonal(hessian)
    elseif size(Q, 1) == size(Q, 2)
        for i in axes(Q, 1), j in axes(Q, 1)
            if i>=j
                hessian[i, j] = (transpose(@view(Q[i, :])) * @view(Q[j, :])) * cos.(x' * Q)[i]
            end
        end
        hessian = Hermitian(hessian + hessian' - Diagonal(hessian))
        # SMatrix{size(hessian, 1), size(hessian,2)}(hessian)
    else
        hessian = zeros(size(Q, 1), size(Q, 1), size(Q, 2))
        for i in axes(Q, 1), j in axes(Q, 1), k in axes(Q, 2)
            if i>=j
                hessian[i, j, k] = (transpose(@view(Q[i, :])) * @view(Q[j, :])) * cos.(x' * Q)[k]
            end
        end
        return hessian
    end
end
##############################
#### Computing Spectra #######
##############################

"""
    gauss_sum(z)

Computes the addition of 2 numbers in (natural) log-space using the definition [here](https://en.wikipedia.org/wiki/Gaussian_logarithm).\n
#Examples
```julia-repl
julia> CYAxiverse.generate.gauss_sum(10.)
10.000045398899218

julia> CYAxiverse.generate.gauss_sum(1000.)
1000.0
```
"""
function gauss_sum(z::Float64)
    log2 = log(2)
    if abs(z)>600.
        return 0.5*z +abs(0.5*z)
    else
        return log2 + 0.5*z + log(cosh(0.5*z))
    end
end
"""
    gauss_diff(z)

Computes the difference of 2 numbers in (natural) log-space using the definition [here](https://en.wikipedia.org/wiki/Gaussian_logarithm).\n
#Examples
```julia-repl
julia> CYAxiverse.generate.gauss_diff(10.)
9.99995459903963

julia> CYAxiverse.generate.gauss_diff(1000.)
1000.0
```
"""
function gauss_diff(z::Float64)
    log2 = log(2)
    if abs(z)>600.
        return 0.5*z +abs(0.5*z)
    else
        return log2 + 0.5*z + log(abs(sinh(0.5*z)))
    end
end

"""
    gauss_log_split(sign, log)

Algorithm to compute Gaussian logarithms, as detailed [here](https://en.wikipedia.org/wiki/Gaussian_logarithm).\n
#Examples
```julia-repl
julia> CYAxiverse.generate.gauss_diff(10.)
9.99995459903963

julia> CYAxiverse.generate.gauss_diff(1000.)
1000.0
```
"""
function gauss_log_split(sb::Vector{Int},logb::Vector{Float64})
    # loga = log(|A|); logb = log(|B|); sa = sign(A); sb = sign(B)
    temp = hcat(sb,logb)
    temp = temp[sortperm(temp[:,2]),:]
    sb::Vector{Int} = temp[:,1]
    logb::Vector{Float64} = temp[:,2]
    i = 1
    sa = sb[i]
    loga = logb[i]
    while i < size(sb,1)
#         println(i)
#         println([sa,sb[i+1],loga, logb[i+1]])
        if (sa==0 && sb[i+1]==0) ## A == B == 0
        elseif sa==0 ## A==0 --> B
            sa = sb[i+1]
            loga = logb[i+1]
        elseif sb[i+1]==0 ## B==0 --> A
        elseif (sa<0 && sb[i+1]>0) ## B-A
            if loga<logb[i+1] ## |A|<|B|
                sa = 1
                loga = logb[i+1]+gauss_diff(loga-logb[i+1])
            elseif loga == logb[i+1]
                sa = 0
                loga = 0
            else ## |A|>|B|
                sa = -1
                loga = logb[i+1]+gauss_diff(loga-logb[i+1])
            end
        elseif (sa>0 && sb[i+1]<0) ## A-B
            if loga>logb[i+1] ## |A|>|B|
                sa =1
                loga = loga+gauss_diff(-loga+logb[i+1])
            elseif loga == logb[i+1]
                sa = 0
                loga = 0
            else ## |A|<|B|
                sa = -1
                loga = loga+gauss_diff(-loga+logb[i+1])
            end
        elseif (sa<0 && sb[i+1]<0) ## -A-B
            sa = -1
            loga = loga + gauss_sum(-loga+logb[i+1])
        else ## A+B
            sa = 1
            loga = loga + gauss_sum(-loga+logb[i+1])
        end
        i+=1
    end
    return Int(sa), Float64(loga)
end

function gauss_log(sb,logb)
    if size(sb[sb .== 0.],1) == size(sb,1)
        return 0,-Inf
    elseif size(sb[sb .> 0.],1) == 0
        test = -1
#     elseif size(sb[sb .< 0.],1) == 0
#         test = 1
    else
        test = 1
    end
    temp = hcat(sb,logb)
    signed_mask::Vector{Bool} = temp[:,1] .== test
    temp1 = temp[signed_mask,:]
    temp1 = temp1[sortperm(temp1[:,2]),:]
    sb::Vector{Int} = temp1[:,1]
    logb::Vector{Float64} = temp1[:,2]
    sa1::Int,loga1::Float64 = gauss_log_split(sb,logb)
    if size(temp1,1) != size(temp,1)
        signed_mask = Bool.(true .- signed_mask)
        temp2 = temp[signed_mask,:]
        temp2 = temp2[sortperm(temp2[:,2]),:]
        sba::Vector{Int} = temp2[:,1]
        logba::Vector{Float64} = temp2[:,2]
        sa2::Int,loga2::Float64 = gauss_log_split(sba,logba)
        sa3::Vector{Int} = vcat(sa1,sa2)
        loga3::Vector{Float64} = vcat(loga1,loga2)
        sa::Int,loga = gauss_log_split(sa3,loga3)
        return Int(sa),Float64(loga)
    else
        return Int(sa1), Float64(loga1)
    end
end

function V(x; L, Q)
    potential = dot(L, (1. - cos(Q * x)))
end

function _hp_selected_potential(L::AbstractMatrix{<:Real},
        Q::AbstractMatrix{<:Integer}, selection::Symbol)
    selection in (:raw, :hp_effective) ||
        throw(ArgumentError("selection must be :raw or :hp_effective"))
    size(L, 1) == 2 || throw(DimensionMismatch("L must be 2 × N: sign and log10 scale by column"))
    size(Q, 2) == size(L, 2) || throw(DimensionMismatch("L and Q must contain the same number of instantons"))
    Lnative = Matrix{Float64}(L)
    Qnative = Matrix{Int}(Q)
    selection === :raw && return Lnative, Qnative

    selected = LQtildebar(Lnative, Qnative)
    Matrix{Float64}(selected["Lhat"]), Matrix{Int}(selected["Qhat"])
end

"""
    hp_spectrum(K, L, Q; prec=1000, quartics=true, selection=:raw)

Uses potential data generated by CYTools (or randomly generated) to compute
axion spectra—masses, quartic couplings, and decay constants—at the requested
arbitrary precision. Hessian diagonalization and the signed quartic
contractions are completed at `prec` before their reported logarithms are
converted to `Float64`.

`L` is a `2 × N` matrix whose columns contain the instanton sign and
`log10` scale, and `Q` is an `h11 × N` charge matrix with one instanton per
column. Set `quartics=false` for a mass/sign/kinetic-decay calculation that
skips the quartic contractions and returns empty quartic fields.
`selection=:raw` uses all supplied instantons; `selection=:hp_effective`
applies the legacy `LQtildebar` hierarchy selection before diagonalization.

#Examples
```julia-repl
julia> pot_data = CYAxiverse.read.potential(4,10,1)
julia> hp_spec_data = CYAxiverse.generate.hp_spectrum(4, 10, 1)
Dict{String, Any} with 12 entries:
    "msign" => []
    "m" => []
    "fK" => []
    "fpert" => []
    "λselfsign" => []
    "λself" => []
    "λ31_i" => []
    "λ31sign" => []
    "λ31" => []
    "λ22_i" => []
    "λ22sign" => []
    "λ22" => []
```
"""
function hp_spectrum(K::Hermitian{Float64, Matrix{Float64}},
        L::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Integer};
        prec::Int=1000, quartics::Bool=true, selection::Symbol=:raw)
    L, Q = _hp_selected_potential(L, Q, selection)
    h11::Int = size(K, 1)
    size(K, 2) == h11 || throw(DimensionMismatch("K must be square"))
    size(L, 1) == 2 || throw(DimensionMismatch("L must be 2 × N: sign and log10 scale by column"))
    size(Q, 1) == h11 || throw(DimensionMismatch("Q must be h11 × N: one charge vector per column"))
    size(Q, 2) == size(L, 2) || throw(DimensionMismatch("L and Q must contain the same number of instantons"))

    setprecision(ArbFloat; digits=prec)
    ninstanton::Int = size(Q, 2)
    Lh::Vector{ArbFloat} = Vector{ArbFloat}(undef, ninstanton)
    ten = ArbFloat(10)
    @inbounds for instanton in 1:ninstanton
        Lh[instanton] = ArbFloat(L[1, instanton]) *
            ten^ArbFloat(L[2, instanton])
    end

    # Accumulate instanton rank-one contributions column by column. This keeps
    # the hot input traversal aligned with Julia's column-major storage.
    grad2::Matrix{ArbFloat} = zeros(ArbFloat, h11, h11)
    @inbounds for instanton in 1:ninstanton
        scale = Lh[instanton]
        for i in 1:h11
            qi = Q[i, instanton]
            for j in 1:i
                grad2[i, j] += scale * qi * Q[j, instanton]
            end
        end
    end
    hessfull = Hermitian(grad2 + transpose(grad2) - Diagonal(grad2))

    # Compute the generalized eigensystem at arbitrary precision.
    Ktest = Hermitian(ArbFloat.(Matrix(K)))
    Kfactor = cholesky(Ktest)
    kinetic_eigenvalues = eigvals!(Ktest)
    fK::Vector{Float64} = Float64.(log10.(sqrt.(kinetic_eigenvalues)))
    whitened_hessian = Hermitian(Kfactor.L \ Matrix(hessfull) / Kfactor.L')
    Vls::Vector{ArbFloat}, eigenvectors::Matrix{ArbFloat} = eigen(whitened_hessian)
    Hsign::Vector{Int64} = @.(sign(Vls))
    Hvals::Vector{Float64} = @.(log10(sqrt(abs(Vls))))

    constant_data = constants()
    mplanck_log = Float64(log10(constant_data["MPlanck"]))
    log2π = Float64(constant_data["log2π"])
    log10e = log10(exp(1))
    masses = Hvals .+ mplanck_log .+ 9 .+ log2π
    if !quartics
        vals = Hsign, masses, fK .+ mplanck_log .- log2π, Float64[], Int[], Float64[],
            zeros(Int, 4, 0), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[]
        keys = ["msign", "m", "fK", "fpert", "λselfsign", "λself", "λ31_i",
            "λ31sign", "λ31", "λ22_i", "λ22sign", "λ22"]
        return Dict(zip(keys, vals))
    end

    # Keep high-precision scales and contractions until the final conversion
    # to the legacy Float64 logarithmic output fields.
    Tls::Matrix{ArbFloat} = Kfactor.L' \ eigenvectors
    Qrows::Matrix{ArbFloat} = ArbFloat.(transpose(Q))
    QMs::Matrix{ArbFloat} = Qrows * Tls
    qindq31::Vector{NTuple{4, Int}} = [(x, x, x, y) for x=1:h11,y=1:h11 if x!=y]
    qindq22::Vector{NTuple{4, Int}} = [(x, x, y, y) for x=1:h11,y=1:h11 if x>y]
    quart31log::Vector{Float64} = zeros(Float64, length(qindq31))
    quart22log::Vector{Float64} = zeros(Float64, length(qindq22))
    quartdiaglog::Vector{Float64} = zeros(Float64, h11)
    quart31sign::Vector{Int} = zeros(Int, length(qindq31))
    quart22sign::Vector{Int} = zeros(Int, length(qindq22))
    quartdiagsign::Vector{Int} = zeros(Int, h11)
    quart31values = zeros(ArbFloat, length(qindq31))
    quart22values = zeros(ArbFloat, length(qindq22))
    quartdiagvalues = zeros(ArbFloat, h11)
    qmode_squared = zeros(ArbFloat, h11)
    qmode_cubed = zeros(ArbFloat, h11)

    signed_log(value::ArbFloat) = begin
        value_sign = Int(sign(value))
        value_sign, value_sign == 0 ? -Inf : Float64(log(abs(value)))
    end

    # Fuse the instanton loop across all quartic components. The previous
    # component-by-component sums repeatedly traversed Lh and allocated a
    # reduction state for every component.
    @inbounds for instanton in eachindex(Lh)
        scale = Lh[instanton]
        for i in 1:h11
            qmode = QMs[instanton, i]
            qmode_squared[i] = qmode^2
            qmode_cubed[i] = qmode_squared[i] * qmode
            quartdiagvalues[i] += scale * qmode_squared[i]^2
        end

        index31 = 1
        index22 = 1
        for i in 1:h11
            for j in 1:h11
                if i != j
                    quart31values[index31] += scale * qmode_cubed[i] * QMs[instanton, j]
                    index31 += 1
                end
            end
            for j in 1:i-1
                quart22values[index22] += scale * qmode_squared[i] * qmode_squared[j]
                index22 += 1
            end
        end
    end
    @inbounds for k in eachindex(qindq31)
        quart31sign[k], quart31log[k] = signed_log(quart31values[k])
    end
    @inbounds for k in eachindex(qindq22)
        quart22sign[k], quart22log[k] = signed_log(quart22values[k])
    end
    @inbounds for k in 1:h11
        quartdiagsign[k], quartdiaglog[k] = signed_log(quartdiagvalues[k])
    end

    fpert::Vector{Float64} = @.(Hvals + mplanck_log -
        (0.5 * quartdiaglog * log10e))
    qindq31_output = zeros(Int, 4, length(qindq31))
    qindq22_output = zeros(Int, 4, length(qindq22))
    @inbounds for column in eachindex(qindq31)
        for row in 1:4
            qindq31_output[row, column] = qindq31[column][row] - 1
        end
    end
    @inbounds for column in eachindex(qindq22)
        for row in 1:4
            qindq22_output[row, column] = qindq22[column][row] - 1
        end
    end
    vals = Hsign, masses, fK .+ mplanck_log .- log2π, fpert .- log2π,
        quartdiagsign, quartdiaglog .* log10e .+ 4 * log2π,
        qindq31_output, quart31sign,
        quart31log .* log10e .+ 4 * log2π,
        qindq22_output, quart22sign,
        quart22log .* log10e .+ 4 * log2π

    keys = ["msign", "m", "fK", "fpert", "λselfsign", "λself", "λ31_i",
        "λ31sign", "λ31", "λ22_i", "λ22sign", "λ22"]
    Dict(zip(keys, vals))
end

function hp_spectrum(h11::Int, tri::Int, cy::Int=1;
        prec::Int=1000, quartics::Bool=true,
        selection::Symbol=:hp_effective)
    pot_data = potential(h11,tri,cy);
    hp_spectrum(pot_data.K, pot_data.L, pot_data.Q;
        prec = prec, quartics = quartics, selection = selection)
end

function hp_spectrum(geom_idx::GeometryIndex;
        prec::Int=1000, quartics::Bool=true,
        selection::Symbol=:hp_effective)
    pot_data = potential(geom_idx);
    hp_spectrum(pot_data.K, pot_data.L, pot_data.Q;
        prec = prec, quartics = quartics, selection = selection)
end


"""
    hp_spectrum_save(h11, tri, cy; prec=1000, quartics=true,
        selection=:hp_effective, phase=zeros(h11))

Compute and persist the high-precision spectrum and the cubic interaction
tensor. The cubic tensor is evaluated in the charge basis selected by
`selection` and is stored under `spectrum/cubic/tensor`; the phase vector is
stored under `spectrum/cubic/phase`. The default `:hp_effective` selection
uses the legacy `LQtildebar` basis.
"""
function hp_spectrum_save(h11::Int, tri::Int, cy::Int=1; prec = 1000,
                          quartics::Bool=true,
                          selection::Symbol=:hp_effective,
                          phase::AbstractVector{<:Real}=zeros(Float64, h11))
    if h11 != 0
        pot_data = potential(h11, tri, cy)
        K = pot_data.K
        Ltilde, Qtilde = _hp_selected_potential(pot_data.L, pot_data.Q, selection)
        spectrum_data = hp_spectrum(K, Ltilde, Qtilde;
            prec = prec, quartics = quartics, selection = :raw)
        cubic_data = cubic(phase, Ltilde, Qtilde)

        h5open(cyax_file(h11, tri, cy), "r+") do file
            f2 = haskey(file, "spectrum") ? file["spectrum"] : create_group(file, "spectrum")
            f2a = haskey(f2, "quartdiag") ? f2["quartdiag"] : create_group(f2, "quartdiag")
            f2a["log10", deflate=9] = spectrum_data["λself"]
            f2a["sign", deflate=9] = spectrum_data["λselfsign"]
            f2e = haskey(f2, "decay") ? f2["decay"] : create_group(f2, "decay")
            f2e["fpert", deflate=9] = spectrum_data["fpert"]
            f2e["fK", deflate=9] = spectrum_data["fK"]

            f2b = haskey(f2, "quart31") ? f2["quart31"] : create_group(f2, "quart31")
            f2b["log10", deflate=9] = spectrum_data["λ31"]
            f2b["sign", deflate=9] = spectrum_data["λ31sign"]
            f2b["index", deflate=9] = spectrum_data["λ31_i"]

            f2c = haskey(f2, "quart22") ? f2["quart22"] : create_group(f2, "quart22")
            f2c["log10", deflate=9] = spectrum_data["λ22"]
            f2c["sign", deflate=9] = spectrum_data["λ22sign"]
            f2c["index", deflate=9] = spectrum_data["λ22_i"]

            f2d = haskey(f2, "masses") ? f2["masses"] : create_group(f2, "masses")
            f2d["log10", deflate=9] = spectrum_data["m"]
            f2d["sign", deflate=9] = spectrum_data["msign"]

            f2g = haskey(f2, "cubic") ? f2["cubic"] : create_group(f2, "cubic")
            f2g["tensor", deflate=9] = cubic_data
            f2g["phase", deflate=9] = Float64.(phase)
        end
    end

end


"""
    project_out(v::Vector)

Takes the direction to be projected out as input and returns a projector of the form

``\\Pi\\bigl(\\vec{v}\\bigr) = \\mathbb{1}_{h^{1,1}} - \\frac{\\bigl|\\vec{v}\\bigr\\rangle\\bigl\\langle\\vec{v}\\bigr|}{||\\vec{v}||^2}``
"""
function project_out(v::Vector{T} where T<:Union{Rational{Int64}, Integer})
    idd = Matrix{Rational}(I(size(v,1)))
    norm2 = dot(v,v)
    proj = 1 // norm2 * (v * v')
    proj = @.(ifelse(abs(proj) < 1e-5, zero(proj), proj))
    # TODO: #16 Need to remove floating point errors
    Projector(proj, idd - proj)
end

function project_out(projector::Matrix, v::Vector{Int})
    norm2 = dot(projector * v, projector * v)
    proj = 1. / norm2 * ((projector * v) * (v' * projector))
    projector = projector - proj
    @.(ifelse(abs(projector) < 1e-5, zero(projector), projector))
end

function project_out(v::Vector{Float64})
    idd = Matrix{Float64}(I(size(v,1)))
    norm2 = dot(v,v)
    proj = 1. /norm2 * (v * v')
    proj = @.(ifelse(abs(proj) < 1e-5, zero(proj), proj))
    idd_proj = idd - proj
    Projector(proj, @.(ifelse(abs(idd_proj) < 1e-5, zero(idd_proj), idd_proj)))
end

function project_out(projector::Matrix, v::Vector{Float64})
    norm2 = dot(projector * v, projector * v)
    proj = 1. / norm2 * ((projector * v) * (v' * projector))
    proj = @.(ifelse(abs(proj) < eps(), zero(proj), proj))
    idd_proj = projector - proj
    @.(ifelse(abs(idd_proj) < 1e-5, zero(idd_proj), idd_proj))
end

"""
    project_out(orth_basis::Matrix)

TBW
"""
function project_out(orth_basis::Matrix)
    projector = I(size(orth_basis, 1))
    for i in 1:size(orth_basis, 2)
        P = @view(orth_basis[:, i]) * transpose(@view(orth_basis[:, i]))
        projector -= P
    end
    @.(ifelse(abs(projector) < 1e-10, zero(projector), projector))
end

"""
    orth_basis(vec::Vector)

Uses the projector defined in [`project_out(v)`](@ref) to construct an orthonormal basis (same method as [scipy.linalg.orth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orth.html))
"""
function orth_basis(vec::Vector)
    proj = project_out(vec)
    #this is the scipy.linalg.orth function written out
    u, s, vh = svd(proj.Πperp,full=true)
    M, N = size(u,1), size(vh,2)
    rcond = eps() * max(M, N)
    tol = maximum(s) * rcond
    num = Int.(round(sum(s .> tol)))
    T = u[:, 1:num]
    @.(ifelse(abs(T) < tol, zero(T), T))
end

"""
    orth_basis(Q)
Takes a set of vectors (columns of `Q`) and constructs an orthonormal basis
"""
function orth_basis(Q::Matrix)
   #this is the scipy.linalg.orth function written out
   u, s, vh = svd(Q, full=true)
   M, N = size(u,1), size(vh,2)
   rcond = eps() * max(M, N)
   tol = maximum(s) * rcond
   num = Int.(round(sum(s .> tol)))
   T = u[:, 1:num]
   @.(ifelse(abs(T) < tol, zero(T), T))
end 

"""Build the sequential PQ frame and approximate masses in canonical space."""
function pq_canonical_frame(Qleading::Matrix{Float64}, Ltilde::Matrix{Float64})
    h11 = size(Qleading, 1)
    P = zeros(Float64, h11, h11)
    fapprox = zeros(Float64, h11)
    mapprox = zeros(Float64, h11)
    for i in 1:h11
        direction = copy(@view Qleading[i, :])
        for j in 1:i-1
            direction .-= dot(direction, @view(P[:, j])) .* @view(P[:, j])
        end
        direction_norm = norm(direction)
        @assert direction_norm > 0 "PQ-selected charges must be linearly independent"
        fapprox[i] = log10(1 / (2π * direction_norm^2))
        mapprox[i] = 0.5 * (Ltilde[2, i] - fapprox[i] - log10(2π))
        P[:, i] .= direction ./ direction_norm
    end
    return fapprox, mapprox, P
end

"""Accumulate the first `n` logarithmic magnitudes after an in-place sort."""
function logsum_sorted!(logs::Vector{Float64}, n::Int)
    n == 0 && return -Inf
    sort!(@view(logs[1:n]))
    total = logs[1]
    @inbounds for i in 2:n
        total += gauss_sum(logs[i] - total)
    end
    return total
end

"""Evaluate one signed PQ quartic with reusable Float64 log-space buffers."""
function pq_contracted_log!(positive_logs::Vector{Float64}, negative_logs::Vector{Float64}, scale_sign::Vector{Int}, scale_log::AbstractVector{Float64}, Qpq::Matrix{Float64}, exponents::NTuple{4, Int})
    npositive = 0
    nnegative = 0
    @inbounds for a in eachindex(scale_sign)
        term_sign = scale_sign[a]
        term_log = scale_log[a]
        for mode in exponents
            charge = Qpq[a, mode]
            if charge == 0
                term_sign = 0
                break
            end
            term_sign *= charge > 0 ? 1 : -1
            term_log += log(abs(charge))
        end
        if term_sign > 0
            npositive += 1
            positive_logs[npositive] = term_log
        elseif term_sign < 0
            nnegative += 1
            negative_logs[nnegative] = term_log
        end
    end

    positive_sum = logsum_sorted!(positive_logs, npositive)
    negative_sum = logsum_sorted!(negative_logs, nnegative)
    if npositive == 0
        return nnegative == 0 ? 0 : -1, negative_sum, negative_sum
    elseif nnegative == 0
        return 1, positive_sum, positive_sum
    elseif positive_sum > negative_sum
        return 1, positive_sum + gauss_diff(negative_sum - positive_sum), positive_sum + gauss_sum(negative_sum - positive_sum)
    elseif negative_sum > positive_sum
        return -1, negative_sum + gauss_diff(positive_sum - negative_sum), negative_sum + gauss_sum(positive_sum - negative_sum)
    end
    return 0, 0.0, positive_sum + log(2)
end

function leading_hessian_matrix_float64(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int})
    h11 = size(K, 1)
    scales = @view(L[1, :]) .* (10.0 .^ @view(L[2, :]))
    H = zeros(Float64, h11, h11)
    for a in eachindex(scales), i in 1:h11, j in 1:h11
        H[i, j] += scales[a] * Q[i, a] * Q[j, a]
    end
    Kfactor = cholesky(K)
    Hermitian(Kfactor.L \ H / Kfactor.L')
end

"""Build a Float64 seed Hessian after a common log-scale rescaling."""
function leading_hessian_matrix_float64_scaled(K::Hermitian{Float64, Matrix{Float64}},
        L::Matrix{Float64}, Q::Matrix{Int})
    h11 = size(K, 1)
    isempty(L) && return Hermitian(zeros(Float64, h11, h11))
    reference_log = maximum(@view L[2, :])
    floor_log = log10(floatmin(Float64))
    scales = Float64[
        L[1, a] * 10.0^max(L[2, a] - reference_log, floor_log)
        for a in axes(L, 2)
    ]
    H = zeros(Float64, h11, h11)
    for a in eachindex(scales), i in 1:h11, j in 1:h11
        H[i, j] += scales[a] * Q[i, a] * Q[j, a]
    end
    Kfactor = cholesky(K)
    Hermitian(Kfactor.L \ H / Kfactor.L')
end

"""Construct the PQ leading Hessian in canonical fields at arbitrary precision.

Only nonzero entries of each instanton charge vector are accumulated. This is
mathematically identical to a dense rank-one update but is essential for the
sparse charge matrices encountered at large h11.
"""
function high_precision_leading_hessian(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; prec::Int=1_000)
    setprecision(ArbFloat; digits=prec)
    T = typeof(ArbFloat(0))
    h11 = size(K, 1)
    scales = T.(L[1, :]) .* (T(10) .^ T.(L[2, :]))
    H = zeros(T, h11, h11)
    for a in eachindex(scales)
        support = Int[]
        for i in axes(Q, 1)
            Q[i, a] != 0 && push!(support, i)
        end
        for j in support, i in support
            H[i, j] += scales[a] * Q[i, a] * Q[j, a]
        end
    end
    Kfactor = cholesky(Hermitian(T.(Matrix(K))))
    Hermitian(Kfactor.L \ H / Kfactor.L'), Kfactor
end

function mass_basis_accuracy(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}, basis::Matrix{Float64})
    W = leading_hessian_matrix_float64(K, L, Q)
    Wbasis = W * basis
    eigenvalues = vec(sum(basis .* Wbasis; dims=1))
    scale = opnorm(W, Inf)
    eigenpair_residuals = [norm(@view(Wbasis[:, i]) .- eigenvalues[i] .* @view(basis[:, i])) / max(abs(eigenvalues[i]), eps(Float64) * scale) for i in axes(basis, 2)]
    nearest_relative_gaps = [minimum(abs(eigenvalues[i] - eigenvalues[j]) / max(abs(eigenvalues[i]), abs(eigenvalues[j]), eps(Float64) * scale) for j in axes(basis, 2) if j != i; init=Inf) for i in axes(basis, 2)]
    orthogonality_error = opnorm(basis' * basis - I, Inf)
    MassBasisDiagnostics(eigenpair_residuals, nearest_relative_gaps, orthogonality_error)
end

"""
    instanton_scale_blocks(L; gap_log10=1.0, min_block_size=1)

Sort instantons by descending `L[2, :]` and cluster adjacent entries whose
log-scale gap is at most `gap_log10`. A gap larger than the configured value
starts a new block. Small blocks are merged with the adjacent block across the
smaller boundary gap until every block has at least `min_block_size` entries.
The returned block indices are zero-based indices into the original `L`.

The result contains `blocks`, `sorted_indices`, `sorted_log_scales`,
`inter_block_gaps`, and a `diagnostics` named tuple. No scale is materialized,
so this routine remains safe for extreme log-scale spans.
"""
function instanton_scale_blocks(L::AbstractMatrix{<:Real};
        gap_log10::Real=1.0, min_block_size::Int=1)
    size(L, 1) >= 2 || throw(DimensionMismatch("L must have at least two rows: sign and log10 scale"))
    min_block_size >= 1 || throw(ArgumentError("min_block_size must be positive"))
    gap = Float64(gap_log10)
    isfinite(gap) && gap >= 0 || throw(ArgumentError("gap_log10 must be finite and nonnegative"))
    logs = Float64.(L[2, :])
    all(isfinite, logs) || throw(ArgumentError("L log10 scales must be finite"))
    sorted_positions = sortperm(logs; rev=true)
    sorted_logs = logs[sorted_positions]
    ranges = UnitRange{Int}[]
    start = 1
    for position in 1:max(length(sorted_logs) - 1, 0)
        if sorted_logs[position] - sorted_logs[position + 1] > gap
            push!(ranges, start:position)
            start = position + 1
        end
    end
    !isempty(sorted_logs) && push!(ranges, start:length(sorted_logs))

    # Preserve the clustering rule while preventing undersized boundary blocks.
    while length(ranges) > 1
        small = findfirst(range -> length(range) < min_block_size, ranges)
        small === nothing && break
        left_gap = small > 1 ? sorted_logs[first(ranges[small - 1])] -
            sorted_logs[last(ranges[small - 1]) + 1] : Inf
        right_gap = small < length(ranges) ? sorted_logs[last(ranges[small])] -
            sorted_logs[first(ranges[small + 1])] : Inf
        if left_gap <= right_gap
            ranges[small - 1] = first(ranges[small - 1]):last(ranges[small])
            deleteat!(ranges, small)
        else
            ranges[small] = first(ranges[small]):last(ranges[small + 1])
            deleteat!(ranges, small + 1)
        end
    end

    blocks = InstantonScaleBlock[]
    for range in ranges
        push!(blocks, InstantonScaleBlock(
            Int[sorted_positions[position] - 1 for position in range],
            collect(range),
            collect(sorted_logs[range])))
    end
    inter_block_gaps = Float64[
        sorted_logs[last(ranges[i - 1])] - sorted_logs[first(ranges[i])]
        for i in 2:length(ranges)
    ]
    return (; blocks, sorted_indices=Int.(sorted_positions .- 1),
        sorted_log_scales=sorted_logs, inter_block_gaps,
        gaps=inter_block_gaps, gap_log10=gap, min_block_size,
        diagnostics=(block_count=length(blocks),
            log_scale_span=isempty(sorted_logs) ? 0.0 : maximum(sorted_logs) - minimum(sorted_logs),
            largest_inter_block_gap=isempty(inter_block_gaps) ? 0.0 : maximum(inter_block_gaps)))
end

function _instanton_split_diagnostics(K::Hermitian{Float64, Matrix{Float64}},
        Q::Matrix{Int}, hierarchy)
    isempty(hierarchy.blocks) && return PerturbativeSplitDiagnostics[]
    Kfactor = cholesky(K).L
    Qcanonical = Matrix(Q') / Kfactor'
    splits = PerturbativeSplitDiagnostics[]
    for boundary in 1:length(hierarchy.blocks)-1
        left = reduce(vcat, (block.indices .+ 1 for block in hierarchy.blocks[1:boundary]))
        right = reduce(vcat, (block.indices .+ 1 for block in hierarchy.blocks[boundary+1:end]))
        left_charge = @view Qcanonical[left, :]
        right_charge = @view Qcanonical[right, :]
        left_norm = opnorm(left_charge)
        right_norm = opnorm(right_charge)
        off_block_norm = opnorm(Matrix(left_charge * right_charge'))
        coupling = off_block_norm / max(left_norm * right_norm, eps(Float64))
        separation_gap = hierarchy.inter_block_gaps[boundary]
        coupling_to_gap_ratio = coupling == 0.0 ? 0.0 :
            separation_gap > 300.0 ? 0.0 : coupling / 10.0^separation_gap
        # Charge-space alignment is checked independently of the scale gap:
        # parallel canonical charges are not a certified perturbative split.
        certified_safe = separation_gap >= 6.0 && coupling <= 1e-6
        push!(splits, PerturbativeSplitDiagnostics(Float64(off_block_norm),
            Float64(separation_gap), Float64(coupling_to_gap_ratio), certified_safe))
    end
    splits
end

"""
    instanton_hierarchy_diagnostics(L; gap_log10=1.0, min_block_size=1)

Return scale-block metadata in addition to the historical leading-gap and span
screening fields. This overload has no charge geometry, so perturbative split
certificates are empty. Use `instanton_hierarchy_diagnostics(K, L, Q; kwargs...)`
to include canonical charge-coupling diagnostics.
"""
function instanton_hierarchy_diagnostics(L::AbstractMatrix{<:Real};
        gap_log10::Real=1.0, min_block_size::Int=1)
    scales = Float64.(L[2, :])
    largest = isempty(scales) ? -Inf : maximum(scales)
    second_largest = length(scales) < 2 ? -Inf : sort(scales; rev=true)[2]
    smallest = isempty(scales) ? Inf : minimum(scales)
    leading_log_gap = length(scales) < 2 ? 0.0 : largest - second_largest
    log_scale_span = isempty(scales) ? 0.0 : largest - smallest
    hierarchy = instanton_scale_blocks(L; gap_log10, min_block_size)
    InstantonHierarchyDiagnostics(leading_log_gap, log_scale_span,
        leading_log_gap >= 30 && log_scale_span >= 1_000, hierarchy.blocks,
        hierarchy.inter_block_gaps, PerturbativeSplitDiagnostics[],
        Float64(gap_log10), min_block_size)
end

"""Return hierarchy blocks and charge-coupling diagnostics for a potential."""
function instanton_hierarchy_diagnostics(K::Hermitian{Float64, Matrix{Float64}},
        L::Matrix{Float64}, Q::Matrix{Int}; gap_log10::Real=1.0,
        min_block_size::Int=1)
    hierarchy = instanton_hierarchy_diagnostics(L; gap_log10, min_block_size)
    splits = _instanton_split_diagnostics(K, Q,
        instanton_scale_blocks(L; gap_log10, min_block_size))
    InstantonHierarchyDiagnostics(hierarchy.leading_log_gap,
        hierarchy.log_scale_span, hierarchy.heuristic_strong_hierarchy,
        hierarchy.blocks, hierarchy.inter_block_gaps, splits, Float64(gap_log10),
        min_block_size)
end

"""
    leading_hessian_mass_basis(K, L, Q; prec=1_000)

Return the high-precision masses, eigenvalue signs, and canonical eigenbasis of
the Hessian formed from a PQ-selected leading instanton set. Results are
ordered by ascending mass and converted to Float64 only after diagonalization.
"""
function leading_hessian_mass_basis(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; prec::Int=1_000)
    W, _ = high_precision_leading_hessian(K, L, Q; prec)
    eigenvalues, eigenvectors = eigen(W)
    masses = Float64.(0.5 .* log10.(abs.(eigenvalues))) .+ 9 .+ Float64(log10(constants()["MPlanck"])) .+ Float64(constants()["log2π"])
    order = sortperm(masses)
    return masses[order], Int.(sign.(eigenvalues[order])), Float64.(eigenvectors[:, order])
end

"""
    leading_hessian_mass_basis_float64(K, L, Q)

Float64 counterpart to [`leading_hessian_mass_basis`](@ref). It is suitable
only for well-resolved leading-Hessian modes.
"""
function leading_hessian_mass_basis_float64(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int})
    eigenvalues, eigenvectors = eigen(leading_hessian_matrix_float64(K, L, Q))
    masses = 0.5 .* log10.(abs.(eigenvalues)) .+ 9 .+ log10(2.435e18) .+ log10(2π)
    order = sortperm(masses)
    return masses[order], Int.(sign.(eigenvalues[order])), eigenvectors[:, order]
end

"""
    pq_spectrum(K,L,Q; mixing_correction=:float64, prec=1_000,
                quartic_diagnostics=false, mass_basis_diagnostics=false,
                hierarchy_diagnostics=false)

Compute a PQ-selected axion spectrum. The PQ procedure first selects the
leading linearly independent instantons. Every returned quartic includes all
instanton scales in `L`, while the requested `mixing_correction` determines
the field basis used for masses and quartics.

# Basis and precision modes

| `mixing_correction` | `m` basis | quartic basis |
|:--|:--|:--|
| `:float64` (default) | Float64 eigensystem of the PQ-selected leading Hessian | same Float64 leading-Hessian mass basis |
| `true` or `:high_precision` | arbitrary-precision eigensystem of the PQ-selected leading Hessian | same high-precision leading-Hessian mass basis, converted to Float64 for the final log-space contraction |
| `false` | original sequential PQ mass estimate | original sequential PQ basis |

`f` and `fK` always retain their original PQ/kinetic-basis definitions; they
are not rotated by a mixing correction.

`quartic_diagnostics=false` is the efficient default. Set it to `true` to
return an additional cancellation assessment for every quartic component.
That assessment repeats the all-scale contraction and is therefore intended
for validation rather than large production scans.

`mass_basis_diagnostics=false` is likewise opt-in. In a leading-Hessian mass
basis it evaluates Float64 eigenpair residuals, nearest relative eigenvalue
gaps, and orthogonality error. It is unavailable for `mixing_correction=false`,
whose sequential PQ directions are not Hessian eigenvectors.

`hierarchy_diagnostics=false` keeps this additional linear-time scan out of
the default path. When requested, `instanton_hierarchy` reports a
physics-informed sanity check. Its provisional `heuristic_strong_hierarchy`
flag requires a leading instanton-scale gap of at least 30 and a full
log-scale span of at least 1,000. These values were empirically useful at
h11=10 only; they do not certify numerical accuracy and must be revalidated
at other dimensions.

# Numerical interpretation

The default is fast and is appropriate for masses and self-interactions in
the Float64-resolved sector. It does not make light eigenvectors below the
relative Float64 resolution floor reliable. A mass being resolved also does
not automatically certify every mixed quartic: a `λ31` or `λ22` component can
lose precision through cancellation among instanton contributions. Use
`:high_precision` when such components need controlled accuracy.

# Return value

Returns an `AxionSpectrum` with masses `m`, aligned Hessian signs `msign`,
decay quantities `f` and `fK`, and signed base-10 logarithms for `λself`,
`λ31`, and `λ22`. The `λ31_i` and
`λ22_i` matrices give zero-based mode indices. `quartic_diagnostics` is
`nothing` by default; when requested, it is aligned with these component
families and reports final-sum cancellation in the Float64 log-space
contraction. Signs of `λ31` additionally depend on the arbitrary sign
convention for individual mass eigenvectors.
"""
function pq_spectrum(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; mixing_correction::Union{Bool, Symbol}=:float64, prec::Int=1_000, quartic_diagnostics::Bool=false, mass_basis_diagnostics::Bool=false, hierarchy_diagnostics::Bool=false)
    h11::Int = size(K,1)
    fK::Vector{Float64} = log10.(sqrt.(eigen(K).values))
    Kls = cholesky(K).L
    
    LQtild = LQtilde(Q, L)
    Ltilde = LQtild.Ltilde
    Qtilde = LQtild.Qtilde
    # `Q * inv(Kls')` is the charge matrix in the canonically normalized basis.
    Qcanonical = Matrix(Q') / Kls'
    Qleading = Matrix(Qtilde') / Kls'
    fapprox, mapprox, P = pq_canonical_frame(Qleading, Ltilde)

    # Match the mass ordering used in the returned PQ spectrum.
    order = sortperm(mapprox)
    masses = mapprox[order] .+ 9 .+ Float64(log10(constants()["MPlanck"])) .+ Float64(constants()["log2π"])
    mass_signs = Int.(sign.(@view Ltilde[1, order]))
    quartic_basis = P[:, order]
    correction_mode = mixing_correction === true ? :high_precision : mixing_correction === false ? :none : mixing_correction
    @assert correction_mode in (:none, :float64, :high_precision) "mixing_correction must be false, true, :float64, or :high_precision"
    if correction_mode === :high_precision
        # This is the exact eigensystem of PQ's own selected leading-Hessian,
        # not an all-instanton HP calculation.
        masses, mass_signs, quartic_basis = leading_hessian_mass_basis(K, Ltilde, Qtilde; prec=prec)
    elseif correction_mode === :float64
        masses, mass_signs, quartic_basis = leading_hessian_mass_basis_float64(K, Ltilde, Qtilde)
    end
    if mass_basis_diagnostics && correction_mode === :none
        throw(ArgumentError("mass_basis_diagnostics requires a leading-Hessian mass basis; set mixing_correction to :float64 or :high_precision"))
    end
    mass_diagnostics = mass_basis_diagnostics ? mass_basis_accuracy(K, Ltilde, Qtilde, quartic_basis) : nothing
    hierarchy = hierarchy_diagnostics ?
        instanton_hierarchy_diagnostics(K, L, Q) : nothing
    Qpq = Qcanonical * quartic_basis
    scale_sign = Int.(sign.(@view L[1, :]))
    scale_log = log(10) .* @view(L[2, :])
    positive_logs = zeros(Float64, length(scale_sign))
    negative_logs = zeros(Float64, length(scale_sign))

    qindq31 = [(i, i, i, j) for i in 1:h11 for j in 1:h11 if i != j]
    qindq22 = [(i, i, j, j) for i in 1:h11 for j in 1:i-1]
    quartdiagsign = zeros(Int, h11)
    quartdiaglog = zeros(Float64, h11)
    quart31sign = zeros(Int, length(qindq31))
    quart31log = zeros(Float64, length(qindq31))
    quart22sign = zeros(Int, length(qindq22))
    quart22log = zeros(Float64, length(qindq22))

    for i in 1:h11
        quartdiagsign[i], quartdiaglog[i], _ = pq_contracted_log!(positive_logs, negative_logs, scale_sign, scale_log, Qpq, (i, i, i, i))
    end
    for i in eachindex(qindq31)
        quart31sign[i], quart31log[i], _ = pq_contracted_log!(positive_logs, negative_logs, scale_sign, scale_log, Qpq, qindq31[i])
    end
    for i in eachindex(qindq22)
        quart22sign[i], quart22log[i], _ = pq_contracted_log!(positive_logs, negative_logs, scale_sign, scale_log, Qpq, qindq22[i])
    end

    log2π = Float64(constants()["log2π"])
    diagnostics = nothing
    if quartic_diagnostics
        function component_diagnostics(exponents)
            result_sign, result_log, absolute_sum_log = pq_contracted_log!(positive_logs, negative_logs, scale_sign, scale_log, Qpq, exponents)
            if result_sign == 0
                return Inf, -Inf, false, true
            end
            orders_lost = max(0.0, (absolute_sum_log - result_log) / log(10))
            digits_remaining = -log10(eps(Float64)) - orders_lost
            return orders_lost, digits_remaining, digits_remaining >= 3, false
        end
        self_diagnostics = [component_diagnostics((i, i, i, i)) for i in 1:h11]
        three_one_diagnostics = [component_diagnostics(qindq31[i]) for i in eachindex(qindq31)]
        two_two_diagnostics = [component_diagnostics(qindq22[i]) for i in eachindex(qindq22)]
        diagnostics = QuarticDiagnostics(
            QuarticComponentDiagnostics(getindex.(self_diagnostics, 1), getindex.(self_diagnostics, 2), BitVector(getindex.(self_diagnostics, 3)), BitVector(getindex.(self_diagnostics, 4))),
            QuarticComponentDiagnostics(getindex.(three_one_diagnostics, 1), getindex.(three_one_diagnostics, 2), BitVector(getindex.(three_one_diagnostics, 3)), BitVector(getindex.(three_one_diagnostics, 4))),
            QuarticComponentDiagnostics(getindex.(two_two_diagnostics, 1), getindex.(two_two_diagnostics, 2), BitVector(getindex.(two_two_diagnostics, 3)), BitVector(getindex.(two_two_diagnostics, 4))),
        )
    end
    AxionSpectrum(masses, mass_signs, 0.5 .* fapprox[order] .+ Float64(log10(constants()["MPlanck"])), fK .+ Float64(log10(constants()["MPlanck"])) .- log2π,
    quartdiagsign, quartdiaglog .* log10(exp(1)) .+ 4 * log2π,
    isempty(qindq31) ? zeros(Int, 4, 0) : hcat(collect.(qindq31)...) .- 1, quart31sign, quart31log .* log10(exp(1)) .+ 4 * log2π,
    isempty(qindq22) ? zeros(Int, 4, 0) : hcat(collect.(qindq22)...) .- 1, quart22sign, quart22log .* log10(exp(1)) .+ 4 * log2π, diagnostics, mass_diagnostics, hierarchy)
end

"""
    pq_spectrum(h11, tri, cy; kwargs...)

Load the potential for one geometry and delegate to
[`pq_spectrum(K, L, Q; kwargs...)`](@ref).
"""
function pq_spectrum(h11::Int,tri::Int,cy::Int; kwargs...)
    pot_data = potential(h11,tri,cy)
    K,L,Q = pot_data.K, pot_data.L, pot_data.Q
    pq_spectrum(K, L, Q; kwargs...)
end

function pq_spectrum(geom_idx::GeometryIndex; kwargs...)
    pot_data = potential(geom_idx)
    pq_spectrum(pot_data.K, pot_data.L, pot_data.Q; kwargs...)
end

"""
    pq_physical_spectrum(K, L, Q; threshold_log10=log10(H₀), prec=1_000,
                         quartics=true)

Compute a high-precision PQ leading-Hessian spectrum and retain only modes
whose base-10 mass logarithm is at least `threshold_log10`; by default this is
the package Hubble scale. Every instanton is retained in the quartic
contractions when `quartics=true`, but quartics are returned only for the
retained physical modes. Set `quartics=false` for a mass-only reference
calculation; the masses, mode indices, and eigenvectors are unchanged.

This is a conservative reference implementation for the physical sector. It
diagonalizes the full leading Hessian at arbitrary precision and is not the
future threshold-targeted hybrid solver.
"""
function pq_physical_spectrum(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; threshold_log10::Float64=Float64(log10(constants()["Hubble"])), prec::Int=1_000, quartics::Bool=true)
    LQtild = LQtilde(Q, L)
    Ltilde, Qtilde = LQtild.Ltilde, LQtild.Qtilde
    W, Kfactor = high_precision_leading_hessian(K, Ltilde, Qtilde; prec)
    T = eltype(W)
    h11 = size(K, 1)
    eigenvalues, eigenvectors = eigen(W)
    masses = Float64.(0.5 .* log10.(abs.(eigenvalues))) .+ 9 .+ Float64(log10(constants()["MPlanck"])) .+ Float64(constants()["log2π"])
    order = sortperm(masses)
    masses = masses[order]
    eigenvectors = eigenvectors[:, order]
    retained = findall(masses .>= threshold_log10)
    physical_masses = masses[retained]
    !quartics && return PhysicalAxionSpectrum(physical_masses, retained .- 1,
        Float64.(eigenvectors[:, retained]), Int[], Float64[], zeros(Int, 4, 0),
        Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[], threshold_log10, prec)
    Qmass = (T.(Q') / Kfactor.L') * eigenvectors[:, retained]
    quartic_scales = T.(L[1, :]) .* (T(10) .^ T.(L[2, :]))
    physical_count = length(retained)
    qindq31 = [(i, i, i, j) for i in 1:physical_count for j in 1:physical_count if i != j]
    qindq22 = [(i, i, j, j) for i in 1:physical_count for j in 1:i-1]

    function signed_quartic(exponents::NTuple{4, Int})
        value = zero(T)
        for a in eachindex(quartic_scales)
            value += quartic_scales[a] * Qmass[a, exponents[1]] * Qmass[a, exponents[2]] * Qmass[a, exponents[3]] * Qmass[a, exponents[4]]
        end
        value_sign = Int(sign(value))
        return value_sign, value_sign == 0 ? -Inf : Float64(log10(abs(value)))
    end

    self_sign = zeros(Int, physical_count)
    self_log = zeros(Float64, physical_count)
    for i in 1:physical_count
        self_sign[i], self_log[i] = signed_quartic((i, i, i, i))
    end
    three_one_sign = zeros(Int, length(qindq31))
    three_one_log = zeros(Float64, length(qindq31))
    for i in eachindex(qindq31)
        three_one_sign[i], three_one_log[i] = signed_quartic(qindq31[i])
    end
    two_two_sign = zeros(Int, length(qindq22))
    two_two_log = zeros(Float64, length(qindq22))
    for i in eachindex(qindq22)
        two_two_sign[i], two_two_log[i] = signed_quartic(qindq22[i])
    end

    log2π = Float64(constants()["log2π"])
    PhysicalAxionSpectrum(physical_masses, retained .- 1, Float64.(eigenvectors[:, retained]),
        self_sign, self_log .+ 4 * log2π,
        isempty(qindq31) ? zeros(Int, 4, 0) : hcat(collect.(qindq31)...) .- 1, three_one_sign, three_one_log .+ 4 * log2π,
        isempty(qindq22) ? zeros(Int, 4, 0) : hcat(collect.(qindq22)...) .- 1, two_two_sign, two_two_log .+ 4 * log2π,
        threshold_log10, prec)
end

"""Load one geometry and delegate to [`pq_physical_spectrum(K, L, Q; kwargs...)`](@ref)."""
function pq_physical_spectrum(h11::Int, tri::Int, cy::Int; kwargs...)
    pot_data = potential(h11, tri, cy)
    pq_physical_spectrum(pot_data.K, pot_data.L, pot_data.Q; kwargs...)
end

"""Load one indexed geometry and delegate to [`pq_physical_spectrum(K, L, Q; kwargs...)`](@ref)."""
function pq_physical_spectrum(geom_idx::GeometryIndex; kwargs...)
    pot_data = potential(geom_idx)
    pq_physical_spectrum(pot_data.K, pot_data.L, pot_data.Q; kwargs...)
end

function high_precision_orthonormalize!(basis::Matrix{T}) where {T}
    for j in axes(basis, 2)
        for i in 1:j-1
            basis[:, j] .-= dot(@view(basis[:, i]), @view(basis[:, j])) .* @view(basis[:, i])
        end
        basis[:, j] ./= norm(@view(basis[:, j]))
    end
    basis
end

function schur_physical_basis(W::Hermitian{T, Matrix{T}}, float_basis::Matrix{Float64}, physical_count::Int; maxiter::Int, residual_tolerance::Float64) where {T}
    h11 = size(W, 1)
    B = T.(float_basis[:, end-physical_count+1:end])
    U = T.(float_basis[:, 1:end-physical_count])
    A = Hermitian(B' * W * B)
    coupling = B' * W * U
    C = Hermitian(U' * W * U)
    eigenvalues = eigen(A).values
    basis = similar(B)
    residuals = fill(Inf, physical_count)
    for _ in 1:maxiter
        next_values = similar(eigenvalues)
        coefficients = similar(B, physical_count, physical_count)
        available = collect(1:physical_count)
        for i in eachindex(eigenvalues)
            S = Hermitian(Matrix(A) - coupling * ((Matrix(C) - eigenvalues[i] * I) \ coupling'))
            values, vectors = eigen(S)
            local_index = argmin(abs.(values[available] .- eigenvalues[i]))
            j = available[local_index]
            deleteat!(available, local_index)
            next_values[i] = values[j]
            coefficients[:, i] .= vectors[:, j]
        end
        eigenvalues = next_values
        for i in eachindex(eigenvalues)
            y = -(Matrix(C) - eigenvalues[i] * I) \ (coupling' * @view(coefficients[:, i]))
            basis[:, i] .= B * @view(coefficients[:, i]) + U * y
            basis[:, i] ./= norm(@view(basis[:, i]))
        end
        residuals = [Float64(norm(W * @view(basis[:, i]) - eigenvalues[i] .* @view(basis[:, i])) / abs(eigenvalues[i])) for i in eachindex(eigenvalues)]
        maximum(residuals) <= residual_tolerance && return eigenvalues, basis, residuals
    end
    eigenvalues, basis, residuals
end

"""Check whether the Float64-seeded complement lies below the physical threshold."""
function schur_admissible_float64(W::Hermitian, float_basis::Matrix{Float64}, physical_count::Int, threshold_log10::Float64)
    h11 = size(W, 1)
    physical_count == h11 && return true
    complement = @view float_basis[:, 1:end-physical_count]
    complement_matrix = Symmetric(complement' * Float64.(Matrix(W)) * complement)
    complement_eigenvalues = eigvals(complement_matrix)
    all(isfinite, complement_eigenvalues) || return true
    mass_offset = 9.0 + Float64(log10(constants()["MPlanck"])) + Float64(constants()["log2π"])
    threshold_eigenvalue = 10.0 ^ (2 * (threshold_log10 - mass_offset))
    maximum(complement_eigenvalues) < threshold_eigenvalue
end

"""Select the dense or sparse quartic contraction backend for `Q`."""
function select_quartic_backend(Q::Matrix{Int}, backend::Symbol)
    backend in (:auto, :dense, :sparse) || throw(ArgumentError("quartic_backend must be :auto, :dense, or :sparse"))
    backend !== :auto && return backend
    entries = length(Q)
    density = entries == 0 ? 0.0 : count(value -> !iszero(value), Q) / entries
    entries >= 100_000 && density <= 0.10 ? :sparse : :dense
end

"""Transform charges into the canonical physical-mode basis."""
function quartic_charge_basis(Q::Matrix{Int}, Kfactor, basis::Matrix{T}, backend::Symbol) where {T}
    transformed_basis = transpose(Kfactor.L) \ basis
    backend === :sparse ? sparse(transpose(Q)) * transformed_basis : T.(transpose(Q)) * transformed_basis
end

"""Compute signed log-space diagonal quartics for the retained modes."""
function diagonal_quartics(Q::Matrix{Int}, L::Matrix{Float64}, Kfactor, basis::Matrix{T}, backend::Symbol) where {T}
    transformed_basis = transpose(Kfactor.L) \ basis
    physical_count = size(basis, 2)
    values = zeros(T, physical_count)
    quartic_scales = T.(L[1, :]) .* (T(10) .^ T.(L[2, :]))
    if backend === :sparse
        sparse_Q = sparse(Q)
        charges = zeros(T, physical_count)
        for instanton in axes(Q, 2)
            fill!(charges, zero(T))
            for pointer in nzrange(sparse_Q, instanton)
                row = sparse_Q.rowval[pointer]
                charge = T(sparse_Q.nzval[pointer])
                for mode in 1:physical_count
                    charges[mode] += charge * transformed_basis[row, mode]
                end
            end
            scale = quartic_scales[instanton]
            for mode in 1:physical_count
                values[mode] += scale * charges[mode]^4
            end
        end
    else
        charge_basis = T.(transpose(Q)) * transformed_basis
        for instanton in axes(Q, 2), mode in 1:physical_count
            values[mode] += quartic_scales[instanton] * charge_basis[instanton, mode]^4
        end
    end
    signs = Int.(sign.(values))
    logs = [signs[mode] == 0 ? -Inf : Float64(log10(abs(values[mode]))) for mode in 1:physical_count]
    signs, logs
end

"""
    pq_hybrid_physical_spectrum(K, L, Q; threshold_log10=log10(H₀), prec=1_000,
                                maxiter=100, residual_tolerance=1e-30,
                                schur_acceleration=true, oversampling=8,
                                quartics=true, mixed_quartics=true,
                                quartic_backend=:auto)

Compute only the PQ leading-Hessian modes above `threshold_log10` with a
sequential-PQ-seeded, arbitrary-precision block subspace iteration. The physical
mode count is obtained from high-precision inertia, then the corresponding
largest eigenpairs are refined without a full arbitrary-precision
eigendecomposition. Returned quartics include every instanton, as in
[`pq_physical_spectrum`](@ref).

With `schur_acceleration=true` (the default), the solver first checks whether
the sequential-PQ-seeded complement lies wholly below the physical threshold. If so,
it uses a Schur refinement; otherwise it uses block subspace iteration.

Set `quartics=false` to return only physical masses, mode indices, and
eigenvectors. This avoids the rapidly growing physical-sector quartic output
for large numbers of retained modes.

Set `mixed_quartics=false` with `quartics=true` to compute only diagonal
`lambda_iiii` self-couplings. This is the compact production mode for large
ensemble scans.

`quartic_backend=:auto` selects the dense implementation for small or dense
charge matrices and the sparse implementation for large sparse matrices. Use
`:dense` or `:sparse` to override the dispatch for benchmarking.

When the block-subspace fallback is used, `oversampling` adds a small number
of sub-threshold vectors to the iteration and discards them at the end. This
improves convergence when the spectral gap at the physical threshold is small.

The returned `PhysicalAxionSpectrum` has the same fields as the full
high-precision reference routine. If the requested residual tolerance is not
met by `maxiter`, the solver warns and falls back to the full high-precision
eigensystem. A provisional result is returned only if that fallback also fails
the residual check.
"""
function pq_hybrid_physical_spectrum(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; threshold_log10::Float64=Float64(log10(constants()["Hubble"])), prec::Int=1_000, maxiter::Int=100, residual_tolerance::Float64=1e-30, schur_acceleration::Bool=true, oversampling::Int=8, quartics::Bool=true, mixed_quartics::Bool=true, quartic_backend::Symbol=:auto, hierarchy_gap_log10::Float64=1.0, hierarchy_min_block_size::Int=1, label::AbstractString="matrix input")
    LQtild = LQtilde(Q, L)
    Ltilde, Qtilde = LQtild.Ltilde, LQtild.Qtilde
    W, Kfactor = high_precision_leading_hessian(K, Ltilde, Qtilde; prec)
    physical_count = physical_mode_inertia_count(W, threshold_log10)
    physical_count = confirm_physical_mode_count(
        physical_count, K, Ltilde, Qtilde, threshold_log10, prec, 4_000, label)
    # Confirmation raises ArbFloat's default precision. Restore the requested
    # working precision before allocating refinement temporaries; `W` and its
    # factor retain the precision with which they were constructed.
    setprecision(ArbFloat; digits=prec)
    physical_count == 0 && return PhysicalAxionSpectrum(Float64[], Int[], zeros(Float64, size(K, 1), 0), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[], threshold_log10, prec)

    T = eltype(W)
    h11 = size(K, 1)

    Kls = cholesky(K).L
    Qleading = Matrix(Qtilde') / Kls'
    _, pq_masses, pq_basis = pq_canonical_frame(Qleading, Ltilde)
    seed_basis = pq_basis[:, sortperm(pq_masses)]
    hierarchy = instanton_hierarchy_diagnostics(Ltilde;
        gap_log10=hierarchy_gap_log10, min_block_size=hierarchy_min_block_size)
    hierarchy_safe = if length(hierarchy.blocks) > 32
        false
    else
        charge_hierarchy = instanton_hierarchy_diagnostics(K, Ltilde, Qtilde;
            gap_log10=hierarchy_gap_log10, min_block_size=hierarchy_min_block_size)
        isempty(charge_hierarchy.perturbative_splits) ||
            all(split.certified_safe for split in charge_hierarchy.perturbative_splits)
    end
    schur_safe = false
    if schur_acceleration && physical_count < h11
        if hierarchy_safe && schur_admissible_float64(W, seed_basis, physical_count, threshold_log10)
            complement = T.(seed_basis[:, 1:end-physical_count])
            C = Hermitian(complement' * W * complement)
            mass_offset = T(9) + log10(T(constants()["MPlanck"])) + T(constants()["log2π"])
            threshold_eigenvalue = T(10) ^ (2 * (T(threshold_log10) - mass_offset))
            schur_safe = positive_inertia(bunchkaufman(Hermitian(Matrix(C) - threshold_eigenvalue * I))) == 0
        end
    end
    if schur_safe
        eigenvalues, basis, residuals = schur_physical_basis(W, seed_basis, physical_count; maxiter, residual_tolerance)
    else
        subspace_count = min(h11, physical_count + max(oversampling, 0))
        basis = high_precision_orthonormalize!(T.(seed_basis[:, end-subspace_count+1:end]))
        eigenvalues = zeros(T, physical_count)
        residuals = fill(Inf, physical_count)
        for _ in 1:maxiter
            basis = high_precision_orthonormalize!(Matrix(W) * basis)
            ritz_values, coefficients = eigen(Hermitian(basis' * W * basis))
            basis = basis * coefficients
            eigenvalues = ritz_values[end-physical_count+1:end]
            physical_basis = @view basis[:, end-physical_count+1:end]
            residuals = [Float64(norm(W * @view(physical_basis[:, i]) - eigenvalues[i] .* @view(physical_basis[:, i])) / abs(eigenvalues[i])) for i in eachindex(eigenvalues)]
            maximum(residuals) <= residual_tolerance && break
        end
        basis = Matrix(@view basis[:, end-physical_count+1:end])
    end
    if maximum(residuals) > residual_tolerance
        @warn "hybrid physical spectrum did not reach residual_tolerance=$(residual_tolerance) for geometry=$(label); maximum relative residual=$(maximum(residuals)). Falling back to the full high-precision eigensystem."
        all_values, all_vectors = eigen(W)
        all_masses = Float64.(0.5 .* log10.(abs.(all_values))) .+ 9 .+
            Float64(log10(constants()["MPlanck"])) .+
            Float64(constants()["log2π"])
        all_order = sortperm(all_masses)
        retained = [index for index in all_order if all_values[index] > 0 &&
            all_masses[index] >= threshold_log10]
        length(retained) == physical_count || throw(
            ArgumentError("hybrid full-eigensystem fallback found $(length(retained)) physical modes; expected $(physical_count) for geometry=$(label)"))
        eigenvalues = all_values[retained]
        basis = all_vectors[:, retained]
        residuals = [Float64(norm(W * @view(basis[:, i]) -
            eigenvalues[i] .* @view(basis[:, i])) /
            max(abs(eigenvalues[i]), eps(T))) for i in axes(basis, 2)]
        maximum(residuals) > residual_tolerance && @warn "hybrid full-eigensystem fallback remained above residual_tolerance=$(residual_tolerance) for geometry=$(label); maximum relative residual=$(maximum(residuals)). Returning a provisional result."
    end
    masses = Float64.(0.5 .* log10.(abs.(eigenvalues))) .+ 9 .+ Float64(log10(constants()["MPlanck"])) .+ Float64(constants()["log2π"])
    !quartics && return PhysicalAxionSpectrum(masses, collect(h11-physical_count:h11-1), Float64.(basis), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[], threshold_log10, prec)
    selected_backend = select_quartic_backend(Q, quartic_backend)
    log2π = Float64(constants()["log2π"])
    if !mixed_quartics
        self_sign, self_log = diagonal_quartics(Q, L, Kfactor, basis, selected_backend)
        return PhysicalAxionSpectrum(masses, collect(h11-physical_count:h11-1), Float64.(basis),
            self_sign, self_log .+ 4 * log2π,
            zeros(Int, 4, 0), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[],
            threshold_log10, prec)
    end
    Qmass = quartic_charge_basis(Q, Kfactor, basis, selected_backend)
    quartic_scales = T.(L[1, :]) .* (T(10) .^ T.(L[2, :]))
    qindq31 = mixed_quartics ? [(i, i, i, j) for i in 1:physical_count for j in 1:physical_count if i != j] : Tuple{Int,Int,Int,Int}[]
    qindq22 = mixed_quartics ? [(i, i, j, j) for i in 1:physical_count for j in 1:i-1] : Tuple{Int,Int,Int,Int}[]
    function signed_quartic(exponents::NTuple{4, Int})
        value = zero(T)
        for a in eachindex(quartic_scales)
            value += quartic_scales[a] * Qmass[a, exponents[1]] * Qmass[a, exponents[2]] * Qmass[a, exponents[3]] * Qmass[a, exponents[4]]
        end
        value_sign = Int(sign(value))
        value_sign, value_sign == 0 ? -Inf : Float64(log10(abs(value)))
    end
    self_sign = zeros(Int, physical_count); self_log = zeros(Float64, physical_count)
    for i in 1:physical_count
        self_sign[i], self_log[i] = signed_quartic((i, i, i, i))
    end
    three_one_sign = zeros(Int, length(qindq31)); three_one_log = zeros(Float64, length(qindq31))
    for i in eachindex(qindq31)
        three_one_sign[i], three_one_log[i] = signed_quartic(qindq31[i])
    end
    two_two_sign = zeros(Int, length(qindq22)); two_two_log = zeros(Float64, length(qindq22))
    for i in eachindex(qindq22)
        two_two_sign[i], two_two_log[i] = signed_quartic(qindq22[i])
    end
    PhysicalAxionSpectrum(masses, collect(h11-physical_count:h11-1), Float64.(basis),
        self_sign, self_log .+ 4 * log2π,
        isempty(qindq31) ? zeros(Int, 4, 0) : hcat(collect.(qindq31)...) .- 1, three_one_sign, three_one_log .+ 4 * log2π,
        isempty(qindq22) ? zeros(Int, 4, 0) : hcat(collect.(qindq22)...) .- 1, two_two_sign, two_two_log .+ 4 * log2π,
        threshold_log10, prec)
end

"""Load one geometry and delegate to [`pq_hybrid_physical_spectrum(K, L, Q; kwargs...)`](@ref)."""
function pq_hybrid_physical_spectrum(h11::Int, tri::Int, cy::Int; label::AbstractString="h11=$(h11), polytope=$(tri), frst=$(cy)", kwargs...)
    pot_data = potential(h11, tri, cy)
    pq_hybrid_physical_spectrum(pot_data.K, pot_data.L, pot_data.Q; label, kwargs...)
end

"""Load one indexed geometry and delegate to [`pq_hybrid_physical_spectrum(K, L, Q; kwargs...)`](@ref)."""
function pq_hybrid_physical_spectrum(geom_idx::GeometryIndex; label::AbstractString="h11=$(geom_idx.h11), polytope=$(geom_idx.polytope), frst=$(geom_idx.frst)", kwargs...)
    pot_data = potential(geom_idx)
    pq_hybrid_physical_spectrum(pot_data.K, pot_data.L, pot_data.Q; label, kwargs...)
end

function _empty_physical_spectrum(threshold_log10::Float64, prec::Int, h11::Int;
        min_log10_mass::Float64=threshold_log10,
        max_log10_mass::Float64=Inf,
        diagnostics::Union{Nothing, SpectrumWindowDiagnostics}=nothing)
    PhysicalAxionSpectrum(Float64[], Int[], zeros(Float64, h11, 0), Int[], Float64[],
        zeros(Int, 4, 0), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[],
        threshold_log10, prec, min_log10_mass, max_log10_mass, diagnostics)
end

"""Contract quartics and assemble a physical spectrum from a refined basis."""
function _physical_spectrum_from_basis(K::Hermitian{Float64, Matrix{Float64}},
        L::Matrix{Float64}, Q::Matrix{Int}, basis::Matrix{T}, masses::Vector{Float64},
        mode_indices::Vector{Int}; threshold_log10::Float64,
        min_log10_mass::Float64=threshold_log10, max_log10_mass::Float64=Inf,
        prec::Int, quartics::Bool, mixed_quartics::Bool,
        quartic_backend::Symbol, diagnostics=nothing) where {T}
    !quartics && return PhysicalAxionSpectrum(masses, mode_indices, Float64.(basis),
        Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[],
        zeros(Int, 4, 0), Int[], Float64[], threshold_log10, prec,
        min_log10_mass, max_log10_mass, diagnostics)
    Kfactor = cholesky(K)
    selected_backend = select_quartic_backend(Q, quartic_backend)
    log2π = Float64(constants()["log2π"])
    if !mixed_quartics
        self_sign, self_log = diagonal_quartics(Q, L, Kfactor, basis, selected_backend)
        return PhysicalAxionSpectrum(masses, mode_indices, Float64.(basis),
            self_sign, self_log .+ 4 * log2π, zeros(Int, 4, 0), Int[], Float64[],
            zeros(Int, 4, 0), Int[], Float64[], threshold_log10, prec,
            min_log10_mass, max_log10_mass, diagnostics)
    end
    Qmass = quartic_charge_basis(Q, Kfactor, basis, selected_backend)
    quartic_scales = T.(L[1, :]) .* (T(10) .^ T.(L[2, :]))
    physical_count = length(masses)
    qindq31 = [(i, i, i, j) for i in 1:physical_count for j in 1:physical_count if i != j]
    qindq22 = [(i, i, j, j) for i in 1:physical_count for j in 1:i-1]
    function signed_quartic(exponents::NTuple{4, Int})
        value = zero(T)
        for a in eachindex(quartic_scales)
            value += quartic_scales[a] * Qmass[a, exponents[1]] *
                Qmass[a, exponents[2]] * Qmass[a, exponents[3]] * Qmass[a, exponents[4]]
        end
        value_sign = Int(sign(value))
        value_sign, value_sign == 0 ? -Inf : Float64(log10(abs(value)))
    end
    self_sign = zeros(Int, physical_count)
    self_log = zeros(Float64, physical_count)
    for i in 1:physical_count
        self_sign[i], self_log[i] = signed_quartic((i, i, i, i))
    end
    three_one_sign = zeros(Int, length(qindq31))
    three_one_log = zeros(Float64, length(qindq31))
    for i in eachindex(qindq31)
        three_one_sign[i], three_one_log[i] = signed_quartic(qindq31[i])
    end
    two_two_sign = zeros(Int, length(qindq22))
    two_two_log = zeros(Float64, length(qindq22))
    for i in eachindex(qindq22)
        two_two_sign[i], two_two_log[i] = signed_quartic(qindq22[i])
    end
    PhysicalAxionSpectrum(masses, mode_indices, Float64.(basis),
        self_sign, self_log .+ 4 * log2π,
        isempty(qindq31) ? zeros(Int, 4, 0) : hcat(collect.(qindq31)...) .- 1,
        three_one_sign, three_one_log .+ 4 * log2π,
        isempty(qindq22) ? zeros(Int, 4, 0) : hcat(collect.(qindq22)...) .- 1,
        two_two_sign, two_two_log .+ 4 * log2π, threshold_log10, prec,
        min_log10_mass, max_log10_mass, diagnostics)
end

function _window_counts(W::Hermitian, min_log10_mass::Float64,
        max_log10_mass::Float64, boundary_margin_log10::Float64)
    T = eltype(W)
    mass_offset = T(9) + log10(T(constants()["MPlanck"])) + T(constants()["log2π"])
    # Bounds arrive as Float64 mass logs while the inertia matrix is arbitrary
    # precision; retain a small decimal margin so a round-tripped boundary mode
    # is included rather than lost to the Float64 conversion.
    inclusive_margin = T(boundary_margin_log10)
    function count_above(bound::Float64, margin)
        threshold = T(10) ^ (2 * (T(bound) + margin - mass_offset))
        positive_inertia(bunchkaufman(Hermitian(Matrix(W) - threshold * I)))
    end
    h11 = size(W, 1)
    lower_count = isinf(min_log10_mass) && min_log10_mass < 0 ? h11 :
        count_above(min_log10_mass, -inclusive_margin)
    upper_count = isinf(max_log10_mass) && max_log10_mass > 0 ? 0 :
        count_above(max_log10_mass, inclusive_margin)
    lower_count, upper_count
end

function _window_counts_at_precision(K::Hermitian{Float64, Matrix{Float64}},
        Ltilde::Matrix{Float64}, Qtilde::Matrix{Int}, min_log10_mass::Float64,
        max_log10_mass::Float64, boundary_margin_log10::Float64, prec::Int)
    W, Kfactor = high_precision_leading_hessian(K, Ltilde, Qtilde; prec)
    lower_count, upper_count = _window_counts(W, min_log10_mass, max_log10_mass,
        boundary_margin_log10)
    W, Kfactor, lower_count, upper_count
end

function _confirm_window_counts(K::Hermitian{Float64, Matrix{Float64}},
        Ltilde::Matrix{Float64}, Qtilde::Matrix{Int}, min_log10_mass::Float64,
    max_log10_mass::Float64, boundary_margin_log10::Float64, prec::Int,
        max_prec::Int, confirm::Bool, label::AbstractString)
    max_prec >= prec || throw(ArgumentError("max_prec must be at least prec"))
    W, Kfactor, lower_count, upper_count = _window_counts_at_precision(
        K, Ltilde, Qtilde, min_log10_mass, max_log10_mass,
        boundary_margin_log10, prec)
    records = NTuple{3,Int}[(prec, lower_count, upper_count)]
    stable = true
    working_prec = prec
    if confirm
        stable = false
        while working_prec < max_prec
            working_prec = min(2 * working_prec, max_prec)
            next_W, next_Kfactor, next_lower, next_upper = _window_counts_at_precision(
                K, Ltilde, Qtilde, min_log10_mass, max_log10_mass,
                boundary_margin_log10, working_prec)
            push!(records, (working_prec, next_lower, next_upper))
            W, Kfactor = next_W, next_Kfactor
            if next_lower == lower_count && next_upper == upper_count
                stable = true
                break
            end
            lower_count, upper_count = next_lower, next_upper
        end
        if !stable
            @warn "window mode counts did not stabilize by max_prec=$(max_prec) for geometry=$(label); returning a provisional result"
        end
    end
    W, Kfactor, lower_count, upper_count, records, stable
end

function _mass_values(eigenvalues, mass_offset::Float64)
    Float64.(0.5 .* log10.(abs.(eigenvalues))) .+ mass_offset
end

"""
    pq_window_spectrum(K, L, Q; min_log10_mass=-Inf, max_log10_mass=Inf,
                       prec=1_000, maxiter=100, residual_tolerance=1e-30)

Compute only the leading-Hessian modes in a two-sided physical mass window.
Both boundaries are counted with arbitrary-precision inertia. PQ-seeded Schur
refinement is then applied to an oversampled band around the requested
eigenspace; a full arbitrary-precision reference eigensystem is used only
when the targeted refinement fails interval or residual validation.

`oversampling` adds candidates on both boundaries, `quartics=false` skips all
quartic contractions, and `mixed_quartics=false` retains only self-couplings.
`boundary_margin_log10` is the symmetric inclusive boundary guard used when
converting Float64 mass bounds to arbitrary-precision eigenvalue bounds. The
returned `PhysicalAxionSpectrum.diagnostics` records precision counts,
boundary validation, convergence, and fallback status.
"""
function pq_window_spectrum(K::Hermitian{Float64, Matrix{Float64}},
        L::Matrix{Float64}, Q::Matrix{Int};
        min_log10_mass::Real=-Inf, max_log10_mass::Real=Inf,
        prec::Int=1_000, maxiter::Int=100, residual_tolerance::Float64=1e-30,
        oversampling::Int=8, quartics::Bool=true, mixed_quartics::Bool=true,
        quartic_backend::Symbol=:auto, confirm::Bool=true, max_prec::Int=4_000,
        schur_acceleration::Bool=true, hierarchy_gap_log10::Float64=1.0,
        hierarchy_min_block_size::Int=1, boundary_margin_log10::Real=1e-10,
        label::AbstractString="matrix input")
    min_log10_mass = Float64(min_log10_mass)
    max_log10_mass = Float64(max_log10_mass)
    boundary_margin_log10 = Float64(boundary_margin_log10)
    isfinite(min_log10_mass) || min_log10_mass == -Inf ||
        throw(ArgumentError("min_log10_mass must be finite or -Inf"))
    isfinite(max_log10_mass) || max_log10_mass == Inf ||
        throw(ArgumentError("max_log10_mass must be finite or Inf"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
    oversampling >= 0 || throw(ArgumentError("oversampling must be nonnegative"))
    isfinite(boundary_margin_log10) && boundary_margin_log10 >= 0 ||
        throw(ArgumentError("boundary_margin_log10 must be finite and nonnegative"))

    LQtild = LQtilde(Q, L)
    Ltilde, Qtilde = LQtild.Ltilde, LQtild.Qtilde
    hierarchy = instanton_hierarchy_diagnostics(Ltilde;
        gap_log10=hierarchy_gap_log10, min_block_size=hierarchy_min_block_size)
    if length(hierarchy.blocks) <= 32
        hierarchy = instanton_hierarchy_diagnostics(K, Ltilde, Qtilde;
            gap_log10=hierarchy_gap_log10, min_block_size=hierarchy_min_block_size)
    end
    if min_log10_mass > max_log10_mass
        diagnostics = SpectrumWindowDiagnostics(NTuple{3,Int}[], 0, 0, Inf, Inf,
            boundary_margin_log10, 0.0, true, false, true, false, hierarchy)
        return _empty_physical_spectrum(min_log10_mass, prec, size(K, 1);
            min_log10_mass, max_log10_mass, diagnostics)
    end

    W, Kfactor, lower_count, upper_count, count_records, counts_stable =
        _confirm_window_counts(K, Ltilde, Qtilde, min_log10_mass,
            max_log10_mass, boundary_margin_log10, prec, max_prec, confirm, label)
    target_count = max(lower_count - upper_count, 0)
    if target_count == 0
        diagnostics = SpectrumWindowDiagnostics(count_records, lower_count, upper_count,
            Inf, Inf, boundary_margin_log10, 0.0, true, false, counts_stable,
            !counts_stable, hierarchy)
        return _empty_physical_spectrum(min_log10_mass, prec, size(K, 1);
            min_log10_mass, max_log10_mass, diagnostics)
    end
    setprecision(ArbFloat; digits=prec)
    T = eltype(W)
    h11 = size(K, 1)
    mass_offset = 9.0 + Float64(log10(constants()["MPlanck"])) +
        Float64(constants()["log2π"])

    # Treat charge-coupling failures as a failed hierarchy certificate.  The
    # targeted Schur path remains valid only when every proposed split has
    # passed the conservative scale-and-coupling screen; otherwise use the
    # complete high-precision reference eigensystem.
    hierarchy_safe = length(hierarchy.blocks) <= 32 &&
        all(split.certified_safe for split in hierarchy.perturbative_splits)
    fallback_used = !schur_acceleration || !hierarchy_safe
    refined_values = T[]
    refined_basis = zeros(T, h11, 0)
    residuals = Float64[]
    if schur_acceleration
        float_values, float_vectors = eigen(leading_hessian_matrix_float64_scaled(K, Ltilde, Qtilde))
        positive_float_order = sortperm(
            findall(value -> value > 0, float_values);
            by=index -> float_values[index])
        if length(positive_float_order) >= lower_count
            positive_count = length(positive_float_order)
            target_global_start = positive_count - lower_count + 1
            target_global_end = positive_count - upper_count
            candidate_start = max(1, target_global_start - oversampling)
            candidate_end = min(positive_count, target_global_end + oversampling)
            candidates = positive_float_order[candidate_start:candidate_end]
            target_start = findfirst(==(positive_float_order[target_global_start]), candidates)
            target_end = findfirst(==(positive_float_order[target_global_end]), candidates)
            if target_start !== nothing && target_end !== nothing
                complement = setdiff(collect(1:h11), candidates)
                candidate_basis = hcat(float_vectors[:, complement],
                    float_vectors[:, candidates])
                refine_count = length(candidates)
                try
                    refined_values, refined_basis, residuals = schur_physical_basis(
                        W, candidate_basis, refine_count; maxiter, residual_tolerance)
                catch err
                    if err isa SingularException || err isa PosDefException
                        fallback_used = true
                    else
                        rethrow()
                    end
                end
                if !fallback_used
                    refined_masses = _mass_values(refined_values, mass_offset)
                    inside = findall(mass -> mass >= min_log10_mass &&
                        mass <= max_log10_mass, refined_masses)
                    if length(inside) != target_count ||
                            any(!isfinite, refined_masses[inside])
                        fallback_used = true
                    end
                end
            else
                fallback_used = true
            end
        else
            fallback_used = true
        end
    end

    if fallback_used
        all_values, all_vectors = eigen(W)
        all_masses = _mass_values(all_values, mass_offset)
        all_order = sortperm(all_masses)
        inside = [index for index in all_order if all_values[index] > 0 &&
            all_masses[index] >= min_log10_mass && all_masses[index] <= max_log10_mass]
        refined_values = all_values[inside]
        refined_basis = all_vectors[:, inside]
        masses = all_masses[inside]
        mode_indices = Int[findfirst(==(index), all_order) - 1 for index in inside]
        residuals = [Float64(norm(W * @view(refined_basis[:, i]) -
            refined_values[i] .* @view(refined_basis[:, i])) /
            max(abs(refined_values[i]), eps(T))) for i in axes(refined_basis, 2)]
    else
        refined_masses = _mass_values(refined_values, mass_offset)
        inside = findall(mass -> mass >= min_log10_mass && mass <= max_log10_mass,
            refined_masses)
        refined_values = refined_values[inside]
        refined_basis = refined_basis[:, inside]
        masses = refined_masses[inside]
        mode_indices = collect(h11 - lower_count:h11 - upper_count - 1)
    end
    order = sortperm(masses)
    masses = masses[order]
    refined_values = refined_values[order]
    refined_basis = refined_basis[:, order]
    mode_indices = mode_indices[order]
    interval_ok = all(mass -> mass >= min_log10_mass && mass <= max_log10_mass, masses)
    max_residual = isempty(residuals) ? 0.0 : maximum(residuals)
    converged = max_residual <= residual_tolerance
    provisional = !counts_stable || !converged || !interval_ok ||
        length(masses) != target_count
    if provisional
        @warn "window spectrum did not satisfy residual or interval validation for geometry=$(label); returning a provisional result"
    end
    lower_boundary_gap = isempty(masses) || isinf(min_log10_mass) ? Inf :
        minimum(abs.(masses .- min_log10_mass))
    upper_boundary_gap = isempty(masses) || isinf(max_log10_mass) ? Inf :
        minimum(abs.(masses .- max_log10_mass))
    diagnostics = SpectrumWindowDiagnostics(count_records, lower_count, upper_count,
        lower_boundary_gap, upper_boundary_gap, boundary_margin_log10,
        max_residual, converged, fallback_used, !provisional, provisional,
        hierarchy)
    _physical_spectrum_from_basis(K, L, Q, refined_basis, masses, mode_indices;
        threshold_log10=min_log10_mass, min_log10_mass, max_log10_mass, prec,
        quartics, mixed_quartics, quartic_backend, diagnostics)
end

"""Load one geometry and delegate to [`pq_window_spectrum`](@ref)."""
function pq_window_spectrum(h11::Int, tri::Int, cy::Int; kwargs...)
    pot_data = potential(h11, tri, cy)
    pq_window_spectrum(pot_data.K, pot_data.L, pot_data.Q; kwargs...)
end

"""Load one indexed geometry and delegate to [`pq_window_spectrum`](@ref)."""
function pq_window_spectrum(geom_idx::GeometryIndex; kwargs...)
    pot_data = potential(geom_idx)
    pq_window_spectrum(pot_data.K, pot_data.L, pot_data.Q; kwargs...)
end

function positive_inertia(factor::BunchKaufman)
    D, pivots = factor.D, factor.ipiv
    positive = 0
    i = 1
    while i <= length(pivots)
        if pivots[i] > 0
            positive += D[i, i] > 0
            i += 1
        else
            block = Hermitian([D[i, i] D[i + 1, i]; D[i + 1, i] D[i + 1, i + 1]])
            positive += count(eigen(block).values .> 0)
            i += 2
        end
    end
    positive
end

"""
    pq_physical_mode_count(K, L, Q; threshold_log10=log10(H₀), prec=1_000,
                           confirm=true, max_prec=4_000, label="matrix input")

Count PQ leading-Hessian modes above `threshold_log10` using the inertia of
the arbitrary-precision shifted Hessian. This avoids computing the complete
eigensystem and is a building block for a future threshold-targeted physical
sector solver. It does not return eigenvectors or quartics.

With `confirm=true` (the default), the count must agree after increasing the
working precision before it is returned. If it does not stabilize by
`max_prec`, a warning is emitted and the latest count is returned as
provisional. This keeps long scans running while flagging geometries that
need a higher-precision follow-up. `label` identifies the input in that
warning; geometry-based overloads supply it automatically.
"""
function physical_mode_inertia_count(W::Hermitian, threshold_log10::Float64)
    T = eltype(W)
    mass_offset = T(9) + log10(T(constants()["MPlanck"])) + T(constants()["log2π"])
    threshold_eigenvalue = T(10) ^ (2 * (T(threshold_log10) - mass_offset))
    positive_inertia(bunchkaufman(Hermitian(Matrix(W) - threshold_eigenvalue * I)))
end

"""Count physical modes after constructing the high-precision leading Hessian."""
function physical_mode_inertia_count(K::Hermitian{Float64, Matrix{Float64}}, Ltilde::Matrix{Float64}, Qtilde::Matrix{Int}, threshold_log10::Float64, prec::Int)
    W, _ = high_precision_leading_hessian(K, Ltilde, Qtilde; prec)
    physical_mode_inertia_count(W, threshold_log10)
end

"""Confirm a physical-mode count by repeating it at increasing precision."""
function confirm_physical_mode_count(count_at_prec::Int, K::Hermitian{Float64, Matrix{Float64}}, Ltilde::Matrix{Float64}, Qtilde::Matrix{Int}, threshold_log10::Float64, prec::Int, max_prec::Int, label::AbstractString)
    working_prec = prec
    while working_prec < max_prec
        working_prec = min(2 * working_prec, max_prec)
        confirmed_count = physical_mode_inertia_count(K, Ltilde, Qtilde, threshold_log10, working_prec)
        confirmed_count == count_at_prec && return count_at_prec
        count_at_prec = confirmed_count
    end
    @warn "physical-mode count did not stabilize by max_prec=$(max_prec) for geometry=$(label); returning provisional count $(count_at_prec). Increase max_prec or inspect the threshold neighbourhood."
    count_at_prec
end

"""
    pq_physical_mode_count(K, L, Q; kwargs...)

Count leading-Hessian modes above the physical mass threshold, optionally
confirming the count at increasing arbitrary precision.
"""
function pq_physical_mode_count(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; threshold_log10::Float64=Float64(log10(constants()["Hubble"])), prec::Int=1_000, confirm::Bool=true, max_prec::Int=4_000, label::AbstractString="matrix input")
    LQtild = LQtilde(Q, L)
    Ltilde, Qtilde = LQtild.Ltilde, LQtild.Qtilde
    count_at_prec = physical_mode_inertia_count(K, Ltilde, Qtilde, threshold_log10, prec)
    !confirm && return count_at_prec
    confirm_physical_mode_count(count_at_prec, K, Ltilde, Qtilde, threshold_log10, prec, max_prec, label)
end

"""Load one geometry and delegate to [`pq_physical_mode_count(K, L, Q; kwargs...)`](@ref), supplying its identifier as the warning label."""
function pq_physical_mode_count(h11::Int, tri::Int, cy::Int; label::AbstractString="h11=$(h11), polytope=$(tri), frst=$(cy)", kwargs...)
    pot_data = potential(h11, tri, cy)
    pq_physical_mode_count(pot_data.K, pot_data.L, pot_data.Q; label, kwargs...)
end

"""Load one indexed geometry and delegate to [`pq_physical_mode_count(K, L, Q; kwargs...)`](@ref), supplying its identifier as the warning label."""
function pq_physical_mode_count(geom_idx::GeometryIndex; label::AbstractString="h11=$(geom_idx.h11), polytope=$(geom_idx.polytope), frst=$(geom_idx.frst)", kwargs...)
    pot_data = potential(geom_idx)
    pq_physical_mode_count(pot_data.K, pot_data.L, pot_data.Q; label, kwargs...)
end

"""
    pq_schur_admissible(K, L, Q; threshold_log10=log10(H₀), prec=1_000)

Return whether a Float64-seeded physical/complement split is safe for a Schur
solver at `threshold_log10`. The check is performed at arbitrary precision:
after selecting the physical-count largest Float64 seed vectors, it verifies
by inertia that the complementary high-precision block has no eigenvalue above
the threshold. Only in that case is its Schur resolvent nonsingular throughout
the requested physical sector.

This is an opt-in diagnostic for a future Schur accelerator. It is not cheap:
it needs a dense arbitrary-precision complement factorization, so it should
not be enabled in a large default scan.
"""
function pq_schur_admissible(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; threshold_log10::Float64=Float64(log10(constants()["Hubble"])), prec::Int=1_000, label::AbstractString="matrix input")
    LQtild = LQtilde(Q, L)
    Ltilde, Qtilde = LQtild.Ltilde, LQtild.Qtilde
    physical_count = pq_physical_mode_count(K, L, Q; threshold_log10, prec, label)
    physical_count == 0 && return true
    setprecision(ArbFloat; digits=prec)
    T = typeof(ArbFloat(0))
    h11 = size(K, 1)
    physical_count == h11 && return true
    W, _ = high_precision_leading_hessian(K, Ltilde, Qtilde; prec)
    _, float_vectors = eigen(leading_hessian_matrix_float64(K, Ltilde, Qtilde))
    complement = T.(float_vectors[:, 1:end-physical_count])
    C = Hermitian(complement' * W * complement)
    mass_offset = T(9) + log10(T(constants()["MPlanck"])) + T(constants()["log2π"])
    threshold_eigenvalue = T(10) ^ (2 * (T(threshold_log10) - mass_offset))
    positive_inertia(bunchkaufman(Hermitian(Matrix(C) - threshold_eigenvalue * I))) == 0
end

"""Load one geometry and delegate to [`pq_schur_admissible(K, L, Q; kwargs...)`](@ref)."""
function pq_schur_admissible(h11::Int, tri::Int, cy::Int; label::AbstractString="h11=$(h11), polytope=$(tri), frst=$(cy)", kwargs...)
    pot_data = potential(h11, tri, cy)
    pq_schur_admissible(pot_data.K, pot_data.L, pot_data.Q; label, kwargs...)
end

"""Load one indexed geometry and delegate to [`pq_schur_admissible(K, L, Q; kwargs...)`](@ref)."""
function pq_schur_admissible(geom_idx::GeometryIndex; label::AbstractString="h11=$(geom_idx.h11), polytope=$(geom_idx.polytope), frst=$(geom_idx.frst)", kwargs...)
    pot_data = potential(geom_idx)
    pq_schur_admissible(pot_data.K, pot_data.L, pot_data.Q; label, kwargs...)
end

"""
    pq_hp_alignment(K, L, Q; prec=1_000)

Refines the full PQ mode subspace against the high-precision canonical Hessian
with a PQ-seeded Rayleigh--Ritz step. The result labels the refined modes by
the PQ ordering: `permutation[i]` is the high-precision eigenmode assigned to
PQ mode `i`, and the corresponding column has been sign-fixed for positive
overlap with that PQ direction. This is a high-precision validation utility,
not part of the default `pq_spectrum` path.

Returns a named tuple containing the PQ frame, arbitrary-precision
eigenvalues, sign/permutation alignment, overlaps, and eigenpair residuals.
"""
function pq_hp_alignment(K::Hermitian{Float64, Matrix{Float64}}, L::Matrix{Float64}, Q::Matrix{Int}; prec=1_000)
    h11 = size(K, 1)
    Kls = cholesky(K).L
    LQtild = LQtilde(Q, L)
    Qtilde, Ltilde = LQtild.Qtilde, LQtild.Ltilde

    # Reproduce PQ's mass ordering and build its canonical orthonormal frame.
    Qleading = Matrix(Qtilde') / Kls'
    _, mapprox, P = pq_canonical_frame(Qleading, Ltilde)
    P = P[:, sortperm(mapprox)]

    W, _ = high_precision_leading_hessian(K, L, Q; prec)
    T = eltype(W)

    # Orthonormalize the Float64 PQ frame again at the requested precision.
    Pseed = T.(P)
    for i in 1:h11
        for j in 1:i-1
            Pseed[:, i] .-= dot(@view(Pseed[:, j]), @view(Pseed[:, i])) .* @view(Pseed[:, j])
        end
        Pseed[:, i] ./= sqrt(dot(@view(Pseed[:, i]), @view(Pseed[:, i])))
    end

    # A full-space Rayleigh--Ritz refinement is equivalent to diagonalizing W,
    # but preserves an explicit connection to the PQ seed directions.
    projected = Hermitian(Pseed' * W * Pseed)
    eigenvalues, coefficients = eigen(projected)
    refined = Pseed * coefficients
    overlap = Float64.(Pseed' * refined)

    # Greedy maximum-overlap assignment; process the least ambiguous PQ modes
    # first so a clean one-to-one sign/permutation map is obtained.
    permutation = zeros(Int, h11)
    available = trues(h11)
    confidence = [maximum(abs.(@view overlap[i, :])) for i in 1:h11]
    for i in sortperm(confidence, rev=true)
        candidates = findall(available)
        j = candidates[argmax(abs.(@view overlap[i, candidates]))]
        permutation[i] = j
        available[j] = false
    end
    signs = [overlap[i, permutation[i]] < 0 ? -1 : 1 for i in 1:h11]
    aligned_modes = refined[:, permutation] * Diagonal(ArbFloat.(signs))
    aligned_overlap = [Float64(dot(@view(Pseed[:, i]), @view(aligned_modes[:, i]))) for i in 1:h11]
    residuals = [Float64(norm(W * @view(aligned_modes[:, i]) - eigenvalues[permutation[i]] * @view(aligned_modes[:, i]))) for i in 1:h11]

    return (; pq_basis=P, eigenvalues, permutation, signs, overlap, aligned_overlap, residuals)
end


"""
	spectra_generator(h11_min, h11_max, h11list)
Generates multiple axion spectra for a given set of geometries identified in `h11list` between `h11_min` and `h11_max`.

⚠️ Will generate **all** geometries with `potential` data generated between `h11_min` and `h11_max` so will be slow if this is a lot! ⚠️
"""
function pq_spectra_generator(h11_min::Int, h11_max::Int, h11list::Matrix{Int})
	spectra = []
	for col in eachcol(h11list[:, h11_min .≤ h11list[1, :] .≤ h11_max])
		geom_idx = GeometryIndex(col...)
		push!(spectra, (geom_idx, pq_spectrum(geom_idx)))
	end
	spectra
end


"""
    pq_spectrum_save(h11, tri, cy; phase=zeros(h11))

Compute and persist the fast PQ spectrum, including masses, decay constants,
mass signs, and the cubic interaction tensor in the PQ mass basis.  `phase`
is the phase-space evaluation point in the selected charge basis.  Outputs
are written under `spectrum/`, with the cubic tensor at
`spectrum/cubic/tensor` and its phase at `spectrum/cubic/phase`.

This production path uses the Float64 PQ leading-Hessian basis.  Use
[`hp_spectrum_save`](@ref) when arbitrary-precision spectrum data are needed.
"""
function pq_spectrum_save(h11::Int, tri::Int, cy::Int=1;
                          phase::AbstractVector{<:Real}=zeros(Float64, h11))
    h11 == 0 && return nothing
    pot_data = potential(h11, tri, cy)
    L, Q, K = pot_data.L, pot_data.Q, pot_data.K
    spectrum_data = pq_spectrum(K, L, Q)
    LQtild = LQtilde(Q, L)
    Kls = cholesky(K).L
    _, _, basis = leading_hessian_mass_basis_float64(K, LQtild.Ltilde, LQtild.Qtilde)
    Qmass = (Matrix(LQtild.Qtilde') / Kls') * basis
    cubic_data = cubic(phase, LQtild.Ltilde, Qmass)
    h5open(cyax_file(h11, tri, cy), "r+") do file
        f2 = haskey(file, "spectrum") ? file["spectrum"]::HDF5.Group : create_group(file, "spectrum")
        f2e = haskey(f2, "decay") ? f2["decay"]::HDF5.Group : create_group(f2, "decay")
        f2e["fpert",deflate=9] = spectrum_data.f
        f2e["fK",deflate=9] = spectrum_data.fK
        f2d = haskey(f2, "masses") ? f2["masses"]::HDF5.Group : create_group(f2, "masses")
        f2d["log10",deflate=9] = spectrum_data.m
        f2d["sign",deflate=9] = spectrum_data.msign
        f2g = haskey(f2, "cubic") ? f2["cubic"]::HDF5.Group : create_group(f2, "cubic")
        f2g["tensor",deflate=9] = cubic_data
        f2g["phase",deflate=9] = Float64.(phase)
    end
end

function Base.convert(::Type{Matrix{Int}}, x::Nemo.ZZMatrix)
    m,n = size(x)
    mat = Int[x[i,j] for i = 1:m, j = 1:n]
    return mat
end
function Base.convert(::Type{Matrix{BigInt}}, x::Nemo.ZZMatrix)
    m,n = size(x)
    mat = BigInt[x[i,j] for i = 1:m, j = 1:n]
    return mat
end
# Base.convert(::Type{Matrix{Int}}, x::Nemo.ZZMatrix) = convert(Matrix{Int}, x)
# Base.convert(::Type{Matrix{BigInt}}, x::Nemo.ZZMatrix) = convert(Matrix{BigInt}, x)


"""
    vacua(L,Q; threshold)

Compute the number of vacua given an instanton charge matrix `Q` and 2-column matrix of instanton scales `L` (in the form [sign; exponent]) and a threshold for:

``\\frac{\\Lambda_a}{|\\Lambda_j|}``

_i.e._ is the instanton contribution large enough to affect the minima.

For small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.

For larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential.\n
#Examples
```julia-repl
julia> using CYAxiverse
julia> h11,tri,cy = 10,20,1;
julia> pot_data = CYAxiverse.read.potential(h11,tri,cy);
julia> vacua_data = CYAxiverse.generate.vacua(pot_data["L"],pot_data["Q"])
Dict{String, Any} with 3 entries:
  "θ∥"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]
  "vacua"  => 3
  "Qtilde" => [0 0 … 1 0; 0 0 … 0 0; … ; 1 1 … 0 0; 0 0 … 0 0]
```
"""
function vacua(L::Matrix{Float64},Q::Matrix{Int}; threshold::Float64=0.5)
    h11::Int = size(Q,2)
    θparalleltest::Matrix{Rational} = zeros(Rational, 0, 0)
    if h11 <= 50
        snf_data = vacua_SNF(Q)
        θparalleltest = snf_data.θparallel
    end
    data = LQtildebar(L,Q; threshold=threshold)
    Qtilde = data["Qtilde"]
    
    if h11 <= 50
        vacua = Int(round(abs(det(θparalleltest) / det(inv(Qtilde)))))
        thparallel::Matrix{Rational} = Rational.(round.(θparalleltest; digits=5))
        keys = ["vacua","θ∥","Qtilde"]
        vals = [abs(vacua), thparallel, Qtilde]
        return Dict(zip(keys,vals))
    else
        vacua = Int(round(abs(1 / det(inv(Qtilde)))))
        keys = ["vacua","Qtilde"]
        vals = [abs(vacua), Qtilde]
        return Dict(zip(keys,vals))
    end
end

"""Select independent charge columns using a reusable Gram--Schmidt workspace."""
function leading_independent_mask!(tilde_mask::AbstractVector{Bool}, Q::AbstractMatrix{Int}, order::AbstractVector{Int}, orthogonal_span::AbstractMatrix{Float64}, residual::AbstractVector{Float64})
    h11, ncols = size(Q)
    length(order) == ncols || throw(DimensionMismatch("order must contain one entry per charge column"))
    length(tilde_mask) == ncols || throw(DimensionMismatch("tilde_mask must contain one entry per charge column"))
    fill!(tilde_mask, false)
    fill!(orthogonal_span, 0.0)
    current_rank = 0

    @inbounds for ordered_idx in 1:ncols
        idx = order[ordered_idx]
        original_norm_squared = 0.0
        for i in 1:h11
            value = Float64(Q[i, idx])
            residual[i] = value
            original_norm_squared += value * value
        end

        # A second modified Gram--Schmidt pass keeps the residual reliable when
        # the already-selected charge vectors are poorly conditioned.
        for _ in 1:2, j in 1:current_rank
            projection = 0.0
            for i in 1:h11
                projection += orthogonal_span[i, j] * residual[i]
            end
            for i in 1:h11
                residual[i] -= projection * orthogonal_span[i, j]
            end
        end

        residual_norm_squared = 0.0
        for i in 1:h11
            residual_norm_squared += residual[i] * residual[i]
        end
        if residual_norm_squared > eps(Float64) * original_norm_squared
            current_rank += 1
            inverse_norm = inv(sqrt(residual_norm_squared))
            for i in 1:h11
                orthogonal_span[i, current_rank] = residual[i] * inverse_norm
            end
            tilde_mask[idx] = true
            current_rank == h11 && break
        end
    end
    current_rank
end

"""Backward-compatible wrapper for callers that already hold ordered charges."""
function leading_independent_mask!(tilde_mask::AbstractVector{Bool}, Qordered::AbstractMatrix{Int},
        orthogonal_span::AbstractMatrix{Float64}, residual::AbstractVector{Float64})
    leading_independent_mask!(tilde_mask, Qordered, Base.OneTo(size(Qordered, 2)),
        orthogonal_span, residual)
end

"""
    LQtilde(Q, L)

Order instantons by decreasing `L[2, :]`, select the first `h11` linearly
independent charge columns, and return the selected (`tilde`) and remaining
(`bar`) potential data. Selection uses a preallocated, reorthogonalized
Gram--Schmidt workspace, so scanning candidates does not allocate per column.
"""
function LQtilde(Q::AbstractMatrix{Int}, L::AbstractMatrix{Float64})
    @assert size(Q, 1) < size(Q, 2) "Looks like you need to transpose..."
    h11 = size(Q, 1)
    
    # Keep only the ordering permutation. Materializing Q[:, perm] and L[:, perm]
    # here duplicates the full instanton data, which becomes significant for
    # large-h11 geometries with many subleading instantons.
    perm = sortperm(@view(L[2, :]), rev=true)

    ncols = size(Q, 2)
    tilde_mask = fill(false, ncols)
    
    # Maintain an orthonormal span while scanning. Repeated calls to `rank`
    # trigger a dense decomposition for every candidate and do not scale to
    # the large instanton sets at high h11.
    orthogonal_span = zeros(Float64, h11, h11)
    residual = zeros(Float64, h11)
    leading_independent_mask!(tilde_mask, Q, perm, orthogonal_span, residual)

    selected_indices = [index for index in perm if tilde_mask[index]]
    remaining_indices = [index for index in perm if !tilde_mask[index]]
    Qtilde = Q[:, selected_indices]
    Ltilde = L[:, selected_indices]
    Qbar = Q[:, remaining_indices]
    Lbar = L[:, remaining_indices]

    return LQLinearlyIndependent(Qtilde, Qbar, Lbar, Ltilde)
end

"""Load one geometry and select its leading linearly independent instantons."""
function LQtilde(h11::Int, tri::Int, cy::Int; hilbert = false)
    pot_data = potential(h11, tri, cy; hilbert = hilbert)
    return LQtilde(Matrix{Int}(pot_data.Q), Matrix{Float64}(pot_data.L))
end	

"""Select leading instantons for a geometry identified by `geom_idx`."""
function LQtilde(geom_idx::GeometryIndex; hilbert = false)
    pot_data = potential(geom_idx; hilbert = hilbert)
    return LQtilde(Matrix{Int}(pot_data.Q), Matrix{Float64}(pot_data.L))
end	

"""
    reduced_critical_points(L, Q; kwargs...)

Deterministically find and classify critical points in the leading-charge
coordinates used in the axion-minima papers. Each stationarity equation is
scaled by its corresponding leading instanton amplitude, avoiding loss of the
hierarchically suppressed directions.
"""
function reduced_critical_points(L::AbstractMatrix{Float64}, Q::AbstractMatrix{Int}; kwargs...)
    selected = LQtilde(Q, L)
    Lordered = hcat(selected.Ltilde, selected.Lbar)
    Qordered = hcat(selected.Qtilde, selected.Qbar)
    leading_logs = @view selected.Ltilde[2, :]
    equation_scales = 10.0 .^ clamp.(leading_logs .- maximum(leading_logs),
        log10(floatmin(Float64)), 0.0)
    critical_points(Lordered, Qordered;
        coordinate_basis=selected.Qtilde, equation_scales=equation_scales, kwargs...)
end

"""Return the sup-norm distance between two points on the unit torus."""
function _torus_distance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    maximum(min.(abs.(a .- b), 1 .- abs.(a .- b)))
end

"""Compute an exact integer determinant without dense BigInt work when safe."""
function _exact_integer_determinant(matrix::AbstractMatrix{Int})
    size(matrix, 1) == size(matrix, 2) ||
        throw(ArgumentError("exact integer determinant requires a square matrix"))
    dimension = size(matrix, 1)
    dimension == 0 && return big(1)
    work = Matrix{Int}(matrix)
    sign = 1
    denominator = 1
    try
        @inbounds for pivot_index in 1:(dimension - 1)
            pivot_row = pivot_index
            while pivot_row <= dimension && iszero(work[pivot_row, pivot_index])
                pivot_row += 1
            end
            pivot_row > dimension && return big(0)
            if pivot_row != pivot_index
                for column in 1:dimension
                    work[pivot_index, column], work[pivot_row, column] =
                        work[pivot_row, column], work[pivot_index, column]
                end
                sign = -sign
            end
            pivot = work[pivot_index, pivot_index]
            for row in (pivot_index + 1):dimension
                entry = work[row, pivot_index]
                work[row, pivot_index] = 0
                for column in (pivot_index + 1):dimension
                    numerator = Base.checked_sub(
                        Base.checked_mul(work[row, column], pivot),
                        Base.checked_mul(entry, work[pivot_index, column]))
                    if denominator != 1
                        rem(numerator, denominator) == 0 ||
                            return det(Matrix{BigInt}(matrix))
                        numerator ÷= denominator
                    end
                    work[row, column] = numerator
                end
            end
            denominator = pivot
        end
        BigInt(Base.checked_mul(sign, work[dimension, dimension]))
    catch error
        error isa OverflowError || rethrow()
        det(Matrix{BigInt}(matrix))
    end
end

"""Check whether `θ` is already represented among the first `count` columns."""
function _contains_torus_point(points::AbstractMatrix{<:Real}, θ::AbstractVector{<:Real},
        count::Int; tol::Float64)
    for i in 1:count
        _torus_distance(@view(points[:, i]), θ) <= tol && return true
    end
    false
end

"""
    leading_lattice_offsets(selected; tolerance=1e-8)

Enumerate the finite quotient-lattice offsets solving
`Qtilde' * θ = 0 mod 1` for the leading selected charge matrix.  The returned
matrix has one axion-space offset per column and `abs(det(Qtilde))` columns for
full-rank square `Qtilde`.
"""
function leading_lattice_offsets(selected::LQLinearlyIndependent;
        tolerance::Float64=1e-8,
        det_qtilde_big::Union{Nothing,BigInt}=nothing)
    h11 = size(selected.Qtilde, 1)
    size(selected.Qtilde, 2) == h11 ||
        throw(ArgumentError("Qtilde must be square to enumerate leading lattice offsets"))

    exact_det = det_qtilde_big === nothing ?
        abs(_exact_integer_determinant(selected.Qtilde)) : abs(det_qtilde_big)
    exact_det <= BigInt(typemax(Int)) ||
        throw(ArgumentError("abs(det(Qtilde)) is too large to materialize lattice offsets"))
    det_qtilde = Int(exact_det)
    det_qtilde > 0 || throw(ArgumentError("Qtilde must be nonsingular"))

    inverse_transpose = inv(transpose(Float64.(selected.Qtilde)))
    offsets = zeros(Float64, h11, det_qtilde)
    offsets[:, 1] .= 0.0
    count = 1
    cursor = 1
    while cursor <= count
        base = @view offsets[:, cursor]
        for i in 1:h11
            θ = mod.(base .+ @view(inverse_transpose[:, i]), 1.0)
            if !_contains_torus_point(offsets, θ, count; tol=tolerance)
                count += 1
                count <= det_qtilde ||
                    error("lattice enumeration exceeded abs(det(Qtilde))")
                offsets[:, count] .= θ
            end
        end
        cursor += 1
    end
    count == det_qtilde ||
        @warn "lattice enumeration did not reach abs(det(Qtilde))" found=count expected=det_qtilde
    offsets[:, 1:count]
end

"""Return the exact absolute determinant of a square integer charge matrix."""
function _leading_det_qtilde(selected::LQLinearlyIndependent)
    h11 = size(selected.Qtilde, 1)
    size(selected.Qtilde, 2) == h11 ||
        throw(ArgumentError("Qtilde must be square"))
    abs(_exact_integer_determinant(selected.Qtilde))
end

"""Normalize the optional leading-index filter and validate its bounds."""
function _leading_negative_mode_range(h11::Int,
        negative_mode_range::Union{Nothing,UnitRange{Int}},
        max_negative_modes::Union{Nothing,Int})
    negative_mode_range === nothing || max_negative_modes === nothing ||
        throw(ArgumentError("specify only one of negative_mode_range and max_negative_modes"))
    if max_negative_modes !== nothing
        0 <= max_negative_modes <= h11 ||
            throw(ArgumentError("max_negative_modes must lie between 0 and h11"))
        return 0:max_negative_modes
    end
    negative_mode_range === nothing && return nothing
    isempty(negative_mode_range) &&
        throw(ArgumentError("negative_mode_range must not be empty"))
    first(negative_mode_range) >= 0 && last(negative_mode_range) <= h11 ||
        throw(ArgumentError("negative_mode_range must lie between 0 and h11"))
    negative_mode_range
end

"""Count half-integer masks in an exact leading-index range."""
function _leading_mask_count(h11::Int, signs::AbstractVector{<:Real},
        negative_mode_range::Union{Nothing,UnitRange{Int}})
    total_masks = BigInt(2)^h11
    negative_mode_range === nothing && return total_masks
    nonzero_count = count(!iszero, signs)
    zero_count = h11 - nonzero_count
    sum(index <= nonzero_count ? binomial(BigInt(nonzero_count), index) : big(0)
        for index in negative_mode_range; init=big(0)) * (BigInt(2)^zero_count)
end

"""Visit masks in increasing integer order without materializing a mask list."""
function _foreach_leading_mask(callback::Function, bit::Int, mask::BigInt,
        baseline::BigInt, zero_signs::BigInt, negative_modes::Int,
        minimum_index::Int, maximum_index::Int)
    remaining = bit + 1
    negative_modes > maximum_index && return nothing
    negative_modes + remaining < minimum_index && return nothing
    if bit < 0
        minimum_index <= negative_modes <= maximum_index && callback(mask, negative_modes)
        return nothing
    end

    baseline_bit = Int((baseline >> bit) & big(1))
    zero_sign = ((zero_signs >> bit) & big(1)) == big(1)
    # The zero branch is visited first so that the resulting BigInt mask is
    # visited in increasing numeric order, matching the legacy mask loop.
    zero_flip = zero_sign ? 0 : baseline_bit
    _foreach_leading_mask(callback, bit - 1, mask,
        baseline, zero_signs, negative_modes + zero_flip,
        minimum_index, maximum_index)

    one_flip = zero_sign ? 0 : 1 - baseline_bit
    bit_mask = big(1) << bit
    _foreach_leading_mask(callback, bit - 1, mask | bit_mask,
        baseline, zero_signs, negative_modes + one_flip,
        minimum_index, maximum_index)
    nothing
end

"""Visit filtered masks without allocating BigInt intermediates per node."""
function _foreach_leading_mask_bits(callback::Function, bit::Int,
        selected::BitVector, baseline::BitVector, zero_signs::BitVector,
        negative_modes::Int, minimum_index::Int, maximum_index::Int)
    remaining = bit + 1
    negative_modes > maximum_index && return nothing
    negative_modes + remaining < minimum_index && return nothing
    if bit < 0
        minimum_index <= negative_modes <= maximum_index &&
            callback(selected, negative_modes)
        return nothing
    end

    index = bit + 1
    selected[index] = false
    zero_flip = zero_signs[index] ? 0 : (baseline[index] ? 1 : 0)
    _foreach_leading_mask_bits(callback, bit - 1, selected, baseline,
        zero_signs, negative_modes + zero_flip, minimum_index, maximum_index)

    selected[index] = true
    one_flip = zero_signs[index] ? 0 : (baseline[index] ? 0 : 1)
    _foreach_leading_mask_bits(callback, bit - 1, selected, baseline,
        zero_signs, negative_modes + one_flip, minimum_index, maximum_index)
    selected[index] = false
    nothing
end

"""Return a compact report for a streamed leading-branch search."""
function _leading_stream_report(branch_count::BigInt, mask_count::BigInt,
        masks_visited::BigInt, det_qtilde::BigInt,
        negative_mode_range::Union{Nothing,UnitRange{Int}})
    (; branch_count, mask_count, masks_visited,
       masks_skipped=mask_count - masks_visited,
       lattice_copy_count=det_qtilde,
       lattice_copies_visited=masks_visited * det_qtilde,
       negative_mode_range,
       search_classification=negative_mode_range === nothing ?
           :complete_enumeration : :deterministic_low_index_enumeration)
end

"""
    foreach_leading_critical_branch(callback, selected; tolerance=1e-8,
        max_branches=1_000_000, negative_mode_range=nothing,
        max_negative_modes=nothing)

Stream the leading half-integer critical branches to `callback` without
materializing the full coordinate matrix. The callback receives
`(theta, leading_negative_modes)`; `theta` is a reused mutable vector and must
not be retained or mutated. `negative_mode_range=1:2` keeps only branches
whose leading selected potential has one or two negative modes, while
`max_negative_modes=2` is shorthand for `0:2`. The exact branch count is
checked against `max_branches` before allocating lattice offsets or branch
workspaces. The return value reports the exact branch count, masks visited and
skipped, lattice-copy counts, and the search classification.

This is the memory-bounded counterpart to [`leading_critical_branches`](@ref).
"""
function foreach_leading_critical_branch(callback::Function,
        selected::LQLinearlyIndependent; tolerance::Float64=1e-8,
        max_branches::Int=1_000_000,
        negative_mode_range::Union{Nothing,UnitRange{Int}}=nothing,
        max_negative_modes::Union{Nothing,Int}=nothing)
    h11 = size(selected.Qtilde, 1)
    max_branches > 0 || throw(ArgumentError("max_branches must be positive"))
    index_range = _leading_negative_mode_range(h11,
        negative_mode_range, max_negative_modes)
    signs = @view selected.Ltilde[1, :]
    mask_count = _leading_mask_count(h11, signs, index_range)
    det_qtilde = _leading_det_qtilde(selected)
    det_qtilde > 0 || throw(ArgumentError("Qtilde must be nonsingular"))
    branch_count = det_qtilde * mask_count
    branch_count <= BigInt(max_branches) ||
        throw(ArgumentError("leading branch enumeration would create $branch_count branches; increase max_branches explicitly if intended"))

    # An index range can be disjoint from the attainable inertia values (for
    # example, a zero-amplitude selected instanton contributes no negative
    # mode).  In that case the exact result is empty and no numerical buffers
    # or lattice offsets need to be allocated.
    mask_count == 0 && return _leading_stream_report(
        branch_count, BigInt(2)^h11, big(0), det_qtilde, index_range)

    offsets = leading_lattice_offsets(selected; tolerance=tolerance,
        det_qtilde_big=det_qtilde)

    inverse_transpose = inv(transpose(Float64.(selected.Qtilde)))
    half_phase = zeros(Float64, h11)
    base = zeros(Float64, h11)
    theta = zeros(Float64, h11)
    visit_mask = function(mask::BigInt, negative_modes::Int)
        @inbounds for i in 1:h11
            half_phase[i] = ((mask >> (i - 1)) & big(1)) == big(1) ? 0.5 : 0.0
        end
        LinearAlgebra.mul!(base, inverse_transpose, half_phase)
        for j in axes(offsets, 2)
            @inbounds for i in 1:h11
                theta[i] = mod(offsets[i, j] + base[i], 1.0)
            end
            callback(theta, negative_modes)
        end
        nothing
    end

    if index_range === nothing
        mask_count_int = Int(mask_count)
        for mask in 0:(mask_count_int - 1)
            visit_mask(BigInt(mask), count(i -> signs[i] *
                (((mask >> (i - 1)) & 1) == 0 ? 1.0 : -1.0) < 0.0, 1:h11))
        end
    else
        baseline = BitVector(undef, h11)
        zero_signs = BitVector(undef, h11)
        @inbounds for i in 1:h11
            baseline[i] = signs[i] < 0.0
            zero_signs[i] = iszero(signs[i])
        end
        visit_bits = function(mask_bits::BitVector, negative_modes::Int)
            @inbounds for i in 1:h11
                half_phase[i] = mask_bits[i] ? 0.5 : 0.0
            end
            LinearAlgebra.mul!(base, inverse_transpose, half_phase)
            for j in axes(offsets, 2)
                @inbounds for i in 1:h11
                    theta[i] = mod(offsets[i, j] + base[i], 1.0)
                end
                callback(theta, negative_modes)
            end
            nothing
        end
        _foreach_leading_mask_bits(visit_bits, h11 - 1,
            falses(h11), baseline, zero_signs, 0,
            first(index_range), last(index_range))
    end
    masks_visited = index_range === nothing ? mask_count :
        _leading_mask_count(h11, signs, index_range)
    _leading_stream_report(branch_count, BigInt(2)^h11,
        masks_visited,
        det_qtilde, index_range)
end

"""Stream leading critical branches with the selected matrix as first argument."""
function foreach_leading_critical_branch(selected::LQLinearlyIndependent,
        callback::Function; kwargs...)
    foreach_leading_critical_branch(callback, selected; kwargs...)
end

"""
    leading_critical_branches(selected; tolerance=1e-8, max_branches=1_000_000)

Enumerate the leading half-integer critical branches
`Qtilde' * θ ∈ {0, 1/2}^h11 mod 1`, including the quotient-lattice copies from
`abs(det(Qtilde))`.  This gives a cheap deterministic prefilter for vacua and
inflation scans when the leading instantons are strongly hierarchical.  The
returned coordinates are in the original axion torus, one branch per column.

The `leading_negative_modes` entry is the Hessian inertia of the leading
selected potential only; downstream callers should still evaluate the full
potential/Hessian on any retained branches.
"""
function leading_critical_branches(selected::LQLinearlyIndependent;
        tolerance::Float64=1e-8, max_branches::Int=1_000_000,
        negative_mode_range::Union{Nothing,UnitRange{Int}}=nothing,
        max_negative_modes::Union{Nothing,Int}=nothing)
    h11 = size(selected.Qtilde, 1)
    index_range = _leading_negative_mode_range(h11,
        negative_mode_range, max_negative_modes)
    branch_count = _leading_det_qtilde(selected) *
        _leading_mask_count(h11, @view(selected.Ltilde[1, :]), index_range)
    branch_count <= BigInt(max_branches) ||
        throw(ArgumentError("leading branch enumeration would create $branch_count branches; increase max_branches explicitly if intended"))

    branch_count_int = Int(branch_count)
    coordinates = zeros(Float64, h11, branch_count_int)
    leading_negative_modes = Vector{Int}(undef, branch_count_int)
    branch_cursor = Ref(0)
    stream_report = foreach_leading_critical_branch(selected;
        tolerance, max_branches, negative_mode_range, max_negative_modes) do theta, negative_modes
        branch_cursor[] += 1
        coordinates[:, branch_cursor[]] .= theta
        leading_negative_modes[branch_cursor[]] = negative_modes
    end

    (; coordinates,
       leading_negative_modes,
       branch_count=branch_cursor[],
       leading_minima_count=count(==(0), leading_negative_modes),
       det_Qtilde=Int(_leading_det_qtilde(selected)),
       stream_report)
end

"""Enumerate leading critical branches after selecting independent charges."""
function leading_critical_branches(Q::AbstractMatrix{Int}, L::AbstractMatrix{Float64}; kwargs...)
    leading_critical_branches(LQtilde(Q, L); kwargs...)
end

"""
    αmatrix(LQtilde::NamedTuple; threshold::Float64=0.5)

TBW
"""
function αmatrix(LQ::LQLinearlyIndependent; threshold::Float64=0.5)
    Qhat = Matrix{Rational}(LQ.Qtilde)
    h11 = size(Qhat, 1)
    Qbar = LQ.Qbar
    Lhat = LQ.Ltilde
    Lbar = LQ.Lbar

    Ltilde_min = minimum(@view(Lhat[2, :]))
    Ldiff_limit = log10(threshold)

    # Use views for filtering valid components
    valid_cols = @view(Lbar[2, :]) .>= (Ltilde_min + Ldiff_limit)
    Qbar_v = @view(Qbar[:, valid_cols])
    Lbar_v = @view(Lbar[:, valid_cols])

    Qinv::Matrix{Rational} = inv(Qhat)
    @. Qinv = ifelse(abs(Qinv) < 1e-4, zero(Rational), Rational(Qinv))

    α::Matrix{Rational} = Matrix{Rational}((Qinv * Qbar_v)')
    @. α = ifelse(abs(α) < 1e-4, zero(Rational), Rational(α))
    @. α = ifelse(mod(α, 1) < 1e-3, round(α), α)

    # Pre-allocate effective vectors efficiently
    αeff_cols = Vector{Rational}[]
    αeff_indices = Int[]
    
    @inbounds for i in axes(α, 1)
        keep = false
        for j in axes(α, 2)
            if abs(α[i, j]) > 1e-3
                Ldiff = round(Lbar_v[2, i] - Lhat[2, j], digits=3)
                if Ldiff <= Ldiff_limit
                    α[i, j] = zero(Rational)
                else
                    keep = true
                end
            else
                α[i, j] = zero(Rational)
            end
        end
        if keep
            push!(αeff_cols, α[i, :])
            push!(αeff_indices, i)
        end
    end

    if !isempty(αeff_cols)
        αeff::Matrix{Rational} = hcat(αeff_cols...)
        Qbar_eff = Qbar_v[:, αeff_indices]
        Lbar_eff = Lbar_v[:, αeff_indices]
        perturbation_anchor = Lbar_v[2, first(αeff_indices)]
        αrowmask = [(L - perturbation_anchor) < -Ldiff_limit for L in @view(Lhat[2, 1:h11])]
        αcolmask = [any(!iszero, col) for col in eachcol(αeff[αrowmask, :])]
        return Canonicalα(Matrix{Int}(Qhat), Matrix{Int}(Qbar_eff), Matrix{Float64}(Lhat), Matrix{Float64}(Lbar_eff), Matrix{Rational}(αeff), Matrix{Rational}(αeff), Vector{Bool}(αrowmask), Vector{Bool}(αcolmask))
    else
        return CanonicalQBasis(Matrix{Int}(Qhat), Matrix{Int}(Qbar), Matrix{Float64}(Lhat), Matrix{Float64}(Lbar))
    end
end

function αmatrix(h11::Int, tri::Int, cy::Int; threshold::Float64 = 0.5, hilbert = false)
    αmatrix(LQtilde(h11, tri, cy; hilbert = hilbert); threshold = threshold)
end

function αmatrix(geom_idx::GeometryIndex; threshold::Float64 = 0.5, hilbert = false)
    αmatrix(LQtilde(geom_idx; hilbert = hilbert); threshold = threshold)
end

function αmatrix(Q, L; threshold::Float64 = 0.5)
    αmatrix(LQtilde(Q, L); threshold = threshold)
end
"""
    ωnorm2(LQ::CanonicalQBasis)

TBW
"""
function ωnorm2(LQ::CanonicalQBasis)
	Qhat = LQ.Qhat
	ωnorm = zeros(size(Qhat, 2))
	for i in axes(Qhat, 2)
		if length(Qhat[:, i][Qhat[:, i] .== 0]) < size(Qhat, 2) - 1
			ωnorm[i] = norm(Qhat[:, i])^2
		end
	end
	sum(ωnorm) / size(Qhat, 2)
end

function ωnorm2(LQ::Canonicalα)
    ωnorm2(CanonicalQBasis(LQ.Qhat, LQ.Qbar, LQ.Lhat, LQ.Lbar))
end

function ωnorm2(geom_idx::GeometryIndex; threshold::Float64 = 0.5)
    ωnorm2(αmatrix(LQtilde(geom_idx); threshold = threshold))
end

"""
    LQtildebar(L,Q; threshold)

Compute the linearly independent leading instantons that generate the axion potential, including any subleading instantons that are within `threshold` of their basis instanton.  Also returns `α` which is a vector of zeros if `Qhat` is square, or is a matrix with additional non-zero columns if `Qhat` is not square.\n
#Examples
```julia-repl
julia> h11,tri,cy = 12, 7, 1;
julia> pot_data = CYAxiverse.read.potential(h11,tri,cy);
julia> vacua_data = CYAxiverse.generate.LQtildebar(pot_data["L"],pot_data["Q"]; threshold=1e-2)
Dict{String, Matrix}(
"Lbar" => 2×51 Matrix{Float64}:
    1.0       1.0       1.0      -1.0    …      1.0      -1.0       1.0       1.0
 -101.342  -110.839  -156.784  -271.595     -1113.02  -1118.28  -1118.47  -1144.78

"Qhat" => 12×13 Matrix{Int64}:
 0   0  0  0  0  0  -1   0  0  0  0  0  1
 0  -2  0  0  0  0   1   0  0  0  0  0  0
 0   0  0  0  1  0  -1   2  0  0  0  0  0
 0   1  0  0  0  0  -1   2  0  1  0  0  0
 0   1  0  0  0  0   1  -2  0  0  0  0  0
 0   1  0  0  0  0  -1   0  1  0  0  0  0
 0   0  0  0  0  0   0   1  0  0  0  1  0
 0  -1  0  1  0  0   0   1  0  0  0  0  0
 0   1  0  0  0  1   0  -1  0  0  0  0  0
 0   1  1  0  0  0  -1   1  0  0  0  0  0
 1   0  0  0  0  0  -1   1  0  0  0  0  0
 0   1  0  0  0  0   0   0  0  0  1  0  0

"Lhat" => 2×13 Matrix{Float64}:
   1.0       1.0       1.0        1.0    …     1.0       1.0        1.0       1.0
 -31.7319  -77.6752  -87.1719  -249.058     -693.394  -872.027  -1143.42  -1144.78

"Qbar" => 12×51 Matrix{Int64}:
  0   0   0   0   0   0   0   0   0   0   0  …   0   0   0   0   1   0   0   0   0  1
 -2   0  -2   0   0   0   2   2  -2   0   0      0   0  -2   0  -1   0   0   0   0  0
  0   0   0   0   1   0   0   1   0   0   1      1   0   0   0   1  -2   0   0   1  0
  1   0   1   0   0   0  -1  -1   1   0   0     -1   1   1   0   2  -1   0   0   0  0
  1   0   1   0   0   0  -1  -1   1   0   0      0   0   1   0  -1   2   0   0   0  0
  1   0   1   0   0   0  -1  -1   1   0   0  …   0   0   1   0   1   0   0   0   0  0
  0   0   0   0   0   0   0   0   0   0   0      0   0   0   0   0  -1   0   0   0  0
 -1   0  -1   1   0   0   2   1  -1   1   0      0   0  -1   0   0  -1   1   0   0  0
  1   0   1   0   0   1  -1  -1   0   0   0      0  -1   1   0   0   1   0   1   0  0
  1   1   0   0   0   0  -1  -1   1  -1  -1      0   0   1   1   1  -1   0   0   0  0
 -1  -1   0  -1  -1  -1   0   0   0   0   0  …   0   0   0   0   1  -1   0   0   0  0
  1   0   1   0   0   0  -1  -1   1   0   0      0   0   0  -1   0   0  -1  -1  -1  0

"α" => 12×2 Matrix{Rational}:
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  0//1
 0//1  3//4
 )
```
"""
function LQtildebar(L::Matrix{Float64},Q::Matrix{Int}; threshold = 0.5)
    @assert size(L, 1) == 2 && size(L, 2) == size(Q, 2)
    instanton_order = sortperm(@view(L[2, :]), rev=true)
    Qsorted_test = Matrix{Int}(Q[:, instanton_order])
    Lsorted_test = Matrix{Float64}(L[:, instanton_order])
    Qtilde::Matrix{Int} = hcat(zeros(Int, size(Qsorted_test, 1)), Qsorted_test[:, 1])
    Ltilde::Matrix{Float64} = hcat(zeros(Float64, size(Lsorted_test, 1)), Lsorted_test[:, 1])
    
    S = Nemo.matrix_space(Nemo.ZZ, 1, 1)
    m::Nemo.ZZMatrix = matrix(Nemo.ZZ,zeros(1,1))
    d::Int = 1
    Qbar::Matrix{Int} = zeros(Int, size(Qsorted_test, 1), 1)
    Lbar::Matrix{Float64} = zeros(Float64, size(Lsorted_test, 1), 1)
    for i=2:size(Qsorted_test, 2)
        S = Nemo.matrix_space(Nemo.ZZ, size(Qtilde)...)
        m = S(hcat(@view(Qtilde[:, 2:end]), @view(Qsorted_test[:, i])))
        d = Nemo.nullspace(m)[1]
        if d == 0
            Qtilde = hcat(Qtilde, @view(Qsorted_test[:, i]))
            Ltilde = hcat(Ltilde, @view(Lsorted_test[:, i]))
        else
            Qbar = hcat(Qbar, @view(Qsorted_test[:, i]))
            Lbar = hcat(Lbar, @view(Lsorted_test[:, i]))
        end
    end
    Qtilde = Matrix{Rational}(@view(Qtilde[:,2:end]))
    Qbar = Matrix{Int}(@view(Qbar[:,2:end]))
    Ltilde = @view(Ltilde[:,2:end])
    Lbar = @view(Lbar[:,2:end])
    Ltilde_min::Float64 = minimum(@view(Ltilde[2,:]))
    Ldiff_limit::Float64 = log10(threshold)
    Qbar = @view(Qbar[:, @view(Lbar[2,:]) .>= (Ltilde_min + Ldiff_limit)])
    Lbar = @view(Lbar[:, @view(Lbar[2,:]) .>= (Ltilde_min + Ldiff_limit)])
    Qinv = (inv(Qtilde))
    Qinv = @.(ifelse(abs(Qinv) < 1e-10, zero(Qinv), round(Qinv; digits=4)))
    Qhat::Matrix{Int} = deepcopy(Qtilde)
    Lhat = deepcopy(Ltilde)
    αeff::Matrix{Rational} = zeros(size(Q, 1),1)
    α::Matrix{Rational} = (Qinv * Qbar)' ##Is this the same as JLM's? YES
    for i in axes(α,1)
        for j in axes(α,2)
            if abs(α[i,j]) > 1e-3
                Ldiff::Float64 = round(Lbar[2,i] - Lhat[2,j], digits=3)
                if Ldiff > Ldiff_limit
                else
                    α[i,j] = zero(Rational)
                end
            else
                α[i,j] = zero(Rational)
            end
        end
        if α[i,:] == zeros(size(α,2))
        else
            Qhat = hcat(Qhat, @view(Qbar[:,i]))
            Lhat = hcat(Lhat, @view(Lbar[:,i]))
            αeff = hcat(αeff,@view(α[i,:]))
        end
    end
    keys = ["Qhat", "Qbar", "Lhat", "Lbar", "α"]
    vals = [Qhat, Qbar, Lhat, Lbar, αeff[:,2:end]]
    return Dict(zip(keys,vals))
end

"""
    LQtildebar(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5)

TBW
"""
function LQtildebar(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5)
    pot_data = potential(h11,tri,cy)
    Q::Matrix{Int}, L::Matrix{Float64} = pot_data.Q, pot_data.L 
    LQtildebar(L, Q; threshold=threshold)
end

function LQtildebar(geom_idx::GeometryIndex; threshold::Float64=0.5)
    pot_data = potential(geom_idx)
    Q::Matrix{Int}, L::Matrix{Float64} = pot_data.Q, pot_data.L 
    LQtildebar(L, Q; threshold=threshold)
end

"""
    vacua_id_basis(L::Matrix{Float64},Q::Matrix{Int}; threshold::Float64=0.5)

Compute the number of vacua given an instanton charge matrix `Q` and 2-column matrix of instanton scales `L` (in the form [sign; exponent])  and a threshold for:

``\frac{Lambda_a}{|Lambda_j|}``

_i.e._ is the instanton contribution large enough to affect the minima.  This function uses JLM's method outlined in [TO APPEAR].

#Examples
```julia-repl
julia> using CYAxiverse
julia> h11,tri,cy = 10,20,1;
julia> pot_data = CYAxiverse.read.potential(h11,tri,cy);
julia> vacua_data = CYAxiverse.generate.vacua_id_basis(pot_data["L"],pot_data["Q"]; threshold=0.01)
Dict{String, Any} with 3 entries:
  "θ∥"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]
  "vacua"  => 11552.0
  "Qtilde" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]
```
"""
function vacua_id_basis(L::Matrix{Float64},Q::Matrix{Int}; threshold::Float64=0.5)
    if @isdefined h11
    else
        h11::Int = size(Q,1)
    end
    data = LQtildebar(L,Q; threshold=threshold)
    Leff = data["Lhat"]
    Qtilde = Matrix{Rational}(data["Qhat"][:, 1:h11])
    Qbar = Matrix{Int}(data["Qbar"])
    Qinv = Matrix{Rational}(inv(Qtilde))
    Qinv = @.(ifelse(abs(Qinv) < 1e-5, zero(Rational), Rational(Qinv)))
    αeff = data["α"]
    if αeff == zeros(Float64,size(Q,1),1)
        keys = ["θ̃∥", "vac"]
        vals = [unique(Qinv, dims=2), abs(det(Qtilde))]
        return Dict(zip(keys,vals))
    else
        αeff = @view(αeff[:,2:end])
        Qeff = hcat((1//1 * I(size(αeff,1))),αeff)
        Qrowmask = [sum(i .== zero(i[1])) < size(Qeff,2)-1 for i in eachrow(Qeff)]
        Qcolmask = [any(col .!= zero(col[1])) for col in eachcol(Qeff[Qrowmask,:])]
        keys = ["Qtilde_inv", "α", "Qeff","Leff", "Qrowmask", "Qcolmask"]
        vals = [inv(Matrix{Rational}(@view(Qtilde[:,1:size(Qtilde,1)]))), (inv(Matrix{Rational}(@view(Qtilde[:,1:size(Qtilde,1)]))) * Qbar), Qeff, Leff, Qrowmask, Qcolmask]
        return Dict(zip(keys,vals))
    end
end

function vacua_id_basis(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5)
    pot_data = potential(h11,tri,cy)
    Q::Matrix{Int}, L::Matrix{Float64} = pot_data.Q, pot_data.L 
    vacua_id_basis(L, Q; threshold=threshold)
end
"""
    vacua_id(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5, phase::Vector=zero(Q[1, :]))

TBW
"""
function vacua_id(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5,
        phase::Vector=zeros(size(Q, 1)), runs::Int=10_000)
    runs > 0 || throw(ArgumentError("runs must be positive"))
    # TODO: #4 add phases @vmmhep
    if @isdefined h11
    else
        h11::Int = size(Q,1)
    end
    id_basis = vacua_id_basis(L, Q; threshold)
    if haskey(id_basis, "Qeff")
        Qeff = Matrix(id_basis["Qeff"])
        xmin = []
        for (i,row) in enumerate(eachrow(Qeff))
            if sum(iszero.(row)) == (size(row, 1)) - 1
                push!(xmin, zeros(Float64, h11))
            elseif maximum(denominator.(row)) == 1
                push!(xmin, zeros(Float64, h11))
            else
                Leff = id_basis["Leff"][:, @.(!iszero(row))]
                Lsubdiff = @view(Leff[2,:]) .- @view(Leff[2,1])
                Lfull = Leff[1,:] .* 10. .^ Lsubdiff;
                res = subspace_minimize(Lfull, Matrix(row[row .!= 0]'); runs=runs,
                    phase=phase[i])
                if typeof(res) <: Vector
                    res = reshape(res, length(res), 1)
                end
                subspace_min = zeros(h11, size(res, 1))
                subspace_min[i, :] = hcat(@view(res[:, 1])...)
                subspace_min = subspace_min' * id_basis["Qtilde_inv"]
                push!(xmin, Matrix(subspace_min'))
            end
        end
        keys = ["θ̃∥", "vac"]
        xmin = hcat(xmin...)
        xmin = sort(xmin, dims = 2)
        min_num = 1
        while min_num < size(xmin, 2)
            if all(abs.(@view(xmin[:, min_num+1]) .- @view(xmin[:, min_num])) .< 1e-10) 
                xmin[:, min_num] = zero(@view(xmin[:, min_num]))
            end
            min_num += 1
        end
        xmin = unique(xmin, dims = 2)
        vac = size(xmin, 2)
        vals = [xmin, vac]
        return Dict(zip(keys, vals))
    else
        θ̃min = id_basis["θ̃∥"]
        for col in axes(θ̃min, 2)
            if sum(θ̃min[:, col] .== zero(θ̃min[:, col][1])) == size(θ̃min, 1) - 1
                θ̃min[:, col] = zero(θ̃min[:, col])
            else
                for i in 1:maximum(denominator.(θ̃min[:, col]))
                    θ̃min = hcat(θ̃min, i .+ θ̃min[:, col])
                end
            end
        end
        xmin = unique(θ̃min, dims=2)
        xmin = unique(@.(ifelse(all(xmin != 0), mod(xmin, 1), xmin)), dims=2)
        keys = ["θ̃min", "θ̃∥", "vac"]
        vals = [θ̃min, xmin, id_basis["vac"]]
        Dict(zip(keys, vals))
    end
end

"""
    vacua_id(h11::Int, tri::Int, cy::Int; threshold, phase::Vector)

TBW
"""
function vacua_id(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5,
    phase::Vector=zeros(h11), runs::Int=10_000)
    pot_data = potential(h11,tri,cy)
    Q::Matrix{Int}, L::Matrix{Float64} = pot_data.Q, pot_data.L 
    vacua_id(L, Q; threshold=threshold, phase=phase, runs=runs)
end

"""
    basis_snf(rays::Matrix{Int})

This function is useful for checking if the identity matrix is contained within the charge matrix, _i.e._ that the fundamental domain is the unit cube
"""
function basis_snf(rays::Matrix{Int})
    h11::Int = size(rays,2)
    ###### Nemo SNF #####
    Qtemp::Nemo.ZZMatrix = matrix(Nemo.ZZ,rays)
    T::Nemo.ZZMatrix = snf_with_transform(Qtemp)[2]
    Tparallel1::Nemo.ZZMatrix = inv(T)[:, 1:h11]
    Tparallel::Matrix{Rational} = zeros(1,1)
    if maximum(abs.(Tparallel1)) < 2^60
        Tparallel = Matrix{Int}(Tparallel1)
        θparalleltest = Matrix{Rational}(inv(transpose(Rational.(rays)) * Rational.(rays)) * transpose(Rational.(rays)) * Tparallel)
        θparalleltest = @.(ifelse(abs(θparalleltest) < 1e-4, zero(θparalleltest), Rational(θparalleltest)))
		θparalleltestinv = @.(ifelse(abs(inv(θparalleltest)) < 1e-4, zero(θparalleltest), Rational(θparalleltest)))
    else
        Tparallel = Matrix{BigInt}(Tparallel1)
        θparalleltest = Matrix{Rational{BigInt}}(inv(transpose(Rational.(rays)) * Rational.(rays)) * transpose(Rational.(rays)) * Tparallel)
        θparalleltest = @.(ifelse(abs(θparalleltest) < 1e-4, zero(θparalleltest), Rational{BigInt}(θparalleltest)))
		θparalleltestinv = @.(ifelse(abs(inv(θparalleltest)) < 1e-4, zero(θparalleltest), Rational{BigInt}(θparalleltest)))
    end
	vol_basis = abs(det(θparalleltest))
    return BasisSNF(vol_basis, θparalleltest, θparalleltestinv)
end

function vacua_SNF(Q::AbstractMatrix{<:Integer})
    h11::Int = size(Q,2)
    ###### Nemo SNF #####
    Qtemp::Nemo.ZZMatrix = matrix(Nemo.ZZ,Q)
    T::Nemo.ZZMatrix = snf_with_transform(Qtemp)[2]
    Tparallel1::Nemo.ZZMatrix = inv(T)[:, 1:h11]
    Tparallel::Matrix{Rational} = zeros(1,1)
    if maximum(abs.(Tparallel1)) < 2^60
        Tparallel = Matrix{Int}(Tparallel1)
        θparalleltest = Matrix{Rational}(inv(transpose(Rational.(Q)) * Rational.(Q)) * transpose(Rational.(Q)) * Tparallel)
        θparalleltest = @.(ifelse(abs(θparalleltest) < 1e-4, zero(θparalleltest), Rational(θparalleltest)))
    else
        Tparallel = Matrix{BigInt}(Tparallel1)
        θparalleltest = Matrix{Rational{BigInt}}(inv(transpose(Rational.(Q)) * Rational.(Q)) * transpose(Rational.(Q)) * Tparallel)
        θparalleltest = @.(ifelse(abs(θparalleltest) < 1e-4, zero(θparalleltest), Rational{BigInt}(θparalleltest)))
    end
    # keys = ["T∥", "θ∥"]
    # vals = [Tparallel,θparalleltest]
    return RationalQSNF(Tparallel,θparalleltest)
end
"""
    vacua_TB(L,Q)

Compute the number of vacua given an instanton charge matrix `Q` and 2-column matrix of instanton scales `L` (in the form [sign; exponent])

For small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.

For larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential.
#Examples
```julia-repl
julia> using CYAxiverse
julia> h11,tri,cy = 10,20,1;
julia> pot_data = CYAxiverse.read.potential(h11,tri,cy);
julia> vacua_data = CYAxiverse.generate.vacua_TB(pot_data["L"],pot_data["Q"])
Dict{String, Any} with 3 entries:
  "θ∥"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]
  "vacua"  => 11552.0
  "Qtilde" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]
```
"""
function vacua_TB(L::Matrix{Float64},Q::Matrix{Int}; threshold::Float64=0.5)
    
    h11::Int = size(Q,2)
    θparalleltest::Matrix{Rational} = zeros(Rational, 0, 0)
    if h11 <= 50
        snf_data = vacua_SNF(Q)
        θparalleltest = snf_data.θparallel
    end
    data = LQtildebar(L,Q; threshold=threshold)
    Qtilde = data["Qtilde"]
    Qbar = data["Qbar"]
    Ltilde = data["Ltilde"]
    Lbar = data["Lbar"]
    α = data["α"]
    if h11 <= 50
        if size(Qtilde,1) == size(Qtilde,2)
            vacua = abs(det(θparalleltest) / det(inv(Qtilde)))
        else
            vacua = abs(det(θparalleltest) / (1/sqrt(det(Qtilde * Qtilde'))))
        end
        thparallel::Matrix{Rational} = Rational.(round.(θparalleltest; digits=5))
        keys = ["vacua","θ∥","Qtilde"]
        vals = [abs(vacua), thparallel, Qtilde]
        return Dict(zip(keys,vals))
    else
        if size(Qtilde,1) == size(Qtilde,2)
            vacua = abs(1 / det(inv(Qtilde)))
        else
            vacua = abs(sqrt(det(Qtilde * Qtilde')))
        end
        
        keys = ["vacua","Qtilde"]
        vals = [abs(vacua), Qtilde]
        return Dict(zip(keys,vals))
    end
end

"""
    vacua_TB(h11,tri,cy)

Compute the number of vacua given a geometry from the KS database.

For small systems (Nax<=50) the algorithm computes the ratio of volumes of the fundamental domain of the leading potential and the full potential.

For larger systems, the algorithm only computes the volume of the fundamental domain of the leading potential.
#Examples
```julia-repl
julia> using CYAxiverse
julia> h11,tri,cy = 10,20,1;
julia> vacua_data = CYAxiverse.generate.vacua_TB(h11,tri,cy)
Dict{String, Any} with 3 entries:
  "θ∥"     => Rational[1//1 0//1 … 0//1 0//1; 0//1 1//1 … 0//1 0//1; … ; 0//1 0//1 … 1//1 0//1; 0//1 0//1 … 0//1 1//1]
  "vacua"  => 11552.0
  "Qtilde" => [0 0 … 0 1; 0 0 … 0 0; … ; 1 1 … -1 -1; 0 0 … 0 0]
```
"""
function vacua_TB(h11::Int,tri::Int,cy::Int; threshold::Float64=0.5)
    pot_data = potential(h11,tri,cy)
    Q::Matrix{Int}, L::Matrix{Float64} = pot_data.Q, pot_data.L 
    vacua_TB(L, Q; threshold=threshold)
end


function vacua_save(h11::Int,tri::Int,cy::Int=1; threshold::Float64=0.5)
    file_open::Bool = 0
    h5open(cyax_file(h11,tri,cy), "r") do file
        if haskey(file, "vacua")
            file_open = 1
            return nothing
        end
    end
    if file_open == 0
        pot_data = potential(h11,tri,cy)
        vacua_data = vacua(pot_data.L,pot_data.Q; threshold=threshold)
        h5open(cyax_file(h11,tri,cy), "r+") do file
            f3 = create_group(file, "vacua")
            f3["vacua",deflate=9] = vacua_data["vacua"]
            f3["Qtilde",deflate=9] = vacua_data["Qtilde"]
            if h11 <=50
                f3a = create_group(f3, "thparallel")
                f3a["numerator",deflate=9] = numerator.(vacua_data["θ∥"])
                f3a["denominator",deflate=9] = denominator.(vacua_data["θ∥"])
            end
        end
    end
end



function vacua_save_TB(h11::Int,tri::Int,cy::Int=1; threshold::Float64=0.5)
    file_open::Bool = 0
    h5open(cyax_file(h11,tri,cy), "r") do file
        if haskey(file, "vacua_TB")
            file_open = 1
            return nothing
        end
    end
    if file_open == 0
        pot_data = potential(h11,tri,cy)
        vacua_data = vacua_TB(pot_data.L,pot_data.Q; threshold=threshold)
        h5open(cyax_file(h11,tri,cy), "r+") do file
            f3 = create_group(file, "vacua_TB")
            f3["vacua",deflate=9] = vacua_data["vacua"]
            f3["Qtilde",deflate=9] = vacua_data["Qtilde"]
            if h11 <=50
                f3a = create_group(f3, "thparallel")
                f3a["numerator",deflate=9] = numerator.(vacua_data["θ∥"])
                f3a["denominator",deflate=9] = denominator.(vacua_data["θ∥"])
            end
        end
    end
end


"""
    vacua_MK(L,Q; threshold=1e-2)
Uses the projection method of _PQ Axiverse_ [paper](https://arxiv.org/abs/2112.04503) (Appendix A) on ``\\mathcal{Q}`` to compute the locations of vacua.
!!! note
    Finding the lattice of minima when numerical minimisation is required has not yet been implemented.
"""
function vacua_MK(L::Matrix{Float64}, Q::Matrix{Int}; threshold = 1e-2)
	setprecision(ArbFloat; digits=5_000)
    h11 = size(Q, 2)
    LQtilde = LQtildebar(L, Q; threshold=threshold)
	Ltilde = LQtilde["Ltilde"][:,sortperm(LQtilde["Ltilde"][2,:], rev=true)]
    Qtilde = LQtilde["Qtilde"]'[sortperm(Ltilde[2,:], rev=true), :]
	Qtilde = Matrix{Int}(Qtilde)
    basis_vectors = zeros(size(Qtilde,2), size(Qtilde,2))
	idx = 1
    println("size Qtilde: ", size(Qtilde))
    while idx < size(Q,2)
        println("start ", idx)
		Qsub = Qtilde[idx, :]
		Lsub = Ltilde[:, idx]
		while Ltilde[2, idx+1] - Ltilde[2, idx] ≥ threshold && dot(Qtilde[idx+1, :], Qtilde[idx, :]) != 0
			Lsub = hcat(Lsub, Ltilde[:, idx+1])
			Qsub = hcat(Qsub, Qtilde[idx+1, :])
			idx += 1
            println("while ", idx)
		end
		if size(Qsub,2) == 1
			basis_vectors[idx, :] = Qsub
			idx += 1
            println("if ", idx)

		else
			Lsubdiff = Lsub[2,:] .- Lsub[2,1]
			Lfull = Lsubdiff[1,:] .* 10. .^ Lsubdiff[2,:];
			Qsubmask = [sum(i .== 0) < size(Qsub,1) for i in eachcol(Qsub)]
				Qsub = Qsub[:,Qsubmask]
			res = nothing
				for run_number = 1:10_000
					x0 = rand(Uniform(0,2π),h11) .* rand(Float64,h11)
					raw_result = minimize(Lfull, Qsub, x0) ##need to write subsystem minimizer
					res = _MKMinimizeResult(raw_result["xmin"],
						raw_result["Vmin_log"] .+ Lsub[2,1])
				end
			xmin = hcat(res.xmin...)
			for i in eachcol(xmin)
				i[:] = @.(mod(i / 2π, 1) ≈ 1 || mod(i / 2π, 1) ≈ 0 ? 0 : i)
			end
			xmin = xmin[:, [sum(i)/size(i,1) > eps() for i in eachcol(xmin)]]
			xmin = xmin[:,sortperm([norm(i,Inf) for i in eachcol(xmin)])]
            xmin[xmin .< 10. * eps()] .= 0.
            println(size(xmin))
				lattice_vecs = minima_lattice(xmin) ##need to write lattice minimizer
			basis_vectors[idx-size(lattice_vecs["lattice_vectors"],2):idx, :] = lattice_vecs["lattice_vectors"]
		end
        T = orth_basis(Qtilde[idx, :])
        Qtilde_i = zeros(size(Qtilde, 1), size(T, 2))
        LinearAlgebra.mul!(Qtilde_i, Qtilde, T)
        Qtilde = copy(Qtilde_i)
        println("size(Qtilde): ", size(Qtilde))
    end
    keys = ["minima_lattice_vectors"]
    vals = [basis_vectors]
    return Dict(zip(keys,vals))
end

"""
    vacua_MK(L,Q; threshold=1e-2)
Uses the projection method of _PQ Axiverse_ [paper](https://arxiv.org/abs/2112.04503) (Appendix A) on ``\\mathcal{Q}`` to compute the locations of vacua.
!!! note
    Finding the lattice of minima when numerical minimisation is required has not yet been implemented.
"""
function vacua_MK(h11::Int,tri::Int,cy::Int)
    pot_data = potential(h11,tri,cy)
    K,L,Q = pot_data.K, pot_data.L, pot_data.Q
    vacua_MK(L, Q)
end

function simple_rationals(min, max)
    if max < 1  # J ⊂ (0, 1)
        return 1/(simple_rationals(1 / max, 1 / min))
    elseif 1 < min  # J ⊂ (1, ∞):
        q = ceil(min) - 1  # largest q satisfying q < left
        return q + simple_rationals(abs(min - q), abs(min - q))
    else  #  left <= 1 <= right, so 1 ∈ J
        return 1/1
    end
end

"""
    vacua_projector(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5)

This applies the projection method to square Q̂ to verify procedure
"""
function vacua_projector(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5, phase::Vector = [[0.]])
    # TODO: #5 fix phases
    if @isdefined h11
    else
        h11::Int = size(Q, 2)
    end
    LQtilde = LQtildebar(L, Q; threshold=threshold)
    Qhat = Matrix{Int}(LQtilde["Qhat"])
    Lhat = LQtilde["Lhat"]
    if size(Qhat, 1) == size(Qhat, 2)
        Qhat = Qhat[:, sortperm(Lhat[2,:], rev=true)]
        LQtilde["Qhat"] = copy(Qhat)
        Lhat = Lhat[:, sortperm(Lhat[2,:], rev=true)]
        idx = 1
        θmin_list = []
        Qsub_list = []
        projectedQ_list = []
        projector = zeros(h11, h11)
        grad(Q::Vector, θ::Float64, δ::Float64) = sin(norm(Q) * θ - δ)
        while idx ≤ size(Qhat, 2)
            # TODO: #7 check if projected Qhat is required at each iteration
            println("COLUMN", idx, ":")
            Qhat = (I(h11) - projector) * Qhat
            Qhat = @.(ifelse(abs(Qhat) < 1e-10, 0., Qhat))
            Qsub = Qhat[:, idx]
            println("Qsub: ", Qsub)
            if Lhat[2, idx == size(Qhat,2) ? idx : idx+1] - Lhat[2, idx] ≥ threshold && dot(Qhat[:, idx == size(Qhat,2) ? idx : idx+1], Qhat[:, idx]) != 0
                return "Sorry, there are degeneracies.  Please try another example."
            else
                min_list = []
                θmin(n::Int, phase_values, qsub) =
                    [(2π*n - δ) / norm(qsub) for δ in phase_values]
                # TODO: #10 check gradient / hessian
                # TODO: #9 Lambdas have different signså
                m = 0
                esub = Qsub ./ norm(Qsub)
                limit = ifelse(any(0. .< abs.(Qsub) .< 1.), 2π/minimum(abs.(esub[esub .!= 0.])), 2π)
                println("limit: ", limit)
                while all(i -> i < limit, θmin(m, phase[idx], Qsub))
                    # TODO: #12 Check condition on periodicity
                    push!(min_list, θmin(m, phase[idx], Qsub))
                    m+=1
                    println("θmin: ", θmin(m, phase[idx], Qsub))
                end
                # min_list = hcat(min_list...)
                push!(θmin_list, min_list)
                println(zip(phase, min_list...))
                # grad_list = [grad(Qsub, θ, δ) for (δ,θ) in zip(hcat(δlist...), hcat(min_list...))]
                # println("gradients: ", grad_list)
                # println("size(gradients[gradients .== 0]): ", grad_list[grad_list .== 0.])
            end
            projector = I(h11) - project_out(Qsub)
            # TODO: #14 Check products of projectors are projectors
            push!(projectedQ_list, hcat([norm(col) for col in eachcol(projector * Qhat)]...))
            if idx < size(Q, 2)
                phase = reshape(norm(projector * Qhat[:, idx+1]) .* hcat(min_list...), size(min_list))
                # TODO: #13 Phase is sum of all previous phases
            end
            push!(Qsub_list, Qsub)
            println("projectedQ: ", projectedQ_list[idx])
            # TODO: #11 construct θ_min
            idx +=1
            println("projector: ", projector)
            println("projector[projector .!= 0]: ", projector[projector .!=0])
            println("size(projector): ", size(projector))
        end
        (θmin = θmin_list, vacua_estimate = abs(det(LQtilde["Qhat"])), Qhat = LQtilde["Qhat"])
    end
end

function vacua_projector(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5)
    pot_data = potential(h11, tri, cy)
    L, Q = pot_data.L, pot_data.Q
    vacua_projector(L, Q; threshold=threshold)
end

function vacuaΩ(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5, phase::Vector=[[0.]])
    # TODO: #5 fix phases
    if @isdefined h11
    else
        h11::Int = size(Q, 2)
    end
    LQtilde = LQtildebar(L, Q; threshold=threshold)
    Qhat = Matrix{Int}(LQtilde["Qhat"])
    Lhat = LQtilde["Lhat"]
    if size(Qhat, 1) == size(Qhat, 2)
        Qhat = Qhat[:, sortperm(Lhat[2,:], rev=true)]
        LQtilde["Qhat"] = copy(Qhat)
        Lhat = Lhat[:, sortperm(Lhat[2,:], rev=true)]
        idx = 1
        θmin_list = []
        Qsub_list = []
        projectedQ_list = []
        projector = zeros(h11, h11)
        while idx ≤ size(Qhat, 2)
            # TODO: #7 check if projected Qhat is required at each iteration
            println("COLUMN", idx, ":")
            Qhat = (I(h11) - projector) * Qhat
            Qhat = @.(ifelse(abs(Qhat) < 1e-10, 0., Qhat))
            if Lhat[2, idx == size(Qhat,2) ? idx : idx+1] - Lhat[2, idx] ≥ threshold && dot(Qhat[:, idx == size(Qhat,2) ? idx : idx+1], Qhat[:, idx]) != 0
                return "Sorry, there are degeneracies.  Please try another example."
            else
                Qsub = Qhat[:, idx]
                push!(Qsub_list, [norm(col) for col in eachcol(Qhat)])
                println("Qsub: ", Qsub)
                min_list = []
                δ(θmin::Float64) = sum(norm(projector * Qsub))
                # TODO: introduce δ
                θmin(n::Int) = [(2π*n-δ)/norm(Qsub) for δ in hcat(phase...)]
                # TODO: #10 check gradient / hessian
                # TODO: #9 Lambdas have different signså
                m = 0
                esub = Qsub ./ norm(Qsub)
                limit = ifelse(any(0. .< abs.(Qsub) .< 1.), 2π/minimum(abs.(esub[esub .!= 0.])), 2π)
                println("limit: ", limit)
                while all(i -> i < limit, θmin(m))
                    # TODO: #12 Check condition on periodicity
                    push!(min_list, θmin(m))
                    m+=1
                    println("θmin: ", θmin(m))
                end
                # min_list = hcat(min_list...)
                push!(θmin_list, min_list)
                println(zip(phase, min_list...))
                # grad_list = [grad(Qsub, θ, δ) for (δ,θ) in zip(hcat(phase...), hcat(min_list...))]
                # println("gradients: ", grad_list)
                # println("size(gradients[gradients .== 0]): ", grad_list[grad_list .== 0.])
            end
            projector = I(h11) - project_out(Qsub)
            # TODO: #14 Check products of projectors are projectors
            push!(projectedQ_list, hcat([norm(col) for col in eachcol(projector * Qhat)]...))
            if idx < size(Q, 2)
                phase = reshape(norm(projector * Qhat[:, idx+1]) .* hcat(min_list...), size(min_list))
                # TODO: #13 Phase is sum of all previous phases
            end
            push!(Qsub_list, Qsub)
            println("projectedQ: ", projectedQ_list[idx])
            # TODO: #11 construct θ_min
            idx +=1
            println("phases: ", phase[idx])
            println("size(phases): ", size(phase[idx]))
            println("projector: ", projector)
            println("projector[projector .!= 0]: ", projector[projector .!=0])
            println("size(projector): ", size(projector))
        end
        (θmin = θmin_list, vacua_estimate = abs(det(LQtilde["Qhat"])), Qhat = LQtilde["Qhat"])
    end
end
"""
    omega(Ω::Matrix{Int})

TBW
"""
function omega(Ω::Matrix{Int})
    if @isdefined h11
    else
        h11 = size(Ω, 2)
    end
    Ωperp = Matrix{Rational}(deepcopy(Ω))
    Ωparallel = []
    for (i, col) in enumerate(eachcol(Ω))
        # TODO: #15 Π function
        Ωperp[:, i+1:end] = project_out(Vector(col)).Πperp * Ωperp[:, i+1:end]
        Ωperp = @.(ifelse(abs(Ωperp) < 1e-5, zero(Ωperp), Ωperp))
        if i < h11
            push!(Ωparallel, vcat(zeros(Float64, i), mapslices(norm, project_out(Vector(col)).Π * Ω[:, i+1:end]; dims=1)'))
        end
    end
    #TODO #49: check construction
    Ωparallel = hcat(zeros(h11), Ωparallel...)
    Ωparallel = @.(ifelse(abs(Ωparallel) < 1e-5, zero(Ωparallel), Ωparallel))
    ProjectedQ(sparse(Ωperp), sparse(Ωparallel))
end

function omega(geom_idx::GeometryIndex)
    h11 = geom_idx.h11
    omega(αmatrix(geom_idx).Qhat)
end

"""
    norm2(Ω::Union{AbstractMatrix, SparseArrays.AbstractSparseMatrix}; column = true, average = false, product = true)

TBW
"""
function norm2(Ω::Union{AbstractMatrix, SparseArrays.AbstractSparseMatrix}; column = true, average = false, product = true)
    if @isdefined h11
    else
        h11 = size(Ω, 2)
    end
    norm2Ω = zeros(Float64, h11)
	for i in ifelse(column == true, axes(Ω, 2), axes(Ω, 1))
		norm2Ω[i] = ifelse(column == true, norm(Ω[:, i])^2, norm(Ω[i, :])^2)
	end
	if product == true && average == false
        return prod(norm2Ω; dims = 1)
    elseif product == false && average == true
        return sum(norm2Ω; dims = 1) / length(norm2Ω)
    elseif product == false && average == false
        return norm2Ω
    else
        return throw(ArgumentError("average and product kwargs cannot both be $average"))
    end
end


function norm2(Ω::ProjectedQ; column = true, average = false, product = true)
    Ω = Ω.Ωperp
    if @isdefined h11
    else
        h11 = size(Ω, 2)
    end
    norm2Ω = zeros(Float64, h11)
	for i in ifelse(column == true, axes(Ω, 2), axes(Ω, 1))
		norm2Ω[i] = ifelse(column == true, norm(Ω[:, i])^2, norm(Ω[i, :])^2)
	end
	if product == true && average == false
        return prod(norm2Ω; dims = 1)
    elseif product == false && average == true
        return sum(norm2Ω; dims = 1) / length(norm2Ω)
    elseif product == false && average == false
        return norm2Ω
    else
        return throw(ArgumentError("average and product kwargs cannot both be $average"))
    end
end

function norm2(geom_idx::GeometryIndex; column = true, average = false, product = true)
    h11 = geom_idx.h11
    norm2(omega(geom_idx); column = column, average = average, product = product)
end
"""
    norm2minus1(Ω::Union{AbstractMatrix, SparseArrays.AbstractSparseMatrix}; col = true)

TBW
"""
function norm2minus1(Ω::Union{AbstractMatrix, SparseArrays.AbstractSparseMatrix}; column = true, average = false, product = true)
    if @isdefined h11
    else
        h11 = size(Ω, 2)
    end
    norm2Ω = zeros(Float64, h11)
	for i in ifelse(column == true, axes(Ω, 2), axes(Ω, 1))
		norm2Ω[i] = ifelse(column == true, norm(Ω[:, i])^2 - 1, norm(Ω[i, :])^2 - 1)
	end
    norm2Ω = norm2Ω[norm2Ω .!= 0.]
	if product == true && average == false
        return prod(norm2Ω; dims = 1)
    elseif product == false && average == true
        return sum(norm2Ω; dims = 1) / length(norm2Ω)
    elseif product == false && average == false
        return norm2Ω
    else
        return throw(ArgumentError("average and product kwargs cannot both be $average"))
    end
end

function norm2minus1(Ω::ProjectedQ; column = true, average = false, product = true)
    Ω = Ω.Ωperp
    if @isdefined h11
    else
        h11 = size(Ω, 2)
    end
    norm2Ω = zeros(Float64, h11)
	for i in ifelse(column == true, axes(Ω, 2), axes(Ω, 1))
		norm2Ω[i] = ifelse(column == true, norm(Ω[:, i])^2 - 1, norm(Ω[i, :])^2 - 1)
	end
    norm2Ω = norm2Ω[norm2Ω .!= 0.]
	if product == true && average == false
        return prod(norm2Ω; dims = 1)
    elseif product == false && average == true
        return sum(norm2Ω; dims = 1) / length(norm2Ω)
    elseif product == false && average == false
        return norm2Ω
    else
        return throw(ArgumentError("average and product kwargs cannot both be $average"))
    end
end

"""
    θmin(Ω::ProjectedQ; phase=zeros(size(Ω.Ωperp, 2)), n::Vector=zeros(size(Ω.Ωperp, 2)))

TBW
"""
function θmin(Ω::ProjectedQ; phase=zeros(size(Ω.Ωperp, 2)), n::Vector=zeros(size(Ω.Ωperp, 2)))
    min = zeros(size(Ω.Ωperp, 2))
    for i ∈ axes(Ω.Ωperp, 2)
        n_i = 0
        while 0 ≤ min[i] < 2π
            min[i] = 2π * n_i - phase[i] / norm(Ω.Ωperp[:, i])
            n_i += 1
        end
        ei = hcat([Ω.Ωperp[:, i] / norm(Ω.Ωperp[:, i]) for _ in axes(Ω.Ωperp, 1)]...)
    end
end


"""
    θmin_tree(Ω::ProjectedQ; phase=zeros(size(Ω.Ωperp, 2)))

TBW
"""
function θmin_tree(Ω::ProjectedQ; phase=zeros(size(Ω.Ωperp, 2)))
    tree = MyTree(0)
    for i ∈ axes(Ω.Ωperp, 2)
        min = tree.data - phase[i] / norm(Ω.Ωperp[:, i])
        phase[i+1] = min * Ω.Ωparallel
        tree = MyTree(min, tree)
    end
    ei = hcat([ΩpΩ.Ωperperp[:, i] / norm(Ω.Ωperp[:, i]) for i in axes(Ωperp, 1)]...)
end
"""
    vacuaΠ(L, Q; threshold=0.5, phase=zeros(size(Q,2)))

TBW
"""
function vacuaΠ(L, Q; threshold=0.5, phase=zeros(size(Q,2)))
    if @isdefined h11
    else
        h11::Int = size(Q, 2)
    end
    LQtilde = LQtildebar(L, Q; threshold=threshold)
    if size(LQtilde["Qhat"], 1) == size(LQtilde["Qhat"], 2)
        Qhat = LQtilde["Qhat"][:, sortperm(LQtilde["Lhat"][2,:], rev=true)]
        Lhat = LQtilde["Lhat"][:, sortperm(LQtilde["Lhat"][2,:], rev=true)]
        Ω = Matrix{Int}(Qhat)
        Ω = omega(Ω)
    else
        "Ω is not square"
    end
end

function vacuaΠ(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5, phase=zeros(h11))
    pot_data = potential(h11, tri, cy)
    L, Q = pot_data.L, pot_data.Q
    vacuaΠ(L, Q; threshold=threshold, phase=phase)
end


"""
    vacua_full(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5, phase::Vector{Float64}=zeros(Float64, size(Q,2)))
New implementation of MK's algorithm -- testing!
"""
function vacua_full(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5, phase::Vector{Float64}=zeros(Float64, size(Q,2)), runs = 100_000)
    # TODO: #6 projections of square Qhat
    # TODO: #4 add phases @vmmhep
    if @isdefined h11
    else
        h11::Int = size(Q, 2)
    end
    LQtilde = LQtildebar(L, Q; threshold=threshold)
    Qhat = Matrix{Int}(LQtilde["Qhat"])
    Lhat = LQtilde["Lhat"]
    if size(Qhat, 1) == size(Qhat, 2)
        Qinv = Matrix{Rational}(inv(Qhat))
        Qinv = @.(ifelse(abs(Qinv) < 1e-5, zero(Qinv), simple_rationals(round(Qinv; digits=4) - 1e-4, round(Qinv; digits=4) + 1e-4)))
        for col in axes(Qinv, 2)
            if sum(Qinv[:, col] .== zero(Qinv[:, col][1])) == size(Qinv, 1)-1
                Qinv[:, col] = zero(Qinv[:, col])
            end
        end
        return unique(mod.(Qinv, 1), dims=2), abs(det(Qhat)), phase
    else
        Lhat = Lhat[:, sortperm(Lhat[2,:], rev=true)]
        Qhat = Qhat[:, sortperm(Lhat[2,:], rev=true)]
        θmin = []
        vac = 0
        idx = 1
        while idx < size(Qhat, 2)
            Qsub = Qhat[:, idx]
            Lsub = Lhat[:, idx]
            while Lhat[2, idx+1] - Lhat[2, idx] ≥ threshold && dot(Qhat[:, idx+1], Qhat[:, idx]) != 0
                Lsub = hcat(Lsub, Lhat[:, idx+1])
                Qsub = hcat(Qsub, Qhat[:, idx+1])
                idx += 1
            end
            if size(Qsub, 2) == 1 && sum(Qsub .== 0) == size(Qsub, 1)-1
                push!(θmin, zeros(Float64, h11))
                Qhat = project_out(Qsub) * Qhat
                Qhat = @.(ifelse(abs(Qhat) < 1e-5, zero(Qhat), Qhat))
            else
                # Lsub = Lsub[:, @.(!iszero(Qsub))]
                Lsubdiff = @view(Lsub[2,:]) .- Lsub[2,1]
                Lfull = Lsub[1,:] .* 10. .^ Lsubdiff;
                if size(Qsub, 2) == 1
                    Qsub = reshape(Qsub, h11,1)
                end
                println("size(phase): ", size(phase))
                println("phases: ", phase)
                println("size(phase) without zeros: ", size(phase[phase .!= 0]))
                xmin = subspace_minimize(Lfull, Qsub; runs = runs, phase=phase)
                xmin = hcat(xmin...)
                println("number of minima found with $runs random initialisations: ", size(xmin))
                xmin = sort(xmin, dims = 2)
                min_num = 1
                while min_num < size(xmin, 2)
                    if all(abs.(@view(xmin[:, min_num+1]) .- @view(xmin[:, min_num])) .< 1e-10) 
                        xmin[:, min_num] = zero(@view(xmin[:, min_num]))
                    end
                    min_num += 1
                end
                xmin = unique(xmin, dims = 2)
                vac += size(xmin, 2)
                push!(θmin, xmin)
                Qsub = orth_basis(Qsub)
                Qhat = project_out(Qsub) * Qhat
                Qhat = @.(ifelse(abs(Qhat) < 1e-10, zero(Qhat), Qhat))
                # phase::Array{Rational} = I(size(phase,1)) .- project_out(Qsub)
                # phase = @.(ifelse(abs(phase) < 1e-10, zero(phase), phase))
            end
            idx += 1
        end
        θmin = unique(hcat(θmin...), dims = 2)
        vac = size(θmin, 2)
        return θmin, vac, phase
    end
end

function vacua_full(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5, phase::Vector{Float64}=zeros(h11))
    pot_data = potential(h11, tri, cy)
    L, Q = pot_data.L, pot_data.Q
    vacua_full(L, Q; threshold=threshold, phase=phase)
end


"""
    vacua_no_optim(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5, phase::Vector{Float64}=zeros(Float64, size(Q,2)))

TBW
"""
function vacua_no_optim(L::Matrix{Float64}, Q::Matrix{Int}; threshold::Float64=0.5, phase::Vector{Float64}=zeros(Float64, size(Q,2)))
    if @isdefined h11
    else
        h11::Int = size(Q, 2)
    end
    LQtilde = LQtildebar(L, Q; threshold=threshold)
    Qhat = Matrix{Int}(LQtilde["Qhat"])
    Lhat = LQtilde["Lhat"]
    if size(Qhat, 1) == size(Qhat, 2)
        Qinv = Matrix{Rational}(inv(Qhat))
        Qinv = @.(ifelse(abs(Qinv) < 1e-10, zero(Qinv), Rational(round(Qinv; digits=4))))
        for col in axes(Qinv, 2)
            if sum(Qinv[:, col] .== zero(Qinv[:, col][1])) == size(Qinv, 1)-1
                Qinv[:, col] = zero(Qinv[:, col])
            end
        end
        return unique(mod.(Qinv, 1), dims=2), abs(det(Qhat)), phase
    else
        Lhat = Lhat[:, sortperm(Lhat[2,:], rev=true)]
        Qhat = Qhat[:, sortperm(Lhat[2,:], rev=true)]
        Ω = Matrix{Int}(@view(Qhat[:, 1:h11]))
        Ωinv = Matrix{Rational}(inv(Ω))
        Ωinv = @.(ifelse(abs(Ωinv) < 1e-10, zero(Ωinv), Rational(round(Ωinv; digits=4))))
        Ωhat = (Ωinv * Qhat)'
        for col in eachcol(Ωhat)
        end
    end
end


"""
    phase(h11, α::Canonicalα)

TBW
"""
function phase(h11, α::Canonicalα)
    phase_vector = []
	for (i, item) in enumerate(α.Lhat[1, 1:h11])
		if α.:αrowmask[i] == false && item == -1
			push!(phase_vector, π)
		else
			push!(phase_vector, 0)
		end
	end
	phase_vector::Vector = vec([phase_vector' * α.:α_complete]...)
end



function jlm_vacua_db(; n=size(paths_cy()[2], 2), h11 = nothing)
	vac_square = []
	vac_1D = []
	vac_ND = []
    no_vac = []
    geom_list = []
    if h11 === nothing
        geom_list = [GeometryIndex(col...) for col in eachcol(paths_cy()[2][:, 1:n])]
    elseif h11 !== nothing && n != size(paths_cy()[2], 2)
        geom_list = [GeometryIndex(col...) for col in eachcol(paths_cy()[2][:, paths_cy()[2][1, :] .== h11][:, 1:n])]
    else
        geom_list = [GeometryIndex(col...) for col in eachcol(paths_cy()[2][:, paths_cy()[2][1, :] .== h11])]
    end
	for geom_idx in geom_list
		# println(geom_idx)
		if isfile(minfile(geom_idx))
            try
                vac_test = vacua_jlm(geom_idx)
                if typeof(vac_test) <: Min_JLM_Square
                    push!(vac_square, [geom_idx.h11, geom_idx.polytope, geom_idx.frst, vac_test.N_min, vac_test.det_QTilde])
                elseif typeof(vac_test) == Min_JLM_1D
                    push!(vac_1D, [geom_idx.h11, geom_idx.polytope, geom_idx.frst, vac_test.N_min, vac_test.min_coords, vac_test.extra_rows, vac_test.det_QTilde])
                elseif typeof(vac_test) == Min_JLM_ND
                    push!(vac_ND, [geom_idx.h11, geom_idx.polytope, geom_idx.frst, vac_test.N_min, vac_test.min_coords, vac_test.extra_rows, vac_test.det_QTilde])
                end
            catch e
                push!(no_vac, [geom_idx.h11, geom_idx.polytope, geom_idx.frst, 0])
            end
        else
            push!(no_vac, [geom_idx.h11, geom_idx.polytope, geom_idx.frst, 0])
		end
        # Qtilde = LQtilde(geom_idx).Qtilde
        # det_Q_tilde = Int(abs(round(det(Qtilde))))
        # push!(detQtilde, [geom_idx.h11, geom_idx.polytope, geom_idx.frst, det_Q_tilde])
	end
	return (; square = vac_square, one_dim = vac_1D, n_dim = vac_ND, err = no_vac)
end

"""
    vacua_estimate(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5)

Uses `LQtildebar` function to make Q̂.  If Q̂ is square, returns number of vacua as `|det(Q̂)|`
otherwise returns number of vacua as `√|det(Q̂'Q̂)|`.
"""
function vacua_estimate(geom_idx::GeometryIndex; threshold::Float64=0.5)
    data = αmatrix(geom_idx; threshold=threshold)
    if size(data.Qhat, 1) == size(data.Qhat, 2)
        vac = Int(round(abs(det(data.Qhat))))
        return (; vac = vac, issquare = 1)
    else
        vac = Int(floor(sqrt(abs(det(data.Qhat * data.Qhat')))))
        return (; vac = vac, issquare = 0, extrarows = size(data.Qhat, 2) - geom_idx.h11)
    end
end

function vacua_estimate(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5)
    geom_idx = GeometryIndex(h11, tri, cy)
    vacua_estimate(geom_idx; threshold)
end

function vacua_estimate_save(geom_idx::GeometryIndex; threshold::Float64=0.5)
    vac_data = vacua_estimate(geom_idx; threshold=threshold)
    if isfile(minfile(geom_idx))
        h5open(joinpath(minfile(geom_idx)), "r+") do f
            f["issquare", deflate=9] = vac_data.issquare
            f["det_QTilde", deflate=9] = vac_data.vac
        end
    else
        h5open(joinpath(minfile(geom_idx)), "cw") do f
            f["issquare", deflate=9] = vac_data.issquare
            f["det_QTilde", deflate=9] = vac_data.vac
        end
    end
end

function vacua_estimate_save(h11::Int, tri::Int, cy::Int; threshold::Float64=0.5)
    geom_idx = GeometryIndex(h11, tri, cy)
    vacua_estimate_save(geom_idx; threshold)
end

end
