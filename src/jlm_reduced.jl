"""
    CYAxiverse.jlm_reduced

Julia-native preparation and solving utilities for the reduced JLM minima
problem. This module keeps the legacy `Min_JLM_*` result shape, but separates
the expensive data reduction from the numerical solve so large scans can reuse
selected charge data.
"""
module jlm_reduced

using HDF5
using LinearAlgebra
using Nemo
using SparseArrays

using ..filestructure: minfile
using ..generate: LQtilde, αmatrix, phase, vacua_SNF
using ..minimizer: critical_points
using ..read: potential
using ..structs: GeometryIndex, Canonicalα, Min_JLM_1D, Min_JLM_ND, Min_JLM_Square

export ReducedJLMProblem, prepare, minimize, minimize_save

"""Preprocessed charge, scale, and symmetry data for a reduced JLM solve."""
struct ReducedJLMProblem
    Q_reduced::SparseMatrixCSC{Float64, Int}
    L_reduced::Matrix{Float64}
    phases::Vector{Float64}
    det_QTilde::Int
    multiplicity::Float64
    integer_charges::Bool
    square_vacua::Union{Nothing, Int}
    extra_rows::Int
end

"""Return the exact absolute determinant of an integer-valued square matrix."""
function _det_int(Q::AbstractMatrix{<:Integer})
    size(Q, 1) == size(Q, 2) || throw(DimensionMismatch("determinant requires a square matrix"))
    value = Nemo.det(Nemo.matrix(Nemo.ZZ, Matrix{Int}(Q)))
    Int(abs(BigInt(value)))
end

"""Determine whether reduced charges are integral and compute multiplicity."""
function _symmetry_multiplicity(det_QTilde::Int, Q_reduced::AbstractMatrix)
    if maximum(denominator.(Matrix(Q_reduced))) == 1
        return true, Float64(det_QTilde)
    end

    αrescaled = Matrix{Integer}(det_QTilde .* Matrix(Q_reduced))
    θparallel = vacua_SNF(αrescaled).θparallel .* Rational(det_QTilde)
    volume = Float64(abs(det(θparallel)))
    return false, abs(det_QTilde / volume)
end

"""Return a mask selecting rows containing at least one nonzero charge."""
function _nonzero_row_mask(Q::AbstractMatrix)
    [any(!iszero, row) for row in eachrow(Q)]
end

"""Normalize potential matrices to axion-by-instanton orientation."""
function _oriented_potential_matrices(pot_data)
    Q = Matrix{Int}(pot_data.Q)
    L = Matrix{Float64}(pot_data.L)
    if size(L, 1) != 2 && size(L, 2) == 2
        L = Matrix(L')
    end
    if size(Q, 2) != size(L, 2) && size(Q, 1) == size(L, 2)
        Q = Matrix(Q')
    end
    size(L, 1) == 2 || throw(DimensionMismatch("L must have two rows after orientation"))
    size(Q, 2) == size(L, 2) || throw(DimensionMismatch("Q columns must match L columns after orientation"))
    size(Q, 1) < size(Q, 2) || throw(ArgumentError("reduced JLM preparation requires more instantons than axions"))
    Q, L
end

"""
    prepare(Q, L; threshold=0.01)

Build the reduced JLM problem once from an axion-by-instanton charge matrix
`Q` and the matching 2-by-instanton scale matrix `L`. The returned problem
stores the reduced charge matrix sparsely and can be passed repeatedly to
`minimize`.
"""
function prepare(Q::AbstractMatrix{Int}, L::AbstractMatrix{Float64}; threshold::Float64=0.01)
    selected = LQtilde(Q, L)
    αtest = αmatrix(selected; threshold=threshold)

    if !(αtest isa Canonicalα)
        vacua = _det_int(αtest.Qhat)
        return ReducedJLMProblem(spzeros(Float64, 0, 0), zeros(2, 0), Float64[],
            vacua, Float64(vacua), true, vacua, 0)
    end

    det_QTilde = _det_int(αtest.Qhat)
    h11 = size(αtest.Qhat, 1)
    rowmask = αtest.αrowmask
    colmask = αtest.αcolmask
    n_axions = count(rowmask)

    # Decide whether the effective charge set is square without constructing a
    # second dense rational matrix. Only perturbations involving at least two
    # leading axions add independent rows to the reduced problem.
    nontrivial_extra_rows = 0
    @inbounds for col in axes(αtest.α, 2)
        αtest.αcolmask[col] || continue
        nonzero_count = 0
        for row in axes(αtest.α, 1)
            αtest.αrowmask[row] && !iszero(αtest.α[row, col]) && (nonzero_count += 1)
        end
        nonzero_count > 1 && (nontrivial_extra_rows += 1)
    end

    if nontrivial_extra_rows == 0
        return ReducedJLMProblem(spzeros(Float64, 0, 0), zeros(2, 0), Float64[],
            det_QTilde, Float64(det_QTilde), true, det_QTilde, 0)
    end

    Q_reduced = hcat(Matrix{Rational}(I, n_axions, n_axions),
        αtest.α_complete[rowmask, colmask])'
    L_reduced = Matrix(hcat(αtest.Lhat[:, 1:h11][:, rowmask],
        αtest.Lbar[:, colmask])')
    phases = Float64.(phase(h11, αtest)[colmask])
    integer_charges, multiplicity = _symmetry_multiplicity(det_QTilde, Q_reduced)

    # Keep the exact rational representation through classification and only
    # convert values at the numerical solver boundary. Constructing the sparse
    # matrix directly avoids a dense Float64 copy of the full reduced charge
    # matrix.
    Q_reduced_sparse = SparseMatrixCSC{Float64, Int}(sparse(Q_reduced))
    ReducedJLMProblem(Q_reduced_sparse, L_reduced, phases,
        det_QTilde, multiplicity, integer_charges, nothing,
        size(Q_reduced, 1) - size(Q_reduced, 2))
end

"""Load a geometry's potential data and prepare its reduced JLM problem."""
function prepare(geom_idx::GeometryIndex; threshold::Float64=0.01, hilbert::Bool=false)
    pot_data = potential(geom_idx; hilbert=hilbert)
    Q, L = _oriented_potential_matrices(pot_data)
    prepare(Q, L; threshold=threshold)
end

"""Solve a prepared problem and construct the corresponding legacy result."""
function _minimize_reduced(problem::ReducedJLMProblem; starts::Int=100_000,
        residual_tolerance::Float64=1e-9, merge_tolerance::Float64=1e-6,
        max_iterations::Int=300)
    problem.square_vacua === nothing || return Min_JLM_Square(problem.square_vacua, problem.det_QTilde)

    n_axions = size(problem.Q_reduced, 2)
    phases = vcat(zeros(n_axions), problem.phases)
    leading_signs = @view problem.L_reduced[1:n_axions, 1]
    leading_seed = [sign < 0 ? 0.5 : 0.0 for sign in leading_signs]
    seed_points = [leading_seed]
    for axion in 1:n_axions
        for displacement in (-0.125, 0.125)
            seed = copy(leading_seed)
            seed[axion] = mod(seed[axion] + displacement, 1.0)
            push!(seed_points, seed)
        end
    end
    initial_points = hcat(seed_points...)
    solved = critical_points(Matrix(problem.L_reduced'), Matrix(transpose(problem.Q_reduced));
        phases=phases, starts=starts, residual_tolerance=residual_tolerance,
        merge_tolerance=merge_tolerance, max_iterations=max_iterations,
        initial_points=initial_points,
    )

    minima_count = Int(abs(round(problem.multiplicity * solved.minima_count)))
    coords = Matrix((2π .* solved.minima)')
    if n_axions == 1
        return Min_JLM_1D(minima_count, vec(coords), problem.extra_rows, problem.det_QTilde)
    end
    Min_JLM_ND(minima_count, coords, problem.extra_rows, problem.det_QTilde)
end

"""
    minimize(problem; starts=100_000, ...)

Solve a prepared reduced JLM problem using the Julia-native deterministic
critical-point search and return a legacy `Min_JLM_*` result.
"""
minimize(problem::ReducedJLMProblem; kwargs...) = _minimize_reduced(problem; kwargs...)

"""Prepare and solve a reduced JLM problem from charge and scale matrices."""
function minimize(Q::AbstractMatrix{Int}, L::AbstractMatrix{Float64};
        threshold::Float64=0.01, kwargs...)
    minimize(prepare(Q, L; threshold=threshold); kwargs...)
end

"""Prepare and solve a reduced JLM problem for one indexed geometry."""
function minimize(geom_idx::GeometryIndex; threshold::Float64=0.01,
        hilbert::Bool=false, kwargs...)
    minimize(prepare(geom_idx; threshold=threshold, hilbert=hilbert); kwargs...)
end

"""Replace the reduced-minima datasets in an open HDF5 group."""
function _write_result!(group::Union{HDF5.File, HDF5.Group},
        min_data::Union{Min_JLM_Square, Min_JLM_1D, Min_JLM_ND})
    for key in ("Nvac", "vac_coords", "extra_rows", "det_QTilde", "issquare")
        haskey(group, key) && HDF5.delete_object(group, key)
    end
    group["Nvac", deflate=9] = min_data.N_min
    group["det_QTilde", deflate=9] = min_data.det_QTilde
    if min_data isa Min_JLM_Square
        group["issquare", deflate=9] = 1
    else
        group["vac_coords", deflate=9] = min_data.min_coords
        group["extra_rows", deflate=9] = min_data.extra_rows
        group["issquare", deflate=9] = 0
    end
end

"""
    minimize_save(geom_idx; threshold=0.01, hilbert=false, kwargs...)

Solve a geometry's reduced JLM problem and persist the result to its minima
file, using the `hilbert` subgroup when requested.
"""
function minimize_save(geom_idx::GeometryIndex; threshold::Float64=0.01,
        hilbert::Bool=false, kwargs...)
    min_data = minimize(geom_idx; threshold=threshold, hilbert=hilbert, kwargs...)
    h5open(minfile(geom_idx), isfile(minfile(geom_idx)) ? "r+" : "cw") do file
        if hilbert
            haskey(file, "hilbert") || create_group(file, "hilbert")
            _write_result!(file["hilbert"]::HDF5.Group, min_data)
        else
            _write_result!(file, min_data)
        end
    end
    min_data
end

end
