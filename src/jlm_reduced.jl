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
using ..generate: LQtilde, LQLinearlyIndependent, αmatrix, phase, vacua_SNF
using ..minimizer: critical_points
using ..read: potential
using ..structs: GeometryIndex, Canonicalα, Min_JLM_1D, Min_JLM_ND, Min_JLM_Square

export ReducedJLMProblem, prepare, minimize, minimize_save, critical_ensemble

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
    reduction::Symbol
    coordinate_scale::Vector{Int}
    lift_matrix::Matrix{Float64}
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

"""Round a floating-point alpha entry in the same tolerance as the author code."""
function _author_alpha(Qhat::AbstractMatrix{Int}, Qbar::AbstractMatrix{Int})
    inverse = inv(Matrix{Rational}(Qhat))
    alpha = Matrix{Rational}(transpose(inverse * Matrix{Rational}(Qbar)))
    for index in eachindex(alpha)
        value = Float64(alpha[index])
        abs(value) < 1e-3 && (alpha[index] = zero(Rational))
        abs(value - round(value)) < 1e-3 &&
            (alpha[index] = Rational(round(Int, value)))
    end
    alpha
end

"""Return the rightmost nonzero alpha coordinate, or zero for a zero row."""
function _author_smallest_coordinate(row::AbstractVector{<:Number})
    for index in reverse(eachindex(row))
        !iszero(row[index]) && return index
    end
    0
end

"""Integerize a rational charge matrix and return its coordinate periods."""
function _integerize_author_charges(Q::AbstractMatrix{Rational})
    scales = Int[]
    for column in eachcol(Q)
        scale = foldl(lcm, denominator.(column); init=1)
        push!(scales, scale)
    end
    integer = Matrix{Int}(undef, size(Q))
    for column in axes(Q, 2), row in axes(Q, 1)
        integer[row, column] = Int(Q[row, column] * scales[column])
    end
    integer, scales
end

"""Return a reduced problem using the author paper's row-level alpha filter."""
function _prepare_author(selected::LQLinearlyIndependent; threshold::Float64)
    threshold > 0 || throw(ArgumentError("threshold must be positive"))
    Qhat = Matrix{Int}(selected.Qtilde)
    Qbar = Matrix{Int}(selected.Qbar)
    Lhat = Matrix{Float64}(selected.Ltilde)
    Lbar = Matrix{Float64}(selected.Lbar)
    h11 = size(Qhat, 1)
    det_QTilde = _det_int(Qhat)
    alpha_total = _author_alpha(Qhat, Qbar)

    kept = Int[]
    smallest = [_author_smallest_coordinate(row) for row in eachrow(alpha_total)]
    for row in axes(alpha_total, 1)
        coordinate = smallest[row]
        coordinate == 0 && continue
        Lbar[2, row] - Lhat[2, coordinate] > log10(threshold) && push!(kept, row)
    end
    isempty(kept) && return ReducedJLMProblem(spzeros(Float64, 0, 0), zeros(2, 0),
        Float64[], det_QTilde, Float64(det_QTilde), true, det_QTilde, 0,
        :author, Int[], zeros(Float64, h11, 0))

    alpha = alpha_total[kept, :]
    Lextra = Lbar[:, kept]
    signs = vcat(Lhat[1, :], Lextra[1, :])
    logs = vcat(Lhat[2, :], Lextra[2, :])
    matrix_coord = vcat(Matrix{Float64}(I, h11, h11), Float64.(alpha))
    col_min = findall(count(!iszero, column) > 1 for column in eachcol(matrix_coord))
    isempty(col_min) && return ReducedJLMProblem(spzeros(Float64, 0, 0), zeros(2, 0),
        Float64[], det_QTilde, Float64(det_QTilde), true, det_QTilde, 0,
        :author, Int[], zeros(Float64, h11, 0))

    qeff = vcat(Matrix{Float64}(I, length(col_min), length(col_min)),
        Float64.(alpha[:, col_min]))
    first_alpha = copy(@view qeff[length(col_min) + 1, :])
    non_relevant = findall(index ->
        abs(first_alpha[index]) >= 1e-3 &&
        logs[h11 + 1] - Lhat[2, col_min[index]] - log10(threshold) < 0,
        eachindex(first_alpha))
    # This is the author's positional truncation: the first suppressed
    # columns in the active list are removed before later rows are filtered.
    drop_count = length(non_relevant)
    drop_count < length(col_min) || return ReducedJLMProblem(spzeros(Float64, 0, 0),
        zeros(2, 0), Float64[], det_QTilde, Float64(det_QTilde), true,
        det_QTilde, 0, :author, Int[], zeros(Float64, h11, 0))
    alpha_truncated = Float64.(qeff[length(col_min) + 1:end, drop_count + 1:end])
    col_truncated = col_min[drop_count + 1:end]
    Ltilde_truncated = logs[col_truncated]
    Lbar_truncated = logs[h11 + 1:end]

    if size(alpha_truncated, 1) > 1
        for row in 2:size(alpha_truncated, 1)
            prior = vec(sum(abs.(@view alpha_truncated[1:row - 1, :]), dims=1))
            suppression = Lbar_truncated[row] .- Ltilde_truncated .-
                log10(threshold) .> 0
            alpha_truncated[row, :] .*= Float64.((prior .> 1e-3) .| suppression)
        end
    end

    alpha_reduced = copy(alpha_truncated)
    rows_non_relevant = findall(count(!iszero, row) == 1 for row in eachrow(alpha_reduced))
    for row in rows_non_relevant, column in axes(alpha_truncated, 2)
        abs(abs(alpha_truncated[row, column]) - 1) <= 1e-4 &&
            (alpha_truncated[row, column] = 0.0)
    end
    rows_zeros = findall(count(!iszero, row) == 0 for row in eachrow(alpha_truncated))
    keep_rows = setdiff(axes(alpha_truncated, 1), rows_zeros)
    alpha_low = alpha_truncated[keep_rows, :]
    col_reduced = findall(any(!iszero, column) for column in eachcol(alpha_low))
    isempty(col_reduced) && return ReducedJLMProblem(spzeros(Float64, 0, 0), zeros(2, 0),
        Float64[], det_QTilde, Float64(det_QTilde), true, det_QTilde, 0,
        :author, Int[], zeros(Float64, h11, 0))

    Qrational = vcat(Matrix{Rational}(I, length(col_reduced), length(col_reduced)),
        Matrix{Rational}(alpha_low[:, col_reduced]))
    integer_Q, coordinate_scale = _integerize_author_charges(Qrational)

    # Effective coefficients, including the author's cancellation correction
    # for a one-coordinate row that was removed from the optimizer.
    tilde_logs = Float64.(logs[col_truncated])
    tilde_signs = Float64.(signs[col_truncated])
    for row in rows_non_relevant
        coordinate = findall(abs.(abs.(alpha_reduced[row, :]) .- 1) .< 1e-4)
        length(coordinate) == 1 || continue
        coordinate = only(coordinate)
        ratio = Lbar_truncated[row] - Ltilde_truncated[coordinate]
        signed_factor = Lextra[1, row] * tilde_signs[coordinate]
        factor = 1 + signed_factor * 10.0^ratio
        if abs(factor) < 1e-300
            tilde_logs[coordinate] = -Inf
            tilde_signs[coordinate] = 1.0
        else
            tilde_logs[coordinate] += log10(abs(factor))
            tilde_signs[coordinate] *= sign(factor)
        end
    end
    tilde_logs = tilde_logs[col_reduced]
    tilde_signs = tilde_signs[col_reduced]
    bar_logs = Lbar_truncated[keep_rows]
    bar_signs = Lextra[1, keep_rows]
    Lreduced = Matrix(hcat(vcat(tilde_signs, bar_signs),
        vcat(tilde_logs, bar_logs)))

    location = [signs[index] < 0 ? π : 0.0 for index in col_min]
    for index in col_reduced
        index <= length(location) && (location[index] = 0.0)
    end
    phases = alpha[:, col_min] * location
    phases = phases[keep_rows]

    # x_full = Qhat' * theta_raw.  The reduced coordinates retain only the
    # selected leading directions, so this matrix lifts them back to raw theta.
    retained = col_truncated[col_reduced]
    embedding = zeros(Float64, h11, length(retained))
    for (index, original) in enumerate(retained)
        embedding[original, index] = 1.0
    end
    lift = inv(Matrix{Float64}(Qhat')) * embedding

    ReducedJLMProblem(sparse(Float64.(integer_Q)), Lreduced, Float64.(phases),
        det_QTilde, 1.0, false, nothing,
        size(integer_Q, 1) - size(integer_Q, 2), :author,
        coordinate_scale, lift)
end

"""
    prepare(Q, L; threshold=0.01, reduction=:alphamatrix)

Build the reduced JLM problem once from an axion-by-instanton charge matrix
`Q` and the matching 2-by-instanton scale matrix `L`. The returned problem
stores the reduced charge matrix sparsely and can be passed repeatedly to
`minimize`.
"""
function prepare(Q::AbstractMatrix{Int}, L::AbstractMatrix{Float64};
        threshold::Float64=0.01, reduction::Symbol=:alphamatrix)
    reduction in (:alphamatrix, :author) ||
        throw(ArgumentError("reduction must be :alphamatrix or :author"))
    selected = LQtilde(Q, L)
    reduction == :author && return _prepare_author(selected; threshold)
    αtest = αmatrix(selected; threshold=threshold)

    if !(αtest isa Canonicalα)
        vacua = _det_int(αtest.Qhat)
        return ReducedJLMProblem(spzeros(Float64, 0, 0), zeros(2, 0), Float64[],
        vacua, Float64(vacua), true, vacua, 0, :alphamatrix, Int[],
        zeros(Float64, size(αtest.Qhat, 1), 0))
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
            det_QTilde, Float64(det_QTilde), true, det_QTilde, 0,
            :alphamatrix, Int[], zeros(Float64, h11, 0))
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
        size(Q_reduced, 1) - size(Q_reduced, 2), :alphamatrix,
        ones(Int, size(Q_reduced, 2)), Matrix{Float64}(I, size(Q_reduced, 2), size(Q_reduced, 2)))
end

"""Load a geometry's potential data and prepare its reduced JLM problem."""
function prepare(geom_idx::GeometryIndex; threshold::Float64=0.01,
        hilbert::Bool=false, reduction::Symbol=:alphamatrix)
    pot_data = potential(geom_idx; hilbert=hilbert)
    Q, L = _oriented_potential_matrices(pot_data)
    prepare(Q, L; threshold=threshold, reduction=reduction)
end

"""Solve a non-square reduced model and return its complete critical ensemble."""
function critical_ensemble(problem::ReducedJLMProblem; starts::Int=100_000,
        residual_tolerance::Float64=1e-9, merge_tolerance::Float64=1e-6,
        max_iterations::Int=300)
    problem.square_vacua === nothing || return (
        coordinates=zeros(Float64, size(problem.Q_reduced, 2), 0),
        reduced_coordinates=zeros(Float64, size(problem.Q_reduced, 2), 0),
        inertia=NTuple{3, Int}[], critical_count=0, minima_count=problem.square_vacua)
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
    solved = critical_points(Matrix(problem.L_reduced'), Matrix(transpose(problem.Q_reduced));
        phases=phases, starts=starts, residual_tolerance=residual_tolerance,
        merge_tolerance=merge_tolerance, max_iterations=max_iterations,
        initial_points=hcat(seed_points...))
    reduced_coordinates = problem.coordinate_scale .* solved.coordinates
    coordinates = problem.reduction == :author ?
        mod.(problem.lift_matrix * reduced_coordinates, 1.0) : reduced_coordinates
    merge(solved, (; coordinates, reduced_coordinates))
end

"""Solve a prepared problem and construct the corresponding legacy result."""
function _minimize_reduced(problem::ReducedJLMProblem; starts::Int=100_000,
        residual_tolerance::Float64=1e-9, merge_tolerance::Float64=1e-6,
        max_iterations::Int=300)
    problem.square_vacua === nothing || return Min_JLM_Square(problem.square_vacua, problem.det_QTilde)

    n_axions = size(problem.Q_reduced, 2)
    solved = critical_ensemble(problem; starts, residual_tolerance,
        merge_tolerance, max_iterations)

    minima_count = Int(abs(round(problem.multiplicity * solved.minima_count)))
    minima_mask = [entry == (0, 0, n_axions) for entry in solved.inertia]
    coords = Matrix((2π .* solved.coordinates[:, minima_mask])')
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
        threshold::Float64=0.01, reduction::Symbol=:alphamatrix, kwargs...)
    minimize(prepare(Q, L; threshold=threshold, reduction=reduction); kwargs...)
end

"""Prepare and solve a reduced JLM problem for one indexed geometry."""
function minimize(geom_idx::GeometryIndex; threshold::Float64=0.01,
        hilbert::Bool=false, reduction::Symbol=:alphamatrix, kwargs...)
    minimize(prepare(geom_idx; threshold=threshold, hilbert=hilbert,
        reduction=reduction); kwargs...)
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
