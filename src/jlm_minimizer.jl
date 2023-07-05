"""
    CYAxiverse.jlm_minimizer
Here we define functions related to JLM's python minimization methods

"""
module jlm_minimizer
using HDF5
using LinearAlgebra
using GenericLinearAlgebra

using ..jlm_python: one_dim_axion_solver, multi_axion_solver
using ..generate: αmatrix, LQtilde, phase, vacua_SNF

using ..structs: GeometryIndex, Canonicalα, Solver1D, SolverND

"""
    jlm_minimize(geom_idx::GeometryIndex)

If the effective instanton charge matrix, `Q`, is not square, this function will compute the number of vacua in the potential using the methods outlined in `arXiv: 2306.XXXXX`.
"""
function minimize(geom_idx::GeometryIndex; random_phase=false)
    αtest = αmatrix(geom_idx; threshold=0.01)
    if typeof(αtest)<:Canonicalα
        Qtilde = LQtilde(geom_idx).Qtilde
        det_Q_tilde = Int(abs(round(det(Qtilde))))
        n_axions = size(αtest.α[αtest.αrowmask, αtest.αcolmask], 1)
        Q_reduced = hcat(1//1 * I(n_axions), αtest.α_complete[αtest.αrowmask, αtest.αcolmask])'
        Q_reduced_temp = hcat(1//1 * I(n_axions), αtest.α[αtest.αrowmask, αtest.αcolmask])'
        for (i,row) in enumerate(eachrow(Q_reduced[n_axions+1:end, :]))
            if sum(abs.(row) .== 0) == (size(row, 1) - 1)
                Q_reduced_temp[n_axions+i, :] .= 0
            end
        end
        Qrowmask = [any(row .!= 0) for row in eachrow(Q_reduced_temp)]
        Q_reduced_temp = Q_reduced_temp[Qrowmask, :]
        if size(Q_reduced_temp, 1) == size(Q_reduced_temp, 2)
            return det_Q_tilde
        else
            phase_vector = phase(geom_idx.h11, αtest)
            if random_phase
                phase_vector = mod.(phase_vector .+ rand(Uniform(0, 2π), size(phase_vector, 1)), 2π)
            end
            L_reduced = Matrix(hcat(αtest.Lhat[:, 1:geom_idx.h11][:, αtest.αrowmask], αtest.Lhat[:, geom_idx.h11+1:end][:, αtest.αcolmask])')
            # L_reduced = L_reduced[Qrowmask, :]
            flag_int = ifelse(maximum(denominator.(Matrix(Q_reduced))) == 1, 1, 0)
            αrescaled = Matrix{Integer}(det_Q_tilde .* Matrix(Q_reduced))
            θparallel = vacua_SNF(αrescaled).:θparallel .* Rational(det_Q_tilde)
            basis_inverse = []
            if abs(maximum(denominator.(θparallel)) * maximum(numerator.(abs.(θparallel)))) > 2^60
                θparallel::Matrix{Rational{BigInt}} = θparallel
                basis_inverse = ifelse(size(inv(θparallel)) == (1,1), Rational{BigInt}(inv(θparallel)[1,1]), Matrix{Rational{BigInt}}(inv(θparallel)))
            else
                basis_inverse = ifelse(size(inv(θparallel)) == (1,1), Rational(inv(θparallel)[1,1]), Matrix{Rational}(inv(θparallel)))
            end
            vol_basis = Rational(det(θparallel))
            if size(Q_reduced, 2) == 1
                to_solve1D = Solver1D(10π, Float64.(vec(Q_reduced)), L_reduced[:, 2], L_reduced[:, 1], det_Q_tilde, phase_vector, flag_int, basis_inverse, vol_basis)
                return one_dim_axion_solver(to_solve1D)
                # return to_solve1D
            else
                to_solveND = SolverND(100_000, Float64.(Matrix(Q_reduced)), L_reduced[:, 2], L_reduced[:, 1], det_Q_tilde, phase_vector, flag_int, basis_inverse, vol_basis)
                return multi_axion_solver(to_solveND)
                # return to_solveND
            end
        end
    else
        return Int(abs(round(det(αtest.Qhat))))
    end
end

function minimize_save(geom_idx::GeometryIndex; random_phase=false)
    min_data = jlm_minimize(geom_idx; random_phase=random_phase)
    if isfile(minfile(geom_idx))
        rm(minfile(geom_idx))
    end
    if typeof(min_data) <: Int
        h5open(minfile(geom_idx),isfile(minfile(geom_idx)) ? "r+" : "cw") do file
            file["Nvac", deflate=9] = min_data
        end
    elseif typeof(min_data) <: Min_JLM_1D || typeof(min_data) <: Min_JLM_ND
        h5open(minfile(geom_idx),isfile(minfile(geom_idx)) ? "r+" : "cw") do file
            file["Nvac", deflate = 9] = min_data.N_min
            file["vac_coords", deflate = 9] = min_data.min_coords
            file["extra_rows", deflate = 9] = min_data.extra_rows
        end
    end
    return
end


end