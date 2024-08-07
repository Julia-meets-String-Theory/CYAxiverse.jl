"""
    CYAxiverse.jlm_minimizer
Here we define functions related to JLM's python minimization methods

"""
module jlm_minimizer_test
using HDF5
using LinearAlgebra, Distributions
using GenericLinearAlgebra

using ..jlm_python_test: one_dim_axion_solver, multi_axion_solver
using ..generate: αmatrix, LQtilde, phase, vacua_SNF
using ..filestructure: minfile, paths_cy
using ..structs: GeometryIndex, Canonicalα, Solver1D, SolverND, Min_JLM_1D, Min_JLM_ND, Min_JLM_Square

"""
    jlm_minimize(geom_idx::GeometryIndex)

If the effective instanton charge matrix, `Q`, is not square, this function will compute the number of vacua in the potential using the methods outlined in `arXiv: 2306.XXXXX`.
"""
function minimize(geom_idx::GeometryIndex; random_phase=false, threshold = 0.01, hilbert = false)
    αtest = αmatrix(geom_idx; threshold=threshold, hilbert = hilbert)
    if typeof(αtest)<:Canonicalα
        Qtilde = LQtilde(geom_idx; hilbert = hilbert).Qtilde
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
            println("Q_reduced_temp is square")
            return Min_JLM_Square(det_Q_tilde, Int(floor(sqrt(abs(det(αtest.Qhat * αtest.Qhat'))))))
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
        println("Natively square")
        return Min_JLM_Square(Int(abs(round(det(αtest.Qhat)))), Int(floor(sqrt(abs(det(αtest.Qhat * αtest.Qhat'))))))
    end
end

function minimize(Q::Matrix{Int}, L::Matrix{Float64}; random_phase=false, threshold = 0.01)
    αtest = αmatrix(Q, L; threshold=threshold)
    if typeof(αtest)<:Canonicalα
        Qtilde = LQtilde(Q, L).Qtilde
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
            return Min_JLM_Square(det_Q_tilde, Int(floor(sqrt(abs(det(αtest.Qhat * αtest.Qhat'))))))
        else
            phase_vector = phase(size(Qtilde, 2), αtest)
            if random_phase
                phase_vector = mod.(phase_vector .+ rand(Uniform(0, 2π), size(phase_vector, 1)), 2π)
            end
            L_reduced = Matrix(hcat(αtest.Lhat[:, axes(Qtilde, 2)][:, αtest.αrowmask], αtest.Lhat[:, size(Qtilde, 2)+1:end][:, αtest.αcolmask])')
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
        return Min_JLM_Square(Int(abs(round(det(αtest.Qhat)))), Int(floor(sqrt(abs(det(αtest.Qhat * αtest.Qhat'))))))
    end
end

function minimize_save(geom_idx::GeometryIndex; random_phase=false, threshold = 0.01, hilbert = false)
    min_data = minimize(geom_idx; random_phase=random_phase, threshold = threshold, hilbert = hilbert)
    if hilbert
        if typeof(min_data) <: Min_JLM_Square
            h5open(minfile(geom_idx),isfile(minfile(geom_idx)) ? "r+" : "cw") do file
                if haskey(file, "hilbert")
                    HDF5.delete_object(file, "hilbert/")
                    f1 = create_group(file, "hilbert/")
                    f1["Nvac", deflate=9] = min_data.N_min
                    f1["det_QTilde", deflate=9] = min_data.det_QTilde
                    f1["issquare", deflate=9] = 1
                else
                    f1 = create_group(file, "hilbert/")
                    f1["Nvac", deflate=9] = min_data.N_min
                    f1["det_QTilde", deflate=9] = min_data.det_QTilde
                    f1["issquare", deflate=9] = 1
                end
            end
        elseif typeof(min_data) <: Min_JLM_1D || typeof(min_data) <: Min_JLM_ND
            h5open(minfile(geom_idx),isfile(minfile(geom_idx)) ? "r+" : "cw") do file
                if haskey(file, "hilbert")
                    HDF5.delete_object(file, "hilbert/")
                    f1 = create_group(file, "hilbert/")
                    f1["Nvac", deflate = 9] = min_data.N_min
                    f1["vac_coords", deflate = 9] = min_data.min_coords
                    f1["extra_rows", deflate = 9] = min_data.extra_rows
                    f1["det_QTilde", deflate = 9] = min_data.det_QTilde
                    f1["issquare", deflate=9] = 0
                else
                    f1 = create_group(file, "hilbert/")
                    f1["Nvac", deflate = 9] = min_data.N_min
                    f1["vac_coords", deflate = 9] = min_data.min_coords
                    f1["extra_rows", deflate = 9] = min_data.extra_rows
                    f1["det_QTilde", deflate = 9] = min_data.det_QTilde
                    f1["issquare", deflate=9] = 0
                end
            end
        end
    else
        if typeof(min_data) <: Min_JLM_Square
            h5open(minfile(geom_idx),isfile(minfile(geom_idx)) ? "r+" : "cw") do file
                if haskey(file, "Nvac")
                    HDF5.delete_object(file, "Nvac")
                    file["Nvac", deflate=9] = min_data.N_min
                else
                    file["Nvac", deflate=9] = min_data.N_min
                end
                if haskey(file, "issquare")
                    HDF5.delete_object(file, "issquare")
                    file["issquare", deflate=9] = 1
                else
                    file["issquare", deflate=9] = 1
                end
                if haskey(file, "det_QTilde")
                    HDF5.delete_object(file, "det_QTilde")
                    file["det_QTilde", deflate=9] = min_data.det_QTilde
                else
                    file["det_QTilde", deflate=9] = min_data.det_QTilde
                end
            end
        elseif typeof(min_data) <: Min_JLM_1D || typeof(min_data) <: Min_JLM_ND
            h5open(minfile(geom_idx),isfile(minfile(geom_idx)) ? "r+" : "cw") do file
                if haskey(file, "Nvac")
                    HDF5.delete_object(file, "Nvac")
                    file["Nvac", deflate = 9] = min_data.N_min
                else
                    file["Nvac", deflate = 9] = min_data.N_min
                end
                if haskey(file, "issquare")
                    HDF5.delete_object(file, "issquare")
                    file["issquare", deflate=9] = 0
                else
                    file["issquare", deflate=9] = 0
                end
                if haskey(file, "vac_coords")
                    HDF5.delete_object(file, "vac_coords")
                    file["vac_coords", deflate=9] = min_data.min_coords
                else
                    file["vac_coords", deflate=9] = min_data.min_coords
                end
                if haskey(file, "extra_rows")
                    HDF5.delete_object(file, "extra_rows")
                    file["extra_rows", deflate=9] = min_data.extra_rows
                else
                    file["extra_rows", deflate=9] = min_data.extra_rows
                end
                if haskey(file, "det_QTilde")
                    HDF5.delete_object(file, "det_QTilde")
                    file["det_QTilde", deflate=9] = min_data.det_QTilde
                else
                    file["det_QTilde", deflate=9] = min_data.det_QTilde
                end
            end
        end
    end
end


end