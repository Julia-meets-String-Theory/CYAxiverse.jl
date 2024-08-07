"""
    CYAxiverse.read
Functions that access the database.

"""
module read
using HDF5
using LinearAlgebra
using ..filestructure: cyax_file, minfile, geom_dir_read
using ..structs: GeometryIndex, TopologicalData, GeometricData, AxionPotential, Min_JLM_1D, Min_JLM_ND, Min_JLM_Square
###########################
##### Read CYTools data ###
###########################

function topology(h11::Int,tri::Int,cy::Int=1)
    poly_points::Matrix{Int}, simplices::Matrix{Int} = h5open(cyax_file(h11,tri,cy), "r") do file
    HDF5.read(file, "cytools/geometric/points"),HDF5.read(file, "cytools/geometric/simplices")
    end
    return TopologicalData(poly_points, simplices)
end

function topology(geom_idx::GeometryIndex)
    h11, tri, cy = geom_idx.h11, geom_idx.polytope, geom_idx.frst
    topology(h11, tri, cy)
end

function geometry(h11::Int,tri::Int,cy::Int=1; hilbert = false)
    tip_prefactor = nothing
    hilbert_basis = nothing
    h21::Int = 0
    glsm::Matrix{Int} = zeros(Int, h11, h11)
    basis::Vector{Int} = zeros(Int, h11)
    tip::Vector{Float64} = zeros(Float64, h11)
    CY_Volume::Float64 = 0.
    divisor_volumes::Vector{Float64} = zeros(Float64, h11)
    Kinv::Matrix{Float64} = zeros(Float64, h11, h11)
    if hilbert
        h21, glsm, basis, tip, CY_Volume, divisor_volumes, Kinv = 
        h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "cytools/geometric/h21"),HDF5.read(file, "cytools/geometric/glsm"),
            HDF5.read(file, "cytools/geometric/basis"),HDF5.read(file, "cytools/hilbert/geometric/tip"),
            HDF5.read(file, "cytools/hilbert/geometric/CY_volume"),HDF5.read(file, "cytools/hilbert/geometric/divisor_volumes"),
            HDF5.read(file, "cytools/hilbert/geometric/Kinv")
        end
        h5open(cyax_file(h11,tri,cy), "r") do file
            if haskey(file, "cytools/geometric/tip_prefactor")
                tip_prefactor = HDF5.read(file, "cytools/hilbert/geometric/tip_prefactor")
            end
            if haskey(file, "cytools/geometric/hilbert_basis")
                hilbert_basis = HDF5.read(file, "cytools/geometric/hilbert_basis")
            end
        end
        if tip_prefactor !== nothing && hilbert_basis !== nothing
            # keys = ["h21","glsm_charges","basis","tip","tip_prefactor", "CYvolume","τ_volumes","Kinv"]
            # vals = [h21,
            # glsm,basis,
            # tip,tip_prefactor, CY_Volume,divisor_volumes,
            # Kinv]
            # return Dict(zip(keys,vals))
            return GeometricData(tip_prefactor, divisor_volumes, h21, CY_Volume, glsm, basis, tip, Kinv, hilbert_basis)
        else
            # keys = ["h21","glsm_charges","basis","tip", "CYvolume","τ_volumes","Kinv"]
            # vals = [h21,
            # glsm,basis,
            # tip, CY_Volume,divisor_volumes,
            # Kinv]
            # return Dict(zip(keys,vals))
            tip_prefactor = ones(Float64, 2)
            hilbert_basis = zeros(h11, h11)
            return GeometricData(tip_prefactor, divisor_volumes, h21, CY_Volume, glsm, basis, tip, Kinv, hilbert_basis)
        end
    else
        h21, glsm, basis, tip, CY_Volume, divisor_volumes, Kinv = 
        h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "cytools/geometric/h21"),HDF5.read(file, "cytools/geometric/glsm"),
            HDF5.read(file, "cytools/geometric/basis"),HDF5.read(file, "cytools/geometric/tip"),
            HDF5.read(file, "cytools/geometric/CY_volume"),HDF5.read(file, "cytools/geometric/divisor_volumes"),
            HDF5.read(file, "cytools/geometric/Kinv")
        end
        h5open(cyax_file(h11,tri,cy), "r") do file
            if haskey(file, "cytools/geometric/tip_prefactor")
                tip_prefactor = HDF5.read(file, "cytools/geometric/tip_prefactor")
            end
            if haskey(file, "cytools/geometric/hilbert_basis")
                hilbert_basis = HDF5.read(file, "cytools/geometric/hilbert_basis")
            end
        end
        if tip_prefactor !== nothing && hilbert_basis !== nothing
            # keys = ["h21","glsm_charges","basis","tip","tip_prefactor", "CYvolume","τ_volumes","Kinv"]
            # vals = [h21,
            # glsm,basis,
            # tip,tip_prefactor, CY_Volume,divisor_volumes,
            # Kinv]
            # return Dict(zip(keys,vals))
            return GeometricData(tip_prefactor, divisor_volumes, h21, CY_Volume, glsm, basis, tip, Kinv, hilbert_basis)
        else
            # keys = ["h21","glsm_charges","basis","tip", "CYvolume","τ_volumes","Kinv"]
            # vals = [h21,
            # glsm,basis,
            # tip, CY_Volume,divisor_volumes,
            # Kinv]
            # return Dict(zip(keys,vals))
            tip_prefactor = ones(Float64, 2)
            hilbert_basis = zeros(h11, h11)
            return GeometricData(tip_prefactor, divisor_volumes, h21, CY_Volume, glsm, basis, tip, Kinv, hilbert_basis)
        end
    end
end


function geometry(geom_idx::GeometryIndex; hilbert = false)
    h11, tri, cy = geom_idx.h11, geom_idx.polytope, geom_idx.frst
    geometry(h11, tri, cy; hilbert = hilbert)
end

function hilbert_basis(geom_idx::GeometryIndex)
    basis = zeros(geom_idx.h11, geom_idx.h11)
    h5open(cyax_file(geom_idx), "r") do file
        if haskey(file, "cytools/geometric/hilbert_basis")
            basis = HDF5.read(file, "cytools/geometric/hilbert_basis")
        end
    end
    basis
end

#############################
##### Read Geometric data ###
#############################

function potential(geom_idx::GeometryIndex; hilbert = false)
    L::Matrix{Float64} = zeros(Float64, 1, 1)
    Q::Matrix{Int} = zeros(Int, 1, 1)
    Kinv::Matrix{Float64} = zeros(Float64, geom_idx.h11, geom_idx.h11)
    if hilbert
        L, Q, Kinv = 
        h5open(cyax_file(geom_idx), "r") do file
            HDF5.read(file, "cytools/hilbert/potential/L"),HDF5.read(file, "cytools/hilbert/potential/Q"),
            HDF5.read(file, "cytools/hilbert/geometric/Kinv")
        end
        AxionPotential(L, Q, Hermitian(inv(Kinv)))
    else
        L, Q, Kinv = 
        h5open(cyax_file(geom_idx), "r") do file
            HDF5.read(file, "cytools/potential/L"),HDF5.read(file, "cytools/potential/Q"),
            HDF5.read(file, "cytools/geometric/Kinv")
        end
        AxionPotential(L, Q, Hermitian(inv(Kinv)))
    end
end


function potential(h11::Int,tri::Int,cy::Int=1; hilbert = false)
    geom_idx = GeometryIndex(h11, tri, cy)
    potential(geom_idx; hilbert = hilbert)
end

function Q(h11::Int,tri::Int,cy::Int=1; hilbert = false)
    Q::Matrix{Int} = h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/potential/Q")
    end
    return Q
end

function K(h11::Int,tri::Int,cy::Int=1; hilbert = false)
    K::Matrix{Float64} = h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/potential/K")
    end
    K = 0.5.* (K+transpose(K))
    return Hermitian(K)
end

function L_log(h11::Int,tri::Int,cy::Int=1; hilbert = false)
    L::Matrix{Float64} = h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/potential/L")
    end
    return L
end

function L_arb(h11::Int,tri::Int,cy::Int=1)
    L::Matrix{Float64} = h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/potential/L")
    end
    Ltemp::Vector{ArbFloat} = zeros(ArbFloat,size(L,2))
    @inbounds for i in axes(L,1)
        Ltemp[i] = ArbFloat.(L[i,1]) .* ArbFloat(10.) .^ ArbFloat.(L[i,2])
    end
    return Ltemp
end


##############################
##### HDF5.read Vacua data ###
##############################

function qshape(h11::Int,tri::Int,cy::Int=1)
    square, vacua, extrarows, ωnorm2 = 0, 0, 0, 0
    h5open(joinpath(geom_dir_read(h11,tri,cy),"qshape.h5"), "r") do file
        square = HDF5.read(file, "square")
        vacua = HDF5.read(file, "vacua_estimate")
        if haskey(file, "extra_rows")
            extrarows = HDF5.read(file, "extra_rows")
        end
        if haskey(file, "ωnorm2_estimate")
            ωnorm2 =  HDF5.read(file, "ωnorm2_estimate")
        end
    end
    (issquare = square, vacua_det = vacua, lengthα = extrarows, ωnorm2 = ωnorm2)
end

function qshape(geom_idx::GeometryIndex)
    h11, tri, cy = geom_idx.h11, geom_idx.polytope, geom_idx.frst
    qshape(h11, tri, cy)
end

function vacua(h11::Int,tri::Int,cy::Int=1)
    vacua::Float64 = 0.
    θparallel_num::Matrix{Int} = zeros(Int,1,1)
    θparallel_den::Matrix{Int} = zeros(Int,1,1)
    Qtilde::Matrix{Int} = zeros(Int,1,1)
    θparallel::Matrix{Float32} = zeros(Float32,1,1)
    if h11 <= 50
        vacua, θparallel_num, θparallel_den, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua/vacua"),HDF5.read(file, "vacua/thparallel/numerator"),HDF5.read(file, "vacua/thparallel/denominator"),HDF5.read(file, "vacua/Qtilde")
        end
        keys = ["vacua","θ∥","Qtilde"]
        vals = [abs(vacua), θparallel_num .// θparallel_den, Qtilde]
        return Dict(zip(keys,vals))
    else
        vacua, θparallel, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua/vacua"),HDF5.read(file, "vacua/thparallel"),HDF5.read(file, "vacua/Qtilde")
        end
        keys = ["vacua","θ∥","Qtilde"]
        vals = [abs(vacua), Rational.(round.(θparallel; digits=8)), Qtilde]
        return Dict(zip(keys,vals))
    end
end

function vacua_TB(h11::Int,tri::Int,cy::Int=1)
    vacua::Float64 = 0
    θparallel_num::Matrix{Int} = zeros(Int,1,1)
    θparallel_den::Matrix{Int} = zeros(Int,1,1)
    Qtilde::Matrix{Int} = zeros(Int,1,1)
    θparallel::Matrix{Float32} = zeros(Float32,1,1)
    if h11 <= 50
        vacua, θparallel_num, θparallel_den, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua_TB/vacua"),HDF5.read(file, "vacua_TB/thparallel/numerator"),HDF5.read(file, "vacua_TB/thparallel/denominator"),HDF5.read(file, "vacua_TB/Qtilde")
        end
        keys = ["vacua","θ∥","Qtilde"]
        vals = [abs(vacua), θparallel_num .// θparallel_den, Qtilde]
        return Dict(zip(keys,vals))
    else
        vacua, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua_TB/vacua"),HDF5.read(file, "vacua_TB/Qtilde")
        end
        keys = ["vacua","Qtilde"]
        vals = [abs(vacua), Qtilde]
        return Dict(zip(keys,vals))
    end
end

function vacua_jlm(geom_idx::GeometryIndex; hilbert = false)
    Nvac = 0
    min_coords = zeros(1,1)
    extra_rows = 0
    det_Qtilde = 0
    if hilbert
        h5open(minfile(geom_idx), "r") do file
            Nvac = HDF5.read(file, "hilbert/Nvac")
            if haskey(file, "hilbert/extra_rows")
                min_coords = HDF5.read(file, "hilbert/vac_coords")
                extra_rows = HDF5.read(file, "hilbert/extra_rows")
            end
            if haskey(file, "hilbert/det_QTilde")
                det_Qtilde = HDF5.read(file, "hilbert/det_QTilde")
            end
        end
    else
        h5open(minfile(geom_idx), "r") do file
            Nvac = HDF5.read(file, "Nvac")
            if haskey(file, "extra_rows")
                min_coords = HDF5.read(file, "vac_coords")
                extra_rows = HDF5.read(file, "extra_rows")
            end
            if haskey(file, "det_QTilde")
                det_Qtilde = HDF5.read(file, "det_QTilde")
            end
        end
    end
    if extra_rows == 0
        return Min_JLM_Square(Nvac, det_Qtilde)
    elseif extra_rows == 1
        return Min_JLM_1D(Nvac, vec(min_coords), extra_rows, det_Qtilde)
    else
        return Min_JLM_ND(Nvac, min_coords, extra_rows, det_Qtilde)
    end
end

################################
##### HDF5.read Spectra data ###
################################

function pq_spectrum(h11::Int,tri::Int,cy::Int=1)
    Hvals::Vector{Float64}, fK::Vector{Float64}, fpert::Vector{Float64} = 
    h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "spectrum/masses/log10"),
        HDF5.read(file, "spectrum/decay/fK"), HDF5.read(file, "spectrum/decay/fpert")
    end
    keys = ["m", "fK", "fpert"]
    vals = [Hvals, fK, fpert]
    return Dict(zip(keys,vals))
end

function hp_spectrum(h11::Int,tri::Int,cy::Int=1)
    Hsign::Vector{Int64}, Hvals::Vector{Float64}, fK::Vector{Float64}, fpert::Vector{Float64},
    quartdiagsign::Vector{Int64},quartdiaglog::Vector{Float64},
    quart22_index,quart22_sign::Vector{Int},quart22_log10::Vector{Float64},quart31_index,
    quart31_sign::Vector{Int},
    quart31_log10::Vector{Float64} = h5open(cyax_file(h11,tri,cy), "r") do file
    HDF5.read(file, "spectrum/masses/sign"),HDF5.read(file, "spectrum/masses/log10"),
    HDF5.read(file, "spectrum/decay/fK"), HDF5.read(file, "spectrum/decay/fpert"),HDF5.read(file, "spectrum/quartdiag/sign"),
        HDF5.read(file, "spectrum/quartdiag/log10"),HDF5.read(file, "spectrum/quart31/index"),HDF5.read(file, "spectrum/quart31/sign"),
        HDF5.read(file, "spectrum/quart31/log10"),HDF5.read(file, "spectrum/quart22/index"),HDF5.read(file, "spectrum/quart22/sign"),
        HDF5.read(file, "spectrum/quart22/log10")
    end
    keys = ["msign","m", "fK", "fpert","λselfsign", "λself","λ31_i","λ31sign","λ31", "λ22_i","λ22sign","λ22"]
    vals = [Hsign,Hvals, fK, fpert,quartdiagsign, quartdiaglog,
    quart22_index,quart22_log10 ,quart31_index,quart31_log10]
    return Dict(zip(keys,vals))
end

end

