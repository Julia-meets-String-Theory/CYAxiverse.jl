"""
    CYAxiverse.read
Functions that access the database.

"""
module read
using HDF5
using LinearAlgebra
using ..filestructure: cyax_file, minfile
###########################
##### Read CYTools data ###
###########################

function topology(h11::Int,tri::Int,cy::Int=1)
    poly_points::Matrix{Int}, simplices::Matrix{Int} = h5open(cyax_file(h11,tri,cy), "r") do file
    HDF5.read(file, "cytools/geometric/points"),HDF5.read(file, "cytools/geometric/simplices")
    end
    keys = ["points","simplices"]
    vals = [poly_points, simplices]
    return Dict(zip(keys,vals))
end
        
function geometry(h11::Int,tri::Int,cy::Int=1)
    h21::Int,
    glsm::Matrix{Int},basis::Vector{Int},
    tip::Vector{Float64},CY_Volume::Float64,divisor_volumes::Vector{Float64},
    Kinv::Matrix{Float64}= h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/geometric/h21"),HDF5.read(file, "cytools/geometric/glsm"),
        HDF5.read(file, "cytools/geometric/basis"),HDF5.read(file, "cytools/geometric/tip"),
        HDF5.read(file, "cytools/geometric/CY_volume"),HDF5.read(file, "cytools/geometric/divisor_volumes"),
        HDF5.read(file, "cytools/geometric/Kinv")
    end
    keys = ["h21","glsm_charges","basis","tip","CYvolume","τ_volumes","Kinv"]
    vals = [h21,
    glsm,basis,
    tip,CY_Volume,divisor_volumes,
    Kinv]
    return Dict(zip(keys,vals))
end


#############################
##### Read Geometric data ###
#############################

function potential(h11::Int,tri::Int,cy::Int=1)
    L::Matrix{Float64}, Q::Matrix{Int},
    Kinv::Matrix{Float64}= h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/potential/L"),HDF5.read(file, "cytools/potential/Q"),
        HDF5.read(file, "cytools/geometric/Kinv")
    end
    keys = ["L","Q","K"]
    vals = [L, Q, Hermitian(inv(Kinv))]
    return Dict(zip(keys,vals))
end

function Q(h11::Int,tri::Int,cy::Int=1)
    Q::Matrix{Int} = h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/potential/Q")
    end
    return Q
end

function K(h11::Int,tri::Int,cy::Int=1)
    K::Matrix{Float64} = h5open(cyax_file(h11,tri,cy), "r") do file
        HDF5.read(file, "cytools/potential/K")
    end
    K = 0.5.* (K+transpose(K))
    return Hermitian(K)
end

function L_log(h11::Int,tri::Int,cy::Int=1)
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
    @inbounds for i=1:size(L,1)
        Ltemp[i] = ArbFloat.(L[i,1]) .* ArbFloat(10.) .^ ArbFloat.(L[i,2])
    end
    return Ltemp
end


##############################
##### HDF5.read Vacua data ###
##############################

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

