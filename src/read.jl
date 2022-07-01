module read
using HDF5
using ..filestructure: cyax_file, minfile
###########################
##### Read CYTools data ###
###########################

function topology(h11::Int,tri::Int,cy::Int=1)
    poly_points::Matrix{Int}, simplices::Matrix{Int} = h5open(cyax_file(h11,tri,cy), "r") do file
    read(file, "cytools/geometric/points"),read(file, "cytools/geometric/simplices")
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
        read(file, "cytools/geometric/h21"),read(file, "cytools/geometric/glsm"),
        read(file, "cytools/geometric/basis"),read(file, "cytools/geometric/tip"),
        read(file, "cytools/geometric/CY_volume"),read(file, "cytools/geometric/divisor_volumes"),
        read(file, "cytools/geometric/Kinv")
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
        read(file, "cytools/potential/L"),read(file, "cytools/potential/Q"),
        read(file, "cytools/geometric/Kinv")
    end
    L = L'
    Q = Q'
    keys = ["L","Q","K"]
    vals = [L', Q', Hermitian(inv(Kinv))]
    return Dict(zip(keys,vals))
end

function Q(h11::Int,tri::Int,cy::Int=1)
    Q::Matrix{Int} = h5open(cyax_file(h11,tri,cy), "r") do file
        read(file, "cytools/potential/Q")
    end
    Q = Q'
    return Q
end

function K(h11::Int,tri::Int,cy::Int=1)
    K::Matrix{Float64} = h5open(cyax_file(h11,tri,cy), "r") do file
        read(file, "cytools/potential/K")
    end
    K = 0.5.* (K+transpose(K))
    return Hermitian(K)
end

function L(h11::Int,tri::Int,cy::Int=1)
    L::Matrix{Float64} = h5open(cyax_file(h11,tri,cy), "r") do file
        read(file, "cytools/potential/L")
    end
    L = L'
    Ltemp::Vector{ArbFloat} = zeros(ArbFloat,size(L,2))
    @inbounds for i=1:size(L,2)
        Ltemp[i] = ArbFloat.(L[1,i]) .* ArbFloat(10.) .^ ArbFloat.(L[2,i])
    end
    return Ltemp
end


#########################
##### Read Vacua data ###
#########################

function vacua(h11::Int,tri::Int,cy::Int=1)
    vacua::Int, θparallel_num::Matrix{Int}, θparallel_den::Matrix{Int}, Qtilde::Matrix{Int} = h5open(cyax_file(h11,tri,cy), "r") do file
    read(file, "vacua/vacua"),read(file, "vacua/thetaparallel/numerator"),
        read(file, "vacua/thetaparallel/denominator"),read(file, "vacua/Qtilde")
    end
    keys = ["vacua","θ||","Qtilde"]
    vals = [abs(vacua), θparallel_num .// θparallel_den, Qtilde]
    return Dict(zip(keys,vals))
end



function pq_spectrum(h11::Int,tri::Int,cy::Int=1)
    Hvals::Vector{Float64}, fK::Vector{Float64}, fpert::Vector{Float64} = 
    h5open(cyax_file(h11,tri,cy), "r") do file
        read(file, "spectrum/Heigvals/log10"),
        read(file, "spectrum/decay/fK"), read(file, "spectrum/decay/fpert")
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
    read(file, "spectrum/Heigvals/sign"),read(file, "spectrum/Heigvals/log10"),
    read(file, "spectrum/decay/fK"), read(file, "spectrum/decay/fpert"),read(file, "spectrum/quartdiag/sign"),
        read(file, "spectrum/quartdiag/log10"),read(file, "spectrum/quart31/index"),read(file, "spectrum/quart31/sign"),
        read(file, "spectrum/quart31/log10"),read(file, "spectrum/quart22/index"),read(file, "spectrum/quart22/sign"),
        read(file, "spectrum/quart22/log10")
    end
    keys = ["msign","m", "fK", "fpert","λselfsign", "λself","λ31_i","λ31sign","λ31", "λ22_i","λ22sign","λ22"]
    vals = [Hsign,Hvals, fK, fpert,quartdiagsign, quartdiaglog,
    quart22_index,quart22_log10 ,quart31_index,quart31_log10]
    return Dict(zip(keys,vals))
end

end

