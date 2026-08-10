"""
    CYAxiverse.read
Functions that access the database.

"""
module read
using HDF5
using LinearAlgebra
using ArbNumerics: ArbFloat
using ..filestructure: cyax_file, minfile, geom_dir_read
using ..structs: GeometryIndex, TopologicalData, GeometricData, AxionPotential, Min_JLM_1D, Min_JLM_ND, Min_JLM_Square

function _read_dataset(group::HDF5.Group, name::AbstractString)
    HDF5.read(group[name]::HDF5.Dataset)
end
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

function geometry(h11::Int, tri::Int, cy::Int=1; hilbert=false)
    h5open(cyax_file(h11, tri, cy), "r") do file
        if hilbert
            h21 = HDF5.read(file, "cytools/geometric/h21")
            glsm = HDF5.read(file, "cytools/geometric/glsm")
            basis = HDF5.read(file, "cytools/geometric/basis")
            tip = HDF5.read(file, "cytools/hilbert/geometric/tip")
            CY_Volume = HDF5.read(file, "cytools/hilbert/geometric/CY_volume")
            divisor_volumes = HDF5.read(file, "cytools/hilbert/geometric/divisor_volumes")
            Kinv = HDF5.read(file, "cytools/hilbert/geometric/Kinv")

            tip_prefactor = haskey(file, "cytools/geometric/tip_prefactor") ? HDF5.read(file, "cytools/hilbert/geometric/tip_prefactor") : ones(Float64, 2)
        else
            h21 = HDF5.read(file, "cytools/geometric/h21")
            glsm = HDF5.read(file, "cytools/geometric/glsm")
            basis = HDF5.read(file, "cytools/geometric/basis")
            tip = HDF5.read(file, "cytools/geometric/tip")
            CY_Volume = HDF5.read(file, "cytools/geometric/CY_volume")
            divisor_volumes = HDF5.read(file, "cytools/geometric/divisor_volumes")
            Kinv = HDF5.read(file, "cytools/geometric/Kinv")

            tip_prefactor = haskey(file, "cytools/geometric/tip_prefactor") ? HDF5.read(file, "cytools/geometric/tip_prefactor") : ones(Float64, 2)
        end
        
        hilbert_basis::Matrix{Int} = haskey(file, "cytools/geometric/hilbert_basis") ? HDF5.read(file, "cytools/geometric/hilbert_basis") : zeros(Int, h11, h11)

        return GeometricData(tip_prefactor::Vector{Float64}, divisor_volumes::Vector{Float64}, h21::Int, CY_Volume::Float64, glsm::Matrix{Int}, basis::Vector{Int}, tip::Vector{Float64}, Kinv::Matrix{Float64}, hilbert_basis)
    end
end


function geometry(geom_idx::GeometryIndex; hilbert = false)
    h11, tri, cy = geom_idx.h11, geom_idx.polytope, geom_idx.frst
    geometry(h11, tri, cy; hilbert = hilbert)
end

function hilbert_basis(geom_idx::GeometryIndex)
    basis = zeros(Int, geom_idx.h11, geom_idx.h11)
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
    if hilbert
        L::Matrix{Float64}, Q::Matrix{Int}, Kinv::Matrix{Float64} = 
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

"""
    oriented_potential(geom_idx::GeometryIndex)

Load a geometry potential and return its canonical screening orientation as a
named tuple `(Q, L, K)`: `Q` is `h11 × n_instantons`, `L` is
`2 × n_instantons`, and `K` is an `h11 × h11` Hermitian Float64 matrix.

The helper accepts the two matrix orientations emitted by older geometry
files, validates dimensions and finite kinetic/log-scale data, and owns the
normalization used by numerical screening callers. It does not apply scan
thresholds or candidate policy.
"""
function oriented_potential(geom_idx::GeometryIndex)
    potential_data = potential(geom_idx)
    Q = Matrix{Int}(potential_data.Q)
    L = Matrix{Float64}(potential_data.L)
    if size(L, 1) != 2 && size(L, 2) == 2
        L = Matrix(L')
    end
    if size(Q, 2) != size(L, 2) && size(Q, 1) == size(L, 2)
        Q = Matrix(Q')
    end
    size(L, 1) == 2 || throw(DimensionMismatch("L must have two rows"))
    size(Q, 2) == size(L, 2) ||
        throw(DimensionMismatch("Q and L must have the same instanton count"))
    K = Hermitian(Matrix{Float64}(potential_data.K))
    size(K, 1) == size(K, 2) ||
        throw(DimensionMismatch("K must be square"))
    size(Q, 1) == size(K, 1) ||
        throw(DimensionMismatch("Q and K must have the same axion count"))
    all(isfinite, L) || throw(ArgumentError("L contains non-finite values"))
    all(isfinite, Matrix(K)) ||
        throw(ArgumentError("K contains non-finite values"))
    (; Q, L, K)
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
        mantissa = ArbFloat(L[i,1])
        exponent = ArbFloat(10.) ^ ArbFloat(L[i,2])
        Ltemp[i] = mantissa * exponent
        end
    return Ltemp
end

"""
    cubic_tensor(h11, tri, cy=1)

Read the cubic interaction tensor saved by `pq_spectrum_save` or
`hp_spectrum_save`.  Returns a named tuple with `tensor` and the evaluation
`phase`; the tensor is stored in the corresponding saved spectrum basis.
"""
function cubic_tensor(h11::Int, tri::Int, cy::Int=1)
    h5open(cyax_file(h11, tri, cy), "r") do file
        tensor = HDF5.read(file, "spectrum/cubic/tensor")
        phase = HDF5.read(file, "spectrum/cubic/phase")
        return (; tensor=tensor, phase=phase)
    end
end

function cubic_tensor(geom_idx::GeometryIndex)
    cubic_tensor(geom_idx.h11, geom_idx.polytope, geom_idx.frst)
end


##############################
##### HDF5.read Vacua data ###
##############################

function qshape(h11::Int,tri::Int,cy::Int=1)
    extrarows = 0
    ωnorm2 = 0.0
    square = 0
    vacua = 0.0
    root = geom_dir_read(h11, tri, cy)
    root === nothing && throw(ArgumentError("geometry directory is not available for h11=$h11, tri=$tri, cy=$cy"))
    h5open(joinpath(root, "qshape.h5"), "r") do file
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
    if h11 <= 50
        vacua, θparallel_num::Matrix{Int}, θparallel_den::Matrix{Int}, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua/vacua"),HDF5.read(file, "vacua/thparallel/numerator"),HDF5.read(file, "vacua/thparallel/denominator"),HDF5.read(file, "vacua/Qtilde")
        end
        θ_parallel = θparallel_num .//θparallel_den
    else
        vacua, θparallel, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua/vacua"),HDF5.read(file, "vacua/thparallel"),HDF5.read(file, "vacua/Qtilde")
        end
        θ_parallel = Rational.(round.(θparallel; digits=8))
    end
    return (; vacua = abs(vacua::Float64), θ_parallel = θ_parallel::Matrix{Rational}, Qtilde = Qtilde::Matrix{Int})
end

function vacua_TB(h11::Int,tri::Int,cy::Int=1)
    if h11 <= 50
        vacua, θparallel_num::Matrix{Int}, θparallel_den::Matrix{Int}, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua_TB/vacua"),HDF5.read(file, "vacua_TB/thparallel/numerator"),HDF5.read(file, "vacua_TB/thparallel/denominator"),HDF5.read(file, "vacua_TB/Qtilde")
        end
        return (; vacua = abs(vacua::Float64), θ_parallel = Matrix{Rational}(θparallel_num .// θparallel_den), Qtilde = Qtilde::Matrix{Int})
    else
        vacua, Qtilde = h5open(cyax_file(h11,tri,cy), "r") do file
            HDF5.read(file, "vacua_TB/vacua"),HDF5.read(file, "vacua_TB/Qtilde")
        end
        return (; vacua = abs(vacua::Float64), Qtilde = Qtilde::Matrix{Int})
    end
end

function vacua_jlm(geom_idx::GeometryIndex; hilbert = false)
    Nvac = 0
    min_coords = zeros(1,1)
    extra_rows = 0
    det_Qtilde = 0.0
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
    return (; m = Hvals, fK = fK, fpert = fpert)
end

"""
    physical_spectrum(h11, tri, cy=1)

Read the persisted physical spectrum and its metadata from a geometry's
`spectrum/physical` HDF5 group.
"""
function physical_spectrum(h11::Int, tri::Int, cy::Int=1)
    h5open(cyax_file(h11, tri, cy), "r") do file
        physical = file["spectrum/physical"]::HDF5.Group
        metadata = physical["metadata"]::HDF5.Group
        return (; m = _read_dataset(physical, "m"),
            mode_indices = _read_dataset(physical, "mode_indices"),
            mass_signs_or_inertia = haskey(physical, "mass_signs_or_inertia") ? _read_dataset(physical, "mass_signs_or_inertia") : Int[],
            fK = _read_dataset(physical, "fK_log10"),
            λselfsign = haskey(physical, "lambda_self_sign") ? _read_dataset(physical, "lambda_self_sign") : Int[],
            λself = haskey(physical, "lambda_self_log10") ? _read_dataset(physical, "lambda_self_log10") : Float64[],
            fpert = haskey(physical, "fpert_log10") ? _read_dataset(physical, "fpert_log10") : Float64[],
            threshold_log10 = _read_dataset(metadata, "threshold_log10"),
            prec = _read_dataset(metadata, "prec"),
            provisional = _read_dataset(metadata, "provisional"),
            runtime_seconds = _read_dataset(metadata, "runtime_seconds"))
    end
end

"""Read a persisted physical spectrum using a [`GeometryIndex`](@ref)."""
function physical_spectrum(geom_idx::GeometryIndex)
    physical_spectrum(geom_idx.h11, geom_idx.polytope, geom_idx.frst)
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
    return (; m = Hvals, fK = fK, fpert = fpert, λself = quartdiaglog, λ31_i = quart31_index, λ31 = quart31_log10, λ22_i = quart22_index, λ22 = quart22_log10)
end

"""
    pipeline_vacua(h11, tri, cy=1)

Read the vacua estimate and coordinates written by `scripts/vacua_pipeline.jl`.
"""
function pipeline_vacua(h11::Int, tri::Int, cy::Int=1)
    h5open(cyax_file(h11, tri, cy), "r") do file
        group = file["vacua_pipeline"]::HDF5.Group
        threshold = _read_dataset(group, "threshold")
        estimate = _read_dataset(group, "estimate")
        issquare = _read_dataset(group, "issquare")
        extrarows = haskey(group, "extrarows") ? _read_dataset(group, "extrarows") : nothing
        verified = haskey(group, "verified") ? _read_dataset(group, "verified") : nothing
        theta_min = haskey(group, "theta_min") ? begin
            theta_group = group["theta_min"]::HDF5.Group
            _read_dataset(theta_group, "numerator") .// _read_dataset(theta_group, "denominator")
        end : nothing
        theta_parallel = haskey(group, "theta_parallel") ? begin
            theta_group = group["theta_parallel"]::HDF5.Group
            _read_dataset(theta_group, "numerator") .// _read_dataset(theta_group, "denominator")
        end : nothing
        metadata = if haskey(group, "metadata")
            metadata_group = group["metadata"]::HDF5.Group
            read_metadata(name, default=nothing) =
                haskey(metadata_group, name) ? _read_dataset(metadata_group, name) : default
            (; pipeline_version = read_metadata("pipeline_version"),
               threshold = read_metadata("threshold"),
               starts = read_metadata("starts"),
               residual_tolerance = read_metadata("residual_tolerance"),
               merge_tolerance = read_metadata("merge_tolerance"),
               max_iterations = read_metadata("max_iterations"),
               max_branches = read_metadata("max_branches"),
               method = read_metadata("method"),
               branch_method = read_metadata("branch_method"),
               status = read_metadata("status"),
               solver_status = read_metadata("solver_status", read_metadata("status")),
               estimate_status = read_metadata("estimate_status"),
               verification_status = read_metadata("verification_status"),
               julia_version = read_metadata("julia_version"),
               git_revision = read_metadata("git_revision"),
               runtime_seconds = read_metadata("runtime_seconds"),
               completed_at = read_metadata("completed_at"),
               error = read_metadata("error"),
               search_method = read_metadata("search_method"),
               search_classification = read_metadata("search_classification"),
               minimum_count = read_metadata("minimum_count"),
               multiplicity = read_metadata("multiplicity"),
               critical_count = read_metadata("critical_count"),
               branch_count = read_metadata("branch_count"),
               det_Qtilde = read_metadata("det_Qtilde"),
               search_status = read_metadata("search_status"))
        else
            nothing
        end
        return (; threshold, estimate, issquare, extrarows, verified, theta_min,
            theta_parallel, metadata)
    end
end

"""Read vacua pipeline data for a geometry identified by `geom_idx`."""
function pipeline_vacua(geom_idx::GeometryIndex)
    pipeline_vacua(geom_idx.h11, geom_idx.polytope, geom_idx.frst)
end

end
