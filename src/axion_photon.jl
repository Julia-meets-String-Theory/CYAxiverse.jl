"""
    CYAxiverse.axion_photon

Small, replayable implementation of the leading axion hierarchy,
axion--photon coupling, and decay-width calculations. The reader is
deliberately explicit about the local-file adaptation: charge vectors are the
columns of the stored `effective_cone` matrix, and stored potential
coefficients are retained with their signs.

This namespace operates on bounded local geometry scans. When a geometry was
generated with the orientifold-compatible intersecting-D7 visible-sector
policy, it can also use the selected QED divisor and its Euclidean-D3 term to
derive the stringy light threshold.
"""
module axion_photon

using HDF5
using LinearAlgebra

using ..filestructure: resolve_data_dir
using ..structs: GeometryIndex

const M_PLANCK_GEV = 2.435e18
const ALPHA_EM = 1 / 137.035999084
const ELECTRON_MASS_EV = 0.511e6

"""Orientifold-compatible QCD/QED divisor assignment stored with a geometry."""
struct VisibleSectorAssignment{T<:AbstractFloat}
    qcd_divisor_index::Int
    qed_divisor_index::Int
    qcd_image_index::Int
    qed_image_index::Int
    qcd_divisor_volume::T
    qed_divisor_volume::T
    qcd_charge::Vector{Int}
    qed_charge::Vector{Int}
    em_charge::Vector{Int}
    qed_instanton_index::Int
    qed_log10_lambda4::T
    qcd_qed_intersection::Bool
    qcd_invariant::Bool
    qed_invariant::Bool
    policy::Symbol
end

"""Geometry fields needed by the local axion--photon calculation."""
struct GeometryInputs{T<:AbstractFloat}
    path::String
    index::GeometryIndex{Int}
    tip::Vector{T}
    divisor_volumes::Vector{T}
    cy_volume::T
    kinv::Matrix{T}
    direct_charges::Matrix{Int}
    direct_divisor_volumes::Vector{T}
    direct_labels::Vector{Int}
    visible_sector::Union{Nothing,VisibleSectorAssignment{T}}
end

"""Signed, log-scaled potential read from `cytools/potential`."""
struct InstantonData{T<:AbstractFloat}
    Q::Matrix{Int}
    log10_lambda4::Vector{T}
    coefficient_signs::Vector{Int}
    source_indices::Vector{Int}
end

"""Leading-charge hierarchy in the canonical Gram--Schmidt frame.

`Q_reduced` and `q` retain one selected instanton charge vector per column,
with the same `h11 × h11` column convention as the input `Q`. The canonical
charge matrix is `q = X' * Q_reduced` and is upper triangular.
"""
struct LeadingAxionHierarchy{T<:AbstractFloat}
    selected_indices::Vector{Int}
    dependent_indices::Vector{Int}
    # Keep one selected instanton charge vector per column, matching Q.
    Q_reduced::Matrix{Int}
    log10_lambda4::Vector{T}
    coefficient_signs::Vector{Int}
    q::Matrix{T}
    theta_from_canonical::Matrix{T}
    log10_f_GeV::Vector{T}
    log10_mass_eV::Vector{T}
    m_planck_GeV::T
    triangular_residual::T
    metric_residual::T
end

"""Axion--photon couplings and the two leading decay-width estimates."""
struct AxionPhotonObservables{T<:AbstractFloat}
    n_em::Vector{T}
    theta::Matrix{T}
    Cgamma::Vector{T}
    log10_g_GeVinv::Vector{T}
    log10_g_effective_GeVinv::Vector{T}
    log10_photon_width_GeV::Vector{T}
    log10_quartic_width_GeV::Vector{T}
    light_threshold_eV::T
    log10_light_threshold_eV::T
    light_mode_count::Int
    charge_residual::T
end

"""One local scan result, retaining all scientific intermediates."""
struct AxionPhotonResult{T<:AbstractFloat}
    geometry::GeometryInputs{T}
    potential::InstantonData{T}
    hierarchy::LeadingAxionHierarchy{T}
    photons::AxionPhotonObservables{T}
    em_divisor_index::Int
    em_divisor_volume::T
    em_charge_source::Symbol
    light_threshold_policy::Symbol
    status::Symbol
end

function _integer_matrix(values, name::AbstractString)
    values isa Matrix{Int} && return values
    all(isfinite, values) || throw(ArgumentError("$name contains non-finite values"))
    rounded = round.(Int, values)
    all(values .== rounded) || throw(ArgumentError("$name is not integral"))
    Matrix{Int}(rounded)
end

function _integer_vector(values, name::AbstractString)
    values isa Vector{Int} && return values
    all(isfinite, values) || throw(ArgumentError("$name contains non-finite values"))
    rounded = round.(Int, values)
    all(values .== rounded) || throw(ArgumentError("$name is not integral"))
    Vector{Int}(rounded)
end

function _typed_vector(values, ::Type{T}) where {T<:AbstractFloat}
    values isa Vector{T} ? values : Vector{T}(values)
end

function _dataset(parent::HDF5.File, path::AbstractString)
    object = parent[path]
    object isa HDF5.Dataset || throw(ArgumentError(
        "HDF5 object '$path' must be a dataset"))
    object
end

function _dataset(parent::HDF5.Group, name::AbstractString)
    object = parent[name]
    object isa HDF5.Dataset || throw(ArgumentError(
        "HDF5 object '$name' must be a dataset"))
    object
end

function _read_int_scalar(parent::HDF5.Group, name::AbstractString)
    value = HDF5.read(_dataset(parent, name))
    value isa Integer || throw(ArgumentError(
        "visible-sector field '$name' must be an integer scalar"))
    Int(value)
end

function _read_bool_scalar(parent::HDF5.Group, name::AbstractString)
    value = _read_int_scalar(parent, name)
    value in (0, 1) || throw(ArgumentError(
        "visible-sector field '$name' must be 0 or 1"))
    value == 1
end

function _load_visible_sector(file::HDF5.File, geometric::HDF5.Group,
        ::Type{T}, h11::Int, n_prime_divisors::Int) where {T<:AbstractFloat}
    haskey(geometric, "visible_sector") || return nothing
    object = geometric["visible_sector"]
    object isa HDF5.Group || throw(ArgumentError(
        "HDF5 object 'cytools/geometric/visible_sector' must be a group"))
    visible = object
    required = (
        "qcd_divisor_index", "qed_divisor_index", "qcd_image_index",
        "qed_image_index", "qcd_divisor_volume", "qed_divisor_volume",
        "qcd_charge", "qed_charge", "em_charge", "qed_instanton_index",
        "qed_log10_lambda4", "qcd_qed_intersection", "qcd_invariant",
        "qed_invariant")
    all(haskey(visible, name) for name in required) || throw(ArgumentError(
        "visible-sector group is missing one or more required fields"))

    qcd_index = _read_int_scalar(visible, "qcd_divisor_index") + 1
    qed_index = _read_int_scalar(visible, "qed_divisor_index") + 1
    qcd_image = _read_int_scalar(visible, "qcd_image_index") + 1
    qed_image = _read_int_scalar(visible, "qed_image_index") + 1
    all(1 <= index <= n_prime_divisors for index in
        (qcd_index, qed_index, qcd_image, qed_image)) || throw(ArgumentError(
        "visible-sector divisor indices are outside the prime-divisor list"))

    qcd_volume = T(HDF5.read(_dataset(visible, "qcd_divisor_volume")))
    qed_volume = T(HDF5.read(_dataset(visible, "qed_divisor_volume")))
    all(isfinite, (qcd_volume, qed_volume)) &&
        all(>(zero(T)), (qcd_volume, qed_volume)) || throw(ArgumentError(
        "visible-sector divisor volumes must be positive and finite"))

    qcd_charge = _integer_vector(HDF5.read(_dataset(visible, "qcd_charge")),
        "visible_sector/qcd_charge")
    qed_charge = _integer_vector(HDF5.read(_dataset(visible, "qed_charge")),
        "visible_sector/qed_charge")
    em_charge = _integer_vector(HDF5.read(_dataset(visible, "em_charge")),
        "visible_sector/em_charge")
    all(length(charge) == h11 for charge in (qcd_charge, qed_charge, em_charge)) ||
        throw(DimensionMismatch("visible-sector charge vectors must have h11 entries"))
    em_charge == qed_charge || throw(ArgumentError(
        "visible-sector EM charge must match the selected QED divisor charge"))

    qed_instanton_index = _read_int_scalar(visible, "qed_instanton_index") + 1
    qed_instanton_index > 0 || throw(ArgumentError(
        "visible-sector QED instanton index must be non-negative"))
    qed_log10_lambda4 = T(HDF5.read(_dataset(visible, "qed_log10_lambda4")))
    isfinite(qed_log10_lambda4) || throw(ArgumentError(
        "visible-sector QED instanton scale must be finite"))
    VisibleSectorAssignment{T}(qcd_index, qed_index, qcd_image, qed_image,
        qcd_volume, qed_volume, qcd_charge, qed_charge, em_charge,
        qed_instanton_index, qed_log10_lambda4,
        _read_bool_scalar(visible, "qcd_qed_intersection"),
        _read_bool_scalar(visible, "qcd_invariant"),
        _read_bool_scalar(visible, "qed_invariant"), :intersecting_d7)
end

function _read_integer_float_dataset(dataset::HDF5.Dataset,
        ::Type{T}, path::AbstractString) where {T<:AbstractFloat}
    rows, columns = size(dataset)
    output = Matrix{Int}(undef, rows, columns)
    chunk_columns = min(columns, 4096)
    buffer = Matrix{T}(undef, rows, chunk_columns)
    for first_column in 1:chunk_columns:columns
        last_column = min(columns, first_column + chunk_columns - 1)
        width = last_column - first_column + 1
        chunk = @view buffer[:, 1:width]
        HDF5.copyto!(chunk, dataset, :, first_column:last_column)
        @inbounds for column in 1:width, row in 1:rows
            value = chunk[row, column]
            isfinite(value) || throw(ArgumentError(
                "$path contains non-finite values"))
            rounded = round(Int, value)
            value == rounded || throw(ArgumentError(
                "$path is not integral"))
            output[row, first_column + column - 1] = rounded
        end
    end
    output
end

function _read_integer_dataset(file::HDF5.File, path::AbstractString)
    dataset = _dataset(file, path)
    stored_type = eltype(dataset)
    stored_type <: Integer && return HDF5.read(dataset)
    stored_type === Float64 &&
        return _read_integer_float_dataset(dataset, Float64, path)
    stored_type === Float32 &&
        return _read_integer_float_dataset(dataset, Float32, path)
    _integer_matrix(HDF5.read(dataset), path)
end

function _normalise_potential(Q, L, ::Type{T}) where {T<:AbstractFloat}
    q = _integer_matrix(Q, "Q")
    l = Matrix{T}(L)
    size(l, 1) == 2 || throw(DimensionMismatch(
        "L must have package shape 2 × N; transposed L data are not accepted"))
    size(q, 2) == size(l, 2) ||
        throw(DimensionMismatch(
            "Q must have package shape h11 × N and match L's instanton count; " *
            "transposed Q data are not accepted"))
    coefficient = @view l[1, :]
    exponent = @view l[2, :]
    all(isfinite, coefficient) || throw(ArgumentError("L coefficients are non-finite"))
    all(isfinite, exponent) || throw(ArgumentError("L exponents are non-finite"))
    all(!iszero, coefficient) ||
        throw(ArgumentError("L contains zero instanton coefficients"))
    signs = Int[sign(x) for x in coefficient]
    scales = Vector{T}(undef, length(coefficient))
    # CYAxiverse's stored L convention is (sign/mantissa, log10 scale).
    # The first row is normally ±1; retain its sign but rank by the stored
    # second row so this reader agrees with the package's existing spectrum
    # routines.
    scales .= exponent
    all(isfinite, scales) || throw(ArgumentError("computed instanton scales are non-finite"))
    order = sortperm(scales; rev=true, alg=MergeSort)
    InstantonData(q, scales[order], signs[order], order)
end

function _load_potential(file::HDF5.File, ::Type{T}) where {T<:AbstractFloat}
    Q = _read_integer_dataset(file, "cytools/potential/Q")
    L = HDF5.read(file, "cytools/potential/L")
    _normalise_potential(Q, L, T)
end

"""Read local instanton data and convert `L` to log Λ⁴."""
function load_instanton_data(path::AbstractString; T::Type{<:AbstractFloat}=Float64)
    isfile(path) || throw(ArgumentError("geometry file does not exist: $path"))
    h5open(path, "r") do file
        _load_potential(file, T)
    end
end

function _index_from_path(path::AbstractString)
    pieces = splitpath(normpath(path))
    length(pieces) >= 4 || throw(ArgumentError(
        "cannot infer GeometryIndex from path '$path'"))
    cy_name, np_name, h_name = pieces[end - 1], pieces[end - 2], pieces[end - 3]
    h_match = match(r"^h11_(\d+)$", h_name)
    np_match = match(r"^np_(\d+)$", np_name)
    cy_match = match(r"^cy_(\d+)$", cy_name)
    if h_match === nothing || np_match === nothing || cy_match === nothing
        throw(ArgumentError("cannot infer GeometryIndex from path '$path'"))
    end
    h_value = h_match.captures[1]
    np_value = np_match.captures[1]
    cy_value = cy_match.captures[1]
    h_value === nothing && throw(ArgumentError("missing h11 path component"))
    np_value === nothing && throw(ArgumentError("missing polytope path component"))
    cy_value === nothing && throw(ArgumentError("missing triangulation path component"))
    GeometryIndex(parse(Int, h_value), parse(Int, np_value), parse(Int, cy_value))
end

"""Return the local `cyax.h5` path corresponding to a geometry index."""
function geometry_path(index::GeometryIndex; data_dir=nothing)
    root = resolve_data_dir(data_dir)
    joinpath(root, "h11_$(lpad(index.h11, 3, '0'))",
        "np_$(lpad(index.polytope, 7, '0'))",
        "cy_$(lpad(index.frst, 7, '0'))", "cyax.h5")
end

function _required_geometry_fields(file::HDF5.File)
    all(haskey(file, "cytools/geometric/$field") for field in
        ("tip", "divisor_volumes", "CY_volume", "Kinv", "effective_cone",
         "prime_divisor_volumes", "prime_toric_divisors"))
end

function _load_geometry(file::HDF5.File, path::AbstractString,
        index::GeometryIndex, ::Type{T}) where {T<:AbstractFloat}
    _required_geometry_fields(file) || throw(ArgumentError(
        "geometry file lacks the complete local axion--photon fields: $path"))
    geometric_object = file["cytools/geometric"]
    geometric_object isa HDF5.Group || throw(ArgumentError(
        "HDF5 object 'cytools/geometric' must be a group"))
    geometric = geometric_object
    tip = _typed_vector(HDF5.read(_dataset(geometric, "tip")), T)
    divisor_volumes = _typed_vector(
        HDF5.read(_dataset(geometric, "divisor_volumes")), T)
    cy_volume = T(HDF5.read(_dataset(geometric, "CY_volume")))
    kinv_raw = Matrix{T}(HDF5.read(_dataset(geometric, "Kinv")))
    direct = _read_integer_dataset(file, "cytools/geometric/effective_cone")
    h11 = index.h11
    size(direct, 1) == h11 || throw(DimensionMismatch(
        "effective_cone must have package shape h11 × N; transposed " *
        "direct-charge data are not accepted"))
    direct_volumes = _typed_vector(
        HDF5.read(_dataset(geometric, "prime_divisor_volumes")), T)
    labels = _integer_vector(HDF5.read(
        _dataset(geometric, "prime_toric_divisors")),
        "prime_toric_divisors")
    size(direct, 2) == length(direct_volumes) == length(labels) ||
        throw(DimensionMismatch("direct divisor fields have inconsistent lengths"))
    size(kinv_raw) == (h11, h11) ||
        throw(DimensionMismatch("Kinv has the wrong shape"))
    length(tip) == h11 && length(divisor_volumes) == h11 ||
        throw(DimensionMismatch("stored geometry volumes have the wrong length"))
    all(isfinite, kinv_raw) || throw(ArgumentError("Kinv is non-finite"))
    all(isfinite, tip) && all(isfinite, divisor_volumes) ||
        throw(ArgumentError("stored geometry coordinates are non-finite"))
    all(isfinite, direct_volumes) && all(>(zero(T)), direct_volumes) ||
        throw(ArgumentError("direct divisor volumes must be positive"))
    isfinite(cy_volume) && cy_volume > zero(T) ||
        throw(ArgumentError("CY volume must be positive and finite"))
    kinv = copy(kinv_raw)
    @inbounds for column in axes(kinv, 2), row in 1:(column - 1)
        value = (kinv[row, column] + kinv[column, row]) / T(2)
        kinv[row, column] = value
        kinv[column, row] = value
    end
    cholesky(Symmetric(kinv))
    visible_sector = _load_visible_sector(file, geometric, T, h11, length(labels))
    GeometryInputs{T}(normpath(abspath(path)), index, tip, divisor_volumes,
        cy_volume, kinv, direct, direct_volumes, labels, visible_sector)
end

function _load_geometry(path::AbstractString, index::GeometryIndex,
        ::Type{T}) where {T<:AbstractFloat}
    isfile(path) || throw(ArgumentError("geometry file does not exist: $path"))
    h5open(path, "r") do file
        _load_geometry(file, path, index, T)
    end
end

"""Read geometry metadata and direct divisor charges from a local file."""
function load_geometry_inputs(path::AbstractString;
        T::Type{<:AbstractFloat}=Float64,
        index::Union{Nothing,GeometryIndex}=nothing)
    isfile(path) || throw(ArgumentError("geometry file does not exist: $path"))
    selected_index = index === nothing ? _index_from_path(path) : index
    _load_geometry(path, selected_index, T)
end

function load_geometry_inputs(index::GeometryIndex;
        data_dir=nothing, T::Type{<:AbstractFloat}=Float64)
    path = geometry_path(index; data_dir=data_dir)
    load_geometry_inputs(path; T=T, index=index)
end

function _is_complete_geometry(path::AbstractString)
    h5open(path, "r") do file
        haskey(file, "cytools/potential/Q") && haskey(file, "cytools/potential/L") &&
            _required_geometry_fields(file)
    end
end

"""Return deterministic local geometry indices for a bounded scan."""
function local_geometry_indices(data_dir::AbstractString;
        h11s=(15, 100, 200, 300), limit_per_h11::Integer=2,
        require_complete::Bool=true)
    limit_per_h11 > 0 || throw(ArgumentError("limit_per_h11 must be positive"))
    root = normpath(abspath(data_dir))
    isdir(root) || throw(ArgumentError("local data directory does not exist: $root"))
    output = GeometryIndex{Int}[]
    for h11_value in h11s
        h11 = Int(h11_value)
        hdir = joinpath(root, "h11_$(lpad(h11, 3, '0'))")
        isdir(hdir) || continue
        found = 0
        for npdir in sort(readdir(hdir; join=true))
            isdir(npdir) || continue
            startswith(basename(npdir), "np_") || continue
            for cydir in sort(readdir(npdir; join=true))
                isdir(cydir) || continue
                startswith(basename(cydir), "cy_") || continue
                path = joinpath(cydir, "cyax.h5")
                isfile(path) || continue
                require_complete && !_is_complete_geometry(path) && continue
                push!(output, _index_from_path(path))
                found += 1
                found == limit_per_h11 && break
            end
            found == limit_per_h11 && break
        end
    end
    output
end

function local_geometry_indices(; data_dir=nothing,
        h11s=(15, 100, 200, 300), limit_per_h11::Integer=2,
        require_complete::Bool=true)
    local_geometry_indices(resolve_data_dir(data_dir); h11s=h11s,
        limit_per_h11=limit_per_h11, require_complete=require_complete)
end

function _rank_state_append!(basis::Matrix{Int64}, active::BitVector,
        row::AbstractVector{<:Integer}, prime::Int, work::Vector{Int64})
    n = length(row)
    length(work) == n || throw(DimensionMismatch("rank workspace has the wrong length"))
    for i in 1:n
        work[i] = mod(Int64(row[i]), Int64(prime))
    end
    for pivot in 1:n
        active[pivot] || continue
        factor = work[pivot]
        factor == 0 && continue
        @inbounds for column in pivot:n
            work[column] = mod(work[column] - factor * basis[pivot, column], prime)
        end
    end
    pivot = findfirst(value -> !iszero(value), work)
    pivot === nothing && return false
    inverse = invmod(work[pivot], prime)
    @inbounds for column in pivot:n
        work[column] = mod(work[column] * inverse, prime)
    end
    @inbounds for old_pivot in 1:n
        active[old_pivot] || continue
        factor = basis[old_pivot, pivot]
        factor == 0 && continue
        for column in pivot:n
            basis[old_pivot, column] =
                mod(basis[old_pivot, column] - factor * work[column], prime)
        end
    end
    basis[pivot, :] .= work
    active[pivot] = true
    true
end

function _select_independent_terms(potential::InstantonData,
        h11::Int; rank_primes=(1_000_003, 1_000_033))
    size(potential.Q, 1) == h11 || throw(DimensionMismatch(
        "potential charge matrix does not match Kinv"))
    primes = Int[Int(p) for p in rank_primes]
    isempty(primes) && throw(ArgumentError("at least one rank prime is required"))
    bases = [zeros(Int64, h11, h11) for _ in primes]
    active = [falses(h11) for _ in primes]
    workspaces = [zeros(Int64, h11) for _ in primes]
    selected = Int[]
    dependent = Int[]
    for source in eachindex(potential.source_indices)
        column = potential.source_indices[source]
        independent = false
        for prime_index in eachindex(primes)
            independent |= _rank_state_append!(bases[prime_index],
                active[prime_index], @view(potential.Q[:, column]),
                primes[prime_index], workspaces[prime_index])
        end
        if independent
            push!(selected, column)
            if length(selected) == h11
                append!(dependent, potential.source_indices[(source + 1):end])
                break
            end
        else
            push!(dependent, column)
        end
    end
    length(selected) == h11 || throw(ArgumentError(
        "leading charges have rank $(length(selected)); expected h11=$h11"))
    selected, dependent
end

function _canonical_frame(Q_reduced::Matrix{Int}, Kinv::Matrix{T}) where {T<:AbstractFloat}
    n = size(Q_reduced, 1)
    size(Q_reduced, 2) == n || throw(DimensionMismatch(
        "selected charge matrix must be square"))
    factor = cholesky(Symmetric(Kinv))
    lower = factor.L
    Q_float = Matrix{T}(Q_reduced)
    charge_columns = Matrix{T}(undef, n, n)
    mul!(charge_columns, transpose(lower), Q_float)
    qfactor = qr(charge_columns)
    orthogonal = Matrix{T}(qfactor.Q)
    upper = Matrix{T}(qfactor.R)
    q = upper
    rotation = orthogonal
    for i in 1:n
        q[i, i] < zero(T) || continue
        rotation[:, i] .*= -one(T)
        q[i, :] .*= -one(T)
    end
    theta = lower * rotation
    q, theta, factor
end

"""Construct the leading axion charge frame and log-scale masses."""
function leading_hierarchy(potential::InstantonData, kinv::AbstractMatrix{<:Real};
        T::Type{<:AbstractFloat}=Float64,
        signed_scale_policy::Symbol=:require_positive,
        rank_primes=(1_000_003, 1_000_033),
        m_planck_GeV::Real=M_PLANCK_GEV)
    size(kinv, 1) == size(kinv, 2) || throw(DimensionMismatch("Kinv must be square"))
    h11 = size(kinv, 1)
    selected, dependent = _select_independent_terms(potential, h11;
        rank_primes=rank_primes)
    position_by_source = Dict{Int,Int}(
        source => position for (position, source) in enumerate(potential.source_indices))
    selected_positions = [position_by_source[source] for source in selected]
    signs = potential.coefficient_signs[selected_positions]
    selected_scales = potential.log10_lambda4[selected_positions]
    signed_scale_policy in (:require_positive, :absolute) || throw(ArgumentError(
        "signed_scale_policy must be :require_positive or :absolute"))
    signed_scale_policy == :require_positive && any(signs .<= 0) && throw(ArgumentError(
        "selected leading terms include non-positive coefficients; use " *
        "signed_scale_policy=:absolute only for an explicitly adapted run"))
    Kinv = Matrix{T}(kinv)
    Kinv = (Kinv + transpose(Kinv)) / T(2)
    Q_reduced = Matrix{Int}(potential.Q[:, selected])
    q, theta, factor = _canonical_frame(Q_reduced, Kinv)
    diag_q = diag(q)
    all(>(zero(T)), diag_q) || throw(ArgumentError(
        "canonical charge frame has a non-positive diagonal"))
    planck = T(m_planck_GeV)
    planck > zero(T) && isfinite(planck) ||
        throw(ArgumentError("m_planck_GeV must be positive and finite"))
    log_m_planck = log10(planck)
    log10_f = Vector{T}(undef, h11)
    log10_mass = Vector{T}(undef, h11)
    for i in 1:h11
        log10_f[i] = log_m_planck - log10(T(2π)) - log10(diag_q[i])
        log10_mass[i] = T(0.5) * T(selected_scales[i]) +
            log_m_planck + T(9) + log10(T(2π)) + log10(diag_q[i])
    end
    identity = Matrix{T}(I, h11, h11)
    metric_residual = norm(transpose(theta) * (factor \ theta) - identity, Inf)
    scale = max(norm(q, Inf), one(T))
    triangular_residual = norm(tril(q, -1), Inf) / scale
    LeadingAxionHierarchy{T}(selected, dependent, Q_reduced,
        T.(selected_scales), signs, q, theta, log10_f,
        log10_mass, planck, triangular_residual, metric_residual)
end

function leading_hierarchy(path::AbstractString; T::Type{<:AbstractFloat}=Float64,
        kwargs...)
    potential_data = load_instanton_data(path; T=T)
    geometry_data = load_geometry_inputs(path; T=T)
    leading_hierarchy(potential_data, geometry_data.kinv; T=T, kwargs...)
end

"""Return the leading-hierarchy mixing matrix Θ."""
function mixing_matrix(result::LeadingAxionHierarchy{T}) where {T<:AbstractFloat}
    n = length(result.log10_lambda4)
    theta = zeros(T, n, n)
    for a in 1:n, b in 1:n
        if b <= a
            theta[a, b] = result.q[b, a] / result.q[b, b]
        else
            theta[a, b] = -T(10)^(result.log10_lambda4[b] -
                result.log10_lambda4[a]) * result.q[a, b] / result.q[a, a]
        end
    end
    theta
end

function _log10_abs(value::T) where {T<:AbstractFloat}
    iszero(value) ? T(-Inf) : log10(abs(value))
end

function _exp10_or_zero(value::T) where {T<:AbstractFloat}
    value < log10(floatmin(T)) ? zero(T) : T(10)^value
end

"""
    qed_instanton_log10_threshold_eV(geometry)

Compute the QED light threshold from the selected divisor's Euclidean-D3
instanton term. The result is `log10(m_QED/eV)` and therefore remains usable
when the threshold is below the representable range of a regular Float64.
The geometry must contain an `intersecting_d7` visible-sector assignment.
"""
function qed_instanton_log10_threshold_eV(
        geometry::GeometryInputs{T}; m_planck_GeV::Real=M_PLANCK_GEV) where {T<:AbstractFloat}
    assignment = geometry.visible_sector
    assignment === nothing && throw(ArgumentError(
        "geometry has no orientifold-compatible QCD/QED divisor assignment"))
    planck = T(m_planck_GeV)
    planck > zero(T) && isfinite(planck) || throw(ArgumentError(
        "m_planck_GeV must be positive and finite"))
    charge = Vector{T}(undef, length(assignment.qed_charge))
    transformed = similar(charge)
    @inbounds for index in eachindex(charge)
        charge[index] = T(assignment.qed_charge[index])
    end
    mul!(transformed, geometry.kinv, charge)
    norm_squared = dot(charge, transformed)
    norm_squared > zero(T) && isfinite(norm_squared) || throw(ArgumentError(
        "QED divisor charge has a non-positive kinetic norm"))
    T(0.5) * assignment.qed_log10_lambda4 + log10(planck) + T(9) +
        log10(T(2π)) + T(0.5) * log10(norm_squared)
end

"""Compute the QED light threshold in eV, returning zero only on underflow."""
function qed_instanton_threshold_eV(geometry::GeometryInputs{T}; kwargs...) where {T<:AbstractFloat}
    _exp10_or_zero(qed_instanton_log10_threshold_eV(geometry; kwargs...))
end

"""Compute photon couplings, the QED-threshold proxy, and leading widths."""
function photon_observables(result::LeadingAxionHierarchy{T},
        em_charge::AbstractVector{<:Real};
        alpha_em::Real=ALPHA_EM,
        light_threshold_eV::Real=ELECTRON_MASS_EV,
        light_threshold_log10_eV::Union{Nothing,Real}=nothing) where {T<:AbstractFloat}
    n = size(result.Q_reduced, 1)
    length(em_charge) == n || throw(DimensionMismatch(
        "EM charge vector must have h11 entries"))
    em = T.(em_charge)
    n_em = Matrix{T}(result.Q_reduced) \ em
    charge_residual = norm(result.Q_reduced * n_em - em, Inf) /
        max(norm(em, Inf), one(T))
    theta = mixing_matrix(result)
    Cgamma = vec(transpose(n_em) * theta)
    log10_g = Vector{T}(undef, n)
    log10_f = result.log10_f_GeV
    for i in 1:n
        log10_g[i] = log10(T(alpha_em) / T(2π)) - log10_f[i] +
            _log10_abs(Cgamma[i])
    end
    threshold, log_threshold = if light_threshold_log10_eV === nothing
        value = T(light_threshold_eV)
        value > zero(T) && isfinite(value) || throw(ArgumentError(
            "light threshold must be positive and finite"))
        value, log10(value)
    else
        value = T(light_threshold_log10_eV)
        isfinite(value) || throw(ArgumentError(
            "log10 light threshold must be finite"))
        _exp10_or_zero(value), value
    end
    log10_g_effective = copy(log10_g)
    light_count = 0
    for i in 1:n
        if result.log10_mass_eV[i] <= log_threshold
            light_count += 1
            log10_g_effective[i] += T(2) *
                (result.log10_mass_eV[i] - log_threshold)
        end
    end
    log10_mass_GeV = result.log10_mass_eV .- T(9)
    log10_photon_width = T(3) .* log10_mass_GeV .+ T(2) .* log10_g_effective .-
        log10(T(64π))
    log10_quartic_width = fill(T(-Inf), n)
    log_m_planck = log10(result.m_planck_GeV)
    for a in 1:(n - 1)
        b = a + 1
        log_lambda = result.log10_lambda4[b] + T(4) * log_m_planck -
            result.log10_f_GeV[a] - T(3) * result.log10_f_GeV[b] +
            _log10_abs(theta[b, a])
        log10_quartic_width[a] = T(2) * log_lambda + log10_mass_GeV[a] -
            log10(T(128π^3))
    end
    AxionPhotonObservables{T}(n_em, theta, Cgamma, log10_g,
        log10_g_effective, log10_photon_width, log10_quartic_width,
        threshold, log_threshold, light_count, charge_residual)
end

function _em_selection(geometry::GeometryInputs, em_divisor_index)
    assignment = geometry.visible_sector
    if em_divisor_index === nothing && assignment !== nothing
        return assignment.qed_divisor_index, assignment.em_charge,
            assignment.qed_divisor_volume, :visible_sector_qed
    end
    em_index = em_divisor_index === nothing ?
        findmin(geometry.direct_divisor_volumes)[2] : Int(em_divisor_index)
    1 <= em_index <= size(geometry.direct_charges, 2) ||
        throw(BoundsError(geometry.direct_charges, (:, em_index)))
    em_index, geometry.direct_charges[:, em_index],
        geometry.direct_divisor_volumes[em_index], :direct_effective_cone
end

function _validate_visible_sector_potential(
        geometry::GeometryInputs, potential::InstantonData)
    assignment = geometry.visible_sector
    assignment === nothing && throw(ArgumentError(
        "visible-sector QED metadata are unavailable"))
    source = assignment.qed_instanton_index
    1 <= source <= size(potential.Q, 2) || throw(ArgumentError(
        "visible-sector QED instanton index is outside the stored potential"))
    all(index -> potential.Q[index, source] == assignment.qed_charge[index],
        eachindex(assignment.qed_charge)) || throw(ArgumentError(
        "visible-sector QED charge does not match its stored instanton column"))
    sorted_position = findfirst(==(source), potential.source_indices)
    sorted_position === nothing && throw(ArgumentError(
        "visible-sector QED instanton index is absent from the sorted potential"))
    isapprox(potential.log10_lambda4[sorted_position],
        assignment.qed_log10_lambda4; rtol=zero(eltype(potential.log10_lambda4)),
        atol=8eps(eltype(potential.log10_lambda4))) || throw(ArgumentError(
        "visible-sector QED instanton scale does not match its stored potential column"))
    nothing
end

function _run_local_scan(path::AbstractString;
        T::Type{<:AbstractFloat}=Float64,
        em_divisor_index::Union{Nothing,Integer}=nothing,
        light_threshold_eV::Real=ELECTRON_MASS_EV,
        qed_threshold_policy::Symbol=:electron_proxy,
        signed_scale_policy::Symbol=:require_positive)
    isfile(path) || throw(ArgumentError("geometry file does not exist: $path"))
    index = _index_from_path(path)
    geometry_data, potential_data = h5open(path, "r") do file
        _load_geometry(file, path, index, T), _load_potential(file, T)
    end
    hierarchy_data = leading_hierarchy(potential_data, geometry_data.kinv; T=T,
        signed_scale_policy=signed_scale_policy)
    qed_threshold_policy in (:electron_proxy, :divisor_instanton) || throw(ArgumentError(
        "qed_threshold_policy must be :electron_proxy or :divisor_instanton"))
    em_index, em_charge, em_volume, em_source = _em_selection(
        geometry_data, em_divisor_index)
    threshold, threshold_log = if qed_threshold_policy == :divisor_instanton
        geometry_data.visible_sector === nothing && throw(ArgumentError(
            "qed_threshold_policy=:divisor_instanton requires visible-sector metadata"))
        _validate_visible_sector_potential(geometry_data, potential_data)
        log_threshold = qed_instanton_log10_threshold_eV(geometry_data;
            m_planck_GeV=hierarchy_data.m_planck_GeV)
        _exp10_or_zero(log_threshold), log_threshold
    else
        value = T(light_threshold_eV)
        value > zero(T) && isfinite(value) || throw(ArgumentError(
            "light_threshold_eV must be positive and finite"))
        value, log10(value)
    end
    photons = photon_observables(hierarchy_data, em_charge;
        light_threshold_eV=threshold, light_threshold_log10_eV=threshold_log)
    status = qed_threshold_policy == :divisor_instanton ?
        :visible_sector_instanton_threshold :
        (signed_scale_policy == :absolute ? :adapted_absolute_scale :
         :adapted_local_geometry)
    AxionPhotonResult{T}(geometry_data, potential_data, hierarchy_data,
        photons, em_index, em_volume, em_source, qed_threshold_policy, status)
end

"""Run a bounded local axion--photon scan on deterministic h11 slices."""
function run_local_scan(; data_dir=nothing, h11s=(15, 100, 200, 300),
        limit_per_h11::Integer=2, require_complete::Bool=true,
        T::Type{<:AbstractFloat}=Float64,
        em_divisor_index::Union{Nothing,Integer}=nothing,
        light_threshold_eV::Real=ELECTRON_MASS_EV,
        qed_threshold_policy::Symbol=:electron_proxy,
        signed_scale_policy::Symbol=:require_positive)
    root = resolve_data_dir(data_dir)
    indices = local_geometry_indices(root; h11s=h11s,
        limit_per_h11=limit_per_h11, require_complete=require_complete)
    isempty(indices) && throw(ArgumentError(
        "no matching complete local geometries were found in $root"))
    results = AxionPhotonResult{T}[]
    for index in indices
        push!(results, _run_local_scan(geometry_path(index; data_dir=root);
            T=T, em_divisor_index=em_divisor_index,
            light_threshold_eV=light_threshold_eV,
            qed_threshold_policy=qed_threshold_policy,
            signed_scale_policy=signed_scale_policy))
    end
    results
end

function _max_log10_abs(values::AbstractVector{T}) where {T<:AbstractFloat}
    maximum((_log10_abs(value) for value in values); init=T(-Inf))
end

"""Write one compact CSV summary; detailed arrays remain in Julia."""
function write_scan_csv(path::AbstractString, results::AbstractVector{<:AxionPhotonResult})
    parent = dirname(normpath(path))
    isdir(parent) || throw(ArgumentError("CSV parent directory does not exist: $parent"))
    fields = ("path", "h11", "polytope", "frst", "status", "n_input",
        "n_selected", "n_dependent", "em_divisor_index", "em_divisor_volume",
        "em_charge_source", "light_threshold_policy", "light_threshold_eV",
        "log10_light_threshold_eV", "light_mode_count", "max_log10_g_GeVinv",
        "max_log10_Cgamma",
        "triangular_residual", "metric_residual", "charge_residual")
    open(path, "w") do io
        println(io, join(fields, ','))
        for result in results
            g = result.geometry
            h = result.hierarchy
            p = result.photons
            values = (g.path, g.index.h11, g.index.polytope, g.index.frst,
                result.status, length(result.potential.source_indices),
                length(h.selected_indices), length(h.dependent_indices),
                result.em_divisor_index, result.em_divisor_volume,
                result.em_charge_source, result.light_threshold_policy,
                p.light_threshold_eV, p.log10_light_threshold_eV,
                p.light_mode_count, maximum(p.log10_g_GeVinv),
                _max_log10_abs(p.Cgamma), h.triangular_residual,
                h.metric_residual, p.charge_residual)
            println(io, join(string.(values), ','))
        end
    end
    normpath(abspath(path))
end

"""Compatibility names retained for existing local analyses."""
const GlimmersGeometry = GeometryInputs
const GlimmersPotential = InstantonData
const GlimmersHierarchy = LeadingAxionHierarchy
const GlimmersPhotonObservables = AxionPhotonObservables
const GlimmersPilotResult = AxionPhotonResult
const load_geometry = load_geometry_inputs
const load_potential = load_instanton_data
const hierarchy = leading_hierarchy
const run_local_pilot = run_local_scan
const write_pilot_csv = write_scan_csv

export VisibleSectorAssignment, GeometryInputs, InstantonData,
    LeadingAxionHierarchy, AxionPhotonObservables, AxionPhotonResult,
    load_geometry_inputs,
    load_instanton_data, geometry_path, local_geometry_indices,
    leading_hierarchy, mixing_matrix, photon_observables,
    qed_instanton_log10_threshold_eV, qed_instanton_threshold_eV,
    run_local_scan, write_scan_csv

end
