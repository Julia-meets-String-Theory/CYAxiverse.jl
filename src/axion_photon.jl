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
using SHA
import Nemo

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

"""Exact ordered rational-rank evidence for leading-charge selection.

The certificate records the source-column order, the selected and dependent
columns, the exact rank after every ordered prefix, and the exact determinant
of the selected square charge matrix. Recompute the prefix ranks from the
stored integer charge matrix to independently replay the selection.
"""
struct RationalRankCertificate
    algorithm::String
    matrix_shape::NTuple{2,Int}
    ordered_source_indices::Vector{Int}
    selected_indices::Vector{Int}
    dependent_indices::Vector{Int}
    prefix_ranks::Vector{Int}
    selected_determinant::BigInt
end

"""Leading-charge hierarchy in the canonical Gram--Schmidt frame.

`Q_reduced` and `q` retain one selected instanton charge vector per column,
with the same `h11 × h11` column convention as the input `Q`. The canonical
charge matrix is `q = X' * Q_reduced` and is upper triangular. The exact
ordered rational-rank evidence is retained in `rank_certificate`.
"""
struct LeadingAxionHierarchy{T<:AbstractFloat}
    selected_indices::Vector{Int}
    dependent_indices::Vector{Int}
    rank_certificate::RationalRankCertificate
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

"""Configuration identity for one persisted axion--photon calculation.

Keep the threshold as a typed textual value so the persisted contract records
the caller's configuration without introducing a scientific rounding or
normalisation rule.
"""
struct AxionPhotonConfiguration
    em_divisor_index::Union{Nothing,Int}
    light_threshold_eV::String
    light_threshold_eV_effective::String
    qed_threshold_policy::Symbol
    signed_scale_policy::Symbol
    precision::String
end

"""Identity binding a result to its geometry, inputs, and configuration."""
struct AxionPhotonIdentity
    schema_version::Int
    geometry_index::GeometryIndex{Int}
    geometry_digest::String
    potential_digest::String
    geometry_snapshot_digest::String
    potential_snapshot_digest::String
    configuration::AxionPhotonConfiguration
    configuration_digest::String
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
    identity::AxionPhotonIdentity
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

const _AXION_PHOTON_IDENTITY_SCHEMA_VERSION = 1

function _identity_write_bytes!(io::IO, bytes::AbstractVector{UInt8})
    write(io, bytes)
    nothing
end

function _identity_write_bytes!(context::SHA.SHA256_CTX,
        bytes::AbstractVector{UInt8})
    SHA.update!(context, bytes)
    nothing
end

function _identity_write_length!(sink, length::Integer)
    value = UInt64(length)
    bytes = UInt8[(value >> shift) & 0xff for shift in 0:8:56]
    _identity_write_bytes!(sink, bytes)
end

function _identity_write_string(sink, value::AbstractString)
    bytes = Vector{UInt8}(codeunits(value))
    _identity_write_length!(sink, length(bytes))
    _identity_write_bytes!(sink, bytes)
    nothing
end

function _identity_write_scalar(sink, value)
    if value isa AbstractFloat
        _identity_write_string(sink, "float")
        _identity_write_string(sink, string(typeof(value)))
        _identity_write_string(sink, repr(value))
    elseif value isa Integer
        _identity_write_string(sink, "integer")
        _identity_write_string(sink, string(typeof(value)))
        _identity_write_string(sink, string(value))
    elseif value isa Bool
        _identity_write_string(sink, "bool")
        _identity_write_string(sink, value ? "true" : "false")
    elseif value isa AbstractString
        _identity_write_string(sink, "string")
        _identity_write_string(sink, value)
    elseif value === nothing
        _identity_write_string(sink, "nothing")
    else
        _identity_write_string(sink, "repr")
        _identity_write_string(sink, string(typeof(value)))
        _identity_write_string(sink, repr(value))
    end
    nothing
end

function _identity_write_value(sink, value)
    if value isa AbstractArray
        _identity_write_string(sink, "array")
        _identity_write_string(sink, string(eltype(value)))
        _identity_write_string(sink, string(ndims(value)))
        for dimension in size(value)
            _identity_write_string(sink, string(dimension))
        end
        for entry in value
            _identity_write_scalar(sink, entry)
        end
    else
        _identity_write_scalar(sink, value)
    end
    nothing
end

function _identity_write_group!(sink, group::HDF5.Group, label::AbstractString)
    _identity_write_string(sink, "group")
    _identity_write_string(sink, label)
    names = sort!(String.(collect(keys(group))))
    _identity_write_string(sink, string(length(names)))
    for name in names
        object = group[name]
        try
            if object isa HDF5.Group
                _identity_write_group!(sink, object, name)
            elseif object isa HDF5.Dataset
                _identity_write_string(sink, "dataset")
                _identity_write_string(sink, name)
                _identity_write_string(sink, string(eltype(object)))
                _identity_write_string(sink, string(ndims(object)))
                for dimension in size(object)
                    _identity_write_string(sink, string(dimension))
                end
                _identity_write_value(sink, HDF5.read(object))
            else
                throw(ArgumentError("unsupported HDF5 object '$name'"))
            end
        finally
            close(object)
        end
    end
    nothing
end

function _identity_group_digest(group::HDF5.Group)
    context = SHA.SHA256_CTX()
    _identity_write_group!(context, group, "/")
    bytes2hex(SHA.digest!(context))
end

function _input_identity_digests(file::HDF5.File)
    geometric_object = file["cytools/geometric"]
    potential_object = file["cytools/potential"]
    geometric_object isa HDF5.Group || begin
        close(geometric_object)
        throw(ArgumentError("HDF5 object 'cytools/geometric' must be a group"))
    end
    potential_object isa HDF5.Group || begin
        close(geometric_object)
        close(potential_object)
        throw(ArgumentError("HDF5 object 'cytools/potential' must be a group"))
    end
    try
        geometry_digest = _identity_group_digest(geometric_object)
        potential_digest = _identity_group_digest(potential_object)
        geometry_digest, potential_digest
    finally
        close(geometric_object)
        close(potential_object)
    end
end

function _identity_write_named_value!(context::SHA.SHA256_CTX,
        name::AbstractString, value)
    _identity_write_string(context, name)
    _identity_write_value(context, value)
end

function _identity_write_visible_sector!(context::SHA.SHA256_CTX,
        assignment::Nothing)
    _identity_write_string(context, "visible_sector")
    _identity_write_scalar(context, nothing)
end

function _identity_write_visible_sector!(context::SHA.SHA256_CTX,
        assignment::VisibleSectorAssignment)
    _identity_write_string(context, "visible_sector")
    _identity_write_named_value!(context, "qcd_divisor_index",
        assignment.qcd_divisor_index)
    _identity_write_named_value!(context, "qed_divisor_index",
        assignment.qed_divisor_index)
    _identity_write_named_value!(context, "qcd_image_index", assignment.qcd_image_index)
    _identity_write_named_value!(context, "qed_image_index", assignment.qed_image_index)
    _identity_write_named_value!(context, "qcd_divisor_volume",
        assignment.qcd_divisor_volume)
    _identity_write_named_value!(context, "qed_divisor_volume",
        assignment.qed_divisor_volume)
    _identity_write_named_value!(context, "qcd_charge", assignment.qcd_charge)
    _identity_write_named_value!(context, "qed_charge", assignment.qed_charge)
    _identity_write_named_value!(context, "em_charge", assignment.em_charge)
    _identity_write_named_value!(context, "qed_instanton_index",
        assignment.qed_instanton_index)
    _identity_write_named_value!(context, "qed_log10_lambda4",
        assignment.qed_log10_lambda4)
    _identity_write_named_value!(context, "qcd_qed_intersection",
        assignment.qcd_qed_intersection)
    _identity_write_named_value!(context, "qcd_invariant", assignment.qcd_invariant)
    _identity_write_named_value!(context, "qed_invariant", assignment.qed_invariant)
    _identity_write_named_value!(context, "policy", String(assignment.policy))
end

function _geometry_snapshot_digest(geometry::GeometryInputs)
    context = SHA.SHA256_CTX()
    _identity_write_string(context, "axion-photon-geometry-snapshot-v1")
    _identity_write_named_value!(context, "index", (
        geometry.index.h11, geometry.index.polytope, geometry.index.frst))
    _identity_write_named_value!(context, "tip", geometry.tip)
    _identity_write_named_value!(context, "divisor_volumes", geometry.divisor_volumes)
    _identity_write_named_value!(context, "cy_volume", geometry.cy_volume)
    _identity_write_named_value!(context, "kinv", geometry.kinv)
    _identity_write_named_value!(context, "direct_charges", geometry.direct_charges)
    _identity_write_named_value!(context, "direct_divisor_volumes",
        geometry.direct_divisor_volumes)
    _identity_write_named_value!(context, "direct_labels", geometry.direct_labels)
    _identity_write_visible_sector!(context, geometry.visible_sector)
    bytes2hex(SHA.digest!(context))
end

function _potential_snapshot_digest(potential::InstantonData)
    context = SHA.SHA256_CTX()
    _identity_write_string(context, "axion-photon-potential-snapshot-v1")
    _identity_write_named_value!(context, "Q", potential.Q)
    _identity_write_named_value!(context, "log10_lambda4", potential.log10_lambda4)
    _identity_write_named_value!(context, "coefficient_signs", potential.coefficient_signs)
    _identity_write_named_value!(context, "source_indices", potential.source_indices)
    bytes2hex(SHA.digest!(context))
end

function _configuration_value(value)
    value === nothing && return "nothing"
    string(typeof(value), ":", repr(value))
end

function _configuration_selection(index::Union{Nothing,Int})
    index === nothing ? "auto" : string("index:", index)
end

function _configuration_digest(configuration::AxionPhotonConfiguration)
    context = SHA.SHA256_CTX()
    _identity_write_string(context, "axion-photon-configuration-v1")
    _identity_write_string(context,
        _configuration_selection(configuration.em_divisor_index))
    _identity_write_string(context, configuration.light_threshold_eV)
    _identity_write_string(context, configuration.light_threshold_eV_effective)
    _identity_write_string(context, String(configuration.qed_threshold_policy))
    _identity_write_string(context, String(configuration.signed_scale_policy))
    _identity_write_string(context, configuration.precision)
    bytes2hex(SHA.digest!(context))
end

function _configuration_precision_type(precision::AbstractString)
    precision == string(Float16) && return Float16
    precision == string(Float32) && return Float32
    precision == string(Float64) && return Float64
    precision == string(BigFloat) && return BigFloat
    throw(ArgumentError("unsupported persisted computation precision '$precision'"))
end

function _make_axion_photon_configuration(::Type{T}, em_divisor_index,
        light_threshold_eV, qed_threshold_policy::Symbol,
        signed_scale_policy::Symbol) where {T<:AbstractFloat}
    qed_threshold_policy in (:electron_proxy, :divisor_instanton) || throw(ArgumentError(
        "qed_threshold_policy must be :electron_proxy or :divisor_instanton"))
    signed_scale_policy in (:require_positive, :absolute) || throw(ArgumentError(
        "signed_scale_policy must be :require_positive or :absolute"))
    em_index = em_divisor_index === nothing ? nothing : Int(em_divisor_index)
    effective_threshold = qed_threshold_policy == :electron_proxy ?
        _configuration_value(T(light_threshold_eV)) : "unused"
    AxionPhotonConfiguration(em_index, _configuration_value(light_threshold_eV),
        effective_threshold,
        qed_threshold_policy, signed_scale_policy, string(T))
end

function _configuration_with_effective_threshold(
        configuration::AxionPhotonConfiguration, threshold::AbstractFloat)
    configuration.qed_threshold_policy == :electron_proxy || return configuration
    AxionPhotonConfiguration(configuration.em_divisor_index,
        configuration.light_threshold_eV, _configuration_value(threshold),
        configuration.qed_threshold_policy, configuration.signed_scale_policy,
        configuration.precision)
end

function _configuration_matches_request(stored::AxionPhotonConfiguration,
        requested::AxionPhotonConfiguration)
    stored.em_divisor_index == requested.em_divisor_index &&
        stored.light_threshold_eV == requested.light_threshold_eV &&
        stored.qed_threshold_policy == requested.qed_threshold_policy &&
        stored.signed_scale_policy == requested.signed_scale_policy &&
        stored.precision == requested.precision
end

function _make_axion_photon_identity(index::GeometryIndex{Int},
        geometry_digest::AbstractString, potential_digest::AbstractString,
        geometry::GeometryInputs, potential::InstantonData,
        configuration::AxionPhotonConfiguration)
    AxionPhotonIdentity(_AXION_PHOTON_IDENTITY_SCHEMA_VERSION, index,
        String(geometry_digest), String(potential_digest),
        _geometry_snapshot_digest(geometry), _potential_snapshot_digest(potential),
        configuration, _configuration_digest(configuration))
end

function _parse_configuration_selection(value::AbstractString)
    value == "auto" && return nothing
    startswith(value, "index:") || throw(ArgumentError(
        "invalid persisted EM-divisor selection"))
    parsed = try
        parse(Int, value[7:end])
    catch
        throw(ArgumentError("invalid persisted EM-divisor selection"))
    end
    parsed > 0 || throw(ArgumentError("persisted EM-divisor index must be positive"))
    parsed
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

"""Return a JSON-compatible payload for an exact rank certificate."""
function rank_certificate_payload(certificate::RationalRankCertificate)
    (algorithm=certificate.algorithm,
        matrix_shape=collect(certificate.matrix_shape),
        ordered_source_indices=copy(certificate.ordered_source_indices),
        selected_indices=copy(certificate.selected_indices),
        dependent_indices=copy(certificate.dependent_indices),
        prefix_ranks=copy(certificate.prefix_ranks),
        selected_determinant=string(certificate.selected_determinant))
end

const _RANK_SCREENING_PRIME = 1_000_003

function _modular_rank_state_append!(basis::Matrix{Int64}, active::BitVector,
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
        basis[pivot, column] = work[column]
    end
    active[pivot] = true
    true
end

function _exact_rank_state_append!(basis::Matrix{Rational{BigInt}}, active::BitVector,
        row::AbstractVector{<:Integer}, work::Vector{Rational{BigInt}})
    n = length(row)
    length(work) == n || throw(DimensionMismatch("rank workspace has the wrong length"))
    for i in 1:n
        work[i] = Rational{BigInt}(BigInt(row[i]), BigInt(1))
    end
    for pivot in 1:n
        active[pivot] || continue
        factor = work[pivot]
        factor == 0 && continue
        @inbounds for column in pivot:n
            work[column] -= factor * basis[pivot, column]
        end
    end
    pivot = findfirst(value -> !iszero(value), work)
    pivot === nothing && return false
    pivot_value = work[pivot]
    @inbounds for column in pivot:n
        work[column] /= pivot_value
        basis[pivot, column] = work[column]
    end
    active[pivot] = true
    true
end

function _exact_rank_columns(Q::Matrix{Int}, columns::AbstractVector{Int})
    isempty(columns) && return 0
    Int(Nemo.rank(Nemo.matrix(Nemo.ZZ, Matrix{Int}(Q[:, columns]))))
end

function _first_exactly_independent_pending(Q::Matrix{Int}, selected::Vector{Int},
        pending::Vector{Int})
    isempty(pending) && return 0
    selected_rank = length(selected)
    _exact_rank_columns(Q, vcat(selected, pending)) == selected_rank && return 0
    first = 1
    last = length(pending)
    while first < last
        midpoint = (first + last) ÷ 2
        prefix = vcat(selected, pending[1:midpoint])
        if _exact_rank_columns(Q, prefix) > selected_rank
            last = midpoint
        else
            first = midpoint + 1
        end
    end
    first
end

function _initialise_exact_rank_state(Q::Matrix{Int}, selected::Vector{Int})
    n = size(Q, 1)
    basis = zeros(Rational{BigInt}, n, n)
    active = falses(n)
    workspace = zeros(Rational{BigInt}, n)
    for column in selected
        _exact_rank_state_append!(basis, active, @view(Q[:, column]), workspace) ||
            throw(ArgumentError("selected columns are not exactly independent"))
    end
    basis, active, workspace
end

function _process_exact_columns!(selected::Vector{Int}, dependent::Vector{Int},
        prefix_ranks::Vector{Int}, Q::Matrix{Int}, positions, columns,
        basis::Matrix{Rational{BigInt}}, active::BitVector,
        workspace::Vector{Rational{BigInt}})
    for (position, column) in zip(positions, columns)
        if length(selected) == size(Q, 1)
            push!(dependent, column)
            prefix_ranks[position] = length(selected)
            continue
        end
        independent = _exact_rank_state_append!(basis, active,
            @view(Q[:, column]), workspace)
        if independent
            push!(selected, column)
        else
            push!(dependent, column)
        end
        prefix_ranks[position] = length(selected)
    end
end

function _exact_integer_determinant(matrix::AbstractMatrix{<:Integer})
    size(matrix, 1) == size(matrix, 2) || throw(ArgumentError(
        "exact integer determinant requires a square matrix"))
    BigInt(Nemo.det(Nemo.matrix(Nemo.ZZ, Matrix{Int}(matrix))))
end

function _validate_source_order(potential::InstantonData)
    ncolumns = size(potential.Q, 2)
    length(potential.source_indices) == ncolumns || throw(DimensionMismatch(
        "source_indices must contain one entry per charge column"))
    sort(potential.source_indices) == collect(1:ncolumns) || throw(ArgumentError(
        "source_indices must be a permutation of the charge-column indices"))
    nothing
end

function _select_independent_terms(potential::InstantonData, h11::Int)
    size(potential.Q, 1) == h11 || throw(DimensionMismatch(
        "potential charge matrix does not match Kinv"))
    h11 > 0 || throw(ArgumentError("h11 must be positive"))
    _validate_source_order(potential)
    ncolumns = size(potential.Q, 2)
    modular_basis = zeros(Int64, h11, h11)
    modular_active = falses(h11)
    modular_workspace = zeros(Int64, h11)
    modular_rank = 0
    selected = Int[]
    dependent = Int[]
    prefix_ranks = Vector{Int}(undef, ncolumns)
    pending_positions = Int[]
    pending_columns = Int[]
    exact_mode = false
    exact_basis = Matrix{Rational{BigInt}}(undef, 0, 0)
    exact_active = BitVector()
    exact_workspace = Vector{Rational{BigInt}}()
    for source_position in eachindex(potential.source_indices)
        column = potential.source_indices[source_position]
        if length(selected) == h11
            push!(dependent, column)
            prefix_ranks[source_position] = h11
            continue
        end

        if exact_mode
            independent = _exact_rank_state_append!(exact_basis, exact_active,
                @view(potential.Q[:, column]), exact_workspace)
            if independent
                push!(selected, column)
            else
                push!(dependent, column)
            end
            prefix_ranks[source_position] = length(selected)
            continue
        end

        # A rank increase modulo one prime proves rational independence. Buffer
        # modularly dependent columns and resolve them as one exact batch.
        modular_independent = modular_rank == length(selected) &&
            _modular_rank_state_append!(modular_basis, modular_active,
                @view(potential.Q[:, column]), _RANK_SCREENING_PRIME,
                modular_workspace)
        if !modular_independent
            push!(pending_positions, source_position)
            push!(pending_columns, column)
            continue
        end

        hidden_position = _first_exactly_independent_pending(
            potential.Q, selected, pending_columns)
        if hidden_position != 0
            for index in 1:(hidden_position - 1)
                push!(dependent, pending_columns[index])
                prefix_ranks[pending_positions[index]] = length(selected)
            end
            exact_basis, exact_active, exact_workspace =
                _initialise_exact_rank_state(potential.Q, selected)
            _process_exact_columns!(selected, dependent, prefix_ranks, potential.Q,
                pending_positions[hidden_position:end],
                pending_columns[hidden_position:end], exact_basis, exact_active,
                exact_workspace)
            empty!(pending_positions)
            empty!(pending_columns)
            exact_mode = true
            independent = _exact_rank_state_append!(exact_basis, exact_active,
                @view(potential.Q[:, column]), exact_workspace)
            if independent
                push!(selected, column)
            else
                push!(dependent, column)
            end
            prefix_ranks[source_position] = length(selected)
            continue
        end

        for (position, pending_column) in zip(pending_positions, pending_columns)
            push!(dependent, pending_column)
            prefix_ranks[position] = length(selected)
        end
        empty!(pending_positions)
        empty!(pending_columns)
        push!(selected, column)
        modular_rank += 1
        prefix_ranks[source_position] = length(selected)
    end

    if !isempty(pending_columns)
        hidden_position = _first_exactly_independent_pending(
            potential.Q, selected, pending_columns)
        if hidden_position == 0
            for (position, pending_column) in zip(pending_positions, pending_columns)
                push!(dependent, pending_column)
                prefix_ranks[position] = length(selected)
            end
        else
            for index in 1:(hidden_position - 1)
                push!(dependent, pending_columns[index])
                prefix_ranks[pending_positions[index]] = length(selected)
            end
            exact_basis, exact_active, exact_workspace =
                _initialise_exact_rank_state(potential.Q, selected)
            _process_exact_columns!(selected, dependent, prefix_ranks, potential.Q,
                pending_positions[hidden_position:end],
                pending_columns[hidden_position:end], exact_basis, exact_active,
                exact_workspace)
        end
    end
    length(selected) == h11 || throw(ArgumentError(
        "leading charges have rank $(length(selected)); expected h11=$h11"))
    selected_determinant = _exact_integer_determinant(potential.Q[:, selected])
    selected_determinant != 0 || throw(ArgumentError(
        "exact rank certificate found a zero selected determinant"))
    certificate = RationalRankCertificate(
        "modular_screen_with_exact_rational_fallback_v1", (h11, ncolumns),
        copy(potential.source_indices), copy(selected), copy(dependent),
        prefix_ranks, selected_determinant)
    selected, dependent, certificate
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
        m_planck_GeV::Real=M_PLANCK_GEV)
    size(kinv, 1) == size(kinv, 2) || throw(DimensionMismatch("Kinv must be square"))
    h11 = size(kinv, 1)
    selected, dependent, rank_certificate = _select_independent_terms(potential, h11)
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
    LeadingAxionHierarchy{T}(selected, dependent, rank_certificate, Q_reduced,
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
    configuration = _make_axion_photon_configuration(T, em_divisor_index,
        light_threshold_eV, qed_threshold_policy, signed_scale_policy)
    geometry_data, potential_data, geometry_digest, potential_digest =
        h5open(path, "r") do file
            loaded_geometry = _load_geometry(file, path, index, T)
            loaded_potential = _load_potential(file, T)
            raw_geometry_digest, raw_potential_digest = _input_identity_digests(file)
            loaded_geometry, loaded_potential, raw_geometry_digest, raw_potential_digest
        end
    hierarchy_data = leading_hierarchy(potential_data, geometry_data.kinv; T=T,
        signed_scale_policy=signed_scale_policy)
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
    configuration = _configuration_with_effective_threshold(
        configuration, photons.light_threshold_eV)
    identity = _make_axion_photon_identity(index, geometry_digest,
        potential_digest, geometry_data, potential_data, configuration)
    status = qed_threshold_policy == :divisor_instanton ?
        :visible_sector_instanton_threshold :
        (signed_scale_policy == :absolute ? :adapted_absolute_scale :
         :adapted_local_geometry)
    AxionPhotonResult{T}(geometry_data, potential_data, hierarchy_data,
        photons, em_index, em_volume, em_source, qed_threshold_policy, status,
        identity)
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

function _write_axion_photon_group(group::HDF5.Group, result::AxionPhotonResult)
    _write_axion_photon_identity(group, result.identity)
    group["em_divisor_index"] = Int(result.em_divisor_index)
    group["em_divisor_volume"] = result.em_divisor_volume
    group["em_charge_source"] = String(result.em_charge_source)
    group["light_threshold_policy"] = String(result.light_threshold_policy)
    group["status"] = String(result.status)

    h = result.hierarchy
    hg = create_group(group, "hierarchy")
    hg["selected_indices", deflate=9] = h.selected_indices
    hg["dependent_indices", deflate=9] = h.dependent_indices
    hg["Q_reduced", deflate=9] = h.Q_reduced
    hg["log10_lambda4", deflate=9] = h.log10_lambda4
    hg["coefficient_signs", deflate=9] = h.coefficient_signs
    hg["q", deflate=9] = h.q
    hg["theta_from_canonical", deflate=9] = h.theta_from_canonical
    hg["log10_f_GeV", deflate=9] = h.log10_f_GeV
    hg["log10_mass_eV", deflate=9] = h.log10_mass_eV
    hg["m_planck_GeV"] = h.m_planck_GeV
    hg["triangular_residual"] = h.triangular_residual
    hg["metric_residual"] = h.metric_residual

    rc = h.rank_certificate
    rg = create_group(hg, "rank_certificate")
    rg["algorithm"] = rc.algorithm
    rg["matrix_shape"] = collect(rc.matrix_shape)
    rg["selected_determinant"] = string(rc.selected_determinant)

    p = result.photons
    pg = create_group(group, "photons")
    pg["n_em", deflate=9] = p.n_em
    pg["Cgamma", deflate=9] = p.Cgamma
    pg["log10_g_GeVinv", deflate=9] = p.log10_g_GeVinv
    pg["log10_g_effective_GeVinv", deflate=9] = p.log10_g_effective_GeVinv
    pg["log10_photon_width_GeV", deflate=9] = p.log10_photon_width_GeV
    pg["log10_quartic_width_GeV", deflate=9] = p.log10_quartic_width_GeV
    pg["light_threshold_eV"] = p.light_threshold_eV
    pg["log10_light_threshold_eV"] = p.log10_light_threshold_eV
    pg["light_mode_count"] = Int(p.light_mode_count)
    pg["charge_residual"] = p.charge_residual
    nothing
end

function _required_axion_photon_dataset(group::HDF5.Group, path::AbstractString)
    object = group[path]
    object isa HDF5.Dataset || begin
        close(object)
        throw(ArgumentError("HDF5 object '$path' must be a dataset"))
    end
    try
        HDF5.read(object)
    finally
        close(object)
    end
end

function _required_axion_photon_group(group::HDF5.Group, path::AbstractString)
    object = group[path]
    object isa HDF5.Group || begin
        close(object)
        throw(ArgumentError("HDF5 object '$path' must be a group"))
    end
    object
end

function _real_vector(value)
    value isa AbstractVector && all(entry -> entry isa Real, value)
end

function _real_matrix(value)
    value isa AbstractMatrix && all(entry -> entry isa Real, value)
end

function _integer_vector(value)
    value isa AbstractVector && all(_integer_value, value)
end

function _integer_matrix(value)
    value isa AbstractMatrix && all(_integer_value, value)
end

function _integer_value(value)
    value isa Integer || return false
    try
        converted = Int(value)
        converted == value
    catch
        false
    end
end

function _write_axion_photon_identity(group::HDF5.Group,
        identity::AxionPhotonIdentity)
    identity_group = create_group(group, "identity")
    identity_group["schema_version"] = identity.schema_version
    identity_group["geometry_h11"] = identity.geometry_index.h11
    identity_group["geometry_polytope"] = identity.geometry_index.polytope
    identity_group["geometry_frst"] = identity.geometry_index.frst
    identity_group["geometry_digest"] = identity.geometry_digest
    identity_group["potential_digest"] = identity.potential_digest
    identity_group["geometry_snapshot_digest"] = identity.geometry_snapshot_digest
    identity_group["potential_snapshot_digest"] = identity.potential_snapshot_digest
    identity_group["configuration_digest"] = identity.configuration_digest

    configuration = identity.configuration
    configuration_group = create_group(identity_group, "configuration")
    configuration_group["em_divisor_selection"] =
        _configuration_selection(configuration.em_divisor_index)
    configuration_group["light_threshold_eV"] = configuration.light_threshold_eV
    configuration_group["light_threshold_eV_effective"] =
        configuration.light_threshold_eV_effective
    configuration_group["qed_threshold_policy"] =
        String(configuration.qed_threshold_policy)
    configuration_group["signed_scale_policy"] =
        String(configuration.signed_scale_policy)
    configuration_group["precision"] = configuration.precision
    nothing
end

function _is_sha256_identity(value)
    value isa AbstractString && occursin(r"^[0-9a-f]{64}$", value)
end

function _read_axion_photon_identity(group::HDF5.Group)
    identity_group = _required_axion_photon_group(group, "identity")
    try
        schema_version = _required_axion_photon_dataset(identity_group,
            "schema_version")
        geometry_h11 = _required_axion_photon_dataset(identity_group,
            "geometry_h11")
        geometry_polytope = _required_axion_photon_dataset(identity_group,
            "geometry_polytope")
        geometry_frst = _required_axion_photon_dataset(identity_group,
            "geometry_frst")
        geometry_digest = _required_axion_photon_dataset(identity_group,
            "geometry_digest")
        potential_digest = _required_axion_photon_dataset(identity_group,
            "potential_digest")
        geometry_snapshot_digest = _required_axion_photon_dataset(identity_group,
            "geometry_snapshot_digest")
        potential_snapshot_digest = _required_axion_photon_dataset(identity_group,
            "potential_snapshot_digest")
        configuration_digest = _required_axion_photon_dataset(identity_group,
            "configuration_digest")
        _integer_value(schema_version) || throw(ArgumentError(
            "persisted axion-photon schema_version must be an integer"))
        Int(schema_version) == _AXION_PHOTON_IDENTITY_SCHEMA_VERSION ||
            throw(ArgumentError("unsupported persisted axion-photon identity schema"))
        all(_integer_value, (geometry_h11, geometry_polytope, geometry_frst)) ||
            throw(ArgumentError("persisted geometry identity must contain integer indices"))
        index_values = Int[Int(geometry_h11), Int(geometry_polytope), Int(geometry_frst)]
        all(>(0), index_values) || throw(ArgumentError(
            "persisted geometry identity indices must be positive"))
        _is_sha256_identity(geometry_digest) || throw(ArgumentError(
            "persisted geometry identity digest is invalid"))
        _is_sha256_identity(potential_digest) || throw(ArgumentError(
            "persisted potential identity digest is invalid"))
        _is_sha256_identity(geometry_snapshot_digest) || throw(ArgumentError(
            "persisted geometry snapshot digest is invalid"))
        _is_sha256_identity(potential_snapshot_digest) || throw(ArgumentError(
            "persisted potential snapshot digest is invalid"))
        _is_sha256_identity(configuration_digest) || throw(ArgumentError(
            "persisted configuration identity digest is invalid"))

        configuration_group = _required_axion_photon_group(identity_group,
            "configuration")
        try
            em_selection = _required_axion_photon_dataset(configuration_group,
                "em_divisor_selection")
            light_threshold = _required_axion_photon_dataset(configuration_group,
                "light_threshold_eV")
            light_threshold_effective = _required_axion_photon_dataset(
                configuration_group, "light_threshold_eV_effective")
            qed_policy = _required_axion_photon_dataset(configuration_group,
                "qed_threshold_policy")
            signed_policy = _required_axion_photon_dataset(configuration_group,
                "signed_scale_policy")
            precision = _required_axion_photon_dataset(configuration_group,
                "precision")
            all(value -> value isa AbstractString,
                (em_selection, light_threshold, light_threshold_effective,
                 qed_policy, signed_policy, precision)) ||
                throw(ArgumentError("persisted axion-photon configuration has invalid fields"))
            configuration = AxionPhotonConfiguration(
                _parse_configuration_selection(em_selection),
                String(light_threshold), String(light_threshold_effective),
                Symbol(String(qed_policy)),
                Symbol(String(signed_policy)), String(precision))
            configuration.qed_threshold_policy in (:electron_proxy, :divisor_instanton) ||
                throw(ArgumentError("persisted QED threshold policy is invalid"))
            configuration.signed_scale_policy in (:require_positive, :absolute) ||
                throw(ArgumentError("persisted signed-scale policy is invalid"))
            isempty(configuration.precision) && throw(ArgumentError(
                "persisted computation precision is empty"))
            _configuration_precision_type(configuration.precision)
            expected_digest = _configuration_digest(configuration)
            String(configuration_digest) == expected_digest || throw(ArgumentError(
                "persisted configuration identity digest does not match its fields"))
            AxionPhotonIdentity(_AXION_PHOTON_IDENTITY_SCHEMA_VERSION,
                GeometryIndex(index_values[1], index_values[2], index_values[3]),
                String(geometry_digest), String(potential_digest),
                String(geometry_snapshot_digest), String(potential_snapshot_digest),
                configuration, String(configuration_digest))
        finally
            close(configuration_group)
        end
    finally
        close(identity_group)
    end
end

"""Validate the complete persisted axion-photon group."""
function _validate_axion_photon_group(group::HDF5.Group)
    nested_groups = HDF5.Group[]
    try
        identity = _read_axion_photon_identity(group)
        em_divisor_index = _required_axion_photon_dataset(group, "em_divisor_index")
        em_divisor_volume = _required_axion_photon_dataset(group, "em_divisor_volume")
        em_charge_source = _required_axion_photon_dataset(group, "em_charge_source")
        light_threshold_policy =
            _required_axion_photon_dataset(group, "light_threshold_policy")
        status = _required_axion_photon_dataset(group, "status")
        _integer_value(em_divisor_index) || return false
        em_divisor_volume isa Real || return false
        em_charge_source isa AbstractString || return false
        light_threshold_policy isa AbstractString || return false
        status isa AbstractString || return false
        String(light_threshold_policy) ==
            String(identity.configuration.qed_threshold_policy) || return false
        identity.configuration.em_divisor_index === nothing ||
            Int(em_divisor_index) == identity.configuration.em_divisor_index || return false
        if identity.configuration.em_divisor_index !== nothing
            String(em_charge_source) == "direct_effective_cone" || return false
        end

        hierarchy = _required_axion_photon_group(group, "hierarchy")
        push!(nested_groups, hierarchy)
        selected_indices = _required_axion_photon_dataset(hierarchy, "selected_indices")
        dependent_indices = _required_axion_photon_dataset(hierarchy, "dependent_indices")
        Q_reduced = _required_axion_photon_dataset(hierarchy, "Q_reduced")
        log10_lambda4 = _required_axion_photon_dataset(hierarchy, "log10_lambda4")
        coefficient_signs =
            _required_axion_photon_dataset(hierarchy, "coefficient_signs")
        q = _required_axion_photon_dataset(hierarchy, "q")
        theta_from_canonical =
            _required_axion_photon_dataset(hierarchy, "theta_from_canonical")
        log10_f_GeV = _required_axion_photon_dataset(hierarchy, "log10_f_GeV")
        log10_mass_eV = _required_axion_photon_dataset(hierarchy, "log10_mass_eV")
        m_planck_GeV = _required_axion_photon_dataset(hierarchy, "m_planck_GeV")
        triangular_residual =
            _required_axion_photon_dataset(hierarchy, "triangular_residual")
        metric_residual = _required_axion_photon_dataset(hierarchy, "metric_residual")

        _integer_vector(selected_indices) || return false
        _integer_vector(dependent_indices) || return false
        _integer_matrix(Q_reduced) || return false
        _real_vector(log10_lambda4) || return false
        _integer_vector(coefficient_signs) || return false
        _real_matrix(q) || return false
        _real_matrix(theta_from_canonical) || return false
        _real_vector(log10_f_GeV) || return false
        _real_vector(log10_mass_eV) || return false
        m_planck_GeV isa Real || return false
        triangular_residual isa Real || return false
        metric_residual isa Real || return false

        h11 = length(selected_indices)
        size(Q_reduced) == (h11, h11) || return false
        length(log10_lambda4) == h11 || return false
        length(coefficient_signs) == h11 || return false
        size(q) == (h11, h11) || return false
        size(theta_from_canonical) == (h11, h11) || return false
        length(log10_f_GeV) == h11 || return false
        length(log10_mass_eV) == h11 || return false

        rank_certificate = _required_axion_photon_group(
            hierarchy, "rank_certificate")
        push!(nested_groups, rank_certificate)
        algorithm = _required_axion_photon_dataset(rank_certificate, "algorithm")
        matrix_shape = _required_axion_photon_dataset(rank_certificate, "matrix_shape")
        selected_determinant = _required_axion_photon_dataset(
            rank_certificate, "selected_determinant")
        algorithm isa AbstractString || return false
        _integer_vector(matrix_shape) && length(matrix_shape) == 2 || return false
        all(>(0), matrix_shape) || return false
        selected_determinant isa AbstractString || return false
        parse(BigInt, selected_determinant)

        photons = _required_axion_photon_group(group, "photons")
        push!(nested_groups, photons)
        n_em = _required_axion_photon_dataset(photons, "n_em")
        Cgamma = _required_axion_photon_dataset(photons, "Cgamma")
        log10_g_GeVinv = _required_axion_photon_dataset(
            photons, "log10_g_GeVinv")
        log10_g_effective_GeVinv = _required_axion_photon_dataset(
            photons, "log10_g_effective_GeVinv")
        log10_photon_width_GeV = _required_axion_photon_dataset(
            photons, "log10_photon_width_GeV")
        log10_quartic_width_GeV = _required_axion_photon_dataset(
            photons, "log10_quartic_width_GeV")
        light_threshold_eV = _required_axion_photon_dataset(
            photons, "light_threshold_eV")
        log10_light_threshold_eV = _required_axion_photon_dataset(
            photons, "log10_light_threshold_eV")
        light_mode_count = _required_axion_photon_dataset(photons, "light_mode_count")
        charge_residual = _required_axion_photon_dataset(photons, "charge_residual")

        _real_vector(n_em) || return false
        _real_vector(Cgamma) || return false
        _real_vector(log10_g_GeVinv) || return false
        _real_vector(log10_g_effective_GeVinv) || return false
        _real_vector(log10_photon_width_GeV) || return false
        _real_vector(log10_quartic_width_GeV) || return false
        light_threshold_eV isa Real || return false
        log10_light_threshold_eV isa Real || return false
        _integer_value(light_mode_count) || return false
        charge_residual isa Real || return false
        all(length(value) == h11 for value in (
            n_em, Cgamma, log10_g_GeVinv, log10_g_effective_GeVinv,
            log10_photon_width_GeV, log10_quartic_width_GeV)) || return false
        if identity.configuration.qed_threshold_policy == :electron_proxy
            identity.configuration.light_threshold_eV_effective ==
                _configuration_value(light_threshold_eV) || return false
        else
            identity.configuration.light_threshold_eV_effective == "unused" || return false
        end
        true
    catch err
        err isa InterruptException && rethrow()
        false
    finally
        for nested_group in reverse(nested_groups)
            close(nested_group)
        end
    end
end

function _assert_persisted_identity_matches!(file::HDF5.File,
        path::AbstractString, identity::AxionPhotonIdentity)
    expected_index = _index_from_path(path)
    identity.geometry_index == expected_index || throw(ArgumentError(
        "persisted axion-photon result belongs to geometry " *
        "$(identity.geometry_index), not destination geometry $expected_index"))
    geometry_digest, potential_digest = _input_identity_digests(file)
    identity.geometry_digest == geometry_digest || throw(ArgumentError(
        "persisted axion-photon result does not match destination geometry inputs"))
    identity.potential_digest == potential_digest || throw(ArgumentError(
        "persisted axion-photon result does not match destination potential inputs"))
    nothing
end

function _assert_snapshot_identity_matches!(file::HDF5.File,
        path::AbstractString, identity::AxionPhotonIdentity)
    stored_T = _configuration_precision_type(identity.configuration.precision)
    index = _index_from_path(path)
    stored_geometry = _load_geometry(file, path, index, stored_T)
    stored_potential = _load_potential(file, stored_T)
    identity.geometry_snapshot_digest == _geometry_snapshot_digest(stored_geometry) ||
        throw(ArgumentError(
            "persisted axion-photon result does not match loaded geometry snapshot"))
    identity.potential_snapshot_digest == _potential_snapshot_digest(stored_potential) ||
        throw(ArgumentError(
            "persisted axion-photon result does not match loaded potential snapshot"))
    stored_geometry, stored_potential
end

function _assert_result_identity_matches!(file::HDF5.File,
        path::AbstractString, result::AxionPhotonResult)
    identity = result.identity
    identity.schema_version == _AXION_PHOTON_IDENTITY_SCHEMA_VERSION || throw(ArgumentError(
        "axion-photon result has an unsupported identity schema"))
    identity.geometry_index == result.geometry.index || throw(ArgumentError(
        "axion-photon result identity does not match its retained geometry"))
    identity.configuration_digest == _configuration_digest(identity.configuration) ||
        throw(ArgumentError("axion-photon result has inconsistent configuration identity"))
    identity.geometry_snapshot_digest == _geometry_snapshot_digest(result.geometry) ||
        throw(ArgumentError("axion-photon result geometry was changed after computation"))
    identity.potential_snapshot_digest == _potential_snapshot_digest(result.potential) ||
        throw(ArgumentError("axion-photon result potential was changed after computation"))
    _assert_persisted_identity_matches!(file, path, identity)
    nothing
end

function _axion_photon_temp_path(path::AbstractString)
    string(path, ".axion_photon.tmp-", getpid(), "-", time_ns())
end

function _axion_photon_link_name(parent::HDF5.Group, prefix::AbstractString)
    name = string(prefix, getpid(), "-", time_ns())
    while haskey(parent, name)
        name = string(prefix, getpid(), "-", time_ns(), "-", rand(UInt))
    end
    name
end

function _rollback_axion_photon_publication!(file::HDF5.File,
        spectrum, stage_name, backup_name, spectrum_created, new_moved, old_moved)
    rollback_errors = Any[]
    if spectrum !== nothing
        if stage_name !== nothing && haskey(spectrum, stage_name)
            try
                HDF5.delete_object(spectrum, stage_name)
            catch err
                push!(rollback_errors, err)
            end
        end
        if new_moved && haskey(spectrum, "axion_photon")
            try
                HDF5.delete_object(spectrum, "axion_photon")
            catch err
                push!(rollback_errors, err)
            end
        end
        if old_moved && backup_name !== nothing &&
                !haskey(spectrum, "axion_photon") && haskey(spectrum, backup_name)
            try
                HDF5.move_link(spectrum, backup_name, spectrum, "axion_photon")
            catch err
                push!(rollback_errors, err)
            end
        end
        if spectrum_created && !haskey(spectrum, "axion_photon") &&
                isempty(keys(spectrum))
            try
                HDF5.delete_object(file, "spectrum")
            catch err
                push!(rollback_errors, err)
            end
        end
    end
    rollback_errors
end

function _publish_axion_photon_group!(file::HDF5.File, source::HDF5.File)
    spectrum = nothing
    spectrum_created = false
    stage_name = nothing
    backup_name = nothing
    new_moved = false
    old_moved = false
    try
        if haskey(file, "spectrum")
            spectrum = file["spectrum"]
            spectrum isa HDF5.Group || throw(ArgumentError(
                "HDF5 object 'spectrum' must be a group"))
        else
            spectrum = create_group(file, "spectrum")
            spectrum_created = true
        end

        stage_name = _axion_photon_link_name(spectrum, "__axion_photon_stage-")
        HDF5.copy_object(source, "axion_photon", spectrum, stage_name)
        staged_group = _required_axion_photon_group(spectrum, stage_name)
        try
            _validate_axion_photon_group(staged_group) || throw(ArgumentError(
                "staged axion-photon result failed validation"))
        finally
            close(staged_group)
        end

        if haskey(spectrum, "axion_photon")
            backup_name = _axion_photon_link_name(spectrum, "__axion_photon_backup-")
            HDF5.move_link(spectrum, "axion_photon", spectrum, backup_name)
            old_moved = true
        end
        HDF5.move_link(spectrum, stage_name, spectrum, "axion_photon")
        new_moved = true
        published_group = _required_axion_photon_group(spectrum, "axion_photon")
        try
            _validate_axion_photon_group(published_group) ||
                throw(ArgumentError("published axion-photon result failed validation"))
        finally
            close(published_group)
        end

        if old_moved && backup_name !== nothing
            HDF5.delete_object(spectrum, backup_name)
            old_moved = false
        end
        nothing
    catch err
        rollback_errors = _rollback_axion_photon_publication!(file, spectrum,
            stage_name, backup_name, spectrum_created, new_moved, old_moved)
        isempty(rollback_errors) || throw(ErrorException(string(
            "axion-photon publication failed and rollback also failed: ",
            sprint(showerror, err), "; rollback: ",
            join(sprint.(showerror, rollback_errors), "; "))))
        rethrow()
    end
end

"""Write an axion-photon result into the geometry's HDF5 file."""
function write_axion_photon_result(path::AbstractString, result::AxionPhotonResult;
        force::Bool=false)
    isfile(path) || throw(ArgumentError("geometry file does not exist: $path"))
    h5open(path, "r") do file
        _assert_result_identity_matches!(file, path, result)
    end
    !force && _has_axion_photon(path) && throw(ArgumentError(
        "spectrum/axion_photon already exists; use force=true to overwrite"))

    temporary = _axion_photon_temp_path(path)
    try
        h5open(temporary, "w") do staged_file
            staged_group = create_group(staged_file, "axion_photon")
            _write_axion_photon_group(staged_group, result)
            _validate_axion_photon_group(staged_group) || throw(ArgumentError(
                "serialized axion-photon result failed validation"))
        end

        h5open(temporary, "r") do source
            h5open(path, "r+") do file
                _assert_result_identity_matches!(file, path, result)
                !force && _has_axion_photon(file;
                    expected_index=_index_from_path(path), path=path) && throw(ArgumentError(
                    "spectrum/axion_photon already exists; use force=true to overwrite"))
                _publish_axion_photon_group!(file, source)
            end
        end
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    path
end

function write_axion_photon_result(result::AxionPhotonResult; force::Bool=false)
    write_axion_photon_result(result.geometry.path, result; force=force)
end

"""Read a persisted axion-photon result from a geometry's HDF5 file."""
function read_axion_photon_result(path::AbstractString;
        T::Type{<:AbstractFloat}=Float64)
    isfile(path) || throw(ArgumentError("geometry file does not exist: $path"))
    h5open(path, "r") do file
        haskey(file, "spectrum") && haskey(file["spectrum"], "axion_photon") ||
            throw(ArgumentError(
                "spectrum/axion_photon group not found in file: $path"))

        ap = file["spectrum/axion_photon"]::HDF5.Group
        _validate_axion_photon_group(ap) || throw(ArgumentError(
            "spectrum/axion_photon is incomplete or lacks required identity metadata; " *
            "recompute the result"))
        identity = _read_axion_photon_identity(ap)
        _assert_persisted_identity_matches!(file, path, identity)
        stored_T = _configuration_precision_type(identity.configuration.precision)

        em_divisor_index = Int(_required_axion_photon_dataset(ap, "em_divisor_index"))
        em_divisor_volume = T(_required_axion_photon_dataset(ap, "em_divisor_volume"))
        em_charge_source = Symbol(String(_required_axion_photon_dataset(ap, "em_charge_source")))
        light_threshold_policy = Symbol(String(_required_axion_photon_dataset(ap, "light_threshold_policy")))
        status = Symbol(String(_required_axion_photon_dataset(ap, "status")))

        hg = ap["hierarchy"]::HDF5.Group
        selected_indices = Vector{Int}(_required_axion_photon_dataset(hg, "selected_indices"))
        dependent_indices = Vector{Int}(_required_axion_photon_dataset(hg, "dependent_indices"))
        Q_reduced = Matrix{Int}(_required_axion_photon_dataset(hg, "Q_reduced"))
        log10_lambda4 = Vector{T}(_required_axion_photon_dataset(hg, "log10_lambda4"))
        coefficient_signs = Vector{Int}(_required_axion_photon_dataset(hg, "coefficient_signs"))
        q = Matrix{T}(_required_axion_photon_dataset(hg, "q"))
        theta_from_canonical = Matrix{T}(_required_axion_photon_dataset(hg, "theta_from_canonical"))
        log10_f_GeV = Vector{T}(_required_axion_photon_dataset(hg, "log10_f_GeV"))
        log10_mass_eV = Vector{T}(_required_axion_photon_dataset(hg, "log10_mass_eV"))
        m_planck_GeV = T(_required_axion_photon_dataset(hg, "m_planck_GeV"))
        triangular_residual = T(_required_axion_photon_dataset(hg, "triangular_residual"))
        metric_residual = T(_required_axion_photon_dataset(hg, "metric_residual"))

        rg = hg["rank_certificate"]::HDF5.Group
        algorithm = String(_required_axion_photon_dataset(rg, "algorithm"))
        matrix_shape = Tuple(Vector{Int}(_required_axion_photon_dataset(rg, "matrix_shape")))
        selected_determinant = parse(BigInt, String(_required_axion_photon_dataset(rg, "selected_determinant")))

        rank_cert = RationalRankCertificate(
            algorithm,
            matrix_shape,
            Int[],
            copy(selected_indices),
            copy(dependent_indices),
            Int[],
            selected_determinant)

        hierarchy = LeadingAxionHierarchy{T}(
            selected_indices,
            dependent_indices,
            rank_cert,
            Q_reduced,
            log10_lambda4,
            coefficient_signs,
            q,
            theta_from_canonical,
            log10_f_GeV,
            log10_mass_eV,
            m_planck_GeV,
            triangular_residual,
            metric_residual)

        theta = mixing_matrix(hierarchy)

        pg = ap["photons"]::HDF5.Group
        n_em = Vector{T}(_required_axion_photon_dataset(pg, "n_em"))
        Cgamma = Vector{T}(_required_axion_photon_dataset(pg, "Cgamma"))
        log10_g_GeVinv = Vector{T}(_required_axion_photon_dataset(pg, "log10_g_GeVinv"))
        log10_g_effective_GeVinv = Vector{T}(_required_axion_photon_dataset(pg, "log10_g_effective_GeVinv"))
        log10_photon_width_GeV = Vector{T}(_required_axion_photon_dataset(pg, "log10_photon_width_GeV"))
        log10_quartic_width_GeV = Vector{T}(_required_axion_photon_dataset(pg, "log10_quartic_width_GeV"))
        light_threshold_eV = T(_required_axion_photon_dataset(pg, "light_threshold_eV"))
        log10_light_threshold_eV = T(_required_axion_photon_dataset(pg, "log10_light_threshold_eV"))
        light_mode_count = Int(_required_axion_photon_dataset(pg, "light_mode_count"))
        charge_residual = T(_required_axion_photon_dataset(pg, "charge_residual"))

        photons = AxionPhotonObservables{T}(
            n_em,
            theta,
            Cgamma,
            log10_g_GeVinv,
            log10_g_effective_GeVinv,
            log10_photon_width_GeV,
            log10_quartic_width_GeV,
            light_threshold_eV,
            log10_light_threshold_eV,
            light_mode_count,
            charge_residual)

        index = _index_from_path(path)
        stored_geometry, stored_potential =
            _assert_snapshot_identity_matches!(file, path, identity)
        geometry = T === stored_T ? stored_geometry : _load_geometry(file, path, index, T)
        potential = T === stored_T ? stored_potential : _load_potential(file, T)

        AxionPhotonResult{T}(
            geometry,
            potential,
            hierarchy,
            photons,
            em_divisor_index,
            em_divisor_volume,
            em_charge_source,
            light_threshold_policy,
            status,
            identity)
    end
end

function read_axion_photon_result(index::GeometryIndex;
        data_dir=nothing, T::Type{<:AbstractFloat}=Float64)
    path = geometry_path(index; data_dir=data_dir)
    read_axion_photon_result(path; T=T)
end

function _has_axion_photon(file::HDF5.File; expected_index=nothing, path=nothing)
    opened_groups = HDF5.Group[]
    try
        haskey(file, "spectrum") || return false
        spectrum = file["spectrum"]
        spectrum isa HDF5.Group || begin
            close(spectrum)
            return false
        end
        push!(opened_groups, spectrum)
        haskey(spectrum, "axion_photon") || return false
        group = _required_axion_photon_group(spectrum, "axion_photon")
        push!(opened_groups, group)
        _validate_axion_photon_group(group) || return false
        identity = _read_axion_photon_identity(group)
        expected_index === nothing || identity.geometry_index == expected_index || return false
        geometry_digest, potential_digest = _input_identity_digests(file)
        identity.geometry_digest == geometry_digest || return false
        identity.potential_digest == potential_digest || return false
        if path !== nothing
            _assert_snapshot_identity_matches!(file, path, identity)
        end
        return true
    catch err
        err isa InterruptException && rethrow()
        return false
    finally
        for group in reverse(opened_groups)
            close(group)
        end
    end
end

function _has_axion_photon(path::AbstractString; configuration=nothing)
    isfile(path) || return false
    expected_index = try
        _index_from_path(path)
    catch
        return false
    end
    try
        h5open(path, "r") do file
            _has_axion_photon(file; expected_index=expected_index, path=path) || return false
            group = file["spectrum/axion_photon"]
            group isa HDF5.Group || begin
                close(group)
                return false
            end
            try
                identity = _read_axion_photon_identity(group)
                return configuration === nothing || identity.configuration == configuration
            finally
                close(group)
            end
        end
    catch err
        err isa InterruptException && rethrow()
        false
    end
end

"""Run an axion-photon scan over local geometries and persist each result to HDF5."""
function run_batch_axion_photon(; data_dir=nothing,
        h11s=(15, 100, 200, 300), limit_per_h11::Integer=2,
        require_complete::Bool=true,
        T::Type{<:AbstractFloat}=Float64,
        em_divisor_index::Union{Nothing,Integer}=nothing,
        light_threshold_eV::Real=ELECTRON_MASS_EV,
        qed_threshold_policy::Symbol=:electron_proxy,
        signed_scale_policy::Symbol=:require_positive,
        force::Bool=false,
        skip_complete::Bool=true,
        verbose::Bool=false)
    root = resolve_data_dir(data_dir)
    indices = local_geometry_indices(root; h11s=h11s,
        limit_per_h11=limit_per_h11, require_complete=require_complete)
    isempty(indices) && throw(ArgumentError(
        "no matching complete local geometries were found in $root"))
    requested_configuration = _make_axion_photon_configuration(T,
        em_divisor_index, light_threshold_eV, qed_threshold_policy,
        signed_scale_policy)
    summaries = Vector{@NamedTuple{index::GeometryIndex{Int}, path::String,
        status::Symbol, error::String}}()
    for index in indices
        path = geometry_path(index; data_dir=root)
        if skip_complete && !force && _has_axion_photon(path)
            stored_configuration = h5open(path, "r") do file
                ap = file["spectrum/axion_photon"]::HDF5.Group
                try
                    _read_axion_photon_identity(ap).configuration
                finally
                    close(ap)
                end
            end
            if _configuration_matches_request(
                    stored_configuration, requested_configuration)
                verbose && println("skip $path (axion_photon exists)")
                push!(summaries, (index=index, path=path, status=:skipped, error=""))
                continue
            end
            message = "valid axion-photon result has a different configuration; " *
                "use force=true to recompute"
            verbose && println(message)
            push!(summaries, (index=index, path=path, status=:failed, error=message))
            continue
        end
        try
            result = _run_local_scan(path; T=T,
                em_divisor_index=em_divisor_index,
                light_threshold_eV=light_threshold_eV,
                qed_threshold_policy=qed_threshold_policy,
                signed_scale_policy=signed_scale_policy)
            write_axion_photon_result(path, result; force=force)
            verbose && println("wrote $path")
            push!(summaries, (index=index, path=path, status=:written, error=""))
        catch err
            verbose && println(err)
            push!(summaries, (index=index, path=path, status=:failed,
                error=string(err)))
        end
    end
    summaries
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
    RationalRankCertificate, LeadingAxionHierarchy, AxionPhotonObservables,
    AxionPhotonConfiguration, AxionPhotonIdentity,
    AxionPhotonResult, rank_certificate_payload,
    load_geometry_inputs,
    load_instanton_data, geometry_path, local_geometry_indices,
    leading_hierarchy, mixing_matrix, photon_observables,
    qed_instanton_log10_threshold_eV, qed_instanton_threshold_eV,
    run_local_scan, write_scan_csv,
    write_axion_photon_result,
    read_axion_photon_result,
    run_batch_axion_photon

end
