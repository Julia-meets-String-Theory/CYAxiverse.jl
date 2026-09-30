"""Tuple-backed array storage that cannot be changed after construction."""
struct FrozenArray{T,N,L} <: AbstractArray{T,N}
    values::NTuple{L,T}
    dimensions::NTuple{N,Int}
end

function FrozenArray(array::AbstractArray{T,N}) where {T,N}
    values = Tuple(deepcopy(value) for value in array)
    FrozenArray{T,N,length(values)}(values, size(array))
end

function Base.getproperty(array::FrozenArray, name::Symbol)
    name === :values && return Tuple(deepcopy(value) for value in getfield(array, :values))
    getfield(array, name)
end

Base.size(array::FrozenArray) = getfield(array, :dimensions)
Base.IndexStyle(::Type{<:FrozenArray}) = IndexLinear()
Base.getindex(array::FrozenArray, index::Int) = deepcopy(getfield(array, :values)[index])
function Base.getindex(array::FrozenArray{T,N}, indices::Vararg{Int,N}) where {T,N}
    linear_index = LinearIndices(array)[indices...]
    deepcopy(getfield(array, :values)[linear_index])
end
Base.copy(array::FrozenArray) = Array(reshape(
    [deepcopy(value) for value in getfield(array, :values)], size(array)))

"""A sparse, canonical representation of a symmetric triple-intersection tensor."""
struct CanonicalIntersectionTensor{T<:Number}
    n::Int
    triples::Tuple{Vararg{NTuple{3,Int}}}
    coefficients::FrozenArray{T,1}
end

_widen_exact(value::Integer) = BigInt(value)
_widen_exact(value::Rational) = Rational{BigInt}(value)
_widen_exact(value) = value
_widen_exact_array(values::AbstractArray) = map(_widen_exact, values)

function CanonicalIntersectionTensor(n::Integer, terms::AbstractVector{<:Pair})
    n > 0 || throw(ArgumentError("the divisor dimension must be positive"))
    accum = Dict{NTuple{3,Int},Any}()
    for item in terms
        length(item.first) == 3 || throw(ArgumentError("intersection keys must be triples"))
        key = Tuple(sort!(collect(Int, item.first)))
        all(i -> 1 <= i <= n, key) || throw(BoundsError(1:n, key))
        value = _widen_exact(item.second)
        accum[key] = get(accum, key, zero(value)) + value
    end
    keys_sorted = sort!(collect(keys(accum)))
    isempty(keys_sorted) && throw(ArgumentError("the intersection tensor cannot be empty"))
    values_sorted = [accum[key] for key in keys_sorted if !iszero(accum[key])]
    triples_sorted = [key for key in keys_sorted if !iszero(accum[key])]
    isempty(triples_sorted) && throw(ArgumentError("the intersection tensor cannot be identically zero"))
    T = promote_type(map(typeof, values_sorted)...)
    CanonicalIntersectionTensor{T}(Int(n), Tuple(triples_sorted), FrozenArray(T.(values_sorted)))
end

function _permuted_entries(i::Int, j::Int, k::Int)
    if i == k
        return ((i, j, k),)
    elseif i == j
        return ((i, i, k), (i, k, i), (k, i, i))
    elseif j == k
        return ((i, j, j), (j, i, j), (j, j, i))
    end
    ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i))
end

function _full_entries(tensor::CanonicalIntersectionTensor)
    ((i, j, k, value)
     for (triple, value) in zip(tensor.triples, tensor.coefficients)
     for (i, j, k) in _permuted_entries(triple...))
end

function dense_intersections(tensor::CanonicalIntersectionTensor{T}) where {T}
    dense = zeros(T, tensor.n, tensor.n, tensor.n)
    for (i, j, k, value) in _full_entries(tensor)
        dense[i, j, k] = value
    end
    dense
end

"""Immutable identity and importer provenance for one geometry source."""
struct GeometrySourceIdentity
    source_kind::Symbol
    source_id::String
    source_revision::String
    source_locator::String
    source_sha256::String
    polytope_identity::String
    triangulation_identity::String
    cytools_revision::Union{Nothing,String}
    importer_id::String
end

function GeometrySourceIdentity(; source_kind::Symbol, source_id::AbstractString,
        source_revision::AbstractString, source_locator::AbstractString,
        source_sha256::AbstractString, polytope_identity="not_applicable:native_fixture",
        triangulation_identity="not_applicable:native_fixture", cytools_revision=nothing,
        importer_id::AbstractString)
    source_kind in (:cytools_export, :native_fixture, :other) ||
        throw(ArgumentError("unsupported geometry source kind: $source_kind"))
    all(!isempty, (source_id, source_revision, source_locator, importer_id,
        polytope_identity, triangulation_identity)) ||
        throw(ArgumentError("geometry source identity strings must be nonempty"))
    occursin(r"^[0-9a-f]{64}$", source_sha256) ||
        throw(ArgumentError("source_sha256 must be 64 lowercase hexadecimal characters"))
    cytools = cytools_revision === nothing ? nothing : String(cytools_revision)
    source_kind === :cytools_export && (cytools === nothing || isempty(cytools)) &&
        throw(ArgumentError("a CYTools export must record its exact CYTools revision"))
    source_kind === :cytools_export &&
        (startswith(polytope_identity, "not_applicable:") ||
         startswith(triangulation_identity, "not_applicable:")) &&
        throw(ArgumentError("a CYTools export must identify its polytope and triangulation"))
    GeometrySourceIdentity(source_kind, String(source_id), String(source_revision),
        String(source_locator), String(source_sha256), String(polytope_identity),
        String(triangulation_identity), cytools, String(importer_id))
end

"""Provenance for imported cone inequalities. A toric inference is not completeness evidence."""
struct ConeProvenance
    status::Symbol
    construction::String
    normalization::String
    completeness::Symbol
    source_locator::String
    function ConeProvenance(status::Symbol, construction::AbstractString,
            normalization::AbstractString, completeness::Symbol,
            source_locator::AbstractString)
        status in (:torically_inferred, :independently_established, :unknown) ||
            throw(ArgumentError("invalid cone provenance status"))
        completeness in (:established, :not_established, :unknown) ||
            throw(ArgumentError("invalid cone completeness status"))
        all(!isempty, (construction, normalization, source_locator)) ||
            throw(ArgumentError("cone provenance fields must be nonempty"))
        status === :torically_inferred && completeness === :established &&
            throw(ArgumentError("toric inference alone cannot establish cone completeness"))
        new(status, String(construction), String(normalization), completeness,
            String(source_locator))
    end
end

"""Versioned native geometry data with immutable tensor and array storage."""
struct GeometryRecord{K<:Number,C<:Number}
    schema_version::String
    intersections::CanonicalIntersectionTensor{K}
    euler_characteristic::Int
    ordered_divisors::Tuple{Vararg{String}}
    ordered_curves::Tuple{Vararg{String}}
    divisor_basis_map::FrozenArray{BigInt,2}
    dual_curve_basis_map::FrozenArray{BigInt,2}
    domain_inequalities::FrozenArray{C,2}
    cone_provenance::ConeProvenance
    precision::String
    exactness::Symbol
    units::String
    source::GeometrySourceIdentity
    basis_history::Tuple{Vararg{String}}
    artifact_sha256::String

    function GeometryRecord{K,C}(schema_version::String,
            intersections::CanonicalIntersectionTensor{K}, euler_characteristic::Int,
            ordered_divisors::Tuple{Vararg{String}}, ordered_curves::Tuple{Vararg{String}},
            divisor_basis_map::FrozenArray{BigInt,2}, dual_curve_basis_map::FrozenArray{BigInt,2},
            domain_inequalities::FrozenArray{C,2}, cone_provenance::ConeProvenance,
            precision::String, exactness::Symbol, units::String,
            source::GeometrySourceIdentity, basis_history::Tuple{Vararg{String}}) where {K<:Number,C<:Number}
        artifact_sha = _geometry_digest(schema_version, intersections, euler_characteristic,
            ordered_divisors, ordered_curves, divisor_basis_map, dual_curve_basis_map,
            domain_inequalities, cone_provenance, precision, exactness, units, source,
            basis_history)
        new{K,C}(schema_version, intersections, euler_characteristic, ordered_divisors,
            ordered_curves, divisor_basis_map, dual_curve_basis_map, domain_inequalities,
            cone_provenance, precision, exactness, units, source, basis_history,
            artifact_sha)
    end
end

function _integer_determinant(matrix::AbstractMatrix{<:Integer})
    n, m = size(matrix)
    n == m || throw(DimensionMismatch("basis maps must be square"))
    A = BigInt.(matrix)
    n == 0 && return BigInt(1)
    sign = BigInt(1)
    previous = BigInt(1)
    for k in 1:(n - 1)
        pivot_row = findfirst(i -> !iszero(A[i, k]), k:n)
        pivot_row === nothing && return BigInt(0)
        pivot = first(pivot_row) + k - 1
        if pivot != k
            A[k, :], A[pivot, :] = copy(A[pivot, :]), copy(A[k, :])
            sign = -sign
        end
        pivot_value = A[k, k]
        for i in (k + 1):n, j in (k + 1):n
            A[i, j] = div(A[i, j] * pivot_value - A[i, k] * A[k, j], previous)
        end
        for i in (k + 1):n
            A[i, k] = 0
        end
        previous = pivot_value
    end
    sign * A[n, n]
end

function _integer_inverse(matrix::AbstractMatrix{<:Integer})
    abs(_integer_determinant(matrix)) == 1 ||
        throw(ArgumentError("basis covariance requires a unimodular integer matrix"))
    rational_inverse = inv(Rational{BigInt}.(matrix))
    all(x -> denominator(x) == 1, rational_inverse) ||
        throw(ArgumentError("unimodular inverse unexpectedly has nonintegral entries"))
    BigInt.(numerator.(rational_inverse))
end

function _push_digest_matrix!(parts::Vector{String}, name::String, matrix::AbstractMatrix)
    push!(parts, name, string(size(matrix, 1)), string(size(matrix, 2)), string(eltype(matrix)))
    append!(parts, (repr(value) for value in matrix))
end

function _geometry_digest(schema, intersections, euler, divisors, curves,
        divisor_map, curve_map, inequalities, cone, precision, exactness,
        units, source, basis_history)
    parts = String["cyax0191-geometry-digest-v2", string(schema),
        string(intersections.n), string(eltype(intersections.coefficients)),
        string(length(intersections.triples))]
    for i in eachindex(intersections.triples)
        push!(parts, repr(intersections.triples[i]), repr(intersections.coefficients[i]))
    end
    append!(parts, (string(euler), string(length(divisors))))
    append!(parts, divisors)
    push!(parts, string(length(curves)))
    append!(parts, curves)
    _push_digest_matrix!(parts, "divisor_basis_map", divisor_map)
    _push_digest_matrix!(parts, "dual_curve_basis_map", curve_map)
    _push_digest_matrix!(parts, "domain_inequalities", inequalities)
    append!(parts, (string(cone.status), cone.construction, cone.normalization,
        string(cone.completeness), cone.source_locator, precision,
        string(exactness), units, string(source.source_kind), source.source_id,
        source.source_revision, source.source_locator, source.source_sha256,
        source.polytope_identity, source.triangulation_identity))
    if source.cytools_revision === nothing
        push!(parts, "cytools_revision:none", "")
    else
        push!(parts, "cytools_revision:string", source.cytools_revision)
    end
    append!(parts, (source.importer_id,
        string(length(basis_history))))
    append!(parts, basis_history)
    io = IOBuffer()
    for part in parts
        encoded = codeunits(part)
        write(io, string(length(encoded)), ':')
        write(io, encoded)
    end
    bytes2hex(sha256(take!(io)))
end

function GeometryRecord(intersections::CanonicalIntersectionTensor{K};
        schema_version="cyax0191-geometry-v1", euler_characteristic::Integer,
        ordered_divisors, ordered_curves, divisor_basis_map,
        dual_curve_basis_map, domain_inequalities, cone_provenance::ConeProvenance,
        precision::AbstractString, exactness::Symbol, units::AbstractString,
        source::GeometrySourceIdentity, basis_history=()) where {K}
    n = intersections.n
    divisors = Tuple(String.(ordered_divisors))
    curves = Tuple(String.(ordered_curves))
    length(divisors) == n || throw(DimensionMismatch("one ordered divisor is required per modulus"))
    length(curves) == n || throw(DimensionMismatch("one dual curve is required per modulus"))
    length(unique(divisors)) == n || throw(ArgumentError("ordered divisor labels must be unique"))
    length(unique(curves)) == n || throw(ArgumentError("ordered curve labels must be unique"))
    dmap = BigInt.(divisor_basis_map)
    cmap = BigInt.(dual_curve_basis_map)
    size(dmap) == (n, n) || throw(DimensionMismatch("divisor basis map must be n×n"))
    size(cmap) == (n, n) || throw(DimensionMismatch("dual curve basis map must be n×n"))
    _integer_determinant(dmap) != 0 || throw(ArgumentError("divisor basis map is singular"))
    _integer_determinant(cmap) != 0 || throw(ArgumentError("dual curve basis map is singular"))
    BigInt.(dmap)' * BigInt.(cmap) == Matrix{BigInt}(I, n, n) ||
        throw(ArgumentError("divisor and dual curve basis maps are not dual"))
    inequalities = Matrix(domain_inequalities)
    size(inequalities, 2) == n || throw(DimensionMismatch("cone inequalities must have n columns"))
    size(inequalities, 1) > 0 || throw(ArgumentError("at least one imported cone inequality is required"))
    all(isfinite, inequalities) || throw(ArgumentError("cone inequalities must be finite"))
    exactness in (:exact, :rational, :approximate) || throw(ArgumentError("invalid geometry exactness"))
    all(!isempty, (schema_version, precision, units)) || throw(ArgumentError("geometry metadata must be nonempty"))
    history = Tuple(String.(basis_history))
    all(!isempty, history) || throw(ArgumentError("basis history entries must be nonempty"))
    GeometryRecord{K,eltype(inequalities)}(String(schema_version), intersections,
        Int(euler_characteristic), divisors, curves, FrozenArray(dmap), FrozenArray(cmap),
        FrozenArray(inequalities), cone_provenance, String(precision), exactness,
        String(units), source, history)
end

abstract type AbstractGeometryImporter end

"""Importer for a serialized CYTools/native payload; it never calls Python."""
struct NativePayloadImporter <: AbstractGeometryImporter end

function import_geometry(::NativePayloadImporter, payload::NamedTuple)
    required = (:intersections, :euler_characteristic, :ordered_divisors,
        :ordered_curves, :divisor_basis_map, :dual_curve_basis_map,
        :domain_inequalities, :cone_provenance, :precision, :exactness,
        :units, :source)
    all(key -> hasproperty(payload, key), required) ||
        throw(ArgumentError("serialized geometry payload is missing required fields"))
    GeometryRecord(payload.intersections;
        euler_characteristic=payload.euler_characteristic,
        ordered_divisors=payload.ordered_divisors,
        ordered_curves=payload.ordered_curves,
        divisor_basis_map=payload.divisor_basis_map,
        dual_curve_basis_map=payload.dual_curve_basis_map,
        domain_inequalities=payload.domain_inequalities,
        cone_provenance=payload.cone_provenance, precision=payload.precision,
        exactness=payload.exactness, units=payload.units, source=payload.source,
        schema_version=hasproperty(payload, :schema_version) ? payload.schema_version : "cyax0191-geometry-v1",
        basis_history=hasproperty(payload, :basis_history) ? payload.basis_history : ())
end

"""Return the dense symmetric tensor, retaining exact element type where possible."""
function _dense(tensor::CanonicalIntersectionTensor{T}) where {T}
    dense_intersections(tensor)
end

"""Calabi–Yau volume from the canonical sparse tensor."""
function calabi_yau_volume(geometry::GeometryRecord, t::AbstractVector)
    length(t) == geometry.intersections.n || throw(DimensionMismatch("wrong two-cycle vector length"))
    coordinates = _widen_exact_array(t)
    total = zero(coordinates[1]^3 * geometry.intersections.coefficients[1] * (1 // 6))
    for (i, j, k, value) in _full_entries(geometry.intersections)
        total += (value * coordinates[i] * coordinates[j] * coordinates[k]) * (1 // 6)
    end
    total
end

"""Divisor four-cycle volumes tau_i = 1/2 kappa_ijk t^j t^k."""
function divisor_volumes(geometry::GeometryRecord, t::AbstractVector)
    n = geometry.intersections.n
    length(t) == n || throw(DimensionMismatch("wrong two-cycle vector length"))
    coordinates = _widen_exact_array(t)
    zero_volume = zero(coordinates[1]^2 * geometry.intersections.coefficients[1] * (1 // 2))
    values = fill(zero_volume, n)
    for (i, j, k, value) in _full_entries(geometry.intersections)
        values[i] += (value * coordinates[j] * coordinates[k]) * (1 // 2)
    end
    values
end

"""Jacobian d tau_i/d t_j = kappa_ijk t^k."""
function divisor_volume_jacobian(geometry::GeometryRecord, t::AbstractVector)
    n = geometry.intersections.n
    length(t) == n || throw(DimensionMismatch("wrong two-cycle vector length"))
    coordinates = _widen_exact_array(t)
    J = fill(zero(coordinates[1] * geometry.intersections.coefficients[1]), n, n)
    for (i, j, k, value) in _full_entries(geometry.intersections)
        J[i, j] += value * coordinates[k]
    end
    J
end

"""Imported cone margins A*t. These are not physical curve volumes."""
function cone_margins(geometry::GeometryRecord, t::AbstractVector)
    inequalities = _widen_exact_array(geometry.domain_inequalities)
    coordinates = _widen_exact_array(t)
    inequalities * coordinates
end

function imported_domain_status(geometry::GeometryRecord, t::AbstractVector)
    margins = cone_margins(geometry, t)
    all(isfinite, margins) && all(>(zero(eltype(margins))), margins) ? :PASS : :FAIL
end

function geometry_identity(geometry::GeometryRecord)
    (; schema_version=geometry.schema_version,
       artifact_sha256=geometry.artifact_sha256,
       source_id=geometry.source.source_id,
       source_revision=geometry.source.source_revision,
       source_locator=geometry.source.source_locator,
       source_sha256=geometry.source.source_sha256,
       source_kind=geometry.source.source_kind,
       polytope_identity=geometry.source.polytope_identity,
       triangulation_identity=geometry.source.triangulation_identity,
       cytools_revision=geometry.source.cytools_revision,
       importer_id=geometry.source.importer_id,
       euler_characteristic=geometry.euler_characteristic,
       ordered_divisors=geometry.ordered_divisors,
       ordered_curves=geometry.ordered_curves,
       divisor_basis_map=copy(geometry.divisor_basis_map),
       dual_curve_basis_map=copy(geometry.dual_curve_basis_map),
       cone_status=geometry.cone_provenance.status,
       cone_construction=geometry.cone_provenance.construction,
       cone_normalization=geometry.cone_provenance.normalization,
       cone_source_locator=geometry.cone_provenance.source_locator,
       cone_completeness=geometry.cone_provenance.completeness,
       basis_history=geometry.basis_history,
       precision=geometry.precision, exactness=geometry.exactness,
       units=geometry.units)
end

"""Apply a unimodular divisor-basis change D_new = B*D_old."""
function change_divisor_basis(geometry::GeometryRecord, B::AbstractMatrix{<:Integer})
    n = geometry.intersections.n
    size(B) == (n, n) || throw(DimensionMismatch("basis change must be n×n"))
    basis = Matrix{BigInt}(B)
    Binv = _integer_inverse(basis)
    basis_exact = basis
    nonzero_rows = [findall(!iszero, view(basis, :, i)) for i in 1:n]
    transformed = Dict{NTuple{3,Int},Any}()
    for (i, j, k, value) in _full_entries(geometry.intersections)
        for a in nonzero_rows[i], b in nonzero_rows[j], c in nonzero_rows[k]
            key = (a, b, c)
            contribution = basis_exact[a, i] * basis_exact[b, j] * basis_exact[c, k] * value
            transformed[key] = get(transformed, key, zero(contribution)) + contribution
        end
    end
    terms = Pair{NTuple{3,Int},Any}[]
    for (key, value) in transformed
        key[1] <= key[2] <= key[3] && !iszero(value) && push!(terms, key => value)
    end
    tensor = CanonicalIntersectionTensor(n, terms)
    divisors = ["basis$(length(geometry.basis_history)+1)_D$i" for i in 1:n]
    curves = ["basis$(length(geometry.basis_history)+1)_C$i" for i in 1:n]
    history_entry = "D_new=$(repr(basis))*D_old"
    GeometryRecord(tensor;
        schema_version=geometry.schema_version,
        euler_characteristic=geometry.euler_characteristic,
        ordered_divisors=divisors, ordered_curves=curves,
        divisor_basis_map=basis_exact * Matrix{BigInt}(geometry.divisor_basis_map),
        dual_curve_basis_map=Binv' * Matrix{BigInt}(geometry.dual_curve_basis_map),
        domain_inequalities=_widen_exact_array(geometry.domain_inequalities) * basis_exact',
        cone_provenance=geometry.cone_provenance,
        precision=geometry.precision, exactness=geometry.exactness,
        units=geometry.units, source=geometry.source,
        basis_history=(geometry.basis_history..., history_entry))
end

"""A small exact native fixture used by Gate B and the local architecture tests."""
function synthetic_geometry_fixture()
    locator = "src/research/cyax0191/fixtures/cyax0191-two-modulus-v1.txt"
    fixture_path = joinpath(@__DIR__, "fixtures", "cyax0191-two-modulus-v1.txt")
    fixture_bytes = read(fixture_path)
    fixture_sha256 = bytes2hex(sha256(fixture_bytes))
    fields = Dict{String,String}()
    for line in split(String(copy(fixture_bytes)), '\n')
        isempty(strip(line)) && continue
        key, value = split(line, '='; limit=2)
        fields[key] = value
    end
    pair_values(key) = parse.(Int, split(fields[key], ','))
    tensor = CanonicalIntersectionTensor(2, Pair[
        (1, 1, 1) => parse(Int, fields["kappa_111"]),
        (1, 1, 2) => parse(Int, fields["kappa_112"]),
        (1, 2, 2) => parse(Int, fields["kappa_122"]),
        (2, 2, 2) => parse(Int, fields["kappa_222"]),
    ])
    source = GeometrySourceIdentity(source_kind=:native_fixture,
        source_id=fields["source_id"], source_revision=fields["source_revision"],
        source_locator=locator, source_sha256=fixture_sha256,
        polytope_identity="not_applicable:synthetic-two-modulus-fixture-v1",
        triangulation_identity="not_applicable:synthetic-two-modulus-fixture-v1",
        cytools_revision=nothing, importer_id="CYAX0191.NativePayloadImporter/v1")
    cone = ConeProvenance(Symbol(fields["cone_status"]), fields["cone_construction"],
        fields["cone_normalization"], Symbol(fields["cone_completeness"]), locator)
    payload = (; intersections=tensor,
        euler_characteristic=parse(Int, fields["euler_characteristic"]),
        ordered_divisors=Tuple(split(fields["ordered_divisors"], ',')),
        ordered_curves=Tuple(split(fields["ordered_curves"], ',')),
        divisor_basis_map=Matrix{Int}(I, 2, 2),
        dual_curve_basis_map=Matrix{Int}(I, 2, 2),
        domain_inequalities=permutedims(hcat(pair_values("domain_row_1"),
            pair_values("domain_row_2"))), cone_provenance=cone,
        precision="exact integer intersections; exact rational coordinates supported",
        exactness=:exact, units="dimensionless synthetic geometry",
        source=source, schema_version="cyax0191-geometry-v1")
    import_geometry(NativePayloadImporter(), payload)
end
