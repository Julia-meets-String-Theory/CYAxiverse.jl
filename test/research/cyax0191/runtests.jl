using LinearAlgebra
using SHA
using Test

include(joinpath(@__DIR__, "../../../src/research/cyax0191/CYAX0191.jl"))
using .CYAX0191

mutable struct MutableManifestString <: AbstractString
    value::String
end

Base.ncodeunits(value::MutableManifestString) = ncodeunits(value.value)
Base.codeunit(::Type{MutableManifestString}) = UInt8
Base.codeunit(value::MutableManifestString, index::Integer) = codeunit(value.value, index)
Base.isvalid(value::MutableManifestString, index::Integer) = isvalid(value.value, index)
Base.iterate(value::MutableManifestString) = iterate(value.value)
Base.iterate(value::MutableManifestString, state::Integer) = iterate(value.value, state)
Base.length(value::MutableManifestString) = length(value.value)
Base.getindex(value::MutableManifestString, index::Integer) = getindex(value.value, index)

struct DomainRejectingBackend <: DifferentiationBackend end

function CYAX0191.gradient!(out::AbstractVector{T}, ::DomainRejectingBackend, f,
        x::AbstractVector{T}; scales=ones(T, length(x))) where {T<:AbstractFloat}
    f(x)
    out[1] = x[1] - T(2)
    out
end

function CYAX0191.hessian!(out::AbstractMatrix{T}, ::DomainRejectingBackend, f,
        x::AbstractVector{T}; scales=ones(T, length(x))) where {T<:AbstractFloat}
    f(x)
    fill!(out, zero(T))
    out[1, 1] = one(T)
    out
end

function fixture_convention()
    ModelConvention(
        coordinate_map="T_paper=rho-i*tau; T_Julia=tau+i*rho=i*T_paper",
        axion_periodicity="synthetic 2pi representatives; no condensate quotient",
        condensate_branch="explicit synthetic branch label",
        frame="Einstein frame assumption for synthetic fixture",
        length_and_alpha_prime_units="dimensionless fixture units with alpha-prime convention recorded",
        planck_normalization="M_Pl=1 synthetic convention",
        kcs_treatment="constant K_cs retained in absolute potential",
        active_fields=("tau", "rho"), frozen_fields=("S", "complex-structure moduli"),
        flux_assumptions=("D_S W0=0", "D_alpha W0=0"),
        heavy_field_reduction="full S-T Hermitian metric; retained inverse from Schur complement",
        source_convention="Issue 191 source model conventions; synthetic local fixture")
end

function fixture_model(n; charges=Matrix{Int}(I, n, n), w0=1.2, theta0=0.21,
        gs=0.2, kcs=0.4, amplitudes=fill(0.0, size(charges, 1)),
        actions=fill(0.8, size(charges, 1)), phases=zeros(size(charges, 1)),
        switches=ModelSwitches(), uplift=nothing, identity="gateb-synthetic-model-v1")
    KahlerModel(; w0_magnitude=w0, theta0=theta0, gs=gs, kcs=kcs,
        amplitudes, actions, phases, charges, convention=fixture_convention(),
        switches, uplift, identity)
end

function one_modulus_geometry(; kappa=1, chi=-2, inequality=reshape([1], 1, 1), id="one-modulus-fixture-v1")
    tensor = CanonicalIntersectionTensor(1, [(1, 1, 1) => kappa])
    source = GeometrySourceIdentity(source_kind=:native_fixture, source_id=id,
        source_revision="fixture-v1", source_locator="test/research/cyax0191/runtests.jl",
        source_sha256=bytes2hex(sha256(read(@__FILE__))),
        polytope_identity="not_applicable:one-modulus-synthetic-fixture",
        triangulation_identity="not_applicable:one-modulus-synthetic-fixture",
        cytools_revision=nothing,
        importer_id="CYAX0191.NativePayloadImporter/v1")
    cone = ConeProvenance(:torically_inferred, "one-modulus synthetic inequality",
        "unit row normalization", :not_established,
        "test/research/cyax0191/runtests.jl")
    GeometryRecord(tensor; euler_characteristic=chi, ordered_divisors=("D1",),
        ordered_curves=("C1",), divisor_basis_map=reshape([1], 1, 1),
        dual_curve_basis_map=reshape([1], 1, 1), domain_inequalities=inequality,
        cone_provenance=cone, precision="exact integer intersections",
        exactness=:exact, units="dimensionless synthetic geometry", source)
end

function geometry_with_source(geometry::GeometryRecord, source::GeometrySourceIdentity)
    GeometryRecord(geometry.intersections;
        euler_characteristic=geometry.euler_characteristic,
        ordered_divisors=geometry.ordered_divisors, ordered_curves=geometry.ordered_curves,
        divisor_basis_map=geometry.divisor_basis_map,
        dual_curve_basis_map=geometry.dual_curve_basis_map,
        domain_inequalities=geometry.domain_inequalities,
        cone_provenance=geometry.cone_provenance, precision=geometry.precision,
        exactness=geometry.exactness, units=geometry.units, source)
end

function geometry_with_intersections(geometry::GeometryRecord,
        intersections::CanonicalIntersectionTensor)
    GeometryRecord(intersections;
        euler_characteristic=geometry.euler_characteristic,
        ordered_divisors=geometry.ordered_divisors, ordered_curves=geometry.ordered_curves,
        divisor_basis_map=geometry.divisor_basis_map,
        dual_curve_basis_map=geometry.dual_curve_basis_map,
        domain_inequalities=geometry.domain_inequalities,
        cone_provenance=geometry.cone_provenance, precision=geometry.precision,
        exactness=geometry.exactness, units=geometry.units, source=geometry.source)
end

@testset "CYAX-0191 Gate B geometry and importer boundary" begin
    geometry = synthetic_geometry_fixture()
    source = geometry.source
    @test_throws ArgumentError GeometrySourceIdentity(:invalid, source.source_id,
        source.source_revision, source.source_locator, source.source_sha256,
        source.polytope_identity, source.triangulation_identity,
        source.cytools_revision, source.importer_id)
    @test_throws ArgumentError CanonicalIntersectionTensor(1,
        [(1, 1, 1) => ComplexF64(1)])
    @test_throws ArgumentError CanonicalIntersectionTensor{ComplexF64}(1,
        ((1, 1, 1),), CYAX0191.FrozenArray(ComplexF64[1 + 0im]))
    @test_throws ArgumentError GeometryRecord(geometry.intersections;
        euler_characteristic=geometry.euler_characteristic,
        ordered_divisors=geometry.ordered_divisors, ordered_curves=geometry.ordered_curves,
        divisor_basis_map=geometry.divisor_basis_map,
        dual_curve_basis_map=geometry.dual_curve_basis_map,
        domain_inequalities=complex.(geometry.domain_inequalities),
        cone_provenance=geometry.cone_provenance, precision=geometry.precision,
        exactness=geometry.exactness, units=geometry.units, source=geometry.source)
    duplicate_terms = CanonicalIntersectionTensor(1, Pair[
        (1, 1, 1) => typemax(Int), (1, 1, 1) => 1])
    @test collect(duplicate_terms.coefficients) == BigInt[BigInt(typemax(Int)) + 1]
    t_exact = Rational{Int}[2, 1]
    @test calabi_yau_volume(geometry, t_exact) == 23 // 3
    @test divisor_volumes(geometry, t_exact) == Rational{Int}[17 // 2, 6]
    @test dot(t_exact, divisor_volumes(geometry, t_exact)) ==
        3 * calabi_yau_volume(geometry, t_exact)
    tensor = dense_intersections(geometry.intersections)
    @test tensor == permutedims(tensor, (2, 1, 3))
    @test tensor == permutedims(tensor, (3, 2, 1))
    @test geometry_identity(geometry).source_sha256 == geometry.source.source_sha256
    @test geometry_identity(geometry).source_kind == :native_fixture
    fixture_path = joinpath(@__DIR__, "../../../src/research/cyax0191/fixtures/cyax0191-two-modulus-v1.txt")
    @test geometry.source.source_locator == "src/research/cyax0191/fixtures/cyax0191-two-modulus-v1.txt"
    @test geometry.source.source_sha256 == bytes2hex(sha256(read(fixture_path)))
    @test geometry.cone_provenance.source_locator == geometry.source.source_locator
    @test geometry_identity(geometry).cytools_revision === nothing
    @test geometry.exactness == :exact
    @test geometry.precision != ""
    @test geometry.cone_provenance.status == :torically_inferred
    @test geometry.cone_provenance.completeness == :not_established
    protected_digest = geometry.artifact_sha256
    protected_coefficient = geometry.intersections.coefficients[1]
    Base.GMP.MPZ.set!(protected_coefficient, BigInt(999))
    protected_coefficient_from_values =
        getfield(geometry.intersections.coefficients, :values)[1]
    @test protected_coefficient_from_values isa Tuple{Vararg{UInt8}}
    @test_throws MethodError Base.GMP.MPZ.set!(protected_coefficient_from_values, BigInt(998))
    protected_basis_entry = geometry.divisor_basis_map[1, 1]
    Base.GMP.MPZ.set!(protected_basis_entry, BigInt(77))
    protected_basis_entry_from_values = getfield(geometry.divisor_basis_map, :values)[1]
    @test protected_basis_entry_from_values isa Tuple{Vararg{UInt8}}
    @test_throws MethodError Base.GMP.MPZ.set!(protected_basis_entry_from_values, BigInt(76))
    @test_throws MethodError CYAX0191.FrozenArray{BigInt,1,1}((BigInt(3),), (1,))
    rational_frozen = CYAX0191.FrozenArray(reshape(Rational{BigInt}[1 // 3], 1, 1))
    @test rational_frozen[1, 1] == BigInt(1) // BigInt(3)
    setprecision(BigFloat, 256) do
        bigfloat_value = BigFloat(1) / BigFloat(7)
        bigfloat_frozen = CYAX0191.FrozenArray([bigfloat_value])
        @test bigfloat_frozen[1] == bigfloat_value
        @test precision(bigfloat_frozen[1]) == 256
        @test getfield(bigfloat_frozen, :values)[1][1] == 256
        @test getfield(bigfloat_frozen, :values)[1][2] isa Tuple{Vararg{UInt8}}
    end
    high_precision_coefficient, high_precision_coordinate = setprecision(BigFloat, 512) do
        (BigFloat(1) / BigFloat(7), BigFloat(1) / BigFloat(3))
    end
    high_precision_geometry_base = one_modulus_geometry(id="bigfloat-intersection-precision-v1")
    high_precision_tensor = setprecision(BigFloat, 512) do
        CanonicalIntersectionTensor(1, Pair[(1, 1, 1) => high_precision_coefficient])
    end
    high_precision_geometry = geometry_with_intersections(high_precision_geometry_base,
        high_precision_tensor)
    expected_duplicate_coefficient = setprecision(BigFloat, 512) do
        high_precision_coefficient + high_precision_coefficient
    end
    expected_volume = calabi_yau_volume(high_precision_geometry,
        [high_precision_coordinate])
    expected_tau = divisor_volumes(high_precision_geometry,
        [high_precision_coordinate])[1]
    expected_jacobian = divisor_volume_jacobian(high_precision_geometry,
        [high_precision_coordinate])[1, 1]
    high_precision_cone_coordinates = setprecision(BigFloat, 512) do
        BigFloat[BigFloat(1) / 3, BigFloat(1) / 5]
    end
    expected_cone_margins = cone_margins(geometry, high_precision_cone_coordinates)
    setprecision(BigFloat, 128) do
        imported_tensor = CanonicalIntersectionTensor(1,
            Pair[(1, 1, 1) => high_precision_coefficient])
        imported_coefficient = imported_tensor.coefficients[1]
        @test imported_coefficient == high_precision_coefficient
        @test precision(imported_coefficient) == 512
        duplicate_tensor = CanonicalIntersectionTensor(1, Pair[
            (1, 1, 1) => high_precision_coefficient,
            (1, 1, 1) => high_precision_coefficient])
        @test duplicate_tensor.coefficients[1] == expected_duplicate_coefficient
        @test precision(duplicate_tensor.coefficients[1]) == 512
        imported_geometry = geometry_with_intersections(high_precision_geometry_base,
            imported_tensor)
        @test imported_geometry.artifact_sha256 == high_precision_geometry.artifact_sha256
        @test imported_geometry.intersections.coefficients[1] == high_precision_coefficient
        @test precision(calabi_yau_volume(imported_geometry,
            [high_precision_coordinate])) == 512
        @test calabi_yau_volume(imported_geometry,
            [high_precision_coordinate]) == expected_volume
        tau = divisor_volumes(imported_geometry, [high_precision_coordinate])
        @test precision(tau[1]) == 512
        @test tau[1] == expected_tau
        jacobian = divisor_volume_jacobian(imported_geometry,
            [high_precision_coordinate])
        @test precision(jacobian[1, 1]) == 512
        @test jacobian[1, 1] == expected_jacobian
        rebased = change_divisor_basis(imported_geometry, reshape([1], 1, 1))
        @test rebased.intersections.coefficients[1] == high_precision_coefficient
        @test precision(rebased.intersections.coefficients[1]) == 512
        margins = cone_margins(geometry, high_precision_cone_coordinates)
        @test all(value -> precision(value) == 512, margins)
        @test margins == expected_cone_margins
    end
    @test geometry.artifact_sha256 == protected_digest
    @test geometry.intersections.coefficients[1] == 3
    @test geometry.divisor_basis_map[1, 1] == 1
    caller_basis = BigInt[1 0; 0 1]
    copied_geometry = GeometryRecord(geometry.intersections;
        euler_characteristic=geometry.euler_characteristic,
        ordered_divisors=geometry.ordered_divisors, ordered_curves=geometry.ordered_curves,
        divisor_basis_map=caller_basis, dual_curve_basis_map=caller_basis,
        domain_inequalities=geometry.domain_inequalities,
        cone_provenance=geometry.cone_provenance, precision=geometry.precision,
        exactness=geometry.exactness, units=geometry.units, source=geometry.source)
    Base.GMP.MPZ.set!(caller_basis[1, 1], BigInt(7))
    @test copied_geometry.divisor_basis_map[1, 1] == 1
    @test cone_margins(geometry, [2.0, 1.0]) == [3.0, 1.0]
    @test imported_domain_status(geometry, [2.0, 1.0]) == :PASS
    large_exact_coordinates = [2_000_000, 1]
    @test calabi_yau_volume(geometry, large_exact_coordinates) ==
        parse(BigInt, "12000006000003000002") // BigInt(3)
    @test divisor_volumes(geometry, large_exact_coordinates) == Rational{BigInt}[
        parse(BigInt, "12000004000001") // BigInt(2), BigInt(2_000_002_000_002) // BigInt(1)]
    cone_edge = [typemax(Int), typemax(Int) - 1]
    @test cone_margins(geometry, cone_edge) == BigInt[
        2BigInt(typemax(Int)) - 1, 1]
    @test imported_domain_status(geometry, cone_edge) == :PASS

    # D_new = B*D_old leaves the same point inside the imported cone, even
    # though one native two-cycle coordinate is negative.
    B = Int[1 3; 0 1]
    t_old, rho_old = [2.0, 1.0], [0.31, -0.27]
    t_new, rho_new = change_coordinate_basis(t_old, rho_old, B)
    transformed = change_divisor_basis(geometry, B)
    @test t_new[2] < 0
    @test imported_domain_status(transformed, t_new) == :PASS
    @test calabi_yau_volume(transformed, t_new) ≈ calabi_yau_volume(geometry, t_old)
    @test divisor_volumes(transformed, t_new) ≈ B * divisor_volumes(geometry, t_old)
    @test divisor_volume_jacobian(transformed, t_new) ≈
        B * divisor_volume_jacobian(geometry, t_old) * B'
    @test transformed.divisor_basis_map == B
    @test transformed.dual_curve_basis_map == inv(B)'
    @test transformed.divisor_basis_map' * transformed.dual_curve_basis_map == I
    @test transformed.artifact_sha256 != geometry.artifact_sha256
    B2 = Int[1 0; 2 1]
    sequential = change_divisor_basis(transformed, B2)
    combined = B2 * B
    t_sequential, rho_sequential = change_coordinate_basis(t_new, rho_new, B2)
    @test sequential.divisor_basis_map == combined
    @test sequential.dual_curve_basis_map == inv(combined)'
    @test sequential.divisor_basis_map' * sequential.dual_curve_basis_map == I
    large_shear_value = BigInt(5_000_000_000)
    large_shear_first = BigInt[1 large_shear_value; 0 1]
    large_shear_second = BigInt[1 0; large_shear_value 1]
    large_shear_combined = large_shear_second * large_shear_first
    large_shear_geometry = change_divisor_basis(
        change_divisor_basis(geometry, large_shear_first), large_shear_second)
    @test maximum(abs, large_shear_combined) > BigInt(typemax(Int))
    @test Matrix{BigInt}(large_shear_geometry.divisor_basis_map) == large_shear_combined
    @test Matrix{BigInt}(large_shear_geometry.dual_curve_basis_map) ==
        CYAX0191._integer_inverse(large_shear_combined)'
    @test large_shear_geometry.divisor_basis_map' *
        large_shear_geometry.dual_curve_basis_map == Matrix{BigInt}(I, 2, 2)
    beyond_int_basis = BigInt[1 BigInt(typemax(Int)) + 1; 0 1]
    beyond_int_geometry = change_divisor_basis(geometry, beyond_int_basis)
    @test Matrix{BigInt}(beyond_int_geometry.divisor_basis_map) == beyond_int_basis
    @test occursin(string(beyond_int_basis), last(beyond_int_geometry.basis_history))
    @test calabi_yau_volume(sequential, t_sequential) ≈ calabi_yau_volume(geometry, t_old)
    @test divisor_volumes(sequential, t_sequential) ≈ combined * divisor_volumes(geometry, t_old)
    large_basis = Int[10_000_000 1; 1 0]
    large_basis_geometry = change_divisor_basis(geometry, large_basis)
    k111_index = findfirst(==((1, 1, 1)), large_basis_geometry.intersections.triples)
    @test k111_index !== nothing
    @test large_basis_geometry.intersections.coefficients[k111_index] ==
        parse(BigInt, "3000000300000030000004")
    @test_throws Base.CanonicalIndexError setindex!(geometry.intersections.coefficients, 9, 1)
    @test_throws Base.CanonicalIndexError setindex!(geometry.divisor_basis_map, 9, 1, 1)
    @test_throws Base.CanonicalIndexError setindex!(geometry.domain_inequalities, 9, 1, 1)
    @test_throws ArgumentError GeometrySourceIdentity(source_kind=:cytools_export,
        source_id="bad", source_revision="r1", source_locator="fixture",
        source_sha256=repeat("0", 64), cytools_revision=nothing, importer_id="test")

    alternate_source = GeometrySourceIdentity(source_kind=:other,
        source_id=geometry.source.source_id,
        source_revision=geometry.source.source_revision,
        source_locator=geometry.source.source_locator,
        source_sha256=geometry.source.source_sha256,
        polytope_identity=geometry.source.polytope_identity,
        triangulation_identity=geometry.source.triangulation_identity,
        cytools_revision=nothing, importer_id=geometry.source.importer_id)
    alternate_kind = geometry_with_source(geometry, alternate_source)
    @test alternate_kind.artifact_sha256 != geometry.artifact_sha256
    string_revision_source = GeometrySourceIdentity(source_kind=:native_fixture,
        source_id=geometry.source.source_id,
        source_revision=geometry.source.source_revision,
        source_locator=geometry.source.source_locator,
        source_sha256=geometry.source.source_sha256,
        polytope_identity=geometry.source.polytope_identity,
        triangulation_identity=geometry.source.triangulation_identity,
        cytools_revision="nothing", importer_id=geometry.source.importer_id)
    string_revision_geometry = geometry_with_source(geometry, string_revision_source)
    @test string_revision_geometry.artifact_sha256 != geometry.artifact_sha256
    collision_left_source = GeometrySourceIdentity(source_kind=:native_fixture,
        source_id="A|B", source_revision="C", source_locator=geometry.source.source_locator,
        source_sha256=geometry.source.source_sha256,
        polytope_identity=geometry.source.polytope_identity,
        triangulation_identity=geometry.source.triangulation_identity,
        cytools_revision=nothing, importer_id=geometry.source.importer_id)
    collision_right_source = GeometrySourceIdentity(source_kind=:native_fixture,
        source_id="A", source_revision="B|C", source_locator=geometry.source.source_locator,
        source_sha256=geometry.source.source_sha256,
        polytope_identity=geometry.source.polytope_identity,
        triangulation_identity=geometry.source.triangulation_identity,
        cytools_revision=nothing, importer_id=geometry.source.importer_id)
    @test geometry_with_source(geometry, collision_left_source).artifact_sha256 !=
        geometry_with_source(geometry, collision_right_source).artifact_sha256

    one_geometry = one_modulus_geometry()
    overflowing_dual_divisor = reshape([Int(4_294_967_297)], 1, 1)
    overflowing_dual_curve = reshape([Int(-4_294_967_295)], 1, 1)
    @test BigInt(overflowing_dual_divisor[1]) * BigInt(overflowing_dual_curve[1]) != 1
    @test_throws ArgumentError GeometryRecord(one_geometry.intersections;
        euler_characteristic=one_geometry.euler_characteristic,
        ordered_divisors=one_geometry.ordered_divisors,
        ordered_curves=one_geometry.ordered_curves,
        divisor_basis_map=overflowing_dual_divisor,
        dual_curve_basis_map=overflowing_dual_curve,
        domain_inequalities=one_geometry.domain_inequalities,
        cone_provenance=one_geometry.cone_provenance,
        precision=one_geometry.precision, exactness=one_geometry.exactness,
        units=one_geometry.units, source=one_geometry.source)

    geometry = synthetic_geometry_fixture()
    K = eltype(geometry.intersections.coefficients)
    C = eltype(geometry.domain_inequalities)
    malformed_cone = CYAX0191.FrozenArray(reshape([1, 0, 0], 1, 3))
    @test_throws DimensionMismatch GeometryRecord{K,C}(
        geometry.schema_version, geometry.intersections, geometry.euler_characteristic,
        geometry.ordered_divisors, geometry.ordered_curves, geometry.divisor_basis_map,
        geometry.dual_curve_basis_map, malformed_cone, geometry.cone_provenance,
        geometry.precision, geometry.exactness, geometry.units, geometry.source,
        geometry.basis_history)
    complex_cone = CYAX0191.FrozenArray(ComplexF64.(geometry.domain_inequalities))
    @test_throws ArgumentError GeometryRecord{K,ComplexF64}(
        geometry.schema_version, geometry.intersections, geometry.euler_characteristic,
        geometry.ordered_divisors, geometry.ordered_curves, geometry.divisor_basis_map,
        geometry.dual_curve_basis_map, complex_cone, geometry.cone_provenance,
        geometry.precision, geometry.exactness, geometry.units, geometry.source,
        geometry.basis_history)

    large_n = 24
    sparse_tensor = CanonicalIntersectionTensor(large_n,
        Pair[(1, 1, 1) => 2, (large_n, large_n, large_n) => 3])
    large_source = GeometrySourceIdentity(source_kind=:native_fixture,
        source_id="sparse-basis-transform-fixture-v1", source_revision="fixture-v1",
        source_locator="test/research/cyax0191/runtests.jl",
        source_sha256=bytes2hex(sha256(read(@__FILE__))),
        importer_id="CYAX0191.NativePayloadImporter/v1")
    large_cone = ConeProvenance(:torically_inferred, "sparse test cone",
        "unit row normalization", :not_established, large_source.source_locator)
    large_geometry = GeometryRecord(sparse_tensor;
        euler_characteristic=-2,
        ordered_divisors=Tuple("D$i" for i in 1:large_n),
        ordered_curves=Tuple("C$i" for i in 1:large_n),
        divisor_basis_map=Matrix{Int}(I, large_n, large_n),
        dual_curve_basis_map=Matrix{Int}(I, large_n, large_n),
        domain_inequalities=Matrix{Int}(I, large_n, large_n),
        cone_provenance=large_cone, precision="exact sparse fixture", exactness=:exact,
        units="dimensionless synthetic geometry", source=large_source)
    sparse_B = Matrix{Int}(I, large_n, large_n)
    sparse_B[2, 1] = 1
    sparse_changed = change_divisor_basis(large_geometry, sparse_B)
    @test length(sparse_changed.intersections.triples) == 5
    @test calabi_yau_volume(sparse_changed, inv(sparse_B)' * ones(large_n)) ≈
        calabi_yau_volume(large_geometry, ones(large_n))
end

@testset "CYAX-0191 Gate B no-scale, BBHL, and full-metric discriminator" begin
    convention = fixture_convention()
    @test_throws ArgumentError ModelConvention("", convention.axion_periodicity,
        convention.condensate_branch, convention.frame,
        convention.length_and_alpha_prime_units, convention.planck_normalization,
        convention.kcs_treatment, convention.active_fields, convention.frozen_fields,
        convention.flux_assumptions, convention.heavy_field_reduction,
        convention.source_convention)
    geometry = one_modulus_geometry()
    common = (; w0=1.0, theta0=0.0, gs=0.5, kcs=0.1,
        amplitudes=[0.0], actions=[1.0], phases=[0.0], charges=reshape([1], 1, 1))
    corrected_model = fixture_model(1; common...,
        switches=ModelSwitches(bbhl_correction_enabled=true,
            np_linear_enabled=false, np_quadratic_enabled=false))
    t, rho = [1.0], [0.0]
    corrected = evaluate_potential(corrected_model, geometry, t, rho)
    data = CYAX0191._kahler_data(corrected_model, geometry, t)
    x = data.xihat / data.V
    c_full = 3x * (1 + 7x + x^2) / ((1 - x) * (2 + x)^2)
    c_frozen = 3x / (4 - x)
    c_difference = 81x^2 / ((4 - x) * (1 - x) * (2 + x)^2)
    full_from_metric = data.k_t[1]^2 * data.full_inverse_tt_metric[1, 1] - 3
    frozen_from_metric = data.k_t[1]^2 / data.parent_metric[2, 2] - 3
    @test full_from_metric ≈ c_full rtol=1e-11 atol=1e-12
    @test frozen_from_metric ≈ c_frozen rtol=1e-11 atol=1e-12
    @test full_from_metric - frozen_from_metric ≈ c_difference rtol=1e-10 atol=1e-12
    @test !isapprox(frozen_from_metric, c_full; rtol=1e-6, atol=1e-9)
    @test corrected.value ≈ data.expK * corrected_model.w0_magnitude^2 * c_full
    n = 3x * (1 + 7x + x^2)
    d = (1 - x) * (2 + x)^2
    nprime = 3 + 42x + 9x^2
    dprime = -6x - 3x^2
    c_full_prime = (nprime * d - n * dprime) / d^2
    x_tau = -x * data.tau[1] / data.V
    kt = -2data.tau[1] / data.Y
    independent_t_derivative = data.expK * corrected_model.w0_magnitude^2 *
        (kt * c_full + c_full_prime * x_tau)
    numerical_t_derivative = finite_difference_gradient(CentralDifferenceBackend(),
        tvar -> potential(corrected_model, geometry, tvar, rho), t)[1]
    @test numerical_t_derivative ≈ independent_t_derivative rtol=2e-6 atol=2e-9
    c_frozen_prime = 12 / (4 - x)^2
    frozen_t_derivative = data.expK * corrected_model.w0_magnitude^2 *
        (kt * c_frozen + c_frozen_prime * x_tau)
    @test !isapprox(numerical_t_derivative, frozen_t_derivative; rtol=1e-3, atol=1e-8)
    @test corrected.parent_metric_assessment.status == :PASS
    @test corrected.retained_metric_assessment.status == :PASS
    @test abs(corrected.parent_metric[1, 2]) > 1e-8
    @test corrected.retained_kinetic_metric ≈ corrected.parent_metric[2:2, 2:2]
    @test corrected.retained_inverse_metric ≈ inv(data.schur_complement_metric)
    @test data.schur_complement_metric ≈
        data.retained_kinetic_metric - data.parent_metric[2:2, 1:1] *
        data.parent_metric[1:1, 2:2] / data.parent_metric[1, 1]
    @test !isapprox(corrected.retained_inverse_metric,
        inv(corrected.retained_kinetic_metric); rtol=1e-8, atol=1e-12)
    @test size(corrected.parent_metric) == (2, 2)
    @test size(corrected.retained_kinetic_metric) == (1, 1)
    @test size(corrected.retained_real_kinetic_metric) == (2, 2)
    @test corrected.retained_real_kinetic_metric ≈
        2 .* Matrix{Float64}(I, 2, 2) .* corrected.retained_kinetic_metric[1, 1]
    nonunit_eval = evaluate_potential(corrected_model, geometry, [1.2], rho)
    nonunit_jacobian = divisor_volume_jacobian(geometry, [1.2])
    @test nonunit_eval.retained_real_kinetic_metric[1, 1] /
        nonunit_eval.retained_real_kinetic_metric[2, 2] ≈ 1.2^2
    @test nonunit_eval.retained_real_kinetic_metric[1, 1] ≈
        2 * nonunit_jacobian[1, 1]^2 * nonunit_eval.retained_kinetic_metric[1, 1]
    @test nonunit_eval.retained_real_kinetic_metric[2, 2] ≈
        2 * nonunit_eval.retained_kinetic_metric[1, 1]
    nonunit_masses = fluctuation_analysis(GeneralizedEigenBackend(),
        Matrix{Float64}(I, 2, 2), nonunit_eval.retained_real_kinetic_metric)
    expected_nonunit_masses = sort(inv.(diag(nonunit_eval.retained_real_kinetic_metric)))
    @test nonunit_masses.generalized_mass_eigenvalues ≈ expected_nonunit_masses

    uncorrected_model = fixture_model(1; common...,
        switches=ModelSwitches(bbhl_correction_enabled=false,
            np_linear_enabled=false, np_quadratic_enabled=false))
    uncorrected = evaluate_potential(uncorrected_model, geometry, t, rho)
    @test uncorrected.value ≈ 0 atol=2e-14
    @test uncorrected.xi == corrected.xi
    @test uncorrected.xihat == corrected.xihat
    @test uncorrected.xihat_over_two == corrected.xihat_over_two
    @test uncorrected.Y == uncorrected.volume
    @test corrected.Y == corrected.volume + corrected.xihat_over_two

    # Turning both nonperturbative contributions off is the no-scale limit.
    no_np = evaluate_potential(corrected_model, geometry, t, rho)
    @test no_np.active_contributions.np_linear == 0
    @test no_np.active_contributions.np_quadratic == 0
    @test_throws ArgumentError fixture_model(1; common...,
        switches=ModelSwitches(uplift_enabled=true))
    mutable_uplift_capture = [0.125]
    mutable_uplift = args -> mutable_uplift_capture[1]
    @test_throws ArgumentError UpliftSpec(mutable_uplift,
        "mutable synthetic uplift", "must not retain mutable state")
    mutable_uplift_ref = Ref(0.125)
    mutable_ref_uplift = args -> mutable_uplift_ref[]
    @test_throws ArgumentError UpliftSpec(mutable_ref_uplift,
        "mutable Ref uplift", "must not retain mutable state")
    @test_throws ArgumentError UpliftSpec(42,
        "non-callable uplift", "must fail at construction")
    @test_throws ArgumentError fixture_model(1; common...,
        switches=ModelSwitches(bbhl_correction_enabled=false,
            np_linear_enabled=false, np_quadratic_enabled=false, uplift_enabled=true),
        uplift=42)
    uplift = UpliftSpec(args -> 0.125, "constant synthetic uplift", "Gate B switch fixture")
    uplift_model = fixture_model(1; common...,
        switches=ModelSwitches(bbhl_correction_enabled=false,
            np_linear_enabled=false, np_quadratic_enabled=false, uplift_enabled=true),
        uplift)
    uplift_value = evaluate_potential(uplift_model, geometry, t, rho)
    @test uplift_value.contributions.optional_uplift == 0.125
    @test uplift_value.value ≈ 0.125

    @test_throws ArgumentError fixture_model(1; merge(common, (kcs=Inf,))...)
    @test_throws ArgumentError fixture_model(1; merge(common, (theta0=NaN,))...)
    @test_throws ArgumentError fixture_model(1; merge(common, (amplitudes=[Inf],))...)
    overflowing_model = fixture_model(1; w0=1e308,
        switches=ModelSwitches(bbhl_correction_enabled=true,
            np_linear_enabled=false, np_quadratic_enabled=false))
    @test_throws DomainError evaluate_potential(overflowing_model, one_modulus_geometry(),
        [1.0], [0.0])

    split_model = fixture_model(1; w0=1.0, theta0=0.2, gs=0.5, kcs=0.1,
        amplitudes=[0.4], actions=[0.7], phases=[0.3], charges=reshape([1], 1, 1))
    split = evaluate_potential(split_model, geometry, [1.0], [0.4])
    linear_only = evaluate_potential(fixture_model(1; w0=1.0, theta0=0.2,
        gs=0.5, kcs=0.1, amplitudes=[0.4], actions=[0.7], phases=[0.3],
        charges=reshape([1], 1, 1), switches=ModelSwitches(
            bbhl_correction_enabled=true, np_linear_enabled=true,
            np_quadratic_enabled=false)), geometry, [1.0], [0.4])
    quadratic_only = evaluate_potential(fixture_model(1; w0=1.0, theta0=0.2,
        gs=0.5, kcs=0.1, amplitudes=[0.4], actions=[0.7], phases=[0.3],
        charges=reshape([1], 1, 1), switches=ModelSwitches(
            bbhl_correction_enabled=true, np_linear_enabled=false,
            np_quadratic_enabled=true)), geometry, [1.0], [0.4])
    @test split.value ≈ linear_only.active_contributions.alpha3 +
        linear_only.active_contributions.np_linear +
        quadratic_only.active_contributions.np_quadratic
    @test split.contributions.np_linear != 0
    @test split.contributions.np_quadratic != 0
end

@testset "CYAX-0191 Gate B derivative and phase oracles" begin
    geometry = synthetic_geometry_fixture()
    model = fixture_model(2; charges=Matrix{Int}(I, 2, 2), w0=1.3,
        theta0=0.19, gs=0.2, kcs=0.4, amplitudes=[0.8, 0.55],
        actions=[0.7, 1.1], phases=[0.37, -0.23])
    t, rho = [2.0, 1.0], [0.31, -0.27]
    analytic = analytic_axion_gradient(model, geometry, t, rho)
    oracle = finite_difference_gradient(CentralDifferenceBackend(),
        r -> potential(model, geometry, t, r), rho)
    @test analytic ≈ oracle rtol=2e-6 atol=2e-9

    i, j = 1, 2
    adopted = quadratic_phase(model.actions[i], rho[i], model.phases[i],
        model.actions[j], rho[j], model.phases[j])
    direct = direct_complex_interference_phase(model.actions[i],
        model.charges[i, :], rho, model.phases[i], model.actions[j],
        model.charges[j, :], model.phases[j])
    printed_equal_index = model.actions[i] * rho[i] - model.actions[j] * rho[j]
    tau = divisor_volumes(geometry, t)
    z_i = model.amplitudes[i] * exp(complex(
        -model.actions[i] * dot(model.charges[i, :], tau),
        model.phases[i] - model.actions[i] * dot(model.charges[i, :], rho)))
    z_j = model.amplitudes[j] * exp(complex(
        -model.actions[j] * dot(model.charges[j, :], tau),
        model.phases[j] - model.actions[j] * dot(model.charges[j, :], rho)))
    direct_cross = real(z_i * conj(z_j))
    closed_form_cross = abs(z_i) * abs(z_j) * cos(adopted)
    printed_cross = abs(z_i) * abs(z_j) * cos(printed_equal_index)
    @test cos(adopted) ≈ cos(direct) atol=1e-14
    @test direct_cross ≈ closed_form_cross rtol=1e-14 atol=1e-15
    @test abs(direct_cross - printed_cross) > 1e-3 * abs(z_i * z_j)
    @test model.phases[i] != 0 && model.phases[j] != 0 && rho != zeros(2)
end

@testset "CYAX-0191 Gate B generic-Q oracle and basis covariance" begin
    geometry = one_modulus_geometry(id="generic-charge-one-modulus-v1")
    source_model = source_basis_model(1; w0_magnitude=0.83,
        theta0=0.2, gs=0.4, kcs=0.3, amplitudes=[0.6], actions=[0.9],
        phases=[0.17], convention=fixture_convention())
    generic_model = fixture_model(1; charges=reshape([1, 2], 2, 1), w0=0.83,
        theta0=0.2, gs=0.4, kcs=0.3, amplitudes=[0.6, 0.25],
        actions=[0.9, 0.7], phases=[0.17, -0.31])
    t, rho = [1.2], [0.43]
    evaluation = evaluate_potential(generic_model, geometry, t, rho)
    data = CYAX0191._kahler_data(generic_model, geometry, t)
    w0 = generic_model.w0_magnitude * cis(generic_model.theta0)
    tau = data.tau[1]
    z1 = generic_model.amplitudes[1] * exp(complex(
        -generic_model.actions[1] * tau,
        generic_model.phases[1] - generic_model.actions[1] * rho[1]))
    z2 = generic_model.amplitudes[2] * exp(complex(
        -2generic_model.actions[2] * tau,
        generic_model.phases[2] - 2generic_model.actions[2] * rho[1]))
    W = w0 + z1 + z2
    D = data.k_t[1] * W - generic_model.actions[1] * z1 -
        2generic_model.actions[2] * z2
    independent_q_oracle = data.expK *
        (data.full_inverse_tt_metric[1, 1] * abs2(D) - 3abs2(W))
    @test evaluation.value ≈ independent_q_oracle rtol=1e-13 atol=1e-14

    identity_eval = evaluate_potential(source_model, geometry, t, rho)
    single_z = source_model.amplitudes[1] * exp(complex(
        -source_model.actions[1] * tau,
        source_model.phases[1] - source_model.actions[1] * rho[1]))
    single_W = w0 + single_z
    single_D = data.k_t[1] * single_W - source_model.actions[1] * single_z
    source_basis_oracle = data.expK *
        (data.full_inverse_tt_metric[1, 1] * abs2(single_D) - 3abs2(single_W))
    @test identity_eval.value ≈ source_basis_oracle rtol=1e-13 atol=1e-14

    geometry2 = synthetic_geometry_fixture()
    Q = Int[1 0; 0 1; 1 1]
    model2 = fixture_model(2; charges=Q, w0=1.3, theta0=0.19,
        gs=0.2, kcs=0.4, amplitudes=[0.8, 0.55, 0.18],
        actions=[0.7, 1.1, 0.4], phases=[0.37, -0.23, 0.11])
    t_old, rho_old = [2.0, 1.0], [0.31, -0.27]
    B = Int[1 3; 0 1]
    geometry_new = change_divisor_basis(geometry2, B)
    model_new = change_model_basis(model2, B)
    t_new, rho_new = change_coordinate_basis(t_old, rho_old, B)
    tau_old = divisor_volumes(geometry2, t_old)
    tau_new = divisor_volumes(geometry_new, t_new)
    @test tau_new ≈ B * tau_old
    @test change_charge_basis(Q, B) == model_new.charges
    @test dot(Q[3, :], tau_old) ≈ dot(model_new.charges[3, :], tau_new)
    @test charge_coordinates(Q, tau_old, rho_old).tau ≈ Q * tau_old
    @test charge_metric_contraction(Q, Matrix{Float64}(I, 2, 2)) == Q * Q'
    large_charge = reshape([10_000_000_000], 1, 1)
    large_charged_coordinates = charge_coordinates(large_charge,
        BigInt[10_000_000_000], BigInt[0])
    @test large_charged_coordinates.tau == reshape([big(10)^20], 1)
    @test charge_metric_contraction(large_charge, reshape([1], 1, 1)) ==
        reshape([big(10)^20], 1, 1)

    charge_limit = BigInt(typemax(Int))
    overflowing_q = reshape([typemax(Int), 1], 1, 2)
    overflowing_basis = Int[1 0; -typemax(Int) 1]
    @test change_charge_basis(overflowing_q, overflowing_basis) ==
        reshape([2charge_limit, BigInt(1)], 1, 2)
    transformed_t, transformed_rho = change_coordinate_basis(
        [typemax(Int), 1], [0, typemax(Int)], overflowing_basis)
    @test transformed_t == BigInt[2charge_limit, 1]
    @test transformed_rho == BigInt[0, charge_limit]

    large_shear = Int[1 4_000_000_000 0; 0 1 4_000_000_000; 0 0 1]
    large_shear_inverse = CYAX0191._integer_inverse(large_shear)
    shear = BigInt(4_000_000_000)
    expected_shear_inverse = BigInt[1 -shear shear^2; 0 1 -shear; 0 0 1]
    @test large_shear_inverse == expected_shear_inverse
    @test BigInt.(large_shear) * large_shear_inverse == Matrix{BigInt}(I, 3, 3)
    @test change_charge_basis(Int[1 0 0], large_shear) ==
        reshape(expected_shear_inverse[1, :], 1, 3)
    shear_t, _ = change_coordinate_basis([1, 0, 0], [0, 0, 0], large_shear)
    @test shear_t == expected_shear_inverse[1, :]
    shear_model = fixture_model(3; charges=Matrix{Int}(I, 3, 3))
    @test change_model_basis(shear_model, large_shear).charges[1, :] ==
        expected_shear_inverse[1, :]
    beyond_int_model_basis = BigInt[1 BigInt(typemax(Int)) + 1; 0 1]
    beyond_int_model = change_model_basis(model2, beyond_int_model_basis)
    @test beyond_int_model.charges == change_charge_basis(Q, beyond_int_model_basis)
    @test occursin(string(beyond_int_model_basis[1, 2]), beyond_int_model.identity)
    @test potential(model2, geometry2, t_old, rho_old) ≈
        potential(model_new, geometry_new, t_new, rho_new) rtol=2e-12 atol=1e-13
    state = critical_point_state(geometry2, t_old, rho_old)
    @test state.t == t_old && state.tau ≈ tau_old && state.rho == rho_old
    @test state.geometry_artifact_sha256 == geometry2.artifact_sha256
    @test !(:stable in fieldnames(typeof(state)))
    @test !(:controlled in fieldnames(typeof(state)))

    f_old = x -> potential(model2, geometry2, view(x, 1:2), view(x, 3:4))
    f_new = x -> potential(model_new, geometry_new, view(x, 1:2), view(x, 3:4))
    x_old, x_new = [t_old; rho_old], [t_new; rho_new]
    grad_old = finite_difference_gradient(CentralDifferenceBackend(), f_old, x_old)
    grad_new = finite_difference_gradient(CentralDifferenceBackend(), f_new, x_new)
    Binv = inv(B)
    @test grad_new[1:2] ≈ B * grad_old[1:2] rtol=2e-5 atol=2e-8
    @test grad_new[3:4] ≈ Binv' * grad_old[3:4] rtol=2e-5 atol=2e-8
end

@testset "CYAX-0191 Gate B numerical interfaces, reports, and frozen policy" begin
    backend = CentralDifferenceBackend()
    @test_throws ArgumentError SearchCriteria{Float64}([1.0], 1.0, Inf,
        10, 0.5, 1e-5, nothing, "")
    source_scales = [1.0, 2.0]
    frozen_criteria = SearchCriteria{Float64}(source_scales, 1.0, 1e-8,
        10, 0.5, 1e-5, nothing, "")
    source_scales[1] = 99.0
    @test frozen_criteria.field_scales == [1.0, 2.0]
    @test getfield(frozen_criteria, :field_scales) isa CYAX0191.FrozenArray{Float64,1}
    @test_throws Base.CanonicalIndexError setindex!(frozen_criteria.field_scales, 7.0, 1)
    mutable_scale_capture = [1.0]
    mutable_scale_rule = x -> mutable_scale_capture[1]
    @test_throws ArgumentError SearchCriteria([1.0], 1.0, 1e-8;
        potential_scale_rule=mutable_scale_rule,
        potential_scale_rule_identity="mutable-scale-test")
    mutable_field_scale_capture = Ref(1.0)
    mutable_field_scale_rule = x -> [mutable_field_scale_capture[]]
    @test_throws ArgumentError SearchCriteria([1.0], 1.0, 1e-8;
        field_scale_rule=mutable_field_scale_rule,
        field_scale_rule_identity="mutable-field-scale-test")

    f(x) = x[1]^2 + 3x[1] * x[2] + 2x[2]^2
    point = [0.4, -0.7]
    grad = finite_difference_gradient(backend, f, point)
    hess = finite_difference_hessian(backend, f, point)
    @test grad ≈ [2point[1] + 3point[2], 3point[1] + 4point[2]] atol=1e-7
    @test hess ≈ [2.0 3.0; 3.0 4.0] atol=2e-5

    objective(x) = (x[1]^2 + 2x[2]^2) / 2
    criteria = SearchCriteria([1.0, 1.0], 1.0, 1e-7; max_iterations=20)
    search = search_stationary(DampedNewtonSearch(), objective, [2.0, -1.0], criteria)
    @test search.status == :converged
    @test search.point ≈ zeros(2) atol=1e-6
    @test search.scaled_residual <= criteria.stationarity_tolerance
    positive_domain_objective(x) = x[1] > 0 ? x[1]^2 :
        throw(DomainError(x[1], "coordinate must stay positive"))
    near_domain_search = search_stationary(DampedNewtonSearch(),
        positive_domain_objective, [1e-8],
        SearchCriteria([1.0], 1.0, 1e-8; max_iterations=10))
    @test near_domain_search.status == :failed
    @test near_domain_search.point == [1e-8]
    @test near_domain_search.value ≈ 1e-16
    @test isinf(near_domain_search.scaled_residual)
    @test any(occursin("rejected the current iterate", failure)
        for failure in near_domain_search.failures)

    hessian_domain_objective(x) = x[1] < 1 ? x[1]^2 :
        throw(DomainError(x[1], "outside test domain"))
    hessian_domain_search = search_stationary(DampedNewtonSearch(),
        hessian_domain_objective, [0.99999],
        SearchCriteria([1.0], 1.0, 1e-12; max_iterations=3))
    @test hessian_domain_search.status == :failed
    @test hessian_domain_search.point == [0.99999]
    @test hessian_domain_search.value ≈ 0.99999^2
    @test any(occursin("Hessian evaluation rejected", failure)
        for failure in hessian_domain_search.failures)
    nonfinite_objective(x) = iszero(x[1]) ? Inf : x[1]^2
    nonfinite_search = search_stationary(DampedNewtonSearch(), nonfinite_objective,
        [0.0], SearchCriteria([1.0], 1.0, 1e-8; max_iterations=2))
    @test nonfinite_search.status == :failed
    @test isinf(nonfinite_search.value)
    @test any(occursin("objective returned a nonfinite value", failure)
        for failure in nonfinite_search.failures)
    final_recheck_calls = Ref(0)
    function final_recheck_objective(x)
        final_recheck_calls[] += 1
        final_recheck_calls[] == 9 &&
            throw(DomainError(x[1], "unexpected post-budget reevaluation"))
        x[1] < 1 ? x[1]^2 : throw(DomainError(x[1], "outside test domain"))
    end
    exhausted_search = search_stationary(
        DampedNewtonSearch(differentiation=DomainRejectingBackend()),
        final_recheck_objective, [0.0], SearchCriteria([1.0], 1.0, 1e-8;
            max_iterations=1, minimum_step=1 / 8))
    @test exhausted_search.status == :failed
    @test exhausted_search.point == [0.5]
    @test final_recheck_calls[] == 8
    @test any(occursin("iteration budget exhausted", failure)
        for failure in exhausted_search.failures)

    fluct = fluctuation_analysis(GeneralizedEigenBackend(),
        [-0.5 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 5e-11 0.0; 0.0 0.0 0.0 2.0],
        [2.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0];
        absolute_zero_threshold=1e-10,
        relative_zero_threshold=1e-10, active_charges=Int[0 1])
    @test fluct.generalized_mass_eigenvalues ≈ [-0.25, 0.0, 5e-11, 2.0]
    @test [mode.disposition for mode in fluct.mode_dispositions] ==
        [:tachyonic, :numerically_unresolved_near_zero,
         :numerically_unresolved_near_zero, :lifted]
    @test [mode.sign_status for mode in fluct.mode_dispositions] ==
        [:negative, :zero_within_threshold, :zero_within_threshold, :positive]
    @test fluct.active_axionic_shift_directions ==
        reshape(Rational{BigInt}[0, 0, 1, 0], 4, 1)
    @test fluct.symmetry_kernel_assessment.status == :FAIL
    exact_shift = fluctuation_analysis(GeneralizedEigenBackend(),
        [1.0 0.0; 0.0 0.0], Matrix{Float64}(I, 2, 2);
        active_charges=zeros(Int, 0, 1), absolute_zero_threshold=1e-12)
    @test [mode.disposition for mode in exact_shift.mode_dispositions] ==
        [:symmetry_protected_exact_zero, :lifted]
    @test exact_shift.symmetry_kernel_assessment.status == :PASS
    @test exact_shift.mode_dispositions[1].disposition == :symmetry_protected_exact_zero
    @test exact_shift.mode_dispositions[1].sign_status == :zero_within_threshold
    charge_row = Int[1 3]
    charge_hessian = zeros(4, 4)
    charge_vector = Float64[1, 3]
    charge_hessian[3:4, 3:4] .= 0.1 .* (charge_vector * charge_vector')
    rounded_exact_shift = fluctuation_analysis(GeneralizedEigenBackend(),
        charge_hessian, Matrix{Float64}(I, 4, 4); active_charges=charge_row,
        absolute_zero_threshold=1e-12)
    @test rounded_exact_shift.symmetry_kernel_assessment.status == :PASS
    @test rounded_exact_shift.symmetry_mode_assessment.status == :PASS
    @test abs.(rounded_exact_shift.active_axionic_shift_directions) ==
        reshape(Rational{BigInt}[0, 0, 3, 1], 4, 1)
    @test all(mode -> mode.disposition != :symmetry_protected_exact_zero,
        rounded_exact_shift.mode_dispositions)
    @test any(mode -> mode.disposition == :numerically_unresolved_near_zero,
        rounded_exact_shift.mode_dispositions)
    for invalid_threshold in (NaN, Inf, -1.0)
        @test_throws ArgumentError fluctuation_analysis(GeneralizedEigenBackend(),
            Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2);
            absolute_zero_threshold=invalid_threshold)
        @test_throws ArgumentError fluctuation_analysis(GeneralizedEigenBackend(),
            Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2);
            relative_zero_threshold=invalid_threshold)
    end
    tiny_lift = fluctuation_analysis(GeneralizedEigenBackend(),
        [1.0 0.0; 0.0 1e-15], Matrix{Float64}(I, 2, 2);
        active_charges=zeros(Int, 0, 1), absolute_zero_threshold=1e-12)
    @test tiny_lift.symmetry_kernel_assessment.status == :FAIL
    @test tiny_lift.symmetry_mode_assessment.status == :NOT_ASSESSED
    @test tiny_lift.mode_dispositions[1].disposition == :numerically_unresolved_near_zero
    degenerate_shift = fluctuation_analysis(GeneralizedEigenBackend(),
        [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0],
        Matrix{Float64}(I, 4, 4); active_charges=Int[1 1])
    @test degenerate_shift.symmetry_kernel_assessment.status == :PASS
    @test degenerate_shift.symmetry_mode_assessment.status == :PASS
    @test degenerate_shift.active_axionic_shift_directions ==
        reshape(Rational{BigInt}[0, 0, -1, 1], 4, 1)
    @test [mode.disposition for mode in degenerate_shift.mode_dispositions] ==
        [:numerically_unresolved_near_zero, :numerically_unresolved_near_zero,
         :lifted, :lifted]
    policy_b1 = frozen_policy(:P0_B1)
    scaled_fluct = fluctuation_analysis(GeneralizedEigenBackend(),
        [1e-15 0.0; 0.0 1e-13], Matrix{Float64}(I, 2, 2);
        potential_scale=policy_b1.potential_scale,
        field_scales=policy_b1.field_scales,
        absolute_zero_threshold=policy_b1.absolute_spectral_threshold,
        relative_zero_threshold=policy_b1.relative_spectral_threshold)
    @test scaled_fluct.normalized_generalized_mass_eigenvalues ≈
        scaled_fluct.generalized_mass_eigenvalues ./ Float64(policy_b1.potential_scale)
    @test scaled_fluct.normalized_generalized_mass_eigenvalues[1] > 1e-10
    @test scaled_fluct.mode_dispositions[1].disposition == :lifted
    @test exact_axionic_shift_basis(Int[0 1]) ==
        reshape(Rational{BigInt}[1, 0], 2, 1)
    basis = Int[1 3; 0 1]
    kernel_old = exact_axionic_shift_basis(Int[1 0])
    transformed_charges = change_charge_basis(Int[1 0], basis)
    transformed_kernel = exact_axionic_shift_basis(transformed_charges)
    @test transformed_charges * transformed_kernel == zeros(Rational{BigInt}, 1, 1)
    @test transformed_kernel[:, 1] == basis * kernel_old[:, 1]
    @test exact_axionic_shift_basis(zeros(Int, 0, 2)) == Matrix{Rational{BigInt}}(I, 2, 2)
    @test fluct.physical_mass_assessment.status == :PASS
    @test fluct.bf_assessment.status == :NOT_APPLICABLE
    @test_throws ArgumentError fluctuation_analysis(GeneralizedEigenBackend(),
        Matrix{Float64}(I, 2, 2), Matrix{Float64}(I, 2, 2);
        at_critical_point=false)

    bounded_objective(x) = x[1] < 1 ? x[1]^2 : throw(DomainError(x[1], "outside test domain"))
    boundary_search = search_stationary(
        DampedNewtonSearch(differentiation=DomainRejectingBackend()),
        bounded_objective, [0.0], SearchCriteria([1.0], 1.0, 1e-8;
            max_iterations=1, minimum_step=1 / 8))
    @test boundary_search.status == :failed
    @test boundary_search.point == [0.5]
    @test any(occursin("domain rejected", failure) for failure in boundary_search.failures)

    rejecting_scale = x -> x[1] > 1 ? throw(DomainError(x[1], "scale outside test domain")) : 1.0
    scale_domain_search = search_stationary(
        DampedNewtonSearch(differentiation=DomainRejectingBackend()), x -> x[1]^2,
        [0.0], SearchCriteria([1.0], 1.0, 1e-8; max_iterations=1,
            minimum_step=1 / 8, potential_scale_rule=rejecting_scale,
            potential_scale_rule_identity="test-scale-domain-v1"))
    @test scale_domain_search.status == :failed
    @test scale_domain_search.point == [1.0]
    @test any(occursin("pointwise scale rejected", failure)
        for failure in scale_domain_search.failures)

    geometry = synthetic_geometry_fixture()
    model = fixture_model(2; charges=Matrix{Int}(I, 2, 2),
        amplitudes=[0.2, 0.3], actions=[0.7, 0.9], phases=[0.1, -0.2])
    @test_throws ArgumentError CYAX0191.KahlerModel{Float64,Nothing}(
        CYAX0191.FrozenScalar(-1.0), getfield(model, :theta0), getfield(model, :gs),
        getfield(model, :kcs), getfield(model, :amplitudes), getfield(model, :actions),
        getfield(model, :phases), getfield(model, :charges), model.convention,
        model.switches, nothing, model.identity)
    @test_throws ArgumentError CYAX0191.KahlerModel{Float64,Int}(
        getfield(model, :w0_magnitude), getfield(model, :theta0),
        getfield(model, :gs), getfield(model, :kcs), getfield(model, :amplitudes),
        getfield(model, :actions), getfield(model, :phases), getfield(model, :charges),
        model.convention, model.switches, 42, model.identity)
    manifest = replay_manifest(NativeReplayBackend(), model, geometry;
        code_revision="synthetic-code-revision", selected_source_route="analytic-fixture",
        counting_unit="one synthetic model state", numeric_type="Float64",
        precision_bits=53, backend_versions=(; differentiation="central-difference-v1"),
        solver_configuration=(; method="damped-newton", budget=20),
        scales=(; fields=(1.0, 1.0, 1.0, 1.0), potential=1.0), seed=17, budget=20)
    @test manifest.schema_version == "cyax0191-replay-v2"
    @test manifest.geometry_artifact_sha256 == geometry.artifact_sha256
    @test manifest.selected_source_route == "analytic-fixture"
    @test manifest.counting_unit == "one synthetic model state"
    @test manifest.model_switches == model.switches
    @test manifest.model_parameters.w0_magnitude == string(model.w0_magnitude)
    @test manifest.model_parameters.amplitudes == Tuple(string.(model.amplitudes))
    @test Tuple(Tuple(value.decimal for value in row)
        for row in manifest.model_parameters.charges) == (("1", "0"), ("0", "1"))
    @test manifest.model_parameters.uplift === nothing
    mutable_string = MutableManifestString("backend-origin-a")
    string_manifest = replay_manifest(NativeReplayBackend(), model, geometry;
        code_revision="synthetic-code-revision", selected_source_route="analytic-fixture",
        counting_unit="one synthetic model state", numeric_type="Float64",
        precision_bits=53, backend_versions=(; origin=mutable_string),
        solver_configuration=(; method="damped-newton"),
        scales=(; fields=(1.0, 1.0, 1.0, 1.0), potential=1.0))
    mutable_string.value = "backend-origin-b"
    @test string_manifest.backend_versions.origin == "backend-origin-a"
    @test string_manifest.backend_versions.origin isa String
    heterogeneous_components = Any["alpha", 7]
    heterogeneous_nested = Any["nested", 11]
    heterogeneous_manifest = replay_manifest(NativeReplayBackend(), model, geometry;
        code_revision="synthetic-code-revision", selected_source_route="analytic-fixture",
        counting_unit="one synthetic model state", numeric_type="Float64",
        precision_bits=53,
        backend_versions=(; components=heterogeneous_components,
            nested=(; values=heterogeneous_nested)),
        solver_configuration=(; method="damped-newton"),
        scales=(; fields=(1.0, 1.0, 1.0, 1.0), potential=1.0))
    heterogeneous_components[1] = "changed"
    heterogeneous_components[2] = 99
    heterogeneous_nested[1] = "changed-nested"
    heterogeneous_nested[2] = 99
    @test heterogeneous_manifest.backend_versions.components[1] == "alpha"
    @test heterogeneous_manifest.backend_versions.components[2] == 7
    @test heterogeneous_manifest.backend_versions.nested.values[1] == "nested"
    @test heterogeneous_manifest.backend_versions.nested.values[2] == 11
    @test heterogeneous_manifest.backend_versions.components isa
        CYAX0191.FrozenManifestArray{1}
    backend_versions_input = ["backend-v1", "numerics-v1"]
    solver_steps_input = [8, 16]
    field_scales_input = [1.0, 2.0, 3.0, 4.0]
    nested_manifest = replay_manifest(NativeReplayBackend(), model, geometry;
        code_revision="synthetic-code-revision", selected_source_route="analytic-fixture",
        counting_unit="one synthetic model state", numeric_type="Float64",
        precision_bits=53,
        backend_versions=(; components=backend_versions_input,
            provenance=(; tags=["cpu", "native"])),
        solver_configuration=(; steps=solver_steps_input,
            options=(; flags=[true, false])),
        scales=(; fields=field_scales_input, blocks=(; axions=[1.0, 1.0])),
        seed=17, budget=20)
    backend_versions_input[1] = "changed"
    solver_steps_input[1] = 999
    field_scales_input[1] = 999.0
    @test nested_manifest.backend_versions.components[1] == "backend-v1"
    @test nested_manifest.backend_versions.provenance.tags[2] == "native"
    @test nested_manifest.solver_configuration.steps[1] == 8
    @test nested_manifest.solver_configuration.options.flags[2] == false
    @test nested_manifest.scales.fields[1] == 1.0
    @test nested_manifest.scales.blocks.axions[2] == 1.0
    @test nested_manifest.source.divisor_basis_map[1, 1] == 1
    @test_throws Base.CanonicalIndexError setindex!(nested_manifest.scales.fields, 42.0, 1)
    source_charges = BigInt[1 0]
    source_amplitudes = [0.2]
    source_actions = [0.7]
    source_phases = [0.1]
    model_w0_input = BigFloat("0.75")
    immutable_model = fixture_model(2; charges=source_charges,
        amplitudes=source_amplitudes, actions=source_actions, phases=source_phases,
        w0=model_w0_input)
    immutable_geometry = synthetic_geometry_fixture()
    immutable_point = ([2.0, 1.0], [0.2, -0.3])
    immutable_value = potential(immutable_model, immutable_geometry, immutable_point...)
    immutable_identity = immutable_model.identity
    Base.GMP.MPZ.set!(source_charges[1, 1], BigInt(99))
    source_amplitudes[1] = 8.0
    source_actions[1] = 9.0
    source_phases[1] = 7.0
    exposed_charge = immutable_model.charges[1, 1]
    Base.GMP.MPZ.set!(exposed_charge, BigInt(88))
    exposed_charge_from_values = getfield(immutable_model.charges, :values)[1]
    @test exposed_charge_from_values isa Tuple{Vararg{UInt8}}
    @test_throws MethodError Base.GMP.MPZ.set!(exposed_charge_from_values, BigInt(77))
    @test_throws MethodError CYAX0191.FrozenArray{BigInt,1,1}((BigInt(3),), (1,))
    exposed_w0 = immutable_model.w0_magnitude
    @test exposed_w0 == model_w0_input
    @test precision(exposed_w0) == precision(model_w0_input)
    @test getfield(immutable_model, :w0_magnitude) isa CYAX0191.FrozenScalar{BigFloat}
    @test getfield(getfield(immutable_model, :w0_magnitude), :encoded)[1] ==
        precision(model_w0_input)
    @test_throws MethodError Base.MPFR.nextfloat!(getfield(immutable_model, :w0_magnitude))
    for field in (:w0_magnitude, :theta0, :gs, :kcs)
        @test getfield(immutable_model, field) isa CYAX0191.FrozenScalar{BigFloat}
        @test_throws MethodError Base.MPFR.nextfloat!(getfield(immutable_model, field))
    end
    Base.MPFR.nextfloat!(exposed_w0)
    @test immutable_model.w0_magnitude == model_w0_input
    @test immutable_model.charges[1, 1] == 1
    @test immutable_model.amplitudes[1] == 0.2
    @test immutable_model.actions[1] == 0.7
    @test immutable_model.phases[1] == 0.1
    @test immutable_model.identity == immutable_identity
    @test potential(immutable_model, immutable_geometry, immutable_point...) == immutable_value
    @test_throws Base.CanonicalIndexError setindex!(immutable_model.amplitudes, 4.0, 1)
    @test unassessed_controls().heavy_sector_stability.status == :NOT_ASSESSED
    @test unassessed_controls().global_compactification_consistency.status == :NOT_ASSESSED

    policies = FROZEN_GATE_C_POLICIES
    @test length(policies) == 3
    @test map(p -> p.benchmark_id, policies) == (:P0_B1, :P0_B2, :P0_B3)
    @test all(p -> p.precision_bits == 256 && p.frozen_before_gate_c, policies)
    @test all(p -> length(p.coordinate_order) == length(p.field_scales), policies)
    @test all(p -> p.potential_scale > 0 && p.potential_scale_rule !== "", policies)
    @test all(p -> p.stationarity_scale_rule !== "", policies)
    @test all(p -> p.stationarity_field_scale_rule !== "", policies)
    @test policy_manifest(frozen_policy(:P0_B3)).frozen_before_gate_c
    @test policy_manifest(frozen_policy(:P0_B3)).stationarity_field_scale_rule ==
        frozen_policy(:P0_B3).stationarity_field_scale_rule
    policy_copy = policy_b1.potential_scale
    @test policy_copy isa BigFloat
    @test precision(policy_copy) == 256
    @test getfield(policy_b1, :potential_scale) isa CYAX0191.FrozenScalar{BigFloat}
    @test getfield(getfield(policy_b1, :potential_scale), :encoded)[1] == 256
    @test_throws MethodError Base.MPFR.nextfloat!(getfield(policy_b1, :potential_scale))
    for field in (:potential_scale, :stationarity_tolerance, :root_tolerance,
            :optimizer_gradient_tolerance, :backtracking_factor, :minimum_step,
            :absolute_spectral_threshold, :relative_spectral_threshold)
        @test getfield(policy_b1, field) isa CYAX0191.FrozenScalar{BigFloat}
        @test_throws MethodError Base.MPFR.nextfloat!(getfield(policy_b1, field))
    end
    @test all(value isa CYAX0191.FrozenScalar{BigFloat}
        for value in getfield(policy_b1, :field_scales))
    original_policy_scale = string(policy_b1.potential_scale)
    Base.MPFR.nextfloat!(policy_copy)
    @test string(policy_b1.potential_scale) == original_policy_scale
    @test policy_manifest(policy_b1).stationarity_scale_rule == policy_b1.stationarity_scale_rule
    @test_throws KeyError frozen_policy(:unknown)
    @test scaled_stationarity_residual([1e-7, 1e-7], [1.0, 2.0], 1e-6) ≈ 0.2

    setprecision(BigFloat, 256) do
        scale_geometry = one_modulus_geometry(chi=-126, id="pointwise-scale-regression-v1")
        scale_model = fixture_model(1; w0=BigFloat(1), gs=BigFloat("0.1"),
            kcs=BigFloat(1), amplitudes=BigFloat[0], actions=BigFloat[1],
            phases=BigFloat[0], switches=ModelSwitches(bbhl_correction_enabled=true,
                np_linear_enabled=false, np_quadratic_enabled=false))
        scale_policy = CYAX0191._frozen_policy(:SYNTHETIC_SCALE_REGRESSION,
            -126, "0.1", "1.0", "1.0", true, ("t_1", "rho_1"))
        t_large = cbrt(BigFloat(6 * 154711))
        x_large = BigFloat[t_large, 0]
        scale_objective(x) = potential(scale_model, scale_geometry, view(x, 1:1), view(x, 2:2))
        static_criteria = SearchCriteria(collect(scale_policy.field_scales),
            scale_policy.potential_scale, scale_policy.stationarity_tolerance;
            max_iterations=1)
        static_search = search_stationary(DampedNewtonSearch(), scale_objective,
            x_large, static_criteria)
        large_gradient = finite_difference_gradient(CentralDifferenceBackend(),
            scale_objective, x_large)
        pointwise_criteria = policy_search_criteria(scale_policy, scale_model, scale_geometry)
        pointwise_search = search_stationary(DampedNewtonSearch(), scale_objective,
            x_large, pointwise_criteria)
        @test static_search.status == :converged
        @test static_search.scaled_residual <= scale_policy.stationarity_tolerance
        @test abs(large_gradient[1]) > BigFloat("1e-18")
        @test pointwise_search.status != :converged
        @test pointwise_search.scaled_residual > scale_policy.stationarity_tolerance
        local_scale = characteristic_potential_scale(scale_model, scale_geometry,
            view(x_large, 1:1), view(x_large, 2:2))
        @test local_scale < scale_policy.potential_scale / 1_000_000
        active_at_reference = evaluate_potential(scale_model, scale_geometry,
            view(x_large, 1:1), view(x_large, 2:2)).active_contributions
        @test local_scale == sum(abs, values(active_at_reference))
        @test pointwise_criteria.potential_scale_rule_identity ==
            scale_policy.stationarity_scale_rule
        @test pointwise_criteria.field_scale_rule_identity ==
            scale_policy.stationarity_field_scale_rule

        x_asymptotic = BigFloat[1000, 0]
        x_tail = BigFloat[5000, 0]
        legacy_scale_rule = x -> begin
            data = CYAX0191._kahler_data(scale_model, scale_geometry, view(x, 1:1))
            terms = CYAX0191._model_terms(scale_model, data, view(x, 2:2))
            abs(data.expK) * (abs(terms.w0) + sum(abs, terms.z_terms))^2
        end
        legacy_criteria = SearchCriteria(collect(scale_policy.field_scales),
            scale_policy.potential_scale, scale_policy.stationarity_tolerance;
            max_iterations=1, potential_scale_rule=legacy_scale_rule,
            potential_scale_rule_identity="legacy-superpotential-envelope")
        legacy_search = search_stationary(DampedNewtonSearch(), scale_objective,
            x_tail, legacy_criteria)
        active_search = search_stationary(DampedNewtonSearch(), scale_objective,
            x_asymptotic, pointwise_criteria)
        asymptotic_gradient = finite_difference_gradient(CentralDifferenceBackend(),
            scale_objective, x_asymptotic)
        asymptotic_active_scale = characteristic_potential_scale(scale_model,
            scale_geometry, view(x_asymptotic, 1:1), view(x_asymptotic, 2:2))
        asymptotic_legacy_scale = legacy_scale_rule(x_asymptotic)
        tail_gradient = finite_difference_gradient(CentralDifferenceBackend(),
            scale_objective, x_tail)
        @test abs(asymptotic_gradient[1]) > BigFloat("1e-32")
        @test abs(tail_gradient[1]) > BigFloat("1e-40")
        @test legacy_search.status == :converged
        @test legacy_search.scaled_residual <= scale_policy.stationarity_tolerance
        @test active_search.status != :converged
        @test active_search.scaled_residual > scale_policy.stationarity_tolerance
        @test asymptotic_active_scale < asymptotic_legacy_scale / 1_000_000

        runaway_criteria = policy_search_criteria(scale_policy, scale_model,
            scale_geometry; max_iterations=1)
        for t_runaway in (BigFloat("1e13"), BigFloat("1e14"))
            x_runaway = BigFloat[t_runaway, 0]
            runaway_scales = CYAX0191._search_field_scales(runaway_criteria, x_runaway)
            @test runaway_scales == BigFloat[t_runaway, 1]
            runaway_search = search_stationary(DampedNewtonSearch(), scale_objective,
                x_runaway, runaway_criteria)
            @test runaway_search.status == :failed
            @test runaway_search.scaled_residual > scale_policy.stationarity_tolerance
        end
    end

    setprecision(BigFloat, 128) do
        geometry_big = one_modulus_geometry(id="bigfloat-geometry-v1")
        model_big = fixture_model(1; charges=reshape([1], 1, 1),
            w0=BigFloat("0.75"), theta0=BigFloat("0.2"),
            gs=BigFloat("0.3"), kcs=BigFloat("0.1"),
            amplitudes=BigFloat[BigFloat("0.2")],
            actions=BigFloat[BigFloat("0.6")], phases=BigFloat[BigFloat("0.4")])
        t_big, rho_big = BigFloat[1.25], BigFloat[0.3]
        eval_big = evaluate_potential(model_big, geometry_big, t_big, rho_big)
        grad_big = analytic_axion_gradient(model_big, geometry_big, t_big, rho_big)
        state_big = critical_point_state(geometry_big, t_big, rho_big)
        @test eval_big.value isa BigFloat
        @test eltype(eval_big.retained_inverse_metric) === BigFloat
        @test eltype(grad_big) === BigFloat
        @test eltype(state_big.tau) === BigFloat
        @test precision(eval_big.value) >= 128
        @test CYAX0191._metric_assessment(eval_big.parent_metric).status == :PASS
        exposed_big_w0 = model_big.w0_magnitude
        @test exposed_big_w0 isa BigFloat
        @test precision(exposed_big_w0) == 128
        @test getfield(model_big, :w0_magnitude) isa CYAX0191.FrozenScalar{BigFloat}
        big_hessian = BigFloat[1 0; 0 2]
        big_fluctuation = fluctuation_analysis(GeneralizedEigenBackend(),
            big_hessian, Matrix{BigFloat}(I, 2, 2))
        @test eltype(big_fluctuation.generalized_mass_eigenvalues) === BigFloat
        @test big_fluctuation.generalized_mass_eigenvalues == BigFloat[1, 2]
        big_policy_scale = policy_b1.potential_scale
        @test big_policy_scale isa BigFloat
        @test getfield(policy_b1, :potential_scale) isa CYAX0191.FrozenScalar{BigFloat}
        big_criteria_scales = BigFloat[1, 2]
        big_criteria = SearchCriteria(big_criteria_scales, BigFloat(1), BigFloat("1e-8");
            max_iterations=10)
        big_criteria_scales[1] = BigFloat(99)
        @test big_criteria.field_scales[1] == 1
        @test getfield(big_criteria, :potential_scale) isa CYAX0191.FrozenScalar{BigFloat}
        @test getfield(big_criteria, :potential_scale).encoded[1] == 128
        @test_throws MethodError Base.MPFR.nextfloat!(getfield(big_criteria, :potential_scale))
    end

    high_precision_model_inputs = setprecision(BigFloat, 512) do
        model = fixture_model(1; charges=reshape([1], 1, 1),
            w0=BigFloat("0.75"), theta0=BigFloat("0.2"),
            gs=BigFloat("0.3"), kcs=BigFloat("0.1"),
            amplitudes=BigFloat[BigFloat("0.2")],
            actions=BigFloat[BigFloat("0.6")], phases=BigFloat[BigFloat("0.4")])
        t = BigFloat[BigFloat("1.234567890123456789")]
        rho = BigFloat[BigFloat("0.314159265358979323")]
        (; model, t, rho)
    end
    high_precision_model_reference = setprecision(BigFloat, 512) do
        model = high_precision_model_inputs.model
        geometry = one_modulus_geometry(id="model-precision-context-v1")
        t, rho = high_precision_model_inputs.t, high_precision_model_inputs.rho
        (; geometry,
           evaluation=evaluate_potential(model, geometry, t, rho),
           axion_gradient=analytic_axion_gradient(model, geometry, t, rho),
           characteristic_scale=characteristic_potential_scale(model, geometry, t, rho),
           kahler_data=CYAX0191._kahler_data(model, geometry, t; rho),
           critical_state=critical_point_state(geometry, t, rho))
    end
    default_precision_model_evaluation = evaluate_potential(
        high_precision_model_inputs.model, high_precision_model_reference.geometry,
        high_precision_model_inputs.t, high_precision_model_inputs.rho)
    high_ambient_model_evaluation = setprecision(BigFloat, 1024) do
        evaluate_potential(high_precision_model_inputs.model,
            high_precision_model_reference.geometry,
            high_precision_model_inputs.t, high_precision_model_inputs.rho)
    end
    default_precision_axion_gradient = analytic_axion_gradient(
        high_precision_model_inputs.model, high_precision_model_reference.geometry,
        high_precision_model_inputs.t, high_precision_model_inputs.rho)
    default_precision_characteristic_scale = characteristic_potential_scale(
        high_precision_model_inputs.model, high_precision_model_reference.geometry,
        high_precision_model_inputs.t, high_precision_model_inputs.rho)
    default_precision_kahler_data = CYAX0191._kahler_data(
        high_precision_model_inputs.model, high_precision_model_reference.geometry,
        high_precision_model_inputs.t; rho=high_precision_model_inputs.rho)
    default_precision_critical_state = critical_point_state(
        high_precision_model_reference.geometry, high_precision_model_inputs.t,
        high_precision_model_inputs.rho)
    @test precision(BigFloat) == 256
    @test precision(default_precision_model_evaluation.value) == 512
    @test default_precision_model_evaluation.value ==
        high_precision_model_reference.evaluation.value
    @test precision(high_ambient_model_evaluation.value) == 512
    @test high_ambient_model_evaluation.value == high_precision_model_reference.evaluation.value
    @test default_precision_model_evaluation.xi == high_precision_model_reference.evaluation.xi
    @test precision(default_precision_model_evaluation.xi) == 512
    @test default_precision_model_evaluation.retained_inverse_metric ==
        high_precision_model_reference.evaluation.retained_inverse_metric
    @test default_precision_axion_gradient == high_precision_model_reference.axion_gradient
    @test all(value -> precision(value) == 512, default_precision_axion_gradient)
    @test default_precision_characteristic_scale ==
        high_precision_model_reference.characteristic_scale
    @test precision(default_precision_characteristic_scale) == 512
    @test default_precision_kahler_data.xi == high_precision_model_reference.kahler_data.xi
    @test precision(default_precision_kahler_data.V) == 512
    @test default_precision_critical_state.tau ==
        high_precision_model_reference.critical_state.tau
    @test all(value -> precision(value) == 512, default_precision_critical_state.tau)

    high_precision_numeric_inputs = setprecision(BigFloat, 512) do
        x = BigFloat[BigFloat("1.125")]
        scales = BigFloat[1]
        target = BigFloat("1.25")
        criteria = SearchCriteria(scales, BigFloat(1), BigFloat("1e-40");
            max_iterations=8, minimum_step=BigFloat("1e-30"))
        f = let target=target
            point -> (point[1] - target)^2
        end
        (; x, scales, target, criteria, f)
    end
    high_precision_numeric_reference = setprecision(BigFloat, 512) do
        inputs = high_precision_numeric_inputs
        gradient = finite_difference_gradient(CentralDifferenceBackend(),
            inputs.f, inputs.x; scales=inputs.scales)
        hessian = finite_difference_hessian(CentralDifferenceBackend(),
            inputs.f, inputs.x; scales=inputs.scales)
        residual = scaled_stationarity_residual(gradient, inputs.scales, BigFloat(1))
        search = search_stationary(DampedNewtonSearch(), inputs.f, inputs.x,
            inputs.criteria)
        (; gradient, hessian, residual, search)
    end
    default_precision_numeric_gradient = finite_difference_gradient(
        CentralDifferenceBackend(), high_precision_numeric_inputs.f,
        high_precision_numeric_inputs.x; scales=high_precision_numeric_inputs.scales)
    default_precision_numeric_hessian = finite_difference_hessian(
        CentralDifferenceBackend(), high_precision_numeric_inputs.f,
        high_precision_numeric_inputs.x; scales=high_precision_numeric_inputs.scales)
    default_precision_numeric_residual = scaled_stationarity_residual(
        default_precision_numeric_gradient, high_precision_numeric_inputs.scales, BigFloat(1))
    default_precision_numeric_search = search_stationary(DampedNewtonSearch(),
        high_precision_numeric_inputs.f, high_precision_numeric_inputs.x,
        high_precision_numeric_inputs.criteria)
    @test precision(default_precision_numeric_gradient[1]) == 512
    @test default_precision_numeric_gradient == high_precision_numeric_reference.gradient
    @test precision(default_precision_numeric_hessian[1, 1]) == 512
    @test default_precision_numeric_hessian == high_precision_numeric_reference.hessian
    @test precision(default_precision_numeric_residual) == 512
    @test default_precision_numeric_residual == high_precision_numeric_reference.residual
    @test default_precision_numeric_search.status == :converged
    @test default_precision_numeric_search.point == high_precision_numeric_reference.search.point
    @test default_precision_numeric_search.value == high_precision_numeric_reference.search.value
    @test precision(default_precision_numeric_search.value) == 512

    high_precision_fluctuation_inputs = setprecision(BigFloat, 512) do
        (; hessian=BigFloat[1 0; 0 2], metric=Matrix{BigFloat}(I, 2, 2))
    end
    high_precision_fluctuation_reference = setprecision(BigFloat, 512) do
        fluctuation_analysis(GeneralizedEigenBackend(),
            high_precision_fluctuation_inputs.hessian,
            high_precision_fluctuation_inputs.metric)
    end
    default_precision_fluctuation = fluctuation_analysis(GeneralizedEigenBackend(),
        high_precision_fluctuation_inputs.hessian,
        high_precision_fluctuation_inputs.metric)
    @test precision(default_precision_fluctuation.generalized_mass_eigenvalues[1]) == 512
    @test default_precision_fluctuation.generalized_mass_eigenvalues ==
        high_precision_fluctuation_reference.generalized_mass_eigenvalues

    zeta3_512 = setprecision(BigFloat, 512) do
        CYAX0191._zeta3(BigFloat)
    end
    zeta3_256 = setprecision(BigFloat, 256) do
        CYAX0191._zeta3(BigFloat)
    end
    @test precision(zeta3_512) == 512
    @test setprecision(BigFloat, 512) do
        zeta3_512 != BigFloat(zeta3_256) &&
            abs(zeta3_512 - parse(BigFloat, CYAX0191._ZETA3_DECIMAL)) < BigFloat("1e-76")
    end
end
