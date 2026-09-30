using LinearAlgebra
using SHA
using Test

include(joinpath(@__DIR__, "../../../src/research/cyax0191/CYAX0191.jl"))
using .CYAX0191

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

@testset "CYAX-0191 Gate B geometry and importer boundary" begin
    geometry = synthetic_geometry_fixture()
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
    @test cone_margins(geometry, [2.0, 1.0]) == [3.0, 1.0]
    @test imported_domain_status(geometry, [2.0, 1.0]) == :PASS

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
    @test calabi_yau_volume(sequential, t_sequential) ≈ calabi_yau_volume(geometry, t_old)
    @test divisor_volumes(sequential, t_sequential) ≈ combined * divisor_volumes(geometry, t_old)
    @test_throws MethodError setindex!(geometry.intersections.coefficients, 9, 1)
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
    uplift = UpliftSpec(args -> 0.125, "constant synthetic uplift", "Gate B switch fixture")
    uplift_model = fixture_model(1; common...,
        switches=ModelSwitches(bbhl_correction_enabled=false,
            np_linear_enabled=false, np_quadratic_enabled=false, uplift_enabled=true),
        uplift)
    uplift_value = evaluate_potential(uplift_model, geometry, t, rho)
    @test uplift_value.contributions.optional_uplift == 0.125
    @test uplift_value.value ≈ 0.125

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

    geometry = synthetic_geometry_fixture()
    model = fixture_model(2; charges=Matrix{Int}(I, 2, 2),
        amplitudes=[0.2, 0.3], actions=[0.7, 0.9], phases=[0.1, -0.2])
    manifest = replay_manifest(NativeReplayBackend(), model, geometry;
        code_revision="synthetic-code-revision", selected_source_route="analytic-fixture",
        counting_unit="one synthetic model state", numeric_type="Float64",
        precision_bits=53, backend_versions=(; differentiation="central-difference-v1"),
        solver_configuration=(; method="damped-newton", budget=20),
        scales=(; fields=(1.0, 1.0, 1.0, 1.0), potential=1.0), seed=17, budget=20)
    @test manifest.geometry_artifact_sha256 == geometry.artifact_sha256
    @test manifest.selected_source_route == "analytic-fixture"
    @test manifest.counting_unit == "one synthetic model state"
    @test manifest.model_switches == model.switches
    @test manifest.model_parameters.w0_magnitude == string(model.w0_magnitude)
    @test manifest.model_parameters.amplitudes == Tuple(string.(model.amplitudes))
    @test manifest.model_parameters.charges == ((1, 0), (0, 1))
    @test manifest.model_parameters.uplift === nothing
    @test unassessed_controls().heavy_sector_stability.status == :NOT_ASSESSED
    @test unassessed_controls().global_compactification_consistency.status == :NOT_ASSESSED

    policies = FROZEN_GATE_C_POLICIES
    @test length(policies) == 3
    @test map(p -> p.benchmark_id, policies) == (:P0_B1, :P0_B2, :P0_B3)
    @test all(p -> p.precision_bits == 256 && p.frozen_before_gate_c, policies)
    @test all(p -> length(p.coordinate_order) == length(p.field_scales), policies)
    @test all(p -> p.potential_scale > 0 && p.potential_scale_rule !== "", policies)
    @test policy_manifest(frozen_policy(:P0_B3)).frozen_before_gate_c
    @test_throws KeyError frozen_policy(:unknown)
    @test scaled_stationarity_residual([1e-7, 1e-7], [1.0, 2.0], 1e-6) ≈ 0.2

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
    end
end
