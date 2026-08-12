@testset "Glimmers hierarchy and coupling kernels" begin
    glimmers = CYAxiverse.glimmers

    Q = Int[1 2 3; 0 1 1]
    potential = glimmers.GlimmersPotential(Q,
        Float64[5.0, 4.0, 3.0], Int[1, 1, 1], Int[2, 1, 3])
    kinv = Float64[1.0 0.2; 0.2 1.5]
    hierarchy = glimmers.hierarchy(potential, kinv)

    @test hierarchy.selected_indices == [2, 1]
    @test hierarchy.dependent_indices == [3]
    @test hierarchy.Q_reduced == Int[2 1; 1 0]
    @test hierarchy.q[2, 1] == 0
    @test hierarchy.q[1, 2] != 0
    @test hierarchy.q ≈ hierarchy.theta_from_canonical' * hierarchy.Q_reduced
    @test hierarchy.triangular_residual < 1e-12
    @test hierarchy.metric_residual < 1e-12
    @test all(>(0), hierarchy.log10_f_GeV)
    @test all(isfinite, hierarchy.log10_mass_eV)

    theta = glimmers.mixing_matrix(hierarchy)
    @test theta[1, 1] ≈ 1.0
    @test theta[2, 2] ≈ 1.0
    @test theta[2, 1] ≈ hierarchy.q[1, 2] / hierarchy.q[1, 1]
    @test theta[1, 2] ≈ -10.0^(4.0 - 5.0) * hierarchy.q[1, 2] /
        hierarchy.q[1, 1]

    photons = glimmers.photon_observables(hierarchy, hierarchy.Q_reduced[:, 1];
        light_threshold_eV=1.0e-30)
    @test photons.charge_residual < 1e-12
    @test photons.Cgamma[1] ≈ 1.0
    @test all(isfinite, photons.log10_g_GeVinv)
    @test all(isfinite, photons.log10_photon_width_GeV)
    @test photons.light_mode_count == 0

    signed = glimmers.GlimmersPotential(Q,
        Float64[5.0, 4.0, 3.0], Int[-1, 1, 1], Int[2, 1, 3])
    @test_throws ArgumentError glimmers.hierarchy(signed, kinv)
    adapted = glimmers.hierarchy(signed, kinv; signed_scale_policy=:absolute)
    @test adapted.coefficient_signs[1] == -1

    local_L = Float64[5.0 4.0 3.0; 0.0 -1.0 -2.0]
    @test_throws DimensionMismatch glimmers._normalise_potential(
        Q, Matrix(transpose(local_L)), Float64)
    @test_throws DimensionMismatch glimmers._normalise_potential(
        Matrix(transpose(Q)), local_L, Float64)
end
