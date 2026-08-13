using LinearAlgebra
using HDF5

@testset "Axion hierarchy and photon-coupling kernels" begin
    axion_photon = CYAxiverse.axion_photon

    Q = Int[1 2 3; 0 1 1]
    potential = axion_photon.InstantonData(Q,
        Float64[5.0, 4.0, 3.0], Int[1, 1, 1], Int[2, 1, 3])
    kinv = Float64[1.0 0.2; 0.2 1.5]
    hierarchy = axion_photon.leading_hierarchy(potential, kinv)

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

    one_axion = axion_photon.leading_hierarchy(
        axion_photon.InstantonData(reshape(Int[3], 1, 1),
            Float64[-20.0], Int[1], Int[1]),
        reshape(Float64[0.25], 1, 1))
    one_q = 3.0 / 2.0
    @test one_axion.log10_f_GeV[1] ≈
        log10(axion_photon.M_PLANCK_GEV) - log10(2π) - log10(one_q)
    @test one_axion.log10_mass_eV[1] ≈
        -10.0 + log10(axion_photon.M_PLANCK_GEV) + 9.0 +
        log10(2π) + log10(one_q)

    theta = axion_photon.mixing_matrix(hierarchy)
    @test theta[1, 1] ≈ 1.0
    @test theta[2, 2] ≈ 1.0
    @test theta[2, 1] ≈ hierarchy.q[1, 2] / hierarchy.q[1, 1]
    @test theta[1, 2] ≈ -10.0^(4.0 - 5.0) * hierarchy.q[1, 2] /
        hierarchy.q[1, 1]

    photons = axion_photon.photon_observables(hierarchy, hierarchy.Q_reduced[:, 1];
        light_threshold_eV=1.0e-30)
    @test photons.charge_residual < 1e-12
    @test photons.Cgamma[1] ≈ 1.0
    @test all(isfinite, photons.log10_g_GeVinv)
    @test all(isfinite, photons.log10_photon_width_GeV)
    @test photons.light_mode_count == 0

    signed = axion_photon.InstantonData(Q,
        Float64[5.0, 4.0, 3.0], Int[-1, 1, 1], Int[2, 1, 3])
    @test_throws ArgumentError axion_photon.leading_hierarchy(signed, kinv)
    adapted = axion_photon.leading_hierarchy(signed, kinv; signed_scale_policy=:absolute)
    @test adapted.coefficient_signs[1] == -1

    local_L = Float64[5.0 4.0 3.0; 0.0 -1.0 -2.0]
    @test_throws DimensionMismatch axion_photon._normalise_potential(
        Q, Matrix(transpose(local_L)), Float64)
    @test_throws DimensionMismatch axion_photon._normalise_potential(
        Matrix(transpose(Q)), local_L, Float64)

    assignment = axion_photon.VisibleSectorAssignment{Float64}(
        1, 2, 1, 2, 40.0, 12.0, Int[1, 0], Int[0, 1], Int[0, 1],
        3, -20.0, true, true, true, :intersecting_d7)
    geometry = axion_photon.GeometryInputs{Float64}(
        "/tmp/cyaxiverse-visible-sector/cyax.h5",
        CYAxiverse.structs.GeometryIndex(2, 1, 1),
        Float64[1.0, 1.0], Float64[2.0, 3.0], 5.0,
        Matrix{Float64}(I, 2, 2), Int[1 0; 0 1], Float64[1.0, 2.0],
        Int[1, 2], assignment)
    expected_log_threshold = 0.5 * assignment.qed_log10_lambda4 +
        log10(axion_photon.M_PLANCK_GEV) + 9.0 + log10(2π)
    @test axion_photon.qed_instanton_log10_threshold_eV(geometry) ≈
        expected_log_threshold
    @test axion_photon.qed_instanton_threshold_eV(geometry) ≈
        10.0^expected_log_threshold
end

@testset "Visible-sector metadata roundtrip" begin
    axion_photon = CYAxiverse.axion_photon
    mktempdir() do root
        geometry_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geometry_dir)
        path = joinpath(geometry_dir, "cyax.h5")
        h5open(path, "w") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            potential["Q"] = Int[1 0; 0 1]
            potential["L"] = Float64[1.0 1.0; -5.0 -10.0]
            geometric["tip"] = Float64[1.0, 1.0]
            geometric["divisor_volumes"] = Float64[2.0, 3.0]
            geometric["CY_volume"] = 10.0
            geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
            geometric["effective_cone"] = Int[1 0; 0 1]
            geometric["prime_divisor_volumes"] = Float64[40.0, 12.0]
            geometric["prime_toric_divisors"] = Int[0, 1]
            visible = create_group(geometric, "visible_sector")
            visible["qcd_divisor_index"] = 0
            visible["qed_divisor_index"] = 1
            visible["qcd_image_index"] = 0
            visible["qed_image_index"] = 1
            visible["qcd_divisor_volume"] = 40.0
            visible["qed_divisor_volume"] = 12.0
            visible["qcd_charge"] = Int[1, 0]
            visible["qed_charge"] = Int[0, 1]
            visible["em_charge"] = Int[0, 1]
            visible["qed_instanton_index"] = 1
            visible["qed_log10_lambda4"] = -10.0
            visible["qcd_qed_intersection"] = 1
            visible["qcd_invariant"] = 1
            visible["qed_invariant"] = 1
        end

        loaded = axion_photon.load_geometry_inputs(path)
        @test loaded.visible_sector !== nothing
        @test loaded.visible_sector.qed_divisor_index == 2
        @test loaded.visible_sector.qed_instanton_index == 2
        @test loaded.visible_sector.em_charge == Int[0, 1]
        result = axion_photon._run_local_scan(path;
            qed_threshold_policy=:divisor_instanton)
        @test result.em_charge_source == :visible_sector_qed
        @test result.light_threshold_policy == :divisor_instanton
        @test result.status == :visible_sector_instanton_threshold
        @test result.em_divisor_index == 2
        @test result.photons.log10_light_threshold_eV ≈
            axion_photon.qed_instanton_log10_threshold_eV(loaded)
    end
end
