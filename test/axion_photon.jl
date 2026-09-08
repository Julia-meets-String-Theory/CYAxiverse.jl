using LinearAlgebra
using HDF5
import Nemo
using Random

function _axion_photon_nemo_rank(matrix::AbstractMatrix{Int})
    isempty(matrix) && return 0
    Nemo.rank(Nemo.matrix(Nemo.ZZ, Matrix{Int}(matrix)))
end

@testset "Axion hierarchy and photon-coupling kernels" begin
    axion_photon = CYAxiverse.axion_photon

    Q = Int[1 2 3; 0 1 1]
    potential = axion_photon.InstantonData(Q,
        Float64[5.0, 4.0, 3.0], Int[1, 1, 1], Int[2, 1, 3])
    kinv = Float64[1.0 0.2; 0.2 1.5]
    hierarchy = axion_photon.leading_hierarchy(potential, kinv)

    @test hierarchy.selected_indices == [2, 1]
    @test hierarchy.dependent_indices == [3]
    @test hierarchy.rank_certificate.ordered_source_indices == [2, 1, 3]
    @test hierarchy.rank_certificate.prefix_ranks == [1, 2, 2]
    @test hierarchy.rank_certificate.selected_determinant == -1
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

@testset "Exact ordered rational rank certificate" begin
    axion_photon = CYAxiverse.axion_photon

    pathological_Q = Int[1_000_003 1 0; 0 0 1]
    pathological = axion_photon.InstantonData(
        pathological_Q, Float64[3.0, 2.0, 1.0], Int[1, 1, 1], Int[1, 2, 3])
    selected, dependent, certificate = axion_photon._select_independent_terms(
        pathological, 2)
    @test selected == [1, 3]
    @test dependent == [2]
    @test certificate.prefix_ranks == [1, 1, 2]
    @test certificate.selected_determinant == BigInt(1_000_003)
    @test _axion_photon_nemo_rank(pathological_Q[:, selected]) == 2

    hierarchy = axion_photon.leading_hierarchy(pathological,
        Matrix{Float64}(I, 2, 2))
    @test hierarchy.selected_indices == [1, 3]
    @test hierarchy.Q_reduced == Int[1_000_003 0; 0 1]
    @test hierarchy.rank_certificate.selected_determinant == BigInt(1_000_003)
    @test _axion_photon_nemo_rank(hierarchy.Q_reduced) == 2
    payload = axion_photon.rank_certificate_payload(hierarchy.rank_certificate)
    @test payload.algorithm == "modular_screen_with_exact_rational_fallback_v1"
    @test payload.matrix_shape == [2, 3]
    @test payload.selected_determinant == "1000003"

    # Hand-computed case: the first charge is (2, 1), the second is (1, 0),
    # and Gram--Schmidt gives q = [sqrt(5) 2/sqrt(5); 0 1/sqrt(5)].
    hand = axion_photon.leading_hierarchy(
        axion_photon.InstantonData(Int[2 1; 1 0], Float64[7.0, 6.0],
            Int[1, 1], Int[1, 2]), Matrix{Float64}(I, 2, 2))
    @test hand.selected_indices == [1, 2]
    @test hand.dependent_indices == Int[]
    @test hand.rank_certificate.prefix_ranks == [1, 2]
    @test hand.rank_certificate.selected_determinant == -1
    @test hand.q ≈ [sqrt(5.0) 2 / sqrt(5.0); 0.0 1 / sqrt(5.0)]
    @test hand.theta_from_canonical ≈
        [2 / sqrt(5.0) 1 / sqrt(5.0); 1 / sqrt(5.0) -2 / sqrt(5.0)]

    rng = MersenneTwister(0x03_0e_aa)
    for _ in 1:40
        h11 = rand(rng, 1:4)
        ncolumns = h11 + rand(rng, 0:4)
        Q = hcat(Matrix{Int}(I, h11, h11),
            rand(rng, -3:3, h11, ncolumns - h11))
        order = randperm(rng, ncolumns)
        scales = zeros(Float64, ncolumns)
        for (position, column) in enumerate(order)
            scales[column] = Float64(ncolumns - position)
        end
        potential = axion_photon.InstantonData(Q, scales, ones(Int, ncolumns), order)

        expected_selected = Int[]
        expected_dependent = Int[]
        expected_prefix_ranks = Int[]
        for column in order
            if length(expected_selected) == h11
                push!(expected_dependent, column)
                push!(expected_prefix_ranks, h11)
                continue
            end
            candidate = vcat(expected_selected, column)
            candidate_rank = _axion_photon_nemo_rank(Q[:, candidate])
            if candidate_rank > length(expected_selected)
                push!(expected_selected, column)
            else
                push!(expected_dependent, column)
            end
            push!(expected_prefix_ranks, candidate_rank)
        end

        selected, dependent, certificate = axion_photon._select_independent_terms(
            potential, h11)
        @test selected == expected_selected
        @test dependent == expected_dependent
        @test certificate.ordered_source_indices == order
        @test certificate.prefix_ranks == expected_prefix_ranks
        @test _axion_photon_nemo_rank(Q[:, selected]) == h11
        @test certificate.selected_determinant ==
            det(Matrix{BigInt}(Q[:, selected]))
    end
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

@testset "Axion-photon HDF5 write-read roundtrip" begin
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
            geometric["prime_divisor_volumes"] = Float64[1.0, 2.0]
            geometric["prime_toric_divisors"] = Int[0, 1]
        end

        original = axion_photon._run_local_scan(path)

        axion_photon.write_axion_photon_result(path, original)

        restored = axion_photon.read_axion_photon_result(path)

        @test restored.em_divisor_index == original.em_divisor_index
        @test restored.em_divisor_volume == original.em_divisor_volume
        @test restored.em_charge_source == original.em_charge_source
        @test restored.light_threshold_policy == original.light_threshold_policy
        @test restored.status == original.status

        @test restored.hierarchy.selected_indices == original.hierarchy.selected_indices
        @test restored.hierarchy.dependent_indices == original.hierarchy.dependent_indices
        @test restored.hierarchy.Q_reduced == original.hierarchy.Q_reduced
        @test restored.hierarchy.log10_lambda4 == original.hierarchy.log10_lambda4
        @test restored.hierarchy.coefficient_signs == original.hierarchy.coefficient_signs
        @test restored.hierarchy.q ≈ original.hierarchy.q
        @test restored.hierarchy.theta_from_canonical ≈ original.hierarchy.theta_from_canonical
        @test restored.hierarchy.log10_f_GeV ≈ original.hierarchy.log10_f_GeV
        @test restored.hierarchy.log10_mass_eV ≈ original.hierarchy.log10_mass_eV
        @test restored.hierarchy.m_planck_GeV == original.hierarchy.m_planck_GeV
        @test restored.hierarchy.triangular_residual == original.hierarchy.triangular_residual
        @test restored.hierarchy.metric_residual == original.hierarchy.metric_residual

        @test restored.hierarchy.rank_certificate.algorithm == original.hierarchy.rank_certificate.algorithm
        @test restored.hierarchy.rank_certificate.matrix_shape == original.hierarchy.rank_certificate.matrix_shape
        @test restored.hierarchy.rank_certificate.selected_determinant == original.hierarchy.rank_certificate.selected_determinant
        @test isempty(restored.hierarchy.rank_certificate.ordered_source_indices)
        @test isempty(restored.hierarchy.rank_certificate.prefix_ranks)
        @test restored.hierarchy.rank_certificate.selected_indices == original.hierarchy.selected_indices
        @test restored.hierarchy.rank_certificate.dependent_indices == original.hierarchy.dependent_indices

        @test restored.photons.n_em ≈ original.photons.n_em
        @test restored.photons.Cgamma ≈ original.photons.Cgamma
        @test restored.photons.log10_g_GeVinv ≈ original.photons.log10_g_GeVinv
        @test restored.photons.log10_g_effective_GeVinv ≈ original.photons.log10_g_effective_GeVinv
        @test restored.photons.log10_photon_width_GeV ≈ original.photons.log10_photon_width_GeV
        @test restored.photons.log10_quartic_width_GeV ≈ original.photons.log10_quartic_width_GeV
        @test restored.photons.light_threshold_eV == original.photons.light_threshold_eV
        @test restored.photons.log10_light_threshold_eV ≈ original.photons.log10_light_threshold_eV
        @test restored.photons.light_mode_count == original.photons.light_mode_count
        @test restored.photons.charge_residual == original.photons.charge_residual
        @test restored.photons.theta ≈ original.photons.theta

        @test restored.geometry.index == original.geometry.index
        @test restored.geometry.tip == original.geometry.tip
        @test restored.geometry.cy_volume == original.geometry.cy_volume
        @test restored.potential.Q == original.potential.Q
        @test restored.potential.log10_lambda4 == original.potential.log10_lambda4
        @test restored.identity == original.identity
        @test restored.identity.geometry_index == original.geometry.index
        @test restored.identity.geometry_digest == original.identity.geometry_digest
        @test restored.identity.potential_digest == original.identity.potential_digest
        @test restored.identity.geometry_snapshot_digest ==
            original.identity.geometry_snapshot_digest
        @test restored.identity.potential_snapshot_digest ==
            original.identity.potential_snapshot_digest
        @test restored.identity.configuration == original.identity.configuration
        @test restored.identity.configuration.signed_scale_policy == :require_positive
        @test restored.identity.configuration.precision == "Float64"

        @test_throws ArgumentError axion_photon.write_axion_photon_result(path, original)
        axion_photon.write_axion_photon_result(path, original; force=true)
        restored2 = axion_photon.read_axion_photon_result(path)
        @test restored2.hierarchy.selected_indices == original.hierarchy.selected_indices
    end
end

function _write_axion_photon_test_geometry(path)
    mkpath(dirname(path))
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
        geometric["prime_divisor_volumes"] = Float64[1.0, 2.0]
        geometric["prime_toric_divisors"] = Int[0, 1]
    end
    path
end

@testset "Axion-photon persisted identity guards" begin
    axion_photon = CYAxiverse.axion_photon
    mktempdir() do root
        path = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
            "np_0000001", "cy_0000001", "cyax.h5"))
        other_path = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
            "np_0000002", "cy_0000001", "cyax.h5"))
        original = axion_photon._run_local_scan(path)

        other_bytes = read(other_path)
        @test_throws ArgumentError axion_photon.write_axion_photon_result(
            other_path, original)
        @test read(other_path) == other_bytes

        changed_write_path = _write_axion_photon_test_geometry(joinpath(root,
            "h11_002", "np_0000003", "cy_0000001", "cyax.h5"))
        changed_write_result = axion_photon._run_local_scan(changed_write_path)
        h5open(changed_write_path, "r+") do file
            geometric = file["cytools/geometric"]
            try
                HDF5.delete_object(geometric, "tip")
                geometric["tip"] = Float64[1.25, 1.0]
            finally
                close(geometric)
            end
        end
        changed_bytes = read(changed_write_path)
        @test_throws ArgumentError axion_photon.write_axion_photon_result(
            changed_write_path, changed_write_result)
        @test read(changed_write_path) == changed_bytes

        changed_read_path = _write_axion_photon_test_geometry(joinpath(root,
            "h11_002", "np_0000004", "cy_0000001", "cyax.h5"))
        read_result = axion_photon._run_local_scan(changed_read_path)
        axion_photon.write_axion_photon_result(changed_read_path, read_result)
        h5open(changed_read_path, "r+") do file
            potential = file["cytools/potential"]
            try
                HDF5.delete_object(potential, "L")
                potential["L"] = Float64[1.0 1.0; -5.0 -11.0]
            finally
                close(potential)
            end
        end
        @test_throws ArgumentError axion_photon.read_axion_photon_result(
            changed_read_path)
        @test !axion_photon._has_axion_photon(changed_read_path)

        legacy_path = _write_axion_photon_test_geometry(joinpath(root,
            "h11_002", "np_0000005", "cy_0000001", "cyax.h5"))
        legacy_result = axion_photon._run_local_scan(legacy_path)
        axion_photon.write_axion_photon_result(legacy_path, legacy_result)
        h5open(legacy_path, "r+") do file
            result_group = file["spectrum/axion_photon"]
            try
                HDF5.delete_object(result_group, "identity")
            finally
                close(result_group)
            end
        end
        @test_throws ArgumentError axion_photon.read_axion_photon_result(legacy_path)
        @test !axion_photon._has_axion_photon(legacy_path)
        summaries = axion_photon.run_batch_axion_photon(data_dir=root,
            h11s=(2,), limit_per_h11=10, skip_complete=true)
        legacy_summary = only(filter(summary -> summary.path == legacy_path, summaries))
        @test legacy_summary.status == :written
        @test axion_photon.read_axion_photon_result(legacy_path).identity ==
            legacy_result.identity
    end
end

@testset "Axion-photon configuration and input identity guards" begin
    axion_photon = CYAxiverse.axion_photon

    @testset "Copied result is stale for a different geometry index" begin
        mktempdir() do root
            path_one = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
                "np_0000001", "cy_0000001", "cyax.h5"))
            path_two = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
                "np_0000002", "cy_0000001", "cyax.h5"))
            result_one = axion_photon._run_local_scan(path_one)
            axion_photon.write_axion_photon_result(path_one, result_one)

            h5open(path_one, "r") do source
                source_spectrum = source["spectrum"]
                try
                    h5open(path_two, "r+") do destination
                        destination_spectrum = create_group(destination, "spectrum")
                        try
                            HDF5.copy_object(source_spectrum, "axion_photon",
                                destination_spectrum, "axion_photon")
                        finally
                            close(destination_spectrum)
                        end
                    end
                finally
                    close(source_spectrum)
                end
            end

            @test_throws ArgumentError axion_photon.read_axion_photon_result(path_two)
            @test !axion_photon._has_axion_photon(path_two)

            summaries = axion_photon.run_batch_axion_photon(data_dir=root,
                h11s=(2,), limit_per_h11=10, skip_complete=true)
            summary_two = only(filter(summary -> summary.path == path_two, summaries))
            @test summary_two.status == :written
            @test axion_photon.read_axion_photon_result(path_two).geometry.index ==
                axion_photon._index_from_path(path_two)
        end
    end

    @testset "Batch configuration and requested read precision" begin
        mktempdir() do root
            path = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
                "np_0000001", "cy_0000001", "cyax.h5"))

            first = axion_photon.run_batch_axion_photon(data_dir=root,
                h11s=(2,), limit_per_h11=1, light_threshold_eV=1.0,
                skip_complete=true)
            @test only(first).status == :written

            conflicting = axion_photon.run_batch_axion_photon(data_dir=root,
                h11s=(2,), limit_per_h11=1, light_threshold_eV=2.0,
                skip_complete=true)
            @test only(conflicting).status == :failed
            @test occursin("different configuration", only(conflicting).error)

            forced = axion_photon.run_batch_axion_photon(data_dir=root,
                h11s=(2,), limit_per_h11=1, light_threshold_eV=2.0,
                force=true, skip_complete=true)
            @test only(forced).status == :written
            restored32 = axion_photon.read_axion_photon_result(path; T=Float32)
            @test restored32.geometry.tip isa Vector{Float32}
            @test restored32.potential.log10_lambda4 isa Vector{Float32}
            @test restored32.photons.light_threshold_eV == Float32(2.0)
            @test restored32.identity.configuration.precision == "Float64"
        end
    end

    @testset "Retained snapshots and raw source precision" begin
        mktempdir() do root
            retained_path = _write_axion_photon_test_geometry(joinpath(root,
                "h11_002", "np_0000001", "cy_0000001", "cyax.h5"))
            retained_result = axion_photon._run_local_scan(retained_path)
            retained_result.geometry.tip[1] += 0.25
            retained_bytes = read(retained_path)
            @test_throws ArgumentError axion_photon.write_axion_photon_result(
                retained_path, retained_result)
            @test read(retained_path) == retained_bytes

            float32_path = _write_axion_photon_test_geometry(joinpath(root,
                "h11_002", "np_0000002", "cy_0000001", "cyax.h5"))
            float32_result = axion_photon._run_local_scan(float32_path; T=Float32)
            axion_photon.write_axion_photon_result(float32_path, float32_result)
            @test Float32(1.0 + 1.0e-8) == Float32(1.0)
            h5open(float32_path, "r+") do file
                geometric = file["cytools/geometric"]
                try
                    HDF5.delete_object(geometric, "tip")
                    geometric["tip"] = Float64[1.0 + 1.0e-8, 1.0]
                finally
                    close(geometric)
                end
            end
            changed_bytes = read(float32_path)
            @test_throws ArgumentError axion_photon.read_axion_photon_result(
                float32_path; T=Float32)
            @test !axion_photon._has_axion_photon(float32_path)
            @test_throws ArgumentError axion_photon.write_axion_photon_result(
                float32_path, float32_result; force=true)
            @test read(float32_path) == changed_bytes
        end
    end

    @testset "Persisted snapshot digests are verified before batch skip" begin
        mktempdir() do root
            path = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
                "np_0000001", "cy_0000001", "cyax.h5"))
            result = axion_photon._run_local_scan(path)
            axion_photon.write_axion_photon_result(path, result)

            h5open(path, "r+") do file
                identity = file["spectrum/axion_photon/identity"]
                try
                    HDF5.delete_object(identity, "geometry_snapshot_digest")
                    identity["geometry_snapshot_digest"] = repeat("0", 64)
                finally
                    close(identity)
                end
            end

            @test_throws ArgumentError axion_photon.read_axion_photon_result(path)
            @test !axion_photon._has_axion_photon(path)
            summaries = axion_photon.run_batch_axion_photon(data_dir=root,
                h11s=(2,), limit_per_h11=1, skip_complete=true)
            @test only(summaries).status == :written
            @test axion_photon._has_axion_photon(path)
            @test axion_photon.read_axion_photon_result(path).identity == result.identity
        end
    end

    @testset "Persisted configuration cannot diverge from result metadata" begin
        mktempdir() do root
            path = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
                "np_0000001", "cy_0000001", "cyax.h5"))
            result = axion_photon._run_local_scan(path; em_divisor_index=1)
            axion_photon.write_axion_photon_result(path, result)
            original_configuration = result.identity.configuration
            tampered_configuration = axion_photon.AxionPhotonConfiguration(
                original_configuration.em_divisor_index,
                original_configuration.light_threshold_eV,
                "unused",
                :divisor_instanton,
                original_configuration.signed_scale_policy,
                original_configuration.precision)

            h5open(path, "r+") do file
                identity = file["spectrum/axion_photon/identity"]
                configuration = identity["configuration"]
                try
                    HDF5.delete_object(configuration, "light_threshold_eV_effective")
                    configuration["light_threshold_eV_effective"] = "unused"
                    HDF5.delete_object(configuration, "qed_threshold_policy")
                    configuration["qed_threshold_policy"] = "divisor_instanton"
                    HDF5.delete_object(identity, "configuration_digest")
                    identity["configuration_digest"] =
                        axion_photon._configuration_digest(tampered_configuration)
                finally
                    close(configuration)
                    close(identity)
                end
            end

            @test_throws ArgumentError axion_photon.read_axion_photon_result(path)
            @test !axion_photon._has_axion_photon(path)
        end
    end
end

@testset "Axion-photon HDF5 publication failure recovery" begin
    axion_photon = CYAxiverse.axion_photon
    mktempdir() do root
        path = _write_axion_photon_test_geometry(joinpath(root, "h11_002",
            "np_0000001", "cy_0000001", "cyax.h5"))
        original = axion_photon._run_local_scan(path)
        unsupported = axion_photon._run_local_scan(path; T=BigFloat)

        @test_throws Exception axion_photon.write_axion_photon_result(
            path, unsupported)
        @test !axion_photon._has_axion_photon(path)
        @test h5open(path, "r") do file
            !haskey(file, "spectrum")
        end
        @test !any(name -> occursin(".axion_photon.tmp-", name),
            readdir(dirname(path)))

        axion_photon.write_axion_photon_result(path, original)
        before_failed_force = axion_photon.read_axion_photon_result(path)
        bytes_before_failed_force = read(path)
        @test_throws Exception axion_photon.write_axion_photon_result(
            path, unsupported; force=true)
        after_failed_force = axion_photon.read_axion_photon_result(path)
        @test after_failed_force.hierarchy.selected_indices ==
            before_failed_force.hierarchy.selected_indices
        @test read(path) == bytes_before_failed_force
        @test axion_photon._has_axion_photon(path)

        h5open(path, "r+") do file
            spectrum = file["spectrum"]
            HDF5.delete_object(spectrum, "axion_photon")
            partial = create_group(spectrum, "axion_photon")
            partial["status"] = "partial"
        end
        @test !axion_photon._has_axion_photon(path)
        summaries = axion_photon.run_batch_axion_photon(data_dir=root,
            h11s=(2,), limit_per_h11=1, skip_complete=true)
        @test length(summaries) == 1
        @test summaries[1].status == :written
        @test axion_photon._has_axion_photon(path)
        @test axion_photon.read_axion_photon_result(path).status == original.status

        h5open(path, "r+") do file
            hierarchy = file["spectrum/axion_photon/hierarchy"]
            HDF5.delete_object(hierarchy, "q")
            hierarchy["q"] = Float64[1.0]
        end
        @test !axion_photon._has_axion_photon(path)
        malformed_summaries = axion_photon.run_batch_axion_photon(data_dir=root,
            h11s=(2,), limit_per_h11=1, skip_complete=true)
        @test malformed_summaries[1].status == :written
        skipped_summaries = axion_photon.run_batch_axion_photon(data_dir=root,
            h11s=(2,), limit_per_h11=1, skip_complete=true)
        @test skipped_summaries[1].status == :skipped

        h5open(path, "r") do file
            spectrum = file["spectrum"]
            @test all(name -> name == "axion_photon", keys(spectrum))
        end
    end
end
