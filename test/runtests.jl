using CYAxiverse
using LinearAlgebra
using SparseArrays
using Test
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "vacua_pipeline.jl"))
include(joinpath(@__DIR__, "..", "scripts", "batch_vacua_pipeline.jl"))
include(joinpath(@__DIR__, "..", "scripts", "batch_physical_spectrum.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_refinement_common.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_scan_prep.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_scan_pilot.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_scale_continuation.jl"))

@testset "Scale-continuation pilot diagnostics" begin
    Q = Int[1 2]
    L = Float64[1.0 1.0; 0.0 log10(0.25)]
    K = reshape([1.0], 1, 1)

    @test pilot_parse_scale_grid("1.0,0.9,1.0") == [0.9, 1.0]
    defaults = _pilot_parse_args(String[])
    @test defaults[:scale_status] == :physical
    @test defaults[:volume_normalization] == :full
    @test pilot_homotopy_scale(L, 0.9)[2, 2] ≈ 0.9 * L[2, 2]
    @test_throws ArgumentError pilot_homotopy_scale(L, 1.0;
        scale_status=:physical)

    # The physical generic path follows the author laws:
    # tau -> k tau, Kinv -> k^2 Kinv, and either fixed or full CY volume.
    base_Q = Int[1 0 1 -1 0 1; 0 1 1 1 1 0]
    base_geometry = (; τ_volumes=[1.0, 2.0], kinv=Matrix{Float64}(I, 2, 2),
        cy_volume=10.0)
    base_L = _pilot_author_potential(base_Q, base_geometry.τ_volumes,
        base_geometry.kinv, base_geometry.cy_volume)
    cross_coefficient = (8π / base_geometry.cy_volume^2) *
        (π * dot(base_Q[:, 1], base_geometry.kinv * base_Q[:, 2]) +
         dot(base_Q[:, 1] + base_Q[:, 2], base_geometry.τ_volumes))
    @test base_L[2, 4] ≈ log10(abs(cross_coefficient)) -
        2π * log10(exp(1.0)) *
        dot(base_Q[:, 1] + base_Q[:, 2], base_geometry.τ_volumes)
    base_K = Matrix{Float64}(I, 2, 2)
    fixed = pilot_scaled_inputs(base_Q, base_L, base_K, 0.9;
        scale_status=:physical, geometry=base_geometry,
        volume_normalization=:fixed)
    full = pilot_scaled_inputs(base_Q, base_L, base_K, 0.9;
        scale_status=:physical, geometry=base_geometry,
        volume_normalization=:full)
    @test fixed.volume ≈ 10.0
    @test full.volume ≈ 10.0 * 0.9^(3 / 2)
    @test log(full.volume) - log(fixed.volume) ≈ (3 / 2) * log(0.9)
    @test fixed.tau ≈ 0.9 .* base_geometry.τ_volumes
    @test fixed.kinv ≈ 0.9^2 .* base_geometry.kinv
    @test fixed.K ≈ base_K / 0.9^2
    @test fixed.L != full.L
    @test fixed.scale_status == :physical
    @test fixed.volume_normalization == :fixed

    # A common amplitude normalization rescales derivatives but cannot move
    # their zeros or change Hessian signatures in the N=5 potential.
    n5 = CYAxiverse.paper_benchmarks.n5_potential(k=0.6745)
    θ5 = collect(range(0.07, 0.63; length=5))
    base_eval = CYAxiverse.generate.structured_charge_evaluator(n5.Q, n5.L)
    scaled_L5 = copy(n5.L); scaled_L5[2, :] .+= 7.0
    scaled_eval = CYAxiverse.generate.structured_charge_evaluator(n5.Q, scaled_L5)
    base_derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
        base_eval, θ5, n5.Q)
    scaled_derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
        scaled_eval, θ5, n5.Q)
    # The production log-shifted evaluator removes this common factor for
    # numerical conditioning, so its derivative arrays are identical.
    @test scaled_derivatives.gradient ≈ base_derivatives.gradient
    @test scaled_derivatives.hessian ≈ base_derivatives.hessian
    @test sign.(eigvals(Symmetric(base_derivatives.hessian))) ==
        sign.(eigvals(Symmetric(scaled_derivatives.hessian)))
    @test _pilot_periodic_distance([0.99], [0.01]) ≈ 0.02
    @test_throws ArgumentError pilot_select_geometries("/private/tmp",
        [CYAxiverse.structs.GeometryIndex(491, 1, 1)])
    resource_capped = pilot_collect_seeds(Q, L; max_branches=100,
        max_stage_allocated_bytes=1)
    @test resource_capped.status == :resource_cap
    @test resource_capped.estimate == 2
    @test isempty(resource_capped.seeds)

    factor = cholesky(K).L
    below = _pilot_records([[0.5]], [0], Q,
        pilot_homotopy_scale(L, 0.99), factor;
        residual_tolerance=1e-10, max_iterations=20, duplicate_tolerance=1e-7)
    above = _pilot_records([[0.5]], [0], Q,
        pilot_homotopy_scale(L, 1.01), factor;
        residual_tolerance=1e-10, max_iterations=20, duplicate_tolerance=1e-7)
    _pilot_init_branch_ids!(below)
    matches = pilot_match_records!(below, above; matching_tolerance=0.1)
    @test length(matches) == 1
    brackets = _pilot_mark_crossings!(below, above, matches, 0.99, 1.01;
        zero_eigenvalue_tolerance=1e-6, bracket_number=1,
        previous_minima=1, current_minima=0)
    @test brackets == 1
    @test below[1].near_catastrophe && above[1].near_catastrophe
    @test below[1].catastrophe_bracket == "bracket-0001"
    @test below[1].classification.hessian_min > 0
    @test above[1].classification.hessian_min < 0
    locked = _classify_point([0.5], Q, L, cholesky(K))
    @test below[1].classification.value ≈ locked.value
    @test below[1].classification.negative_modes == 0

    augmented = pilot_augmented_catastrophe([0.5], [1.0], 1.0, Q, L, K;
        tolerance=1e-10, scale_status=:homotopy_only)
    @test augmented.converged
    @test augmented.gradient_residual < 1e-10
    @test augmented.null_residual < 1e-10
    @test augmented.normalized_null_vector_residual < 1e-10
    @test abs(augmented.hessian_eigenvalues[1]) < 1e-10

    benchmark = pilot_benchmark_regression()
    @test benchmark.passed
    @test benchmark.n5_critical_scale ≈ 0.674506370003365 atol=1e-12
    @test benchmark.n5_ratio ≈ 0.25 atol=1e-12
    @test benchmark.n8_zero_mode
    @test benchmark.n8_positive_heavy_modes
    @test benchmark.n8_detuned_negative_modes == 1

    mktempdir() do root
        path = joinpath(root, "report.csv")
        pilot_prepare_csv(path, PILOT_SUMMARY_FIELDS)
        pilot_append_csv(path, (; row_type=:scale, h11=8, polytope=1,
            frst=1, sampled_scale=1.0), PILOT_SUMMARY_FIELDS)
        @test length(readlines(path)) == 2
        @test _pilot_completed_scales(path) == Set([(8, 1, 1, 1.0)])
        pilot_prepare_csv(path, PILOT_SUMMARY_FIELDS; append=true)
        @test_throws ArgumentError pilot_prepare_csv(path, PILOT_SUMMARY_FIELDS)
        @test_throws ArgumentError pilot_prepare_csv(path, (:wrong_schema,); append=true)
    end
end

@testset "Geometry-level LQtilde orientation" begin
    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        h5open(joinpath(geom_dir, "cyax.h5"), "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            Q = Int[1 0 1; 0 1 1]
            L = Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0]
            potential["Q"] = Q
            potential["L"] = L
            geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
        end
        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            geom_idx = CYAxiverse.structs.GeometryIndex(2, 1, 1)
            from_geometry = CYAxiverse.generate.LQtilde(geom_idx)
            from_matrices = CYAxiverse.generate.LQtilde(
                Int[1 0 1; 0 1 1], Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0])
            oriented = CYAxiverse.read.oriented_potential(geom_idx)
            @test oriented.Q == Int[1 0 1; 0 1 1]
            @test oriented.L == Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0]
            @test Matrix(oriented.K) == Matrix{Float64}(I, 2, 2)
            @test from_geometry.Qtilde == from_matrices.Qtilde
            @test from_geometry.Ltilde == from_matrices.Ltilde
            @test from_geometry.Qbar == from_matrices.Qbar
            @test from_geometry.Lbar == from_matrices.Lbar

            transposed_dir = joinpath(root, "h11_002", "np_0000002", "cy_0000001")
            mkpath(transposed_dir)
            h5open(joinpath(transposed_dir, "cyax.h5"), "cw") do file
                cytools = create_group(file, "cytools")
                potential = create_group(cytools, "potential")
                geometric = create_group(cytools, "geometric")
                potential["Q"] = Int[1 0; 0 1; 1 1]
                potential["L"] = Float64[1.0 0.0; 1.0 -1.0; 1.0 -10.0]
                geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
            end
            transposed = CYAxiverse.read.oriented_potential(
                CYAxiverse.structs.GeometryIndex(2, 2, 1))
            @test size(transposed.Q) == (2, 3)
            @test size(transposed.L) == (2, 3)
            @test transposed.Q == Int[1 0 1; 0 1 1]
            @test transposed.L == Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0]
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
    end
end

@testset "Log-shifted derivative workspace" begin
    Q = Int[1 0 1; 0 1 1]
    L = Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0]
    theta = [0.13, 0.27]
    workspace = CYAxiverse.generate.logshifted_derivative_workspace(Q, L)
    derivatives = CYAxiverse.generate.logshifted_derivatives!(workspace, theta, Q)
    shift = maximum(L[2, :])
    amplitudes = L[1, :] .* 10.0 .^ (L[2, :] .- shift)
    arguments = 2π .* (Q' * theta)
    expected_value = sum(amplitudes .* (1 .- cos.(arguments)))
    expected_gradient = 2π .* Q * (amplitudes .* sin.(arguments))
    expected_hessian = (2π)^2 .* Q *
        Diagonal(amplitudes .* cos.(arguments)) * Q'
    @test derivatives.log_shift == shift
    @test derivatives.value ≈ expected_value
    @test derivatives.gradient ≈ expected_gradient
    @test derivatives.hessian ≈ expected_hessian
    first_gradient = derivatives.gradient
    second = CYAxiverse.generate.logshifted_derivatives!(workspace, [0.21, 0.31], Q)
    @test second.gradient === first_gradient
    @test second.hessian === derivatives.hessian
end

@testset "Structured pairwise charge evaluator" begin
    h11 = 2
    base_count = h11 + 4
    base_Q = Int[1 0 1 2 1 0;
                 0 1 1 1 0 2]
    Q = let columns = [base_Q[:, index] for index in 1:base_count]
        for i in 1:(base_count - 1), j in (i + 1):base_count
            orientation = iseven(i + j) ? 1 : -1
            push!(columns, orientation .* (base_Q[:, j] - base_Q[:, i]))
        end
        hcat(columns...)
    end
    L = zeros(Float64, 2, size(Q, 2))
    L[1, :] .= 1.0
    L[2, :] .= collect(0.0:-1.0:-(size(Q, 2) - 1))
    representation = CYAxiverse.generate.structured_charge_representation(Q, L)
    @test representation.validated
    @test representation.base_count == base_count
    @test length(representation.pair_i) == binomial(base_count, 2)
    @test representation.L === L

    structured = CYAxiverse.generate.structured_charge_evaluator(Q, L)
    generic_workspace = CYAxiverse.generate.logshifted_derivative_workspace(Q, L)
    @test !structured.uses_generic_fallback
    for theta in ([0.123, 0.456], [0.731, 0.219], [0.0, 0.5])
        structured_derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
            structured, theta, Q)
        generic_derivatives = CYAxiverse.generate.logshifted_derivatives!(
            generic_workspace, theta, Q)
        @test isapprox(structured_derivatives.value, generic_derivatives.value;
            rtol=1e-13, atol=1e-13)
        @test isapprox(structured_derivatives.gradient, generic_derivatives.gradient;
            rtol=1e-13, atol=1e-13)
        @test isapprox(structured_derivatives.hessian, generic_derivatives.hessian;
            rtol=1e-13, atol=1e-13)
    end

    invalid_Q = copy(Q)
    invalid_Q[1, end] += 1
    fallback = CYAxiverse.generate.structured_charge_evaluator(invalid_Q, L)
    @test !fallback.representation.validated
    @test fallback.uses_generic_fallback
    fallback_derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
        fallback, [0.123, 0.456], invalid_Q)
    expected_fallback = CYAxiverse.generate.logshifted_derivatives!(
        CYAxiverse.generate.logshifted_derivative_workspace(invalid_Q, L),
        [0.123, 0.456], invalid_Q)
    @test fallback_derivatives.value == expected_fallback.value
    @test fallback_derivatives.gradient == expected_fallback.gradient
    @test fallback_derivatives.hessian == expected_fallback.hessian
end

@testset "Cached screening Hessian eigensolver" begin
    for dimension in (3, 8)
        matrix = randn(dimension, dimension)
        matrix = (matrix + matrix') / 2
        expected = eigvals(Symmetric(matrix))
        workspace = _symmetric_eigen_workspace(dimension)
        eigenvalues = _symmetric_eigenvalues!(workspace, copy(matrix))
        @test eigenvalues === workspace.eigenvalues
        @test eigenvalues ≈ expected

        warmed_matrix = copy(matrix)
        _symmetric_eigenvalues!(workspace, warmed_matrix)
        @test @allocated(_symmetric_eigenvalues!(workspace, warmed_matrix)) <= 4096
    end
end

@testset "Inflation append-only scan shards" begin
    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        h5open(joinpath(geom_dir, "cyax.h5"), "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            potential["Q"] = Int[1 0 1; 0 1 1]
            potential["L"] = Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0]
            geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
        end

        shard_dir = joinpath(root, "shards")
        options = _scan_prep_parse_args([
            "--data-dir", root, "--geometry", "2,1,1",
            "--max-branches", "100000", "--shard-dir", shard_dir,
            "--run-id", "test-stage5"])
        @test run_scan_prep(options)
        paths = inflation_shard_paths(shard_dir)
        @test length(paths) == 1
        @test count(==('\n'), read(paths[1], String)) == 2
        @test (2, 1, 1) in inflation_completed_shard_geometries(
            shard_dir; data_dir=abspath(root), max_branches=100000)

        resume_options = _scan_prep_parse_args([
            "--data-dir", root, "--geometry", "2,1,1",
            "--max-branches", "100000", "--shard-dir", shard_dir,
            "--resume"])
        @test run_scan_prep(resume_options)
        @test count(==('\n'), read(paths[1], String)) == 2

        low_index_shard_dir = joinpath(root, "low-index-shards")
        low_index_options = _scan_prep_parse_args([
            "--data-dir", root, "--geometry", "2,1,1",
            "--max-branches", "100000", "--negative-mode-range", "1:1",
            "--shard-dir", low_index_shard_dir])
        @test run_scan_prep(low_index_options)
        low_index_path = inflation_shard_paths(low_index_shard_dir)[1]
        low_index_text = read(low_index_path, String)
        @test occursin("negative_mode_range", first(split(low_index_text, '\n')))
        @test occursin(",1:1,", low_index_text)
        @test (2, 1, 1) in inflation_completed_shard_geometries(
            low_index_shard_dir; data_dir=abspath(root), max_branches=100000,
            negative_mode_range=1:1)

        merged = joinpath(root, "merged.csv")
        @test inflation_merge_shards(paths, merged) == merged
        @test readlines(merged)[1] == INFLATION_SHARD_HEADER
        @test_throws ArgumentError inflation_merge_shards(paths, merged)

        failed_dir = joinpath(root, "failed-shards")
        failed_options = _scan_prep_parse_args([
            "--data-dir", root, "--geometry", "9,1,1",
            "--shard-dir", failed_dir, "--retries", "1"])
        @test !run_scan_prep(failed_options)
        failed_path = only(inflation_shard_paths(failed_dir))
        @test count(==('\n'), read(failed_path, String)) == 3
        @test occursin("attempt", read(failed_path, String))
        @test isempty(inflation_completed_shard_geometries(
            failed_dir; data_dir=abspath(root), max_branches=1_000_000))
    end
end

@testset "Inflation stratified pilot selection and report" begin
    @test _leading_branch_det_qtilde(4, 2) == 1
    @test _leading_branch_det_qtilde(1, 150) == 0
    @test inflation_screening_tier(50) == :normal
    @test inflation_screening_tier(100) == :middle
    @test inflation_screening_tier(150) == :high_memory_queue
    @test inflation_branch_estimate_lower_bound(150) == big(2)^150
    @test !inflation_refinement_eligible((status=:success, h11=150,
        candidate_count=1, allocated_bytes=1, output_bytes=1))
    @test inflation_refinement_eligible((status=:success, h11=15,
        candidate_count=1, allocated_bytes=1, output_bytes=1))
    high_h11_selected = CYAxiverse.structs.LQLinearlyIndependent(
        Matrix{Int}(I, 150, 150), zeros(Int, 150, 0),
        zeros(Float64, 2, 0),
        vcat(ones(Float64, 1, 150), zeros(Float64, 1, 150)))
    high_h11_error = try
        CYAxiverse.generate.foreach_leading_critical_branch(
            high_h11_selected; max_branches=100_000) do _, _
            nothing
        end
        nothing
    catch error
        error
    end
    @test high_h11_error isa ArgumentError
    @test occursin("1427247692705959881058285969449495136382746624 branches",
        sprint(showerror, high_h11_error))
    mktempdir() do root
        for (h11, count) in ((4, 3), (8, 2))
            for index in 1:count
                path = joinpath(root, string("h11_", lpad(h11, 3, '0')),
                    string("np_", lpad(index, 7, '0')), "cy_0000001", "cyax.h5")
                mkpath(dirname(path))
                open(path, "w") do io
                    write(io, "synthetic pilot placeholder")
                end
            end
        end
        selected = inflation_pilot_select_geometries(root;
            sample_per_h11=2)
        @test length(selected) == 4
        @test sort(unique(geom.h11 for geom in selected)) == [4, 8]
        capped = inflation_pilot_select_geometries(root;
            sample_per_h11=2, max_geometries=3)
        @test length(capped) == 3
        @test capped[1].h11 == 4
        middle_only = inflation_pilot_select_geometries(root;
            h11_min=8, h11_max=8, sample_per_h11=2)
        @test length(middle_only) == 2
        @test all(geom -> geom.h11 == 8, middle_only)

        rows = [
            (h11=4, polytope=1, frst=1, status=:success, attempt=1,
                instantons=40, strong_hierarchy=true, leading_log_gap=5.0,
                log_scale_span=12.0, branch_count=8, candidate_count=2,
                total_seconds=0.25, allocated_bytes=1000, output_bytes=200,
                error=""),
            (h11=4, polytope=2, frst=1, status=:failed, attempt=2,
                instantons=42, strong_hierarchy=false, leading_log_gap=2.0,
                log_scale_span=6.0, branch_count=0, candidate_count=0,
                total_seconds=0.5, allocated_bytes=0, output_bytes=0,
                error="synthetic failure")]
        reports = _inflation_pilot_report_rows(rows)
        @test length(reports) == 2
        @test sum(report.geometries for report in reports) == 2
        @test sum(report.failures for report in reports) == 1
        report_path = joinpath(root, "pilot-report.csv")
        @test _inflation_pilot_write_report(report_path, reports) == report_path
        @test occursin("mean_allocated_bytes", read(report_path, String))
    end
end

@testset "CYAxiverse.jl" begin
    @testset "core" begin
        @test CYAxiverse.greet_CYAxiverse() == "Hello CYAxiverse!"
        @test CYAxiverse.greet_CYAxiverse() != "Hello world!"
    end

    if get(ENV, "CYAXIVERSE_TEST_CYTOOLS", "0") == "1"
        @testset "cytools wrapper regression repro" begin
            include(joinpath(@__DIR__, "..", "scripts", "cytools_wrapper_repro.jl"))
        end
    end
end

@testset "Vacua pipeline persistence and validation" begin
    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        path = joinpath(geom_dir, "cyax.h5")
        h5open(path, "cw") do file
            spectrum = create_group(file, "spectrum")
            spectrum["sentinel"] = Int[17, 23]
        end

        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            vacuum_geom_dir = joinpath(root, "h11_002", "np_0000002", "cy_0000001")
            mkpath(vacuum_geom_dir)
            h5open(joinpath(vacuum_geom_dir, "cyax.h5"), "cw") do file
                cytools = create_group(file, "cytools")
                potential = create_group(cytools, "potential")
                geometric = create_group(cytools, "geometric")
                potential["Q"] = Int[1 0 1; 0 1 1]
                potential["L"] = Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0]
                geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
            end
            vacuum_only = compute_vacua_data(
                CYAxiverse.structs.GeometryIndex(2, 2, 1), root;
                method=:auto, starts=64, save=false)
            @test !haskey(vacuum_only, "spectrum")
            @test vacuum_only["search"].auto_selected_method == "exact_determinant"
            @test vacuum_only["vacua_estimate"].vac == 1
            @test haskey(vacuum_only["timings"], "potential_load_seconds")
            @test haskey(vacuum_only["timings"], "vacua_search_seconds")
            geom_idx = CYAxiverse.structs.GeometryIndex(2, 1, 1)
            spectrum = nothing
            estimate = (vac=3, issquare=1)
            identified = Dict("vac" => 3)
            save_axion_data(geom_idx, spectrum, estimate, identified;
                threshold=0.5, starts=17, residual_tolerance=1e-9,
                merge_tolerance=1e-6, max_iterations=12, force=false,
                search_metadata=(search_method="legacy",
                    search_classification="finite_search_lower_bound",
                    minimum_count=3, multiplicity=1.0, critical_count=-1,
                    branch_count=-1, det_Qtilde=-1, search_status="completed"))

            h5open(path, "r") do file
                @test read(file["spectrum/sentinel"]) == [17, 23]
                @test read(file["vacua_pipeline/estimate"]) == 3
                @test read(file["vacua_pipeline/verified"]) == 3
                @test read(file["vacua_pipeline/metadata/status"]) == "completed"
                @test read(file["vacua_pipeline/metadata/solver_status"]) == "completed"
                @test read(file["vacua_pipeline/metadata/starts"]) == 17
                @test read(file["vacua_pipeline/metadata/method"]) == "legacy"
            end
            @test _has_pipeline_result(path)
            config = _pipeline_config(threshold=0.5, starts=17,
                residual_tolerance=1e-9, merge_tolerance=1e-6,
                max_iterations=12, method=:legacy)
            @test _has_pipeline_result(path; config=config)
            @test _vacua_result_state(path, config) == :matching
            matching_summary = joinpath(root, "matching.csv")
            matching_options = _vacua_parse_args(["--data-dir", root,
                "--geometry", "2,1,1", "--summary", matching_summary, "--threshold", "0.5",
                "--starts", "17", "--residual-tolerance", "1e-9",
                "--merge-tolerance", "1e-6", "--max-iterations", "12", "--dry-run"])
            @test run_vacua_batch(matching_options)
            @test occursin("skipped", read(matching_summary, String))

            parallel_summary = joinpath(root, "parallel.csv")
            parallel_options = _vacua_parse_args(["--data-dir", root,
                "--geometry", "2,1,1", "--workers", "2", "--blas-threads", "1",
                "--summary", parallel_summary, "--force"])
            @test parallel_options isa NamedTuple
            @test parallel_options[:workers] == 2
            @test parallel_options[:blas_threads] == 1
            default_blas_options = _vacua_parse_args(["--data-dir", root,
                "--geometry", "2,1,1"])
            @test default_blas_options[:blas_threads] == 1
            @test !run_vacua_batch(parallel_options)
            @test occursin("failed", read(parallel_summary, String))

            mismatched_options = _vacua_parse_args(["--data-dir", root,
                "--geometry", "2,1,1", "--threshold", "0.6", "--dry-run"])
            @test _vacua_result_state(path, _vacua_config(mismatched_options)) != :matching
            @test !run_vacua_batch(mismatched_options)
            pipeline_data = CYAxiverse.read.pipeline_vacua(geom_idx)
            @test pipeline_data.estimate == 3
            @test pipeline_data.metadata.status == "completed"
            @test pipeline_data.metadata.starts == 17
            @test pipeline_data.metadata.method == "legacy"
            @test pipeline_data.metadata.verification_status == "not_applicable"
            @test pipeline_data.metadata.search_method == "legacy"
            @test_throws ArgumentError save_axion_data(geom_idx, spectrum, estimate, identified;
                threshold=0.5, force=false)

            bad_potential = (Q=zeros(Int, 3, 4), L=zeros(2, 4),
                K=Hermitian(Matrix{Float64}(I, 2, 2)))
            @test_throws DimensionMismatch _validate_potential(geom_idx, bad_potential)
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
    end
end

@testset "Physical spectrum batch persistence" begin
    spectrum = CYAxiverse.structs.PhysicalAxionSpectrum(
        [1.0, 2.0], [0, 1], zeros(2, 2), [1, -1], [3.0, 4.0],
        zeros(Int, 4, 0), Int[], Float64[], zeros(Int, 4, 0), Int[], Float64[],
        -30.0, 200)
    geom_idx = CYAxiverse.structs.GeometryIndex(2, 3, 4)
    output_dir = mktempdir()
    output_path = joinpath(output_dir, "h11_002", "np_0000003", "cy_0000004", "cyax.h5")
    mkpath(dirname(output_path))
    h5open(output_path, "cw") do file
        create_group(file, "cytools")
    end
    _write_result(output_path, geom_idx, spectrum; prec=200, threshold_log10=-30.0,
        quartics=true, runtime_seconds=0.1, provisional=false, fK=[5.0, 6.0])
    HDF5.h5open(output_path, "r") do file
        @test HDF5.haskey(file, "spectrum/physical/m")
        @test read(file["spectrum/physical/fK_log10"]) == [5.0, 6.0]
        @test read(file["spectrum/physical/fpert_log10"]) ≈ [-0.5, 0.0]
    end
    old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
    ENV["CYAXIVERSE_DATA_DIR"] = output_dir
    try
        loaded = CYAxiverse.read.physical_spectrum(geom_idx)
        @test loaded.m == spectrum.m
        @test loaded.fpert ≈ [-0.5, 0.0]
        @test loaded.mass_signs_or_inertia == Int[]
    finally
        old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
            (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
    end

    _write_result(output_path, geom_idx, spectrum; prec=200, threshold_log10=-30.0,
        quartics=false, runtime_seconds=0.2, provisional=false, fK=[5.0, 6.0])
    HDF5.h5open(output_path, "r") do file
        @test !HDF5.haskey(file, "spectrum/physical/mass_signs_or_inertia")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_self_sign")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_self_log10")
        @test !HDF5.haskey(file, "spectrum/physical/fpert_log10")
        @test read(file["spectrum/physical/metadata/quartics"]) == false
    end

    physical_options = _parse_args(["--geometry", "2,3,4", "--quartics"])
    @test physical_options isa NamedTuple
    @test physical_options[:geometries] == [geom_idx]
    @test physical_options[:quartics]

    summary_path = joinpath(output_dir, "summary.csv")
    _write_summary_header(summary_path)
    _append_summary(summary_path, geom_idx; status="failed", error="synthetic failure",
        prec=200, threshold_log10=-30.0)
    @test count(==( '\n'), read(summary_path, String)) == 2
    @test occursin("failed", read(summary_path, String))
end

@testset "Paper reproduction benchmarks" begin
    @testset "Locally scaled critical-point solver" begin
        extreme = CYAxiverse.minimizer.critical_points(
            [1.0 1.0; 0.0 -400.0], [1 0; 0 1]; starts=64)
        @test extreme.critical_count == 4
        @test extreme.minima_count == 1
        @test all(isfinite, reduce(vcat, extreme.hessian_eigenvalues))

        displaced = CYAxiverse.minimizer.critical_points(
            [1.0 1.0 -1.0;
             -13.988802496336225 -13.988802497269443 -13.991896359391525],
            [1 0 1; 0 1 1]; starts=256)
        @test displaced.critical_count == 6
        @test displaced.minima_count == 2
        @test all(isfinite, reduce(vcat, displaced.hessian_eigenvalues))

        n = 26
        high_dimensional_q = zeros(Int, n, n + 1)
        high_dimensional_q[:, 1:n] .= Matrix{Int}(I, n, n)
        high_dimensional_q[1, n + 1] = 1
        high_dimensional_q[2, n + 1] = 1
        high_dimensional_l = zeros(2, n + 1)
        high_dimensional_l[1, 1:n] .= 1.0
        high_dimensional_l[1, n + 1] = -1.0
        high_dimensional_l[2, n + 1] = -0.003
        high_dimensional = CYAxiverse.minimizer.critical_points(
            high_dimensional_l, high_dimensional_q; starts=64,
            initial_points=zeros(n, 1))
        @test high_dimensional.minima_count == 2
    end

    n5 = CYAxiverse.paper_benchmarks.n5_potential()
    @test size(n5.Q) == (5, 8)
    @test n5.qdotτ == [6, 6.25, 24, 26, 31.875, 32, 36.125, 162.125]
    n5_selected = CYAxiverse.generate.LQtilde(n5.Q, n5.L)
    @test size(n5_selected.Qtilde) == (5, 5)
    @test rank(n5_selected.Qtilde) == 5
    @test n5_selected.Qtilde == n5.Q[:, 1:5]

    kc = CYAxiverse.paper_benchmarks.n5_critical_scale()
    @test isapprox(CYAxiverse.paper_benchmarks.n5_reduced_ratio(kc), 1 / 4; atol=1e-14)
    @test CYAxiverse.paper_benchmarks.n5_reduced_critical_points(kc - 1e-4).minima == 2
    @test CYAxiverse.paper_benchmarks.n5_reduced_critical_points(kc + 1e-4).minima == 1
    @test CYAxiverse.paper_benchmarks.n5_reduced_critical_points(kc).hessian_sign[2] == 0

    for (k, expected_critical, expected_minima) in
            ((kc - 1e-4, 4, 2), (kc + 1e-4, 2, 1))
        ratio = CYAxiverse.paper_benchmarks.n5_reduced_ratio(k)
        solved = CYAxiverse.minimizer.critical_points(
            [1.0 1.0; 0.0 log10(ratio)], [1 2]; starts=64)
        @test solved.critical_count == expected_critical
        @test solved.minima_count == expected_minima
        @test maximum(solved.residuals) <= 1e-10
    end

    n8 = CYAxiverse.paper_benchmarks.n8_potential()
    @test size(n8.Q) == (8, 12)
    @test n8.Q[:, 7] == [0, 1, -1, -1, 1, 1, 0, 0]
    @test n8.qdotτ == [14, 14.5, 14.5, 15.5, 15.5, 15.5, 15.5, 16, 17, 17, 25, 45]
    n8_selected = CYAxiverse.generate.LQtilde(n8.Q, n8.L)
    @test size(n8_selected.Qtilde) == (8, 8)
    @test rank(n8_selected.Qtilde) == 8
    @test abs(round(Int, det(n8_selected.Qtilde))) == 1

    # The hierarchy-preconditioned production entry point must agree with the
    # generic solver on a square, unimodular potential.
    square = CYAxiverse.generate.reduced_critical_points(
        [1.0 1.0 0.0; 0.0 -1.0 -100.0], [1 0 0; 0 1 0]; starts=32)
    @test square.critical_count == 4
    @test square.minima_count == 1

    # Section 4.2 reports five minima below the N=8 catastrophe and one above
    # it. Two thousand deterministic starts are sufficient to recover both
    # sides with the Table 1 truncation.
    for (k, expected_minima) in ((0.66, 5), (0.68, 1))
        potential = CYAxiverse.paper_benchmarks.n8_potential(k=k)
        solved = CYAxiverse.generate.reduced_critical_points(
            potential.L, potential.Q; starts=2048,
            residual_tolerance=1e-9, merge_tolerance=1e-6,
            max_iterations=300)
        @test solved.minima_count == expected_minima
    end
    scan = CYAxiverse.paper_benchmarks.n8_minima_scan(starts=2048)
    @test Tuple(result.minima_count for result in scan) == (5, 1)

    reduced_problem = CYAxiverse.jlm_reduced.prepare(n8.Q, n8.L)
    @test reduced_problem.Q_reduced isa SparseArrays.AbstractSparseMatrix
    @test reduced_problem.extra_rows == size(reduced_problem.Q_reduced, 1) - size(reduced_problem.Q_reduced, 2)

    square_jlm = CYAxiverse.jlm_reduced.minimize(
        [1 0 0; 0 1 0], [1.0 1.0 0.0; 0.0 -1.0 -100.0])
    @test square_jlm.N_min == 1
    @test square_jlm.det_QTilde == 1
    scaled_square_jlm = CYAxiverse.jlm_reduced.minimize(
        [2 0 1; 0 2 0], [1.0 1.0 1.0; 0.0 -10.0 -1.0])
    @test scaled_square_jlm.N_min == 4
    @test scaled_square_jlm.det_QTilde == 4

    # The legacy alpha-matrix reduction remains the default, while the
    # author-compatible reduction retains rational coordinates and enlarges
    # their fundamental domain before solving.
    author_fixture = (Q=Int[2 0 1 1; 0 2 1 0; 0 0 1 1],
        L=Float64[1.0 1.0 1.0 1.0; 0.0 -1.0 -2.0 -3.0])
    alpha_problem = CYAxiverse.jlm_reduced.prepare(author_fixture.Q, author_fixture.L)
    author_problem = CYAxiverse.jlm_reduced.prepare(author_fixture.Q, author_fixture.L;
        reduction=:author)
    @test alpha_problem.reduction == :alphamatrix
    @test author_problem.reduction == :author
    @test author_problem.coordinate_scale == [2, 1]
    @test Matrix(author_problem.Q_reduced) == [2 0; 0 1; -1 1]
    author_ensemble = CYAxiverse.jlm_reduced.critical_ensemble(author_problem; starts=256)
    @test author_ensemble.critical_count == 8
    @test author_ensemble.minima_count == 2

    synthetic = (Q=Int[2 0 1; 0 2 1],
        L=Float64[1.0 1.0 1.0; 0.0 -1.0 -10.0])
    synthetic_geom = CYAxiverse.structs.GeometryIndex(2, 1, 1)
    leading_estimate, _, leading_search = _search_vacua(synthetic_geom, synthetic;
        threshold=0.5, starts=64, residual_tolerance=1e-9,
        merge_tolerance=1e-6, max_iterations=100,
        method=:leading_branches, max_branches=1_000)
    @test leading_estimate.vac == 4
    @test leading_search.search_classification == "certified_selected_branch_set"
    @test leading_search.branch_count == 16
    @test_throws ArgumentError _search_vacua(synthetic_geom, synthetic;
        threshold=0.5, starts=64, residual_tolerance=1e-9,
        merge_tolerance=1e-6, max_iterations=100,
        method=:leading_branches, max_branches=8)

    reduced_estimate, _, reduced_search = _search_vacua(synthetic_geom, synthetic;
        threshold=0.5, starts=64, residual_tolerance=1e-9,
        merge_tolerance=1e-6, max_iterations=100,
        method=:reduced_jlm, max_branches=1_000)
    @test reduced_estimate.vac == 4
    @test reduced_search.search_classification == "exact_determinant_branch"
    @test reduced_search.multiplicity == 4.0

    lattice_selected = CYAxiverse.generate.LQtilde(
        [2 0 1; 0 2 1], [1.0 1.0 1.0; 0.0 -1.0 -10.0])
    exact_det_matrix = Int[-3 -1; 0 -1]
    @test CYAxiverse.generate._exact_integer_determinant(exact_det_matrix) == 3
    @test CYAxiverse.generate._exact_integer_determinant(Int[0 1; 1 0]) == -1
    @test CYAxiverse.generate._exact_integer_determinant(exact_det_matrix) ==
        det(Matrix{BigInt}(exact_det_matrix))
    overflow_det_matrix = Int[typemax(Int) 0; 0 2]
    @test CYAxiverse.generate._exact_integer_determinant(overflow_det_matrix) ==
        BigInt(typemax(Int)) * 2
    lattice_offsets = CYAxiverse.generate.leading_lattice_offsets(lattice_selected)
    @test size(lattice_offsets) == (2, 4)
    @test all(maximum(abs.(mod.(lattice_selected.Qtilde' * lattice_offsets, 1.0)); dims=1) .< 1e-12)
    lattice_branches = CYAxiverse.generate.leading_critical_branches(lattice_selected)
    @test lattice_branches.det_Qtilde == 4
    @test lattice_branches.branch_count == 16
    @test lattice_branches.leading_minima_count == 4
    @test count(==(0), lattice_branches.leading_negative_modes) == 4
    @test count(==(1), lattice_branches.leading_negative_modes) == 8
    @test count(==(2), lattice_branches.leading_negative_modes) == 4
    streamed_coordinates = zeros(Float64, size(lattice_branches.coordinates))
    streamed_modes = Int[]
    callback_theta = Ref{Any}(nothing)
    callback_reused = Ref(true)
    CYAxiverse.generate.foreach_leading_critical_branch(lattice_selected;
            max_branches=1_000) do theta, negative_modes
        callback_theta[] === nothing ||
            (callback_reused[] &= callback_theta[] === theta)
        callback_theta[] = theta
        push!(streamed_modes, negative_modes)
        streamed_coordinates[:, length(streamed_modes)] .= theta
    end
    @test streamed_coordinates == lattice_branches.coordinates
    @test streamed_modes == lattice_branches.leading_negative_modes
    @test callback_reused[]

    # A low-index search is a deterministic subset of the legacy numeric-mask
    # ordering.  Its report keeps the exact full mask count and the selected
    # mask/lattice coverage separate.
    low_index_coordinates = Matrix{Float64}(undef, 2, 8)
    low_index_modes = Int[]
    low_index_cursor = 0
    low_index_report = CYAxiverse.generate.foreach_leading_critical_branch(
            lattice_selected; max_branches=1_000,
            negative_mode_range=1:1) do theta, negative_modes
        low_index_cursor += 1
        low_index_coordinates[:, low_index_cursor] .= theta
        push!(low_index_modes, negative_modes)
    end
    expected_low_index = findall(==(1), lattice_branches.leading_negative_modes)
    @test low_index_coordinates == lattice_branches.coordinates[:, expected_low_index]
    @test low_index_modes == lattice_branches.leading_negative_modes[expected_low_index]
    @test low_index_report.branch_count == 8
    @test low_index_report.mask_count == 4
    @test low_index_report.masks_visited == 2
    @test low_index_report.masks_skipped == 2
    @test low_index_report.lattice_copy_count == 4
    @test low_index_report.lattice_copies_visited == 8
    @test low_index_report.search_classification == :deterministic_low_index_enumeration

    low_index_branches = CYAxiverse.generate.leading_critical_branches(
        lattice_selected; max_branches=1_000, negative_mode_range=1:1)
    @test low_index_branches.branch_count == 8
    @test low_index_branches.leading_minima_count == 0
    @test low_index_branches.stream_report == low_index_report
    @test_throws ArgumentError CYAxiverse.generate.foreach_leading_critical_branch(
        lattice_selected; max_branches=1_000,
        negative_mode_range=1:1, max_negative_modes=2) do theta, negative_modes
        nothing
    end

    zero_sign_selected = CYAxiverse.generate.LQtilde(
        [1 0 1; 0 1 1], [0.0 1.0 1.0; 0.0 -1.0 -10.0])
    zero_sign_branches = CYAxiverse.generate.leading_critical_branches(
        zero_sign_selected; max_branches=1_000)
    @test sort(zero_sign_branches.leading_negative_modes) == [0, 0, 1, 1]
    @test CYAxiverse.generate.leading_critical_branches(zero_sign_selected;
        max_branches=1_000, negative_mode_range=1:1).branch_count == 2

    # The exact count is checked before lattice-offset allocation, even when
    # h11 is large enough that 2^h11 cannot be represented by an Int.
    high_h11 = 150
    high_Q = hcat(Matrix{Int}(I, high_h11, high_h11),
        zeros(Int, high_h11, 1))
    high_L = zeros(Float64, 2, high_h11 + 1)
    high_L[1, :] .= 1.0
    high_L[2, 1:high_h11] .= collect(0.0:-1.0:-149.0)
    high_L[2, end] = -1000.0
    high_selected = CYAxiverse.generate.LQtilde(high_Q, high_L)
    high_report = CYAxiverse.generate.foreach_leading_critical_branch(
        high_selected; max_branches=1_000, negative_mode_range=1:1) do theta, negative_modes
        nothing
    end
    @test high_report.branch_count == 150
    @test high_report.masks_visited == 150
    @test high_report.mask_count == BigInt(2)^high_h11
    @test_throws ArgumentError CYAxiverse.generate.foreach_leading_critical_branch(
        high_selected; max_branches=1_000) do theta, negative_modes
        nothing
    end

    signed_selected = CYAxiverse.generate.LQtilde(
        [1 0 1; 0 1 1], [-1.0 1.0 1.0; 0.0 -1.0 -10.0])
    signed_branches = CYAxiverse.generate.leading_critical_branches(signed_selected)
    @test signed_branches.branch_count == 4
    @test signed_branches.leading_minima_count == 1
    @test sort(signed_branches.leading_negative_modes) == [0, 1, 1, 2]

    catastrophe = CYAxiverse.paper_benchmarks.n8_degenerate_point(
        [0.0, 0.00499839, 0.99500161, 0.75995156,
         0.75004523, 0.24995477, 0.0, 0.75495317])
    @test catastrophe.converged
    @test isapprox(catastrophe.k, 0.6745063700; atol=1e-9)
    @test catastrophe.gradient_residual < 1e-10
    @test catastrophe.null_residual < 1e-10
    @test abs(catastrophe.eigenvalues[1]) < 1e-9
    @test catastrophe.eigenvalues[2] > 0

    # Inflation uses the paper's equation-(25) truncation by default.  The
    # equation-(19) reconstruction remains available as an explicit diagnostic.
    theta_glsm = Matrix{Float64}(n8_selected.Qtilde') \ catastrophe.theta
    truncated = CYAxiverse.paper_benchmarks.n8_potential_derivatives(theta_glsm, catastrophe.k)
    full = CYAxiverse.paper_benchmarks.n8_potential_derivatives(theta_glsm, catastrophe.k; full=true)
    fixed_volume = CYAxiverse.paper_benchmarks.n8_full_potential(
        k=2.0, volume_normalization=:fixed)
    full_volume = CYAxiverse.paper_benchmarks.n8_full_potential(k=2.0)
    @test length(truncated.amplitudes) == 12
    @test length(full.amplitudes) == 78
    @test norm(truncated.gradient, Inf) / maximum(abs, truncated.amplitudes) < 1e-10
    @test full_volume.volume_normalization == :full
    @test fixed_volume.volume_normalization == :fixed
    @test isapprox(fixed_volume.volume, 126.0; atol=0)
    @test isapprox(full_volume.volume, 126.0 * 2.0^(3 / 2); rtol=1e-14)
    @test full_volume.Q == fixed_volume.Q
    @test full_volume.L[1, :] == fixed_volume.L[1, :]
    @test all(isapprox.(full_volume.L[2, :] .- fixed_volume.L[2, :],
        fill(-3log10(2.0), 78); atol=1e-12))
    fixed_derivatives = CYAxiverse.paper_benchmarks.n8_potential_derivatives(
        zeros(8), 2.0; full=true, volume_normalization=:fixed)
    full_derivatives = CYAxiverse.paper_benchmarks.n8_potential_derivatives(
        zeros(8), 2.0; full=true)
    # The leading 12 terms avoid subnormal Float64 log-to-amplitude
    # conversion in the far-tail cross terms.
    @test all(isapprox.(full_derivatives.amplitudes[1:12] ./
        fixed_derivatives.amplitudes[1:12], fill(2.0^-3, 12);
        rtol=1e-12))
    @test_throws ArgumentError CYAxiverse.paper_benchmarks.n8_full_potential(
        volume_normalization=:author)

    initial = CYAxiverse.paper_benchmarks.n8_inflation_initial_condition(
        catastrophe.k + 1e-7)
    @test !initial.follow_hilltop
    @test isapprox(initial.theta_critical, theta_glsm; atol=1e-12)

    local_form = CYAxiverse.paper_benchmarks.n8_local_hilltop_coefficients()
    @test isapprox(local_form.beta1, 3.71e3; rtol=2e-3)
    @test isapprox(local_form.c4, 2.3724e10; rtol=2e-4)
    analytic_efolds = CYAxiverse.paper_benchmarks.n8_hilltop_efolds(1e-7)
    @test isapprox(analytic_efolds.efolds, 6.819e3; rtol=2e-3)

    geometry = CYAxiverse.paper_benchmarks.n8_geometry()
    @test geometry.h11 == 8
    @test geometry.h21 == 28
    @test geometry.volume == 126
    @test geometry.divisor_volumes == [45, 17, 17, 14.5, 14.5, 15.5, 15.5, 25]
    expected_kinetic_eigenvalues = sort([8.20e-4, 6.35e-4, 5.97e-4, 3.13e-4,
                                         1.24e-4, 9.15e-5, 8.30e-5, 5.84e-5])
    @test all(isapprox.(eigvals(geometry.kinetic), expected_kinetic_eigenvalues; rtol=6e-3))
end

@testset "HP spectrum: one-axion analytic mass" begin
    # V = 10^-20 * (1 - cos(3θ)); second instanton is exactly absent.
    K = Hermitian(reshape([4.0], 1, 1))
    Q = reshape(Int[3, 6], 1, 2)
    L = [1.0 0.0;
         -20.0 -1000.0]

    # hp_spectrum's low-level interface uses one instanton per row.
    hp = CYAxiverse.generate.hp_spectrum(K, Matrix(L'), Matrix(Q'); prec=200)

    # λ = Λ q² / K, followed by hp_spectrum's output-unit conversion.
    expected_mass = 0.5 * (-20.0 + log10(3^2 / 4.0)) +
                    log10(2.435e18) + 9.0 + log10(2π)

    @test isapprox(hp["m"][1], expected_mass; atol=1e-10)
    @test hp["msign"][1] == 1

    # The canonical charge is q/sqrt(K) = 3/2, so the fourth derivative is
    # Lambda * (3/2)^4 in hp_spectrum's reported units.
    expected_quartic = -20.0 + 4 * log10(3 / 2) + 4 * log10(2π)
    @test hp["λselfsign"][1] == 1
    @test isapprox(hp["λself"][1], expected_quartic; atol=1e-10)
end
@testset "PQ and HP spectrum: diagonal two-axion comparison" begin
    # The leading instantons act on separate axions, so the PQ construction is
    # exact. The third, zero-sign instanton only satisfies N > h11 and makes
    # no contribution to the potential.
    K = Hermitian([4.0 0.0;
                   0.0 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]

    selected = CYAxiverse.generate.LQtilde(Q, L)
    @test size(selected.Qtilde) == (2, 2)
    @test rank(selected.Qtilde) == 2
    @test selected.Qtilde == Q[:, 1:2]
    @test selected.Ltilde == L[:, 1:2]

    # The candidate scan itself reuses its workspaces, including for dependent
    # columns encountered before the basis is complete.
    Qsorted = Q[:, [1, 3, 2]]
    mask = falses(3)
    span = zeros(2, 2)
    residual = zeros(2)
    CYAxiverse.generate.leading_independent_mask!(mask, Qsorted, span, residual)
    @test mask == Bool[true, false, true]
    @test @allocated(CYAxiverse.generate.leading_independent_mask!(mask, Qsorted, span, residual)) == 0

    pq = CYAxiverse.generate.pq_spectrum(K, L, Q; quartic_diagnostics=true, mass_basis_diagnostics=true, hierarchy_diagnostics=true)
    hp = CYAxiverse.generate.hp_spectrum(K, Matrix(L'), Matrix(Q'); prec=200)

    expected = sort([
        0.5 * (-20.0 + log10(3^2 / 4.0)),
        0.5 * (-30.0 + log10(5^2 / 9.0)),
    ] .+ log10(2.435e18) .+ 9.0 .+ log10(2π))

    @test all(isapprox.(pq.m, expected; atol=1e-10))
    @test pq.msign == [1, 1]
    @test all(isapprox.(hp["m"], expected; atol=1e-10))
    @test pq.msign == hp["msign"]
    @test all(isapprox.(pq.m, hp["m"]; atol=1e-6))
    @test pq.λselfsign == hp["λselfsign"]
    @test all(isapprox.(pq.λself, hp["λself"]; atol=1e-10))
    @test all(isapprox.(pq.quartic_diagnostics.self.orders_lost, zeros(2); atol=1e-12))
    @test all(pq.quartic_diagnostics.self.reliable)
    @test !any(pq.quartic_diagnostics.self.exact_zero)
    @test maximum(pq.mass_basis_diagnostics.eigenpair_residuals) < 1e-12
    @test pq.mass_basis_diagnostics.orthogonality_error < 1e-12
    @test minimum(pq.mass_basis_diagnostics.nearest_relative_gaps) > 0.9
    @test pq.instanton_hierarchy.leading_log_gap == 10.0
    @test pq.instanton_hierarchy.log_scale_span == 980.0
    @test !pq.instanton_hierarchy.heuristic_strong_hierarchy
    @test CYAxiverse.generate.pq_spectrum(K, L, Q).quartic_diagnostics === nothing
    @test CYAxiverse.generate.pq_spectrum(K, L, Q).mass_basis_diagnostics === nothing
    @test CYAxiverse.generate.pq_spectrum(K, L, Q).instanton_hierarchy === nothing

    physical = CYAxiverse.generate.pq_physical_spectrum(K, L, Q; prec=200)
    @test physical.mode_indices == [0, 1]
    @test all(isapprox.(physical.m, expected; atol=1e-10))
    @test physical.λselfsign == hp["λselfsign"]
    @test all(isapprox.(physical.λself, hp["λself"]; atol=1e-10))
    hybrid = CYAxiverse.generate.pq_hybrid_physical_spectrum(K, L, Q; prec=200)
    @test hybrid.mode_indices == [0, 1]
    @test all(isapprox.(hybrid.m, expected; atol=1e-10))
    @test hybrid.λselfsign == hp["λselfsign"]
    @test all(isapprox.(hybrid.λself, hp["λself"]; atol=1e-10))
    hybrid_sparse = CYAxiverse.generate.pq_hybrid_physical_spectrum(
        K, L, Q; prec=200, quartics=true, mixed_quartics=false,
        quartic_backend=:sparse)
    @test all(isapprox.(hybrid_sparse.m, hybrid.m; atol=1e-10))
    @test hybrid_sparse.λselfsign == hybrid.λselfsign
    @test all(isapprox.(hybrid_sparse.λself, hybrid.λself; atol=1e-10))
    @test CYAxiverse.generate.select_quartic_backend(Q, :auto) == :dense
    dispatch_probe = zeros(Int, 200, 500)
    for column in axes(dispatch_probe, 2)
        dispatch_probe[mod1(column, 200), column] = 1
    end
    @test CYAxiverse.generate.select_quartic_backend(dispatch_probe, :auto) == :sparse
    hybrid_masses_only = CYAxiverse.generate.pq_hybrid_physical_spectrum(K, L, Q; prec=200, quartics=false)
    @test all(isapprox.(hybrid_masses_only.m, expected; atol=1e-10))
    @test isempty(hybrid_masses_only.λself)
    @test CYAxiverse.generate.pq_physical_mode_count(K, L, Q; prec=200) == 2
    @test CYAxiverse.generate.pq_schur_admissible(K, L, Q; prec=200)
    @test_logs (:warn, r"geometry=diagonal test") @test CYAxiverse.generate.pq_physical_mode_count(K, L, Q; prec=100, max_prec=100, label="diagonal test") == 2
end

@testset "PQ spectrum: mass-sign propagation" begin
    K = Hermitian(reshape([4.0], 1, 1))
    Q = reshape(Int[3, 6], 1, 2)
    L = [-1.0 0.0;
         -20.0 -1000.0]

    @test CYAxiverse.generate.pq_spectrum(K, L, Q).msign == [-1]
    @test CYAxiverse.generate.pq_spectrum(K, L, Q; mixing_correction=:high_precision, prec=200).msign == [-1]
    @test CYAxiverse.generate.pq_spectrum(K, L, Q; mixing_correction=false).msign == [-1]
end

@testset "PQ spectrum: non-diagonal kinetic matrix" begin
    # A non-diagonal K catches the side on which the Cholesky inverse acts.
    K = Hermitian([4.0 1.2;
                   1.2 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]

    pq_legacy = CYAxiverse.generate.pq_spectrum(K, L, Q; mixing_correction=false)
    pq = CYAxiverse.generate.pq_spectrum(K, L, Q)
    pq_corrected = CYAxiverse.generate.pq_spectrum(K, L, Q; mixing_correction=true, prec=200)
    hp = CYAxiverse.generate.hp_spectrum(K, Matrix(L'), Matrix(Q'); prec=200)

    @test all(isapprox.(pq_legacy.m, hp["m"]; atol=0.1))
    @test all(isapprox.(pq.m, hp["m"]; atol=1e-6))
    @test all(isapprox.(pq.λself, hp["λself"]; atol=1e-10))
    @test all(isapprox.(pq_corrected.m, hp["m"]; atol=1e-10))
    @test pq_corrected.λselfsign == hp["λselfsign"]
    @test all(isapprox.(pq_corrected.λself, hp["λself"]; atol=1e-10))
    @test pq_corrected.λ22sign == hp["λ22sign"]
    @test all(isapprox.(pq_corrected.λ22, hp["λ22"]; atol=1e-10))
end

@testset "PQ-seeded HP alignment" begin
    K = Hermitian([4.0 0.0;
                   0.0 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]

    alignment = CYAxiverse.generate.pq_hp_alignment(K, L, Q; prec=200)

    @test alignment.permutation == [1, 2]
    @test all(isapprox.(alignment.aligned_overlap, ones(2); atol=1e-12))
    @test all(alignment.residuals .< 1e-100)
end

@testset "PQ vacua-pipeline spectrum persistence" begin
    K = Hermitian([4.0 0.0;
                   0.0 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]
    spectrum = CYAxiverse.generate.pq_spectrum(K, L, Q)
    geom_idx = CYAxiverse.structs.GeometryIndex(2, 1, 1)
    vac_est = (vac=3.0, issquare=1, extrarows=0)
    vac_id = Dict{String, Any}(
        "vac" => 3,
        "θ̃∥" => Rational{Int}[1//1 0//1; 0//1 1//1],
    )

    previous_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
    try
        mktempdir() do data_dir
            ENV["CYAXIVERSE_DATA_DIR"] = data_dir
            mkpath(joinpath(data_dir, "h11_002", "np_0000001", "cy_0000001"))
            path = CYAxiverse.filestructure.cyax_file(geom_idx)
            h5open(path, "w") do file
                spectrum_group = create_group(file, "spectrum")
                masses = create_group(spectrum_group, "masses")
                masses["log10"] = spectrum.m
                masses["sign"] = spectrum.msign
                decay = create_group(spectrum_group, "decay")
                decay["fK"] = spectrum.fK
                decay["fpert"] = spectrum.f
                quartdiag = create_group(spectrum_group, "quartdiag")
                quartdiag["log10"] = spectrum.λself
                quartdiag["sign"] = spectrum.λselfsign
            end

            save_axion_data(geom_idx, spectrum, vac_est, vac_id; threshold=0.5)

            h5open(path, "r") do file
                @test HDF5.read(file, "spectrum/masses/log10") == spectrum.m
                @test HDF5.read(file, "spectrum/masses/sign") == spectrum.msign
                @test HDF5.read(file, "spectrum/decay/fK") == spectrum.fK
                @test HDF5.read(file, "spectrum/decay/fpert") == spectrum.f
                @test HDF5.read(file, "spectrum/quartdiag/log10") == spectrum.λself
                @test HDF5.read(file, "spectrum/quartdiag/sign") == spectrum.λselfsign
            end

            saved_vacua = CYAxiverse.read.pipeline_vacua(geom_idx)
            @test saved_vacua.threshold == 0.5
            @test saved_vacua.estimate == 3.0
            @test saved_vacua.verified == 3
            @test saved_vacua.theta_parallel == vac_id["θ̃∥"]
        end
    finally
        if previous_data_dir === nothing
            pop!(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        else
            ENV["CYAXIVERSE_DATA_DIR"] = previous_data_dir
        end
    end

    @testset "inflation reproduction contracts" begin
        benchmark = CYAxiverse.paper_benchmarks
        poly102 = benchmark.poly102_inflation
        n8 = benchmark.n8_potential(k=benchmark.N8_KC; trajectory=true)
        @test size(n8.Q) == (8, 10)
        @test n8.phases == zeros(10)
        @test n8.qdotτ == [14.0, 14.5, 14.5, 15.5, 15.5, 15.5, 15.5, 16.0, 17.0, 17.0]
        appendix = benchmark.n8_potential(k=benchmark.N8_KC)
        @test size(appendix.Q) == (8, 12)
        @test appendix.qdotτ[end-1:end] == [25.0, 45.0]

        geometry = benchmark.n8_geometry()
        expected_metric_eigenvalues = sort([
            8.20e-4, 6.35e-4, 5.97e-4, 3.13e-4,
            1.24e-4, 9.15e-5, 8.30e-5, 5.84e-5,
        ])
        @test all(isapprox.(eigvals(geometry.kinetic), expected_metric_eigenvalues; rtol=6e-3))
        k_detuned = benchmark.N8_KC + 1e-3
        @test Matrix(benchmark.n8_kinetic_matrix(k_detuned)) ≈
            Matrix(benchmark.n8_kinetic_matrix(benchmark.N8_KC)) *
            (benchmark.N8_KC / k_detuned)^2
        @test Matrix(benchmark.n8_kinetic_matrix(1.0)) ≈ Matrix(geometry.kinetic)

        critical = poly102.n8_degenerate_point()
        @test critical.converged
        @test isapprox(critical.k, 0.674506370003365; atol=1e-15)
        @test critical.gradient_residual < 1e-10
        @test critical.null_residual < 1e-10

        mass_basis = poly102.n8_mass_eigenbasis(critical.k)
        @test mass_basis.basis == :mass_eigenbasis
        @test mass_basis.orthonormality_residual < 1e-10
        @test mass_basis.generalized_residual < 1e-10
        @test all(isapprox.(mass_basis.raw_eigenvectors' *
            mass_basis.metric * mass_basis.raw_eigenvectors, Matrix(I, 8, 8);
            atol=1e-10))
        mass_direction = poly102.n8_unstable_direction(
            critical.k; basis=:mass_eigenbasis)
        canonical_direction = poly102.n8_unstable_direction(
            critical.k; basis=:canonical_hessian)
        @test mass_direction.basis == :mass_eigenbasis
        @test canonical_direction.basis == :canonical_hessian
        @test abs(dot(mass_direction.raw,
            mass_direction.metric * canonical_direction.raw)) > 1 - 1e-10
        @test norm(mass_direction.hessian_theta * mass_direction.raw -
            mass_direction.metric * mass_direction.raw *
            mass_direction.eigenvalues[mass_direction.index]) < 1e-10

        initial = poly102.n8_inflation_initial_condition(critical.k + 1e-7)
        @test isapprox(initial.canonical_norm, 1e-8; rtol=1e-8)
        @test isapprox(initial.canonical_norm,
            sqrt(dot(initial.theta - initial.theta_critical,
                Matrix(poly102.n8_kinetic_matrix(critical.k + 1e-7)) *
                (initial.theta - initial.theta_critical))); rtol=1e-8)
        mass_initial = poly102.n8_inflation_initial_condition(
            critical.k + 1e-7; basis=:mass_eigenbasis)
        @test mass_initial.basis == :mass_eigenbasis
        @test isapprox(mass_initial.canonical_norm, 1e-8; rtol=1e-8)
        @test mass_initial.basis_theta == poly102.N8_BEST_X
        @test mass_initial.basis_k == critical.k + 1e-7

        audit = poly102.n8_basis_directions(critical.k + 1e-7)
        @test hasproperty(audit.directions, :E_mass_eigenbasis)
        @test audit.equivalent_mass_direction > 1 - 1e-10

        n8_tuned = poly102.n8_hilltop_normal_form_efolds(1e-7).efolds
        @test isapprox(n8_tuned, 463115.0; rtol=0.01)
        n8_sixty = poly102.n8_hilltop_normal_form_efolds(1.5320548620798324e-3).efolds
        @test isapprox(n8_sixty, 60.0; rtol=0.08)

        @test isapprox(poly102.n5_critical_scale(), 0.674506370003365; atol=1e-15)
        @test isapprox(poly102.n5_reduced_ratio(poly102.n5_critical_scale()), 0.25; atol=1e-15)
        n5_geometry = poly102.n5_geometry()
        @test n5_geometry.h11 == 5
        @test n5_geometry.h21 == 75
        @test n5_geometry.euler == -140
        @test isapprox(n5_geometry.volume, 149.3958333333367; rtol=1e-13)
        @test n5_geometry.divisor_volumes == [6.0, 24.0, 36.125, 32.0, 6.25]
        @test n5_geometry.vertices == poly102.N5_VERTICES
        n5_light = poly102.n5_light_direction()
        @test isapprox(dot(n5_light.direction,
            n5_light.metric * n5_light.direction), 1.0; atol=1e-12)
        @test Matrix(poly102.n5_kinetic_matrix(1.0)) ≈ Matrix(n5_geometry.kinetic)
        n8_probe = poly102.n8_hilltop_normal_form(1e-7; sample_count=5)
        @test n8_probe.end_event == :local_normal_form
        @test length(n8_probe.samples) == 5
        @test isapprox(n8_probe.efolds, poly102.n8_hilltop_normal_form_efolds(1e-7).efolds)
        mass_trajectory = poly102.n8_efold_gradient_flow(
            1e-7; basis=:mass_eigenbasis, max_efolds=0)
        @test mass_trajectory.basis == :mass_eigenbasis
        @test mass_trajectory.steps == 0
        @test isapprox(poly102.n5_hilltop_normal_form_efolds(1e-7).efolds, 27349.0; rtol=1e-10)
        @test isapprox(poly102.n5_hilltop_normal_form_efolds(6.65e-5).efolds, 60.0; rtol=0.02)

        @testset "inflation trajectory contracts" begin
            maps = poly102.n8_coordinate_maps(k_detuned)
            @test maps.raw_to_canonical' * maps.raw_to_canonical ≈ maps.metric atol=1e-12
            @test maps.canonical_to_raw * maps.raw_to_canonical ≈ Matrix(I, 8, 8) atol=1e-12

            theta = poly102.N8_BEST_X .+ 0.01 .* collect(1:8)
            derivatives = poly102.n8_potential_derivatives(
                theta, k_detuned; trajectory=true)
            finite_difference_step = 1e-6
            finite_difference_raw = [
                let plus=copy(theta), minus=copy(theta)
                    plus[index] += finite_difference_step
                    minus[index] -= finite_difference_step
                    (poly102.n8_potential_derivatives(
                        plus, k_detuned; trajectory=true).value -
                     poly102.n8_potential_derivatives(
                        minus, k_detuned; trajectory=true).value) /
                    (2finite_difference_step)
                end
                for index in eachindex(theta)
            ]
            @test norm(derivatives.gradient - finite_difference_raw, Inf) < 1e-8

            chi = maps.raw_to_canonical * theta
            canonical_gradient = maps.canonical_to_raw' * derivatives.gradient
            finite_difference_canonical = [
                let plus=copy(chi), minus=copy(chi)
                    plus[index] += finite_difference_step
                    minus[index] -= finite_difference_step
                    (poly102.n8_potential_derivatives(
                        maps.canonical_to_raw * plus, k_detuned;
                        trajectory=true).value -
                     poly102.n8_potential_derivatives(
                        maps.canonical_to_raw * minus, k_detuned;
                        trajectory=true).value) /
                    (2finite_difference_step)
                end
                for index in eachindex(chi)
            ]
            @test norm(canonical_gradient - finite_difference_canonical, Inf) < 1e-8

            trajectory_initial = poly102.n8_inflation_initial_condition(
                k_detuned; basis=:mass_eigenbasis)
            metric = Matrix(poly102.n8_kinetic_matrix(k_detuned))
            @test isapprox(dot(trajectory_initial.initial_tangent,
                metric * trajectory_initial.initial_tangent), 1.0; atol=1e-10)
            moving_basis = poly102.n8_mass_eigenbasis(
                k_detuned; theta=trajectory_initial.theta)
            fixed_index = argmin(trajectory_initial.basis_eigenvalues)
            moving_index = argmin(moving_basis.eigenvalues)
            @test trajectory_initial.basis_theta == poly102.N8_BEST_X
            @test norm(trajectory_initial.basis_raw_eigenvectors[:, fixed_index] -
                moving_basis.raw_eigenvectors[:, moving_index]) > 1e-6

            refinement_config = inflation_refinement_config(
                precision_bits=64, max_time=10, max_step=1,
                initial_step=1e-5, sample_count=1, reltol=1e-8,
                abstol=1e-10, maxiters=1_000_000,
                measurement_scope=:cold)
            refinement_candidate = inflation_refinement_candidate(
                "trajectory-contract"; delta_k=1.5320548620798324e-3,
                screening=(status=:candidate, measurement_scope=:cold,
                    value=1.0, epsilon=0.5, min_eta=-0.5, negative_modes=1,
                    wall_seconds=0.001, allocated_bytes=123,
                    output_bytes=64))
            refined = refine_inflation_candidate(refinement_candidate;
                config=refinement_config)
            @test refined.summary.refinement_status == :completed
            @test refined.summary.event_policy == :final_finite_exit
            @test refined.summary.measurement_status == :completed
            @test refined.summary.measurement_scope == :cold
            @test refined.summary.accepted_steps > 0
            @test _refinement_solver_status(ReturnCode.Success) ==
                (:completed, "")
            @test first(_refinement_solver_status(ReturnCode.MaxIters)) ==
                :failed
            short_flow = refined.trajectory
            @test short_flow.entered_slow_roll
            @test short_flow.end_event == :tmax
            @test !short_flow.terminated
            diagnostic_row = inflation_refinement_diagnostic_row(
                refinement_candidate, refined)
            serialization = inflation_stage_measure(
                () -> inflation_diagnostic_csv_line(diagnostic_row);
                measurement_scope=:warm)
            diagnostic_row = inflation_refinement_diagnostic_row(
                refinement_candidate, refined; serialization)
            @test diagnostic_row.screen_status == :candidate
            @test diagnostic_row.screen_epsilon == 0.5
            @test diagnostic_row.refinement_status == :completed
            @test diagnostic_row.serialization_status == :completed
            @test diagnostic_row.serialization_measurement_scope == :warm
            @test diagnostic_row.serialization_output_bytes > 0
            @test occursin("candidate_id", inflation_diagnostic_csv_line(
                diagnostic_row; header=true))
            diagnostic_path = joinpath(mktempdir(), "stage4.csv")
            written = inflation_append_diagnostic_row(diagnostic_path,
                diagnostic_row; measurement_scope=:warm, header=true)
            @test written.status == :completed
            @test written.measurement_scope == :warm
            @test isfile(diagnostic_path)
            @test count(==('\n'), read(diagnostic_path, String)) == 2
            failed_measurement = inflation_stage_measure(
                () -> throw(ArgumentError("stage-4 failure"));
                measurement_scope=:warm, capture_errors=true)
            @test failed_measurement.status == :failed
            @test occursin("stage-4 failure", failed_measurement.error)
            not_selected = refine_inflation_candidate(
                inflation_refinement_candidate("not-selected";
                    delta_k=1e-3, accepted=false); config=refinement_config)
            @test not_selected.summary.refinement_status == :not_selected
            @test not_selected.summary.allocated_bytes == 0
            unsupported = refine_inflation_candidate(
                inflation_refinement_candidate("unsupported";
                    model=:unregistered, delta_k=1e-3);
                config=refinement_config)
            @test unsupported.summary.refinement_status == :unsupported_model
            @test_throws ArgumentError poly102.n8_physical_gradient_flow(
                1.5320548620798324e-3; precision_bits=64, max_time=1,
                method=:FBDF)
        end
    end
end
