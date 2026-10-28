# Full-only testsets. Included once from runtests.jl under `if _FULL`, so a
# misplaced `end` cannot silently change a test's mode.

@testset "Cached screening Hessian eigensolver" begin
    rng = MersenneTwister(20261026)
    for dimension in (3, 8)
        matrix = randn(rng, dimension, dimension)
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
    withenv("CYAXIVERSE_DATA_DIR" => nothing) do
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

@testset "Vacua pipeline persistence and validation" begin
    migrated_metadata = _normalize_vacua_search_metadata((
        search_classification="exact_determinant_branch", search_method="reduced_jlm"))
    @test migrated_metadata.search_classification ==
        "square_reduced_potential_determinant_count"
    @test migrated_metadata.legacy_search_classification ==
        "exact_determinant_branch"
    @test migrated_metadata.classification_schema_version ==
        "cyaxiverse-vacua-classification-2"

    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        path = joinpath(geom_dir, "cyax.h5")
        h5open(path, "cw") do file
            spectrum = create_group(file, "spectrum")
            spectrum["sentinel"] = Int[17, 23]
        end

        withenv("CYAXIVERSE_DATA_DIR" => root) do
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
                    search_classification="finite_multistart_minimum_lower_bound",
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
                "--merge-tolerance", "1e-6", "--max-iterations", "12", "--method", "legacy",
                "--dry-run"])
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
            @test default_blas_options[:method] == :auto
            @test default_blas_options[:blas_threads] == 1
            repository_root = normpath(joinpath(@__DIR__, ".."))
            if in_git_checkout(repository_root)
                coverage_files_cmd = `git -C $repository_root ls-files --others --exclude-standard -- '*.cov'`
                coverage_files_before = Set(readlines(coverage_files_cmd))
            end
            @test !run_vacua_batch(parallel_options;
                worker_exeflags=Cmd(child_coverage_args()))
            if in_git_checkout(repository_root)
                @test Set(readlines(coverage_files_cmd)) == coverage_files_before
            end
            @test occursin("failed", read(parallel_summary, String))

            # Geometry 2,1,1 above has no potential, so failure is expected there. The
            # valid geometry 2,2,1 built earlier must succeed through the same workers.
            # method=auto: the legacy vacua_id_basis raises BoundsError on this
            # potential (src/generate.jl:3878, reported separately).
            valid_summary = joinpath(root, "parallel-valid.csv")
            valid_options = _vacua_parse_args(["--data-dir", root,
                "--geometry", "2,2,1", "--workers", "2", "--blas-threads", "1",
                "--method", "auto", "--summary", valid_summary, "--force"])
            @test run_vacua_batch(valid_options; worker_exeflags=Cmd(child_coverage_args()))
            @test !occursin("failed", read(valid_summary, String))
            # One minimum: asserted above for this same potential via compute_vacua_data
            # (Q has an identity leading block, so the determinant count is 1).
            @test CYAxiverse.read.pipeline_vacua(
                CYAxiverse.structs.GeometryIndex(2, 2, 1)).estimate == 1

            mismatched_options = _vacua_parse_args(["--data-dir", root,
                "--geometry", "2,1,1", "--threshold", "0.6", "--dry-run"])
            @test _vacua_result_state(path, _vacua_config(mismatched_options)) != :matching
            @test !run_vacua_batch(mismatched_options)
            pipeline_data = CYAxiverse.read.pipeline_vacua(geom_idx)
            @test pipeline_data.estimate == 3
            @test pipeline_data.metadata.status == "completed"
            @test pipeline_data.metadata.terminal_status == "completed"
            @test pipeline_data.metadata.configuration_digest isa AbstractString
            @test pipeline_data.metadata.starts == 17
            @test pipeline_data.metadata.method == "legacy"
            @test pipeline_data.metadata.verification_status == "not_applicable"
            @test pipeline_data.metadata.search_method == "legacy"
            @test pipeline_data.metadata.search_classification ==
                "finite_multistart_minimum_lower_bound"
            @test pipeline_data.metadata.legacy_search_classification ==
                "finite_search_lower_bound"
            @test pipeline_data.metadata.classification_schema_version ==
                "cyaxiverse-vacua-classification-2"
            @test pipeline_data.metadata.model_scope ==
                "hierarchy_truncated_axion_potential"
            @test pipeline_data.metadata.full_potential_status == "not_validated"
            @test_throws ArgumentError save_axion_data(geom_idx, spectrum, estimate, identified;
                threshold=0.5, force=false)
            save_axion_data(geom_idx, spectrum, estimate, identified;
                threshold=0.5, force=true)

            # A failed terminal record is explicit persisted state. Repairing
            # it requires the same explicit force acknowledgement as a
            # completed record.
            h5open(path, "r+") do file
                HDF5.delete_object(file, "vacua_pipeline/metadata/status")
                file["vacua_pipeline/metadata/status"] = "failed"
                HDF5.delete_object(file, "vacua_pipeline/metadata/terminal_status")
                file["vacua_pipeline/metadata/terminal_status"] = "failed"
            end
            @test !_has_pipeline_result(path)
            @test !_has_pipeline_group(path)
            @test_throws ArgumentError save_axion_data(geom_idx, spectrum, estimate, identified;
                threshold=0.5, starts=17, residual_tolerance=1e-9,
                merge_tolerance=1e-6, max_iterations=12, force=false,
                search_metadata=(search_method="legacy",
                    search_classification="finite_multistart_minimum_lower_bound",
                    minimum_count=3, multiplicity=1.0, critical_count=-1,
                    branch_count=-1, det_Qtilde=-1, search_status="completed"))
            save_axion_data(geom_idx, spectrum, estimate, identified;
                threshold=0.5, starts=17, residual_tolerance=1e-9,
                merge_tolerance=1e-6, max_iterations=12, force=true,
                search_metadata=(search_method="legacy",
                    search_classification="finite_multistart_minimum_lower_bound",
                    minimum_count=3, multiplicity=1.0, critical_count=-1,
                    branch_count=-1, det_Qtilde=-1, search_status="completed"))
            @test _has_pipeline_result(path)

            bad_potential = (Q=zeros(Int, 3, 4), L=zeros(2, 4),
                K=Hermitian(Matrix{Float64}(I, 2, 2)))
            @test_throws DimensionMismatch _validate_potential(geom_idx, bad_potential)
        end
    end
end


@testset "Orientifold inflation persistence preserves precision and requires force" begin
    mktempdir() do data_dir
        withenv("CYAXIVERSE_DATA_DIR" => data_dir) do
            geom_idx = CYAxiverse.structs.GeometryIndex(2, 7, 1)
            geom_dir = joinpath(data_dir, "h11_002", "np_0000007", "cy_0000001")
            mkpath(geom_dir)
            path = joinpath(geom_dir, "cyax.h5")
            h5open(path, "w") do file
                file["geometry_sentinel"] = Int[1]
            end
            precise = setprecision(BigFloat, 256) do
                parse(BigFloat, "1.234567890123456789012345678901234567890123456789")
            end
            slow_precise = setprecision(BigFloat, 256) do
                parse(BigFloat, "2.34567890123456789012345678901234567890123456789")
            end
            flow_row = (; precision_bits=256, flow_status=:completed,
                flow_end_event=:exit, flow_efolds=precise,
                flow_slow_roll_efolds=slow_precise, flow_accepted=true,
                candidate_index=1, displacement_sign=1,
                leading_negative_modes=1)
            catastrophe = (; status=:success,
                result=(leading_minima_count=1, saddle_count=0, branch_count=1,
                    qtilde_det=1, mass_min=1.0, mass_max=2.0,
                    negative_mass_count=1, candidate_slowroll_saddles=0))
            efolds = (; status=:success,
                result=(refinement=(search=(search_status=:completed,
                    coverage_status=:complete, refinement_status=:completed),
                    candidates=[flow_row]), flow=(rows=[flow_row], qualified=1)))
            write_inflation_group!(geom_idx, catastrophe, efolds;
                catastrophe_settings=(; max_branches=10),
                efold_settings=(; max_branches=10, precision_bits=256,
                    float_tolerance=1e-10, high_tolerance=1e-40,
                    max_points=10, min_efolds=50, max_efolds=60,
                    flow_step=1e-3, flow_displacement=1e-8),
                pipeline_config_digest="fixture-digest")

            h5open(path, "r") do file
                trajectories = file["inflation/efolds/trajectories"]
                flow_bytes = read(trajectories["flow_efolds"])
                flow_lengths = read(trajectories["flow_efolds_lengths"])
                slow_bytes = read(trajectories["flow_slow_roll_efolds"])
                slow_lengths = read(trajectories["flow_slow_roll_efolds_lengths"])
                decode = (bytes, lengths, index) ->
                    String(bytes[1:lengths[index], index])
                flow_text = decode(flow_bytes, flow_lengths, 1)
                slow_text = decode(slow_bytes, slow_lengths, 1)
                @test flow_text == string(precise)
                @test slow_text == string(slow_precise)
                flow_round_trip = setprecision(
                    Int(read(trajectories["flow_precision_bits_by_row"])[1])
                ) do
                    parse(BigFloat, flow_text)
                end
                slow_round_trip = setprecision(
                    Int(read(trajectories["flow_precision_bits_by_row"])[1])
                ) do
                    parse(BigFloat, slow_text)
                end
                @test flow_round_trip == precise
                @test slow_round_trip == slow_precise
                @test read(trajectories["flow_precision_bits"]) == 256
                @test read(trajectories["flow_precision_bits_by_row"]) == [256]
                @test read(trajectories["flow_numeric_encoding"]) == "decimal_string"
                @test read(file["inflation/terminal_status"]) == "completed"
                @test read(file["inflation/configuration_digest"]) == "fixture-digest"
                @test read(file["inflation/schema_version"]) ==
                    "cyaxiverse-phase3-orientifold-inflation-2.3"
                @test read(file["inflation/domain_certificate_version"]) ==
                    "physical-domain-certificate-2"
                @test read(file["inflation/scale_status"]) == "homotopy_only"
                @test read(file["inflation/domain_status"]) == "out_of_model"
                @test read(file["inflation/fixed_point_status"]) == "homotopy_only"
                @test read(file["inflation/trajectory_status"]) == "homotopy_only"
                @test read(file["inflation/moduli_status"]) == "not_established"
                @test read(file["inflation/normalization"]) == "homotopy_only"
                @test read(file["inflation/physical_units_contract"]) ==
                    "M_s=M_Pl;k=dimensionless"
                @test read(file["inflation/physical_scaling_gate_status"]) ==
                    "not_applicable"
                @test occursin("does not evaluate", read(file[
                    "inflation/physical_scaling_gate_reason"]))
                @test startswith(read(file["inflation/physical_scaling_gate_provenance"]),
                    "scripts/inflation_scale_continuation.jl::pilot_homotopy_scale")
                @test read(file["inflation/physical_control_gate_status"]) ==
                    "not_applicable"
                @test read(file["inflation/physical_viability_status"]) ==
                    "not_applicable"
                @test read(file["inflation/catastrophes/physical_scaling_gate_status"]) ==
                    "not_applicable"
                @test read(file["inflation/catastrophes/physical_control_gate_status"]) ==
                    "not_applicable"
                @test read(file["inflation/efolds/physical_viability_status"]) ==
                    "not_applicable"
                @test startswith(read(file["inflation/source_identity"]),
                    "scripts/build_orientifold_vacua_inflation.jl@")
                @test read(file["inflation/precision_bits"]) == 256
            end
            h5open(path, "r+") do file
                write(file["inflation/physical_control_gate_status"], "bogus")
            end
            @test !_has_inflation_group(path; config_digest="fixture-digest")
            @test_throws ArgumentError write_inflation_group!(geom_idx, catastrophe, efolds;
                catastrophe_settings=(; max_branches=10),
                efold_settings=(; max_branches=10, precision_bits=256,
                    float_tolerance=1e-10, high_tolerance=1e-40,
                    max_points=10, min_efolds=50, max_efolds=60,
                    flow_step=1e-3, flow_displacement=1e-8),
                pipeline_config_digest="fixture-digest", force=false)
            write_inflation_group!(geom_idx, catastrophe, efolds;
                catastrophe_settings=(; max_branches=10),
                efold_settings=(; max_branches=10, precision_bits=256,
                    float_tolerance=1e-10, high_tolerance=1e-40,
                    max_points=10, min_efolds=50, max_efolds=60,
                    flow_step=1e-3, flow_displacement=1e-8),
                pipeline_config_digest="fixture-digest", force=true)
            @test _has_inflation_group(path; config_digest="fixture-digest")
        end
    end
end

@testset "Physical spectrum batch persistence" begin
    @test _fpert_log10([18.0], [4.0]) ≈ [7.0]
    # Non-empty λ31/λ22 (one cross-quartic term each) so the round-trip
    # below actually exercises the new lambda_31_*/lambda_22_* datasets,
    # not just their empty-array degenerate case.
    spectrum = CYAxiverse.structs.PhysicalAxionSpectrum(
        [1.0, 2.0], [0, 1], zeros(2, 2), [1, -1], [3.0, 4.0],
        reshape([1, 1, 1, 2], 4, 1), [1], [0.5],
        reshape([1, 1, 2, 2], 4, 1), [-1], [0.7],
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
    completed_config = _physical_spectrum_config(; prec=200,
        threshold_log10=-30.0, quartics=true)
    @test _has_physical_spectrum(output_path; config=completed_config)
    @test !_has_physical_spectrum(output_path; config=_physical_spectrum_config(
        prec=201, threshold_log10=-30.0, quartics=true))
    HDF5.h5open(output_path, "r") do file
        @test HDF5.haskey(file, "spectrum/physical/m")
        @test read(file["spectrum/physical/fK_log10"]) == [5.0, 6.0]
        @test read(file["spectrum/physical/fpert_log10"]) ≈ [-9.5, -9.0]
        @test read(file["spectrum/physical/metadata/mass_log10_unit"]) == "eV"
        @test read(file["spectrum/physical/metadata/fpert_log10_unit"]) == "GeV"
        @test read(file["spectrum/physical/metadata/fpert_formula"]) ==
            "log10(fpert/GeV)=log10(m/eV)-9-0.5*log10(abs(lambda_self))"
        @test read(file["spectrum/physical/metadata/log_domain_policy"]) ==
            "not_recorded_for_fixture"
        @test read(file["spectrum/physical/lambda_31_sign"]) == [1]
        @test read(file["spectrum/physical/lambda_31_log10"]) == [0.5]
        @test read(file["spectrum/physical/lambda_31_indices"]) == reshape([1, 1, 1, 2], 4, 1)
        @test read(file["spectrum/physical/lambda_22_sign"]) == [-1]
        @test read(file["spectrum/physical/lambda_22_log10"]) == [0.7]
        @test read(file["spectrum/physical/lambda_22_indices"]) == reshape([1, 1, 2, 2], 4, 1)
    end
    withenv("CYAXIVERSE_DATA_DIR" => output_dir) do
        loaded = CYAxiverse.read.physical_spectrum(geom_idx)
        @test loaded.m == spectrum.m
        @test loaded.fpert ≈ [-9.5, -9.0]
        @test loaded.schema_version == "cyaxiverse-physical-spectrum-3"
        @test loaded.fpert_log10_unit == "GeV"
        @test loaded.fpert_convention == "lambda_self_abs_log10"
        @test loaded.mass_signs_or_inertia == Int[]
    end

    _write_result(output_path, geom_idx, spectrum; prec=200, threshold_log10=-30.0,
        quartics=false, runtime_seconds=0.2, provisional=false, fK=[5.0, 6.0])
    HDF5.h5open(output_path, "r") do file
        @test !HDF5.haskey(file, "spectrum/physical/mass_signs_or_inertia")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_self_sign")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_self_log10")
        @test !HDF5.haskey(file, "spectrum/physical/fpert_log10")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_31_sign")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_31_log10")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_31_indices")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_22_sign")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_22_log10")
        @test !HDF5.haskey(file, "spectrum/physical/lambda_22_indices")
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

    mktempdir() do data_dir
        withenv("CYAXIVERSE_DATA_DIR" => data_dir) do
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
    end
end

@testset "Quartic index migration: relabel old two-dimensional order" begin
    # The pre-fix `hp_spectrum` built its quartic index lists with a
    # two-dimensional comprehension, which iterates column-major, while the
    # fused accumulation loop that fills the corresponding values runs the
    # first index slowest. This fixture writes an `index` dataset in that old
    # order and checks that the migration rewrites it to the order the
    # current source actually produces, obtained from a real call into
    # `hp_spectrum` rather than a formula duplicated in this test.
    h11 = 4
    Q = [1  0  0  0  1 -1  2  0;
         0  1  0  0  1  0  1  1;
         0  0  1  0  0  1  0  2;
         0  0  0  1  1  1  1  0]
    L = [   1.0    1.0    1.0    1.0    1.0   -1.0    1.0    1.0;
          -20.0  -22.0  -25.0  -27.0  -30.0  -33.0  -36.0  -40.0]
    K = Hermitian([2.0 0.3 0.1 0.0;
                   0.3 1.5 0.2 0.1;
                   0.1 0.2 1.8 0.4;
                   0.0 0.1 0.4 1.2])
    hp = CYAxiverse.generate.hp_spectrum(K, L, Q; prec = 120)
    current_index31 = hp["λ31_i"]
    current_index22 = hp["λ22_i"]
    @test size(current_index31, 2) == h11 * (h11 - 1)
    @test size(current_index22, 2) == h11 * (h11 - 1) ÷ 2

    # The pre-fix column-major comprehension, exactly as it read before the
    # source fix (see commit "Fix quartic component labelling and basis_snf
    # inverse"). This is independent of the migration script's own copy of
    # the same formula: it is typed here from the historical source, not
    # imported from the script under test.
    old_components_31 = [(x, x, x, y) for x = 1:h11, y = 1:h11 if x != y]
    old_components_22 = [(x, x, y, y) for x = 1:h11, y = 1:h11 if x > y]
    to_zero_based_matrix(components) = begin
        matrix = zeros(Int, 4, length(components))
        for column in eachindex(components), row in 1:4
            matrix[row, column] = components[column][row] - 1
        end
        matrix
    end
    old_index31 = to_zero_based_matrix(old_components_31)
    old_index22 = to_zero_based_matrix(old_components_22)
    @test old_index31 != current_index31
    @test old_index22 != current_index22

    # Known, arbitrary values unrelated to the physics: only their byte
    # identity across the migration is being tested.
    log10_31 = collect(1.0:size(old_index31, 2))
    sign_31 = [isodd(k) ? 1 : -1 for k in 1:size(old_index31, 2)]
    log10_22 = collect(101.0:100.0 + size(old_index22, 2))
    sign_22 = [isodd(k) ? -1 : 1 for k in 1:size(old_index22, 2)]

    mktempdir() do root
        geom_dir = joinpath(root, "h11_004", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        path = joinpath(geom_dir, "cyax.h5")
        h5open(path, "cw") do file
            spectrum = create_group(file, "spectrum")
            quart31 = create_group(spectrum, "quart31")
            quart31["log10", deflate=9] = log10_31
            quart31["sign", deflate=9] = sign_31
            quart31["index", deflate=9] = old_index31
            quart22 = create_group(spectrum, "quart22")
            quart22["log10", deflate=9] = log10_22
            quart22["sign", deflate=9] = sign_22
            quart22["index", deflate=9] = old_index22
        end

        # Unrecognised path: a second geometry whose stored index matches
        # neither the old nor the current order. It must be left untouched
        # by every run below, including the final --apply pass.
        garbage_dir = joinpath(root, "h11_004", "np_0000002", "cy_0000001")
        mkpath(garbage_dir)
        garbage_path = joinpath(garbage_dir, "cyax.h5")
        garbage_index31 = old_index31 .+ 99
        h5open(garbage_path, "cw") do file
            spectrum = create_group(file, "spectrum")
            quart31 = create_group(spectrum, "quart31")
            quart31["log10", deflate=9] = log10_31
            quart31["sign", deflate=9] = sign_31
            quart31["index", deflate=9] = garbage_index31
        end

        begin
            # Default is report-only: the on-disk old order must survive a
            # run without --apply.
            dry_options = _quartic_parse_args(["--data-dir", root, "--h11", "4"])
            dry_result = run_quartic_index_migration(dry_options)
            @test dry_result.migrated == 2
            @test dry_result.files_written == 0
            @test dry_result.success
            h5open(path, "r") do file
                @test read(file, "spectrum/quart31/index") == old_index31
                @test read(file, "spectrum/quart22/index") == old_index22
            end

            apply_options = _quartic_parse_args(["--data-dir", root, "--h11", "4", "--apply"])
            apply_result = run_quartic_index_migration(apply_options)
            @test apply_result.migrated == 2
            @test apply_result.already_correct == 0
            @test apply_result.unrecognised == 1
            @test apply_result.files_written == 1
            @test apply_result.success
            @test any(occursin("np_0000002", entry) for entry in apply_result.unrecognised_paths)

            h5open(path, "r") do file
                # The index datasets now carry the labelling the current
                # source actually produces.
                @test read(file, "spectrum/quart31/index") == current_index31
                @test read(file, "spectrum/quart22/index") == current_index22
                # log10 and sign were never opened for writing and must be
                # bit-identical to what was written before migration.
                @test read(file, "spectrum/quart31/log10") == log10_31
                @test read(file, "spectrum/quart31/sign") == sign_31
                @test read(file, "spectrum/quart22/log10") == log10_22
                @test read(file, "spectrum/quart22/sign") == sign_22
            end
            h5open(garbage_path, "r") do file
                @test read(file, "spectrum/quart31/index") == garbage_index31
                @test read(file, "spectrum/quart31/log10") == log10_31
                @test read(file, "spectrum/quart31/sign") == sign_31
            end

            # Re-running must be a no-op: everything is already correct, and
            # the unrecognised dataset is still left untouched.
            second_result = run_quartic_index_migration(apply_options)
            @test second_result.migrated == 0
            @test second_result.already_correct == 2
            @test second_result.unrecognised == 1
            @test second_result.files_written == 0
            @test second_result.success

            h5open(path, "r") do file
                @test read(file, "spectrum/quart31/index") == current_index31
                @test read(file, "spectrum/quart22/index") == current_index22
                @test read(file, "spectrum/quart31/log10") == log10_31
                @test read(file, "spectrum/quart31/sign") == sign_31
                @test read(file, "spectrum/quart22/log10") == log10_22
                @test read(file, "spectrum/quart22/sign") == sign_22
            end
            h5open(garbage_path, "r") do file
                @test read(file, "spectrum/quart31/index") == garbage_index31
            end
        end
    end
end

@testset "Orientifold Pipeline 2 (vacua + inflation) writing boundary" begin
    # A minimal synthetic geometry: 1 axion, 3 instantons (enough for
    # `LQtilde`/`run_geometry`/`scan_geometry_for_inflation` to have a
    # well-posed reduced problem), mirroring the "Visible-sector metadata
    # roundtrip" / "Vacua pipeline persistence" synthetic-fixture pattern
    # used elsewhere in this file rather than depending on real CYTools data.
    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        path = joinpath(geom_dir, "cyax.h5")
        h5open(path, "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            # 2 axions, 3 instantons: `_validate_potential` requires strictly
            # more instantons than axions, matching the real orientifold
            # geometries' (h11, N)=(2, 6)-shaped potentials in miniature.
            potential["Q"] = Int[1 0 1; 0 1 1]
            potential["L"] = Float64[1.0 1.0 1.0; -1.0 -2.0 -3.0]
            geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
        end
        geom_idx = CYAxiverse.structs.GeometryIndex(2, 1, 1)
        withenv("CYAXIVERSE_DATA_DIR" => root) do
            # 1. The established vacua_pipeline engine, called with
            # method=:auto (this test's own guard against ever silently
            # reverting to the :legacy default this driver was built to
            # avoid).
            vacua_result = compute_vacua_data(geom_idx, root; method=:auto,
                save=true, starts=64)
            @test vacua_result["search"].search_status == "completed"

            catastrophe_outcome = (; status=:success,
                result=run_geometry(geom_idx; max_branches=1_000))
            efold_outcome = (; status=:success,
                result=scan_geometry_for_inflation(geom_idx; max_branches=1_000,
                    precision_bits=128, float_tolerance=1e-10,
                    high_tolerance=1e-25, max_points=100, min_efolds=0.01,
                    max_efolds=0.05, flow_step=1e-3, flow_displacement=1e-3))

            # 2. Round-trip: write once, then read every field back.
            write_inflation_group!(geom_idx, catastrophe_outcome, efold_outcome;
                catastrophe_settings=(; max_branches=1_000),
                efold_settings=(; max_branches=1_000, precision_bits=128,
                    float_tolerance=1e-10, high_tolerance=1e-25,
                    max_points=100, min_efolds=0.01, max_efolds=0.05,
                    flow_step=1e-3, flow_displacement=1e-3))

            pv = CYAxiverse.read.pipeline_vacua(geom_idx)
            @test pv.metadata.status == "completed"
            @test pv.metadata.method == "auto"

            h5open(path, "r") do file
                @test haskey(file, "inflation")
                @test haskey(file, "vacua_pipeline")
                @test String(HDF5.read(file["inflation/scale_status"])) == "homotopy_only"
                @test String(HDF5.read(file["inflation/claim_boundary"])) ==
                    "homotopy-only diagnostics; physical domain certificate not passed"
                cat = file["inflation/catastrophes"]
                @test String(HDF5.read(cat["status"])) == "completed"
                @test HDF5.read(cat["leading_minima_count"]) ==
                    catastrophe_outcome.result.leading_minima_count
                @test HDF5.read(cat["leading_branch_saddle_count"]) ==
                    catastrophe_outcome.result.saddle_count
                @test HDF5.read(cat["leading_branch_saddles_present"]) ==
                    Int(catastrophe_outcome.result.saddle_count > 0)
                @test HDF5.read(cat["legacy_v1/saddle_count"]) ==
                    catastrophe_outcome.result.saddle_count
                @test HDF5.read(cat["legacy_v1/catastrophes_present"]) ==
                    Int(catastrophe_outcome.result.saddle_count > 0)

                ef = file["inflation/efolds"]
                @test String(HDF5.read(ef["search_status"])) ==
                    string(efold_outcome.result.refinement.search.search_status)
                @test HDF5.read(ef["n_candidates"]) ==
                    length(efold_outcome.result.refinement.candidates)
                @test HDF5.read(ef["n_qualified"]) == efold_outcome.result.flow.qualified
            end

            # 3. No-overwrite guard: a second write without force=true is
            # rejected rather than silently replacing the first result.
            @test_throws ArgumentError write_inflation_group!(geom_idx,
                catastrophe_outcome, efold_outcome;
                catastrophe_settings=(; max_branches=1_000),
                efold_settings=(; max_branches=1_000, precision_bits=128,
                    float_tolerance=1e-10, high_tolerance=1e-25,
                    max_points=100, min_efolds=0.01, max_efolds=0.05,
                    flow_step=1e-3, flow_displacement=1e-3))

            # The rejected retry must not have touched the file: the original
            # result is still exactly what is on disk.
            h5open(path, "r") do file
                cat = file["inflation/catastrophes"]
                @test HDF5.read(cat["leading_branch_saddle_count"]) ==
                    catastrophe_outcome.result.saddle_count
            end

            # 4. A failed upstream stage is recorded with explicit sentinel
            # values and its error message, not silently skipped or
            # confused with a genuine zero/completed result.
            failed_catastrophe = (; status=:failed, message="synthetic failure for testing")
            failed_efold = (; status=:failed, message="synthetic failure for testing")
            write_inflation_group!(geom_idx, failed_catastrophe, failed_efold;
                catastrophe_settings=(; max_branches=1_000),
                efold_settings=(; max_branches=1_000, precision_bits=128,
                    float_tolerance=1e-10, high_tolerance=1e-25,
                    max_points=100, min_efolds=0.01, max_efolds=0.05,
                    flow_step=1e-3, flow_displacement=1e-3), force=true)
            h5open(path, "r") do file
                cat = file["inflation/catastrophes"]
                @test String(HDF5.read(cat["status"])) == "failed"
                @test HDF5.read(cat["leading_branch_saddle_count"]) == -1
                @test HDF5.read(cat["legacy_v1/saddle_count"]) == -1
                @test String(HDF5.read(cat["error"])) == "synthetic failure for testing"
                ef = file["inflation/efolds"]
                @test String(HDF5.read(ef["search_status"])) == "failed"
                @test HDF5.read(ef["n_candidates"]) == -1
            end
            @test !_has_inflation_group(path)
            @test !_pipeline2_complete(path)
            @test _inflation_write_blocked(path)
            # Failed terminal records are explicit persisted state. Replacing
            # one requires the same explicit force acknowledgement as a
            # completed record.
            @test_throws ArgumentError write_inflation_group!(geom_idx,
                catastrophe_outcome, efold_outcome;
                catastrophe_settings=(; max_branches=1_000),
                efold_settings=(; max_branches=1_000, precision_bits=128,
                    float_tolerance=1e-10, high_tolerance=1e-25,
                    max_points=100, min_efolds=0.01, max_efolds=0.05,
                    flow_step=1e-3, flow_displacement=1e-3))
            write_inflation_group!(geom_idx, catastrophe_outcome, efold_outcome;
                catastrophe_settings=(; max_branches=1_000),
                efold_settings=(; max_branches=1_000, precision_bits=128,
                    float_tolerance=1e-10, high_tolerance=1e-25,
                    max_points=100, min_efolds=0.01, max_efolds=0.05,
                    flow_step=1e-3, flow_displacement=1e-3), force=true)
            @test _has_inflation_group(path)

            # 5. A geometry file that does not exist yet is rejected outright
            # -- this writer never creates a geometry, only appends to one.
            missing_idx = CYAxiverse.structs.GeometryIndex(2, 2, 1)
            @test_throws ArgumentError write_inflation_group!(missing_idx,
                catastrophe_outcome, efold_outcome;
                catastrophe_settings=(; max_branches=1_000),
                efold_settings=(; max_branches=1_000, precision_bits=128,
                    float_tolerance=1e-10, high_tolerance=1e-25,
                    max_points=100, min_efolds=0.01, max_efolds=0.05,
                    flow_step=1e-3, flow_displacement=1e-3))
        end
    end

    # Real-data round-trip, when the h11=2 orientifold database is present
    # on this machine (mirrors the Phase 1 skip-gracefully pattern above).
    db_root = normpath(joinpath(@__DIR__, "..", "..", "data",
        "orientifold_axiverse_database_20260821"))
    h11_dir = joinpath(db_root, "h11_002")
    if isdir(h11_dir)
        np_dirs = sort(filter(name -> startswith(name, "np_"), readdir(h11_dir)))
        if !isempty(np_dirs)
            np_index = parse(Int, np_dirs[1][4:end])
            np_dir = joinpath(h11_dir, np_dirs[1])
            cy_dirs = sort(filter(name -> startswith(name, "cy_"), readdir(np_dir)))
            if !isempty(cy_dirs)
                cy_index = parse(Int, cy_dirs[1][4:end])
                geom_idx = CYAxiverse.structs.GeometryIndex(2, np_index, cy_index)
                withenv("CYAXIVERSE_DATA_DIR" => db_root) do
                    pv = CYAxiverse.read.pipeline_vacua(geom_idx)
                    @test pv.metadata.status == "completed"
                    @test pv.metadata.method == "auto"
                    @test pv.estimate >= 0
                    h5open(CYAxiverse.filestructure.cyax_file(geom_idx), "r") do file
                        @test haskey(file, "inflation")
                        @test haskey(file, "inflation/catastrophes")
                        @test haskey(file, "inflation/efolds")
                        @test String(HDF5.read(file["inflation/catastrophes/status"])) ==
                            "completed"
                    end
                end
            end
        end
    else
        @info "skipping Pipeline 2 real-data round-trip test: " *
            "$h11_dir not present on this machine"
    end
end
