using CYAxiverse
using LinearAlgebra
using SparseArrays
using Test
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "vacua_pipeline.jl"))
include(joinpath(@__DIR__, "..", "scripts", "batch_vacua_pipeline.jl"))
include(joinpath(@__DIR__, "..", "scripts", "batch_physical_spectrum.jl"))

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
            @test from_geometry.Qtilde == from_matrices.Qtilde
            @test from_geometry.Ltilde == from_matrices.Ltilde
            @test from_geometry.Qbar == from_matrices.Qbar
            @test from_geometry.Lbar == from_matrices.Lbar
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
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
end
