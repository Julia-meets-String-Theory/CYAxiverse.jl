using CYAxiverse
using LinearAlgebra
using Random
using SparseArrays
using Test
using HDF5
using ArbNumerics: ArbFloat, precision, setprecision

@testset "Core plotting API stays optional" begin
    @test Base.get_extension(CYAxiverse, :CYAxiverseCairoMakieExt) === nothing
    @test isempty(methods(CYAxiverse.plotting.scatterplot))
    @test isempty(methods(CYAxiverse.plotting.functionplot))

    style = CYAxiverse.plotting.paper_style(accent_colors = (:gold, :teal))
    @test style.background == "#E8E8F0"
    @test style.accent_colors == ("gold", "teal")
    @test CYAxiverse.plotting.curve(1:2, [3, 4]; label = "curve").label == "curve"
    @test_throws DimensionMismatch CYAxiverse.plotting.curve(1:2, [3])
    @test_throws ArgumentError CYAxiverse.plotting.reference_line(1; orientation = :diagonal)
end

include(joinpath(@__DIR__, "optional_plotting.jl"))

include(joinpath(@__DIR__, "..", "scripts", "vacua_pipeline.jl"))
include(joinpath(@__DIR__, "..", "scripts", "batch_vacua_pipeline.jl"))
include(joinpath(@__DIR__, "..", "scripts", "batch_physical_spectrum.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_refinement_common.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_scan_prep.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_scan_pilot.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_scale_continuation.jl"))
include(joinpath(@__DIR__, "..", "scripts", "inflation_candidate_refinement.jl"))
include(joinpath(@__DIR__, "..", "scripts", "migrate_quartic_index_ordering.jl"))
include(joinpath(@__DIR__, "..", "scripts", "build_orientifold_vacua_inflation.jl"))
include(joinpath(@__DIR__, "axion_photon.jl"))

@testset "Data directory resolution" begin
    filestructure = CYAxiverse.filestructure
    expected_default = normpath(joinpath(@__DIR__, "..", "..", "data"))
    old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
    old_newargs = get(ENV, "newARGS", nothing)
    try
        delete!(ENV, "CYAXIVERSE_DATA_DIR")
        delete!(ENV, "newARGS")
        @test filestructure.default_data_dir() == expected_default
        @test filestructure.resolve_data_dir() == expected_default

        ENV["CYAXIVERSE_DATA_DIR"] = "/tmp/from-environment"
        @test filestructure.resolve_data_dir() == "/tmp/from-environment"
        @test filestructure.resolve_data_dir("/tmp/explicit") == "/tmp/explicit"
        @test filestructure.present_dir() == "/tmp/from-environment/"
        @test filestructure.present_dir("/tmp/explicit") == "/tmp/explicit/"

        delete!(ENV, "CYAXIVERSE_DATA_DIR")
        ENV["newARGS"] = "unknown-alias"
        @test_throws ArgumentError filestructure.resolve_data_dir()
        ENV["newARGS"] = "docker"
        @test filestructure.resolve_data_dir() == "/scratch/database"
    finally
        old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
            (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        old_newargs === nothing ? delete!(ENV, "newARGS") :
            (ENV["newARGS"] = old_newargs)
    end
end

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
    physical_records = _pilot_records([[0.5]], [0], Q,
        pilot_homotopy_scale(L, 1.01), factor;
        residual_tolerance=1e-10, max_iterations=20, duplicate_tolerance=1e-7,
        scale_status=:physical)
    @test all(record -> record.candidate_status == :none, physical_records)
    @test any(occursin("withheld", record.candidate_reason) for record in physical_records)
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

@testset "Generic inflation point correction and precision boundary" begin
    Q = Int[1 0; 0 1]
    L = Float64[1.0 -0.1; 0.0 0.0]
    K = Float64[2.0 0.3; 0.3 1.0]
    points = CYAxiverse.inflation_points

    context = points.prepare_context(Q, L, K)
    @test points.basis_policy().working_basis == :periodic_string
    @test points.basis_policy().physical_basis == :mass_eigenbasis
    corrected = points.correct_stationary_point(context, [1.5, -0.5];
        residual_tolerance=1e-12)
    @test corrected.status == :converged
    @test corrected.working_basis == :periodic_string
    @test corrected.theta ≈ [0.5, 0.5] atol=1e-12
    @test corrected.residual <= 1e-12

    diagnostic = points.diagnose(context, corrected.theta)
    @test diagnostic.gradient_residual <= 1e-12
    @test diagnostic.negative_modes == 1
    @test diagnostic.zeroish_modes == 0
    @test diagnostic.positive_modes == 1
    @test diagnostic.value > 0
    @test diagnostic.physical_basis == :mass_eigenbasis

    scalar_mass_basis = points.mass_eigenbasis(context, corrected.theta)
    @test !hasproperty(scalar_mass_basis, :raw_eigenvectors)
    vector_mass_basis = points.mass_eigenbasis(context, corrected.theta; vectors=true)
    @test vector_mass_basis.basis == :mass_eigenbasis
    @test vector_mass_basis.eigenvalues ≈ diagnostic.hessian_eigenvalues
    @test vector_mass_basis.metric_residual < 1e-12
    @test vector_mass_basis.generalized_residual < 1e-12

    comparison = points.compare_precision([1.5, -0.5], Q, L, K;
        precision_bits=128, float_residual_tolerance=1e-12,
        high_residual_tolerance=1e-25)
    @test comparison.float_correction.status == :converged
    @test comparison.high_correction.status == :converged
    @test comparison.residual_agreement
    @test comparison.inertia_agreement
    @test comparison.accepted
    @test comparison.float_diagnostics.negative_modes ==
        comparison.high_diagnostics.negative_modes

    @test_throws PosDefException points.prepare_context(Q, L, [-1.0 0.0; 0.0 1.0])
    @test_throws DimensionMismatch points.prepare_context(Q, L[:, 1:1], K)
end

@testset "Generic geometry mass-basis gradient flow" begin
    points = CYAxiverse.inflation_points
    Q = reshape(Int[1, 2], 1, 2)
    L = Float64[1.0 1.0; 0.0 log10(0.25)]
    K = Matrix{Float64}(I, 1, 1)
    context = points.prepare_context(Q, L, K)
    flow = points.gradient_flow(context, [0.5]; displacement=1e-3,
        max_efolds=0.1, step=1e-3)
    @test flow.status == :completed
    @test flow.basis == :mass_eigenbasis
    @test flow.coordinate_chart == :canonical_cholesky
    @test flow.efolds > 0
    @test length(flow.mass_direction) == 1
    @test flow.mass_eigenvalue ≈ 0 atol=1e-10

    mktempdir() do root
        geom_dir = joinpath(root, "h11_001", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        scan_L = copy(L)
        scan_L[2, 2] = log10(0.24)
        h5open(joinpath(geom_dir, "cyax.h5"), "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            potential["Q"] = Q
            potential["L"] = scan_L
            geometric["Kinv"] = K
        end
        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            geom_idx = CYAxiverse.structs.GeometryIndex(1, 1, 1)
            prepared = points.prepare_geometry_context(geom_idx)
            @test prepared.geometry == geom_idx
            @test prepared.input_basis == :periodic_string
            @test prepared.source == :oriented_potential
            geometry_flow = points.gradient_flow(geom_idx, [0.5];
                displacement=1e-3, max_efolds=0.1, step=1e-3)
            @test geometry_flow.geometry == geom_idx
            @test geometry_flow.input_basis == :periodic_string
            @test geometry_flow.source == :oriented_potential
            @test geometry_flow.status == :completed
            @test geometry_flow.basis == :mass_eigenbasis

            scan = scan_geometry_for_inflation(geom_idx;
                max_branches=100, precision_bits=128,
                float_tolerance=1e-10, high_tolerance=1e-25,
                max_points=100, min_efolds=0.05, max_efolds=0.1,
                flow_step=1e-3, flow_displacement=1e-3)
            @test scan.refinement.search.search_status == :completed
            @test !isempty(scan.refinement.candidates)
            @test length(scan.flow.rows) == 2 * length(scan.refinement.candidates)
            @test Set(row.displacement_sign for row in scan.flow.rows) == Set((-1, 1))
            @test all(row.physical_basis == :mass_eigenbasis for row in scan.flow.rows)
            @test any(row.flow_accepted for row in scan.flow.rows)
            flow_report = joinpath(root, "flows.csv")
            _write_csv(flow_report, scan.flow.rows, FLOW_FIELDS)
            @test length(readlines(flow_report)) == length(scan.flow.rows) + 1
        finally
            if old_data_dir === nothing
                delete!(ENV, "CYAXIVERSE_DATA_DIR")
            else
                ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir
            end
        end
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

            duplicate_dir = joinpath(root, "h11_002", "np_0000003", "cy_0000001")
            mkpath(duplicate_dir)
            h5open(joinpath(duplicate_dir, "cyax.h5"), "cw") do file
                cytools = create_group(file, "cytools")
                potential = create_group(cytools, "potential")
                geometric = create_group(cytools, "geometric")
                # Three leading charges, with the second and third redundant;
                # the remaining columns are their ordered pairwise terms.
                potential["Q"] = Int[
                    1 0 0 -1 -1 0;
                    0 1 1 1 1 0;
                ]
                potential["L"] = Float64[
                    1.0 2.0 2.0 4.0 4.0 6.0;
                    0.0 -1.0 -1.0 -3.0 -3.0 -5.0;
                ]
                geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
            end
            duplicate = CYAxiverse.read.oriented_potential(
                CYAxiverse.structs.GeometryIndex(2, 3, 1))
            @test duplicate.Q == Int[1 0 -1; 0 1 1]
            @test duplicate.L == Float64[1.0 2.0 4.0; 0.0 -1.0 -3.0]
            raw_duplicate = CYAxiverse.read.oriented_potential(
                CYAxiverse.structs.GeometryIndex(2, 3, 1);
                canonicalize_charge_rows=false)
            @test size(raw_duplicate.Q) == (2, 6)
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
    end
end

@testset "Kinetic matrix construction and section-3 positive-definiteness invariant" begin
    # `.copilot/AGENTS.md` section 3 requires the kinetic matrix
    # K = 1/2 g to be symmetric positive-definite, throwing a domain error
    # when it is not. `read.potential` previously built K as
    # `Hermitian(inv(Kinv))`: a plain, symmetry-unaware `inv` on the
    # `Matrix` read from HDF5, wrapped in `Hermitian` only afterwards. That
    # selects a general LU factorisation and can leave K badly indefinite on
    # realistic, ill-conditioned `Kinv`, even though the wrapped result is
    # always exactly symmetric. This testset pins the fixed construction
    # (`Hermitian(inv(Hermitian(Kinv)))`, a symmetry-aware Bunch-Kaufman
    # inverse) together with the new validation that enforces the
    # invariant.
    make_kinv(n, logcond; rng) = begin
        V = Matrix(qr(randn(rng, n, n)).Q)
        D = Diagonal(10.0 .^ range(0, -logcond, length=n))
        V * D * V'
    end
    write_potential_fixture(root, h11, np, cy, Kinv) = begin
        geom_dir = joinpath(root, "h11_$(lpad(h11, 3, '0'))",
            "np_$(lpad(np, 7, '0'))", "cy_$(lpad(cy, 7, '0'))")
        mkpath(geom_dir)
        n = size(Kinv, 1)
        h5open(joinpath(geom_dir, "cyax.h5"), "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            potential["Q"] = Matrix{Int}(I, n, n)
            potential["L"] = vcat(ones(1, n), -ones(1, n))
            geometric["Kinv"] = Kinv
        end
    end

    mktempdir() do root
        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            rng = MersenneTwister(20260816)

            # A well-conditioned Kinv reads fine and yields a
            # positive-definite K.
            write_potential_fixture(root, 21, 1, 1, make_kinv(6, 2; rng=rng))
            well_conditioned = CYAxiverse.read.potential(21, 1, 1)
            @test isposdef(well_conditioned.K)

            # Ill-conditioned Kinv at cond = 1e10, 1e12, 1e14: the fixed
            # construction reads fine and yields a positive-definite K. The
            # previous `Hermitian(inv(Kinv))` construction failed
            # `cholesky` on a large majority of draws at these condition
            # numbers (up to 20/20 at n=100); the fix eliminates that
            # failure mode entirely for this range.
            for (index, logcond) in enumerate((10, 12, 14))
                # n=100: the measurement backing this fix (240 draws
                # across n in 20, 40, 100 and cond in 1e10..1e16) found the
                # old construction fails cholesky on 20/20 draws at n=100
                # for every condition number tested, so the "bug
                # reproduces" assertion below is not a statistical fluke.
                Kinv = make_kinv(100, logcond; rng=rng)
                write_potential_fixture(root, 22, index, 1, Kinv)
                pot = CYAxiverse.read.potential(22, index, 1)
                @test isposdef(pot.K)

                # The bug reproduces on the same stored data: the old,
                # symmetry-unaware construction is badly indefinite here.
                @test !isposdef(Hermitian(inv(Kinv)))
            end

            # A genuinely non-positive-definite stored Kinv (a negative
            # eigenvalue of real size, far larger than floating-point
            # noise) throws DomainError, and the message names the
            # geometry.
            n = 6
            V = Matrix(qr(randn(rng, n, n)).Q)
            D_bad = Diagonal(vcat([1.0, 0.5, 0.3, 0.1, 0.05], [-0.2]))
            Kinv_bad = V * D_bad * V'
            write_potential_fixture(root, 23, 1, 1, Kinv_bad)
            err = try
                CYAxiverse.read.potential(23, 1, 1)
                nothing
            catch caught
                caught
            end
            @test err isa DomainError
            @test occursin("h11=23", err.msg)
            @test occursin("polytope=1", err.msg)
            @test occursin("frst=1", err.msg)

            # Opt-out: validate=false returns the same (invalid) K without
            # throwing, for callers that have already validated the corpus
            # or need to bypass the check.
            skipped = CYAxiverse.read.potential(23, 1, 1; validate=false)
            @test !isposdef(skipped.K)

            # Chosen behaviour: validate isposdef(K) with no tolerance,
            # because section 3 is a statement about K and this is exactly
            # that statement. A Kinv eigenvalue that is negative even at
            # -1e-10 produces a K with smallest eigenvalue around -1e10, so
            # it is rejected rather than tolerated.
            #
            # An earlier revision fell back to Kinv's own spectrum with a
            # sqrt(eps)-scaled tolerance. That was removed after measurement:
            # it altered the outcome only beyond cond(Kinv) = 1e16, where
            # rejection is correct anyway, while wrongly accepting a stored
            # eigenvalue of -1e-9 that yields a K with smallest eigenvalue
            # -1e9 -- precisely the state section 3 forbids.
            Random.seed!(2024)
            n = 6
            V_marginal = Matrix(qr(randn(n, n)).Q)
            D_marginal = Diagonal(vcat([1.0, 0.6, 0.4, 0.2, 0.1], [-1e-10]))
            Kinv_marginal = V_marginal * D_marginal * V_marginal'
            @test !isposdef(Hermitian(inv(Hermitian(Kinv_marginal))))
            write_potential_fixture(root, 24, 1, 1, Kinv_marginal)
            @test_throws DomainError CYAxiverse.read.potential(24, 1, 1)
            # ... and validate=false still bypasses the check entirely.
            @test CYAxiverse.read.potential(24, 1, 1; validate = false).K isa
                  Hermitian{Float64, Matrix{Float64}}

            # A positive-definite Kinv approaching a Kaehler-cone wall must
            # still be accepted: its smallest eigenvalue tends to zero from
            # above, which makes K's LARGEST eigenvalue diverge while K
            # stays positive definite.
            V_wall = Matrix(qr(randn(30, 30)).Q)
            Kinv_wall = V_wall * Diagonal(10.0 .^ range(0, -16; length = 30)) * V_wall'
            write_potential_fixture(root, 25, 1, 1, Kinv_wall)
            near_wall = CYAxiverse.read.potential(25, 1, 1)
            @test isposdef(near_wall.K)
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
    HDF5.h5open(output_path, "r") do file
        @test HDF5.haskey(file, "spectrum/physical/m")
        @test read(file["spectrum/physical/fK_log10"]) == [5.0, 6.0]
        @test read(file["spectrum/physical/fpert_log10"]) ≈ [-0.5, 0.0]
        @test read(file["spectrum/physical/lambda_31_sign"]) == [1]
        @test read(file["spectrum/physical/lambda_31_log10"]) == [0.5]
        @test read(file["spectrum/physical/lambda_31_indices"]) == reshape([1, 1, 1, 2], 4, 1)
        @test read(file["spectrum/physical/lambda_22_sign"]) == [-1]
        @test read(file["spectrum/physical/lambda_22_log10"]) == [0.7]
        @test read(file["spectrum/physical/lambda_22_indices"]) == reshape([1, 1, 2, 2], 4, 1)
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
    # catastrophe-paper reduction retains rational coordinates and enlarges
    # their fundamental domain before solving.
    catastrophe_fixture = (Q=Int[2 0 1 1; 0 2 1 0; 0 0 1 1],
        L=Float64[1.0 1.0 1.0 1.0; 0.0 -1.0 -2.0 -3.0])
    alpha_problem = CYAxiverse.jlm_reduced.prepare(catastrophe_fixture.Q, catastrophe_fixture.L)
    catastrophe_problem = CYAxiverse.jlm_reduced.prepare(
        catastrophe_fixture.Q, catastrophe_fixture.L; reduction=:catastrophe)
    @test alpha_problem.reduction == :alphamatrix
    @test catastrophe_problem.reduction == :catastrophe
    @test catastrophe_problem.coordinate_scale == [2, 1]
    @test Matrix(catastrophe_problem.Q_reduced) == [2 0; 0 1; -1 1]
    catastrophe_ensemble = CYAxiverse.jlm_reduced.critical_ensemble(
        catastrophe_problem; starts=256)
    @test catastrophe_ensemble.critical_count == 8
    @test catastrophe_ensemble.minima_count == 2

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

    @testset "Fuzzy-axion mass-scale formulas (arXiv:2412.12012)" begin
        mplanck_ev = Float64(CYAxiverse.generate.constants()["MPlanck"]) * 1e9

        @test CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(1.0) == 0.0
        @test isapprox(CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(exp(1)), -2.0)
        @test isapprox(CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(exp(2)), -4.0)
        @test_throws ArgumentError CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(0.0)
        @test_throws ArgumentError CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(-1.0)

        @test isapprox(CYAxiverse.paper_benchmarks.fuzzy_axion_prefactor_P(1.0), 1.0 / 128.0)
        # arXiv:2412.12012: "For the reasonable reference value gs = 0.5, we
        # have P ~ 5 * 10^-4, and this is the value we have used in our main
        # analysis." Quoted to one significant figure.
        p_reference = CYAxiverse.paper_benchmarks.fuzzy_axion_prefactor_P(0.5)
        @test isapprox(p_reference, 0.5^4 / 128.0)
        @test isapprox(round(p_reference; digits=4), 5e-4)
        @test_throws ArgumentError CYAxiverse.paper_benchmarks.fuzzy_axion_prefactor_P(0.0)
        @test_throws ArgumentError CYAxiverse.paper_benchmarks.fuzzy_axion_prefactor_P(-0.5)

        w_default = CYAxiverse.paper_benchmarks.fuzzy_axion_flux_superpotential(1e-5)
        @test isapprox(real(w_default), 1e-5)
        @test imag(w_default) == 0.0
        w_real_terms = CYAxiverse.paper_benchmarks.fuzzy_axion_flux_superpotential(
            1e-5, [0.1, -0.05])
        @test isapprox(real(w_real_terms), 1e-5 + 0.1 - 0.05)
        w_complex_terms = CYAxiverse.paper_benchmarks.fuzzy_axion_flux_superpotential(
            0.0, [0.1 + 0.2im])
        @test isapprox(real(w_complex_terms), 0.1)
        @test isapprox(imag(w_complex_terms), 0.2)

        # P=1, K=0, |W|=1 -> m_3/2 = Mpl exactly.
        @test isapprox(
            CYAxiverse.paper_benchmarks.fuzzy_axion_gravitino_mass(1.0, 0.0, 1.0),
            mplanck_ev)
        # P=1, K=-2 (fuzzy_axion_kahler_potential(e)), |W|=1 -> m_3/2 = Mpl / e.
        @test isapprox(
            CYAxiverse.paper_benchmarks.fuzzy_axion_gravitino_mass(
                1.0, CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(exp(1)), 1.0),
            mplanck_ev / exp(1))
        # |W| = |3+4i| = 5.
        @test isapprox(
            CYAxiverse.paper_benchmarks.fuzzy_axion_gravitino_mass(1.0, 0.0, 3.0 + 4.0im),
            5.0 * mplanck_ev)
        @test_throws ArgumentError CYAxiverse.paper_benchmarks.fuzzy_axion_gravitino_mass(
            -1.0, 0.0, 1.0)
    end

    @testset "Fuzzy-axion model-stage evaluator (arXiv:2412.12012 Algorithm 1)" begin
        PB = CYAxiverse.paper_benchmarks
        G = CYAxiverse.generate

        Q = Int[
            1 0 -1 2 6 0 0 4
            0 1  1 0 0 0 0 -2
            0 0  0 1 3 1 0 1
            0 0 -1 1 3 0 1 2
        ]
        tau = Float64[0.5, 8.0, 0.5, 11.0, 33.0, 3.0, 7.0, 3.0000001]
        Kinv = [
            1.0 0.1 0.05 0.0
            0.1 1.0 0.1 0.05
            0.05 0.1 1.0 0.1
            0.0 0.05 0.1 1.0
        ]

        # `leading_axion_reference_data`'s Cholesky/canonical-frame/unit-conversion
        # plumbing must exactly reproduce the established public
        # `pq_spectrum(K, L, Q; mixing_correction=false)` API when the eq. 3.18
        # prefactor injection is neutralized (8*pi*sqrt(P)*m32/V == 1). This
        # isolates "did I wire the existing, already-used mass machinery
        # correctly" from "is eq. 3.19's own normalization convention exactly
        # right", the latter being inherited, untouched, from the existing
        # pq_canonical_frame/pq_spectrum implementation.
        P_neutral = 1.0
        V_neutral = 1.0
        m32_neutral = 1.0 / (8π)
        reference_neutral = PB.leading_axion_reference_data(
            Q, tau, V_neutral, P_neutral, m32_neutral, Kinv)
        Kmetric = Hermitian(inv(Hermitian(Kinv)))
        L_raw = PB.instanton_scales(tau, 1.0)
        established = G.pq_spectrum(Kmetric, L_raw, Q; mixing_correction=false)
        @test isapprox(
            sort(reference_neutral.mass_log10_ev_reference), established.m; atol=1e-8)

        # The eq. 3.18 prefactor log10(8*pi*sqrt(P)*m32/V) is a uniform
        # additive shift to log10(Lambda^4) that changes by 0.5*Δlog10(P)
        # when only P changes; since m ~ sqrt(Lambda^4), every reference mass
        # must then shift by half of *that*, i.e. 0.25*Δlog10(P) overall.
        P_other = 4.0
        reference_other = PB.leading_axion_reference_data(
            Q, tau, V_neutral, P_other, m32_neutral, Kinv)
        expected_shift = 0.25 * (log10(P_other) - log10(P_neutral))
        @test all(isapprox.(
            reference_other.mass_log10_ev_reference .-
                reference_neutral.mass_log10_ev_reference,
            expected_shift; atol=1e-8))

        # Every selected leading charge column must be an exact original
        # column of Q (LQtilde selects raw columns, never transforms them),
        # and each reference tau must be minable back from that column's
        # position in the input tau vector.
        for a in axes(reference_neutral.Qtilde, 2)
            column = @view reference_neutral.Qtilde[:, a]
            match = findfirst(j -> @view(Q[:, j]) == column, axes(Q, 2))
            @test match !== nothing
            @test reference_neutral.tau_reference[a] == tau[match]
        end

        # Closed-form lambda root: exact inversion of eq. 3.24-3.27's
        # m(lambda)^2 = m(1)^2 * exp(-2*pi*tau(1)*(lambda^2-1)).
        m_ref = 1e10
        tau_ref = 5.0
        lambda = PB.fuzzy_axion_dilation_root(m_ref, tau_ref)
        @test lambda !== nothing
        m_at_lambda = sqrt(m_ref^2 * exp(-2π * tau_ref * (lambda^2 - 1)))
        @test isapprox(m_at_lambda, PB.FUZZY_AXION_MASS_TARGET_EV; rtol=1e-10)
        # Algorithm 1 quantifies over lambda in R+ with no lower bound of 1
        # (Sec. 4.1's literal "for lambda in {lambda in R+ | ...}"), so a
        # reference mass already at or below the target can still have a
        # valid *contracting* root (lambda < 1) -- verified directly against
        # the same closed-form relation used above.
        lambda_contract = PB.fuzzy_axion_dilation_root(1e-19, tau_ref)
        @test lambda_contract !== nothing
        @test lambda_contract < 1.0
        m_at_lambda_contract = sqrt(1e-19^2 * exp(-2π * tau_ref * (lambda_contract^2 - 1)))
        @test isapprox(m_at_lambda_contract, PB.FUZZY_AXION_MASS_TARGET_EV; rtol=1e-10)
        # Genuinely no real root: the reference mass is far enough below the
        # target that even lambda -> 0 cannot raise it back up.
        @test PB.fuzzy_axion_dilation_root(1e-30, tau_ref) === nothing
        @test_throws ArgumentError PB.fuzzy_axion_dilation_root(-1.0, tau_ref)
        @test_throws ArgumentError PB.fuzzy_axion_dilation_root(m_ref, -1.0)
        @test_throws ArgumentError PB.fuzzy_axion_dilation_root(
            m_ref, tau_ref; mass_target_ev=0.0)

        # Criterion 1: tau -> lambda^2*tau uniformly; boundary is inclusive.
        @test PB.fuzzy_axion_criterion_one([1.0, 2.0, 0.5], 1.0) == false
        @test PB.fuzzy_axion_criterion_one([1.0, 2.0, 1.0], 1.0) == true
        @test PB.fuzzy_axion_criterion_one([0.25, 2.0], 2.0) == true
        @test_throws ArgumentError PB.fuzzy_axion_criterion_one([1.0], -1.0)

        # Criterion 2: inclusive [25, 40] window on the QCD divisor's volume.
        @test PB.fuzzy_axion_criterion_two(30.0, 1.0) == true
        @test PB.fuzzy_axion_criterion_two(24.999, 1.0) == false
        @test PB.fuzzy_axion_criterion_two(40.001, 1.0) == false
        @test PB.fuzzy_axion_criterion_two(25.0, 1.0) == true
        @test PB.fuzzy_axion_criterion_two(40.0, 1.0) == true
        @test_throws ArgumentError PB.fuzzy_axion_criterion_two(30.0, 0.0)

        # End-to-end: every returned (D, a) model must independently satisfy
        # all three criteria at its own lambda -- re-derived here from the
        # function's own inputs/outputs, not merely trusted from its return
        # value, to catch a criterion evaluated against the wrong lambda or
        # the wrong divisor index.
        P_real = CYAxiverse.paper_benchmarks.fuzzy_axion_prefactor_P(0.5)
        V_real = 9.0
        K_real = CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(V_real)
        W_real = CYAxiverse.paper_benchmarks.fuzzy_axion_flux_superpotential(1e-5)
        m32_real = CYAxiverse.paper_benchmarks.fuzzy_axion_gravitino_mass(
            P_real, K_real, abs(W_real); mplanck_ev=1.0)
        models = PB.enumerate_fuzzy_axion_models(Q, tau, V_real, P_real, m32_real, Kinv)
        @test !isempty(models)
        for model in models
            @test PB.fuzzy_axion_criterion_one(tau, model.lambda)
            @test PB.fuzzy_axion_criterion_two(
                tau[model.qcd_divisor_index], model.lambda)
            # Re-derived in log10 space, matching how enumerate_fuzzy_axion_models
            # itself solves for lambda -- the linear form can underflow to
            # exactly 0.0 for a strongly-suppressed sub-leading instanton
            # (see fuzzy_axion_dilation_root's docstring).
            log10_m_check = model.mass_reference_log10_ev -
                (π * model.tau_reference * (model.lambda^2 - 1)) / log(10)
            @test isapprox(log10_m_check, log10(PB.FUZZY_AXION_MASS_TARGET_EV); atol=1e-8)
        end
        # Every model's qcd_divisor_index and axion_index must be valid,
        # in-range references into the input arrays.
        @test all(1 <= m.qcd_divisor_index <= length(tau) for m in models)
        @test all(1 <= m.axion_index <= size(reference_neutral.Qtilde, 2) for m in models)

        @test_throws ArgumentError PB.leading_axion_reference_data(
            Q, tau, -1.0, P_real, m32_real, Kinv)
        @test_throws ArgumentError PB.leading_axion_reference_data(
            Q, tau, V_real, -1.0, m32_real, Kinv)
        @test_throws ArgumentError PB.leading_axion_reference_data(
            Q, tau, V_real, P_real, -1.0, Kinv)
        @test_throws DimensionMismatch PB.leading_axion_reference_data(
            Q, tau, V_real, P_real, m32_real, Kinv[1:3, 1:3])
        @test_throws DimensionMismatch PB.leading_axion_reference_data(
            Q, tau[1:end-1], V_real, P_real, m32_real, Kinv)

        # Regression for a real full-h11=4-population failure (priority 4,
        # record index 46 of the 285-record h21_plus_zero-accepted export):
        # a sub-leading (4th-ranked) instanton with tau_reference=306.0 gave
        # mass_log10_ev_reference=-390.15, and 10.0^(-390.15) underflows to
        # exactly 0.0 in Float64 (Float64's smallest positive value is
        # ~5e-324) -- enumerate_fuzzy_axion_models used to compute
        # `10.0^log10_mass` before root-solving and crashed with
        # "mass_reference_ev must be positive" on this exact input. Confirmed
        # live against the real export record, not fabricated.
        @test 10.0^(-390.1476386436961) == 0.0
        lambda_underflow_avoided = PB.fuzzy_axion_dilation_root_log10(
            -390.1476386436961, 306.0000364594718)
        @test lambda_underflow_avoided !== nothing
        @test isfinite(lambda_underflow_avoided)
        # Hand-derived expectation: argument = 1 + ln(10)*(log10_m - log10(target))/(pi*tau)
        expected_argument = 1.0 + log(10.0) *
            (-390.1476386436961 - log10(PB.FUZZY_AXION_MASS_TARGET_EV)) /
            (π * 306.0000364594718)
        @test isapprox(lambda_underflow_avoided, sqrt(expected_argument); rtol=1e-10)

        Q_record46 = Int[
            1 0 0 1 0 0 0 1
            0 1 1 -2 0 0 0 -6
            0 0 1 -3 1 1 0 -8
            0 0 0 -1 0 0 1 -3
        ]
        tau_record46 = Float64[
            3039.5003719815713, 97.5000132125037, 403.5000496719755,
            1925.0002361800668, 306.0000364594718, 306.0000364594718,
            1.499999998081662, 2.0000010365299246,
        ]
        cy_volume_record46 = 5741.334389020104
        Kinv_record46 = [
            1.2358372022962462e7 312722.4226720411 1.3089883116565086e6 18237.00220856627
            312722.4226720411 428435.77269930055 -271070.73200224846 -22380.337504440147
            1.3089883116565086e6 -271070.73200224846 374544.0892527923 -21129.337304245895
            18237.00220856627 -22380.337504440147 -21129.337304245895 114835.68772785066
        ]
        P_record46 = PB.fuzzy_axion_prefactor_P(0.5)
        K_record46 = PB.fuzzy_axion_kahler_potential(cy_volume_record46)
        W_record46 = PB.fuzzy_axion_flux_superpotential(1.0)
        m32_record46 = PB.fuzzy_axion_gravitino_mass(
            P_record46, K_record46, abs(W_record46); mplanck_ev=1.0)
        models_record46 = PB.enumerate_fuzzy_axion_models(
            Q_record46, tau_record46, cy_volume_record46, P_record46, m32_record46,
            Kinv_record46)
        for model in models_record46
            @test PB.fuzzy_axion_criterion_one(tau_record46, model.lambda)
            @test PB.fuzzy_axion_criterion_two(
                tau_record46[model.qcd_divisor_index], model.lambda)
            @test isfinite(model.mass_reference_log10_ev)
        end

        # Self-reference is NOT excluded: a candidate QCD divisor equal to
        # axion `a`'s own divisor is a valid Algorithm-1 model, matching the
        # paper's literal text (see enumerate_fuzzy_axion_models's
        # docstring). Real geometry, not synthetic: the paper's own h1,1=2
        # worked example (Sec. 4.2.1, eq. 4.2-4.4), evaluated at the genuine
        # Algorithm-1 canonical Kahler-cone-tip point (CYTools
        # `kahler_cone.tip_of_stretched_cone(1.0)`, NOT the paper's own
        # hand-picked illustrative t*) -- confirmed against the paper's text
        # that toric divisor D6 (tau=0.5 here) is "the QCD axion" and D2
        # (tau=2.5) is "the fuzzy axion". Of this geometry's 3 generated
        # models, 2 are self-referential (axion 1 paired with its own
        # divisor D6, axion 2 paired with its own divisor D2) and 1 is not
        # (axion 2 paired with the distinct-but-same-volume divisor D3),
        # per
        # validation/fuzzy_axions_2412_12012_model_count_gap_scope_20260818.md
        # Sec. 0/0b's population-wide measurement that self-referential
        # pairing is the modal outcome (69% mean, present in every one of
        # 267 records), not a rare case worth excluding.
        Q_h11_2 = [7 1 1 2 3 0; 2 0 0 0 1 1]
        tau_h11_2 = [18.499999999999993, 2.499999999999999, 2.499999999999999,
            4.999999999999998, 7.9999999999999964, 0.49999999999999956]
        cy_volume_h11_2 = 3.4999999999999982
        Kinv_h11_2 = [11.0 -9.0; -9.0 43.0]
        P_h11_2 = PB.fuzzy_axion_prefactor_P(0.5029733)
        K_h11_2 = PB.fuzzy_axion_kahler_potential(cy_volume_h11_2)
        W_h11_2 = PB.fuzzy_axion_flux_superpotential(1.0)
        m32_h11_2 = PB.fuzzy_axion_gravitino_mass(
            P_h11_2, K_h11_2, abs(W_h11_2); mplanck_ev=1.0)
        reference_h11_2 = PB.leading_axion_reference_data(
            Q_h11_2, tau_h11_2, cy_volume_h11_2, P_h11_2, m32_h11_2, Kinv_h11_2)
        # divisor_index must round-trip exactly back through Q/tau.
        for a in eachindex(reference_h11_2.divisor_index)
            @test @view(Q_h11_2[:, reference_h11_2.divisor_index[a]]) ==
                @view(reference_h11_2.Qtilde[:, a])
            @test tau_h11_2[reference_h11_2.divisor_index[a]] ==
                reference_h11_2.tau_reference[a]
        end
        models_h11_2 = PB.enumerate_fuzzy_axion_models(
            Q_h11_2, tau_h11_2, cy_volume_h11_2, P_h11_2, m32_h11_2, Kinv_h11_2)
        @test length(models_h11_2) == 3
        self_referential = [
            model.qcd_divisor_index == reference_h11_2.divisor_index[model.axion_index]
            for model in models_h11_2
        ]
        @test count(self_referential) == 2
        @test any(
            model.axion_index == 1 && model.qcd_divisor_index == 6
            for model in models_h11_2
        )
        @test any(
            model.axion_index == 2 && model.qcd_divisor_index == 2
            for model in models_h11_2
        )
        # A distinct divisor that merely happens to share the same volume
        # (D2 and D3 both have tau=2.5 here) is a different physical 4-cycle
        # and is not self-referential.
        @test any(
            model.axion_index == 2 && model.qcd_divisor_index == 3
            for model in models_h11_2
        )

        # `qcd_divisor_domain=:leading_nonself` -- the opt-in candidate
        # restriction of Algorithm 1's `for D` loop to the h1,1 leading-
        # instanton divisors minus the fuzzy axion's own
        # (validation/fuzzy_axions_2412_12012_sampler_reverse_engineering_20260818.md
        # Sec. 3.3). It must not perturb the default path, must reject an
        # unknown domain, and must produce exactly the subset the rule
        # names -- checked here against a real h1,1=3 export record so the
        # restricted set is non-empty, not vacuously so.
        @test PB.FUZZY_AXION_QCD_DIVISOR_DOMAINS == (:all_prime, :leading_nonself)
        @test PB.enumerate_fuzzy_axion_models(Q_h11_2, tau_h11_2, cy_volume_h11_2,
            P_h11_2, m32_h11_2, Kinv_h11_2; qcd_divisor_domain=:all_prime) == models_h11_2
        @test_throws ArgumentError PB.enumerate_fuzzy_axion_models(
            Q_h11_2, tau_h11_2, cy_volume_h11_2, P_h11_2, m32_h11_2, Kinv_h11_2;
            qcd_divisor_domain=:leading)
        # Both of this geometry's leading divisors (D6, tau=0.5; D2, tau=2.5)
        # fail criterion 2 at the *other* axion's lambda, so the restriction
        # empties it: all 3 default models here are self-paired or pair with
        # D3, which hosts no leading instanton.
        @test isempty(PB.enumerate_fuzzy_axion_models(Q_h11_2, tau_h11_2, cy_volume_h11_2,
            P_h11_2, m32_h11_2, Kinv_h11_2; qcd_divisor_domain=:leading_nonself))

        # Real h1,1=3 export record (record 0 of the full h1,1=3 population,
        # validation/fuzzy_axions_supp/model_count_gap_20260818/h11_3_detail.json).
        # Leading divisors are D1, D2, D7; the default domain accepts 10
        # models, the restriction keeps exactly (axion 1, D2) and
        # (axion 2, D1) -- axion 3's lambda puts the two remaining leading
        # divisors at tau ~ 22, below criterion 2's floor of 25.
        Q_h11_3 = [1 0 0 4 1 0 2; 0 1 1 0 0 0 -2; 0 0 0 2 0 1 1]
        tau_h11_3 = [2.000000000261836, 2.000000000397189, 2.000000000397189,
            14.000000002208573, 2.000000000261836, 3.0000000005806142,
            3.000000000309908]
        cy_volume_h11_3 = 3.6666666675483235
        Kinv_h11_3 = [16.000000004189374 1.3333333337988176 -20.0000000053531
            1.3333333337988176 16.000000006355023 9.333333337232721
            -20.0000000053531 9.333333337232721 94.66666669585791]
        P_h11_3 = PB.fuzzy_axion_prefactor_P(0.5)
        K_h11_3 = PB.fuzzy_axion_kahler_potential(cy_volume_h11_3)
        W_h11_3 = PB.fuzzy_axion_flux_superpotential(1.0)
        m32_h11_3 = PB.fuzzy_axion_gravitino_mass(
            P_h11_3, K_h11_3, abs(W_h11_3); mplanck_ev=1.0)
        reference_h11_3 = PB.leading_axion_reference_data(
            Q_h11_3, tau_h11_3, cy_volume_h11_3, P_h11_3, m32_h11_3, Kinv_h11_3)
        @test reference_h11_3.divisor_index == [1, 2, 7]
        models_h11_3 = PB.enumerate_fuzzy_axion_models(Q_h11_3, tau_h11_3,
            cy_volume_h11_3, P_h11_3, m32_h11_3, Kinv_h11_3)
        @test length(models_h11_3) == 10
        restricted_h11_3 = PB.enumerate_fuzzy_axion_models(Q_h11_3, tau_h11_3,
            cy_volume_h11_3, P_h11_3, m32_h11_3, Kinv_h11_3;
            qcd_divisor_domain=:leading_nonself)
        @test [(m.axion_index, m.qcd_divisor_index) for m in restricted_h11_3] ==
            [(1, 2), (2, 1)]
        # The restriction only ever removes candidates: same lambda, same
        # reference mass, same tau_reference for every surviving pair.
        for restricted in restricted_h11_3
            match = only(filter(m -> m.axion_index == restricted.axion_index &&
                    m.qcd_divisor_index == restricted.qcd_divisor_index, models_h11_3))
            @test restricted == match
        end
        # Every surviving pair satisfies the rule's own three conditions.
        for restricted in restricted_h11_3
            @test restricted.qcd_divisor_index in reference_h11_3.divisor_index
            @test restricted.qcd_divisor_index !=
                reference_h11_3.divisor_index[restricted.axion_index]
            @test PB.fuzzy_axion_criterion_two(
                tau_h11_3[restricted.qcd_divisor_index], restricted.lambda)
        end
    end
end

@testset "HP spectrum: one-axion analytic mass" begin
    # V = 10^-20 * (1 - cos(3θ)); second instanton is exactly absent.
    K = Hermitian(reshape([4.0], 1, 1))
    Q = reshape(Int[3, 6], 1, 2)
    L = [1.0 0.0;
         -20.0 -1000.0]

    # hp_spectrum uses the native column-oriented potential convention.
    hp = CYAxiverse.generate.hp_spectrum(K, L, Q; prec=200)

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

    hp_mass_only = CYAxiverse.generate.hp_spectrum(K, L, Q;
        prec=200, quartics=false)
    @test hp_mass_only["m"] == hp["m"]
    @test hp_mass_only["msign"] == hp["msign"]
    @test hp_mass_only["fK"] == hp["fK"]
    @test isempty(hp_mass_only["fpert"])
    @test isempty(hp_mass_only["λself"])
    @test size(hp_mass_only["λ31_i"]) == (4, 0)
    @test size(hp_mass_only["λ22_i"]) == (4, 0)
end

@testset "HP spectrum: quartic labels match accumulated values" begin
    # The quartic index lists must enumerate in the same order as the fused
    # instanton accumulation loop, which runs the first index slowest. A
    # two-dimensional comprehension iterates column-major and transposes the
    # labels relative to the values. That is invisible at h11 == 1 (both
    # families are empty) and at h11 <= 3 for λ22, so this fixture uses
    # h11 == 4 and checks every component against an independent reference.
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

    prec = 120
    hp = CYAxiverse.generate.hp_spectrum(K, L, Q; prec = prec)

    # Reference: component-at-a-time contraction with no loop fusion and no
    # index bookkeeping shared with the implementation under test.
    setprecision(ArbFloat; digits = prec)
    AF = typeof(ArbFloat(0))
    ninst = size(Q, 2)
    Lh = AF[AF(L[1, a]) * AF(10)^AF(L[2, a]) for a in 1:ninst]
    hess = zeros(AF, h11, h11)
    for a in 1:ninst, i in 1:h11, j in 1:h11
        hess[i, j] += Lh[a] * Q[i, a] * Q[j, a]
    end
    Kf = cholesky(Hermitian(AF.(Matrix(K))))
    whitened = Hermitian(Kf.L \ Matrix(Hermitian(hess)) / Kf.L')
    Tls = Kf.L' \ eigen(whitened).vectors
    QMs = AF.(transpose(Q)) * Tls
    to_out(v) = Float64(log10(abs(v))) + 4 * log10(2π)

    # λ31 is not symmetric under i <-> j, so transposed labels are a genuine
    # mislabelling rather than a reordering of equal values.
    @test size(hp["λ31_i"], 2) == h11 * (h11 - 1)
    for k in axes(hp["λ31_i"], 2)
        i, j = hp["λ31_i"][1, k] + 1, hp["λ31_i"][4, k] + 1
        ref = sum(Lh[a] * QMs[a, i]^3 * QMs[a, j] for a in 1:ninst)
        @test isapprox(hp["λ31"][k], to_out(ref); atol = 1e-9)
        @test hp["λ31sign"][k] == Int(sign(ref))
    end

    @test size(hp["λ22_i"], 2) == h11 * (h11 - 1) ÷ 2
    for k in axes(hp["λ22_i"], 2)
        i, j = hp["λ22_i"][1, k] + 1, hp["λ22_i"][4, k] + 1
        ref = sum(Lh[a] * QMs[a, i]^2 * QMs[a, j]^2 for a in 1:ninst)
        @test isapprox(hp["λ22"][k], to_out(ref); atol = 1e-9)
        @test hp["λ22sign"][k] == Int(sign(ref))
    end

    for i in 1:h11
        ref = sum(Lh[a] * QMs[a, i]^4 for a in 1:ninst)
        @test isapprox(hp["λself"][i], to_out(ref); atol = 1e-9)
    end

    # hp_spectrum and pq_spectrum must agree on the component labelling.
    pq = CYAxiverse.generate.pq_spectrum(K, L, Q)
    @test hp["λ31_i"] == pq.λ31_i
    @test hp["λ22_i"] == pq.λ22_i
end

@testset "basis_snf: id_coords is the matrix inverse" begin
    # `inv` inside a `@.` broadcast applies elementwise, which made id_coords a
    # thresholded copy of basis rather than its inverse.
    for rays in (Matrix{Int}(I, 3, 3), [1 0 0; 1 1 0; 0 1 1], [2 1 0; 0 1 0; 0 0 3])
        basis = CYAxiverse.generate.basis_snf(Matrix{Int}(rays))
        @test basis.basis * basis.id_coords == I
        @test basis.volume == abs(det(basis.basis))
    end
    # A case where the inverse genuinely differs from the basis itself.
    nontrivial = CYAxiverse.generate.basis_snf([2 1 0; 0 1 0; 0 0 3])
    @test nontrivial.basis != nontrivial.id_coords
end

@testset "read.L_arb: column-oriented instanton scales" begin
    mktempdir() do root
        geom = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom)
        # L is stored 2 × N: sign/mantissa on row 1, log10 scale on row 2.
        L = [   1.0   -1.0    1.0;
              -20.0  -25.0  -30.0]
        h5open(joinpath(geom, "cyax.h5"), "cw") do file
            potential = create_group(create_group(file, "cytools"), "potential")
            potential["L"] = L
        end
        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            setprecision(ArbFloat; digits = 60)
            scales = CYAxiverse.read.L_arb(2, 1, 1)
            @test length(scales) == size(L, 2)
            for a in axes(L, 2)
                @test isapprox(Float64(log10(abs(scales[a]))), L[2, a]; atol = 1e-12)
                @test Int(sign(scales[a])) == Int(sign(L[1, a]))
            end
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
    end
end

@testset "Concrete arbitrary-precision scales and L_arb layout" begin
    original_precision = precision(ArbFloat)
    setprecision(ArbFloat; digits=80)
    try
        scales = CYAxiverse.generate.pseudo_L(1, 1; log=false)
        @test isconcretetype(eltype(scales))
        @test eltype(scales) == typeof(ArbFloat(0))

        legacy_hessian = CYAxiverse.minimizer._legacy_hessian(
            Float64[1.0, -2.0], Int[1 2; 3 4], Float64[0.1, 0.2])
        @test isconcretetype(eltype(legacy_hessian))
        @test eltype(legacy_hessian) == typeof(ArbFloat(0))
        legacy_gradient = CYAxiverse.minimizer._legacy_gradient(
            Float64[1.0, -2.0], Int[1 2; 3 4], Float64[0.1, 0.2])
        @test isconcretetype(eltype(legacy_gradient))
        @test eltype(legacy_gradient) == typeof(ArbFloat(0))

        mktempdir() do root
            geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
            mkpath(geom_dir)
            stored_L = Float64[1.0  -2.0  3.0;
                               -4.0 -6.0 -8.0]
            h5open(joinpath(geom_dir, "cyax.h5"), "cw") do file
                potential = create_group(create_group(file, "cytools"), "potential")
                potential["L"] = stored_L
            end

            old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
            ENV["CYAXIVERSE_DATA_DIR"] = root
            try
                loaded = CYAxiverse.read.L_arb(2, 1, 1)
                expected = ArbFloat.(stored_L[1, :]) .*
                    ArbFloat(10) .^ ArbFloat.(stored_L[2, :])
                @test isconcretetype(eltype(loaded))
                @test eltype(loaded) == typeof(ArbFloat(0))
                @test loaded == expected
            finally
                old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                    (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
            end
        end
    finally
        setprecision(ArbFloat; bits=original_precision)
    end
end

@testset "Phase-hoisted Hessian parity" begin
    minimizer = CYAxiverse.minimizer
    generate = CYAxiverse.generate

    function legacy_phase_hessian(LV, QV, x)
        T = promote_type(eltype(LV), eltype(QV), eltype(x))
        hessian = zeros(T, size(QV, 1), size(QV, 1))
        for i in axes(QV, 1), j in axes(QV, 1)
            if i >= j
                hessian[i, j] = sum(LV' *
                    (@view(QV[i, :]) .* @view(QV[j, :]) .* cos.(x' * QV)))
            end
        end
        hessian
    end

    LV = Float64[1.25, -0.75, 0.5]
    QV = Float64[1.0 2.0 -1.0;
                 0.5 -3.0 4.0]
    x = Float64[0.37, -0.22]
    expected = legacy_phase_hessian(LV, QV, x)
    actual = zeros(Float64, 2, 2)
    minimizer._phase_hessian!(actual, LV, QV, cos.(x' * QV))
    @test actual == expected
    @test actual + actual' - Diagonal(actual) ==
        expected + expected' - Diagonal(expected)

    original_precision = precision(ArbFloat)
    setprecision(ArbFloat; digits=80)
    try
        T = typeof(ArbFloat(0))
        LV_arb = T.(LV)
        QV_arb = T.(QV)
        x_arb = T.(x)
        expected_arb = legacy_phase_hessian(LV_arb, QV_arb, x_arb)
        actual_arb = zeros(T, 2, 2)
        minimizer._phase_hessian!(actual_arb, LV_arb, QV_arb,
            cos.(x_arb' * QV_arb))
        @test actual_arb == expected_arb
        @test eltype(actual_arb) == T
    finally
        setprecision(ArbFloat; bits=original_precision)
    end

    function legacy_hessian_norm(x, Q::Matrix)
        hessian = zeros(size(Q, 1), size(Q, 1))
        if size(Q, 1) == 1
            for i in axes(Q, 1), j in axes(Q, 1)
                if i >= j
                    hessian[i, j] = (transpose(@view(Q[i, :])) *
                        @view(Q[j, :])) * cos.(x' * Q)[i]
                end
            end
            hessian = hessian + hessian' - Diagonal(hessian)
        elseif size(Q, 1) == size(Q, 2)
            for i in axes(Q, 1), j in axes(Q, 1)
                if i >= j
                    hessian[i, j] = (transpose(@view(Q[i, :])) *
                        @view(Q[j, :])) * cos.(x' * Q)[i]
                end
            end
            hessian = Hermitian(hessian + hessian' - Diagonal(hessian))
        else
            hessian = zeros(size(Q, 1), size(Q, 1), size(Q, 2))
            for i in axes(Q, 1), j in axes(Q, 1), k in axes(Q, 2)
                if i >= j
                    hessian[i, j, k] = (transpose(@view(Q[i, :])) *
                        @view(Q[j, :])) * cos.(x' * Q)[k]
                end
            end
            return hessian
        end
    end

    Qone = reshape(Float64[1.0, -2.0, 3.0], 1, 3)
    @test generate.hessian_norm([0.31], Qone) ==
        legacy_hessian_norm([0.31], Qone)

    Qsquare = Float64[1.0 2.0; 3.0 4.0]
    xsquare = Float64[0.17, -0.23]
    @test Matrix(generate.hessian_norm(xsquare, Qsquare)) ==
        Matrix(legacy_hessian_norm(xsquare, Qsquare))

    Qrect = Float64[1.0 2.0 3.0; -1.0 4.0 0.5]
    xrect = Float64[0.12, -0.41]
    @test generate.hessian_norm(xrect, Qrect) ==
        legacy_hessian_norm(xrect, Qrect)

    Qempty = zeros(Float64, 2, 0)
    @test size(generate.hessian_norm([0.2, -0.1], Qempty)) == (2, 2, 0)
end

@testset "Factored Kinv whitening remains stable when Kinv is ill-conditioned" begin
    # This Kinv has condition number about 4 × 10^10.  Forming K = inv(Kinv)
    # before whitening loses the small kinetic direction on older paths.
    Kinv = Float64[1.0 1.0 - 5e-11;
                   1.0 - 5e-11 1.0]
    Q = Int[1 0 1;
            0 1 1]
    L = Float64[1.0 1.0 1.0;
                -20.0 -21.0 -22.0]

    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        h5open(joinpath(geom_dir, "cyax.h5"), "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            potential["Q"] = Q
            potential["L"] = L
            geometric["Kinv"] = Kinv
        end

        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            geom_idx = CYAxiverse.structs.GeometryIndex(2, 1, 1)
            factored = CYAxiverse.read.potential_factored(geom_idx)
            @test factored.Kinv ≈ Kinv
            @test factored.C * transpose(factored.C) ≈ Kinv rtol=1e-12 atol=1e-15

            hp = CYAxiverse.generate.hp_spectrum(geom_idx;
                prec=200, quartics=false, selection=:raw)
            reference = setprecision(BigFloat, 400) do
                Cbig = cholesky(Symmetric(BigFloat.(Kinv))).L
                Hbig = BigFloat.(Q) *
                    Diagonal(BigFloat.(L[1, :]) .* (BigFloat(10) .^ BigFloat.(L[2, :]))) *
                    transpose(BigFloat.(Q))
                values = eigvals(Symmetric(transpose(Cbig) * Hbig * Cbig))
                Float64.(0.5 .* log10.(abs.(values)) .+ 9 .+
                    log10(BigFloat("2.435e18")) .+ log10(2BigFloat(π)))
            end
            @test hp["m"] ≈ reference atol=5e-6
        finally
            if old_data_dir === nothing
                delete!(ENV, "CYAXIVERSE_DATA_DIR")
            else
                ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir
            end
        end
    end
end

@testset "HP spectrum: raw and effective selections" begin
    K = Hermitian([4.0 0.0;
                   0.0 9.0])
    Q = [3 0 3;
         0 5 0]
    L = [1.0 1.0 1.0;
         -20.0 -30.0 -100.0]

    selected = CYAxiverse.generate.LQtildebar(L, Q)
    @test size(selected["Qhat"], 2) == 2
    hp_effective = CYAxiverse.generate.hp_spectrum(K, L, Q;
        prec=200, quartics=false, selection=:hp_effective)
    hp_provided = CYAxiverse.generate.hp_spectrum(K,
        selected["Lhat"], selected["Qhat"]; prec=200,
        quartics=false, selection=:raw)

    @test hp_effective["m"] == hp_provided["m"]
    @test hp_effective["msign"] == hp_provided["msign"]
    @test_throws ArgumentError CYAxiverse.generate.hp_spectrum(
        K, L, Q; quartics=false, selection=:unknown)
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
    hp = CYAxiverse.generate.hp_spectrum(K, L, Q; prec=200)

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
    physical_masses_only = CYAxiverse.generate.pq_physical_spectrum(
        K, L, Q; prec=200, quartics=false)
    @test physical_masses_only.mode_indices == physical.mode_indices
    @test all(isapprox.(physical_masses_only.m, physical.m; atol=1e-10))
    @test all(isapprox.(physical_masses_only.eigenvectors, physical.eigenvectors; atol=1e-10))
    @test isempty(physical_masses_only.λself)
    @test isempty(physical_masses_only.λselfsign)
    @test size(physical_masses_only.λ31_i) == (4, 0)
    @test size(physical_masses_only.λ22_i) == (4, 0)
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
    @test_logs (:warn, r"geometry=diagonal test") @test CYAxiverse.generate.pq_physical_mode_count(
        K, L, Q; threshold_log10=expected[1], prec=100, max_prec=100,
        label="diagonal test") == 1
end

@testset "PQ quartic cache and linear log-sum" begin
    G = CYAxiverse.generate

    Qpq = Float64[1.0 0.0 -2.0;
                  -2.0 3.0 0.5]
    charge_sign, charge_logabs = G._cache_charge_sign_logabs(Qpq)
    @test charge_sign isa Matrix{Int8}
    @test Int.(charge_sign) == [1 0 -1; -1 1 1]
    @test charge_logabs[1, 1] == 0.0
    @test charge_logabs[1, 2] == -Inf
    @test charge_logabs[1, 3] ≈ log(2.0)
    @test charge_logabs[2, 3] ≈ log(0.5)

    logs = [log(1.0), log(3.0), log(2.0), log(1e-200)]
    original_logs = copy(logs)
    expected_lse = log(sum(exp.(logs)))
    @test G.logsum_sorted!(logs, length(logs)) ≈ expected_lse rtol=1e-14
    @test logs == original_logs
    @test G.logsum_sorted!(Float64[], 0) == -Inf
    @test G.logsum_sorted!([-Inf, -Inf], 2) == -Inf
    @test G.logsum_sorted!([Inf, 0.0], 2) == Inf
    @test isnan(G.logsum_sorted!([NaN, 0.0], 2))

    sorted_reference = sort(copy(logs))
    reference_lse = sorted_reference[1]
    for i in 2:length(sorted_reference)
        reference_lse += G.gauss_sum(sorted_reference[i] - reference_lse)
    end
    @test G.logsum_sorted!(copy(logs), length(logs)) ≈ reference_lse rtol=1e-14

    scale_sign = Int[1, -1, 1, -1]
    scale_log = zeros(4)
    positive_logs = zeros(4)
    negative_logs = zeros(4)
    all_one_charges = ones(Float64, 4, 1)
    all_one_sign, all_one_logabs = G._cache_charge_sign_logabs(all_one_charges)
    exact_cancel = G.pq_contracted_log!(positive_logs, negative_logs,
        scale_sign, scale_log, all_one_sign, all_one_logabs, (1, 1, 1, 1))
    @test exact_cancel[1] == 0
    @test exact_cancel[2] == 0.0
    @test exact_cancel[3] ≈ log(4.0)

    near_cancel = G.pq_contracted_log!(positive_logs, negative_logs,
        scale_sign[1:2], [0.0, -log(2.0)], all_one_sign, all_one_logabs,
        (1, 1, 1, 1))
    @test near_cancel[1] == 1
    @test near_cancel[2] ≈ log(0.5) atol=1e-14
    @test near_cancel[3] ≈ log(1.5) atol=1e-14

    shrinking = G.pq_contracted_log!(positive_logs, negative_logs,
        @view(scale_sign[1:1]), @view(scale_log[1:1]), all_one_sign,
        all_one_logabs, (1, 1, 1, 1))
    @test shrinking == (1, 0.0, 0.0)

    K = Hermitian(reshape([4.0], 1, 1))
    Q = reshape(Int[3, 6], 1, 2)
    L = [1.0 1.0;
         -20.0 -1000.0]
    pq = G.pq_spectrum(K, L, Q; mixing_correction=:high_precision,
        prec=200, quartic_diagnostics=true)
    hp = G.hp_spectrum(K, L, Q; prec=200)
    @test pq.λselfsign == hp["λselfsign"]
    @test pq.λself ≈ hp["λself"] atol=1e-10
end

@testset "Float64 inertia certificate and Arb fallback" begin
    G = CYAxiverse.generate
    C = Matrix{Float64}(I, 2, 2)
    Q = Int[1 0 0;
            0 1 0]
    L = Float64[1.0 1.0 0.0;
                -20.0 -30.0 -400.0]
    Ltilde = L[:, 1:2]
    Qtilde = Q[:, 1:2]
    threshold = Float64(log10(G.constants()["Hubble"]))

    certificate = G._float64_leading_hessian_certificate(C, Ltilde, Qtilde)
    @test certificate !== nothing
    @test G._certified_float64_inertia_count(certificate, threshold) == 2

    mass_offset = 9.0 + Float64(log10(G.constants()["MPlanck"])) +
        Float64(G.constants()["log2π"])
    ambiguous_threshold = 0.5 * log10(abs(certificate.eigenvalues[1])) + mass_offset
    @test G._certified_float64_inertia_count(certificate, ambiguous_threshold) === nothing
    @test G._float64_leading_hessian_certificate(
        C, Float64[1.0 1.0; -400.0 -30.0], Qtilde) === nothing
    @test G._float64_leading_hessian_certificate(
        C, Float64[1.0 1.0; 400.0 30.0], Qtilde) === nothing

    @test G._certified_float64_window_counts(C, Ltilde, Qtilde,
        0.0, 40.0, 0.0) == (2, 0)
    @test G._next_confirmation_precision(1_000, 4_000) == 1_500
    @test G._next_confirmation_precision(1_500, 4_000) == 2_250
    @test G._next_confirmation_precision(3_375, 4_000) == 4_000
    W0, Cprecision0 = G.high_precision_leading_hessian(C, Ltilde, Qtilde; prec=80)
    _, _, _, _, records, _ = G._confirm_window_counts(W0, Cprecision0, C,
        Ltilde, Qtilde, ambiguous_threshold, ambiguous_threshold, 0.0,
        80, 120, true, "B2 precision schedule")
    @test first(records)[1] == 80
    @test last(records)[1] == 120

    @test G.pq_physical_mode_count(Hermitian(2.0 .* Matrix{Float64}(I, 2, 2)),
        L, Q; prec=120, confirm=true) == 2
    window = G.pq_window_spectrum(Hermitian(2.0 .* Matrix{Float64}(I, 2, 2)),
        L, Q; min_log10_mass=0.0, max_log10_mass=40.0, prec=120,
        confirm=true, quartics=false)
    @test window.diagnostics.counts_by_precision == [(120, 2, 0)]

    fallback_count = G._pq_physical_mode_count_factored(C, L, Q;
        threshold_log10=ambiguous_threshold, prec=80, confirm=false)
    reference_count = G.physical_mode_inertia_count(C, Ltilde, Qtilde,
        ambiguous_threshold, 80)
    @test fallback_count == reference_count
end

@testset "Hybrid spectrum: full fallback after nonconvergence" begin
    K = Hermitian([
        0.7946177070131799 0.16842382565791195 0.007532002811287394;
        0.16842382565791195 1.4905142746092241 -1.030351626502046;
        0.007532002811287394 -1.030351626502046 3.4403322647473655
    ])
    Q = [
        -1 1 -2 2 3 2;
        -3 0 0 -2 -1 1;
        -3 1 0 -1 2 -2
    ]
    L = vcat(ones(1, 6), reshape(-2.0 .* collect(1:6), 1, 6))
    reference = CYAxiverse.generate.pq_physical_spectrum(
        K, L, Q; threshold_log10=26.5, prec=100)
    hybrid = @test_logs (:warn, r"Falling back to the full high-precision eigensystem") CYAxiverse.generate.pq_hybrid_physical_spectrum(
        K, L, Q; threshold_log10=26.5, prec=100, maxiter=1,
        oversampling=0, schur_acceleration=false, quartics=false,
        label="nonconverged fallback test")
    @test hybrid.mode_indices == reference.mode_indices
    @test all(isapprox.(hybrid.m, reference.m; atol=1e-10))
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

@testset "Instanton hierarchy blocks" begin
    L = Float64[1.0 1.0 1.0 1.0;
                0.0 -0.2 -0.4 -10.0]
    hierarchy = CYAxiverse.generate.instanton_scale_blocks(L; gap_log10=0.5)
    @test [block.indices for block in hierarchy.blocks] == [[0, 1, 2], [3]]
    @test hierarchy.inter_block_gaps == [9.6]
    @test hierarchy.sorted_indices == [0, 1, 2, 3]

    merged = CYAxiverse.generate.instanton_scale_blocks(L; gap_log10=0.1,
        min_block_size=2)
    @test all(length(block.indices) >= 2 for block in merged.blocks)

    permutation = [3, 1, 4, 2]
    permuted = CYAxiverse.generate.instanton_scale_blocks(L[:, permutation];
        gap_log10=0.5)
    @test sort(vcat([block.indices for block in permuted.blocks]...)) == [0, 1, 2, 3]

    extreme = CYAxiverse.generate.instanton_scale_blocks(
        Float64[1.0 1.0 1.0; 1.0 -1.0 -1.0e6]; gap_log10=10.0)
    @test all(isfinite, extreme.inter_block_gaps)
    @test all(isfinite, extreme.sorted_log_scales)

    K = Hermitian(Matrix{Float64}(I, 2, 2))
    mixed_Q = Int[1 1 0; 0 0 1]
    mixed = CYAxiverse.generate.instanton_hierarchy_diagnostics(
        K, Float64[1.0 1.0 1.0; 0.0 -10.0 -10.2], mixed_Q;
        gap_log10=0.5)
    @test length(mixed.perturbative_splits) == 1
    @test mixed.perturbative_splits[1].off_block_norm > 0
    @test !mixed.perturbative_splits[1].certified_safe
end

@testset "PQ mass-window spectrum" begin
    K = Hermitian([4.0 0.0;
                   0.0 9.0])
    Q = [3 0 0;
         0 5 0]
    L = [1.0 1.0 0.0;
         -20.0 -30.0 -1000.0]
    reference = CYAxiverse.generate.pq_physical_spectrum(K, L, Q; prec=120)
    full_window = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=0.0, max_log10_mass=40.0, prec=120, confirm=false)
    @test full_window.mode_indices == reference.mode_indices
    @test all(isapprox.(full_window.m, reference.m; atol=1e-10))
    @test full_window.diagnostics.certified
    @test !full_window.diagnostics.fallback_used

    lower = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=reference.m[1] - 1.0, max_log10_mass=Inf,
        prec=120, confirm=false, quartics=true, mixed_quartics=false)
    hybrid_lower = CYAxiverse.generate.pq_hybrid_physical_spectrum(K, L, Q;
        threshold_log10=reference.m[1] - 1.0, prec=120, quartics=false)
    @test lower.mode_indices == [0, 1]
    @test lower.mode_indices == hybrid_lower.mode_indices
    @test all(isapprox.(lower.m, hybrid_lower.m; atol=1e-10))
    @test all(isapprox.(lower.λself, reference.λself; atol=1e-10))

    narrow = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=reference.m[1], max_log10_mass=reference.m[1],
        prec=120, confirm=false, quartics=false)
    @test narrow.mode_indices == [0]
    @test narrow.m[1] ≈ reference.m[1]

    empty_window = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=reference.m[end] + 1.0,
        max_log10_mass=reference.m[end] + 2.0, prec=120, confirm=false)
    @test isempty(empty_window.m)
    @test size(empty_window.eigenvectors) == (2, 0)
    @test empty_window.diagnostics.certified

    reversed_window = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=20.0, max_log10_mass=10.0, prec=120, confirm=false,
        quartics=false)
    @test isempty(reversed_window.m)
    @test reversed_window.diagnostics.certified
end

@testset "PQ mass-window scientific validation edges" begin
    K = Hermitian(Matrix{Float64}(I, 3, 3))
    Q = [1 0 0 0;
         0 1 0 0;
         0 0 1 0]
    L = [1.0 1.0 1.0 1.0;
         -20.0 -22.0 -24.0 -30.0]
    reference = CYAxiverse.generate.pq_physical_spectrum(K, L, Q; prec=160)

    full_window = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=reference.m[1] - 1.0,
        max_log10_mass=reference.m[end] + 1.0, prec=160,
        confirm=false, quartics=false)
    @test full_window.mode_indices == reference.mode_indices
    @test all(isapprox.(full_window.m, reference.m; atol=1e-10))
    @test full_window.diagnostics.certified
    @test !full_window.diagnostics.provisional

    exact_upper = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=-Inf, max_log10_mass=reference.m[1], prec=160,
        confirm=false, quartics=false)
    @test exact_upper.mode_indices == [0]
    @test exact_upper.m[1] ≈ reference.m[1]
    @test exact_upper.diagnostics.certified

    for boundary_margin in (0.0, 1e-8)
        margin_window = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
            min_log10_mass=-Inf, max_log10_mass=reference.m[1],
            boundary_margin_log10=boundary_margin, prec=160,
            confirm=false, quartics=false)
        @test margin_window.mode_indices == [0]
        @test margin_window.diagnostics.boundary_margin_log10 == boundary_margin
        @test margin_window.diagnostics.certified
    end
    @test_throws ArgumentError CYAxiverse.generate.pq_window_spectrum(
        K, L, Q; boundary_margin_log10=-1.0)

    interior = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=reference.m[2], max_log10_mass=reference.m[2],
        prec=160, confirm=false, quartics=false)
    @test interior.mode_indices == [1]
    @test interior.m[1] ≈ reference.m[2]
    @test interior.diagnostics.certified

    permutation = [3, 1, 4, 2]
    permuted = CYAxiverse.generate.pq_window_spectrum(K, L[:, permutation],
        Q[:, permutation]; min_log10_mass=reference.m[1] - 1.0,
        max_log10_mass=reference.m[end] + 1.0, prec=160,
        confirm=false, quartics=true, mixed_quartics=false)
    unpermuted = CYAxiverse.generate.pq_window_spectrum(K, L, Q;
        min_log10_mass=reference.m[1] - 1.0,
        max_log10_mass=reference.m[end] + 1.0, prec=160,
        confirm=false, quartics=true, mixed_quartics=false)
    @test permuted.mode_indices == unpermuted.mode_indices
    @test all(isapprox.(permuted.m, unpermuted.m; atol=1e-10))
    @test permuted.λselfsign == unpermuted.λselfsign
    @test all(isapprox.(permuted.λself, unpermuted.λself; atol=1e-10))

    nearly_degenerate = CYAxiverse.generate.instanton_scale_blocks(
        Float64[1.0 1.0 1.0 1.0;
                0.0 -0.1 -0.2 -10.0]; gap_log10=0.5)
    @test [length(block.indices) for block in nearly_degenerate.blocks] == [3, 1]
    @test nearly_degenerate.inter_block_gaps == [9.8]

    mixed_Q = Int[1 1 0;
                  0 1 1]
    mixed_L = Float64[1.0 1.0 1.0;
                     0.0 -10.0 -20.0]
    mixed_diagnostics = CYAxiverse.generate.instanton_hierarchy_diagnostics(
        Hermitian(Matrix{Float64}(I, 2, 2)), mixed_L, mixed_Q; gap_log10=1.0)
    @test length(mixed_diagnostics.perturbative_splits) == 2
    @test all(split -> split.off_block_norm > 0,
        mixed_diagnostics.perturbative_splits)
    @test all(split -> !split.certified_safe,
        mixed_diagnostics.perturbative_splits)

    mixed_window = CYAxiverse.generate.pq_window_spectrum(
        Hermitian(Matrix{Float64}(I, 2, 2)), mixed_L, mixed_Q;
        min_log10_mass=0.0, max_log10_mass=40.0, prec=160,
        confirm=false, quartics=false)
    @test mixed_window.diagnostics.fallback_used
    @test mixed_window.diagnostics.certified
    @test !mixed_window.diagnostics.provisional

    extreme = CYAxiverse.generate.pq_window_spectrum(
        Hermitian(Matrix{Float64}(I, 3, 3)),
        Float64[1.0 1.0 1.0 1.0;
                -1.0e6 -1.0e6 + 1.0 -1.0e6 + 2.0 -1.0e6 + 3.0],
        Q; min_log10_mass=-600_000.0, max_log10_mass=40.0,
        prec=160, confirm=false, quartics=false)
    @test length(extreme.m) == 3
    @test all(isfinite, extreme.m)
    @test all(isfinite, extreme.eigenvectors)
    @test extreme.diagnostics.certified
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
    hp = CYAxiverse.generate.hp_spectrum(K, L, Q; prec=200)

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

@testset "analyze_inflation_candidates: whitened point classification" begin
    # The script defines `derivatives`, `classify_point` and `main`, names that
    # would collide with the scan scripts already included at top level, so it
    # is loaded into its own module.
    candidates = Module(:AnalyzeInflationCandidates)
    Base.include(candidates, joinpath(@__DIR__, "..", "scripts",
        "analyze_inflation_candidates.jl"))

    # The implementation exactly as it stood before the whitening change, kept
    # here so the equivalence is pinned rather than asserted.
    function classify_point_explicit_inverse(theta, Q, L, K)
        d = candidates.derivatives(theta, Q, L)
        factor = cholesky(Hermitian(K)).L
        to_theta = inv(factor')
        canonical_hessian = to_theta' * d.hessian * to_theta
        eigs = eigvals(Hermitian(canonical_hessian))
        gradnorm = sqrt(max(dot(d.gradient, inv(K) * d.gradient), 0.0))
        (; value=d.value, gradnorm, hessian_eigenvalues=eigs)
    end

    function classification_fixture(h11, condition_number; seed = 42)
        rng = MersenneTwister(seed)
        Q = Matrix(CYAxiverse.generate.pseudo_Q(h11, 1)')
        L = Matrix(CYAxiverse.generate.pseudo_L(h11, 1)')
        rotation = qr(randn(rng, h11, h11)).Q
        spectrum = exp10.(range(0, -log10(condition_number); length = h11))
        K = Matrix(Hermitian(rotation * Diagonal(spectrum) * rotation'))
        Q, L, K
    end

    @testset "reproduces the explicit-inverse form" begin
        for h11 in (6, 10, 20), condition_number in (1e2, 1e6)
            Q, L, K = classification_fixture(h11, condition_number)
            theta = 0.37 .* ones(h11) .+ 0.11 .* (1:h11) ./ h11
            expected = classify_point_explicit_inverse(theta, Q, L, K)
            actual = candidates.classify_point(theta, Q, L, cholesky(Hermitian(K)))
            @test actual.value == expected.value
            @test actual.gradnorm ≈ expected.gradnorm rtol=1e-8
            @test actual.hessian_eigenvalues ≈ expected.hessian_eigenvalues atol=
                1e-8 * max(maximum(abs, expected.hessian_eigenvalues), 1.0)
        end
    end

    @testset "gradnorm is the K-inverse norm of the gradient" begin
        # norm(L \ g) == sqrt(g' K^-1 g) is the identity the whitened form
        # relies on; without it `epsilon` would not be the slow-roll parameter.
        for h11 in (8, 16)
            Q, L, K = classification_fixture(h11, 1e4)
            theta = 0.29 .* ones(h11) .+ 0.07 .* (1:h11) ./ h11
            d = candidates.derivatives(theta, Q, L)
            actual = candidates.classify_point(theta, Q, L, cholesky(Hermitian(K)))
            @test actual.gradnorm ≈ sqrt(dot(d.gradient, K \ d.gradient)) rtol=1e-9
            @test actual.gradnorm >= 0
        end
    end

    @testset "factorization may be hoisted out of a point loop" begin
        # `analyze_geometry` factors K once and reuses it for every critical
        # point; that must agree with factoring per point.
        h11 = 12
        Q, L, K = classification_fixture(h11, 1e5)
        factor = cholesky(Hermitian(K))
        for i in 1:4
            theta = 0.3 .+ 0.4 .* rand(MersenneTwister(i), h11)
            hoisted = candidates.classify_point(theta, Q, L, factor)
            per_point = candidates.classify_point(theta, Q, L, K)
            @test hoisted.gradnorm == per_point.gradnorm
            @test hoisted.epsilon == per_point.epsilon
            @test hoisted.hessian_eigenvalues == per_point.hessian_eigenvalues
            @test hoisted.negative_modes == per_point.negative_modes
        end
    end

    @testset "mode counts are tolerance-aware and partition the spectrum" begin
        for h11 in (8, 12, 16)
            Q, L, K = classification_fixture(h11, 1e5)
            theta = 0.31 .* ones(h11) .+ 0.09 .* (1:h11) ./ h11
            c = candidates.classify_point(theta, Q, L, cholesky(Hermitian(K)))

            # The invariant `analyze_geometry` now asserts for every point.
            @test c.negative_modes + c.zeroish_modes + c.positive_modes == h11

            # The previous overlapping triple could not satisfy it, because a
            # zeroish eigenvalue was also counted as negative or positive.
            scale = max(maximum(abs, c.hessian_eigenvalues), 1.0)
            tolerance = 1e-10 * scale
            old_negative = count(<(0), c.hessian_eigenvalues)
            old_zeroish = count(x -> abs(x) <= tolerance, c.hessian_eigenvalues)
            old_positive = count(>(0), c.hessian_eigenvalues)
            if old_zeroish > 0
                @test old_negative + old_zeroish + old_positive > h11
                @test c.negative_modes <= old_negative
            end

            # No mode is called a tachyon unless it clears the band.
            @test c.negative_modes ==
                  count(x -> x < -tolerance, c.hessian_eigenvalues)
        end
    end

    @testset "the flatness metric excludes curvature-free directions" begin
        # abs_min_eta/min_eta used to range over the whole spectrum, so a
        # curvature-free direction drove them to zero (or spuriously
        # negative on noise). They are now restricted to directions with
        # resolvable curvature (DECISION-ETA); the naive whole-spectrum
        # minimum is never larger, and strictly smaller when a zeroish mode
        # is present.
        for h11 in (10, 14)
            Q, L, K = classification_fixture(h11, 1e5)
            theta = 0.23 .* ones(h11) .+ 0.13 .* (1:h11) ./ h11
            c = candidates.classify_point(theta, Q, L, cholesky(Hermitian(K)))
            naive_abs_min_eta = minimum(abs.(c.eta_values))
            @test c.abs_min_eta >= naive_abs_min_eta
            if c.zeroish_modes > 0
                @test c.abs_min_eta > naive_abs_min_eta
            end
        end
    end
end

@testset "spectrum_mode_counts partitions a symmetric spectrum" begin
    counts = CYAxiverse.generate.spectrum_mode_counts

    @testset "the three counts partition the spectrum" begin
        # This is the property the previous `count(<(0))` / toleranced-zeroish
        # / `count(>(0))` triple could not offer: those overlapped, so they
        # summed to more than the mode count and no invariant was available.
        for spectrum in (
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [-1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1e-18, -1e-18, 5.0],
                randn(MersenneTwister(11), 50),
                exp10.(range(-20, 3; length = 25)))
            modes = counts(spectrum)
            @test modes.negative + modes.zeroish + modes.positive ==
                  length(spectrum)
        end
    end

    @testset "the zeroish band is relative to the largest mode" begin
        # 1e-17 relative to a scale of 1e6 is inside a 1e-10 band, so it is
        # neither a tachyon nor a positive mode.
        modes = counts([1e6, -1e-11, 1e-11])
        @test modes.scale == 1e6
        @test modes.tolerance == 1e-4
        @test modes.negative == 0
        @test modes.positive == 1
        @test modes.zeroish == 2

        # The same two entries against a scale of 1 are resolvable.
        resolved = counts([1.0, -1e-11, 1e-11]; relative_tolerance = 1e-13)
        @test resolved.negative == 1
        @test resolved.positive == 2
        @test resolved.zeroish == 0
    end

    @testset "scale has an absolute floor of one" begin
        # A uniformly tiny spectrum must not manufacture a huge relative
        # tolerance from its own smallness.
        modes = counts([1e-30, -1e-30])
        @test modes.scale == 1.0
        @test modes.tolerance == 1e-10
        @test modes.zeroish == 2
    end

    @testset "boundary is inclusive for zeroish" begin
        # |eig| == tolerance counts as zeroish, matching the `<=` the previous
        # zeroish test used.
        modes = counts([1.0, 1e-10, -1e-10]; relative_tolerance = 1e-10)
        @test modes.zeroish == 2
        @test modes.negative == 0
        @test modes.positive == 1
    end

    @testset "a sign test disagrees exactly where it should" begin
        # The regression this function exists to prevent: a mode at 1e-17
        # relative is reported as a tachyon by a sign test.
        spectrum = [5.0, 3.0, -1e-16]
        @test count(<(0), spectrum) == 1          # old behaviour
        @test counts(spectrum).negative == 0      # new behaviour
        @test counts(spectrum).zeroish == 1
    end

    @testset "degenerate inputs" begin
        empty_modes = counts(Float64[])
        @test empty_modes.negative == 0
        @test empty_modes.zeroish == 0
        @test empty_modes.positive == 0

        # A zero tolerance reduces to an exact sign test with zeros grouped
        # as zeroish, still partitioning.
        exact = counts([-1.0, 0.0, 1.0]; relative_tolerance = 0.0)
        @test (exact.negative, exact.zeroish, exact.positive) == (1, 1, 1)

        @test_throws ArgumentError counts([1.0]; relative_tolerance = -1e-10)
    end

    @testset "integer spectra are accepted" begin
        modes = counts([-2, 0, 3])
        @test (modes.negative, modes.zeroish, modes.positive) == (1, 1, 1)
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

        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        try
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
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
    end
end

@testset "Tier-A optimisations preserve values" begin
    # A2: _quartic_index_matrix replaces `hcat(collect.(idx)...) .- 1`.
    let
        legacy(indices) = isempty(indices) ? zeros(Int, 4, 0) :
                          hcat(collect.(indices)...) .- 1
        for h11 in (1, 2, 4, 7)
            i31 = [(i, i, i, j) for i in 1:h11 for j in 1:h11 if i != j]
            i22 = [(i, i, j, j) for i in 1:h11 for j in 1:i-1]
            @test CYAxiverse.generate._quartic_index_matrix(i31) == legacy(i31)
            @test CYAxiverse.generate._quartic_index_matrix(i22) == legacy(i22)
        end
        # The empty case must keep its shape, not collapse to a 0x0.
        @test size(CYAxiverse.generate._quartic_index_matrix(NTuple{4, Int}[])) == (4, 0)
    end

    # A2: _hcat_columns replaces `hcat(blocks...)`.
    let blocks = [rand(3, 2) for _ in 1:5]
        @test CYAxiverse.generate._hcat_columns(blocks) == hcat(blocks...)
    end
    let blocks = [rand(4) for _ in 1:6]
        @test CYAxiverse.generate._hcat_columns(blocks) == hcat(blocks...)
    end

    # A9: sandwiching between orthonormal-column QR factors leaves the
    # operator norm unchanged, so the thin-QR form is exact.
    let
        for (nl, nr, h11) in ((300, 200, 12), (150, 150, 8), (40, 90, 20))
            A = randn(nl, h11)
            B = randn(nr, h11)
            RA = qr(A).R
            RB = qr(B).R
            @test isapprox(opnorm(RA), opnorm(A); rtol = 1e-12)
            @test isapprox(opnorm(RB), opnorm(B); rtol = 1e-12)
            @test isapprox(opnorm(RA * RB'), opnorm(A * B'); rtol = 1e-10)
        end
    end

    # A11: _hp_selected_potential now routes through LQtilde + αmatrix
    # instead of LQtildebar. The combined (Lhat, Qhat) pair must be
    # identical, since αmatrix keeps Qhat and Qbar separate where
    # LQtildebar pre-concatenates them.
    let
        for h11 in (4, 6, 8, 10, 12, 15)
            Q = Matrix(CYAxiverse.generate.pseudo_Q(h11, 1)')
            L = Matrix(CYAxiverse.generate.pseudo_L(h11, 1)')
            legacy = CYAxiverse.generate.LQtildebar(Matrix{Float64}(L), Matrix{Int}(Q))
            legacy_L = Matrix{Float64}(legacy["Lhat"])
            legacy_Q = Matrix{Int}(legacy["Qhat"])
            fast_L, fast_Q = CYAxiverse.generate._hp_selected_potential(L, Q, :hp_effective)
            @test size(fast_Q) == size(legacy_Q)
            @test fast_Q == legacy_Q
            @test fast_L == legacy_L
            # LQtilde uses a floating-point rank test where LQtildebar uses
            # exact Nemo.nullspace. Pin that the fast selector never accepts
            # a rationally dependent column.
            @test rank(Matrix{Rational{BigInt}}(fast_Q)) == size(fast_Q, 2)
        end
    end
end

@testset "Orientifold axiverse database bridge (Phase 1, h11=2)" begin
    # A synthetic malformed-dimension geometry must be rejected rather than
    # silently accepted: Q's instanton count disagrees with L's, and K's
    # axion count disagrees with Q's, exercising the two DimensionMismatch
    # guards `oriented_potential` places at the read boundary before any
    # numerical code sees the arrays.
    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        path = joinpath(geom_dir, "cyax.h5")
        h5open(path, "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            # Q has 3 instanton columns; L has 4: an instanton-count mismatch
            # that no orientation-guessing transpose can repair (2 rows each,
            # so neither candidate transpose changes the column counts).
            potential["Q"] = Int[1 0 1; 0 1 1]
            potential["L"] = Float64[1.0 1.0 1.0 1.0; -5.0 -10.0 -1.0 -2.0]
            geometric["Kinv"] = Matrix{Float64}(I, 2, 2)
        end
        geom_idx = CYAxiverse.structs.GeometryIndex(2, 1, 1)
        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            @test_throws DimensionMismatch CYAxiverse.read.oriented_potential(geom_idx)
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
    end

    # K built from a non-square Kinv must also be rejected.
    mktempdir() do root
        geom_dir = joinpath(root, "h11_002", "np_0000001", "cy_0000001")
        mkpath(geom_dir)
        path = joinpath(geom_dir, "cyax.h5")
        h5open(path, "cw") do file
            cytools = create_group(file, "cytools")
            potential = create_group(cytools, "potential")
            geometric = create_group(cytools, "geometric")
            potential["Q"] = Int[1 0; 0 1]
            potential["L"] = Float64[1.0 1.0; -5.0 -10.0]
            geometric["Kinv"] = Float64[1.0 0.0 0.0; 0.0 1.0 0.0]
        end
        geom_idx = CYAxiverse.structs.GeometryIndex(2, 1, 1)
        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
            @test_throws DimensionMismatch CYAxiverse.read.oriented_potential(geom_idx)
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
        end
    end

    # Round-trip one real, ledger-verified h11=2 trilayer geometry written by
    # scripts/build_orientifold_axion_database.py, when that database is
    # present on this machine.  The database lives under the workspace
    # parent's `data/` directory (outside the package Git checkout, per
    # `.github/copilot-instructions.md`'s documentation-placement rule), so
    # this test skips gracefully -- rather than failing -- wherever that
    # local data is not present, e.g. in CI.
    db_root = normpath(joinpath(@__DIR__, "..", "..", "data",
        "orientifold_axiverse_database_20260821"))
    h11_dir = joinpath(db_root, "h11_002")
    if isdir(h11_dir)
        np_dirs = sort(filter(name -> startswith(name, "np_"), readdir(h11_dir)))
        @test !isempty(np_dirs)
        if !isempty(np_dirs)
            np_index = parse(Int, np_dirs[1][4:end])
            np_dir = joinpath(h11_dir, np_dirs[1])
            cy_dirs = sort(filter(name -> startswith(name, "cy_"), readdir(np_dir)))
            @test !isempty(cy_dirs)
            cy_index = parse(Int, cy_dirs[1][4:end])
            geom_idx = CYAxiverse.structs.GeometryIndex(2, np_index, cy_index)

            old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
            ENV["CYAXIVERSE_DATA_DIR"] = db_root
            try
                # cytools/geometric round-trip.
                geometry = CYAxiverse.read.geometry(geom_idx)
                @test geometry.h21 > 0
                @test length(geometry.τ_volumes) == 2
                @test size(geometry.kinv) == (2, 2)
                @test geometry.cy_volume > 0

                # cytools/potential round-trip, both raw and orientation-safe.
                raw = CYAxiverse.read.potential(geom_idx)
                normalised = CYAxiverse.read.oriented_potential(geom_idx)
                @test size(normalised.Q, 1) == 2
                @test size(normalised.L, 1) == 2
                @test size(normalised.Q, 2) == size(normalised.L, 2)
                # Confirm the boundary handles whichever raw HDF5.jl
                # orientation this cross-language (h5py -> HDF5.jl) artifact
                # produced: either raw already matches the canonical
                # (h11, N)/(2, N) shape, or its transpose does.
                raw_or_transposed_matches = (
                    (size(raw.Q, 1) == 2 && size(raw.L, 1) == 2) ||
                    (size(raw.Q, 2) == 2 && size(raw.L, 2) == 2)
                )
                @test raw_or_transposed_matches

                # Kinetic matrix K = (1/2) g must be symmetric positive-definite.
                @test isposdef(normalised.K)
                @test issymmetric(Matrix(normalised.K))

                # Visible-sector QCD divisor volume must be normalized to 40
                # under the homogeneous-qcd-volume-40-v1 convention.
                visible = CYAxiverse.read.visible_sector(geom_idx)
                @test visible !== nothing
                @test isapprox(visible.qcd_divisor_volume, 40.0; atol=1e-6)
                @test visible.qcd_invariant

                # orientifold/ provenance group: every required key present
                # and non-empty.
                h5open(CYAxiverse.filestructure.cyax_file(geom_idx), "r") do file
                    @test haskey(file, "orientifold")
                    group = file["orientifold"]
                    for name in ("h2_involution_matrix", "lattice_matrix",
                            "torus_shift_numerator", "torus_shift_denominator",
                            "lambda_f", "h11_minus", "h11_plus", "h21_plus")
                        @test haskey(group, name)
                    end
                    group_attrs = HDF5.attributes(group)
                    for name in ("polytope_normal_form_id", "frst_hash",
                            "source_ledger_path", "source_ledger_sha256",
                            "source_commit", "cytools_version",
                            "normalization_map_version",
                            "orientifold_provenance_schema_version",
                            "bridge_schema_version")
                        @test haskey(group_attrs, name)
                        @test !isempty(String(HDF5.read(group_attrs[name])))
                    end
                    @test Bool(HDF5.read(HDF5.attributes(file)["orientifold_provenance_complete"]))
                end
            finally
                old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                    (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
            end
        end
    else
        @info "skipping orientifold axiverse database round-trip test: " *
            "$h11_dir not present on this machine"
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
        old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
        ENV["CYAXIVERSE_DATA_DIR"] = root
        try
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
                cat = file["inflation/catastrophes"]
                @test String(HDF5.read(cat["status"])) == "completed"
                @test HDF5.read(cat["leading_minima_count"]) ==
                    catastrophe_outcome.result.leading_minima_count
                @test HDF5.read(cat["saddle_count"]) ==
                    catastrophe_outcome.result.saddle_count
                @test HDF5.read(cat["catastrophes_present"]) ==
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
                @test HDF5.read(cat["saddle_count"]) ==
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
                @test HDF5.read(cat["saddle_count"]) == -1
                @test String(HDF5.read(cat["error"])) == "synthetic failure for testing"
                ef = file["inflation/efolds"]
                @test String(HDF5.read(ef["search_status"])) == "failed"
                @test HDF5.read(ef["n_candidates"]) == -1
            end

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
        finally
            old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
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
                old_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
                ENV["CYAXIVERSE_DATA_DIR"] = db_root
                try
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
                finally
                    old_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
                        (ENV["CYAXIVERSE_DATA_DIR"] = old_data_dir)
                end
            end
        end
    else
        @info "skipping Pipeline 2 real-data round-trip test: " *
            "$h11_dir not present on this machine"
    end
end
