using CYAxiverse
using LinearAlgebra
using SparseArrays
using Test
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "vacua_pipeline.jl"))

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

@testset "Paper reproduction benchmarks" begin
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
    @test length(truncated.amplitudes) == 12
    @test length(full.amplitudes) == 78
    @test norm(truncated.gradient, Inf) / maximum(abs, truncated.amplitudes) < 1e-10

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
            h5open(path, "w") do _ end

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
