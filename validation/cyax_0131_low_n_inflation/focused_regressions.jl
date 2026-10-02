#!/usr/bin/env julia

using CYAxiverse
using Test

include(joinpath(@__DIR__, "..", "..", "scripts",
    "inflation_refinement_common.jl"))
include(joinpath(@__DIR__, "..", "..", "scripts",
    "inflation_diagnostics_common.jl"))

const AUTHOR = CYAxiverse.paper_benchmarks.author_inflation
const DELTA_K = 1.5320548620798324e-3

@testset "CYAX-0131 author trajectory gates" begin
    calibration = AUTHOR.n8_row2_phase_catastrophe()
    @test calibration.status === :completed
    @test calibration.phase_vector == [0.0, 0.04, zeros(8)...]
    @test length(calibration.phase_path) == 400
    @test all(step.converged for step in calibration.phase_path)
    @test calibration.refined.solver.precision_bits == 128
    @test calibration.refined.gradient_residual < big"1e-30"
    @test calibration.refined.null_residual < big"1e-30"
    @test count(>(0), calibration.refined.eigenvalues) == 7
    @test length(calibration.stationary_path) == 12
    @test calibration.sign_change
    @test calibration.bracket.lower_minimum_eigenvalue > 0
    @test calibration.bracket.upper_minimum_eigenvalue < 0

    config = inflation_refinement_config(precision_bits=64,
        max_time=10, max_step=1, sample_count=2, maxiters=1_000_000,
        reltol=1e-8, abstol=1e-10,
        critical_k=calibration.refined.k,
        phases=calibration.phase_vector,
        basis_theta=calibration.refined.theta,
        measurement_scope=:cold)
    candidate = inflation_refinement_candidate(
        "cyax-0131-focused-author-route"; delta_k=DELTA_K,
        screening=(status=:reference_anchor, measurement_scope=:cold))
    refined = refine_inflation_candidate(candidate; config)
    @test refined.summary.refinement_status === :completed
    @test refined.summary.solver_retcode ==
        string(RefinementReturnCode.Success)
    @test refined.summary.entered_slow_roll
    @test refined.summary.accepted_steps > 0
    @test refined.summary.physical_k == refined.trajectory.k
    expected_physical_k = setprecision(BigFloat, 64) do
        BigFloat(calibration.refined.k) + BigFloat(DELTA_K)
    end
    @test refined.trajectory.k == expected_physical_k
    @test length(refined.trajectory.samples) == 2

    diagnostics = inflation_refinement_author_diagnostics(
        refined; sample_index=2)
    @test diagnostics.sample_index == 2
    @test diagnostics.sample_n == refined.trajectory.samples[2].n
    @test diagnostics.ne == refined.summary.slow_roll_efolds
    expected_window_efolds = setprecision(BigFloat,
        refined.summary.precision_bits) do
        BigFloat(refined.summary.end_n) - BigFloat(refined.summary.entry_n)
    end
    @test diagnostics.ne == expected_window_efolds
    @test diagnostics.pivot_interpretation === :none
    @test diagnostics.observational_acceptance === :not_evaluated
    @test diagnostics.physical_k_reestablished

    failed_refinement = merge(refined, (; summary=merge(
        refined.summary, (; refinement_status=:failed))))
    @test_throws ArgumentError inflation_refinement_author_diagnostics(
        failed_refinement; sample_index=2)
    zero_steps = merge(refined.trajectory, (; solver=merge(
        refined.trajectory.solver, (; accepted_steps=0))))
    @test_throws ArgumentError inflation_author_trajectory_diagnostics(
        zero_steps; sample_index=2)
    failed_solver = merge(refined.trajectory, (; solver=merge(
        refined.trajectory.solver, (; retcode=RefinementReturnCode.MaxIters))))
    @test_throws ArgumentError inflation_author_trajectory_diagnostics(
        failed_solver; sample_index=2)
    wrong_scale = merge(refined.trajectory, (; k=refined.trajectory.k + big"1e-10"))
    @test_throws ArgumentError inflation_author_trajectory_diagnostics(
        wrong_scale; sample_index=2)
end
