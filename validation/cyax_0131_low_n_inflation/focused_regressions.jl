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
    @test calibration.refined_phase_vector ==
        setprecision(BigFloat, 128) do
            BigFloat.(calibration.phase_vector)
        end
    @test length(calibration.refined.eigenvalues) == 8
    @test minimum(abs, calibration.refined.eigenvalues) <=
        sqrt(length(calibration.refined.null_vector)) *
        calibration.refined.null_residual
    @test length(calibration.stationary_path) == 12
    @test calibration.sign_change
    @test calibration.bracket.lower_minimum_eigenvalue > 0
    @test calibration.bracket.upper_minimum_eigenvalue < 0
    @test calibration.refined.k > calibration.bracket.lower_k
    @test calibration.refined.k < calibration.bracket.upper_k
    println((record=:repaired_n8_row2_calibration,
        model=calibration.model, phase_vector=calibration.phase_vector,
        refined_phase_vector=calibration.refined_phase_vector,
        refined_k=calibration.refined.k,
        gradient_residual=calibration.refined.gradient_residual,
        null_residual=calibration.refined.null_residual,
        eigenvalues=calibration.refined.eigenvalues,
        minimum_hessian_signs=(lower=calibration.bracket.lower_minimum_eigenvalue,
            upper=calibration.bracket.upper_minimum_eigenvalue),
        bracket=(lower_k=calibration.bracket.lower_k,
            upper_k=calibration.bracket.upper_k,
            width=calibration.bracket.width),
        phase_steps=length(calibration.phase_path),
        stationary_points=length(calibration.stationary_path)))

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
    @test refined.summary.refinement_status === :censored
    @test refined.summary.solver_retcode ==
        string(RefinementReturnCode.Success)
    @test refined.summary.entered_slow_roll
    @test refined.summary.accepted_steps > 0
    @test refined.summary.physical_k == refined.trajectory.k
    expected_physical_k = setprecision(BigFloat, 64) do
        BigFloat(calibration.refined.k) + BigFloat(DELTA_K)
    end
    @test refined.trajectory.k == expected_physical_k
    @test refined.trajectory.entered_slow_roll
    @test refined.trajectory.end_event === :tmax
    @test !refined.trajectory.terminated
    @test isempty(refined.trajectory.samples)
    @test refined.summary.slow_roll_efolds === nothing
    @test refined.summary.censored_slow_roll_duration > 0
    println((record=:bounded_open_window_gate_case,
        model=refined.summary.model_route, precision_bits=refined.summary.precision_bits,
        max_time=refined.summary.max_time, maxiters=refined.summary.maxiters,
        reltol=refined.summary.reltol, abstol=refined.summary.abstol,
        solver_method=refined.summary.solver_method,
        solver_retcode=refined.summary.solver_retcode,
        accepted_steps=refined.summary.accepted_steps,
        rejected_steps=refined.summary.rejected_steps,
        rhs_evaluations=refined.summary.rhs_evaluations,
        jacobian_evaluations=refined.summary.jacobian_evaluations,
        end_event=refined.summary.end_event, terminated=refined.summary.terminated,
        censored_slow_roll_duration=refined.summary.censored_slow_roll_duration,
        completed_ne=refined.summary.slow_roll_efolds,
        sample_count=length(refined.trajectory.samples)))
    @test_throws ArgumentError inflation_refinement_author_diagnostics(
        refined; sample_index=1, physical_witness=calibration)

    synthetic_samples = [
        (n=BigFloat(2), epsilon=BigFloat("0.01"),
         eta_parallel=BigFloat("-0.01"), potential=BigFloat(1),
         tangent=BigFloat[1, 0, 0, 0, 0, 0, 0, 0]),
        (n=BigFloat(8), epsilon=BigFloat("0.02"),
         eta_parallel=BigFloat("-0.02"), potential=BigFloat(1),
         tangent=BigFloat[0, 1, 0, 0, 0, 0, 0, 0]),
    ]
    complete_trajectory = merge(refined.trajectory, (; entered_slow_roll=true,
        terminated=true, end_event=:epsilon, entry_n=BigFloat(2),
        end_n=BigFloat(8), efolds=BigFloat(8),
        slow_roll_efolds=BigFloat(6), samples=synthetic_samples))
    complete_refinement = merge(refined, (; trajectory=complete_trajectory,
        summary=merge(refined.summary, (; refinement_status=:completed,
            entered_slow_roll=true, terminated=true, end_event=:epsilon,
            entry_n=BigFloat(2), end_n=BigFloat(8), efolds=BigFloat(8),
            slow_roll_efolds=BigFloat(6),
            ne_window=(BigFloat(2), BigFloat(8)),
            physical_k=complete_trajectory.k))))
    diagnostics = inflation_refinement_author_diagnostics(
        complete_refinement; sample_index=2, physical_witness=calibration)
    @test diagnostics.sample_index == 2
    @test diagnostics.sample_n == complete_trajectory.samples[2].n
    @test diagnostics.ne == complete_trajectory.slow_roll_efolds
    @test diagnostics.pivot_interpretation === :none
    @test diagnostics.observational_acceptance === :not_evaluated
    @test diagnostics.physical_k_reestablished
    @test diagnostics.ne_window == (BigFloat(2), BigFloat(8))

    failed_refinement = merge(complete_refinement, (; summary=merge(
        complete_refinement.summary, (; refinement_status=:failed))))
    @test_throws ArgumentError inflation_refinement_author_diagnostics(
        failed_refinement; sample_index=2, physical_witness=calibration)
    @test_throws ArgumentError inflation_refinement_author_diagnostics(
        complete_refinement; sample_index=2, physical_witness=nothing)
    wrong_phase_witness = merge(calibration,
        (; phase_vector=copy(calibration.phase_vector)))
    wrong_phase_witness.phase_vector[2] = nextfloat(0.04)
    @test_throws ArgumentError inflation_refinement_author_diagnostics(
        complete_refinement; sample_index=2, physical_witness=wrong_phase_witness)
    wrong_metric_witness = merge(calibration, (; metric_basis=:other_basis))
    @test_throws ArgumentError inflation_refinement_author_diagnostics(
        complete_refinement; sample_index=2, physical_witness=wrong_metric_witness)
    wrong_k_refinement = merge(complete_refinement, (; summary=merge(
        complete_refinement.summary,
        (; critical_k=complete_refinement.summary.critical_k + big"1e-10"))))
    @test_throws ArgumentError inflation_refinement_author_diagnostics(
        wrong_k_refinement; sample_index=2, physical_witness=calibration)
    wrong_coordinates = merge(complete_trajectory,
        (; basis_theta=complete_trajectory.basis_theta .+ big"1e-20"))
    wrong_coordinates = merge(wrong_coordinates,
        (; critical_theta=copy(wrong_coordinates.basis_theta)))
    @test_throws ArgumentError inflation_author_trajectory_diagnostics(
        wrong_coordinates; sample_index=2, physical_witness=calibration)
    zero_steps = merge(complete_trajectory, (; solver=merge(
        complete_trajectory.solver, (; accepted_steps=0))))
    @test_throws ArgumentError inflation_author_trajectory_diagnostics(
        zero_steps; sample_index=2, physical_witness=calibration)
    failed_solver = merge(complete_trajectory, (; solver=merge(
        complete_trajectory.solver, (; retcode=RefinementReturnCode.MaxIters))))
    @test_throws ArgumentError inflation_author_trajectory_diagnostics(
        failed_solver; sample_index=2, physical_witness=calibration)
    wrong_scale = merge(complete_trajectory,
        (; k=complete_trajectory.k + big"1e-10"))
    @test_throws ArgumentError inflation_author_trajectory_diagnostics(
        wrong_scale; sample_index=2, physical_witness=calibration)
end
