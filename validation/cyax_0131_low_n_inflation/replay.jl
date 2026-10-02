#!/usr/bin/env julia

"""Bounded CYAX-0131 low-N calibration and author-trajectory replay."""

using CYAxiverse
using LinearAlgebra
using Random
using SHA

include(joinpath(@__DIR__, "..", "..", "scripts",
    "inflation_refinement_common.jl"))

const BENCHMARK = CYAxiverse.paper_benchmarks
const AUTHOR = BENCHMARK.author_inflation
const ROW2_PHASES = [0.0, 0.04, zeros(8)...]
const N8_DELTA_K_REFERENCE = 1.5320548620798324e-3

_sha256(path) = bytes2hex(sha256(read(path)))

function _n5_stationary_scan(model, k::BigFloat, phase::BigFloat;
        intervals::Int=10_000, bisection_iterations::Int=256)
    period = BigFloat(2) * BigFloat(π)
    nodes = range(zero(BigFloat), period; length=intervals + 1)
    values = [model.n5_reduced_phase_gradient(theta, k, phase) for theta in nodes]
    roots = BigFloat[]
    for index in 1:intervals
        left, right = nodes[index], nodes[index + 1]
        fleft, fright = values[index], values[index + 1]
        if iszero(fleft)
            push!(roots, left)
            continue
        end
        signbit(fleft) == signbit(fright) && continue
        for _ in 1:bisection_iterations
            midpoint = (left + right) / 2
            fmid = model.n5_reduced_phase_gradient(midpoint, k, phase)
            if iszero(fmid)
                left = midpoint
                right = midpoint
                break
            elseif signbit(fleft) == signbit(fmid)
                left, fleft = midpoint, fmid
            else
                right = midpoint
            end
        end
        root = mod((left + right) / 2, period)
        if all(min(abs(root - old), period - abs(root - old)) > big"1e-60"
                for old in roots)
            push!(roots, root)
        end
    end
    sort!(roots)
    (; roots, intervals, nodes=length(nodes), bisection_iterations)
end

function main()
    repo_root = dirname(dirname(@__DIR__))
    project_hash = _sha256(joinpath(repo_root,
        "validation", "p0_numerical_equivalence", "environment", "Project.toml"))
    manifest_hash = _sha256(joinpath(repo_root,
        "validation", "p0_numerical_equivalence", "environment", "Manifest.toml"))
    source_path = pathof(CYAxiverse)
    source_relative = relpath(source_path, repo_root)
    println((record=:runtime, julia=string(VERSION), source_path=source_relative,
        package_source_sha256=_sha256(source_path),
        p0_project_sha256=project_hash, p0_manifest_sha256=manifest_hash,
        environment_profile="candidate active project + existing P0 pinned environment + stdlib"))

    zero = AUTHOR.n8_degenerate_point()
    zero_model = BENCHMARK.n8_potential(k=zero.k; trajectory=true)
    @assert zero.converged
    @assert isapprox(zero.k, 0.674506370003365; atol=1e-15)
    @assert zero.gradient_residual < 1e-10
    @assert zero.null_residual < 1e-10
    @assert size(zero_model.Q) == (8, 10)
    @assert zero_model.phases == zeros(10)
    println((record=:n8_zero_phase_author10, route=:author_inflation_n8_degenerate_point,
        row_count=size(zero_model.Q, 2), phases=zero_model.phases,
        coordinate_convention=:raw_radians,
        metric_convention=:rounded_reconstructed_author_metric,
        scale_category=:author_model_catastrophe_calibration,
        k=zero.k, theta=zero.theta, gradient_residual=zero.gradient_residual,
        null_residual=zero.null_residual, converged=zero.converged))

    shifted = AUTHOR.n8_row2_phase_catastrophe()
    @assert shifted.status == :completed && shifted.converged
    @assert length(shifted.phase_path) == 400
    @assert all(step.converged for step in shifted.phase_path)
    @assert shifted.phase_vector == ROW2_PHASES
    @assert shifted.refined.gradient_residual < 1e-10
    @assert shifted.refined.null_residual < 1e-10
    @assert shifted.refined.converged
    @assert length(shifted.refined.eigenvalues) == 8
    @assert count(>(0), shifted.refined.eigenvalues) == 7
    @assert shifted.sign_change
    @assert length(shifted.stationary_path) == 12
    @assert all(entry.point.converged for entry in shifted.stationary_path)
    @assert shifted.refined.k > shifted.bracket.lower_k
    @assert shifted.refined.k < shifted.bracket.upper_k
    lower = shifted.stationary_path[findfirst(
        entry -> entry.offset == -1, shifted.stationary_path)].point
    upper = shifted.stationary_path[findfirst(
        entry -> entry.offset == 1, shifted.stationary_path)].point
    println((record=:n8_row2_phase_catastrophe, route=:author_inflation_n8_row2_phase_catastrophe,
        model=shifted.model, row_count=shifted.row_count,
        phase_convention=shifted.phase_convention, phase_vector=shifted.phase_vector,
        continuation=(attempted=length(shifted.phase_path),
            converged=count(step -> step.converged, shifted.phase_path),
            failed=count(step -> !step.converged, shifted.phase_path),
            phase_increment=0.04 / 400),
        float_location=(k=shifted.catastrophe.k, theta=shifted.catastrophe.theta,
            gradient_residual=shifted.catastrophe.gradient_residual,
            null_residual=shifted.catastrophe.null_residual),
        refined=(precision_bits=shifted.refined.solver.precision_bits,
            theta=shifted.refined.theta, null_vector=shifted.refined.null_vector,
            k=shifted.refined.k,
            gradient_residual=shifted.refined.gradient_residual,
            null_residual=shifted.refined.null_residual,
            normalized_null_residual=shifted.refined.normalized_null_residual,
            transverse_eigenvalues=shifted.refined.eigenvalues[2:end]),
        branch=(attempted=length(shifted.stationary_path),
            converged=count(entry -> entry.point.converged, shifted.stationary_path),
            offsets=[entry.offset for entry in shifted.stationary_path],
            lower=(k=lower.k, theta=lower.theta,
                gradient=lower.gradient_residual,
                min_eigenvalue=lower.minimum_hessian_eigenvalue),
            upper=(k=upper.k, theta=upper.theta,
                gradient=upper.gradient_residual,
                min_eigenvalue=upper.minimum_hessian_eigenvalue)),
        bracket=shifted.bracket, sign_change=shifted.sign_change,
        scale_category=shifted.scale_category))

    setprecision(BigFloat, 256) do
        fold = AUTHOR.n5_reduced_phase_fold(second_phase=BigFloat(π) / 4,
            theta0=big"2.1", k0=big"1.03", ftol=big"1e-70",
            xtol=big"1e-70", max_iterations=1_000)
        @assert fold.converged
        @assert fold.phase_assignment == :second_cosine
        @assert fold.second_phase == BigFloat(π) / 4
        @assert abs(fold.gradient) <= big"1e-9"
        @assert abs(fold.hessian) <= big"1e-10"
        @assert fold.residual <= big"1e-70"
        lower_k = fold.k - big"1e-4"
        upper_k = fold.k + big"1e-4"
        lower_roots = _n5_stationary_scan(AUTHOR, lower_k,
            fold.second_phase)
        upper_roots = _n5_stationary_scan(AUTHOR, upper_k,
            fold.second_phase)
        lower_local = filter(theta -> abs(theta - fold.theta) < big"0.1",
            lower_roots.roots)
        upper_local = filter(theta -> abs(theta - fold.theta) < big"0.1",
            upper_roots.roots)
        @assert length(lower_roots.roots) == 4
        @assert length(upper_roots.roots) == 2
        @assert length(lower_local) == 2
        @assert isempty(upper_local)
        lower_points = [AUTHOR.n5_reduced_phase_critical_point(lower_k, theta;
            second_phase=fold.second_phase, tolerance=big"1e-70")
            for theta in lower_local]
        @assert all(point.converged for point in lower_points)
        @assert all(abs(point.gradient) <= big"1e-9" for point in lower_points)
        @assert lower_points[1].hessian * lower_points[2].hessian < 0
        print((record=:n5_pi_over_four_reduced_fold,
            model=fold.model, phase_assignment=fold.phase_assignment,
            second_phase=fold.second_phase, precision_bits=fold.solver.precision_bits,
            solver=(method=fold.solver.method,
                iterations=fold.solver.iterations, ftol=fold.solver.ftol,
                xtol=fold.solver.xtol,
                max_iterations=fold.solver.max_iterations),
            fold=(theta=fold.theta, k=fold.k, ratio=fold.ratio,
                gradient=fold.gradient, hessian=fold.hessian,
                residual=fold.residual),
            lower_side=(k=lower_k, intervals=lower_roots.intervals,
                nodes=lower_roots.nodes, sign_change_roots=lower_roots.roots,
                local_roots=[(theta=point.theta, gradient=point.gradient,
                    hessian=point.hessian) for point in lower_points]),
            upper_side=(k=upper_k, intervals=upper_roots.intervals,
                nodes=upper_roots.nodes, sign_change_roots=upper_roots.roots,
                local_roots=upper_local),
            full_eight_row_phase_mapping=:NOT_VERIFIABLE))
    end
    println()

    Random.seed!(131)
    p96_model = BENCHMARK.n8_potential(k=0.68)
    @assert size(p96_model.Q, 2) == 12
    p96 = BENCHMARK.n8_pseudo_arclength_continuation(zeros(8), 0.68;
        n_steps=6, ds=1e-5, tolerance=1e-10, k_bounds=(0.67, 0.69))
    @assert length(p96.steps) == 7
    @assert all(step.converged for step in p96.steps)
    @assert all(step.gradient_residual <= 1e-10 for step in p96.steps)
    println((record=:p96_table1_crosscheck_only,
        route=:paper_benchmarks_n8_pseudo_arclength_continuation,
        source_rows=size(p96_model.Q, 2), phase_vector=zeros(12),
        initial_point=zeros(8), initial_k=0.68,
        configured_steps=6, retained_steps=length(p96.steps),
        converged=count(step -> step.converged, p96.steps),
        final_k=last(p96.steps).k,
        gradient_residuals=[step.gradient_residual for step in p96.steps],
        status=p96.status, catastrophe_bracket=p96.catastrophe_bracket,
        author10_gate_contribution=:none))

    delta = N8_DELTA_K_REFERENCE
    config = inflation_refinement_config(precision_bits=100,
        max_time=1e6, scan_step=5, max_step=100, initial_step=1e-5,
        sample_count=20, maxiters=10^8,
        critical_k=shifted.refined.k, phases=shifted.phase_vector,
        basis_theta=shifted.refined.theta, measurement_scope=:cold)
    candidate = inflation_refinement_candidate(
        "n8-row2-author-reference-delta"; delta_k=delta,
        screening=(status=:reference_anchor, measurement_scope=:cold))
    refined = refine_inflation_candidate(candidate; config)
    summary = refined.summary
    println((record=:n8_physical_author_trajectory_gate,
        route=summary.model_route, model=summary.model, row_count=summary.row_count,
        phase_vector=summary.phases, critical_k=summary.critical_k,
        delta_k=summary.delta_k, physical_k=summary.physical_k,
        scale_category=summary.scale_category,
        refinement_status=summary.refinement_status,
        entered_slow_roll=summary.entered_slow_roll,
        accepted_steps=summary.accepted_steps,
        rejected_steps=summary.rejected_steps,
        solver_method=summary.solver_method,
        solver_retcode=summary.solver_retcode,
        precision_bits=summary.precision_bits, reltol=summary.reltol,
        abstol=summary.abstol, maxiters=summary.maxiters,
        max_time=summary.max_time, scan_step=summary.scan_step,
        max_step=summary.max_step, initial_step=summary.initial_step,
        end_event=summary.end_event, terminated=summary.terminated,
        entry_n=summary.entry_n, end_n=summary.end_n,
        slow_roll_efolds=summary.slow_roll_efolds,
        measurement_scope=summary.measurement_scope,
        wall_seconds=summary.wall_seconds,
        allocated_bytes=summary.allocated_bytes,
        output_bytes=summary.output_bytes))
    eligible = summary.refinement_status === :completed &&
        summary.entered_slow_roll && summary.accepted_steps > 0
    if !eligible
        println((record=:stop, reason=:trajectory_eligibility_failed,
            refinement_status=summary.refinement_status,
            entered_slow_roll=summary.entered_slow_roll,
            accepted_steps=summary.accepted_steps,
            error=summary.error))
        exit(2)
    end
    @assert length(refined.trajectory.samples) == 20
    println((record=:n8_physical_author_trajectory_diagnostics,
        ne_definition=:slow_roll_efolds,
        ne_units=:e_folds, ne=summary.slow_roll_efolds,
        ne_window=(summary.entry_n, summary.end_n),
        reference_anchor=60.0,
        reference_discrepancy=summary.slow_roll_efolds - 60.0,
        sample_count=length(refined.trajectory.samples),
        sample_selection=:all_returned_samples_no_pivot))
    for sample_index in eachindex(refined.trajectory.samples)
        diagnostic = inflation_refinement_author_diagnostics(refined;
            sample_index)
        println((record=:n8_sample, model=diagnostic.model,
            row_count=diagnostic.row_count,
            sample_index=diagnostic.sample_index,
            sample_n=diagnostic.sample_n, sample_n_units=diagnostic.sample_n_units,
            n_s=diagnostic.n_s, n_s_units=diagnostic.n_s_units,
            paper_delta_H=diagnostic.paper_delta_H,
            paper_delta_H_units=diagnostic.paper_delta_H_units,
            scalar_amplitude_convention=diagnostic.scalar_amplitude_convention,
            cumulative_turning=diagnostic.cumulative_turning,
            cumulative_turning_units=diagnostic.cumulative_turning_units,
            coordinate_convention=diagnostic.coordinate_convention,
            metric_convention=diagnostic.metric_convention,
            scale_category=diagnostic.scale_category,
            physical_k_reestablished=diagnostic.physical_k_reestablished,
            pivot_interpretation=diagnostic.pivot_interpretation,
            observational_acceptance=diagnostic.observational_acceptance))
    end
    println((record=:bounded_execution_stop, reason=:configured_slice_complete,
        n8_phase_steps=400, n8_branch_point_count=12,
        n5_scan_intervals_per_side=10_000, p96_steps=6,
        trajectory_candidates=1, samples=20,
        observational_status=(new_As_conversion=:NOT_REACHED,
            new_As_acceptance=:NOT_REACHED, new_ns_acceptance=:NOT_REACHED,
            tensor_to_scalar_r=:NOT_REACHED,
            full_physical_n5_trajectory=:NOT_REACHED,
            issue131_observational_stretch_goal=:NOT_REACHED),
        n5_full_eight_row_phase_mapping=:NOT_VERIFIABLE,
        claim_boundary=:fixed_saxion_effective_theory_only,
        population_prevalence=:NOT_ESTABLISHED,
        dynamical_saxion_stabilization=:NOT_ESTABLISHED,
        full_ks_claim=:NOT_ESTABLISHED))
end

main()
