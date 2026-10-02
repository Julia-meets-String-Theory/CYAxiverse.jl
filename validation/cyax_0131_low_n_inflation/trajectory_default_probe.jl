#!/usr/bin/env julia

"""Bounded diagnostics for the active 100-bit author trajectory route."""

using CYAxiverse
using SHA

const AUTHOR = CYAxiverse.paper_benchmarks.author_inflation
const DELTA_K = 1.5320548620798324e-3
const PROBE_MAX_TIME = 10.0
const PROBE_MAXITERS = 1_000
const PROBE_MAX_STEP = 100.0
const PROBE_INITIAL_STEP = 1e-5
const PROBE_SCAN_STEP = 5.0
const PROBE_SAMPLE_COUNT = 2

_sha256(path) = bytes2hex(sha256(read(path)))
_emit(record) = (println(record); flush(stdout); nothing)

function _record_probe(label, calibration)
    _emit((record=:trajectory_probe_started, label,
        precision_bits=100, tolerance_source=:author_default_for_precision,
        simulated_horizon=PROBE_MAX_TIME, maxiters=PROBE_MAXITERS))
    measured = @timed try
        AUTHOR.n8_author_trajectory(DELTA_K;
            critical_k=calibration.refined.k,
            phases=calibration.phase_vector,
            basis_theta=calibration.refined.theta,
            displacement=1e-8,
            displacement_sign=-1,
            method=:Rodas5P,
            precision_bits=100,
            max_time=PROBE_MAX_TIME,
            scan_step=PROBE_SCAN_STEP,
            max_step=PROBE_MAX_STEP,
            initial_step=PROBE_INITIAL_STEP,
            sample_count=PROBE_SAMPLE_COUNT,
            maxiters=PROBE_MAXITERS)
    catch error
        (; probe_exception=sprint(showerror, error), exception_type=string(typeof(error)))
    end

    result = measured.value
    if hasproperty(result, :solver)
        solver = result.solver
        retcode_success = solver.retcode == AUTHOR.OrdinaryDiffEq.ReturnCode.Success
        _emit((record=:trajectory_probe, label,
            precision_bits=result.precision_bits,
            tolerance_source=:author_default_for_precision,
            reltol=solver.reltol, abstol=solver.abstol,
            configured_tspan=(0.0, PROBE_MAX_TIME),
            tspan_completed=retcode_success,
            returned_simulation_time=retcode_success ? PROBE_MAX_TIME : nothing,
            simulation_time_limit=PROBE_MAX_TIME,
            simulation_time_field_exposed=false,
            maxiters=PROBE_MAXITERS,
            max_step=PROBE_MAX_STEP,
            initial_step=PROBE_INITIAL_STEP,
            scan_step=PROBE_SCAN_STEP,
            sample_count=PROBE_SAMPLE_COUNT,
            retcode=string(solver.retcode),
            accepted_steps=solver.accepted_steps,
            rejected_steps=solver.rejected_steps,
            rhs_evaluations=solver.rhs_evaluations,
            jacobian_evaluations=solver.jacobian_evaluations,
            entered_slow_roll=result.entered_slow_roll,
            end_event=result.end_event,
            efolds=result.efolds,
            slow_roll_efolds=result.slow_roll_efolds,
            returned_fields=propertynames(result),
            wall_seconds=measured.time,
            gc_seconds=measured.gctime,
            allocated_bytes=measured.bytes,
            returned_object_bytes=Base.summarysize(result)))
    else
        _emit((record=:trajectory_probe, label,
            precision_bits=100,
            tolerance_source=:author_default_for_precision,
            configured_tspan=(0.0, PROBE_MAX_TIME),
            simulation_time_limit=PROBE_MAX_TIME,
            simulation_time_field_exposed=false,
            maxiters=PROBE_MAXITERS,
            retcode=:exception_before_solver_record,
            accepted_steps=nothing, rejected_steps=nothing,
            rhs_evaluations=nothing, jacobian_evaluations=nothing,
            returned_simulation_time=nothing,
            error=result.probe_exception,
            exception_type=result.exception_type,
            wall_seconds=measured.time,
            gc_seconds=measured.gctime,
            allocated_bytes=measured.bytes))
    end
    nothing
end

function main()
    repo_root = dirname(dirname(@__DIR__))
    _emit((record=:probe_runtime,
        julia=string(VERSION),
        source_path=relpath(pathof(CYAxiverse), repo_root),
        trajectory_source_sha256=_sha256(joinpath(repo_root,
            "src", "paper_benchmarks", "poly102_inflation.jl")),
        launch_mode=:default_julia_compile,
        threads=Threads.nthreads(),
        max_time=PROBE_MAX_TIME,
        maxiters=PROBE_MAXITERS,
        save_everystep=true,
        dense=true))

    setprecision(BigFloat, 100) do
        reltol = BigFloat(10)^(-100 ÷ 2)
        abstol = BigFloat(10)^(-100 * 2 ÷ 3)
        epsilon_at_one = eps(BigFloat(1))
        _emit((record=:default_tolerance_scale,
            precision_bits=precision(BigFloat(1)),
            reltol=reltol,
            abstol=abstol,
            epsilon_at_one=epsilon_at_one,
            reltol_over_epsilon=reltol / epsilon_at_one,
            abstol_over_epsilon=abstol / epsilon_at_one,
            note=:epsilon_is_unit_scale_spacing_not_a_solver_acceptance_rule))
    end

    _emit((record=:calibration_started,
        route=:author_inflation_n8_row2_phase_catastrophe))
    calibration_time = @timed AUTHOR.n8_row2_phase_catastrophe()
    calibration = calibration_time.value
    _emit((record=:shared_calibration,
        status=calibration.status,
        model=calibration.model,
        phase_vector=calibration.phase_vector,
        critical_k=calibration.refined.k,
        basis_theta=calibration.refined.theta,
        precision_bits=calibration.refined.solver.precision_bits,
        wall_seconds=calibration_time.time,
        allocated_bytes=calibration_time.bytes))
    calibration.status === :completed || return

    _record_probe(:first_100_bit_default_tolerance, calibration)
    _record_probe(:repeat_100_bit_default_tolerance, calibration)
end

main()
