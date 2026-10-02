"""Shared script-level measurements for the inflation scan stages."""

const INFLATION_DIAGNOSTIC_SCHEMA_VERSION = "1"
const _INFLATION_MEASUREMENT_SCOPES = (:cold, :warm, :unspecified)

function _inflation_validate_measurement_scope(scope::Symbol)
    scope in _INFLATION_MEASUREMENT_SCOPES ||
        throw(ArgumentError("measurement_scope must be :cold, :warm, or :unspecified"))
    scope
end

"""Measure one stage without retaining its result in the diagnostic record."""
function inflation_stage_measure(f; measurement_scope::Symbol=:unspecified,
        capture_errors::Bool=false)
    scope = _inflation_validate_measurement_scope(measurement_scope)
    GC.gc(false)
    started = time_ns()
    try
        measured = @timed f()
        (; value=measured.value, status=:completed, error="",
           diagnostic_schema_version=INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
           measurement_scope=scope, seconds=measured.time, bytes=measured.bytes,
           output_bytes=Base.summarysize(measured.value))
    catch error
        capture_errors || rethrow()
        (; value=nothing, status=:failed, error=sprint(showerror, error),
           diagnostic_schema_version=INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
           measurement_scope=scope,
           seconds=(time_ns() - started) / 1e9, bytes=0, output_bytes=0)
    end
end

function _inflation_diagnostic_property(record, field::Symbol, default=nothing)
    record !== nothing && hasproperty(record, field) ?
        getproperty(record, field) : default
end

"""Flatten screening and refinement results into one candidate diagnostic row."""
function inflation_refinement_diagnostic_row(candidate, refined;
        serialization=nothing)
    screening = candidate.screening
    summary = refined.summary
    (; diagnostic_schema_version=INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
       candidate_id=candidate.candidate_id, model=candidate.model,
       screen_status=_inflation_diagnostic_property(screening, :status,
           candidate.accepted ? :accepted : :rejected),
       screen_accepted=candidate.accepted,
       screen_measurement_scope=_inflation_diagnostic_property(
           screening, :measurement_scope, :unspecified),
       screen_value=_inflation_diagnostic_property(screening, :value),
       screen_epsilon=_inflation_diagnostic_property(screening, :epsilon),
       screen_min_eta=_inflation_diagnostic_property(screening, :min_eta),
       screen_negative_modes=_inflation_diagnostic_property(
           screening, :negative_modes),
       screen_wall_seconds=_inflation_diagnostic_property(
           screening, :wall_seconds, 0.0),
       screen_allocated_bytes=_inflation_diagnostic_property(
           screening, :allocated_bytes, 0),
       screen_output_bytes=_inflation_diagnostic_property(
           screening, :output_bytes, 0),
       refinement_status=summary.refinement_status,
       refinement_error=summary.error,
       refinement_measurement_status=summary.measurement_status,
       refinement_measurement_scope=summary.measurement_scope,
       refinement_precision_bits=summary.precision_bits,
       refinement_model_route=summary.model_route,
       refinement_row_count=summary.row_count,
       refinement_phase_convention=summary.phase_convention,
       refinement_coordinate_convention=summary.coordinate_convention,
       refinement_metric_convention=summary.metric_convention,
       refinement_scale_category=summary.scale_category,
       refinement_critical_k=summary.critical_k,
       refinement_physical_k=summary.physical_k,
       refinement_phases=summary.phases,
       refinement_solver_method=summary.solver_method,
       refinement_solver_retcode=summary.solver_retcode,
       refinement_reltol=summary.reltol,
       refinement_abstol=summary.abstol,
       refinement_maxiters=summary.maxiters,
       refinement_event_policy=summary.event_policy,
       refinement_entered_slow_roll=summary.entered_slow_roll,
       refinement_end_event=summary.end_event,
       refinement_terminated=summary.terminated,
       refinement_entry_n=summary.entry_n,
       refinement_end_n=summary.end_n,
       refinement_efolds=summary.efolds,
       refinement_slow_roll_efolds=summary.slow_roll_efolds,
       refinement_censored_slow_roll_duration=summary.censored_slow_roll_duration,
       refinement_ne_definition=summary.ne_definition,
       refinement_ne_window=summary.ne_window,
       refinement_accepted_steps=summary.accepted_steps,
       refinement_rejected_steps=summary.rejected_steps,
       refinement_rhs_evaluations=summary.rhs_evaluations,
       refinement_jacobian_evaluations=summary.jacobian_evaluations,
       refinement_wall_seconds=summary.wall_seconds,
       refinement_allocated_bytes=summary.allocated_bytes,
       refinement_output_bytes=summary.output_bytes,
       serialization_status=serialization === nothing ? :not_measured :
           serialization.status,
       serialization_measurement_scope=serialization === nothing ? :unspecified :
           serialization.measurement_scope,
       serialization_wall_seconds=serialization === nothing ? 0.0 :
           serialization.seconds,
       serialization_allocated_bytes=serialization === nothing ? 0 :
           serialization.bytes,
       serialization_output_bytes=serialization === nothing ? 0 :
           serialization.output_bytes)
end

"""
Return explicitly selected diagnostics from the N8 author trajectory.

`N_e` is the selected slow-roll window duration. Sample diagnostics always
carry the exact returned sample index and `samples[index].n`; selecting a
sample does not define a pivot or an observational acceptance window.
"""
function _inflation_require_author_witness(witness, trajectory; summary=nothing)
    witness === nothing &&
        throw(ArgumentError("physical diagnostics require a same-model catastrophe witness"))
    witness.status === :completed && witness.converged && witness.sign_change ||
        throw(ArgumentError("catastrophe witness did not pass its source gates"))
    witness.model === :n8_author_trajectory_10_row && witness.row_count == 10 ||
        throw(ArgumentError("catastrophe witness is not the ten-row N8 author model"))
    witness.phase_convention === :additive_argument_radians ||
        throw(ArgumentError("catastrophe witness phase convention does not match"))
    witness.coordinate_convention === :raw_radians ||
        throw(ArgumentError("catastrophe witness coordinates do not match"))
    witness.metric_basis === :canonical_hessian_from_reconstructed_author_metric ||
        throw(ArgumentError("catastrophe witness metric or basis does not match"))
    trajectory.model === witness.model && trajectory.scale_category === :physical_author_radial_k ||
        throw(ArgumentError("trajectory does not use the witnessed author model and physical scale"))
    trajectory.coordinate_convention === witness.coordinate_convention &&
        trajectory.metric_convention === :rounded_reconstructed_author_metric ||
        throw(ArgumentError("trajectory coordinates or metric do not match the catastrophe witness"))
    trajectory.basis === :canonical_hessian && trajectory.basis_k == trajectory.k ||
        throw(ArgumentError("trajectory basis does not match the catastrophe witness"))
    trajectory.critical_theta == trajectory.basis_theta ||
        throw(ArgumentError("trajectory critical coordinates differ from its fixed basis coordinates"))
    precision = trajectory.precision_bits
    witness_values = setprecision(BigFloat, precision) do
        (phases=BigFloat.(witness.phase_vector),
         refined_phases=BigFloat.(witness.refined_phase_vector),
         theta=BigFloat.(witness.refined.theta),
         critical_k=BigFloat(witness.refined.k))
    end
    trajectory.phases == witness_values.phases == witness_values.refined_phases ||
        throw(ArgumentError("trajectory phase encoding differs from the catastrophe witness"))
    trajectory.basis_theta == witness_values.theta ||
        throw(ArgumentError("trajectory coordinates differ from the refined catastrophe witness"))
    trajectory.critical_k == witness_values.critical_k ||
        throw(ArgumentError("trajectory critical k differs from the refined catastrophe witness"))
    if summary !== nothing
        summary.phases == witness.phase_vector ||
            throw(ArgumentError("refinement phase encoding differs from the catastrophe witness"))
        summary.basis === :canonical_hessian &&
            summary.metric_convention === :rounded_reconstructed_author_metric &&
            summary.coordinate_convention === :raw_radians ||
            throw(ArgumentError("refinement coordinate, metric, or basis differs from the witness"))
        summary_theta = setprecision(BigFloat, precision) do
            BigFloat.(summary.basis_theta)
        end
        summary_theta == witness_values.theta ||
            throw(ArgumentError("refinement coordinates differ from the refined catastrophe witness"))
        summary_k = setprecision(BigFloat, precision) do
            BigFloat(summary.critical_k)
        end
        summary_k == witness_values.critical_k ||
            throw(ArgumentError("refinement critical k differs from the refined catastrophe witness"))
    end
    nothing
end

function inflation_author_trajectory_diagnostics(trajectory;
        sample_index::Int, physical_witness)
    trajectory.model === :n8_author_trajectory_10_row ||
        throw(ArgumentError("diagnostics require the ten-row N8 author trajectory"))
    trajectory.scale_category === :physical_author_radial_k ||
        throw(ArgumentError("physical k must be established before diagnostics"))
    trajectory.entered_slow_roll ||
        throw(ArgumentError("trajectory did not enter a slow-roll window"))
    trajectory.terminated && trajectory.end_event !== :tmax ||
        throw(ArgumentError("trajectory has no completed finite-exit slow-roll window"))
    hasproperty(trajectory, :solver) ||
        throw(ArgumentError("trajectory does not contain solver status"))
    success = CYAxiverse.paper_benchmarks.author_inflation.OrdinaryDiffEq.ReturnCode.Success
    trajectory.solver.retcode == success ||
        throw(ArgumentError("trajectory solver did not complete successfully"))
    trajectory.solver.accepted_steps > 0 ||
        throw(ArgumentError("trajectory has no accepted solver steps"))
    _inflation_require_author_witness(physical_witness, trajectory)
    expected_physical_k = setprecision(BigFloat, trajectory.precision_bits) do
        BigFloat(trajectory.critical_k) + BigFloat(trajectory.delta_k)
    end
    trajectory.k == expected_physical_k ||
        throw(ArgumentError("trajectory physical k does not match critical_k + delta_k"))
    hasproperty(trajectory, :entry_n) && hasproperty(trajectory, :end_n) ||
        throw(ArgumentError("trajectory does not contain a completed slow-roll window"))
    1 <= sample_index <= length(trajectory.samples) ||
        throw(BoundsError(trajectory.samples, sample_index))
    sample = trajectory.samples[sample_index]
    observables = CYAxiverse.paper_benchmarks.trajectory_observables(
        trajectory; sample_index)
    cumulative_turning = CYAxiverse.paper_benchmarks.cumulative_turning(
        trajectory.samples[1:sample_index])
    (; model=:n8_author_trajectory_10_row, row_count=10,
       source_route=:author_inflation_n8_author_trajectory,
       phase_convention=:additive_argument_radians,
       phases=copy(trajectory.phases),
       coordinate_convention=:raw_radians,
       metric_convention=:rounded_reconstructed_author_metric,
       scale_category=trajectory.scale_category,
       critical_k=trajectory.critical_k, physical_k=trajectory.k,
       ne_definition=:slow_roll_efolds,
       ne_window=(trajectory.entry_n, trajectory.end_n),
       ne_units=:e_folds,
       ne=trajectory.slow_roll_efolds,
       sample_n_units=:e_folds,
       n_s_units=:dimensionless,
       sample_index, sample_n=sample.n,
       n_s=observables.n_s, paper_delta_H=observables.delta_H,
       paper_delta_H_units=:dimensionless,
       scalar_amplitude_convention=observables.scalar_amplitude_convention,
       cumulative_turning,
       cumulative_turning_units=:radians,
       cumulative_turning_sample_index=sample_index,
       cumulative_turning_sample_n=sample.n,
       physical_k_reestablished=true,
       pivot_interpretation=:none,
       observational_acceptance=:not_evaluated)
end

"""Apply the refinement eligibility gate before author-trajectory reporting."""
function inflation_refinement_author_diagnostics(refined;
        sample_index::Int, physical_witness)
    summary = refined.summary
    summary.refinement_status === :completed ||
        throw(ArgumentError("refinement status is not completed"))
    summary.entered_slow_roll ||
        throw(ArgumentError("refinement did not enter slow roll"))
    summary.terminated && summary.end_event !== :tmax ||
        throw(ArgumentError("refinement has no completed finite-exit slow-roll window"))
    summary.accepted_steps > 0 ||
        throw(ArgumentError("refinement has no accepted solver steps"))
    summary.model_route === :author_inflation_n8_author_trajectory ||
        throw(ArgumentError("refinement did not use the N8 author trajectory route"))
    summary.scale_category === :physical_author_radial_k ||
        throw(ArgumentError("refinement k is not an author-model physical radial scale"))
    refined.trajectory === nothing &&
        throw(ArgumentError("completed refinement has no trajectory"))
    expected_physical_k = setprecision(BigFloat, summary.precision_bits) do
        BigFloat(summary.critical_k) + BigFloat(summary.delta_k)
    end
    expected_physical_k == refined.trajectory.k ||
        throw(ArgumentError("refinement physical k does not match critical_k + delta_k"))
    summary.physical_k == refined.trajectory.k ||
        throw(ArgumentError("refinement physical k differs from author trajectory k"))
    _inflation_require_author_witness(
        physical_witness, refined.trajectory; summary)
    diagnostics = inflation_author_trajectory_diagnostics(
        refined.trajectory; sample_index, physical_witness)
    merge(diagnostics, (; refinement_status=summary.refinement_status,
        accepted_steps=summary.accepted_steps,
        solver_retcode=summary.solver_retcode))
end

function _inflation_diagnostic_csv_escape(value)
    value === nothing && return ""
    text = replace(string(value), '"' => "\"\"")
    occursin(r"[,\"\n\r]", text) ? string('"', text, '"') : text
end

"""Serialize a flat diagnostic row; file writes can be measured separately."""
function inflation_diagnostic_csv_line(row; header::Bool=false)
    names = propertynames(row)
    values = (_inflation_diagnostic_csv_escape(getproperty(row, name))
        for name in names)
    header ? join(string.(names), ',') : join(values, ',')
end

"""Append one diagnostic row and measure formatting plus the file write."""
function inflation_append_diagnostic_row(path::AbstractString, row;
        measurement_scope::Symbol=:unspecified, header::Bool=false)
    path = abspath(expanduser(path))
    mkpath(dirname(path))
    inflation_stage_measure(
        () -> begin
            line = inflation_diagnostic_csv_line(row)
            open(path, "a") do io
                header && println(io,
                    inflation_diagnostic_csv_line(row; header=true))
                println(io, line)
                flush(io)
            end
            line
        end; measurement_scope)
end
