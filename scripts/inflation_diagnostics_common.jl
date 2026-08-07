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
       refinement_solver_method=summary.solver_method,
       refinement_solver_retcode=summary.solver_retcode,
       refinement_event_policy=summary.event_policy,
       refinement_entered_slow_roll=summary.entered_slow_roll,
       refinement_end_event=summary.end_event,
       refinement_terminated=summary.terminated,
       refinement_efolds=summary.efolds,
       refinement_slow_roll_efolds=summary.slow_roll_efolds,
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
