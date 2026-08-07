"""Script-level Stage 3 refinement boundary for the poly-102 benchmark.

The scan-prep classifier is geometry-generic, but the available arbitrary-
precision trajectory solver is currently the validated n8/poly-102 model. This
adapter makes that boundary explicit without pretending that the solver accepts
an arbitrary Q/L/K geometry yet.
"""

if !isdefined(@__MODULE__, :INFLATION_DIAGNOSTIC_SCHEMA_VERSION)
    include(joinpath(@__DIR__, "inflation_diagnostics_common.jl"))
end

using CYAxiverse
using OrdinaryDiffEq

const Poly102RefinementModel = CYAxiverse.axion_benchmarks.poly102_inflation
const N8_POLY102_REFINEMENT_MODEL = :n8_poly102

"""Build a validated, serializable configuration for one refinement call."""
function inflation_refinement_config(; precision_bits::Int=100,
        max_time::Real=1e6, scan_step::Real=5.0, max_step::Real=100.0,
        initial_step::Real=1e-5, sample_count::Int=20,
        reltol=nothing, abstol=nothing, maxiters::Int=10^8,
        displacement::Real=1e-8, displacement_sign::Real=-1.0,
        basis::Symbol=:canonical_hessian,
        measurement_scope::Symbol=:unspecified,
        basis_theta::AbstractVector{<:Real}=Poly102RefinementModel.N8_BEST_X)
    precision_bits >= 64 ||
        throw(ArgumentError("precision_bits must be at least 64"))
    max_time > 0 || throw(ArgumentError("max_time must be positive"))
    scan_step > 0 || throw(ArgumentError("scan_step must be positive"))
    max_step > 0 || throw(ArgumentError("max_step must be positive"))
    initial_step > 0 || throw(ArgumentError("initial_step must be positive"))
    sample_count > 0 || throw(ArgumentError("sample_count must be positive"))
    maxiters > 0 || throw(ArgumentError("maxiters must be positive"))
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    displacement_sign != 0 ||
        throw(ArgumentError("displacement_sign must be nonzero"))
    basis === :canonical_hessian || basis === :mass_eigenbasis ||
        throw(ArgumentError("unsupported refinement basis: $basis"))
    _inflation_validate_measurement_scope(measurement_scope)
    all(isfinite, basis_theta) && length(basis_theta) == 8 ||
        throw(DimensionMismatch("poly-102 refinement requires eight finite basis coordinates"))
    reltol === nothing || reltol > 0 || throw(ArgumentError("reltol must be positive"))
    abstol === nothing || abstol > 0 || throw(ArgumentError("abstol must be positive"))
    (; precision_bits, max_time=Float64(max_time), scan_step=Float64(scan_step),
       max_step=Float64(max_step), initial_step=Float64(initial_step),
       sample_count, reltol=reltol === nothing ? nothing : Float64(reltol),
       abstol=abstol === nothing ? nothing : Float64(abstol), maxiters,
       displacement=Float64(displacement),
       displacement_sign=Float64(displacement_sign), basis,
       measurement_scope,
       basis_theta=Float64.(basis_theta),
       solver_method=:Rodas5P, event_policy=:final_finite_exit)
end

"""Create the candidate record consumed by `refine_inflation_candidate`."""
function inflation_refinement_candidate(candidate_id::AbstractString;
        model::Symbol=N8_POLY102_REFINEMENT_MODEL, delta_k::Real,
        accepted::Bool=true, screening=NamedTuple())
    isfinite(delta_k) && delta_k > 0 ||
        throw(ArgumentError("delta_k must be positive and finite"))
    (; candidate_id=String(candidate_id), model,
       delta_k=Float64(delta_k), accepted, screening)
end

function _refinement_summary(candidate, config, status::Symbol;
        error_message="", measured=nothing, trajectory=nothing)
    solver = trajectory === nothing ? nothing : trajectory.solver
    (; candidate_id=candidate.candidate_id, model=candidate.model,
       delta_k=candidate.delta_k, screen_accepted=candidate.accepted,
       refinement_status=status, error=error_message,
       measurement_status=measured === nothing ? :not_measured : measured.status,
       diagnostic_schema_version=INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
       measurement_scope=measured === nothing ? config.measurement_scope :
           measured.measurement_scope,
       precision_bits=config.precision_bits,
       solver_method=config.solver_method, event_policy=config.event_policy,
       solver_retcode=solver === nothing ? nothing : string(solver.retcode),
       reltol=config.reltol, abstol=config.abstol,
       max_time=config.max_time, scan_step=config.scan_step,
       max_step=config.max_step, initial_step=config.initial_step,
       displacement=config.displacement,
       displacement_sign=config.displacement_sign, basis=config.basis,
       entered_slow_roll=trajectory === nothing ? false : trajectory.entered_slow_roll,
       end_event=trajectory === nothing ? :not_run : trajectory.end_event,
       terminated=trajectory === nothing ? false : trajectory.terminated,
       efolds=trajectory === nothing ? nothing : trajectory.efolds,
       slow_roll_efolds=trajectory === nothing ? nothing : trajectory.slow_roll_efolds,
       accepted_steps=solver === nothing ? 0 : solver.accepted_steps,
       rejected_steps=solver === nothing ? 0 : solver.rejected_steps,
       rhs_evaluations=solver === nothing ? 0 : solver.rhs_evaluations,
       jacobian_evaluations=solver === nothing ? 0 : solver.jacobian_evaluations,
       wall_seconds=measured === nothing ? 0.0 : measured.seconds,
       allocated_bytes=measured === nothing ? 0 : measured.bytes,
       output_bytes=measured === nothing ? 0 : measured.output_bytes)
end

function _refinement_solver_status(retcode)
    retcode == ReturnCode.Success && return (:completed, "")
    (:failed, "trajectory solver retcode: $retcode")
end

"""Run one eligible high-precision candidate and return summary plus trajectory."""
function refine_inflation_candidate(candidate;
        config=inflation_refinement_config())
    candidate.accepted || return (
        summary=_refinement_summary(candidate, config, :not_selected),
        trajectory=nothing)
    candidate.model === N8_POLY102_REFINEMENT_MODEL || return (
        summary=_refinement_summary(candidate, config, :unsupported_model),
        trajectory=nothing)

    measured = inflation_stage_measure(
        () -> Poly102RefinementModel.n8_physical_gradient_flow(
            candidate.delta_k; displacement=config.displacement,
            displacement_sign=config.displacement_sign,
            max_time=config.max_time, scan_step=config.scan_step,
            sample_count=config.sample_count, max_step=config.max_step,
            initial_step=config.initial_step, basis=config.basis,
            basis_theta=config.basis_theta, precision_bits=config.precision_bits,
            reltol=config.reltol, abstol=config.abstol,
            maxiters=config.maxiters);
        measurement_scope=config.measurement_scope, capture_errors=true)
    if measured.status === :failed
        return (; summary=_refinement_summary(candidate, config, :failed;
                   error_message=measured.error, measured), trajectory=nothing)
    end
    try
        trajectory = measured.value
        status, error_message = _refinement_solver_status(
            trajectory.solver.retcode)
        (; summary=_refinement_summary(candidate, config, status;
               error_message, measured, trajectory), trajectory)
    catch error
        (; summary=_refinement_summary(candidate, config, :failed;
               error_message=sprint(showerror, error)), trajectory=nothing)
    end
end
