#!/usr/bin/env julia

"""Bounded scale-continuation and catastrophe diagnostic pilot.

Generic geometry rows may use the owner-selected homogeneous continuation only
after a same-scale physical-domain certificate passes. The author's
fixed-volume convention remains an explicit nonphysical comparison diagnostic,
and the old mathematical continuation remains available explicitly with
`scale_status=homotopy_only`.

The script is intentionally script-local.  It reuses the locked orientation,
leading-branch streamer, log-shifted derivatives, and screen conventions from
`inflation_scan_common.jl`, but does not modify the fixed-potential scan.
"""

using LinearAlgebra
using NLsolve
using Printf
using Statistics
using HDF5
using SHA

include(joinpath(@__DIR__, "inflation_scan_common.jl"))

const PILOT_SCHEMA_VERSION = "6"
const PILOT_DOMAIN_CERTIFICATE_VERSION = "physical-domain-certificate-3"
const PILOT_PHYSICAL_NORMALIZATION = "homogeneous_full_volume_k32"
const PILOT_PHYSICAL_UNITS = "M_s=M_Pl;k=dimensionless"
const PILOT_STORED_NUMERIC_TYPE = "Float64"
const PILOT_STORED_PRECISION_BITS = 53
const PILOT_TARGET_NUMERIC_TYPE = "Float64"
const PILOT_TARGET_PRECISION_BITS = 53
const PILOT_CONVERSION_TOLERANCE = "1e-12"
const PILOT_CONVERSION_POLICY_VERSION = "kinv-mixed-tolerance-v1"
const PILOT_KINV_CONVERSION_RULE =
    "max_absolute_error <= 1e-12 OR max_relative_error <= 1e-12"
const PILOT_AUTHOR_SOURCE_IDENTITY =
    "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CN_Axiverse_code/" *
    "ks_axiverse_python_collaborator/src/cytools_catastrophe_scan.py@sha256:" *
    "d820dd3e19d2833bac0691d74c2f99d2461c8eb0ef1620062f70d3daffd3bcf4"
const PILOT_HOMOTOPY_SOURCE_IDENTITY =
    "scripts/inflation_scale_continuation.jl::pilot_homotopy_scale"
const PILOT_GATE_STATUS_VALUES = (
    :passed, :failed, :not_established, :not_applicable, :missing_evidence,
    :numerical_failure, :out_of_model)
const PILOT_VIABILITY_STATUS_VALUES = (
    :not_applicable, :not_evaluated, :blocked_scaling_gate,
    :not_established, :not_candidate, :eligible_not_validated,
    :near_catastrophe_not_validated)
const PILOT_DEFAULT_SCALE_GRID = (0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10)
const PILOT_DEFAULT_REPORT = "/private/tmp/inflation-scale-continuation/report.csv"
const PILOT_DEFAULT_SHARDS = "/private/tmp/inflation-scale-continuation/shards"
const PILOT_DEFAULT_CERTIFICATE_DIR = ""
const PILOT_DEFAULT_CHECKPOINT_DIR = ""

const PILOT_COMMON_FIELDS = (
    :row_type, :schema_version, :run_id, :data_root, :geometry_path,
    :h11, :polytope, :frst, :reference_scale, :sampled_scale,
    :scale_grid, :scale_source, :scale_status, :volume_normalization,
    :domain_certificate_version, :domain_status, :domain_reason,
    :physical_scaling_gate_status, :physical_scaling_gate_reason,
    :physical_scaling_gate_provenance,
    :physical_control_gate_status, :physical_control_gate_reason,
    :physical_control_gate_provenance,
    :physical_viability_status, :physical_viability_reason,
    :fixed_point_status, :trajectory_status, :coverage_status,
    :moduli_status, :phase_convention, :units, :normalization,
    :source_identity, :precision_bits, :source_numeric_type,
    :source_precision_bits, :target_numeric_type, :target_precision_bits,
    :conversion_status, :conversion_error_bound, :conversion_tolerance,
    :conversion_comparison, :conversion_policy_version,
    :stored_reference_max_log10_error, :stored_reference_sign_mismatches,
    :leading_log_gap, :log_scale_span,
    :strong_hierarchy, :search_mode, :branch_coverage_status,
    :branch_estimate, :branch_count, :mask_count, :masks_visited,
    :masks_skipped, :lattice_copy_count, :lattice_copies_visited,
    :estimated_stage_allocated_bytes, :max_stage_allocated_bytes,
    :reference_uncorrected_screen_candidates, :branch_seed_index,
    :leading_negative_modes, :leading_seed_theta,
    :corrected_theta, :correction_status, :correction_residual,
    :correction_iterations, :branch_provenance_id, :branch_match_id,
    :matching_status, :value, :gradient_norm, :epsilon, :min_eta,
    :max_eta, :abs_min_eta, :hessian_min, :hessian_max,
    :zeroish_modes, :negative_modes, :positive_modes, :near_catastrophe,
    :catastrophe_bracket, :catastrophe_reason, :candidate_status,
    :candidate_reason, :seed_count, :corrected_count, :matched_count,
    :lost_count, :new_count, :duplicate_count, :correction_failed_count,
    :near_catastrophe_brackets, :screen_candidate_count,
    :corrected_candidate_count, :physical_eligible_count, :minima_count, :wall_seconds,
    :allocated_bytes, :output_bytes, :failure)

const PILOT_SUMMARY_FIELDS = PILOT_COMMON_FIELDS
const PILOT_BRANCH_FIELDS = PILOT_COMMON_FIELDS

"""Parse a comma-separated positive Float64 scale grid."""
function pilot_parse_scale_grid(value::AbstractString)
    values = Float64.(parse.(Float64, split(strip(value), ',')))
    isempty(values) && throw(ArgumentError("scale grid must not be empty"))
    all(isfinite, values) && all(>(0), values) ||
        throw(ArgumentError("scale grid values must be finite and positive"))
    sort!(unique(values))
end

"""Apply the legacy generic mathematical homotopy explicitly."""
function pilot_homotopy_scale(L::AbstractMatrix{<:Real}, scale::Real;
        scale_status::Symbol=:homotopy_only)
    scale_status == :homotopy_only ||
        throw(ArgumentError("physical generic scale transformation is unsupported: " *
            "geometry metadata does not document how L and K transform"))
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("homotopy scale must be finite and positive"))
    scaled = Matrix{Float64}(L)
    scaled[2, :] .= Float64(scale) .* scaled[2, :]
    all(isfinite, scaled) || throw(ArgumentError("scaled L is non-finite"))
    scaled
end

function _pilot_author_term_count(term_count::Int)
    base_count = (isqrt(8 * term_count + 1) - 1) ÷ 2
    base_count > 0 && base_count * (base_count + 1) ÷ 2 == term_count ||
        throw(ArgumentError("potential term count is not base-plus-pairwise: $term_count"))
    base_count
end

function _pilot_author_potential(Q::AbstractMatrix{<:Integer},
        tau::AbstractVector{T}, kinv::AbstractMatrix{T}, cy_volume::T) where
        {T<:AbstractFloat}
    h11, term_count = size(Q)
    length(tau) == h11 || throw(DimensionMismatch("tau and Q dimensions disagree"))
    size(kinv) == (h11, h11) || throw(DimensionMismatch("Kinv and Q dimensions disagree"))
    isfinite(cy_volume) && cy_volume > 0 ||
        throw(ArgumentError("CY volume must be finite and positive"))
    all(isfinite, tau) && all(>(0), tau) ||
        throw(ArgumentError("author divisor volumes must be finite and positive"))
    all(isfinite, kinv) || throw(ArgumentError("author Kinv contains non-finite values"))
    base_count = _pilot_author_term_count(term_count)
    expected_Q = zeros(Int, h11, term_count)
    expected_Q[:, 1:base_count] .= Q[:, 1:base_count]
    index = base_count + 1
    for i in 1:(base_count - 1), j in (i + 1):base_count
        expected_Q[:, index] .= @view(Q[:, j]) .- @view(Q[:, i])
        index += 1
    end
    expected_Q == Q || throw(ArgumentError(
        "stored charges do not have the author's leading-plus-difference ordering"))

    result = zeros(T, 2, term_count)
    prefactor = T(8) * T(π) / cy_volume^2
    log10e = log10(exp(one(T)))
    for column in 1:base_count
        charge = @view Q[:, column]
        exponent = -T(2) * T(π) * log10e * dot(charge, tau)
        coefficient = prefactor * dot(charge, tau)
        if coefficient == 0
            result[1, column] = zero(T)
            result[2, column] = -T(Inf)
        else
            result[1, column] = sign(coefficient)
            result[2, column] = log10(abs(coefficient)) + exponent
        end
    end
    index = base_count + 1
    for i in 1:(base_count - 1), j in (i + 1):base_count
        qi = @view Q[:, i]
        qj = @view Q[:, j]
        charge_sum = qi .+ qj
        exponent = -T(2) * T(π) * log10e * dot(charge_sum, tau)
        coefficient = prefactor *
            (T(π) * dot(qi, kinv * qj) + dot(charge_sum, tau))
        if coefficient == 0
            result[1, index] = zero(T)
            result[2, index] = -T(Inf)
        else
            result[1, index] = sign(coefficient)
            result[2, index] = log10(abs(coefficient)) + exponent
        end
        index += 1
    end
    result
end

function _pilot_reference_diagnostic(Q, L, tau::AbstractVector{T},
        kinv::AbstractMatrix{T}, cy_volume::T) where {T<:AbstractFloat}
    expected = _pilot_author_potential(Q, tau, kinv, cy_volume)
    size(L) == size(expected) || throw(DimensionMismatch("L dimensions disagree"))
    L_sign = sign.(L[1, :])
    expected_sign = sign.(expected[1, :])
    finite = isfinite.(L[2, :]) .& isfinite.(expected[2, :])
    compatible_nonfinite = all(finite .| (L[2, :] .== expected[2, :]))
    compatible_nonfinite || throw(ArgumentError(
        "stored potential has incompatible non-finite logs"))
    (; expected, max_log10_error=any(finite) ?
           maximum(abs.(L[2, finite] .- expected[2, finite])) : zero(T),
           sign_mismatches=count(index -> L_sign[index] != expected_sign[index],
           eachindex(L_sign)))
end

struct PilotPhysicalDomainError <: Exception
    certificate::NamedTuple
end

function Base.showerror(io::IO, error::PilotPhysicalDomainError)
    print(io, "physical-domain certificate ", error.certificate.status, ": ",
        error.certificate.domain_reason)
end

function _pilot_get_field(object, names::Tuple)
    for name in names
        hasproperty(object, name) && return true, getproperty(object, name)
    end
    false, nothing
end

function _pilot_nonempty_metadata(value)
    value === nothing && return false
    ismissing(value) && return false
    value isa AbstractString && begin
        text = lowercase(strip(value))
        return !isempty(text) && !(text in ("missing", "unknown", "not_recorded", "none"))
    end
    value isa Symbol && return !(value in (:missing, :unknown, :not_recorded))
    value isa AbstractArray && return !isempty(value)
    value isa AbstractDict && return !isempty(value)
    value isa Bool && return value
    true
end

"""Infer the numeric type and represented precision of one stored value."""
function _pilot_numeric_type_precision(value)
    numeric_type = value isa AbstractArray ? eltype(value) : typeof(value)
    if numeric_type === Float64
        return (; source_numeric_type=PILOT_STORED_NUMERIC_TYPE,
            source_precision_bits=PILOT_STORED_PRECISION_BITS)
    elseif numeric_type <: BigFloat
        sample = value isa AbstractArray && isempty(value) ? BigFloat(0) :
            value isa AbstractArray ? first(value) : value
        return (; source_numeric_type="BigFloat",
            source_precision_bits=precision(sample))
    elseif numeric_type <: AbstractFloat
        bits = try
            precision(zero(numeric_type))
        catch
            0
        end
        return (; source_numeric_type=string(numeric_type),
            source_precision_bits=Int(bits))
    end
    (; source_numeric_type=string(numeric_type), source_precision_bits=0)
end

"""Describe the source precision without claiming more than the arrays carry."""
function _pilot_source_numeric_provenance(values)
    entries = [_pilot_numeric_type_precision(value) for value in values]
    names = unique(String[entry.source_numeric_type for entry in entries])
    bits = [entry.source_precision_bits for entry in entries if
        entry.source_precision_bits > 0]
    source_numeric_type = length(names) == 1 ? only(names) :
        string("mixed[", join(names, ","), "]")
    (; source_numeric_type,
       source_precision_bits=isempty(bits) ? 0 : minimum(bits))
end

"""Convert one certified arbitrary-precision value and measure round-trip loss."""
function _pilot_convert_float64(value, label::AbstractString)
    target = value isa AbstractArray ? Array{Float64}(undef, size(value)) : nothing
    max_absolute_error = zero(BigFloat)
    max_relative_error = zero(BigFloat)
    failure = nothing
    assign!(index, source) = begin
        source_big = try
            BigFloat(source)
        catch error
            failure = string(label, " cannot be represented as BigFloat: ",
                sprint(showerror, error))
            return
        end
        target_value = try
            Float64(source_big)
        catch error
            failure = string(label, " cannot be represented as Float64: ",
                sprint(showerror, error))
            return
        end
        if isnan(source_big) || (isfinite(source_big) && !isfinite(target_value))
            failure = string(label, " conversion is non-finite")
            return
        elseif isfinite(source_big)
            round_trip = BigFloat(target_value)
            absolute_error = abs(source_big - round_trip)
            relative_error = absolute_error / max(abs(source_big), one(BigFloat))
            max_absolute_error = max(max_absolute_error, absolute_error)
            max_relative_error = max(max_relative_error, relative_error)
        elseif !(isinf(target_value) && signbit(source_big) == signbit(target_value))
            failure = string(label, " infinity changed sign or finiteness")
            return
        end
        target === nothing ? nothing : (target[index] = target_value)
    end
    if target === nothing
        assign!(nothing, value)
        converted = failure === nothing ? Float64(value) : NaN
    else
        for index in eachindex(value)
            assign!(index, value[index])
            failure === nothing || break
        end
        converted = target
    end
    (; value=converted, max_absolute_error, max_relative_error, failure)
end

"""Audit the one intentional BigFloat-to-Float64 evaluator boundary."""
function _pilot_float64_conversion_audit(certificate, scaled_L)
    reference_tolerance = BigFloat(certificate.reference_tolerance)
    metric_tolerance = hasproperty(certificate, :checks) &&
        certificate.checks !== nothing && hasproperty(certificate.checks, :spd_tolerance) ?
        BigFloat(certificate.checks.spd_tolerance) :
        BigFloat(PILOT_CONVERSION_TOLERANCE)
    relative_tolerance = BigFloat(PILOT_CONVERSION_TOLERANCE)
    converted_L = _pilot_convert_float64(scaled_L, "L")
    converted_K = _pilot_convert_float64(certificate.K, "K")
    converted_kinv = _pilot_convert_float64(certificate.kinv, "Kinv")
    converted_tau = _pilot_convert_float64(certificate.tau, "divisor volumes")
    converted_volume = _pilot_convert_float64(certificate.volume, "CY volume")
    comparison = (; L=converted_L.failure === nothing &&
            converted_L.max_absolute_error <= reference_tolerance ? :passed : :unsafe,
        K=converted_K.failure === nothing &&
            converted_K.max_absolute_error <= metric_tolerance ? :passed : :unsafe,
        kinv=converted_kinv.failure === nothing &&
            (converted_kinv.max_absolute_error <= metric_tolerance ||
             converted_kinv.max_relative_error <= relative_tolerance) ? :passed : :unsafe,
        tau=converted_tau.failure === nothing &&
            converted_tau.max_relative_error <= relative_tolerance ? :passed : :unsafe,
        volume=converted_volume.failure === nothing &&
            converted_volume.max_relative_error <= relative_tolerance ? :passed : :unsafe)
    status = all(value -> value == :passed, values(comparison)) ? :passed : :unsafe
    reason = status == :passed ? "Float64 evaluator conversion is within declared tolerances" :
        string("Float64 evaluator conversion exceeds a declared tolerance: ", comparison)
    (; status, reason,
       source_numeric_type=certificate.source_numeric_type,
       source_precision_bits=certificate.source_precision_bits,
       target_numeric_type=PILOT_TARGET_NUMERIC_TYPE,
       target_precision_bits=PILOT_TARGET_PRECISION_BITS,
       conversion_policy_version=PILOT_CONVERSION_POLICY_VERSION,
       conversion_error_bound=(; L=converted_L.max_absolute_error,
           K=converted_K.max_absolute_error, kinv=converted_kinv.max_absolute_error,
           kinv_absolute=converted_kinv.max_absolute_error,
           kinv_relative=converted_kinv.max_relative_error,
           tau=converted_tau.max_relative_error,
           volume=converted_volume.max_relative_error),
       conversion_tolerance=(; absolute_L=reference_tolerance,
           absolute_metric=metric_tolerance, relative=relative_tolerance,
           kinv_rule=PILOT_KINV_CONVERSION_RULE),
       conversion_comparison=comparison,
       L=converted_L.value, K=converted_K.value, tau=converted_tau.value,
       kinv=converted_kinv.value, volume=converted_volume.value)
end

function _pilot_normalize_gate_status(value, label::AbstractString)
    value === nothing && throw(ArgumentError("$label status is missing"))
    text = lowercase(strip(string(value)))
    text = replace(text, ' ' => '_', '-' => '_')
    status = Symbol(startswith(text, ":") ? text[2:end] : text)
    status in PILOT_GATE_STATUS_VALUES ||
        throw(ArgumentError("malformed $label status: $value"))
    status
end

function _pilot_certificate_gate_status(status::Symbol)
    status == :passed && return :passed
    status == :missing_evidence && return :missing_evidence
    status == :out_of_model && return :out_of_model
    status == :numerical_failure && return :numerical_failure
    status == :domain_failure && return :failed
    :failed
end

function _pilot_normalize_viability_status(value)
    value === nothing && throw(ArgumentError("physical_viability status is missing"))
    text = lowercase(strip(string(value)))
    text = replace(text, ' ' => '_', '-' => '_')
    status = Symbol(startswith(text, ":") ? text[2:end] : text)
    status in PILOT_VIABILITY_STATUS_VALUES ||
        throw(ArgumentError("malformed physical_viability status: $value"))
    status
end

function _pilot_control_status(values)
    statuses = Symbol[]
    for (value, label, allow_not_applicable) in values
        if value === nothing
            push!(statuses, :not_established)
            continue
        end
        if label == "potent_curve_volumes" && value isa AbstractArray
            if isempty(value)
                push!(statuses, :not_established)
            elseif all(entry -> entry isa Real && isfinite(entry), value) &&
                    minimum(value) > 1
                push!(statuses, :passed)
            else
                push!(statuses, :failed)
            end
            continue
        end
        text = lowercase(strip(string(value)))
        text = replace(text, ' ' => '_', '-' => '_')
        if text in ("passed", "validated", "complete", "completed", "true")
            push!(statuses, :passed)
        elseif allow_not_applicable && text in ("not_applicable", "not applicable")
            push!(statuses, :passed)
        elseif text in ("not_established", "unknown", "missing", "not_recorded", "none")
            push!(statuses, :not_established)
        elseif text in ("failed", "false", "out_of_model", "numerical_failure")
            push!(statuses, :failed)
        else
            push!(statuses, :failed)
        end
    end
    _pilot_control_aggregate(statuses)
end

function _pilot_control_aggregate(statuses::AbstractVector{<:Symbol})
    status = any(value -> value == :failed, statuses) ? :failed :
        all(value -> value == :passed, statuses) ? :passed : :not_established
    (; status, component_statuses=Tuple(statuses))
end

const PILOT_NONPHYSICAL_SCALING_REASON =
    "physical scaling gate is not applicable to a nonphysical diagnostic path"
const PILOT_NONPHYSICAL_CONTROL_REASON =
    "physical control gate is not applicable to a nonphysical diagnostic path"
const PILOT_NONPHYSICAL_VIABILITY_REASON =
    "nonphysical diagnostic output is not a physical viability result"

function _pilot_domain_result(status::Symbol, reason::AbstractString;
        scale=big(0), precision_bits::Int=0, tau=nothing, kinv=nothing,
        volume=nothing, K=nothing, phase_convention="missing", units="missing",
        normalization="missing", source_identity="missing",
        configuration_digest="missing", moduli_status=:not_established,
        physical_scaling_gate_status=status,
        physical_scaling_gate_reason=reason,
        physical_scaling_gate_provenance="scripts/inflation_scale_continuation.jl::pilot_physical_domain_certificate",
        physical_control_gate_status=:not_established,
        physical_control_gate_reason="physical control evidence was not evaluated",
        physical_control_gate_provenance="scripts/inflation_scale_continuation.jl::pilot_physical_domain_certificate",
        physical_viability_status=:not_evaluated,
        physical_viability_reason="physical viability was not evaluated",
        reference_diagnostic=nothing, checks=nothing,
        reference_tolerance=BigFloat("1e-10"),
        source_numeric_type="not_recorded", source_precision_bits::Int=0,
        target_numeric_type=PILOT_TARGET_NUMERIC_TYPE,
        target_precision_bits::Int=PILOT_TARGET_PRECISION_BITS,
        conversion_status=:not_attempted, conversion_error_bound=nothing,
        conversion_tolerance=nothing, conversion_comparison=:not_attempted,
        conversion_policy_version=PILOT_CONVERSION_POLICY_VERSION)
    scaling_status = _pilot_normalize_gate_status(
        _pilot_certificate_gate_status(Symbol(physical_scaling_gate_status)),
        "physical_scaling_gate")
    control_status = _pilot_normalize_gate_status(physical_control_gate_status,
        "physical_control_gate")
    viability_status = _pilot_normalize_viability_status(physical_viability_status)
    (; certificate_version=PILOT_DOMAIN_CERTIFICATE_VERSION, status,
       scale_status=status == :passed ? :physical : :unsupported,
       domain_status=status, domain_reason=String(reason),
       volume_normalization=status == :passed ? :full : :none,
       domain_certificate_version=PILOT_DOMAIN_CERTIFICATE_VERSION,
       fixed_point_status=:not_run, trajectory_status=:not_run,
       coverage_status=:not_started, moduli_status,
       phase_convention, units, normalization, source_identity,
       configuration_digest, precision_bits, tau, kinv, volume, K,
       reference_diagnostic, checks, reference_tolerance,
       physical_scaling_gate_status=scaling_status,
       physical_scaling_gate_reason=String(physical_scaling_gate_reason),
       physical_scaling_gate_provenance=String(physical_scaling_gate_provenance),
       physical_control_gate_status=control_status,
       physical_control_gate_reason=String(physical_control_gate_reason),
       physical_control_gate_provenance=String(physical_control_gate_provenance),
       physical_viability_status=viability_status,
       physical_viability_reason=String(physical_viability_reason),
       source_numeric_type, source_precision_bits, target_numeric_type,
       target_precision_bits, conversion_status, conversion_error_bound,
       conversion_tolerance, conversion_comparison, conversion_policy_version)
end

function _pilot_domain_missing_result(missing; scale=big(0), precision_bits::Int=0,
        phase_convention="missing", units="missing", normalization="missing",
        source_identity="missing", configuration_digest="missing",
        moduli_status=:not_established)
    reason = string("required physical-domain evidence is missing: ",
        join(sort!(unique(String.(missing))), ", "))
    _pilot_domain_result(:missing_evidence, reason; scale, precision_bits,
        phase_convention, units, normalization, source_identity,
        configuration_digest, moduli_status)
end

function _pilot_physical_domain_certificate_big(geometry, Q, L, K, scale,
        precision_bits::Int; source_identity=nothing, configuration_digest=nothing,
        phase_convention=nothing, units=nothing, normalization=nothing,
        reference_tolerance=BigFloat("1e-10"))
    get_or_override(names, override) = override === nothing ?
        _pilot_get_field(geometry, names) : (true, override)
    found_tau, raw_tau = _pilot_get_field(geometry,
        (:τ_volumes, :tau_volumes, :divisor_volumes))
    found_volume, raw_volume = _pilot_get_field(geometry, (:cy_volume, :CY_volume))
    found_kinv, raw_kinv = _pilot_get_field(geometry, (:kinv, :Kinv))
    found_prime, raw_prime = _pilot_get_field(geometry,
        (:prime_divisor_volumes, :direct_divisor_volumes))
    found_effective, raw_effective = _pilot_get_field(geometry,
        (:effective_divisor_volumes, :effective_volumes))
    found_curves, raw_curves = _pilot_get_field(geometry, (:curve_volumes,))
    found_potent, raw_potent = _pilot_get_field(geometry,
        (:potent_curve_volumes,))
    found_margin, raw_margin = _pilot_get_field(geometry,
        (:kahler_margin, :kaehler_margin, :kahler_cone_margin,
         :kaehler_cone_interior_margin))
    found_basis, raw_basis = _pilot_get_field(geometry,
        (:basis_identity, :basis_convention, :basis))
    found_orientation, raw_orientation = _pilot_get_field(geometry,
        (:charge_orientation, :charge_convention))
    found_phase, raw_phase = get_or_override((:phase_convention, :phases),
        phase_convention)
    found_units, raw_units = get_or_override((:units, :unit_convention), units)
    found_normalization, raw_normalization = get_or_override(
        (:normalization, :volume_normalization), normalization)
    found_source, raw_source = get_or_override(
        (:source_identity, :source_commit, :source_hash), source_identity)
    found_config, raw_config = configuration_digest === nothing ?
        _pilot_get_field(geometry, (:configuration_digest, :config_digest)) :
        (true, configuration_digest)
    found_moduli, raw_moduli = _pilot_get_field(geometry, (:moduli_status,))
    found_instanton, raw_instanton = _pilot_get_field(geometry,
        (:instanton_control, :instanton_status))
    found_perturbative, raw_perturbative = _pilot_get_field(geometry,
        (:perturbative_control, :eft_control, :perturbative_status))
    found_visible, raw_visible = _pilot_get_field(geometry,
        (:visible_sector_status, :qcd_status))
    found_spd, raw_spd = _pilot_get_field(geometry,
        (:spd_tolerance, :kinetic_spd_tolerance))

    control = _pilot_control_status((
        (found_potent ? raw_potent : nothing, "potent_curve_volumes", false),
        (found_instanton ? raw_instanton : nothing, "instanton_control", false),
        (found_perturbative ? raw_perturbative : nothing, "perturbative_control", false),
        (found_moduli ? raw_moduli : nothing, "moduli_status", false),
        (found_visible ? raw_visible : nothing, "visible_sector_status", true)))
    control_provenance =
        "geometry metadata: potent_curve_volumes, instanton_control, " *
        "perturbative_control, moduli_status, visible_sector_status"
    control_reason = string("control evidence status=", control.status,
        "; components=", control.component_statuses)

    missing = String[]
    for (found, label) in ((found_tau, "divisor_volumes"),
            (found_volume, "cy_volume"), (found_kinv, "kinv"),
            (found_prime, "prime_divisor_volumes"),
            (found_effective, "effective_divisor_volumes"),
            (found_curves, "curve_volumes"),
            (found_margin, "kahler_margin"), (found_basis, "basis_identity"),
            (found_orientation, "charge_orientation"),
            (found_phase, "phase_convention"), (found_units, "units"),
            (found_normalization, "normalization"),
            (found_source, "source_identity"),
            (found_config, "configuration_digest"),
            (found_spd, "spd_tolerance"))
        found || push!(missing, label)
    end
    isempty(missing) || return _pilot_domain_result(:missing_evidence,
        string("required physical-domain evidence is missing: ",
            join(sort!(unique(String.(missing))), ", ")); scale, precision_bits,
        phase_convention=raw_phase, units=raw_units, normalization=raw_normalization,
        source_identity=raw_source, configuration_digest=raw_config,
        moduli_status=raw_moduli,
        physical_control_gate_status=control.status,
        physical_control_gate_reason=control_reason,
        physical_control_gate_provenance=control_provenance)

    for (value, label) in ((raw_tau, "divisor_volumes"),
            (raw_volume, "cy_volume"), (raw_kinv, "kinv"),
            (raw_prime, "prime_divisor_volumes"),
            (raw_effective, "effective_divisor_volumes"),
            (raw_curves, "curve_volumes"),
            (raw_margin, "kahler_margin"),
            (raw_basis, "basis_identity"), (raw_orientation, "charge_orientation"),
            (raw_phase, "phase_convention"), (raw_units, "units"),
            (raw_normalization, "normalization"), (raw_source, "source_identity"),
            (raw_config, "configuration_digest"), (raw_spd, "spd_tolerance"))
        _pilot_nonempty_metadata(value) || push!(missing, label)
    end
    isempty(missing) || return _pilot_domain_result(:missing_evidence,
        string("required physical-domain evidence is missing: ",
            join(sort!(unique(String.(missing))), ", ")); scale, precision_bits,
        phase_convention=raw_phase, units=raw_units, normalization=raw_normalization,
        source_identity=raw_source, configuration_digest=raw_config,
        moduli_status=raw_moduli,
        physical_control_gate_status=control.status,
        physical_control_gate_reason=control_reason,
        physical_control_gate_provenance=control_provenance)

    source_provenance = _pilot_source_numeric_provenance(
        (raw_tau, raw_kinv, raw_volume, L, K))
    cert_result(status, reason; kwargs...) = _pilot_domain_result(status, reason;
        merge((; scale, precision_bits, phase_convention=raw_phase, units=raw_units,
                normalization=raw_normalization, source_identity=raw_source,
                configuration_digest=raw_config, moduli_status=raw_moduli,
                reference_tolerance,
                source_numeric_type=source_provenance.source_numeric_type,
                source_precision_bits=source_provenance.source_precision_bits,
                physical_control_gate_status=control.status,
                physical_control_gate_reason=control_reason,
                physical_control_gate_provenance=control_provenance),
            (; kwargs...))...)

    _pilot_nonempty_metadata(raw_basis) ||
        return cert_result(:missing_evidence,
            "basis identity is empty"; scale, precision_bits)
    _pilot_nonempty_metadata(raw_orientation) ||
        return cert_result(:missing_evidence,
            "charge orientation is empty"; scale, precision_bits)
    _pilot_nonempty_metadata(raw_phase) ||
        return cert_result(:missing_evidence,
            "phase convention is empty"; scale, precision_bits)
    _pilot_nonempty_metadata(raw_units) ||
        return cert_result(:missing_evidence,
            "units metadata is empty"; scale, precision_bits)
    string(raw_units) == PILOT_PHYSICAL_UNITS ||
        return cert_result(:out_of_model,
            string("physical mode requires the exact units contract ",
                PILOT_PHYSICAL_UNITS); scale, precision_bits)
    _pilot_nonempty_metadata(raw_source) ||
        return cert_result(:missing_evidence,
            "source identity is empty"; scale, precision_bits)
    _pilot_nonempty_metadata(raw_config) ||
        return cert_result(:missing_evidence,
            "configuration digest is empty"; scale, precision_bits)

    normalization_text = lowercase(strip(string(raw_normalization)))
    normalization_text == lowercase(PILOT_PHYSICAL_NORMALIZATION) ||
        return cert_result(:out_of_model,
            "physical mode requires the owner-selected homogeneous normalization";
            scale, precision_bits, normalization=raw_normalization,
            source_identity=raw_source, configuration_digest=raw_config)
    try
        k = BigFloat(scale)
        isfinite(k) && k > 0 || return cert_result(:domain_failure,
            "scale must be finite and positive"; scale=k, precision_bits)
        tau = BigFloat.(collect(raw_tau))
        kinv = BigFloat.(Matrix(raw_kinv))
        prime = BigFloat.(collect(raw_prime))
        effective = BigFloat.(collect(raw_effective))
        curves = BigFloat.(collect(raw_curves))
        potent = if found_potent
            try
                BigFloat.(collect(raw_potent))
            catch
                BigFloat[]
            end
        else
            BigFloat[]
        end
        volume = BigFloat(raw_volume)
        margin = BigFloat(raw_margin)
        spd_tolerance = BigFloat(raw_spd)
            all(isfinite, tau) && all(isfinite, kinv) && all(isfinite, prime) &&
            all(isfinite, effective) && all(isfinite, curves) &&
            all(isfinite, potent) &&
            isfinite(volume) && isfinite(margin) &&
            isfinite(spd_tolerance) || return cert_result(
                :domain_failure, "physical-domain values are non-finite";
                scale=k, precision_bits, normalization=raw_normalization,
                source_identity=raw_source, configuration_digest=raw_config)
        h11 = size(Q, 1)
        length(tau) == h11 || return cert_result(:domain_failure,
            "divisor-volume and charge dimensions disagree"; scale=k,
            precision_bits, normalization=raw_normalization,
            source_identity=raw_source, configuration_digest=raw_config)
        size(kinv) == (h11, h11) || return cert_result(:domain_failure,
            "Kinv and charge dimensions disagree"; scale=k, precision_bits,
            normalization=raw_normalization, source_identity=raw_source,
            configuration_digest=raw_config)
        volume > 0 && margin > 0 && spd_tolerance > 0 ||
            return cert_result(:domain_failure,
                "volume, Kähler-cone margin, and SPD tolerance must be positive";
                scale=k, precision_bits, normalization=raw_normalization,
                source_identity=raw_source, configuration_digest=raw_config)

        scaled_tau = k .* tau
        scaled_kinv = k^2 .* kinv
        scaled_volume = k^(BigFloat(3) / BigFloat(2)) * volume
        scaled_prime = k .* prime
        scaled_effective = k .* effective
        scaled_curves = sqrt(k) .* curves
        scaled_potent = sqrt(k) .* potent
        scaled_margin = sqrt(k) * margin
        minimum(scaled_prime) > 0 && minimum(scaled_effective) > 0 &&
            minimum(scaled_curves) > 0 &&
            scaled_margin > 0 ||
            return cert_result(:domain_failure,
                "effective-curve/divisor or Kähler-cone evidence failed";
                scale=k, precision_bits, tau=scaled_tau, kinv=scaled_kinv,
                volume=scaled_volume, phase_convention=raw_phase, units=raw_units,
                normalization=raw_normalization, source_identity=raw_source,
                configuration_digest=raw_config)

        control_components = collect(control.component_statuses)
        potent_status = _pilot_control_status((
            (found_potent ? scaled_potent : nothing, "potent_curve_volumes", false),))
        control_components[1] = potent_status.component_statuses[1]
        instanton_status = control_components[2]
        if instanton_status == :passed && minimum(scaled_prime) <= 1
            instanton_status = :failed
        end
        control_components[2] = instanton_status
        control = _pilot_control_aggregate(control_components)
        control_reason = string("control evidence status=", control.status,
            "; components=", control.component_statuses,
            "; scale-dependent volume checks evaluated at k=", k)

        base_K = BigFloat.(Matrix(K))
        size(base_K) == (h11, h11) || return cert_result(
            :domain_failure, "K and charge dimensions disagree"; scale=k,
            precision_bits, normalization=raw_normalization,
            source_identity=raw_source, configuration_digest=raw_config)
        symmetry_K = maximum(abs.(base_K - base_K'))
        symmetry_Kinv = maximum(abs.(kinv - kinv'))
        symmetry_K <= spd_tolerance && symmetry_Kinv <= spd_tolerance ||
            return cert_result(:domain_failure,
                "K and Kinv are not symmetric within the recorded tolerance";
                scale=k, precision_bits, normalization=raw_normalization,
                source_identity=raw_source, configuration_digest=raw_config)
        identity = Matrix{BigFloat}(I, h11, h11)
        inverse_residual = maximum(abs.(base_K * kinv - identity))
        inverse_residual <= spd_tolerance || return cert_result(
            :domain_failure, "K and Kinv are not reciprocal in one basis";
            scale=k, precision_bits, normalization=raw_normalization,
            source_identity=raw_source, configuration_digest=raw_config)
        scaled_K = base_K / k^2
        shifted_spd = try
            cholesky(Symmetric(scaled_K - spd_tolerance * identity))
            cholesky(Symmetric(scaled_kinv - spd_tolerance * identity))
            true
        catch
            false
        end
        shifted_spd ||
            return cert_result(:domain_failure,
                "the scaled kinetic metric is not positive definite"; scale=k,
                precision_bits, tau=scaled_tau, kinv=scaled_kinv,
                volume=scaled_volume, K=scaled_K,
                phase_convention=raw_phase, units=raw_units,
                normalization=raw_normalization, source_identity=raw_source,
                configuration_digest=raw_config)

        reference = try
            _pilot_reference_diagnostic(Q, L, tau, kinv, volume)
        catch error
            return cert_result(:domain_failure,
                string("stored Q/L reference is invalid: ", sprint(showerror, error));
                scale=k, precision_bits, tau=scaled_tau, kinv=scaled_kinv,
                volume=scaled_volume, K=scaled_K,
                phase_convention=raw_phase, units=raw_units,
                normalization=raw_normalization, source_identity=raw_source,
                configuration_digest=raw_config)
        end
        reference.sign_mismatches == 0 || return cert_result(
            :domain_failure, "stored Q/L reference has sign mismatches"; scale=k,
            precision_bits, tau=scaled_tau, kinv=scaled_kinv, volume=scaled_volume,
            K=scaled_K, phase_convention=raw_phase, units=raw_units,
            normalization=raw_normalization, source_identity=raw_source,
            configuration_digest=raw_config, reference_diagnostic=reference)
        BigFloat(reference.max_log10_error) <= BigFloat(reference_tolerance) ||
            return cert_result(:domain_failure,
                "stored Q/L reference exceeds the recorded tolerance"; scale=k,
                precision_bits, tau=scaled_tau, kinv=scaled_kinv,
                volume=scaled_volume, K=scaled_K,
                phase_convention=raw_phase, units=raw_units,
                normalization=raw_normalization, source_identity=raw_source,
                configuration_digest=raw_config, reference_diagnostic=reference)
        cert_result(:passed, "physical-domain certificate passed";
            scale=k, precision_bits, tau=scaled_tau, kinv=scaled_kinv,
            volume=scaled_volume, K=scaled_K, phase_convention=raw_phase,
            units=raw_units, normalization=raw_normalization,
            source_identity=raw_source, configuration_digest=raw_config,
            moduli_status=:not_established, reference_diagnostic=reference,
            checks=(; spd_tolerance, inverse_residual,
                spd_check="shifted_cholesky",
                kahler_margin=scaled_margin,
                minimum_prime_divisor_volume=minimum(scaled_prime),
                minimum_effective_divisor_volume=minimum(scaled_effective),
                minimum_curve_volume=minimum(scaled_curves)))
    catch error
        cert_result(:numerical_failure,
            string("physical-domain certificate evaluation failed: ",
                sprint(showerror, error)); scale=scale, precision_bits)
    end
end

function pilot_physical_domain_certificate(geometry, Q, L, K, scale::Real;
        source_identity=nothing, configuration_digest=nothing,
        phase_convention=nothing, units=nothing, normalization=nothing,
        precision_bits=nothing, reference_tolerance=BigFloat("1e-10"))
    geometry === nothing && return _pilot_domain_result(:missing_evidence,
        "physical scaling requires complete geometry-domain evidence"; scale)
    for (value, label) in ((Q, "charges"), (L, "stored_potential"), (K, "K"))
        _pilot_nonempty_metadata(value) || return _pilot_domain_result(
            :missing_evidence, "required physical-domain evidence is missing: $label";
            scale)
    end
    found_precision, raw_precision = _pilot_get_field(geometry,
        (:precision_bits,))
    precision_value = precision_bits === nothing ? raw_precision : precision_bits
    found_precision || precision_bits !== nothing ||
        return _pilot_domain_result(:missing_evidence,
            "precision_bits is missing from the physical-domain evidence"; scale)
    parsed_precision = try
        Int(precision_value)
    catch
        return _pilot_domain_result(:missing_evidence,
            "precision_bits is not an integer"; scale)
    end
    parsed_precision >= 128 || return _pilot_domain_result(:out_of_model,
        "physical refinement requires at least 128 bits"; scale,
        precision_bits=parsed_precision)
    setprecision(parsed_precision) do
        _pilot_physical_domain_certificate_big(geometry, Q, L, K, scale,
            parsed_precision; source_identity, configuration_digest,
            phase_convention, units, normalization, reference_tolerance)
    end
end

function _pilot_fixed_volume_diagnostic(Q, L, K, scale::Real, geometry)
    found_tau, raw_tau = _pilot_get_field(geometry,
        (:τ_volumes, :tau_volumes, :divisor_volumes))
    found_kinv, raw_kinv = _pilot_get_field(geometry, (:kinv, :Kinv))
    found_volume, raw_volume = _pilot_get_field(geometry, (:cy_volume, :CY_volume))
    found_tau && found_kinv && found_volume || throw(ArgumentError(
        "fixed-volume diagnostic requires divisor, Kinv, and CY-volume metadata"))
    T = BigFloat
    tau = T.(collect(raw_tau)); kinv = T.(Matrix(raw_kinv)); volume = T(raw_volume)
    k = T(scale)
    reference = _pilot_reference_diagnostic(Q, L, tau, kinv, volume)
    reference.sign_mismatches == 0 && reference.max_log10_error <= T("1e-10") ||
        throw(ArgumentError("stored Q/L reference does not match fixed diagnostic"))
    scaled_tau = k .* tau
    scaled_kinv = k^2 .* kinv
    scaled_volume = volume
    scaled_K = Hermitian(T.(Matrix(K)) / k^2)
    source_provenance = _pilot_source_numeric_provenance(
        (raw_tau, raw_kinv, raw_volume, L, K))
    (; Q=Matrix{Int}(Q), L=_pilot_author_potential(Q, scaled_tau, scaled_kinv,
           scaled_volume), K=scaled_K, tau=scaled_tau, kinv=scaled_kinv,
       volume=scaled_volume, reference=reference.expected,
       reference_diagnostic=reference, scale_source="author_fixed_volume_diagnostic",
       scale_status=:unsupported, volume_normalization=:fixed,
       domain_certificate_version=PILOT_DOMAIN_CERTIFICATE_VERSION,
       domain_status=:out_of_model,
       domain_reason="fixed CY-volume comparison is not the selected physical path",
       physical_scaling_gate_status=:not_applicable,
       physical_scaling_gate_reason=PILOT_NONPHYSICAL_SCALING_REASON,
       physical_scaling_gate_provenance="scripts/inflation_scale_continuation.jl::_pilot_fixed_volume_diagnostic",
       physical_control_gate_status=:not_applicable,
       physical_control_gate_reason=PILOT_NONPHYSICAL_CONTROL_REASON,
       physical_control_gate_provenance="scripts/inflation_scale_continuation.jl::_pilot_fixed_volume_diagnostic",
       physical_viability_status=:not_applicable,
       physical_viability_reason=PILOT_NONPHYSICAL_VIABILITY_REASON,
       fixed_point_status=:not_run, trajectory_status=:not_run,
       coverage_status=:not_started, moduli_status=:not_established,
       phase_convention="not_recorded", units="not_recorded",
       normalization="fixed_CY_volume_comparison", source_identity=PILOT_AUTHOR_SOURCE_IDENTITY,
       precision_bits=precision(BigFloat(0)), configuration_digest="not_recorded",
       source_numeric_type=source_provenance.source_numeric_type,
       source_precision_bits=source_provenance.source_precision_bits,
       target_numeric_type=PILOT_TARGET_NUMERIC_TYPE,
       target_precision_bits=PILOT_TARGET_PRECISION_BITS,
       conversion_status=:not_attempted, conversion_error_bound=nothing,
       conversion_tolerance=nothing, conversion_comparison=:not_attempted,
       conversion_policy_version=PILOT_CONVERSION_POLICY_VERSION,
       domain_certificate=nothing)
end

"""Apply the legacy homotopy or a certified homogeneous physical CY path."""
function pilot_scaled_inputs(Q::AbstractMatrix{<:Integer}, L::AbstractMatrix{<:Real},
        K::AbstractMatrix{<:Real}, scale::Real; scale_status::Symbol=:physical,
        geometry=nothing, volume_normalization::Symbol=:full,
        source_identity=nothing, configuration_digest=nothing,
        phase_convention=nothing, units=nothing, normalization=nothing,
        precision_bits=nothing)
    isfinite(scale) && scale > 0 || throw(ArgumentError("scale must be finite and positive"))
    if scale_status == :homotopy_only
        source_provenance = _pilot_source_numeric_provenance((L, K))
        return (; Q=Matrix{Int}(Q), L=pilot_homotopy_scale(L, scale), K,
            scale_source="generic_log10_amplitude_exponent_stretch",
            scale_status=:homotopy_only, volume_normalization=:none,
            domain_certificate_version=PILOT_DOMAIN_CERTIFICATE_VERSION,
            domain_status=:out_of_model,
            domain_reason="historical logarithmic homotopy is outside the physical domain",
            physical_scaling_gate_status=:not_applicable,
            physical_scaling_gate_reason=PILOT_NONPHYSICAL_SCALING_REASON,
            physical_scaling_gate_provenance=PILOT_HOMOTOPY_SOURCE_IDENTITY,
            physical_control_gate_status=:not_applicable,
            physical_control_gate_reason=PILOT_NONPHYSICAL_CONTROL_REASON,
            physical_control_gate_provenance=PILOT_HOMOTOPY_SOURCE_IDENTITY,
            physical_viability_status=:not_applicable,
            physical_viability_reason=PILOT_NONPHYSICAL_VIABILITY_REASON,
            fixed_point_status=:not_run, trajectory_status=:not_run,
            coverage_status=:not_started, moduli_status=:not_established,
            phase_convention="not_applicable", units="not_applicable",
            normalization="homotopy_only", source_identity=PILOT_HOMOTOPY_SOURCE_IDENTITY,
            precision_bits=0, configuration_digest="not_recorded",
            source_numeric_type=source_provenance.source_numeric_type,
            source_precision_bits=source_provenance.source_precision_bits,
            target_numeric_type=PILOT_TARGET_NUMERIC_TYPE,
            target_precision_bits=PILOT_TARGET_PRECISION_BITS,
            conversion_status=:not_attempted, conversion_error_bound=nothing,
            conversion_tolerance=nothing, conversion_comparison=:not_attempted,
            conversion_policy_version=PILOT_CONVERSION_POLICY_VERSION,
            domain_certificate=nothing)
    end
    if scale_status == :unsupported && volume_normalization == :fixed
        geometry === nothing && throw(ArgumentError(
            "fixed-volume diagnostic requires geometry metadata"))
        return _pilot_fixed_volume_diagnostic(Q, L, K, scale, geometry)
    end
    scale_status == :physical || throw(ArgumentError(
        "scale_status must be :homotopy_only, :physical, or :unsupported"))
    volume_normalization == :full || throw(ArgumentError(
        "fixed CY-volume normalization is diagnostic-only; physical mode requires :full"))
    certificate = pilot_physical_domain_certificate(geometry, Q, L, K, scale;
        source_identity, configuration_digest, phase_convention, units,
        normalization, precision_bits)
    certificate.status == :passed &&
        certificate.physical_scaling_gate_status == :passed ||
        throw(PilotPhysicalDomainError(certificate))
    scaled_L = _pilot_author_potential(Q, certificate.tau, certificate.kinv,
        certificate.volume)
    conversion = _pilot_float64_conversion_audit(certificate, scaled_L)
    conversion.status == :passed || begin
        failed_certificate = merge(certificate, (; status=:numerical_failure,
            scale_status=:unsupported, volume_normalization=:none,
            domain_status=:numerical_failure, domain_reason=conversion.reason,
            conversion_status=conversion.status,
            conversion_error_bound=conversion.conversion_error_bound,
            conversion_tolerance=conversion.conversion_tolerance,
            conversion_comparison=conversion.conversion_comparison))
        throw(PilotPhysicalDomainError(failed_certificate))
    end
    certified_certificate = merge(certificate, (; conversion_status=conversion.status,
        conversion_error_bound=conversion.conversion_error_bound,
        conversion_tolerance=conversion.conversion_tolerance,
        conversion_comparison=conversion.conversion_comparison,
        target_numeric_type=conversion.target_numeric_type,
        target_precision_bits=conversion.target_precision_bits))
    (; Q=Matrix{Int}(Q), L=conversion.L, K=Hermitian(conversion.K),
       tau=certificate.tau, kinv=certificate.kinv, volume=certificate.volume,
       reference=certificate.reference_diagnostic.expected,
       reference_diagnostic=certificate.reference_diagnostic,
       scale_source="owner_homogeneous_divisor_volume_path",
       scale_status=:physical, volume_normalization=:full,
       domain_certificate_version=certificate.domain_certificate_version,
       domain_status=certificate.domain_status,
       domain_reason=certificate.domain_reason,
       physical_scaling_gate_status=certificate.physical_scaling_gate_status,
       physical_scaling_gate_reason=certificate.physical_scaling_gate_reason,
       physical_scaling_gate_provenance=certificate.physical_scaling_gate_provenance,
       physical_control_gate_status=certificate.physical_control_gate_status,
       physical_control_gate_reason=certificate.physical_control_gate_reason,
       physical_control_gate_provenance=certificate.physical_control_gate_provenance,
       physical_viability_status=:not_evaluated,
       physical_viability_reason="branch viability is evaluated only after scale calculation",
       fixed_point_status=certificate.fixed_point_status,
       trajectory_status=certificate.trajectory_status,
       coverage_status=certificate.coverage_status,
       moduli_status=certificate.moduli_status,
       phase_convention=certificate.phase_convention, units=certificate.units,
       normalization=certificate.normalization,
       source_identity=certificate.source_identity,
       precision_bits=certificate.precision_bits,
       configuration_digest=certificate.configuration_digest,
       source_numeric_type=certificate.source_numeric_type,
       source_precision_bits=certificate.source_precision_bits,
       target_numeric_type=conversion.target_numeric_type,
       target_precision_bits=conversion.target_precision_bits,
       conversion_status=conversion.status,
       conversion_error_bound=conversion.conversion_error_bound,
       conversion_tolerance=conversion.conversion_tolerance,
       conversion_comparison=conversion.conversion_comparison,
       conversion_policy_version=conversion.conversion_policy_version,
       domain_certificate=certified_certificate)
end

function _pilot_data_dir(path::AbstractString)
    isempty(strip(path)) && throw(ArgumentError("--data-dir is required"))
    data_dir = realpath(abspath(expanduser(path)))
    data_dir in ("/", homedir()) &&
        throw(ArgumentError("refusing to use root-like data directory: $data_dir"))
    isdir(data_dir) || throw(ArgumentError("data directory does not exist: $data_dir"))
    data_dir
end

function _pilot_geometry_path(data_dir::AbstractString, geom::GeometryIndex)
    joinpath(data_dir, string("h11_", lpad(geom.h11, 3, '0')),
        string("np_", lpad(geom.polytope, 7, '0')),
        string("cy_", lpad(geom.frst, 7, '0')), "cyax.h5")
end

function _pilot_parse_prefixed_int(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        nothing
    end
end

function _pilot_discover(data_dir::AbstractString; h11_filter=nothing)
    geometries = GeometryIndex[]
    for h11_name in sort(filter(name -> startswith(name, "h11_"), readdir(data_dir)))
        h11 = _pilot_parse_prefixed_int(h11_name, "h11_")
        h11 === nothing && continue
        h11_filter === nothing || h11 == h11_filter || continue
        h11_path = joinpath(data_dir, h11_name)
        isdir(h11_path) || continue
        for np_name in sort(filter(name -> startswith(name, "np_"), readdir(h11_path)))
            polytope = _pilot_parse_prefixed_int(np_name, "np_")
            polytope === nothing && continue
            np_path = joinpath(h11_path, np_name)
            isdir(np_path) || continue
            for cy_name in sort(filter(name -> startswith(name, "cy_"), readdir(np_path)))
                frst = _pilot_parse_prefixed_int(cy_name, "cy_")
                frst === nothing && continue
                isfile(joinpath(np_path, cy_name, "cyax.h5")) || continue
                push!(geometries, GeometryIndex(h11, polytope, frst))
            end
        end
    end
    geometries
end

function pilot_select_geometries(data_dir::AbstractString,
        requested::AbstractVector{<:GeometryIndex};
        h11_filter=nothing, max_geometries=nothing)
    geometries = isempty(requested) ? _pilot_discover(data_dir; h11_filter) : unique(requested)
    any(geom -> geom.h11 == 491, geometries) &&
        throw(ArgumentError("h11=491 is prohibited: no local geometry group exists"))
    sort!(geometries, by=geom -> (geom.h11, geom.polytope, geom.frst))
    max_geometries === nothing ? geometries : geometries[1:min(max_geometries, length(geometries))]
end

function _pilot_branch_estimate(selected, negative_mode_range)
    h11 = size(selected.Qtilde, 1)
    determinant = abs(det(Matrix{BigInt}(selected.Qtilde)))
    full_mask_count = BigInt(2)^h11
    negative_mode_range === nothing && return determinant * full_mask_count
    nonzero_count = count(!iszero, @view selected.Ltilde[1, :])
    zero_count = h11 - nonzero_count
    masks = sum(index <= nonzero_count ?
        binomial(BigInt(nonzero_count), index) : big(0)
        for index in negative_mode_range; init=big(0)) * (BigInt(2)^zero_count)
    determinant * masks
end

function _pilot_mask_coverage(selected, negative_mode_range)
    h11 = size(selected.Qtilde, 1)
    full = BigInt(2)^h11
    negative_mode_range === nothing && return (mask_count=full, masks_visited=full)
    nonzero_count = count(!iszero, @view selected.Ltilde[1, :])
    zero_count = h11 - nonzero_count
    visited = sum(index <= nonzero_count ?
        binomial(BigInt(nonzero_count), index) : big(0)
        for index in negative_mode_range; init=big(0)) * (BigInt(2)^zero_count)
    (; mask_count=full, masks_visited=visited)
end

"""Stream and copy bounded reference seeds; the package streamer reuses theta."""
function pilot_collect_seeds(Q::Matrix{Int}, L::Matrix{Float64};
        max_branches::Int, negative_mode_range=nothing, max_negative_modes=nothing,
        max_stage_allocated_bytes::Int=750_000_000)
    selected = CYAxiverse.generate.LQtilde(Q, L)
    range = max_negative_modes === nothing ? negative_mode_range : 0:max_negative_modes
    coverage = _pilot_mask_coverage(selected, range)
    determinant = abs(det(Matrix{BigInt}(selected.Qtilde)))
    estimate = _pilot_branch_estimate(selected, range)
    # Conservative preflight: the bounded corrector retains hessian-sized
    # workspaces and NLsolve temporaries per retained branch.  This is a guard
    # against materializing a run that has already exceeded the handoff's
    # 750-MB stage-allocation envelope; measured allocation is still reported.
    h11 = size(Q, 1)
    estimated_bytes = estimate * BigInt(h11)^2 * 4_000
    if estimated_bytes > BigInt(max_stage_allocated_bytes)
        coverage = _pilot_mask_coverage(selected, range)
        message = string("correction preflight estimate ", estimated_bytes,
            " bytes exceeds max_stage_allocated_bytes=", max_stage_allocated_bytes)
        fallback_stream = (; branch_count=estimate, mask_count=coverage.mask_count,
            masks_visited=big(0), masks_skipped=coverage.mask_count,
            lattice_copy_count=determinant, lattice_copies_visited=big(0),
            negative_mode_range=range,
            search_classification=range === nothing ?
                :complete_enumeration : :deterministic_low_index_enumeration)
        return (; selected, seeds=Vector{Vector{Float64}}(), modes=Int[],
            stream=fallback_stream, determinant, estimate, estimated_bytes,
            status=:resource_cap, error=message, seconds=0.0)
    end
    seeds = Vector{Vector{Float64}}()
    modes = Int[]
    started = time_ns()
    try
        stream = CYAxiverse.generate.foreach_leading_critical_branch(
            selected; max_branches, negative_mode_range, max_negative_modes) do theta, index
            push!(seeds, copy(theta))
            push!(modes, index)
        end
        return (; selected, seeds, modes, stream, determinant, estimate,
            estimated_bytes, status=:completed, error="",
            seconds=(time_ns() - started) / 1e9)
    catch error
        message = sprint(showerror, error)
        status = occursin("leading branch enumeration would create", message) ?
            :branch_cap : :failed
        # The bounded enumerator throws before exposing a completion report;
        # do not infer mask coverage from the unconstrained estimate.
        fallback_stream = (; branch_count=estimate, mask_count=coverage.mask_count,
            masks_visited=big(0),
            masks_skipped=coverage.mask_count,
            lattice_copy_count=determinant,
            lattice_copies_visited=big(0),
            negative_mode_range=range,
            search_classification=range === nothing ?
                :complete_enumeration : :deterministic_low_index_enumeration)
        return (; selected, seeds, modes, stream=fallback_stream, determinant,
            estimate, estimated_bytes, status, error=message,
            seconds=(time_ns() - started) / 1e9)
    end
end

function _pilot_periodic_distance(left::AbstractVector, right::AbstractVector)
    maximum(min.(abs.(left .- right), 1 .- abs.(left .- right)))
end

function _pilot_classify(theta, Q::Matrix{Int}, L::Matrix{Float64}, factor,
        evaluator=nothing)
    evaluator = evaluator === nothing ?
        CYAxiverse.generate.structured_charge_evaluator(Q, L) : evaluator
    derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
        evaluator, theta, Q)
    lower = factor isa AbstractMatrix ? factor : Matrix(factor.L)
    canonical_hessian = lower \ derivatives.hessian / lower'
    eigenvalues = eigvals(Symmetric(canonical_hessian))
    inverse_metric_gradient = lower' \ (lower \ derivatives.gradient)
    gradient_norm = sqrt(max(dot(derivatives.gradient, inverse_metric_gradient), 0.0))
    value = derivatives.value
    epsilon = value == 0 ? Inf : 0.5 * (gradient_norm / abs(value))^2
    eta_values = value == 0 ? fill(Inf, length(eigenvalues)) : eigenvalues ./ value
    modes = CYAxiverse.generate.spectrum_mode_counts(eigenvalues)
    (; value, gradient_norm, epsilon,
       min_eta=minimum(eta_values), max_eta=maximum(eta_values),
       abs_min_eta=minimum(abs.(eta_values)),
       hessian_min=minimum(eigenvalues), hessian_max=maximum(eigenvalues),
       abs_min_hessian=minimum(abs.(eigenvalues)),
       negative_modes=modes.negative,
       zeroish_modes=modes.zeroish,
       positive_modes=modes.positive, eigenvalues)
end

_pilot_screen_pass(classification) = classification.value > 0 &&
    classification.negative_modes > 0 && classification.epsilon < 1 &&
    abs(classification.min_eta) < 1

function _pilot_correct(seed, Q::Matrix{Int}, L::Matrix{Float64};
        residual_tolerance::Float64, max_iterations::Int, evaluator=nothing)
    evaluator = evaluator === nothing ?
        CYAxiverse.generate.structured_charge_evaluator(Q, L) : evaluator
    function residual!(out, theta)
        derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
            evaluator, theta, Q)
        out .= derivatives.gradient
        nothing
    end
    function jacobian!(out, theta)
        derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
            evaluator, theta, Q)
        out .= derivatives.hessian
        nothing
    end
    started = time_ns()
    try
        result = nlsolve(residual!, jacobian!, Float64.(seed);
            method=:trust_region, ftol=residual_tolerance,
            xtol=residual_tolerance, iterations=max_iterations)
        theta = mod.(result.zero, 1.0)
        derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
            evaluator, theta, Q)
        residual = norm(derivatives.gradient, Inf)
        converged = (result.f_converged || result.x_converged) &&
            isfinite(residual) && residual <= residual_tolerance
        status = converged ? :converged : :residual_failed
        (; theta, status, residual, iterations=result.iterations,
           seconds=(time_ns() - started) / 1e9, error="")
    catch error
        (; theta=mod.(Float64.(seed), 1.0), status=:failed,
           residual=Inf, iterations=0, seconds=(time_ns() - started) / 1e9,
           error=sprint(showerror, error))
    end
end

mutable struct _PilotBranchRecord
    seed_index::Int
    leading_negative_modes::Int
    seed_theta::Vector{Float64}
    corrected_theta::Vector{Float64}
    correction_status::Symbol
    correction_residual::Float64
    correction_iterations::Int
    correction_seconds::Float64
    branch_provenance_id::String
    branch_match_id::String
    matching_status::Symbol
    classification::Union{Nothing,NamedTuple}
    near_catastrophe::Bool
    catastrophe_bracket::String
    catastrophe_reason::String
    candidate_status::Symbol
    candidate_reason::String
    physical_viability_status::Symbol
    physical_viability_reason::String
    failure::String
end

function _pilot_physical_viability(scale_status::Symbol, classification;
        physical_scaling_gate_status=nothing,
        physical_control_gate_status=nothing)
    screen = classification !== nothing && _pilot_screen_pass(classification)
    if scale_status == :homotopy_only || scale_status == :unsupported
        return (; candidate_status=scale_status == :homotopy_only && screen ?
                :corrected_candidate : :none,
            candidate_reason=scale_status == :homotopy_only && screen ?
                "corrected point passes existing Float64 screen in a nonphysical homotopy" :
                "nonphysical diagnostic output is not a physical candidate",
            physical_viability_status=:not_applicable,
            physical_viability_reason=PILOT_NONPHYSICAL_VIABILITY_REASON)
    end
    scale_status == :physical || return (; candidate_status=:none,
        candidate_reason="unknown scale status; candidate classification is blocked",
        physical_viability_status=:blocked_scaling_gate,
        physical_viability_reason="malformed scale status blocks physical viability")
    scaling = try
        _pilot_normalize_gate_status(physical_scaling_gate_status,
            "physical_scaling_gate")
    catch error
        return (; candidate_status=:none,
            candidate_reason=sprint(showerror, error),
            physical_viability_status=:blocked_scaling_gate,
            physical_viability_reason=sprint(showerror, error))
    end
    scaling == :passed || return (; candidate_status=:none,
        candidate_reason="physical scaling gate is not passed; candidate classification is blocked",
        physical_viability_status=:blocked_scaling_gate,
        physical_viability_reason="physical_scaling_gate_status=$scaling")
    control = try
        _pilot_normalize_gate_status(physical_control_gate_status,
            "physical_control_gate")
    catch error
        return (; candidate_status=:none,
            candidate_reason=sprint(showerror, error),
            physical_viability_status=:not_established,
            physical_viability_reason=sprint(showerror, error))
    end
    control == :passed || return (; candidate_status=:none,
        candidate_reason="physical control gate is not passed; candidate classification is blocked",
        physical_viability_status=:not_established,
        physical_viability_reason="physical_control_gate_status=$control")
    screen ? (; candidate_status=:physical_candidate_eligible,
        candidate_reason="screen eligibility only; not a validated or physically viable candidate",
        physical_viability_status=:eligible_not_validated,
        physical_viability_reason="both gates passed and the existing Float64 screen passed") :
        (; candidate_status=:none,
        candidate_reason="corrected point does not pass existing Float64 screen",
        physical_viability_status=:not_candidate,
        physical_viability_reason="screen eligibility was not met")
end

function _pilot_records(seeds, modes, Q, L, factor; residual_tolerance,
        max_iterations, duplicate_tolerance, scale_status::Symbol=:homotopy_only,
        physical_scaling_gate_status=nothing,
        physical_control_gate_status=nothing)
    evaluator = CYAxiverse.generate.structured_charge_evaluator(Q, L)
    records = _PilotBranchRecord[]
    for index in eachindex(seeds)
        corrected = _pilot_correct(seeds[index], Q, L;
            residual_tolerance, max_iterations, evaluator)
        classification = nothing
        status = corrected.status
        failure = corrected.error
        if corrected.status == :converged
            try
                classification = _pilot_classify(corrected.theta, Q, L, factor, evaluator)
                if !all(isfinite, classification.eigenvalues)
                    status = :inertia_failed
                    failure = "generalized Hessian eigenvalues are non-finite"
                end
            catch error
                status = :inertia_failed
                failure = sprint(showerror, error)
            end
        end
        duplicate = corrected.status == :converged && any(record ->
            record.correction_status == :converged &&
            _pilot_periodic_distance(record.corrected_theta, corrected.theta) <=
                duplicate_tolerance, records)
        status = duplicate && status == :converged ? :duplicate : status
        viability = _pilot_physical_viability(scale_status, classification;
            physical_scaling_gate_status, physical_control_gate_status)
        push!(records, _PilotBranchRecord(index, modes[index], copy(seeds[index]),
            copy(corrected.theta), status, corrected.residual, corrected.iterations,
            corrected.seconds, string("seed-", lpad(index, 8, '0')),
            string("branch-", lpad(index, 8, '0')), :unmatched, classification,
            false, "", "", viability.candidate_status,
            viability.candidate_reason, viability.physical_viability_status,
            viability.physical_viability_reason, failure))
    end
    records
end

"""Greedy deterministic periodic matching, with distance then index tie-breaks."""
function pilot_match_records!(previous, current; matching_tolerance::Float64)
    pairs = Tuple{Float64, Int, Int}[]
    for (left_index, left) in enumerate(previous), (right_index, right) in enumerate(current)
        left.correction_status == :converged || continue
        right.correction_status == :converged || continue
        distance = _pilot_periodic_distance(left.corrected_theta, right.corrected_theta)
        distance <= matching_tolerance && push!(pairs, (distance, left_index, right_index))
    end
    sort!(pairs, by=pair -> pair)
    used_previous = falses(length(previous))
    used_current = falses(length(current))
    matches = Tuple{Int, Int, Float64}[]
    for (distance, left_index, right_index) in pairs
        used_previous[left_index] && continue
        used_current[right_index] && continue
        used_previous[left_index] = true
        used_current[right_index] = true
        push!(matches, (left_index, right_index, distance))
        current[right_index].matching_status = :matched
        current[right_index].branch_match_id = previous[left_index].branch_match_id
    end
    for (index, record) in enumerate(previous)
        record.correction_status == :converged && !used_previous[index] &&
            (record.matching_status = :lost)
    end
    for (index, record) in enumerate(current)
        record.matching_status == :unmatched && record.correction_status == :converged &&
            (record.matching_status = :new)
    end
    matches
end

function _pilot_mark_crossings!(previous, current, matches, scale_left, scale_right;
        zero_eigenvalue_tolerance::Float64, bracket_number::Int,
        previous_minima::Int, current_minima::Int)
    branch_change = count(record -> record.correction_status == :converged, previous) !=
        count(record -> record.correction_status == :converged, current)
    minima_change = previous_minima != current_minima
    bracket = string(scale_left, ":", scale_right)
    marked = 0
    for (left_index, right_index, _) in matches
        left, right = previous[left_index], current[right_index]
        left.classification === nothing && continue
        right.classification === nothing && continue
        left_min = left.classification.hessian_min
        right_min = right.classification.hessian_min
        sign_change = (left_min <= 0 <= right_min) || (right_min <= 0 <= left_min)
        scale = max(1.0, abs(left.classification.hessian_max),
            abs(right.classification.hessian_max))
        near_zero = min(abs(left_min), abs(right_min)) <=
            zero_eigenvalue_tolerance * scale
        flag = sign_change || (near_zero && (branch_change || minima_change))
        flag || continue
        marked += 1
        reason = sign_change ? "smallest generalized Hessian eigenvalue sign change" :
            branch_change ? "near-zero smallest eigenvalue with branch-count change" :
            "near-zero smallest eigenvalue with minima-count change"
        for record in (left, right)
            record.near_catastrophe = true
            record.catastrophe_bracket = string("bracket-", lpad(bracket_number, 4, '0'))
            record.catastrophe_reason = reason
            if !_pilot_screen_pass(record.classification) &&
                    record.physical_viability_status == :not_applicable
                record.candidate_status = :near_catastrophe_only
            elseif record.physical_viability_status == :eligible_not_validated
                record.physical_viability_status = :near_catastrophe_not_validated
                record.physical_viability_reason =
                    "screen-eligible point lies in a flagged catastrophe bracket"
            end
            record.candidate_reason = reason
        end
    end
    marked
end

function _pilot_init_branch_ids!(records)
    for record in records
        record.branch_match_id = record.branch_provenance_id
        record.matching_status = :reference
    end
end

function _pilot_scalar(value)
    value === nothing && return ""
    value isa AbstractVector && return join(string.(value), ';')
    value isa Symbol && return string(value)
    value
end

function _pilot_csv_escape(value)
    value === nothing && return ""
    text = string(_pilot_scalar(value))
    text = replace(text, '"' => "\"\"")
    occursin(r"[,\"\n\r]", text) ? string('"', text, '"') : text
end

function _pilot_csv_fields(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if character == '"'
            next_index = nextind(line, index)
            if quoted && next_index <= lastindex(line) && line[next_index] == '"'
                write(buffer, '"')
                index = nextind(line, next_index)
                continue
            end
            quoted = !quoted
        elseif character == ',' && !quoted
            push!(fields, String(take!(buffer)))
        else
            write(buffer, character)
        end
        index = nextind(line, index)
    end
    push!(fields, String(take!(buffer)))
    fields
end

function pilot_prepare_csv(path::AbstractString, fields; append::Bool=false)
    path = abspath(expanduser(path))
    mkpath(dirname(path))
    header = join(string.(fields), ',')
    if !append && isfile(path)
        throw(ArgumentError("refusing to overwrite append-only pilot CSV: $path; " *
            "choose a new path or use --resume"))
    elseif append && isfile(path)
        first_line = open(path, "r") do io
            eof(io) ? "" : chomp(readline(io))
        end
        first_line == header || throw(ArgumentError("CSV schema mismatch: $path"))
    else
        open(path, "w") do io
            println(io, header)
        end
    end
    path
end

function pilot_append_csv(path::AbstractString, row, fields)
    values = (_pilot_csv_escape(hasproperty(row, field) ? getproperty(row, field) : nothing)
        for field in fields)
    open(path, "a") do io
        println(io, join(values, ','))
        flush(io)
    end
end

function _pilot_completed_scales(path::AbstractString)
    completed = Set{Tuple{Int, Int, Int, Float64}}()
    isfile(path) || return completed
    lines = readlines(path)
    length(lines) < 2 && return completed
    header = _pilot_csv_fields(lines[1])
    positions = Dict(Symbol(name) => index for (index, name) in enumerate(header))
    required = (:row_type, :h11, :polytope, :frst, :sampled_scale)
    all(name -> haskey(positions, name), required) || return completed
    for line in @view lines[2:end]
        isempty(strip(line)) && continue
        fields = _pilot_csv_fields(line)
        length(fields) == length(header) || continue
        fields[positions[:row_type]] == "scale" || continue
        try
            push!(completed, (parse(Int, fields[positions[:h11]]),
                parse(Int, fields[positions[:polytope]]),
                parse(Int, fields[positions[:frst]]),
                parse(Float64, fields[positions[:sampled_scale]])))
        catch
        end
    end
    completed
end

function _pilot_empty_classification()
    (; value=NaN, gradient_norm=NaN, epsilon=NaN, min_eta=NaN,
       max_eta=NaN, abs_min_eta=NaN, hessian_min=NaN, hessian_max=NaN,
       zeroish_modes=0, negative_modes=0, positive_modes=0)
end

function _pilot_branch_row(record, context; wall_seconds=record.correction_seconds,
        failure=record.failure)
    classification = record.classification === nothing ?
        _pilot_empty_classification() : record.classification
    merge(context, (; row_type=:branch,
        branch_seed_index=record.seed_index,
        leading_negative_modes=record.leading_negative_modes,
        leading_seed_theta=record.seed_theta, corrected_theta=record.corrected_theta,
        correction_status=record.correction_status,
        correction_residual=record.correction_residual,
        correction_iterations=record.correction_iterations,
        branch_provenance_id=record.branch_provenance_id,
        branch_match_id=record.branch_match_id,
        matching_status=record.matching_status,
        value=classification.value, gradient_norm=classification.gradient_norm,
        epsilon=classification.epsilon, min_eta=classification.min_eta,
        max_eta=classification.max_eta, abs_min_eta=classification.abs_min_eta,
        hessian_min=classification.hessian_min, hessian_max=classification.hessian_max,
        zeroish_modes=classification.zeroish_modes,
        negative_modes=classification.negative_modes,
        positive_modes=classification.positive_modes,
        near_catastrophe=record.near_catastrophe,
        catastrophe_bracket=record.catastrophe_bracket,
        catastrophe_reason=record.catastrophe_reason,
        candidate_status=record.candidate_status,
        candidate_reason=record.candidate_reason,
        physical_viability_status=record.physical_viability_status,
        physical_viability_reason=record.physical_viability_reason,
        wall_seconds, failure))
end

function _pilot_summary_row(context, records, seed_info; bracket_count=0,
        failure="", wall_seconds=0.0, allocated_bytes=0, output_bytes=0)
    converged = [record for record in records if record.correction_status == :converged]
    matched = count(record -> record.matching_status == :matched, records)
    lost = count(record -> record.matching_status == :lost, records)
    new = count(record -> record.matching_status == :new, records)
    duplicates = count(record -> record.correction_status == :duplicate, records)
    failed = count(record -> !(record.correction_status in (:converged, :duplicate)), records)
    corrected_candidates = count(record -> record.candidate_status ==
        :corrected_candidate, records)
    candidates = count(record -> record.candidate_status in
        (:corrected_candidate, :physical_candidate_eligible), records)
    physical_eligible = count(record -> record.candidate_status ==
        :physical_candidate_eligible, records)
    summary_viability_status = context.physical_viability_status
    summary_viability_reason = context.physical_viability_reason
    if context.scale_status == :physical
        if context.physical_scaling_gate_status != :passed
            summary_viability_status = :blocked_scaling_gate
            summary_viability_reason = "physical scaling gate is not passed"
        elseif context.physical_control_gate_status != :passed
            summary_viability_status = :not_established
            summary_viability_reason = "physical control gate is not passed"
        elseif physical_eligible > 0
            summary_viability_status = :eligible_not_validated
            summary_viability_reason =
                "one or more branches passed the screen; no validated candidate claim"
        else
            summary_viability_status = :not_candidate
            summary_viability_reason = "no branch passed the existing Float64 screen"
        end
    end
    minima = count(record -> record.classification !== nothing &&
        record.classification.negative_modes == 0, records)
    merge(context, (; row_type=:scale, branch_seed_index=nothing,
        leading_negative_modes=nothing, leading_seed_theta=nothing,
        corrected_theta=nothing, correction_status=nothing,
        correction_residual=nothing, correction_iterations=nothing,
        branch_provenance_id=nothing, branch_match_id=nothing,
        matching_status=nothing, value=nothing, gradient_norm=nothing,
        epsilon=nothing, min_eta=nothing, max_eta=nothing,
        abs_min_eta=nothing, hessian_min=nothing, hessian_max=nothing,
        zeroish_modes=nothing, negative_modes=nothing, positive_modes=nothing,
        near_catastrophe=bracket_count > 0,
        catastrophe_bracket=bracket_count == 0 ? "" : "count=$bracket_count",
        catastrophe_reason=bracket_count == 0 ? "" : "matched branch bracket flagged",
        candidate_status=nothing, candidate_reason=nothing,
        physical_viability_status=summary_viability_status,
        physical_viability_reason=summary_viability_reason,
        seed_count=length(seed_info.seeds), corrected_count=length(converged),
        matched_count=matched, lost_count=lost, new_count=new,
        duplicate_count=duplicates, correction_failed_count=failed,
        near_catastrophe_brackets=bracket_count,
        screen_candidate_count=candidates,
        corrected_candidate_count=corrected_candidates,
        physical_eligible_count=physical_eligible,
        minima_count=minima,
        wall_seconds, allocated_bytes, output_bytes, failure))
end

function _pilot_context(geom, data_dir, geometry_path, scale, hierarchy,
        seed_info, options; scaled=nothing)
    stream = seed_info.stream
    certificate = scaled !== nothing && hasproperty(scaled, :domain_certificate) ?
        scaled.domain_certificate : get(options, :physical_domain_certificate, nothing)
    actual_scale_status = if scaled !== nothing
        scaled.scale_status
    elseif options[:scale_status] == :physical &&
            scale == 1.0 && certificate !== nothing && certificate.status == :passed
        :physical
    elseif options[:scale_status] == :physical
        :unsupported
    else
        options[:scale_status]
    end
    provenance = scaled !== nothing ? scaled : certificate
    domain_status = if scaled !== nothing && hasproperty(scaled, :domain_status)
        scaled.domain_status
    elseif actual_scale_status == :physical && certificate !== nothing
        certificate.domain_status
    elseif options[:scale_status] == :physical
        :missing_evidence
    else
        get(options, :domain_status, :out_of_model)
    end
    domain_reason = if scaled !== nothing && hasproperty(scaled, :domain_reason)
        scaled.domain_reason
    elseif actual_scale_status == :physical && certificate !== nothing
        certificate.domain_reason
    elseif options[:scale_status] == :physical
        "same-scale physical-domain certificate was not evaluated"
    else
        get(options, :domain_reason, "domain certificate not applicable")
    end
    metadata(name, default) = provenance !== nothing && hasproperty(provenance, name) ?
        getproperty(provenance, name) : default
    default_gate_status = options[:scale_status] == :physical ?
        :missing_evidence : :not_applicable
    scaling_gate_status = _pilot_normalize_gate_status(
        metadata(:physical_scaling_gate_status, default_gate_status),
        "physical_scaling_gate")
    control_gate_status = _pilot_normalize_gate_status(
        metadata(:physical_control_gate_status,
            options[:scale_status] == :physical ? :not_established : :not_applicable),
        "physical_control_gate")
    viability_default = actual_scale_status == :physical ? :not_evaluated :
        actual_scale_status in (:homotopy_only, :unsupported) ? :not_applicable :
        :blocked_scaling_gate
    viability_status = _pilot_normalize_viability_status(
        metadata(:physical_viability_status, viability_default))
    coverage_status = seed_info.status == :completed ?
        (stream.search_classification == :complete_enumeration ?
         :complete : :partial_index_range) : seed_info.status
    scale_source = metadata(:scale_source,
        actual_scale_status == :physical ? "owner_homogeneous_divisor_volume_path" :
        actual_scale_status == :unsupported ? "nonphysical_diagnostic" :
        "generic_log10_amplitude_exponent_stretch")
    (; row_type=:scale, schema_version=PILOT_SCHEMA_VERSION,
       run_id=options[:run_id], data_root=data_dir, geometry_path,
       h11=geom.h11, polytope=geom.polytope, frst=geom.frst,
       reference_scale=1.0, sampled_scale=scale,
       scale_grid=join(string.(options[:scale_grid]), ';'),
       scale_source, scale_status=actual_scale_status,
       volume_normalization=metadata(:volume_normalization,
           actual_scale_status == :physical ? :full : :none),
       domain_certificate_version=PILOT_DOMAIN_CERTIFICATE_VERSION,
       domain_status, domain_reason,
       physical_scaling_gate_status=scaling_gate_status,
       physical_scaling_gate_reason=metadata(:physical_scaling_gate_reason,
           scaling_gate_status == :not_applicable ? PILOT_NONPHYSICAL_SCALING_REASON :
           "physical scaling gate status=$scaling_gate_status"),
       physical_scaling_gate_provenance=metadata(:physical_scaling_gate_provenance,
           "scripts/inflation_scale_continuation.jl::pilot_physical_domain_certificate"),
       physical_control_gate_status=control_gate_status,
       physical_control_gate_reason=metadata(:physical_control_gate_reason,
           control_gate_status == :not_applicable ? PILOT_NONPHYSICAL_CONTROL_REASON :
           "physical control gate status=$control_gate_status"),
       physical_control_gate_provenance=metadata(:physical_control_gate_provenance,
           "scripts/inflation_scale_continuation.jl::pilot_physical_domain_certificate"),
       physical_viability_status=viability_status,
       physical_viability_reason=metadata(:physical_viability_reason,
           viability_status == :not_applicable ? PILOT_NONPHYSICAL_VIABILITY_REASON :
           "physical viability status=$viability_status"),
       fixed_point_status=metadata(:fixed_point_status, :not_run),
       trajectory_status=metadata(:trajectory_status, :not_run),
       coverage_status,
       moduli_status=metadata(:moduli_status, :not_established),
       phase_convention=metadata(:phase_convention, "not_recorded"),
       units=metadata(:units, "not_recorded"),
       normalization=metadata(:normalization, "not_recorded"),
       source_identity=metadata(:source_identity, "not_recorded"),
       precision_bits=metadata(:precision_bits, 0),
       source_numeric_type=metadata(:source_numeric_type, "not_recorded"),
       source_precision_bits=metadata(:source_precision_bits, 0),
       target_numeric_type=metadata(:target_numeric_type,
           PILOT_TARGET_NUMERIC_TYPE),
       target_precision_bits=metadata(:target_precision_bits,
           PILOT_TARGET_PRECISION_BITS),
       conversion_status=metadata(:conversion_status, :not_attempted),
       conversion_error_bound=metadata(:conversion_error_bound, nothing),
       conversion_tolerance=metadata(:conversion_tolerance, nothing),
       conversion_comparison=metadata(:conversion_comparison, :not_attempted),
       conversion_policy_version=metadata(:conversion_policy_version,
           PILOT_CONVERSION_POLICY_VERSION),
       stored_reference_max_log10_error=get(options,
           :stored_reference_max_log10_error, nothing),
       stored_reference_sign_mismatches=get(options,
           :stored_reference_sign_mismatches, nothing),
       leading_log_gap=hierarchy.leading_log_gap,
       log_scale_span=hierarchy.log_scale_span,
       strong_hierarchy=hierarchy.heuristic_strong_hierarchy,
       search_mode=stream.search_classification,
       branch_coverage_status=coverage_status,
       branch_estimate=seed_info.estimate,
       branch_count=stream.branch_count, mask_count=stream.mask_count,
       masks_visited=stream.masks_visited,
       masks_skipped=stream.masks_skipped,
       lattice_copy_count=stream.lattice_copy_count,
       lattice_copies_visited=stream.lattice_copies_visited,
       estimated_stage_allocated_bytes=seed_info.estimated_bytes,
       max_stage_allocated_bytes=options[:max_stage_allocated_bytes])
end

function _pilot_scale_records(geom, data_dir, geometry_path, Q, L, K, hierarchy,
        seed_info, scale, previous, previous_minima, options, geometry_data=nothing)
    scaled = pilot_scaled_inputs(Q, L, K, scale;
        scale_status=options[:scale_status], geometry=geometry_data,
        volume_normalization=options[:volume_normalization],
        source_identity=get(options, :source_identity, nothing),
        configuration_digest=get(options, :configuration_digest, nothing),
        phase_convention=get(options, :phase_convention, nothing),
        units=get(options, :units, nothing),
        normalization=get(options, :normalization, nothing),
        precision_bits=get(options, :precision_bits, nothing))
    scaled_L = scaled.L
    factor = cholesky(scaled.K).L
    records = _pilot_records(seed_info.seeds, seed_info.modes, scaled.Q, scaled_L, factor;
        residual_tolerance=options[:correction_tolerance],
        max_iterations=options[:correction_iterations],
        duplicate_tolerance=options[:duplicate_tolerance],
        scale_status=scaled.scale_status,
        physical_scaling_gate_status=scaled.physical_scaling_gate_status,
        physical_control_gate_status=scaled.physical_control_gate_status)
    if previous === nothing
        _pilot_init_branch_ids!(records)
        matches = Tuple{Int, Int, Float64}[]
    else
        matches = pilot_match_records!(previous, records;
            matching_tolerance=options[:matching_tolerance])
    end
    minima = count(record -> record.classification !== nothing &&
        record.classification.negative_modes == 0, records)
    bracket_count = previous === nothing ? 0 : _pilot_mark_crossings!(
        previous, records, matches, options[:previous_scale], scale;
        zero_eigenvalue_tolerance=options[:zero_eigenvalue_tolerance],
        bracket_number=options[:bracket_number] + 1,
        previous_minima=previous_minima, current_minima=minima)
    options[:previous_scale] = scale
    options[:bracket_number] += bracket_count > 0 ? 1 : 0
    context = _pilot_context(geom, data_dir, geometry_path, scale, hierarchy,
        seed_info, options; scaled)
    reference_candidates = if scale == 1.0
        evaluator = CYAxiverse.generate.structured_charge_evaluator(scaled.Q, scaled_L)
        count(seed_info.seeds) do seed
            _pilot_screen_pass(_pilot_classify(seed, scaled.Q, scaled_L, factor, evaluator))
        end
    else
        nothing
    end
    context = merge(context, (; reference_uncorrected_screen_candidates=reference_candidates))
    (; records, context, summary=_pilot_summary_row(context, records, seed_info;
        bracket_count), minima, bracket_count, matches)
end

function _pilot_augmented_residuals(state, Q, L, K;
        scale_status::Symbol=:physical, geometry=nothing,
        volume_normalization::Symbol=:full)
    n = size(Q, 1)
    theta = @view state[1:n]
    null_vector = @view state[(n + 1):(2n)]
    scale = exp(state[end])
    scaled = pilot_scaled_inputs(Q, L, K, scale;
        scale_status, geometry, volume_normalization)
    evaluator = CYAxiverse.generate.structured_charge_evaluator(scaled.Q, scaled.L)
    derivatives = CYAxiverse.generate.structured_logshifted_derivatives!(
        evaluator, theta, scaled.Q)
    hessian_scale = max(maximum(abs, derivatives.hessian), 1.0)
    (; gradient=derivatives.gradient, hessian=derivatives.hessian,
       null_vector, hessian_scale, K=scaled.K)
end

"""Solve the optional augmented gradient/Hessian-null system for diagnostics."""
function pilot_augmented_catastrophe(initial_theta::AbstractVector{<:Real},
        initial_null_vector::AbstractVector{<:Real}, initial_scale::Real,
        Q::Matrix{Int}, L::Matrix{Float64}, K::AbstractMatrix{<:Real};
        tolerance::Float64=1e-10, max_iterations::Int=200,
        scale_status::Symbol=:physical, geometry=nothing,
        volume_normalization::Symbol=:full)
    n = size(Q, 1)
    length(initial_theta) == n || throw(DimensionMismatch("theta length mismatch"))
    length(initial_null_vector) == n || throw(DimensionMismatch("null-vector length mismatch"))
    initial = vcat(Float64.(initial_theta),
        Float64.(initial_null_vector) ./ norm(initial_null_vector), log(Float64(initial_scale)))
    function equations!(out, state)
        data = _pilot_augmented_residuals(state, Q, L, K;
            scale_status, geometry, volume_normalization)
        out[1:n] .= data.gradient
        out[(n + 1):(2n)] .= data.hessian * data.null_vector ./ data.hessian_scale
        out[end] = dot(data.null_vector, data.null_vector) - 1
        nothing
    end
    started = time_ns()
    result = nlsolve(equations!, initial; method=:trust_region,
        ftol=tolerance, xtol=tolerance, iterations=max_iterations)
    state = result.zero
    data = _pilot_augmented_residuals(state, Q, L, K;
        scale_status, geometry, volume_normalization)
    factor = Matrix(cholesky(Hermitian(Matrix{Float64}(data.K))).L)
    theta = mod.(state[1:n], 1.0)
    null_vector = state[(n + 1):(2n)]
    (; theta, null_vector, scale=exp(state[end]),
       gradient_residual=norm(data.gradient, Inf),
       null_residual=norm(data.hessian * null_vector, Inf),
       normalized_null_vector_residual=abs(dot(null_vector, null_vector) - 1),
       converged=(result.f_converged || result.x_converged) &&
           norm(data.gradient, Inf) <= tolerance &&
           norm(data.hessian * null_vector, Inf) <= tolerance &&
           abs(dot(null_vector, null_vector) - 1) <= tolerance,
       iterations=result.iterations, seconds=(time_ns() - started) / 1e9,
       hessian_eigenvalues=eigvals(Symmetric(
           factor \ data.hessian / factor')))
end

function pilot_benchmark_regression()
    benchmark = CYAxiverse.paper_benchmarks.poly102_inflation
    n5_kc = benchmark.n5_critical_scale()
    n5_at = benchmark.n5_reduced_critical_points(n5_kc)
    n8_seed = copy(benchmark.N8_BEST_X)
    n8 = benchmark.n8_degenerate_point(n8_seed)
    detuned = CYAxiverse.paper_benchmarks.n8_hilltop(n8.k + 1e-7)
    (; n5_critical_scale=n5_kc,
       n5_ratio=benchmark.n5_reduced_ratio(n5_kc),
       n5_zero_curvature=n5_at.hessian_sign[2] == 0,
       n8_critical_scale=n8.k,
       n8_gradient_residual=n8.gradient_residual,
       n8_null_residual=n8.null_residual,
       n8_zero_mode=abs(n8.eigenvalues[1]) < 1e-9,
       n8_positive_heavy_modes=all(>(0), n8.eigenvalues[2:end]),
       n8_detuned_negative_modes=count(<(0), detuned.eigenvalues),
       n8_detuned_eigenvalues=detuned.eigenvalues,
       passed=isapprox(n5_kc, 0.674506370003365; atol=1e-12) &&
           isapprox(benchmark.n5_reduced_ratio(n5_kc), 0.25; atol=1e-12) &&
           n5_at.hessian_sign[2] == 0 && isapprox(n8.k, n5_kc; atol=1e-9) &&
           n8.gradient_residual < 1e-10 && n8.null_residual < 1e-10 &&
           abs(n8.eigenvalues[1]) < 1e-9 && all(>(0), n8.eigenvalues[2:end]) &&
           count(<(0), detuned.eigenvalues) == 1)
end

function _pilot_parse_args(args)
    options = Dict{Symbol, Any}(
        :data_dir => get(ENV, "CYAXIVERSE_DATA_DIR", ""),
        :geometries => GeometryIndex[], :h11 => nothing, :max_geometries => nothing,
        :scale_grid => collect(PILOT_DEFAULT_SCALE_GRID), :max_branches => 400_000,
        :max_stage_allocated_bytes => 750_000_000,
        :negative_mode_range => nothing, :max_negative_modes => nothing,
        :scale_status => :physical, :volume_normalization => :full,
        :report => PILOT_DEFAULT_REPORT,
        :shard_dir => PILOT_DEFAULT_SHARDS, :shard_index => 1, :shard_count => 1,
        :run_id => "", :resume => false, :correction_tolerance => 1e-9,
        :correction_iterations => 100, :matching_tolerance => 0.1,
        :duplicate_tolerance => 1e-7, :zero_eigenvalue_tolerance => 1e-8,
        :benchmarks_only => false)
    valued = ("--data-dir", "--geometry", "--h11", "--max-geometries",
        "--scale-grid", "--max-branches", "--max-stage-allocated-bytes",
        "--negative-mode-range",
        "--max-negative-modes", "--scale-status", "--volume-normalization",
        "--report", "--shard-dir",
        "--shard-index", "--shard-count", "--run-id", "--correction-tolerance",
        "--correction-iterations", "--matching-tolerance", "--duplicate-tolerance",
        "--zero-eigenvalue-tolerance")
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            println("""
Usage: julia --project=. scripts/inflation_scale_continuation.jl [options]
  --data-dir PATH, --geometry H,P,F (repeatable), --h11 N, --max-geometries N
  --scale-grid a,b,c, --max-branches N
  --max-stage-allocated-bytes N (default 750000000)
  --negative-mode-range K[:K] or --max-negative-modes K
  --scale-status homotopy_only|physical|unsupported (default physical)
  --volume-normalization fixed|full (fixed is diagnostic-only; default full)
  --report PATH, --shard-dir PATH, --shard-index N, --shard-count N
  --correction-tolerance T, --correction-iterations N
  --matching-tolerance T, --duplicate-tolerance T, --zero-eigenvalue-tolerance T
  --resume, --benchmarks-only
""")
            exit(0)
        elseif arg == "--resume"
            options[:resume] = true
        elseif arg == "--benchmarks-only"
            options[:benchmarks_only] = true
        elseif arg in valued
            index == length(args) && error("missing value for $arg")
            value = args[index + 1]
            if arg == "--data-dir"
                options[:data_dir] = value
            elseif arg == "--geometry"
                parts = parse.(Int, split(value, ',')); length(parts) == 3 ||
                    error("--geometry must be H,P,F")
                push!(options[:geometries], GeometryIndex(parts...))
            elseif arg == "--h11"
                options[:h11] = parse(Int, value)
            elseif arg == "--max-geometries"
                options[:max_geometries] = parse(Int, value)
            elseif arg == "--scale-grid"
                options[:scale_grid] = pilot_parse_scale_grid(value)
            elseif arg == "--max-branches"
                options[:max_branches] = parse(Int, value)
            elseif arg == "--max-stage-allocated-bytes"
                options[:max_stage_allocated_bytes] = parse(Int, value)
            elseif arg == "--negative-mode-range"
                options[:negative_mode_range] = inflation_parse_negative_mode_range(value)
            elseif arg == "--max-negative-modes"
                options[:max_negative_modes] = parse(Int, value)
            elseif arg == "--scale-status"
                options[:scale_status] = Symbol(value)
            elseif arg == "--volume-normalization"
                options[:volume_normalization] = Symbol(value)
            elseif arg == "--report"
                options[:report] = value
            elseif arg == "--shard-dir"
                options[:shard_dir] = value
            elseif arg == "--shard-index"
                options[:shard_index] = parse(Int, value)
            elseif arg == "--shard-count"
                options[:shard_count] = parse(Int, value)
            elseif arg == "--run-id"
                options[:run_id] = value
            elseif arg == "--correction-tolerance"
                options[:correction_tolerance] = parse(Float64, value)
            elseif arg == "--correction-iterations"
                options[:correction_iterations] = parse(Int, value)
            elseif arg == "--matching-tolerance"
                options[:matching_tolerance] = parse(Float64, value)
            elseif arg == "--duplicate-tolerance"
                options[:duplicate_tolerance] = parse(Float64, value)
            elseif arg == "--zero-eigenvalue-tolerance"
                options[:zero_eigenvalue_tolerance] = parse(Float64, value)
            end
            index += 1
        else
            error("unknown option: $arg")
        end
        index += 1
    end
    options[:scale_status] in (:homotopy_only, :physical, :unsupported) ||
        throw(ArgumentError("--scale-status must be homotopy_only, physical, or unsupported"))
    options[:volume_normalization] in (:fixed, :full) ||
        throw(ArgumentError("--volume-normalization must be fixed or full"))
    options[:scale_status] == :physical && options[:volume_normalization] == :fixed &&
        throw(ArgumentError("fixed CY-volume normalization is diagnostic-only; use --scale-status unsupported"))
    options[:max_branches] > 0 || error("--max-branches must be positive")
    options[:max_stage_allocated_bytes] > 0 ||
        error("--max-stage-allocated-bytes must be positive")
    options[:correction_tolerance] > 0 || error("correction tolerance must be positive")
    options[:correction_iterations] > 0 || error("correction iterations must be positive")
    options[:matching_tolerance] > 0 || error("matching tolerance must be positive")
    options[:duplicate_tolerance] > 0 || error("duplicate tolerance must be positive")
    options[:zero_eigenvalue_tolerance] > 0 || error("zero eigenvalue tolerance must be positive")
    options[:negative_mode_range] === nothing || options[:max_negative_modes] === nothing ||
        error("use only one of --negative-mode-range and --max-negative-modes")
    options[:shard_count] > 0 && 1 <= options[:shard_index] <= options[:shard_count] ||
        error("invalid shard index/count")
    options
end

function _pilot_partition(geometries, index, count)
    [geom for (position, geom) in enumerate(geometries)
        if mod(position - 1, count) + 1 == index]
end

function run_scale_continuation(options)
    data_dir = _pilot_data_dir(
        CYAxiverse.filestructure.resolve_data_dir(options[:data_dir]))
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    geometries = pilot_select_geometries(data_dir, options[:geometries];
        h11_filter=options[:h11], max_geometries=options[:max_geometries])
    geometries = _pilot_partition(geometries, options[:shard_index], options[:shard_count])
    isempty(geometries) && throw(ArgumentError("no geometries selected"))
    report = abspath(expanduser(options[:report]))
    shard_dir = abspath(expanduser(options[:shard_dir]))
    shard_path = joinpath(shard_dir, string("inflation_scale_continuation_shard_",
        lpad(options[:shard_index], 4, '0'), "_of_", lpad(options[:shard_count], 4, '0'), ".csv"))
    if options[:resume]
        isfile(report) || throw(ArgumentError("--resume requires an existing report: $report"))
        isfile(shard_path) || throw(ArgumentError("--resume requires an existing shard: $shard_path"))
    end
    pilot_prepare_csv(report, PILOT_SUMMARY_FIELDS; append=options[:resume])
    pilot_prepare_csv(shard_path, PILOT_BRANCH_FIELDS; append=options[:resume])
    completed = options[:resume] ? _pilot_completed_scales(report) :
        Set{Tuple{Int, Int, Int, Float64}}()
    run_id = isempty(options[:run_id]) ?
        string("scale-continuation-", getpid(), "-", round(Int, time())) : options[:run_id]
    options[:run_id] = run_id
    @printf("Scale-continuation pilot: %d geometries, %d scales, %s (%s)\n",
        length(geometries), length(options[:scale_grid]), options[:scale_status],
        options[:volume_normalization])
    for geom in geometries
        geometry_path = _pilot_geometry_path(data_dir, geom)
        try
            loaded = CYAxiverse.read.oriented_potential(geom)
            Q, L, K = loaded.Q, loaded.L, loaded.K
            geometry_data = options[:scale_status] == :physical ?
                CYAxiverse.read.geometry(geom) : nothing
            if options[:scale_status] == :physical
                reference = pilot_scaled_inputs(Q, L, K, 1.0;
                    scale_status=:physical, geometry=geometry_data,
                    volume_normalization=options[:volume_normalization])
                L = reference.L
                options[:physical_domain_certificate] = reference.domain_certificate
                options[:domain_status] = reference.domain_status
                options[:domain_reason] = reference.domain_reason
                options[:physical_scaling_gate_status] =
                    reference.physical_scaling_gate_status
                options[:physical_scaling_gate_reason] =
                    reference.physical_scaling_gate_reason
                options[:physical_scaling_gate_provenance] =
                    reference.physical_scaling_gate_provenance
                options[:physical_control_gate_status] =
                    reference.physical_control_gate_status
                options[:physical_control_gate_reason] =
                    reference.physical_control_gate_reason
                options[:physical_control_gate_provenance] =
                    reference.physical_control_gate_provenance
                options[:physical_viability_status] =
                    reference.physical_viability_status
                options[:physical_viability_reason] =
                    reference.physical_viability_reason
                options[:stored_reference_max_log10_error] =
                    reference.reference_diagnostic.max_log10_error
                options[:stored_reference_sign_mismatches] =
                    reference.reference_diagnostic.sign_mismatches
            else
                options[:stored_reference_max_log10_error] = nothing
                options[:stored_reference_sign_mismatches] = nothing
            end
            hierarchy = CYAxiverse.generate.instanton_hierarchy_diagnostics(L)
            seed_info = pilot_collect_seeds(Q, L;
                max_branches=options[:max_branches],
                negative_mode_range=options[:negative_mode_range],
                max_negative_modes=options[:max_negative_modes],
                max_stage_allocated_bytes=options[:max_stage_allocated_bytes])
            previous = nothing
            previous_minima = 0
            for scale in options[:scale_grid]
                key = (geom.h11, geom.polytope, geom.frst, scale)
                key in completed && continue
                options[:previous_scale] = previous === nothing ? scale : options[:previous_scale]
                options[:bracket_number] = get(options, :bracket_number, 0)
                if seed_info.status != :completed
                    context = _pilot_context(geom, data_dir, geometry_path, scale,
                        hierarchy, seed_info, options)
                    summary = _pilot_summary_row(context, _PilotBranchRecord[], seed_info;
                        failure=seed_info.error)
                    pilot_append_csv(report, summary, PILOT_SUMMARY_FIELDS)
                    previous = nothing
                    continue
                end
                measured = @timed _pilot_scale_records(geom, data_dir, geometry_path,
                    Q, L, K, hierarchy, seed_info, scale, previous,
                    previous_minima, options, geometry_data)
                result = measured.value
                context = result.context
                summary = merge(result.summary, (; run_id, wall_seconds=measured.time,
                    allocated_bytes=measured.bytes,
                    output_bytes=Base.summarysize(result.records)))
                pilot_append_csv(report, summary, PILOT_SUMMARY_FIELDS)
                for record in result.records
                    pilot_append_csv(shard_path, _pilot_branch_row(record, context),
                        PILOT_BRANCH_FIELDS)
                end
                previous = result.records
                previous_minima = result.minima
                options[:previous_scale] = scale
            end
            println("completed ", geom.h11, ",", geom.polytope, ",", geom.frst,
                " seed_status=", seed_info.status, " seeds=", length(seed_info.seeds))
        catch error
            message = sprint(showerror, error)
            certificate = error isa PilotPhysicalDomainError ? error.certificate : nothing
            fallback_scale_status = options[:scale_status] == :physical ?
                :unsupported : options[:scale_status]
            fallback_domain_status = certificate === nothing ?
                (options[:scale_status] == :physical ? :numerical_failure :
                 get(options, :domain_status, :out_of_model)) : certificate.domain_status
            fallback_domain_reason = certificate === nothing ? message :
                certificate.domain_reason
            fallback_scale_source = fallback_scale_status == :unsupported ?
                "physical_domain_certificate_failure" :
                "generic_log10_amplitude_exponent_stretch"
            fallback_scaling_gate_status = certificate === nothing ?
                (options[:scale_status] == :physical ? :missing_evidence : :not_applicable) :
                certificate.physical_scaling_gate_status
            fallback_control_gate_status = certificate === nothing ?
                (options[:scale_status] == :physical ? :not_established : :not_applicable) :
                certificate.physical_control_gate_status
            fallback_scaling_gate_reason = certificate === nothing ?
                "physical scaling gate was not evaluated: $message" :
                certificate.physical_scaling_gate_reason
            fallback_control_gate_reason = certificate === nothing ?
                "physical control gate was not evaluated" :
                certificate.physical_control_gate_reason
            fallback_viability_status = if options[:scale_status] != :physical
                :not_applicable
            elseif fallback_scaling_gate_status == :passed
                :not_evaluated
            else
                :blocked_scaling_gate
            end
            fallback_viability_reason = fallback_viability_status == :not_applicable ?
                PILOT_NONPHYSICAL_VIABILITY_REASON :
                fallback_viability_status == :not_evaluated ?
                "physical viability was not evaluated after the geometry-level failure" :
                "physical scaling gate is not passed"
            for scale in options[:scale_grid]
                key = (geom.h11, geom.polytope, geom.frst, scale)
                key in completed && continue
                fallback = (; row_type=:scale, schema_version=PILOT_SCHEMA_VERSION,
                    run_id, data_root=data_dir, geometry_path,
                    h11=geom.h11, polytope=geom.polytope, frst=geom.frst,
                    reference_scale=1.0, sampled_scale=scale,
                    scale_grid=join(string.(options[:scale_grid]), ';'),
                    scale_source=fallback_scale_source,
                    scale_status=fallback_scale_status,
                    volume_normalization=fallback_scale_status == :unsupported &&
                        options[:volume_normalization] == :fixed ? :fixed : :none,
                    domain_certificate_version=PILOT_DOMAIN_CERTIFICATE_VERSION,
                    domain_status=fallback_domain_status,
                    domain_reason=fallback_domain_reason,
                    physical_scaling_gate_status=fallback_scaling_gate_status,
                    physical_scaling_gate_reason=fallback_scaling_gate_reason,
                    physical_scaling_gate_provenance=certificate === nothing ?
                        "scripts/inflation_scale_continuation.jl::pilot_physical_domain_certificate" :
                        certificate.physical_scaling_gate_provenance,
                    physical_control_gate_status=fallback_control_gate_status,
                    physical_control_gate_reason=fallback_control_gate_reason,
                    physical_control_gate_provenance=certificate === nothing ?
                        "scripts/inflation_scale_continuation.jl::pilot_physical_domain_certificate" :
                        certificate.physical_control_gate_provenance,
                    physical_viability_status=fallback_viability_status,
                    physical_viability_reason=fallback_viability_reason,
                    fixed_point_status=:not_run, trajectory_status=:not_run,
                    coverage_status=:not_started, moduli_status=:not_established,
                    phase_convention=certificate === nothing ? "not_recorded" :
                        certificate.phase_convention,
                    units=certificate === nothing ? "not_recorded" : certificate.units,
                    normalization=certificate === nothing ? "not_recorded" :
                        certificate.normalization,
                    source_identity=certificate === nothing ? "not_recorded" :
                        certificate.source_identity,
                    precision_bits=certificate === nothing ? 0 : certificate.precision_bits,
                    source_numeric_type=certificate === nothing ? "not_recorded" :
                        certificate.source_numeric_type,
                    source_precision_bits=certificate === nothing ? 0 :
                        certificate.source_precision_bits,
                    target_numeric_type=certificate === nothing ?
                        PILOT_TARGET_NUMERIC_TYPE : certificate.target_numeric_type,
                    target_precision_bits=certificate === nothing ?
                        PILOT_TARGET_PRECISION_BITS : certificate.target_precision_bits,
                    conversion_status=certificate === nothing ? :not_attempted :
                        certificate.conversion_status,
                    conversion_error_bound=certificate === nothing ? nothing :
                        certificate.conversion_error_bound,
                    conversion_tolerance=certificate === nothing ? nothing :
                        certificate.conversion_tolerance,
                    conversion_comparison=certificate === nothing ? :not_attempted :
                        certificate.conversion_comparison,
                    conversion_policy_version=certificate === nothing ?
                        PILOT_CONVERSION_POLICY_VERSION :
                        certificate.conversion_policy_version,
                    leading_log_gap=NaN,
                    log_scale_span=NaN, strong_hierarchy=false,
                    search_mode=:unsupported, branch_coverage_status=:failed,
                    branch_estimate=nothing, branch_count=nothing, mask_count=nothing,
                    masks_visited=nothing, masks_skipped=nothing,
                    lattice_copy_count=nothing, lattice_copies_visited=nothing,
                    failure=message)
                pilot_append_csv(report, _pilot_summary_row(fallback,
                    _PilotBranchRecord[], (; seeds=Any[]); failure=message), PILOT_SUMMARY_FIELDS)
            end
            @warn "scale-continuation geometry failed" geometry=geom error=message
        end
    end
    true
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = _pilot_parse_args(ARGS)
    if options[:benchmarks_only]
        result = pilot_benchmark_regression()
        println(result)
        result.passed || exit(1)
    else
        run_scale_continuation(options) || exit(1)
    end
end
