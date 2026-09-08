#!/usr/bin/env julia

"""
    CatastrophicInflationPopulationStudy

Framework and fail-closed preflight harness for measuring catastrophic-inflation
occurrence in a validated Uniform-FRST ensemble (Issue #133).

Enforces explicit preconditions, frozen ensemble contracts, separated phase strata,
explicit success/failure denominators, attached search recall, and strict physical
claim boundaries.
"""
module CatastrophicInflationPopulationStudy

using CYAxiverse
using LinearAlgebra
using Printf
using SHA

export SCHEMA_VERSION, TASK_ID,
    PopulationPreconditionStatus, EnsembleSpecification, SearchConfiguration,
    DenominatorAccounting, PopulationStudyResult,
    evaluate_preconditions, assert_preconditions,
    default_ensemble_specification, default_search_configuration,
    create_empty_accounting, record_search_outcome!, verify_accounting_integrity,
    run_population_preflight, run_bounded_diagnostic,
    json_value, save_report

const SCHEMA_VERSION = "catastrophic-inflation-population-study-v1"
const TASK_ID = "catastrophic-inflation-population-study"
const SOURCE_ISSUE = "Julia-meets-String-Theory/CYAxiverse.jl#133"

# --- JSON serialization without external dependencies ---

function json_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif UInt32(c) < 0x20
            @printf(io, "\\u%04x", UInt32(c))
        else
            print(io, c)
        end
    end
    String(take!(io))
end

json_value(::Nothing) = "null"
json_value(::Missing) = "null"
json_value(x::Bool) = x ? "true" : "false"
json_value(x::Integer) = string(x)
json_value(x::AbstractFloat) = isfinite(x) ? repr(x) : string('"', json_escape(string(x)), '"')
json_value(x::Symbol) = string('"', json_escape(string(x)), '"')
json_value(x::AbstractString) = string('"', json_escape(x), '"')
json_value(x::AbstractArray) = string('[', join(json_value.(collect(x)), ','), ']')
json_value(x::Tuple) = json_value(collect(x))

function json_value(x::AbstractDict)
    pairs_sorted = sort!(collect(pairs(x)), by=p -> string(p[1]))
    string('{', join((string(json_value(string(k)), ':', json_value(v)) for (k, v) in pairs_sorted), ','), '}')
end

function json_value(x::NamedTuple)
    keys_sorted = sort!(collect(keys(x)), by=string)
    string('{', join((string(json_value(string(k)), ':', json_value(getfield(x, k))) for k in keys_sorted), ','), '}')
end

# --- Data Structures ---

"""
    PopulationPreconditionStatus

Records the verification gate status of each explicit precondition required before
launching a scientific population study under Issue #133.
"""
struct PopulationPreconditionStatus
    ensemble_contract_status::Symbol       # :passed or :blocked_unvalidated (Issue #115)
    recall_calibration_status::Symbol     # :calibrated or :uncalibrated (Issue #113)
    classification_stability_status::Symbol # :stable or :unverified (Issue #130/131)
    scale_status::Symbol                   # :homotopy_only or :physical
    physical_control_gate::Symbol          # :not_established or :established
    can_execute_production::Bool
    can_execute_diagnostic::Bool
    blocking_reasons::Vector{String}
end

function json_value(x::PopulationPreconditionStatus)
    json_value((
        ensemble_contract_status=x.ensemble_contract_status,
        recall_calibration_status=x.recall_calibration_status,
        classification_stability_status=x.classification_stability_status,
        scale_status=x.scale_status,
        physical_control_gate=x.physical_control_gate,
        can_execute_production=x.can_execute_production,
        can_execute_diagnostic=x.can_execute_diagnostic,
        blocking_reasons=x.blocking_reasons,
    ))
end

"""
    EnsembleSpecification

Specifies the authoritative target ensemble and keeps it strictly distinct from
biased comparison ensembles (e.g. NTFE, fast-heuristic / Glimmers).
"""
struct EnsembleSpecification
    target_ensemble::Symbol              # :Uniform_FRST
    sampling_measure::Symbol             # :uniform_frst_canonical_two_face
    deduplication_policy::Symbol         # :canonical_two_face_dedup
    comparison_ensembles::Vector{Symbol} # [:ntfe, :glimmers_fast]
    seed::Int
    favorable::Bool
    lattice::Symbol                      # :N
end

function json_value(x::EnsembleSpecification)
    json_value((
        target_ensemble=x.target_ensemble,
        sampling_measure=x.sampling_measure,
        deduplication_policy=x.deduplication_policy,
        comparison_ensembles=x.comparison_ensembles,
        seed=x.seed,
        favorable=x.favorable,
        lattice=x.lattice,
    ))
end

"""
    SearchConfiguration

Frozen configuration for the local catastrophe search and bisection refinement.
"""
struct SearchConfiguration
    k_min::Float64
    k_max::Float64
    k_steps::Int
    tolerance::Float64
    bisection_tolerance::Float64
    precision_bits::Int
    gradient_tolerance::Float64
    hessian_tolerance::Float64
    derivative_tolerance::Float64
end

function json_value(x::SearchConfiguration)
    json_value((
        k_min=x.k_min,
        k_max=x.k_max,
        k_steps=x.k_steps,
        tolerance=x.tolerance,
        bisection_tolerance=x.bisection_tolerance,
        precision_bits=x.precision_bits,
        gradient_tolerance=x.gradient_tolerance,
        hessian_tolerance=x.hessian_tolerance,
        derivative_tolerance=x.derivative_tolerance,
    ))
end

"""
    DenominatorAccounting

Tracks explicit success and failure denominators for a specific phase stratum.
Ensures every candidate count has an unambiguous denominator and conservation.
"""
mutable struct DenominatorAccounting
    stratum::Symbol                         # :zero_phase, :specified_random_phases, :deliberately_optimized_phases
    total_geometries_attempted::Int
    geometries_loaded::Int
    triangulations_attempted::Int
    triangulations_valid::Int
    searches_evaluated::Int
    catastrophes_found::Int
    cusp_count::Int
    fold_count::Int
    unresolved_count::Int
    sixty_efolds_count::Union{Int, Missing} # Missing under homotopy_only
    search_recall_attached::String          # Explicit recall attached to result, e.g. "0/0 (uncalibrated)"
    physical_stabilized_models::Int        # Must remain 0 while physical_control_gate is :not_established
    failure_breakdown::Dict{String, Int}
end

function json_value(x::DenominatorAccounting)
    json_value((
        stratum=x.stratum,
        total_geometries_attempted=x.total_geometries_attempted,
        geometries_loaded=x.geometries_loaded,
        triangulations_attempted=x.triangulations_attempted,
        triangulations_valid=x.triangulations_valid,
        searches_evaluated=x.searches_evaluated,
        catastrophes_found=x.catastrophes_found,
        cusp_count=x.cusp_count,
        fold_count=x.fold_count,
        unresolved_count=x.unresolved_count,
        sixty_efolds_count=x.sixty_efolds_count,
        search_recall_attached=x.search_recall_attached,
        physical_stabilized_models=x.physical_stabilized_models,
        failure_breakdown=x.failure_breakdown,
    ))
end

"""
    PopulationStudyResult

The comprehensive audit or study outcome, recording all preconditions, specifications,
and stratified denominator accounting.
"""
struct PopulationStudyResult
    schema_version::String
    task_id::String
    source_issue::String
    preconditions::PopulationPreconditionStatus
    ensemble::EnsembleSpecification
    search_config::SearchConfiguration
    strata_accounting::Dict{Symbol, DenominatorAccounting}
    claim_boundary::NamedTuple
    status::Symbol
end

function json_value(x::PopulationStudyResult)
    json_value((
        schema_version=x.schema_version,
        task_id=x.task_id,
        source_issue=x.source_issue,
        preconditions=x.preconditions,
        ensemble=x.ensemble,
        search_config=x.search_config,
        strata_accounting=x.strata_accounting,
        claim_boundary=x.claim_boundary,
        status=x.status,
    ))
end

# --- Precondition and Gate Evaluation ---

"""
    evaluate_preconditions(; kwargs...)

Evaluate the four normative gates required by Issue #133.
Defaults to the current verified state of the repository.
"""
function evaluate_preconditions(;
        ensemble_contract_validated::Bool=false,
        recall_calibrated::Bool=false,
        classification_stable::Bool=true,
        scale_status::Symbol=:homotopy_only,
        physical_control_gate::Symbol=:not_established)

    blocking = String[]

    ensemble_status = ensemble_contract_validated ? :passed : :blocked_unvalidated
    if !ensemble_contract_validated
        push!(blocking, "Issue #115: Uniform-FRST ensemble contract is not validated.")
    end

    recall_status = recall_calibrated ? :calibrated : :uncalibrated
    if !recall_calibrated
        push!(blocking, "Issue #113: High-h11 search recall is not calibrated against nonzero positive references (historical recall 0/0).")
    end

    classification_status = classification_stable ? :stable : :unverified
    if !classification_stable
        push!(blocking, "Catastrophe classification or refinement is unstable.")
    end

    if scale_status !== :physical
        push!(blocking, "Scale parameter is homotopy_only; physical divisor-volume transformation is not established.")
    end

    if physical_control_gate !== :established
        push!(blocking, "physical_control_gate is not_established; physical viability claims are prohibited.")
    end

    can_production = ensemble_contract_validated && recall_calibrated && classification_stable &&
                     (scale_status === :physical) && (physical_control_gate === :established)
    can_diagnostic = classification_stable

    PopulationPreconditionStatus(
        ensemble_status,
        recall_status,
        classification_status,
        scale_status,
        physical_control_gate,
        can_production,
        can_diagnostic,
        blocking
    )
end

"""
    assert_preconditions(status::PopulationPreconditionStatus; allow_diagnostic_mode=false)

Fails closed if the production scientific run is attempted while preconditions are unfulfilled.
"""
function assert_preconditions(status::PopulationPreconditionStatus; allow_diagnostic_mode::Bool=false)
    if !status.can_execute_production
        if allow_diagnostic_mode && status.can_execute_diagnostic
            return true
        end
        reasons = join([" - " * r for r in status.blocking_reasons], "\n")
        error("CatastrophicInflationPopulationStudy: Scientific population study cannot proceed because preconditions are not met:\n$reasons")
    end
    true
end

# --- Defaults ---

function default_ensemble_specification(; seed::Int=42)
    EnsembleSpecification(
        :Uniform_FRST,
        :uniform_frst_canonical_two_face,
        :canonical_two_face_dedup,
        [:ntfe, :glimmers_fast],
        seed,
        true,
        :N
    )
end

function default_search_configuration()
    SearchConfiguration(
        0.5,     # k_min
        1.5,     # k_max
        101,     # k_steps
        1e-8,    # tolerance
        1e-12,   # bisection_tolerance
        120,     # precision_bits
        1e-8,    # gradient_tolerance
        1e-8,    # hessian_tolerance
        1e-8     # derivative_tolerance
    )
end

function create_empty_accounting(stratum::Symbol; recall_str::String="0/0 (uncalibrated)")
    DenominatorAccounting(
        stratum,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        missing,
        recall_str,
        0,
        Dict{String, Int}()
    )
end

"""
    verify_accounting_integrity(acc::DenominatorAccounting)

Verify that accounting numbers obey all conservation laws.
"""
function verify_accounting_integrity(acc::DenominatorAccounting)
    acc.catastrophes_found == (acc.cusp_count + acc.fold_count + acc.unresolved_count) ||
        throw(ArgumentError("Catastrophe count $(acc.catastrophes_found) does not equal sum of cusp ($(acc.cusp_count)), fold ($(acc.fold_count)), and unresolved ($(acc.unresolved_count))"))

    acc.searches_evaluated <= acc.triangulations_valid ||
        throw(ArgumentError("Searches evaluated ($(acc.searches_evaluated)) exceeds valid triangulations ($(acc.triangulations_valid))"))

    acc.triangulations_valid <= acc.triangulations_attempted ||
        throw(ArgumentError("Valid triangulations ($(acc.triangulations_valid)) exceeds attempted ($(acc.triangulations_attempted))"))

    acc.geometries_loaded <= acc.total_geometries_attempted ||
        throw(ArgumentError("Loaded geometries ($(acc.geometries_loaded)) exceeds total attempted ($(acc.total_geometries_attempted))"))

    true
end

"""
    record_search_outcome!(acc::DenominatorAccounting, outcome::Symbol; failure_reason=nothing)

Increment appropriate accounting counters for an evaluated geometry or triangulation search.
"""
function record_search_outcome!(acc::DenominatorAccounting, outcome::Symbol; failure_reason=nothing)
    if outcome === :geometry_attempted
        acc.total_geometries_attempted += 1
    elseif outcome === :geometry_loaded
        acc.geometries_loaded += 1
    elseif outcome === :triangulation_attempted
        acc.triangulations_attempted += 1
    elseif outcome === :triangulation_valid
        acc.triangulations_valid += 1
    elseif outcome === :search_evaluated
        acc.searches_evaluated += 1
    elseif outcome === :cusp
        acc.catastrophes_found += 1
        acc.cusp_count += 1
    elseif outcome === :fold
        acc.catastrophes_found += 1
        acc.fold_count += 1
    elseif outcome === :unresolved
        acc.catastrophes_found += 1
        acc.unresolved_count += 1
    elseif outcome === :no_catastrophe
        # search evaluated, but no catastrophe found
    elseif outcome === :failure
        if failure_reason !== nothing
            reason = string(failure_reason)
            acc.failure_breakdown[reason] = get(acc.failure_breakdown, reason, 0) + 1
        end
    else
        throw(ArgumentError("Unknown search outcome: $outcome"))
    end
end

"""
    run_population_preflight(; kwargs...)

Execute a read-only audit of the population study preconditions and return
a PopulationStudyResult recording the fail-closed status.
"""
function run_population_preflight(;
        ensemble_contract_validated::Bool=false,
        recall_calibrated::Bool=false,
        classification_stable::Bool=true,
        seed::Int=42)

    preconditions = evaluate_preconditions(;
        ensemble_contract_validated,
        recall_calibrated,
        classification_stable,
        scale_status=:homotopy_only,
        physical_control_gate=:not_established
    )

    ensemble = default_ensemble_specification(; seed)
    search_config = default_search_configuration()

    strata = Dict{Symbol, DenominatorAccounting}(
        :zero_phase => create_empty_accounting(:zero_phase),
        :specified_random_phases => create_empty_accounting(:specified_random_phases),
        :deliberately_optimized_phases => create_empty_accounting(:deliberately_optimized_phases),
    )

    claim_boundary = (
        fixed_saxions=true,
        scale_status=:homotopy_only,
        physical_control_gate=:not_established,
        moduli_stabilization=:not_established,
        authoritative_population_claims=:prohibited_pending_preconditions,
        conflation_rule="Deliberately optimized phase searches must not be reported as random-phase prevalence."
    )

    status = preconditions.can_execute_production ? :authorized : :blocked_by_preconditions

    PopulationStudyResult(
        SCHEMA_VERSION,
        TASK_ID,
        SOURCE_ISSUE,
        preconditions,
        ensemble,
        search_config,
        strata,
        claim_boundary,
        status
    )
end

"""
    run_bounded_diagnostic(reference_models; seed=42)

Run a bounded diagnostic using reference models (e.g. N=5 and test fold) to verify
that catastrophe diagnostics and denominator accounting operate as expected in the
diagnostic harness without violating the preconditions.
"""
function run_bounded_diagnostic(reference_models::Vector; seed::Int=42)
    preflight = run_population_preflight(; seed)
    acc = preflight.strata_accounting[:zero_phase]

    for model in reference_models
        record_search_outcome!(acc, :geometry_attempted)
        record_search_outcome!(acc, :geometry_loaded)
        record_search_outcome!(acc, :triangulation_attempted)
        record_search_outcome!(acc, :triangulation_valid)
        record_search_outcome!(acc, :search_evaluated)

        diag = CYAxiverse.paper_benchmarks.local_catastrophe_diagnostic(
            model.theta, model.Q, model.amplitudes, model.metric;
            phases=get(model, :phases, zeros(length(model.amplitudes))),
            precision_bits=120, tolerance=1e-8
        )

        if diag.classification === :cusp
            record_search_outcome!(acc, :cusp)
        elseif diag.classification === :fold
            record_search_outcome!(acc, :fold)
        elseif diag.classification === :unresolved
            record_search_outcome!(acc, :unresolved)
        else
            record_search_outcome!(acc, :no_catastrophe)
        end
    end

    verify_accounting_integrity(acc)
    preflight
end

"""
    save_report(result::PopulationStudyResult, path::String)

Write the structured JSON report to `path`.
"""
function save_report(result::PopulationStudyResult, path::String)
    open(path, "w") do io
        println(io, json_value(result))
    end
end

function main(args=ARGS)
    report_path = nothing
    run_diag = false
    enforce_gates = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--report" && i < length(args)
            i += 1
            report_path = args[i]
        elseif arg == "--diagnostic"
            run_diag = true
        elseif arg == "--enforce-gates"
            enforce_gates = true
        end
        i += 1
    end

    bench = CYAxiverse.paper_benchmarks
    poly102 = bench.poly102_inflation

    kc5 = poly102.n5_critical_scale()
    ratio5 = poly102.n5_reduced_ratio(kc5)

    reference_models = [
        (
            theta=[π],
            Q=reshape([1, 2], 1, 2),
            amplitudes=[1.0, ratio5],
            metric=reshape([1.0], 1, 1),
            phases=zeros(2),
        ),
        (
            theta=[0.0],
            Q=reshape([1, 2], 1, 2),
            amplitudes=[2.0, 1.0],
            metric=reshape([1.0], 1, 1),
            phases=[π / 2, -π / 2],
        )
    ]

    result = run_diag ?
        run_bounded_diagnostic(reference_models) :
        run_population_preflight()

    if report_path !== nothing
        save_report(result, report_path)
        println("Report written to: $report_path")
    else
        println(json_value(result))
    end

    if enforce_gates
        assert_preconditions(result.preconditions)
    end
end

end # module CatastrophicInflationPopulationStudy

if abspath(PROGRAM_FILE) == @__FILE__
    using .CatastrophicInflationPopulationStudy
    CatastrophicInflationPopulationStudy.main()
end
