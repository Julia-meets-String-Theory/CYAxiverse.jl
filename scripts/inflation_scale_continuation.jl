#!/usr/bin/env julia

"""Bounded scale-continuation and catastrophe diagnostic pilot.

Generic geometry rows use the author-reconstructed physical continuation by
default, with `volume_normalization=full`. The author's fixed-volume
convention remains available with `volume_normalization=fixed`, and the old
mathematical continuation remains available explicitly with
`scale_status=homotopy_only`.

The script is intentionally script-local.  It reuses the locked orientation,
leading-branch streamer, log-shifted derivatives, and screen conventions from
`inflation_scan_common.jl`, but does not modify the fixed-potential scan.
"""

include(joinpath(@__DIR__, "inflation_scan_common.jl"))

using LinearAlgebra
using NLsolve
using Printf
using Statistics

const PILOT_SCHEMA_VERSION = "3"
const PILOT_DEFAULT_SCALE_GRID = (0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10)
const PILOT_DEFAULT_REPORT = "/private/tmp/inflation-scale-continuation/report.csv"
const PILOT_DEFAULT_SHARDS = "/private/tmp/inflation-scale-continuation/shards"

const PILOT_COMMON_FIELDS = (
    :row_type, :schema_version, :run_id, :data_root, :geometry_path,
    :h11, :polytope, :frst, :reference_scale, :sampled_scale,
    :scale_grid, :scale_source, :scale_status, :volume_normalization,
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
    :corrected_candidate_count, :minima_count, :wall_seconds,
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

function _pilot_author_potential(Q::Matrix{Int}, tau::Vector{Float64},
        kinv::Matrix{Float64}, cy_volume::Float64)
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

    result = zeros(Float64, 2, term_count)
    prefactor = 8π / cy_volume^2
    log10e = log10(exp(1.0))
    for column in 1:base_count
        charge = @view Q[:, column]
        exponent = -2π * log10e * dot(charge, tau)
        coefficient = prefactor * dot(charge, tau)
        if coefficient == 0
            result[1, column] = 0.0
            result[2, column] = -Inf
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
        exponent = -2π * log10e * dot(charge_sum, tau)
        coefficient = (8π / cy_volume^2) *
            (π * dot(qi, kinv * qj) + dot(charge_sum, tau))
        if coefficient == 0
            result[1, index] = 0.0
            result[2, index] = -Inf
        else
            result[1, index] = sign(coefficient)
            result[2, index] = log10(abs(coefficient)) + exponent
        end
        index += 1
    end
    result
end

function _pilot_reference_diagnostic(Q, L, tau, kinv, cy_volume)
    expected = _pilot_author_potential(Q, tau, kinv, cy_volume)
    size(L) == size(expected) || throw(DimensionMismatch("L dimensions disagree"))
    L_sign = sign.(L[1, :])
    expected_sign = sign.(expected[1, :])
    finite = isfinite.(L[2, :]) .& isfinite.(expected[2, :])
    compatible_nonfinite = all(finite .| (L[2, :] .== expected[2, :]))
    compatible_nonfinite || throw(ArgumentError(
        "stored potential has incompatible non-finite logs"))
    (; expected, max_log10_error=any(finite) ?
           maximum(abs.(L[2, finite] .- expected[2, finite])) : 0.0,
       sign_mismatches=count(index -> L_sign[index] != expected_sign[index],
           eachindex(L_sign)))
end

"""Apply either the legacy homotopy or the author's physical CY path."""
function pilot_scaled_inputs(Q::Matrix{Int}, L::Matrix{Float64}, K,
        scale::Real; scale_status::Symbol=:physical,
        geometry=nothing, volume_normalization::Symbol=:full)
    isfinite(scale) && scale > 0 || throw(ArgumentError("scale must be finite and positive"))
    if scale_status == :homotopy_only
        volume_normalization == :full || volume_normalization == :fixed ||
            throw(ArgumentError("volume_normalization must be :fixed or :full"))
        return (; Q, L=pilot_homotopy_scale(L, scale), K,
            scale_source="generic_log10_amplitude_exponent_stretch",
            scale_status=:homotopy_only, volume_normalization=:none)
    end
    scale_status == :physical || throw(ArgumentError(
        "scale_status must be :homotopy_only or :physical"))
    geometry === nothing && throw(ArgumentError(
        "physical scaling requires CY volume, divisor volumes, and Kinv metadata"))
    volume_normalization in (:fixed, :full) || throw(ArgumentError(
        "volume_normalization must be :fixed or :full"))
    tau = Float64.(geometry.τ_volumes)
    kinv = Matrix{Float64}(geometry.kinv)
    cy_volume = Float64(geometry.cy_volume)
    reference = _pilot_reference_diagnostic(Q, L, tau, kinv, cy_volume)
    reference.sign_mismatches == 0 || throw(ArgumentError(
        "stored potential sign normalization does not match the physical author path"))
    reference.max_log10_error <= 1e-10 || throw(ArgumentError(
        "stored potential log normalization does not match the physical author path"))
    scale_float = Float64(scale)
    scaled_tau = scale_float .* tau
    scaled_kinv = scale_float^2 .* kinv
    scaled_volume = volume_normalization == :fixed ? cy_volume :
        cy_volume * scale_float^(3 / 2)
    scaled_L = _pilot_author_potential(Q, scaled_tau, scaled_kinv, scaled_volume)
    scaled_K = Hermitian(Matrix{Float64}(K) / scale_float^2)
    (; Q, L=scaled_L, K=scaled_K, tau=scaled_tau, kinv=scaled_kinv,
       volume=scaled_volume, reference=reference.expected,
       reference_diagnostic=reference,
       scale_source="author_divisor_volume_path", scale_status=:physical,
       volume_normalization)
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
    eigen_scale = max(maximum(abs, eigenvalues), 1.0)
    (; value, gradient_norm, epsilon,
       min_eta=minimum(eta_values), max_eta=maximum(eta_values),
       abs_min_eta=minimum(abs.(eta_values)),
       hessian_min=minimum(eigenvalues), hessian_max=maximum(eigenvalues),
       abs_min_hessian=minimum(abs.(eigenvalues)),
       negative_modes=count(<(0), eigenvalues),
       zeroish_modes=count(x -> abs(x) <= 1e-10 * eigen_scale, eigenvalues),
       positive_modes=count(>(0), eigenvalues), eigenvalues)
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
    failure::String
end

function _pilot_records(seeds, modes, Q, L, factor; residual_tolerance,
        max_iterations, duplicate_tolerance, scale_status::Symbol=:homotopy_only)
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
        candidate = scale_status == :homotopy_only && classification !== nothing &&
            _pilot_screen_pass(classification)
        candidate_reason = if candidate
            "corrected point passes existing Float64 screen"
        elseif classification !== nothing && _pilot_screen_pass(classification) &&
                scale_status == :physical
            "physical-mode screen hit withheld from candidate classification"
        else
            "corrected point does not pass existing Float64 screen"
        end
        push!(records, _PilotBranchRecord(index, modes[index], copy(seeds[index]),
            copy(corrected.theta), status, corrected.residual, corrected.iterations,
            corrected.seconds, string("seed-", lpad(index, 8, '0')),
            string("branch-", lpad(index, 8, '0')), :unmatched, classification,
            false, "", "", candidate ? :corrected_candidate : :none,
            candidate_reason, failure))
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
            _pilot_screen_pass(record.classification) ||
                (record.candidate_status = :near_catastrophe_only)
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
    candidates = count(record -> record.candidate_status == :corrected_candidate, records)
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
        seed_count=length(seed_info.seeds), corrected_count=length(converged),
        matched_count=matched, lost_count=lost, new_count=new,
        duplicate_count=duplicates, correction_failed_count=failed,
        near_catastrophe_brackets=bracket_count,
        screen_candidate_count=candidates,
        corrected_candidate_count=candidates, minima_count=minima,
        wall_seconds, allocated_bytes, output_bytes, failure))
end

function _pilot_context(geom, data_dir, geometry_path, scale, hierarchy,
        seed_info, options)
    stream = seed_info.stream
    (; row_type=:scale, schema_version=PILOT_SCHEMA_VERSION,
       run_id=options[:run_id], data_root=data_dir, geometry_path,
       h11=geom.h11, polytope=geom.polytope, frst=geom.frst,
       reference_scale=1.0, sampled_scale=scale,
       scale_grid=join(string.(options[:scale_grid]), ';'),
       scale_source=options[:scale_status] == :physical ?
           "author_divisor_volume_path" :
           "generic_log10_amplitude_exponent_stretch",
       scale_status=options[:scale_status],
       volume_normalization=options[:scale_status] == :physical ?
           options[:volume_normalization] : :none,
       stored_reference_max_log10_error=get(options,
           :stored_reference_max_log10_error, nothing),
       stored_reference_sign_mismatches=get(options,
           :stored_reference_sign_mismatches, nothing),
       leading_log_gap=hierarchy.leading_log_gap,
       log_scale_span=hierarchy.log_scale_span,
       strong_hierarchy=hierarchy.heuristic_strong_hierarchy,
       search_mode=stream.search_classification,
       branch_coverage_status=seed_info.status == :completed ?
           (stream.search_classification == :complete_enumeration ?
            :complete : :partial_index_range) : seed_info.status,
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
        volume_normalization=options[:volume_normalization])
    scaled_L = scaled.L
    factor = cholesky(scaled.K).L
    records = _pilot_records(seed_info.seeds, seed_info.modes, scaled.Q, scaled_L, factor;
        residual_tolerance=options[:correction_tolerance],
        max_iterations=options[:correction_iterations],
        duplicate_tolerance=options[:duplicate_tolerance],
        scale_status=options[:scale_status])
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
        seed_info, options)
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
  --scale-status homotopy_only|physical (default physical)
  --volume-normalization fixed|full (physical mode only; default full)
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
    options[:scale_status] in (:homotopy_only, :physical) ||
        throw(ArgumentError("--scale-status must be homotopy_only or physical"))
    options[:volume_normalization] in (:fixed, :full) ||
        throw(ArgumentError("--volume-normalization must be fixed or full"))
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
            for scale in options[:scale_grid]
                key = (geom.h11, geom.polytope, geom.frst, scale)
                key in completed && continue
                fallback = (; row_type=:scale, schema_version=PILOT_SCHEMA_VERSION,
                    run_id, data_root=data_dir, geometry_path,
                    h11=geom.h11, polytope=geom.polytope, frst=geom.frst,
                    reference_scale=1.0, sampled_scale=scale,
                    scale_grid=join(string.(options[:scale_grid]), ';'),
                    scale_source=options[:scale_status] == :physical ?
                        "author_divisor_volume_path" :
                        "generic_log10_amplitude_exponent_stretch",
                    scale_status=options[:scale_status],
                    volume_normalization=options[:scale_status] == :physical ?
                        options[:volume_normalization] : :none,
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
