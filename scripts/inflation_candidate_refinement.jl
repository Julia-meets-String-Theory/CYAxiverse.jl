#!/usr/bin/env julia

"""Bounded generic stationary-point correction, precision audit, and flow scan.

This driver owns search policy and provenance.  The numerical correction and
diagnostic/flow boundaries live in `CYAxiverse.inflation_points`.  It does
not claim that a finite search is exhaustive.
"""

using CYAxiverse
using LinearAlgebra
using Printf
using Random

const GeometryIndex = CYAxiverse.structs.GeometryIndex
const POINTS = CYAxiverse.inflation_points
const REFINEMENT_SCHEMA_VERSION = "3"
const FLOW_SCHEMA_VERSION = "1"

const REFINEMENT_FIELDS = (
    :row_type, :schema_version, :run_id, :data_dir, :code_commit,
    :max_branches, :negative_mode_range, :random_starts, :random_seed,
    :max_points, :float_tolerance, :high_tolerance, :duplicate_tolerance,
    :working_basis, :physical_basis, :physical_vectors, :scale_status,
    :h11, :polytope, :frst,
    :search_status, :coverage_status, :refinement_status, :search_mode, :branch_source,
    :branch_estimate, :branch_count, :mask_count, :masks_visited,
    :branch_seed_index, :leading_negative_modes, :seed_theta,
    :float_correction_status, :float_correction_residual,
    :float_correction_iterations, :float_value, :float_gradient_residual,
    :float_gradient_norm, :float_epsilon, :float_min_eta,
    :float_negative_modes, :float_zeroish_modes, :float_positive_modes,
    :high_precision_bits, :high_correction_status,
    :high_correction_residual, :high_correction_iterations, :high_value,
    :high_gradient_residual, :high_gradient_norm, :high_epsilon,
    :high_min_eta, :high_negative_modes, :high_zeroish_modes,
    :high_positive_modes, :residual_agreement, :inertia_agreement,
    :accepted, :candidate_status, :candidate_reason, :wall_seconds,
    :allocated_bytes, :output_bytes, :failure)

const FLOW_FIELDS = (
    :row_type, :schema_version, :run_id, :data_dir, :code_commit,
    :h11, :polytope, :frst, :candidate_index, :branch_seed_index,
    :leading_negative_modes, :seed_theta, :float_theta, :high_theta,
    :precision_bits, :working_basis, :physical_basis, :scale_status,
    :coordinate_chart,
    :mass_mode_index, :displacement_sign, :mass_eigenvalue,
    :flow_theta_initial, :flow_theta_final,
    :flow_status, :flow_end_event, :flow_efolds, :flow_slow_roll_efolds,
    :flow_entry_efolds, :flow_exit_efolds, :flow_max_efolds, :flow_step,
    :flow_accepted, :flow_reason, :flow_error, :flow_steps)

function _csv_field(value)
    value === nothing && return ""
    value isa AbstractFloat && !isfinite(value) && return string(value)
    value isa AbstractArray && return _csv_field(join(string.(value), ';'))
    text = string(value)
    occursin(r"[,\n\"]", text) ? "\"" * replace(text, "\"" => "\"\"") * "\"" : text
end

function _write_csv(path::AbstractString, rows, fields=REFINEMENT_FIELDS)
    open(path, "w") do io
        println(io, join(string.(fields), ","))
        for row in rows
            println(io, join((_csv_field(getproperty(row, field))
                for field in fields), ","))
        end
    end
    path
end

function _parse_range(value::AbstractString)
    parts = split(value, ':')
    length(parts) in (1, 2) || error("negative-mode range must be K or K:K")
    first_mode = parse(Int, parts[1])
    last_mode = length(parts) == 1 ? first_mode : parse(Int, parts[2])
    first_mode:last_mode
end

function _parse_geometry(value::AbstractString)
    parts = parse.(Int, split(value, ','))
    length(parts) == 3 || error("geometry must be h11,polytope,frst")
    GeometryIndex(parts...)
end

function _screen_pass(diagnostic)
    diagnostic !== nothing && diagnostic.value > 0 &&
        diagnostic.negative_modes > 0 && diagnostic.epsilon < 1 &&
        abs(minimum(diagnostic.eta_values)) < 1
end

function _periodic_distance(left::AbstractVector, right::AbstractVector)
    maximum(min.(abs.(left .- right), 1 .- abs.(left .- right)))
end

function _search_metadata(Q, L, negative_mode_range)
    selected = CYAxiverse.generate.LQtilde(Q, L)
    signs = @view selected.Ltilde[1, :]
    mask_count = CYAxiverse.generate._leading_mask_count(
        size(Q, 1), signs, negative_mode_range)
    determinant = CYAxiverse.generate._leading_det_qtilde(selected)
    (; selected, mask_count, branch_estimate=mask_count * determinant,
       determinant)
end

function _collect_exact_seeds(Q, L; max_branches::Int,
        negative_mode_range=nothing)
    metadata = _search_metadata(Q, L, negative_mode_range)
    seeds = Vector{Vector{Float64}}()
    modes = Int[]
    started = time_ns()
    try
        report = CYAxiverse.generate.foreach_leading_critical_branch(
            metadata.selected; max_branches, negative_mode_range) do theta, mode
            push!(seeds, copy(theta))
            push!(modes, mode)
        end
        return merge(metadata, (; seeds, modes, report, search_status=:completed,
            coverage_status=report.search_classification == :complete_enumeration ?
                :complete : :low_index, error="",
            seconds=(time_ns() - started) / 1e9))
    catch error
        message = sprint(showerror, error)
        status = occursin("leading branch enumeration would create", message) ?
            :resource_capped : :failed
        return merge(metadata, (; seeds, modes, report=nothing,
            search_status=status, coverage_status=status, error=message,
            seconds=(time_ns() - started) / 1e9))
    end
end

function _diagnostic_fields(diagnostic, prefix::AbstractString)
    diagnostic === nothing && return NamedTuple{
        (Symbol(prefix, "_value"), Symbol(prefix, "_gradient_residual"),
         Symbol(prefix, "_gradient_norm"), Symbol(prefix, "_epsilon"),
         Symbol(prefix, "_min_eta"), Symbol(prefix, "_negative_modes"),
         Symbol(prefix, "_zeroish_modes"), Symbol(prefix, "_positive_modes"))}(
        (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing))
    NamedTuple{
        (Symbol(prefix, "_value"), Symbol(prefix, "_gradient_residual"),
         Symbol(prefix, "_gradient_norm"), Symbol(prefix, "_epsilon"),
         Symbol(prefix, "_min_eta"), Symbol(prefix, "_negative_modes"),
         Symbol(prefix, "_zeroish_modes"), Symbol(prefix, "_positive_modes"))}(
        (diagnostic.value, diagnostic.gradient_residual, diagnostic.gradient_norm,
         diagnostic.epsilon, minimum(diagnostic.eta_values),
         diagnostic.negative_modes, diagnostic.zeroish_modes,
         diagnostic.positive_modes))
end

function _point_row(geom, search, source::Symbol, index::Int, mode::Int,
        seed, comparison, precision_bits, settings; duplicate=false,
        refinement_status=:completed, wall_seconds=0.0,
        allocated_bytes=0, output_bytes=0, failure="")
    float = comparison.float_correction
    high = comparison.high_correction
    float_fields = _diagnostic_fields(comparison.float_diagnostics, "float")
    high_fields = _diagnostic_fields(comparison.high_diagnostics, "high")
    float_screen = _screen_pass(comparison.float_diagnostics)
    high_screen = _screen_pass(comparison.high_diagnostics)
    accepted = comparison.accepted && high_screen && !duplicate
    candidate_status = duplicate ? :duplicate : (accepted ? :refined_candidate :
        (float_screen ? :screen_candidate : :none))
    candidate_reason = duplicate ? "torus-distance duplicate of an earlier corrected point" :
        accepted ?
        "Float64 and arbitrary-precision correction and inertia agree" :
        (float_screen ? "Float64 screen hit not accepted at precision boundary" :
         "corrected point does not pass the screening thresholds")
    merge((; row_type=:point, schema_version=REFINEMENT_SCHEMA_VERSION,
        run_id=settings.run_id, data_dir=settings.data_dir,
        code_commit=settings.code_commit, max_branches=settings.max_branches,
        negative_mode_range=settings.negative_mode_range,
        random_starts=settings.random_starts, random_seed=settings.random_seed,
        max_points=settings.max_points, float_tolerance=settings.float_tolerance,
        high_tolerance=settings.high_tolerance,
        duplicate_tolerance=settings.duplicate_tolerance,
        working_basis=:periodic_string, physical_basis=:mass_eigenbasis,
        physical_vectors=:deferred, scale_status=:unknown,
        h11=geom.h11, polytope=geom.polytope, frst=geom.frst,
        search_status=search.search_status, coverage_status=search.coverage_status,
        refinement_status,
        search_mode=search.report === nothing ? :not_completed :
            search.report.search_classification, branch_source=source,
        branch_estimate=search.branch_estimate,
        branch_count=search.report === nothing ? nothing : search.report.branch_count,
        mask_count=search.report === nothing ? search.mask_count : search.report.mask_count,
        masks_visited=search.report === nothing ? nothing : search.report.masks_visited,
        branch_seed_index=index, leading_negative_modes=mode, seed_theta=seed,
        float_correction_status=float.status,
        float_correction_residual=float.residual,
        float_correction_iterations=float.iterations,
        high_precision_bits=precision_bits,
        high_correction_status=high.status,
        high_correction_residual=high.residual,
        high_correction_iterations=high.iterations,
        residual_agreement=comparison.residual_agreement,
        inertia_agreement=comparison.inertia_agreement, accepted,
        candidate_status, candidate_reason, wall_seconds, allocated_bytes,
        output_bytes, failure), float_fields, high_fields)
end

function _random_seeds(h11::Int, count::Int, seed::Int)
    rng = MersenneTwister(seed)
    [rand(rng, h11) for _ in 1:count]
end

function _current_code_commit()
    get(ENV, "CYAXIVERSE_CODE_COMMIT", try
        readchomp(`git rev-parse HEAD`)
    catch
        "unrecorded"
    end)
end

function _refine_geometry(geom; max_branches::Int=100_000,
        negative_mode_range=nothing, random_starts::Int=0,
        random_seed::Int=20260810, precision_bits::Int=256,
        float_tolerance::Float64=1e-10, high_tolerance::Float64=1e-40,
        duplicate_tolerance::Float64=1e-7, max_points::Int=100_000,
        settings=(; run_id="", data_dir=get(ENV, "CYAXIVERSE_DATA_DIR", ""),
            code_commit="unrecorded", max_branches, negative_mode_range,
            random_starts, random_seed, max_points, float_tolerance,
            high_tolerance, duplicate_tolerance))
    loaded = CYAxiverse.read.oriented_potential(geom)
    Q, L, K = loaded.Q, loaded.L, loaded.K
    search = _collect_exact_seeds(Q, L; max_branches, negative_mode_range)
    rows = NamedTuple[]
    candidates = NamedTuple[]
    refinement_status = search.search_status == :completed ? :completed : :not_started
    if search.search_status == :completed
        seeds = [(seed, search.modes[index], :exact, index)
            for (index, seed) in enumerate(search.seeds)]
        for (index, seed) in enumerate(_random_seeds(size(Q, 1), random_starts, random_seed))
            push!(seeds, (seed, -1, :stochastic, index))
        end
        length(seeds) > max_points && (refinement_status = :resource_capped)
        retained = Vector{Vector{Float64}}()
        for (index, (seed, mode, source, source_index)) in enumerate(seeds)
            index > max_points && break
            started = time_ns()
            measured = @timed POINTS.compare_precision(seed, Q, L, K;
                precision_bits, float_residual_tolerance=float_tolerance,
                high_residual_tolerance=high_tolerance)
            comparison = measured.value
            duplicate = false
            if comparison.float_correction.status == :converged
                duplicate = any(point -> _periodic_distance(
                    point, comparison.float_correction.theta) <= duplicate_tolerance,
                    retained)
                duplicate || push!(retained, comparison.float_correction.theta)
            end
            row = _point_row(geom, search, source, source_index, mode, seed,
                comparison, precision_bits, settings; duplicate,
                refinement_status,
                wall_seconds=(time_ns() - started) / 1e9,
                allocated_bytes=measured.bytes,
                output_bytes=Base.summarysize(comparison))
            push!(rows, row)
            if row.candidate_status == :refined_candidate
                push!(candidates, (; row_index=length(rows), seed=copy(seed), mode,
                    source, source_index, comparison))
            end
        end
    end
    (; search=merge(search, (; refinement_status)), rows, candidates,
        inputs=(Q=Q, L=L, K=K))
end

function _flow_acceptance(flow; min_efolds::Real,
        require_finite_exit::Bool)
    flow.status == :failed && return (false, "flow failed")
    flow.slow_roll_efolds < min_efolds &&
        return (false, "insufficient slow-roll e-folds")
    require_finite_exit && flow.end_event == :max_efolds &&
        return (false, "slow-roll window reached max_efolds without exit")
    if flow.end_event == :max_efolds
        return (true, "at least the requested slow-roll e-folds; finite exit unobserved")
    end
    (true, "slow-roll window meets the requested e-fold threshold")
end

function _flow_row(geom, candidate, candidate_index::Int, flow, accepted::Bool,
        reason::AbstractString, settings, precision_bits::Int)
    comparison = candidate.comparison
    high = comparison.high_correction
    (; row_type=:flow, schema_version=FLOW_SCHEMA_VERSION,
        run_id=settings.run_id, data_dir=settings.data_dir,
        code_commit=settings.code_commit, h11=geom.h11,
        polytope=geom.polytope, frst=geom.frst, candidate_index,
        branch_seed_index=candidate.source_index,
        leading_negative_modes=candidate.mode, seed_theta=candidate.seed,
        float_theta=comparison.float_correction.theta, high_theta=high.theta,
        precision_bits, working_basis=:periodic_string,
        physical_basis=:mass_eigenbasis,
        scale_status=:unknown,
        coordinate_chart=get(flow, :coordinate_chart, nothing),
        mass_mode_index=get(flow, :mode_index, nothing),
        displacement_sign=get(flow, :displacement_sign, nothing),
        mass_eigenvalue=get(flow, :mass_eigenvalue, nothing),
        flow_theta_initial=get(flow, :theta_initial, nothing),
        flow_theta_final=get(flow, :theta_final, nothing),
        flow_status=get(flow, :status, :failed),
        flow_end_event=get(flow, :end_event, :failed),
        flow_efolds=get(flow, :efolds, zero(BigFloat)),
        flow_slow_roll_efolds=get(flow, :slow_roll_efolds, zero(BigFloat)),
        flow_entry_efolds=get(flow, :entry_efolds, nothing),
        flow_exit_efolds=get(flow, :exit_efolds, nothing),
        flow_max_efolds=get(flow, :max_efolds, nothing),
        flow_step=get(flow, :step, nothing), flow_accepted=accepted,
        flow_reason=reason, flow_error=get(flow, :error, ""),
        flow_steps=get(flow, :steps, nothing))
end

function _flow_geometry(geom, refinement; precision_bits::Int,
        min_efolds::Real=50, max_efolds::Real=60, step::Real=1e-3,
        displacement::Real=1e-8, require_finite_exit::Bool=false,
        settings)
    inputs = refinement.inputs
    context = POINTS.prepare_context(inputs.Q, inputs.L, inputs.K;
        precision_bits)
    rows = NamedTuple[]
    qualified = 0
    for (candidate_index, candidate) in enumerate(refinement.candidates)
        hilltop = candidate.comparison.high_correction.theta
        diagnostic = candidate.comparison.high_diagnostics
        mass = POINTS.mass_eigenbasis(context, hilltop; vectors=true)
        negative_indices = findall(value -> value < -diagnostic.zero_tolerance,
            mass.eigenvalues)
        if isempty(negative_indices)
            flow = (; status=:no_negative_mass_mode, error="", end_event=:none,
                coordinate_chart=:canonical_cholesky, mode_index=nothing,
                displacement_sign=nothing, mass_eigenvalue=nothing,
                theta_initial=nothing, theta_final=nothing, efolds=zero(BigFloat),
                slow_roll_efolds=zero(BigFloat), entry_efolds=nothing,
                exit_efolds=nothing, max_efolds=BigFloat(max_efolds),
                step=BigFloat(step), steps=0)
            push!(rows, _flow_row(geom, candidate, candidate_index, flow, false,
                "no negative physical mass mode", settings, precision_bits))
            continue
        end
        for mode_index in negative_indices
            for displacement_sign in (-1, 1)
                flow = POINTS.gradient_flow(context, hilltop;
                    displacement, displacement_sign, mode_index, mass_basis=mass,
                    max_efolds, step)
                accepted, reason = _flow_acceptance(flow;
                    min_efolds, require_finite_exit)
                accepted && (qualified += 1)
                push!(rows, _flow_row(geom, candidate, candidate_index, flow,
                    accepted, reason, settings, precision_bits))
            end
        end
    end
    (; rows, qualified)
end

"""Run candidate refinement and physical-mode flow checks for one geometry.

The refinement stage remains finite-search and records its coverage.  Only
precision-approved stationary points enter the flow stage.  Each negative
physical mass mode is tested on both displacement sides because eigenvector
signs are arbitrary and different tachyonic modes can have different flow
histories.
"""
function scan_geometry_for_inflation(geom; max_branches::Int=100_000,
        negative_mode_range=nothing, random_starts::Int=0,
        random_seed::Int=20260810, precision_bits::Int=256,
        float_tolerance::Float64=1e-10, high_tolerance::Float64=1e-40,
        max_points::Int=100_000, min_efolds::Real=50,
        max_efolds::Real=60, flow_step::Real=1e-3,
        flow_displacement::Real=1e-8, require_finite_exit::Bool=false,
        settings=nothing)
    settings === nothing && (settings=(;
        run_id=string("inflation-scan-", getpid(), "-", round(Int, time())),
        data_dir=get(ENV, "CYAXIVERSE_DATA_DIR", ""),
        code_commit=_current_code_commit(), max_branches,
        negative_mode_range, random_starts, random_seed, max_points,
        float_tolerance, high_tolerance, duplicate_tolerance=1e-7))
    refinement = _refine_geometry(geom; max_branches, negative_mode_range,
        random_starts, random_seed, precision_bits, float_tolerance,
        high_tolerance, max_points, settings)
    flow = _flow_geometry(geom, refinement; precision_bits, min_efolds,
        max_efolds, step=flow_step, displacement=flow_displacement,
        require_finite_exit, settings)
    (; geometry=geom, refinement, flow)
end

function _summary_row(geom, search, rows, settings; wall_seconds=0.0,
        allocated_bytes=0, output_bytes=0)
    refined = count(row -> row.candidate_status == :refined_candidate, rows)
    screened = count(row -> row.candidate_status == :screen_candidate, rows)
    merge((; row_type=:summary, schema_version=REFINEMENT_SCHEMA_VERSION,
        run_id=settings.run_id, data_dir=settings.data_dir,
        code_commit=settings.code_commit, max_branches=settings.max_branches,
        negative_mode_range=settings.negative_mode_range,
        random_starts=settings.random_starts, random_seed=settings.random_seed,
        max_points=settings.max_points, float_tolerance=settings.float_tolerance,
        high_tolerance=settings.high_tolerance,
        duplicate_tolerance=settings.duplicate_tolerance,
        working_basis=:periodic_string, physical_basis=:mass_eigenbasis,
        physical_vectors=:deferred, scale_status=:unknown,
        h11=geom.h11, polytope=geom.polytope, frst=geom.frst,
        search_status=search.search_status, coverage_status=search.coverage_status,
        refinement_status=search.refinement_status,
        search_mode=search.report === nothing ? :not_completed :
            search.report.search_classification, branch_source=:summary,
        branch_estimate=search.branch_estimate,
        branch_count=search.report === nothing ? nothing : search.report.branch_count,
        mask_count=search.mask_count,
        masks_visited=search.report === nothing ? nothing : search.report.masks_visited,
        branch_seed_index=nothing, leading_negative_modes=nothing, seed_theta=nothing,
        float_correction_status=nothing, float_correction_residual=nothing,
        float_correction_iterations=nothing, high_precision_bits=nothing,
        high_correction_status=nothing, high_correction_residual=nothing,
        high_correction_iterations=nothing, high_value=nothing,
        high_gradient_residual=nothing, high_gradient_norm=nothing,
        high_epsilon=nothing, high_min_eta=nothing, high_negative_modes=nothing,
        high_zeroish_modes=nothing, high_positive_modes=nothing,
        residual_agreement=nothing, inertia_agreement=nothing, accepted=nothing,
        candidate_status=refined > 0 ? :refined_candidate :
            (screened > 0 ? :screen_candidate : :none),
        candidate_reason="summary; candidate recall remains unmeasured",
        wall_seconds, allocated_bytes, output_bytes,
        failure=search.error),
        _diagnostic_fields(nothing, "float"))
end

function _candidate_parse_args(args)
    options = Dict{Symbol, Any}(
        :data_dir => get(ENV, "CYAXIVERSE_DATA_DIR",
            normpath(joinpath(@__DIR__, "..", "..", "data"))),
        :geometries => GeometryIndex[],
        :output => "/private/tmp/inflation-candidate-refinement.csv",
        :flow_output => nothing,
        :max_branches => 100_000, :negative_mode_range => nothing,
        :random_starts => 0, :random_seed => 20260810, :precision_bits => 256,
        :max_points => 100_000, :float_tolerance => 1e-10,
        :high_tolerance => 1e-40, :flow_min_efolds => 50.0,
        :flow_max_efolds => 60.0, :flow_step => 1e-3,
        :flow_displacement => 1e-8, :require_finite_exit => false)
    valued = ("--data-dir", "--geometry", "--output", "--flow-output",
        "--max-branches", "--negative-mode-range",
        "--random-starts", "--random-seed", "--precision-bits", "--max-points",
        "--float-tolerance", "--high-tolerance", "--flow-min-efolds",
        "--flow-max-efolds", "--flow-step", "--flow-displacement")
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--require-finite-exit"
            options[:require_finite_exit] = true
            index += 1
            continue
        end
        arg == "--help" && (println("""
Usage: julia --project=. scripts/inflation_candidate_refinement.jl --geometry H,P,F [options]
  --data-dir PATH, --output PATH, --flow-output PATH
  --max-branches N, --negative-mode-range K[:K]
  --random-starts N, --random-seed N, --precision-bits N, --max-points N
  --float-tolerance T, --high-tolerance T
  --flow-min-efolds N, --flow-max-efolds N, --flow-step N,
  --flow-displacement N, --require-finite-exit
"""); return nothing)
        arg in valued || error("unknown argument: $arg")
        index += 1
        index <= length(args) || error("missing value for $arg")
        value = args[index]
        if arg == "--data-dir"
            options[:data_dir] = value
        elseif arg == "--geometry"
            push!(options[:geometries], _parse_geometry(value))
        elseif arg == "--output"
            options[:output] = value
        elseif arg == "--flow-output"
            options[:flow_output] = value
        elseif arg == "--max-branches"
            options[:max_branches] = parse(Int, value)
        elseif arg == "--negative-mode-range"
            options[:negative_mode_range] = _parse_range(value)
        elseif arg == "--random-starts"
            options[:random_starts] = parse(Int, value)
        elseif arg == "--random-seed"
            options[:random_seed] = parse(Int, value)
        elseif arg == "--precision-bits"
            options[:precision_bits] = parse(Int, value)
        elseif arg == "--max-points"
            options[:max_points] = parse(Int, value)
        elseif arg == "--float-tolerance"
            options[:float_tolerance] = parse(Float64, value)
        elseif arg == "--high-tolerance"
            options[:high_tolerance] = parse(Float64, value)
        elseif arg == "--flow-min-efolds"
            options[:flow_min_efolds] = parse(Float64, value)
        elseif arg == "--flow-max-efolds"
            options[:flow_max_efolds] = parse(Float64, value)
        elseif arg == "--flow-step"
            options[:flow_step] = parse(Float64, value)
        elseif arg == "--flow-displacement"
            options[:flow_displacement] = parse(Float64, value)
        end
        index += 1
    end
    isempty(options[:geometries]) && error("at least one --geometry is required")
    if options[:flow_output] === nothing
        base, extension = splitext(options[:output])
        options[:flow_output] = string(base, ".flows", extension)
    end
    options
end

function main(args=ARGS)
    options = _candidate_parse_args(args)
    options === nothing && return nothing
    ENV["CYAXIVERSE_DATA_DIR"] = abspath(expanduser(options[:data_dir]))
    run_id = string("candidate-refinement-", getpid(), "-", round(Int, time()))
    settings = (; run_id, data_dir=ENV["CYAXIVERSE_DATA_DIR"],
        code_commit=_current_code_commit(),
        max_branches=options[:max_branches],
        negative_mode_range=options[:negative_mode_range],
        random_starts=options[:random_starts], random_seed=options[:random_seed],
        max_points=options[:max_points], float_tolerance=options[:float_tolerance],
        high_tolerance=options[:high_tolerance], duplicate_tolerance=1e-7)
    rows = NamedTuple[]
    flow_rows = NamedTuple[]
    for geom in options[:geometries]
        @printf("refining h11=%d polytope=%d frst=%d\n",
            geom.h11, geom.polytope, geom.frst)
        started = time_ns()
        measured = @timed scan_geometry_for_inflation(geom;
            max_branches=options[:max_branches],
            negative_mode_range=options[:negative_mode_range],
            random_starts=options[:random_starts], random_seed=options[:random_seed],
            precision_bits=options[:precision_bits],
            float_tolerance=options[:float_tolerance],
            high_tolerance=options[:high_tolerance], max_points=options[:max_points],
            min_efolds=options[:flow_min_efolds],
            max_efolds=options[:flow_max_efolds],
            flow_step=options[:flow_step],
            flow_displacement=options[:flow_displacement],
            require_finite_exit=options[:require_finite_exit],
            settings=settings)
        result = measured.value
        points = result.refinement.rows
        append!(rows, points)
        append!(flow_rows, result.flow.rows)
        push!(rows, _summary_row(geom, result.refinement.search, points, settings;
            wall_seconds=(time_ns() - started) / 1e9,
            allocated_bytes=measured.bytes,
            output_bytes=Base.summarysize(points)))
    end
    mkpath(dirname(abspath(options[:output])))
    _write_csv(options[:output], rows)
    mkpath(dirname(abspath(options[:flow_output])))
    _write_csv(options[:flow_output], flow_rows, FLOW_FIELDS)
    println(options[:output])
    println(options[:flow_output])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
