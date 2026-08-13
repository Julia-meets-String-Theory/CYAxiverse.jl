#!/usr/bin/env julia

"""Run the draft CN catastrophe scan on a local relaxed-QCD sample.

This is a local-data adapter for the draft's `run_high_h11_chunk.sh` policy:
the generated HDF5 records are used as-is, the CN catastrophe reduction is
prepared independently at every k, and only the reduced minima count is used
for candidate promotion.  A candidate requires a changed minima count, a
consistent reduced model across the solved k values, and a changed critical
count.  The latter two checks retain the draft's model-switch and
minima-without-critical-change diagnostics.

The default k policy is the draft's adaptive policy: solve 0.5 and 1.0,
continue only when the first reduced minima count is greater than one, and
extend through 3.0 while the most recent count remains greater than one.
"""

using Dates
using LinearAlgebra
using SHA
using Statistics

const BOOT_OPTIONS = let
    options = Dict{String,String}(
        "data-dir" => "/private/tmp/cyaxiverse-relaxed-qcd-70",
    )
    for argument in ARGS
        startswith(argument, "--") || continue
        body = argument[3:end]
        occursin("=", body) || continue
        key, value = split(body, "="; limit=2)
        options[key] = value
    end
    options
end
ENV["CYAXIVERSE_DATA_DIR"] = normpath(abspath(expanduser(BOOT_OPTIONS["data-dir"])))
using CYAxiverse

const GeometryIndex = CYAxiverse.structs.GeometryIndex
const REDUCTION_THRESHOLD = 0.01
const DEFAULT_BASE_K = "0.5,1.0"
const DEFAULT_EXTENSION_K = "1.25,1.5,1.75,2.0,2.5,3.0"
const DEFAULT_STARTS = 2048
const DEFAULT_MAX_ITERATIONS = 300
const DEFAULT_RESIDUAL_TOLERANCE = 1e-8
const DEFAULT_MERGE_TOLERANCE = 1e-4

function parse_options(args)
    options = Dict{String,String}(
        "data-dir" => get(BOOT_OPTIONS, "data-dir", "/private/tmp/cyaxiverse-relaxed-qcd-70"),
        "output-dir" => joinpath("/private/tmp",
            "cyaxiverse-relaxed-qcd-scan-" * Dates.format(now(), "yyyymmdd-HHMMSS")),
        "base-k" => DEFAULT_BASE_K,
        "extension-k" => DEFAULT_EXTENSION_K,
        "qdot-gate" => "1.0",
        "adaptive" => "true",
        "starts" => string(DEFAULT_STARTS),
        "max-iterations" => string(DEFAULT_MAX_ITERATIONS),
        "residual-tolerance" => string(DEFAULT_RESIDUAL_TOLERANCE),
        "merge-tolerance" => string(DEFAULT_MERGE_TOLERANCE),
    )
    index = 1
    while index <= length(args)
        argument = args[index]
        startswith(argument, "--") || throw(ArgumentError("unknown argument $argument"))
        body = argument[3:end]
        if occursin("=", body)
            key, value = split(body, "="; limit=2)
            options[key] = value
        else
            index < length(args) || throw(ArgumentError("missing value for --$body"))
            index += 1
            options[body] = args[index]
        end
        index += 1
    end
    options
end

function parse_grid(value::AbstractString)
    grid = Float64.(parse.(Float64, filter(!isempty, split(strip(value), ','))))
    isempty(grid) && throw(ArgumentError("k grid must not be empty"))
    all(isfinite, grid) && all(>(0), grid) ||
        throw(ArgumentError("k grid must contain finite positive values"))
    grid
end

function parse_id(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        nothing
    end
end

function discover_geometries(data_dir::AbstractString)
    geometries = GeometryIndex[]
    for h11_name in sort(filter(name -> startswith(name, "h11_"), readdir(data_dir)))
        h11 = parse_id(h11_name, "h11_")
        h11 === nothing && continue
        4 <= h11 <= 10 || continue
        h11_dir = joinpath(data_dir, h11_name)
        isdir(h11_dir) || continue
        for polytope_name in sort(readdir(h11_dir))
            polytope = parse_id(polytope_name, "np_")
            polytope === nothing && continue
            polytope_dir = joinpath(h11_dir, polytope_name)
            isdir(polytope_dir) || continue
            for frst_name in sort(readdir(polytope_dir))
                frst = parse_id(frst_name, "cy_")
                frst === nothing && continue
                isfile(joinpath(polytope_dir, frst_name, "cyax.h5")) || continue
                push!(geometries, GeometryIndex(h11, polytope, frst))
            end
        end
    end
    sort!(geometries; by=geom -> (geom.h11, geom.polytope, geom.frst))
end

function csv_value(value)
    value === missing && return ""
    text = string(value)
    occursin(r"[,\"\n]", text) ? "\"" * replace(text, "\"" => "\"\"") * "\"" : text
end

function write_csv(path::AbstractString, rows, fields)
    open(path, "w") do io
        write(io, join(string.(fields), ','), '\n')
        for row in rows
            write(io, join((csv_value(get(row, field, missing)) for field in fields), ','), '\n')
        end
    end
end

function load_geometry(geometry)
    potential = CYAxiverse.read.oriented_potential(geometry)
    geometric = CYAxiverse.read.geometry(geometry)
    (; Q=Matrix{Int}(potential.Q), L=Matrix{Float64}(potential.L),
       K=Matrix{Float64}(potential.K), tau=Vector{Float64}(geometric.τ_volumes),
       kinv=Matrix{Float64}(geometric.kinv), cy_volume=Float64(geometric.cy_volume),
       glsm=Matrix{Int}(geometric.glsm_charges))
end

function draft_scaled_inputs(loaded, k::Float64)
    tau = k .* loaded.tau
    kinv = k^2 .* loaded.kinv
    # This is intentional: the draft's catastrophe scan holds CY_volume fixed
    # while scaling divisor volumes and Kinv along the k path.
    L = _draft_potential(loaded.Q, tau, kinv, loaded.cy_volume)
    (; Q=loaded.Q, L, K=loaded.K ./ k^2)
end

function _draft_term_count(term_count::Int)
    base_count = (isqrt(8 * term_count + 1) - 1) ÷ 2
    base_count > 0 && base_count * (base_count + 1) ÷ 2 == term_count ||
        throw(ArgumentError("potential is not base-plus-pairwise: $term_count"))
    base_count
end

function _draft_potential(Q::Matrix{Int}, tau::Vector{Float64},
        kinv::Matrix{Float64}, cy_volume::Float64)
    h11, term_count = size(Q)
    base_count = _draft_term_count(term_count)
    result = zeros(Float64, 2, term_count)
    prefactor = 8π / cy_volume^2
    log10e = log10(exp(1.0))
    for column in 1:base_count
        charge = @view Q[:, column]
        qtau = dot(charge, tau)
        coefficient = prefactor * qtau
        result[1, column] = sign(coefficient)
        result[2, column] = coefficient == 0 ? -Inf : log10(abs(coefficient)) -
            2π * log10e * qtau
    end
    column = base_count + 1
    for i in 1:(base_count - 1), j in (i + 1):base_count
        qi = @view Q[:, i]
        qj = @view Q[:, j]
        qsum = qi .+ qj
        qtau = dot(qsum, tau)
        coefficient = prefactor * (π * dot(qi, kinv * qj) + qtau)
        result[1, column] = sign(coefficient)
        result[2, column] = coefficient == 0 ? -Inf : log10(abs(coefficient)) -
            2π * log10e * qtau
        column += 1
    end
    result
end

function model_signature(problem)
    payload = if problem.square_vacua === nothing
        (kind=:nonsquare, Q=round.(Matrix(problem.Q_reduced); digits=8),
         signs=sign.(problem.L_reduced[:, 1]), phases=round.(problem.phases; digits=8),
         coordinate_scale=problem.coordinate_scale,
         lift=round.(problem.lift_matrix; digits=8), extra_rows=problem.extra_rows)
    else
        (kind=:square, square_vacua=problem.square_vacua, det=problem.det_QTilde)
    end
    bytes2hex(sha256(repr(payload)))
end

function qdot_tau_min(loaded, k::Float64)
    charges = unique(loaded.glsm; dims=1)
    minimum(vec(Float64.(charges) * (k .* loaded.tau)))
end

function solve_reduced(loaded, k::Float64, options; allow_nonpositive::Bool=false)
    inputs = draft_scaled_inputs(loaded, k)
    problem = CYAxiverse.jlm_reduced.prepare(inputs.Q, inputs.L;
        threshold=REDUCTION_THRESHOLD, reduction=:catastrophe)
    signs = size(problem.L_reduced, 2) == 0 ? Float64[] : problem.L_reduced[:, 1]
    nonpositive = !isempty(signs) && !all(>(0), signs)
    nonpositive && !allow_nonpositive &&
        return (; k, qdot_tau_min=qdot_tau_min(loaded, k), problem,
            model_signature=model_signature(problem), valid=false,
            invalid_reason="nonpositive_reduced_sign", minima=0, critical=0,
            found_minima=0, reduction_dimension=size(problem.Q_reduced, 2))
    solved = CYAxiverse.jlm_reduced.critical_ensemble(problem;
        starts=options.starts, residual_tolerance=options.residual_tolerance,
        merge_tolerance=options.merge_tolerance, max_iterations=options.max_iterations)
    minima = solved.minima_count
    critical = problem.square_vacua === nothing ? solved.critical_count :
        problem.square_vacua * 2^size(loaded.Q, 1)
    (; k, qdot_tau_min=qdot_tau_min(loaded, k), problem,
       model_signature=model_signature(problem), valid=minima > 0,
       invalid_reason=minima > 0 ? "" : "no_minima_found", minima, critical,
       found_minima=minima, reduction_dimension=size(problem.Q_reduced, 2),
       nonpositive_reduced_sign=nonpositive)
end

function scan_one(geometry, loaded, base_grid, extension_grid, options)
    results = NamedTuple[]
    first_gate = options.adaptive && options.qdot_gate > 0 ? options.qdot_gate : nothing
    for k in base_grid
        result = solve_reduced(loaded, k, options; allow_nonpositive=!isempty(results))
        push!(results, result)
        if isempty(results)
            error("unreachable")
        end
        if options.adaptive && length(results) == 1
            if first_gate !== nothing && result.qdot_tau_min < first_gate
                result = merge(result, (; valid=false,
                    invalid_reason="qdot_tau_below_first_gate"))
                results[end] = result
                break
            end
            if !result.valid || result.minima <= 1
                break
            end
        end
    end
    if options.adaptive && length(results) == length(base_grid) &&
            results[end].valid && results[end].minima > 1
        for k in extension_grid
            result = solve_reduced(loaded, k, options; allow_nonpositive=true)
            push!(results, result)
            result.valid && result.minima > 1 || break
        end
    end
    valid = length(results) >= length(base_grid) && all(result.valid for result in results)
    minima = [result.minima for result in results]
    critical = [result.critical for result in results]
    signatures = [result.model_signature for result in results]
    minima_changed = valid && length(unique(minima)) > 1
    critical_changed = valid && length(unique(critical)) > 1
    model_consistent = length(unique(signatures)) <= 1
    raw_candidate = minima_changed
    model_switch = raw_candidate && !model_consistent
    minima_without_critical = raw_candidate && !critical_changed
    candidate = raw_candidate && model_consistent && critical_changed
    (; geometry, results, valid, minima, critical, model_consistent, minima_changed,
       critical_changed, model_switch, minima_without_critical, candidate,
       qdot_first=first(results).qdot_tau_min,
       reduction_dimensions=[result.reduction_dimension for result in results])
end

function main(args)
    parsed = parse_options(args)
    data_dir = normpath(abspath(expanduser(parsed["data-dir"])))
    output_dir = normpath(abspath(expanduser(parsed["output-dir"])))
    isdir(data_dir) || throw(ArgumentError("data directory does not exist: $data_dir"))
    mkpath(output_dir)
    base_grid = parse_grid(parsed["base-k"])
    extension_grid = parse_grid(parsed["extension-k"])
    options = (; starts=parse(Int, parsed["starts"]),
        max_iterations=parse(Int, parsed["max-iterations"]),
        residual_tolerance=parse(Float64, parsed["residual-tolerance"]),
        merge_tolerance=parse(Float64, parsed["merge-tolerance"]),
        qdot_gate=parse(Float64, parsed["qdot-gate"]))
    options = merge(options, (; adaptive=lowercase(parsed["adaptive"]) in ("1", "true", "yes", "y")))
    geometries = discover_geometries(data_dir)
    rows = Dict{Symbol,Any}[]
    summaries = Dict{Symbol,Any}[]
    errors = Dict{Symbol,Any}[]
    for (rank, geometry) in enumerate(geometries)
        print("[$rank/$(length(geometries))] scanning $geometry ... ")
        try
            loaded = load_geometry(geometry)
            scan = scan_one(geometry, loaded, base_grid, extension_grid, options)
            for result in scan.results
                push!(rows, Dict{Symbol,Any}(
                    :sample_rank=>rank, :h11=>geometry.h11, :polytope=>geometry.polytope,
                    :frst=>geometry.frst, :k=>result.k,
                    :qdot_tau_min=>result.qdot_tau_min, :valid=>result.valid,
                    :invalid_reason=>result.invalid_reason, :minima=>result.minima,
                    :critical=>result.critical, :reduction_dimension=>result.reduction_dimension,
                    :model_signature=>result.model_signature,
                    :nonpositive_reduced_sign=>get(result, :nonpositive_reduced_sign, false)))
            end
            push!(summaries, Dict{Symbol,Any}(
                :sample_rank=>rank, :h11=>geometry.h11, :polytope=>geometry.polytope,
                :frst=>geometry.frst, :k_values=>join((result.k for result in scan.results), ';'),
                :minima_counts=>join(scan.minima, ';'),
                :critical_counts=>join(scan.critical, ';'), :qdot_first=>scan.qdot_first,
                :model_consistent=>scan.model_consistent, :minima_changed=>scan.minima_changed,
                :critical_changed=>scan.critical_changed, :candidate=>scan.candidate,
                :model_switch=>scan.model_switch,
                :minima_without_critical=>scan.minima_without_critical,
                :reduction_dimensions=>join(scan.reduction_dimensions, ';'),
                :valid=>scan.valid, :status=>"scanned"))
            println(scan.candidate ? "CANDIDATE" : "screened")
        catch error
            push!(errors, Dict{Symbol,Any}(:sample_rank=>rank, :h11=>geometry.h11,
                :polytope=>geometry.polytope, :frst=>geometry.frst,
                :error=>sprint(showerror, error), :status=>"failed"))
            println("FAILED: ", sprint(showerror, error))
        end
    end
    row_fields = (:sample_rank,:h11,:polytope,:frst,:k,:qdot_tau_min,:valid,
        :invalid_reason,:minima,:critical,:reduction_dimension,:model_signature,
        :nonpositive_reduced_sign)
    summary_fields = (:sample_rank,:h11,:polytope,:frst,:k_values,:minima_counts,
        :critical_counts,:qdot_first,:model_consistent,:minima_changed,
        :critical_changed,:candidate,:model_switch,:minima_without_critical,
        :reduction_dimensions,:valid,:status)
    error_fields = (:sample_rank,:h11,:polytope,:frst,:error,:status)
    write_csv(joinpath(output_dir, "scan_rows.csv"), rows, row_fields)
    write_csv(joinpath(output_dir, "geometry_summary.csv"), summaries, summary_fields)
    write_csv(joinpath(output_dir, "errors.csv"), errors, error_fields)
    open(joinpath(output_dir, "run_metadata.txt"), "w") do io
        println(io, "data_dir=$(data_dir)")
        println(io, "geometries=$(length(geometries))")
        println(io, "base_k=$(join(base_grid, ','))")
        println(io, "extension_k=$(join(extension_grid, ','))")
        println(io, "qdot_gate=$(options.qdot_gate)")
        println(io, "adaptive=$(options.adaptive)")
        println(io, "reduction=catastrophe")
        println(io, "reduction_threshold=$(REDUCTION_THRESHOLD)")
        println(io, "candidate_type=minima")
        println(io, "minima_field=reduced")
        println(io, "volume_normalization=fixed_draft_CY_volume")
        println(io, "generated_at=$(Dates.now())")
    end
    println("output_dir=$(output_dir)")
    println("geometries=$(length(geometries)) scanned=$(length(summaries)) errors=$(length(errors))")
    println("candidates=$(count(row -> get(row, :candidate, false), summaries))")
    println("model_switch_symptoms=$(count(row -> get(row, :model_switch, false), summaries))")
    println("minima_without_critical_symptoms=$(count(row -> get(row, :minima_without_critical, false), summaries))")
    println("first_gate_failures=$(count(row -> get(row, :status, "") == "scanned" &&
        get(row, :qdot_first, Inf) < options.qdot_gate, summaries))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
