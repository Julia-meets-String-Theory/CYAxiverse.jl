#!/usr/bin/env julia

"""Compare full and CN-reduced minima counts along the physical k scan.

The primary scan uses the author-reconstructed potential with
`volume_normalization=:full`: divisor volumes and `Kinv` scale with `k`, while
the Calabi-Yau volume scales as `k^(3/2)`.  The full model is solved with all
stored instantons.  The comparison model applies the CN catastrophe reduction
with threshold `0.01` to the same scaled potential.

This is a screening scan, not an exhaustive proof of minima counts.  Both
models use the same deterministic Halton-start budget and the output records
that budget, the reduction structure, the stored-potential normalization
diagnostic, and a fixed-volume control for each grid point.
"""

using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

const DEFAULT_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data"))
const DEFAULT_GRID = "0.5,1.0,1.25,1.5,1.75,2.0,2.5,3.0"
const DEFAULT_SEED = 20260812
const DEFAULT_STARTS = 2048
const DEFAULT_MAX_ITERATIONS = 300
const DEFAULT_RESIDUAL_TOLERANCE = 1e-8
const DEFAULT_MERGE_TOLERANCE = 1e-4
const H11_VALUES = 4:10
const SAMPLE_QUOTAS = (8, 10, 12, 14, 16, 18, 22)
const REDUCTION_THRESHOLD = 0.01

function parse_options(args)
    options = Dict{String,String}(
        "data-dir" => DEFAULT_DATA_DIR,
        "output-dir" => joinpath("/private/tmp",
            "cyaxiverse-reduction-comparison-" * Dates.format(now(), "yyyymmdd-HHMMSS")),
        "grid" => DEFAULT_GRID,
        "seed" => string(DEFAULT_SEED),
        "starts" => string(DEFAULT_STARTS),
        "max-iterations" => string(DEFAULT_MAX_ITERATIONS),
        "residual-tolerance" => string(DEFAULT_RESIDUAL_TOLERANCE),
        "merge-tolerance" => string(DEFAULT_MERGE_TOLERANCE),
        "fixed-control" => "true",
        "sample-limit" => "0",
    )
    index = 1
    while index <= length(args)
        argument = args[index]
        startswith(argument, "--") || throw(ArgumentError("unknown argument $argument"))
        body = argument[3:end]
        if occursin("=", body)
            key, value = split(body, "="; limit=2)
            options[key] = value
        elseif body == "no-fixed-control"
            options["fixed-control"] = "false"
        else
            index < length(args) || throw(ArgumentError("missing value for --$body"))
            index += 1
            options[body] = args[index]
        end
        index += 1
    end
    options
end

# Load the package and the shared physical-path helper before defining the
# geometry-facing methods below.  This avoids Julia world-age issues caused by
# defining `GeometryIndex` constructors after those methods are compiled.
const BOOT_OPTIONS = parse_options(ARGS)
ENV["CYAXIVERSE_DATA_DIR"] = normpath(abspath(expanduser(BOOT_OPTIONS["data-dir"])))
using CYAxiverse
include(joinpath(@__DIR__, "inflation_scale_continuation.jl"))

function parse_grid(value::AbstractString)
    grid = sort!(unique(Float64.(parse.(Float64, split(strip(value), ',')))))
    isempty(grid) && throw(ArgumentError("the k grid must not be empty"))
    all(isfinite, grid) && all(>(0), grid) ||
        throw(ArgumentError("the k grid must contain finite positive values"))
    grid
end

function parse_bool(value::AbstractString)
    lowercase(strip(value)) in ("1", "true", "yes", "y")
end

function parse_id(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    suffix = name[(lastindex(prefix) + 1):end]
    try
        parse(Int, suffix)
    catch
        nothing
    end
end

function discover_geometries(data_dir::AbstractString, h11::Int)
    h11_dir = joinpath(data_dir, "h11_$(lpad(h11, 3, '0'))")
    isdir(h11_dir) || throw(ArgumentError("missing h11 directory: $h11_dir"))
    geometries = GeometryIndex[]
    for polytope_name in sort(readdir(h11_dir))
        polytope = parse_id(polytope_name, "np_")
        polytope === nothing && continue
        polytope_dir = joinpath(h11_dir, polytope_name)
        isdir(polytope_dir) || continue
        for frst_name in sort(readdir(polytope_dir))
            frst = parse_id(frst_name, "cy_")
            frst === nothing && continue
            geometry_path = joinpath(polytope_dir, frst_name, "cyax.h5")
            isfile(geometry_path) || continue
            push!(geometries, GeometryIndex(h11, polytope, frst))
        end
    end
    sort!(geometries; by=geometry -> (geometry.polytope, geometry.frst))
    geometries
end

function sample_geometries(data_dir::AbstractString, seed::Int)
    rng = MersenneTwister(seed)
    sample = GeometryIndex[]
    manifest = NamedTuple[]
    for (h11, quota) in zip(H11_VALUES, SAMPLE_QUOTAS)
        available = discover_geometries(data_dir, h11)
        length(available) >= quota || throw(ArgumentError(
            "h11=$h11 has only $(length(available)) geometries; need $quota"))
        selected_indices = randperm(rng, length(available))[1:quota]
        for index in selected_indices
            geometry = available[index]
            push!(sample, geometry)
            push!(manifest, (; sample_rank=length(sample), h11=geometry.h11,
                polytope=geometry.polytope, frst=geometry.frst, quota,
                seed))
        end
    end
    sample, manifest
end

function csv_value(value)
    value === missing && return ""
    value isa AbstractVector && return join(string.(value), ';')
    value isa AbstractMatrix && return join(string.(vec(value)), ';')
    text = string(value)
    occursin(r"[,\"\n]", text) ? "\"" * replace(text, "\"" => "\"\"") * "\"" : text
end

function write_csv(path::AbstractString, rows, fields::Tuple)
    open(path, "w") do io
        write(io, join(string.(fields), ','), '\n')
        for row in rows
            write(io, join((csv_value(get(row, field, missing)) for field in fields), ','), '\n')
        end
    end
end

function write_manifest(path::AbstractString, manifest)
    fields = (:sample_rank, :h11, :polytope, :frst, :quota, :seed)
    write_csv(path, [Dict{Symbol,Any}(field => getproperty(row, field) for field in fields)
        for row in manifest], fields)
end

function reduction_signature(problem)
    problem.square_vacua === nothing ?
        "n=$(size(problem.Q_reduced, 2));m=$(size(problem.Q_reduced, 1));extra=$(problem.extra_rows);integer=$(problem.integer_charges)" :
        "square=$(problem.square_vacua);det=$(problem.det_QTilde)"
end

function finite_count(value)
    value === missing || (value isa Real && isfinite(value))
end

function count_vector(rows, model::Symbol, normalization)
    selected = filter(row -> row[:model] == string(model) &&
        row[:normalization] == string(normalization),
        rows)
    sort!(selected; by=row -> row[:k])
    [row[:total_minima] for row in selected]
end

function changes_count(counts)
    values = filter(x -> x !== missing && x isa Real && isfinite(x), counts)
    length(values) == length(counts) && length(values) >= 2 && length(unique(values)) > 1
end

function vector_text(values)
    join((value === missing ? "missing" : string(value) for value in values), ';')
end

function load_geometry(geometry)
    potential = CYAxiverse.read.oriented_potential(geometry)
    geometric = CYAxiverse.read.geometry(geometry)
    Q = Matrix{Int}(potential.Q)
    L_stored = Matrix{Float64}(potential.L)
    K = Matrix{Float64}(potential.K)
    tau = Vector{Float64}(geometric.τ_volumes)
    kinv = Matrix{Float64}(geometric.kinv)
    cy_volume = Float64(geometric.cy_volume)
    expected = _pilot_author_potential(Q, tau, kinv, cy_volume)
    finite_logs = isfinite.(L_stored[2, :]) .& isfinite.(expected[2, :])
    nonfinite_compatible = all(finite_logs .| (L_stored[2, :] .== expected[2, :]))
    max_error = any(finite_logs) ?
        maximum(abs.(L_stored[2, finite_logs] .- expected[2, finite_logs])) : 0.0
    sign_mismatches = count(index -> sign(L_stored[1, index]) != sign(expected[1, index]),
        eachindex(L_stored[1, :]))
    (; Q, L_stored, K, tau, kinv, cy_volume, reference=expected,
       reference_max_log10_error=max_error, reference_sign_mismatches=sign_mismatches,
       reference_nonfinite_compatible=nonfinite_compatible)
end

function scaled_inputs(loaded, k::Float64, normalization::Symbol)
    scaled_tau = k .* loaded.tau
    scaled_kinv = k^2 .* loaded.kinv
    scaled_volume = normalization == :full ? loaded.cy_volume * k^(3 / 2) : loaded.cy_volume
    L = _pilot_author_potential(loaded.Q, scaled_tau, scaled_kinv, scaled_volume)
    K = Matrix{Float64}(loaded.K) / k^2
    (; Q=loaded.Q, L, K, volume=scaled_volume)
end

function solve_full(inputs, starts, residual_tolerance, merge_tolerance, max_iterations)
    solved = CYAxiverse.minimizer.critical_points(inputs.L, inputs.Q;
        starts, residual_tolerance, merge_tolerance, max_iterations)
    context = CYAxiverse.inflation_points.prepare_context(inputs.Q, inputs.L, inputs.K)
    full_diagnostics = [CYAxiverse.inflation_points.diagnose(context, solved.minima[:, index])
        for index in axes(solved.minima, 2)]
    full_nonnegative = count(diagnostic -> diagnostic.negative_modes == 0, full_diagnostics)
    (; critical_count=solved.critical_count, found_minima=solved.minima_count,
       total_minima=solved.minima_count, full_nonnegative, reduced_full_minima=missing,
       reduced_full_negative=missing, reduced_full_zeroish=missing,
       reduction_dimension=size(inputs.Q, 1), reduction_signature="full",
       multiplicity=1.0, extra_rows=size(inputs.Q, 2) - size(inputs.Q, 1),
       solver_status="finite_start_search", residual_max=maximum_or_zero(solved.residuals),
       coordinates=solved.coordinates)
end

function maximum_or_zero(values)
    isempty(values) ? 0.0 : maximum(values)
end

function solve_reduced(inputs, starts, residual_tolerance, merge_tolerance, max_iterations)
    problem = CYAxiverse.jlm_reduced.prepare(inputs.Q, inputs.L;
        threshold=REDUCTION_THRESHOLD, reduction=:catastrophe)
    solved = CYAxiverse.jlm_reduced.critical_ensemble(problem;
        starts, residual_tolerance, merge_tolerance, max_iterations)
    context = CYAxiverse.inflation_points.prepare_context(inputs.Q, inputs.L, inputs.K)
    minima_mask = [entry == (0, 0, size(problem.Q_reduced, 2)) for entry in solved.inertia]
    reduced_coordinates = solved.coordinates[:, minima_mask]
    reduced_diagnostics = [CYAxiverse.inflation_points.diagnose(context,
        reduced_coordinates[:, index]) for index in axes(reduced_coordinates, 2)]
    full_nonnegative = count(diagnostic -> diagnostic.negative_modes == 0, reduced_diagnostics)
    full_negative = count(diagnostic -> diagnostic.negative_modes > 0, reduced_diagnostics)
    full_zeroish = count(diagnostic -> diagnostic.zeroish_modes > 0, reduced_diagnostics)
    total_minima = problem.square_vacua === nothing ?
        problem.multiplicity * solved.minima_count : Float64(problem.square_vacua)
    (; critical_count=solved.critical_count, found_minima=solved.minima_count,
       total_minima, full_nonnegative, reduced_full_minima=full_nonnegative,
       reduced_full_negative=full_negative, reduced_full_zeroish=full_zeroish,
       reduction_dimension=size(problem.Q_reduced, 2),
       reduction_signature=reduction_signature(problem),
       multiplicity=problem.multiplicity, extra_rows=problem.extra_rows,
       solver_status=problem.square_vacua === nothing ? "finite_start_search" : "square_lattice_count",
       residual_max=missing, coordinates=reduced_coordinates)
end

function run_one_model(loaded, k, normalization, model, options)
    inputs = scaled_inputs(loaded, k, normalization)
    solver = model == :full ? solve_full(inputs, options.starts, options.residual_tolerance,
        options.merge_tolerance, options.max_iterations) :
        solve_reduced(inputs, options.starts, options.residual_tolerance,
            options.merge_tolerance, options.max_iterations)
    (; solver, inputs)
end

function result_row(geometry, k, normalization, model, loaded, solved, elapsed, options)
    Dict{Symbol,Any}(
        :h11 => geometry.h11, :polytope => geometry.polytope, :frst => geometry.frst,
        :k => k, :normalization => string(normalization), :model => string(model),
        :threshold => model == :reduced ? REDUCTION_THRESHOLD : missing,
        :critical_count => solved.critical_count, :found_minima => solved.found_minima,
        :total_minima => solved.total_minima, :full_nonnegative => solved.full_nonnegative,
        :reduced_full_minima => solved.reduced_full_minima,
        :reduced_full_negative => solved.reduced_full_negative,
        :reduced_full_zeroish => solved.reduced_full_zeroish,
        :reduction_dimension => solved.reduction_dimension,
        :reduction_signature => solved.reduction_signature,
        :multiplicity => solved.multiplicity, :extra_rows => solved.extra_rows,
        :solver_status => solved.solver_status, :residual_max => solved.residual_max,
        :elapsed_seconds => elapsed, :starts => options.starts,
        :residual_tolerance => options.residual_tolerance,
        :merge_tolerance => options.merge_tolerance,
        :reference_max_log10_error => loaded.reference_max_log10_error,
        :reference_sign_mismatches => loaded.reference_sign_mismatches,
        :reference_nonfinite_compatible => loaded.reference_nonfinite_compatible,
    )
end

function run_scan(sample, grid, options; checkpoint=nothing)
    rows = Dict{Symbol,Any}[]
    errors = Dict{Symbol,Any}[]
    for (sample_rank, geometry) in enumerate(sample)
        @info "Scanning geometry $(geometry)" sample_rank total=length(sample)
        loaded = try
            load_geometry(geometry)
        catch error
            push!(errors, Dict{Symbol,Any}(:sample_rank => sample_rank,
                :h11 => geometry.h11, :polytope => geometry.polytope, :frst => geometry.frst,
                :stage => "load", :error => sprint(showerror, error)))
            continue
        end
        normalizations = options.fixed_control ? (:full, :fixed) : (:full,)
        for normalization in normalizations, k in grid, model in (:full, :reduced)
            started = time()
            try
                result = run_one_model(loaded, k, normalization, model, options)
                push!(rows, result_row(geometry, k, normalization, model, loaded,
                    result.solver, time() - started, options) |>
                    row -> merge(row, Dict{Symbol,Any}(:sample_rank => sample_rank)))
            catch error
                push!(errors, Dict{Symbol,Any}(:sample_rank => sample_rank,
                    :h11 => geometry.h11, :polytope => geometry.polytope,
                    :frst => geometry.frst, :k => k,
                    :normalization => string(normalization), :model => string(model),
                    :stage => "solve", :error => sprint(showerror, error)))
                @warn "Solver failure" sample_rank geometry k normalization model exception=(error, catch_backtrace())
            end
        end
        checkpoint === nothing || checkpoint(rows, errors, sample_rank)
    end
    rows, errors
end

function summarize(rows, sample, grid, options)
    summary = Dict{Symbol,Any}[]
    for (sample_rank, geometry) in enumerate(sample)
        full_rows = filter(row -> row[:sample_rank] == sample_rank &&
            row[:normalization] == "full", rows)
        full_counts = Dict(model => count_vector(full_rows, model, "full")
            for model in (:full, :reduced))
        fixed_counts = Dict(model => count_vector(
            filter(row -> row[:sample_rank] == sample_rank, rows), model, "fixed")
            for model in (:full, :reduced))
        cn_signatures = [row[:reduction_signature] for row in full_rows if row[:model] == "reduced"]
        cn_reduction_switch = length(unique(cn_signatures)) > 1
        cn_candidate = changes_count(full_counts[:reduced])
        full_candidate = changes_count(full_counts[:full])
        reduction_count_difference = length(full_counts[:full]) == length(full_counts[:reduced]) &&
            any(full_counts[:full][index] != full_counts[:reduced][index]
                for index in eachindex(full_counts[:full]))
        normalization_difference = options.fixed_control &&
            any(full_counts[model] != fixed_counts[model] for model in (:full, :reduced))
        reduced_full_mismatch = any(row[:model] == "reduced" &&
            row[:reduced_full_negative] !== missing && row[:reduced_full_negative] > 0
            for row in full_rows)
        push!(summary, Dict{Symbol,Any}(
            :sample_rank => sample_rank, :h11 => geometry.h11,
            :polytope => geometry.polytope, :frst => geometry.frst,
            :grid => vector_text(grid), :full_counts => vector_text(full_counts[:full]),
            :cn_counts => vector_text(full_counts[:reduced]),
            :fixed_full_counts => vector_text(fixed_counts[:full]),
            :fixed_cn_counts => vector_text(fixed_counts[:reduced]),
            :full_candidate => full_candidate, :cn_candidate => cn_candidate,
            :both_candidate => full_candidate && cn_candidate,
            :cn_only_candidate => cn_candidate && !full_candidate,
            :full_only_candidate => full_candidate && !cn_candidate,
            :reduction_count_difference => reduction_count_difference,
            :reduction_structural_switch => cn_reduction_switch,
            :reduced_minima_not_full_minima => reduced_full_mismatch,
            :normalization_difference => normalization_difference,
            :reference_max_log10_error => isempty(full_rows) ? missing :
                full_rows[1][:reference_max_log10_error],
            :reference_sign_mismatches => isempty(full_rows) ? missing :
                full_rows[1][:reference_sign_mismatches],
            :rows_completed => length(full_rows),
        ))
    end
    summary
end

function main(args)
    options_dict = parse_options(args)
    data_dir = normpath(abspath(expanduser(options_dict["data-dir"])))
    output_dir = normpath(abspath(expanduser(options_dict["output-dir"])))
    isdir(data_dir) || throw(ArgumentError("data directory does not exist: $data_dir"))
    mkpath(output_dir)
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir

    options = (; data_dir, output_dir, grid=parse_grid(options_dict["grid"]),
        seed=parse(Int, options_dict["seed"]), starts=parse(Int, options_dict["starts"]),
        max_iterations=parse(Int, options_dict["max-iterations"]),
        residual_tolerance=parse(Float64, options_dict["residual-tolerance"]),
        merge_tolerance=parse(Float64, options_dict["merge-tolerance"]),
        fixed_control=parse_bool(options_dict["fixed-control"]))
    options.starts > 0 || throw(ArgumentError("starts must be positive"))
    options.max_iterations > 0 || throw(ArgumentError("max-iterations must be positive"))

    sample, manifest = sample_geometries(data_dir, options.seed)
    sample_limit = parse(Int, options_dict["sample-limit"])
    sample_limit >= 0 || throw(ArgumentError("sample-limit must be nonnegative"))
    if sample_limit > 0
        sample_limit = min(sample_limit, length(sample))
        sample = sample[1:sample_limit]
        manifest = manifest[1:sample_limit]
        manifest = [merge(row, (; sample_rank=index)) for (index, row) in enumerate(manifest)]
    end
    write_manifest(joinpath(output_dir, "sample_manifest.csv"), manifest)
    open(joinpath(output_dir, "run_metadata.txt"), "w") do io
        println(io, "data_dir=$(data_dir)")
        println(io, "grid=$(join(options.grid, ','))")
        println(io, "seed=$(options.seed)")
        println(io, "quotas=$(join(SAMPLE_QUOTAS, ','))")
        println(io, "starts=$(options.starts)")
        println(io, "max_iterations=$(options.max_iterations)")
        println(io, "residual_tolerance=$(options.residual_tolerance)")
        println(io, "merge_tolerance=$(options.merge_tolerance)")
        println(io, "fixed_control=$(options.fixed_control)")
        println(io, "reduction=catastrophe")
        println(io, "reduction_threshold=$(REDUCTION_THRESHOLD)")
        println(io, "physical_volume_rule=CY_volume(k)=CY_volume(1)*k^(3/2)")
        println(io, "generated_at=$(Dates.now())")
    end

    result_fields = (:sample_rank, :h11, :polytope, :frst, :k, :normalization, :model,
        :threshold, :critical_count, :found_minima, :total_minima, :full_nonnegative,
        :reduced_full_minima, :reduced_full_negative, :reduced_full_zeroish,
        :reduction_dimension, :reduction_signature, :multiplicity, :extra_rows,
        :solver_status, :residual_max, :elapsed_seconds, :starts, :residual_tolerance,
        :merge_tolerance, :reference_max_log10_error, :reference_sign_mismatches,
        :reference_nonfinite_compatible)
    error_fields = (:sample_rank, :h11, :polytope, :frst, :k, :normalization, :model,
        :stage, :error)
    checkpoint = (current_rows, current_errors, sample_rank) -> begin
        write_csv(joinpath(output_dir, "scan_rows.checkpoint.csv"), current_rows, result_fields)
        write_csv(joinpath(output_dir, "errors.checkpoint.csv"), current_errors, error_fields)
        open(joinpath(output_dir, "checkpoint.txt"), "w") do io
            println(io, "completed_geometries=$(sample_rank)")
            println(io, "updated_at=$(Dates.now())")
        end
    end
    rows, errors = run_scan(sample, options.grid, options; checkpoint)
    summary = summarize(rows, sample, options.grid, options)
    summary_fields = (:sample_rank, :h11, :polytope, :frst, :grid, :full_counts,
        :cn_counts, :fixed_full_counts, :fixed_cn_counts, :full_candidate, :cn_candidate,
        :both_candidate, :cn_only_candidate, :full_only_candidate,
        :reduction_count_difference, :reduction_structural_switch,
        :reduced_minima_not_full_minima, :normalization_difference,
        :reference_max_log10_error, :reference_sign_mismatches, :rows_completed)
    write_csv(joinpath(output_dir, "scan_rows.csv"), rows, result_fields)
    write_csv(joinpath(output_dir, "geometry_summary.csv"), summary, summary_fields)
    write_csv(joinpath(output_dir, "errors.csv"), errors, error_fields)

    println("output_dir=$(output_dir)")
    println("rows=$(length(rows)) errors=$(length(errors)) geometries=$(length(sample))")
    println("full_volume_cn_candidates=$(count(row -> row[:cn_candidate], summary))")
    println("full_volume_full_candidates=$(count(row -> row[:full_candidate], summary))")
    println("full_volume_both_candidates=$(count(row -> row[:both_candidate], summary))")
    println("normalization_difference_geometries=$(count(row -> row[:normalization_difference], summary))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
