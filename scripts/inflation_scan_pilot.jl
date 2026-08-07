#!/usr/bin/env julia

"""Run a bounded, h11-stratified pilot on real inflation geometries.

The pilot samples evenly spaced geometries within each available h11 group,
then uses the Stage 5 scan-prep shard writer.  Instanton count, hierarchy, and
candidate-count strata are measured from the resulting screening rows rather
than guessed before the numerical screen.
"""

include(joinpath(@__DIR__, "inflation_scan_prep.jl"))

using Printf
using Statistics

function _inflation_pilot_usage()
    println("""
    Usage:
      julia --project=. scripts/inflation_scan_pilot.jl [options]

    Options:
      --data-dir PATH       Data root containing h11_*/np_*/cy_*/cyax.h5.
      --h11 N               Restrict the pilot to one h11 value.
      --h11-min N           Restrict the pilot to h11 >= N.
      --h11-max N           Restrict the pilot to h11 <= N.
      --sample-per-h11 N    Evenly spaced geometries per h11. Default: 1.
      --max-geometries N    Global cap, sampled evenly across selected h11 groups.
      --max-branches N      Bound leading branch enumeration. Default: 100000.
      --negative-mode-range K[:K]
                            Search only leading branches in an index range.
      --max-negative-modes K
                            Search leading branches with index 0 through K.
      --shard-dir PATH      Stage 5 shard directory.
      --report PATH         Aggregated pilot report CSV.
      --run-id ID           Provenance label for shard rows.
      --retries N           Retry failed geometries N times. Default: 0.
      --resume              Resume from existing shard rows.
    """)
end

function _inflation_pilot_parse_args(args)
    options = Dict{Symbol, Any}(
        :data_dir => get(ENV, "CYAXIVERSE_DATA_DIR", ""),
        :h11 => nothing, :h11_min => nothing, :h11_max => nothing,
        :sample_per_h11 => 1, :max_geometries => nothing,
        :max_branches => 100_000, :shard_dir => "", :report => "",
        :negative_mode_range => nothing, :max_negative_modes => nothing,
        :run_id => "", :retries => 0, :resume => false)
    valued = ("--data-dir", "--h11", "--h11-min", "--h11-max",
        "--sample-per-h11", "--max-geometries",
        "--max-branches", "--negative-mode-range", "--max-negative-modes",
        "--shard-dir", "--report", "--run-id", "--retries")
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            _inflation_pilot_usage()
            exit(0)
        elseif arg == "--resume"
            options[:resume] = true
        elseif arg in valued
            index == length(args) && error("missing value for $arg")
            value = args[index + 1]
            if arg == "--data-dir"
                options[:data_dir] = value
            elseif arg == "--h11"
                options[:h11] = parse(Int, value)
            elseif arg == "--h11-min"
                options[:h11_min] = parse(Int, value)
            elseif arg == "--h11-max"
                options[:h11_max] = parse(Int, value)
            elseif arg == "--sample-per-h11"
                options[:sample_per_h11] = parse(Int, value)
            elseif arg == "--max-geometries"
                options[:max_geometries] = parse(Int, value)
            elseif arg == "--max-branches"
                options[:max_branches] = parse(Int, value)
            elseif arg == "--negative-mode-range"
                options[:negative_mode_range] =
                    inflation_parse_negative_mode_range(value)
            elseif arg == "--max-negative-modes"
                options[:max_negative_modes] = parse(Int, value)
            elseif arg == "--shard-dir"
                options[:shard_dir] = value
            elseif arg == "--report"
                options[:report] = value
            elseif arg == "--run-id"
                options[:run_id] = value
            elseif arg == "--retries"
                options[:retries] = parse(Int, value)
            end
            index += 1
        else
            error("unknown argument $arg")
        end
        index += 1
    end
    isempty(options[:data_dir]) && error("--data-dir or CYAXIVERSE_DATA_DIR is required")
    options[:data_dir] = abspath(expanduser(options[:data_dir]))
    options[:h11] === nothing || options[:h11] > 0 || error("--h11 must be positive")
    options[:h11_min] === nothing || options[:h11_min] > 0 ||
        error("--h11-min must be positive")
    options[:h11_max] === nothing || options[:h11_max] > 0 ||
        error("--h11-max must be positive")
    options[:h11] === nothing || options[:h11_min] === nothing ||
        options[:h11] >= options[:h11_min] ||
        error("--h11 must be at least --h11-min")
    options[:h11] === nothing || options[:h11_max] === nothing ||
        options[:h11] <= options[:h11_max] ||
        error("--h11 must be at most --h11-max")
    options[:h11_min] === nothing || options[:h11_max] === nothing ||
        options[:h11_min] <= options[:h11_max] ||
        error("--h11-min must not exceed --h11-max")
    options[:sample_per_h11] > 0 || error("--sample-per-h11 must be positive")
    options[:max_geometries] === nothing || options[:max_geometries] > 0 ||
        error("--max-geometries must be positive")
    options[:max_branches] > 0 || error("--max-branches must be positive")
    options[:max_negative_modes] === nothing || options[:max_negative_modes] >= 0 ||
        error("--max-negative-modes must be nonnegative")
    options[:negative_mode_range] === nothing || options[:max_negative_modes] === nothing ||
        error("use only one of --negative-mode-range and --max-negative-modes")
    options[:retries] >= 0 || error("--retries must be nonnegative")
    options
end

function _inflation_pilot_h11_values(data_dir::AbstractString, h11_filter;
        h11_min=nothing, h11_max=nothing)
    values = Int[]
    for name in sort(readdir(data_dir))
        h11 = _scan_prep_parse_prefixed_int(name, "h11_")
        h11 === nothing && continue
        h11_filter === nothing || h11 == h11_filter || continue
        h11_min === nothing || h11 >= h11_min || continue
        h11_max === nothing || h11 <= h11_max || continue
        isdir(joinpath(data_dir, name)) && push!(values, h11)
    end
    sort!(unique(values))
end

function _inflation_pilot_group_count(data_dir::AbstractString, h11::Int)
    h11_path = joinpath(data_dir, string("h11_", lpad(h11, 3, '0')))
    count = 0
    for np_dir in sort(filter(name -> startswith(name, "np_"), readdir(h11_path)))
        np = _scan_prep_parse_prefixed_int(np_dir, "np_")
        np === nothing && continue
        np_path = joinpath(h11_path, np_dir)
        isdir(np_path) || continue
        for cy_dir in sort(filter(name -> startswith(name, "cy_"), readdir(np_path)))
            frst = _scan_prep_parse_prefixed_int(cy_dir, "cy_")
            frst === nothing && continue
            isfile(joinpath(np_path, cy_dir, "cyax.h5")) && (count += 1)
        end
    end
    count
end

function _inflation_pilot_sample_positions(total::Int, requested::Int)
    total > 0 || return Int[]
    count = min(total, requested)
    count == 1 && return [cld(total, 2)]
    positions = [round(Int, 1 + (index - 1) * (total - 1) / (count - 1))
        for index in 1:count]
    sort!(unique(positions))
end

function _inflation_pilot_group_sample(data_dir::AbstractString, h11::Int,
        positions::AbstractVector{<:Integer})
    wanted = Set(Int.(positions))
    selected = GeometryIndex[]
    h11_path = joinpath(data_dir, string("h11_", lpad(h11, 3, '0')))
    index = 0
    for np_dir in sort(filter(name -> startswith(name, "np_"), readdir(h11_path)))
        polytope = _scan_prep_parse_prefixed_int(np_dir, "np_")
        polytope === nothing && continue
        np_path = joinpath(h11_path, np_dir)
        isdir(np_path) || continue
        for cy_dir in sort(filter(name -> startswith(name, "cy_"), readdir(np_path)))
            frst = _scan_prep_parse_prefixed_int(cy_dir, "cy_")
            frst === nothing && continue
            isfile(joinpath(np_path, cy_dir, "cyax.h5")) || continue
            index += 1
            index in wanted && push!(selected, GeometryIndex(h11, polytope, frst))
        end
    end
    selected
end

"""Select evenly spaced geometries per h11, with an optional round-robin cap."""
function inflation_pilot_select_geometries(data_dir::AbstractString;
        h11_filter=nothing, h11_min=nothing, h11_max=nothing,
        sample_per_h11::Int=1, max_geometries=nothing)
    h11_values = _inflation_pilot_h11_values(data_dir, h11_filter;
        h11_min, h11_max)
    groups = Dict{Int, Vector{GeometryIndex}}()
    for h11 in h11_values
        total = _inflation_pilot_group_count(data_dir, h11)
        groups[h11] = _inflation_pilot_group_sample(data_dir, h11,
            _inflation_pilot_sample_positions(total, sample_per_h11))
    end
    selected = GeometryIndex[]
    for h11 in h11_values
        append!(selected, groups[h11])
    end
    if max_geometries !== nothing && length(selected) > max_geometries
        positions = _inflation_pilot_sample_positions(
            length(selected), max_geometries)
        selected = selected[positions]
    end
    sort!(selected, by=geom -> (geom.h11, geom.polytope, geom.frst))
    selected
end

function _inflation_pilot_parse_value(fields, positions, name, default)
    index = get(positions, name, 0)
    if index == 0 || index > length(fields) || isempty(fields[index])
        return default
    end
    try
        default isa Integer ? parse(Int, fields[index]) : parse(Float64, fields[index])
    catch
        default
    end
end

function _inflation_pilot_terminal_rows(shard_dir::AbstractString)
    rows = Dict{Tuple{Int, Int, Int}, NamedTuple}()
    for path in inflation_shard_paths(shard_dir)
        _inflation_shard_validate_header(path)
        positions = Dict(field => index for
            (index, field) in enumerate(INFLATION_SHARD_FIELDS))
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                isempty(strip(line)) && continue
                fields = _inflation_shard_csv_fields(line)
                length(fields) == length(INFLATION_SHARD_FIELDS) || continue
                h11 = _inflation_pilot_parse_value(fields, positions, :h11, 0)
                polytope = _inflation_pilot_parse_value(
                    fields, positions, :polytope, 0)
                frst = _inflation_pilot_parse_value(fields, positions, :frst, 0)
                h11 > 0 && polytope > 0 && frst > 0 || continue
                key = (h11, polytope, frst)
                row = (; h11, polytope, frst,
                    status=Symbol(fields[positions[:status]]),
                    attempt=_inflation_pilot_parse_value(
                        fields, positions, :attempt, 0),
                    instantons=_inflation_pilot_parse_value(
                        fields, positions, :instantons, 0),
                    strong_hierarchy=lowercase(
                        fields[positions[:strong_hierarchy]]) == "true",
                    leading_log_gap=_inflation_pilot_parse_value(
                        fields, positions, :leading_log_gap, NaN),
                    log_scale_span=_inflation_pilot_parse_value(
                        fields, positions, :log_scale_span, NaN),
                    branch_count=_inflation_pilot_parse_value(
                        fields, positions, :branch_count, 0),
                    candidate_count=_inflation_pilot_parse_value(
                        fields, positions, :candidate_slowroll_saddles, 0),
                    total_seconds=_inflation_pilot_parse_value(
                        fields, positions, :total_seconds, NaN),
                    allocated_bytes=_inflation_pilot_parse_value(
                        fields, positions, :stage_allocated_bytes, 0),
                    output_bytes=_inflation_pilot_parse_value(
                        fields, positions, :stage_output_bytes, 0),
                    error=fields[positions[:error]])
                previous = get(rows, key, nothing)
                previous === nothing || row.attempt >= previous.attempt || continue
                rows[key] = row
            end
        end
    end
    sort!(collect(values(rows)), by=row -> (row.h11, row.polytope, row.frst))
end

function _inflation_pilot_instanton_bin(count::Int)
    count <= 50 ? "0-50" : count <= 100 ? "51-100" :
        count <= 200 ? "101-200" : ">200"
end

function _inflation_pilot_candidate_bin(count::Int)
    count == 0 ? "0" : count <= 10 ? "1-10" : count <= 100 ? "11-100" : ">100"
end

function _inflation_pilot_mean(rows, field)
    values = [getproperty(row, field) for row in rows]
    finite = filter(isfinite, Float64.(values))
    isempty(finite) ? NaN : mean(finite)
end

function _inflation_pilot_report_rows(rows)
    groups = Dict{Tuple{Int, String, String, String}, Vector{NamedTuple}}()
    for row in rows
        key = (row.h11, _inflation_pilot_instanton_bin(row.instantons),
            row.strong_hierarchy ? "strong" : "not_strong",
            _inflation_pilot_candidate_bin(row.candidate_count))
        push!(get!(groups, key, NamedTuple[]), row)
    end
    reports = NamedTuple[]
    for key in sort(collect(keys(groups)))
        group = groups[key]
        statuses = [row.status for row in group]
        push!(reports, (; h11=key[1], screening_tier=string(
                inflation_screening_tier(key[1])), instanton_bin=key[2],
            hierarchy_bin=key[3], candidate_bin=key[4], geometries=length(group),
            successes=count(==(Symbol(:success)), statuses),
            branch_caps=count(==(Symbol(:branch_cap)), statuses),
            failures=count(==(Symbol(:failed)), statuses),
            empty_enumerations=count(==(Symbol(:empty_enumeration)), statuses),
            refinement_eligible=count(inflation_refinement_eligible, group),
            mean_instantons=_inflation_pilot_mean(group, :instantons),
            mean_leading_log_gap=_inflation_pilot_mean(group, :leading_log_gap),
            mean_log_scale_span=_inflation_pilot_mean(group, :log_scale_span),
            mean_branch_count=_inflation_pilot_mean(group, :branch_count),
            mean_candidate_count=_inflation_pilot_mean(group, :candidate_count),
            mean_total_seconds=_inflation_pilot_mean(group, :total_seconds),
            mean_allocated_bytes=_inflation_pilot_mean(group, :allocated_bytes),
            max_allocated_bytes=maximum(row.allocated_bytes for row in group),
            mean_output_bytes=_inflation_pilot_mean(group, :output_bytes)))
    end
    reports
end

const INFLATION_PILOT_REPORT_FIELDS = (
    :h11, :screening_tier, :instanton_bin, :hierarchy_bin, :candidate_bin,
    :geometries, :successes, :branch_caps, :failures, :empty_enumerations,
    :refinement_eligible, :mean_instantons,
    :mean_leading_log_gap, :mean_log_scale_span, :mean_branch_count,
    :mean_candidate_count, :mean_total_seconds, :mean_allocated_bytes,
    :max_allocated_bytes, :mean_output_bytes)

function _inflation_pilot_write_report(path::AbstractString, reports)
    path = abspath(expanduser(path))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(INFLATION_PILOT_REPORT_FIELDS), ','))
        for report in reports
            values = (_scan_prep_csv_escape(getproperty(report, field))
                for field in INFLATION_PILOT_REPORT_FIELDS)
            println(io, join(values, ','))
        end
    end
    path
end

function _inflation_pilot_scan_options(options, geometries, shard_dir)
    args = String["--data-dir", options[:data_dir],
        "--max-branches", string(options[:max_branches]),
        "--shard-dir", shard_dir, "--shard-index", "1", "--shard-count", "1"]
    if options[:max_negative_modes] !== nothing
        append!(args, ["--max-negative-modes", string(options[:max_negative_modes])])
    elseif options[:negative_mode_range] !== nothing
        append!(args, ["--negative-mode-range",
            inflation_negative_mode_range_label(options[:negative_mode_range])])
    end
    isempty(options[:run_id]) || append!(args, ["--run-id", options[:run_id]])
    options[:retries] == 0 || append!(args, ["--retries", string(options[:retries])])
    options[:resume] && push!(args, "--resume")
    for geom in geometries
        push!(args, "--geometry")
        push!(args, string(geom.h11, ',', geom.polytope, ',', geom.frst))
    end
    _scan_prep_parse_args(args)
end

function run_inflation_pilot(options)
    data_dir = _scan_prep_validate_data_dir(options[:data_dir])
    selected = inflation_pilot_select_geometries(data_dir;
        h11_filter=options[:h11], h11_min=options[:h11_min],
        h11_max=options[:h11_max], sample_per_h11=options[:sample_per_h11],
        max_geometries=options[:max_geometries])
    isempty(selected) && throw(ArgumentError("pilot selected no geometries"))
    shard_dir = isempty(options[:shard_dir]) ?
        joinpath(data_dir, "logs", "inflation_scan_pilot_shards") :
        abspath(expanduser(options[:shard_dir]))
    report_path = isempty(options[:report]) ?
        joinpath(data_dir, "logs", "inflation_scan_pilot_report.csv") :
        abspath(expanduser(options[:report]))
    scan_ok = run_scan_prep(_inflation_pilot_scan_options(
        options, selected, shard_dir))
    rows = _inflation_pilot_terminal_rows(shard_dir)
    reports = _inflation_pilot_report_rows(rows)
    _inflation_pilot_write_report(report_path, reports)
    @printf("Pilot selected=%d completed_rows=%d strata=%d report=%s\n",
        length(selected), length(rows), length(reports), report_path)
    for report in reports
        @printf("h11=%d instantons=%s hierarchy=%s candidates=%s n=%d success=%d branch_cap=%d failed=%d mean_s=%.4g mean_alloc=%g\n",
            report.h11, report.instanton_bin, report.hierarchy_bin,
            report.candidate_bin, report.geometries, report.successes,
            report.branch_caps, report.failures, report.mean_total_seconds,
            report.mean_allocated_bytes)
    end
    (; selected, rows, reports, report_path, scan_ok)
end

if abspath(PROGRAM_FILE) == @__FILE__
    pilot = run_inflation_pilot(_inflation_pilot_parse_args(ARGS))
    pilot.scan_ok || exit(1)
end
