#!/usr/bin/env julia

"""Prepare bounded inflation-screening inputs one geometry at a time.

The numerical sequence lives in `inflation_scan_common.jl` and is deliberately
script-only.  This driver adds deterministic geometry selection, streamed CSV
summaries, append-only shard persistence, and a conservative resume check.  It
does not run trajectory refinement, create workers, or write geometry files.
"""

# The pilot and shard-merge CLIs include this driver. Keep repeated includes
# idempotent when those entry points are loaded together in tests.
if !isdefined(@__MODULE__, :INFLATION_SCAN_PREP_LOADED)
const INFLATION_SCAN_PREP_LOADED = true

include(joinpath(@__DIR__, "inflation_scan_common.jl"))

using Printf

const SCAN_PREP_FIELDS = (
    :contract_version, :diagnostic_schema_version, :measurement_scope,
    :data_dir, :max_branches,
    :h11, :polytope, :frst, :status, :error,
    :instantons, :selected_instantons, :qtilde_det, :negative_mode_range,
    :structured_charge_validated, :structured_fallback_reason,
    :search_classification, :mask_count, :masks_visited, :masks_skipped,
    :lattice_copy_count, :lattice_copies_visited,
    :leading_log_gap, :log_scale_span, :strong_hierarchy,
    :branch_count, :leading_minima_count, :saddle_count,
    :candidate_slowroll_saddles, :least_tachyonic_min_eta,
    :least_tachyonic_epsilon, :best_min_eta, :best_epsilon,
    :best_abs_min_eta, :mass_min, :mass_max, :negative_mass_count,
    :stage_load_s, :stage_select_s, :stage_hierarchy_s, :stage_factor_s,
    :stage_mass_basis_s, :stage_branches_s, :stage_classify_s,
    :stage_load_output_bytes, :stage_select_output_bytes,
    :stage_hierarchy_output_bytes, :stage_factor_output_bytes,
    :stage_mass_basis_output_bytes, :stage_classify_output_bytes,
    :stage_allocated_bytes, :stage_output_bytes, :total_seconds)
const SCAN_PREP_HEADER = join(string.(SCAN_PREP_FIELDS), ',')

include(joinpath(@__DIR__, "inflation_scan_shards_common.jl"))

function _scan_prep_usage()
    println("""
    Usage:
      julia --project=. scripts/inflation_scan_prep.jl [options]

    Options:
      --data-dir PATH       Data root containing h11_*/np_*/cy_*/cyax.h5.
      --h11 N               Restrict discovered geometries to one h11.
      --limit N             Process at most N selected geometries.
      --offset N            Skip the first N selected geometries.
      --geometry H,P,F      Process an explicit geometry; may be repeated.
      --max-branches N      Bound leading branch enumeration. Default: 1000000.
      --negative-mode-range K[:K]
                            Search only leading branches in an index range;
                            default: all.
      --max-negative-modes K
                            Search leading branches with index 0 through K.
      --summary PATH        Stream one preparation row per geometry to PATH.
      --append-summary      Append to an existing compatible summary.
      --resume              Skip successful/branch-cap rows matching this run configuration.
      --shard-dir PATH      Write one append-only shard in this directory.
      --shard-index N       One-based shard index; default: 1.
      --shard-count N       Total deterministic shard count; default: 1.
      --run-id ID           Provenance label written to every shard row.
      --retries N           Retry failed geometries N times; default: 0.
    """)
end

function _scan_prep_parse_args(args)
    options = Dict{Symbol, Any}(
        :data_dir => get(ENV, "CYAXIVERSE_DATA_DIR", ""),
        :h11 => nothing, :limit => nothing, :offset => 0,
        :geometries => GeometryIndex[], :max_branches => 1_000_000,
        :negative_mode_range => nothing, :max_negative_modes => nothing,
        :summary => "", :append_summary => false, :resume => false,
        :shard_dir => "", :shard_index => 1, :shard_count => 1,
        :run_id => "", :retries => 0)
    valued = ("--data-dir", "--h11", "--limit", "--offset", "--geometry",
        "--max-branches", "--negative-mode-range", "--max-negative-modes",
        "--summary", "--shard-dir", "--shard-index",
        "--shard-count", "--run-id", "--retries")
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            _scan_prep_usage()
            exit(0)
        elseif arg == "--append-summary"
            options[:append_summary] = true
        elseif arg == "--resume"
            options[:resume] = true
        elseif arg in valued
            index == length(args) && error("missing value for $arg")
            value = args[index + 1]
            if arg == "--data-dir"
                options[:data_dir] = value
            elseif arg == "--h11"
                options[:h11] = parse(Int, value)
            elseif arg == "--limit"
                options[:limit] = parse(Int, value)
            elseif arg == "--offset"
                options[:offset] = parse(Int, value)
            elseif arg == "--geometry"
                parts = parse.(Int, split(value, ','))
                length(parts) == 3 || error("--geometry must be H,P,F")
                push!(options[:geometries], GeometryIndex(parts...))
            elseif arg == "--max-branches"
                options[:max_branches] = parse(Int, value)
            elseif arg == "--negative-mode-range"
                options[:negative_mode_range] =
                    inflation_parse_negative_mode_range(value)
            elseif arg == "--max-negative-modes"
                options[:max_negative_modes] = parse(Int, value)
            elseif arg == "--summary"
                options[:summary] = value
            elseif arg == "--shard-dir"
                options[:shard_dir] = value
            elseif arg == "--shard-index"
                options[:shard_index] = parse(Int, value)
            elseif arg == "--shard-count"
                options[:shard_count] = parse(Int, value)
            elseif arg == "--run-id"
                options[:run_id] = value
            elseif arg == "--retries"
                options[:retries] = parse(Int, value)
            end
            index += 1
        else
            error("unknown option: $arg")
        end
        index += 1
    end
    isempty(options[:data_dir]) && error("--data-dir or CYAXIVERSE_DATA_DIR is required")
    options[:data_dir] = abspath(expanduser(options[:data_dir]))
    options[:h11] === nothing || options[:h11] > 0 || error("--h11 must be positive")
    options[:limit] === nothing || options[:limit] > 0 || error("--limit must be positive")
    options[:offset] >= 0 || error("--offset must be nonnegative")
    options[:max_branches] > 0 || error("--max-branches must be positive")
    options[:negative_mode_range] === nothing ||
        (first(options[:negative_mode_range]) >= 0 &&
         last(options[:negative_mode_range]) >= first(options[:negative_mode_range])) ||
        error("--negative-mode-range must be nonnegative and ordered")
    options[:max_negative_modes] === nothing ||
        options[:max_negative_modes] >= 0 ||
        error("--max-negative-modes must be nonnegative")
    options[:negative_mode_range] === nothing || options[:max_negative_modes] === nothing ||
        error("use only one of --negative-mode-range and --max-negative-modes")
    isempty(options[:summary]) || isempty(options[:shard_dir]) ||
        error("--summary and --shard-dir are mutually exclusive")
    options[:shard_count] > 0 || error("--shard-count must be positive")
    1 <= options[:shard_index] <= options[:shard_count] ||
        error("--shard-index must be between 1 and --shard-count")
    isempty(options[:shard_dir]) && options[:shard_count] != 1 &&
        error("--shard-count requires --shard-dir")
    options[:retries] >= 0 || error("--retries must be nonnegative")
    options
end

function _scan_prep_validate_data_dir(data_dir::AbstractString)
    isempty(strip(data_dir)) && throw(ArgumentError("data_dir must be explicitly provided"))
    data_dir in ("/", homedir()) &&
        throw(ArgumentError("refusing to use root-like data directory: $data_dir"))
    isdir(data_dir) || throw(ArgumentError("data directory does not exist: $data_dir"))
    data_dir
end

function _scan_prep_parse_prefixed_int(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        nothing
    end
end

"""Discover existing geometry files in deterministic h11/np/cy order."""
function _scan_prep_discover(data_dir::AbstractString, h11_filter)
    h11_dirs = filter(name -> startswith(name, "h11_"), readdir(data_dir))
    geoms = GeometryIndex[]
    for h11_dir in sort(h11_dirs)
        h11 = _scan_prep_parse_prefixed_int(h11_dir, "h11_")
        h11 === nothing && continue
        h11_filter === nothing || h11 == h11_filter || continue
        h11_path = joinpath(data_dir, h11_dir)
        isdir(h11_path) || continue
        for np_dir in sort(filter(name -> startswith(name, "np_"), readdir(h11_path)))
            polytope = _scan_prep_parse_prefixed_int(np_dir, "np_")
            polytope === nothing && continue
            np_path = joinpath(h11_path, np_dir)
            isdir(np_path) || continue
            for cy_dir in sort(filter(name -> startswith(name, "cy_"), readdir(np_path)))
                frst = _scan_prep_parse_prefixed_int(cy_dir, "cy_")
                frst === nothing && continue
                path = joinpath(np_path, cy_dir, "cyax.h5")
                isfile(path) && push!(geoms, GeometryIndex(h11, polytope, frst))
            end
        end
    end
    geoms
end

function _scan_prep_selected_geometries(options)
    geoms = isempty(options[:geometries]) ?
        _scan_prep_discover(options[:data_dir], options[:h11]) :
        copy(options[:geometries])
    geoms = unique(geoms)
    sort!(geoms, by=geom -> (geom.h11, geom.polytope, geom.frst))
    first_index = min(options[:offset] + 1, length(geoms) + 1)
    geoms = geoms[first_index:end]
    geoms = options[:limit] === nothing ? geoms :
        geoms[1:min(options[:limit], length(geoms))]
    options[:shard_count] == 1 && return geoms
    [geom for (index, geom) in enumerate(geoms)
        if mod(index - 1, options[:shard_count]) + 1 == options[:shard_index]]
end

function _scan_prep_csv_escape(value)
    value === nothing && return ""
    text = replace(string(value), '"' => "\"\"")
    occursin(r"[,\"\n\r]", text) ? string('"', text, '"') : text
end

function _scan_prep_csv_fields(line::AbstractString)
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

function _scan_prep_write_header(path::AbstractString; append::Bool=false)
    path = abspath(expanduser(path))
    mkpath(dirname(path))
    if append && isfile(path)
        first_line = open(path, "r") do io
            eof(io) ? "" : chomp(readline(io))
        end
        first_line == SCAN_PREP_HEADER ||
            throw(ArgumentError("summary header does not match scan-prep schema: $path"))
        return path
    end
    open(path, "w") do io
        println(io, SCAN_PREP_HEADER)
    end
    path
end

function _scan_prep_completed(path::AbstractString; data_dir, max_branches,
        negative_mode_range=nothing, max_negative_modes=nothing)
    completed = Set{Tuple{Int, Int, Int}}()
    isfile(path) || return completed
    lines = readlines(path)
    isempty(lines) && return completed
    header = _scan_prep_csv_fields(lines[1])
    positions = Dict(Symbol(name) => index for (index, name) in enumerate(header))
    required = (:contract_version, :data_dir, :max_branches,
        :negative_mode_range, :h11, :polytope, :frst, :status)
    all(name -> haskey(positions, name), required) ||
        throw(ArgumentError("summary is missing scan-prep resume fields: $path"))
    for line in @view lines[2:end]
        isempty(strip(line)) && continue
        fields = _scan_prep_csv_fields(line)
        length(fields) >= length(header) || continue
        fields[positions[:contract_version]] == INFLATION_SCAN_CONTRACT_VERSION || continue
        fields[positions[:data_dir]] == data_dir || continue
        try
            parse(Int, fields[positions[:max_branches]]) == max_branches || continue
            expected_range = negative_mode_range === nothing && max_negative_modes === nothing ?
                "all" : max_negative_modes === nothing ?
                inflation_negative_mode_range_label(negative_mode_range) :
                inflation_negative_mode_range_label(0:max_negative_modes)
            fields[positions[:negative_mode_range]] == expected_range || continue
            status = fields[positions[:status]]
            status in ("success", "branch_cap") || continue
            key = (parse(Int, fields[positions[:h11]]),
                parse(Int, fields[positions[:polytope]]),
                parse(Int, fields[positions[:frst]]))
            push!(completed, key)
        catch
            continue
        end
    end
    completed
end

function _scan_prep_append(path::AbstractString, summary; data_dir, max_branches,
        negative_mode_range=nothing, max_negative_modes=nothing, error="")
    values = Any[]
    for field in SCAN_PREP_FIELDS
        value = if field === :data_dir
            data_dir
        elseif field === :max_branches
            max_branches
        elseif field === :negative_mode_range
            max_negative_modes === nothing ?
                inflation_negative_mode_range_label(negative_mode_range) :
                inflation_negative_mode_range_label(0:max_negative_modes)
        elseif field === :error
            error
        elseif hasproperty(summary, field)
            getproperty(summary, field)
        else
            nothing
        end
        push!(values, _scan_prep_csv_escape(value))
    end
    open(path, "a") do io
        println(io, join(values, ','))
        flush(io)
    end
end

function run_scan_prep(options)
    data_dir = _scan_prep_validate_data_dir(options[:data_dir])
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    geoms = _scan_prep_selected_geometries(options)
    isempty(geoms) && throw(ArgumentError("no geometries selected"))
    shard_mode = !isempty(options[:shard_dir])
    summary_path = shard_mode ? "" : (isempty(options[:summary]) ?
        joinpath(data_dir, "logs", "inflation_scan_prep.csv") :
        abspath(expanduser(options[:summary])))
    shard_path = ""
    run_id = isempty(options[:run_id]) ?
        string("scan-prep-", getpid(), "-", round(Int, time())) : options[:run_id]
    if shard_mode
        shard_dir = abspath(expanduser(options[:shard_dir]))
        shard_path = joinpath(shard_dir, string("inflation_scan_prep_shard_",
            lpad(options[:shard_index], 4, '0'), "_of_",
            lpad(options[:shard_count], 4, '0'), ".csv"))
        options[:resume] && !isfile(shard_path) &&
            throw(ArgumentError("--resume requires an existing shard: $shard_path"))
        inflation_prepare_shard(shard_path;
            append=options[:append_summary] || options[:resume])
        resumable = options[:resume] ? inflation_completed_shard_geometries(
            shard_dir; data_dir, max_branches=options[:max_branches],
            negative_mode_range=options[:negative_mode_range],
            max_negative_modes=options[:max_negative_modes]) :
            Set{Tuple{Int, Int, Int}}()
    else
        options[:resume] && !isfile(summary_path) &&
            throw(ArgumentError("--resume requires an existing summary: $summary_path"))
        _scan_prep_write_header(summary_path;
            append=options[:append_summary] || options[:resume])
        resumable = options[:resume] ? _scan_prep_completed(summary_path;
            data_dir, max_branches=options[:max_branches],
            negative_mode_range=options[:negative_mode_range],
            max_negative_modes=options[:max_negative_modes]) : Set{Tuple{Int, Int, Int}}()
    end

    range_label = options[:max_negative_modes] === nothing ?
        inflation_negative_mode_range_label(options[:negative_mode_range]) :
        inflation_negative_mode_range_label(0:options[:max_negative_modes])
    @printf("Inflation scan-prep: %d geometries max_branches=%d leading_index=%s\n",
        length(geoms), options[:max_branches], range_label)
    if shard_mode
        @printf("data_dir=%s\nshard=%s\nrun_id=%s\n", data_dir, shard_path, run_id)
    else
        @printf("data_dir=%s\nsummary=%s\n", data_dir, summary_path)
    end
    successes = 0
    branch_caps = 0
    failed = 0
    skipped = 0
    for (index, geom_idx) in enumerate(geoms)
        key = (geom_idx.h11, geom_idx.polytope, geom_idx.frst)
        if key in resumable
            skipped += 1
            @printf("[%d/%d] h11=%d polytope=%d frst=%d skipped resume\n",
                index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst)
            continue
        end
        final_summary = nothing
        final_message = ""
        final_attempt = 0
        for attempt in 1:(options[:retries] + 1)
            final_attempt = attempt
            started = time()
            measurement_scope = index == 1 ? :cold : :warm
            summary = try
                run_geometry(geom_idx; max_branches=options[:max_branches],
                    measurement_scope,
                    negative_mode_range=options[:negative_mode_range],
                    max_negative_modes=options[:max_negative_modes])
            catch error
                failure = _scan_prep_error_status(error)
                (; contract_version=INFLATION_SCAN_CONTRACT_VERSION,
                   diagnostic_schema_version=INFLATION_DIAGNOSTIC_SCHEMA_VERSION,
                   measurement_scope,
                   h11=geom_idx.h11, polytope=geom_idx.polytope, frst=geom_idx.frst,
                   status=failure.status, total_seconds=time() - started), failure.message
            end
            if !(summary isa NamedTuple)
                result, message = summary
                final_summary, final_message = result, message
            else
                final_summary = summary
                final_message = ""
            end
            if shard_mode
                inflation_append_shard_row(shard_path, final_summary;
                    run_id, shard_index=options[:shard_index],
                    shard_count=options[:shard_count], attempt,
                    started_unix_s=started, finished_unix_s=time(), data_dir,
                    max_branches=options[:max_branches],
                    negative_mode_range=options[:negative_mode_range],
                    max_negative_modes=options[:max_negative_modes],
                    error_message=final_message)
            else
                _scan_prep_append(summary_path, final_summary; data_dir,
                    max_branches=options[:max_branches],
                    negative_mode_range=options[:negative_mode_range],
                    max_negative_modes=options[:max_negative_modes],
                    error=final_message)
            end
            final_summary.status in (:failed,) && attempt <= options[:retries] &&
                continue
            break
        end
        if final_summary.status == :failed
            failed += 1
        elseif final_summary.status == :branch_cap
            branch_caps += 1
        else
            successes += 1
        end
        branches = hasproperty(final_summary, :branch_count) ? final_summary.branch_count : 0
        @printf("[%d/%d] h11=%d polytope=%d frst=%d status=%s attempts=%d branches=%d seconds=%.3f\n",
            index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst,
            final_summary.status, final_attempt, branches, final_summary.total_seconds)
    end
    @printf("Finished: success=%d branch_cap=%d skipped=%d failed=%d\n",
        successes, branch_caps, skipped, failed)
    failed == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_scan_prep(_scan_prep_parse_args(ARGS)) || exit(1)
end

end
