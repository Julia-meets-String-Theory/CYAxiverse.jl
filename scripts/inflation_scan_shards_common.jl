"""Append-only shard persistence for the bounded inflation scan.

This is deliberately a script-level persistence boundary.  One scan process
owns one shard file, appends one row per attempt, and flushes after each row.
The row contains both the scan-prep result and execution provenance so that a
later merge can retain failures and retries rather than silently replacing
them.
"""

if !isdefined(@__MODULE__, :INFLATION_SHARD_SCHEMA_VERSION)
    const INFLATION_SHARD_SCHEMA_VERSION = "2"
    const INFLATION_SHARD_METADATA_FIELDS = (
        :shard_schema_version, :run_id, :shard_index, :shard_count,
        :attempt, :started_unix_s, :finished_unix_s)
    const INFLATION_SHARD_FIELDS = (
        INFLATION_SHARD_METADATA_FIELDS..., SCAN_PREP_FIELDS...)
    const INFLATION_SHARD_HEADER = join(string.(INFLATION_SHARD_FIELDS), ',')

    function _inflation_shard_csv_escape(value)
        value === nothing && return ""
        text = replace(string(value), '"' => "\"\"")
        occursin(r"[,\"\n\r]", text) ? string('"', text, '"') : text
    end

    function _inflation_shard_csv_fields(line::AbstractString)
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

    function _inflation_shard_validate_header(path::AbstractString)
        first_line = open(path, "r") do io
            eof(io) ? "" : chomp(readline(io))
        end
        first_line == INFLATION_SHARD_HEADER ||
            throw(ArgumentError("inflation shard header does not match schema: $path"))
        INFLATION_SHARD_FIELDS
    end

    """Create a shard or validate an existing shard for append/resume."""
    function inflation_prepare_shard(path::AbstractString; append::Bool=false)
        path = abspath(expanduser(path))
        mkpath(dirname(path))
        if isfile(path)
            append || throw(ArgumentError(
                "inflation shard already exists; use append/resume: $path"))
            _inflation_shard_validate_header(path)
            return path
        end
        open(path, "w") do io
            write(io, INFLATION_SHARD_HEADER)
            write(io, '\n')
        end
        path
    end

    function _inflation_shard_property(summary, field::Symbol, default=nothing)
        summary !== nothing && hasproperty(summary, field) ?
            getproperty(summary, field) : default
    end

    """Append one result/provenance row and flush it before returning."""
    function inflation_append_shard_row(path::AbstractString, summary;
            run_id::AbstractString, shard_index::Int, shard_count::Int,
            attempt::Int, started_unix_s::Real, finished_unix_s::Real,
            data_dir::AbstractString, max_branches::Int,
            negative_mode_range=nothing, max_negative_modes=nothing,
            error_message="")
        shard_index >= 1 || throw(ArgumentError("shard_index must be positive"))
        shard_count >= shard_index ||
            throw(ArgumentError("shard_index must not exceed shard_count"))
        attempt > 0 || throw(ArgumentError("attempt must be positive"))
        values = map(INFLATION_SHARD_FIELDS) do field
            if field === :shard_schema_version
                INFLATION_SHARD_SCHEMA_VERSION
            elseif field === :run_id
                run_id
            elseif field === :shard_index
                shard_index
            elseif field === :shard_count
                shard_count
            elseif field === :attempt
                attempt
            elseif field === :started_unix_s
                started_unix_s
            elseif field === :finished_unix_s
                finished_unix_s
            elseif field === :data_dir
                data_dir
            elseif field === :max_branches
                max_branches
            elseif field === :negative_mode_range
                max_negative_modes === nothing ?
                    inflation_negative_mode_range_label(negative_mode_range) :
                    inflation_negative_mode_range_label(0:max_negative_modes)
            elseif field === :error && !isempty(error_message)
                error_message
            else
                _inflation_shard_property(summary, field)
            end
        end
        line = join(_inflation_shard_csv_escape.(values), ',')
        open(path, "a") do io
            write(io, line)
            write(io, '\n')
            flush(io)
        end
        path
    end

    function inflation_shard_paths(shard_dir::AbstractString)
        shard_dir = abspath(expanduser(shard_dir))
        isdir(shard_dir) || return String[]
        sort!(filter(path -> isfile(path) && endswith(path, ".csv"),
            joinpath.(shard_dir, readdir(shard_dir))))
    end

    """Return geometry keys with a matching successful terminal row."""
    function inflation_completed_shard_geometries(shard_dir::AbstractString;
            data_dir::AbstractString, max_branches::Int,
            negative_mode_range=nothing, max_negative_modes=nothing,
            contract_version::AbstractString=INFLATION_SCAN_CONTRACT_VERSION)
        completed = Set{Tuple{Int, Int, Int}}()
        required = (:contract_version, :data_dir, :max_branches,
            :negative_mode_range, :h11,
            :polytope, :frst, :status)
        for path in inflation_shard_paths(shard_dir)
            _inflation_shard_validate_header(path)
            positions = Dict(field => index for
                (index, field) in enumerate(INFLATION_SHARD_FIELDS))
            all(field -> haskey(positions, field), required) ||
                throw(ArgumentError("inflation shard is missing resume fields: $path"))
            open(path, "r") do io
                readline(io)
                for line in eachline(io)
                    isempty(strip(line)) && continue
                    fields = _inflation_shard_csv_fields(line)
                    length(fields) == length(INFLATION_SHARD_FIELDS) || continue
                    fields[positions[:contract_version]] == contract_version || continue
                    fields[positions[:data_dir]] == data_dir || continue
                    try
                        parse(Int, fields[positions[:max_branches]]) == max_branches || continue
                        expected_range = negative_mode_range === nothing && max_negative_modes === nothing ?
                            "all" : max_negative_modes === nothing ?
                            inflation_negative_mode_range_label(negative_mode_range) :
                            inflation_negative_mode_range_label(0:max_negative_modes)
                        fields[positions[:negative_mode_range]] == expected_range || continue
                        fields[positions[:status]] in ("success", "branch_cap") || continue
                        key = (parse(Int, fields[positions[:h11]]),
                            parse(Int, fields[positions[:polytope]]),
                            parse(Int, fields[positions[:frst]]))
                        push!(completed, key)
                    catch
                        continue
                    end
                end
            end
        end
        completed
    end

    """Deterministically concatenate validated shard rows into one CSV."""
    function inflation_merge_shards(paths::AbstractVector{<:AbstractString},
            output::AbstractString; overwrite::Bool=false)
        normalized = sort!(abspath.(expanduser.(String.(paths))))
        isempty(normalized) && throw(ArgumentError("no inflation shards supplied"))
        all(isfile, normalized) || throw(ArgumentError("an inflation shard is missing"))
        output = abspath(expanduser(output))
        !overwrite && isfile(output) && throw(ArgumentError(
            "merged output already exists; pass overwrite=true: $output"))
        mkpath(dirname(output))
        temporary = string(output, ".tmp.", getpid())
        try
            open(temporary, "w") do destination
                write(destination, INFLATION_SHARD_HEADER)
                write(destination, '\n')
                for path in normalized
                    _inflation_shard_validate_header(path)
                    open(path, "r") do source
                        readline(source)
                        for line in eachline(source)
                            isempty(strip(line)) || begin
                                write(destination, line)
                                write(destination, '\n')
                            end
                        end
                    end
                end
                flush(destination)
            end
            mv(temporary, output; force=overwrite)
        catch
            isfile(temporary) && rm(temporary)
            rethrow()
        end
        output
    end
end
