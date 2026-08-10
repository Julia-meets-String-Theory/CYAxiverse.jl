#!/usr/bin/env julia

"""Deterministically merge append-only inflation scan shards.

The merge preserves every attempt row in lexicographic shard-path order.  A
downstream analysis can select the final successful terminal row for each
geometry while retaining failed attempts for audit and retry diagnostics.
"""

include(joinpath(@__DIR__, "inflation_scan_prep.jl"))

function _inflation_merge_usage()
    println("""
    Usage:
      julia --project=. scripts/inflation_scan_merge_shards.jl \\
        --shard-dir PATH --output PATH [--overwrite]
    """)
end

function _inflation_merge_parse_args(args)
    shard_dir = ""
    output = ""
    overwrite = false
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            _inflation_merge_usage()
            exit(0)
        elseif arg == "--overwrite"
            overwrite = true
        elseif arg in ("--shard-dir", "--output")
            index == length(args) && error("missing value for $arg")
            if arg == "--shard-dir"
                shard_dir = args[index + 1]
            else
                output = args[index + 1]
            end
            index += 1
        else
            error("unknown argument $arg")
        end
        index += 1
    end
    isempty(shard_dir) && error("--shard-dir is required")
    isempty(output) && error("--output is required")
    (; shard_dir=abspath(expanduser(shard_dir)),
       output=abspath(expanduser(output)), overwrite)
end

function main(args)
    options = _inflation_merge_parse_args(args)
    paths = inflation_shard_paths(options.shard_dir)
    isempty(paths) && error("no CSV shards found in $(options.shard_dir)")
    inflation_merge_shards(paths, options.output; overwrite=options.overwrite)
    println("merged $(length(paths)) shard(s) into $(options.output)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
