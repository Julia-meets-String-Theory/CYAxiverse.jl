#!/usr/bin/env julia

using CYAxiverse

const GLIMMERS = CYAxiverse.glimmers

function _usage()
    println("Usage: julia --project=. scripts/glimmers_local_pilot.jl [options]")
    println("  --data-dir PATH              local CYAxiverse data root")
    println("  --h11 LIST                   comma-separated slices (default: 15,100,200,300)")
    println("  --limit N                    geometries per slice (default: 2)")
    println("  --em-divisor-index N         1-based direct-divisor column; default: minimum volume")
    println("  --output PATH                write a compact CSV summary")
    println("  --absolute-scales            permit non-positive selected local coefficients")
    println("  --require-positive           require paper-style positive selected coefficients (default)")
    println("  --help                       show this message")
end

function _next_argument(args, index, option)
    index < length(args) || throw(ArgumentError("$option requires a value"))
    args[index + 1], index + 1
end

function _parse_h11(value)
    parsed = parse.(Int, split(value, ','))
    isempty(parsed) && throw(ArgumentError("--h11 must not be empty"))
    Tuple(parsed)
end

function _parse_args(args)
    data_dir = nothing
    h11s = (15, 100, 200, 300)
    limit = 2
    em_divisor_index = nothing
    output = nothing
    signed_scale_policy = :require_positive
    i = 1
    while i <= length(args)
        option = args[i]
        if option == "--help"
            _usage()
            exit(0)
        elseif option == "--data-dir"
            value, i = _next_argument(args, i, option)
            data_dir = value
        elseif option == "--h11"
            value, i = _next_argument(args, i, option)
            h11s = _parse_h11(value)
        elseif option == "--limit"
            value, i = _next_argument(args, i, option)
            limit = parse(Int, value)
        elseif option == "--em-divisor-index"
            value, i = _next_argument(args, i, option)
            em_divisor_index = parse(Int, value)
        elseif option == "--output"
            output, i = _next_argument(args, i, option)
        elseif option == "--absolute-scales"
            signed_scale_policy = :absolute
        elseif option == "--require-positive"
            signed_scale_policy = :require_positive
        else
            throw(ArgumentError("unknown option '$option'"))
        end
        i += 1
    end
    (; data_dir, h11s, limit, em_divisor_index, output, signed_scale_policy)
end

options = _parse_args(ARGS)
results = GLIMMERS.run_local_pilot(; data_dir=options.data_dir,
    h11s=options.h11s, limit_per_h11=options.limit,
    em_divisor_index=options.em_divisor_index,
    signed_scale_policy=options.signed_scale_policy)

for result in results
    h = result.hierarchy
    p = result.photons
    println(result.geometry.path,
        " h11=", result.geometry.index.h11,
        " selected=", length(h.selected_indices),
        " light=", p.light_mode_count,
        " max_log10_g_GeVinv=", maximum(p.log10_g_GeVinv),
        " status=", result.status)
end

if options.output !== nothing
    println("Wrote ", GLIMMERS.write_pilot_csv(options.output, results))
end
