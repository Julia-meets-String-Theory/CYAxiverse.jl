#!/usr/bin/env julia

using CYAxiverse
using HDF5
using Printf

const GeometryIndex = CYAxiverse.structs.GeometryIndex

function _usage()
    println("""
    Usage:
      julia --project=. scripts/batch_jlm_reduced.jl [options]

    Options:
      --data-dir PATH        Data root containing h11_*/np_*/cy_*/cyax.h5.
      --h11 N                Restrict paths_cy() selection to one h11.
      --limit N              Process at most N geometries.
      --offset N             Skip the first N selected geometries.
      --geometry H,N,C       Add one explicit geometry. May be repeated.
      --threshold X          Alpha-reduction threshold. Default: 0.01.
      --starts N             Deterministic starts for non-square solves. Default: 100000.
      --force                Recompute even when minima.h5 already has Nvac.
      --hilbert              Use Hilbert-basis potential.
      --summary PATH         CSV summary output path.
      --append-summary       Append to an existing summary instead of replacing it.
    """)
end

function _parse_args(args)
    options = (data_dir="", h11=nothing, limit=nothing, offset=0,
        geometries=GeometryIndex[], threshold=0.01, starts=100_000,
        force=false, hilbert=false, summary="", append_summary=false)

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            _usage()
            exit(0)
        elseif arg == "--force"
            options = merge(options, (; force=true))
        elseif arg == "--hilbert"
            options = merge(options, (; hilbert=true))
        elseif arg == "--append-summary"
            options = merge(options, (; append_summary=true))
        elseif arg in ("--data-dir", "--h11", "--limit", "--offset", "--geometry",
                       "--threshold", "--starts", "--summary")
            i == length(args) && error("missing value for $arg")
            value = args[i + 1]
            if arg == "--data-dir"
                options = merge(options, (; data_dir=value))
            elseif arg == "--h11"
                options = merge(options, (; h11=parse(Int, value)))
            elseif arg == "--limit"
                options = merge(options, (; limit=parse(Int, value)))
            elseif arg == "--offset"
                options = merge(options, (; offset=parse(Int, value)))
            elseif arg == "--geometry"
                parts = parse.(Int, split(value, ","))
                length(parts) == 3 || error("--geometry must be H,N,C")
                push!(options.geometries, GeometryIndex(parts...))
            elseif arg == "--threshold"
                options = merge(options, (; threshold=parse(Float64, value)))
            elseif arg == "--starts"
                options = merge(options, (; starts=parse(Int, value)))
            elseif arg == "--summary"
                options = merge(options, (; summary=value))
            end
            i += 1
        else
            error("unknown option: $arg")
        end
        i += 1
    end

    options
end

function _selected_geometries(options)
    if !isempty(options[:geometries])
        geoms = copy(options[:geometries])
    else
        indexed = _indexed_geometries(options[:h11])
        scanned = _scanned_geometries(options[:h11])
        geoms = length(scanned) > length(indexed) ? scanned : indexed
    end

    offset = options[:offset]
    offset > 0 && (geoms = geoms[(min(offset + 1, length(geoms) + 1)):end])
    if options[:limit] !== nothing
        geoms = geoms[1:min(options[:limit], length(geoms))]
    end
    geoms
end

function _indexed_geometries(h11_filter)
    try
        _, pathinds = CYAxiverse.filestructure.paths_cy()
        columns = eachcol(pathinds)
        if h11_filter === nothing
            return [GeometryIndex(col...) for col in columns]
        end
        return [GeometryIndex(col...) for col in columns if col[1] == h11_filter]
    catch
        return GeometryIndex[]
    end
end

function _parse_prefixed_int(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        return parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        return nothing
    end
end

function _scanned_geometries(h11_filter)
    root = CYAxiverse.filestructure.present_dir()
    h11_dirs = if h11_filter === nothing
        filter(name -> startswith(name, "h11_"), readdir(root))
    else
        [string("h11_", lpad(h11_filter, 3, "0"))]
    end

    geoms = GeometryIndex[]
    for h11_dir in sort(h11_dirs)
        h11 = _parse_prefixed_int(h11_dir, "h11_")
        h11 === nothing && continue
        h11_path = joinpath(root, h11_dir)
        isdir(h11_path) || continue
        for np_dir in sort(filter(name -> startswith(name, "np_"), readdir(h11_path)))
            polytope = _parse_prefixed_int(np_dir, "np_")
            polytope === nothing && continue
            np_path = joinpath(h11_path, np_dir)
            for cy_dir in sort(filter(name -> startswith(name, "cy_"), readdir(np_path)))
                frst = _parse_prefixed_int(cy_dir, "cy_")
                frst === nothing && continue
                isfile(joinpath(np_path, cy_dir, "cyax.h5")) || continue
                push!(geoms, GeometryIndex(h11, polytope, frst))
            end
        end
    end
    geoms
end

function _has_completed_minimum(geom_idx; hilbert::Bool=false)
    path = CYAxiverse.filestructure.minfile(geom_idx)
    isfile(path) || return false
    h5open(path, "r") do file
        hilbert ? haskey(file, "hilbert/Nvac") : haskey(file, "Nvac")
    end
end

function _write_summary_header(path; append::Bool=false)
    isempty(path) && return
    append && isfile(path) && return
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "h11,polytope,frst,status,Nvac,issquare,extra_rows,det_QTilde,seconds,message")
    end
end

function _append_summary(path, geom_idx, status, min_data, seconds, message="")
    isempty(path) && return
    nmin = min_data === nothing ? "" : string(min_data.N_min)
    issquare = min_data === nothing ? "" : string(Int(min_data isa CYAxiverse.structs.Min_JLM_Square))
    extra_rows = (min_data === nothing || min_data isa CYAxiverse.structs.Min_JLM_Square) ? "" : string(min_data.extra_rows)
    det_qtilde = min_data === nothing ? "" : string(min_data.det_QTilde)
    clean_message = replace(message, "," => ";", "\n" => " ")
    open(path, "a") do io
        @printf(io, "%d,%d,%d,%s,%s,%s,%s,%s,%.6f,%s\n",
            geom_idx.h11, geom_idx.polytope, geom_idx.frst, status, nmin,
            issquare, extra_rows, det_qtilde, seconds, clean_message)
    end
end

function run_batch(options)
    data_dir = CYAxiverse.filestructure.resolve_data_dir(options[:data_dir])
    options = merge(options, (; data_dir))
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir

    geoms = _selected_geometries(options)
    isempty(geoms) && error("no geometries selected")

    summary = options[:summary]
    if isempty(summary)
        summary = joinpath(CYAxiverse.filestructure.log_dir(), "jlm_reduced_batch.csv")
    end
    _write_summary_header(summary; append=options[:append_summary])

    @printf("JLM reduced batch: %d geometries\n", length(geoms))
    @printf("data_dir=%s\n", CYAxiverse.filestructure.present_dir())
    @printf("threshold=%g starts=%d hilbert=%s force=%s\n",
        options[:threshold], options[:starts], string(options[:hilbert]), string(options[:force]))
    @printf("summary=%s\n", summary)

    completed = 0
    skipped = 0
    failed = 0
    total_start = time()

    for (idx, geom_idx) in enumerate(geoms)
        @printf("[%d/%d] h11=%d polytope=%d frst=%d ",
            idx, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst)
        start = time()
        if !options[:force] && _has_completed_minimum(geom_idx; hilbert=options[:hilbert])
            seconds = time() - start
            skipped += 1
            println("skipped")
            _append_summary(summary, geom_idx, "skipped", nothing, seconds)
            continue
        end

        try
            min_data = CYAxiverse.jlm_reduced.minimize_save(geom_idx;
                threshold=options[:threshold], starts=options[:starts],
                hilbert=options[:hilbert])
            seconds = time() - start
            completed += 1
            @printf("done Nvac=%d det_QTilde=%d %.3fs\n",
                min_data.N_min, min_data.det_QTilde, seconds)
            _append_summary(summary, geom_idx, "done", min_data, seconds)
        catch err
            seconds = time() - start
            failed += 1
            println("failed")
            _append_summary(summary, geom_idx, "failed", nothing, seconds,
                sprint(showerror, err))
        end
        GC.gc(false)
    end

    @printf("Finished: done=%d skipped=%d failed=%d elapsed=%.3fs\n",
        completed, skipped, failed, time() - total_start)
    failed == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = _parse_args(ARGS)
    success = run_batch(options)
    success || exit(1)
end
