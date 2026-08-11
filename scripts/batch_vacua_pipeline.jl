#!/usr/bin/env julia

using CYAxiverse
using Distributed
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "vacua_pipeline.jl"))

const GeometryIndex = CYAxiverse.structs.GeometryIndex
const DEFAULT_BLAS_THREADS = 1
const VacuaJobOptions = NamedTuple{(:threshold, :starts, :residual_tolerance,
    :merge_tolerance, :max_iterations, :method, :max_branches, :force),
    Tuple{Float64, Int, Float64, Float64, Int, Symbol, Int, Bool}}
const VacuaJob = NamedTuple{(:geom_idx, :data_dir, :options),
    Tuple{GeometryIndex, String, VacuaJobOptions}}

"""Print command-line usage for the vacua batch runner."""
function _vacua_usage()
    println("""
    Usage:
      julia --project=. scripts/batch_vacua_pipeline.jl [options]

    Options:
      --data-dir PATH              Required data root containing h11_*/np_*/cy_*/cyax.h5.
      --h11 N                      Restrict selection to one h11.
      --limit N                    Process at most N selected geometries.
      --offset N                   Skip the first N selected geometries.
      --geometry H,N,C             Add one explicit geometry. May be repeated.
      --threshold X                Vacua threshold. Default: 0.5.
        --starts N                   Legacy subspace runs/search budget. Default: 10000.
      --residual-tolerance X       Recorded residual tolerance. Default: 1e-10.
      --merge-tolerance X          Recorded merge tolerance. Default: 1e-7.
      --max-iterations N           Recorded iteration bound. Default: 200.
        --method NAME                Search method: auto, legacy, leading_branches, or reduced_jlm.
                            Default: legacy.
        --max-branches N             Bound leading-branch enumeration. Default: 1000000.
      --summary PATH               CSV summary output path.
      --append-summary             Append to an existing summary.
        --workers N                  Geometry worker processes. Default: 1.
        --blas-threads N             BLAS threads per worker. Default: 1.
        --batch-size N               Maximum geometries dispatched before checkpointing. Default: 16.
      --force                      Recompute and replace an existing matching result.
      --dry-run                    Select and classify geometries without numerical work.
    """)
end

"""Parse batch-runner command-line arguments into a typed options tuple."""
function _vacua_parse_args(args)
    options = (data_dir="", h11=nothing, limit=nothing, offset=0,
        geometries=GeometryIndex[], threshold=0.5, starts=10_000,
        residual_tolerance=1e-10, merge_tolerance=1e-7, max_iterations=200,
        method=:legacy, max_branches=1_000_000, summary="",
        append_summary=false, workers=1, blas_threads=DEFAULT_BLAS_THREADS, batch_size=16,
        force=false, dry_run=false)

    valued = ("--data-dir", "--h11", "--limit", "--offset", "--geometry",
        "--threshold", "--starts", "--residual-tolerance", "--merge-tolerance",
        "--max-iterations", "--method", "--max-branches", "--summary", "--workers",
        "--blas-threads", "--batch-size")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            _vacua_usage()
            exit(0)
        elseif arg == "--force"
            options = merge(options, (; force=true))
        elseif arg == "--dry-run"
            options = merge(options, (; dry_run=true))
        elseif arg == "--append-summary"
            options = merge(options, (; append_summary=true))
        elseif arg in valued
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
            elseif arg == "--residual-tolerance"
                options = merge(options, (; residual_tolerance=parse(Float64, value)))
            elseif arg == "--merge-tolerance"
                options = merge(options, (; merge_tolerance=parse(Float64, value)))
            elseif arg == "--max-iterations"
                options = merge(options, (; max_iterations=parse(Int, value)))
            elseif arg == "--method"
                options = merge(options, (; method=Symbol(value)))
            elseif arg == "--max-branches"
                options = merge(options, (; max_branches=parse(Int, value)))
            elseif arg == "--summary"
                options = merge(options, (; summary=value))
            elseif arg == "--workers"
                options = merge(options, (; workers=parse(Int, value)))
            elseif arg == "--blas-threads"
                options = merge(options, (; blas_threads=parse(Int, value)))
            elseif arg == "--batch-size"
                options = merge(options, (; batch_size=parse(Int, value)))
            end
            i += 1
        else
            error("unknown option: $arg")
        end
        i += 1
    end
    options
end

"""Validate batch-runner options before selecting or processing geometries."""
function _vacua_validate_options(options)
    data_dir = CYAxiverse.filestructure.resolve_data_dir(options[:data_dir])
    _validate_data_dir(data_dir)
    options[:threshold] >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    options[:starts] > 0 || throw(ArgumentError("starts must be positive"))
    options[:residual_tolerance] > 0 || throw(ArgumentError("residual-tolerance must be positive"))
    options[:merge_tolerance] > 0 || throw(ArgumentError("merge-tolerance must be positive"))
    options[:max_iterations] > 0 || throw(ArgumentError("max-iterations must be positive"))
    options[:offset] >= 0 || throw(ArgumentError("offset must be nonnegative"))
    options[:limit] === nothing || options[:limit] > 0 ||
        throw(ArgumentError("limit must be positive"))
    options[:method] in (:auto, :legacy, :leading_branches, :reduced_jlm) ||
        throw(ArgumentError("unsupported method: $(options[:method])"))
    options[:max_branches] > 0 || throw(ArgumentError("max-branches must be positive"))
    options[:workers] > 0 || throw(ArgumentError("workers must be positive"))
    options[:blas_threads] === nothing || options[:blas_threads] > 0 ||
        throw(ArgumentError("blas-threads must be positive"))
    options[:batch_size] > 0 || throw(ArgumentError("batch-size must be positive"))
    merge(options, (; data_dir))
end

"""Load indexed geometries, optionally restricted to one h11 value."""
function _vacua_indexed_geometries(h11_filter)
    try
        _, pathinds = CYAxiverse.filestructure.paths_cy()
        geoms = [GeometryIndex(col...) for col in eachcol(pathinds)]
        return sort!([geom for geom in geoms if h11_filter === nothing || geom.h11 == h11_filter],
            by=geom -> (geom.h11, geom.polytope, geom.frst))
    catch
        GeometryIndex[]
    end
end

"""Parse an integer suffix when a name starts with the requested prefix."""
function _vacua_parse_prefixed_int(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        nothing
    end
end

"""Discover geometry files by scanning the configured data directory."""
function _vacua_scanned_geometries(h11_filter)
    root = CYAxiverse.filestructure.present_dir()
    h11_dirs = h11_filter === nothing ?
        filter(name -> startswith(name, "h11_"), readdir(root)) :
        [string("h11_", lpad(h11_filter, 3, "0"))]
    geoms = GeometryIndex[]
    for h11_dir in sort(h11_dirs)
        h11 = _vacua_parse_prefixed_int(h11_dir, "h11_")
        h11 === nothing && continue
        h11_path = joinpath(root, h11_dir)
        isdir(h11_path) || continue
        for np_dir in sort(filter(name -> startswith(name, "np_"), readdir(h11_path)))
            polytope = _vacua_parse_prefixed_int(np_dir, "np_")
            polytope === nothing && continue
            np_path = joinpath(h11_path, np_dir)
            for cy_dir in sort(filter(name -> startswith(name, "cy_"), readdir(np_path)))
                frst = _vacua_parse_prefixed_int(cy_dir, "cy_")
                frst === nothing && continue
                isfile(joinpath(np_path, cy_dir, "cyax.h5")) || continue
                push!(geoms, GeometryIndex(h11, polytope, frst))
            end
        end
    end
    geoms
end

"""Select, offset, and limit the geometries requested by batch options."""
function _vacua_selected_geometries(options)
    geoms = if isempty(options[:geometries])
        indexed = _vacua_indexed_geometries(options[:h11])
        scanned = _vacua_scanned_geometries(options[:h11])
        length(scanned) > length(indexed) ? scanned : indexed
    else
        copy(options[:geometries])
    end
    start_index = options[:offset] + 1
    geoms = start_index > length(geoms) ? GeometryIndex[] : geoms[start_index:end]
    options[:limit] === nothing && return geoms
    geoms[1:min(options[:limit], length(geoms))]
end

"""Construct the persisted pipeline configuration for batch options."""
function _vacua_config(options)
    _pipeline_config(; threshold=options[:threshold], starts=options[:starts],
        residual_tolerance=options[:residual_tolerance],
        merge_tolerance=options[:merge_tolerance],
        max_iterations=options[:max_iterations], method=options[:method],
        max_branches=options[:max_branches])
end

"""Create a CSV summary header unless an append operation already has one."""
function _vacua_summary_header(path; append=false)
    isempty(path) && return
    append && isfile(path) && return
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "h11,polytope,frst,status,estimate,verified,issquare,seconds,message")
    end
end

"""Convert a batch-summary field to a single CSV-safe string."""
function _vacua_csv_value(value)
    value === nothing && return ""
    replace(string(value), '"' => "\"\"", '\n' => ' ', '\r' => ' ')
end

"""Append and flush one geometry result to the batch CSV summary."""
function _vacua_append_summary(path, geom_idx, status; estimate=nothing,
        verified=nothing, issquare=nothing, seconds=0.0, message="")
    isempty(path) && return
    open(path, "a") do io
        values = (geom_idx.h11, geom_idx.polytope, geom_idx.frst, status,
            estimate, verified, issquare, @sprintf("%.6f", seconds), message)
        println(io, join(("\"$(_vacua_csv_value(value))\"" for value in values), ","))
        flush(io)
    end
end

"""Classify a geometry result as missing, new, incomplete, mismatched, or reusable."""
function _vacua_result_state(path, config)
    isfile(path) || return :missing
    _has_pipeline_result(path; config=config) && return :matching
    _has_pipeline_result(path) && return :config_mismatch
    _has_pipeline_group(path) ? :incomplete : :new
end

"""Run one geometry job and return a serializable success or failure record."""
function _vacua_worker_job(job)
    geom_idx, data_dir, options = job
    started_at = time()
    try
        result = compute_vacua_data(geom_idx.h11, geom_idx.polytope, geom_idx.frst,
            data_dir; threshold=options.threshold, starts=options.starts,
            residual_tolerance=options.residual_tolerance,
            merge_tolerance=options.merge_tolerance,
            max_iterations=options.max_iterations, method=options.method,
            max_branches=options.max_branches, force=options.force)
        estimate = result["vacua_estimate"]
        locations = result["vacua_locations"]
        return (; geom_idx, status="done", estimate=estimate.vac,
            verified=haskey(locations, "vac") ? locations["vac"] : nothing,
            issquare=estimate.issquare, seconds=time() - started_at, message="")
    catch err
        return (; geom_idx, status="failed", estimate=nothing, verified=nothing,
            issquare=nothing, seconds=time() - started_at,
            message=sprint(showerror, err))
    finally
        GC.gc(false)
    end
end

"""Start geometry workers and configure their BLAS thread counts."""
function _vacua_start_workers(options)
    workers = options[:workers]
    if workers == 1
        options[:blas_threads] === nothing ||
            LinearAlgebra.BLAS.set_num_threads(options[:blas_threads])
        return Int[]
    end
    project = Base.active_project()
    project === nothing && throw(ArgumentError("parallel execution requires an active Julia project"))
    worker_ids = addprocs(workers; exeflags="--project=$(project)")
    script = joinpath(@__DIR__, "vacua_pipeline.jl")
    blas_threads = options[:blas_threads]
    worker_setup = quote
        using CYAxiverse
        using LinearAlgebra
        include($script)
        function _vacua_worker_job(job)
            geom_idx, data_dir, options = job
            started_at = time()
            try
                result = compute_vacua_data(geom_idx.h11, geom_idx.polytope, geom_idx.frst,
                    data_dir; threshold=options.threshold, starts=options.starts,
                    residual_tolerance=options.residual_tolerance,
                    merge_tolerance=options.merge_tolerance,
                    max_iterations=options.max_iterations, method=options.method,
                    max_branches=options.max_branches, force=options.force)
                estimate = result["vacua_estimate"]
                locations = result["vacua_locations"]
                (; geom_idx, status="done", estimate=estimate.vac,
                    verified=haskey(locations, "vac") ? locations["vac"] : nothing,
                    issquare=estimate.issquare, seconds=time() - started_at, message="")
            catch err
                (; geom_idx, status="failed", estimate=nothing, verified=nothing,
                    issquare=nothing, seconds=time() - started_at,
                    message=sprint(showerror, err))
            finally
                GC.gc(false)
            end
        end
    end
    try
        for worker_id in worker_ids
            remotecall_eval(Main, worker_id, worker_setup)
            blas_threads === nothing ||
                remotecall_wait(LinearAlgebra.BLAS.set_num_threads, worker_id, blas_threads)
        end
    catch
        rmprocs(worker_ids; waitfor=5)
        rethrow()
    end
    worker_ids
end

"""Run the bounded, resumable vacua batch and stream results to CSV."""
function run_vacua_batch(options)
    options = _vacua_validate_options(options)
    ENV["CYAXIVERSE_DATA_DIR"] = options[:data_dir]
    geoms = _vacua_selected_geometries(options)
    isempty(geoms) && error("no geometries selected")

    summary = isempty(options[:summary]) ?
        joinpath(CYAxiverse.filestructure.present_dir(), "logs", "vacua_pipeline_batch.csv") :
        abspath(expanduser(options[:summary]))
    _vacua_summary_header(summary; append=options[:append_summary])
    config = _vacua_config(options)

    if options[:force]
        println("Force targets:")
        for geom_idx in geoms
            println("  ", CYAxiverse.filestructure.cyax_file(geom_idx))
        end
    end

    @printf("Vacua pipeline batch: %d geometries\n", length(geoms))
    @printf("data_dir=%s\n", CYAxiverse.filestructure.present_dir())
    @printf("threshold=%g starts=%d method=%s workers=%d blas_threads=%s force=%s dry_run=%s\n",
        options[:threshold], options[:starts], options[:method],
        options[:workers], options[:blas_threads] === nothing ? "default" : options[:blas_threads],
        options[:force], options[:dry_run])
    @printf("summary=%s\n", summary)

    done = 0
    skipped = 0
    blocked = 0
    failed = 0
    total_start = time()
    worker_ids = Int[]
    active_jobs = VacuaJob[]
    try
        for (index, geom_idx) in enumerate(geoms)
            path = CYAxiverse.filestructure.cyax_file(geom_idx)
            start = time()
            state = try
                _vacua_result_state(path, config)
            catch err
                failed += 1
                seconds = time() - start
                message = sprint(showerror, err)
                @printf("[%d/%d] h11=%d polytope=%d frst=%d failed: %s\n",
                    index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst, message)
                _vacua_append_summary(summary, geom_idx, "failed"; seconds=seconds,
                    message=message)
                continue
            end
            if state == :matching && !options[:force]
                skipped += 1
                @printf("[%d/%d] h11=%d polytope=%d frst=%d skipped matching configuration\n",
                    index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst)
                _vacua_append_summary(summary, geom_idx, "skipped"; seconds=time() - start)
                continue
            elseif state in (:config_mismatch, :incomplete) && !options[:force]
                blocked += 1
                message = state == :config_mismatch ?
                    "completed result has different configuration; use --force" :
                    "existing pipeline result is incomplete; use --force"
                @printf("[%d/%d] h11=%d polytope=%d frst=%d blocked: %s\n",
                    index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst, message)
                _vacua_append_summary(summary, geom_idx, string(state); seconds=time() - start,
                    message=message)
                continue
            elseif state == :missing && !isfile(path)
                failed += 1
                message = "geometry file does not exist"
                @printf("[%d/%d] h11=%d polytope=%d frst=%d missing\n",
                    index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst)
                _vacua_append_summary(summary, geom_idx, "missing_data"; seconds=time() - start,
                    message=message)
                continue
            elseif options[:dry_run]
                @printf("[%d/%d] h11=%d polytope=%d frst=%d %s\n",
                    index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst,
                    state == :matching ? "would recompute" : "would run")
                _vacua_append_summary(summary, geom_idx,
                    state == :matching ? "would_recompute" : "would_run";
                    seconds=time() - start)
                continue
            end

            job_options = VacuaJobOptions((
                threshold=options[:threshold], starts=options[:starts],
                residual_tolerance=options[:residual_tolerance],
                merge_tolerance=options[:merge_tolerance], max_iterations=options[:max_iterations],
                method=options[:method], max_branches=options[:max_branches], force=options[:force]))
            push!(active_jobs, VacuaJob((geom_idx=geom_idx, data_dir=options[:data_dir],
                options=job_options)))
        end

        worker_ids = isempty(active_jobs) ? Int[] : _vacua_start_workers(options)
        record_result = function (result)
            index = findfirst(==(result.geom_idx), geoms)
            @printf("[%d/%d] h11=%d polytope=%d frst=%d ", index, length(geoms),
                result.geom_idx.h11, result.geom_idx.polytope, result.geom_idx.frst)
            if result.status == "done"
                done += 1
                @printf("done estimate=%s verified=%s %.3fs\n",
                    result.estimate, result.verified, result.seconds)
                _vacua_append_summary(summary, result.geom_idx, "done";
                    estimate=result.estimate, verified=result.verified,
                    issquare=result.issquare, seconds=result.seconds)
            else
                failed += 1
                println("failed: ", result.message)
                _vacua_append_summary(summary, result.geom_idx, "failed";
                    seconds=result.seconds, message=result.message)
            end
        end
        batch_size = options[:batch_size]
        worker_pool = isempty(worker_ids) ? nothing : WorkerPool(worker_ids)
        for first_index in 1:batch_size:length(active_jobs)
            last_index = min(first_index + batch_size - 1, length(active_jobs))
            batch_jobs = active_jobs[first_index:last_index]
            if options[:workers] == 1
                for job in batch_jobs
                    record_result(_vacua_worker_job(job))
                end
            else
                for result in Distributed.pgenerate(worker_pool, _vacua_worker_job, batch_jobs)
                    record_result(result)
                end
            end
            GC.gc(false)
        end
    finally
        if !isempty(worker_ids)
            try
                rmprocs(worker_ids; waitfor=5)
            catch err
                @warn "some Distributed workers did not terminate cleanly" exception=(err, catch_backtrace())
            end
        end
    end

    @printf("Finished: done=%d skipped=%d blocked=%d failed=%d elapsed=%.3fs\n",
        done, skipped, blocked, failed, time() - total_start)
    failed == 0 && blocked == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = _vacua_parse_args(ARGS)
    success = run_vacua_batch(options)
    success || exit(1)
end
