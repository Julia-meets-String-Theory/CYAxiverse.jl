#!/usr/bin/env julia

using CYAxiverse
using HDF5
using LinearAlgebra
using Printf
using Statistics
using Logging

const GeometryIndex = CYAxiverse.structs.GeometryIndex

struct _BatchWarningLogger <: AbstractLogger
    messages::Vector{String}
end

Logging.min_enabled_level(::_BatchWarningLogger) = Logging.Debug
Logging.shouldlog(::_BatchWarningLogger, args...) = true
Logging.catch_exceptions(::_BatchWarningLogger) = false
function Logging.handle_message(logger::_BatchWarningLogger, level, message, args...; kwargs...)
    level >= Logging.Warn && push!(logger.messages, string(message))
end

function _usage()
    println("""
    Usage:
      julia --project=. scripts/batch_physical_spectrum.jl [options]

    Options:
      --data-dir PATH        Data root containing h11_*/np_*/cy_*/cyax.h5.
      --h11 N                Restrict selection to one h11.
      --limit N              Process at most N geometries.
      --offset N             Skip the first N selected geometries.
      --geometry H,P,F       Add one explicit geometry. May be repeated.
      --prec N               Arbitrary precision digits. Default: 200.
      --threshold-log10 X    Physical mass threshold. Default: log10(Hubble).
      --quartics             Compute diagonal quartics and fpert.
      --mass-only            Explicitly select mass-only mode (the default).
      --force                Recompute existing output files.
      --summary PATH         CSV summary output path.
      --append-summary       Append to an existing summary instead of replacing it.
      --hilbert              Accepted for CLI compatibility; unused by spectrum loading.
    """)
end

function _parse_args(args)
    options = Dict{Symbol, Any}(
        :data_dir => "", :h11 => nothing, :limit => nothing, :offset => 0,
        :geometries => GeometryIndex[], :prec => 200,
        :threshold_log10 => nothing, :quartics => false, :force => false,
        :summary => "", :append_summary => false, :hilbert => false)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            _usage(); exit(0)
        elseif arg == "--quartics"
            options[:quartics] = true
        elseif arg == "--mass-only"
            options[:quartics] = false
        elseif arg == "--force"
            options[:force] = true
        elseif arg == "--append-summary"
            options[:append_summary] = true
        elseif arg == "--hilbert"
            options[:hilbert] = true
        elseif arg in ("--data-dir", "--h11", "--limit", "--offset", "--geometry",
                       "--prec", "--threshold-log10", "--summary")
            i == length(args) && error("missing value for $arg")
            value = args[i + 1]
            if arg == "--data-dir"
                options[:data_dir] = value
            elseif arg == "--h11"
                options[:h11] = parse(Int, value)
            elseif arg == "--limit"
                options[:limit] = parse(Int, value)
            elseif arg == "--offset"
                options[:offset] = parse(Int, value)
            elseif arg == "--geometry"
                parts = parse.(Int, split(value, ","))
                length(parts) == 3 || error("--geometry must be H,P,F")
                push!(options[:geometries], GeometryIndex(parts...))
            elseif arg == "--prec"
                options[:prec] = parse(Int, value)
            elseif arg == "--threshold-log10"
                options[:threshold_log10] = parse(Float64, value)
            elseif arg == "--summary"
                options[:summary] = value
            end
            i += 1
        else
            error("unknown option: $arg")
        end
        i += 1
    end
    options[:prec] > 0 || error("--prec must be positive")
    options[:offset] >= 0 || error("--offset must be nonnegative")
    options
end

function _parse_prefixed_int(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        nothing
    end
end

function _scanned_geometries(h11_filter)
    root = CYAxiverse.filestructure.present_dir()
    h11_dirs = h11_filter === nothing ? filter(name -> startswith(name, "h11_"), readdir(root)) : [string("h11_", lpad(h11_filter, 3, "0"))]
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

function _indexed_geometries(h11_filter)
    try
        _, pathinds = CYAxiverse.filestructure.paths_cy()
        return [GeometryIndex(col...) for col in eachcol(pathinds)
                if h11_filter === nothing || col[1] == h11_filter]
    catch
        GeometryIndex[]
    end
end

function _selected_geometries(options)
    geoms = if isempty(options[:geometries])
        indexed = _indexed_geometries(options[:h11])
        scanned = _scanned_geometries(options[:h11])
        length(scanned) > length(indexed) ? scanned : indexed
    else
        copy(options[:geometries])
    end
    first_index = min(options[:offset] + 1, length(geoms) + 1)
    geoms = geoms[first_index:end]
    options[:limit] === nothing ? geoms : geoms[1:min(options[:limit], length(geoms))]
end

function _output_path(root, geom_idx)
    CYAxiverse.filestructure.cyax_file(geom_idx)
end

function _has_physical_spectrum(path)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "spectrum/physical/m")
    end
end

function _fK_log10(K)
    log10.(sqrt.(eigen(K).values)) .+
    Float64(log10(CYAxiverse.generate.constants()["MPlanck"])) .-
        Float64(CYAxiverse.generate.constants()["log2π"])
end

function _write_result(path, geom_idx, spectrum; prec, threshold_log10, quartics, runtime_seconds, provisional, fK=Float64[])
    h5open(path, "r+") do file
        spectrum_group = haskey(file, "spectrum") ? file["spectrum"] : create_group(file, "spectrum")
        physical = haskey(spectrum_group, "physical") ? spectrum_group["physical"] : create_group(spectrum_group, "physical")
        metadata = haskey(physical, "metadata") ? physical["metadata"] : create_group(physical, "metadata")
        metadata["h11"] = geom_idx.h11
        metadata["polytope"] = geom_idx.polytope
        metadata["frst"] = geom_idx.frst
        metadata["threshold_log10"] = threshold_log10
        metadata["prec"] = prec
        metadata["quartics"] = quartics
        metadata["provisional"] = provisional
        metadata["runtime_seconds"] = runtime_seconds
        physical["m"] = spectrum.m
        physical["mode_indices"] = spectrum.mode_indices
        physical["mass_signs_or_inertia"] = Int.(spectrum.m .>= threshold_log10)
        physical["fK_log10"] = fK
        if quartics
            physical["lambda_self_sign"] = spectrum.λselfsign
            physical["lambda_self_log10"] = spectrum.λself
            physical["fpert_log10"] = spectrum.m .- 0.5 .* spectrum.λself
        end
    end
end

function _csv_escape(value)
    text = replace(string(value), '"' => "\"\"")
    occursin(r"[,\"\n]", text) ? string('"', text, '"') : text
end

const SUMMARY_HEADER = "h11,polytope,frst,status,error,runtime_seconds,prec,threshold_log10,instantons,physical_count,massless_count,min_mass_log10,max_mass_log10,median_mass_log10,quartics,negative_lambda_count,positive_lambda_count,min_fpert_log10,max_fpert_log10,median_fpert_log10,min_fK_log10,max_fK_log10,median_fK_log10,provisional,output"

function _write_summary_header(path; append=false)
    append && isfile(path) && return
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, SUMMARY_HEADER)
    end
end

function _median_or_empty(values)
    isempty(values) ? "" : median(values)
end

function _append_summary(path, geom_idx; status, error="", runtime_seconds=0.0, prec, threshold_log10,
                         instantons="", spectrum=nothing, quartics=false, provisional=false, output="", fK=Float64[])
    masses = spectrum === nothing ? Float64[] : spectrum.m
    lambda = quartics && spectrum !== nothing ? spectrum.λself : Float64[]
    fpert = quartics && spectrum !== nothing ? masses .- 0.5 .* lambda : Float64[]
    values = [geom_idx.h11, geom_idx.polytope, geom_idx.frst, status, error, runtime_seconds,
        prec, threshold_log10, instantons, length(masses), count(<(threshold_log10), masses),
        isempty(masses) ? "" : minimum(masses), isempty(masses) ? "" : maximum(masses), _median_or_empty(masses),
        quartics, count(<(0), lambda), count(>(0), lambda), isempty(fpert) ? "" : minimum(fpert),
        isempty(fpert) ? "" : maximum(fpert), _median_or_empty(fpert),
        isempty(fK) ? "" : minimum(fK), isempty(fK) ? "" : maximum(fK), _median_or_empty(fK),
        provisional, output]
    open(path, "a") do io
        println(io, join(_csv_escape.(values), ','))
        flush(io)
    end
end

function run_batch(options)
    !isempty(options[:data_dir]) && (ENV["CYAXIVERSE_DATA_DIR"] = options[:data_dir])
    threshold = something(options[:threshold_log10], Float64(log10(CYAxiverse.generate.constants()["Hubble"])))
    geoms = _selected_geometries(options)
    isempty(geoms) && error("no geometries selected")
    root = CYAxiverse.filestructure.present_dir()
    summary = isempty(options[:summary]) ? joinpath(root, "logs", "physical_spectrum.csv") : options[:summary]
    _write_summary_header(summary; append=options[:append_summary])
    @printf("Physical spectrum batch: %d geometries, quartics=%s\n", length(geoms), options[:quartics])

    failed = 0
    for (index, geom_idx) in enumerate(geoms)
        path = _output_path(root, geom_idx)
        @printf("[%d/%d] h11=%d polytope=%d frst=%d ", index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst)
        if _has_physical_spectrum(path) && !options[:force]
            println("skipped")
            _append_summary(summary, geom_idx; status="skipped", prec=options[:prec], threshold_log10=threshold, quartics=options[:quartics], output=path)
            continue
        end
        started = time()
        try
            potential = CYAxiverse.read.potential(geom_idx)
            warning_messages = String[]
            spectrum = Logging.with_logger(_BatchWarningLogger(warning_messages)) do
                CYAxiverse.generate.pq_hybrid_physical_spectrum(geom_idx;
                    threshold_log10=threshold, prec=options[:prec], quartics=options[:quartics],
                    mixed_quartics=false,
                    label="h11=$(geom_idx.h11),polytope=$(geom_idx.polytope),frst=$(geom_idx.frst)")
            end
            runtime = time() - started
            fK = _fK_log10(potential.K)
            provisional = any(occursin("provisional", lowercase(message)) for message in warning_messages)
            _write_result(path, geom_idx, spectrum; prec=options[:prec], threshold_log10=threshold,
                quartics=options[:quartics], runtime_seconds=runtime, provisional=provisional, fK=fK)
            _append_summary(summary, geom_idx; status="success", runtime_seconds=runtime, prec=options[:prec],
                threshold_log10=threshold, instantons=size(potential.L, 2), spectrum=spectrum,
                quartics=options[:quartics], provisional=provisional, output=path, fK=fK)
            @printf("success physical=%d %.3fs\n", length(spectrum.m), runtime)
        catch err
            runtime = time() - started
            failed += 1
            message = sprint(showerror, err)
            _append_summary(summary, geom_idx; status="failed", error=message, runtime_seconds=runtime,
                prec=options[:prec], threshold_log10=threshold, quartics=options[:quartics])
            println("failed: ", message)
        end
        GC.gc(false)
    end
    failed == 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_batch(_parse_args(ARGS)) || exit(1)
end
