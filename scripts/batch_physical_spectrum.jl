#!/usr/bin/env julia

using CYAxiverse
using HDF5
using LinearAlgebra
using Printf
using Statistics
using Logging

const GeometryIndex = CYAxiverse.structs.GeometryIndex

"""Collect warning messages emitted while processing one geometry."""
struct _BatchWarningLogger <: AbstractLogger
    messages::Vector{String}
end

Logging.min_enabled_level(::_BatchWarningLogger) = Logging.Debug
Logging.shouldlog(::_BatchWarningLogger, args...) = true
Logging.catch_exceptions(::_BatchWarningLogger) = false
"""Store warning-level messages in the batch logger."""
function Logging.handle_message(logger::_BatchWarningLogger, level, message, args...; kwargs...)
    level >= Logging.Warn && push!(logger.messages, string(message))
end

"""Print command-line usage for the physical-spectrum batch runner."""
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

"""Parse batch-runner command-line arguments into a typed options tuple."""
function _parse_args(args)
    options = (data_dir="", h11=nothing, limit=nothing, offset=0,
        geometries=GeometryIndex[], prec=200, threshold_log10=nothing,
        quartics=false, force=false, summary="", append_summary=false,
        hilbert=false)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            _usage(); exit(0)
        elseif arg == "--quartics"
            options = merge(options, (; quartics=true))
        elseif arg == "--mass-only"
            options = merge(options, (; quartics=false))
        elseif arg == "--force"
            options = merge(options, (; force=true))
        elseif arg == "--append-summary"
            options = merge(options, (; append_summary=true))
        elseif arg == "--hilbert"
            options = merge(options, (; hilbert=true))
        elseif arg in ("--data-dir", "--h11", "--limit", "--offset", "--geometry",
                       "--prec", "--threshold-log10", "--summary")
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
                length(parts) == 3 || error("--geometry must be H,P,F")
                push!(options.geometries, GeometryIndex(parts...))
            elseif arg == "--prec"
                options = merge(options, (; prec=parse(Int, value)))
            elseif arg == "--threshold-log10"
                options = merge(options, (; threshold_log10=parse(Float64, value)))
            elseif arg == "--summary"
                options = merge(options, (; summary=value))
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

"""Parse the integer suffix of a directory name with the given prefix."""
function _parse_prefixed_int(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        nothing
    end
end

"""Find geometries by scanning the selected database directory tree."""
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

"""Find geometries from the package path index, returning an empty fallback."""
function _indexed_geometries(h11_filter)
    try
        _, pathinds = CYAxiverse.filestructure.paths_cy()
        return [GeometryIndex(col...) for col in eachcol(pathinds)
                if h11_filter === nothing || col[1] == h11_filter]
    catch
        GeometryIndex[]
    end
end

"""Combine explicit, indexed, and scanned selections with offset and limit."""
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

"""Return the existing `cyax.h5` path for a geometry."""
function _output_path(root, geom_idx)
    CYAxiverse.filestructure.cyax_file(geom_idx)
end

"""Return whether an HDF5 file contains a completed physical mass dataset."""
function _has_physical_spectrum(path)
    isfile(path) || return false
    h5open(path, "r") do file
        haskey(file, "spectrum/physical/m")
    end
end

"""Convert the eigenvalues of `K` into base-10 Kahler decay constants."""
function _fK_log10(K)
    log10.(sqrt.(eigen(K).values)) .+
    Float64(log10(CYAxiverse.generate.constants()["MPlanck"])) .-
        Float64(CYAxiverse.generate.constants()["log2π"])
end

"""Replace one HDF5 dataset, deleting an existing dataset first."""
function _replace_dataset(group, name, value)
    haskey(group, name) && HDF5.delete_object(group, name)
    group[name] = value
end

"""Write one physical spectrum and metadata into its existing geometry file."""
function _write_result(path, geom_idx, spectrum; prec, threshold_log10, quartics, runtime_seconds, provisional, fK=Float64[])
    h5open(path, "r+") do file
        spectrum_group = haskey(file, "spectrum") ? file["spectrum"] : create_group(file, "spectrum")
        physical = haskey(spectrum_group, "physical") ? spectrum_group["physical"] : create_group(spectrum_group, "physical")
        metadata = haskey(physical, "metadata") ? physical["metadata"] : create_group(physical, "metadata")
        _replace_dataset(metadata, "h11", geom_idx.h11)
        _replace_dataset(metadata, "polytope", geom_idx.polytope)
        _replace_dataset(metadata, "frst", geom_idx.frst)
        _replace_dataset(metadata, "threshold_log10", threshold_log10)
        _replace_dataset(metadata, "prec", prec)
        _replace_dataset(metadata, "quartics", quartics)
        _replace_dataset(metadata, "provisional", provisional)
        _replace_dataset(metadata, "runtime_seconds", runtime_seconds)
        _replace_dataset(physical, "m", spectrum.m)
        _replace_dataset(physical, "mode_indices", spectrum.mode_indices)
        _replace_dataset(physical, "fK_log10", fK)
        if quartics
            _replace_dataset(physical, "lambda_self_sign", spectrum.λselfsign)
            _replace_dataset(physical, "lambda_self_log10", spectrum.λself)
            _replace_dataset(physical, "fpert_log10", spectrum.m .- 0.5 .* spectrum.λself)
        else
            for dataset_name in ("lambda_self_sign", "lambda_self_log10", "fpert_log10")
                haskey(physical, dataset_name) && HDF5.delete_object(physical, dataset_name)
            end
        end
    end
end

"""Quote a value when needed for the batch summary CSV."""
function _csv_escape(value)
    text = replace(string(value), '"' => "\"\"")
    occursin(r"[,\"\n]", text) ? string('"', text, '"') : text
end

const SUMMARY_HEADER = "h11,polytope,frst,status,error,runtime_seconds,prec,threshold_log10,instantons,physical_count,massless_count,min_mass_log10,max_mass_log10,median_mass_log10,quartics,negative_lambda_count,positive_lambda_count,min_fpert_log10,max_fpert_log10,median_fpert_log10,min_fK_log10,max_fK_log10,median_fK_log10,provisional,output"

"""Create the batch summary CSV and write its header when needed."""
function _write_summary_header(path; append=false)
    append && isfile(path) && return
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, SUMMARY_HEADER)
    end
end

"""Return a median or an empty field for an empty collection."""
function _median_or_empty(values)
    isempty(values) ? "" : median(values)
end

"""Append one geometry's processing result and summary statistics to CSV."""
function _append_summary(path, geom_idx; status, error="", runtime_seconds=0.0, prec, threshold_log10,
                         instantons="", spectrum=nothing, quartics=false, provisional=false, output="", fK=Float64[])
    masses = spectrum === nothing ? Float64[] : spectrum.m
    lambda = quartics && spectrum !== nothing ? spectrum.λself : Float64[]
    fpert = quartics && spectrum !== nothing ? masses .- 0.5 .* lambda : Float64[]
    values = (geom_idx.h11, geom_idx.polytope, geom_idx.frst, status, error, runtime_seconds,
        prec, threshold_log10, instantons, length(masses), count(<(threshold_log10), masses),
        isempty(masses) ? "" : minimum(masses), isempty(masses) ? "" : maximum(masses), _median_or_empty(masses),
        quartics, count(<(0), lambda), count(>(0), lambda), isempty(fpert) ? "" : minimum(fpert),
        isempty(fpert) ? "" : maximum(fpert), _median_or_empty(fpert),
        isempty(fK) ? "" : minimum(fK), isempty(fK) ? "" : maximum(fK), _median_or_empty(fK),
        provisional, output)
    open(path, "a") do io
        println(io, join(_csv_escape.(values), ','))
        flush(io)
    end
end

"""
    run_batch(options)

Process the selected geometries and persist physical spectra in place. Return
`true` only when every selected geometry completed successfully or was skipped.
"""
function run_batch(options)
    data_dir = CYAxiverse.filestructure.resolve_data_dir(options[:data_dir])
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir
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
