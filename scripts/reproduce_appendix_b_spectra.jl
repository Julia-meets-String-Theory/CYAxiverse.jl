#!/usr/bin/env julia

using CairoMakie
using HDF5
using Printf
using Statistics

set_theme!(theme_latexfonts())

"""Print command-line usage for the Appendix B reproduction script."""
function _usage()
    println("""
    Usage:
      julia --project=. scripts/reproduce_appendix_b_spectra.jl [options]

    Options:
      --data-dir PATH     Data root containing physical_spectrum/.
      --output-dir PATH   Directory for plots and CSV output.
      --h11-min N         Minimum h11 to include. Default: 4.
      --h11-max N         Maximum h11 to include. Default: 30.
      --bins N            Number of log-space bins. Default: 160.
    """)
end

"""Parse plotting options into a dictionary."""
function _parse_args(args)
    options = Dict{Symbol,Any}(:data_dir => "", :output_dir => "", :h11_min => 4,
        :h11_max => 30, :bins => 160)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            _usage()
            exit(0)
        end
        arg in ("--data-dir", "--output-dir", "--h11-min", "--h11-max", "--bins") ||
            error("unknown option: $arg")
        i == length(args) && error("missing value for $arg")
        value = args[i + 1]
        if arg == "--data-dir"
            options[:data_dir] = value
        elseif arg == "--output-dir"
            options[:output_dir] = value
        elseif arg == "--h11-min"
            options[:h11_min] = parse(Int, value)
        elseif arg == "--h11-max"
            options[:h11_max] = parse(Int, value)
        else
            options[:bins] = parse(Int, value)
        end
        i += 2
    end
    options[:h11_min] <= options[:h11_max] || error("h11-min must not exceed h11-max")
    options[:bins] > 1 || error("bins must exceed one")
    options
end

"""Find HDF5 files under `root` whose h11 values lie in the requested range."""
function _physical_files(root, h11_min, h11_max)
    files = Tuple{Int,String}[]
    for h11_dir in readdir(root; join=true)
        name = basename(h11_dir)
        startswith(name, "h11_") || continue
        h11 = try
            parse(Int, name[5:end])
        catch
            continue
        end
        h11_min <= h11 <= h11_max || continue
        for (dir, _, names) in walkdir(h11_dir)
            for filename in names
                endswith(filename, ".h5") || continue
                push!(files, (h11, joinpath(dir, filename)))
            end
        end
    end
    sort!(files; by=first)
    files
end

"""Read completed physical-spectrum values grouped by h11."""
function _read_values(files)
    masses = Dict{Int,Vector{Float64}}()
    fpert = Dict{Int,Vector{Float64}}()
    fK = Dict{Int,Vector{Float64}}()
    completed = Dict{Int,Int}()
    for (h11, path) in files
        try
            h5open(path, "r") do file
                physical = file["spectrum/physical"]
                haskey(physical, "m") || return
                append!(get!(masses, h11, Float64[]), read(physical["m"]))
                haskey(physical, "fK_log10") && append!(get!(fK, h11, Float64[]), read(physical["fK_log10"]))
                haskey(physical, "fpert_log10") && append!(get!(fpert, h11, Float64[]), read(physical["fpert_log10"]))
                completed[h11] = get(completed, h11, 0) + 1
            end
        catch err
            @warn "Skipping unreadable spectrum output" path exception=(err, catch_backtrace())
        end
    end
    masses, fpert, fK, completed
end

"""Compute a normalized histogram for finite values and supplied bin edges."""
function _histogram(values, edges)
    counts = zeros(Float64, length(edges) - 1)
    for value in values
        isfinite(value) || continue
        index = searchsortedlast(edges, value)
        1 <= index < length(edges) || continue
        counts[index] += 1
    end
    total = sum(counts)
    total == 0 ? counts : counts ./ total
end

"""Build one normalized histogram column for each h11 value."""
function _matrix(values, h11s, edges)
    matrix = zeros(Float64, length(edges) - 1, length(h11s))
    for (column, h11) in enumerate(h11s)
        matrix[:, column] = _histogram(get(values, h11, Float64[]), edges)
    end
    matrix
end

"""Save a PDF heatmap of the spectrum distribution by h11."""
function _plot_heatmap(path, matrix, h11s, edges, title, xlabel)
    h11_edges = if length(h11s) == 1
        [h11s[1] - 0.5, h11s[1] + 0.5]
    else
        midpoints = (h11s[1:end-1] .+ h11s[2:end]) ./ 2
        [h11s[1] - (midpoints[1] - h11s[1]); midpoints; h11s[end] + (h11s[end] - midpoints[end])]
    end
    figure = Figure(size=(1100, 700))
    axis = Axis(figure[1, 1], title=title, xlabel=xlabel, ylabel=L"$h^{1,1}$",
        titlesize=20, xlabelsize=16, ylabelsize=16)
    heatmap!(axis, edges, h11_edges, matrix; colormap=:viridis)
    Colorbar(figure[1, 2], label="fraction per log10 bin")
    save(path, figure)
end

"""Return a quantile or an empty CSV field for empty values."""
function _quantile_or_empty(values, q)
    isempty(values) ? "" : quantile(values, q)
end

"""Write per-h11 quantiles and completion counts to CSV."""
function _write_summary(path, h11s, masses, fpert, fK, completed)
    open(path, "w") do io
        println(io, "h11,geometries,mass_count,mass_q05,mass_median,mass_q95,fpert_count,fpert_q05,fpert_median,fpert_q95,fK_count,fK_q05,fK_median,fK_q95")
        for h11 in h11s
            mass = get(masses, h11, Float64[])
            self = get(fpert, h11, Float64[])
            metric = get(fK, h11, Float64[])
            println(io, join((h11, get(completed, h11, 0), length(mass),
                _quantile_or_empty(mass, 0.05), _quantile_or_empty(mass, 0.5), _quantile_or_empty(mass, 0.95),
                length(self), _quantile_or_empty(self, 0.05), _quantile_or_empty(self, 0.5), _quantile_or_empty(self, 0.95),
                length(metric), _quantile_or_empty(metric, 0.05), _quantile_or_empty(metric, 0.5), _quantile_or_empty(metric, 0.95)), ','))
        end
    end
end

"""Generate Appendix B spectrum heatmaps and quantile summaries."""
function main(options)
    data_dir = isempty(options[:data_dir]) ? pwd() : abspath(options[:data_dir])
    output_dir = isempty(options[:output_dir]) ? joinpath(data_dir, "physical_spectrum", "appendix_b") : abspath(options[:output_dir])
    mkpath(output_dir)
    files = _physical_files(data_dir, options[:h11_min], options[:h11_max])
    isempty(files) && error("no physical spectrum outputs found")
    masses, fpert, fK, completed = _read_values(files)
    h11s = collect(options[:h11_min]:options[:h11_max])
    bins = options[:bins]
    mass_edges = range(-33.0, 28.0; length=bins + 1)
    fpert_edges = range(-2.0, 28.0; length=bins + 1)
    fK_edges = range(-2.0, 19.0; length=bins + 1)
    _plot_heatmap(joinpath(output_dir, "masses_by_h11.pdf"), _matrix(masses, h11s, mass_edges), h11s, mass_edges,
        "Appendix B spectrum reproduction: masses", "log10(m / GeV)")
    _plot_heatmap(joinpath(output_dir, "fpert_by_h11.pdf"), _matrix(fpert, h11s, fpert_edges), h11s, fpert_edges,
        "Appendix B spectrum reproduction: perturbative decay constants", "log10(fpert / GeV)")
    _plot_heatmap(joinpath(output_dir, "fK_by_h11.pdf"), _matrix(fK, h11s, fK_edges), h11s, fK_edges,
        "Appendix B spectrum reproduction: Kahler decay constants", "log10(fK / GeV)")
    _write_summary(joinpath(output_dir, "appendix_b_quantiles.csv"), h11s, masses, fpert, fK, completed)
    @printf("Read %d spectrum outputs across h11=%d..%d\n", length(files), options[:h11_min], options[:h11_max])
    println("Wrote plots and quantiles to ", output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(_parse_args(ARGS))
end
